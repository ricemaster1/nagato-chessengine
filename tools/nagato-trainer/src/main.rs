use bullet_lib::{
    game::{
        formats::bulletformat::ChessBoard,
        inputs::{ChessBucketsMirrored, get_num_buckets},
        outputs::OutputBuckets,
    },
    nn::{
        InitSettings, Shape,
        optimiser::{AdamW, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::SfBinpackLoader},
};
use sfbinpack::{
    TrainingDataEntry,
    chess::{piece::Piece, r#move::MoveType},
};

const L1_SIZE: usize = 256;
const L1_PAIR: usize = L1_SIZE / 2;
const L2_INPUT: usize = 2 * L1_PAIR;
const L2_SIZE: usize = 32;
const NUM_OUTPUT_BUCKETS: usize = 4;
const NUM_PSQT_BUCKETS: usize = 4;
const SKIP_SIZE: usize = 8;

#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
    3, 3, 3, 3,
    4, 1, 1, 1,
    7, 1, 0, 0,
    9, 1, 0, 0,
    9, 1, 0, 0,
    7, 1, 1, 1,
    5, 1, 1, 1,
    6, 6, 6, 6,
];

const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);
const FT_SIZE: usize = 768 * NUM_INPUT_BUCKETS;

const MIN_PLY: u16 = 16;
const MAX_ABS_SCORE: u16 = 10_000;

fn is_capture(entry: &TrainingDataEntry) -> bool {
    match entry.mv.mtype() {
        MoveType::EnPassant => true,
        MoveType::Castle => false,
        _ => entry.pos.piece_at(entry.mv.to()) != Piece::none(),
    }
}

fn keep_entry(entry: &TrainingDataEntry) -> bool {
    entry.ply >= MIN_PLY
        && entry.score.unsigned_abs() <= MAX_ABS_SCORE
        && !entry.pos.is_checked(entry.pos.side_to_move())
        && !is_capture(entry)
}

const NAGT_MAGIC: &[u8; 4] = b"NAGT";
const NAGT_VERSION: u32 = 4;

const NET_ID: &str = "nagato-halfka-v4f";
const SUPERBATCHES: usize = 600;
const BATCH_SIZE: usize = 16_384;
const BATCHES_PER_SB: usize = 6104;
const INITIAL_LR: f32 = 0.001;
const WDL_PROPORTION: f32 = 0.75;
const EVAL_SCALE: f32 = 400.0;
const SAVE_RATE: usize = 10;
const THREADS: usize = 12;
const BINPACK_BUFFER_MB: usize = 2048;
const BINPACK_WORKERS: usize = 12;
const OUTPUT_DIR: &str = "checkpoints";

#[derive(Clone, Copy, Default)]
struct NagatoMaterialBuckets;

impl OutputBuckets<ChessBoard> for NagatoMaterialBuckets {
    const BUCKETS: usize = NUM_OUTPUT_BUCKETS;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        let count = pos.occ().count_ones() as usize;
        (((count.max(1) - 1) / 8).min(NUM_OUTPUT_BUCKETS - 1)) as u8
    }
}

fn halves_perm(out_idx: usize, pair: usize) -> usize {
    if out_idx < pair { 2 * out_idx } else { 2 * (out_idx - pair) + 1 }
}

fn permute_halves_rows(src: &[f32], row_size: usize) -> Vec<f32> {
    let pair = row_size / 2;
    let rows = src.len() / row_size;
    let mut out = vec![0.0f32; src.len()];
    for r in 0..rows {
        let base = r * row_size;
        for j in 0..row_size {
            out[base + j] = src[base + halves_perm(j, pair)];
        }
    }
    out
}

fn permute_halves_1d(src: Vec<f32>) -> Vec<f32> {
    let pair = src.len() / 2;
    (0..src.len()).map(|j| src[halves_perm(j, pair)]).collect()
}

fn nagt_save_format() -> Vec<SavedFormat> {
    let mut fmt: Vec<SavedFormat> = Vec::new();

    fmt.push(SavedFormat::custom(NAGT_MAGIC.to_vec()));
    fmt.push(SavedFormat::custom(NAGT_VERSION.to_le_bytes().to_vec()));

    fmt.push(
        SavedFormat::id("l0w")
            .transform(|store, weights| {
                let factoriser = store.get("l0f").values.f32().repeat(NUM_INPUT_BUCKETS);
                weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
            })
            .transpose()
            .transform(|_, w| permute_halves_rows(&w, L1_SIZE)),
    );

    fmt.push(SavedFormat::id("l0b").transform(|_, w| permute_halves_1d(w)));

    fmt.push(SavedFormat::custom(vec![0u8; FT_SIZE * NUM_PSQT_BUCKETS * 4]));

    for s in 0..NUM_OUTPUT_BUCKETS {
        fmt.push(SavedFormat::id("l1w").transpose().transform(move |_, w| {
            let stride = NUM_OUTPUT_BUCKETS * L2_SIZE;
            let mut out = Vec::with_capacity(L2_INPUT * L2_SIZE);
            for in_i in 0..L2_INPUT {
                for j in 0..L2_SIZE {
                    out.push(w[in_i * stride + s * L2_SIZE + j]);
                }
            }
            out
        }));

        fmt.push(SavedFormat::id("l1b").transform(move |_, w| w[s * L2_SIZE..(s + 1) * L2_SIZE].to_vec()));

        fmt.push(SavedFormat::id("l2w").transform(move |_, w| w[s * L2_SIZE..(s + 1) * L2_SIZE].to_vec()));

        fmt.push(SavedFormat::id("l2b").transform(move |_, w| vec![w[s]]));

        fmt.push(SavedFormat::custom(vec![0u8; SKIP_SIZE * 4]));
    }

    fmt
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: nagato-trainer <data.binpack> [more.binpack ...]");
        std::process::exit(2);
    }
    let data_paths: Vec<&str> = args.iter().map(String::as_str).collect();

    let final_lr = INITIAL_LR * 0.3f32.powi(5);
    let fmt = nagt_save_format();

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(NagatoMaterialBuckets)
        .save_format(&fmt)
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0f = builder.new_weights("l0f", Shape::new(L1_SIZE, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, L1_SIZE);
            l0.weights = l0.weights + expanded_factoriser;

            let l1 = builder.new_affine("l1", L1_SIZE, NUM_OUTPUT_BUCKETS * L2_SIZE);
            let l2 = builder.new_affine("l2", L2_SIZE, NUM_OUTPUT_BUCKETS);

            let stm_h = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_h = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let hl1 = stm_h.concat(ntm_h);
            let hl2 = l1.forward(hl1).select(output_buckets).screlu();
            l2.forward(hl2).select(output_buckets)
        });

    let stricter = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter);
    trainer.optimiser.set_params_for_weight("l0f", stricter);

    let schedule = TrainingSchedule {
        net_id: NET_ID.to_string(),
        eval_scale: EVAL_SCALE,
        steps: TrainingSteps {
            batch_size: BATCH_SIZE,
            batches_per_superbatch: BATCHES_PER_SB,
            start_superbatch: 1,
            end_superbatch: SUPERBATCHES,
        },
        wdl_scheduler: wdl::ConstantWDL { value: WDL_PROPORTION },
        lr_scheduler: lr::CosineDecayLR { initial_lr: INITIAL_LR, final_lr, final_superbatch: SUPERBATCHES },
        save_rate: SAVE_RATE,
    };

    let settings = LocalSettings {
        threads: THREADS,
        test_set: None,
        output_directory: OUTPUT_DIR,
        batch_queue_size: 32,
    };

    let dataloader = SfBinpackLoader::new_concat_multiple(
        &data_paths,
        BINPACK_BUFFER_MB,
        BINPACK_WORKERS,
        keep_entry,
    );

    trainer.run(&schedule, &settings, &dataloader);
}
