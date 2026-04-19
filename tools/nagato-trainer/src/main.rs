use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, get_num_buckets},
        outputs::MaterialCount,
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

const L1_SIZE: usize = 256;
const L2_SIZE: usize = 32;
const NUM_OUTPUT_BUCKETS: usize = 4;

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

const NET_ID: &str = "nagato-halfka-v4";
const SUPERBATCHES: usize = 800;
const BATCH_SIZE: usize = 16_384;
const BATCHES_PER_SB: usize = 6104;
const INITIAL_LR: f32 = 0.001;
const WDL_PROPORTION: f32 = 0.75;
const EVAL_SCALE: f32 = 400.0;
const SAVE_RATE: usize = 10;
const THREADS: usize = 4;
const BINPACK_BUFFER_MB: usize = 2048;
const BINPACK_WORKERS: usize = 4;
const OUTPUT_DIR: &str = "checkpoints";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: nagato-trainer <data.binpack> [more.binpack ...]");
        std::process::exit(2);
    }
    let data_paths: Vec<&str> = args.iter().map(String::as_str).collect();

    let final_lr = INITIAL_LR * 0.3f32.powi(5);

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w")
                .transform(|store, weights| {
                    let factoriser = store.get("l0f").values.f32().repeat(NUM_INPUT_BUCKETS);
                    weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
                })
                .round()
                .quantise::<i16>(255),
            SavedFormat::id("l0b").round().quantise::<i16>(255),
            SavedFormat::id("l1w").transpose().round().quantise::<i8>(64),
            SavedFormat::id("l1b"),
            SavedFormat::id("l2w").transpose(),
            SavedFormat::id("l2b"),
        ])
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
        |_entry| true,
    );

    trainer.run(&schedule, &settings, &dataloader);
}
