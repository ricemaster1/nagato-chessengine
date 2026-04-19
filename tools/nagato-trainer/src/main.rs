use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, get_num_buckets},
        outputs::MaterialCount,
    },
    nn::{
        InitSettings, Shape,
        optimiser::{AdamW, AdamWParams},
    },
    trainer::save::SavedFormat,
    value::ValueTrainerBuilder,
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

fn main() {
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

    let _ = trainer;
    eprintln!("nagato-trainer: network defined — training schedule pending (chunk 3)");
    std::process::exit(1);
}
