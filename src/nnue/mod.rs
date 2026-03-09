pub mod features;
pub mod accumulator;
pub mod network;
pub mod simd;
pub mod trainer;

pub const L1_SIZE: usize = 256;
pub const L1_PAIR: usize = L1_SIZE / 2;
pub const L2_INPUT: usize = 2 * L1_PAIR;
pub const L2_SIZE: usize = 32;
pub const INPUT_SIZE: usize = 768;
pub const NUM_PSQT_BUCKETS: usize = 4;
pub const NUM_LAYER_STACKS: usize = 4;
pub const SKIP_SIZE: usize = 8;

pub const QA: i32 = 255;
pub const QB: i32 = 64;

pub use accumulator::{
    Accumulator,
    AccumulatorQ,
    AccStackQ,
    DirtyPiece,
    SQ_NONE,
    FinnyCache,
    refresh_accumulator,
    refresh_accumulator_q,
    refresh_half_q,
    finny_update_half,
    apply_dirty_half_q,
    accumulator_add,
    accumulator_add_q,
    accumulator_remove,
    accumulator_remove_q,
    accumulator_move,
    accumulator_move_q,
};
pub use features::{feature_index_white, feature_index_black, king_bucket_of};
pub use network::{init, is_active, weight_version, evaluate, evaluate_q, forward, forward_q, psqt_bucket};
