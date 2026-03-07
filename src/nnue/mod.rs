pub mod features;
pub mod accumulator;
pub mod network;

pub const L1_SIZE: usize = 128;
pub const L2_SIZE: usize = 32;
pub const INPUT_SIZE: usize = 768;

pub const QA: i32 = 255;
pub const QB: i32 = 64;

pub use accumulator::{
    Accumulator,
    AccumulatorQ,
    refresh_accumulator,
    refresh_accumulator_q,
    accumulator_add,
    accumulator_add_q,
    accumulator_remove,
    accumulator_remove_q,
    accumulator_move,
    accumulator_move_q,
};
pub use features::{feature_index_white, feature_index_black};
pub use network::{init, is_active, evaluate, evaluate_q, forward, forward_q};
