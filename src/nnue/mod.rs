pub mod features;
pub mod accumulator;
pub mod network;

pub const L1_SIZE: usize = 128;
pub const L2_SIZE: usize = 32;
pub const INPUT_SIZE: usize = 768;

pub use accumulator::{
    Accumulator,
    refresh_accumulator,
    accumulator_add,
    accumulator_remove,
    accumulator_move,
};
pub use features::{feature_index_white, feature_index_black};
pub use network::{init, is_active, evaluate, forward};
