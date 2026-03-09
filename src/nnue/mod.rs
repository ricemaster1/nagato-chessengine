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

pub struct ArchConfig {
    pub l1_size: usize,
    pub l2_size: usize,
    pub num_layer_stacks: usize,
    pub skip_size: usize,
    pub num_psqt_buckets: usize,
    pub king_buckets: usize,
}

impl ArchConfig {
    pub const DEFAULT: Self = ArchConfig {
        l1_size: L1_SIZE,
        l2_size: L2_SIZE,
        num_layer_stacks: NUM_LAYER_STACKS,
        skip_size: SKIP_SIZE,
        num_psqt_buckets: NUM_PSQT_BUCKETS,
        king_buckets: 10,
    };

    pub const fn ft_size(&self) -> usize {
        self.king_buckets * 5 * 64 * 2
    }

    pub const fn l1_pair(&self) -> usize {
        self.l1_size / 2
    }

    pub const fn l2_input(&self) -> usize {
        2 * self.l1_pair()
    }

    pub const fn total_params(&self) -> usize {
        let ft = self.ft_size() * self.l1_size + self.l1_size;
        let psqt = self.ft_size() * self.num_psqt_buckets;
        let per_stack = self.l2_input() * self.l2_size + self.l2_size + self.l2_size + 1 + self.skip_size;
        ft + psqt + self.num_layer_stacks * per_stack
    }
}

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
