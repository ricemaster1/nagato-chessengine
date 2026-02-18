//! NNUE (Efficiently Updatable Neural Network) evaluation module.
//!
//! Architecture: "Baby NNUE" — Phase 1
//!   Input:  768 piece-square features (color × piece × square)
//!   Layer1: 768 → L1_SIZE (per-perspective, ClippedReLU)
//!   Concat: 2 × L1_SIZE from both perspectives
//!   Layer2: 2*L1_SIZE → L2_SIZE (ClippedReLU)
//!   Output: L2_SIZE → 1 (linear)

pub mod features;
pub mod accumulator;
pub mod network;

// ─── Shared constants ────────────────────────────────────────

/// First hidden layer size (per perspective)
pub const L1_SIZE: usize = 128;
/// Second hidden layer size
pub const L2_SIZE: usize = 32;
/// Input feature count (2 colors × 6 pieces × 64 squares)
pub const INPUT_SIZE: usize = 768;

// ─── Public API re-exports ──────────────────────────────────

pub use accumulator::{
    Accumulator,
    refresh_accumulator,
    accumulator_add,
    accumulator_remove,
    accumulator_move,
};
pub use features::{feature_index_white, feature_index_black};
pub use network::{init, is_active, evaluate, forward};
