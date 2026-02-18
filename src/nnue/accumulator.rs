//! NNUE accumulator — incrementally updated layer-1 pre-activations.
//!
//! The accumulator stores per-perspective hidden layer values that are
//! incrementally maintained as pieces move on the board.

use crate::bitboard::*;
use crate::board::Board;

use super::L1_SIZE;
use super::features::{feature_index_white, feature_index_black};
use super::network::weights;

/// The accumulator stores the pre-activation values for layer 1.
/// It can be incrementally updated when pieces move.
#[derive(Clone)]
pub struct Accumulator {
    /// Layer 1 values for white's perspective
    pub white: [f32; L1_SIZE],
    /// Layer 1 values for black's perspective
    pub black: [f32; L1_SIZE],
}

impl Accumulator {
    pub fn new() -> Self {
        Accumulator {
            white: [0.0; L1_SIZE],
            black: [0.0; L1_SIZE],
        }
    }
}

/// Compute the full accumulator from scratch for a given board position.
/// This is used when setting up a new position (e.g., from FEN or at the start).
pub fn refresh_accumulator(board: &Board, acc: &mut Accumulator) {
    let w = weights();

    // Start with biases
    acc.white = w.l1_biases;
    acc.black = w.l1_biases;

    // Add each piece's feature contribution
    for color_idx in 0..COLOR_COUNT {
        let color = if color_idx == 0 { Color::White } else { Color::Black };
        for piece_idx in 0..PIECE_COUNT {
            let piece: Piece = unsafe { std::mem::transmute(piece_idx as u8) };
            let mut bb = board.pieces[color_idx][piece_idx];
            while bb != 0 {
                let sq = pop_lsb(&mut bb);
                let wi = feature_index_white(piece, color, sq);
                let bi = feature_index_black(piece, color, sq);
                for j in 0..L1_SIZE {
                    acc.white[j] += w.l1_weights[wi][j];
                    acc.black[j] += w.l1_weights[bi][j];
                }
            }
        }
    }
}

/// Incrementally add a piece feature to the accumulator.
#[inline]
pub fn accumulator_add(acc: &mut Accumulator, piece: Piece, color: Color, sq: u8) {
    let w = weights();
    let wi = feature_index_white(piece, color, sq);
    let bi = feature_index_black(piece, color, sq);
    for j in 0..L1_SIZE {
        acc.white[j] += w.l1_weights[wi][j];
        acc.black[j] += w.l1_weights[bi][j];
    }
}

/// Incrementally remove a piece feature from the accumulator.
#[inline]
pub fn accumulator_remove(acc: &mut Accumulator, piece: Piece, color: Color, sq: u8) {
    let w = weights();
    let wi = feature_index_white(piece, color, sq);
    let bi = feature_index_black(piece, color, sq);
    for j in 0..L1_SIZE {
        acc.white[j] -= w.l1_weights[wi][j];
        acc.black[j] -= w.l1_weights[bi][j];
    }
}

/// Incrementally move a piece (remove from `from`, add to `to`).
#[inline]
pub fn accumulator_move(acc: &mut Accumulator, piece: Piece, color: Color, from: u8, to: u8) {
    let w = weights();
    let wi_from = feature_index_white(piece, color, from);
    let wi_to   = feature_index_white(piece, color, to);
    let bi_from = feature_index_black(piece, color, from);
    let bi_to   = feature_index_black(piece, color, to);
    for j in 0..L1_SIZE {
        acc.white[j] += w.l1_weights[wi_to][j] - w.l1_weights[wi_from][j];
        acc.black[j] += w.l1_weights[bi_to][j] - w.l1_weights[bi_from][j];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_new() {
        let acc = Accumulator::new();
        assert!(acc.white.iter().all(|&v| v == 0.0));
        assert!(acc.black.iter().all(|&v| v == 0.0));
    }
}
