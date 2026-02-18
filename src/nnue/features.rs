//! Feature index encoding for NNUE inputs.
//!
//! Features are encoded as: color * 384 + piece * 64 + square
//! White's perspective uses the raw encoding.
//! Black's perspective flips color and mirrors the square vertically.

use crate::bitboard::*;

use super::INPUT_SIZE;

/// Compute the feature index for a piece on a square from white's perspective.
///
/// White pieces: piece * 64 + sq
/// Black pieces: 384 + piece * 64 + sq
#[inline]
pub fn feature_index_white(piece: Piece, color: Color, sq: u8) -> usize {
    let color_offset = match color {
        Color::White => 0,
        Color::Black => 384,
    };
    color_offset + piece.index() * 64 + sq as usize
}

/// Compute the feature index for a piece on a square from black's perspective.
///
/// Black pieces: piece * 64 + flip(sq)
/// White pieces: 384 + piece * 64 + flip(sq)
#[inline]
pub fn feature_index_black(piece: Piece, color: Color, sq: u8) -> usize {
    let flipped = sq ^ 56; // vertical mirror
    let color_offset = match color {
        Color::Black => 0,   // black's own pieces first
        Color::White => 384, // opponent's pieces second
    };
    color_offset + piece.index() * 64 + flipped as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_index_bounds() {
        // White pawn on a1 from white's perspective
        let idx = feature_index_white(Piece::Pawn, Color::White, 0);
        assert_eq!(idx, 0);

        // Black king on h8 from white's perspective
        let idx = feature_index_white(Piece::King, Color::Black, 63);
        assert_eq!(idx, 384 + 5 * 64 + 63);
        assert!(idx < INPUT_SIZE);

        // From black's perspective: black pawn on a8 (sq=56), after flip = sq=0
        let idx = feature_index_black(Piece::Pawn, Color::Black, 56);
        assert_eq!(idx, 0 * 64 + 0); // color_offset=0, piece=0, flipped=0

        // White king on e1 (sq=4) from black's perspective: flip = 60
        let idx = feature_index_black(Piece::King, Color::White, 4);
        assert_eq!(idx, 384 + 5 * 64 + 60);
        assert!(idx < INPUT_SIZE);
    }

    #[test]
    fn test_feature_index_symmetry() {
        // A white pawn on e2 from white's perspective should map to the same
        // feature as a black pawn on e7 from black's perspective (mirrored).
        let w_idx = feature_index_white(Piece::Pawn, Color::White, sq::E2);
        let b_idx = feature_index_black(Piece::Pawn, Color::Black, sq::E7);
        assert_eq!(w_idx, b_idx, "Symmetric positions should have same feature index");
    }
}
