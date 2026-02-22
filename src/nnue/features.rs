//! Feature index encoding for NNUE inputs.
//!
//! Features are encoded as: color * 384 + piece * 64 + square
//! White's perspective uses the raw encoding.
//! Black's perspective flips color and mirrors the square vertically.

use crate::bitboard::*;

use super::INPUT_SIZE;

/// Number of king buckets used by the planned king-relative feature encoding.
/// We start with a compact mapping of 10 buckets (center, edges, corners, etc.).
pub const KING_BUCKETS: usize = 10;

/// Map a king square (0..63) to a king-bucket index (0..KING_BUCKETS-1).
/// This mapping uses horizontal mirroring: if the king is on files e..h
/// (4..7) we mirror the file so symmetric squares share the same bucket.
#[inline]
pub fn king_bucket_of(sq: u8) -> usize {
    // File (0..7) and rank (0..7)
    let file = sq & 7;
    let rank = sq >> 3;

    // Mirror horizontally so files 4..7 map to 3..0
    let file_m = if file >= 4 { 7 - file } else { file };

    // Simple handcrafted partition into 10 zones. The goal is compactness
    // and to separate center/castled/corner-like locations.
    // bucket 0: center 3x3 (files 2..4, ranks 2..4) -> after mirroring files 2..3
    if file_m >= 2 && file_m <= 3 && rank >= 2 && rank <= 4 {
        return 0;
    }
    // bucket 1: near-center ring
    if file_m >= 1 && file_m <= 4 && rank >= 1 && rank <= 6 {
        return 1;
    }
    // bucket 2: kingside (mirrored) near edges
    if file_m >= 3 && rank >= 2 && rank <= 5 {
        return 2;
    }
    // bucket 3: first rank (castled-ish)
    if rank == 0 { return 3; }
    // bucket 4: second rank
    if rank == 1 { return 4; }
    // bucket 5: seventh rank
    if rank == 6 { return 5; }
    // bucket 6: eighth rank
    if rank == 7 { return 6; }
    // bucket 7: queenside near corners
    if file_m <= 1 && (rank <= 2 || rank >= 5) { return 7; }
    // bucket 8: far edges
    if file_m >= 4 || rank <= 0 || rank >= 7 { return 8; }
    // fallback bucket 9: everything else
    9
}

/// Number of piece types excluding the king
pub const PIECES_EX_KING: usize = 5; // pawn, knight, bishop, rook, queen
/// Squares per piece
pub const SQUARES_PER_PIECE: usize = 64;
/// Per-color features per bucket: 5 piece types * 64 squares
pub const PER_COLOR_BUCKET: usize = PIECES_EX_KING * SQUARES_PER_PIECE; // 320
/// Total features per bucket (both colors)
pub const PER_BUCKET_FEATURES: usize = PER_COLOR_BUCKET * 2; // 640

/// Compute index for piece type excluding king. Returns None for `Piece::King`.
#[inline]
pub fn piece_index_no_king(piece: Piece) -> Option<usize> {
    match piece {
        Piece::Pawn => Some(0),
        Piece::Knight => Some(1),
        Piece::Bishop => Some(2),
        Piece::Rook => Some(3),
        Piece::Queen => Some(4),
        Piece::King => None,
    }
}

/// Half-KP feature index for white's perspective.
/// Layout per bucket: (own 5*64) then (opp 5*64)
#[inline]
pub fn feature_index_halfkp_white(piece: Piece, color: Color, sq: u8, king_sq: u8) -> usize {
    let bucket = king_bucket_of(king_sq);
    let piece_no_king = piece_index_no_king(piece).expect("King has no HalfKP feature");
    let color_offset = match color {
        Color::White => 0,
        Color::Black => PER_COLOR_BUCKET,
    };
    bucket * PER_BUCKET_FEATURES + color_offset + piece_no_king * 64 + sq as usize
}

/// Half-KP feature index for black's perspective. Squares are vertically flipped.
#[inline]
pub fn feature_index_halfkp_black(piece: Piece, color: Color, sq: u8, king_sq: u8) -> usize {
    let flipped = sq ^ 56;
    let bucket = king_bucket_of(king_sq);
    let piece_no_king = piece_index_no_king(piece).expect("King has no HalfKP feature");
    let color_offset = match color {
        Color::Black => 0,
        Color::White => PER_COLOR_BUCKET,
    };
    bucket * PER_BUCKET_FEATURES + color_offset + piece_no_king * 64 + flipped as usize
}

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

    #[test]
    fn test_king_bucket_mapping() {
        // Center square e4 should be in bucket 0 (center)
        assert_eq!(king_bucket_of(sq::E4), 0);

        // Corner / first-rank castled-like squares map to rank-based buckets
        assert_eq!(king_bucket_of(sq::H1), 3);
        assert_eq!(king_bucket_of(sq::A1), 3);

        // Top rank squares map to bucket 6
        assert_eq!(king_bucket_of(sq::E8), 6);

        // A square far from center on queenside should map to a non-center bucket
        let b = king_bucket_of(sq::A7);
        assert!(b < KING_BUCKETS);
    }
}
