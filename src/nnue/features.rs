use crate::bitboard::*;
use crate::board::Board;

pub const KING_BUCKETS: usize = 10;

// Mapper indexed by [rank][mirrored_file], where mirrored_file maps A/H->0, B/G->1, C/F->2, D/E->3.
pub const KING_BUCKET_MAPPER: [[usize; 4]; 8] = [
    [3, 3, 3, 3],
    [4, 1, 1, 1],
    [7, 1, 0, 0],
    [9, 1, 0, 0],
    [9, 1, 0, 0],
    [7, 1, 1, 1],
    [5, 1, 1, 1],
    [6, 6, 6, 6],
];

#[inline]
pub fn king_bucket_of(sq: u8) -> usize {
    let file = sq & 7;
    let rank = sq >> 3;
    let file_m = if file >= 4 { 7 - file } else { file };
    KING_BUCKET_MAPPER[rank as usize][file_m as usize]
}

pub const PIECES: usize = 6;
pub const SQUARES_PER_PIECE: usize = 64;
pub const PER_COLOR_BUCKET: usize = PIECES * SQUARES_PER_PIECE;
pub const PER_BUCKET_FEATURES: usize = PER_COLOR_BUCKET * 2;
pub const FT_SIZE: usize = KING_BUCKETS * PER_BUCKET_FEATURES;

#[inline]
pub fn feature_index_halfka_white(piece: Piece, color: Color, sq: u8, king_sq: u8) -> usize {
    let bucket = king_bucket_of(king_sq);
    let color_offset = match color { Color::White => 0, Color::Black => PER_COLOR_BUCKET };
    bucket * PER_BUCKET_FEATURES + color_offset + piece.index() * 64 + sq as usize
}

#[inline]
pub fn feature_index_halfka_black(piece: Piece, color: Color, sq: u8, king_sq: u8) -> usize {
    let flipped = sq ^ 56;
    let flipped_king = king_sq ^ 56;
    let bucket = king_bucket_of(flipped_king);
    let color_offset = match color { Color::Black => 0, Color::White => PER_COLOR_BUCKET };
    bucket * PER_BUCKET_FEATURES + color_offset + piece.index() * 64 + flipped as usize
}

pub struct HalfKaFeatures {
    pub white: Vec<usize>,
    pub black: Vec<usize>,
}

pub fn transform_halfka(board: &Board) -> HalfKaFeatures {
    let wk = board.king_sq(Color::White);
    let bk = board.king_sq(Color::Black);

    let mut white = Vec::with_capacity(32);
    let mut black = Vec::with_capacity(32);

    for color_idx in 0..2 {
        let color = if color_idx == 0 { Color::White } else { Color::Black };
        for piece_idx in 0..PIECE_COUNT {
            let piece = Piece::from_index(piece_idx);
            let mut bb = board.pieces[color.index()][piece_idx];
            while bb != 0 {
                let sq = pop_lsb(&mut bb);
                white.push(feature_index_halfka_white(piece, color, sq, wk));
                black.push(feature_index_halfka_black(piece, color, sq, bk));
            }
        }
    }

    HalfKaFeatures { white, black }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_king_bucket_mapping() {
        assert_eq!(king_bucket_of(sq::E4), 0);
        assert_eq!(king_bucket_of(sq::H1), 3);
        assert_eq!(king_bucket_of(sq::A1), 3);
        assert_eq!(king_bucket_of(sq::E8), 6);
        let b = king_bucket_of(sq::A7);
        assert!(b < KING_BUCKETS);
    }

    #[test]
    fn test_king_bucket_mapper_layout() {
        assert_eq!(KING_BUCKET_MAPPER[0], [3, 3, 3, 3]);
        assert_eq!(KING_BUCKET_MAPPER[1], [4, 1, 1, 1]);
        assert_eq!(KING_BUCKET_MAPPER[2], [7, 1, 0, 0]);
        assert_eq!(KING_BUCKET_MAPPER[3], [9, 1, 0, 0]);
        assert_eq!(KING_BUCKET_MAPPER[4], [9, 1, 0, 0]);
        assert_eq!(KING_BUCKET_MAPPER[5], [7, 1, 1, 1]);
        assert_eq!(KING_BUCKET_MAPPER[6], [5, 1, 1, 1]);
        assert_eq!(KING_BUCKET_MAPPER[7], [6, 6, 6, 6]);
    }

    #[test]
    fn test_king_bucket_all_squares_in_range() {
        for sq in 0..64u8 {
            let b = king_bucket_of(sq);
            assert!(b < KING_BUCKETS, "sq {} mapped to bucket {} (>= {})", sq, b, KING_BUCKETS);
        }
    }

    #[test]
    fn test_halfka_white_index_bounds() {
        let total = KING_BUCKETS * PER_BUCKET_FEATURES;
        for king in [sq::E1, sq::A1, sq::H1, sq::D4, sq::G8] {
            for &piece in &[Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King] {
                for &color in &[Color::White, Color::Black] {
                    for sq in 0..64u8 {
                        let idx = feature_index_halfka_white(piece, color, sq, king);
                        assert!(idx < total, "white idx {} >= {} for king={} piece={:?} color={:?} sq={}",
                            idx, total, king, piece, color, sq);
                    }
                }
            }
        }
    }

    #[test]
    fn test_halfka_black_index_bounds() {
        let total = KING_BUCKETS * PER_BUCKET_FEATURES;
        for king in [sq::E8, sq::A8, sq::H8, sq::D5, sq::G1] {
            for &piece in &[Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King] {
                for &color in &[Color::White, Color::Black] {
                    for sq in 0..64u8 {
                        let idx = feature_index_halfka_black(piece, color, sq, king);
                        assert!(idx < total, "black idx {} >= {} for king={} piece={:?} color={:?} sq={}",
                            idx, total, king, piece, color, sq);
                    }
                }
            }
        }
    }

    #[test]
    fn test_halfka_perspective_symmetry() {
        let w = feature_index_halfka_white(Piece::Pawn, Color::White, sq::E2, sq::E1);
        let b = feature_index_halfka_black(Piece::Pawn, Color::Black, sq::E7, sq::E8);
        assert_eq!(w, b);
    }

    #[test]
    fn test_halfka_different_buckets_differ() {
        let idx_a = feature_index_halfka_white(Piece::Pawn, Color::White, sq::E2, sq::E1);
        let idx_b = feature_index_halfka_white(Piece::Pawn, Color::White, sq::E2, sq::E4);
        assert_ne!(idx_a, idx_b);
    }

    #[test]
    fn test_halfka_includes_king() {
        let idx_wk = feature_index_halfka_white(Piece::King, Color::White, sq::E1, sq::E1);
        let idx_bk = feature_index_halfka_white(Piece::King, Color::Black, sq::E8, sq::E1);
        assert_ne!(idx_wk, idx_bk);
        let total = KING_BUCKETS * PER_BUCKET_FEATURES;
        assert!(idx_wk < total);
        assert!(idx_bk < total);
    }

}
