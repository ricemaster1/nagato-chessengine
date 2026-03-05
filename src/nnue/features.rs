use crate::bitboard::*;
use super::INPUT_SIZE;

pub const KING_BUCKETS: usize = 10;

#[inline]
pub fn king_bucket_of(sq: u8) -> usize {
    let file = sq & 7;
    let rank = sq >> 3;
    let file_m = if file >= 4 { 7 - file } else { file };
    if file_m >= 2 && file_m <= 3 && rank >= 2 && rank <= 4 {
        0
    } else if file_m >= 1 && file_m <= 4 && rank >= 1 && rank <= 6 {
        1
    } else if file_m >= 3 && rank >= 2 && rank <= 5 {
        2
    } else if rank == 0 {
        3
    } else if rank == 1 {
        4
    } else if rank == 6 {
        5
    } else if rank == 7 {
        6
    } else if file_m <= 1 && (rank <= 2 || rank >= 5) {
        7
    } else if file_m >= 4 || rank <= 0 || rank >= 7 {
        8
    } else {
        9
    }
}

pub const PIECES_EX_KING: usize = 5;
pub const SQUARES_PER_PIECE: usize = 64;
pub const PER_COLOR_BUCKET: usize = PIECES_EX_KING * SQUARES_PER_PIECE;
pub const PER_BUCKET_FEATURES: usize = PER_COLOR_BUCKET * 2;

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

#[inline]
pub fn feature_index_halfkp_white(piece: Piece, color: Color, sq: u8, king_sq: u8) -> usize {
    let bucket = king_bucket_of(king_sq);
    let piece_no_king = piece_index_no_king(piece).expect("King has no HalfKP feature");
    let color_offset = match color { Color::White => 0, Color::Black => PER_COLOR_BUCKET };
    bucket * PER_BUCKET_FEATURES + color_offset + piece_no_king * 64 + sq as usize
}

#[inline]
pub fn feature_index_halfkp_black(piece: Piece, color: Color, sq: u8, king_sq: u8) -> usize {
    let flipped = sq ^ 56;
    let flipped_king = king_sq ^ 56;
    let bucket = king_bucket_of(flipped_king);
    let piece_no_king = piece_index_no_king(piece).expect("King has no HalfKP feature");
    let color_offset = match color { Color::Black => 0, Color::White => PER_COLOR_BUCKET };
    bucket * PER_BUCKET_FEATURES + color_offset + piece_no_king * 64 + flipped as usize
}

#[inline]
pub fn feature_index_white(piece: Piece, color: Color, sq: u8) -> usize {
    let color_offset = match color { Color::White => 0, Color::Black => 384 };
    color_offset + piece.index() * 64 + sq as usize
}

#[inline]
pub fn feature_index_black(piece: Piece, color: Color, sq: u8) -> usize {
    let flipped = sq ^ 56;
    let color_offset = match color { Color::Black => 0, Color::White => 384 };
    color_offset + piece.index() * 64 + flipped as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_index_bounds() {
        let idx = feature_index_white(Piece::Pawn, Color::White, 0);
        assert_eq!(idx, 0);

        let idx = feature_index_white(Piece::King, Color::Black, 63);
        assert_eq!(idx, 384 + 5 * 64 + 63);
        assert!(idx < INPUT_SIZE);

        let idx = feature_index_black(Piece::Pawn, Color::Black, 56);
        assert_eq!(idx, 0 * 64 + 0);

        let idx = feature_index_black(Piece::King, Color::White, 4);
        assert_eq!(idx, 384 + 5 * 64 + 60);
        assert!(idx < INPUT_SIZE);
    }

    #[test]
    fn test_feature_index_symmetry() {
        let w_idx = feature_index_white(Piece::Pawn, Color::White, sq::E2);
        let b_idx = feature_index_black(Piece::Pawn, Color::Black, sq::E7);
        assert_eq!(w_idx, b_idx);
    }

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
    fn test_king_bucket_all_squares_in_range() {
        for sq in 0..64u8 {
            let b = king_bucket_of(sq);
            assert!(b < KING_BUCKETS, "sq {} mapped to bucket {} (>= {})", sq, b, KING_BUCKETS);
        }
    }

    #[test]
    fn test_halfkp_white_index_bounds() {
        let total = KING_BUCKETS * PER_BUCKET_FEATURES;
        for king in [sq::E1, sq::A1, sq::H1, sq::D4, sq::G8] {
            for &piece in &[Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen] {
                for &color in &[Color::White, Color::Black] {
                    for sq in 0..64u8 {
                        let idx = feature_index_halfkp_white(piece, color, sq, king);
                        assert!(idx < total, "white idx {} >= {} for king={} piece={:?} color={:?} sq={}",
                            idx, total, king, piece, color, sq);
                    }
                }
            }
        }
    }

    #[test]
    fn test_halfkp_black_index_bounds() {
        let total = KING_BUCKETS * PER_BUCKET_FEATURES;
        for king in [sq::E8, sq::A8, sq::H8, sq::D5, sq::G1] {
            for &piece in &[Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen] {
                for &color in &[Color::White, Color::Black] {
                    for sq in 0..64u8 {
                        let idx = feature_index_halfkp_black(piece, color, sq, king);
                        assert!(idx < total, "black idx {} >= {} for king={} piece={:?} color={:?} sq={}",
                            idx, total, king, piece, color, sq);
                    }
                }
            }
        }
    }

    #[test]
    fn test_halfkp_perspective_symmetry() {
        let w = feature_index_halfkp_white(Piece::Pawn, Color::White, sq::E2, sq::E1);
        let b = feature_index_halfkp_black(Piece::Pawn, Color::Black, sq::E7, sq::E8);
        assert_eq!(w, b);
    }

    #[test]
    fn test_halfkp_different_buckets_differ() {
        let idx_a = feature_index_halfkp_white(Piece::Pawn, Color::White, sq::E2, sq::E1);
        let idx_b = feature_index_halfkp_white(Piece::Pawn, Color::White, sq::E2, sq::E4);
        assert_ne!(idx_a, idx_b);
    }

    #[test]
    #[should_panic(expected = "King has no HalfKP feature")]
    fn test_halfkp_rejects_king_piece() {
        feature_index_halfkp_white(Piece::King, Color::White, sq::E1, sq::E1);
    }

    #[test]
    fn test_piece_index_no_king() {
        assert_eq!(piece_index_no_king(Piece::Pawn), Some(0));
        assert_eq!(piece_index_no_king(Piece::Knight), Some(1));
        assert_eq!(piece_index_no_king(Piece::Bishop), Some(2));
        assert_eq!(piece_index_no_king(Piece::Rook), Some(3));
        assert_eq!(piece_index_no_king(Piece::Queen), Some(4));
        assert_eq!(piece_index_no_king(Piece::King), None);
    }
}
