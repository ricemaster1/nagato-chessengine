use crate::bitboard::*;
use crate::board::Board;
#[cfg(test)]
use super::INPUT_SIZE;

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
pub fn feature_index_halfka_white(piece: Piece, color: Color, sq: u8, king_sq: u8) -> usize {
    let bucket = king_bucket_of(king_sq);
    let piece_no_king = piece_index_no_king(piece).expect("King has no HalfKP feature");
    let color_offset = match color { Color::White => 0, Color::Black => PER_COLOR_BUCKET };
    bucket * PER_BUCKET_FEATURES + color_offset + piece_no_king * 64 + sq as usize
}

#[inline]
pub fn feature_index_halfka_black(piece: Piece, color: Color, sq: u8, king_sq: u8) -> usize {
    let flipped = sq ^ 56;
    let flipped_king = king_sq ^ 56;
    let bucket = king_bucket_of(flipped_king);
    let piece_no_king = piece_index_no_king(piece).expect("King has no HalfKP feature");
    let color_offset = match color { Color::Black => 0, Color::White => PER_COLOR_BUCKET };
    bucket * PER_BUCKET_FEATURES + color_offset + piece_no_king * 64 + flipped as usize
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
            if piece_index_no_king(piece).is_none() {
                continue;
            }

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

pub const THREAT_PIECES: usize = 5;
pub const THREAT_PER_COLOR: usize = THREAT_PIECES * 64;
pub const THREAT_FEATURES: usize = THREAT_PER_COLOR * 2;

pub const FT_SIZE: usize = KING_BUCKETS * PER_BUCKET_FEATURES;
pub const FT_SIZE_THREATS: usize = FT_SIZE + THREAT_FEATURES;

#[inline]
pub fn threat_feature_white(piece: Piece, sq: u8, attacker_is_white: bool) -> usize {
    let pi = piece_index_no_king(piece).expect("King has no threat feature");
    let color_off = if attacker_is_white { 0 } else { THREAT_PER_COLOR };
    FT_SIZE + color_off + pi * 64 + sq as usize
}

#[inline]
pub fn threat_feature_black(piece: Piece, sq: u8, attacker_is_white: bool) -> usize {
    let pi = piece_index_no_king(piece).expect("King has no threat feature");
    let flipped = sq ^ 56;
    let color_off = if attacker_is_white { THREAT_PER_COLOR } else { 0 };
    FT_SIZE + color_off + pi * 64 + flipped as usize
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
    fn test_halfka_black_index_bounds() {
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
    fn test_halfka_perspective_symmetry() {
        let w = feature_index_halfkp_white(Piece::Pawn, Color::White, sq::E2, sq::E1);
        let b = feature_index_halfkp_black(Piece::Pawn, Color::Black, sq::E7, sq::E8);
        assert_eq!(w, b);
    }

    #[test]
    fn test_halfka_different_buckets_differ() {
        let idx_a = feature_index_halfkp_white(Piece::Pawn, Color::White, sq::E2, sq::E1);
        let idx_b = feature_index_halfkp_white(Piece::Pawn, Color::White, sq::E2, sq::E4);
        assert_ne!(idx_a, idx_b);
    }

    #[test]
    fn test_transform_halfka_startpos_size() {
        let board = Board::start_pos();
        let transformed = transform_halfkp(&board);
        assert_eq!(transformed.white.len(), 30);
        assert_eq!(transformed.black.len(), 30);
    }

    #[test]
    #[should_panic(expected = "King has no HalfKP feature")]
    fn test_halfka_rejects_king_piece() {
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

    #[test]
    fn test_threat_feature_bounds() {
        for &piece in &[Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen] {
            for sq in 0..64u8 {
                let w = threat_feature_white(piece, sq, true);
                let b = threat_feature_black(piece, sq, false);
                assert!(w < FT_SIZE_THREATS, "white threat idx {} >= {}", w, FT_SIZE_THREATS);
                assert!(b < FT_SIZE_THREATS, "black threat idx {} >= {}", b, FT_SIZE_THREATS);
            }
        }
    }

    #[test]
    fn test_threat_feature_symmetry() {
        let w = threat_feature_white(Piece::Pawn, sq::E4, true);
        let b = threat_feature_black(Piece::Pawn, sq::E5, false);
        assert_eq!(w, b);
    }
}
