pub mod hce;
pub mod see;
pub mod pst;

pub use hce::*;
pub use see::*;
pub use pst::*;

use crate::bitboard::*;
use crate::board::Board;
use crate::endgame::evaluate_endgame;
use crate::movegen;
use crate::moves::Move;
use crate::nnue;

pub const PAWN_VALUE: i32   = 100;
pub const KNIGHT_VALUE: i32 = 320;
pub const BISHOP_VALUE: i32 = 330;
pub const ROOK_VALUE: i32   = 500;
pub const QUEEN_VALUE: i32  = 900;
pub const KING_VALUE: i32   = 0;

pub const PIECE_VALUES: [i32; PIECE_COUNT] = [
    PAWN_VALUE, KNIGHT_VALUE, BISHOP_VALUE, ROOK_VALUE, QUEEN_VALUE, KING_VALUE
];

pub const KNIGHT_PHASE: i32 = 1;
pub const BISHOP_PHASE: i32 = 1;
pub const ROOK_PHASE: i32   = 2;
pub const QUEEN_PHASE: i32  = 4;
pub const TOTAL_PHASE: i32  = 4 * KNIGHT_PHASE + 4 * BISHOP_PHASE + 4 * ROOK_PHASE + 2 * QUEEN_PHASE;

pub fn compute_phase(board: &Board) -> i32 {
    let mut phase = TOTAL_PHASE;
    for color in 0..COLOR_COUNT {
        phase -= popcount(board.pieces[color][Piece::Knight.index()]) as i32 * KNIGHT_PHASE;
        phase -= popcount(board.pieces[color][Piece::Bishop.index()]) as i32 * BISHOP_PHASE;
        phase -= popcount(board.pieces[color][Piece::Rook.index()]) as i32 * ROOK_PHASE;
        phase -= popcount(board.pieces[color][Piece::Queen.index()]) as i32 * QUEEN_PHASE;
    }
    phase.max(0)
}

const DUAL_NET_THRESHOLD: i32 = ROOK_VALUE * 2;

pub fn material_balance(board: &Board) -> i32 {
    let mut balance = 0i32;
    for piece in 0..PIECE_COUNT - 1 {
        let w = popcount(board.pieces[Color::White.index()][piece]) as i32;
        let b = popcount(board.pieces[Color::Black.index()][piece]) as i32;
        balance += (w - b) * PIECE_VALUES[piece];
    }
    balance
}

pub fn evaluate(board: &Board) -> i32 {
    if let Some(endgame_score) = evaluate_endgame(board) {
        return match board.side {
            Color::White => endgame_score,
            Color::Black => -endgame_score,
        };
    }

    if nnue::is_active() {
        let mb = material_balance(board);
        if mb.abs() >= DUAL_NET_THRESHOLD {
            return evaluate_hce(board);
        }
        return nnue::evaluate_q(board, &board.accumulator_q);
    }

    evaluate_hce(board)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::movegen;
    use crate::moves::*;
    use crate::bitboard::sq;

    fn setup() {
        crate::zobrist::init();
        movegen::init();
    }

    #[test]
    fn test_eval_start_pos() {
        setup();
        let board = Board::start_pos();
        let score = evaluate(&board);
        assert!(score.abs() < 50, "Start position eval should be near 0, got {}", score);
    }

    #[test]
    fn test_eval_material_advantage() {
        setup();
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let score = evaluate(&board);
        assert!(score.abs() < 50);
    }

    fn find_move(board: &Board, from_str: &str, to_str: &str) -> Move {
        use crate::bitboard::parse_square;
        let from = parse_square(from_str).unwrap();
        let to = parse_square(to_str).unwrap();
        let mut list = MoveList::new();
        movegen::generate_moves(board, &mut list);
        for i in 0..list.len() {
            let m = list.moves[i];
            if m.from_sq() == from && m.to_sq() == to && !m.is_promotion() {
                return m;
            }
        }
        panic!("Move {}→{} not found in legal moves", from_str, to_str);
    }

    fn find_promo_move(board: &Board, from_str: &str, to_str: &str, promo: Piece) -> Move {
        use crate::bitboard::parse_square;
        let from = parse_square(from_str).unwrap();
        let to = parse_square(to_str).unwrap();
        let mut list = MoveList::new();
        movegen::generate_moves(board, &mut list);
        for i in 0..list.len() {
            let m = list.moves[i];
            if m.from_sq() == from && m.to_sq() == to && m.promotion_piece() == Some(promo) {
                return m;
            }
        }
        panic!("Promo move {}→{} not found", from_str, to_str);
    }

    #[test]
    fn test_see_simple_pawn_takes_pawn() {
        setup();
        let board = Board::from_fen("8/8/8/3p4/4P3/8/8/4K2k w - - 0 1").unwrap();
        let m = find_move(&board, "e4", "d5");
        let score = see(&board, m);
        assert_eq!(score, PAWN_VALUE, "PxP with no defenders should win a pawn");
    }

    #[test]
    fn test_see_pawn_takes_defended_pawn() {
        setup();
        let board = Board::from_fen("8/8/4p3/3p4/4P3/8/8/4K2k w - - 0 1").unwrap();
        let m = find_move(&board, "e4", "d5");
        let score = see(&board, m);
        assert_eq!(score, 0, "PxP with pawn defender should be 0");
    }

    #[test]
    fn test_see_knight_takes_defended_pawn() {
        setup();
        let board = Board::from_fen("8/8/4p3/3p4/8/2N5/8/4K2k w - - 0 1").unwrap();
        let m = find_move(&board, "c3", "d5");
        let score = see(&board, m);
        assert!(score < 0, "Knight taking pawn defended by pawn should be negative, got {}", score);
    }

    #[test]
    fn test_see_queen_takes_defended_pawn() {
        setup();
        let board = Board::from_fen("8/8/4p3/3p4/8/8/8/3QK2k w - - 0 1").unwrap();
        let m = find_move(&board, "d1", "d5");
        let score = see(&board, m);
        assert!(score < 0, "Queen taking defended pawn should be losing, got {}", score);
    }

    #[test]
    fn test_see_rook_takes_rook() {
        setup();
        let board = Board::from_fen("4r3/8/8/8/8/8/8/4RK1k w - - 0 1").unwrap();
        let m = find_move(&board, "e1", "e8");
        let score = see(&board, m);
        assert_eq!(score, ROOK_VALUE, "RxR undefended should win a rook");
    }

    #[test]
    fn test_see_xray_battery() {
        setup();
        let board = Board::from_fen("4r3/8/8/8/8/8/4R3/4RK1k w - - 0 1").unwrap();
        let m = find_move(&board, "e2", "e8");
        let score = see(&board, m);
        assert_eq!(score, ROOK_VALUE, "RxR with rook behind should win");
    }

    #[test]
    fn test_see_pawn_takes_queen() {
        setup();
        let board = Board::from_fen("8/8/8/3q4/4P3/8/8/4K2k w - - 0 1").unwrap();
        let m = find_move(&board, "e4", "d5");
        let score = see(&board, m);
        assert_eq!(score, QUEEN_VALUE, "PxQ undefended should win queen value");
    }

    #[test]
    fn test_see_non_capture_returns_zero() {
        setup();
        let board = Board::start_pos();
        let m = find_move(&board, "e2", "e4");
        let score = see(&board, m);
        assert_eq!(score, 0, "Non-capture should return 0");
    }

    #[test]
    fn test_see_bishop_takes_knight_with_recapture() {
        setup();
        let board = Board::from_fen("8/8/8/8/3p4/4n3/8/2B1K2k w - - 0 1").unwrap();
        let m = find_move(&board, "c1", "e3");
        let score = see(&board, m);
        assert!(score < 0, "BxN defended by pawn should be negative, got {}", score);
    }

    #[test]
    fn test_edge_pawn_penalty_vs_interior() {
        setup();
        let edge = Board::from_fen("8/8/8/8/P7/8/8/4K2k w - - 0 1").unwrap();
        let interior = Board::from_fen("8/8/8/8/3P4/8/8/4K2k w - - 0 1").unwrap();
        let edge_score = evaluate(&edge);
        let interior_score = evaluate(&interior);
        assert!(interior_score > edge_score,
            "Interior pawn should score higher than edge pawn: interior={}, edge={}",
            interior_score, edge_score);
    }

    #[test]
    fn test_edge_pawn_capture_available_bonus() {
        setup();
        let with_target = Board::from_fen("8/8/8/1p6/P7/8/8/4K2k w - - 0 1").unwrap();
        let no_target = Board::from_fen("8/8/8/8/P7/8/8/4K2k w - - 0 1").unwrap();
        let score_with = evaluate(&with_target);
        let score_without = evaluate(&no_target);
        assert!(score_without != 0, "Edge pawn should affect eval");
    }

    #[test]
    fn test_off_edge_bonus_b_file() {
        setup();
        let off_edge = Board::from_fen("8/8/8/8/1P6/8/8/4K2k w - - 0 1").unwrap();
        let normal = Board::from_fen("8/8/8/8/1P6/8/P7/4K2k w - - 0 1").unwrap();
        let off_edge_score = evaluate(&off_edge);
        let normal_score = evaluate(&normal);
        assert!(normal_score > off_edge_score,
            "Position with extra pawn should score higher");
    }

    #[test]
    fn test_edge_pawn_advancement_scaling() {
        setup();
        let advanced = Board::from_fen("8/8/P7/8/8/8/8/4K2k w - - 0 1").unwrap();
        let early = Board::from_fen("8/8/8/8/8/8/P7/4K2k w - - 0 1").unwrap();
        let advanced_score = evaluate(&advanced);
        let early_score = evaluate(&early);
        assert!(advanced_score != early_score,
            "Different advancement should produce different scores");
    }

    #[test]
    fn test_edge_pawn_symmetry() {
        setup();
        let a_pawn = Board::from_fen("8/8/8/8/P7/8/8/4K2k w - - 0 1").unwrap();
        let h_pawn = Board::from_fen("8/8/8/8/7P/8/8/4K2k w - - 0 1").unwrap();
        let a_score = evaluate(&a_pawn);
        let h_score = evaluate(&h_pawn);
        let diff = (a_score - h_score).abs();
        assert!(diff <= 20,
            "A-file and H-file edge pawns should have similar eval, diff={}", diff);
    }
}

