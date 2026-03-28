pub mod kbnk;

use crate::bitboard::*;
use crate::board::Board;

pub fn evaluate_endgame(board: &Board) -> Option<i32> {
    #[cfg(not(feature = "kbnk-scaling"))]
    {
        let _ = board;
        return None;
    }

    #[cfg(feature = "kbnk-scaling")]
    {
    let w_pawns = popcount(board.pieces[Color::White.index()][Piece::Pawn.index()]);
    let b_pawns = popcount(board.pieces[Color::Black.index()][Piece::Pawn.index()]);
    if w_pawns > 0 || b_pawns > 0 {
        return None;
    }

    let w_rooks = popcount(board.pieces[Color::White.index()][Piece::Rook.index()]);
    let b_rooks = popcount(board.pieces[Color::Black.index()][Piece::Rook.index()]);
    let w_queens = popcount(board.pieces[Color::White.index()][Piece::Queen.index()]);
    let b_queens = popcount(board.pieces[Color::Black.index()][Piece::Queen.index()]);
    if w_rooks > 0 || b_rooks > 0 || w_queens > 0 || b_queens > 0 {
        return None;
    }

    let w_bishops = popcount(board.pieces[Color::White.index()][Piece::Bishop.index()]);
    let b_bishops = popcount(board.pieces[Color::Black.index()][Piece::Bishop.index()]);
    let w_knights = popcount(board.pieces[Color::White.index()][Piece::Knight.index()]);
    let b_knights = popcount(board.pieces[Color::Black.index()][Piece::Knight.index()]);

    let w_minor_count = w_bishops + w_knights;
    let b_minor_count = b_bishops + b_knights;

    if w_minor_count == 2 && w_bishops == 1 && w_knights == 1 && b_minor_count == 0 {
        return Some(kbnk::eval_kbnk(board, Color::White));
    }

    if b_minor_count == 2 && b_bishops == 1 && b_knights == 1 && w_minor_count == 0 {
        return Some(-kbnk::eval_kbnk(board, Color::Black));
    }

    None
    }
}
