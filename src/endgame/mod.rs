pub mod kbnk;

use crate::bitboard::*;
use crate::board::Board;

#[cfg(feature = "kbnk-scaling")]
const KBNK_WHITE: u64 = (1u64 << 24) | (1u64 << 32);
#[cfg(feature = "kbnk-scaling")]
const KBNK_BLACK: u64 = (1u64 << 28) | (1u64 << 36);

#[cfg(feature = "kbnk-scaling")]
fn material_key(board: &Board) -> u64 {
    let wp = popcount(board.pieces[Color::White.index()][Piece::Pawn.index()]) as u64;
    let bp = popcount(board.pieces[Color::Black.index()][Piece::Pawn.index()]) as u64;
    let wq = popcount(board.pieces[Color::White.index()][Piece::Queen.index()]) as u64;
    let bq = popcount(board.pieces[Color::Black.index()][Piece::Queen.index()]) as u64;
    let wr = popcount(board.pieces[Color::White.index()][Piece::Rook.index()]) as u64;
    let br = popcount(board.pieces[Color::Black.index()][Piece::Rook.index()]) as u64;
    let wb = popcount(board.pieces[Color::White.index()][Piece::Bishop.index()]) as u64;
    let bb = popcount(board.pieces[Color::Black.index()][Piece::Bishop.index()]) as u64;
    let wn = popcount(board.pieces[Color::White.index()][Piece::Knight.index()]) as u64;
    let bn = popcount(board.pieces[Color::Black.index()][Piece::Knight.index()]) as u64;

    wp | (bp << 4) | (wq << 8) | (bq << 12) | (wr << 16) | (br << 20)
        | (wb << 24) | (bb << 28) | (wn << 32) | (bn << 36)
}

pub fn evaluate_endgame(board: &Board) -> Option<i32> {
    #[cfg(not(feature = "kbnk-scaling"))]
    {
        let _ = board;
        return None;
    }

    #[cfg(feature = "kbnk-scaling")]
    {
        let key = material_key(board);
        match key {
            k if k == KBNK_WHITE => Some(kbnk::eval_kbnk(board, Color::White)),
            k if k == KBNK_BLACK => Some(-kbnk::eval_kbnk(board, Color::Black)),
            _ => None,
        }
    }
}
