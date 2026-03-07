use crate::bitboard::*;
use crate::board::Board;

use super::L1_SIZE;
use super::features::{
    feature_index_white,
    feature_index_black,
    feature_index_halfkp_white,
    feature_index_halfkp_black,
};
use super::network::{weights, weights_q};

#[derive(Clone)]
pub struct Accumulator {
    pub white: [f32; L1_SIZE],
    pub black: [f32; L1_SIZE],
}

impl Accumulator {
    pub fn new() -> Self {
        Accumulator { white: [0.0; L1_SIZE], black: [0.0; L1_SIZE] }
    }
}

#[derive(Clone)]
pub struct AccumulatorQ {
    pub white: [i16; L1_SIZE],
    pub black: [i16; L1_SIZE],
}

impl AccumulatorQ {
    pub fn new() -> Self {
        AccumulatorQ { white: [0; L1_SIZE], black: [0; L1_SIZE] }
    }
}

pub fn refresh_accumulator(board: &Board, acc: &mut Accumulator) {
    let w = weights();
    acc.white = w.l1_biases;
    acc.black = w.l1_biases;
    if w.version == 1 {
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
    } else {
        let white_king = board.king_sq(Color::White);
        let black_king = board.king_sq(Color::Black);
        for color_idx in 0..COLOR_COUNT {
            let color = if color_idx == 0 { Color::White } else { Color::Black };
            for piece_idx in 0..PIECE_COUNT {
                let piece: Piece = unsafe { std::mem::transmute(piece_idx as u8) };
                if piece == Piece::King { continue; }
                let mut bb = board.pieces[color_idx][piece_idx];
                while bb != 0 {
                    let sq = pop_lsb(&mut bb);
                    let wi = feature_index_halfkp_white(piece, color, sq, white_king);
                    let bi = feature_index_halfkp_black(piece, color, sq, black_king);
                    for j in 0..L1_SIZE {
                        acc.white[j] += w.l1_weights[wi][j];
                        acc.black[j] += w.l1_weights[bi][j];
                    }
                }
            }
        }
    }
}

#[inline]
pub fn accumulator_add(acc: &mut Accumulator, piece: Piece, color: Color, sq: u8, white_king: u8, black_king: u8) {
    let w = weights();
    if w.version == 1 {
        let wi = feature_index_white(piece, color, sq);
        let bi = feature_index_black(piece, color, sq);
        for j in 0..L1_SIZE { acc.white[j] += w.l1_weights[wi][j]; acc.black[j] += w.l1_weights[bi][j]; }
    } else {
        if piece == Piece::King { return; }
        let wi = feature_index_halfkp_white(piece, color, sq, white_king);
        let bi = feature_index_halfkp_black(piece, color, sq, black_king);
        for j in 0..L1_SIZE { acc.white[j] += w.l1_weights[wi][j]; acc.black[j] += w.l1_weights[bi][j]; }
    }
}

#[inline]
pub fn accumulator_remove(acc: &mut Accumulator, piece: Piece, color: Color, sq: u8, white_king: u8, black_king: u8) {
    let w = weights();
    if w.version == 1 {
        let wi = feature_index_white(piece, color, sq);
        let bi = feature_index_black(piece, color, sq);
        for j in 0..L1_SIZE { acc.white[j] -= w.l1_weights[wi][j]; acc.black[j] -= w.l1_weights[bi][j]; }
    } else {
        if piece == Piece::King { return; }
        let wi = feature_index_halfkp_white(piece, color, sq, white_king);
        let bi = feature_index_halfkp_black(piece, color, sq, black_king);
        for j in 0..L1_SIZE { acc.white[j] -= w.l1_weights[wi][j]; acc.black[j] -= w.l1_weights[bi][j]; }
    }
}

#[inline]
pub fn accumulator_move(acc: &mut Accumulator, piece: Piece, color: Color, from: u8, to: u8, white_king: u8, black_king: u8) {
    let w = weights();
    if w.version == 1 {
        let wi_from = feature_index_white(piece, color, from);
        let wi_to   = feature_index_white(piece, color, to);
        let bi_from = feature_index_black(piece, color, from);
        let bi_to   = feature_index_black(piece, color, to);
        for j in 0..L1_SIZE { acc.white[j] += w.l1_weights[wi_to][j] - w.l1_weights[wi_from][j]; acc.black[j] += w.l1_weights[bi_to][j] - w.l1_weights[bi_from][j]; }
    } else {
        if piece == Piece::King { return; }
        let wi_from = feature_index_halfkp_white(piece, color, from, white_king);
        let wi_to   = feature_index_halfkp_white(piece, color, to, white_king);
        let bi_from = feature_index_halfkp_black(piece, color, from, black_king);
        let bi_to   = feature_index_halfkp_black(piece, color, to, black_king);
        for j in 0..L1_SIZE { acc.white[j] += w.l1_weights[wi_to][j] - w.l1_weights[wi_from][j]; acc.black[j] += w.l1_weights[bi_to][j] - w.l1_weights[bi_from][j]; }
    }
}

pub fn refresh_accumulator_q(board: &Board, acc: &mut AccumulatorQ) {
    let wq = weights_q();
    acc.white = wq.ft_biases;
    acc.black = wq.ft_biases;
    if wq.version == 1 {
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
                        acc.white[j] += wq.ft_weights[wi][j];
                        acc.black[j] += wq.ft_weights[bi][j];
                    }
                }
            }
        }
    } else {
        let white_king = board.king_sq(Color::White);
        let black_king = board.king_sq(Color::Black);
        for color_idx in 0..COLOR_COUNT {
            let color = if color_idx == 0 { Color::White } else { Color::Black };
            for piece_idx in 0..PIECE_COUNT {
                let piece: Piece = unsafe { std::mem::transmute(piece_idx as u8) };
                if piece == Piece::King { continue; }
                let mut bb = board.pieces[color_idx][piece_idx];
                while bb != 0 {
                    let sq = pop_lsb(&mut bb);
                    let wi = feature_index_halfkp_white(piece, color, sq, white_king);
                    let bi = feature_index_halfkp_black(piece, color, sq, black_king);
                    for j in 0..L1_SIZE {
                        acc.white[j] += wq.ft_weights[wi][j];
                        acc.black[j] += wq.ft_weights[bi][j];
                    }
                }
            }
        }
    }
}

#[inline]
pub fn accumulator_add_q(acc: &mut AccumulatorQ, piece: Piece, color: Color, sq: u8, white_king: u8, black_king: u8) {
    let wq = weights_q();
    if wq.version == 1 {
        let wi = feature_index_white(piece, color, sq);
        let bi = feature_index_black(piece, color, sq);
        for j in 0..L1_SIZE { acc.white[j] += wq.ft_weights[wi][j]; acc.black[j] += wq.ft_weights[bi][j]; }
    } else {
        if piece == Piece::King { return; }
        let wi = feature_index_halfkp_white(piece, color, sq, white_king);
        let bi = feature_index_halfkp_black(piece, color, sq, black_king);
        for j in 0..L1_SIZE { acc.white[j] += wq.ft_weights[wi][j]; acc.black[j] += wq.ft_weights[bi][j]; }
    }
}

#[inline]
pub fn accumulator_remove_q(acc: &mut AccumulatorQ, piece: Piece, color: Color, sq: u8, white_king: u8, black_king: u8) {
    let wq = weights_q();
    if wq.version == 1 {
        let wi = feature_index_white(piece, color, sq);
        let bi = feature_index_black(piece, color, sq);
        for j in 0..L1_SIZE { acc.white[j] -= wq.ft_weights[wi][j]; acc.black[j] -= wq.ft_weights[bi][j]; }
    } else {
        if piece == Piece::King { return; }
        let wi = feature_index_halfkp_white(piece, color, sq, white_king);
        let bi = feature_index_halfkp_black(piece, color, sq, black_king);
        for j in 0..L1_SIZE { acc.white[j] -= wq.ft_weights[wi][j]; acc.black[j] -= wq.ft_weights[bi][j]; }
    }
}

#[inline]
pub fn accumulator_move_q(acc: &mut AccumulatorQ, piece: Piece, color: Color, from: u8, to: u8, white_king: u8, black_king: u8) {
    let wq = weights_q();
    if wq.version == 1 {
        let wi_from = feature_index_white(piece, color, from);
        let wi_to   = feature_index_white(piece, color, to);
        let bi_from = feature_index_black(piece, color, from);
        let bi_to   = feature_index_black(piece, color, to);
        for j in 0..L1_SIZE { acc.white[j] += wq.ft_weights[wi_to][j] - wq.ft_weights[wi_from][j]; acc.black[j] += wq.ft_weights[bi_to][j] - wq.ft_weights[bi_from][j]; }
    } else {
        if piece == Piece::King { return; }
        let wi_from = feature_index_halfkp_white(piece, color, from, white_king);
        let wi_to   = feature_index_halfkp_white(piece, color, to, white_king);
        let bi_from = feature_index_halfkp_black(piece, color, from, black_king);
        let bi_to   = feature_index_halfkp_black(piece, color, to, black_king);
        for j in 0..L1_SIZE { acc.white[j] += wq.ft_weights[wi_to][j] - wq.ft_weights[wi_from][j]; acc.black[j] += wq.ft_weights[bi_to][j] - wq.ft_weights[bi_from][j]; }
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

    #[test]
    fn test_accumulator_q_new() {
        let acc = AccumulatorQ::new();
        assert!(acc.white.iter().all(|&v| v == 0));
        assert!(acc.black.iter().all(|&v| v == 0));
    }
}
