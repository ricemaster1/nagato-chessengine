use crate::bitboard::*;
use crate::board::Board;

use super::L1_SIZE;
use super::NUM_PSQT_BUCKETS;
use super::features::{
    feature_index_white,
    feature_index_black,
    feature_index_halfkp_white,
    feature_index_halfkp_black,
    king_bucket_of,
};
use super::network::{weights, weights_q};
use super::simd;

pub const SQ_NONE: u8 = 64;

#[derive(Clone, Copy)]
pub struct DirtyPiece {
    pub piece: Piece,
    pub color: Color,
    pub from: u8,
    pub to: u8,
}

impl DirtyPiece {
    pub const EMPTY: Self = DirtyPiece { piece: Piece::Pawn, color: Color::White, from: SQ_NONE, to: SQ_NONE };
}

#[derive(Clone)]
pub struct Accumulator {
    pub white: [f32; L1_SIZE],
    pub black: [f32; L1_SIZE],
    pub psqt_white: [f32; NUM_PSQT_BUCKETS],
    pub psqt_black: [f32; NUM_PSQT_BUCKETS],
}

impl Accumulator {
    pub fn new() -> Self {
        Accumulator {
            white: [0.0; L1_SIZE],
            black: [0.0; L1_SIZE],
            psqt_white: [0.0; NUM_PSQT_BUCKETS],
            psqt_black: [0.0; NUM_PSQT_BUCKETS],
        }
    }
}

#[derive(Clone)]
pub struct AccumulatorQ {
    pub white: [i16; L1_SIZE],
    pub black: [i16; L1_SIZE],
    pub psqt_white: [i32; NUM_PSQT_BUCKETS],
    pub psqt_black: [i32; NUM_PSQT_BUCKETS],
}

impl AccumulatorQ {
    pub fn new() -> Self {
        AccumulatorQ {
            white: [0; L1_SIZE],
            black: [0; L1_SIZE],
            psqt_white: [0; NUM_PSQT_BUCKETS],
            psqt_black: [0; NUM_PSQT_BUCKETS],
        }
    }
}

const MAX_PLY: usize = 128;

pub struct AccStackQ {
    entries: Box<[AccumulatorQ; MAX_PLY]>,
    sp: usize,
}

impl Clone for AccStackQ {
    fn clone(&self) -> Self {
        AccStackQ {
            entries: self.entries.clone(),
            sp: self.sp,
        }
    }
}

impl AccStackQ {
    pub fn new() -> Self {
        AccStackQ {
            entries: Box::new(std::array::from_fn(|_| AccumulatorQ::new())),
            sp: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, acc: &AccumulatorQ) {
        debug_assert!(self.sp < MAX_PLY);
        self.entries[self.sp].white = acc.white;
        self.entries[self.sp].black = acc.black;
        self.entries[self.sp].psqt_white = acc.psqt_white;
        self.entries[self.sp].psqt_black = acc.psqt_black;
        self.sp += 1;
    }

    #[inline]
    pub fn pop(&mut self) -> &AccumulatorQ {
        debug_assert!(self.sp > 0);
        self.sp -= 1;
        &self.entries[self.sp]
    }

    #[inline]
    pub fn clear(&mut self) {
        self.sp = 0;
    }
}

pub fn refresh_accumulator(board: &Board, acc: &mut Accumulator) {
    let w = weights();
    acc.white = w.l1_biases;
    acc.black = w.l1_biases;
    acc.psqt_white = [0.0; NUM_PSQT_BUCKETS];
    acc.psqt_black = [0.0; NUM_PSQT_BUCKETS];
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
                    for b in 0..NUM_PSQT_BUCKETS {
                        acc.psqt_white[b] += w.psqt_weights[wi][b];
                        acc.psqt_black[b] += w.psqt_weights[bi][b];
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
                    for b in 0..NUM_PSQT_BUCKETS {
                        acc.psqt_white[b] += w.psqt_weights[wi][b];
                        acc.psqt_black[b] += w.psqt_weights[bi][b];
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
        for b in 0..NUM_PSQT_BUCKETS { acc.psqt_white[b] += w.psqt_weights[wi][b]; acc.psqt_black[b] += w.psqt_weights[bi][b]; }
    } else {
        if piece == Piece::King { return; }
        let wi = feature_index_halfkp_white(piece, color, sq, white_king);
        let bi = feature_index_halfkp_black(piece, color, sq, black_king);
        for j in 0..L1_SIZE { acc.white[j] += w.l1_weights[wi][j]; acc.black[j] += w.l1_weights[bi][j]; }
        for b in 0..NUM_PSQT_BUCKETS { acc.psqt_white[b] += w.psqt_weights[wi][b]; acc.psqt_black[b] += w.psqt_weights[bi][b]; }
    }
}

#[inline]
pub fn accumulator_remove(acc: &mut Accumulator, piece: Piece, color: Color, sq: u8, white_king: u8, black_king: u8) {
    let w = weights();
    if w.version == 1 {
        let wi = feature_index_white(piece, color, sq);
        let bi = feature_index_black(piece, color, sq);
        for j in 0..L1_SIZE { acc.white[j] -= w.l1_weights[wi][j]; acc.black[j] -= w.l1_weights[bi][j]; }
        for b in 0..NUM_PSQT_BUCKETS { acc.psqt_white[b] -= w.psqt_weights[wi][b]; acc.psqt_black[b] -= w.psqt_weights[bi][b]; }
    } else {
        if piece == Piece::King { return; }
        let wi = feature_index_halfkp_white(piece, color, sq, white_king);
        let bi = feature_index_halfkp_black(piece, color, sq, black_king);
        for j in 0..L1_SIZE { acc.white[j] -= w.l1_weights[wi][j]; acc.black[j] -= w.l1_weights[bi][j]; }
        for b in 0..NUM_PSQT_BUCKETS { acc.psqt_white[b] -= w.psqt_weights[wi][b]; acc.psqt_black[b] -= w.psqt_weights[bi][b]; }
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
        for b in 0..NUM_PSQT_BUCKETS { acc.psqt_white[b] += w.psqt_weights[wi_to][b] - w.psqt_weights[wi_from][b]; acc.psqt_black[b] += w.psqt_weights[bi_to][b] - w.psqt_weights[bi_from][b]; }
    } else {
        if piece == Piece::King { return; }
        let wi_from = feature_index_halfkp_white(piece, color, from, white_king);
        let wi_to   = feature_index_halfkp_white(piece, color, to, white_king);
        let bi_from = feature_index_halfkp_black(piece, color, from, black_king);
        let bi_to   = feature_index_halfkp_black(piece, color, to, black_king);
        for j in 0..L1_SIZE { acc.white[j] += w.l1_weights[wi_to][j] - w.l1_weights[wi_from][j]; acc.black[j] += w.l1_weights[bi_to][j] - w.l1_weights[bi_from][j]; }
        for b in 0..NUM_PSQT_BUCKETS { acc.psqt_white[b] += w.psqt_weights[wi_to][b] - w.psqt_weights[wi_from][b]; acc.psqt_black[b] += w.psqt_weights[bi_to][b] - w.psqt_weights[bi_from][b]; }
    }
}

pub fn refresh_accumulator_q(board: &Board, acc: &mut AccumulatorQ) {
    let wq = weights_q();
    acc.white = wq.ft_biases;
    acc.black = wq.ft_biases;
    acc.psqt_white = [0; NUM_PSQT_BUCKETS];
    acc.psqt_black = [0; NUM_PSQT_BUCKETS];
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
                    simd::vec_add_i16(&mut acc.white, &wq.ft_weights[wi]);
                    simd::vec_add_i16(&mut acc.black, &wq.ft_weights[bi]);
                    for b in 0..NUM_PSQT_BUCKETS { acc.psqt_white[b] += wq.psqt_weights[wi][b]; acc.psqt_black[b] += wq.psqt_weights[bi][b]; }
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
                    simd::vec_add_i16(&mut acc.white, &wq.ft_weights[wi]);
                    simd::vec_add_i16(&mut acc.black, &wq.ft_weights[bi]);
                    for b in 0..NUM_PSQT_BUCKETS { acc.psqt_white[b] += wq.psqt_weights[wi][b]; acc.psqt_black[b] += wq.psqt_weights[bi][b]; }
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
        simd::vec_add_i16(&mut acc.white, &wq.ft_weights[wi]);
        simd::vec_add_i16(&mut acc.black, &wq.ft_weights[bi]);
        for b in 0..NUM_PSQT_BUCKETS { acc.psqt_white[b] += wq.psqt_weights[wi][b]; acc.psqt_black[b] += wq.psqt_weights[bi][b]; }
    } else {
        if piece == Piece::King { return; }
        let wi = feature_index_halfkp_white(piece, color, sq, white_king);
        let bi = feature_index_halfkp_black(piece, color, sq, black_king);
        simd::vec_add_i16(&mut acc.white, &wq.ft_weights[wi]);
        simd::vec_add_i16(&mut acc.black, &wq.ft_weights[bi]);
        for b in 0..NUM_PSQT_BUCKETS { acc.psqt_white[b] += wq.psqt_weights[wi][b]; acc.psqt_black[b] += wq.psqt_weights[bi][b]; }
    }
}

#[inline]
pub fn accumulator_remove_q(acc: &mut AccumulatorQ, piece: Piece, color: Color, sq: u8, white_king: u8, black_king: u8) {
    let wq = weights_q();
    if wq.version == 1 {
        let wi = feature_index_white(piece, color, sq);
        let bi = feature_index_black(piece, color, sq);
        simd::vec_sub_i16(&mut acc.white, &wq.ft_weights[wi]);
        simd::vec_sub_i16(&mut acc.black, &wq.ft_weights[bi]);
        for b in 0..NUM_PSQT_BUCKETS { acc.psqt_white[b] -= wq.psqt_weights[wi][b]; acc.psqt_black[b] -= wq.psqt_weights[bi][b]; }
    } else {
        if piece == Piece::King { return; }
        let wi = feature_index_halfkp_white(piece, color, sq, white_king);
        let bi = feature_index_halfkp_black(piece, color, sq, black_king);
        simd::vec_sub_i16(&mut acc.white, &wq.ft_weights[wi]);
        simd::vec_sub_i16(&mut acc.black, &wq.ft_weights[bi]);
        for b in 0..NUM_PSQT_BUCKETS { acc.psqt_white[b] -= wq.psqt_weights[wi][b]; acc.psqt_black[b] -= wq.psqt_weights[bi][b]; }
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
        simd::vec_add_sub_i16(&mut acc.white, &wq.ft_weights[wi_to], &wq.ft_weights[wi_from]);
        simd::vec_add_sub_i16(&mut acc.black, &wq.ft_weights[bi_to], &wq.ft_weights[bi_from]);
        for b in 0..NUM_PSQT_BUCKETS { acc.psqt_white[b] += wq.psqt_weights[wi_to][b] - wq.psqt_weights[wi_from][b]; acc.psqt_black[b] += wq.psqt_weights[bi_to][b] - wq.psqt_weights[bi_from][b]; }
    } else {
        if piece == Piece::King { return; }
        let wi_from = feature_index_halfkp_white(piece, color, from, white_king);
        let wi_to   = feature_index_halfkp_white(piece, color, to, white_king);
        let bi_from = feature_index_halfkp_black(piece, color, from, black_king);
        let bi_to   = feature_index_halfkp_black(piece, color, to, black_king);
        simd::vec_add_sub_i16(&mut acc.white, &wq.ft_weights[wi_to], &wq.ft_weights[wi_from]);
        simd::vec_add_sub_i16(&mut acc.black, &wq.ft_weights[bi_to], &wq.ft_weights[bi_from]);
        for b in 0..NUM_PSQT_BUCKETS { acc.psqt_white[b] += wq.psqt_weights[wi_to][b] - wq.psqt_weights[wi_from][b]; acc.psqt_black[b] += wq.psqt_weights[bi_to][b] - wq.psqt_weights[bi_from][b]; }
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
        assert!(acc.psqt_white.iter().all(|&v| v == 0.0));
        assert!(acc.psqt_black.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_accumulator_q_new() {
        let acc = AccumulatorQ::new();
        assert!(acc.white.iter().all(|&v| v == 0));
        assert!(acc.black.iter().all(|&v| v == 0));
        assert!(acc.psqt_white.iter().all(|&v| v == 0));
        assert!(acc.psqt_black.iter().all(|&v| v == 0));
    }
}
