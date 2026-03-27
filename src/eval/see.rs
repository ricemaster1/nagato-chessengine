use crate::bitboard::*;
use crate::board::Board;
use crate::movegen;
use crate::moves::Move;
use super::{PAWN_VALUE, PIECE_VALUES};

pub fn see(board: &Board, m: Move) -> i32 {
    if !m.is_capture() {
        return 0;
    }

    let from = m.from_sq();
    let to = m.to_sq();

    let mut gain = [0i32; 32];
    let mut d: usize = 0;

    gain[0] = if m.is_en_passant() {
        PAWN_VALUE
    } else {
        PIECE_VALUES[m.captured_piece().index()]
    };

    if m.is_promotion() {
        gain[0] += PIECE_VALUES[m.promotion_piece().unwrap().index()] - PAWN_VALUE;
    }

    let mut piece_on_target = if m.is_promotion() {
        PIECE_VALUES[m.promotion_piece().unwrap().index()]
    } else {
        PIECE_VALUES[m.piece().index()]
    };

    let mut occ = board.all_occupancy;
    occ ^= square_bb(from);

    if m.is_en_passant() {
        let ep_cap_sq = match board.side {
            Color::White => to - 8,
            Color::Black => to + 8,
        };
        occ ^= square_bb(ep_cap_sq);
    }

    let mut side = board.side.flip();

    loop {
        d += 1;
        if d >= 32 {
            break;
        }

        let (attacker_sq, piece) = match least_valuable_attacker(board, to, side, occ) {
            Some(result) => result,
            None => break,
        };

        gain[d] = piece_on_target - gain[d - 1];

        if (-gain[d - 1]).max(gain[d]) < 0 {
            break;
        }

        piece_on_target = PIECE_VALUES[piece.index()];

        occ ^= square_bb(attacker_sq);

        side = side.flip();
    }

    while d > 1 {
        d -= 1;
        gain[d - 1] = -((-gain[d - 1]).max(gain[d]));
    }

    gain[0]
}

fn least_valuable_attacker(board: &Board, sq: u8, side: Color, occ: Bitboard) -> Option<(u8, Piece)> {
    let si = side.index();

    let pawn_attackers = movegen::pawn_attacks(sq, side.flip()) & board.pieces[si][Piece::Pawn.index()] & occ;
    if pawn_attackers != 0 {
        return Some((lsb(pawn_attackers), Piece::Pawn));
    }

    let knight_attackers = movegen::knight_attacks(sq) & board.pieces[si][Piece::Knight.index()] & occ;
    if knight_attackers != 0 {
        return Some((lsb(knight_attackers), Piece::Knight));
    }

    let bishop_attacks = movegen::bishop_attacks(sq, occ);
    let bishop_attackers = bishop_attacks & board.pieces[si][Piece::Bishop.index()] & occ;
    if bishop_attackers != 0 {
        return Some((lsb(bishop_attackers), Piece::Bishop));
    }

    let rook_attacks = movegen::rook_attacks(sq, occ);
    let rook_attackers = rook_attacks & board.pieces[si][Piece::Rook.index()] & occ;
    if rook_attackers != 0 {
        return Some((lsb(rook_attackers), Piece::Rook));
    }

    let queen_attackers = (bishop_attacks | rook_attacks) & board.pieces[si][Piece::Queen.index()] & occ;
    if queen_attackers != 0 {
        return Some((lsb(queen_attackers), Piece::Queen));
    }

    let king_attackers = movegen::king_attacks(sq) & board.pieces[si][Piece::King.index()] & occ;
    if king_attackers != 0 {
        return Some((lsb(king_attackers), Piece::King));
    }

    None
}

pub fn mvv_lva_score(m: Move) -> i32 {
    if !m.is_capture() {
        return 0;
    }

    let victim_val = if m.is_en_passant() {
        PAWN_VALUE
    } else {
        PIECE_VALUES[m.captured_piece().index()]
    };
    let attacker_val = PIECE_VALUES[m.piece().index()];
    victim_val * 10 - attacker_val
}

pub const INFINITY: i32 = 30000;
pub const MATE_SCORE: i32 = 29000;
pub const MATE_THRESHOLD: i32 = 28000;

#[inline]
pub fn is_mate_score(score: i32) -> bool {
    score.abs() > MATE_THRESHOLD
}

pub fn mate_in(score: i32) -> i32 {
    if score > MATE_THRESHOLD {
        (MATE_SCORE - score + 1) / 2
    } else if score < -MATE_THRESHOLD {
        -(MATE_SCORE + score + 1) / 2
    } else {
        0
    }
}
