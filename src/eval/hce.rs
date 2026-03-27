use crate::bitboard::*;
use crate::board::Board;
use crate::movegen;
use super::pst::*;
use super::{PIECE_VALUES, TOTAL_PHASE, compute_phase};

#[inline]
fn flip_sq(sq: u8) -> u8 {
    sq ^ 56
}


pub fn evaluate_hce(board: &Board) -> i32 {
    let mut mg_score: [i32; 2] = [0, 0];
    let mut eg_score: [i32; 2] = [0, 0];

    for color in 0..COLOR_COUNT {
        for piece in 0..PIECE_COUNT {
            let mut bb = board.pieces[color][piece];
            while bb != 0 {
                let sq = pop_lsb(&mut bb);
                mg_score[color] += PIECE_VALUES[piece];
                eg_score[color] += PIECE_VALUES[piece];

                let pst_sq = if color == 0 { sq } else { flip_sq(sq) };
                mg_score[color] += PST[piece][pst_sq as usize][0];
                eg_score[color] += PST[piece][pst_sq as usize][1];
            }
        }
    }

    for color in 0..COLOR_COUNT {
        if popcount(board.pieces[color][Piece::Bishop.index()]) >= 2 {
            mg_score[color] += 30;
            eg_score[color] += 50;
        }
    }

    eval_pawn_structure(board, &mut mg_score, &mut eg_score);

    eval_rooks(board, &mut mg_score, &mut eg_score);

    eval_mobility(board, &mut mg_score, &mut eg_score);

    eval_king_safety(board, &mut mg_score, &mut eg_score);

    eval_edge_pawns(board, &mut mg_score, &mut eg_score);

    let phase = compute_phase(board);
    let mg = mg_score[Color::White.index()] - mg_score[Color::Black.index()];
    let eg = eg_score[Color::White.index()] - eg_score[Color::Black.index()];

    let score = (mg * (TOTAL_PHASE - phase) + eg * phase) / TOTAL_PHASE;

    match board.side {
        Color::White => score,
        Color::Black => -score,
    }
}

pub fn eval_pawn_structure(board: &Board, mg: &mut [i32; 2], eg: &mut [i32; 2]) {
    for color in 0..COLOR_COUNT {
        let our_pawns = board.pieces[color][Piece::Pawn.index()];
        let their_pawns = board.pieces[color ^ 1][Piece::Pawn.index()];

        let mut pawns = our_pawns;
        while pawns != 0 {
            let sq = pop_lsb(&mut pawns);
            let file = file_of(sq) as usize;
            let file_mask = FILES[file];

            if popcount(our_pawns & file_mask) > 1 {
                mg[color] -= 8;
                eg[color] -= 12;
            }

            let adjacent_files = match file {
                0 => FILES[1],
                7 => FILES[6],
                f => FILES[f - 1] | FILES[f + 1],
            };
            if our_pawns & adjacent_files == 0 {
                mg[color] -= 10;
                eg[color] -= 15;
            }

            let rank = rank_of(sq);
            let ahead_mask = if color == 0 {
                let mut m: Bitboard = 0;
                for r in (rank + 1)..8 {
                    m |= RANKS[r as usize];
                }
                m & (file_mask | adjacent_files)
            } else {
                let mut m: Bitboard = 0;
                for r in 0..rank {
                    m |= RANKS[r as usize];
                }
                m & (file_mask | adjacent_files)
            };

            if their_pawns & ahead_mask == 0 {
                let advancement = if color == 0 { rank } else { 7 - rank };
                let bonus = match advancement {
                    1 => [5, 10],
                    2 => [5, 12],
                    3 => [10, 20],
                    4 => [20, 40],
                    5 => [40, 70],
                    6 => [60, 120],
                    _ => [0, 0],
                };
                mg[color] += bonus[0];
                eg[color] += bonus[1];
            }
        }
    }
}

pub fn eval_rooks(board: &Board, mg: &mut [i32; 2], eg: &mut [i32; 2]) {
    for color in 0..COLOR_COUNT {
        let mut rooks = board.pieces[color][Piece::Rook.index()];
        let our_pawns = board.pieces[color][Piece::Pawn.index()];
        let their_pawns = board.pieces[color ^ 1][Piece::Pawn.index()];

        while rooks != 0 {
            let sq = pop_lsb(&mut rooks);
            let file_mask = FILES[file_of(sq) as usize];

            if our_pawns & file_mask == 0 {
                if their_pawns & file_mask == 0 {
                    mg[color] += 20;
                    eg[color] += 15;
                } else {
                    mg[color] += 10;
                    eg[color] += 8;
                }
            }
        }
    }
}

pub fn eval_mobility(board: &Board, mg: &mut [i32; 2], eg: &mut [i32; 2]) {
    for color in 0..COLOR_COUNT {
        let _us_color = if color == 0 { Color::White } else { Color::Black };
        let our_occ = board.occupancy[color];
        let all_occ = board.all_occupancy;

        let mut knights = board.pieces[color][Piece::Knight.index()];
        while knights != 0 {
            let sq = pop_lsb(&mut knights);
            let moves = popcount(movegen::knight_attacks(sq) & !our_occ) as i32;
            mg[color] += (moves - 4) * 3;
            eg[color] += (moves - 4) * 3;
        }

        let mut bishops = board.pieces[color][Piece::Bishop.index()];
        while bishops != 0 {
            let sq = pop_lsb(&mut bishops);
            let moves = popcount(movegen::bishop_attacks(sq, all_occ) & !our_occ) as i32;
            mg[color] += (moves - 6) * 4;
            eg[color] += (moves - 6) * 3;
        }

        let mut rooks = board.pieces[color][Piece::Rook.index()];
        while rooks != 0 {
            let sq = pop_lsb(&mut rooks);
            let moves = popcount(movegen::rook_attacks(sq, all_occ) & !our_occ) as i32;
            mg[color] += (moves - 7) * 2;
            eg[color] += (moves - 7) * 3;
        }

        let mut queens = board.pieces[color][Piece::Queen.index()];
        while queens != 0 {
            let sq = pop_lsb(&mut queens);
            let moves = popcount(movegen::queen_attacks(sq, all_occ) & !our_occ) as i32;
            mg[color] += (moves - 14) * 1;
            eg[color] += (moves - 14) * 2;
        }
    }
}

pub fn eval_king_safety(board: &Board, mg: &mut [i32; 2], _eg: &mut [i32; 2]) {
    for color in 0..COLOR_COUNT {
        let king_sq = board.king_sq(if color == 0 { Color::White } else { Color::Black });
        let king_file = file_of(king_sq) as usize;
        let our_pawns = board.pieces[color][Piece::Pawn.index()];

        if king_file <= 2 || king_file >= 5 {
            let shield_files: Vec<usize> = match king_file {
                0 => vec![0, 1, 2],
                1 => vec![0, 1, 2],
                2 => vec![1, 2, 3],
                5 => vec![4, 5, 6],
                6 => vec![5, 6, 7],
                7 => vec![5, 6, 7],
                _ => vec![],
            };

            let shield_rank = if color == 0 { RANK_2 | RANK_3 } else { RANK_6 | RANK_7 };

            for &f in &shield_files {
                if our_pawns & FILES[f] & shield_rank != 0 {
                    mg[color] += 10;
                } else {
                    mg[color] -= 15;
                }
            }
        }
    }
}

pub fn eval_edge_pawns(board: &Board, mg: &mut [i32; 2], eg: &mut [i32; 2]) {
    for color in 0..COLOR_COUNT {
        let our_pawns = board.pieces[color][Piece::Pawn.index()];
        let their_occ = board.occupancy[color ^ 1];
        let us_color = if color == 0 { Color::White } else { Color::Black };

        let mut a_pawns = our_pawns & FILE_A;
        while a_pawns != 0 {
            let sq = pop_lsb(&mut a_pawns);
            let rank = rank_of(sq);
            let advancement = if color == 0 { rank } else { 7 - rank };

            let penalty_mg = 3 + advancement as i32;
            let penalty_eg = 5 + 2 * advancement as i32;
            mg[color] -= penalty_mg;
            eg[color] -= penalty_eg;

            let capture_sq = match us_color {
                Color::White => if rank < 7 { Some(make_square(1, rank + 1)) } else { None },
                Color::Black => if rank > 0 { Some(make_square(1, rank - 1)) } else { None },
            };
            if let Some(csq) = capture_sq {
                if get_bit(their_occ, csq) {
                    mg[color] += 8;
                    eg[color] += 12;
                }
            }
        }

        let mut h_pawns = our_pawns & FILE_H;
        while h_pawns != 0 {
            let sq = pop_lsb(&mut h_pawns);
            let rank = rank_of(sq);
            let advancement = if color == 0 { rank } else { 7 - rank };

            let penalty_mg = 3 + advancement as i32;
            let penalty_eg = 5 + 2 * advancement as i32;
            mg[color] -= penalty_mg;
            eg[color] -= penalty_eg;

            let capture_sq = match us_color {
                Color::White => if rank < 7 { Some(make_square(6, rank + 1)) } else { None },
                Color::Black => if rank > 0 { Some(make_square(6, rank - 1)) } else { None },
            };
            if let Some(csq) = capture_sq {
                if get_bit(their_occ, csq) {
                    mg[color] += 8;
                    eg[color] += 12;
                }
            }
        }

        let b_pawns = our_pawns & FILE_B;
        if b_pawns != 0 && (our_pawns & FILE_A) == 0 {
            let count = popcount(b_pawns) as i32;
            mg[color] += count * 5;
            eg[color] += count * 8;
        }

        let g_pawns = our_pawns & FILE_G;
        if g_pawns != 0 && (our_pawns & FILE_H) == 0 {
            let count = popcount(g_pawns) as i32;
            mg[color] += count * 5;
            eg[color] += count * 8;
        }
    }
}

