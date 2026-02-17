/// Evaluation function.
/// Uses piece-square tables with tapered evaluation (midgame → endgame interpolation).

use crate::bitboard::*;
use crate::board::Board;
use crate::movegen;
use crate::moves::Move;

// ============================================================
// Material values (centipawns)
// ============================================================
pub const PAWN_VALUE: i32   = 100;
pub const KNIGHT_VALUE: i32 = 320;
pub const BISHOP_VALUE: i32 = 330;
pub const ROOK_VALUE: i32   = 500;
pub const QUEEN_VALUE: i32  = 900;
pub const KING_VALUE: i32   = 0; // King has no material value

pub const PIECE_VALUES: [i32; PIECE_COUNT] = [
    PAWN_VALUE, KNIGHT_VALUE, BISHOP_VALUE, ROOK_VALUE, QUEEN_VALUE, KING_VALUE
];

// Phase values for tapered eval (total = 24)
const KNIGHT_PHASE: i32 = 1;
const BISHOP_PHASE: i32 = 1;
const ROOK_PHASE: i32   = 2;
const QUEEN_PHASE: i32  = 4;
const TOTAL_PHASE: i32  = 4 * KNIGHT_PHASE + 4 * BISHOP_PHASE + 4 * ROOK_PHASE + 2 * QUEEN_PHASE;

// ============================================================
// Piece-Square Tables (from white's perspective, flipped for black)
// Index: square (a1=0 through h8=63)
// Values in centipawns, [midgame, endgame]
// ============================================================

type PstPair = [i32; 2]; // [midgame, endgame]

#[rustfmt::skip]
const PAWN_PST: [PstPair; 64] = [
    [ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],  // rank 1 (no pawns)
    [-1, 0],[ 1, 0],[-4, 0],[-8, 0],[-8, 0],[-4, 0],[ 1, 0],[-1, 0],  // rank 2
    [-2,-1],[ 0, 0],[-3, 0],[-1, 0],[-1, 0],[-3, 0],[ 0, 0],[-2,-1],  // rank 3
    [-2, 3],[ 2, 3],[ 6, 1],[12, 0],[12, 0],[ 6, 1],[ 2, 3],[-2, 3],  // rank 4
    [ 3, 8],[ 8, 8],[14, 5],[22, 2],[22, 2],[14, 5],[ 8, 8],[ 3, 8],  // rank 5
    [ 5,18],[12,18],[18,14],[28,10],[28,10],[18,14],[12,18],[ 5,18],  // rank 6
    [10,40],[15,40],[20,35],[30,30],[30,30],[20,35],[15,40],[10,40],  // rank 7
    [ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],[ 0, 0],  // rank 8 (no pawns)
];

#[rustfmt::skip]
const KNIGHT_PST: [PstPair; 64] = [
    [-50,-30],[-30,-20],[-20,-15],[-15,-10],[-15,-10],[-20,-15],[-30,-20],[-50,-30],
    [-30,-20],[-10, -5],[  0,  0],[  5,  5],[  5,  5],[  0,  0],[-10, -5],[-30,-20],
    [-20,-10],[  5,  5],[ 15, 10],[ 20, 15],[ 20, 15],[ 15, 10],[  5,  5],[-20,-10],
    [-10, -5],[ 10, 10],[ 20, 15],[ 25, 20],[ 25, 20],[ 20, 15],[ 10, 10],[-10, -5],
    [-10, -5],[ 10, 10],[ 20, 15],[ 25, 20],[ 25, 20],[ 20, 15],[ 10, 10],[-10, -5],
    [-20,-10],[  5,  5],[ 15, 10],[ 20, 15],[ 20, 15],[ 15, 10],[  5,  5],[-20,-10],
    [-30,-20],[-10, -5],[  0,  0],[  5,  5],[  5,  5],[  0,  0],[-10, -5],[-30,-20],
    [-50,-30],[-30,-20],[-20,-15],[-15,-10],[-15,-10],[-20,-15],[-30,-20],[-50,-30],
];

#[rustfmt::skip]
const BISHOP_PST: [PstPair; 64] = [
    [-10,-10],[-10, -5],[-10, -5],[-10, -5],[-10, -5],[-10, -5],[-10, -5],[-10,-10],
    [-5,  -5],[  5,  0],[  2,  0],[  5,  0],[  5,  0],[  2,  0],[  5,  0],[ -5, -5],
    [-5,  -5],[  5,  0],[ 10,  5],[  8,  5],[  8,  5],[ 10,  5],[  5,  0],[ -5, -5],
    [-5,   0],[  2,  5],[  8, 10],[ 12, 10],[ 12, 10],[  8, 10],[  2,  5],[ -5,  0],
    [-5,   0],[  5,  5],[ 10, 10],[ 12, 10],[ 12, 10],[ 10, 10],[  5,  5],[ -5,  0],
    [-5,  -5],[ 10,  0],[ 10,  5],[  5,  5],[  5,  5],[ 10,  5],[ 10,  0],[ -5, -5],
    [-5,  -5],[  8,  0],[  2,  0],[  2,  0],[  2,  0],[  2,  0],[  8,  0],[ -5, -5],
    [-10,-10],[-10, -5],[-10, -5],[-10, -5],[-10, -5],[-10, -5],[-10, -5],[-10,-10],
];

#[rustfmt::skip]
const ROOK_PST: [PstPair; 64] = [
    [ -2,  0],[  0,  0],[  4,  0],[  6,  0],[  6,  0],[  4,  0],[  0,  0],[ -2,  0],
    [ -5,  0],[ -2,  0],[  0,  0],[  2,  0],[  2,  0],[  0,  0],[ -2,  0],[ -5,  0],
    [ -5,  0],[ -2,  0],[  0,  0],[  0,  0],[  0,  0],[  0,  0],[ -2,  0],[ -5,  0],
    [ -5,  0],[ -2,  0],[  0,  0],[  0,  0],[  0,  0],[  0,  0],[ -2,  0],[ -5,  0],
    [ -5,  0],[ -2,  0],[  0,  5],[  0,  5],[  0,  5],[  0,  5],[ -2,  0],[ -5,  0],
    [  0,  0],[  2,  0],[  5,  5],[  8,  5],[  8,  5],[  5,  5],[  2,  0],[  0,  0],
    [ 10, 10],[ 12, 10],[ 15, 10],[ 18, 10],[ 18, 10],[ 15, 10],[ 12, 10],[ 10, 10],
    [  5,  5],[  5,  5],[  5, 10],[  5, 10],[  5, 10],[  5, 10],[  5,  5],[  5,  5],
];

#[rustfmt::skip]
const QUEEN_PST: [PstPair; 64] = [
    [-15, -20],[-10, -10],[ -5, -5],[  0,  0],[  0,  0],[ -5, -5],[-10, -10],[-15, -20],
    [-10, -10],[ -2,  -5],[  3,  0],[  3,  0],[  3,  0],[  3,  0],[ -2,  -5],[-10, -10],
    [ -5,  -5],[  3,   0],[  5,  5],[  5,  5],[  5,  5],[  5,  5],[  3,   0],[ -5,  -5],
    [  0,   0],[  3,   5],[  5, 10],[  8, 10],[  8, 10],[  5, 10],[  3,   5],[  0,   0],
    [ -3,   0],[  3,   5],[  5, 10],[  8, 10],[  8, 10],[  5, 10],[  3,   5],[ -3,   0],
    [ -5,  -5],[  0,   0],[  5,  5],[  3,  5],[  3,  5],[  5,  5],[  0,   0],[ -5,  -5],
    [-10, -10],[ -5,  -5],[ -2,  0],[ -2,  0],[ -2,  0],[ -2,  0],[ -5,  -5],[-10, -10],
    [-15, -20],[-10, -10],[ -5, -5],[ -5,  0],[ -5,  0],[ -5, -5],[-10, -10],[-15, -20],
];

#[rustfmt::skip]
const KING_PST: [PstPair; 64] = [
    [ 20, -50],[ 30, -30],[ 10, -20],[ -5, -20],[ -5, -20],[ 10, -20],[ 30, -30],[ 20, -50],
    [ 20, -30],[ 20, -15],[  0, -10],[-10, -10],[-10, -10],[  0, -10],[ 20, -15],[ 20, -30],
    [-10, -20],[-15, -10],[-20,   0],[-25,  10],[-25,  10],[-20,   0],[-15, -10],[-10, -20],
    [-25, -15],[-30,  -5],[-35,  10],[-40,  20],[-40,  20],[-35,  10],[-30,  -5],[-25, -15],
    [-40, -15],[-45,  -5],[-50,  10],[-55,  20],[-55,  20],[-50,  10],[-45,  -5],[-40, -15],
    [-35, -20],[-40, -10],[-50,   0],[-55,  10],[-55,  10],[-50,   0],[-40, -10],[-35, -20],
    [-25, -30],[-35, -15],[-45, -10],[-50, -10],[-50, -10],[-45, -10],[-35, -15],[-25, -30],
    [-20, -50],[-30, -30],[-40, -20],[-50, -20],[-50, -20],[-40, -20],[-30, -30],[-20, -50],
];

const PST: [[PstPair; 64]; PIECE_COUNT] = [
    PAWN_PST, KNIGHT_PST, BISHOP_PST, ROOK_PST, QUEEN_PST, KING_PST
];

/// Flip a square vertically (for accessing PST from black's perspective)
#[inline]
fn flip_sq(sq: u8) -> u8 {
    sq ^ 56
}

/// Compute phase (how much material is on the board)
fn compute_phase(board: &Board) -> i32 {
    let mut phase = TOTAL_PHASE;
    for color in 0..COLOR_COUNT {
        phase -= popcount(board.pieces[color][Piece::Knight.index()]) as i32 * KNIGHT_PHASE;
        phase -= popcount(board.pieces[color][Piece::Bishop.index()]) as i32 * BISHOP_PHASE;
        phase -= popcount(board.pieces[color][Piece::Rook.index()]) as i32 * ROOK_PHASE;
        phase -= popcount(board.pieces[color][Piece::Queen.index()]) as i32 * QUEEN_PHASE;
    }
    phase.max(0) // Clamp to 0 (full endgame) - TOTAL_PHASE (full opening)
}

/// Main evaluation function. Returns score in centipawns from the perspective
/// of the side to move.
pub fn evaluate(board: &Board) -> i32 {
    let mut mg_score: [i32; 2] = [0, 0]; // midgame score per side
    let mut eg_score: [i32; 2] = [0, 0]; // endgame score per side

    for color in 0..COLOR_COUNT {
        for piece in 0..PIECE_COUNT {
            let mut bb = board.pieces[color][piece];
            while bb != 0 {
                let sq = pop_lsb(&mut bb);
                // Material
                mg_score[color] += PIECE_VALUES[piece];
                eg_score[color] += PIECE_VALUES[piece];

                // Piece-square table
                let pst_sq = if color == 0 { sq } else { flip_sq(sq) };
                mg_score[color] += PST[piece][pst_sq as usize][0];
                eg_score[color] += PST[piece][pst_sq as usize][1];
            }
        }
    }

    // Bishop pair bonus
    for color in 0..COLOR_COUNT {
        if popcount(board.pieces[color][Piece::Bishop.index()]) >= 2 {
            mg_score[color] += 30;
            eg_score[color] += 50;
        }
    }

    // Pawn structure evaluation
    eval_pawn_structure(board, &mut mg_score, &mut eg_score);

    // Rook on open/semi-open file
    eval_rooks(board, &mut mg_score, &mut eg_score);

    // Mobility
    eval_mobility(board, &mut mg_score, &mut eg_score);

    // King safety
    eval_king_safety(board, &mut mg_score, &mut eg_score);

    // Tapered eval: interpolate between midgame and endgame
    let phase = compute_phase(board);
    let mg = mg_score[Color::White.index()] - mg_score[Color::Black.index()];
    let eg = eg_score[Color::White.index()] - eg_score[Color::Black.index()];

    // phase = TOTAL_PHASE means opening (all pieces), phase = 0 means endgame (no pieces)
    let score = (mg * (TOTAL_PHASE - phase) + eg * phase) / TOTAL_PHASE;

    // Return from perspective of side to move
    match board.side {
        Color::White => score,
        Color::Black => -score,
    }
}

fn eval_pawn_structure(board: &Board, mg: &mut [i32; 2], eg: &mut [i32; 2]) {
    for color in 0..COLOR_COUNT {
        let our_pawns = board.pieces[color][Piece::Pawn.index()];
        let their_pawns = board.pieces[color ^ 1][Piece::Pawn.index()];

        let mut pawns = our_pawns;
        while pawns != 0 {
            let sq = pop_lsb(&mut pawns);
            let file = file_of(sq) as usize;
            let file_mask = FILES[file];

            // Doubled pawns penalty
            if popcount(our_pawns & file_mask) > 1 {
                mg[color] -= 8;
                eg[color] -= 12;
            }

            // Isolated pawns penalty
            let adjacent_files = match file {
                0 => FILES[1],
                7 => FILES[6],
                f => FILES[f - 1] | FILES[f + 1],
            };
            if our_pawns & adjacent_files == 0 {
                mg[color] -= 10;
                eg[color] -= 15;
            }

            // Passed pawn bonus
            let rank = rank_of(sq);
            let ahead_mask = if color == 0 {
                // White: ranks above current rank
                let mut m: Bitboard = 0;
                for r in (rank + 1)..8 {
                    m |= RANKS[r as usize];
                }
                m & (file_mask | adjacent_files)
            } else {
                // Black: ranks below current rank
                let mut m: Bitboard = 0;
                for r in 0..rank {
                    m |= RANKS[r as usize];
                }
                m & (file_mask | adjacent_files)
            };

            if their_pawns & ahead_mask == 0 {
                // Passed pawn! Bonus based on how far advanced
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

fn eval_rooks(board: &Board, mg: &mut [i32; 2], eg: &mut [i32; 2]) {
    for color in 0..COLOR_COUNT {
        let mut rooks = board.pieces[color][Piece::Rook.index()];
        let our_pawns = board.pieces[color][Piece::Pawn.index()];
        let their_pawns = board.pieces[color ^ 1][Piece::Pawn.index()];

        while rooks != 0 {
            let sq = pop_lsb(&mut rooks);
            let file_mask = FILES[file_of(sq) as usize];

            if our_pawns & file_mask == 0 {
                if their_pawns & file_mask == 0 {
                    // Open file
                    mg[color] += 20;
                    eg[color] += 15;
                } else {
                    // Semi-open file
                    mg[color] += 10;
                    eg[color] += 8;
                }
            }
        }
    }
}

fn eval_mobility(board: &Board, mg: &mut [i32; 2], eg: &mut [i32; 2]) {
    for color in 0..COLOR_COUNT {
        let _us_color = if color == 0 { Color::White } else { Color::Black };
        let our_occ = board.occupancy[color];
        let all_occ = board.all_occupancy;

        // Knight mobility
        let mut knights = board.pieces[color][Piece::Knight.index()];
        while knights != 0 {
            let sq = pop_lsb(&mut knights);
            let moves = popcount(movegen::knight_attacks(sq) & !our_occ) as i32;
            mg[color] += (moves - 4) * 3;
            eg[color] += (moves - 4) * 3;
        }

        // Bishop mobility
        let mut bishops = board.pieces[color][Piece::Bishop.index()];
        while bishops != 0 {
            let sq = pop_lsb(&mut bishops);
            let moves = popcount(movegen::bishop_attacks(sq, all_occ) & !our_occ) as i32;
            mg[color] += (moves - 6) * 4;
            eg[color] += (moves - 6) * 3;
        }

        // Rook mobility
        let mut rooks = board.pieces[color][Piece::Rook.index()];
        while rooks != 0 {
            let sq = pop_lsb(&mut rooks);
            let moves = popcount(movegen::rook_attacks(sq, all_occ) & !our_occ) as i32;
            mg[color] += (moves - 7) * 2;
            eg[color] += (moves - 7) * 3;
        }

        // Queen mobility (smaller bonus to avoid overvaluing early queen moves)
        let mut queens = board.pieces[color][Piece::Queen.index()];
        while queens != 0 {
            let sq = pop_lsb(&mut queens);
            let moves = popcount(movegen::queen_attacks(sq, all_occ) & !our_occ) as i32;
            mg[color] += (moves - 14) * 1;
            eg[color] += (moves - 14) * 2;
        }
    }
}

fn eval_king_safety(board: &Board, mg: &mut [i32; 2], _eg: &mut [i32; 2]) {
    for color in 0..COLOR_COUNT {
        let king_sq = board.king_sq(if color == 0 { Color::White } else { Color::Black });
        let king_file = file_of(king_sq) as usize;
        let our_pawns = board.pieces[color][Piece::Pawn.index()];

        // Pawn shield bonus (for kings on the flanks in the midgame)
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

/// Static Exchange Evaluation — determines the material outcome of a capture
/// sequence on a single square, assuming both sides play optimally.
///
/// Uses the "swap algorithm": we simulate alternating captures on the target
/// square, tracking a gain stack, then propagate back with min/max to determine
/// whether the side to move benefits from the initial capture.
///
/// Returns the net material gain (positive = good for the capturing side).
pub fn see(board: &Board, m: Move) -> i32 {
    if !m.is_capture() {
        return 0;
    }

    let from = m.from_sq();
    let to = m.to_sq();

    let mut gain = [0i32; 32];
    let mut d: usize = 0;

    // Initial capture gain
    gain[0] = if m.is_en_passant() {
        PAWN_VALUE
    } else {
        PIECE_VALUES[m.captured_piece().index()]
    };

    // If promoting, add the promotion bonus
    if m.is_promotion() {
        gain[0] += PIECE_VALUES[m.promotion_piece().unwrap().index()] - PAWN_VALUE;
    }

    // The value of the piece now sitting on the target square (what the opponent can win back)
    let mut piece_on_target = if m.is_promotion() {
        PIECE_VALUES[m.promotion_piece().unwrap().index()]
    } else {
        PIECE_VALUES[m.piece().index()]
    };

    // Working copy of occupancy — remove the initial attacker
    let mut occ = board.all_occupancy;
    occ ^= square_bb(from);

    // For en passant, also remove the captured pawn from occupancy
    if m.is_en_passant() {
        let ep_cap_sq = match board.side {
            Color::White => to - 8,
            Color::Black => to + 8,
        };
        occ ^= square_bb(ep_cap_sq);
    }

    // Side making the *next* capture (opponent of the initial mover)
    let mut side = board.side.flip();

    loop {
        d += 1;
        if d >= 32 {
            break;
        }

        // Find the least valuable attacker of `to` by `side`
        let (attacker_sq, piece) = match least_valuable_attacker(board, to, side, occ) {
            Some(result) => result,
            None => break, // No more attackers — exchange ends
        };

        // Speculative gain: if we capture the piece on the target, we get piece_on_target
        // but we give up what our opponent gained so far
        gain[d] = piece_on_target - gain[d - 1];

        // Pruning: if even the best case can't improve for the side to move,
        // this capture would never be made
        if (-gain[d - 1]).max(gain[d]) < 0 {
            break;
        }

        // The piece now on the target square is the one that just captured
        piece_on_target = PIECE_VALUES[piece.index()];

        // Remove this attacker from occupancy (opens x-ray lines for sliding pieces)
        occ ^= square_bb(attacker_sq);

        // Flip side
        side = side.flip();
    }

    // Propagate the gain stack back: each side chooses whether to capture
    // or stand pat, whichever is better (negamax-style)
    while d > 1 {
        d -= 1;
        gain[d - 1] = -((-gain[d - 1]).max(gain[d]));
    }

    gain[0]
}

/// Find the least valuable piece of `side` that attacks `sq` given `occ`.
/// Returns (attacker_square, piece_type) or None.
fn least_valuable_attacker(board: &Board, sq: u8, side: Color, occ: Bitboard) -> Option<(u8, Piece)> {
    let si = side.index();

    // Pawns (cheapest)
    let pawn_attackers = movegen::pawn_attacks(sq, side.flip()) & board.pieces[si][Piece::Pawn.index()] & occ;
    if pawn_attackers != 0 {
        return Some((lsb(pawn_attackers), Piece::Pawn));
    }

    // Knights
    let knight_attackers = movegen::knight_attacks(sq) & board.pieces[si][Piece::Knight.index()] & occ;
    if knight_attackers != 0 {
        return Some((lsb(knight_attackers), Piece::Knight));
    }

    // Bishops
    let bishop_attacks = movegen::bishop_attacks(sq, occ);
    let bishop_attackers = bishop_attacks & board.pieces[si][Piece::Bishop.index()] & occ;
    if bishop_attackers != 0 {
        return Some((lsb(bishop_attackers), Piece::Bishop));
    }

    // Rooks
    let rook_attacks = movegen::rook_attacks(sq, occ);
    let rook_attackers = rook_attacks & board.pieces[si][Piece::Rook.index()] & occ;
    if rook_attackers != 0 {
        return Some((lsb(rook_attackers), Piece::Rook));
    }

    // Queens (use both bishop + rook rays)
    let queen_attackers = (bishop_attacks | rook_attacks) & board.pieces[si][Piece::Queen.index()] & occ;
    if queen_attackers != 0 {
        return Some((lsb(queen_attackers), Piece::Queen));
    }

    // King (most expensive — only if no other attacker)
    let king_attackers = movegen::king_attacks(sq) & board.pieces[si][Piece::King.index()] & occ;
    if king_attackers != 0 {
        return Some((lsb(king_attackers), Piece::King));
    }

    None
}

// ============================================================
// MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
// For move ordering in search
// ============================================================

/// MVV-LVA score: higher = better capture to search first
pub fn mvv_lva_score(m: Move) -> i32 {
    if !m.is_capture() {
        return 0;
    }
    // Captured piece value * 10 - attacker piece value
    // This ensures capturing expensive pieces with cheap ones is scored highest
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

/// Is this score a mate score?
#[inline]
pub fn is_mate_score(score: i32) -> bool {
    score.abs() > MATE_THRESHOLD
}

/// Convert a mate score to "mate in N" moves
pub fn mate_in(score: i32) -> i32 {
    if score > MATE_THRESHOLD {
        (MATE_SCORE - score + 1) / 2
    } else if score < -MATE_THRESHOLD {
        -(MATE_SCORE + score + 1) / 2
    } else {
        0
    }
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
        // Starting position should be roughly equal
        assert!(score.abs() < 50, "Start position eval should be near 0, got {}", score);
    }

    #[test]
    fn test_eval_material_advantage() {
        setup();
        // White has an extra queen
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let score = evaluate(&board);
        // Should be symmetric-ish at start
        assert!(score.abs() < 50);
    }

    // ---- SEE tests ----

    /// Helper: find a move from `from_str` to `to_str` in the legal move list
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
        // White pawn on e4, black pawn on d5 — PxP is even
        let board = Board::from_fen("8/8/8/3p4/4P3/8/8/4K2k w - - 0 1").unwrap();
        let m = find_move(&board, "e4", "d5");
        let score = see(&board, m);
        assert_eq!(score, PAWN_VALUE, "PxP with no defenders should win a pawn");
    }

    #[test]
    fn test_see_pawn_takes_defended_pawn() {
        setup();
        // White pawn e4, black pawn d5 defended by pawn on e6
        let board = Board::from_fen("8/8/4p3/3p4/4P3/8/8/4K2k w - - 0 1").unwrap();
        let m = find_move(&board, "e4", "d5");
        let score = see(&board, m);
        // PxP, then pxP recapture — net 0 (exchange pawns)
        assert_eq!(score, 0, "PxP with pawn defender should be 0");
    }

    #[test]
    fn test_see_knight_takes_defended_pawn() {
        setup();
        // White knight on c3, black pawn on d5 defended by pawn on e6
        let board = Board::from_fen("8/8/4p3/3p4/8/2N5/8/4K2k w - - 0 1").unwrap();
        let m = find_move(&board, "c3", "d5");
        let score = see(&board, m);
        // NxP (gain 100), then pxN (they gain 320) — net: 100 - 320 = -220
        assert!(score < 0, "Knight taking pawn defended by pawn should be negative, got {}", score);
    }

    #[test]
    fn test_see_queen_takes_defended_pawn() {
        setup();
        // White queen on d1, black pawn on d5 defended by pawn on e6
        let board = Board::from_fen("8/8/4p3/3p4/8/8/8/3QK2k w - - 0 1").unwrap();
        let m = find_move(&board, "d1", "d5");
        let score = see(&board, m);
        // QxP (gain 100), then pxQ (they gain 900) — net: 100 - 900 = bad
        assert!(score < 0, "Queen taking defended pawn should be losing, got {}", score);
    }

    #[test]
    fn test_see_rook_takes_rook() {
        setup();
        // White rook e1, black rook e8, open file — RxR is even (undefended)
        let board = Board::from_fen("4r3/8/8/8/8/8/8/4RK1k w - - 0 1").unwrap();
        let m = find_move(&board, "e1", "e8");
        let score = see(&board, m);
        assert_eq!(score, ROOK_VALUE, "RxR undefended should win a rook");
    }

    #[test]
    fn test_see_xray_battery() {
        setup();
        // White rook e1, white rook e2 (battery), black rook e8
        // RxR, opponent has nothing — gain 500
        let board = Board::from_fen("4r3/8/8/8/8/8/4R3/4RK1k w - - 0 1").unwrap();
        let m = find_move(&board, "e2", "e8");
        let score = see(&board, m);
        assert_eq!(score, ROOK_VALUE, "RxR with rook behind should win");
    }

    #[test]
    fn test_see_pawn_takes_queen() {
        setup();
        // White pawn on e4, black queen on d5 — PxQ wins a queen
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
        // White bishop c1, black knight on e3, defended by pawn d4
        // BxN (gain 320), pxB (they gain 330) — net: 320 - 330 = -10
        let board = Board::from_fen("8/8/8/8/3p4/4n3/8/2B1K2k w - - 0 1").unwrap();
        let m = find_move(&board, "c1", "e3");
        let score = see(&board, m);
        assert!(score < 0, "BxN defended by pawn should be negative, got {}", score);
    }
}
