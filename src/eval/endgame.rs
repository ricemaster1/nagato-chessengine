use crate::bitboard::*;
use crate::board::Board;

/// Detect specific endgames and return their heuristic score if applicable.
/// Score is returned relative to White (positive means White is winning).
pub fn evaluate_endgame(board: &Board) -> Option<i32> {
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

    // Detect KBNvK
    if w_minor_count == 2 && w_bishops == 1 && w_knights == 1 && b_minor_count == 0 {
        return Some(eval_kbnk(board, Color::White));
    } else if b_minor_count == 2 && b_bishops == 1 && b_knights == 1 && w_minor_count == 0 {
        return Some(-eval_kbnk(board, Color::Black));
    }

    None
}

/// Evaluates Bishop + Knight vs King.
/// Uses mathematical scaling (Delétang's Triangle method) to drive the king
/// algebraically to the correct mating corner avoiding the "wrong corner" heuristic trap.
fn eval_kbnk(board: &Board, winning_side: Color) -> i32 {
    let losing_side = winning_side.flip();

    let winner_king = board.king_sq(winning_side);
    let loser_king = board.king_sq(losing_side);
    let bishop_bb = board.pieces[winning_side.index()][Piece::Bishop.index()];
    let bishop_sq = bishop_bb.trailing_zeros() as u8;

    // Base massive score for guaranteed endgame win (assuming checkmate is forced)
    let base_score = 5000;

    // Is the bishop on a light or dark square?
    // Formula: (file + rank) is even -> dark square
    let file = bishop_sq & 7;
    let rank = bishop_sq >> 3;
    let is_light_squared = (file + rank) % 2 != 0;

    let loser_file = loser_king & 7;
    let loser_rank = loser_king >> 3;

    // Calculate Chebyshev distance (max(dx, dy)) to the closest TRUE mating corner
    // For light-squared bishop, true mating corners are a8 (file 0, rank 7) and h1 (file 7, rank 0).
    // For dark-squared bishop, true mating corners are a1 (file 0, rank 0) and h8 (file 7, rank 7).
    let dist_to_corner1: u8;
    let dist_to_corner2: u8;

    if is_light_squared {
        // a8
        dist_to_corner1 = loser_file.max(7_u8.saturating_sub(loser_rank));
        // h1
        dist_to_corner2 = (7_u8.saturating_sub(loser_file)).max(loser_rank);
    } else {
        // a1
        dist_to_corner1 = loser_file.max(loser_rank);
        // h8
        dist_to_corner2 = (7_u8.saturating_sub(loser_file)).max(7_u8.saturating_sub(loser_rank));
    }

    let min_dist_to_corner = dist_to_corner1.min(dist_to_corner2) as i32;

    // Exponential scaling to actively shrink the Delétang triangles.
    // The shorter the distance to the correct corner, the exponentially higher the reward.
    let push_score = (7 - min_dist_to_corner) * (7 - min_dist_to_corner) * 20;

    // Force our King to be close to their King.
    let winner_file = winner_king & 7;
    let winner_rank = winner_king >> 3;
    let k_dist_file = (winner_file as i32 - loser_file as i32).abs();
    let k_dist_rank = (winner_rank as i32 - loser_rank as i32).abs();
    let king_distance = k_dist_file.max(k_dist_rank);

    let king_proximity_score = (7 - king_distance) * 10;

    base_score + push_score + king_proximity_score
}

