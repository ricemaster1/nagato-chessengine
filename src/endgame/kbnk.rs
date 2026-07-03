use crate::bitboard::*;
use crate::board::Board;

pub fn eval_kbnk(board: &Board, winning_side: Color) -> i32 {
    let losing_side = winning_side.flip();

    let winner_king = board.king_sq(winning_side);
    let loser_king = board.king_sq(losing_side);
    let bishop_bb = board.pieces[winning_side.index()][Piece::Bishop.index()];
    let bishop_sq = bishop_bb.trailing_zeros() as u8;

    let base_score = 5000;

    let file = bishop_sq & 7;
    let rank = bishop_sq >> 3;
    let is_light_squared = (file + rank) % 2 != 0;

    let loser_file = loser_king & 7;
    let loser_rank = loser_king >> 3;

    let dist_to_corner1: u8;
    let dist_to_corner2: u8;

    if is_light_squared {
        dist_to_corner1 = loser_file.max(7_u8.saturating_sub(loser_rank));
        dist_to_corner2 = (7_u8.saturating_sub(loser_file)).max(loser_rank);
    } else {
        dist_to_corner1 = loser_file.max(loser_rank);
        dist_to_corner2 = (7_u8.saturating_sub(loser_file)).max(7_u8.saturating_sub(loser_rank));
    }

    let min_dist_to_corner = dist_to_corner1.min(dist_to_corner2) as i32;
    let push_score = (7 - min_dist_to_corner) * (7 - min_dist_to_corner) * 20;

    let winner_file = winner_king & 7;
    let winner_rank = winner_king >> 3;
    let k_dist_file = (winner_file as i32 - loser_file as i32).abs();
    let k_dist_rank = (winner_rank as i32 - loser_rank as i32).abs();
    let king_distance = k_dist_file.max(k_dist_rank);

    let king_proximity_score = (7 - king_distance) * 10;

    base_score + push_score + king_proximity_score
}
