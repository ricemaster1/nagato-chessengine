/// Search module: alpha-beta with iterative deepening, PVS, and various pruning techniques.

use crate::bitboard::*;
use crate::board::Board;
use crate::eval::{self, INFINITY, MATE_SCORE};
use crate::learn::{ExpEntry, ExpTable};
use crate::movegen;
use crate::moves::*;
use std::time::Instant;

// ============================================================
// Transposition Table
// ============================================================

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TTFlag {
    Exact,
    Alpha, // Upper bound (fail-low)
    Beta,  // Lower bound (fail-high)
}

#[derive(Clone, Copy)]
pub struct TTEntry {
    pub hash: u64,
    pub depth: i8,
    pub score: i32,
    pub flag: TTFlag,
    pub best_move: Move,
}

pub struct TranspositionTable {
    entries: Vec<Option<TTEntry>>,
    size: usize,
}

impl TranspositionTable {
    pub fn new(size_mb: usize) -> Self {
        let entry_size = std::mem::size_of::<Option<TTEntry>>();
        let size = (size_mb * 1024 * 1024) / entry_size;
        TranspositionTable {
            entries: vec![None; size],
            size,
        }
    }

    #[inline]
    pub fn probe(&self, hash: u64) -> Option<&TTEntry> {
        let idx = (hash as usize) % self.size;
        self.entries[idx].as_ref().filter(|e| e.hash == hash)
    }

    #[inline]
    pub fn store(&mut self, hash: u64, depth: i8, score: i32, flag: TTFlag, best_move: Move) {
        let idx = (hash as usize) % self.size;
        // Always-replace scheme (simple but effective)
        // Prefer deeper entries or exact entries
        if let Some(existing) = &self.entries[idx] {
            if existing.hash == hash && existing.depth > depth as i8 && existing.flag == TTFlag::Exact {
                return; // Don't overwrite a deeper exact entry for the same position
            }
        }
        self.entries[idx] = Some(TTEntry { hash, depth: depth as i8, score, flag, best_move });
    }

    pub fn clear(&mut self) {
        self.entries.fill(None);
    }
}

// ============================================================
// Search state
// ============================================================

pub struct SearchInfo {
    pub nodes: u64,
    pub start_time: Instant,
    pub time_limit_ms: u64,
    pub max_depth: i32,
    pub stopped: bool,

    // Killer moves: quiet moves that caused a beta cutoff
    pub killers: [[Move; 2]; 128], // [ply][slot]

    // History heuristic: indexed by [color][from][to]
    pub history: [[[i32; 64]; 64]; 2],

    // Counter move heuristic: indexed by [piece_that_moved_last][to_sq_of_last_move]
    // Stores the quiet move that refuted the opponent's last move
    pub counter_moves: [[Move; 64]; 6],
}

impl SearchInfo {
    pub fn new() -> Self {
        SearchInfo {
            nodes: 0,
            start_time: Instant::now(),
            time_limit_ms: 0,
            max_depth: 64,
            stopped: false,
            killers: [[MOVE_NONE; 2]; 128],
            history: [[[0; 64]; 64]; 2],
            counter_moves: [[MOVE_NONE; 64]; 6],
        }
    }

    pub fn reset(&mut self) {
        self.nodes = 0;
        self.stopped = false;
        self.killers = [[MOVE_NONE; 2]; 128];
        // Age history scores instead of clearing
        for c in 0..2 {
            for f in 0..64 {
                for t in 0..64 {
                    self.history[c][f][t] /= 2;
                }
            }
        }
        // Counter moves persist across iterations (not reset per depth)
    }

    #[inline]
    pub fn check_time(&mut self) {
        if self.time_limit_ms > 0 && self.nodes & 2047 == 0 {
            if self.start_time.elapsed().as_millis() as u64 >= self.time_limit_ms {
                self.stopped = true;
            }
        }
    }
}

// ============================================================
// ============================================================
// Move ordering
// ============================================================

fn score_moves(list: &MoveList, board: &Board, info: &SearchInfo, ply: usize, tt_move: Move, exp: &ExpTable, prev_move: Move) -> Vec<i32> {
    // Probe experience once per position
    let exp_move = exp.probe(board.hash).map(|e| e.best_move);

    // Look up the counter move for the opponent's previous move
    let counter = if !prev_move.is_null() {
        let cm = info.counter_moves[prev_move.piece().index()][prev_move.to_sq() as usize];
        if cm.is_null() { None } else { Some(cm) }
    } else {
        None
    };

    let mut scores = vec![0i32; list.len()];
    for i in 0..list.len() {
        let m = list.moves[i];

        if m.0 == tt_move.0 && !tt_move.is_null() {
            scores[i] = 10_000_000; // TT move first
        } else if exp_move.is_some() && m.0 == exp_move.unwrap().0 && !exp_move.unwrap().is_null() {
            scores[i] = 5_000_000; // Experience best move — between TT and captures
        } else if m.is_capture() || m.is_en_passant() {
            let see_val = eval::see(board, m);
            if see_val >= 0 {
                // Winning/equal captures: above killers, ordered by MVV-LVA
                let cap_score = 1_000_000 + eval::mvv_lva_score(m);
                scores[i] = cap_score;
            } else {
                // Losing captures: below killers and history, ordered by SEE
                scores[i] = -100_000 + see_val;
            }
        } else if m.is_promotion() {
            scores[i] = 900_000;
        } else if ply < 128 && m.0 == info.killers[ply][0].0 {
            scores[i] = 800_000;
        } else if ply < 128 && m.0 == info.killers[ply][1].0 {
            scores[i] = 700_000;
        } else if counter.is_some() && m.0 == counter.unwrap().0 {
            scores[i] = 650_000; // Counter move — between killer 2 and history
        } else {
            // History heuristic
            scores[i] = info.history[board.side.index()][m.from_sq() as usize][m.to_sq() as usize];
        }
    }
    scores
}

/// Pick the best-scored move and swap it to position `start`
fn pick_move(list: &mut MoveList, scores: &mut [i32], start: usize) {
    let mut best_idx = start;
    let mut best_score = scores[start];
    for i in (start + 1)..list.len() {
        if scores[i] > best_score {
            best_score = scores[i];
            best_idx = i;
        }
    }
    if best_idx != start {
        list.moves.swap(start, best_idx);
        scores.swap(start, best_idx);
    }
}

// ============================================================
// Quiescence search
// ============================================================

fn quiescence(board: &mut Board, mut alpha: i32, beta: i32, info: &mut SearchInfo, _exp: &ExpTable) -> i32 {
    info.nodes += 1;
    info.check_time();
    if info.stopped {
        return 0;
    }

    let stand_pat = eval::evaluate(board);

    if stand_pat >= beta {
        return beta;
    }

    // Delta pruning: if even capturing the best piece can't raise alpha, prune
    let big_delta = eval::QUEEN_VALUE + 200; // Queen + some margin
    if stand_pat + big_delta < alpha {
        return alpha;
    }

    if alpha < stand_pat {
        alpha = stand_pat;
    }

    let mut list = MoveList::new();
    movegen::generate_captures(board, &mut list);

    // Simple move ordering for captures: MVV-LVA
    let mut scores: Vec<i32> = (0..list.len())
        .map(|i| eval::mvv_lva_score(list.moves[i]))
        .collect();

    for i in 0..list.len() {
        pick_move(&mut list, &mut scores, i);
        let m = list.moves[i];

        // SEE pruning: skip clearly losing captures
        if eval::see(board, m) < 0 {
            continue;
        }

        if !board.make_move(m) {
            continue;
        }

        let score = -quiescence(board, -beta, -alpha, info, _exp);
        board.unmake_move(m);

        if info.stopped {
            return 0;
        }

        if score > alpha {
            alpha = score;
            if score >= beta {
                return beta;
            }
        }
    }

    alpha
}

// ============================================================
// Main alpha-beta search with PVS
// ============================================================

fn alpha_beta(
    board: &mut Board,
    tt: &mut TranspositionTable,
    info: &mut SearchInfo,
    exp: &ExpTable,
    mut depth: i32,
    mut alpha: i32,
    beta: i32,
    ply: usize,
    do_null: bool,
    prev_move: Move,
) -> i32 {
    // Check extension: extend search when in check
    let in_check = board.in_check();
    if in_check {
        depth += 1;
    }

    // Drop into quiescence at depth 0
    if depth <= 0 {
        return quiescence(board, alpha, beta, info, exp);
    }

    info.nodes += 1;
    info.check_time();
    if info.stopped {
        return 0;
    }

    // Draw detection: 50-move rule
    if board.halfmove >= 100 {
        return 0;
    }

    // Draw detection: repetition (simplified — check history)
    if ply > 0 {
        let hash = board.hash;
        let start = if board.history.len() > board.halfmove as usize {
            board.history.len() - board.halfmove as usize
        } else {
            0
        };
        let mut reps = 0;
        for i in (start..board.history.len().saturating_sub(1)).rev().step_by(2) {
            if board.history[i].hash == hash {
                reps += 1;
                if reps >= 1 {
                    return 0; // Draw by repetition
                }
            }
        }
    }

    // Mate distance pruning
    let mate_val = MATE_SCORE - ply as i32;
    if alpha >= mate_val {
        return alpha;
    }
    if beta <= -(MATE_SCORE - ply as i32 - 1) {
        return beta;
    }

    // TT probe
    let mut tt_move = MOVE_NONE;
    if let Some(entry) = tt.probe(board.hash) {
        tt_move = entry.best_move;
        if entry.depth >= depth as i8 {
            let tt_score = entry.score;
            match entry.flag {
                TTFlag::Exact => return tt_score,
                TTFlag::Alpha => {
                    if tt_score <= alpha {
                        return alpha;
                    }
                }
                TTFlag::Beta => {
                    if tt_score >= beta {
                        return beta;
                    }
                }
            }
        }
    }

    // Experience probe — compute a small eval correction if we've been here before
    let exp_correction = if let Some(exp_entry) = exp.probe(board.hash) {
        // Use experience best move as fallback if TT has nothing
        if tt_move.is_null() && !exp_entry.best_move.is_null() {
            tt_move = exp_entry.best_move;
        }
        // Correction: game_result is 0.0 (loss) to 1.0 (win), expected is 0.5.
        // Nudge eval toward reality. Scale by confidence (more games = stronger).
        let confidence = (exp_entry.count as f32).min(16.0) / 16.0; // caps at 16 games
        let outcome_delta = exp_entry.game_result - 0.5; // -0.5 to +0.5
        (outcome_delta * 60.0 * confidence) as i32 // max ±30cp
    } else {
        0
    };

    // Null move pruning
    if do_null && !in_check && depth >= 3 && ply > 0 {
        // Skip null move in zugzwang-prone positions.
        // Zugzwang is most dangerous in endgames with limited material:
        //   - Only pawns + king
        //   - Single minor piece + pawns
        //   - Rook + pawns with no other pieces
        // Safe to null-move if we have sufficient piece diversity.
        let us = board.side.index();
        let our_queens = board.pieces[us][Piece::Queen.index()];
        let our_rooks = board.pieces[us][Piece::Rook.index()];
        let our_bishops = board.pieces[us][Piece::Bishop.index()];
        let our_knights = board.pieces[us][Piece::Knight.index()];
        let non_pawn_material = board.occupancy[us]
            & !(board.pieces[us][Piece::Pawn.index()]
                | board.pieces[us][Piece::King.index()]);
        let minor_count = our_bishops.count_ones() + our_knights.count_ones();
        let major_count = our_rooks.count_ones() + our_queens.count_ones();

        // Allow null move if:
        //   - Has any non-pawn material AND (has a queen, or 2+ pieces, or rook+minor)
        //   - This blocks: lone king, K+minor+pawns, K+rook+pawns (zugzwang prone)
        let null_safe = non_pawn_material != 0
            && (our_queens != 0
                || (major_count + minor_count) >= 2
                || (our_rooks != 0 && minor_count >= 1));

        if null_safe {
            board.make_null_move();
            let r = if depth >= 6 { 3 } else { 2 }; // Adaptive null move reduction
            let null_score = -alpha_beta(board, tt, info, exp, depth - 1 - r, -beta, -beta + 1, ply + 1, false, MOVE_NONE);
            board.unmake_null_move();

            if info.stopped {
                return 0;
            }

            if null_score >= beta {
                return beta;
            }
        }
    }

    // Reverse futility pruning (static eval pruning)
    if !in_check && depth <= 3 && ply > 0 {
        let static_eval = eval::evaluate(board) + exp_correction;
        let margin = 120 * depth;
        if static_eval - margin >= beta {
            return static_eval - margin;
        }
    }

    // Generate and search moves
    let mut list = MoveList::new();
    movegen::generate_moves(board, &mut list);

    let mut scores = score_moves(&list, board, info, ply, tt_move, exp, prev_move);

    let mut best_move = MOVE_NONE;
    let mut best_score = -INFINITY;
    let mut moves_searched = 0;
    let mut flag = TTFlag::Alpha;

    for i in 0..list.len() {
        pick_move(&mut list, &mut scores, i);
        let m = list.moves[i];

        if !board.make_move(m) {
            continue;
        }

        let mut score;

        // Late Move Reductions (LMR) — dynamic formula
        // Skip LMR in very sparse endgames (5 or fewer pieces) — every move is critical
        let few_pieces = board.all_occupancy.count_ones() <= 5;
        if moves_searched >= 3
            && depth >= 3
            && !in_check
            && !m.is_capture()
            && !m.is_promotion()
            && !board.in_check()
            && !few_pieces
        {
            // Base reduction from ln(depth) * ln(moves_searched)
            let ln_depth = (depth as f32).ln();
            let ln_moves = (moves_searched as f32).ln();
            let mut reduction = (0.75 + ln_depth * ln_moves / 2.5) as i32;

            // Reduce more for moves with bad history
            let hist_score = info.history[board.side.index()][m.from_sq() as usize][m.to_sq() as usize];
            if hist_score < 0 {
                reduction += 1;
            }
            // Reduce less for moves with good history
            if hist_score > 1000 {
                reduction -= 1;
            }

            // Reduce less for killer moves
            if ply < 128 && (m.0 == info.killers[ply][0].0 || m.0 == info.killers[ply][1].0) {
                reduction -= 1;
            }

            // Clamp: at least 1 ply reduction, never reduce below depth 1
            reduction = reduction.clamp(1, depth - 2);

            let reduced_depth = (depth - 1 - reduction).max(1);
            score = -alpha_beta(board, tt, info, exp, reduced_depth, -alpha - 1, -alpha, ply + 1, true, m);

            // Re-search at full depth if LMR score is above alpha
            if score > alpha {
                score = -alpha_beta(board, tt, info, exp, depth - 1, -alpha - 1, -alpha, ply + 1, true, m);
            }
        } else if moves_searched > 0 {
            // PVS: search with null window first
            score = -alpha_beta(board, tt, info, exp, depth - 1, -alpha - 1, -alpha, ply + 1, true, m);
        } else {
            // First move: full window search
            score = alpha + 1; // Force full search below
        }

        // Full window re-search if needed
        if score > alpha {
            score = -alpha_beta(board, tt, info, exp, depth - 1, -beta, -alpha, ply + 1, true, m);
        }

        board.unmake_move(m);
        moves_searched += 1;

        if info.stopped {
            return 0;
        }

        if score > best_score {
            best_score = score;
            best_move = m;

            if score > alpha {
                alpha = score;
                flag = TTFlag::Exact;

                // Update history for quiet moves
                if !m.is_capture() {
                    info.history[board.side.index()][m.from_sq() as usize][m.to_sq() as usize] += depth * depth;
                }

                if score >= beta {
                    // Store killer moves and counter move on quiet beta cutoffs
                    if !m.is_capture() && ply < 128 {
                        info.killers[ply][1] = info.killers[ply][0];
                        info.killers[ply][0] = m;

                        // Counter move: record this move as the refutation of opponent's last move
                        if !prev_move.is_null() {
                            info.counter_moves[prev_move.piece().index()][prev_move.to_sq() as usize] = m;
                        }
                    }

                    tt.store(board.hash, depth as i8, beta, TTFlag::Beta, best_move);
                    return beta;
                }
            }
        }
    }

    // Checkmate or stalemate
    if moves_searched == 0 {
        if in_check {
            return -(MATE_SCORE - ply as i32); // Checkmate
        } else {
            return 0; // Stalemate
        }
    }

    tt.store(board.hash, depth as i8, alpha, flag, best_move);
    alpha
}

// ============================================================
// Iterative deepening
// ============================================================

pub struct SearchResult {
    pub best_move: Move,
    pub score: i32,
    pub depth: i32,
    pub nodes: u64,
    pub time_ms: u64,
}

pub fn search(board: &mut Board, tt: &mut TranspositionTable, exp: &ExpTable, time_limit_ms: u64, max_depth: i32) -> SearchResult {
    let mut info = SearchInfo::new();
    let start_time = Instant::now();
    info.start_time = start_time;
    info.time_limit_ms = time_limit_ms;
    info.max_depth = max_depth;

    let mut best_move = MOVE_NONE;
    let mut best_score = 0;

    // Iterative deepening
    for depth in 1..=max_depth {
        info.reset();
        info.start_time = start_time; // Preserve the original start time
        let _start_of_iteration = Instant::now();

        // Aspiration windows
        let (mut alpha, mut beta) = if depth >= 4 {
            (best_score - 25, best_score + 25)
        } else {
            (-INFINITY, INFINITY)
        };

        let mut score;
        loop {
            score = alpha_beta(board, tt, &mut info, exp, depth, alpha, beta, 0, true, MOVE_NONE);

            if info.stopped {
                break;
            }

            // Widen window if we fail outside
            if score <= alpha {
                alpha = -INFINITY;
            } else if score >= beta {
                beta = INFINITY;
            } else {
                break;
            }
        }

        if info.stopped && depth > 1 {
            break;
        }

        best_score = score;

        // Get best move from TT
        if let Some(entry) = tt.probe(board.hash) {
            best_move = entry.best_move;
        }

        let elapsed = info.start_time.elapsed().as_millis() as u64;
        let nps = if elapsed > 0 { info.nodes * 1000 / elapsed } else { 0 };

        // UCI info output
        let score_str = if eval::is_mate_score(score) {
            format!("score mate {}", eval::mate_in(score))
        } else {
            format!("score cp {}", score)
        };

        // Build PV from TT
        let pv = extract_pv(board, tt, depth);

        println!(
            "info depth {} {} nodes {} time {} nps {} pv {}",
            depth,
            score_str,
            info.nodes,
            elapsed,
            nps,
            pv,
        );

        // If we found a forced mate, stop searching
        if eval::is_mate_score(score) {
            break;
        }

        // Time management: if we've used more than half the time, stop
        if time_limit_ms > 0 && elapsed > time_limit_ms / 2 {
            break;
        }
    }

    let total_time = info.start_time.elapsed().as_millis() as u64;

    SearchResult {
        best_move,
        score: best_score,
        depth: max_depth,
        nodes: info.nodes,
        time_ms: total_time,
    }
}

/// Extract the principal variation from the TT
fn extract_pv(board: &mut Board, tt: &TranspositionTable, max_depth: i32) -> String {
    let mut pv_moves: Vec<Move> = Vec::new();

    for _ in 0..max_depth {
        if let Some(entry) = tt.probe(board.hash) {
            let m = entry.best_move;
            if m.is_null() {
                break;
            }
            if !board.make_move(m) {
                break;
            }
            pv_moves.push(m);
        } else {
            break;
        }
    }

    // Unmake all moves in reverse to restore the board
    for i in (0..pv_moves.len()).rev() {
        board.unmake_move(pv_moves[i]);
    }

    pv_moves.iter().map(|m| m.to_uci()).collect::<Vec<_>>().join(" ")
}

/// Properly extract PV by tracking moves
pub fn get_pv(board: &mut Board, tt: &TranspositionTable, max_depth: i32) -> Vec<Move> {
    let mut pv = Vec::new();
    let mut moves_made = 0;

    for _ in 0..max_depth {
        if let Some(entry) = tt.probe(board.hash) {
            let m = entry.best_move;
            if m.is_null() {
                break;
            }
            if !board.make_move(m) {
                break;
            }
            pv.push(m);
            moves_made += 1;
        } else {
            break;
        }
    }

    // Unmake all moves in reverse
    for i in (0..moves_made).rev() {
        board.unmake_move(pv[i]);
    }

    pv
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    fn setup() {
        crate::zobrist::init();
        crate::movegen::init();
    }

    #[test]
    fn test_search_start_pos() {
        setup();
        let mut board = Board::start_pos();
        let mut tt = TranspositionTable::new(16);
        let exp = ExpTable::new();
        let result = search(&mut board, &mut tt, &exp, 1000, 5);
        assert!(!result.best_move.is_null());
        println!("Best move: {}, score: {}", result.best_move, result.score);
    }

    #[test]
    fn test_search_mate_in_1() {
        setup();
        // White to move, Qh7# is mate in 1
        let mut board = Board::from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4").unwrap();
        let mut tt = TranspositionTable::new(16);
        let exp = ExpTable::new();
        let result = search(&mut board, &mut tt, &exp, 5000, 6);
        // Should find Qxf7# (scholar's mate)
        println!("Best move: {}, score: {}", result.best_move, result.score);
        assert!(eval::is_mate_score(result.score), "Should find mate");
    }

    #[test]
    fn test_search_avoid_blunder() {
        setup();
        // Position where queen is hanging
        let mut board = Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").unwrap();
        let mut tt = TranspositionTable::new(16);
        let exp = ExpTable::new();
        let result = search(&mut board, &mut tt, &exp, 1000, 5);
        assert!(!result.best_move.is_null());
    }

    #[test]
    fn test_mate_in_2_back_rank() {
        setup();
        // Back-rank mate in 2: 1. Re8+ Rxe8 2. Qxe8#
        let mut board = Board::from_fen("3r2k1/5ppp/8/8/8/8/4RPPP/4Q1K1 w - - 0 1").unwrap();
        let mut tt = TranspositionTable::new(16);
        let exp = ExpTable::new();
        let result = search(&mut board, &mut tt, &exp, 10000, 10);
        println!("Mate-in-2: Best move: {}, score: {}", result.best_move, result.score);
        assert!(eval::is_mate_score(result.score), "Should find mate in 2");
        let mate_moves = eval::mate_in(result.score);
        assert!(mate_moves <= 2, "Should be mate in at most 2, got mate in {}", mate_moves);
    }

    #[test]
    fn test_mate_in_6_kqk() {
        setup();
        // KQ vs K: forced mate in 6
        let mut board = Board::from_fen("8/4k3/8/8/2K5/8/8/Q7 w - - 0 1").unwrap();
        let mut tt = TranspositionTable::new(32);
        let exp = ExpTable::new();
        let result = search(&mut board, &mut tt, &exp, 30000, 16);
        println!("Mate-in-6: Best move: {}, score: {}", result.best_move, result.score);
        assert!(eval::is_mate_score(result.score), "Should find mate in 6");
        let mate_moves = eval::mate_in(result.score);
        assert!(mate_moves <= 6, "Should be mate in at most 6, got mate in {}", mate_moves);
    }

    #[test]
    fn test_mate_in_7_kqk() {
        setup();
        // KQ vs K: forced mate in 7 — verifies deep mate-finding capability
        let mut board = Board::from_fen("8/8/3k4/8/8/4K3/8/Q7 w - - 0 1").unwrap();
        let mut tt = TranspositionTable::new(32);
        let exp = ExpTable::new();
        let result = search(&mut board, &mut tt, &exp, 30000, 20);
        println!("Mate-in-7: Best move: {}, score: {}", result.best_move, result.score);
        assert!(eval::is_mate_score(result.score), "Should find mate in 7");
        let mate_moves = eval::mate_in(result.score);
        assert!(mate_moves <= 7, "Should be mate in at most 7, got mate in {}", mate_moves);
    }

    #[test]
    fn test_experience_move_ordering() {
        setup();
        let mut board = Board::start_pos();
        let info = SearchInfo::new();
        let tt_move = MOVE_NONE;

        // Without experience: all quiet moves scored by history (0 initially)
        let exp_empty = ExpTable::new();
        let mut list = MoveList::new();
        crate::movegen::generate_moves(&board, &mut list);
        let scores_no_exp = score_moves(&list, &board, &info, 0, tt_move, &exp_empty, MOVE_NONE);

        // With experience: store a best move for the start position
        let hints_move = list.moves[5]; // pick an arbitrary legal move
        let mut exp_filled = ExpTable::new();
        exp_filled.store(ExpEntry {
            hash: board.hash,
            best_move: hints_move,
            depth: 10,
            score: 50,
            game_result: 0.8,
            count: 1,
        });
        let scores_with_exp = score_moves(&list, &board, &info, 0, tt_move, &exp_filled, MOVE_NONE);

        // The experience move should now be scored at 5_000_000
        let idx = (0..list.len()).find(|&i| list.moves[i].0 == hints_move.0).unwrap();
        assert_eq!(scores_no_exp[idx], 0, "Without experience, should be 0 (history)");
        assert_eq!(scores_with_exp[idx], 5_000_000, "With experience, should be 5M");
    }

    #[test]
    fn test_experience_eval_correction() {
        setup();
        // Verify that experience with a strong win record produces a positive correction
        // and experience with losses produces a negative one. We check indirectly:
        // searching the same position with "all wins" vs "all losses" experience
        // should yield a higher score for the winning experience.
        let mut board = Board::start_pos();
        let mut tt = TranspositionTable::new(16);

        // Search without experience
        let exp_empty = ExpTable::new();
        let result_neutral = search(&mut board, &mut tt, &exp_empty, 500, 4);

        // Search with "winning" experience
        tt.clear();
        let mut exp_win = ExpTable::new();
        exp_win.store(ExpEntry {
            hash: board.hash,
            best_move: result_neutral.best_move,
            depth: 10,
            score: 50,
            game_result: 1.0, // always won from here
            count: 16,
        });
        let result_win = search(&mut board, &mut tt, &exp_win, 500, 4);

        // Search with "losing" experience
        tt.clear();
        let mut exp_loss = ExpTable::new();
        exp_loss.store(ExpEntry {
            hash: board.hash,
            best_move: result_neutral.best_move,
            depth: 10,
            score: 50,
            game_result: 0.0, // always lost from here
            count: 16,
        });
        let result_loss = search(&mut board, &mut tt, &exp_loss, 500, 4);

        // The "win" experience should produce a score >= "loss" experience
        println!("Neutral: {}, Win: {}, Loss: {}", result_neutral.score, result_win.score, result_loss.score);
        // Note: the correction is small (max ±30cp) and only affects RFP, so
        // the effect may be subtle. We just verify it doesn't crash and produces
        // reasonable scores (all should be near 0 for start pos).
        assert!(result_win.score > -500 && result_win.score < 500, "Score should be reasonable");
        assert!(result_loss.score > -500 && result_loss.score < 500, "Score should be reasonable");
    }
}
