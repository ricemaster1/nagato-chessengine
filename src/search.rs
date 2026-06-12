
use crate::bitboard::*;
use crate::board::Board;
use crate::eval::{self, INFINITY, MATE_SCORE};
use crate::nnue;
use crate::learn::ExpTable;
use crate::movegen;
use crate::moves::*;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Instant;

const TT_BUCKET_SIZE: usize = 4;
const LMP_DEPTH_MAX: i32 = 1;
const LMP_BASE: usize = 6;
const LMP_STEP: usize = 2;
const FUTILITY_DEPTH_MAX: i32 = 2;
const FUTILITY_MARGIN_BASE: i32 = 140;
const FUTILITY_MARGIN_STEP: i32 = 60;
const FUTILITY_IMPROVING_BONUS: i32 = 20;
const PROBCUT_MIN_DEPTH: i32 = 5;
const PROBCUT_MARGIN: i32 = 180;
const PROBCUT_REDUCTION: i32 = 3;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TTFlag {
    None,
    Exact,
    Alpha,
    Beta,
}

#[derive(Clone, Copy)]
pub struct TTEntry {
    pub hash: u64,
    pub depth: i8,
    pub score: i32,
    pub flag: TTFlag,
    pub best_move: Move,
    pub generation: u8,
}

impl TTEntry {
    const EMPTY: TTEntry = TTEntry {
        hash: 0,
        depth: 0,
        score: 0,
        flag: TTFlag::None,
        best_move: MOVE_NONE,
        generation: 0,
    };
}

pub struct TranspositionTable {
    buckets: Vec<[TTEntry; TT_BUCKET_SIZE]>,
    num_buckets: usize,
    pub generation: u8,
}

impl TranspositionTable {
    pub fn new(size_mb: usize) -> Self {
        let bucket_size = std::mem::size_of::<[TTEntry; TT_BUCKET_SIZE]>();
        let num_buckets = (size_mb * 1024 * 1024) / bucket_size;
        TranspositionTable {
            buckets: vec![[TTEntry::EMPTY; TT_BUCKET_SIZE]; num_buckets],
            num_buckets,
            generation: 0,
        }
    }

    pub fn new_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    #[inline]
    pub fn probe(&self, hash: u64) -> Option<&TTEntry> {
        let idx = (hash as usize) % self.num_buckets;
        let bucket = &self.buckets[idx];
        for entry in bucket.iter() {
            if entry.hash == hash && entry.flag != TTFlag::None {
                return Some(entry);
            }
        }
        None
    }

    #[inline]
    pub fn store(&mut self, hash: u64, depth: i8, score: i32, flag: TTFlag, best_move: Move) {
        let idx = (hash as usize) % self.num_buckets;
        let bucket = &mut self.buckets[idx];
        let gen = self.generation;

        let mut replace_idx = 0;
        let mut worst_value = i32::MAX;

        for (i, entry) in bucket.iter().enumerate() {
            if entry.flag == TTFlag::None {
                replace_idx = i;
                break;
            }
            if entry.hash == hash {
                replace_idx = i;
                break;
            }
            let age_penalty = if entry.generation != gen { 4 } else { 0 };
            let value = entry.depth as i32 - age_penalty;
            if value < worst_value {
                worst_value = value;
                replace_idx = i;
            }
        }

        bucket[replace_idx] = TTEntry {
            hash,
            depth,
            score,
            flag,
            best_move,
            generation: gen,
        };
    }

    pub fn clear(&mut self) {
        for bucket in self.buckets.iter_mut() {
            *bucket = [TTEntry::EMPTY; TT_BUCKET_SIZE];
        }
    }
}

pub struct SearchInfo {
    pub nodes: u64,
    pub start_time: Instant,
    pub time_limit_ms: u64,
    pub max_depth: i32,
    pub stopped: bool,

    pub killers: [[Move; 2]; 128],

    pub history: [[[i32; 64]; 64]; 2],

    pub counter_moves: [[Move; 64]; 6],

    pub eval_stack: [i32; 128],

    pub root_best: Move,
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
            eval_stack: [0; 128],
            root_best: MOVE_NONE,
        }
    }

    pub fn reset(&mut self) {
        self.nodes = 0;
        self.stopped = false;
        self.root_best = MOVE_NONE;
        self.killers = [[MOVE_NONE; 2]; 128];
        for c in 0..2 {
            for f in 0..64 {
                for t in 0..64 {
                    self.history[c][f][t] /= 2;
                }
            }
        }
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

fn score_moves(list: &MoveList, board: &Board, info: &SearchInfo, ply: usize, tt_move: Move, exp: &ExpTable, prev_move: Move) -> Vec<i32> {
    let exp_move = exp.probe(board.hash).map(|e| e.best_move);

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
            scores[i] = 10_000_000;
        } else if exp_move.is_some() && m.0 == exp_move.unwrap().0 && !exp_move.unwrap().is_null() {
            scores[i] = 5_000_000;
        } else if m.is_capture() || m.is_en_passant() {
            let see_val = eval::see(board, m);
            if see_val >= 0 {
                let cap_score = 1_000_000 + eval::mvv_lva_score(m);
                scores[i] = cap_score;
            } else {
                scores[i] = -100_000 + see_val;
            }
        } else if m.is_promotion() {
            scores[i] = 900_000;
        } else if ply < 128 && m.0 == info.killers[ply][0].0 {
            scores[i] = 800_000;
        } else if ply < 128 && m.0 == info.killers[ply][1].0 {
            scores[i] = 700_000;
        } else if counter.is_some() && m.0 == counter.unwrap().0 {
            scores[i] = 650_000;
        } else {
            scores[i] = info.history[board.side.index()][m.from_sq() as usize][m.to_sq() as usize];
        }
    }
    scores
}

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

fn quiescence(board: &mut Board, mut alpha: i32, beta: i32, info: &mut SearchInfo, _exp: &ExpTable, ply: usize) -> i32 {
    if ply >= nnue::MAX_PLY {
        board.ensure_acc_computed();
        return eval::evaluate(board);
    }

    info.nodes += 1;
    info.check_time();
    if info.stopped {
        return 0;
    }

    board.ensure_acc_computed();
    let stand_pat = eval::evaluate(board);

    if stand_pat >= beta {
        return beta;
    }

    let big_delta = eval::QUEEN_VALUE + 200;
    if stand_pat + big_delta < alpha {
        return alpha;
    }

    if alpha < stand_pat {
        alpha = stand_pat;
    }

    let mut list = MoveList::new();
    movegen::generate_captures(board, &mut list);

    let mut scores: Vec<i32> = (0..list.len())
        .map(|i| eval::mvv_lva_score(list.moves[i]))
        .collect();

    for i in 0..list.len() {
        pick_move(&mut list, &mut scores, i);
        let m = list.moves[i];

        if eval::see(board, m) < 0 {
            continue;
        }

        if !board.make_move(m) {
            continue;
        }

        let score = -quiescence(board, -beta, -alpha, info, _exp, ply + 1);
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
    if ply >= nnue::MAX_PLY {
        board.ensure_acc_computed();
        return eval::evaluate(board);
    }

    let in_check = board.in_check();
    if in_check {
        depth += 1;
    }

    if depth <= 0 {
        return quiescence(board, alpha, beta, info, exp, ply);
    }

    info.nodes += 1;
    info.check_time();
    if info.stopped {
        return 0;
    }

    if board.halfmove >= 100 {
        return 0;
    }

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
                    return 0;
                }
            }
        }
    }

    let mate_val = MATE_SCORE - ply as i32;
    if alpha >= mate_val {
        return alpha;
    }
    if beta <= -(MATE_SCORE - ply as i32 - 1) {
        return beta;
    }

    if let Some(tb_score) = crate::syzygy::probe_wdl_score(board, depth, ply) {
        return tb_score;
    }

    let mut tt_move = MOVE_NONE;
    if let Some(entry) = tt.probe(board.hash) {
        tt_move = entry.best_move;
        if ply > 0 && entry.depth >= depth as i8 {
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
                TTFlag::None => {}
            }
        }
    }

    let is_pv = beta - alpha > 1;

    if depth >= 4 && tt_move.is_null() && !in_check && !is_pv {
        // Internal iterative reduction when we have no strong TT guidance.
        depth -= 1;
    }

    let exp_correction = if let Some(exp_entry) = exp.probe(board.hash) {
        if tt_move.is_null() && !exp_entry.best_move.is_null() {
            tt_move = exp_entry.best_move;
        }
        let confidence = (exp_entry.count as f32).min(16.0) / 16.0;
        let outcome_delta = exp_entry.game_result - 0.5;
        (outcome_delta * 60.0 * confidence) as i32
    } else {
        0
    };

    if do_null && !in_check && depth >= 3 && ply > 0 {
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

        let null_safe = non_pawn_material != 0
            && (our_queens != 0
                || (major_count + minor_count) >= 2
                || (our_rooks != 0 && minor_count >= 1));

        if null_safe {
            board.make_null_move();
            let r = if depth >= 6 { 3 } else { 2 };
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

    let static_eval = if !in_check {
        board.ensure_acc_computed();
        eval::evaluate(board) + exp_correction
    } else {
        0
    };

    if ply < 128 {
        info.eval_stack[ply] = static_eval;
    }

    let improving = !in_check && ply >= 2 && ply < 128 && static_eval > info.eval_stack[ply - 2];

    if !in_check && depth <= 3 && ply > 0 {
        let margin = if improving { 100 * depth } else { 120 * depth };
        if static_eval - margin >= beta {
            return static_eval - margin;
        }
    }

    if !in_check
        && !is_pv
        && depth >= PROBCUT_MIN_DEPTH
        && beta < MATE_SCORE - 512
    {
        let prob_beta = beta + PROBCUT_MARGIN;
        if static_eval >= prob_beta - 120 {
            let mut prob_list = MoveList::new();
            movegen::generate_captures(board, &mut prob_list);
            let mut prob_scores: Vec<i32> = (0..prob_list.len())
                .map(|i| eval::mvv_lva_score(prob_list.moves[i]))
                .collect();

            for i in 0..prob_list.len() {
                pick_move(&mut prob_list, &mut prob_scores, i);
                let m = prob_list.moves[i];

                if eval::see(board, m) < 0 {
                    continue;
                }
                if !board.make_move(m) {
                    continue;
                }

                let reduced_depth = (depth - 1 - PROBCUT_REDUCTION).max(1);
                let score = -alpha_beta(
                    board,
                    tt,
                    info,
                    exp,
                    reduced_depth,
                    -prob_beta,
                    -prob_beta + 1,
                    ply + 1,
                    false,
                    m,
                );

                board.unmake_move(m);

                if info.stopped {
                    return 0;
                }
                if score >= prob_beta {
                    return score;
                }
            }
        }
    }

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

        let quiet_move = !m.is_capture() && !m.is_promotion() && !m.is_en_passant();
        let is_killer = ply < 128 && (m.0 == info.killers[ply][0].0 || m.0 == info.killers[ply][1].0);
        let protected_quiet = !quiet_move || m.0 == tt_move.0 || is_killer;

        if !in_check && !is_pv && !protected_quiet {
            if depth <= LMP_DEPTH_MAX {
                let lmp_limit = LMP_BASE + LMP_STEP * (depth as usize);
                if moves_searched >= lmp_limit {
                    continue;
                }
            }

            if depth <= FUTILITY_DEPTH_MAX && moves_searched > 0 {
                let mut futility_margin = FUTILITY_MARGIN_BASE + FUTILITY_MARGIN_STEP * depth;
                if improving {
                    futility_margin -= FUTILITY_IMPROVING_BONUS;
                }
                if static_eval + futility_margin <= alpha {
                    continue;
                }
            }
        }

        let mover = board.side.index();
        let from_sq = m.from_sq() as usize;
        let to_sq = m.to_sq() as usize;
        let alpha_before_move = alpha;

        if !board.make_move(m) {
            continue;
        }

        let mut score;

        let few_pieces = board.all_occupancy.count_ones() <= 5;
        if moves_searched >= 3
            && depth >= 3
            && !in_check
            && !m.is_capture()
            && !m.is_promotion()
            && !board.in_check()
            && !few_pieces
        {
            let ln_depth = (depth as f32).ln();
            let ln_moves = (moves_searched as f32).ln();
            let mut reduction = (0.75 + ln_depth * ln_moves / 2.5) as i32;

            if is_pv {
                reduction -= 1;
            }

            if improving {
                reduction -= 1;
            }

            let hist_score = info.history[board.side.index()][m.from_sq() as usize][m.to_sq() as usize];
            if hist_score < -500 {
                reduction += 1;
            } else if hist_score > 2000 {
                reduction -= 1;
            }

            if ply < 128 && (m.0 == info.killers[ply][0].0 || m.0 == info.killers[ply][1].0) {
                reduction -= 1;
            }

            reduction = reduction.clamp(1, depth - 2);

            let reduced_depth = (depth - 1 - reduction).max(1);
            score = -alpha_beta(board, tt, info, exp, reduced_depth, -alpha - 1, -alpha, ply + 1, true, m);

            if score > alpha {
                score = -alpha_beta(board, tt, info, exp, depth - 1, -alpha - 1, -alpha, ply + 1, true, m);
            }
        } else if moves_searched > 0 {
            score = -alpha_beta(board, tt, info, exp, depth - 1, -alpha - 1, -alpha, ply + 1, true, m);
        } else {
            score = alpha + 1;
        }

        if score > alpha {
            score = -alpha_beta(board, tt, info, exp, depth - 1, -beta, -alpha, ply + 1, true, m);
        }

        board.unmake_move(m);
        moves_searched += 1;

        if quiet_move && score <= alpha_before_move {
            info.history[mover][from_sq][to_sq] = (info.history[mover][from_sq][to_sq] - depth * depth).clamp(-32_000, 32_000);
        }

        if info.stopped {
            return 0;
        }

        if score > best_score {
            best_score = score;
            best_move = m;
            if ply == 0 {
                info.root_best = m;
            }

            if score > alpha {
                alpha = score;
                flag = TTFlag::Exact;

                if !m.is_capture() {
                    info.history[board.side.index()][m.from_sq() as usize][m.to_sq() as usize] =
                        (info.history[board.side.index()][m.from_sq() as usize][m.to_sq() as usize] + depth * depth)
                            .clamp(-32_000, 32_000);
                }

                if score >= beta {
                    if !m.is_capture() && ply < 128 {
                        info.killers[ply][1] = info.killers[ply][0];
                        info.killers[ply][0] = m;

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

    if moves_searched == 0 {
        if in_check {
            return -(MATE_SCORE - ply as i32);
        } else {
            return 0;
        }
    }

    tt.store(board.hash, depth as i8, alpha, flag, best_move);
    alpha
}

pub struct SearchResult {
    pub best_move: Move,
    pub score: i32,
    pub depth: i32,
    pub nodes: u64,
    pub time_ms: u64,
}

#[derive(Clone, Copy)]
struct WorkerResult {
    best_move: Move,
    score: i32,
    depth: i32,
    nodes: u64,
}

pub fn search_threads(
    board: &Board,
    exp: &ExpTable,
    time_limit_ms: u64,
    max_depth: i32,
    threads: usize,
    hash_mb: usize,
) -> SearchResult {
    let start_time = Instant::now();
    let mut root = board.clone();
    let mut root_list = MoveList::new();
    movegen::generate_moves(&root, &mut root_list);
    let root_moves: Vec<Move> = (0..root_list.len())
        .map(|i| root_list.moves[i])
        .filter(|m| {
            if root.make_move(*m) {
                root.unmake_move(*m);
                true
            } else {
                false
            }
        })
        .collect();

    if root_moves.is_empty() {
        return SearchResult {
            best_move: MOVE_NONE,
            score: 0,
            depth: 0,
            nodes: 0,
            time_ms: 0,
        };
    }

    let worker_count = threads.max(1);
    let per_thread_hash = (hash_mb / worker_count).max(8);
    let shared_stop = AtomicBool::new(false);
    let worker_results: Vec<WorkerResult> = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        let root_moves_ref = &root_moves;
        let shared_stop_ref = &shared_stop;

        for worker_id in 0..worker_count {
            handles.push(scope.spawn(move || {
                let mut local_board = board.clone();
                let mut local_tt = TranspositionTable::new(per_thread_hash);
                local_tt.new_generation();

                let mut root_order = root_moves_ref.clone();
                if !root_order.is_empty() {
                    let n = root_order.len();
                    root_order.rotate_left(worker_id % n);
                }

                let mut total_nodes = 0u64;
                let mut worker_best_move = root_order[0];
                let mut worker_best_score = -INFINITY;
                let mut worker_reached_depth = 0;
                let mut info = SearchInfo::new();
                info.start_time = start_time;
                info.time_limit_ms = time_limit_ms;
                info.max_depth = max_depth;

                for depth in 1..=max_depth {
                    if shared_stop_ref.load(Ordering::Relaxed) {
                        break;
                    }

                    info.reset();
                    info.start_time = start_time;
                    info.time_limit_ms = time_limit_ms;
                    info.max_depth = depth;

                    let (mut alpha, mut beta) = if depth >= 4 {
                        (worker_best_score - 25, worker_best_score + 25)
                    } else {
                        (-INFINITY, INFINITY)
                    };

                    let mut depth_best_move = MOVE_NONE;
                    let mut depth_best_score = -INFINITY;

                    for &m in &root_order {
                        if shared_stop_ref.load(Ordering::Relaxed) {
                            break;
                        }

                        if !local_board.make_move(m) {
                            continue;
                        }

                        let mut score;
                        loop {
                            score = -alpha_beta(
                                &mut local_board,
                                &mut local_tt,
                                &mut info,
                                exp,
                                depth - 1,
                                alpha,
                                beta,
                                1,
                                true,
                                m,
                            );

                            if info.stopped {
                                break;
                            }

                            if score <= alpha {
                                alpha = -INFINITY;
                            } else if score >= beta {
                                beta = INFINITY;
                            } else {
                                break;
                            }
                        }

                        local_board.unmake_move(m);

                        if info.stopped {
                            shared_stop_ref.store(true, Ordering::Relaxed);
                            break;
                        }

                        if score > depth_best_score {
                            depth_best_score = score;
                            depth_best_move = m;
                            if eval::is_mate_score(score) {
                                shared_stop_ref.store(true, Ordering::Relaxed);
                            }
                        }
                    }

                    total_nodes = total_nodes.saturating_add(info.nodes);

                    if info.stopped {
                        break;
                    }

                    if !depth_best_move.is_null() {
                        worker_best_move = depth_best_move;
                        worker_best_score = depth_best_score;
                        worker_reached_depth = depth;
                    }

                    let elapsed = start_time.elapsed().as_millis() as u64;
                    if time_limit_ms > 0 && elapsed > time_limit_ms / 2 {
                        shared_stop_ref.store(true, Ordering::Relaxed);
                        break;
                    }
                }

                WorkerResult {
                    best_move: worker_best_move,
                    score: worker_best_score,
                    depth: worker_reached_depth,
                    nodes: total_nodes,
                }
            }));
        }

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut best = WorkerResult {
        best_move: root_moves[0],
        score: -INFINITY,
        depth: 0,
        nodes: 0,
    };
    let mut global_nodes = 0u64;

    for wr in worker_results {
        global_nodes = global_nodes.saturating_add(wr.nodes);

        let best_is_win = eval::is_mate_score(best.score) && best.score > 0;
        let wr_is_win = eval::is_mate_score(wr.score) && wr.score > 0;
        let best_is_loss = eval::is_mate_score(best.score) && best.score < 0;
        let wr_is_loss = eval::is_mate_score(wr.score) && wr.score < 0;

        let choose_wr = if best_is_win {
            wr_is_win && wr.score > best.score
        } else if best_is_loss {
            wr_is_loss && wr.score < best.score
        } else if wr_is_win {
            true
        } else if wr_is_loss {
            false
        } else {
            wr.score > best.score || (wr.score == best.score && wr.depth > best.depth)
        };

        if choose_wr {
            best = wr;
        }
    }

    let elapsed = start_time.elapsed().as_millis() as u64;
    let nps = if elapsed > 0 { global_nodes * 1000 / elapsed } else { 0 };
    let score_str = if eval::is_mate_score(best.score) {
        format!("score mate {}", eval::mate_in(best.score))
    } else {
        format!("score cp {}", best.score)
    };
    println!(
        "info depth {} {} nodes {} time {} nps {} pv {}",
        best.depth,
        score_str,
        global_nodes,
        elapsed,
        nps,
        best.best_move.to_uci(),
    );
    let _ = std::io::stdout().flush();

    SearchResult {
        best_move: best.best_move,
        score: best.score,
        depth: best.depth,
        nodes: global_nodes,
        time_ms: elapsed,
    }
}

pub fn search(board: &mut Board, tt: &mut TranspositionTable, exp: &ExpTable, time_limit_ms: u64, max_depth: i32) -> SearchResult {
    let mut info = SearchInfo::new();
    let start_time = Instant::now();
    info.start_time = start_time;
    info.time_limit_ms = time_limit_ms;
    info.max_depth = max_depth;

    tt.new_generation();

    let mut best_move = MOVE_NONE;
    let mut best_score = 0;

    for depth in 1..=max_depth {
        info.reset();
        info.start_time = start_time;
        let _start_of_iteration = Instant::now();

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

        if !info.root_best.is_null() {
            best_move = info.root_best;
        } else if let Some(entry) = tt.probe(board.hash) {
            best_move = entry.best_move;
        }

        let elapsed = info.start_time.elapsed().as_millis() as u64;
        let nps = if elapsed > 0 { info.nodes * 1000 / elapsed } else { 0 };

        let score_str = if eval::is_mate_score(score) {
            format!("score mate {}", eval::mate_in(score))
        } else {
            format!("score cp {}", score)
        };

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
        let _ = std::io::stdout().flush();

        if eval::is_mate_score(score) {
            break;
        }

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

    for i in (0..pv_moves.len()).rev() {
        board.unmake_move(pv_moves[i]);
    }

    pv_moves.iter().map(|m| m.to_uci()).collect::<Vec<_>>().join(" ")
}

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

    for i in (0..moves_made).rev() {
        board.unmake_move(pv[i]);
    }

    pv
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::learn::ExpEntry;

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
        let mut board = Board::from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4").unwrap();
        let mut tt = TranspositionTable::new(16);
        let exp = ExpTable::new();
        let result = search(&mut board, &mut tt, &exp, 5000, 6);
        println!("Best move: {}, score: {}", result.best_move, result.score);
        assert!(eval::is_mate_score(result.score), "Should find mate");
    }

    #[test]
    fn test_search_avoid_blunder() {
        setup();
        let mut board = Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").unwrap();
        let mut tt = TranspositionTable::new(16);
        let exp = ExpTable::new();
        let result = search(&mut board, &mut tt, &exp, 1000, 5);
        assert!(!result.best_move.is_null());
    }

    #[test]
    fn test_mate_in_2_back_rank() {
        setup();
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
    #[ignore]
    fn test_mate_in_6_kqk() {
        setup();
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

        let exp_empty = ExpTable::new();
        let mut list = MoveList::new();
        crate::movegen::generate_moves(&board, &mut list);
        let scores_no_exp = score_moves(&list, &board, &info, 0, tt_move, &exp_empty, MOVE_NONE);

        let hints_move = list.moves[5];
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

        let idx = (0..list.len()).find(|&i| list.moves[i].0 == hints_move.0).unwrap();
        assert_eq!(scores_no_exp[idx], 0, "Without experience, should be 0 (history)");
        assert_eq!(scores_with_exp[idx], 5_000_000, "With experience, should be 5M");
    }

    #[test]
    fn test_experience_eval_correction() {
        setup();
        let mut board = Board::start_pos();
        let mut tt = TranspositionTable::new(16);

        let exp_empty = ExpTable::new();
        let result_neutral = search(&mut board, &mut tt, &exp_empty, 500, 4);

        tt.clear();
        let mut exp_win = ExpTable::new();
        exp_win.store(ExpEntry {
            hash: board.hash,
            best_move: result_neutral.best_move,
            depth: 10,
            score: 50,
            game_result: 1.0,
            count: 16,
        });
        let result_win = search(&mut board, &mut tt, &exp_win, 500, 4);

        tt.clear();
        let mut exp_loss = ExpTable::new();
        exp_loss.store(ExpEntry {
            hash: board.hash,
            best_move: result_neutral.best_move,
            depth: 10,
            score: 50,
            game_result: 0.0,
            count: 16,
        });
        let result_loss = search(&mut board, &mut tt, &exp_loss, 500, 4);

        println!("Neutral: {}, Win: {}, Loss: {}", result_neutral.score, result_win.score, result_loss.score);
        assert!(result_win.score > -500 && result_win.score < 500, "Score should be reasonable");
        assert!(result_loss.score > -500 && result_loss.score < 500, "Score should be reasonable");
    }
}
