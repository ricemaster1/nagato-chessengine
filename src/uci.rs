
use crate::bitboard::*;
use crate::board::Board;
use crate::eval;
use crate::learn::{self, ExpTable, GameRecorder};
use crate::movegen;
use crate::moves::*;
use crate::nnue;
use crate::search::{self, TranspositionTable};
use std::io::{self, BufRead};
use std::path::PathBuf;

const ENGINE_NAME: &str = "Nagato";
const ENGINE_AUTHOR: &str = "Nagato Team";

pub fn uci_loop() {
    let stdin = io::stdin();
    let mut board = Board::start_pos();
    let mut hash_mb = 64usize;
    let mut tt = TranspositionTable::new(hash_mb);

    // Temporary stabilization: keep engine behavior single-threaded.
    let max_threads = 1usize;
    let mut threads = 1usize;

    let mut exp_path = PathBuf::from("nagato.exp");
    let mut exp_table = ExpTable::new();
    let mut recorder = GameRecorder::new();
    let mut use_experience = true;

    match exp_table.load(&exp_path) {
        Ok(n) if n > 0 => eprintln!("info string loaded {} experience entries", n),
        Ok(_) => {}
        Err(e) => eprintln!("info string could not load experience: {}", e),
    }

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }

        match tokens[0] {
            "uci" => {
                println!("id name {}", ENGINE_NAME);
                println!("id author {}", ENGINE_AUTHOR);
                println!("option name Hash type spin default 64 min 1 max 4096");
                println!("option name Threads type spin default 1 min 1 max {}", max_threads);
                println!("option name SyzygyPath type string default");
                println!("option name SyzygyProbeDepth type spin default 1 min 1 max 64");
                println!("option name ExperienceFile type string default nagato.exp");
                println!("option name Experience type check default true");
                println!("uciok");
            }
            "isready" => {
                println!("readyok");
            }
            "ucinewgame" => {
                if use_experience && recorder.recorded_count() > 0 {
                    recorder.flush(&mut exp_table, learn::GameResult::Draw);
                    if let Err(e) = exp_table.save(&exp_path) {
                        eprintln!("info string could not save experience: {}", e);
                    }
                }
                board = Board::start_pos();
                tt.clear();
                recorder.clear();
            }
            "position" => {
                parse_position(&tokens, &mut board);
            }
            "go" => {
                let (time_ms, depth) = parse_go(&tokens, &board);
                if use_experience {
                    recorder.set_our_color(board.side.index() as u8);
                }
                let exp_ref = if use_experience { &exp_table } else { &ExpTable::new() };
                let _ = threads;
                let _ = hash_mb;
                let result = search::search(&mut board, &mut tt, exp_ref, time_ms, depth);
                if use_experience {
                    recorder.record(
                        board.hash,
                        result.best_move,
                        result.depth as i8,
                        result.score.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                        board.side.index() as u8,
                    );
                }
                println!("bestmove {}", result.best_move);
            }
            "gameover" => {
                if use_experience {
                    let result = if tokens.len() >= 2 {
                        match tokens[1] {
                            "win" => learn::GameResult::Win,
                            "loss" => learn::GameResult::Loss,
                            _ => learn::GameResult::Draw,
                        }
                    } else {
                        learn::GameResult::Draw
                    };
                    recorder.flush(&mut exp_table, result);
                    if let Err(e) = exp_table.save(&exp_path) {
                        eprintln!("info string could not save experience: {}", e);
                    } else {
                        eprintln!("info string experience saved ({} entries)", exp_table.len());
                    }
                }
            }
            "quit" => {
                if use_experience && recorder.recorded_count() > 0 {
                    recorder.flush(&mut exp_table, learn::GameResult::Draw);
                    let _ = exp_table.save(&exp_path);
                }
                break;
            }
            "d" | "display" => {
                board.print();
            }
            "perft" => {
                if tokens.len() >= 2 {
                    if let Ok(depth) = tokens[1].parse::<u32>() {
                        let start = std::time::Instant::now();
                        let count = movegen::perft_divide(&mut board, depth);
                        let elapsed = start.elapsed().as_millis();
                        let nps = if elapsed > 0 { count * 1000 / elapsed as u64 } else { 0 };
                        println!("\nNodes: {} Time: {}ms NPS: {}", count, elapsed, nps);
                    }
                }
            }
            "eval" => {
                board.ensure_acc_computed();
                let score = eval::evaluate(&board);
                println!("Eval: {} cp (from {}'s perspective)",
                    score,
                    match board.side { Color::White => "white", Color::Black => "black" }
                );
            }
            "bench" => {
                run_bench(&mut tt);
            }
            "datagen" => {
                let mut num_games = 1000u32;
                let mut depth = 6i32;
                let mut output = "training_data.bin".to_string();
                let mut random_plies = 8u32;

                let mut i = 1;
                while i < tokens.len() {
                    match tokens[i] {
                        "games" => { i += 1; if i < tokens.len() { num_games = tokens[i].parse().unwrap_or(1000); } }
                        "depth" => { i += 1; if i < tokens.len() { depth = tokens[i].parse().unwrap_or(6); } }
                        "output" => { i += 1; if i < tokens.len() { output = tokens[i].to_string(); } }
                        "random_plies" => { i += 1; if i < tokens.len() { random_plies = tokens[i].parse().unwrap_or(8); } }
                        _ => {}
                    }
                    i += 1;
                }

                crate::datagen::generate(num_games, depth, &output, random_plies);
            }
            "train" => {
                let mut input = "training_data.bin".to_string();
                let mut output = "nn.bin".to_string();
                let mut epochs = 10u32;
                let mut batch_size = 16384usize;
                let mut lr = 0.001f32;
                let mut lambda = 0.75f32;

                let mut i = 1;
                while i < tokens.len() {
                    match tokens[i] {
                        "input" => { i += 1; if i < tokens.len() { input = tokens[i].to_string(); } }
                        "output" => { i += 1; if i < tokens.len() { output = tokens[i].to_string(); } }
                        "epochs" => { i += 1; if i < tokens.len() { epochs = tokens[i].parse().unwrap_or(10); } }
                        "batch" => { i += 1; if i < tokens.len() { batch_size = tokens[i].parse().unwrap_or(16384); } }
                        "lr" => { i += 1; if i < tokens.len() { lr = tokens[i].parse().unwrap_or(0.001); } }
                        "lambda" => { i += 1; if i < tokens.len() { lambda = tokens[i].parse().unwrap_or(0.75); } }
                        _ => {}
                    }
                    i += 1;
                }

                eprintln!("info string loading training data from {}", input);
                let samples = nnue::trainer::load_samples(&input);
                eprintln!("info string loaded {} samples, training...", samples.len());
                let w = nnue::trainer::train(&samples, epochs, batch_size, lr, lambda);
                match nnue::trainer::save_weights(&w, &output) {
                    Ok(_) => eprintln!("info string weights saved to {}", output),
                    Err(e) => eprintln!("info string save error: {}", e),
                }
            }
            "setoption" => {
                parse_setoption(
                    &tokens,
                    &mut tt,
                    &mut hash_mb,
                    &mut threads,
                    max_threads,
                    &mut exp_path,
                    &mut exp_table,
                    &mut use_experience,
                );
            }
            _ => {
            }
        }
    }
}

fn parse_setoption(
    tokens: &[&str],
    tt: &mut TranspositionTable,
    hash_mb: &mut usize,
    threads: &mut usize,
    max_threads: usize,
    exp_path: &mut PathBuf,
    exp_table: &mut ExpTable,
    use_experience: &mut bool,
) {
    let mut name = String::new();
    let mut value = String::new();
    let mut reading_name = false;
    let mut reading_value = false;

    for &token in &tokens[1..] {
        match token {
            "name" => { reading_name = true; reading_value = false; }
            "value" => { reading_name = false; reading_value = true; }
            _ => {
                if reading_name {
                    if !name.is_empty() { name.push(' '); }
                    name.push_str(token);
                } else if reading_value {
                    if !value.is_empty() { value.push(' '); }
                    value.push_str(token);
                }
            }
        }
    }

    let name_lower = name.to_lowercase();
    match name_lower.as_str() {
        "hash" => {
            if let Ok(mb) = value.parse::<usize>() {
                *hash_mb = mb.clamp(1, 4096);
                *tt = TranspositionTable::new(*hash_mb);
            }
        }
        "threads" => {
            let requested = value.parse::<usize>().ok().unwrap_or(1);
            if requested != 1 {
                eprintln!("info string Threads is temporarily locked to 1 for stability");
            }
            let _ = max_threads;
            *threads = 1;
        }
        "syzygypath" => {
            crate::syzygy::set_path(&value);
            if value.is_empty() {
                eprintln!("info string SyzygyPath cleared");
            } else {
                eprintln!("info string SyzygyPath set to {}", value);
            }
        }
        "syzygyprobedepth" => {
            let depth = value.parse::<i32>().ok().unwrap_or(1).clamp(1, 64);
            crate::syzygy::set_probe_depth(depth);
            eprintln!("info string SyzygyProbeDepth set to {}", depth);
        }
        "experiencefile" => {
            if !value.is_empty() {
                *exp_path = PathBuf::from(&value);
                *exp_table = ExpTable::new();
                match exp_table.load(exp_path) {
                    Ok(n) if n > 0 => eprintln!("info string loaded {} experience entries from {}", n, value),
                    Ok(_) => {}
                    Err(e) => eprintln!("info string could not load experience: {}", e),
                }
            }
        }
        "experience" => {
            *use_experience = value.eq_ignore_ascii_case("true");
            if !*use_experience {
                eprintln!("info string experience learning disabled");
            }
        }
        _ => {}
    }
}

fn parse_position(tokens: &[&str], board: &mut Board) {
    let mut idx = 1;
    if idx >= tokens.len() {
        return;
    }

    if tokens[idx] == "startpos" {
        *board = Board::start_pos();
        idx += 1;
    } else if tokens[idx] == "fen" {
        idx += 1;
        let mut fen_parts = Vec::new();
        while idx < tokens.len() && tokens[idx] != "moves" {
            fen_parts.push(tokens[idx]);
            idx += 1;
        }
        let fen = fen_parts.join(" ");
        if let Ok(b) = Board::from_fen(&fen) {
            *board = b;
        }
    }

    if idx < tokens.len() && tokens[idx] == "moves" {
        idx += 1;
        while idx < tokens.len() {
            if let Some(m) = parse_uci_move(board, tokens[idx]) {
                board.make_move(m);
            }
            idx += 1;
        }
    }
}

fn parse_uci_move(board: &Board, uci: &str) -> Option<Move> {
    if uci.len() < 4 {
        return None;
    }

    let from = parse_square(&uci[0..2])?;
    let to = parse_square(&uci[2..4])?;
    let promo = if uci.len() == 5 {
        match uci.as_bytes()[4] {
            b'n' => Some(Piece::Knight),
            b'b' => Some(Piece::Bishop),
            b'r' => Some(Piece::Rook),
            b'q' => Some(Piece::Queen),
            _ => None,
        }
    } else {
        None
    };

    let mut list = MoveList::new();
    movegen::generate_moves(board, &mut list);

    for i in 0..list.len() {
        let m = list.moves[i];
        if m.from_sq() == from && m.to_sq() == to {
            if let Some(p) = promo {
                if m.promotion_piece() == Some(p) {
                    return Some(m);
                }
            } else if !m.is_promotion() {
                return Some(m);
            }
        }
    }

    None
}

fn parse_go(tokens: &[&str], board: &Board) -> (u64, i32) {
    let mut time_ms: u64 = 0;
    let mut depth: i32 = 64;
    let mut movestogo: u64 = 0;
    let mut movetime: u64 = 0;
    let mut infinite = false;

    let mut wtime: u64 = 0;
    let mut btime: u64 = 0;
    let mut winc: u64 = 0;
    let mut binc: u64 = 0;
    let mut mate_search: u64 = 0;

    let mut i = 1;
    while i < tokens.len() {
        match tokens[i] {
            "depth" => {
                i += 1;
                if i < tokens.len() {
                    depth = tokens[i].parse().unwrap_or(64);
                }
            }
            "movetime" => {
                i += 1;
                if i < tokens.len() {
                    movetime = tokens[i].parse().unwrap_or(0);
                }
            }
            "wtime" => {
                i += 1;
                if i < tokens.len() {
                    wtime = tokens[i].parse().unwrap_or(0);
                }
            }
            "btime" => {
                i += 1;
                if i < tokens.len() {
                    btime = tokens[i].parse().unwrap_or(0);
                }
            }
            "winc" => {
                i += 1;
                if i < tokens.len() {
                    winc = tokens[i].parse().unwrap_or(0);
                }
            }
            "binc" => {
                i += 1;
                if i < tokens.len() {
                    binc = tokens[i].parse().unwrap_or(0);
                }
            }
            "movestogo" => {
                i += 1;
                if i < tokens.len() {
                    movestogo = tokens[i].parse().unwrap_or(0);
                }
            }
            "mate" => {
                i += 1;
                if i < tokens.len() {
                    mate_search = tokens[i].parse().unwrap_or(0);
                }
            }
            "infinite" => {
                infinite = true;
            }
            _ => {}
        }
        i += 1;
    }

    if mate_search > 0 {
        depth = (mate_search * 2 + 2) as i32;
        time_ms = 0;
        return (time_ms, depth);
    }

    if movetime > 0 {
        time_ms = movetime;
    } else if infinite {
        time_ms = 0;
    } else {
        let (our_time, our_inc) = match board.side {
            Color::White => (wtime, winc),
            Color::Black => (btime, binc),
        };

        if our_time > 0 {
            let piece_count = board.all_occupancy.count_ones() as u64;

            let estimated_moves = if movestogo > 0 {
                movestogo
            } else if piece_count > 24 {
                35
            } else if piece_count > 14 {
                25
            } else {
                20
            };

            let phase_factor: f64 = if piece_count > 26 {
                0.7
            } else if piece_count > 14 {
                1.3
            } else {
                0.9
            };

            let base_time = our_time / estimated_moves + our_inc * 3 / 4;

            time_ms = (base_time as f64 * phase_factor) as u64;

            time_ms = time_ms.min(our_time / 4);

            time_ms = time_ms.max(50);

            let safety = (our_time / 40).max(30).min(200);
            if time_ms > safety {
                time_ms -= safety;
            }
        } else if depth == 64 {
            time_ms = 5000;
        }
    }

    (time_ms, depth)
}

fn run_bench(tt: &mut TranspositionTable) {
    let positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    ];

    let depth = 7;
    let mut total_nodes: u64 = 0;
    let start = std::time::Instant::now();

    for fen in &positions {
        let mut board = Board::from_fen(fen).unwrap();
        tt.clear();
        let exp = learn::ExpTable::new();
        let result = search::search(&mut board, tt, &exp, 0, depth);
        total_nodes += result.nodes;
    }

    let elapsed = start.elapsed().as_millis() as u64;
    let nps = if elapsed > 0 { total_nodes * 1000 / elapsed } else { 0 };

    println!("===========================");
    println!("Total nodes: {}", total_nodes);
    println!("Total time:  {} ms", elapsed);
    println!("NPS:         {}", nps);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() {
        crate::zobrist::init();
        crate::movegen::init();
    }

    #[test]
    fn test_parse_uci_move() {
        setup();
        let board = Board::start_pos();
        let m = parse_uci_move(&board, "e2e4");
        assert!(m.is_some());
        let m = m.unwrap();
        assert_eq!(m.from_sq(), sq::E2);
        assert_eq!(m.to_sq(), sq::E4);
    }

    #[test]
    fn test_parse_position_startpos() {
        setup();
        let mut board = Board::empty();
        let tokens: Vec<&str> = "position startpos".split_whitespace().collect();
        parse_position(&tokens, &mut board);
        assert_eq!(board.to_fen(), crate::board::START_FEN);
    }

    #[test]
    fn test_parse_position_startpos_moves() {
        setup();
        let mut board = Board::empty();
        let tokens: Vec<&str> = "position startpos moves e2e4 e7e5".split_whitespace().collect();
        parse_position(&tokens, &mut board);
        assert_eq!(board.to_fen(), "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2");
    }
}
