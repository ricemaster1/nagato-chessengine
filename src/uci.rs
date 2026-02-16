/// UCI (Universal Chess Interface) protocol implementation.
/// This allows the engine to communicate with any UCI-compatible GUI.

use crate::bitboard::*;
use crate::board::Board;
use crate::eval;
use crate::movegen;
use crate::moves::*;
use crate::search::{self, TranspositionTable};
use std::io::{self, BufRead};

const ENGINE_NAME: &str = "Nagato";
const ENGINE_AUTHOR: &str = "Nagato Team";

pub fn uci_loop() {
    let stdin = io::stdin();
    let mut board = Board::start_pos();
    let mut tt = TranspositionTable::new(64); // 64 MB default

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
                println!("uciok");
            }
            "isready" => {
                println!("readyok");
            }
            "ucinewgame" => {
                board = Board::start_pos();
                tt.clear();
            }
            "position" => {
                parse_position(&tokens, &mut board);
            }
            "go" => {
                let (time_ms, depth) = parse_go(&tokens, &board);
                let result = search::search(&mut board, &mut tt, time_ms, depth);
                println!("bestmove {}", result.best_move);
            }
            "quit" => {
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
                let score = eval::evaluate(&board);
                println!("Eval: {} cp (from {}'s perspective)",
                    score,
                    match board.side { Color::White => "white", Color::Black => "black" }
                );
            }
            "bench" => {
                run_bench(&mut tt);
            }
            _ => {
                // Unknown command, ignore silently per UCI spec
            }
        }
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

    // Parse moves
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

    // Generate all legal moves and find the matching one
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

    // Handle "go mate N" — search for mate in N moves
    if mate_search > 0 {
        depth = (mate_search * 2 + 2) as i32; // Enough ply to find the mate
        time_ms = 0; // No time limit for mate search
        return (time_ms, depth);
    }

    if movetime > 0 {
        time_ms = movetime;
    } else if infinite {
        time_ms = 0; // No time limit, but depth is still capped
    } else {
        // Time management
        let (our_time, our_inc) = match board.side {
            Color::White => (wtime, winc),
            Color::Black => (btime, binc),
        };

        if our_time > 0 {
            // Estimate game phase from piece count
            let piece_count = board.all_occupancy.count_ones() as u64;

            // Estimate moves remaining based on game phase
            let estimated_moves = if movestogo > 0 {
                movestogo
            } else if piece_count > 24 {
                // Opening: many pieces, moves are more "bookish"
                35
            } else if piece_count > 14 {
                // Middlegame: complex positions need more time
                25
            } else {
                // Endgame: fewer pieces, search is faster
                20
            };

            // Phase-based time scaling factor (spend more in middlegame)
            let phase_factor: f64 = if piece_count > 26 {
                0.7  // Opening: play faster
            } else if piece_count > 14 {
                1.3  // Middlegame: think harder
            } else {
                0.9  // Endgame: search is deep but fast
            };

            // Base time allocation
            let base_time = our_time / estimated_moves + our_inc * 3 / 4;

            // Apply phase scaling
            time_ms = (base_time as f64 * phase_factor) as u64;

            // Hard limits: never use more than 1/4 of remaining time
            time_ms = time_ms.min(our_time / 4);

            // Minimum think time
            time_ms = time_ms.max(50);

            // Safety margin to avoid flagging
            let safety = (our_time / 40).max(30).min(200);
            if time_ms > safety {
                time_ms -= safety;
            }
        } else if depth == 64 {
            // No time control and no explicit depth — use a sensible default
            time_ms = 5000; // 5 seconds per move
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
        let result = search::search(&mut board, tt, 0, depth);
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
        // After 1.e4 e5
        assert_eq!(board.to_fen(), "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2");
    }
}
