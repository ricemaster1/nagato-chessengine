/// Data generation for NNUE training.
///
/// Runs self-play games using the current HCE evaluation, recording
/// (position, search_score, game_result) tuples to a binary file.
///
/// Binary format per entry (32 bytes):
///   - FEN-compressed position: 24 bytes (custom packed format)
///   - search score: i16 (2 bytes) — centipawns from STM perspective
///   - game result: i8 (1 byte) — 1=white win, 0=draw, -1=black win
///   - padding: 5 bytes
///
/// We use a simpler "packed board" representation:
///   - 32 bytes per piece-square encoding (4 bits per square, 64 squares = 32 bytes)
///     But that's complex. Instead we use a compact text format.
///
/// Actually, for simplicity and portability, we'll write to a simple binary format:
///   - Each entry is exactly 40 bytes:
///     - 32 bytes: piece list encoding (piece_on_sq[64] as 4-bit nibbles = 32 bytes)
///       Encoding: 0=empty, 1=WP,2=WN,3=WB,4=WR,5=WQ,6=WK,7=BP,8=BN,9=BB,10=BR,11=BQ,12=BK
///     - 1 byte: side to move (0=white, 1=black)
///     - 1 byte: castling rights
///     - 1 byte: en passant file (0-7, or 255 for none)
///     - 1 byte: padding
///     - 2 bytes: score (i16, centipawns from white's perspective)
///     - 2 bytes: game result + wdl encoding:
///       - result: i8 (1=white win, 0=draw, -1=black win)
///       - padding: 1 byte

use crate::bitboard::*;
use crate::board::Board;
use crate::eval;
use crate::learn::ExpTable;
use crate::movegen;
use crate::moves::*;
use crate::search;
use std::io::Write;

/// Size of each training entry in bytes
pub const ENTRY_SIZE: usize = 40;

/// Pack a board position into the 32-byte piece-list format.
fn pack_board(board: &Board) -> [u8; 32] {
    let mut packed = [0u8; 32];
    for sq in 0..64u8 {
        let nibble = match board.piece_at(sq) {
            None => 0u8,
            Some((piece, color)) => {
                let base = match piece {
                    Piece::Pawn => 1,
                    Piece::Knight => 2,
                    Piece::Bishop => 3,
                    Piece::Rook => 4,
                    Piece::Queen => 5,
                    Piece::King => 6,
                };
                match color {
                    Color::White => base,
                    Color::Black => base + 6,
                }
            }
        };
        let byte_idx = sq as usize / 2;
        if sq % 2 == 0 {
            packed[byte_idx] |= nibble; // low nibble
        } else {
            packed[byte_idx] |= nibble << 4; // high nibble
        }
    }
    packed
}

/// Unpack a 32-byte piece list back to a Board (for verification).
pub fn unpack_board(packed: &[u8; 32], side: Color, castling: u8, ep_file: u8) -> Board {
    let mut board = Board::empty();
    for sq in 0..64u8 {
        let byte_idx = sq as usize / 2;
        let nibble = if sq % 2 == 0 {
            packed[byte_idx] & 0x0F
        } else {
            (packed[byte_idx] >> 4) & 0x0F
        };
        if nibble == 0 {
            continue;
        }
        let (piece, color) = match nibble {
            1 => (Piece::Pawn, Color::White),
            2 => (Piece::Knight, Color::White),
            3 => (Piece::Bishop, Color::White),
            4 => (Piece::Rook, Color::White),
            5 => (Piece::Queen, Color::White),
            6 => (Piece::King, Color::White),
            7 => (Piece::Pawn, Color::Black),
            8 => (Piece::Knight, Color::Black),
            9 => (Piece::Bishop, Color::Black),
            10 => (Piece::Rook, Color::Black),
            11 => (Piece::Queen, Color::Black),
            12 => (Piece::King, Color::Black),
            _ => continue,
        };
        board.put_piece(piece, color, sq);
    }
    board.side = side;
    board.castling = castling;
    if ep_file < 8 {
        let ep_rank = match side {
            Color::White => 5u8, // ep target is on rank 6 (index 5)
            Color::Black => 2u8, // ep target is on rank 3 (index 2)
        };
        board.ep_square = Some(make_square(ep_file, ep_rank));
    }
    board.hash = board.compute_hash();
    board
}

/// Write a single training entry to the output buffer.
fn write_entry(buf: &mut Vec<u8>, board: &Board, score_white: i16, result: i8) {
    let packed = pack_board(board);
    buf.extend_from_slice(&packed);  // 32 bytes
    buf.push(board.side as u8);       // 1 byte
    buf.push(board.castling);         // 1 byte
    buf.push(board.ep_square.map_or(255, |sq| file_of(sq))); // 1 byte
    buf.push(0);                      // padding
    buf.extend_from_slice(&score_white.to_le_bytes()); // 2 bytes
    buf.push(result as u8);           // 1 byte
    buf.push(0);                      // padding
    debug_assert_eq!(buf.len() % ENTRY_SIZE, 0);
}

/// Run data generation: self-play games with the HCE, recording positions.
///
/// Parameters:
/// - `num_games`: number of self-play games to generate
/// - `depth`: search depth for scoring positions
/// - `output_path`: path to write the binary training data
/// - `random_plies`: number of random opening plies (for position diversity)
pub fn generate(num_games: u32, depth: i32, output_path: &str, random_plies: u32) {
    use rand::Rng;

    let mut tt = search::TranspositionTable::new(32); // Small TT for datagen
    let exp = ExpTable::new(); // No experience for datagen
    let mut rng = rand::thread_rng();
    let mut buf: Vec<u8> = Vec::with_capacity(1024 * 1024);
    let mut total_positions = 0u64;
    let mut total_games = 0u32;

    let start_time = std::time::Instant::now();

    eprintln!("info string datagen: {} games, depth {}, random_plies {}, output {}",
        num_games, depth, random_plies, output_path);

    for game_idx in 0..num_games {
        let mut board = Board::start_pos();
        let mut positions: Vec<(Board, i16)> = Vec::new();
        let mut ply = 0u32;

        // Play random opening moves for diversity
        for _ in 0..random_plies {
            let mut list = MoveList::new();
            movegen::generate_moves(&board, &mut list);
            if list.len() == 0 {
                break;
            }

            // Try random legal moves
            let mut made = false;
            for _ in 0..10 {
                let idx = rng.gen_range(0..list.len());
                let m = list.moves[idx];
                if board.make_move(m) {
                    ply += 1;
                    made = true;
                    break;
                }
            }
            if !made {
                break;
            }
        }

        // Play the actual game with search
        let mut result: i8 = 0; // 0 = draw
        let mut consecutive_no_progress = 0u32;

        loop {
            // Check for game-ending conditions
            let mut list = MoveList::new();
            movegen::generate_moves(&board, &mut list);

            // Check if any move is legal
            let mut has_legal = false;
            let mut temp = board.clone();
            for i in 0..list.len() {
                if temp.make_move(list.moves[i]) {
                    temp.unmake_move(list.moves[i]);
                    has_legal = true;
                    break;
                }
                temp = board.clone();
            }

            if !has_legal {
                if board.in_check() {
                    // Checkmate
                    result = match board.side {
                        Color::White => -1, // Black wins
                        Color::Black => 1,  // White wins
                    };
                }
                // else stalemate = draw (result stays 0)
                break;
            }

            // 50-move rule
            if board.halfmove >= 100 {
                result = 0;
                break;
            }

            // Adjudication: too many moves
            if ply > 400 {
                result = 0;
                break;
            }

            // Search the position
            tt.clear();
            let search_result = search::search(&mut board, &mut tt, &exp, 0, depth);
            let score = search_result.score;

            // Adjudicate large scores as wins
            if score.abs() > eval::MATE_THRESHOLD {
                if score > 0 {
                    result = match board.side {
                        Color::White => 1,
                        Color::Black => -1,
                    };
                } else {
                    result = match board.side {
                        Color::White => -1,
                        Color::Black => 1,
                    };
                }
                break;
            }

            // Adjudicate large eval advantages after enough plies
            if ply > 40 && score.abs() > 1000 {
                consecutive_no_progress += 1;
                if consecutive_no_progress > 5 {
                    if score > 0 {
                        result = match board.side {
                            Color::White => 1,
                            Color::Black => -1,
                        };
                    } else {
                        result = match board.side {
                            Color::White => -1,
                            Color::Black => 1,
                        };
                    }
                    break;
                }
            } else {
                consecutive_no_progress = 0;
            }

            // Record the position if not in check and not a capture
            // (quiet positions are better for training)
            if !board.in_check() && score.abs() < 5000 {
                // Convert score to white's perspective
                let score_white = match board.side {
                    Color::White => score as i16,
                    Color::Black => -score as i16,
                };
                positions.push((board.clone(), score_white));
            }

            // Make the best move
            let best_move = search_result.best_move;
            if best_move == MOVE_NONE {
                break;
            }
            board.make_move(best_move);
            ply += 1;
        }

        // Write all positions from this game with the game result
        for (pos, score) in &positions {
            write_entry(&mut buf, pos, *score, result);
            total_positions += 1;
        }

        total_games += 1;

        // Progress update every 10 games
        if (game_idx + 1) % 10 == 0 {
            let elapsed = start_time.elapsed().as_secs();
            let games_per_sec = if elapsed > 0 { total_games as f64 / elapsed as f64 } else { 0.0 };
            eprintln!(
                "info string datagen: {}/{} games, {} positions, {:.1} games/sec",
                total_games, num_games, total_positions, games_per_sec
            );
        }

        // Flush to disk periodically (every 100 games)
        if total_games % 100 == 0 && !buf.is_empty() {
            flush_buf(&mut buf, output_path, total_games == 100);
        }
    }

    // Final flush
    if !buf.is_empty() {
        flush_buf(&mut buf, output_path, total_games <= 100);
    }

    let elapsed = start_time.elapsed().as_secs_f64();
    eprintln!(
        "info string datagen complete: {} games, {} positions, {:.1}s ({:.1} games/sec, {:.0} pos/sec)",
        total_games, total_positions, elapsed,
        total_games as f64 / elapsed,
        total_positions as f64 / elapsed
    );
}

fn flush_buf(buf: &mut Vec<u8>, path: &str, create: bool) {
    use std::fs::OpenOptions;
    let file = if create {
        std::fs::File::create(path)
    } else {
        OpenOptions::new().append(true).open(path)
    };
    match file {
        Ok(mut f) => {
            if let Err(e) = f.write_all(buf) {
                eprintln!("info string datagen write error: {}", e);
            }
        }
        Err(e) => {
            eprintln!("info string datagen file error: {}", e);
        }
    }
    buf.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() {
        crate::zobrist::init();
        crate::movegen::init();
    }

    #[test]
    fn test_pack_roundtrip_startpos() {
        setup();
        let board = Board::start_pos();
        let packed = pack_board(&board);
        let restored = unpack_board(&packed, board.side, board.castling,
            board.ep_square.map_or(255, |sq| file_of(sq)));

        // Verify all pieces match
        for sq in 0..64u8 {
            assert_eq!(board.piece_at(sq), restored.piece_at(sq),
                "Mismatch at square {}", SQUARE_NAMES[sq as usize]);
        }
    }

    #[test]
    fn test_pack_empty_board() {
        setup();
        let board = Board::empty();
        let packed = pack_board(&board);
        assert!(packed.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_write_entry_size() {
        setup();
        let board = Board::start_pos();
        let mut buf = Vec::new();
        write_entry(&mut buf, &board, 15, 1);
        assert_eq!(buf.len(), ENTRY_SIZE);
    }

    #[test]
    fn test_pack_roundtrip_mid_game() {
        setup();
        let board = Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
        let packed = pack_board(&board);
        let restored = unpack_board(&packed, board.side, board.castling,
            board.ep_square.map_or(255, |sq| file_of(sq)));

        for sq in 0..64u8 {
            assert_eq!(board.piece_at(sq), restored.piece_at(sq),
                "Mismatch at square {}", SQUARE_NAMES[sq as usize]);
        }
    }
}
