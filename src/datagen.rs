
use crate::bitboard::*;
use crate::board::Board;
use crate::eval;
use crate::learn::ExpTable;
use crate::movegen;
use crate::moves::*;
use crate::search;
use std::io::Write;
pub const ENTRY_SIZE: usize = 40;
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
            packed[byte_idx] |= nibble;
        } else {
            packed[byte_idx] |= nibble << 4;
        }
    }
    packed
}
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
            Color::White => 5u8,
            Color::Black => 2u8,
        };
        board.ep_square = Some(make_square(ep_file, ep_rank));
    }
    board.hash = board.compute_hash();
    board
}
fn write_entry(buf: &mut Vec<u8>, board: &Board, score_white: i16, wdl: u8) {
    let packed = pack_board(board);
    buf.extend_from_slice(&packed);
    buf.push(board.side as u8);
    buf.push(board.castling);
    buf.push(board.ep_square.map_or(255, |sq| file_of(sq)));
    buf.push(0);
    buf.extend_from_slice(&score_white.to_le_bytes());
    buf.push(wdl);
    buf.push(0);
    debug_assert_eq!(buf.len() % ENTRY_SIZE, 0);
}
pub fn write_entry_pub(buf: &mut Vec<u8>, board: &Board, score_white: i16, wdl: u8) {
    write_entry(buf, board, score_white, wdl);
}
pub fn generate(num_games: u32, depth: i32, output_path: &str, random_plies: u32) {
    use rand::Rng;

    const OPENING_FENS: &[&str] = &[
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
        "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
        "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
        "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
        "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1",
        "rnbqkbnr/pppp1ppp/4p3/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
        "rnbqkb1r/pppppp1p/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq - 1 3",
        "rnbqkb1r/p1pppppp/1p6/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
    ];

    let mut tt = search::TranspositionTable::new(32);
    let exp = ExpTable::new();
    let mut rng = rand::thread_rng();
    let mut buf: Vec<u8> = Vec::with_capacity(1024 * 1024);
    let mut total_positions = 0u64;
    let mut total_games = 0u32;

    let start_time = std::time::Instant::now();

    eprintln!("info string datagen: {} games, depth {}, random_plies {}, output {}",
        num_games, depth, random_plies, output_path);

    for game_idx in 0..num_games {
        let fen_idx = rng.gen_range(0..OPENING_FENS.len());
        let mut board = Board::from_fen(OPENING_FENS[fen_idx]).unwrap_or_else(|_| Board::start_pos());
        let mut positions: Vec<(Board, i16)> = Vec::new();
        let mut ply = 0u32;
        for _ in 0..random_plies {
            let mut list = MoveList::new();
            movegen::generate_moves(&board, &mut list);
            if list.len() == 0 {
                break;
            }
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
        let mut wdl: u8 = 1;
        let mut win_adj = 0u32;
        let mut draw_adj = 0u32;

        loop {
            let mut list = MoveList::new();
            movegen::generate_moves(&board, &mut list);
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
                    wdl = if board.side == Color::White { 0 } else { 2 };
                }
                break;
            }
            if board.halfmove >= 100 || ply > 400 {
                break;
            }
            tt.clear();
            let search_result = search::search(&mut board, &mut tt, &exp, 0, depth);
            let score = search_result.score;
            if score.abs() > eval::MATE_THRESHOLD {
                wdl = if (score > 0) == (board.side == Color::White) { 2 } else { 0 };
                break;
            }
            if score.abs() > 1000 {
                win_adj += 1;
                draw_adj = 0;
                if win_adj >= 3 {
                    wdl = if (score > 0) == (board.side == Color::White) { 2 } else { 0 };
                    break;
                }
            } else if score.abs() < 10 {
                draw_adj += 1;
                win_adj = 0;
                if draw_adj >= 8 && ply > 60 {
                    break;
                }
            } else {
                win_adj = 0;
                draw_adj = 0;
            }
            if !board.in_check() && score.abs() < 5000 {
                let score_white = match board.side {
                    Color::White => score as i16,
                    Color::Black => -score as i16,
                };
                positions.push((board.clone(), score_white));
            }
            let best_move = search_result.best_move;
            if best_move == MOVE_NONE {
                break;
            }
            board.make_move(best_move);
            ply += 1;
        }
        for (pos, score) in &positions {
            write_entry(&mut buf, pos, *score, wdl);
            total_positions += 1;
        }

        total_games += 1;
        if (game_idx + 1) % 10 == 0 {
            let elapsed = start_time.elapsed().as_secs();
            let games_per_sec = if elapsed > 0 { total_games as f64 / elapsed as f64 } else { 0.0 };
            eprintln!(
                "info string datagen: {}/{} games, {} positions, {:.1} games/sec",
                total_games, num_games, total_positions, games_per_sec
            );
        }
        if total_games % 100 == 0 && !buf.is_empty() {
            flush_buf(&mut buf, output_path, total_games == 100);
        }
    }
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

    #[test]
    fn test_read_lichess_pipeline_output() {
        setup();
        let path = "lichess_test_data.bin";
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping: {} not found (run pipeline.py first)", path);
            return;
        }
        let data = std::fs::read(path).unwrap();
        assert_eq!(data.len() % ENTRY_SIZE, 0, "File size not multiple of {}", ENTRY_SIZE);
        let n = data.len() / ENTRY_SIZE;
        assert!(n > 0, "Empty file");
        for i in 0..n {
            let off = i * ENTRY_SIZE;
            let entry = &data[off..off + ENTRY_SIZE];
            let packed: [u8; 32] = entry[0..32].try_into().unwrap();
            let side_byte = entry[32];
            let castling = entry[33];
            let ep_file = entry[34];
            let score = i16::from_le_bytes([entry[36], entry[37]]);
            let wdl = entry[38];

            assert!(side_byte <= 1, "entry {}: bad side {}", i, side_byte);
            assert!(wdl <= 2, "entry {}: bad wdl {}", i, wdl);
            assert!(score.abs() <= 30000, "entry {}: score {} out of range", i, score);

            let side = if side_byte == 0 { Color::White } else { Color::Black };
            let board = unpack_board(&packed, side, castling, ep_file);
            // Verify board has exactly 1 king per side
            assert_ne!(board.king_sq(Color::White), 64, "entry {}: no white king", i);
            assert_ne!(board.king_sq(Color::Black), 64, "entry {}: no black king", i);
        }
        eprintln!("Verified {} entries from {}", n, path);
    }
}
