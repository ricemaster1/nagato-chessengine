/// Tablebase probing module.
///
/// Supports:
/// - Syzygy endgame tablebases (WDL + DTZ) via local files for 3-7 pieces
/// - Lomonosov 7-piece tablebases via online API (fallback for analysis)

use crate::board::Board;
use crate::eval::MATE_THRESHOLD;

use shakmaty::{CastlingMode, Chess, Position};
use shakmaty_syzygy::{AmbiguousWdl, Dtz, MaybeRounded, Tablebase as SyzygyTB};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

// ============================================================
// Syzygy Tablebase
// ============================================================

/// Global Syzygy tablebase instance
static SYZYGY: OnceLock<Mutex<SyzygyTB<Chess>>> = OnceLock::new();

/// Maximum number of pieces for Syzygy probing (typically 6 or 7)
static MAX_PIECES: OnceLock<u32> = OnceLock::new();

/// Whether Syzygy tablebases are available
static SYZYGY_ENABLED: OnceLock<bool> = OnceLock::new();

/// Tablebase probe result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TbResult {
    /// Win for the side to move (with DTZ distance if known)
    Win(i32),
    /// Draw
    Draw,
    /// Loss for the side to move (with DTZ distance if known)
    Loss(i32),
    /// Probe failed (position not in tablebase)
    Failed,
}

impl TbResult {
    /// Convert to a search score for use in alpha-beta
    pub fn to_score(self, ply: usize) -> Option<i32> {
        match self {
            TbResult::Win(dtz) => {
                // Score as mate-like but less than actual checkmate
                // Use MATE_THRESHOLD - 1 as base to distinguish from real mates
                let tb_win_score = MATE_THRESHOLD - ply as i32 - dtz.unsigned_abs() as i32;
                Some(tb_win_score.max(MATE_THRESHOLD - 500))
            }
            TbResult::Draw => Some(0),
            TbResult::Loss(dtz) => {
                let tb_loss_score = -(MATE_THRESHOLD - ply as i32 - dtz.unsigned_abs() as i32);
                Some(tb_loss_score.min(-(MATE_THRESHOLD - 500)))
            }
            TbResult::Failed => None,
        }
    }
}

/// Initialize the Syzygy tablebase with the given path(s).
/// Paths can be separated by `:` (Unix) or `;` (Windows).
pub fn init_syzygy(paths: &str) {
    let mut tb = SyzygyTB::new();
    let mut total = 0usize;

    for path in paths.split([':', ';']) {
        let path = path.trim();
        if path.is_empty() {
            continue;
        }
        if !Path::new(path).exists() {
            eprintln!("info string Syzygy path not found: {}", path);
            continue;
        }
        match tb.add_directory(path) {
            Ok(count) => {
                total += count;
                eprintln!("info string Loaded {} tablebases from {}", count, path);
            }
            Err(e) => {
                eprintln!("info string Error loading tablebases from {}: {}", path, e);
            }
        }
    }

    if total > 0 {
        // Determine max pieces from loaded tables
        let max = tb.max_pieces();
        eprintln!("info string Syzygy: {} tablebases loaded, up to {} pieces", total, max);
        let _ = MAX_PIECES.set(max as u32);
        let _ = SYZYGY_ENABLED.set(true);
    } else {
        eprintln!("info string No Syzygy tablebases found");
        let _ = MAX_PIECES.set(0);
        let _ = SYZYGY_ENABLED.set(false);
    }

    let _ = SYZYGY.set(Mutex::new(tb));
}

/// Check if Syzygy tablebases are available
pub fn syzygy_available() -> bool {
    *SYZYGY_ENABLED.get().unwrap_or(&false)
}

/// Get the maximum number of pieces supported by loaded tables
pub fn max_pieces() -> u32 {
    *MAX_PIECES.get().unwrap_or(&0)
}

/// Convert our Board to a shakmaty Chess position
fn board_to_shakmaty(board: &Board) -> Option<Chess> {
    use shakmaty::fen::Fen;

    // The easiest and most reliable way: convert to FEN and parse
    let fen_str = board.to_fen();
    let fen: Fen = fen_str.parse().ok()?;
    let pos: Chess = fen.into_position(CastlingMode::Standard).ok()?;
    Some(pos)
}

/// Probe Syzygy WDL (Win/Draw/Loss) for the given position.
/// Returns the WDL result from the perspective of the side to move.
pub fn probe_wdl(board: &Board) -> TbResult {
    if !syzygy_available() {
        return TbResult::Failed;
    }

    // Only probe positions with few enough pieces
    let piece_count = board.all_occupancy.count_ones();
    if piece_count > max_pieces() {
        return TbResult::Failed;
    }

    // Don't probe if there are castling rights (tablebases don't handle this)
    if board.castling != 0 {
        return TbResult::Failed;
    }

    let pos = match board_to_shakmaty(board) {
        Some(p) => p,
        None => return TbResult::Failed,
    };

    let tb = match SYZYGY.get() {
        Some(tb) => tb,
        None => return TbResult::Failed,
    };

    let tb = match tb.lock() {
        Ok(tb) => tb,
        Err(_) => return TbResult::Failed,
    };

    match tb.probe_wdl(&pos) {
        Ok(wdl) => match wdl {
            AmbiguousWdl::Win | AmbiguousWdl::MaybeWin => TbResult::Win(0),
            AmbiguousWdl::CursedWin => TbResult::Draw, // Treat cursed wins as draws (50-move rule)
            AmbiguousWdl::Draw => TbResult::Draw,
            AmbiguousWdl::BlessedLoss => TbResult::Draw, // Treat blessed losses as draws
            AmbiguousWdl::MaybeLoss | AmbiguousWdl::Loss => TbResult::Loss(0),
        },
        Err(_) => TbResult::Failed,
    }
}

/// Probe Syzygy DTZ (Distance To Zeroing move) for the given position.
/// This gives a more precise score for tablebase positions.
pub fn probe_dtz(board: &Board) -> TbResult {
    if !syzygy_available() {
        return TbResult::Failed;
    }

    let piece_count = board.all_occupancy.count_ones();
    if piece_count > max_pieces() {
        return TbResult::Failed;
    }

    if board.castling != 0 {
        return TbResult::Failed;
    }

    let pos = match board_to_shakmaty(board) {
        Some(p) => p,
        None => return TbResult::Failed,
    };

    let tb = match SYZYGY.get() {
        Some(tb) => tb,
        None => return TbResult::Failed,
    };

    let tb = match tb.lock() {
        Ok(tb) => tb,
        Err(_) => return TbResult::Failed,
    };

    match tb.probe_dtz(&pos) {
        Ok(MaybeRounded::Precise(dtz)) | Ok(MaybeRounded::Rounded(dtz)) => {
            let plies = dtz.0;
            if plies > 0 {
                TbResult::Win(plies)
            } else if plies < 0 {
                TbResult::Loss(plies)
            } else {
                TbResult::Draw
            }
        }
        Err(_) => TbResult::Failed,
    }
}

/// Probe the best move from Syzygy tablebases at the root.
/// Returns the best move UCI string and its WDL result.
pub fn probe_root(board: &Board) -> Option<(String, TbResult)> {
    if !syzygy_available() {
        return None;
    }

    let piece_count = board.all_occupancy.count_ones();
    if piece_count > max_pieces() {
        return None;
    }

    if board.castling != 0 {
        return None;
    }

    let pos = match board_to_shakmaty(board) {
        Some(p) => p,
        None => return None,
    };

    let tb = match SYZYGY.get() {
        Some(tb) => tb,
        None => return None,
    };

    let tb = match tb.lock() {
        Ok(tb) => tb,
        Err(_) => return None,
    };

    // Probe DTZ to find the best root move
    match tb.best_move(&pos) {
        Ok(Some((m, dtz))) => {
            let uci_move = format!("{}", m);
            let plies = dtz.ignore_rounding().0;
            let result = if plies > 0 {
                TbResult::Win(plies)
            } else if plies < 0 {
                TbResult::Loss(plies)
            } else {
                TbResult::Draw
            };
            Some((uci_move, result))
        }
        Ok(None) => None, // No legal moves (checkmate/stalemate)
        Err(_) => None,
    }
}

// ============================================================
// Lomonosov 7-piece online API
// ============================================================

/// Query the Lomonosov 7-piece tablebase via the online API.
/// This is slow (network round-trip) and only suitable for analysis mode.
/// Returns WDL result from the perspective of the side to move.
pub fn probe_lomonosov(board: &Board) -> TbResult {
    let piece_count = board.all_occupancy.count_ones();
    if piece_count > 7 || piece_count < 3 {
        return TbResult::Failed;
    }

    // Skip if Syzygy can already handle this many pieces
    if syzygy_available() && piece_count <= max_pieces() {
        return TbResult::Failed;
    }

    let fen = board.to_fen();

    // Lomonosov API endpoint
    let url = format!(
        "http://tb7.chessok.com/probe?fen={}",
        fen.replace(' ', "%20")
    );

    match ureq::get(&url).call() {
        Ok(response) => {
            let body = match response.into_body().read_to_string() {
                Ok(s) => s,
                Err(_) => return TbResult::Failed,
            };

            // Parse the response — Lomonosov returns text like:
            // "Win in X" / "Draw" / "Loss in X" / "Checkmate" / "Stalemate"
            let body_lower = body.to_lowercase();

            if body_lower.contains("checkmate") {
                // Current side is checkmated
                return TbResult::Loss(0);
            }
            if body_lower.contains("stalemate") {
                return TbResult::Draw;
            }
            if body_lower.contains("draw") {
                return TbResult::Draw;
            }

            // Try to extract "win in N" or "loss in N"
            if body_lower.contains("win") {
                let dtz = extract_number(&body_lower).unwrap_or(1);
                return TbResult::Win(dtz);
            }
            if body_lower.contains("loss") || body_lower.contains("lose") {
                let dtz = extract_number(&body_lower).unwrap_or(1);
                return TbResult::Loss(-dtz);
            }

            TbResult::Failed
        }
        Err(_) => TbResult::Failed,
    }
}

/// Extract the first number from a string
fn extract_number(s: &str) -> Option<i32> {
    let mut num_str = String::new();
    let mut found = false;
    for ch in s.chars() {
        if ch.is_ascii_digit() {
            num_str.push(ch);
            found = true;
        } else if found {
            break;
        }
    }
    if found {
        num_str.parse().ok()
    } else {
        None
    }
}

/// Combined probe: try Syzygy first, then fall back to Lomonosov for 7-piece positions.
/// `use_lomonosov` should only be true in analysis mode (not during timed games).
pub fn probe_position(board: &Board, use_lomonosov: bool) -> TbResult {
    // Try Syzygy first (fast, local)
    let result = probe_wdl(board);
    if result != TbResult::Failed {
        return result;
    }

    // Fall back to Lomonosov for 7-piece (slow, online)
    if use_lomonosov {
        return probe_lomonosov(board);
    }

    TbResult::Failed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_board_to_shakmaty() {
        crate::zobrist::init();
        crate::movegen::init();

        // Test that our FEN round-trips through shakmaty correctly
        let board = Board::start_pos();
        let sm_pos = board_to_shakmaty(&board);
        assert!(sm_pos.is_some(), "Should convert starting position");
    }

    #[test]
    fn test_board_to_shakmaty_endgame() {
        crate::zobrist::init();
        crate::movegen::init();

        // KQ vs K endgame
        let board = Board::from_fen("8/8/3k4/8/8/4K3/8/Q7 w - - 0 1").unwrap();
        let sm_pos = board_to_shakmaty(&board);
        assert!(sm_pos.is_some(), "Should convert KQ vs K position");
    }

    #[test]
    fn test_tb_result_to_score() {
        // Win should give a large positive score
        let win = TbResult::Win(10);
        let score = win.to_score(0).unwrap();
        assert!(score > 0, "Win should be positive");
        assert!(score > 20000, "Win should be a high score");

        // Loss should give a large negative score
        let loss = TbResult::Loss(-10);
        let score = loss.to_score(0).unwrap();
        assert!(score < 0, "Loss should be negative");

        // Draw should be 0
        let draw = TbResult::Draw;
        let score = draw.to_score(0).unwrap();
        assert_eq!(score, 0, "Draw should be 0");

        // Failed should return None
        let failed = TbResult::Failed;
        assert!(failed.to_score(0).is_none(), "Failed should be None");
    }

    #[test]
    fn test_probe_wdl_no_tables() {
        // Without tables loaded, probe should return Failed
        let board = Board::from_fen("8/8/3k4/8/8/4K3/8/Q7 w - - 0 1").unwrap();
        let result = probe_wdl(&board);
        assert_eq!(result, TbResult::Failed);
    }

    #[test]
    fn test_extract_number() {
        assert_eq!(extract_number("win in 42 moves"), Some(42));
        assert_eq!(extract_number("loss in 7"), Some(7));
        assert_eq!(extract_number("draw"), None);
        assert_eq!(extract_number("checkmate"), None);
    }
}
