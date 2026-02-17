/// Experience-based learning: Nagato remembers positions from past games.
///
/// After each game, notable positions (PV nodes, large eval swings) are persisted
/// to disk. In future games, the stored best move and score act as hints — improving
/// move ordering and providing a small eval correction when we've "been here before."

use crate::moves::Move;

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

// ============================================================
// Game result (from engine's perspective)
// ============================================================

/// Outcome of a finished game, from the engine's perspective.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GameResult {
    Win,
    Draw,
    Loss,
}

// ============================================================
// Experience entry
// ============================================================

/// A single piece of experience: what we learned about a position.
#[derive(Clone, Copy, Debug)]
pub struct ExpEntry {
    /// Zobrist hash of the position
    pub hash: u64,
    /// The best move we found last time
    pub best_move: Move,
    /// Search depth at which we found it
    pub depth: i8,
    /// The engine's evaluation (centipawns, from side-to-move)
    pub score: i16,
    /// How the game actually ended: 1.0 = win, 0.5 = draw, 0.0 = loss
    /// (from the perspective of the side to move in this position)
    pub game_result: f32,
    /// How many times we've seen this position across games
    pub count: u16,
}

// ============================================================
// Experience table
// ============================================================

/// Fixed number of buckets — kept small so the file stays portable.
/// Each bucket holds one entry (direct-mapped). Collisions are resolved
/// by keeping the higher-depth entry.
const EXP_TABLE_SIZE: usize = 1 << 16; // 65536 entries

/// The experience table: a position cache that persists across games.
pub struct ExpTable {
    entries: Vec<Option<ExpEntry>>,
}

impl ExpTable {
    pub fn new() -> Self {
        ExpTable {
            entries: vec![None; EXP_TABLE_SIZE],
        }
    }

    #[inline]
    fn index(hash: u64) -> usize {
        (hash as usize) % EXP_TABLE_SIZE
    }

    /// Look up a position's experience. Returns None if we've never seen it.
    pub fn probe(&self, hash: u64) -> Option<&ExpEntry> {
        self.entries[Self::index(hash)]
            .as_ref()
            .filter(|e| e.hash == hash)
    }

    /// Record or update experience for a position.
    /// If we've seen it before, we keep the deeper entry and increment the count.
    pub fn store(&mut self, entry: ExpEntry) {
        let idx = Self::index(entry.hash);
        if let Some(existing) = &mut self.entries[idx] {
            if existing.hash == entry.hash {
                // Same position — merge: keep deeper search, accumulate count
                existing.count = existing.count.saturating_add(1);
                // Blend game result as a running average
                let n = existing.count as f32;
                existing.game_result =
                    existing.game_result * ((n - 1.0) / n) + entry.game_result * (1.0 / n);
                // Update move/score only if the new entry searched deeper
                if entry.depth >= existing.depth {
                    existing.best_move = entry.best_move;
                    existing.depth = entry.depth;
                    existing.score = entry.score;
                }
            } else if entry.depth >= existing.depth {
                // Collision with a different position — replace if deeper
                self.entries[idx] = Some(entry);
            }
        } else {
            self.entries[idx] = Some(entry);
        }
    }

    /// Number of occupied slots (for diagnostics).
    pub fn len(&self) -> usize {
        self.entries.iter().filter(|e| e.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ============================================================
// Binary file persistence
// ============================================================
//
// File format (nagato.exp):
//   Magic:   4 bytes "NEXP"
//   Version: 1 byte
//   Count:   4 bytes (u32 LE) — number of entries
//   Entries: count × 24 bytes each:
//     hash:        8 bytes (u64 LE)
//     best_move:   4 bytes (u32 LE — Move raw bits)
//     depth:       1 byte  (i8)
//     score:       2 bytes (i16 LE)
//     game_result: 4 bytes (f32 LE)
//     count:       2 bytes (u16 LE)
//     _padding:    3 bytes (reserved)

const EXP_MAGIC: &[u8; 4] = b"NEXP";
const EXP_VERSION: u8 = 1;
const ENTRY_BYTES: usize = 24;

impl ExpTable {
    /// Save all non-empty entries to a binary file.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        // Header
        w.write_all(EXP_MAGIC)?;
        w.write_all(&[EXP_VERSION])?;

        let occupied: Vec<&ExpEntry> = self.entries.iter().filter_map(|e| e.as_ref()).collect();
        w.write_all(&(occupied.len() as u32).to_le_bytes())?;

        // Entries
        for e in &occupied {
            w.write_all(&e.hash.to_le_bytes())?;
            w.write_all(&e.best_move.0.to_le_bytes())?;
            w.write_all(&[e.depth as u8])?;
            w.write_all(&e.score.to_le_bytes())?;
            w.write_all(&e.game_result.to_le_bytes())?;
            w.write_all(&e.count.to_le_bytes())?;
            w.write_all(&[0u8; 3])?; // padding
        }

        w.flush()?;
        Ok(())
    }

    /// Load entries from a binary file, merging into the current table.
    pub fn load(&mut self, path: &Path) -> io::Result<usize> {
        if !path.exists() {
            return Ok(0);
        }

        let file = File::open(path)?;
        let mut r = BufReader::new(file);

        // Check magic
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != EXP_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not an experience file"));
        }

        // Check version
        let mut ver = [0u8; 1];
        r.read_exact(&mut ver)?;
        if ver[0] != EXP_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported experience file version {}", ver[0]),
            ));
        }

        // Entry count
        let mut count_buf = [0u8; 4];
        r.read_exact(&mut count_buf)?;
        let count = u32::from_le_bytes(count_buf) as usize;

        let mut loaded = 0;
        let mut buf = [0u8; ENTRY_BYTES];

        for _ in 0..count {
            r.read_exact(&mut buf)?;

            let hash = u64::from_le_bytes(buf[0..8].try_into().unwrap());
            let move_bits = u32::from_le_bytes(buf[8..12].try_into().unwrap());
            let depth = buf[12] as i8;
            let score = i16::from_le_bytes(buf[13..15].try_into().unwrap());
            let game_result = f32::from_le_bytes(buf[15..19].try_into().unwrap());
            let entry_count = u16::from_le_bytes(buf[19..21].try_into().unwrap());
            // buf[21..24] is padding

            self.store(ExpEntry {
                hash,
                best_move: Move(move_bits),
                depth,
                score,
                game_result,
                count: entry_count,
            });
            loaded += 1;
        }

        Ok(loaded)
    }
}

// ============================================================
// Game recorder
// ============================================================

/// A snapshot of one position during a game.
/// We don't know the game result yet — that gets filled in at the end.
#[derive(Clone, Copy)]
struct PositionRecord {
    hash: u64,
    best_move: Move,
    depth: i8,
    score: i16,
    /// Which color was to move (0 = White, 1 = Black)
    side: u8,
}

/// Records notable positions as the engine plays a game.
/// When the game ends, we learn the outcome and flush everything
/// into the experience table.
pub struct GameRecorder {
    positions: Vec<PositionRecord>,
    /// Which color Nagato is playing (set when the first "go" arrives)
    our_color: Option<u8>,
}

impl GameRecorder {
    pub fn new() -> Self {
        GameRecorder {
            positions: Vec::with_capacity(128),
            our_color: None,
        }
    }

    /// Call this when the engine starts thinking.
    /// `side` is the color to move (0 = White, 1 = Black).
    pub fn set_our_color(&mut self, side: u8) {
        if self.our_color.is_none() {
            self.our_color = Some(side);
        }
    }

    /// Record a position after the engine finishes searching it.
    pub fn record(&mut self, hash: u64, best_move: Move, depth: i8, score: i16, side: u8) {
        // Only record positions searched to a reasonable depth
        if depth < 4 {
            return;
        }
        self.positions.push(PositionRecord {
            hash,
            best_move,
            depth,
            score,
            side,
        });
    }

    /// Flush all recorded positions into the experience table
    /// now that we know how the game ended.
    pub fn flush(&mut self, table: &mut ExpTable, result: GameResult) {
        let our_color = match self.our_color {
            Some(c) => c,
            None => {
                self.positions.clear();
                return;
            }
        };

        for pos in &self.positions {
            // Game result from the perspective of the side to move in this position
            let result_for_side = if pos.side == our_color {
                match result {
                    GameResult::Win => 1.0f32,
                    GameResult::Draw => 0.5,
                    GameResult::Loss => 0.0,
                }
            } else {
                // Opponent was to move — flip the result
                match result {
                    GameResult::Win => 0.0f32,
                    GameResult::Draw => 0.5,
                    GameResult::Loss => 1.0,
                }
            };

            table.store(ExpEntry {
                hash: pos.hash,
                best_move: pos.best_move,
                depth: pos.depth,
                score: pos.score,
                game_result: result_for_side,
                count: 1,
            });
        }

        self.positions.clear();
        self.our_color = None;
    }

    /// Discard recorded positions without learning (e.g. game aborted).
    pub fn clear(&mut self) {
        self.positions.clear();
        self.our_color = None;
    }

    /// How many positions have been recorded this game.
    pub fn recorded_count(&self) -> usize {
        self.positions.len()
    }
}
