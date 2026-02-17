/// Experience-based learning: Nagato remembers positions from past games.
///
/// After each game, notable positions (PV nodes, large eval swings) are persisted
/// to disk. In future games, the stored best move and score act as hints — improving
/// move ordering and providing a small eval correction when we've "been here before."

use crate::moves::Move;

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
