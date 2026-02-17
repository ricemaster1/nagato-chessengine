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
