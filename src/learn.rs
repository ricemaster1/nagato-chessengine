
use crate::moves::Move;

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GameResult {
    Win,
    Draw,
    Loss,
}

#[derive(Clone, Copy, Debug)]
pub struct ExpEntry {
    pub hash: u64,
    pub best_move: Move,
    pub depth: i8,
    pub score: i16,
    pub game_result: f32,
    pub count: u16,
}

const EXP_TABLE_SIZE: usize = 1 << 16;

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

    pub fn probe(&self, hash: u64) -> Option<&ExpEntry> {
        self.entries[Self::index(hash)]
            .as_ref()
            .filter(|e| e.hash == hash)
    }

    pub fn store(&mut self, entry: ExpEntry) {
        let idx = Self::index(entry.hash);
        if let Some(existing) = &mut self.entries[idx] {
            if existing.hash == entry.hash {
                existing.count = existing.count.saturating_add(1);
                let n = existing.count as f32;
                existing.game_result =
                    existing.game_result * ((n - 1.0) / n) + entry.game_result * (1.0 / n);
                if entry.depth >= existing.depth {
                    existing.best_move = entry.best_move;
                    existing.depth = entry.depth;
                    existing.score = entry.score;
                }
            } else if entry.depth >= existing.depth {
                self.entries[idx] = Some(entry);
            }
        } else {
            self.entries[idx] = Some(entry);
        }
    }

    pub fn len(&self) -> usize {
        self.entries.iter().filter(|e| e.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

const EXP_MAGIC: &[u8; 4] = b"NEXP";
const EXP_VERSION: u8 = 1;
const ENTRY_BYTES: usize = 24;

impl ExpTable {
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        w.write_all(EXP_MAGIC)?;
        w.write_all(&[EXP_VERSION])?;

        let occupied: Vec<&ExpEntry> = self.entries.iter().filter_map(|e| e.as_ref()).collect();
        w.write_all(&(occupied.len() as u32).to_le_bytes())?;

        for e in &occupied {
            w.write_all(&e.hash.to_le_bytes())?;
            w.write_all(&e.best_move.0.to_le_bytes())?;
            w.write_all(&[e.depth as u8])?;
            w.write_all(&e.score.to_le_bytes())?;
            w.write_all(&e.game_result.to_le_bytes())?;
            w.write_all(&e.count.to_le_bytes())?;
            w.write_all(&[0u8; 3])?;
        }

        w.flush()?;
        Ok(())
    }

    pub fn load(&mut self, path: &Path) -> io::Result<usize> {
        if !path.exists() {
            return Ok(0);
        }

        let file = File::open(path)?;
        let mut r = BufReader::new(file);

        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != EXP_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not an experience file"));
        }

        let mut ver = [0u8; 1];
        r.read_exact(&mut ver)?;
        if ver[0] != EXP_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported experience file version {}", ver[0]),
            ));
        }

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

#[derive(Clone, Copy)]
struct PositionRecord {
    hash: u64,
    best_move: Move,
    depth: i8,
    score: i16,
    side: u8,
}

pub struct GameRecorder {
    positions: Vec<PositionRecord>,
    our_color: Option<u8>,
}

impl GameRecorder {
    pub fn new() -> Self {
        GameRecorder {
            positions: Vec::with_capacity(128),
            our_color: None,
        }
    }

    pub fn set_our_color(&mut self, side: u8) {
        if self.our_color.is_none() {
            self.our_color = Some(side);
        }
    }

    pub fn record(&mut self, hash: u64, best_move: Move, depth: i8, score: i16, side: u8) {
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

    pub fn flush(&mut self, table: &mut ExpTable, result: GameResult) {
        let our_color = match self.our_color {
            Some(c) => c,
            None => {
                self.positions.clear();
                return;
            }
        };

        for pos in &self.positions {
            let result_for_side = if pos.side == our_color {
                match result {
                    GameResult::Win => 1.0f32,
                    GameResult::Draw => 0.5,
                    GameResult::Loss => 0.0,
                }
            } else {
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

    pub fn clear(&mut self) {
        self.positions.clear();
        self.our_color = None;
    }

    pub fn recorded_count(&self) -> usize {
        self.positions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::MOVE_NONE;
    use std::path::PathBuf;

    fn make_entry(hash: u64, depth: i8, score: i16, result: f32) -> ExpEntry {
        ExpEntry {
            hash,
            best_move: MOVE_NONE,
            depth,
            score,
            game_result: result,
            count: 1,
        }
    }

    #[test]
    fn test_store_and_probe() {
        let mut table = ExpTable::new();
        let entry = make_entry(0xDEADBEEF, 8, 150, 1.0);
        table.store(entry);

        let probed = table.probe(0xDEADBEEF);
        assert!(probed.is_some());
        let p = probed.unwrap();
        assert_eq!(p.hash, 0xDEADBEEF);
        assert_eq!(p.depth, 8);
        assert_eq!(p.score, 150);
        assert_eq!(p.count, 1);
    }

    #[test]
    fn test_probe_miss() {
        let table = ExpTable::new();
        assert!(table.probe(0x12345678).is_none());
    }

    #[test]
    fn test_merge_same_position_keeps_deeper() {
        let mut table = ExpTable::new();
        table.store(make_entry(0xAAAA, 6, 100, 1.0));
        table.store(make_entry(0xAAAA, 10, 200, 0.0));

        let p = table.probe(0xAAAA).unwrap();
        assert_eq!(p.depth, 10, "should keep the deeper entry's depth");
        assert_eq!(p.score, 200, "should keep the deeper entry's score");
        assert_eq!(p.count, 2, "count should increment");
    }

    #[test]
    fn test_merge_blends_game_result() {
        let mut table = ExpTable::new();
        table.store(make_entry(0xBBBB, 8, 100, 1.0));
        table.store(make_entry(0xBBBB, 8, 100, 0.0));

        let p = table.probe(0xBBBB).unwrap();
        assert!((p.game_result - 0.5).abs() < 0.01, "game result should blend to ~0.5");
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let mut table = ExpTable::new();
        table.store(make_entry(0x1111, 5, 50, 1.0));
        table.store(make_entry(0x2222, 10, -30, 0.0));
        table.store(make_entry(0x3333, 7, 0, 0.5));

        let path = PathBuf::from("/tmp/nagato_test_exp.bin");
        table.save(&path).expect("save should succeed");

        let mut loaded = ExpTable::new();
        let count = loaded.load(&path).expect("load should succeed");
        assert_eq!(count, 3);

        let p1 = loaded.probe(0x1111).unwrap();
        assert_eq!(p1.depth, 5);
        assert_eq!(p1.score, 50);

        let p2 = loaded.probe(0x2222).unwrap();
        assert_eq!(p2.depth, 10);
        assert_eq!(p2.score, -30);

        let p3 = loaded.probe(0x3333).unwrap();
        assert_eq!(p3.depth, 7);
        assert!((p3.game_result - 0.5).abs() < 0.01);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_nonexistent_returns_zero() {
        let mut table = ExpTable::new();
        let count = table.load(Path::new("/tmp/nagato_doesnt_exist.bin")).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_game_recorder_minimum_depth() {
        let mut recorder = GameRecorder::new();
        recorder.set_our_color(0);

        recorder.record(0xAAAA, MOVE_NONE, 3, 100, 0);
        assert_eq!(recorder.recorded_count(), 0);

        recorder.record(0xBBBB, MOVE_NONE, 4, 200, 0);
        assert_eq!(recorder.recorded_count(), 1);
    }

    #[test]
    fn test_game_recorder_flush_win() {
        let mut recorder = GameRecorder::new();
        let mut table = ExpTable::new();
        recorder.set_our_color(0);

        recorder.record(0xAAAA, MOVE_NONE, 8, 100, 0);
        recorder.record(0xBBBB, MOVE_NONE, 6, -50, 1);

        recorder.flush(&mut table, GameResult::Win);

        let p1 = table.probe(0xAAAA).unwrap();
        assert!((p1.game_result - 1.0).abs() < 0.01);

        let p2 = table.probe(0xBBBB).unwrap();
        assert!((p2.game_result - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_table_len() {
        let mut table = ExpTable::new();
        assert_eq!(table.len(), 0);
        assert!(table.is_empty());

        table.store(make_entry(0x1111, 5, 50, 1.0));
        assert_eq!(table.len(), 1);
        assert!(!table.is_empty());
    }
}
