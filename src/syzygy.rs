use crate::board::Board;
use crate::eval::MATE_SCORE;
use shakmaty::{fen::Fen, CastlingMode, Chess};
use shakmaty_syzygy::{Tablebase, Wdl};
use std::sync::{Mutex, OnceLock};

#[derive(Default)]
struct SyzygyConfig {
    path: String,
    probe_depth: i32,
}

#[derive(Default)]
struct TablebaseCache {
    path: String,
    tables: Option<Tablebase<Chess>>,
}

fn config() -> &'static Mutex<SyzygyConfig> {
    static CONFIG: OnceLock<Mutex<SyzygyConfig>> = OnceLock::new();
    CONFIG.get_or_init(|| {
        Mutex::new(SyzygyConfig {
            path: String::new(),
            probe_depth: 1,
        })
    })
}

fn cache() -> &'static Mutex<TablebaseCache> {
    static CACHE: OnceLock<Mutex<TablebaseCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(TablebaseCache::default()))
}

pub fn set_path(path: &str) {
    if let Ok(mut cfg) = config().lock() {
        cfg.path = path.trim().to_string();
    }
}

pub fn set_probe_depth(depth: i32) {
    if let Ok(mut cfg) = config().lock() {
        cfg.probe_depth = depth.max(1);
    }
}

fn get_config_snapshot() -> Option<(String, i32)> {
    let cfg = config().lock().ok()?;
    Some((cfg.path.clone(), cfg.probe_depth))
}

fn with_tablebase<R>(path: &str, f: impl FnOnce(&Tablebase<Chess>) -> R) -> Option<R> {
    let mut cache = cache().lock().ok()?;

    if cache.path != path {
        let mut tables = Tablebase::<Chess>::new();
        let loaded = tables.add_directory(path).ok()?;
        if loaded == 0 {
            return None;
        }
        cache.path = path.to_string();
        cache.tables = Some(tables);
    }

    cache.tables.as_ref().map(f)
}

pub fn probe_wdl_score(board: &Board, depth: i32, ply: usize) -> Option<i32> {
    let (path, probe_depth) = get_config_snapshot()?;
    if path.is_empty() || depth < probe_depth {
        return None;
    }

    if board.all_occupancy.count_ones() > 7 {
        return None;
    }

    let fen = board.to_fen();
    let pos: Chess = fen
        .parse::<Fen>()
        .ok()?
        .into_position(CastlingMode::Standard)
        .ok()?;

    let wdl = with_tablebase(&path, |tables| tables.probe_wdl_after_zeroing(&pos).ok())??;

    let ply_i32 = ply as i32;
    let score = match wdl {
        Wdl::Win => MATE_SCORE - ply_i32,
        Wdl::CursedWin => 200,
        Wdl::Draw => 0,
        Wdl::BlessedLoss => -200,
        Wdl::Loss => -MATE_SCORE + ply_i32,
    };
    Some(score)
}
