use crate::bitboard::*;
use crate::datagen::{unpack_board, ENTRY_SIZE};
use super::features::{feature_index_halfkp_white, feature_index_halfkp_black,
                       piece_index_no_king, KING_BUCKETS, PER_BUCKET_FEATURES};

pub const FT_SIZE: usize = KING_BUCKETS * PER_BUCKET_FEATURES;

pub struct TrainingSample {
    pub white_features: Vec<usize>,
    pub black_features: Vec<usize>,
    pub score: f32,
    pub wdl: f32,
    pub stm: Color,
    pub piece_count: u32,
}

pub fn parse_entry(data: &[u8]) -> Option<TrainingSample> {
    if data.len() < ENTRY_SIZE { return None; }

    let packed: [u8; 32] = data[0..32].try_into().ok()?;
    let side = if data[32] == 0 { Color::White } else { Color::Black };
    let castling = data[33];
    let ep_file = data[34];
    let score_raw = i16::from_le_bytes([data[36], data[37]]);
    let wdl_byte = data[38];

    let board = unpack_board(&packed, side, castling, ep_file);
    let wk = board.king_sq(Color::White);
    let bk = board.king_sq(Color::Black);

    let mut white_feats = Vec::with_capacity(32);
    let mut black_feats = Vec::with_capacity(32);
    let mut piece_count = 0u32;

    for color_idx in 0..2 {
        let c = if color_idx == 0 { Color::White } else { Color::Black };
        for piece_idx in 0..PIECE_COUNT {
            let piece = Piece::from_index(piece_idx);
            if piece_index_no_king(piece).is_none() { continue; }
            let mut bb = board.pieces[c.index()][piece_idx];
            while bb != 0 {
                let sq = pop_lsb(&mut bb);
                white_feats.push(feature_index_halfkp_white(piece, c, sq, wk));
                black_feats.push(feature_index_halfkp_black(piece, c, sq, bk));
                piece_count += 1;
            }
        }
    }

    let wdl = match wdl_byte { 2 => 1.0, 1 => 0.5, _ => 0.0 };
    let score = score_raw as f32;

    Some(TrainingSample {
        white_features: white_feats,
        black_features: black_feats,
        score,
        wdl,
        stm: side,
        piece_count,
    })
}

pub fn load_samples(path: &str) -> Vec<TrainingSample> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).expect("cannot open training data");
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).expect("cannot read training data");

    let n = buf.len() / ENTRY_SIZE;
    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * ENTRY_SIZE;
        if let Some(s) = parse_entry(&buf[off..off + ENTRY_SIZE]) {
            samples.push(s);
        }
    }
    samples
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::datagen;

    fn setup() {
        crate::zobrist::init();
        crate::movegen::init();
    }

    #[test]
    fn test_parse_startpos_entry() {
        setup();
        let board = Board::start_pos();
        let mut buf = Vec::new();
        datagen::write_entry_pub(&mut buf, &board, 25, 2);
        let sample = parse_entry(&buf).unwrap();
        assert_eq!(sample.white_features.len(), 30);
        assert_eq!(sample.black_features.len(), 30);
        assert!((sample.score - 25.0).abs() < 0.01);
        assert!((sample.wdl - 1.0).abs() < 0.01);
        assert_eq!(sample.piece_count, 30);
    }
}
