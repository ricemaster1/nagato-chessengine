use super::{L1_SIZE, L1_PAIR, L2_SIZE, L2_INPUT, NUM_PSQT_BUCKETS, NUM_LAYER_STACKS, SKIP_SIZE};
use super::network::psqt_bucket;
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

pub struct TrainableWeights {
    pub ft_w: Vec<[f32; L1_SIZE]>,
    pub ft_b: [f32; L1_SIZE],
    pub psqt_w: Vec<[f32; NUM_PSQT_BUCKETS]>,
    pub l2_w: [Vec<[f32; L2_SIZE]>; NUM_LAYER_STACKS],
    pub l2_b: [[f32; L2_SIZE]; NUM_LAYER_STACKS],
    pub out_w: [[f32; L2_SIZE]; NUM_LAYER_STACKS],
    pub out_b: [f32; NUM_LAYER_STACKS],
    pub skip_w: [[f32; SKIP_SIZE]; NUM_LAYER_STACKS],
}

impl TrainableWeights {
    pub fn new_random() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = 0.01f32;

        let mut ft_w = vec![[0.0f32; L1_SIZE]; FT_SIZE];
        for row in ft_w.iter_mut() {
            for v in row.iter_mut() { *v = rng.gen_range(-scale..scale); }
        }
        let mut ft_b = [0.0f32; L1_SIZE];
        for v in ft_b.iter_mut() { *v = rng.gen_range(-scale..scale); }

        let psqt_w = vec![[0.0f32; NUM_PSQT_BUCKETS]; FT_SIZE];

        let l2_w: [Vec<[f32; L2_SIZE]>; NUM_LAYER_STACKS] = std::array::from_fn(|_| {
            let mut w = vec![[0.0f32; L2_SIZE]; L2_INPUT];
            for row in w.iter_mut() { for v in row.iter_mut() { *v = rng.gen_range(-scale..scale); } }
            w
        });
        let mut l2_b = [[0.0f32; L2_SIZE]; NUM_LAYER_STACKS];
        for s in l2_b.iter_mut() { for v in s.iter_mut() { *v = rng.gen_range(-scale..scale); } }

        let mut out_w = [[0.0f32; L2_SIZE]; NUM_LAYER_STACKS];
        for s in out_w.iter_mut() { for v in s.iter_mut() { *v = rng.gen_range(-scale..scale); } }
        let out_b = [0.0f32; NUM_LAYER_STACKS];
        let skip_w = [[0.0f32; SKIP_SIZE]; NUM_LAYER_STACKS];

        TrainableWeights { ft_w, ft_b, psqt_w, l2_w, l2_b, out_w, out_b, skip_w }
    }
}

pub struct ForwardResult {
    pub l1_white: [f32; L1_SIZE],
    pub l1_black: [f32; L1_SIZE],
    pub pw: [f32; L1_PAIR],
    pub l2_in: [f32; L2_INPUT],
    pub l2_out: [f32; L2_SIZE],
    pub psqt_stm: f32,
    pub psqt_opp: f32,
    pub positional: f32,
    pub output: f32,
    pub stack: usize,
}

fn crelu(x: f32) -> f32 { x.clamp(0.0, 1.0) }

pub fn forward(w: &TrainableWeights, s: &TrainingSample) -> ForwardResult {
    let stack = psqt_bucket(s.piece_count);

    let mut l1_w = w.ft_b;
    let mut l1_b = w.ft_b;
    for &fi in &s.white_features { for j in 0..L1_SIZE { l1_w[j] += w.ft_w[fi][j]; } }
    for &fi in &s.black_features { for j in 0..L1_SIZE { l1_b[j] += w.ft_w[fi][j]; } }

    let (stm, opp) = match s.stm {
        Color::White => (&l1_w, &l1_b),
        Color::Black => (&l1_b, &l1_w),
    };

    let mut pw = [0.0f32; L1_PAIR];
    for i in 0..L1_PAIR {
        pw[i] = crelu(stm[i * 2]) * crelu(stm[i * 2 + 1])
              + crelu(opp[i * 2]) * crelu(opp[i * 2 + 1]);
    }

    let mut l2_in = [0.0f32; L2_INPUT];
    for i in 0..L1_PAIR { l2_in[i] = crelu(stm[i]) * crelu(stm[i + L1_PAIR]); }
    for i in 0..L1_PAIR { l2_in[L1_PAIR + i] = crelu(opp[i]) * crelu(opp[i + L1_PAIR]); }

    let mut l2_out = w.l2_b[stack];
    for i in 0..L2_INPUT {
        for j in 0..L2_SIZE { l2_out[j] += l2_in[i] * w.l2_w[stack][i][j]; }
    }
    for v in l2_out.iter_mut() { *v = crelu(*v); }

    let mut positional = w.out_b[stack];
    for j in 0..L2_SIZE { positional += l2_out[j] * w.out_w[stack][j]; }
    for j in 0..SKIP_SIZE { positional += l2_in[j] * w.skip_w[stack][j]; }

    let mut psqt_stm = 0.0f32;
    let mut psqt_opp = 0.0f32;
    let (stm_feats, opp_feats) = match s.stm {
        Color::White => (&s.white_features, &s.black_features),
        Color::Black => (&s.black_features, &s.white_features),
    };
    for &fi in stm_feats { psqt_stm += w.psqt_w[fi][stack]; }
    for &fi in opp_feats { psqt_opp += w.psqt_w[fi][stack]; }

    let output = positional + psqt_stm - psqt_opp;

    ForwardResult { l1_white: l1_w, l1_black: l1_b, pw, l2_in, l2_out, psqt_stm, psqt_opp, positional, output, stack }
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
