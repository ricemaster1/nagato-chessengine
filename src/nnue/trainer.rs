use super::{L1_SIZE, L1_PAIR, L2_SIZE, L2_INPUT, NUM_PSQT_BUCKETS, NUM_LAYER_STACKS, SKIP_SIZE};
use super::network::psqt_bucket;
use crate::bitboard::*;
use crate::datagen::{unpack_board, ENTRY_SIZE};
use super::features::{transform_halfka, KING_BUCKETS, PER_BUCKET_FEATURES};

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
    let transformed = transform_halfka(&board);
    let white_feats = transformed.white;
    let black_feats = transformed.black;
    let piece_count = white_feats.len() as u32;

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
        let ft_scale = 0.01f32;

        let mut ft_w = vec![[0.0f32; L1_SIZE]; FT_SIZE];
        for row in ft_w.iter_mut() {
            for v in row.iter_mut() { *v = rng.gen_range(-ft_scale..ft_scale); }
        }
        let mut ft_b = [0.0f32; L1_SIZE];
        for v in ft_b.iter_mut() { *v = 0.5 + rng.gen_range(-0.05..0.05); }

        let psqt_w = vec![[0.0f32; NUM_PSQT_BUCKETS]; FT_SIZE];

        let l2_scale = (2.0 / L2_INPUT as f32).sqrt();
        let l2_w: [Vec<[f32; L2_SIZE]>; NUM_LAYER_STACKS] = std::array::from_fn(|_| {
            let mut w = vec![[0.0f32; L2_SIZE]; L2_INPUT];
            for row in w.iter_mut() { for v in row.iter_mut() { *v = rng.gen_range(-l2_scale..l2_scale); } }
            w
        });
        let l2_b = [[0.0f32; L2_SIZE]; NUM_LAYER_STACKS];

        let out_scale = (2.0 / L2_SIZE as f32).sqrt();
        let mut out_w = [[0.0f32; L2_SIZE]; NUM_LAYER_STACKS];
        for s in out_w.iter_mut() { for v in s.iter_mut() { *v = rng.gen_range(-out_scale..out_scale); } }
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

const SIGMOID_K: f32 = 400.0;

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x / SIGMOID_K).exp()) }

pub fn loss(pred: f32, target_score: f32, wdl: f32, lambda: f32) -> f32 {
    let p = sigmoid(pred);
    let t_eval = sigmoid(target_score);
    let mse = (p - t_eval).powi(2);
    let ce = -(wdl * p.max(1e-7).ln() + (1.0 - wdl) * (1.0 - p).max(1e-7).ln());
    lambda * mse + (1.0 - lambda) * ce
}

pub struct Gradients {
    pub ft_w: Vec<[f32; L1_SIZE]>,
    pub ft_b: [f32; L1_SIZE],
    pub psqt_w: Vec<[f32; NUM_PSQT_BUCKETS]>,
    pub l2_w: [Vec<[f32; L2_SIZE]>; NUM_LAYER_STACKS],
    pub l2_b: [[f32; L2_SIZE]; NUM_LAYER_STACKS],
    pub out_w: [[f32; L2_SIZE]; NUM_LAYER_STACKS],
    pub out_b: [f32; NUM_LAYER_STACKS],
    pub skip_w: [[f32; SKIP_SIZE]; NUM_LAYER_STACKS],
    pub count: usize,
}

impl Gradients {
    pub fn new() -> Self {
        Gradients {
            ft_w: vec![[0.0; L1_SIZE]; FT_SIZE],
            ft_b: [0.0; L1_SIZE],
            psqt_w: vec![[0.0; NUM_PSQT_BUCKETS]; FT_SIZE],
            l2_w: std::array::from_fn(|_| vec![[0.0; L2_SIZE]; L2_INPUT]),
            l2_b: [[0.0; L2_SIZE]; NUM_LAYER_STACKS],
            out_w: [[0.0; L2_SIZE]; NUM_LAYER_STACKS],
            out_b: [0.0; NUM_LAYER_STACKS],
            skip_w: [[0.0; SKIP_SIZE]; NUM_LAYER_STACKS],
            count: 0,
        }
    }

    pub fn zero(&mut self) {
        for r in self.ft_w.iter_mut() { *r = [0.0; L1_SIZE]; }
        self.ft_b = [0.0; L1_SIZE];
        for r in self.psqt_w.iter_mut() { *r = [0.0; NUM_PSQT_BUCKETS]; }
        for s in 0..NUM_LAYER_STACKS {
            for r in self.l2_w[s].iter_mut() { *r = [0.0; L2_SIZE]; }
            self.l2_b[s] = [0.0; L2_SIZE];
            self.out_w[s] = [0.0; L2_SIZE];
            self.out_b[s] = 0.0;
            self.skip_w[s] = [0.0; SKIP_SIZE];
        }
        self.count = 0;
    }
}

fn dcrelu(x: f32) -> f32 { if x > 0.0 && x < 1.0 { 1.0 } else { 0.0 } }

pub fn backward(w: &TrainableWeights, s: &TrainingSample, fwd: &ForwardResult, lambda: f32, g: &mut Gradients) {
    let p = sigmoid(fwd.output);
    let t = sigmoid(s.score);
    let d_out = lambda * 2.0 * (p - t) + (1.0 - lambda) * (p - s.wdl);
    let stack = fwd.stack;

    g.out_b[stack] += d_out;
    for j in 0..L2_SIZE { g.out_w[stack][j] += d_out * fwd.l2_out[j]; }
    for j in 0..SKIP_SIZE { g.skip_w[stack][j] += d_out * fwd.l2_in[j]; }

    let mut d_l2_out = [0.0f32; L2_SIZE];
    for j in 0..L2_SIZE { d_l2_out[j] = d_out * w.out_w[stack][j] * dcrelu(fwd.l2_out[j]); }

    let mut d_l2_in = [0.0f32; L2_INPUT];
    for i in 0..L2_INPUT {
        for j in 0..L2_SIZE {
            d_l2_in[i] += d_l2_out[j] * w.l2_w[stack][i][j];
            g.l2_w[stack][i][j] += d_l2_out[j] * fwd.l2_in[i];
        }
    }
    for j in 0..L2_SIZE { g.l2_b[stack][j] += d_l2_out[j]; }
    for j in 0..SKIP_SIZE { d_l2_in[j] += d_out * w.skip_w[stack][j]; }

    let (stm_l1, opp_l1) = match s.stm {
        Color::White => (&fwd.l1_white, &fwd.l1_black),
        Color::Black => (&fwd.l1_black, &fwd.l1_white),
    };

    let mut d_stm = [0.0f32; L1_SIZE];
    let mut d_opp = [0.0f32; L1_SIZE];
    for i in 0..L1_PAIR {
        let a = crelu(stm_l1[i]);
        let b = crelu(stm_l1[i + L1_PAIR]);
        d_stm[i] += d_l2_in[i] * b * dcrelu(stm_l1[i]);
        d_stm[i + L1_PAIR] += d_l2_in[i] * a * dcrelu(stm_l1[i + L1_PAIR]);
        let a2 = crelu(opp_l1[i]);
        let b2 = crelu(opp_l1[i + L1_PAIR]);
        d_opp[i] += d_l2_in[L1_PAIR + i] * b2 * dcrelu(opp_l1[i]);
        d_opp[i + L1_PAIR] += d_l2_in[L1_PAIR + i] * a2 * dcrelu(opp_l1[i + L1_PAIR]);
    }

    let (d_white, d_black) = match s.stm {
        Color::White => (&d_stm, &d_opp),
        Color::Black => (&d_opp, &d_stm),
    };
    for j in 0..L1_SIZE { g.ft_b[j] += d_white[j] + d_black[j]; }
    for &fi in &s.white_features { for j in 0..L1_SIZE { g.ft_w[fi][j] += d_white[j]; } }
    for &fi in &s.black_features { for j in 0..L1_SIZE { g.ft_w[fi][j] += d_black[j]; } }

    let d_psqt_stm = d_out;
    let d_psqt_opp = -d_out;
    let (stm_feats, opp_feats) = match s.stm {
        Color::White => (&s.white_features, &s.black_features),
        Color::Black => (&s.black_features, &s.white_features),
    };
    for &fi in stm_feats { g.psqt_w[fi][stack] += d_psqt_stm; }
    for &fi in opp_feats { g.psqt_w[fi][stack] += d_psqt_opp; }

    g.count += 1;
}

pub fn sgd_step(w: &mut TrainableWeights, g: &Gradients, lr: f32) {
    if g.count == 0 { return; }
    let scale = lr / g.count as f32;

    for i in 0..w.ft_w.len() {
        for j in 0..L1_SIZE { w.ft_w[i][j] -= scale * g.ft_w[i][j]; }
    }
    for j in 0..L1_SIZE { w.ft_b[j] -= scale * g.ft_b[j]; }
    for i in 0..w.psqt_w.len() {
        for b in 0..NUM_PSQT_BUCKETS { w.psqt_w[i][b] -= scale * g.psqt_w[i][b]; }
    }
    for s in 0..NUM_LAYER_STACKS {
        for i in 0..L2_INPUT {
            for j in 0..L2_SIZE { w.l2_w[s][i][j] -= scale * g.l2_w[s][i][j]; }
        }
        for j in 0..L2_SIZE { w.l2_b[s][j] -= scale * g.l2_b[s][j]; }
        for j in 0..L2_SIZE { w.out_w[s][j] -= scale * g.out_w[s][j]; }
        w.out_b[s] -= scale * g.out_b[s];
        for j in 0..SKIP_SIZE { w.skip_w[s][j] -= scale * g.skip_w[s][j]; }
    }
}

pub fn train(samples: &[TrainingSample], epochs: u32, batch_size: usize, lr: f32, lambda: f32) -> TrainableWeights {
    use rand::seq::SliceRandom;

    let mut w = TrainableWeights::new_random();
    let mut g = Gradients::new();
    let mut indices: Vec<usize> = (0..samples.len()).collect();
    let mut rng = rand::thread_rng();
    let mut current_lr = lr;
    let drop_interval = (epochs / 4).max(1);

    for epoch in 0..epochs {
        if epoch > 0 && epoch % drop_interval == 0 {
            current_lr *= 0.5;
            eprintln!("info string lr decay -> {:.6}", current_lr);
        }

        indices.shuffle(&mut rng);
        let mut epoch_loss = 0.0f64;
        let mut epoch_count = 0usize;

        for batch_start in (0..indices.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(indices.len());
            g.zero();

            for &idx in &indices[batch_start..batch_end] {
                let s = &samples[idx];
                let fwd = forward(&w, s);
                epoch_loss += loss(fwd.output, s.score, s.wdl, lambda) as f64;
                epoch_count += 1;
                backward(&w, s, &fwd, lambda, &mut g);
            }
            sgd_step(&mut w, &g, current_lr);
        }

        let avg_loss = if epoch_count > 0 { epoch_loss / epoch_count as f64 } else { 0.0 };
        eprintln!("info string epoch {}/{} loss {:.6}", epoch + 1, epochs, avg_loss);
    }
    w
}

pub fn save_weights(w: &TrainableWeights, path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    f.write_all(b"NAGT")?;
    f.write_all(&3u32.to_le_bytes())?;

    for i in 0..w.ft_w.len() {
        for j in 0..L1_SIZE { f.write_all(&w.ft_w[i][j].to_le_bytes())?; }
    }
    for j in 0..L1_SIZE { f.write_all(&w.ft_b[j].to_le_bytes())?; }
    for i in 0..w.psqt_w.len() {
        for b in 0..NUM_PSQT_BUCKETS { f.write_all(&w.psqt_w[i][b].to_le_bytes())?; }
    }
    for s in 0..NUM_LAYER_STACKS {
        for i in 0..L2_INPUT {
            for j in 0..L2_SIZE { f.write_all(&w.l2_w[s][i][j].to_le_bytes())?; }
        }
        for j in 0..L2_SIZE { f.write_all(&w.l2_b[s][j].to_le_bytes())?; }
        for j in 0..L2_SIZE { f.write_all(&w.out_w[s][j].to_le_bytes())?; }
        f.write_all(&w.out_b[s].to_le_bytes())?;
        for j in 0..SKIP_SIZE { f.write_all(&w.skip_w[s][j].to_le_bytes())?; }
    }
    f.flush()
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
        assert_eq!(sample.white_features.len(), 32);
        assert_eq!(sample.black_features.len(), 32);
        assert!((sample.score - 25.0).abs() < 0.01);
        assert!((sample.wdl - 1.0).abs() < 0.01);
        assert_eq!(sample.piece_count, 32);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let w = TrainableWeights::new_random();
        let path = "/tmp/nagato_test_nn.bin";
        save_weights(&w, path).unwrap();
        let loaded = super::super::network::load_weights_from_file(std::path::Path::new(path)).unwrap();
        assert_eq!(loaded.version, 3);
        for j in 0..L1_SIZE {
            assert!((w.ft_b[j] - loaded.l1_biases[j]).abs() < 1e-6);
        }
        for j in 0..L2_SIZE {
            assert!((w.out_w[0][j] - loaded.output_weights[0][j]).abs() < 1e-6);
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_load_lichess_samples() {
        setup();
        let path = "lichess_test_data.bin";
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping: {} not found", path);
            return;
        }
        let samples = load_samples(path);
        assert!(!samples.is_empty(), "No samples loaded");
        for (i, s) in samples.iter().enumerate() {
            assert!(!s.white_features.is_empty(), "sample {}: empty white features", i);
            assert!(!s.black_features.is_empty(), "sample {}: empty black features", i);
            assert!(s.piece_count > 0, "sample {}: zero pieces", i);
        }
        eprintln!("Loaded {} training samples from {}", samples.len(), path);
    }
}
