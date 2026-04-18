use crate::bitboard::Color;
use crate::board::Board;
use crate::nnue::features::{KING_BUCKETS, PER_BUCKET_FEATURES, king_bucket_of};

use super::{L1_SIZE, L1_PAIR, L2_INPUT, L2_SIZE, NUM_PSQT_BUCKETS, NUM_LAYER_STACKS, SKIP_SIZE, QA, QB};
#[cfg(test)]
use super::INPUT_SIZE;
use super::accumulator::{Accumulator, AccumulatorQ};
use super::simd;

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

pub const NNUE_FORMAT_VERSION: u32 = 4;

pub struct NnueWeights {
    pub version: u32,
    pub l1_weights: Vec<[f32; L1_SIZE]>,
    pub l1_biases: [f32; L1_SIZE],
    pub psqt_weights: Vec<[f32; NUM_PSQT_BUCKETS]>,
    pub l2_weights: [Vec<[f32; L2_SIZE]>; NUM_LAYER_STACKS],
    pub l2_biases: [[f32; L2_SIZE]; NUM_LAYER_STACKS],
    pub output_weights: [[f32; L2_SIZE]; NUM_LAYER_STACKS],
    pub output_bias: [f32; NUM_LAYER_STACKS],
    pub skip_weights: [[f32; SKIP_SIZE]; NUM_LAYER_STACKS],
}

pub struct NnueWeightsQ {
    pub version: u32,
    pub ft_weights: Vec<[i16; L1_SIZE]>,
    pub ft_biases: [i16; L1_SIZE],
    pub psqt_weights: Vec<[i32; NUM_PSQT_BUCKETS]>,
    pub l2_weights: [Vec<[i8; L2_SIZE]>; NUM_LAYER_STACKS],
    pub l2_weights_t: [[[i8; L2_INPUT]; L2_SIZE]; NUM_LAYER_STACKS],
    pub l2_biases: [[i32; L2_SIZE]; NUM_LAYER_STACKS],
    pub out_weights: [[i16; L2_SIZE]; NUM_LAYER_STACKS],
    pub out_bias: [i32; NUM_LAYER_STACKS],
    pub skip_weights: [[i16; SKIP_SIZE]; NUM_LAYER_STACKS],
}

pub fn quantize_weights(w: &NnueWeights) -> NnueWeightsQ {
    let qa = QA as f32;
    let qb = QB as f32;
    let qa_qb = qa * qb;
    let qb_qb = qb * qb;

    let mut ft_weights = vec![[0i16; L1_SIZE]; w.l1_weights.len()];
    for i in 0..w.l1_weights.len() {
        for j in 0..L1_SIZE {
            ft_weights[i][j] = (w.l1_weights[i][j] * qa).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        }
    }

    let mut ft_biases = [0i16; L1_SIZE];
    for j in 0..L1_SIZE {
        ft_biases[j] = (w.l1_biases[j] * qa).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
    }

    let mut psqt_weights = vec![[0i32; NUM_PSQT_BUCKETS]; w.psqt_weights.len()];
    for i in 0..w.psqt_weights.len() {
        for b in 0..NUM_PSQT_BUCKETS {
            psqt_weights[i][b] = (w.psqt_weights[i][b] * qa).round() as i32;
        }
    }

    let concat = L2_INPUT;
    let mut l2_weights: [Vec<[i8; L2_SIZE]>; NUM_LAYER_STACKS] = std::array::from_fn(|_| vec![[0i8; L2_SIZE]; concat]);
    let mut l2_biases = [[0i32; L2_SIZE]; NUM_LAYER_STACKS];
    let mut out_weights = [[0i16; L2_SIZE]; NUM_LAYER_STACKS];
    let mut out_bias = [0i32; NUM_LAYER_STACKS];
    let mut skip_weights = [[0i16; SKIP_SIZE]; NUM_LAYER_STACKS];
    let mut l2_weights_t = [[[0i8; L2_INPUT]; L2_SIZE]; NUM_LAYER_STACKS];

    for s in 0..NUM_LAYER_STACKS {
        for i in 0..concat {
            for j in 0..L2_SIZE {
                l2_weights[s][i][j] = (w.l2_weights[s][i][j] * qb).round().clamp(i8::MIN as f32, i8::MAX as f32) as i8;
            }
        }
        for j in 0..L2_SIZE {
            l2_biases[s][j] = (w.l2_biases[s][j] * qa_qb).round() as i32;
        }
        for j in 0..L2_SIZE {
            out_weights[s][j] = (w.output_weights[s][j] * qb).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        }
        out_bias[s] = (w.output_bias[s] * qb_qb).round() as i32;
        for j in 0..SKIP_SIZE {
            skip_weights[s][j] = (w.skip_weights[s][j] * qb).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        }
        for j in 0..L2_SIZE {
            for i in 0..concat {
                l2_weights_t[s][j][i] = l2_weights[s][i][j];
            }
        }
    }

    NnueWeightsQ {
        version: w.version,
        ft_weights,
        ft_biases,
        psqt_weights,
        l2_weights,
        l2_weights_t,
        l2_biases,
        out_weights,
        out_bias,
        skip_weights,
    }
}

static NNUE_STATE: OnceLock<NnueWeights> = OnceLock::new();
static NNUE_STATE_Q: OnceLock<NnueWeightsQ> = OnceLock::new();
static NNUE_LOADED: AtomicBool = AtomicBool::new(false);

pub fn init() {
    let path = std::path::Path::new("nn.bin");
    if path.exists() {
        match load_weights_from_file(path) {
            Ok(w) => {
                let q = quantize_weights(&w);
                let _ = NNUE_STATE.set(w);
                let _ = NNUE_STATE_Q.set(q);
                NNUE_LOADED.store(true, Ordering::Relaxed);
                eprintln!("info string NNUE loaded from nn.bin");
            }
            Err(e) => {
                eprintln!("info string NNUE load failed: {} — using HCE", e);
            }
        }
    } else {
        eprintln!("info string nn.bin not found — using HCE");
    }
}

#[inline]
pub fn is_active() -> bool {
    NNUE_LOADED.load(Ordering::Relaxed)
}

#[inline]
pub fn weight_version() -> u32 {
    weights_q().version
}

#[inline]
pub(super) fn weights() -> &'static NnueWeights {
    NNUE_STATE.get().unwrap()
}

#[inline]
pub(super) fn weights_q() -> &'static NnueWeightsQ {
    NNUE_STATE_Q.get().unwrap()
}

pub fn load_weights_from_file(path: &std::path::Path) -> Result<NnueWeights, String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|e| format!("open: {}", e))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).map_err(|e| format!("read: {}", e))?;
    load_weights_from_bytes(&buf)
}

pub fn load_weights_from_bytes(data: &[u8]) -> Result<NnueWeights, String> {
    let mut cursor = 4usize;
    let mut v2_header_bytes = 0usize;

    let read_u32 = |cursor: &mut usize, data: &[u8]| -> Result<u32, String> {
        if *cursor + 4 > data.len() {
            return Err("unexpected EOF reading u32".into());
        }
        let val = u32::from_le_bytes([data[*cursor], data[*cursor+1], data[*cursor+2], data[*cursor+3]]);
        *cursor += 4;
        Ok(val)
    };

    let read_f32 = |cursor: &mut usize, data: &[u8]| -> Result<f32, String> {
        if *cursor + 4 > data.len() {
            return Err("unexpected EOF reading f32".into());
        }
        let val = f32::from_le_bytes([data[*cursor], data[*cursor+1], data[*cursor+2], data[*cursor+3]]);
        *cursor += 4;
        Ok(val)
    };

    if data.len() < 8 {
        return Err("file too small".into());
    }
    if &data[0..4] != b"NAGT" {
        return Err("bad magic".into());
    }

    let version = read_u32(&mut cursor, data)?;
    if version != 1 && version != 2 && version != 3 && version != 4 {
        return Err(format!("unsupported version: {}", version));
    }

    // Optional v2 header for explicit architecture metadata.
    // Legacy v2 files without this header remain supported.
    if version == 2 && cursor + 20 <= data.len() && &data[cursor..cursor + 4] == b"V2H0" {
        cursor += 4;
        v2_header_bytes = 20;
        let hdr_l1_rows = read_u32(&mut cursor, data)? as usize;
        let hdr_l1_size = read_u32(&mut cursor, data)? as usize;
        let hdr_l2_size = read_u32(&mut cursor, data)? as usize;
        let hdr_psqt_buckets = read_u32(&mut cursor, data)? as usize;
        let expected_l1_rows = KING_BUCKETS * PER_BUCKET_FEATURES;
        if hdr_l1_rows != expected_l1_rows
            || hdr_l1_size != L1_SIZE
            || hdr_l2_size != L2_SIZE
            || hdr_psqt_buckets != NUM_PSQT_BUCKETS
        {
            return Err(format!(
                "v2 header mismatch: rows={} l1={} l2={} psqt={}",
                hdr_l1_rows, hdr_l1_size, hdr_l2_size, hdr_psqt_buckets
            ));
        }
    }

    let l1_rows = if version == 1 {
        super::INPUT_SIZE
    } else {
        KING_BUCKETS * PER_BUCKET_FEATURES
    };
    let mut l1_weights = vec![[0.0f32; L1_SIZE]; l1_rows];
    for i in 0..l1_rows {
        for j in 0..L1_SIZE {
            l1_weights[i][j] = read_f32(&mut cursor, data)?;
        }
    }

    let mut l1_biases = [0.0f32; L1_SIZE];
    for j in 0..L1_SIZE {
        l1_biases[j] = read_f32(&mut cursor, data)?;
    }

    let mut psqt_weights = vec![[0.0f32; NUM_PSQT_BUCKETS]; l1_rows];
    for i in 0..l1_rows {
        for b in 0..NUM_PSQT_BUCKETS {
            psqt_weights[i][b] = read_f32(&mut cursor, data)?;
        }
    }

    let concat_size = L2_INPUT;
    let num_stacks = if version <= 2 { 1 } else { NUM_LAYER_STACKS };
    let mut l2_weights: [Vec<[f32; L2_SIZE]>; NUM_LAYER_STACKS] = std::array::from_fn(|_| vec![[0.0f32; L2_SIZE]; concat_size]);
    let mut l2_biases = [[0.0f32; L2_SIZE]; NUM_LAYER_STACKS];
    let mut output_weights = [[0.0f32; L2_SIZE]; NUM_LAYER_STACKS];
    let mut output_bias = [0.0f32; NUM_LAYER_STACKS];
    let mut skip_weights = [[0.0f32; SKIP_SIZE]; NUM_LAYER_STACKS];

    for s in 0..num_stacks {
        for i in 0..concat_size {
            for j in 0..L2_SIZE {
                l2_weights[s][i][j] = read_f32(&mut cursor, data)?;
            }
        }
        for j in 0..L2_SIZE {
            l2_biases[s][j] = read_f32(&mut cursor, data)?;
        }
        for j in 0..L2_SIZE {
            output_weights[s][j] = read_f32(&mut cursor, data)?;
        }
        output_bias[s] = read_f32(&mut cursor, data)?;
        if version >= 3 {
            for j in 0..SKIP_SIZE {
                skip_weights[s][j] = read_f32(&mut cursor, data)?;
            }
        }
    }

    if version <= 2 {
        for s in 1..NUM_LAYER_STACKS {
            l2_weights[s] = l2_weights[0].clone();
            l2_biases[s] = l2_biases[0];
            output_weights[s] = output_weights[0];
            output_bias[s] = output_bias[0];
        }
    }

    let fc_floats_per_stack = concat_size * L2_SIZE + L2_SIZE + L2_SIZE + 1;
    let skip_floats = if version >= 3 { SKIP_SIZE } else { 0 };
    let expected = 4 + 4 + v2_header_bytes
        + (l1_rows * L1_SIZE) * 4
        + L1_SIZE * 4
        + (l1_rows * NUM_PSQT_BUCKETS) * 4
        + num_stacks * (fc_floats_per_stack + skip_floats) * 4;
    if cursor != expected {
        return Err(format!("size mismatch: read {} expected {}", cursor, expected));
    }

    Ok(NnueWeights {
        version,
        l1_weights,
        l1_biases,
        psqt_weights,
        l2_weights,
        l2_biases,
        output_weights,
        output_bias,
        skip_weights,
    })
}

#[inline]
fn clipped_relu(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

const PSQT_ALPHA: i32 = 125;
const PSQT_BETA: i32 = 131;
const PSQT_GAMMA: i32 = 128;

#[inline]
pub fn psqt_bucket(piece_count: u32) -> usize {
    let count = piece_count.max(1) as usize;
    ((count - 1) / 8).min(NUM_PSQT_BUCKETS - 1)
}

pub fn forward(acc: &Accumulator, side: Color, king_sq_white: u8, king_sq_black: u8, psqt: i32, piece_count: u32) -> i32 {
    let w = weights();
    let bucket = psqt_bucket(piece_count);
    let wb = king_bucket_of(king_sq_white);
    let bb = king_bucket_of(king_sq_black);

    let (stm_acc, opp_acc) = match side {
        Color::White => (&acc.white[wb], &acc.black[bb]),
        Color::Black => (&acc.black[bb], &acc.white[wb]),
    };

    let mut l2_out = w.l2_biases[bucket];

    for i in 0..L1_PAIR {
        let lo = clipped_relu(stm_acc[i]);
        let hi = clipped_relu(stm_acc[L1_PAIR + i]);
        let activated = lo * hi;
        if activated != 0.0 {
            for j in 0..L2_SIZE {
                l2_out[j] += activated * w.l2_weights[bucket][i][j];
            }
        }
    }

    for i in 0..L1_PAIR {
        let lo = clipped_relu(opp_acc[i]);
        let hi = clipped_relu(opp_acc[L1_PAIR + i]);
        let activated = lo * hi;
        if activated != 0.0 {
            for j in 0..L2_SIZE {
                l2_out[j] += activated * w.l2_weights[bucket][L1_PAIR + i][j];
            }
        }
    }

    let mut skip_val = 0.0f32;
    for j in 0..SKIP_SIZE {
        skip_val += l2_out[j] * w.skip_weights[bucket][j];
    }

    let mut output = w.output_bias[bucket];
    for j in 0..L2_SIZE {
        output += clipped_relu(l2_out[j]) * w.output_weights[bucket][j];
    }
    output += skip_val;

    let positional = (output * 400.0) as i32;
    (PSQT_ALPHA as i64 * psqt as i64 + PSQT_BETA as i64 * positional as i64) as i32 / PSQT_GAMMA
}

pub fn evaluate(board: &Board, acc: &Accumulator) -> i32 {
    let bucket = psqt_bucket(board.piece_count());
    let (stm_psqt, opp_psqt) = match board.side {
        Color::White => (acc.psqt_white[bucket], acc.psqt_black[bucket]),
        Color::Black => (acc.psqt_black[bucket], acc.psqt_white[bucket]),
    };
    let psqt = ((stm_psqt - opp_psqt) * 400.0) as i32;
    forward(acc, board.side, board.king_sq(Color::White), board.king_sq(Color::Black), psqt, board.piece_count())
}

pub fn forward_q(acc: &AccumulatorQ, side: Color, king_sq_white: u8, king_sq_black: u8, psqt: i32, piece_count: u32) -> i32 {
    let wq = weights_q();
    let bucket = psqt_bucket(piece_count);
    let wb = king_bucket_of(king_sq_white);
    let bb = king_bucket_of(king_sq_black);

    let (stm_acc, opp_acc) = match side {
        Color::White => (&acc.white[wb], &acc.black[bb]),
        Color::Black => (&acc.black[bb], &acc.white[wb]),
    };

    let mut l2_out = [0i32; L2_SIZE];
    simd::affine_l2(stm_acc, opp_acc, &wq.l2_weights_t[bucket], &wq.l2_biases[bucket], &mut l2_out);

    let mut skip_val = 0i64;
    for j in 0..SKIP_SIZE {
        skip_val += l2_out[j] as i64 * wq.skip_weights[bucket][j] as i64;
    }
    skip_val /= QA as i64;

    let output = simd::output_layer(&l2_out, &wq.out_weights[bucket], wq.out_bias[bucket]) + skip_val;

    let positional = (output * 400 / (QB as i64 * QB as i64)) as i32;
    (PSQT_ALPHA as i64 * psqt as i64 + PSQT_BETA as i64 * positional as i64) as i32 / PSQT_GAMMA
}

pub fn evaluate_q(board: &Board, acc: &AccumulatorQ) -> i32 {
    let bucket = psqt_bucket(board.piece_count());
    let (stm_psqt, opp_psqt) = match board.side {
        Color::White => (acc.psqt_white[bucket], acc.psqt_black[bucket]),
        Color::Black => (acc.psqt_black[bucket], acc.psqt_white[bucket]),
    };
    let psqt = (stm_psqt - opp_psqt) * 400 / (QA * QA);
    forward_q(acc, board.side, board.king_sq(Color::White), board.king_sq(Color::Black), psqt, board.piece_count())
}

pub fn leb128_encode_i32(val: i32, buf: &mut Vec<u8>) {
    let mut v = val;
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if (v == 0 && byte & 0x40 == 0) || (v == -1 && byte & 0x40 != 0) {
            buf.push(byte);
            return;
        }
        buf.push(byte | 0x80);
    }
}

pub fn leb128_decode_i32(data: &[u8], cursor: &mut usize) -> Result<i32, String> {
    let mut result: i32 = 0;
    let mut shift = 0u32;
    loop {
        if *cursor >= data.len() { return Err("unexpected EOF in LEB128".into()); }
        let byte = data[*cursor];
        *cursor += 1;
        result |= ((byte & 0x7f) as i32) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            if shift < 32 && byte & 0x40 != 0 {
                result |= !0i32 << shift;
            }
            return Ok(result);
        }
        if shift >= 35 { return Err("LEB128 overflow".into()); }
    }
}

pub fn save_weights_leb128(w: &NnueWeights, path: &str) -> Result<(), String> {
    use std::io::Write;
    let qa = QA as f32;
    let qb = QB as f32;
    let mut buf = Vec::new();
    buf.extend_from_slice(b"NAGL");
    buf.extend_from_slice(&w.version.to_le_bytes());

    let l1_rows = w.l1_weights.len();
    for i in 0..l1_rows {
        for j in 0..L1_SIZE {
            leb128_encode_i32((w.l1_weights[i][j] * qa).round() as i32, &mut buf);
        }
    }
    for j in 0..L1_SIZE {
        leb128_encode_i32((w.l1_biases[j] * qa).round() as i32, &mut buf);
    }
    for i in 0..l1_rows {
        for b in 0..NUM_PSQT_BUCKETS {
            leb128_encode_i32((w.psqt_weights[i][b] * qa).round() as i32, &mut buf);
        }
    }
    let num_stacks = if w.version <= 2 { 1 } else { NUM_LAYER_STACKS };
    for s in 0..num_stacks {
        for i in 0..L2_INPUT {
            for j in 0..L2_SIZE {
                leb128_encode_i32((w.l2_weights[s][i][j] * qb).round() as i32, &mut buf);
            }
        }
        for j in 0..L2_SIZE {
            leb128_encode_i32((w.l2_biases[s][j] * qa * qb).round() as i32, &mut buf);
        }
        for j in 0..L2_SIZE {
            leb128_encode_i32((w.output_weights[s][j] * qb).round() as i32, &mut buf);
        }
        leb128_encode_i32((w.output_bias[s] * qb * qb).round() as i32, &mut buf);
        if w.version >= 3 {
            for j in 0..SKIP_SIZE {
                leb128_encode_i32((w.skip_weights[s][j] * qb).round() as i32, &mut buf);
            }
        }
    }
    let mut file = std::fs::File::create(path).map_err(|e| format!("create: {}", e))?;
    file.write_all(&buf).map_err(|e| format!("write: {}", e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psqt_bucket() {
        assert_eq!(psqt_bucket(0), 0);
        assert_eq!(psqt_bucket(1), 0);
        assert_eq!(psqt_bucket(8), 0);
        assert_eq!(psqt_bucket(9), 1);
        assert_eq!(psqt_bucket(16), 1);
        assert_eq!(psqt_bucket(17), 2);
        assert_eq!(psqt_bucket(24), 2);
        assert_eq!(psqt_bucket(25), 3);
        assert_eq!(psqt_bucket(32), 3);
    }

    #[test]
    fn test_clipped_relu() {
        assert_eq!(clipped_relu(-1.0), 0.0);
        assert_eq!(clipped_relu(0.0), 0.0);
        assert_eq!(clipped_relu(0.5), 0.5);
        assert_eq!(clipped_relu(1.0), 1.0);
        assert_eq!(clipped_relu(2.0), 1.0);
    }

    #[test]
    fn test_weight_file_size_v1() {
        let l1_rows = INPUT_SIZE;
        let ft_floats = l1_rows * L1_SIZE + L1_SIZE + l1_rows * NUM_PSQT_BUCKETS;
        let fc_floats = L2_INPUT * L2_SIZE + L2_SIZE + L2_SIZE + 1;
        let expected_bytes = 8 + (ft_floats + fc_floats) * 4;
        assert_eq!(expected_bytes, 832_780);
    }

    #[test]
    fn test_weight_file_size_v4() {
        let l1_rows = KING_BUCKETS * PER_BUCKET_FEATURES;
        let ft_floats = l1_rows * L1_SIZE + L1_SIZE + l1_rows * NUM_PSQT_BUCKETS;
        let fc_floats_per_stack = L2_INPUT * L2_SIZE + L2_SIZE + L2_SIZE + 1 + SKIP_SIZE;
        let expected_bytes = 8 + (ft_floats + NUM_LAYER_STACKS * fc_floats_per_stack) * 4;
        assert_eq!(expected_bytes, 8_120_472);
    }

    #[test]
    fn test_load_v1_roundtrip() {
        let l1_rows = INPUT_SIZE;
        let total_floats = l1_rows * L1_SIZE + L1_SIZE + l1_rows * NUM_PSQT_BUCKETS + L2_INPUT * L2_SIZE + L2_SIZE + L2_SIZE + 1;
        let mut buf: Vec<u8> = Vec::with_capacity(8 + total_floats * 4);
        buf.extend_from_slice(b"NAGT");
        buf.extend_from_slice(&1u32.to_le_bytes());
        for i in 0..total_floats {
            buf.extend_from_slice(&(i as f32 * 0.001).to_le_bytes());
        }
        let w = load_weights_from_bytes(&buf).expect("v1 load failed");
        assert_eq!(w.version, 1);
        assert_eq!(w.l1_weights.len(), l1_rows);
        assert_eq!(w.l2_weights[0].len(), L2_INPUT);
        for s in 1..NUM_LAYER_STACKS {
            assert_eq!(w.l2_weights[s][0][0], w.l2_weights[0][0][0]);
        }
        assert_eq!(w.skip_weights, [[0.0; SKIP_SIZE]; NUM_LAYER_STACKS]);
        let first_l1 = w.l1_weights[0][0];
        assert!((first_l1 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_load_v2_roundtrip() {
        let l1_rows = KING_BUCKETS * PER_BUCKET_FEATURES;
        let total_floats = l1_rows * L1_SIZE + L1_SIZE + l1_rows * NUM_PSQT_BUCKETS + L2_INPUT * L2_SIZE + L2_SIZE + L2_SIZE + 1;
        let mut buf: Vec<u8> = Vec::with_capacity(8 + total_floats * 4);
        buf.extend_from_slice(b"NAGT");
        buf.extend_from_slice(&2u32.to_le_bytes());
        for i in 0..total_floats {
            buf.extend_from_slice(&(i as f32 * 0.0001).to_le_bytes());
        }
        let w = load_weights_from_bytes(&buf).expect("v2 load failed");
        assert_eq!(w.version, 2);
        assert_eq!(w.l1_weights.len(), l1_rows);
        assert_eq!(w.l2_weights[0].len(), L2_INPUT);
    }

    #[test]
    fn test_load_v2_with_header_roundtrip() {
        let l1_rows = KING_BUCKETS * PER_BUCKET_FEATURES;
        let total_floats = l1_rows * L1_SIZE + L1_SIZE + l1_rows * NUM_PSQT_BUCKETS + L2_INPUT * L2_SIZE + L2_SIZE + L2_SIZE + 1;
        let mut buf: Vec<u8> = Vec::with_capacity(8 + 20 + total_floats * 4);
        buf.extend_from_slice(b"NAGT");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(b"V2H0");
        buf.extend_from_slice(&(l1_rows as u32).to_le_bytes());
        buf.extend_from_slice(&(L1_SIZE as u32).to_le_bytes());
        buf.extend_from_slice(&(L2_SIZE as u32).to_le_bytes());
        buf.extend_from_slice(&(NUM_PSQT_BUCKETS as u32).to_le_bytes());
        for i in 0..total_floats {
            buf.extend_from_slice(&(i as f32 * 0.0001).to_le_bytes());
        }
        let w = load_weights_from_bytes(&buf).expect("v2 header load failed");
        assert_eq!(w.version, 2);
        assert_eq!(w.l1_weights.len(), l1_rows);
        assert_eq!(w.l2_weights[0].len(), L2_INPUT);
    }

    #[test]
    fn test_load_v4_roundtrip() {
        let l1_rows = KING_BUCKETS * PER_BUCKET_FEATURES;
        let ft_floats = l1_rows * L1_SIZE + L1_SIZE + l1_rows * NUM_PSQT_BUCKETS;
        let fc_floats_per_stack = L2_INPUT * L2_SIZE + L2_SIZE + L2_SIZE + 1 + SKIP_SIZE;
        let total_floats = ft_floats + NUM_LAYER_STACKS * fc_floats_per_stack;
        let mut buf: Vec<u8> = Vec::with_capacity(8 + total_floats * 4);
        buf.extend_from_slice(b"NAGT");
        buf.extend_from_slice(&NNUE_FORMAT_VERSION.to_le_bytes());
        for i in 0..total_floats {
            buf.extend_from_slice(&(i as f32 * 0.0001).to_le_bytes());
        }
        let w = load_weights_from_bytes(&buf).expect("v4 load failed");
        assert_eq!(w.version, NNUE_FORMAT_VERSION);
        assert_eq!(w.l1_weights.len(), l1_rows);
        for s in 0..NUM_LAYER_STACKS {
            assert_eq!(w.l2_weights[s].len(), L2_INPUT);
        }
        assert_ne!(w.l2_weights[0][0][0], w.l2_weights[1][0][0]);
        assert_ne!(w.skip_weights[0][0], 0.0);
    }

    #[test]
    fn test_load_bad_magic() {
        let mut buf = vec![0u8; 8];
        buf[0..4].copy_from_slice(b"XXXX");
        assert!(load_weights_from_bytes(&buf).is_err());
    }

    #[test]
    fn test_load_bad_version() {
        let mut buf = vec![0u8; 12];
        buf[0..4].copy_from_slice(b"NAGT");
        buf[4..8].copy_from_slice(&99u32.to_le_bytes());
        assert!(load_weights_from_bytes(&buf).is_err());
    }

    #[test]
    fn test_load_truncated() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"NAGT");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 16]);
        assert!(load_weights_from_bytes(&buf).is_err());
    }

    fn make_synthetic_weights(version: u32) -> NnueWeights {
        let l1_rows = if version == 1 { INPUT_SIZE } else { KING_BUCKETS * PER_BUCKET_FEATURES };
        NnueWeights {
            version,
            l1_weights: vec![[0.01f32; L1_SIZE]; l1_rows],
            l1_biases: [0.0f32; L1_SIZE],
            psqt_weights: vec![[0.0f32; NUM_PSQT_BUCKETS]; l1_rows],
            l2_weights: std::array::from_fn(|_| vec![[0.01f32; L2_SIZE]; L2_INPUT]),
            l2_biases: [[0.0f32; L2_SIZE]; NUM_LAYER_STACKS],
            output_weights: [[0.01f32; L2_SIZE]; NUM_LAYER_STACKS],
            output_bias: [0.0; NUM_LAYER_STACKS],
            skip_weights: [[0.0; SKIP_SIZE]; NUM_LAYER_STACKS],
        }
    }

    #[test]
    fn test_quantize_weights_ft() {
        let w = make_synthetic_weights(1);
        let q = quantize_weights(&w);
        assert_eq!(q.version, 1);
        let expected_ft = (0.01f32 * QA as f32).round() as i16;
        assert_eq!(q.ft_weights[0][0], expected_ft);
        assert_eq!(q.ft_biases[0], 0);
    }

    #[test]
    fn test_quantize_weights_l2() {
        let w = make_synthetic_weights(1);
        let q = quantize_weights(&w);
        let expected_l2 = (0.01f32 * QB as f32).round() as i8;
        assert_eq!(q.l2_weights[0][0][0], expected_l2);
        let expected_l2_bias = (0.0f32 * (QA as f32 * QB as f32)).round() as i32;
        assert_eq!(q.l2_biases[0][0], expected_l2_bias);
    }

    #[test]
    fn test_quantize_weights_output() {
        let w = make_synthetic_weights(1);
        let q = quantize_weights(&w);
        let expected_out_w = (0.01f32 * QB as f32).round() as i16;
        assert_eq!(q.out_weights[0][0], expected_out_w);
        assert_eq!(q.out_bias[0], 0);
    }

    #[test]
    fn test_forward_q_vs_forward() {
        let w = make_synthetic_weights(1);
        let q = quantize_weights(&w);
        let _ = NNUE_STATE.set(w);
        let _ = NNUE_STATE_Q.set(q);
        NNUE_LOADED.store(true, Ordering::Relaxed);

        let mut acc_f = Accumulator::new();
        for bucket in 0..KING_BUCKETS {
            acc_f.white[bucket] = [0.5; L1_SIZE];
            acc_f.black[bucket] = [0.3; L1_SIZE];
        }

        let mut acc_q = AccumulatorQ::new();
        for bucket in 0..KING_BUCKETS {
            for j in 0..L1_SIZE {
                acc_q.white[bucket][j] = (0.5 * QA as f32).round() as i16;
                acc_q.black[bucket][j] = (0.3 * QA as f32).round() as i16;
            }
        }

        let sq_bucket0 = 18u8; // C3 maps to bucket 0
        let f32_result = forward(&acc_f, Color::White, sq_bucket0, sq_bucket0, 0, 32);
        let q_result = forward_q(&acc_q, Color::White, sq_bucket0, sq_bucket0, 0, 32);

        let diff = (f32_result - q_result).abs();
        assert!(diff <= 100, "f32={} q={} diff={} (uniform 0.01 weights have high quant error)", f32_result, q_result, diff);

        NNUE_LOADED.store(false, Ordering::Relaxed);
    }

    #[test]
    fn test_forward_q_zero_acc() {
        let w = make_synthetic_weights(1);
        let q = quantize_weights(&w);
        let _ = NNUE_STATE.set(w);
        let _ = NNUE_STATE_Q.set(q);
        NNUE_LOADED.store(true, Ordering::Relaxed);

        let acc_q = AccumulatorQ::new();
        let sq_bucket0 = 18u8; // C3 maps to bucket 0
        let result = forward_q(&acc_q, Color::White, sq_bucket0, sq_bucket0, 0, 32);
        assert_eq!(result, 0);

        NNUE_LOADED.store(false, Ordering::Relaxed);
    }

    #[test]
    fn test_forward_q_exact_arithmetic() {
        let w = NnueWeights {
            version: 1,
            l1_weights: vec![[0.0f32; L1_SIZE]; INPUT_SIZE],
            l1_biases: [0.0f32; L1_SIZE],
            psqt_weights: vec![[0.0f32; NUM_PSQT_BUCKETS]; INPUT_SIZE],
            l2_weights: std::array::from_fn(|_| vec![[1.0 / QB as f32; L2_SIZE]; L2_INPUT]),
            l2_biases: [[0.0f32; L2_SIZE]; NUM_LAYER_STACKS],
            output_weights: [[1.0 / QB as f32; L2_SIZE]; NUM_LAYER_STACKS],
            output_bias: [0.0; NUM_LAYER_STACKS],
            skip_weights: [[0.0; SKIP_SIZE]; NUM_LAYER_STACKS],
        };
        let q = quantize_weights(&w);

        assert_eq!(q.l2_weights[0][0][0], 1i8);
        assert_eq!(q.out_weights[0][0], 1i16);

        let _ = NNUE_STATE.set(w);
        let _ = NNUE_STATE_Q.set(q);
        NNUE_LOADED.store(true, Ordering::Relaxed);

        let mut acc_q = AccumulatorQ::new();
        for j in 0..L1_SIZE {
            acc_q.white[0][j] = 200;
            acc_q.black[0][j] = 100;
        }

        let sq_bucket0 = 18u8; // C3 maps to bucket 0
        let result = forward_q(&acc_q, Color::White, sq_bucket0, sq_bucket0, 0, 32);
        let stm_pw: i32 = (200u16 as u32 * 200u16 as u32 >> 8) as i32;
        let opp_pw: i32 = (100u16 as u32 * 100u16 as u32 >> 8) as i32;
        let l2_dot = L1_PAIR as i32 * stm_pw * 1 + L1_PAIR as i32 * opp_pw * 1;
        let l2_crelu = std::cmp::min(l2_dot, QA * QB) / QA;
        let output_dot = L2_SIZE as i64 * l2_crelu as i64 * 1i64;
        let positional = (output_dot * 400 / (QB as i64 * QB as i64)) as i32;
        let expected = (PSQT_BETA as i64 * positional as i64) as i32 / PSQT_GAMMA;
        assert_eq!(result, expected, "result={} expected={}", result, expected);

        NNUE_LOADED.store(false, Ordering::Relaxed);
    }

    #[test]
    #[ignore]
    fn bench_forward_pass() {
        use std::time::Instant;
        let w = make_synthetic_weights(1);
        let _ = NNUE_STATE.set(w);
        NNUE_LOADED.store(true, Ordering::Relaxed);

        let mut acc = Accumulator::new();
        for bucket in 0..KING_BUCKETS {
            acc.white[bucket] = [0.5; L1_SIZE];
            acc.black[bucket] = [0.3; L1_SIZE];
        }

        let sq_bucket0 = 18u8; // C3 maps to bucket 0
        let iterations = 1_000_000;
        let start = Instant::now();
        let mut sum = 0i64;
        for _ in 0..iterations {
            sum += forward(&acc, Color::White, sq_bucket0, sq_bucket0, 0, 32) as i64;
        }
        let elapsed = start.elapsed();
        let ns_per = elapsed.as_nanos() as f64 / iterations as f64;
        println!("forward: {} iterations in {:.2?} ({:.0} ns/iter, {:.1} M evals/s) [sum={}]",
            iterations, elapsed, ns_per, 1e9 / ns_per / 1e6, sum);
    }

    #[test]
    #[ignore]
    fn bench_refresh_accumulator() {
        use std::time::Instant;
        use crate::board::Board;

        crate::zobrist::init();
        crate::movegen::init();

        let w = make_synthetic_weights(1);
        let _ = NNUE_STATE.set(w);
        NNUE_LOADED.store(true, Ordering::Relaxed);

        let board = Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
        let mut acc = Accumulator::new();

        let iterations = 500_000;
        let start = Instant::now();
        for _ in 0..iterations {
            crate::nnue::accumulator::refresh_accumulator(&board, &mut acc);
        }
        let elapsed = start.elapsed();
        let ns_per = elapsed.as_nanos() as f64 / iterations as f64;
        println!("refresh_accumulator: {} iterations in {:.2?} ({:.0} ns/iter, {:.1}K refreshes/s)",
            iterations, elapsed, ns_per, 1e9 / ns_per / 1e3);
    }

    #[test]
    #[ignore]
    fn bench_forward_q_pass() {
        use std::time::Instant;
        let w = make_synthetic_weights(1);
        let q = quantize_weights(&w);
        let _ = NNUE_STATE.set(w);
        let _ = NNUE_STATE_Q.set(q);
        NNUE_LOADED.store(true, Ordering::Relaxed);

        let mut acc_q = AccumulatorQ::new();
        for j in 0..L1_SIZE {
            acc_q.white[0][j] = (0.5 * QA as f32).round() as i16;
            acc_q.black[0][j] = (0.3 * QA as f32).round() as i16;
        }

        let sq_bucket0 = 18u8; // C3 maps to bucket 0
        let iterations = 1_000_000;
        let start = Instant::now();
        let mut sum = 0i64;
        for _ in 0..iterations {
            sum += forward_q(&acc_q, Color::White, sq_bucket0, sq_bucket0, 0, 32) as i64;
        }
        let elapsed = start.elapsed();
        let ns_per = elapsed.as_nanos() as f64 / iterations as f64;
        println!("forward_q: {} iterations in {:.2?} ({:.0} ns/iter, {:.1} M evals/s) [sum={}]",
            iterations, elapsed, ns_per, 1e9 / ns_per / 1e6, sum);
    }

    #[test]
    fn test_leb128_roundtrip() {
        for val in [0, 1, -1, 127, -128, 255, -256, 32767, -32768, i32::MAX, i32::MIN] {
            let mut buf = Vec::new();
            leb128_encode_i32(val, &mut buf);
            let mut cursor = 0;
            let decoded = leb128_decode_i32(&buf, &mut cursor).unwrap();
            assert_eq!(val, decoded, "LEB128 roundtrip failed for {}", val);
            assert_eq!(cursor, buf.len());
        }
    }

    #[test]
    fn test_leb128_small_values_compress() {
        let mut buf = Vec::new();
        leb128_encode_i32(0, &mut buf);
        assert_eq!(buf.len(), 1);
        buf.clear();
        leb128_encode_i32(63, &mut buf);
        assert_eq!(buf.len(), 1);
        buf.clear();
        leb128_encode_i32(-64, &mut buf);
        assert_eq!(buf.len(), 1);
    }
}
