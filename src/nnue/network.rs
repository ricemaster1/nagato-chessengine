use crate::bitboard::Color;
use crate::board::Board;
use crate::nnue::features::{KING_BUCKETS, PER_BUCKET_FEATURES};

use super::{L1_SIZE, L2_SIZE, INPUT_SIZE, QA, QB};
use super::accumulator::{Accumulator, AccumulatorQ};

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

pub struct NnueWeights {
    pub l1_weights: Vec<[f32; L1_SIZE]>,
    pub version: u32,
    pub l1_biases: [f32; L1_SIZE],
    pub l2_weights: Vec<[f32; L2_SIZE]>,
    pub l2_biases: [f32; L2_SIZE],
    pub output_weights: [f32; L2_SIZE],
    pub output_bias: f32,
}

pub struct NnueWeightsQ {
    pub version: u32,
    pub ft_weights: Vec<[i16; L1_SIZE]>,
    pub ft_biases: [i16; L1_SIZE],
    pub l2_weights: Vec<[i8; L2_SIZE]>,
    pub l2_biases: [i32; L2_SIZE],
    pub out_weights: [i16; L2_SIZE],
    pub out_bias: i32,
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

    let concat = 2 * L1_SIZE;
    let mut l2_weights = vec![[0i8; L2_SIZE]; concat];
    for i in 0..concat {
        for j in 0..L2_SIZE {
            l2_weights[i][j] = (w.l2_weights[i][j] * qb).round().clamp(i8::MIN as f32, i8::MAX as f32) as i8;
        }
    }

    let mut l2_biases = [0i32; L2_SIZE];
    for j in 0..L2_SIZE {
        l2_biases[j] = (w.l2_biases[j] * qa_qb).round() as i32;
    }

    let mut out_weights = [0i16; L2_SIZE];
    for j in 0..L2_SIZE {
        out_weights[j] = (w.output_weights[j] * qb).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
    }

    let out_bias = (w.output_bias * qb_qb).round() as i32;

    NnueWeightsQ {
        version: w.version,
        ft_weights,
        ft_biases,
        l2_weights,
        l2_biases,
        out_weights,
        out_bias,
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
    let mut cursor = 0usize;

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
    cursor = 4;

    let version = read_u32(&mut cursor, data)?;
    if version != 1 && version != 2 {
        return Err(format!("unsupported version: {}", version));
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

    let concat_size = 2 * L1_SIZE;
    let mut l2_weights = vec![[0.0f32; L2_SIZE]; concat_size];
    for i in 0..concat_size {
        for j in 0..L2_SIZE {
            l2_weights[i][j] = read_f32(&mut cursor, data)?;
        }
    }

    let mut l2_biases = [0.0f32; L2_SIZE];
    for j in 0..L2_SIZE {
        l2_biases[j] = read_f32(&mut cursor, data)?;
    }

    let mut output_weights = [0.0f32; L2_SIZE];
    for j in 0..L2_SIZE {
        output_weights[j] = read_f32(&mut cursor, data)?;
    }

    let output_bias = read_f32(&mut cursor, data)?;

    let expected = 4 + 4
        + (l1_rows * L1_SIZE) * 4
        + L1_SIZE * 4
        + (concat_size * L2_SIZE) * 4
        + L2_SIZE * 4
        + L2_SIZE * 4
        + 4;
    if cursor != expected {
        return Err(format!("size mismatch: read {} expected {}", cursor, expected));
    }

    Ok(NnueWeights {
        version,
        l1_weights,
        l1_biases,
        l2_weights,
        l2_biases,
        output_weights,
        output_bias,
    })
}

#[inline]
fn clipped_relu(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

pub fn forward(acc: &Accumulator, side: Color) -> i32 {
    let w = weights();

    let (stm_acc, opp_acc) = match side {
        Color::White => (&acc.white, &acc.black),
        Color::Black => (&acc.black, &acc.white),
    };

    let mut l2_out = w.l2_biases;

    for i in 0..L1_SIZE {
        let activated = clipped_relu(stm_acc[i]);
        if activated != 0.0 {
            for j in 0..L2_SIZE {
                l2_out[j] += activated * w.l2_weights[i][j];
            }
        }
    }

    for i in 0..L1_SIZE {
        let activated = clipped_relu(opp_acc[i]);
        if activated != 0.0 {
            for j in 0..L2_SIZE {
                l2_out[j] += activated * w.l2_weights[L1_SIZE + i][j];
            }
        }
    }

    let mut output = w.output_bias;
    for j in 0..L2_SIZE {
        output += clipped_relu(l2_out[j]) * w.output_weights[j];
    }

    (output * 400.0) as i32
}

pub fn evaluate(board: &Board, acc: &Accumulator) -> i32 {
    forward(acc, board.side)
}

pub fn forward_q(acc: &AccumulatorQ, side: Color) -> i32 {
    let wq = weights_q();
    let qa = QA;
    let qa_qb = QA * QB;

    let (stm_acc, opp_acc) = match side {
        Color::White => (&acc.white, &acc.black),
        Color::Black => (&acc.black, &acc.white),
    };

    let mut l2_out = wq.l2_biases;

    for i in 0..L1_SIZE {
        let clamped = stm_acc[i].max(0).min(qa as i16) as u8;
        if clamped != 0 {
            for j in 0..L2_SIZE {
                l2_out[j] += clamped as i32 * wq.l2_weights[i][j] as i32;
            }
        }
    }

    for i in 0..L1_SIZE {
        let clamped = opp_acc[i].max(0).min(qa as i16) as u8;
        if clamped != 0 {
            for j in 0..L2_SIZE {
                l2_out[j] += clamped as i32 * wq.l2_weights[L1_SIZE + i][j] as i32;
            }
        }
    }

    let mut output = wq.out_bias as i64;
    for j in 0..L2_SIZE {
        let activated = l2_out[j].max(0).min(qa_qb) / qa;
        output += activated as i64 * wq.out_weights[j] as i64;
    }

    (output * 400 / (QB as i64 * QB as i64)) as i32
}

pub fn evaluate_q(board: &Board, acc: &AccumulatorQ) -> i32 {
    forward_q(acc, board.side)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clipped_relu() {
        assert_eq!(clipped_relu(-1.0), 0.0);
        assert_eq!(clipped_relu(0.0), 0.0);
        assert_eq!(clipped_relu(0.5), 0.5);
        assert_eq!(clipped_relu(1.0), 1.0);
        assert_eq!(clipped_relu(2.0), 1.0);
    }

    #[test]
    fn test_weight_file_size() {
        let concat_size = 2 * L1_SIZE;
        let expected_floats = INPUT_SIZE * L1_SIZE
            + L1_SIZE
            + concat_size * L2_SIZE
            + L2_SIZE
            + L2_SIZE
            + 1;
        let expected_bytes = 8 + expected_floats * 4;
        assert_eq!(expected_bytes, 426_764);
    }

    #[test]
    fn test_load_v1_roundtrip() {
        let l1_rows = INPUT_SIZE;
        let concat = 2 * L1_SIZE;
        let total_floats = l1_rows * L1_SIZE + L1_SIZE + concat * L2_SIZE + L2_SIZE + L2_SIZE + 1;
        let mut buf: Vec<u8> = Vec::with_capacity(8 + total_floats * 4);
        buf.extend_from_slice(b"NAGT");
        buf.extend_from_slice(&1u32.to_le_bytes());
        for i in 0..total_floats {
            buf.extend_from_slice(&(i as f32 * 0.001).to_le_bytes());
        }
        let w = load_weights_from_bytes(&buf).expect("v1 load failed");
        assert_eq!(w.version, 1);
        assert_eq!(w.l1_weights.len(), l1_rows);
        assert_eq!(w.l2_weights.len(), concat);
        let first_l1 = w.l1_weights[0][0];
        assert!((first_l1 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_load_v2_roundtrip() {
        let l1_rows = KING_BUCKETS * PER_BUCKET_FEATURES;
        let concat = 2 * L1_SIZE;
        let total_floats = l1_rows * L1_SIZE + L1_SIZE + concat * L2_SIZE + L2_SIZE + L2_SIZE + 1;
        let mut buf: Vec<u8> = Vec::with_capacity(8 + total_floats * 4);
        buf.extend_from_slice(b"NAGT");
        buf.extend_from_slice(&2u32.to_le_bytes());
        for i in 0..total_floats {
            buf.extend_from_slice(&(i as f32 * 0.0001).to_le_bytes());
        }
        let w = load_weights_from_bytes(&buf).expect("v2 load failed");
        assert_eq!(w.version, 2);
        assert_eq!(w.l1_weights.len(), l1_rows);
        assert_eq!(w.l2_weights.len(), concat);
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
        let concat = 2 * L1_SIZE;
        NnueWeights {
            version,
            l1_weights: vec![[0.01f32; L1_SIZE]; l1_rows],
            l1_biases: [0.0f32; L1_SIZE],
            l2_weights: vec![[0.01f32; L2_SIZE]; concat],
            l2_biases: [0.0f32; L2_SIZE],
            output_weights: [0.01f32; L2_SIZE],
            output_bias: 0.0,
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
        assert_eq!(q.l2_weights[0][0], expected_l2);
        let expected_l2_bias = (0.0f32 * (QA as f32 * QB as f32)).round() as i32;
        assert_eq!(q.l2_biases[0], expected_l2_bias);
    }

    #[test]
    fn test_quantize_weights_output() {
        let w = make_synthetic_weights(1);
        let q = quantize_weights(&w);
        let expected_out_w = (0.01f32 * QB as f32).round() as i16;
        assert_eq!(q.out_weights[0], expected_out_w);
        assert_eq!(q.out_bias, 0);
    }

    #[test]
    fn test_forward_q_vs_forward() {
        let w = make_synthetic_weights(1);
        let q = quantize_weights(&w);
        let _ = NNUE_STATE.set(w);
        let _ = NNUE_STATE_Q.set(q);
        NNUE_LOADED.store(true, Ordering::Relaxed);

        let mut acc_f = Accumulator::new();
        acc_f.white = [0.5; L1_SIZE];
        acc_f.black = [0.3; L1_SIZE];

        let mut acc_q = AccumulatorQ::new();
        for j in 0..L1_SIZE {
            acc_q.white[j] = (0.5 * QA as f32).round() as i16;
            acc_q.black[j] = (0.3 * QA as f32).round() as i16;
        }

        let f32_result = forward(&acc_f, Color::White);
        let q_result = forward_q(&acc_q, Color::White);

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
        let result = forward_q(&acc_q, Color::White);
        assert_eq!(result, 0);

        NNUE_LOADED.store(false, Ordering::Relaxed);
    }

    #[test]
    fn test_forward_q_exact_arithmetic() {
        let concat = 2 * L1_SIZE;
        let w = NnueWeights {
            version: 1,
            l1_weights: vec![[0.0f32; L1_SIZE]; INPUT_SIZE],
            l1_biases: [0.0f32; L1_SIZE],
            l2_weights: vec![[1.0 / QB as f32; L2_SIZE]; concat],
            l2_biases: [0.0f32; L2_SIZE],
            output_weights: [1.0 / QB as f32; L2_SIZE],
            output_bias: 0.0,
        };
        let q = quantize_weights(&w);

        assert_eq!(q.l2_weights[0][0], 1i8);
        assert_eq!(q.out_weights[0], 1i16);

        let _ = NNUE_STATE.set(w);
        let _ = NNUE_STATE_Q.set(q);
        NNUE_LOADED.store(true, Ordering::Relaxed);

        let mut acc_q = AccumulatorQ::new();
        for j in 0..L1_SIZE {
            acc_q.white[j] = 20;
            acc_q.black[j] = 10;
        }

        let result = forward_q(&acc_q, Color::White);
        let l2_dot = 128 * 20 * 1 + 128 * 10 * 1;
        let l2_crelu = std::cmp::min(l2_dot, QA * QB) / QA;
        let output_dot = L2_SIZE as i64 * l2_crelu as i64 * 1i64;
        let expected = (output_dot * 400 / (QB as i64 * QB as i64)) as i32;
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
        acc.white = [0.5; L1_SIZE];
        acc.black = [0.3; L1_SIZE];

        let iterations = 1_000_000;
        let start = Instant::now();
        let mut sum = 0i64;
        for _ in 0..iterations {
            sum += forward(&acc, Color::White) as i64;
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
            acc_q.white[j] = (0.5 * QA as f32).round() as i16;
            acc_q.black[j] = (0.3 * QA as f32).round() as i16;
        }

        let iterations = 1_000_000;
        let start = Instant::now();
        let mut sum = 0i64;
        for _ in 0..iterations {
            sum += forward_q(&acc_q, Color::White) as i64;
        }
        let elapsed = start.elapsed();
        let ns_per = elapsed.as_nanos() as f64 / iterations as f64;
        println!("forward_q: {} iterations in {:.2?} ({:.0} ns/iter, {:.1} M evals/s) [sum={}]",
            iterations, elapsed, ns_per, 1e9 / ns_per / 1e6, sum);
    }
}
