use crate::bitboard::Color;
use crate::board::Board;
use crate::nnue::features::{KING_BUCKETS, PER_BUCKET_FEATURES};

use super::{L1_SIZE, L2_SIZE, INPUT_SIZE};
use super::accumulator::Accumulator;

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

static NNUE_STATE: OnceLock<NnueWeights> = OnceLock::new();
static NNUE_LOADED: AtomicBool = AtomicBool::new(false);

pub fn init() {
    let path = std::path::Path::new("nn.bin");
    if path.exists() {
        match load_weights_from_file(path) {
            Ok(w) => {
                let _ = NNUE_STATE.set(w);
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
}
