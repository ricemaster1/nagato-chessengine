//! NNUE network weights, loading, and forward pass.

use crate::bitboard::Color;
use crate::board::Board;

use super::{L1_SIZE, L2_SIZE, INPUT_SIZE};
use super::accumulator::Accumulator;

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

// ============================================================
// Network weights
// ============================================================

/// The NNUE network weights, loaded from a binary file.
pub struct NnueWeights {
    /// Layer 1 weights: INPUT_SIZE × L1_SIZE (feature transform, shared for both perspectives)
    pub l1_weights: Vec<[f32; L1_SIZE]>, // indexed by [feature_idx][hidden_idx]
    /// Layer 1 biases: L1_SIZE
    pub l1_biases: [f32; L1_SIZE],
    /// Layer 2 weights: (2 * L1_SIZE) × L2_SIZE
    pub l2_weights: Vec<[f32; L2_SIZE]>, // indexed by [concat_idx][l2_idx]
    /// Layer 2 biases: L2_SIZE
    pub l2_biases: [f32; L2_SIZE],
    /// Output weights: L2_SIZE → 1
    pub output_weights: [f32; L2_SIZE],
    /// Output bias
    pub output_bias: f32,
}

// ============================================================
// Global state
// ============================================================

static NNUE_STATE: OnceLock<NnueWeights> = OnceLock::new();
static NNUE_LOADED: AtomicBool = AtomicBool::new(false);

/// Initialize NNUE from nn.bin. Call once at startup.
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

/// Check if NNUE is loaded and active.
#[inline]
pub fn is_active() -> bool {
    NNUE_LOADED.load(Ordering::Relaxed)
}

/// Get a reference to the loaded weights (panics if not loaded).
#[inline]
pub(super) fn weights() -> &'static NnueWeights {
    NNUE_STATE.get().unwrap()
}

// ============================================================
// Weight loading — binary format
// ============================================================

/// Binary format (little-endian f32):
///   - Magic: b"NAGT" (4 bytes)
///   - Version: u32 (4 bytes)
///   - L1 weights: INPUT_SIZE * L1_SIZE floats
///   - L1 biases: L1_SIZE floats
///   - L2 weights: (2 * L1_SIZE) * L2_SIZE floats
///   - L2 biases: L2_SIZE floats
///   - Output weights: L2_SIZE floats
///   - Output bias: 1 float
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

    // Magic
    if data.len() < 8 {
        return Err("file too small".into());
    }
    if &data[0..4] != b"NAGT" {
        return Err("bad magic".into());
    }
    cursor = 4;

    // Version
    let version = read_u32(&mut cursor, data)?;
    if version != 1 {
        return Err(format!("unsupported version: {}", version));
    }

    // L1 weights: INPUT_SIZE rows × L1_SIZE columns
    let mut l1_weights = vec![[0.0f32; L1_SIZE]; INPUT_SIZE];
    for i in 0..INPUT_SIZE {
        for j in 0..L1_SIZE {
            l1_weights[i][j] = read_f32(&mut cursor, data)?;
        }
    }

    // L1 biases
    let mut l1_biases = [0.0f32; L1_SIZE];
    for j in 0..L1_SIZE {
        l1_biases[j] = read_f32(&mut cursor, data)?;
    }

    // L2 weights: (2 * L1_SIZE) rows × L2_SIZE columns
    let concat_size = 2 * L1_SIZE;
    let mut l2_weights = vec![[0.0f32; L2_SIZE]; concat_size];
    for i in 0..concat_size {
        for j in 0..L2_SIZE {
            l2_weights[i][j] = read_f32(&mut cursor, data)?;
        }
    }

    // L2 biases
    let mut l2_biases = [0.0f32; L2_SIZE];
    for j in 0..L2_SIZE {
        l2_biases[j] = read_f32(&mut cursor, data)?;
    }

    // Output weights
    let mut output_weights = [0.0f32; L2_SIZE];
    for j in 0..L2_SIZE {
        output_weights[j] = read_f32(&mut cursor, data)?;
    }

    // Output bias
    let output_bias = read_f32(&mut cursor, data)?;

    // Verify we consumed the right amount
    let expected = 4 + 4 // magic + version
        + (INPUT_SIZE * L1_SIZE) * 4  // l1 weights
        + L1_SIZE * 4                  // l1 biases
        + (concat_size * L2_SIZE) * 4  // l2 weights
        + L2_SIZE * 4                  // l2 biases
        + L2_SIZE * 4                  // output weights
        + 4;                           // output bias
    if cursor != expected {
        return Err(format!("size mismatch: read {} expected {}", cursor, expected));
    }

    Ok(NnueWeights {
        l1_weights,
        l1_biases,
        l2_weights,
        l2_biases,
        output_weights,
        output_bias,
    })
}

// ============================================================
// Forward pass
// ============================================================

/// ClippedReLU activation: clamp(x, 0, 1)
#[inline]
fn clipped_relu(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Run the forward pass from the accumulated layer-1 values.
/// `side` is the side to move — determines which perspective goes first in the concat.
///
/// Returns the evaluation score in centipawns (from the side-to-move's perspective).
pub fn forward(acc: &Accumulator, side: Color) -> i32 {
    let w = weights();

    // Determine perspective order: STM first, then opponent
    let (stm_acc, opp_acc) = match side {
        Color::White => (&acc.white, &acc.black),
        Color::Black => (&acc.black, &acc.white),
    };

    // Layer 2 input: ClippedReLU of [stm_accumulator | opp_accumulator]
    let mut l2_out = w.l2_biases;

    // STM perspective (indices 0..L1_SIZE)
    for i in 0..L1_SIZE {
        let activated = clipped_relu(stm_acc[i]);
        if activated != 0.0 {
            for j in 0..L2_SIZE {
                l2_out[j] += activated * w.l2_weights[i][j];
            }
        }
    }

    // Opponent perspective (indices L1_SIZE..2*L1_SIZE)
    for i in 0..L1_SIZE {
        let activated = clipped_relu(opp_acc[i]);
        if activated != 0.0 {
            for j in 0..L2_SIZE {
                l2_out[j] += activated * w.l2_weights[L1_SIZE + i][j];
            }
        }
    }

    // Output layer: dot product with ClippedReLU of l2
    let mut output = w.output_bias;
    for j in 0..L2_SIZE {
        output += clipped_relu(l2_out[j]) * w.output_weights[j];
    }

    // Scale to centipawns (the network outputs are trained in a [0,1]-ish range
    // from the sigmoid — we scale by 400 to get centipawns)
    (output * 400.0) as i32
}

// ============================================================
// High-level evaluation
// ============================================================

/// Evaluate the position using NNUE. Requires a pre-computed accumulator.
/// Returns score in centipawns from the side-to-move's perspective.
pub fn evaluate(board: &Board, acc: &Accumulator) -> i32 {
    forward(acc, board.side)
}

// ============================================================
// Tests
// ============================================================

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
        // Verify expected binary file size
        let concat_size = 2 * L1_SIZE;
        let expected_floats = INPUT_SIZE * L1_SIZE  // l1 weights
            + L1_SIZE                               // l1 biases
            + concat_size * L2_SIZE                 // l2 weights
            + L2_SIZE                               // l2 biases
            + L2_SIZE                               // output weights
            + 1;                                    // output bias
        let expected_bytes = 8 + expected_floats * 4; // 8 = magic + version
        // 768*128 + 128 + 256*32 + 32 + 32 + 1 = 98_304 + 128 + 8_192 + 32 + 32 + 1 = 106_689 floats
        // 8 + 106_689 * 4 = 426_764 bytes (~417 KB)
        assert_eq!(expected_bytes, 426_764);
    }
}
