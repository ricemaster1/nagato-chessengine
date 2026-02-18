/// NNUE (Efficiently Updatable Neural Network) evaluation module.
///
/// Architecture: "Baby NNUE" — Phase 1
///   Input:  768 piece-square features (color × piece × square)
///   Layer1: 768 → HIDDEN (per-perspective, ClippedReLU)
///   Concat: 2 × HIDDEN from both perspectives
///   Layer2: 2*HIDDEN → L2_SIZE (ClippedReLU)
///   Output: L2_SIZE → 1 (linear)
///
/// Features are encoded as: color * 384 + piece * 64 + square
/// White's perspective uses the raw encoding.
/// Black's perspective flips color and mirrors the square vertically.
///
/// The accumulator is incrementally updated on make/unmake for efficiency.

use crate::bitboard::*;
use crate::board::Board;

// ============================================================
// Network dimensions
// ============================================================

/// First hidden layer size (per perspective)
pub const L1_SIZE: usize = 128;
/// Second hidden layer size
pub const L2_SIZE: usize = 32;
/// Input feature count (2 colors × 6 pieces × 64 squares)
pub const INPUT_SIZE: usize = 768;

// ============================================================
// Accumulator — one per perspective (white/black)
// ============================================================

/// The accumulator stores the pre-activation values for layer 1.
/// It can be incrementally updated when pieces move.
#[derive(Clone)]
pub struct Accumulator {
    /// Layer 1 values for white's perspective
    pub white: [f32; L1_SIZE],
    /// Layer 1 values for black's perspective
    pub black: [f32; L1_SIZE],
}

impl Accumulator {
    pub fn new() -> Self {
        Accumulator {
            white: [0.0; L1_SIZE],
            black: [0.0; L1_SIZE],
        }
    }
}

// ============================================================
// Feature index encoding
// ============================================================

/// Compute the feature index for a piece on a square from a given perspective.
///
/// From white's perspective:
///   white pieces: piece * 64 + sq
///   black pieces: 384 + piece * 64 + sq
///
/// From black's perspective (mirror):
///   black pieces: piece * 64 + flip(sq)
///   white pieces: 384 + piece * 64 + flip(sq)
#[inline]
pub fn feature_index_white(piece: Piece, color: Color, sq: u8) -> usize {
    let color_offset = match color {
        Color::White => 0,
        Color::Black => 384,
    };
    color_offset + piece.index() * 64 + sq as usize
}

#[inline]
pub fn feature_index_black(piece: Piece, color: Color, sq: u8) -> usize {
    let flipped = sq ^ 56; // vertical mirror
    let color_offset = match color {
        Color::Black => 0,   // black's own pieces first
        Color::White => 384, // opponent's pieces second
    };
    color_offset + piece.index() * 64 + flipped as usize
}

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

/// Global singleton for the loaded NNUE weights.
/// None if NNUE is not loaded (falls back to HCE).
use std::sync::OnceLock;

static NNUE_STATE: OnceLock<NnueWeights> = OnceLock::new();
static NNUE_LOADED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Initialize NNUE from embedded weights or external file.
/// Call this once at startup.
pub fn init() {
    // Try to load from the default embedded path or external file.
    // If no weights file exists, NNUE remains inactive and HCE is used.
    let path = std::path::Path::new("nn.bin");
    if path.exists() {
        match load_weights_from_file(path) {
            Ok(weights) => {
                let _ = NNUE_STATE.set(weights);
                NNUE_LOADED.store(true, std::sync::atomic::Ordering::Relaxed);
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

/// Check if NNUE is loaded and active
#[inline]
pub fn is_active() -> bool {
    NNUE_LOADED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Get a reference to the loaded weights (panics if not active)
#[inline]
fn weights() -> &'static NnueWeights {
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
// Accumulator operations
// ============================================================

/// Compute the full accumulator from scratch for a given board position.
/// This is used when setting up a new position (e.g., from FEN or at the start).
pub fn refresh_accumulator(board: &Board, acc: &mut Accumulator) {
    let w = weights();

    // Start with biases
    acc.white = w.l1_biases;
    acc.black = w.l1_biases;

    // Add each piece's feature contribution
    for color_idx in 0..COLOR_COUNT {
        let color = if color_idx == 0 { Color::White } else { Color::Black };
        for piece_idx in 0..PIECE_COUNT {
            let piece: Piece = unsafe { std::mem::transmute(piece_idx as u8) };
            let mut bb = board.pieces[color_idx][piece_idx];
            while bb != 0 {
                let sq = pop_lsb(&mut bb);
                let wi = feature_index_white(piece, color, sq);
                let bi = feature_index_black(piece, color, sq);
                for j in 0..L1_SIZE {
                    acc.white[j] += w.l1_weights[wi][j];
                    acc.black[j] += w.l1_weights[bi][j];
                }
            }
        }
    }
}

/// Incrementally add a piece feature to the accumulator.
#[inline]
pub fn accumulator_add(acc: &mut Accumulator, piece: Piece, color: Color, sq: u8) {
    let w = weights();
    let wi = feature_index_white(piece, color, sq);
    let bi = feature_index_black(piece, color, sq);
    for j in 0..L1_SIZE {
        acc.white[j] += w.l1_weights[wi][j];
        acc.black[j] += w.l1_weights[bi][j];
    }
}

/// Incrementally remove a piece feature from the accumulator.
#[inline]
pub fn accumulator_remove(acc: &mut Accumulator, piece: Piece, color: Color, sq: u8) {
    let w = weights();
    let wi = feature_index_white(piece, color, sq);
    let bi = feature_index_black(piece, color, sq);
    for j in 0..L1_SIZE {
        acc.white[j] -= w.l1_weights[wi][j];
        acc.black[j] -= w.l1_weights[bi][j];
    }
}

/// Incrementally move a piece (remove from `from`, add to `to`).
#[inline]
pub fn accumulator_move(acc: &mut Accumulator, piece: Piece, color: Color, from: u8, to: u8) {
    let w = weights();
    let wi_from = feature_index_white(piece, color, from);
    let wi_to   = feature_index_white(piece, color, to);
    let bi_from = feature_index_black(piece, color, from);
    let bi_to   = feature_index_black(piece, color, to);
    for j in 0..L1_SIZE {
        acc.white[j] += w.l1_weights[wi_to][j] - w.l1_weights[wi_from][j];
        acc.black[j] += w.l1_weights[bi_to][j] - w.l1_weights[bi_from][j];
    }
}

// ============================================================
// Forward pass — evaluate from the accumulator
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
// High-level evaluation — called from eval.rs
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
    fn test_feature_index_bounds() {
        // White pawn on a1 from white's perspective
        let idx = feature_index_white(Piece::Pawn, Color::White, 0);
        assert_eq!(idx, 0);

        // Black king on h8 from white's perspective
        let idx = feature_index_white(Piece::King, Color::Black, 63);
        assert_eq!(idx, 384 + 5 * 64 + 63);
        assert!(idx < INPUT_SIZE);

        // From black's perspective: black pawn on a8 (sq=56), after flip = sq=0
        let idx = feature_index_black(Piece::Pawn, Color::Black, 56);
        assert_eq!(idx, 0 * 64 + 0); // color_offset=0, piece=0, flipped=0

        // White king on e1 (sq=4) from black's perspective: flip = 60
        let idx = feature_index_black(Piece::King, Color::White, 4);
        assert_eq!(idx, 384 + 5 * 64 + 60);
        assert!(idx < INPUT_SIZE);
    }

    #[test]
    fn test_feature_index_symmetry() {
        // A white pawn on e2 from white's perspective should map to the same
        // feature as a black pawn on e7 from black's perspective (mirrored).
        let w_idx = feature_index_white(Piece::Pawn, Color::White, sq::E2);
        let b_idx = feature_index_black(Piece::Pawn, Color::Black, sq::E7);
        assert_eq!(w_idx, b_idx, "Symmetric positions should have same feature index");
    }

    #[test]
    fn test_accumulator_new() {
        let acc = Accumulator::new();
        assert!(acc.white.iter().all(|&v| v == 0.0));
        assert!(acc.black.iter().all(|&v| v == 0.0));
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
