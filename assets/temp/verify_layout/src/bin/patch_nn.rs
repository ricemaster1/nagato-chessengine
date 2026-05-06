// Patches an existing NAGT v4 nn.bin produced by the buggy trainer to the
// layout Nagato's runtime actually expects.
//
// Bugs in the trainer save format (now confirmed by verify_layout):
//   l0w: spurious .transpose() before permute_halves_rows — reverses input/neuron axes
//   l0b: spurious permute_halves_1d — shuffles bias across wrong neurons
//   l1w: spurious .transpose() with a col-major extraction formula — reads wrong bytes
//   l2w: missing .transpose() before per-bucket slice — produces column not row per bucket
//
// This patcher inverts the wrong transformations and writes a corrected file.

use std::{env, fs, io::{self, Read, Write}};

// ── NAGT v4 parameters ───────────────────────────────────────────────────────

const L1_SIZE: usize = 256;
const L1_PAIR: usize = L1_SIZE / 2;
const L2_INPUT: usize = L1_SIZE;  // 2 * L1_PAIR = L1_SIZE
const L2_SIZE: usize = 32;
const NUM_OUTPUT_BUCKETS: usize = 4;
const NUM_PSQT_BUCKETS: usize = 4;
const SKIP_SIZE: usize = 8;
const KING_BUCKETS: usize = 10;
const PER_BUCKET_FEATURES: usize = 768;
const FT_ROWS: usize = KING_BUCKETS * PER_BUCKET_FEATURES;  // 7680

// ── Math helpers (verbatim from trainer) ─────────────────────────────────────

fn halves_perm(out_idx: usize, pair: usize) -> usize {
    if out_idx < pair { 2 * out_idx } else { 2 * (out_idx - pair) + 1 }
}

fn inv_perm(j: usize, pair: usize) -> usize {
    if j % 2 == 0 { j / 2 } else { pair + (j - 1) / 2 }
}

fn un_permute_halves_rows(src: &[f32], row_size: usize) -> Vec<f32> {
    let pair = row_size / 2;
    let mut out = vec![0.0f32; src.len()];
    for r in 0..(src.len() / row_size) {
        let base = r * row_size;
        for k in 0..row_size {
            // Forward: saved[j] = original[halves_perm(j)]
            // Inverse: original[k] = saved[inv_perm(k)]
            out[base + k] = src[base + inv_perm(k, pair)];
        }
    }
    out
}

fn un_permute_halves_1d(src: &[f32]) -> Vec<f32> {
    let pair = src.len() / 2;
    (0..src.len()).map(|k| src[inv_perm(k, pair)]).collect()
}

// Applies transpose: converts between col-major and row-major for shape (rows, cols).
// Applying twice with same shape = identity.
fn transpose(buf: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(buf.len(), rows * cols);
    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[cols * i + j] = buf[rows * j + i];
        }
    }
    out
}

// ── Binary I/O ───────────────────────────────────────────────────────────────

fn read_f32s(buf: &[u8], offset: &mut usize, count: usize) -> Vec<f32> {
    let end = *offset + count * 4;
    let floats: Vec<f32> = buf[*offset..end]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    *offset = end;
    floats
}

fn write_f32s(out: &mut Vec<u8>, values: &[f32]) {
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let (input_path, output_path) = match args.as_slice() {
        [_, i, o] => (i.as_str(), o.as_str()),
        [_, i] => (i.as_str(), i.as_str()),
        _ => {
            eprintln!("usage: patch_nn <input.bin> [output.bin]");
            std::process::exit(1);
        }
    };

    let mut data = Vec::new();
    fs::File::open(input_path)?.read_to_end(&mut data)?;

    let mut cursor = 0usize;

    // Magic + version
    assert_eq!(&data[0..4], b"NAGT", "Not a NAGT file");
    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    if version != 4 {
        eprintln!("Error: only NAGT v4 files can be patched (file is v{version}).");
        eprintln!("The v4 file is on the machine where training ran. Copy it here first.");
        std::process::exit(1);
    }
    cursor += 8;

    println!("Patching NAGT v4 file: {input_path}");
    println!("  FT rows: {FT_ROWS}, L1: {L1_SIZE}, L2: {L2_SIZE}");
    println!("  Output buckets: {NUM_OUTPUT_BUCKETS}, PSQT buckets: {NUM_PSQT_BUCKETS}");

    let mut patched: Vec<u8> = Vec::with_capacity(data.len());
    patched.extend_from_slice(&data[0..8]);  // keep magic + version

    // ── l0w: (native → transpose → permute) → invert → native ───────────────
    {
        let raw = read_f32s(&data, &mut cursor, FT_ROWS * L1_SIZE);
        // Step 1: un-permute the halves (each row of L1_SIZE)
        let un_permed = un_permute_halves_rows(&raw, L1_SIZE);
        // Step 2: un-transpose from (L1_SIZE=256 rows, FT_ROWS=7680 cols) row-major
        //         back to col-major via transpose with swapped dimensions
        let native = transpose(&un_permed, FT_ROWS, L1_SIZE);
        // Verify shapes: native should be FT_ROWS*L1_SIZE floats, col-major (out fast, in slow)
        assert_eq!(native.len(), FT_ROWS * L1_SIZE);
        write_f32s(&mut patched, &native);
        println!("  l0w: un-permute + un-transpose applied ({} f32s)", native.len());
    }

    // ── l0b: (native → permute_halves_1d) → invert → native ─────────────────
    {
        let raw = read_f32s(&data, &mut cursor, L1_SIZE);
        let native = un_permute_halves_1d(&raw);
        write_f32s(&mut patched, &native);
        println!("  l0b: un-permute_halves_1d applied ({} f32s)", native.len());
    }

    // ── PSQT: zeros, copy as-is ──────────────────────────────────────────────
    {
        let psqt_size = FT_ROWS * NUM_PSQT_BUCKETS;
        let psqt = read_f32s(&data, &mut cursor, psqt_size);
        write_f32s(&mut patched, &psqt);
        println!("  psqt: copied as-is ({} f32s)", psqt.len());
    }

    // ── Per output-bucket stacks ─────────────────────────────────────────────

    // l1w: we need all 4 buckets' data to reconstruct and fix.
    // Collect all extracted bucket data first, then reconstruct.
    let mut l1w_extracted: Vec<Vec<f32>> = Vec::with_capacity(NUM_OUTPUT_BUCKETS);
    let mut l1b_data: Vec<Vec<f32>> = Vec::with_capacity(NUM_OUTPUT_BUCKETS);
    let mut l2w_data: Vec<Vec<f32>> = Vec::with_capacity(NUM_OUTPUT_BUCKETS);
    let mut l2b_data: Vec<Vec<f32>> = Vec::with_capacity(NUM_OUTPUT_BUCKETS);
    let mut skip_data: Vec<Vec<f32>> = Vec::with_capacity(NUM_OUTPUT_BUCKETS);

    for _ in 0..NUM_OUTPUT_BUCKETS {
        l1w_extracted.push(read_f32s(&data, &mut cursor, L2_INPUT * L2_SIZE));
        l1b_data.push(read_f32s(&data, &mut cursor, L2_SIZE));
        l2w_data.push(read_f32s(&data, &mut cursor, L2_SIZE));
        l2b_data.push(read_f32s(&data, &mut cursor, 1));
        skip_data.push(read_f32s(&data, &mut cursor, SKIP_SIZE));
    }

    // Reconstruct post-transpose l1w buffer from the wrong extraction.
    // Wrong extraction: extracted_s[in_i * L2_SIZE + j] = post_transpose[in_i * stride + s * L2_SIZE + j]
    // where stride = NUM_OUTPUT_BUCKETS * L2_SIZE = 128.
    // So: post_transpose[in_i * 128 + global_out] = extracted_{global_out/L2_SIZE}[in_i * L2_SIZE + global_out % L2_SIZE]
    let total_l1w = L2_INPUT * NUM_OUTPUT_BUCKETS * L2_SIZE;  // 256 * 128
    let stride = NUM_OUTPUT_BUCKETS * L2_SIZE;
    let mut post_transpose = vec![0.0f32; total_l1w];
    for s in 0..NUM_OUTPUT_BUCKETS {
        for in_i in 0..L2_INPUT {
            for j in 0..L2_SIZE {
                let global_out = s * L2_SIZE + j;
                post_transpose[in_i * stride + global_out] = l1w_extracted[s][in_i * L2_SIZE + j];
            }
        }
    }

    // Un-transpose to get native col-major of l1w (shape: output=128, input=256)
    let l1w_native = transpose(&post_transpose, L2_INPUT, stride);  // (256 rows, 128 cols) → native (128, 256)
    assert_eq!(l1w_native.len(), total_l1w);

    println!("  l1w: reconstructed post-transpose buffer, un-transposed, re-extracted per bucket");

    // l2w: reconstruct full L2-weight buffer from the 4 per-bucket raw slices (each is col-major
    // chunk), concatenate = full col-major matrix, then transpose to row-major for Nagato.
    let all_l2w: Vec<f32> = l2w_data.iter().flat_map(|b| b.iter().copied()).collect();
    let l2w_transposed = transpose(&all_l2w, NUM_OUTPUT_BUCKETS, L2_SIZE);
    println!("  l2w: concatenated {NUM_OUTPUT_BUCKETS} × {L2_SIZE} chunks, transposed, re-sliced");

    for s in 0..NUM_OUTPUT_BUCKETS {
        // l1w correct per-bucket (borrow l1w_native, no move)
        let mut correct_l1w = Vec::with_capacity(L2_INPUT * L2_SIZE);
        for in_i in 0..L2_INPUT {
            for j in 0..L2_SIZE {
                let global_out = s * L2_SIZE + j;
                correct_l1w.push(l1w_native[in_i * stride + global_out]);
            }
        }
        write_f32s(&mut patched, &correct_l1w);
        write_f32s(&mut patched, &l1b_data[s]);
        write_f32s(&mut patched, &l2w_transposed[s * L2_SIZE..(s + 1) * L2_SIZE]);
        write_f32s(&mut patched, &l2b_data[s]);
        write_f32s(&mut patched, &skip_data[s]);
    }

    assert_eq!(patched.len(), data.len(),
        "Size mismatch: patched={} original={}", patched.len(), data.len());

    fs::File::create(output_path)?.write_all(&patched)?;
    println!("Written to {output_path} ({} bytes)", patched.len());

    Ok(())
}
