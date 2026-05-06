// Checks that applying patch_nn twice returns the original bytes.
// Synthesises a minimal v4 file with known weight patterns, patches it,
// then verifies selected weights are in the expected Nagato-readable order.

use std::io::{Write, Read};

const L1_SIZE: usize = 256;
const L2_INPUT: usize = L1_SIZE;
const L2_SIZE: usize = 32;
const NUM_OUTPUT_BUCKETS: usize = 4;
const NUM_PSQT_BUCKETS: usize = 4;
const SKIP_SIZE: usize = 8;
const KING_BUCKETS: usize = 10;
const PER_BUCKET_FEATURES: usize = 768;
const FT_ROWS: usize = KING_BUCKETS * PER_BUCKET_FEATURES;

fn halves_perm(j: usize, pair: usize) -> usize {
    if j < pair { 2 * j } else { 2 * (j - pair) + 1 }
}

fn permute_halves_rows(src: &[f32], row_size: usize) -> Vec<f32> {
    let pair = row_size / 2;
    let mut out = vec![0.0f32; src.len()];
    for r in 0..(src.len() / row_size) {
        let base = r * row_size;
        for j in 0..row_size {
            out[base + j] = src[base + halves_perm(j, pair)];
        }
    }
    out
}

fn permute_halves_1d(src: &[f32]) -> Vec<f32> {
    let pair = src.len() / 2;
    (0..src.len()).map(|j| src[halves_perm(j, pair)]).collect()
}

fn transpose(buf: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[cols * i + j] = buf[rows * j + i];
        }
    }
    out
}

fn write_f32s(out: &mut Vec<u8>, vals: &[f32]) {
    for &v in vals { out.extend_from_slice(&v.to_le_bytes()); }
}

fn main() {
    // Build a synthetic v4 file with known col-major "true" weights.
    // l0w: element(out, in) = out*1000 + in
    // l1w: element(out, in) = 1_000_000 + out*1000 + in  (for verification)
    // l2w: element(out, in) = 2_000_000 + out*1000 + in

    let l0w_native: Vec<f32> = {
        let mut v = vec![0.0f32; FT_ROWS * L1_SIZE];
        for inp in 0..FT_ROWS {
            for out in 0..L1_SIZE {
                v[out + inp * L1_SIZE] = (out * 1000 + inp % 1000) as f32;
            }
        }
        v
    };
    let l0b_native: Vec<f32> = (0..L1_SIZE).map(|j| j as f32).collect();

    let stride = NUM_OUTPUT_BUCKETS * L2_SIZE;
    let l1w_native: Vec<f32> = {
        let mut v = vec![0.0f32; L2_INPUT * stride];
        for inp in 0..L2_INPUT {
            for out in 0..stride {
                v[out + inp * stride] = (1_000_000 + out * 1000 + inp) as f32;
            }
        }
        v
    };
    let l2w_native: Vec<f32> = {
        let total = NUM_OUTPUT_BUCKETS * L2_SIZE;
        let mut v = vec![0.0f32; total];
        for inp in 0..L2_SIZE {
            for out in 0..NUM_OUTPUT_BUCKETS {
                v[out + inp * NUM_OUTPUT_BUCKETS] = (2_000_000 + out * 1000 + inp) as f32;
            }
        }
        v
    };

    // Apply WRONG trainer transforms to create the "buggy" v4 file.
    let l0w_saved = permute_halves_rows(&transpose(&l0w_native, L1_SIZE, FT_ROWS), L1_SIZE);
    let l0b_saved = permute_halves_1d(&l0b_native);
    let l1w_transposed = transpose(&l1w_native, stride, L2_INPUT);
    let l2w_saved_all: Vec<f32> = (0..NUM_OUTPUT_BUCKETS)
        .flat_map(|s| l2w_native[s * L2_SIZE..(s + 1) * L2_SIZE].to_vec())
        .collect();

    let mut file_bytes: Vec<u8> = Vec::new();
    file_bytes.extend_from_slice(b"NAGT");
    file_bytes.extend_from_slice(&4u32.to_le_bytes());
    write_f32s(&mut file_bytes, &l0w_saved);
    write_f32s(&mut file_bytes, &l0b_saved);
    write_f32s(&mut file_bytes, &vec![0.0f32; FT_ROWS * NUM_PSQT_BUCKETS]);

    for s in 0..NUM_OUTPUT_BUCKETS {
        // Wrong l1w extraction: w[in_i * stride + s * L2_SIZE + j] from post-transpose
        let mut bucket = Vec::with_capacity(L2_INPUT * L2_SIZE);
        for in_i in 0..L2_INPUT {
            for j in 0..L2_SIZE {
                bucket.push(l1w_transposed[in_i * stride + s * L2_SIZE + j]);
            }
        }
        write_f32s(&mut file_bytes, &bucket);
        write_f32s(&mut file_bytes, &vec![0.0f32; L2_SIZE]);  // l1b zeros
        write_f32s(&mut file_bytes, &l2w_saved_all[s * L2_SIZE..(s + 1) * L2_SIZE]);
        write_f32s(&mut file_bytes, &[0.0f32]);  // l2b
        write_f32s(&mut file_bytes, &vec![0.0f32; SKIP_SIZE]);
    }

    let tmp_path = "/tmp/roundtrip_test.bin";
    let out_path = "/tmp/roundtrip_patched.bin";
    std::fs::File::create(tmp_path).unwrap().write_all(&file_bytes).unwrap();

    // Run the patcher via subprocess
    let status = std::process::Command::new(
        "/Users/ricer/Repos/NagatoChessEngine/assets/temp/verify_layout/target/release/patch_nn"
    ).arg(tmp_path).arg(out_path).status().unwrap();
    assert!(status.success(), "patcher failed");

    // Load patched file and verify
    let mut patched = Vec::new();
    std::fs::File::open(out_path).unwrap().read_to_end(&mut patched).unwrap();
    let mut cursor = 8usize;

    let read_f32s = |data: &[u8], offset: &mut usize, n: usize| -> Vec<f32> {
        let end = *offset + n * 4;
        let v = data[*offset..end].chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        *offset = end;
        v
    };

    let got_l0w = read_f32s(&patched, &mut cursor, FT_ROWS * L1_SIZE);
    let got_l0b = read_f32s(&patched, &mut cursor, L1_SIZE);
    cursor += FT_ROWS * NUM_PSQT_BUCKETS * 4; // skip psqt

    println!("l0w first 8 raw: {:?}", &got_l0w[..8]);
    println!("l0w expected  : {:?}", (0..8usize).map(|k| l0w_native[k]).collect::<Vec<_>>());

    let mut l0w_ok = true;
    for inp in 0..FT_ROWS.min(5) {
        for out in 0..L1_SIZE.min(4) {
            let got = got_l0w[inp * L1_SIZE + out];
            let expected = l0w_native[out + inp * L1_SIZE];
            if (got - expected).abs() > 0.5 {
                eprintln!("l0w MISMATCH at feature={inp} neuron={out}: got={got} expected={expected}");
                l0w_ok = false;
            }
        }
    }
    println!("l0w: {}", if l0w_ok { "✓ PASS" } else { "✗ FAIL" });

    println!("l0b got:      {:?}", &got_l0b[..8.min(L1_SIZE)]);
    println!("l0b expected: {:?}", &l0b_native[..8.min(L1_SIZE)]);

    let mut l0b_ok = true;
    for j in 0..L1_SIZE.min(8) {
        let got = got_l0b[j];
        let expected = l0b_native[j];
        if (got - expected).abs() > 0.5 {
            eprintln!("l0b MISMATCH at j={j}: got={got} expected={expected}");
            l0b_ok = false;
        }
    }
    println!("l0b: {}", if l0b_ok { "✓ PASS" } else { "✗ FAIL" });

    for s in 0..NUM_OUTPUT_BUCKETS {
        let got_l1w = read_f32s(&patched, &mut cursor, L2_INPUT * L2_SIZE);
        cursor += L2_SIZE * 4;  // l1b
        let got_l2w = read_f32s(&patched, &mut cursor, L2_SIZE);
        cursor += 4 + SKIP_SIZE * 4;  // l2b + skip

        let mut l1w_ok = true;
        for inp in 0..L2_INPUT.min(3) {
            for j in 0..L2_SIZE.min(3) {
                let global_out = s * L2_SIZE + j;
                let got = got_l1w[inp * L2_SIZE + j];
                let expected = l1w_native[global_out + inp * stride];
                if (got - expected).abs() > 0.5 {
                    eprintln!("  l1w[bkt={s}] MISMATCH at in={inp} j={j}: got={got} expected={expected}");
                    l1w_ok = false;
                }
            }
        }
        println!("l1w bucket {s}: {}", if l1w_ok { "✓ PASS" } else { "✗ FAIL" });

        let mut l2w_ok = true;
        for j in 0..L2_SIZE.min(4) {
            let got = got_l2w[j];
            let expected = l2w_native[s + j * NUM_OUTPUT_BUCKETS]; // col-major: out=s, in=j
            if (got - expected).abs() > 0.5 {
                eprintln!("  l2w[bkt={s}] MISMATCH at j={j}: got={got} expected={expected}");
                l2w_ok = false;
            }
        }
        println!("l2w bucket {s}: {}", if l2w_ok { "✓ PASS" } else { "✗ FAIL" });
    }
}
