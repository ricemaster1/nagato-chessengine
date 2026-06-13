// Layout verification for HalfKA NNUE trainer save format.
//
// Replicates three functions verbatim from Bullet and nagato-trainer, applies
// them to tiny matrices with known values, and checks whether the byte order
// in the resulting file matches what Nagato's loader expects.
//
// Bullet docs (assets/bullet/docs/4-saved-networks.md):
//   "Every weight has an associated shape MxN and is written in column-major format."
//   ".transpose() converts to row-major."
//   "An affine layer's weights have shape (output_size, input_size)."
//
// Nagato reads l0w (feature weights) as:
//   for i in 0..FT_SIZE  { for j in 0..L1_SIZE { l1_weights[i][j] = read_f32() } }
// Expected byte order: (i=feature slow, j=neuron fast), stride L1_SIZE per feature.

// ── Bullet functions (verbatim copies) ───────────────────────────────────────

fn transpose_impl(weights: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(weights.len(), rows * cols);
    let mut new_buf = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            new_buf[cols * i + j] = weights[rows * j + i];
        }
    }
    new_buf
}

// Trainer function (verbatim from tools/nagato-trainer/src/main.rs)
fn halves_perm(out_idx: usize, pair: usize) -> usize {
    if out_idx < pair { 2 * out_idx } else { 2 * (out_idx - pair) + 1 }
}

fn permute_halves_rows(src: &[f32], row_size: usize) -> Vec<f32> {
    let pair = row_size / 2;
    let rows = src.len() / row_size;
    let mut out = vec![0.0f32; src.len()];
    for r in 0..rows {
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

// Trainer l1w per-bucket extraction (verbatim)
fn extract_l1w_bucket(w: &[f32], s: usize, num_out_buckets: usize, l2: usize, l2_input: usize) -> Vec<f32> {
    let stride = num_out_buckets * l2;
    (0..l2_input).flat_map(|in_i| (0..l2).map(move |j| w[in_i * stride + s * l2 + j])).collect()
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn col_major_value(out: usize, inp: usize, out_size: usize) -> f32 {
    (out * 1000 + inp) as f32
}

fn make_col_major(out_size: usize, in_size: usize) -> Vec<f32> {
    // element (out, in) at index = out + in*out_size
    let mut buf = vec![0.0f32; out_size * in_size];
    for inp in 0..in_size {
        for out in 0..out_size {
            buf[out + inp * out_size] = col_major_value(out, inp, out_size);
        }
    }
    buf
}

fn print_matrix(label: &str, buf: &[f32], row_size: usize) {
    println!("{label} (rows of {row_size}):");
    for (i, chunk) in buf.chunks(row_size).enumerate() {
        let vals: Vec<String> = chunk.iter().map(|v| format!("{:7.0}", v)).collect();
        println!("  row {:3}: [{}]", i, vals.join(", "));
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

fn test_l0w() {
    println!("\n=== L0W layout test ===");
    println!("Architecture: output=4, input=6  (col-major: element(out,in) = out*1000+in)");

    let out_size = 4;
    let in_size = 6;

    let native = make_col_major(out_size, in_size);
    print_matrix("Native col-major", &native, out_size);

    // ── CURRENT TRAINER: .transpose() then permute_halves_rows ──────────────
    let after_transpose = transpose_impl(&native, out_size, in_size);
    let current_file = permute_halves_rows(&after_transpose, out_size);

    print_matrix("Current trainer output (transpose+permute)", &current_file, out_size);

    println!("What Nagato reads as l1_weights[feature][neuron]:");
    for i in 0..in_size {
        let row: Vec<String> = (0..out_size)
            .map(|j| format!("{:7.0}", current_file[i * out_size + j]))
            .collect();
        println!("  l1_weights[feature={i}] = [{}]  (expected: {:?})",
            row.join(", "),
            (0..out_size).map(|out| col_major_value(out, i, out_size)).collect::<Vec<_>>());
    }

    // ── PROPOSED FIX: no transpose, permute_halves_rows only ────────────────
    let fixed_file = permute_halves_rows(&native, out_size);
    print_matrix("\nFixed trainer output (permute only, no transpose)", &fixed_file, out_size);

    println!("What Nagato reads as l1_weights[feature][neuron]:");
    for i in 0..in_size {
        let row: Vec<String> = (0..out_size)
            .map(|j| format!("{:7.0}", fixed_file[i * out_size + j]))
            .collect();
        let expected: Vec<f32> = (0..out_size).map(|out| col_major_value(halves_perm(out, out_size/2), i, out_size)).collect();
        println!("  l1_weights[feature={i}] = [{}]  (permuted-expected: {:?})",
            row.join(", "), expected);
    }

    // ── NO transform at all ─────────────────────────────────────────────────
    print_matrix("\nNo-transform output (pure col-major)", &native, out_size);
    println!("What Nagato reads:");
    for i in 0..in_size {
        let row: Vec<String> = (0..out_size)
            .map(|j| format!("{:7.0}", native[i * out_size + j]))
            .collect();
        let expected: Vec<String> = (0..out_size)
            .map(|out| format!("{:7.0}", col_major_value(out, i, out_size)))
            .collect();
        let matches = (0..out_size).all(|j| native[i * out_size + j] == col_major_value(j, i, out_size));
        println!("  l1_weights[feature={i}] = [{}]  expected=[{}]  {}",
            row.join(", "), expected.join(", "),
            if matches { "✓ CORRECT" } else { "✗ WRONG" });
    }
}

fn test_l0b() {
    println!("\n=== L0B layout test ===");
    println!("Architecture: output=8 (L1_SIZE for bias test)");

    let l1 = 8;
    let bias: Vec<f32> = (0..l1).map(|j| j as f32).collect();
    println!("Native bias (expected order, neuron j = j): {:?}", bias);

    let current = permute_halves_1d(&bias);
    println!("Current trainer (permute_halves_1d): {:?}", current);
    println!("Nagato reads: l1_biases[j] = current[j]");
    println!("  current[0] = {:.0} (expected neuron 0)", current[0]);
    println!("  current[1] = {:.0} (expected neuron 1 but is {})", current[1],
        if current[1] == 1.0 { "✓" } else { "✗ wrong" });

    let no_perm = bias.clone();
    println!("Fixed trainer (no permute): {:?}", no_perm);
    println!("  no_perm[0] = {:.0} (expected neuron 0) {}",
        no_perm[0], if no_perm[0] == 0.0 { "✓" } else { "✗" });
}

fn test_l1w() {
    println!("\n=== L1W layout test ===");
    println!("Architecture: output=8 (=2 buckets × L2=4), input=8 (L2_INPUT), buckets=2");

    let out_total = 8;   // num_out_buckets * l2 = 2*4
    let in_total = 8;    // l2_input
    let num_buckets = 2;
    let l2 = 4;

    let native = make_col_major(out_total, in_total);
    print_matrix("Native l1w col-major (out_total=8, in_total=8)", &native, out_total);
    println!("  Element(out,in) = out*1000+in");

    // Current: .transpose() then extract
    let after_t = transpose_impl(&native, out_total, in_total);
    print_matrix("After .transpose()", &after_t, in_total);

    println!("\nCurrent trainer output per bucket (formula: w[in_i*8 + s*4 + j]):");
    for s in 0..num_buckets {
        let extracted = extract_l1w_bucket(&after_t, s, num_buckets, l2, in_total);
        println!("  Bucket {s}: {:?}", extracted);
        println!("    Expected (bucket {s} = out {}..{}): {:?}", s*l2, (s+1)*l2,
            (0..in_total).flat_map(|inp| (s*l2..(s+1)*l2).map(move |out| col_major_value(out, inp, out_total))).collect::<Vec<_>>());
    }

    // Fixed: NO .transpose(), then extract
    println!("\nFixed trainer output per bucket (no transpose, same formula):");
    for s in 0..num_buckets {
        let extracted = extract_l1w_bucket(&native, s, num_buckets, l2, in_total);
        let expected: Vec<f32> = (0..in_total).flat_map(|inp| {
            (s*l2..(s+1)*l2).map(move |out| col_major_value(out, inp, out_total))
        }).collect();
        let matches = extracted == expected;
        println!("  Bucket {s}: {:?}", extracted);
        println!("    Expected:  {:?}  {}", expected, if matches { "✓ CORRECT" } else { "✗ WRONG" });
    }
}

fn test_l2w() {
    println!("\n=== L2W layout test ===");
    println!("Architecture: output=2 (buckets), input=4 (l2)");

    let out_size = 2;  // num_out_buckets
    let in_size = 4;   // l2

    let native = make_col_major(out_size, in_size);
    print_matrix("Native l2w col-major (out=buckets=2, in=l2=4)", &native, out_size);

    let after_t = transpose_impl(&native, out_size, in_size);
    print_matrix("After .transpose()", &after_t, in_size);

    println!("\nNagato expects output_weights[s][j] = weight for output-bucket s, L2-neuron j:");
    for s in 0..out_size {
        let expected: Vec<f32> = (0..in_size).map(|j| col_major_value(s, j, out_size)).collect();
        println!("  Bucket {s} expected: {:?}", expected);
    }

    println!("\nCurrent trainer (NO transpose, slice w[s*4..(s+1)*4]):");
    for s in 0..out_size {
        let extracted: Vec<f32> = native[s*in_size..(s+1)*in_size].to_vec();
        let expected: Vec<f32> = (0..in_size).map(|j| col_major_value(s, j, out_size)).collect();
        let matches = extracted == expected;
        println!("  Bucket {s}: {:?}  {}", extracted, if matches { "✓" } else { "✗ WRONG" });
    }

    println!("\nFixed trainer (WITH transpose, slice w[s*4..(s+1)*4]):");
    for s in 0..out_size {
        let extracted: Vec<f32> = after_t[s*in_size..(s+1)*in_size].to_vec();
        let expected: Vec<f32> = (0..in_size).map(|j| col_major_value(s, j, out_size)).collect();
        let matches = extracted == expected;
        println!("  Bucket {s}: {:?}  {}", extracted, if matches { "✓ CORRECT" } else { "✗" });
    }
}

fn main() {
    println!("Bullet layout: column-major, shape=(output, input), element(out,in) at idx = out + in*output_size");
    println!("Nagato reads:  l1_weights[feature=in][neuron=out] sequentially (in slow, out fast)");
    println!("               → same byte order as Bullet col-major iff no .transpose() applied");

    test_l0w();
    test_l0b();
    test_l1w();
    test_l2w();

    println!("\n=== Summary ===");
    println!("l0w: .transpose() scrambles layout. Fix: remove .transpose().");
    println!("l0b: permute_halves_1d scrambles bias. Fix: remove it.");
    println!("l1w: .transpose() makes col-major formula read wrong bytes. Fix: remove .transpose().");
    println!("l2w: missing .transpose() means per-bucket slice reads mixed (out,in) pairs. Fix: add .transpose().");
}
