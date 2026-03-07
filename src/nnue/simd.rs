use super::{L1_SIZE, L1_PAIR, L2_INPUT, L2_SIZE, QA, QB};

const CHUNK_NEON: usize = 8;
const ITERS_NEON: usize = L1_SIZE / CHUNK_NEON;
const CONCAT: usize = L2_INPUT;
const ITERS_PAIR_NEON: usize = L1_PAIR / CHUNK_NEON;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_add_i16(dst: &mut [i16; L1_SIZE], src: &[i16; L1_SIZE]) {
    unsafe {
        for k in 0..ITERS_NEON {
            let off = k * CHUNK_NEON;
            let a = vld1q_s16(dst.as_ptr().add(off));
            let b = vld1q_s16(src.as_ptr().add(off));
            vst1q_s16(dst.as_mut_ptr().add(off), vaddq_s16(a, b));
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_sub_i16(dst: &mut [i16; L1_SIZE], src: &[i16; L1_SIZE]) {
    unsafe {
        for k in 0..ITERS_NEON {
            let off = k * CHUNK_NEON;
            let a = vld1q_s16(dst.as_ptr().add(off));
            let b = vld1q_s16(src.as_ptr().add(off));
            vst1q_s16(dst.as_mut_ptr().add(off), vsubq_s16(a, b));
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_add_sub_i16(
    dst: &mut [i16; L1_SIZE],
    add_src: &[i16; L1_SIZE],
    sub_src: &[i16; L1_SIZE],
) {
    unsafe {
        for k in 0..ITERS_NEON {
            let off = k * CHUNK_NEON;
            let d = vld1q_s16(dst.as_ptr().add(off));
            let a = vld1q_s16(add_src.as_ptr().add(off));
            let s = vld1q_s16(sub_src.as_ptr().add(off));
            let r = vsubq_s16(vaddq_s16(d, a), s);
            vst1q_s16(dst.as_mut_ptr().add(off), r);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn pairwise_crelu(
    stm: &[i16; L1_SIZE],
    opp: &[i16; L1_SIZE],
    out: &mut [u8; CONCAT],
) {
    unsafe {
        let zero = vdupq_n_s16(0);
        let max = vdupq_n_s16(QA as i16);
        for k in 0..ITERS_PAIR_NEON {
            let off = k * CHUNK_NEON;
            let lo = vminq_s16(vmaxq_s16(vld1q_s16(stm.as_ptr().add(off)), zero), max);
            let hi = vminq_s16(vmaxq_s16(vld1q_s16(stm.as_ptr().add(L1_PAIR + off)), zero), max);
            let lo_u8 = vmovn_u16(vreinterpretq_u16_s16(lo));
            let hi_u8 = vmovn_u16(vreinterpretq_u16_s16(hi));
            let prod = vmull_u8(lo_u8, hi_u8);
            let scaled = vmovn_u16(vshrq_n_u16::<8>(prod));
            vst1_u8(out.as_mut_ptr().add(off), scaled);
        }
        for k in 0..ITERS_PAIR_NEON {
            let off = k * CHUNK_NEON;
            let lo = vminq_s16(vmaxq_s16(vld1q_s16(opp.as_ptr().add(off)), zero), max);
            let hi = vminq_s16(vmaxq_s16(vld1q_s16(opp.as_ptr().add(L1_PAIR + off)), zero), max);
            let lo_u8 = vmovn_u16(vreinterpretq_u16_s16(lo));
            let hi_u8 = vmovn_u16(vreinterpretq_u16_s16(hi));
            let prod = vmull_u8(lo_u8, hi_u8);
            let scaled = vmovn_u16(vshrq_n_u16::<8>(prod));
            vst1_u8(out.as_mut_ptr().add(L1_PAIR + off), scaled);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn dot_u8_i8(acts: &[u8; CONCAT], weights_t: &[i8], bias: i32) -> i32 {
    unsafe {
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        let mut k = 0usize;
        while k < CONCAT {
            let a = vld1_u8(acts.as_ptr().add(k));
            let w = vld1_s8(weights_t.as_ptr().add(k));
            let a_lo = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(a)));
            let a_hi = vget_high_s16(vreinterpretq_s16_u16(vmovl_u8(a)));
            let w_lo = vget_low_s16(vmovl_s8(w));
            let w_hi = vget_high_s16(vmovl_s8(w));
            acc0 = vmlal_s16(acc0, a_lo, w_lo);
            acc1 = vmlal_s16(acc1, a_hi, w_hi);
            k += 8;
        }
        let sum = vaddq_s32(acc0, acc1);
        bias + vaddvq_s32(sum)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn affine_l2(
    stm: &[i16; L1_SIZE],
    opp: &[i16; L1_SIZE],
    weights_t: &[[i8; CONCAT]; L2_SIZE],
    biases: &[i32; L2_SIZE],
    out: &mut [i32; L2_SIZE],
) {
    let mut acts = [0u8; CONCAT];
    pairwise_crelu(stm, opp, &mut acts);
    for j in 0..L2_SIZE {
        out[j] = dot_u8_i8(&acts, &weights_t[j], biases[j]);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn output_layer(
    l2_out: &[i32; L2_SIZE],
    out_weights: &[i16; L2_SIZE],
    out_bias: i32,
) -> i64 {
    let qa = QA;
    let qa_qb = QA * QB;
    let mut output = out_bias as i64;
    for j in 0..L2_SIZE {
        let activated = l2_out[j].max(0).min(qa_qb) / qa;
        output += activated as i64 * out_weights[j] as i64;
    }
    output
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const CHUNK_AVX2: usize = 16;
const ITERS_AVX2: usize = L1_SIZE / CHUNK_AVX2;
const ITERS_PAIR_AVX2: usize = L1_PAIR / CHUNK_AVX2;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn vec_add_i16_avx2(dst: &mut [i16; L1_SIZE], src: &[i16; L1_SIZE]) {
    for k in 0..ITERS_AVX2 {
        let off = k * CHUNK_AVX2;
        let a = _mm256_loadu_si256(dst.as_ptr().add(off) as *const __m256i);
        let b = _mm256_loadu_si256(src.as_ptr().add(off) as *const __m256i);
        _mm256_storeu_si256(dst.as_mut_ptr().add(off) as *mut __m256i, _mm256_add_epi16(a, b));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn vec_sub_i16_avx2(dst: &mut [i16; L1_SIZE], src: &[i16; L1_SIZE]) {
    for k in 0..ITERS_AVX2 {
        let off = k * CHUNK_AVX2;
        let a = _mm256_loadu_si256(dst.as_ptr().add(off) as *const __m256i);
        let b = _mm256_loadu_si256(src.as_ptr().add(off) as *const __m256i);
        _mm256_storeu_si256(dst.as_mut_ptr().add(off) as *mut __m256i, _mm256_sub_epi16(a, b));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn vec_add_sub_i16_avx2(
    dst: &mut [i16; L1_SIZE],
    add_src: &[i16; L1_SIZE],
    sub_src: &[i16; L1_SIZE],
) {
    for k in 0..ITERS_AVX2 {
        let off = k * CHUNK_AVX2;
        let d = _mm256_loadu_si256(dst.as_ptr().add(off) as *const __m256i);
        let a = _mm256_loadu_si256(add_src.as_ptr().add(off) as *const __m256i);
        let s = _mm256_loadu_si256(sub_src.as_ptr().add(off) as *const __m256i);
        let r = _mm256_sub_epi16(_mm256_add_epi16(d, a), s);
        _mm256_storeu_si256(dst.as_mut_ptr().add(off) as *mut __m256i, r);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pairwise_crelu_avx2(
    stm: &[i16; L1_SIZE],
    opp: &[i16; L1_SIZE],
    out: &mut [u8; CONCAT],
) {
    let zero = _mm256_setzero_si256();
    let max = _mm256_set1_epi16(QA as i16);
    for k in 0..ITERS_PAIR_AVX2 {
        let off = k * CHUNK_AVX2;
        let lo = _mm256_min_epi16(_mm256_max_epi16(
            _mm256_loadu_si256(stm.as_ptr().add(off) as *const __m256i), zero), max);
        let hi = _mm256_min_epi16(_mm256_max_epi16(
            _mm256_loadu_si256(stm.as_ptr().add(L1_PAIR + off) as *const __m256i), zero), max);
        let prod = _mm256_mullo_epi16(lo, hi);
        let shifted = _mm256_srli_epi16(prod, 8);
        let packed = _mm256_packus_epi16(shifted, _mm256_setzero_si256());
        let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);
        let lo128 = _mm256_castsi256_si128(packed);
        _mm_storeu_si128(out.as_mut_ptr().add(off) as *mut __m128i, lo128);
    }
    for k in 0..ITERS_PAIR_AVX2 {
        let off = k * CHUNK_AVX2;
        let lo = _mm256_min_epi16(_mm256_max_epi16(
            _mm256_loadu_si256(opp.as_ptr().add(off) as *const __m256i), zero), max);
        let hi = _mm256_min_epi16(_mm256_max_epi16(
            _mm256_loadu_si256(opp.as_ptr().add(L1_PAIR + off) as *const __m256i), zero), max);
        let prod = _mm256_mullo_epi16(lo, hi);
        let shifted = _mm256_srli_epi16(prod, 8);
        let packed = _mm256_packus_epi16(shifted, _mm256_setzero_si256());
        let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);
        let lo128 = _mm256_castsi256_si128(packed);
        _mm_storeu_si128(out.as_mut_ptr().add(L1_PAIR + off) as *mut __m128i, lo128);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dot_u8_i8_avx2(acts: &[u8; CONCAT], weights_t: &[i8], bias: i32) -> i32 {
    let mut acc = _mm256_setzero_si256();
    let ones = _mm256_set1_epi16(1);
    let mut k = 0usize;
    while k < CONCAT {
        let a = _mm256_loadu_si256(acts.as_ptr().add(k) as *const __m256i);
        let w = _mm256_loadu_si256(weights_t.as_ptr().add(k) as *const __m256i);
        let prod = _mm256_maddubs_epi16(a, w);
        let widened = _mm256_madd_epi16(prod, ones);
        acc = _mm256_add_epi32(acc, widened);
        k += 32;
    }
    let hi = _mm256_extracti128_si256(acc, 1);
    let lo = _mm256_castsi256_si128(acc);
    let sum128 = _mm_add_epi32(lo, hi);
    let hi64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi32(sum128, hi64);
    let hi32 = _mm_shuffle_epi32(sum64, 0b00_00_00_01);
    let sum32 = _mm_add_epi32(sum64, hi32);
    bias + _mm_cvtsi128_si32(sum32)
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn vec_add_i16(dst: &mut [i16; L1_SIZE], src: &[i16; L1_SIZE]) {
    if is_x86_feature_detected!("avx2") {
        unsafe { vec_add_i16_avx2(dst, src) }
    } else {
        for j in 0..L1_SIZE { dst[j] += src[j]; }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn vec_sub_i16(dst: &mut [i16; L1_SIZE], src: &[i16; L1_SIZE]) {
    if is_x86_feature_detected!("avx2") {
        unsafe { vec_sub_i16_avx2(dst, src) }
    } else {
        for j in 0..L1_SIZE { dst[j] -= src[j]; }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn vec_add_sub_i16(
    dst: &mut [i16; L1_SIZE],
    add_src: &[i16; L1_SIZE],
    sub_src: &[i16; L1_SIZE],
) {
    if is_x86_feature_detected!("avx2") {
        unsafe { vec_add_sub_i16_avx2(dst, add_src, sub_src) }
    } else {
        for j in 0..L1_SIZE {
            dst[j] = dst[j].wrapping_add(add_src[j]).wrapping_sub(sub_src[j]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn pairwise_crelu(
    stm: &[i16; L1_SIZE],
    opp: &[i16; L1_SIZE],
    out: &mut [u8; CONCAT],
) {
    if is_x86_feature_detected!("avx2") {
        unsafe { pairwise_crelu_avx2(stm, opp, out) }
    } else {
        for i in 0..L1_PAIR {
            let lo = stm[i].max(0).min(QA as i16) as u16;
            let hi = stm[L1_PAIR + i].max(0).min(QA as i16) as u16;
            out[i] = ((lo * hi) >> 8) as u8;
        }
        for i in 0..L1_PAIR {
            let lo = opp[i].max(0).min(QA as i16) as u16;
            let hi = opp[L1_PAIR + i].max(0).min(QA as i16) as u16;
            out[L1_PAIR + i] = ((lo * hi) >> 8) as u8;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dot_u8_i8(acts: &[u8; CONCAT], weights_t: &[i8], bias: i32) -> i32 {
    if is_x86_feature_detected!("avx2") {
        unsafe { dot_u8_i8_avx2(acts, weights_t, bias) }
    } else {
        let mut sum = bias;
        for i in 0..CONCAT { sum += acts[i] as i32 * weights_t[i] as i32; }
        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn affine_l2(
    stm: &[i16; L1_SIZE],
    opp: &[i16; L1_SIZE],
    weights_t: &[[i8; CONCAT]; L2_SIZE],
    biases: &[i32; L2_SIZE],
    out: &mut [i32; L2_SIZE],
) {
    let mut acts = [0u8; CONCAT];
    pairwise_crelu(stm, opp, &mut acts);
    for j in 0..L2_SIZE {
        out[j] = dot_u8_i8(&acts, &weights_t[j], biases[j]);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn output_layer(
    l2_out: &[i32; L2_SIZE],
    out_weights: &[i16; L2_SIZE],
    out_bias: i32,
) -> i64 {
    let qa = QA;
    let qa_qb = QA * QB;
    let mut output = out_bias as i64;
    for j in 0..L2_SIZE {
        let activated = l2_out[j].max(0).min(qa_qb) / qa;
        output += activated as i64 * out_weights[j] as i64;
    }
    output
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline]
pub fn vec_add_i16(dst: &mut [i16; L1_SIZE], src: &[i16; L1_SIZE]) {
    for j in 0..L1_SIZE {
        dst[j] += src[j];
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline]
pub fn vec_sub_i16(dst: &mut [i16; L1_SIZE], src: &[i16; L1_SIZE]) {
    for j in 0..L1_SIZE {
        dst[j] -= src[j];
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline]
pub fn vec_add_sub_i16(
    dst: &mut [i16; L1_SIZE],
    add_src: &[i16; L1_SIZE],
    sub_src: &[i16; L1_SIZE],
) {
    for j in 0..L1_SIZE {
        dst[j] = dst[j].wrapping_add(add_src[j]).wrapping_sub(sub_src[j]);
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline]
pub fn pairwise_crelu(
    stm: &[i16; L1_SIZE],
    opp: &[i16; L1_SIZE],
    out: &mut [u8; CONCAT],
) {
    for i in 0..L1_PAIR {
        let lo = stm[i].max(0).min(QA as i16) as u16;
        let hi = stm[L1_PAIR + i].max(0).min(QA as i16) as u16;
        out[i] = ((lo * hi) >> 8) as u8;
    }
    for i in 0..L1_PAIR {
        let lo = opp[i].max(0).min(QA as i16) as u16;
        let hi = opp[L1_PAIR + i].max(0).min(QA as i16) as u16;
        out[L1_PAIR + i] = ((lo * hi) >> 8) as u8;
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline]
pub fn dot_u8_i8(acts: &[u8; CONCAT], weights_t: &[i8], bias: i32) -> i32 {
    let mut sum = bias;
    for i in 0..CONCAT {
        sum += acts[i] as i32 * weights_t[i] as i32;
    }
    sum
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline]
pub fn affine_l2(
    stm: &[i16; L1_SIZE],
    opp: &[i16; L1_SIZE],
    weights_t: &[[i8; CONCAT]; L2_SIZE],
    biases: &[i32; L2_SIZE],
    out: &mut [i32; L2_SIZE],
) {
    let mut acts = [0u8; CONCAT];
    pairwise_crelu(stm, opp, &mut acts);
    for j in 0..L2_SIZE {
        out[j] = dot_u8_i8(&acts, &weights_t[j], biases[j]);
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline]
pub fn output_layer(
    l2_out: &[i32; L2_SIZE],
    out_weights: &[i16; L2_SIZE],
    out_bias: i32,
) -> i64 {
    let qa = QA;
    let qa_qb = QA * QB;
    let mut output = out_bias as i64;
    for j in 0..L2_SIZE {
        let activated = l2_out[j].max(0).min(qa_qb) / qa;
        output += activated as i64 * out_weights[j] as i64;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_add_i16() {
        let mut dst = [0i16; L1_SIZE];
        let src = [1i16; L1_SIZE];
        vec_add_i16(&mut dst, &src);
        assert!(dst.iter().all(|&v| v == 1));
        vec_add_i16(&mut dst, &src);
        assert!(dst.iter().all(|&v| v == 2));
    }

    #[test]
    fn test_vec_sub_i16() {
        let mut dst = [10i16; L1_SIZE];
        let src = [3i16; L1_SIZE];
        vec_sub_i16(&mut dst, &src);
        assert!(dst.iter().all(|&v| v == 7));
    }

    #[test]
    fn test_vec_add_sub_i16() {
        let mut dst = [10i16; L1_SIZE];
        let add = [5i16; L1_SIZE];
        let sub = [3i16; L1_SIZE];
        vec_add_sub_i16(&mut dst, &add, &sub);
        assert!(dst.iter().all(|&v| v == 12));
    }

    #[test]
    fn test_vec_add_varied() {
        let mut dst = [0i16; L1_SIZE];
        let mut src = [0i16; L1_SIZE];
        for i in 0..L1_SIZE {
            src[i] = i as i16;
        }
        vec_add_i16(&mut dst, &src);
        for i in 0..L1_SIZE {
            assert_eq!(dst[i], i as i16);
        }
        vec_add_i16(&mut dst, &src);
        for i in 0..L1_SIZE {
            assert_eq!(dst[i], 2 * i as i16);
        }
    }

    #[test]
    fn test_vec_negative_values() {
        let mut dst = [0i16; L1_SIZE];
        let mut src = [0i16; L1_SIZE];
        for i in 0..L1_SIZE {
            src[i] = -(i as i16);
        }
        vec_add_i16(&mut dst, &src);
        for i in 0..L1_SIZE {
            assert_eq!(dst[i], -(i as i16));
        }
    }

    #[test]
    fn test_pairwise_crelu_basic() {
        let mut stm = [0i16; L1_SIZE];
        let mut opp = [0i16; L1_SIZE];
        for i in 0..L1_PAIR {
            stm[i] = 100;
            stm[L1_PAIR + i] = 200;
            opp[i] = 50;
            opp[L1_PAIR + i] = 50;
        }
        let mut out = [0u8; CONCAT];
        pairwise_crelu(&stm, &opp, &mut out);
        let expected_stm = ((100u16 * 200) >> 8) as u8;
        let expected_opp = ((50u16 * 50) >> 8) as u8;
        for i in 0..L1_PAIR {
            assert_eq!(out[i], expected_stm, "stm mismatch at {}", i);
        }
        for i in 0..L1_PAIR {
            assert_eq!(out[L1_PAIR + i], expected_opp, "opp mismatch at {}", i);
        }
    }

    #[test]
    fn test_pairwise_crelu_saturation() {
        let stm = [255i16; L1_SIZE];
        let opp = [0i16; L1_SIZE];
        let mut out = [0u8; CONCAT];
        pairwise_crelu(&stm, &opp, &mut out);
        let max_val = ((255u16 * 255) >> 8) as u8;
        assert!(out[..L1_PAIR].iter().all(|&v| v == max_val));
        assert!(out[L1_PAIR..].iter().all(|&v| v == 0));
    }

    #[test]
    fn test_dot_u8_i8_basic() {
        let mut acts = [0u8; CONCAT];
        let mut wts = [0i8; CONCAT];
        for i in 0..CONCAT {
            acts[i] = 1;
            wts[i] = 2;
        }
        let result = dot_u8_i8(&acts, &wts, 10);
        assert_eq!(result, 10 + (CONCAT as i32) * 2);
    }

    #[test]
    fn test_dot_u8_i8_mixed_signs() {
        let mut acts = [0u8; CONCAT];
        let mut wts = [0i8; CONCAT];
        for i in 0..CONCAT {
            acts[i] = 10;
            wts[i] = if i % 2 == 0 { 1 } else { -1 };
        }
        let result = dot_u8_i8(&acts, &wts, 0);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_affine_l2_zero() {
        let stm = [0i16; L1_SIZE];
        let opp = [0i16; L1_SIZE];
        let weights_t = [[0i8; CONCAT]; L2_SIZE];
        let biases = [42i32; L2_SIZE];
        let mut out = [0i32; L2_SIZE];
        affine_l2(&stm, &opp, &weights_t, &biases, &mut out);
        assert!(out.iter().all(|&v| v == 42));
    }
}
