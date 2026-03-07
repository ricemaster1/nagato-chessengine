use super::{L1_SIZE, L2_SIZE, QA, QB};

const CHUNK_NEON: usize = 8;
const ITERS_NEON: usize = L1_SIZE / CHUNK_NEON;
const CONCAT: usize = 2 * L1_SIZE;

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
pub fn crelu_pack(
    stm: &[i16; L1_SIZE],
    opp: &[i16; L1_SIZE],
    out: &mut [u8; CONCAT],
    qa: i16,
) {
    unsafe {
        let zero = vdupq_n_s16(0);
        let max = vdupq_n_s16(qa);
        for k in 0..ITERS_NEON {
            let off = k * CHUNK_NEON;
            let v = vld1q_s16(stm.as_ptr().add(off));
            let clamped = vminq_s16(vmaxq_s16(v, zero), max);
            let narrow = vmovn_u16(vreinterpretq_u16_s16(clamped));
            vst1_u8(out.as_mut_ptr().add(off), narrow);
        }
        for k in 0..ITERS_NEON {
            let off = k * CHUNK_NEON;
            let v = vld1q_s16(opp.as_ptr().add(off));
            let clamped = vminq_s16(vmaxq_s16(v, zero), max);
            let narrow = vmovn_u16(vreinterpretq_u16_s16(clamped));
            vst1_u8(out.as_mut_ptr().add(L1_SIZE + off), narrow);
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
    crelu_pack(stm, opp, &mut acts, QA as i16);
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
pub fn crelu_pack(
    stm: &[i16; L1_SIZE],
    opp: &[i16; L1_SIZE],
    out: &mut [u8; CONCAT],
    qa: i16,
) {
    for i in 0..L1_SIZE {
        out[i] = stm[i].max(0).min(qa) as u8;
    }
    for i in 0..L1_SIZE {
        out[L1_SIZE + i] = opp[i].max(0).min(qa) as u8;
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
    crelu_pack(stm, opp, &mut acts, QA as i16);
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
    fn test_crelu_pack_basic() {
        let mut stm = [0i16; L1_SIZE];
        let mut opp = [0i16; L1_SIZE];
        for i in 0..L1_SIZE {
            stm[i] = i as i16 * 3;
            opp[i] = -(i as i16);
        }
        let mut out = [0u8; CONCAT];
        crelu_pack(&stm, &opp, &mut out, QA as i16);
        for i in 0..L1_SIZE {
            let expected = (stm[i].max(0).min(QA as i16)) as u8;
            assert_eq!(out[i], expected, "stm mismatch at {}", i);
        }
        for i in 0..L1_SIZE {
            assert_eq!(out[L1_SIZE + i], 0, "negative should clamp to 0");
        }
    }

    #[test]
    fn test_crelu_pack_saturation() {
        let stm = [500i16; L1_SIZE];
        let opp = [200i16; L1_SIZE];
        let mut out = [0u8; CONCAT];
        crelu_pack(&stm, &opp, &mut out, QA as i16);
        assert!(out[..L1_SIZE].iter().all(|&v| v == QA as u8));
        assert!(out[L1_SIZE..].iter().all(|&v| v == 200));
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
