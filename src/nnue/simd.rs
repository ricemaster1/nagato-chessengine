use super::L1_SIZE;

const CHUNK_NEON: usize = 8;
const ITERS_NEON: usize = L1_SIZE / CHUNK_NEON;

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
}
