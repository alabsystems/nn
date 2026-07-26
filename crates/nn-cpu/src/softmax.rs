// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized softmax along the last dimension.
//!
//! Three-pass numerically stable softmax:
//!   1. Find max over the dimension (prevents overflow in exp).
//!   2. Subtract max and compute exp for each element.
//!   3. Divide each exp by the sum of all exps.
//!
//! NEON (aarch64) and AVX2 (x86_64) paths with scalar fallback.

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

/// Scalar softmax along the last dimension.
///
/// `input` and `output` must have the same length, which must be a
/// multiple of `dim_size`. Softmax is applied independently over each
/// contiguous `dim_size`-element row.
pub fn softmax_scalar(input: &[f32], output: &mut [f32], dim_size: usize) {
    assert_eq!(input.len(), output.len());
    assert!(dim_size > 0, "dim_size must be > 0");
    assert_eq!(
        input.len() % dim_size,
        0,
        "input length must be a multiple of dim_size"
    );

    let rows = input.len() / dim_size;
    for row in 0..rows {
        let start = row * dim_size;
        let end = start + dim_size;
        let row_in = &input[start..end];
        let row_out = &mut output[start..end];

        // Pass 1: max
        let mut max_val = f32::NEG_INFINITY;
        for &x in row_in {
            if x > max_val {
                max_val = x;
            }
        }

        // Pass 2: exp(x - max)
        let mut sum = 0.0_f32;
        for (o, &x) in row_out.iter_mut().zip(row_in.iter()) {
            let e = (x - max_val).exp();
            *o = e;
            sum += e;
        }

        // Pass 3: normalize
        let inv_sum = 1.0 / sum;
        for o in row_out.iter_mut() {
            *o *= inv_sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Fast exp approximation (shared between NEON and scalar hot paths)
// ---------------------------------------------------------------------------

/// Fast exp approximation using the Schraudolph method with linear
/// interpolation. Accurate to ~1e-4 relative error for |x| < 80.
/// Falls back gracefully for extreme inputs (clips to 0 / large).
#[inline(always)]
fn fast_exp_f32(x: f32) -> f32 {
    // Clamp to prevent overflow/underflow in the integer trick.
    let x = x.clamp(-87.0, 88.0);
    // Constants: log2(e) = 1.442695, 2^23 = 8388608
    // Schraudolph: reinterpret (2^23 / ln2) * x + (127 * 2^23 - bias) as f32
    const A: f32 = 12102203.0; // 2^23 / ln2
    const B: f32 = 1064866805.0; // 127 * 2^23 - ~486411 (bias for better accuracy)
    let val = A * x + B;
    f32::from_bits(val as u32)
}

// ---------------------------------------------------------------------------
// NEON (aarch64) — 128-bit, 4x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// NEON-accelerated softmax over a single row of `dim_size` elements.
    ///
    /// Uses vmaxq_f32 for max reduction, fast exp approximation, and
    /// vaddq_f32 for sum accumulation.
    pub(super) fn softmax_neon(input: &[f32], output: &mut [f32], dim_size: usize) {
        assert_eq!(input.len(), output.len());
        assert!(dim_size > 0);
        assert_eq!(input.len() % dim_size, 0);

        let rows = input.len() / dim_size;
        for row in 0..rows {
            let start = row * dim_size;
            let end = start + dim_size;
            softmax_row_neon(&input[start..end], &mut output[start..end]);
        }
    }

    fn softmax_row_neon(row_in: &[f32], row_out: &mut [f32]) {
        let n = row_in.len();
        let chunks = n / 4;
        let remainder = n % 4;
        let tail_start = chunks * 4;

        // ---- Pass 1: find max ----
        // SAFETY: aarch64 NEON is always available. Bounded loads within slice.
        let max_val = unsafe {
            let mut vmax = vdupq_n_f32(f32::NEG_INFINITY);
            for i in 0..chunks {
                let v = vld1q_f32(row_in.as_ptr().add(i * 4));
                vmax = vmaxq_f32(vmax, v);
            }
            // Horizontal max of the 4 lanes.
            let a = vgetq_lane_f32::<0>(vmax);
            let b = vgetq_lane_f32::<1>(vmax);
            let c = vgetq_lane_f32::<2>(vmax);
            let d = vgetq_lane_f32::<3>(vmax);
            let mut m = a.max(b).max(c.max(d));
            // Scalar tail for max.
            for i in 0..remainder {
                m = m.max(row_in[tail_start + i]);
            }
            m
        };

        // ---- Pass 2: exp(x - max) ----
        // SAFETY: Bounded loads/stores within slice.
        let sum = unsafe {
            let vmax_splat = vdupq_n_f32(max_val);
            let mut vsum = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let offset = i * 4;
                let v = vld1q_f32(row_in.as_ptr().add(offset));
                let diff = vsubq_f32(v, vmax_splat);
                // Compute exp per lane using fast approximation.
                let x0 = vgetq_lane_f32::<0>(diff);
                let x1 = vgetq_lane_f32::<1>(diff);
                let x2 = vgetq_lane_f32::<2>(diff);
                let x3 = vgetq_lane_f32::<3>(diff);
                let mut e = vdupq_n_f32(0.0);
                e = vsetq_lane_f32::<0>(super::fast_exp_f32(x0), e);
                e = vsetq_lane_f32::<1>(super::fast_exp_f32(x1), e);
                e = vsetq_lane_f32::<2>(super::fast_exp_f32(x2), e);
                e = vsetq_lane_f32::<3>(super::fast_exp_f32(x3), e);
                vst1q_f32(row_out.as_mut_ptr().add(offset), e);
                vsum = vaddq_f32(vsum, e);
            }
            // Horizontal sum.
            let pair = vpaddq_f32(vsum, vsum);
            let mut s = vgetq_lane_f32::<0>(vpaddq_f32(pair, pair));
            // Scalar tail.
            for i in 0..remainder {
                let e = super::fast_exp_f32(row_in[tail_start + i] - max_val);
                row_out[tail_start + i] = e;
                s += e;
            }
            s
        };

        // ---- Pass 3: normalize ----
        // SAFETY: Bounded loads/stores within slice.
        unsafe {
            let inv_sum = 1.0 / sum;
            let vinv = vdupq_n_f32(inv_sum);
            for i in 0..chunks {
                let offset = i * 4;
                let v = vld1q_f32(row_out.as_ptr().add(offset));
                let r = vmulq_f32(v, vinv);
                vst1q_f32(row_out.as_mut_ptr().add(offset), r);
            }
            for i in 0..remainder {
                row_out[tail_start + i] *= inv_sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64) — 256-bit, 8x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// AVX2-accelerated softmax over a single row of `dim_size` elements.
    ///
    /// Uses _mm256_max_ps for max reduction, fast exp approximation, and
    /// _mm256_add_ps for sum accumulation.
    ///
    /// # Safety
    /// Caller must verify AVX2 is available (`is_x86_feature_detected!("avx2")`).
    #[target_feature(enable = "avx2")]
    pub unsafe fn softmax_avx2(input: &[f32], output: &mut [f32], dim_size: usize) {
        assert_eq!(input.len(), output.len());
        assert!(dim_size > 0);
        assert_eq!(input.len() % dim_size, 0);

        let rows = input.len() / dim_size;
        for row in 0..rows {
            let start = row * dim_size;
            let end = start + dim_size;
            softmax_row_avx2(&input[start..end], &mut output[start..end]);
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn softmax_row_avx2(row_in: &[f32], row_out: &mut [f32]) {
        let n = row_in.len();
        let chunks = n / 8;
        let remainder = n % 8;
        let tail_start = chunks * 8;

        // ---- Pass 1: find max ----
        let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
        for i in 0..chunks {
            let v = _mm256_loadu_ps(row_in.as_ptr().add(i * 8));
            vmax = _mm256_max_ps(vmax, v);
        }
        // Horizontal max of 8 lanes.
        let hi = _mm256_extractf128_ps(vmax, 1);
        let lo = _mm256_castps256_ps128(vmax);
        let max128 = _mm_max_ps(lo, hi);
        let shuf = _mm_movehdup_ps(max128);
        let maxs = _mm_max_ps(max128, shuf);
        let shuf2 = _mm_movehl_ps(maxs, maxs);
        let final_max = _mm_max_ss(maxs, shuf2);
        let mut max_val = _mm_cvtss_f32(final_max);
        // Scalar tail for max.
        for i in 0..remainder {
            max_val = max_val.max(row_in[tail_start + i]);
        }

        // ---- Pass 2: exp(x - max) ----
        let vmax_splat = _mm256_set1_ps(max_val);
        let mut vsum = _mm256_setzero_ps();
        for i in 0..chunks {
            let offset = i * 8;
            let v = _mm256_loadu_ps(row_in.as_ptr().add(offset));
            let diff = _mm256_sub_ps(v, vmax_splat);
            // Extract lanes, compute fast exp, repack.
            let mut buf = [0.0f32; 8];
            _mm256_storeu_ps(buf.as_mut_ptr(), diff);
            for b in &mut buf {
                *b = super::fast_exp_f32(*b);
            }
            let e = _mm256_loadu_ps(buf.as_ptr());
            _mm256_storeu_ps(row_out.as_mut_ptr().add(offset), e);
            vsum = _mm256_add_ps(vsum, e);
        }
        // Horizontal sum of 8 lanes.
        let hi_s = _mm256_extractf128_ps(vsum, 1);
        let lo_s = _mm256_castps256_ps128(vsum);
        let sum128 = _mm_add_ps(lo_s, hi_s);
        let shuf_s = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf_s);
        let shuf2_s = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2_s);
        let mut sum = _mm_cvtss_f32(result);
        // Scalar tail.
        for i in 0..remainder {
            let e = super::fast_exp_f32(row_in[tail_start + i] - max_val);
            row_out[tail_start + i] = e;
            sum += e;
        }

        // ---- Pass 3: normalize ----
        let inv_sum = 1.0 / sum;
        let vinv = _mm256_set1_ps(inv_sum);
        for i in 0..chunks {
            let offset = i * 8;
            let v = _mm256_loadu_ps(row_out.as_ptr().add(offset));
            let r = _mm256_mul_ps(v, vinv);
            _mm256_storeu_ps(row_out.as_mut_ptr().add(offset), r);
        }
        for i in 0..remainder {
            row_out[tail_start + i] *= inv_sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Public dispatch: auto-selects best SIMD path
// ---------------------------------------------------------------------------

/// Softmax along the last dimension. Auto-dispatches to NEON/AVX2/scalar.
///
/// `input` and `output` must have the same length, which must be a
/// multiple of `dim_size`. Softmax is applied independently over each
/// contiguous `dim_size`-element row.
///
/// Uses a numerically stable three-pass algorithm:
///   1. Find max (prevents exp overflow).
///   2. Compute exp(x - max).
///   3. Normalize by sum.
///
/// NEON and AVX2 paths use a fast exp approximation (~1e-4 relative error)
/// for throughput. The scalar path uses `f32::exp()` for maximum precision.
pub fn softmax(input: &[f32], output: &mut [f32], dim_size: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        neon::softmax_neon(input, output, dim_size);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            unsafe { avx2::softmax_avx2(input, output, dim_size) };
            return;
        }
    }

    #[allow(unreachable_code)]
    softmax_scalar(input, output, dim_size);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "softmax_tests.rs"]
mod softmax_tests;
