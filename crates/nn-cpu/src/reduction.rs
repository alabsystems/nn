// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-vectorized reduction operations: sum, max, dot product.
//!
//! NEON (aarch64) and AVX2 (x86_64) paths with scalar fallback.

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

/// Scalar sum of all elements.
pub fn sum_scalar(input: &[f32]) -> f32 {
    input.iter().copied().fold(0.0, |a, b| a + b)
}

/// Scalar max of all elements. Returns `f32::NEG_INFINITY` for empty input.
pub fn max_scalar(input: &[f32]) -> f32 {
    input.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// Scalar dot product of two equal-length slices.
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * y)
        .fold(0.0, |s, v| s + v)
}

// ---------------------------------------------------------------------------
// NEON (aarch64)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// NEON horizontal sum of a float32x4 vector.
    ///
    /// # Safety
    /// NEON is always available on aarch64.
    #[inline]
    unsafe fn hsum_f32x4(v: float32x4_t) -> f32 { unsafe {
        let pair = vpaddq_f32(v, v); // [a+b, c+d, a+b, c+d]
        vgetq_lane_f32::<0>(vpaddq_f32(pair, pair))
    }}

    /// NEON sum reduction.
    pub(super) fn sum_neon(input: &[f32]) -> f32 {
        let n = input.len();
        let chunks = n / 4;
        let remainder = n % 4;

        // SAFETY: NEON always available on aarch64. Bounded loads.
        let mut acc = unsafe { vdupq_n_f32(0.0) };
        unsafe {
            for i in 0..chunks {
                let v = vld1q_f32(input.as_ptr().add(i * 4));
                acc = vaddq_f32(acc, v);
            }
        }
        // SAFETY: horizontal sum of NEON register.
        let mut result = unsafe { hsum_f32x4(acc) };
        let tail_start = chunks * 4;
        for i in 0..remainder {
            result += input[tail_start + i];
        }
        result
    }

    /// NEON max reduction.
    pub(super) fn max_neon(input: &[f32]) -> f32 {
        if input.is_empty() {
            return f32::NEG_INFINITY;
        }
        let n = input.len();
        let chunks = n / 4;
        let remainder = n % 4;

        // SAFETY: NEON always available on aarch64. Bounded loads.
        let mut acc = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };
        unsafe {
            for i in 0..chunks {
                let v = vld1q_f32(input.as_ptr().add(i * 4));
                acc = vmaxq_f32(acc, v);
            }
        }
        // Horizontal max: extract lanes and fold.
        // SAFETY: extracting lanes from NEON register.
        let mut result = unsafe {
            let a = vgetq_lane_f32::<0>(acc);
            let b = vgetq_lane_f32::<1>(acc);
            let c = vgetq_lane_f32::<2>(acc);
            let d = vgetq_lane_f32::<3>(acc);
            a.max(b).max(c.max(d))
        };
        let tail_start = chunks * 4;
        for i in 0..remainder {
            result = result.max(input[tail_start + i]);
        }
        result
    }

    /// NEON dot product.
    pub(super) fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let n = a.len();
        let chunks = n / 4;
        let remainder = n % 4;

        // SAFETY: NEON always available on aarch64. Bounded loads.
        let mut acc = unsafe { vdupq_n_f32(0.0) };
        unsafe {
            for i in 0..chunks {
                let va = vld1q_f32(a.as_ptr().add(i * 4));
                let vb = vld1q_f32(b.as_ptr().add(i * 4));
                acc = vfmaq_f32(acc, va, vb); // fused multiply-add
            }
        }
        // SAFETY: horizontal sum of NEON register.
        let mut result = unsafe { hsum_f32x4(acc) };
        let tail_start = chunks * 4;
        for i in 0..remainder {
            result += a[tail_start + i] * b[tail_start + i];
        }
        result
    }
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// Horizontal sum of an __m256 register.
    ///
    /// # Safety
    /// Caller must have verified AVX2 availability.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hsum_m256(v: __m256) -> f32 {
        // [a0+a4, a1+a5, a2+a6, a3+a7] via hadd pattern
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128); // [s1, s1, s3, s3]
        let sums = _mm_add_ps(sum128, shuf); // [s0+s1, _, s2+s3, _]
        let shuf2 = _mm_movehl_ps(sums, sums); // [s2+s3, _, _, _]
        let result = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(result)
    }

    /// AVX2 sum reduction.
    ///
    /// # Safety
    /// Caller must verify AVX2 is available.
    #[target_feature(enable = "avx2")]
    pub unsafe fn sum_avx2(input: &[f32]) -> f32 {
        let n = input.len();
        let chunks = n / 8;
        let remainder = n % 8;

        let mut acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let v = _mm256_loadu_ps(input.as_ptr().add(i * 8));
            acc = _mm256_add_ps(acc, v);
        }
        let mut result = hsum_m256(acc);
        let tail_start = chunks * 8;
        for i in 0..remainder {
            result += input[tail_start + i];
        }
        result
    }

    /// AVX2 max reduction.
    ///
    /// # Safety
    /// Caller must verify AVX2 is available.
    #[target_feature(enable = "avx2")]
    pub unsafe fn max_avx2(input: &[f32]) -> f32 {
        if input.is_empty() {
            return f32::NEG_INFINITY;
        }
        let n = input.len();
        let chunks = n / 8;
        let remainder = n % 8;

        let mut acc = _mm256_set1_ps(f32::NEG_INFINITY);
        for i in 0..chunks {
            let v = _mm256_loadu_ps(input.as_ptr().add(i * 8));
            acc = _mm256_max_ps(acc, v);
        }
        // Horizontal max
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let max128 = _mm_max_ps(lo, hi);
        let shuf = _mm_movehdup_ps(max128);
        let maxs = _mm_max_ps(max128, shuf);
        let shuf2 = _mm_movehl_ps(maxs, maxs);
        let final_max = _mm_max_ss(maxs, shuf2);
        let mut result = _mm_cvtss_f32(final_max);
        let tail_start = chunks * 8;
        for i in 0..remainder {
            result = result.max(input[tail_start + i]);
        }
        result
    }

    /// AVX2 dot product using FMA.
    ///
    /// # Safety
    /// Caller must verify AVX2 (and FMA) is available.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        let mut acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }
        let mut result = hsum_m256(acc);
        let tail_start = chunks * 8;
        for i in 0..remainder {
            result += a[tail_start + i] * b[tail_start + i];
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Public dispatch: auto-selects best SIMD path
// ---------------------------------------------------------------------------

/// Sum all elements. Auto-dispatches to NEON/AVX2/scalar.
pub fn sum(input: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return neon::sum_neon(input);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            return unsafe { avx2::sum_avx2(input) };
        }
    }

    #[allow(unreachable_code)]
    sum_scalar(input)
}

/// Max of all elements. Auto-dispatches to NEON/AVX2/scalar.
pub fn max(input: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return neon::max_neon(input);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            return unsafe { avx2::max_avx2(input) };
        }
    }

    #[allow(unreachable_code)]
    max_scalar(input)
}

/// Dot product of two equal-length slices. Auto-dispatches to NEON/AVX2/scalar.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return neon::dot_neon(a, b);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA detected above.
            return unsafe { avx2::dot_avx2(a, b) };
        }
    }

    #[allow(unreachable_code)]
    dot_scalar(a, b)
}
