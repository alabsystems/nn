// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized flat reduction operations with explicit entry points.
//!
//! This module provides named entry points for each SIMD tier:
//! - `*_f32_neon` — NEON-optimized (aarch64)
//! - `*_f32_avx2` — AVX2-optimized (x86_64)
//! - `*_f32_scalar` — pure scalar fallback
//! - `*_f32` — auto-dispatch to best available
//!
//! Operations (flat — entire slice, no row/dim_size parameter):
//! - `sum_f32` — sum of all elements
//! - `max_f32` — maximum element
//! - `min_f32` — minimum element
//! - `dot_f32` — dot product of two slices
//! - `l2_norm_f32` — L2 (Euclidean) norm: sqrt(sum(x^2))
//!
//! The existing `crate::reduction` module provides sum/max/dot with the same
//! SIMD paths; this module adds `min_f32` and `l2_norm_f32` and exposes
//! per-tier entry points for benchmarking and differential testing.

// ---------------------------------------------------------------------------
// Scalar fallback (always compiled, no cfg gate)
// ---------------------------------------------------------------------------

/// Scalar sum of all elements.
#[inline]
pub fn sum_f32_scalar(x: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for &v in x {
        acc += v;
    }
    acc
}

/// Scalar max of all elements. Returns `f32::NEG_INFINITY` for empty input.
#[inline]
pub fn max_f32_scalar(x: &[f32]) -> f32 {
    let mut acc = f32::NEG_INFINITY;
    for &v in x {
        acc = acc.max(v);
    }
    acc
}

/// Scalar min of all elements. Returns `f32::INFINITY` for empty input.
#[inline]
pub fn min_f32_scalar(x: &[f32]) -> f32 {
    let mut acc = f32::INFINITY;
    for &v in x {
        acc = acc.min(v);
    }
    acc
}

/// Scalar dot product: `sum(a[i] * b[i])`.
#[inline]
pub fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "a and b must have equal length");
    let mut acc = 0.0_f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// Scalar L2 norm: `sqrt(sum(x[i]^2))`.
#[inline]
pub fn l2_norm_f32_scalar(x: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for &v in x {
        acc += v * v;
    }
    acc.sqrt()
}

// ---------------------------------------------------------------------------
// NEON (aarch64) — 128-bit, 4x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
pub fn sum_f32_neon(x: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = x.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: aarch64 NEON is always available. Bounded loads within slice.
    let mut result = unsafe {
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(x.as_ptr().add(i * 4));
            acc = vaddq_f32(acc, v);
        }
        // Horizontal sum: pairwise add twice.
        let pair = vpaddq_f32(acc, acc);
        vgetq_lane_f32::<0>(vpaddq_f32(pair, pair))
    };
    let tail = chunks * 4;
    for i in 0..remainder {
        result += x[tail + i];
    }
    result
}

#[cfg(not(target_arch = "aarch64"))]
pub fn sum_f32_neon(x: &[f32]) -> f32 {
    sum_f32_scalar(x)
}

#[cfg(target_arch = "aarch64")]
pub fn max_f32_neon(x: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    if x.is_empty() {
        return f32::NEG_INFINITY;
    }
    let n = x.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: aarch64 NEON is always available. Bounded loads within slice.
    let mut result = unsafe {
        let mut acc = vdupq_n_f32(f32::NEG_INFINITY);
        for i in 0..chunks {
            let v = vld1q_f32(x.as_ptr().add(i * 4));
            acc = vmaxq_f32(acc, v);
        }
        // Horizontal max: extract 4 lanes and fold.
        let a = vgetq_lane_f32::<0>(acc);
        let b = vgetq_lane_f32::<1>(acc);
        let c = vgetq_lane_f32::<2>(acc);
        let d = vgetq_lane_f32::<3>(acc);
        a.max(b).max(c.max(d))
    };
    let tail = chunks * 4;
    for i in 0..remainder {
        result = result.max(x[tail + i]);
    }
    result
}

#[cfg(not(target_arch = "aarch64"))]
pub fn max_f32_neon(x: &[f32]) -> f32 {
    max_f32_scalar(x)
}

#[cfg(target_arch = "aarch64")]
pub fn min_f32_neon(x: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    if x.is_empty() {
        return f32::INFINITY;
    }
    let n = x.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: aarch64 NEON is always available. Bounded loads within slice.
    let mut result = unsafe {
        let mut acc = vdupq_n_f32(f32::INFINITY);
        for i in 0..chunks {
            let v = vld1q_f32(x.as_ptr().add(i * 4));
            acc = vminq_f32(acc, v);
        }
        // Horizontal min: extract 4 lanes and fold.
        let a = vgetq_lane_f32::<0>(acc);
        let b = vgetq_lane_f32::<1>(acc);
        let c = vgetq_lane_f32::<2>(acc);
        let d = vgetq_lane_f32::<3>(acc);
        a.min(b).min(c.min(d))
    };
    let tail = chunks * 4;
    for i in 0..remainder {
        result = result.min(x[tail + i]);
    }
    result
}

#[cfg(not(target_arch = "aarch64"))]
pub fn min_f32_neon(x: &[f32]) -> f32 {
    min_f32_scalar(x)
}

#[cfg(target_arch = "aarch64")]
pub fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    assert_eq!(a.len(), b.len(), "a and b must have equal length");
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: aarch64 NEON is always available. vfmaq_f32 = acc + va*vb.
    let mut result = unsafe {
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            acc = vfmaq_f32(acc, va, vb);
        }
        let pair = vpaddq_f32(acc, acc);
        vgetq_lane_f32::<0>(vpaddq_f32(pair, pair))
    };
    let tail = chunks * 4;
    for i in 0..remainder {
        result += a[tail + i] * b[tail + i];
    }
    result
}

#[cfg(not(target_arch = "aarch64"))]
pub fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    dot_f32_scalar(a, b)
}

#[cfg(target_arch = "aarch64")]
pub fn l2_norm_f32_neon(x: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = x.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: aarch64 NEON is always available. vfmaq_f32(acc, v, v) = acc + v^2.
    let mut sum_sq = unsafe {
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(x.as_ptr().add(i * 4));
            acc = vfmaq_f32(acc, v, v);
        }
        let pair = vpaddq_f32(acc, acc);
        vgetq_lane_f32::<0>(vpaddq_f32(pair, pair))
    };
    let tail = chunks * 4;
    for i in 0..remainder {
        let v = x[tail + i];
        sum_sq += v * v;
    }
    sum_sq.sqrt()
}

#[cfg(not(target_arch = "aarch64"))]
pub fn l2_norm_f32_neon(x: &[f32]) -> f32 {
    l2_norm_f32_scalar(x)
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64) — 256-bit, 8x f32 lanes
// ---------------------------------------------------------------------------

/// Horizontal sum of an `__m256` register. Shared by AVX2 reductions.
///
/// # Safety
/// Caller must have verified AVX2 availability. Must be called within a
/// `#[target_feature(enable = "avx2")]` context.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_m256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(result)
}

#[cfg(target_arch = "x86_64")]
pub fn sum_f32_avx2(x: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 detected above.
        unsafe { sum_f32_avx2_inner(x) }
    } else {
        sum_f32_scalar(x)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_f32_avx2_inner(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = x.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut acc = _mm256_setzero_ps();
    for i in 0..chunks {
        // SAFETY: i*8 + 8 <= n from loop bound. Unaligned load.
        let v = _mm256_loadu_ps(x.as_ptr().add(i * 8));
        acc = _mm256_add_ps(acc, v);
    }
    let mut result = hsum_m256(acc);
    let tail = chunks * 8;
    for i in 0..remainder {
        result += x[tail + i];
    }
    result
}

#[cfg(not(target_arch = "x86_64"))]
pub fn sum_f32_avx2(x: &[f32]) -> f32 {
    sum_f32_scalar(x)
}

#[cfg(target_arch = "x86_64")]
pub fn max_f32_avx2(x: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 detected above.
        unsafe { max_f32_avx2_inner(x) }
    } else {
        max_f32_scalar(x)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn max_f32_avx2_inner(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    if x.is_empty() {
        return f32::NEG_INFINITY;
    }
    let n = x.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut acc = _mm256_set1_ps(f32::NEG_INFINITY);
    for i in 0..chunks {
        // SAFETY: i*8 + 8 <= n from loop bound. Unaligned load.
        let v = _mm256_loadu_ps(x.as_ptr().add(i * 8));
        acc = _mm256_max_ps(acc, v);
    }
    // Horizontal max.
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let max128 = _mm_max_ps(lo, hi);
    let shuf = _mm_movehdup_ps(max128);
    let maxs = _mm_max_ps(max128, shuf);
    let shuf2 = _mm_movehl_ps(maxs, maxs);
    let final_max = _mm_max_ss(maxs, shuf2);
    let mut result = _mm_cvtss_f32(final_max);
    let tail = chunks * 8;
    for i in 0..remainder {
        result = result.max(x[tail + i]);
    }
    result
}

#[cfg(not(target_arch = "x86_64"))]
pub fn max_f32_avx2(x: &[f32]) -> f32 {
    max_f32_scalar(x)
}

#[cfg(target_arch = "x86_64")]
pub fn min_f32_avx2(x: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 detected above.
        unsafe { min_f32_avx2_inner(x) }
    } else {
        min_f32_scalar(x)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn min_f32_avx2_inner(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    if x.is_empty() {
        return f32::INFINITY;
    }
    let n = x.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut acc = _mm256_set1_ps(f32::INFINITY);
    for i in 0..chunks {
        // SAFETY: i*8 + 8 <= n from loop bound. Unaligned load.
        let v = _mm256_loadu_ps(x.as_ptr().add(i * 8));
        acc = _mm256_min_ps(acc, v);
    }
    // Horizontal min.
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let min128 = _mm_min_ps(lo, hi);
    let shuf = _mm_movehdup_ps(min128);
    let mins = _mm_min_ps(min128, shuf);
    let shuf2 = _mm_movehl_ps(mins, mins);
    let final_min = _mm_min_ss(mins, shuf2);
    let mut result = _mm_cvtss_f32(final_min);
    let tail = chunks * 8;
    for i in 0..remainder {
        result = result.min(x[tail + i]);
    }
    result
}

#[cfg(not(target_arch = "x86_64"))]
pub fn min_f32_avx2(x: &[f32]) -> f32 {
    min_f32_scalar(x)
}

#[cfg(target_arch = "x86_64")]
pub fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: AVX2+FMA detected above.
        unsafe { dot_f32_avx2_inner(a, b) }
    } else {
        dot_f32_scalar(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_f32_avx2_inner(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    assert_eq!(a.len(), b.len(), "a and b must have equal length");
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut acc = _mm256_setzero_ps();
    for i in 0..chunks {
        // SAFETY: i*8 + 8 <= n from loop bound. Unaligned loads.
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    let mut result = hsum_m256(acc);
    let tail = chunks * 8;
    for i in 0..remainder {
        result += a[tail + i] * b[tail + i];
    }
    result
}

#[cfg(not(target_arch = "x86_64"))]
pub fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    dot_f32_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
pub fn l2_norm_f32_avx2(x: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: AVX2+FMA detected above.
        unsafe { l2_norm_f32_avx2_inner(x) }
    } else {
        l2_norm_f32_scalar(x)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn l2_norm_f32_avx2_inner(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = x.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut acc = _mm256_setzero_ps();
    for i in 0..chunks {
        // SAFETY: i*8 + 8 <= n from loop bound. Unaligned load.
        let v = _mm256_loadu_ps(x.as_ptr().add(i * 8));
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    let mut sum_sq = hsum_m256(acc);
    let tail = chunks * 8;
    for i in 0..remainder {
        let v = x[tail + i];
        sum_sq += v * v;
    }
    sum_sq.sqrt()
}

#[cfg(not(target_arch = "x86_64"))]
pub fn l2_norm_f32_avx2(x: &[f32]) -> f32 {
    l2_norm_f32_scalar(x)
}

// ---------------------------------------------------------------------------
// Public dispatch: auto-selects best SIMD path
// ---------------------------------------------------------------------------

/// Sum all elements of `x`. Auto-dispatches to NEON/AVX2/scalar.
#[inline]
pub fn sum_f32(x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return sum_f32_neon(x);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return sum_f32_avx2(x);
        }
    }

    #[allow(unreachable_code)]
    sum_f32_scalar(x)
}

/// Maximum element of `x`. Returns `f32::NEG_INFINITY` for empty input.
/// Auto-dispatches to NEON/AVX2/scalar.
#[inline]
pub fn max_f32(x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return max_f32_neon(x);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return max_f32_avx2(x);
        }
    }

    #[allow(unreachable_code)]
    max_f32_scalar(x)
}

/// Minimum element of `x`. Returns `f32::INFINITY` for empty input.
/// Auto-dispatches to NEON/AVX2/scalar.
#[inline]
pub fn min_f32(x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return min_f32_neon(x);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return min_f32_avx2(x);
        }
    }

    #[allow(unreachable_code)]
    min_f32_scalar(x)
}

/// Dot product of `a` and `b`: `sum(a[i] * b[i])`.
/// Auto-dispatches to NEON/AVX2/scalar.
///
/// Uses hardware FMA on supported platforms.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return dot_f32_neon(a, b);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return dot_f32_avx2(a, b);
        }
    }

    #[allow(unreachable_code)]
    dot_f32_scalar(a, b)
}

/// L2 (Euclidean) norm: `sqrt(sum(x[i]^2))`.
/// Auto-dispatches to NEON/AVX2/scalar.
///
/// Uses hardware FMA for the squared-sum accumulation.
#[inline]
pub fn l2_norm_f32(x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return l2_norm_f32_neon(x);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return l2_norm_f32_avx2(x);
        }
    }

    #[allow(unreachable_code)]
    l2_norm_f32_scalar(x)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "simd_reduce_tests.rs"]
mod simd_reduce_tests;
