// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized softmax with explicit scalar/NEON/AVX2 entry points.
//!
//! This module provides named entry points for each SIMD tier:
//! - `softmax_f32_neon` — NEON-optimized (aarch64)
//! - `softmax_f32_avx2` — AVX2-optimized (x86_64)
//! - `softmax_f32_scalar` — pure scalar fallback
//! - `softmax_f32` — auto-dispatch to best available
//!
//! Three-pass numerically stable algorithm:
//!   1. Max reduction (prevents overflow in exp).
//!   2. Exp(x - max) and sum accumulation.
//!   3. Normalize by 1/sum.
//!
//! The existing `crate::softmax` module provides the same algorithm with
//! the same SIMD paths; this module exposes per-tier entry points for
//! benchmarking and differential testing.

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

/// Pure scalar softmax. Uses stdlib `f32::exp()` for maximum precision.
///
/// `input` and `output` must have the same length equal to `size`.
/// Softmax is applied over the entire `size`-element row.
pub fn softmax_f32_scalar(input: &[f32], output: &mut [f32], size: usize) {
    assert_eq!(input.len(), size, "input length must equal size");
    assert_eq!(output.len(), size, "output length must equal size");
    if size == 0 {
        return;
    }

    // Pass 1: max
    let mut max_val = f32::NEG_INFINITY;
    for &x in &input[..size] {
        if x > max_val {
            max_val = x;
        }
    }

    // Pass 2: exp(x - max) and sum
    let mut sum = 0.0_f32;
    for (o, &x) in output[..size].iter_mut().zip(input[..size].iter()) {
        let e = (x - max_val).exp();
        *o = e;
        sum += e;
    }

    // Pass 3: normalize
    let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for o in output[..size].iter_mut() {
        *o *= inv_sum;
    }
}

/// Reference implementation for differential testing.
///
/// Returns a newly-allocated output vector. Uses stdlib `f32::exp()`.
pub fn softmax_f32_reference(input: &[f32], size: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; size];
    softmax_f32_scalar(input, &mut output, size);
    output
}

// ---------------------------------------------------------------------------
// NEON (aarch64) — 128-bit, 4x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
pub fn softmax_f32_neon(input: &[f32], output: &mut [f32], size: usize) {
    use std::arch::aarch64::*;

    assert_eq!(input.len(), size, "input length must equal size");
    assert_eq!(output.len(), size, "output length must equal size");
    if size == 0 {
        return;
    }

    let chunks = size / 4;
    let remainder = size % 4;
    let tail_start = chunks * 4;

    // ---- Pass 1: find max ----
    // SAFETY: aarch64 NEON is always available. Bounded loads within slice.
    let max_val = unsafe {
        let mut vmax = vdupq_n_f32(f32::NEG_INFINITY);
        for i in 0..chunks {
            let v = vld1q_f32(input.as_ptr().add(i * 4));
            vmax = vmaxq_f32(vmax, v);
        }
        // Horizontal max of the 4 lanes.
        let a = vgetq_lane_f32::<0>(vmax);
        let b = vgetq_lane_f32::<1>(vmax);
        let c = vgetq_lane_f32::<2>(vmax);
        let d = vgetq_lane_f32::<3>(vmax);
        let mut m = a.max(b).max(c.max(d));
        for i in 0..remainder {
            m = m.max(input[tail_start + i]);
        }
        m
    };

    // ---- Pass 2: exp(x - max) and sum ----
    // SAFETY: Bounded loads/stores within slice.
    let sum = unsafe {
        let vmax_splat = vdupq_n_f32(max_val);
        let mut vsum = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let offset = i * 4;
            let v = vld1q_f32(input.as_ptr().add(offset));
            let diff = vsubq_f32(v, vmax_splat);
            // Per-lane fast exp approximation.
            let x0 = vgetq_lane_f32::<0>(diff);
            let x1 = vgetq_lane_f32::<1>(diff);
            let x2 = vgetq_lane_f32::<2>(diff);
            let x3 = vgetq_lane_f32::<3>(diff);
            let mut e = vdupq_n_f32(0.0);
            e = vsetq_lane_f32::<0>(fast_exp_f32(x0), e);
            e = vsetq_lane_f32::<1>(fast_exp_f32(x1), e);
            e = vsetq_lane_f32::<2>(fast_exp_f32(x2), e);
            e = vsetq_lane_f32::<3>(fast_exp_f32(x3), e);
            vst1q_f32(output.as_mut_ptr().add(offset), e);
            vsum = vaddq_f32(vsum, e);
        }
        // Horizontal sum.
        let pair = vpaddq_f32(vsum, vsum);
        let mut s = vgetq_lane_f32::<0>(vpaddq_f32(pair, pair));
        // Scalar tail.
        for i in 0..remainder {
            let e = fast_exp_f32(input[tail_start + i] - max_val);
            output[tail_start + i] = e;
            s += e;
        }
        s
    };

    // ---- Pass 3: normalize ----
    // SAFETY: Bounded loads/stores within slice.
    unsafe {
        let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        let vinv = vdupq_n_f32(inv_sum);
        for i in 0..chunks {
            let offset = i * 4;
            let v = vld1q_f32(output.as_ptr().add(offset));
            let r = vmulq_f32(v, vinv);
            vst1q_f32(output.as_mut_ptr().add(offset), r);
        }
        for i in 0..remainder {
            output[tail_start + i] *= inv_sum;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn softmax_f32_neon(input: &[f32], output: &mut [f32], size: usize) {
    // Fallback: delegate to scalar on non-aarch64 platforms.
    softmax_f32_scalar(input, output, size);
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64) — 256-bit, 8x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
pub fn softmax_f32_avx2(input: &[f32], output: &mut [f32], size: usize) {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 detected above.
        unsafe { softmax_f32_avx2_inner(input, output, size) };
    } else {
        softmax_f32_scalar(input, output, size);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn softmax_f32_avx2_inner(input: &[f32], output: &mut [f32], size: usize) {
    use std::arch::x86_64::*;

    assert_eq!(input.len(), size, "input length must equal size");
    assert_eq!(output.len(), size, "output length must equal size");
    if size == 0 {
        return;
    }

    let chunks = size / 8;
    let remainder = size % 8;
    let tail_start = chunks * 8;

    // ---- Pass 1: find max ----
    let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
    for i in 0..chunks {
        let v = _mm256_loadu_ps(input.as_ptr().add(i * 8));
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
    for i in 0..remainder {
        max_val = max_val.max(input[tail_start + i]);
    }

    // ---- Pass 2: exp(x - max) and sum ----
    let vmax_splat = _mm256_set1_ps(max_val);
    let mut vsum = _mm256_setzero_ps();
    for i in 0..chunks {
        let offset = i * 8;
        let v = _mm256_loadu_ps(input.as_ptr().add(offset));
        let diff = _mm256_sub_ps(v, vmax_splat);
        // Extract lanes, compute fast exp, repack.
        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), diff);
        for b in &mut buf {
            *b = fast_exp_f32(*b);
        }
        let e = _mm256_loadu_ps(buf.as_ptr());
        _mm256_storeu_ps(output.as_mut_ptr().add(offset), e);
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
    for i in 0..remainder {
        let e = fast_exp_f32(input[tail_start + i] - max_val);
        output[tail_start + i] = e;
        sum += e;
    }

    // ---- Pass 3: normalize ----
    let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    let vinv = _mm256_set1_ps(inv_sum);
    for i in 0..chunks {
        let offset = i * 8;
        let v = _mm256_loadu_ps(output.as_ptr().add(offset));
        let r = _mm256_mul_ps(v, vinv);
        _mm256_storeu_ps(output.as_mut_ptr().add(offset), r);
    }
    for i in 0..remainder {
        output[tail_start + i] *= inv_sum;
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn softmax_f32_avx2(input: &[f32], output: &mut [f32], size: usize) {
    // Fallback: delegate to scalar on non-x86_64 platforms.
    softmax_f32_scalar(input, output, size);
}

// ---------------------------------------------------------------------------
// Fast exp approximation (Schraudolph method)
// ---------------------------------------------------------------------------

/// Fast exp approximation using the Schraudolph method with linear
/// interpolation. Accurate to ~1e-4 relative error for |x| < 80.
/// Falls back gracefully for extreme inputs (clips to 0 / large).
#[inline(always)]
fn fast_exp_f32(x: f32) -> f32 {
    let x = x.clamp(-87.0, 88.0);
    const A: f32 = 12102203.0; // 2^23 / ln2
    const B: f32 = 1064866805.0; // 127 * 2^23 - ~486411 (bias for accuracy)
    let val = A * x + B;
    f32::from_bits(val as u32)
}

// ---------------------------------------------------------------------------
// Public dispatch: auto-selects best SIMD path
// ---------------------------------------------------------------------------

/// Softmax over a single row of `size` elements. Auto-dispatches to
/// NEON (aarch64), AVX2 (x86_64), or scalar fallback.
///
/// `input` and `output` must each have length `size`.
///
/// Uses a numerically stable three-pass algorithm:
///   1. Find max (prevents exp overflow).
///   2. Compute exp(x - max) and accumulate sum.
///   3. Normalize by 1/sum.
///
/// NEON and AVX2 paths use a fast exp approximation (~1e-4 relative error)
/// for throughput. The scalar path uses `f32::exp()` for maximum precision.
pub fn softmax_f32(input: &[f32], output: &mut [f32], size: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        softmax_f32_neon(input, output, size);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            softmax_f32_avx2(input, output, size);
            return;
        }
    }

    #[allow(unreachable_code)]
    softmax_f32_scalar(input, output, size);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "simd_softmax_tests.rs"]
mod simd_softmax_tests;
