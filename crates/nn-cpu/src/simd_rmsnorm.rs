// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized RMS Normalization with explicit scalar/NEON/AVX2 entry points.
//!
//! This module provides:
//! - `rmsnorm` — single vector RMSNorm, auto-dispatch
//! - `rmsnorm_batch` — batched RMSNorm, auto-dispatch
//! - `rmsnorm_reference` — pure scalar reference (returns new Vec)
//!
//! RMSNorm: `rms = sqrt(mean(x^2) + eps)`, `out = (x / rms) * weight`
//!
//! Used in modern LLMs (LLaMA, Qwen, etc.) as a simpler alternative to LayerNorm.

// ---------------------------------------------------------------------------
// Scalar reference (returns new Vec)
// ---------------------------------------------------------------------------

/// Pure scalar RMSNorm reference implementation returning a new Vec.
///
/// Computes `rms = sqrt(mean(x^2) + eps)`, then `out[i] = (x[i] / rms) * weight[i]`.
pub fn rmsnorm_reference(input: &[f32], weight: &[f32], hidden_size: usize, eps: f32) -> Vec<f32> {
    assert_eq!(
        input.len(),
        hidden_size,
        "input length must equal hidden_size"
    );
    assert_eq!(
        weight.len(),
        hidden_size,
        "weight length must equal hidden_size"
    );

    // Accumulate in f64 for numerical stability
    let sum_sq: f64 = input.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let rms = ((sum_sq / hidden_size as f64) + f64::from(eps)).sqrt();
    let inv_rms = 1.0 / rms;

    input
        .iter()
        .zip(weight.iter())
        .map(|(&x, &w)| ((f64::from(x) * inv_rms) * f64::from(w)) as f32)
        .collect()
}

// ---------------------------------------------------------------------------
// Scalar fallback (out-of-place)
// ---------------------------------------------------------------------------

/// Scalar RMSNorm for a single vector.
fn rmsnorm_scalar(input: &[f32], weight: &[f32], output: &mut [f32], hidden_size: usize, eps: f32) {
    // f64 accumulation for numerical stability
    let sum_sq: f64 = input[..hidden_size]
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum();
    let rms = ((sum_sq / hidden_size as f64) + f64::from(eps)).sqrt();
    let inv_rms = (1.0 / rms) as f32;

    for i in 0..hidden_size {
        output[i] = input[i] * inv_rms * weight[i];
    }
}

// ---------------------------------------------------------------------------
// NEON (aarch64) — 128-bit, 4x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
fn rmsnorm_neon(input: &[f32], weight: &[f32], output: &mut [f32], hidden_size: usize, eps: f32) {
    use std::arch::aarch64::*;

    let chunks = hidden_size / 4;
    let remainder = hidden_size % 4;
    let tail_start = chunks * 4;

    // Pass 1: compute sum of squares
    // SAFETY: aarch64 NEON always available. All loads bounded by hidden_size.
    let sum_sq = unsafe {
        let mut vsum = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(input.as_ptr().add(i * 4));
            vsum = vfmaq_f32(vsum, v, v);
        }
        // Horizontal sum of 4 lanes
        let pair = vpaddq_f32(vsum, vsum);
        let mut s = vgetq_lane_f32::<0>(vpaddq_f32(pair, pair));
        for i in 0..remainder {
            let x = input[tail_start + i];
            s += x * x;
        }
        s
    };

    let rms = (sum_sq / hidden_size as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    // Pass 2: normalize and scale
    // SAFETY: All loads/stores bounded by hidden_size.
    unsafe {
        let vinv = vdupq_n_f32(inv_rms);
        for i in 0..chunks {
            let offset = i * 4;
            let v = vld1q_f32(input.as_ptr().add(offset));
            let w = vld1q_f32(weight.as_ptr().add(offset));
            let normed = vmulq_f32(v, vinv);
            let scaled = vmulq_f32(normed, w);
            vst1q_f32(output.as_mut_ptr().add(offset), scaled);
        }
    }
    for i in 0..remainder {
        let idx = tail_start + i;
        output[idx] = input[idx] * inv_rms * weight[idx];
    }
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64) — 256-bit, 8x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn rmsnorm_avx2_inner(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    hidden_size: usize,
    eps: f32,
) {
    use std::arch::x86_64::*;

    let chunks = hidden_size / 8;
    let remainder = hidden_size % 8;
    let tail_start = chunks * 8;

    // Pass 1: compute sum of squares
    let mut vsum = _mm256_setzero_ps();
    for i in 0..chunks {
        // SAFETY: offset + 8 <= hidden_size from loop bound.
        let v = _mm256_loadu_ps(input.as_ptr().add(i * 8));
        vsum = _mm256_fmadd_ps(v, v, vsum);
    }
    let mut s = hsum_avx2(vsum);
    for i in 0..remainder {
        let x = input[tail_start + i];
        s += x * x;
    }

    let rms = (s / hidden_size as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    // Pass 2: normalize and scale
    let vinv = _mm256_set1_ps(inv_rms);
    for i in 0..chunks {
        let offset = i * 8;
        // SAFETY: offset + 8 <= hidden_size from loop bound.
        let v = _mm256_loadu_ps(input.as_ptr().add(offset));
        let w = _mm256_loadu_ps(weight.as_ptr().add(offset));
        let normed = _mm256_mul_ps(v, vinv);
        let scaled = _mm256_mul_ps(normed, w);
        _mm256_storeu_ps(output.as_mut_ptr().add(offset), scaled);
    }
    for i in 0..remainder {
        let idx = tail_start + i;
        output[idx] = input[idx] * inv_rms * weight[idx];
    }
}

/// Horizontal sum of 8 f32 lanes in a __m256.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx2(v: std::arch::x86_64::__m256) -> f32 {
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

// ---------------------------------------------------------------------------
// Public dispatch: single vector RMSNorm
// ---------------------------------------------------------------------------

/// RMS Normalization for a single vector of `hidden_size` elements.
///
/// Computes `rms = sqrt(mean(x^2) + eps)`, then `out[i] = (x[i] / rms) * weight[i]`.
///
/// Auto-dispatches to NEON (aarch64), AVX2 (x86_64), or scalar fallback.
///
/// # Panics
/// Panics if `input.len() != hidden_size`, `weight.len() != hidden_size`, or
/// `output.len() != hidden_size`.
pub fn rmsnorm(input: &[f32], weight: &[f32], output: &mut [f32], hidden_size: usize, eps: f32) {
    assert_eq!(
        input.len(),
        hidden_size,
        "input length must equal hidden_size"
    );
    assert_eq!(
        weight.len(),
        hidden_size,
        "weight length must equal hidden_size"
    );
    assert_eq!(
        output.len(),
        hidden_size,
        "output length must equal hidden_size"
    );
    assert!(hidden_size > 0, "hidden_size must be > 0");

    #[cfg(target_arch = "aarch64")]
    {
        rmsnorm_neon(input, weight, output, hidden_size, eps);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA detected above.
            unsafe {
                rmsnorm_avx2_inner(input, weight, output, hidden_size, eps);
            }
            return;
        }
    }

    #[allow(unreachable_code)]
    rmsnorm_scalar(input, weight, output, hidden_size, eps);
}

// ---------------------------------------------------------------------------
// Batched RMSNorm
// ---------------------------------------------------------------------------

/// Batched RMS Normalization.
///
/// Applies RMSNorm independently to each row of `batch_size` rows, each of
/// `hidden_size` elements. `weight` is shared across all rows.
///
/// Layout: `input` and `output` are `[batch_size, hidden_size]` flattened.
///
/// # Panics
/// Panics if slice lengths are inconsistent with `batch_size * hidden_size`.
pub fn rmsnorm_batch(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let total = batch_size * hidden_size;
    assert_eq!(
        input.len(),
        total,
        "input length must equal batch_size * hidden_size"
    );
    assert_eq!(
        output.len(),
        total,
        "output length must equal batch_size * hidden_size"
    );
    assert_eq!(
        weight.len(),
        hidden_size,
        "weight length must equal hidden_size"
    );

    for b in 0..batch_size {
        let start = b * hidden_size;
        let end = start + hidden_size;
        rmsnorm(
            &input[start..end],
            weight,
            &mut output[start..end],
            hidden_size,
            eps,
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "simd_rmsnorm_tests.rs"]
mod simd_rmsnorm_tests;
