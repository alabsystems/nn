// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized LayerNorm with explicit scalar/NEON/AVX2 entry points.
//!
//! This module provides named entry points for each SIMD tier:
//! - `layer_norm_f32_neon` — NEON-optimized (aarch64)
//! - `layer_norm_f32_avx2` — AVX2-optimized (x86_64)
//! - `layer_norm_f32_scalar` — pure scalar fallback
//! - `layer_norm_f32` — auto-dispatch to best available
//!
//! Two-pass algorithm:
//!   1. Compute mean and variance (two-pass for SIMD: sum then var).
//!   2. Normalize: `(x - mean) / sqrt(var + eps) * gamma + beta`.
//!
//! The existing `crate::layernorm` module provides the same algorithm;
//! this module exposes per-tier entry points with a simpler API signature
//! for benchmarking and differential testing.

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

/// Pure scalar LayerNorm. Uses f64 accumulation for numerical stability.
///
/// Processes `input[0..normalized_shape]` as a single row, writing
/// normalized + affine-transformed output.
///
/// # Arguments
/// * `input` — input slice of length `normalized_shape`
/// * `output` — output slice of length `normalized_shape`
/// * `gamma` — per-feature scale of length `normalized_shape`
/// * `beta` — per-feature shift of length `normalized_shape`
/// * `normalized_shape` — size of the normalization dimension
/// * `eps` — small constant for numerical stability (typically 1e-5)
pub fn layer_norm_f32_scalar(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    normalized_shape: usize,
    eps: f32,
) {
    assert!(normalized_shape > 0, "normalized_shape must be > 0");
    assert_eq!(
        input.len(),
        normalized_shape,
        "input length must equal normalized_shape"
    );
    assert_eq!(
        output.len(),
        normalized_shape,
        "output length must equal normalized_shape"
    );
    assert_eq!(
        gamma.len(),
        normalized_shape,
        "gamma length must equal normalized_shape"
    );
    assert_eq!(
        beta.len(),
        normalized_shape,
        "beta length must equal normalized_shape"
    );

    // Pass 1: mean (f64 for precision)
    let mut sum = 0.0_f64;
    for &x in &input[..normalized_shape] {
        sum += f64::from(x);
    }
    let mean = sum / normalized_shape as f64;

    // Pass 2: variance
    let mut var_sum = 0.0_f64;
    for &x in &input[..normalized_shape] {
        let d = f64::from(x) - mean;
        var_sum += d * d;
    }
    let variance = var_sum / normalized_shape as f64;
    let inv_std = 1.0 / (variance + f64::from(eps)).sqrt();

    let mean_f32 = mean as f32;
    let inv_std_f32 = inv_std as f32;

    // Pass 3: normalize + affine
    for i in 0..normalized_shape {
        let normalized = (input[i] - mean_f32) * inv_std_f32;
        output[i] = normalized * gamma[i] + beta[i];
    }
}

/// Reference implementation returning a new Vec.
pub fn layer_norm_f32_reference(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    normalized_shape: usize,
    eps: f32,
) -> Vec<f32> {
    let mut output = vec![0.0f32; normalized_shape];
    layer_norm_f32_scalar(input, &mut output, gamma, beta, normalized_shape, eps);
    output
}

// ---------------------------------------------------------------------------
// NEON (aarch64) — 128-bit, 4x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
pub fn layer_norm_f32_neon(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    normalized_shape: usize,
    eps: f32,
) {
    use std::arch::aarch64::*;

    assert!(normalized_shape > 0);
    assert_eq!(input.len(), normalized_shape);
    assert_eq!(output.len(), normalized_shape);
    assert_eq!(gamma.len(), normalized_shape);
    assert_eq!(beta.len(), normalized_shape);

    let n = normalized_shape;
    let chunks = n / 4;
    let remainder = n % 4;
    let tail_start = chunks * 4;
    let n_f32 = n as f32;

    // ---- Pass 1: compute mean ----
    // SAFETY: aarch64 NEON is always available. Bounded loads within slice.
    let mean = unsafe {
        let mut vsum = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(input.as_ptr().add(i * 4));
            vsum = vaddq_f32(vsum, v);
        }
        let pair = vpaddq_f32(vsum, vsum);
        let mut s = vgetq_lane_f32::<0>(vpaddq_f32(pair, pair));
        for i in 0..remainder {
            s += input[tail_start + i];
        }
        s / n_f32
    };

    // ---- Pass 2: compute variance ----
    // SAFETY: Bounded loads within slice.
    let variance = unsafe {
        let vmean = vdupq_n_f32(mean);
        let mut vvar = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(input.as_ptr().add(i * 4));
            let diff = vsubq_f32(v, vmean);
            vvar = vfmaq_f32(vvar, diff, diff);
        }
        let pair = vpaddq_f32(vvar, vvar);
        let mut s = vgetq_lane_f32::<0>(vpaddq_f32(pair, pair));
        for i in 0..remainder {
            let d = input[tail_start + i] - mean;
            s += d * d;
        }
        s / n_f32
    };

    let inv_std = 1.0 / (variance + eps).sqrt();

    // ---- Pass 3: normalize + affine ----
    // SAFETY: Bounded loads/stores within slice.
    unsafe {
        let vmean = vdupq_n_f32(mean);
        let vinv_std = vdupq_n_f32(inv_std);
        for i in 0..chunks {
            let offset = i * 4;
            let v = vld1q_f32(input.as_ptr().add(offset));
            let g = vld1q_f32(gamma.as_ptr().add(offset));
            let b = vld1q_f32(beta.as_ptr().add(offset));
            let diff = vsubq_f32(v, vmean);
            let normed = vmulq_f32(diff, vinv_std);
            let scaled = vfmaq_f32(b, normed, g); // b + normed * g
            vst1q_f32(output.as_mut_ptr().add(offset), scaled);
        }
        for i in 0..remainder {
            let idx = tail_start + i;
            let normalized = (input[idx] - mean) * inv_std;
            output[idx] = normalized * gamma[idx] + beta[idx];
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn layer_norm_f32_neon(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    normalized_shape: usize,
    eps: f32,
) {
    layer_norm_f32_scalar(input, output, gamma, beta, normalized_shape, eps);
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64) — 256-bit, 8x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
pub fn layer_norm_f32_avx2(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    normalized_shape: usize,
    eps: f32,
) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: AVX2+FMA detected above.
        unsafe {
            layer_norm_f32_avx2_inner(input, output, gamma, beta, normalized_shape, eps);
        }
    } else {
        layer_norm_f32_scalar(input, output, gamma, beta, normalized_shape, eps);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn layer_norm_f32_avx2_inner(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    normalized_shape: usize,
    eps: f32,
) {
    use std::arch::x86_64::*;

    assert!(normalized_shape > 0);
    assert_eq!(input.len(), normalized_shape);
    assert_eq!(output.len(), normalized_shape);
    assert_eq!(gamma.len(), normalized_shape);
    assert_eq!(beta.len(), normalized_shape);

    let n = normalized_shape;
    let chunks = n / 8;
    let remainder = n % 8;
    let tail_start = chunks * 8;
    let n_f32 = n as f32;

    // ---- Pass 1: compute mean ----
    let mut vsum = _mm256_setzero_ps();
    for i in 0..chunks {
        let v = _mm256_loadu_ps(input.as_ptr().add(i * 8));
        vsum = _mm256_add_ps(vsum, v);
    }
    let mut s = hsum_avx2(vsum);
    for i in 0..remainder {
        s += input[tail_start + i];
    }
    let mean = s / n_f32;

    // ---- Pass 2: compute variance ----
    let vmean = _mm256_set1_ps(mean);
    let mut vvar = _mm256_setzero_ps();
    for i in 0..chunks {
        let v = _mm256_loadu_ps(input.as_ptr().add(i * 8));
        let diff = _mm256_sub_ps(v, vmean);
        vvar = _mm256_fmadd_ps(diff, diff, vvar);
    }
    let mut v = hsum_avx2(vvar);
    for i in 0..remainder {
        let d = input[tail_start + i] - mean;
        v += d * d;
    }
    let variance = v / n_f32;
    let inv_std = 1.0 / (variance + eps).sqrt();

    // ---- Pass 3: normalize + affine ----
    let vinv_std = _mm256_set1_ps(inv_std);
    for i in 0..chunks {
        let offset = i * 8;
        let vi = _mm256_loadu_ps(input.as_ptr().add(offset));
        let g = _mm256_loadu_ps(gamma.as_ptr().add(offset));
        let b = _mm256_loadu_ps(beta.as_ptr().add(offset));
        let diff = _mm256_sub_ps(vi, vmean);
        let normed = _mm256_mul_ps(diff, vinv_std);
        let scaled = _mm256_fmadd_ps(normed, g, b); // normed * g + b
        _mm256_storeu_ps(output.as_mut_ptr().add(offset), scaled);
    }
    for i in 0..remainder {
        let idx = tail_start + i;
        let normalized = (input[idx] - mean) * inv_std;
        output[idx] = normalized * gamma[idx] + beta[idx];
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

#[cfg(not(target_arch = "x86_64"))]
pub fn layer_norm_f32_avx2(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    normalized_shape: usize,
    eps: f32,
) {
    layer_norm_f32_scalar(input, output, gamma, beta, normalized_shape, eps);
}

// ---------------------------------------------------------------------------
// Public dispatch: auto-selects best SIMD path
// ---------------------------------------------------------------------------

/// LayerNorm over a single row of `normalized_shape` elements.
/// Auto-dispatches to NEON (aarch64), AVX2 (x86_64), or scalar fallback.
///
/// # Arguments
/// * `input` — input slice of length `normalized_shape`
/// * `output` — output slice of length `normalized_shape`
/// * `gamma` — per-feature scale of length `normalized_shape`
/// * `beta` — per-feature shift of length `normalized_shape`
/// * `normalized_shape` — normalization dimension size
/// * `eps` — numerical stability constant (typically 1e-5)
pub fn layer_norm_f32(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    normalized_shape: usize,
    eps: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        layer_norm_f32_neon(input, output, gamma, beta, normalized_shape, eps);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            layer_norm_f32_avx2(input, output, gamma, beta, normalized_shape, eps);
            return;
        }
    }

    #[allow(unreachable_code)]
    layer_norm_f32_scalar(input, output, gamma, beta, normalized_shape, eps);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "simd_layernorm_tests.rs"]
mod simd_layernorm_tests;
