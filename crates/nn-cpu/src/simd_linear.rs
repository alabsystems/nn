// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized fused linear layer: y = x @ W^T + b.
//!
//! This module provides:
//! - `linear` — y = x @ W^T + b (single vector)
//! - `linear_no_bias` — y = x @ W^T (single vector)
//! - `linear_batched` — batched linear with bias
//!
//! Weight layout: `[out_features, in_features]` (row-major, each row is one output neuron).
//!
//! NEON (aarch64) and AVX2 (x86_64) paths use vectorized dot products.
//! Scalar fallbacks are always available.

// ---------------------------------------------------------------------------
// Scalar reference implementations
// ---------------------------------------------------------------------------

/// Reference linear: y = x @ W^T + b.
///
/// `weight` is `[out_features, in_features]` row-major.
/// `input` has length `in_features`.
/// `bias` has length `out_features`.
/// `output` has length `out_features`.
pub fn linear_reference(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    in_features: usize,
    out_features: usize,
) {
    assert_eq!(
        input.len(),
        in_features,
        "input length must equal in_features"
    );
    assert_eq!(
        weight.len(),
        out_features * in_features,
        "weight length must equal out_features * in_features"
    );
    assert_eq!(
        bias.len(),
        out_features,
        "bias length must equal out_features"
    );
    assert_eq!(
        output.len(),
        out_features,
        "output length must equal out_features"
    );

    for o in 0..out_features {
        let mut acc = bias[o];
        let row_offset = o * in_features;
        for k in 0..in_features {
            acc += input[k] * weight[row_offset + k];
        }
        output[o] = acc;
    }
}

/// Reference linear without bias: y = x @ W^T.
pub fn linear_no_bias_reference(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    in_features: usize,
    out_features: usize,
) {
    assert_eq!(
        input.len(),
        in_features,
        "input length must equal in_features"
    );
    assert_eq!(
        weight.len(),
        out_features * in_features,
        "weight length must equal out_features * in_features"
    );
    assert_eq!(
        output.len(),
        out_features,
        "output length must equal out_features"
    );

    for o in 0..out_features {
        let mut acc = 0.0f32;
        let row_offset = o * in_features;
        for k in 0..in_features {
            acc += input[k] * weight[row_offset + k];
        }
        output[o] = acc;
    }
}

/// Reference batched linear: y[b] = x[b] @ W^T + bias for each batch element.
pub fn linear_batched_reference(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    batch: usize,
    in_features: usize,
    out_features: usize,
) {
    assert_eq!(
        input.len(),
        batch * in_features,
        "input length must equal batch * in_features"
    );
    assert_eq!(
        weight.len(),
        out_features * in_features,
        "weight length must equal out_features * in_features"
    );
    assert_eq!(
        bias.len(),
        out_features,
        "bias length must equal out_features"
    );
    assert_eq!(
        output.len(),
        batch * out_features,
        "output length must equal batch * out_features"
    );

    for b in 0..batch {
        let in_offset = b * in_features;
        let out_offset = b * out_features;
        linear_reference(
            &input[in_offset..in_offset + in_features],
            weight,
            bias,
            &mut output[out_offset..out_offset + out_features],
            in_features,
            out_features,
        );
    }
}

// ---------------------------------------------------------------------------
// SIMD dot product helpers
// ---------------------------------------------------------------------------

/// Scalar dot product.
#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// NEON dot product (aarch64).
#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut acc;
    // SAFETY: aarch64 NEON is always available. Bounded loads within slice.
    unsafe {
        let mut vacc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));
            vacc = vfmaq_f32(vacc, va, vb);
        }
        acc = vaddvq_f32(vacc);
    }
    let tail = chunks * 4;
    for i in 0..remainder {
        acc += a[tail + i] * b[tail + i];
    }
    acc
}

/// AVX2 dot product (x86_64) — inner function with target_feature.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_avx2_inner(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut vacc = _mm256_setzero_ps();
    for i in 0..chunks {
        let offset = i * 8;
        // SAFETY: offset + 8 <= n from loop bound.
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        vacc = _mm256_fmadd_ps(va, vb, vacc);
    }
    // Horizontal sum of 8 lanes
    let hi = _mm256_extractf128_ps::<1>(vacc);
    let lo = _mm256_castps256_ps128(vacc);
    let sum4 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum4);
    let sum2 = _mm_add_ss(sum4, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sum2);
    let sum1 = _mm_add_ss(sum2, shuf2);
    let mut acc = _mm_cvtss_f32(sum1);

    let tail = chunks * 8;
    for i in 0..remainder {
        acc += a[tail + i] * b[tail + i];
    }
    acc
}

/// Best-available dot product.
#[inline]
fn dot_best(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return dot_neon(a, b);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA detected above.
            return unsafe { dot_avx2_inner(a, b) };
        }
    }

    #[allow(unreachable_code)]
    dot_scalar(a, b)
}

// ---------------------------------------------------------------------------
// Public API: linear
// ---------------------------------------------------------------------------

/// Fused linear layer: `output[o] = dot(input, weight[o]) + bias[o]`.
///
/// `weight` is `[out_features, in_features]` row-major.
/// Uses NEON/AVX2 vectorized dot products where available.
pub fn linear(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    in_features: usize,
    out_features: usize,
) {
    assert_eq!(
        input.len(),
        in_features,
        "input length must equal in_features"
    );
    assert_eq!(
        weight.len(),
        out_features * in_features,
        "weight length must equal out_features * in_features"
    );
    assert_eq!(
        bias.len(),
        out_features,
        "bias length must equal out_features"
    );
    assert_eq!(
        output.len(),
        out_features,
        "output length must equal out_features"
    );

    for o in 0..out_features {
        let row_start = o * in_features;
        let row = &weight[row_start..row_start + in_features];
        output[o] = dot_best(input, row) + bias[o];
    }
}

/// Linear without bias: `output[o] = dot(input, weight[o])`.
///
/// `weight` is `[out_features, in_features]` row-major.
/// Uses NEON/AVX2 vectorized dot products where available.
pub fn linear_no_bias(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    in_features: usize,
    out_features: usize,
) {
    assert_eq!(
        input.len(),
        in_features,
        "input length must equal in_features"
    );
    assert_eq!(
        weight.len(),
        out_features * in_features,
        "weight length must equal out_features * in_features"
    );
    assert_eq!(
        output.len(),
        out_features,
        "output length must equal out_features"
    );

    for o in 0..out_features {
        let row_start = o * in_features;
        let row = &weight[row_start..row_start + in_features];
        output[o] = dot_best(input, row);
    }
}

/// Batched linear: `output[b, o] = dot(input[b], weight[o]) + bias[o]` for each batch.
///
/// `input` is `[batch, in_features]` row-major.
/// `weight` is `[out_features, in_features]` row-major (shared across batch).
/// `bias` is `[out_features]` (shared across batch).
/// `output` is `[batch, out_features]` row-major.
pub fn linear_batched(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    batch: usize,
    in_features: usize,
    out_features: usize,
) {
    assert_eq!(
        input.len(),
        batch * in_features,
        "input length must equal batch * in_features"
    );
    assert_eq!(
        weight.len(),
        out_features * in_features,
        "weight length must equal out_features * in_features"
    );
    assert_eq!(
        bias.len(),
        out_features,
        "bias length must equal out_features"
    );
    assert_eq!(
        output.len(),
        batch * out_features,
        "output length must equal batch * out_features"
    );

    for b in 0..batch {
        let in_offset = b * in_features;
        let out_offset = b * out_features;
        linear(
            &input[in_offset..in_offset + in_features],
            weight,
            bias,
            &mut output[out_offset..out_offset + out_features],
            in_features,
            out_features,
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "simd_linear_tests.rs"]
mod simd_linear_tests;
