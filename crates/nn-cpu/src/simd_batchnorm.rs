// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized BatchNorm with explicit scalar/NEON/AVX2 entry points.
//!
//! BatchNorm normalizes using pre-computed running mean/variance (inference mode):
//!   For each channel c in [0, channels):
//!     For each spatial position s in [0, spatial):
//!       output[c * spatial + s] = gamma[c] * (input[c * spatial + s] - mean[c]) / sqrt(var[c] + eps) + beta[c]
//!
//! Input layout: `[channels, spatial]` row-major (single batch element).
//! `mean`, `var`, `gamma`, `beta` are per-channel vectors of length `channels`.
//!
//! NEON (aarch64) and AVX2 (x86_64) paths with scalar fallback.
//! Below `BATCHNORM_SIMD_THRESHOLD`, the scalar path is used unconditionally
//! since SIMD setup overhead dominates at small spatial sizes.

/// Below this spatial size, scalar fallback is used unconditionally.
pub const BATCHNORM_SIMD_THRESHOLD: usize = 16;

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

/// Pure scalar BatchNorm (inference mode).
///
/// Input layout: `[channels, spatial]` row-major, so `input.len() == channels * spatial`.
/// `mean`, `var`, `gamma`, `beta` are per-channel vectors of length `channels`.
///
/// # Arguments
/// * `input` -- flat input of length `channels * spatial`
/// * `mean` -- per-channel running mean, length `channels`
/// * `var` -- per-channel running variance, length `channels`
/// * `gamma` -- per-channel scale, length `channels`
/// * `beta` -- per-channel shift, length `channels`
/// * `channels` -- number of channels
/// * `spatial` -- spatial dimension per channel
/// * `eps` -- numerical stability constant (typically 1e-5)
/// * `out` -- flat output of length `channels * spatial`
pub fn batchnorm_f32_scalar(
    input: &[f32],
    mean: &[f32],
    var: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    spatial: usize,
    eps: f32,
    out: &mut [f32],
) {
    let total = channels * spatial;
    assert_eq!(
        input.len(),
        total,
        "input length must equal channels * spatial"
    );
    assert_eq!(
        out.len(),
        total,
        "output length must equal channels * spatial"
    );
    assert_eq!(mean.len(), channels, "mean length must equal channels");
    assert_eq!(var.len(), channels, "var length must equal channels");
    assert_eq!(gamma.len(), channels, "gamma length must equal channels");
    assert_eq!(beta.len(), channels, "beta length must equal channels");

    for c in 0..channels {
        let inv_std = 1.0 / (var[c] + eps).sqrt();
        let scale = gamma[c] * inv_std;
        let shift = beta[c] - gamma[c] * mean[c] * inv_std;
        let start = c * spatial;
        for s in 0..spatial {
            out[start + s] = input[start + s] * scale + shift;
        }
    }
}

/// Reference implementation returning a new Vec.
pub fn batchnorm_reference(
    input: &[f32],
    mean: &[f32],
    var: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    spatial: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; channels * spatial];
    batchnorm_f32_scalar(
        input, mean, var, gamma, beta, channels, spatial, eps, &mut out,
    );
    out
}

// ---------------------------------------------------------------------------
// NEON (aarch64) -- 128-bit, 4x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
pub fn batchnorm_f32_neon(
    input: &[f32],
    mean: &[f32],
    var: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    spatial: usize,
    eps: f32,
    out: &mut [f32],
) {
    use std::arch::aarch64::*;

    let total = channels * spatial;
    assert_eq!(input.len(), total);
    assert_eq!(out.len(), total);
    assert_eq!(mean.len(), channels);
    assert_eq!(var.len(), channels);
    assert_eq!(gamma.len(), channels);
    assert_eq!(beta.len(), channels);

    if spatial < BATCHNORM_SIMD_THRESHOLD || channels == 0 || spatial == 0 {
        batchnorm_f32_scalar(input, mean, var, gamma, beta, channels, spatial, eps, out);
        return;
    }

    let chunks = spatial / 4;
    let remainder = spatial % 4;
    let tail_start = chunks * 4;

    for c in 0..channels {
        let inv_std = 1.0 / (var[c] + eps).sqrt();
        let scale = gamma[c] * inv_std;
        let shift = beta[c] - gamma[c] * mean[c] * inv_std;
        let start = c * spatial;

        // SAFETY: aarch64 NEON is always available. Bounded loads/stores within slice.
        unsafe {
            let vscale = vdupq_n_f32(scale);
            let vshift = vdupq_n_f32(shift);
            let inp = input.as_ptr().add(start);
            let outp = out.as_mut_ptr().add(start);

            for i in 0..chunks {
                let offset = i * 4;
                let v = vld1q_f32(inp.add(offset));
                let r = vfmaq_f32(vshift, v, vscale); // v * scale + shift
                vst1q_f32(outp.add(offset), r);
            }
        }
        // Scalar tail.
        for s in 0..remainder {
            let idx = start + tail_start + s;
            out[idx] = input[idx] * scale + shift;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn batchnorm_f32_neon(
    input: &[f32],
    mean: &[f32],
    var: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    spatial: usize,
    eps: f32,
    out: &mut [f32],
) {
    batchnorm_f32_scalar(input, mean, var, gamma, beta, channels, spatial, eps, out);
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64) -- 256-bit, 8x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
pub fn batchnorm_f32_avx2(
    input: &[f32],
    mean: &[f32],
    var: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    spatial: usize,
    eps: f32,
    out: &mut [f32],
) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: AVX2+FMA detected above.
        unsafe {
            batchnorm_f32_avx2_inner(input, mean, var, gamma, beta, channels, spatial, eps, out);
        }
    } else {
        batchnorm_f32_scalar(input, mean, var, gamma, beta, channels, spatial, eps, out);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn batchnorm_f32_avx2_inner(
    input: &[f32],
    mean: &[f32],
    var: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    spatial: usize,
    eps: f32,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;

    let total = channels * spatial;
    assert_eq!(input.len(), total);
    assert_eq!(out.len(), total);
    assert_eq!(mean.len(), channels);
    assert_eq!(var.len(), channels);
    assert_eq!(gamma.len(), channels);
    assert_eq!(beta.len(), channels);

    if spatial < BATCHNORM_SIMD_THRESHOLD || channels == 0 || spatial == 0 {
        batchnorm_f32_scalar(input, mean, var, gamma, beta, channels, spatial, eps, out);
        return;
    }

    let chunks = spatial / 8;
    let remainder = spatial % 8;
    let tail_start = chunks * 8;

    for c in 0..channels {
        let inv_std = 1.0 / (var[c] + eps).sqrt();
        let scale = gamma[c] * inv_std;
        let shift = beta[c] - gamma[c] * mean[c] * inv_std;
        let start = c * spatial;

        let vscale = _mm256_set1_ps(scale);
        let vshift = _mm256_set1_ps(shift);
        let inp = input.as_ptr().add(start);
        let outp = out.as_mut_ptr().add(start);

        for i in 0..chunks {
            let offset = i * 8;
            let v = _mm256_loadu_ps(inp.add(offset));
            let r = _mm256_fmadd_ps(v, vscale, vshift); // v * scale + shift
            _mm256_storeu_ps(outp.add(offset), r);
        }
        // Scalar tail.
        for s in 0..remainder {
            let idx = start + tail_start + s;
            out[idx] = input[idx] * scale + shift;
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn batchnorm_f32_avx2(
    input: &[f32],
    mean: &[f32],
    var: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    spatial: usize,
    eps: f32,
    out: &mut [f32],
) {
    batchnorm_f32_scalar(input, mean, var, gamma, beta, channels, spatial, eps, out);
}

// ---------------------------------------------------------------------------
// Public dispatch: auto-selects best SIMD path
// ---------------------------------------------------------------------------

/// BatchNorm (inference mode) over `[channels, spatial]` layout.
/// Auto-dispatches to NEON (aarch64), AVX2 (x86_64), or scalar fallback.
///
/// Uses pre-computed running mean/variance. Each channel is normalized
/// independently with per-channel affine parameters (gamma, beta).
///
/// # Arguments
/// * `input` -- flat input of length `channels * spatial`
/// * `mean` -- per-channel running mean, length `channels`
/// * `var` -- per-channel running variance, length `channels`
/// * `gamma` -- per-channel scale, length `channels`
/// * `beta` -- per-channel shift, length `channels`
/// * `channels` -- number of channels
/// * `spatial` -- spatial dimension per channel
/// * `eps` -- numerical stability constant (typically 1e-5)
/// * `out` -- flat output of length `channels * spatial`
#[inline]
pub fn batchnorm_f32(
    input: &[f32],
    mean: &[f32],
    var: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    spatial: usize,
    eps: f32,
    out: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    {
        batchnorm_f32_neon(input, mean, var, gamma, beta, channels, spatial, eps, out);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            batchnorm_f32_avx2(input, mean, var, gamma, beta, channels, spatial, eps, out);
            return;
        }
    }

    #[allow(unreachable_code)]
    batchnorm_f32_scalar(input, mean, var, gamma, beta, channels, spatial, eps, out);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "simd_batchnorm_tests.rs"]
mod simd_batchnorm_tests;
