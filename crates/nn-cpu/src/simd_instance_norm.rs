// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized InstanceNorm with explicit scalar/NEON/AVX2 entry points.
//!
//! InstanceNorm normalizes each channel independently:
//!   For each channel c in [0, channels):
//!     slice = input[c * spatial .. (c+1) * spatial]
//!     mean_c = mean(slice)
//!     var_c = var(slice)
//!     output[c * spatial + i] = (slice[i] - mean_c) / sqrt(var_c + eps)
//!
//! Input layout: `[channels, spatial]` row-major (single batch).
//! No affine parameters (gamma/beta) in this version; for affine instance
//! norm, compose with element-wise scale+shift.
//!
//! NEON (aarch64) and AVX2 (x86_64) paths with scalar fallback.

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

/// Pure scalar InstanceNorm. Uses f64 accumulation for numerical stability.
///
/// Input layout: `[channels, spatial]` row-major, so `input.len() == channels * spatial`.
/// Output has the same layout and length.
///
/// # Arguments
/// * `input` — flat input of length `channels * spatial`
/// * `output` — flat output of length `channels * spatial`
/// * `channels` — number of channels
/// * `spatial` — spatial dimension per channel
/// * `eps` — numerical stability constant (typically 1e-5)
pub fn instance_norm_f32_scalar(
    input: &[f32],
    output: &mut [f32],
    channels: usize,
    spatial: usize,
    eps: f32,
) {
    let total = channels * spatial;
    assert_eq!(
        input.len(),
        total,
        "input length must equal channels * spatial"
    );
    assert_eq!(
        output.len(),
        total,
        "output length must equal channels * spatial"
    );
    if spatial == 0 || channels == 0 {
        return;
    }

    for c in 0..channels {
        let start = c * spatial;
        let end = start + spatial;
        let ch_in = &input[start..end];
        let ch_out = &mut output[start..end];

        // Pass 1: mean (f64)
        let mut sum = 0.0_f64;
        for &x in ch_in {
            sum += f64::from(x);
        }
        let mean = sum / spatial as f64;

        // Pass 2: variance
        let mut var_sum = 0.0_f64;
        for &x in ch_in {
            let d = f64::from(x) - mean;
            var_sum += d * d;
        }
        let variance = var_sum / spatial as f64;
        let inv_std = 1.0 / (variance + f64::from(eps)).sqrt();

        let mean_f32 = mean as f32;
        let inv_std_f32 = inv_std as f32;

        // Pass 3: normalize
        for i in 0..spatial {
            ch_out[i] = (ch_in[i] - mean_f32) * inv_std_f32;
        }
    }
}

/// Reference implementation returning a new Vec.
pub fn instance_norm_f32_reference(
    input: &[f32],
    channels: usize,
    spatial: usize,
    eps: f32,
) -> Vec<f32> {
    let mut output = vec![0.0f32; channels * spatial];
    instance_norm_f32_scalar(input, &mut output, channels, spatial, eps);
    output
}

// ---------------------------------------------------------------------------
// NEON (aarch64) — 128-bit, 4x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
pub fn instance_norm_f32_neon(
    input: &[f32],
    output: &mut [f32],
    channels: usize,
    spatial: usize,
    eps: f32,
) {
    use std::arch::aarch64::*;

    let total = channels * spatial;
    assert_eq!(input.len(), total);
    assert_eq!(output.len(), total);
    if spatial == 0 || channels == 0 {
        return;
    }

    let chunks = spatial / 4;
    let remainder = spatial % 4;
    let tail_start = chunks * 4;
    let sp_f32 = spatial as f32;

    for c in 0..channels {
        let start = c * spatial;
        let ch_in = &input[start..start + spatial];
        let ch_out = &mut output[start..start + spatial];

        // ---- Pass 1: mean ----
        // SAFETY: aarch64 NEON always available. Bounded loads.
        let mean = unsafe {
            let mut vsum = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let v = vld1q_f32(ch_in.as_ptr().add(i * 4));
                vsum = vaddq_f32(vsum, v);
            }
            let pair = vpaddq_f32(vsum, vsum);
            let mut s = vgetq_lane_f32::<0>(vpaddq_f32(pair, pair));
            for i in 0..remainder {
                s += ch_in[tail_start + i];
            }
            s / sp_f32
        };

        // ---- Pass 2: variance ----
        // SAFETY: Bounded loads.
        let variance = unsafe {
            let vmean = vdupq_n_f32(mean);
            let mut vvar = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let v = vld1q_f32(ch_in.as_ptr().add(i * 4));
                let diff = vsubq_f32(v, vmean);
                vvar = vfmaq_f32(vvar, diff, diff);
            }
            let pair = vpaddq_f32(vvar, vvar);
            let mut s = vgetq_lane_f32::<0>(vpaddq_f32(pair, pair));
            for i in 0..remainder {
                let d = ch_in[tail_start + i] - mean;
                s += d * d;
            }
            s / sp_f32
        };

        let inv_std = 1.0 / (variance + eps).sqrt();

        // ---- Pass 3: normalize ----
        // SAFETY: Bounded loads/stores.
        unsafe {
            let vmean = vdupq_n_f32(mean);
            let vinv_std = vdupq_n_f32(inv_std);
            for i in 0..chunks {
                let offset = i * 4;
                let v = vld1q_f32(ch_in.as_ptr().add(offset));
                let diff = vsubq_f32(v, vmean);
                let normed = vmulq_f32(diff, vinv_std);
                vst1q_f32(ch_out.as_mut_ptr().add(offset), normed);
            }
            for i in 0..remainder {
                let idx = tail_start + i;
                ch_out[idx] = (ch_in[idx] - mean) * inv_std;
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn instance_norm_f32_neon(
    input: &[f32],
    output: &mut [f32],
    channels: usize,
    spatial: usize,
    eps: f32,
) {
    instance_norm_f32_scalar(input, output, channels, spatial, eps);
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64) — 256-bit, 8x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
pub fn instance_norm_f32_avx2(
    input: &[f32],
    output: &mut [f32],
    channels: usize,
    spatial: usize,
    eps: f32,
) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: AVX2+FMA detected above.
        unsafe {
            instance_norm_f32_avx2_inner(input, output, channels, spatial, eps);
        }
    } else {
        instance_norm_f32_scalar(input, output, channels, spatial, eps);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn instance_norm_f32_avx2_inner(
    input: &[f32],
    output: &mut [f32],
    channels: usize,
    spatial: usize,
    eps: f32,
) {
    use std::arch::x86_64::*;

    let total = channels * spatial;
    assert_eq!(input.len(), total);
    assert_eq!(output.len(), total);
    if spatial == 0 || channels == 0 {
        return;
    }

    let chunks = spatial / 8;
    let remainder = spatial % 8;
    let tail_start = chunks * 8;
    let sp_f32 = spatial as f32;

    for c in 0..channels {
        let start = c * spatial;
        let ch_in = &input[start..start + spatial];
        let ch_out = &mut output[start..start + spatial];

        // ---- Pass 1: mean ----
        let mut vsum = _mm256_setzero_ps();
        for i in 0..chunks {
            let v = _mm256_loadu_ps(ch_in.as_ptr().add(i * 8));
            vsum = _mm256_add_ps(vsum, v);
        }
        let mut s = hsum_avx2(vsum);
        for i in 0..remainder {
            s += ch_in[tail_start + i];
        }
        let mean = s / sp_f32;

        // ---- Pass 2: variance ----
        let vmean = _mm256_set1_ps(mean);
        let mut vvar = _mm256_setzero_ps();
        for i in 0..chunks {
            let v = _mm256_loadu_ps(ch_in.as_ptr().add(i * 8));
            let diff = _mm256_sub_ps(v, vmean);
            vvar = _mm256_fmadd_ps(diff, diff, vvar);
        }
        let mut v = hsum_avx2(vvar);
        for i in 0..remainder {
            let d = ch_in[tail_start + i] - mean;
            v += d * d;
        }
        let variance = v / sp_f32;
        let inv_std = 1.0 / (variance + eps).sqrt();

        // ---- Pass 3: normalize ----
        let vinv_std = _mm256_set1_ps(inv_std);
        for i in 0..chunks {
            let offset = i * 8;
            let vi = _mm256_loadu_ps(ch_in.as_ptr().add(offset));
            let diff = _mm256_sub_ps(vi, vmean);
            let normed = _mm256_mul_ps(diff, vinv_std);
            _mm256_storeu_ps(ch_out.as_mut_ptr().add(offset), normed);
        }
        for i in 0..remainder {
            let idx = tail_start + i;
            ch_out[idx] = (ch_in[idx] - mean) * inv_std;
        }
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
pub fn instance_norm_f32_avx2(
    input: &[f32],
    output: &mut [f32],
    channels: usize,
    spatial: usize,
    eps: f32,
) {
    instance_norm_f32_scalar(input, output, channels, spatial, eps);
}

// ---------------------------------------------------------------------------
// Public dispatch: auto-selects best SIMD path
// ---------------------------------------------------------------------------

/// InstanceNorm over `[channels, spatial]` layout. Auto-dispatches to
/// NEON (aarch64), AVX2 (x86_64), or scalar fallback.
///
/// Normalizes each channel independently: for each channel, computes
/// mean and variance over the spatial dimension, then normalizes.
///
/// # Arguments
/// * `input` — flat input of length `channels * spatial`
/// * `output` — flat output of length `channels * spatial`
/// * `channels` — number of channels
/// * `spatial` — spatial dimension per channel
/// * `eps` — numerical stability constant (typically 1e-5)
pub fn instance_norm_f32(
    input: &[f32],
    output: &mut [f32],
    channels: usize,
    spatial: usize,
    eps: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        instance_norm_f32_neon(input, output, channels, spatial, eps);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            instance_norm_f32_avx2(input, output, channels, spatial, eps);
            return;
        }
    }

    #[allow(unreachable_code)]
    instance_norm_f32_scalar(input, output, channels, spatial, eps);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "simd_instance_norm_tests.rs"]
mod simd_instance_norm_tests;
