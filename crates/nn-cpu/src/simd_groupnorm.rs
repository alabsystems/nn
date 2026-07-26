// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized GroupNorm with explicit scalar/NEON/AVX2 entry points.
//!
//! GroupNorm divides channels into groups and normalizes within each group:
//!   For each group g in [0, groups):
//!     channels_per_group = channels / groups
//!     group_slice = input[g * cpg * spatial .. (g+1) * cpg * spatial]
//!     mean_g, var_g = mean/var over group_slice
//!     For each channel c in the group:
//!       output[c * spatial + s] = gamma[c] * (input[c * spatial + s] - mean_g) / sqrt(var_g + eps) + beta[c]
//!
//! Input layout: `[channels, spatial]` row-major (single batch element).
//! `gamma`, `beta` are per-channel vectors of length `channels`.
//! `channels` must be divisible by `groups`.
//!
//! NEON (aarch64) and AVX2 (x86_64) paths with scalar fallback.

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

/// Pure scalar GroupNorm. Uses f64 accumulation for numerical stability.
///
/// Input layout: `[channels, spatial]` row-major, so `input.len() == channels * spatial`.
/// `gamma`, `beta` are per-channel vectors of length `channels`.
///
/// # Arguments
/// * `input` -- flat input of length `channels * spatial`
/// * `gamma` -- per-channel scale, length `channels`
/// * `beta` -- per-channel shift, length `channels`
/// * `groups` -- number of groups (must divide `channels`)
/// * `channels` -- number of channels
/// * `spatial` -- spatial dimension per channel
/// * `eps` -- numerical stability constant (typically 1e-5)
/// * `out` -- flat output of length `channels * spatial`
pub fn groupnorm_f32_scalar(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    groups: usize,
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
    assert_eq!(gamma.len(), channels, "gamma length must equal channels");
    assert_eq!(beta.len(), channels, "beta length must equal channels");
    assert!(groups > 0, "groups must be > 0");
    assert_eq!(channels % groups, 0, "channels must be divisible by groups");

    if channels == 0 || spatial == 0 {
        return;
    }

    let cpg = channels / groups; // channels per group
    let group_size = cpg * spatial;

    for g in 0..groups {
        let group_start = g * cpg * spatial;

        // Pass 1: mean over the group (f64 accumulation)
        let mut sum = 0.0_f64;
        for i in 0..group_size {
            sum += f64::from(input[group_start + i]);
        }
        let mean = sum / group_size as f64;

        // Pass 2: variance
        let mut var_sum = 0.0_f64;
        for i in 0..group_size {
            let d = f64::from(input[group_start + i]) - mean;
            var_sum += d * d;
        }
        let variance = var_sum / group_size as f64;
        let inv_std = 1.0 / (variance + f64::from(eps)).sqrt();

        let mean_f32 = mean as f32;
        let inv_std_f32 = inv_std as f32;

        // Pass 3: normalize + per-channel affine
        for c_in_group in 0..cpg {
            let c = g * cpg + c_in_group;
            let ch_start = c * spatial;
            for s in 0..spatial {
                let normalized = (input[ch_start + s] - mean_f32) * inv_std_f32;
                out[ch_start + s] = normalized * gamma[c] + beta[c];
            }
        }
    }
}

/// Reference implementation returning a new Vec.
pub fn groupnorm_reference(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    groups: usize,
    channels: usize,
    spatial: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; channels * spatial];
    groupnorm_f32_scalar(input, gamma, beta, groups, channels, spatial, eps, &mut out);
    out
}

// ---------------------------------------------------------------------------
// NEON (aarch64) -- 128-bit, 4x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
pub fn groupnorm_f32_neon(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    groups: usize,
    channels: usize,
    spatial: usize,
    eps: f32,
    out: &mut [f32],
) {
    use std::arch::aarch64::*;

    let total = channels * spatial;
    assert_eq!(input.len(), total);
    assert_eq!(out.len(), total);
    assert_eq!(gamma.len(), channels);
    assert_eq!(beta.len(), channels);
    assert!(groups > 0);
    assert_eq!(channels % groups, 0);

    if channels == 0 || spatial == 0 {
        return;
    }

    let cpg = channels / groups;
    let group_size = cpg * spatial;
    let gs_f32 = group_size as f32;

    for g in 0..groups {
        let group_start = g * cpg * spatial;
        let group_data = &input[group_start..group_start + group_size];

        let chunks = group_size / 4;
        let remainder = group_size % 4;
        let tail_start = chunks * 4;

        // ---- Pass 1: mean ----
        // SAFETY: aarch64 NEON always available. Bounded loads.
        let mean = unsafe {
            let mut vsum = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let v = vld1q_f32(group_data.as_ptr().add(i * 4));
                vsum = vaddq_f32(vsum, v);
            }
            let pair = vpaddq_f32(vsum, vsum);
            let mut s = vgetq_lane_f32::<0>(vpaddq_f32(pair, pair));
            for i in 0..remainder {
                s += group_data[tail_start + i];
            }
            s / gs_f32
        };

        // ---- Pass 2: variance ----
        // SAFETY: Bounded loads.
        let variance = unsafe {
            let vmean = vdupq_n_f32(mean);
            let mut vvar = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let v = vld1q_f32(group_data.as_ptr().add(i * 4));
                let diff = vsubq_f32(v, vmean);
                vvar = vfmaq_f32(vvar, diff, diff);
            }
            let pair = vpaddq_f32(vvar, vvar);
            let mut s = vgetq_lane_f32::<0>(vpaddq_f32(pair, pair));
            for i in 0..remainder {
                let d = group_data[tail_start + i] - mean;
                s += d * d;
            }
            s / gs_f32
        };

        let inv_std = 1.0 / (variance + eps).sqrt();

        // ---- Pass 3: normalize + per-channel affine ----
        for c_in_group in 0..cpg {
            let c = g * cpg + c_in_group;
            let ch_start = c * spatial;
            let scale = gamma[c] * inv_std;
            let shift = beta[c] - gamma[c] * mean * inv_std;

            let sp_chunks = spatial / 4;
            let sp_remainder = spatial % 4;
            let sp_tail = sp_chunks * 4;

            // SAFETY: Bounded loads/stores within slice.
            unsafe {
                let vscale = vdupq_n_f32(scale);
                let vshift = vdupq_n_f32(shift);
                let inp = input.as_ptr().add(ch_start);
                let outp = out.as_mut_ptr().add(ch_start);

                for i in 0..sp_chunks {
                    let offset = i * 4;
                    let v = vld1q_f32(inp.add(offset));
                    let r = vfmaq_f32(vshift, v, vscale);
                    vst1q_f32(outp.add(offset), r);
                }
            }
            for s in 0..sp_remainder {
                let idx = ch_start + sp_tail + s;
                out[idx] = input[idx] * scale + shift;
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn groupnorm_f32_neon(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    groups: usize,
    channels: usize,
    spatial: usize,
    eps: f32,
    out: &mut [f32],
) {
    groupnorm_f32_scalar(input, gamma, beta, groups, channels, spatial, eps, out);
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64) -- 256-bit, 8x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
pub fn groupnorm_f32_avx2(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    groups: usize,
    channels: usize,
    spatial: usize,
    eps: f32,
    out: &mut [f32],
) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: AVX2+FMA detected above.
        unsafe {
            groupnorm_f32_avx2_inner(input, gamma, beta, groups, channels, spatial, eps, out);
        }
    } else {
        groupnorm_f32_scalar(input, gamma, beta, groups, channels, spatial, eps, out);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn groupnorm_f32_avx2_inner(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    groups: usize,
    channels: usize,
    spatial: usize,
    eps: f32,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;

    let total = channels * spatial;
    assert_eq!(input.len(), total);
    assert_eq!(out.len(), total);
    assert_eq!(gamma.len(), channels);
    assert_eq!(beta.len(), channels);
    assert!(groups > 0);
    assert_eq!(channels % groups, 0);

    if channels == 0 || spatial == 0 {
        return;
    }

    let cpg = channels / groups;
    let group_size = cpg * spatial;
    let gs_f32 = group_size as f32;

    for g in 0..groups {
        let group_start = g * cpg * spatial;
        let group_data = &input[group_start..group_start + group_size];

        let chunks = group_size / 8;
        let remainder = group_size % 8;
        let tail_start = chunks * 8;

        // ---- Pass 1: mean ----
        let mut vsum = _mm256_setzero_ps();
        for i in 0..chunks {
            let v = _mm256_loadu_ps(group_data.as_ptr().add(i * 8));
            vsum = _mm256_add_ps(vsum, v);
        }
        let mut s = hsum_avx2(vsum);
        for i in 0..remainder {
            s += group_data[tail_start + i];
        }
        let mean = s / gs_f32;

        // ---- Pass 2: variance ----
        let vmean = _mm256_set1_ps(mean);
        let mut vvar = _mm256_setzero_ps();
        for i in 0..chunks {
            let v = _mm256_loadu_ps(group_data.as_ptr().add(i * 8));
            let diff = _mm256_sub_ps(v, vmean);
            vvar = _mm256_fmadd_ps(diff, diff, vvar);
        }
        let mut v = hsum_avx2(vvar);
        for i in 0..remainder {
            let d = group_data[tail_start + i] - mean;
            v += d * d;
        }
        let variance = v / gs_f32;
        let inv_std = 1.0 / (variance + eps).sqrt();

        // ---- Pass 3: normalize + per-channel affine ----
        for c_in_group in 0..cpg {
            let c = g * cpg + c_in_group;
            let ch_start = c * spatial;
            let scale = gamma[c] * inv_std;
            let shift = beta[c] - gamma[c] * mean * inv_std;

            let sp_chunks = spatial / 8;
            let sp_remainder = spatial % 8;
            let sp_tail = sp_chunks * 8;

            let vscale = _mm256_set1_ps(scale);
            let vshift = _mm256_set1_ps(shift);
            let inp = input.as_ptr().add(ch_start);
            let outp = out.as_mut_ptr().add(ch_start);

            for i in 0..sp_chunks {
                let offset = i * 8;
                let vi = _mm256_loadu_ps(inp.add(offset));
                let r = _mm256_fmadd_ps(vi, vscale, vshift);
                _mm256_storeu_ps(outp.add(offset), r);
            }
            for s_idx in 0..sp_remainder {
                let idx = ch_start + sp_tail + s_idx;
                out[idx] = input[idx] * scale + shift;
            }
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
pub fn groupnorm_f32_avx2(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    groups: usize,
    channels: usize,
    spatial: usize,
    eps: f32,
    out: &mut [f32],
) {
    groupnorm_f32_scalar(input, gamma, beta, groups, channels, spatial, eps, out);
}

// ---------------------------------------------------------------------------
// Public dispatch: auto-selects best SIMD path
// ---------------------------------------------------------------------------

/// GroupNorm over `[channels, spatial]` layout. Auto-dispatches to
/// NEON (aarch64), AVX2 (x86_64), or scalar fallback.
///
/// Divides channels into `groups`, normalizes within each group
/// (across all channels and spatial positions in the group), then
/// applies per-channel affine parameters (gamma, beta).
///
/// # Arguments
/// * `input` -- flat input of length `channels * spatial`
/// * `gamma` -- per-channel scale, length `channels`
/// * `beta` -- per-channel shift, length `channels`
/// * `groups` -- number of groups (must divide `channels`)
/// * `channels` -- number of channels
/// * `spatial` -- spatial dimension per channel
/// * `eps` -- numerical stability constant (typically 1e-5)
/// * `out` -- flat output of length `channels * spatial`
#[inline]
pub fn groupnorm_f32(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    groups: usize,
    channels: usize,
    spatial: usize,
    eps: f32,
    out: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    {
        groupnorm_f32_neon(input, gamma, beta, groups, channels, spatial, eps, out);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            groupnorm_f32_avx2(input, gamma, beta, groups, channels, spatial, eps, out);
            return;
        }
    }

    #[allow(unreachable_code)]
    groupnorm_f32_scalar(input, gamma, beta, groups, channels, spatial, eps, out);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "simd_groupnorm_tests.rs"]
mod simd_groupnorm_tests;
