// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized LayerNorm and RMSNorm along the last dimension.
//!
//! **LayerNorm** — two-pass Welford algorithm:
//!   1. Compute mean and variance via Welford's online algorithm (numerically stable).
//!   2. Normalize: `(x - mean) / sqrt(var + eps) * gamma + beta`.
//!
//! **RMSNorm** — single-pass:
//!   1. Compute mean of squares: `sum(x^2) / n`.
//!   2. Normalize: `x / sqrt(mean_sq + eps) * gamma`.
//!
//! NEON (aarch64) and AVX2 (x86_64) paths with scalar fallback.

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

/// Scalar LayerNorm using Welford's algorithm for numerically stable
/// mean/variance computation.
///
/// Processes `input` in contiguous rows of `normalized_shape` elements.
/// Each row is independently normalized with the provided `gamma` (scale)
/// and optional `beta` (shift) parameters.
pub fn layernorm_scalar(
    input: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    eps: f32,
    normalized_shape: usize,
) -> Vec<f32> {
    assert!(normalized_shape > 0, "normalized_shape must be > 0");
    assert_eq!(
        input.len() % normalized_shape,
        0,
        "input length must be a multiple of normalized_shape"
    );
    assert_eq!(gamma.len(), normalized_shape);
    if let Some(b) = beta {
        assert_eq!(b.len(), normalized_shape);
    }

    let rows = input.len() / normalized_shape;
    let mut output = vec![0.0f32; input.len()];

    for row in 0..rows {
        let start = row * normalized_shape;
        let end = start + normalized_shape;
        let row_in = &input[start..end];
        let row_out = &mut output[start..end];

        // Pass 1: Welford mean and variance
        let mut mean = 0.0_f64;
        let mut m2 = 0.0_f64;
        for (i, &x) in row_in.iter().enumerate() {
            let n = (i + 1) as f64;
            let delta = f64::from(x) - mean;
            mean += delta / n;
            let delta2 = f64::from(x) - mean;
            m2 += delta * delta2;
        }
        let variance = m2 / normalized_shape as f64;
        let inv_std = 1.0 / (variance + f64::from(eps)).sqrt();
        let mean_f32 = mean as f32;
        let inv_std_f32 = inv_std as f32;

        // Pass 2: normalize with affine transform
        for i in 0..normalized_shape {
            let normalized = (row_in[i] - mean_f32) * inv_std_f32;
            row_out[i] = normalized * gamma[i] + beta.map_or(0.0, |b| b[i]);
        }
    }

    output
}

/// Scalar RMSNorm: `x / sqrt(mean(x^2) + eps) * gamma`.
///
/// Single-pass computation of root-mean-square, then element-wise scaling.
pub fn rmsnorm_scalar(input: &[f32], gamma: &[f32], eps: f32, normalized_shape: usize) -> Vec<f32> {
    assert!(normalized_shape > 0, "normalized_shape must be > 0");
    assert_eq!(
        input.len() % normalized_shape,
        0,
        "input length must be a multiple of normalized_shape"
    );
    assert_eq!(gamma.len(), normalized_shape);

    let rows = input.len() / normalized_shape;
    let mut output = vec![0.0f32; input.len()];

    for row in 0..rows {
        let start = row * normalized_shape;
        let end = start + normalized_shape;
        let row_in = &input[start..end];
        let row_out = &mut output[start..end];

        // Single pass: sum of squares
        let mut sum_sq = 0.0_f64;
        for &x in row_in {
            sum_sq += f64::from(x) * f64::from(x);
        }
        let mean_sq = sum_sq / normalized_shape as f64;
        let inv_rms = 1.0 / (mean_sq + f64::from(eps)).sqrt();
        let inv_rms_f32 = inv_rms as f32;

        // Normalize with scale
        for i in 0..normalized_shape {
            row_out[i] = row_in[i] * inv_rms_f32 * gamma[i];
        }
    }

    output
}

// ---------------------------------------------------------------------------
// NEON (aarch64) — 128-bit, 4x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// NEON-accelerated LayerNorm over contiguous rows of `normalized_shape`
    /// elements. Uses Welford's algorithm with 4-wide SIMD accumulation.
    pub(super) fn layernorm_neon(
        input: &[f32],
        gamma: &[f32],
        beta: Option<&[f32]>,
        eps: f32,
        normalized_shape: usize,
    ) -> Vec<f32> {
        assert!(normalized_shape > 0);
        assert_eq!(input.len() % normalized_shape, 0);
        assert_eq!(gamma.len(), normalized_shape);
        if let Some(b) = beta {
            assert_eq!(b.len(), normalized_shape);
        }

        let rows = input.len() / normalized_shape;
        let mut output = vec![0.0f32; input.len()];

        for row in 0..rows {
            let start = row * normalized_shape;
            let end = start + normalized_shape;
            layernorm_row_neon(
                &input[start..end],
                &mut output[start..end],
                gamma,
                beta,
                eps,
            );
        }

        output
    }

    fn layernorm_row_neon(
        row_in: &[f32],
        row_out: &mut [f32],
        gamma: &[f32],
        beta: Option<&[f32]>,
        eps: f32,
    ) {
        let n = row_in.len();
        let chunks = n / 4;
        let remainder = n % 4;
        let tail_start = chunks * 4;
        let n_f32 = n as f32;

        // ---- Pass 1: compute mean and variance ----
        // Two-pass for SIMD: first compute mean (sum/n), then variance (sum((x-mean)^2)/n).
        // This is more SIMD-friendly than Welford while remaining numerically stable.

        // SAFETY: aarch64 NEON is always available. Bounded loads within slice.
        let mean = unsafe {
            let mut vsum = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let v = vld1q_f32(row_in.as_ptr().add(i * 4));
                vsum = vaddq_f32(vsum, v);
            }
            // Horizontal sum of 4 lanes.
            let pair = vpaddq_f32(vsum, vsum);
            let mut s = vgetq_lane_f32::<0>(vpaddq_f32(pair, pair));
            // Scalar tail for sum.
            for i in 0..remainder {
                s += row_in[tail_start + i];
            }
            s / n_f32
        };

        // Compute variance: sum((x - mean)^2) / n
        // SAFETY: Bounded loads within slice.
        let variance = unsafe {
            let vmean = vdupq_n_f32(mean);
            let mut vvar = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let v = vld1q_f32(row_in.as_ptr().add(i * 4));
                let diff = vsubq_f32(v, vmean);
                vvar = vfmaq_f32(vvar, diff, diff); // vvar += diff * diff
            }
            // Horizontal sum.
            let pair = vpaddq_f32(vvar, vvar);
            let mut s = vgetq_lane_f32::<0>(vpaddq_f32(pair, pair));
            // Scalar tail.
            for i in 0..remainder {
                let d = row_in[tail_start + i] - mean;
                s += d * d;
            }
            s / n_f32
        };

        let inv_std = 1.0 / (variance + eps).sqrt();

        // ---- Pass 2: normalize with affine transform ----
        // SAFETY: Bounded loads/stores within slice.
        unsafe {
            let vmean = vdupq_n_f32(mean);
            let vinv_std = vdupq_n_f32(inv_std);
            for i in 0..chunks {
                let offset = i * 4;
                let v = vld1q_f32(row_in.as_ptr().add(offset));
                let g = vld1q_f32(gamma.as_ptr().add(offset));
                let diff = vsubq_f32(v, vmean);
                let normed = vmulq_f32(diff, vinv_std);
                let scaled = vmulq_f32(normed, g);
                let result = if let Some(b) = beta {
                    let bv = vld1q_f32(b.as_ptr().add(offset));
                    vaddq_f32(scaled, bv)
                } else {
                    scaled
                };
                vst1q_f32(row_out.as_mut_ptr().add(offset), result);
            }
            // Scalar tail.
            for i in 0..remainder {
                let idx = tail_start + i;
                let normalized = (row_in[idx] - mean) * inv_std;
                row_out[idx] = normalized * gamma[idx] + beta.map_or(0.0, |b| b[idx]);
            }
        }
    }

    /// NEON-accelerated RMSNorm over contiguous rows.
    pub(super) fn rmsnorm_neon(
        input: &[f32],
        gamma: &[f32],
        eps: f32,
        normalized_shape: usize,
    ) -> Vec<f32> {
        assert!(normalized_shape > 0);
        assert_eq!(input.len() % normalized_shape, 0);
        assert_eq!(gamma.len(), normalized_shape);

        let rows = input.len() / normalized_shape;
        let mut output = vec![0.0f32; input.len()];

        for row in 0..rows {
            let start = row * normalized_shape;
            let end = start + normalized_shape;
            rmsnorm_row_neon(&input[start..end], &mut output[start..end], gamma, eps);
        }

        output
    }

    fn rmsnorm_row_neon(row_in: &[f32], row_out: &mut [f32], gamma: &[f32], eps: f32) {
        let n = row_in.len();
        let chunks = n / 4;
        let remainder = n % 4;
        let tail_start = chunks * 4;
        let n_f32 = n as f32;

        // ---- Single pass: sum of squares ----
        // SAFETY: aarch64 NEON is always available. Bounded loads within slice.
        let mean_sq = unsafe {
            let mut vsum_sq = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let v = vld1q_f32(row_in.as_ptr().add(i * 4));
                vsum_sq = vfmaq_f32(vsum_sq, v, v); // vsum_sq += v * v
            }
            // Horizontal sum.
            let pair = vpaddq_f32(vsum_sq, vsum_sq);
            let mut s = vgetq_lane_f32::<0>(vpaddq_f32(pair, pair));
            // Scalar tail.
            for i in 0..remainder {
                let x = row_in[tail_start + i];
                s += x * x;
            }
            s / n_f32
        };

        let inv_rms = 1.0 / (mean_sq + eps).sqrt();

        // ---- Normalize with scale ----
        // SAFETY: Bounded loads/stores within slice.
        unsafe {
            let vinv_rms = vdupq_n_f32(inv_rms);
            for i in 0..chunks {
                let offset = i * 4;
                let v = vld1q_f32(row_in.as_ptr().add(offset));
                let g = vld1q_f32(gamma.as_ptr().add(offset));
                let normed = vmulq_f32(v, vinv_rms);
                let scaled = vmulq_f32(normed, g);
                vst1q_f32(row_out.as_mut_ptr().add(offset), scaled);
            }
            // Scalar tail.
            for i in 0..remainder {
                let idx = tail_start + i;
                row_out[idx] = row_in[idx] * inv_rms * gamma[idx];
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

    /// Horizontal sum of 8 f32 lanes in a __m256.
    ///
    /// # Safety
    /// Caller must verify AVX2 is available.
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_ps_avx2(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(result)
    }

    /// AVX2-accelerated LayerNorm.
    ///
    /// # Safety
    /// Caller must verify AVX2 is available (`is_x86_feature_detected!("avx2")`).
    #[target_feature(enable = "avx2")]
    pub unsafe fn layernorm_avx2(
        input: &[f32],
        gamma: &[f32],
        beta: Option<&[f32]>,
        eps: f32,
        normalized_shape: usize,
    ) -> Vec<f32> {
        assert!(normalized_shape > 0);
        assert_eq!(input.len() % normalized_shape, 0);
        assert_eq!(gamma.len(), normalized_shape);
        if let Some(b) = beta {
            assert_eq!(b.len(), normalized_shape);
        }

        let rows = input.len() / normalized_shape;
        let mut output = vec![0.0f32; input.len()];

        for row in 0..rows {
            let start = row * normalized_shape;
            let end = start + normalized_shape;
            layernorm_row_avx2(
                &input[start..end],
                &mut output[start..end],
                gamma,
                beta,
                eps,
            );
        }

        output
    }

    #[target_feature(enable = "avx2")]
    unsafe fn layernorm_row_avx2(
        row_in: &[f32],
        row_out: &mut [f32],
        gamma: &[f32],
        beta: Option<&[f32]>,
        eps: f32,
    ) {
        let n = row_in.len();
        let chunks = n / 8;
        let remainder = n % 8;
        let tail_start = chunks * 8;
        let n_f32 = n as f32;

        // ---- Pass 1a: compute mean ----
        let mut vsum = _mm256_setzero_ps();
        for i in 0..chunks {
            let v = _mm256_loadu_ps(row_in.as_ptr().add(i * 8));
            vsum = _mm256_add_ps(vsum, v);
        }
        let mut s = hsum_ps_avx2(vsum);
        for i in 0..remainder {
            s += row_in[tail_start + i];
        }
        let mean = s / n_f32;

        // ---- Pass 1b: compute variance ----
        let vmean = _mm256_set1_ps(mean);
        let mut vvar = _mm256_setzero_ps();
        for i in 0..chunks {
            let v = _mm256_loadu_ps(row_in.as_ptr().add(i * 8));
            let diff = _mm256_sub_ps(v, vmean);
            // FMA: vvar += diff * diff
            vvar = _mm256_fmadd_ps(diff, diff, vvar);
        }
        let mut v = hsum_ps_avx2(vvar);
        for i in 0..remainder {
            let d = row_in[tail_start + i] - mean;
            v += d * d;
        }
        let variance = v / n_f32;
        let inv_std = 1.0 / (variance + eps).sqrt();

        // ---- Pass 2: normalize with affine transform ----
        let vinv_std = _mm256_set1_ps(inv_std);
        for i in 0..chunks {
            let offset = i * 8;
            let vi = _mm256_loadu_ps(row_in.as_ptr().add(offset));
            let g = _mm256_loadu_ps(gamma.as_ptr().add(offset));
            let diff = _mm256_sub_ps(vi, vmean);
            let normed = _mm256_mul_ps(diff, vinv_std);
            let scaled = _mm256_mul_ps(normed, g);
            let result = if let Some(b) = beta {
                let bv = _mm256_loadu_ps(b.as_ptr().add(offset));
                _mm256_add_ps(scaled, bv)
            } else {
                scaled
            };
            _mm256_storeu_ps(row_out.as_mut_ptr().add(offset), result);
        }
        // Scalar tail.
        for i in 0..remainder {
            let idx = tail_start + i;
            let normalized = (row_in[idx] - mean) * inv_std;
            row_out[idx] = normalized * gamma[idx] + beta.map_or(0.0, |b| b[idx]);
        }
    }

    /// AVX2-accelerated RMSNorm.
    ///
    /// # Safety
    /// Caller must verify AVX2 is available (`is_x86_feature_detected!("avx2")`).
    #[target_feature(enable = "avx2")]
    pub unsafe fn rmsnorm_avx2(
        input: &[f32],
        gamma: &[f32],
        eps: f32,
        normalized_shape: usize,
    ) -> Vec<f32> {
        assert!(normalized_shape > 0);
        assert_eq!(input.len() % normalized_shape, 0);
        assert_eq!(gamma.len(), normalized_shape);

        let rows = input.len() / normalized_shape;
        let mut output = vec![0.0f32; input.len()];

        for row in 0..rows {
            let start = row * normalized_shape;
            let end = start + normalized_shape;
            rmsnorm_row_avx2(&input[start..end], &mut output[start..end], gamma, eps);
        }

        output
    }

    #[target_feature(enable = "avx2")]
    unsafe fn rmsnorm_row_avx2(row_in: &[f32], row_out: &mut [f32], gamma: &[f32], eps: f32) {
        let n = row_in.len();
        let chunks = n / 8;
        let remainder = n % 8;
        let tail_start = chunks * 8;
        let n_f32 = n as f32;

        // ---- Single pass: sum of squares ----
        let mut vsum_sq = _mm256_setzero_ps();
        for i in 0..chunks {
            let v = _mm256_loadu_ps(row_in.as_ptr().add(i * 8));
            // FMA: vsum_sq += v * v
            vsum_sq = _mm256_fmadd_ps(v, v, vsum_sq);
        }
        let mut s = hsum_ps_avx2(vsum_sq);
        for i in 0..remainder {
            let x = row_in[tail_start + i];
            s += x * x;
        }
        let mean_sq = s / n_f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();

        // ---- Normalize with scale ----
        let vinv_rms = _mm256_set1_ps(inv_rms);
        for i in 0..chunks {
            let offset = i * 8;
            let v = _mm256_loadu_ps(row_in.as_ptr().add(offset));
            let g = _mm256_loadu_ps(gamma.as_ptr().add(offset));
            let normed = _mm256_mul_ps(v, vinv_rms);
            let scaled = _mm256_mul_ps(normed, g);
            _mm256_storeu_ps(row_out.as_mut_ptr().add(offset), scaled);
        }
        // Scalar tail.
        for i in 0..remainder {
            let idx = tail_start + i;
            row_out[idx] = row_in[idx] * inv_rms * gamma[idx];
        }
    }
}

// ---------------------------------------------------------------------------
// Public dispatch: auto-selects best SIMD path
// ---------------------------------------------------------------------------

/// LayerNorm along the last dimension. Auto-dispatches to NEON/AVX2/scalar.
///
/// Processes `input` in contiguous rows of `normalized_shape` elements.
/// Each row is independently normalized using Welford's algorithm for
/// numerically stable mean/variance computation, then scaled by `gamma`
/// and shifted by `beta`.
///
/// # Arguments
/// * `input` — flattened input tensor; length must be a multiple of `normalized_shape`
/// * `gamma` — per-feature scale; length must equal `normalized_shape`
/// * `beta` — optional per-feature shift; length must equal `normalized_shape` if provided
/// * `eps` — small constant for numerical stability (typically 1e-5)
/// * `normalized_shape` — size of the last dimension to normalize over
///
/// # Returns
/// Normalized output with the same shape as `input`.
pub fn layernorm(
    input: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    eps: f32,
    normalized_shape: usize,
) -> Vec<f32> {
    #[cfg(target_arch = "aarch64")]
    {
        return neon::layernorm_neon(input, gamma, beta, eps, normalized_shape);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            return unsafe { avx2::layernorm_avx2(input, gamma, beta, eps, normalized_shape) };
        }
    }

    #[allow(unreachable_code)]
    layernorm_scalar(input, gamma, beta, eps, normalized_shape)
}

/// RMSNorm along the last dimension. Auto-dispatches to NEON/AVX2/scalar.
///
/// Computes `x / sqrt(mean(x^2) + eps) * gamma` for each contiguous row
/// of `normalized_shape` elements.
///
/// # Arguments
/// * `input` — flattened input tensor; length must be a multiple of `normalized_shape`
/// * `gamma` — per-feature scale; length must equal `normalized_shape`
/// * `eps` — small constant for numerical stability (typically 1e-5)
/// * `normalized_shape` — size of the last dimension to normalize over
///
/// # Returns
/// Normalized output with the same shape as `input`.
pub fn rmsnorm(input: &[f32], gamma: &[f32], eps: f32, normalized_shape: usize) -> Vec<f32> {
    #[cfg(target_arch = "aarch64")]
    {
        return neon::rmsnorm_neon(input, gamma, eps, normalized_shape);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            return unsafe { avx2::rmsnorm_avx2(input, gamma, eps, normalized_shape) };
        }
    }

    #[allow(unreachable_code)]
    rmsnorm_scalar(input, gamma, eps, normalized_shape)
}

#[cfg(test)]
#[path = "layernorm_tests.rs"]
mod layernorm_tests;
