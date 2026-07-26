// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized 1D convolution with full feature support.
//!
//! Extends the basic `conv1d` module with support for **groups** and
//! **dilation**, matching PyTorch's `torch.nn.Conv1d` parameter set.
//!
//! Layout:
//! - Input:  `[in_channels, in_length]` (single-batch, row-major)
//! - Weight: `[out_channels, in_channels/groups, kernel_size]` (row-major)
//! - Bias:   `[out_channels]` (optional)
//! - Output: `[out_channels, out_length]` (row-major)
//!
//! `out_length = (in_length + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1`
//!
//! NEON (aarch64) and AVX2 (x86_64) paths with scalar fallback.

use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during conv1d.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Conv1dError {
    /// Stride must be > 0.
    ZeroStride,
    /// Kernel size must be > 0.
    ZeroKernelSize,
    /// Input channels must be > 0.
    ZeroInChannels,
    /// Output channels must be > 0.
    ZeroOutChannels,
    /// Groups must be > 0.
    ZeroGroups,
    /// Dilation must be > 0.
    ZeroDilation,
    /// in_channels must be divisible by groups.
    InChannelsNotDivisibleByGroups { in_ch: usize, groups: usize },
    /// out_channels must be divisible by groups.
    OutChannelsNotDivisibleByGroups { out_ch: usize, groups: usize },
    /// Input length is not a multiple of in_channels.
    InvalidInputShape { input_len: usize, in_ch: usize },
    /// Weight length does not match `[out_ch, in_ch/groups, kernel_size]`.
    InvalidWeightShape { weight_len: usize, expected: usize },
    /// Bias length does not match out_channels.
    InvalidBiasLength { bias_len: usize, out_ch: usize },
    /// Padded input is smaller than the effective kernel size.
    PaddedInputTooSmall {
        padded_len: usize,
        effective_kernel: usize,
    },
}

impl fmt::Display for Conv1dError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroStride => write!(f, "stride must be > 0"),
            Self::ZeroKernelSize => write!(f, "kernel_size must be > 0"),
            Self::ZeroInChannels => write!(f, "in_channels must be > 0"),
            Self::ZeroOutChannels => write!(f, "out_channels must be > 0"),
            Self::ZeroGroups => write!(f, "groups must be > 0"),
            Self::ZeroDilation => write!(f, "dilation must be > 0"),
            Self::InChannelsNotDivisibleByGroups { in_ch, groups } => write!(
                f,
                "in_channels ({in_ch}) must be divisible by groups ({groups})"
            ),
            Self::OutChannelsNotDivisibleByGroups { out_ch, groups } => write!(
                f,
                "out_channels ({out_ch}) must be divisible by groups ({groups})"
            ),
            Self::InvalidInputShape { input_len, in_ch } => write!(
                f,
                "input length ({input_len}) must be a multiple of in_channels ({in_ch})"
            ),
            Self::InvalidWeightShape {
                weight_len,
                expected,
            } => write!(
                f,
                "weight length ({weight_len}) does not match expected ({expected})"
            ),
            Self::InvalidBiasLength { bias_len, out_ch } => write!(
                f,
                "bias length ({bias_len}) does not match out_channels ({out_ch})"
            ),
            Self::PaddedInputTooSmall {
                padded_len,
                effective_kernel,
            } => write!(
                f,
                "padded input length ({padded_len}) < effective kernel size ({effective_kernel})"
            ),
        }
    }
}

impl std::error::Error for Conv1dError {}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a 1D convolution operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv1dConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Conv1dConfig {
    /// Effective kernel size accounting for dilation.
    #[inline]
    pub fn effective_kernel_size(&self) -> usize {
        self.dilation * (self.kernel_size - 1) + 1
    }

    /// Compute the output length given an input length.
    #[inline]
    pub fn output_length(&self, in_len: usize) -> usize {
        let padded = in_len + 2 * self.padding;
        let ek = self.effective_kernel_size();
        (padded - ek) / self.stride + 1
    }

    /// Number of input channels per group.
    #[inline]
    pub fn in_channels_per_group(&self) -> usize {
        self.in_channels / self.groups
    }

    /// Number of output channels per group.
    #[inline]
    pub fn out_channels_per_group(&self) -> usize {
        self.out_channels / self.groups
    }
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

fn validate(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    cfg: &Conv1dConfig,
) -> Result<usize, Conv1dError> {
    if cfg.stride == 0 {
        return Err(Conv1dError::ZeroStride);
    }
    if cfg.kernel_size == 0 {
        return Err(Conv1dError::ZeroKernelSize);
    }
    if cfg.in_channels == 0 {
        return Err(Conv1dError::ZeroInChannels);
    }
    if cfg.out_channels == 0 {
        return Err(Conv1dError::ZeroOutChannels);
    }
    if cfg.groups == 0 {
        return Err(Conv1dError::ZeroGroups);
    }
    if cfg.dilation == 0 {
        return Err(Conv1dError::ZeroDilation);
    }
    if !cfg.in_channels.is_multiple_of(cfg.groups) {
        return Err(Conv1dError::InChannelsNotDivisibleByGroups {
            in_ch: cfg.in_channels,
            groups: cfg.groups,
        });
    }
    if !cfg.out_channels.is_multiple_of(cfg.groups) {
        return Err(Conv1dError::OutChannelsNotDivisibleByGroups {
            out_ch: cfg.out_channels,
            groups: cfg.groups,
        });
    }
    if !input.len().is_multiple_of(cfg.in_channels) {
        return Err(Conv1dError::InvalidInputShape {
            input_len: input.len(),
            in_ch: cfg.in_channels,
        });
    }
    let in_len = input.len() / cfg.in_channels;
    let ic_per_g = cfg.in_channels_per_group();
    let expected_weight = cfg.out_channels * ic_per_g * cfg.kernel_size;
    if weight.len() != expected_weight {
        return Err(Conv1dError::InvalidWeightShape {
            weight_len: weight.len(),
            expected: expected_weight,
        });
    }
    if let Some(b) = bias {
        if b.len() != cfg.out_channels {
            return Err(Conv1dError::InvalidBiasLength {
                bias_len: b.len(),
                out_ch: cfg.out_channels,
            });
        }
    }
    let padded = in_len + 2 * cfg.padding;
    let ek = cfg.effective_kernel_size();
    if padded < ek {
        return Err(Conv1dError::PaddedInputTooSmall {
            padded_len: padded,
            effective_kernel: ek,
        });
    }
    Ok(in_len)
}

// ---------------------------------------------------------------------------
// Scalar reference implementation
// ---------------------------------------------------------------------------

/// Pure scalar conv1d reference with groups and dilation support.
///
/// Intended as a ground-truth for differential testing against SIMD paths.
pub fn conv1d_full_reference(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    cfg: &Conv1dConfig,
) -> Result<Vec<f32>, Conv1dError> {
    let in_len = validate(input, weight, bias, cfg)?;
    let out_len = cfg.output_length(in_len);
    let mut output = vec![0.0f32; cfg.out_channels * out_len];

    conv1d_scalar_inner(input, weight, &mut output, cfg, in_len, out_len);

    if let Some(b) = bias {
        add_bias_scalar(&mut output, b, cfg.out_channels, out_len);
    }
    Ok(output)
}

/// Inner scalar convolution loop with groups and dilation.
fn conv1d_scalar_inner(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    cfg: &Conv1dConfig,
    in_len: usize,
    out_len: usize,
) {
    let ic_per_g = cfg.in_channels_per_group();
    let oc_per_g = cfg.out_channels_per_group();

    for g in 0..cfg.groups {
        let ic_start = g * ic_per_g;
        let oc_start = g * oc_per_g;

        for oc_local in 0..oc_per_g {
            let oc = oc_start + oc_local;
            for o in 0..out_len {
                let mut acc = 0.0f32;
                let out_pos = o * cfg.stride;
                for ic_local in 0..ic_per_g {
                    let ic = ic_start + ic_local;
                    let w_base = oc * ic_per_g * cfg.kernel_size + ic_local * cfg.kernel_size;
                    let i_base = ic * in_len;
                    for k in 0..cfg.kernel_size {
                        let in_pos = out_pos + k * cfg.dilation;
                        if in_pos >= cfg.padding && in_pos < cfg.padding + in_len {
                            acc += weight[w_base + k] * input[i_base + in_pos - cfg.padding];
                        }
                    }
                }
                output[oc * out_len + o] = acc;
            }
        }
    }
}

/// Scalar bias addition.
fn add_bias_scalar(output: &mut [f32], bias: &[f32], out_ch: usize, out_len: usize) {
    for oc in 0..out_ch {
        let b = bias[oc];
        let row = &mut output[oc * out_len..(oc + 1) * out_len];
        for v in row.iter_mut() {
            *v += b;
        }
    }
}

// ---------------------------------------------------------------------------
// NEON (aarch64)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
fn conv1d_neon_inner(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    cfg: &Conv1dConfig,
    in_len: usize,
    out_len: usize,
) {
    use std::arch::aarch64::*;

    let ic_per_g = cfg.in_channels_per_group();
    let oc_per_g = cfg.out_channels_per_group();

    for g in 0..cfg.groups {
        let ic_start = g * ic_per_g;
        let oc_start = g * oc_per_g;

        for oc_local in 0..oc_per_g {
            let oc = oc_start + oc_local;
            let chunks = out_len / 4;

            // SIMD path: 4 output positions at a time.
            for chunk in 0..chunks {
                let o_base = chunk * 4;
                // SAFETY: NEON is always available on aarch64. Bounded by
                // chunk/remainder split and inner position checks.
                unsafe {
                    let mut acc = vdupq_n_f32(0.0);
                    for ic_local in 0..ic_per_g {
                        let ic = ic_start + ic_local;
                        let w_base = oc * ic_per_g * cfg.kernel_size + ic_local * cfg.kernel_size;
                        let i_base = ic * in_len;
                        for k in 0..cfg.kernel_size {
                            let wv = vdupq_n_f32(weight[w_base + k]);
                            let mut iv = [0.0f32; 4];
                            for lane in 0..4 {
                                let in_pos = (o_base + lane) * cfg.stride + k * cfg.dilation;
                                if in_pos >= cfg.padding && in_pos < cfg.padding + in_len {
                                    iv[lane] = input[i_base + in_pos - cfg.padding];
                                }
                            }
                            let input_v = vld1q_f32(iv.as_ptr());
                            acc = vfmaq_f32(acc, wv, input_v);
                        }
                    }
                    let out_ptr = output.as_mut_ptr().add(oc * out_len + o_base);
                    vst1q_f32(out_ptr, vaddq_f32(vld1q_f32(out_ptr), acc));
                }
            }

            // Scalar tail.
            let tail_start = chunks * 4;
            for o in tail_start..out_len {
                let mut acc = 0.0f32;
                let out_pos = o * cfg.stride;
                for ic_local in 0..ic_per_g {
                    let ic = ic_start + ic_local;
                    let w_base = oc * ic_per_g * cfg.kernel_size + ic_local * cfg.kernel_size;
                    let i_base = ic * in_len;
                    for k in 0..cfg.kernel_size {
                        let in_pos = out_pos + k * cfg.dilation;
                        if in_pos >= cfg.padding && in_pos < cfg.padding + in_len {
                            acc += weight[w_base + k] * input[i_base + in_pos - cfg.padding];
                        }
                    }
                }
                output[oc * out_len + o] = acc;
            }
        }
    }
}

/// NEON bias addition.
#[cfg(target_arch = "aarch64")]
fn add_bias_neon(output: &mut [f32], bias: &[f32], out_ch: usize, out_len: usize) {
    use std::arch::aarch64::*;

    for oc in 0..out_ch {
        let b = bias[oc];
        let row_start = oc * out_len;
        let chunks = out_len / 4;
        let remainder = out_len % 4;

        // SAFETY: NEON always available on aarch64. Bounded loads/stores.
        unsafe {
            let bv = vdupq_n_f32(b);
            for i in 0..chunks {
                let offset = row_start + i * 4;
                let ptr = output.as_mut_ptr().add(offset);
                let v = vld1q_f32(ptr);
                vst1q_f32(ptr, vaddq_f32(v, bv));
            }
        }
        let tail_start = row_start + chunks * 4;
        for i in 0..remainder {
            output[tail_start + i] += b;
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64)
// ---------------------------------------------------------------------------

/// AVX2 conv1d inner loop with groups and dilation.
///
/// # Safety
/// Caller must verify AVX2 and FMA are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn conv1d_avx2_inner(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    cfg: &Conv1dConfig,
    in_len: usize,
    out_len: usize,
) {
    use std::arch::x86_64::*;

    let ic_per_g = cfg.in_channels_per_group();
    let oc_per_g = cfg.out_channels_per_group();

    for g in 0..cfg.groups {
        let ic_start = g * ic_per_g;
        let oc_start = g * oc_per_g;

        for oc_local in 0..oc_per_g {
            let oc = oc_start + oc_local;
            let chunks = out_len / 8;

            for chunk in 0..chunks {
                let o_base = chunk * 8;
                let mut acc = _mm256_setzero_ps();

                for ic_local in 0..ic_per_g {
                    let ic = ic_start + ic_local;
                    let w_base = oc * ic_per_g * cfg.kernel_size + ic_local * cfg.kernel_size;
                    let i_base = ic * in_len;
                    for k in 0..cfg.kernel_size {
                        let wv = _mm256_set1_ps(weight[w_base + k]);
                        let mut iv = [0.0f32; 8];
                        for lane in 0..8 {
                            let in_pos = (o_base + lane) * cfg.stride + k * cfg.dilation;
                            if in_pos >= cfg.padding && in_pos < cfg.padding + in_len {
                                iv[lane] = input[i_base + in_pos - cfg.padding];
                            }
                        }
                        let input_v = _mm256_loadu_ps(iv.as_ptr());
                        acc = _mm256_fmadd_ps(wv, input_v, acc);
                    }
                }

                let out_ptr = output.as_mut_ptr().add(oc * out_len + o_base);
                _mm256_storeu_ps(out_ptr, _mm256_add_ps(_mm256_loadu_ps(out_ptr), acc));
            }

            // Scalar tail.
            let tail_start = chunks * 8;
            for o in tail_start..out_len {
                let mut acc = 0.0f32;
                let out_pos = o * cfg.stride;
                for ic_local in 0..ic_per_g {
                    let ic = ic_start + ic_local;
                    let w_base = oc * ic_per_g * cfg.kernel_size + ic_local * cfg.kernel_size;
                    let i_base = ic * in_len;
                    for k in 0..cfg.kernel_size {
                        let in_pos = out_pos + k * cfg.dilation;
                        if in_pos >= cfg.padding && in_pos < cfg.padding + in_len {
                            acc += weight[w_base + k] * input[i_base + in_pos - cfg.padding];
                        }
                    }
                }
                output[oc * out_len + o] = acc;
            }
        }
    }
}

/// AVX2 bias addition.
///
/// # Safety
/// Caller must verify AVX2 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_bias_avx2(output: &mut [f32], bias: &[f32], out_ch: usize, out_len: usize) {
    use std::arch::x86_64::*;

    for oc in 0..out_ch {
        let b = bias[oc];
        let row_start = oc * out_len;
        let chunks = out_len / 8;
        let remainder = out_len % 8;

        let bv = _mm256_set1_ps(b);
        for i in 0..chunks {
            let offset = row_start + i * 8;
            let ptr = output.as_mut_ptr().add(offset);
            _mm256_storeu_ps(ptr, _mm256_add_ps(_mm256_loadu_ps(ptr), bv));
        }
        let tail_start = row_start + chunks * 8;
        for i in 0..remainder {
            *output.get_unchecked_mut(tail_start + i) += b;
        }
    }
}

// ---------------------------------------------------------------------------
// Public dispatch
// ---------------------------------------------------------------------------

/// SIMD-optimized 1D convolution with groups and dilation support.
///
/// Matches PyTorch `torch.nn.Conv1d` semantics:
/// - `input`:  `[in_channels, in_length]` row-major
/// - `weight`: `[out_channels, in_channels/groups, kernel_size]` row-major
/// - `bias`:   optional `[out_channels]`
/// - Returns:  `[out_channels, out_length]` row-major
///
/// `out_length = (in_length + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`
///
/// Auto-dispatches to NEON (aarch64), AVX2 (x86_64), or scalar fallback.
pub fn conv1d_full(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    cfg: &Conv1dConfig,
) -> Result<Vec<f32>, Conv1dError> {
    let in_len = validate(input, weight, bias, cfg)?;
    let out_len = cfg.output_length(in_len);
    let mut output = vec![0.0f32; cfg.out_channels * out_len];

    #[cfg(target_arch = "aarch64")]
    {
        conv1d_neon_inner(input, weight, &mut output, cfg, in_len, out_len);
        if let Some(b) = bias {
            add_bias_neon(&mut output, b, cfg.out_channels, out_len);
        }
        return Ok(output);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA detected above.
            unsafe {
                conv1d_avx2_inner(input, weight, &mut output, cfg, in_len, out_len);
            }
            if let Some(b) = bias {
                // SAFETY: AVX2 detected above.
                unsafe {
                    add_bias_avx2(&mut output, b, cfg.out_channels, out_len);
                }
            }
            return Ok(output);
        }
    }

    #[allow(unreachable_code)]
    {
        conv1d_scalar_inner(input, weight, &mut output, cfg, in_len, out_len);
        if let Some(b) = bias {
            add_bias_scalar(&mut output, b, cfg.out_channels, out_len);
        }
        Ok(output)
    }
}

/// Convenience wrapper that creates a [`Conv1dConfig`] from individual
/// parameters and dispatches to [`conv1d_full`].
pub fn conv1d_grouped(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Vec<f32>, Conv1dError> {
    let cfg = Conv1dConfig {
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
    };
    conv1d_full(input, weight, bias, &cfg)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "simd_conv1d_tests.rs"]
mod simd_conv1d_tests;
