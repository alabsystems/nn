// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized 2D convolution.
//!
//! Layout (NCHW, row-major):
//! - Input:  `[batch, in_channels, h, w]` flattened
//! - Weight: `[out_channels, in_channels, kh, kw]` flattened
//! - Bias:   `[out_channels]` (optional)
//! - Output: `[batch, out_channels, out_h, out_w]` flattened
//!
//! `out_h = (h + 2*pad_h - kh) / stride_h + 1`
//! `out_w = (w + 2*pad_w - kw) / stride_w + 1`
//!
//! NEON (aarch64) and AVX2 (x86_64) paths with scalar fallback.

use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during conv2d.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Conv2dError {
    /// Stride must be > 0.
    ZeroStride,
    /// Kernel size must be > 0.
    ZeroKernelSize,
    /// Input channels must be > 0.
    ZeroInChannels,
    /// Output channels must be > 0.
    ZeroOutChannels,
    /// Spatial dims must be > 0.
    ZeroSpatialDim,
    /// Batch must be > 0.
    ZeroBatch,
    /// Input length mismatch.
    InvalidInputLength { got: usize, expected: usize },
    /// Weight length mismatch.
    InvalidWeightLength { got: usize, expected: usize },
    /// Bias length mismatch.
    InvalidBiasLength { got: usize, expected: usize },
    /// Output length mismatch.
    InvalidOutputLength { got: usize, expected: usize },
    /// Padded spatial dim too small for kernel.
    PaddedTooSmall {
        padded: usize,
        kernel: usize,
        dim: &'static str,
    },
}

impl fmt::Display for Conv2dError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroStride => write!(f, "stride must be > 0"),
            Self::ZeroKernelSize => write!(f, "kernel size must be > 0"),
            Self::ZeroInChannels => write!(f, "in_channels must be > 0"),
            Self::ZeroOutChannels => write!(f, "out_channels must be > 0"),
            Self::ZeroSpatialDim => write!(f, "spatial dimensions must be > 0"),
            Self::ZeroBatch => write!(f, "batch must be > 0"),
            Self::InvalidInputLength { got, expected } => {
                write!(f, "input length {got}, expected {expected}")
            }
            Self::InvalidWeightLength { got, expected } => {
                write!(f, "weight length {got}, expected {expected}")
            }
            Self::InvalidBiasLength { got, expected } => {
                write!(f, "bias length {got}, expected {expected}")
            }
            Self::InvalidOutputLength { got, expected } => {
                write!(f, "output length {got}, expected {expected}")
            }
            Self::PaddedTooSmall {
                padded,
                kernel,
                dim,
            } => {
                write!(f, "padded {dim} ({padded}) < kernel {dim} ({kernel})")
            }
        }
    }
}

impl std::error::Error for Conv2dError {}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

#[inline]
fn out_dim(spatial: usize, kernel: usize, stride: usize, pad: usize) -> usize {
    (spatial + 2 * pad - kernel) / stride + 1
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    batch: usize,
    in_ch: usize,
    out_ch: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Result<(usize, usize), Conv2dError> {
    if batch == 0 {
        return Err(Conv2dError::ZeroBatch);
    }
    if in_ch == 0 {
        return Err(Conv2dError::ZeroInChannels);
    }
    if out_ch == 0 {
        return Err(Conv2dError::ZeroOutChannels);
    }
    if h == 0 || w == 0 {
        return Err(Conv2dError::ZeroSpatialDim);
    }
    if kh == 0 || kw == 0 {
        return Err(Conv2dError::ZeroKernelSize);
    }
    if stride_h == 0 || stride_w == 0 {
        return Err(Conv2dError::ZeroStride);
    }

    let padded_h = h + 2 * pad_h;
    let padded_w = w + 2 * pad_w;
    if padded_h < kh {
        return Err(Conv2dError::PaddedTooSmall {
            padded: padded_h,
            kernel: kh,
            dim: "h",
        });
    }
    if padded_w < kw {
        return Err(Conv2dError::PaddedTooSmall {
            padded: padded_w,
            kernel: kw,
            dim: "w",
        });
    }

    let expected_input = batch * in_ch * h * w;
    if input.len() != expected_input {
        return Err(Conv2dError::InvalidInputLength {
            got: input.len(),
            expected: expected_input,
        });
    }
    let expected_weight = out_ch * in_ch * kh * kw;
    if weight.len() != expected_weight {
        return Err(Conv2dError::InvalidWeightLength {
            got: weight.len(),
            expected: expected_weight,
        });
    }
    if let Some(b) = bias {
        if b.len() != out_ch {
            return Err(Conv2dError::InvalidBiasLength {
                got: b.len(),
                expected: out_ch,
            });
        }
    }
    let oh = out_dim(h, kh, stride_h, pad_h);
    let ow = out_dim(w, kw, stride_w, pad_w);
    let expected_output = batch * out_ch * oh * ow;
    if output.len() != expected_output {
        return Err(Conv2dError::InvalidOutputLength {
            got: output.len(),
            expected: expected_output,
        });
    }
    Ok((oh, ow))
}

// ---------------------------------------------------------------------------
// Scalar reference
// ---------------------------------------------------------------------------

/// Pure scalar 2D convolution reference implementation.
///
/// Writes into caller-provided `output` buffer.
/// Layout: NCHW (batch, channels, height, width).
#[allow(clippy::too_many_arguments)]
pub fn conv2d_reference(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    batch: usize,
    in_ch: usize,
    out_ch: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Result<(), Conv2dError> {
    let (oh, ow) = validate(
        input, weight, bias, output, batch, in_ch, out_ch, h, w, kh, kw, stride_h, stride_w, pad_h,
        pad_w,
    )?;
    conv2d_scalar_inner(
        input, weight, bias, output, batch, in_ch, out_ch, h, w, kh, kw, stride_h, stride_w, pad_h,
        pad_w, oh, ow,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn conv2d_scalar_inner(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    batch: usize,
    in_ch: usize,
    out_ch: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    oh: usize,
    ow: usize,
) {
    for b in 0..batch {
        let in_batch = b * in_ch * h * w;
        let out_batch = b * out_ch * oh * ow;
        for oc in 0..out_ch {
            let bias_val = bias.map_or(0.0, |bv| bv[oc]);
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = bias_val;
                    for ic in 0..in_ch {
                        let w_base = (oc * in_ch + ic) * kh * kw;
                        let i_base = in_batch + ic * h * w;
                        for ky in 0..kh {
                            let iy = oy * stride_h + ky;
                            if iy < pad_h || iy >= pad_h + h {
                                continue;
                            }
                            let iy_raw = iy - pad_h;
                            for kx in 0..kw {
                                let ix = ox * stride_w + kx;
                                if ix < pad_w || ix >= pad_w + w {
                                    continue;
                                }
                                let ix_raw = ix - pad_w;
                                acc += weight[w_base + ky * kw + kx]
                                    * input[i_base + iy_raw * w + ix_raw];
                            }
                        }
                    }
                    output[out_batch + oc * oh * ow + oy * ow + ox] = acc;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NEON (aarch64)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn conv2d_neon_inner(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    batch: usize,
    in_ch: usize,
    out_ch: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    oh: usize,
    ow: usize,
) {
    use std::arch::aarch64::*;

    for b in 0..batch {
        let in_batch = b * in_ch * h * w;
        let out_batch = b * out_ch * oh * ow;
        for oc in 0..out_ch {
            let bias_val = bias.map_or(0.0, |bv| bv[oc]);
            for oy in 0..oh {
                let out_row = out_batch + oc * oh * ow + oy * ow;
                let chunks = ow / 4;
                let tail_start = chunks * 4;

                for chunk in 0..chunks {
                    let ox_base = chunk * 4;
                    // SAFETY: NEON always available on aarch64. Bounded by
                    // chunk split. Lane loads are bounds-checked.
                    unsafe {
                        let mut acc = vdupq_n_f32(bias_val);
                        for ic in 0..in_ch {
                            let w_base = (oc * in_ch + ic) * kh * kw;
                            let i_base = in_batch + ic * h * w;
                            for ky in 0..kh {
                                let iy = oy * stride_h + ky;
                                if iy < pad_h || iy >= pad_h + h {
                                    continue;
                                }
                                let iy_raw = iy - pad_h;
                                for kx in 0..kw {
                                    let wv = vdupq_n_f32(weight[w_base + ky * kw + kx]);
                                    let mut iv = [0.0f32; 4];
                                    for lane in 0..4 {
                                        let ix = (ox_base + lane) * stride_w + kx;
                                        if ix >= pad_w && ix < pad_w + w {
                                            iv[lane] = input[i_base + iy_raw * w + ix - pad_w];
                                        }
                                    }
                                    let input_v = vld1q_f32(iv.as_ptr());
                                    acc = vfmaq_f32(acc, wv, input_v);
                                }
                            }
                        }
                        vst1q_f32(output.as_mut_ptr().add(out_row + ox_base), acc);
                    }
                }

                // Scalar tail.
                for ox in tail_start..ow {
                    let mut acc = bias_val;
                    for ic in 0..in_ch {
                        let w_base = (oc * in_ch + ic) * kh * kw;
                        let i_base = in_batch + ic * h * w;
                        for ky in 0..kh {
                            let iy = oy * stride_h + ky;
                            if iy < pad_h || iy >= pad_h + h {
                                continue;
                            }
                            let iy_raw = iy - pad_h;
                            for kx in 0..kw {
                                let ix = ox * stride_w + kx;
                                if ix < pad_w || ix >= pad_w + w {
                                    continue;
                                }
                                acc += weight[w_base + ky * kw + kx]
                                    * input[i_base + iy_raw * w + ix - pad_w];
                            }
                        }
                    }
                    output[out_row + ox] = acc;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn conv2d_avx2_inner(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    batch: usize,
    in_ch: usize,
    out_ch: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    oh: usize,
    ow: usize,
) {
    use std::arch::x86_64::*;

    for b in 0..batch {
        let in_batch = b * in_ch * h * w;
        let out_batch = b * out_ch * oh * ow;
        for oc in 0..out_ch {
            let bias_val = bias.map_or(0.0, |bv| bv[oc]);
            for oy in 0..oh {
                let out_row = out_batch + oc * oh * ow + oy * ow;
                let chunks = ow / 8;
                let tail_start = chunks * 8;

                for chunk in 0..chunks {
                    let ox_base = chunk * 8;
                    let mut acc = _mm256_set1_ps(bias_val);
                    for ic in 0..in_ch {
                        let w_base = (oc * in_ch + ic) * kh * kw;
                        let i_base = in_batch + ic * h * w;
                        for ky in 0..kh {
                            let iy = oy * stride_h + ky;
                            if iy < pad_h || iy >= pad_h + h {
                                continue;
                            }
                            let iy_raw = iy - pad_h;
                            for kx in 0..kw {
                                let wv = _mm256_set1_ps(weight[w_base + ky * kw + kx]);
                                let mut iv = [0.0f32; 8];
                                for lane in 0..8 {
                                    let ix = (ox_base + lane) * stride_w + kx;
                                    if ix >= pad_w && ix < pad_w + w {
                                        iv[lane] = input[i_base + iy_raw * w + ix - pad_w];
                                    }
                                }
                                let input_v = _mm256_loadu_ps(iv.as_ptr());
                                acc = _mm256_fmadd_ps(wv, input_v, acc);
                            }
                        }
                    }
                    _mm256_storeu_ps(output.as_mut_ptr().add(out_row + ox_base), acc);
                }

                // Scalar tail.
                for ox in tail_start..ow {
                    let mut acc = bias_val;
                    for ic in 0..in_ch {
                        let w_base = (oc * in_ch + ic) * kh * kw;
                        let i_base = in_batch + ic * h * w;
                        for ky in 0..kh {
                            let iy = oy * stride_h + ky;
                            if iy < pad_h || iy >= pad_h + h {
                                continue;
                            }
                            let iy_raw = iy - pad_h;
                            for kx in 0..kw {
                                let ix = ox * stride_w + kx;
                                if ix < pad_w || ix >= pad_w + w {
                                    continue;
                                }
                                acc += weight[w_base + ky * kw + kx]
                                    * input[i_base + iy_raw * w + ix - pad_w];
                            }
                        }
                    }
                    output[out_row + ox] = acc;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public dispatch
// ---------------------------------------------------------------------------

/// SIMD-optimized 2D convolution (NCHW layout).
///
/// - `input`:  `[batch, in_ch, h, w]` flattened row-major
/// - `weight`: `[out_ch, in_ch, kh, kw]` flattened row-major
/// - `bias`:   optional `[out_ch]`
/// - `output`: `[batch, out_ch, out_h, out_w]` caller-allocated
///
/// Auto-dispatches to NEON (aarch64), AVX2 (x86_64), or scalar fallback.
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    batch: usize,
    in_ch: usize,
    out_ch: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Result<(), Conv2dError> {
    let (oh, ow) = validate(
        input, weight, bias, output, batch, in_ch, out_ch, h, w, kh, kw, stride_h, stride_w, pad_h,
        pad_w,
    )?;

    #[cfg(target_arch = "aarch64")]
    {
        conv2d_neon_inner(
            input, weight, bias, output, batch, in_ch, out_ch, h, w, kh, kw, stride_h, stride_w,
            pad_h, pad_w, oh, ow,
        );
        return Ok(());
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA detected above.
            unsafe {
                conv2d_avx2_inner(
                    input, weight, bias, output, batch, in_ch, out_ch, h, w, kh, kw, stride_h,
                    stride_w, pad_h, pad_w, oh, ow,
                );
            }
            return Ok(());
        }
    }

    #[allow(unreachable_code)]
    {
        conv2d_scalar_inner(
            input, weight, bias, output, batch, in_ch, out_ch, h, w, kh, kw, stride_h, stride_w,
            pad_h, pad_w, oh, ow,
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "simd_conv2d_tests.rs"]
mod simd_conv2d_tests;
