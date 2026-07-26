// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized 1D convolution for the CPU backend.
//!
//! Computes `conv1d` over input layout `[batch, in_channels, length]` stored as
//! a flat `[in_ch * in_len]` slice (single-batch). Weight layout is
//! `[out_ch, in_ch, kernel_size]` (row-major).
//!
//! Two strategies:
//! - **Direct convolution** (default): NEON `vfmaq_f32` / AVX2 `_mm256_fmadd_ps`
//!   accumulation in the inner (in_ch * kernel_size) loop.
//! - **im2col + matmul** (large kernels, `kernel_size > IM2COL_THRESHOLD`): unfold
//!   the input into a column matrix then delegate to `crate::matmul::matmul_tiled`.
//!
//! Both paths share the same public API and produce identical results.

/// Kernel size threshold above which im2col + matmul is used instead of direct
/// convolution. Chosen empirically: for large kernels the column-matrix layout
/// gives better cache behavior and amortises SIMD over the full matmul.
const IM2COL_THRESHOLD: usize = 7;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// 1D convolution: `[in_ch, in_len] * [out_ch, in_ch, kernel_size] -> [out_ch, out_len]`.
///
/// `input`  — row-major `[in_ch, in_len]`.
/// `weight` — row-major `[out_ch, in_ch, kernel_size]`.
/// `bias`   — optional per-output-channel bias `[out_ch]`.
///
/// Returns a `Vec<f32>` of shape `[out_ch, out_len]` where
/// `out_len = (in_len + 2*padding - kernel_size) / stride + 1`.
///
/// Auto-dispatches to NEON (aarch64), AVX2 (x86_64), or scalar fallback.
#[must_use]
pub fn conv1d(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_ch: usize,
    out_ch: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    assert!(stride > 0, "stride must be > 0");
    assert!(kernel_size > 0, "kernel_size must be > 0");
    assert!(in_ch > 0, "in_ch must be > 0");
    assert!(out_ch > 0, "out_ch must be > 0");

    let in_len = input.len() / in_ch;
    assert_eq!(
        input.len(),
        in_ch * in_len,
        "input length must be in_ch * in_len"
    );
    assert_eq!(
        weight.len(),
        out_ch * in_ch * kernel_size,
        "weight must be [out_ch, in_ch, kernel_size]"
    );
    if let Some(b) = bias {
        assert_eq!(b.len(), out_ch, "bias must be [out_ch]");
    }

    let padded_len = in_len + 2 * padding;
    assert!(
        padded_len >= kernel_size,
        "padded input length ({padded_len}) must be >= kernel_size ({kernel_size})"
    );
    let out_len = (padded_len - kernel_size) / stride + 1;

    if kernel_size > IM2COL_THRESHOLD {
        return conv1d_im2col(
            input,
            weight,
            bias,
            in_ch,
            out_ch,
            kernel_size,
            stride,
            padding,
            in_len,
            out_len,
        );
    }

    // Direct convolution path with SIMD dispatch.
    let mut output = vec![0.0f32; out_ch * out_len];

    #[cfg(target_arch = "aarch64")]
    {
        conv1d_direct_neon(
            input,
            weight,
            &mut output,
            in_ch,
            out_ch,
            kernel_size,
            stride,
            padding,
            in_len,
            out_len,
        );
        if let Some(b) = bias {
            add_bias_neon(&mut output, b, out_ch, out_len);
        }
        return output;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA detected above.
            unsafe {
                conv1d_direct_avx2(
                    input,
                    weight,
                    &mut output,
                    in_ch,
                    out_ch,
                    kernel_size,
                    stride,
                    padding,
                    in_len,
                    out_len,
                );
            }
            if let Some(b) = bias {
                // SAFETY: AVX2 detected above.
                unsafe { add_bias_avx2(&mut output, b, out_ch, out_len) };
            }
            return output;
        }
    }

    #[allow(unreachable_code)]
    {
        conv1d_direct_scalar(
            input,
            weight,
            &mut output,
            in_ch,
            out_ch,
            kernel_size,
            stride,
            padding,
            in_len,
            out_len,
        );
        if let Some(b) = bias {
            add_bias_scalar(&mut output, b, out_ch, out_len);
        }
        output
    }
}

/// Naive scalar conv1d for use as a test reference.
///
/// Same signature and semantics as [`conv1d`] but always uses scalar code.
#[must_use]
pub fn conv1d_scalar_reference(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_ch: usize,
    out_ch: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    let in_len = input.len() / in_ch;
    let padded_len = in_len + 2 * padding;
    assert!(padded_len >= kernel_size);
    let out_len = (padded_len - kernel_size) / stride + 1;
    let mut output = vec![0.0f32; out_ch * out_len];

    conv1d_direct_scalar(
        input,
        weight,
        &mut output,
        in_ch,
        out_ch,
        kernel_size,
        stride,
        padding,
        in_len,
        out_len,
    );
    if let Some(b) = bias {
        add_bias_scalar(&mut output, b, out_ch, out_len);
    }
    output
}

// ---------------------------------------------------------------------------
// Scalar direct convolution
// ---------------------------------------------------------------------------

/// Scalar direct conv1d accumulation (no SIMD).
fn conv1d_direct_scalar(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    in_ch: usize,
    out_ch: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    in_len: usize,
    out_len: usize,
) {
    for oc in 0..out_ch {
        for o in 0..out_len {
            let mut acc = 0.0f32;
            let out_pos = o * stride;
            for ic in 0..in_ch {
                let w_base = oc * in_ch * kernel_size + ic * kernel_size;
                let i_base = ic * in_len;
                for k in 0..kernel_size {
                    let in_pos = out_pos + k;
                    // Apply zero-padding: positions outside [padding, padding+in_len)
                    // contribute zero.
                    if in_pos >= padding && in_pos < padding + in_len {
                        acc += weight[w_base + k] * input[i_base + in_pos - padding];
                    }
                }
            }
            output[oc * out_len + o] = acc;
        }
    }
}

/// Scalar bias addition: `output[oc, :] += bias[oc]`.
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
// NEON direct convolution (aarch64)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
fn conv1d_direct_neon(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    in_ch: usize,
    out_ch: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    in_len: usize,
    out_len: usize,
) {
    use std::arch::aarch64::*;

    for oc in 0..out_ch {
        // Process 4 output positions at a time with NEON.
        let chunks = out_len / 4;

        for chunk in 0..chunks {
            let o_base = chunk * 4;
            // SAFETY: NEON is always available on aarch64. Bounds are checked
            // by the chunk/remainder split and the inner position check.
            unsafe {
                let mut acc = vdupq_n_f32(0.0);
                for ic in 0..in_ch {
                    let w_base = oc * in_ch * kernel_size + ic * kernel_size;
                    let i_base = ic * in_len;
                    for k in 0..kernel_size {
                        let wv = vdupq_n_f32(weight[w_base + k]);
                        // Gather 4 input values for the 4 output positions.
                        let mut iv = [0.0f32; 4];
                        for lane in 0..4 {
                            let in_pos = (o_base + lane) * stride + k;
                            if in_pos >= padding && in_pos < padding + in_len {
                                iv[lane] = input[i_base + in_pos - padding];
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

        // Scalar tail for remaining positions.
        let tail_start = chunks * 4;
        for o in tail_start..out_len {
            let mut acc = 0.0f32;
            let out_pos = o * stride;
            for ic in 0..in_ch {
                let w_base = oc * in_ch * kernel_size + ic * kernel_size;
                let i_base = ic * in_len;
                for k in 0..kernel_size {
                    let in_pos = out_pos + k;
                    if in_pos >= padding && in_pos < padding + in_len {
                        acc += weight[w_base + k] * input[i_base + in_pos - padding];
                    }
                }
            }
            output[oc * out_len + o] = acc;
        }
    }
}

/// NEON bias addition: `vaddq_f32` over output rows.
#[cfg(target_arch = "aarch64")]
fn add_bias_neon(output: &mut [f32], bias: &[f32], out_ch: usize, out_len: usize) {
    use std::arch::aarch64::*;

    for oc in 0..out_ch {
        let b = bias[oc];
        let row_start = oc * out_len;
        let chunks = out_len / 4;
        let remainder = out_len % 4;

        // SAFETY: NEON is always available on aarch64. Bounded loads/stores.
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
// AVX2 direct convolution (x86_64)
// ---------------------------------------------------------------------------

/// AVX2 direct conv1d with FMA accumulation.
///
/// # Safety
/// Caller must verify AVX2 and FMA are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn conv1d_direct_avx2(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    in_ch: usize,
    out_ch: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    in_len: usize,
    out_len: usize,
) {
    use std::arch::x86_64::*;

    for oc in 0..out_ch {
        let chunks = out_len / 8;
        let remainder = out_len % 8;

        for chunk in 0..chunks {
            let o_base = chunk * 8;
            let mut acc = _mm256_setzero_ps();

            for ic in 0..in_ch {
                let w_base = oc * in_ch * kernel_size + ic * kernel_size;
                let i_base = ic * in_len;
                for k in 0..kernel_size {
                    let wv = _mm256_set1_ps(weight[w_base + k]);
                    // Gather 8 input values.
                    let mut iv = [0.0f32; 8];
                    for lane in 0..8 {
                        let in_pos = (o_base + lane) * stride + k;
                        if in_pos >= padding && in_pos < padding + in_len {
                            iv[lane] = input[i_base + in_pos - padding];
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
            let out_pos = o * stride;
            for ic in 0..in_ch {
                let w_base = oc * in_ch * kernel_size + ic * kernel_size;
                let i_base = ic * in_len;
                for k in 0..kernel_size {
                    let in_pos = out_pos + k;
                    if in_pos >= padding && in_pos < padding + in_len {
                        acc += weight[w_base + k] * input[i_base + in_pos - padding];
                    }
                }
            }
            output[oc * out_len + o] = acc;
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
// im2col + matmul path (large kernels)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "conv1d_tests.rs"]
mod conv1d_tests;

/// Conv1d via im2col: unfold input into columns, then matmul.
///
/// For each output position, the column matrix contains the flattened
/// `[in_ch, kernel_size]` input patch. The weight matrix is
/// `[out_ch, in_ch * kernel_size]`.
///
/// The matmul is `[out_ch, in_ch * kernel_size] x [in_ch * kernel_size, out_len]`
/// = `[out_ch, out_len]`.
#[allow(clippy::too_many_arguments)]
fn conv1d_im2col(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_ch: usize,
    out_ch: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    in_len: usize,
    out_len: usize,
) -> Vec<f32> {
    let col_rows = in_ch * kernel_size;
    let col_cols = out_len;

    // Build the column matrix: each column is one output position's input patch.
    let mut col = vec![0.0f32; col_rows * col_cols];
    for o in 0..out_len {
        for ic in 0..in_ch {
            for k in 0..kernel_size {
                let in_pos = o * stride + k;
                let val = if in_pos >= padding && in_pos < padding + in_len {
                    input[ic * in_len + in_pos - padding]
                } else {
                    0.0
                };
                // Column-matrix is row-major [col_rows, col_cols].
                col[(ic * kernel_size + k) * col_cols + o] = val;
            }
        }
    }

    // Matmul: weight [out_ch, col_rows] * col [col_rows, col_cols] = output [out_ch, col_cols].
    let mut output = vec![0.0f32; out_ch * out_len];
    crate::matmul::matmul_tiled(weight, &col, &mut output, out_ch, col_rows, col_cols);

    // Add bias.
    if let Some(b) = bias {
        // Use platform-dispatched bias addition.
        #[cfg(target_arch = "aarch64")]
        {
            add_bias_neon(&mut output, b, out_ch, out_len);
            return output;
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 detected above.
                unsafe { add_bias_avx2(&mut output, b, out_ch, out_len) };
                return output;
            }
        }

        #[allow(unreachable_code)]
        add_bias_scalar(&mut output, b, out_ch, out_len);
    }

    output
}
