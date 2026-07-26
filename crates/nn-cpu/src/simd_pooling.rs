// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized pooling operations.
//!
//! Provides 1D and 2D max/average pooling with padding support.
//! - NEON-optimized (aarch64)
//! - AVX2-optimized (x86_64)
//! - Pure scalar fallback
//!
//! Layout: `[batch, channels, spatial...]` (NCHW for 2D, NCL for 1D).

// ---------------------------------------------------------------------------
// Output length helper
// ---------------------------------------------------------------------------

/// Computes the output length for a pooling dimension.
#[inline]
fn pool_output_len(input_len: usize, kernel_size: usize, stride: usize, padding: usize) -> usize {
    (input_len + 2 * padding - kernel_size) / stride + 1
}

// ---------------------------------------------------------------------------
// Reference (scalar) implementations
// ---------------------------------------------------------------------------

/// Reference 1D max pooling (scalar, no SIMD).
pub fn max_pool1d_reference(
    input: &[f32],
    output: &mut [f32],
    batch: usize,
    channels: usize,
    input_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) {
    let out_len = pool_output_len(input_len, kernel_size, stride, padding);
    assert_eq!(input.len(), batch * channels * input_len);
    assert_eq!(output.len(), batch * channels * out_len);

    for b in 0..batch {
        for c in 0..channels {
            let in_offset = (b * channels + c) * input_len;
            let out_offset = (b * channels + c) * out_len;
            for o in 0..out_len {
                let start = o * stride;
                let mut max_val = f32::NEG_INFINITY;
                for k in 0..kernel_size {
                    let idx = start + k;
                    if idx >= padding && idx < input_len + padding {
                        let val = input[in_offset + idx - padding];
                        if val > max_val {
                            max_val = val;
                        }
                    }
                    // Padded positions contribute NEG_INFINITY (identity for max).
                }
                output[out_offset + o] = max_val;
            }
        }
    }
}

/// Reference 1D average pooling (scalar, no SIMD).
pub fn avg_pool1d_reference(
    input: &[f32],
    output: &mut [f32],
    batch: usize,
    channels: usize,
    input_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) {
    let out_len = pool_output_len(input_len, kernel_size, stride, padding);
    assert_eq!(input.len(), batch * channels * input_len);
    assert_eq!(output.len(), batch * channels * out_len);

    for b in 0..batch {
        for c in 0..channels {
            let in_offset = (b * channels + c) * input_len;
            let out_offset = (b * channels + c) * out_len;
            for o in 0..out_len {
                let start = o * stride;
                let mut sum = 0.0f32;
                for k in 0..kernel_size {
                    let idx = start + k;
                    if idx >= padding && idx < input_len + padding {
                        sum += input[in_offset + idx - padding];
                    }
                    // Padded positions contribute 0.0 (identity for sum).
                }
                output[out_offset + o] = sum / kernel_size as f32;
            }
        }
    }
}

/// Reference 2D max pooling (scalar, no SIMD).
pub fn max_pool2d_reference(
    input: &[f32],
    output: &mut [f32],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) {
    let out_h = pool_output_len(h, kh, stride_h, pad_h);
    let out_w = pool_output_len(w, kw, stride_w, pad_w);
    assert_eq!(input.len(), batch * channels * h * w);
    assert_eq!(output.len(), batch * channels * out_h * out_w);

    for b in 0..batch {
        for c in 0..channels {
            let in_base = (b * channels + c) * h * w;
            let out_base = (b * channels + c) * out_h * out_w;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut max_val = f32::NEG_INFINITY;
                    for fh in 0..kh {
                        for fw in 0..kw {
                            let ih = oh * stride_h + fh;
                            let iw = ow * stride_w + fw;
                            if ih >= pad_h && ih < h + pad_h && iw >= pad_w && iw < w + pad_w {
                                let val = input[in_base + (ih - pad_h) * w + (iw - pad_w)];
                                if val > max_val {
                                    max_val = val;
                                }
                            }
                        }
                    }
                    output[out_base + oh * out_w + ow] = max_val;
                }
            }
        }
    }
}

/// Reference 2D average pooling (scalar, no SIMD).
pub fn avg_pool2d_reference(
    input: &[f32],
    output: &mut [f32],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) {
    let out_h = pool_output_len(h, kh, stride_h, pad_h);
    let out_w = pool_output_len(w, kw, stride_w, pad_w);
    assert_eq!(input.len(), batch * channels * h * w);
    assert_eq!(output.len(), batch * channels * out_h * out_w);

    let k_area = (kh * kw) as f32;
    for b in 0..batch {
        for c in 0..channels {
            let in_base = (b * channels + c) * h * w;
            let out_base = (b * channels + c) * out_h * out_w;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0f32;
                    for fh in 0..kh {
                        for fw in 0..kw {
                            let ih = oh * stride_h + fh;
                            let iw = ow * stride_w + fw;
                            if ih >= pad_h && ih < h + pad_h && iw >= pad_w && iw < w + pad_w {
                                sum += input[in_base + (ih - pad_h) * w + (iw - pad_w)];
                            }
                        }
                    }
                    output[out_base + oh * out_w + ow] = sum / k_area;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SIMD-accelerated 1D max pooling
// ---------------------------------------------------------------------------

/// 1D max pooling. Auto-dispatches to NEON/AVX2/scalar.
///
/// Layout: `[batch, channels, input_len]`.
/// Output length: `(input_len + 2*padding - kernel_size) / stride + 1`.
#[allow(clippy::too_many_arguments)]
pub fn max_pool1d(
    input: &[f32],
    output: &mut [f32],
    batch: usize,
    channels: usize,
    input_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) {
    let out_len = pool_output_len(input_len, kernel_size, stride, padding);
    assert_eq!(input.len(), batch * channels * input_len);
    assert_eq!(output.len(), batch * channels * out_len);

    // When padding == 0 and kernel elements are contiguous, we can use SIMD
    // max reduction over the kernel window. With padding, fall back to reference.
    if padding > 0 {
        max_pool1d_reference(
            input,
            output,
            batch,
            channels,
            input_len,
            kernel_size,
            stride,
            padding,
        );
        return;
    }

    for b in 0..batch {
        for c in 0..channels {
            let in_offset = (b * channels + c) * input_len;
            let out_offset = (b * channels + c) * out_len;
            for o in 0..out_len {
                let window_start = o * stride;
                let window =
                    &input[in_offset + window_start..in_offset + window_start + kernel_size];
                output[out_offset + o] = simd_max_slice(window);
            }
        }
    }
}

/// 1D average pooling. Auto-dispatches to NEON/AVX2/scalar.
#[allow(clippy::too_many_arguments)]
pub fn avg_pool1d(
    input: &[f32],
    output: &mut [f32],
    batch: usize,
    channels: usize,
    input_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) {
    let out_len = pool_output_len(input_len, kernel_size, stride, padding);
    assert_eq!(input.len(), batch * channels * input_len);
    assert_eq!(output.len(), batch * channels * out_len);

    if padding > 0 {
        avg_pool1d_reference(
            input,
            output,
            batch,
            channels,
            input_len,
            kernel_size,
            stride,
            padding,
        );
        return;
    }

    let inv_k = 1.0 / kernel_size as f32;
    for b in 0..batch {
        for c in 0..channels {
            let in_offset = (b * channels + c) * input_len;
            let out_offset = (b * channels + c) * out_len;
            for o in 0..out_len {
                let window_start = o * stride;
                let window =
                    &input[in_offset + window_start..in_offset + window_start + kernel_size];
                output[out_offset + o] = simd_sum_slice(window) * inv_k;
            }
        }
    }
}

/// 2D max pooling. Auto-dispatches to NEON/AVX2/scalar.
///
/// Layout: `[batch, channels, h, w]`.
#[allow(clippy::too_many_arguments)]
pub fn max_pool2d(
    input: &[f32],
    output: &mut [f32],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) {
    let out_h = pool_output_len(h, kh, stride_h, pad_h);
    let out_w = pool_output_len(w, kw, stride_w, pad_w);
    assert_eq!(input.len(), batch * channels * h * w);
    assert_eq!(output.len(), batch * channels * out_h * out_w);

    if pad_h > 0 || pad_w > 0 {
        max_pool2d_reference(
            input, output, batch, channels, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w,
        );
        return;
    }

    for b in 0..batch {
        for c in 0..channels {
            let in_base = (b * channels + c) * h * w;
            let out_base = (b * channels + c) * out_h * out_w;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut max_val = f32::NEG_INFINITY;
                    for fh in 0..kh {
                        let row_start = in_base + (oh * stride_h + fh) * w + ow * stride_w;
                        let row_slice = &input[row_start..row_start + kw];
                        let row_max = simd_max_slice(row_slice);
                        if row_max > max_val {
                            max_val = row_max;
                        }
                    }
                    output[out_base + oh * out_w + ow] = max_val;
                }
            }
        }
    }
}

/// 2D average pooling. Auto-dispatches to NEON/AVX2/scalar.
#[allow(clippy::too_many_arguments)]
pub fn avg_pool2d(
    input: &[f32],
    output: &mut [f32],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) {
    let out_h = pool_output_len(h, kh, stride_h, pad_h);
    let out_w = pool_output_len(w, kw, stride_w, pad_w);
    assert_eq!(input.len(), batch * channels * h * w);
    assert_eq!(output.len(), batch * channels * out_h * out_w);

    if pad_h > 0 || pad_w > 0 {
        avg_pool2d_reference(
            input, output, batch, channels, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w,
        );
        return;
    }

    let inv_k = 1.0 / (kh * kw) as f32;
    for b in 0..batch {
        for c in 0..channels {
            let in_base = (b * channels + c) * h * w;
            let out_base = (b * channels + c) * out_h * out_w;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0f32;
                    for fh in 0..kh {
                        let row_start = in_base + (oh * stride_h + fh) * w + ow * stride_w;
                        let row_slice = &input[row_start..row_start + kw];
                        sum += simd_sum_slice(row_slice);
                    }
                    output[out_base + oh * out_w + ow] = sum * inv_k;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SIMD micro-kernels for max and sum over a small slice
// ---------------------------------------------------------------------------

/// SIMD-accelerated max over a contiguous f32 slice.
#[inline]
fn simd_max_slice(data: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return simd_max_slice_neon(data);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            return unsafe { simd_max_slice_avx2(data) };
        }
    }

    #[allow(unreachable_code)]
    simd_max_slice_scalar(data)
}

/// SIMD-accelerated sum over a contiguous f32 slice.
#[inline]
fn simd_sum_slice(data: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return simd_sum_slice_neon(data);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            return unsafe { simd_sum_slice_avx2(data) };
        }
    }

    #[allow(unreachable_code)]
    simd_sum_slice_scalar(data)
}

// --- Scalar ---

fn simd_max_slice_scalar(data: &[f32]) -> f32 {
    let mut m = f32::NEG_INFINITY;
    for &v in data {
        if v > m {
            m = v;
        }
    }
    m
}

fn simd_sum_slice_scalar(data: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for &v in data {
        s += v;
    }
    s
}

// --- NEON (aarch64) ---

#[cfg(target_arch = "aarch64")]
fn simd_max_slice_neon(data: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = data.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: aarch64 NEON is always available. Bounded loads within slice.
    unsafe {
        let mut acc = vdupq_n_f32(f32::NEG_INFINITY);
        for i in 0..chunks {
            let v = vld1q_f32(data.as_ptr().add(i * 4));
            acc = vmaxq_f32(acc, v);
        }
        // Horizontal max of 4 lanes.
        let mut m = vmaxvq_f32(acc);
        let tail = chunks * 4;
        for i in 0..remainder {
            let v = data[tail + i];
            if v > m {
                m = v;
            }
        }
        m
    }
}

#[cfg(target_arch = "aarch64")]
fn simd_sum_slice_neon(data: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = data.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: aarch64 NEON is always available. Bounded loads within slice.
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(data.as_ptr().add(i * 4));
            acc = vaddq_f32(acc, v);
        }
        let mut s = vaddvq_f32(acc);
        let tail = chunks * 4;
        for i in 0..remainder {
            s += data[tail + i];
        }
        s
    }
}

// --- AVX2 (x86_64) ---

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_max_slice_avx2(data: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = data.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut acc = _mm256_set1_ps(f32::NEG_INFINITY);
    for i in 0..chunks {
        // SAFETY: offset + 8 <= n from loop bound. Unaligned loads.
        let v = _mm256_loadu_ps(data.as_ptr().add(i * 8));
        acc = _mm256_max_ps(acc, v);
    }

    // Horizontal max: reduce 8 lanes to scalar.
    let hi = _mm256_extractf128_ps::<1>(acc);
    let lo = _mm256_castps256_ps128(acc);
    let max128 = _mm_max_ps(hi, lo);
    let shuf = _mm_movehdup_ps(max128);
    let max64 = _mm_max_ps(max128, shuf);
    let shuf2 = _mm_movehl_ps(max64, max64);
    let max32 = _mm_max_ss(max64, shuf2);
    let mut m = _mm_cvtss_f32(max32);

    let tail = chunks * 8;
    for i in 0..remainder {
        let v = data[tail + i];
        if v > m {
            m = v;
        }
    }
    m
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_sum_slice_avx2(data: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = data.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut acc = _mm256_setzero_ps();
    for i in 0..chunks {
        // SAFETY: offset + 8 <= n from loop bound. Unaligned loads.
        let v = _mm256_loadu_ps(data.as_ptr().add(i * 8));
        acc = _mm256_add_ps(acc, v);
    }

    // Horizontal sum: reduce 8 lanes to scalar.
    let hi = _mm256_extractf128_ps::<1>(acc);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(hi, lo);
    let shuf = _mm_movehdup_ps(sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sum64, sum64);
    let sum32 = _mm_add_ss(sum64, shuf2);
    let mut s = _mm_cvtss_f32(sum32);

    let tail = chunks * 8;
    for i in 0..remainder {
        s += data[tail + i];
    }
    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "simd_pooling_tests.rs"]
mod simd_pooling_tests;
