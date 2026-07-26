// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! 2-D pooling operations (max pool, average pool, adaptive average pool).
//!
//! Input shape: `[batch, channels, height, width]`
//! Output shape: `[batch, channels, out_height, out_width]`

use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::{gpu_backend_dispatch, DynTensor};
use crate::error::{Result, TensorError};
use crate::Device;
use ndarray::{ArrayD, IxDyn};

/// Compute the output length for a pooling dimension.
///
/// Same formula as convolution with dilation=1:
/// `out = (input + 2*padding - kernel_size) / stride + 1`
pub(crate) fn pool2d_out_len(
    input_len: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
    ceil_mode: bool,
) -> Result<usize> {
    if kernel_size == 0 {
        return Err(TensorError::ConvParameterInvalid {
            param: "kernel_size",
            value: 0,
            reason: "must be > 0",
        });
    }
    if stride == 0 {
        return Err(TensorError::ConvParameterInvalid {
            param: "stride",
            value: 0,
            reason: "must be > 0",
        });
    }
    let padded = input_len
        .checked_add(2usize.checked_mul(padding).ok_or_else(|| {
            TensorError::InvalidShape(format!("pool2d: padding overflow (padding={padding})"))
        })?)
        .ok_or_else(|| {
            TensorError::InvalidShape(format!(
                "pool2d: padded input length overflow (input_len={input_len}, padding={padding})"
            ))
        })?;
    if padded < kernel_size {
        return Err(TensorError::InvalidShape(format!(
            "pool2d: padded input length {padded} < kernel_size {kernel_size}"
        )));
    }
    let numerator = padded - kernel_size;
    if ceil_mode {
        Ok(numerator.div_ceil(stride) + 1)
    } else {
        Ok(numerator / stride + 1)
    }
}

impl DynTensor {
    /// 1-D max pooling.
    ///
    /// Input shape: `[batch, channels, length]`
    /// Output shape: `[batch, channels, out_length]`
    ///
    /// Selects the maximum value in each pooling window along the last dimension.
    pub fn max_pool1d(&self, kernel_size: usize, stride: usize, padding: usize) -> Result<Self> {
        self.max_pool1d_ceil(kernel_size, stride, padding, false)
    }

    /// 1-D max pooling with ceil_mode option.
    pub fn max_pool1d_ceil(
        &self,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        ceil_mode: bool,
    ) -> Result<Self> {
        let (batch, channels, in_len) = self.dims3()?;
        let out_len = pool2d_out_len(in_len, kernel_size, padding, stride, ceil_mode)?;

        let mut result = if self.device().is_gpu() {
            let original_device = self.device();
            let cpu_input = self.to_device(&Device::Cpu)?;
            let r = cpu_input.max_pool1d_ceil(kernel_size, stride, padding, ceil_mode)?;
            r.to_device(&original_device)?
        } else {
            let input_dtype = self.dtype;
            let input_c = self.contiguous()?;
            let input_data = input_c.to_f32_array()?;
            let inp = input_data.as_slice().ok_or_else(|| {
                TensorError::InvalidShape("max_pool1d: input not contiguous".into())
            })?;

            let buf_len = super::checked_buffer_len(&[batch, channels, out_len], "max_pool1d")?;
            let mut output = vec![f32::NEG_INFINITY; buf_len];

            for b in 0..batch {
                for c in 0..channels {
                    for ol in 0..out_len {
                        let mut max_val = f32::NEG_INFINITY;
                        for k in 0..kernel_size {
                            let il = ol * stride + k;
                            if il >= padding && il - padding < in_len {
                                let idx = b * channels * in_len + c * in_len + (il - padding);
                                let val = inp[idx];
                                if val > max_val {
                                    max_val = val;
                                }
                            }
                        }
                        let out_idx = b * channels * out_len + c * out_len + ol;
                        output[out_idx] = max_val;
                    }
                }
            }

            Self::from_f32_result(
                ArrayD::from_shape_vec(IxDyn(&[batch, channels, out_len]), output)?,
                input_dtype,
            )?
        };

        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::MaxPool1d {
                    kernel_size,
                    stride,
                    padding,
                },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }

        Ok(result)
    }

    /// 2-D max pooling.
    ///
    /// Input shape: `[batch, channels, height, width]`
    /// Output shape: `[batch, channels, out_height, out_width]`
    ///
    /// Selects the maximum value in each pooling window.
    pub fn max_pool2d(&self, kernel_size: usize, stride: usize, padding: usize) -> Result<Self> {
        self.max_pool2d_ceil(kernel_size, stride, padding, false)
    }

    /// 2-D max pooling with ceil_mode option.
    pub fn max_pool2d_ceil(
        &self,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        ceil_mode: bool,
    ) -> Result<Self> {
        let (batch, channels, in_h, in_w) = self.dims4()?;
        let out_h = pool2d_out_len(in_h, kernel_size, padding, stride, ceil_mode)?;
        let out_w = pool2d_out_len(in_w, kernel_size, padding, stride, ceil_mode)?;

        let mut result = if self.device().is_gpu() {
            // Try native GPU dispatch; fall back to CPU round-trip.
            if !ceil_mode {
                if let Some(gpu_result) =
                    gpu_backend_dispatch(|b| b.max_pool2d(self, kernel_size, stride, padding))
                {
                    gpu_result?
                } else {
                    let original_device = self.device();
                    let cpu_input = self.to_device(&Device::Cpu)?;
                    let r = cpu_input.max_pool2d_ceil(kernel_size, stride, padding, ceil_mode)?;
                    r.to_device(&original_device)?
                }
            } else {
                // ceil_mode: CPU round-trip (GPU kernel only supports floor mode).
                let original_device = self.device();
                let cpu_input = self.to_device(&Device::Cpu)?;
                let r = cpu_input.max_pool2d_ceil(kernel_size, stride, padding, ceil_mode)?;
                r.to_device(&original_device)?
            }
        } else {
            let input_dtype = self.dtype;
            let input_c = self.contiguous()?;
            let input_data = input_c.to_f32_array()?;
            let inp = input_data.as_slice().ok_or_else(|| {
                TensorError::InvalidShape("max_pool2d: input not contiguous".into())
            })?;

            let buf_len =
                super::checked_buffer_len(&[batch, channels, out_h, out_w], "max_pool2d")?;
            let mut output = vec![f32::NEG_INFINITY; buf_len];

            for b in 0..batch {
                for c in 0..channels {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut max_val = f32::NEG_INFINITY;
                            for kh in 0..kernel_size {
                                for kw in 0..kernel_size {
                                    let ih = oh * stride + kh;
                                    let iw = ow * stride + kw;
                                    if ih >= padding
                                        && ih - padding < in_h
                                        && iw >= padding
                                        && iw - padding < in_w
                                    {
                                        let idx = b * channels * in_h * in_w
                                            + c * in_h * in_w
                                            + (ih - padding) * in_w
                                            + (iw - padding);
                                        let val = inp[idx];
                                        if val > max_val {
                                            max_val = val;
                                        }
                                    }
                                }
                            }
                            let out_idx =
                                b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                            output[out_idx] = max_val;
                        }
                    }
                }
            }

            Self::from_f32_result(
                ArrayD::from_shape_vec(IxDyn(&[batch, channels, out_h, out_w]), output)?,
                input_dtype,
            )?
        };

        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::MaxPool2d {
                    kernel_size: [kernel_size, kernel_size],
                    stride: [stride, stride],
                    padding: [padding, padding],
                },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }

        Ok(result)
    }

    /// 2-D average pooling.
    ///
    /// Input shape: `[batch, channels, height, width]`
    /// Output shape: `[batch, channels, out_height, out_width]`
    ///
    /// Computes the mean value in each pooling window. Padding positions are
    /// excluded from the count (count_include_pad=false, matching PyTorch default).
    pub fn avg_pool2d(&self, kernel_size: usize, stride: usize, padding: usize) -> Result<Self> {
        self.avg_pool2d_ceil(kernel_size, stride, padding, false)
    }

    /// 2-D average pooling with ceil_mode option.
    pub fn avg_pool2d_ceil(
        &self,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        ceil_mode: bool,
    ) -> Result<Self> {
        let (batch, channels, in_h, in_w) = self.dims4()?;
        let out_h = pool2d_out_len(in_h, kernel_size, padding, stride, ceil_mode)?;
        let out_w = pool2d_out_len(in_w, kernel_size, padding, stride, ceil_mode)?;

        let mut result = if self.device().is_gpu() {
            // Try native GPU dispatch; fall back to CPU round-trip.
            if !ceil_mode {
                if let Some(gpu_result) =
                    gpu_backend_dispatch(|b| b.avg_pool2d(self, kernel_size, stride, padding))
                {
                    gpu_result?
                } else {
                    let original_device = self.device();
                    let cpu_input = self.to_device(&Device::Cpu)?;
                    let r = cpu_input.avg_pool2d_ceil(kernel_size, stride, padding, ceil_mode)?;
                    r.to_device(&original_device)?
                }
            } else {
                // ceil_mode: CPU round-trip (GPU kernel only supports floor mode).
                let original_device = self.device();
                let cpu_input = self.to_device(&Device::Cpu)?;
                let r = cpu_input.avg_pool2d_ceil(kernel_size, stride, padding, ceil_mode)?;
                r.to_device(&original_device)?
            }
        } else {
            let input_dtype = self.dtype;
            let input_c = self.contiguous()?;
            let input_data = input_c.to_f32_array()?;
            let inp = input_data.as_slice().ok_or_else(|| {
                TensorError::InvalidShape("avg_pool2d: input not contiguous".into())
            })?;

            let buf_len =
                super::checked_buffer_len(&[batch, channels, out_h, out_w], "avg_pool2d")?;
            let mut output = vec![0.0f32; buf_len];

            for b in 0..batch {
                for c in 0..channels {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut sum = 0.0f32;
                            let mut count = 0u32;
                            for kh in 0..kernel_size {
                                for kw in 0..kernel_size {
                                    let ih = oh * stride + kh;
                                    let iw = ow * stride + kw;
                                    if ih >= padding
                                        && ih - padding < in_h
                                        && iw >= padding
                                        && iw - padding < in_w
                                    {
                                        let idx = b * channels * in_h * in_w
                                            + c * in_h * in_w
                                            + (ih - padding) * in_w
                                            + (iw - padding);
                                        sum += inp[idx];
                                        count += 1;
                                    }
                                }
                            }
                            let out_idx =
                                b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                            output[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                        }
                    }
                }
            }

            Self::from_f32_result(
                ArrayD::from_shape_vec(IxDyn(&[batch, channels, out_h, out_w]), output)?,
                input_dtype,
            )?
        };

        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::AvgPool2d {
                    kernel_size: [kernel_size, kernel_size],
                    stride: [stride, stride],
                    padding: [padding, padding],
                },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }

        Ok(result)
    }

    /// Adaptive 2-D average pooling.
    ///
    /// Input shape: `[batch, channels, height, width]`
    /// Output shape: `[batch, channels, out_h, out_w]`
    ///
    /// Automatically computes kernel size, stride, and padding to produce
    /// the target output size. Matches PyTorch's `nn.AdaptiveAvgPool2d`.
    pub fn adaptive_avg_pool2d(&self, out_h: usize, out_w: usize) -> Result<Self> {
        let (batch, channels, in_h, in_w) = self.dims4()?;

        if out_h == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "out_h",
                value: 0,
                reason: "must be > 0",
            });
        }
        if out_w == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "out_w",
                value: 0,
                reason: "must be > 0",
            });
        }

        let mut result = if self.device().is_gpu() {
            // Try native GPU dispatch; fall back to CPU round-trip.
            if let Some(gpu_result) =
                gpu_backend_dispatch(|b| b.adaptive_avg_pool2d(self, out_h, out_w))
            {
                gpu_result?
            } else {
                let original_device = self.device();
                let cpu_input = self.to_device(&Device::Cpu)?;
                let r = cpu_input.adaptive_avg_pool2d(out_h, out_w)?;
                r.to_device(&original_device)?
            }
        } else {
            let input_dtype = self.dtype;
            let input_c = self.contiguous()?;
            let input_data = input_c.to_f32_array()?;
            let inp = input_data.as_slice().ok_or_else(|| {
                TensorError::InvalidShape("adaptive_avg_pool2d: input not contiguous".into())
            })?;

            let buf_len =
                super::checked_buffer_len(&[batch, channels, out_h, out_w], "adaptive_avg_pool2d")?;
            let mut output = vec![0.0f32; buf_len];

            for b in 0..batch {
                for c in 0..channels {
                    for oh in 0..out_h {
                        // PyTorch ATen adaptive pooling window: floor/ceil ensures
                        // at least one element per window even when out > in.
                        let start_h = (oh * in_h) / out_h;
                        let end_h = ((oh + 1) * in_h).div_ceil(out_h);
                        for ow in 0..out_w {
                            let start_w = (ow * in_w) / out_w;
                            let end_w = ((ow + 1) * in_w).div_ceil(out_w);

                            let mut sum = 0.0f32;
                            let mut count = 0u32;
                            for ih in start_h..end_h {
                                for iw in start_w..end_w {
                                    let idx = b * channels * in_h * in_w
                                        + c * in_h * in_w
                                        + ih * in_w
                                        + iw;
                                    sum += inp[idx];
                                    count += 1;
                                }
                            }
                            let out_idx =
                                b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                            output[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                        }
                    }
                }
            }

            Self::from_f32_result(
                ArrayD::from_shape_vec(IxDyn(&[batch, channels, out_h, out_w]), output)?,
                input_dtype,
            )?
        };

        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::AdaptiveAvgPool2d {
                    output_size: [out_h, out_w],
                },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
#[path = "pool_tests.rs"]
mod tests;
