// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tracked tensor pool operations: max_pool1d, max_pool2d, avg_pool2d, adaptive_avg_pool2d.
//!
//! Extracted from `tracked_composite_ops.rs` for 500-line compliance.

use std::sync::Arc;

use super::TrackedTensor;
use crate::error::Result;
use crate::op::Op;

impl TrackedTensor {
    /// 1-D max pooling with argmax indices for backward pass.
    ///
    /// Computes both the pooled output and the argmax flat indices in a single pass.
    /// Input shape: `[batch, channels, length]`
    pub fn max_pool1d(
        self: &Arc<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Arc<Self>> {
        use nn_core::dyn_tensor::DynTensor;
        use nn_core::Device;

        let input = self.tensor();
        let dims = input.dims();
        if dims.len() != 3 {
            return Err(crate::error::AutodiffError::WrongInputRank {
                op: "max_pool1d",
                expected: 3,
                actual: dims.len(),
            });
        }
        let (batch, channels, in_len) = (dims[0], dims[1], dims[2]);

        if kernel_size == 0 || stride == 0 {
            return Err(crate::error::AutodiffError::InvalidConfig {
                op: "max_pool1d",
                reason: format!("kernel_size ({kernel_size}) and stride ({stride}) must be > 0"),
            });
        }
        let padded = in_len + 2 * padding;
        if padded < kernel_size {
            return Err(crate::error::AutodiffError::InvalidConfig {
                op: "max_pool1d",
                reason: format!("padded input ({padded}) smaller than kernel ({kernel_size})"),
            });
        }
        let out_len = (padded - kernel_size) / stride + 1;

        // GPU→CPU round-trip if needed
        let device = input.device();
        let cpu_input = if device.is_gpu() {
            input.to_device(&Device::Cpu)?
        } else {
            input.clone()
        };
        let input_c = cpu_input.contiguous()?;
        let input_data = input_c.to_f32_array()?;
        let inp = input_data
            .as_slice()
            .ok_or(crate::error::AutodiffError::NotContiguous { op: "max_pool1d" })?;

        let total = batch * channels * out_len;
        let mut output = vec![f32::NEG_INFINITY; total];
        let mut indices = vec![0u32; total];

        for b in 0..batch {
            for c in 0..channels {
                for ol in 0..out_len {
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx: u32 = 0;
                    for k in 0..kernel_size {
                        let il = ol * stride + k;
                        if il >= padding && il - padding < in_len {
                            let idx = b * channels * in_len + c * in_len + (il - padding);
                            let val = inp[idx];
                            if val > max_val {
                                max_val = val;
                                max_idx = u32::try_from(idx).map_err(|_| {
                                    crate::error::AutodiffError::IndexOverflow {
                                        op: "max_pool1d",
                                        index: idx,
                                        max: u32::MAX,
                                    }
                                })?;
                            }
                        }
                    }
                    let out_idx = b * channels * out_len + c * out_len + ol;
                    output[out_idx] = max_val;
                    indices[out_idx] = max_idx;
                }
            }
        }

        let out_shape = &[batch, channels, out_len];
        let mut data = DynTensor::from_vec(output, out_shape, &Device::Cpu)?;
        if device.is_gpu() {
            data = data.to_device(&device)?;
        }
        let mut indices_t = DynTensor::from_vec_u32(indices, out_shape, &Device::Cpu)?;
        if device.is_gpu() {
            indices_t = indices_t.to_device(&device)?;
        }

        Ok(Arc::new(Self::from_op(
            data,
            Op::MaxPool1d {
                input: Arc::clone(self),
                indices: indices_t,
                kernel_size,
                stride,
                padding,
            },
        )))
    }

    /// 2-D max pooling with argmax indices for backward pass.
    ///
    /// Computes both the pooled output and the argmax flat indices in a single pass.
    /// Input shape: `[batch, channels, height, width]`
    pub fn max_pool2d(
        self: &Arc<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Arc<Self>> {
        use nn_core::dyn_tensor::DynTensor;
        use nn_core::Device;

        let input = self.tensor();
        let dims = input.dims();
        if dims.len() != 4 {
            return Err(crate::error::AutodiffError::WrongInputRank {
                op: "max_pool2d",
                expected: 4,
                actual: dims.len(),
            });
        }
        let (batch, channels, in_h, in_w) = (dims[0], dims[1], dims[2], dims[3]);

        // Compute output dimensions (same as pool2d_out_len in nn-core)
        if kernel_size == 0 || stride == 0 {
            return Err(crate::error::AutodiffError::InvalidConfig {
                op: "max_pool2d",
                reason: format!("kernel_size ({kernel_size}) and stride ({stride}) must be > 0"),
            });
        }
        let padded_h = in_h + 2 * padding;
        let padded_w = in_w + 2 * padding;
        if padded_h < kernel_size || padded_w < kernel_size {
            return Err(crate::error::AutodiffError::InvalidConfig {
                op: "max_pool2d",
                reason: format!(
                    "padded input ({padded_h}x{padded_w}) smaller than kernel ({kernel_size})"
                ),
            });
        }
        let out_h = (padded_h - kernel_size) / stride + 1;
        let out_w = (padded_w - kernel_size) / stride + 1;

        // GPU→CPU round-trip if needed
        let device = input.device();
        let cpu_input = if device.is_gpu() {
            input.to_device(&Device::Cpu)?
        } else {
            input.clone()
        };
        let input_c = cpu_input.contiguous()?;
        let input_data = input_c.to_f32_array()?;
        let inp = input_data
            .as_slice()
            .ok_or(crate::error::AutodiffError::NotContiguous { op: "max_pool2d" })?;

        let out_len = batch * channels * out_h * out_w;
        let mut output = vec![f32::NEG_INFINITY; out_len];
        let mut indices = vec![0u32; out_len];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx: u32 = 0;
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
                                        max_idx = u32::try_from(idx).map_err(|_| {
                                            crate::error::AutodiffError::IndexOverflow {
                                                op: "max_pool2d",
                                                index: idx,
                                                max: u32::MAX,
                                            }
                                        })?;
                                    }
                                }
                            }
                        }
                        let out_idx =
                            b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        output[out_idx] = max_val;
                        indices[out_idx] = max_idx;
                    }
                }
            }
        }

        let out_shape = &[batch, channels, out_h, out_w];
        let mut data = DynTensor::from_vec(output, out_shape, &Device::Cpu)?;
        if device.is_gpu() {
            data = data.to_device(&device)?;
        }
        let mut indices_t = DynTensor::from_vec_u32(indices, out_shape, &Device::Cpu)?;
        if device.is_gpu() {
            indices_t = indices_t.to_device(&device)?;
        }

        Ok(Arc::new(Self::from_op(
            data,
            Op::MaxPool2d {
                input: Arc::clone(self),
                indices: indices_t,
                kernel_size,
                stride,
                padding,
            },
        )))
    }

    /// 2-D average pooling.
    ///
    /// Input shape: `[batch, channels, height, width]`
    pub fn avg_pool2d(
        self: &Arc<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Arc<Self>> {
        let data = self.tensor().avg_pool2d(kernel_size, stride, padding)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::AvgPool2d {
                input: Arc::clone(self),
                kernel_size,
                stride,
                padding,
            },
        )))
    }

    /// Adaptive 2-D average pooling.
    ///
    /// Input shape: `[batch, channels, height, width]`
    /// Produces output of shape `[batch, channels, output_h, output_w]`.
    pub fn adaptive_avg_pool2d(
        self: &Arc<Self>,
        output_h: usize,
        output_w: usize,
    ) -> Result<Arc<Self>> {
        let data = self.tensor().adaptive_avg_pool2d(output_h, output_w)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::AdaptiveAvgPool2d {
                input: Arc::clone(self),
                output_h,
                output_w,
            },
        )))
    }
}
