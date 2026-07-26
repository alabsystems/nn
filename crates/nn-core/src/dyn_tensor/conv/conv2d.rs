// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! 2-D convolution operation for [`DynTensor`].
//!
//! Extracted from `conv/mod.rs` for 500-line compliance (#1280 Direction 2).

use super::Conv2dParams;
use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::{gpu_backend_dispatch, DynTensor};
use crate::error::{Result, TensorError};
use crate::Device;
use ndarray::{ArrayD, IxDyn};

/// Compute the output length of one spatial dimension of a Conv2d operation.
///
/// Returns `Err` if `kernel_size`, `stride`, or `dilation` is zero,
/// or if the padded input is smaller than the effective kernel.
pub fn conv2d_out_len(
    input_len: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
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
    if dilation == 0 {
        return Err(TensorError::ConvParameterInvalid {
            param: "dilation",
            value: 0,
            reason: "must be > 0",
        });
    }
    let effective_k = (kernel_size - 1)
        .checked_mul(dilation)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| {
            TensorError::InvalidShape(format!(
                "conv2d: effective kernel size overflow (kernel_size={kernel_size}, dilation={dilation})"
            ))
        })?;
    let padded = input_len
        .checked_add(2usize.checked_mul(padding).ok_or_else(|| {
            TensorError::InvalidShape(format!("conv2d: padding overflow (padding={padding})"))
        })?)
        .ok_or_else(|| {
            TensorError::InvalidShape(format!(
                "conv2d: padded input length overflow (input_len={input_len}, padding={padding})"
            ))
        })?;
    if padded < effective_k {
        return Err(TensorError::InvalidShape(format!(
            "conv2d: padded input length {padded} < effective kernel size {effective_k}"
        )));
    }
    Ok((padded - effective_k) / stride + 1)
}

impl DynTensor {
    /// 2-D convolution.
    ///
    /// Input shape: `[batch, in_channels, height, width]`
    /// Kernel shape: `[out_channels, in_channels/groups, kH, kW]`
    /// Output shape: `[batch, out_channels, out_height, out_width]`
    pub fn conv2d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let (batch, in_ch, in_h, in_w) = self.dims4()?;
        let (out_ch, k_in_ch, k_h, k_w) = kernel.dims4()?;

        if stride == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "stride",
                value: 0,
                reason: "must be > 0",
            });
        }
        if dilation == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "dilation",
                value: 0,
                reason: "must be > 0",
            });
        }
        if groups == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "groups",
                value: 0,
                reason: "must be > 0",
            });
        }
        if in_ch % groups != 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "groups",
                value: groups,
                reason: "must divide in_channels",
            });
        }
        if out_ch % groups != 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "groups",
                value: groups,
                reason: "must divide out_channels",
            });
        }
        if k_in_ch != in_ch / groups {
            return Err(TensorError::shape_mismatch(
                vec![out_ch, in_ch / groups, k_h, k_w],
                kernel.dims().to_vec(),
            ));
        }

        let out_h = conv2d_out_len(in_h, k_h, padding, stride, dilation)?;
        let out_w = conv2d_out_len(in_w, k_w, padding, stride, dilation)?;

        // Try native GPU dispatch; fall back to CPU round-trip for GPU tensors.
        let mut result = if self.device().is_gpu() || kernel.device().is_gpu() {
            if let Some(gpu_result) = gpu_backend_dispatch(|b| {
                b.conv2d(self, kernel, None, padding, stride, dilation, groups)
            }) {
                gpu_result?
            } else {
                let original_device = self.device();
                let cpu_input = self.to_device(&Device::Cpu)?;
                let cpu_kernel = kernel.to_device(&Device::Cpu)?;
                let cpu_result =
                    cpu_input.conv2d(&cpu_kernel, padding, stride, dilation, groups)?;
                cpu_result.to_device(&original_device)?
            }
        } else {
            let input_dtype = self.dtype;
            let input_c = self.contiguous()?;
            let kernel_c = kernel.contiguous()?;
            let input_data = input_c.to_f32_array()?;
            let kernel_data = kernel_c.to_f32_array()?;
            let inp = input_data.as_slice().ok_or_else(|| {
                TensorError::InvalidShape("conv2d: input not contiguous after contiguous()".into())
            })?;
            let ker = kernel_data.as_slice().ok_or_else(|| {
                TensorError::InvalidShape("conv2d: kernel not contiguous after contiguous()".into())
            })?;

            let ch_per_group = in_ch / groups;
            let out_ch_per_group = out_ch / groups;
            let buf_len = super::checked_buffer_len(&[batch, out_ch, out_h, out_w], "conv2d")?;
            let mut output = vec![0.0f32; buf_len];

            for b in 0..batch {
                for g in 0..groups {
                    let in_ch_start = g * ch_per_group;
                    let out_ch_start = g * out_ch_per_group;

                    for oc in 0..out_ch_per_group {
                        let abs_oc = out_ch_start + oc;
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let mut sum = 0.0f32;
                                for ic in 0..ch_per_group {
                                    let abs_ic = in_ch_start + ic;
                                    for kh in 0..k_h {
                                        for kw in 0..k_w {
                                            let ih = oh * stride + kh * dilation;
                                            let iw = ow * stride + kw * dilation;
                                            if ih >= padding
                                                && ih - padding < in_h
                                                && iw >= padding
                                                && iw - padding < in_w
                                            {
                                                let input_idx = b * in_ch * in_h * in_w
                                                    + abs_ic * in_h * in_w
                                                    + (ih - padding) * in_w
                                                    + (iw - padding);
                                                let kernel_idx = abs_oc * ch_per_group * k_h * k_w
                                                    + ic * k_h * k_w
                                                    + kh * k_w
                                                    + kw;
                                                sum += inp[input_idx] * ker[kernel_idx];
                                            }
                                        }
                                    }
                                }
                                let out_idx = b * out_ch * out_h * out_w
                                    + abs_oc * out_h * out_w
                                    + oh * out_w
                                    + ow;
                                output[out_idx] = sum;
                            }
                        }
                    }
                }
            }

            Self::from_f32_result(
                ArrayD::from_shape_vec(IxDyn(&[batch, out_ch, out_h, out_w]), output)?,
                input_dtype,
            )?
        };

        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self, kernel])?;
            if let Some(id) = trace::record_op(
                TraceOp::Conv2d {
                    weight: kernel.to_weight_ref()?,
                    bias: None,
                    padding: [padding, padding],
                    stride: [stride, stride],
                    dilation: [dilation, dilation],
                    groups,
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

    /// 2-D convolution with named parameter struct.
    ///
    /// Identical to [`conv2d`](Self::conv2d) but uses [`Conv2dParams`] to prevent
    /// parameter-order mistakes.
    pub fn conv2d_with(&self, kernel: &Self, params: Conv2dParams) -> Result<Self> {
        self.conv2d(
            kernel,
            params.padding,
            params.stride,
            params.dilation,
            params.groups,
        )
    }
}
