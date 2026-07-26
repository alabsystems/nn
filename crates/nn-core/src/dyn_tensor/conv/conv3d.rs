// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! 3-D convolution operation for [`DynTensor`].
//!
//! Follows the Conv2d pattern from `conv2d.rs`. CPU reference implementation
//! using direct nested loops. Needed by dpdf for Qwen3-VL 3D patch embedding.

use super::Conv3dParams;
use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::{gpu_backend_dispatch, DynTensor};
use crate::error::{Result, TensorError};
use crate::Device;
use ndarray::{ArrayD, IxDyn};

/// Compute the output length of one spatial dimension of a Conv3d operation.
///
/// Returns `Err` if `kernel_size`, `stride`, or `dilation` is zero,
/// or if the padded input is smaller than the effective kernel.
pub fn conv3d_out_len(
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
                "conv3d: effective kernel size overflow \
                 (kernel_size={kernel_size}, dilation={dilation})"
            ))
        })?;
    let padded = input_len
        .checked_add(2usize.checked_mul(padding).ok_or_else(|| {
            TensorError::InvalidShape(format!("conv3d: padding overflow (padding={padding})"))
        })?)
        .ok_or_else(|| {
            TensorError::InvalidShape(format!(
                "conv3d: padded input length overflow \
                 (input_len={input_len}, padding={padding})"
            ))
        })?;
    if padded < effective_k {
        return Err(TensorError::InvalidShape(format!(
            "conv3d: padded input length {padded} \
             < effective kernel size {effective_k}"
        )));
    }
    Ok((padded - effective_k) / stride + 1)
}

impl DynTensor {
    /// 3-D convolution.
    ///
    /// Input shape: `[batch, in_channels, depth, height, width]`
    /// Kernel shape: `[out_channels, in_channels/groups, kD, kH, kW]`
    /// Output shape: `[batch, out_channels, out_depth, out_height, out_width]`
    ///
    /// This is needed for 3D patch embeddings (e.g., Qwen3-VL vision encoder).
    pub fn conv3d(
        &self,
        kernel: &Self,
        padding: [usize; 3],
        stride: [usize; 3],
        dilation: [usize; 3],
        groups: usize,
    ) -> Result<Self> {
        let (batch, in_ch, in_d, in_h, in_w) = self.dims5()?;
        let k_dims = kernel.dims();
        if k_dims.len() != 5 {
            return Err(TensorError::RankMismatch {
                expected: 5,
                actual: k_dims.len(),
            });
        }
        let out_ch = k_dims[0];
        let k_in_ch = k_dims[1];
        let k_d = k_dims[2];
        let k_h = k_dims[3];
        let k_w = k_dims[4];

        for (name, val) in [
            ("stride[0]", stride[0]),
            ("stride[1]", stride[1]),
            ("stride[2]", stride[2]),
        ] {
            if val == 0 {
                return Err(TensorError::ConvParameterInvalid {
                    param: name,
                    value: 0,
                    reason: "must be > 0",
                });
            }
        }
        for (name, val) in [
            ("dilation[0]", dilation[0]),
            ("dilation[1]", dilation[1]),
            ("dilation[2]", dilation[2]),
        ] {
            if val == 0 {
                return Err(TensorError::ConvParameterInvalid {
                    param: name,
                    value: 0,
                    reason: "must be > 0",
                });
            }
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
        if !out_ch.is_multiple_of(groups) {
            return Err(TensorError::ConvParameterInvalid {
                param: "groups",
                value: groups,
                reason: "must divide out_channels",
            });
        }
        if k_in_ch != in_ch / groups {
            return Err(TensorError::shape_mismatch(
                vec![out_ch, in_ch / groups, k_d, k_h, k_w],
                kernel.dims().to_vec(),
            ));
        }

        let out_d = conv3d_out_len(in_d, k_d, padding[0], stride[0], dilation[0])?;
        let out_h = conv3d_out_len(in_h, k_h, padding[1], stride[1], dilation[1])?;
        let out_w = conv3d_out_len(in_w, k_w, padding[2], stride[2], dilation[2])?;

        // Try native GPU dispatch; fall back to CPU for GPU tensors.
        if self.device().is_gpu() || kernel.device().is_gpu() {
            if let Some(gpu_result) = gpu_backend_dispatch(|b| {
                b.conv3d(self, kernel, None, padding, stride, dilation, groups)
            }) {
                let mut result = gpu_result?;
                if trace::is_tracing() {
                    let input_ids = Self::trace_input_ids(&[self, kernel])?;
                    if let Some(id) = trace::record_op(
                        TraceOp::Conv3d {
                            weight: kernel.to_weight_ref()?,
                            bias: None,
                            padding,
                            stride,
                            dilation,
                            groups,
                        },
                        &input_ids,
                        result.dims(),
                        result.dtype(),
                    ) {
                        result.set_trace_id(id);
                    }
                }
                return Ok(result);
            }
            // No GPU backend or backend returned None: CPU fallback.
            let original_device = self.device();
            let cpu_input = self.to_device(&Device::Cpu)?;
            let cpu_kernel = kernel.to_device(&Device::Cpu)?;
            let cpu_result = cpu_input.conv3d(&cpu_kernel, padding, stride, dilation, groups)?;
            return cpu_result.to_device(&original_device);
        }

        let input_dtype = self.dtype;
        let input_c = self.contiguous()?;
        let kernel_c = kernel.contiguous()?;
        let input_data = input_c.to_f32_array()?;
        let kernel_data = kernel_c.to_f32_array()?;
        let inp = input_data.as_slice().ok_or_else(|| {
            TensorError::InvalidShape("conv3d: input not contiguous after contiguous()".into())
        })?;
        let ker = kernel_data.as_slice().ok_or_else(|| {
            TensorError::InvalidShape("conv3d: kernel not contiguous after contiguous()".into())
        })?;

        let ch_per_group = in_ch / groups;
        let out_ch_per_group = out_ch / groups;
        let buf_len = super::checked_buffer_len(&[batch, out_ch, out_d, out_h, out_w], "conv3d")?;
        let mut output = vec![0.0f32; buf_len];

        let pad_d = padding[0];
        let pad_h = padding[1];
        let pad_w = padding[2];
        let str_d = stride[0];
        let str_h = stride[1];
        let str_w = stride[2];
        let dil_d = dilation[0];
        let dil_h = dilation[1];
        let dil_w = dilation[2];

        for b in 0..batch {
            for g in 0..groups {
                let in_ch_start = g * ch_per_group;
                let out_ch_start = g * out_ch_per_group;

                for oc in 0..out_ch_per_group {
                    let abs_oc = out_ch_start + oc;
                    for od in 0..out_d {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let mut sum = 0.0f32;
                                for ic in 0..ch_per_group {
                                    let abs_ic = in_ch_start + ic;
                                    for kd in 0..k_d {
                                        let id = od * str_d + kd * dil_d;
                                        if id < pad_d || id - pad_d >= in_d {
                                            continue;
                                        }
                                        let id_real = id - pad_d;
                                        for kh in 0..k_h {
                                            let ih = oh * str_h + kh * dil_h;
                                            if ih < pad_h || ih - pad_h >= in_h {
                                                continue;
                                            }
                                            let ih_real = ih - pad_h;
                                            for kw in 0..k_w {
                                                let iw = ow * str_w + kw * dil_w;
                                                if iw < pad_w || iw - pad_w >= in_w {
                                                    continue;
                                                }
                                                let iw_real = iw - pad_w;
                                                let input_idx = b * in_ch * in_d * in_h * in_w
                                                    + abs_ic * in_d * in_h * in_w
                                                    + id_real * in_h * in_w
                                                    + ih_real * in_w
                                                    + iw_real;
                                                let kernel_idx =
                                                    abs_oc * ch_per_group * k_d * k_h * k_w
                                                        + ic * k_d * k_h * k_w
                                                        + kd * k_h * k_w
                                                        + kh * k_w
                                                        + kw;
                                                sum += inp[input_idx] * ker[kernel_idx];
                                            }
                                        }
                                    }
                                }
                                let out_idx = b * out_ch * out_d * out_h * out_w
                                    + abs_oc * out_d * out_h * out_w
                                    + od * out_h * out_w
                                    + oh * out_w
                                    + ow;
                                output[out_idx] = sum;
                            }
                        }
                    }
                }
            }
        }

        let mut result = Self::from_f32_result(
            ArrayD::from_shape_vec(IxDyn(&[batch, out_ch, out_d, out_h, out_w]), output)?,
            input_dtype,
        )?;

        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self, kernel])?;
            if let Some(id) = trace::record_op(
                TraceOp::Conv3d {
                    weight: kernel.to_weight_ref()?,
                    bias: None,
                    padding,
                    stride,
                    dilation,
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

    /// 3-D convolution with named parameter struct.
    ///
    /// Identical to [`conv3d`](Self::conv3d) but uses [`Conv3dParams`] to
    /// prevent parameter-order mistakes.
    pub fn conv3d_with(&self, kernel: &Self, params: Conv3dParams) -> Result<Self> {
        self.conv3d(
            kernel,
            params.padding,
            params.stride,
            params.dilation,
            params.groups,
        )
    }
}
