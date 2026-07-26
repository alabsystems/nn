// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv1d, Conv2d, Conv3d, pooling, and padding operations for [`DynTensor`].
//!
//! CPU implementation uses im2col (unfold + matmul) for Conv1d.
//! Conv2d lives in `conv2d.rs`. Conv3d in `conv3d.rs`.
//! ConvTranspose1d in `transpose.rs`. Pool2d in `pool.rs`.
//! Padding ops live in `padding.rs`.

use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::{gpu_backend_dispatch, DynTensor};
use crate::error::{Result, TensorError};
use crate::Device;
use ndarray::{ArrayD, IxDyn};

mod conv2d;
pub(crate) mod conv3d;
mod im2col;
mod padding;
pub(crate) mod params;
pub(crate) mod pool;
mod transpose;
mod transpose2d;

pub use conv2d::conv2d_out_len;
pub use conv3d::conv3d_out_len;
pub use params::{
    Conv1dParams, Conv2dParams, Conv3dParams, ConvTranspose1dParams, ConvTranspose2dParams,
};
pub use transpose::conv_transpose1d_out_len;
pub use transpose2d::conv_transpose2d_out_len;

/// Checked multiply of multiple `usize` factors, returning `TensorError::InvalidShape` on overflow.
pub(crate) fn checked_buffer_len(factors: &[usize], context: &str) -> Result<usize> {
    factors.iter().try_fold(1usize, |acc, &f| {
        acc.checked_mul(f).ok_or_else(|| {
            TensorError::InvalidShape(format!(
                "{context}: buffer size overflow computing {factors:?}"
            ))
        })
    })
}

/// Compute the output length of a Conv1d operation.
///
/// Returns `Err` if `kernel_size`, `stride`, or `dilation` is zero,
/// or if the padded input is smaller than the effective kernel.
pub fn conv1d_out_len(
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
                "conv1d: effective kernel size overflow (kernel_size={kernel_size}, dilation={dilation})"
            ))
        })?;
    let padded = input_len
        .checked_add(2usize.checked_mul(padding).ok_or_else(|| {
            TensorError::InvalidShape(format!("conv1d: padding overflow (padding={padding})"))
        })?)
        .ok_or_else(|| {
            TensorError::InvalidShape(format!(
                "conv1d: padded input length overflow (input_len={input_len}, padding={padding})"
            ))
        })?;
    if padded < effective_k {
        return Err(TensorError::InvalidShape(format!(
            "conv1d: padded input length {padded} < effective kernel size {effective_k}"
        )));
    }
    Ok((padded - effective_k) / stride + 1)
}

impl DynTensor {
    /// 1-D convolution.
    ///
    /// Input shape: `[batch, in_channels, length]`
    /// Kernel shape: `[out_channels, in_channels/groups, kernel_size]`
    /// Output shape: `[batch, out_channels, out_length]`
    pub fn conv1d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let (batch, in_ch, in_len) = self.dims3()?;
        let (out_ch, k_in_ch, k_size) = kernel.dims3()?;

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
                vec![out_ch, in_ch / groups, k_size],
                kernel.dims().to_vec(),
            ));
        }

        let out_len = conv1d_out_len(in_len, k_size, padding, stride, dilation)?;

        // Try native GPU dispatch; fall back to CPU round-trip for GPU tensors.
        let mut result = if self.device().is_gpu() || kernel.device().is_gpu() {
            if let Some(gpu_result) = gpu_backend_dispatch(|b| {
                b.conv1d(self, kernel, None, padding, stride, dilation, groups)
            }) {
                gpu_result?
            } else {
                let original_device = self.device();
                let cpu_input = self.to_device(&Device::Cpu)?;
                let cpu_kernel = kernel.to_device(&Device::Cpu)?;
                let cpu_result =
                    cpu_input.conv1d(&cpu_kernel, padding, stride, dilation, groups)?;
                cpu_result.to_device(&original_device)?
            }
        } else {
            let input_dtype = self.dtype;
            let input_c = self.contiguous()?;
            let kernel_c = kernel.contiguous()?;
            let input_data = input_c.to_f32_array()?;
            let kernel_data = kernel_c.to_f32_array()?;
            let inp = input_data.as_slice().ok_or_else(|| {
                TensorError::InvalidShape("conv1d: input not contiguous after contiguous()".into())
            })?;
            let ker = kernel_data.as_slice().ok_or_else(|| {
                TensorError::InvalidShape("conv1d: kernel not contiguous after contiguous()".into())
            })?;

            let ch_per_group = in_ch / groups;
            let out_ch_per_group = out_ch / groups;
            let buf_len = checked_buffer_len(&[batch, out_ch, out_len], "conv1d")?;
            let mut output = vec![0.0f32; buf_len];

            for b in 0..batch {
                for g in 0..groups {
                    let in_ch_start = g * ch_per_group;
                    let out_ch_start = g * out_ch_per_group;

                    for oc in 0..out_ch_per_group {
                        let abs_oc = out_ch_start + oc;
                        for ol in 0..out_len {
                            let mut sum = 0.0f32;
                            for ic in 0..ch_per_group {
                                let abs_ic = in_ch_start + ic;
                                for kl in 0..k_size {
                                    let il = ol * stride + kl * dilation;
                                    if il >= padding && il - padding < in_len {
                                        let input_idx =
                                            b * in_ch * in_len + abs_ic * in_len + (il - padding);
                                        let kernel_idx =
                                            abs_oc * ch_per_group * k_size + ic * k_size + kl;
                                        sum += inp[input_idx] * ker[kernel_idx];
                                    }
                                }
                            }
                            let out_idx = b * out_ch * out_len + abs_oc * out_len + ol;
                            output[out_idx] = sum;
                        }
                    }
                }
            }

            Self::from_f32_result(
                ArrayD::from_shape_vec(IxDyn(&[batch, out_ch, out_len]), output)?,
                input_dtype,
            )?
        };

        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self, kernel])?;
            if let Some(id) = trace::record_op(
                TraceOp::Conv1d {
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

    /// 1-D convolution with named parameter struct.
    ///
    /// Identical to [`conv1d`](Self::conv1d) but uses [`Conv1dParams`] to prevent
    /// parameter-order mistakes.
    pub fn conv1d_with(&self, kernel: &Self, params: Conv1dParams) -> Result<Self> {
        self.conv1d(
            kernel,
            params.padding,
            params.stride,
            params.dilation,
            params.groups,
        )
    }
}

#[cfg(kani)]
#[path = "kani_conv3d_proofs.rs"]
mod kani_conv3d_proofs;
#[cfg(kani)]
#[path = "kani_dpdf_conv_transpose2d_proofs.rs"]
mod kani_dpdf_conv_transpose2d_proofs;
#[cfg(kani)]
#[path = "kani_params.rs"]
mod kani_params;
#[cfg(kani)]
#[path = "kani_pool.rs"]
mod kani_pool_conv;

#[cfg(test)]
#[path = "conv3d_tests.rs"]
mod conv3d_tests;
#[cfg(test)]
mod tests;
