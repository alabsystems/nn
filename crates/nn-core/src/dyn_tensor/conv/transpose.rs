// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ConvTranspose1d (deconvolution) for [`DynTensor`].
//!
//! Extracted from `conv/mod.rs` for file-size compliance.

use super::ConvTranspose1dParams;
use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::{gpu_backend_dispatch, DynTensor};
use crate::error::{Result, TensorError};
use crate::Device;
use ndarray::{ArrayD, IxDyn};

/// Compute the output length of a ConvTranspose1d operation.
///
/// Returns `Err` if `2 * padding` exceeds the positive terms, which would
/// cause a usize underflow.
pub fn conv_transpose1d_out_len(
    input_len: usize,
    kernel_size: usize,
    padding: usize,
    output_padding: usize,
    stride: usize,
    dilation: usize,
) -> Result<usize> {
    if input_len == 0 {
        return Err(TensorError::ConvParameterInvalid {
            param: "input_len",
            value: 0,
            reason: "must be > 0",
        });
    }
    if kernel_size == 0 {
        return Err(TensorError::ConvParameterInvalid {
            param: "kernel_size",
            value: 0,
            reason: "must be > 0",
        });
    }
    if stride > 0 && output_padding >= stride {
        return Err(TensorError::ConvParameterInvalid {
            param: "output_padding",
            value: output_padding,
            reason: "must be < stride",
        });
    }
    let positive = (input_len - 1)
        .checked_mul(stride)
        .and_then(|v| {
            dilation
                .checked_mul(kernel_size - 1)
                .and_then(|dk| v.checked_add(dk))
        })
        .and_then(|v| v.checked_add(output_padding))
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| {
            TensorError::InvalidShape(format!(
                "conv_transpose1d: output length overflow (input_len={input_len}, \
                 kernel_size={kernel_size}, stride={stride}, dilation={dilation}, \
                 output_padding={output_padding})"
            ))
        })?;
    let negative = 2usize.checked_mul(padding).ok_or_else(|| {
        TensorError::InvalidShape(format!(
            "conv_transpose1d: padding overflow (padding={padding})"
        ))
    })?;
    if negative >= positive {
        return Err(TensorError::InvalidShape(format!(
            "conv_transpose1d: 2*padding ({negative}) meets or exceeds output length \
             terms ({positive}), would produce zero or negative output length"
        )));
    }
    Ok(positive - negative)
}

impl DynTensor {
    /// 1-D transposed convolution (deconvolution).
    ///
    /// Input shape: `[batch, in_channels, length]`
    /// Kernel shape: `[in_channels, out_channels/groups, kernel_size]`
    /// Output shape: `[batch, out_channels, out_length]`
    pub fn conv_transpose1d(
        &self,
        kernel: &Self,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        self.conv_transpose1d_validate(kernel, padding, output_padding, stride, dilation, groups)?;
        trace::traced_forward(
            &[self],
            || {
                Ok(TraceOp::ConvTranspose1d {
                    weight: kernel.to_weight_ref()?,
                    bias: None,
                    padding,
                    output_padding,
                    stride,
                    dilation,
                    groups,
                })
            },
            || {
                self.conv_transpose1d_compute(
                    kernel,
                    padding,
                    output_padding,
                    stride,
                    dilation,
                    groups,
                )
            },
        )
    }

    /// Validate conv_transpose1d parameters. Returns the validated dimensions.
    fn conv_transpose1d_validate(
        &self,
        kernel: &Self,
        _padding: usize,
        _output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<()> {
        let (_, in_ch, _) = self.dims3()?;
        let (k_in_ch, k_out_ch_per_g, k_size) = kernel.dims3()?;
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
        if k_in_ch != in_ch {
            return Err(TensorError::shape_mismatch(
                vec![in_ch, k_out_ch_per_g, k_size],
                kernel.dims().to_vec(),
            ));
        }
        if in_ch % groups != 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "groups",
                value: groups,
                reason: "must divide in_channels",
            });
        }
        Ok(())
    }

    /// Compute body for conv_transpose1d. Called within trace suppression.
    fn conv_transpose1d_compute(
        &self,
        kernel: &Self,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let (batch, in_ch, in_len) = self.dims3()?;
        let (_, k_out_ch_per_g, k_size) = kernel.dims3()?;
        let out_ch = k_out_ch_per_g * groups;
        let out_len =
            conv_transpose1d_out_len(in_len, k_size, padding, output_padding, stride, dilation)?;

        // Try native GPU dispatch. Metal supports all parameter combinations
        // including output_padding > 0 (#1957). Falls back to CPU round-trip
        // only if the backend returns None (currently no known configs do).
        if self.device().is_gpu() || kernel.device().is_gpu() {
            if let Some(result) = gpu_backend_dispatch(|b| {
                b.conv_transpose1d(
                    self,
                    kernel,
                    None,
                    padding,
                    output_padding,
                    stride,
                    dilation,
                    groups,
                )
            }) {
                return result;
            }
            // CPU round-trip fallback when backend returns None.
            let original_device = self.device();
            let cpu_input = self.to_device(&Device::Cpu)?;
            let cpu_kernel = kernel.to_device(&Device::Cpu)?;
            let result = cpu_input.conv_transpose1d_compute(
                &cpu_kernel,
                padding,
                output_padding,
                stride,
                dilation,
                groups,
            )?;
            return result.to_device(&original_device);
        }

        let input_dtype = self.dtype;
        let input_c = self.contiguous()?;
        let kernel_c = kernel.contiguous()?;
        let input_data = input_c.to_f32_array()?;
        let kernel_data = kernel_c.to_f32_array()?;
        let input_slice = input_data.as_slice().ok_or_else(|| {
            TensorError::InvalidShape(
                "conv_transpose1d: input not contiguous after contiguous()".into(),
            )
        })?;
        let kernel_slice = kernel_data.as_slice().ok_or_else(|| {
            TensorError::InvalidShape(
                "conv_transpose1d: kernel not contiguous after contiguous()".into(),
            )
        })?;

        let in_ch_per_group = in_ch / groups;
        let buf_len = super::checked_buffer_len(&[batch, out_ch, out_len], "conv_transpose1d")?;
        let mut output = vec![0.0f32; buf_len];

        for b in 0..batch {
            for g in 0..groups {
                let in_ch_start = g * in_ch_per_group;
                let out_ch_start = g * k_out_ch_per_g;

                for ic in 0..in_ch_per_group {
                    let abs_ic = in_ch_start + ic;
                    for il in 0..in_len {
                        let input_val = input_slice[b * in_ch * in_len + abs_ic * in_len + il];
                        for oc in 0..k_out_ch_per_g {
                            let abs_oc = out_ch_start + oc;
                            for kl in 0..k_size {
                                let ol_raw = il * stride + kl * dilation;
                                if ol_raw >= padding && ol_raw - padding < out_len {
                                    let ol = ol_raw - padding;
                                    let kernel_idx =
                                        abs_ic * k_out_ch_per_g * k_size + oc * k_size + kl;
                                    let out_idx = b * out_ch * out_len + abs_oc * out_len + ol;
                                    output[out_idx] += input_val * kernel_slice[kernel_idx];
                                }
                            }
                        }
                    }
                }
            }
        }

        Self::from_f32_result(
            ArrayD::from_shape_vec(IxDyn(&[batch, out_ch, out_len]), output)?,
            input_dtype,
        )
    }

    /// 1-D transposed convolution with named parameter struct.
    ///
    /// Identical to [`conv_transpose1d`](Self::conv_transpose1d) but uses
    /// [`ConvTranspose1dParams`] to prevent parameter-order mistakes.
    pub fn conv_transpose1d_with(
        &self,
        kernel: &Self,
        params: ConvTranspose1dParams,
    ) -> Result<Self> {
        self.conv_transpose1d(
            kernel,
            params.padding,
            params.output_padding,
            params.stride,
            params.dilation,
            params.groups,
        )
    }
}
