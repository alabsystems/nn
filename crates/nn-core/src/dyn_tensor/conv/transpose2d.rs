// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ConvTranspose2d (2-D transposed convolution / deconvolution) for [`DynTensor`].

use super::params::ConvTranspose2dParams;
use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::DynTensor;
use crate::error::{Result, TensorError};
use crate::Device;
use ndarray::{ArrayD, IxDyn};

/// Compute the output length of one spatial dimension of a ConvTranspose2d operation.
///
/// Formula: `(input_len - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1`
pub fn conv_transpose2d_out_len(
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
                "conv_transpose2d: output length overflow (input_len={input_len}, \
                 kernel_size={kernel_size}, stride={stride}, dilation={dilation}, \
                 output_padding={output_padding})"
            ))
        })?;
    let negative = 2usize.checked_mul(padding).ok_or_else(|| {
        TensorError::InvalidShape(format!(
            "conv_transpose2d: padding overflow (padding={padding})"
        ))
    })?;
    if negative >= positive {
        return Err(TensorError::InvalidShape(format!(
            "conv_transpose2d: 2*padding ({negative}) meets or exceeds output length \
             terms ({positive}), would produce zero or negative output length"
        )));
    }
    Ok(positive - negative)
}

impl DynTensor {
    /// 2-D transposed convolution (deconvolution).
    ///
    /// Spatial parameters are `[height, width]` to support non-square
    /// transposed convolutions (e.g., stride=[2,1]).
    pub fn conv_transpose2d(
        &self,
        kernel: &Self,
        padding: [usize; 2],
        output_padding: [usize; 2],
        stride: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Result<Self> {
        self.conv_transpose2d_validate(kernel, stride, dilation, groups)?;
        trace::traced_forward(
            &[self],
            || {
                Ok(TraceOp::ConvTranspose2d {
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
                self.conv_transpose2d_compute(
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

    fn conv_transpose2d_validate(
        &self,
        kernel: &Self,
        stride: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Result<()> {
        let (_, in_ch, _, _) = self.dims4()?;
        let (k_in_ch, k_out_ch_per_g, k_h, k_w) = kernel.dims4()?;
        if stride[0] == 0 || stride[1] == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "stride",
                value: 0,
                reason: "must be > 0",
            });
        }
        if dilation[0] == 0 || dilation[1] == 0 {
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
                vec![in_ch, k_out_ch_per_g, k_h, k_w],
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

    fn conv_transpose2d_compute(
        &self,
        kernel: &Self,
        padding: [usize; 2],
        output_padding: [usize; 2],
        stride: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Result<Self> {
        let (batch, in_ch, in_h, in_w) = self.dims4()?;
        let (_, k_out_ch_per_g, k_h, k_w) = kernel.dims4()?;
        let out_ch = k_out_ch_per_g * groups;
        let out_h = conv_transpose2d_out_len(
            in_h,
            k_h,
            padding[0],
            output_padding[0],
            stride[0],
            dilation[0],
        )?;
        let out_w = conv_transpose2d_out_len(
            in_w,
            k_w,
            padding[1],
            output_padding[1],
            stride[1],
            dilation[1],
        )?;

        if self.device().is_gpu() || kernel.device().is_gpu() {
            let original_device = self.device();
            let cpu_input = self.to_device(&Device::Cpu)?;
            let cpu_kernel = kernel.to_device(&Device::Cpu)?;
            let result = cpu_input.conv_transpose2d_compute(
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
                "conv_transpose2d: input not contiguous after contiguous()".into(),
            )
        })?;
        let kernel_slice = kernel_data.as_slice().ok_or_else(|| {
            TensorError::InvalidShape(
                "conv_transpose2d: kernel not contiguous after contiguous()".into(),
            )
        })?;

        let in_ch_per_group = in_ch / groups;
        let buf_len =
            super::checked_buffer_len(&[batch, out_ch, out_h, out_w], "conv_transpose2d")?;
        let mut output = vec![0.0f32; buf_len];

        for b in 0..batch {
            for g in 0..groups {
                let in_ch_start = g * in_ch_per_group;
                let out_ch_start = g * k_out_ch_per_g;
                for ic in 0..in_ch_per_group {
                    let abs_ic = in_ch_start + ic;
                    for ih in 0..in_h {
                        for iw in 0..in_w {
                            let input_val = input_slice
                                [b * in_ch * in_h * in_w + abs_ic * in_h * in_w + ih * in_w + iw];
                            for oc in 0..k_out_ch_per_g {
                                let abs_oc = out_ch_start + oc;
                                for kh in 0..k_h {
                                    let oh_raw = ih * stride[0] + kh * dilation[0];
                                    if oh_raw < padding[0] || oh_raw - padding[0] >= out_h {
                                        continue;
                                    }
                                    let oh = oh_raw - padding[0];
                                    for kw in 0..k_w {
                                        let ow_raw = iw * stride[1] + kw * dilation[1];
                                        if ow_raw < padding[1] || ow_raw - padding[1] >= out_w {
                                            continue;
                                        }
                                        let ow = ow_raw - padding[1];
                                        let kernel_idx = abs_ic * k_out_ch_per_g * k_h * k_w
                                            + oc * k_h * k_w
                                            + kh * k_w
                                            + kw;
                                        let out_idx = b * out_ch * out_h * out_w
                                            + abs_oc * out_h * out_w
                                            + oh * out_w
                                            + ow;
                                        output[out_idx] += input_val * kernel_slice[kernel_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Self::from_f32_result(
            ArrayD::from_shape_vec(IxDyn(&[batch, out_ch, out_h, out_w]), output)?,
            input_dtype,
        )
    }

    /// 2-D transposed convolution with named parameter struct.
    pub fn conv_transpose2d_with(
        &self,
        kernel: &Self,
        params: ConvTranspose2dParams,
    ) -> Result<Self> {
        self.conv_transpose2d(
            kernel,
            params.padding,
            params.output_padding,
            params.stride,
            params.dilation,
            params.groups,
        )
    }
}
