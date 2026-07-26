// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU-native convolution implementations for [`MetalDynBackend`].
//!
//! Extracted from `dyn_tensor_metal_shape_ops.rs` to stay under the 500-line limit.
//!
//! - `gpu_conv1d`: 1-D convolution (forward pass)
//! - `gpu_conv2d`: 2-D convolution (forward pass)
//! - `gpu_conv_transpose1d`: 1-D transposed convolution (deconvolution)
//!
//! `gpu_index_select` was extracted to `dyn_tensor_metal_select_ops.rs` (not a conv op).
//!
//! Part of #1101 (GPU op elimination).

use nn_core::dyn_tensor::DynTensor;
use nn_core::{Result, TensorError};

use nn_dsl::TensorBlockBuilder;

use super::MetalTensorData;

impl super::MetalDynBackend {
    /// GPU-native 1-D convolution.
    ///
    /// Input shape: `[batch, in_channels, length]`
    /// Kernel shape: `[out_channels, in_channels/groups, kernel_size]`
    /// Output shape: `[batch, out_channels, out_length]`
    pub(super) fn gpu_conv1d(
        input: &DynTensor,
        kernel: &DynTensor,
        bias: Option<&DynTensor>,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<DynTensor> {
        Self::validate_same_float_dtype(input, kernel, "gpu_conv1d")?;
        if let Some(b) = bias {
            Self::validate_same_float_dtype(input, b, "gpu_conv1d")?;
        }

        // Defense-in-depth: validate conv parameters before arithmetic.
        // The CPU path validates via `conv1d_out_len` in nn-core, but the GPU
        // path bypasses that and computes output length inline.
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

        let in_shape = input.dims();
        let k_shape = kernel.dims();

        if in_shape.len() != 3 || k_shape.len() != 3 {
            return Err(TensorError::InvalidShape(format!(
                "gpu_conv1d: requires 3D tensors, got input={in_shape:?} kernel={k_shape:?}"
            )));
        }

        let input_data = input.gpu_data::<MetalTensorData>()?;
        let kernel_data = kernel.gpu_data::<MetalTensorData>()?;

        let batch = in_shape[0];
        let out_ch = k_shape[0];
        let in_len = in_shape[2];
        let k_size = k_shape[2];

        // Compute conv1d output length: (L + 2*P - D*(K-1) - 1) / S + 1
        let effective_k = (k_size - 1) * dilation + 1;
        let padded = in_len + 2 * padding;
        if padded < effective_k {
            return Err(TensorError::InvalidShape(format!(
                "gpu_conv1d: padded input length {padded} < effective kernel size {effective_k}"
            )));
        }
        let out_len = (padded - effective_k) / stride + 1;
        let out_shape = vec![batch, out_ch, out_len];

        // Route to direct sliding-window Conv1d for Kokoro K=3 shapes (#4264).
        // Avoids im2col buffer allocation + blit, saving 1 dispatch per Conv1d.
        if Self::should_use_direct_conv1d_k3(
            in_shape, k_shape, out_len, groups, stride, dilation, input.dtype(),
        ) {
            return Self::gpu_direct_conv1d_k3(input, kernel, bias, padding, &out_shape);
        }

        // Route to im2col + simdgroup GEMM for large standard convolutions (#3002).
        if Self::should_use_conv1d_gemm(in_shape, k_shape, out_len, groups, input.dtype()) {
            return Self::gpu_conv1d_gemm(
                input, kernel, bias, padding, stride, dilation, &out_shape,
            );
        }

        let has_bias = u64::from(bias.is_some());
        let bias_dims: Option<Vec<usize>> = bias.map(|b| b.dims().to_vec());
        let def = crate::kernel_def_cache::get_or_build(
            "conv1d",
            &[in_shape, k_shape],
            &[
                stride as u64,
                padding as u64,
                dilation as u64,
                groups as u64,
                has_bias,
            ],
            input.dtype(),
            || {
                let mut b = TensorBlockBuilder::new("dyn_conv1d");
                let in_node = b.add_input("input", in_shape);
                let k_node = b.add_input("kernel", k_shape);
                let bias_node = bias_dims.as_ref().map(|d| b.add_input("bias", d));
                let out = b.add_conv1d_full(
                    in_node, k_node, bias_node, stride, padding, dilation, groups, &out_shape,
                );
                crate::build_kernel(b, out)
            },
        )?;

        let mut dispatch_inputs: Vec<(&str, crate::gpu_slice::GpuSlice)> = vec![
            ("input", input_data.as_gpu_slice()),
            ("kernel", kernel_data.as_gpu_slice()),
        ];

        if let Some(bias_t) = bias {
            let bias_data = bias_t.gpu_data::<MetalTensorData>()?;
            dispatch_inputs.push(("bias", bias_data.as_gpu_slice()));
        }

        Self::dispatch_def(&def, &dispatch_inputs, &out_shape, input.dtype())
    }

    /// GPU-native 2-D convolution.
    ///
    /// Input shape: `[batch, in_channels, height, width]`
    /// Kernel shape: `[out_channels, in_channels/groups, kH, kW]`
    /// Output shape: `[batch, out_channels, out_height, out_width]`
    pub(super) fn gpu_conv2d(
        input: &DynTensor,
        kernel: &DynTensor,
        bias: Option<&DynTensor>,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<DynTensor> {
        Self::validate_same_float_dtype(input, kernel, "gpu_conv2d")?;
        if let Some(b) = bias {
            Self::validate_same_float_dtype(input, b, "gpu_conv2d")?;
        }

        // Defense-in-depth: validate conv parameters before arithmetic.
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

        let in_shape = input.dims();
        let k_shape = kernel.dims();
        let input_data = input.gpu_data::<MetalTensorData>()?;
        let kernel_data = kernel.gpu_data::<MetalTensorData>()?;

        if in_shape.len() != 4 || k_shape.len() != 4 {
            return Err(TensorError::InvalidShape(format!(
                "gpu_conv2d: requires 4D tensors, got input={in_shape:?} kernel={k_shape:?}"
            )));
        }

        let batch = in_shape[0];
        let in_h = in_shape[2];
        let in_w = in_shape[3];
        let out_ch = k_shape[0];
        let k_h = k_shape[2];
        let k_w = k_shape[3];

        // Compute conv2d output spatial dims.
        let effective_kh = (k_h - 1) * dilation + 1;
        let effective_kw = (k_w - 1) * dilation + 1;
        let padded_h = in_h + 2 * padding;
        let padded_w = in_w + 2 * padding;
        if padded_h < effective_kh || padded_w < effective_kw {
            return Err(TensorError::InvalidShape(format!(
                "gpu_conv2d: padded dims ({padded_h},{padded_w}) < effective kernel ({effective_kh},{effective_kw})"
            )));
        }
        let out_h = (padded_h - effective_kh) / stride + 1;
        let out_w = (padded_w - effective_kw) / stride + 1;
        let out_shape = vec![batch, out_ch, out_h, out_w];

        let has_bias = u64::from(bias.is_some());
        let bias_dims: Option<Vec<usize>> = bias.map(|bt| bt.dims().to_vec());
        let def = crate::kernel_def_cache::get_or_build(
            "conv2d",
            &[in_shape, k_shape],
            &[
                stride as u64,
                padding as u64,
                dilation as u64,
                groups as u64,
                has_bias,
            ],
            input.dtype(),
            || {
                let mut b = TensorBlockBuilder::new("dyn_conv2d");
                let in_node = b.add_input("input", in_shape);
                let k_node = b.add_input("kernel", k_shape);
                let bias_node = bias_dims.as_ref().map(|d| b.add_input("bias", d));
                let out = b.add_conv2d_full(
                    in_node, k_node, bias_node, stride, stride, padding, padding, dilation,
                    dilation, groups, &out_shape,
                );
                crate::build_kernel(b, out)
            },
        )?;

        let mut dispatch_inputs: Vec<(&str, crate::gpu_slice::GpuSlice)> = vec![
            ("input", input_data.as_gpu_slice()),
            ("kernel", kernel_data.as_gpu_slice()),
        ];

        if let Some(bias_t) = bias {
            let bias_data = bias_t.gpu_data::<MetalTensorData>()?;
            dispatch_inputs.push(("bias", bias_data.as_gpu_slice()));
        }

        Self::dispatch_def(&def, &dispatch_inputs, &out_shape, input.dtype())
    }

    /// GPU-native 1-D transposed convolution (deconvolution).
    ///
    /// Input shape: `[batch, in_channels, length]`
    /// Kernel shape: `[in_channels, out_channels, kernel_size]`
    /// Output shape: `[batch, out_channels, out_length]`
    ///
    /// Supports all dilation, groups, and stride combinations. The caller
    /// handles `output_padding` via the output-length formula; the GPU kernel
    /// operates on the final `out_len` directly.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gpu_conv_transpose1d(
        input: &DynTensor,
        kernel: &DynTensor,
        bias: Option<&DynTensor>,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<DynTensor> {
        Self::validate_same_float_dtype(input, kernel, "gpu_conv_transpose1d")?;
        if let Some(b) = bias {
            Self::validate_same_float_dtype(input, b, "gpu_conv_transpose1d")?;
        }

        let in_shape = input.dims();
        let k_shape = kernel.dims();
        let input_data = input.gpu_data::<MetalTensorData>()?;
        let kernel_data = kernel.gpu_data::<MetalTensorData>()?;

        if in_shape.len() != 3 || k_shape.len() != 3 {
            return Err(TensorError::InvalidShape(format!(
                "gpu_conv_transpose1d: requires 3D tensors, got input={in_shape:?} kernel={k_shape:?}"
            )));
        }

        let batch = in_shape[0];
        let in_ch = in_shape[1];
        let k_out_ch_per_g = k_shape[1];
        let in_len = in_shape[2];
        let k_size = k_shape[2];

        if groups == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "groups",
                value: 0,
                reason: "must be > 0",
            });
        }
        if !in_ch.is_multiple_of(groups) {
            return Err(TensorError::ConvParameterInvalid {
                param: "groups",
                value: groups,
                reason: "must divide in_channels",
            });
        }
        let out_ch = k_out_ch_per_g * groups;

        // ConvTranspose1d output length:
        // out_len = (in_len - 1) * stride + dilation * (kernel_size - 1) + 1
        //           - 2 * padding + output_padding
        if output_padding >= stride {
            return Err(TensorError::ConvParameterInvalid {
                param: "output_padding",
                value: output_padding,
                reason: "must be < stride",
            });
        }
        let positive = (in_len - 1)
            .checked_mul(stride)
            .and_then(|v| {
                dilation
                    .checked_mul(k_size.checked_sub(1)?)
                    .and_then(|dk| v.checked_add(dk))
            })
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| {
                TensorError::InvalidShape(format!(
                    "gpu_conv_transpose1d: output length overflow (in_len={in_len}, \
                     k_size={k_size}, stride={stride}, dilation={dilation})"
                ))
            })?;
        let negative = 2usize
            .checked_mul(padding)
            .ok_or(TensorError::ConvParameterInvalid {
                param: "padding",
                value: padding,
                reason: "2 * padding overflows usize",
            })?;
        if negative >= positive {
            return Err(TensorError::ConvParameterInvalid {
                param: "padding",
                value: padding,
                reason: "2 * padding >= base output length (zero-length output)",
            });
        }
        // Full output length including output_padding.
        // The GPU kernel operates on the full out_len so that convolution
        // writes landing in the output_padding region are captured correctly
        // (matching the CPU path which allocates out_len and writes into it).
        let out_len = positive - negative + output_padding;
        let out_shape = vec![batch, out_ch, out_len];

        let has_bias = u64::from(bias.is_some());
        let bias_dims: Option<Vec<usize>> = bias.map(|bt| bt.dims().to_vec());
        let def = crate::kernel_def_cache::get_or_build(
            "conv_transpose1d",
            &[in_shape, k_shape],
            &[
                stride as u64,
                padding as u64,
                output_padding as u64,
                dilation as u64,
                groups as u64,
                has_bias,
            ],
            input.dtype(),
            || {
                let mut b = TensorBlockBuilder::new("dyn_conv_transpose1d");
                let in_node = b.add_input("input", in_shape);
                let k_node = b.add_input("kernel", k_shape);
                let bias_node = bias_dims.as_ref().map(|d| b.add_input("bias", d));
                let out = b.add_conv_transpose_1d(
                    in_node,
                    k_node,
                    bias_node,
                    stride,
                    padding,
                    dilation,
                    groups,
                    output_padding,
                    &out_shape,
                );
                crate::build_kernel(b, out)
            },
        )?;

        let mut dispatch_inputs: Vec<(&str, crate::gpu_slice::GpuSlice)> = vec![
            ("input", input_data.as_gpu_slice()),
            ("kernel", kernel_data.as_gpu_slice()),
        ];

        if let Some(bias_t) = bias {
            let bias_data = bias_t.gpu_data::<MetalTensorData>()?;
            dispatch_inputs.push(("bias", bias_data.as_gpu_slice()));
        }

        Self::dispatch_def(&def, &dispatch_inputs, &out_shape, input.dtype())
    }
}
