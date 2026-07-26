// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU Conv3d dispatch for [`MetalDynBackend`].
//!
//! Establishes the Conv3d GPU dispatch path for Qwen3-VL vision patch embedding.
//! Initial implementation uses CPU fallback (read to CPU, conv3d, upload).
//! Native MSL kernel optimization is a follow-up.
//!
//! Part of #3866.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{conv3d_out_len, Device, Result, TensorError};

impl super::MetalDynBackend {
    /// GPU Conv3d dispatch.
    ///
    /// Input shape: `[batch, in_channels, depth, height, width]`
    /// Kernel shape: `[out_channels, in_channels/groups, kD, kH, kW]`
    /// Output shape: `[batch, out_channels, out_depth, out_height, out_width]`
    ///
    /// Currently delegates to CPU conv3d and uploads the result back to GPU.
    /// This establishes the dispatch path; a native MSL im2col + GEMM kernel
    /// is a follow-up optimization.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gpu_conv3d(
        input: &DynTensor,
        kernel: &DynTensor,
        bias: Option<&DynTensor>,
        padding: [usize; 3],
        stride: [usize; 3],
        dilation: [usize; 3],
        groups: usize,
    ) -> Result<DynTensor> {
        // Validate dtypes.
        Self::validate_same_float_dtype(input, kernel, "gpu_conv3d")?;
        if let Some(b) = bias {
            Self::validate_same_float_dtype(input, b, "gpu_conv3d")?;
        }

        let in_shape = input.dims();
        let k_shape = kernel.dims();

        // Validate ranks.
        if in_shape.len() != 5 {
            return Err(TensorError::InvalidShape(format!(
                "gpu_conv3d: input must be 5D [B,C,D,H,W], got {in_shape:?}"
            )));
        }
        if k_shape.len() != 5 {
            return Err(TensorError::InvalidShape(format!(
                "gpu_conv3d: kernel must be 5D [C_out,C_in/g,kD,kH,kW], got {k_shape:?}"
            )));
        }

        // Validate conv parameters (defense-in-depth).
        for (i, &s) in stride.iter().enumerate() {
            if s == 0 {
                return Err(TensorError::ConvParameterInvalid {
                    param: "stride",
                    value: 0,
                    reason: "must be > 0",
                });
            }
            let _ = i; // suppress unused
        }
        for &d in &dilation {
            if d == 0 {
                return Err(TensorError::ConvParameterInvalid {
                    param: "dilation",
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

        let in_ch = in_shape[1];
        let out_ch = k_shape[0];

        if !in_ch.is_multiple_of(groups) {
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

        // Compute output dimensions (validates kernel/padding/dilation combos).
        let _out_d = conv3d_out_len(
            in_shape[2], k_shape[2], padding[0], stride[0], dilation[0],
        )?;
        let _out_h = conv3d_out_len(
            in_shape[3], k_shape[3], padding[1], stride[1], dilation[1],
        )?;
        let _out_w = conv3d_out_len(
            in_shape[4], k_shape[4], padding[2], stride[2], dilation[2],
        )?;

        // CPU fallback: transfer to CPU, run conv3d, upload back.
        // This is the initial dispatch path. A native MSL kernel will replace
        // this fallback for production performance.
        let original_device = input.device();
        let cpu_input = input.to_device(&Device::Cpu)?;
        let cpu_kernel = kernel.to_device(&Device::Cpu)?;

        let cpu_result = cpu_input.conv3d(
            &cpu_kernel, padding, stride, dilation, groups,
        )?;

        // Add bias on CPU if present, then upload.
        let biased = if let Some(bias_t) = bias {
            let cpu_bias = bias_t.to_device(&Device::Cpu)?;
            // Bias shape: [out_ch]. Reshape to [1, out_ch, 1, 1, 1] for broadcast.
            let bias_bcast = cpu_bias.reshape([1, out_ch, 1, 1, 1])?;
            cpu_result.add(&bias_bcast)?
        } else {
            cpu_result
        };

        biased.to_device(&original_device)
    }
}

#[cfg(test)]
#[path = "dyn_tensor_metal_conv3d_tests.rs"]
mod conv3d_tests;
