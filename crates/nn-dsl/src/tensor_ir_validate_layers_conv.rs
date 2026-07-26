// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv1d, Conv2d, and ConvTranspose1d tensor IR validators.
//!
//! Extracted from `tensor_ir_validate_layers.rs` (#827 Direction 4) to keep
//! the layer validators under 400 lines.

use super::super::{TensorIRConvError, TensorIRError, TensorKernelDef, TensorNodeId};

impl TensorKernelDef {
    pub(super) fn validate_conv1d(
        &self,
        current: TensorNodeId,
        input: TensorNodeId,
        weight: TensorNodeId,
        bias: Option<TensorNodeId>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, input)?;
        self.check_ref(current, weight)?;

        // Dilation must be >= 1
        if dilation == 0 {
            return Err(TensorIRConvError::Conv1dZeroDilation.into());
        }

        // Groups must be >= 1
        if groups == 0 {
            return Err(TensorIRConvError::Conv1dZeroGroups.into());
        }

        // Input must have at least 2 dimensions: [in_channels, in_length]
        let input_shape = &self.nodes[input.index()].shape;
        if input_shape.len() < 2 {
            return Err(TensorIRConvError::Conv1dInputRankTooLow {
                rank: input_shape.len(),
            }
            .into());
        }

        // Weight must be 3D: [out_channels, in_channels/groups, kernel_size]
        let weight_shape = &self.nodes[weight.index()].shape;
        if weight_shape.len() != 3 {
            return Err(TensorIRConvError::Conv1dWeightShape {
                shape: weight_shape.clone(),
            }
            .into());
        }

        // Input channels must be divisible by groups
        let in_channels = input_shape[input_shape.len() - 2];
        if groups > 1 && !in_channels.is_multiple_of(groups) {
            return Err(TensorIRConvError::Conv1dGroupsChannelMismatch {
                in_channels,
                groups,
            }
            .into());
        }

        // Output channels must be divisible by groups
        let out_channels = weight_shape[0];
        if groups > 1 && !out_channels.is_multiple_of(groups) {
            return Err(TensorIRConvError::Conv1dGroupsOutputMismatch {
                out_channels,
                groups,
            }
            .into());
        }

        // Weight in_channels must equal in_channels / groups
        let expected_weight_in = in_channels / groups;
        let weight_in_channels = weight_shape[1];
        if weight_in_channels != expected_weight_in {
            return Err(TensorIRConvError::Conv1dGroupsWeightMismatch {
                weight_in_channels,
                expected: expected_weight_in,
            }
            .into());
        }

        // Stride must be >= 1
        if stride == 0 {
            return Err(TensorIRConvError::Conv1dZeroStride.into());
        }

        // Compute effective kernel with dilation (checked arithmetic)
        let kernel_size = weight_shape[2];
        if kernel_size == 0 {
            return Err(TensorIRConvError::Conv1dZeroKernelSize.into());
        }
        // kernel_size >= 1 guaranteed above, so kernel_size - 1 is safe.
        let effective_kernel = dilation
            .checked_mul(kernel_size - 1)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| {
                TensorIRError::from(TensorIRConvError::Conv1dArithmeticOverflow {
                    context: format!(
                        "effective_kernel: dilation={dilation} * (kernel_size={kernel_size} - 1) + 1"
                    ),
                })
            })?;
        let in_len = input_shape[input_shape.len() - 1];
        let padded = padding
            .checked_mul(2)
            .and_then(|v| v.checked_add(in_len))
            .ok_or_else(|| {
                TensorIRError::from(TensorIRConvError::Conv1dArithmeticOverflow {
                    context: format!("padded: in_len={in_len} + 2 * padding={padding}"),
                })
            })?;
        if padded < effective_kernel {
            return Err(TensorIRConvError::Conv1dKernelTooLarge {
                kernel_size: effective_kernel,
                padded_len: padded,
                in_len,
                padding,
            }
            .into());
        }

        // Validate bias if present
        if let Some(bias_id) = bias {
            self.check_ref(current, bias_id)?;
            let bias_shape = &self.nodes[bias_id.index()].shape;
            if bias_shape != &[out_channels] {
                return Err(TensorIRConvError::Conv1dBiasShape {
                    expected: out_channels,
                    got_shape: bias_shape.clone(),
                }
                .into());
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn validate_conv2d(
        &self,
        current: TensorNodeId,
        input: TensorNodeId,
        weight: TensorNodeId,
        bias: Option<TensorNodeId>,
        stride_h: usize,
        stride_w: usize,
        _padding_h: usize,
        _padding_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        groups: usize,
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, input)?;
        self.check_ref(current, weight)?;

        // Dilation must be >= 1
        if dilation_h == 0 || dilation_w == 0 {
            return Err(TensorIRConvError::Conv2dZeroDilation {
                dilation_h,
                dilation_w,
            }
            .into());
        }

        // Groups must be >= 1
        if groups == 0 {
            return Err(TensorIRConvError::Conv2dZeroGroups.into());
        }

        // Input must have at least 3 dimensions: [in_channels, height, width]
        let input_shape = &self.nodes[input.index()].shape;
        if input_shape.len() < 3 {
            return Err(TensorIRConvError::Conv2dInputRankTooLow {
                rank: input_shape.len(),
            }
            .into());
        }

        // Weight must be 4D: [out_channels, in_channels/groups, kernel_h, kernel_w]
        let weight_shape = &self.nodes[weight.index()].shape;
        if weight_shape.len() != 4 {
            return Err(TensorIRConvError::Conv2dWeightShape {
                shape: weight_shape.clone(),
            }
            .into());
        }

        // Input channels must be divisible by groups
        let in_channels = input_shape[input_shape.len() - 3];
        if groups > 1 && !in_channels.is_multiple_of(groups) {
            return Err(TensorIRConvError::Conv2dGroupsChannelMismatch {
                in_channels,
                groups,
            }
            .into());
        }

        // Output channels must be divisible by groups
        let out_channels = weight_shape[0];
        if groups > 1 && !out_channels.is_multiple_of(groups) {
            return Err(TensorIRConvError::Conv2dGroupsOutputMismatch {
                out_channels,
                groups,
            }
            .into());
        }

        // Weight in_channels must equal in_channels / groups
        let expected_weight_in = in_channels / groups;
        let weight_in_channels = weight_shape[1];
        if weight_in_channels != expected_weight_in {
            return Err(TensorIRConvError::Conv2dGroupsWeightMismatch {
                weight_in_channels,
                expected: expected_weight_in,
            }
            .into());
        }

        // Stride must be >= 1
        if stride_h == 0 || stride_w == 0 {
            return Err(TensorIRConvError::Conv2dZeroStride { stride_h, stride_w }.into());
        }

        // Kernel dimensions must be >= 1
        let kernel_h = weight_shape[2];
        let kernel_w = weight_shape[3];
        if kernel_h == 0 || kernel_w == 0 {
            return Err(TensorIRConvError::Conv2dZeroKernelSize { kernel_h, kernel_w }.into());
        }

        // Validate bias if present
        if let Some(bias_id) = bias {
            self.check_ref(current, bias_id)?;
            let bias_shape = &self.nodes[bias_id.index()].shape;
            if bias_shape != &[out_channels] {
                return Err(TensorIRConvError::Conv2dBiasShape {
                    expected: out_channels,
                    got_shape: bias_shape.clone(),
                }
                .into());
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn validate_conv_transpose_1d(
        &self,
        current: TensorNodeId,
        input: TensorNodeId,
        weight: TensorNodeId,
        bias: Option<TensorNodeId>,
        stride: usize,
        _padding: usize,
        dilation: usize,
        groups: usize,
        output_padding: usize,
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, input)?;
        self.check_ref(current, weight)?;

        let input_shape = &self.nodes[input.index()].shape;
        if input_shape.len() < 2 {
            return Err(TensorIRConvError::ConvTranspose1dInputRankTooLow {
                rank: input_shape.len(),
            }
            .into());
        }

        let weight_shape = &self.nodes[weight.index()].shape;
        if weight_shape.len() != 3 {
            return Err(TensorIRConvError::ConvTranspose1dWeightShape {
                shape: weight_shape.clone(),
            }
            .into());
        }

        if stride == 0 {
            return Err(TensorIRConvError::ConvTranspose1dZeroStride.into());
        }

        // PyTorch constraint: output_padding must be < stride.
        if output_padding >= stride {
            return Err(TensorIRConvError::ConvTranspose1dArithmeticOverflow {
                context: format!("output_padding={output_padding} must be < stride={stride}"),
            }
            .into());
        }

        if dilation == 0 {
            return Err(TensorIRConvError::ConvTranspose1dZeroDilation.into());
        }

        if groups == 0 {
            return Err(TensorIRConvError::ConvTranspose1dZeroGroups.into());
        }

        let in_channels = input_shape[input_shape.len() - 2];
        let weight_in_channels = weight_shape[0];
        if weight_in_channels != in_channels {
            return Err(TensorIRConvError::ConvTranspose1dChannelMismatch {
                expected: weight_in_channels,
                got: in_channels,
            }
            .into());
        }

        // For grouped conv_transpose, in_channels must be divisible by groups.
        if !in_channels.is_multiple_of(groups) {
            return Err(TensorIRConvError::ConvTranspose1dGroupChannelMismatch {
                in_channels,
                groups,
            }
            .into());
        }

        let out_ch_per_group = weight_shape[1];
        let out_channels = out_ch_per_group * groups;
        if let Some(bias_id) = bias {
            self.check_ref(current, bias_id)?;
            let bias_shape = &self.nodes[bias_id.index()].shape;
            if bias_shape != &[out_channels] {
                return Err(TensorIRConvError::ConvTranspose1dBiasShape {
                    expected: out_channels,
                    got_shape: bias_shape.clone(),
                }
                .into());
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn validate_conv_transpose_2d(
        &self,
        current: TensorNodeId,
        input: TensorNodeId,
        weight: TensorNodeId,
        bias: Option<TensorNodeId>,
        stride_h: usize,
        stride_w: usize,
        _padding_h: usize,
        _padding_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        groups: usize,
        output_padding_h: usize,
        output_padding_w: usize,
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, input)?;
        self.check_ref(current, weight)?;

        let input_shape = &self.nodes[input.index()].shape;
        if input_shape.len() < 3 {
            return Err(TensorIRConvError::ConvTranspose2dInputRankTooLow {
                rank: input_shape.len(),
            }
            .into());
        }

        let weight_shape = &self.nodes[weight.index()].shape;
        if weight_shape.len() != 4 {
            return Err(TensorIRConvError::ConvTranspose2dWeightShape {
                shape: weight_shape.clone(),
            }
            .into());
        }

        if stride_h == 0 || stride_w == 0 {
            return Err(TensorIRConvError::ConvTranspose2dZeroStride { stride_h, stride_w }.into());
        }

        if output_padding_h >= stride_h || output_padding_w >= stride_w {
            return Err(TensorIRConvError::ConvTranspose2dArithmeticOverflow {
                context: format!(
                    "output_padding=({output_padding_h},{output_padding_w}) must be < stride=({stride_h},{stride_w})"
                ),
            }
            .into());
        }

        if dilation_h == 0 || dilation_w == 0 {
            return Err(TensorIRConvError::ConvTranspose2dZeroDilation {
                dilation_h,
                dilation_w,
            }
            .into());
        }

        if groups == 0 {
            return Err(TensorIRConvError::ConvTranspose2dZeroGroups.into());
        }

        let in_channels = input_shape[input_shape.len() - 3];
        let weight_in_channels = weight_shape[0];
        if weight_in_channels != in_channels {
            return Err(TensorIRConvError::ConvTranspose2dChannelMismatch {
                expected: weight_in_channels,
                got: in_channels,
            }
            .into());
        }

        if !in_channels.is_multiple_of(groups) {
            return Err(TensorIRConvError::ConvTranspose2dGroupChannelMismatch {
                in_channels,
                groups,
            }
            .into());
        }

        let out_ch_per_group = weight_shape[1];
        let out_channels = out_ch_per_group * groups;
        if let Some(bias_id) = bias {
            self.check_ref(current, bias_id)?;
            let bias_shape = &self.nodes[bias_id.index()].shape;
            if bias_shape != &[out_channels] {
                return Err(TensorIRConvError::ConvTranspose2dBiasShape {
                    expected: out_channels,
                    got_shape: bias_shape.clone(),
                }
                .into());
            }
        }

        Ok(())
    }
}
