// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Convolution dispatch parameters extracted from `dispatch_step.rs`.
//!
//! Contains the `Conv1dParams`, `Conv2dParams`, and `ConvTranspose1dParams`
//! structs that hold the convolution-specific fields for the corresponding
//! `DispatchStep` variants.

use crate::ir::ScalarType;
use crate::tensor_ir::TensorNodeId;

/// Parameters for a 1-D convolution dispatch step.
#[non_exhaustive]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "plan-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Conv1dParams {
    /// Name of the generated MSL kernel function.
    pub kernel_name: String,
    /// Scalar type (f32 or f16).
    pub dtype: ScalarType,
    /// Tensor node for input data.
    pub input: TensorNodeId,
    /// Tensor node for weight tensor.
    pub weight: TensorNodeId,
    /// Optional tensor node for bias.
    pub bias: Option<TensorNodeId>,
    /// Tensor node for output.
    pub output: TensorNodeId,
    /// Input channels.
    pub in_channels: usize,
    /// Output channels.
    pub out_channels: usize,
    /// Convolution kernel size.
    pub kernel_size: usize,
    /// Input length (time/spatial dimension).
    pub in_length: usize,
    /// Total output elements (out_channels * out_length).
    pub total_elements: usize,
    /// Convolution stride.
    pub stride: usize,
    /// Zero-padding on each side.
    pub padding: usize,
    /// Dilation factor.
    pub dilation: usize,
    /// Number of channel groups.
    pub groups: usize,
}

#[allow(clippy::too_many_arguments)]
impl Conv1dParams {
    pub fn new(
        kernel_name: String,
        dtype: ScalarType,
        input: TensorNodeId,
        weight: TensorNodeId,
        bias: Option<TensorNodeId>,
        output: TensorNodeId,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        in_length: usize,
        total_elements: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Self {
        Self {
            kernel_name,
            dtype,
            input,
            weight,
            bias,
            output,
            in_channels,
            out_channels,
            kernel_size,
            in_length,
            total_elements,
            stride,
            padding,
            dilation,
            groups,
        }
    }
}

/// Parameters for a 2-D convolution dispatch step.
#[non_exhaustive]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "plan-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Conv2dParams {
    /// Name of the generated MSL kernel function.
    pub kernel_name: String,
    /// Scalar type (f32 or f16).
    pub dtype: ScalarType,
    /// Tensor node for input data.
    pub input: TensorNodeId,
    /// Tensor node for weight tensor.
    pub weight: TensorNodeId,
    /// Optional tensor node for bias.
    pub bias: Option<TensorNodeId>,
    /// Tensor node for output.
    pub output: TensorNodeId,
    /// Input channels.
    pub in_channels: usize,
    /// Output channels.
    pub out_channels: usize,
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
    /// Input height.
    pub in_height: usize,
    /// Input width.
    pub in_width: usize,
    /// Total output elements (out_channels * out_h * out_w).
    pub total_elements: usize,
    /// Convolution stride per dimension.
    pub stride_h: usize,
    pub stride_w: usize,
    /// Zero-padding on each side per dimension.
    pub padding_h: usize,
    pub padding_w: usize,
    /// Dilation factor per dimension.
    pub dilation_h: usize,
    pub dilation_w: usize,
    /// Number of channel groups.
    pub groups: usize,
}

#[allow(clippy::too_many_arguments)]
impl Conv2dParams {
    pub fn new(
        kernel_name: String,
        dtype: ScalarType,
        input: TensorNodeId,
        weight: TensorNodeId,
        bias: Option<TensorNodeId>,
        output: TensorNodeId,
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        in_height: usize,
        in_width: usize,
        total_elements: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        groups: usize,
    ) -> Self {
        Self {
            kernel_name,
            dtype,
            input,
            weight,
            bias,
            output,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            in_height,
            in_width,
            total_elements,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups,
        }
    }
}

/// Parameters for a 1-D transposed convolution dispatch step.
#[non_exhaustive]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "plan-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConvTranspose1dParams {
    /// Name of the generated MSL kernel function.
    pub kernel_name: String,
    /// Scalar type (f32 or f16).
    pub dtype: ScalarType,
    /// Tensor node for input data.
    pub input: TensorNodeId,
    /// Tensor node for weight tensor.
    pub weight: TensorNodeId,
    /// Optional tensor node for bias.
    pub bias: Option<TensorNodeId>,
    /// Tensor node for output.
    pub output: TensorNodeId,
    /// Input channels.
    pub in_channels: usize,
    /// Output channels.
    pub out_channels: usize,
    /// Convolution kernel size.
    pub kernel_size: usize,
    /// Input length (time/spatial dimension).
    pub in_length: usize,
    /// Total output elements (out_channels * out_length).
    pub total_elements: usize,
    /// Convolution stride.
    pub stride: usize,
    /// Zero-padding on each side.
    pub padding: usize,
    /// Dilation (spacing between kernel elements).
    pub dilation: usize,
    /// Number of channel groups.
    pub groups: usize,
    /// Extra elements added to one side of the output (must be < stride).
    pub output_padding: usize,
}

#[allow(clippy::too_many_arguments)]
impl ConvTranspose1dParams {
    pub fn new(
        kernel_name: String,
        dtype: ScalarType,
        input: TensorNodeId,
        weight: TensorNodeId,
        bias: Option<TensorNodeId>,
        output: TensorNodeId,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        in_length: usize,
        total_elements: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        output_padding: usize,
    ) -> Self {
        Self {
            kernel_name,
            dtype,
            input,
            weight,
            bias,
            output,
            in_channels,
            out_channels,
            kernel_size,
            in_length,
            total_elements,
            stride,
            padding,
            dilation,
            groups,
            output_padding,
        }
    }
}
