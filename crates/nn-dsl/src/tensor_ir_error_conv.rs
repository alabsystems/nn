// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Convolution-specific validation error variants for the tensor IR.
//!
//! Extracted from `tensor_ir_error_layer.rs` to keep that file under the
//! 500-line limit. Covers Conv1d, Conv2d, and ConvTranspose1d validation.
//!
//! Part of #1342.

use thiserror::Error;

/// Convolution-specific tensor IR validation errors.
///
/// Wrapped by `TensorIRLayerError::Conv(TensorIRConvError)`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TensorIRConvError {
    // --- Conv1d ---
    #[error("Conv1d weight must have 3 dimensions [out_ch, in_ch, kernel_size], got {shape:?}")]
    Conv1dWeightShape { shape: Vec<usize> },

    #[error("Conv1d stride must be >= 1, got 0")]
    Conv1dZeroStride,

    #[error(
        "Conv1d kernel size ({kernel_size}) larger than padded input length ({padded_len}): in_len={in_len}, padding={padding}"
    )]
    Conv1dKernelTooLarge {
        kernel_size: usize,
        padded_len: usize,
        in_len: usize,
        padding: usize,
    },

    #[error("Conv1d bias must have shape [{expected}], got {got_shape:?}")]
    Conv1dBiasShape {
        expected: usize,
        got_shape: Vec<usize>,
    },

    #[error("Conv1d input must have at least 2 dimensions, got {rank}")]
    Conv1dInputRankTooLow { rank: usize },

    #[error("Conv1d kernel_size must be >= 1, got 0")]
    Conv1dZeroKernelSize,

    #[error("Conv1d arithmetic overflow computing output shape: {context}")]
    Conv1dArithmeticOverflow { context: String },

    #[error("Conv1d dilation must be >= 1, got 0")]
    Conv1dZeroDilation,

    #[error("Conv1d groups must be >= 1, got 0")]
    Conv1dZeroGroups,

    #[error("Conv1d input channels ({in_channels}) must be divisible by groups ({groups})")]
    Conv1dGroupsChannelMismatch { in_channels: usize, groups: usize },

    #[error("Conv1d output channels ({out_channels}) must be divisible by groups ({groups})")]
    Conv1dGroupsOutputMismatch { out_channels: usize, groups: usize },

    #[error("Conv1d weight in_channels ({weight_in_channels}) must equal in_channels/groups ({expected})")]
    Conv1dGroupsWeightMismatch {
        weight_in_channels: usize,
        expected: usize,
    },

    // --- Conv2d ---
    #[error("Conv2d weight must have 4 dimensions [out_ch, in_ch, kH, kW], got {shape:?}")]
    Conv2dWeightShape { shape: Vec<usize> },

    #[error("Conv2d input must have at least 3 dimensions [C, H, W], got {rank}")]
    Conv2dInputRankTooLow { rank: usize },

    #[error("Conv2d stride must be >= 1 in both dimensions, got stride=({stride_h},{stride_w})")]
    Conv2dZeroStride { stride_h: usize, stride_w: usize },

    #[error(
        "Conv2d dilation must be >= 1 in both dimensions, got dilation=({dilation_h},{dilation_w})"
    )]
    Conv2dZeroDilation {
        dilation_h: usize,
        dilation_w: usize,
    },

    #[error("Conv2d groups must be >= 1, got 0")]
    Conv2dZeroGroups,

    #[error(
        "Conv2d kernel size must be >= 1 in both dimensions, got kernel=({kernel_h},{kernel_w})"
    )]
    Conv2dZeroKernelSize { kernel_h: usize, kernel_w: usize },

    #[error("Conv2d input channels ({in_channels}) must be divisible by groups ({groups})")]
    Conv2dGroupsChannelMismatch { in_channels: usize, groups: usize },

    #[error("Conv2d output channels ({out_channels}) must be divisible by groups ({groups})")]
    Conv2dGroupsOutputMismatch { out_channels: usize, groups: usize },

    #[error("Conv2d weight in_channels ({weight_in_channels}) must equal in_channels/groups ({expected})")]
    Conv2dGroupsWeightMismatch {
        weight_in_channels: usize,
        expected: usize,
    },

    #[error("Conv2d bias must have shape [{expected}], got {got_shape:?}")]
    Conv2dBiasShape {
        expected: usize,
        got_shape: Vec<usize>,
    },

    #[error("Conv2d arithmetic overflow computing output shape: {context}")]
    Conv2dArithmeticOverflow { context: String },

    // --- ConvTranspose1d ---
    #[error(
        "ConvTranspose1d weight must have 3 dimensions [in_ch, out_ch, kernel_size], got {shape:?}"
    )]
    ConvTranspose1dWeightShape { shape: Vec<usize> },

    #[error("ConvTranspose1d stride must be >= 1, got 0")]
    ConvTranspose1dZeroStride,

    #[error("ConvTranspose1d input channels mismatch: weight expects {expected}, input has {got}")]
    ConvTranspose1dChannelMismatch { expected: usize, got: usize },

    #[error("ConvTranspose1d bias must have shape [{expected}], got {got_shape:?}")]
    ConvTranspose1dBiasShape {
        expected: usize,
        got_shape: Vec<usize>,
    },

    #[error("ConvTranspose1d input must have at least 2 dimensions, got {rank}")]
    ConvTranspose1dInputRankTooLow { rank: usize },

    #[error("ConvTranspose1d output length must be >= 1, got {out_length} (in_len={in_length}, stride={stride}, kernel={kernel_size}, padding={padding})")]
    ConvTranspose1dOutputNonPositive {
        out_length: isize,
        in_length: usize,
        stride: usize,
        kernel_size: usize,
        padding: usize,
    },

    #[error("ConvTranspose1d arithmetic overflow computing output shape: {context}")]
    ConvTranspose1dArithmeticOverflow { context: String },

    #[error("ConvTranspose1d dilation must be >= 1, got 0")]
    ConvTranspose1dZeroDilation,

    #[error("ConvTranspose1d groups must be >= 1, got 0")]
    ConvTranspose1dZeroGroups,

    #[error("ConvTranspose1d in_channels ({in_channels}) must be divisible by groups ({groups})")]
    ConvTranspose1dGroupChannelMismatch { in_channels: usize, groups: usize },

    // --- ConvTranspose2d ---
    #[error(
        "ConvTranspose2d weight must have 4 dimensions [in_ch, out_ch, kH, kW], got {shape:?}"
    )]
    ConvTranspose2dWeightShape { shape: Vec<usize> },

    #[error("ConvTranspose2d stride must be >= 1, got stride=({stride_h},{stride_w})")]
    ConvTranspose2dZeroStride { stride_h: usize, stride_w: usize },

    #[error("ConvTranspose2d input channels mismatch: weight expects {expected}, input has {got}")]
    ConvTranspose2dChannelMismatch { expected: usize, got: usize },

    #[error("ConvTranspose2d bias must have shape [{expected}], got {got_shape:?}")]
    ConvTranspose2dBiasShape {
        expected: usize,
        got_shape: Vec<usize>,
    },

    #[error("ConvTranspose2d input must have at least 3 dimensions [C, H, W], got {rank}")]
    ConvTranspose2dInputRankTooLow { rank: usize },

    #[error("ConvTranspose2d arithmetic overflow computing output shape: {context}")]
    ConvTranspose2dArithmeticOverflow { context: String },

    #[error("ConvTranspose2d dilation must be >= 1, got dilation=({dilation_h},{dilation_w})")]
    ConvTranspose2dZeroDilation {
        dilation_h: usize,
        dilation_w: usize,
    },

    #[error("ConvTranspose2d groups must be >= 1, got 0")]
    ConvTranspose2dZeroGroups,

    #[error("ConvTranspose2d in_channels ({in_channels}) must be divisible by groups ({groups})")]
    ConvTranspose2dGroupChannelMismatch { in_channels: usize, groups: usize },

    // --- Pool2d (AvgPool2d / MaxPool2d) ---
    #[error("Pool2d input must have at least 3 dimensions [C, H, W], got {rank}")]
    Pool2dInputRankTooLow { rank: usize },

    #[error("Pool2d stride must be >= 1 in both dimensions, got stride=({stride_h},{stride_w})")]
    Pool2dZeroStride { stride_h: usize, stride_w: usize },

    #[error(
        "Pool2d kernel size must be >= 1 in both dimensions, got kernel=({kernel_h},{kernel_w})"
    )]
    Pool2dZeroKernelSize { kernel_h: usize, kernel_w: usize },

    #[error(
        "Pool2d kernel ({kernel_h},{kernel_w}) larger than padded input ({padded_h},{padded_w})"
    )]
    Pool2dKernelTooLarge {
        kernel_h: usize,
        kernel_w: usize,
        padded_h: usize,
        padded_w: usize,
    },

    #[error("Pool2d arithmetic overflow computing output shape: {context}")]
    Pool2dArithmeticOverflow { context: String },
}
