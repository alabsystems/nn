// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv1d tensor kernel builder.
//!
//! Builds a `TensorKernelDef` for 1D convolution that maps to NY's
//! `Layer::Conv1d(Conv1dLayer)`. Weight and bias inputs are treated as fixed
//! parameters (not verified as variables).
//!
//! # dvoice parameter coverage
//!
//! Stride, padding, kernel_size, optional bias, and dilation are fully
//! supported. Dilation uses kernel expansion (zero-insertion) to work around
//! NY's lack of native dilation support (#582). Groups != 1 are
//! rejected until NY upstream adds support (NY#3170).

use crate::tensor_ir::{
    TensorIRConvError, TensorIRError, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind,
};

/// Build a Conv1d tensor kernel definition with default dilation=1, groups=1.
///
/// Delegates to [`build_conv1d_full`] with `dilation=1` and `groups=1`.
/// Use `build_conv1d_full` when you need dilated or grouped convolutions.
///
/// # Errors
///
/// Returns `TensorIRError` if parameters are invalid (zero stride/kernel_size,
/// arithmetic overflow, or padded input smaller than kernel).
pub fn build_conv1d(
    name: &str,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    in_length: usize,
    stride: usize,
    padding: usize,
    has_bias: bool,
) -> Result<TensorKernelDef, TensorIRError> {
    build_conv1d_full(
        name,
        in_channels,
        out_channels,
        kernel_size,
        in_length,
        stride,
        padding,
        1, // dilation
        1, // groups
        has_bias,
    )
}

/// Build a Conv1d tensor kernel definition with full parameter control.
///
/// Inputs (in order): `[data, weight]` or `[data, weight, bias]`.
/// Weight and bias are treated as constant parameters during verification.
///
/// # Arguments
///
/// * `name` — Kernel name for diagnostics and node naming.
/// * `in_channels` — Number of input channels.
/// * `out_channels` — Number of output channels (filters).
/// * `kernel_size` — Spatial extent of the convolution kernel (must be >= 1).
/// * `in_length` — Spatial length of the input tensor.
/// * `stride` — Convolution stride (must be >= 1).
/// * `padding` — Zero-padding applied to both sides.
/// * `dilation` — Spacing between kernel elements (must be >= 1).
/// * `groups` — Number of input channel groups (must be >= 1).
/// * `has_bias` — Whether to include a bias input node.
///
/// # Errors
///
/// Returns `TensorIRError` if any parameter is zero when it must be >= 1,
/// if `in_channels` is not divisible by `groups`, if arithmetic overflows,
/// or if the effective padded input is smaller than the effective kernel.
pub fn build_conv1d_full(
    name: &str,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    in_length: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    has_bias: bool,
) -> Result<TensorKernelDef, TensorIRError> {
    if stride == 0 {
        return Err(TensorIRConvError::Conv1dZeroStride.into());
    }
    if kernel_size == 0 {
        return Err(TensorIRConvError::Conv1dZeroKernelSize.into());
    }
    if dilation == 0 {
        return Err(TensorIRConvError::Conv1dZeroDilation.into());
    }
    if groups == 0 {
        return Err(TensorIRConvError::Conv1dZeroGroups.into());
    }
    if !in_channels.is_multiple_of(groups) {
        return Err(TensorIRConvError::Conv1dGroupsChannelMismatch {
            in_channels,
            groups,
        }
        .into());
    }

    // effective_kernel = dilation * (kernel_size - 1) + 1
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

    // padded = in_length + 2 * padding
    let padded = padding
        .checked_mul(2)
        .and_then(|v| v.checked_add(in_length))
        .ok_or_else(|| {
            TensorIRError::from(TensorIRConvError::Conv1dArithmeticOverflow {
                context: format!("padded: in_length={in_length} + 2 * padding={padding}"),
            })
        })?;

    if padded < effective_kernel {
        return Err(TensorIRConvError::Conv1dKernelTooLarge {
            kernel_size: effective_kernel,
            padded_len: padded,
            in_len: in_length,
            padding,
        }
        .into());
    }

    // out_length = (padded - effective_kernel) / stride + 1
    // padded >= effective_kernel guaranteed above, subtraction is safe.
    let out_length = (padded - effective_kernel) / stride + 1;
    let weight_in_channels = in_channels / groups;

    let mut nodes = vec![
        // %0 = input data: [in_channels, in_length]
        TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: crate::input_names::DATA.into(),
                shape: vec![in_channels, in_length],
            },
            vec![in_channels, in_length],
        ),
        // %1 = input weight: [out_channels, in_channels/groups, kernel_size]
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Input {
                name: "weight".into(),
                shape: vec![out_channels, weight_in_channels, kernel_size],
            },
            vec![out_channels, weight_in_channels, kernel_size],
        ),
    ];

    let bias_node = if has_bias {
        // %2 = input bias: [out_channels]
        nodes.push(TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::Input {
                name: "bias".into(),
                shape: vec![out_channels],
            },
            vec![out_channels],
        ));
        Some(TensorNodeId::new(2))
    } else {
        None
    };

    let conv_id = TensorNodeId::new(nodes.len());
    nodes.push(TensorNode::new(
        conv_id,
        TensorOpKind::Conv1d {
            input: TensorNodeId::new(0),
            weight: TensorNodeId::new(1),
            bias: bias_node,
            stride,
            padding,
            dilation,
            groups,
        },
        vec![out_channels, out_length],
    ));

    Ok(TensorKernelDef::new(name, nodes, conv_id))
}

#[cfg(test)]
#[path = "conv1d_tests.rs"]
mod tests;
