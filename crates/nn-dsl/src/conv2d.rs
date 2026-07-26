// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv2d tensor kernel builder.
//!
//! Builds a `TensorKernelDef` for 2D convolution that maps to NY's
//! `Layer::Conv2d(Conv2dLayer)`. Weight and bias inputs are treated as fixed
//! parameters (not verified as variables).
//!
//! # Demucs parameter coverage
//!
//! Stride, padding, kernel_size (asymmetric H×W), optional bias, dilation, and
//! groups are fully supported. Dilation uses kernel expansion (zero-insertion)
//! for NY translation. Groups != 1 are rejected until NY
//! upstream adds support.

use crate::tensor_ir::{
    TensorIRConvError, TensorIRError, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind,
};

/// Build a Conv2d tensor kernel definition with default dilation=1, groups=1.
///
/// Delegates to [`build_conv2d_full`] with `dilation_h=1`, `dilation_w=1`, `groups=1`.
///
/// # Errors
///
/// Returns `TensorIRError` if parameters are invalid (zero stride/kernel_size,
/// arithmetic overflow, or padded input smaller than kernel).
pub fn build_conv2d(
    name: &str,
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    in_height: usize,
    in_width: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    has_bias: bool,
) -> Result<TensorKernelDef, TensorIRError> {
    build_conv2d_full(
        name,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        in_height,
        in_width,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        1, // dilation_h
        1, // dilation_w
        1, // groups
        has_bias,
    )
}

/// Build a Conv2d tensor kernel definition with full parameter control.
///
/// Inputs (in order): `[data, weight]` or `[data, weight, bias]`.
/// Weight and bias are treated as constant parameters during verification.
///
/// # Arguments
///
/// * `name` — Kernel name for diagnostics and node naming.
/// * `in_channels` — Number of input channels.
/// * `out_channels` — Number of output channels (filters).
/// * `kernel_h` — Vertical extent of the convolution kernel (must be >= 1).
/// * `kernel_w` — Horizontal extent of the convolution kernel (must be >= 1).
/// * `in_height` — Spatial height of the input tensor.
/// * `in_width` — Spatial width of the input tensor.
/// * `stride_h` — Vertical convolution stride (must be >= 1).
/// * `stride_w` — Horizontal convolution stride (must be >= 1).
/// * `padding_h` — Zero-padding applied to top and bottom.
/// * `padding_w` — Zero-padding applied to left and right.
/// * `dilation_h` — Vertical spacing between kernel elements (must be >= 1).
/// * `dilation_w` — Horizontal spacing between kernel elements (must be >= 1).
/// * `groups` — Number of input channel groups (must be >= 1).
/// * `has_bias` — Whether to include a bias input node.
///
/// # Errors
///
/// Returns `TensorIRError` if any parameter is zero when it must be >= 1,
/// if `in_channels` is not divisible by `groups`, if arithmetic overflows,
/// or if the effective padded input is smaller than the effective kernel.
#[allow(clippy::too_many_arguments)]
pub fn build_conv2d_full(
    name: &str,
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    in_height: usize,
    in_width: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    groups: usize,
    has_bias: bool,
) -> Result<TensorKernelDef, TensorIRError> {
    if stride_h == 0 || stride_w == 0 {
        return Err(TensorIRConvError::Conv2dZeroStride { stride_h, stride_w }.into());
    }
    if kernel_h == 0 || kernel_w == 0 {
        return Err(TensorIRConvError::Conv2dZeroKernelSize { kernel_h, kernel_w }.into());
    }
    if dilation_h == 0 || dilation_w == 0 {
        return Err(TensorIRConvError::Conv2dZeroDilation {
            dilation_h,
            dilation_w,
        }
        .into());
    }
    if groups == 0 {
        return Err(TensorIRConvError::Conv2dZeroGroups.into());
    }
    if !in_channels.is_multiple_of(groups) {
        return Err(TensorIRConvError::Conv2dGroupsChannelMismatch {
            in_channels,
            groups,
        }
        .into());
    }
    if !out_channels.is_multiple_of(groups) {
        return Err(TensorIRConvError::Conv2dGroupsOutputMismatch {
            out_channels,
            groups,
        }
        .into());
    }

    // effective_kernel_h = dilation_h * (kernel_h - 1) + 1
    let eff_kh = dilation_h
        .checked_mul(kernel_h - 1)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| {
            TensorIRError::from(TensorIRConvError::Conv2dArithmeticOverflow {
                context: format!(
                    "effective_kernel_h: dilation_h={dilation_h} * (kernel_h={kernel_h} - 1) + 1"
                ),
            })
        })?;

    let eff_kw = dilation_w
        .checked_mul(kernel_w - 1)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| {
            TensorIRError::from(TensorIRConvError::Conv2dArithmeticOverflow {
                context: format!(
                    "effective_kernel_w: dilation_w={dilation_w} * (kernel_w={kernel_w} - 1) + 1"
                ),
            })
        })?;

    // padded_h = in_height + 2 * padding_h
    let padded_h = padding_h
        .checked_mul(2)
        .and_then(|v| v.checked_add(in_height))
        .ok_or_else(|| {
            TensorIRError::from(TensorIRConvError::Conv2dArithmeticOverflow {
                context: format!("padded_h: in_height={in_height} + 2 * padding_h={padding_h}"),
            })
        })?;

    let padded_w = padding_w
        .checked_mul(2)
        .and_then(|v| v.checked_add(in_width))
        .ok_or_else(|| {
            TensorIRError::from(TensorIRConvError::Conv2dArithmeticOverflow {
                context: format!("padded_w: in_width={in_width} + 2 * padding_w={padding_w}"),
            })
        })?;

    if padded_h < eff_kh {
        return Err(TensorIRConvError::Conv2dArithmeticOverflow {
            context: format!("out_h: padded_h={padded_h} < eff_kh={eff_kh}"),
        }
        .into());
    }
    if padded_w < eff_kw {
        return Err(TensorIRConvError::Conv2dArithmeticOverflow {
            context: format!("out_w: padded_w={padded_w} < eff_kw={eff_kw}"),
        }
        .into());
    }

    // out_h = (padded_h - eff_kh) / stride_h + 1
    let out_h = (padded_h - eff_kh) / stride_h + 1;
    let out_w = (padded_w - eff_kw) / stride_w + 1;
    let weight_in_channels = in_channels / groups;

    let mut nodes = vec![
        // %0 = input data: [in_channels, in_height, in_width]
        TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: crate::input_names::DATA.into(),
                shape: vec![in_channels, in_height, in_width],
            },
            vec![in_channels, in_height, in_width],
        ),
        // %1 = input weight: [out_channels, in_channels/groups, kernel_h, kernel_w]
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Input {
                name: "weight".into(),
                shape: vec![out_channels, weight_in_channels, kernel_h, kernel_w],
            },
            vec![out_channels, weight_in_channels, kernel_h, kernel_w],
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
        TensorOpKind::Conv2d {
            input: TensorNodeId::new(0),
            weight: TensorNodeId::new(1),
            bias: bias_node,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups,
        },
        vec![out_channels, out_h, out_w],
    ));

    Ok(TensorKernelDef::new(name, nodes, conv_id))
}

#[cfg(test)]
#[path = "conv2d_tests.rs"]
mod tests;

#[cfg(kani)]
#[path = "conv2d_kani.rs"]
mod kani_proofs;
