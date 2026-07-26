// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv dispatch step builders (Conv1d, Conv2d, ConvTranspose1d).
//!
//! Extracted from `codegen_msl_tensor_ops.rs` to keep that file under the
//! 450-line limit. Functions use `pub(in super::super)` — visible to the
//! grandparent `codegen_msl_tensor` module for re-export.

use crate::ir::ScalarType;
use crate::tensor_ir::{TensorKernelDef, TensorNodeId};

use super::super::{
    node_shape, shape_total, Conv1dParams, Conv2dParams, ConvTranspose1dParams, DispatchStep,
    TensorMSLCodegenError,
};

/// Build a `DispatchStep::Conv1d` from a Conv1d node.
#[allow(clippy::too_many_arguments)]
pub(in super::super) fn build_conv1d_step(
    effective: &TensorKernelDef,
    node_id: TensorNodeId,
    node_shape_out: &[usize],
    input: &TensorNodeId,
    weight: &TensorNodeId,
    bias: &Option<TensorNodeId>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    dtype: ScalarType,
) -> Result<DispatchStep, TensorMSLCodegenError> {
    let input_shape = node_shape(effective, *input)?;
    let weight_shape = node_shape(effective, *weight)?;
    let in_channels = input_shape[input_shape.len() - 2];
    let in_length = input_shape[input_shape.len() - 1];
    let out_channels = weight_shape[0];
    let kernel_size = weight_shape[2];
    let total_elements = shape_total(node_shape_out)?;
    Ok(DispatchStep::Conv1d(Conv1dParams {
        kernel_name: format!("{}_conv1d_n{}", effective.name, node_id.index()),
        dtype,
        input: *input,
        weight: *weight,
        bias: *bias,
        output: node_id,
        in_channels,
        out_channels,
        kernel_size,
        in_length,
        total_elements,
        stride,
        padding,
        dilation,
        groups,
    }))
}

/// Build a `DispatchStep::Conv2d` from a Conv2d node.
#[allow(clippy::too_many_arguments)]
pub(in super::super) fn build_conv2d_step(
    effective: &TensorKernelDef,
    node_id: TensorNodeId,
    node_shape_out: &[usize],
    input: &TensorNodeId,
    weight: &TensorNodeId,
    bias: &Option<TensorNodeId>,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    groups: usize,
    dtype: ScalarType,
) -> Result<DispatchStep, TensorMSLCodegenError> {
    let input_shape = node_shape(effective, *input)?;
    let weight_shape = node_shape(effective, *weight)?;
    let in_channels = input_shape[input_shape.len() - 3];
    let in_height = input_shape[input_shape.len() - 2];
    let in_width = input_shape[input_shape.len() - 1];
    let out_channels = weight_shape[0];
    let kernel_h = weight_shape[2];
    let kernel_w = weight_shape[3];
    let total_elements = shape_total(node_shape_out)?;
    Ok(DispatchStep::Conv2d(Conv2dParams {
        kernel_name: format!("{}_conv2d_n{}", effective.name, node_id.index()),
        dtype,
        input: *input,
        weight: *weight,
        bias: *bias,
        output: node_id,
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
    }))
}

/// Build a `DispatchStep::ConvTranspose1d` from a ConvTranspose1d node.
#[allow(clippy::too_many_arguments)]
pub(in super::super) fn build_conv_transpose_1d_step(
    effective: &TensorKernelDef,
    node_id: TensorNodeId,
    node_shape_out: &[usize],
    input: &TensorNodeId,
    weight: &TensorNodeId,
    bias: &Option<TensorNodeId>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    output_padding: usize,
    dtype: ScalarType,
) -> Result<DispatchStep, TensorMSLCodegenError> {
    let input_shape = node_shape(effective, *input)?;
    let weight_shape = node_shape(effective, *weight)?;
    let in_channels = input_shape[input_shape.len() - 2];
    let in_length = input_shape[input_shape.len() - 1];
    // weight_shape = [in_ch, out_ch_per_group, kernel_size]
    // Total out_channels = out_ch_per_group * groups.
    let out_channels = weight_shape[1].checked_mul(groups).ok_or_else(|| {
        TensorMSLCodegenError::ShapeProductOverflow {
            shape: weight_shape.to_vec(),
        }
    })?;
    let kernel_size = weight_shape[2];
    let total_elements = shape_total(node_shape_out)?;
    Ok(DispatchStep::ConvTranspose1d(ConvTranspose1dParams {
        kernel_name: format!("{}_conv_transpose_1d_n{}", effective.name, node_id.index()),
        dtype,
        input: *input,
        weight: *weight,
        bias: *bias,
        output: node_id,
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
    }))
}
