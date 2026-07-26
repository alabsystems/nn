// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-op dispatch step builders for `build_dispatch_plan`.
//!
//! Extracted from `codegen_msl_tensor.rs` to keep the dispatch planner under
//! 500 lines (#753). Each function takes a tensor node and its context, and
//! returns the corresponding `DispatchStep`.

use crate::ir::ScalarType;
use crate::tensor_ir::{ReduceOp, TensorKernelDef, TensorNodeId};

use super::{
    node_shape, shape_total, DispatchStep, SimdgroupLinearParams, SimdgroupMatMulParams,
    TensorMSLCodegenError, TiledLinearParams, TiledMatMulParams,
};

#[path = "codegen_msl_tensor_ops_conv.rs"]
mod conv;
pub(super) use conv::{build_conv1d_step, build_conv2d_step, build_conv_transpose_1d_step};

/// Build a `DispatchStep::Reduce` from a Reduce node.
pub(super) fn build_reduce_step(
    effective: &TensorKernelDef,
    node_id: TensorNodeId,
    op: &ReduceOp,
    input: &TensorNodeId,
    axis: usize,
    keepdim: bool,
    dtype: ScalarType,
) -> Result<DispatchStep, TensorMSLCodegenError> {
    let input_shape = node_shape(effective, *input)?;
    if axis + 1 != input_shape.len() {
        return Err(TensorMSLCodegenError::NonLastAxisReduce {
            node_id,
            axis,
            shape: input_shape.to_vec(),
        });
    }
    let reduce_dim = input_shape[axis];
    let outer_size: usize = input_shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != axis)
        .try_fold(1usize, |acc, (_, &d)| acc.checked_mul(d))
        .ok_or_else(|| TensorMSLCodegenError::ShapeProductOverflow {
            shape: input_shape.to_vec(),
        })?;
    let op_name = match op {
        ReduceOp::Sum => "sum",
        ReduceOp::Mean => "mean",
        ReduceOp::Max => "max",
        ReduceOp::Min => "min",
    };
    Ok(DispatchStep::Reduce {
        kernel_name: format!("{}_reduce_{}_n{}", effective.name, op_name, node_id.index()),
        op: *op,
        dtype,
        input: *input,
        output: node_id,
        reduce_dim,
        outer_size,
        keepdim,
    })
}

/// Build a `DispatchStep::Softmax` from a Softmax node.
pub(super) fn build_softmax_step(
    effective: &TensorKernelDef,
    node_id: TensorNodeId,
    input: &TensorNodeId,
    axis: i32,
    dtype: ScalarType,
) -> Result<DispatchStep, TensorMSLCodegenError> {
    let input_shape = node_shape(effective, *input)?;
    let rank = input_shape.len();
    let rank_i32 = i32::try_from(rank)
        .map_err(|_| TensorMSLCodegenError::AxisOutOfBounds { axis: 0, rank })?;
    if axis < -rank_i32 || axis >= rank_i32 {
        // Report the resolved positive axis when possible, otherwise 0 as placeholder.
        let display_axis = if axis >= 0 { axis as usize } else { 0 };
        return Err(TensorMSLCodegenError::AxisOutOfBounds {
            axis: display_axis,
            rank,
        });
    }
    let resolved_axis = if axis < 0 {
        (rank_i32 + axis) as usize
    } else {
        axis as usize
    };
    let axis_size = input_shape[resolved_axis];
    let outer_size: usize = input_shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != resolved_axis)
        .try_fold(1usize, |acc, (_, &d)| acc.checked_mul(d))
        .ok_or_else(|| TensorMSLCodegenError::ShapeProductOverflow {
            shape: input_shape.to_vec(),
        })?;
    Ok(DispatchStep::Softmax {
        kernel_name: format!("{}_softmax_n{}", effective.name, node_id.index()),
        dtype,
        input: *input,
        output: node_id,
        axis: resolved_axis,
        axis_size,
        outer_size,
    })
}

/// Returns true if the simdgroup GEMM kernel should be used for these dims.
///
/// Mirrors `MetalDynBackend::should_use_simdgroup()` from nn-metal.
/// Criteria (from `designs/2026-03-08-simdgroup-matmul-dispatch-strategy.md`):
/// - All dimensions must be multiples of 8 (simdgroup_matrix requirement)
/// - M×N must be ≥ 16,384 (compute must dominate dispatch overhead)
/// - K must be ≥ 128 (amortize shared memory tiling cost)
///
/// Part of #2275.
pub(super) fn should_use_simdgroup(m: usize, k: usize, n: usize) -> bool {
    m.is_multiple_of(8) && k.is_multiple_of(8) && n.is_multiple_of(8) && m * n >= 16_384 && k >= 128
}

/// Returns true if the tiled GEMM kernel should be used for these dims.
///
/// Targets shapes large enough to fill at least one 16×16 threadgroup tile
/// but below simdgroup requirements. Part of #3230 (Gap 1).
pub(super) fn should_use_tiled(m: usize, k: usize, n: usize) -> bool {
    m >= 16 && n >= 16 && k >= 8
}

/// Build a `DispatchStep::Linear`, `SimdgroupLinear`, or `TiledLinear`.
///
/// Routes: simdgroup → tiled → naive. Part of #3230 (Gap 1).
pub(super) fn build_linear_step(
    effective: &TensorKernelDef,
    node_id: TensorNodeId,
    node_shape_out: &[usize],
    input: &TensorNodeId,
    weight: &TensorNodeId,
    bias: &Option<TensorNodeId>,
    dtype: ScalarType,
) -> Result<DispatchStep, TensorMSLCodegenError> {
    let input_shape = node_shape(effective, *input)?;
    let weight_shape = node_shape(effective, *weight)?;
    let in_features = weight_shape[1];
    let out_features = weight_shape[0];
    let batch_size: usize = input_shape[..input_shape.len() - 1]
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| TensorMSLCodegenError::ShapeProductOverflow {
            shape: input_shape.to_vec(),
        })?;
    // Route: M=batch_size, K=in_features, N=out_features.
    if should_use_simdgroup(batch_size, in_features, out_features) {
        return Ok(DispatchStep::SimdgroupLinear(SimdgroupLinearParams {
            kernel_name: format!("{}_simd_linear_n{}", effective.name, node_id.index()),
            dtype,
            input: *input,
            weight: *weight,
            bias: *bias,
            output: node_id,
            in_features,
            out_features,
            batch_size,
        }));
    }
    if should_use_tiled(batch_size, in_features, out_features) {
        return Ok(DispatchStep::TiledLinear(TiledLinearParams {
            kernel_name: format!("{}_tiled_linear_n{}", effective.name, node_id.index()),
            dtype,
            input: *input,
            weight: *weight,
            bias: *bias,
            output: node_id,
            in_features,
            out_features,
            batch_size,
        }));
    }
    let total_elements = shape_total(node_shape_out)?;
    Ok(DispatchStep::Linear {
        kernel_name: format!("{}_linear_n{}", effective.name, node_id.index()),
        dtype,
        input: *input,
        weight: *weight,
        bias: *bias,
        output: node_id,
        in_features,
        out_features,
        batch_size,
        total_elements,
    })
}

/// Build a `DispatchStep::MatMul`, `SimdgroupMatMul`, or `TiledMatMul`.
///
/// Routes: simdgroup → tiled → naive. Part of #3230 (Gap 1).
/// (all dims % 8, M×N ≥ 16384, K ≥ 128). Falls back to naive `MatMul`.
pub(super) fn build_matmul_step(
    effective: &TensorKernelDef,
    node_id: TensorNodeId,
    node_shape_out: &[usize],
    left: &TensorNodeId,
    right: &TensorNodeId,
    transpose_right: bool,
    scale: Option<f32>,
    dtype: ScalarType,
) -> Result<DispatchStep, TensorMSLCodegenError> {
    let left_shape = node_shape(effective, *left)?;
    let right_shape = node_shape(effective, *right)?;
    let m = left_shape[left_shape.len() - 2];
    let k = left_shape[left_shape.len() - 1];
    let n = if transpose_right {
        right_shape[right_shape.len() - 2]
    } else {
        right_shape[right_shape.len() - 1]
    };
    let batch_size: usize = left_shape[..left_shape.len() - 2]
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| TensorMSLCodegenError::ShapeProductOverflow {
            shape: left_shape.to_vec(),
        })?;
    // Detect when right tensor has fewer batch dims than left — the MSL kernel
    // must broadcast right (use offset 0) instead of indexing into non-existent batches.
    let right_batch_dims = &right_shape[..right_shape.len() - 2];
    let broadcast_right = right_batch_dims.is_empty()
        || (right_batch_dims.len() < left_shape.len() - 2)
        || right_batch_dims.iter().all(|&d| d == 1);
    // Route to simdgroup when shapes conform.
    if should_use_simdgroup(m, k, n) {
        return Ok(DispatchStep::SimdgroupMatMul(SimdgroupMatMulParams {
            kernel_name: format!("{}_simd_matmul_n{}", effective.name, node_id.index()),
            dtype,
            left: *left,
            right: *right,
            output: node_id,
            m,
            k,
            n,
            batch_size,
            transpose_right,
            broadcast_right,
            scale,
        }));
    }
    if should_use_tiled(m, k, n) {
        return Ok(DispatchStep::TiledMatMul(TiledMatMulParams {
            kernel_name: format!("{}_tiled_matmul_n{}", effective.name, node_id.index()),
            dtype,
            left: *left,
            right: *right,
            output: node_id,
            m,
            k,
            n,
            batch_size,
            transpose_right,
            broadcast_right,
            scale,
        }));
    }
    let total_elements = shape_total(node_shape_out)?;
    Ok(DispatchStep::MatMul {
        kernel_name: format!("{}_matmul_n{}", effective.name, node_id.index()),
        dtype,
        left: *left,
        right: *right,
        output: node_id,
        m,
        k,
        n,
        batch_size,
        transpose_right,
        broadcast_right,
        scale,
        total_elements,
    })
}

/// Build a `DispatchStep::ZeroPad1d` from a ZeroPad1d node.
pub(super) fn build_zero_pad_step(
    effective: &TensorKernelDef,
    node_id: TensorNodeId,
    input: &TensorNodeId,
    pad_left: usize,
    pad_right: usize,
    dtype: ScalarType,
) -> Result<DispatchStep, TensorMSLCodegenError> {
    let input_shape = node_shape(effective, *input)?;
    let in_length = input_shape[input_shape.len() - 1];
    let channels: usize = input_shape[..input_shape.len() - 1]
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| TensorMSLCodegenError::ShapeProductOverflow {
            shape: input_shape.to_vec(),
        })?;
    let out_length = in_length
        .checked_add(pad_left)
        .and_then(|v| v.checked_add(pad_right))
        .ok_or_else(|| TensorMSLCodegenError::ShapeProductOverflow {
            shape: vec![in_length, pad_left, pad_right],
        })?;
    Ok(DispatchStep::ZeroPad1d {
        kernel_name: format!("{}_zero_pad_1d_n{}", effective.name, node_id.index()),
        dtype,
        input: *input,
        output: node_id,
        channels,
        in_length,
        pad_left,
        out_length,
    })
}
