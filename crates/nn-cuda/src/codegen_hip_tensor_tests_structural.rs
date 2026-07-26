// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for HIP structural op emission (reduce, broadcast, narrow, transpose, concat).

use crate::codegen_hip_tensor_emit_structural::*;
use nn_dsl::{BroadcastAlignment, ReduceOp, ScalarType};

// --- Reduce tests ---

#[test]
fn test_reduce_sum_f32() {
    let src = emit_reduce_kernel("reduce_sum", ReduceOp::Sum, ScalarType::F32).unwrap();
    assert!(src.contains("extern \"C\" __global__ void reduce_sum"));
    assert!(src.contains("__shared__"));
    assert!(src.contains("__syncthreads()"));
    assert!(src.contains("partial = partial + val"));
    assert!(!src.contains("HUGE_VALF"));
}

#[test]
fn test_reduce_mean_f32() {
    let src = emit_reduce_kernel("reduce_mean", ReduceOp::Mean, ScalarType::F32).unwrap();
    assert!(src.contains("/ (float)reduce_dim"));
}

#[test]
fn test_reduce_max_f32() {
    let src = emit_reduce_kernel("reduce_max", ReduceOp::Max, ScalarType::F32).unwrap();
    assert!(src.contains("-HUGE_VALF"));
    assert!(src.contains("fmaxf"));
}

#[test]
fn test_reduce_min_f32() {
    let src = emit_reduce_kernel("reduce_min", ReduceOp::Min, ScalarType::F32).unwrap();
    assert!(src.contains("HUGE_VALF"));
    assert!(src.contains("fminf"));
}

#[test]
fn test_reduce_sum_f16_casts() {
    let src = emit_reduce_kernel("reduce_sum_f16", ReduceOp::Sum, ScalarType::F16).unwrap();
    // Should accumulate in float, cast loads.
    assert!(src.contains("float partial"));
    assert!(src.contains("(float)input"));
    // Should cast output back to half.
    assert!(src.contains("(half)"));
}

// --- Broadcast tests ---

#[test]
fn test_broadcast_right_aligned() {
    let src = emit_broadcast_kernel(
        "bcast_right",
        ScalarType::F32,
        &[4],
        &[2, 3, 4],
        BroadcastAlignment::Right,
    )
    .unwrap();
    assert!(src.contains("extern \"C\" __global__ void bcast_right"));
    assert!(src.contains("in_idx"));
    assert!(src.contains("remainder"));
}

#[test]
fn test_broadcast_left_aligned() {
    let src = emit_broadcast_kernel(
        "bcast_left",
        ScalarType::F32,
        &[2, 3],
        &[2, 3, 4],
        BroadcastAlignment::Left,
    )
    .unwrap();
    assert!(src.contains("extern \"C\" __global__ void bcast_left"));
    // Left-aligned: input dims [2,3] map to output dims [2,3,4] at prefix.
    assert!(src.contains("coord_0"));
    assert!(src.contains("coord_1"));
}

#[test]
fn test_broadcast_scalar_to_tensor() {
    let src = emit_broadcast_kernel(
        "bcast_scalar",
        ScalarType::F32,
        &[1],
        &[4, 8],
        BroadcastAlignment::Right,
    )
    .unwrap();
    // Input dim is 1, so no `in_idx += coord` line for it (size-1 dims are skipped).
    assert!(src.contains("output[tid] = input[in_idx]"));
}

// --- Narrow tests ---

#[test]
fn test_narrow_axis0() {
    let src = emit_narrow_kernel("narrow_0", ScalarType::F32, &[10, 8], 0, 2, 5).unwrap();
    assert!(src.contains("extern \"C\" __global__ void narrow_0"));
    // Axis 0 offset by start=2.
    assert!(src.contains("c0 + 2"));
}

#[test]
fn test_narrow_axis1() {
    let src = emit_narrow_kernel("narrow_1", ScalarType::F32, &[4, 16, 8], 1, 4, 8).unwrap();
    assert!(src.contains("c1 + 4"));
}

#[test]
fn test_narrow_axis_out_of_bounds() {
    let result = emit_narrow_kernel("bad", ScalarType::F32, &[4, 8], 2, 0, 4);
    assert!(result.is_err());
}

// --- Transpose tests ---

#[test]
fn test_transpose_2d() {
    let src = emit_transpose_kernel("transpose_2d", ScalarType::F32, &[3, 5], &[1, 0]).unwrap();
    assert!(src.contains("extern \"C\" __global__ void transpose_2d"));
    // For [3,5] with axes [1,0]: output shape is [5,3].
    // Output strides: [3, 1]. Input strides: [5, 1].
    // c0 maps to input axis 1 (stride 1), c1 maps to input axis 0 (stride 5).
    assert!(src.contains("in_idx += c0 * 1"));
    assert!(src.contains("in_idx += c1 * 5"));
}

#[test]
fn test_transpose_3d_permute() {
    let src =
        emit_transpose_kernel("transpose_3d", ScalarType::F32, &[2, 3, 4], &[0, 2, 1]).unwrap();
    assert!(src.contains("extern \"C\" __global__ void transpose_3d"));
    // axes [0,2,1]: output shape [2,4,3].
    // Input strides: [12, 4, 1]. Output strides: [12, 3, 1].
    // c0 -> axis 0 (stride 12), c1 -> axis 2 (stride 1), c2 -> axis 1 (stride 4).
    assert!(src.contains("in_idx += c0 * 12"));
    assert!(src.contains("in_idx += c1 * 1"));
    assert!(src.contains("in_idx += c2 * 4"));
}

// --- Concat tests ---

#[test]
fn test_concat_two_inputs_axis0() {
    let src = emit_concat_kernel("concat_a0", ScalarType::F32, &[3, 4], &[3, 5], 0).unwrap();
    assert!(src.contains("extern \"C\" __global__ void concat_a0"));
    assert!(src.contains("input0"));
    assert!(src.contains("input1"));
    assert!(src.contains("which_input"));
}

#[test]
fn test_concat_three_inputs_axis1() {
    let src = emit_concat_kernel("concat_a1", ScalarType::F32, &[2, 4, 8], &[4, 6, 3], 1).unwrap();
    assert!(src.contains("input0"));
    assert!(src.contains("input1"));
    assert!(src.contains("input2"));
    assert!(src.contains("which_input == 1"));
}

#[test]
fn test_concat_single_input() {
    let src = emit_concat_kernel("concat_one", ScalarType::F32, &[4, 8], &[4], 0).unwrap();
    // Single input: no which_input conditional.
    assert!(src.contains("input0[in_idx]"));
    assert!(!src.contains("which_input"));
}

#[test]
fn test_concat_empty_inputs() {
    let result = emit_concat_kernel("bad", ScalarType::F32, &[4, 8], &[], 0);
    assert!(result.is_err());
}

#[test]
fn test_concat_axis_out_of_bounds() {
    let result = emit_concat_kernel("bad", ScalarType::F32, &[4, 8], &[4], 2);
    assert!(result.is_err());
}
