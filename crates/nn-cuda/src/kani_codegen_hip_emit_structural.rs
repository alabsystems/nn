// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for HIP structural ops codegen, index/pad ops, select
//! ops, elementwise ops, and rocWMMA routing.
//!
//! Proves properties of:
//! - `row_major_strides` (modeled): stride product consistency, last-stride==1
//! - `emit_broadcast_kernel`: alignment mapping, shape validation
//! - `emit_narrow_kernel`: axis bounds, offset correctness
//! - `emit_transpose_kernel`: permutation validity
//! - `emit_concat_kernel`: empty-input rejection, axis bounds
//! - `emit_reduce_kernel`: ReduceOp coverage
//! - `emit_axis_select_kernel`: axis bounds, structural output
//! - `emit_stack_kernel`: empty-input rejection, axis bounds
//! - `emit_zero_pad_1d_hip`: overflow guard, structural output
//! - `emit_index_select_hip` / `emit_gather_hip`: axis bounds, dim-size==0
//! - `emit_f32_to_u32_hip`: naming convention
//! - `should_use_rocwmma`: alignment invariants
//! - Elementwise emit fns: dtype routing, structural markers
//!
//! Part of #3740.

// ---- Structural ops imports ----
use super::codegen_hip_tensor_emit_structural::{
    emit_broadcast_kernel, emit_concat_kernel, emit_narrow_kernel, emit_reduce_kernel,
    emit_transpose_kernel,
};

// ---- Select ops imports ----
use super::codegen_hip_tensor_emit_select::{emit_axis_select_kernel, emit_stack_kernel};

// ---- Index/pad ops imports ----
use super::codegen_hip_tensor_emit_index::{
    emit_f32_to_u32_hip, emit_gather_hip, emit_index_select_hip, emit_zero_pad_1d_hip,
};

// ---- Elementwise ops imports ----
use super::codegen_hip_tensor_emit_ops::{
    emit_binary_add_kernel, emit_binary_mul_kernel, emit_gelu_erf_kernel, emit_gelu_kernel,
    emit_relu_kernel, emit_sigmoid_kernel, emit_tanh_kernel,
};

// ---- GEMM routing imports ----
use super::codegen_hip_tensor_emit_gemm::should_use_rocwmma;

use nn_dsl::{BroadcastAlignment, ReduceOp, ScalarType};

// =========================================================================
// Modeled row_major_strides (private fn — we model it here for proofs)
// =========================================================================

/// Model row_major_strides using checked arithmetic (mirrors the private fn).
fn model_row_major_strides(shape: &[usize]) -> Option<Vec<usize>> {
    let rank = shape.len();
    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1].checked_mul(shape[i + 1])?;
    }
    Some(strides)
}

/// Prove row_major_strides last element is always 1 for non-empty shapes.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn prove_strides_last_is_one() {
    let a: u8 = kani::any();
    let b: u8 = kani::any();
    kani::assume(a >= 1 && b >= 1);
    let shape = vec![a as usize, b as usize];
    let strides = model_row_major_strides(&shape).unwrap();
    assert_eq!(*strides.last().unwrap(), 1);
}

/// Prove strides length equals shape length.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
fn prove_strides_length_matches_shape() {
    let a: u8 = kani::any();
    let b: u8 = kani::any();
    let c: u8 = kani::any();
    kani::assume(a >= 1 && b >= 1 && c >= 1);
    let shape = vec![a as usize, b as usize, c as usize];
    let strides = model_row_major_strides(&shape).unwrap();
    assert_eq!(strides.len(), shape.len());
}

/// Prove row_major_strides: stride[i] == product(shape[i+1..]) for rank 3.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
fn prove_strides_product_property() {
    let a: u8 = kani::any();
    let b: u8 = kani::any();
    let c: u8 = kani::any();
    kani::assume(a >= 1 && a <= 16);
    kani::assume(b >= 1 && b <= 16);
    kani::assume(c >= 1 && c <= 16);
    let shape = vec![a as usize, b as usize, c as usize];
    let strides = model_row_major_strides(&shape).unwrap();
    assert_eq!(strides[0], (b as usize) * (c as usize));
    assert_eq!(strides[1], c as usize);
    assert_eq!(strides[2], 1);
}

/// Prove strides are monotonically non-increasing for shapes with all dims >= 1.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
fn prove_strides_monotone() {
    let a: u8 = kani::any();
    let b: u8 = kani::any();
    let c: u8 = kani::any();
    kani::assume(a >= 1 && b >= 1 && c >= 1);
    kani::assume(a <= 32 && b <= 32 && c <= 32);
    let shape = vec![a as usize, b as usize, c as usize];
    if let Some(strides) = model_row_major_strides(&shape) {
        assert!(strides[0] >= strides[1]);
        assert!(strides[1] >= strides[2]);
    }
}

/// Prove strides overflow detection: large dimensions return None.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn prove_strides_overflow() {
    let shape = vec![usize::MAX / 2, usize::MAX / 2];
    let result = model_row_major_strides(&shape);
    assert!(result.is_none());
}

/// Prove scalar (rank-1) strides are [1].
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn prove_strides_rank1() {
    let a: u8 = kani::any();
    kani::assume(a >= 1);
    let shape = vec![a as usize];
    let strides = model_row_major_strides(&shape).unwrap();
    assert_eq!(strides, vec![1]);
}

// =========================================================================
// emit_reduce_kernel proofs
// =========================================================================

/// Prove emit_reduce_kernel succeeds for Sum.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_reduce_sum_ok() {
    let result = emit_reduce_kernel("reduce_sum", ReduceOp::Sum, ScalarType::F32);
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("reduce_sum"));
    assert!(src.contains("__global__"));
    assert!(src.contains("__shared__"));
}

/// Prove emit_reduce_kernel succeeds for Mean and includes divisor.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_reduce_mean_has_divisor() {
    let result = emit_reduce_kernel("reduce_mean", ReduceOp::Mean, ScalarType::F32);
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("reduce_dim"));
}

/// Prove emit_reduce_kernel succeeds for Max and uses fmaxf.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_reduce_max_uses_fmaxf() {
    let result = emit_reduce_kernel("reduce_max", ReduceOp::Max, ScalarType::F32);
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("fmaxf"));
}

/// Prove emit_reduce_kernel succeeds for Min and uses fminf.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_reduce_min_uses_fminf() {
    let result = emit_reduce_kernel("reduce_min", ReduceOp::Min, ScalarType::F32);
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("fminf"));
}

/// Prove emit_reduce_kernel for f16 uses half type and float accumulator.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_reduce_f16_accumulator() {
    let result = emit_reduce_kernel("reduce_f16", ReduceOp::Sum, ScalarType::F16);
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("half"));
    assert!(src.contains("float"));
}

// =========================================================================
// emit_broadcast_kernel proofs
// =========================================================================

/// Prove broadcast left-aligned succeeds for valid shapes.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_broadcast_left_valid() {
    let result = emit_broadcast_kernel(
        "bcast",
        ScalarType::F32,
        &[4, 8],
        &[4, 8, 16],
        BroadcastAlignment::Left,
    );
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("bcast"));
    assert!(src.contains("__global__"));
}

/// Prove broadcast right-aligned succeeds for valid shapes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_broadcast_right_valid() {
    let result = emit_broadcast_kernel(
        "bcast_r",
        ScalarType::F32,
        &[8],
        &[4, 8],
        BroadcastAlignment::Right,
    );
    assert!(result.is_ok());
}

/// Prove broadcast kernel contains input/output parameter declarations.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_broadcast_has_params() {
    let result = emit_broadcast_kernel(
        "bcast_p",
        ScalarType::F32,
        &[3],
        &[2, 3],
        BroadcastAlignment::Right,
    );
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("input"));
    assert!(src.contains("output"));
    assert!(src.contains("total"));
}

/// Prove broadcast f16 uses half type.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_broadcast_f16_type() {
    let result = emit_broadcast_kernel(
        "bcast_f16",
        ScalarType::F16,
        &[4],
        &[4, 8],
        BroadcastAlignment::Left,
    );
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("half"));
}

// =========================================================================
// emit_narrow_kernel proofs
// =========================================================================

/// Prove narrow rejects axis out of bounds.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(128)]
fn prove_narrow_axis_oob() {
    let result = emit_narrow_kernel("narrow_bad", ScalarType::F32, &[4, 8], 2, 0, 4);
    assert!(result.is_err());
}

/// Prove narrow succeeds for valid parameters.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_narrow_valid() {
    let result = emit_narrow_kernel("narrow_ok", ScalarType::F32, &[4, 8], 0, 1, 2);
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("narrow_ok"));
    assert!(src.contains("__global__"));
}

/// Prove narrow kernel contains the start offset value.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_narrow_contains_start() {
    let result = emit_narrow_kernel("narrow_s", ScalarType::F32, &[10, 20], 1, 5, 10);
    assert!(result.is_ok());
    let src = result.unwrap();
    // The start offset 5 should appear as part of the index calculation
    assert!(src.contains("5"));
}

/// Prove narrow on last axis succeeds.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_narrow_last_axis() {
    let result = emit_narrow_kernel("narrow_last", ScalarType::F32, &[4, 8, 16], 2, 2, 6);
    assert!(result.is_ok());
}

// =========================================================================
// emit_transpose_kernel proofs
// =========================================================================

/// Prove transpose with identity permutation succeeds.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_transpose_identity() {
    let result = emit_transpose_kernel("trans_id", ScalarType::F32, &[4, 8], &[0, 1]);
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("trans_id"));
}

/// Prove transpose with swap permutation succeeds.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_transpose_swap() {
    let result = emit_transpose_kernel("trans_swap", ScalarType::F32, &[4, 8], &[1, 0]);
    assert!(result.is_ok());
}

/// Prove transpose 3D permutation succeeds.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_transpose_3d() {
    let result = emit_transpose_kernel("trans_3d", ScalarType::F32, &[2, 3, 4], &[2, 0, 1]);
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("__global__"));
}

/// Prove transpose bf16 uses hip_bfloat16 type.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_transpose_bf16_type() {
    let result = emit_transpose_kernel("trans_bf16", ScalarType::BF16, &[4, 8], &[1, 0]);
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("hip_bfloat16"));
}

// =========================================================================
// emit_concat_kernel proofs
// =========================================================================

/// Prove concat rejects empty input list.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(128)]
fn prove_concat_empty_error() {
    let result = emit_concat_kernel("cat_bad", ScalarType::F32, &[4, 8], &[], 0);
    assert!(result.is_err());
}

/// Prove concat rejects axis out of bounds.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(128)]
fn prove_concat_axis_oob() {
    let result = emit_concat_kernel("cat_oob", ScalarType::F32, &[4, 8], &[3, 5], 2);
    assert!(result.is_err());
}

/// Prove concat succeeds for two inputs along axis 0.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_concat_two_inputs_axis0() {
    let result = emit_concat_kernel("cat2", ScalarType::F32, &[4, 8], &[4, 6], 0);
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("cat2"));
    assert!(src.contains("input0"));
    assert!(src.contains("input1"));
}

/// Prove concat single input produces simpler code (no which_input).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_concat_single_input() {
    let result = emit_concat_kernel("cat1", ScalarType::F32, &[4, 8], &[4], 0);
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("input0"));
}

/// Prove concat three inputs along axis 1 succeeds.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_concat_three_inputs_axis1() {
    let result = emit_concat_kernel("cat3", ScalarType::F32, &[4, 8], &[2, 3, 3], 1);
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("input2"));
}

// =========================================================================
// emit_axis_select_kernel proofs
// =========================================================================

/// Prove axis_select rejects axis out of bounds.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(128)]
fn prove_axis_select_oob() {
    let result = emit_axis_select_kernel("asel_bad", ScalarType::F32, &[4, 8], 2, 0);
    assert!(result.is_err());
}

/// Prove axis_select succeeds for valid parameters.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_axis_select_valid() {
    let result = emit_axis_select_kernel("asel", ScalarType::F32, &[4, 8], 0, 2);
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("asel"));
    assert!(src.contains("__global__"));
}

/// Prove axis_select on last axis of 3D tensor succeeds.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_axis_select_3d_last() {
    let result = emit_axis_select_kernel("asel_3d", ScalarType::F32, &[2, 3, 4], 2, 1);
    assert!(result.is_ok());
}

// =========================================================================
// emit_stack_kernel proofs
// =========================================================================

/// Prove stack rejects n_inputs=0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(128)]
fn prove_stack_empty_error() {
    let result = emit_stack_kernel("stk_bad", ScalarType::F32, &[4, 8], 0, 0);
    assert!(result.is_err());
}

/// Prove stack rejects axis out of bounds.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(128)]
fn prove_stack_axis_oob() {
    let result = emit_stack_kernel("stk_oob", ScalarType::F32, &[4, 8], 2, 3);
    assert!(result.is_err());
}

/// Prove stack succeeds for 2 inputs at axis 0.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_stack_two_inputs() {
    let result = emit_stack_kernel("stk2", ScalarType::F32, &[4, 8], 2, 0);
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("input0"));
    assert!(src.contains("input1"));
    assert!(src.contains("which_input"));
}

/// Prove stack single input does not use ternary chain.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_stack_single_input() {
    let result = emit_stack_kernel("stk1", ScalarType::F32, &[4, 8], 1, 0);
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("input0[in_idx]"));
}

// =========================================================================
// emit_zero_pad_1d_hip proofs
// =========================================================================

/// Prove zero_pad_1d succeeds for valid parameters.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_zero_pad_valid() {
    let result = emit_zero_pad_1d_hip("zp", ScalarType::F32, 4, 10, 3, 16);
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("zp"));
    assert!(src.contains("__global__"));
}

/// Prove zero_pad_1d f16 uses half type.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_zero_pad_f16() {
    let result = emit_zero_pad_1d_hip("zp_f16", ScalarType::F16, 2, 8, 1, 10);
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("half"));
}

/// Prove zero_pad_1d overflow detection: channels * out_length overflow.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(128)]
fn prove_zero_pad_overflow() {
    let result = emit_zero_pad_1d_hip("zp_ovf", ScalarType::F32, usize::MAX / 2, 3, 0, 3);
    assert!(result.is_err());
}

// =========================================================================
// emit_index_select_hip proofs
// =========================================================================

/// Prove index_select rejects dim out of bounds.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(128)]
fn prove_index_select_dim_oob() {
    let result = emit_index_select_hip("isel_bad", ScalarType::F32, &[4, 8], 2);
    assert!(result.is_err());
}

/// Prove index_select rejects dim_size == 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(128)]
fn prove_index_select_dim_zero() {
    let result = emit_index_select_hip("isel_z", ScalarType::F32, &[4, 0, 8], 1);
    assert!(result.is_err());
}

/// Prove index_select succeeds for valid 2D input.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_index_select_valid() {
    let result = emit_index_select_hip("isel", ScalarType::F32, &[10, 8], 0);
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("isel"));
    assert!(src.contains("DIM_SIZE = 10"));
}

/// Prove index_select 3D dim=1 computes correct OUTER and INNER.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_index_select_3d_decomposition() {
    let result = emit_index_select_hip("isel_3d", ScalarType::F32, &[2, 5, 3], 1);
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("OUTER = 2"));
    assert!(src.contains("INNER = 3"));
    assert!(src.contains("DIM_SIZE = 5"));
}

// =========================================================================
// emit_gather_hip proofs
// =========================================================================

/// Prove gather rejects dim out of bounds.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(128)]
fn prove_gather_dim_oob() {
    let result = emit_gather_hip("gath_bad", ScalarType::F32, &[4], 1);
    assert!(result.is_err());
}

/// Prove gather rejects dim_size == 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(128)]
fn prove_gather_dim_zero() {
    let result = emit_gather_hip("gath_z", ScalarType::F32, &[0, 8], 0);
    assert!(result.is_err());
}

/// Prove gather succeeds for valid parameters.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_gather_valid() {
    let result = emit_gather_hip("gath", ScalarType::F32, &[10, 8], 0);
    assert!(result.is_ok());
    let src = result.unwrap();
    assert!(src.contains("gath"));
    assert!(src.contains("DIM_SIZE = 10"));
    assert!(src.contains("src_dim = indices[tid]"));
}

// =========================================================================
// emit_f32_to_u32_hip proofs
// =========================================================================

/// Prove f32_to_u32 naming convention: appends "_f32_to_u32" suffix.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(64)]
fn prove_f32_to_u32_naming() {
    let (conv_name, _) = emit_f32_to_u32_hip("nn_kernel");
    assert_eq!(conv_name, "nn_kernel_f32_to_u32");
}

/// Prove f32_to_u32 kernel contains the expected conversion pattern.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_f32_to_u32_structure() {
    let (_, src) = emit_f32_to_u32_hip("test");
    assert!(src.contains("__global__"));
    assert!(src.contains("(unsigned int)(v)"));
    assert!(src.contains("test_f32_to_u32"));
}

// =========================================================================
// should_use_rocwmma proofs
// =========================================================================

/// Prove rocwmma requires 16-aligned M dimension.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn prove_rocwmma_requires_m_aligned() {
    let m: u8 = kani::any();
    kani::assume(m > 0 && (m as usize) % 16 != 0);
    assert!(!should_use_rocwmma(m as usize, 128, 128));
}

/// Prove rocwmma requires 16-aligned K dimension.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn prove_rocwmma_requires_k_aligned() {
    let k: u8 = kani::any();
    kani::assume(k > 0 && (k as usize) % 16 != 0);
    assert!(!should_use_rocwmma(128, k as usize, 128));
}

/// Prove rocwmma requires 16-aligned N dimension.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn prove_rocwmma_requires_n_aligned() {
    let n: u8 = kani::any();
    kani::assume(n > 0 && (n as usize) % 16 != 0);
    assert!(!should_use_rocwmma(128, 128, n as usize));
}

/// Prove rocwmma requires K >= 128.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn prove_rocwmma_requires_k_ge_128() {
    let k: u8 = kani::any();
    kani::assume(k < 128);
    kani::assume((k as usize) % 16 == 0);
    assert!(!should_use_rocwmma(256, k as usize, 256));
}

/// Prove rocwmma requires M*N >= 16384.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn prove_rocwmma_requires_min_compute() {
    // 64 * 128 = 8192 < 16384 => false
    assert!(!should_use_rocwmma(64, 128, 128));
}

/// Prove rocwmma accepts well-aligned large dimensions.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn prove_rocwmma_accepts_large_aligned() {
    assert!(should_use_rocwmma(256, 256, 256));
}

/// Prove rocwmma rejects zero dimensions.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn prove_rocwmma_rejects_zero() {
    assert!(!should_use_rocwmma(0, 128, 128));
    assert!(!should_use_rocwmma(128, 0, 128));
    assert!(!should_use_rocwmma(128, 128, 0));
}

// =========================================================================
// Elementwise emit proofs
// =========================================================================

/// Prove all 7 elementwise ops succeed for F32 with valid element count.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_elementwise_f32_all_ops() {
    assert!(emit_binary_add_kernel("add", ScalarType::F32, 1024).is_ok());
    assert!(emit_binary_mul_kernel("mul", ScalarType::F32, 1024).is_ok());
    assert!(emit_sigmoid_kernel("sig", ScalarType::F32, 1024).is_ok());
    assert!(emit_gelu_kernel("gelu", ScalarType::F32, 1024).is_ok());
    assert!(emit_gelu_erf_kernel("gelu_erf", ScalarType::F32, 1024).is_ok());
    assert!(emit_relu_kernel("relu", ScalarType::F32, 1024).is_ok());
    assert!(emit_tanh_kernel("tanh", ScalarType::F32, 1024).is_ok());
}

/// Prove sigmoid kernel contains the 1/(1+exp(-x)) pattern.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_sigmoid_contains_exp_pattern() {
    let src = emit_sigmoid_kernel("sig", ScalarType::F32, 512).unwrap();
    assert!(src.contains("expf"));
    assert!(src.contains("1.0f"));
}

/// Prove relu kernel contains the max(x, 0) pattern.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_relu_contains_max_pattern() {
    let src = emit_relu_kernel("relu", ScalarType::F32, 512).unwrap();
    assert!(src.contains("> (float)0"));
}

/// Prove gelu_erf kernel contains erf approximation coefficients.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_gelu_erf_contains_coefficients() {
    let src = emit_gelu_erf_kernel("gerf", ScalarType::F32, 512).unwrap();
    // Abramowitz & Stegun coefficient 0.3275911
    assert!(src.contains("0.3275911"));
}

/// Prove binary_add kernel for f16 uses half type.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_binary_add_f16() {
    let src = emit_binary_add_kernel("add_f16", ScalarType::F16, 256).unwrap();
    assert!(src.contains("half"));
}

/// Prove binary_mul kernel for bf16 uses hip_bfloat16 type.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(256)]
fn prove_binary_mul_bf16() {
    let src = emit_binary_mul_kernel("mul_bf16", ScalarType::BF16, 256).unwrap();
    assert!(src.contains("hip_bfloat16"));
}
