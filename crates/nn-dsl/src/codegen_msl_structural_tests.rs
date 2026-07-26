// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for MSL code generation of structural tensor operations.
//!
//! Extracted from `codegen_msl_structural.rs` via `#[path]` pattern (#571 AC2).

use super::*;
use crate::codegen_msl_structural_conv::{emit_conv1d_kernel, emit_conv_transpose_1d_kernel};

#[test]
fn test_axis_select_kernel_contains_kernel_attribute() {
    let msl =
        emit_axis_select_kernel("test_sel", ScalarType::F32, &[2, 3, 4], 1, 0).expect("codegen");
    assert!(msl.contains("[[kernel]]"), "must have kernel attribute");
    assert!(msl.contains("test_sel"), "must contain kernel name");
    assert!(msl.contains("in_idx"), "must compute input index");
}

#[test]
fn test_axis_select_kernel_last_axis_pair() {
    // RoPE case: select from [BH, S, D/2, 2] at axis=3, index=0
    let msl =
        emit_axis_select_kernel("rope_sel", ScalarType::F32, &[2, 4, 3, 2], 3, 0).expect("codegen");
    assert!(msl.contains("rope_sel"));
    // The fixed dimension (axis=3) should contribute `0 * 1` to in_idx.
    assert!(msl.contains("in_idx += 0 * 1"));
}

#[test]
fn test_stack_kernel_two_inputs() {
    let msl = emit_stack_kernel("test_stack", ScalarType::F32, &[2, 3], 2, 1).expect("codegen");
    assert!(msl.contains("[[kernel]]"));
    assert!(msl.contains("input0"));
    assert!(msl.contains("input1"));
    assert!(msl.contains("which_input"));
}

#[test]
fn test_stack_kernel_rope_pattern() {
    // RoPE case: stack 2 inputs of [BH, S, D/2] at axis=3 → [BH, S, D/2, 2]
    let msl = emit_stack_kernel("rope_stack", ScalarType::F32, &[2, 4, 3], 2, 3).expect("codegen");
    assert!(msl.contains("rope_stack"));
    assert!(msl.contains("buffer(0)"));
    assert!(msl.contains("buffer(1)"));
    assert!(msl.contains("buffer(2)")); // output
    assert!(msl.contains("buffer(3)")); // total
}

#[test]
fn test_row_major_strides() {
    assert_eq!(row_major_strides(&[2, 3, 4]).unwrap(), vec![12, 4, 1]);
    assert_eq!(row_major_strides(&[5]).unwrap(), vec![1]);
    assert_eq!(row_major_strides(&[]).unwrap(), Vec::<usize>::new());
}

#[test]
fn test_row_major_strides_overflow_returns_err() {
    // 3 dimensions: strides[1] = usize::MAX, strides[0] = usize::MAX * usize::MAX → overflow
    let huge = &[2, usize::MAX, usize::MAX];
    assert!(row_major_strides(huge).is_err());
}

/// emit_stack_kernel with n_inputs=0 must return EmptyStack error (#270).
#[test]
fn test_stack_kernel_zero_inputs_returns_err() {
    let result = emit_stack_kernel("empty", ScalarType::F32, &[2, 3], 0, 1);
    assert!(result.is_err(), "n_inputs=0 must fail");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("n_inputs=0"),
        "error should mention n_inputs=0, got: {err_msg}"
    );
}

/// Stack kernel with 29 inputs (31 total buffers) should be accepted (#290).
#[test]
fn test_stack_kernel_29_inputs_accepted() {
    let result = emit_stack_kernel("big_stack", ScalarType::F32, &[2, 3], 29, 1);
    assert!(
        result.is_ok(),
        "29-input stack should be accepted (31 buffers), got: {result:?}"
    );
}

/// Stack kernel with 30 inputs (exceeds MAX_DIRECT_BINDING_INPUTS) should
/// produce a packed kernel variant using 4 buffer slots (#1649).
#[test]
fn test_stack_kernel_30_inputs_packed() {
    let result = emit_stack_kernel("huge_stack", ScalarType::F32, &[2, 3], 30, 1);
    let msl = result.expect("30-input stack should succeed via packed kernel");
    assert!(
        msl.contains("packed_inputs"),
        "packed kernel should use packed_inputs buffer, got: {msl}"
    );
    assert!(
        msl.contains("offsets"),
        "packed kernel should use offsets buffer, got: {msl}"
    );
    // Packed kernel uses only 4 buffer slots: packed_inputs(0), offsets(1), output(2), total(3)
    assert!(
        !msl.contains("[[buffer(4)]]"),
        "packed kernel should not use buffer index >= 4, got: {msl}"
    );
}

#[test]
fn test_safe_msl_uint_rejects_large_value() {
    let val = u32::MAX as usize + 1;
    let err = safe_msl_uint(val).expect_err("value > u32::MAX must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("u32::MAX"),
        "error should mention u32::MAX, got: {msg}"
    );
}

#[test]
fn test_safe_msl_uint_accepts_boundary() {
    let val = u32::MAX as usize;
    let s = safe_msl_uint(val).expect("u32::MAX must be accepted");
    assert_eq!(s, "4294967295");
}

#[test]
fn test_axis_select_out_of_bounds_returns_err() {
    // axis=3 for rank-3 shape [2, 3, 4] is out of bounds
    let result = emit_axis_select_kernel("bad_sel", ScalarType::F32, &[2, 3, 4], 3, 0);
    let err = result.expect_err("axis >= rank must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("out of bounds"),
        "error should mention out of bounds, got: {msg}"
    );
}

#[test]
fn test_axis_select_empty_shape_returns_err() {
    let result = emit_axis_select_kernel("empty_sel", ScalarType::F32, &[], 0, 0);
    let err = result.expect_err("axis=0 for empty shape must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("out of bounds"),
        "error should mention out of bounds, got: {msg}"
    );
}

#[test]
fn test_stack_kernel_axis_out_of_bounds_returns_err() {
    // axis=3 for rank-2 shape [2, 3] is out of bounds (max valid is 2)
    let result = emit_stack_kernel("bad_stack", ScalarType::F32, &[2, 3], 2, 3);
    let err = result.expect_err("axis > rank must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("out of bounds"),
        "error should mention out of bounds, got: {msg}"
    );
}

#[test]
fn test_stack_kernel_axis_at_end_accepted() {
    // axis=2 for rank-2 shape [2, 3] is valid (appends new dim at end)
    let result = emit_stack_kernel("end_stack", ScalarType::F32, &[2, 3], 2, 2);
    assert!(
        result.is_ok(),
        "axis == rank should be valid for stack, got: {result:?}"
    );
}

// --- Conv1d kernel tests ---

#[test]
fn test_conv1d_kernel_contains_kernel_attribute() {
    let msl = emit_conv1d_kernel(
        "conv1d_test",
        ScalarType::F32,
        4,
        2,
        3,
        8,
        1,
        0,
        1,
        1,
        false,
    )
    .expect("codegen");
    assert!(msl.contains("[[kernel]]"), "must have kernel attribute");
    assert!(msl.contains("conv1d_test"), "must contain kernel name");
}

#[test]
fn test_conv1d_kernel_no_bias_buffer_layout() {
    let msl = emit_conv1d_kernel("conv1d_nb", ScalarType::F32, 4, 2, 3, 8, 1, 0, 1, 1, false)
        .expect("codegen");
    assert!(msl.contains("buffer(0)"), "input buffer");
    assert!(msl.contains("buffer(1)"), "weight buffer");
    assert!(msl.contains("buffer(2)"), "output buffer (no bias)");
    assert!(msl.contains("buffer(3)"), "total elements buffer");
    assert!(!msl.contains("bias["), "no bias indexing in no-bias mode");
}

#[test]
fn test_conv1d_kernel_with_bias_buffer_layout() {
    let msl = emit_conv1d_kernel("conv1d_b", ScalarType::F32, 4, 2, 3, 8, 1, 0, 1, 1, true)
        .expect("codegen");
    assert!(msl.contains("buffer(0)"), "input buffer");
    assert!(msl.contains("buffer(1)"), "weight buffer");
    assert!(msl.contains("buffer(2)"), "bias buffer");
    assert!(msl.contains("buffer(3)"), "output buffer (with bias)");
    assert!(msl.contains("buffer(4)"), "total elements buffer");
    assert!(msl.contains("bias"), "must reference bias");
}

#[test]
fn test_conv1d_kernel_bakes_stride_and_padding() {
    // dvoice pattern: in_len=24000, stride=4, padding=2
    let msl = emit_conv1d_kernel(
        "conv1d_dv",
        ScalarType::F32,
        1,
        48,
        8,
        24000,
        4,
        2,
        1,
        1,
        false,
    )
    .expect("codegen");
    assert!(msl.contains("STRIDE = 4"), "stride must be baked");
    assert!(msl.contains("PADDING = 2"), "padding must be baked");
    assert!(msl.contains("IN_LENGTH = 24000"), "in_length must be baked");
}

#[test]
fn test_conv1d_kernel_f16_type() {
    let msl = emit_conv1d_kernel("conv1d_h", ScalarType::F16, 2, 4, 3, 8, 1, 0, 1, 1, false)
        .expect("codegen");
    assert!(msl.contains("half"), "f16 must use 'half' for I/O buffers");
    // f16 inputs accumulate in f32 to avoid catastrophic precision loss (#2557).
    assert!(msl.contains("float sum"), "f16 must accumulate in f32");
    assert!(
        msl.contains("half(sum)"),
        "f16 must cast accumulator back to half on store"
    );
}

#[test]
fn test_conv1d_kernel_bf16_accumulates_in_f32() {
    let msl = emit_conv1d_kernel("conv1d_bf", ScalarType::BF16, 2, 4, 3, 8, 1, 0, 1, 1, true)
        .expect("codegen");
    // BF16 inputs/outputs use bfloat but accumulate in float (#2557).
    assert!(msl.contains("float sum"), "bf16 must accumulate in f32");
    assert!(
        msl.contains("float(input[in_idx]) * float(weight[w_idx])"),
        "bf16 must cast loads to float"
    );
    assert!(
        msl.contains("float(bias[oc_local])"),
        "bf16 must cast bias load to float"
    );
}

#[test]
fn test_conv1d_kernel_f32_no_cast() {
    let msl = emit_conv1d_kernel("conv1d_f", ScalarType::F32, 2, 4, 3, 8, 1, 0, 1, 1, false)
        .expect("codegen");
    // F32 should use float directly, no casts.
    assert!(msl.contains("float sum"), "f32 uses float accumulator");
    assert!(
        msl.contains("sum += input[in_idx] * weight[w_idx]"),
        "f32 should not have cast wrappers"
    );
}

#[test]
fn test_conv_transpose_1d_bf16_accumulates_in_f32() {
    let msl = emit_conv_transpose_1d_kernel(
        "ct1d_bf",
        ScalarType::BF16,
        4,
        8,
        3,
        10,
        2,
        1,
        1,
        1,
        0,
        true,
    )
    .expect("codegen");
    assert!(
        msl.contains("float sum"),
        "bf16 ConvTranspose1d must accumulate in f32"
    );
    assert!(
        msl.contains("float(input[in_idx]) * float(weight[w_idx])"),
        "bf16 ConvTranspose1d must cast loads to float"
    );
}

#[test]
fn test_conv1d_kernel_dilation_baked() {
    let msl = emit_conv1d_kernel("conv1d_d", ScalarType::F32, 2, 4, 3, 16, 1, 0, 3, 1, false)
        .expect("codegen");
    assert!(msl.contains("DILATION = 3"), "dilation must be baked");
}

// --- Concat kernel tests ---

#[test]
fn test_concat_kernel_two_inputs_same_axis_size() {
    let msl =
        emit_concat_kernel("test_cat", ScalarType::F32, &[2, 4, 8], &[4, 4], 1).expect("codegen");
    assert!(msl.contains("[[kernel]]"), "must have kernel attribute");
    assert!(msl.contains("test_cat"), "must contain kernel name");
    assert!(msl.contains("input0"), "must reference input0");
    assert!(msl.contains("input1"), "must reference input1");
    assert!(msl.contains("which_input"), "must select input buffer");
    assert!(msl.contains("local_axis"), "must compute local axis offset");
}

#[test]
fn test_concat_kernel_different_axis_sizes() {
    // KV cache pattern: existing [1, 8, 64] + new [1, 1, 64] along axis=1
    let msl =
        emit_concat_kernel("kv_cat", ScalarType::F32, &[1, 8, 64], &[8, 1], 1).expect("codegen");
    assert!(msl.contains("kv_cat"));
    assert!(msl.contains("input0"));
    assert!(msl.contains("input1"));
    // Cumulative boundary at 8: axis_coord >= 8 selects input1
    assert!(msl.contains("axis_coord >= 8"));
}

#[test]
fn test_concat_kernel_axis_zero() {
    // Batch concat: [4, 8] + [4, 8] along axis=0 → [8, 8]
    let msl =
        emit_concat_kernel("batch_cat", ScalarType::F32, &[4, 8], &[4, 4], 0).expect("codegen");
    assert!(msl.contains("batch_cat"));
    // inner_stride = product of dims after axis 0 = 8
    assert!(msl.contains("% 8"), "inner stride should be 8");
}

#[test]
fn test_concat_kernel_last_axis() {
    // Feature concat: [2, 3] + [2, 5] along axis=1 → [2, 8]
    let msl =
        emit_concat_kernel("feat_cat", ScalarType::F32, &[2, 3], &[3, 5], 1).expect("codegen");
    assert!(msl.contains("feat_cat"));
    // inner_stride for last axis = 1
    assert!(
        msl.contains("% 1"),
        "inner stride for last axis should be 1"
    );
}

#[test]
fn test_concat_kernel_three_inputs() {
    let msl =
        emit_concat_kernel("tri_cat", ScalarType::F32, &[1, 2, 8], &[2, 3, 4], 1).expect("codegen");
    assert!(msl.contains("input0"));
    assert!(msl.contains("input1"));
    assert!(msl.contains("input2"));
    // Boundaries: 2, 5 (cumulative)
    assert!(msl.contains("axis_coord >= 2"));
    assert!(msl.contains("axis_coord >= 5"));
}

#[test]
fn test_concat_kernel_empty_inputs_returns_err() {
    let result = emit_concat_kernel("empty", ScalarType::F32, &[2, 3], &[], 0);
    assert!(result.is_err(), "empty inputs must fail");
}

#[test]
fn test_concat_kernel_axis_out_of_bounds_returns_err() {
    let result = emit_concat_kernel("bad_axis", ScalarType::F32, &[2, 3], &[3, 3], 2);
    let err = result.expect_err("axis >= rank must fail");
    let msg = err.to_string();
    assert!(msg.contains("out of bounds"), "got: {msg}");
}

// --- ConvTranspose1d MSL codegen tests ---

#[test]
fn test_conv_transpose_1d_kernel_basic() {
    let msl = emit_conv_transpose_1d_kernel(
        "ct1d_test",
        ScalarType::F32,
        4, // in_channels
        2, // out_channels
        3, // kernel_size
        8, // in_length
        2, // stride
        1, // padding
        1, // dilation
        1, // groups
        0, // output_padding
        false,
    )
    .expect("codegen");
    assert!(msl.contains("[[kernel]]"), "must have kernel attribute");
    assert!(msl.contains("ct1d_test"), "must contain kernel name");
    assert!(msl.contains("STRIDE = 2"), "stride must be baked");
    assert!(msl.contains("PADDING = 1"), "padding must be baked");
    assert!(msl.contains("IN_LENGTH = 8"), "in_length must be baked");
    assert!(msl.contains("KERNEL_SIZE = 3"), "kernel_size must be baked");
    assert!(msl.contains("IN_CHANNELS = 4"), "in_channels must be baked");
    assert!(
        msl.contains("OUT_CHANNELS = 2"),
        "out_channels must be baked"
    );
    // out_length = (8-1)*2 + 3 - 2*1 = 15
    assert!(msl.contains("OUT_LENGTH = 15"), "out_length must be baked");
}

#[test]
fn test_conv_transpose_1d_kernel_with_bias() {
    let msl = emit_conv_transpose_1d_kernel(
        "ct1d_bias",
        ScalarType::F32,
        2, // in_channels
        4, // out_channels
        3, // kernel_size
        8, // in_length
        1, // stride
        0, // padding
        1, // dilation
        1, // groups
        0, // output_padding
        true,
    )
    .expect("codegen");
    assert!(msl.contains("bias"), "must reference bias buffer");
    assert!(
        msl.contains("bias[oc_local]"),
        "must index bias by output channel (oc_local for batch safety)"
    );
    assert!(msl.contains("buffer(2)"), "bias at buffer 2");
    assert!(msl.contains("buffer(3)"), "output at buffer 3");
    assert!(msl.contains("buffer(4)"), "total at buffer 4");
}

#[test]
fn test_conv_transpose_1d_kernel_no_bias_buffer_layout() {
    let msl = emit_conv_transpose_1d_kernel(
        "ct1d_nobias",
        ScalarType::F32,
        1,
        1,
        2,
        4,
        1,
        0,
        1,
        1,
        0, // output_padding
        false,
    )
    .expect("codegen");
    // No bias: output at buffer(2), total at buffer(3)
    assert!(msl.contains("buffer(2)"), "output at buffer 2");
    assert!(msl.contains("buffer(3)"), "total at buffer 3");
    assert!(!msl.contains("bias["), "no bias indexing in no-bias mode");
}

#[test]
fn test_conv_transpose_1d_kernel_kokoro_scale() {
    // Kokoro Generator: stride=10, kernel=20, no padding
    let msl = emit_conv_transpose_1d_kernel(
        "kokoro_up",
        ScalarType::F32,
        512, // in_channels
        256, // out_channels
        20,  // kernel_size
        8,   // in_length
        10,  // stride
        0,   // padding
        1,   // dilation
        1,   // groups
        0,   // output_padding
        true,
    )
    .expect("codegen");
    assert!(msl.contains("STRIDE = 10"));
    assert!(msl.contains("KERNEL_SIZE = 20"));
    // out_length = (8-1)*10 + 20 - 0 = 90
    assert!(msl.contains("OUT_LENGTH = 90"));
}
