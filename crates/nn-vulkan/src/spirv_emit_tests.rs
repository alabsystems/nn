// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! External tests for `spirv_emit` module.
//!
//! Tests the public API for GLSL compute shader generation and SPIR-V
//! emission constants.

use crate::spirv_emit::{
    emit_elementwise_glsl, emit_matmul_glsl, emit_reduction_glsl, emit_softmax_glsl, glsl_type,
    spirv_type_bytes, ReductionOp, DEFAULT_WORKGROUP_SIZE, GLSL_COMPUTE_VERSION, SPIRV_MAGIC,
    SPIRV_VERSION_1_5,
};
use nn_dsl::ScalarType;

// ---- Constants ----

#[test]
fn test_spirv_magic_value() {
    assert_eq!(SPIRV_MAGIC, 0x0723_0203, "SPIR-V magic must be 0x07230203");
}

#[test]
fn test_spirv_version_1_5_value() {
    assert_eq!(
        SPIRV_VERSION_1_5, 0x0001_0500,
        "SPIR-V 1.5 version word must be 0x00010500"
    );
}

#[test]
fn test_default_workgroup_size_power_of_two() {
    assert!(
        DEFAULT_WORKGROUP_SIZE.is_power_of_two(),
        "DEFAULT_WORKGROUP_SIZE ({DEFAULT_WORKGROUP_SIZE}) must be power of 2"
    );
}

#[test]
fn test_default_workgroup_size_within_limits() {
    assert!(DEFAULT_WORKGROUP_SIZE > 0);
    assert!(
        DEFAULT_WORKGROUP_SIZE <= 1024,
        "DEFAULT_WORKGROUP_SIZE ({DEFAULT_WORKGROUP_SIZE}) exceeds Vulkan guaranteed minimum"
    );
}

#[test]
fn test_glsl_compute_version_header() {
    assert_eq!(GLSL_COMPUTE_VERSION, "#version 450\n");
}

// ---- glsl_type ----

#[test]
fn test_glsl_type_f32() {
    assert_eq!(glsl_type(ScalarType::F32).unwrap(), "float");
}

#[test]
fn test_glsl_type_f16() {
    assert_eq!(glsl_type(ScalarType::F16).unwrap(), "float16_t");
}

#[test]
fn test_glsl_type_bf16_unsupported() {
    let result = glsl_type(ScalarType::BF16);
    assert!(result.is_err(), "bf16 has no native GLSL type");
}

// ---- spirv_type_bytes ----

#[test]
fn test_spirv_type_bytes_f32() {
    assert_eq!(spirv_type_bytes(ScalarType::F32).unwrap(), 4);
}

#[test]
fn test_spirv_type_bytes_f16() {
    assert_eq!(spirv_type_bytes(ScalarType::F16).unwrap(), 2);
}

#[test]
fn test_spirv_type_bytes_bf16() {
    assert_eq!(spirv_type_bytes(ScalarType::BF16).unwrap(), 2);
}

// ---- emit_elementwise_glsl ----

#[test]
fn test_elementwise_glsl_relu() {
    let glsl = emit_elementwise_glsl("relu", "max(x, 0.0)", 256).unwrap();
    assert!(
        glsl.starts_with("#version 450"),
        "must start with GLSL version"
    );
    assert!(glsl.contains("relu"), "must contain kernel name in comment");
    assert!(
        glsl.contains("local_size_x = 256"),
        "must set workgroup size"
    );
    assert!(
        glsl.contains("max(x, 0.0)"),
        "must contain the operation expression"
    );
    assert!(glsl.contains("void main()"), "must have main entry point");
    assert!(
        glsl.contains("gl_GlobalInvocationID"),
        "must use global invocation ID"
    );
}

#[test]
fn test_elementwise_glsl_custom_workgroup() {
    let glsl = emit_elementwise_glsl("sigmoid", "1.0 / (1.0 + exp(-x))", 64).unwrap();
    assert!(glsl.contains("local_size_x = 64"));
    assert!(glsl.contains("1.0 / (1.0 + exp(-x))"));
}

#[test]
fn test_elementwise_glsl_zero_workgroup_rejected() {
    let result = emit_elementwise_glsl("bad", "x", 0);
    assert!(result.is_err(), "workgroup_size=0 must be rejected");
}

#[test]
fn test_elementwise_glsl_has_buffer_layout() {
    let glsl = emit_elementwise_glsl("test", "x * 2.0", 128).unwrap();
    assert!(glsl.contains("input_buffer"), "must declare input buffer");
    assert!(glsl.contains("output_buffer"), "must declare output buffer");
    assert!(
        glsl.contains("push_constant"),
        "must declare push constants"
    );
    assert!(
        glsl.contains("total_elements"),
        "must have total_elements push constant"
    );
}

// ---- emit_matmul_glsl ----

#[test]
fn test_matmul_glsl_tile_16() {
    let glsl = emit_matmul_glsl(16).unwrap();
    assert!(glsl.starts_with("#version 450"));
    assert!(glsl.contains("local_size_x = 16"));
    assert!(glsl.contains("local_size_y = 16"));
    assert!(glsl.contains("tileA"), "must declare shared memory tile A");
    assert!(glsl.contains("tileB"), "must declare shared memory tile B");
    assert!(
        glsl.contains("barrier()"),
        "must have barrier for shared memory sync"
    );
}

#[test]
fn test_matmul_glsl_tile_8() {
    let glsl = emit_matmul_glsl(8).unwrap();
    assert!(glsl.contains("local_size_x = 8"));
    assert!(glsl.contains("local_size_y = 8"));
}

#[test]
fn test_matmul_glsl_non_power_of_two_rejected() {
    let result = emit_matmul_glsl(15);
    assert!(result.is_err(), "non-power-of-2 tile_size must be rejected");
}

#[test]
fn test_matmul_glsl_zero_tile_rejected() {
    let result = emit_matmul_glsl(0);
    assert!(result.is_err(), "tile_size=0 must be rejected");
}

#[test]
fn test_matmul_glsl_has_dimensions() {
    let glsl = emit_matmul_glsl(16).unwrap();
    assert!(glsl.contains("uint M"), "must have M dimension");
    assert!(glsl.contains("uint N"), "must have N dimension");
    assert!(glsl.contains("uint K"), "must have K dimension");
}

// ---- emit_reduction_glsl ----

#[test]
fn test_reduction_glsl_sum() {
    let glsl = emit_reduction_glsl("sum_reduce", ReductionOp::Sum, 256).unwrap();
    assert!(glsl.starts_with("#version 450"));
    assert!(glsl.contains("local_size_x = 256"));
    assert!(
        glsl.contains("shared float sdata"),
        "must use shared memory"
    );
    assert!(glsl.contains("barrier()"), "must have barrier");
    assert!(glsl.contains("Sum"), "must mention Sum op in comment");
}

#[test]
fn test_reduction_glsl_max() {
    let glsl = emit_reduction_glsl("max_reduce", ReductionOp::Max, 128).unwrap();
    assert!(glsl.contains("local_size_x = 128"));
    assert!(
        glsl.contains("max(a, b)"),
        "Max reduction must use max(a, b)"
    );
    assert!(glsl.contains("Max"), "must mention Max op in comment");
}

#[test]
fn test_reduction_glsl_min() {
    let glsl = emit_reduction_glsl("min_reduce", ReductionOp::Min, 64).unwrap();
    assert!(glsl.contains("local_size_x = 64"));
    assert!(
        glsl.contains("min(a, b)"),
        "Min reduction must use min(a, b)"
    );
    assert!(glsl.contains("Min"), "must mention Min op in comment");
}

#[test]
fn test_reduction_glsl_zero_workgroup_rejected() {
    let result = emit_reduction_glsl("bad", ReductionOp::Sum, 0);
    assert!(result.is_err(), "workgroup_size=0 must be rejected");
}

#[test]
fn test_reduction_glsl_non_power_of_two_rejected() {
    let result = emit_reduction_glsl("bad", ReductionOp::Sum, 100);
    assert!(
        result.is_err(),
        "non-power-of-2 workgroup_size must be rejected"
    );
}

#[test]
fn test_reduction_glsl_has_row_params() {
    let glsl = emit_reduction_glsl("test", ReductionOp::Sum, 256).unwrap();
    assert!(
        glsl.contains("row_size"),
        "must have row_size push constant"
    );
    assert!(
        glsl.contains("num_rows"),
        "must have num_rows push constant"
    );
}

// ---- emit_softmax_glsl ----

#[test]
fn test_softmax_glsl_valid() {
    let glsl = emit_softmax_glsl(256).unwrap();
    assert!(glsl.starts_with("#version 450"));
    assert!(glsl.contains("local_size_x = 256"));
    assert!(
        glsl.contains("smax"),
        "must have shared memory for max reduction"
    );
    assert!(
        glsl.contains("ssum"),
        "must have shared memory for sum reduction"
    );
    assert!(glsl.contains("exp("), "must compute exp for softmax");
    assert!(glsl.contains("barrier()"), "must have barrier");
}

#[test]
fn test_softmax_glsl_custom_workgroup() {
    let glsl = emit_softmax_glsl(64).unwrap();
    assert!(glsl.contains("local_size_x = 64"));
}

#[test]
fn test_softmax_glsl_zero_workgroup_rejected() {
    let result = emit_softmax_glsl(0);
    assert!(result.is_err(), "workgroup_size=0 must be rejected");
}

#[test]
fn test_softmax_glsl_non_power_of_two_rejected() {
    let result = emit_softmax_glsl(100);
    assert!(
        result.is_err(),
        "non-power-of-2 workgroup_size must be rejected"
    );
}

#[test]
fn test_softmax_glsl_three_pass_structure() {
    let glsl = emit_softmax_glsl(128).unwrap();
    // Pass 1: find max.
    assert!(
        glsl.contains("row_max") || glsl.contains("local_max"),
        "must compute row max"
    );
    // Pass 2: exp and sum.
    assert!(
        glsl.contains("local_sum") || glsl.contains("row_sum"),
        "must compute sum of exp"
    );
    // Pass 3: normalize.
    assert!(
        glsl.contains("/ row_sum"),
        "must divide by sum for normalization"
    );
}

// ---- ReductionOp variants ----

#[test]
fn test_reduction_op_debug() {
    assert_eq!(format!("{:?}", ReductionOp::Sum), "Sum");
    assert_eq!(format!("{:?}", ReductionOp::Max), "Max");
    assert_eq!(format!("{:?}", ReductionOp::Min), "Min");
}

#[test]
fn test_reduction_op_clone_eq() {
    let a = ReductionOp::Sum;
    let b = a;
    assert_eq!(a, b);
    assert_ne!(ReductionOp::Sum, ReductionOp::Max);
    assert_ne!(ReductionOp::Max, ReductionOp::Min);
}
