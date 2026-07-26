// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for HIP elementwise op emission.

use crate::codegen_hip_tensor_emit_ops::*;
use nn_dsl::ScalarType;

#[test]
fn test_binary_add_f32() {
    let src = emit_binary_add_kernel("add_test", ScalarType::F32, 1024).unwrap();
    assert!(src.contains("extern \"C\" __global__ void add_test"));
    assert!(src.contains("left[tid] + right[tid]"));
    assert!(src.contains("1024u"));
}

#[test]
fn test_binary_mul_f32() {
    let src = emit_binary_mul_kernel("mul_test", ScalarType::F32, 512).unwrap();
    assert!(src.contains("left[tid] * right[tid]"));
}

#[test]
fn test_sigmoid_f32() {
    let src = emit_sigmoid_kernel("sig_test", ScalarType::F32, 256).unwrap();
    assert!(src.contains("expf(-x)"));
    assert!(src.contains("1.0f / (1.0f"));
}

#[test]
fn test_gelu_f32() {
    let src = emit_gelu_kernel("gelu_test", ScalarType::F32, 128).unwrap();
    assert!(src.contains("0.7978845608028654f"));
    assert!(src.contains("0.044715f"));
    assert!(src.contains("expf(2.0f * inner)"));
}

#[test]
fn test_gelu_erf_f32() {
    let src = emit_gelu_erf_kernel("gelu_erf_test", ScalarType::F32, 64).unwrap();
    assert!(src.contains("Abramowitz"));
    assert!(src.contains("0.3275911f"));
    assert!(src.contains("erf_val"));
}

#[test]
fn test_relu_f32() {
    let src = emit_relu_kernel("relu_test", ScalarType::F32, 2048).unwrap();
    assert!(src.contains("(x > (float)0)"));
}

#[test]
fn test_tanh_f32() {
    let src = emit_tanh_kernel("tanh_test", ScalarType::F32, 512).unwrap();
    assert!(src.contains("tanhf("));
}

#[test]
fn test_sigmoid_f16_casts() {
    let src = emit_sigmoid_kernel("sig_f16", ScalarType::F16, 100).unwrap();
    // f16 should cast to float for computation, then back to half.
    assert!(src.contains("(float)input[tid]"));
    assert!(src.contains("(half)"));
}
