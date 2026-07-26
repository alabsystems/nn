// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for HIP complex op emission (linear, matmul, softmax, embedding).

use crate::codegen_hip_tensor_emit_complex::*;
use nn_dsl::ScalarType;

#[test]
fn test_linear_kernel_no_bias() {
    let src = emit_linear_kernel("linear_test", ScalarType::F32, 128, 64, false).unwrap();
    assert!(src.contains("extern \"C\" __global__ void linear_test"));
    assert!(src.contains("IN_FEATURES = 128"));
    assert!(src.contains("OUT_FEATURES = 64"));
    assert!(!src.contains("bias"));
}

#[test]
fn test_linear_kernel_with_bias() {
    let src = emit_linear_kernel("linear_bias", ScalarType::F32, 256, 128, true).unwrap();
    assert!(src.contains("bias"));
    assert!(src.contains("sum += bias[col]"));
}

#[test]
fn test_linear_kernel_f16_accumulation() {
    let src = emit_linear_kernel("linear_f16", ScalarType::F16, 64, 32, true).unwrap();
    // Should accumulate in float, not half.
    assert!(src.contains("float sum"));
    // Should cast loads.
    assert!(src.contains("(float)input"));
    assert!(src.contains("(float)weight"));
    // Should cast output.
    assert!(src.contains("(half)sum"));
}

#[test]
fn test_matmul_kernel_basic() {
    let src = emit_matmul_kernel("mm_test", ScalarType::F32, 8, 16, 4, false, false, None).unwrap();
    assert!(src.contains("M = 8"));
    assert!(src.contains("K = 16"));
    assert!(src.contains("N = 4"));
    // Normal (non-transposed) right index.
    assert!(src.contains("kk * N + j"));
}

#[test]
fn test_matmul_kernel_transpose_right() {
    let src = emit_matmul_kernel("mm_tr", ScalarType::F32, 4, 8, 4, true, false, None).unwrap();
    // Transposed right index.
    assert!(src.contains("j * K + kk"));
}

#[test]
fn test_matmul_kernel_broadcast_right() {
    let src = emit_matmul_kernel("mm_br", ScalarType::F32, 4, 8, 4, false, true, None).unwrap();
    assert!(src.contains("batch_offset_r = 0"));
}

#[test]
fn test_matmul_kernel_with_scale() {
    let src = emit_matmul_kernel(
        "mm_scale",
        ScalarType::F32,
        4,
        8,
        4,
        false,
        false,
        Some(0.125),
    )
    .unwrap();
    assert!(src.contains("sum *= (float)"));
    assert!(src.contains("0.125"));
}

#[test]
fn test_softmax_kernel_f32() {
    let src = emit_softmax_kernel("softmax_test", ScalarType::F32).unwrap();
    assert!(src.contains("__shared__"));
    assert!(src.contains("__syncthreads()"));
    assert!(src.contains("expf("));
    assert!(src.contains("-HUGE_VALF"));
    assert!(src.contains("Phase 1"));
    assert!(src.contains("Phase 2"));
    assert!(src.contains("Phase 3"));
}

#[test]
fn test_embedding_kernel() {
    let src = emit_embedding_kernel("embed_test", ScalarType::F32, 512).unwrap();
    assert!(src.contains("EMBEDDING_DIM = 512"));
    assert!(src.contains("(unsigned int)indices[seq_idx]"));
}
