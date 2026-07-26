// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the HIP tensor codegen entry point.

use nn_dsl::ScalarType;

#[test]
fn test_emit_gemm_hip_f32() {
    let src = crate::codegen_hip_tensor::emit_gemm_hip("test_gemm", ScalarType::F32, 64, 128, 32)
        .expect("should generate GEMM kernel");

    // Should include the HIP prelude.
    assert!(src.contains("#include <hip/hip_runtime.h>"));
    // Should include the kernel declaration.
    assert!(src.contains("extern \"C\" __global__ void test_gemm"));
    // Should use float type.
    assert!(src.contains("float"));
    // Should include matrix dimensions.
    assert!(src.contains("M = 64"));
    assert!(src.contains("K = 128"));
    assert!(src.contains("N = 32"));
}

#[test]
fn test_emit_gemm_hip_f16() {
    let src = crate::codegen_hip_tensor::emit_gemm_hip("gemm_f16", ScalarType::F16, 32, 64, 16)
        .expect("should generate f16 GEMM kernel");

    // f16 should include fp16 header.
    assert!(src.contains("#include <hip/hip_fp16.h>"));
    // Should use half type for parameters.
    assert!(src.contains("half"));
    // Accumulation should be in float.
    assert!(src.contains("float sum"));
}

#[test]
fn test_emit_tensor_hip_from_ir() {
    // Build a simple matmul tensor IR graph: C = A @ B
    // A: [2, 4], B: [4, 3], C: [2, 3]
    let a = nn_dsl::TensorNode::new(
        nn_dsl::TensorNodeId::new(0),
        nn_dsl::TensorOpKind::Input {
            name: "a".to_string(),
            shape: vec![2, 4],
        },
        vec![2, 4],
    );
    let b = nn_dsl::TensorNode::new(
        nn_dsl::TensorNodeId::new(1),
        nn_dsl::TensorOpKind::Input {
            name: "b".to_string(),
            shape: vec![4, 3],
        },
        vec![4, 3],
    );
    let c = nn_dsl::TensorNode::new(
        nn_dsl::TensorNodeId::new(2),
        nn_dsl::TensorOpKind::MatMul {
            left: nn_dsl::TensorNodeId::new(0),
            right: nn_dsl::TensorNodeId::new(1),
            transpose_right: false,
            scale: None,
        },
        vec![2, 3],
    );

    let kernel =
        nn_dsl::TensorKernelDef::new("test_matmul", vec![a, b, c], nn_dsl::TensorNodeId::new(2));

    let src = crate::codegen_hip_tensor::emit_tensor_hip(&kernel, ScalarType::F32)
        .expect("should generate HIP source from tensor IR");

    assert!(src.contains("__global__"));
    assert!(src.contains("M = 2"));
    assert!(src.contains("K = 4"));
    assert!(src.contains("N = 3"));
}
