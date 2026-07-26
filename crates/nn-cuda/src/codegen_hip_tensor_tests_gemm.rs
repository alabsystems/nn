// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Structural tests for rocWMMA tiled GEMM codegen.

use super::*;
use nn_dsl::ScalarType;

// --- Routing tests ---

#[test]
fn test_should_use_rocwmma_aligned() {
    // M=256, K=256, N=256 — all 16-aligned, large enough.
    assert!(should_use_rocwmma(256, 256, 256));
}

#[test]
fn test_should_use_rocwmma_misaligned_m() {
    // M=255 not divisible by 16.
    assert!(!should_use_rocwmma(255, 256, 256));
}

#[test]
fn test_should_use_rocwmma_small_output() {
    // M*N = 16*16 = 256 < 16384 threshold.
    assert!(!should_use_rocwmma(16, 256, 16));
}

#[test]
fn test_should_use_rocwmma_small_k() {
    // K=64 < 128 threshold.
    assert!(!should_use_rocwmma(128, 64, 128));
}

#[test]
fn test_should_use_rocwmma_threshold_exact() {
    // M*N = 128*128 = 16384 = threshold, K=128 = threshold.
    assert!(should_use_rocwmma(128, 128, 128));
}

// --- MatMul kernel tests ---

#[test]
fn test_rocwmma_matmul_f32_basic() {
    let src = emit_rocwmma_matmul_kernel(
        "mm_f32",
        ScalarType::F32,
        64,
        128,
        64,
        1,
        false,
        false,
        None,
    )
    .unwrap();
    assert!(
        src.contains("rocwmma/rocwmma.hpp"),
        "missing rocWMMA include"
    );
    assert!(src.contains("extern \"C\" __global__ void mm_f32"));
    assert!(src.contains("rocwmma::fragment<rocwmma::accumulator"));
    assert!(src.contains("rocwmma::fill_fragment(acc, 0.0f)"));
    assert!(src.contains("rocwmma::load_matrix_sync"));
    assert!(src.contains("rocwmma::mma_sync(acc, a_frag, b_frag, acc)"));
    assert!(src.contains("rocwmma::store_matrix_sync"));
    assert!(src.contains("__shared__"));
    assert!(src.contains("__syncthreads()"));
    assert!(src.contains("M_DIM = 64"));
    assert!(src.contains("K_DIM = 128"));
    assert!(src.contains("N_DIM = 64"));
}

#[test]
fn test_rocwmma_matmul_f16() {
    let src = emit_rocwmma_matmul_kernel(
        "mm_f16",
        ScalarType::F16,
        32,
        128,
        32,
        1,
        false,
        false,
        None,
    )
    .unwrap();
    assert!(src.contains("rocwmma::float16_t"), "f16 fragment type");
    assert!(src.contains("const half*"), "f16 buffer type");
    assert!(src.contains("float> acc"), "f32 accumulator");
}

#[test]
fn test_rocwmma_matmul_bf16() {
    let src = emit_rocwmma_matmul_kernel(
        "mm_bf16",
        ScalarType::BF16,
        32,
        128,
        32,
        1,
        false,
        false,
        None,
    )
    .unwrap();
    assert!(src.contains("rocwmma::bfloat16_t"), "bf16 fragment type");
    assert!(src.contains("const hip_bfloat16*"), "bf16 buffer type");
}

#[test]
fn test_rocwmma_matmul_transpose_right() {
    let src =
        emit_rocwmma_matmul_kernel("mm_tr", ScalarType::F32, 64, 128, 64, 1, true, false, None)
            .unwrap();
    // Transposed B: shared memory load uses gc * K_DIM + gr.
    assert!(src.contains("gc * 128u + gr"), "transposed B index");
}

#[test]
fn test_rocwmma_matmul_broadcast_right() {
    let src =
        emit_rocwmma_matmul_kernel("mm_br", ScalarType::F32, 64, 128, 64, 4, false, true, None)
            .unwrap();
    assert!(src.contains("b_offset = 0"), "broadcast B offset");
    assert!(src.contains("BATCH_COUNT = 4"), "batch count");
}

#[test]
fn test_rocwmma_matmul_with_scale() {
    let src = emit_rocwmma_matmul_kernel(
        "mm_sc",
        ScalarType::F32,
        64,
        128,
        64,
        1,
        false,
        false,
        Some(0.125),
    )
    .unwrap();
    assert!(src.contains("val *= 0.125"), "scale multiplication");
}

// --- Linear kernel tests ---

#[test]
fn test_rocwmma_linear_with_bias() {
    let src = emit_rocwmma_linear_kernel("lin_bias", ScalarType::F32, 256, 128, 32, true).unwrap();
    assert!(
        src.contains("const float* __restrict__ bias"),
        "bias parameter"
    );
    assert!(src.contains("val += bias[gc]"), "bias addition");
    // Linear: M=batch=32, K=in_features=256, N=out_features=128.
    assert!(src.contains("M_DIM = 32"), "M = batch_size");
    assert!(src.contains("K_DIM = 256"), "K = in_features");
    assert!(src.contains("N_DIM = 128"), "N = out_features");
}

#[test]
fn test_rocwmma_linear_no_bias() {
    let src = emit_rocwmma_linear_kernel("lin_nb", ScalarType::F32, 256, 128, 32, false).unwrap();
    assert!(!src.contains("bias"), "no bias parameter or addition");
}

// --- Structural invariant tests ---

#[test]
fn test_rocwmma_warp_bounds_check() {
    let src =
        emit_rocwmma_matmul_kernel("mm_wb", ScalarType::F32, 64, 128, 64, 1, false, false, None)
            .unwrap();
    // Must have warp_id >= 4 early-return for RDNA3 (8 warps of 32).
    assert!(
        src.contains("if (warp_id >= 4u) return"),
        "warp bounds check"
    );
}

#[test]
fn test_rocwmma_output_bounds_check() {
    let src =
        emit_rocwmma_matmul_kernel("mm_bc", ScalarType::F32, 64, 128, 64, 1, false, false, None)
            .unwrap();
    // Must have bounds check before writing to global C.
    assert!(
        src.contains("if (gr < M_DIM && gc < N_DIM)"),
        "output bounds check"
    );
}

#[test]
fn test_rocwmma_batch_bounds_check() {
    let src = emit_rocwmma_matmul_kernel(
        "mm_batch",
        ScalarType::F32,
        64,
        128,
        64,
        8,
        false,
        false,
        None,
    )
    .unwrap();
    assert!(
        src.contains("if (batch_idx >= BATCH_COUNT) return"),
        "batch bounds"
    );
}
