// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for MXFP4 GEMM kernel emission.

use super::*;

#[test]
fn test_mxfp4_gemm_basic_generation() {
    let src = emit_mxfp4_gemm_kernel("mxfp4_gemm_test", 256, 128, 256, 1).unwrap();

    // Verify includes.
    assert!(
        src.contains("#include <hip/hip_runtime.h>"),
        "missing HIP runtime include"
    );
    assert!(
        src.contains("#include <rocwmma/rocwmma.hpp>"),
        "missing rocWMMA include"
    );

    // Verify MXFP4 helpers are present.
    assert!(src.contains("MXFP4_E2M1_LUT"), "missing E2M1 LUT");
    assert!(src.contains("mxfp4_unpack"), "missing unpack function");
    assert!(src.contains("mxfp4_scale"), "missing scale function");
    assert!(src.contains("mxfp4_dequant"), "missing dequant function");

    // Verify kernel signature.
    assert!(src.contains("extern \"C\" __global__ void mxfp4_gemm_test("));
    assert!(src.contains("const unsigned char* __restrict__ A_packed"));
    assert!(src.contains("const unsigned char* __restrict__ A_scales"));
    assert!(src.contains("const unsigned char* __restrict__ B_packed"));
    assert!(src.contains("const unsigned char* __restrict__ B_scales"));
    assert!(src.contains("float* __restrict__ C"));
}

#[test]
fn test_mxfp4_gemm_dimensions_in_source() {
    let src = emit_mxfp4_gemm_kernel("k1", 512, 1024, 768, 1).unwrap();

    assert!(src.contains("M_DIM = 512"), "wrong M dimension");
    assert!(src.contains("K_DIM = 1024"), "wrong K dimension");
    assert!(src.contains("N_DIM = 768"), "wrong N dimension");
}

#[test]
fn test_mxfp4_gemm_batch_support() {
    let src = emit_mxfp4_gemm_kernel("k_batched", 256, 128, 256, 8).unwrap();

    assert!(src.contains("BATCH_COUNT = 8"), "wrong batch count");
    assert!(
        src.contains("batch_idx = blockIdx.z"),
        "missing batch indexing"
    );
}

#[test]
fn test_mxfp4_gemm_dequant_in_tile_load() {
    let src = emit_mxfp4_gemm_kernel("k_dq", 32, 32, 32, 1).unwrap();

    // Verify A tile load includes dequantization.
    assert!(src.contains("A_packed[byte_idx]"), "missing packed A load");
    assert!(src.contains("A_scales[scale_idx]"), "missing A scale load");
    assert!(
        src.contains("mxfp4_dequant(packed, sub, scale)"),
        "missing dequant call in A load"
    );

    // Verify B tile load includes dequantization.
    assert!(src.contains("B_packed[byte_idx]"), "missing packed B load");
    assert!(src.contains("B_scales[scale_idx]"), "missing B scale load");
}

#[test]
fn test_mxfp4_gemm_rocwmma_mma() {
    let src = emit_mxfp4_gemm_kernel("k_mma", 256, 256, 256, 1).unwrap();

    assert!(
        src.contains("rocwmma::fragment<rocwmma::accumulator"),
        "missing accumulator"
    );
    assert!(
        src.contains("rocwmma::fragment<rocwmma::matrix_a"),
        "missing matrix_a fragment"
    );
    assert!(
        src.contains("rocwmma::fragment<rocwmma::matrix_b"),
        "missing matrix_b fragment"
    );
    assert!(
        src.contains("rocwmma::mma_sync(acc, a_frag, b_frag, acc)"),
        "missing mma_sync"
    );
    assert!(
        src.contains("rocwmma::store_matrix_sync"),
        "missing store_matrix_sync"
    );
}

#[test]
fn test_mxfp4_gemm_alignment_error_m() {
    let result = emit_mxfp4_gemm_kernel("bad", 33, 32, 32, 1);
    assert!(result.is_err(), "M=33 should fail alignment check");
    let err = result.unwrap_err().to_string();
    assert!(err.contains("multiples of 32"), "{err}");
}

#[test]
fn test_mxfp4_gemm_alignment_error_k() {
    let result = emit_mxfp4_gemm_kernel("bad", 32, 48, 32, 1);
    assert!(result.is_err(), "K=48 should fail alignment check");
}

#[test]
fn test_mxfp4_gemm_alignment_error_n() {
    let result = emit_mxfp4_gemm_kernel("bad", 32, 32, 100, 1);
    assert!(result.is_err(), "N=100 should fail alignment check");
}

#[test]
fn test_mxfp4_gemm_launch_config() {
    let cfg = mxfp4_gemm_launch_config(256, 128, 4);
    assert_eq!(cfg.grid.x, 4); // ceil(128/32)
    assert_eq!(cfg.grid.y, 8); // ceil(256/32)
    assert_eq!(cfg.grid.z, 4); // batch
    assert_eq!(cfg.block.x, 256);
}

#[test]
fn test_mxfp4_gemm_large_competition_dimensions() {
    // Typical competition GEMM sizes for DeepSeek/Kimi.
    let src = emit_mxfp4_gemm_kernel("mxfp4_gemm_4096", 4096, 4096, 4096, 1).unwrap();

    assert!(src.contains("M_DIM = 4096"));
    assert!(src.contains("K_DIM = 4096"));
    assert!(src.contains("N_DIM = 4096"));

    // Verify the kernel compiles (string-level: balanced braces).
    let opens = src.matches('{').count();
    let closes = src.matches('}').count();
    assert_eq!(opens, closes, "unbalanced braces in generated kernel");
}

#[test]
fn test_mxfp4_gemm_standalone() {
    let src = emit_mxfp4_gemm_standalone("qualifier_gemm", 256, 256, 256, 1).unwrap();

    // Standalone should be a complete compilable file.
    assert!(src.contains("#include <hip/hip_runtime.h>"));
    assert!(src.contains("extern \"C\" __global__"));
}

#[test]
fn test_mxfp4_gemm_nibble_addressing() {
    let src = emit_mxfp4_gemm_kernel("k_nibble", 32, 32, 32, 1).unwrap();

    // Verify nibble sub-indexing: gc & 1u for even/odd selection.
    assert!(src.contains("gc & 1u"), "missing nibble sub-index");
    // Verify byte addressing: gc / 2u for packed byte offset.
    assert!(src.contains("gc / 2u"), "missing byte addressing");
}

#[test]
fn test_mxfp4_gemm_scale_addressing() {
    let src = emit_mxfp4_gemm_kernel("k_scale", 64, 64, 64, 1).unwrap();

    // Verify scale block addressing: gc / MX_BLK for scale index.
    assert!(
        src.contains("gc / MX_BLK"),
        "missing scale block addressing"
    );
}
