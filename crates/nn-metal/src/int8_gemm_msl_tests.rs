// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for INT8 W8A16 GEMM MSL code generation.

use super::{generate_int8_gemm_msl, int8_gemm_input_count, int8_gemm_threadgroup_bytes, Int8GemmInfo};

fn info_no_bias() -> Int8GemmInfo {
    Int8GemmInfo { m: 64, k: 128, n: 256, has_bias: false }
}

fn info_with_bias() -> Int8GemmInfo {
    Int8GemmInfo { m: 64, k: 128, n: 256, has_bias: true }
}

// ---- generate_int8_gemm_msl: basic structure ----

#[test]
fn test_int8_gemm_msl_contains_kernel_name() {
    let msl = generate_int8_gemm_msl(&info_no_bias());
    assert!(
        msl.contains("kernel void int8_matmul_dequant("),
        "MSL must contain the kernel function name"
    );
}

#[test]
fn test_int8_gemm_msl_includes_metal_headers() {
    let msl = generate_int8_gemm_msl(&info_no_bias());
    assert!(msl.contains("#include <metal_stdlib>"));
    assert!(msl.contains("#include <metal_simdgroup_matrix>"));
    assert!(msl.contains("using namespace metal;"));
}

#[test]
fn test_int8_gemm_msl_contains_buffer_bindings_no_bias() {
    let msl = generate_int8_gemm_msl(&info_no_bias());
    assert!(msl.contains("[[buffer(0)]]"), "A activation buffer");
    assert!(msl.contains("[[buffer(1)]]"), "W weight buffer");
    assert!(msl.contains("[[buffer(2)]]"), "scale buffer");
    assert!(msl.contains("[[buffer(3)]]"), "zero_point buffer");
    assert!(msl.contains("[[buffer(4)]]"), "output buffer (no bias)");
    // No buffer(5) when bias is absent.
    assert!(!msl.contains("[[buffer(5)]]"), "no buffer(5) without bias");
}

#[test]
fn test_int8_gemm_msl_contains_buffer_bindings_with_bias() {
    let msl = generate_int8_gemm_msl(&info_with_bias());
    assert!(msl.contains("[[buffer(0)]]"), "A activation buffer");
    assert!(msl.contains("[[buffer(1)]]"), "W weight buffer");
    assert!(msl.contains("[[buffer(2)]]"), "scale buffer");
    assert!(msl.contains("[[buffer(3)]]"), "zero_point buffer");
    assert!(msl.contains("[[buffer(4)]]"), "bias buffer");
    assert!(msl.contains("[[buffer(5)]]"), "output buffer (with bias)");
}

// ---- Buffer index correctness: output buffer placement ----

#[test]
fn test_int8_gemm_output_buffer_index_no_bias() {
    let msl = generate_int8_gemm_msl(&info_no_bias());
    // Output C should be at buffer(4) when no bias.
    assert!(
        msl.contains("device float*        C              [[buffer(4)]]"),
        "output C at buffer(4) without bias"
    );
}

#[test]
fn test_int8_gemm_output_buffer_index_with_bias() {
    let msl = generate_int8_gemm_msl(&info_with_bias());
    // Output C should be at buffer(5) when bias present.
    assert!(
        msl.contains("device float*        C              [[buffer(5)]]"),
        "output C at buffer(5) with bias"
    );
}

#[test]
fn test_int8_gemm_bias_param_present_when_has_bias() {
    let msl = generate_int8_gemm_msl(&info_with_bias());
    assert!(
        msl.contains("device const float* bias           [[buffer(4)]]"),
        "bias parameter at buffer(4)"
    );
}

#[test]
fn test_int8_gemm_bias_param_absent_when_no_bias() {
    let msl = generate_int8_gemm_msl(&info_no_bias());
    assert!(
        !msl.contains("device const float* bias"),
        "no bias parameter declaration without has_bias"
    );
    assert!(
        !msl.contains("val += bias[gc]"),
        "no bias addition without has_bias"
    );
}

// ---- Dimension constants ----

#[test]
fn test_int8_gemm_msl_dimension_constants() {
    let info = Int8GemmInfo { m: 32, k: 512, n: 1024, has_bias: false };
    let msl = generate_int8_gemm_msl(&info);
    assert!(msl.contains("constant uint M_DIM = 32u;"), "M_DIM embedded");
    assert!(msl.contains("constant uint K_DIM = 512u;"), "K_DIM embedded");
    assert!(msl.contains("constant uint N_DIM = 1024u;"), "N_DIM embedded");
}

#[test]
fn test_int8_gemm_msl_dimension_constants_different_values() {
    let info = Int8GemmInfo { m: 1, k: 768, n: 3072, has_bias: true };
    let msl = generate_int8_gemm_msl(&info);
    assert!(msl.contains("constant uint M_DIM = 1u;"));
    assert!(msl.contains("constant uint K_DIM = 768u;"));
    assert!(msl.contains("constant uint N_DIM = 3072u;"));
}

// ---- int8_gemm_input_count ----

#[test]
fn test_int8_gemm_input_count_without_bias() {
    assert_eq!(int8_gemm_input_count(false), 4);
}

#[test]
fn test_int8_gemm_input_count_with_bias() {
    assert_eq!(int8_gemm_input_count(true), 5);
}

// ---- int8_gemm_threadgroup_bytes ----

#[test]
fn test_int8_gemm_threadgroup_bytes_value() {
    // As (half): 32 * 33 * 2 = 2112
    // Ws (half): 32 * 33 * 2 = 2112
    // tile_out (float): 32 * 33 * 4 = 4224
    // Total: 2112 + 2112 + 4224 = 8448
    assert_eq!(int8_gemm_threadgroup_bytes(), 8448);
}

// ---- Bias-add logic in MSL output section ----

#[test]
fn test_int8_gemm_msl_bias_add_present_when_has_bias() {
    let msl = generate_int8_gemm_msl(&info_with_bias());
    assert!(
        msl.contains("val += bias[gc];"),
        "bias addition must be present when has_bias"
    );
}

#[test]
fn test_int8_gemm_msl_bias_add_absent_when_no_bias() {
    let msl = generate_int8_gemm_msl(&info_no_bias());
    assert!(
        !msl.contains("val += bias[gc];"),
        "bias addition must not be present without has_bias"
    );
}

// ---- Dequantization pattern ----

#[test]
fn test_int8_gemm_msl_contains_dequantization_logic() {
    let msl = generate_int8_gemm_msl(&info_no_bias());
    // Verify the INT8 dequantization sequence is present.
    assert!(msl.contains("as_type<char>(w_raw)"), "reinterpret uchar as signed char");
    assert!(msl.contains("zero_point[gn]"), "per-channel zero point subtraction");
    assert!(msl.contains("scale[gn]"), "per-channel scale multiplication");
}

// ---- Tile constants ----

#[test]
fn test_int8_gemm_msl_tile_constants() {
    let msl = generate_int8_gemm_msl(&info_no_bias());
    assert!(msl.contains("constant uint TILE = 32;"));
    assert!(msl.contains("constant uint SIMD_SIZE = 32;"));
    assert!(msl.contains("constant uint PADDED = TILE + 1;"));
}
