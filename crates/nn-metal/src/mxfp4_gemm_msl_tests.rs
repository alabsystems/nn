// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for MXFP4 dequantizing GEMM MSL code generation.

use super::{generate_mxfp4_gemm_msl, mxfp4_gemm_input_count, mxfp4_gemm_threadgroup_bytes, Mxfp4GemmInfo};

fn info_no_bias() -> Mxfp4GemmInfo {
    Mxfp4GemmInfo {
        m: 64,
        k: 128,
        n: 256,
        block_size: 32,
        has_bias: false,
    }
}

fn info_with_bias() -> Mxfp4GemmInfo {
    Mxfp4GemmInfo {
        m: 64,
        k: 128,
        n: 256,
        block_size: 32,
        has_bias: true,
    }
}

// ---- MSL generation validity ----

#[test]
fn test_msl_generation_valid() {
    let msl = generate_mxfp4_gemm_msl(&info_no_bias());
    assert!(
        msl.contains("kernel void mxfp4_matmul_dequant("),
        "MSL must contain the kernel function name"
    );
    assert!(msl.contains("#include <metal_stdlib>"));
    assert!(msl.contains("using namespace metal;"));
    // Verify dimension constants are embedded.
    assert!(msl.contains("constant uint M_DIM = 64u;"), "M_DIM embedded");
    assert!(msl.contains("constant uint K_DIM = 128u;"), "K_DIM embedded");
    assert!(msl.contains("constant uint N_DIM = 256u;"), "N_DIM embedded");
    assert!(
        msl.contains("constant uint BLOCK_SIZE = 32u;"),
        "BLOCK_SIZE embedded"
    );
}

// ---- FP4 LUT completeness ----

#[test]
fn test_fp4_lut_completeness() {
    let msl = generate_mxfp4_gemm_msl(&info_no_bias());
    // The LUT must contain all 16 entries (8 positive + 8 negative).
    assert!(msl.contains("constant float fp4_lut[16]"), "LUT declaration");
    // Check representative positive values.
    assert!(msl.contains("0.0f"), "zero");
    assert!(msl.contains("0.5f"), "0.5");
    assert!(msl.contains("1.0f"), "1.0");
    assert!(msl.contains("1.5f"), "1.5");
    assert!(msl.contains("2.0f"), "2.0");
    assert!(msl.contains("3.0f"), "3.0");
    assert!(msl.contains("4.0f"), "4.0");
    assert!(msl.contains("6.0f"), "6.0");
    // Check negative values present.
    assert!(msl.contains("-0.5f"), "-0.5");
    assert!(msl.contains("-6.0f"), "-6.0");
}

// ---- Buffer count ----

#[test]
fn test_buffer_count_with_bias() {
    // A + packed_weights + shared_exponents + bias = 4 input buffers.
    assert_eq!(mxfp4_gemm_input_count(true), 4);
}

#[test]
fn test_buffer_count_without_bias() {
    // A + packed_weights + shared_exponents = 3 input buffers.
    assert_eq!(mxfp4_gemm_input_count(false), 3);
}

// ---- Threadgroup bytes ----

#[test]
fn test_threadgroup_bytes() {
    // Naive per-element kernel uses no threadgroup memory.
    assert_eq!(mxfp4_gemm_threadgroup_bytes(), 0);
}

// ---- Buffer binding indices ----

#[test]
fn test_buffer_bindings_no_bias() {
    let msl = generate_mxfp4_gemm_msl(&info_no_bias());
    assert!(msl.contains("[[buffer(0)]]"), "A activation buffer");
    assert!(msl.contains("[[buffer(1)]]"), "packed_w buffer");
    assert!(msl.contains("[[buffer(2)]]"), "shared_exp buffer");
    assert!(msl.contains("[[buffer(3)]]"), "output buffer (no bias)");
    // No buffer(4) without bias.
    assert!(!msl.contains("[[buffer(4)]]"), "no buffer(4) without bias");
}

#[test]
fn test_buffer_bindings_with_bias() {
    let msl = generate_mxfp4_gemm_msl(&info_with_bias());
    assert!(msl.contains("[[buffer(0)]]"), "A activation buffer");
    assert!(msl.contains("[[buffer(1)]]"), "packed_w buffer");
    assert!(msl.contains("[[buffer(2)]]"), "shared_exp buffer");
    assert!(msl.contains("[[buffer(3)]]"), "bias buffer");
    assert!(msl.contains("[[buffer(4)]]"), "output buffer (with bias)");
}

#[test]
fn test_bias_add_present_when_has_bias() {
    let msl = generate_mxfp4_gemm_msl(&info_with_bias());
    assert!(
        msl.contains("acc += bias[col]"),
        "bias addition must be present when has_bias"
    );
}

#[test]
fn test_bias_add_absent_when_no_bias() {
    let msl = generate_mxfp4_gemm_msl(&info_no_bias());
    assert!(
        !msl.contains("acc += bias[col]"),
        "bias addition must not be present without has_bias"
    );
    assert!(
        !msl.contains("device const float* bias"),
        "no bias parameter without has_bias"
    );
}

// ---- Dequantization pattern ----

#[test]
fn test_dequantization_logic_present() {
    let msl = generate_mxfp4_gemm_msl(&info_no_bias());
    // Nibble unpacking.
    assert!(
        msl.contains("packed_byte & 0x0Fu"),
        "low nibble extraction"
    );
    assert!(msl.contains("packed_byte >> 4u"), "high nibble extraction");
    // LUT lookup.
    assert!(msl.contains("fp4_lut[nibble_lo]"), "LUT lookup low");
    assert!(msl.contains("fp4_lut[nibble_hi]"), "LUT lookup high");
    // Block exponent application.
    assert!(
        msl.contains("SHARED_EXP_BIAS"),
        "shared exponent bias constant"
    );
    assert!(msl.contains("exp2("), "power-of-two block scale");
}

// ---- Different dimensions ----

#[test]
fn test_different_dimensions() {
    let info = Mxfp4GemmInfo {
        m: 1,
        k: 768,
        n: 3072,
        block_size: 32,
        has_bias: true,
    };
    let msl = generate_mxfp4_gemm_msl(&info);
    assert!(msl.contains("constant uint M_DIM = 1u;"));
    assert!(msl.contains("constant uint K_DIM = 768u;"));
    assert!(msl.contains("constant uint N_DIM = 3072u;"));
    assert!(
        msl.contains("constant uint BLOCKS_PER_ROW = 24u;"),
        "768/32 = 24 blocks"
    );
}
