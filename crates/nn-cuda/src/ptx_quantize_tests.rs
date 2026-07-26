// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for PTX quantization kernel generation and CPU references.

use crate::ptx_quantize::{
    dequantize_reference, generate_dequantize_int8_to_f32_ptx, generate_quantize_f32_to_int8_ptx,
    quantize_reference, QUANTIZE_BLOCK_SIZE,
};

// ---------------------------------------------------------------------------
// PTX validity
// ---------------------------------------------------------------------------

#[test]
fn test_quantize_ptx_validity() {
    let ptx = generate_quantize_f32_to_int8_ptx(1024, 0.1, 0);

    assert!(!ptx.is_empty(), "PTX output must not be empty");
    assert!(ptx.contains(".version"), "missing PTX version directive");
    assert!(ptx.contains(".target"), "missing PTX target directive");
    assert!(
        ptx.contains(".address_size 64"),
        "missing 64-bit address size"
    );
    assert!(
        ptx.contains(".visible .entry quantize_f32_to_int8"),
        "missing quantize entry point"
    );
    assert!(ptx.contains("param_input"), "missing input parameter");
    assert!(ptx.contains("param_output"), "missing output parameter");
    assert!(ptx.contains("param_n"), "missing n parameter");
    assert!(ptx.contains("ret;"), "missing return instruction");
    assert!(
        !ptx.contains("__global__"),
        "must not contain CUDA C++ artifacts"
    );
    assert!(!ptx.contains("#include"), "must not contain C/C++ includes");

    // Balanced braces
    let open = ptx.matches('{').count();
    let close = ptx.matches('}').count();
    assert_eq!(
        open, close,
        "unbalanced braces: {open} open vs {close} close"
    );
}

#[test]
fn test_dequantize_ptx_validity() {
    let ptx = generate_dequantize_int8_to_f32_ptx(1024, 0.1, 0);

    assert!(!ptx.is_empty(), "PTX output must not be empty");
    assert!(ptx.contains(".version"), "missing PTX version directive");
    assert!(ptx.contains(".target"), "missing PTX target directive");
    assert!(
        ptx.contains(".address_size 64"),
        "missing 64-bit address size"
    );
    assert!(
        ptx.contains(".visible .entry dequantize_int8_to_f32"),
        "missing dequantize entry point"
    );
    assert!(ptx.contains("param_input"), "missing input parameter");
    assert!(ptx.contains("param_output"), "missing output parameter");
    assert!(ptx.contains("param_n"), "missing n parameter");
    assert!(ptx.contains("ret;"), "missing return instruction");

    // Balanced braces
    let open = ptx.matches('{').count();
    let close = ptx.matches('}').count();
    assert_eq!(
        open, close,
        "unbalanced braces: {open} open vs {close} close"
    );
}

// ---------------------------------------------------------------------------
// Reference roundtrip: quantize -> dequantize ~ original within tolerance
// ---------------------------------------------------------------------------

#[test]
fn test_quantize_dequantize_roundtrip() {
    let scale = 0.1_f32;
    let zero_point = 0_i32;
    let input: Vec<f32> = vec![0.0, 0.5, 1.0, -1.0, 5.0, -5.0, 12.0, -12.5];

    let quantized = quantize_reference(&input, scale, zero_point);
    let dequantized = dequantize_reference(&quantized, scale, zero_point);

    // Roundtrip error bounded by scale/2 (quantization step)
    let tolerance = scale / 2.0 + 1e-6;
    for (i, (&orig, &deq)) in input.iter().zip(dequantized.iter()).enumerate() {
        // Clamped values may differ more, but unclamped values should be close
        if orig.abs() <= 12.7 {
            // Within int8 range for this scale
            assert!(
                (orig - deq).abs() <= tolerance,
                "roundtrip error at index {i}: orig={orig}, dequantized={deq}, tol={tolerance}"
            );
        }
    }
}

#[test]
fn test_quantize_dequantize_roundtrip_with_zero_point() {
    let scale = 0.05_f32;
    let zero_point = 10_i32;
    let input: Vec<f32> = vec![0.0, 1.0, -1.0, 3.0, -3.0];

    let quantized = quantize_reference(&input, scale, zero_point);
    let dequantized = dequantize_reference(&quantized, scale, zero_point);

    let tolerance = scale / 2.0 + 1e-6;
    for (i, (&orig, &deq)) in input.iter().zip(dequantized.iter()).enumerate() {
        if orig.abs() <= 5.9 {
            // Within range for this scale + zero_point
            assert!(
                (orig - deq).abs() <= tolerance,
                "roundtrip error at index {i}: orig={orig}, dequantized={deq}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Clamping to [-128, 127]
// ---------------------------------------------------------------------------

#[test]
fn test_quantize_clamping() {
    let scale = 0.1_f32;
    let zero_point = 0_i32;

    // Values that would exceed int8 range:
    // 100.0 / 0.1 = 1000 -> clamp to 127
    // -100.0 / 0.1 = -1000 -> clamp to -128
    let input = vec![100.0, -100.0, 12.7, -12.8];
    let quantized = quantize_reference(&input, scale, zero_point);

    assert_eq!(quantized[0], 127, "positive overflow must clamp to 127");
    assert_eq!(quantized[1], -128, "negative overflow must clamp to -128");
    assert_eq!(quantized[2], 127, "12.7/0.1=127 should be exactly 127");
    assert_eq!(quantized[3], -128, "-12.8/0.1=-128 should be exactly -128");
}

#[test]
fn test_quantize_clamping_with_zero_point() {
    let scale = 1.0_f32;
    let zero_point = 100_i32;

    // With zero_point=100: q = round(x/1.0) + 100
    // x=50 -> q=150 -> clamp to 127
    // x=-250 -> q=-150 -> clamp to -128
    let input = vec![50.0, -250.0];
    let quantized = quantize_reference(&input, scale, zero_point);

    assert_eq!(quantized[0], 127, "positive with zero_point overflow");
    assert_eq!(quantized[1], -128, "negative with zero_point overflow");
}

// ---------------------------------------------------------------------------
// Zero point correctness
// ---------------------------------------------------------------------------

#[test]
fn test_quantize_zero_point_offset() {
    let scale = 1.0_f32;

    // With zero_point=0: q(0.0) = round(0.0/1.0) + 0 = 0
    let q0 = quantize_reference(&[0.0], scale, 0);
    assert_eq!(q0[0], 0);

    // With zero_point=10: q(0.0) = round(0.0/1.0) + 10 = 10
    let q10 = quantize_reference(&[0.0], scale, 10);
    assert_eq!(q10[0], 10);

    // With zero_point=-5: q(0.0) = round(0.0/1.0) + (-5) = -5
    let qn5 = quantize_reference(&[0.0], scale, -5);
    assert_eq!(qn5[0], -5);

    // Dequantize should recover 0.0 for all zero points
    let dq0 = dequantize_reference(&q0, scale, 0);
    let dq10 = dequantize_reference(&q10, scale, 10);
    let dqn5 = dequantize_reference(&qn5, scale, -5);
    assert!((dq0[0]).abs() < 1e-6, "dequant(0, zp=0) should be 0.0");
    assert!((dq10[0]).abs() < 1e-6, "dequant(10, zp=10) should be 0.0");
    assert!((dqn5[0]).abs() < 1e-6, "dequant(-5, zp=-5) should be 0.0");
}

#[test]
fn test_dequantize_zero_point_applied() {
    let scale = 0.5_f32;
    let zero_point = 20_i32;

    // q=20 should dequantize to 0.0 (since q - zp = 0)
    let deq = dequantize_reference(&[20], scale, zero_point);
    assert!((deq[0]).abs() < 1e-6, "q=zp should dequantize to 0.0");

    // q=30 should dequantize to (30-20)*0.5 = 5.0
    let deq2 = dequantize_reference(&[30], scale, zero_point);
    assert!(
        (deq2[0] - 5.0).abs() < 1e-6,
        "q=30, zp=20, scale=0.5 -> 5.0"
    );
}

// ---------------------------------------------------------------------------
// Different scales
// ---------------------------------------------------------------------------

#[test]
fn test_quantize_different_scales() {
    let zero_point = 0_i32;

    // Scale = 1.0: q(5.0) = round(5.0/1.0) + 0 = 5
    let q1 = quantize_reference(&[5.0], 1.0, zero_point);
    assert_eq!(q1[0], 5);

    // Scale = 0.5: q(5.0) = round(5.0/0.5) + 0 = 10
    let q05 = quantize_reference(&[5.0], 0.5, zero_point);
    assert_eq!(q05[0], 10);

    // Scale = 2.0: q(5.0) = round(5.0/2.0) + 0 = round(2.5) = 2 (banker's rounding)
    // Note: Rust's f32::round() rounds away from zero, so 2.5 -> 3
    let q2 = quantize_reference(&[5.0], 2.0, zero_point);
    assert_eq!(q2[0], 3); // 2.5 rounds to 3 in Rust

    // Scale = 0.01: q(1.0) = round(1.0/0.01) + 0 = 100
    let q001 = quantize_reference(&[1.0], 0.01, zero_point);
    assert_eq!(q001[0], 100);
}

#[test]
fn test_quantize_ptx_different_scales_produce_different_ptx() {
    let ptx_a = generate_quantize_f32_to_int8_ptx(1024, 0.1, 0);
    let ptx_b = generate_quantize_f32_to_int8_ptx(1024, 0.5, 0);
    assert_ne!(ptx_a, ptx_b, "different scales must produce different PTX");
}

#[test]
fn test_dequantize_ptx_different_zero_points_produce_different_ptx() {
    let ptx_a = generate_dequantize_int8_to_f32_ptx(1024, 0.1, 0);
    let ptx_b = generate_dequantize_int8_to_f32_ptx(1024, 0.1, 10);
    assert_ne!(
        ptx_a, ptx_b,
        "different zero_points must produce different PTX"
    );
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#[test]
fn test_quantize_block_size() {
    assert_eq!(QUANTIZE_BLOCK_SIZE, 256);
}

// ---------------------------------------------------------------------------
// PTX contains grid-stride loop structure
// ---------------------------------------------------------------------------

#[test]
fn test_quantize_ptx_has_grid_stride_loop() {
    let ptx = generate_quantize_f32_to_int8_ptx(1024, 0.1, 0);
    assert!(ptx.contains("Q_LOOP:"), "quantize must have loop label");
    assert!(ptx.contains("Q_EXIT:"), "quantize must have exit label");
}

#[test]
fn test_dequantize_ptx_has_grid_stride_loop() {
    let ptx = generate_dequantize_int8_to_f32_ptx(1024, 0.1, 0);
    assert!(ptx.contains("DQ_LOOP:"), "dequantize must have loop label");
    assert!(ptx.contains("DQ_EXIT:"), "dequantize must have exit label");
}

// ---------------------------------------------------------------------------
// PTX contains s8 load/store for int8
// ---------------------------------------------------------------------------

#[test]
fn test_quantize_ptx_uses_s8_store() {
    let ptx = generate_quantize_f32_to_int8_ptx(256, 0.1, 0);
    assert!(
        ptx.contains("st.global.s8"),
        "quantize PTX must store as s8"
    );
    assert!(
        ptx.contains("ld.global.f32"),
        "quantize PTX must load as f32"
    );
}

#[test]
fn test_dequantize_ptx_uses_s8_load() {
    let ptx = generate_dequantize_int8_to_f32_ptx(256, 0.1, 0);
    assert!(
        ptx.contains("ld.global.s8"),
        "dequantize PTX must load as s8"
    );
    assert!(
        ptx.contains("st.global.f32"),
        "dequantize PTX must store as f32"
    );
}

// ---------------------------------------------------------------------------
// Edge: empty input
// ---------------------------------------------------------------------------

#[test]
fn test_quantize_empty_input() {
    let q = quantize_reference(&[], 1.0, 0);
    assert!(q.is_empty());
}

#[test]
fn test_dequantize_empty_input() {
    let d = dequantize_reference(&[], 1.0, 0);
    assert!(d.is_empty());
}

// ---------------------------------------------------------------------------
// PTX header contains parameter metadata
// ---------------------------------------------------------------------------

#[test]
fn test_quantize_ptx_header_metadata() {
    let ptx = generate_quantize_f32_to_int8_ptx(512, 0.25, 5);
    assert!(ptx.contains("n=512"), "header should contain n");
    assert!(ptx.contains("scale=0.25"), "header should contain scale");
    assert!(
        ptx.contains("zero_point=5"),
        "header should contain zero_point"
    );
}

#[test]
fn test_dequantize_ptx_header_metadata() {
    let ptx = generate_dequantize_int8_to_f32_ptx(512, 0.25, 5);
    assert!(ptx.contains("n=512"), "header should contain n");
    assert!(ptx.contains("scale=0.25"), "header should contain scale");
    assert!(
        ptx.contains("zero_point=5"),
        "header should contain zero_point"
    );
}
