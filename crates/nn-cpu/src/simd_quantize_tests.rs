// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SIMD quantization operations.

use super::*;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn assert_approx_f32(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "index {i}: actual={a}, expected={e}, diff={}",
            (a - e).abs()
        );
    }
}

fn assert_eq_i8(actual: &[i8], expected: &[i8]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(a, e, "index {i}: actual={a}, expected={e}");
    }
}

// ---------------------------------------------------------------------------
// quantize_f32_to_i8 — basic
// ---------------------------------------------------------------------------

#[test]
fn test_quantize_basic() {
    // scale=0.1, zero_point=0
    // q = clamp(round(x / 0.1) + 0, -128, 127)
    let input = [0.0, 0.1, 0.5, 1.0, -0.3, -1.0, 2.5];
    let mut output = [0i8; 7];
    quantize_f32_to_i8(&input, &mut output, 0.1, 0);

    // 0.0/0.1=0 → 0, 0.1/0.1=1 → 1, 0.5/0.1=5 → 5, 1.0/0.1=10 → 10,
    // -0.3/0.1=-3 → -3, -1.0/0.1=-10 → -10, 2.5/0.1=25 → 25
    assert_eq_i8(&output, &[0, 1, 5, 10, -3, -10, 25]);
}

#[test]
fn test_quantize_with_zero_point() {
    // scale=0.5, zero_point=10
    // q = clamp(round(x / 0.5) + 10, -128, 127)
    let input = [0.0, 1.0, -1.0, 5.0];
    let mut output = [0i8; 4];
    quantize_f32_to_i8(&input, &mut output, 0.5, 10);

    // 0.0/0.5+10=10, 1.0/0.5+10=12, -1.0/0.5+10=8, 5.0/0.5+10=20
    assert_eq_i8(&output, &[10, 12, 8, 20]);
}

// ---------------------------------------------------------------------------
// dequantize_i8_to_f32 — basic
// ---------------------------------------------------------------------------

#[test]
fn test_dequantize_basic() {
    // scale=0.1, zero_point=0
    // x = (q - 0) * 0.1
    let input = [0i8, 1, 5, 10, -3, -10, 25];
    let mut output = [0.0f32; 7];
    dequantize_i8_to_f32(&input, &mut output, 0.1, 0);

    assert_approx_f32(&output, &[0.0, 0.1, 0.5, 1.0, -0.3, -1.0, 2.5], 1e-6);
}

#[test]
fn test_dequantize_with_zero_point() {
    // scale=0.5, zero_point=10
    // x = (q - 10) * 0.5
    let input = [10i8, 12, 8, 20];
    let mut output = [0.0f32; 4];
    dequantize_i8_to_f32(&input, &mut output, 0.5, 10);

    assert_approx_f32(&output, &[0.0, 1.0, -1.0, 5.0], 1e-6);
}

// ---------------------------------------------------------------------------
// Roundtrip: quantize → dequantize
// ---------------------------------------------------------------------------

#[test]
fn test_quantize_dequantize_roundtrip() {
    // Quantize then dequantize should be close to original (within quantization step).
    let scale = 0.1;
    let zero_point = 0i8;
    let input = [0.0, 0.15, 0.5, 1.0, -0.3, -1.0, 2.5, -5.0, 10.0, 12.7];
    let mut quantized = [0i8; 10];
    let mut dequantized = [0.0f32; 10];

    quantize_f32_to_i8(&input, &mut quantized, scale, zero_point);
    dequantize_i8_to_f32(&quantized, &mut dequantized, scale, zero_point);

    // Maximum error is 0.5 * scale = 0.05 (rounding error)
    let tol = 0.5 * scale + 1e-6;
    for (i, (&original, &recovered)) in input.iter().zip(dequantized.iter()).enumerate() {
        // Values that get clamped will have larger error
        let expected_q = (original / scale).round();
        if (-128.0..=127.0).contains(&expected_q) {
            assert!(
                (original - recovered).abs() < tol,
                "index {i}: original={original}, recovered={recovered}, diff={}",
                (original - recovered).abs()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Clamping: values outside i8 range
// ---------------------------------------------------------------------------

#[test]
fn test_quantize_clamping() {
    // scale=1.0, zero_point=0 — values outside [-128, 127] clamp.
    let input = [200.0, -200.0, 127.0, -128.0, 127.4, -128.4, 500.0];
    let mut output = [0i8; 7];
    quantize_f32_to_i8(&input, &mut output, 1.0, 0);

    // 200→127(clamp), -200→-128(clamp), 127→127, -128→-128,
    // 127.4→round(127.4)=127→127, -128.4→round(-128.4)=-128→-128, 500→127
    assert_eq_i8(&output, &[127, -128, 127, -128, 127, -128, 127]);
}

#[test]
fn test_quantize_clamping_with_zero_point() {
    // scale=1.0, zero_point=100
    // q = round(x/1.0) + 100 clamped to [-128, 127]
    let input = [100.0, -100.0, 0.0, 27.0, -228.0];
    let mut output = [0i8; 5];
    quantize_f32_to_i8(&input, &mut output, 1.0, 100);

    // 100+100=200→127, -100+100=0→0, 0+100=100→100, 27+100=127→127, -228+100=-128→-128
    assert_eq_i8(&output, &[127, 0, 100, 127, -128]);
}

// ---------------------------------------------------------------------------
// Per-channel quantization
// ---------------------------------------------------------------------------

#[test]
fn test_per_channel_quantize() {
    // 2 channels, 4 elements each
    let input = [
        // Channel 0: scale=0.1, zp=0
        0.0, 0.1, 0.5, 1.0, // Channel 1: scale=0.5, zp=10
        0.0, 1.0, -1.0, 5.0,
    ];
    let scales = [0.1, 0.5];
    let zero_points = [0i8, 10];
    let mut output = [0i8; 8];

    quantize_per_channel(&input, &mut output, &scales, &zero_points, 2, 4);

    // Channel 0: [0, 1, 5, 10]
    // Channel 1: [10, 12, 8, 20]
    assert_eq_i8(&output, &[0, 1, 5, 10, 10, 12, 8, 20]);
}

#[test]
fn test_per_channel_quantize_matches_reference() {
    let channels = 3;
    let elems = 16;
    let input: Vec<f32> = (0..(channels * elems))
        .map(|i| (i as f32 * 0.3).sin() * 5.0)
        .collect();
    let scales = [0.1, 0.2, 0.05];
    let zero_points = [0i8, 5, -10];
    let mut out_dispatch = vec![0i8; channels * elems];
    let mut out_ref = vec![0i8; channels * elems];

    quantize_per_channel(
        &input,
        &mut out_dispatch,
        &scales,
        &zero_points,
        channels,
        elems,
    );
    quantize_per_channel_reference(&input, &mut out_ref, &scales, &zero_points, channels, elems);

    assert_eq_i8(&out_dispatch, &out_ref);
}

// ---------------------------------------------------------------------------
// Dispatch matches reference
// ---------------------------------------------------------------------------

#[test]
fn test_quantize_dispatch_matches_reference() {
    let input: Vec<f32> = (0..33).map(|i| (i as f32 - 16.0) * 0.3).collect();
    let mut out_dispatch = vec![0i8; 33];
    let mut out_ref = vec![0i8; 33];
    quantize_f32_to_i8(&input, &mut out_dispatch, 0.1, 5);
    quantize_f32_to_i8_reference(&input, &mut out_ref, 0.1, 5);
    assert_eq_i8(&out_dispatch, &out_ref);
}

#[test]
fn test_dequantize_dispatch_matches_reference() {
    let input: Vec<i8> = (-16..17).collect();
    let mut out_dispatch = vec![0.0f32; 33];
    let mut out_ref = vec![0.0f32; 33];
    dequantize_i8_to_f32(&input, &mut out_dispatch, 0.25, 3);
    dequantize_i8_to_f32_reference(&input, &mut out_ref, 0.25, 3);
    assert_approx_f32(&out_dispatch, &out_ref, 1e-6);
}

// ---------------------------------------------------------------------------
// Large input (exercises main SIMD loop + scalar tail)
// ---------------------------------------------------------------------------

#[test]
fn test_quantize_large_input() {
    let n = 1024 + 7;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin() * 10.0).collect();
    let mut out_dispatch = vec![0i8; n];
    let mut out_ref = vec![0i8; n];
    quantize_f32_to_i8(&input, &mut out_dispatch, 0.1, 0);
    quantize_f32_to_i8_reference(&input, &mut out_ref, 0.1, 0);
    assert_eq_i8(&out_dispatch, &out_ref);
}

#[test]
fn test_dequantize_large_input() {
    let n = 512 + 3;
    let input: Vec<i8> = (0..n).map(|i| (i % 256) as u8 as i8).collect();
    let mut out_dispatch = vec![0.0f32; n];
    let mut out_ref = vec![0.0f32; n];
    dequantize_i8_to_f32(&input, &mut out_dispatch, 0.05, -10);
    dequantize_i8_to_f32_reference(&input, &mut out_ref, 0.05, -10);
    assert_approx_f32(&out_dispatch, &out_ref, 1e-6);
}
