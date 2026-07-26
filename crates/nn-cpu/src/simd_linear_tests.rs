// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SIMD linear layer operations.

use super::*;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn assert_approx(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "index {i}: actual={a}, expected={e}, diff={}",
            (a - e).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// test_linear_identity
// ---------------------------------------------------------------------------

#[test]
fn test_linear_identity() {
    // 3x3 identity weight matrix, zero bias
    let in_features = 3;
    let out_features = 3;
    #[rustfmt::skip]
    let weight = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ];
    let bias = [0.0, 0.0, 0.0];
    let input = [2.0, 3.0, 5.0];
    let mut output = [0.0f32; 3];

    linear(
        &input,
        &weight,
        &bias,
        &mut output,
        in_features,
        out_features,
    );
    assert_approx(&output, &[2.0, 3.0, 5.0], 1e-6);
}

// ---------------------------------------------------------------------------
// test_linear_with_bias
// ---------------------------------------------------------------------------

#[test]
fn test_linear_with_bias() {
    // 2x3 weight matrix (out=2, in=3)
    let in_features = 3;
    let out_features = 2;
    #[rustfmt::skip]
    let weight = [
        1.0, 2.0, 3.0,  // row 0
        4.0, 5.0, 6.0,  // row 1
    ];
    let bias = [0.5, -0.5];
    let input = [1.0, 1.0, 1.0];
    let mut output = [0.0f32; 2];

    linear(
        &input,
        &weight,
        &bias,
        &mut output,
        in_features,
        out_features,
    );
    // output[0] = 1*1 + 2*1 + 3*1 + 0.5 = 6.5
    // output[1] = 4*1 + 5*1 + 6*1 - 0.5 = 14.5
    assert_approx(&output, &[6.5, 14.5], 1e-6);
}

// ---------------------------------------------------------------------------
// test_linear_no_bias
// ---------------------------------------------------------------------------

#[test]
fn test_linear_no_bias() {
    let in_features = 3;
    let out_features = 2;
    #[rustfmt::skip]
    let weight = [
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ];
    let bias = [0.0, 0.0];
    let input = [1.0, 1.0, 1.0];
    let mut out_no_bias = [0.0f32; 2];
    let mut out_with_zero_bias = [0.0f32; 2];

    linear_no_bias(&input, &weight, &mut out_no_bias, in_features, out_features);
    linear(
        &input,
        &weight,
        &bias,
        &mut out_with_zero_bias,
        in_features,
        out_features,
    );

    assert_approx(&out_no_bias, &out_with_zero_bias, 1e-6);
    // Also check known values: 1+2+3=6, 4+5+6=15
    assert_approx(&out_no_bias, &[6.0, 15.0], 1e-6);
}

// ---------------------------------------------------------------------------
// test_linear_batched
// ---------------------------------------------------------------------------

#[test]
fn test_linear_batched() {
    let batch = 3;
    let in_features = 2;
    let out_features = 2;
    #[rustfmt::skip]
    let weight = [
        1.0, 0.0,  // row 0: pass-through x[0]
        0.0, 1.0,  // row 1: pass-through x[1]
    ];
    let bias = [10.0, 20.0];
    #[rustfmt::skip]
    let input = [
        1.0, 2.0,  // batch 0
        3.0, 4.0,  // batch 1
        5.0, 6.0,  // batch 2
    ];
    let mut output = [0.0f32; 6];

    linear_batched(
        &input,
        &weight,
        &bias,
        &mut output,
        batch,
        in_features,
        out_features,
    );
    #[rustfmt::skip]
    let expected = [
        11.0, 22.0,  // 1+10, 2+20
        13.0, 24.0,  // 3+10, 4+20
        15.0, 26.0,  // 5+10, 6+20
    ];
    assert_approx(&output, &expected, 1e-6);
}

// ---------------------------------------------------------------------------
// test_linear_matches_matmul
// ---------------------------------------------------------------------------

#[test]
fn test_linear_matches_matmul() {
    // Compare linear against manual matmul + bias
    let in_features = 4;
    let out_features = 3;
    #[rustfmt::skip]
    let weight = [
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 1.1, 1.2,
    ];
    let bias = [0.01, 0.02, 0.03];
    let input = [1.0, 2.0, 3.0, 4.0];
    let mut output = [0.0f32; 3];

    linear(
        &input,
        &weight,
        &bias,
        &mut output,
        in_features,
        out_features,
    );

    // Manual computation:
    // output[0] = 0.1*1 + 0.2*2 + 0.3*3 + 0.4*4 + 0.01 = 0.1+0.4+0.9+1.6+0.01 = 3.01
    // output[1] = 0.5*1 + 0.6*2 + 0.7*3 + 0.8*4 + 0.02 = 0.5+1.2+2.1+3.2+0.02 = 7.02
    // output[2] = 0.9*1 + 1.0*2 + 1.1*3 + 1.2*4 + 0.03 = 0.9+2.0+3.3+4.8+0.03 = 11.03
    assert_approx(&output, &[3.01, 7.02, 11.03], 1e-5);
}

// ---------------------------------------------------------------------------
// test_linear_dispatch_matches_reference
// ---------------------------------------------------------------------------

#[test]
fn test_linear_dispatch_matches_reference() {
    let in_features = 33; // not multiple of 4 or 8 — exercises scalar tail
    let out_features = 5;
    let weight: Vec<f32> = (0..out_features * in_features)
        .map(|i| ((i as f32) * 0.01).sin())
        .collect();
    let bias: Vec<f32> = (0..out_features).map(|i| i as f32 * 0.1).collect();
    let input: Vec<f32> = (0..in_features).map(|i| (i as f32 - 16.0) * 0.05).collect();

    let mut out_dispatch = vec![0.0f32; out_features];
    let mut out_ref = vec![0.0f32; out_features];

    linear(
        &input,
        &weight,
        &bias,
        &mut out_dispatch,
        in_features,
        out_features,
    );
    linear_reference(
        &input,
        &weight,
        &bias,
        &mut out_ref,
        in_features,
        out_features,
    );

    assert_approx(&out_dispatch, &out_ref, 1e-4);
}

// ---------------------------------------------------------------------------
// test_linear_no_bias_dispatch_matches_reference
// ---------------------------------------------------------------------------

#[test]
fn test_linear_no_bias_dispatch_matches_reference() {
    let in_features = 17;
    let out_features = 7;
    let weight: Vec<f32> = (0..out_features * in_features)
        .map(|i| ((i as f32) * 0.03).cos())
        .collect();
    let input: Vec<f32> = (0..in_features).map(|i| (i as f32) * 0.2).collect();

    let mut out_dispatch = vec![0.0f32; out_features];
    let mut out_ref = vec![0.0f32; out_features];

    linear_no_bias(
        &input,
        &weight,
        &mut out_dispatch,
        in_features,
        out_features,
    );
    linear_no_bias_reference(&input, &weight, &mut out_ref, in_features, out_features);

    assert_approx(&out_dispatch, &out_ref, 1e-4);
}

// ---------------------------------------------------------------------------
// test_linear_batched_dispatch_matches_reference
// ---------------------------------------------------------------------------

#[test]
fn test_linear_batched_dispatch_matches_reference() {
    let batch = 4;
    let in_features = 16;
    let out_features = 8;
    let weight: Vec<f32> = (0..out_features * in_features)
        .map(|i| ((i as f32) * 0.02).sin())
        .collect();
    let bias: Vec<f32> = (0..out_features).map(|i| i as f32 * 0.05).collect();
    let input: Vec<f32> = (0..batch * in_features)
        .map(|i| ((i as f32) * 0.1).cos())
        .collect();

    let mut out_dispatch = vec![0.0f32; batch * out_features];
    let mut out_ref = vec![0.0f32; batch * out_features];

    linear_batched(
        &input,
        &weight,
        &bias,
        &mut out_dispatch,
        batch,
        in_features,
        out_features,
    );
    linear_batched_reference(
        &input,
        &weight,
        &bias,
        &mut out_ref,
        batch,
        in_features,
        out_features,
    );

    assert_approx(&out_dispatch, &out_ref, 1e-4);
}

// ---------------------------------------------------------------------------
// Large input — exercises SIMD main loop + scalar tail
// ---------------------------------------------------------------------------

#[test]
fn test_linear_large_in_features() {
    let in_features = 512 + 7; // not multiple of 4 or 8
    let out_features = 3;
    let weight: Vec<f32> = (0..out_features * in_features)
        .map(|i| ((i as f32) * 0.001).sin())
        .collect();
    let bias = [0.1, 0.2, 0.3];
    let input: Vec<f32> = (0..in_features).map(|i| (i as f32) * 0.01).collect();

    let mut out_dispatch = vec![0.0f32; out_features];
    let mut out_ref = vec![0.0f32; out_features];

    linear(
        &input,
        &weight,
        &bias,
        &mut out_dispatch,
        in_features,
        out_features,
    );
    linear_reference(
        &input,
        &weight,
        &bias,
        &mut out_ref,
        in_features,
        out_features,
    );

    assert_approx(&out_dispatch, &out_ref, 1e-3);
}
