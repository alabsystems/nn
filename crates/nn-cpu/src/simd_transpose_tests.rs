// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SIMD-optimized 2D matrix transpose.

use super::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff < tol,
            "{label}[{i}]: {x} vs {y}, diff={diff} > tol={tol}"
        );
    }
}

// ---------------------------------------------------------------------------
// test_transpose_square: 4x4 and 8x8 matrices
// ---------------------------------------------------------------------------

#[test]
fn test_transpose_square() {
    // 4x4
    let input_4x4: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let ref_4x4 = transpose_reference(&input_4x4, 4, 4);
    let mut out_4x4 = vec![0.0f32; 16];
    transpose_2d(&input_4x4, &mut out_4x4, 4, 4);
    assert_close(&out_4x4, &ref_4x4, 1e-6, "transpose_4x4");

    // Verify known values: input[r][c] -> output[c][r]
    // input[0][1] = 1 -> output[1][0] = 1 -> out[1*4 + 0] = out[4]
    assert_eq!(out_4x4[4], 1.0, "4x4: element [0][1] should move to [1][0]");

    // 8x8
    let input_8x8: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let ref_8x8 = transpose_reference(&input_8x8, 8, 8);
    let mut out_8x8 = vec![0.0f32; 64];
    transpose_2d(&input_8x8, &mut out_8x8, 8, 8);
    assert_close(&out_8x8, &ref_8x8, 1e-6, "transpose_8x8");
}

// ---------------------------------------------------------------------------
// test_transpose_rectangular: 3x7 and 16x5 matrices
// ---------------------------------------------------------------------------

#[test]
fn test_transpose_rectangular() {
    // 3x7
    let input_3x7: Vec<f32> = (0..21).map(|i| i as f32 * 0.5).collect();
    let ref_3x7 = transpose_reference(&input_3x7, 3, 7);
    let mut out_3x7 = vec![0.0f32; 21];
    transpose_2d(&input_3x7, &mut out_3x7, 3, 7);
    assert_close(&out_3x7, &ref_3x7, 1e-6, "transpose_3x7");

    // 16x5
    let input_16x5: Vec<f32> = (0..80).map(|i| (i as f32 * 0.13).sin()).collect();
    let ref_16x5 = transpose_reference(&input_16x5, 16, 5);
    let mut out_16x5 = vec![0.0f32; 80];
    transpose_2d(&input_16x5, &mut out_16x5, 16, 5);
    assert_close(&out_16x5, &ref_16x5, 1e-6, "transpose_16x5");
}

// ---------------------------------------------------------------------------
// test_transpose_simd_matches_reference: random matrix, compare results
// ---------------------------------------------------------------------------

#[test]
fn test_transpose_simd_matches_reference() {
    let rows = 37;
    let cols = 53;
    let n = rows * cols;

    // Generate pseudo-random data using a simple LCG
    let mut seed: u64 = 12345;
    let input: Vec<f32> = (0..n)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();

    let reference = transpose_reference(&input, rows, cols);
    let mut output = vec![0.0f32; n];
    transpose_2d(&input, &mut output, rows, cols);
    assert_close(&output, &reference, 1e-6, "transpose_random_37x53");
}

// ---------------------------------------------------------------------------
// test_transpose_identity: 1x1, 1xN, Nx1
// ---------------------------------------------------------------------------

#[test]
fn test_transpose_identity() {
    // 1x1
    let input_1x1 = vec![42.0f32];
    let ref_1x1 = transpose_reference(&input_1x1, 1, 1);
    let mut out_1x1 = vec![0.0f32; 1];
    transpose_2d(&input_1x1, &mut out_1x1, 1, 1);
    assert_close(&out_1x1, &ref_1x1, 1e-6, "transpose_1x1");
    assert_eq!(out_1x1[0], 42.0, "1x1 should be identity");

    // 1xN (row vector)
    let n = 16;
    let input_1xn: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let ref_1xn = transpose_reference(&input_1xn, 1, n);
    let mut out_1xn = vec![0.0f32; n];
    transpose_2d(&input_1xn, &mut out_1xn, 1, n);
    assert_close(&out_1xn, &ref_1xn, 1e-6, "transpose_1xN");
    // 1xN transposed = Nx1 -> same elements but layout [col][row] = [i][0] = i*1+0 = i
    for i in 0..n {
        assert_eq!(out_1xn[i], i as f32, "1xN->Nx1: element {i}");
    }

    // Nx1 (column vector)
    let input_nx1: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let ref_nx1 = transpose_reference(&input_nx1, n, 1);
    let mut out_nx1 = vec![0.0f32; n];
    transpose_2d(&input_nx1, &mut out_nx1, n, 1);
    assert_close(&out_nx1, &ref_nx1, 1e-6, "transpose_Nx1");
    for i in 0..n {
        assert_eq!(out_nx1[i], i as f32, "Nx1->1xN: element {i}");
    }
}

// ---------------------------------------------------------------------------
// Double transpose is identity
// ---------------------------------------------------------------------------

#[test]
fn test_transpose_double_is_identity() {
    let rows = 13;
    let cols = 17;
    let n = rows * cols;
    let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.3 - 5.0).collect();

    let mut first = vec![0.0f32; n];
    transpose_2d(&input, &mut first, rows, cols);

    let mut second = vec![0.0f32; n];
    transpose_2d(&first, &mut second, cols, rows);

    assert_close(&second, &input, 1e-6, "double_transpose");
}

// ---------------------------------------------------------------------------
// Large matrix with SIMD blocks
// ---------------------------------------------------------------------------

#[test]
fn test_transpose_large() {
    let rows = 64;
    let cols = 128;
    let n = rows * cols;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.007).sin()).collect();

    let reference = transpose_reference(&input, rows, cols);
    let mut output = vec![0.0f32; n];
    transpose_2d(&input, &mut output, rows, cols);
    assert_close(&output, &reference, 1e-6, "transpose_64x128");
}
