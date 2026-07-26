// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SIMD pooling operations.

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
// max_pool1d
// ---------------------------------------------------------------------------

#[test]
fn test_max_pool1d_basic() {
    // 1 batch, 1 channel, 8 elements, kernel=3, stride=1, padding=0
    // out_len = (8 - 3) / 1 + 1 = 6
    let input = [1.0, 3.0, 2.0, 5.0, 4.0, 1.0, 7.0, 6.0];
    let mut output = [0.0f32; 6];
    max_pool1d(&input, &mut output, 1, 1, 8, 3, 1, 0);

    // windows: [1,3,2]=3, [3,2,5]=5, [2,5,4]=5, [5,4,1]=5, [4,1,7]=7, [1,7,6]=7
    assert_approx(&output, &[3.0, 5.0, 5.0, 5.0, 7.0, 7.0], 1e-6);
}

#[test]
fn test_max_pool1d_stride2() {
    // 1 batch, 1 channel, 8 elements, kernel=2, stride=2, padding=0
    // out_len = (8 - 2) / 2 + 1 = 4
    let input = [1.0, 4.0, 2.0, 3.0, 5.0, 0.0, 7.0, 6.0];
    let mut output = [0.0f32; 4];
    max_pool1d(&input, &mut output, 1, 1, 8, 2, 2, 0);

    // windows: [1,4]=4, [2,3]=3, [5,0]=5, [7,6]=7
    assert_approx(&output, &[4.0, 3.0, 5.0, 7.0], 1e-6);
}

// ---------------------------------------------------------------------------
// avg_pool1d
// ---------------------------------------------------------------------------

#[test]
fn test_avg_pool1d_basic() {
    // 1 batch, 1 channel, 6 elements, kernel=3, stride=1, padding=0
    // out_len = (6 - 3) / 1 + 1 = 4
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut output = [0.0f32; 4];
    avg_pool1d(&input, &mut output, 1, 1, 6, 3, 1, 0);

    // windows: [1,2,3]/3=2, [2,3,4]/3=3, [3,4,5]/3=4, [4,5,6]/3=5
    assert_approx(&output, &[2.0, 3.0, 4.0, 5.0], 1e-6);
}

#[test]
fn test_avg_pool1d_stride2() {
    // 1 batch, 1 channel, 8 elements, kernel=2, stride=2, padding=0
    // out_len = (8 - 2) / 2 + 1 = 4
    let input = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    let mut output = [0.0f32; 4];
    avg_pool1d(&input, &mut output, 1, 1, 8, 2, 2, 0);

    // windows: [2,4]/2=3, [6,8]/2=7, [10,12]/2=11, [14,16]/2=15
    assert_approx(&output, &[3.0, 7.0, 11.0, 15.0], 1e-6);
}

// ---------------------------------------------------------------------------
// max_pool2d
// ---------------------------------------------------------------------------

#[test]
fn test_max_pool2d_basic() {
    // 1 batch, 1 channel, 4x4 input, 2x2 kernel, stride 2, no padding
    // out: 2x2
    #[rustfmt::skip]
    let input = [
        1.0, 3.0, 2.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 2.0, 1.0, 3.0,
        4.0, 10.0, 5.0, 6.0,
    ];
    let mut output = [0.0f32; 4];
    max_pool2d(&input, &mut output, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0);

    // top-left 2x2: max(1,3,5,6)=6
    // top-right 2x2: max(2,4,7,8)=8
    // bottom-left 2x2: max(9,2,4,10)=10
    // bottom-right 2x2: max(1,3,5,6)=6
    assert_approx(&output, &[6.0, 8.0, 10.0, 6.0], 1e-6);
}

#[test]
fn test_max_pool2d_multi_channel() {
    // 1 batch, 2 channels, 4x4 each, 2x2 kernel, stride 2
    #[rustfmt::skip]
    let input = [
        // channel 0
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
        // channel 1
        16.0, 15.0, 14.0, 13.0,
        12.0, 11.0, 10.0, 9.0,
        8.0, 7.0, 6.0, 5.0,
        4.0, 3.0, 2.0, 1.0,
    ];
    let mut output = [0.0f32; 8]; // 2 channels * 2*2
    max_pool2d(&input, &mut output, 1, 2, 4, 4, 2, 2, 2, 2, 0, 0);

    // ch0: max(1,2,5,6)=6, max(3,4,7,8)=8, max(9,10,13,14)=14, max(11,12,15,16)=16
    // ch1: max(16,15,12,11)=16, max(14,13,10,9)=14, max(8,7,4,3)=8, max(6,5,2,1)=6
    assert_approx(&output, &[6.0, 8.0, 14.0, 16.0, 16.0, 14.0, 8.0, 6.0], 1e-6);
}

// ---------------------------------------------------------------------------
// avg_pool2d
// ---------------------------------------------------------------------------

#[test]
fn test_avg_pool2d_basic() {
    // 1 batch, 1 channel, 4x4 input, 2x2 kernel, stride 2, no padding
    #[rustfmt::skip]
    let input = [
        1.0, 3.0, 2.0, 4.0,
        5.0, 7.0, 6.0, 8.0,
        9.0, 11.0, 10.0, 12.0,
        13.0, 15.0, 14.0, 16.0,
    ];
    let mut output = [0.0f32; 4];
    avg_pool2d(&input, &mut output, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0);

    // top-left: (1+3+5+7)/4=4
    // top-right: (2+4+6+8)/4=5
    // bottom-left: (9+11+13+15)/4=12
    // bottom-right: (10+12+14+16)/4=13
    assert_approx(&output, &[4.0, 5.0, 12.0, 13.0], 1e-6);
}

// ---------------------------------------------------------------------------
// Pooling with padding
// ---------------------------------------------------------------------------

#[test]
fn test_pool_with_padding_max1d() {
    // 1 batch, 1 channel, 4 elements, kernel=3, stride=1, padding=1
    // out_len = (4 + 2*1 - 3) / 1 + 1 = 4
    let input = [2.0, 5.0, 1.0, 3.0];
    let mut output = [0.0f32; 4];
    max_pool1d(&input, &mut output, 1, 1, 4, 3, 1, 1);

    // padded input: [-inf, 2, 5, 1, 3, -inf]
    // windows: [-inf,2,5]=5, [2,5,1]=5, [5,1,3]=5, [1,3,-inf]=3
    assert_approx(&output, &[5.0, 5.0, 5.0, 3.0], 1e-6);
}

#[test]
fn test_pool_with_padding_avg1d() {
    // 1 batch, 1 channel, 4 elements, kernel=3, stride=1, padding=1
    // out_len = (4 + 2*1 - 3) / 1 + 1 = 4
    let input = [3.0, 6.0, 9.0, 12.0];
    let mut output = [0.0f32; 4];
    avg_pool1d(&input, &mut output, 1, 1, 4, 3, 1, 1);

    // padded input: [0, 3, 6, 9, 12, 0]
    // windows (div by kernel_size=3):
    //   [0,3,6]/3=3, [3,6,9]/3=6, [6,9,12]/3=9, [9,12,0]/3=7
    assert_approx(&output, &[3.0, 6.0, 9.0, 7.0], 1e-6);
}

#[test]
fn test_pool_with_padding_max2d() {
    // 1 batch, 1 channel, 2x2 input, 2x2 kernel, stride=1, padding=1
    // out_h = (2 + 2 - 2) / 1 + 1 = 3
    // out_w = (2 + 2 - 2) / 1 + 1 = 3
    #[rustfmt::skip]
    let input = [
        1.0, 2.0,
        3.0, 4.0,
    ];
    let mut output = [0.0f32; 9];
    max_pool2d(&input, &mut output, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1);

    // Padded 4x4 (with -inf padding):
    //   -inf -inf -inf -inf
    //   -inf  1    2   -inf
    //   -inf  3    4   -inf
    //   -inf -inf -inf -inf
    //
    // 2x2 windows with stride 1 → 3x3 output:
    // (0,0): max(-inf,-inf,-inf,1) = 1
    // (0,1): max(-inf,-inf,1,2)   = 2
    // (0,2): max(-inf,-inf,2,-inf)= 2
    // (1,0): max(-inf,1,-inf,3)   = 3
    // (1,1): max(1,2,3,4)         = 4
    // (1,2): max(2,-inf,4,-inf)   = 4
    // (2,0): max(-inf,3,-inf,-inf)= 3
    // (2,1): max(3,4,-inf,-inf)   = 4
    // (2,2): max(4,-inf,-inf,-inf)= 4
    assert_approx(
        &output,
        &[1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 3.0, 4.0, 4.0],
        1e-6,
    );
}

// ---------------------------------------------------------------------------
// Dispatch matches reference
// ---------------------------------------------------------------------------

#[test]
fn test_max_pool1d_matches_reference() {
    let input: Vec<f32> = (0..48).map(|i| (i as f32 * 0.7).sin()).collect();
    let out_len = pool_output_len(48, 4, 2, 0);
    let mut out_dispatch = vec![0.0f32; out_len];
    let mut out_ref = vec![0.0f32; out_len];
    max_pool1d(&input, &mut out_dispatch, 1, 1, 48, 4, 2, 0);
    max_pool1d_reference(&input, &mut out_ref, 1, 1, 48, 4, 2, 0);
    assert_approx(&out_dispatch, &out_ref, 1e-6);
}

#[test]
fn test_avg_pool1d_matches_reference() {
    let input: Vec<f32> = (0..48).map(|i| (i as f32 * 0.3).cos()).collect();
    let out_len = pool_output_len(48, 4, 2, 0);
    let mut out_dispatch = vec![0.0f32; out_len];
    let mut out_ref = vec![0.0f32; out_len];
    avg_pool1d(&input, &mut out_dispatch, 1, 1, 48, 4, 2, 0);
    avg_pool1d_reference(&input, &mut out_ref, 1, 1, 48, 4, 2, 0);
    assert_approx(&out_dispatch, &out_ref, 1e-5);
}

#[test]
fn test_max_pool2d_matches_reference() {
    let input: Vec<f32> = (0..64).map(|i| (i as f32 * 0.5).sin()).collect();
    let out_h = pool_output_len(8, 2, 2, 0);
    let out_w = pool_output_len(8, 2, 2, 0);
    let mut out_dispatch = vec![0.0f32; out_h * out_w];
    let mut out_ref = vec![0.0f32; out_h * out_w];
    max_pool2d(&input, &mut out_dispatch, 1, 1, 8, 8, 2, 2, 2, 2, 0, 0);
    max_pool2d_reference(&input, &mut out_ref, 1, 1, 8, 8, 2, 2, 2, 2, 0, 0);
    assert_approx(&out_dispatch, &out_ref, 1e-6);
}

#[test]
fn test_avg_pool2d_matches_reference() {
    let input: Vec<f32> = (0..64).map(|i| (i as f32 * 0.2).cos()).collect();
    let out_h = pool_output_len(8, 2, 2, 0);
    let out_w = pool_output_len(8, 2, 2, 0);
    let mut out_dispatch = vec![0.0f32; out_h * out_w];
    let mut out_ref = vec![0.0f32; out_h * out_w];
    avg_pool2d(&input, &mut out_dispatch, 1, 1, 8, 8, 2, 2, 2, 2, 0, 0);
    avg_pool2d_reference(&input, &mut out_ref, 1, 1, 8, 8, 2, 2, 2, 2, 0, 0);
    assert_approx(&out_dispatch, &out_ref, 1e-5);
}
