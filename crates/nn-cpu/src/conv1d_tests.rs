// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for the CPU SIMD conv1d implementation.
//!
//! Covers: basic convolution, strides, padding, multi-channel, groups
//! (depthwise), pointwise (kernel_size=1), im2col path (large kernels),
//! output length formula validation, batch simulation, and SIMD vs scalar
//! reference parity.

use crate::conv1d::{conv1d, conv1d_scalar_reference};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Naive reference conv1d computed entirely inline — independent of the crate
/// implementation. Used as a ground-truth oracle for all tests.
fn naive_conv1d(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_ch: usize,
    out_ch: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    let in_len = input.len() / in_ch;
    let padded_len = in_len + 2 * padding;
    assert!(padded_len >= kernel_size);
    let out_len = (padded_len - kernel_size) / stride + 1;
    let mut output = vec![0.0f32; out_ch * out_len];

    for oc in 0..out_ch {
        for o in 0..out_len {
            let mut acc = 0.0f32;
            for ic in 0..in_ch {
                for k in 0..kernel_size {
                    let in_pos = o * stride + k;
                    if in_pos >= padding && in_pos < padding + in_len {
                        let w_idx = oc * in_ch * kernel_size + ic * kernel_size + k;
                        let i_idx = ic * in_len + (in_pos - padding);
                        acc += weight[w_idx] * input[i_idx];
                    }
                }
            }
            if let Some(b) = bias {
                acc += b[oc];
            }
            output[oc * out_len + o] = acc;
        }
    }
    output
}

/// Assert two slices are element-wise close within `tol`.
fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (va - vb).abs();
        assert!(
            diff <= tol,
            "{label}[{i}]: {va} vs {vb} (diff={diff}, tol={tol})"
        );
    }
}

/// Expected output length from conv1d parameters.
fn expected_out_len(in_len: usize, kernel_size: usize, stride: usize, padding: usize) -> usize {
    (in_len + 2 * padding - kernel_size) / stride + 1
}

// ---------------------------------------------------------------------------
// Basic conv1d (single channel, kernel_size=3, stride=1, no padding)
// ---------------------------------------------------------------------------

#[test]
fn test_basic_conv1d_single_channel() {
    // input: [1, 8], weight: [1, 1, 3]
    let input: Vec<f32> = (1..=8).map(|x| x as f32).collect(); // [1,2,3,4,5,6,7,8]
    let weight = vec![1.0, 0.0, -1.0]; // simple difference filter
    let out = conv1d(&input, &weight, None, 1, 1, 3, 1, 0);

    // out_len = (8 - 3)/1 + 1 = 6
    assert_eq!(out.len(), 6);
    // Manual: out[i] = input[i]*1 + input[i+1]*0 + input[i+2]*(-1)
    //       = input[i] - input[i+2]
    let expected: Vec<f32> = (0..6).map(|i| input[i] - input[i + 2]).collect();
    assert_close(&out, &expected, 1e-6, "basic_single_channel");
}

#[test]
fn test_basic_conv1d_matches_naive() {
    let input: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let weight = vec![0.5, -0.3, 0.2];
    let out = conv1d(&input, &weight, None, 1, 1, 3, 1, 0);
    let expected = naive_conv1d(&input, &weight, None, 1, 1, 3, 1, 0);
    assert_close(&out, &expected, 1e-6, "basic_matches_naive");
}

// ---------------------------------------------------------------------------
// Stride > 1
// ---------------------------------------------------------------------------

#[test]
fn test_conv1d_stride2() {
    let input: Vec<f32> = (1..=10).map(|x| x as f32).collect();
    let weight = vec![1.0, 1.0, 1.0]; // sum of 3 consecutive
    let out = conv1d(&input, &weight, None, 1, 1, 3, 2, 0);

    // out_len = (10 - 3)/2 + 1 = 4
    assert_eq!(out.len(), 4);
    // out[0] = 1+2+3=6, out[1]=3+4+5=12, out[2]=5+6+7=18, out[3]=7+8+9=24
    assert_close(&out, &[6.0, 12.0, 18.0, 24.0], 1e-6, "stride2");
}

#[test]
fn test_conv1d_stride3() {
    let input: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let weight = vec![1.0, -1.0]; // kernel_size=2
    let out = conv1d(&input, &weight, None, 1, 1, 2, 3, 0);

    // out_len = (12 - 2)/3 + 1 = 4
    assert_eq!(out.len(), 4);
    let expected = naive_conv1d(&input, &weight, None, 1, 1, 2, 3, 0);
    assert_close(&out, &expected, 1e-6, "stride3");
}

// ---------------------------------------------------------------------------
// Padding
// ---------------------------------------------------------------------------

#[test]
fn test_conv1d_padding_same_output_length() {
    // kernel_size=3, padding=1, stride=1 -> out_len = in_len
    let input: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let weight = vec![1.0, 1.0, 1.0];
    let out = conv1d(&input, &weight, None, 1, 1, 3, 1, 1);

    let out_len = expected_out_len(8, 3, 1, 1);
    assert_eq!(out_len, 8, "same-padding should preserve length");
    assert_eq!(out.len(), 8);

    // First output: pad(0)*1 + 1*1 + 2*1 = 3
    // Last output: 7*1 + 8*1 + pad(0)*1 = 15
    let expected = naive_conv1d(&input, &weight, None, 1, 1, 3, 1, 1);
    assert_close(&out, &expected, 1e-6, "padding_same");
}

#[test]
fn test_conv1d_large_padding() {
    // padding=3 with kernel_size=3 — output sees mostly zeros at edges
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0, 1.0, 1.0];
    let out = conv1d(&input, &weight, None, 1, 1, 3, 1, 3);

    // out_len = (4 + 6 - 3)/1 + 1 = 8
    assert_eq!(out.len(), 8);
    let expected = naive_conv1d(&input, &weight, None, 1, 1, 3, 1, 3);
    assert_close(&out, &expected, 1e-6, "large_padding");
}

#[test]
fn test_conv1d_padding_with_stride() {
    let input: Vec<f32> = (0..10).map(|x| x as f32).collect();
    let weight = vec![0.5, 0.5, 0.5];
    let out = conv1d(&input, &weight, None, 1, 1, 3, 2, 1);

    // out_len = (10 + 2 - 3)/2 + 1 = 5
    assert_eq!(out.len(), 5);
    let expected = naive_conv1d(&input, &weight, None, 1, 1, 3, 2, 1);
    assert_close(&out, &expected, 1e-6, "padding_with_stride");
}

// ---------------------------------------------------------------------------
// Multiple input/output channels
// ---------------------------------------------------------------------------

#[test]
fn test_conv1d_multi_in_channels() {
    // 2 input channels, 1 output channel, kernel_size=3
    let in_ch = 2;
    let in_len = 6;
    let input: Vec<f32> = (0..in_ch * in_len).map(|x| x as f32 * 0.1).collect();
    let weight = vec![1.0, 0.0, -1.0, 0.5, 0.5, 0.5]; // [1, 2, 3]

    let out = conv1d(&input, &weight, None, in_ch, 1, 3, 1, 0);
    let expected = naive_conv1d(&input, &weight, None, in_ch, 1, 3, 1, 0);
    assert_eq!(out.len(), 4); // (6 - 3)/1 + 1 = 4
    assert_close(&out, &expected, 1e-5, "multi_in_ch");
}

#[test]
fn test_conv1d_multi_out_channels() {
    // 1 input channel, 3 output channels, kernel_size=2
    let in_ch = 1;
    let out_ch = 3;
    let input: Vec<f32> = (1..=5).map(|x| x as f32).collect();
    // weight: [3, 1, 2] = 3 filters of size 2
    let weight = vec![
        1.0, -1.0, // filter 0: difference
        1.0, 1.0, // filter 1: sum
        0.5, 0.5, // filter 2: average
    ];

    let out = conv1d(&input, &weight, None, in_ch, out_ch, 2, 1, 0);
    let expected = naive_conv1d(&input, &weight, None, in_ch, out_ch, 2, 1, 0);
    assert_eq!(out.len(), out_ch * 4); // (5 - 2)/1 + 1 = 4
    assert_close(&out, &expected, 1e-6, "multi_out_ch");
}

#[test]
fn test_conv1d_multi_in_out_channels() {
    // 3 input channels, 2 output channels, kernel_size=3
    let in_ch = 3;
    let out_ch = 2;
    let kernel_size = 3;
    let in_len = 8;
    let input: Vec<f32> = (0..in_ch * in_len)
        .map(|i| (i as f32) * 0.1 - 1.0)
        .collect();
    let weight: Vec<f32> = (0..out_ch * in_ch * kernel_size)
        .map(|i| (i as f32) * 0.05 - 0.4)
        .collect();
    let bias = vec![0.1, -0.2];

    let out = conv1d(
        &input,
        &weight,
        Some(&bias),
        in_ch,
        out_ch,
        kernel_size,
        1,
        0,
    );
    let expected = naive_conv1d(
        &input,
        &weight,
        Some(&bias),
        in_ch,
        out_ch,
        kernel_size,
        1,
        0,
    );
    let out_len = expected_out_len(in_len, kernel_size, 1, 0);
    assert_eq!(out.len(), out_ch * out_len);
    assert_close(&out, &expected, 1e-4, "multi_in_out_ch");
}

// ---------------------------------------------------------------------------
// Groups (depthwise convolution)
// ---------------------------------------------------------------------------

#[test]
fn test_conv1d_depthwise_per_channel() {
    // Depthwise conv: groups = in_ch = out_ch. Each channel uses its own filter.
    // We simulate depthwise by running per-channel conv1d and concatenating.
    let channels = 3;
    let in_len = 8;
    let kernel_size = 3;

    // Per-channel inputs and weights
    let input: Vec<f32> = (0..channels * in_len).map(|i| (i as f32) * 0.2).collect();
    let per_ch_weights: Vec<Vec<f32>> = (0..channels)
        .map(|c| {
            (0..kernel_size)
                .map(|k| ((c * kernel_size + k) as f32) * 0.1 + 0.1)
                .collect()
        })
        .collect();

    // Run each channel independently (in_ch=1, out_ch=1)
    let out_len = expected_out_len(in_len, kernel_size, 1, 0);
    let mut depthwise_out = Vec::with_capacity(channels * out_len);
    for c in 0..channels {
        let ch_input = &input[c * in_len..(c + 1) * in_len];
        let ch_out = conv1d(ch_input, &per_ch_weights[c], None, 1, 1, kernel_size, 1, 0);
        depthwise_out.extend_from_slice(&ch_out);
    }

    // Verify against naive per-channel reference
    let mut expected = Vec::with_capacity(channels * out_len);
    for c in 0..channels {
        let ch_input = &input[c * in_len..(c + 1) * in_len];
        let ch_out = naive_conv1d(ch_input, &per_ch_weights[c], None, 1, 1, kernel_size, 1, 0);
        expected.extend_from_slice(&ch_out);
    }
    assert_close(&depthwise_out, &expected, 1e-5, "depthwise");
}

// ---------------------------------------------------------------------------
// Output length formula validation
// ---------------------------------------------------------------------------

#[test]
fn test_output_length_formula() {
    // Verify the formula: out_len = (in_len + 2*padding - kernel_size) / stride + 1
    let cases: Vec<(usize, usize, usize, usize, usize)> = vec![
        // (in_len, kernel_size, stride, padding, expected_out_len)
        (8, 3, 1, 0, 6),
        (8, 3, 2, 0, 3),
        (8, 3, 1, 1, 8), // same padding
        (8, 5, 1, 2, 8), // same padding for k=5
        (10, 3, 2, 1, 5),
        (10, 1, 1, 0, 10), // pointwise
        (10, 1, 2, 0, 5),  // pointwise with stride
        (3, 3, 1, 0, 1),   // minimal: input == kernel
        (8, 8, 1, 0, 1),   // input == kernel
        (16, 3, 4, 0, 4),
        (7, 3, 1, 0, 5),
        (7, 5, 2, 2, 4),
    ];

    for (in_len, ks, stride, pad, expected) in &cases {
        let out = expected_out_len(*in_len, *ks, *stride, *pad);
        assert_eq!(
            out, *expected,
            "out_len({in_len}, ks={ks}, s={stride}, p={pad}): got {out}, expected {expected}"
        );
    }
}

#[test]
fn test_output_length_matches_actual_output() {
    // Run conv1d and verify the returned length matches the formula.
    let cases: Vec<(usize, usize, usize, usize)> = vec![
        (8, 3, 1, 0),
        (8, 3, 2, 0),
        (8, 3, 1, 1),
        (10, 5, 2, 2),
        (16, 1, 1, 0),
        (3, 3, 1, 0),
    ];

    for (in_len, ks, stride, pad) in cases {
        let input: Vec<f32> = (0..in_len).map(|i| i as f32).collect();
        let weight: Vec<f32> = vec![1.0; ks];
        let out = conv1d(&input, &weight, None, 1, 1, ks, stride, pad);
        let formula_len = expected_out_len(in_len, ks, stride, pad);
        assert_eq!(
            out.len(),
            formula_len,
            "length mismatch for in_len={in_len}, ks={ks}, stride={stride}, pad={pad}"
        );
    }
}

// ---------------------------------------------------------------------------
// Pointwise convolution (kernel_size = 1)
// ---------------------------------------------------------------------------

#[test]
fn test_conv1d_pointwise_single_channel() {
    // kernel_size=1 acts as element-wise scaling
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weight = vec![2.0]; // scale by 2
    let out = conv1d(&input, &weight, None, 1, 1, 1, 1, 0);
    assert_eq!(out.len(), 5);
    let expected: Vec<f32> = input.iter().map(|x| x * 2.0).collect();
    assert_close(&out, &expected, 1e-6, "pointwise_single");
}

#[test]
fn test_conv1d_pointwise_multi_channel() {
    // kernel_size=1, 2 in_ch, 3 out_ch: acts like a linear layer per position
    let in_ch = 2;
    let out_ch = 3;
    let in_len = 4;
    let input: Vec<f32> = (0..in_ch * in_len).map(|i| (i + 1) as f32).collect();
    // weight: [3, 2, 1]
    let weight = vec![
        1.0, 0.0, // out_ch 0: take ch0
        0.0, 1.0, // out_ch 1: take ch1
        1.0, 1.0, // out_ch 2: sum both channels
    ];

    let out = conv1d(&input, &weight, None, in_ch, out_ch, 1, 1, 0);
    let expected = naive_conv1d(&input, &weight, None, in_ch, out_ch, 1, 1, 0);
    assert_eq!(out.len(), out_ch * in_len);
    assert_close(&out, &expected, 1e-6, "pointwise_multi_ch");
}

#[test]
fn test_conv1d_pointwise_with_bias() {
    let input = vec![1.0, 2.0, 3.0];
    let weight = vec![1.0];
    let bias = vec![10.0];
    let out = conv1d(&input, &weight, Some(&bias), 1, 1, 1, 1, 0);
    let expected: Vec<f32> = input.iter().map(|x| x + 10.0).collect();
    assert_close(&out, &expected, 1e-6, "pointwise_bias");
}

// ---------------------------------------------------------------------------
// Input length exactly equal to kernel_size
// ---------------------------------------------------------------------------

#[test]
fn test_conv1d_input_equals_kernel() {
    // in_len = kernel_size -> single output position
    let input = vec![1.0, 2.0, 3.0];
    let weight = vec![1.0, 1.0, 1.0];
    let out = conv1d(&input, &weight, None, 1, 1, 3, 1, 0);
    assert_eq!(out.len(), 1);
    assert_close(&out, &[6.0], 1e-6, "input_equals_kernel");
}

#[test]
fn test_conv1d_input_equals_kernel_multi_channel() {
    let in_ch = 2;
    let out_ch = 2;
    let kernel_size = 4;
    let in_len = 4;
    let input: Vec<f32> = (0..in_ch * in_len).map(|i| (i + 1) as f32).collect();
    let weight: Vec<f32> = vec![1.0; out_ch * in_ch * kernel_size];
    let out = conv1d(&input, &weight, None, in_ch, out_ch, kernel_size, 1, 0);
    assert_eq!(out.len(), out_ch); // out_len = 1
    let expected = naive_conv1d(&input, &weight, None, in_ch, out_ch, kernel_size, 1, 0);
    assert_close(&out, &expected, 1e-5, "input_equals_kernel_multi");
}

// ---------------------------------------------------------------------------
// SIMD vs scalar reference parity
// ---------------------------------------------------------------------------

#[test]
fn test_simd_matches_scalar_small() {
    // in_ch=2, out_ch=2, kernel_size=3 => weight size = 2*2*3 = 12
    let input: Vec<f32> = (0..16).map(|i| (i as f32) * 0.3 - 2.0).collect();
    let weight: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 + 0.1).collect();
    let bias = vec![0.5, -0.5];

    let simd_out = conv1d(&input, &weight, Some(&bias), 2, 2, 3, 1, 0);
    let scalar_out = conv1d_scalar_reference(&input, &weight, Some(&bias), 2, 2, 3, 1, 0);
    assert_close(&simd_out, &scalar_out, 1e-5, "simd_vs_scalar_small");
}

#[test]
fn test_simd_matches_scalar_medium() {
    // Larger input to exercise SIMD vector paths (NEON: 4-wide, AVX2: 8-wide)
    let in_ch = 4;
    let out_ch = 8;
    let kernel_size = 5;
    let in_len = 64;
    let input: Vec<f32> = (0..in_ch * in_len)
        .map(|i| ((i * 7 + 3) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let weight: Vec<f32> = (0..out_ch * in_ch * kernel_size)
        .map(|i| ((i * 13 + 11) % 200) as f32 * 0.005 - 0.5)
        .collect();
    let bias: Vec<f32> = (0..out_ch).map(|i| (i as f32) * 0.1 - 0.3).collect();

    let simd_out = conv1d(
        &input,
        &weight,
        Some(&bias),
        in_ch,
        out_ch,
        kernel_size,
        1,
        1,
    );
    let scalar_out = conv1d_scalar_reference(
        &input,
        &weight,
        Some(&bias),
        in_ch,
        out_ch,
        kernel_size,
        1,
        1,
    );
    assert_close(&simd_out, &scalar_out, 1e-4, "simd_vs_scalar_medium");
}

#[test]
fn test_simd_matches_scalar_stride_padding() {
    let in_ch = 3;
    let out_ch = 4;
    let kernel_size = 3;
    let in_len = 20;
    let stride = 2;
    let padding = 1;

    let input: Vec<f32> = (0..in_ch * in_len).map(|i| (i as f32).sin()).collect();
    let weight: Vec<f32> = (0..out_ch * in_ch * kernel_size)
        .map(|i| (i as f32).cos() * 0.3)
        .collect();

    let simd_out = conv1d(
        &input,
        &weight,
        None,
        in_ch,
        out_ch,
        kernel_size,
        stride,
        padding,
    );
    let scalar_out = conv1d_scalar_reference(
        &input,
        &weight,
        None,
        in_ch,
        out_ch,
        kernel_size,
        stride,
        padding,
    );
    assert_close(&simd_out, &scalar_out, 1e-4, "simd_vs_scalar_stride_pad");
}

#[test]
fn test_simd_matches_scalar_non_aligned_length() {
    // Length 13 — not divisible by NEON (4) or AVX2 (8), exercises tail handling
    let in_len = 13;
    let input: Vec<f32> = (0..in_len).map(|i| (i as f32) * 0.7 - 4.0).collect();
    let weight = vec![0.3, -0.6, 0.9];

    let simd_out = conv1d(&input, &weight, None, 1, 1, 3, 1, 0);
    let scalar_out = conv1d_scalar_reference(&input, &weight, None, 1, 1, 3, 1, 0);
    assert_close(&simd_out, &scalar_out, 1e-5, "simd_vs_scalar_non_aligned");
}

// ---------------------------------------------------------------------------
// im2col path (kernel_size > IM2COL_THRESHOLD = 7)
// ---------------------------------------------------------------------------

#[test]
fn test_conv1d_im2col_basic() {
    // kernel_size=8 triggers im2col path
    let kernel_size = 8;
    let in_len = 16;
    let input: Vec<f32> = (0..in_len).map(|i| (i + 1) as f32).collect();
    let weight: Vec<f32> = (0..kernel_size).map(|i| (i as f32) * 0.1 + 0.1).collect();

    let out = conv1d(&input, &weight, None, 1, 1, kernel_size, 1, 0);
    let expected = naive_conv1d(&input, &weight, None, 1, 1, kernel_size, 1, 0);
    let out_len = expected_out_len(in_len, kernel_size, 1, 0);
    assert_eq!(out.len(), out_len);
    assert_close(&out, &expected, 1e-3, "im2col_basic");
}

#[test]
fn test_conv1d_im2col_multi_channel_with_bias() {
    let in_ch = 2;
    let out_ch = 3;
    let kernel_size = 9; // well above threshold
    let in_len = 24;

    let input: Vec<f32> = (0..in_ch * in_len)
        .map(|i| ((i * 3 + 7) % 50) as f32 * 0.04 - 1.0)
        .collect();
    let weight: Vec<f32> = (0..out_ch * in_ch * kernel_size)
        .map(|i| ((i * 5 + 2) % 40) as f32 * 0.025 - 0.5)
        .collect();
    let bias: Vec<f32> = vec![0.1, -0.2, 0.3];

    let out = conv1d(
        &input,
        &weight,
        Some(&bias),
        in_ch,
        out_ch,
        kernel_size,
        1,
        0,
    );
    let expected = naive_conv1d(
        &input,
        &weight,
        Some(&bias),
        in_ch,
        out_ch,
        kernel_size,
        1,
        0,
    );
    assert_close(&out, &expected, 1e-3, "im2col_multi_ch_bias");
}

#[test]
fn test_conv1d_im2col_with_stride_and_padding() {
    let kernel_size = 10;
    let in_len = 32;
    let stride = 3;
    let padding = 4;

    let input: Vec<f32> = (0..in_len).map(|i| (i as f32).sin()).collect();
    let weight: Vec<f32> = (0..kernel_size).map(|i| (i as f32).cos() * 0.2).collect();

    let out = conv1d(&input, &weight, None, 1, 1, kernel_size, stride, padding);
    let expected = naive_conv1d(&input, &weight, None, 1, 1, kernel_size, stride, padding);
    assert_close(&out, &expected, 1e-3, "im2col_stride_padding");
}

#[test]
fn test_conv1d_im2col_matches_scalar_reference() {
    let in_ch = 2;
    let out_ch = 2;
    let kernel_size = 12;
    let in_len = 30;

    let input: Vec<f32> = (0..in_ch * in_len)
        .map(|i| (i as f32) * 0.05 - 1.5)
        .collect();
    let weight: Vec<f32> = (0..out_ch * in_ch * kernel_size)
        .map(|i| ((i * 7) % 100) as f32 * 0.01 - 0.5)
        .collect();

    let simd_out = conv1d(&input, &weight, None, in_ch, out_ch, kernel_size, 2, 3);
    let scalar_out =
        conv1d_scalar_reference(&input, &weight, None, in_ch, out_ch, kernel_size, 2, 3);
    assert_close(&simd_out, &scalar_out, 1e-3, "im2col_vs_scalar");
}

// ---------------------------------------------------------------------------
// Bias
// ---------------------------------------------------------------------------

#[test]
fn test_conv1d_with_bias() {
    let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let weight = vec![1.0, 0.0, 0.0]; // identity-like: picks input[i]
    let bias = vec![100.0];

    let out = conv1d(&input, &weight, Some(&bias), 1, 1, 3, 1, 0);
    let expected = naive_conv1d(&input, &weight, Some(&bias), 1, 1, 3, 1, 0);
    assert_close(&out, &expected, 1e-6, "with_bias");
    // Each output should be input[i] + 100
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - (i as f32 + 100.0)).abs() < 1e-6,
            "bias offset wrong at {i}"
        );
    }
}

#[test]
fn test_conv1d_no_bias_vs_zero_bias() {
    let input: Vec<f32> = (0..10).map(|i| i as f32 * 0.5).collect();
    let weight = vec![1.0, -1.0];
    let zero_bias = vec![0.0];

    let out_no_bias = conv1d(&input, &weight, None, 1, 1, 2, 1, 0);
    let out_zero_bias = conv1d(&input, &weight, Some(&zero_bias), 1, 1, 2, 1, 0);
    assert_close(&out_no_bias, &out_zero_bias, 1e-6, "no_bias_vs_zero_bias");
}

// ---------------------------------------------------------------------------
// Batch simulation
// ---------------------------------------------------------------------------

#[test]
fn test_conv1d_batch_independence() {
    // The API is single-batch [in_ch, in_len]. Verify that running two
    // different inputs independently produces results consistent with
    // separate calls (no state leakage).
    let in_ch = 2;
    let out_ch = 2;
    let kernel_size = 3;
    let in_len = 8;

    let weight: Vec<f32> = (0..out_ch * in_ch * kernel_size)
        .map(|i| (i as f32) * 0.1)
        .collect();

    let input_a: Vec<f32> = (0..in_ch * in_len).map(|i| (i as f32) * 0.2).collect();
    let input_b: Vec<f32> = (0..in_ch * in_len)
        .map(|i| (i as f32) * -0.3 + 1.0)
        .collect();

    let out_a = conv1d(&input_a, &weight, None, in_ch, out_ch, kernel_size, 1, 0);
    let out_b = conv1d(&input_b, &weight, None, in_ch, out_ch, kernel_size, 1, 0);

    // Run again — results should be identical (no state mutation)
    let out_a2 = conv1d(&input_a, &weight, None, in_ch, out_ch, kernel_size, 1, 0);
    let out_b2 = conv1d(&input_b, &weight, None, in_ch, out_ch, kernel_size, 1, 0);

    assert_close(&out_a, &out_a2, 0.0, "batch_a_deterministic");
    assert_close(&out_b, &out_b2, 0.0, "batch_b_deterministic");

    // Outputs for different inputs should differ
    assert!(
        out_a
            .iter()
            .zip(out_b.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6),
        "different inputs should produce different outputs"
    );
}

#[test]
fn test_conv1d_simulated_batch() {
    // Simulate batch processing by running conv1d per batch element
    // and concatenating results.
    let batch_size = 4;
    let in_ch = 2;
    let out_ch = 3;
    let kernel_size = 3;
    let in_len = 8;

    let weight: Vec<f32> = (0..out_ch * in_ch * kernel_size)
        .map(|i| (i as f32) * 0.05 - 0.3)
        .collect();
    let bias: Vec<f32> = vec![0.1, -0.1, 0.0];

    let out_len = expected_out_len(in_len, kernel_size, 1, 0);
    let mut batch_output = Vec::with_capacity(batch_size * out_ch * out_len);

    for b in 0..batch_size {
        let input: Vec<f32> = (0..in_ch * in_len)
            .map(|i| (b * in_ch * in_len + i) as f32 * 0.02 - 0.5)
            .collect();
        let out = conv1d(
            &input,
            &weight,
            Some(&bias),
            in_ch,
            out_ch,
            kernel_size,
            1,
            0,
        );
        assert_eq!(out.len(), out_ch * out_len, "batch elem {b} length");
        batch_output.extend_from_slice(&out);
    }

    assert_eq!(batch_output.len(), batch_size * out_ch * out_len);

    // Verify each batch element against naive reference
    for b in 0..batch_size {
        let input: Vec<f32> = (0..in_ch * in_len)
            .map(|i| (b * in_ch * in_len + i) as f32 * 0.02 - 0.5)
            .collect();
        let expected = naive_conv1d(
            &input,
            &weight,
            Some(&bias),
            in_ch,
            out_ch,
            kernel_size,
            1,
            0,
        );
        let start = b * out_ch * out_len;
        let end = start + out_ch * out_len;
        assert_close(
            &batch_output[start..end],
            &expected,
            1e-4,
            &format!("batch_elem_{b}"),
        );
    }
}

// ---------------------------------------------------------------------------
// Threshold boundary: direct vs im2col
// ---------------------------------------------------------------------------

#[test]
fn test_conv1d_threshold_boundary() {
    // kernel_size=7 should use direct path, kernel_size=8 should use im2col.
    // Both must produce the same results against naive reference.
    let in_len = 24;

    for kernel_size in [7, 8] {
        let input: Vec<f32> = (0..in_len).map(|i| (i as f32) * 0.3).collect();
        let weight: Vec<f32> = (0..kernel_size).map(|i| (i as f32) * 0.1 + 0.05).collect();
        let bias = vec![0.5];

        let out = conv1d(&input, &weight, Some(&bias), 1, 1, kernel_size, 1, 0);
        let expected = naive_conv1d(&input, &weight, Some(&bias), 1, 1, kernel_size, 1, 0);
        assert_close(
            &out,
            &expected,
            1e-3,
            &format!("threshold_boundary_ks{kernel_size}"),
        );
    }
}

// ---------------------------------------------------------------------------
// Edge cases: zeros, negative weights, large values
// ---------------------------------------------------------------------------

#[test]
fn test_conv1d_all_zeros_input() {
    let input = vec![0.0f32; 8];
    let weight = vec![1.0, 2.0, 3.0];
    let out = conv1d(&input, &weight, None, 1, 1, 3, 1, 0);
    assert!(
        out.iter().all(|&v| v == 0.0),
        "zero input should produce zero output"
    );
}

#[test]
fn test_conv1d_all_zeros_weight() {
    let input: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let weight = vec![0.0f32; 3];
    let out = conv1d(&input, &weight, None, 1, 1, 3, 1, 0);
    assert!(
        out.iter().all(|&v| v == 0.0),
        "zero weight should produce zero output"
    );
}

#[test]
fn test_conv1d_negative_weights() {
    let input: Vec<f32> = (1..=6).map(|x| x as f32).collect();
    let weight = vec![-1.0, -1.0, -1.0];
    let out = conv1d(&input, &weight, None, 1, 1, 3, 1, 0);
    let expected = naive_conv1d(&input, &weight, None, 1, 1, 3, 1, 0);
    assert_close(&out, &expected, 1e-6, "negative_weights");
    // All outputs should be negative (input is positive, weight is negative)
    assert!(
        out.iter().all(|&v| v < 0.0),
        "all outputs should be negative"
    );
}

#[test]
fn test_conv1d_large_values() {
    let input = vec![1e6, 2e6, 3e6, 4e6, 5e6];
    let weight = vec![1.0, 1.0, 1.0];
    let out = conv1d(&input, &weight, None, 1, 1, 3, 1, 0);
    let expected = naive_conv1d(&input, &weight, None, 1, 1, 3, 1, 0);
    // Wider tolerance for large values due to floating-point precision
    assert_close(&out, &expected, 1.0, "large_values");
}

// ---------------------------------------------------------------------------
// Identity filter
// ---------------------------------------------------------------------------

#[test]
fn test_conv1d_identity_filter() {
    // A kernel [0, 1, 0] with same-padding should approximate identity
    let input: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let weight = vec![0.0, 1.0, 0.0];
    let out = conv1d(&input, &weight, None, 1, 1, 3, 1, 1);
    // With padding=1 and kernel [0,1,0], output should equal input
    assert_close(&out, &input, 1e-6, "identity_filter");
}
