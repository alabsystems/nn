// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for SIMD-accelerated 1D convolution.
//!
//! Validates that the dispatched (NEON/AVX2/im2col) path produces identical
//! results to the scalar reference across various configurations: kernel sizes,
//! strides, padding, multi-channel, bias/no-bias.

use nn_cpu::conv1d;

// ============================================================================
// Helpers
// ============================================================================

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

/// Naive triple-loop conv1d for ground truth.
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
                        let w = weight[oc * in_ch * kernel_size + ic * kernel_size + k];
                        let x = input[ic * in_len + in_pos - padding];
                        acc += w * x;
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

// ============================================================================
// 1x1 conv (equivalent to linear / pointwise)
// ============================================================================

#[test]
fn test_conv1d_1x1_single_channel() {
    // 1x1 conv with 1 input channel, 1 output channel = scalar multiply.
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weight = vec![2.0]; // [out_ch=1, in_ch=1, kernel_size=1]
    let result = conv1d::conv1d(&input, &weight, None, 1, 1, 1, 1, 0);
    let expected: Vec<f32> = input.iter().map(|x| x * 2.0).collect();
    assert_close(&result, &expected, 1e-6, "conv1d_1x1_single");
}

#[test]
fn test_conv1d_1x1_multi_channel() {
    // 1x1 conv: [in_ch=2, len=4] * [out_ch=3, in_ch=2, kernel_size=1] = [3, 4]
    // This is equivalent to a linear layer applied to each position.
    let in_ch = 2;
    let out_ch = 3;
    let in_len = 4;
    // input: ch0=[1,2,3,4], ch1=[5,6,7,8]
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // weight: [3, 2, 1] = 6 values
    let weight: Vec<f32> = vec![0.5, -0.5, 1.0, 0.0, 0.0, 1.0];

    let result = conv1d::conv1d(&input, &weight, None, in_ch, out_ch, 1, 1, 0);
    let expected = naive_conv1d(&input, &weight, None, in_ch, out_ch, 1, 1, 0);
    assert_close(&result, &expected, 1e-5, "conv1d_1x1_multi_ch");
    assert_eq!(result.len(), out_ch * in_len);
}

// ============================================================================
// 3x1 conv with padding
// ============================================================================

#[test]
fn test_conv1d_k3_padding1() {
    // kernel_size=3, padding=1 => output length = input length (stride=1).
    let in_ch = 1;
    let out_ch = 1;
    let kernel_size = 3;
    let padding = 1;
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    // weight = [1, 1, 1] => sum of 3 neighbors (with zero-padding at edges)
    let weight = vec![1.0, 1.0, 1.0];

    let result = conv1d::conv1d(
        &input,
        &weight,
        None,
        in_ch,
        out_ch,
        kernel_size,
        1,
        padding,
    );
    let expected = naive_conv1d(
        &input,
        &weight,
        None,
        in_ch,
        out_ch,
        kernel_size,
        1,
        padding,
    );
    assert_close(&result, &expected, 1e-6, "conv1d_k3_pad1");
    // Output length should equal input length.
    assert_eq!(result.len(), input.len());
    // First element: 0*1 + 1*1 + 2*1 = 3
    assert!((result[0] - 3.0).abs() < 1e-6);
    // Last element: 4*1 + 5*1 + 0*1 = 9
    assert!((result[4] - 9.0).abs() < 1e-6);
}

#[test]
fn test_conv1d_k3_no_padding() {
    // kernel_size=3, padding=0 => output length = input_len - 2.
    let in_ch = 1;
    let out_ch = 1;
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weight = vec![1.0, 0.0, -1.0]; // difference filter

    let result = conv1d::conv1d(&input, &weight, None, in_ch, out_ch, 3, 1, 0);
    let expected = naive_conv1d(&input, &weight, None, in_ch, out_ch, 3, 1, 0);
    assert_close(&result, &expected, 1e-6, "conv1d_k3_no_pad");
    assert_eq!(result.len(), 6); // 8 - 3 + 1 = 6
}

// ============================================================================
// Stride > 1
// ============================================================================

#[test]
fn test_conv1d_stride2() {
    let in_ch = 1;
    let out_ch = 1;
    let kernel_size = 3;
    let stride = 2;
    let padding = 1;
    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let weight = vec![0.25, 0.5, 0.25];

    let result = conv1d::conv1d(
        &input,
        &weight,
        None,
        in_ch,
        out_ch,
        kernel_size,
        stride,
        padding,
    );
    let expected = naive_conv1d(
        &input,
        &weight,
        None,
        in_ch,
        out_ch,
        kernel_size,
        stride,
        padding,
    );
    assert_close(&result, &expected, 1e-5, "conv1d_stride2");
    // out_len = (16 + 2*1 - 3) / 2 + 1 = 8
    assert_eq!(result.len(), 8);
}

#[test]
fn test_conv1d_stride4() {
    let in_ch = 2;
    let out_ch = 4;
    let kernel_size = 5;
    let stride = 4;
    let padding = 2;
    let in_len = 64;
    let input: Vec<f32> = (0..in_ch * in_len)
        .map(|i| ((i % 13) as f32) * 0.1 - 0.6)
        .collect();
    let weight: Vec<f32> = (0..out_ch * in_ch * kernel_size)
        .map(|i| ((i % 7) as f32) * 0.2 - 0.5)
        .collect();

    let result = conv1d::conv1d(
        &input,
        &weight,
        None,
        in_ch,
        out_ch,
        kernel_size,
        stride,
        padding,
    );
    let expected = naive_conv1d(
        &input,
        &weight,
        None,
        in_ch,
        out_ch,
        kernel_size,
        stride,
        padding,
    );
    assert_close(&result, &expected, 1e-4, "conv1d_stride4");
}

// ============================================================================
// Multi-channel input/output
// ============================================================================

#[test]
fn test_conv1d_multi_channel() {
    let in_ch = 4;
    let out_ch = 8;
    let kernel_size = 3;
    let stride = 1;
    let padding = 1;
    let in_len = 32;
    let input: Vec<f32> = (0..in_ch * in_len)
        .map(|i| ((i % 11) as f32) * 0.1 - 0.5)
        .collect();
    let weight: Vec<f32> = (0..out_ch * in_ch * kernel_size)
        .map(|i| ((i % 17) as f32) * 0.05 - 0.4)
        .collect();

    let result = conv1d::conv1d(
        &input,
        &weight,
        None,
        in_ch,
        out_ch,
        kernel_size,
        stride,
        padding,
    );
    let expected = naive_conv1d(
        &input,
        &weight,
        None,
        in_ch,
        out_ch,
        kernel_size,
        stride,
        padding,
    );
    assert_close(&result, &expected, 1e-3, "conv1d_multi_ch");
    assert_eq!(result.len(), out_ch * in_len);
}

#[test]
fn test_conv1d_large_channels() {
    // Exercise deeper accumulation for many input channels.
    let in_ch = 48;
    let out_ch = 96;
    let kernel_size = 3;
    let stride = 1;
    let padding = 1;
    let in_len = 16;
    let input: Vec<f32> = (0..in_ch * in_len)
        .map(|i| ((i % 23) as f32) * 0.02 - 0.2)
        .collect();
    let weight: Vec<f32> = (0..out_ch * in_ch * kernel_size)
        .map(|i| ((i % 31) as f32) * 0.01 - 0.15)
        .collect();

    let result = conv1d::conv1d(
        &input,
        &weight,
        None,
        in_ch,
        out_ch,
        kernel_size,
        stride,
        padding,
    );
    let expected = naive_conv1d(
        &input,
        &weight,
        None,
        in_ch,
        out_ch,
        kernel_size,
        stride,
        padding,
    );
    assert_close(&result, &expected, 1e-2, "conv1d_large_ch");
}

// ============================================================================
// Bias vs no-bias
// ============================================================================

#[test]
fn test_conv1d_with_bias() {
    let in_ch = 2;
    let out_ch = 3;
    let kernel_size = 3;
    let in_len = 10;
    let padding = 1;
    let input: Vec<f32> = (0..in_ch * in_len).map(|i| i as f32 * 0.1).collect();
    let weight: Vec<f32> = (0..out_ch * in_ch * kernel_size)
        .map(|i| i as f32 * 0.05 - 0.3)
        .collect();
    let bias = vec![0.1, -0.2, 0.3];

    let result = conv1d::conv1d(
        &input,
        &weight,
        Some(&bias),
        in_ch,
        out_ch,
        kernel_size,
        1,
        padding,
    );
    let expected = naive_conv1d(
        &input,
        &weight,
        Some(&bias),
        in_ch,
        out_ch,
        kernel_size,
        1,
        padding,
    );
    assert_close(&result, &expected, 1e-4, "conv1d_with_bias");
}

#[test]
fn test_conv1d_no_bias() {
    let in_ch = 2;
    let out_ch = 3;
    let kernel_size = 3;
    let in_len = 10;
    let input: Vec<f32> = (0..in_ch * in_len).map(|i| i as f32 * 0.1).collect();
    let weight: Vec<f32> = (0..out_ch * in_ch * kernel_size)
        .map(|i| i as f32 * 0.05 - 0.3)
        .collect();

    let result = conv1d::conv1d(&input, &weight, None, in_ch, out_ch, kernel_size, 1, 1);
    let result_ref =
        conv1d::conv1d_scalar_reference(&input, &weight, None, in_ch, out_ch, kernel_size, 1, 1);
    assert_close(&result, &result_ref, 1e-4, "conv1d_no_bias");
}

// ============================================================================
// Reference comparison: SIMD-dispatched vs scalar reference
// ============================================================================

#[test]
fn test_conv1d_simd_matches_scalar_reference() {
    // Comprehensive check: dispatched path vs scalar_reference.
    let configs: Vec<(usize, usize, usize, usize, usize)> = vec![
        // (in_ch, out_ch, kernel_size, stride, padding)
        (1, 1, 1, 1, 0),
        (1, 1, 3, 1, 1),
        (2, 4, 3, 1, 1),
        (4, 8, 5, 2, 2),
        (3, 6, 7, 1, 3),
        (1, 1, 3, 2, 0),
        (8, 16, 3, 1, 1),
    ];

    for (in_ch, out_ch, kernel_size, stride, padding) in configs {
        let in_len = 33; // non-power-of-2 to exercise tails
        let input: Vec<f32> = (0..in_ch * in_len)
            .map(|i| ((i * 7 + 3) % 19) as f32 * 0.1 - 0.9)
            .collect();
        let weight: Vec<f32> = (0..out_ch * in_ch * kernel_size)
            .map(|i| ((i * 11 + 5) % 23) as f32 * 0.08 - 0.8)
            .collect();
        let bias: Vec<f32> = (0..out_ch).map(|i| i as f32 * 0.1 - 0.3).collect();

        let label = format!("ic={in_ch},oc={out_ch},ks={kernel_size},s={stride},p={padding}");

        let dispatched = conv1d::conv1d(
            &input,
            &weight,
            Some(&bias),
            in_ch,
            out_ch,
            kernel_size,
            stride,
            padding,
        );
        let scalar = conv1d::conv1d_scalar_reference(
            &input,
            &weight,
            Some(&bias),
            in_ch,
            out_ch,
            kernel_size,
            stride,
            padding,
        );
        assert_close(&dispatched, &scalar, 1e-4, &label);
    }
}

// ============================================================================
// im2col path (large kernels, kernel_size > 7)
// ============================================================================

#[test]
fn test_conv1d_im2col_large_kernel() {
    // kernel_size=9 triggers im2col path (threshold is 7).
    let in_ch = 2;
    let out_ch = 4;
    let kernel_size = 9;
    let stride = 1;
    let padding = 4;
    let in_len = 20;
    let input: Vec<f32> = (0..in_ch * in_len)
        .map(|i| ((i % 11) as f32) * 0.1 - 0.5)
        .collect();
    let weight: Vec<f32> = (0..out_ch * in_ch * kernel_size)
        .map(|i| ((i % 7) as f32) * 0.1 - 0.3)
        .collect();
    let bias = vec![0.1, -0.1, 0.2, -0.2];

    let result = conv1d::conv1d(
        &input,
        &weight,
        Some(&bias),
        in_ch,
        out_ch,
        kernel_size,
        stride,
        padding,
    );
    let expected = naive_conv1d(
        &input,
        &weight,
        Some(&bias),
        in_ch,
        out_ch,
        kernel_size,
        stride,
        padding,
    );
    assert_close(&result, &expected, 1e-3, "conv1d_im2col_large_kernel");
}

#[test]
fn test_conv1d_im2col_stride2() {
    let in_ch = 3;
    let out_ch = 6;
    let kernel_size = 11;
    let stride = 2;
    let padding = 5;
    let in_len = 40;
    let input: Vec<f32> = (0..in_ch * in_len)
        .map(|i| ((i % 17) as f32) * 0.05 - 0.4)
        .collect();
    let weight: Vec<f32> = (0..out_ch * in_ch * kernel_size)
        .map(|i| ((i % 13) as f32) * 0.03 - 0.15)
        .collect();

    let result = conv1d::conv1d(
        &input,
        &weight,
        None,
        in_ch,
        out_ch,
        kernel_size,
        stride,
        padding,
    );
    let expected = naive_conv1d(
        &input,
        &weight,
        None,
        in_ch,
        out_ch,
        kernel_size,
        stride,
        padding,
    );
    assert_close(&result, &expected, 1e-2, "conv1d_im2col_stride2");
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn test_conv1d_single_output() {
    // Input length equals kernel size, no padding => single output.
    let input = vec![1.0, 2.0, 3.0];
    let weight = vec![1.0, 1.0, 1.0];
    let result = conv1d::conv1d(&input, &weight, None, 1, 1, 3, 1, 0);
    assert_eq!(result.len(), 1);
    assert!((result[0] - 6.0).abs() < 1e-6);
}

#[test]
fn test_conv1d_output_len_formula() {
    // Verify the output length formula: (in_len + 2*padding - kernel_size) / stride + 1
    let in_len = 100;
    let input = vec![0.0f32; in_len];
    let weight = vec![0.0f32; 8]; // ks=8

    let result = conv1d::conv1d(&input, &weight, None, 1, 1, 8, 4, 2);
    // out_len = (100 + 2*2 - 8) / 4 + 1 = 96/4 + 1 = 25
    assert_eq!(result.len(), 25);
}

#[test]
fn test_conv1d_non_aligned_lengths() {
    // Input length 13, which is not a multiple of 4 (NEON) or 8 (AVX2).
    let in_ch = 1;
    let out_ch = 2;
    let kernel_size = 3;
    let in_len = 13;
    let input: Vec<f32> = (0..in_ch * in_len).map(|i| i as f32).collect();
    let weight: Vec<f32> = (0..out_ch * in_ch * kernel_size)
        .map(|i| (i as f32) * 0.1)
        .collect();

    let result = conv1d::conv1d(&input, &weight, None, in_ch, out_ch, kernel_size, 1, 1);
    let expected = naive_conv1d(&input, &weight, None, in_ch, out_ch, kernel_size, 1, 1);
    assert_close(&result, &expected, 1e-5, "conv1d_non_aligned");
    assert_eq!(result.len(), out_ch * in_len);
}
