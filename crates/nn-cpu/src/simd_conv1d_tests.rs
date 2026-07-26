// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SIMD conv1d with groups and dilation support.
//!
//! Strategy: compare `conv1d_full` (SIMD-dispatched) against
//! `conv1d_full_reference` (pure scalar) and a fully independent naive
//! implementation defined here.

use crate::simd_conv1d::{
    conv1d_full, conv1d_full_reference, conv1d_grouped, Conv1dConfig, Conv1dError,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Fully independent naive conv1d with groups + dilation, for oracle testing.
fn naive_grouped_conv1d(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    cfg: &Conv1dConfig,
) -> Vec<f32> {
    let in_len = input.len() / cfg.in_channels;
    let out_len = cfg.output_length(in_len);
    let ic_per_g = cfg.in_channels / cfg.groups;
    let oc_per_g = cfg.out_channels / cfg.groups;
    let mut output = vec![0.0f32; cfg.out_channels * out_len];

    for g in 0..cfg.groups {
        for oc_local in 0..oc_per_g {
            let oc = g * oc_per_g + oc_local;
            for o in 0..out_len {
                let mut acc = 0.0f32;
                for ic_local in 0..ic_per_g {
                    let ic = g * ic_per_g + ic_local;
                    for k in 0..cfg.kernel_size {
                        let in_pos = o * cfg.stride + k * cfg.dilation;
                        if in_pos >= cfg.padding && in_pos < cfg.padding + in_len {
                            let w_idx =
                                oc * ic_per_g * cfg.kernel_size + ic_local * cfg.kernel_size + k;
                            let i_idx = ic * in_len + (in_pos - cfg.padding);
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
    }
    output
}

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

fn cfg(
    in_ch: usize,
    out_ch: usize,
    ks: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Conv1dConfig {
    Conv1dConfig {
        in_channels: in_ch,
        out_channels: out_ch,
        kernel_size: ks,
        stride,
        padding,
        dilation,
        groups,
    }
}

// ---------------------------------------------------------------------------
// Basic: no groups, no dilation (parity with existing conv1d)
// ---------------------------------------------------------------------------

#[test]
fn test_basic_single_channel() {
    let c = cfg(1, 1, 3, 1, 0, 1, 1);
    let input: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let weight = vec![1.0, 0.0, -1.0];
    let out = conv1d_full(&input, &weight, None, &c).unwrap();
    let expected = naive_grouped_conv1d(&input, &weight, None, &c);
    assert_close(&out, &expected, 1e-6, "basic_single_ch");
}

#[test]
fn test_basic_multi_channel() {
    let c = cfg(3, 2, 3, 1, 0, 1, 1);
    let in_len = 8;
    let input: Vec<f32> = (0..c.in_channels * in_len)
        .map(|i| (i as f32) * 0.1 - 1.0)
        .collect();
    let weight: Vec<f32> = (0..c.out_channels * c.in_channels * c.kernel_size)
        .map(|i| (i as f32) * 0.05 - 0.4)
        .collect();
    let bias = vec![0.1, -0.2];

    let out = conv1d_full(&input, &weight, Some(&bias), &c).unwrap();
    let expected = naive_grouped_conv1d(&input, &weight, Some(&bias), &c);
    assert_close(&out, &expected, 1e-4, "basic_multi_ch");
}

#[test]
fn test_stride_2() {
    let c = cfg(1, 1, 3, 2, 0, 1, 1);
    let input: Vec<f32> = (1..=10).map(|x| x as f32).collect();
    let weight = vec![1.0, 1.0, 1.0];
    let out = conv1d_full(&input, &weight, None, &c).unwrap();
    assert_eq!(out.len(), 4); // (10-3)/2+1 = 4
    let expected = naive_grouped_conv1d(&input, &weight, None, &c);
    assert_close(&out, &expected, 1e-6, "stride_2");
}

#[test]
fn test_padding_same() {
    let c = cfg(1, 1, 3, 1, 1, 1, 1);
    let input: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let weight = vec![1.0, 1.0, 1.0];
    let out = conv1d_full(&input, &weight, None, &c).unwrap();
    assert_eq!(out.len(), 8); // same padding preserves length
    let expected = naive_grouped_conv1d(&input, &weight, None, &c);
    assert_close(&out, &expected, 1e-6, "padding_same");
}

// ---------------------------------------------------------------------------
// Dilation
// ---------------------------------------------------------------------------

#[test]
fn test_dilation_2() {
    // dilation=2, kernel_size=3: effective kernel = 2*(3-1)+1 = 5
    let c = cfg(1, 1, 3, 1, 0, 2, 1);
    let input: Vec<f32> = (0..10).map(|x| x as f32).collect();
    let weight = vec![1.0, 1.0, 1.0];
    // out_len = (10 - 5)/1 + 1 = 6
    let out = conv1d_full(&input, &weight, None, &c).unwrap();
    assert_eq!(out.len(), 6);

    // Manual: out[i] = input[i] + input[i+2] + input[i+4]
    let expected: Vec<f32> = (0..6)
        .map(|i| (i as f32) + ((i + 2) as f32) + ((i + 4) as f32))
        .collect();
    assert_close(&out, &expected, 1e-6, "dilation_2_manual");
}

#[test]
fn test_dilation_3_multi_channel() {
    let c = cfg(2, 3, 2, 1, 0, 3, 1);
    let in_len = 12;
    let input: Vec<f32> = (0..c.in_channels * in_len)
        .map(|i| (i as f32) * 0.1)
        .collect();
    let weight: Vec<f32> = (0..c.out_channels * c.in_channels * c.kernel_size)
        .map(|i| (i as f32) * 0.2 - 0.5)
        .collect();

    let out = conv1d_full(&input, &weight, None, &c).unwrap();
    let ref_out = conv1d_full_reference(&input, &weight, None, &c).unwrap();
    let naive = naive_grouped_conv1d(&input, &weight, None, &c);
    assert_close(&out, &ref_out, 1e-5, "dilation_3_simd_vs_ref");
    assert_close(&out, &naive, 1e-5, "dilation_3_simd_vs_naive");
}

#[test]
fn test_dilation_with_padding() {
    let c = cfg(1, 1, 3, 1, 2, 2, 1);
    let input: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let weight = vec![0.5, 0.3, 0.2];
    let out = conv1d_full(&input, &weight, None, &c).unwrap();
    let expected = naive_grouped_conv1d(&input, &weight, None, &c);
    assert_close(&out, &expected, 1e-5, "dilation_padding");
}

#[test]
fn test_dilation_with_stride() {
    let c = cfg(1, 1, 3, 2, 1, 2, 1);
    let input: Vec<f32> = (0..16).map(|x| x as f32 * 0.5).collect();
    let weight = vec![1.0, -0.5, 0.25];
    let out = conv1d_full(&input, &weight, None, &c).unwrap();
    let expected = naive_grouped_conv1d(&input, &weight, None, &c);
    assert_close(&out, &expected, 1e-5, "dilation_stride");
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

#[test]
fn test_groups_depthwise() {
    // Depthwise: groups = in_ch = out_ch
    let channels = 4;
    let c = cfg(channels, channels, 3, 1, 0, 1, channels);
    let in_len = 8;
    let input: Vec<f32> = (0..channels * in_len).map(|i| (i as f32) * 0.2).collect();
    // Weight: [4, 1, 3] — each channel has its own 1-channel filter
    let weight: Vec<f32> = (0..channels * c.kernel_size)
        .map(|i| (i as f32) * 0.1 + 0.1)
        .collect();

    let out = conv1d_full(&input, &weight, None, &c).unwrap();
    let expected = naive_grouped_conv1d(&input, &weight, None, &c);
    assert_close(&out, &expected, 1e-5, "depthwise");
}

#[test]
fn test_groups_depthwise_with_bias() {
    let channels = 3;
    let c = cfg(channels, channels, 3, 1, 1, 1, channels);
    let in_len = 10;
    let input: Vec<f32> = (0..channels * in_len).map(|i| (i as f32).sin()).collect();
    let weight: Vec<f32> = (0..channels * c.kernel_size)
        .map(|i| (i as f32).cos() * 0.3)
        .collect();
    let bias: Vec<f32> = vec![0.1, -0.2, 0.3];

    let out = conv1d_full(&input, &weight, Some(&bias), &c).unwrap();
    let expected = naive_grouped_conv1d(&input, &weight, Some(&bias), &c);
    assert_close(&out, &expected, 1e-5, "depthwise_bias");
}

#[test]
fn test_groups_2() {
    // 2 groups: in_ch=4, out_ch=6 => 2 per group in, 3 per group out
    let c = cfg(4, 6, 3, 1, 0, 1, 2);
    let in_len = 8;
    let ic_per_g = c.in_channels_per_group(); // 2
    let input: Vec<f32> = (0..c.in_channels * in_len)
        .map(|i| (i as f32) * 0.1 - 1.0)
        .collect();
    let weight: Vec<f32> = (0..c.out_channels * ic_per_g * c.kernel_size)
        .map(|i| (i as f32) * 0.05 - 0.3)
        .collect();

    let out = conv1d_full(&input, &weight, None, &c).unwrap();
    let ref_out = conv1d_full_reference(&input, &weight, None, &c).unwrap();
    let naive = naive_grouped_conv1d(&input, &weight, None, &c);
    assert_close(&out, &ref_out, 1e-4, "groups_2_simd_vs_ref");
    assert_close(&out, &naive, 1e-4, "groups_2_simd_vs_naive");
}

#[test]
fn test_groups_with_dilation() {
    // Groups + dilation combined
    let c = cfg(4, 4, 3, 1, 0, 2, 4);
    let in_len = 10;
    let input: Vec<f32> = (0..c.in_channels * in_len)
        .map(|i| (i as f32) * 0.15)
        .collect();
    let weight: Vec<f32> = (0..c.out_channels * c.kernel_size)
        .map(|i| (i as f32) * 0.2 - 0.5)
        .collect();

    let out = conv1d_full(&input, &weight, None, &c).unwrap();
    let expected = naive_grouped_conv1d(&input, &weight, None, &c);
    assert_close(&out, &expected, 1e-5, "groups_dilation");
}

// ---------------------------------------------------------------------------
// SIMD vs scalar reference parity
// ---------------------------------------------------------------------------

#[test]
fn test_simd_vs_reference_medium() {
    let c = cfg(4, 8, 5, 1, 2, 1, 1);
    let in_len = 64;
    let input: Vec<f32> = (0..c.in_channels * in_len)
        .map(|i| ((i * 7 + 3) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let weight: Vec<f32> = (0..c.out_channels * c.in_channels * c.kernel_size)
        .map(|i| ((i * 13 + 11) % 200) as f32 * 0.005 - 0.5)
        .collect();
    let bias: Vec<f32> = (0..c.out_channels)
        .map(|i| (i as f32) * 0.1 - 0.3)
        .collect();

    let simd_out = conv1d_full(&input, &weight, Some(&bias), &c).unwrap();
    let ref_out = conv1d_full_reference(&input, &weight, Some(&bias), &c).unwrap();
    assert_close(&simd_out, &ref_out, 1e-4, "simd_vs_ref_medium");
}

#[test]
fn test_simd_vs_reference_non_aligned() {
    // out_len=13: not divisible by NEON(4) or AVX2(8)
    let c = cfg(1, 1, 3, 1, 0, 1, 1);
    let input: Vec<f32> = (0..15).map(|i| (i as f32) * 0.7 - 4.0).collect();
    let weight = vec![0.3, -0.6, 0.9];

    let simd_out = conv1d_full(&input, &weight, None, &c).unwrap();
    let ref_out = conv1d_full_reference(&input, &weight, None, &c).unwrap();
    assert_eq!(simd_out.len(), 13);
    assert_close(&simd_out, &ref_out, 1e-5, "simd_vs_ref_non_aligned");
}

#[test]
fn test_simd_vs_reference_groups_dilation_stride_padding() {
    // Full feature combo: groups=2, dilation=2, stride=2, padding=3
    let c = cfg(4, 6, 3, 2, 3, 2, 2);
    let in_len = 20;
    let input: Vec<f32> = (0..c.in_channels * in_len)
        .map(|i| (i as f32).sin())
        .collect();
    let ic_per_g = c.in_channels_per_group();
    let weight: Vec<f32> = (0..c.out_channels * ic_per_g * c.kernel_size)
        .map(|i| (i as f32).cos() * 0.3)
        .collect();
    let bias: Vec<f32> = (0..c.out_channels).map(|i| (i as f32) * 0.05).collect();

    let simd_out = conv1d_full(&input, &weight, Some(&bias), &c).unwrap();
    let ref_out = conv1d_full_reference(&input, &weight, Some(&bias), &c).unwrap();
    let naive = naive_grouped_conv1d(&input, &weight, Some(&bias), &c);
    assert_close(&simd_out, &ref_out, 1e-4, "full_combo_simd_vs_ref");
    assert_close(&simd_out, &naive, 1e-4, "full_combo_simd_vs_naive");
}

// ---------------------------------------------------------------------------
// conv1d_grouped convenience API
// ---------------------------------------------------------------------------

#[test]
fn test_conv1d_grouped_api() {
    let in_ch = 4;
    let out_ch = 4;
    let ks = 3;
    let in_len = 10;
    let input: Vec<f32> = (0..in_ch * in_len).map(|i| (i as f32) * 0.2).collect();
    let weight: Vec<f32> = (0..out_ch * ks).map(|i| (i as f32) * 0.1).collect();

    let out = conv1d_grouped(&input, &weight, None, in_ch, out_ch, ks, 1, 0, 1, 4).unwrap();

    let c = cfg(in_ch, out_ch, ks, 1, 0, 1, 4);
    let expected = conv1d_full(&input, &weight, None, &c).unwrap();
    assert_close(&out, &expected, 0.0, "grouped_api");
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_all_zeros_input() {
    let c = cfg(1, 1, 3, 1, 0, 1, 1);
    let input = vec![0.0f32; 8];
    let weight = vec![1.0, 2.0, 3.0];
    let out = conv1d_full(&input, &weight, None, &c).unwrap();
    assert!(out.iter().all(|&v| v == 0.0), "zero input -> zero output");
}

#[test]
fn test_all_zeros_weight() {
    let c = cfg(1, 1, 3, 1, 0, 1, 1);
    let input: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let weight = vec![0.0f32; 3];
    let out = conv1d_full(&input, &weight, None, &c).unwrap();
    assert!(out.iter().all(|&v| v == 0.0), "zero weight -> zero output");
}

#[test]
fn test_identity_filter() {
    let c = cfg(1, 1, 3, 1, 1, 1, 1);
    let input: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let weight = vec![0.0, 1.0, 0.0]; // identity with same-padding
    let out = conv1d_full(&input, &weight, None, &c).unwrap();
    assert_close(&out, &input, 1e-6, "identity_filter");
}

#[test]
fn test_input_equals_kernel() {
    let c = cfg(1, 1, 5, 1, 0, 1, 1);
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let out = conv1d_full(&input, &weight, None, &c).unwrap();
    assert_eq!(out.len(), 1);
    assert_close(&out, &[15.0], 1e-6, "input_eq_kernel");
}

#[test]
fn test_pointwise() {
    let c = cfg(2, 3, 1, 1, 0, 1, 1);
    let in_len = 5;
    let input: Vec<f32> = (0..c.in_channels * in_len)
        .map(|i| (i + 1) as f32)
        .collect();
    let weight: Vec<f32> = (0..c.out_channels * c.in_channels)
        .map(|i| (i as f32) * 0.5)
        .collect();
    let out = conv1d_full(&input, &weight, None, &c).unwrap();
    let expected = naive_grouped_conv1d(&input, &weight, None, &c);
    assert_eq!(out.len(), c.out_channels * in_len);
    assert_close(&out, &expected, 1e-5, "pointwise");
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

#[test]
fn test_error_zero_stride() {
    let c = cfg(1, 1, 3, 0, 0, 1, 1);
    let r = conv1d_full(&[1.0; 8], &[1.0; 3], None, &c);
    assert!(matches!(r, Err(Conv1dError::ZeroStride)));
}

#[test]
fn test_error_zero_dilation() {
    let c = cfg(1, 1, 3, 1, 0, 0, 1);
    let r = conv1d_full(&[1.0; 8], &[1.0; 3], None, &c);
    assert!(matches!(r, Err(Conv1dError::ZeroDilation)));
}

#[test]
fn test_error_zero_groups() {
    let c = cfg(1, 1, 3, 1, 0, 1, 0);
    let r = conv1d_full(&[1.0; 8], &[1.0; 3], None, &c);
    assert!(matches!(r, Err(Conv1dError::ZeroGroups)));
}

#[test]
fn test_error_in_ch_not_divisible_by_groups() {
    let c = cfg(3, 6, 3, 1, 0, 1, 2);
    let r = conv1d_full(&[1.0; 24], &[1.0; 27], None, &c);
    assert!(matches!(
        r,
        Err(Conv1dError::InChannelsNotDivisibleByGroups { .. })
    ));
}

#[test]
fn test_error_out_ch_not_divisible_by_groups() {
    let c = cfg(4, 5, 3, 1, 0, 1, 2);
    let r = conv1d_full(&[1.0; 32], &[1.0; 30], None, &c);
    assert!(matches!(
        r,
        Err(Conv1dError::OutChannelsNotDivisibleByGroups { .. })
    ));
}

#[test]
fn test_error_wrong_weight_len() {
    let c = cfg(2, 3, 3, 1, 0, 1, 1);
    // Expected: 3 * 2 * 3 = 18, provide 10
    let r = conv1d_full(&[1.0; 16], &[1.0; 10], None, &c);
    assert!(matches!(r, Err(Conv1dError::InvalidWeightShape { .. })));
}

#[test]
fn test_error_wrong_bias_len() {
    let c = cfg(1, 2, 3, 1, 0, 1, 1);
    let r = conv1d_full(&[1.0; 8], &[1.0; 6], Some(&[1.0; 3]), &c);
    assert!(matches!(r, Err(Conv1dError::InvalidBiasLength { .. })));
}

#[test]
fn test_error_padded_too_small() {
    // dilation=5, ks=3 => effective=11, in_len=4, pad=0 => padded=4 < 11
    let c = cfg(1, 1, 3, 1, 0, 5, 1);
    let r = conv1d_full(&[1.0; 4], &[1.0; 3], None, &c);
    assert!(matches!(r, Err(Conv1dError::PaddedInputTooSmall { .. })));
}

// ---------------------------------------------------------------------------
// Output length formula
// ---------------------------------------------------------------------------

#[test]
fn test_output_length_with_dilation() {
    // out_len = (in_len + 2*pad - dilation*(ks-1) - 1) / stride + 1
    let cases: Vec<(usize, usize, usize, usize, usize, usize)> = vec![
        // (in_len, ks, stride, pad, dilation, expected_out_len)
        (10, 3, 1, 0, 1, 8),  // standard
        (10, 3, 1, 0, 2, 6),  // dilation=2, eff_ks=5
        (10, 3, 1, 0, 3, 4),  // dilation=3, eff_ks=7
        (20, 3, 2, 1, 2, 9),  // stride+pad+dilation
        (16, 5, 1, 4, 2, 16), // same-like padding with dilation
        (8, 2, 1, 0, 4, 4),   // dilation=4, eff_ks=5
    ];

    for (in_len, ks, stride, pad, dil, expected) in cases {
        let c = cfg(1, 1, ks, stride, pad, dil, 1);
        let computed = c.output_length(in_len);
        assert_eq!(
            computed, expected,
            "out_len(in={in_len}, ks={ks}, s={stride}, p={pad}, d={dil}): \
             got {computed}, expected {expected}"
        );
    }
}

// ---------------------------------------------------------------------------
// Kokoro-representative sizes
// ---------------------------------------------------------------------------

#[test]
fn test_kokoro_encoder_like() {
    // Kokoro encoder: in_ch=1, out_ch=48, kernel_size=8, stride=4
    let c = cfg(1, 48, 8, 4, 2, 1, 1);
    let in_len = 256;
    let input: Vec<f32> = (0..c.in_channels * in_len)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let ic_per_g = c.in_channels_per_group();
    let weight: Vec<f32> = (0..c.out_channels * ic_per_g * c.kernel_size)
        .map(|i| (i as f32 * 0.01).cos() * 0.1)
        .collect();
    let bias: Vec<f32> = (0..c.out_channels).map(|i| (i as f32) * 0.001).collect();

    let simd_out = conv1d_full(&input, &weight, Some(&bias), &c).unwrap();
    let ref_out = conv1d_full_reference(&input, &weight, Some(&bias), &c).unwrap();
    assert_close(&simd_out, &ref_out, 1e-3, "kokoro_encoder_like");
}

#[test]
fn test_kokoro_depthwise_like() {
    // Kokoro uses depthwise separable convolutions
    let channels = 48;
    let c = cfg(channels, channels, 3, 1, 1, 1, channels);
    let in_len = 64;
    let input: Vec<f32> = (0..channels * in_len)
        .map(|i| (i as f32 * 0.005).sin())
        .collect();
    let weight: Vec<f32> = (0..channels * c.kernel_size)
        .map(|i| (i as f32 * 0.02).cos() * 0.15)
        .collect();

    let simd_out = conv1d_full(&input, &weight, None, &c).unwrap();
    let ref_out = conv1d_full_reference(&input, &weight, None, &c).unwrap();
    assert_eq!(simd_out.len(), channels * in_len); // same-padding
    assert_close(&simd_out, &ref_out, 1e-4, "kokoro_depthwise_like");
}

// ---------------------------------------------------------------------------
// Determinism / batch independence
// ---------------------------------------------------------------------------

#[test]
fn test_deterministic() {
    let c = cfg(2, 3, 3, 1, 1, 1, 1);
    let input: Vec<f32> = (0..20).map(|i| (i as f32) * 0.3).collect();
    let weight: Vec<f32> = (0..18).map(|i| (i as f32) * 0.1).collect();

    let out1 = conv1d_full(&input, &weight, None, &c).unwrap();
    let out2 = conv1d_full(&input, &weight, None, &c).unwrap();
    assert_close(&out1, &out2, 0.0, "deterministic");
}
