// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for SPIR-V normalization and activation kernels.
//!
//! Covers:
//! - BatchNorm reference correctness (1D, 2D, batched, various num_features)
//! - GroupNorm reference correctness (groups=1, groups=channels, groups divides channels)
//! - InstanceNorm reference correctness
//! - RMSNorm reference correctness (various eps)
//! - Softmax reference correctness (1D, 2D, large values for numerical stability)
//! - GELU/SiLU/Snake activation reference correctness
//! - SPIR-V generation for each norm (valid magic number, entry point)
//! - Workgroup size validation
//! - Edge cases: single element, very large channels, eps=0

use crate::spirv_activations::{
    gelu_reference, generate_gelu_spirv, generate_silu_spirv, generate_snake_spirv, silu_reference,
    snake_reference, ACTIVATION_WORKGROUP_SIZE,
};
use crate::spirv_binary::{find_entry_point_name, find_workgroup_size};
use crate::spirv_emit::SPIRV_MAGIC;
use crate::spirv_norms::{
    batchnorm_reference, generate_batchnorm_spirv, generate_groupnorm_spirv,
    generate_instancenorm_spirv, groupnorm_reference, instancenorm_reference, NORM_WORKGROUP_SIZE,
};
use crate::spirv_rmsnorm::{
    generate_rmsnorm_separate_io_spirv, rmsnorm_reference, RmsNormConfig, RMSNORM_WORKGROUP_SIZE,
};
use crate::spirv_softmax::{
    generate_softmax_separate_io_spirv, reference_softmax, SOFTMAX_WORKGROUP_SIZE,
};

// ---- SPIR-V constants for structural assertions ----

const TEST_SPIRV_VERSION_1_0: u32 = 0x0001_0000;
const TEST_GENERATOR_MAGIC: u32 = 0x4E4E_0000;

// ---- Helpers ----

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(
        bytes.len() % 4,
        0,
        "SPIR-V byte length must be multiple of 4"
    );
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn assert_valid_spirv_header(words: &[u32], label: &str) {
    assert!(words.len() >= 5, "{label}: module too short");
    assert_eq!(
        words[0], SPIRV_MAGIC,
        "{label}: wrong magic (expected 0x07230203)"
    );
    assert_eq!(words[1], TEST_SPIRV_VERSION_1_0, "{label}: wrong version");
    assert_eq!(
        words[2], TEST_GENERATOR_MAGIC,
        "{label}: wrong generator magic"
    );
    assert!(words[3] > 0, "{label}: bound must be > 0");
    assert_eq!(words[4], 0, "{label}: schema must be 0");
}

fn assert_all_finite(values: &[f32], label: &str) {
    for (i, &v) in values.iter().enumerate() {
        assert!(v.is_finite(), "{label}: output[{i}] = {v} must be finite");
    }
}

// ====================================================================
// BatchNorm extended reference tests
// ====================================================================

#[test]
fn test_batchnorm_ref_1d_single_channel() {
    // [N=1, C=1, S=8]
    let input: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let mean = vec![4.5];
    let var = vec![5.25]; // variance of 1..8
    let weight = vec![1.0];
    let bias = vec![0.0];
    let eps = 1e-5;
    let output = batchnorm_reference(&input, &mean, &var, &weight, &bias, 1, 1, 8, eps);
    assert_eq!(output.len(), 8);
    let inv_std = 1.0 / (5.25 + eps).sqrt();
    for (i, &v) in output.iter().enumerate() {
        let expected = (input[i] - 4.5) * inv_std;
        assert!(
            (v - expected).abs() < 1e-4,
            "batchnorm 1d ch1: output[{i}] = {v}, expected {expected}"
        );
    }
}

#[test]
fn test_batchnorm_ref_2d_two_channels() {
    // [N=1, C=2, S=4] simulating 2D spatial (2x2)
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mean = vec![2.5, 6.5];
    let var = vec![1.25, 1.25];
    let weight = vec![1.0, 1.0];
    let bias = vec![0.0, 0.0];
    let eps = 1e-5;
    let output = batchnorm_reference(&input, &mean, &var, &weight, &bias, 1, 2, 4, eps);
    assert_eq!(output.len(), 8);
    assert_all_finite(&output, "batchnorm_2d_2ch");

    let inv_std = 1.0 / (1.25 + eps).sqrt();
    // channel 0
    for i in 0..4 {
        let expected = (input[i] - 2.5) * inv_std;
        assert!((output[i] - expected).abs() < 1e-4);
    }
    // channel 1
    for i in 4..8 {
        let expected = (input[i] - 6.5) * inv_std;
        assert!((output[i] - expected).abs() < 1e-4);
    }
}

#[test]
fn test_batchnorm_ref_batched_n4() {
    // [N=4, C=2, S=1]
    let input = vec![
        1.0, 10.0, // batch 0
        2.0, 20.0, // batch 1
        3.0, 30.0, // batch 2
        4.0, 40.0, // batch 3
    ];
    let mean = vec![0.0, 0.0];
    let var = vec![1.0, 1.0];
    let weight = vec![1.0, 1.0];
    let bias = vec![0.0, 0.0];
    let eps = 0.0;
    let output = batchnorm_reference(&input, &mean, &var, &weight, &bias, 4, 2, 1, eps);
    assert_eq!(output.len(), 8);
    // With zero mean, unit var, eps=0: output == input
    for (i, (&out, &inp)) in output.iter().zip(input.iter()).enumerate() {
        assert!(
            (out - inp).abs() < 1e-6,
            "output[{i}] = {out}, expected {inp}"
        );
    }
}

#[test]
fn test_batchnorm_ref_various_num_features() {
    for num_features in [1, 3, 16, 64, 128] {
        let spatial = 4;
        let batch = 2;
        let n = batch * num_features * spatial;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let mean = vec![0.0; num_features];
        let var = vec![1.0; num_features];
        let weight = vec![1.0; num_features];
        let bias = vec![0.0; num_features];
        let output = batchnorm_reference(
            &input,
            &mean,
            &var,
            &weight,
            &bias,
            batch,
            num_features,
            spatial,
            1e-5,
        );
        assert_eq!(output.len(), n);
        assert_all_finite(&output, &format!("batchnorm_nf{num_features}"));
    }
}

#[test]
fn test_batchnorm_ref_large_spatial() {
    // [N=1, C=1, S=1024]
    let spatial = 1024;
    let input: Vec<f32> = (0..spatial).map(|i| (i as f32) * 0.01 - 5.0).collect();
    let mean = vec![0.0];
    let var = vec![10.0];
    let weight = vec![2.0];
    let bias = vec![1.0];
    let output = batchnorm_reference(&input, &mean, &var, &weight, &bias, 1, 1, spatial, 1e-5);
    assert_eq!(output.len(), spatial);
    assert_all_finite(&output, "batchnorm_large_spatial");
}

// ====================================================================
// GroupNorm extended reference tests
// ====================================================================

#[test]
fn test_groupnorm_ref_groups_1_is_layernorm_like() {
    // groups=1 means one group containing all channels
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [N=1, C=6, S=1]
    let weight = vec![1.0; 6];
    let bias = vec![0.0; 6];
    let eps = 1e-5;
    let output = groupnorm_reference(&input, &weight, &bias, 1, 1, 6, 1, eps);

    // Single group: mean over all 6 values
    let mean: f32 = input.iter().sum::<f32>() / 6.0;
    let var_val: f32 = input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / 6.0;
    let inv_std = 1.0 / (var_val + eps).sqrt();
    for (i, &v) in output.iter().enumerate() {
        let expected = (input[i] - mean) * inv_std;
        assert!(
            (v - expected).abs() < 1e-4,
            "gn g=1: output[{i}] = {v}, expected {expected}"
        );
    }
}

#[test]
fn test_groupnorm_ref_groups_eq_channels_is_instancenorm() {
    // groups == channels means each channel is its own group
    for channels in [2, 4, 8] {
        let spatial = 4;
        let n = channels * spatial;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let weight = vec![1.0; channels];
        let bias = vec![0.0; channels];
        let eps = 1e-5;
        let gn = groupnorm_reference(&input, &weight, &bias, 1, channels, channels, spatial, eps);
        let inst = instancenorm_reference(&input, &weight, &bias, 1, channels, spatial, eps);
        for (i, (&g, &in_val)) in gn.iter().zip(inst.iter()).enumerate() {
            assert!(
                (g - in_val).abs() < 1e-4,
                "gn(g=C={channels}): output[{i}]: gn={g}, in={in_val}"
            );
        }
    }
}

#[test]
fn test_groupnorm_ref_groups_divides_channels() {
    // groups=4, channels=16 => 4 channels per group
    let channels = 16;
    let groups = 4;
    let spatial = 2;
    let n = channels * spatial;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    let weight = vec![1.0; channels];
    let bias = vec![0.0; channels];
    let eps = 1e-5;
    let output = groupnorm_reference(&input, &weight, &bias, 1, groups, channels, spatial, eps);
    assert_eq!(output.len(), n);
    assert_all_finite(&output, "gn_g4_c16");
}

#[test]
fn test_groupnorm_ref_2_groups_4_channels_known() {
    // [N=1, C=4, S=2], groups=2 => group0=[ch0,ch1], group1=[ch2,ch3]
    let input = vec![
        1.0, 2.0, // ch0
        3.0, 4.0, // ch1
        5.0, 6.0, // ch2
        7.0, 8.0, // ch3
    ];
    let weight = vec![1.0; 4];
    let bias = vec![0.0; 4];
    let eps = 1e-5;
    let output = groupnorm_reference(&input, &weight, &bias, 1, 2, 4, 2, eps);

    // Group 0: channels 0,1 = [1,2,3,4]. mean=2.5
    let g0_vals = [1.0, 2.0, 3.0, 4.0];
    let g0_mean = 2.5_f32;
    let g0_var: f32 = g0_vals.iter().map(|x| (x - g0_mean).powi(2)).sum::<f32>() / 4.0;
    let g0_inv = 1.0 / (g0_var + eps).sqrt();
    for i in 0..4 {
        let expected = (g0_vals[i] - g0_mean) * g0_inv;
        assert!(
            (output[i] - expected).abs() < 1e-4,
            "gn g0: output[{i}] = {}, expected {expected}",
            output[i]
        );
    }

    // Group 1: channels 2,3 = [5,6,7,8]. mean=6.5
    let g1_vals = [5.0, 6.0, 7.0, 8.0];
    let g1_mean = 6.5_f32;
    let g1_var: f32 = g1_vals.iter().map(|x| (x - g1_mean).powi(2)).sum::<f32>() / 4.0;
    let g1_inv = 1.0 / (g1_var + eps).sqrt();
    for i in 0..4 {
        let expected = (g1_vals[i] - g1_mean) * g1_inv;
        assert!(
            (output[4 + i] - expected).abs() < 1e-4,
            "gn g1: output[{}] = {}, expected {expected}",
            4 + i,
            output[4 + i]
        );
    }
}

#[test]
fn test_groupnorm_ref_multi_batch_independence() {
    let channels = 4;
    let groups = 2;
    let spatial = 2;
    let weight = vec![1.0; channels];
    let bias = vec![0.0; channels];
    let eps = 1e-5;

    let single_input: Vec<f32> = (0..channels * spatial).map(|i| i as f32).collect();
    let single_output = groupnorm_reference(
        &single_input,
        &weight,
        &bias,
        1,
        groups,
        channels,
        spatial,
        eps,
    );

    // Batch 2: first batch is same as single
    let mut multi_input = single_input;
    multi_input.extend((0..channels * spatial).map(|i| (i as f32) * 10.0));
    let multi_output = groupnorm_reference(
        &multi_input,
        &weight,
        &bias,
        2,
        groups,
        channels,
        spatial,
        eps,
    );

    for i in 0..single_output.len() {
        assert!(
            (multi_output[i] - single_output[i]).abs() < 1e-5,
            "batch independence: multi[{i}]={}, single[{i}]={}",
            multi_output[i],
            single_output[i]
        );
    }
}

// ====================================================================
// InstanceNorm extended reference tests
// ====================================================================

#[test]
fn test_instancenorm_ref_single_element_spatial() {
    // [N=1, C=2, S=1] -- single spatial element, variance = 0
    let input = vec![5.0, 10.0];
    let weight = vec![1.0, 1.0];
    let bias = vec![0.0, 0.0];
    let eps = 1e-5;
    let output = instancenorm_reference(&input, &weight, &bias, 1, 2, 1, eps);
    // With single element: mean = x, var = 0, normalized = 0
    for (i, &v) in output.iter().enumerate() {
        assert!(
            v.abs() < 1e-3,
            "instancenorm single-spatial: output[{i}] = {v}, expected ~0"
        );
    }
}

#[test]
fn test_instancenorm_ref_large_channels() {
    let channels = 512;
    let spatial = 4;
    let n = channels * spatial;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
    let weight = vec![1.0; channels];
    let bias = vec![0.0; channels];
    let eps = 1e-5;
    let output = instancenorm_reference(&input, &weight, &bias, 1, channels, spatial, eps);
    assert_eq!(output.len(), n);
    assert_all_finite(&output, "instancenorm_512ch");
}

#[test]
fn test_instancenorm_ref_negative_inputs() {
    let input = vec![-10.0, -5.0, -1.0, 0.0, 1.0, 5.0]; // [N=1, C=2, S=3]
    let weight = vec![1.0, 1.0];
    let bias = vec![0.0, 0.0];
    let eps = 1e-5;
    let output = instancenorm_reference(&input, &weight, &bias, 1, 2, 3, eps);
    assert_all_finite(&output, "instancenorm_neg");
}

// ====================================================================
// RMSNorm extended reference tests
// ====================================================================

#[test]
fn test_rmsnorm_ref_eps_1e_6() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0; 4];
    let eps = 1e-6;
    let output = rmsnorm_reference(&input, &weight, 1, 4, eps);
    assert_all_finite(&output, "rmsnorm_eps_1e-6");
    let mean_sq = (1.0 + 4.0 + 9.0 + 16.0) / 4.0;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    for (i, &v) in output.iter().enumerate() {
        let expected = input[i] * inv_rms;
        assert!(
            (v - expected).abs() < 1e-5,
            "rmsnorm eps=1e-6: output[{i}]={v}, exp={expected}"
        );
    }
}

#[test]
fn test_rmsnorm_ref_eps_1e_8() {
    let input = vec![0.5, -0.5, 1.5, -1.5];
    let weight = vec![2.0; 4];
    let eps = 1e-8;
    let output = rmsnorm_reference(&input, &weight, 1, 4, eps);
    assert_all_finite(&output, "rmsnorm_eps_1e-8");
    let mean_sq = (0.25 + 0.25 + 2.25 + 2.25) / 4.0;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    for (i, &v) in output.iter().enumerate() {
        let expected = input[i] * inv_rms * 2.0;
        assert!(
            (v - expected).abs() < 1e-5,
            "rmsnorm eps=1e-8: output[{i}]={v}, exp={expected}"
        );
    }
}

#[test]
fn test_rmsnorm_ref_eps_0_nonzero_input() {
    // eps=0 with nonzero input should still produce valid results
    let input = vec![3.0, 4.0];
    let weight = vec![1.0, 1.0];
    let eps = 0.0;
    let output = rmsnorm_reference(&input, &weight, 1, 2, eps);
    let mean_sq: f32 = f32::midpoint(9.0, 16.0);
    let inv_rms = 1.0 / mean_sq.sqrt();
    for (i, &v) in output.iter().enumerate() {
        let expected = input[i] * inv_rms;
        assert!(
            (v - expected).abs() < 1e-5,
            "rmsnorm eps=0: output[{i}]={v}, exp={expected}"
        );
    }
}

#[test]
fn test_rmsnorm_ref_hidden_dim_1() {
    // Single element: mean(x^2) = x^2, inv_rms = 1/|x|, output = sign(x) * weight
    let input = vec![3.0];
    let weight = vec![2.0];
    let eps = 0.0;
    let output = rmsnorm_reference(&input, &weight, 1, 1, eps);
    // inv_rms = 1/sqrt(9) = 1/3, output = 3 * 1/3 * 2 = 2
    assert!(
        (output[0] - 2.0).abs() < 1e-5,
        "rmsnorm dim=1: output={}, exp=2.0",
        output[0]
    );
}

#[test]
fn test_rmsnorm_ref_hidden_dim_1024() {
    let dim = 1024;
    let input: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
    let weight = vec![1.0; dim];
    let eps = 1e-5;
    let output = rmsnorm_reference(&input, &weight, 1, dim, eps);
    assert_eq!(output.len(), dim);
    assert_all_finite(&output, "rmsnorm_dim1024");
}

// ====================================================================
// Softmax extended reference tests
// ====================================================================

#[test]
fn test_softmax_ref_1d_uniform() {
    // All same values -> uniform distribution
    let input = vec![3.0; 5];
    let output = reference_softmax(&input, 1, 5);
    for (i, &v) in output.iter().enumerate() {
        assert!(
            (v - 0.2).abs() < 1e-6,
            "softmax uniform: output[{i}]={v}, exp=0.2"
        );
    }
}

#[test]
fn test_softmax_ref_1d_one_hot() {
    // One large value, rest small -> approximately one-hot
    let input = vec![0.0, 0.0, 100.0, 0.0];
    let output = reference_softmax(&input, 1, 4);
    assert!(
        output[2] > 0.99,
        "softmax one-hot: max prob should be ~1.0, got {}",
        output[2]
    );
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum={sum}");
}

#[test]
fn test_softmax_ref_2d_rows_independent() {
    let input = vec![
        1.0, 2.0, 3.0, // row 0
        -1.0, -2.0, -3.0, // row 1
    ];
    let output = reference_softmax(&input, 2, 3);

    // Each row sums to 1
    let sum0: f32 = output[0..3].iter().sum();
    let sum1: f32 = output[3..6].iter().sum();
    assert!((sum0 - 1.0).abs() < 1e-6, "row0 sum={sum0}");
    assert!((sum1 - 1.0).abs() < 1e-6, "row1 sum={sum1}");

    // Row 0 is increasing, row 1 is decreasing
    assert!(
        output[2] > output[1] && output[1] > output[0],
        "row0 monotonic"
    );
    assert!(
        output[3] > output[4] && output[4] > output[5],
        "row1 monotonic"
    );
}

#[test]
fn test_softmax_ref_large_positive_stability() {
    let input = vec![1e6, 1e6 + 1.0, 1e6 + 2.0];
    let output = reference_softmax(&input, 1, 3);
    assert_all_finite(&output, "softmax_large_pos");
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax large pos sum={sum}");
}

#[test]
fn test_softmax_ref_large_negative_stability() {
    let input = vec![-1e6, -1e6 + 1.0, -1e6 + 2.0];
    let output = reference_softmax(&input, 1, 3);
    assert_all_finite(&output, "softmax_large_neg");
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax large neg sum={sum}");
}

#[test]
fn test_softmax_ref_single_element() {
    let output = reference_softmax(&[42.0], 1, 1);
    assert!(
        (output[0] - 1.0).abs() < 1e-6,
        "softmax single={}",
        output[0]
    );
}

#[test]
fn test_softmax_ref_wide_row_512() {
    let cols = 512;
    let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01).collect();
    let output = reference_softmax(&input, 1, cols);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "softmax 512 sum={sum}");
    assert_all_finite(&output, "softmax_512");
}

// ====================================================================
// GELU/SiLU/Snake activation extended reference tests
// ====================================================================

#[test]
fn test_gelu_ref_symmetry_approx() {
    // GELU(-x) + GELU(x) ~ 0 for small x (approximately odd around 0)
    // Actually GELU is not odd, but GELU(x) + GELU(-x) ~ x for small x... let's just check values
    let vals = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
    for &x in &vals {
        let r = gelu_reference(x);
        assert!(r.is_finite(), "GELU({x}) = {r} must be finite");
    }
    // GELU(0) = 0
    assert!(gelu_reference(0.0).abs() < 1e-6);
}

#[test]
fn test_gelu_ref_positive_for_large_x() {
    for x in [5.0, 10.0, 50.0, 100.0] {
        let r = gelu_reference(x);
        assert!(
            (r - x).abs() < 0.01 * x.abs().max(1.0),
            "GELU({x}) ~ {x} for large positive, got {r}"
        );
    }
}

#[test]
fn test_gelu_ref_near_zero_for_large_negative() {
    for x in [-5.0, -10.0, -50.0] {
        let r = gelu_reference(x);
        assert!(r.abs() < 0.01, "GELU({x}) ~ 0 for large negative, got {r}");
    }
}

#[test]
fn test_silu_ref_positive_for_positive_x() {
    for x in [0.1, 0.5, 1.0, 5.0, 10.0] {
        let r = silu_reference(x);
        assert!(r > 0.0, "SiLU({x}) should be > 0, got {r}");
    }
}

#[test]
fn test_silu_ref_negative_for_small_negative_x() {
    // SiLU(-1) < 0 (minimum is around x ~ -1.28)
    let r = silu_reference(-1.0);
    assert!(r < 0.0, "SiLU(-1) should be < 0, got {r}");
}

#[test]
fn test_silu_ref_approaches_zero_for_large_neg() {
    let r = silu_reference(-20.0);
    assert!(r.abs() < 1e-6, "SiLU(-20) ~ 0, got {r}");
}

#[test]
fn test_snake_ref_alpha_variations() {
    let x = 1.0;
    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
        let r = snake_reference(x, alpha);
        assert!(r.is_finite(), "Snake({x}, {alpha}) must be finite, got {r}");
        // Snake(x) >= x (since sin^2 >= 0)
        assert!(r >= x - 1e-6, "Snake({x}, {alpha}) = {r} should be >= {x}");
    }
}

#[test]
fn test_snake_ref_x_plus_bounded_correction() {
    // The sin^2 term is bounded: 0 <= sin^2 <= 1
    // So Snake(x, alpha) is in [x, x + 1/alpha]
    let alpha = 2.0;
    for i in -50..=50 {
        let x = i as f32 * 0.1;
        let r = snake_reference(x, alpha);
        assert!(r >= x - 1e-6, "Snake lower bound violated at x={x}");
        assert!(
            r <= x + 1.0 / alpha + 1e-6,
            "Snake upper bound violated at x={x}: {r}"
        );
    }
}

#[test]
fn test_snake_ref_negative_x() {
    let r = snake_reference(-5.0, 1.0);
    // Snake(-5) = -5 + sin(-5)^2
    let expected = -5.0 + (-5.0_f32).sin().powi(2);
    assert!(
        (r - expected).abs() < 1e-5,
        "Snake(-5, 1) = {r}, expected {expected}"
    );
}

// ====================================================================
// SPIR-V generation: BatchNorm
// ====================================================================

#[test]
fn test_batchnorm_spirv_magic_various_channels() {
    for ch in [1, 4, 16, 32, 64, 128, 256, 512] {
        let words = generate_batchnorm_spirv(ch, NORM_WORKGROUP_SIZE);
        assert_eq!(words[0], SPIRV_MAGIC, "batchnorm ch={ch}: wrong magic");
    }
}

#[test]
fn test_batchnorm_spirv_entry_point_all_sizes() {
    for ch in [8, 32, 128] {
        let words = generate_batchnorm_spirv(ch, NORM_WORKGROUP_SIZE);
        let name = find_entry_point_name(&words).expect("must have entry point");
        assert_eq!(
            name, "main",
            "batchnorm ch={ch}: entry point should be main"
        );
    }
}

#[test]
fn test_batchnorm_spirv_workgroup_default() {
    let words = generate_batchnorm_spirv(32, NORM_WORKGROUP_SIZE);
    let wg = find_workgroup_size(&words).expect("must have wg size");
    assert_eq!(wg, [NORM_WORKGROUP_SIZE, 1, 1]);
}

// ====================================================================
// SPIR-V generation: GroupNorm
// ====================================================================

#[test]
fn test_groupnorm_spirv_magic_various_configs() {
    let configs = [(1, 16), (2, 32), (4, 64), (8, 128), (16, 256), (32, 256)];
    for (g, c) in configs {
        let words = generate_groupnorm_spirv(g, c, NORM_WORKGROUP_SIZE);
        assert_eq!(words[0], SPIRV_MAGIC, "groupnorm g={g} c={c}: wrong magic");
    }
}

#[test]
fn test_groupnorm_spirv_entry_point() {
    let words = generate_groupnorm_spirv(4, 32, NORM_WORKGROUP_SIZE);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_groupnorm_spirv_workgroup_size_validation() {
    let words = generate_groupnorm_spirv(8, 64, NORM_WORKGROUP_SIZE);
    let wg = find_workgroup_size(&words).expect("must have wg size");
    assert_eq!(wg, [NORM_WORKGROUP_SIZE, 1, 1]);
}

// ====================================================================
// SPIR-V generation: InstanceNorm
// ====================================================================

#[test]
fn test_instancenorm_spirv_magic_various() {
    for ch in [1, 8, 32, 64, 256, 512] {
        let words = generate_instancenorm_spirv(ch, NORM_WORKGROUP_SIZE);
        assert_eq!(words[0], SPIRV_MAGIC, "instancenorm ch={ch}: wrong magic");
    }
}

#[test]
fn test_instancenorm_spirv_entry_point_all() {
    for ch in [16, 64, 256] {
        let words = generate_instancenorm_spirv(ch, NORM_WORKGROUP_SIZE);
        let name = find_entry_point_name(&words).expect("must have entry point");
        assert_eq!(name, "main", "instancenorm ch={ch}");
    }
}

#[test]
fn test_instancenorm_spirv_workgroup_size_validation() {
    let words = generate_instancenorm_spirv(128, NORM_WORKGROUP_SIZE);
    let wg = find_workgroup_size(&words).expect("must have wg size");
    assert_eq!(wg, [NORM_WORKGROUP_SIZE, 1, 1]);
}

// ====================================================================
// SPIR-V generation: RMSNorm
// ====================================================================

#[test]
fn test_rmsnorm_spirv_magic_various_dims() {
    for dim in [1, 64, 128, 256, 512, 768, 1024, 4096] {
        let config = RmsNormConfig::new(dim, 1e-5);
        let bytes = generate_rmsnorm_separate_io_spirv(&config);
        let words = bytes_to_words(&bytes);
        assert_eq!(words[0], SPIRV_MAGIC, "rmsnorm dim={dim}: wrong magic");
    }
}

#[test]
fn test_rmsnorm_spirv_entry_point_default() {
    let config = RmsNormConfig::new(768, 1e-5);
    let bytes = generate_rmsnorm_separate_io_spirv(&config);
    let words = bytes_to_words(&bytes);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_rmsnorm_spirv_workgroup_size_validation() {
    let config = RmsNormConfig::new(768, 1e-5);
    let bytes = generate_rmsnorm_separate_io_spirv(&config);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("must have wg size");
    assert_eq!(wg, [RMSNORM_WORKGROUP_SIZE, 1, 1]);
}

// ====================================================================
// SPIR-V generation: Softmax
// ====================================================================

#[test]
fn test_softmax_spirv_magic_various() {
    for (r, c) in [(1, 1), (1, 4), (4, 16), (32, 128), (64, 4096)] {
        let bytes = generate_softmax_separate_io_spirv(r, c);
        let words = bytes_to_words(&bytes);
        assert_eq!(words[0], SPIRV_MAGIC, "softmax {r}x{c}: wrong magic");
    }
}

#[test]
fn test_softmax_spirv_entry_point() {
    let bytes = generate_softmax_separate_io_spirv(16, 64);
    let words = bytes_to_words(&bytes);
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_softmax_spirv_workgroup_size_validation() {
    let bytes = generate_softmax_separate_io_spirv(32, 128);
    let words = bytes_to_words(&bytes);
    let wg = find_workgroup_size(&words).expect("must have wg size");
    assert_eq!(wg, [SOFTMAX_WORKGROUP_SIZE, 1, 1]);
}

// ====================================================================
// SPIR-V generation: Activations
// ====================================================================

#[test]
fn test_gelu_spirv_magic_and_entry() {
    let bytes = generate_gelu_spirv(ACTIVATION_WORKGROUP_SIZE);
    let words = bytes_to_words(&bytes);
    assert_eq!(words[0], SPIRV_MAGIC, "gelu: wrong magic");
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_silu_spirv_magic_and_entry() {
    let bytes = generate_silu_spirv(ACTIVATION_WORKGROUP_SIZE);
    let words = bytes_to_words(&bytes);
    assert_eq!(words[0], SPIRV_MAGIC, "silu: wrong magic");
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_snake_spirv_magic_and_entry() {
    let bytes = generate_snake_spirv(ACTIVATION_WORKGROUP_SIZE);
    let words = bytes_to_words(&bytes);
    assert_eq!(words[0], SPIRV_MAGIC, "snake: wrong magic");
    let name = find_entry_point_name(&words).expect("must have entry point");
    assert_eq!(name, "main");
}

#[test]
fn test_gelu_spirv_workgroup_size_validation() {
    for wg in [64, 128, 256, 512] {
        let bytes = generate_gelu_spirv(wg);
        let words = bytes_to_words(&bytes);
        let found = find_workgroup_size(&words).expect("must have wg size");
        assert_eq!(found, [wg, 1, 1], "gelu wg={wg}: wrong workgroup size");
    }
}

#[test]
fn test_silu_spirv_workgroup_size_validation() {
    let bytes = generate_silu_spirv(256);
    let words = bytes_to_words(&bytes);
    let found = find_workgroup_size(&words).expect("must have wg size");
    assert_eq!(found, [256, 1, 1]);
}

#[test]
fn test_snake_spirv_workgroup_size_validation() {
    let bytes = generate_snake_spirv(256);
    let words = bytes_to_words(&bytes);
    let found = find_workgroup_size(&words).expect("must have wg size");
    assert_eq!(found, [256, 1, 1]);
}

// ====================================================================
// Edge cases: single element
// ====================================================================

#[test]
fn test_batchnorm_ref_single_element() {
    let input = vec![5.0]; // [N=1, C=1, S=1]
    let mean = vec![3.0];
    let var = vec![4.0];
    let weight = vec![2.0];
    let bias = vec![1.0];
    let eps = 1e-5;
    let output = batchnorm_reference(&input, &mean, &var, &weight, &bias, 1, 1, 1, eps);
    let expected = (5.0 - 3.0) / (4.0 + eps).sqrt() * 2.0 + 1.0;
    assert!(
        (output[0] - expected).abs() < 1e-4,
        "bn single: {}, exp {expected}",
        output[0]
    );
}

#[test]
fn test_groupnorm_ref_single_element() {
    // [N=1, C=1, S=1] groups=1 -> variance=0, normalized=0
    let input = vec![7.0];
    let weight = vec![1.0];
    let bias = vec![2.0];
    let eps = 1e-5;
    let output = groupnorm_reference(&input, &weight, &bias, 1, 1, 1, 1, eps);
    // mean=7, var=0, normalized=(7-7)/sqrt(0+eps)=0, output=0*1+2=2
    assert!(
        (output[0] - 2.0).abs() < 1e-3,
        "gn single: {}, exp 2.0",
        output[0]
    );
}

#[test]
fn test_rmsnorm_ref_single_element() {
    let input = vec![4.0];
    let weight = vec![1.0];
    let eps = 0.0;
    let output = rmsnorm_reference(&input, &weight, 1, 1, eps);
    // mean(x^2)=16, inv_rms=1/4, output=4*1/4*1=1
    assert!(
        (output[0] - 1.0).abs() < 1e-5,
        "rmsnorm single: {}, exp 1.0",
        output[0]
    );
}

// ====================================================================
// Edge cases: very large channels
// ====================================================================

#[test]
fn test_batchnorm_spirv_large_channels() {
    let words = generate_batchnorm_spirv(1024, NORM_WORKGROUP_SIZE);
    assert_valid_spirv_header(&words, "batchnorm_1024ch");
}

#[test]
fn test_groupnorm_spirv_large_channels() {
    let words = generate_groupnorm_spirv(32, 1024, NORM_WORKGROUP_SIZE);
    assert_valid_spirv_header(&words, "groupnorm_32g_1024ch");
}

#[test]
fn test_instancenorm_spirv_large_channels() {
    let words = generate_instancenorm_spirv(1024, NORM_WORKGROUP_SIZE);
    assert_valid_spirv_header(&words, "instancenorm_1024ch");
}

// ====================================================================
// Edge cases: eps=0 (boundary behavior)
// ====================================================================

#[test]
fn test_batchnorm_ref_eps_0_with_nonzero_var() {
    let input = vec![1.0, 2.0, 3.0, 4.0]; // [N=1, C=2, S=2]
    let mean = vec![1.5, 3.5];
    let var = vec![0.25, 0.25];
    let weight = vec![1.0, 1.0];
    let bias = vec![0.0, 0.0];
    let eps = 0.0;
    let output = batchnorm_reference(&input, &mean, &var, &weight, &bias, 1, 2, 2, eps);
    assert_all_finite(&output, "batchnorm_eps0");
    let inv_std = 1.0 / 0.25_f32.sqrt();
    let expected_0 = (1.0 - 1.5) * inv_std;
    assert!((output[0] - expected_0).abs() < 1e-4);
}

#[test]
fn test_instancenorm_ref_eps_0_with_varying_input() {
    let input = vec![1.0, 3.0, 5.0, 7.0]; // [N=1, C=1, S=4]
    let weight = vec![1.0];
    let bias = vec![0.0];
    let eps = 0.0;
    let output = instancenorm_reference(&input, &weight, &bias, 1, 1, 4, eps);
    assert_all_finite(&output, "instancenorm_eps0");
}

// ====================================================================
// Workgroup size constant validation
// ====================================================================

#[test]
fn test_norm_workgroup_size_is_256() {
    assert_eq!(NORM_WORKGROUP_SIZE, 256);
}

#[test]
fn test_rmsnorm_workgroup_size_is_256() {
    assert_eq!(RMSNORM_WORKGROUP_SIZE, 256);
}

#[test]
fn test_softmax_workgroup_size_is_256() {
    assert_eq!(SOFTMAX_WORKGROUP_SIZE, 256);
}

#[test]
fn test_activation_workgroup_size_is_256() {
    assert_eq!(ACTIVATION_WORKGROUP_SIZE, 256);
}

// ====================================================================
// Cross-norm consistency tests
// ====================================================================

#[test]
fn test_instancenorm_equals_groupnorm_with_groups_eq_channels_extended() {
    // Verify for multiple channel counts and spatial sizes
    for (channels, spatial) in [(2, 4), (4, 8), (8, 16), (16, 4)] {
        let n = channels * spatial;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 2.0).collect();
        let weight = vec![1.0; channels];
        let bias = vec![0.0; channels];
        let eps = 1e-5;
        let gn = groupnorm_reference(&input, &weight, &bias, 1, channels, channels, spatial, eps);
        let inst = instancenorm_reference(&input, &weight, &bias, 1, channels, spatial, eps);
        for (i, (&g, &in_v)) in gn.iter().zip(inst.iter()).enumerate() {
            assert!(
                (g - in_v).abs() < 1e-4,
                "C={channels} S={spatial}: gn[{i}]={g} != in[{i}]={in_v}"
            );
        }
    }
}

#[test]
fn test_all_norms_produce_finite_output_for_standard_input() {
    let batch = 2;
    let channels = 8;
    let spatial = 4;
    let n = batch * channels * spatial;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 3.0).collect();
    let weight = vec![1.0; channels];
    let bias = vec![0.0; channels];
    let eps = 1e-5;

    // BatchNorm
    let mean = vec![0.0; channels];
    let var = vec![1.0; channels];
    let bn = batchnorm_reference(
        &input, &mean, &var, &weight, &bias, batch, channels, spatial, eps,
    );
    assert_all_finite(&bn, "all_norms_bn");

    // GroupNorm (groups=4)
    let gn = groupnorm_reference(&input, &weight, &bias, batch, 4, channels, spatial, eps);
    assert_all_finite(&gn, "all_norms_gn");

    // InstanceNorm
    let inst = instancenorm_reference(&input, &weight, &bias, batch, channels, spatial, eps);
    assert_all_finite(&inst, "all_norms_inst");
}
