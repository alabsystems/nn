#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional validation and numerical correctness tests for ViT.
//!
//! Core shape/trait tests live in `vit_tests.rs`.
//! VarBuilder loading tests live in `vit_tests_loading.rs`.

use super::*;

// -- Additional validation tests ----------------------------------------------

#[test]
fn test_config_validate_zero_intermediate_size() {
    let mut config = small_config(false);
    config.intermediate_size = 0;
    let err = config.validate().unwrap_err().to_string();
    assert!(err.contains("intermediate_size"), "error: {err}");
}

#[test]
fn test_config_validate_zero_num_channels() {
    let mut config = small_config(false);
    config.num_channels = 0;
    let err = config.validate().unwrap_err().to_string();
    assert!(err.contains("num_channels"), "error: {err}");
}

#[test]
fn test_config_validate_zero_num_heads() {
    let mut config = small_config(false);
    config.num_heads = 0;
    let err = config.validate().unwrap_err().to_string();
    assert!(err.contains("num_heads"), "error: {err}");
}

#[test]
fn test_config_validate_negative_eps() {
    let mut config = small_config(false);
    config.layer_norm_eps = -1.0;
    let err = config.validate().unwrap_err().to_string();
    assert!(err.contains("eps"), "error: {err}");
}

// -- PatchEmbedding Module trait ----------------------------------------------

#[test]
fn test_patch_embed_module_trait() {
    let config = small_config(false);
    let pe = make_patch_embed(&config);
    let img = make_image(1, &config, 0.0);
    let out = Module::forward(&pe, &img).unwrap();
    assert_eq!(out.dims(), &[1, 4, 32]);
}

// -- Numerical correctness tests (#1319) --------------------------------------

/// PatchEmbedding: verify Conv2d produces non-trivial output and values differ
/// across patches (not all zeros or all identical).
#[test]
fn test_patch_embed_numerical_nontrivial() {
    let config = small_config(false);
    let pe = make_patch_embed(&config);
    let img = make_image(1, &config, 0.0);
    let out = pe.forward(&img).unwrap();
    let vals = out.to_flat_vec::<f32>().unwrap();
    // Output should not be all zeros (weights are non-zero, input is non-zero).
    let all_zero = vals.iter().all(|&v| v.abs() < 1e-12);
    assert!(!all_zero, "PatchEmbedding output should not be all zeros");
    // Different patches should have different representations.
    // out shape: [1, 4, 32] — compare patch 0 and patch 1.
    let patch0 = &vals[..32];
    let patch1 = &vals[32..64];
    let diff: f32 = patch0
        .iter()
        .zip(patch1.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 1e-6, "Patches should differ, total diff = {diff}");
}

/// EncoderBlock residual: with near-identity weights (LayerNorm weight~1, bias~0),
/// output should be close to input plus small perturbation.
#[test]
fn test_encoder_block_residual_numerically() {
    let d = 16;
    let block = make_encoder_block(d, 2, 32);
    // Use a uniform input to make behavior predictable through layer norm.
    let x_data: Vec<f32> = (0..d).map(|i| i as f32 * 0.1).collect();
    let x = DynTensor::from_vec(x_data.clone(), &[1, 1, d], &Device::Cpu).unwrap();
    let out = block.forward(&x).unwrap();
    let out_vals = out.to_flat_vec::<f32>().unwrap();
    // With small det_data weights, the residual connection means output ≈ input + small delta.
    // Verify that output has meaningful values (not zeros, not garbage).
    let max_val = out_vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_val > 1e-6 && max_val < 100.0,
        "Output values should be reasonable, max_abs = {max_val}"
    );
    // Output should differ from input (attention + MLP add something).
    let diff: f32 = x_data
        .iter()
        .zip(out_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-6,
        "Output should differ from input, total diff = {diff}"
    );
}

/// Attention scale: verify 1/sqrt(head_dim) is applied. With head_dim=8 (d=16, heads=2),
/// scale = 1/sqrt(8) ≈ 0.3536. The encoder block stores this as `self.scale`.
#[test]
fn test_encoder_block_scale_factor() {
    let d = 16;
    let num_heads = 2;
    let block = make_encoder_block(d, num_heads, 32);
    let expected_scale = 1.0 / ((d / num_heads) as f64).sqrt();
    assert!(
        (block.scale - expected_scale).abs() < 1e-10,
        "scale should be 1/sqrt(head_dim), expected {expected_scale}, got {}",
        block.scale
    );
}

/// Full encoder: deterministic output with fixed config and input.
/// Verify specific values to catch sign errors, wrong activations, etc.
#[test]
fn test_vit_encoder_deterministic_output() {
    let config = small_config(false);
    let encoder = make_encoder(&config, 1); // 1 block for simplicity
    let img = make_image(1, &config, 0.0);
    let out = encoder.forward(&img, PoolingStrategy::Mean).unwrap();
    assert_eq!(out.dims(), &[1, 32]);
    let vals = out.to_flat_vec::<f32>().unwrap();
    // Run twice: output must be identical (deterministic).
    let out2 = encoder.forward(&img, PoolingStrategy::Mean).unwrap();
    let vals2 = out2.to_flat_vec::<f32>().unwrap();
    for (i, (a, b)) in vals.iter().zip(vals2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-7,
            "Output should be deterministic: val[{i}] = {a} vs {b}"
        );
    }
    // Verify output is not trivial (not all zeros, not all same).
    let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
    let variance: f32 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
    assert!(
        variance > 1e-10,
        "Output should have variance, got {variance}"
    );
    // Hardcoded reference values from det_data seed=0.0 input, seed=1.0-20.0 weights,
    // d=32, 1 block, mean pooling. These catch sign errors, wrong activations, and
    // incorrect attention scale. Tolerance 1e-5 accounts for float non-associativity.
    let expected = [0.44534808_f32, 1.6679101, 2.3274307, 2.2813983];
    for (i, (&actual, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - exp).abs() < 1e-5,
            "Output mismatch at [{i}]: actual={actual}, expected={exp}, diff={}",
            (actual - exp).abs()
        );
    }
}
