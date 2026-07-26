// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for BatchNorm2d Metal GPU kernel dispatch (#4324).
//!
//! Covers 9 areas of the fused BatchNorm2d kernel pipeline:
//! 1. Forward pass configuration: BatchNorm2d construction and parameter validation
//! 2. Running mean/variance buffer handling: shape checks, channel alignment
//! 3. Training vs inference mode dispatch: eval-only GPU path
//! 4. Epsilon parameter handling: valid, zero, negative, NaN, large values
//! 5. Channel dimension dispatch (NCHW layout): channel indexing correctness
//! 6. Affine parameter (weight/bias) handling: all 4 combinations
//! 7. Momentum parameter in training mode: config builder coverage
//! 8. Multi-batch dimension handling: batch independence
//! 9. DType compatibility: F32, F16, BF16 scalar type mapping
//!
//! All tests are structure/config tests that exercise CPU-side validation
//! logic and construction paths. Tests that need live GPU Metal dispatch
//! are guarded by MetalContext availability.
//!
//! Part of #4324.

use nn_core::layers::{BatchNorm2d, BatchNormConfig, Module};
use nn_core::{DType, Device, DynTensor};
use nn_dsl::ir::ScalarType;
use nn_dsl::trace_compile::NativeOpKind;

// ═══════════════════════════════════════════════════════════════════════
// 1. BatchNorm2d forward pass configuration
// ═══════════════════════════════════════════════════════════════════════

/// Construct BatchNorm2d with zero mean and unit variance (identity normalization).
#[test]
fn batchnorm2d_forward_identity_normalization() {
    let channels = 4;
    let mean = DynTensor::zeros(&[channels], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[channels], DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm2d::new(channels, mean, var, None, None, 1e-5).unwrap();

    let x = DynTensor::ones(&[1, 4, 8, 8], DType::F32, &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 4, 8, 8]);

    // With zero mean and unit variance, output should be near input.
    let vals = y.to_flat_vec::<f32>().unwrap();
    for v in &vals {
        assert!(
            (v - 1.0).abs() < 1e-3,
            "identity BN should preserve input: got {v}"
        );
    }
}

/// BatchNorm2d rejects rank-3 input (expects exactly 4D).
#[test]
fn batchnorm2d_forward_rejects_rank3() {
    let channels = 2;
    let mean = DynTensor::zeros(&[channels], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[channels], DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm2d::new(channels, mean, var, None, None, 1e-5).unwrap();

    let x = DynTensor::ones(&[1, 2, 16], DType::F32, &Device::Cpu).unwrap();
    let err = bn.forward(&x).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("Rank"),
        "should reject rank-3 input: {msg}"
    );
}

/// BatchNorm2d rejects rank-5 input.
#[test]
fn batchnorm2d_forward_rejects_rank5() {
    let channels = 2;
    let mean = DynTensor::zeros(&[channels], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[channels], DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm2d::new(channels, mean, var, None, None, 1e-5).unwrap();

    let x = DynTensor::ones(&[1, 2, 3, 4, 5], DType::F32, &Device::Cpu).unwrap();
    let err = bn.forward(&x).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("Rank"),
        "should reject rank-5 input: {msg}"
    );
}

/// BatchNorm2d rejects channel mismatch (num_features != input C dim).
#[test]
fn batchnorm2d_forward_rejects_channel_mismatch() {
    let channels = 8;
    let mean = DynTensor::zeros(&[channels], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[channels], DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm2d::new(channels, mean, var, None, None, 1e-5).unwrap();

    // Input has 4 channels, but BN expects 8.
    let x = DynTensor::ones(&[1, 4, 8, 8], DType::F32, &Device::Cpu).unwrap();
    let err = bn.forward(&x).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("Shape") || msg.contains("mismatch"),
        "should reject channel mismatch: {msg}"
    );
}

/// BatchNorm2d with known values produces correct output.
#[test]
fn batchnorm2d_forward_known_values() {
    // mean=[1.0, 2.0], var=[4.0, 1.0], eps=0
    // Input [1, 2, 1, 1]: [3.0, 4.0]
    // ch0: (3.0-1.0)/sqrt(4.0) = 2.0/2.0 = 1.0
    // ch1: (4.0-2.0)/sqrt(1.0) = 2.0/1.0 = 2.0
    let mean = DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap();
    let var = DynTensor::new(&[4.0, 1.0], &[2], &Device::Cpu).unwrap();
    let bn = BatchNorm2d::new(2, mean, var, None, None, 0.0).unwrap();

    let x = DynTensor::new(&[3.0, 4.0], &[1, 2, 1, 1], &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 1.0).abs() < 1e-5, "ch0: expected 1.0, got {}", vals[0]);
    assert!((vals[1] - 2.0).abs() < 1e-5, "ch1: expected 2.0, got {}", vals[1]);
}

// ═══════════════════════════════════════════════════════════════════════
// 2. Running mean/variance buffer handling
// ═══════════════════════════════════════════════════════════════════════

/// Running statistics must have shape [C] matching num_features.
#[test]
fn batchnorm2d_running_stats_shape_accessors() {
    let channels = 16;
    let mean = DynTensor::zeros(&[channels], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[channels], DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm2d::new(channels, mean, var, None, None, 1e-5).unwrap();

    assert_eq!(bn.running_mean().dims(), &[channels]);
    assert_eq!(bn.running_var().dims(), &[channels]);
    assert_eq!(bn.num_features(), channels);
}

/// Running statistics tensor values are preserved after construction.
#[test]
fn batchnorm2d_running_stats_values_preserved() {
    let mean_data = vec![0.5, -0.5, 1.0, 2.0];
    let var_data = vec![1.0, 2.0, 0.5, 4.0];
    let mean = DynTensor::new(&mean_data, &[4], &Device::Cpu).unwrap();
    let var = DynTensor::new(&var_data, &[4], &Device::Cpu).unwrap();
    let bn = BatchNorm2d::new(4, mean, var, None, None, 1e-5).unwrap();

    let rm = bn.running_mean().to_flat_vec::<f32>().unwrap();
    let rv = bn.running_var().to_flat_vec::<f32>().unwrap();
    assert_eq!(rm, mean_data);
    assert_eq!(rv, var_data);
}

/// Single-channel running statistics.
#[test]
fn batchnorm2d_running_stats_single_channel() {
    let mean = DynTensor::new(&[3.14], &[1], &Device::Cpu).unwrap();
    let var = DynTensor::new(&[2.0], &[1], &Device::Cpu).unwrap();
    let bn = BatchNorm2d::new(1, mean, var, None, None, 1e-5).unwrap();

    let x = DynTensor::new(&[3.14], &[1, 1, 1, 1], &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // (3.14 - 3.14) / sqrt(2 + 1e-5) = 0
    assert!(vals[0].abs() < 1e-4, "zero-mean input: expected ~0, got {}", vals[0]);
}

/// Many-channel (256) running statistics.
#[test]
fn batchnorm2d_running_stats_many_channels() {
    let channels = 256;
    let mean = DynTensor::zeros(&[channels], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[channels], DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm2d::new(channels, mean, var, None, None, 1e-5).unwrap();

    assert_eq!(bn.running_mean().dims(), &[channels]);
    assert_eq!(bn.running_var().dims(), &[channels]);

    // Forward should work on standard ResNet-like shapes.
    let x = DynTensor::ones(&[2, 256, 7, 7], DType::F32, &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 256, 7, 7]);
}

// ═══════════════════════════════════════════════════════════════════════
// 3. Training vs inference mode dispatch
// ═══════════════════════════════════════════════════════════════════════

/// BatchNorm.forward uses inference path (frozen running stats).
#[test]
fn batchnorm2d_inference_mode_frozen_stats() {
    let mean = DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap();
    let var = DynTensor::new(&[1.0, 1.0], &[2], &Device::Cpu).unwrap();
    let bn = BatchNorm2d::new(2, mean, var, None, None, 1e-5).unwrap();

    // Run forward twice: stats should not change (inference mode).
    let x = DynTensor::from_vec(
        vec![10.0, 20.0, 30.0, 40.0],
        &[1, 2, 1, 2],
        &Device::Cpu,
    )
    .unwrap();
    let y1 = bn.forward(&x).unwrap();
    let y2 = bn.forward(&x).unwrap();

    let v1 = y1.to_flat_vec::<f32>().unwrap();
    let v2 = y2.to_flat_vec::<f32>().unwrap();
    assert_eq!(v1, v2, "inference mode should produce deterministic output");

    // Running stats should be unchanged.
    let rm = bn.running_mean().to_flat_vec::<f32>().unwrap();
    let rv = bn.running_var().to_flat_vec::<f32>().unwrap();
    assert_eq!(rm, vec![1.0, 2.0], "running_mean should be frozen");
    assert_eq!(rv, vec![1.0, 1.0], "running_var should be frozen");
}

/// NativeOpKind::BatchNorm2d has estimated dispatch count of 1 (fused kernel).
#[test]
fn batchnorm2d_native_op_dispatch_count_is_one() {
    let op = NativeOpKind::BatchNorm2d {
        eps: 1e-5,
        num_channels: 64,
        input_shape: vec![1, 64, 32, 32],
        has_weight: true,
        has_bias: true,
    };
    assert_eq!(
        op.estimated_metal_dispatches(),
        1,
        "fused BatchNorm2d should be a single dispatch"
    );
}

/// NativeOpKind::BatchNorm2d variant_name is "BatchNorm2d".
#[test]
fn batchnorm2d_native_op_variant_name() {
    let op = NativeOpKind::BatchNorm2d {
        eps: 1e-5,
        num_channels: 32,
        input_shape: vec![2, 32, 16, 16],
        has_weight: false,
        has_bias: false,
    };
    assert_eq!(op.variant_name(), "BatchNorm2d");
}

// ═══════════════════════════════════════════════════════════════════════
// 4. Epsilon parameter handling
// ═══════════════════════════════════════════════════════════════════════

/// BatchNorm construction with zero epsilon succeeds (valid edge case).
#[test]
fn batchnorm2d_eps_zero_succeeds() {
    let mean = DynTensor::zeros(&[2], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[2], DType::F32, &Device::Cpu).unwrap();
    let result = BatchNorm2d::new(2, mean, var, None, None, 0.0);
    assert!(result.is_ok(), "eps=0 is valid (boundary case)");
}

/// BatchNorm construction with negative epsilon fails.
#[test]
fn batchnorm2d_eps_negative_rejected() {
    let mean = DynTensor::zeros(&[2], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[2], DType::F32, &Device::Cpu).unwrap();
    let result = BatchNorm2d::new(2, mean, var, None, None, -1e-5);
    assert!(result.is_err(), "negative eps should be rejected");
}

/// BatchNorm construction with NaN epsilon fails.
#[test]
fn batchnorm2d_eps_nan_rejected() {
    let mean = DynTensor::zeros(&[2], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[2], DType::F32, &Device::Cpu).unwrap();
    let result = BatchNorm2d::new(2, mean, var, None, None, f64::NAN);
    assert!(result.is_err(), "NaN eps should be rejected");
}

/// BatchNorm construction with infinite epsilon fails.
#[test]
fn batchnorm2d_eps_inf_rejected() {
    let mean = DynTensor::zeros(&[2], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[2], DType::F32, &Device::Cpu).unwrap();
    let result = BatchNorm2d::new(2, mean, var, None, None, f64::INFINITY);
    assert!(result.is_err(), "infinite eps should be rejected");
}

/// Large epsilon should produce valid (but attenuated) output.
#[test]
fn batchnorm2d_eps_large_value_attenuates() {
    let mean = DynTensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[1], DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm2d::new(1, mean, var, None, None, 100.0).unwrap();

    let x = DynTensor::new(&[10.0], &[1, 1, 1, 1], &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // (10 - 0) / sqrt(1 + 100) = 10 / ~10.05 = ~0.995
    let expected = 10.0 / (101.0_f32).sqrt();
    assert!(
        (vals[0] - expected).abs() < 1e-3,
        "large eps: expected ~{expected}, got {}",
        vals[0]
    );
}

/// Standard PyTorch epsilon 1e-5 with default config.
#[test]
fn batchnorm2d_eps_default_pytorch() {
    let config = BatchNormConfig::default();
    assert!(
        (config.eps - 1e-5).abs() < 1e-10,
        "default eps should be 1e-5, got {}",
        config.eps
    );
}

/// Custom epsilon via BatchNormConfig::new().
#[test]
fn batchnorm2d_eps_custom_config() {
    let config = BatchNormConfig::new(1e-3);
    assert!(
        (config.eps - 1e-3).abs() < 1e-10,
        "custom eps should be 1e-3, got {}",
        config.eps
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 5. Channel dimension dispatch (NCHW layout)
// ═══════════════════════════════════════════════════════════════════════

/// Per-channel normalization: each channel gets its own mean/var.
#[test]
fn batchnorm2d_nchw_per_channel_normalization() {
    // 3 channels: mean=[0, 10, 100], var=[1, 1, 1]
    // Input all zeros => output ch0=0, ch1=-10, ch2=-100
    let mean = DynTensor::new(&[0.0, 10.0, 100.0], &[3], &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[3], DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm2d::new(3, mean, var, None, None, 0.0).unwrap();

    let x = DynTensor::zeros(&[1, 3, 2, 2], DType::F32, &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();

    // ch0 (indices 0-3): (0 - 0) / 1 = 0
    for i in 0..4 {
        assert!(vals[i].abs() < 1e-5, "ch0[{i}]: expected 0, got {}", vals[i]);
    }
    // ch1 (indices 4-7): (0 - 10) / 1 = -10
    for i in 4..8 {
        assert!(
            (vals[i] - (-10.0)).abs() < 1e-4,
            "ch1[{i}]: expected -10, got {}",
            vals[i]
        );
    }
    // ch2 (indices 8-11): (0 - 100) / 1 = -100
    for i in 8..12 {
        assert!(
            (vals[i] - (-100.0)).abs() < 1e-3,
            "ch2[{i}]: expected -100, got {}",
            vals[i]
        );
    }
}

/// Spatial dimensions (H, W) are independent -- same channel value across pixels.
#[test]
fn batchnorm2d_nchw_spatial_independence() {
    let mean = DynTensor::new(&[5.0], &[1], &Device::Cpu).unwrap();
    let var = DynTensor::new(&[4.0], &[1], &Device::Cpu).unwrap();
    let bn = BatchNorm2d::new(1, mean, var, None, None, 0.0).unwrap();

    // All spatial positions have the same value => all outputs should be equal.
    let x = DynTensor::full(&[1, 1, 4, 4], 7.0, DType::F32, &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();

    // (7 - 5) / sqrt(4) = 2 / 2 = 1.0
    for (i, v) in vals.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-5,
            "spatial[{i}]: expected 1.0, got {v}"
        );
    }
}

/// NativeOpKind::BatchNorm2d correctly stores the NCHW input shape.
#[test]
fn batchnorm2d_nchw_native_op_shape_encoding() {
    let shape = vec![2, 64, 32, 32];
    let op = NativeOpKind::BatchNorm2d {
        eps: 1e-5,
        num_channels: 64,
        input_shape: shape.clone(),
        has_weight: true,
        has_bias: true,
    };
    match op {
        NativeOpKind::BatchNorm2d {
            input_shape,
            num_channels,
            ..
        } => {
            assert_eq!(input_shape, shape);
            assert_eq!(num_channels, 64);
        }
        _ => panic!("expected BatchNorm2d variant"),
    }
}

/// Channel index computation: flat_idx -> channel via (idx / spatial) % C.
#[test]
fn batchnorm2d_nchw_channel_index_formula() {
    // Verify the formula used in the MSL kernel:
    // channel = (flat_idx / spatial_size) % num_channels
    let n = 2;
    let c = 3;
    let h = 4;
    let w = 4;
    let spatial = h * w;

    for batch in 0..n {
        for ch in 0..c {
            for s in 0..spatial {
                let flat = batch * c * spatial + ch * spatial + s;
                let computed_ch = (flat / spatial) % c;
                assert_eq!(
                    computed_ch, ch,
                    "flat={flat}: expected ch={ch}, got {computed_ch}"
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 6. Affine parameter (weight/bias) handling
// ═══════════════════════════════════════════════════════════════════════

/// No weight, no bias: pure normalization.
#[test]
fn batchnorm2d_affine_none() {
    let mean = DynTensor::zeros(&[2], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[2], DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm2d::new(2, mean, var, None, None, 1e-5).unwrap();

    assert!(bn.weight().is_none());
    assert!(bn.bias().is_none());

    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 1, 2], &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 2, 1, 2]);
}

/// Weight only (no bias): output = normalized * weight.
#[test]
fn batchnorm2d_affine_weight_only() {
    let mean = DynTensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[1], DType::F32, &Device::Cpu).unwrap();
    let weight = DynTensor::full(&[1], 3.0, DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm2d::new(1, mean, var, Some(weight), None, 0.0).unwrap();

    assert!(bn.weight().is_some());
    assert!(bn.bias().is_none());

    let x = DynTensor::new(&[2.0], &[1, 1, 1, 1], &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // (2 - 0) / 1 * 3 = 6
    assert!((vals[0] - 6.0).abs() < 1e-5, "weight-only: expected 6, got {}", vals[0]);
}

/// Bias only (no weight): output = normalized + bias.
#[test]
fn batchnorm2d_affine_bias_only() {
    let mean = DynTensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[1], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::full(&[1], 10.0, DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm2d::new(1, mean, var, None, Some(bias), 0.0).unwrap();

    assert!(bn.weight().is_none());
    assert!(bn.bias().is_some());

    let x = DynTensor::new(&[5.0], &[1, 1, 1, 1], &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // (5 - 0) / 1 + 10 = 15
    assert!((vals[0] - 15.0).abs() < 1e-5, "bias-only: expected 15, got {}", vals[0]);
}

/// Weight + bias: full affine transform.
#[test]
fn batchnorm2d_affine_weight_and_bias() {
    let mean = DynTensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[1], DType::F32, &Device::Cpu).unwrap();
    let weight = DynTensor::full(&[1], 2.0, DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::full(&[1], 7.0, DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm2d::new(1, mean, var, Some(weight), Some(bias), 0.0).unwrap();

    assert!(bn.weight().is_some());
    assert!(bn.bias().is_some());

    let x = DynTensor::new(&[3.0], &[1, 1, 1, 1], &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // (3 - 0) / 1 * 2 + 7 = 13
    assert!((vals[0] - 13.0).abs() < 1e-5, "affine: expected 13, got {}", vals[0]);
}

/// Per-channel affine with different weight/bias per channel.
#[test]
fn batchnorm2d_affine_per_channel() {
    let mean = DynTensor::zeros(&[3], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[3], DType::F32, &Device::Cpu).unwrap();
    let weight = DynTensor::new(&[1.0, 2.0, 0.5], &[3], &Device::Cpu).unwrap();
    let bias = DynTensor::new(&[0.0, 10.0, -5.0], &[3], &Device::Cpu).unwrap();
    let bn = BatchNorm2d::new(3, mean, var, Some(weight), Some(bias), 0.0).unwrap();

    let x = DynTensor::new(&[1.0, 1.0, 1.0], &[1, 3, 1, 1], &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();

    // ch0: 1*1 + 0 = 1
    assert!((vals[0] - 1.0).abs() < 1e-5, "ch0: {}", vals[0]);
    // ch1: 1*2 + 10 = 12
    assert!((vals[1] - 12.0).abs() < 1e-5, "ch1: {}", vals[1]);
    // ch2: 1*0.5 + (-5) = -4.5
    assert!((vals[2] - (-4.5)).abs() < 1e-5, "ch2: {}", vals[2]);
}

/// NativeOpKind tracks has_weight and has_bias flags.
#[test]
fn batchnorm2d_affine_native_op_flags() {
    let with_affine = NativeOpKind::BatchNorm2d {
        eps: 1e-5,
        num_channels: 32,
        input_shape: vec![1, 32, 8, 8],
        has_weight: true,
        has_bias: true,
    };
    let without_affine = NativeOpKind::BatchNorm2d {
        eps: 1e-5,
        num_channels: 32,
        input_shape: vec![1, 32, 8, 8],
        has_weight: false,
        has_bias: false,
    };

    match with_affine {
        NativeOpKind::BatchNorm2d {
            has_weight,
            has_bias,
            ..
        } => {
            assert!(has_weight, "with_affine: has_weight should be true");
            assert!(has_bias, "with_affine: has_bias should be true");
        }
        _ => panic!("expected BatchNorm2d"),
    }
    match without_affine {
        NativeOpKind::BatchNorm2d {
            has_weight,
            has_bias,
            ..
        } => {
            assert!(!has_weight, "without_affine: has_weight should be false");
            assert!(!has_bias, "without_affine: has_bias should be false");
        }
        _ => panic!("expected BatchNorm2d"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 7. Momentum parameter in training mode
// ═══════════════════════════════════════════════════════════════════════

/// BatchNormConfig momentum defaults to 0.1 (PyTorch default).
#[test]
fn batchnorm2d_momentum_default() {
    let config = BatchNormConfig::default();
    assert!(
        (config.momentum - 0.1).abs() < 1e-10,
        "default momentum should be 0.1, got {}",
        config.momentum
    );
}

/// BatchNormConfig with_momentum builder.
#[test]
fn batchnorm2d_momentum_custom() {
    let config = BatchNormConfig::default().with_momentum(0.01);
    assert!(
        (config.momentum - 0.01).abs() < 1e-10,
        "custom momentum should be 0.01, got {}",
        config.momentum
    );
}

/// BatchNormConfig with_affine builder.
#[test]
fn batchnorm2d_config_affine_builder() {
    let config = BatchNormConfig::default().with_affine(false);
    assert!(!config.affine, "affine should be false after with_affine(false)");
}

/// BatchNormConfig with_remove_mean builder.
#[test]
fn batchnorm2d_config_remove_mean_builder() {
    let config = BatchNormConfig::default().with_remove_mean(false);
    assert!(
        !config.remove_mean,
        "remove_mean should be false after with_remove_mean(false)"
    );
}

/// BatchNormConfig chained builder preserves all settings.
#[test]
fn batchnorm2d_config_chained_builder() {
    let config = BatchNormConfig::default()
        .with_eps(1e-3)
        .with_momentum(0.05)
        .with_affine(false)
        .with_remove_mean(false);

    assert!((config.eps - 1e-3).abs() < 1e-10);
    assert!((config.momentum - 0.05).abs() < 1e-10);
    assert!(!config.affine);
    assert!(!config.remove_mean);
}

/// BatchNorm2d::with_config uses the config's eps.
#[test]
fn batchnorm2d_with_config_uses_eps() {
    let config = BatchNormConfig::new(1e-3);
    let mean = DynTensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[4], DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm2d::with_config(4, mean, var, None, None, config).unwrap();

    // Forward should succeed with the custom eps.
    let x = DynTensor::ones(&[1, 4, 2, 2], DType::F32, &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 4, 2, 2]);
}

// ═══════════════════════════════════════════════════════════════════════
// 8. Multi-batch dimension handling
// ═══════════════════════════════════════════════════════════════════════

/// Batch dimension is independent: same normalization across batches.
#[test]
fn batchnorm2d_multi_batch_independence() {
    let mean = DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap();
    let var = DynTensor::new(&[1.0, 4.0], &[2], &Device::Cpu).unwrap();
    let bn = BatchNorm2d::new(2, mean, var, None, None, 0.0).unwrap();

    // Batch of 3, 2 channels, 1x1 spatial.
    // All items have the same value per channel.
    let x = DynTensor::from_vec(
        vec![
            3.0, 4.0, // batch 0
            3.0, 4.0, // batch 1
            3.0, 4.0, // batch 2
        ],
        &[3, 2, 1, 1],
        &Device::Cpu,
    )
    .unwrap();
    let y = bn.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();

    // All batches should produce the same result.
    // ch0: (3-1)/1 = 2, ch1: (4-2)/2 = 1
    for b in 0..3 {
        let ch0 = vals[b * 2];
        let ch1 = vals[b * 2 + 1];
        assert!(
            (ch0 - 2.0).abs() < 1e-5,
            "batch {b} ch0: expected 2.0, got {ch0}"
        );
        assert!(
            (ch1 - 1.0).abs() < 1e-5,
            "batch {b} ch1: expected 1.0, got {ch1}"
        );
    }
}

/// Single batch (N=1) with larger spatial.
#[test]
fn batchnorm2d_multi_batch_n1_large_spatial() {
    let channels = 8;
    let mean = DynTensor::zeros(&[channels], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[channels], DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm2d::new(channels, mean, var, None, None, 1e-5).unwrap();

    let x = DynTensor::ones(&[1, 8, 32, 32], DType::F32, &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 8, 32, 32]);
}

/// Large batch (N=16).
#[test]
fn batchnorm2d_multi_batch_n16() {
    let channels = 4;
    let mean = DynTensor::zeros(&[channels], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[channels], DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm2d::new(channels, mean, var, None, None, 1e-5).unwrap();

    let x = DynTensor::ones(&[16, 4, 8, 8], DType::F32, &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[16, 4, 8, 8]);
    assert_eq!(y.elem_count(), 16 * 4 * 8 * 8);
}

/// Output shape matches input shape for all batch sizes.
#[test]
fn batchnorm2d_multi_batch_output_shape_preserved() {
    let mean = DynTensor::zeros(&[3], DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[3], DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm2d::new(3, mean, var, None, None, 1e-5).unwrap();

    for batch_size in [1, 2, 4, 8] {
        let x = DynTensor::ones(&[batch_size, 3, 4, 4], DType::F32, &Device::Cpu).unwrap();
        let y = bn.forward(&x).unwrap();
        assert_eq!(
            y.dims(),
            &[batch_size, 3, 4, 4],
            "output shape mismatch for batch_size={batch_size}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 9. DType compatibility
// ═══════════════════════════════════════════════════════════════════════

/// ScalarType F32 maps to "float" MSL type for BatchNorm kernel.
#[test]
fn batchnorm2d_dtype_f32_msl_mapping() {
    let (msl_type, byte_size) = crate::dtype_to_msl(DType::F32).unwrap();
    assert_eq!(msl_type, "float");
    assert_eq!(byte_size, 4);
}

/// ScalarType F16 maps to "half" MSL type.
#[test]
fn batchnorm2d_dtype_f16_msl_mapping() {
    let (msl_type, byte_size) = crate::dtype_to_msl(DType::F16).unwrap();
    assert_eq!(msl_type, "half");
    assert_eq!(byte_size, 2);
}

/// ScalarType BF16 maps to "half" on Apple GPUs.
#[test]
fn batchnorm2d_dtype_bf16_msl_mapping() {
    let (msl_type, byte_size) = crate::dtype_to_msl(DType::BF16).unwrap();
    assert_eq!(msl_type, "half");
    assert_eq!(byte_size, 2);
}

/// Integer dtypes are rejected by dtype_to_msl (no Metal scalar equivalent).
#[test]
fn batchnorm2d_dtype_integers_rejected() {
    for dtype in [DType::I32, DType::I64, DType::U32, DType::U8] {
        assert!(
            crate::dtype_to_msl(dtype).is_err(),
            "{dtype:?} should have no ScalarType equivalent"
        );
    }
}

/// F32 BatchNorm2d MSL source contains the kernel function name.
#[test]
fn batchnorm2d_dtype_f32_msl_source_valid() {
    let src = crate::dyn_tensor_metal::batch_norm_msl_source();
    assert!(
        src.contains("fused_batch_norm_float"),
        "F32 MSL should contain kernel name 'fused_batch_norm_float'"
    );
    assert!(
        src.contains("running_mean"),
        "MSL should reference running_mean buffer"
    );
    assert!(
        src.contains("running_var"),
        "MSL should reference running_var buffer"
    );
    assert!(
        src.contains("has_weight"),
        "MSL should have has_weight parameter"
    );
    assert!(
        src.contains("has_bias"),
        "MSL should have has_bias parameter"
    );
    assert!(
        src.contains("rsqrt"),
        "MSL should use rsqrt for inverse sqrt"
    );
}

/// F16 BatchNorm2d MSL source contains the half-precision kernel name.
#[test]
fn batchnorm2d_dtype_f16_msl_source_valid() {
    let src = crate::dyn_tensor_metal::batch_norm_f16_msl_source();
    assert!(
        src.contains("fused_batch_norm_half"),
        "F16 MSL should contain kernel name 'fused_batch_norm_half'"
    );
    // Accumulators should always be float for precision.
    assert!(
        src.contains("float"),
        "F16 MSL should use float accumulators"
    );
}

/// NativeOpKind::BatchNorm2d stores eps as f32 (matches GPU dispatch precision).
#[test]
fn batchnorm2d_dtype_native_op_eps_precision() {
    let op = NativeOpKind::BatchNorm2d {
        eps: 1e-5,
        num_channels: 64,
        input_shape: vec![1, 64, 32, 32],
        has_weight: true,
        has_bias: true,
    };
    match op {
        NativeOpKind::BatchNorm2d { eps, .. } => {
            assert!(
                (eps - 1e-5).abs() < 1e-10,
                "eps should be stored as f32: got {eps}"
            );
        }
        _ => panic!("expected BatchNorm2d"),
    }
}

/// ScalarType accumulator is always "float" for all scalar types (precision guarantee).
#[test]
fn batchnorm2d_dtype_accumulator_always_float() {
    for st in [ScalarType::F32, ScalarType::F16, ScalarType::BF16] {
        assert_eq!(
            st.msl_accumulator_str(),
            "float",
            "accumulator should be 'float' for {st:?}"
        );
    }
}
