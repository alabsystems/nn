// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended normalization layer tests for nn-core.
//!
//! Part of #4186. Tests LayerNorm, RmsNorm, BatchNorm, GroupNorm, InstanceNorm
//! with focus on numerical properties (mean, variance), config defaults, and
//! shape preservation.

use crate::layers::{
    BatchNorm, BatchNormConfig, GroupNorm, InstanceNorm, LayerNorm, LayerNormConfig, Module,
    RmsNorm,
};
use crate::test_prng::rand_f32_vec;
use crate::{DType, Device, DynTensor};

/// Helper: create a DynTensor with deterministic pseudo-random data.
fn rand_tensor(seed: u64, dims: &[usize]) -> DynTensor {
    let numel: usize = dims.iter().product();
    let data = rand_f32_vec(seed, numel, -2.0, 2.0);
    DynTensor::from_vec(data, dims, &Device::Cpu).unwrap()
}

// ============================================================================
// LayerNorm
// ============================================================================

#[test]
fn test_layer_norm_output_shape() {
    let dim = 8;
    let weight = DynTensor::ones(&[dim], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::zeros(&[dim], DType::F32, &Device::Cpu).unwrap();
    let ln = LayerNorm::new(weight, bias, 1e-5).unwrap();

    // 2D input [batch, features]
    let x2d = rand_tensor(42, &[3, dim]);
    let y2d = ln.forward(&x2d).unwrap();
    assert_eq!(y2d.dims(), x2d.dims(), "2D output shape must match input");

    // 3D input [batch, seq, features]
    let x3d = rand_tensor(43, &[2, 5, dim]);
    let y3d = ln.forward(&x3d).unwrap();
    assert_eq!(y3d.dims(), x3d.dims(), "3D output shape must match input");
}

#[test]
fn test_layer_norm_mean_near_zero() {
    let dim = 16;
    let weight = DynTensor::ones(&[dim], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::zeros(&[dim], DType::F32, &Device::Cpu).unwrap();
    let ln = LayerNorm::new(weight, bias, 1e-5).unwrap();

    let x = rand_tensor(100, &[4, dim]);
    let y = ln.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();

    // Check mean of each row (last dim) is near zero
    for row in 0..4 {
        let start = row * dim;
        let row_vals = &vals[start..start + dim];
        let mean: f32 = row_vals.iter().sum::<f32>() / dim as f32;
        assert!(
            mean.abs() < 1e-4,
            "LayerNorm row {row} mean = {mean}, expected ~0"
        );
    }
}

#[test]
fn test_layer_norm_var_near_one() {
    let dim = 32;
    let weight = DynTensor::ones(&[dim], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::zeros(&[dim], DType::F32, &Device::Cpu).unwrap();
    let ln = LayerNorm::new(weight, bias, 1e-5).unwrap();

    let x = rand_tensor(200, &[2, dim]);
    let y = ln.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();

    for row in 0..2 {
        let start = row * dim;
        let row_vals = &vals[start..start + dim];
        let mean: f32 = row_vals.iter().sum::<f32>() / dim as f32;
        let var: f32 = row_vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
        assert!(
            (var - 1.0).abs() < 0.05,
            "LayerNorm row {row} variance = {var}, expected ~1.0"
        );
    }
}

#[test]
fn test_layer_norm_config_defaults() {
    let cfg = LayerNormConfig::default();
    assert!(
        (cfg.eps - 1e-5).abs() < 1e-12,
        "LayerNormConfig default eps should be 1e-5, got {}",
        cfg.eps
    );

    let custom = LayerNormConfig::new(1e-6);
    assert!(
        (custom.eps - 1e-6).abs() < 1e-12,
        "LayerNormConfig::new(1e-6) should set eps to 1e-6"
    );
}

#[test]
fn test_layer_norm_with_affine() {
    let dim = 4;
    let weight = DynTensor::from_vec(vec![2.0, 0.5, 3.0, 1.0], &[dim], &Device::Cpu).unwrap();
    let bias = DynTensor::from_vec(vec![1.0, -1.0, 0.0, 5.0], &[dim], &Device::Cpu).unwrap();
    let ln = LayerNorm::new(weight, bias, 0.0).unwrap();

    // Input where mean=0, var=1: [1, -1, 1, -1] (std-normalized already)
    let x = DynTensor::from_vec(vec![1.0, -1.0, 1.0, -1.0], &[1, dim], &Device::Cpu).unwrap();
    let y = ln.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();

    // Normalized input is [1, -1, 1, -1] (since mean=0, var=1)
    // After affine: [1*2+1, -1*0.5-1, 1*3+0, -1*1+5] = [3, -1.5, 3, 4]
    assert!(
        (vals[0] - 3.0).abs() < 1e-4,
        "affine[0]: expected 3.0, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - (-1.5)).abs() < 1e-4,
        "affine[1]: expected -1.5, got {}",
        vals[1]
    );
    assert!(
        (vals[2] - 3.0).abs() < 1e-4,
        "affine[2]: expected 3.0, got {}",
        vals[2]
    );
    assert!(
        (vals[3] - 4.0).abs() < 1e-4,
        "affine[3]: expected 4.0, got {}",
        vals[3]
    );
}

// ============================================================================
// RmsNorm
// ============================================================================

#[test]
fn test_rms_norm_output_shape() {
    let dim = 6;
    let weight = DynTensor::ones(&[dim], DType::F32, &Device::Cpu).unwrap();
    let rn = RmsNorm::new(weight, 1e-5).unwrap();

    let x2d = rand_tensor(300, &[3, dim]);
    let y2d = rn.forward(&x2d).unwrap();
    assert_eq!(y2d.dims(), x2d.dims(), "2D output shape must match input");

    let x3d = rand_tensor(301, &[2, 4, dim]);
    let y3d = rn.forward(&x3d).unwrap();
    assert_eq!(y3d.dims(), x3d.dims(), "3D output shape must match input");
}

#[test]
fn test_rms_norm_scale_invariance() {
    let dim = 8;
    let weight = DynTensor::ones(&[dim], DType::F32, &Device::Cpu).unwrap();
    let rn = RmsNorm::new(weight, 1e-8).unwrap();

    let x = rand_tensor(400, &[1, dim]);
    let y = rn.forward(&x).unwrap();

    // Scale input by 5.0 -- RmsNorm normalizes by RMS, so output should be identical
    let x_scaled = x.mul_scalar(5.0).unwrap();
    let y_scaled = rn.forward(&x_scaled).unwrap();

    let v1 = y.to_flat_vec::<f32>().unwrap();
    let v2 = y_scaled.to_flat_vec::<f32>().unwrap();
    for (i, (&a, &b)) in v1.iter().zip(v2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "RmsNorm scale invariance failed at [{i}]: {a} vs {b}"
        );
    }
}

#[test]
fn test_rms_norm_unit_weight() {
    let dim = 4;
    let weight = DynTensor::ones(&[dim], DType::F32, &Device::Cpu).unwrap();
    let rn = RmsNorm::new(weight, 1e-8).unwrap();

    let x = DynTensor::from_vec(vec![3.0, 4.0, 0.0, 0.0], &[1, dim], &Device::Cpu).unwrap();
    let y = rn.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();

    // RMS of [3, 4, 0, 0] = sqrt((9 + 16 + 0 + 0) / 4) = sqrt(6.25) = 2.5
    // Normalized: [3/2.5, 4/2.5, 0, 0] = [1.2, 1.6, 0, 0]
    // With unit weight, output == normalized
    assert!(
        (vals[0] - 1.2).abs() < 1e-4,
        "unit weight [0]: expected 1.2, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - 1.6).abs() < 1e-4,
        "unit weight [1]: expected 1.6, got {}",
        vals[1]
    );
    assert!(
        vals[2].abs() < 1e-4,
        "unit weight [2]: expected 0, got {}",
        vals[2]
    );
    assert!(
        vals[3].abs() < 1e-4,
        "unit weight [3]: expected 0, got {}",
        vals[3]
    );
}

// ============================================================================
// BatchNorm
// ============================================================================

#[test]
fn test_batch_norm_training_mode() {
    // BatchNorm in nn uses frozen running stats (eval mode only).
    // Verify that forward works with running stats and produces deterministic output.
    let channels = 3;
    let running_mean = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[channels], &Device::Cpu).unwrap();
    let running_var = DynTensor::from_vec(vec![1.0, 1.0, 1.0], &[channels], &Device::Cpu).unwrap();
    let bn = BatchNorm::new(running_mean, running_var, None, None, 1e-5).unwrap();

    let x =
        DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 3, 2], &Device::Cpu).unwrap();

    // First forward
    let y1 = bn.forward(&x).unwrap();
    // Second forward with same input -- must be identical (frozen stats)
    let y2 = bn.forward(&x).unwrap();

    let v1 = y1.to_flat_vec::<f32>().unwrap();
    let v2 = y2.to_flat_vec::<f32>().unwrap();
    for (i, (&a, &b)) in v1.iter().zip(v2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-8,
            "BatchNorm eval mode: output differs at [{i}]: {a} vs {b}"
        );
    }
}

#[test]
fn test_batch_norm_eval_mode() {
    // Eval mode uses running statistics. Verify known values.
    let running_mean = DynTensor::from_vec(vec![0.0, 10.0], &[2], &Device::Cpu).unwrap();
    let running_var = DynTensor::from_vec(vec![4.0, 1.0], &[2], &Device::Cpu).unwrap();
    let bn = BatchNorm::new(running_mean, running_var, None, None, 0.0).unwrap();

    // x: [B=1, C=2, T=1], values [2.0, 12.0]
    let x = DynTensor::from_vec(vec![2.0, 12.0], &[1, 2, 1], &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();

    // ch0: (2 - 0) / sqrt(4) = 1.0
    assert!(
        (vals[0] - 1.0).abs() < 1e-5,
        "eval ch0: expected 1.0, got {}",
        vals[0]
    );
    // ch1: (12 - 10) / sqrt(1) = 2.0
    assert!(
        (vals[1] - 2.0).abs() < 1e-5,
        "eval ch1: expected 2.0, got {}",
        vals[1]
    );
}

#[test]
fn test_batch_norm_config() {
    let cfg = BatchNormConfig::default();
    assert!((cfg.eps - 1e-5).abs() < 1e-12, "default eps = 1e-5");
    assert!(cfg.remove_mean, "default remove_mean = true");
    assert!(cfg.affine, "default affine = true");
    assert!((cfg.momentum - 0.1).abs() < 1e-12, "default momentum = 0.1");

    // Builder methods
    let custom = BatchNormConfig::new(1e-3)
        .with_remove_mean(false)
        .with_affine(false)
        .with_momentum(0.01);
    assert!((custom.eps - 1e-3).abs() < 1e-12);
    assert!(!custom.remove_mean);
    assert!(!custom.affine);
    assert!((custom.momentum - 0.01).abs() < 1e-12);
}

// ============================================================================
// GroupNorm
// ============================================================================

#[test]
fn test_group_norm_output_shape() {
    let channels = 8;
    let groups = 4;
    let w = DynTensor::ones(&[channels], DType::F32, &Device::Cpu).unwrap();
    let b = DynTensor::zeros(&[channels], DType::F32, &Device::Cpu).unwrap();
    let gn = GroupNorm::new(groups, channels, w, b, 1e-5).unwrap();

    // 3D input [B, C, T]
    let x3d = rand_tensor(500, &[2, channels, 10]);
    let y3d = gn.forward(&x3d).unwrap();
    assert_eq!(y3d.dims(), x3d.dims(), "3D output shape must match input");

    // 4D input [B, C, H, W]
    let x4d = rand_tensor(501, &[1, channels, 4, 4]);
    let y4d = gn.forward(&x4d).unwrap();
    assert_eq!(y4d.dims(), x4d.dims(), "4D output shape must match input");
}

#[test]
fn test_group_norm_groups_must_divide_channels() {
    let channels = 7;
    let groups = 3; // 7 not divisible by 3
    let w = DynTensor::ones(&[channels], DType::F32, &Device::Cpu).unwrap();
    let b = DynTensor::zeros(&[channels], DType::F32, &Device::Cpu).unwrap();
    let err = GroupNorm::new(groups, channels, w, b, 1e-5).unwrap_err();
    // validate_divisible returns ValueOutOfRange
    let msg = format!("{err:?}");
    assert!(
        msg.contains("ValueOutOfRange") || msg.contains("divisible"),
        "Expected divisibility error, got: {msg}"
    );
}

// ============================================================================
// InstanceNorm
// ============================================================================

#[test]
fn test_instance_norm_output_shape() {
    let norm = InstanceNorm::new(1e-5).unwrap();

    // 3D [B, C, T]
    let x3d = rand_tensor(600, &[2, 3, 16]);
    let y3d = norm.forward(&x3d).unwrap();
    assert_eq!(y3d.dims(), x3d.dims(), "3D output shape must match input");

    // 4D [B, C, H, W]
    let x4d = rand_tensor(601, &[1, 4, 8, 8]);
    let y4d = norm.forward(&x4d).unwrap();
    assert_eq!(y4d.dims(), x4d.dims(), "4D output shape must match input");
}

#[test]
fn test_instance_norm_per_channel() {
    let norm = InstanceNorm::new(1e-5).unwrap();

    // [B=1, C=2, T=6]: channel 0 has small values, channel 1 has large values
    let data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // ch0
        100.0, 200.0, 300.0, 400.0, 500.0, 600.0, // ch1
    ];
    let x = DynTensor::from_vec(data, &[1, 2, 6], &Device::Cpu).unwrap();
    let y = norm.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();

    // Each channel is normalized independently, so both channels should
    // have the same normalized pattern despite different scales
    let ch0 = &vals[0..6];
    let ch1 = &vals[6..12];

    // Mean of each channel should be ~0
    let mean0: f32 = ch0.iter().sum::<f32>() / 6.0;
    let mean1: f32 = ch1.iter().sum::<f32>() / 6.0;
    assert!(mean0.abs() < 1e-4, "ch0 mean should be ~0, got {mean0}");
    assert!(mean1.abs() < 1e-4, "ch1 mean should be ~0, got {mean1}");

    // Both channels have the same relative distribution [1..6] and [100..600],
    // so their normalized values should be identical
    for (i, (&a, &b)) in ch0.iter().zip(ch1.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "ch0[{i}]={a} should equal ch1[{i}]={b} (same relative distribution)"
        );
    }
}
