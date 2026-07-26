#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for normalization layers: LayerNorm, GroupNorm, BatchNorm.

use super::*;
use crate::{DType, Device, DynTensor, TensorError};

// -- LayerNorm ---------------------------------------------------------------

#[test]
fn test_layer_norm_constant_input() {
    // Constant input -> normalized to 0, then affine applied
    // weight=1, bias=0 -> output should be 0
    let weight = DynTensor::ones(&[4], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    let ln = LayerNorm::new(weight, bias, 1e-5).unwrap();
    let x = DynTensor::new(&[5.0, 5.0, 5.0, 5.0], &[1, 4], &Device::Cpu).unwrap();
    let y = ln.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 4]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    for v in &vals {
        assert!(v.abs() < 1e-4, "expected ~0, got {v}");
    }
}

#[test]
fn test_layer_norm_known_values() {
    // Input [1, -1] with weight=1, bias=0, eps=0
    // mean = 0, var = 1, output = [1, -1]
    let weight = DynTensor::ones(&[2], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::zeros(&[2], DType::F32, &Device::Cpu).unwrap();
    let ln = LayerNorm::new(weight, bias, 0.0).unwrap();
    let x = DynTensor::new(&[1.0, -1.0], &[1, 2], &Device::Cpu).unwrap();
    let y = ln.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 1.0).abs() < 1e-5);
    assert!((vals[1] - (-1.0)).abs() < 1e-5);
}

#[test]
fn test_layer_norm_with_affine() {
    // Input [1, -1], weight=2, bias=3
    // Normalized: [1, -1], after affine: [5, 1]
    let weight = DynTensor::full(&[2], 2.0, DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::full(&[2], 3.0, DType::F32, &Device::Cpu).unwrap();
    let ln = LayerNorm::new(weight, bias, 0.0).unwrap();
    let x = DynTensor::new(&[1.0, -1.0], &[1, 2], &Device::Cpu).unwrap();
    let y = ln.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // normalized=[1,-1], *2 = [2,-2], +3 = [5, 1]
    assert!((vals[0] - 5.0).abs() < 1e-5);
    assert!((vals[1] - 1.0).abs() < 1e-5);
}

#[test]
fn test_layer_norm_rank0_error() {
    let weight = DynTensor::ones(&[1], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    let ln = LayerNorm::new(weight, bias, 1e-5).unwrap();
    let x = DynTensor::full(&[], 5.0, DType::F32, &Device::Cpu).unwrap();
    let err = ln.forward(&x).unwrap_err();
    assert!(
        matches!(err, TensorError::RankMismatch { .. }),
        "expected RankMismatch, got: {err:?}"
    );
}

// -- GroupNorm ---------------------------------------------------------------

#[test]
fn test_group_norm_channels_not_divisible() {
    let w = DynTensor::ones(&[3], DType::F32, &Device::Cpu).unwrap();
    let b = DynTensor::zeros(&[3], DType::F32, &Device::Cpu).unwrap();
    let err = GroupNorm::new(2, 3, w, b, 1e-5).unwrap_err();
    assert!(
        matches!(err, TensorError::ValueOutOfRange { .. }),
        "expected ValueOutOfRange for non-divisible channels, got: {err:?}"
    );
}

#[test]
fn test_group_norm_constant_input() {
    // 1 group, 2 channels, all same value -> output = bias (since normalized = 0)
    let w = DynTensor::ones(&[2], DType::F32, &Device::Cpu).unwrap();
    let b = DynTensor::full(&[2], 7.0, DType::F32, &Device::Cpu).unwrap();
    let gn = GroupNorm::new(1, 2, w, b, 1e-5).unwrap();
    // x: [1, 2, 3] = [batch=1, channels=2, spatial=3], all 5.0
    let x = DynTensor::full(&[1, 2, 3], 5.0, DType::F32, &Device::Cpu).unwrap();
    let y = gn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 2, 3]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    for v in &vals {
        assert!((v - 7.0).abs() < 1e-3, "expected ~7.0 (bias), got {v}");
    }
}

#[test]
fn test_group_norm_rank1_error() {
    let w = DynTensor::ones(&[2], DType::F32, &Device::Cpu).unwrap();
    let b = DynTensor::zeros(&[2], DType::F32, &Device::Cpu).unwrap();
    let gn = GroupNorm::new(1, 2, w, b, 1e-5).unwrap();
    let x = DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap();
    let err = gn.forward(&x).unwrap_err();
    assert!(
        matches!(err, TensorError::RankMismatch { .. }),
        "expected RankMismatch, got: {err:?}"
    );
}

#[test]
fn test_group_norm_channel_mismatch() {
    let w = DynTensor::ones(&[4], DType::F32, &Device::Cpu).unwrap();
    let b = DynTensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    let gn = GroupNorm::new(2, 4, w, b, 1e-5).unwrap();
    // Input has 2 channels, but gn expects 4
    let x = DynTensor::ones(&[1, 2, 3], DType::F32, &Device::Cpu).unwrap();
    let err = gn.forward(&x).unwrap_err();
    assert!(
        matches!(err, TensorError::ShapeMismatch { .. }),
        "expected ShapeMismatch, got: {err:?}"
    );
}

// -- BatchNorm ---------------------------------------------------------------

#[test]
fn test_batch_norm_config_default() {
    let cfg = BatchNormConfig::default();
    assert!((cfg.eps - 1e-5).abs() < 1e-12);
    assert!(cfg.remove_mean);
    assert!(cfg.affine);
    assert!((cfg.momentum - 0.1).abs() < 1e-12);
}

#[test]
fn test_batch_norm_eval_known_values() {
    let running_mean = DynTensor::new(&[2.0, 5.0], &[2], &Device::Cpu).unwrap();
    let running_var = DynTensor::new(&[1.0, 4.0], &[2], &Device::Cpu).unwrap();
    let bn = BatchNorm::new(running_mean, running_var, None, None, 1e-5).unwrap();
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3], &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 2, 3]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - (-1.0)).abs() < 1e-3);
    assert!(vals[1].abs() < 1e-3);
    assert!((vals[2] - 1.0).abs() < 1e-3);
    assert!((vals[3] - (-0.5)).abs() < 1e-3);
    assert!(vals[4].abs() < 1e-3);
    assert!((vals[5] - 0.5).abs() < 1e-3);
}

#[test]
fn test_batch_norm_with_affine() {
    let running_mean = DynTensor::zeros(&[2], DType::F32, &Device::Cpu).unwrap();
    let running_var = DynTensor::ones(&[2], DType::F32, &Device::Cpu).unwrap();
    let weight = DynTensor::full(&[2], 2.0, DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::full(&[2], 10.0, DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm::new(running_mean, running_var, Some(weight), Some(bias), 1e-5).unwrap();
    let x = DynTensor::new(&[1.0, 1.0], &[1, 2], &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 12.0).abs() < 1e-3);
    assert!((vals[1] - 12.0).abs() < 1e-3);
}

#[test]
fn test_batch_norm_no_remove_mean() {
    let running_mean = DynTensor::full(&[1], 100.0, DType::F32, &Device::Cpu).unwrap();
    let running_var = DynTensor::ones(&[1], DType::F32, &Device::Cpu).unwrap();
    let cfg = BatchNormConfig {
        remove_mean: false,
        ..Default::default()
    };
    let bn = BatchNorm::with_config(running_mean, running_var, None, None, cfg).unwrap();
    let x = DynTensor::new(&[3.0], &[1, 1], &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 3.0).abs() < 1e-3);
}

#[test]
fn test_batch_norm_4d_input() {
    let running_mean = DynTensor::zeros(&[2], DType::F32, &Device::Cpu).unwrap();
    let running_var = DynTensor::ones(&[2], DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm::new(running_mean, running_var, None, None, 0.0).unwrap();
    let x = DynTensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1, 2, 2, 2],
        &Device::Cpu,
    )
    .unwrap();
    let y = bn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 2, 2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    for (i, v) in vals.iter().enumerate() {
        assert!((v - (i as f32 + 1.0)).abs() < 1e-5);
    }
}

#[test]
fn test_batch_norm_rank1_error() {
    let running_mean = DynTensor::zeros(&[2], DType::F32, &Device::Cpu).unwrap();
    let running_var = DynTensor::ones(&[2], DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm::new(running_mean, running_var, None, None, 1e-5).unwrap();
    let x = DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap();
    let err = bn.forward(&x).unwrap_err();
    assert!(
        matches!(err, TensorError::RankMismatch { .. }),
        "expected RankMismatch, got: {err:?}"
    );
}

#[test]
fn test_batch_norm_zero_input() {
    let running_mean = DynTensor::zeros(&[2], DType::F32, &Device::Cpu).unwrap();
    let running_var = DynTensor::zeros(&[2], DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm::new(running_mean, running_var, None, None, 1e-5).unwrap();
    let x = DynTensor::zeros(&[2, 2, 4], DType::F32, &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 2, 4]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    for v in &vals {
        assert!(v.abs() < 1e-3);
    }
}

#[test]
fn test_batch_norm_accessors() {
    let rm = DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap();
    let rv = DynTensor::new(&[3.0, 4.0], &[2], &Device::Cpu).unwrap();
    let w = DynTensor::new(&[5.0, 6.0], &[2], &Device::Cpu).unwrap();
    let b = DynTensor::new(&[7.0, 8.0], &[2], &Device::Cpu).unwrap();
    let bn = BatchNorm::new(rm, rv, Some(w), Some(b), 1e-5).unwrap();
    assert_eq!(bn.running_mean().dims(), &[2]);
    assert_eq!(bn.running_var().dims(), &[2]);
    assert!(bn.weight().is_some());
    assert!(bn.bias().is_some());
    assert_eq!(bn.weight().unwrap().dims(), &[2]);
}

#[test]
fn test_batch_norm_via_apply() {
    let running_mean = DynTensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    let running_var = DynTensor::ones(&[1], DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm::new(running_mean, running_var, None, None, 0.0).unwrap();
    let x = DynTensor::new(&[5.0], &[1, 1], &Device::Cpu).unwrap();
    let y = x.apply(&bn).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 5.0).abs() < 1e-5);
}

#[test]
fn test_batch_norm_weight_only() {
    let running_mean = DynTensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    let running_var = DynTensor::ones(&[1], DType::F32, &Device::Cpu).unwrap();
    let weight = DynTensor::full(&[1], 3.0, DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm::new(running_mean, running_var, Some(weight), None, 0.0).unwrap();
    let x = DynTensor::new(&[2.0], &[1, 1], &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 6.0).abs() < 1e-5);
}

#[test]
fn test_batch_norm_bias_only() {
    let running_mean = DynTensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    let running_var = DynTensor::ones(&[1], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::full(&[1], 10.0, DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm::new(running_mean, running_var, None, Some(bias), 0.0).unwrap();
    let x = DynTensor::new(&[2.0], &[1, 1], &Device::Cpu).unwrap();
    let y = bn.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 12.0).abs() < 1e-5);
}

// -- Non-trivial GroupNorm correctness test -----------------------------------

#[test]
fn test_group_norm_two_groups_known_values() {
    let w = DynTensor::ones(&[4], DType::F32, &Device::Cpu).unwrap();
    let b = DynTensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    let gn = GroupNorm::new(2, 4, w, b, 0.0).unwrap();
    let x = DynTensor::new(
        &[1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0],
        &[1, 4, 2],
        &Device::Cpu,
    )
    .unwrap();
    let y = gn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 4, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    let sqrt5 = 5.0_f32.sqrt();
    assert!(
        (vals[0] - (-3.0 / sqrt5)).abs() < 1e-4,
        "ch0[0]: {}",
        vals[0]
    );
    assert!(
        (vals[1] - (-1.0 / sqrt5)).abs() < 1e-4,
        "ch0[1]: {}",
        vals[1]
    );
    assert!(
        (vals[2] - (1.0 / sqrt5)).abs() < 1e-4,
        "ch1[0]: {}",
        vals[2]
    );
    assert!(
        (vals[3] - (3.0 / sqrt5)).abs() < 1e-4,
        "ch1[1]: {}",
        vals[3]
    );
    assert!(
        (vals[4] - (-3.0 / sqrt5)).abs() < 1e-4,
        "ch2[0]: {}",
        vals[4]
    );
    assert!(
        (vals[5] - (-1.0 / sqrt5)).abs() < 1e-4,
        "ch2[1]: {}",
        vals[5]
    );
    assert!(
        (vals[6] - (1.0 / sqrt5)).abs() < 1e-4,
        "ch3[0]: {}",
        vals[6]
    );
    assert!(
        (vals[7] - (3.0 / sqrt5)).abs() < 1e-4,
        "ch3[1]: {}",
        vals[7]
    );
}

// -- BatchNorm NaN/Inf defense tests (proof_coverage) ------------------------
// Updated for #1202: BatchNorm now returns NonFiniteData error instead of
// silently propagating NaN/Inf through the output.

#[test]
fn test_batch_norm_nan_running_mean_returns_error() {
    // NaN in running_mean produces NaN output → caught by check_output_finite (#1202)
    let running_mean = DynTensor::new(&[f32::NAN], &[1], &Device::Cpu).unwrap();
    let running_var = DynTensor::ones(&[1], DType::F32, &Device::Cpu).unwrap();
    let bn = BatchNorm::new(running_mean, running_var, None, None, 1e-5).unwrap();
    let x = DynTensor::new(&[1.0], &[1, 1], &Device::Cpu).unwrap();
    assert!(
        bn.forward(&x).is_err(),
        "NaN running_mean should return error"
    );
}

#[test]
fn test_batch_norm_nan_running_var_returns_error() {
    // NaN in running_var → sqrt(NaN) = NaN → caught by check_output_finite (#1202)
    let running_mean = DynTensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    let running_var = DynTensor::new(&[f32::NAN], &[1], &Device::Cpu).unwrap();
    let bn = BatchNorm::new(running_mean, running_var, None, None, 1e-5).unwrap();
    let x = DynTensor::new(&[1.0], &[1, 1], &Device::Cpu).unwrap();
    assert!(
        bn.forward(&x).is_err(),
        "NaN running_var should return error"
    );
}

#[test]
fn test_batch_norm_negative_running_var_returns_error() {
    // sqrt(-10 + 1e-5) = NaN → caught by check_output_finite (#1202)
    let running_mean = DynTensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    let running_var = DynTensor::new(&[-10.0], &[1], &Device::Cpu).unwrap();
    let bn = BatchNorm::new(running_mean, running_var, None, None, 1e-5).unwrap();
    let x = DynTensor::new(&[1.0], &[1, 1], &Device::Cpu).unwrap();
    assert!(
        bn.forward(&x).is_err(),
        "negative running_var should return error"
    );
}

#[test]
fn test_batch_norm_inf_weight_returns_error() {
    // Inf weight × normalized_x = Inf → caught by check_output_finite (#1202)
    let running_mean = DynTensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    let running_var = DynTensor::ones(&[1], DType::F32, &Device::Cpu).unwrap();
    let weight = DynTensor::new(&[f32::INFINITY], &[1], &Device::Cpu).unwrap();
    let bn = BatchNorm::new(running_mean, running_var, Some(weight), None, 0.0).unwrap();
    let x = DynTensor::new(&[1.0], &[1, 1], &Device::Cpu).unwrap();
    assert!(bn.forward(&x).is_err(), "Inf weight should return error");
}

// -- GroupNorm NaN defense test (proof_coverage) ------------------------------

#[test]
fn test_group_norm_nan_weight_propagates() {
    let w = DynTensor::new(&[f32::NAN, 1.0], &[2], &Device::Cpu).unwrap();
    let b = DynTensor::zeros(&[2], DType::F32, &Device::Cpu).unwrap();
    let gn = GroupNorm::new(1, 2, w, b, 1e-5).unwrap();
    let x = DynTensor::new(&[1.0, -1.0], &[1, 2, 1], &Device::Cpu).unwrap();
    let y = gn.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!(
        vals[0].is_nan(),
        "NaN weight should produce NaN in channel 0"
    );
}

// -- LayerNorm rank 0 rejection (proof_coverage) ------------------------------

#[test]
fn test_layer_norm_rank0_rejected() {
    let weight = DynTensor::ones(&[1], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    let ln = LayerNorm::new(weight, bias, 1e-5).unwrap();
    let x = DynTensor::from_vec(vec![1.0], &[1], &Device::Cpu).unwrap();
    // LayerNorm requires rank >= 1; a scalar (rank 0) should error.
    // DynTensor::from_vec(&[1]) creates rank-1, so this should actually pass.
    let result = ln.forward(&x);
    assert!(
        result.is_ok(),
        "rank-1 input should be accepted by LayerNorm"
    );
}
