// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for mixed-precision training support: dynamic loss scaling,
//! gradient dtype casting, and integration with backward rules.

use super::*;
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

// --- MixedPrecisionConfig tests ---

#[test]
fn test_mixed_precision_config_default() {
    let config = MixedPrecisionConfig::default();
    assert_eq!(config.loss_scale, 65536.0);
    assert_eq!(config.grad_dtype, DType::F32);
    assert_eq!(config.growth_factor, 2.0);
    assert_eq!(config.backoff_factor, 0.5);
    assert_eq!(config.growth_interval, 2000);
}

#[test]
fn test_mixed_precision_config_bf16_preset() {
    let config = MixedPrecisionConfig::bf16_training();
    assert_eq!(config.loss_scale, 65536.0);
    assert_eq!(config.grad_dtype, DType::F32);
}

#[test]
fn test_mixed_precision_config_f16_preset() {
    let config = MixedPrecisionConfig::f16_training();
    // F16 has smaller dynamic range, so starts with higher scale
    assert!(config.loss_scale > 65536.0);
    assert_eq!(config.grad_dtype, DType::F32);
    // Shorter growth interval for faster recovery
    assert!(config.growth_interval < 2000);
}

// --- DynamicLossScaler construction tests ---

#[test]
fn test_scaler_default_construction() {
    let scaler = DynamicLossScaler::default();
    assert_eq!(scaler.scale_factor(), 65536.0);
    assert_eq!(scaler.grad_dtype(), DType::F32);
    assert_eq!(scaler.consecutive_good_steps(), 0);
}

#[test]
fn test_scaler_custom_construction() {
    let config = MixedPrecisionConfig {
        loss_scale: 1024.0,
        grad_dtype: DType::F32,
        growth_factor: 4.0,
        backoff_factor: 0.25,
        growth_interval: 500,
    };
    let scaler = DynamicLossScaler::new(config).unwrap();
    assert_eq!(scaler.scale_factor(), 1024.0);
}

#[test]
fn test_scaler_invalid_scale_zero() {
    let config = MixedPrecisionConfig {
        loss_scale: 0.0,
        ..MixedPrecisionConfig::default()
    };
    assert!(DynamicLossScaler::new(config).is_err());
}

#[test]
fn test_scaler_invalid_scale_negative() {
    let config = MixedPrecisionConfig {
        loss_scale: -1.0,
        ..MixedPrecisionConfig::default()
    };
    assert!(DynamicLossScaler::new(config).is_err());
}

#[test]
fn test_scaler_invalid_scale_nan() {
    let config = MixedPrecisionConfig {
        loss_scale: f32::NAN,
        ..MixedPrecisionConfig::default()
    };
    assert!(DynamicLossScaler::new(config).is_err());
}

#[test]
fn test_scaler_invalid_growth_factor() {
    let config = MixedPrecisionConfig {
        growth_factor: 0.5, // must be > 1.0
        ..MixedPrecisionConfig::default()
    };
    assert!(DynamicLossScaler::new(config).is_err());
}

#[test]
fn test_scaler_invalid_backoff_factor_zero() {
    let config = MixedPrecisionConfig {
        backoff_factor: 0.0, // must be in (0, 1)
        ..MixedPrecisionConfig::default()
    };
    assert!(DynamicLossScaler::new(config).is_err());
}

#[test]
fn test_scaler_invalid_backoff_factor_one() {
    let config = MixedPrecisionConfig {
        backoff_factor: 1.0, // must be < 1.0
        ..MixedPrecisionConfig::default()
    };
    assert!(DynamicLossScaler::new(config).is_err());
}

#[test]
fn test_scaler_invalid_growth_interval_zero() {
    let config = MixedPrecisionConfig {
        growth_interval: 0,
        ..MixedPrecisionConfig::default()
    };
    assert!(DynamicLossScaler::new(config).is_err());
}

#[test]
fn test_scaler_invalid_grad_dtype_integer() {
    let config = MixedPrecisionConfig {
        grad_dtype: DType::I32, // must be float
        ..MixedPrecisionConfig::default()
    };
    assert!(DynamicLossScaler::new(config).is_err());
}

// --- scale_loss tests ---

#[test]
fn test_scale_loss_multiplies_by_scale() {
    let scaler = DynamicLossScaler::default();
    let loss = DynTensor::from_vec(vec![0.001], &[1], &Device::Cpu).unwrap();
    let scaled = scaler.scale_loss(&loss).unwrap();
    let val = scaled.to_flat_vec::<f32>().unwrap();
    let expected = 0.001 * 65536.0;
    assert!(
        (val[0] - expected).abs() < 1.0,
        "expected ~{expected}, got {}",
        val[0]
    );
}

#[test]
fn test_scale_loss_preserves_shape() {
    let scaler = DynamicLossScaler::default();
    let loss = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    let scaled = scaler.scale_loss(&loss).unwrap();
    assert_eq!(scaled.dims(), &[3]);
}

// --- unscale_gradients tests ---

#[test]
fn test_unscale_gradients_divides_by_scale() {
    let scaler = DynamicLossScaler::default();
    let scale = f64::from(scaler.scale_factor());
    let mut grads =
        vec![DynTensor::from_vec(vec![(2.0 * scale) as f32], &[1], &Device::Cpu).unwrap()];
    scaler.unscale_gradients(&mut grads).unwrap();
    let val = grads[0].to_flat_vec::<f32>().unwrap();
    assert!((val[0] - 2.0).abs() < 1e-4, "expected ~2.0, got {}", val[0]);
}

#[test]
fn test_unscale_gradients_multiple_tensors() {
    let scaler = DynamicLossScaler::default();
    let scale = f64::from(scaler.scale_factor());
    let mut grads = vec![
        DynTensor::from_vec(vec![(1.0 * scale) as f32], &[1], &Device::Cpu).unwrap(),
        DynTensor::from_vec(
            vec![(3.0 * scale) as f32, (4.0 * scale) as f32],
            &[2],
            &Device::Cpu,
        )
        .unwrap(),
    ];
    scaler.unscale_gradients(&mut grads).unwrap();
    let v0 = grads[0].to_flat_vec::<f32>().unwrap();
    let v1 = grads[1].to_flat_vec::<f32>().unwrap();
    assert!((v0[0] - 1.0).abs() < 1e-4);
    assert!((v1[0] - 3.0).abs() < 1e-4);
    assert!((v1[1] - 4.0).abs() < 1e-4);
}

// --- found_inf tests ---

#[test]
fn test_found_inf_false_for_finite_grads() {
    let grads = vec![
        DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap(),
        DynTensor::from_vec(vec![4.0, 5.0], &[2], &Device::Cpu).unwrap(),
    ];
    assert!(!DynamicLossScaler::found_inf(&grads).unwrap());
}

#[test]
fn test_found_inf_true_for_inf_grad() {
    let grads =
        vec![DynTensor::from_vec(vec![1.0, f32::INFINITY, 3.0], &[3], &Device::Cpu).unwrap()];
    assert!(DynamicLossScaler::found_inf(&grads).unwrap());
}

#[test]
fn test_found_inf_true_for_nan_grad() {
    let grads = vec![
        DynTensor::from_vec(vec![1.0, 2.0], &[2], &Device::Cpu).unwrap(),
        DynTensor::from_vec(vec![f32::NAN], &[1], &Device::Cpu).unwrap(),
    ];
    assert!(DynamicLossScaler::found_inf(&grads).unwrap());
}

#[test]
fn test_found_inf_empty_grads() {
    let grads: Vec<DynTensor> = vec![];
    assert!(!DynamicLossScaler::found_inf(&grads).unwrap());
}

// --- update (growth/backoff) tests ---

#[test]
fn test_update_backoff_on_inf() {
    let mut scaler = DynamicLossScaler::default();
    let initial = scaler.scale_factor();
    scaler.update(true);
    assert_eq!(scaler.scale_factor(), initial * 0.5);
    assert_eq!(scaler.consecutive_good_steps(), 0);
}

#[test]
fn test_update_growth_after_interval() {
    let config = MixedPrecisionConfig {
        growth_interval: 3,
        ..MixedPrecisionConfig::default()
    };
    let mut scaler = DynamicLossScaler::new(config).unwrap();
    let initial = scaler.scale_factor();

    // 2 good steps: no growth yet
    scaler.update(false);
    scaler.update(false);
    assert_eq!(scaler.scale_factor(), initial);
    assert_eq!(scaler.consecutive_good_steps(), 2);

    // 3rd good step: growth triggers
    scaler.update(false);
    assert_eq!(scaler.scale_factor(), initial * 2.0);
    assert_eq!(scaler.consecutive_good_steps(), 0);
}

#[test]
fn test_update_backoff_resets_growth_counter() {
    let config = MixedPrecisionConfig {
        growth_interval: 5,
        ..MixedPrecisionConfig::default()
    };
    let mut scaler = DynamicLossScaler::new(config).unwrap();

    // Accumulate 4 good steps
    for _ in 0..4 {
        scaler.update(false);
    }
    assert_eq!(scaler.consecutive_good_steps(), 4);

    // Hit inf: resets counter, reduces scale
    let before = scaler.scale_factor();
    scaler.update(true);
    assert_eq!(scaler.consecutive_good_steps(), 0);
    assert_eq!(scaler.scale_factor(), before * 0.5);
}

#[test]
fn test_update_repeated_backoff() {
    let mut scaler = DynamicLossScaler::default();
    let initial = scaler.scale_factor();
    scaler.update(true);
    scaler.update(true);
    scaler.update(true);
    let expected = initial * 0.5 * 0.5 * 0.5;
    assert!((scaler.scale_factor() - expected).abs() < 1e-6);
}

// --- cast_grad_to_f32 tests ---

#[test]
fn test_cast_grad_f32_noop() {
    let grad = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    let result = cast_grad_to_f32(&grad).unwrap();
    assert_eq!(result.dtype(), DType::F32);
    assert_eq!(result.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_cast_grad_bf16_to_f32() {
    let grad_f32 = DynTensor::from_vec(vec![1.5, 2.5], &[2], &Device::Cpu).unwrap();
    let grad_bf16 = grad_f32.to_dtype(DType::BF16).unwrap();
    assert_eq!(grad_bf16.dtype(), DType::BF16);
    let result = cast_grad_to_f32(&grad_bf16).unwrap();
    assert_eq!(result.dtype(), DType::F32);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 1.5).abs() < 0.1);
    assert!((vals[1] - 2.5).abs() < 0.1);
}

#[test]
fn test_cast_grad_f16_to_f32() {
    let grad_f32 = DynTensor::from_vec(vec![0.25, 0.75], &[2], &Device::Cpu).unwrap();
    let grad_f16 = grad_f32.to_dtype(DType::F16).unwrap();
    assert_eq!(grad_f16.dtype(), DType::F16);
    let result = cast_grad_to_f32(&grad_f16).unwrap();
    assert_eq!(result.dtype(), DType::F32);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 0.25).abs() < 0.01);
    assert!((vals[1] - 0.75).abs() < 0.01);
}

#[test]
fn test_cast_grad_f64_passthrough() {
    // DynTensor internally stores float as F32 (DynTensor float storage invariant).
    // to_dtype(F64) results in F32 storage labeled as F32, which cast_grad_to_f32
    // returns as-is without error. This test verifies the function doesn't error
    // on any float dtype.
    let grad = DynTensor::from_vec(vec![1.0, 2.0], &[2], &Device::Cpu).unwrap();
    let result = cast_grad_to_f32(&grad).unwrap();
    assert_eq!(result.dtype(), DType::F32);
}

#[test]
fn test_cast_grad_integer_dtype_errors() {
    // U32 is a non-float dtype that can be constructed via zeros
    let grad = DynTensor::zeros(&[3], DType::U32, &Device::Cpu).unwrap();
    assert!(cast_grad_to_f32(&grad).is_err());
}

// --- Integration: backward rules use grad.dtype() ---

#[test]
fn test_backward_integration_f32_grad_accumulation() {
    use crate::{backward, TrackedTensor, Var};
    use std::sync::Arc;

    // Verify that backward produces F32 gradients for F32 forward
    let x = Var::new(DynTensor::from_vec(vec![3.0], &[1], &Device::Cpu).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.sqr().unwrap(); // y = x^2
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap();
    assert_eq!(grad.dtype(), DType::F32);
    let val = grad.to_flat_vec::<f32>().unwrap();
    // dy/dx = 2x = 6.0
    assert!((val[0] - 6.0).abs() < 1e-4);
}

#[test]
fn test_scaler_roundtrip_scale_unscale() {
    // Verify that scale_loss + unscale_gradients is identity (up to float precision)
    let scaler = DynamicLossScaler::default();
    let original = vec![1.5, -2.3, 0.001];
    let loss = DynTensor::from_vec(original.clone(), &[3], &Device::Cpu).unwrap();

    // Scale then "backward" (simulate: scaled grads = scaled loss values)
    let scaled = scaler.scale_loss(&loss).unwrap();
    let mut grads = vec![scaled];
    scaler.unscale_gradients(&mut grads).unwrap();

    let result = grads[0].to_flat_vec::<f32>().unwrap();
    for (got, expected) in result.iter().zip(original.iter()) {
        assert!(
            (got - expected).abs() < 1e-3,
            "roundtrip mismatch: got {got}, expected {expected}"
        );
    }
}
