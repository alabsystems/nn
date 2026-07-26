// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`super::validation`] helpers.

use super::*;
use crate::dyn_tensor::DynTensor;
use crate::{DType, Device, TensorError};

// -- validate_heads -----------------------------------------------------------

#[test]
fn test_validate_heads_positive_ok() {
    assert!(validate_heads(1, "test").is_ok());
    assert!(validate_heads(8, "test").is_ok());
    assert!(validate_heads(64, "test").is_ok());
}

#[test]
fn test_validate_heads_zero_rejected() {
    let err = validate_heads(0, "test").unwrap_err();
    assert!(
        matches!(err, TensorError::ValueOutOfRange { .. }),
        "expected ValueOutOfRange, got: {err:?}"
    );
}

// -- validate_eps -------------------------------------------------------------

#[test]
fn test_validate_eps_valid_ok() {
    assert!(validate_eps(1e-5, "test").is_ok());
    assert!(validate_eps(0.0, "test").is_ok());
    assert!(validate_eps(1e-12, "test").is_ok());
}

#[test]
fn test_validate_eps_negative_rejected() {
    let err = validate_eps(-1e-5, "test").unwrap_err();
    assert!(matches!(err, TensorError::ValueOutOfRange { .. }));
}

#[test]
fn test_validate_eps_nan_rejected() {
    let err = validate_eps(f64::NAN, "test").unwrap_err();
    assert!(matches!(err, TensorError::ValueOutOfRange { .. }));
}

#[test]
fn test_validate_eps_infinity_rejected() {
    let err = validate_eps(f64::INFINITY, "test").unwrap_err();
    assert!(matches!(err, TensorError::ValueOutOfRange { .. }));
}

#[test]
fn test_validate_eps_neg_infinity_rejected() {
    let err = validate_eps(f64::NEG_INFINITY, "test").unwrap_err();
    assert!(matches!(err, TensorError::ValueOutOfRange { .. }));
}

// -- validate_divisible -------------------------------------------------------

#[test]
fn test_validate_divisible_even_ok() {
    assert!(validate_divisible(12, 4, "a", "b", "test").is_ok());
    assert!(validate_divisible(1, 1, "a", "b", "test").is_ok());
    assert!(validate_divisible(0, 5, "a", "b", "test").is_ok());
}

#[test]
fn test_validate_divisible_not_divisible_rejected() {
    let err = validate_divisible(7, 3, "a", "b", "test").unwrap_err();
    assert!(matches!(err, TensorError::ValueOutOfRange { .. }));
}

// -- validate_weight_finite ---------------------------------------------------

#[test]
fn test_validate_weight_finite_clean_ok() {
    let t = DynTensor::new(&[1.0f32, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    assert!(validate_weight_finite(&t, "weight").is_ok());
}

#[test]
fn test_validate_weight_finite_nan_rejected() {
    let t = DynTensor::new(&[1.0f32, f32::NAN, 3.0], &[3], &Device::Cpu).unwrap();
    let err = validate_weight_finite(&t, "weight").unwrap_err();
    match err {
        TensorError::NonFiniteData { count, .. } => {
            assert_eq!(count, 1, "should detect exactly 1 NaN");
        }
        other => panic!("expected NonFiniteData, got: {other:?}"),
    }
}

#[test]
fn test_validate_weight_finite_inf_rejected() {
    let t = DynTensor::new(&[f32::INFINITY, f32::NEG_INFINITY, 1.0], &[3], &Device::Cpu).unwrap();
    let err = validate_weight_finite(&t, "weight").unwrap_err();
    match err {
        TensorError::NonFiniteData { count, .. } => {
            assert_eq!(count, 2, "should detect 2 non-finite values");
        }
        other => panic!("expected NonFiniteData, got: {other:?}"),
    }
}

#[test]
fn test_validate_weight_finite_zeros_ok() {
    let t = DynTensor::zeros(&[10], DType::F32, &Device::Cpu).unwrap();
    assert!(validate_weight_finite(&t, "zeros").is_ok());
}

// -- CpuRoundTrip (CPU-only, no GPU needed) -----------------------------------

#[test]
fn test_cpu_round_trip_f32_cpu_no_roundtrip() {
    let t = DynTensor::new(&[1.0f32, 2.0], &[2], &Device::Cpu).unwrap();
    let rt = CpuRoundTrip::new(&t);
    assert!(
        !rt.need_roundtrip,
        "f32 CPU tensor should not need round-trip"
    );

    let prepared = rt.prepare(&t).unwrap();
    assert_eq!(prepared.dims(), &[2]);

    let restored = rt.restore(prepared).unwrap();
    assert_eq!(restored.dims(), &[2]);
}

#[test]
fn test_cpu_round_trip_prepare_param_cpu_no_roundtrip() {
    let x = DynTensor::new(&[1.0f32], &[1], &Device::Cpu).unwrap();
    let rt = CpuRoundTrip::new(&x);

    let param = DynTensor::new(&[0.5f32, 0.5], &[2], &Device::Cpu).unwrap();
    let prepared = rt.prepare_param(&param).unwrap();
    assert_eq!(prepared.dims(), &[2]);
}
