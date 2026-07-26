#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for DynTensor random constructors (training feature).

use crate::dyn_tensor::test_helpers::cpu;
use crate::dyn_tensor::DynTensor;
use crate::DType;

#[test]
fn test_randn_basic_shape() {
    let t = DynTensor::randn(0.0, 1.0, &[2, 3], &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(t.numel(), 6);
}

#[test]
fn test_randn_1d() {
    let t = DynTensor::randn(0.0, 1.0, &[100], &cpu()).unwrap();
    assert_eq!(t.dims(), &[100]);
}

#[test]
fn test_randn_empty() {
    let t = DynTensor::randn(0.0, 1.0, &[0], &cpu()).unwrap();
    assert_eq!(t.dims(), &[0]);
    assert_eq!(t.numel(), 0);
}

#[test]
fn test_randn_statistical_properties() {
    // 10k samples should have mean ~0 and std ~1
    let t = DynTensor::randn(0.0, 1.0, &[10000], &cpu()).unwrap();
    let data = t.to_flat_vec::<f32>().unwrap();

    let mean: f64 = data.iter().map(|&x| f64::from(x)).sum::<f64>() / data.len() as f64;
    let variance: f64 = data
        .iter()
        .map(|&x| {
            let d = f64::from(x) - mean;
            d * d
        })
        .sum::<f64>()
        / data.len() as f64;
    let std_dev = variance.sqrt();

    // Mean should be within 0.05 of 0 with high probability
    assert!(mean.abs() < 0.05, "mean {mean} too far from 0");
    // Std dev should be within 0.05 of 1
    assert!(
        (std_dev - 1.0).abs() < 0.05,
        "std_dev {std_dev} too far from 1"
    );
}

#[test]
fn test_randn_values_differ() {
    let t1 = DynTensor::randn(0.0, 1.0, &[10], &cpu()).unwrap();
    let t2 = DynTensor::randn(0.0, 1.0, &[10], &cpu()).unwrap();
    let d1 = t1.to_flat_vec::<f32>().unwrap();
    let d2 = t2.to_flat_vec::<f32>().unwrap();
    // Extremely unlikely that two random tensors are identical
    assert_ne!(d1, d2, "two randn calls should produce different values");
}

#[test]
fn test_randn_all_finite() {
    let t = DynTensor::randn(0.0, 1.0, &[1000], &cpu()).unwrap();
    let data = t.to_flat_vec::<f32>().unwrap();
    for &v in &data {
        assert!(v.is_finite(), "randn produced non-finite: {v}");
    }
}

#[test]
fn test_randn_custom_mean_std() {
    // N(5.0, 0.1) — mean ~5.0, std ~0.1
    let t = DynTensor::randn(5.0, 0.1, &[10000], &cpu()).unwrap();
    let data = t.to_flat_vec::<f32>().unwrap();

    let mean: f64 = data.iter().map(|&x| f64::from(x)).sum::<f64>() / data.len() as f64;
    let variance: f64 = data
        .iter()
        .map(|&x| {
            let d = f64::from(x) - mean;
            d * d
        })
        .sum::<f64>()
        / data.len() as f64;
    let std_dev = variance.sqrt();

    assert!((mean - 5.0).abs() < 0.05, "mean {mean} too far from 5.0");
    assert!(
        (std_dev - 0.1).abs() < 0.01,
        "std_dev {std_dev} too far from 0.1"
    );
}

#[test]
fn test_rand_like_basic() {
    let reference = DynTensor::zeros(&[3, 4], DType::F32, &cpu()).unwrap();
    let t = reference.rand_like().unwrap();
    assert_eq!(t.dims(), &[3, 4]);
    assert_eq!(t.dtype(), DType::F32);
    // rand_like should produce non-zero values (extremely unlikely all zeros)
    let data = t.to_flat_vec::<f32>().unwrap();
    let any_nonzero = data.iter().any(|&x| x != 0.0);
    assert!(any_nonzero, "rand_like should produce non-zero values");
}

#[test]
fn test_rand_like_preserves_shape() {
    let reference = DynTensor::ones(&[2, 5, 3], DType::F32, &cpu()).unwrap();
    let t = reference.rand_like().unwrap();
    assert_eq!(t.dims(), reference.dims());
}

#[test]
fn test_randn_high_dimensional() {
    let t = DynTensor::randn(0.0, 1.0, &[2, 3, 4, 5], &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 3, 4, 5]);
    assert_eq!(t.numel(), 120);
}

// -- rand (uniform) tests -----------------------------------------------------

#[test]
fn test_rand_basic_shape() {
    let t = DynTensor::rand(0.0, 1.0, &[2, 3], &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(t.elem_count(), 6);
}

#[test]
fn test_rand_values_in_unit_interval() {
    let t = DynTensor::rand(0.0, 1.0, &[10000], &cpu()).unwrap();
    let data = t.to_flat_vec::<f32>().unwrap();
    for &v in &data {
        assert!((0.0..1.0).contains(&v), "rand value {v} outside [0, 1)");
    }
}

#[test]
fn test_rand_statistical_properties() {
    // 10k uniform [0, 1) samples: mean ≈ 0.5, std ≈ 1/sqrt(12) ≈ 0.2887
    let t = DynTensor::rand(0.0, 1.0, &[10000], &cpu()).unwrap();
    let data = t.to_flat_vec::<f32>().unwrap();
    let mean: f64 = data.iter().map(|&x| f64::from(x)).sum::<f64>() / data.len() as f64;
    assert!(
        (mean - 0.5).abs() < 0.02,
        "uniform mean {mean} too far from 0.5"
    );
}

#[test]
fn test_rand_custom_range() {
    // U(-1.0, 1.0): all values in [-1.0, 1.0), mean ≈ 0.0
    let t = DynTensor::rand(-1.0, 1.0, &[10000], &cpu()).unwrap();
    let data = t.to_flat_vec::<f32>().unwrap();
    for &v in &data {
        assert!((-1.0..1.0).contains(&v), "rand value {v} outside [-1, 1)");
    }
    let mean: f64 = data.iter().map(|&x| f64::from(x)).sum::<f64>() / data.len() as f64;
    assert!(mean.abs() < 0.05, "U(-1,1) mean {mean} too far from 0");
}

#[test]
fn test_rand_like_values_in_unit_interval() {
    let reference = DynTensor::zeros(&[1000], DType::F32, &cpu()).unwrap();
    let t = reference.rand_like().unwrap();
    let data = t.to_flat_vec::<f32>().unwrap();
    for &v in &data {
        assert!((0.0..1.0).contains(&v), "rand_like value {v} outside [0, 1)");
    }
}

// -- randn_like tests ---------------------------------------------------------

#[test]
fn test_randn_like_basic() {
    let reference = DynTensor::zeros(&[3, 4], DType::F32, &cpu()).unwrap();
    let t = reference.randn_like().unwrap();
    assert_eq!(t.dims(), &[3, 4]);
    assert_eq!(t.dtype(), DType::F32);
}

#[test]
fn test_randn_like_statistical_properties() {
    // randn_like delegates to randn(0, 1, ...) — verify distribution independently.
    // 10k samples: mean ~0, std ~1 (same thresholds as randn test).
    let reference = DynTensor::zeros(&[10000], DType::F32, &cpu()).unwrap();
    let t = reference.randn_like().unwrap();
    let data = t.to_flat_vec::<f32>().unwrap();

    let mean: f64 = data.iter().map(|&x| f64::from(x)).sum::<f64>() / data.len() as f64;
    let variance: f64 = data
        .iter()
        .map(|&x| {
            let d = f64::from(x) - mean;
            d * d
        })
        .sum::<f64>()
        / data.len() as f64;
    let std_dev = variance.sqrt();

    assert!(mean.abs() < 0.05, "randn_like mean {mean} too far from 0");
    assert!(
        (std_dev - 1.0).abs() < 0.05,
        "randn_like std_dev {std_dev} too far from 1.0"
    );
    // Also verify negative count (original check)
    let negatives = data.iter().filter(|&&x| x < 0.0).count();
    assert!(
        negatives > 4000 && negatives < 6000,
        "randn_like should produce ~50% negatives, got {negatives}"
    );
}

// -- f64-to-f32 overflow guards -----------------------------------------------

#[test]
fn test_rand_f64_overflow_lo_rejects() {
    let result = DynTensor::rand(1e40, 1e41, &[2], &cpu());
    assert!(
        result.is_err(),
        "rand with f64 bounds overflowing f32 must fail"
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("overflows f32"),
        "error should mention overflow: {msg}"
    );
}

#[test]
fn test_rand_f64_overflow_hi_rejects() {
    let result = DynTensor::rand(0.0, 1e40, &[2], &cpu());
    assert!(result.is_err(), "rand with hi overflowing f32 must fail");
}

#[test]
fn test_rand_f64_neg_overflow_lo_rejects() {
    let result = DynTensor::rand(-1e40, 0.0, &[2], &cpu());
    assert!(
        result.is_err(),
        "rand with negative f64 lo overflowing f32 must fail"
    );
}

#[test]
fn test_rand_f64_normal_bounds_ok() {
    // Normal f64 values that fit in f32 should succeed
    let result = DynTensor::rand(-100.0, 100.0, &[2], &cpu());
    assert!(result.is_ok());
}

#[test]
fn test_randn_f64_overflow_mean_rejects() {
    let result = DynTensor::randn(1e40, 1.0, &[2], &cpu());
    assert!(result.is_err(), "randn with mean overflowing f32 must fail");
}

#[test]
fn test_randn_f64_overflow_std_rejects() {
    let result = DynTensor::randn(0.0, 1e40, &[2], &cpu());
    assert!(result.is_err(), "randn with std overflowing f32 must fail");
}

#[test]
fn test_randn_f64_neg_overflow_mean_rejects() {
    let result = DynTensor::randn(-1e40, 1.0, &[2], &cpu());
    assert!(
        result.is_err(),
        "randn with negative mean overflowing f32 must fail"
    );
}

#[test]
fn test_randn_f64_normal_params_ok() {
    let result = DynTensor::randn(5.0, 0.1, &[2], &cpu());
    assert!(result.is_ok());
}

// -- full() f64-to-f32 overflow guard -----------------------------------------

#[test]
fn test_full_f64_overflow_rejects() {
    // 1e40 overflows f32 to Inf
    let result = DynTensor::full(&[2, 3], 1e40, DType::F32, &cpu());
    assert!(
        result.is_err(),
        "full() with f64 value overflowing f32 must fail"
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("overflows f32"),
        "error should mention overflow: {msg}"
    );
}

#[test]
fn test_full_f64_neg_overflow_rejects() {
    let result = DynTensor::full(&[2], -1e40, DType::F32, &cpu());
    assert!(
        result.is_err(),
        "full() with negative f64 overflowing f32 must fail"
    );
}

#[test]
fn test_full_f64_normal_value_ok() {
    let result = DynTensor::full(&[2, 3], 3.14, DType::F32, &cpu());
    assert!(result.is_ok());
    let t = result.unwrap();
    let data = t.to_flat_vec::<f32>().unwrap();
    assert!((data[0] - 3.14_f32).abs() < 1e-6);
}

#[test]
fn test_full_nan_allowed() {
    // NaN as f64 stays NaN as f32 — both are non-finite, so guard doesn't trigger
    let result = DynTensor::full(&[2], f64::NAN, DType::F32, &cpu());
    assert!(
        result.is_ok(),
        "NaN should be allowed (NaN->NaN, not overflow)"
    );
}

#[test]
fn test_full_inf_allowed() {
    // Inf as f64 stays Inf as f32 — both are non-finite, so guard doesn't trigger
    let result = DynTensor::full(&[2], f64::INFINITY, DType::F32, &cpu());
    assert!(
        result.is_ok(),
        "Inf should be allowed (Inf->Inf, not overflow)"
    );
}

// -- elem_count tests ---------------------------------------------------------

#[test]
fn test_elem_count_matches_numel() {
    let t = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    assert_eq!(t.elem_count(), t.numel());
    assert_eq!(t.elem_count(), 24);
}
