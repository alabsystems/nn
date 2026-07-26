#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for DynTensor math operations: maximum, minimum, repair_non_finite,
//! f64-to-f32 overflow guards. Activation tests (leaky_relu, snake) in submodule.

use crate::dyn_tensor::test_helpers::{cpu, t1d, t2d};
use crate::DynTensor;

// -- maximum ------------------------------------------------------------------

#[test]
fn test_maximum_same_shape() {
    let a = t1d(&[1.0, 5.0, 3.0]);
    let b = t1d(&[4.0, 2.0, 6.0]);
    let c = a.maximum(&b).unwrap();
    assert_eq!(c.to_flat_vec::<f32>().unwrap(), vec![4.0, 5.0, 6.0]);
}

#[test]
fn test_maximum_broadcast_scalar() {
    // Scalar [1] broadcast to [3]
    let a = t1d(&[1.0, 5.0, 3.0]);
    let b = DynTensor::from_vec(vec![2.5], &[1], &cpu()).unwrap();
    let c = a.maximum(&b).unwrap();
    assert_eq!(c.to_flat_vec::<f32>().unwrap(), vec![2.5, 5.0, 3.0]);
}

#[test]
fn test_maximum_broadcast_row() {
    // [1, 3] broadcast to [2, 3]
    let a = t2d(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let b = DynTensor::from_vec(vec![3.0, 3.0, 3.0], &[1, 3], &cpu()).unwrap();
    let c = a.maximum(&b).unwrap();
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![3.0, 5.0, 3.0, 4.0, 3.0, 6.0]
    );
}

#[test]
fn test_maximum_shape_mismatch_error() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[1.0, 2.0]); // incompatible shape
    let err = a.maximum(&b).unwrap_err();
    let msg = format!("{err}").to_lowercase();
    assert!(msg.contains("shape"), "expected ShapeMismatch, got: {err}");
}

#[test]
fn test_maximum_negative_values() {
    let a = t1d(&[-5.0, -1.0, 0.0]);
    let b = t1d(&[-3.0, -2.0, -1.0]);
    let c = a.maximum(&b).unwrap();
    assert_eq!(c.to_flat_vec::<f32>().unwrap(), vec![-3.0, -1.0, 0.0]);
}

// -- minimum ------------------------------------------------------------------

#[test]
fn test_minimum_same_shape() {
    let a = t1d(&[1.0, 5.0, 3.0]);
    let b = t1d(&[4.0, 2.0, 6.0]);
    let c = a.minimum(&b).unwrap();
    assert_eq!(c.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_minimum_broadcast_scalar() {
    let a = t1d(&[1.0, 5.0, 3.0]);
    let b = DynTensor::from_vec(vec![2.5], &[1], &cpu()).unwrap();
    let c = a.minimum(&b).unwrap();
    assert_eq!(c.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.5, 2.5]);
}

#[test]
fn test_minimum_broadcast_column() {
    // [2, 1] broadcast to [2, 3]
    let a = t2d(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let b = DynTensor::from_vec(vec![3.0, 5.0], &[2, 1], &cpu()).unwrap();
    let c = a.minimum(&b).unwrap();
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 3.0, 3.0, 4.0, 2.0, 5.0]
    );
}

#[test]
fn test_minimum_shape_mismatch_error() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2); // incompatible shape [3] vs [2,2]
    let err = a.minimum(&b).unwrap_err();
    let msg = format!("{err}").to_lowercase();
    assert!(msg.contains("shape"), "expected ShapeMismatch, got: {err}");
}

#[test]
fn test_minimum_negative_values() {
    let a = t1d(&[-5.0, -1.0, 0.0]);
    let b = t1d(&[-3.0, -2.0, -1.0]);
    let c = a.minimum(&b).unwrap();
    assert_eq!(c.to_flat_vec::<f32>().unwrap(), vec![-5.0, -2.0, -1.0]);
}

// -- combined -----------------------------------------------------------------

#[test]
fn test_maximum_minimum_clamp_pattern() {
    // Common ML pattern: clamp(x, lo, hi) = minimum(maximum(x, lo), hi)
    let x = t1d(&[-2.0, 0.5, 1.5, 3.0]);
    let lo = DynTensor::from_vec(vec![0.0], &[1], &cpu()).unwrap();
    let hi = DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap();
    let clamped = x.maximum(&lo).unwrap().minimum(&hi).unwrap();
    assert_eq!(
        clamped.to_flat_vec::<f32>().unwrap(),
        vec![0.0, 0.5, 1.0, 1.0]
    );
}

// -- repair_non_finite --------------------------------------------------------

#[test]
fn test_repair_non_finite_all_finite() {
    let x = t1d(&[1.0, -2.0, 3.0, 0.0]);
    let r = x.repair_non_finite(999.0).unwrap();
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), vec![1.0, -2.0, 3.0, 0.0]);
}

#[test]
fn test_repair_non_finite_nan() {
    let x = t1d(&[1.0, f32::NAN, 3.0]);
    let r = x.repair_non_finite(-100.0).unwrap();
    let vals = r.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals[0], 1.0);
    assert_eq!(vals[1], -100.0);
    assert_eq!(vals[2], 3.0);
}

#[test]
fn test_repair_non_finite_inf() {
    let x = t1d(&[f32::INFINITY, -1.0, f32::NEG_INFINITY]);
    let r = x.repair_non_finite(50.0).unwrap();
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), vec![50.0, -1.0, 50.0]);
}

#[test]
fn test_repair_non_finite_mixed() {
    // Matches NY's repair_non_finite_lower pattern: NaN/Inf → -FALLBACK
    let x = t1d(&[f32::NAN, 1.5, f32::INFINITY, -3.0, f32::NEG_INFINITY]);
    let fallback = -1e6_f64;
    let r = x.repair_non_finite(fallback).unwrap();
    let vals = r.to_flat_vec::<f32>().unwrap();
    let fallback_f32 = fallback as f32;
    assert_eq!(
        vals,
        vec![fallback_f32, 1.5, fallback_f32, -3.0, fallback_f32]
    );
}

#[test]
fn test_repair_non_finite_2d() {
    let x = DynTensor::from_vec(vec![1.0, f32::NAN, f32::INFINITY, 4.0], &[2, 2], &cpu()).unwrap();
    let r = x.repair_non_finite(0.0).unwrap();
    assert_eq!(r.dims(), &[2, 2]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), vec![1.0, 0.0, 0.0, 4.0]);
}

#[test]
fn test_repair_non_finite_empty() {
    let x = DynTensor::from_vec(vec![], &[0], &cpu()).unwrap();
    let r = x.repair_non_finite(42.0).unwrap();
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), Vec::<f32>::new());
}

#[test]
fn test_repair_non_finite_u32_dtype_error() {
    let x = DynTensor::from_vec_u32(vec![1, 2, 3], &[3], &cpu()).unwrap();
    let err = x.repair_non_finite(0.0).unwrap_err();
    let msg = format!("{err}").to_lowercase();
    assert!(
        msg.contains("data type"),
        "expected DTypeMismatch, got: {err}"
    );
}

// -- f64-to-f32 overflow guards -----------------------------------------------
// These tests verify that functions taking f64 parameters reject values that
// overflow f32 (|val| > ~3.4e38) rather than silently producing Inf.

#[test]
fn test_elu_alpha_overflow_rejects() {
    let x = t1d(&[-1.0, 0.0, 1.0]);
    let err = x.elu(1e40).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("overflows f32"),
        "expected overflow error, got: {err}"
    );
}

#[test]
fn test_elu_neg_alpha_overflow_rejects() {
    let x = t1d(&[-1.0, 0.0, 1.0]);
    let err = x.elu(-1e40).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("overflows f32"),
        "expected overflow error, got: {err}"
    );
}

#[test]
fn test_clamp_min_overflow_rejects() {
    let x = t1d(&[1.0, 2.0, 3.0]);
    let err = x.clamp(1e40, 1e41).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("overflows f32"),
        "expected overflow error, got: {err}"
    );
}

#[test]
fn test_clamp_max_overflow_rejects() {
    let x = t1d(&[1.0, 2.0, 3.0]);
    let err = x.clamp(-1.0, 1e40).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("overflows f32"),
        "expected overflow error, got: {err}"
    );
}

#[test]
fn test_clamp_min_only_overflow_rejects() {
    let x = t1d(&[1.0, 2.0, 3.0]);
    let err = x.clamp_min(1e40).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("overflows f32"),
        "expected overflow error, got: {err}"
    );
}

#[test]
fn test_clamp_max_only_overflow_rejects() {
    let x = t1d(&[1.0, 2.0, 3.0]);
    let err = x.clamp_max(-1e40).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("overflows f32"),
        "expected overflow error, got: {err}"
    );
}

#[test]
fn test_powf_exponent_overflow_rejects() {
    let x = t1d(&[1.0, 2.0]);
    let err = x.powf(1e40).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("overflows f32"),
        "expected overflow error, got: {err}"
    );
}

#[test]
fn test_powf_neg_exponent_overflow_rejects() {
    let x = t1d(&[1.0, 2.0]);
    let err = x.powf(-1e40).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("overflows f32"),
        "expected overflow error, got: {err}"
    );
}

#[test]
fn test_repair_non_finite_fallback_overflow_rejects() {
    let x = t1d(&[f32::NAN, 1.0]);
    let err = x.repair_non_finite(1e40).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("overflows f32"),
        "expected overflow error, got: {err}"
    );
}

#[test]
fn test_compare_scalar_overflow_rejects() {
    let x = t1d(&[1.0, 2.0]);
    let err = x.ge(1e40).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("overflows f32"),
        "expected overflow error, got: {err}"
    );
}

#[test]
fn test_compare_scalar_neg_overflow_rejects() {
    let x = t1d(&[1.0, 2.0]);
    let err = x.lt(-1e40).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("overflows f32"),
        "expected overflow error, got: {err}"
    );
}

// Verify normal values still work (no false positives from guard)
#[test]
fn test_elu_normal_alpha_works() {
    let x = t1d(&[-1.0, 0.0, 1.0]);
    let r = x.elu(1.0).unwrap();
    let vals = r.to_flat_vec::<f32>().unwrap();
    assert!(vals[0] < 0.0, "elu(-1, alpha=1) should be negative");
    assert_eq!(vals[1], 0.0);
    assert_eq!(vals[2], 1.0);
}

#[test]
fn test_clamp_normal_bounds_works() {
    let x = t1d(&[-5.0, 0.5, 10.0]);
    let r = x.clamp(0.0, 1.0).unwrap();
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), vec![0.0, 0.5, 1.0]);
}

#[test]
fn test_powf_normal_exponent_works() {
    let x = t1d(&[4.0, 9.0]);
    let r = x.powf(0.5).unwrap();
    let vals = r.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 2.0).abs() < 1e-5);
    assert!((vals[1] - 3.0).abs() < 1e-5);
}

// -- CPU/GPU behavioral divergence documentation tests -------------------------
// These tests assert the documented CPU/GPU behavioral asymmetry (#1492).

#[test]
fn test_recip_zero_cpu_returns_error() {
    let x = t1d(&[1.0, 0.0, 2.0]);
    let err = x.recip().unwrap_err();
    let msg = format!("{err}").to_lowercase();
    assert!(
        msg.contains("non-finite"),
        "CPU recip(0) should return non-finite error, got: {err}"
    );
}

#[test]
fn test_recip_normal_values_cpu() {
    let x = t1d(&[2.0, 4.0, 0.5]);
    let r = x.recip().unwrap();
    let vals = r.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 0.5).abs() < 1e-6);
    assert!((vals[1] - 0.25).abs() < 1e-6);
    assert!((vals[2] - 2.0).abs() < 1e-6);
}

#[test]
fn test_maximum_nan_cpu_returns_non_nan() {
    // CPU f32::max() returns the non-NaN operand when one is NaN.
    let a = t1d(&[1.0, f32::NAN, 3.0]);
    let b = t1d(&[2.0, 2.0, f32::NAN]);
    let c = a.maximum(&b).unwrap();
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals[0], 2.0);
    assert_eq!(vals[1], 2.0); // NaN vs 2.0 → 2.0
    assert_eq!(vals[2], 3.0); // 3.0 vs NaN → 3.0
}

#[test]
fn test_minimum_nan_cpu_returns_non_nan() {
    // CPU f32::min() returns the non-NaN operand when one is NaN.
    let a = t1d(&[1.0, f32::NAN, 3.0]);
    let b = t1d(&[2.0, 2.0, f32::NAN]);
    let c = a.minimum(&b).unwrap();
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals[0], 1.0);
    assert_eq!(vals[1], 2.0); // NaN vs 2.0 → 2.0
    assert_eq!(vals[2], 3.0); // 3.0 vs NaN → 3.0
}

// -- any_non_finite tests -------------------------------------------------------

#[test]
fn test_any_non_finite_all_finite() {
    let x = t1d(&[1.0, 2.0, 3.0]);
    assert!(!x.any_non_finite().unwrap());
}

#[test]
fn test_any_non_finite_with_nan() {
    let x = t1d(&[1.0, f32::NAN, 3.0]);
    assert!(x.any_non_finite().unwrap());
}

#[test]
fn test_any_non_finite_with_inf() {
    let x = t1d(&[1.0, f32::INFINITY, 3.0]);
    assert!(x.any_non_finite().unwrap());
}

#[test]
fn test_any_non_finite_with_neg_inf() {
    let x = t1d(&[f32::NEG_INFINITY, 2.0, 3.0]);
    assert!(x.any_non_finite().unwrap());
}

#[test]
fn test_any_non_finite_empty() {
    let x = DynTensor::from_vec(Vec::<f32>::new(), &[0], &cpu()).unwrap();
    assert!(!x.any_non_finite().unwrap());
}

#[test]
fn test_any_non_finite_u32_always_false() {
    let x = DynTensor::from_vec_u32(vec![1, 2, 3], &[3], &cpu()).unwrap();
    assert!(!x.any_non_finite().unwrap());
}

#[test]
fn test_any_non_finite_scalar() {
    let finite = DynTensor::from_vec(vec![42.0_f32], &[], &cpu()).unwrap();
    assert!(!finite.any_non_finite().unwrap());

    let nan = DynTensor::from_vec(vec![f32::NAN], &[], &cpu()).unwrap();
    assert!(nan.any_non_finite().unwrap());
}

// -- atan2 --------------------------------------------------------------------

#[test]
fn test_atan2_same_shape() {
    let y = t1d(&[1.0, -1.0, 0.0, 1.0]);
    let x = t1d(&[1.0, 1.0, 1.0, -1.0]);
    let r = y.atan2(&x).unwrap();
    let vals = r.to_flat_vec::<f32>().unwrap();
    let expected: Vec<f32> = [1.0_f32, -1.0, 0.0, 1.0]
        .iter()
        .zip([1.0_f32, 1.0, 1.0, -1.0].iter())
        .map(|(y, x)| y.atan2(*x))
        .collect();
    for (got, want) in vals.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
    }
}

#[test]
fn test_atan2_broadcast_scalar() {
    let y = t1d(&[1.0, 0.0, -1.0]);
    let x = DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap();
    let r = y.atan2(&x).unwrap();
    assert_eq!(r.dims(), &[3]);
    let vals = r.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
    assert!(vals[1].abs() < 1e-6);
    assert!((vals[2] + std::f32::consts::FRAC_PI_4).abs() < 1e-6);
}

#[test]
fn test_atan2_quadrants() {
    // Verify all four quadrants
    let y = t1d(&[1.0, 1.0, -1.0, -1.0]);
    let x = t1d(&[1.0, -1.0, -1.0, 1.0]);
    let r = y.atan2(&x).unwrap();
    let vals = r.to_flat_vec::<f32>().unwrap();
    // Q1: atan2(1,1) = pi/4
    assert!((vals[0] - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
    // Q2: atan2(1,-1) = 3*pi/4
    assert!((vals[1] - 3.0 * std::f32::consts::FRAC_PI_4).abs() < 1e-6);
    // Q3: atan2(-1,-1) = -3*pi/4
    assert!((vals[2] + 3.0 * std::f32::consts::FRAC_PI_4).abs() < 1e-6);
    // Q4: atan2(-1,1) = -pi/4
    assert!((vals[3] + std::f32::consts::FRAC_PI_4).abs() < 1e-6);
}

// -- round (IEEE 754 round-ties-even) -----------------------------------------
// Verifies DynTensor::round() uses f32::round_ties_even() (not f32::round()).
// PyTorch torch.round() uses IEEE 754 roundTiesToEven: .5 rounds to nearest even.

#[test]
fn test_round_ties_even_boundaries() {
    // IEEE 754 roundTiesToEven: .5 rounds to nearest EVEN integer.
    // f32::round() would round 0.5→1, 1.5→2, 2.5→3 (half away from zero).
    // f32::round_ties_even() rounds 0.5→0, 1.5→2, 2.5→2 (half to even).
    let x = t1d(&[0.5, 1.5, 2.5, 3.5, 4.5]);
    let r = x.round().unwrap();
    let vals = r.to_flat_vec::<f32>().unwrap();
    assert_eq!(
        vals,
        vec![0.0, 2.0, 2.0, 4.0, 4.0],
        "positive .5 boundaries"
    );
}

#[test]
fn test_round_ties_even_negative_boundaries() {
    // Negative .5 boundaries: -0.5→0, -1.5→-2, -2.5→-2, -3.5→-4
    let x = t1d(&[-0.5, -1.5, -2.5, -3.5, -4.5]);
    let r = x.round().unwrap();
    let vals = r.to_flat_vec::<f32>().unwrap();
    assert_eq!(
        vals,
        vec![0.0, -2.0, -2.0, -4.0, -4.0],
        "negative .5 boundaries"
    );
}

#[test]
fn test_round_non_boundary_values() {
    // Non-.5 values round normally (same for both rounding modes).
    let x = t1d(&[0.3, 1.7, -0.8, 2.1, -3.9]);
    let r = x.round().unwrap();
    let vals = r.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 2.0, -1.0, 2.0, -4.0]);
}

#[test]
fn test_round_integers_unchanged() {
    let x = t1d(&[0.0, 1.0, -1.0, 100.0, -100.0]);
    let r = x.round().unwrap();
    let vals = r.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 1.0, -1.0, 100.0, -100.0]);
}

// -- Activation-style tests (leaky_relu, snake, snake_tensor) extracted to tests_math_activation.rs --
#[path = "tests_math_activation.rs"]
mod activation_tests;

// -- New ops tests (tan, ceil, sign, softsign, reflection_pad2d, constant_pad_nd) --
#[path = "tests_new_ops.rs"]
mod new_ops_tests;
