// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Numerical correctness tests for normalization bounds functions.
//!
//! Per design doc line ~283: "Every `prove_bounds` analytical function needs both
//! error-path AND numerical correctness tests." These tests verify that the
//! computed output values match hand-calculated expected results for:
//! - rms_norm_scalar_output_bounds
//! - norm_affine_output_bounds (shared by layer_norm_scalar, instance_norm_affine_scalar)
//! - adain_output_bounds
//! - instance_norm_output_bounds

use super::*;

const TOL: f64 = 1e-6;

fn assert_close(actual: f64, expected: f64, msg: &str) {
    crate::bounds::assert_close_f64(actual, expected, TOL, msg);
}

// ============================================================================
// RMSNorm scalar output bounds (AC6)
// ============================================================================

#[test]
fn test_rms_norm_positive_coeff() {
    // output = x * rms_inv * weight
    // rms_inv = 2.0, weight = 0.5 → coeff = 1.0
    // x in [-3, 3] → output in [-3, 3]
    let (lo, hi) = rms_norm_scalar_output_bounds(2.0, 0.5, -3.0, 3.0).unwrap();
    assert_close(lo, -3.0, "rms_norm lo coeff=1");
    assert_close(hi, 3.0, "rms_norm hi coeff=1");
}

#[test]
fn test_rms_norm_negative_coeff() {
    // rms_inv = -1.0, weight = 2.0 → coeff = -2.0
    // x in [1, 4] → output = x * (-2) → in [-8, -2]
    let (lo, hi) = rms_norm_scalar_output_bounds(-1.0, 2.0, 1.0, 4.0).unwrap();
    assert_close(lo, -8.0, "rms_norm lo coeff=-2");
    assert_close(hi, -2.0, "rms_norm hi coeff=-2");
}

#[test]
fn test_rms_norm_zero_weight() {
    // rms_inv = 5.0, weight = 0.0 → coeff = 0.0
    // x in [-100, 100] → output = 0
    let (lo, hi) = rms_norm_scalar_output_bounds(5.0, 0.0, -100.0, 100.0).unwrap();
    assert_eq!(lo, 0.0, "rms_norm lo zero_weight");
    assert_eq!(hi, 0.0, "rms_norm hi zero_weight");
}

// ============================================================================
// norm_affine output bounds (AC7) — shared by LayerNorm/InstanceNormAffine
// ============================================================================

#[test]
fn test_norm_affine_identity_transform() {
    // mean=0, var=1, eps=0, gamma=1, beta=0
    // output = (x - 0) * 1/sqrt(1) * 1 + 0 = x
    // x in [-2, 5] → output in [-2, 5]
    let (lo, hi) = norm_affine_output_bounds(0.0, 1.0, 0.0, 1.0, 0.0, -2.0, 5.0).unwrap();
    assert_close(lo, -2.0, "norm_affine lo identity");
    assert_close(hi, 5.0, "norm_affine hi identity");
}

#[test]
fn test_norm_affine_with_shift_and_scale() {
    // mean=1.0, var=4.0, eps=0.0, gamma=2.0, beta=3.0
    // inv_std = 1/sqrt(4) = 0.5
    // slope = 0.5 * 2 = 1.0
    // intercept = -1.0 * 0.5 * 2.0 + 3.0 = -1.0 + 3.0 = 2.0
    // output = 1.0 * x + 2.0
    // x in [0, 10] → output in [2, 12]
    let (lo, hi) = norm_affine_output_bounds(1.0, 4.0, 0.0, 2.0, 3.0, 0.0, 10.0).unwrap();
    assert_close(lo, 2.0, "norm_affine lo scale+shift");
    assert_close(hi, 12.0, "norm_affine hi scale+shift");
}

#[test]
fn test_norm_affine_negative_gamma() {
    // mean=0, var=1, eps=0, gamma=-1, beta=0
    // slope = 1.0 * (-1) = -1.0, intercept = 0
    // output = -x → x in [1, 5] → output in [-5, -1]
    let (lo, hi) = norm_affine_output_bounds(0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 5.0).unwrap();
    assert_close(lo, -5.0, "norm_affine lo neg_gamma");
    assert_close(hi, -1.0, "norm_affine hi neg_gamma");
}

#[test]
fn test_norm_affine_with_eps() {
    // mean=0, var=0, eps=4.0, gamma=1, beta=0
    // inv_std = 1/sqrt(0 + 4) = 0.5
    // slope = 0.5, intercept = 0
    // x in [-4, 6] → output in [-2, 3]
    let (lo, hi) = norm_affine_output_bounds(0.0, 0.0, 4.0, 1.0, 0.0, -4.0, 6.0).unwrap();
    assert_close(lo, -2.0, "norm_affine lo with_eps");
    assert_close(hi, 3.0, "norm_affine hi with_eps");
}

// ============================================================================
// AdaIN output bounds (AC8)
// ============================================================================

#[test]
fn test_adain_identity() {
    // mu=0, var=1, gamma=1, beta=0, eps=0
    // output = 1 * (x - 0) / sqrt(1) + 0 = x
    // x in [-3, 7] → output in [-3, 7]
    let (lo, hi) = adain_output_bounds(0.0, 1.0, 1.0, 0.0, 0.0, -3.0, 7.0).unwrap();
    assert_close(lo, -3.0, "adain lo identity");
    assert_close(hi, 7.0, "adain hi identity");
}

#[test]
fn test_adain_scale_shift() {
    // mu=2.0, var=9.0, gamma=3.0, beta=1.0, eps=0.0
    // inv_std = 1/sqrt(9) = 1/3
    // slope = 3 * (1/3) = 1.0
    // intercept = -2 * 3 * (1/3) + 1 = -2 + 1 = -1.0
    // output = x - 1.0
    // x in [0, 10] → output in [-1, 9]
    let (lo, hi) = adain_output_bounds(2.0, 9.0, 3.0, 1.0, 0.0, 0.0, 10.0).unwrap();
    assert_close(lo, -1.0, "adain lo scale_shift");
    assert_close(hi, 9.0, "adain hi scale_shift");
}

#[test]
fn test_adain_negative_gamma() {
    // mu=0, var=4, gamma=-2, beta=5, eps=0
    // inv_std = 1/sqrt(4) = 0.5
    // slope = -2 * 0.5 = -1.0
    // intercept = -0 * (-2) * 0.5 + 5 = 5.0
    // output = -x + 5
    // x in [0, 10] → output at x=0: 5, at x=10: -5 → in [-5, 5]
    let (lo, hi) = adain_output_bounds(0.0, 4.0, -2.0, 5.0, 0.0, 0.0, 10.0).unwrap();
    assert_close(lo, -5.0, "adain lo neg_gamma");
    assert_close(hi, 5.0, "adain hi neg_gamma");
}

#[test]
fn test_adain_with_production_eps() {
    // mu=0, var=1, gamma=1, beta=0, eps=1e-5 (standard production value)
    // denom = var + eps = 1.00001
    // inv_std = 1/sqrt(1.00001) ≈ 0.999995
    // output ≈ x (very close to identity)
    // x in [-3, 7] → output ≈ [-3, 7]
    let (lo, hi) = adain_output_bounds(0.0, 1.0, 1.0, 0.0, 1e-5, -3.0, 7.0).unwrap();
    assert!((lo - (-3.0)).abs() < 0.001, "adain lo with eps=1e-5: {lo}");
    assert!((hi - 7.0).abs() < 0.001, "adain hi with eps=1e-5: {hi}");
}

#[test]
fn test_adain_eps_dominated() {
    // mu=0, var=0, gamma=1, beta=0, eps=4.0 (eps dominates)
    // denom = var + eps = 4.0
    // inv_std = 1/sqrt(4) = 0.5
    // output = 0.5 * x
    // x in [-6, 10] → output in [-3, 5]
    let (lo, hi) = adain_output_bounds(0.0, 0.0, 1.0, 0.0, 4.0, -6.0, 10.0).unwrap();
    assert_close(lo, -3.0, "adain lo eps_dominated");
    assert_close(hi, 5.0, "adain hi eps_dominated");
}

// ============================================================================
// AdaIN / InstanceNorm denom<=0 error-path tests
// ============================================================================

#[test]
fn test_adain_denom_negative_returns_error() {
    // var=-1.0, eps=0.5 → denom = -0.5, guard fires
    let err = adain_output_bounds(0.0, -1.0, 1.0, 0.0, 0.5, -1.0, 1.0);
    assert!(err.is_err(), "adain should reject denom <= 0");
}

#[test]
fn test_adain_denom_zero_returns_error() {
    // var=-2.0, eps=2.0 → denom = 0.0, guard fires
    let err = adain_output_bounds(0.0, -2.0, 1.0, 0.0, 2.0, -1.0, 1.0);
    assert!(err.is_err(), "adain should reject denom == 0");
}

#[test]
fn test_norm_affine_denom_negative_returns_error() {
    // mean=0, var=-2, eps=1, gamma=1, beta=0 → denom = -1.0
    let err = norm_affine_output_bounds(0.0, -2.0, 1.0, 1.0, 0.0, -1.0, 1.0);
    assert!(err.is_err(), "norm_affine should reject denom <= 0");
}

#[test]
fn test_instance_norm_denom_negative_returns_error() {
    // var=-1.0, eps=0.5 → denom = -0.5
    let err = instance_norm_output_bounds(0.0, -1.0, 0.5, -1.0, 1.0);
    assert!(err.is_err(), "instance_norm should reject denom <= 0");
}

#[test]
fn test_instance_norm_denom_zero_returns_error() {
    // var=-3.0, eps=3.0 → denom = 0.0
    let err = instance_norm_output_bounds(0.0, -3.0, 3.0, -1.0, 1.0);
    assert!(err.is_err(), "instance_norm should reject denom == 0");
}

// ============================================================================
// InstanceNorm output bounds (AC9)
// ============================================================================

#[test]
fn test_instance_norm_unit_variance() {
    // mean=0, var=1, eps=0
    // inv_std = 1/sqrt(1) = 1.0
    // output = (x - 0) * 1 = x
    // x in [-5, 5] → output in [-5, 5]
    let (lo, hi) = instance_norm_output_bounds(0.0, 1.0, 0.0, -5.0, 5.0).unwrap();
    assert_close(lo, -5.0, "instnorm lo unit_var");
    assert_close(hi, 5.0, "instnorm hi unit_var");
}

#[test]
fn test_instance_norm_with_mean_shift() {
    // mean=3.0, var=4.0, eps=0.0
    // inv_std = 1/sqrt(4) = 0.5
    // slope = 0.5
    // intercept = -3.0 * 0.5 = -1.5
    // output = 0.5 * x - 1.5
    // x in [0, 10] → output at x=0: -1.5, at x=10: 3.5
    let (lo, hi) = instance_norm_output_bounds(3.0, 4.0, 0.0, 0.0, 10.0).unwrap();
    assert_close(lo, -1.5, "instnorm lo mean_shift");
    assert_close(hi, 3.5, "instnorm hi mean_shift");
}

#[test]
fn test_instance_norm_small_variance_large_eps() {
    // mean=0, var=0, eps=16.0
    // inv_std = 1/sqrt(16) = 0.25
    // output = 0.25 * x
    // x in [-8, 8] → output in [-2, 2]
    let (lo, hi) = instance_norm_output_bounds(0.0, 0.0, 16.0, -8.0, 8.0).unwrap();
    assert_close(lo, -2.0, "instnorm lo large_eps");
    assert_close(hi, 2.0, "instnorm hi large_eps");
}

// ============================================================================
// Error-path tests: NaN/Inf constant params and input bounds
// ============================================================================

// --- RMSNorm NaN/Inf constant param rejection ---

#[test]
fn test_rms_norm_nan_rms_inv_rejected() {
    // rms_inv_const (index 1) is NaN → NonFiniteConstantParam
    assert!(rms_norm_scalar_output_bounds(f32::NAN, 1.0, -1.0, 1.0).is_err());
}

#[test]
fn test_rms_norm_inf_rms_inv_rejected() {
    assert!(rms_norm_scalar_output_bounds(f32::INFINITY, 1.0, -1.0, 1.0).is_err());
}

#[test]
fn test_rms_norm_nan_weight_rejected() {
    // weight_const (index 2) is NaN → NonFiniteConstantParam
    assert!(rms_norm_scalar_output_bounds(1.0, f32::NAN, -1.0, 1.0).is_err());
}

#[test]
fn test_rms_norm_inf_weight_rejected() {
    assert!(rms_norm_scalar_output_bounds(1.0, f32::INFINITY, -1.0, 1.0).is_err());
}

#[test]
fn test_rms_norm_nan_lower_rejected() {
    assert!(rms_norm_scalar_output_bounds(1.0, 1.0, f64::NAN, 1.0).is_err());
}

#[test]
fn test_rms_norm_inf_upper_rejected() {
    assert!(rms_norm_scalar_output_bounds(1.0, 1.0, -1.0, f64::INFINITY).is_err());
}

#[test]
fn test_rms_norm_inverted_bounds_rejected() {
    assert!(rms_norm_scalar_output_bounds(1.0, 1.0, 5.0, -5.0).is_err());
}

// --- norm_affine NaN/Inf constant param rejection ---

#[test]
fn test_norm_affine_nan_mean_rejected() {
    assert!(norm_affine_output_bounds(f32::NAN, 1.0, 0.0, 1.0, 0.0, -1.0, 1.0).is_err());
}

#[test]
fn test_norm_affine_nan_gamma_rejected() {
    assert!(norm_affine_output_bounds(0.0, 1.0, 0.0, f32::NAN, 0.0, -1.0, 1.0).is_err());
}

#[test]
fn test_norm_affine_inf_beta_rejected() {
    assert!(norm_affine_output_bounds(0.0, 1.0, 0.0, 1.0, f32::INFINITY, -1.0, 1.0).is_err());
}

#[test]
fn test_norm_affine_nan_lower_rejected() {
    assert!(norm_affine_output_bounds(0.0, 1.0, 0.0, 1.0, 0.0, f64::NAN, 1.0).is_err());
}

#[test]
fn test_norm_affine_inverted_bounds_rejected() {
    assert!(norm_affine_output_bounds(0.0, 1.0, 0.0, 1.0, 0.0, 5.0, -5.0).is_err());
}

// --- AdaIN NaN/Inf constant param rejection ---

#[test]
fn test_adain_nan_mu_rejected() {
    assert!(adain_output_bounds(f32::NAN, 1.0, 1.0, 0.0, 0.0, -1.0, 1.0).is_err());
}

#[test]
fn test_adain_nan_gamma_rejected() {
    assert!(adain_output_bounds(0.0, 1.0, f32::NAN, 0.0, 0.0, -1.0, 1.0).is_err());
}

#[test]
fn test_adain_inf_beta_rejected() {
    assert!(adain_output_bounds(0.0, 1.0, 1.0, f32::INFINITY, 0.0, -1.0, 1.0).is_err());
}

#[test]
fn test_adain_nan_lower_rejected() {
    assert!(adain_output_bounds(0.0, 1.0, 1.0, 0.0, 0.0, f64::NAN, 1.0).is_err());
}

#[test]
fn test_adain_inverted_bounds_rejected() {
    assert!(adain_output_bounds(0.0, 1.0, 1.0, 0.0, 0.0, 5.0, -5.0).is_err());
}

// --- InstanceNorm NaN/Inf constant param rejection ---

#[test]
fn test_instance_norm_nan_mean_rejected() {
    assert!(instance_norm_output_bounds(f32::NAN, 1.0, 0.0, -1.0, 1.0).is_err());
}

#[test]
fn test_instance_norm_inf_mean_rejected() {
    assert!(instance_norm_output_bounds(f32::INFINITY, 1.0, 0.0, -1.0, 1.0).is_err());
}

#[test]
fn test_instance_norm_nan_lower_rejected() {
    assert!(instance_norm_output_bounds(0.0, 1.0, 0.0, f64::NAN, 1.0).is_err());
}

#[test]
fn test_instance_norm_inverted_bounds_rejected() {
    assert!(instance_norm_output_bounds(0.0, 1.0, 0.0, 5.0, -5.0).is_err());
}
