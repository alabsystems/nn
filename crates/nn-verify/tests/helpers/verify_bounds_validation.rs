// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Validation and edge-case tests for kernel bounds verification.
//!
//! Covers: VerifyConfig threshold validation, NaN/Inf input bounds rejection,
//! and SpecVerification provenance.
//!
//! Extracted from verify_bounds.rs to keep files under 500 lines.

use nn_dsl::lower::Lowerer;
use nn_dsl::snake_scalar_bounds;
use nn_verify::{
    multi_scalar_input_bounds, scalar_input_bounds, Bound, ParamBinding, PropMethod,
    VerificationResult, VerifyConfig, VerifyError, VerifyRequest,
};

use super::common::snake_kernel;

// --- VerifyConfig threshold validation tests ---

#[test]
fn test_verify_config_rejects_nan_threshold() {
    let err = VerifyConfig::with_threshold(f32::NAN).expect_err("NaN should be rejected");
    assert!(
        matches!(err, VerifyError::InvalidThreshold { .. }),
        "expected InvalidThreshold for NaN, got: {err}"
    );
}

#[test]
fn test_verify_config_rejects_negative_threshold() {
    let err = VerifyConfig::with_threshold(-1.0).expect_err("negative should be rejected");
    assert!(
        matches!(err, VerifyError::InvalidThreshold { value } if value == -1.0),
        "expected InvalidThreshold for -1.0, got: {err}"
    );
}

#[test]
fn test_verify_config_rejects_infinite_threshold() {
    let err = VerifyConfig::with_threshold(f32::INFINITY).expect_err("infinity should be rejected");
    assert!(
        matches!(err, VerifyError::InvalidThreshold { .. }),
        "expected InvalidThreshold for infinity, got: {err}"
    );
}

#[test]
fn test_verify_config_rejects_neg_infinite_threshold() {
    let err = VerifyConfig::with_threshold(f32::NEG_INFINITY)
        .expect_err("neg infinity should be rejected");
    assert!(
        matches!(err, VerifyError::InvalidThreshold { .. }),
        "expected InvalidThreshold for neg infinity, got: {err}"
    );
}

#[test]
fn test_verify_config_accepts_zero_threshold() {
    let config = VerifyConfig::with_threshold(0.0).expect("zero is valid");
    assert_eq!(config.escalation_threshold(), 0.0);
}

// --- NaN/Inf input bounds rejection tests (#66) ---

#[test]
fn test_scalar_input_bounds_rejects_nan_lower() {
    let err = scalar_input_bounds(f32::NAN, 1.0).expect_err("NaN lower should be rejected");
    assert!(
        matches!(err, VerifyError::InvalidInputBounds { .. }),
        "expected InvalidInputBounds for NaN lower, got: {err}"
    );
}

#[test]
fn test_scalar_input_bounds_rejects_nan_upper() {
    let err = scalar_input_bounds(1.0, f32::NAN).expect_err("NaN upper should be rejected");
    assert!(
        matches!(err, VerifyError::InvalidInputBounds { .. }),
        "expected InvalidInputBounds for NaN upper, got: {err}"
    );
}

#[test]
fn test_scalar_input_bounds_rejects_both_nan() {
    let err = scalar_input_bounds(f32::NAN, f32::NAN).expect_err("both NaN should be rejected");
    assert!(
        matches!(err, VerifyError::InvalidInputBounds { .. }),
        "expected InvalidInputBounds for both NaN, got: {err}"
    );
}

#[test]
fn test_scalar_input_bounds_rejects_inf_lower() {
    let err = scalar_input_bounds(f32::INFINITY, 1.0).expect_err("Inf lower should be rejected");
    assert!(
        matches!(err, VerifyError::InvalidInputBounds { .. }),
        "expected InvalidInputBounds for Inf lower, got: {err}"
    );
}

#[test]
fn test_scalar_input_bounds_rejects_neg_inf_upper() {
    let err = scalar_input_bounds(-1.0, f32::NEG_INFINITY)
        .expect_err("NEG_INFINITY upper should be rejected");
    assert!(
        matches!(err, VerifyError::InvalidInputBounds { .. }),
        "expected InvalidInputBounds for NEG_INFINITY upper, got: {err}"
    );
}

#[test]
fn test_multi_scalar_input_bounds_rejects_nan_lower() {
    let err = multi_scalar_input_bounds(&[(f32::NAN, 1.0)])
        .expect_err("NaN lower in multi bounds should be rejected");
    assert!(
        matches!(err, VerifyError::InvalidInputBounds { .. }),
        "expected InvalidInputBounds for NaN lower in multi bounds, got: {err}"
    );
}

#[test]
fn test_multi_scalar_input_bounds_rejects_nan_upper_non_first() {
    let err = multi_scalar_input_bounds(&[(-1.0, 1.0), (0.0, f32::NAN)])
        .expect_err("NaN upper in non-first pair should be rejected");
    assert!(
        matches!(err, VerifyError::InvalidInputBounds { .. }),
        "expected InvalidInputBounds for NaN upper in non-first pair, got: {err}"
    );
}

#[test]
fn test_multi_scalar_input_bounds_rejects_inf() {
    let err = multi_scalar_input_bounds(&[(-1.0, 1.0), (0.0, f32::INFINITY)])
        .expect_err("Inf in multi bounds should be rejected");
    assert!(
        matches!(err, VerifyError::InvalidInputBounds { .. }),
        "expected InvalidInputBounds for Inf in multi bounds, got: {err}"
    );
}

#[test]
fn test_multi_scalar_input_bounds_rejects_inverted() {
    // Inverted bounds (lower > upper) should be rejected, even when both are finite.
    let err =
        multi_scalar_input_bounds(&[(5.0, 1.0)]).expect_err("inverted bounds should be rejected");
    assert!(
        matches!(
            err,
            VerifyError::InvalidInputBounds {
                lower: 5.0,
                upper: 1.0
            }
        ),
        "expected InvalidInputBounds for inverted bounds, got: {err}"
    );
}

#[test]
fn test_multi_scalar_input_bounds_rejects_inverted_non_first() {
    // Inverted bounds in non-first pair should also be rejected.
    let err = multi_scalar_input_bounds(&[(-1.0, 1.0), (3.0, -3.0)])
        .expect_err("inverted bounds in second pair should be rejected");
    assert!(
        matches!(
            err,
            VerifyError::InvalidInputBounds {
                lower: 3.0,
                upper: -3.0
            }
        ),
        "expected InvalidInputBounds for inverted second pair, got: {err}"
    );
}

#[test]
fn test_scalar_input_bounds_accepts_valid_finite() {
    scalar_input_bounds(-10.0, 10.0).expect("valid finite bounds should be accepted");
    scalar_input_bounds(0.0, 0.0).expect("equal bounds should be accepted");
    scalar_input_bounds(-1e30, 1e30).expect("large finite bounds should be accepted");
}

// --- SpecVerification provenance tests (#52) ---

#[test]
fn test_spec_verification_ibp_only_provenance() {
    let kernel = snake_kernel();
    let input_bounds = scalar_input_bounds(-10.0, 10.0).expect("input bounds");
    let (out_lower, out_upper) =
        snake_scalar_bounds(-10.0, 10.0, 0.01, 100.0).expect("finite bounds");
    let output_spec = vec![Bound::new(out_lower, out_upper)];
    let config = VerifyConfig::default();

    let spec_v = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&input_bounds)
        .required_output_bounds(&output_spec)
        .config(config)
        .verify_spec()
        .expect("spec verification should succeed");

    assert_eq!(
        spec_v.method,
        PropMethod::Ibp,
        "default threshold should use IBP only"
    );
    assert!(
        spec_v.crown_fallback_reason.is_none(),
        "CROWN should not have been attempted with default threshold"
    );
}

#[test]
fn test_spec_verification_crown_escalation_provenance() {
    let kernel = snake_kernel();
    let input_bounds = scalar_input_bounds(-10.0, 10.0).expect("input bounds");
    let tight_spec = vec![Bound::new(-10.0, 10.0)];
    let config = VerifyConfig::with_threshold(0.0).expect("valid threshold");

    let spec_v = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&input_bounds)
        .required_output_bounds(&tight_spec)
        .config(config)
        .verify_spec()
        .expect("spec verification should return a result");

    assert_eq!(
        spec_v.method,
        PropMethod::Crown,
        "tight spec + threshold=0 should escalate to CROWN"
    );
    assert!(
        spec_v.crown_fallback_reason.is_none(),
        "CROWN should not have errored, got fallback: {:?}",
        spec_v.crown_fallback_reason
    );
}

#[test]
fn test_spec_verification_bindings_provenance() {
    let src = "fn add(x: f32, y: f32) -> f32 { x + y }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");

    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    let output_spec = vec![Bound::new(-3.0, 3.0)];
    let config = VerifyConfig::with_threshold(0.0).expect("valid threshold");

    let spec_v = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(&[(-1.0, 1.0), (-2.0, 2.0)])
        .required_output_bounds(&output_spec)
        .config(config)
        .verify_spec()
        .expect("spec verification");

    assert!(
        matches!(spec_v.method, PropMethod::Ibp | PropMethod::Crown),
        "method should be Ibp or Crown, got {:?}",
        spec_v.method
    );
    assert!(
        matches!(spec_v.result, VerificationResult::Verified { .. }),
        "expected Verified, got {:?}",
        spec_v.result
    );
}
