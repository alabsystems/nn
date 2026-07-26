// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `VerifyRequest` builder.
//!
//! Covers both error paths (missing required fields) and happy paths
//! (single-variable and multi-variable bounds + spec verification).

use nn_dsl::lower::Lowerer;
use nn_verify::{
    scalar_input_bounds, Bound, ParamBinding, VerificationResult, VerifyConfig, VerifyError,
    VerifyRequest,
};

use super::common::snake_kernel;

// ---------------------------------------------------------------------------
// Error paths: missing required fields
// ---------------------------------------------------------------------------

#[test]
fn test_verify_bounds_missing_input_bounds() {
    let kernel = snake_kernel();
    let err = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        // no input_bounds set
        .verify_bounds()
        .expect_err("should require input_bounds for single-variable path");
    assert!(
        matches!(err, VerifyError::InvalidInput(ref msg) if msg.contains("input_bounds")),
        "expected InvalidInput about input_bounds, got: {err}"
    );
}

#[test]
fn test_verify_bounds_bindings_without_variable_bounds() {
    let src = "fn add(x: f32, y: f32) -> f32 { x + y }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");

    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    let err = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        // no variable_bounds set
        .verify_bounds()
        .expect_err("should require variable_bounds when bindings are set");
    assert!(
        matches!(err, VerifyError::InvalidInput(ref msg) if msg.contains("variable_bounds")),
        "expected InvalidInput about variable_bounds, got: {err}"
    );
}

#[test]
fn test_verify_spec_missing_output_bounds() {
    let kernel = snake_kernel();
    let ib = scalar_input_bounds(-1.0, 1.0).expect("bounds");
    let err = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&ib)
        // no required_output_bounds set
        .verify_spec()
        .expect_err("should require required_output_bounds for spec verification");
    assert!(
        matches!(err, VerifyError::InvalidInput(ref msg) if msg.contains("required_output_bounds")),
        "expected InvalidInput about required_output_bounds, got: {err}"
    );
}

#[test]
fn test_verify_spec_bindings_without_variable_bounds() {
    let src = "fn add(x: f32, y: f32) -> f32 { x + y }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");

    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    let required = [Bound::new(-10.0, 10.0)];
    let err = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .required_output_bounds(&required)
        // no variable_bounds set
        .verify_spec()
        .expect_err("should require variable_bounds when bindings are set");
    assert!(
        matches!(err, VerifyError::InvalidInput(ref msg) if msg.contains("variable_bounds")),
        "expected InvalidInput about variable_bounds, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Happy paths: single-variable bounds verification
// ---------------------------------------------------------------------------

#[test]
fn test_verify_bounds_single_variable_snake() {
    let kernel = snake_kernel();
    let ib = scalar_input_bounds(-1.0, 1.0).expect("bounds");
    let result = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&ib)
        .verify_bounds()
        .expect("single-variable bounds verification should succeed");

    assert_eq!(result.kernel_name, "snake");
    assert!(result.is_finite, "Snake output bounds should be finite");
}

#[test]
fn test_verify_bounds_single_variable_with_config() {
    let kernel = snake_kernel();
    let ib = scalar_input_bounds(-1.0, 1.0).expect("bounds");
    let config = VerifyConfig::with_threshold(100.0).expect("config");
    let result = VerifyRequest::new(&kernel)
        .config(config)
        .constant_params(&[1.0])
        .input_bounds(&ib)
        .verify_bounds()
        .expect("single-variable bounds with config should succeed");

    assert!(result.is_finite, "bounds should be finite");
}

// ---------------------------------------------------------------------------
// Happy paths: multi-variable bounds verification
// ---------------------------------------------------------------------------

#[test]
fn test_verify_bounds_multi_variable_add() {
    let src = "fn add(x: f32, y: f32) -> f32 { x + y }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");

    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    let vb = [(-1.0, 1.0), (-2.0, 2.0)];
    let result = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(&vb)
        .verify_bounds()
        .expect("multi-variable bounds verification should succeed");

    assert!(result.is_finite, "bounds should be finite");
    assert!(
        result.output_lower <= -3.0 + 0.01,
        "lower should be <= -3: got {}",
        result.output_lower
    );
    assert!(
        result.output_upper >= 3.0 - 0.01,
        "upper should be >= 3: got {}",
        result.output_upper
    );
}

// ---------------------------------------------------------------------------
// Happy paths: spec verification
// ---------------------------------------------------------------------------

#[test]
fn test_verify_spec_single_variable_snake() {
    let kernel = snake_kernel();
    let ib = scalar_input_bounds(-1.0, 1.0).expect("bounds");
    let required = [Bound::new(-10.0, 10.0)];
    let result = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&ib)
        .required_output_bounds(&required)
        .verify_spec()
        .expect("spec verification should succeed");

    assert!(
        matches!(result.result, VerificationResult::Verified { .. }),
        "Snake [-1,1] → [-10,10] should verify, got: {:?}",
        result.result
    );
}

#[test]
fn test_verify_spec_multi_variable() {
    let src = "fn add(x: f32, y: f32) -> f32 { x + y }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");

    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    let vb = [(-1.0, 1.0), (-2.0, 2.0)];
    let required = [Bound::new(-10.0, 10.0)];
    let result = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(&vb)
        .required_output_bounds(&required)
        .verify_spec()
        .expect("multi-variable spec verification should succeed");

    assert!(
        matches!(result.result, VerificationResult::Verified { .. }),
        "add([-1,1], [-2,2]) → [-10,10] should verify, got: {:?}",
        result.result
    );
}

// ---------------------------------------------------------------------------
// Builder equivalence: verify_bounds matches legacy API
// ---------------------------------------------------------------------------

#[test]
fn test_builder_scalar_bounds_snake() {
    let kernel = snake_kernel();
    let ib = scalar_input_bounds(-1.0, 1.0).expect("bounds");

    let result = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&ib)
        .verify_bounds()
        .expect("builder");

    assert_eq!(result.kernel_name, "snake");
    assert!(result.is_finite, "Snake output bounds should be finite");
    assert!(
        result.output_lower >= -2.0,
        "lower bound too loose: {}",
        result.output_lower
    );
    assert!(
        result.output_upper <= 3.0,
        "upper bound too loose: {}",
        result.output_upper
    );
}

// ---------------------------------------------------------------------------
// Error paths: validation failures through the builder (#187)
// ---------------------------------------------------------------------------

#[test]
fn test_verify_spec_single_variable_missing_input_bounds() {
    let kernel = snake_kernel();
    let required = [Bound::new(-10.0, 10.0)];
    let err = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .required_output_bounds(&required)
        // no input_bounds set
        .verify_spec()
        .expect_err("should require input_bounds for single-variable spec path");
    assert!(
        matches!(err, VerifyError::InvalidInput(ref msg) if msg.contains("input_bounds")),
        "expected InvalidInput about input_bounds, got: {err}"
    );
}

#[test]
fn test_verify_bounds_variable_bounds_count_mismatch() {
    let src = "fn add(x: f32, y: f32) -> f32 { x + y }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");

    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    // 2 Variable bindings but only 1 bounds pair → mismatch
    let vb = [(-1.0, 1.0)];
    let err = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(&vb)
        .verify_bounds()
        .expect_err("should reject mismatched variable bounds count");
    assert!(
        matches!(err, VerifyError::VariableBoundsMismatch { .. }),
        "expected VariableBoundsMismatch, got: {err:?}"
    );
}

#[test]
fn test_verify_bounds_all_constant_bindings_rejected() {
    let src = "fn add(x: f32, y: f32) -> f32 { x + y }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");

    // All bindings are Constant → NoVariableBindings error
    let bindings = [ParamBinding::Constant(1.0), ParamBinding::Constant(2.0)];
    let vb: &[(f32, f32)] = &[];
    let err = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(vb)
        .verify_bounds()
        .expect_err("should reject all-constant bindings");
    assert!(
        matches!(err, VerifyError::NoVariableBindings),
        "expected NoVariableBindings, got: {err:?}"
    );
}

// ---------------------------------------------------------------------------
// require_sound = true: sound result passes, no error (#187 AC3)
// ---------------------------------------------------------------------------

#[test]
fn test_verify_spec_require_sound_with_sound_result_succeeds() {
    // Current scalar kernels always produce Sound provenance.
    // This test verifies that require_sound=true does NOT reject sound results.
    let kernel = snake_kernel();
    let ib = scalar_input_bounds(-1.0, 1.0).expect("bounds");
    let required = [Bound::new(-10.0, 10.0)];
    let config = VerifyConfig::with_threshold(1e6)
        .expect("config")
        .with_require_sound(true);
    let result = VerifyRequest::new(&kernel)
        .config(config)
        .constant_params(&[1.0])
        .input_bounds(&ib)
        .required_output_bounds(&required)
        .verify_spec()
        .expect("require_sound=true should pass when result is Sound");

    assert!(
        matches!(result.result, VerificationResult::Verified { .. }),
        "Snake [-1,1] → [-10,10] should verify, got: {:?}",
        result.result
    );
}

#[test]
fn test_verify_spec_require_sound_false_default_succeeds() {
    // Confirm the default config has require_sound=false and passes.
    let kernel = snake_kernel();
    let ib = scalar_input_bounds(-1.0, 1.0).expect("bounds");
    let required = [Bound::new(-10.0, 10.0)];
    let result = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&ib)
        .required_output_bounds(&required)
        .verify_spec()
        .expect("default config (require_sound=false) should succeed");

    assert!(
        matches!(result.result, VerificationResult::Verified { .. }),
        "expected Verified, got: {:?}",
        result.result
    );
}
