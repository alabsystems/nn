// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-variable input bounds verification tests.
//!
//! Extracted from verify_bounds.rs to keep files under 500 lines.

use nn_dsl::lower::Lowerer;
use nn_verify::{
    scalar_input_bounds, Bound, ParamBinding, VerificationResult, VerifyConfig, VerifyError,
    VerifyRequest,
};

use super::common::snake_kernel;

#[test]
fn test_multi_variable_two_inputs_add() {
    let src = "fn add(x: f32, y: f32) -> f32 { x + y }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");

    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    let result = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(&[(-1.0, 1.0), (-2.0, 2.0)])
        .verify_bounds()
        .expect("verification");

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

#[test]
fn test_multi_variable_two_inputs_mul() {
    let src = "fn mul(x: f32, y: f32) -> f32 { x * y }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");

    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    let result = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(&[(-2.0, 2.0), (-3.0, 3.0)])
        .verify_bounds()
        .expect("verification");

    assert!(result.is_finite, "bounds should be finite");
    assert!(
        result.output_lower <= -6.0 + 0.01,
        "lower should be <= -6: got {}",
        result.output_lower
    );
    assert!(
        result.output_upper >= 6.0 - 0.01,
        "upper should be >= 6: got {}",
        result.output_upper
    );
}

#[test]
fn test_multi_variable_three_inputs() {
    let src = "fn triple(x: f32, y: f32, z: f32) -> f32 { x + y + z }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");

    let bindings = vec![
        ParamBinding::Variable,
        ParamBinding::Variable,
        ParamBinding::Variable,
    ];
    let result = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(&[(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0)])
        .verify_bounds()
        .expect("verification");

    assert!(result.is_finite, "bounds should be finite");
    assert!(
        result.output_lower <= -6.0 + 0.01,
        "lower should be <= -6: got {}",
        result.output_lower
    );
    assert!(
        result.output_upper >= 6.0 - 0.01,
        "upper should be >= 6: got {}",
        result.output_upper
    );
}

#[test]
fn test_multi_variable_mixed_variable_and_constant() {
    let src = "fn scaled_add(x: f32, y: f32, scale: f32) -> f32 { scale * (x + y) }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");

    let bindings = vec![
        ParamBinding::Variable,
        ParamBinding::Variable,
        ParamBinding::Constant(2.0),
    ];
    let result = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(&[(-1.0, 1.0), (-1.0, 1.0)])
        .verify_bounds()
        .expect("verification");

    assert!(result.is_finite, "bounds should be finite");
    assert!(
        result.output_lower <= -4.0 + 0.01,
        "lower should be <= -4: got {}",
        result.output_lower
    );
    assert!(
        result.output_upper >= 4.0 - 0.01,
        "upper should be >= 4: got {}",
        result.output_upper
    );
}

#[test]
fn test_multi_variable_snake_both_variable() {
    let kernel = snake_kernel();

    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    let result = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(&[(-5.0, 5.0), (0.5, 2.0)])
        .verify_bounds()
        .expect("verification");

    assert!(result.is_finite, "bounds should be finite");
    assert!(
        result.output_lower <= -5.0 + 0.01,
        "lower should be <= -5: got {}",
        result.output_lower
    );
}

#[test]
fn test_single_variable_via_multi_api_backward_compat() {
    let kernel = snake_kernel();

    let ib = scalar_input_bounds(-10.0, 10.0).expect("input bounds");
    let single_result = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&ib)
        .verify_bounds()
        .expect("single-variable verification");

    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(1.0)];
    let multi_result = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(&[(-10.0, 10.0)])
        .verify_bounds()
        .expect("multi verification");

    assert_eq!(single_result.method, multi_result.method);
    assert!(
        (single_result.output_lower - multi_result.output_lower).abs() < 1e-6,
        "lower bounds should match: single={} multi={}",
        single_result.output_lower,
        multi_result.output_lower,
    );
    assert!(
        (single_result.output_upper - multi_result.output_upper).abs() < 1e-6,
        "upper bounds should match: single={} multi={}",
        single_result.output_upper,
        multi_result.output_upper,
    );
}

#[test]
fn test_multi_variable_escalation_parity_with_single_variable_path() {
    let kernel = snake_kernel();
    let config = VerifyConfig::with_threshold(0.0).expect("valid threshold");
    let single_input_bounds = scalar_input_bounds(-10.0, 10.0).expect("bounds");

    let single_result = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&single_input_bounds)
        .config(config.clone())
        .verify_bounds()
        .expect("single-variable verification");
    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(1.0)];
    let multi_result = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(&[(-10.0, 10.0)])
        .config(config)
        .verify_bounds()
        .expect("multi-variable verification");

    assert_eq!(single_result.method, multi_result.method);
    assert_eq!(
        single_result.crown_fallback_reason.is_some(),
        multi_result.crown_fallback_reason.is_some(),
        "fallback behavior must match between paths",
    );
    assert!(
        (single_result.output_lower - multi_result.output_lower).abs() < 1e-6,
        "lower bounds should match: single={} multi={}",
        single_result.output_lower,
        multi_result.output_lower
    );
    assert!(
        (single_result.output_upper - multi_result.output_upper).abs() < 1e-6,
        "upper bounds should match: single={} multi={}",
        single_result.output_upper,
        multi_result.output_upper
    );
}

#[test]
fn test_verify_kernel_spec_with_bindings_multi_variable_verified() {
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
        matches!(spec_v.result, VerificationResult::Verified { .. }),
        "expected Verified for add(x,y) bounds, got {:?}",
        spec_v.result
    );
}

// --- VariableBoundsMismatch edge case tests ---

#[test]
fn test_variable_bounds_mismatch_too_many_bounds() {
    let src = "fn add(x: f32, y: f32) -> f32 { x + y }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");

    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(1.0)];
    let err = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(&[(-1.0, 1.0), (-2.0, 2.0)])
        .verify_bounds()
        .expect_err("should reject mismatched bounds count");

    assert!(
        matches!(
            err,
            VerifyError::VariableBoundsMismatch {
                variable_count: 1,
                bounds_count: 2
            }
        ),
        "expected VariableBoundsMismatch(1, 2), got: {err}"
    );
}

#[test]
fn test_variable_bounds_mismatch_too_few_bounds() {
    let src = "fn add(x: f32, y: f32) -> f32 { x + y }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");

    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    let err = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(&[(-1.0, 1.0)])
        .verify_bounds()
        .expect_err("should reject mismatched bounds count");

    assert!(
        matches!(
            err,
            VerifyError::VariableBoundsMismatch {
                variable_count: 2,
                bounds_count: 1
            }
        ),
        "expected VariableBoundsMismatch(2, 1), got: {err}"
    );
}

#[test]
fn test_no_variable_bindings_empty_bounds_rejected() {
    let src = "fn add(x: f32, y: f32) -> f32 { x + y }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");

    let bindings = vec![ParamBinding::Constant(1.0), ParamBinding::Constant(2.0)];
    let err = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(&[])
        .verify_bounds()
        .expect_err("all-constant bindings with no variable bounds should error");

    assert!(matches!(err, VerifyError::NoVariableBindings));
}

#[test]
fn test_no_variable_bindings_with_bounds_rejected() {
    let src = "fn add(x: f32, y: f32) -> f32 { x + y }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");

    let bindings = vec![ParamBinding::Constant(1.0), ParamBinding::Constant(2.0)];
    let err = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(&[(-1.0, 1.0)])
        .verify_bounds()
        .expect_err("should reject bounds when no variables exist");

    assert!(matches!(err, VerifyError::NoVariableBindings));
}

#[test]
fn test_spec_with_bindings_mismatch_rejected() {
    let src = "fn add(x: f32, y: f32) -> f32 { x + y }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");

    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(1.0)];
    let output_spec = vec![Bound::new(-5.0, 5.0)];

    let err = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(&[(-1.0, 1.0), (-2.0, 2.0)])
        .required_output_bounds(&output_spec)
        .verify_spec()
        .expect_err("should reject mismatched bounds in spec path");

    assert!(
        matches!(
            err,
            VerifyError::VariableBoundsMismatch {
                variable_count: 1,
                bounds_count: 2
            }
        ),
        "expected VariableBoundsMismatch(1, 2) in spec path, got: {err}"
    );
}

#[test]
fn test_spec_with_bindings_zero_variables_rejected() {
    let src = "fn add(x: f32, y: f32) -> f32 { x + y }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");

    let bindings = vec![ParamBinding::Constant(1.0), ParamBinding::Constant(2.0)];
    let output_spec = vec![Bound::new(3.0, 3.0)];

    let err = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(&[])
        .required_output_bounds(&output_spec)
        .verify_spec()
        .expect_err("should reject all-constant spec verification path");

    assert!(matches!(err, VerifyError::NoVariableBindings));
}

#[test]
fn test_spec_with_bindings_tight_property_unknown() {
    let src = "fn add(x: f32, y: f32) -> f32 { x + y }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");

    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    let tight_spec = vec![Bound::new(-2.0, 2.0)];
    let config = VerifyConfig::with_threshold(0.0).expect("valid threshold");

    let spec_v = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(&[(-1.0, 1.0), (-2.0, 2.0)])
        .required_output_bounds(&tight_spec)
        .config(config)
        .verify_spec()
        .expect("spec verification should return a result");

    assert!(
        matches!(spec_v.result, VerificationResult::Unknown { .. }),
        "tight spec should be Unknown for add(x,y) with wide input bounds, got {:?}",
        spec_v.result
    );
}

// --- Default bindings spec wrapper test ---

#[test]
fn test_verify_kernel_spec_with_bindings_default_config() {
    let src = "fn add(x: f32, y: f32) -> f32 { x + y }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");

    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    let output_spec = vec![Bound::new(-3.0, 3.0)];

    let spec_v = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(&[(-1.0, 1.0), (-2.0, 2.0)])
        .required_output_bounds(&output_spec)
        .verify_spec()
        .expect("default-config spec verification");

    assert!(
        matches!(spec_v.result, VerificationResult::Verified { .. }),
        "expected Verified for add(x,y) with default config, got {:?}",
        spec_v.result
    );
}
