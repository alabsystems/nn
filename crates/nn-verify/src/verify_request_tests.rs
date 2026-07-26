// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `verify_request.rs` — builder validation and error paths.

use super::*;
use crate::error::VerifyError;
use ny_api::{Bound, BoundedTensor};
use nn_dsl::test_kernels::{parse_kernel, square_kernel};
use ndarray::{ArrayD, IxDyn};

/// Build a simple BoundedTensor for scalar input bounds.
fn scalar_bt(lower: f32, upper: f32) -> BoundedTensor {
    let lo = ArrayD::from_elem(IxDyn(&[1]), lower);
    let hi = ArrayD::from_elem(IxDyn(&[1]), upper);
    BoundedTensor::new(lo, hi).unwrap()
}

// ---------------------------------------------------------------------------
// VerifyRequest::new — builder defaults
// ---------------------------------------------------------------------------

#[test]
fn test_new_creates_default_config() {
    let kernel = square_kernel();
    let req = VerifyRequest::new(&kernel);
    // Verify builder is created with defaults (no panics).
    assert!(req.bindings.is_none());
    assert!(req.constant_params.is_none());
    assert!(req.input_bounds.is_none());
    assert!(req.variable_bounds.is_none());
    assert!(req.required_output_bounds.is_none());
}

// ---------------------------------------------------------------------------
// verify_bounds — missing input_bounds (single-variable path)
// ---------------------------------------------------------------------------

#[test]
fn test_verify_bounds_missing_input_bounds() {
    let kernel = square_kernel();
    let err = VerifyRequest::new(&kernel).verify_bounds().unwrap_err();
    assert!(matches!(err, VerifyError::InvalidInput(ref msg) if msg.contains("input_bounds")));
}

// ---------------------------------------------------------------------------
// verify_bounds — missing variable_bounds with bindings (multi-variable path)
// ---------------------------------------------------------------------------

#[test]
fn test_verify_bounds_missing_variable_bounds_with_bindings() {
    let kernel = parse_kernel("fn f(x: f32, y: f32) -> f32 { x + y }");
    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    let err = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .verify_bounds()
        .unwrap_err();
    assert!(matches!(err, VerifyError::InvalidInput(ref msg) if msg.contains("variable_bounds")));
}

// ---------------------------------------------------------------------------
// verify_spec — missing required_output_bounds
// ---------------------------------------------------------------------------

#[test]
fn test_verify_spec_missing_required_output_bounds() {
    let kernel = square_kernel();
    let bt = scalar_bt(-1.0, 1.0);
    let err = VerifyRequest::new(&kernel)
        .input_bounds(&bt)
        .verify_spec()
        .unwrap_err();
    assert!(
        matches!(err, VerifyError::InvalidInput(ref msg) if msg.contains("required_output_bounds"))
    );
}

// ---------------------------------------------------------------------------
// verify_spec — missing variable_bounds with bindings
// ---------------------------------------------------------------------------

#[test]
fn test_verify_spec_missing_variable_bounds_with_bindings() {
    let kernel = parse_kernel("fn f(x: f32, y: f32) -> f32 { x + y }");
    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    let output_bound = Bound::try_new(-2.0, 2.0).unwrap();
    let err = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .required_output_bounds(&[output_bound])
        .verify_spec()
        .unwrap_err();
    assert!(matches!(err, VerifyError::InvalidInput(ref msg) if msg.contains("variable_bounds")));
}

// ---------------------------------------------------------------------------
// resolve_constant_params — multi-param kernel without constants
// ---------------------------------------------------------------------------

#[test]
fn test_resolve_constant_params_multi_param_missing_constants() {
    // Snake has 2 params — single-variable path requires 1 constant.
    let kernel = parse_kernel("fn f(x: f32, alpha: f32) -> f32 { x + alpha }");
    let bt = scalar_bt(-1.0, 1.0);
    let err = VerifyRequest::new(&kernel)
        .input_bounds(&bt)
        .verify_bounds()
        .unwrap_err();
    assert!(matches!(err, VerifyError::InvalidInput(ref msg) if msg.contains("constant_params")));
}

#[test]
fn test_resolve_constant_params_single_param_ok_without_constants() {
    // Single-param kernel should succeed without constant_params.
    // (This exercises resolve_constant_params returning Ok(&[]).)
    let kernel = parse_kernel("fn f(x: f32) -> f32 { x + 1.0 }");
    let bt = scalar_bt(-1.0, 1.0);
    // This should NOT fail at the resolve_constant_params stage.
    // It may fail later in graph translation or propagation, but
    // we're testing that the builder validation passes.
    let result = VerifyRequest::new(&kernel)
        .input_bounds(&bt)
        .verify_bounds();
    // Should succeed or fail at a later stage — not at resolve_constant_params.
    match result {
        Ok(_) => {} // fine
        Err(VerifyError::InvalidInput(msg)) => {
            panic!("should not get InvalidInput for single-param kernel: {msg}");
        }
        Err(_) => {} // downstream errors are acceptable
    }
}

// ---------------------------------------------------------------------------
// Builder chaining
// ---------------------------------------------------------------------------

#[test]
fn test_builder_config_chaining() {
    let kernel = square_kernel();
    let config = VerifyConfig::with_threshold(10.0)
        .unwrap()
        .with_require_sound(true);
    let bt = scalar_bt(-1.0, 1.0);
    // Verify chaining compiles and doesn't panic.
    let result = VerifyRequest::new(&kernel)
        .config(config)
        .input_bounds(&bt)
        .verify_bounds();
    // Should produce a verification result (not a builder validation error).
    assert!(
        !matches!(result, Err(VerifyError::InvalidInput(_))),
        "config chaining should not produce InvalidInput"
    );
}

// ---------------------------------------------------------------------------
// verify_bounds — variable bounds mismatch
// ---------------------------------------------------------------------------

#[test]
fn test_verify_bounds_variable_bounds_count_mismatch() {
    let kernel = parse_kernel("fn f(x: f32, y: f32) -> f32 { x + y }");
    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    // Provide only 1 bound for 2 variables.
    let bounds = [(0.0, 1.0)];
    let err = VerifyRequest::new(&kernel)
        .bindings(&bindings)
        .variable_bounds(&bounds)
        .verify_bounds()
        .unwrap_err();
    assert!(matches!(
        err,
        VerifyError::VariableBoundsMismatch {
            variable_count: 2,
            bounds_count: 1
        }
    ));
}
