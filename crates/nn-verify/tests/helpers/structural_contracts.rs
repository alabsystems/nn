// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Structural contract tests: verify function signatures and error paths
//! survive merge reverts.
//!
//! These tests guard against the regression-of-fix pattern (#580): a merge
//! silently reverts a `Result`-returning function to a panicking version.
//! Unlike behavioral tests (which pass with either version), these tests
//! ONLY pass when the function returns `Result` and its error paths are live.
//!
//! ## Categories
//!
//! 1. **Compile-time signature guards**: Force the compiler to check that
//!    public API functions return `Result`. A merge revert → compile error.
//! 2. **Error-path guards**: Call functions with invalid inputs (NaN, Inf,
//!    empty shapes) and assert `Err`. A merge revert to panic → test panics.
//! 3. **Boundary constant-fold guards**: Verify Ge/Le equality boundary
//!    through the full public API path.
//!
//! Part of #580 (regression-of-fix pattern)

use nn_dsl::ir::CompareOpKind;
use nn_verify::{
    compose_sequential, kernel_to_graph, kernel_to_graph_multi, ParamBinding, SequentialSpec,
    VerifyError,
};

use super::common;

// ===================================================================
// Category 1: Compile-time signature guards
//
// Each test binds the return type to `Result<_, VerifyError>`.
// If a merge reverts any function from Result to bare value, these
// fail to compile — catching the revert at build time, not runtime.
// ===================================================================

/// Compile-time guard: `kernel_to_graph` returns `Result<GraphNetwork, VerifyError>`.
///
/// If a merge reverts the graph translation pipeline to panic-based error
/// handling, this test fails to compile.
#[test]
fn contract_kernel_to_graph_returns_result() {
    let kernel = common::snake_kernel();
    let _: Result<_, VerifyError> = kernel_to_graph(&kernel, &[1.0]);
}

/// Compile-time guard: `kernel_to_graph_multi` returns `Result<GraphNetwork, VerifyError>`.
#[test]
fn contract_kernel_to_graph_multi_returns_result() {
    let kernel = common::snake_kernel();
    let bindings = [ParamBinding::Variable, ParamBinding::Constant(1.0)];
    let _: Result<_, VerifyError> = kernel_to_graph_multi(&kernel, &bindings);
}

/// Compile-time guard: `compose_sequential` returns `Result<GraphNetwork, VerifyError>`.
#[test]
fn contract_compose_sequential_returns_result() {
    let snake = common::snake_kernel();
    let spec = SequentialSpec::new(&snake, &snake, &[1.0], &[1.0], 0).expect("valid spec");
    let _: Result<_, VerifyError> = compose_sequential(&spec);
}

/// Compile-time guard: `verify_kernel_smt` returns `Result`.
#[test]
#[cfg(feature = "ay-smt")]
fn contract_verify_kernel_smt_returns_result() {
    use nn_verify::{verify_kernel_smt, ScalarInputBounds, SmtStatusRecord};

    let kernel = common::parse_kernel("fn scale(x: f32) -> f32 { x * 2.0 }");
    let bounds = ScalarInputBounds::new(-1.0, 1.0).expect("bounds");
    let _: Result<SmtStatusRecord, VerifyError> = verify_kernel_smt(&kernel, &[], bounds);
}

/// Compile-time guard: `verify_kernel_smt_with_bounds` returns `Result`.
#[test]
#[cfg(feature = "ay-smt")]
fn contract_verify_kernel_smt_with_bounds_returns_result() {
    use nn_verify::{verify_kernel_smt_with_bounds, ScalarInputBounds, SmtStatusRecord};

    let kernel = common::parse_kernel("fn scale(x: f32) -> f32 { x * 2.0 }");
    let bounds = ScalarInputBounds::new(-1.0, 1.0).expect("bounds");
    let _: Result<SmtStatusRecord, VerifyError> =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds, Some((-3.0, 3.0)));
}

// ===================================================================
// Category 2: Error-path guards
//
// Call public API functions with known-invalid inputs and assert they
// return `Err` (not panic). If a merge reverts to panic-based error
// handling, these tests panic → test failure → regression detected.
// ===================================================================

/// NaN constant parameter must return Err, not panic.
///
/// Guards the `scalar_array` → `NonFiniteConstant` error path through
/// the public `kernel_to_graph` API.
#[test]
fn contract_nan_constant_returns_err() {
    let kernel = common::snake_kernel();
    let result = kernel_to_graph(&kernel, &[f32::NAN]);
    assert!(result.is_err(), "NaN constant should return Err, not Ok");
    let err = result.unwrap_err();
    assert!(
        matches!(err, VerifyError::NonFiniteConstant { .. }),
        "Expected NonFiniteConstant, got {err:?}"
    );
}

/// Infinity constant parameter must return Err, not panic.
#[test]
fn contract_inf_constant_returns_err() {
    let kernel = common::snake_kernel();
    let result = kernel_to_graph(&kernel, &[f32::INFINITY]);
    assert!(result.is_err(), "Inf constant should return Err, not Ok");
    let err = result.unwrap_err();
    assert!(
        matches!(err, VerifyError::NonFiniteConstant { .. }),
        "Expected NonFiniteConstant, got {err:?}"
    );
}

/// Negative infinity constant parameter must return Err.
#[test]
fn contract_neg_inf_constant_returns_err() {
    let kernel = common::snake_kernel();
    let result = kernel_to_graph(&kernel, &[f32::NEG_INFINITY]);
    assert!(result.is_err(), "NEG_INFINITY constant should return Err");
    let err = result.unwrap_err();
    assert!(
        matches!(err, VerifyError::NonFiniteConstant { .. }),
        "Expected NonFiniteConstant, got {err:?}"
    );
}

/// NaN in multi-binding constant must return Err.
///
/// Guards the `kernel_to_graph_multi` constant validation path.
#[test]
fn contract_multi_binding_nan_returns_err() {
    let kernel = common::snake_kernel();
    let bindings = [ParamBinding::Variable, ParamBinding::Constant(f32::NAN)];
    let result = kernel_to_graph_multi(&kernel, &bindings);
    assert!(
        result.is_err(),
        "NaN in multi-binding constant should return Err"
    );
}

/// Wrong parameter count must return Err, not panic.
///
/// Guards the `ParamCountMismatch` error path.
#[test]
fn contract_param_count_mismatch_returns_err() {
    let kernel = common::snake_kernel(); // expects 1 constant (alpha)
    let result = kernel_to_graph(&kernel, &[]); // provide 0
    assert!(
        result.is_err(),
        "Wrong param count should return Err, not panic"
    );
    let err = result.unwrap_err();
    assert!(
        matches!(err, VerifyError::ParamCountMismatch { .. }),
        "Expected ParamCountMismatch, got {err:?}"
    );
}

// ===================================================================
// Category 3: Boundary constant-fold guards
//
// These exercise the full public API path for Compare boundary cases
// that have regressed (#568 regressed #559). They complement the
// regression_sentinels.rs tests by using a different API path.
// ===================================================================

/// Ge equality boundary: 5.0 >= 5.0 must fold to true (1.0).
///
/// Tests through eval_compare_fold which uses the full kernel_to_graph
/// public API. Regression #568 made Ge(5,5) return 0.0 instead of 1.0.
#[test]
fn contract_ge_equality_folds_to_true() {
    let result = common::eval_compare_fold(CompareOpKind::Ge, 5.0, 5.0);
    assert!(
        (result - 1.0).abs() < 0.01,
        "Ge(5,5) should fold to 1.0 (true). Got {result}. \
         If 0.0, the Ge equality case returned false (regression #568)."
    );
}

/// Le equality boundary: 5.0 <= 5.0 must fold to true (1.0).
#[test]
fn contract_le_equality_folds_to_true() {
    let result = common::eval_compare_fold(CompareOpKind::Le, 5.0, 5.0);
    assert!(
        (result - 1.0).abs() < 0.01,
        "Le(5,5) should fold to 1.0 (true). Got {result}."
    );
}

/// Ge strict inequality: 4.0 >= 5.0 must fold to false (0.0).
///
/// Ensures the fix for the equality boundary didn't break the strict case.
#[test]
fn contract_ge_strict_less_folds_to_false() {
    let result = common::eval_compare_fold(CompareOpKind::Ge, 4.0, 5.0);
    assert!(
        result.abs() < 0.01,
        "Ge(4,5) should fold to 0.0 (false). Got {result}."
    );
}

/// Le strict inequality: 6.0 <= 5.0 must fold to false (0.0).
#[test]
fn contract_le_strict_greater_folds_to_false() {
    let result = common::eval_compare_fold(CompareOpKind::Le, 6.0, 5.0);
    assert!(
        result.abs() < 0.01,
        "Le(6,5) should fold to 0.0 (false). Got {result}."
    );
}
