// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error-path tests for bounds functions (#575).
//!
//! These call bounds functions directly (bypassing the registry length check in
//! `compute_output_bounds_heuristic`) to exercise the defense-in-depth guard.
//! Each test passes too-few constant params and asserts `InternalTranslationError`.
//!
//! Extracted from `prove_tests_bounds_dispatch.rs` for file size (#763).

use crate::error::VerifyError;

/// Helper: assert that a bounds function result is InternalTranslationError
/// containing the expected function name.
fn assert_require_params_error(result: Result<Option<(f64, f64)>, VerifyError>, expected_fn: &str) {
    match result {
        Err(VerifyError::InternalTranslationError { context }) => {
            assert!(
                context.contains(expected_fn),
                "error context should mention '{expected_fn}', got: {context}"
            );
            assert!(
                context.contains("constant params"),
                "error context should mention 'constant params', got: {context}"
            );
        }
        other => unreachable!(
            "{expected_fn} with insufficient params should return InternalTranslationError, got {other:?}"
        ),
    }
}

// --- Scalar kernel error paths ---

#[test]
fn test_require_params_snake_empty() {
    let result = super::prove_dispatch::bounds_snake(&[], -1.0, 1.0);
    assert_require_params_error(result, "bounds_snake");
}

#[test]
fn test_require_params_silu_mul_empty() {
    let result = super::prove_dispatch::bounds_silu_mul(&[], -1.0, 1.0);
    assert_require_params_error(result, "bounds_silu_mul");
}

#[test]
fn test_require_params_rope_cos_one_param() {
    // Requires 2, provide 1.
    let result = super::prove_dispatch::bounds_rope_cos(&[0.5], -1.0, 1.0);
    assert_require_params_error(result, "bounds_rope_cos");
}

#[test]
fn test_require_params_rope_sin_empty() {
    let result = super::prove_dispatch::bounds_rope_sin(&[], -1.0, 1.0);
    assert_require_params_error(result, "bounds_rope_sin");
}

#[test]
fn test_require_params_rms_norm_scalar_one_param() {
    // Requires 2, provide 1.
    let result = super::prove_dispatch::bounds_rms_norm_scalar(&[0.5], -1.0, 1.0);
    assert_require_params_error(result, "bounds_rms_norm_scalar");
}

#[test]
fn test_require_params_norm_affine_three_params() {
    // Requires 5, provide 3.
    let result = super::prove_dispatch::bounds_norm_affine(&[0.0, 1.0, 1e-5], -1.0, 1.0);
    assert_require_params_error(result, "bounds_norm_affine");
}

#[test]
fn test_require_params_instance_norm_one_param() {
    // Requires 3, provide 1.
    let result = super::prove_dispatch::bounds_instance_norm(&[0.0], -1.0, 1.0);
    assert_require_params_error(result, "bounds_instance_norm");
}

#[test]
fn test_require_params_adain_two_params() {
    // Requires 5, provide 2.
    let result = super::prove_dispatch::bounds_adain(&[0.0, 1.0], -1.0, 1.0);
    assert_require_params_error(result, "bounds_adain");
}

// --- LeakyReLU error path (requires 1 constant param) ---

#[test]
fn test_require_params_leaky_relu_empty() {
    let result = super::prove_dispatch::bounds_leaky_relu(&[], -1.0, 1.0);
    assert_require_params_error(result, "bounds_leaky_relu");
}

// --- Fused kernel error path ---

#[test]
fn test_require_params_adain_snake_four_params() {
    // Requires 6, provide 4.
    let result = super::prove_dispatch::bounds_adain_snake(&[0.0, 1.0, 1.0, 0.0], -1.0, 1.0);
    assert_require_params_error(result, "bounds_adain_snake");
}
