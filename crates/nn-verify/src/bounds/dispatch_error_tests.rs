// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error-path tests for bounds functions (#575, #859).
//!
//! These call bounds functions directly (bypassing the registry length check in
//! `compute_output_bounds_heuristic`) to exercise the defense-in-depth guard.
//! Each test passes too-few constant params and asserts `InternalTranslationError`.

use crate::bounds::{
    bounds_ada_layer_norm, bounds_adain, bounds_adain_leaky_relu, bounds_adain_snake,
    bounds_instance_norm, bounds_leaky_relu, bounds_norm_affine, bounds_rms_norm_scalar,
    bounds_rope_cos, bounds_rope_sin, bounds_silu_mul, bounds_snake,
};
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

#[test]
fn test_require_params_snake_empty() {
    let result = bounds_snake(&[], -1.0, 1.0);
    assert_require_params_error(result, "bounds_snake");
}

#[test]
fn test_require_params_silu_mul_empty() {
    let result = bounds_silu_mul(&[], -1.0, 1.0);
    assert_require_params_error(result, "bounds_silu_mul");
}

#[test]
fn test_require_params_rope_cos_one_param() {
    let result = bounds_rope_cos(&[0.5], -1.0, 1.0);
    assert_require_params_error(result, "bounds_rope_cos");
}

#[test]
fn test_require_params_rope_sin_empty() {
    let result = bounds_rope_sin(&[], -1.0, 1.0);
    assert_require_params_error(result, "bounds_rope_sin");
}

#[test]
fn test_require_params_rms_norm_scalar_one_param() {
    let result = bounds_rms_norm_scalar(&[0.5], -1.0, 1.0);
    assert_require_params_error(result, "bounds_rms_norm_scalar");
}

#[test]
fn test_require_params_norm_affine_three_params() {
    let result = bounds_norm_affine(&[0.0, 1.0, 1e-5], -1.0, 1.0);
    assert_require_params_error(result, "bounds_norm_affine");
}

#[test]
fn test_require_params_instance_norm_one_param() {
    let result = bounds_instance_norm(&[0.0], -1.0, 1.0);
    assert_require_params_error(result, "bounds_instance_norm");
}

#[test]
fn test_require_params_adain_two_params() {
    let result = bounds_adain(&[0.0, 1.0], -1.0, 1.0);
    assert_require_params_error(result, "bounds_adain");
}

#[test]
fn test_require_params_adain_snake_four_params() {
    let result = bounds_adain_snake(&[0.0, 1.0, 1.0, 0.0], -1.0, 1.0);
    assert_require_params_error(result, "bounds_adain_snake");
}

// --- LeakyReLU error path (requires 1 constant param) ---

#[test]
fn test_require_params_leaky_relu_empty() {
    let result = bounds_leaky_relu(&[], -1.0, 1.0);
    assert_require_params_error(result, "bounds_leaky_relu");
}

// --- Fused AdaIN+LeakyReLU error path (requires 6 constant params) ---

#[test]
fn test_require_params_adain_leaky_relu_four_params() {
    let result = bounds_adain_leaky_relu(&[0.0, 1.0, 1.0, 0.0], -1.0, 1.0);
    assert_require_params_error(result, "bounds_adain_leaky_relu");
}

// --- Fused AdaLayerNorm error path (requires 7 constant params) ---

#[test]
fn test_require_params_ada_layer_norm_five_params() {
    let result = bounds_ada_layer_norm(&[0.0, 1.0, 1e-5, 1.0, 0.0], -1.0, 1.0);
    assert_require_params_error(result, "bounds_ada_layer_norm");
}
