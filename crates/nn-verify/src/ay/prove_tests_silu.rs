// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SiLU-mul bounds tests. Extracted from prove_tests_execution.rs (#418).

use super::prove_dispatch::silu_mul_output_bounds;
use super::*;
use crate::test_helpers::bounds;

/// Helper: build a SiLU-Mul kernel for heuristic bounds testing.
fn silu_mul_kernel() -> KernelDef {
    nn_dsl::silu_mul::build_silu_mul_kernel().expect("silu_mul kernel must build")
}

#[test]
fn test_silu_mul_uses_tight_heuristic_bounds() {
    // #459: silu_mul(x, up) — x is variable in [-10, 10], up=1.0 is constant.
    // Analytical bounds: silu(-10)*1 ≈ -0.000454, silu(10)*1 ≈ 9.999546.
    let kernel = silu_mul_kernel();
    let smt2 =
        kernel_to_smt2(&kernel, &[1.0], bounds(-10.0, 10.0)).expect("smt2 generation for silu_mul");
    assert!(
        !smt2.contains("1000010"),
        "silu_mul should use analytical bounds, not ±1e6 fallback. SMT2:\n{smt2}"
    );
    // silu(10)*1 ≈ 9.999546, widened by SMT_QUANTIZATION_MARGIN (#539) to
    // ~9.999646 → encoded as (/ 9999646 1000000).
    assert!(
        smt2.contains("9999646"),
        "silu_mul bounds should contain silu(10) ≈ 9.9996 encoding (with margin). SMT2:\n{smt2}"
    );
}

#[test]
fn test_silu_mul_bounds_negative_up_const() {
    // #459: silu_mul_output_bounds(up_const=-5.0, x_lower=-10.0, x_upper=10.0).
    // silu has a global minimum at x≈-1.278 where silu≈-0.278.
    // silu range over [-10, 10] = [-0.278, 9.9995].
    // Output = silu(x) * (-5.0):
    //   lower = 9.9995 * (-5.0) ≈ -49.998
    //   upper = (-0.278) * (-5.0) ≈ 1.39
    let result = silu_mul_output_bounds(-5.0, -10.0, 10.0).expect("negative up_const bounds");
    assert!(
        result.0 < -40.0,
        "lower bound should be < -40, got {}",
        result.0
    );
    assert!(
        result.1 > 1.0 && result.1 < 2.0,
        "upper bound should be ~1.39 (silu minimum * -up), got {}",
        result.1
    );
}

#[test]
fn test_silu_mul_bounds_zero_input_range() {
    let result = silu_mul_output_bounds(1.0, 0.0, 0.0).expect("zero-range bounds");
    assert_eq!(result.0, 0.0);
    assert_eq!(result.1, 0.0);
}

// ======================== Input finiteness guard tests (#374) ========================

#[test]
fn test_silu_mul_bounds_nan_up_const_returns_non_finite_constant_param() {
    // #459: first param is up_const (kernel param 1 = index 1).
    let err = silu_mul_output_bounds(f64::NAN, -10.0, 10.0).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite constant parameter"),
        "NaN up_const should produce NonFiniteConstantParam, got: {msg}"
    );
    assert!(
        msg.contains("index 1"),
        "error should cite parameter index 1 (up_const), got: {msg}"
    );
}

#[test]
fn test_silu_mul_bounds_positive_infinity_up_const_returns_error() {
    let err = silu_mul_output_bounds(f64::INFINITY, -10.0, 10.0).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite constant parameter"),
        "+Inf up_const should produce NonFiniteConstantParam, got: {msg}"
    );
}

#[test]
fn test_silu_mul_bounds_negative_infinity_up_const_returns_error() {
    let err = silu_mul_output_bounds(f64::NEG_INFINITY, -10.0, 10.0).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite constant parameter"),
        "-Inf up_const should produce NonFiniteConstantParam, got: {msg}"
    );
}

#[test]
fn test_silu_mul_bounds_non_finite_input_lower_returns_error() {
    let err = silu_mul_output_bounds(1.0, f64::NAN, 10.0).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite output bounds"),
        "NaN input_lower should produce NonFiniteBound, got: {msg}"
    );
}

#[test]
fn test_silu_mul_bounds_non_finite_input_upper_returns_error() {
    let err = silu_mul_output_bounds(1.0, -10.0, f64::INFINITY).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite output bounds"),
        "+Inf input_upper should produce NonFiniteBound, got: {msg}"
    );
}
