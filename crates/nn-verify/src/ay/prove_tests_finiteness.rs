// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Constant parameter finiteness validation and UF detail tests.
//! Extracted from prove_tests.rs (#418).

use super::*;
use crate::test_helpers::bounds;
use nn_dsl::test_kernels::snake_kernel;

// --- constant_params finiteness validation (#235) ---

#[test]
fn test_verify_nan_constant_param_rejected() {
    let kernel = snake_kernel();
    // NaN constant_params should be caught at translate_kernel entry.
    // build_smt_query convention: param 0 = Variable, params 1..N = Constant (#448).
    // So constant_params[0] maps to kernel param index 1 (alpha), not index 0 (x).
    let err = verify_kernel_smt(&kernel, &[f32::NAN], bounds(-10.0, 10.0)).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite constant parameter"),
        "NaN constant param should produce NonFiniteConstantParam, got: {msg}"
    );
    assert!(
        msg.contains("index 1"),
        "error should cite kernel param index 1 (alpha, the first constant), got: {msg}"
    );
}

#[test]
fn test_verify_inf_constant_param_rejected() {
    let kernel = snake_kernel();
    let err = verify_kernel_smt(&kernel, &[f32::INFINITY], bounds(-10.0, 10.0)).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite constant parameter"),
        "Inf constant param should produce NonFiniteConstantParam, got: {msg}"
    );
}

#[test]
fn test_verify_neg_inf_constant_param_rejected() {
    let kernel = snake_kernel();
    let err = verify_kernel_smt(&kernel, &[f32::NEG_INFINITY], bounds(-10.0, 10.0)).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite constant parameter"),
        "NEG_INFINITY constant param should produce NonFiniteConstantParam, got: {msg}"
    );
}

#[test]
fn test_uf_nonlinear_kernel_reaches_solver() {
    // #2640: Snake has powi(2) on symbolic base → nonlinear. Now routes to
    // ay NRA solver via ALL logic auto-detection instead of Unexecuted.
    let kernel = snake_kernel();
    let result =
        verify_kernel_smt(&kernel, &[1.0], bounds(-10.0, 10.0)).expect("UF kernel verification");
    assert_ne!(
        result.outcome,
        SmtOutcome::Unexecuted,
        "snake should no longer be Unexecuted (#2640), got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
    assert_eq!(
        result.solver, "ay-direct",
        "nonlinear UF kernel should use ay-direct (#2640)"
    );
}

#[test]
fn test_direct_execution_snake_uses_nra() {
    // #2640: Snake routes to ay NRA solver via ALL logic.
    let kernel = snake_kernel();
    let result =
        verify_kernel_smt(&kernel, &[1.0], bounds(-10.0, 10.0)).expect("snake UF verification");
    assert_eq!(result.encoding, SmtEncodingKind::UfApprox);
    assert_ne!(
        result.outcome,
        SmtOutcome::Unexecuted,
        "nonlinear UF kernel should reach NRA solver (#2640), got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
}
