// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `smt_error.rs` — SmtError Display formatting.
//!
//! Extracted from `ay/error_tests.rs` (#859) to run without `ay-smt` feature.

use super::*;

// ---------------------------------------------------------------------------
// Display formatting for each SmtError variant
// ---------------------------------------------------------------------------

#[test]
fn test_unsupported_op_display() {
    let err = SmtError::UnsupportedOp {
        op_description: "custom_op".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("unsupported IR operation"));
    assert!(msg.contains("custom_op"));
}

#[test]
fn test_solver_error_display() {
    let err = SmtError::SolverError {
        reason: "timeout".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("ay solver error"));
    assert!(msg.contains("timeout"));
}

#[test]
fn test_no_parameters_display() {
    let err = SmtError::NoParameters;
    let msg = err.to_string();
    assert!(msg.contains("kernel has no parameters"));
}

#[test]
fn test_non_finite_literal_display() {
    let err = SmtError::NonFiniteLiteral(f64::NAN);
    let msg = err.to_string();
    assert!(msg.contains("non-finite literal value"));
}

#[test]
fn test_non_finite_literal_infinity() {
    let err = SmtError::NonFiniteLiteral(f64::INFINITY);
    let msg = err.to_string();
    assert!(msg.contains("non-finite literal value"));
    assert!(msg.contains("inf"));
}

#[test]
fn test_non_finite_constant_param_display() {
    let err = SmtError::NonFiniteConstantParam {
        index: 1,
        value: f64::NAN,
    };
    let msg = err.to_string();
    assert!(msg.contains("non-finite constant parameter"));
    assert!(msg.contains("index 1"));
}

#[test]
fn test_value_too_large_display() {
    let err = SmtError::ValueTooLargeForRealEncoding(1e15);
    let msg = err.to_string();
    assert!(msg.contains("value too large for real encoding"));
}

#[test]
fn test_inverted_bounds_display() {
    let err = SmtError::InvertedBounds {
        lower: 5.0,
        upper: 1.0,
    };
    let msg = err.to_string();
    assert!(msg.contains("inverted bounds"));
    assert!(msg.contains("5"));
    assert!(msg.contains("1"));
}

#[test]
fn test_invalid_snake_alpha_display() {
    let err = SmtError::InvalidSnakeAlpha(-1.0);
    let msg = err.to_string();
    assert!(msg.contains("invalid alpha for Snake bounds"));
    assert!(msg.contains("-1"));
}

#[test]
fn test_non_finite_bound_display() {
    let err = SmtError::NonFiniteBound {
        lower: f64::NEG_INFINITY,
        upper: f64::INFINITY,
    };
    let msg = err.to_string();
    assert!(msg.contains("non-finite output bounds"));
}

#[test]
fn test_non_finite_input_bound_display() {
    let err = SmtError::NonFiniteInputBound {
        lower: f64::NAN,
        upper: 1.0,
    };
    let msg = err.to_string();
    assert!(msg.contains("non-finite input bounds"));
}

#[test]
fn test_param_count_mismatch_display() {
    let err = SmtError::ParamCountMismatch {
        ir_count: 3,
        expected: 2,
        provided: 5,
    };
    let msg = err.to_string();
    assert!(msg.contains("constant_params count mismatch"));
    assert!(msg.contains("3 params"));
    assert!(msg.contains("5"));
}

#[test]
fn test_index_out_of_bounds_display() {
    let err = SmtError::IndexOutOfBounds {
        context: "node_exprs",
        index: 10,
        length: 5,
    };
    let msg = err.to_string();
    assert!(msg.contains("index out of bounds"));
    assert!(msg.contains("node_exprs"));
    assert!(msg.contains("index 10"));
    assert!(msg.contains("length 5"));
}

// ---------------------------------------------------------------------------
// SmtError → VerifyError conversion
// ---------------------------------------------------------------------------

#[test]
fn test_smt_error_into_verify_error() {
    use crate::error::VerifyError;
    let smt_err = SmtError::NoParameters;
    let verify_err: VerifyError = smt_err.into();
    match verify_err {
        VerifyError::Smt(inner) => {
            let msg = inner.to_string();
            assert!(msg.contains("no parameters"));
        }
        other => panic!("expected VerifyError::Smt, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Debug derive
// ---------------------------------------------------------------------------

#[test]
fn test_smt_error_debug() {
    let err = SmtError::NoParameters;
    let debug = format!("{err:?}");
    assert!(debug.contains("NoParameters"));
}
