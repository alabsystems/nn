// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for sum_reduce, error paths, and composition in graph_ops.
//!
//! Split from graph_ops_tests.rs (#478). Strategy: build KernelDef via Lowerer,
//! translate to NY GraphNetwork, propagate IBP bounds, verify output
//! bounds are mathematically correct.

use crate::graph::{kernel_to_graph, kernel_to_graph_multi, ParamBinding};
use crate::test_helpers::{parse_kernel, propagate_single};

// ---------------------------------------------------------------------------
// SumReduce translation tests (sum_reduce.rs)
// ---------------------------------------------------------------------------

#[test]
fn test_sum_reduce_all_constants() {
    // f(x) = x + sum_reduce([a, b, c]) with a=1, b=2, c=3 → x + 6
    // x in [0, 1] → [6, 7]
    let (lo, hi) = propagate_single(
        "fn f(x: f32, a: f32, b: f32, c: f32) -> f32 { x + nn_dsl::sum_reduce([a, b, c]) }",
        &[1.0, 2.0, 3.0],
        0.0,
        1.0,
    );
    assert!((lo - 6.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 7.0).abs() < 1e-5, "upper: {hi}");
}

// ---------------------------------------------------------------------------
// Error path tests — verify proper errors instead of panics
// ---------------------------------------------------------------------------

#[test]
fn test_param_count_mismatch_errors() {
    let kernel = parse_kernel("fn f(x: f32, alpha: f32) -> f32 { x + alpha }");
    // Provide 0 constants for a 2-param kernel (expects 1 constant)
    let result = kernel_to_graph(&kernel, &[]);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("mismatch") || err.contains("Mismatch"),
        "expected param count mismatch error, got: {err}"
    );
}

#[test]
fn test_nonfinite_constant_binding_errors() {
    let kernel = parse_kernel("fn f(x: f32, alpha: f32) -> f32 { x + alpha }");
    let result = kernel_to_graph_multi(
        &kernel,
        &[ParamBinding::Variable, ParamBinding::Constant(f32::NAN)],
    );
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.to_lowercase().contains("finite") || err.to_lowercase().contains("nan"),
        "expected non-finite constant error, got: {err}"
    );
}

#[test]
fn test_nonfinite_constant_inf_binding_errors() {
    let kernel = parse_kernel("fn f(x: f32, alpha: f32) -> f32 { x + alpha }");
    let result = kernel_to_graph_multi(
        &kernel,
        &[
            ParamBinding::Variable,
            ParamBinding::Constant(f32::INFINITY),
        ],
    );
    assert!(result.is_err(), "Inf constant binding should be rejected");
}

// ---------------------------------------------------------------------------
// Composition tests — multi-operation translation
// ---------------------------------------------------------------------------

#[test]
fn test_composition_add_then_clamp() {
    // f(x) = (x + 5.0).clamp(0.0, 10.0), x in [-10, 10]
    // x + 5 in [-5, 15], clamped to [0, 10]
    let (lo, hi) = propagate_single(
        "fn f(x: f32) -> f32 { (x + 5.0).clamp(0.0, 10.0) }",
        &[],
        -10.0,
        10.0,
    );
    assert!((lo - 0.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 10.0).abs() < 1e-5, "upper: {hi}");
}

#[test]
fn test_composition_mul_then_abs() {
    // f(x) = (x * (-1.0)).abs() = |x|, x in [-3, 5]
    // x * -1 in [-5, 3], |...| in [0, 5]
    let (lo, hi) = propagate_single("fn f(x: f32) -> f32 { (x * (-1.0)).abs() }", &[], -3.0, 5.0);
    assert!((lo - 0.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 5.0).abs() < 1e-5, "upper: {hi}");
}

#[test]
fn test_composition_exp_then_sqrt() {
    // f(x) = x.exp().sqrt() = exp(x/2), x in [0, 2]
    // exp([0,2]) = [1, e^2], sqrt([1,e^2]) = [1, e]
    let (lo, hi) = propagate_single("fn f(x: f32) -> f32 { x.exp().sqrt() }", &[], 0.0, 2.0);
    assert!((lo - 1.0).abs() < 0.1, "lower: {lo}");
    let e = std::f32::consts::E;
    assert!((hi - e).abs() < 0.3, "upper: {hi}");
}
