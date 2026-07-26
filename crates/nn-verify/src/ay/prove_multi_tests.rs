// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `ay/prove_multi.rs` — multi-variable SMT verification.

use crate::graph::ParamBinding;
use crate::status::{SmtEncodingKind, SmtOutcome};
use nn_dsl::test_kernels::parse_kernel;

// ---------------------------------------------------------------------------
// verify_kernel_smt_multi — basic invocations
// ---------------------------------------------------------------------------

#[test]
fn test_multi_add_kernel_two_variables() {
    // fn f(x, y) -> f32 { x + y }
    let kernel = parse_kernel("fn f(x: f32, y: f32) -> f32 { x + y }");
    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    let bounds = [(-1.0, 1.0), (-1.0, 1.0)];
    let result = super::verify_kernel_smt_multi(&kernel, &bindings, &bounds, Some((-2.0, 2.0)));
    let record = result.unwrap();
    assert!(record.solver.starts_with("ay"), "solver: {}", record.solver);
    // Pure addition (no real_mul), caller-provided bounds → ay-direct path.
    // ay#5357 fixed: pure QF_LRA reaches Proven.
    assert_eq!(
        record.outcome,
        SmtOutcome::Proven,
        "multi-variable x+y with caller bounds must reach Proven, got: {:?}",
        record.outcome,
    );
}

#[test]
fn test_multi_kernel_with_constant_binding() {
    // fn f(x, alpha) -> f32 { x + alpha }
    let kernel = parse_kernel("fn f(x: f32, alpha: f32) -> f32 { x + alpha }");
    let bindings = [ParamBinding::Variable, ParamBinding::Constant(2.0)];
    let bounds = [(-1.0, 1.0)]; // Only 1 variable bound (x)
    let result = super::verify_kernel_smt_multi(&kernel, &bindings, &bounds, Some((-5.0, 5.0)));
    let record = result.unwrap();
    assert!(record.solver.starts_with("ay"), "solver: {}", record.solver);
}

#[test]
fn test_multi_kernel_heuristic_output_bounds() {
    // When expected_output_bounds is None, uses ±1e6 heuristic fallback.
    let kernel = parse_kernel("fn f(x: f32, y: f32) -> f32 { x + y }");
    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    let bounds = [(-1.0, 1.0), (-1.0, 1.0)];
    let result = super::verify_kernel_smt_multi(&kernel, &bindings, &bounds, None);
    let record = result.unwrap();
    // With heuristic bounds, the query should still be generated.
    assert!(record.solver.starts_with("ay"), "solver: {}", record.solver);
}

#[test]
fn test_multi_kernel_single_variable_with_mul() {
    // fn f(x, y) -> f32 { x * y } — nonlinear with both variable
    let kernel = parse_kernel("fn f(x: f32, y: f32) -> f32 { x * y }");
    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    let bounds = [(-1.0, 1.0), (0.0, 2.0)];
    let result = super::verify_kernel_smt_multi(&kernel, &bindings, &bounds, Some((-2.0, 2.0)));
    // Should succeed (or fail gracefully) — nonlinear is valid for SMT.
    let record = result.unwrap();
    assert!(record.solver.starts_with("ay"), "solver: {}", record.solver);
}

// ---------------------------------------------------------------------------
// verify_kernel_smt_multi — encoding and bounds source
// ---------------------------------------------------------------------------

#[test]
fn test_multi_exact_encoding_for_linear_kernel() {
    // A purely linear kernel (x + y) should use Exact encoding (no UFs).
    let kernel = parse_kernel("fn f(x: f32, y: f32) -> f32 { x + y }");
    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    let bounds = [(-1.0, 1.0), (-1.0, 1.0)];
    let result = super::verify_kernel_smt_multi(&kernel, &bindings, &bounds, Some((-2.0, 2.0)));
    let record = result.unwrap();
    assert_eq!(record.encoding, SmtEncodingKind::Exact);
}

#[test]
fn test_multi_kernel_mul_encoding() {
    // x * y is still exact (Real arithmetic supports mul).
    let kernel = parse_kernel("fn f(x: f32, y: f32) -> f32 { x * y }");
    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    let bounds = [(0.0, 1.0), (0.0, 1.0)];
    let result = super::verify_kernel_smt_multi(&kernel, &bindings, &bounds, Some((0.0, 1.0)));
    let record = result.unwrap();
    assert_eq!(record.encoding, SmtEncodingKind::Exact);
}

// ---------------------------------------------------------------------------
// Error paths
// ---------------------------------------------------------------------------

#[test]
fn test_multi_kernel_empty_bindings_no_panic() {
    // Empty bindings — translate_kernel may return an error.
    let kernel = parse_kernel("fn f(x: f32) -> f32 { x + 1.0 }");
    let bindings: &[ParamBinding] = &[];
    let bounds: &[(f32, f32)] = &[];
    // This may error (NoParameters or similar), but should not panic.
    let _ = super::verify_kernel_smt_multi(&kernel, bindings, bounds, None);
}

// ---------------------------------------------------------------------------
// variable_bounds length validation
// ---------------------------------------------------------------------------

#[test]
fn test_multi_bounds_too_few_returns_error() {
    // Two Variable bindings but only one bounds entry — must reject.
    let kernel = parse_kernel("fn f(x: f32, y: f32) -> f32 { x + y }");
    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    let bounds = [(-1.0_f32, 1.0)]; // Missing bounds for y
    let result = super::verify_kernel_smt_multi(&kernel, &bindings, &bounds, Some((-2.0, 2.0)));
    let err = result.expect_err("should reject mismatched variable_bounds length");
    let msg = err.to_string();
    assert!(
        msg.contains("variable_bounds count mismatch"),
        "unexpected error: {}",
        msg
    );
}

#[test]
fn test_multi_bounds_too_many_returns_error() {
    // One Variable binding but two bounds entries — must reject.
    let kernel = parse_kernel("fn f(x: f32, alpha: f32) -> f32 { x + alpha }");
    let bindings = [ParamBinding::Variable, ParamBinding::Constant(1.0)];
    let bounds = [(-1.0_f32, 1.0), (-2.0, 2.0)]; // Excess bounds
    let result = super::verify_kernel_smt_multi(&kernel, &bindings, &bounds, Some((-5.0, 5.0)));
    let err = result.expect_err("should reject excess variable_bounds");
    let msg = err.to_string();
    assert!(
        msg.contains("variable_bounds count mismatch"),
        "unexpected error: {}",
        msg
    );
}

#[test]
fn test_multi_bounds_zero_variables_zero_bounds_rejects() {
    // All Constant bindings with no bounds — translate_kernel rejects (no variables).
    let kernel = parse_kernel("fn f(x: f32, y: f32) -> f32 { x + y }");
    let bindings = [ParamBinding::Constant(1.0), ParamBinding::Constant(2.0)];
    let bounds: &[(f32, f32)] = &[];
    let result = super::verify_kernel_smt_multi(&kernel, &bindings, bounds, Some((0.0, 5.0)));
    // Passes bounds validation (0 == 0) but translate_kernel rejects: no variables.
    assert!(result.is_err(), "should reject all-constant bindings");
}

// ---------------------------------------------------------------------------
// Analytical bounds dispatch (#514)
// ---------------------------------------------------------------------------

#[test]
fn test_multi_single_var_rms_norm_uses_analytical_bounds() {
    // rms_norm_scalar(x, rms_inv, weight) = x * rms_inv * weight
    // Via multi-variable API with 1 Variable (weight) + 2 Constants (x=1.0, rms_inv=0.5).
    // With no explicit output bounds, should dispatch to analytical bounds via
    // compute_output_bounds_heuristic (#514), not the ±1e6 fallback.
    let kernel = nn_dsl::build_rms_norm_scalar_kernel().expect("rms_norm_scalar kernel");
    let bindings = [
        ParamBinding::Variable,
        ParamBinding::Constant(1.0),
        ParamBinding::Constant(0.5),
    ];
    let bounds = [(-10.0_f32, 10.0)];
    let result = super::verify_kernel_smt_multi(&kernel, &bindings, &bounds, None);
    let record = result.unwrap();
    // Analytical bounds should be used, not heuristic → not Unexecuted from vacuous gate.
    assert_eq!(
        record.bounds_source,
        crate::status::BoundsSource::Analytical,
        "single-variable multi-path should use analytical bounds, got: {:?} detail: {:?}",
        record.bounds_source,
        record.detail,
    );
}

#[test]
fn test_multi_two_var_kernel_falls_back_to_heuristic() {
    // fn f(x, y) -> f32 { x + y } — two Variables.
    // compute_output_bounds_heuristic only handles single-variable (#514),
    // so two-variable kernels still fall back to ±1e6 heuristic.
    let kernel = parse_kernel("fn f(x: f32, y: f32) -> f32 { x + y }");
    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    let bounds = [(-1.0, 1.0), (-1.0, 1.0)];
    let result = super::verify_kernel_smt_multi(&kernel, &bindings, &bounds, None);
    let record = result.unwrap();
    // Two variables → heuristic fallback → Unexecuted (vacuous gate).
    assert_eq!(
        record.outcome,
        SmtOutcome::Unexecuted,
        "two-variable kernel without explicit bounds should be Unexecuted (heuristic), got: {:?}",
        record.outcome,
    );
}

#[test]
fn test_multi_single_var_non_zero_index_falls_back() {
    // fn f(a, b, x) -> f32 { a * x + b } — Variable at index 2, not 0.
    // compute_output_bounds_heuristic requires #448 convention (Variable at
    // index 0). When Variable is at a different position, constant_params
    // order is wrong and we must fall back to ±1e6 heuristic.
    let kernel = parse_kernel("fn f(a: f32, b: f32, x: f32) -> f32 { a * x + b }");
    let bindings = [
        ParamBinding::Constant(2.0),
        ParamBinding::Constant(1.0),
        ParamBinding::Variable,
    ];
    let bounds = [(-5.0, 5.0)];
    let result = super::verify_kernel_smt_multi(&kernel, &bindings, &bounds, None);
    let record = result.unwrap();
    // Variable at index 2, not 0 → falls back to heuristic ±1e6 → Unexecuted.
    assert_eq!(
        record.outcome,
        SmtOutcome::Unexecuted,
        "variable-not-at-index-0 should fall back to heuristic, got: {:?}",
        record.outcome,
    );
}

// ---------------------------------------------------------------------------
// Conv1d-k1 scalar proof (#2917 AC4)
// ---------------------------------------------------------------------------

#[test]
fn test_conv1d_k1_scalar_reaches_proven() {
    // Conv1d with kernel_size=1 reduces to: out = x * weight + bias
    // This is a linear function of x (weight and bias are constants).
    // ay proves this exactly in QF_LRA (quantifier-free linear real arithmetic).
    let kernel = parse_kernel(
        "fn conv1d_k1_scalar(x: f32, weight: f32, bias: f32) -> f32 { x * weight + bias }",
    );
    let bindings = [
        ParamBinding::Variable,      // x: input activation
        ParamBinding::Constant(0.5), // weight: model parameter
        ParamBinding::Constant(0.1), // bias: model parameter
    ];
    let bounds = [(-10.0_f32, 10.0)]; // x bounds

    // Exact analytical bounds: out = x * 0.5 + 0.1
    // x in [-10, 10] → out in [-10*0.5+0.1, 10*0.5+0.1] = [-4.9, 5.1]
    let result = super::verify_kernel_smt_multi(&kernel, &bindings, &bounds, Some((-4.9, 5.1)));
    let record = result.unwrap();
    assert_eq!(
        record.outcome,
        SmtOutcome::Proven,
        "conv1d_k1_scalar (linear in x) must reach Proven, got: {:?}",
        record.outcome,
    );
    assert_eq!(
        record.encoding,
        SmtEncodingKind::Exact,
        "linear kernel should use Exact encoding"
    );
}

#[test]
fn test_conv1d_k1_scalar_with_negative_weight() {
    // Negative weight flips the output interval direction.
    // out = x * (-2.0) + 3.0, x in [0, 5] → out in [-7, 3]
    let kernel = parse_kernel(
        "fn conv1d_k1_scalar(x: f32, weight: f32, bias: f32) -> f32 { x * weight + bias }",
    );
    let bindings = [
        ParamBinding::Variable,
        ParamBinding::Constant(-2.0),
        ParamBinding::Constant(3.0),
    ];
    let bounds = [(0.0_f32, 5.0)];
    let result = super::verify_kernel_smt_multi(&kernel, &bindings, &bounds, Some((-7.0, 3.0)));
    let record = result.unwrap();
    assert_eq!(
        record.outcome,
        SmtOutcome::Proven,
        "conv1d_k1_scalar with negative weight must reach Proven, got: {:?}",
        record.outcome,
    );
}

#[test]
fn test_conv1d_k1_scalar_analytical_bounds_dispatch() {
    // When no explicit output bounds are provided, the BOUNDS_REGISTRY
    // entry "conv1d_k1_scalar" should provide analytical bounds via
    // compute_output_bounds_heuristic (#448 single-variable convention).
    let kernel = parse_kernel(
        "fn conv1d_k1_scalar(x: f32, weight: f32, bias: f32) -> f32 { x * weight + bias }",
    );
    let bindings = [
        ParamBinding::Variable,
        ParamBinding::Constant(1.0),
        ParamBinding::Constant(0.0),
    ];
    let bounds = [(-1.0_f32, 1.0)];
    // No explicit output bounds — should dispatch to BOUNDS_REGISTRY.
    let result = super::verify_kernel_smt_multi(&kernel, &bindings, &bounds, None);
    let record = result.unwrap();
    assert_eq!(
        record.bounds_source,
        crate::status::BoundsSource::Analytical,
        "conv1d_k1_scalar should use analytical bounds from BOUNDS_REGISTRY, got: {:?}",
        record.bounds_source,
    );
    assert_eq!(
        record.outcome,
        SmtOutcome::Proven,
        "conv1d_k1_scalar with analytical bounds must reach Proven, got: {:?}",
        record.outcome,
    );
}

// ---------------------------------------------------------------------------
// Binary add with explicit caller bounds (#2917 AC3)
// ---------------------------------------------------------------------------

#[test]
fn test_binary_add_ay_proof_with_exact_bounds() {
    // Binary add: out = x + y, both symbolic.
    // With caller-provided exact bounds, ay proves in QF_LRA.
    let kernel = parse_kernel("fn binary_add(x: f32, y: f32) -> f32 { x + y }");
    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    let bounds = [(-5.0_f32, 5.0), (-3.0, 7.0)];
    // Exact: out in [-5+(-3), 5+7] = [-8, 12]
    let result = super::verify_kernel_smt_multi(&kernel, &bindings, &bounds, Some((-8.0, 12.0)));
    let record = result.unwrap();
    assert_eq!(
        record.outcome,
        SmtOutcome::Proven,
        "binary_add with exact bounds must reach Proven, got: {:?}",
        record.outcome,
    );
}
