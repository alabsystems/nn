// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for ay direct execution and heuristic bounds dispatch.
//! Extracted from prove_tests.rs (#356). SiLU-mul and counterexample tests
//! further extracted to prove_tests_silu.rs and prove_tests_counterexample.rs (#418).

use super::prove_dispatch::compute_output_bounds_heuristic;
use super::*;
use crate::test_helpers::bounds;
use nn_dsl::ir::{BinOpKind, IRNode, IRNodeKind, NodeId, Param, ScalarType};
use nn_dsl::test_kernels::{identity_kernel, square_kernel};

/// Helper: 3-param kernel `f(a, b, x) = a * x + b` for testing constant param
/// validation at index > 0 while leaving at least one symbolic variable.
fn three_param_linear_kernel() -> KernelDef {
    KernelDef::new(
        "linear_3p",
        vec![
            Param::new("a", ScalarType::F32),
            Param::new("b", ScalarType::F32),
            Param::new("x", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)), // a
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)), // b
            IRNode::new(NodeId::new(2), IRNodeKind::Param(2)), // x
            IRNode::new(
                NodeId::new(3),
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(2),
                },
            ), // a * x
            IRNode::new(
                NodeId::new(4),
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(3),
                    rhs: NodeId::new(1),
                },
            ), // a * x + b
        ],
        NodeId::new(4),
    )
}

#[test]
fn test_verify_second_constant_param_nan_rejected() {
    let kernel = three_param_linear_kernel();
    // three_param_linear_kernel params: x, a, b.
    // build_smt_query convention: param 0 (x) = Variable, params 1..N = Constant (#448).
    // constant_params = [1.0, NaN] → bindings = [Variable, Constant(1.0), Constant(NaN)].
    // NaN is at kernel param index 2 (b), not index 1.
    let err = verify_kernel_smt(&kernel, &[1.0, f32::NAN], bounds(-10.0, 10.0)).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite constant parameter"),
        "NaN at param b should produce NonFiniteConstantParam, got: {msg}"
    );
    assert!(
        msg.contains("index 2"),
        "error should cite kernel param index 2 (b, the second constant), got: {msg}"
    );
}

// --- Direct execution tests ---

#[test]
fn test_direct_execution_identity_proven() {
    let kernel = identity_kernel();
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-10.0, 10.0), Some((-10.0, 10.0)))
            .unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(result.solver, "ay-direct");
    // ay#5357 fix landed (6aac039): strict Proven assertion.
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "ay#5357 fixed: identity kernel must reach Proven, got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
}

#[test]
fn test_direct_execution_identity_counterexample() {
    let kernel = identity_kernel();
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-10.0, 10.0), Some((-5.0, 5.0)))
            .unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(result.solver, "ay-direct");
    // ay#5357 fix landed (6aac039): strict Counterexample assertion.
    assert_eq!(
        result.outcome,
        SmtOutcome::Counterexample,
        "ay#5357 fixed: too-tight bounds must find Counterexample, got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
}

#[test]
fn test_nonlinear_kernel_reaches_nra_solver() {
    // #2640: square kernel (x*x) is nonlinear but now routes to ay NRA solver
    // via ALL logic auto-detection instead of being gated as Unexecuted.
    let kernel = square_kernel();
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-5.0, 5.0), Some((0.0, 25.0))).unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_ne!(
        result.outcome,
        SmtOutcome::Unexecuted,
        "nonlinear kernel should no longer be Unexecuted (#2640), got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
    assert_eq!(
        result.solver, "ay-direct",
        "nonlinear kernel should use ay-direct with ALL logic (#2640)"
    );
}

// --- UF direct execution tests (#2617) ---

#[test]
fn test_uf_tanh_direct_execution_proven() {
    // Tanh: single UF call with range [-1, 1]. Truly linear in the SMT sense
    // (no division by non-ground, no symbolic*symbolic multiplication).
    // With #2617, QF_UFLRA direct execution proves tanh_approx ∈ [-1, 1].
    let kernel = nn_dsl::tanh_kernel::build_tanh_kernel().expect("tanh kernel");
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-10.0, 10.0), Some((-1.0, 1.0)))
            .unwrap();
    assert_eq!(
        result.encoding,
        SmtEncodingKind::UfApprox,
        "tanh uses tanh_approx UF"
    );
    assert_eq!(
        result.solver, "ay-direct",
        "UfApprox linear kernel should use direct execution (#2617)"
    );
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "tanh ∈ [-1,1] with tanh_approx range axiom should be provable via QF_UFLRA, \
         got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
    assert_eq!(result.bounds_source, BoundsSource::CallerProvided);
}

#[test]
fn test_uf_sigmoid_too_tight_bounds_counterexample() {
    // Explicit bounds (0.4, 0.6) are too tight for sigmoid with x ∈ [-10, 10].
    // The solver should find a counterexample (e.g., sigmoid(-10) ≈ 0.000045 < 0.4).
    let kernel = nn_dsl::sigmoid::build_sigmoid_kernel().expect("sigmoid kernel");
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-10.0, 10.0), Some((0.4, 0.6))).unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::UfApprox);
    assert_eq!(
        result.solver, "ay-direct",
        "should use direct execution even for counterexample"
    );
    assert_eq!(
        result.outcome,
        SmtOutcome::Counterexample,
        "too-tight bounds should produce counterexample, got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
}

#[test]
fn test_uf_snake_reaches_nra_solver() {
    // #2640: Snake has powi(2) on symbolic base → uses_nonlinear = true.
    // Now routes to ay NRA solver via ALL logic auto-detection instead
    // of being gated as Unexecuted. The NRA solver may return Proven,
    // Counterexample, or Unknown depending on its handling of UF+NRA.
    let kernel = nn_dsl::test_kernels::snake_kernel();
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[1.0], bounds(-10.0, 10.0), Some((-20.0, 20.0)))
            .unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::UfApprox);
    assert_ne!(
        result.outcome,
        SmtOutcome::Unexecuted,
        "snake should no longer be Unexecuted (#2640), got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
    assert_eq!(
        result.solver, "ay-direct",
        "nonlinear UF kernel should use ay-direct with ALL logic (#2640)"
    );
}

// --- heuristic bounds dispatch tests (#251) ---

#[test]
fn test_exact_snake_name_match_not_substring() {
    let kernel = KernelDef::new(
        "snake_variant",
        vec![
            Param::new("alpha", ScalarType::F32),
            Param::new("x", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );

    let smt2 = kernel_to_smt2(&kernel, &[1.0], bounds(-10.0, 10.0)).unwrap();
    assert!(
        smt2.contains("1000000") || smt2.contains("1e6") || smt2.contains("1000010"),
        "snake_variant should use ±1e6 fallback, not snake analytical bounds"
    );
}

// ======================== Heuristic bounds tests (#385) ========================

#[test]
fn test_heuristic_fallback_produces_unexecuted_not_proven() {
    // add_one: output = x + 1. Name "add_one" is not in the analytical match
    // arms, so compute_output_bounds_heuristic falls back to ±1e6.
    // Without #385 fix, this would hit direct execution and return Proven
    // (trivially — no realistic output violates bounds widened by a million).
    // With #385 fix, heuristic bounds are detected and Unexecuted is returned.
    let kernel = KernelDef::new(
        "add_one",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(1.0)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    // No explicit bounds → triggers heuristic fallback.
    let result = verify_kernel_smt(&kernel, &[], bounds(-10.0, 10.0)).unwrap();
    assert_eq!(
        result.outcome,
        SmtOutcome::Unexecuted,
        "heuristic ±1e6 bounds should produce Unexecuted, not Proven. \
         Detail: {:?}",
        result.detail,
    );
    let detail = result.detail.as_deref().unwrap_or("");
    assert!(
        detail.contains("heuristic"),
        "detail should mention heuristic fallback, got: {detail}"
    );
    // #383: heuristic fallback → BoundsSource::Heuristic.
    assert_eq!(
        result.bounds_source,
        BoundsSource::Heuristic,
        "add_one is not in BOUNDS_REGISTRY → should produce Heuristic"
    );
}

#[test]
fn test_explicit_bounds_still_produce_proven() {
    // Same kernel with explicit tight bounds should still produce a
    // meaningful proof (Proven or Unknown from ay solver limitations).
    let kernel = KernelDef::new(
        "add_one",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(1.0)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    // Explicit bounds: add_one with x ∈ [-10, 10] → output ∈ [-9, 11].
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-10.0, 10.0), Some((-9.0, 11.0)))
            .unwrap();
    // ay#5357 fix landed (6aac039): strict Proven assertion.
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "ay#5357 fixed: add_one kernel must reach Proven, got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
}

// ay#5357 fix landed (ay commit 6aac039): ay-direct now reaches Proven and
// Counterexample for QF_LRA kernels. Assertions tightened from `Proven | Unknown`
// to strict `Proven` / `Counterexample`. The `ay_direct_fixed` feature gate and
// 5 gated tests were removed as dead code (#544) before the fix landed.

// ======================== Direct heuristic guard tests (#469) ========================

#[test]
fn test_heuristic_nan_constant_param_reports_kernel_index() {
    // compute_output_bounds_heuristic receives constant_params[i] = kernel param i+1
    // (#448 variable-first: param 0 is variable). Its NaN guard must report
    // the kernel param index (i+1), not the array index (i).
    let kernel = three_param_linear_kernel(); // params: a, b, x
                                              // constant_params = [NaN, 2.0] → NaN at array index 0 = kernel param 1
    let err = compute_output_bounds_heuristic(&kernel, &[f32::NAN, 2.0], -10.0, 10.0).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite constant parameter"),
        "NaN constant should produce NonFiniteConstantParam, got: {msg}"
    );
    assert!(
        msg.contains("index 1"),
        "error should report kernel param index 1 (not array index 0), got: {msg}"
    );
}

#[test]
fn test_heuristic_second_nan_constant_param_reports_correct_index() {
    let kernel = three_param_linear_kernel();
    // constant_params = [1.0, NaN] → NaN at array index 1 = kernel param 2
    let err = compute_output_bounds_heuristic(&kernel, &[1.0, f32::NAN], -10.0, 10.0).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("index 2"),
        "error should report kernel param index 2 (not array index 1), got: {msg}"
    );
}

// ======================== Input bounds finiteness guard (#471) ========================

#[test]
fn test_heuristic_nan_input_lower_rejected() {
    let kernel = identity_kernel();
    let err = compute_output_bounds_heuristic(&kernel, &[], f32::NAN, 10.0).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite input bounds"),
        "NaN input_lower should produce NonFiniteInputBound, got: {msg}"
    );
}

#[test]
fn test_heuristic_nan_input_upper_rejected() {
    let kernel = identity_kernel();
    let err = compute_output_bounds_heuristic(&kernel, &[], -10.0, f32::NAN).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite input bounds"),
        "NaN input_upper should produce NonFiniteInputBound, got: {msg}"
    );
}

#[test]
fn test_heuristic_inf_input_bounds_rejected() {
    let kernel = identity_kernel();
    let err = compute_output_bounds_heuristic(&kernel, &[], f32::NEG_INFINITY, f32::INFINITY)
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite input bounds"),
        "+/-Inf input bounds should produce NonFiniteInputBound, got: {msg}"
    );
}
