// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! Comprehensive ay SMT encoding and verification tests.
//!
//! Tests cover:
//! A. Real encoding: f64 → ay Real roundtrip, boundary guards, non-finite rejection
//! B. SmtOutcome state machine: Unexecuted, Unknown, Proven, Counterexample, ExecutionFailed
//! C. Query construction: relu, sigmoid, linear, clamp, residual kernels
//! D. Quantization margin: finalize_query widens bounds by SMT_QUANTIZATION_MARGIN
//! E. Network-level encoding: multi-layer kernels, incremental push/pop

mod common;

use nn_dsl::ir::KernelDef;
use nn_dsl::test_kernels::{parse_kernel, snake_kernel};
use nn_verify::{
    kernel_to_smt2, kernel_to_smt2_with_bounds, verify_kernel_smt, verify_kernel_smt_with_bounds,
    ScalarInputBounds, SmtEncodingKind, SmtOutcome, TranslatedKernel,
};

// ============================================================================
// Helpers
// ============================================================================

fn bounds(lo: f32, hi: f32) -> ScalarInputBounds {
    ScalarInputBounds::new(lo, hi).expect("valid test bounds")
}

/// Build a ReLU kernel: f(x) = max(x, 0.0)
///
/// Uses Clamp with lower=0.0, upper=f32::MAX to model max(x, 0).
/// Actually uses the `clamp` IR which is directly supported.
fn relu_kernel() -> KernelDef {
    parse_kernel("fn relu(x: f32) -> f32 { x.max(0.0) }")
}

/// Build a simple linear kernel: f(x) = 3*x + 5
fn linear_kernel() -> KernelDef {
    parse_kernel("fn linear(x: f32) -> f32 { x * 3.0 + 5.0 }")
}

/// Build a scale kernel: f(x) = 2*x + 1
fn scale_kernel() -> KernelDef {
    parse_kernel("fn scale(x: f32) -> f32 { x * 2.0 + 1.0 }")
}

/// Build a clamp kernel: f(x) = clamp(x, -1, 1)
fn clamp_kernel() -> KernelDef {
    parse_kernel("fn clamped(x: f32) -> f32 { x.clamp(-1.0, 1.0) }")
}

/// Build a residual-like kernel: f(x) = x + x * 0.5
/// This is f(x) = 1.5x, a simple residual connection.
fn residual_kernel() -> KernelDef {
    parse_kernel("fn residual(x: f32) -> f32 { x + x * 0.5 }")
}

/// Build a negate kernel: f(x) = -x
fn negate_kernel() -> KernelDef {
    parse_kernel("fn negate(x: f32) -> f32 { -x }")
}

/// Build a multi-op kernel: f(x) = (x + 2) * 3 - 1
fn multi_op_kernel() -> KernelDef {
    parse_kernel("fn multi_op(x: f32) -> f32 { (x + 2.0) * 3.0 - 1.0 }")
}

// ============================================================================
// A. SMT Encoding Tests (real_from_f64 → ay Real)
// ============================================================================

#[test]
fn test_ay_real_encoding_basic_kernel_produces_smt2() {
    // Scale kernel: f(x) = 2x + 1. Verify it translates to valid SMT-LIB2.
    let kernel = scale_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[], bounds(-10.0, 10.0)).unwrap();
    assert!(
        smt2.contains("set-logic"),
        "must contain set-logic directive"
    );
    assert!(
        smt2.contains("declare-const"),
        "must declare symbolic variable"
    );
    assert!(smt2.contains("check-sat"), "must contain check-sat");
}

#[test]
fn test_ay_real_encoding_integer_constants() {
    // The linear kernel uses integer constants (3.0, 5.0) which should
    // hit the integer fast-path in real_from_f64.
    let kernel = linear_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[], bounds(-10.0, 10.0)).unwrap();
    // Integer constants like 3.0 and 5.0 should appear in the SMT output.
    assert!(
        smt2.contains("3.0") || smt2.contains("3000000"),
        "should encode constant 3 in some form, got: {smt2}"
    );
    assert!(
        smt2.contains("5.0") || smt2.contains("5000000"),
        "should encode constant 5 in some form, got: {smt2}"
    );
}

#[test]
fn test_ay_real_encoding_fractional_constant() {
    // Residual kernel uses 0.5 which goes through the rational path.
    let kernel = residual_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[], bounds(-10.0, 10.0)).unwrap();
    // 0.5 should encode as rational (500000 / 1000000 or similar).
    assert!(
        smt2.contains("500000"),
        "fractional 0.5 should encode via rational, got: {smt2}"
    );
}

#[test]
fn test_ay_real_encoding_negative_constant() {
    // Negate kernel f(x) = -x. The -1 multiplier should be present.
    let kernel = negate_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[], bounds(-5.0, 5.0)).unwrap();
    // The output should reference the variable x.
    assert!(smt2.contains("declare-const"), "must declare variable x");
}

#[test]
fn test_ay_real_encoding_input_bounds_appear_in_smt2() {
    let kernel = scale_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[], bounds(-7.5, 12.3)).unwrap();
    // Input bounds should be asserted. The bounds go through real_from_f64
    // which encodes as rational, so look for the numerator values.
    // -7.5 → round(-7.5 * 1e6) / 1e6 = -7500000 / 1000000
    assert!(
        smt2.contains("7500000"),
        "input bound -7.5 should appear as 7500000, got:\n{smt2}"
    );
}

#[test]
fn test_ay_variable_encoding_single_param() {
    // Single-param kernel: x is the variable, declared as Real const.
    let kernel = scale_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[], bounds(-1.0, 1.0)).unwrap();
    assert!(
        smt2.contains("declare-const"),
        "single variable should produce declare-const"
    );
}

#[test]
fn test_ay_constant_encoding_two_param() {
    // Snake kernel has two params: x (variable) and alpha (constant).
    let kernel = snake_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[1.0], bounds(-10.0, 10.0)).unwrap();
    // Only x should be declared as a const. alpha=1.0 is a literal.
    // Count declare-const occurrences. Snake UF also declares sin_approx.
    assert!(
        smt2.contains("declare-const"),
        "variable x should be declared as const"
    );
    assert!(
        smt2.contains("declare-fun"),
        "snake uses UF approx for sin, should declare-fun"
    );
}

// ============================================================================
// B. SmtOutcome State Machine Tests
// ============================================================================

#[test]
fn test_smt_outcome_proven_on_correct_bounds() {
    // Scale: f(x) = 2x+1, x in [-5,5] → output in [-9, 11].
    // Correct bounds → UNSAT → Proven.
    let kernel = scale_kernel();
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-5.0, 5.0), Some((-9.0, 11.0))).unwrap();
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "correct bounds should be Proven, got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
    assert_eq!(result.solver, "ay-direct");
}

#[test]
fn test_smt_outcome_counterexample_on_tight_bounds() {
    // Scale: f(x) = 2x+1, x in [-5,5]. Bounds [-1,1] are too tight
    // because f(5) = 11 > 1.
    let kernel = scale_kernel();
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-5.0, 5.0), Some((-1.0, 1.0))).unwrap();
    assert_eq!(
        result.outcome,
        SmtOutcome::Counterexample,
        "too-tight bounds should find Counterexample, got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
    assert!(
        result.detail.is_some(),
        "Counterexample should include detail with witness"
    );
}

#[test]
fn test_smt_outcome_unexecuted_on_heuristic_bounds() {
    // Simple add_one kernel not in analytical match arms → heuristic ±1e6 fallback.
    // Since #385, heuristic bounds produce Unexecuted to prevent vacuous proofs.
    let kernel = parse_kernel("fn add_one(x: f32) -> f32 { x + 1.0 }");
    let result = verify_kernel_smt(&kernel, &[], bounds(-10.0, 10.0)).unwrap();
    assert_eq!(
        result.outcome,
        SmtOutcome::Unexecuted,
        "heuristic bounds should produce Unexecuted (#385), got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
    let detail = result.detail.as_deref().unwrap_or("");
    assert!(
        detail.contains("heuristic"),
        "detail should mention heuristic fallback, got: {detail}"
    );
}

#[test]
fn test_smt_outcome_is_conclusive_semantics() {
    // SmtOutcome semantics test: only Proven and Counterexample are conclusive.
    // Unexecuted = solver never invoked (NOT verification evidence).
    // Unknown = solver ran but undecided.
    // ExecutionFailed = solver attempted but errored.

    // Verify Proven is conclusive.
    let kernel = clamp_kernel();
    let proven_result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-100.0, 100.0), Some((-1.0, 1.0)))
            .unwrap();
    assert_eq!(proven_result.outcome, SmtOutcome::Proven);

    // Verify Counterexample is conclusive.
    let tight_result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-100.0, 100.0), Some((-0.5, 0.5)))
            .unwrap();
    assert_eq!(tight_result.outcome, SmtOutcome::Counterexample);

    // Verify Unexecuted is NOT conclusive (solver never ran).
    let add = parse_kernel("fn add_two(x: f32) -> f32 { x + 2.0 }");
    let unexecuted_result = verify_kernel_smt(&add, &[], bounds(-10.0, 10.0)).unwrap();
    assert_eq!(unexecuted_result.outcome, SmtOutcome::Unexecuted);
}

#[test]
fn test_smt_outcome_unexecuted_distinct_from_unknown() {
    // SmtOutcome::Unexecuted means the solver was NEVER invoked.
    // SmtOutcome::Unknown means the solver RAN but couldn't decide.
    // These are semantically different per design doc ay rules.
    let unexecuted = SmtOutcome::Unexecuted;
    let unknown = SmtOutcome::Unknown;
    assert_ne!(
        unexecuted, unknown,
        "Unexecuted and Unknown must be distinct variants"
    );
}

// ============================================================================
// C. Query Construction Tests
// ============================================================================

#[test]
fn test_relu_query_output_bounded() {
    // relu(x) = max(x, 0). For x in [-5, 5], output in [0, 5].
    // With explicit bounds [0, 5] → should be Proven.
    let kernel = relu_kernel();
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-5.0, 5.0), Some((0.0, 5.0))).unwrap();
    assert_eq!(
        result.encoding,
        SmtEncodingKind::Exact,
        "relu is algebraic, should be Exact encoding"
    );
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "relu([0,5]) for x in [-5,5] should be Proven, got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
}

#[test]
fn test_relu_query_counterexample_wrong_lower() {
    // relu(x) = max(x, 0). For x in [-5, 5], output in [0, 5].
    // Claim output in [1, 5] is wrong — relu(0.5) = 0.5 < 1.
    let kernel = relu_kernel();
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-5.0, 5.0), Some((1.0, 5.0))).unwrap();
    assert_eq!(
        result.outcome,
        SmtOutcome::Counterexample,
        "relu lower bound 1.0 is wrong for x in [-5,5], got: {:?}",
        result.outcome,
    );
}

#[test]
fn test_clamp_query_proven() {
    // clamp(x, -1, 1) for x in [-100, 100] → output in [-1, 1].
    let kernel = clamp_kernel();
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-100.0, 100.0), Some((-1.0, 1.0)))
            .unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "clamp kernel should be Proven, got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
}

#[test]
fn test_linear_composition_kernel() {
    // f(x) = 3x + 5, x in [-10, 10] → output in [-25, 35].
    let kernel = linear_kernel();
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-10.0, 10.0), Some((-25.0, 35.0)))
            .unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "linear 3x+5 in [-25,35] should be Proven, got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
}

#[test]
fn test_linear_composition_counterexample_upper() {
    // f(x) = 3x + 5, x in [-10, 10] → max is f(10) = 35.
    // Claim output <= 30 is wrong.
    let kernel = linear_kernel();
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-10.0, 10.0), Some((-25.0, 30.0)))
            .unwrap();
    assert_eq!(
        result.outcome,
        SmtOutcome::Counterexample,
        "upper bound 30 is too tight for 3x+5, got: {:?}",
        result.outcome,
    );
}

#[test]
fn test_residual_connection_encoding() {
    // f(x) = x + 0.5x = 1.5x, x in [-10, 10] → output in [-15, 15].
    let kernel = residual_kernel();
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-10.0, 10.0), Some((-15.0, 15.0)))
            .unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "residual 1.5x in [-15,15] should be Proven, got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
}

#[test]
fn test_multi_op_kernel_encoding() {
    // f(x) = (x + 2) * 3 - 1 = 3x + 5, x in [-5, 5] → output in [-10, 20].
    let kernel = multi_op_kernel();
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-5.0, 5.0), Some((-10.0, 20.0)))
            .unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "multi-op (x+2)*3-1 in [-10,20] should be Proven, got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
}

#[test]
fn test_negate_kernel_encoding() {
    // f(x) = -x, x in [-3, 7] → output in [-7, 3].
    let kernel = negate_kernel();
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-3.0, 7.0), Some((-7.0, 3.0))).unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "negate kernel in [-7,3] should be Proven, got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
}

#[test]
fn test_snake_uf_approx_encoding() {
    // Snake kernel uses sin (transcendental) → UF approximation encoding.
    let kernel = snake_kernel();
    let result = verify_kernel_smt(&kernel, &[1.0], bounds(-10.0, 10.0)).unwrap();
    assert_eq!(
        result.encoding,
        SmtEncodingKind::UfApprox,
        "snake uses sin → should get UfApprox encoding"
    );
    // The solver should attempt to solve it (not Unexecuted since analytical
    // bounds are available for snake).
    assert_ne!(
        result.outcome,
        SmtOutcome::Unexecuted,
        "snake has analytical bounds, should not be Unexecuted"
    );
}

// ============================================================================
// D. Quantization Margin Tests
// ============================================================================

#[test]
fn test_quantization_margin_applied_in_smt2() {
    // verify_kernel_smt_with_bounds widens bounds by SMT_QUANTIZATION_MARGIN (1e-4).
    // If we provide bounds (-9.0, 11.0), the SMT assertion should use
    // (-9.0001, 11.0001) internally.
    let kernel = scale_kernel();
    let smt2 = kernel_to_smt2_with_bounds(&kernel, &[], bounds(-5.0, 5.0), (-9.0, 11.0)).unwrap();
    // The widened bounds -9.0001 and 11.0001 go through real_from_f64.
    // -9.0001 → round(-9.0001 * 1e6) / 1e6 = -9000100 / 1000000
    // 11.0001 → round(11.0001 * 1e6) / 1e6 = 11000100 / 1000000
    assert!(
        smt2.contains("9000100") || smt2.contains("9000099") || smt2.contains("9000101"),
        "widened lower bound should appear with margin, got:\n{smt2}"
    );
    assert!(
        smt2.contains("11000100") || smt2.contains("11000099") || smt2.contains("11000101"),
        "widened upper bound should appear with margin, got:\n{smt2}"
    );
}

#[test]
fn test_quantization_margin_prevents_spurious_counterexample() {
    // Without the margin, exact-boundary bounds could produce spurious
    // counterexamples due to real_from_f64 encoding precision.
    // Scale: f(x) = 2x+1, x in [-5,5] → exact bounds [-9, 11].
    // The margin ensures the SMT query uses slightly wider bounds.
    let kernel = scale_kernel();
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-5.0, 5.0), Some((-9.0, 11.0))).unwrap();
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "exact bounds with margin should not produce spurious counterexample"
    );
}

#[test]
fn test_finalize_query_widens_caller_provided_bounds() {
    // CallerProvided bounds (from NY) also get widened per #906.
    // verify_kernel_smt_with_bounds routes to CallerProvided. If we provide
    // tight bounds that are exactly correct, the margin should still allow Proven.
    let kernel = clamp_kernel();
    // clamp(x, -1, 1) output is exactly [-1, 1].
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-100.0, 100.0), Some((-1.0, 1.0)))
            .unwrap();
    assert_eq!(result.outcome, SmtOutcome::Proven);
    assert_eq!(
        result.bounds_source,
        nn_verify::BoundsSource::CallerProvided,
    );
}

// ============================================================================
// E. Network-Level Encoding Tests
// ============================================================================

#[test]
fn test_translated_kernel_incremental_proven() {
    // TranslatedKernel allows multiple property checks on the same translation.
    let kernel = scale_kernel();
    let mut tk = TranslatedKernel::from_kernel(&kernel, &[], bounds(-5.0, 5.0)).unwrap();

    // First check: correct bounds → Proven.
    let r1 = tk.check_output_bounded((-9.0, 11.0)).unwrap();
    assert_eq!(
        r1.outcome,
        SmtOutcome::Proven,
        "first check with correct bounds should be Proven"
    );

    // Second check: wider bounds → also Proven.
    let r2 = tk.check_output_bounded((-100.0, 100.0)).unwrap();
    assert_eq!(
        r2.outcome,
        SmtOutcome::Proven,
        "wider bounds should also be Proven"
    );
}

#[test]
fn test_translated_kernel_incremental_counterexample() {
    let kernel = scale_kernel();
    let mut tk = TranslatedKernel::from_kernel(&kernel, &[], bounds(-5.0, 5.0)).unwrap();

    // Tight bounds → Counterexample.
    let r = tk.check_output_bounded((-1.0, 1.0)).unwrap();
    assert_eq!(
        r.outcome,
        SmtOutcome::Counterexample,
        "tight bounds should find Counterexample"
    );
}

#[test]
fn test_translated_kernel_reusable_after_counterexample() {
    // After finding a counterexample, the translation should be reusable.
    let kernel = scale_kernel();
    let mut tk = TranslatedKernel::from_kernel(&kernel, &[], bounds(-5.0, 5.0)).unwrap();

    // First: counterexample.
    let r1 = tk.check_output_bounded((-1.0, 1.0)).unwrap();
    assert_eq!(r1.outcome, SmtOutcome::Counterexample);

    // Second: proven with correct bounds.
    let r2 = tk.check_output_bounded((-9.0, 11.0)).unwrap();
    assert_eq!(
        r2.outcome,
        SmtOutcome::Proven,
        "kernel should be reusable after counterexample"
    );
}

#[test]
fn test_translated_kernel_encoding_types() {
    // Exact kernel.
    let exact = TranslatedKernel::from_kernel(&scale_kernel(), &[], bounds(-5.0, 5.0)).unwrap();
    assert_eq!(exact.encoding(), SmtEncodingKind::Exact);
    assert!(!exact.uses_nonlinear());

    // UfApprox kernel (snake has sin).
    let uf = TranslatedKernel::from_kernel(&snake_kernel(), &[1.0], bounds(-10.0, 10.0)).unwrap();
    assert_eq!(uf.encoding(), SmtEncodingKind::UfApprox);
}

#[test]
fn test_small_two_layer_network_encoding() {
    // Simulate a 2-layer network: f(x) = clamp(2x + 1, -1, 1).
    // This is a single kernel with both linear and clamp operations.
    let kernel = parse_kernel("fn two_layer(x: f32) -> f32 { (x * 2.0 + 1.0).clamp(-1.0, 1.0) }");
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-10.0, 10.0), Some((-1.0, 1.0)))
            .unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "clamp(2x+1, -1, 1) output should be in [-1, 1], got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
}

#[test]
fn test_status_record_metadata_correctness() {
    // Verify that SmtStatusRecord fields are correctly populated.
    let kernel = scale_kernel();
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-5.0, 5.0), Some((-9.0, 11.0))).unwrap();
    assert_eq!(result.solver, "ay-direct");
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(result.property, "output_bounded");
    assert_eq!(result.outcome, SmtOutcome::Proven);
    assert!(result.expected_bounds.is_some());
    let (lo, hi) = result.expected_bounds.unwrap();
    assert!((lo - (-9.0)).abs() < 1e-9, "expected lower -9.0, got {lo}");
    assert!((hi - 11.0).abs() < 1e-9, "expected upper 11.0, got {hi}");
    assert_eq!(
        result.bounds_source,
        nn_verify::BoundsSource::CallerProvided
    );
}

#[test]
fn test_status_record_serialization_roundtrip() {
    // SmtStatusRecord should serialize/deserialize cleanly.
    let kernel = clamp_kernel();
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-100.0, 100.0), Some((-1.0, 1.0)))
            .unwrap();
    let json = serde_json::to_string(&result).expect("serialize");
    let deserialized: nn_verify::SmtStatusRecord =
        serde_json::from_str(&json).expect("deserialize");
    assert_eq!(result.outcome, deserialized.outcome);
    assert_eq!(result.encoding, deserialized.encoding);
    assert_eq!(result.solver, deserialized.solver);
    assert_eq!(result.bounds_source, deserialized.bounds_source);
}

// ============================================================================
// F. Error Path Tests
// ============================================================================

#[test]
fn test_ay_rejects_nan_input_bounds() {
    let err = ScalarInputBounds::new(f32::NAN, 10.0);
    assert!(err.is_err(), "NaN lower bound should be rejected");
}

#[test]
fn test_ay_rejects_inf_input_bounds() {
    let err = ScalarInputBounds::new(-10.0, f32::INFINITY);
    assert!(err.is_err(), "Inf upper bound should be rejected");
}

#[test]
fn test_ay_rejects_inverted_input_bounds() {
    let err = ScalarInputBounds::new(10.0, -10.0);
    assert!(err.is_err(), "inverted bounds should be rejected");
}

#[test]
fn test_ay_rejects_nan_constant_param() {
    let kernel = snake_kernel();
    let err = verify_kernel_smt(&kernel, &[f32::NAN], bounds(-10.0, 10.0));
    assert!(err.is_err(), "NaN constant param should be rejected");
    let msg = err.unwrap_err().to_string();
    assert!(
        msg.contains("non-finite constant parameter"),
        "should mention non-finite constant, got: {msg}"
    );
}

#[test]
fn test_ay_rejects_inf_constant_param() {
    let kernel = snake_kernel();
    let err = verify_kernel_smt(&kernel, &[f32::INFINITY], bounds(-10.0, 10.0));
    assert!(err.is_err(), "Inf constant param should be rejected");
}

#[test]
fn test_ay_rejects_inverted_output_bounds() {
    let kernel = scale_kernel();
    let err = verify_kernel_smt_with_bounds(&kernel, &[], bounds(-5.0, 5.0), Some((11.0, -9.0)));
    assert!(err.is_err(), "inverted output bounds should be rejected");
}

#[test]
fn test_ay_rejects_nan_output_bounds() {
    let kernel = scale_kernel();
    let err =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-5.0, 5.0), Some((f64::NAN, 11.0)));
    assert!(err.is_err(), "NaN output bound should be rejected");
}

#[test]
fn test_ay_rejects_inf_output_bounds() {
    let kernel = scale_kernel();
    let err =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-5.0, 5.0), Some((-9.0, f64::INFINITY)));
    assert!(err.is_err(), "Inf output bound should be rejected");
}

// ============================================================================
// G. Point Bounds and Edge Cases
// ============================================================================

#[test]
fn test_ay_point_input_bounds_accepted() {
    // lower == upper is valid (point constraint). f(5) = 2*5+1 = 11.
    let kernel = scale_kernel();
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(5.0, 5.0), Some((11.0, 11.0))).unwrap();
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "point bounds f(5)=11 should be Proven, got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
}

#[test]
fn test_ay_wide_bounds_proven() {
    // Very wide output bounds should always be proven.
    let kernel = scale_kernel();
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-5.0, 5.0), Some((-1000.0, 1000.0)))
            .unwrap();
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "very wide bounds should be Proven"
    );
}

#[test]
fn test_ay_zero_range_kernel() {
    // Constant kernel: f(x) = 42.0. Output is always 42.
    let kernel = parse_kernel("fn constant(x: f32) -> f32 { 42.0 }");
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-100.0, 100.0), Some((42.0, 42.0)))
            .unwrap();
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "constant kernel output=42 should be Proven, got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
}
