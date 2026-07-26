// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT regression sentinel tests (extracted from regression_sentinels.rs).
//!
//! Sentinel 3: ay SMT translation round-trip (#566 regression)
//!
//! The regression: ay translation/execution could return Unknown(internal-error)
//! after environmental changes. This sentinel verifies that the full path from
//! KernelDef → SMT-LIB2 → solver execution produces a valid SmtOutcome for
//! a kernel with exact arithmetic (no transcendentals).

#[cfg(feature = "ay-smt")]
use super::common;
#[cfg(feature = "ay-smt")]
use nn_dsl::ir::UnaryFnKind;
#[cfg(feature = "ay-smt")]
use nn_verify::ScalarInputBounds;

/// Verify ay SMT translation round-trip for a linear kernel.
///
/// f(x) = 2*x + 1 is pure arithmetic → exact encoding → ay-direct solver.
/// The result must be a valid SmtOutcome (Proven, Unknown, or Unexecuted),
/// never an internal error.
#[test]
#[cfg(feature = "ay-smt")]
fn sentinel_ay_linear_kernel_round_trip() {
    use nn_verify::{verify_kernel_smt_with_bounds, SmtOutcome};

    let kernel = common::parse_kernel("fn scale(x: f32) -> f32 { x * 2.0 + 1.0 }");

    let bounds = ScalarInputBounds::new(-5.0, 5.0).expect("bounds");
    // True output: 2*(-5)+1=-9 to 2*5+1=11, widen slightly for soundness
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds, Some((-10.0, 12.0))).expect("ay");

    // ay#5605 fix landed: real_mul incompleteness resolved — ay-direct now
    // handles constant multiplication. Scale kernel (2x+1) must reach Proven.
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "ay#5605 fixed: scale kernel (2x+1) must reach Proven, got: {:?} (detail: {:?}). \
         If Unknown, the ay real_mul fix may have regressed.",
        result.outcome,
        result.detail,
    );
}

/// Verify ay SMT translation catches counterexamples for tight bounds.
///
/// When claimed bounds are too tight, ay should find a counterexample
/// (or report Unknown). It should NOT report Proven.
#[test]
#[cfg(feature = "ay-smt")]
fn sentinel_ay_linear_kernel_counterexample() {
    use nn_verify::{verify_kernel_smt_with_bounds, SmtOutcome};

    let kernel = common::parse_kernel("fn scale(x: f32) -> f32 { x * 2.0 + 1.0 }");

    let bounds = ScalarInputBounds::new(-5.0, 5.0).expect("bounds");
    // Claimed bounds [-1, 1] are too tight for f(x)=2x+1 on [-5,5]
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds, Some((-1.0, 1.0))).expect("ay");

    assert!(
        !matches!(result.outcome, SmtOutcome::Proven),
        "ay should NOT prove too-tight bounds as correct. Got Proven for \
         f(x)=2x+1 on [-5,5] with claimed bounds [-1,1]. \
         If Proven, the SMT query is malformed (#566)."
    );
}

/// Verify ay SMT translation for a transcendental kernel (UF approximation).
///
/// sin(x) requires uninterpreted function approximation. The result should
/// be Unexecuted (Phase A — ay-direct doesn't support FuncApp), not an error.
#[test]
#[cfg(feature = "ay-smt")]
fn sentinel_ay_transcendental_uf_encoding() {
    use nn_verify::{verify_kernel_smt, SmtEncodingKind, SmtOutcome};

    let kernel = common::unary_fn_kernel(UnaryFnKind::Sin);

    #[allow(clippy::approx_constant)]
    let bounds = ScalarInputBounds::new(-3.14, 3.14).expect("bounds");
    let result = verify_kernel_smt(&kernel, &[], bounds).expect("ay sin");

    assert_eq!(
        result.encoding,
        SmtEncodingKind::UfApprox,
        "sin(x) should use UF approximation encoding"
    );

    // UF-encoded kernels produce Unexecuted (Phase A: SMT-LIB2 generated,
    // solver not invoked because ay-direct doesn't support FuncApp).
    assert_eq!(
        result.outcome,
        SmtOutcome::Unexecuted,
        "UF-encoded kernel should be Unexecuted (Phase A), got {:?}. \
         If Unknown(internal-error), the ay translation regressed (#566).",
        result.outcome
    );
}
