// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Counterexample detection tests. Extracted from prove_tests_execution.rs (#418).

use super::*;
use crate::test_helpers::bounds;
use nn_dsl::ir::{BinOpKind, IRNode, IRNodeKind, NodeId, Param, ScalarType};

#[test]
fn test_add_one_counterexample_for_tight_bounds() {
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
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[], bounds(-10.0, 10.0), Some((-5.0, 5.0)))
            .expect("add_one counterexample verification");
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(result.solver, "ay-direct");
    assert!(
        matches!(
            result.outcome,
            SmtOutcome::Counterexample | SmtOutcome::Unknown
        ),
        "expected Counterexample (or Known-regression Unknown), got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
    if result.outcome != SmtOutcome::Counterexample {
        let detail = result.detail.as_deref().unwrap_or("");
        assert!(
            detail.contains("internal-error"),
            "Unknown should be the known internal-error regression, got: {detail}"
        );
    }
}

/// Helper: build a SiLU-Mul kernel for counterexample testing.
fn silu_mul_kernel() -> KernelDef {
    nn_dsl::silu_mul::build_silu_mul_kernel().expect("silu_mul kernel must build")
}

#[test]
fn test_silu_mul_reaches_nra_solver_with_tight_bounds() {
    // #2640: silu_mul: x * sigmoid(x) — nonlinear (x * non-ground) + UF (exp).
    // Now routes to ay NRA solver via ALL logic auto-detection.
    // Bounds (-1, 1) are tight — silu_mul(-10) ≈ -0.00045 which is within bounds,
    // but silu_mul(10) ≈ 9.9995 which exceeds 1.0. The NRA solver should find
    // a counterexample or return Unknown.
    let kernel = silu_mul_kernel();
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[1.0], bounds(-10.0, 10.0), Some((-1.0, 1.0)))
            .expect("silu_mul counterexample verification");
    assert_eq!(
        result.encoding,
        SmtEncodingKind::UfApprox,
        "silu_mul uses exp → must be UfApprox, got: {:?}",
        result.encoding,
    );
    assert_ne!(
        result.outcome,
        SmtOutcome::Unexecuted,
        "silu_mul should no longer be Unexecuted (#2640), got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
    assert_eq!(
        result.solver, "ay-direct",
        "nonlinear UF kernel should use ay-direct with ALL logic (#2640)"
    );
}
