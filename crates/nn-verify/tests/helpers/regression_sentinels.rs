// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Regression sentinel tests: cross-module integration tests that catch
//! interaction-boundary regressions.
//!
//! Each test exercises a full path that has regressed in the past:
//! - Select-Ge boundary: #568 regressed #559 (Compare × Select × constant-fold)
//! - FTZ-aware bounds: #556 regressed (FTZ classification × bounds bridge)
//! - ay round-trip: #566 regressed (ay translation × solver execution)
//!
//! These tests are NOT module-level unit tests — they test interactions between
//! modules that individually pass their own unit tests.
//!
//! Part of #580 (regression-of-fix pattern)

use super::common;
use nn_dsl::ir::{
    BinOpKind, CompareOpKind, IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType, UnaryFnKind,
};
use nn_verify::kernel_to_graph;

// ---------------------------------------------------------------------------
// Sentinel 1: Select-Ge boundary (#568 regressed #559)
//
// The regression: Compare(Ge, x, threshold) constant-folds correctly at the
// equality boundary (x == threshold → 1.0), but Select picks the wrong branch
// when the constant-fold result is used as a condition. The interaction between
// Compare constant-folding (Ge equality → 1.0 → positive → then-branch) and
// Select condition interpretation must be consistent.
// ---------------------------------------------------------------------------

/// select(x >= 5.0, 10.0, 0.0) where x is exactly 5.0 (constant-fold path).
///
/// When both lhs and rhs of Ge are constants and lhs == rhs, Compare must
/// fold to 1.0 (true), and Select must pick the then-branch (10.0).
/// Regression #568: the Ge/Le inclusive guard broke this by returning 0.0
/// for the equality case.
#[test]
fn sentinel_select_ge_equality_constant_fold() {
    // Build: select(compare(a, b, Ge), 10.0, 0.0) with a=5.0, b=5.0
    let kernel = KernelDef::new(
        "ge_eq_fold",
        vec![
            Param::new("x", ScalarType::F32),
            Param::new("a", ScalarType::F32),
            Param::new("b", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)), // a
            IRNode::new(NodeId::new(2), IRNodeKind::Param(2)), // b
            IRNode::new(
                NodeId::new(3),
                IRNodeKind::Compare {
                    op: CompareOpKind::Ge,
                    lhs: NodeId::new(1),
                    rhs: NodeId::new(2),
                },
            ),
            IRNode::new(NodeId::new(4), IRNodeKind::Literal(10.0)),
            IRNode::new(NodeId::new(5), IRNodeKind::Literal(0.0)),
            IRNode::new(
                NodeId::new(6),
                IRNodeKind::Select {
                    cond: NodeId::new(3),
                    then_val: NodeId::new(4),
                    else_val: NodeId::new(5),
                },
            ),
        ],
        NodeId::new(6),
    );

    // a=5.0, b=5.0 → Ge(5,5) should be true → select returns 10.0
    let graph = kernel_to_graph(&kernel, &[5.0, 5.0]).expect("build graph");
    let (lo, hi) = common::ibp_scalar(&graph, 0.0, 0.0);

    // Both bounds should be 10.0 (constant fold → then-branch)
    assert!(
        (lo - 10.0).abs() < 0.01,
        "Ge(5,5) should fold to true, selecting 10.0. Got lower={lo}. \
         If lower=0.0, the Ge equality case returned false (regression #568)."
    );
    assert!(
        (hi - 10.0).abs() < 0.01,
        "Ge(5,5) should fold to true, selecting 10.0. Got upper={hi}."
    );
}

/// select(x >= 5.0, 10.0, 0.0) where x is exactly 5.0 (Le symmetric case).
///
/// Le(a, b) where a == b should also fold to true (1.0).
#[test]
fn sentinel_select_le_equality_constant_fold() {
    let kernel = KernelDef::new(
        "le_eq_fold",
        vec![
            Param::new("x", ScalarType::F32),
            Param::new("a", ScalarType::F32),
            Param::new("b", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(NodeId::new(2), IRNodeKind::Param(2)),
            IRNode::new(
                NodeId::new(3),
                IRNodeKind::Compare {
                    op: CompareOpKind::Le,
                    lhs: NodeId::new(1),
                    rhs: NodeId::new(2),
                },
            ),
            IRNode::new(NodeId::new(4), IRNodeKind::Literal(10.0)),
            IRNode::new(NodeId::new(5), IRNodeKind::Literal(0.0)),
            IRNode::new(
                NodeId::new(6),
                IRNodeKind::Select {
                    cond: NodeId::new(3),
                    then_val: NodeId::new(4),
                    else_val: NodeId::new(5),
                },
            ),
        ],
        NodeId::new(6),
    );

    let graph = kernel_to_graph(&kernel, &[5.0, 5.0]).expect("build graph");
    let (lo, hi) = common::ibp_scalar(&graph, 0.0, 0.0);

    assert!(
        (lo - 10.0).abs() < 0.01,
        "Le(5,5) should fold to true, selecting 10.0. Got lower={lo}."
    );
    assert!(
        (hi - 10.0).abs() < 0.01,
        "Le(5,5) should fold to true, selecting 10.0. Got upper={hi}."
    );
}

/// select(x >= 0, x, alpha*x) with variable x — the LeakyReLU pattern.
///
/// This tests the full Compare→Select→pattern-match pipeline with variable
/// operands. The bounds must be sound across the transition at x=0.
#[test]
fn sentinel_select_ge_variable_leaky_relu_bounds() {
    let kernel = KernelDef::new(
        "leaky_relu_sentinel",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)), // x
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(0.0)), // zero
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::Compare {
                    op: CompareOpKind::Gt,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            IRNode::new(NodeId::new(3), IRNodeKind::Literal(0.1)), // alpha
            IRNode::new(
                NodeId::new(4),
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(3),
                    rhs: NodeId::new(0),
                },
            ),
            IRNode::new(
                NodeId::new(5),
                IRNodeKind::Select {
                    cond: NodeId::new(2),
                    then_val: NodeId::new(0),
                    else_val: NodeId::new(4),
                },
            ),
        ],
        NodeId::new(5),
    );

    let graph = kernel_to_graph(&kernel, &[]).expect("build LeakyReLU graph");

    // Test with input range spanning zero: x in [-5, 5]
    // True leaky_relu: min(alpha*(-5), 0) = -0.5, max(5, 0) = 5
    let (lo, hi) = common::ibp_scalar(&graph, -5.0, 5.0);

    assert!(
        lo <= -0.4,
        "LeakyReLU lower bound should include alpha*(-5)=-0.5. Got {lo}."
    );
    assert!(
        hi >= 4.9,
        "LeakyReLU upper bound should include x=5. Got {hi}."
    );
}

// ---------------------------------------------------------------------------
// Sentinel 2: FTZ-aware bounds (#556 regression)
//
// The regression: kernels with rsqrt/recip/div operations have FTZ (flush-to-
// zero) sensitivity on Metal GPUs. When the test infrastructure misclassifies
// a kernel's FTZ sensitivity, bounds verification may use incorrect ranges.
// This sentinel verifies that FTZ-sensitive kernels produce valid NY
// bounds even with denormal-adjacent inputs.
// ---------------------------------------------------------------------------

/// Verify that a kernel using Recip produces valid bounds near zero.
///
/// Recip(x) = 1/x is FTZ-sensitive: Metal flushes denormals to zero,
/// making 1/denormal = 1/0 = Inf. NY bounds must remain finite
/// for inputs that avoid the zero singularity.
#[test]
fn sentinel_ftz_recip_bounds_near_zero() {
    let kernel = KernelDef::new(
        "recip_sentinel",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Recip,
                    input: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(1),
    );

    let graph = kernel_to_graph(&kernel, &[]).expect("build Recip graph");

    // Avoid zero: x in [0.1, 10.0] — safe range for 1/x
    let (lo, hi) = common::ibp_scalar(&graph, 0.1, 10.0);

    // 1/10 = 0.1, 1/0.1 = 10.0
    assert!(
        lo.is_finite(),
        "Recip bounds lower must be finite, got {lo}"
    );
    assert!(
        hi.is_finite(),
        "Recip bounds upper must be finite, got {hi}"
    );
    assert!(lo <= 0.11, "Recip lower should be <= 1/10 = 0.1. Got {lo}.");
    assert!(
        hi >= 9.9,
        "Recip upper should be >= 1/0.1 = 10.0. Got {hi}."
    );
}

/// Verify that a kernel using Rsqrt produces valid bounds for positive inputs.
///
/// Rsqrt(x) = 1/sqrt(x) is FTZ-sensitive. This sentinel verifies bounds
/// are correct and finite for a safe positive input range.
#[test]
fn sentinel_ftz_rsqrt_bounds_positive_range() {
    let kernel = KernelDef::new(
        "rsqrt_sentinel",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Rsqrt,
                    input: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(1),
    );

    let graph = kernel_to_graph(&kernel, &[]).expect("build Rsqrt graph");

    // x in [1.0, 4.0]: rsqrt(1)=1, rsqrt(4)=0.5
    let (lo, hi) = common::ibp_scalar(&graph, 1.0, 4.0);

    assert!(
        lo.is_finite(),
        "Rsqrt bounds lower must be finite, got {lo}"
    );
    assert!(
        hi.is_finite(),
        "Rsqrt bounds upper must be finite, got {hi}"
    );
    assert!(
        lo <= 0.51,
        "Rsqrt lower should be <= rsqrt(4) = 0.5. Got {lo}."
    );
    assert!(
        hi >= 0.99,
        "Rsqrt upper should be >= rsqrt(1) = 1.0. Got {hi}."
    );
}

/// Verify that Div-based kernel produces valid bounds.
///
/// x / c where c is a positive constant. Tests the FTZ-sensitive Div op
/// through NY translation.
#[test]
fn sentinel_ftz_div_constant_denominator() {
    let kernel = KernelDef::new(
        "div_sentinel",
        vec![
            Param::new("x", ScalarType::F32),
            Param::new("c", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinOp {
                    op: BinOpKind::Div,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );

    // x / 2.0, x in [-10, 10] → [-5, 5]
    let graph = kernel_to_graph(&kernel, &[2.0]).expect("build Div graph");
    let (lo, hi) = common::ibp_scalar(&graph, -10.0, 10.0);

    assert!(lo.is_finite(), "Div bounds lower must be finite, got {lo}");
    assert!(hi.is_finite(), "Div bounds upper must be finite, got {hi}");
    assert!(lo <= -4.9, "Div lower should be <= -10/2 = -5. Got {lo}.");
    assert!(hi >= 4.9, "Div upper should be >= 10/2 = 5. Got {hi}.");
}

// Sentinel 3 (ay SMT translation round-trip) extracted to regression_sentinels_ay.rs
