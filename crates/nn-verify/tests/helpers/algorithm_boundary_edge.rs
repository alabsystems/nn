// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Edge-case algorithm boundary tests: MinMax, Powi sign behavior, Rsqrt/Recip
//! boundary conditions, and SumReduce mixed-path overflow.

use super::common::{ibp_scalar, square_kernel, unary_fn_kernel};
use nn_dsl::ir::{IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId, Param, ScalarType};
use nn_verify::{kernel_to_graph_multi, ParamBinding, VerifyError};

// --- MinMax boundary conditions ---

#[test]
fn test_minmax_preserves_finite_constants() {
    // min(MAX, -MAX) = -MAX, which is finite and should not be rejected
    let kernel = KernelDef::new(
        "minmax_extreme",
        vec![
            Param::new("a", ScalarType::F32),
            Param::new("b", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::MinMax {
                    op: MinMaxKind::Min,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    let graph = kernel_to_graph_multi(
        &kernel,
        &[
            ParamBinding::Constant(f32::MAX),
            ParamBinding::Constant(f32::MIN),
        ],
    )
    .expect("min(MAX, MIN) is finite and should succeed");
    // min(MAX, MIN) = MIN = -3.4028235e38, verify via IBP
    let (lo, hi) = ibp_scalar(&graph, 0.0, 0.0);
    assert!(
        lo == f32::MIN && hi == f32::MIN,
        "min(MAX, MIN) should fold to f32::MIN, got bounds [{lo}, {hi}]"
    );
}

// --- Square constant fold: exact value ---

#[test]
fn test_square_constant_fold_exact() {
    // 7.0 * 7.0 = 49.0 should be computed exactly by constant folding.
    // Verify via IBP: for a fully-constant graph the output bounds equal the folded value.
    let kernel = square_kernel();
    let graph = kernel_to_graph_multi(&kernel, &[ParamBinding::Constant(7.0)]).expect("7^2 = 49");
    let (lo, hi) = ibp_scalar(&graph, 0.0, 0.0);
    assert!(
        (lo - 49.0).abs() < 1e-6 && (hi - 49.0).abs() < 1e-6,
        "7.0^2 should fold to 49.0, got bounds [{lo}, {hi}]"
    );
}

// --- Non-finite constant param rejected at entry ---

#[test]
fn test_nan_constant_param_rejected() {
    let kernel = unary_fn_kernel(nn_dsl::ir::UnaryFnKind::Abs);
    let result = kernel_to_graph_multi(&kernel, &[ParamBinding::Constant(f32::NAN)]);
    assert!(
        matches!(result, Err(VerifyError::NonFiniteConstant { .. })),
        "NaN constant param should be rejected at entry, got {result:?}"
    );
}

#[test]
fn test_neg_infinity_constant_param_rejected() {
    let kernel = unary_fn_kernel(nn_dsl::ir::UnaryFnKind::Abs);
    let result = kernel_to_graph_multi(&kernel, &[ParamBinding::Constant(f32::NEG_INFINITY)]);
    assert!(
        matches!(result, Err(VerifyError::NonFiniteConstant { .. })),
        "NEG_INFINITY constant param should be rejected at entry, got {result:?}"
    );
}

// --- Rsqrt boundary conditions ---

#[test]
fn test_rsqrt_zero_constant_fold_rejected() {
    // rsqrt(0.0) = 1/sqrt(0) = 1/0 = Inf → checked_constant rejects.
    // This is distinct from the rsqrt(-1) NaN case: rsqrt(0) produces
    // Infinity (division by zero), not NaN.
    let kernel = unary_fn_kernel(nn_dsl::ir::UnaryFnKind::Rsqrt);
    let result = kernel_to_graph_multi(&kernel, &[ParamBinding::Constant(0.0)]);
    match result {
        Err(VerifyError::NonFiniteConstant { value, .. }) => {
            assert!(
                value.is_infinite(),
                "rsqrt(0) should produce Inf, got {value}"
            );
        }
        other => panic!("expected NonFiniteConstant for rsqrt(0), got {other:?}"),
    }
}

// --- Recip negative-zero boundary ---

#[test]
fn test_recip_neg_zero_constant_fold_rejected() {
    // 1.0 / -0.0 = -Inf in IEEE 754 → checked_constant rejects.
    // This is the negative-zero companion to test_recip_zero_constant_fold_rejected.
    let kernel = unary_fn_kernel(nn_dsl::ir::UnaryFnKind::Recip);
    let result = kernel_to_graph_multi(&kernel, &[ParamBinding::Constant(-0.0)]);
    match result {
        Err(VerifyError::NonFiniteConstant { value, .. }) => {
            assert!(
                value.is_infinite(),
                "recip(-0.0) should produce -Inf, got {value}"
            );
            assert!(
                value.is_sign_negative(),
                "recip(-0.0) should produce negative infinity, got {value}"
            );
        }
        other => panic!("expected NonFiniteConstant for recip(-0.0), got {other:?}"),
    }
}

// --- Powi sign behavior ---

#[test]
fn test_powi_negative_base_even_exponent_positive() {
    // (-3.0).powi(2) = 9.0 — negative base with even exponent gives positive result.
    let kernel = KernelDef::new(
        "powi_neg_even",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::Powi {
                    base: NodeId::new(0),
                    exp: 2,
                },
            ),
        ],
        NodeId::new(1),
    );
    let graph =
        kernel_to_graph_multi(&kernel, &[ParamBinding::Constant(-3.0)]).expect("(-3)^2 = 9");
    let (lo, hi) = ibp_scalar(&graph, 0.0, 0.0);
    assert!(
        (lo - 9.0).abs() < 1e-6 && (hi - 9.0).abs() < 1e-6,
        "(-3.0).powi(2) should fold to 9.0, got bounds [{lo}, {hi}]"
    );
}

#[test]
fn test_powi_negative_base_odd_exponent_negative() {
    // (-2.0).powi(3) = -8.0 — negative base with odd exponent gives negative result.
    let kernel = KernelDef::new(
        "powi_neg_odd",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::Powi {
                    base: NodeId::new(0),
                    exp: 3,
                },
            ),
        ],
        NodeId::new(1),
    );
    let graph =
        kernel_to_graph_multi(&kernel, &[ParamBinding::Constant(-2.0)]).expect("(-2)^3 = -8");
    let (lo, hi) = ibp_scalar(&graph, 0.0, 0.0);
    assert!(
        (lo - (-8.0)).abs() < 1e-6 && (hi - (-8.0)).abs() < 1e-6,
        "(-2.0).powi(3) should fold to -8.0, got bounds [{lo}, {hi}]"
    );
}

// --- SumReduce mixed-path constant overflow ---

#[test]
fn test_sum_reduce_mixed_constant_overflow_rejected() {
    // sum_reduce([x, MAX_literal, MAX_literal]) where x is a variable.
    // The two MAX literals sum to Inf during constant accumulation in the
    // mixed path. This must be caught even though var_refs is non-empty.
    let kernel = KernelDef::new(
        "sum_reduce_mixed_overflow",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(f64::from(f32::MAX))),
            IRNode::new(NodeId::new(2), IRNodeKind::Literal(f64::from(f32::MAX))),
            IRNode::new(
                NodeId::new(3),
                IRNodeKind::SumReduce {
                    inputs: vec![NodeId::new(0), NodeId::new(1), NodeId::new(2)],
                },
            ),
        ],
        NodeId::new(3),
    );

    let result = kernel_to_graph_multi(&kernel, &[ParamBinding::Variable]);
    match result {
        Err(VerifyError::NonFiniteConstant { value, context }) => {
            assert!(
                value.is_infinite(),
                "MAX + MAX should overflow, got {value}"
            );
            assert!(
                context.contains("SumReduce"),
                "expected SumReduce context, got {context}"
            );
        }
        other => panic!("expected NonFiniteConstant from mixed SumReduce overflow, got {other:?}"),
    }
}

/// Regression: variable / constant(0.0) must be rejected as InternalTranslationError,
/// not silently create a DivConstant(0.0) layer. Guards added in 94efb7e.
///
/// This tests the `translate_var_const` Div guard specifically (not the constant-fold
/// path tested by `test_div_by_zero_constant_fold_rejected`).
#[test]
fn test_div_by_zero_var_const_rejected() {
    let kernel = KernelDef::new(
        "div_var_zero",
        vec![
            Param::new("x", ScalarType::F32),
            Param::new("zero", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinOp {
                    op: nn_dsl::ir::BinOpKind::Div,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    // Variable / Constant(0.0) — exercises translate_var_const Div guard
    let result = kernel_to_graph_multi(
        &kernel,
        &[ParamBinding::Variable, ParamBinding::Constant(0.0)],
    );
    assert!(
        matches!(result, Err(VerifyError::InternalTranslationError { .. })),
        "variable / constant(0.0) should be rejected, got {result:?}"
    );
}
