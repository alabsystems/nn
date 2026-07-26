// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Algorithm boundary condition tests: checked_constant rejection paths,
//! overflow in constant folding, and Powi boundary conditions.
//!
//! Edge-case tests (MinMax, Powi sign, Rsqrt/Recip, SumReduce mixed-path):
//! algorithm_boundary_edge.rs

use super::common::{ibp_scalar, unary_fn_kernel};
use nn_dsl::ir::{
    BinOpKind, IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType, UnaryFnKind,
};
use nn_verify::{kernel_to_graph_multi, ParamBinding, VerifyError};

// --- checked_constant rejection via constant folding ---

#[test]
fn test_exp_overflow_constant_fold_rejected() {
    // exp(1000.0) overflows f32 → Inf → checked_constant rejects
    let kernel = unary_fn_kernel(UnaryFnKind::Exp);
    let result = kernel_to_graph_multi(&kernel, &[ParamBinding::Constant(1000.0)]);
    match result {
        Err(VerifyError::NonFiniteConstant { value, .. }) => {
            assert!(value.is_infinite(), "should be Inf, got {value}");
        }
        other => panic!("expected NonFiniteConstant, got {other:?}"),
    }
}

#[test]
fn test_sqrt_negative_constant_fold_rejected() {
    // sqrt(-1.0) = NaN → checked_constant rejects
    let kernel = unary_fn_kernel(UnaryFnKind::Sqrt);
    let result = kernel_to_graph_multi(&kernel, &[ParamBinding::Constant(-1.0)]);
    match result {
        Err(VerifyError::NonFiniteConstant { value, .. }) => {
            assert!(value.is_nan(), "should be NaN, got {value}");
        }
        other => panic!("expected NonFiniteConstant, got {other:?}"),
    }
}

#[test]
fn test_recip_zero_constant_fold_rejected() {
    // 1.0 / 0.0 = Inf → checked_constant rejects
    let kernel = unary_fn_kernel(UnaryFnKind::Recip);
    let result = kernel_to_graph_multi(&kernel, &[ParamBinding::Constant(0.0)]);
    match result {
        Err(VerifyError::NonFiniteConstant { value, .. }) => {
            assert!(value.is_infinite(), "should be Inf, got {value}");
        }
        other => panic!("expected NonFiniteConstant, got {other:?}"),
    }
}

#[test]
fn test_rsqrt_negative_constant_fold_rejected() {
    // rsqrt(-1.0) = 1/sqrt(-1) = NaN → checked_constant rejects
    let kernel = unary_fn_kernel(UnaryFnKind::Rsqrt);
    let result = kernel_to_graph_multi(&kernel, &[ParamBinding::Constant(-1.0)]);
    assert!(
        matches!(result, Err(VerifyError::NonFiniteConstant { .. })),
        "rsqrt(-1.0) should be rejected as non-finite, got {result:?}"
    );
}

// --- Overflow via BinOp constant folding ---

#[test]
fn test_mul_overflow_constant_fold_rejected() {
    // f32::MAX * 2.0 overflows → checked_constant rejects
    let kernel = KernelDef::new(
        "mul_overflow",
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
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    let result = kernel_to_graph_multi(
        &kernel,
        &[
            ParamBinding::Constant(f32::MAX),
            ParamBinding::Constant(2.0),
        ],
    );
    assert!(
        matches!(result, Err(VerifyError::NonFiniteConstant { .. })),
        "MAX * 2.0 overflow should be rejected, got {result:?}"
    );
}

#[test]
fn test_div_by_zero_constant_fold_rejected() {
    let kernel = KernelDef::new(
        "div_zero",
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
                IRNodeKind::BinOp {
                    op: BinOpKind::Div,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    let result = kernel_to_graph_multi(
        &kernel,
        &[ParamBinding::Constant(1.0), ParamBinding::Constant(0.0)],
    );
    assert!(
        matches!(result, Err(VerifyError::InternalTranslationError { .. })),
        "1.0 / 0.0 should be rejected with division-by-zero error, got {result:?}"
    );
}

#[test]
fn test_sum_reduce_constant_overflow_rejected() {
    // SumReduce constant folding accumulates with f32 arithmetic.
    // MAX + MAX overflows to Inf and must be rejected by checked_constant.
    let kernel = KernelDef::new(
        "sum_reduce_overflow",
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
                IRNodeKind::SumReduce {
                    inputs: vec![NodeId::new(0), NodeId::new(1)],
                },
            ),
        ],
        NodeId::new(2),
    );

    let result = kernel_to_graph_multi(
        &kernel,
        &[
            ParamBinding::Constant(f32::MAX),
            ParamBinding::Constant(f32::MAX),
        ],
    );
    match result {
        Err(VerifyError::NonFiniteConstant { value, context }) => {
            assert!(
                value.is_infinite(),
                "MAX + MAX should overflow, got {value}"
            );
            assert!(
                context.contains("SumReduce"),
                "expected SumReduce checked_constant context, got {context}"
            );
        }
        other => panic!("expected NonFiniteConstant overflow from SumReduce, got {other:?}"),
    }
}

// --- Powi boundary conditions ---

#[test]
fn test_powi_large_exponent_overflow_constant_fold_rejected() {
    // exp=100 exceeds POWI_MAX_EXPONENT (64), so IR validation rejects before
    // constant folding is attempted. This is the correct (stricter) behavior.
    let kernel = KernelDef::new(
        "powi_overflow",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::Powi {
                    base: NodeId::new(0),
                    exp: 100,
                },
            ),
        ],
        NodeId::new(1),
    );
    let result = kernel_to_graph_multi(&kernel, &[ParamBinding::Constant(100.0)]);
    assert!(
        matches!(result, Err(VerifyError::IrValidation(..))),
        "exp=100 should be rejected by IR validation, got {result:?}"
    );
}

#[test]
fn test_powi_valid_exponent_overflow_constant_fold_rejected() {
    // exp=64 is within POWI_MAX_EXPONENT, but 1e10^64 overflows f32 → Inf.
    // This exercises the constant-fold overflow path (checked_constant).
    let kernel = KernelDef::new(
        "powi_valid_exp_overflow",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::Powi {
                    base: NodeId::new(0),
                    exp: 64,
                },
            ),
        ],
        NodeId::new(1),
    );
    let result = kernel_to_graph_multi(&kernel, &[ParamBinding::Constant(1e10)]);
    assert!(
        matches!(result, Err(VerifyError::NonFiniteConstant { .. })),
        "1e10^64 overflow should be rejected by checked_constant, got {result:?}"
    );
}

#[test]
fn test_powi_zero_exponent_constant_fold_gives_one() {
    // x.powi(0) = 1.0 for any finite x
    let kernel = KernelDef::new(
        "powi_zero",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::Powi {
                    base: NodeId::new(0),
                    exp: 0,
                },
            ),
        ],
        NodeId::new(1),
    );
    // With constant binding: 42.0.powi(0) = 1.0 → constant fold should produce exactly 1.0.
    // Verify via IBP: for a fully-constant graph the output bounds equal the folded value.
    let graph = kernel_to_graph_multi(&kernel, &[ParamBinding::Constant(42.0)])
        .expect("powi(0) should be valid");
    let (lo, hi) = ibp_scalar(&graph, 0.0, 0.0);
    assert!(
        (lo - 1.0).abs() < 1e-6 && (hi - 1.0).abs() < 1e-6,
        "42.0.powi(0) should fold to 1.0, got bounds [{lo}, {hi}]"
    );
}
