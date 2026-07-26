// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for HIP elementwise IR emission (extracted from emit_elementwise.rs).

use super::*;
use nn_dsl::codegen_shared::powi_stmts;
use nn_dsl::{ir::Param, IRNode, IRNodeKind, KernelDef, NodeId};

fn make_add_kernel() -> KernelDef {
    KernelDef::new(
        "nn_add",
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
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    )
}

#[test]
fn test_elementwise_add_kernel() {
    let kernel = make_add_kernel();
    let src = emit_elementwise_hip(&kernel).unwrap();
    assert!(src.contains("__device__"));
    assert!(src.contains("_nn_nn_add"));
    assert!(src.contains("extern \"C\" __global__ void nn_add_kernel"));
    assert!(src.contains("a[tid]"));
    assert!(src.contains("b[tid]"));
    assert!(src.contains("a + b"));
}

#[test]
fn test_elementwise_with_literal() {
    let kernel = KernelDef::new(
        "scale",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(2.0)),
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
    let src = emit_elementwise_hip(&kernel).unwrap();
    assert!(src.contains("2.0"));
    assert!(src.contains("x * t1"));
}

#[test]
fn test_elementwise_unary_sin() {
    let kernel = KernelDef::new(
        "apply_sin",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Sin,
                    input: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(1),
    );
    let src = emit_elementwise_hip(&kernel).unwrap();
    assert!(src.contains("sinf(x)"));
}

// --- round-ties-to-even test ---

#[test]
fn test_elementwise_round_uses_rintf() {
    // rintf is IEEE 754 round-ties-to-even (matches MSL metal::rint and PyTorch torch.round).
    // roundf would be wrong — it uses round-ties-away-from-zero.
    let kernel = KernelDef::new(
        "apply_round",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Round,
                    input: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(1),
    );
    let src = emit_elementwise_hip(&kernel).unwrap();
    assert!(
        src.contains("rintf(x)"),
        "must use rintf (ties-to-even), not roundf (ties-away)"
    );
    assert!(
        !src.contains("roundf"),
        "roundf has wrong rounding semantics"
    );
}

// --- powi tests ---

#[test]
fn test_powi_zero() {
    let s = powi_stmts("x", 0, "float", 5);
    assert!(s.contains("float(1)"));
}

#[test]
fn test_powi_positive_small() {
    let s = powi_stmts("x", 2, "float", 5);
    assert!(s.contains("x * x"));
}

#[test]
fn test_powi_negative() {
    let s = powi_stmts("x", -2, "float", 5);
    assert!(s.contains("float(1) / (x * x)"));
}

#[test]
fn test_powi_large() {
    let s = powi_stmts("x", 8, "float", 5);
    // Shared powi_stmts uses t{tid}_p{power} naming convention.
    assert!(s.contains("t5_p2"));
    assert!(s.contains("t5_p4"));
    assert!(s.contains("t5_p8"));
}
