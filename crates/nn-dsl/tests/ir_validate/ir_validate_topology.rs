// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Topological ordering and node ID consistency validation tests.
//!
//! Tests that KernelDef::validate() catches forward references,
//! self-references, and mismatched node IDs. Companion to
//! `ir_validate_advanced.rs` (Compare/Select/SumReduce validation).

use nn_dsl::ir::{
    BinOpKind, CompareOpKind, IRError, IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType,
    UnaryFnKind,
};

// ======================== Helpers ========================

fn param(name: &str) -> Param {
    Param::new(name, ScalarType::F32)
}

fn node(id: usize, kind: IRNodeKind) -> IRNode {
    IRNode::new(NodeId::new(id), kind)
}

// ======================== Property: topological ordering ========================

#[test]
fn property_backward_reference_passes() {
    // Node 1 references Node 0 — a valid backward reference (topological order).
    let kernel = KernelDef::new(
        "backward_ref",
        vec![param("x")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),
            node(
                1,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Abs,
                    input: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(1),
    );
    assert!(
        kernel.validate().is_ok(),
        "backward reference should pass (topological order)"
    );
}

#[test]
fn property_self_referencing_node_rejected() {
    // Node 1 references itself (NodeId::new(1)) — a self-reference that violates
    // topological ordering. validate() must reject this with ForwardRef error.
    let kernel = KernelDef::new(
        "self_ref",
        vec![param("x")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),
            node(
                1,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Abs,
                    input: NodeId::new(1), // self-reference
                },
            ),
        ],
        NodeId::new(1),
    );
    kernel
        .validate()
        .expect_err("self-referencing node must fail validation");
}

#[test]
fn property_forward_reference_rejected() {
    // Node 1 references Node 2 (forward ref) — violates topological order.
    let kernel = KernelDef::new(
        "forward_ref",
        vec![param("x")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),
            node(
                1,
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(2), // forward reference
                },
            ),
            node(2, IRNodeKind::Literal(1.0)),
        ],
        NodeId::new(1),
    );
    kernel
        .validate()
        .expect_err("forward reference must fail validation");
}

#[test]
fn property_self_ref_in_binop_both_operands() {
    // Both operands of a BinOp reference the node itself.
    let kernel = KernelDef::new(
        "self_ref_binop",
        vec![param("x")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),
            node(
                1,
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(1), // self-reference
                    rhs: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(1),
    );
    let err = kernel
        .validate()
        .expect_err("BinOp with self-referencing lhs must fail");
    assert!(
        matches!(err, IRError::ForwardRef(a, b) if a == NodeId::new(1) && b == NodeId::new(1)),
        "self-ref: expected ForwardRef(1, 1), got: {err:?}"
    );
}

#[test]
fn property_forward_ref_in_select_cond() {
    // Select condition references a later node.
    let kernel = KernelDef::new(
        "forward_select",
        vec![param("x"), param("y")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),
            node(1, IRNodeKind::Param(1)),
            node(
                2,
                IRNodeKind::Select {
                    cond: NodeId::new(3), // forward reference to Compare node
                    then_val: NodeId::new(0),
                    else_val: NodeId::new(1),
                },
            ),
            node(
                3,
                IRNodeKind::Compare {
                    op: CompareOpKind::Gt,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    let err = kernel
        .validate()
        .expect_err("Select with forward-referencing cond must fail");
    assert!(
        matches!(err, IRError::ForwardRef(a, b) if a == NodeId::new(2) && b == NodeId::new(3)),
        "Select forward-ref: expected ForwardRef(2, 3), got: {err:?}"
    );
}

#[test]
fn property_forward_ref_in_sum_reduce() {
    // SumReduce with a forward reference in its inputs list.
    let kernel = KernelDef::new(
        "forward_reduce",
        vec![param("x")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),
            node(
                1,
                IRNodeKind::SumReduce {
                    inputs: vec![NodeId::new(0), NodeId::new(2)], // NodeId::new(2) is forward
                },
            ),
            node(2, IRNodeKind::Literal(1.0)),
        ],
        NodeId::new(1),
    );
    let err = kernel
        .validate()
        .expect_err("SumReduce with forward reference must fail");
    assert!(
        matches!(err, IRError::ForwardRef(a, b) if a == NodeId::new(1) && b == NodeId::new(2)),
        "SumReduce forward-ref: expected ForwardRef(1, 2), got: {err:?}"
    );
}

#[test]
fn property_empty_nodes_with_output_ref_fails() {
    let kernel = KernelDef::new(
        "empty",
        vec![param("x")],
        ScalarType::F32,
        vec![],
        NodeId::new(0),
    );
    let err = kernel
        .validate()
        .expect_err("empty kernel with output ref should fail");
    assert!(matches!(err, IRError::InvalidNodeRef(id) if id == NodeId::new(0)));
}

// ======================== Property: node ID / index consistency ========================

#[test]
fn property_mismatched_node_id_rejected() {
    // Node at index 0 has id=5 — violates the invariant that nodes[i].id == NodeId(i).
    // Without this check, validate() and codegen would use the wrong indices
    // for topological ordering and MSL variable references.
    let kernel = KernelDef::new(
        "bad_ids",
        vec![param("x")],
        ScalarType::F32,
        vec![IRNode::new(NodeId::new(5), IRNodeKind::Param(0))],
        NodeId::new(5),
    );
    let err = kernel
        .validate()
        .expect_err("mismatched node ID must fail validation");
    assert!(
        matches!(
            err,
            IRError::MismatchedNodeId {
                found,
                expected_index: 0,
            } if found == NodeId::new(5)
        ),
        "expected MismatchedNodeId, got {err:?}"
    );
}

#[test]
fn property_swapped_node_ids_rejected() {
    // Nodes with swapped IDs: index 0 has id=1, index 1 has id=0.
    // This would fool the topological check into thinking backward references
    // are forward references and vice versa.
    let kernel = KernelDef::new(
        "swapped_ids",
        vec![param("x")],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(1), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(0),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Sin,
                    input: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(0),
    );
    let err = kernel
        .validate()
        .expect_err("swapped node IDs must fail validation");
    assert!(
        matches!(
            err,
            IRError::MismatchedNodeId {
                found,
                expected_index: 0,
            } if found == NodeId::new(1)
        ),
        "first mismatched node should be caught, got {err:?}"
    );
}
