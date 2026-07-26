// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use nn_dsl::{
    ir_pretty_print, BinOpKind, IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType,
};

fn simple_add_kernel() -> KernelDef {
    KernelDef::new(
        "add",
        vec![
            Param::new("x", ScalarType::F32),
            Param::new("y", ScalarType::F32),
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
fn test_validate_rejects_empty_sum_reduce() {
    let mut kernel = simple_add_kernel();
    kernel.nodes.push(IRNode::new(
        NodeId::new(3),
        IRNodeKind::SumReduce { inputs: Vec::new() },
    ));
    kernel.output = NodeId::new(3);
    let err = kernel.validate().expect_err("empty sum-reduce should fail");
    assert!(
        err.to_string().contains("must have at least one input"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_ir_pretty_print_sum_reduce() {
    let kernel = KernelDef::new(
        "sum3",
        vec![
            Param::new("a", ScalarType::F32),
            Param::new("b", ScalarType::F32),
            Param::new("c", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(NodeId::new(2), IRNodeKind::Param(2)),
            IRNode::new(
                NodeId::new(3),
                IRNodeKind::SumReduce {
                    inputs: vec![NodeId::new(0), NodeId::new(1), NodeId::new(2)],
                },
            ),
        ],
        NodeId::new(3),
    );
    let pp = ir_pretty_print(&kernel);
    assert!(
        pp.contains("sum_reduce(%0, %1, %2)"),
        "pretty-print should include sum-reduce node, got:\n{pp}"
    );
}
