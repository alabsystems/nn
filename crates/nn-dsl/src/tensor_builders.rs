// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared tensor-level kernel builder helpers.
//!
//! These helpers construct scalar `KernelDef` and tensor `TensorNode` building
//! blocks used by multiple norm kernel modules (`instance_norm`, `layer_norm`,
//! `rms_norm`). Centralizing them here eliminates ~500 lines of pure
//! duplication and ensures bug fixes propagate uniformly.
//!
//! See `designs/2026-02-27-kernel-tensor-builders.md`.

use crate::ir::{
    BinOpKind, BinaryFnKind, CompareOpKind, IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId,
    Param, ScalarType, UnaryFnKind,
};
use crate::tensor_ir::{BroadcastAlignment, ReduceOp, TensorNode, TensorNodeId, TensorOpKind};

// --- Scalar kernel builders ---

/// Build a 1-param scalar kernel applying a unary function.
#[must_use]
pub(crate) fn unary_kernel(name: &str, op: UnaryFnKind) -> KernelDef {
    KernelDef {
        name: name.into(),
        params: vec![Param {
            name: "x".into(),
            ty: ScalarType::F32,
        }],
        return_type: ScalarType::F32,
        nodes: vec![
            IRNode {
                id: NodeId::new(0),
                kind: IRNodeKind::Param(0),
            },
            IRNode {
                id: NodeId::new(1),
                kind: IRNodeKind::UnaryFn {
                    op,
                    input: NodeId::new(0),
                },
            },
        ],
        output: NodeId::new(1),
    }
}

/// Build a 2-param scalar kernel applying a binary op.
#[must_use]
pub(crate) fn binop_kernel(name: &str, op: BinOpKind) -> KernelDef {
    KernelDef {
        name: name.into(),
        params: vec![
            Param {
                name: "a".into(),
                ty: ScalarType::F32,
            },
            Param {
                name: "b".into(),
                ty: ScalarType::F32,
            },
        ],
        return_type: ScalarType::F32,
        nodes: vec![
            IRNode {
                id: NodeId::new(0),
                kind: IRNodeKind::Param(0),
            },
            IRNode {
                id: NodeId::new(1),
                kind: IRNodeKind::Param(1),
            },
            IRNode {
                id: NodeId::new(2),
                kind: IRNodeKind::BinOp {
                    op,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            },
        ],
        output: NodeId::new(2),
    }
}

/// Build a 2-param scalar kernel applying min or max.
#[must_use]
pub(crate) fn minmax_kernel(name: &str, op: MinMaxKind) -> KernelDef {
    KernelDef {
        name: name.into(),
        params: vec![
            Param {
                name: "a".into(),
                ty: ScalarType::F32,
            },
            Param {
                name: "b".into(),
                ty: ScalarType::F32,
            },
        ],
        return_type: ScalarType::F32,
        nodes: vec![
            IRNode {
                id: NodeId::new(0),
                kind: IRNodeKind::Param(0),
            },
            IRNode {
                id: NodeId::new(1),
                kind: IRNodeKind::Param(1),
            },
            IRNode {
                id: NodeId::new(2),
                kind: IRNodeKind::MinMax {
                    op,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            },
        ],
        output: NodeId::new(2),
    }
}

/// Build a 2-param scalar kernel applying a binary function (function-call syntax).
#[must_use]
pub(crate) fn binary_fn_kernel(name: &str, op: BinaryFnKind) -> KernelDef {
    KernelDef {
        name: name.into(),
        params: vec![
            Param {
                name: "a".into(),
                ty: ScalarType::F32,
            },
            Param {
                name: "b".into(),
                ty: ScalarType::F32,
            },
        ],
        return_type: ScalarType::F32,
        nodes: vec![
            IRNode {
                id: NodeId::new(0),
                kind: IRNodeKind::Param(0),
            },
            IRNode {
                id: NodeId::new(1),
                kind: IRNodeKind::Param(1),
            },
            IRNode {
                id: NodeId::new(2),
                kind: IRNodeKind::BinaryFn {
                    op,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            },
        ],
        output: NodeId::new(2),
    }
}

/// Build a 2-param scalar kernel: `select(a <op> b, 1.0, 0.0)`.
///
/// Compare produces Bool; Select converts to F32 for the kernel return.
/// Used by `TraceOp::Compare` compilation (#3214).
#[must_use]
pub(crate) fn compare_select_kernel(name: &str, op: CompareOpKind) -> KernelDef {
    KernelDef {
        name: name.into(),
        params: vec![
            Param {
                name: "a".into(),
                ty: ScalarType::F32,
            },
            Param {
                name: "b".into(),
                ty: ScalarType::F32,
            },
        ],
        return_type: ScalarType::F32,
        nodes: vec![
            IRNode {
                id: NodeId::new(0),
                kind: IRNodeKind::Param(0),
            },
            IRNode {
                id: NodeId::new(1),
                kind: IRNodeKind::Param(1),
            },
            IRNode {
                id: NodeId::new(2),
                kind: IRNodeKind::Compare {
                    op,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            },
            IRNode {
                id: NodeId::new(3),
                kind: IRNodeKind::Literal(1.0),
            },
            IRNode {
                id: NodeId::new(4),
                kind: IRNodeKind::Literal(0.0),
            },
            IRNode {
                id: NodeId::new(5),
                kind: IRNodeKind::Select {
                    cond: NodeId::new(2),
                    then_val: NodeId::new(3),
                    else_val: NodeId::new(4),
                },
            },
        ],
        output: NodeId::new(5),
    }
}

/// `square(x) = x * x` — 1-param, self-multiply.
#[must_use]
pub(crate) fn square_kernel() -> KernelDef {
    KernelDef {
        name: "square".into(),
        params: vec![Param {
            name: "x".into(),
            ty: ScalarType::F32,
        }],
        return_type: ScalarType::F32,
        nodes: vec![
            IRNode {
                id: NodeId::new(0),
                kind: IRNodeKind::Param(0),
            },
            IRNode {
                id: NodeId::new(1),
                kind: IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(0),
                },
            },
        ],
        output: NodeId::new(1),
    }
}

// --- Tensor node helpers ---

/// Build a `TensorOpKind::Input` node.
#[must_use]
pub(crate) fn input_node(id: usize, name: &str, shape: &[usize]) -> TensorNode {
    TensorNode {
        id: TensorNodeId::new(id),
        kind: TensorOpKind::Input {
            name: name.into(),
            shape: shape.to_vec(),
        },
        shape: shape.to_vec(),
    }
}

/// Build a `TensorOpKind::Reduce` node.
#[must_use]
pub(crate) fn reduce_node(
    id: usize,
    op: ReduceOp,
    input: usize,
    axis: usize,
    shape: &[usize],
) -> TensorNode {
    TensorNode {
        id: TensorNodeId::new(id),
        kind: TensorOpKind::Reduce {
            op,
            input: TensorNodeId::new(input),
            axis,
            keepdim: false,
        },
        shape: shape.to_vec(),
    }
}

/// Build a `TensorOpKind::Elementwise` node.
#[must_use]
pub(crate) fn elementwise_node(
    id: usize,
    kernel: KernelDef,
    inputs: &[usize],
    shape: &[usize],
) -> TensorNode {
    TensorNode {
        id: TensorNodeId::new(id),
        kind: TensorOpKind::Elementwise {
            kernel,
            inputs: inputs.iter().map(|&i| TensorNodeId::new(i)).collect(),
        },
        shape: shape.to_vec(),
    }
}

/// Build a `TensorOpKind::Broadcast` node.
#[must_use]
pub(crate) fn broadcast_node(
    id: usize,
    input: usize,
    target: &[usize],
    alignment: BroadcastAlignment,
) -> TensorNode {
    TensorNode {
        id: TensorNodeId::new(id),
        kind: TensorOpKind::Broadcast {
            input: TensorNodeId::new(input),
            target_shape: target.to_vec(),
            alignment,
        },
        shape: target.to_vec(),
    }
}
