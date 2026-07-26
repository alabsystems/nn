// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Scalar kernel builders for DynTensor GPU Elementwise dispatch.
//!
//! Each function creates a [`KernelDef`] (scalar IR graph) that the
//! `Elementwise` dispatch step applies per-element on the GPU. Used by
//! `dyn_tensor_metal.rs` to extend GPU-native coverage beyond the dedicated
//! `BinaryAdd`/`BinaryMul`/`Relu`/`Gelu`/`Sigmoid`/`Tanh` dispatch steps.
//!
//! Complex multi-node kernels (compare, where_cond, maximum, minimum, gelu_erf)
//! are in `dyn_tensor_metal_kernels_complex.rs`.

use nn_dsl::ir::{
    BinOpKind, IRNode, IRNodeKind, KernelDef, NodeId as IrNodeId, Param, ScalarType, UnaryFnKind,
};

#[path = "dyn_tensor_metal_kernels_complex.rs"]
mod complex;
pub(super) use complex::{
    make_atan2_kernel, make_clamp_kernel, make_clamp_max_kernel, make_clamp_min_kernel,
    make_compare_scalar_kernel, make_compare_tensor_kernel, make_gelu_erf_kernel,
    make_maximum_kernel, make_minimum_kernel, make_scalar_binop_kernel, make_where_cond_kernel,
};

/// Build a 1-param scalar kernel: `fn name(x: f32) -> f32 { op(x) }`
pub(super) fn make_unary_kernel(name: &str, op: UnaryFnKind) -> KernelDef {
    KernelDef::new(
        name,
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(IrNodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                IrNodeId::new(1),
                IRNodeKind::UnaryFn {
                    op,
                    input: IrNodeId::new(0),
                },
            ),
        ],
        IrNodeId::new(1),
    )
}

/// Build a 2-param scalar kernel: `fn name(a: f32, b: f32) -> f32 { a op b }`
pub(super) fn make_binop_kernel(name: &str, op: BinOpKind) -> KernelDef {
    KernelDef::new(
        name,
        vec![
            Param::new("a", ScalarType::F32),
            Param::new("b", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(IrNodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(IrNodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                IrNodeId::new(2),
                IRNodeKind::BinOp {
                    op,
                    lhs: IrNodeId::new(0),
                    rhs: IrNodeId::new(1),
                },
            ),
        ],
        IrNodeId::new(2),
    )
}

/// Build `fn neg(x: f32) -> f32 { 0.0 - x }`
pub(super) fn make_neg_kernel() -> KernelDef {
    KernelDef::new(
        "neg",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(IrNodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(IrNodeId::new(1), IRNodeKind::Literal(0.0)),
            IRNode::new(
                IrNodeId::new(2),
                IRNodeKind::BinOp {
                    op: BinOpKind::Sub,
                    lhs: IrNodeId::new(1),
                    rhs: IrNodeId::new(0),
                },
            ),
        ],
        IrNodeId::new(2),
    )
}

/// Build `fn log(x: f32) -> f32 { x.ln() }`
pub(super) fn build_log_kernel() -> KernelDef {
    make_unary_kernel("log", UnaryFnKind::Log)
}

/// Build `fn sqr(x: f32) -> f32 { x * x }`
pub(super) fn make_sqr_kernel() -> KernelDef {
    KernelDef::new(
        "sqr",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(IrNodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                IrNodeId::new(1),
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: IrNodeId::new(0),
                    rhs: IrNodeId::new(0),
                },
            ),
        ],
        IrNodeId::new(1),
    )
}
