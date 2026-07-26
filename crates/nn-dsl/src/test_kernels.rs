// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared kernel definitions for testing across nn crates.
//!
//! Exposed under the `test-utils` feature flag (or `#[cfg(test)]`).
//! Not part of the public API — test-only helpers.
//!
//! # Usage
//!
//! From within `nn-dsl` tests:
//! ```text
//! use nn_dsl::test_kernels::{square_kernel, parse_kernel};
//! ```
//!
//! From `nn-verify` integration tests (requires `test-utils` feature):
//! ```text
//! use nn_dsl::test_kernels::{square_kernel, unary_fn_kernel};
//! ```

use crate::ir::{
    BinOpKind, CompareOpKind, IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType, UnaryFnKind,
};
use crate::lower::Lowerer;
use crate::snake::SNAKE_MIN_ALPHA;

/// Parse a Rust fn source string and lower to `KernelDef`.
///
/// Panics on parse or lower errors — intended for tests only.
#[must_use]
pub fn parse_kernel(src: &str) -> KernelDef {
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    Lowerer::lower_fn(&func).expect("lower")
}

/// Snake kernel matching the production `build_snake_scalar_kernel()`.
///
/// Uses [`SNAKE_MIN_ALPHA`] via `format!` — same pattern as production —
/// so the clamp value stays in sync automatically (see #325).
#[must_use]
pub fn snake_kernel() -> KernelDef {
    parse_kernel(&format!(
        "fn snake(x: f32, alpha: f32) -> f32 {{ \
         let a = alpha.max({SNAKE_MIN_ALPHA:e}); \
         x + (1.0 / a) * (a * x).sin().powi(2) }}",
    ))
}

/// `fn id(x: f32) -> f32 { x }` — identity pass-through.
#[must_use]
pub fn identity_kernel() -> KernelDef {
    parse_kernel("fn id(x: f32) -> f32 { x }")
}

/// `fn exp_grow(x: f32) -> f32 { x.exp() }`
#[must_use]
pub fn exp_kernel() -> KernelDef {
    parse_kernel("fn exp_grow(x: f32) -> f32 { x.exp() }")
}

/// `fn square(x: f32) -> f32 { x * x }` — single-param self-multiply.
#[must_use]
pub fn square_kernel() -> KernelDef {
    parse_kernel("fn square(x: f32) -> f32 { x * x }")
}

/// `fn sub(a: f32, b: f32) -> f32 { a - b }` — two-param subtraction.
#[must_use]
pub fn sub_kernel() -> KernelDef {
    parse_kernel("fn sub(a: f32, b: f32) -> f32 { a - b }")
}

/// Build a 1-param f32 kernel applying a unary function: `fn f(x) -> f32 { op(x) }`
///
/// Hand-built: takes a runtime `op` parameter that cannot be expressed as
/// a static Rust source string for `parse_kernel`. See #144 AC1.
#[must_use]
pub fn unary_fn_kernel(op: UnaryFnKind) -> KernelDef {
    KernelDef::new(
        format!("unary_{op:?}"),
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::UnaryFn {
                    op,
                    input: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(1),
    )
}

/// Build a 2-param f32 kernel applying a binary op: `fn f(x, y) -> f32 { x <op> y }`
///
/// Hand-built: takes a runtime `op` parameter that cannot be expressed as
/// a static Rust source string for `parse_kernel`. See #144 AC1.
#[must_use]
pub fn binop_var_var_kernel(op: BinOpKind) -> KernelDef {
    KernelDef::new(
        format!("binop_vv_{op:?}"),
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
                    op,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    )
}

/// Build a kernel with constant-constant binop: `fn f(x, a, b) -> f32 { (a <op> b) + x }`
///
/// Param 0 = variable x, params 1,2 = constants a,b.
///
/// Hand-built: takes a runtime `op` parameter that cannot be expressed as
/// a static Rust source string for `parse_kernel`. See #144 AC1.
#[must_use]
pub fn binop_const_const_kernel(op: BinOpKind) -> KernelDef {
    KernelDef::new(
        format!("binop_cc_{op:?}"),
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
                IRNodeKind::BinOp {
                    op,
                    lhs: NodeId::new(1),
                    rhs: NodeId::new(2),
                },
            ),
            IRNode::new(
                NodeId::new(4),
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(3),
                },
            ),
        ],
        NodeId::new(4),
    )
}

/// Build a kernel with constant-constant compare:
/// `fn f(x, a, b) -> f32 { select(compare(a, b), 1.0, 0.0) + x }`
///
/// Compare produces Bool; Select converts to F32 before the BinOp Add.
///
/// Hand-built: takes a runtime `op` parameter and uses IR constructs
/// (`Compare`, `Select`) not yet supported in the Rust→IR lowerer. See #144 AC1.
#[must_use]
pub fn compare_const_fold_kernel(op: CompareOpKind) -> KernelDef {
    KernelDef::new(
        format!("cmp_{op:?}"),
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
                    op,
                    lhs: NodeId::new(1),
                    rhs: NodeId::new(2),
                },
            ),
            IRNode::new(NodeId::new(4), IRNodeKind::Literal(1.0)),
            IRNode::new(NodeId::new(5), IRNodeKind::Literal(0.0)),
            IRNode::new(
                NodeId::new(6),
                IRNodeKind::Select {
                    cond: NodeId::new(3),
                    then_val: NodeId::new(4),
                    else_val: NodeId::new(5),
                },
            ),
            IRNode::new(
                NodeId::new(7),
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(6),
                },
            ),
        ],
        NodeId::new(7),
    )
}

/// Build a kernel: `fn f(x) -> f32 { select(compare(x, 0.0), 1.0, 0.0) }`
///
/// Compare produces Bool; Select converts to F32 for the kernel return.
///
/// Hand-built: takes a runtime `op` parameter and uses IR constructs
/// (`Compare`, `Select`) not yet supported in the Rust→IR lowerer. See #144 AC1.
#[must_use]
pub fn compare_var_kernel(op: CompareOpKind) -> KernelDef {
    KernelDef::new(
        format!("cmp_var_{op:?}"),
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(0.0)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::Compare {
                    op,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            IRNode::new(NodeId::new(3), IRNodeKind::Literal(1.0)),
            IRNode::new(
                NodeId::new(4),
                IRNodeKind::Select {
                    cond: NodeId::new(2),
                    then_val: NodeId::new(3),
                    else_val: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(4),
    )
}
