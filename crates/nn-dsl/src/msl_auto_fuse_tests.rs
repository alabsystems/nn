// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for auto-generated fused MSL codegen.
//!
//! Part of #3518.

use super::*;
use crate::ir::{
    BinOpKind, CompareOpKind, IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId, Param, ScalarType,
    UnaryFnKind,
};
use crate::tensor_ir::BroadcastAlignment;

/// Build a single-unary KernelDef: `f(x) = tanh(x)`.
fn build_tanh_kernel() -> KernelDef {
    let params = vec![Param::new("p0", ScalarType::F32)];
    let nodes = vec![
        IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
        IRNode::new(
            NodeId::new(1),
            IRNodeKind::UnaryFn {
                op: UnaryFnKind::Tanh,
                input: NodeId::new(0),
            },
        ),
    ];
    KernelDef::new(
        "fused_tanh_x1",
        params,
        ScalarType::F32,
        nodes,
        NodeId::new(1),
    )
}

/// Build a 3-op chain KernelDef: `f(x, y) = tanh(x * y + x)`.
/// Represents: Add(bias=y), Mul(x, y), Tanh.
/// Actually: p0=x, p1=y => t2 = p0*p1, t3 = t2+p0, t4 = tanh(t3).
fn build_add_mul_tanh_kernel() -> KernelDef {
    let params = vec![
        Param::new("p0", ScalarType::F32),
        Param::new("p1", ScalarType::F32),
    ];
    let nodes = vec![
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
        IRNode::new(
            NodeId::new(3),
            IRNodeKind::BinOp {
                op: BinOpKind::Add,
                lhs: NodeId::new(2),
                rhs: NodeId::new(0),
            },
        ),
        IRNode::new(
            NodeId::new(4),
            IRNodeKind::UnaryFn {
                op: UnaryFnKind::Tanh,
                input: NodeId::new(3),
            },
        ),
    ];
    KernelDef::new(
        "fused_mul_add_tanh",
        params,
        ScalarType::F32,
        nodes,
        NodeId::new(4),
    )
}

/// Build a chain with a literal constant: `f(x) = x + 1.0`.
fn build_add_constant_kernel() -> KernelDef {
    let params = vec![Param::new("p0", ScalarType::F32)];
    let nodes = vec![
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
    ];
    KernelDef::new(
        "fused_add_const",
        params,
        ScalarType::F32,
        nodes,
        NodeId::new(2),
    )
}

#[test]
fn test_single_unary_generates_valid_msl() {
    let kernel = build_tanh_kernel();
    let meta = FusedKernelMeta::new(
        vec![vec![1, 4, 8]],
        vec![1, 4, 8],
        BroadcastAlignment::Right,
        ScalarType::F32,
    );

    let result = generate_fused_msl(&kernel, &meta).expect("codegen should succeed");

    // Kernel name should be derived from KernelDef name.
    assert_eq!(result.kernel_name, "fused_tanh_x1_kernel");

    // Should contain Metal prelude.
    assert!(
        result.msl_source.contains("#include <metal_stdlib>"),
        "missing Metal prelude"
    );

    // Should contain kernel function declaration.
    assert!(
        result
            .msl_source
            .contains("[[kernel]] void fused_tanh_x1_kernel("),
        "missing kernel declaration"
    );

    // Should have buffer bindings: 1 input + 1 output + 1 total = 3.
    assert_eq!(result.buffer_count, 3);

    // Should contain tanh call.
    assert!(
        result.msl_source.contains("tanh"),
        "missing tanh function call"
    );

    // Should contain bounds check.
    assert!(
        result.msl_source.contains("if (tid >= total) return;"),
        "missing bounds check"
    );

    // Should write to out[tid].
    assert!(
        result.msl_source.contains("out[tid]"),
        "missing output write"
    );
}

#[test]
fn test_chain_add_mul_tanh_generates_valid_msl() {
    let kernel = build_add_mul_tanh_kernel();
    let meta = FusedKernelMeta::new(
        vec![vec![1, 4, 8], vec![1, 4, 8]],
        vec![1, 4, 8],
        BroadcastAlignment::Right,
        ScalarType::F32,
    );

    let result = generate_fused_msl(&kernel, &meta).expect("codegen should succeed");

    // 2 inputs + 1 output + 1 total = 4 buffers.
    assert_eq!(result.buffer_count, 4);

    // Should contain both buffer bindings.
    assert!(result
        .msl_source
        .contains("device const float* p0 [[buffer(0)]]"));
    assert!(result
        .msl_source
        .contains("device const float* p1 [[buffer(1)]]"));
    assert!(result
        .msl_source
        .contains("device float* out [[buffer(2)]]"));
    assert!(result
        .msl_source
        .contains("constant uint& total [[buffer(3)]]"));

    // Should contain the mul, add, and tanh operations.
    assert!(result.msl_source.contains("*"), "missing multiply op");
    assert!(result.msl_source.contains("+"), "missing add op");
    assert!(result.msl_source.contains("tanh"), "missing tanh call");
}

#[test]
fn test_broadcast_input_generates_modular_indexing() {
    let kernel = build_add_mul_tanh_kernel();
    // p0 has shape [1, 4, 8] (full), p1 has shape [1, 1, 8] (broadcast).
    let meta = FusedKernelMeta::new(
        vec![vec![1, 4, 8], vec![1, 1, 8]],
        vec![1, 4, 8],
        BroadcastAlignment::Right,
        ScalarType::F32,
    );

    let result = generate_fused_msl(&kernel, &meta).expect("codegen should succeed");

    // p0 should use flat indexing (same shape as output).
    assert!(
        result.msl_source.contains("p0[tid]"),
        "p0 should use flat tid indexing"
    );

    // p1 should use broadcast indexing (different shape from output).
    assert!(
        result.msl_source.contains("p1_idx"),
        "p1 should have broadcast index variable"
    );
    assert!(
        result.msl_source.contains("p1[p1_idx]"),
        "p1 should use broadcast index"
    );
}

#[test]
fn test_add_constant_no_extra_buffer() {
    let kernel = build_add_constant_kernel();
    let meta = FusedKernelMeta::new(
        vec![vec![1, 4, 8]],
        vec![1, 4, 8],
        BroadcastAlignment::Right,
        ScalarType::F32,
    );

    let result = generate_fused_msl(&kernel, &meta).expect("codegen should succeed");

    // Only 1 input param, so 1 + 1 output + 1 total = 3 buffers.
    assert_eq!(result.buffer_count, 3);

    // The literal 1.0 should appear in the MSL body.
    assert!(
        result.msl_source.contains("1.0"),
        "should contain literal 1.0"
    );
}

#[test]
fn test_shape_param_mismatch_returns_error() {
    let kernel = build_tanh_kernel();
    // Provide 2 shapes for 1 parameter.
    let meta = FusedKernelMeta::new(
        vec![vec![1, 4, 8], vec![1, 4, 8]],
        vec![1, 4, 8],
        BroadcastAlignment::Right,
        ScalarType::F32,
    );

    let result = generate_fused_msl(&kernel, &meta);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("shape count"),
        "error should mention shape mismatch: {err}"
    );
}

#[test]
fn test_total_elements_computed_correctly() {
    let meta = FusedKernelMeta::new(
        vec![vec![2, 3, 4]],
        vec![2, 3, 4],
        BroadcastAlignment::Right,
        ScalarType::F32,
    );
    assert_eq!(meta.total_elements(), 24);
}

#[test]
fn test_threadgroup_size_is_256() {
    let kernel = build_tanh_kernel();
    let meta = FusedKernelMeta::new(
        vec![vec![1, 4, 8]],
        vec![1, 4, 8],
        BroadcastAlignment::Right,
        ScalarType::F32,
    );

    let result = generate_fused_msl(&kernel, &meta).expect("codegen should succeed");
    assert_eq!(result.threadgroup_size, 256);
}

#[test]
fn test_f16_dtype_uses_half() {
    let kernel = build_tanh_kernel();
    let meta = FusedKernelMeta::new(
        vec![vec![1, 4, 8]],
        vec![1, 4, 8],
        BroadcastAlignment::Right,
        ScalarType::F16,
    );

    let result = generate_fused_msl(&kernel, &meta).expect("codegen should succeed");

    // F16 should emit "half" buffer types.
    assert!(
        result.msl_source.contains("device const half* p0"),
        "F16 should use 'half' type"
    );
    assert!(
        result.msl_source.contains("device half* out"),
        "output should use 'half' type"
    );

    // Should promote to float for accumulation.
    assert!(
        result.msl_source.contains("float p0_f"),
        "F16 should promote to float accumulator"
    );
}

#[test]
fn test_left_aligned_broadcast() {
    // Build a binary add: f(x, y) = x + y.
    let params = vec![
        Param::new("p0", ScalarType::F32),
        Param::new("p1", ScalarType::F32),
    ];
    let nodes = vec![
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
    ];
    let kernel = KernelDef::new("fused_add", params, ScalarType::F32, nodes, NodeId::new(2));

    // p0 has shape [4, 8] (full), p1 has shape [4] (left-aligned broadcast).
    let meta = FusedKernelMeta::new(
        vec![vec![4, 8], vec![4]],
        vec![4, 8],
        BroadcastAlignment::Left,
        ScalarType::F32,
    );

    let result = generate_fused_msl(&kernel, &meta).expect("codegen should succeed");

    // p1 should have broadcast indexing.
    assert!(
        result.msl_source.contains("p1_idx"),
        "broadcast input should have index variable"
    );
}

#[test]
fn test_relu_via_minmax_generates_valid_msl() {
    // relu(x) = max(x, 0) — uses MinMax IR node.
    let params = vec![Param::new("p0", ScalarType::F32)];
    let nodes = vec![
        IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
        IRNode::new(NodeId::new(1), IRNodeKind::Literal(0.0)),
        IRNode::new(
            NodeId::new(2),
            IRNodeKind::MinMax {
                op: MinMaxKind::Max,
                lhs: NodeId::new(0),
                rhs: NodeId::new(1),
            },
        ),
    ];
    let kernel = KernelDef::new("fused_relu", params, ScalarType::F32, nodes, NodeId::new(2));

    let meta = FusedKernelMeta::new(
        vec![vec![1, 4, 8]],
        vec![1, 4, 8],
        BroadcastAlignment::Right,
        ScalarType::F32,
    );

    let result = generate_fused_msl(&kernel, &meta).expect("codegen should succeed");
    assert!(result.msl_source.contains("max("), "should contain max()");
}

#[test]
fn test_select_generates_ternary() {
    // leaky_relu(x, 0.1) = x > 0 ? x : 0.1*x
    let params = vec![Param::new("p0", ScalarType::F32)];
    let nodes = vec![
        IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
        IRNode::new(NodeId::new(1), IRNodeKind::Literal(0.0)),
        IRNode::new(
            NodeId::new(2),
            IRNodeKind::Compare {
                op: CompareOpKind::Gt,
                lhs: NodeId::new(0),
                rhs: NodeId::new(1),
            },
        ),
        IRNode::new(NodeId::new(3), IRNodeKind::Literal(0.1)),
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
    ];
    let kernel = KernelDef::new(
        "fused_leaky_relu",
        params,
        ScalarType::F32,
        nodes,
        NodeId::new(5),
    );

    let meta = FusedKernelMeta::new(
        vec![vec![1, 4, 8]],
        vec![1, 4, 8],
        BroadcastAlignment::Right,
        ScalarType::F32,
    );

    let result = generate_fused_msl(&kernel, &meta).expect("codegen should succeed");
    assert!(result.msl_source.contains("?"), "should contain ternary op");
    assert!(result.msl_source.contains(">"), "should contain comparison");
}

#[test]
fn test_multiple_broadcast_inputs() {
    // f(x, y, z) = x + y + z with different shapes.
    let params = vec![
        Param::new("p0", ScalarType::F32),
        Param::new("p1", ScalarType::F32),
        Param::new("p2", ScalarType::F32),
    ];
    let nodes = vec![
        IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
        IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
        IRNode::new(NodeId::new(2), IRNodeKind::Param(2)),
        IRNode::new(
            NodeId::new(3),
            IRNodeKind::BinOp {
                op: BinOpKind::Add,
                lhs: NodeId::new(0),
                rhs: NodeId::new(1),
            },
        ),
        IRNode::new(
            NodeId::new(4),
            IRNodeKind::BinOp {
                op: BinOpKind::Add,
                lhs: NodeId::new(3),
                rhs: NodeId::new(2),
            },
        ),
    ];
    let kernel = KernelDef::new("fused_add3", params, ScalarType::F32, nodes, NodeId::new(4));

    // p0 = [2, 3, 4] (full), p1 = [1, 3, 1] (broadcast), p2 = [1, 1, 4] (broadcast).
    let meta = FusedKernelMeta::new(
        vec![vec![2, 3, 4], vec![1, 3, 1], vec![1, 1, 4]],
        vec![2, 3, 4],
        BroadcastAlignment::Right,
        ScalarType::F32,
    );

    let result = generate_fused_msl(&kernel, &meta).expect("codegen should succeed");

    // p0 should use flat indexing.
    assert!(result.msl_source.contains("p0[tid]"));

    // p1 and p2 should have broadcast indices.
    assert!(
        result.msl_source.contains("p1_idx"),
        "p1 should have broadcast index"
    );
    assert!(
        result.msl_source.contains("p2_idx"),
        "p2 should have broadcast index"
    );

    // 3 inputs + 1 output + 1 total = 5 buffers.
    assert_eq!(result.buffer_count, 5);
}
