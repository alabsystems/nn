// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the auto-fuse elementwise chain peephole pass (#3517).

use std::collections::HashMap;

use crate::ir::{BinOpKind, IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType, UnaryFnKind};
use crate::tensor_block_builder::TensorBlockBuilder;
use crate::tensor_ir::TensorOpKind;
use crate::trace_compile::{CompiledKernel, CompiledStep, NativeOpKind};

use super::{fuse_elementwise_chains, is_single_elementwise_dispatch, AutoFuseStats};

// -- Helpers ------------------------------------------------------------------

/// Build a scalar KernelDef for a unary op: `out = op(x)`.
fn make_unary_scalar_kernel(name: &str, op: UnaryFnKind) -> KernelDef {
    let params = vec![Param::new("p0", ScalarType::F32)];
    let nodes = vec![
        IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
        IRNode::new(
            NodeId::new(1),
            IRNodeKind::UnaryFn {
                op,
                input: NodeId::new(0),
            },
        ),
    ];
    KernelDef::new(name, params, ScalarType::F32, nodes, NodeId::new(1))
}

/// Build a scalar KernelDef for a binary add: `out = x + y`.
fn make_add_scalar_kernel() -> KernelDef {
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
    KernelDef::new("add", params, ScalarType::F32, nodes, NodeId::new(2))
}

/// Build a scalar KernelDef for a binary mul: `out = x * y`.
fn make_mul_scalar_kernel() -> KernelDef {
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
    ];
    KernelDef::new("mul", params, ScalarType::F32, nodes, NodeId::new(2))
}

/// Wrap a scalar KernelDef into a CompiledStep::Dispatch with elementwise
/// tensor-level IR. `num_inputs` is the number of tensor-level inputs
/// (1 for unary ops, 2 for binary ops).
fn make_elementwise_dispatch(
    scalar_kernel: KernelDef,
    shape: &[usize],
    num_inputs: usize,
    ext_ids: Option<Vec<u64>>,
) -> CompiledStep {
    let mut b = TensorBlockBuilder::new(&scalar_kernel.name);
    let mut inputs = Vec::with_capacity(num_inputs);
    for i in 0..num_inputs {
        inputs.push(b.add_input(&format!("input_{i}"), shape));
    }
    let out = b.add_elementwise(scalar_kernel, &inputs, shape);
    let def = b.build(out).expect("valid elementwise IR");

    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: ext_ids,
    }
}

/// Build use_counts for a linear chain of `n` steps where each consumes
/// the previous (every intermediate step has use_count == 1).
fn linear_use_counts(n: usize) -> Vec<usize> {
    let mut counts = vec![0; n];
    for c in counts.iter_mut().take(n.saturating_sub(1)) {
        *c = 1;
    }
    counts
}

/// Count non-passthrough Dispatch steps.
fn count_dispatches(steps: &[CompiledStep]) -> usize {
    steps
        .iter()
        .filter(|s| matches!(s, CompiledStep::Dispatch { .. }))
        .count()
}

/// Extract the scalar KernelDef from a Dispatch step's Elementwise node.
fn extract_scalar_kernel(step: &CompiledStep) -> Option<&KernelDef> {
    match step {
        CompiledStep::Dispatch { kernel, .. } => {
            for node in &kernel.def().nodes {
                if let TensorOpKind::Elementwise { kernel, .. } = &node.kind {
                    return Some(kernel);
                }
            }
            None
        }
        _ => None,
    }
}

// -- Tests --------------------------------------------------------------------

#[test]
fn test_auto_fuse_chain_of_three_unary_ops() {
    // Chain: [exp, tanh, exp] → should fuse into one dispatch.
    let shape = &[1, 4, 8];
    let mut steps = vec![
        make_elementwise_dispatch(
            make_unary_scalar_kernel("exp", UnaryFnKind::Exp),
            shape,
            1,
            Some(vec![0]),
        ),
        make_elementwise_dispatch(
            make_unary_scalar_kernel("tanh", UnaryFnKind::Tanh),
            shape,
            1,
            Some(vec![0]),
        ),
        make_elementwise_dispatch(
            make_unary_scalar_kernel("exp", UnaryFnKind::Exp),
            shape,
            1,
            Some(vec![1]),
        ),
    ];

    let use_counts = linear_use_counts(3);
    let stats = fuse_elementwise_chains(&mut steps, &use_counts);

    assert_eq!(stats.chains_fused, 1);
    assert_eq!(stats.ops_fused, 3);
    assert_eq!(stats.chains_skipped, 0);
    assert_eq!(count_dispatches(&steps), 1, "should be 1 fused dispatch");

    // First two steps should be IdentityPassthrough.
    assert!(matches!(steps[0], CompiledStep::IdentityPassthrough));
    assert!(matches!(steps[1], CompiledStep::IdentityPassthrough));
    // Last step should be the fused dispatch.
    assert!(matches!(steps[2], CompiledStep::Dispatch { .. }));

    // Verify the fused kernel name.
    let fused_kernel = extract_scalar_kernel(&steps[2]).expect("fused kernel");
    assert!(
        fused_kernel.name.starts_with("fused_"),
        "name should start with fused_, got: {}",
        fused_kernel.name
    );
    assert!(
        fused_kernel.name.contains("_x3"),
        "name should contain _x3, got: {}",
        fused_kernel.name
    );
}

#[test]
fn test_auto_fuse_add_mul_tanh_chain() {
    // Chain: [add, mul, tanh] → should fuse into one dispatch.
    let shape = &[1, 4, 8];
    let mut steps = vec![
        make_elementwise_dispatch(make_add_scalar_kernel(), shape, 2, Some(vec![0, 1])),
        make_elementwise_dispatch(make_mul_scalar_kernel(), shape, 2, Some(vec![0, 2])),
        make_elementwise_dispatch(
            make_unary_scalar_kernel("tanh", UnaryFnKind::Tanh),
            shape,
            1,
            Some(vec![1]),
        ),
    ];

    let use_counts = linear_use_counts(3);
    let stats = fuse_elementwise_chains(&mut steps, &use_counts);

    assert_eq!(stats.chains_fused, 1);
    assert_eq!(stats.ops_fused, 3);
    assert_eq!(count_dispatches(&steps), 1);
}

#[test]
fn test_auto_fuse_chain_broken_by_matmul() {
    // [exp, tanh, MATMUL, exp, tanh]
    // Should produce TWO fused chains (before and after the matmul).
    // But the matmul step is NativeOp, which is not elementwise,
    // so it breaks the chain into [exp, tanh] and [exp, tanh].
    let shape = &[1, 4, 8];
    let mut steps = vec![
        // Chain 1: exp, tanh
        make_elementwise_dispatch(
            make_unary_scalar_kernel("exp", UnaryFnKind::Exp),
            shape,
            1,
            Some(vec![0]),
        ),
        make_elementwise_dispatch(
            make_unary_scalar_kernel("tanh", UnaryFnKind::Tanh),
            shape,
            1,
            Some(vec![0]),
        ),
        // Matmul (materialization point) — non-elementwise NativeOp.
        CompiledStep::NativeOp {
            op: NativeOpKind::LayerNorm {
                eps: 1e-5,
                input_shape: shape.to_vec(),
                hidden_dim: 8,
            },
            weight_data: HashMap::new(),
        },
        // Chain 2: exp, tanh
        make_elementwise_dispatch(
            make_unary_scalar_kernel("exp", UnaryFnKind::Exp),
            shape,
            1,
            Some(vec![2]),
        ),
        make_elementwise_dispatch(
            make_unary_scalar_kernel("tanh", UnaryFnKind::Tanh),
            shape,
            1,
            Some(vec![3]),
        ),
    ];

    // use_counts: each intermediate has fan-out 1.
    let use_counts = vec![1, 1, 1, 1, 0];
    let stats = fuse_elementwise_chains(&mut steps, &use_counts);

    assert_eq!(stats.chains_fused, 2, "two separate chains should be fused");
    assert_eq!(stats.ops_fused, 4);
    // 2 fused dispatches + 1 NativeOp = 3 non-passthrough steps remaining.
    assert_eq!(count_dispatches(&steps), 2);
    assert!(matches!(steps[2], CompiledStep::NativeOp { .. }));
}

#[test]
fn test_auto_fuse_single_op_not_fused() {
    // A single elementwise op (chain of length 1) should NOT be fused.
    let shape = &[1, 4, 8];
    let mut steps = vec![make_elementwise_dispatch(
        make_unary_scalar_kernel("exp", UnaryFnKind::Exp),
        shape,
        1,
        Some(vec![0]),
    )];

    let use_counts = vec![0];
    let stats = fuse_elementwise_chains(&mut steps, &use_counts);

    assert_eq!(stats.chains_fused, 0);
    assert_eq!(stats.ops_fused, 0);
    assert_eq!(count_dispatches(&steps), 1, "single op should remain");
}

#[test]
fn test_auto_fuse_branching_breaks_chain() {
    // [exp, tanh, exp] where exp has fan-out 2 (used by both tanh AND exp).
    // The chain should NOT extend past the branch point.
    let shape = &[1, 4, 8];
    let mut steps = vec![
        make_elementwise_dispatch(
            make_unary_scalar_kernel("exp", UnaryFnKind::Exp),
            shape,
            1,
            Some(vec![0]),
        ),
        make_elementwise_dispatch(
            make_unary_scalar_kernel("tanh", UnaryFnKind::Tanh),
            shape,
            1,
            Some(vec![0]),
        ),
        make_elementwise_dispatch(
            make_unary_scalar_kernel("exp", UnaryFnKind::Exp),
            shape,
            1,
            Some(vec![0]),
        ),
    ];

    // Fan-out: step 0 has 2 consumers (steps 1 and 2).
    let use_counts = vec![2, 0, 0];
    let stats = fuse_elementwise_chains(&mut steps, &use_counts);

    assert_eq!(
        stats.chains_fused, 0,
        "no chains should be fused due to branching"
    );
    assert_eq!(
        count_dispatches(&steps),
        3,
        "all 3 dispatches should remain"
    );
}

#[test]
fn test_auto_fuse_skips_non_elementwise_dispatch() {
    // Non-elementwise Dispatch (e.g., a conv1d) should not be part of a chain.
    let shape = &[1, 4, 8];
    let mut steps = vec![
        make_elementwise_dispatch(
            make_unary_scalar_kernel("exp", UnaryFnKind::Exp),
            shape,
            1,
            Some(vec![0]),
        ),
        // A Dispatch with a non-elementwise tensor op (e.g., matmul).
        // We simulate this with a Passthrough step, which is NOT elementwise.
        CompiledStep::Passthrough {
            op_name: "reshape".into(),
            output_shape: shape.to_vec(),
        },
        make_elementwise_dispatch(
            make_unary_scalar_kernel("tanh", UnaryFnKind::Tanh),
            shape,
            1,
            Some(vec![1]),
        ),
    ];

    let use_counts = vec![1, 1, 0];
    let stats = fuse_elementwise_chains(&mut steps, &use_counts);

    // The Passthrough breaks the chain, but since Passthrough is not
    // "non-passthrough" in our scan (we skip IdentityPassthrough, but
    // Passthrough is NOT IdentityPassthrough), the chain breaks.
    assert_eq!(stats.chains_fused, 0);
}

#[test]
fn test_is_single_elementwise_dispatch_rejects_non_dispatch() {
    assert!(!is_single_elementwise_dispatch(
        &CompiledStep::IdentityPassthrough
    ));
    assert!(!is_single_elementwise_dispatch(
        &CompiledStep::Passthrough {
            op_name: "reshape".into(),
            output_shape: vec![1, 4, 8],
        }
    ));
    assert!(!is_single_elementwise_dispatch(&CompiledStep::InputForward));
}

#[test]
fn test_is_single_elementwise_dispatch_accepts_elementwise() {
    let shape = &[1, 4, 8];
    let step = make_elementwise_dispatch(
        make_unary_scalar_kernel("exp", UnaryFnKind::Exp),
        shape,
        1,
        None,
    );
    assert!(is_single_elementwise_dispatch(&step));
}

#[test]
fn test_auto_fuse_preserves_external_node_ids() {
    // Chain: [add(ext:10,20), tanh] → fused should have external IDs [10, 20].
    let shape = &[1, 4, 8];
    let mut steps = vec![
        make_elementwise_dispatch(make_add_scalar_kernel(), shape, 2, Some(vec![10, 20])),
        make_elementwise_dispatch(
            make_unary_scalar_kernel("tanh", UnaryFnKind::Tanh),
            shape,
            1,
            Some(vec![0]),
        ),
    ];

    let use_counts = vec![1, 0];
    let stats = fuse_elementwise_chains(&mut steps, &use_counts);

    assert_eq!(stats.chains_fused, 1);

    // The fused step should have external node IDs from the add's inputs.
    match &steps[1] {
        CompiledStep::Dispatch {
            external_node_ids: Some(ids),
            ..
        } => {
            assert_eq!(ids.len(), 2, "two external inputs from the add");
            assert_eq!(ids[0], 10);
            assert_eq!(ids[1], 20);
        }
        other => panic!("expected Dispatch with external_node_ids, got: {other:?}"),
    }
}

#[test]
fn test_auto_fuse_identity_passthrough_between_chain_members() {
    // [exp, IdentityPassthrough, tanh] → chain should be [exp, tanh].
    let shape = &[1, 4, 8];
    let mut steps = vec![
        make_elementwise_dispatch(
            make_unary_scalar_kernel("exp", UnaryFnKind::Exp),
            shape,
            1,
            Some(vec![0]),
        ),
        CompiledStep::IdentityPassthrough,
        make_elementwise_dispatch(
            make_unary_scalar_kernel("tanh", UnaryFnKind::Tanh),
            shape,
            1,
            Some(vec![0]),
        ),
    ];

    let use_counts = vec![1, 0, 0];
    let stats = fuse_elementwise_chains(&mut steps, &use_counts);

    assert_eq!(stats.chains_fused, 1);
    assert_eq!(stats.ops_fused, 2);
    assert_eq!(count_dispatches(&steps), 1, "should be 1 fused dispatch");
}

#[test]
fn test_auto_fuse_empty_steps() {
    let mut steps: Vec<CompiledStep> = vec![];
    let use_counts: Vec<usize> = vec![];
    let stats = fuse_elementwise_chains(&mut steps, &use_counts);
    assert_eq!(stats, AutoFuseStats::default());
}
