// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for peephole pass: BinaryAdd + LayerNorm → AddLayerNorm.
//! Part of #1815 Tier 5 D2, #3089.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use super::super::super::{CompiledKernel, CompiledStep, NativeOpKind};

fn test_node(id: u64, name: &str, inputs: Vec<u64>, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        TraceOp::Relu,
        inputs,
        shape,
        DType::F32,
    )
}

fn test_input_node(id: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("input_{id}"),
        TraceOp::Input,
        vec![],
        shape,
        DType::F32,
    )
}

/// Build a Dispatch{add} step.
fn make_add_dispatch(input_shape: &[usize]) -> CompiledStep {
    use crate::tensor_block_builder::TensorBlockBuilder;

    let mut b = TensorBlockBuilder::new("add");
    let a = b.add_input("input_0", input_shape);
    let bb = b.add_input("input_1", input_shape);
    let output = b.add_binary_add(a, bb, input_shape);
    let def = b.build(output).expect("valid add IR");

    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: None,
    }
}

/// Build a LayerNorm NativeOp step.
fn make_layernorm_native(input_shape: &[usize], hidden_dim: usize, eps: f32) -> CompiledStep {
    let mut weight_data = HashMap::new();
    weight_data.insert(
        "weight".to_string(),
        WeightRef::new(vec![1.0f32; hidden_dim], vec![hidden_dim]).expect("valid weight"),
    );
    weight_data.insert(
        "bias".to_string(),
        WeightRef::new(vec![0.0f32; hidden_dim], vec![hidden_dim]).expect("valid bias"),
    );
    CompiledStep::NativeOp {
        op: NativeOpKind::LayerNorm {
            eps,
            input_shape: input_shape.to_vec(),
            hidden_dim,
        },
        weight_data,
    }
}

// -- Add + LayerNorm fusion ---------------------------------------------------

#[test]
fn test_peephole_fuses_add_layernorm() {
    let shape = vec![4, 768];
    let hidden_dim = 768;

    let mut steps = vec![
        CompiledStep::InputForward,                      // 0: input
        CompiledStep::InputForward,                      // 1: residual
        make_add_dispatch(&shape),                       // 2: add(0, 1)
        make_layernorm_native(&shape, hidden_dim, 1e-5), // 3: LN(2)
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, shape.clone()),
        test_input_node(1, shape.clone()),
        test_node(2, "add", vec![0, 1], shape.clone()),
        test_node(3, "layernorm", vec![2], shape),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    match &steps[2] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::AddLayerNorm {
                    eps,
                    hidden_dim: hd,
                    ..
                },
            weight_data,
        } => {
            assert!((eps - 1e-5).abs() < 1e-10);
            assert_eq!(*hd, 768);
            assert!(weight_data.contains_key("weight"), "LN weight missing");
            assert!(weight_data.contains_key("bias"), "LN bias missing");
        }
        other => panic!(
            "expected AddLayerNorm at step 2, got: {:?}",
            std::mem::discriminant(other)
        ),
    }

    assert!(
        matches!(&steps[3], CompiledStep::IdentityPassthrough),
        "expected IdentityPassthrough at step 3"
    );
}

// -- Fan-out > 1 prevents fusion ----------------------------------------------

#[test]
fn test_peephole_no_fuse_add_fanout_gt_1() {
    let shape = vec![4, 768];
    let hidden_dim = 768;

    let mut steps = vec![
        CompiledStep::InputForward,                      // 0: input
        CompiledStep::InputForward,                      // 1: residual
        make_add_dispatch(&shape),                       // 2: add(0, 1) — 2 consumers
        make_layernorm_native(&shape, hidden_dim, 1e-5), // 3: LN(2)
        CompiledStep::IdentityPassthrough,               // 4: second consumer of add
    ];

    // Node 2 (add) has TWO consumers: node 3 and node 4.
    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, shape.clone()),
        test_input_node(1, shape.clone()),
        test_node(2, "add", vec![0, 1], shape.clone()),
        test_node(3, "layernorm", vec![2], shape.clone()),
        test_node(4, "other", vec![2], shape),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // Should NOT fuse — add output has fan-out > 1.
    assert!(
        matches!(&steps[2], CompiledStep::Dispatch { .. }),
        "add should remain unfused with fan-out > 1"
    );
}

// -- Non-LayerNorm after add doesn't fuse -------------------------------------

#[test]
fn test_peephole_no_fuse_add_instance_norm() {
    let shape = vec![4, 768];

    let mut steps = vec![
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        make_add_dispatch(&shape),
        CompiledStep::NativeOp {
            op: NativeOpKind::InstanceNorm {
                eps: 1e-5,
                input_shape: shape.clone(),
            },
            weight_data: HashMap::new(),
        },
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, shape.clone()),
        test_input_node(1, shape.clone()),
        test_node(2, "add", vec![0, 1], shape.clone()),
        test_node(3, "instance_norm", vec![2], shape),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // InstanceNorm doesn't match — should remain unfused.
    assert!(
        matches!(&steps[2], CompiledStep::Dispatch { .. }),
        "add+InstanceNorm should not fuse"
    );
}

// -- Dispatch count -----------------------------------------------------------

#[test]
fn test_add_layer_norm_single_dispatch() {
    let op = NativeOpKind::AddLayerNorm {
        eps: 1e-5,
        input_shape: vec![4, 768],
        hidden_dim: 768,
    };
    assert_eq!(
        op.estimated_metal_dispatches(),
        1,
        "AddLayerNorm should be 1 dispatch"
    );
}
