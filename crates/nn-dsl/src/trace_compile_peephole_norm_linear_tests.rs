// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Structural tests for peephole pass 5: Norm + Linear → NormLinear.
//!
//! Tests verify that the peephole optimizer correctly fuses LayerNorm NativeOp +
//! Linear Dispatch and RmsNorm Dispatch + Linear Dispatch into NormLinear
//! NativeOps. Part of #3089.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

// super = norm_linear, super::super = peephole, super::super::super = trace_compile
use super::super::super::{CompiledKernel, CompiledStep, FusedNormKind, NativeOpKind};

/// Helper: build a TraceNode for test graph construction.
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

/// Build a LayerNorm NativeOp step with weight and bias.
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

/// Build an rms_norm Dispatch step (Qwen3 pattern).
fn make_rms_norm_dispatch(input_shape: &[usize], hidden_dim: usize, eps: f32) -> CompiledStep {
    use crate::tensor_block_builder::TensorBlockBuilder;

    let axis = if input_shape.is_empty() {
        0
    } else {
        input_shape.len() - 1
    };

    let mut b = TensorBlockBuilder::new("rms_norm");
    let input = b.add_input("input_0", input_shape);
    let eps_node = b.add_input("eps", &[1]);
    let w = b.add_input("weight", &[hidden_dim]);
    let output = b.add_rms_norm(input, eps_node, axis, w, input_shape);
    let def = b.build(output).expect("valid rms_norm IR");

    let mut weight_data = HashMap::new();
    weight_data.insert(
        "eps".to_string(),
        WeightRef::new(vec![eps], vec![1]).expect("valid eps"),
    );
    weight_data.insert(
        "weight".to_string(),
        WeightRef::new(vec![1.0f32; hidden_dim], vec![hidden_dim]).expect("valid weight"),
    );

    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: None,
    }
}

/// Build a Linear Dispatch step.
fn make_linear_dispatch(
    input_shape: &[usize],
    out_features: usize,
    has_bias: bool,
) -> CompiledStep {
    use crate::tensor_block_builder::TensorBlockBuilder;

    let in_features = *input_shape.last().unwrap();
    let mut output_shape = input_shape.to_vec();
    *output_shape.last_mut().unwrap() = out_features;
    let weight_shape = [out_features, in_features];

    let mut b = TensorBlockBuilder::new("linear");
    let input = b.add_input("input_0", input_shape);
    let w = b.add_input("weight", &weight_shape);
    let bi = if has_bias {
        Some(b.add_input("bias", &[out_features]))
    } else {
        None
    };
    let output = b.add_linear(input, w, bi, &output_shape);
    let def = b.build(output).expect("valid linear IR");

    let mut weight_data = HashMap::new();
    weight_data.insert(
        "weight".to_string(),
        WeightRef::new(
            vec![0.0f32; out_features * in_features],
            weight_shape.to_vec(),
        )
        .expect("valid weight"),
    );
    if has_bias {
        weight_data.insert(
            "bias".to_string(),
            WeightRef::new(vec![0.0f32; out_features], vec![out_features]).expect("valid bias"),
        );
    }

    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: None,
    }
}

// -- LayerNorm + Linear fusion ------------------------------------------------

#[test]
fn test_peephole_fuses_layernorm_linear() {
    let input_shape = [4, 768];
    let hidden_dim = 768;
    let out_features = 3072;

    let mut steps = vec![
        CompiledStep::InputForward,
        make_layernorm_native(&input_shape, hidden_dim, 1e-5),
        make_linear_dispatch(&input_shape, out_features, true),
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, input_shape.to_vec()),
        test_node(1, "layernorm", vec![0], input_shape.to_vec()),
        test_node(2, "linear", vec![1], vec![4, 3072]),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // LayerNorm + Linear now produces FusedLayerNormLinear (#4252),
    // not NormLinear{LayerNorm}. Both share the same executor.
    match &steps[1] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::FusedLayerNormLinear {
                    eps,
                    hidden_dim: hd,
                    out_features: of,
                    has_bias,
                    ..
                },
            weight_data,
        } => {
            assert!((eps - 1e-5).abs() < 1e-10);
            assert_eq!(*hd, 768);
            assert_eq!(*of, 3072);
            assert!(*has_bias);
            assert!(
                weight_data.contains_key("norm_weight"),
                "norm_weight missing"
            );
            assert!(weight_data.contains_key("norm_bias"), "norm_bias missing");
            assert!(weight_data.contains_key("weight"), "linear weight missing");
            assert!(weight_data.contains_key("bias"), "linear bias missing");
        }
        other => panic!(
            "expected FusedLayerNormLinear at step 1, got: {:?}",
            std::mem::discriminant(other)
        ),
    }

    assert!(
        matches!(&steps[2], CompiledStep::IdentityPassthrough),
        "expected IdentityPassthrough at step 2"
    );
}

// -- RmsNorm + Linear fusion --------------------------------------------------

#[test]
fn test_peephole_fuses_rms_norm_linear() {
    let input_shape = [1, 32, 512];
    let hidden_dim = 512;
    let out_features = 2048;

    let mut steps = vec![
        CompiledStep::InputForward,
        make_rms_norm_dispatch(&input_shape, hidden_dim, 1e-6),
        make_linear_dispatch(&input_shape, out_features, false),
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, input_shape.to_vec()),
        test_node(1, "rms_norm", vec![0], input_shape.to_vec()),
        test_node(2, "linear", vec![1], vec![1, 32, 2048]),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    match &steps[1] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::NormLinear {
                    norm_kind,
                    eps,
                    hidden_dim: hd,
                    out_features: of,
                    has_bias,
                    ..
                },
            weight_data,
        } => {
            assert_eq!(*norm_kind, FusedNormKind::RmsNorm);
            assert!((eps - 1e-6).abs() < 1e-10);
            assert_eq!(*hd, 512);
            assert_eq!(*of, 2048);
            assert!(!*has_bias);
            assert!(
                weight_data.contains_key("norm_weight"),
                "norm_weight missing"
            );
            assert!(
                !weight_data.contains_key("norm_bias"),
                "RmsNorm should not have norm_bias"
            );
            assert!(weight_data.contains_key("weight"), "linear weight missing");
        }
        other => panic!(
            "expected NormLinear at step 1, got: {:?}",
            std::mem::discriminant(other)
        ),
    }

    assert!(matches!(&steps[2], CompiledStep::IdentityPassthrough));
}

// -- Fan-out > 1 prevents fusion ----------------------------------------------

#[test]
fn test_peephole_no_fuse_norm_fanout_gt_1() {
    let input_shape = [4, 768];
    let hidden_dim = 768;
    let out_features = 3072;

    let mut steps = vec![
        CompiledStep::InputForward,
        make_layernorm_native(&input_shape, hidden_dim, 1e-5),
        make_linear_dispatch(&input_shape, out_features, true),
        CompiledStep::IdentityPassthrough, // second consumer of norm
    ];

    // Node 1 (norm) has TWO consumers: node 2 and node 3.
    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, input_shape.to_vec()),
        test_node(1, "layernorm", vec![0], input_shape.to_vec()),
        test_node(2, "linear", vec![1], vec![4, 3072]),
        test_node(3, "other", vec![1], input_shape.to_vec()),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    assert!(
        matches!(
            &steps[1],
            CompiledStep::NativeOp {
                op: NativeOpKind::LayerNorm { .. },
                ..
            }
        ),
        "LayerNorm should remain unfused with fan-out > 1"
    );
}

// -- Hidden dim > 7680 prevents fusion ----------------------------------------

#[test]
fn test_peephole_no_fuse_hidden_dim_too_large() {
    let hidden_dim = 8192; // > 7680 threadgroup memory limit
    let input_shape = [2, hidden_dim];
    let out_features = 1024;

    let mut steps = vec![
        CompiledStep::InputForward,
        make_layernorm_native(&input_shape, hidden_dim, 1e-5),
        make_linear_dispatch(&input_shape, out_features, true),
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, input_shape.to_vec()),
        test_node(1, "layernorm", vec![0], input_shape.to_vec()),
        test_node(2, "linear", vec![1], vec![2, out_features]),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    assert!(
        matches!(
            &steps[1],
            CompiledStep::NativeOp {
                op: NativeOpKind::LayerNorm { .. },
                ..
            }
        ),
        "LayerNorm should remain unfused with hidden_dim > 7680"
    );
}

// -- Dispatch count -----------------------------------------------------------

#[test]
fn test_fused_layer_norm_linear_single_dispatch() {
    let op = NativeOpKind::FusedLayerNormLinear {
        eps: 1e-5,
        input_shape: vec![4, 768],
        hidden_dim: 768,
        out_features: 3072,
        has_bias: true,
    };
    assert_eq!(
        op.estimated_metal_dispatches(),
        1,
        "FusedLayerNormLinear should be 1 dispatch for small shapes"
    );
}
