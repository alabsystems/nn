// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Trace compilation tests for weighted ops (linear, conv1d, instance_norm,
//! embedding), CompiledPlan, cumsum, repeat_interleave, and constants.
//!
//! Extracted from `trace_compile_tests.rs` to keep files under 500 lines.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use crate::trace_compile::{compile_trace, compile_trace_to_plan, CompiledStep, NativeOpKind};

// -- Helpers (duplicated from trace_compile_tests.rs) -------------------------

fn graph_from_nodes(nodes: Vec<TraceNode>) -> ComputationGraph {
    ComputationGraph::from_nodes(nodes)
}

fn input_node(id: u64, shape: &[usize]) -> TraceNode {
    TraceNode::new(
        id,
        format!("input_{id}"),
        TraceOp::Input,
        vec![],
        shape.to_vec(),
        DType::F32,
    )
}

fn binary_node(
    id: u64,
    name: &str,
    op: TraceOp,
    lhs_id: u64,
    rhs_id: u64,
    shape: &[usize],
) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        op,
        vec![lhs_id, rhs_id],
        shape.to_vec(),
        DType::F32,
    )
}

// -- Weighted op tests --------------------------------------------------------

#[test]
fn test_compile_linear_with_weights() {
    let weight = WeightRef::new(vec![1.0; 12], vec![3, 4]).expect("test data");
    let bias = Some(WeightRef::new(vec![0.0; 3], vec![3]).expect("test data"));
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 4]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear { weight, bias },
            vec![0],
            vec![2, 3],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("linear should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } => {
            assert!(!weight_data.is_empty(), "linear should have weight_data");
            assert!(
                weight_data.contains_key("weight"),
                "weight_data should contain 'weight'"
            );
            assert_eq!(kernel.name(), "linear");
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

#[test]
fn test_compile_conv1d_with_weights() {
    let weight = WeightRef::new(vec![1.0; 24], vec![4, 3, 2]).expect("test data");
    let bias = Some(WeightRef::new(vec![0.0; 4], vec![4]).expect("test data"));
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 3, 8]),
        TraceNode::new(
            1,
            "conv1d_0".into(),
            TraceOp::Conv1d {
                weight,
                bias,
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
            vec![0],
            vec![1, 4, 7],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("conv1d should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::Dispatch { weight_data, .. } => {
            assert!(weight_data.contains_key("weight"));
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

// -- Depthwise Conv1d → NativeOp routing (#3538) -----------------------------

/// Depthwise Conv1d with kernel_size=1 routes to NativeOp Conv1dGemm.
#[test]
fn test_compile_depthwise_conv1d_k1() {
    // Depthwise: groups == in_channels == 8, weight: [8, 1, 1]
    let weight = WeightRef::new(vec![1.0; 8], vec![8, 1, 1]).expect("test data");
    let bias = Some(WeightRef::new(vec![0.0; 8], vec![8]).expect("test data"));
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 8, 16]),
        TraceNode::new(
            1,
            "conv1d_dw_k1".into(),
            TraceOp::Conv1d {
                weight,
                bias,
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 8,
            },
            vec![0],
            vec![1, 8, 16],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("depthwise conv1d k=1 should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::NativeOp { op, weight_data } => {
            match op {
                NativeOpKind::Conv1dGemm {
                    groups,
                    kernel_size,
                    out_channels,
                    has_bias,
                    ..
                } => {
                    assert_eq!(*groups, 8, "should preserve depthwise groups");
                    assert_eq!(*kernel_size, 1);
                    assert_eq!(*out_channels, 8);
                    assert!(*has_bias, "should have bias");
                }
                other => panic!("expected Conv1dGemm, got {other:?}"),
            }
            assert!(weight_data.contains_key("weight"));
            assert!(weight_data.contains_key("bias"));
        }
        other => panic!("expected NativeOp for depthwise conv1d, got {other:?}"),
    }
}

/// Depthwise Conv1d with kernel_size=3 routes to NativeOp Conv1dGemm.
#[test]
fn test_compile_depthwise_conv1d_k3() {
    // Depthwise: groups == in_channels == 4, weight: [4, 1, 3]
    let weight = WeightRef::new(vec![1.0; 12], vec![4, 1, 3]).expect("test data");
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 4, 10]),
        TraceNode::new(
            1,
            "conv1d_dw_k3".into(),
            TraceOp::Conv1d {
                weight,
                bias: None,
                padding: 1,
                stride: 1,
                dilation: 1,
                groups: 4,
            },
            vec![0],
            vec![1, 4, 10],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("depthwise conv1d k=3 should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::NativeOp { op, .. } => match op {
            NativeOpKind::Conv1dGemm {
                groups,
                kernel_size,
                padding,
                has_bias,
                ..
            } => {
                assert_eq!(*groups, 4);
                assert_eq!(*kernel_size, 3);
                assert_eq!(*padding, 1);
                assert!(!*has_bias, "should have no bias");
            }
            other => panic!("expected Conv1dGemm, got {other:?}"),
        },
        other => panic!("expected NativeOp for depthwise conv1d k=3, got {other:?}"),
    }
}

/// Depthwise Conv1d with kernel_size=4 and causal padding routes to NativeOp.
/// Causal padding: left-pad = kernel_size - 1, right-pad = 0 → total padding
/// is kernel_size - 1 on one side. In the Conv1d trace, this appears as
/// padding = kernel_size - 1 = 3 (the model pads the input before Conv1d).
#[test]
fn test_compile_depthwise_conv1d_k4_causal() {
    // Depthwise: groups == in_channels == 16, weight: [16, 1, 4]
    // Causal padding = kernel_size - 1 = 3
    let weight = WeightRef::new(vec![1.0; 64], vec![16, 1, 4]).expect("test data");
    let bias = Some(WeightRef::new(vec![0.0; 16], vec![16]).expect("test data"));
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 16, 32]),
        TraceNode::new(
            1,
            "conv1d_dw_k4_causal".into(),
            TraceOp::Conv1d {
                weight,
                bias,
                padding: 3, // causal: kernel_size - 1
                stride: 1,
                dilation: 1,
                groups: 16,
            },
            vec![0],
            // Output length: (32 + 2*3 - 4) / 1 + 1 = 35
            vec![1, 16, 35],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("depthwise conv1d k=4 causal should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::NativeOp { op, weight_data } => {
            match op {
                NativeOpKind::Conv1dGemm {
                    groups,
                    kernel_size,
                    padding,
                    stride,
                    out_channels,
                    has_bias,
                    ..
                } => {
                    assert_eq!(*groups, 16);
                    assert_eq!(*kernel_size, 4);
                    assert_eq!(*padding, 3, "causal padding = kernel_size - 1");
                    assert_eq!(*stride, 1);
                    assert_eq!(*out_channels, 16);
                    assert!(*has_bias);
                }
                other => panic!("expected Conv1dGemm, got {other:?}"),
            }
            assert!(weight_data.contains_key("weight"));
            assert!(weight_data.contains_key("bias"));
        }
        other => panic!("expected NativeOp for depthwise conv1d k=4, got {other:?}"),
    }
}

/// Standard Conv1d with groups=1 below FLOP threshold stays as Dispatch (not NativeOp).
#[test]
fn test_compile_conv1d_groups1_small_stays_dispatch() {
    // Small standard conv: groups=1, below FLOP threshold
    let weight = WeightRef::new(vec![1.0; 24], vec![4, 3, 2]).expect("test data");
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 3, 8]),
        TraceNode::new(
            1,
            "conv1d_small".into(),
            TraceOp::Conv1d {
                weight,
                bias: None,
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
            vec![0],
            vec![1, 4, 7],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("small conv1d should compile");
    assert_eq!(steps.len(), 2);
    // groups=1 and small FLOPs → generic Dispatch path (not NativeOp)
    assert!(
        matches!(steps[1], CompiledStep::Dispatch { .. }),
        "small groups=1 conv1d should use Dispatch, got {:?}",
        std::mem::discriminant(&steps[1])
    );
}

#[test]
fn test_compile_instance_norm() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 3, 8]),
        TraceNode::new(
            1,
            "instance_norm_0".into(),
            TraceOp::InstanceNorm { eps: 1e-5 },
            vec![0],
            vec![1, 3, 8],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("instance_norm should compile");
    assert_eq!(steps.len(), 2);
    // Fused InstanceNorm for rank >= 3 emits NativeOp (#2472), not Dispatch.
    assert!(matches!(steps[1], CompiledStep::NativeOp { .. }));
}

#[test]
fn test_compile_embedding_with_weight() {
    let weight = WeightRef::new(vec![1.0; 30], vec![10, 3]).expect("test data");
    let graph = graph_from_nodes(vec![
        input_node(0, &[4]),
        TraceNode::new(
            1,
            "embedding_0".into(),
            TraceOp::Embedding { weight },
            vec![0],
            vec![4, 3],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("embedding should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::Dispatch { weight_data, .. } => {
            assert!(weight_data.contains_key("weight"));
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

// -- CompiledPlan tests -------------------------------------------------------

/// Build an MLP trace graph: Input([batch, in_dim]) -> (Linear -> Relu) x N layers.
fn build_mlp_graph(batch: usize, layer_dims: &[(usize, usize)]) -> ComputationGraph {
    let mut nodes = vec![input_node(0, &[batch, layer_dims[0].0])];
    let mut next_id: u64 = 1;
    let mut prev_id: u64 = 0;
    for (i, &(in_d, out_d)) in layer_dims.iter().enumerate() {
        let w = WeightRef::new(vec![1.0f32; in_d * out_d], vec![out_d, in_d]).expect("w");
        let b = Some(WeightRef::new(vec![0.0f32; out_d], vec![out_d]).expect("b"));
        let lid = next_id;
        nodes.push(TraceNode::new(
            lid,
            format!("linear_{i}"),
            TraceOp::Linear { weight: w, bias: b },
            vec![prev_id],
            vec![batch, out_d],
            DType::F32,
        ));
        next_id += 1;
        let rid = next_id;
        nodes.push(TraceNode::new(
            rid,
            format!("relu_{i}"),
            TraceOp::Relu,
            vec![lid],
            vec![batch, out_d],
            DType::F32,
        ));
        next_id += 1;
        prev_id = rid;
    }
    graph_from_nodes(nodes)
}

/// 5-layer MLP: Input -> (Linear -> Relu) x 5.
/// Verifies step count, input shapes, output index, weights, and step pattern.
#[test]
fn test_compile_plan_5_layer_mlp() {
    let dims = [(16, 32), (32, 32), (32, 32), (32, 32), (32, 8)];
    let graph = build_mlp_graph(2, &dims);
    let plan = compile_trace_to_plan(&graph).expect("5-layer MLP should compile");

    assert_eq!(plan.steps.len(), 11, "1 input + 5 linear + 5 relu = 11");
    assert_eq!(plan.input_shapes, vec![vec![2, 16]]);
    assert_eq!(plan.output_step, 10);
    assert!(plan.weight_names.contains(&"weight".to_string()));
    assert!(plan.weight_names.contains(&"bias".to_string()));

    // Step pattern: InputForward, then alternating Dispatch(linear), Dispatch(relu)
    assert!(matches!(plan.steps[0], CompiledStep::InputForward));
    for i in 0..5 {
        let li = 1 + i * 2;
        assert!(
            matches!(&plan.steps[li], CompiledStep::Dispatch { kernel, .. } if kernel.name() == "linear"),
            "step {li} should be linear"
        );
        assert!(matches!(&plan.steps[li + 1], CompiledStep::Dispatch { .. }));
    }
}

/// Verifies `CompiledPlan` collects weights from multiple distinct layers.
#[test]
fn test_compile_plan_weight_inventory() {
    // Single linear layer to verify basic weight collection
    let weight = WeightRef::new(vec![1.0; 6], vec![3, 2]).expect("test data");
    let bias = Some(WeightRef::new(vec![0.0; 3], vec![3]).expect("test data"));
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 2]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear { weight, bias },
            vec![0],
            vec![1, 3],
            DType::F32,
        ),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("linear plan should compile");
    assert_eq!(plan.steps.len(), 2);
    assert_eq!(plan.input_shapes, vec![vec![1, 2]]);
    assert_eq!(plan.output_step, 1);
    assert!(plan.weight_names.contains(&"weight".to_string()));
    assert!(plan.weight_names.contains(&"bias".to_string()));
}

/// Empty graph produces a valid (empty) plan.
#[test]
fn test_compile_plan_empty() {
    let graph = graph_from_nodes(vec![]);
    let plan = compile_trace_to_plan(&graph).expect("empty plan should compile");
    assert!(plan.steps.is_empty());
    assert!(plan.input_shapes.is_empty());
    assert_eq!(plan.output_step, 0);
    assert!(plan.weight_names.is_empty());
}

// -- Cumsum / RepeatInterleave compilation ------------------------------------

#[test]
fn test_compile_cumsum_1d() {
    // cumsum([a, b, c, d], dim=0) = [a, a+b, a+b+c, a+b+c+d]
    let graph = graph_from_nodes(vec![
        input_node(0, &[4]),
        TraceNode::new(
            1,
            "cumsum_0".into(),
            TraceOp::Cumsum { dim: 0 },
            vec![0],
            vec![4],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("cumsum should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[0], CompiledStep::InputForward));
    match &steps[1] {
        CompiledStep::NativeOp { op, weight_data } => {
            assert!(matches!(op, NativeOpKind::Cumsum { dim: 0, .. }));
            assert!(weight_data.is_empty());
        }
        other => panic!("expected NativeOp Cumsum, got {other:?}"),
    }
}

#[test]
fn test_compile_cumsum_2d() {
    // cumsum along dim=1 of shape [2, 3] -> same output shape [2, 3]
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 3]),
        TraceNode::new(
            1,
            "cumsum_0".into(),
            TraceOp::Cumsum { dim: 1 },
            vec![0],
            vec![2, 3],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("cumsum 2d should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::NativeOp { .. }));
}

#[test]
fn test_compile_cumsum_single_element() {
    // cumsum of a single element is identity
    let graph = graph_from_nodes(vec![
        input_node(0, &[1]),
        TraceNode::new(
            1,
            "cumsum_0".into(),
            TraceOp::Cumsum { dim: 0 },
            vec![0],
            vec![1],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("cumsum single should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::NativeOp { .. }));
}

#[test]
fn test_compile_repeat_interleave() {
    // repeat_interleave([a, b, c], repeats=2, dim=0) = [a, a, b, b, c, c]
    // Two-input form (input + counts tensor): always emits RuntimeOp because
    // counts are data-dependent even when uniform. Fixes #2452.
    let graph = graph_from_nodes(vec![
        input_node(0, &[3]),
        input_node(1, &[3]), // repeats tensor (uniform: all 2s)
        TraceNode::new(
            2,
            "repeat_interleave_0".into(),
            TraceOp::RepeatInterleave { dim: 0 },
            vec![0, 1],
            vec![6],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("repeat_interleave should compile");
    assert_eq!(steps.len(), 3);
    assert!(
        matches!(steps[2], CompiledStep::RuntimeOp { .. }),
        "two-input repeat_interleave should emit RuntimeOp, got {:?}",
        std::mem::discriminant(&steps[2])
    );
}

#[test]
fn test_compile_repeat_interleave_2d() {
    // Input [2, 3], dim=1, repeats=2 -> output [2, 6]
    // Two-input form: always RuntimeOp (counts are data-dependent). Fixes #2452.
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 3]),
        input_node(1, &[3]), // repeats tensor
        TraceNode::new(
            2,
            "repeat_interleave_0".into(),
            TraceOp::RepeatInterleave { dim: 1 },
            vec![0, 1],
            vec![2, 6],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("repeat_interleave 2d should compile");
    assert_eq!(steps.len(), 3);
    assert!(matches!(steps[2], CompiledStep::RuntimeOp { .. }));
}

#[test]
fn test_compile_repeat_interleave_variable_emits_runtime_op() {
    // Variable repeats: input dim=3, output dim=7 (7 is not divisible by 3).
    // Should emit RuntimeOp instead of erroring (#2234).
    let graph = graph_from_nodes(vec![
        input_node(0, &[3]),
        input_node(1, &[3]),
        TraceNode::new(
            2,
            "repeat_interleave_0".into(),
            TraceOp::RepeatInterleave { dim: 0 },
            vec![0, 1],
            vec![7], // Not divisible by 3 -> variable repeats -> RuntimeOp
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("variable repeats should compile as RuntimeOp");
    assert_eq!(steps.len(), 3);
    match &steps[2] {
        CompiledStep::RuntimeOp { op } => match op {
            crate::trace_compile::RuntimeOpKind::RepeatInterleave {
                dim,
                input_shape,
                counts_shape,
            } => {
                assert_eq!(*dim, 0);
                assert_eq!(input_shape, &[3]);
                assert_eq!(counts_shape, &[3]);
            }
        },
        other => panic!("expected RuntimeOp for variable repeats, got {other:?}"),
    }
}

#[test]
fn test_compile_plan_with_runtime_op_has_no_extra_weight_names() {
    // RuntimeOp has no weight_data, so weight_names should be empty.
    let graph = graph_from_nodes(vec![
        input_node(0, &[3]),
        input_node(1, &[3]),
        TraceNode::new(
            2,
            "repeat_interleave_0".into(),
            TraceOp::RepeatInterleave { dim: 0 },
            vec![0, 1],
            vec![7],
            DType::F32,
        ),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("plan should compile");
    assert_eq!(plan.steps.len(), 3);
    assert!(
        plan.weight_names.is_empty(),
        "RuntimeOp should have no weights"
    );
    assert!(matches!(&plan.steps[2], CompiledStep::RuntimeOp { .. }));
}

// -- Constant compilation tests -----------------------------------------------

/// Regression #2212: Constant must produce ConstantValue, not InputForward.
#[test]
fn test_compile_constant_produces_constant_value() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[4]),
        TraceNode::new(
            1,
            "c".into(),
            TraceOp::Constant { value: 2.0 },
            vec![],
            vec![4],
            DType::F32,
        ),
        binary_node(2, "add_0", TraceOp::Add, 0, 1, &[4]),
    ]);
    let steps = compile_trace(&graph).expect("constant + add should compile");
    assert_eq!(steps.len(), 3);
    assert!(matches!(steps[0], CompiledStep::InputForward));
    assert!(
        matches!(steps[1], CompiledStep::ConstantValue { value, .. } if (value - 2.0).abs() < f64::EPSILON),
        "constant should produce ConstantValue, got {:?}",
        steps[1]
    );
}

/// Regression #2212: build_plan counts only Input nodes, not Constant.
#[test]
fn test_compile_plan_constant_not_counted_as_input() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[4]),
        TraceNode::new(
            1,
            "c".into(),
            TraceOp::Constant { value: 1.0 },
            vec![],
            vec![4],
            DType::F32,
        ),
        binary_node(2, "add_0", TraceOp::Add, 0, 1, &[4]),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("should compile");
    assert_eq!(
        plan.input_shapes.len(),
        1,
        "constant should not count as input"
    );
    assert_eq!(plan.input_shapes[0], vec![4]);
}
