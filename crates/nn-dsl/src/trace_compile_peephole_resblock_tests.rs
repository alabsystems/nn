// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for FusedResBlock peephole fusion (Pass 2).

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use super::super::super::{CompiledKernel, CompiledStep, NativeOpKind, NormActivation};

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

fn test_constant_node(id: u64, value: f64) -> TraceNode {
    TraceNode::new(
        id,
        format!("const_{id}"),
        TraceOp::Constant { value },
        vec![],
        vec![1],
        DType::F32,
    )
}

/// Helper: build a NormActivConv1d NativeOp step (simulating post-Pass-1 output).
fn make_norm_activ_conv1d(
    activation: NormActivation,
    input_shape: &[usize],
    output_channels: usize,
    kernel_size: usize,
    dilation: usize,
) -> CompiledStep {
    let mut weight_data = HashMap::new();
    weight_data.insert(
        "conv_weight".to_string(),
        WeightRef::new(
            vec![0.0f32; output_channels * input_shape[1] * kernel_size],
            vec![output_channels, input_shape[1], kernel_size],
        )
        .expect("valid weight"),
    );
    weight_data.insert(
        "conv_bias".to_string(),
        WeightRef::new(vec![0.0f32; output_channels], vec![output_channels]).expect("valid bias"),
    );
    if matches!(activation, NormActivation::Snake) {
        weight_data.insert(
            "alpha".to_string(),
            WeightRef::new(vec![1.0f32; input_shape[1]], vec![input_shape[1]])
                .expect("valid alpha"),
        );
    }
    CompiledStep::NativeOp {
        op: NativeOpKind::NormActivConv1d {
            activation,
            eps: 1e-5,
            conv_dilation: dilation,
            conv_padding: dilation, // padding = dilation for causal
            input_shape: input_shape.to_vec(),
            output_channels,
            kernel_size,
            external_node_ids: None,
        },
        weight_data,
    }
}

/// Helper: build an "add" Dispatch step (residual).
fn make_add_dispatch(shape: &[usize]) -> CompiledStep {
    use crate::tensor_block_builder::TensorBlockBuilder;

    let mut b = TensorBlockBuilder::new("add");
    let a = b.add_input("input_0", shape);
    let b_input = b.add_input("input_1", shape);
    let out = b.add_binary_add(a, b_input, shape);
    let def = b.build(out).expect("valid add IR");

    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: None,
    }
}

/// Helper: build a "mul" Dispatch step (for residual scale).
fn make_mul_dispatch(shape: &[usize]) -> CompiledStep {
    use crate::tensor_block_builder::TensorBlockBuilder;

    let mut b = TensorBlockBuilder::new("mul");
    let a = b.add_input("input_0", shape);
    let b_input = b.add_input("input_1", shape);
    let out = b.add_binary_mul(a, b_input, shape);
    let def = b.build(out).expect("valid mul IR");

    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: None,
    }
}

/// Build the standard 5-step ResBlock pattern with graph and steps (consecutive).
///
/// After Pass 1 NormActivConv1d peephole, the pattern is:
///   NormActivConv1d (at AdaIN1 pos) + IdentityPassthrough (at conv1 pos)
///   + NormActivConv1d (at AdaIN2 pos) + IdentityPassthrough (at conv2 pos)
///   + Dispatch "add"
///
/// Graph topology:
///   0: input x         [1, C, T]
///   1: input gamma1    [1, C, 1]
///   2: input beta1     [1, C, 1]
///   3: input gamma2    [1, C, 1]
///   4: input beta2     [1, C, 1]
///   5: adain1 (x, gamma1, beta1)       → NormActivConv1d (post-Pass-1)
///   6: conv1 (from adain1)             → IdentityPassthrough (post-Pass-1)
///   7: adain2 (conv1_out, gamma2, beta2) → NormActivConv1d (post-Pass-1)
///   8: conv2 (from adain2)             → IdentityPassthrough (post-Pass-1)
///   9: add (x, conv2_out)              → Dispatch "add"
fn build_resblock_fixture(
    channels: usize,
    time: usize,
    dilation1: usize,
    dilation2: usize,
    activation: NormActivation,
) -> (Vec<CompiledStep>, ComputationGraph) {
    let shape = vec![1, channels, time];
    let gamma_shape = vec![1, channels, 1];

    let steps = vec![
        CompiledStep::InputForward, // 0: x
        CompiledStep::InputForward, // 1: gamma1
        CompiledStep::InputForward, // 2: beta1
        CompiledStep::InputForward, // 3: gamma2
        CompiledStep::InputForward, // 4: beta2
        make_norm_activ_conv1d(activation.clone(), &shape, channels, 3, dilation1), // 5: at adain1 pos
        CompiledStep::IdentityPassthrough, // 6: at conv1 pos
        make_norm_activ_conv1d(activation, &shape, channels, 3, dilation2), // 7: at adain2 pos
        CompiledStep::IdentityPassthrough, // 8: at conv2 pos
        make_add_dispatch(&shape),         // 9: residual add
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, shape.clone()),
        test_input_node(1, gamma_shape.clone()),
        test_input_node(2, gamma_shape.clone()),
        test_input_node(3, gamma_shape.clone()),
        test_input_node(4, gamma_shape),
        // Node 5: first adain — inputs are [x=0, gamma1=1, beta1=2]
        test_node(5, "adain1", vec![0, 1, 2], shape.clone()),
        // Node 6: conv1 — input is [adain1_out=5]
        test_node(6, "conv1", vec![5], shape.clone()),
        // Node 7: second adain — inputs are [conv1_out=6, gamma2=3, beta2=4]
        test_node(7, "adain2", vec![6, 3, 4], shape.clone()),
        // Node 8: conv2 — input is [adain2_out=7]
        test_node(8, "conv2", vec![7], shape.clone()),
        // Node 9: add — inputs are [x=0, conv2_out=8]
        test_node(9, "add", vec![0, 8], shape),
    ]);

    (steps, graph)
}

#[test]
fn test_resblock_fuses_snake_pattern() {
    let (mut steps, graph) = build_resblock_fixture(512, 100, 1, 1, NormActivation::Snake);

    super::super::apply_peephole(&mut steps, &graph);

    // Steps 5-8 should all be IdentityPassthrough.
    for (idx, step) in steps[5..=8].iter().enumerate() {
        assert!(
            matches!(step, CompiledStep::IdentityPassthrough),
            "expected IdentityPassthrough at step {}, got {:?}",
            idx + 5,
            std::mem::discriminant(step)
        );
    }

    // Step 9 should be FusedResBlock.
    match &steps[9] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::FusedResBlock {
                    phase1,
                    phase2,
                    input_steps,
                    residual_scale,
                    style_proj,
                    shortcut_step,
                    ..
                },
            weight_data,
        } => {
            // No style projection absorbed in this pattern.
            assert!(style_proj.is_none(), "expected no style_proj");
            assert!(shortcut_step.is_none(), "expected identity shortcut");
            // Phase 1 params.
            assert_eq!(phase1.activation, NormActivation::Snake);
            assert!((phase1.eps - 1e-5).abs() < 1e-10);
            assert_eq!(phase1.conv_dilation, 1);
            assert_eq!(phase1.conv_padding, 1);
            assert_eq!(phase1.output_channels, 512);
            assert_eq!(phase1.kernel_size, 3);

            // Phase 2 params.
            assert_eq!(phase2.activation, NormActivation::Snake);
            assert_eq!(phase2.conv_dilation, 1);

            // input_steps: [x=0, gamma1=1, beta1=2, gamma2=3, beta2=4]
            assert_eq!(input_steps, &[0, 1, 2, 3, 4]);

            // No post-add scale for consecutive pattern.
            assert!((*residual_scale - 1.0).abs() < f32::EPSILON);

            // Weight data should have p1_ and p2_ prefixed keys.
            assert!(
                weight_data.contains_key("p1_conv_weight"),
                "missing p1_conv_weight"
            );
            assert!(
                weight_data.contains_key("p1_conv_bias"),
                "missing p1_conv_bias"
            );
            assert!(weight_data.contains_key("p1_alpha"), "missing p1_alpha");
            assert!(
                weight_data.contains_key("p2_conv_weight"),
                "missing p2_conv_weight"
            );
            assert!(
                weight_data.contains_key("p2_conv_bias"),
                "missing p2_conv_bias"
            );
            assert!(weight_data.contains_key("p2_alpha"), "missing p2_alpha");
        }
        other => panic!(
            "expected FusedResBlock at step 9, got: {:?}",
            std::mem::discriminant(other)
        ),
    }
}

#[test]
fn test_resblock_fuses_leaky_relu_pattern() {
    let (mut steps, graph) =
        build_resblock_fixture(256, 200, 1, 3, NormActivation::LeakyRelu { slope: 0.2 });

    super::super::apply_peephole(&mut steps, &graph);

    // Step 9 should be FusedResBlock with LeakyRelu.
    match &steps[9] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::FusedResBlock {
                    phase1,
                    phase2,
                    input_steps,
                    residual_scale,
                    style_proj,
                    shortcut_step,
                    ..
                },
            weight_data,
        } => {
            assert!(style_proj.is_none(), "expected no style_proj");
            assert!(shortcut_step.is_none(), "expected identity shortcut");
            assert_eq!(phase1.activation, NormActivation::LeakyRelu { slope: 0.2 });
            assert_eq!(phase1.conv_dilation, 1);
            assert_eq!(phase2.activation, NormActivation::LeakyRelu { slope: 0.2 });
            assert_eq!(phase2.conv_dilation, 3);
            assert_eq!(input_steps, &[0, 1, 2, 3, 4]);
            assert!((*residual_scale - 1.0).abs() < f32::EPSILON);
            // LeakyRelu has no alpha weight.
            assert!(!weight_data.contains_key("p1_alpha"));
            assert!(!weight_data.contains_key("p2_alpha"));
            assert!(weight_data.contains_key("p1_conv_weight"));
            assert!(weight_data.contains_key("p2_conv_weight"));
        }
        other => panic!(
            "expected FusedResBlock, got: {:?}",
            std::mem::discriminant(other)
        ),
    }
}

#[test]
fn test_resblock_rejects_mixed_activation() {
    let shape = vec![1, 256, 100];
    let gamma_shape = vec![1, 256, 1];

    // NormActivConv1d at adain positions, IdentityPassthrough at conv positions.
    let mut steps = vec![
        CompiledStep::InputForward,                                       // 0
        CompiledStep::InputForward,                                       // 1
        CompiledStep::InputForward,                                       // 2
        CompiledStep::InputForward,                                       // 3
        CompiledStep::InputForward,                                       // 4
        make_norm_activ_conv1d(NormActivation::Snake, &shape, 256, 3, 1), // 5
        CompiledStep::IdentityPassthrough,                                // 6
        make_norm_activ_conv1d(NormActivation::LeakyRelu { slope: 0.2 }, &shape, 256, 3, 1), // 7
        CompiledStep::IdentityPassthrough,                                // 8
        make_add_dispatch(&shape),                                        // 9
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, shape.clone()),
        test_input_node(1, gamma_shape.clone()),
        test_input_node(2, gamma_shape.clone()),
        test_input_node(3, gamma_shape.clone()),
        test_input_node(4, gamma_shape),
        test_node(5, "adain1", vec![0, 1, 2], shape.clone()),
        test_node(6, "conv1", vec![5], shape.clone()),
        test_node(7, "adain2", vec![6, 3, 4], shape.clone()),
        test_node(8, "conv2", vec![7], shape.clone()),
        test_node(9, "add", vec![0, 8], shape),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // Should NOT fuse — mixed activations.
    assert!(
        !matches!(
            &steps[9],
            CompiledStep::NativeOp {
                op: NativeOpKind::FusedResBlock { .. },
                ..
            }
        ),
        "should not fuse mixed activation families"
    );
}

#[test]
fn test_resblock_rejects_fanout_gt_1() {
    let shape = vec![1, 512, 100];
    let gamma_shape = vec![1, 512, 1];

    let mut steps = vec![
        CompiledStep::InputForward,                                       // 0
        CompiledStep::InputForward,                                       // 1
        CompiledStep::InputForward,                                       // 2
        CompiledStep::InputForward,                                       // 3
        CompiledStep::InputForward,                                       // 4
        make_norm_activ_conv1d(NormActivation::Snake, &shape, 512, 3, 1), // 5
        CompiledStep::IdentityPassthrough,                                // 6
        make_norm_activ_conv1d(NormActivation::Snake, &shape, 512, 3, 1), // 7
        CompiledStep::IdentityPassthrough,                                // 8
        make_add_dispatch(&shape),                                        // 9
        CompiledStep::IdentityPassthrough, // 10: extra consumer of step 6
    ];

    // Node 10 also consumes node 6 → fan-out = 2 for step 6 (the conv1
    // passthrough). Pass 2 checks use_counts at conv1 (=6), which is 2.
    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, shape.clone()),
        test_input_node(1, gamma_shape.clone()),
        test_input_node(2, gamma_shape.clone()),
        test_input_node(3, gamma_shape.clone()),
        test_input_node(4, gamma_shape),
        test_node(5, "adain1", vec![0, 1, 2], shape.clone()),
        test_node(6, "conv1", vec![5], shape.clone()),
        test_node(7, "adain2", vec![6, 3, 4], shape.clone()),
        test_node(8, "conv2", vec![7], shape.clone()),
        test_node(9, "add", vec![0, 8], shape.clone()),
        test_node(10, "other", vec![6], shape),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // Should NOT fuse — first NormActivConv1d's output (via conv1 passthrough)
    // has fan-out > 1.
    assert!(
        !matches!(
            &steps[9],
            CompiledStep::NativeOp {
                op: NativeOpKind::FusedResBlock { .. },
                ..
            }
        ),
        "should not fuse when NormActivConv1d has fan-out > 1"
    );
}

#[test]
fn test_resblock_too_few_steps() {
    // Only 4 steps — not enough for the 5-step pattern.
    let shape = vec![1, 256, 100];
    let mut steps = vec![
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        make_norm_activ_conv1d(NormActivation::Snake, &shape, 256, 3, 1),
        CompiledStep::IdentityPassthrough,
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, shape.clone()),
        test_input_node(1, vec![1, 256, 1]),
        test_node(2, "adain1", vec![0, 1], shape.clone()),
        test_node(3, "conv1", vec![2], shape),
    ]);

    // Should return without panic.
    super::super::apply_peephole(&mut steps, &graph);
    assert_eq!(steps.len(), 4);
}

/// Test graph-based detection with non-consecutive steps (F0-like pattern).
///
/// The F0 model has style projection steps (Linear, Narrow, Reshape) between
/// the two NormActivConv1d steps. The graph topology is the same as the
/// consecutive case, but the step indices are not adjacent.
///
/// Layout:
///   0: x, 1: gamma1, 2: beta1, 3: gamma2, 4: beta2
///   5: NormActivConv1d (phase1)
///   6: IdentityPassthrough (conv1)
///   7-9: style projection (Linear, Narrow, Narrow) — intervening steps
///   10: NormActivConv1d (phase2)
///   11: IdentityPassthrough (conv2)
///   12: add
#[test]
fn test_resblock_fuses_non_consecutive_f0_pattern() {
    let shape = vec![1, 512, 100];
    let gamma_shape = vec![1, 512, 1];

    let mut steps = vec![
        CompiledStep::InputForward, // 0: x
        CompiledStep::InputForward, // 1: gamma1
        CompiledStep::InputForward, // 2: beta1
        CompiledStep::InputForward, // 3: style (for phase2 projection)
        CompiledStep::InputForward, // 4: (placeholder)
        make_norm_activ_conv1d(NormActivation::LeakyRelu { slope: 0.2 }, &shape, 512, 3, 1), // 5
        CompiledStep::IdentityPassthrough, // 6: conv1 passthrough
        // Steps 7-9: intervening style projection (treated as opaque Dispatch steps)
        CompiledStep::IdentityPassthrough, // 7: style linear output
        CompiledStep::IdentityPassthrough, // 8: gamma2 (narrow)
        CompiledStep::IdentityPassthrough, // 9: beta2 (narrow)
        make_norm_activ_conv1d(NormActivation::LeakyRelu { slope: 0.2 }, &shape, 512, 3, 1), // 10
        CompiledStep::IdentityPassthrough, // 11: conv2 passthrough
        make_add_dispatch(&shape),         // 12: residual add
    ];

    // Graph topology: same as standard ResBlock, just non-consecutive step indices.
    // The style projection nodes (7-9) are in the graph but don't affect topology.
    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, shape.clone()),
        test_input_node(1, gamma_shape.clone()),
        test_input_node(2, gamma_shape.clone()),
        test_input_node(3, vec![1, 128]), // style
        test_input_node(4, vec![1]),      // placeholder
        // Node 5: first adain — inputs are [x=0, gamma1=1, beta1=2]
        test_node(5, "adain1", vec![0, 1, 2], shape.clone()),
        // Node 6: conv1 — input is [adain1_out=5]
        test_node(6, "conv1", vec![5], shape.clone()),
        // Nodes 7-9: style projection (gamma2/beta2 computation)
        test_node(7, "style_linear", vec![3], vec![1, 1024]),
        test_node(8, "narrow_gamma2", vec![7], gamma_shape.clone()),
        test_node(9, "narrow_beta2", vec![7], gamma_shape),
        // Node 10: second adain — inputs [conv1_out=6, gamma2=8, beta2=9]
        test_node(10, "adain2", vec![6, 8, 9], shape.clone()),
        // Node 11: conv2 — input is [adain2_out=10]
        test_node(11, "conv2", vec![10], shape.clone()),
        // Node 12: add — inputs are [x=0, conv2_out=11]
        test_node(12, "add", vec![0, 11], shape),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // FusedResBlock should be at step 12 (the add position).
    match &steps[12] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::FusedResBlock {
                    phase1,
                    phase2,
                    input_steps,
                    residual_scale,
                    style_proj,
                    shortcut_step,
                    ..
                },
            ..
        } => {
            assert!(style_proj.is_none(), "expected no style_proj");
            assert!(shortcut_step.is_none(), "expected identity shortcut");
            assert_eq!(phase1.activation, NormActivation::LeakyRelu { slope: 0.2 });
            assert_eq!(phase2.activation, NormActivation::LeakyRelu { slope: 0.2 });
            // input_steps: [x=0, gamma1=1, beta1=2, gamma2=8, beta2=9]
            assert_eq!(input_steps, &[0, 1, 2, 8, 9]);
            assert!((*residual_scale - 1.0).abs() < f32::EPSILON);
        }
        other => panic!(
            "expected FusedResBlock at step 12, got: {:?}",
            std::mem::discriminant(other)
        ),
    }

    // Steps 5, 6, 10, 11 should be IdentityPassthrough (fused away).
    for idx in [5, 6, 10, 11] {
        assert!(
            matches!(&steps[idx], CompiledStep::IdentityPassthrough),
            "expected IdentityPassthrough at step {idx}"
        );
    }

    // Steps 7, 8, 9 should remain unchanged (style projection still needed).
    // (They were already IdentityPassthrough in this test, but in production
    // they'd be Dispatch steps for Linear/Narrow.)
}

/// Test that the post-add mul_scalar pattern is absorbed into residual_scale.
///
/// Layout: standard ResBlock + ConstantValue + Dispatch "mul" at the end.
#[test]
fn test_resblock_absorbs_post_add_scale() {
    let shape = vec![1, 256, 100];
    let gamma_shape = vec![1, 256, 1];
    let inv_sqrt2: f64 = 1.0 / std::f64::consts::SQRT_2;

    let mut steps = vec![
        CompiledStep::InputForward, // 0: x
        CompiledStep::InputForward, // 1: gamma1
        CompiledStep::InputForward, // 2: beta1
        CompiledStep::InputForward, // 3: gamma2
        CompiledStep::InputForward, // 4: beta2
        make_norm_activ_conv1d(NormActivation::LeakyRelu { slope: 0.2 }, &shape, 256, 3, 1), // 5
        CompiledStep::IdentityPassthrough, // 6
        make_norm_activ_conv1d(NormActivation::LeakyRelu { slope: 0.2 }, &shape, 256, 3, 1), // 7
        CompiledStep::IdentityPassthrough, // 8
        make_add_dispatch(&shape),  // 9: add
        CompiledStep::ConstantValue {
            value: inv_sqrt2,
            shape: vec![1],
        }, // 10: inv_sqrt2
        make_mul_dispatch(&shape),  // 11: mul (scale)
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, shape.clone()),
        test_input_node(1, gamma_shape.clone()),
        test_input_node(2, gamma_shape.clone()),
        test_input_node(3, gamma_shape.clone()),
        test_input_node(4, gamma_shape),
        test_node(5, "adain1", vec![0, 1, 2], shape.clone()),
        test_node(6, "conv1", vec![5], shape.clone()),
        test_node(7, "adain2", vec![6, 3, 4], shape.clone()),
        test_node(8, "conv2", vec![7], shape.clone()),
        test_node(9, "add", vec![0, 8], shape.clone()),
        test_constant_node(10, inv_sqrt2),
        // Node 11: mul — inputs are [add_out=9, const=10]
        test_node(11, "mul", vec![9, 10], shape),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // FusedResBlock should be at step 11 (mul position), absorbing the scale.
    match &steps[11] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::FusedResBlock {
                    residual_scale,
                    input_steps,
                    ..
                },
            ..
        } => {
            assert!(
                (*residual_scale - inv_sqrt2 as f32).abs() < 1e-6,
                "expected residual_scale ≈ {inv_sqrt2}, got {residual_scale}"
            );
            assert_eq!(input_steps, &[0, 1, 2, 3, 4]);
        }
        other => panic!(
            "expected FusedResBlock at step 11, got: {:?}",
            std::mem::discriminant(other)
        ),
    }

    // Steps 5-8 should be IdentityPassthrough (fused NormActivConv1d + conv pairs).
    for (idx, step) in steps[5..=8].iter().enumerate() {
        assert!(
            matches!(step, CompiledStep::IdentityPassthrough),
            "expected IdentityPassthrough at step {}, got {:?}",
            idx + 5,
            std::mem::discriminant(step)
        );
    }
    // Step 9 (add) should be IdentityPassthrough (absorbed into FusedResBlock).
    assert!(
        matches!(&steps[9], CompiledStep::IdentityPassthrough),
        "expected IdentityPassthrough at step 9 (add)"
    );
    // Step 10 (ConstantValue) should remain as-is — root nodes have no
    // edge_map entries, so replacing with IP would fail at execution.
    assert!(
        matches!(&steps[10], CompiledStep::ConstantValue { .. }),
        "ConstantValue at step 10 should be preserved, got {:?}",
        std::mem::discriminant(&steps[10])
    );
}

/// Test that the post-add mul_scalar with ConstantWeight is absorbed.
///
/// This is the F0 AdainResBlk1d pattern: `mul_scalar(1/√2)` traces as
/// ConstantWeight (not ConstantValue) because `scalar_like` creates a
/// tensor with `trace_node_id: None` that gets auto-registered.
#[test]
fn test_resblock_absorbs_constant_weight_scale() {
    use nn_core::dyn_tensor::trace::WeightRef;

    let shape = vec![1, 256, 100];
    let gamma_shape = vec![1, 256, 1];
    let inv_sqrt2_f32 = 1.0_f32 / std::f32::consts::SQRT_2;

    // Build ConstantWeight NativeOp (same as compile_node produces).
    let cw_weight = WeightRef::new(vec![inv_sqrt2_f32], vec![]).unwrap();
    let mut cw_weight_data = HashMap::new();
    cw_weight_data.insert("constant_weight".to_string(), cw_weight.clone());

    let mut steps = vec![
        CompiledStep::InputForward, // 0: x
        CompiledStep::InputForward, // 1: gamma1
        CompiledStep::InputForward, // 2: beta1
        CompiledStep::InputForward, // 3: gamma2
        CompiledStep::InputForward, // 4: beta2
        make_norm_activ_conv1d(NormActivation::LeakyRelu { slope: 0.2 }, &shape, 256, 3, 1), // 5
        CompiledStep::IdentityPassthrough, // 6
        make_norm_activ_conv1d(NormActivation::LeakyRelu { slope: 0.2 }, &shape, 256, 3, 1), // 7
        CompiledStep::IdentityPassthrough, // 8
        make_add_dispatch(&shape),  // 9: add
        CompiledStep::NativeOp {
            op: NativeOpKind::ConstantWeight {
                name: "constant_weight".into(),
                shape: vec![],
            },
            weight_data: cw_weight_data,
        }, // 10: scalar ConstantWeight (1/√2)
        make_mul_dispatch(&shape),  // 11: mul (scale)
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, shape.clone()),
        test_input_node(1, gamma_shape.clone()),
        test_input_node(2, gamma_shape.clone()),
        test_input_node(3, gamma_shape.clone()),
        test_input_node(4, gamma_shape),
        test_node(5, "adain1", vec![0, 1, 2], shape.clone()),
        test_node(6, "conv1", vec![5], shape.clone()),
        test_node(7, "adain2", vec![6, 3, 4], shape.clone()),
        test_node(8, "conv2", vec![7], shape.clone()),
        test_node(9, "add", vec![0, 8], shape.clone()),
        // Node 10: ConstantWeight (scalar 1/√2)
        TraceNode::new(
            10,
            "inv_sqrt2".into(),
            TraceOp::ConstantWeight { weight: cw_weight },
            vec![],
            vec![],
            DType::F32,
        ),
        // Node 11: mul — inputs are [add_out=9, const_weight=10]
        test_node(11, "mul", vec![9, 10], shape),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // FusedResBlock should be at step 11 (mul position), absorbing the scale.
    match &steps[11] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::FusedResBlock {
                    residual_scale,
                    input_steps,
                    ..
                },
            ..
        } => {
            assert!(
                (*residual_scale - inv_sqrt2_f32).abs() < 1e-6,
                "expected residual_scale ≈ {inv_sqrt2_f32}, got {residual_scale}"
            );
            assert_eq!(input_steps, &[0, 1, 2, 3, 4]);
        }
        other => panic!(
            "expected FusedResBlock at step 11, got: {:?}",
            std::mem::discriminant(other)
        ),
    }

    // Step 9 (add) should be IdentityPassthrough (absorbed).
    assert!(
        matches!(&steps[9], CompiledStep::IdentityPassthrough),
        "expected IdentityPassthrough at step 9 (add), got {:?}",
        std::mem::discriminant(&steps[9])
    );
}

/// Test that upsample ResBlocks fuse into FusedResBlock with pool_step (#3510).
///
/// Upsample blocks have the pattern:
///   AdainLeakyRelu → ConvTranspose1d(pool) → Conv1d → NormActivConv1d(phase2) → conv2_IP → add → const → mul
///
/// Phase1 was NOT fused by Pass 1 because a pool sits between AdaIN and Conv1d.
/// Pass 2 should detect the unfused pattern, produce FusedResBlock with
/// `pool_step = Some(pool_idx)`, and leave adain1 + pool as live steps.
#[test]
fn test_resblock_fuses_upsample_with_pool_step() {
    use nn_core::dyn_tensor::trace::WeightRef;

    let channels = 256;
    let time = 100;
    // After pool (ConvTranspose1d stride=2), time doubles.
    let pool_time = time * 2;
    let shape = vec![1, channels, time];
    let pool_shape = vec![1, channels, pool_time];
    let gamma_shape = vec![1, channels, 1];
    let inv_sqrt2: f64 = 1.0 / std::f64::consts::SQRT_2;

    // Build a standalone AdainLeakyRelu NativeOp (phase1 norm+activation, NOT fused).
    let adain1_step = CompiledStep::NativeOp {
        op: NativeOpKind::AdainLeakyRelu {
            eps: 1e-5,
            slope: 0.2,
            input_shape: shape.clone(),
            external_node_ids: None,
        },
        weight_data: HashMap::new(),
    };

    // Build a pool step (ConvTranspose1d). The peephole doesn't inspect the
    // pool's kernel type — it just needs a single-input single-output intermediate.
    // Use a placeholder Dispatch "conv_transpose_1d" step.
    let pool_step = {
        use crate::tensor_block_builder::TensorBlockBuilder;

        let mut b = TensorBlockBuilder::new("conv_transpose_1d");
        let inp = b.add_input("input_0", &shape);
        // The builder creates a dummy output for our test; the actual ConvTranspose1d
        // params don't matter for fusion detection — only graph topology matters.
        let w_id = b.add_input("weight", &[channels, channels, 4]);
        let out = b.add_conv_transpose_1d(
            inp,
            w_id,
            None,
            2, // stride
            1, // padding
            1, // dilation
            1, // groups
            0, // output_padding
            &pool_shape,
        );
        let def = b.build(out).expect("valid conv_transpose_1d IR");

        let mut wd = HashMap::new();
        wd.insert(
            "weight".to_string(),
            WeightRef::new(
                vec![0.0f32; channels * channels * 4],
                vec![channels, channels, 4],
            )
            .expect("valid pool weight"),
        );
        CompiledStep::Dispatch {
            kernel: CompiledKernel::new(def),
            weight_data: wd,
            external_node_ids: None,
        }
    };

    // Build an unfused conv1d Dispatch step (phase1 conv, NOT fused with adain1).
    let conv1_step = {
        use crate::tensor_block_builder::TensorBlockBuilder;

        let mut b = TensorBlockBuilder::new("conv1d");
        let inp = b.add_input("input_0", &pool_shape);
        let w_id = b.add_input("weight", &[channels, channels, 3]);
        let b_id = b.add_input("bias", &[channels]);
        let out = b.add_conv1d(inp, w_id, Some(b_id), 1, 1, &pool_shape);
        let def = b.build(out).expect("valid conv1d IR");

        let mut wd = HashMap::new();
        wd.insert(
            "weight".to_string(),
            WeightRef::new(
                vec![0.0f32; channels * channels * 3],
                vec![channels, channels, 3],
            )
            .expect("valid conv weight"),
        );
        wd.insert(
            "bias".to_string(),
            WeightRef::new(vec![0.0f32; channels], vec![channels]).expect("valid conv bias"),
        );
        CompiledStep::Dispatch {
            kernel: CompiledKernel::new(def),
            weight_data: wd,
            external_node_ids: None,
        }
    };

    let mut steps = vec![
        CompiledStep::InputForward, // 0: x
        CompiledStep::InputForward, // 1: gamma1
        CompiledStep::InputForward, // 2: beta1
        CompiledStep::InputForward, // 3: gamma2
        CompiledStep::InputForward, // 4: beta2
        adain1_step,                // 5: standalone AdainLeakyRelu (unfused phase1 norm+activ)
        pool_step,                  // 6: ConvTranspose1d (pool/upsample)
        conv1_step,                 // 7: unfused Conv1d (phase1 conv)
        make_norm_activ_conv1d(
            // 8: fused NormActivConv1d (phase2)
            NormActivation::LeakyRelu { slope: 0.2 },
            &pool_shape,
            channels,
            3,
            1,
        ),
        CompiledStep::IdentityPassthrough, // 9: conv2 passthrough (post-Pass-1)
        make_add_dispatch(&pool_shape),    // 10: residual add
        CompiledStep::ConstantValue {
            // 11: inv_sqrt2 constant
            value: inv_sqrt2,
            shape: vec![1],
        },
        make_mul_dispatch(&pool_shape), // 12: mul (residual scale)
    ];

    // Graph topology: adain1(x,g1,b1) → pool → conv1 → adain2(_, g2, b2) → conv2 → add(x, _) → mul(_, const)
    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, shape.clone()),
        test_input_node(1, gamma_shape.clone()),
        test_input_node(2, gamma_shape.clone()),
        test_input_node(3, gamma_shape.clone()),
        test_input_node(4, gamma_shape),
        // Node 5: standalone adain1 — inputs [x=0, gamma1=1, beta1=2]
        test_node(5, "adain1", vec![0, 1, 2], shape),
        // Node 6: pool (ConvTranspose1d) — input [adain1_out=5]
        test_node(6, "pool", vec![5], pool_shape.clone()),
        // Node 7: conv1 (unfused) — input [pool_out=6]
        test_node(7, "conv1", vec![6], pool_shape.clone()),
        // Node 8: adain2 (NormActivConv1d) — inputs [conv1_out=7, gamma2=3, beta2=4]
        test_node(8, "adain2", vec![7, 3, 4], pool_shape.clone()),
        // Node 9: conv2 (IdentityPassthrough) — input [adain2_out=8]
        test_node(9, "conv2", vec![8], pool_shape.clone()),
        // Node 10: add — inputs [x=0, conv2_out=9]
        test_node(10, "add", vec![0, 9], pool_shape.clone()),
        // Node 11: constant (1/sqrt(2))
        test_constant_node(11, inv_sqrt2),
        // Node 12: mul — inputs [add_out=10, const=11]
        test_node(12, "mul", vec![10, 11], pool_shape),
    ]);

    super::super::apply_peephole(&mut steps, &graph);

    // FusedResBlock should be at step 12 (mul position, absorbing scale).
    match &steps[12] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::FusedResBlock {
                    phase1,
                    phase2,
                    input_steps,
                    residual_scale,
                    style_proj,
                    shortcut_step,
                    pool_step: fused_pool_step,
                    ..
                },
            weight_data,
        } => {
            // Pool step should reference the ConvTranspose1d at index 6.
            assert_eq!(
                *fused_pool_step,
                Some(6),
                "expected pool_step = Some(6), got {fused_pool_step:?}"
            );

            assert!(style_proj.is_none(), "expected no style_proj");
            assert!(shortcut_step.is_none(), "expected identity shortcut");

            // Phase 1 params (from standalone AdaIN + Conv1d).
            assert_eq!(phase1.activation, NormActivation::LeakyRelu { slope: 0.2 });
            assert!((phase1.eps - 1e-5).abs() < 1e-10);
            assert_eq!(phase1.conv_dilation, 1);
            assert_eq!(phase1.conv_padding, 1);
            assert_eq!(phase1.output_channels, channels);
            assert_eq!(phase1.kernel_size, 3);

            // Phase 2 params (from fused NormActivConv1d).
            assert_eq!(phase2.activation, NormActivation::LeakyRelu { slope: 0.2 });
            assert_eq!(phase2.conv_dilation, 1);
            assert_eq!(phase2.output_channels, channels);

            // input_steps: [x=0, gamma1=1, beta1=2, gamma2=3, beta2=4]
            assert_eq!(input_steps, &[0, 1, 2, 3, 4]);

            // Residual scale = 1/sqrt(2).
            assert!(
                (*residual_scale - inv_sqrt2 as f32).abs() < 1e-6,
                "expected residual_scale ≈ {inv_sqrt2}, got {residual_scale}"
            );

            // Weight data should have p1_ and p2_ prefixed keys.
            assert!(
                weight_data.contains_key("p1_conv_weight"),
                "missing p1_conv_weight"
            );
            assert!(
                weight_data.contains_key("p1_conv_bias"),
                "missing p1_conv_bias"
            );
            assert!(
                weight_data.contains_key("p2_conv_weight"),
                "missing p2_conv_weight"
            );
            assert!(
                weight_data.contains_key("p2_conv_bias"),
                "missing p2_conv_bias"
            );
            // LeakyRelu has no alpha weight.
            assert!(!weight_data.contains_key("p1_alpha"));
        }
        other => panic!(
            "expected FusedResBlock at step 12, got: {:?}",
            std::mem::discriminant(other)
        ),
    }

    // Step 5 (adain1) should REMAIN as a live NativeOp (not replaced).
    assert!(
        matches!(
            &steps[5],
            CompiledStep::NativeOp {
                op: NativeOpKind::AdainLeakyRelu { .. },
                ..
            }
        ),
        "adain1 at step 5 should remain live (not replaced with IdentityPassthrough)"
    );

    // Step 6 (pool) should REMAIN as a live Dispatch step (not replaced).
    assert!(
        matches!(&steps[6], CompiledStep::Dispatch { .. }),
        "pool at step 6 should remain live (not replaced with IdentityPassthrough)"
    );

    // Step 7 (conv1, unfused) should be IdentityPassthrough (absorbed into FusedResBlock).
    assert!(
        matches!(&steps[7], CompiledStep::IdentityPassthrough),
        "conv1 at step 7 should be IdentityPassthrough (absorbed)"
    );

    // Step 8 (NormActivConv1d phase2) should be IdentityPassthrough.
    assert!(
        matches!(&steps[8], CompiledStep::IdentityPassthrough),
        "phase2 NormActivConv1d at step 8 should be IdentityPassthrough (absorbed)"
    );

    // Step 9 (conv2 passthrough) should be IdentityPassthrough.
    assert!(
        matches!(&steps[9], CompiledStep::IdentityPassthrough),
        "conv2 at step 9 should be IdentityPassthrough (absorbed)"
    );

    // Step 10 (add) should be IdentityPassthrough (absorbed into scale).
    assert!(
        matches!(&steps[10], CompiledStep::IdentityPassthrough),
        "add at step 10 should be IdentityPassthrough (absorbed)"
    );
}
