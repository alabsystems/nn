// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the peephole optimizer (NormActivConv1d fusion).

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use super::super::{CompiledKernel, CompiledStep, GemmActivation, NativeOpKind, NormActivation};

/// Helper: build a TraceNode for test graph construction.
fn test_node(id: u64, name: &str, inputs: Vec<u64>, shape: Vec<usize>) -> TraceNode {
    // Use Relu as a dummy op — the peephole only inspects CompiledSteps,
    // not TraceOps. The graph is only used for use-count analysis.
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

/// Helper: build a fake Conv1d CompiledStep::Dispatch with the right kernel name
/// and weight data.
fn make_conv1d_dispatch(
    input_shape: &[usize],
    output_shape: &[usize],
    weight_shape: &[usize],
    padding: usize,
    dilation: usize,
) -> CompiledStep {
    use crate::tensor_block_builder::TensorBlockBuilder;

    let bias_shape = [weight_shape[0]];
    let mut b = TensorBlockBuilder::new("conv1d");
    let input = b.add_input("input_0", input_shape);
    let w = b.add_input("weight", weight_shape);
    let bi = b.add_input("bias", &bias_shape);
    let output = b.add_conv1d_full(input, w, Some(bi), 1, padding, dilation, 1, output_shape);
    let def = b.build(output).expect("valid conv1d IR");

    let mut weight_data = HashMap::new();
    weight_data.insert(
        "weight".to_string(),
        WeightRef::new(
            vec![0.0f32; weight_shape.iter().product()],
            weight_shape.to_vec(),
        )
        .expect("valid weight"),
    );
    weight_data.insert(
        "bias".to_string(),
        WeightRef::new(vec![0.0f32; bias_shape[0]], bias_shape.to_vec()).expect("valid bias"),
    );

    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: None,
    }
}

/// Helper: build an AdainLeakyRelu NativeOp step.
fn make_adain_leaky_relu(input_shape: &[usize]) -> CompiledStep {
    CompiledStep::NativeOp {
        op: NativeOpKind::AdainLeakyRelu {
            eps: 1e-5,
            slope: 0.2,
            input_shape: input_shape.to_vec(),
            external_node_ids: None,
        },
        weight_data: HashMap::new(),
    }
}

/// Helper: build an AdainSnake NativeOp step with alpha weight.
fn make_adain_snake(input_shape: &[usize], channels: usize) -> CompiledStep {
    let mut weight_data = HashMap::new();
    weight_data.insert(
        "alpha".to_string(),
        WeightRef::new(vec![1.0f32; channels], vec![channels]).expect("valid alpha"),
    );
    CompiledStep::NativeOp {
        op: NativeOpKind::AdainSnake {
            eps: 1e-5,
            input_shape: input_shape.to_vec(),
            channels,
            residual_gamma: true,
            external_node_ids: None,
        },
        weight_data,
    }
}

#[test]
fn test_peephole_fuses_adain_leaky_relu_conv1d() {
    let input_shape = [1, 512, 100];
    let output_shape = [1, 512, 100];
    let weight_shape = [512, 512, 3];

    let mut steps = vec![
        CompiledStep::InputForward,          // node 0: x
        CompiledStep::InputForward,          // node 1: gamma
        CompiledStep::InputForward,          // node 2: beta
        make_adain_leaky_relu(&input_shape), // node 3: adain
        make_conv1d_dispatch(&input_shape, &output_shape, &weight_shape, 1, 1), // node 4: conv
    ];

    // Build graph: node 3 consumes node 0,1,2. Node 4 consumes only node 3.
    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, input_shape.to_vec()),
        test_input_node(1, vec![1, 512, 1]),
        test_input_node(2, vec![1, 512, 1]),
        test_node(3, "adain", vec![0, 1, 2], input_shape.to_vec()),
        test_node(4, "conv1d", vec![3], output_shape.to_vec()),
    ]);

    super::apply_peephole(&mut steps, &graph);

    // Step 3 should be NormActivConv1d (at the AdaIN position, which has
    // 3 edge_map entries [x, gamma, beta] matching executor needs).
    match &steps[3] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::NormActivConv1d {
                    activation,
                    eps,
                    conv_dilation,
                    conv_padding,
                    input_shape: fused_shape,
                    output_channels,
                    kernel_size,
                    ..
                },
            weight_data,
        } => {
            assert_eq!(*activation, NormActivation::LeakyRelu { slope: 0.2 });
            assert!((eps - 1e-5).abs() < 1e-10);
            assert_eq!(*conv_dilation, 1);
            assert_eq!(*conv_padding, 1);
            assert_eq!(fused_shape, &input_shape);
            assert_eq!(*output_channels, 512);
            assert_eq!(*kernel_size, 3);
            assert!(weight_data.contains_key("conv_weight"));
            assert!(weight_data.contains_key("conv_bias"));
        }
        other => panic!(
            "expected NormActivConv1d at step 3, got: {:?}",
            std::mem::discriminant(other)
        ),
    }

    // Step 4 should be IdentityPassthrough (conv1d position passthrough).
    assert!(
        matches!(&steps[4], CompiledStep::IdentityPassthrough),
        "expected IdentityPassthrough at step 4, got: {:?}",
        std::mem::discriminant(&steps[4])
    );
}

#[test]
fn test_peephole_does_not_fuse_when_fanout_gt_1() {
    let input_shape = [1, 512, 100];
    let output_shape = [1, 512, 100];
    let weight_shape = [512, 512, 3];

    let mut steps = vec![
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        make_adain_leaky_relu(&input_shape),
        make_conv1d_dispatch(&input_shape, &output_shape, &weight_shape, 1, 1),
        CompiledStep::IdentityPassthrough, // placeholder for node 5
    ];

    // Graph: node 3 has TWO consumers (node 4 AND node 5) → should NOT fuse.
    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, input_shape.to_vec()),
        test_input_node(1, vec![1, 512, 1]),
        test_input_node(2, vec![1, 512, 1]),
        test_node(3, "adain", vec![0, 1, 2], input_shape.to_vec()),
        test_node(4, "conv1d", vec![3], output_shape.to_vec()),
        test_node(5, "other", vec![3], input_shape.to_vec()),
    ]);

    super::apply_peephole(&mut steps, &graph);

    // Should NOT fuse because AdaIN has fan-out > 1.
    assert!(
        matches!(
            &steps[3],
            CompiledStep::NativeOp {
                op: NativeOpKind::AdainLeakyRelu { .. },
                ..
            }
        ),
        "expected AdainLeakyRelu to remain unfused"
    );
}

#[test]
fn test_peephole_fuses_adain_snake_conv1d() {
    let input_shape = [1, 256, 200];
    let output_shape = [1, 256, 200];
    let weight_shape = [256, 256, 3];

    let mut steps = vec![
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        make_adain_snake(&input_shape, 256),
        make_conv1d_dispatch(&input_shape, &output_shape, &weight_shape, 1, 1),
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, input_shape.to_vec()),
        test_input_node(1, vec![1, 256, 1]),
        test_input_node(2, vec![1, 256, 1]),
        test_node(3, "adain_snake", vec![0, 1, 2], input_shape.to_vec()),
        test_node(4, "conv1d", vec![3], output_shape.to_vec()),
    ]);

    super::apply_peephole(&mut steps, &graph);

    // Step 3 should be NormActivConv1d with Snake activation (at AdaIN pos).
    match &steps[3] {
        CompiledStep::NativeOp {
            op: NativeOpKind::NormActivConv1d { activation, .. },
            weight_data,
        } => {
            assert_eq!(*activation, NormActivation::Snake);
            assert!(
                weight_data.contains_key("alpha"),
                "alpha weight should be preserved"
            );
            assert!(weight_data.contains_key("conv_weight"));
        }
        other => panic!(
            "expected NormActivConv1d at step 3, got: {:?}",
            std::mem::discriminant(other)
        ),
    }

    // Step 4 should be IdentityPassthrough.
    assert!(matches!(&steps[4], CompiledStep::IdentityPassthrough));
}

// -- Linear + Activation fusion tests (Pass 4) -------------------------------

/// Helper: build a fake Linear CompiledStep::Dispatch.
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

/// Helper: build a single-op activation Dispatch (e.g., "relu", "gelu").
fn make_activation_dispatch(name: &str, shape: &[usize]) -> CompiledStep {
    use crate::tensor_block_builder::TensorBlockBuilder;

    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("input_0", shape);
    let output = match name {
        "relu" => b.add_relu(input, shape),
        "gelu" => b.add_gelu(input, shape),
        "gelu_erf" => b.add_gelu_erf(input, shape),
        "sigmoid" => b.add_sigmoid(input, shape),
        "tanh" => {
            let kernel = crate::tensor_builders::unary_kernel("tanh", crate::ir::UnaryFnKind::Tanh);
            b.add_elementwise(kernel, &[input], shape)
        }
        _ => panic!("unsupported test activation: {name}"),
    };
    let def = b.build(output).expect("valid activation IR");

    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: None,
    }
}

#[test]
fn test_peephole_fuses_linear_relu() {
    let input_shape = [4, 768];
    let out_features = 3072;
    let output_shape = [4, 3072];

    let mut steps = vec![
        CompiledStep::InputForward,                             // node 0: input
        make_linear_dispatch(&input_shape, out_features, true), // node 1: linear
        make_activation_dispatch("relu", &output_shape),        // node 2: relu
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, input_shape.to_vec()),
        test_node(1, "linear", vec![0], output_shape.to_vec()),
        test_node(2, "relu", vec![1], output_shape.to_vec()),
    ]);

    super::apply_peephole(&mut steps, &graph);

    match &steps[1] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::LinearActivation {
                    activation,
                    in_features,
                    out_features: out_f,
                    has_bias,
                    input_shape: fused_input,
                },
            weight_data,
        } => {
            assert_eq!(*activation, GemmActivation::Relu);
            assert_eq!(*in_features, 768);
            assert_eq!(*out_f, 3072);
            assert!(*has_bias);
            assert_eq!(fused_input, &input_shape);
            assert!(weight_data.contains_key("weight"));
            assert!(weight_data.contains_key("bias"));
        }
        other => panic!(
            "expected LinearActivation at step 1, got: {:?}",
            std::mem::discriminant(other)
        ),
    }

    assert!(
        matches!(&steps[2], CompiledStep::IdentityPassthrough),
        "expected IdentityPassthrough at step 2"
    );
}

#[test]
fn test_peephole_fuses_linear_gelu_erf() {
    let input_shape = [1, 32, 768];
    let out_features = 3072;
    let output_shape = [1, 32, 3072];

    let mut steps = vec![
        CompiledStep::InputForward,
        make_linear_dispatch(&input_shape, out_features, true),
        make_activation_dispatch("gelu_erf", &output_shape),
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, input_shape.to_vec()),
        test_node(1, "linear", vec![0], output_shape.to_vec()),
        test_node(2, "gelu_erf", vec![1], output_shape.to_vec()),
    ]);

    super::apply_peephole(&mut steps, &graph);

    match &steps[1] {
        CompiledStep::NativeOp {
            op: NativeOpKind::LinearActivation { activation, .. },
            ..
        } => {
            assert_eq!(*activation, GemmActivation::GeluErf);
        }
        other => panic!(
            "expected LinearActivation(GeluErf) at step 1, got: {:?}",
            std::mem::discriminant(other)
        ),
    }
}

#[test]
fn test_peephole_no_fuse_linear_fanout_gt_1() {
    let input_shape = [4, 768];
    let out_features = 3072;
    let output_shape = [4, 3072];

    let mut steps = vec![
        CompiledStep::InputForward,
        make_linear_dispatch(&input_shape, out_features, true),
        make_activation_dispatch("relu", &output_shape),
        CompiledStep::IdentityPassthrough, // node 3: second consumer of linear
    ];

    // Node 1 (linear) has TWO consumers: node 2 and node 3.
    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, input_shape.to_vec()),
        test_node(1, "linear", vec![0], output_shape.to_vec()),
        test_node(2, "relu", vec![1], output_shape.to_vec()),
        test_node(3, "other", vec![1], output_shape.to_vec()),
    ]);

    super::apply_peephole(&mut steps, &graph);

    // Should NOT fuse — linear has fan-out > 1.
    assert!(
        matches!(&steps[1], CompiledStep::Dispatch { kernel, .. } if kernel.name() == "linear"),
        "linear should remain unfused with fan-out > 1"
    );
}

#[test]
fn test_peephole_fuses_linear_no_bias() {
    let input_shape = [4, 768];
    let out_features = 3072;
    let output_shape = [4, 3072];

    let mut steps = vec![
        CompiledStep::InputForward,
        make_linear_dispatch(&input_shape, out_features, false),
        make_activation_dispatch("sigmoid", &output_shape),
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, input_shape.to_vec()),
        test_node(1, "linear", vec![0], output_shape.to_vec()),
        test_node(2, "sigmoid", vec![1], output_shape.to_vec()),
    ]);

    super::apply_peephole(&mut steps, &graph);

    match &steps[1] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::LinearActivation {
                    activation,
                    has_bias,
                    ..
                },
            ..
        } => {
            assert_eq!(*activation, GemmActivation::Sigmoid);
            assert!(!has_bias);
        }
        other => panic!(
            "expected LinearActivation(Sigmoid) at step 1, got: {:?}",
            std::mem::discriminant(other)
        ),
    }
}

#[test]
fn test_peephole_all_activation_variants() {
    let cases = [
        ("relu", GemmActivation::Relu),
        ("gelu", GemmActivation::Gelu),
        ("gelu_erf", GemmActivation::GeluErf),
        ("sigmoid", GemmActivation::Sigmoid),
        ("tanh", GemmActivation::Tanh),
    ];

    let input_shape = [2, 64];
    let out_features = 128;
    let output_shape = [2, 128];

    for (name, expected) in &cases {
        let mut steps = vec![
            CompiledStep::InputForward,
            make_linear_dispatch(&input_shape, out_features, true),
            make_activation_dispatch(name, &output_shape),
        ];

        let graph = ComputationGraph::from_nodes(vec![
            test_input_node(0, input_shape.to_vec()),
            test_node(1, "linear", vec![0], output_shape.to_vec()),
            test_node(2, name, vec![1], output_shape.to_vec()),
        ]);

        super::apply_peephole(&mut steps, &graph);

        match &steps[1] {
            CompiledStep::NativeOp {
                op: NativeOpKind::LinearActivation { activation, .. },
                ..
            } => {
                assert_eq!(activation, expected, "failed for activation: {name}");
            }
            other => panic!(
                "expected LinearActivation({name}) at step 1, got: {:?}",
                std::mem::discriminant(other)
            ),
        }
    }
}

// -- Pass ordering interaction: AddLayerNorm (Pass 6) before NormLinear (Pass 7) --

/// Build a Dispatch{add} step for pass-ordering tests.
fn make_add_dispatch_for_ordering(input_shape: &[usize]) -> CompiledStep {
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

/// Build a LayerNorm NativeOp step for pass-ordering tests.
fn make_layernorm_native_for_ordering(
    input_shape: &[usize],
    hidden_dim: usize,
    eps: f32,
) -> CompiledStep {
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

/// Build a Linear Dispatch step for pass-ordering tests.
fn make_linear_dispatch_for_ordering(
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

/// Regression test: `Add → LayerNorm → Linear` must produce `AddLayerNorm`,
/// NOT `NormLinear`. Pass 6 (AddLayerNorm) must run before Pass 7 (NormLinear)
/// so the LayerNorm is consumed by AddLayerNorm first. If the ordering is
/// reversed, NormLinear would consume the LayerNorm, leaving the Add unfused.
///
/// Graph: input_0 ─┐
///        input_1 ─┤→ Add → LayerNorm → Linear → [output]
///
/// Expected after peephole (with add_norm_linear pass enabled):
///   step 0: InputForward
///   step 1: InputForward
///   step 2: NativeOp{AddNormLinear}  (fused Add + LayerNorm + Linear)
///   step 3: IdentityPassthrough      (LayerNorm position consumed)
///   step 4: IdentityPassthrough      (Linear position consumed)
#[test]
fn test_pass_ordering_add_ln_before_norm_linear() {
    let shape = vec![4, 768];
    let hidden_dim = 768;
    let out_features = 3072;

    let mut steps = vec![
        CompiledStep::InputForward,                                   // 0: input_0
        CompiledStep::InputForward,                                   // 1: input_1
        make_add_dispatch_for_ordering(&shape),                       // 2: add(0, 1)
        make_layernorm_native_for_ordering(&shape, hidden_dim, 1e-5), // 3: LN(2)
        make_linear_dispatch_for_ordering(&shape, out_features, true), // 4: linear(3)
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, shape.clone()),
        test_input_node(1, shape.clone()),
        test_node(2, "add", vec![0, 1], shape.clone()),
        test_node(3, "layernorm", vec![2], shape),
        test_node(4, "linear", vec![3], vec![4, out_features]),
    ]);

    super::apply_peephole(&mut steps, &graph);

    // Step 2: must be AddNormLinear (Add + LN + Linear fused).
    // Pass 6 fuses Add+LN → AddLayerNorm, then pass 8 fuses AddLayerNorm+Linear → AddNormLinear.
    match &steps[2] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::AddNormLinear {
                    eps,
                    hidden_dim: hd,
                    out_features: of,
                    ..
                },
            weight_data,
        } => {
            assert!((eps - 1e-5).abs() < 1e-10, "AddNormLinear eps mismatch");
            assert_eq!(*hd, hidden_dim, "AddNormLinear hidden_dim mismatch");
            assert_eq!(*of, out_features, "AddNormLinear out_features mismatch");
            assert!(
                weight_data.contains_key("norm_weight"),
                "norm_weight missing"
            );
            assert!(weight_data.contains_key("norm_bias"), "norm_bias missing");
            assert!(weight_data.contains_key("weight"), "linear weight missing");
        }
        CompiledStep::NativeOp {
            op: NativeOpKind::NormLinear { .. },
            ..
        } => {
            panic!(
                "step 2 became NormLinear — pass ordering is wrong! \
                 AddLayerNorm (Pass 6) must run before NormLinear (Pass 7)."
            );
        }
        other => panic!(
            "expected AddNormLinear at step 2, got: {:?}",
            std::mem::discriminant(other)
        ),
    }

    // Step 3: must be IdentityPassthrough (LayerNorm consumed by AddLayerNorm → AddNormLinear).
    assert!(
        matches!(&steps[3], CompiledStep::IdentityPassthrough),
        "expected IdentityPassthrough at step 3 (consumed LayerNorm)"
    );

    // Step 4: must be IdentityPassthrough (Linear consumed by AddNormLinear).
    assert!(
        matches!(&steps[4], CompiledStep::IdentityPassthrough),
        "expected IdentityPassthrough at step 4 (consumed by AddNormLinear)"
    );
}

/// Verify that isolated `LayerNorm → Linear` (no preceding Add) still fuses
/// to NormLinear. Confirms NormLinear fusion works when AddLayerNorm doesn't match.
#[test]
fn test_norm_linear_fuses_without_preceding_add() {
    let shape = vec![4, 768];
    let hidden_dim = 768;
    let out_features = 3072;

    let mut steps = vec![
        CompiledStep::InputForward,                                    // 0: input
        make_layernorm_native_for_ordering(&shape, hidden_dim, 1e-5),  // 1: LN(0)
        make_linear_dispatch_for_ordering(&shape, out_features, true), // 2: linear(1)
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, shape.clone()),
        test_node(1, "layernorm", vec![0], shape),
        test_node(2, "linear", vec![1], vec![4, out_features]),
    ]);

    super::apply_peephole(&mut steps, &graph);

    // Step 1: must be FusedLayerNormLinear (LayerNorm NativeOp + Linear fused, #4252).
    match &steps[1] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::FusedLayerNormLinear {
                    hidden_dim: hd,
                    out_features: of,
                    ..
                },
            ..
        } => {
            assert_eq!(*hd, hidden_dim);
            assert_eq!(*of, out_features);
        }
        other => panic!(
            "expected FusedLayerNormLinear at step 1, got: {:?}",
            std::mem::discriminant(other)
        ),
    }

    assert!(matches!(&steps[2], CompiledStep::IdentityPassthrough));
}
