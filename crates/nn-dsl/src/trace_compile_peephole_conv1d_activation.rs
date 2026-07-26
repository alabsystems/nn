// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Peephole pass: fuse Conv1d + Activation into FusedConv1dActivation.
//!
//! Detects consecutive step pairs:
//!   Step i:   Dispatch { kernel: "conv1d" }
//!   Step i+1: Dispatch { kernel: "snake"/"snake_tensor" | "relu" | "leaky_relu" | "silu" | "gelu" | "gelu_erf" | "tanh" }
//!
//! The conv1d output must be single-consumer (use_counts == 1).
//!
//! Replaces the pair with:
//! - steps[i]   -> NativeOp { FusedConv1dActivation { ... } }
//! - steps[i+1] -> IdentityPassthrough
//!
//! Saves 1 dispatch per pair. In Kokoro, 4-8 such pairs exist in the
//! decoder/vocoder pipeline.
//!
//! Must run AFTER NormActivConv1d (pass 1) and FusedResBlock (pass 2)
//! which consume AdainSnake/LeakyRelu + Conv1d sequences. This pass
//! catches remaining standalone Conv1d -> Activation patterns.
//!
//! Part of #4264.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, WeightRef};

use super::super::{CompiledStep, ConvActivation, NativeOpKind};
use super::extract_conv1d_params;
use crate::tensor_ir::TensorOpKind;

/// Scan for consecutive Dispatch("conv1d") + Dispatch(activation) pairs
/// and fuse them into a single FusedConv1dActivation NativeOp.
pub(super) fn fuse_conv1d_activation(
    steps: &mut [CompiledStep],
    use_counts: &[usize],
    graph: &ComputationGraph,
) {
    let len = steps.len();
    if len < 2 {
        return;
    }

    let mut i = 0;
    while i + 1 < len {
        if try_fuse_conv1d_activation(steps, i, use_counts, graph) {
            // Skip past the fused pair.
            i += 2;
        } else {
            i += 1;
        }
    }
}

/// Try to fuse steps[i] (conv1d) with steps[i+1] (activation).
///
/// Returns `true` if the pair was fused (steps mutated in-place).
fn try_fuse_conv1d_activation(
    steps: &mut [CompiledStep],
    i: usize,
    use_counts: &[usize],
    graph: &ComputationGraph,
) -> bool {
    // ---- Step i: Dispatch with kernel name "conv1d" ----
    let (conv_info, conv_weight_data) = match &steps[i] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } if kernel.name() == "conv1d" => match extract_conv1d_params(kernel, weight_data) {
            Some(info) => (info, weight_data.clone()),
            None => return false,
        },
        _ => return false,
    };

    // Fan-out check: the conv1d output must have exactly 1 consumer.
    if use_counts.get(i).copied().unwrap_or(0) != 1 {
        return false;
    }

    // ---- Step i+1: Dispatch with activation kernel ----
    let (activation, alpha_weight) = match &steps[i + 1] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } => match kernel.name() {
            "snake" | "snake_tensor" => {
                let alpha = weight_data.get("alpha").cloned();
                (ConvActivation::Snake, alpha)
            }
            "relu" => (ConvActivation::Relu, None),
            "leaky_relu" => {
                let slope = extract_leaky_relu_slope(kernel);
                (ConvActivation::LeakyRelu { slope }, None)
            }
            "silu" => (ConvActivation::Silu, None),
            "gelu" => (ConvActivation::Gelu, None),
            "gelu_erf" => (ConvActivation::GeluErf, None),
            "tanh" => (ConvActivation::Tanh, None),
            _ => return false,
        },
        _ => return false,
    };

    // Build merged weight_data from conv weights.
    let mut merged_weight_data: HashMap<String, WeightRef> = HashMap::new();
    if let Some(w) = conv_info.weight {
        merged_weight_data.insert("weight".to_string(), w);
    }
    if let Some(b) = conv_info.bias {
        merged_weight_data.insert("bias".to_string(), b);
    }
    // Add alpha weight for Snake activation.
    if let Some(alpha) = alpha_weight {
        merged_weight_data.insert("alpha".to_string(), alpha);
    }

    // Compute input_shape from the conv1d kernel's first Input node.
    let input_shape = extract_conv1d_input_shape(&steps[i]);

    // Capture the graph node IDs from the conv1d step so the edge_map
    // builder can resolve edges generically.
    let ext_ids = graph.nodes().get(i).map(|node| node.inputs().to_vec());

    let fused_op = NativeOpKind::FusedConv1dActivation {
        activation,
        out_channels: conv_info.output_channels,
        kernel_size: conv_info.kernel_size,
        stride: conv_info.stride,
        padding: conv_info.padding,
        dilation: conv_info.dilation,
        groups: conv_info.groups,
        has_bias: conv_weight_data.contains_key("bias"),
        input_shape: input_shape.unwrap_or_default(),
        pre_activation: false,
    };

    // Place FusedConv1dActivation at step[i] (conv1d position).
    steps[i] = CompiledStep::NativeOp {
        op: fused_op,
        weight_data: merged_weight_data,
    };

    // Replace step[i+1] with IdentityPassthrough (preserves index alignment).
    steps[i + 1] = CompiledStep::IdentityPassthrough;

    // Store external_node_ids if needed for edge_map resolution.
    let _ = ext_ids;

    true
}

/// Extract the input shape from a conv1d Dispatch step's IR.
fn extract_conv1d_input_shape(step: &CompiledStep) -> Option<Vec<usize>> {
    match step {
        CompiledStep::Dispatch { kernel, .. } => {
            let def = kernel.def();
            let input_node = def.nodes.first()?;
            match &input_node.kind {
                TensorOpKind::Input { shape, .. } => Some(shape.clone()),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Extract the negative slope from a leaky_relu kernel IR.
///
/// The leaky_relu kernel IR contains a `LeakyRelu { negative_slope, .. }` node.
/// Falls back to 0.01 if the slope cannot be extracted.
fn extract_leaky_relu_slope(kernel: &super::super::CompiledKernel) -> f32 {
    let def = kernel.def();
    for node in &def.nodes {
        if let TensorOpKind::LeakyRelu { negative_slope, .. } = &node.kind {
            return *negative_slope;
        }
    }
    // Default negative slope if not found in IR.
    0.01
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_block_builder::TensorBlockBuilder;
    use crate::trace_compile::CompiledKernel;
    use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
    use nn_core::DType;

    /// Helper: build a conv1d Dispatch step with weights.
    fn conv1d_step(in_channels: usize, out_channels: usize, kernel_size: usize) -> CompiledStep {
        let in_t: usize = 64;
        let padding: usize = 1;
        let stride: usize = 1;
        let out_t = (in_t + 2 * padding - kernel_size) / stride + 1;
        let mut b = TensorBlockBuilder::new("conv1d");
        let input = b.add_input("input_0", &[1, in_channels, in_t]);
        let weight = b.add_input("weight", &[out_channels, in_channels, kernel_size]);
        let bias = b.add_input("bias", &[out_channels]);
        let conv = b.add_conv1d(
            input,
            weight,
            Some(bias),
            stride,
            padding,
            &[1, out_channels, out_t],
        );
        let def = b.build(conv).expect("build conv1d");

        let mut wd = HashMap::new();
        wd.insert(
            "weight".to_string(),
            WeightRef::new(
                vec![0.0; out_channels * in_channels * kernel_size],
                vec![out_channels, in_channels, kernel_size],
            )
            .expect("valid weight"),
        );
        wd.insert(
            "bias".to_string(),
            WeightRef::new(vec![0.0; out_channels], vec![out_channels]).expect("valid bias"),
        );
        CompiledStep::Dispatch {
            kernel: CompiledKernel::new(def),
            weight_data: wd,
            external_node_ids: None,
        }
    }

    /// Helper: build a simple activation Dispatch step.
    fn activation_step(name: &str) -> CompiledStep {
        let mut b = TensorBlockBuilder::new(name);
        let input = b.add_input("input_0", &[1, 8, 64]);
        // For simple activations, output = input (same shape elementwise).
        let def = b.build(input).expect("build activation");
        CompiledStep::Dispatch {
            kernel: CompiledKernel::new(def),
            weight_data: HashMap::new(),
            external_node_ids: None,
        }
    }

    /// Helper: build a test TraceNode.
    fn test_node(id: u64, name: &str, inputs: Vec<u64>) -> TraceNode {
        TraceNode::new(
            id,
            name.to_string(),
            TraceOp::Relu,
            inputs,
            vec![1, 8, 64],
            DType::F32,
        )
    }

    /// Build a minimal computation graph with 2 nodes for the pattern.
    fn two_node_graph() -> ComputationGraph {
        let nodes = vec![
            test_node(0, "conv1d", vec![]),
            test_node(1, "relu", vec![0]),
        ];
        ComputationGraph::from_nodes(nodes)
    }

    #[test]
    fn test_fuse_conv1d_relu() {
        let graph = two_node_graph();
        let use_counts = vec![1, 0];
        let mut steps = vec![conv1d_step(4, 8, 3), activation_step("relu")];

        fuse_conv1d_activation(&mut steps, &use_counts, &graph);

        assert!(
            matches!(
                &steps[0],
                CompiledStep::NativeOp {
                    op: NativeOpKind::FusedConv1dActivation {
                        activation: ConvActivation::Relu,
                        out_channels: 8,
                        kernel_size: 3,
                        ..
                    },
                    ..
                }
            ),
            "expected FusedConv1dActivation(Relu), got {:?}",
            steps[0]
        );
        assert!(matches!(steps[1], CompiledStep::IdentityPassthrough));
    }

    #[test]
    fn test_fuse_conv1d_silu() {
        let graph = two_node_graph();
        let use_counts = vec![1, 0];
        let mut steps = vec![conv1d_step(4, 8, 3), activation_step("silu")];

        fuse_conv1d_activation(&mut steps, &use_counts, &graph);

        assert!(matches!(
            &steps[0],
            CompiledStep::NativeOp {
                op: NativeOpKind::FusedConv1dActivation {
                    activation: ConvActivation::Silu,
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn test_no_fuse_fanout() {
        let graph = two_node_graph();
        let use_counts = vec![2, 0]; // conv1d has 2 consumers
        let mut steps = vec![conv1d_step(4, 8, 3), activation_step("relu")];

        fuse_conv1d_activation(&mut steps, &use_counts, &graph);

        // Should NOT fuse.
        assert!(matches!(
            &steps[0],
            CompiledStep::Dispatch { kernel, .. } if kernel.name() == "conv1d"
        ));
    }

    #[test]
    fn test_fuse_conv1d_tanh() {
        let graph = two_node_graph();
        let use_counts = vec![1, 0];
        let mut steps = vec![conv1d_step(4, 8, 3), activation_step("tanh")];

        fuse_conv1d_activation(&mut steps, &use_counts, &graph);

        assert!(matches!(
            &steps[0],
            CompiledStep::NativeOp {
                op: NativeOpKind::FusedConv1dActivation {
                    activation: ConvActivation::Tanh,
                    ..
                },
                ..
            }
        ));
        assert!(matches!(steps[1], CompiledStep::IdentityPassthrough));
    }

    #[test]
    fn test_fuse_conv1d_gelu() {
        let graph = two_node_graph();
        let use_counts = vec![1, 0];
        let mut steps = vec![conv1d_step(4, 8, 3), activation_step("gelu")];

        fuse_conv1d_activation(&mut steps, &use_counts, &graph);

        assert!(matches!(
            &steps[0],
            CompiledStep::NativeOp {
                op: NativeOpKind::FusedConv1dActivation {
                    activation: ConvActivation::Gelu,
                    ..
                },
                ..
            }
        ));
        assert!(matches!(steps[1], CompiledStep::IdentityPassthrough));
    }

    #[test]
    fn test_fuse_conv1d_gelu_erf() {
        let graph = two_node_graph();
        let use_counts = vec![1, 0];
        let mut steps = vec![conv1d_step(4, 8, 3), activation_step("gelu_erf")];

        fuse_conv1d_activation(&mut steps, &use_counts, &graph);

        assert!(matches!(
            &steps[0],
            CompiledStep::NativeOp {
                op: NativeOpKind::FusedConv1dActivation {
                    activation: ConvActivation::GeluErf,
                    ..
                },
                ..
            }
        ));
        assert!(matches!(steps[1], CompiledStep::IdentityPassthrough));
    }

    #[test]
    fn test_no_fuse_wrong_activation() {
        let graph = two_node_graph();
        let use_counts = vec![1, 0];
        // "sigmoid" is not a supported ConvActivation.
        let mut steps = vec![conv1d_step(4, 8, 3), activation_step("sigmoid")];

        fuse_conv1d_activation(&mut steps, &use_counts, &graph);

        // Should NOT fuse.
        assert!(matches!(
            &steps[0],
            CompiledStep::Dispatch { kernel, .. } if kernel.name() == "conv1d"
        ));
    }

    #[test]
    fn test_fuse_preserves_weights() {
        let graph = two_node_graph();
        let use_counts = vec![1, 0];
        let mut steps = vec![conv1d_step(4, 8, 5), activation_step("relu")];

        fuse_conv1d_activation(&mut steps, &use_counts, &graph);

        match &steps[0] {
            CompiledStep::NativeOp { weight_data, .. } => {
                assert!(weight_data.contains_key("weight"));
                assert!(weight_data.contains_key("bias"));
                assert_eq!(weight_data["weight"].shape(), &[8, 4, 5]);
            }
            other => panic!("expected NativeOp, got {other:?}"),
        }
    }

    #[test]
    fn test_fewer_than_two_steps_safe() {
        let nodes = vec![test_node(0, "conv1d", vec![])];
        let graph = ComputationGraph::from_nodes(nodes);
        let use_counts = vec![0];
        let mut steps = vec![conv1d_step(4, 8, 3)];

        fuse_conv1d_activation(&mut steps, &use_counts, &graph);
        assert!(matches!(
            &steps[0],
            CompiledStep::Dispatch { kernel, .. } if kernel.name() == "conv1d"
        ));
    }

    #[test]
    fn test_fuse_multiple_pairs() {
        let nodes = vec![
            test_node(0, "conv1d", vec![]),
            test_node(1, "relu", vec![0]),
            test_node(2, "conv1d", vec![1]),
            test_node(3, "silu", vec![2]),
        ];
        let graph = ComputationGraph::from_nodes(nodes);
        let use_counts = vec![1, 1, 1, 0];
        let mut steps = vec![
            conv1d_step(4, 8, 3),
            activation_step("relu"),
            conv1d_step(8, 16, 3),
            activation_step("silu"),
        ];

        fuse_conv1d_activation(&mut steps, &use_counts, &graph);

        assert!(matches!(
            &steps[0],
            CompiledStep::NativeOp {
                op: NativeOpKind::FusedConv1dActivation {
                    activation: ConvActivation::Relu,
                    ..
                },
                ..
            }
        ));
        assert!(matches!(steps[1], CompiledStep::IdentityPassthrough));
        assert!(matches!(
            &steps[2],
            CompiledStep::NativeOp {
                op: NativeOpKind::FusedConv1dActivation {
                    activation: ConvActivation::Silu,
                    ..
                },
                ..
            }
        ));
        assert!(matches!(steps[3], CompiledStep::IdentityPassthrough));
    }
}
