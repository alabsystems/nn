// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Peephole pass: fuse Activation + Conv1d into FusedConv1dActivation (pre-conv activation).
//!
//! Detects consecutive step pairs where an activation precedes a Conv1d:
//!   Step i:   Dispatch { kernel: "leaky_relu" | "relu" | "silu" | "gelu" | "tanh" }
//!   Step i+1: Dispatch { kernel: "conv1d" }
//!
//! The activation output must be single-consumer (use_counts == 1).
//!
//! This is the REVERSE pattern of FusedConv1dActivation (pass "fuse_conv1d_activation").
//! Both patterns produce the same NativeOp: `FusedConv1dActivation`.
//! The executor applies activation BEFORE conv1d when `pre_activation: true`.
//!
//! In Kokoro, this catches:
//!   - Generator output stage: `leaky_relu(0.01) -> conv_post` (1 site)
//!   - FullDecoder Stage1ResBlk residual path: after norm1/norm2 the LeakyReLU
//!     output feeds directly into a conv1d when NormActivConv1d doesn't absorb it.
//!
//! Must run AFTER NormActivConv1d (pass 1), FusedResBlock (pass 2), and
//! FusedConv1dActivation (which handles Conv1d -> Activation). This pass catches
//! remaining Activation -> Conv1d pairs that previous passes missed.
//!
//! Part of #4264.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, WeightRef};

use super::super::{CompiledStep, ConvActivation, NativeOpKind};
use super::extract_conv1d_params;
use crate::tensor_ir::TensorOpKind;

/// Scan for consecutive Dispatch(activation) + Dispatch("conv1d") pairs
/// and fuse them into a single FusedConv1dActivation NativeOp with pre_activation.
pub(super) fn fuse_activation_conv1d(
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
        if try_fuse_activation_conv1d(steps, i, use_counts, graph) {
            // Skip past the fused pair.
            i += 2;
        } else {
            i += 1;
        }
    }
}

/// Try to fuse steps[i] (activation) with steps[i+1] (conv1d).
///
/// Returns `true` if the pair was fused (steps mutated in-place).
fn try_fuse_activation_conv1d(
    steps: &mut [CompiledStep],
    i: usize,
    use_counts: &[usize],
    _graph: &ComputationGraph,
) -> bool {
    // ---- Step i: Dispatch with activation kernel ----
    let activation = match &steps[i] {
        CompiledStep::Dispatch { kernel, .. } => match kernel.name() {
            "relu" => ConvActivation::Relu,
            "leaky_relu" => {
                let slope = extract_leaky_relu_slope(kernel);
                ConvActivation::LeakyRelu { slope }
            }
            "silu" => ConvActivation::Silu,
            "gelu" => ConvActivation::Gelu,
            "gelu_erf" => ConvActivation::GeluErf,
            "tanh" => ConvActivation::Tanh,
            _ => return false,
        },
        _ => return false,
    };

    // Fan-out check: activation output must have exactly 1 consumer (the conv1d).
    if use_counts.get(i).copied().unwrap_or(0) != 1 {
        return false;
    }

    // ---- Step i+1: Dispatch with kernel name "conv1d" ----
    let (conv_info, conv_weight_data) = match &steps[i + 1] {
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

    // Build merged weight_data from conv weights.
    let mut merged_weight_data: HashMap<String, WeightRef> = HashMap::new();
    if let Some(w) = conv_info.weight {
        merged_weight_data.insert("weight".to_string(), w);
    }
    if let Some(b) = conv_info.bias {
        merged_weight_data.insert("bias".to_string(), b);
    }

    // Compute input_shape from the activation step's first Input node.
    let input_shape = extract_activation_input_shape(&steps[i]);

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
        pre_activation: true,
    };

    // Place FusedConv1dActivation at step[i] (activation position).
    steps[i] = CompiledStep::NativeOp {
        op: fused_op,
        weight_data: merged_weight_data,
    };

    // Replace step[i+1] with IdentityPassthrough.
    steps[i + 1] = CompiledStep::IdentityPassthrough;

    true
}

/// Extract the input shape from an activation Dispatch step's IR.
fn extract_activation_input_shape(step: &CompiledStep) -> Option<Vec<usize>> {
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
/// Falls back to 0.01 if the slope cannot be extracted.
fn extract_leaky_relu_slope(kernel: &super::super::CompiledKernel) -> f32 {
    let def = kernel.def();
    for node in &def.nodes {
        if let TensorOpKind::LeakyRelu { negative_slope, .. } = &node.kind {
            return *negative_slope;
        }
    }
    0.01
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_block_builder::TensorBlockBuilder;
    use crate::trace_compile::CompiledKernel;
    use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
    use nn_core::DType;

    /// Helper: build a conv1d Dispatch step.
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
        let def = b.build(input).expect("build activation");
        CompiledStep::Dispatch {
            kernel: CompiledKernel::new(def),
            weight_data: HashMap::new(),
            external_node_ids: None,
        }
    }

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

    fn two_node_graph() -> ComputationGraph {
        let nodes = vec![
            test_node(0, "leaky_relu", vec![]),
            test_node(1, "conv1d", vec![0]),
        ];
        ComputationGraph::from_nodes(nodes)
    }

    #[test]
    fn test_fuse_leaky_relu_conv1d() {
        let graph = two_node_graph();
        let use_counts = vec![1, 0];
        let mut steps = vec![activation_step("leaky_relu"), conv1d_step(8, 16, 3)];

        fuse_activation_conv1d(&mut steps, &use_counts, &graph);

        assert!(
            matches!(
                &steps[0],
                CompiledStep::NativeOp {
                    op: NativeOpKind::FusedConv1dActivation {
                        activation: ConvActivation::LeakyRelu { .. },
                        out_channels: 16,
                        kernel_size: 3,
                        ..
                    },
                    ..
                }
            ),
            "expected FusedConv1dActivation(LeakyRelu), got {:?}",
            steps[0]
        );
        assert!(matches!(steps[1], CompiledStep::IdentityPassthrough));
    }

    #[test]
    fn test_fuse_relu_conv1d() {
        let graph = two_node_graph();
        let use_counts = vec![1, 0];
        let mut steps = vec![activation_step("relu"), conv1d_step(8, 16, 3)];

        fuse_activation_conv1d(&mut steps, &use_counts, &graph);

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
    }

    #[test]
    fn test_no_fuse_fanout() {
        let graph = two_node_graph();
        let use_counts = vec![2, 0]; // activation has 2 consumers
        let mut steps = vec![activation_step("leaky_relu"), conv1d_step(8, 16, 3)];

        fuse_activation_conv1d(&mut steps, &use_counts, &graph);

        // Should NOT fuse.
        assert!(matches!(
            &steps[0],
            CompiledStep::Dispatch { kernel, .. } if kernel.name() == "leaky_relu"
        ));
    }

    #[test]
    fn test_no_fuse_wrong_activation() {
        let graph = two_node_graph();
        let use_counts = vec![1, 0];
        let mut steps = vec![activation_step("sigmoid"), conv1d_step(8, 16, 3)];

        fuse_activation_conv1d(&mut steps, &use_counts, &graph);

        // sigmoid is not a supported ConvActivation.
        assert!(matches!(
            &steps[0],
            CompiledStep::Dispatch { kernel, .. } if kernel.name() == "sigmoid"
        ));
    }

    #[test]
    fn test_fuse_preserves_conv_weights() {
        let graph = two_node_graph();
        let use_counts = vec![1, 0];
        let mut steps = vec![activation_step("leaky_relu"), conv1d_step(8, 16, 5)];

        fuse_activation_conv1d(&mut steps, &use_counts, &graph);

        match &steps[0] {
            CompiledStep::NativeOp { weight_data, .. } => {
                assert!(weight_data.contains_key("weight"));
                assert!(weight_data.contains_key("bias"));
                assert_eq!(weight_data["weight"].shape(), &[16, 8, 5]);
            }
            other => panic!("expected NativeOp, got {other:?}"),
        }
    }
}
