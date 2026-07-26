// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Peephole pass: fuse ConvTranspose1d + Activation into FusedConvTranspose1dActivation.
//!
//! Detects consecutive step pairs:
//!   Step i:   Dispatch { kernel: "conv_transpose1d" }
//!   Step i+1: Dispatch { kernel: "snake"/"snake_tensor" | "relu" | "leaky_relu" | "silu" | "gelu" | "gelu_erf" | "tanh" }
//!
//! The conv_transpose1d output must be single-consumer (use_counts == 1).
//!
//! Replaces the pair with:
//! - steps[i]   -> NativeOp { FusedConvTranspose1dActivation { ... } }
//! - steps[i+1] -> IdentityPassthrough
//!
//! Saves 1 dispatch per pair. In Kokoro, 4 such pairs exist in the
//! Generator upsample stages (ConvTranspose1d stride=2 + LeakyReLU/Snake),
//! and additional pairs in F0EnergyPredictor upsampling blocks.
//!
//! Must run AFTER NormActivConv1d (pass 1) and FusedResBlock (pass 2)
//! which consume AdainSnake/LeakyRelu + Conv1d sequences. This pass
//! catches remaining standalone ConvTranspose1d → Activation patterns.
//!
//! Part of #4264.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, WeightRef};

use super::super::{CompiledStep, ConvActivation, NativeOpKind};
use crate::tensor_ir::TensorOpKind;

/// Scan for consecutive Dispatch("conv_transpose1d") + Dispatch(activation) pairs
/// and fuse them into a single FusedConvTranspose1dActivation NativeOp.
pub(super) fn fuse_conv_transpose1d_activation(
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
        if try_fuse_conv_transpose1d_activation(steps, i, use_counts, graph) {
            // Skip past the fused pair.
            i += 2;
        } else {
            i += 1;
        }
    }
}

/// Try to fuse steps[i] (conv_transpose1d) with steps[i+1] (activation).
///
/// Returns `true` if the pair was fused (steps mutated in-place).
fn try_fuse_conv_transpose1d_activation(
    steps: &mut [CompiledStep],
    i: usize,
    use_counts: &[usize],
    graph: &ComputationGraph,
) -> bool {
    // ---- Step i: Dispatch with kernel name "conv_transpose1d" ----
    let (ct_info, ct_weight_data) = match &steps[i] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } if kernel.name() == "conv_transpose1d" => {
            match extract_conv_transpose1d_params(kernel, weight_data) {
                Some(info) => (info, weight_data.clone()),
                None => return false,
            }
        }
        _ => return false,
    };

    // Fan-out check: the conv_transpose1d output must have exactly 1 consumer.
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

    // Build merged weight_data from conv_transpose1d weights.
    let mut merged_weight_data: HashMap<String, WeightRef> = HashMap::new();
    if let Some(w) = ct_info.weight {
        merged_weight_data.insert("weight".to_string(), w);
    }
    if let Some(b) = ct_info.bias {
        merged_weight_data.insert("bias".to_string(), b);
    }
    // Add alpha weight for Snake activation.
    if let Some(alpha) = alpha_weight {
        merged_weight_data.insert("alpha".to_string(), alpha);
    }

    // Compute input_shape from the conv_transpose1d kernel's first Input node.
    let input_shape = extract_conv_transpose1d_input_shape(&steps[i]);

    // Capture the graph node IDs from the conv_transpose1d step so the edge_map
    // builder can resolve edges generically.
    let _ext_ids = graph.nodes().get(i).map(|node| node.inputs().to_vec());

    let fused_op = NativeOpKind::FusedConvTranspose1dActivation {
        activation,
        out_channels: ct_info.output_channels,
        kernel_size: ct_info.kernel_size,
        stride: ct_info.stride,
        padding: ct_info.padding,
        dilation: ct_info.dilation,
        groups: ct_info.groups,
        output_padding: ct_info.output_padding,
        has_bias: ct_weight_data.contains_key("bias"),
        input_shape: input_shape.unwrap_or_default(),
    };

    // Place FusedConvTranspose1dActivation at step[i] (conv_transpose1d position).
    steps[i] = CompiledStep::NativeOp {
        op: fused_op,
        weight_data: merged_weight_data,
    };

    // Replace step[i+1] with IdentityPassthrough (preserves index alignment).
    steps[i + 1] = CompiledStep::IdentityPassthrough;

    true
}

/// Extracted ConvTranspose1d parameters.
struct ConvTranspose1dInfo {
    /// Zero-padding on each side.
    padding: usize,
    /// Dilation factor.
    dilation: usize,
    /// Convolution stride.
    stride: usize,
    /// Number of channel groups.
    groups: usize,
    /// Extra output padding (must be < stride).
    output_padding: usize,
    /// Number of output channels (from weight shape[1]).
    output_channels: usize,
    /// Convolution kernel size.
    kernel_size: usize,
    /// Weight reference (if available from weight_data).
    weight: Option<WeightRef>,
    /// Bias reference (if available from weight_data).
    bias: Option<WeightRef>,
}

/// Extract conv_transpose1d parameters from a compiled kernel + weight data.
fn extract_conv_transpose1d_params(
    kernel: &super::super::CompiledKernel,
    weight_data: &HashMap<String, WeightRef>,
) -> Option<ConvTranspose1dInfo> {
    let def = kernel.def();

    // Find the ConvTranspose1d IR node to extract parameters.
    let ct_node = def
        .nodes
        .iter()
        .find(|n| matches!(n.kind, TensorOpKind::ConvTranspose1d { .. }))?;

    let (padding, dilation, stride, groups, output_padding) = match &ct_node.kind {
        TensorOpKind::ConvTranspose1d {
            padding,
            dilation,
            stride,
            groups,
            output_padding,
            ..
        } => (*padding, *dilation, *stride, *groups, *output_padding),
        _ => return None,
    };

    // Extract output channels and kernel size from the conv weight shape.
    // ConvTranspose1d weight shape is [C_in, C_out/groups, K].
    let weight_ref = weight_data.get("weight")?;
    let weight_shape = weight_ref.shape();
    if weight_shape.len() != 3 {
        return None;
    }
    let output_channels = weight_shape[1] * groups;
    let kernel_size = weight_shape[2];

    Some(ConvTranspose1dInfo {
        padding,
        dilation,
        stride,
        groups,
        output_padding,
        output_channels,
        kernel_size,
        weight: weight_data.get("weight").cloned(),
        bias: weight_data.get("bias").cloned(),
    })
}

/// Extract the input shape from a conv_transpose1d Dispatch step's IR.
fn extract_conv_transpose1d_input_shape(step: &CompiledStep) -> Option<Vec<usize>> {
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

    /// Helper: build a conv_transpose1d Dispatch step with weights.
    fn conv_transpose1d_step(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
    ) -> CompiledStep {
        let in_t: usize = 16;
        let padding: usize = 1;
        let output_padding: usize = if stride > 1 { stride - 1 } else { 0 };
        // out_t = (in_t - 1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
        let dilation: usize = 1;
        let out_t =
            (in_t - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
        let mut b = TensorBlockBuilder::new("conv_transpose1d");
        let input = b.add_input("input_0", &[1, in_channels, in_t]);
        let weight = b.add_input("weight", &[in_channels, out_channels, kernel_size]);
        let bias = b.add_input("bias", &[out_channels]);
        let conv = b.add_conv_transpose_1d(
            input,
            weight,
            Some(bias),
            stride,
            padding,
            dilation,
            1, // groups
            output_padding,
            &[1, out_channels, out_t],
        );
        let def = b.build(conv).expect("build conv_transpose1d");

        let mut wd = HashMap::new();
        wd.insert(
            "weight".to_string(),
            WeightRef::new(
                vec![0.0; in_channels * out_channels * kernel_size],
                vec![in_channels, out_channels, kernel_size],
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
        let input = b.add_input("input_0", &[1, 8, 32]);
        let def = b.build(input).expect("build activation");
        CompiledStep::Dispatch {
            kernel: CompiledKernel::new(def),
            weight_data: HashMap::new(),
            external_node_ids: None,
        }
    }

    fn leaky_relu_step(slope: f32) -> CompiledStep {
        let mut b = TensorBlockBuilder::new("leaky_relu");
        let input = b.add_input("input_0", &[1, 8, 32]);
        let output = b.add_leaky_relu(input, slope, &[1, 8, 32]);
        let def = b.build(output).expect("build leaky_relu");
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
            vec![1, 8, 32],
            DType::F32,
        )
    }

    /// Build a minimal computation graph with 2 nodes for the pattern.
    fn two_node_graph() -> ComputationGraph {
        let nodes = vec![
            test_node(0, "conv_transpose1d", vec![]),
            test_node(1, "leaky_relu", vec![0]),
        ];
        ComputationGraph::from_nodes(nodes)
    }

    #[test]
    fn test_fuse_conv_transpose1d_leaky_relu() {
        let graph = two_node_graph();
        let use_counts = vec![1, 0];
        let mut steps = vec![conv_transpose1d_step(4, 8, 3, 2), leaky_relu_step(0.2)];

        fuse_conv_transpose1d_activation(&mut steps, &use_counts, &graph);

        assert!(
            matches!(
                &steps[0],
                CompiledStep::NativeOp {
                    op: NativeOpKind::FusedConvTranspose1dActivation {
                        activation: ConvActivation::LeakyRelu { slope },
                        out_channels: 8,
                        kernel_size: 3,
                        stride: 2,
                        ..
                    },
                    ..
                } if (*slope - 0.2).abs() < 1e-6
            ),
            "expected FusedConvTranspose1dActivation(LeakyRelu), got {:?}",
            steps[0]
        );
        assert!(matches!(steps[1], CompiledStep::IdentityPassthrough));
    }

    #[test]
    fn test_fuse_conv_transpose1d_relu() {
        let graph = two_node_graph();
        let use_counts = vec![1, 0];
        let mut steps = vec![conv_transpose1d_step(4, 8, 3, 2), activation_step("relu")];

        fuse_conv_transpose1d_activation(&mut steps, &use_counts, &graph);

        assert!(matches!(
            &steps[0],
            CompiledStep::NativeOp {
                op: NativeOpKind::FusedConvTranspose1dActivation {
                    activation: ConvActivation::Relu,
                    ..
                },
                ..
            }
        ));
        assert!(matches!(steps[1], CompiledStep::IdentityPassthrough));
    }

    #[test]
    fn test_no_fuse_fanout() {
        let graph = two_node_graph();
        let use_counts = vec![2, 0]; // conv_transpose1d has 2 consumers
        let mut steps = vec![conv_transpose1d_step(4, 8, 3, 2), leaky_relu_step(0.2)];

        fuse_conv_transpose1d_activation(&mut steps, &use_counts, &graph);

        // Should NOT fuse.
        assert!(matches!(
            &steps[0],
            CompiledStep::Dispatch { kernel, .. } if kernel.name() == "conv_transpose1d"
        ));
    }

    #[test]
    fn test_no_fuse_wrong_kernel() {
        let graph = two_node_graph();
        let use_counts = vec![1, 0];
        // First step is conv1d, not conv_transpose1d.
        let mut b = TensorBlockBuilder::new("conv1d");
        let input = b.add_input("input_0", &[1, 4, 16]);
        let def = b.build(input).expect("build");
        let wrong_step = CompiledStep::Dispatch {
            kernel: CompiledKernel::new(def),
            weight_data: HashMap::new(),
            external_node_ids: None,
        };
        let mut steps = vec![wrong_step, leaky_relu_step(0.2)];

        fuse_conv_transpose1d_activation(&mut steps, &use_counts, &graph);

        assert!(matches!(
            &steps[0],
            CompiledStep::Dispatch { kernel, .. } if kernel.name() == "conv1d"
        ));
    }

    #[test]
    fn test_fuse_preserves_weights() {
        let graph = two_node_graph();
        let use_counts = vec![1, 0];
        let mut steps = vec![conv_transpose1d_step(4, 8, 5, 2), leaky_relu_step(0.2)];

        fuse_conv_transpose1d_activation(&mut steps, &use_counts, &graph);

        match &steps[0] {
            CompiledStep::NativeOp { weight_data, .. } => {
                assert!(weight_data.contains_key("weight"));
                assert!(weight_data.contains_key("bias"));
                assert_eq!(weight_data["weight"].shape(), &[4, 8, 5]);
            }
            other => panic!("expected NativeOp, got {other:?}"),
        }
    }

    #[test]
    fn test_fewer_than_two_steps_safe() {
        let nodes = vec![test_node(0, "conv_transpose1d", vec![])];
        let graph = ComputationGraph::from_nodes(nodes);
        let use_counts = vec![0];
        let mut steps = vec![conv_transpose1d_step(4, 8, 3, 2)];

        fuse_conv_transpose1d_activation(&mut steps, &use_counts, &graph);
        assert!(matches!(
            &steps[0],
            CompiledStep::Dispatch { kernel, .. } if kernel.name() == "conv_transpose1d"
        ));
    }
}
