// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Peephole pass: fuse Conv1d + Snake + InstanceNorm → FusedConv1dSnakeNorm.
//!
//! Detects the 3-step pattern in compiled steps:
//!   Step i:   Dispatch { kernel: "conv1d" }
//!   Step i+1: Dispatch { kernel: "snake_tensor" }
//!   Step i+2: NativeOp { InstanceNorm { eps, input_shape } }
//!
//! Both intermediate outputs must be single-consumer (use_counts == 1).
//!
//! Replaces the 3 steps with:
//! - steps[i]   → NativeOp { FusedConv1dSnakeNorm { ... } }
//! - steps[i+1] → IdentityPassthrough
//! - steps[i+2] → IdentityPassthrough
//!
//! Saves 2 Metal dispatches per Conv1d+Snake+InstanceNorm triple. In the
//! Kokoro generator, blocks where the deeper FusedResBlock or NormActivConv1d
//! patterns don't fire can still benefit from this 3-step fusion.
//!
//! Must run BEFORE FusedSnakeInstanceNorm and FusedConv1dActivation passes
//! because this is a longer pattern that subsumes both 2-step fusions.
//!
//! Part of #4264.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, WeightRef};

use super::super::{CompiledStep, NativeOpKind};
use super::extract_conv1d_params;
use crate::tensor_ir::TensorOpKind;

/// Scan for Conv1d + snake_tensor + InstanceNorm triples and fuse them.
pub(super) fn fuse_conv1d_snake_norm(
    steps: &mut [CompiledStep],
    use_counts: &[usize],
    _graph: &ComputationGraph,
) {
    let len = steps.len();
    if len < 3 {
        return;
    }

    let mut i = 0;
    while i + 2 < len {
        if try_fuse(steps, i, use_counts) {
            // Skip past the fused triple.
            i += 3;
        } else {
            i += 1;
        }
    }
}

/// Try to fuse steps[i..i+3] as Conv1d + Snake + InstanceNorm.
///
/// Returns `true` if the triple was fused (steps mutated in-place).
fn try_fuse(steps: &mut [CompiledStep], i: usize, use_counts: &[usize]) -> bool {
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

    // Fan-out: conv1d output must have exactly 1 consumer.
    if use_counts.get(i).copied().unwrap_or(0) != 1 {
        return false;
    }

    // Only fuse groups=1 (common case in Kokoro generator).
    if conv_info.groups != 1 {
        return false;
    }

    // ---- Step i+1: Dispatch with kernel name "snake_tensor" ----
    let snake_weight_data = match &steps[i + 1] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } if kernel.name() == "snake_tensor" => Some(weight_data.clone()),
        _ => None,
    };
    let snake_weight_data = match snake_weight_data {
        Some(wd) => wd,
        None => return false,
    };

    // Fan-out: snake output must have exactly 1 consumer.
    if use_counts.get(i + 1).copied().unwrap_or(0) != 1 {
        return false;
    }

    // ---- Step i+2: NativeOp { InstanceNorm { eps, input_shape } } ----
    let (eps, norm_input_shape) = match &steps[i + 2] {
        CompiledStep::NativeOp {
            op: NativeOpKind::InstanceNorm { eps, input_shape },
            ..
        } => (*eps, input_shape.clone()),
        _ => return false,
    };

    // InstanceNorm input shape must be rank >= 3 for [B, C, T].
    if norm_input_shape.len() < 3 {
        return false;
    }

    // Extract alpha weight from snake_tensor's weight_data.
    let alpha = match snake_weight_data.get("alpha") {
        Some(a) => a.clone(),
        None => return false,
    };

    // Extract conv1d input shape for the fused op.
    let input_shape = extract_conv1d_input_shape(&steps[i]).unwrap_or_default();

    // Build merged weight_data.
    let mut merged_weight_data: HashMap<String, WeightRef> = HashMap::new();
    if let Some(w) = conv_info.weight {
        merged_weight_data.insert("conv_weight".to_string(), w);
    }
    if let Some(b) = conv_info.bias {
        merged_weight_data.insert("conv_bias".to_string(), b);
    }
    merged_weight_data.insert("alpha".to_string(), alpha);

    let fused_op = NativeOpKind::FusedConv1dSnakeNorm {
        out_channels: conv_info.output_channels,
        kernel_size: conv_info.kernel_size,
        stride: conv_info.stride,
        padding: conv_info.padding,
        dilation: conv_info.dilation,
        groups: conv_info.groups,
        has_bias: conv_weight_data.contains_key("bias"),
        eps,
        input_shape,
    };

    // Place fused op at step[i] (conv1d position).
    steps[i] = CompiledStep::NativeOp {
        op: fused_op,
        weight_data: merged_weight_data,
    };

    // Replace steps[i+1] and steps[i+2] with IdentityPassthrough.
    steps[i + 1] = CompiledStep::IdentityPassthrough;
    steps[i + 2] = CompiledStep::IdentityPassthrough;

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

    /// Helper: build a snake_tensor Dispatch step with alpha weight.
    fn snake_tensor_step(channels: usize) -> CompiledStep {
        let mut b = TensorBlockBuilder::new("snake_tensor");
        let input = b.add_input("input_0", &[1, channels, 64]);
        let def = b.build(input).expect("build");
        let mut wd = HashMap::new();
        wd.insert(
            "alpha".to_string(),
            WeightRef::new(vec![1.0; channels], vec![channels]).expect("valid alpha"),
        );
        CompiledStep::Dispatch {
            kernel: CompiledKernel::new(def),
            weight_data: wd,
            external_node_ids: None,
        }
    }

    /// Helper: build an InstanceNorm NativeOp step.
    fn instance_norm_step(eps: f32, shape: Vec<usize>) -> CompiledStep {
        CompiledStep::NativeOp {
            op: NativeOpKind::InstanceNorm {
                eps,
                input_shape: shape,
            },
            weight_data: HashMap::new(),
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

    /// Build a minimal computation graph with 3 nodes for the pattern.
    fn three_node_graph() -> ComputationGraph {
        let nodes = vec![
            test_node(0, "conv1d", vec![]),
            test_node(1, "snake_tensor", vec![0]),
            test_node(2, "instance_norm", vec![1]),
        ];
        ComputationGraph::from_nodes(nodes)
    }

    #[test]
    fn test_fuse_conv1d_snake_norm_basic() {
        let graph = three_node_graph();
        let use_counts = vec![1, 1, 0];
        let mut steps = vec![
            conv1d_step(4, 8, 3),
            snake_tensor_step(8),
            instance_norm_step(1e-5, vec![1, 8, 64]),
        ];

        fuse_conv1d_snake_norm(&mut steps, &use_counts, &graph);

        // Step 0 should be FusedConv1dSnakeNorm.
        assert!(
            matches!(&steps[0], CompiledStep::NativeOp {
                op: NativeOpKind::FusedConv1dSnakeNorm { out_channels: 8, eps, .. },
                ..
            } if (*eps - 1e-5).abs() < 1e-8),
            "expected FusedConv1dSnakeNorm, got {:?}",
            steps[0]
        );
        // Steps 1 and 2 should be IdentityPassthrough.
        assert!(matches!(steps[1], CompiledStep::IdentityPassthrough));
        assert!(matches!(steps[2], CompiledStep::IdentityPassthrough));
    }

    #[test]
    fn test_fuse_preserves_weights() {
        let graph = three_node_graph();
        let use_counts = vec![1, 1, 0];
        let mut steps = vec![
            conv1d_step(4, 8, 3),
            snake_tensor_step(8),
            instance_norm_step(1e-5, vec![1, 8, 64]),
        ];

        fuse_conv1d_snake_norm(&mut steps, &use_counts, &graph);

        match &steps[0] {
            CompiledStep::NativeOp { weight_data, .. } => {
                assert!(
                    weight_data.contains_key("conv_weight"),
                    "must have conv_weight"
                );
                assert!(weight_data.contains_key("conv_bias"), "must have conv_bias");
                assert!(weight_data.contains_key("alpha"), "must have alpha");
                assert_eq!(weight_data["conv_weight"].shape(), &[8, 4, 3]);
                assert_eq!(weight_data["alpha"].shape(), &[8]);
            }
            other => panic!("expected NativeOp, got {other:?}"),
        }
    }

    #[test]
    fn test_no_fuse_conv_has_multiple_consumers() {
        let graph = three_node_graph();
        let use_counts = vec![2, 1, 0]; // conv1d output consumed by 2 nodes
        let mut steps = vec![
            conv1d_step(4, 8, 3),
            snake_tensor_step(8),
            instance_norm_step(1e-5, vec![1, 8, 64]),
        ];

        fuse_conv1d_snake_norm(&mut steps, &use_counts, &graph);

        // Should NOT fuse.
        assert!(matches!(
            &steps[0],
            CompiledStep::Dispatch { kernel, .. } if kernel.name() == "conv1d"
        ));
    }

    #[test]
    fn test_no_fuse_snake_has_multiple_consumers() {
        let graph = three_node_graph();
        let use_counts = vec![1, 2, 0]; // snake output consumed by 2 nodes
        let mut steps = vec![
            conv1d_step(4, 8, 3),
            snake_tensor_step(8),
            instance_norm_step(1e-5, vec![1, 8, 64]),
        ];

        fuse_conv1d_snake_norm(&mut steps, &use_counts, &graph);

        // Should NOT fuse.
        assert!(matches!(
            &steps[0],
            CompiledStep::Dispatch { kernel, .. } if kernel.name() == "conv1d"
        ));
    }

    #[test]
    fn test_no_fuse_wrong_middle_step() {
        let graph = three_node_graph();
        let use_counts = vec![1, 1, 0];

        let mut b = TensorBlockBuilder::new("relu");
        let input = b.add_input("input_0", &[1, 8, 64]);
        let def = b.build(input).expect("build");
        let relu_step = CompiledStep::Dispatch {
            kernel: CompiledKernel::new(def),
            weight_data: HashMap::new(),
            external_node_ids: None,
        };

        let mut steps = vec![
            conv1d_step(4, 8, 3),
            relu_step,
            instance_norm_step(1e-5, vec![1, 8, 64]),
        ];

        fuse_conv1d_snake_norm(&mut steps, &use_counts, &graph);

        // Should NOT fuse — middle step is not snake_tensor.
        assert!(matches!(
            &steps[0],
            CompiledStep::Dispatch { kernel, .. } if kernel.name() == "conv1d"
        ));
    }

    #[test]
    fn test_no_fuse_low_rank_instance_norm() {
        let graph = three_node_graph();
        let use_counts = vec![1, 1, 0];
        let mut steps = vec![
            conv1d_step(4, 8, 3),
            snake_tensor_step(8),
            instance_norm_step(1e-5, vec![8, 64]), // rank 2 — too low
        ];

        fuse_conv1d_snake_norm(&mut steps, &use_counts, &graph);

        // Should NOT fuse — InstanceNorm input rank < 3.
        assert!(matches!(
            &steps[0],
            CompiledStep::Dispatch { kernel, .. } if kernel.name() == "conv1d"
        ));
    }

    #[test]
    fn test_fewer_than_three_steps_safe() {
        let nodes = vec![
            test_node(0, "conv1d", vec![]),
            test_node(1, "snake_tensor", vec![0]),
        ];
        let graph = ComputationGraph::from_nodes(nodes);
        let use_counts = vec![1, 0];
        let mut steps = vec![conv1d_step(4, 8, 3), snake_tensor_step(8)];

        // Should not panic.
        fuse_conv1d_snake_norm(&mut steps, &use_counts, &graph);
    }
}
