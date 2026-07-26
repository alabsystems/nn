// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Peephole pass: fuse Snake + InstanceNorm → FusedSnakeInstanceNorm.
//!
//! Detects the 2-step pattern in compiled steps:
//!   Step i:   Dispatch { kernel: "snake_tensor" }  (Snake activation)
//!   Step i+1: NativeOp { InstanceNorm { eps, input_shape } }
//!
//! The Snake output must be single-consumer (use_counts == 1).
//!
//! Replaces both steps with:
//! - steps[i]   → NativeOp { FusedSnakeInstanceNorm { eps, input_shape, channels } }
//! - steps[i+1] → IdentityPassthrough
//!
//! Saves 1 Metal dispatch per Snake+InstanceNorm pair. In the Kokoro
//! generator, blocks where the deeper FusedResBlock or NormActivConv1d
//! patterns don't fire can still benefit from this simpler fusion.
//!
//! Must run BEFORE pass 0 (FusedAdainSnake) to avoid conflicts with
//! patterns that start with InstanceNorm.
//!
//! Part of #4264.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::ComputationGraph;

use super::super::{CompiledStep, NativeOpKind};

/// Scan for Snake + InstanceNorm pairs and fuse them.
pub(super) fn fuse_snake_instance_norm(
    steps: &mut [CompiledStep],
    use_counts: &[usize],
    _graph: &ComputationGraph,
) {
    let len = steps.len();
    if len < 2 {
        return;
    }

    let mut i = 0;
    while i + 1 < len {
        if try_fuse(steps, i, use_counts) {
            // Skip past the fused pair.
            i += 2;
        } else {
            i += 1;
        }
    }
}

/// Try to fuse steps[i..i+2] as Snake + InstanceNorm.
///
/// Returns `true` if the pair was fused (steps mutated in-place).
fn try_fuse(steps: &mut [CompiledStep], i: usize, use_counts: &[usize]) -> bool {
    // ---- Step i: Dispatch with kernel name "snake_tensor" ----
    let snake_weight_data = match &steps[i] {
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

    // Fan-out: Snake output must have exactly 1 consumer.
    if use_counts.get(i).copied().unwrap_or(0) != 1 {
        return false;
    }

    // ---- Step i+1: NativeOp { InstanceNorm { eps, input_shape } } ----
    let (eps, input_shape) = match &steps[i + 1] {
        CompiledStep::NativeOp {
            op: NativeOpKind::InstanceNorm { eps, input_shape },
            ..
        } => (*eps, input_shape.clone()),
        _ => return false,
    };

    // Input shape must be rank >= 3 for the fused kernel ([B, C, T]).
    if input_shape.len() < 3 {
        return false;
    }
    let channels = input_shape[1];

    // Extract alpha weight from snake_tensor's weight_data.
    let alpha = match snake_weight_data.get("alpha") {
        Some(a) => a.clone(),
        None => return false,
    };

    // Build the fused NativeOp.
    let mut weight_data: HashMap<String, nn_core::dyn_tensor::trace::WeightRef> = HashMap::new();
    weight_data.insert("alpha".to_string(), alpha);

    let fused_op = NativeOpKind::FusedSnakeInstanceNorm {
        eps,
        input_shape,
        channels,
    };

    // Place FusedSnakeInstanceNorm at step[i] (Snake position).
    steps[i] = CompiledStep::NativeOp {
        op: fused_op,
        weight_data,
    };

    // Replace step[i+1] (InstanceNorm) with IdentityPassthrough.
    steps[i + 1] = CompiledStep::IdentityPassthrough;

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
    use nn_core::DType;

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

    /// Helper: build a Dispatch step mimicking snake_tensor with alpha weight.
    fn snake_tensor_step(channels: usize) -> CompiledStep {
        use crate::tensor_block_builder::TensorBlockBuilder;
        let mut b = TensorBlockBuilder::new("snake_tensor");
        let input = b.add_input("input_0", &[1, channels, 8]);
        let def = b.build(input).expect("build");
        let mut wd = HashMap::new();
        wd.insert(
            "alpha".to_string(),
            nn_core::dyn_tensor::trace::WeightRef::new(vec![1.0; channels], vec![channels])
                .expect("valid alpha"),
        );
        CompiledStep::Dispatch {
            kernel: crate::trace_compile::CompiledKernel::new(def),
            weight_data: wd,
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
            vec![1, 4, 8],
            DType::F32,
        )
    }

    /// Build a minimal computation graph with 2 nodes for the pattern.
    fn two_node_graph() -> ComputationGraph {
        let nodes = vec![
            test_node(0, "snake_tensor", vec![]),
            test_node(1, "instance_norm", vec![0]),
        ];
        ComputationGraph::from_nodes(nodes)
    }

    #[test]
    fn test_fuse_snake_instance_norm_basic() {
        let graph = two_node_graph();
        let use_counts = vec![1, 0];
        let mut steps = vec![
            snake_tensor_step(4),
            instance_norm_step(1e-5, vec![1, 4, 8]),
        ];

        fuse_snake_instance_norm(&mut steps, &use_counts, &graph);

        // Step 0 should be FusedSnakeInstanceNorm.
        assert!(
            matches!(&steps[0], CompiledStep::NativeOp {
                op: NativeOpKind::FusedSnakeInstanceNorm { eps, channels, .. },
                ..
            } if (*eps - 1e-5).abs() < 1e-8 && *channels == 4),
            "expected FusedSnakeInstanceNorm, got {:?}",
            steps[0]
        );

        // Step 1 should be IdentityPassthrough.
        assert!(matches!(steps[1], CompiledStep::IdentityPassthrough));
    }

    #[test]
    fn test_no_fuse_when_snake_is_multi_consumer() {
        let graph = two_node_graph();
        let use_counts = vec![2, 0]; // Snake has 2 consumers
        let mut steps = vec![
            snake_tensor_step(4),
            instance_norm_step(1e-5, vec![1, 4, 8]),
        ];

        fuse_snake_instance_norm(&mut steps, &use_counts, &graph);

        // Should NOT fuse.
        assert!(matches!(
            &steps[0],
            CompiledStep::Dispatch { kernel, .. } if kernel.name() == "snake_tensor"
        ));
    }

    #[test]
    fn test_no_fuse_wrong_second_step() {
        let graph = two_node_graph();
        let use_counts = vec![1, 0];

        use crate::tensor_block_builder::TensorBlockBuilder;
        let mut b = TensorBlockBuilder::new("add");
        let input = b.add_input("input_0", &[1, 4, 8]);
        let def = b.build(input).expect("build");
        let add_step = CompiledStep::Dispatch {
            kernel: crate::trace_compile::CompiledKernel::new(def),
            weight_data: HashMap::new(),
            external_node_ids: None,
        };

        let mut steps = vec![snake_tensor_step(4), add_step];

        fuse_snake_instance_norm(&mut steps, &use_counts, &graph);

        // Should NOT fuse — second step is not InstanceNorm.
        assert!(matches!(
            &steps[0],
            CompiledStep::Dispatch { kernel, .. } if kernel.name() == "snake_tensor"
        ));
    }

    #[test]
    fn test_no_fuse_rank_2_input() {
        let graph = two_node_graph();
        let use_counts = vec![1, 0];
        let mut steps = vec![
            snake_tensor_step(4),
            instance_norm_step(1e-5, vec![4, 8]), // rank 2 — too low
        ];

        fuse_snake_instance_norm(&mut steps, &use_counts, &graph);

        // Should NOT fuse — rank < 3.
        assert!(matches!(
            &steps[0],
            CompiledStep::Dispatch { kernel, .. } if kernel.name() == "snake_tensor"
        ));
    }

    #[test]
    fn test_fuse_preserves_alpha_weight() {
        let graph = two_node_graph();
        let use_counts = vec![1, 0];
        let mut steps = vec![
            snake_tensor_step(4),
            instance_norm_step(1e-5, vec![1, 4, 8]),
        ];

        fuse_snake_instance_norm(&mut steps, &use_counts, &graph);

        // Verify the fused step has the alpha weight.
        match &steps[0] {
            CompiledStep::NativeOp { weight_data, .. } => {
                assert!(
                    weight_data.contains_key("alpha"),
                    "FusedSnakeInstanceNorm must carry alpha weight"
                );
                assert_eq!(
                    weight_data["alpha"].shape(),
                    &[4],
                    "alpha shape must be [channels]"
                );
            }
            other => panic!("expected NativeOp, got {other:?}"),
        }
    }

    #[test]
    fn test_fewer_than_two_steps_is_safe() {
        let nodes = vec![test_node(0, "a", vec![])];
        let graph = ComputationGraph::from_nodes(nodes);
        let use_counts = vec![0];
        let mut steps = vec![snake_tensor_step(4)];

        // Should not panic.
        fuse_snake_instance_norm(&mut steps, &use_counts, &graph);
    }
}
