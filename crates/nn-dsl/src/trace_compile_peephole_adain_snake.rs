// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pass 0: fuse InstanceNorm + Mul + Add + Snake → FusedAdainSnake.
//!
//! Detects the 4-step pattern in compiled steps:
//!   Step i:   NativeOp { InstanceNorm { eps, input_shape } }
//!   Step i+1: Dispatch { kernel: "mul" }     (scale by gamma)
//!   Step i+2: Dispatch { kernel: "add" }     (shift by beta)
//!   Step i+3: Dispatch { kernel: "snake_tensor" } (Snake activation)
//!
//! Each intermediate must be single-consumer (use_counts == 1).
//!
//! Replaces all 4 steps with:
//! - steps[i]   → NativeOp { FusedAdainSnake { eps, input_shape, channels } }
//! - steps[i+1] → IdentityPassthrough
//! - steps[i+2] → IdentityPassthrough
//! - steps[i+3] → IdentityPassthrough
//!
//! Saves 3 Metal dispatches per AdaIN+Snake block. In Kokoro Generator,
//! 12 blocks that aren't captured by the deeper FusedResBlock or
//! NormActivConv1d passes = 36 dispatches reduced to 12.
//!
//! Must run BEFORE pass 1 (NormActivConv1d) so the pattern is consumed
//! before NormActivConv1d looks for AdainSnake + Conv1d pairs.
//!
//! Part of #4252.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, WeightRef};

use super::super::{CompiledStep, NativeOpKind};

/// Scan for InstanceNorm + Mul + Add + Snake quadruples and fuse them.
pub(super) fn fuse_adain_snake(
    steps: &mut [CompiledStep],
    use_counts: &[usize],
    graph: &ComputationGraph,
) {
    let len = steps.len();
    if len < 4 {
        return;
    }

    let graph_nodes = graph.nodes();

    let mut i = 0;
    while i + 3 < len {
        if try_fuse(steps, i, use_counts, graph_nodes) {
            // Skip past the fused quadruple.
            i += 4;
        } else {
            i += 1;
        }
    }
}

/// Try to fuse steps[i..i+4] as InstanceNorm + Mul + Add + Snake.
///
/// Returns `true` if the quadruple was fused (steps mutated in-place).
fn try_fuse(
    steps: &mut [CompiledStep],
    i: usize,
    use_counts: &[usize],
    graph_nodes: &[nn_core::dyn_tensor::trace::TraceNode],
) -> bool {
    // ---- Step i: NativeOp { InstanceNorm { eps, input_shape } } ----
    let (eps, input_shape) = match &steps[i] {
        CompiledStep::NativeOp {
            op: NativeOpKind::InstanceNorm { eps, input_shape },
            ..
        } => (*eps, input_shape.clone()),
        _ => return false,
    };

    // Fan-out: InstanceNorm output must have exactly 1 consumer.
    if use_counts.get(i).copied().unwrap_or(0) != 1 {
        return false;
    }

    // ---- Step i+1: Dispatch with kernel name "mul" ----
    let is_mul = matches!(
        &steps[i + 1],
        CompiledStep::Dispatch { kernel, .. } if kernel.name() == "mul"
    );
    if !is_mul {
        return false;
    }
    if use_counts.get(i + 1).copied().unwrap_or(0) != 1 {
        return false;
    }

    // ---- Step i+2: Dispatch with kernel name "add" ----
    let is_add = matches!(
        &steps[i + 2],
        CompiledStep::Dispatch { kernel, .. } if kernel.name() == "add"
    );
    if !is_add {
        return false;
    }
    if use_counts.get(i + 2).copied().unwrap_or(0) != 1 {
        return false;
    }

    // ---- Step i+3: Dispatch with kernel name "snake_tensor" ----
    let snake_weight_data = match &steps[i + 3] {
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
    let mut weight_data: HashMap<String, WeightRef> = HashMap::new();
    weight_data.insert("alpha".to_string(), alpha);

    // Extract graph node IDs for gamma (from mul step) and beta (from add step).
    // In Kokoro's generator, gamma and beta are runtime outputs from style
    // projections -- NOT static weights. The edge_map builder uses these IDs
    // to resolve the correct GPU buffers at execution time.
    let ext_ids = {
        // x input comes from InstanceNorm's graph node input
        let x_id = graph_nodes.get(i).and_then(|n| n.inputs().first().copied());
        // gamma comes from the mul node's second input (input[1])
        let gamma_id = graph_nodes
            .get(i + 1)
            .and_then(|n| n.inputs().get(1).copied());
        // beta comes from the add node's second input (input[1])
        let beta_id = graph_nodes
            .get(i + 2)
            .and_then(|n| n.inputs().get(1).copied());
        match (x_id, gamma_id, beta_id) {
            (Some(x), Some(g), Some(b)) => Some(vec![x, g, b]),
            _ => None,
        }
    };

    let fused_op = NativeOpKind::FusedAdainSnake {
        eps,
        input_shape,
        channels,
        external_node_ids: ext_ids,
    };

    // Place FusedAdainSnake at step[i] (InstanceNorm position).
    steps[i] = CompiledStep::NativeOp {
        op: fused_op,
        weight_data,
    };

    // Replace steps i+1..i+3 with IdentityPassthrough.
    steps[i + 1] = CompiledStep::IdentityPassthrough;
    steps[i + 2] = CompiledStep::IdentityPassthrough;
    steps[i + 3] = CompiledStep::IdentityPassthrough;

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

    /// Helper: build a Dispatch step with a given kernel name and no weights.
    fn dispatch_step(name: &str) -> CompiledStep {
        use crate::tensor_block_builder::TensorBlockBuilder;
        let mut b = TensorBlockBuilder::new(name);
        let input = b.add_input("input_0", &[1, 4, 8]);
        let def = b.build(input).expect("build");
        CompiledStep::Dispatch {
            kernel: crate::trace_compile::CompiledKernel::new(def),
            weight_data: HashMap::new(),
            external_node_ids: None,
        }
    }

    /// Helper: build a Dispatch step mimicking snake_tensor with alpha weight.
    fn snake_tensor_step() -> CompiledStep {
        use crate::tensor_block_builder::TensorBlockBuilder;
        let mut b = TensorBlockBuilder::new("snake_tensor");
        let input = b.add_input("input_0", &[1, 4, 8]);
        let def = b.build(input).expect("build");
        let mut wd = HashMap::new();
        wd.insert(
            "alpha".to_string(),
            WeightRef::new(vec![1.0; 4], vec![4])
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
            TraceOp::Relu, // dummy op — peephole only inspects CompiledSteps
            inputs,
            vec![1, 4, 8],
            DType::F32,
        )
    }

    /// Build a minimal computation graph with 4 nodes for the pattern.
    fn four_node_graph() -> ComputationGraph {
        let nodes = vec![
            test_node(0, "instance_norm", vec![]),
            test_node(1, "mul", vec![0]),
            test_node(2, "add", vec![1]),
            test_node(3, "snake", vec![2]),
        ];
        ComputationGraph::from_nodes(nodes)
    }

    #[test]
    fn test_fuse_adain_snake_basic() {
        let graph = four_node_graph();
        let use_counts = vec![1, 1, 1, 0]; // each intermediate single-consumer
        let mut steps = vec![
            instance_norm_step(1e-5, vec![1, 4, 8]),
            dispatch_step("mul"),
            dispatch_step("add"),
            snake_tensor_step(),
        ];

        fuse_adain_snake(&mut steps, &use_counts, &graph);

        // Step 0 should be FusedAdainSnake.
        assert!(
            matches!(&steps[0], CompiledStep::NativeOp {
                op: NativeOpKind::FusedAdainSnake { eps, channels, .. },
                ..
            } if (*eps - 1e-5).abs() < 1e-8 && *channels == 4),
            "expected FusedAdainSnake, got {:?}",
            steps[0]
        );

        // Steps 1-3 should be IdentityPassthrough.
        assert!(matches!(steps[1], CompiledStep::IdentityPassthrough));
        assert!(matches!(steps[2], CompiledStep::IdentityPassthrough));
        assert!(matches!(steps[3], CompiledStep::IdentityPassthrough));
    }

    #[test]
    fn test_no_fuse_when_instance_norm_is_multi_consumer() {
        let graph = four_node_graph();
        let use_counts = vec![2, 1, 1, 0]; // InstanceNorm has 2 consumers
        let mut steps = vec![
            instance_norm_step(1e-5, vec![1, 4, 8]),
            dispatch_step("mul"),
            dispatch_step("add"),
            snake_tensor_step(),
        ];

        fuse_adain_snake(&mut steps, &use_counts, &graph);

        // Should NOT fuse — InstanceNorm output has fan-out > 1.
        assert!(matches!(
            &steps[0],
            CompiledStep::NativeOp {
                op: NativeOpKind::InstanceNorm { .. },
                ..
            }
        ));
    }

    #[test]
    fn test_no_fuse_wrong_kernel_name() {
        let graph = four_node_graph();
        let use_counts = vec![1, 1, 1, 0];
        let mut steps = vec![
            instance_norm_step(1e-5, vec![1, 4, 8]),
            dispatch_step("mul"),
            dispatch_step("sub"), // wrong — should be "add"
            snake_tensor_step(),
        ];

        fuse_adain_snake(&mut steps, &use_counts, &graph);

        // Should NOT fuse.
        assert!(matches!(
            &steps[0],
            CompiledStep::NativeOp {
                op: NativeOpKind::InstanceNorm { .. },
                ..
            }
        ));
    }

    #[test]
    fn test_no_fuse_rank_2_input() {
        let graph = four_node_graph();
        let use_counts = vec![1, 1, 1, 0];
        let mut steps = vec![
            instance_norm_step(1e-5, vec![4, 8]), // rank 2 — too low
            dispatch_step("mul"),
            dispatch_step("add"),
            snake_tensor_step(),
        ];

        fuse_adain_snake(&mut steps, &use_counts, &graph);

        // Should NOT fuse — rank < 3.
        assert!(matches!(
            &steps[0],
            CompiledStep::NativeOp {
                op: NativeOpKind::InstanceNorm { .. },
                ..
            }
        ));
    }

    #[test]
    fn test_fewer_than_four_steps_is_safe() {
        let nodes = vec![test_node(0, "a", vec![])];
        let graph = ComputationGraph::from_nodes(nodes);
        let use_counts = vec![0];
        let mut steps = vec![instance_norm_step(1e-5, vec![1, 4, 8])];

        // Should not panic.
        fuse_adain_snake(&mut steps, &use_counts, &graph);
    }

    #[test]
    fn test_fuse_preserves_alpha_weight() {
        let graph = four_node_graph();
        let use_counts = vec![1, 1, 1, 0];
        let mut steps = vec![
            instance_norm_step(1e-5, vec![1, 4, 8]),
            dispatch_step("mul"),
            dispatch_step("add"),
            snake_tensor_step(),
        ];

        fuse_adain_snake(&mut steps, &use_counts, &graph);

        // Verify the fused step has the alpha weight.
        match &steps[0] {
            CompiledStep::NativeOp { weight_data, .. } => {
                assert!(
                    weight_data.contains_key("alpha"),
                    "FusedAdainSnake must carry alpha weight"
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
}
