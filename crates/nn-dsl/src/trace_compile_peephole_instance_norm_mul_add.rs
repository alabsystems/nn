// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Peephole pass: fuse InstanceNorm + Mul + Add → FusedInstanceNormMulAdd.
//!
//! Detects the 3-step pattern in compiled steps:
//!   Step i:   NativeOp { InstanceNorm { eps, input_shape } }
//!   Step i+1: Dispatch { kernel: "mul" }     (scale by gamma)
//!   Step i+2: Dispatch { kernel: "add" }     (shift by beta)
//!
//! Each intermediate must be single-consumer (use_counts == 1).
//!
//! Replaces all 3 steps with:
//! - steps[i]   → NativeOp { FusedInstanceNormMulAdd { eps, input_shape, channels } }
//! - steps[i+1] → IdentityPassthrough
//! - steps[i+2] → IdentityPassthrough
//!
//! Saves 2 Metal dispatches per AdaIN block (InstanceNorm→Mul→Add fused to 1).
//! In Kokoro, 24 AdaIN blocks = 72 dispatches reduced to 24.
//!
//! Must run BEFORE pass 0 (FusedAdainSnake) because FusedAdainSnake matches
//! the superset pattern InstanceNorm+Mul+Add+Snake. This pass catches the
//! 3-step pattern that lacks a following Snake.
//!
//! Part of #4252.

use nn_core::dyn_tensor::trace::ComputationGraph;

use super::super::{CompiledStep, NativeOpKind};

/// Scan for InstanceNorm + Mul + Add triples and fuse them.
pub(super) fn fuse_instance_norm_mul_add(
    steps: &mut [CompiledStep],
    use_counts: &[usize],
    graph: &ComputationGraph,
) {
    let len = steps.len();
    if len < 3 {
        return;
    }

    let graph_nodes = graph.nodes();

    let mut i = 0;
    while i + 2 < len {
        // Skip if step i+3 is a snake_tensor — that's the FusedAdainSnake
        // pattern and should be handled by pass 0 instead.
        let has_snake_follower = i + 3 < len
            && matches!(
                &steps[i + 3],
                CompiledStep::Dispatch { kernel, .. } if kernel.name() == "snake_tensor"
            );
        if has_snake_follower {
            i += 1;
            continue;
        }

        if try_fuse(steps, i, use_counts, graph_nodes) {
            // Skip past the fused triple.
            i += 3;
        } else {
            i += 1;
        }
    }
}

/// Try to fuse steps[i..i+3] as InstanceNorm + Mul + Add.
///
/// Returns `true` if the triple was fused (steps mutated in-place).
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

    // Input shape must be rank >= 3 for the fused kernel ([B, C, T]).
    if input_shape.len() < 3 {
        return false;
    }
    let channels = input_shape[1];

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

    let fused_op = NativeOpKind::FusedInstanceNormMulAdd {
        eps,
        input_shape,
        channels,
        external_node_ids: ext_ids,
    };

    // Place FusedInstanceNormMulAdd at step[i] (InstanceNorm position).
    // No static weights needed — gamma and beta are graph inputs.
    steps[i] = CompiledStep::NativeOp {
        op: fused_op,
        weight_data: std::collections::HashMap::new(),
    };

    // Replace steps i+1..i+2 with IdentityPassthrough.
    steps[i + 1] = CompiledStep::IdentityPassthrough;
    steps[i + 2] = CompiledStep::IdentityPassthrough;

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
    use nn_core::DType;
    use std::collections::HashMap;

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

    /// Build a minimal computation graph with 3 nodes for the pattern.
    fn three_node_graph() -> ComputationGraph {
        let nodes = vec![
            test_node(0, "instance_norm", vec![]),
            test_node(1, "mul", vec![0]),
            test_node(2, "add", vec![1]),
        ];
        ComputationGraph::from_nodes(nodes)
    }

    #[test]
    fn test_fuse_instance_norm_mul_add_basic() {
        let graph = three_node_graph();
        let use_counts = vec![1, 1, 0]; // each intermediate single-consumer
        let mut steps = vec![
            instance_norm_step(1e-5, vec![1, 4, 8]),
            dispatch_step("mul"),
            dispatch_step("add"),
        ];

        fuse_instance_norm_mul_add(&mut steps, &use_counts, &graph);

        // Step 0 should be FusedInstanceNormMulAdd.
        assert!(
            matches!(&steps[0], CompiledStep::NativeOp {
                op: NativeOpKind::FusedInstanceNormMulAdd { eps, channels, .. },
                ..
            } if (*eps - 1e-5).abs() < 1e-8 && *channels == 4),
            "expected FusedInstanceNormMulAdd, got {:?}",
            steps[0]
        );

        // Steps 1-2 should be IdentityPassthrough.
        assert!(matches!(steps[1], CompiledStep::IdentityPassthrough));
        assert!(matches!(steps[2], CompiledStep::IdentityPassthrough));
    }

    #[test]
    fn test_no_fuse_when_instance_norm_is_multi_consumer() {
        let graph = three_node_graph();
        let use_counts = vec![2, 1, 0]; // InstanceNorm has 2 consumers
        let mut steps = vec![
            instance_norm_step(1e-5, vec![1, 4, 8]),
            dispatch_step("mul"),
            dispatch_step("add"),
        ];

        fuse_instance_norm_mul_add(&mut steps, &use_counts, &graph);

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
        let graph = three_node_graph();
        let use_counts = vec![1, 1, 0];
        let mut steps = vec![
            instance_norm_step(1e-5, vec![1, 4, 8]),
            dispatch_step("mul"),
            dispatch_step("sub"), // wrong — should be "add"
        ];

        fuse_instance_norm_mul_add(&mut steps, &use_counts, &graph);

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
        let graph = three_node_graph();
        let use_counts = vec![1, 1, 0];
        let mut steps = vec![
            instance_norm_step(1e-5, vec![4, 8]), // rank 2 — too low
            dispatch_step("mul"),
            dispatch_step("add"),
        ];

        fuse_instance_norm_mul_add(&mut steps, &use_counts, &graph);

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
    fn test_skip_when_snake_follows() {
        // If snake_tensor follows, this is the FusedAdainSnake pattern — skip.
        let nodes = vec![
            test_node(0, "instance_norm", vec![]),
            test_node(1, "mul", vec![0]),
            test_node(2, "add", vec![1]),
            test_node(3, "snake", vec![2]),
        ];
        let graph = ComputationGraph::from_nodes(nodes);
        let use_counts = vec![1, 1, 1, 0];

        let mut snake_step = dispatch_step("snake_tensor");
        // Need to add alpha weight to match snake_tensor pattern
        if let CompiledStep::Dispatch {
            ref mut weight_data,
            ..
        } = snake_step
        {
            weight_data.insert(
                "alpha".to_string(),
                nn_core::dyn_tensor::trace::WeightRef::new(vec![1.0; 4], vec![4])
                    .expect("valid alpha"),
            );
        }

        let mut steps = vec![
            instance_norm_step(1e-5, vec![1, 4, 8]),
            dispatch_step("mul"),
            dispatch_step("add"),
            snake_step,
        ];

        fuse_instance_norm_mul_add(&mut steps, &use_counts, &graph);

        // Should NOT fuse — snake_tensor follows, defer to FusedAdainSnake pass.
        assert!(matches!(
            &steps[0],
            CompiledStep::NativeOp {
                op: NativeOpKind::InstanceNorm { .. },
                ..
            }
        ));
    }

    #[test]
    fn test_fewer_than_three_steps_is_safe() {
        let nodes = vec![test_node(0, "a", vec![])];
        let graph = ComputationGraph::from_nodes(nodes);
        let use_counts = vec![0];
        let mut steps = vec![instance_norm_step(1e-5, vec![1, 4, 8])];

        // Should not panic.
        fuse_instance_norm_mul_add(&mut steps, &use_counts, &graph);
    }

    #[test]
    fn test_fuse_has_no_static_weights() {
        let graph = three_node_graph();
        let use_counts = vec![1, 1, 0];
        let mut steps = vec![
            instance_norm_step(1e-5, vec![1, 4, 8]),
            dispatch_step("mul"),
            dispatch_step("add"),
        ];

        fuse_instance_norm_mul_add(&mut steps, &use_counts, &graph);

        // Verify the fused step has no static weights (gamma/beta are graph inputs).
        match &steps[0] {
            CompiledStep::NativeOp { weight_data, .. } => {
                assert!(
                    weight_data.is_empty(),
                    "FusedInstanceNormMulAdd should have no static weights"
                );
            }
            other => panic!("expected NativeOp, got {other:?}"),
        }
    }
}
