// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Peephole pass: fuse 2x FusedConv1dSnakeNorm + residual add
//! into a single `FusedConv1dSnakeNormResBlock` NativeOp.
//!
//! Detects the pattern in compiled steps after the FusedConv1dSnakeNorm pass:
//!   Step a: NativeOp { FusedConv1dSnakeNorm { ... } }  (phase 1)
//!   Step b: NativeOp { FusedConv1dSnakeNorm { ... } }  (phase 2)
//!   Step c: Dispatch { kernel: "add" }                  (residual add)
//!
//! Where step a's output feeds step b, and both step a's input (x) and step b's
//! output feed into the add. The pattern represents a complete Kokoro Generator
//! ResBlock without AdaIN style projection:
//!   conv1d -> snake -> instance_norm -> conv1d -> snake -> instance_norm -> add
//!
//! Replaces 3 steps with a single `FusedConv1dSnakeNormResBlock` NativeOp.
//! The executor sequences the same operations internally (2x conv->snake->norm
//! + residual add), but in a single plan step with merged weight data.
//!
//! Must run AFTER the FusedConv1dSnakeNorm pass (which creates the input steps)
//! and BEFORE the FusedResBlock pass 2 (which handles AdaIN-style ResBlocks).
//!
//! Part of #4264.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::ComputationGraph;

use super::super::{CompiledStep, NativeOpKind};

/// Scan for 2x FusedConv1dSnakeNorm + add patterns and fuse them.
pub(super) fn fuse_conv1d_snake_norm_resblock(
    steps: &mut [CompiledStep],
    use_counts: &[usize],
    graph: &ComputationGraph,
) {
    let len = steps.len();
    if len < 3 {
        return;
    }

    let nodes = graph.nodes();
    let id_to_idx: HashMap<u64, usize> =
        nodes.iter().enumerate().map(|(i, n)| (n.id(), i)).collect();

    // Find all Dispatch "add" steps and try to fuse.
    let mut add_candidates: Vec<usize> = (0..len)
        .filter(|&idx| {
            matches!(
                &steps[idx],
                CompiledStep::Dispatch { kernel, .. } if kernel.name() == "add"
            )
        })
        .collect();
    // Process from the end so replacements don't shift indices.
    add_candidates.reverse();

    for add_idx in add_candidates {
        try_fuse_at_add(steps, add_idx, nodes, &id_to_idx, use_counts);
    }
}

/// Extract FusedConv1dSnakeNorm params from a step.
/// Returns (out_channels, kernel_size, stride, padding, dilation, groups, has_bias, eps, input_shape, weight_data).
#[allow(clippy::type_complexity)]
fn extract_conv1d_snake_norm_params(
    step: &CompiledStep,
) -> Option<(
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    bool,
    f32,
    Vec<usize>,
    HashMap<String, nn_core::dyn_tensor::trace::WeightRef>,
)> {
    match step {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::FusedConv1dSnakeNorm {
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                    has_bias,
                    eps,
                    input_shape,
                },
            weight_data,
        } => {
            // Only fuse stride=1, groups=1 (standard ResBlock pattern).
            if *stride != 1 || *groups != 1 {
                return None;
            }
            Some((
                *out_channels,
                *kernel_size,
                *stride,
                *padding,
                *dilation,
                *groups,
                *has_bias,
                *eps,
                input_shape.clone(),
                weight_data.clone(),
            ))
        }
        _ => None,
    }
}

/// Try to fuse a ResBlock pattern anchored at a `Dispatch "add"` step.
fn try_fuse_at_add(
    steps: &mut [CompiledStep],
    add_idx: usize,
    nodes: &[nn_core::dyn_tensor::trace::TraceNode],
    id_to_idx: &HashMap<u64, usize>,
    use_counts: &[usize],
) -> bool {
    // Verify step at add_idx is Dispatch "add".
    if !matches!(
        &steps[add_idx],
        CompiledStep::Dispatch { kernel, .. } if kernel.name() == "add"
    ) {
        return false;
    }

    // add node inputs: [a, b] -- one is x (residual), other is phase2 output.
    let add_inputs = nodes[add_idx].inputs();
    if add_inputs.len() != 2 {
        return false;
    }

    // Try both orderings: input[1] as conv chain first, then input[0].
    let chain = trace_chain(
        add_inputs[1],
        add_inputs[0],
        steps,
        nodes,
        id_to_idx,
        use_counts,
    )
    .or_else(|| {
        trace_chain(
            add_inputs[0],
            add_inputs[1],
            steps,
            nodes,
            id_to_idx,
            use_counts,
        )
    });

    let chain = match chain {
        Some(c) => c,
        None => return false,
    };

    // Merge weight data with phase prefixes.
    let mut merged_weights = HashMap::new();
    for (k, v) in &chain.phase1_weights {
        merged_weights.insert(format!("p1_{k}"), v.clone());
    }
    for (k, v) in &chain.phase2_weights {
        merged_weights.insert(format!("p2_{k}"), v.clone());
    }

    let fused_op = NativeOpKind::FusedConv1dSnakeNormResBlock {
        phase1_out_channels: chain.phase1_out_channels,
        phase1_kernel_size: chain.phase1_kernel_size,
        phase1_padding: chain.phase1_padding,
        phase1_dilation: chain.phase1_dilation,
        phase1_has_bias: chain.phase1_has_bias,
        phase2_out_channels: chain.phase2_out_channels,
        phase2_kernel_size: chain.phase2_kernel_size,
        phase2_padding: chain.phase2_padding,
        phase2_dilation: chain.phase2_dilation,
        phase2_has_bias: chain.phase2_has_bias,
        eps: chain.eps,
        residual_scale: 1.0,
        input_shape: chain.input_shape,
        x_step: chain.x_step,
    };

    // Replace absorbed steps with IdentityPassthrough.
    steps[chain.phase1_idx] = CompiledStep::IdentityPassthrough;
    steps[chain.phase2_idx] = CompiledStep::IdentityPassthrough;

    // Place the fused op at the add position.
    steps[add_idx] = CompiledStep::NativeOp {
        op: fused_op,
        weight_data: merged_weights,
    };

    true
}

/// Result of successfully tracing the two-phase FusedConv1dSnakeNorm chain.
struct ChainResult {
    phase1_out_channels: usize,
    phase1_kernel_size: usize,
    phase1_padding: usize,
    phase1_dilation: usize,
    phase1_has_bias: bool,
    phase1_weights: HashMap<String, nn_core::dyn_tensor::trace::WeightRef>,
    phase2_out_channels: usize,
    phase2_kernel_size: usize,
    phase2_padding: usize,
    phase2_dilation: usize,
    phase2_has_bias: bool,
    phase2_weights: HashMap<String, nn_core::dyn_tensor::trace::WeightRef>,
    eps: f32,
    input_shape: Vec<usize>,
    phase1_idx: usize,
    phase2_idx: usize,
    x_step: usize,
}

/// Trace back from `phase2_candidate_id` to find the two-phase chain.
///
/// `x_candidate_id` is the residual input from the other side of the add.
/// Verifies that phase1's input matches x_candidate_id (the residual connection).
fn trace_chain(
    phase2_candidate_id: u64,
    x_candidate_id: u64,
    steps: &[CompiledStep],
    nodes: &[nn_core::dyn_tensor::trace::TraceNode],
    id_to_idx: &HashMap<u64, usize>,
    use_counts: &[usize],
) -> Option<ChainResult> {
    let phase2_idx = *id_to_idx.get(&phase2_candidate_id)?;

    // Phase 2 must be a FusedConv1dSnakeNorm step.
    let (p2_oc, p2_ks, _p2_stride, p2_pad, p2_dil, _p2_groups, p2_bias, p2_eps, _p2_shape, p2_wd) =
        extract_conv1d_snake_norm_params(&steps[phase2_idx])?;

    // Fan-out: phase2 output feeds only into add.
    if use_counts.get(phase2_idx).copied().unwrap_or(0) != 1 {
        return None;
    }

    // Phase2 node input: must have exactly 1 graph input (the phase1 output).
    let phase2_inputs = nodes[phase2_idx].inputs();
    if phase2_inputs.len() != 1 {
        return None;
    }

    let phase1_idx = *id_to_idx.get(&phase2_inputs[0])?;

    // Phase 1 must also be a FusedConv1dSnakeNorm step.
    let (p1_oc, p1_ks, _p1_stride, p1_pad, p1_dil, _p1_groups, p1_bias, p1_eps, p1_shape, p1_wd) =
        extract_conv1d_snake_norm_params(&steps[phase1_idx])?;

    // Fan-out: phase1 output feeds only into phase2.
    if use_counts.get(phase1_idx).copied().unwrap_or(0) != 1 {
        return None;
    }

    // Phase1 node input: must have exactly 1 graph input (the residual x).
    let phase1_inputs = nodes[phase1_idx].inputs();
    if phase1_inputs.len() != 1 {
        return None;
    }

    // Verify residual connection: phase1's input must match x_candidate.
    if phase1_inputs[0] != x_candidate_id {
        return None;
    }

    let x_step = *id_to_idx.get(&x_candidate_id)?;

    // Use the first phase's eps (both should be the same in practice).
    let eps = p1_eps.min(p2_eps);

    Some(ChainResult {
        phase1_out_channels: p1_oc,
        phase1_kernel_size: p1_ks,
        phase1_padding: p1_pad,
        phase1_dilation: p1_dil,
        phase1_has_bias: p1_bias,
        phase1_weights: p1_wd,
        phase2_out_channels: p2_oc,
        phase2_kernel_size: p2_ks,
        phase2_padding: p2_pad,
        phase2_dilation: p2_dil,
        phase2_has_bias: p2_bias,
        phase2_weights: p2_wd,
        eps,
        input_shape: p1_shape,
        phase1_idx,
        phase2_idx,
        x_step,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
    use nn_core::DType;

    /// Helper: build a FusedConv1dSnakeNorm NativeOp step.
    fn conv1d_snake_norm_step(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        eps: f32,
    ) -> CompiledStep {
        let mut wd = HashMap::new();
        wd.insert(
            "conv_weight".to_string(),
            WeightRef::new(
                vec![0.0; out_channels * in_channels * kernel_size],
                vec![out_channels, in_channels, kernel_size],
            )
            .expect("valid weight"),
        );
        wd.insert(
            "conv_bias".to_string(),
            WeightRef::new(vec![0.0; out_channels], vec![out_channels]).expect("valid bias"),
        );
        wd.insert(
            "alpha".to_string(),
            WeightRef::new(vec![1.0; out_channels], vec![out_channels]).expect("valid alpha"),
        );
        CompiledStep::NativeOp {
            op: NativeOpKind::FusedConv1dSnakeNorm {
                out_channels,
                kernel_size,
                stride: 1,
                padding: kernel_size / 2,
                dilation: 1,
                groups: 1,
                has_bias: true,
                eps,
                input_shape: vec![1, in_channels, 64],
            },
            weight_data: wd,
        }
    }

    /// Helper: build a Dispatch "add" step.
    fn add_step() -> CompiledStep {
        use crate::tensor_block_builder::TensorBlockBuilder;
        let mut b = TensorBlockBuilder::new("add");
        let input = b.add_input("input_0", &[1, 8, 64]);
        let def = b.build(input).expect("build");
        CompiledStep::Dispatch {
            kernel: crate::trace_compile::CompiledKernel::new(def),
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

    /// Build a graph for: x -> phase1 -> phase2 -> add(x, phase2)
    fn resblock_graph() -> ComputationGraph {
        let nodes = vec![
            test_node(0, "x", vec![]),
            test_node(1, "phase1", vec![0]),
            test_node(2, "phase2", vec![1]),
            test_node(3, "add", vec![0, 2]),
        ];
        ComputationGraph::from_nodes(nodes)
    }

    #[test]
    fn test_fuse_two_conv1d_snake_norm_plus_add() {
        let graph = resblock_graph();
        let use_counts = vec![2, 1, 1, 0];
        let mut steps = vec![
            CompiledStep::IdentityPassthrough,
            conv1d_snake_norm_step(8, 8, 3, 1e-5),
            conv1d_snake_norm_step(8, 8, 3, 1e-5),
            add_step(),
        ];

        fuse_conv1d_snake_norm_resblock(&mut steps, &use_counts, &graph);

        assert!(
            matches!(
                &steps[3],
                CompiledStep::NativeOp {
                    op: NativeOpKind::FusedConv1dSnakeNormResBlock { .. },
                    ..
                }
            ),
            "expected FusedConv1dSnakeNormResBlock, got {:?}",
            steps[3]
        );
        assert!(matches!(steps[1], CompiledStep::IdentityPassthrough));
        assert!(matches!(steps[2], CompiledStep::IdentityPassthrough));
    }

    #[test]
    fn test_fuse_preserves_phase_weights() {
        let graph = resblock_graph();
        let use_counts = vec![2, 1, 1, 0];
        let mut steps = vec![
            CompiledStep::IdentityPassthrough,
            conv1d_snake_norm_step(8, 8, 3, 1e-5),
            conv1d_snake_norm_step(8, 8, 5, 1e-5),
            add_step(),
        ];

        fuse_conv1d_snake_norm_resblock(&mut steps, &use_counts, &graph);

        match &steps[3] {
            CompiledStep::NativeOp {
                op:
                    NativeOpKind::FusedConv1dSnakeNormResBlock {
                        phase1_kernel_size,
                        phase2_kernel_size,
                        ..
                    },
                weight_data,
            } => {
                assert_eq!(*phase1_kernel_size, 3);
                assert_eq!(*phase2_kernel_size, 5);
                assert!(weight_data.contains_key("p1_conv_weight"));
                assert!(weight_data.contains_key("p1_alpha"));
                assert!(weight_data.contains_key("p2_conv_weight"));
                assert!(weight_data.contains_key("p2_alpha"));
            }
            other => panic!("expected FusedConv1dSnakeNormResBlock, got {other:?}"),
        }
    }

    #[test]
    fn test_no_fuse_when_phase1_has_multiple_consumers() {
        let graph = resblock_graph();
        let use_counts = vec![2, 2, 1, 0];
        let mut steps = vec![
            CompiledStep::IdentityPassthrough,
            conv1d_snake_norm_step(8, 8, 3, 1e-5),
            conv1d_snake_norm_step(8, 8, 3, 1e-5),
            add_step(),
        ];

        fuse_conv1d_snake_norm_resblock(&mut steps, &use_counts, &graph);

        assert!(matches!(
            &steps[3],
            CompiledStep::Dispatch { kernel, .. } if kernel.name() == "add"
        ));
    }

    #[test]
    fn test_reversed_add_operand_order() {
        let nodes = vec![
            test_node(0, "x", vec![]),
            test_node(1, "phase1", vec![0]),
            test_node(2, "phase2", vec![1]),
            test_node(3, "add", vec![2, 0]),
        ];
        let graph = ComputationGraph::from_nodes(nodes);
        let use_counts = vec![2, 1, 1, 0];
        let mut steps = vec![
            CompiledStep::IdentityPassthrough,
            conv1d_snake_norm_step(8, 8, 3, 1e-5),
            conv1d_snake_norm_step(8, 8, 3, 1e-5),
            add_step(),
        ];

        fuse_conv1d_snake_norm_resblock(&mut steps, &use_counts, &graph);

        assert!(matches!(
            &steps[3],
            CompiledStep::NativeOp {
                op: NativeOpKind::FusedConv1dSnakeNormResBlock { .. },
                ..
            }
        ));
    }

    #[test]
    fn test_fewer_than_three_steps_safe() {
        let nodes = vec![test_node(0, "a", vec![]), test_node(1, "b", vec![0])];
        let graph = ComputationGraph::from_nodes(nodes);
        let use_counts = vec![1, 0];
        let mut steps = vec![conv1d_snake_norm_step(8, 8, 3, 1e-5), add_step()];

        fuse_conv1d_snake_norm_resblock(&mut steps, &use_counts, &graph);
    }
}
