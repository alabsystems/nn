// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Peephole pass: fuse Upsample1d + Conv1d into FusedUpsampleConv1d.
//!
//! Detects consecutive step pairs:
//!   Step i:   Dispatch { kernel: "upsample1d" }
//!   Step i+1: Dispatch { kernel: "conv1d" }
//!
//! The upsample output must be single-consumer (use_counts == 1).
//!
//! Replaces the pair with:
//! - steps[i]   -> NativeOp { FusedUpsampleConv1d { ... } }
//! - steps[i+1] -> IdentityPassthrough
//!
//! Saves 1 plan step per pair. In Kokoro f0_energy segment, 6 such pairs
//! exist, reducing 12 plan steps to 6.
//!
//! Part of #4310.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, WeightRef};

use super::super::{CompiledStep, NativeOpKind};
use super::extract_conv1d_params;
use crate::tensor_ir::TensorOpKind;

/// Scan for consecutive Dispatch("upsample1d") + Dispatch("conv1d") pairs
/// and fuse them into a single FusedUpsampleConv1d NativeOp.
pub(super) fn fuse_upsample_conv1d(
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
        if try_fuse_upsample_conv1d(steps, i, use_counts, graph) {
            // Skip past the fused pair.
            i += 2;
        } else {
            i += 1;
        }
    }
}

/// Try to fuse steps[i] (upsample1d) with steps[i+1] (conv1d).
///
/// Returns `true` if the pair was fused (steps mutated in-place).
fn try_fuse_upsample_conv1d(
    steps: &mut [CompiledStep],
    i: usize,
    use_counts: &[usize],
    graph: &ComputationGraph,
) -> bool {
    // ---- Step i: Dispatch with kernel name "upsample1d" ----
    let upsample_info = match &steps[i] {
        CompiledStep::Dispatch {
            kernel,
            weight_data: _,
            ..
        } if kernel.name() == "upsample1d" => extract_upsample_params(kernel),
        _ => None,
    };

    let upsample_info = match upsample_info {
        Some(info) => info,
        None => return false,
    };

    // Fan-out check: the upsample output must have exactly 1 consumer.
    if use_counts.get(i).copied().unwrap_or(0) != 1 {
        return false;
    }

    // ---- Step i+1: Dispatch with kernel name "conv1d" ----
    let conv_info = match &steps[i + 1] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } if kernel.name() == "conv1d" => extract_conv1d_params(kernel, weight_data),
        _ => None,
    };

    let conv_info = match conv_info {
        Some(info) => info,
        None => return false,
    };

    // Build merged weight_data.
    let mut merged_weight_data: HashMap<String, WeightRef> = HashMap::new();
    if let Some(w) = conv_info.weight {
        merged_weight_data.insert("weight".to_string(), w);
    }
    if let Some(b) = conv_info.bias {
        merged_weight_data.insert("bias".to_string(), b);
    }

    // Capture the graph node IDs from the upsample step so the edge_map
    // builder can resolve edges generically.
    let ext_ids = graph.nodes().get(i).map(|node| node.inputs().to_vec());

    let fused_op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: upsample_info.factor,
        in_channels: upsample_info.in_channels,
        out_channels: conv_info.output_channels,
        kernel_size: conv_info.kernel_size,
        stride: conv_info.stride,
        padding: conv_info.padding,
        input_shape: upsample_info.input_shape,
    };

    // Place FusedUpsampleConv1d at step[i] (upsample position).
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

/// Extracted upsample parameters.
struct UpsampleInfo {
    /// Nearest-neighbor upsample factor.
    factor: usize,
    /// Number of input channels (from input shape dim 1).
    in_channels: usize,
    /// Full input shape `[B, C, T]`.
    input_shape: Vec<usize>,
}

/// Extract upsample parameters from a compiled "upsample1d" kernel.
///
/// The upsample1d kernel IR is:
///   node 0: Input `[..., T]`
///   node 1: Reshape `[..., T, 1]`
///   node 2: Broadcast `[..., T, factor]`
///   node 3: Reshape `[..., T*factor]`
///
/// The factor is the last element of the Broadcast target_shape.
fn extract_upsample_params(kernel: &super::super::CompiledKernel) -> Option<UpsampleInfo> {
    let def = kernel.def();

    // Find the Broadcast node to extract the factor.
    let broadcast_node = def
        .nodes
        .iter()
        .find(|n| matches!(n.kind, TensorOpKind::Broadcast { .. }))?;

    let factor = match &broadcast_node.kind {
        TensorOpKind::Broadcast { target_shape, .. } => {
            // The last dimension of the broadcast target is the factor.
            *target_shape.last()?
        }
        _ => return None,
    };

    if factor == 0 {
        return None;
    }

    // Extract input shape from the first Input node.
    let input_node = def.nodes.first()?;
    let input_shape = match &input_node.kind {
        TensorOpKind::Input { shape, .. } => shape.clone(),
        _ => return None,
    };

    // Input must be at least rank 3 for [B, C, T].
    if input_shape.len() < 3 {
        return None;
    }
    let in_channels = input_shape[input_shape.len() - 2];

    Some(UpsampleInfo {
        factor,
        in_channels,
        input_shape,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_block_builder::TensorBlockBuilder;
    use crate::trace_compile::CompiledKernel;
    use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
    use nn_core::DType;

    /// Helper: build an upsample1d Dispatch step with a given input shape and factor.
    fn upsample1d_step(input_shape: &[usize], factor: usize) -> CompiledStep {
        let rank = input_shape.len();
        let in_t = input_shape[rank - 1];

        let mut b = TensorBlockBuilder::new("upsample1d");
        let input = b.add_input("input_0", input_shape);

        // Reshape: [..., T] -> [..., T, 1]
        let mut unsq = input_shape.to_vec();
        unsq.push(1);
        let r1 = b.add_reshape(input, &unsq);

        // Broadcast: [..., T, 1] -> [..., T, factor]
        let mut exp = input_shape.to_vec();
        exp.push(factor);
        let r2 = b.add_broadcast(r1, &exp);

        // Reshape: [..., T, factor] -> [..., T*factor]
        let mut out_shape = input_shape.to_vec();
        out_shape[rank - 1] = in_t * factor;
        let output = b.add_reshape(r2, &out_shape);

        let def = b.build(output).expect("build upsample1d");
        CompiledStep::Dispatch {
            kernel: CompiledKernel::new(def),
            weight_data: HashMap::new(),
            external_node_ids: None,
        }
    }

    /// Helper: build a conv1d Dispatch step with weights.
    fn conv1d_step(in_channels: usize, out_channels: usize, kernel_size: usize) -> CompiledStep {
        let in_t: usize = 64;
        let padding: usize = 1;
        let stride: usize = 1;
        // out_t = (in_t + 2*padding - kernel_size) / stride + 1
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

    /// Helper: build a test TraceNode.
    fn test_node(id: u64, name: &str, inputs: Vec<u64>) -> TraceNode {
        TraceNode::new(
            id,
            name.to_string(),
            TraceOp::Relu, // dummy op — peephole only inspects CompiledSteps
            inputs,
            vec![1, 4, 64],
            DType::F32,
        )
    }

    /// Build a minimal computation graph with 2 nodes for the pattern.
    fn two_node_graph() -> ComputationGraph {
        let nodes = vec![
            test_node(0, "upsample1d", vec![]),
            test_node(1, "conv1d", vec![0]),
        ];
        ComputationGraph::from_nodes(nodes)
    }

    #[test]
    fn test_fuse_upsample_conv1d_basic() {
        let graph = two_node_graph();
        let use_counts = vec![1, 0]; // upsample has 1 consumer
        let mut steps = vec![upsample1d_step(&[1, 4, 16], 4), conv1d_step(4, 8, 3)];

        fuse_upsample_conv1d(&mut steps, &use_counts, &graph);

        // Step 0 should be FusedUpsampleConv1d.
        assert!(
            matches!(
                &steps[0],
                CompiledStep::NativeOp {
                    op: NativeOpKind::FusedUpsampleConv1d {
                        upsample_factor: 4,
                        in_channels: 4,
                        out_channels: 8,
                        kernel_size: 3,
                        ..
                    },
                    ..
                }
            ),
            "expected FusedUpsampleConv1d, got {:?}",
            steps[0]
        );

        // Step 1 should be IdentityPassthrough.
        assert!(matches!(steps[1], CompiledStep::IdentityPassthrough));
    }

    #[test]
    fn test_no_fuse_when_upsample_has_multiple_consumers() {
        let graph = two_node_graph();
        let use_counts = vec![2, 0]; // upsample has 2 consumers
        let mut steps = vec![upsample1d_step(&[1, 4, 16], 4), conv1d_step(4, 8, 3)];

        fuse_upsample_conv1d(&mut steps, &use_counts, &graph);

        // Should NOT fuse — fan-out > 1.
        assert!(
            matches!(&steps[0], CompiledStep::Dispatch { kernel, .. } if kernel.name() == "upsample1d"),
            "should remain upsample1d Dispatch"
        );
    }

    #[test]
    fn test_no_fuse_wrong_kernel_names() {
        let graph = two_node_graph();
        let use_counts = vec![1, 0];

        // Build a step that is NOT upsample1d.
        let mut b = TensorBlockBuilder::new("relu");
        let input = b.add_input("input_0", &[1, 4, 16]);
        let def = b.build(input).expect("build");
        let wrong_step = CompiledStep::Dispatch {
            kernel: CompiledKernel::new(def),
            weight_data: HashMap::new(),
            external_node_ids: None,
        };

        let mut steps = vec![wrong_step, conv1d_step(4, 8, 3)];

        fuse_upsample_conv1d(&mut steps, &use_counts, &graph);

        // Should NOT fuse — first step is not upsample1d.
        assert!(
            matches!(&steps[0], CompiledStep::Dispatch { kernel, .. } if kernel.name() == "relu"),
        );
    }

    #[test]
    fn test_fuse_preserves_conv_weights() {
        let graph = two_node_graph();
        let use_counts = vec![1, 0];
        let mut steps = vec![upsample1d_step(&[1, 4, 16], 2), conv1d_step(4, 8, 5)];

        fuse_upsample_conv1d(&mut steps, &use_counts, &graph);

        match &steps[0] {
            CompiledStep::NativeOp { weight_data, .. } => {
                assert!(
                    weight_data.contains_key("weight"),
                    "FusedUpsampleConv1d must carry conv weight"
                );
                assert!(
                    weight_data.contains_key("bias"),
                    "FusedUpsampleConv1d must carry conv bias"
                );
                assert_eq!(
                    weight_data["weight"].shape(),
                    &[8, 4, 5],
                    "weight shape must be [out_channels, in_channels, kernel_size]"
                );
                assert_eq!(
                    weight_data["bias"].shape(),
                    &[8],
                    "bias shape must be [out_channels]"
                );
            }
            other => panic!("expected NativeOp, got {other:?}"),
        }
    }

    #[test]
    fn test_fuse_records_input_shape() {
        let graph = two_node_graph();
        let use_counts = vec![1, 0];
        let mut steps = vec![upsample1d_step(&[1, 4, 16], 4), conv1d_step(4, 8, 3)];

        fuse_upsample_conv1d(&mut steps, &use_counts, &graph);

        match &steps[0] {
            CompiledStep::NativeOp {
                op: NativeOpKind::FusedUpsampleConv1d { input_shape, .. },
                ..
            } => {
                assert_eq!(
                    input_shape,
                    &[1, 4, 16],
                    "input_shape should be the upsample input"
                );
            }
            other => panic!("expected FusedUpsampleConv1d, got {other:?}"),
        }
    }

    #[test]
    fn test_fewer_than_two_steps_is_safe() {
        let nodes = vec![test_node(0, "a", vec![])];
        let graph = ComputationGraph::from_nodes(nodes);
        let use_counts = vec![0];
        let mut steps = vec![upsample1d_step(&[1, 4, 16], 2)];

        // Should not panic.
        fuse_upsample_conv1d(&mut steps, &use_counts, &graph);
        assert!(matches!(
            &steps[0],
            CompiledStep::Dispatch { kernel, .. } if kernel.name() == "upsample1d"
        ));
    }

    #[test]
    fn test_fuse_multiple_pairs() {
        // Build a graph with 2 upsample+conv1d pairs: nodes 0-1 and 2-3.
        let nodes = vec![
            test_node(0, "upsample1d", vec![]),
            test_node(1, "conv1d", vec![0]),
            test_node(2, "upsample1d", vec![1]),
            test_node(3, "conv1d", vec![2]),
        ];
        let graph = ComputationGraph::from_nodes(nodes);
        let use_counts = vec![1, 1, 1, 0];
        let mut steps = vec![
            upsample1d_step(&[1, 4, 16], 2),
            conv1d_step(4, 8, 3),
            upsample1d_step(&[1, 8, 32], 2),
            conv1d_step(8, 16, 3),
        ];

        fuse_upsample_conv1d(&mut steps, &use_counts, &graph);

        // Both pairs should be fused.
        assert!(matches!(
            &steps[0],
            CompiledStep::NativeOp {
                op: NativeOpKind::FusedUpsampleConv1d { .. },
                ..
            }
        ));
        assert!(matches!(steps[1], CompiledStep::IdentityPassthrough));
        assert!(matches!(
            &steps[2],
            CompiledStep::NativeOp {
                op: NativeOpKind::FusedUpsampleConv1d { .. },
                ..
            }
        ));
        assert!(matches!(steps[3], CompiledStep::IdentityPassthrough));
    }
}
