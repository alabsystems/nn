// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Peephole pass 13: Transpose(1,2) + LayerNorm + Transpose(1,2) elimination.
//!
//! Detects the pattern:
//! ```text
//! step[i]   = Transpose(1,2)   — [B, C, T] → [B, T, C]
//! step[i+1] = LayerNorm        — normalizes over last dim (C)
//! step[i+2] = Transpose(1,2)   — [B, T, C] → [B, C, T]
//! ```
//!
//! And replaces with:
//! ```text
//! step[i]   = IdentityPassthrough
//! step[i+1] = ChannelsFirstLayerNorm  — normalizes over dim 1 (C) in-place
//! step[i+2] = IdentityPassthrough
//! ```
//!
//! Eliminates 2 data-copy transpose dispatches per occurrence. In Kokoro
//! TextEncoder, this fires 3 times (one per conv-norm iteration) for a
//! total of 6 transposes eliminated.
//!
//! Part of #3457.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceOp};

use super::super::{CompiledStep, NativeOpKind};

/// Absorb Transpose(1,2) pairs surrounding LayerNorm NativeOps.
///
/// Scans graph nodes for the Transpose→LayerNorm→Transpose pattern.
/// When found with matching dims and single-consumer constraints,
/// replaces the triplet with ChannelsFirstLayerNorm.
pub(super) fn absorb_transpose_layer_norm(
    steps: &mut [CompiledStep],
    graph: &ComputationGraph,
    use_counts: &[usize],
) {
    let nodes = graph.nodes();
    if nodes.len() < 3 {
        return;
    }

    // Scan for LayerNorm NativeOps and check if they're bracketed by transposes.
    for ln_idx in 1..nodes.len().saturating_sub(1) {
        // Step must be NativeOp{LayerNorm}.
        let (ln_eps, _ln_input_shape, ln_hidden_dim, ln_weight_data) = match &steps[ln_idx] {
            CompiledStep::NativeOp {
                op:
                    NativeOpKind::LayerNorm {
                        eps,
                        input_shape,
                        hidden_dim,
                    },
                weight_data,
            } => (*eps, input_shape.clone(), *hidden_dim, weight_data.clone()),
            _ => continue,
        };

        // The graph node for the LayerNorm must have exactly one input.
        let ln_node = &nodes[ln_idx];
        let ln_inputs = ln_node.inputs();
        if ln_inputs.len() != 1 {
            continue;
        }

        // Find the predecessor: must be a Transpose(1,2) in the graph.
        let pre_transpose_id = ln_inputs[0];
        let pre_idx = nodes.iter().position(|n| n.id() == pre_transpose_id);
        let pre_idx = match pre_idx {
            Some(idx) => idx,
            None => continue,
        };
        if !is_transpose_1_2(&nodes[pre_idx]) {
            continue;
        }

        // Fan-out: pre-transpose output must feed only the LayerNorm.
        if use_counts.get(pre_idx).copied().unwrap_or(0) != 1 {
            continue;
        }

        // Find the successor: must be a Transpose(1,2) that consumes the LayerNorm output.
        // Fan-out: LayerNorm output must have exactly 1 consumer.
        if use_counts.get(ln_idx).copied().unwrap_or(0) != 1 {
            continue;
        }
        let ln_output_id = ln_node.id();
        let post_idx = nodes
            .iter()
            .position(|n| n.inputs().contains(&ln_output_id) && is_transpose_1_2(n));
        let post_idx = match post_idx {
            Some(idx) => idx,
            None => continue,
        };

        // Validate shapes:
        // Pre-transpose input is [B, C, T]. After transpose → [B, T, C].
        // LayerNorm normalizes over C (hidden_dim). After transpose back → [B, C, T].
        // The channels-first kernel takes [B, C, T] and normalizes over C.
        let pre_node = &nodes[pre_idx];
        let pre_input_shape = pre_node.inputs().first().and_then(|&id| {
            nodes
                .iter()
                .find(|n| n.id() == id)
                .map(nn_core::dyn_tensor::trace::TraceNode::output_shape)
        });
        let channels_first_shape = match pre_input_shape {
            Some(s) if s.len() >= 3 => s.to_vec(),
            _ => continue,
        };
        let channels = channels_first_shape[1];

        // hidden_dim must match channels.
        if ln_hidden_dim != channels {
            continue;
        }

        // All checks passed. Apply the transformation.

        // Check if the post-transpose's consumer is a LeakyRelu we can absorb.
        let post_output_id = nodes[post_idx].id();
        let leaky_relu_idx = if use_counts.get(post_idx).copied().unwrap_or(0) == 1 {
            nodes
                .iter()
                .enumerate()
                .find(|(_, n)| {
                    n.inputs().contains(&post_output_id)
                        && matches!(n.op(), TraceOp::LeakyRelu { .. })
                })
                .map(|(idx, _)| idx)
        } else {
            None
        };
        let leaky_relu_slope = leaky_relu_idx.and_then(|idx| match nodes[idx].op() {
            TraceOp::LeakyRelu { slope } => Some(*slope as f32),
            _ => None,
        });

        // Replace pre-transpose with IdentityPassthrough.
        steps[pre_idx] = CompiledStep::IdentityPassthrough;

        // Replace LayerNorm with ChannelsFirstLayerNorm (with optional fused activation).
        steps[ln_idx] = CompiledStep::NativeOp {
            op: NativeOpKind::ChannelsFirstLayerNorm {
                eps: ln_eps,
                input_shape: channels_first_shape,
                channels,
                leaky_relu_slope,
            },
            weight_data: ln_weight_data,
        };

        // Replace post-transpose with IdentityPassthrough.
        steps[post_idx] = CompiledStep::IdentityPassthrough;

        // If LeakyRelu was absorbed, replace it with IdentityPassthrough too.
        if let Some(lr_idx) = leaky_relu_idx {
            steps[lr_idx] = CompiledStep::IdentityPassthrough;
        }
    }
}

/// Check if a trace node is `Transpose { dim0: 1, dim1: 2 }`.
fn is_transpose_1_2(node: &nn_core::dyn_tensor::trace::TraceNode) -> bool {
    matches!(
        node.op(),
        TraceOp::Transpose { dim0: 1, dim1: 2 } | TraceOp::Transpose { dim0: 2, dim1: 1 }
    )
}

#[cfg(test)]
#[path = "trace_compile_peephole_conv_ln_tests.rs"]
mod tests;
