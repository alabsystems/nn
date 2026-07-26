// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Attention transpose absorption peephole pass.
//!
//! Detects `Transpose(1,2) → FlashAttention → Transpose(1,2)` patterns
//! in compiled step sequences and eliminates the transposes by switching
//! the FlashAttention NativeOp to `SeqFirst` layout.
//!
//! In the standard multi-head attention pattern:
//! ```text
//! Q = Linear → Reshape [B,T,H,D] → Transpose(1,2) → [B,H,T,D]
//! FlashAttention([B,H,T,D]) → [B,H,T,D]
//! Output = Transpose(1,2) → [B,T,H,D] → Reshape [B,T,hidden]
//! ```
//!
//! Each Transpose is a full-data-copy GPU dispatch. With 12 PlBert layers
//! × 4 transposes each (3 QKV + 1 output) = 48 dispatches eliminated.
//!
//! Part of #1815 (Tier 5 D1).

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceOp};

use super::super::{AttentionLayout, CompiledStep, NativeOpKind};

/// Absorb Transpose(1,2) dispatches into FlashAttention NativeOps.
///
/// For each FlashAttention step:
/// 1. Check if all 3 inputs (Q, K, V) are `Transpose { dim0: 1, dim1: 2 }` nodes
/// 2. Check if the single output consumer is also `Transpose { dim0: 1, dim1: 2 }`
/// 3. If both conditions hold, replace all 4 Transposes with IdentityPassthrough
///    and switch the FlashAttention to `SeqFirst` layout with updated shapes
pub(crate) fn absorb_attention_transposes(steps: &mut [CompiledStep], graph: &ComputationGraph) {
    let nodes = graph.nodes();
    if nodes.is_empty() {
        return;
    }

    // Build node ID → step index mapping.
    let id_to_idx: HashMap<u64, usize> =
        nodes.iter().enumerate().map(|(i, n)| (n.id(), i)).collect();

    // Build consumers map: step_idx → list of consumer step indices.
    let mut consumers: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for &input_id in node.inputs() {
            if let Some(&input_idx) = id_to_idx.get(&input_id) {
                consumers.entry(input_idx).or_default().push(i);
            }
        }
    }

    // Scan for FlashAttention NativeOps.
    for step_idx in 0..steps.len() {
        let is_flash = matches!(
            &steps[step_idx],
            CompiledStep::NativeOp {
                op: NativeOpKind::FlashAttention { .. },
                ..
            }
        );
        if !is_flash {
            continue;
        }

        // Get the SDPA node's input IDs.
        let node = &nodes[step_idx];
        let inputs = node.inputs();
        if inputs.len() < 3 {
            continue;
        }

        // Check that all 3 inputs are Transpose(1,2) steps.
        let q_idx = id_to_idx.get(&inputs[0]).copied();
        let k_idx = id_to_idx.get(&inputs[1]).copied();
        let v_idx = id_to_idx.get(&inputs[2]).copied();

        let (q_idx, k_idx, v_idx) = match (q_idx, k_idx, v_idx) {
            (Some(q), Some(k), Some(v)) => (q, k, v),
            _ => continue,
        };

        if !is_transpose_1_2(&nodes[q_idx])
            || !is_transpose_1_2(&nodes[k_idx])
            || !is_transpose_1_2(&nodes[v_idx])
        {
            continue;
        }

        // Check that the output consumer is a single Transpose(1,2).
        let output_consumers = consumers.get(&step_idx);
        let output_transpose_idx = match output_consumers {
            Some(c) if c.len() == 1 => {
                let idx = c[0];
                if is_transpose_1_2(&nodes[idx]) {
                    Some(idx)
                } else {
                    None
                }
            }
            _ => None,
        };

        // We need at minimum the 3 input transposes to be absorbed.
        // The output transpose is optional (it's a bonus if present).
        // For SeqFirst layout, the FlashAttention output is [B,T,H,D]
        // and any downstream Transpose(1,2) would produce [B,H,T,D] —
        // wrong. So we REQUIRE the output transpose too.
        let output_transpose_idx = match output_transpose_idx {
            Some(idx) => idx,
            None => continue,
        };

        // Get the pre-transpose shapes for Q, K, V.
        // Input transposes convert [B,T,H,D] → [B,H,T,D].
        // The pre-transpose shape is [B,T,H,D] (the input to the Transpose node).
        let q_pre_shape = nodes[q_idx].output_shape(); // This is post-transpose [B,H,T,D]
        let k_pre_shape = nodes[k_idx].output_shape();

        // For SeqFirst, we need the shapes BEFORE transpose.
        // The Transpose(1,2) node's input has shape [B,T,H,D].
        // Its output is [B,H,T,D]. So to get [B,T,H,D], swap dims 1,2 back.
        let q_seq_first_shape = swap_dims_1_2(q_pre_shape);
        let k_seq_first_shape = swap_dims_1_2(k_pre_shape);

        if q_seq_first_shape.len() != 4 || k_seq_first_shape.len() != 4 {
            continue;
        }

        // Output shape in SeqFirst: [B, T, H_q, D] (same as Q SeqFirst shape).
        let output_shape = q_seq_first_shape.clone();

        // All checks passed. Apply the transformation.

        // 1. Replace input Transposes with IdentityPassthrough.
        steps[q_idx] = CompiledStep::IdentityPassthrough;
        steps[k_idx] = CompiledStep::IdentityPassthrough;
        steps[v_idx] = CompiledStep::IdentityPassthrough;

        // 2. Replace output Transpose with IdentityPassthrough.
        steps[output_transpose_idx] = CompiledStep::IdentityPassthrough;

        // 3. Update FlashAttention to SeqFirst layout with new shapes.
        if let CompiledStep::NativeOp {
            op:
                NativeOpKind::FlashAttention {
                    ref mut q_shape,
                    ref mut k_shape,
                    output_shape: ref mut out_shape,
                    ref mut input_layout,
                    ..
                },
            ..
        } = steps[step_idx]
        {
            *q_shape = q_seq_first_shape;
            *k_shape = k_seq_first_shape;
            *out_shape = output_shape;
            *input_layout = AttentionLayout::SeqFirst;
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

/// Swap dimensions 1 and 2 in a shape vector.
/// `[B, H, T, D]` → `[B, T, H, D]` (or vice versa).
fn swap_dims_1_2(shape: &[usize]) -> Vec<usize> {
    if shape.len() < 3 {
        return shape.to_vec();
    }
    let mut result = shape.to_vec();
    result.swap(1, 2);
    result
}

#[cfg(test)]
#[path = "trace_compile_peephole_attention_tests.rs"]
mod tests;
