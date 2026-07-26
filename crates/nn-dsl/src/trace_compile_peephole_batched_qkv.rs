// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pass 12: batch parallel Linear projections sharing the same input.
//!
//! Detects groups of 2+ `Dispatch{linear}` steps that consume the same
//! source step (e.g., Q/K/V projections in transformer attention blocks).
//! Concatenates their weights along dim 0 into a single matmul, then
//! narrows the output to recover individual projections (zero-copy).
//!
//! Saves N-1 dispatches per group:
//! - Whisper: ~68 (32 enc + 8 dec self-attn × 2, 4 cross K+V)
//! - Qwen3: ~72 (36 attention blocks × 2)
//! - Kokoro/PlBert: ~24 (12 attention blocks × 2)
//! - GLM-4/5: 0 (already fused QKV)
//!
//! Part of #3269.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, WeightRef};

use super::super::{CompiledStep, NativeOpKind};
use super::LinearInfo;

/// Candidate Linear step for batching.
struct BatchCandidate {
    step_idx: usize,
    source_step: usize,
    info: LinearInfo,
}

/// Pass 12: batch parallel Linear projections sharing the same input.
///
/// Uses the computation graph's edge map to find the actual source step
/// for each Linear projection. Steps consuming the same source are grouped
/// and batched into a single matmul.
pub(super) fn batch_linear_projections(
    steps: &mut [CompiledStep],
    _use_counts: &[usize],
    graph: &ComputationGraph,
) {
    let edge_map = crate::edge_map::compute_edge_map(graph, steps);
    batch_with_edge_map(steps, &edge_map);
}

/// Core batching logic operating on a pre-built edge map.
///
/// Separated for testability: tests provide synthetic edge maps
/// without building a full `ComputationGraph`.
fn batch_with_edge_map(steps: &mut [CompiledStep], edge_map: &[Vec<usize>]) {
    let candidates = find_linear_candidates(steps, edge_map);
    if candidates.is_empty() {
        return;
    }

    let mut groups: HashMap<usize, Vec<BatchCandidate>> = HashMap::new();
    for c in candidates {
        groups.entry(c.source_step).or_default().push(c);
    }

    let mut group_list: Vec<Vec<BatchCandidate>> =
        groups.into_values().filter(|g| g.len() >= 2).collect();
    group_list.sort_by_key(|g| g[0].step_idx);

    for group in group_list {
        let in_features = group[0].info.in_features;
        if group.iter().any(|c| c.info.in_features != in_features) {
            continue;
        }
        let has_bias = group[0].info.has_bias;
        if group.iter().any(|c| c.info.has_bias != has_bias) {
            continue;
        }
        batch_group(steps, &group, in_features, has_bias);
    }
}

/// Find all `Dispatch{linear}` steps and their source input step index.
///
/// Uses the edge map to find the primary input step (first edge) for each
/// Linear dispatch. This correctly handles non-adjacent Linears that share
/// the same hidden state input (e.g., Q/K/V in transformer attention).
fn find_linear_candidates(steps: &[CompiledStep], edge_map: &[Vec<usize>]) -> Vec<BatchCandidate> {
    let mut candidates = Vec::new();
    for (step_idx, step) in steps.iter().enumerate() {
        let info = match step {
            CompiledStep::Dispatch {
                kernel,
                weight_data,
                ..
            } if kernel.name() == "linear" => super::extract_linear_params(kernel, weight_data),
            _ => None,
        };
        let info = match info {
            Some(i) => i,
            None => continue,
        };

        // Primary input is the first edge (the hidden state, not weights).
        let source_step = edge_map
            .get(step_idx)
            .and_then(|edges| edges.first().copied());
        if let Some(source_step) = source_step {
            candidates.push(BatchCandidate {
                step_idx,
                source_step,
                info,
            });
        }
    }
    candidates
}

/// Batch a group of Linear steps sharing the same source into a single matmul.
fn batch_group(
    steps: &mut [CompiledStep],
    group: &[BatchCandidate],
    in_features: usize,
    has_bias: bool,
) {
    let first_step = group[0].step_idx;
    let input_shape = group[0].info.input_shape.clone();

    // Collect per-projection output sizes and concatenate weights.
    let mut projection_sizes = Vec::with_capacity(group.len());
    let mut concat_weight_data: Vec<f32> = Vec::new();
    let mut concat_bias_data: Vec<f32> = Vec::new();

    for c in group {
        projection_sizes.push(c.info.out_features);

        // Weight shape: [out_features, in_features] (row-major).
        if let Some(w) = c.info.weight_data.get("weight") {
            concat_weight_data.extend_from_slice(w.data());
        } else {
            return; // Missing weight — skip.
        }
        if has_bias {
            if let Some(b) = c.info.weight_data.get("bias") {
                concat_bias_data.extend_from_slice(b.data());
            } else {
                return; // Missing bias — skip.
            }
        }
    }

    let total_out: usize = projection_sizes.iter().sum();

    // Validate concatenated weight dimensions.
    if concat_weight_data.len() != total_out * in_features {
        return;
    }
    if has_bias && concat_bias_data.len() != total_out {
        return;
    }

    // CPU transpose: [total_out, in_features] → [in_features, total_out].
    // Eliminates a GPU transpose dispatch in the executor's matmul path.
    let mut transposed = vec![0.0f32; total_out * in_features];
    for r in 0..total_out {
        for c in 0..in_features {
            transposed[c * total_out + r] = concat_weight_data[r * in_features + c];
        }
    }

    let concat_weight_t = WeightRef::new(transposed, vec![in_features, total_out])
        .expect("transposed weight preserves element count");

    let mut batch_weight_data = HashMap::new();
    batch_weight_data.insert("weight_t".to_string(), concat_weight_t);
    if has_bias {
        let concat_bias = WeightRef::new(concat_bias_data, vec![total_out])
            .expect("bias concat preserves element count");
        batch_weight_data.insert("bias".to_string(), concat_bias);
    }

    // Replace first Linear with BatchedLinearProjection NativeOp.
    // The executor does: matmul → optional bias → narrow first projection as
    // step output → stash full output in thread-local for ProjectionSlice steps.
    steps[first_step] = CompiledStep::NativeOp {
        op: NativeOpKind::BatchedLinearProjection {
            in_features,
            total_out_features: total_out,
            projection_sizes: projection_sizes.clone(),
            has_bias,
            input_shape: input_shape.clone(),
        },
        weight_data: batch_weight_data,
    };

    // Replace remaining Linears with ProjectionSlice NativeOps.
    // Each reads the stashed full output from the BatchedLinearProjection step
    // and narrows to extract its projection's slice via GPU dispatch.
    let ndim = input_shape.len();
    let last_dim = if ndim > 0 { ndim - 1 } else { 0 };
    let mut start = projection_sizes[0]; // First projection handled by BatchedLinearProjection.

    for (i, c) in group.iter().enumerate().skip(1) {
        let mut out_shape = input_shape.clone();
        if let Some(last) = out_shape.last_mut() {
            *last = c.info.out_features;
        }
        steps[c.step_idx] = CompiledStep::NativeOp {
            op: NativeOpKind::ProjectionSlice {
                source_step: first_step,
                dim: last_dim,
                start,
                length: projection_sizes[i],
                output_shape: out_shape,
            },
            weight_data: HashMap::new(),
        };
        start += projection_sizes[i];
    }
}

#[cfg(test)]
#[path = "trace_compile_peephole_batched_qkv_tests.rs"]
mod tests;
