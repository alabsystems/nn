// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pass 4: batch style projections across FusedResBlocks in a segment.
//!
//! After Pass 3 absorbs per-block style Linears into FusedResBlocks, this
//! pass groups FusedResBlocks by their shared style embedding input and
//! concatenates the per-block projection weights into a single batched
//! `[total_out, style_dim]` matmul + bias_add. Each FusedResBlock then
//! narrows its gamma/beta from the batched output (zero-copy).
//!
//! Saves ~136 Metal dispatches for Kokoro (35 blocks × 4 → 2 per segment).
//! Part of #1815 Tier 1, #2964.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::WeightRef;

use super::super::super::{CompiledStep, NativeOpKind, StyleBatchOffset};

/// Candidate FusedResBlock with an absorbed style projection.
struct BatchCandidate {
    rb_idx: usize,
    style_step: usize,
    channels1: usize,
    channels2: usize,
    style_dim: usize,
}

/// Pass 4: batch style projections across FusedResBlocks sharing a style input.
///
/// For each group of FusedResBlocks that share the same `input_steps[1]`
/// (style embedding), concatenates their per-block projection weights into
/// a single `BatchedStyleProjection` NativeOp. Each FusedResBlock then
/// narrows its gamma/beta from the batched output via `style_batch_offset`.
pub(crate) fn batch_style_projections(steps: &mut [CompiledStep]) {
    let candidates: Vec<BatchCandidate> = steps
        .iter()
        .enumerate()
        .filter_map(|(idx, step)| match step {
            CompiledStep::NativeOp {
                op:
                    NativeOpKind::FusedResBlock {
                        input_steps,
                        style_proj: Some(sp),
                        ..
                    },
                ..
            } if input_steps.len() >= 2 => Some(BatchCandidate {
                rb_idx: idx,
                style_step: input_steps[1],
                channels1: sp.channels1,
                channels2: sp.channels2,
                style_dim: sp.style_dim,
            }),
            _ => None,
        })
        .collect();

    if candidates.is_empty() {
        return;
    }

    // Group by style_step (one group per segment).
    let mut groups: HashMap<usize, Vec<BatchCandidate>> = HashMap::new();
    for c in candidates {
        groups.entry(c.style_step).or_default().push(c);
    }

    for group in groups.into_values() {
        if group.len() < 2 {
            continue;
        }
        batch_group(steps, &group);
    }
}

/// Batch a group of FusedResBlocks sharing the same style embedding.
fn batch_group(steps: &mut [CompiledStep], group: &[BatchCandidate]) {
    let first_rb = group[0].rb_idx;
    let style_step = group[0].style_step;
    let style_dim = group[0].style_dim;

    // Find an IdentityPassthrough slot between style_step and the first
    // FusedResBlock. Must be after style_step so the embedding is computed.
    let slot = match find_passthrough_slot(steps, style_step, first_rb) {
        Some(s) => s,
        None => return,
    };

    // Compute per-block offsets and concatenate weights.
    let mut offset = 0usize;
    let mut block_offsets = Vec::with_capacity(group.len());
    let mut concat_weight_data: Vec<f32> = Vec::new();
    let mut concat_bias_data: Vec<f32> = Vec::new();

    for c in group {
        // Per-block layout: [style1(2*C1), style2(2*C2)].
        block_offsets.push(StyleBatchOffset::new(offset, c.channels1, c.channels2));

        // Extract and concatenate weight data from the FusedResBlock.
        if let CompiledStep::NativeOp { weight_data, .. } = &steps[c.rb_idx] {
            // Weight concat order: style1_weight [2*C1, style_dim] then
            // style2_weight [2*C2, style_dim]. Row-major → just append.
            if let Some(w) = weight_data.get("style1_weight") {
                concat_weight_data.extend_from_slice(w.data());
            }
            if let Some(w) = weight_data.get("style2_weight") {
                concat_weight_data.extend_from_slice(w.data());
            }
            if let Some(b) = weight_data.get("style1_bias") {
                concat_bias_data.extend_from_slice(b.data());
            }
            if let Some(b) = weight_data.get("style2_bias") {
                concat_bias_data.extend_from_slice(b.data());
            }
        }

        offset += 2 * (c.channels1 + c.channels2);
    }

    let total_out = offset;

    // Validate concatenated weight dimensions.
    let expected_weight_len = total_out * style_dim;
    if concat_weight_data.len() != expected_weight_len || concat_bias_data.len() != total_out {
        return; // Shape mismatch — skip batching for safety.
    }

    // CPU transpose: [total_out, style_dim] → [style_dim, total_out].
    // Eliminates a GPU transpose dispatch in the executor's matmul path.
    let mut transposed = vec![0.0f32; expected_weight_len];
    for r in 0..total_out {
        for c in 0..style_dim {
            transposed[c * total_out + r] = concat_weight_data[r * style_dim + c];
        }
    }

    let concat_weight = WeightRef::new(concat_weight_data, vec![total_out, style_dim])
        .expect("weight concat preserves element count");
    let concat_weight_t = WeightRef::new(transposed, vec![style_dim, total_out])
        .expect("transposed weight preserves element count");
    let concat_bias = WeightRef::new(concat_bias_data, vec![total_out])
        .expect("bias concat preserves element count");

    let mut batch_weight_data = HashMap::new();
    batch_weight_data.insert("weight".to_string(), concat_weight);
    batch_weight_data.insert("weight_t".to_string(), concat_weight_t);
    batch_weight_data.insert("bias".to_string(), concat_bias);

    // Place BatchedStyleProjection at the IdentityPassthrough slot.
    steps[slot] = CompiledStep::NativeOp {
        op: NativeOpKind::BatchedStyleProjection {
            blocks: block_offsets.clone(),
            style_dim,
            total_out,
            style_step,
        },
        weight_data: batch_weight_data,
    };

    // Update each FusedResBlock to use the batched path.
    for (i, c) in group.iter().enumerate() {
        let step = std::mem::replace(&mut steps[c.rb_idx], CompiledStep::IdentityPassthrough);

        if let CompiledStep::NativeOp {
            op:
                NativeOpKind::FusedResBlock {
                    phase1,
                    phase2,
                    input_steps,
                    residual_scale,
                    shortcut_step,
                    pool_step,
                    ..
                },
            mut weight_data,
        } = step
        {
            // Remove per-block style weights (now in the batched step).
            for key in &[
                "style1_weight",
                "style1_bias",
                "style1_weight_t",
                "style2_weight",
                "style2_bias",
                "style2_weight_t",
            ] {
                weight_data.remove(*key);
            }

            steps[c.rb_idx] = CompiledStep::NativeOp {
                op: NativeOpKind::FusedResBlock {
                    phase1,
                    phase2,
                    input_steps: vec![input_steps[0], slot],
                    residual_scale,
                    style_proj: None,
                    shortcut_step,
                    pool_step,
                    style_batch_offset: Some(block_offsets[i].clone()),
                },
                weight_data,
            };
        }
    }
}

/// Find an `IdentityPassthrough` slot between `after_idx` (exclusive) and
/// `before_idx` (exclusive), scanning backwards from `before_idx`.
fn find_passthrough_slot(
    steps: &[CompiledStep],
    after_idx: usize,
    before_idx: usize,
) -> Option<usize> {
    (after_idx + 1..before_idx)
        .rev()
        .find(|&i| matches!(steps[i], CompiledStep::IdentityPassthrough))
}
