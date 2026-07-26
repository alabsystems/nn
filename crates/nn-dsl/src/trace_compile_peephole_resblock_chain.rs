// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Peephole pass: chain nearby FusedResBlocks into FusedResBlockChain.
//!
//! Scans for runs of 2+ `NativeOpKind::FusedResBlock` steps (skipping
//! IdentityPassthrough gaps left by earlier peephole passes) that:
//! 1. Use the `style_batch_offset` path (batched style projection, not per-block)
//! 2. Share the same style projection step (same `input_steps[1]`)
//! 3. Use the same activation type across all phases (all Snake or all LeakyRelu)
//! 4. Have no `pool_step` (not upsample ResBlocks)
//! 5. Only the first block may have a `shortcut_step`
//!
//! When a run of N eligible blocks is found, they are replaced with a single
//! `FusedResBlockChain` NativeOp. The weights from each original block are
//! re-keyed as `block{i}_p{j}_*` in the chain's weight_data.
//!
//! Part of #4264.

use std::collections::HashMap;

use super::{CompiledStep, NativeOpKind};
use crate::{NormActivation, ResBlockChainEntry};

/// Minimum number of consecutive FusedResBlocks to form a chain.
const MIN_CHAIN_LENGTH: usize = 2;

/// Maximum number of FusedResBlocks in a single chain.
const MAX_CHAIN_LENGTH: usize = 4;

/// Scan for runs of FusedResBlock steps (skipping IdentityPassthrough gaps)
/// and replace with FusedResBlockChain.
///
/// After peephole passes 2-4, absorbed Linear/Narrow/Reshape steps become
/// IdentityPassthrough. The FusedResBlocks for different dilation layers
/// within one ResBlock are separated by these identity gaps. We skip them
/// when scanning to find chainable runs.
pub(crate) fn fuse_resblock_chain(steps: &mut [CompiledStep]) {
    let mut i = 0;
    while i < steps.len() {
        // Find the first chainable FusedResBlock.
        if !is_chainable_resblock(&steps[i], true) {
            i += 1;
            continue;
        }

        // Collect indices of chainable FusedResBlocks in this run.
        let mut run_indices: Vec<usize> = vec![i];
        let mut scan = i + 1;

        while scan < steps.len() && run_indices.len() < MAX_CHAIN_LENGTH {
            match &steps[scan] {
                CompiledStep::IdentityPassthrough => {
                    // Skip identity gaps (absorbed Linear/Narrow/Reshape steps).
                    scan += 1;
                    continue;
                }
                _ if is_chainable_resblock(&steps[scan], false)
                    // Check compatibility with the first block in the run.
                    && is_compatible_with_run_at(steps, run_indices[0], scan) => {
                        run_indices.push(scan);
                        scan += 1;
                        continue;
                    }
                _ => {}
            }
            break;
        }

        if run_indices.len() >= MIN_CHAIN_LENGTH {
            merge_run_sparse(steps, &run_indices);
            // Skip past all processed steps.
            i = *run_indices.last().unwrap() + 1;
        } else {
            i += 1;
        }
    }
}

/// Check if a step is a FusedResBlock eligible for chaining.
fn is_chainable_resblock(step: &CompiledStep, is_first: bool) -> bool {
    if let CompiledStep::NativeOp {
        op:
            NativeOpKind::FusedResBlock {
                style_batch_offset: Some(_),
                pool_step: None,
                style_proj: None,
                shortcut_step,
                phase1,
                phase2,
                ..
            },
        ..
    } = step
    {
        // Only the first block in a chain may have a shortcut step.
        if !is_first && shortcut_step.is_some() {
            return false;
        }
        // Both phases must use the same activation type.
        matches!(
            (&phase1.activation, &phase2.activation),
            (NormActivation::Snake, NormActivation::Snake)
                | (
                    NormActivation::LeakyRelu { .. },
                    NormActivation::LeakyRelu { .. }
                )
        )
    } else {
        false
    }
}

/// Check if a candidate step at `cand_idx` is compatible with the first
/// block at `first_idx` (same style step, same activation).
fn is_compatible_with_run_at(steps: &[CompiledStep], first_idx: usize, cand_idx: usize) -> bool {
    let (first_style_step, first_activation) = match &steps[first_idx] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::FusedResBlock {
                    input_steps,
                    phase1,
                    ..
                },
            ..
        } => {
            let style_step = if input_steps.len() >= 2 {
                input_steps[1]
            } else {
                return false;
            };
            (style_step, phase1.activation.clone())
        }
        _ => return false,
    };

    if let CompiledStep::NativeOp {
        op:
            NativeOpKind::FusedResBlock {
                input_steps,
                phase1,
                ..
            },
        ..
    } = &steps[cand_idx]
    {
        let cand_style_step = if input_steps.len() >= 2 {
            input_steps[1]
        } else {
            return false;
        };
        // Same style step and same activation type.
        cand_style_step == first_style_step
            && std::mem::discriminant(&phase1.activation)
                == std::mem::discriminant(&first_activation)
    } else {
        false
    }
}

/// Merge a sparse run of FusedResBlocks (at non-contiguous indices) into a
/// single FusedResBlockChain. The first index gets the chain; all other
/// indices (and any IdentityPassthrough gaps between them) stay as-is
/// (the gaps were already IdentityPassthrough from earlier passes).
fn merge_run_sparse(steps: &mut [CompiledStep], indices: &[usize]) {
    let count = indices.len();
    let mut blocks = Vec::with_capacity(count);
    let mut style_batch_offsets = Vec::with_capacity(count);
    let mut chain_input_steps = Vec::new();
    let mut first_shortcut = None;
    let mut merged_weights: HashMap<String, nn_core::dyn_tensor::trace::WeightRef> =
        HashMap::new();

    for (block_idx, &step_idx) in indices.iter().enumerate() {
        if let CompiledStep::NativeOp {
            op:
                NativeOpKind::FusedResBlock {
                    phase1,
                    phase2,
                    input_steps,
                    residual_scale,
                    shortcut_step,
                    style_batch_offset,
                    ..
                },
            weight_data,
        } = &steps[step_idx]
        {
            blocks.push(ResBlockChainEntry::new(
                phase1.clone(),
                phase2.clone(),
                *residual_scale,
            ));

            if let Some(ref sbo) = style_batch_offset {
                style_batch_offsets.push(sbo.clone());
            }

            if block_idx == 0 {
                chain_input_steps = input_steps.clone();
                first_shortcut = *shortcut_step;
            }

            // Re-key weights for the chain: block{i}_p{j}_*.
            for (key, wref) in weight_data {
                let new_key = remap_weight_key(key, block_idx);
                merged_weights.insert(new_key, wref.clone());
            }
        }
    }

    // Build the chain NativeOp.
    let chain_op = NativeOpKind::FusedResBlockChain {
        blocks,
        input_steps: chain_input_steps,
        style_batch_offsets,
        first_shortcut_step: first_shortcut,
    };

    // Replace the first FusedResBlock with the chain.
    steps[indices[0]] = CompiledStep::NativeOp {
        op: chain_op,
        weight_data: merged_weights,
    };

    // Replace remaining FusedResBlock steps with IdentityPassthrough.
    for &idx in &indices[1..] {
        steps[idx] = CompiledStep::IdentityPassthrough;
    }
}

/// Remap a weight key from the original block to the chain namespace.
///
/// Original keys: `p1_alpha`, `p1_conv_weight`, `p2_conv_bias`, etc.
/// Remapped:      `block0_p1_alpha`, `block0_p1_conv_weight`, etc.
fn remap_weight_key(key: &str, block_idx: usize) -> String {
    format!("block{block_idx}_{key}")
}
