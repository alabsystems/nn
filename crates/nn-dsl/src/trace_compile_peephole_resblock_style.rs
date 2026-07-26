// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pass 3: absorb style projection Linears into FusedResBlock steps.
//!
//! Traces gamma/beta inputs back through `Reshape → Narrow → Linear(style)`
//! and absorbs the Linear weights into the FusedResBlock, replacing
//! `input_steps = [x, gamma1, beta1, gamma2, beta2]` with `[x, style]`.
//!
//! Extracted from `trace_compile_peephole_resblock.rs` — Part of #2218.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceOp};

use nn_core::dyn_tensor::trace::WeightRef;

use super::super::super::{CompiledStep, NativeOpKind, StyleProjectionParams};

/// Pass 3: absorb style projection Linears into FusedResBlock steps.
///
/// For each FusedResBlock with `style_proj: None`, traces gamma/beta
/// inputs back through `Reshape → Narrow → Linear(style)`. If both
/// phases trace to the same `style` input, absorbs the Linear weights
/// into the FusedResBlock and replaces `input_steps` with `[x, style]`.
///
/// This eliminates ~2 Dispatch "linear" steps per block (14 total for
/// F0EnergyPredictor), reducing the compiled plan step count. Part of #2780.
pub(crate) fn absorb_style_projections(steps: &mut [CompiledStep], graph: &ComputationGraph) {
    let nodes = graph.nodes();
    let id_to_idx: HashMap<u64, usize> =
        nodes.iter().enumerate().map(|(i, n)| (n.id(), i)).collect();

    // Collect FusedResBlock indices first to avoid borrow issues.
    let rb_indices: Vec<usize> = (0..steps.len())
        .filter(|&idx| {
            matches!(
                &steps[idx],
                CompiledStep::NativeOp {
                    op: NativeOpKind::FusedResBlock {
                        style_proj: None,
                        ..
                    },
                    ..
                }
            )
        })
        .collect();

    for rb_idx in rb_indices {
        try_absorb_style_at(steps, rb_idx, nodes, &id_to_idx);
    }
}

/// Trace a gamma or beta step back through Reshape → Narrow → Linear.
///
/// Returns `Some((linear_idx, style_id, channels, intermediate_step_indices))`
/// if the pattern `Reshape → Narrow(dim=1, start=expect_start) → Linear → style`
/// is matched.
fn trace_gamma_beta_to_linear(
    step_idx: usize,
    nodes: &[nn_core::dyn_tensor::trace::TraceNode],
    id_to_idx: &HashMap<u64, usize>,
    expect_start: usize,
) -> Option<(usize, u64, usize, Vec<usize>)> {
    let node = &nodes[step_idx];

    // Step should be a reshape (Passthrough) with 1 input.
    let reshape_inputs = node.inputs();
    if reshape_inputs.len() != 1 {
        return None;
    }

    // The input to reshape should be a narrow.
    let narrow_idx = *id_to_idx.get(&reshape_inputs[0])?;
    let narrow_node = &nodes[narrow_idx];

    // Check it's a Narrow op on dim=1 with the expected start offset.
    let (narrow_start, narrow_len) = match narrow_node.op() {
        TraceOp::Narrow {
            dim: 1,
            start,
            length,
        } => (*start, *length),
        _ => return None,
    };
    if narrow_start != expect_start {
        return None;
    }

    // The input to narrow should be a Linear.
    let narrow_inputs = narrow_node.inputs();
    if narrow_inputs.len() != 1 {
        return None;
    }
    let linear_idx = *id_to_idx.get(&narrow_inputs[0])?;
    let linear_node = &nodes[linear_idx];

    match linear_node.op() {
        TraceOp::Linear { .. } | TraceOp::QLinear { .. } => {}
        _ => return None,
    }

    // The Linear should have 1 graph input (the style vector).
    let linear_inputs = linear_node.inputs();
    if linear_inputs.len() != 1 {
        return None;
    }
    let style_id = linear_inputs[0];

    Some((linear_idx, style_id, narrow_len, vec![step_idx, narrow_idx]))
}

/// Try to absorb style projection at a FusedResBlock step.
fn try_absorb_style_at(
    steps: &mut [CompiledStep],
    rb_idx: usize,
    nodes: &[nn_core::dyn_tensor::trace::TraceNode],
    id_to_idx: &HashMap<u64, usize>,
) -> bool {
    // Extract current input_steps from the FusedResBlock.
    let (input_steps, _channels1, _channels2) = match &steps[rb_idx] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::FusedResBlock {
                    input_steps,
                    phase1,
                    phase2,
                    style_proj: None,
                    ..
                },
            ..
        } => (
            input_steps.clone(),
            phase1.input_shape[1],
            phase2.input_shape[1],
        ),
        _ => return false,
    };

    if input_steps.len() < 5 {
        return false;
    }

    let gamma1_step = input_steps[1];
    let beta1_step = input_steps[2];
    let gamma2_step = input_steps[3];
    let beta2_step = input_steps[4];

    // Trace gamma1 back: Reshape → Narrow(start=0) → Linear → style
    let (linear1_idx, style1_id, ch1, intermediates1) =
        match trace_gamma_beta_to_linear(gamma1_step, nodes, id_to_idx, 0) {
            Some(r) => r,
            None => return false,
        };

    // Trace beta1 back: Reshape → Narrow(start=ch1) → same Linear → same style
    let (linear1b_idx, style1b_id, _, intermediates1b) =
        match trace_gamma_beta_to_linear(beta1_step, nodes, id_to_idx, ch1) {
            Some(r) => r,
            None => return false,
        };

    // gamma1 and beta1 must trace to the same Linear.
    if linear1_idx != linear1b_idx || style1_id != style1b_id {
        return false;
    }

    // Trace gamma2 back: Reshape → Narrow(start=0) → Linear → style
    let (linear2_idx, style2_id, ch2, intermediates2) =
        match trace_gamma_beta_to_linear(gamma2_step, nodes, id_to_idx, 0) {
            Some(r) => r,
            None => return false,
        };

    // Trace beta2 back: Reshape → Narrow(start=ch2) → same Linear → same style
    let (linear2b_idx, style2b_id, _, intermediates2b) =
        match trace_gamma_beta_to_linear(beta2_step, nodes, id_to_idx, ch2) {
            Some(r) => r,
            None => return false,
        };

    // gamma2 and beta2 must trace to the same Linear.
    if linear2_idx != linear2b_idx || style2_id != style2b_id {
        return false;
    }

    // Both Linears must take the same style input.
    if style1_id != style2_id {
        return false;
    }

    // Extract weight data from the two Linear Dispatch steps.
    let (linear1_weights, linear2_weights) = match (&steps[linear1_idx], &steps[linear2_idx]) {
        (
            CompiledStep::Dispatch {
                weight_data: w1, ..
            },
            CompiledStep::Dispatch {
                weight_data: w2, ..
            },
        ) => (w1.clone(), w2.clone()),
        _ => return false,
    };

    // Extract style_dim from the projection weight shape: [2*channels, style_dim].
    let style_dim = match linear1_weights.get("weight") {
        Some(w) if w.shape().len() == 2 => w.shape()[1],
        _ => return false,
    };

    // Resolve the style step index.
    let style_step = match id_to_idx.get(&style1_id) {
        Some(&idx) => idx,
        None => return false,
    };

    // --- Build updated FusedResBlock ---
    let step = std::mem::replace(&mut steps[rb_idx], CompiledStep::IdentityPassthrough);
    let (op, mut weight_data) = match step {
        CompiledStep::NativeOp { op, weight_data } => (op, weight_data),
        _ => return false,
    };

    let (phase1, phase2, residual_scale, shortcut_step, pool_step) = match op {
        NativeOpKind::FusedResBlock {
            phase1,
            phase2,
            residual_scale,
            shortcut_step,
            pool_step,
            ..
        } => (phase1, phase2, residual_scale, shortcut_step, pool_step),
        _ => return false,
    };

    // Add style projection weights with prefixed names.
    // Pre-transposed `_t` versions eliminate per-forward-pass GPU transpose
    // dispatches in `run_style_projection` (each `Linear::new()` would
    // otherwise dispatch a GPU transpose kernel). Part of #2218.
    if let Some(w) = linear1_weights.get("weight") {
        if let Some(wt) = transpose_2d_weight(w) {
            weight_data.insert("style1_weight_t".to_string(), wt);
        }
        weight_data.insert("style1_weight".to_string(), w.clone());
    }
    if let Some(b) = linear1_weights.get("bias") {
        weight_data.insert("style1_bias".to_string(), b.clone());
    }
    if let Some(w) = linear2_weights.get("weight") {
        if let Some(wt) = transpose_2d_weight(w) {
            weight_data.insert("style2_weight_t".to_string(), wt);
        }
        weight_data.insert("style2_weight".to_string(), w.clone());
    }
    if let Some(b) = linear2_weights.get("bias") {
        weight_data.insert("style2_bias".to_string(), b.clone());
    }

    let new_op = NativeOpKind::FusedResBlock {
        phase1,
        phase2,
        input_steps: vec![input_steps[0], style_step],
        residual_scale,
        style_proj: Some(StyleProjectionParams {
            channels1: ch1,
            channels2: ch2,
            style_dim,
        }),
        shortcut_step,
        pool_step,
        style_batch_offset: None,
    };

    // Place updated FusedResBlock.
    steps[rb_idx] = CompiledStep::NativeOp {
        op: new_op,
        weight_data,
    };

    // Replace absorbed steps with IdentityPassthrough.
    steps[linear1_idx] = CompiledStep::IdentityPassthrough;
    if linear2_idx != linear1_idx {
        steps[linear2_idx] = CompiledStep::IdentityPassthrough;
    }

    // Replace intermediate Narrow/Reshape steps.
    let mut all_intermediates = Vec::new();
    all_intermediates.extend_from_slice(&intermediates1);
    all_intermediates.extend_from_slice(&intermediates1b);
    all_intermediates.extend_from_slice(&intermediates2);
    all_intermediates.extend_from_slice(&intermediates2b);
    all_intermediates.sort_unstable();
    all_intermediates.dedup();
    for idx in all_intermediates {
        steps[idx] = CompiledStep::IdentityPassthrough;
    }

    true
}

/// CPU transpose of a 2D weight `[rows, cols]` → `[cols, rows]`.
///
/// Returns `None` if the weight is not rank-2 or is a shape-only placeholder.
/// The result is stored as a pre-transposed `WeightRef` so the executor can
/// skip the per-forward-pass GPU transpose dispatch in `run_style_projection`.
fn transpose_2d_weight(w: &WeightRef) -> Option<WeightRef> {
    if w.shape().len() != 2 || w.is_placeholder() {
        return None;
    }
    let rows = w.shape()[0];
    let cols = w.shape()[1];
    let data = w.data();
    let mut out = vec![0.0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    // Element count is preserved; WeightRef::new() cannot fail here.
    Some(WeightRef::new(out, vec![cols, rows]).expect("transpose preserves element count"))
}
