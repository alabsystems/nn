// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ResBlock-level peephole fusion for compiled step sequences.
//!
//! Detects the 2× NormActivConv1d + residual add pattern produced after
//! the NormActivConv1d peephole and fuses into a single
//! `NativeOpKind::FusedResBlock` NativeOp.
//!
//! Uses **graph-topology-based** detection instead of consecutive-step
//! matching, so it works for both:
//!   - Generator ResBlocks (consecutive steps, no style projections)
//!   - F0 ResBlocks (non-consecutive, with intervening style projections)
//!
//! Optionally absorbs a post-add `ConstantValue + Dispatch "mul"` into
//! the `residual_scale` field (for F0's `/ sqrt(2)` pattern).
//!
//! Runs as a second pass after `fuse_norm_activ_conv1d()`. Part of #2218.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceOp};

use super::super::{CompiledStep, NativeOpKind, NormActivConv1dParams, NormActivation};

/// Scan for the ResBlock pattern via graph topology and fuse into FusedResBlock.
///
/// For each `Dispatch "add"` step, traces back through the computation graph
/// to find two NormActivConv1d phases feeding into it with a shared residual
/// input. Works regardless of whether the steps are consecutive.
pub(crate) fn fuse_resblock(
    steps: &mut [CompiledStep],
    graph: &ComputationGraph,
    use_counts: &[usize],
) {
    let len = steps.len();
    if len < 5 {
        return;
    }

    let nodes = graph.nodes();
    let id_to_idx: HashMap<u64, usize> =
        nodes.iter().enumerate().map(|(i, n)| (n.id(), i)).collect();

    // Build consumers map: node_id → list of consumer step indices.
    // Used for detecting optional post-add mul_scalar pattern.
    let mut consumers: HashMap<u64, Vec<usize>> = HashMap::new();
    for (idx, node) in nodes.iter().enumerate() {
        for &input_id in node.inputs() {
            consumers.entry(input_id).or_default().push(idx);
        }
    }

    // Scan all steps for Dispatch "add" candidates.
    // Process from the end so that replacements don't interfere with
    // earlier candidates (each add is independent in the graph).
    let mut add_candidates: Vec<usize> = (0..len)
        .filter(|&idx| {
            matches!(
                &steps[idx],
                CompiledStep::Dispatch { kernel, .. } if kernel.name() == "add"
            )
        })
        .collect();
    // Reverse so we process later indices first (safe for in-place mutation).
    add_candidates.reverse();

    for add_idx in add_candidates {
        try_fuse_at_add(steps, add_idx, nodes, &id_to_idx, use_counts, &consumers);
    }
}

/// Result of successfully tracing the two-phase NormActivConv1d chain.
struct ConvChainResult {
    x_id: u64,
    phase1_params: NormActivConv1dParams,
    phase1_weights: HashMap<String, nn_core::dyn_tensor::trace::WeightRef>,
    phase2_params: NormActivConv1dParams,
    phase2_weights: HashMap<String, nn_core::dyn_tensor::trace::WeightRef>,
    adain1_idx: usize,
    conv1_idx: usize,
    adain2_idx: usize,
    conv2_idx: usize,
    /// adain1 graph inputs: [x_id, gamma1_id, beta1_id]
    adain1_inputs: Vec<u64>,
    /// adain2 graph inputs: [conv1_out_id, gamma2_id, beta2_id]
    adain2_inputs: Vec<u64>,
    /// Step index of a conv1x1 shortcut on the residual side.
    /// `Some(idx)` when the `x_candidate_id` is a Conv1d(k=1) applied to adain1's input.
    shortcut_step: Option<usize>,
    /// Step index of a pool (ConvTranspose1d) between adain1 and conv1,
    /// for upsample ResBlocks. `None` for standard fused phase1.
    /// When `Some`, adain1 and pool remain as live steps (NOT replaced).
    pool_step: Option<usize>,
}

/// Try to trace back from `conv2_candidate_id` through the two-phase
/// NormActivConv1d chain, verifying that adain1's first input matches `x_candidate_id`.
///
/// Returns `Some(ConvChainResult)` if the full chain is found, `None` otherwise.
fn trace_conv_chain(
    conv2_candidate_id: u64,
    x_candidate_id: u64,
    steps: &[CompiledStep],
    nodes: &[nn_core::dyn_tensor::trace::TraceNode],
    id_to_idx: &HashMap<u64, usize>,
    use_counts: &[usize],
) -> Option<ConvChainResult> {
    let conv2_idx = *id_to_idx.get(&conv2_candidate_id)?;

    // conv2_idx must be IdentityPassthrough (was conv2 position after Pass 1).
    if !matches!(&steps[conv2_idx], CompiledStep::IdentityPassthrough) {
        return None;
    }
    // Fan-out: conv2 passthrough feeds only into add.
    if use_counts.get(conv2_idx).copied().unwrap_or(0) != 1 {
        return None;
    }

    // conv2 node → single input → adain2 (NormActivConv1d, second phase).
    let conv2_inputs = nodes[conv2_idx].inputs();
    if conv2_inputs.len() != 1 {
        return None;
    }
    let adain2_idx = *id_to_idx.get(&conv2_inputs[0])?;

    // Extract phase 2 NormActivConv1d params.
    let (phase2_params, phase2_weights) = extract_norm_activ_params(&steps[adain2_idx])?;

    // adain2 node inputs: [conv1_out_id, gamma2_id, beta2_id]
    let adain2_inputs = nodes[adain2_idx].inputs();
    if adain2_inputs.len() < 3 {
        return None;
    }

    let conv1_idx = *id_to_idx.get(&adain2_inputs[0])?;

    // --- Standard fused path: conv1_idx is IdentityPassthrough (Pass 1 fused) ---
    if matches!(&steps[conv1_idx], CompiledStep::IdentityPassthrough) {
        // Fan-out: conv1 passthrough feeds only into second phase.
        if use_counts.get(conv1_idx).copied().unwrap_or(0) != 1 {
            return None;
        }

        // conv1 node → single input → adain1 (NormActivConv1d, first phase).
        let conv1_inputs = nodes[conv1_idx].inputs();
        if conv1_inputs.len() != 1 {
            return None;
        }
        let adain1_idx = *id_to_idx.get(&conv1_inputs[0])?;

        // Extract phase 1 NormActivConv1d params.
        let (phase1_params, phase1_weights) = extract_norm_activ_params(&steps[adain1_idx])?;

        // adain1 node inputs: [x_id_check, gamma1_id, beta1_id]
        let adain1_inputs = nodes[adain1_idx].inputs();
        if adain1_inputs.len() < 3 {
            return None;
        }

        let shortcut_step = if adain1_inputs[0] == x_candidate_id {
            None
        } else {
            detect_conv1x1_shortcut(x_candidate_id, adain1_inputs[0], steps, nodes, id_to_idx)?
        };

        return Some(ConvChainResult {
            x_id: adain1_inputs[0],
            phase1_params,
            phase1_weights,
            phase2_params,
            phase2_weights,
            adain1_idx,
            conv1_idx,
            adain2_idx,
            conv2_idx,
            adain1_inputs: adain1_inputs.to_vec(),
            adain2_inputs: adain2_inputs.to_vec(),
            shortcut_step,
            pool_step: None,
        });
    }

    // --- Unfused path: conv1 is a standalone Dispatch "conv1d" (#3510) ---
    // For upsample ResBlocks, Pass 1 could not fuse adain1+conv1 because a
    // ConvTranspose1d (pool) sits between them. The pattern is:
    //   adain1 → pool → conv1 → adain2 → conv2
    // We detect: conv1 is Dispatch "conv1d", its input chains through exactly
    // one intermediate (pool) back to a standalone AdainLeakyRelu/AdainSnake.
    trace_unfused_phase1(
        conv1_idx,
        x_candidate_id,
        steps,
        nodes,
        id_to_idx,
        use_counts,
        phase2_params,
        phase2_weights,
        adain2_idx,
        conv2_idx,
        adain2_inputs,
    )
}

/// Trace unfused phase1 for upsample ResBlocks (#3510).
///
/// Called when `conv1_idx` is a standalone `Dispatch "conv1d"` (not fused by Pass 1).
/// Looks for the pattern: adain1 → pool → conv1, where pool is exactly one
/// intermediate step (ConvTranspose1d or similar spatial op).
///
/// Builds phase1 params from the separate AdaIN step + Conv1d step and records
/// `pool_step` so the executor knows to split phase1.
#[allow(clippy::too_many_arguments)]
fn trace_unfused_phase1(
    conv1_idx: usize,
    x_candidate_id: u64,
    steps: &[CompiledStep],
    nodes: &[nn_core::dyn_tensor::trace::TraceNode],
    id_to_idx: &HashMap<u64, usize>,
    use_counts: &[usize],
    phase2_params: NormActivConv1dParams,
    phase2_weights: HashMap<String, nn_core::dyn_tensor::trace::WeightRef>,
    adain2_idx: usize,
    conv2_idx: usize,
    adain2_inputs: &[u64],
) -> Option<ConvChainResult> {
    // conv1 must be a Dispatch "conv1d".
    let conv_info = match &steps[conv1_idx] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } if kernel.name() == "conv1d" => super::extract_conv1d_params(kernel, weight_data),
        _ => None,
    }?;

    // Only fuse stride=1, groups=1 Conv1d.
    if conv_info.stride != 1 || conv_info.groups != 1 {
        return None;
    }

    // Fan-out: conv1 feeds only into adain2.
    if use_counts.get(conv1_idx).copied().unwrap_or(0) != 1 {
        return None;
    }

    // conv1 node → single input → pool step (intermediate).
    let conv1_inputs = nodes[conv1_idx].inputs();
    if conv1_inputs.len() != 1 {
        return None;
    }
    let pool_idx = *id_to_idx.get(&conv1_inputs[0])?;

    // Fan-out: pool feeds only into conv1.
    if use_counts.get(pool_idx).copied().unwrap_or(0) != 1 {
        return None;
    }

    // pool node → single input → adain1 step.
    let pool_inputs = nodes[pool_idx].inputs();
    if pool_inputs.len() != 1 {
        return None;
    }
    let adain1_idx = *id_to_idx.get(&pool_inputs[0])?;

    // adain1 must be a standalone AdainLeakyRelu or AdainSnake.
    let (activation, eps, input_shape, adain_weight_data) =
        extract_standalone_adain_params(&steps[adain1_idx])?;

    // adain1 node inputs: [x_id, gamma1_id, beta1_id]
    let adain1_inputs = nodes[adain1_idx].inputs();
    if adain1_inputs.len() < 3 {
        return None;
    }

    // Check residual connection: adain1's x must match x_candidate.
    let shortcut_step = if adain1_inputs[0] == x_candidate_id {
        None
    } else {
        detect_conv1x1_shortcut(x_candidate_id, adain1_inputs[0], steps, nodes, id_to_idx)?
    };

    // Build phase1 NormActivConv1dParams by combining AdaIN + Conv1d info.
    let phase1_params = NormActivConv1dParams {
        activation,
        eps,
        conv_dilation: conv_info.dilation,
        conv_padding: conv_info.padding,
        input_shape,
        output_channels: conv_info.output_channels,
        kernel_size: conv_info.kernel_size,
    };

    // Build phase1 weight_data: AdaIN weights + conv weights with conv_ prefix.
    let mut phase1_weights = adain_weight_data;
    if let Some(w) = conv_info.weight {
        phase1_weights.insert("conv_weight".to_string(), w);
    }
    if let Some(b) = conv_info.bias {
        phase1_weights.insert("conv_bias".to_string(), b);
    }

    Some(ConvChainResult {
        x_id: adain1_inputs[0],
        phase1_params,
        phase1_weights,
        phase2_params,
        phase2_weights,
        adain1_idx,
        conv1_idx,
        adain2_idx,
        conv2_idx,
        adain1_inputs: adain1_inputs.to_vec(),
        adain2_inputs: adain2_inputs.to_vec(),
        shortcut_step,
        pool_step: Some(pool_idx),
    })
}

/// Detect whether `x_candidate_id` is a Conv1d(kernel_size=1) applied to `adain1_x_id`.
///
/// Returns `Some(Some(step_idx))` if a conv1x1 shortcut is found (the step index
/// of the conv1x1 output in the compiled plan). Returns `None` if the pattern
/// doesn't match (signals that `trace_conv_chain` should return `None`).
#[allow(clippy::option_option)]
fn detect_conv1x1_shortcut(
    x_candidate_id: u64,
    adain1_x_id: u64,
    steps: &[CompiledStep],
    nodes: &[nn_core::dyn_tensor::trace::TraceNode],
    id_to_idx: &HashMap<u64, usize>,
) -> Option<Option<usize>> {
    let shortcut_idx = *id_to_idx.get(&x_candidate_id)?;

    // The shortcut step must be a Dispatch (Conv1d compiles to a Dispatch step).
    if !matches!(&steps[shortcut_idx], CompiledStep::Dispatch { kernel, .. } if kernel.name() == "conv1d")
    {
        return None;
    }

    // The shortcut node's graph op must be Conv1d with kernel_size=1.
    match nodes[shortcut_idx].op() {
        TraceOp::Conv1d {
            ref weight,
            stride: 1,
            dilation: 1,
            groups: 1,
            ..
        } if weight.shape().last() == Some(&1) => {}
        _ => return None,
    }

    // The conv1x1's first graph input (data) must be the same node that feeds phase 1.
    // Conv1d nodes have inputs [data, kernel] — kernel is the weight parameter.
    let shortcut_inputs = nodes[shortcut_idx].inputs();
    if shortcut_inputs.is_empty() || shortcut_inputs[0] != adain1_x_id {
        return None;
    }

    Some(Some(shortcut_idx))
}

/// Try to fuse a ResBlock pattern anchored at a `Dispatch "add"` step.
///
/// Traces back through graph topology to find the two-phase NormActivConv1d
/// pattern, then optionally looks forward for a post-add mul_scalar.
fn try_fuse_at_add(
    steps: &mut [CompiledStep],
    add_idx: usize,
    nodes: &[nn_core::dyn_tensor::trace::TraceNode],
    id_to_idx: &HashMap<u64, usize>,
    use_counts: &[usize],
    consumers: &HashMap<u64, Vec<usize>>,
) -> bool {
    // --- Verify step at add_idx is Dispatch "add" ---
    if !matches!(
        &steps[add_idx],
        CompiledStep::Dispatch { kernel, .. } if kernel.name() == "add"
    ) {
        return false;
    }

    // --- Trace back through graph topology ---
    // add node inputs: [a, b] — one is x (residual), other is conv2 output.
    // Try both orderings since `x.add(&h)` vs `h.add(&x)` produce different input order.
    let add_inputs = nodes[add_idx].inputs();
    if add_inputs.len() != 2 {
        return false;
    }

    // Try input[1] as conv2 chain first, then input[0].
    let chain = trace_conv_chain(
        add_inputs[1],
        add_inputs[0],
        steps,
        nodes,
        id_to_idx,
        use_counts,
    )
    .or_else(|| {
        trace_conv_chain(
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

    // Both phases must use the same activation family.
    let same_family = matches!(
        (
            &chain.phase1_params.activation,
            &chain.phase2_params.activation
        ),
        (NormActivation::Snake, NormActivation::Snake)
            | (
                NormActivation::LeakyRelu { .. },
                NormActivation::LeakyRelu { .. }
            )
    );
    if !same_family {
        return false;
    }

    // --- Resolve input step indices from graph topology ---
    let x_step = match id_to_idx.get(&chain.x_id) {
        Some(&idx) => idx,
        None => return false,
    };
    let gamma1_step = match id_to_idx.get(&chain.adain1_inputs[1]) {
        Some(&idx) => idx,
        None => return false,
    };
    let beta1_step = match id_to_idx.get(&chain.adain1_inputs[2]) {
        Some(&idx) => idx,
        None => return false,
    };
    let gamma2_step = match id_to_idx.get(&chain.adain2_inputs[1]) {
        Some(&idx) => idx,
        None => return false,
    };
    let beta2_step = match id_to_idx.get(&chain.adain2_inputs[2]) {
        Some(&idx) => idx,
        None => return false,
    };

    // --- Validate input_steps are before the fused position ---
    // The FusedResBlock reads directly from buffers[input_step]. All inputs
    // must be steps that have already executed (i.e., come before the fusion
    // position in the step sequence).
    // Also verify none of them are steps we're about to replace.
    //
    // For upsample blocks (pool_step is Some), adain1 and pool remain live —
    // only conv1, adain2, and conv2 are replaced.
    let replaced_steps: Vec<usize> = if chain.pool_step.is_some() {
        vec![chain.conv1_idx, chain.adain2_idx, chain.conv2_idx]
    } else {
        vec![
            chain.adain1_idx,
            chain.conv1_idx,
            chain.adain2_idx,
            chain.conv2_idx,
        ]
    };
    for &s in &[x_step, gamma1_step, beta1_step, gamma2_step, beta2_step] {
        if replaced_steps.contains(&s) {
            // An input_step references a step we're replacing with IP — the
            // IP won't produce the right output. Bail out.
            return false;
        }
    }
    // Shortcut step (conv1x1) must not collide with replaced steps.
    if let Some(sc) = chain.shortcut_step {
        if replaced_steps.contains(&sc) {
            return false;
        }
    }
    // Pool step must not collide with replaced steps.
    if let Some(ps) = chain.pool_step {
        if replaced_steps.contains(&ps) {
            return false;
        }
    }

    // --- Check for optional post-add residual scale ---
    // Pattern: add → ConstantValue + Dispatch "mul" (e.g., * 1/√2 in F0).
    let (fused_position, residual_scale, extra_replace) =
        detect_post_add_scale(add_idx, steps, nodes, id_to_idx, consumers, use_counts);

    // --- Merge weight data with phase prefixes ---
    let mut merged_weights = HashMap::new();
    for (k, v) in &chain.phase1_weights {
        merged_weights.insert(format!("p1_{k}"), v.clone());
    }
    for (k, v) in &chain.phase2_weights {
        merged_weights.insert(format!("p2_{k}"), v.clone());
    }

    let fused_op = NativeOpKind::FusedResBlock {
        phase1: chain.phase1_params,
        phase2: chain.phase2_params,
        input_steps: vec![x_step, gamma1_step, beta1_step, gamma2_step, beta2_step],
        residual_scale,
        style_proj: None,
        shortcut_step: chain.shortcut_step,
        pool_step: chain.pool_step,
        style_batch_offset: None,
    };

    // Replace absorbed steps with IdentityPassthrough.
    // For upsample blocks (pool_step is Some), adain1 and pool remain live —
    // they execute before FusedResBlock and produce buffers the executor reads.
    if chain.pool_step.is_none() {
        steps[chain.adain1_idx] = CompiledStep::IdentityPassthrough;
    }
    steps[chain.conv1_idx] = CompiledStep::IdentityPassthrough;
    steps[chain.adain2_idx] = CompiledStep::IdentityPassthrough;
    steps[chain.conv2_idx] = CompiledStep::IdentityPassthrough;

    // Replace extra steps (ConstantValue + mul) if absorbed.
    for idx in &extra_replace {
        steps[*idx] = CompiledStep::IdentityPassthrough;
    }

    // Place FusedResBlock at the outermost fused position.
    steps[fused_position] = CompiledStep::NativeOp {
        op: fused_op,
        weight_data: merged_weights,
    };

    true
}

#[path = "trace_compile_peephole_resblock_style.rs"]
mod style_absorption;
pub(crate) use style_absorption::absorb_style_projections;

#[path = "trace_compile_peephole_resblock_batch.rs"]
mod style_batching;
pub(crate) use style_batching::batch_style_projections;

#[path = "trace_compile_peephole_resblock_helpers.rs"]
mod helpers;
use helpers::{detect_post_add_scale, extract_norm_activ_params, extract_standalone_adain_params};

#[cfg(test)]
#[path = "trace_compile_peephole_resblock_tests.rs"]
mod tests;
