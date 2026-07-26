// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MoeGating execution for `CompiledModel`.
//!
//! Implements `NativeOpKind::MoeGating`: softmax over expert logits, top-k
//! selection, weighted routing through experts, scatter-add back.
//!
//! The MoeGating NativeOp decomposes into existing DynTensor operations:
//! - Gate routing: softmax -> topk -> sum_keepdim -> div (renormalize)
//! - Per-expert: index_select -> expert FFN -> broadcast_mul -> scatter-add
//!
//! Currently implemented as a **decomposed CPU-bridge path** using DynTensor
//! ops. When the step carries router weights (`gate_weight`) and expert
//! weights (`expert_{i}_gate_proj`, etc.), the full MoE forward is executed.
//! When no weights are present (pure gating marker), the input is passed
//! through as identity -- matching the NY verification behavior.
//!
//! TODO(#4287): Fused single-dispatch Metal kernel for MoE gating + expert
//! routing when all experts share the same architecture (common in LLMs).
//!
//! Part of #4287.

use nn_core::Result;

use crate::gpu_slice::GpuSlice;

use super::super::helpers::{dyn_to_slice, native_dispatch_err, slice_to_dyn, weight_to_dyn};
use super::super::CompiledModel;

/// Execute a `NativeOpKind::MoeGating` step.
///
/// Decomposed execution path: resolves the hidden-state input from the
/// preceding step, performs softmax -> topk routing via DynTensor ops.
///
/// When the step has no weights (current default from the peephole pass),
/// the input is passed through unchanged. This matches the buffer planner's
/// assumption that MoeGating output shape == input shape, and the NY
/// verification model (identity passthrough for conservative bounds).
///
/// When weights are wired (future: gate_weight, expert weights), the full
/// MoE routing is executed via DynTensor decomposition.
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_native_moe_gating(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    num_experts: usize,
    top_k: usize,
    input_shape: &[usize],
) -> Result<GpuSlice> {
    let dtype = model.step_dtype(step_idx);
    let step_weights = &model.def.weight_buffers[step_idx];

    // Resolve the hidden-state input (edge 0).
    let input_slice = model.resolve_input_slice(step_idx, 0, buffers)?;

    // Check if the step carries gate/expert weights for full MoE execution.
    let has_gate_weight = step_weights.contains_key("gate_weight");

    if has_gate_weight {
        // Full MoE routing path: gate linear -> softmax -> topk -> scatter/gather.
        execute_moe_with_weights(
            model,
            step_idx,
            &input_slice,
            dtype,
            num_experts,
            top_k,
            input_shape,
        )
    } else {
        // No weights: identity passthrough.
        // The MoE sub-operations are handled by separate compiled steps
        // in the plan. MoeGating acts as a composite marker for dispatch
        // counting and buffer planning.
        //
        // Return the input slice directly -- output shape == input shape
        // per buffer_planner_bytes.rs.
        Ok(GpuSlice::from_ref(input_slice.buffer(), input_slice.byte_offset()))
    }
}

/// Full MoE execution path with weights.
///
/// Decomposes into DynTensor operations:
/// 1. Gate: input @ gate_weight^T -> logits [batch, num_experts]
/// 2. Softmax over last dim -> probabilities
/// 3. Topk(top_k) -> (weights, indices)
/// 4. Renormalize: weights / sum(weights, keepdim=true)
/// 5. Per-expert scatter-gather via DynTensor ops
///
/// TODO(#4287): Replace with fused Metal kernel for better performance.
#[allow(clippy::too_many_arguments)]
fn execute_moe_with_weights(
    model: &CompiledModel,
    step_idx: usize,
    input_slice: &GpuSlice,
    dtype: nn_core::DType,
    num_experts: usize,
    top_k: usize,
    input_shape: &[usize],
) -> Result<GpuSlice> {
    let step_weights = &model.def.weight_buffers[step_idx];

    // Resolve input tensor.
    let hidden = slice_to_dyn(input_slice, input_shape, dtype)?;

    // Gate linear: hidden [..., D] @ gate_weight [num_experts, D]^T -> logits [..., E]
    let model_dim = *input_shape.last().ok_or_else(|| {
        native_dispatch_err(step_idx, "MoeGating: empty input_shape".into())
    })?;
    let gate_weight = weight_to_dyn(
        step_weights,
        "gate_weight",
        &[num_experts, model_dim],
        dtype,
        step_idx,
        "MoeGating",
    )?;
    let logits = hidden.matmul(&gate_weight.t()?)?;
    let last_dim = logits.rank() - 1;

    // Softmax -> routing probabilities.
    let probs = logits.softmax(last_dim)?;

    // Topk selection.
    let (topk_weights, topk_indices) = probs.topk(last_dim, top_k)?;

    // Renormalize topk weights: w_i / sum(w).
    let w_sum = topk_weights.sum_keepdim(last_dim)?;
    let routing_weights = topk_weights.broadcast_div(&w_sum)?;

    // Flatten to [N, D] for scatter-gather.
    let n_tokens: usize = input_shape.iter().rev().skip(1).product();
    let flat_hidden = hidden.reshape([n_tokens, model_dim])?;
    let flat_indices = topk_indices.reshape([n_tokens, top_k])?;
    let flat_weights = routing_weights.reshape([n_tokens, top_k])?;

    // Scatter-gather: per-expert forward via DynTensor ops.
    // Read expert indices to CPU for routing dispatch.
    let idx_arr = flat_indices.as_cpu_u32().map_err(|e| {
        native_dispatch_err(
            step_idx,
            format!("MoeGating: failed to read expert indices: {e}"),
        )
    })?;
    let wt_arr = flat_weights.to_f32_array().map_err(|e| {
        native_dispatch_err(
            step_idx,
            format!("MoeGating: failed to read routing weights: {e}"),
        )
    })?;

    // Group tokens by expert.
    let mut expert_groups: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_experts];
    for token_idx in 0..n_tokens {
        for k_idx in 0..top_k {
            let expert_id = idx_arr[[token_idx, k_idx]] as usize;
            let weight_val = wt_arr[[token_idx, k_idx]];
            if expert_id < num_experts {
                expert_groups[expert_id].push((token_idx, weight_val));
            }
        }
    }

    // Initialize output as zeros [N, D].
    let device = flat_hidden.device();
    let mut output = nn_core::dyn_tensor::DynTensor::zeros(
        &[n_tokens, model_dim],
        nn_core::DType::F32,
        &device,
    )?;

    // Per-expert forward + weighted scatter-add.
    for (expert_idx, assignments) in expert_groups.iter().enumerate() {
        if assignments.is_empty() {
            continue;
        }

        // Check for expert weight keys.
        let gate_key = format!("expert_{expert_idx}_gate_proj");
        let up_key = format!("expert_{expert_idx}_up_proj");
        let down_key = format!("expert_{expert_idx}_down_proj");

        if !step_weights.contains_key(&gate_key) {
            // Expert weights not available -- skip this expert.
            continue;
        }

        // Gather tokens for this expert.
        let token_indices: Vec<u32> = assignments.iter().map(|&(t, _)| t as u32).collect();
        let expert_weights_vec: Vec<f32> = assignments.iter().map(|&(_, w)| w).collect();
        let n_assigned = token_indices.len();

        // index_select: gather tokens assigned to this expert.
        let idx_tensor = nn_core::dyn_tensor::DynTensor::from_vec_u32(
            token_indices,
            &[n_assigned],
            &device,
        )?;
        let gathered = flat_hidden.index_select(&idx_tensor, 0)?;

        // Expert forward: SwiGLU(gate_proj, up_proj) @ down_proj.
        // Expert intermediate size inferred from gate_proj weight buffer.
        let gate_buf = step_weights.get(&gate_key).ok_or_else(|| {
            native_dispatch_err(step_idx, format!("MoeGating: missing '{gate_key}'"))
        })?;
        let gate_buf_bytes = gate_buf.len();
        let elem_bytes = dtype.size_bytes();
        let gate_elements = gate_buf_bytes / elem_bytes;
        let intermediate_size = gate_elements / model_dim;

        let expert_gate = weight_to_dyn(
            step_weights,
            &gate_key,
            &[intermediate_size, model_dim],
            dtype,
            step_idx,
            "MoeGating expert",
        )?;
        let expert_up = weight_to_dyn(
            step_weights,
            &up_key,
            &[intermediate_size, model_dim],
            dtype,
            step_idx,
            "MoeGating expert",
        )?;
        let expert_down = weight_to_dyn(
            step_weights,
            &down_key,
            &[model_dim, intermediate_size],
            dtype,
            step_idx,
            "MoeGating expert",
        )?;

        // SwiGLU: silu(gathered @ gate^T) * (gathered @ up^T)
        let gate_out = gathered.matmul(&expert_gate.t()?)?;
        let up_out = gathered.matmul(&expert_up.t()?)?;
        let silu_gate = gate_out.silu()?;
        let expert_hidden = silu_gate.mul(&up_out)?;

        // Down projection: expert_hidden @ down^T
        let expert_out = expert_hidden.matmul(&expert_down.t()?)?;

        // Weighted scatter-add back to output.
        let weight_tensor = nn_core::dyn_tensor::DynTensor::from_slice(
            &expert_weights_vec,
            &[n_assigned, 1],
            &device,
        )?;
        let weighted_out = expert_out.broadcast_mul(&weight_tensor)?;

        // Scatter-add: for each assigned token, accumulate into output.
        for (local_idx, &(token_idx, _)) in assignments.iter().enumerate() {
            let row = weighted_out.narrow(0, local_idx, 1)?;
            let existing = output.narrow(0, token_idx, 1)?;
            let summed = existing.add(&row)?;
            output = scatter_row(&output, token_idx, &summed)?;
        }
    }

    // Reshape output back to original input shape.
    let output = output.reshape(input_shape)?;
    dyn_to_slice(&output, step_idx, "MoeGating")
}

/// Replace row `idx` in `tensor` with `row`.
///
/// Workaround for lack of in-place scatter in DynTensor.
/// Concatenates [tensor[..idx], row, tensor[idx+1..]] along dim 0.
fn scatter_row(
    tensor: &nn_core::dyn_tensor::DynTensor,
    idx: usize,
    row: &nn_core::dyn_tensor::DynTensor,
) -> Result<nn_core::dyn_tensor::DynTensor> {
    let n = tensor.dims()[0];
    if n == 1 {
        return Ok(row.clone());
    }
    let mut parts: Vec<nn_core::dyn_tensor::DynTensor> = Vec::with_capacity(3);
    if idx > 0 {
        parts.push(tensor.narrow(0, 0, idx)?);
    }
    parts.push(row.clone());
    if idx + 1 < n {
        parts.push(tensor.narrow(0, idx + 1, n - idx - 1)?);
    }
    let refs: Vec<&nn_core::dyn_tensor::DynTensor> = parts.iter().collect();
    nn_core::dyn_tensor::DynTensor::cat(&refs, 0)
}
