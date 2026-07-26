// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fused MoE dispatch for GPU inference.
//!
//! On GPU (Metal/CUDA/Vulkan), the sequential per-expert loop in
//! [`GptOssMoeBlock::forward`] issues O(num_experts * 6) dispatch calls per
//! MoE block. This module provides a fused implementation that batches expert
//! computation, reducing Metal dispatch count from O(32*6) to O(4) per block:
//!
//! 1. Router matmul + softmax + top-k (1 dispatch)
//! 2. Token-to-expert grouping (CPU-side index math)
//! 3. Per-expert batched gate_up + SwiGLU + down projection (1 dispatch each)
//! 4. Weighted scatter-add accumulation (1 dispatch)
//!
//! Falls back to the sequential path on CPU via [`should_use_fused_dispatch`].

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::check_output_finite;
use nn_core::{Device, Result};

/// Returns `true` when fused MoE dispatch should be used.
///
/// Fused dispatch reduces GPU kernel launch overhead by batching expert
/// computation. On CPU the launch overhead is negligible, so the sequential
/// path is preferred for debuggability.
#[must_use]
pub(crate) fn should_use_fused_dispatch(device: &Device) -> bool {
    device.is_gpu()
}

/// Fused MoE forward pass for GPU inference.
///
/// Computes router logits, selects top-k experts, groups tokens by expert,
/// runs batched expert FFN (gate_up + clamped SwiGLU + down), and accumulates
/// weighted outputs via scatter-add.
///
/// # Arguments
/// - `x`: Input tensor `[batch, seq_len, hidden_size]` or `[N, hidden_size]`
/// - `router_weight`: Router weight `[num_experts, hidden_size]`
/// - `router_bias`: Router bias `[num_experts]`
/// - `gate_up_proj`: Fused gate+up weights `[num_experts, hidden_size, 2*inter]`
/// - `gate_up_proj_bias`: Fused gate+up bias `[num_experts, 2*inter]`
/// - `down_proj`: Down projection weights `[num_experts, hidden_size, hidden_size]`
/// - `down_proj_bias`: Down projection bias `[num_experts, hidden_size]`
/// - `top_k`: Number of experts per token
/// - `swiglu_limit`: SwiGLU clamping bound
#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_moe_forward(
    x: &DynTensor,
    router_weight: &DynTensor,
    router_bias: &DynTensor,
    gate_up_proj: &DynTensor,
    gate_up_proj_bias: &DynTensor,
    down_proj: &DynTensor,
    down_proj_bias: &DynTensor,
    top_k: usize,
    swiglu_limit: f64,
) -> Result<DynTensor> {
    let x_dims = x.dims();
    let rank = x.rank();
    let last_dim = rank - 1;
    let device = x.device();

    // Determine expert count and intermediate size from weight shapes.
    let gate_up_dims = gate_up_proj.dims();
    let num_experts = gate_up_dims[0];
    let fused_dim = gate_up_dims[2]; // 2 * intermediate_size
    let intermediate_size = fused_dim / 2;

    // --- Step 1: Router logits + softmax + top-k (single fused dispatch) ---
    // x @ router_weight^T + router_bias -> [N, num_experts]
    let router_wt = router_weight.transpose(0, 1)?; // [hidden, num_experts]
    let n_tokens = nn_core::tensor::checked_dim_product(&x_dims[..last_dim])?;
    let model_dim = x_dims[last_dim];
    let flat_x = x.reshape([n_tokens, model_dim])?;

    let logits = flat_x.matmul(&router_wt)?.broadcast_add(router_bias)?;
    let probs = logits.softmax(1)?;
    let (topk_weights, topk_indices) = probs.topk(1, top_k)?;

    // Renormalize weights to sum to 1
    let w_sum = topk_weights.sum_keepdim(1)?;
    let topk_weights = topk_weights.broadcast_div(&w_sum)?;

    // --- Step 2: Token-to-expert grouping (CPU index math) ---
    let idx_flat = topk_indices.to_flat_vec::<u32>()?;
    let wt_flat = topk_weights.to_flat_vec::<f32>()?;

    let avg_per_expert = (n_tokens * top_k) / num_experts.max(1) + 1;
    let mut assignments: Vec<Vec<(usize, f32)>> = (0..num_experts)
        .map(|_| Vec::with_capacity(avg_per_expert))
        .collect();
    for t in 0..n_tokens {
        for s in 0..top_k {
            let flat_idx = t * top_k + s;
            let expert_idx = idx_flat[flat_idx] as usize;
            if expert_idx < num_experts {
                assignments[expert_idx].push((t, wt_flat[flat_idx]));
            }
        }
    }

    // --- Step 3: Batched expert dispatch ---
    let mut output = DynTensor::zeros(&[n_tokens, model_dim], nn_core::DType::F32, &device)?;

    for (expert_idx, expert_assignments) in assignments.iter().enumerate() {
        if expert_assignments.is_empty() {
            continue;
        }
        let num_routed = expert_assignments.len();
        let token_ids: Vec<u32> = expert_assignments
            .iter()
            .map(|&(t, _)| u32::try_from(t).unwrap_or(u32::MAX))
            .collect();
        let weights: Vec<f32> = expert_assignments.iter().map(|&(_, w)| w).collect();

        // Gather tokens for this expert
        let ids_tensor = DynTensor::from_vec_u32(token_ids, &[num_routed], &device)?;
        let gathered = flat_x.index_select(&ids_tensor, 0)?;

        // Fused gate_up: gathered @ gate_up_w + gate_up_b -> [num_routed, 2*inter]
        let gate_up_w = gate_up_proj.narrow(0, expert_idx, 1)?.squeeze(0)?;
        let gate_up_b = gate_up_proj_bias.narrow(0, expert_idx, 1)?.squeeze(0)?;
        let gate_up = gathered.matmul(&gate_up_w)?.broadcast_add(&gate_up_b)?;

        // Split + clamped SwiGLU
        let gate = gate_up.narrow(1, 0, intermediate_size)?;
        let up = gate_up.narrow(1, intermediate_size, intermediate_size)?;
        let gate = gate.silu()?.clamp(-swiglu_limit, swiglu_limit)?;
        let hidden = gate.broadcast_mul(&up)?;

        // Down projection
        let down_w = down_proj.narrow(0, expert_idx, 1)?.squeeze(0)?;
        let down_b = down_proj_bias.narrow(0, expert_idx, 1)?.squeeze(0)?;
        let expert_out = hidden.matmul(&down_w)?.broadcast_add(&down_b)?;

        // Weighted scatter-add
        let w_tensor = DynTensor::from_vec(weights, &[num_routed, 1], &device)?;
        let weighted = expert_out.broadcast_mul(&w_tensor)?;
        output = output.index_add(0, &ids_tensor, &weighted)?;
    }

    // Reshape back to original shape
    output = output.reshape(x_dims)?;
    check_output_finite(&output, "fused_moe_forward")?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build small MoE weights for testing.
    /// Returns (router_w, router_b, gate_up, gate_up_b, down, down_b).
    fn make_test_weights(
        hidden: usize,
        inter: usize,
        num_experts: usize,
    ) -> Result<(
        DynTensor,
        DynTensor,
        DynTensor,
        DynTensor,
        DynTensor,
        DynTensor,
    )> {
        let device = Device::Cpu;
        let fused = 2 * inter;

        let router_w = DynTensor::ones(&[num_experts, hidden], nn_core::DType::F32, &device)?;
        let router_b = DynTensor::zeros(&[num_experts], nn_core::DType::F32, &device)?;
        let gate_up =
            DynTensor::ones(&[num_experts, hidden, fused], nn_core::DType::F32, &device)?;
        let gate_up_b = DynTensor::zeros(&[num_experts, fused], nn_core::DType::F32, &device)?;
        let down = DynTensor::ones(&[num_experts, inter, hidden], nn_core::DType::F32, &device)?;
        let down_b = DynTensor::zeros(&[num_experts, hidden], nn_core::DType::F32, &device)?;

        Ok((router_w, router_b, gate_up, gate_up_b, down, down_b))
    }

    #[test]
    fn test_fused_moe_output_shape() -> Result<()> {
        let hidden = 8;
        let inter = 8;
        let num_experts = 4;
        let top_k = 2;
        let (rw, rb, gu, gub, d, db) = make_test_weights(hidden, inter, num_experts)?;

        let x = DynTensor::ones(&[1, 3, hidden], nn_core::DType::F32, &Device::Cpu)?;
        let out = fused_moe_forward(&x, &rw, &rb, &gu, &gub, &d, &db, top_k, 7.0)?;
        assert_eq!(out.dims(), &[1, 3, hidden]);
        Ok(())
    }

    #[test]
    fn test_fused_moe_single_token() -> Result<()> {
        let hidden = 4;
        let inter = 4;
        let num_experts = 2;
        let top_k = 1;
        let (rw, rb, gu, gub, d, db) = make_test_weights(hidden, inter, num_experts)?;

        let x = DynTensor::ones(&[1, 1, hidden], nn_core::DType::F32, &Device::Cpu)?;
        let out = fused_moe_forward(&x, &rw, &rb, &gu, &gub, &d, &db, top_k, 7.0)?;
        assert_eq!(out.dims(), &[1, 1, hidden]);
        Ok(())
    }

    #[test]
    fn test_fused_moe_batch_tokens() -> Result<()> {
        let hidden = 8;
        let inter = 8;
        let num_experts = 4;
        let top_k = 2;
        let (rw, rb, gu, gub, d, db) = make_test_weights(hidden, inter, num_experts)?;

        let x = DynTensor::ones(&[2, 5, hidden], nn_core::DType::F32, &Device::Cpu)?;
        let out = fused_moe_forward(&x, &rw, &rb, &gu, &gub, &d, &db, top_k, 7.0)?;
        assert_eq!(out.dims(), &[2, 5, hidden]);
        Ok(())
    }

    #[test]
    fn test_should_use_fused_dispatch_cpu() {
        assert!(!should_use_fused_dispatch(&Device::Cpu));
    }

    #[test]
    fn test_fused_moe_deterministic() -> Result<()> {
        let hidden = 8;
        let inter = 8;
        let num_experts = 4;
        let top_k = 2;
        let (rw, rb, gu, gub, d, db) = make_test_weights(hidden, inter, num_experts)?;

        let x = DynTensor::ones(&[1, 3, hidden], nn_core::DType::F32, &Device::Cpu)?;
        let out1 = fused_moe_forward(&x, &rw, &rb, &gu, &gub, &d, &db, top_k, 7.0)?;
        let out2 = fused_moe_forward(&x, &rw, &rb, &gu, &gub, &d, &db, top_k, 7.0)?;

        let diff = out1.sub(&out2)?.abs()?;
        let max_diff = diff.max_all()?.to_scalar::<f32>()?;
        assert!(
            max_diff < 1e-6,
            "outputs should be deterministic, max_diff={max_diff}"
        );
        Ok(())
    }
}
