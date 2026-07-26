// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU fused MoE scatter-gather dispatch for [`MetalDynBackend`].
//!
//! Implements the MoE scatter-gather pattern entirely on GPU, eliminating
//! per-expert CPU readback of routing indices. The approach:
//!
//! 1. Reads routing indices to CPU once (O(N*K), typically small)
//! 2. Builds per-expert index tensors on CPU
//! 3. Uploads index tensors to GPU
//! 4. Dispatches all expert SwiGLU FFNs on GPU as a batched command sequence
//! 5. Uses GPU index_add for weighted accumulation
//!
//! The key optimization vs the decomposed path: all expert forward passes
//! (matmul, silu, mul) are batched into a single Metal command buffer via
//! lazy batch (#2009), eliminating per-expert commit barriers.
//!
//! Issue: #3547

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result, TensorError};

use super::MetalDynBackend;

impl MetalDynBackend {
    /// Fused MoE scatter-gather on Metal GPU.
    ///
    /// Performs per-expert dispatch with all tensor operations staying on GPU.
    /// Expert forward passes are batched into a single Metal command buffer.
    ///
    /// Returns `None` for non-F32 tensors to trigger CPU fallback.
    ///
    /// Issue: #3547
    pub(super) fn gpu_moe_scatter_gather(
        hidden: &DynTensor,
        indices: &DynTensor,
        weights: &DynTensor,
        expert_gate_weights: &[DynTensor],
        expert_up_weights: &[DynTensor],
        expert_down_weights: &[DynTensor],
        num_experts: usize,
    ) -> Option<Result<DynTensor>> {
        // Only F32 supported; non-F32 falls back to CPU loop.
        if hidden.dtype() != DType::F32 {
            return crate::gpu_fallback("moe_scatter_gather", "non-F32 hidden");
        }
        if expert_gate_weights.len() != num_experts
            || expert_up_weights.len() != num_experts
            || expert_down_weights.len() != num_experts
        {
            return Some(Err(TensorError::DataLengthMismatch {
                expected: num_experts,
                actual: expert_gate_weights.len(),
            }));
        }

        Some(Self::gpu_moe_scatter_gather_inner(
            hidden,
            indices,
            weights,
            expert_gate_weights,
            expert_up_weights,
            expert_down_weights,
            num_experts,
        ))
    }

    /// Inner implementation: batched per-expert GPU dispatch.
    ///
    /// Strategy:
    /// 1. Read routing indices to CPU for O(N*K) grouping (small transfer)
    /// 2. For each expert with assigned tokens:
    ///    a. Build token-ID and weight tensors on GPU
    ///    b. index_select: gather assigned tokens from hidden
    ///    c. SwiGLU FFN: gate=silu(x@gate^T), up=x@up^T, h=gate*up, out=h@down^T
    ///    d. index_add: weighted scatter back to output accumulator
    /// 3. All GPU ops are batched into lazy command buffer (#2009)
    fn gpu_moe_scatter_gather_inner(
        hidden: &DynTensor,
        indices: &DynTensor,
        weights: &DynTensor,
        expert_gate_weights: &[DynTensor],
        expert_up_weights: &[DynTensor],
        expert_down_weights: &[DynTensor],
        num_experts: usize,
    ) -> Result<DynTensor> {
        let hidden_dims = hidden.dims();
        let n_tokens = hidden_dims[0];
        let model_dim = hidden_dims[1];
        let device = hidden.device();
        let k = indices.dims()[1]; // top-k

        // Flush pending GPU work so index readback sees committed data.
        crate::gpu_scope::flush()
            .map_err(|e| TensorError::InvalidShape(e.to_string()))?;

        // Transfer routing indices and weights to CPU for grouping (O(N*K), small).
        let indices_cpu = indices.to_device(&Device::Cpu)?;
        let weights_cpu = weights.to_device(&Device::Cpu)?;
        let idx_arr = indices_cpu.as_cpu_u32()?;
        let wt_arr = weights_cpu.to_f32_array()?;

        // Group tokens by expert in a single O(N*K) pass.
        let avg_per_expert = (n_tokens * k) / num_experts.max(1) + 1;
        let mut assignments: Vec<Vec<(usize, f32)>> = (0..num_experts)
            .map(|_| Vec::with_capacity(avg_per_expert))
            .collect();
        for t in 0..n_tokens {
            for s in 0..k {
                let coord = &[t, s];
                let expert_idx = idx_arr[ndarray::IxDyn(coord)] as usize;
                if expert_idx >= num_experts {
                    return Err(TensorError::DimensionOutOfRange {
                        dim: expert_idx,
                        rank: num_experts,
                    });
                }
                let weight = wt_arr.view()[ndarray::IxDyn(coord)];
                assignments[expert_idx].push((t, weight));
            }
        }

        // Zero-initialized output accumulator on GPU.
        let mut output = DynTensor::zeros(&[n_tokens, model_dim], DType::F32, &device)?;

        // Dispatch each expert's tokens entirely on GPU.
        // All ops are batched into the lazy command buffer (#2009).
        for (expert_idx, expert_assignments) in assignments.iter().enumerate() {
            if expert_assignments.is_empty() {
                continue;
            }

            let num_routed = expert_assignments.len();

            // Build token-ID and weight tensors directly on GPU.
            let token_ids: Vec<u32> = expert_assignments
                .iter()
                .map(|&(t, _)| {
                    u32::try_from(t).map_err(|_| TensorError::ValueOutOfRange {
                        description: "MoE GPU: token index exceeds u32::MAX",
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let expert_wts: Vec<f32> = expert_assignments.iter().map(|&(_, w)| w).collect();

            let ids_tensor = DynTensor::from_vec_u32(token_ids, &[num_routed], &device)?;
            let w_tensor = DynTensor::from_vec(expert_wts, &[num_routed, 1], &device)?;

            // Gather tokens assigned to this expert.
            let gathered = hidden.index_select(&ids_tensor, 0)?; // [num_routed, D]

            // SwiGLU FFN: gate=silu(gathered @ gate_weight^T), up=gathered @ up_weight^T
            //             h=gate*up, expert_out=h @ down_weight^T
            // Use pre-transposed weights: Linear stores [out_features, in_features],
            // so `gathered @ weight^T` = matmul(gathered, weight.transpose(0,1)).
            let gate_w_t = expert_gate_weights[expert_idx].transpose(0, 1)?;
            let up_w_t = expert_up_weights[expert_idx].transpose(0, 1)?;
            let down_w_t = expert_down_weights[expert_idx].transpose(0, 1)?;

            let gate = gathered.matmul(&gate_w_t)?.silu()?;
            let up = gathered.matmul(&up_w_t)?;
            let h = gate.broadcast_mul(&up)?;
            let expert_out = h.matmul(&down_w_t)?; // [num_routed, D]

            // Weight and accumulate.
            let weighted = expert_out.broadcast_mul(&w_tensor)?;
            output = output.index_add(0, &ids_tensor, &weighted)?;
        }

        Ok(output)
    }
}
