// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU-oriented MoE dispatch with explicit scatter/gather phases.
//!
//! [`MoeDispatch`] separates routing, scatter, expert execution, and gather
//! into distinct phases optimized for GPU dispatch:
//!
//! 1. **Gating**: `router_logits = Linear(hidden, num_experts)` then
//!    `top_k_softmax(router_logits, k)` producing `(expert_indices, expert_weights)`.
//! 2. **Scatter**: Route tokens to assigned experts based on top-k indices.
//! 3. **Expert FFN**: Apply per-expert SwiGLU FFN.
//! 4. **Gather**: Weighted combination of expert outputs back to original positions.
//!
//! Unlike [`MoeLayer`] which uses a single combined forward, `MoeDispatch`
//! exposes `compute_routing` and `scatter_gather` as separate operations.
//! This separation enables future GPU kernel fusion of the scatter/gather
//! phases (Metal/CUDA grouped GEMM).
//!
//! # Load balancing auxiliary loss
//!
//! [`MoeDispatch::forward_with_aux_loss`] returns the auxiliary load-balancing
//! loss alongside the output. The loss encourages uniform expert utilization:
//!
//! ```text
//! aux_loss = num_experts * sum_e(f_e * P_e)
//! ```
//!
//! where `f_e` is the fraction of tokens routed to expert `e` and `P_e` is
//! the mean routing probability for expert `e`.

use super::{check_output_finite, Linear, Module, SwiGluExpert};
use crate::dyn_tensor::{gpu_backend_dispatch, DynTensor};
use crate::var_builder::VarBuilder;
use crate::{DType, Result, TensorError};

// -- Configuration -----------------------------------------------------------

/// Configuration for [`MoeDispatch`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MoeDispatchConfig {
    /// Total number of experts.
    pub num_experts: usize,
    /// Number of experts selected per token.
    pub top_k: usize,
    /// Hidden dimension (model dimension).
    pub hidden_size: usize,
    /// Intermediate FFN dimension for each expert.
    pub expert_intermediate_size: usize,
    /// Whether to renormalize top-k probabilities to sum to 1.
    pub norm_topk_prob: bool,
}

impl MoeDispatchConfig {
    /// Create a new configuration.
    ///
    /// Returns an error if `top_k` is 0 or exceeds `num_experts`, or if
    /// `hidden_size` or `expert_intermediate_size` is 0.
    pub fn new(
        num_experts: usize,
        top_k: usize,
        hidden_size: usize,
        expert_intermediate_size: usize,
        norm_topk_prob: bool,
    ) -> Result<Self> {
        if top_k == 0 || top_k > num_experts {
            return Err(TensorError::ValueOutOfRange {
                description: "MoeDispatchConfig: top_k must be in [1, num_experts]",
            });
        }
        if hidden_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "MoeDispatchConfig: hidden_size must be > 0",
            });
        }
        if expert_intermediate_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "MoeDispatchConfig: expert_intermediate_size must be > 0",
            });
        }
        Ok(Self {
            num_experts,
            top_k,
            hidden_size,
            expert_intermediate_size,
            norm_topk_prob,
        })
    }
}

// -- MoeDispatch output with optional aux loss -------------------------------

/// Output from [`MoeDispatch::forward_with_aux_loss`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MoeDispatchOutput {
    /// Hidden states after expert dispatch. Same shape as input.
    pub hidden_states: DynTensor,
    /// Auxiliary load-balancing loss (scalar).
    pub aux_loss: DynTensor,
}

impl MoeDispatchOutput {
    /// Create a dispatch output.
    pub fn new(hidden_states: DynTensor, aux_loss: DynTensor) -> Self {
        Self {
            hidden_states,
            aux_loss,
        }
    }
}

// -- MoeDispatch -------------------------------------------------------------

/// GPU-oriented MoE dispatch with explicit scatter/gather phases.
///
/// Provides the same forward semantics as [`MoeLayer`] but separates
/// routing from scatter/gather for GPU kernel optimization.
#[derive(Debug, Clone)]
pub struct MoeDispatch {
    router: Linear,
    experts: Vec<SwiGluExpert>,
    cfg: MoeDispatchConfig,
}

impl MoeDispatch {
    /// Load from a [`VarBuilder`].
    ///
    /// Weight key layout:
    /// ```text
    /// gate.weight                         -> router Linear
    /// experts.{e}.gate_proj.weight        -> expert gate
    /// experts.{e}.up_proj.weight          -> expert up
    /// experts.{e}.down_proj.weight        -> expert down
    /// ```
    pub fn load(vb: &VarBuilder, cfg: MoeDispatchConfig) -> Result<Self> {
        let router = Linear::load(vb.pp("gate"), cfg.hidden_size, cfg.num_experts)?;
        let experts_vb = vb.pp("experts");
        let experts = (0..cfg.num_experts)
            .map(|e| {
                SwiGluExpert::load(
                    experts_vb.pp(e.to_string()),
                    cfg.hidden_size,
                    cfg.expert_intermediate_size,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            router,
            experts,
            cfg,
        })
    }

    /// Create from pre-built components.
    pub fn new(router: Linear, experts: Vec<SwiGluExpert>, cfg: MoeDispatchConfig) -> Result<Self> {
        if experts.len() != cfg.num_experts {
            return Err(TensorError::DataLengthMismatch {
                expected: cfg.num_experts,
                actual: experts.len(),
            });
        }
        Ok(Self {
            router,
            experts,
            cfg,
        })
    }

    /// Access the configuration.
    #[must_use]
    pub fn config(&self) -> &MoeDispatchConfig {
        &self.cfg
    }

    /// Access the router linear.
    #[must_use]
    pub fn router(&self) -> &Linear {
        &self.router
    }

    /// Compute routing: softmax over experts, top-k selection, optional renormalization.
    ///
    /// Input: `[..., D]` where D = `hidden_size`.
    /// Returns `(expert_indices, expert_weights)`:
    /// - `expert_indices`: `[..., K]` (U32) — selected expert indices per token.
    /// - `expert_weights`: `[..., K]` (F32) — corresponding routing weights.
    pub(crate) fn compute_routing(&self, hidden: &DynTensor) -> Result<(DynTensor, DynTensor)> {
        let logits = self.router.forward(hidden)?; // [..., num_experts]
        let last_dim = logits.rank() - 1;

        // Softmax over all experts, then select top-k.
        let probs = logits.softmax(last_dim)?; // [..., num_experts]
        let (topk_weights, topk_indices) = probs.topk(last_dim, self.cfg.top_k)?;

        // Optionally renormalize so selected weights sum to 1.
        let topk_weights = if self.cfg.norm_topk_prob {
            let w_sum = topk_weights.sum_keepdim(last_dim)?;
            topk_weights.broadcast_div(&w_sum)?
        } else {
            topk_weights
        };

        Ok((topk_indices, topk_weights))
    }

    /// Scatter tokens to experts, run expert FFNs, gather weighted results.
    ///
    /// `hidden`: `[N, D]` flattened token embeddings.
    /// `indices`: `[N, K]` expert indices (U32).
    /// `weights`: `[N, K]` routing weights (F32).
    ///
    /// Returns `[N, D]` with the weighted combination of expert outputs.
    ///
    /// Tries fused GPU dispatch first (#3547), falls back to per-expert loop.
    pub(crate) fn scatter_gather(
        &self,
        hidden: &DynTensor,
        indices: &DynTensor,
        weights: &DynTensor,
    ) -> Result<DynTensor> {
        // Try fused GPU dispatch first (#3547).
        if hidden.device().is_gpu() {
            let gate_ws: Vec<DynTensor> = self
                .experts
                .iter()
                .map(|e| e.gate_proj().weight().clone())
                .collect();
            let up_ws: Vec<DynTensor> = self
                .experts
                .iter()
                .map(|e| e.up_proj().weight().clone())
                .collect();
            let down_ws: Vec<DynTensor> = self
                .experts
                .iter()
                .map(|e| e.down_proj().weight().clone())
                .collect();
            if let Some(result) = gpu_backend_dispatch(|b| {
                b.moe_scatter_gather(
                    hidden,
                    indices,
                    weights,
                    &gate_ws,
                    &up_ws,
                    &down_ws,
                    self.cfg.num_experts,
                )
            }) {
                return result;
            }
        }

        // Fallback: per-expert loop dispatch.
        let dims = hidden.dims();
        let n_tokens = dims[0];
        let model_dim = dims[1];
        let k = self.cfg.top_k;
        let device = hidden.device();

        // Transfer routing to CPU for grouping (O(N*k), small).
        let idx_arr = indices.as_cpu_u32()?;
        let wt_arr = weights.to_f32_array()?;

        let num_experts = self.cfg.num_experts;
        let assignments =
            group_tokens_by_expert(&idx_arr, &wt_arr.view(), n_tokens, k, num_experts)?;

        // Accumulator stays on the input device.
        let mut output = DynTensor::zeros(&[n_tokens, model_dim], DType::F32, &device)?;

        // Dispatch tokens to each expert using tensor ops (device-native).
        for (expert_idx, expert_assignments) in assignments.iter().enumerate() {
            if expert_assignments.is_empty() {
                continue;
            }
            output = dispatch_single_expert(
                &self.experts[expert_idx],
                hidden,
                &output,
                expert_assignments,
                &device,
            )?;
        }

        Ok(output)
    }

    /// Forward with auxiliary load-balancing loss.
    ///
    /// Returns both the output hidden states and a scalar aux loss that
    /// encourages uniform expert utilization.
    pub fn forward_with_aux_loss(&self, hidden: &DynTensor) -> Result<MoeDispatchOutput> {
        let input_dims = hidden.dims();
        let rank = hidden.rank();
        let last_dim = rank - 1;
        let device = hidden.device();

        let n_tokens = crate::tensor::checked_dim_product(&input_dims[..last_dim])?;
        let model_dim = input_dims[last_dim];

        // Compute full routing probabilities for aux loss before topk.
        let logits = self.router.forward(hidden)?;
        let logits_last = logits.rank() - 1;
        let probs = logits.softmax(logits_last)?;
        let (topk_weights, topk_indices) = probs.topk(logits_last, self.cfg.top_k)?;

        let topk_weights = if self.cfg.norm_topk_prob {
            let w_sum = topk_weights.sum_keepdim(logits_last)?;
            topk_weights.broadcast_div(&w_sum)?
        } else {
            topk_weights
        };

        // Flatten to [N, D] and [N, K].
        let flat_hidden = hidden.reshape([n_tokens, model_dim])?;
        let flat_indices = topk_indices.reshape([n_tokens, self.cfg.top_k])?;
        let flat_weights = topk_weights.reshape([n_tokens, self.cfg.top_k])?;

        // Scatter-gather dispatch.
        let output = self.scatter_gather(&flat_hidden, &flat_indices, &flat_weights)?;
        let output = output.reshape(input_dims)?;

        // Compute auxiliary load-balancing loss.
        let flat_probs = probs.reshape([n_tokens, self.cfg.num_experts])?;
        let aux_loss = compute_aux_loss(
            &flat_indices,
            &flat_probs,
            n_tokens,
            self.cfg.num_experts,
            &device,
        )?;

        check_output_finite(&output, "MoeDispatch")?;
        Ok(MoeDispatchOutput::new(output, aux_loss))
    }
}

impl Module for MoeDispatch {
    /// Standard forward pass (no auxiliary loss).
    fn forward(&self, hidden: &DynTensor) -> Result<DynTensor> {
        let input_dims = hidden.dims();
        let rank = hidden.rank();
        let last_dim = rank - 1;

        let n_tokens = crate::tensor::checked_dim_product(&input_dims[..last_dim])?;
        let model_dim = input_dims[last_dim];

        let flat_hidden = hidden.reshape([n_tokens, model_dim])?;
        let (topk_indices, topk_weights) = self.compute_routing(hidden)?;
        let flat_indices = topk_indices.reshape([n_tokens, self.cfg.top_k])?;
        let flat_weights = topk_weights.reshape([n_tokens, self.cfg.top_k])?;

        let output = self.scatter_gather(&flat_hidden, &flat_indices, &flat_weights)?;
        let output = output.reshape(input_dims)?;

        check_output_finite(&output, "MoeDispatch")?;
        Ok(output)
    }
}

// -- Internal helpers --------------------------------------------------------

/// Group tokens by expert index in a single O(N*k) pass.
///
/// Returns a vec of per-expert `(token_id, weight)` lists. Validates that
/// all routing indices are in `[0, num_experts)`.
fn group_tokens_by_expert(
    idx_arr: &ndarray::ArrayViewD<'_, u32>,
    wt_arr: &ndarray::ArrayViewD<'_, f32>,
    n_tokens: usize,
    k: usize,
    num_experts: usize,
) -> Result<Vec<Vec<(usize, f32)>>> {
    let avg_per_expert = (n_tokens * k) / num_experts + 1;
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
            let weight = wt_arr[ndarray::IxDyn(coord)];
            assignments[expert_idx].push((t, weight));
        }
    }
    Ok(assignments)
}

/// Run one expert on its assigned tokens and index-add the weighted result.
fn dispatch_single_expert(
    expert: &SwiGluExpert,
    flat_x: &DynTensor,
    output: &DynTensor,
    assignments: &[(usize, f32)],
    device: &crate::Device,
) -> Result<DynTensor> {
    let num_routed = assignments.len();
    let token_ids: Vec<u32> = assignments
        .iter()
        .map(|&(t, _)| {
            u32::try_from(t).map_err(|_| TensorError::ValueOutOfRange {
                description: "MoeDispatch: token index exceeds u32::MAX",
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let weights: Vec<f32> = assignments.iter().map(|&(_, w)| w).collect();

    let ids_tensor = DynTensor::from_vec_u32(token_ids, &[num_routed], device)?;
    let gathered = flat_x.index_select(&ids_tensor, 0)?;
    let expert_out = expert.forward(&gathered)?;

    // Weight: [num_routed, 1] * [num_routed, D]
    let w_tensor = DynTensor::from_vec(weights, &[num_routed, 1], device)?;
    let weighted = expert_out.broadcast_mul(&w_tensor)?;

    output.index_add(0, &ids_tensor, &weighted)
}

/// Compute auxiliary load-balancing loss.
///
/// `aux_loss = num_experts * sum_e(f_e * P_e)` where:
/// - `f_e` = fraction of tokens routed to expert `e`
/// - `P_e` = mean routing probability for expert `e`
fn compute_aux_loss(
    flat_indices: &DynTensor,
    flat_probs: &DynTensor,
    n_tokens: usize,
    num_experts: usize,
    device: &crate::Device,
) -> Result<DynTensor> {
    if n_tokens == 0 {
        return DynTensor::zeros(&[], DType::F32, device);
    }

    // f_e: fraction of tokens assigned to each expert.
    // One-hot encode indices and average over tokens.
    let idx_arr = flat_indices.as_cpu_u32()?;
    let k = flat_indices.dims()[1];
    let mut expert_counts = vec![0.0f32; num_experts];
    let total_assignments = (n_tokens * k) as f32;

    for t in 0..n_tokens {
        for s in 0..k {
            let e = idx_arr[ndarray::IxDyn(&[t, s])] as usize;
            if e < num_experts {
                expert_counts[e] += 1.0;
            }
        }
    }
    // f_e = count_e / (n_tokens * k)
    let f_e: Vec<f32> = expert_counts
        .iter()
        .map(|&c| c / total_assignments)
        .collect();
    let f_tensor = DynTensor::from_vec(f_e, &[num_experts], device)?;

    // P_e: mean routing probability for expert e (mean over tokens).
    let p_e = flat_probs.mean(0)?; // [num_experts]

    // aux_loss = num_experts * sum(f_e * P_e)
    let fp = f_tensor.broadcast_mul(&p_e)?;
    let sum = fp.sum_all()?;
    let scale = DynTensor::full(&[], num_experts as f64, DType::F32, device)?;
    sum.broadcast_mul(&scale)
}

#[cfg(kani)]
#[path = "moe_dispatch_kani.rs"]
mod kani_proofs;

#[cfg(test)]
#[path = "moe_dispatch_tests.rs"]
mod tests;
