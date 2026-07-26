// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Config-driven Mixture-of-Experts layer for Qwen3-VL-30B-A3B and similar
//! MoE architectures (Qwen3-MoE, DeepSeek-V2/V3, Mixtral).
//!
//! Provides:
//! - [`ExpertFFN`]: SwiGLU-style MLP (gate_proj + up_proj + down_proj with SiLU)
//! - [`ExpertMlp`]: Generic MLP (up_proj + activation + down_proj)
//! - [`MoeLayerConfig`]: Validated configuration struct
//! - [`MoeLayer`]: Full MoE layer combining router, experts, optional shared expert,
//!   and scatter/gather dispatch
//!
//! # Weight naming convention (Qwen3)
//!
//! ```text
//! gate.weight                         -> router Linear
//! experts.{i}.gate_proj.weight        -> expert gate projection
//! experts.{i}.up_proj.weight          -> expert up projection
//! experts.{i}.down_proj.weight        -> expert down projection
//! shared_expert.gate_proj.weight      -> shared expert (optional)
//! shared_expert.up_proj.weight
//! shared_expert.down_proj.weight
//! ```
//!
//! # Forward pass
//!
//! 1. Router logits: `Linear(hidden_states, num_experts)` -> softmax -> top-k
//! 2. Scatter: route tokens to assigned experts based on top-k indices
//! 3. Expert FFN: SwiGLU on each expert's token batch
//! 4. Gather: weighted sum of expert outputs back to original token positions
//! 5. Shared expert: add output of shared FFN on ALL tokens (if enabled)

use super::{check_output_finite, Linear, Module};
use crate::dyn_tensor::{gpu_backend_dispatch, DynTensor};
use crate::var_builder::VarBuilder;
use crate::{DType, Result, TensorError};

// Re-export expert types from moe_experts (500-line extraction).
pub use super::moe_experts::{ExpertFFN, ExpertMlp};

// -- MoeLayerConfig -----------------------------------------------------------

/// Configuration for [`MoeLayer`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MoeLayerConfig {
    /// Total number of experts.
    pub num_experts: usize,
    /// Number of experts selected per token (top-k).
    pub top_k: usize,
    /// Model hidden dimension.
    pub hidden_size: usize,
    /// Intermediate FFN dimension for each expert.
    pub expert_intermediate_size: usize,
    /// Whether to renormalize top-k routing weights to sum to 1.
    pub norm_topk_prob: bool,
    /// Whether to include a shared expert that processes ALL tokens.
    pub shared_expert: bool,
    /// Intermediate size for the shared expert. Defaults to
    /// `expert_intermediate_size` if `None` and `shared_expert` is true.
    pub shared_expert_intermediate_size: Option<usize>,
}

impl MoeLayerConfig {
    /// Create a new configuration.
    ///
    /// Returns an error if invariants are violated:
    /// - `top_k` must be in `[1, num_experts]`
    /// - `hidden_size` and `expert_intermediate_size` must be > 0
    pub fn new(
        num_experts: usize,
        top_k: usize,
        hidden_size: usize,
        expert_intermediate_size: usize,
        norm_topk_prob: bool,
        shared_expert: bool,
    ) -> Result<Self> {
        let cfg = Self {
            num_experts,
            top_k,
            hidden_size,
            expert_intermediate_size,
            norm_topk_prob,
            shared_expert,
            shared_expert_intermediate_size: None,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Create configuration with a custom shared expert intermediate size.
    pub fn with_shared_intermediate_size(mut self, size: usize) -> Result<Self> {
        if size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "MoeLayerConfig: shared_expert_intermediate_size must be > 0",
            });
        }
        self.shared_expert_intermediate_size = Some(size);
        Ok(self)
    }

    /// Validate configuration invariants.
    fn validate(&self) -> Result<()> {
        if self.num_experts == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "MoeLayerConfig: num_experts must be > 0",
            });
        }
        if self.top_k == 0 || self.top_k > self.num_experts {
            return Err(TensorError::ValueOutOfRange {
                description: "MoeLayerConfig: top_k must be in [1, num_experts]",
            });
        }
        if self.hidden_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "MoeLayerConfig: hidden_size must be > 0",
            });
        }
        if self.expert_intermediate_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "MoeLayerConfig: expert_intermediate_size must be > 0",
            });
        }
        Ok(())
    }

    /// Shared expert intermediate size (falls back to expert_intermediate_size).
    #[must_use]
    pub fn shared_ff_dim(&self) -> usize {
        self.shared_expert_intermediate_size
            .unwrap_or(self.expert_intermediate_size)
    }
}

// -- MoeLayer output ----------------------------------------------------------

/// Output from [`MoeLayer::forward_with_aux`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MoeOutput {
    /// Hidden states after expert dispatch. Same shape as input.
    pub hidden_states: DynTensor,
    /// Auxiliary load-balancing loss (scalar).
    pub aux_loss: DynTensor,
}

impl MoeOutput {
    /// Create a MoE output.
    pub fn new(hidden_states: DynTensor, aux_loss: DynTensor) -> Self {
        Self {
            hidden_states,
            aux_loss,
        }
    }
}

// -- MoeLayer -----------------------------------------------------------------

/// Full Mixture-of-Experts layer with config-driven construction.
///
/// Combines:
/// - Router: `Linear(hidden_size, num_experts)` producing logits
/// - `Vec<ExpertFFN>` for the routed experts
/// - Optional `shared_expert: ExpertFFN` that processes ALL tokens
///
/// Forward pass:
/// 1. Compute router logits, softmax, top-k selection
/// 2. Scatter tokens to assigned experts (loop dispatch)
/// 3. Run each expert's SwiGLU FFN on its assigned tokens
/// 4. Gather weighted results back to original positions
/// 5. Optionally add shared expert output
///
/// Routing indices are transferred to CPU for grouping (O(N*k), small).
/// Expert forward, weighting, and scatter-add stay on the input device.
#[derive(Debug, Clone)]
pub struct MoeLayer {
    router: Linear,
    experts: Vec<ExpertFFN>,
    shared_expert: Option<ExpertFFN>,
    cfg: MoeLayerConfig,
}

impl MoeLayer {
    /// Create from pre-built components.
    pub fn new(
        router: Linear,
        experts: Vec<ExpertFFN>,
        shared_expert: Option<ExpertFFN>,
        cfg: MoeLayerConfig,
    ) -> Result<Self> {
        if experts.len() != cfg.num_experts {
            return Err(TensorError::DataLengthMismatch {
                expected: cfg.num_experts,
                actual: experts.len(),
            });
        }
        if cfg.shared_expert && shared_expert.is_none() {
            return Err(TensorError::ValueOutOfRange {
                description:
                    "MoeLayer: config has shared_expert=true but no shared expert provided",
            });
        }
        Ok(Self {
            router,
            experts,
            shared_expert,
            cfg,
        })
    }

    /// Load from a [`VarBuilder`] using Qwen3 weight naming.
    ///
    /// See module-level docs for weight key layout.
    pub fn load(vb: impl AsRef<VarBuilder>, cfg: MoeLayerConfig) -> Result<Self> {
        let vb = vb.as_ref();
        let router = Linear::load(vb.pp("gate"), cfg.hidden_size, cfg.num_experts)?;
        let experts_vb = vb.pp("experts");
        let experts = (0..cfg.num_experts)
            .map(|e| {
                ExpertFFN::load(
                    experts_vb.pp(e.to_string()),
                    cfg.hidden_size,
                    cfg.expert_intermediate_size,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let shared_expert = if cfg.shared_expert {
            let shared_dim = cfg.shared_ff_dim();
            Some(ExpertFFN::load(
                vb.pp("shared_expert"),
                cfg.hidden_size,
                shared_dim,
            )?)
        } else {
            None
        };
        Self::new(router, experts, shared_expert, cfg)
    }

    /// Access the configuration.
    #[must_use]
    pub fn config(&self) -> &MoeLayerConfig {
        &self.cfg
    }

    /// Access the router linear.
    #[must_use]
    pub fn router(&self) -> &Linear {
        &self.router
    }

    /// Access the experts.
    #[must_use]
    pub fn experts(&self) -> &[ExpertFFN] {
        &self.experts
    }

    /// Access the shared expert, if present.
    #[must_use]
    pub fn shared_expert(&self) -> Option<&ExpertFFN> {
        self.shared_expert.as_ref()
    }

    /// Compute routing: softmax over experts, top-k selection, optional renormalization.
    ///
    /// Input: `[..., D]` where D = `hidden_size`.
    /// Returns `(expert_indices, expert_weights)`:
    /// - `expert_indices`: `[..., K]` (U32)
    /// - `expert_weights`: `[..., K]` (F32)
    fn compute_routing(&self, hidden: &DynTensor) -> Result<(DynTensor, DynTensor)> {
        let logits = self.router.forward(hidden)?;
        let last_dim = logits.rank() - 1;
        let probs = logits.softmax(last_dim)?;
        let (topk_weights, topk_indices) = probs.topk(last_dim, self.cfg.top_k)?;
        let topk_weights = if self.cfg.norm_topk_prob {
            let w_sum = topk_weights.sum_keepdim(last_dim)?;
            topk_weights.broadcast_div(&w_sum)?
        } else {
            topk_weights
        };
        Ok((topk_indices, topk_weights))
    }

    /// Scatter tokens to experts, run FFNs, gather weighted results.
    ///
    /// Tries the fused GPU dispatch first (single kernel for all experts).
    /// Falls back to per-expert loop dispatch when GPU returns `None`.
    fn scatter_gather(
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

        let idx_arr = indices.as_cpu_u32()?;
        let wt_arr = weights.to_f32_array()?;

        let assignments =
            group_tokens_by_expert(&idx_arr, &wt_arr.view(), n_tokens, k, self.cfg.num_experts)?;

        let mut output = DynTensor::zeros(&[n_tokens, model_dim], DType::F32, &device)?;

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
    /// Returns both the output hidden states and a scalar aux loss
    /// (`num_experts * sum_e(f_e * P_e)`) encouraging uniform expert utilization.
    pub fn forward_with_aux(&self, hidden: &DynTensor) -> Result<MoeOutput> {
        let input_dims = hidden.dims();
        let rank = hidden.rank();
        let last_dim = rank - 1;
        let device = hidden.device();

        let n_tokens = crate::tensor::checked_dim_product(&input_dims[..last_dim])?;
        let model_dim = input_dims[last_dim];

        // Full routing probabilities (for aux loss computation).
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

        let flat_hidden = hidden.reshape([n_tokens, model_dim])?;
        let flat_indices = topk_indices.reshape([n_tokens, self.cfg.top_k])?;
        let flat_weights = topk_weights.reshape([n_tokens, self.cfg.top_k])?;

        let mut output = self.scatter_gather(&flat_hidden, &flat_indices, &flat_weights)?;
        output = output.reshape(input_dims)?;

        // Add shared expert output if present.
        if let Some(shared) = &self.shared_expert {
            let shared_out = shared.forward(hidden)?;
            output = output.broadcast_add(&shared_out)?;
        }

        // Compute auxiliary loss.
        let flat_probs = probs.reshape([n_tokens, self.cfg.num_experts])?;
        let aux_loss = compute_aux_loss(
            &flat_indices,
            &flat_probs,
            n_tokens,
            self.cfg.num_experts,
            &device,
        )?;

        check_output_finite(&output, "MoeLayer")?;
        Ok(MoeOutput::new(output, aux_loss))
    }
}

impl Module for MoeLayer {
    /// Standard forward pass (no auxiliary loss).
    ///
    /// Input: `[B, T, D]` or `[T, D]`. Output: same shape.
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

        let mut output = self.scatter_gather(&flat_hidden, &flat_indices, &flat_weights)?;
        output = output.reshape(input_dims)?;

        // Add shared expert output if present.
        if let Some(shared) = &self.shared_expert {
            let shared_out = shared.forward(hidden)?;
            output = output.broadcast_add(&shared_out)?;
        }

        check_output_finite(&output, "MoeLayer")?;
        Ok(output)
    }
}

// -- Internal helpers (pub(super) for reuse by moe_mlp_layer) ----------------

/// Group tokens by expert index in a single O(N*k) pass.
pub(super) fn group_tokens_by_expert(
    idx_arr: &ndarray::ArrayViewD<'_, u32>,
    wt_arr: &ndarray::ArrayViewD<'_, f32>,
    n_tokens: usize,
    k: usize,
    num_experts: usize,
) -> Result<Vec<Vec<(usize, f32)>>> {
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
            let weight = wt_arr[ndarray::IxDyn(coord)];
            assignments[expert_idx].push((t, weight));
        }
    }
    Ok(assignments)
}

/// Run one expert on its assigned tokens and index-add the weighted result.
///
/// Generic over any `Module` implementation (ExpertFFN, ExpertMlp, etc.).
pub(super) fn dispatch_single_expert(
    expert: &dyn Module,
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
                description: "MoeLayer: token index exceeds u32::MAX",
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let weights: Vec<f32> = assignments.iter().map(|&(_, w)| w).collect();

    let ids_tensor = DynTensor::from_vec_u32(token_ids, &[num_routed], device)?;
    let gathered = flat_x.index_select(&ids_tensor, 0)?;
    let expert_out = expert.forward(&gathered)?;

    let w_tensor = DynTensor::from_vec(weights, &[num_routed, 1], device)?;
    let weighted = expert_out.broadcast_mul(&w_tensor)?;

    output.index_add(0, &ids_tensor, &weighted)
}

/// Compute auxiliary load-balancing loss.
///
/// `aux_loss = num_experts * sum_e(f_e * P_e)` where:
/// - `f_e` = fraction of tokens routed to expert `e`
/// - `P_e` = mean routing probability for expert `e`
pub(super) fn compute_aux_loss(
    flat_indices: &DynTensor,
    flat_probs: &DynTensor,
    n_tokens: usize,
    num_experts: usize,
    device: &crate::Device,
) -> Result<DynTensor> {
    if n_tokens == 0 {
        return DynTensor::zeros(&[], DType::F32, device);
    }

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
    let f_e: Vec<f32> = expert_counts
        .iter()
        .map(|&c| c / total_assignments)
        .collect();
    let f_tensor = DynTensor::from_vec(f_e, &[num_experts], device)?;

    let p_e = flat_probs.mean(0)?;
    let fp = f_tensor.broadcast_mul(&p_e)?;
    let sum = fp.sum_all()?;
    let scale = DynTensor::full(&[], num_experts as f64, DType::F32, device)?;
    sum.broadcast_mul(&scale)
}

#[cfg(kani)]
#[path = "moe_layer_kani.rs"]
mod kani_proofs;

#[cfg(test)]
#[path = "moe_layer_tests.rs"]
mod tests;
