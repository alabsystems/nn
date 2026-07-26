// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MoE layer variant using generic [`ExpertMlp`] experts (configurable activation).
//!
//! [`MoeMlpLayer`] is the counterpart of [`super::MoeLayer`] (which uses SwiGLU
//! [`super::ExpertFFN`] experts). This variant supports any activation function
//! via the [`Activation`] enum, making it suitable for Mixtral-style architectures
//! that use standard 2-projection MLPs with GELU, ReLU, SiLU, etc.
//!
//! Also provides [`ExpertMlp`]: a generic two-layer MLP expert with configurable
//! activation, the per-expert computation unit for `MoeMlpLayer`.
//!
//! # Forward pass
//!
//! 1. Router logits: `Linear(hidden_states, num_experts)` -> softmax -> top-k
//! 2. Scatter: route tokens to assigned experts based on top-k indices
//! 3. Expert MLP: `down_proj(activation(up_proj(x)))` for each expert's tokens
//! 4. Gather: weighted sum of expert outputs based on routing weights
//!
//! Reuses the scatter-gather helpers from [`super::moe_layer`].

use super::moe_layer::{compute_aux_loss, dispatch_single_expert, group_tokens_by_expert};
use super::{check_output_finite, Activation, Linear, Module};
use crate::dyn_tensor::DynTensor;
use crate::var_builder::VarBuilder;
use crate::{DType, Result, TensorError};

use super::moe_layer::MoeOutput;

// -- ExpertMlp ----------------------------------------------------------------

/// Generic two-layer expert FFN: `down_proj(activation(up_proj(x)))`.
///
/// Unlike [`super::ExpertFFN`] which uses the SwiGLU gated pattern (3 projections),
/// `ExpertMlp` is the standard MLP pattern used in Mixtral and many other
/// MoE architectures: a single up-projection, an element-wise activation,
/// and a down-projection back to model dimension.
///
/// The activation is configurable via the [`Activation`] enum (ReLU, GELU,
/// SiLU, etc.).
#[derive(Debug, Clone)]
pub struct ExpertMlp {
    up_proj: Linear,
    down_proj: Linear,
    activation: Activation,
}

impl ExpertMlp {
    /// Create from pre-built linear projections and an activation function.
    ///
    /// - `up_proj`: `[hidden_size, intermediate_size]` (expands to FFN dimension)
    /// - `down_proj`: `[intermediate_size, hidden_size]` (projects back)
    /// - `activation`: element-wise activation between projections
    pub fn new(up_proj: Linear, down_proj: Linear, activation: Activation) -> Result<Self> {
        let up_out = up_proj.weight().dim(0)?;
        let down_in = down_proj.weight().dim(1)?;
        if up_out != down_in {
            return Err(TensorError::shape_mismatch(vec![up_out], vec![down_in]));
        }
        Ok(Self {
            up_proj,
            down_proj,
            activation,
        })
    }

    /// Load expert weights from a [`VarBuilder`].
    ///
    /// Expects keys: `up_proj.weight`, `down_proj.weight` under the given
    /// prefix. Optional bias tensors are loaded if present.
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        hidden_size: usize,
        intermediate_size: usize,
        activation: Activation,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let up_proj = Linear::load(vb.pp("up_proj"), hidden_size, intermediate_size)?;
        let down_proj = Linear::load(vb.pp("down_proj"), intermediate_size, hidden_size)?;
        Self::new(up_proj, down_proj, activation)
    }

    /// Access the up projection.
    #[must_use]
    pub fn up_proj(&self) -> &Linear {
        &self.up_proj
    }

    /// Access the down projection.
    #[must_use]
    pub fn down_proj(&self) -> &Linear {
        &self.down_proj
    }

    /// Access the activation function.
    #[must_use]
    pub fn activation(&self) -> Activation {
        self.activation
    }
}

impl Module for ExpertMlp {
    /// Forward: `down_proj(activation(up_proj(x)))`.
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let h = self.up_proj.forward(x)?;
        let h = self.activation.forward(&h)?;
        self.down_proj.forward(&h)
    }
}

// -- MoeMlpConfig ------------------------------------------------------------

/// Configuration for [`MoeMlpLayer`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MoeMlpConfig {
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
    /// Activation function for each expert MLP.
    pub activation: Activation,
}

impl MoeMlpConfig {
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
        activation: Activation,
    ) -> Result<Self> {
        let cfg = Self {
            num_experts,
            top_k,
            hidden_size,
            expert_intermediate_size,
            norm_topk_prob,
            activation,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if self.num_experts == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "MoeMlpConfig: num_experts must be > 0",
            });
        }
        if self.top_k == 0 || self.top_k > self.num_experts {
            return Err(TensorError::ValueOutOfRange {
                description: "MoeMlpConfig: top_k must be in [1, num_experts]",
            });
        }
        if self.hidden_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "MoeMlpConfig: hidden_size must be > 0",
            });
        }
        if self.expert_intermediate_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "MoeMlpConfig: expert_intermediate_size must be > 0",
            });
        }
        Ok(())
    }
}

// -- MoeMlpLayer -------------------------------------------------------------

/// MoE layer using generic 2-projection [`ExpertMlp`] experts.
///
/// Unlike [`super::MoeLayer`] which uses SwiGLU (3-projection) experts,
/// this variant uses `down_proj(activation(up_proj(x)))` per expert, with
/// the activation configurable via [`Activation`].
///
/// Forward pass:
/// 1. Compute router logits, softmax, top-k selection
/// 2. Scatter tokens to assigned experts (loop dispatch)
/// 3. Run each expert's MLP on its assigned tokens
/// 4. Gather weighted results back to original positions
///
/// Routing indices are transferred to CPU for grouping (O(N*k), small).
/// Expert forward, weighting, and scatter-add stay on the input device.
#[derive(Debug, Clone)]
pub struct MoeMlpLayer {
    router: Linear,
    experts: Vec<ExpertMlp>,
    cfg: MoeMlpConfig,
}

impl MoeMlpLayer {
    /// Create from pre-built components.
    pub fn new(router: Linear, experts: Vec<ExpertMlp>, cfg: MoeMlpConfig) -> Result<Self> {
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

    /// Load from a [`VarBuilder`] using standard MoE weight naming.
    ///
    /// Weight keys:
    /// ```text
    /// gate.weight                     -> router Linear
    /// experts.{i}.up_proj.weight      -> expert up projection
    /// experts.{i}.down_proj.weight    -> expert down projection
    /// ```
    pub fn load(vb: impl AsRef<VarBuilder>, cfg: MoeMlpConfig) -> Result<Self> {
        let vb = vb.as_ref();
        let router = Linear::load(vb.pp("gate"), cfg.hidden_size, cfg.num_experts)?;
        let experts_vb = vb.pp("experts");
        let experts = (0..cfg.num_experts)
            .map(|e| {
                ExpertMlp::load(
                    experts_vb.pp(e.to_string()),
                    cfg.hidden_size,
                    cfg.expert_intermediate_size,
                    cfg.activation,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Self::new(router, experts, cfg)
    }

    /// Access the configuration.
    #[must_use]
    pub fn config(&self) -> &MoeMlpConfig {
        &self.cfg
    }

    /// Access the router linear.
    #[must_use]
    pub fn router(&self) -> &Linear {
        &self.router
    }

    /// Access the experts.
    #[must_use]
    pub fn experts(&self) -> &[ExpertMlp] {
        &self.experts
    }

    /// Compute routing: softmax over experts, top-k selection, optional renorm.
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

    /// Scatter tokens to experts, run MLPs, gather weighted results.
    fn scatter_gather(
        &self,
        hidden: &DynTensor,
        indices: &DynTensor,
        weights: &DynTensor,
    ) -> Result<DynTensor> {
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
    /// (`num_experts * sum_e(f_e * P_e)`) encouraging uniform utilization.
    pub fn forward_with_aux(&self, hidden: &DynTensor) -> Result<MoeOutput> {
        let input_dims = hidden.dims();
        let rank = hidden.rank();
        let last_dim = rank - 1;
        let device = hidden.device();

        let n_tokens = crate::tensor::checked_dim_product(&input_dims[..last_dim])?;
        let model_dim = input_dims[last_dim];

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

        let output = self.scatter_gather(&flat_hidden, &flat_indices, &flat_weights)?;
        let output = output.reshape(input_dims)?;

        let flat_probs = probs.reshape([n_tokens, self.cfg.num_experts])?;
        let aux_loss = compute_aux_loss(
            &flat_indices,
            &flat_probs,
            n_tokens,
            self.cfg.num_experts,
            &device,
        )?;

        check_output_finite(&output, "MoeMlpLayer")?;
        Ok(MoeOutput::new(output, aux_loss))
    }
}

impl Module for MoeMlpLayer {
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

        let output = self.scatter_gather(&flat_hidden, &flat_indices, &flat_weights)?;
        let output = output.reshape(input_dims)?;

        check_output_finite(&output, "MoeMlpLayer")?;
        Ok(output)
    }
}

#[cfg(test)]
#[path = "moe_mlp_layer_tests.rs"]
mod tests;
