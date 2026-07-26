// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Mixture-of-Experts (MoE) routing and dispatch.
//!
//! Provides [`MoeRouter`] for top-k expert selection with softmax routing,
//! [`SwiGluExpert`] for individual expert FFNs, and [`MoeLayer`] for the
//! full MoE forward pass with optional shared expert (Qwen3.5 pattern).
//!
//! Uses loop dispatch (Strategy A): iterate over experts, gather routed
//! tokens, run expert FFN, scatter-add back. Correct for inference.
//! Grouped GEMM (Strategy B) deferred to nn-metal Phase 5.

use super::{check_output_finite, Linear, Module, SwiGlu};
use crate::dyn_tensor::{gpu_backend_dispatch, DynTensor};
use crate::var_builder::VarBuilder;
use crate::{DType, Result, TensorError};

// -- Routing output -----------------------------------------------------------

/// Output from [`MoeRouter::forward`]: selected expert weights and indices.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MoeRoutingOutput {
    /// Top-k routing weights per token, renormalized to sum to 1.
    /// Shape: `[..., k]` where leading dims match input.
    pub weights: DynTensor,
    /// Top-k expert indices per token (U32 dtype).
    /// Shape: `[..., k]` matching `weights`.
    pub indices: DynTensor,
}

impl MoeRoutingOutput {
    /// Create a routing output from weights and expert indices.
    pub fn new(weights: DynTensor, indices: DynTensor) -> Self {
        Self { weights, indices }
    }
}

// -- MoeRouter ----------------------------------------------------------------

/// Softmax top-k expert router.
///
/// Computes routing logits via a linear projection, applies softmax over
/// all experts, selects top-k, and renormalizes selected weights.
#[derive(Debug, Clone)]
pub struct MoeRouter {
    gate: Linear,
    num_experts: usize,
    top_k: usize,
}

impl MoeRouter {
    /// Create a router from a pre-built gate linear and parameters.
    ///
    /// `gate` projects from `model_dim` to `num_experts`.
    pub fn new(gate: Linear, num_experts: usize, top_k: usize) -> Result<Self> {
        if top_k == 0 || top_k > num_experts {
            return Err(TensorError::ValueOutOfRange {
                description: "MoeRouter: top_k must be in [1, num_experts]",
            });
        }
        Ok(Self {
            gate,
            num_experts,
            top_k,
        })
    }

    /// Number of experts this router selects per token.
    #[must_use]
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Total number of experts.
    #[must_use]
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Route input tokens to experts.
    ///
    /// Input shape: `[B, T, D]` or `[T, D]`.
    /// Returns routing weights and expert indices, both with last dim = `top_k`.
    pub fn forward(&self, x: &DynTensor) -> Result<MoeRoutingOutput> {
        let logits = self.gate.forward(x)?; // [..., num_experts]
        let last_dim = logits.rank() - 1;
        let weights = logits.softmax(last_dim)?; // [..., num_experts]
        let (topk_w, topk_idx) = weights.topk(last_dim, self.top_k)?; // [..., k]
                                                                      // Renormalize: weights / sum(weights, keepdim=true)
        let w_sum = topk_w.sum_keepdim(last_dim)?;
        let topk_w = topk_w.broadcast_div(&w_sum)?;
        check_output_finite(&topk_w, "MoeRouter")?;
        Ok(MoeRoutingOutput {
            weights: topk_w,
            indices: topk_idx,
        })
    }
}

// -- SwiGluExpert -------------------------------------------------------------

/// Individual expert FFN: SwiGLU(gate_proj, up_proj) @ down_proj.
///
/// Wraps [`SwiGlu`] for clarity in the MoE context.
#[derive(Debug, Clone)]
pub struct SwiGluExpert {
    ffn: SwiGlu,
}

impl SwiGluExpert {
    /// Create from pre-built projections.
    pub fn new(gate_proj: Linear, up_proj: Linear, down_proj: Linear) -> Result<Self> {
        Ok(Self {
            ffn: SwiGlu::new(gate_proj, up_proj, down_proj)?,
        })
    }

    /// Load expert weights from a VarBuilder.
    ///
    /// Expects keys: `"gate_proj.weight"`, `"up_proj.weight"`, `"down_proj.weight"`.
    pub fn load(vb: impl AsRef<VarBuilder>, model_dim: usize, ff_dim: usize) -> Result<Self> {
        let vb = vb.as_ref();
        let gate_proj = Linear::load(vb.pp("gate_proj"), model_dim, ff_dim)?;
        let up_proj = Linear::load(vb.pp("up_proj"), model_dim, ff_dim)?;
        let down_proj = Linear::load(vb.pp("down_proj"), ff_dim, model_dim)?;
        Self::new(gate_proj, up_proj, down_proj)
    }

    /// Access the gate projection weight.
    #[must_use]
    pub fn gate_proj(&self) -> &Linear {
        self.ffn.w_gate()
    }

    /// Access the up projection weight.
    #[must_use]
    pub fn up_proj(&self) -> &Linear {
        self.ffn.w_up()
    }

    /// Access the down projection weight.
    #[must_use]
    pub fn down_proj(&self) -> &Linear {
        self.ffn.w_down()
    }
}

impl Module for SwiGluExpert {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        self.ffn.forward(x)
    }
}

// -- MoeLayer -----------------------------------------------------------------

/// Full MoE FFN layer with loop dispatch (Strategy A).
///
/// Forward pass:
/// 1. Route tokens to top-k experts via softmax + topk
/// 2. For each expert: gather routed tokens, run SwiGLU FFN, scatter-add back
/// 3. Add shared expert output if present (Qwen3.5 pattern)
#[derive(Debug, Clone)]
#[allow(dead_code)] // Superseded by MoeLayer in moe_layer.rs; retained for reference.
pub(super) struct MoeLayer {
    router: MoeRouter,
    experts: Vec<SwiGluExpert>,
    shared_expert: Option<SwiGluExpert>,
}

#[allow(dead_code)] // Superseded by MoeLayer in moe_layer.rs; retained for reference.
impl MoeLayer {
    /// Create from pre-built components.
    pub(super) fn new(
        router: MoeRouter,
        experts: Vec<SwiGluExpert>,
        shared_expert: Option<SwiGluExpert>,
    ) -> Result<Self> {
        if experts.len() != router.num_experts() {
            return Err(TensorError::DataLengthMismatch {
                expected: router.num_experts(),
                actual: experts.len(),
            });
        }
        Ok(Self {
            router,
            experts,
            shared_expert,
        })
    }

    /// Load a full MoE layer from a VarBuilder.
    ///
    /// Weight key layout (Qwen3.5 convention):
    /// ```text
    /// gate.weight                         → router Linear
    /// experts.{e}.gate_proj.weight        → expert gate
    /// experts.{e}.up_proj.weight          → expert up
    /// experts.{e}.down_proj.weight        → expert down
    /// shared_expert.gate_proj.weight      → shared expert (optional)
    /// ```
    pub(super) fn load(
        vb: impl AsRef<VarBuilder>,
        model_dim: usize,
        ff_dim: usize,
        num_experts: usize,
        top_k: usize,
        has_shared_expert: bool,
    ) -> Result<Self> {
        Self::load_with_shared_dim(
            vb.as_ref(),
            model_dim,
            ff_dim,
            num_experts,
            top_k,
            has_shared_expert,
            None,
        )
    }

    /// Load a full MoE layer with an optional separate shared expert dimension.
    ///
    /// When `shared_ff_dim` is `Some(dim)`, the shared expert uses `dim` as its
    /// intermediate size instead of `ff_dim`. This supports models like Qwen3-MoE
    /// where the shared expert has a different intermediate size than regular experts.
    pub(super) fn load_with_shared_dim(
        vb: impl AsRef<VarBuilder>,
        model_dim: usize,
        ff_dim: usize,
        num_experts: usize,
        top_k: usize,
        has_shared_expert: bool,
        shared_ff_dim: Option<usize>,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let gate = Linear::load(vb.pp("gate"), model_dim, num_experts)?;
        let router = MoeRouter::new(gate, num_experts, top_k)?;
        let experts_vb = vb.pp("experts");
        let experts = (0..num_experts)
            .map(|e| {
                let evb = experts_vb.pp(e.to_string());
                SwiGluExpert::load(&evb, model_dim, ff_dim)
            })
            .collect::<Result<Vec<_>>>()?;
        let shared_expert = if has_shared_expert {
            let shared_dim = shared_ff_dim.unwrap_or(ff_dim);
            Some(SwiGluExpert::load(
                vb.pp("shared_expert"),
                model_dim,
                shared_dim,
            )?)
        } else {
            None
        };
        Self::new(router, experts, shared_expert)
    }

    /// Access the router.
    #[must_use]
    pub(super) fn router(&self) -> &MoeRouter {
        &self.router
    }
}

/// Group tokens by expert index in a single O(N*k) pass.
///
/// Returns a vec of per-expert `(token_id, weight)` lists. Validates that
/// all routing indices are in `[0, num_experts)`.
#[allow(dead_code)] // Retained for reference; active dispatch in moe_dispatch.rs.
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

/// Run one expert on its assigned tokens and index-add the weighted result
/// into the output accumulator. All tensor ops stay on the input device.
#[allow(dead_code)] // Retained for reference; active dispatch in moe_dispatch.rs.
fn dispatch_expert(
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
                description: "MoE: token index exceeds u32::MAX",
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

    // index_add uses 1D row indices [num_routed] instead of expanded 2D
    // scatter indices [num_routed, D], eliminating O(N*D) u32 allocation.
    output.index_add(0, &ids_tensor, &weighted)
}

impl Module for MoeLayer {
    /// MoE forward pass with loop dispatch.
    ///
    /// Tries fused GPU dispatch first (#3547), falls back to per-expert loop.
    ///
    /// Input: `[B, T, D]`. Output: same shape.
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let routing = self.router.forward(x)?;
        let x_dims = x.dims();
        let rank = x.rank();
        let last_dim = rank - 1;
        let k = self.router.top_k();

        // Flatten to [N, D] where N = product of all leading dims.
        let n_tokens = crate::tensor::checked_dim_product(&x_dims[..last_dim])?;
        let model_dim = x_dims[last_dim];
        let input_device = x.device();
        let flat_x = x.reshape([n_tokens, model_dim])?;

        // Flatten routing to [N, k] so 2D indexing works regardless of input rank.
        let flat_indices = routing.indices.reshape([n_tokens, k])?;
        let flat_weights = routing.weights.reshape([n_tokens, k])?;

        // Try fused GPU dispatch first (#3547).
        let gpu_result = if flat_x.device().is_gpu() {
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
            gpu_backend_dispatch(|b| {
                b.moe_scatter_gather(
                    &flat_x,
                    &flat_indices,
                    &flat_weights,
                    &gate_ws,
                    &up_ws,
                    &down_ws,
                    self.router.num_experts(),
                )
            })
        } else {
            None
        };

        let mut output = if let Some(result) = gpu_result {
            result?
        } else {
            // Fallback: per-expert loop dispatch.
            let idx_arr = flat_indices.as_cpu_u32()?;
            let wt_arr = flat_weights.to_f32_array()?;

            let num_experts = self.router.num_experts();
            let expert_assignments =
                group_tokens_by_expert(&idx_arr, &wt_arr.view(), n_tokens, k, num_experts)?;

            let mut acc = DynTensor::zeros(&[n_tokens, model_dim], DType::F32, &input_device)?;

            for (expert_idx, assignments) in expert_assignments.iter().enumerate() {
                if !assignments.is_empty() {
                    acc = dispatch_expert(
                        &self.experts[expert_idx],
                        &flat_x,
                        &acc,
                        assignments,
                        &input_device,
                    )?;
                }
            }
            acc
        };

        output = output.reshape(x_dims)?;

        // Add shared expert output if present
        if let Some(shared) = &self.shared_expert {
            let shared_out = shared.forward(x)?;
            output = output.broadcast_add(&shared_out)?;
        }
        check_output_finite(&output, "MoeLayer")?;
        Ok(output)
    }
}

#[cfg(kani)]
#[path = "moe_kani.rs"]
mod kani_proofs;

#[cfg(kani)]
#[path = "moe_kani_routing.rs"]
mod kani_routing_proofs;

#[cfg(test)]
#[path = "moe_tests.rs"]
mod tests;
