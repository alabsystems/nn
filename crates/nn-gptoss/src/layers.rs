// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Internal decoder layer components: Attention (GQA with bias), MoE FFN
//! (fused clamped SwiGLU experts), and DecoderLayer with alternating attention.
//!
//! Architecture matches actual Context-1 weights:
//! - 64 Q heads x 64 head_dim = 4096 (attention dim > hidden_size=2880)
//! - 8 KV heads x 64 head_dim = 512
//! - Fused expert tensors: `[num_experts, hidden, fused_dim]`
//! - Router with bias
//! - Per-layer attention sinks `[head_dim]`

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::kv_cache::KvCacheLayer;
use nn_core::layers::{
    self, check_output_finite, repeat_kv, Linear, Module, RmsNorm, RotaryEmbedding,
};
use nn_core::var_builder::VarBuilder;
use nn_core::Result;

use crate::config::{GptOssConfig, LayerType};

// -- GQA Attention with bias --------------------------------------------------

pub(crate) struct GptOssAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// StreamingLLM-style attention sinks, shape `[head_dim]`.
    /// Loaded from `self_attn.sinks`. Application in attention scores deferred
    /// pending reference implementation verification.
    // TODO: Apply sinks to attention scores for first token position.
    // The exact mechanism (additive bias on score[:, :, :, 0]?) needs
    // verification against the Chroma reference code.
    #[allow(dead_code)]
    sinks: DynTensor,
}

impl GptOssAttention {
    pub(crate) fn load(vb: impl AsRef<VarBuilder>, cfg: &GptOssConfig) -> Result<Self> {
        let vb = vb.as_ref();
        let h = cfg.hidden_size;
        let hd = cfg.head_dim;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let attn_dim = nh * hd; // 64 * 64 = 4096
        let kv_dim = nkv * hd; // 8 * 64 = 512

        // gpt-oss has attention_bias=true: Q/K/V/O projections include bias.
        let load_proj = |prefix: &str, out_dim: usize, in_dim: usize| -> Result<Linear> {
            let w = vb.get(&[out_dim, in_dim], &format!("{prefix}.weight"))?;
            let b = if cfg.attention_bias {
                Some(vb.get(&[out_dim], &format!("{prefix}.bias"))?)
            } else {
                None
            };
            Linear::new(w, b)
        };

        // Q: [attn_dim, hidden] = [4096, 2880]
        // K: [kv_dim, hidden]   = [512, 2880]
        // V: [kv_dim, hidden]   = [512, 2880]
        // O: [hidden, attn_dim] = [2880, 4096]
        let q_proj = load_proj("q_proj", attn_dim, h)?;
        let k_proj = load_proj("k_proj", kv_dim, h)?;
        let v_proj = load_proj("v_proj", kv_dim, h)?;
        let o_proj = load_proj("o_proj", h, attn_dim)?;

        // Attention sinks: [head_dim] per layer
        let sinks = vb.get(&[hd], "sinks")?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            sinks,
        })
    }

    pub(crate) fn forward(
        &self,
        x: &DynTensor,
        rope: &RotaryEmbedding,
        positions: &[usize],
        mask: Option<&DynTensor>,
        cache: Option<&mut KvCacheLayer>,
    ) -> Result<DynTensor> {
        let (batch, seq_len, _) = x.dims3()?;

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [batch, heads, seq, head_dim]
        let q = q
            .reshape([batch, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = k
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = v
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])?
            .transpose(1, 2)?;

        // Apply RoPE (half-split / rotate_half convention, matching HuggingFace)
        let (q, k) = rope.apply_pair_half_split(&q, &k, positions)?;

        // KV cache
        let (k, v) = match cache {
            Some(c) => c.append(&k, &v)?,
            None => (k, v),
        };

        // GQA: repeat K, V to match Q heads.
        // 64 Q heads / 8 KV heads = 8 groups (cleanly divisible).
        let repeat_factor = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, repeat_factor)?;
        let v = repeat_kv(&v, repeat_factor)?;

        // Scaled dot-product attention.
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let s_kv = k.dim(2)?;
        let attn_output = if mask.is_some() && seq_len == s_kv {
            layers::sdpa_causal(&q, &k, &v, scale)?
        } else {
            layers::sdpa(&q, &k, &v, mask, scale)?
        };

        // Reshape back: [batch, seq, num_heads * head_dim] = [batch, seq, 4096]
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape([
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ])?;

        // O projection maps from attn_dim (4096) back to hidden_size (2880)
        let out = self.o_proj.forward(&attn_output)?;
        check_output_finite(&out, "GptOssAttention")?;
        Ok(out)
    }
}

// -- Fused MoE Block ----------------------------------------------------------

/// MoE FFN block with fused expert weights and router with bias.
///
/// Actual Context-1 weight layout (fused, NOT per-expert separate):
/// ```text
/// experts.gate_up_proj:      [num_experts, hidden_size, 2*intermediate_size]
/// experts.gate_up_proj_bias: [num_experts, 2*intermediate_size]
/// experts.down_proj:         [num_experts, hidden_size, hidden_size]
/// experts.down_proj_bias:    [num_experts, hidden_size]
/// router.weight:             [num_experts, hidden_size]
/// router.bias:               [num_experts]
/// ```
///
/// For each selected expert, we slice into the fused tensors and apply
/// clamped SwiGLU: `down(clamp(silu(gate(x)), -L, L) * up(x)) + down_bias`.
pub(crate) struct GptOssMoeBlock {
    router: Linear,
    gate_up_proj: DynTensor,
    gate_up_proj_bias: DynTensor,
    down_proj: DynTensor,
    down_proj_bias: DynTensor,
    num_experts: usize,
    top_k: usize,
    swiglu_limit: f64,
    intermediate_size: usize,
}

impl GptOssMoeBlock {
    pub(crate) fn load(vb: impl AsRef<VarBuilder>, cfg: &GptOssConfig) -> Result<Self> {
        let vb = vb.as_ref();
        let ne = cfg.num_local_experts;
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let fused_dim = 2 * inter; // gate + up fused along last dim

        // Router with bias
        let router = Linear::new(
            vb.get(&[ne, h], "router.weight")?,
            Some(vb.get(&[ne], "router.bias")?),
        )?;

        // Fused expert weights
        let experts_vb = vb.pp("experts");
        let gate_up_proj = experts_vb.get(&[ne, h, fused_dim], "gate_up_proj")?;
        let gate_up_proj_bias = experts_vb.get(&[ne, fused_dim], "gate_up_proj_bias")?;
        let down_proj = experts_vb.get(&[ne, h, h], "down_proj")?;
        let down_proj_bias = experts_vb.get(&[ne, h], "down_proj_bias")?;

        Ok(Self {
            router,
            gate_up_proj,
            gate_up_proj_bias,
            down_proj,
            down_proj_bias,
            num_experts: ne,
            top_k: cfg.experts_per_token,
            swiglu_limit: cfg.swiglu_limit,
            intermediate_size: inter,
        })
    }

    /// Run one expert on a batch of tokens.
    ///
    /// Slices into fused weight tensors for expert `expert_idx`, applies
    /// clamped SwiGLU FFN: `down(clamp(silu(gate(x)), -L, L) * up(x)) + bias`.
    fn expert_forward(&self, x: &DynTensor, expert_idx: usize) -> Result<DynTensor> {
        let inter = self.intermediate_size;

        // Slice fused gate_up_proj for this expert: [hidden_size, 2*inter]
        // Weight convention is [in_features, out_features] for fused experts.
        let gate_up_w = self.gate_up_proj.narrow(0, expert_idx, 1)?.squeeze(0)?; // [hidden_size, 2*inter]
        let gate_up_b = self
            .gate_up_proj_bias
            .narrow(0, expert_idx, 1)?
            .squeeze(0)?; // [2*inter]

        // x @ gate_up_w + gate_up_b  →  [N, 2*inter]
        // Weight is [in=2880, out=5760], so no transpose needed.
        let gate_up = x.matmul(&gate_up_w)?.broadcast_add(&gate_up_b)?;

        // Split along last dim: gate = [:, :inter], up = [:, inter:]
        let last = gate_up.rank() - 1;
        let gate = gate_up.narrow(last, 0, inter)?;
        let up = gate_up.narrow(last, inter, inter)?;

        // Clamped SwiGLU: clamp(silu(gate), -limit, limit) * up
        let gate = gate.silu()?.clamp(-self.swiglu_limit, self.swiglu_limit)?;
        let hidden = gate.broadcast_mul(&up)?;

        // Down projection: [hidden_size, hidden_size]
        // Weight is [in=inter, out=hidden], stored as [hidden, hidden] since inter==hidden.
        let down_w = self.down_proj.narrow(0, expert_idx, 1)?.squeeze(0)?; // [hidden_size, hidden_size]
        let down_b = self.down_proj_bias.narrow(0, expert_idx, 1)?.squeeze(0)?; // [hidden_size]

        let out = hidden.matmul(&down_w)?.broadcast_add(&down_b)?;
        check_output_finite(&out, "GptOssMoeBlock::expert_forward")?;
        Ok(out)
    }

    pub(crate) fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        // Use fused dispatch on GPU to reduce Metal dispatch count.
        // Sequential per-expert loop issues O(num_experts * 6) dispatches;
        // fused path batches to O(4) dispatches per MoE block.
        if crate::moe_dispatch::should_use_fused_dispatch(&x.device()) {
            let router_bias =
                self.router
                    .bias()
                    .ok_or_else(|| nn_core::TensorError::ValueOutOfRange {
                        description: "MoE router requires bias",
                    })?;
            return crate::moe_dispatch::fused_moe_forward(
                x,
                self.router.weight(),
                router_bias,
                &self.gate_up_proj,
                &self.gate_up_proj_bias,
                &self.down_proj,
                &self.down_proj_bias,
                self.top_k,
                self.swiglu_limit,
            );
        }

        let x_dims = x.dims();
        let rank = x.rank();
        let last_dim = rank - 1;

        // Router: softmax over experts, top-k selection
        let logits = self.router.forward(x)?;
        let logits_last = logits.rank() - 1;
        let probs = logits.softmax(logits_last)?;
        let (topk_weights, topk_indices) = probs.topk(logits_last, self.top_k)?;

        // Renormalize weights
        let w_sum = topk_weights.sum_keepdim(logits_last)?;
        let topk_weights = topk_weights.broadcast_div(&w_sum)?;

        // Flatten to [N, D]
        let n_tokens = nn_core::tensor::checked_dim_product(&x_dims[..last_dim])?;
        let model_dim = x_dims[last_dim];
        let flat_x = x.reshape([n_tokens, model_dim])?;
        let flat_indices = topk_indices.reshape([n_tokens, self.top_k])?;
        let flat_weights = topk_weights.reshape([n_tokens, self.top_k])?;

        // Per-expert loop dispatch
        let device = x.device();
        let idx_flat = flat_indices.to_flat_vec::<u32>()?;
        let wt_flat = flat_weights.to_flat_vec::<f32>()?;

        let mut output = DynTensor::zeros(&[n_tokens, model_dim], nn_core::DType::F32, &device)?;

        // Group tokens by expert using flat vectors (row-major [N, k])
        let avg_per_expert = (n_tokens * self.top_k) / self.num_experts.max(1) + 1;
        let mut assignments: Vec<Vec<(usize, f32)>> = (0..self.num_experts)
            .map(|_| Vec::with_capacity(avg_per_expert))
            .collect();
        for t in 0..n_tokens {
            for s in 0..self.top_k {
                let flat_idx = t * self.top_k + s;
                let expert_idx = idx_flat[flat_idx] as usize;
                if expert_idx < self.num_experts {
                    let weight = wt_flat[flat_idx];
                    assignments[expert_idx].push((t, weight));
                }
            }
        }

        // Dispatch each expert via fused weight slicing
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

            let ids_tensor = DynTensor::from_vec_u32(token_ids, &[num_routed], &device)?;
            let gathered = flat_x.index_select(&ids_tensor, 0)?;
            let expert_out = self.expert_forward(&gathered, expert_idx)?;

            let w_tensor = DynTensor::from_vec(weights, &[num_routed, 1], &device)?;
            let weighted = expert_out.broadcast_mul(&w_tensor)?;

            output = output.index_add(0, &ids_tensor, &weighted)?;
        }

        output = output.reshape(x_dims)?;
        check_output_finite(&output, "GptOssMoeBlock")?;
        Ok(output)
    }
}

// -- Decoder Layer ------------------------------------------------------------

/// One gpt-oss decoder layer: input_layernorm -> attention -> residual ->
/// post_attention_layernorm -> MoE -> residual.
///
/// The `layer_type` determines the attention mask:
/// - `SlidingAttention` uses a sliding window mask
/// - `FullAttention` uses the standard causal mask passed from the model
pub(crate) struct GptOssDecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: GptOssAttention,
    post_attention_layernorm: RmsNorm,
    moe: GptOssMoeBlock,
    layer_type: LayerType,
}

impl GptOssDecoderLayer {
    pub(crate) fn load(
        vb: impl AsRef<VarBuilder>,
        cfg: &GptOssConfig,
        layer_type: LayerType,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let input_layernorm = RmsNorm::new(
            vb.get(&[cfg.hidden_size], "input_layernorm.weight")?,
            cfg.rms_norm_eps,
        )?;
        let self_attn = GptOssAttention::load(vb.pp("self_attn"), cfg)?;
        let post_attention_layernorm = RmsNorm::new(
            vb.get(&[cfg.hidden_size], "post_attention_layernorm.weight")?,
            cfg.rms_norm_eps,
        )?;
        let moe = GptOssMoeBlock::load(vb.pp("mlp"), cfg)?;

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            moe,
            layer_type,
        })
    }

    /// Forward pass for one decoder layer.
    ///
    /// `causal_mask` is the full causal mask from the model. For sliding
    /// attention layers, a combined causal+sliding mask is used instead.
    pub(crate) fn forward(
        &self,
        x: &DynTensor,
        rope: &RotaryEmbedding,
        positions: &[usize],
        causal_mask: Option<&DynTensor>,
        sliding_mask: Option<&DynTensor>,
        cache: Option<&mut KvCacheLayer>,
    ) -> Result<DynTensor> {
        // Select attention mask based on layer type
        let mask = match self.layer_type {
            LayerType::SlidingAttention => sliding_mask,
            LayerType::FullAttention => causal_mask,
        };

        // Pre-norm attention with residual
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, rope, positions, mask, cache)?;
        let x = residual.broadcast_add(&x)?;

        // Pre-norm MoE with residual
        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.moe.forward(&x)?;
        let output = residual.broadcast_add(&x)?;
        check_output_finite(&output, "GptOssDecoderLayer")?;
        Ok(output)
    }
}

#[cfg(kani)]
#[path = "kani_layers_proofs.rs"]
mod kani_layers_proofs;
