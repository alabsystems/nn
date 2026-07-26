// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Qwen3 MoE (Mixture-of-Experts) decoder-only transformer.
//!
//! Drop-in replacement for `Qwen3Model` that uses [`MoeLayer`] instead of
//! the dense SwiGLU MLP. Supports GQA, QK-Norm, shared expert (Qwen3.5 pattern).
//!
//! Qwen3 MoE variants: Qwen3-30B-A3B (128 experts, 8 active),
//! Qwen3-235B-A22B (128 experts, 8 active).

use crate::forward_common::{
    build_rope, embed_and_unsqueeze, forward_to_logits, forward_to_logits_and_hidden,
    validate_embedding_input, validate_forward_input, DecoderLayer,
};
use crate::layers::Qwen3Attention;
use crate::Qwen3Config;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::kv_cache::{KvCache, KvCacheLayer};
use nn_core::layers::{
    check_output_finite, Embedding, Linear, Module, MoeLayer, MoeLayerConfig, RmsNorm,
    RotaryEmbedding,
};
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Result};

use crate::error::Qwen3Error;

// -- Config -------------------------------------------------------------------

/// Qwen3 MoE model configuration.
///
/// Extends [`Qwen3Config`] with MoE-specific parameters.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Qwen3MoeConfig {
    /// Base transformer config (hidden_size, num_heads, etc.).
    pub base: Qwen3Config,
    /// Total number of experts per MoE layer.
    pub num_experts: usize,
    /// Number of experts active per token (top-k).
    pub num_experts_per_tok: usize,
    /// Whether to include a shared expert (Qwen3.5 pattern).
    pub shared_expert: bool,
    /// Intermediate size for the shared expert (may differ from base).
    /// If `None`, uses `base.intermediate_size`.
    pub shared_expert_intermediate_size: Option<usize>,
}

impl Qwen3MoeConfig {
    /// Create a new Qwen3 MoE configuration.
    ///
    /// Required because `#[non_exhaustive]` prevents struct literal construction
    /// outside this crate.
    #[must_use]
    pub fn new(
        base: Qwen3Config,
        num_experts: usize,
        num_experts_per_tok: usize,
        shared_expert: bool,
        shared_expert_intermediate_size: Option<usize>,
    ) -> Self {
        Self {
            base,
            num_experts,
            num_experts_per_tok,
            shared_expert,
            shared_expert_intermediate_size,
        }
    }

    /// Validate configuration invariants.
    pub fn validate(&self) -> Result<()> {
        self.base.validate()?;
        if self.num_experts == 0 {
            return Err(Qwen3Error::InvalidConfig {
                reason: "num_experts must be > 0".into(),
            }
            .into());
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_experts {
            return Err(Qwen3Error::InvalidConfig {
                reason: format!(
                    "num_experts_per_tok ({}) must be in [1, {}]",
                    self.num_experts_per_tok, self.num_experts
                ),
            }
            .into());
        }
        if self.shared_expert {
            if let Some(dim) = self.shared_expert_intermediate_size {
                if dim == 0 {
                    return Err(Qwen3Error::InvalidConfig {
                        reason: "shared_expert_intermediate_size must be > 0".into(),
                    }
                    .into());
                }
            }
        }
        Ok(())
    }

    /// Shared expert intermediate size (falls back to base if not specified).
    #[must_use]
    pub fn shared_expert_ff_dim(&self) -> usize {
        self.shared_expert_intermediate_size
            .unwrap_or(self.base.intermediate_size)
    }
}

// -- MoE Decoder Layer --------------------------------------------------------

/// Qwen3 MoE decoder layer: attention + MoE FFN (replaces dense MLP).
struct Qwen3MoeDecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: Qwen3Attention,
    post_attention_layernorm: RmsNorm,
    moe: MoeLayer,
}

impl Qwen3MoeDecoderLayer {
    fn load(vb: impl AsRef<VarBuilder>, cfg: &Qwen3MoeConfig) -> Result<Self> {
        let vb = vb.as_ref();
        let input_layernorm = RmsNorm::new(
            vb.get(&[cfg.base.hidden_size], "input_layernorm.weight")?,
            cfg.base.rms_norm_eps,
        )?;
        let self_attn = Qwen3Attention::load(vb.pp("self_attn"), &cfg.base)?;
        let post_attention_layernorm = RmsNorm::new(
            vb.get(&[cfg.base.hidden_size], "post_attention_layernorm.weight")?,
            cfg.base.rms_norm_eps,
        )?;
        let mut moe_cfg = MoeLayerConfig::new(
            cfg.num_experts,
            cfg.num_experts_per_tok,
            cfg.base.hidden_size,
            cfg.base.intermediate_size,
            true, // norm_topk_prob
            cfg.shared_expert,
        )?;
        if cfg.shared_expert {
            moe_cfg = moe_cfg.with_shared_intermediate_size(cfg.shared_expert_ff_dim())?;
        }
        let moe = MoeLayer::load(vb.pp("mlp"), moe_cfg)?;

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            moe,
        })
    }
}

impl DecoderLayer for Qwen3MoeDecoderLayer {
    fn forward(
        &self,
        x: &DynTensor,
        rope: &RotaryEmbedding,
        positions: &[usize],
        mask: Option<&DynTensor>,
        cache: Option<&mut KvCacheLayer>,
    ) -> Result<DynTensor> {
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
        check_output_finite(&output, "Qwen3MoeDecoderLayer")?;
        Ok(output)
    }
}

// -- Full MoE Model -----------------------------------------------------------

/// Qwen3 MoE decoder-only transformer.
///
/// Same architecture as [`Qwen3Model`](crate::Qwen3Model) but with MoE FFN
/// layers instead of dense SwiGLU. Supports GQA, QK-Norm, KV cache.
///
/// Weight names follow HuggingFace convention:
/// ```text
/// model.embed_tokens.weight
/// model.layers.{i}.input_layernorm.weight
/// model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
/// model.layers.{i}.self_attn.{q,k}_norm.weight
/// model.layers.{i}.post_attention_layernorm.weight
/// model.layers.{i}.mlp.gate.weight                       -> router
/// model.layers.{i}.mlp.experts.{e}.gate_proj.weight      -> expert gate
/// model.layers.{i}.mlp.experts.{e}.up_proj.weight        -> expert up
/// model.layers.{i}.mlp.experts.{e}.down_proj.weight      -> expert down
/// model.layers.{i}.mlp.shared_expert.gate_proj.weight    -> shared (optional)
/// model.norm.weight
/// lm_head.weight (absent when tie_word_embeddings)
/// ```
pub struct Qwen3MoeModel {
    embed_tokens: Embedding,
    layers: Vec<Qwen3MoeDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rope: RotaryEmbedding,
    cfg: Qwen3MoeConfig,
    /// Model weight dtype (from VarBuilder). Used to convert embedding inputs
    /// to match weight dtype for bf16/f16 inference (#1734).
    dtype: DType,
}

impl Qwen3MoeModel {
    /// Load a Qwen3 MoE model from weights via VarBuilder.
    pub fn load(vb: impl AsRef<VarBuilder>, cfg: Qwen3MoeConfig) -> Result<Self> {
        let vb = vb.as_ref();
        cfg.validate()?;

        let model_vb = vb.pp("model");

        let embed_weight = model_vb.get(
            &[cfg.base.vocab_size, cfg.base.hidden_size],
            "embed_tokens.weight",
        )?;
        let embed_tokens = Embedding::new(embed_weight.clone())?;

        let mut layers = Vec::with_capacity(cfg.base.num_hidden_layers);
        for i in 0..cfg.base.num_hidden_layers {
            let layer_vb = model_vb.pp(format!("layers.{i}"));
            layers.push(Qwen3MoeDecoderLayer::load(&layer_vb, &cfg)?);
        }

        let norm = RmsNorm::new(
            model_vb.get(&[cfg.base.hidden_size], "norm.weight")?,
            cfg.base.rms_norm_eps,
        )?;

        let lm_head = if cfg.base.tie_word_embeddings {
            Linear::new(embed_weight, None)?
        } else {
            Linear::new(
                vb.get(
                    &[cfg.base.vocab_size, cfg.base.hidden_size],
                    "lm_head.weight",
                )?,
                None,
            )?
        };

        let rope = build_rope(&cfg.base, vb)?;
        let dtype = vb.dtype();

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            cfg,
            dtype,
        })
    }

    /// Forward pass without KV cache: input token IDs -> logits.
    pub fn forward(&self, input_ids: &[usize], positions: &[usize]) -> Result<DynTensor> {
        self.forward_cached(input_ids, positions, None)
    }

    /// Forward pass with optional KV cache for autoregressive decoding.
    ///
    /// Returns logits tensor of shape `[1, seq_len, vocab_size]`.
    pub fn forward_cached(
        &self,
        input_ids: &[usize],
        positions: &[usize],
        cache: Option<&mut KvCache>,
    ) -> Result<DynTensor> {
        validate_forward_input(input_ids, positions)?;
        let x = embed_and_unsqueeze(&self.embed_tokens, input_ids)?;
        forward_to_logits(
            &self.layers,
            &self.norm,
            &self.lm_head,
            &self.rope,
            x,
            positions,
            cache,
            "Qwen3MoeModel",
        )
    }

    /// Forward pass from pre-computed embeddings.
    pub fn forward_from_embeddings(
        &self,
        hidden_states: &DynTensor,
        positions: &[usize],
        cache: Option<&mut KvCache>,
    ) -> Result<DynTensor> {
        validate_embedding_input(hidden_states, positions, self.cfg.base.hidden_size)?;
        let hidden_states = if hidden_states.dtype() != self.dtype {
            hidden_states.to_dtype(self.dtype)?
        } else {
            hidden_states.clone()
        };
        forward_to_logits(
            &self.layers,
            &self.norm,
            &self.lm_head,
            &self.rope,
            hidden_states,
            positions,
            cache,
            "Qwen3MoeModel",
        )
    }

    /// Forward pass from pre-computed embeddings, returning both logits and
    /// normed hidden states (after RMSNorm, before lm_head).
    ///
    /// Needed by TTS models where the hidden states feed into a CodePredictor
    /// for residual codebook generation (tokens 1-15).
    pub fn forward_from_embeddings_with_hidden(
        &self,
        hidden_states: &DynTensor,
        positions: &[usize],
        cache: Option<&mut KvCache>,
    ) -> Result<(DynTensor, DynTensor)> {
        validate_embedding_input(hidden_states, positions, self.cfg.base.hidden_size)?;
        let hidden_states = if hidden_states.dtype() != self.dtype {
            hidden_states.to_dtype(self.dtype)?
        } else {
            hidden_states.clone()
        };
        forward_to_logits_and_hidden(
            &self.layers,
            &self.norm,
            &self.lm_head,
            &self.rope,
            hidden_states,
            positions,
            cache,
            "Qwen3MoeModel",
        )
    }

    /// Create a new KV cache sized for this model.
    #[must_use]
    pub fn new_cache(&self) -> KvCache {
        KvCache::new(self.cfg.base.num_hidden_layers)
    }

    /// Configuration reference.
    #[must_use]
    pub fn config(&self) -> &Qwen3MoeConfig {
        &self.cfg
    }

    /// Embedding layer reference.
    #[must_use]
    pub fn embed_tokens(&self) -> &Embedding {
        &self.embed_tokens
    }

    /// Model weight dtype (from VarBuilder at load time).
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

#[cfg(test)]
#[path = "qwen3_moe_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "moe_tests.rs"]
mod moe_tests;

#[cfg(test)]
#[path = "moe_routing_tests.rs"]
mod moe_routing_tests;
