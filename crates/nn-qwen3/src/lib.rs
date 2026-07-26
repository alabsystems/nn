// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Qwen3 dense decoder-only **text** LLM for nn.
//!
//! This is a standard Qwen3 text decoder — **not** Qwen3-TTS. dvoice uses
//! Qwen3-TTS, which differs architecturally:
//!
//! | Feature           | nn-qwen3 (this crate)   | Qwen3-TTS (dvoice)                |
//! |-------------------|--------------------------|--------------------------------------|
//! | Embedding         | Single vocab             | Dual-track (text + codec, summed)    |
//! | RoPE              | Standard 1D              | 3D Multimodal (mrope_section)        |
//! | Output            | logits, or logits+hidden | logits + hidden states               |
//! | Pipeline          | Single LM                | Talker + CodePredictor + SpeechToken |
//! | Quantization      | None                     | GGML Q4K / Q8_0                      |
//!
//! nn-qwen3 serves as: (a) reference implementation for standard Qwen3,
//! (b) KV cache + attention integration test case, (c) text-only Qwen3 for
//! non-TTS use. The TTS-specific wiring (dual-track embedding, M-RoPE,
//! CodePredictor, SpeechTokenizer) lives dvoice-side using nn framework ops.
//!
//! Drop-in replacement for `candle_transformers::Qwen3` on the DynTensor stack.
//! Supports GQA (Grouped Query Attention) and QK-Norm (Qwen3-specific).
//!
//! ## RoPE Convention
//!
//! This crate uses **half-split** (HuggingFace `rotate_half`) RoPE, which
//! pairs element `(i, i + head_dim/2)`. This matches HuggingFace Qwen3
//! weights directly — **no Q/K weight permutation is needed**. Do NOT use
//! `RopePermutingBackend` or similar interleaved-convention adapters when
//! loading weights into this model. See `rope_convention_tests.rs` and #4327.
//!
//! Reference: Qwen3 Technical Report (arXiv:2505.09388).

mod config;
mod error;
mod forward_common;
mod layers;
mod moe;
pub mod rope_cache;
#[cfg(any(test, feature = "test-helpers"))]
pub mod test_utils;

pub use config::Qwen3Config;
pub use error::Qwen3Error;
pub use nn_core::layers::{causal_mask_dtype as causal_mask, causal_mask_with_offset};
pub use moe::{Qwen3MoeConfig, Qwen3MoeModel};

use forward_common::{
    build_rope, embed_and_unsqueeze, forward_to_logits, forward_to_logits_and_hidden,
    validate_embedding_input, validate_forward_input,
};
use layers::Qwen3DecoderLayer;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::kv_cache::KvCache;
use nn_core::layers::{Embedding, Linear, RmsNorm, RotaryEmbedding};
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device, Result};
use std::path::Path;

// -- Full Model ---------------------------------------------------------------

/// Qwen3 dense decoder-only transformer.
///
/// Supports GQA, QK-Norm, SwiGLU MLP. Loads via [`VarBuilder`] with the same
/// weight names as HuggingFace `Qwen/Qwen3-*` safetensors files.
pub struct Qwen3Model {
    embed_tokens: Embedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rope: RotaryEmbedding,
    cfg: Qwen3Config,
    /// Model weight dtype (from VarBuilder). Used to convert embedding inputs
    /// to match weight dtype for bf16/f16 inference (#1734).
    dtype: DType,
}

impl Qwen3Model {
    /// Load a Qwen3 model from a safetensors file.
    ///
    /// Convenience constructor matching the `WhisperModel::load_safetensors`
    /// and `SileroVad::load` patterns. Reads the file, builds a [`VarBuilder`],
    /// and delegates to [`Self::load`].
    pub fn load_safetensors(path: impl AsRef<Path>, cfg: Qwen3Config) -> Result<Self> {
        let tensors = nn_core::load_safetensors(path)?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        Self::load(&vb, cfg)
    }

    /// Load a Qwen3 model from weights via VarBuilder.
    ///
    /// Weight names follow HuggingFace convention:
    /// - `model.embed_tokens.weight`
    /// - `model.layers.{i}.input_layernorm.weight`
    /// - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
    /// - `model.layers.{i}.self_attn.{q,k}_norm.weight`
    /// - `model.layers.{i}.post_attention_layernorm.weight`
    /// - `model.layers.{i}.mlp.{gate,up,down}_proj.weight`
    /// - `model.norm.weight`
    /// - `lm_head.weight` (absent when `tie_word_embeddings`)
    pub fn load(vb: impl AsRef<VarBuilder>, cfg: Qwen3Config) -> Result<Self> {
        let vb = vb.as_ref();
        cfg.validate()?;

        let model_vb = vb.pp("model");

        let embed_weight =
            model_vb.get(&[cfg.vocab_size, cfg.hidden_size], "embed_tokens.weight")?;
        let embed_tokens = Embedding::new(embed_weight.clone())?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer_vb = model_vb.pp(format!("layers.{i}"));
            layers.push(Qwen3DecoderLayer::load(&layer_vb, &cfg)?);
        }

        let norm = RmsNorm::new(
            model_vb.get(&[cfg.hidden_size], "norm.weight")?,
            cfg.rms_norm_eps,
        )?;

        // lm_head: separate or tied to embedding
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_weight, None)?
        } else {
            Linear::new(
                vb.get(&[cfg.vocab_size, cfg.hidden_size], "lm_head.weight")?,
                None,
            )?
        };

        let rope = build_rope(&cfg, vb)?;

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

    /// Forward pass without KV cache: input token IDs → logits.
    ///
    /// Equivalent to `forward_cached(input_ids, positions, None)`.
    pub fn forward(&self, input_ids: &[usize], positions: &[usize]) -> Result<DynTensor> {
        self.forward_cached(input_ids, positions, None)
    }

    /// Forward pass with optional KV cache for autoregressive decoding.
    ///
    /// - `input_ids`: token IDs as `&[usize]`
    /// - `positions`: position of each token (for RoPE)
    /// - `cache`: optional KV cache (must have `num_hidden_layers` layers)
    ///
    /// Returns logits tensor of shape `[1, seq_len, vocab_size]`.
    ///
    /// When `cache` is `Some`, each attention layer appends its K/V projections
    /// to the per-layer cache and attends over the full cached sequence. This
    /// makes autoregressive decoding O(n) per step instead of O(n²).
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
            "Qwen3Model",
        )
    }

    /// Forward pass from pre-computed embeddings, skipping `embed_tokens`.
    ///
    /// Used by TTS models (e.g., Qwen3-TTS) where input is a dual-track
    /// embedding (text projection + codec embedding summed), not standard
    /// token IDs.
    ///
    /// - `hidden_states`: pre-computed embeddings `[batch, seq, hidden_size]`
    /// - `positions`: position of each token (for RoPE), length must equal `seq`
    /// - `cache`: optional KV cache (must have `num_hidden_layers` layers)
    ///
    /// Returns logits tensor of shape `[batch, seq, vocab_size]`.
    pub fn forward_from_embeddings(
        &self,
        hidden_states: &DynTensor,
        positions: &[usize],
        cache: Option<&mut KvCache>,
    ) -> Result<DynTensor> {
        validate_embedding_input(hidden_states, positions, self.cfg.hidden_size)?;
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
            "Qwen3Model",
        )
    }

    /// Forward pass from pre-computed embeddings, returning both logits and
    /// normed hidden states (after RMSNorm, before lm_head).
    ///
    /// Needed by TTS models where the hidden states feed into a CodePredictor
    /// for residual codebook generation (tokens 1–15).
    ///
    /// - `hidden_states`: pre-computed embeddings `[batch, seq, hidden_size]`
    /// - `positions`: position of each token (for RoPE), length must equal `seq`
    /// - `cache`: optional KV cache (must have `num_hidden_layers` layers)
    ///
    /// Returns `(logits, normed_hidden)` where:
    /// - `logits`: shape `[batch, seq, vocab_size]`
    /// - `normed_hidden`: shape `[batch, seq, hidden_size]` (after final RMSNorm)
    pub fn forward_from_embeddings_with_hidden(
        &self,
        hidden_states: &DynTensor,
        positions: &[usize],
        cache: Option<&mut KvCache>,
    ) -> Result<(DynTensor, DynTensor)> {
        validate_embedding_input(hidden_states, positions, self.cfg.hidden_size)?;
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
            "Qwen3Model",
        )
    }

    /// Create a new KV cache sized for this model.
    #[must_use]
    pub fn new_cache(&self) -> KvCache {
        KvCache::new(self.cfg.num_hidden_layers)
    }

    /// Configuration reference.
    #[must_use]
    pub fn config(&self) -> &Qwen3Config {
        &self.cfg
    }

    /// Embedding layer reference.
    ///
    /// Useful for TTS models that need to access the embedding table
    /// for dual-track embedding (text + codec tokens).
    #[must_use]
    pub fn embed_tokens(&self) -> &Embedding {
        &self.embed_tokens
    }

    /// Model weight dtype (from VarBuilder at load time).
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Device the model weights are on (inferred from embedding weight).
    #[must_use]
    pub fn device(&self) -> Device {
        self.embed_tokens.weight().device()
    }

    /// Greedy autoregressive generation.
    ///
    /// Convenience wrapper that bridges `nn_core::layers::generate` with
    /// Qwen3's `forward_cached`. Computes positions automatically from
    /// the KV cache state.
    pub fn generate_greedy(
        &self,
        prompt_ids: &[usize],
        max_new_tokens: usize,
    ) -> Result<nn_core::layers::GenerationOutput> {
        use nn_core::layers::{generate, GenerationConfig};
        let device = self.device();
        let mut cache = self.new_cache();
        let mut cfg = GenerationConfig::default();
        cfg.max_new_tokens = max_new_tokens;
        generate(
            |input, c| self.model_fn_adapter(input, c),
            prompt_ids,
            &mut cache,
            &cfg,
            &device,
        )
    }

    /// Beam search decoding.
    ///
    /// Convenience wrapper that bridges `nn_core::layers::beam_search` with
    /// Qwen3's `forward_cached`. Computes positions automatically from
    /// the KV cache state.
    pub fn generate_beam(
        &self,
        prompt_ids: &[usize],
        config: &nn_core::layers::BeamSearchConfig,
    ) -> Result<nn_core::layers::BeamSearchOutput> {
        use nn_core::layers::beam_search;
        let device = self.device();
        let mut cache = self.new_cache();
        beam_search(
            |input, c| self.model_fn_adapter(input, c),
            prompt_ids,
            &mut cache,
            config,
            &device,
        )
    }

    /// Adapter converting `beam_search`/`generate` model_fn interface to
    /// `forward_cached`. Extracts U32 token IDs from DynTensor and computes
    /// positions from KV cache seq_len.
    fn model_fn_adapter(&self, input: &DynTensor, cache: &mut KvCache) -> Result<DynTensor> {
        let u32_data = input.to_flat_vec::<u32>()?;
        let ids: Vec<usize> = u32_data.iter().map(|&v| v as usize).collect();
        let offset = cache.seq_len();
        let positions: Vec<usize> = (0..ids.len()).map(|i| offset + i).collect();
        self.forward_cached(&ids, &positions, Some(cache))
    }
}

#[cfg(test)]
#[path = "qwen3_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "qwen3_gen_tests.rs"]
mod gen_tests;

#[cfg(test)]
#[path = "qwen3_expanded_tests.rs"]
mod qwen3_expanded_tests;

#[cfg(test)]
#[path = "attention_tests.rs"]
mod attention_tests;

#[cfg(test)]
#[path = "mlp_tests.rs"]
mod mlp_tests;

#[cfg(test)]
#[path = "generation_tests.rs"]
mod generation_tests;

#[cfg(test)]
#[path = "config_tests.rs"]
mod config_tests;

#[cfg(test)]
#[path = "rope_tests.rs"]
mod rope_tests;

#[cfg(test)]
#[path = "kv_cache_tests.rs"]
mod kv_cache_tests;

#[cfg(test)]
#[path = "qwen3_extended_tests.rs"]
mod qwen3_extended_tests;

#[cfg(test)]
#[path = "rope_convention_tests.rs"]
mod rope_convention_tests;

#[cfg(test)]
#[path = "qwen3_architecture_extended_tests.rs"]
mod qwen3_architecture_extended_tests;

#[cfg(test)]
#[path = "qwen3_config_extended_tests.rs"]
mod qwen3_config_extended_tests;

#[cfg(test)]
#[path = "qwen3_config_tests.rs"]
mod qwen3_config_tests;

#[cfg(test)]
#[path = "qwen3_config_model_sizes_tests.rs"]
mod qwen3_config_model_sizes_tests;

#[cfg(test)]
#[path = "qwen3_pipeline_extended_tests.rs"]
mod qwen3_pipeline_extended_tests;

#[cfg(test)]
#[path = "qwen3_model_extended_tests.rs"]
mod qwen3_model_extended_tests;

#[cfg(kani)]
#[path = "kani_qwen3.rs"]
mod kani_proofs;

#[cfg(kani)]
#[path = "kani_moe_forward_proofs.rs"]
mod kani_moe_forward_proofs;

#[cfg(kani)]
#[path = "kani_moe.rs"]
mod kani_moe;

#[cfg(kani)]
#[path = "kani_forward_common.rs"]
mod kani_forward_common;

#[cfg(kani)]
#[path = "kani_layers.rs"]
mod kani_layers;

#[cfg(kani)]
#[path = "kani_lib.rs"]
mod kani_lib;

#[cfg(kani)]
#[path = "kani_model_invariants.rs"]
mod kani_model_invariants;

#[cfg(kani)]
#[path = "kani_qwen3_wave11.rs"]
mod kani_qwen3_wave11;

#[cfg(kani)]
#[path = "kani_qwen3_vl_projection_proofs.rs"]
mod kani_qwen3_vl_projection_proofs;
