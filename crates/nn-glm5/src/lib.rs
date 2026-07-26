// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GLM-4/5 (ChatGLM) decoder-only LLM for nn.
//!
//! Loads THUDM/glm-4-* and glm-5-* safetensors weights via [`VarBuilder`].
//! Uses the same DynTensor + Module stack as nn-qwen3.
//!
//! Key architectural differences from Llama/Qwen:
//!
//! | Feature          | GLM-4/5 (this crate)           | Qwen3 / Llama        |
//! |------------------|--------------------------------|-----------------------|
//! | QKV projection   | Fused `query_key_value`        | Separate q/k/v_proj   |
//! | QKV bias         | Yes (`add_qkv_bias`)           | No                    |
//! | MLP naming       | `dense_h_to_4h` / `dense_4h_to_h` | gate/up/down_proj |
//! | MLP gate+up      | Fused (split internally)       | Separate projections  |
//! | RoPE             | Partial (first `rot_dim` dims) | Full `head_dim`       |
//! | KV heads config  | `multi_query_group_num`        | `num_key_value_heads` |
//! | Weight prefix    | `transformer.encoder.layers`   | `model.layers`        |
//! | QK-Norm          | No                             | Yes (Qwen3)           |
//!
//! Reference: THUDM/ChatGLM technical reports, HuggingFace model cards.

mod config;
pub mod error;
pub mod kv_cache;
mod layers;
#[cfg(any(test, feature = "test-helpers"))]
pub mod test_utils;

pub use config::Glm5Config;
pub use error::Glm5Error;
pub use nn_core::layers::{causal_mask_dtype as causal_mask, causal_mask_with_offset};

use layers::Glm5DecoderLayer;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::kv_cache::KvCache;
use nn_core::layers::{
    check_output_finite, with_nan_check_policy, Embedding, HalfRotaryEmbedding, Linear, Module,
    NanCheckPolicy, RmsNorm,
};
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device, Result};
use std::path::Path;

// -- Full Model ---------------------------------------------------------------

/// GLM-4/5 decoder-only transformer.
///
/// Supports multi-query attention (MQA/GQA), SwiGLU MLP, partial RoPE.
/// Loads via [`VarBuilder`] with the same weight names as HuggingFace
/// `THUDM/glm-4-*` safetensors files.
pub struct Glm5Model {
    embed_tokens: Embedding,
    layers: Vec<Glm5DecoderLayer>,
    final_layernorm: RmsNorm,
    output_layer: Linear,
    rope: HalfRotaryEmbedding,
    cfg: Glm5Config,
    /// Model weight dtype (from VarBuilder). Used to convert embedding inputs
    /// to match weight dtype for bf16/f16 inference (#1734).
    dtype: DType,
}

impl Glm5Model {
    /// Load a GLM-4/5 model from a safetensors file.
    ///
    /// Convenience constructor matching the `WhisperModel::load_safetensors`
    /// and `SileroVad::load` patterns. Reads the file, builds a [`VarBuilder`],
    /// and delegates to [`Self::load`].
    pub fn load_safetensors(path: impl AsRef<Path>, cfg: Glm5Config) -> Result<Self> {
        let tensors = nn_core::load_safetensors(path)?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        Self::load(&vb, cfg)
    }

    /// Load a GLM-4/5 model from weights via VarBuilder.
    ///
    /// Weight names follow HuggingFace ChatGLM convention:
    /// - `transformer.embedding.word_embeddings.weight`
    /// - `transformer.encoder.layers.{i}.input_layernorm.weight`
    /// - `transformer.encoder.layers.{i}.self_attention.query_key_value.{weight,bias}`
    /// - `transformer.encoder.layers.{i}.self_attention.dense.weight`
    /// - `transformer.encoder.layers.{i}.post_attention_layernorm.weight`
    /// - `transformer.encoder.layers.{i}.mlp.dense_h_to_4h.weight`
    /// - `transformer.encoder.layers.{i}.mlp.dense_4h_to_h.weight`
    /// - `transformer.encoder.final_layernorm.weight`
    /// - `transformer.output_layer.weight`
    pub fn load(vb: impl AsRef<VarBuilder>, cfg: Glm5Config) -> Result<Self> {
        let vb = vb.as_ref();
        cfg.validate()?;

        let transformer_vb = vb.pp("transformer");

        let embed_weight = transformer_vb.get(
            &[cfg.padded_vocab_size, cfg.hidden_size],
            "embedding.word_embeddings.weight",
        )?;
        let embed_tokens = Embedding::new(embed_weight)?;

        let encoder_vb = transformer_vb.pp("encoder");

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let layer_vb = encoder_vb.pp(format!("layers.{i}"));
            layers.push(Glm5DecoderLayer::load(&layer_vb, &cfg)?);
        }

        let final_layernorm = RmsNorm::new(
            encoder_vb.get(&[cfg.hidden_size], "final_layernorm.weight")?,
            cfg.layernorm_epsilon,
        )?;

        let output_layer_bias = if cfg.add_bias_linear {
            Some(transformer_vb.get(&[cfg.padded_vocab_size], "output_layer.bias")?)
        } else {
            None
        };
        let output_layer = Linear::new(
            transformer_vb.get(
                &[cfg.padded_vocab_size, cfg.hidden_size],
                "output_layer.weight",
            )?,
            output_layer_bias,
        )?;

        // GLM uses partial (half) RoPE: first head_dim/2 dims rotated,
        // second half passes through unchanged. HalfRotaryEmbedding handles
        // the split internally: new(head_dim, ...) creates inner RoPE of dim head_dim/2.
        let rope =
            HalfRotaryEmbedding::new(cfg.kv_channels, cfg.seq_length, cfg.rope_theta, vb.device())?;

        let dtype = vb.dtype();

        Ok(Self {
            embed_tokens,
            layers,
            final_layernorm,
            output_layer,
            rope,
            cfg,
            dtype,
        })
    }

    /// Forward pass without KV cache: input token IDs → logits.
    pub fn forward(&self, input_ids: &[usize], positions: &[usize]) -> Result<DynTensor> {
        self.forward_cached(input_ids, positions, None)
    }

    /// Forward pass with optional KV cache for autoregressive decoding.
    ///
    /// - `input_ids`: token IDs as `&[usize]`
    /// - `positions`: position of each token (for RoPE)
    /// - `cache`: optional KV cache (must have `num_layers` layers)
    ///
    /// Returns logits tensor of shape `[1, seq_len, padded_vocab_size]`.
    pub fn forward_cached(
        &self,
        input_ids: &[usize],
        positions: &[usize],
        cache: Option<&mut KvCache>,
    ) -> Result<DynTensor> {
        if input_ids.len() != positions.len() {
            return Err(Glm5Error::InvalidInput {
                reason: format!(
                    "input_ids len ({}) != positions len ({})",
                    input_ids.len(),
                    positions.len()
                ),
            }
            .into());
        }

        let mut x = self.embed_tokens.forward_ids(input_ids)?;
        x = x.unsqueeze(0)?; // [1, seq_len, hidden_size]

        // Skip embedding + per-layer checks. Boundary check catches NaN from any layer.
        let logits = with_nan_check_policy(NanCheckPolicy::Skip, || {
            check_output_finite(&x, "Glm5Model.embedding")?; // no-op in Skip
            self.forward_inner(x, positions, cache)
        })?;
        check_output_finite(&logits, "Glm5Model")?;
        Ok(logits)
    }

    /// Forward pass from pre-computed embeddings, skipping `embed_tokens`.
    pub fn forward_from_embeddings(
        &self,
        hidden_states: &DynTensor,
        positions: &[usize],
        cache: Option<&mut KvCache>,
    ) -> Result<DynTensor> {
        self.validate_embedding_input(hidden_states, positions)?;
        let hidden_states = if hidden_states.dtype() != self.dtype {
            hidden_states.to_dtype(self.dtype)?
        } else {
            hidden_states.clone()
        };
        // Skip per-layer checks. Boundary check catches NaN from any layer.
        let logits = with_nan_check_policy(NanCheckPolicy::Skip, || {
            self.forward_inner(hidden_states, positions, cache)
        })?;
        check_output_finite(&logits, "Glm5Model")?;
        Ok(logits)
    }

    fn validate_embedding_input(
        &self,
        hidden_states: &DynTensor,
        positions: &[usize],
    ) -> Result<()> {
        let (_, seq_len, hidden_size) = hidden_states.dims3()?;
        if seq_len != positions.len() {
            return Err(Glm5Error::InvalidInput {
                reason: format!(
                    "hidden_states seq_len ({seq_len}) != positions len ({})",
                    positions.len()
                ),
            }
            .into());
        }
        if hidden_size != self.cfg.hidden_size {
            return Err(Glm5Error::InvalidInput {
                reason: format!(
                    "hidden_states hidden_size ({hidden_size}) != model hidden_size ({})",
                    self.cfg.hidden_size
                ),
            }
            .into());
        }
        Ok(())
    }

    fn forward_inner(
        &self,
        mut x: DynTensor,
        positions: &[usize],
        mut cache: Option<&mut KvCache>,
    ) -> Result<DynTensor> {
        if let Some(ref c) = cache {
            if c.num_layers() != self.layers.len() {
                return Err(Glm5Error::CacheMismatch {
                    cache_layers: c.num_layers(),
                    model_layers: self.layers.len(),
                }
                .into());
            }
        }
        let seq_len = positions.len();

        let cached_len = cache.as_ref().map_or(0, |c| c.seq_len());
        let total_seq = cached_len + seq_len;
        let mask = if seq_len > 1 && total_seq > 1 {
            Some(causal_mask_with_offset(
                seq_len,
                total_seq,
                x.dtype(),
                &x.device(),
            )?)
        } else {
            None
        };

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = match cache {
                Some(ref mut c) => Some(c.layer_mut(i)?),
                None => None,
            };
            x = layer.forward(&x, &self.rope, positions, mask.as_ref(), layer_cache)?;
        }

        let x = self.final_layernorm.forward(&x)?;
        let logits = self.output_layer.forward(&x)?;
        check_output_finite(&logits, "Glm5Model")?;
        Ok(logits)
    }

    /// Create a new KV cache sized for this model.
    #[must_use]
    pub fn new_cache(&self) -> KvCache {
        KvCache::new(self.cfg.num_layers)
    }

    /// Configuration reference.
    #[must_use]
    pub fn config(&self) -> &Glm5Config {
        &self.cfg
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
    /// GLM-5's `forward_cached`. Computes positions automatically from
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
    /// GLM-5's `forward_cached`. Computes positions automatically from
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
#[path = "glm5_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "glm5_gen_tests.rs"]
mod gen_tests;

#[cfg(test)]
#[path = "glm5_architecture_tests.rs"]
mod glm5_architecture_tests;

#[cfg(test)]
#[path = "glm5_inference_tests.rs"]
mod inference_tests;

#[cfg(test)]
#[path = "decoder_tests.rs"]
mod decoder_tests;

#[cfg(test)]
#[path = "config_tests.rs"]
mod config_tests;

#[cfg(test)]
#[path = "model_tests.rs"]
mod model_tests;

#[cfg(test)]
#[path = "glm5_extended_tests.rs"]
mod glm5_extended_tests;

#[cfg(test)]
#[path = "glm5_decoder_extended_tests.rs"]
mod glm5_decoder_extended_tests;

#[cfg(test)]
#[path = "glm5_config_tests.rs"]
mod glm5_config_tests;

#[cfg(test)]
#[path = "glm5_config_extended_tests.rs"]
mod glm5_config_extended_tests;

#[cfg(test)]
#[path = "glm5_pipeline_extended_tests.rs"]
mod glm5_pipeline_extended_tests;

#[cfg(test)]
#[path = "glm5_model_extended_tests.rs"]
mod glm5_model_extended_tests;

#[cfg(kani)]
#[path = "kani_glm5.rs"]
mod kani_proofs;

#[cfg(kani)]
#[path = "kani_glm5_extra.rs"]
mod kani_proofs_extra;

#[cfg(kani)]
#[path = "kani_glm5_shapes.rs"]
mod kani_proofs_shapes;

#[cfg(kani)]
#[path = "kani_glm5_wave11.rs"]
mod kani_wave11;

#[cfg(kani)]
#[path = "kani_glm5_decoder_proofs.rs"]
mod kani_decoder_proofs;
