// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Chroma Context-1 (gpt-oss-20b) MoE decoder-only transformer for nn.
//!
//! A 20B-parameter Mixture-of-Experts model for agentic search with:
//! - 24 decoder layers with alternating sliding/full attention
//! - GQA: 64 query heads, 8 KV heads, head_dim=64 (attn_dim=4096 > hidden=2880)
//! - Attention bias on Q/K/V/O projections
//! - Per-layer attention sinks (StreamingLLM-style, `[head_dim]` per layer)
//! - 32 fused SwiGLU experts per layer, top-4 routing (router has bias)
//! - Clamped gate activation: `silu(gate).clamp(-7, 7) * up`
//! - Expert weights fused: `[32, 2880, 5760]` gate_up_proj + bias per layer
//! - YaRN RoPE scaling (factor=32, original_max=4096 -> 131K context)
//! - Sliding window attention (128 tokens) on even layers
//!
//! Drop-in replacement architecture on the DynTensor stack.

pub mod agent;
pub(crate) mod attention_sinks;
pub mod batch_inference;
pub mod bench;
pub mod compiled_gptoss;
mod config;
pub(crate) mod context_window;
pub(crate) mod device_utils;
mod error;
pub mod generate;
pub(crate) mod gguf_loader;
pub(crate) mod gpu_dispatch;
pub(crate) mod kv_cache;
mod layers;
pub(crate) mod metal_dispatch;
pub(crate) mod moe_dispatch;
pub mod perf_model;
pub mod quantize;
mod quantized_model;
pub(crate) mod rope_yarn;
pub mod sampling;
pub mod speculative;
pub mod streaming;
#[cfg(feature = "tokenizer")]
pub mod tokenizer;
mod tool_parser;
pub mod verified_search;

pub use agent::{
    AgentConfig, AgentOutput, ContextManager, Document, GrepResult, RetrievedDocument,
    SearchBackend, SearchResult, SearchTool,
};
pub use bench::{
    estimate_kv_cache_memory, estimate_model_memory, estimate_mxfp4_memory, BenchmarkConfig,
    BenchmarkResult,
};
pub use compiled_gptoss::{CompiledGptOss, GenerationOutput, InferenceSession};
pub use config::{GptOssConfig, LayerType};
pub use context_window::{ContextWindow, ContextWindowConfig};
pub use error::GptOssError;
pub use generate::{generate, GenerateConfig};
pub use kv_cache::GptOssKvCache;
pub use perf_model::{ForwardProfile, HardwareProfile, OperationProfile};
pub use quantize::Mxfp4Tensor;
pub use quantized_model::GptOssQuantizedModel;
pub use speculative::{SpeculativeConfig, SpeculativeStats, SpeculativeStep};
pub use streaming::{StreamingConfig, StreamingSession, StreamingToken};
#[cfg(feature = "tokenizer")]
pub use tokenizer::GptOssTokenizer;
pub use verified_search::{
    LogitBounds, SearchQuery, SearchVerificationReport, VerificationStatus, VerifiedSearchResult,
};

#[cfg(kani)]
#[path = "kani_quantize_proofs.rs"]
mod kani_quantize_proofs;

use layers::GptOssDecoderLayer;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::attention::sliding_window_mask;
use nn_core::layers::kv_cache::KvCache;
use nn_core::layers::{
    check_output_finite, with_nan_check_policy, Embedding, Linear, Module, NanCheckPolicy, RmsNorm,
    RotaryEmbedding,
};
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device, Result};

use nn_core::layers::causal_mask_with_offset;

// -- Full Model ---------------------------------------------------------------

/// Chroma Context-1 (gpt-oss-20b) MoE decoder-only transformer.
///
/// Supports GQA, alternating sliding/full attention, clamped SwiGLU MoE,
/// and YaRN extended context. Loads via [`VarBuilder`] with HuggingFace
/// weight names.
pub struct GptOssModel {
    embed_tokens: Embedding,
    layers: Vec<GptOssDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rope: RotaryEmbedding,
    cfg: GptOssConfig,
    dtype: DType,
}

impl GptOssModel {
    /// Load from a safetensors file (F32 on CPU).
    pub fn load_safetensors(path: impl AsRef<std::path::Path>, cfg: GptOssConfig) -> Result<Self> {
        let tensors = nn_core::load_safetensors(path)?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        Self::load(&vb, cfg)
    }

    /// Load from a safetensors file with explicit dtype and device.
    ///
    /// Use `Device::Metal(0)` for GPU inference on Apple Silicon.
    /// Use `DType::BF16` for native BF16 on M4 Max (halves memory, ~same speed).
    ///
    /// The VarBuilder converts and transfers each weight tensor as it is
    /// fetched, so weights land directly on the target device at the target
    /// dtype. All downstream DynTensor operations then dispatch to GPU
    /// automatically.
    pub fn load_safetensors_to_device(
        path: impl AsRef<std::path::Path>,
        cfg: GptOssConfig,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let tensors = nn_core::load_safetensors(path)?;
        let vb = VarBuilder::from_tensors(tensors, dtype, device);
        Self::load(&vb, cfg)
    }

    /// Load from weights via VarBuilder.
    ///
    /// Weight names match actual Context-1 safetensors layout:
    /// ```text
    /// model.embed_tokens.weight                            [201088, 2880]
    /// model.layers.{i}.input_layernorm.weight              [2880]
    /// model.layers.{i}.self_attn.q_proj.weight + .bias     [4096, 2880] + [4096]
    /// model.layers.{i}.self_attn.k_proj.weight + .bias     [512, 2880] + [512]
    /// model.layers.{i}.self_attn.v_proj.weight + .bias     [512, 2880] + [512]
    /// model.layers.{i}.self_attn.o_proj.weight + .bias     [2880, 4096] + [2880]
    /// model.layers.{i}.self_attn.sinks                     [64]
    /// model.layers.{i}.post_attention_layernorm.weight     [2880]
    /// model.layers.{i}.mlp.router.weight + .bias           [32, 2880] + [32]
    /// model.layers.{i}.mlp.experts.gate_up_proj            [32, 2880, 5760]
    /// model.layers.{i}.mlp.experts.gate_up_proj_bias       [32, 5760]
    /// model.layers.{i}.mlp.experts.down_proj               [32, 2880, 2880]
    /// model.layers.{i}.mlp.experts.down_proj_bias          [32, 2880]
    /// model.norm.weight                                    [2880]
    /// lm_head.weight                                       [201088, 2880]
    /// ```
    pub fn load(vb: impl AsRef<VarBuilder>, cfg: GptOssConfig) -> Result<Self> {
        let vb = vb.as_ref();
        cfg.validate()?;

        let model_vb = vb.pp("model");

        let embed_weight =
            model_vb.get(&[cfg.vocab_size, cfg.hidden_size], "embed_tokens.weight")?;
        let embed_tokens = Embedding::new(embed_weight.clone())?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer_vb = model_vb.pp(format!("layers.{i}"));
            let layer_type = cfg.layer_types[i];
            layers.push(GptOssDecoderLayer::load(&layer_vb, &cfg, layer_type)?);
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
        self.forward_inner(x, positions, cache)
    }

    /// Forward pass from pre-computed embeddings.
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
        self.forward_inner(hidden_states, positions, cache)
    }

    /// Shared forward logic: decoder layers -> norm -> lm_head.
    fn forward_inner(
        &self,
        x: DynTensor,
        positions: &[usize],
        cache: Option<&mut KvCache>,
    ) -> Result<DynTensor> {
        let logits = with_nan_check_policy(NanCheckPolicy::Skip, || {
            let normed = self.forward_decoder_and_norm(x, positions, cache)?;
            self.lm_head.forward(&normed)
        })?;
        check_output_finite(&logits, "GptOssModel")?;
        Ok(logits)
    }

    /// Decoder pass: validate cache -> build masks -> decoder layers -> norm.
    fn forward_decoder_and_norm(
        &self,
        mut x: DynTensor,
        positions: &[usize],
        mut cache: Option<&mut KvCache>,
    ) -> Result<DynTensor> {
        validate_cache(cache.as_deref(), self.layers.len())?;

        let seq_len = positions.len();
        let cached_len = cache.as_deref().map_or(0, KvCache::seq_len);
        let total_seq = cached_len + seq_len;

        // Build causal mask (for full attention layers)
        let causal_mask = if seq_len > 1 && total_seq > 1 {
            Some(causal_mask_with_offset(
                seq_len,
                total_seq,
                x.dtype(),
                &x.device(),
            )?)
        } else {
            None
        };

        // Build sliding window mask (for sliding attention layers).
        // Only needed when seq_len > 1 (prompt processing, not single-token decode).
        let sliding_mask = if seq_len > 1 {
            // Combine causal mask with sliding window: each position can only
            // attend to positions within the sliding window AND causally before it.
            let sw_mask = sliding_window_mask(seq_len, self.cfg.sliding_window, &x.device())?;
            // The sliding_window_mask is symmetric. We need to also apply
            // causality (can't attend to future tokens).
            if let Some(ref causal) = causal_mask {
                // Element-wise minimum of two masks (both are 0 or -inf additive)
                // Taking the minimum effectively ANDs the two constraints.
                // minimum() uses broadcast_binary_op internally
                Some(sw_mask.minimum(causal)?)
            } else {
                Some(sw_mask)
            }
        } else {
            None
        };

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = match cache {
                Some(ref mut c) => Some(c.layer_mut(i)?),
                None => None,
            };
            x = layer.forward(
                &x,
                &self.rope,
                positions,
                causal_mask.as_ref(),
                sliding_mask.as_ref(),
                layer_cache,
            )?;
        }

        self.norm.forward(&x)
    }

    /// Create a new KV cache sized for this model.
    #[must_use]
    pub fn new_cache(&self) -> KvCache {
        KvCache::new(self.cfg.num_hidden_layers)
    }

    /// Configuration reference.
    #[must_use]
    pub fn config(&self) -> &GptOssConfig {
        &self.cfg
    }

    /// Embedding layer reference.
    #[must_use]
    pub fn embed_tokens(&self) -> &Embedding {
        &self.embed_tokens
    }

    /// Model weight dtype.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Device the model weights are on.
    #[must_use]
    pub fn device(&self) -> Device {
        self.embed_tokens.weight().device()
    }

    /// Greedy autoregressive generation.
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

    /// Adapter converting generate/beam_search model_fn interface to
    /// forward_cached.
    fn model_fn_adapter(&self, input: &DynTensor, cache: &mut KvCache) -> Result<DynTensor> {
        let u32_data = input.to_flat_vec::<u32>()?;
        let ids: Vec<usize> = u32_data.iter().map(|&v| v as usize).collect();
        let offset = cache.seq_len();
        let positions: Vec<usize> = (0..ids.len()).map(|i| offset + i).collect();
        self.forward_cached(&ids, &positions, Some(cache))
    }
}

// -- Helper functions ---------------------------------------------------------

/// Build RoPE from config.
pub(crate) fn build_rope(
    cfg: &GptOssConfig,
    vb: impl AsRef<VarBuilder>,
) -> Result<RotaryEmbedding> {
    let vb = vb.as_ref();
    match &cfg.rope_scaling {
        Some(yarn) => RotaryEmbedding::new_yarn(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            yarn,
            vb.device(),
        ),
        None => RotaryEmbedding::new(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.device(),
        ),
    }
}

/// Token IDs -> embedded + unsqueezed tensor.
pub(crate) fn embed_and_unsqueeze(
    embed_tokens: &Embedding,
    input_ids: &[usize],
) -> Result<DynTensor> {
    let x = embed_tokens.forward_ids(input_ids)?;
    x.unsqueeze(0)
}

/// Validate forward_cached inputs.
pub(crate) fn validate_forward_input(input_ids: &[usize], positions: &[usize]) -> Result<()> {
    if input_ids.len() != positions.len() {
        return Err(GptOssError::InvalidInput {
            reason: format!(
                "input_ids len ({}) != positions len ({})",
                input_ids.len(),
                positions.len()
            ),
        }
        .into());
    }
    Ok(())
}

/// Validate pre-computed embedding inputs.
fn validate_embedding_input(
    hidden_states: &DynTensor,
    positions: &[usize],
    hidden_size: usize,
) -> Result<()> {
    let (_, seq_len, hs) = hidden_states.dims3()?;
    if seq_len != positions.len() {
        return Err(GptOssError::InvalidInput {
            reason: format!(
                "hidden_states seq_len ({seq_len}) != positions len ({})",
                positions.len()
            ),
        }
        .into());
    }
    if hs != hidden_size {
        return Err(GptOssError::InvalidInput {
            reason: format!(
                "hidden_states hidden_size ({hs}) != model hidden_size ({hidden_size})",
            ),
        }
        .into());
    }
    Ok(())
}

/// Validate KV cache layer count.
pub(crate) fn validate_cache(cache: Option<&KvCache>, num_layers: usize) -> Result<()> {
    if let Some(c) = cache {
        if c.num_layers() != num_layers {
            return Err(GptOssError::CacheMismatch {
                cache_layers: c.num_layers(),
                model_layers: num_layers,
            }
            .into());
        }
    }
    Ok(())
}

#[cfg(kani)]
#[path = "kani_kv_cache_proofs.rs"]
mod kani_kv_cache_proofs;

#[cfg(kani)]
#[path = "kani_kv_cache_advanced_proofs.rs"]
mod kani_kv_cache_advanced_proofs;

#[cfg(kani)]
#[path = "kani_device_proofs.rs"]
mod kani_device_proofs;

#[cfg(kani)]
#[path = "kani_moe_dispatch_proofs.rs"]
mod kani_moe_dispatch_proofs;

#[cfg(kani)]
#[path = "kani_rope_sinks_proofs.rs"]
mod kani_rope_sinks_proofs;

#[cfg(kani)]
#[path = "kani_gpu_dispatch_proofs.rs"]
mod kani_gpu_dispatch_proofs;

#[cfg(kani)]
#[path = "kani_attention_proofs.rs"]
mod kani_attention_proofs;

#[cfg(kani)]
#[path = "kani_generate_proofs.rs"]
mod kani_generate_proofs;

#[cfg(kani)]
#[path = "kani_error_proofs.rs"]
mod kani_error_proofs;

#[cfg(kani)]
#[path = "kani_quantize_advanced_proofs.rs"]
mod kani_quantize_advanced_proofs;

#[cfg(kani)]
#[path = "kani_streaming_proofs.rs"]
mod kani_streaming_proofs;

#[cfg(kani)]
#[path = "kani_sampling_proofs.rs"]
mod kani_sampling_proofs;

#[cfg(kani)]
#[path = "kani_tool_parser_proofs.rs"]
mod kani_tool_parser_proofs;

#[cfg(kani)]
#[path = "kani_agent_proofs.rs"]
mod kani_agent_proofs;

#[cfg(kani)]
#[path = "kani_bench_proofs.rs"]
mod kani_bench_proofs;

#[cfg(kani)]
#[path = "kani_gguf_proofs.rs"]
mod kani_gguf_proofs;

#[cfg(kani)]
#[path = "kani_fused_moe_proofs.rs"]
mod kani_fused_moe_proofs;

#[cfg(kani)]
#[path = "kani_yarn_rope_proofs.rs"]
mod kani_yarn_rope_proofs;

#[cfg(kani)]
#[path = "kani_gguf_mapping_proofs.rs"]
mod kani_gguf_mapping_proofs;

#[cfg(kani)]
#[path = "kani_metal_dispatch_proofs.rs"]
mod kani_metal_dispatch_proofs;

#[cfg(kani)]
#[path = "kani_integration_proofs.rs"]
mod kani_integration_proofs;

#[cfg(kani)]
#[path = "kani_context_window_proofs.rs"]
mod kani_context_window_proofs;

#[cfg(kani)]
#[path = "kani_perf_model_proofs.rs"]
mod kani_perf_model_proofs;

#[cfg(kani)]
#[path = "kani_speculative_proofs.rs"]
mod kani_speculative_proofs;

#[cfg(kani)]
#[path = "kani_batch_inference_proofs.rs"]
mod kani_batch_inference_proofs;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_preset() {
        let cfg = GptOssConfig::gptoss_20b();
        cfg.validate().expect("gptoss_20b preset should validate");
    }
}
