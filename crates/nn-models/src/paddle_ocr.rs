// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PaddleOCR-VL-1.5 vision-language model for document OCR.
//!
//! Architecture:
//! - **Vision encoder:** SigLIP-style 27-layer ViT (1152 hidden, 16 heads,
//!   head_dim 72, patch_size 14) with 2D RoPE and learned position embedding,
//!   plus 2x2 spatial merge projector outputting 1024-dim features.
//!   See [`paddle_ocr_vision`] module.
//! - **Decoder:** ERNIE-4.5 18-layer GQA transformer (1024 hidden, 16 Q heads,
//!   2 KV heads, head_dim 128) with 3-axis MultimodalRoPE
//!   (`mrope_section=[16,24,24]`, theta=500000) + SwiGLU MLP + untied LM head
//!   (vocab 103424).
//!
//! Reference: PaddlePaddle/PaddleOCR-VL-1.5 (HuggingFace).
//!
//! # Weight loading
//!
//! Weights are loaded via [`VarBuilder`] with HuggingFace naming:
//! - `visual.vision_model.*` -- SigLIP vision encoder
//! - `mlp_AR.*` -- 2x2 spatial merge projector
//! - `model.embed_tokens.weight` -- text token embedding
//! - `model.layers.{i}.*` -- decoder layers
//! - `model.norm.weight` -- final RMSNorm
//! - `lm_head.weight` -- language model head

use nn_core::layers::{
    causal_mask_with_offset, check_output_finite, embedding, linear_no_bias, repeat_kv, rms_norm,
    sdpa, sdpa_causal, Embedding, KvCache, KvCacheLayer, Linear, Module, MultimodalRoPE, RmsNorm,
    SwiGlu,
};
use nn_core::{Device, DynTensor, Result, TensorError, VarBuilder};

use crate::paddle_ocr_vision::{PaddleOcrVlVisionConfig, PaddleOcrVlVisionEncoder};

// ---------------------------------------------------------------------------
// Constants -- ERNIE-4.5 decoder
// ---------------------------------------------------------------------------

/// Decoder hidden dimension.
pub const DECODER_HIDDEN: usize = 1024;
/// Decoder SwiGLU intermediate dimension.
pub const DECODER_INTERMEDIATE: usize = 3072;
/// Number of decoder layers.
pub const NUM_DECODER_LAYERS: usize = 18;
/// Number of query attention heads.
pub const NUM_HEADS: usize = 16;
/// Number of key-value attention heads (GQA).
pub const NUM_KV_HEADS: usize = 2;
/// Per-head dimension.
pub const HEAD_DIM: usize = 128;
/// Vocabulary size.
pub const VOCAB_SIZE: usize = 103_424;
/// RMSNorm epsilon.
pub const RMS_NORM_EPS: f64 = 1e-5;
/// RoPE frequency base.
pub const ROPE_THETA: f64 = 500_000.0;
/// Maximum position embeddings.
pub const MAX_POSITION_EMBEDDINGS: usize = 131_072;
/// Multimodal RoPE section sizes: [temporal, height, width].
pub const MROPE_SECTION: [usize; 3] = [16, 24, 24];

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the full PaddleOCR-VL-1.5 model.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PaddleOcrVlConfig {
    /// Vision encoder configuration.
    pub vision: PaddleOcrVlVisionConfig,
    /// Decoder hidden dimension.
    pub decoder_hidden: usize,
    /// Decoder intermediate (SwiGLU) dimension.
    pub decoder_intermediate: usize,
    /// Number of decoder layers.
    pub num_decoder_layers: usize,
    /// Number of query heads.
    pub num_heads: usize,
    /// Number of key-value heads (GQA).
    pub num_kv_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f64,
    /// RoPE frequency base.
    pub rope_theta: f64,
    /// Maximum position embeddings.
    pub max_position_embeddings: usize,
    /// Multimodal RoPE section sizes [temporal, height, width].
    pub mrope_section: [usize; 3],
}

impl PaddleOcrVlConfig {
    /// Create the default PaddleOCR-VL-1.5 configuration.
    #[must_use]
    pub fn default_vl() -> Self {
        Self {
            vision: PaddleOcrVlVisionConfig::default(),
            decoder_hidden: DECODER_HIDDEN,
            decoder_intermediate: DECODER_INTERMEDIATE,
            num_decoder_layers: NUM_DECODER_LAYERS,
            num_heads: NUM_HEADS,
            num_kv_heads: NUM_KV_HEADS,
            head_dim: HEAD_DIM,
            vocab_size: VOCAB_SIZE,
            rms_norm_eps: RMS_NORM_EPS,
            rope_theta: ROPE_THETA,
            max_position_embeddings: MAX_POSITION_EMBEDDINGS,
            mrope_section: MROPE_SECTION,
        }
    }

    /// Validate configuration consistency.
    pub fn validate(&self) -> Result<()> {
        self.vision.validate()?;
        if self.decoder_hidden == 0 {
            return Err(TensorError::InvalidShape(
                "PaddleOcrVlConfig: decoder_hidden must be > 0".into(),
            ));
        }
        if self.num_heads == 0 || self.num_kv_heads == 0 {
            return Err(TensorError::InvalidShape(
                "PaddleOcrVlConfig: num_heads and num_kv_heads must be > 0".into(),
            ));
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err(TensorError::InvalidShape(format!(
                "PaddleOcrVlConfig: num_heads {} not divisible by num_kv_heads {}",
                self.num_heads, self.num_kv_heads
            )));
        }
        if self.num_decoder_layers == 0 {
            return Err(TensorError::InvalidShape(
                "PaddleOcrVlConfig: num_decoder_layers must be > 0".into(),
            ));
        }
        if self.vocab_size == 0 {
            return Err(TensorError::InvalidShape(
                "PaddleOcrVlConfig: vocab_size must be > 0".into(),
            ));
        }
        if self.head_dim == 0 {
            return Err(TensorError::InvalidShape(
                "PaddleOcrVlConfig: head_dim must be > 0".into(),
            ));
        }
        Ok(())
    }

    /// GQA group ratio (num_heads / num_kv_heads).
    #[must_use]
    pub fn gqa_ratio(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }
}

// ---------------------------------------------------------------------------
// Multimodal RoPE position IDs
// ---------------------------------------------------------------------------

/// Per-token 3-axis position IDs for multimodal RoPE.
///
/// PaddleOCR-VL uses three independent position streams:
/// temporal, height, and width. Text-only tokens use the same
/// position on all three axes.
#[derive(Debug, Clone)]
pub struct MropePositionIds {
    temporal: Vec<usize>,
    height: Vec<usize>,
    width: Vec<usize>,
}

impl MropePositionIds {
    /// Create explicit multimodal position IDs.
    pub fn new(temporal: Vec<usize>, height: Vec<usize>, width: Vec<usize>) -> Result<Self> {
        let seq_len = temporal.len();
        if height.len() != seq_len || width.len() != seq_len {
            return Err(TensorError::DataLengthMismatch {
                expected: seq_len,
                actual: height.len().max(width.len()),
            });
        }
        Ok(Self {
            temporal,
            height,
            width,
        })
    }

    /// Create text-style positions where all 3 axes are identical.
    #[must_use]
    pub fn text(start: usize, seq_len: usize) -> Self {
        let positions: Vec<usize> = (start..start + seq_len).collect();
        Self {
            temporal: positions.clone(),
            height: positions.clone(),
            width: positions,
        }
    }

    /// Number of positions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.temporal.len()
    }

    /// Whether positions are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.temporal.is_empty()
    }

    /// Largest position value across all 3 axes.
    #[must_use]
    pub fn max_position(&self) -> usize {
        self.temporal
            .iter()
            .chain(self.height.iter())
            .chain(self.width.iter())
            .copied()
            .max()
            .unwrap_or(0)
    }

    /// Get the three axes as slices.
    pub fn axes(&self) -> [&[usize]; 3] {
        [&self.temporal, &self.height, &self.width]
    }

    /// Validate that the position ID length matches a sequence length.
    pub fn validate_len(&self, seq_len: usize) -> Result<()> {
        if self.len() != seq_len {
            return Err(TensorError::DataLengthMismatch {
                expected: seq_len,
                actual: self.len(),
            });
        }
        Ok(())
    }

    /// HuggingFace-style `rope_delta = max_position + 1 - seq_len`.
    fn rope_delta(&self) -> Result<isize> {
        let max_plus_one = isize::try_from(self.max_position().saturating_add(1))
            .map_err(|_| TensorError::InvalidShape("M-ROPE position overflow".into()))?;
        let seq_len = isize::try_from(self.len())
            .map_err(|_| TensorError::InvalidShape("sequence length overflow".into()))?;
        max_plus_one
            .checked_sub(seq_len)
            .ok_or_else(|| TensorError::InvalidShape("M-ROPE delta underflow".into()))
    }

    /// Text-style decode positions continuing from a multimodal prefill.
    pub fn continuation_from_cache_len(&self, cache_len: usize, seq_len: usize) -> Result<Self> {
        let cache_len = isize::try_from(cache_len)
            .map_err(|_| TensorError::InvalidShape("cache length overflow".into()))?;
        let start = cache_len
            .checked_add(self.rope_delta()?)
            .ok_or_else(|| TensorError::InvalidShape("continuation position overflow".into()))?;
        if start < 0 {
            return Err(TensorError::InvalidShape(
                "continuation position became negative".into(),
            ));
        }
        Ok(Self::text(
            usize::try_from(start)
                .map_err(|_| TensorError::InvalidShape("continuation position overflow".into()))?,
            seq_len,
        ))
    }
}

// ---------------------------------------------------------------------------
// Decoder Layer
// ---------------------------------------------------------------------------

/// Single PaddleOCR-VL decoder block (ERNIE-4.5 GQA + SwiGLU MLP).
#[derive(Clone)]
struct DecoderLayer {
    input_layernorm: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    post_attention_layernorm: RmsNorm,
    mlp: SwiGlu,
}

impl DecoderLayer {
    fn load(vb: &VarBuilder, cfg: &PaddleOcrVlConfig) -> Result<Self> {
        let attn_vb = vb.pp("self_attn");
        let mlp_vb = vb.pp("mlp").with_name_mapping(|name| {
            name.replace(".w_gate.", ".gate_proj.")
                .replace(".w_up.", ".up_proj.")
                .replace(".w_down.", ".down_proj.")
        });

        Ok(Self {
            input_layernorm: rms_norm(
                cfg.decoder_hidden,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            q_proj: linear_no_bias(
                cfg.decoder_hidden,
                cfg.num_heads * cfg.head_dim,
                attn_vb.pp("q_proj"),
            )?,
            k_proj: linear_no_bias(
                cfg.decoder_hidden,
                cfg.num_kv_heads * cfg.head_dim,
                attn_vb.pp("k_proj"),
            )?,
            v_proj: linear_no_bias(
                cfg.decoder_hidden,
                cfg.num_kv_heads * cfg.head_dim,
                attn_vb.pp("v_proj"),
            )?,
            o_proj: linear_no_bias(
                cfg.num_heads * cfg.head_dim,
                cfg.decoder_hidden,
                attn_vb.pp("o_proj"),
            )?,
            post_attention_layernorm: rms_norm(
                cfg.decoder_hidden,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: SwiGlu::load(&mlp_vb, cfg.decoder_hidden, cfg.decoder_intermediate)?,
        })
    }

    fn forward(
        &self,
        x: &DynTensor,
        mrope: &MultimodalRoPE,
        position_ids: &MropePositionIds,
        attention_mask: Option<&DynTensor>,
        cache: Option<&mut KvCacheLayer>,
        cfg: &PaddleOcrVlConfig,
    ) -> Result<DynTensor> {
        let (b, seq_len, _) = x.dims3()?;

        let h = self.input_layernorm.forward(x)?;

        let q = self
            .q_proj
            .forward(&h)?
            .reshape([b, seq_len, cfg.num_heads, cfg.head_dim])?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(&h)?
            .reshape([b, seq_len, cfg.num_kv_heads, cfg.head_dim])?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(&h)?
            .reshape([b, seq_len, cfg.num_kv_heads, cfg.head_dim])?
            .transpose(1, 2)?;

        let [t_positions, h_positions, w_positions] = position_ids.axes();
        let (q, k) = mrope.apply_pair(&q, &k, t_positions, h_positions, w_positions)?;

        let (k, v) = match cache {
            Some(cache) => cache.append(&k, &v)?,
            None => (k, v),
        };

        let k = repeat_kv(&k, cfg.gqa_ratio())?;
        let v = repeat_kv(&v, cfg.gqa_ratio())?;

        let scale = (cfg.head_dim as f64).sqrt().recip();
        let attn_output = match attention_mask {
            Some(mask) => sdpa(&q, &k, &v, Some(mask), scale)?,
            None => sdpa_causal(&q, &k, &v, scale)?,
        };

        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape([b, seq_len, cfg.num_heads * cfg.head_dim])?;
        let attn_output = self.o_proj.forward(&attn_output)?;

        let h = x.broadcast_add(&attn_output)?;
        let h2 = self.post_attention_layernorm.forward(&h)?;
        let h2 = self.mlp.forward(&h2)?;
        h.broadcast_add(&h2)
    }
}

// ---------------------------------------------------------------------------
// Full PaddleOCR-VL Model
// ---------------------------------------------------------------------------

/// PaddleOCR-VL-1.5: vision encoder + ERNIE-4.5 GQA decoder + LM head.
#[derive(Clone)]
pub struct PaddleOcrVl {
    vision_encoder: PaddleOcrVlVisionEncoder,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    embed_tokens: Embedding,
    lm_head: Linear,
    rope: MultimodalRoPE,
    config: PaddleOcrVlConfig,
}

impl std::fmt::Debug for PaddleOcrVl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PaddleOcrVl")
            .field("vision_encoder", &self.vision_encoder)
            .field("decoder_layers", &self.layers.len())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl PaddleOcrVl {
    /// Load the full model from a VarBuilder.
    ///
    /// Weight names follow HuggingFace `PaddlePaddle/PaddleOCR-VL-1.5`.
    pub fn load(vb: &VarBuilder, cfg: PaddleOcrVlConfig) -> Result<Self> {
        cfg.validate()?;

        let vision_encoder = PaddleOcrVlVisionEncoder::load(vb, cfg.vision)?;

        let model_vb = vb.pp("model");
        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.decoder_hidden,
            model_vb.pp("embed_tokens"),
        )?;

        let mut layers = Vec::with_capacity(cfg.num_decoder_layers);
        for i in 0..cfg.num_decoder_layers {
            layers.push(DecoderLayer::load(
                &model_vb.pp("layers").pp(i.to_string()),
                &cfg,
            )?);
        }

        let norm = rms_norm(cfg.decoder_hidden, cfg.rms_norm_eps, model_vb.pp("norm"))?;
        let lm_head = linear_no_bias(cfg.decoder_hidden, cfg.vocab_size, vb.pp("lm_head"))?;
        let rope = MultimodalRoPE::new(
            cfg.head_dim,
            cfg.mrope_section,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.device(),
        )?;

        Ok(Self {
            vision_encoder,
            layers,
            norm,
            embed_tokens,
            lm_head,
            rope,
            config: cfg,
        })
    }

    /// Encode an image through the vision encoder.
    ///
    /// Input: `[batch, 3, H, W]` (H, W divisible by 28).
    /// Output: `[batch, merged_tokens, 1024]`.
    pub fn vision_encode(&self, image: &DynTensor) -> Result<DynTensor> {
        self.vision_encoder.forward(image)
    }

    /// Embed token IDs into the decoder hidden space.
    ///
    /// Returns: `[1, seq_len, decoder_hidden]`.
    pub fn embed_token_ids(&self, ids: &[usize]) -> Result<DynTensor> {
        self.embed_tokens.forward_ids(ids)?.unsqueeze(0)
    }

    /// Run the decoder stack on pre-built embeddings.
    ///
    /// Returns: `[batch, seq_len, decoder_hidden]` (before LM head).
    pub fn decoder_forward(
        &self,
        input_embeds: &DynTensor,
        position_ids: &MropePositionIds,
        mut kv_cache: Option<&mut KvCache>,
    ) -> Result<DynTensor> {
        let (_, seq_len, hidden_size) = input_embeds.dims3()?;
        if hidden_size != self.config.decoder_hidden {
            return Err(TensorError::InvalidShape(format!(
                "PaddleOcrVl expected hidden size {}, got {hidden_size}",
                self.config.decoder_hidden
            )));
        }
        position_ids.validate_len(seq_len)?;

        let cached_tokens = kv_cache.as_deref().map_or(0, KvCache::seq_len);
        let attention_mask = build_attention_mask(
            seq_len,
            cached_tokens,
            input_embeds.dtype(),
            &input_embeds.device(),
        )?;

        let mut h = input_embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = match kv_cache.as_deref_mut() {
                Some(cache) => Some(cache.layer_mut(i)?),
                None => None,
            };
            h = layer.forward(
                &h,
                &self.rope,
                position_ids,
                attention_mask.as_ref(),
                layer_cache,
                &self.config,
            )?;
        }

        let out = self.norm.forward(&h)?;
        check_output_finite(&out, "PaddleOcrVlDecoderStack")?;
        Ok(out)
    }

    /// Apply the LM head to decoder output.
    ///
    /// Returns: `[batch, seq_len, vocab_size]` logits.
    pub fn lm_head_forward(&self, hidden: &DynTensor) -> Result<DynTensor> {
        let logits = self.lm_head.forward(hidden)?;
        check_output_finite(&logits, "PaddleOcrVlLmHead")?;
        Ok(logits)
    }

    /// Full forward pass: embeddings -> decoder -> logits.
    ///
    /// Returns: `[batch, seq_len, vocab_size]` logits.
    pub fn forward(
        &self,
        input_embeds: &DynTensor,
        position_ids: &MropePositionIds,
        kv_cache: Option<&mut KvCache>,
    ) -> Result<DynTensor> {
        let hidden = self.decoder_forward(input_embeds, position_ids, kv_cache)?;
        self.lm_head_forward(&hidden)
    }

    /// Access the model configuration.
    #[must_use]
    pub fn config(&self) -> &PaddleOcrVlConfig {
        &self.config
    }

    /// Access the vision encoder.
    #[must_use]
    pub fn vision_encoder(&self) -> &PaddleOcrVlVisionEncoder {
        &self.vision_encoder
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_attention_mask(
    new_tokens: usize,
    cached_tokens: usize,
    dtype: nn_core::DType,
    device: &Device,
) -> Result<Option<DynTensor>> {
    if cached_tokens == 0 {
        return Ok(None);
    }
    let total_tokens = cached_tokens
        .checked_add(new_tokens)
        .ok_or_else(|| TensorError::InvalidShape("attention sequence length overflow".into()))?;
    Ok(Some(causal_mask_with_offset(
        new_tokens,
        total_tokens,
        dtype,
        device,
    )?))
}

#[cfg(test)]
#[path = "paddle_ocr_tests.rs"]
mod tests;
