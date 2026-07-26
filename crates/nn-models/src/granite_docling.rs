// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Granite-Docling-258M end-to-end model builder for dpdf.
//!
//! Assembles SigLIP2 vision encoder + Granite-165M decoder into a complete
//! vision-language model (VLM) for document understanding.
//!
//! Architecture:
//! - **Vision encoder:** SigLIP2-base-patch16-512 (768 hidden, 12 layers, 12 heads)
//! - **Vision projection:** Linear(768 → 768) bridging encoder to decoder
//! - **Text embedding:** Embedding(49152, 768) for tokenized text
//! - **Decoder:** 12-layer Granite-165M transformer (GQA: 12 heads / 4 KV heads, SwiGLU MLP)
//! - **LM head:** Linear(768 → 49152) for next-token prediction
//!
//! Input: `[B, 3, 512, 512]` image + `[B, text_len]` token IDs
//! Output: `[B, 1024 + text_len, 49152]` logits
//!
//! Reference: `ibm-granite/granite-vision-docling-258m-preview` (HuggingFace).
//!
//! # Weight loading
//!
//! Weights are loaded via [`VarBuilder`] with HuggingFace naming:
//! - `vision_model.*` — SigLIP2 vision encoder (see [`SigLip2VisionEncoder`])
//! - `multi_modal_projector.linear.{weight,bias}` — vision projection
//! - `model.embed_tokens.weight` — text token embedding
//! - `model.layers.{i}.*` — decoder layers
//! - `model.norm.weight` — final RMSNorm
//! - `lm_head.weight` — language model head

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::vision::{PoolingStrategy, SigLip2Config, SigLip2VisionEncoder};
use nn_core::layers::{
    check_output_finite, with_nan_check_policy, Embedding, Linear, Module, MultiHeadAttention,
    NanCheckPolicy, RmsNorm,
};
use nn_core::var_builder::VarBuilder;
use nn_core::{Result, TensorError};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Default image size (512 for Granite-Docling).
pub const IMAGE_SIZE: usize = 512;
/// Patch size in pixels.
pub const PATCH_SIZE: usize = 16;
/// Number of patches: (512 / 16)^2 = 1024.
pub const NUM_PATCHES: usize = (IMAGE_SIZE / PATCH_SIZE) * (IMAGE_SIZE / PATCH_SIZE);
/// Vision encoder hidden dimension.
pub const VISION_HIDDEN: usize = 768;
/// Vision encoder attention heads.
pub const VISION_HEADS: usize = 12;
/// Vision encoder transformer layers.
pub const VISION_LAYERS: usize = 12;
/// Decoder hidden dimension.
pub const DECODER_HIDDEN: usize = 768;
/// Decoder attention heads (Q heads).
pub const DECODER_HEADS: usize = 12;
/// Decoder KV heads (GQA: 12/4 = 3 Q heads per KV head).
pub const DECODER_KV_HEADS: usize = 4;
/// Decoder SwiGLU intermediate dimension.
pub const DECODER_INTERMEDIATE: usize = 2048;
/// Decoder transformer layers.
pub const DECODER_LAYERS: usize = 12;
/// Vocabulary size.
pub const VOCAB_SIZE: usize = 49152;
/// RMSNorm epsilon.
const RMS_NORM_EPS: f64 = 1e-5;

/// Configuration for Granite-Docling-258M.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct GraniteDoclingConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub vision_hidden: usize,
    pub vision_heads: usize,
    pub vision_layers: usize,
    pub decoder_hidden: usize,
    pub decoder_heads: usize,
    pub decoder_kv_heads: usize,
    pub decoder_intermediate: usize,
    pub decoder_layers: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
}

impl GraniteDoclingConfig {
    /// Create the default 258M configuration.
    #[must_use]
    pub fn default_258m() -> Self {
        Self {
            image_size: IMAGE_SIZE,
            patch_size: PATCH_SIZE,
            vision_hidden: VISION_HIDDEN,
            vision_heads: VISION_HEADS,
            vision_layers: VISION_LAYERS,
            decoder_hidden: DECODER_HIDDEN,
            decoder_heads: DECODER_HEADS,
            decoder_kv_heads: DECODER_KV_HEADS,
            decoder_intermediate: DECODER_INTERMEDIATE,
            decoder_layers: DECODER_LAYERS,
            vocab_size: VOCAB_SIZE,
            rms_norm_eps: RMS_NORM_EPS,
        }
    }

    /// Number of vision patches.
    #[must_use]
    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size) * (self.image_size / self.patch_size)
    }

    /// Head dimension for decoder attention.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.decoder_hidden / self.decoder_heads
    }

    /// Validate configuration consistency.
    pub fn validate(&self) -> Result<()> {
        if self.patch_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "GraniteDoclingConfig: patch_size must be > 0",
            });
        }
        if !self.image_size.is_multiple_of(self.patch_size) {
            return Err(TensorError::ValueOutOfRange {
                description: "GraniteDoclingConfig: image_size must be divisible by patch_size",
            });
        }
        if !self.decoder_hidden.is_multiple_of(self.decoder_heads) {
            return Err(TensorError::ValueOutOfRange {
                description:
                    "GraniteDoclingConfig: decoder_hidden must be divisible by decoder_heads",
            });
        }
        if !self.decoder_heads.is_multiple_of(self.decoder_kv_heads) {
            return Err(TensorError::ValueOutOfRange {
                description:
                    "GraniteDoclingConfig: decoder_heads must be divisible by decoder_kv_heads",
            });
        }
        Ok(())
    }

    fn to_siglip2_config(&self) -> Result<SigLip2Config> {
        SigLip2Config::base_patch16(self.image_size)
    }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

/// Single Granite decoder layer: pre-norm attention + pre-norm SwiGLU MLP.
#[derive(Clone)]
pub struct GraniteDecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: MultiHeadAttention,
    post_attention_layernorm: RmsNorm,
    mlp_gate: Linear,
    mlp_up: Linear,
    mlp_down: Linear,
}

impl std::fmt::Debug for GraniteDecoderLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraniteDecoderLayer")
            .field("self_attn", &self.self_attn)
            .finish_non_exhaustive()
    }
}

impl GraniteDecoderLayer {
    /// Load a decoder layer from a VarBuilder scoped to `model.layers.{i}`.
    pub fn load(vb: impl AsRef<VarBuilder>, cfg: &GraniteDoclingConfig) -> Result<Self> {
        let vb = vb.as_ref();
        let h = cfg.decoder_hidden;

        let input_layernorm =
            RmsNorm::new(vb.get(&[h], "input_layernorm.weight")?, cfg.rms_norm_eps)?;

        let self_attn = MultiHeadAttention::load(
            vb.pp("self_attn"),
            h,
            cfg.decoder_heads,
            cfg.decoder_kv_heads,
            false, // no bias for Granite attention projections
        )?;

        let post_attention_layernorm = RmsNorm::new(
            vb.get(&[h], "post_attention_layernorm.weight")?,
            cfg.rms_norm_eps,
        )?;

        let mlp_vb = vb.pp("mlp");
        let i = cfg.decoder_intermediate;
        let mlp_gate = Linear::new(mlp_vb.get(&[i, h], "gate_proj.weight")?, None)?;
        let mlp_up = Linear::new(mlp_vb.get(&[i, h], "up_proj.weight")?, None)?;
        let mlp_down = Linear::new(mlp_vb.get(&[h, i], "down_proj.weight")?, None)?;

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp_gate,
            mlp_up,
            mlp_down,
        })
    }

    /// Forward pass through this decoder layer.
    ///
    /// Input/output: `[B, S, D]`.
    /// Uses causal self-attention with optional mask.
    pub fn forward(&self, x: &DynTensor, mask: Option<&DynTensor>) -> Result<DynTensor> {
        // Pre-norm attention with residual
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h, None, mask, None, 0)?;
        let x = residual.broadcast_add(&h)?;

        // Pre-norm SwiGLU MLP with residual
        let residual = x.clone();
        let h = self.post_attention_layernorm.forward(&x)?;
        let gate = self.mlp_gate.forward(&h)?.silu()?;
        let up = self.mlp_up.forward(&h)?;
        let h = self.mlp_down.forward(&gate.broadcast_mul(&up)?)?;
        let output = residual.broadcast_add(&h)?;
        check_output_finite(&output, "GraniteDecoderLayer")?;
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

/// Granite-Docling-258M: SigLIP2 vision encoder + Granite-165M decoder.
#[derive(Clone)]
pub struct GraniteDocling {
    vision_encoder: SigLip2VisionEncoder,
    vision_projection: Linear,
    embed_tokens: Embedding,
    decoder_layers: Vec<GraniteDecoderLayer>,
    decoder_norm: RmsNorm,
    lm_head: Linear,
    config: GraniteDoclingConfig,
}

impl std::fmt::Debug for GraniteDocling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraniteDocling")
            .field("vision_encoder", &self.vision_encoder)
            .field("decoder_layers", &self.decoder_layers.len())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl GraniteDocling {
    /// Load the full model from a VarBuilder.
    ///
    /// Weight names follow HuggingFace `ibm-granite/granite-vision-docling-258m-preview`.
    pub fn load(vb: impl AsRef<VarBuilder>, cfg: GraniteDoclingConfig) -> Result<Self> {
        let vb = vb.as_ref();
        cfg.validate()?;

        // Vision encoder
        let siglip_cfg = cfg.to_siglip2_config()?;
        let vision_encoder = SigLip2VisionEncoder::load(vb.pp("vision_model"), &siglip_cfg)?;

        // Multi-modal projector: vision → decoder hidden
        let proj_vb = vb.pp("multi_modal_projector").pp("linear");
        let proj_w = proj_vb.get(&[cfg.decoder_hidden, cfg.vision_hidden], "weight")?;
        let proj_b = if proj_vb.contains_tensor("bias") {
            Some(proj_vb.get(&[cfg.decoder_hidden], "bias")?)
        } else {
            None
        };
        let vision_projection = Linear::new(proj_w, proj_b)?;

        // Text embedding
        let model_vb = vb.pp("model");
        let embed_weight =
            model_vb.get(&[cfg.vocab_size, cfg.decoder_hidden], "embed_tokens.weight")?;
        let embed_tokens = Embedding::new(embed_weight)?;

        // Decoder layers
        let mut decoder_layers = Vec::with_capacity(cfg.decoder_layers);
        for i in 0..cfg.decoder_layers {
            let layer_vb = model_vb.pp(format!("layers.{i}"));
            decoder_layers.push(GraniteDecoderLayer::load(&layer_vb, &cfg)?);
        }

        // Final norm + LM head
        let decoder_norm = RmsNorm::new(
            model_vb.get(&[cfg.decoder_hidden], "norm.weight")?,
            cfg.rms_norm_eps,
        )?;
        let lm_head = Linear::new(
            vb.get(&[cfg.vocab_size, cfg.decoder_hidden], "lm_head.weight")?,
            None,
        )?;

        Ok(Self {
            vision_encoder,
            vision_projection,
            embed_tokens,
            decoder_layers,
            decoder_norm,
            lm_head,
            config: cfg,
        })
    }

    /// Full forward pass: image + text token IDs → logits.
    ///
    /// - `image`: `[B, 3, H, W]` pixel values (H, W = image_size)
    /// - `input_ids`: token IDs as `&[usize]` (flat, length = text_len for B=1)
    ///
    /// Returns: `[B, num_patches + text_len, vocab_size]` logits.
    pub fn forward(&self, image: &DynTensor, input_ids: &[usize]) -> Result<DynTensor> {
        // 1. Vision: encode image → [B, num_patches, vision_hidden]
        let vision_out = self.vision_encoder.forward(image, PoolingStrategy::None)?;

        // 2. Project: [B, num_patches, vision_hidden] → [B, num_patches, decoder_hidden]
        let vision_projected = self.vision_projection.forward(&vision_out)?;

        // 3. Embed text tokens: [B, text_len, decoder_hidden]
        let text_embedded = self.embed_tokens.forward_ids(input_ids)?;
        // Ensure 3D: forward_ids returns [text_len, embed_dim], unsqueeze for batch
        let text_embedded = if text_embedded.rank() == 2 {
            text_embedded.unsqueeze(0)?
        } else {
            text_embedded
        };

        // 4. Concatenate vision + text along sequence dimension
        let combined = DynTensor::cat(&[&vision_projected, &text_embedded], 1)?;

        // 5. Build causal mask for the combined sequence
        let total_len = combined.dim(1)?;
        let mask =
            nn_core::layers::causal_mask_dtype(total_len, combined.dtype(), &combined.device())?;

        // 6. Run through decoder layers — skip per-layer NaN checks to avoid
        // N flush+readback cycles that cause Metal GPU timeout.
        let hidden = with_nan_check_policy(NanCheckPolicy::Skip, || -> Result<DynTensor> {
            let mut h = combined;
            for layer in &self.decoder_layers {
                h = layer.forward(&h, Some(&mask))?;
            }
            Ok(h)
        })?;

        // 7. Final norm → LM head → logits
        let hidden = self.decoder_norm.forward(&hidden)?;
        let logits = self.lm_head.forward(&hidden)?;
        // Defense-in-depth: validate final output OUTSIDE the skip scope
        check_output_finite(&logits, "GraniteDocling")?;
        Ok(logits)
    }

    /// Access the model configuration.
    #[must_use]
    pub fn config(&self) -> &GraniteDoclingConfig {
        &self.config
    }

    /// Access the vision encoder.
    #[must_use]
    pub fn vision_encoder(&self) -> &SigLip2VisionEncoder {
        &self.vision_encoder
    }

    /// Access the decoder layers.
    #[must_use]
    pub fn decoder_layers(&self) -> &[GraniteDecoderLayer] {
        &self.decoder_layers
    }
}

#[cfg(test)]
#[path = "granite_docling_tests.rs"]
mod tests;
