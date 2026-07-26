// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GLM-OCR 0.9B model builder with Multi-Token Prediction.
//!
//! SigLIP2 vision encoder + GLM decoder + MTP heads for speculative OCR.
//! Reference: `THUDM/glm-ocr-2b` family (HuggingFace).
//!
//! **Note:** dpdf's current pipeline does NOT use GLM-OCR. dpdf Tier 1B uses
//! PaddleOCR-VL-1.5 for detection+recognition instead. GLM-OCR is retained as
//! an available model but is not wired into dpdf's production pipeline.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::generation::{
    greedy_decode_with_verification, MtpHead, MtpHeadConfig, SpeculativeConfig, SpeculativeOutput,
};
use nn_core::layers::vision::{PoolingStrategy, SigLip2Config, SigLip2VisionEncoder};
use nn_core::layers::{
    check_output_finite, with_nan_check_policy, Embedding, Linear, Module, MultiHeadAttention,
    NanCheckPolicy, RmsNorm,
};
use nn_core::var_builder::VarBuilder;
use nn_core::{Result, TensorError};

// -- Architecture constants (0.9B) --------------------------------------------

pub const HIDDEN: usize = 1536;
pub const NUM_HEADS: usize = 16;
pub const NUM_KV_HEADS: usize = 4;
pub const INTERMEDIATE: usize = 4096;
pub const NUM_LAYERS: usize = 24;
pub const VOCAB_SIZE: usize = 65024;
pub const VISION_HIDDEN: usize = 768;
pub const VISION_LAYERS: usize = 12;
pub const MTP_DEPTH: usize = 3;
pub const IMAGE_SIZE: usize = 384;
pub const PATCH_SIZE: usize = 16;
const RMS_NORM_EPS: f64 = 1e-6;

// -- Configuration ------------------------------------------------------------

/// Configuration for GLM-OCR models.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct GlmOcrConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub vision_hidden: usize,
    pub vision_heads: usize,
    pub vision_layers: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub mtp_depth: usize,
    pub rms_norm_eps: f64,
}

impl GlmOcrConfig {
    /// Create the 0.9B configuration (GLM-OCR tier).
    #[must_use]
    pub fn preset_900m() -> Self {
        Self {
            hidden_size: HIDDEN,
            num_heads: NUM_HEADS,
            num_kv_heads: NUM_KV_HEADS,
            intermediate_size: INTERMEDIATE,
            num_layers: NUM_LAYERS,
            vocab_size: VOCAB_SIZE,
            vision_hidden: VISION_HIDDEN,
            vision_heads: 12,
            vision_layers: VISION_LAYERS,
            image_size: IMAGE_SIZE,
            patch_size: PATCH_SIZE,
            mtp_depth: MTP_DEPTH,
            rms_norm_eps: RMS_NORM_EPS,
        }
    }

    /// Number of vision patches.
    #[must_use]
    pub fn num_patches(&self) -> usize {
        if self.patch_size == 0 {
            return 0;
        }
        (self.image_size / self.patch_size) * (self.image_size / self.patch_size)
    }

    /// Decoder head dimension.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        if self.num_heads == 0 {
            return 0;
        }
        self.hidden_size / self.num_heads
    }

    /// GQA group ratio (Q heads per KV head).
    #[must_use]
    pub fn gqa_ratio(&self) -> usize {
        if self.num_kv_heads == 0 {
            return 0;
        }
        self.num_heads / self.num_kv_heads
    }

    /// Validate configuration consistency.
    pub fn validate(&self) -> Result<()> {
        let bad = |d: &'static str| TensorError::ValueOutOfRange { description: d };
        if self.num_heads == 0 {
            return Err(bad("GlmOcrConfig: num_heads must be > 0"));
        }
        if self.num_kv_heads == 0 {
            return Err(bad("GlmOcrConfig: num_kv_heads must be > 0"));
        }
        if !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(bad(
                "GlmOcrConfig: hidden_size must be divisible by num_heads",
            ));
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err(bad(
                "GlmOcrConfig: num_heads must be divisible by num_kv_heads",
            ));
        }
        if self.patch_size == 0 {
            return Err(bad("GlmOcrConfig: patch_size must be > 0"));
        }
        if !self.image_size.is_multiple_of(self.patch_size) {
            return Err(bad(
                "GlmOcrConfig: image_size must be divisible by patch_size",
            ));
        }
        Ok(())
    }

    fn to_siglip2_config(&self) -> Result<SigLip2Config> {
        SigLip2Config::new(
            3,
            self.vision_hidden,
            self.vision_layers,
            self.vision_heads,
            self.vision_hidden * 4,
            self.patch_size,
            self.image_size,
            self.rms_norm_eps,
        )
    }
}

// -- Decoder layer ------------------------------------------------------------

/// Single GLM decoder layer: pre-norm GQA attention + pre-norm SwiGLU MLP.
#[derive(Clone)]
pub struct GlmDecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: MultiHeadAttention,
    post_attention_layernorm: RmsNorm,
    mlp_gate: Linear,
    mlp_up: Linear,
    mlp_down: Linear,
}

impl std::fmt::Debug for GlmDecoderLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlmDecoderLayer")
            .field("self_attn", &self.self_attn)
            .finish_non_exhaustive()
    }
}

impl GlmDecoderLayer {
    /// Load a decoder layer from a VarBuilder scoped to `model.layers.{i}`.
    pub fn load(vb: impl AsRef<VarBuilder>, cfg: &GlmOcrConfig) -> Result<Self> {
        let vb = vb.as_ref();
        let h = cfg.hidden_size;

        let input_layernorm =
            RmsNorm::new(vb.get(&[h], "input_layernorm.weight")?, cfg.rms_norm_eps)?;

        let self_attn = MultiHeadAttention::load(
            vb.pp("self_attn"),
            h,
            cfg.num_heads,
            cfg.num_kv_heads,
            false, // no bias for GLM attention projections
        )?;

        let post_attention_layernorm = RmsNorm::new(
            vb.get(&[h], "post_attention_layernorm.weight")?,
            cfg.rms_norm_eps,
        )?;

        let mlp_vb = vb.pp("mlp");
        let i = cfg.intermediate_size;
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
    pub fn forward(&self, x: &DynTensor, mask: Option<&DynTensor>) -> Result<DynTensor> {
        // Pre-norm attention with residual.
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h, None, mask, None, 0)?;
        let x = residual.broadcast_add(&h)?;

        // Pre-norm SwiGLU MLP with residual.
        let residual = x.clone();
        let h = self.post_attention_layernorm.forward(&x)?;
        let gate = self.mlp_gate.forward(&h)?.silu()?;
        let up = self.mlp_up.forward(&h)?;
        let h = self.mlp_down.forward(&gate.broadcast_mul(&up)?)?;
        let output = residual.broadcast_add(&h)?;
        check_output_finite(&output, "GlmDecoderLayer")?;
        Ok(output)
    }
}

// -- Output -------------------------------------------------------------------

/// Output from GLM-OCR forward pass.
#[derive(Debug, Clone)]
pub struct GlmOcrOutput {
    /// Next-token logits from the main LM head: `[B, S, vocab_size]`.
    pub logits: DynTensor,
    /// Optional MTP logits: `[B, S, mtp_depth, vocab_size]` (present when
    /// the model has MTP heads and `run_mtp` is true).
    pub mtp_logits: Option<DynTensor>,
}

// -- Full model ---------------------------------------------------------------

/// GLM-OCR: SigLIP2 vision encoder + GLM decoder + MTP heads.
#[derive(Clone)]
pub struct GlmOcr {
    vision_encoder: SigLip2VisionEncoder,
    vision_projection: Linear,
    embed_tokens: Embedding,
    layers: Vec<GlmDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    mtp_heads: Option<MtpHead>,
    config: GlmOcrConfig,
}

impl std::fmt::Debug for GlmOcr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlmOcr")
            .field("vision_encoder", &self.vision_encoder)
            .field("layers", &self.layers.len())
            .field("has_mtp", &self.mtp_heads.is_some())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl GlmOcr {
    /// Load the full model from a VarBuilder.
    ///
    /// Weight names follow HuggingFace `THUDM/glm-ocr-*` convention.
    pub fn load(vb: impl AsRef<VarBuilder>, cfg: GlmOcrConfig) -> Result<Self> {
        let vb = vb.as_ref();
        cfg.validate()?;

        // Vision encoder.
        let siglip_cfg = cfg.to_siglip2_config()?;
        let vision_encoder = SigLip2VisionEncoder::load(vb.pp("vision_model"), &siglip_cfg)?;

        // Vision projection: vision_hidden -> decoder hidden.
        let proj_vb = vb.pp("vision_projection");
        let proj_w = proj_vb.get(&[cfg.hidden_size, cfg.vision_hidden], "weight")?;
        let proj_b = if proj_vb.contains_tensor("bias") {
            Some(proj_vb.get(&[cfg.hidden_size], "bias")?)
        } else {
            None
        };
        let vision_projection = Linear::new(proj_w, proj_b)?;

        // Text embedding.
        let model_vb = vb.pp("model");
        let embed_weight =
            model_vb.get(&[cfg.vocab_size, cfg.hidden_size], "embed_tokens.weight")?;
        let embed_tokens = Embedding::new(embed_weight)?;

        // Decoder layers.
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let layer_vb = model_vb.pp(format!("layers.{i}"));
            layers.push(GlmDecoderLayer::load(&layer_vb, &cfg)?);
        }

        // Final norm.
        let norm = RmsNorm::new(
            model_vb.get(&[cfg.hidden_size], "norm.weight")?,
            cfg.rms_norm_eps,
        )?;

        // LM head.
        let lm_head = Linear::new(
            vb.get(&[cfg.vocab_size, cfg.hidden_size], "lm_head.weight")?,
            None,
        )?;

        // MTP heads (optional: loaded if mtp_depth > 0 and weights exist).
        let mtp_heads = if cfg.mtp_depth > 0 {
            let mtp_vb = vb.pp("mtp");
            let mtp_cfg = MtpHeadConfig {
                num_predict_tokens: cfg.mtp_depth,
                hidden_size: cfg.hidden_size,
                vocab_size: cfg.vocab_size,
                shared_trunk: false,
                per_head_norm: false,
                norm_eps: cfg.rms_norm_eps,
            };
            Some(MtpHead::load(&mtp_vb, mtp_cfg)?)
        } else {
            None
        };

        Ok(Self {
            vision_encoder,
            vision_projection,
            embed_tokens,
            layers,
            norm,
            lm_head,
            mtp_heads,
            config: cfg,
        })
    }

    /// Forward: image `[B,3,H,W]` + token IDs -> [`GlmOcrOutput`].
    pub fn forward(&self, image: &DynTensor, input_ids: &[usize]) -> Result<GlmOcrOutput> {
        let hidden = self.forward_hidden(image, input_ids)?;

        // LM head: [B, S, D] -> [B, S, V].
        let logits = self.lm_head.forward(&hidden)?;
        check_output_finite(&logits, "GlmOcr.logits")?;

        // MTP heads: [B, S, D] -> [B, S, mtp_depth, V].
        let mtp_logits = if let Some(ref mtp) = self.mtp_heads {
            let mtp_out = mtp.forward(&hidden)?;
            check_output_finite(&mtp_out, "GlmOcr.mtp_logits")?;
            Some(mtp_out)
        } else {
            None
        };

        Ok(GlmOcrOutput { logits, mtp_logits })
    }

    /// Vision encode + decoder pass, returning hidden states `[B, S, D]`.
    fn forward_hidden(&self, image: &DynTensor, input_ids: &[usize]) -> Result<DynTensor> {
        // 1. Vision: encode image -> [B, num_patches, vision_hidden].
        let vision_out = self.vision_encoder.forward(image, PoolingStrategy::None)?;

        // 2. Project: [B, N, vision_hidden] -> [B, N, hidden_size].
        let vision_projected = self.vision_projection.forward(&vision_out)?;

        // 3. Embed text tokens: [text_len, hidden_size].
        let text_embedded = self.embed_tokens.forward_ids(input_ids)?;
        let text_embedded = if text_embedded.rank() == 2 {
            text_embedded.unsqueeze(0)?
        } else {
            text_embedded
        };

        // 4. Concatenate vision + text along sequence dimension.
        let combined = DynTensor::cat(&[&vision_projected, &text_embedded], 1)?;

        // 5. Build causal mask.
        let total_len = combined.dim(1)?;
        let mask =
            nn_core::layers::causal_mask_dtype(total_len, combined.dtype(), &combined.device())?;

        // 6. Run through decoder layers — skip per-layer NaN checks to avoid
        // N flush+readback cycles that cause Metal GPU timeout.
        let hidden = with_nan_check_policy(NanCheckPolicy::Skip, || -> Result<DynTensor> {
            let mut h = combined;
            for layer in &self.layers {
                h = layer.forward(&h, Some(&mask))?;
            }
            Ok(h)
        })?;

        // 7. Final norm.
        let hidden = self.norm.forward(&hidden)?;
        // Defense-in-depth: validate final output OUTSIDE the skip scope
        check_output_finite(&hidden, "GlmOcr.hidden")?;
        Ok(hidden)
    }

    /// Generate text from an image using MTP speculative decoding.
    /// Requires `mtp_depth > 0`.
    pub fn generate_with_mtp(
        &self,
        image: &DynTensor,
        prompt_ids: &[usize],
        max_tokens: usize,
    ) -> Result<SpeculativeOutput> {
        let mtp = self.mtp_heads.as_ref().ok_or_else(|| {
            TensorError::InvalidShape("GlmOcr::generate_with_mtp requires mtp_depth > 0".into())
        })?;

        let spec_cfg = SpeculativeConfig::new(max_tokens, mtp.num_predict_tokens());

        // Clone image and prompt for closures.
        let image_ref = image.clone();
        let mut kv_cache = DummyKvCache;

        greedy_decode_with_verification(
            |hidden_states| mtp.forward_per_head(hidden_states),
            |input_ids_tensor, _cache| {
                let ids = input_ids_tensor.to_flat_vec::<u32>()?;
                let ids_usize: Vec<usize> = ids.iter().map(|&v| v as usize).collect();
                let hidden = self.forward_hidden(&image_ref, &ids_usize)?;
                let logits = self.lm_head.forward(&hidden)?;
                Ok((logits, hidden))
            },
            prompt_ids,
            &mut kv_cache,
            &spec_cfg,
            &image.device(),
        )
    }

    /// Access the model configuration.
    #[must_use]
    pub fn config(&self) -> &GlmOcrConfig {
        &self.config
    }

    /// Access the decoder layers.
    #[must_use]
    pub fn decoder_layers(&self) -> &[GlmDecoderLayer] {
        &self.layers
    }

    /// Access the MTP heads (if present).
    #[must_use]
    pub fn mtp_heads(&self) -> Option<&MtpHead> {
        self.mtp_heads.as_ref()
    }

    /// Access the vision encoder.
    #[must_use]
    pub fn vision_encoder(&self) -> &SigLip2VisionEncoder {
        &self.vision_encoder
    }
}

// -- Dummy KV cache (placeholder for speculative decode interface) ------------

/// Minimal KV cache backend — GLM-OCR re-computes hidden states per step.
struct DummyKvCache;

impl nn_core::layers::KvCacheBackend for DummyKvCache {
    fn layer_backend_mut(
        &mut self,
        _index: usize,
    ) -> Result<&mut dyn nn_core::layers::KvCacheLayerBackend> {
        Err(TensorError::InvalidShape(
            "DummyKvCache has no layers".into(),
        ))
    }

    fn num_layers(&self) -> usize {
        0
    }

    fn seq_len(&self) -> usize {
        0
    }

    fn reset(&mut self) {}
}

#[cfg(test)]
#[path = "glm_ocr_tests.rs"]
mod tests;
