// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Qwen3-VL multimodal model builder.
//!
//! Supports two usage modes:
//!
//! 1. **Pre-encoded vision (2B/7B/30B):** Accepts pre-encoded `vision_features`
//!    from an external encoder, projects via a linear merger, and feeds into
//!    the decoder. Used for FireRed-OCR and other config-driven variants.
//!
//! 2. **Full vision encoder (8B):** Includes a 27-layer ViT with 2D RoPE,
//!    Conv3d patch embedding (emulated via Conv2d), and DeepStack feature
//!    extraction at layers 8, 16, 24. DeepStack features are injected
//!    additively into decoder layers 0, 1, 2.
//!
//! Architecture:
//! - **Vision encoder (8B):** 27-layer ViT (1152 hidden, 16 heads, head_dim 72,
//!   patch_size 16) with full attention and DeepStack mergers
//! - **Vision-language merger:** Linear(vision_hidden -> decoder_hidden) for 2B/7B;
//!   PatchMerger (2x2 spatial, MLP) for 8B
//! - **Decoder:** N-layer Qwen3 transformer (GQA, RmsNorm, SwiGLU MLP)
//! - **LM head:** Linear(decoder_hidden -> vocab_size)
//!
//! Reference: `Qwen/Qwen3-VL-2B-Instruct`, `Qwen/Qwen3-VL-8B-Instruct`.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{
    causal_mask_with_offset, check_output_finite, with_nan_check_policy, Embedding, KvCache,
    KvCacheLayer, Linear, Module, MultiHeadAttention, NanCheckPolicy, RmsNorm,
};
use nn_core::var_builder::VarBuilder;
use nn_core::{Result, TensorError};

// -- Vision encoder (8B full ViT with DeepStack) ------------------------------
#[path = "qwen3_vl_vision.rs"]
pub mod vision;
pub use vision::{Qwen3VlVisionEncoder, Qwen3VlVisionOutput};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// RMSNorm epsilon (Qwen3 default).
const RMS_NORM_EPS: f64 = 1e-6;

/// Configuration for Qwen3-VL multimodal models.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Qwen3VLConfig {
    /// Decoder hidden dimension.
    pub hidden_size: usize,
    /// Number of decoder Q attention heads.
    pub num_heads: usize,
    /// Number of decoder KV heads (GQA).
    pub num_kv_heads: usize,
    /// SwiGLU intermediate dimension.
    pub intermediate_size: usize,
    /// Number of decoder transformer layers.
    pub num_layers: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Vision encoder hidden dimension.
    pub vision_hidden: usize,
    /// Vision encoder attention heads.
    pub vision_heads: usize,
    /// Vision encoder layers.
    pub vision_layers: usize,
    /// Vision spatial patch size in pixels.
    pub vision_patch_size: usize,
    /// Vision temporal patch size (frames).
    pub vision_temporal_patch: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f64,
    /// MoE: total number of experts (0 = dense model).
    pub num_experts: usize,
    /// MoE: number of active experts per token.
    pub active_experts: usize,
}

impl Qwen3VLConfig {
    /// Create the 2B configuration (FireRed-OCR tier).
    #[must_use]
    pub fn preset_2b() -> Self {
        Self {
            hidden_size: 1536,
            num_heads: 12,
            num_kv_heads: 2,
            intermediate_size: 8960,
            num_layers: 28,
            vocab_size: 152064,
            vision_hidden: 1280,
            vision_heads: 16,
            vision_layers: 32,
            vision_patch_size: 14,
            vision_temporal_patch: 2,
            rms_norm_eps: RMS_NORM_EPS,
            num_experts: 0,
            active_experts: 0,
        }
    }

    /// Create the 7B configuration.
    #[must_use]
    pub fn preset_7b() -> Self {
        Self {
            hidden_size: 3584,
            num_heads: 28,
            num_kv_heads: 4,
            intermediate_size: 18944,
            num_layers: 28,
            vocab_size: 152064,
            vision_hidden: 1280,
            vision_heads: 16,
            vision_layers: 32,
            vision_patch_size: 14,
            vision_temporal_patch: 2,
            rms_norm_eps: RMS_NORM_EPS,
            num_experts: 0,
            active_experts: 0,
        }
    }

    /// Create the 30B-A3B MoE configuration.
    #[must_use]
    pub fn preset_30b_a3b() -> Self {
        Self {
            hidden_size: 3584,
            num_heads: 28,
            num_kv_heads: 4,
            intermediate_size: 2560,
            num_layers: 48,
            vocab_size: 152064,
            vision_hidden: 1280,
            vision_heads: 16,
            vision_layers: 32,
            vision_patch_size: 14,
            vision_temporal_patch: 2,
            rms_norm_eps: RMS_NORM_EPS,
            num_experts: 128,
            active_experts: 8,
        }
    }

    /// Create the 8B configuration (Qwen3-VL-8B-Instruct).
    ///
    /// This variant has a built-in 27-layer ViT vision encoder with DeepStack
    /// feature extraction. Use [`Qwen3VL::load_with_vision_encoder`] to load
    /// the full model with integrated vision encoding.
    #[must_use]
    pub fn preset_8b() -> Self {
        Self {
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            intermediate_size: 11008,
            num_layers: 32,
            vocab_size: 152064,
            vision_hidden: 1152,
            vision_heads: 16,
            vision_layers: 27,
            vision_patch_size: 16,
            vision_temporal_patch: 2,
            rms_norm_eps: RMS_NORM_EPS,
            num_experts: 0,
            active_experts: 0,
        }
    }

    /// Whether this is a Mixture-of-Experts model.
    #[must_use]
    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
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
        if self.num_heads == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen3VLConfig: num_heads must be > 0",
            });
        }
        if self.num_kv_heads == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen3VLConfig: num_kv_heads must be > 0",
            });
        }
        if !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen3VLConfig: hidden_size must be divisible by num_heads",
            });
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen3VLConfig: num_heads must be divisible by num_kv_heads",
            });
        }
        if self.vision_patch_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen3VLConfig: vision_patch_size must be > 0",
            });
        }
        if self.vision_temporal_patch == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen3VLConfig: vision_temporal_patch must be > 0",
            });
        }
        if self.is_moe() && self.active_experts == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen3VLConfig: MoE requires active_experts > 0",
            });
        }
        if self.is_moe() && self.active_experts > self.num_experts {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen3VLConfig: active_experts must be <= num_experts",
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

/// Single Qwen3 decoder layer: pre-norm GQA attention + pre-norm SwiGLU MLP.
#[derive(Clone)]
pub struct Qwen3DecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: MultiHeadAttention,
    post_attention_layernorm: RmsNorm,
    mlp_gate: Linear,
    mlp_up: Linear,
    mlp_down: Linear,
}

impl std::fmt::Debug for Qwen3DecoderLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3DecoderLayer")
            .field("self_attn", &self.self_attn)
            .finish_non_exhaustive()
    }
}

impl Qwen3DecoderLayer {
    /// Load a decoder layer from a VarBuilder scoped to `model.layers.{i}`.
    pub fn load(vb: impl AsRef<VarBuilder>, cfg: &Qwen3VLConfig) -> Result<Self> {
        let vb = vb.as_ref();
        let h = cfg.hidden_size;

        let input_layernorm =
            RmsNorm::new(vb.get(&[h], "input_layernorm.weight")?, cfg.rms_norm_eps)?;

        let self_attn = MultiHeadAttention::load(
            vb.pp("self_attn"),
            h,
            cfg.num_heads,
            cfg.num_kv_heads,
            false, // no bias for Qwen3 attention projections
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
        check_output_finite(&output, "Qwen3DecoderLayer")?;
        Ok(output)
    }

    /// Forward pass with KV cache for autoregressive decoding.
    ///
    /// Appends new K/V to `cache`, attends over the full cached sequence.
    /// During decode, `x` is typically `[B, 1, D]` (single token).
    pub fn forward_cached(
        &self,
        x: &DynTensor,
        mask: Option<&DynTensor>,
        cache: &mut KvCacheLayer,
    ) -> Result<DynTensor> {
        // Pre-norm attention with residual
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let h = self
            .self_attn
            .forward_kv_cached(&h, None, cache, mask, None, 0)?;
        let x = residual.broadcast_add(&h)?;

        // Pre-norm SwiGLU MLP with residual
        let residual = x.clone();
        let h = self.post_attention_layernorm.forward(&x)?;
        let gate = self.mlp_gate.forward(&h)?.silu()?;
        let up = self.mlp_up.forward(&h)?;
        let h = self.mlp_down.forward(&gate.broadcast_mul(&up)?)?;
        let output = residual.broadcast_add(&h)?;
        check_output_finite(&output, "Qwen3DecoderLayer")?;
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

/// Number of initial decoder layers that receive additive DeepStack features.
const DEEPSTACK_INJECT_LAYERS: usize = vision::DEEPSTACK_INDEXES.len();

/// Qwen3-VL: optional vision encoder + merger + decoder-only language model.
///
/// Two loading modes:
/// - [`Qwen3VL::load`]: Decoder-only with a simple linear vision merger.
///   Accepts pre-encoded `vision_features` from an external encoder.
/// - [`Qwen3VL::load_with_vision_encoder`]: Full model including the 27-layer
///   ViT vision encoder with DeepStack mergers. Accepts raw images.
pub struct Qwen3VL {
    embed_tokens: Embedding,
    vision_merger: Linear,
    vision_encoder: Option<Qwen3VlVisionEncoder>,
    decoder_layers: Vec<Qwen3DecoderLayer>,
    decoder_norm: RmsNorm,
    lm_head: Linear,
    config: Qwen3VLConfig,
}

impl Clone for Qwen3VL {
    fn clone(&self) -> Self {
        Self {
            embed_tokens: self.embed_tokens.clone(),
            vision_merger: self.vision_merger.clone(),
            vision_encoder: None, // Qwen3VlVisionEncoder is not Clone
            decoder_layers: self.decoder_layers.clone(),
            decoder_norm: self.decoder_norm.clone(),
            lm_head: self.lm_head.clone(),
            config: self.config.clone(),
        }
    }
}

impl std::fmt::Debug for Qwen3VL {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3VL")
            .field("decoder_layers", &self.decoder_layers.len())
            .field("has_vision_encoder", &self.vision_encoder.is_some())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl Qwen3VL {
    /// Load the decoder-only model with a simple linear vision merger.
    ///
    /// Weight names follow HuggingFace `Qwen/Qwen3-VL-2B-Instruct`.
    /// Use [`Self::load_with_vision_encoder`] for the 8B model with
    /// integrated vision encoding.
    pub fn load(vb: impl AsRef<VarBuilder>, cfg: Qwen3VLConfig) -> Result<Self> {
        let vb = vb.as_ref();
        cfg.validate()?;

        // Vision-language merger: vision_hidden -> decoder_hidden
        let merger_vb = vb.pp("visual").pp("merger");
        let merger_w = merger_vb.get(&[cfg.hidden_size, cfg.vision_hidden], "weight")?;
        let merger_b = if merger_vb.contains_tensor("bias") {
            Some(merger_vb.get(&[cfg.hidden_size], "bias")?)
        } else {
            None
        };
        let vision_merger = Linear::new(merger_w, merger_b)?;

        // Text embedding
        let model_vb = vb.pp("model");
        let embed_weight =
            model_vb.get(&[cfg.vocab_size, cfg.hidden_size], "embed_tokens.weight")?;
        let embed_tokens = Embedding::new(embed_weight)?;

        // Decoder layers
        let mut decoder_layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let layer_vb = model_vb.pp(format!("layers.{i}"));
            decoder_layers.push(Qwen3DecoderLayer::load(&layer_vb, &cfg)?);
        }

        // Final norm + LM head
        let decoder_norm = RmsNorm::new(
            model_vb.get(&[cfg.hidden_size], "norm.weight")?,
            cfg.rms_norm_eps,
        )?;
        let lm_head = Linear::new(
            vb.get(&[cfg.vocab_size, cfg.hidden_size], "lm_head.weight")?,
            None,
        )?;

        Ok(Self {
            embed_tokens,
            vision_merger,
            vision_encoder: None,
            decoder_layers,
            decoder_norm,
            lm_head,
            config: cfg,
        })
    }

    /// Load the full model including the 27-layer ViT vision encoder.
    ///
    /// This loads the vision encoder from `visual.*` keys, the decoder from
    /// `model.*` keys, and the LM head from `lm_head.*`. The vision merger
    /// from the vision encoder output (`[batch, merged_tokens, 4096]`) is
    /// used directly without an additional linear projection.
    ///
    /// Weight names follow HuggingFace `Qwen/Qwen3-VL-8B-Instruct`.
    ///
    /// # Errors
    ///
    /// Returns an error if weight tensors are missing or have wrong shapes.
    pub fn load_with_vision_encoder(
        vb: impl AsRef<VarBuilder>,
        cfg: Qwen3VLConfig,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        cfg.validate()?;

        // Load the full vision encoder (patch_embed + 27 blocks + mergers)
        let vision_encoder = Qwen3VlVisionEncoder::load(vb)?;

        // Vision encoder's main merger already outputs [B, tokens, 4096].
        // Create a pass-through merger (identity) since projection is done
        // by the PatchMerger. We use a zero-weight + identity-like linear
        // that the forward path will bypass when vision_encoder is present.
        let merger_vb = vb.pp("visual").pp("merger");
        let merger_w = merger_vb.get(&[cfg.hidden_size, cfg.vision_hidden], "linear_fc2.weight")?;
        let vision_merger = Linear::new(merger_w, None)?;

        // Text embedding
        let model_vb = vb.pp("model");
        let embed_weight =
            model_vb.get(&[cfg.vocab_size, cfg.hidden_size], "embed_tokens.weight")?;
        let embed_tokens = Embedding::new(embed_weight)?;

        // Decoder layers
        let mut decoder_layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let layer_vb = model_vb.pp(format!("layers.{i}"));
            decoder_layers.push(Qwen3DecoderLayer::load(&layer_vb, &cfg)?);
        }

        // Final norm + LM head
        let decoder_norm = RmsNorm::new(
            model_vb.get(&[cfg.hidden_size], "norm.weight")?,
            cfg.rms_norm_eps,
        )?;
        let lm_head = Linear::new(
            vb.get(&[cfg.vocab_size, cfg.hidden_size], "lm_head.weight")?,
            None,
        )?;

        Ok(Self {
            embed_tokens,
            vision_merger,
            vision_encoder: Some(vision_encoder),
            decoder_layers,
            decoder_norm,
            lm_head,
            config: cfg,
        })
    }

    /// Encode an image through the vision encoder.
    ///
    /// Only available when loaded with [`Self::load_with_vision_encoder`].
    ///
    /// # Arguments
    ///
    /// * `image` - `[batch, 3, height, width]` where H, W divisible by 32
    ///
    /// # Returns
    ///
    /// [`Qwen3VlVisionOutput`] with main features and 3 DeepStack feature tensors.
    ///
    /// # Errors
    ///
    /// Returns an error if no vision encoder was loaded or image dims are invalid.
    pub fn vision_encode(&self, image: &DynTensor) -> Result<Qwen3VlVisionOutput> {
        let encoder = self.vision_encoder.as_ref().ok_or_else(|| {
            TensorError::InvalidShape(
                "Qwen3VL: vision_encode requires load_with_vision_encoder".into(),
            )
        })?;
        encoder.forward(image)
    }

    /// Whether this model has an integrated vision encoder.
    #[must_use]
    pub fn has_vision_encoder(&self) -> bool {
        self.vision_encoder.is_some()
    }

    /// Access the vision encoder (if loaded).
    #[must_use]
    pub fn vision_encoder(&self) -> Option<&Qwen3VlVisionEncoder> {
        self.vision_encoder.as_ref()
    }

    /// Forward pass: optional vision features + text token IDs -> logits.
    ///
    /// - `vision_features`: `[B, N_vis, vision_hidden]` pre-encoded vision
    ///   tokens. Pass `None` for text-only inference.
    /// - `input_ids`: token IDs as `&[usize]` (flat, length = text_len for B=1)
    ///
    /// Returns: `[B, total_seq_len, vocab_size]` logits.
    pub fn forward(
        &self,
        vision_features: Option<&DynTensor>,
        input_ids: &[usize],
    ) -> Result<DynTensor> {
        self.decoder_forward(vision_features, input_ids, None)
    }

    /// Forward pass with DeepStack feature injection.
    ///
    /// When `deepstack` is provided, features from the vision encoder's
    /// DeepStack mergers are additively injected into decoder layers 0, 1, 2.
    ///
    /// - `vision_features`: Main vision features `[B, N_vis, hidden_size]`.
    /// - `input_ids`: Token IDs.
    /// - `deepstack`: Optional DeepStack features from [`Qwen3VlVisionOutput`].
    ///
    /// Returns: `[B, total_seq_len, vocab_size]` logits.
    pub fn forward_with_deepstack(
        &self,
        vision_features: Option<&DynTensor>,
        input_ids: &[usize],
        deepstack: Option<&[DynTensor]>,
    ) -> Result<DynTensor> {
        self.decoder_forward(vision_features, input_ids, deepstack)
    }

    /// Internal decoder forward with optional DeepStack injection.
    fn decoder_forward(
        &self,
        vision_features: Option<&DynTensor>,
        input_ids: &[usize],
        deepstack: Option<&[DynTensor]>,
    ) -> Result<DynTensor> {
        // 1. Embed text tokens: [text_len, hidden]
        let text_embedded = self.embed_tokens.forward_ids(input_ids)?;
        let text_embedded = if text_embedded.rank() == 2 {
            text_embedded.unsqueeze(0)?
        } else {
            text_embedded
        };

        // 2. Combine vision + text if vision features are provided
        let combined = if let Some(vis) = vision_features {
            // For 8B model, features come from the PatchMerger and are
            // already in decoder hidden space. For 2B/7B, project via merger.
            let vis_projected = if self.vision_encoder.is_some() {
                vis.clone()
            } else {
                self.vision_merger.forward(vis)?
            };
            DynTensor::cat(&[&vis_projected, &text_embedded], 1)?
        } else {
            text_embedded
        };

        // 3. Build causal mask
        let total_len = combined.dim(1)?;
        let mask =
            nn_core::layers::causal_mask_dtype(total_len, combined.dtype(), &combined.device())?;

        // 4. Run through decoder layers with optional DeepStack injection —
        // skip per-layer NaN checks to avoid N flush+readback cycles that
        // cause Metal GPU timeout.
        let hidden = with_nan_check_policy(NanCheckPolicy::Skip, || -> Result<DynTensor> {
            let mut h = combined;
            for (i, layer) in self.decoder_layers.iter().enumerate() {
                h = layer.forward(&h, Some(&mask))?;

                // DeepStack injection at layers 0, 1, 2
                if i < DEEPSTACK_INJECT_LAYERS {
                    if let Some(ds) = deepstack {
                        if let Some(ds_feat) = ds.get(i) {
                            h = h.broadcast_add(ds_feat)?;
                        }
                    }
                }
            }
            Ok(h)
        })?;

        // 5. Final norm -> LM head -> logits
        let hidden = self.decoder_norm.forward(&hidden)?;
        let logits = self.lm_head.forward(&hidden)?;
        // Defense-in-depth: validate final output OUTSIDE the skip scope
        check_output_finite(&logits, "Qwen3VL")?;
        Ok(logits)
    }

    /// Forward pass with KV cache for autoregressive decoding.
    ///
    /// During prefill, pass all vision features and text tokens at once.
    /// During decode, pass `vision_features: None` and a single token ID.
    ///
    /// Returns logits for the **last token position** only: `[B, 1, vocab_size]`.
    pub fn forward_cached(
        &self,
        vision_features: Option<&DynTensor>,
        input_ids: &[usize],
        cache: &mut KvCache,
    ) -> Result<DynTensor> {
        self.decoder_forward_cached(vision_features, input_ids, cache, None)
    }

    /// Forward pass with KV cache and DeepStack feature injection.
    ///
    /// DeepStack features are additively merged into decoder layers 0, 1, 2
    /// during the prefill pass. Pass `deepstack: None` during decode steps.
    ///
    /// Returns logits for the **last token position** only: `[B, 1, vocab_size]`.
    pub fn forward_cached_with_deepstack(
        &self,
        vision_features: Option<&DynTensor>,
        input_ids: &[usize],
        cache: &mut KvCache,
        deepstack: Option<&[DynTensor]>,
    ) -> Result<DynTensor> {
        self.decoder_forward_cached(vision_features, input_ids, cache, deepstack)
    }

    /// Internal cached decoder forward with optional DeepStack injection.
    fn decoder_forward_cached(
        &self,
        vision_features: Option<&DynTensor>,
        input_ids: &[usize],
        cache: &mut KvCache,
        deepstack: Option<&[DynTensor]>,
    ) -> Result<DynTensor> {
        if cache.num_layers() != self.decoder_layers.len() {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen3VL: KV cache layer count does not match model decoder layers",
            });
        }

        // 1. Embed text tokens: [text_len, hidden] -> [1, text_len, hidden]
        let text_embedded = self.embed_tokens.forward_ids(input_ids)?;
        let text_embedded = if text_embedded.rank() == 2 {
            text_embedded.unsqueeze(0)?
        } else {
            text_embedded
        };

        // 2. Combine vision + text if vision features are provided
        let combined = if let Some(vis) = vision_features {
            let vis_projected = if self.vision_encoder.is_some() {
                vis.clone()
            } else {
                self.vision_merger.forward(vis)?
            };
            DynTensor::cat(&[&vis_projected, &text_embedded], 1)?
        } else {
            text_embedded
        };

        // 3. Build causal mask with offset for cached positions
        let seq_len = combined.dim(1)?;
        let cached_len = cache.seq_len();
        let total_seq = cached_len + seq_len;
        let mask = if seq_len > 1 && total_seq > 1 {
            Some(causal_mask_with_offset(
                seq_len,
                total_seq,
                combined.dtype(),
                &combined.device(),
            )?)
        } else {
            None
        };

        // 4. Run through decoder layers with KV cache + DeepStack —
        // skip per-layer NaN checks to avoid N flush+readback cycles that
        // cause Metal GPU timeout.
        let hidden = with_nan_check_policy(NanCheckPolicy::Skip, || -> Result<DynTensor> {
            let mut h = combined;
            for (i, layer) in self.decoder_layers.iter().enumerate() {
                let layer_cache = cache.layer_mut(i)?;
                h = layer.forward_cached(&h, mask.as_ref(), layer_cache)?;

                // DeepStack injection at layers 0, 1, 2
                if i < DEEPSTACK_INJECT_LAYERS {
                    if let Some(ds) = deepstack {
                        if let Some(ds_feat) = ds.get(i) {
                            h = h.broadcast_add(ds_feat)?;
                        }
                    }
                }
            }
            Ok(h)
        })?;

        // 5. Final norm -> LM head -> logits (last position only)
        let hidden = self.decoder_norm.forward(&hidden)?;
        let last_pos = hidden.dim(1)?.saturating_sub(1);
        let last_hidden = hidden.narrow(1, last_pos, 1)?;
        let logits = self.lm_head.forward(&last_hidden)?;
        // Defense-in-depth: validate final output OUTSIDE the skip scope
        check_output_finite(&logits, "Qwen3VL")?;
        Ok(logits)
    }

    /// Create a fresh [`KvCache`] sized for this model.
    #[must_use]
    pub fn create_cache(&self) -> KvCache {
        KvCache::new(self.decoder_layers.len())
    }

    /// Access the model configuration.
    #[must_use]
    pub fn config(&self) -> &Qwen3VLConfig {
        &self.config
    }

    /// Access the decoder layers.
    #[must_use]
    pub fn decoder_layers(&self) -> &[Qwen3DecoderLayer] {
        &self.decoder_layers
    }
}

// -- Generation (KV-cached autoregressive decoding) ---------------------------
#[path = "qwen3_vl_generate.rs"]
pub mod generate;

#[cfg(test)]
#[path = "qwen3_vl_tests.rs"]
mod tests;
