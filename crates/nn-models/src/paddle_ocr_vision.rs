// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PaddleOCR-VL-1.5 vision encoder: SigLIP-style 27-layer ViT with 2D RoPE
//! and 2x2 spatial merge projector.
//!
//! Input: `[batch, 3, H, W]` with H, W divisible by `patch_size * spatial_merge_size` (28).
//! Output: `[batch, merged_tokens, 1024]` visual embeddings.
//!
//! Expected HuggingFace safetensors keys:
//! - `visual.vision_model.embeddings.patch_embedding.{weight,bias}`
//! - `visual.vision_model.embeddings.position_embedding.weight`
//! - `visual.vision_model.encoder.layers.{i}.*`
//! - `visual.vision_model.post_layernorm.{weight,bias}`
//! - `mlp_AR.{pre_norm,linear_1,linear_2}.*`

use nn_core::layers::{
    check_output_finite, rope, sdpa, Conv2d, Conv2dConfig, LayerNorm, Linear, Module,
};
use nn_core::{Device, DynTensor, Result, TensorError, VarBuilder};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Vision encoder hidden dimension.
pub const VISION_HIDDEN: usize = 1152;
/// Vision encoder intermediate (MLP) dimension.
pub const VISION_INTERMEDIATE: usize = 4304;
/// Number of vision transformer layers.
pub const VISION_LAYERS: usize = 27;
/// Number of attention heads in the vision encoder.
pub const VISION_HEADS: usize = 16;
/// Per-head dimension (1152 / 16 = 72).
pub const VISION_HEAD_DIM: usize = VISION_HIDDEN / VISION_HEADS;
/// Patch size in pixels.
pub const VISION_PATCH_SIZE: usize = 14;
/// Default image size for position embedding grid.
pub const VISION_IMAGE_SIZE: usize = 384;
/// LayerNorm epsilon for the vision encoder.
pub const VISION_LN_EPS: f64 = 1e-6;
/// Spatial merge factor (2x2 = 4 patches merged into 1).
pub const SPATIAL_MERGE_SIZE: usize = 2;
/// Output dimension of the merge projector.
pub const MERGE_OUTPUT_DIM: usize = 1024;
/// LayerNorm epsilon for the merge projector.
pub const MERGE_LN_EPS: f64 = 1e-5;
/// RoPE frequency base for the SigLIP vision encoder.
const VISION_ROPE_BASE: f64 = 10_000.0;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// PaddleOCR-VL-1.5 vision encoder configuration.
#[derive(Debug, Clone, Copy)]
pub struct PaddleOcrVlVisionConfig {
    /// Number of input channels (3 for RGB).
    pub num_channels: usize,
    /// Hidden size of the vision transformer.
    pub hidden_size: usize,
    /// Intermediate size for MLP layers.
    pub intermediate_size: usize,
    /// Number of transformer layers.
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Patch size in pixels.
    pub patch_size: usize,
    /// Default image size for position grid.
    pub image_size: usize,
    /// LayerNorm epsilon for the vision encoder.
    pub layer_norm_eps: f64,
    /// Spatial merge size (2x2).
    pub spatial_merge_size: usize,
    /// Output dimension after merging.
    pub merge_output_size: usize,
    /// LayerNorm epsilon for the merge projector.
    pub merge_layer_norm_eps: f64,
}

impl Default for PaddleOcrVlVisionConfig {
    fn default() -> Self {
        Self {
            num_channels: 3,
            hidden_size: VISION_HIDDEN,
            intermediate_size: VISION_INTERMEDIATE,
            num_hidden_layers: VISION_LAYERS,
            num_attention_heads: VISION_HEADS,
            patch_size: VISION_PATCH_SIZE,
            image_size: VISION_IMAGE_SIZE,
            layer_norm_eps: VISION_LN_EPS,
            spatial_merge_size: SPATIAL_MERGE_SIZE,
            merge_output_size: MERGE_OUTPUT_DIM,
            merge_layer_norm_eps: MERGE_LN_EPS,
        }
    }
}

impl PaddleOcrVlVisionConfig {
    /// Validate configuration consistency.
    pub fn validate(self) -> Result<Self> {
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.patch_size == 0
            || self.spatial_merge_size == 0
            || self.merge_output_size == 0
        {
            return Err(TensorError::InvalidShape(
                "PaddleOcrVlVisionConfig: dimensions must be > 0".into(),
            ));
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(TensorError::InvalidShape(format!(
                "PaddleOcrVlVisionConfig: hidden_size {} not divisible by num_attention_heads {}",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if self.image_size / self.patch_size == 0 {
            return Err(TensorError::InvalidShape(format!(
                "PaddleOcrVlVisionConfig: image_size {} too small for patch_size {}",
                self.image_size, self.patch_size
            )));
        }
        Ok(self)
    }

    /// Per-head dimension.
    #[must_use]
    pub fn head_dim(self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Default position grid size (image_size / patch_size).
    fn position_grid_size(self) -> usize {
        self.image_size / self.patch_size
    }

    /// Hidden size after spatial merging (hidden * merge^2).
    fn spatial_merge_hidden_size(self) -> usize {
        self.hidden_size * self.spatial_merge_size * self.spatial_merge_size
    }
}

// ---------------------------------------------------------------------------
// SigLIP-style 2D RoPE
// ---------------------------------------------------------------------------

/// Precomputed cos/sin tables for the SigLIP 2D RoPE.
#[derive(Clone)]
struct SigLIPRoPE {
    /// `[1, 1, seq_len, rope_dim]`
    cos: DynTensor,
    /// `[1, 1, seq_len, rope_dim]`
    sin: DynTensor,
}

impl SigLIPRoPE {
    /// Build cos/sin tables for given grid positions.
    fn new(
        head_dim: usize,
        h_positions: &[usize],
        w_positions: &[usize],
        base: f64,
        device: &Device,
    ) -> Result<Self> {
        let seq_len = h_positions.len();
        if seq_len != w_positions.len() {
            return Err(TensorError::DataLengthMismatch {
                expected: seq_len,
                actual: w_positions.len(),
            });
        }
        if !head_dim.is_multiple_of(2) || head_dim == 0 {
            return Err(TensorError::InvalidShape(
                "SigLIPRoPE: head_dim must be a positive even number".into(),
            ));
        }

        let rope_dim = head_dim / 2;
        let num_freqs = rope_dim / 2;

        let inv_freq: Vec<f64> = (0..num_freqs)
            .map(|i| {
                let exponent = (2 * i) as f64 / rope_dim as f64;
                1.0 / base.powf(exponent)
            })
            .collect();

        let mut cos_data = Vec::with_capacity(seq_len * rope_dim);
        let mut sin_data = Vec::with_capacity(seq_len * rope_dim);

        for t in 0..seq_len {
            for &freq in &inv_freq {
                let angle = h_positions[t] as f64 * freq;
                cos_data.push(angle.cos() as f32);
                sin_data.push(angle.sin() as f32);
            }
            for &freq in &inv_freq {
                let angle = w_positions[t] as f64 * freq;
                cos_data.push(angle.cos() as f32);
                sin_data.push(angle.sin() as f32);
            }
        }

        let cos = DynTensor::from_vec(cos_data, &[1, 1, seq_len, rope_dim], &Device::Cpu)?
            .to_device(device)?;
        let sin = DynTensor::from_vec(sin_data, &[1, 1, seq_len, rope_dim], &Device::Cpu)?
            .to_device(device)?;

        Ok(Self { cos, sin })
    }

    /// Apply SigLIP RoPE to `[batch, num_heads, seq_len, head_dim]`.
    fn apply(&self, x: &DynTensor) -> Result<DynTensor> {
        rope(x, &self.cos, &self.sin)
    }
}

// ---------------------------------------------------------------------------
// Patch Embedding
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct PatchEmbed {
    projection: Conv2d,
    hidden_size: usize,
}

impl PatchEmbed {
    fn load(vb: &VarBuilder, config: PaddleOcrVlVisionConfig) -> Result<Self> {
        let conv_cfg = Conv2dConfig::new(0, config.patch_size, 1);
        Ok(Self {
            projection: Conv2d::load(
                vb,
                config.num_channels,
                config.hidden_size,
                config.patch_size,
                conv_cfg,
            )?,
            hidden_size: config.hidden_size,
        })
    }

    fn forward(&self, image: &DynTensor) -> Result<(DynTensor, usize, usize)> {
        let features = self.projection.forward(image)?;
        let (batch, _channels, grid_h, grid_w) = features.dims4()?;
        let seq_len = grid_h.checked_mul(grid_w).ok_or_else(|| {
            TensorError::InvalidShape("PatchEmbed: patch grid size overflow".into())
        })?;
        let tokens = features.reshape([batch, self.hidden_size, seq_len])?;
        Ok((tokens.transpose(1, 2)?, grid_h, grid_w))
    }
}

// ---------------------------------------------------------------------------
// Vision Transformer Block
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct VisionTransformerBlock {
    layer_norm1: LayerNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    layer_norm2: LayerNorm,
    mlp_fc1: Linear,
    mlp_fc2: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl VisionTransformerBlock {
    fn load(vb: &VarBuilder, config: PaddleOcrVlVisionConfig) -> Result<Self> {
        let head_dim = config.head_dim();
        let scale = 1.0 / (head_dim as f64).sqrt();
        let attn_vb = vb.pp("self_attn");
        let mlp_vb = vb.pp("mlp");

        Ok(Self {
            layer_norm1: LayerNorm::load(
                vb.pp("layer_norm1"),
                config.hidden_size,
                config.layer_norm_eps,
            )?,
            q_proj: Linear::load(attn_vb.pp("q_proj"), config.hidden_size, config.hidden_size)?,
            k_proj: Linear::load(attn_vb.pp("k_proj"), config.hidden_size, config.hidden_size)?,
            v_proj: Linear::load(attn_vb.pp("v_proj"), config.hidden_size, config.hidden_size)?,
            out_proj: Linear::load(
                attn_vb.pp("out_proj"),
                config.hidden_size,
                config.hidden_size,
            )?,
            layer_norm2: LayerNorm::load(
                vb.pp("layer_norm2"),
                config.hidden_size,
                config.layer_norm_eps,
            )?,
            mlp_fc1: Linear::load(
                mlp_vb.pp("fc1"),
                config.hidden_size,
                config.intermediate_size,
            )?,
            mlp_fc2: Linear::load(
                mlp_vb.pp("fc2"),
                config.intermediate_size,
                config.hidden_size,
            )?,
            num_heads: config.num_attention_heads,
            head_dim,
            scale,
        })
    }

    fn forward(&self, x: &DynTensor, siglip_rope: &SigLIPRoPE) -> Result<DynTensor> {
        let (batch, seq_len, _) = x.dims3()?;

        let hidden = self.layer_norm1.forward(x)?;
        let q = self
            .q_proj
            .forward(&hidden)?
            .reshape([batch, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(&hidden)?
            .reshape([batch, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(&hidden)?
            .reshape([batch, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        let q = siglip_rope.apply(&q)?;
        let k = siglip_rope.apply(&k)?;
        let attn = sdpa(&q, &k, &v, None, self.scale)?;
        let attn =
            attn.transpose(1, 2)?
                .reshape([batch, seq_len, self.num_heads * self.head_dim])?;
        let attn = self.out_proj.forward(&attn)?;
        let hidden = x.broadcast_add(&attn)?;

        let mlp = self.layer_norm2.forward(&hidden)?;
        let mlp = self.mlp_fc1.forward(&mlp)?.gelu()?;
        let mlp = self.mlp_fc2.forward(&mlp)?;
        hidden.broadcast_add(&mlp)
    }
}

// ---------------------------------------------------------------------------
// 2x2 Spatial Merge Projector (mlp_AR)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct SpatialMerge {
    pre_norm: LayerNorm,
    linear_1: Linear,
    linear_2: Linear,
    merge_size: usize,
    hidden_size: usize,
}

impl SpatialMerge {
    fn load(vb: &VarBuilder, config: PaddleOcrVlVisionConfig) -> Result<Self> {
        Ok(Self {
            pre_norm: LayerNorm::load(
                vb.pp("pre_norm"),
                config.hidden_size,
                config.merge_layer_norm_eps,
            )?,
            linear_1: Linear::load(
                vb.pp("linear_1"),
                config.spatial_merge_hidden_size(),
                config.spatial_merge_hidden_size(),
            )?,
            linear_2: Linear::load(
                vb.pp("linear_2"),
                config.spatial_merge_hidden_size(),
                config.merge_output_size,
            )?,
            merge_size: config.spatial_merge_size,
            hidden_size: config.hidden_size,
        })
    }

    fn forward(&self, x: &DynTensor, grid_h: usize, grid_w: usize) -> Result<DynTensor> {
        let (batch, seq_len, hidden_size) = x.dims3()?;
        if hidden_size != self.hidden_size {
            return Err(TensorError::shape_mismatch(
                vec![batch, seq_len, self.hidden_size],
                vec![batch, seq_len, hidden_size],
            ));
        }
        let expected_seq_len = grid_h.checked_mul(grid_w).ok_or_else(|| {
            TensorError::InvalidShape("SpatialMerge: patch grid size overflow".into())
        })?;
        if seq_len != expected_seq_len {
            return Err(TensorError::shape_mismatch(
                vec![batch, expected_seq_len, hidden_size],
                vec![batch, seq_len, hidden_size],
            ));
        }
        if !grid_h.is_multiple_of(self.merge_size) || !grid_w.is_multiple_of(self.merge_size) {
            return Err(TensorError::InvalidShape(format!(
                "SpatialMerge: patch grid {grid_h}x{grid_w} not divisible by merge_size {}",
                self.merge_size
            )));
        }

        let merged_h = grid_h / self.merge_size;
        let merged_w = grid_w / self.merge_size;
        let merged_hidden = self
            .hidden_size
            .checked_mul(self.merge_size)
            .and_then(|value| value.checked_mul(self.merge_size))
            .ok_or_else(|| {
                TensorError::InvalidShape("SpatialMerge: merged hidden size overflow".into())
            })?;

        let x = self.pre_norm.forward(x)?;
        let x = x.reshape([
            batch,
            merged_h,
            self.merge_size,
            merged_w,
            self.merge_size,
            self.hidden_size,
        ])?;
        let x = x.permute([0, 1, 3, 2, 4, 5])?;
        let x = x.reshape([batch, merged_h * merged_w, merged_hidden])?;
        let x = self.linear_1.forward(&x)?.gelu()?;
        self.linear_2.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// PaddleOCR-VL Vision Encoder
// ---------------------------------------------------------------------------

/// PaddleOCR-VL-1.5 vision encoder plus the `mlp_AR` 2x2 merge projector.
///
/// Input: `[batch, 3, height, width]`, with `height` and `width` divisible by
/// `patch_size * spatial_merge_size` (28 for PaddleOCR-VL-1.5).
///
/// Output: `[batch, merged_tokens, 1024]` visual embeddings ready to splice
/// into the ERNIE decoder input stream.
#[derive(Clone)]
pub struct PaddleOcrVlVisionEncoder {
    patch_embed: PatchEmbed,
    position_embedding: DynTensor,
    blocks: Vec<VisionTransformerBlock>,
    post_layernorm: LayerNorm,
    spatial_merge: SpatialMerge,
    config: PaddleOcrVlVisionConfig,
}

impl std::fmt::Debug for PaddleOcrVlVisionEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PaddleOcrVlVisionEncoder")
            .field("num_layers", &self.blocks.len())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl PaddleOcrVlVisionEncoder {
    /// Load the vision encoder from the root of a PaddleOCR-VL safetensors checkpoint.
    pub fn load(vb: &VarBuilder, config: PaddleOcrVlVisionConfig) -> Result<Self> {
        let config = config.validate()?;
        let visual_vb = vb.pp("visual").pp("vision_model");
        let embeddings_vb = visual_vb.pp("embeddings");
        let patch_embed = PatchEmbed::load(&embeddings_vb.pp("patch_embedding"), config)?;
        let num_positions = config
            .position_grid_size()
            .checked_mul(config.position_grid_size())
            .ok_or_else(|| {
                TensorError::InvalidShape("PaddleOcrVlVisionEncoder: position grid overflow".into())
            })?;
        let position_embedding = embeddings_vb.get(
            &[num_positions, config.hidden_size],
            "position_embedding.weight",
        )?;

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            blocks.push(VisionTransformerBlock::load(
                &visual_vb.pp("encoder").pp("layers").pp(index.to_string()),
                config,
            )?);
        }

        let post_layernorm = LayerNorm::load(
            visual_vb.pp("post_layernorm"),
            config.hidden_size,
            config.layer_norm_eps,
        )?;
        let spatial_merge = SpatialMerge::load(&vb.pp("mlp_AR"), config)?;

        Ok(Self {
            patch_embed,
            position_embedding,
            blocks,
            post_layernorm,
            spatial_merge,
            config,
        })
    }

    /// Run the vision encoder on an RGB image tensor.
    ///
    /// Input: `[batch, 3, height, width]`.
    /// Output: `[batch, merged_tokens, 1024]`.
    pub fn forward(&self, image: &DynTensor) -> Result<DynTensor> {
        let (_batch, channels, height, width) = image.dims4()?;
        if channels != self.config.num_channels {
            return Err(TensorError::shape_mismatch(
                vec![self.config.num_channels, height, width],
                vec![channels, height, width],
            ));
        }
        let merge_patch = self
            .config
            .patch_size
            .checked_mul(self.config.spatial_merge_size)
            .ok_or_else(|| {
                TensorError::InvalidShape(
                    "PaddleOcrVlVisionEncoder: merge patch size overflow".into(),
                )
            })?;
        if height % merge_patch != 0 || width % merge_patch != 0 {
            return Err(TensorError::InvalidShape(format!(
                "PaddleOcrVlVisionEncoder: image size {height}x{width} must be divisible by {merge_patch}"
            )));
        }

        let (mut hidden, grid_h, grid_w) = self.patch_embed.forward(image)?;
        let position_embedding = self.interpolate_position_embedding(grid_h, grid_w)?;
        hidden = hidden.broadcast_add(&position_embedding)?;

        let (h_positions, w_positions) = build_hw_positions(grid_h, grid_w)?;
        let siglip_rope = SigLIPRoPE::new(
            self.config.head_dim(),
            &h_positions,
            &w_positions,
            VISION_ROPE_BASE,
            &image.device(),
        )?;

        for block in &self.blocks {
            hidden = block.forward(&hidden, &siglip_rope)?;
        }

        hidden = self.post_layernorm.forward(&hidden)?;
        hidden = self.spatial_merge.forward(&hidden, grid_h, grid_w)?;
        check_output_finite(&hidden, "PaddleOcrVlVisionEncoder")?;
        Ok(hidden)
    }

    /// Access the vision config.
    #[must_use]
    pub fn config(&self) -> &PaddleOcrVlVisionConfig {
        &self.config
    }

    fn interpolate_position_embedding(&self, grid_h: usize, grid_w: usize) -> Result<DynTensor> {
        let base_grid = self.config.position_grid_size();
        if grid_h == base_grid && grid_w == base_grid {
            return self.position_embedding.reshape([
                1,
                base_grid * base_grid,
                self.config.hidden_size,
            ]);
        }

        let position_embedding =
            self.position_embedding
                .reshape([1, base_grid, base_grid, self.config.hidden_size])?;
        let position_embedding = position_embedding.permute([0, 3, 1, 2])?;
        let position_embedding = position_embedding.upsample_bilinear_2d(
            grid_h as f64 / base_grid as f64,
            grid_w as f64 / base_grid as f64,
            false,
        )?;
        position_embedding.permute([0, 2, 3, 1])?.reshape([
            1,
            grid_h * grid_w,
            self.config.hidden_size,
        ])
    }
}

/// Build row-major height/width position vectors for a patch grid.
fn build_hw_positions(grid_h: usize, grid_w: usize) -> Result<(Vec<usize>, Vec<usize>)> {
    let seq_len = grid_h.checked_mul(grid_w).ok_or_else(|| {
        TensorError::InvalidShape("build_hw_positions: patch grid size overflow".into())
    })?;
    let mut h_positions = Vec::with_capacity(seq_len);
    let mut w_positions = Vec::with_capacity(seq_len);
    for h in 0..grid_h {
        for w in 0..grid_w {
            h_positions.push(h);
            w_positions.push(w);
        }
    }
    Ok((h_positions, w_positions))
}

#[cfg(test)]
#[path = "paddle_ocr_vision_tests.rs"]
mod tests;
