// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SigLIP2 Vision Encoder for vision-language models.
//!
//! [`SigLip2VisionEncoder`] wraps the standard [`VitEncoder`] building blocks
//! (patch embedding, transformer encoder blocks) with SigLIP2-specific
//! configuration and HuggingFace weight naming conventions.
//!
//! Required by Granite-Docling-258M (dpdf Tier 1) and other VLMs using
//! the `google/siglip2-base-patch16-*` family.
//!
//! HuggingFace weight naming differs from standard ViT:
//! - `vision_model.embeddings.patch_embedding.{weight,bias}` (not `patch_embeddings.projection`)
//! - `vision_model.encoder.layers.{i}.self_attn.{q,k,v,out}_proj.*` (not `attention.attention.query.*`)
//! - `vision_model.encoder.layers.{i}.layer_norm1.*` (not `layernorm_before`)
//! - `vision_model.encoder.layers.{i}.mlp.fc1.*` / `mlp.fc2.*` (not `intermediate.dense`)
//! - `vision_model.post_layernorm.*` (not `layernorm`)
//! - No CLS token (uses all patch outputs or mean pooling)

use super::{PatchEmbedding, PoolingStrategy, VitConfig, VitEncoderBlock};
use crate::dyn_tensor::DynTensor;
use crate::layers::{
    check_output_finite, with_nan_check_policy, LayerNorm, Linear, Module, NanCheckPolicy,
};
use crate::var_builder::VarBuilder;
use crate::{Result, TensorError};

/// Configuration for SigLIP2 vision encoder.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SigLip2Config {
    /// Number of input channels (3 for RGB).
    pub num_channels: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// MLP intermediate dimension.
    pub intermediate_size: usize,
    /// Patch size in pixels.
    pub patch_size: usize,
    /// Input image size in pixels (square).
    pub image_size: usize,
    /// Layer normalization epsilon.
    pub layer_norm_eps: f64,
}

impl SigLip2Config {
    /// SigLIP2-base configuration (768 hidden, 12 layers, 12 heads, patch 16).
    ///
    /// Matches `google/siglip2-base-patch16-224` and
    /// `ibm-granite/granite-vision-docling-258m-preview`.
    pub fn base_patch16(image_size: usize) -> Result<Self> {
        Self::new(3, 768, 12, 12, 3072, 16, image_size, 1e-6)
    }

    /// Create a new SigLIP2 config with validation.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_channels: usize,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        intermediate_size: usize,
        patch_size: usize,
        image_size: usize,
        layer_norm_eps: f64,
    ) -> Result<Self> {
        let config = Self {
            num_channels,
            hidden_size,
            num_layers,
            num_heads,
            intermediate_size,
            patch_size,
            image_size,
            layer_norm_eps,
        };
        config.validate()?;
        Ok(config)
    }

    /// Convert to a [`VitConfig`] for reuse of shared ViT building blocks.
    pub fn to_vit_config(&self) -> Result<VitConfig> {
        VitConfig::new(
            self.num_channels,
            self.hidden_size,
            self.num_layers,
            self.num_heads,
            self.intermediate_size,
            self.patch_size,
            self.image_size,
            self.layer_norm_eps,
            false, // SigLIP2 has no CLS token
        )
    }

    fn validate(&self) -> Result<()> {
        self.to_vit_config()?;
        Ok(())
    }
}

/// SigLIP2 Vision Encoder.
///
/// Loads weights using SigLIP2/HuggingFace naming conventions and wraps
/// standard ViT building blocks (PatchEmbedding, VitEncoderBlock).
#[derive(Clone)]
pub struct SigLip2VisionEncoder {
    patch_embed: PatchEmbedding,
    position_embedding: DynTensor,
    blocks: Vec<VitEncoderBlock>,
    post_layernorm: LayerNorm,
    config: SigLip2Config,
}

impl std::fmt::Debug for SigLip2VisionEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SigLip2VisionEncoder")
            .field("num_layers", &self.blocks.len())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl SigLip2VisionEncoder {
    /// Load from a [`VarBuilder`] with SigLIP2 / HuggingFace weight naming.
    ///
    /// Expected prefix: `vision_model.*` (caller should scope via `vb.pp("vision_model")`
    /// or the VarBuilder can point directly at the vision model scope).
    ///
    /// Weight structure:
    /// - `embeddings.patch_embedding.{weight,bias}`
    /// - `embeddings.position_embedding.weight` `[num_patches, D]` (unsqueezed to `[1, N, D]`)
    /// - `encoder.layers.{i}.self_attn.{q_proj,k_proj,v_proj,out_proj}.{weight,bias}`
    /// - `encoder.layers.{i}.layer_norm1.{weight,bias}`
    /// - `encoder.layers.{i}.layer_norm2.{weight,bias}`
    /// - `encoder.layers.{i}.mlp.fc1.{weight,bias}`
    /// - `encoder.layers.{i}.mlp.fc2.{weight,bias}`
    /// - `post_layernorm.{weight,bias}`
    pub fn load(vb: impl AsRef<VarBuilder>, config: &SigLip2Config) -> Result<Self> {
        let vb = vb.as_ref();
        config.validate()?;
        let vit = config.to_vit_config()?;
        let d = config.hidden_size;
        let num_patches = vit.num_patches();

        // Patch embedding — SigLIP2 uses "patch_embedding" (singular)
        let embed_vb = vb.pp("embeddings");
        let patch_embed = load_siglip2_patch_embedding(&embed_vb, &vit)?;

        // Position embedding — HuggingFace stores as [N, D], unsqueeze to [1, N, D]
        let position_embedding = embed_vb
            .get(&[num_patches, d], "position_embedding.weight")?
            .unsqueeze(0)?;

        // Encoder blocks with SigLIP2 naming
        let enc_vb = vb.pp("encoder");
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer_vb = enc_vb.pp(format!("layers.{i}"));
            let block = load_siglip2_encoder_block(&layer_vb, config)?;
            blocks.push(block);
        }

        // Post-LayerNorm
        let ln_w = vb.get(&[d], "post_layernorm.weight")?;
        let ln_b = vb.get(&[d], "post_layernorm.bias")?;
        let post_layernorm = LayerNorm::new(ln_w, ln_b, config.layer_norm_eps)?;

        Ok(Self {
            patch_embed,
            position_embedding,
            blocks,
            post_layernorm,
            config: config.clone(),
        })
    }

    /// Forward pass through the SigLIP2 encoder.
    ///
    /// Input: `[B, C, H, W]` image tensor.
    /// Output depends on pooling:
    /// - `Mean`: `[B, D]`
    /// - `None`: `[B, num_patches, D]`
    /// - `Cls`: returns error (SigLIP2 has no CLS token)
    pub fn forward(&self, x: &DynTensor, pooling: PoolingStrategy) -> Result<DynTensor> {
        if pooling == PoolingStrategy::Cls {
            return Err(TensorError::ValueOutOfRange {
                description: "SigLip2VisionEncoder: Cls pooling not supported (no CLS token)",
            });
        }

        // Patch embedding: [B, C, H, W] -> [B, N, D]
        let mut x = self.patch_embed.forward(x)?;

        // Add position embedding
        let seq_len = x.dim(1)?;
        let pos_len = self.position_embedding.dim(1)?;
        if seq_len != pos_len {
            return Err(TensorError::shape_mismatch(
                vec![1, pos_len, self.config.hidden_size],
                vec![1, seq_len, self.config.hidden_size],
            ));
        }
        x = x.broadcast_add(&self.position_embedding)?;

        // Encoder blocks — skip per-block NaN checks to avoid 13x flush+readback
        // cycles that cause Metal GPU timeout. Periodic flush every 4 blocks
        // prevents command buffer depth pathology (#4319: 207 encodings in one
        // buffer caused Metal scheduler timeout). Final output check provides
        // defense-in-depth.
        x = with_nan_check_policy(NanCheckPolicy::Skip, || -> Result<DynTensor> {
            let mut h = x;
            for (i, block) in self.blocks.iter().enumerate() {
                h = block.forward(&h)?;
                // Flush every 4 blocks (~68 encodings) to limit command buffer
                // depth. Source: #4319
                if (i + 1) % 4 == 0 && i + 1 < self.blocks.len() {
                    crate::gpu_backend_flush()?;
                }
            }
            Ok(h)
        })?;

        // Post-LayerNorm
        x = self.post_layernorm.forward(&x)?;
        // Defense-in-depth: validate final output OUTSIDE the skip scope
        check_output_finite(&x, "SigLip2VisionEncoder")?;

        match pooling {
            PoolingStrategy::Mean => x.mean_keepdim(1)?.squeeze(1),
            PoolingStrategy::None => Ok(x),
            PoolingStrategy::Cls => unreachable!(),
        }
    }

    /// Forward pass that collects intermediate layer outputs for multi-level
    /// feature fusion (DeepStack).
    ///
    /// See [`VitEncoder::forward_deepstack`] for details. Returns raw block
    /// outputs (pre-post-LayerNorm) at the requested layer indices.
    pub fn forward_deepstack(
        &self,
        x: &DynTensor,
        layer_indices: &[usize],
    ) -> Result<Vec<DynTensor>> {
        if layer_indices.is_empty() {
            return Err(TensorError::InvalidShape(
                "SigLip2VisionEncoder::forward_deepstack: layer_indices must not be empty".into(),
            ));
        }
        let num_blocks = self.blocks.len();
        for &idx in layer_indices {
            if idx >= num_blocks {
                return Err(TensorError::ValueOutOfRange {
                    description:
                        "SigLip2VisionEncoder::forward_deepstack: layer index out of range",
                });
            }
        }

        // Patch embedding + position embedding
        let mut h = self.patch_embed.forward(x)?;
        let seq_len = h.dim(1)?;
        let pos_len = self.position_embedding.dim(1)?;
        if seq_len != pos_len {
            return Err(TensorError::shape_mismatch(
                vec![1, pos_len, self.config.hidden_size],
                vec![1, seq_len, self.config.hidden_size],
            ));
        }
        h = h.broadcast_add(&self.position_embedding)?;

        // Collect intermediate outputs + early exit after last needed layer
        let collect: std::collections::HashSet<usize> = layer_indices.iter().copied().collect();
        let last_needed = layer_indices.iter().copied().max().unwrap_or(0);
        let mut collected = Vec::with_capacity(collect.len());
        let mut index_map: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();

        for (i, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h)?;
            if collect.contains(&i) {
                let pos = collected.len();
                collected.push(h.clone());
                index_map.insert(i, pos);
            }
            // Periodic flush to prevent command buffer depth pathology (#4319)
            if (i + 1) % 4 == 0 {
                crate::gpu_backend_flush()?;
            }
            if i >= last_needed {
                break;
            }
        }

        // Reorder to match layer_indices order
        let mut result = Vec::with_capacity(layer_indices.len());
        for &idx in layer_indices {
            let pos = *index_map.get(&idx).ok_or_else(|| {
                TensorError::InvalidShape(format!(
                    "SigLip2VisionEncoder::forward_deepstack: layer {idx} not collected"
                ))
            })?;
            result.push(collected[pos].clone());
        }

        // Defense-in-depth: validate all collected outputs are finite
        for (i, t) in result.iter().enumerate() {
            check_output_finite(
                t,
                &format!(
                    "SigLip2VisionEncoder::forward_deepstack[layer {}]",
                    layer_indices[i]
                ),
            )?;
        }

        Ok(result)
    }

    /// Access the config.
    #[must_use]
    pub fn config(&self) -> &SigLip2Config {
        &self.config
    }
}

impl Module for SigLip2VisionEncoder {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        self.forward(x, PoolingStrategy::None)
    }
}

/// Load patch embedding with SigLIP2 weight naming (`patch_embedding`, not
/// `patch_embeddings.projection`).
fn load_siglip2_patch_embedding(
    embed_vb: impl AsRef<VarBuilder>,
    vit: &VitConfig,
) -> Result<PatchEmbedding> {
    let embed_vb = embed_vb.as_ref();
    use crate::layers::{Conv2d, Conv2dConfig};
    let proj_vb = embed_vb.pp("patch_embedding");
    let w = proj_vb.get(
        &[
            vit.hidden_size,
            vit.num_channels,
            vit.patch_size,
            vit.patch_size,
        ],
        "weight",
    )?;
    let b = if proj_vb.contains_tensor("bias") {
        Some(proj_vb.get(&[vit.hidden_size], "bias")?)
    } else {
        None
    };
    let conv = Conv2d::new(
        w,
        b,
        Conv2dConfig {
            stride: vit.patch_size,
            padding: 0,
            dilation: 1,
            groups: 1,
        },
    )?;
    PatchEmbedding::new(conv, vit.hidden_size)
}

/// Load a single encoder block with SigLIP2 weight naming.
///
/// SigLIP2 naming:
/// - `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.out_proj`
/// - `layer_norm1`, `layer_norm2`
/// - `mlp.fc1`, `mlp.fc2`
fn load_siglip2_encoder_block(
    vb: impl AsRef<VarBuilder>,
    config: &SigLip2Config,
) -> Result<VitEncoderBlock> {
    let vb = vb.as_ref();
    let d = config.hidden_size;
    let eps = config.layer_norm_eps;
    let num_heads = config.num_heads;
    let head_dim = d / num_heads;

    // LayerNorms
    let ln1_w = vb.get(&[d], "layer_norm1.weight")?;
    let ln1_b = vb.get(&[d], "layer_norm1.bias")?;
    let ln1 = LayerNorm::new(ln1_w, ln1_b, eps)?;

    let ln2_w = vb.get(&[d], "layer_norm2.weight")?;
    let ln2_b = vb.get(&[d], "layer_norm2.bias")?;
    let ln2 = LayerNorm::new(ln2_w, ln2_b, eps)?;

    // Attention: separate Q/K/V projections, fused into QKV for VitEncoderBlock
    let attn_vb = vb.pp("self_attn");
    let q_w = attn_vb.get(&[d, d], "q_proj.weight")?;
    let q_b = attn_vb.get(&[d], "q_proj.bias")?;
    let k_w = attn_vb.get(&[d, d], "k_proj.weight")?;
    let k_b = attn_vb.get(&[d], "k_proj.bias")?;
    let v_w = attn_vb.get(&[d, d], "v_proj.weight")?;
    let v_b = attn_vb.get(&[d], "v_proj.bias")?;

    let qkv_w = DynTensor::cat(&[&q_w, &k_w, &v_w], 0)?;
    let qkv_b = DynTensor::cat(&[&q_b, &k_b, &v_b], 0)?;
    let attn_qkv = Linear::new(qkv_w, Some(qkv_b))?;

    let out_w = attn_vb.get(&[d, d], "out_proj.weight")?;
    let out_b = attn_vb.get(&[d], "out_proj.bias")?;
    let attn_proj = Linear::new(out_w, Some(out_b))?;

    // MLP
    let mlp_vb = vb.pp("mlp");
    let fc1_w = mlp_vb.get(&[config.intermediate_size, d], "fc1.weight")?;
    let fc1_b = mlp_vb.get(&[config.intermediate_size], "fc1.bias")?;
    let mlp_fc1 = Linear::new(fc1_w, Some(fc1_b))?;

    let fc2_w = mlp_vb.get(&[d, config.intermediate_size], "fc2.weight")?;
    let fc2_b = mlp_vb.get(&[d], "fc2.bias")?;
    let mlp_fc2 = Linear::new(fc2_w, Some(fc2_b))?;

    VitEncoderBlock::new(
        ln1, attn_qkv, attn_proj, ln2, mlp_fc1, mlp_fc2, num_heads, head_dim,
    )
}

#[cfg(test)]
#[path = "siglip2_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "siglip2_integration_tests.rs"]
mod integration_tests;
