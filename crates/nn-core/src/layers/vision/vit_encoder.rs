// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Full Vision Transformer encoder — combines patch embedding, positional
//! embeddings, transformer blocks, and final layer normalization.

use super::{load_layer_norm, PatchEmbedding, PoolingStrategy, VitConfig, VitEncoderBlock};
use crate::dyn_tensor::DynTensor;
use crate::layers::{check_output_finite, LayerNorm, Module};
use crate::var_builder::VarBuilder;
use crate::{Result, TensorError};

/// Full Vision Transformer encoder.
///
/// Combines [`PatchEmbedding`], positional embeddings, a stack of
/// [`VitEncoderBlock`]s, and a final [`LayerNorm`].
#[derive(Clone)]
pub struct VitEncoder {
    pub(super) patch_embed: PatchEmbedding,
    /// CLS token: learnable `[1, 1, D]` prepended to patch sequence.
    pub(super) cls_token: Option<DynTensor>,
    /// Positional embedding: `[1, seq_len, D]`.
    pub(super) position_embedding: DynTensor,
    pub(super) blocks: Vec<VitEncoderBlock>,
    pub(super) ln: LayerNorm,
    pub(super) config: VitConfig,
}

impl std::fmt::Debug for VitEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VitEncoder")
            .field("num_layers", &self.blocks.len())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl VitEncoder {
    /// Load from a [`VarBuilder`] with HuggingFace ViT weight naming.
    ///
    /// Expected prefix structure:
    /// - `embeddings.patch_embeddings.projection.{weight,bias}`
    /// - `embeddings.cls_token` (if `use_cls_token`)
    /// - `embeddings.position_embeddings`
    /// - `encoder.layer.{i}.{attention,intermediate,output,layernorm_before,layernorm_after}.*`
    /// - `layernorm.{weight,bias}`
    pub fn load(vb: impl AsRef<VarBuilder>, config: &VitConfig) -> Result<Self> {
        let vb = vb.as_ref();
        config.validate()?;
        let d = config.hidden_size;
        let seq_len = config.seq_len();

        // Patch embedding
        let embed_vb = vb.pp("embeddings");
        let patch_embed = PatchEmbedding::load(embed_vb.pp("patch_embeddings"), config)?;

        // CLS token
        let cls_token = if config.use_cls_token {
            let cls = embed_vb.get(&[1, 1, d], "cls_token")?;
            Some(cls)
        } else {
            None
        };

        // Positional embedding
        let position_embedding = embed_vb.get(&[1, seq_len, d], "position_embeddings")?;

        // Encoder blocks
        let enc_vb = vb.pp("encoder");
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let block = VitEncoderBlock::load(
                enc_vb.pp(format!("layer.{i}")),
                d,
                config.num_heads,
                config.intermediate_size,
                config.layer_norm_eps,
            )?;
            blocks.push(block);
        }

        // Final LayerNorm
        let ln = load_layer_norm(vb, "layernorm", d, config.layer_norm_eps)?;

        Ok(Self {
            patch_embed,
            cls_token,
            position_embedding,
            blocks,
            ln,
            config: config.clone(),
        })
    }

    /// Forward pass through the full ViT encoder.
    ///
    /// Input: `[B, C, H, W]` image tensor.
    /// Output depends on pooling strategy:
    /// - `Cls`: `[B, D]` — CLS token output
    /// - `Mean`: `[B, D]` — mean of patch tokens
    /// - `None`: `[B, seq_len, D]` — all token outputs
    pub fn forward(&self, x: &DynTensor, pooling: PoolingStrategy) -> Result<DynTensor> {
        let (b, _c, _h, _w) = x.dims4()?;

        // Extract patch embeddings: [B, N, D]
        let mut embeddings = self.patch_embed.forward(x)?;

        // Prepend CLS token if configured
        if let Some(ref cls) = self.cls_token {
            // Expand CLS [1, 1, D] to [B, 1, D]
            let cls_expanded = cls.expand([b, 1, self.config.hidden_size])?;
            embeddings = DynTensor::cat(&[&cls_expanded, &embeddings], 1)?;
        }

        // Add positional embedding
        // Handle variable image sizes: interpolate position embeddings if needed
        let seq_len = embeddings.dim(1)?;
        let pos_emb = if seq_len == self.position_embedding.dim(1)? {
            self.position_embedding.clone()
        } else {
            // Interpolate position embeddings for different image sizes
            self.interpolate_position_embeddings(seq_len)?
        };
        let mut x = embeddings.broadcast_add(&pos_emb)?;

        // Encoder blocks — periodic flush prevents command buffer depth pathology
        // on deep models (#4319: 200+ encodings in one buffer caused Metal timeout)
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x)?;
            if (i + 1) % 4 == 0 && i + 1 < self.blocks.len() {
                crate::gpu_backend_flush()?;
            }
        }

        // Final LayerNorm
        x = self.ln.forward(&x)?;
        check_output_finite(&x, "VitEncoder")?;

        // Pooling
        match pooling {
            PoolingStrategy::Cls => {
                if self.cls_token.is_none() {
                    return Err(TensorError::ValueOutOfRange {
                        description: "VitEncoder: Cls pooling requires use_cls_token = true",
                    });
                }
                // CLS is at index 0: [B, seq_len, D] -> [B, D]
                x.narrow(1, 0, 1)?.squeeze(1)
            }
            PoolingStrategy::Mean => {
                let start = if self.cls_token.is_some() { 1 } else { 0 };
                let seq = x.dim(1)?;
                if seq <= start {
                    return Err(TensorError::ValueOutOfRange {
                        description:
                            "VitEncoder: Mean pooling requires sequence length > CLS offset",
                    });
                }
                let patch_count = seq - start;
                let patches = x.narrow(1, start, patch_count)?;
                patches.mean_keepdim(1)?.squeeze(1)
            }
            PoolingStrategy::None => Ok(x),
        }
    }

    /// Interpolate position embeddings for variable image sizes.
    ///
    /// Uses simple nearest-neighbor interpolation by repeating/selecting positions.
    /// For production use with variable sizes, bilinear interpolation would be better.
    fn interpolate_position_embeddings(&self, target_len: usize) -> Result<DynTensor> {
        let pos_len = self.position_embedding.dim(1)?;
        let d = self.config.hidden_size;

        if target_len == 0 {
            return Err(TensorError::ZeroLengthDimension {
                axis: 1,
                operation: "VitEncoder interpolate_pos_embed",
            });
        }
        // Index tensors are U32; validate position count fits.
        if pos_len > u32::MAX as usize {
            return Err(TensorError::ValueOutOfRange {
                description: "VitEncoder: pos_len exceeds u32::MAX",
            });
        }

        if self.cls_token.is_some() {
            if target_len < 2 {
                return Err(TensorError::ValueOutOfRange {
                    description: "VitEncoder: with CLS token, target sequence must be >= 2",
                });
            }
            // Separate CLS position embedding from patch position embeddings
            let cls_pos = self.position_embedding.narrow(1, 0, 1)?;
            let patch_pos = self.position_embedding.narrow(1, 1, pos_len - 1)?;

            // Simple nearest-neighbor: select positions proportionally
            let target_patches = target_len - 1;
            let source_patches = pos_len - 1;
            let mut indices = Vec::with_capacity(target_patches);
            for i in 0..target_patches {
                let src_idx = i * source_patches / target_patches;
                indices.push(src_idx);
            }

            // Gather selected positions
            let idx_tensor = DynTensor::from_vec_u32(
                indices.iter().map(|&i| i as u32).collect::<Vec<_>>(),
                &[target_patches],
                &self.position_embedding.device(),
            )?;
            let interp_pos = patch_pos.squeeze(0)?.index_select(&idx_tensor, 0)?;
            let interp_pos = interp_pos.unsqueeze(0)?;

            // Concatenate CLS + interpolated patch positions
            DynTensor::cat(&[&cls_pos, &interp_pos], 1)
        } else {
            let mut indices = Vec::with_capacity(target_len);
            for i in 0..target_len {
                let src_idx = i * pos_len / target_len;
                indices.push(src_idx);
            }
            let idx_tensor = DynTensor::from_vec_u32(
                indices.iter().map(|&i| i as u32).collect::<Vec<_>>(),
                &[target_len],
                &self.position_embedding.device(),
            )?;
            let interp_pos = self
                .position_embedding
                .squeeze(0)?
                .index_select(&idx_tensor, 0)?;
            Ok(interp_pos.unsqueeze(0)?.reshape([1, target_len, d])?)
        }
    }

    /// Access the ViT config.
    #[must_use]
    pub fn config(&self) -> &VitConfig {
        &self.config
    }

    /// Forward pass that collects intermediate layer outputs for multi-level
    /// feature fusion (DeepStack).
    ///
    /// `layer_indices` specifies which transformer block outputs to collect
    /// (0-indexed). For example, `&[3, 7, 11]` on a 12-layer ViT collects
    /// outputs after blocks 4, 8, and 12.
    ///
    /// Returns a vector of `[B, seq_len, D]` tensors, one per requested layer
    /// index. Outputs are raw block outputs (pre-final-LayerNorm).
    ///
    /// Required by Qwen3-VL DeepStack and other VLMs that fuse multi-layer
    /// ViT features for richer visual representations.
    pub fn forward_deepstack(
        &self,
        x: &DynTensor,
        layer_indices: &[usize],
    ) -> Result<Vec<DynTensor>> {
        if layer_indices.is_empty() {
            return Err(TensorError::InvalidShape(
                "VitEncoder::forward_deepstack: layer_indices must not be empty".into(),
            ));
        }
        let num_blocks = self.blocks.len();
        for &idx in layer_indices {
            if idx >= num_blocks {
                return Err(TensorError::ValueOutOfRange {
                    description: "VitEncoder::forward_deepstack: layer index out of range",
                });
            }
        }

        let (b, _c, _h, _w) = x.dims4()?;

        // Patch embedding + CLS token + position embedding (same as forward)
        let mut embeddings = self.patch_embed.forward(x)?;
        if let Some(ref cls) = self.cls_token {
            let cls_expanded = cls.expand([b, 1, self.config.hidden_size])?;
            embeddings = DynTensor::cat(&[&cls_expanded, &embeddings], 1)?;
        }
        let seq_len = embeddings.dim(1)?;
        let pos_emb = if seq_len == self.position_embedding.dim(1)? {
            self.position_embedding.clone()
        } else {
            self.interpolate_position_embeddings(seq_len)?
        };
        let mut h = embeddings.broadcast_add(&pos_emb)?;

        // Build a fast lookup set for which layers to collect + early exit bound
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

        // Reorder collected outputs to match the order of layer_indices
        let mut result = Vec::with_capacity(layer_indices.len());
        for &idx in layer_indices {
            let pos = *index_map.get(&idx).ok_or_else(|| {
                TensorError::InvalidShape(format!(
                    "VitEncoder::forward_deepstack: layer {idx} not collected"
                ))
            })?;
            result.push(collected[pos].clone());
        }

        // Defense-in-depth: validate all collected outputs are finite
        for (i, t) in result.iter().enumerate() {
            check_output_finite(
                t,
                &format!("VitEncoder::forward_deepstack[layer {}]", layer_indices[i]),
            )?;
        }

        Ok(result)
    }

    /// Access encoder blocks.
    #[must_use]
    pub fn blocks(&self) -> &[VitEncoderBlock] {
        &self.blocks
    }
}

/// Self-attention via [`Module`] trait (uses `PoolingStrategy::None`).
impl Module for VitEncoder {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        self.forward(x, PoolingStrategy::None)
    }
}
