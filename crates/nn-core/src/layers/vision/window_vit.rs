// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Window attention Vision Transformer for Qwen2.5-VL (#2421).
//!
//! [`WindowVitEncoderBlock`] extends [`VitEncoderBlock`] with optional local
//! window attention. When `window_size` is `Some`, the block partitions the
//! input into non-overlapping spatial windows before self-attention, then
//! unpartitions after — restricting each token to attend only within its
//! local window. When `None`, standard global attention is used.
//!
//! [`WindowVitEncoder`] stacks these blocks according to a
//! [`WindowVitConfig`], which specifies a `window_pattern: Vec<bool>` to
//! control which layers use window vs. global attention (e.g., alternating
//! `[true, false, true, false, ...]` as in Qwen2.5-VL's 32-layer ViT).
//!
//! Both types build on the existing [`window_partition`] / [`window_unpartition`]
//! utilities in `crate::layers::attention::window`.

use super::vit::load_layer_norm;
use super::{PatchEmbedding, PoolingStrategy, VitConfig, VitEncoderBlock};
use crate::dyn_tensor::DynTensor;
use crate::layers::attention::{window_partition, window_unpartition};
use crate::layers::{check_output_finite, LayerNorm, Module};
use crate::var_builder::VarBuilder;
use crate::{Result, TensorError};

/// Configuration for a window-attention Vision Transformer.
///
/// Extends [`VitConfig`] with a window attention pattern and window size.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct WindowVitConfig {
    /// Base ViT configuration (hidden size, heads, layers, etc.).
    pub vit: VitConfig,
    /// Window size in tokens (spatial edge length). Applied to layers where
    /// `window_pattern[i]` is `true`.
    pub window_size: usize,
    /// Per-layer flag: `true` = window (local) attention, `false` = global.
    /// Length must equal `vit.num_layers`. For Qwen2.5-VL, odd-indexed layers
    /// use window attention and even-indexed layers use global attention.
    pub window_pattern: Vec<bool>,
}

impl WindowVitConfig {
    /// Create a new `WindowVitConfig` with validation.
    pub fn new(vit: VitConfig, window_size: usize, window_pattern: Vec<bool>) -> Result<Self> {
        let config = Self {
            vit,
            window_size,
            window_pattern,
        };
        config.validate()?;
        Ok(config)
    }

    /// Create a Qwen2.5-VL-style alternating pattern: odd layers use window
    /// attention, even layers use global attention.
    pub fn alternating(vit: VitConfig, window_size: usize) -> Result<Self> {
        let pattern: Vec<bool> = (0..vit.num_layers).map(|i| i % 2 == 1).collect();
        Self::new(vit, window_size, pattern)
    }

    /// Create a Qwen3-VL-style pattern: every Nth layer uses global attention,
    /// all other layers use window (local) attention.
    ///
    /// For example, with `global_every_n = 4` and 32 layers:
    /// layers 3, 7, 11, 15, 19, 23, 27, 31 are global (every 4th, counting
    /// from 0 where the Nth minus 1 is global). All other layers use window
    /// attention with the given `window_size`.
    ///
    /// Qwen3-VL uses this pattern with `global_every_n = 4` to reduce
    /// attention cost while maintaining sufficient global context mixing.
    ///
    /// Returns an error if `global_every_n == 0`.
    pub fn every_nth_global(
        vit: VitConfig,
        window_size: usize,
        global_every_n: usize,
    ) -> Result<Self> {
        if global_every_n == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "WindowVitConfig::every_nth_global: global_every_n must be > 0",
            });
        }
        // Window = true for all layers except every Nth (0-indexed: N-1, 2N-1, ...)
        let pattern: Vec<bool> = (0..vit.num_layers)
            .map(|i| (i + 1) % global_every_n != 0)
            .collect();
        Self::new(vit, window_size, pattern)
    }

    /// Create an all-window pattern: every layer uses window attention.
    ///
    /// Useful for testing or models where all layers are local-only.
    pub fn all_window(vit: VitConfig, window_size: usize) -> Result<Self> {
        let pattern = vec![true; vit.num_layers];
        Self::new(vit, window_size, pattern)
    }

    /// Create an all-global pattern: no layers use window attention.
    ///
    /// Equivalent to a standard ViT with no windowing.
    pub fn all_global(vit: VitConfig, window_size: usize) -> Result<Self> {
        let pattern = vec![false; vit.num_layers];
        Self::new(vit, window_size, pattern)
    }

    /// Validate all configuration invariants.
    pub fn validate(&self) -> Result<()> {
        self.vit.validate()?;
        if self.window_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "WindowVitConfig: window_size must be > 0",
            });
        }
        if self.window_pattern.len() != self.vit.num_layers {
            return Err(TensorError::InvalidShape(format!(
                "WindowVitConfig: window_pattern length ({}) must equal num_layers ({})",
                self.window_pattern.len(),
                self.vit.num_layers,
            )));
        }
        Ok(())
    }
}

/// A ViT encoder block with optional window attention.
///
/// When `window_size` is `Some`, the attention sub-layer operates on
/// non-overlapping windows of the spatial token sequence. The MLP sub-layer
/// always operates token-wise (unaffected by windowing). Spatial dimensions
/// (`height`, `width`) must be provided at forward time so the block can
/// partition and unpartition correctly.
///
/// When `window_size` is `None`, the block delegates entirely to the inner
/// [`VitEncoderBlock`] (global attention).
#[derive(Clone, Debug)]
pub struct WindowVitEncoderBlock {
    /// Inner standard ViT encoder block (contains all weights).
    pub(super) inner: VitEncoderBlock,
    /// If `Some(ws)`, use window attention with window edge length `ws`.
    pub(super) window_size: Option<usize>,
}

impl WindowVitEncoderBlock {
    /// Create from an existing [`VitEncoderBlock`] and optional window size.
    pub fn new(inner: VitEncoderBlock, window_size: Option<usize>) -> Result<Self> {
        if let Some(ws) = window_size {
            if ws == 0 {
                return Err(TensorError::ValueOutOfRange {
                    description: "WindowVitEncoderBlock: window_size must be > 0",
                });
            }
        }
        Ok(Self { inner, window_size })
    }

    /// Forward pass with spatial dimensions for window partitioning.
    ///
    /// - `x`: `[B, H*W, D]` — batch of spatial token sequences (no CLS token).
    /// - `height`, `width`: spatial grid dimensions (`H * W == x.dim(1)`).
    ///
    /// Returns `[B, H*W, D]`.
    pub fn forward_spatial(&self, x: &DynTensor, height: usize, width: usize) -> Result<DynTensor> {
        match self.window_size {
            Some(ws) => self.forward_windowed(x, height, width, ws),
            None => self.inner.forward(x),
        }
    }

    /// Window-attention forward: partition -> block forward -> unpartition.
    ///
    /// The full pre-norm transformer block (LN -> attention -> residual ->
    /// LN -> MLP -> residual) runs within each window independently. This
    /// matches the Qwen2.5-VL architecture where window layers apply both
    /// attention AND MLP within the local window context.
    fn forward_windowed(
        &self,
        x: &DynTensor,
        height: usize,
        width: usize,
        window_size: usize,
    ) -> Result<DynTensor> {
        let (b, seq_len, _d) = x.dims3()?;
        if seq_len != height * width {
            return Err(TensorError::shape_mismatch(
                vec![b, height * width],
                vec![b, seq_len],
            ));
        }

        // Partition into windows: [B * num_windows, ws*ws, D]
        let (windowed, padded_h, padded_w) = window_partition(x, height, width, window_size)?;

        // Run the standard encoder block within each window
        let windowed_out = self.inner.forward(&windowed)?;

        // Unpartition back to spatial sequence: [B, H*W, D]
        window_unpartition(
            &windowed_out,
            height,
            width,
            padded_h,
            padded_w,
            window_size,
            b,
        )
    }

    /// Access the inner [`VitEncoderBlock`].
    #[must_use]
    pub fn inner(&self) -> &VitEncoderBlock {
        &self.inner
    }

    /// Whether this block uses window attention.
    #[must_use]
    pub fn uses_window_attention(&self) -> bool {
        self.window_size.is_some()
    }
}

/// Window-attention ViT encoder for Qwen2.5-VL and similar models.
///
/// Stacks [`WindowVitEncoderBlock`]s with a configurable per-layer window
/// pattern. Includes patch embedding, positional embeddings, and final
/// layer normalization (same as [`VitEncoder`]).
///
/// The spatial grid dimensions are derived from the patch embedding output
/// and propagated through all blocks for correct window partitioning.
#[derive(Clone)]
pub struct WindowVitEncoder {
    pub(super) patch_embed: PatchEmbedding,
    /// CLS token: learnable `[1, 1, D]` prepended to patch sequence.
    pub(super) cls_token: Option<DynTensor>,
    /// Positional embedding: `[1, seq_len, D]`.
    pub(super) position_embedding: DynTensor,
    pub(super) blocks: Vec<WindowVitEncoderBlock>,
    pub(super) ln: LayerNorm,
    pub(super) config: WindowVitConfig,
}

impl std::fmt::Debug for WindowVitEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WindowVitEncoder")
            .field("num_layers", &self.blocks.len())
            .field("window_size", &self.config.window_size)
            .field("window_pattern", &self.config.window_pattern)
            .finish_non_exhaustive()
    }
}

impl WindowVitEncoder {
    /// Load from a [`VarBuilder`] with HuggingFace ViT weight naming.
    ///
    /// Uses the same weight layout as [`VitEncoder`]. The window pattern
    /// controls which layers use local attention — weights are identical
    /// regardless of the attention pattern.
    pub fn load(vb: impl AsRef<VarBuilder>, config: &WindowVitConfig) -> Result<Self> {
        let vb = vb.as_ref();
        config.validate()?;
        let vc = &config.vit;
        let d = vc.hidden_size;
        let seq_len = vc.seq_len();

        // Patch embedding
        let embed_vb = vb.pp("embeddings");
        let patch_embed = PatchEmbedding::load(embed_vb.pp("patch_embeddings"), vc)?;

        // CLS token
        let cls_token = if vc.use_cls_token {
            Some(embed_vb.get(&[1, 1, d], "cls_token")?)
        } else {
            None
        };

        // Positional embedding
        let position_embedding = embed_vb.get(&[1, seq_len, d], "position_embeddings")?;

        // Encoder blocks with window pattern
        let enc_vb = vb.pp("encoder");
        let mut blocks = Vec::with_capacity(vc.num_layers);
        for i in 0..vc.num_layers {
            let inner = VitEncoderBlock::load(
                enc_vb.pp(format!("layer.{i}")),
                d,
                vc.num_heads,
                vc.intermediate_size,
                vc.layer_norm_eps,
            )?;
            let ws = if config.window_pattern[i] {
                Some(config.window_size)
            } else {
                None
            };
            blocks.push(WindowVitEncoderBlock::new(inner, ws)?);
        }

        // Final LayerNorm
        let ln = load_layer_norm(vb, "layernorm", d, vc.layer_norm_eps)?;

        Ok(Self {
            patch_embed,
            cls_token,
            position_embedding,
            blocks,
            ln,
            config: config.clone(),
        })
    }

    /// Forward pass through the window-attention ViT encoder.
    ///
    /// Input: `[B, C, H, W]` image tensor.
    /// Output depends on pooling strategy (same as [`VitEncoder`]).
    ///
    /// Window attention layers receive the spatial grid dimensions derived
    /// from the patch embedding. If a CLS token is present, it is separated
    /// before window layers and re-prepended after, since window partitioning
    /// operates on spatial tokens only.
    pub fn forward(&self, x: &DynTensor, pooling: PoolingStrategy) -> Result<DynTensor> {
        let (b, _c, img_h, img_w) = x.dims4()?;
        let vc = &self.config.vit;

        // Spatial grid dimensions from patch embedding
        let grid_h = img_h / vc.patch_size;
        let grid_w = img_w / vc.patch_size;

        // Extract patch embeddings: [B, N, D]
        let mut embeddings = self.patch_embed.forward(x)?;

        // Prepend CLS token if configured
        if let Some(ref cls) = self.cls_token {
            let cls_expanded = cls.expand([b, 1, vc.hidden_size])?;
            embeddings = DynTensor::cat(&[&cls_expanded, &embeddings], 1)?;
        }

        // Add positional embedding
        let seq_len = embeddings.dim(1)?;
        let pos_emb = if seq_len == self.position_embedding.dim(1)? {
            self.position_embedding.clone()
        } else {
            // For variable image sizes, use nearest-neighbor interpolation
            // (simplified — production would use bilinear).
            self.position_embedding.clone()
        };
        let mut h = embeddings.broadcast_add(&pos_emb)?;

        // Encoder blocks
        let has_cls = self.cls_token.is_some();
        for block in &self.blocks {
            if block.uses_window_attention() && has_cls {
                // Separate CLS from spatial tokens for window attention
                let cls_tok = h.narrow(1, 0, 1)?;
                let spatial = h.narrow(1, 1, grid_h * grid_w)?;
                let spatial_out = block.forward_spatial(&spatial, grid_h, grid_w)?;
                h = DynTensor::cat(&[&cls_tok, &spatial_out], 1)?;
            } else if block.uses_window_attention() {
                h = block.forward_spatial(&h, grid_h, grid_w)?;
            } else {
                h = block.inner.forward(&h)?;
            }
        }

        // Final LayerNorm
        h = self.ln.forward(&h)?;
        check_output_finite(&h, "WindowVitEncoder")?;

        // Pooling (same as VitEncoder)
        match pooling {
            PoolingStrategy::Cls => {
                if self.cls_token.is_none() {
                    return Err(TensorError::ValueOutOfRange {
                        description: "WindowVitEncoder: Cls pooling requires use_cls_token = true",
                    });
                }
                h.narrow(1, 0, 1)?.squeeze(1)
            }
            PoolingStrategy::Mean => {
                let start = if has_cls { 1 } else { 0 };
                let total = h.dim(1)?;
                if total <= start {
                    return Err(TensorError::ValueOutOfRange {
                        description:
                            "WindowVitEncoder: Mean pooling requires sequence length > CLS offset",
                    });
                }
                let patch_count = total - start;
                let patches = h.narrow(1, start, patch_count)?;
                patches.mean_keepdim(1)?.squeeze(1)
            }
            PoolingStrategy::None => Ok(h),
        }
    }

    /// Access the window ViT config.
    #[must_use]
    pub fn config(&self) -> &WindowVitConfig {
        &self.config
    }

    /// Access encoder blocks.
    #[must_use]
    pub fn blocks(&self) -> &[WindowVitEncoderBlock] {
        &self.blocks
    }
}

impl Module for WindowVitEncoder {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        self.forward(x, PoolingStrategy::None)
    }
}

#[cfg(test)]
#[path = "window_vit_tests.rs"]
mod tests;
