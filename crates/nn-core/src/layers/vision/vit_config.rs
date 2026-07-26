// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Configuration for Vision Transformer (ViT) models.
//!
//! Extracted from `vit.rs` to keep files under 400 lines.

use crate::layers::{validate_divisible, validate_eps, validate_heads};
use crate::{Result, TensorError};

/// Configuration for a Vision Transformer.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct VitConfig {
    /// Number of input image channels (typically 3 for RGB).
    pub num_channels: usize,
    /// Hidden dimension (embedding size per patch).
    pub hidden_size: usize,
    /// Number of transformer encoder layers.
    pub num_layers: usize,
    /// Number of attention heads per encoder block.
    pub num_heads: usize,
    /// Intermediate (MLP) dimension. Typically `4 * hidden_size`.
    pub intermediate_size: usize,
    /// Patch size in pixels (square patches, e.g., 16 for ViT-B/16).
    pub patch_size: usize,
    /// Expected input image size in pixels (square, e.g., 224).
    /// Used to determine the number of position embeddings.
    pub image_size: usize,
    /// Layer normalization epsilon.
    pub layer_norm_eps: f64,
    /// Whether to use a [CLS] token prepended to the patch sequence.
    pub use_cls_token: bool,
}

impl VitConfig {
    /// Create a new `VitConfig`, validating all invariants.
    ///
    /// Required because `#[non_exhaustive]` prevents struct literal construction
    /// outside this crate. Calls [`VitConfig::validate()`] internally.
    ///
    /// Returns an error if any parameter is invalid (e.g., `patch_size == 0`,
    /// `hidden_size` not divisible by `num_heads`).
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
        use_cls_token: bool,
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
            use_cls_token,
        };
        config.validate()?;
        Ok(config)
    }

    /// Number of patches for the configured image size:
    /// `(image_size / patch_size)^2`.
    #[must_use]
    pub fn num_patches(&self) -> usize {
        let grid = self.image_size / self.patch_size;
        grid * grid
    }

    /// Total sequence length including optional CLS token.
    #[must_use]
    pub fn seq_len(&self) -> usize {
        let n = self.num_patches();
        if self.use_cls_token {
            n + 1
        } else {
            n
        }
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if self.patch_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "VitConfig: patch_size must be > 0",
            });
        }
        if self.image_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "VitConfig: image_size must be > 0",
            });
        }
        if !self.image_size.is_multiple_of(self.patch_size) {
            return Err(TensorError::ValueOutOfRange {
                description: "VitConfig: image_size must be divisible by patch_size",
            });
        }
        if self.hidden_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "VitConfig: hidden_size must be > 0",
            });
        }
        if self.num_channels == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "VitConfig: num_channels must be > 0",
            });
        }
        if self.intermediate_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "VitConfig: intermediate_size must be > 0",
            });
        }
        validate_heads(self.num_heads, "VitConfig")?;
        validate_divisible(
            self.hidden_size,
            self.num_heads,
            "hidden_size",
            "num_heads",
            "VitConfig",
        )?;
        validate_eps(self.layer_norm_eps, "VitConfig")?;
        Ok(())
    }
}
