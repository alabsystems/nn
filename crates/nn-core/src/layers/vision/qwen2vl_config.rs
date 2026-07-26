// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Configuration for Qwen-VL Vision Transformers with window attention.
//!
//! [`Qwen2VLVitConfig`] — Qwen2.5-VL uses alternating global/window layers:
//! even-indexed layers (0, 2, 4, ...) use full global attention; odd-indexed
//! layers (1, 3, 5, ...) use local window attention.
//!
//! [`Qwen3VLVitConfig`] — Qwen3-VL uses a different pattern: most layers use
//! window (local) attention, with every Nth layer using global attention to
//! provide long-range context. Additionally, Qwen3-VL uses interleaved M-RoPE
//! ([`crate::layers::attention::InterleavedMRoPE`]) and DeepStack multi-level
//! feature fusion ([`crate::layers::vision::DeepStackFusion`]).

use crate::layers::{validate_divisible, validate_eps, validate_heads};
use crate::{Result, TensorError};

/// Configuration for a Qwen2.5-VL Vision Transformer.
///
/// Extends the standard ViT config with window attention parameters:
/// - `window_size`: spatial window edge length for local attention layers
/// - `window_layers`: which layer indices use window attention (default: odd layers)
///
/// # Example (Qwen2.5-VL-7B ViT)
///
/// ```text
/// hidden_size: 1280, num_layers: 32, num_heads: 16
/// window_size: 14, window_layers: [1, 3, 5, ..., 31]
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Qwen2VLVitConfig {
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
    /// Patch size in pixels (square patches).
    pub patch_size: usize,
    /// Temporal patch size for video frames (Qwen2.5-VL specific).
    pub temporal_patch_size: usize,
    /// Layer normalization epsilon.
    pub layer_norm_eps: f64,
    /// Spatial window edge length for window attention layers.
    pub window_size: usize,
    /// Layer indices that use window attention. If empty, defaults to odd layers.
    pub window_layers: Vec<usize>,
}

impl Qwen2VLVitConfig {
    /// Create and validate a Qwen2.5-VL ViT config.
    ///
    /// If `window_layers` is empty, defaults to odd-indexed layers
    /// `[1, 3, 5, ..., num_layers-1]`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_channels: usize,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        intermediate_size: usize,
        patch_size: usize,
        temporal_patch_size: usize,
        layer_norm_eps: f64,
        window_size: usize,
        window_layers: Vec<usize>,
    ) -> Result<Self> {
        let config = Self {
            num_channels,
            hidden_size,
            num_layers,
            num_heads,
            intermediate_size,
            patch_size,
            temporal_patch_size,
            layer_norm_eps,
            window_size,
            window_layers,
        };
        config.validate()?;
        Ok(config)
    }

    /// Create a config with Qwen2.5-VL-7B defaults.
    ///
    /// - 1280 hidden, 32 layers, 16 heads, 5120 intermediate
    /// - patch_size=14, temporal_patch_size=2
    /// - window_size=14, odd layers use window attention
    pub fn qwen25_vl_7b() -> Result<Self> {
        Self::new(
            3,          // num_channels
            1280,       // hidden_size
            32,         // num_layers
            16,         // num_heads
            5120,       // intermediate_size
            14,         // patch_size
            2,          // temporal_patch_size
            1e-6,       // layer_norm_eps
            14,         // window_size
            Vec::new(), // default: odd layers
        )
    }

    /// Check whether a given layer index uses window attention.
    #[must_use]
    pub fn is_window_layer(&self, layer_idx: usize) -> bool {
        if self.window_layers.is_empty() {
            // Default: odd-indexed layers use window attention
            layer_idx % 2 == 1
        } else {
            self.window_layers.contains(&layer_idx)
        }
    }

    /// Validate all configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if self.patch_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen2VLVitConfig: patch_size must be > 0",
            });
        }
        if self.temporal_patch_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen2VLVitConfig: temporal_patch_size must be > 0",
            });
        }
        if self.hidden_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen2VLVitConfig: hidden_size must be > 0",
            });
        }
        if self.num_channels == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen2VLVitConfig: num_channels must be > 0",
            });
        }
        if self.intermediate_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen2VLVitConfig: intermediate_size must be > 0",
            });
        }
        if self.num_layers == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen2VLVitConfig: num_layers must be > 0",
            });
        }
        if self.window_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen2VLVitConfig: window_size must be > 0",
            });
        }
        validate_heads(self.num_heads, "Qwen2VLVitConfig")?;
        validate_divisible(
            self.hidden_size,
            self.num_heads,
            "hidden_size",
            "num_heads",
            "Qwen2VLVitConfig",
        )?;
        validate_eps(self.layer_norm_eps, "Qwen2VLVitConfig")?;

        // Validate window layer indices are in range.
        for &idx in &self.window_layers {
            if idx >= self.num_layers {
                return Err(TensorError::ValueOutOfRange {
                    description: "Qwen2VLVitConfig: window_layers contains index >= num_layers",
                });
            }
        }

        Ok(())
    }

    /// Head dimension: `hidden_size / num_heads`.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}

// -- Qwen3-VL Vision Transformer Configuration --------------------------------

/// Configuration for a Qwen3-VL Vision Transformer.
///
/// Qwen3-VL differs from Qwen2.5-VL in three key ways:
///
/// 1. **Window pattern**: Most layers use window attention; every Nth layer
///    (typically every 4th) uses global attention — instead of alternating.
/// 2. **Interleaved M-RoPE**: Position embeddings use interleaved multimodal
///    RoPE where pair index `i` maps to section `i % 3` (temporal/height/width).
/// 3. **DeepStack fusion**: Intermediate hidden states from multiple ViT layers
///    are concatenated and projected to produce richer visual representations.
///
/// Reference: Qwen3-VL (arXiv:2511.21631).
///
/// # Variants
///
/// | Model | hidden | layers | heads | intermediate | window | global_every |
/// |-------|--------|--------|-------|-------------|--------|-------------|
/// | 2B   | 1280   | 32     | 16    | 5120        | 14     | 4           |
/// | 7B   | 3584   | 32     | 28    | 18944       | 14     | 4           |
/// | 72B  | 3584   | 80     | 28    | 18944       | 14     | 4           |
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Qwen3VLVitConfig {
    /// Number of input image channels (typically 3 for RGB).
    pub num_channels: usize,
    /// Hidden dimension (embedding size per patch).
    pub hidden_size: usize,
    /// Number of transformer encoder layers.
    pub num_layers: usize,
    /// Number of attention heads per encoder block.
    pub num_heads: usize,
    /// Intermediate (MLP) dimension.
    pub intermediate_size: usize,
    /// Patch size in pixels (square patches).
    pub patch_size: usize,
    /// Temporal patch size for video frames.
    pub temporal_patch_size: usize,
    /// Layer normalization epsilon.
    pub layer_norm_eps: f64,
    /// Spatial window edge length for window attention layers.
    pub window_size: usize,
    /// Every Nth layer uses global attention (0 = all window).
    pub global_every_n: usize,
    /// Layer indices for DeepStack feature fusion (e.g., `[7, 15, 23, 31]`
    /// for 4-level fusion across a 32-layer ViT).
    pub deepstack_layers: Vec<usize>,
    /// Output hidden size after DeepStack projection.
    /// Typically matches the LLM's hidden_size (e.g., 1536 for Qwen3-2B).
    pub deepstack_output_size: usize,
}

impl Qwen3VLVitConfig {
    /// Create and validate a Qwen3-VL ViT config.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_channels: usize,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        intermediate_size: usize,
        patch_size: usize,
        temporal_patch_size: usize,
        layer_norm_eps: f64,
        window_size: usize,
        global_every_n: usize,
        deepstack_layers: Vec<usize>,
        deepstack_output_size: usize,
    ) -> Result<Self> {
        let config = Self {
            num_channels,
            hidden_size,
            num_layers,
            num_heads,
            intermediate_size,
            patch_size,
            temporal_patch_size,
            layer_norm_eps,
            window_size,
            global_every_n,
            deepstack_layers,
            deepstack_output_size,
        };
        config.validate()?;
        Ok(config)
    }

    /// Qwen3-VL-2B vision encoder defaults.
    ///
    /// - 1280 hidden, 32 layers, 16 heads, 5120 intermediate
    /// - patch_size=14, temporal_patch_size=2
    /// - window_size=14, global every 4th layer
    /// - DeepStack: layers [7, 15, 23, 31] -> 1536 (Qwen3-2B LLM hidden)
    pub fn qwen3_vl_2b() -> Result<Self> {
        Self::new(
            3,                   // num_channels
            1280,                // hidden_size
            32,                  // num_layers
            16,                  // num_heads
            5120,                // intermediate_size
            14,                  // patch_size
            2,                   // temporal_patch_size
            1e-6,                // layer_norm_eps
            14,                  // window_size
            4,                   // global_every_n
            vec![7, 15, 23, 31], // deepstack_layers
            1536,                // deepstack_output_size (Qwen3-2B LLM hidden_size)
        )
    }

    /// Qwen3-VL-7B vision encoder defaults.
    ///
    /// - 3584 hidden, 32 layers, 28 heads, 18944 intermediate
    /// - patch_size=14, temporal_patch_size=2
    /// - window_size=14, global every 4th layer
    /// - DeepStack: layers [7, 15, 23, 31] -> 3584 (Qwen3-7B LLM hidden)
    pub fn qwen3_vl_7b() -> Result<Self> {
        Self::new(
            3,                   // num_channels
            3584,                // hidden_size
            32,                  // num_layers
            28,                  // num_heads
            18944,               // intermediate_size
            14,                  // patch_size
            2,                   // temporal_patch_size
            1e-6,                // layer_norm_eps
            14,                  // window_size
            4,                   // global_every_n
            vec![7, 15, 23, 31], // deepstack_layers
            3584,                // deepstack_output_size (Qwen3-7B LLM hidden_size)
        )
    }

    /// Qwen3-VL-72B vision encoder defaults.
    ///
    /// - 3584 hidden, 80 layers, 28 heads, 18944 intermediate
    /// - patch_size=14, temporal_patch_size=2
    /// - window_size=14, global every 4th layer
    /// - DeepStack: layers [19, 39, 59, 79] -> 8192 (Qwen3-72B LLM hidden)
    pub fn qwen3_vl_72b() -> Result<Self> {
        Self::new(
            3,                    // num_channels
            3584,                 // hidden_size
            80,                   // num_layers
            28,                   // num_heads
            18944,                // intermediate_size
            14,                   // patch_size
            2,                    // temporal_patch_size
            1e-6,                 // layer_norm_eps
            14,                   // window_size
            4,                    // global_every_n
            vec![19, 39, 59, 79], // deepstack_layers
            8192,                 // deepstack_output_size (Qwen3-72B LLM hidden_size)
        )
    }

    /// Check whether a given layer index uses global attention.
    ///
    /// If `global_every_n == 0`, all layers are window-only (no global).
    /// Otherwise, every Nth layer (0-indexed: N-1, 2N-1, ...) is global.
    #[must_use]
    pub fn is_global_layer(&self, layer_idx: usize) -> bool {
        if self.global_every_n == 0 {
            false
        } else {
            (layer_idx + 1).is_multiple_of(self.global_every_n)
        }
    }

    /// Check whether a given layer index uses window attention.
    #[must_use]
    pub fn is_window_layer(&self, layer_idx: usize) -> bool {
        !self.is_global_layer(layer_idx)
    }

    /// Generate the window pattern as a `Vec<bool>` (true = window, false = global).
    ///
    /// Compatible with [`WindowVitConfig`](super::WindowVitConfig) construction.
    #[must_use]
    pub fn window_pattern(&self) -> Vec<bool> {
        (0..self.num_layers)
            .map(|i| self.is_window_layer(i))
            .collect()
    }

    /// Validate all configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if self.patch_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen3VLVitConfig: patch_size must be > 0",
            });
        }
        if self.temporal_patch_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen3VLVitConfig: temporal_patch_size must be > 0",
            });
        }
        if self.hidden_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen3VLVitConfig: hidden_size must be > 0",
            });
        }
        if self.num_channels == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen3VLVitConfig: num_channels must be > 0",
            });
        }
        if self.intermediate_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen3VLVitConfig: intermediate_size must be > 0",
            });
        }
        if self.num_layers == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen3VLVitConfig: num_layers must be > 0",
            });
        }
        if self.window_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen3VLVitConfig: window_size must be > 0",
            });
        }
        validate_heads(self.num_heads, "Qwen3VLVitConfig")?;
        validate_divisible(
            self.hidden_size,
            self.num_heads,
            "hidden_size",
            "num_heads",
            "Qwen3VLVitConfig",
        )?;
        validate_eps(self.layer_norm_eps, "Qwen3VLVitConfig")?;

        // Validate DeepStack layer indices are in range.
        for &idx in &self.deepstack_layers {
            if idx >= self.num_layers {
                return Err(TensorError::ValueOutOfRange {
                    description: "Qwen3VLVitConfig: deepstack_layers contains index >= num_layers",
                });
            }
        }

        if self.deepstack_output_size == 0 && !self.deepstack_layers.is_empty() {
            return Err(TensorError::ValueOutOfRange {
                description:
                    "Qwen3VLVitConfig: deepstack_output_size must be > 0 when deepstack_layers is non-empty",
            });
        }

        Ok(())
    }

    /// Head dimension: `hidden_size / num_heads`.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}

#[cfg(test)]
#[path = "qwen2vl_config_tests.rs"]
mod tests;
