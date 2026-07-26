// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! VarBuilder constructors for normalization, embedding, and recurrent layers.
//!
//! Extracted from `var_builder_loaders.rs` to stay under the 500-line limit.
//! Wired via `#[path]` submodule in the parent module.

use super::super::{
    BatchNorm, BatchNormConfig, BiLstm, Embedding, GroupNorm, LayerNorm, Lstm, RmsNorm,
};
use crate::var_builder::VarBuilder;
use crate::Result;

// -- LayerNorm ----------------------------------------------------------------

/// Configuration for [`LayerNorm`].
///
/// Matches candle-nn's `LayerNormConfig` for find-and-replace migration.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct LayerNormConfig {
    /// Epsilon for numerical stability. Default: 1e-5.
    pub eps: f64,
}

impl LayerNormConfig {
    /// Create config with custom epsilon.
    #[must_use]
    pub fn new(eps: f64) -> Self {
        Self { eps }
    }
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self { eps: 1e-5 }
    }
}

impl LayerNorm {
    /// Load a LayerNorm layer from a VarBuilder.
    ///
    /// Loads `"weight"` (gamma) and `"bias"` (beta), both required.
    /// Shape: `[normalized_dim]`.
    pub fn load(vb: impl AsRef<VarBuilder>, dim: usize, eps: f64) -> Result<Self> {
        let vb = vb.as_ref();
        let weight = vb.get(&[dim], "weight")?;
        let bias = vb.get(&[dim], "bias")?;
        Self::new(weight, bias, eps)
    }
}

/// Construct a LayerNorm from a VarBuilder.
///
/// Loads `"weight"` (gamma) and `"bias"` (beta), both required.
/// Matches candle-nn's `layer_norm()` free function.
pub fn layer_norm(
    size: usize,
    config: LayerNormConfig,
    vb: impl AsRef<VarBuilder>,
) -> Result<LayerNorm> {
    LayerNorm::load(vb, size, config.eps)
}

// -- GroupNorm ----------------------------------------------------------------

impl GroupNorm {
    /// Load a GroupNorm layer from a VarBuilder.
    ///
    /// Loads `"weight"` (gamma) and `"bias"` (beta), both required.
    /// Shape: `[num_channels]`.
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        num_groups: usize,
        num_channels: usize,
        eps: f64,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let weight = vb.get(&[num_channels], "weight")?;
        let bias = vb.get(&[num_channels], "bias")?;
        Self::new(num_groups, num_channels, weight, bias, eps)
    }
}

/// Construct a GroupNorm from a VarBuilder.
///
/// Loads `"weight"` (gamma) and `"bias"` (beta), both required.
/// Default eps is 1e-5 (same as PyTorch).
/// Matches candle-nn's `group_norm()` free function.
pub fn group_norm(
    num_groups: usize,
    num_channels: usize,
    eps: f64,
    vb: impl AsRef<VarBuilder>,
) -> Result<GroupNorm> {
    GroupNorm::load(vb, num_groups, num_channels, eps)
}

// -- BatchNorm ----------------------------------------------------------------

impl BatchNorm {
    /// Load a BatchNorm layer from a VarBuilder.
    ///
    /// Loads running statistics (`"running_mean"`, `"running_var"`) and
    /// optional affine parameters (`"weight"`, `"bias"`).
    /// Shape: `[num_features]`.
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        num_features: usize,
        config: BatchNormConfig,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let running_mean = vb.get(&[num_features], "running_mean")?;
        let running_var = vb.get(&[num_features], "running_var")?;
        let weight = if config.affine && vb.contains_tensor("weight") {
            Some(vb.get(&[num_features], "weight")?)
        } else {
            None
        };
        let bias = if config.affine && vb.contains_tensor("bias") {
            Some(vb.get(&[num_features], "bias")?)
        } else {
            None
        };
        Self::with_config(running_mean, running_var, weight, bias, config)
    }
}

/// Construct a BatchNorm layer from a VarBuilder.
///
/// Loads running statistics and optional affine parameters.
/// Matches candle-nn's `batch_norm()` free function.
pub fn batch_norm(
    num_features: usize,
    config: BatchNormConfig,
    vb: impl AsRef<VarBuilder>,
) -> Result<BatchNorm> {
    BatchNorm::load(vb, num_features, config)
}

// -- RmsNorm ------------------------------------------------------------------

impl RmsNorm {
    /// Load an RmsNorm layer from a VarBuilder.
    ///
    /// Loads `"weight"` (gamma). Shape: `[dim]`.
    pub fn load(vb: impl AsRef<VarBuilder>, dim: usize, eps: f64) -> Result<Self> {
        let vb = vb.as_ref();
        let weight = vb.get(&[dim], "weight")?;
        Self::new(weight, eps)
    }
}

/// Construct an RmsNorm from a VarBuilder.
///
/// Loads `"weight"` (gamma). Matches candle-nn's `rms_norm()` free function.
pub fn rms_norm(size: usize, eps: f64, vb: impl AsRef<VarBuilder>) -> Result<RmsNorm> {
    RmsNorm::load(vb, size, eps)
}

// -- Embedding ----------------------------------------------------------------

impl Embedding {
    /// Load an Embedding layer from a VarBuilder.
    ///
    /// Loads `"weight"`. Shape: `[vocab_size, embedding_dim]`.
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        vocab_size: usize,
        embedding_dim: usize,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let weight = vb.get(&[vocab_size, embedding_dim], "weight")?;
        Self::new(weight)
    }
}

/// Construct an Embedding from a VarBuilder.
///
/// Loads `"weight"` `[vocab_size, hidden_size]`.
/// Matches candle-nn's `embedding()` free function.
pub fn embedding(
    vocab_size: usize,
    hidden_size: usize,
    vb: impl AsRef<VarBuilder>,
) -> Result<Embedding> {
    Embedding::load(vb, vocab_size, hidden_size)
}

// -- Lstm ---------------------------------------------------------------------

impl Lstm {
    /// Load an LSTM cell from a VarBuilder.
    ///
    /// Loads PyTorch LSTM weight convention:
    /// - `"weight_ih_l0"`: `[4*hidden_size, input_size]`
    /// - `"weight_hh_l0"`: `[4*hidden_size, hidden_size]`
    /// - `"bias_ih_l0"`: `[4*hidden_size]` (optional)
    /// - `"bias_hh_l0"`: `[4*hidden_size]` (optional)
    pub fn load(vb: impl AsRef<VarBuilder>, input_size: usize, hidden_size: usize) -> Result<Self> {
        let vb = vb.as_ref();
        let four_h = 4 * hidden_size;
        let w_ih = vb.get(&[four_h, input_size], "weight_ih_l0")?;
        let w_hh = vb.get(&[four_h, hidden_size], "weight_hh_l0")?;
        let b_ih = if vb.contains_tensor("bias_ih_l0") {
            Some(vb.get(&[four_h], "bias_ih_l0")?)
        } else {
            None
        };
        let b_hh = if vb.contains_tensor("bias_hh_l0") {
            Some(vb.get(&[four_h], "bias_hh_l0")?)
        } else {
            None
        };
        Self::new(w_ih, w_hh, b_ih, b_hh, hidden_size)
    }
}

/// Construct an LSTM cell from a VarBuilder.
///
/// Loads PyTorch LSTM weight convention (`weight_ih_l0`, `weight_hh_l0`,
/// optional `bias_ih_l0`, `bias_hh_l0`).
/// Matches candle-nn's `lstm()` free function for find-and-replace migration.
pub fn lstm(input_size: usize, hidden_size: usize, vb: impl AsRef<VarBuilder>) -> Result<Lstm> {
    Lstm::load(vb, input_size, hidden_size)
}

// -- BiLstm -------------------------------------------------------------------

impl BiLstm {
    /// Load a bidirectional LSTM from a VarBuilder.
    ///
    /// Supports three naming conventions:
    /// - **PyTorch-native:** `"weight_ih_l0"`, `"weight_ih_l0_reverse"`, etc.
    /// - **Keyremap hybrid:** `"forward.weight_ih_l0"`, `"backward.weight_ih_l0"`, etc.
    ///   (directional prefix + PyTorch leaf name, produced by kokoro_reference_keyremap.py)
    /// - **Decomposed (dvoice):** `"forward.weight_ih.weight"`,
    ///   `"backward.weight_ih.weight"`, etc.
    ///
    /// Tries PyTorch-native first, then keyremap hybrid, then decomposed.
    /// Part of #2741, #2691.
    pub fn load(vb: impl AsRef<VarBuilder>, input_size: usize, hidden_size: usize) -> Result<Self> {
        let vb = vb.as_ref();
        let four_h = 4 * hidden_size;

        // Forward direction: PyTorch-native → keyremap hybrid → decomposed
        let w_ih_fwd = vb
            .get(&[four_h, input_size], "weight_ih_l0")
            .or_else(|_| vb.get(&[four_h, input_size], "forward.weight_ih_l0"))
            .or_else(|_| vb.get(&[four_h, input_size], "forward.weight_ih.weight"))?;
        let w_hh_fwd = vb
            .get(&[four_h, hidden_size], "weight_hh_l0")
            .or_else(|_| vb.get(&[four_h, hidden_size], "forward.weight_hh_l0"))
            .or_else(|_| vb.get(&[four_h, hidden_size], "forward.weight_hh.weight"))?;
        let b_ih_fwd = load_optional_bias_3(
            vb,
            four_h,
            "bias_ih_l0",
            "forward.bias_ih_l0",
            "forward.weight_ih.bias",
        );
        let b_hh_fwd = load_optional_bias_3(
            vb,
            four_h,
            "bias_hh_l0",
            "forward.bias_hh_l0",
            "forward.weight_hh.bias",
        );

        // Backward direction: PyTorch-native → keyremap hybrid → decomposed
        let w_ih_rev = vb
            .get(&[four_h, input_size], "weight_ih_l0_reverse")
            .or_else(|_| vb.get(&[four_h, input_size], "backward.weight_ih_l0"))
            .or_else(|_| vb.get(&[four_h, input_size], "backward.weight_ih.weight"))?;
        let w_hh_rev = vb
            .get(&[four_h, hidden_size], "weight_hh_l0_reverse")
            .or_else(|_| vb.get(&[four_h, hidden_size], "backward.weight_hh_l0"))
            .or_else(|_| vb.get(&[four_h, hidden_size], "backward.weight_hh.weight"))?;
        let b_ih_rev = load_optional_bias_3(
            vb,
            four_h,
            "bias_ih_l0_reverse",
            "backward.bias_ih_l0",
            "backward.weight_ih.bias",
        );
        let b_hh_rev = load_optional_bias_3(
            vb,
            four_h,
            "bias_hh_l0_reverse",
            "backward.bias_hh_l0",
            "backward.weight_hh.bias",
        );

        Self::from_weights(
            w_ih_fwd,
            w_hh_fwd,
            b_ih_fwd,
            b_hh_fwd,
            w_ih_rev,
            w_hh_rev,
            b_ih_rev,
            b_hh_rev,
            hidden_size,
        )
    }
}

/// Three-way optional bias: PyTorch-native → keyremap hybrid → decomposed.
/// Part of #2691.
fn load_optional_bias_3(
    vb: &VarBuilder,
    size: usize,
    native_name: &str,
    hybrid_name: &str,
    decomposed_name: &str,
) -> Option<crate::dyn_tensor::DynTensor> {
    if vb.contains_tensor(native_name) {
        vb.get(&[size], native_name).ok()
    } else if vb.contains_tensor(hybrid_name) {
        vb.get(&[size], hybrid_name).ok()
    } else if vb.contains_tensor(decomposed_name) {
        vb.get(&[size], decomposed_name).ok()
    } else {
        None
    }
}
