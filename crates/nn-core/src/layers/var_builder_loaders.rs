// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! [`VarBuilder`]-based constructors for nn layers.
//!
//! Each `Layer::load(vb, ...)` method encodes PyTorch weight naming conventions
//! so model porting code can use one-line construction instead of manual
//! `vb.get()` calls for each parameter.
//!
//! Weight names follow PyTorch defaults: `"weight"`, `"bias"`,
//! `"weight_ih_l0"`, `"weight_hh_l0"`, etc.
//!
//! All functions accept `impl AsRef<VarBuilder>` so callers can pass either
//! `&vb` (borrowed) or `vb` (owned). This eliminates the need for cross-crate
//! wrapper functions that exist solely to convert owned → borrowed.

use super::{
    Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Conv3d, Conv3dConfig, ConvTranspose1d,
    ConvTranspose1dConfig, ConvTranspose2d, ConvTranspose2dConfig, Linear,
};
use crate::var_builder::VarBuilder;
use crate::{Result, TensorError};

/// Validate `groups` before integer division to prevent panic.
/// Checks: groups > 0 and channels divisible by groups.
fn validate_groups(channels: usize, groups: usize, caller: &str) -> Result<()> {
    if groups == 0 {
        return Err(TensorError::InvalidShape(format!(
            "{caller}: groups must be > 0"
        )));
    }
    if !channels.is_multiple_of(groups) {
        return Err(TensorError::InvalidShape(format!(
            "{caller}: channels {channels} not divisible by groups {groups}"
        )));
    }
    Ok(())
}

// -- Linear -------------------------------------------------------------------

impl Linear {
    /// Load a Linear layer from a VarBuilder.
    ///
    /// Loads `"weight"` (required) and `"bias"` (auto-detected, optional).
    /// Weight shape: `[out_features, in_features]`.
    ///
    /// Unlike [`linear()`], this method auto-detects bias: if the VarBuilder
    /// contains a `"bias"` tensor it is loaded, otherwise the layer is
    /// bias-free. Use this when loading weights that may or may not have bias.
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let weight = vb.get(&[out_features, in_features], "weight")?;
        let bias = if vb.contains_tensor("bias") {
            Some(vb.get(&[out_features], "bias")?)
        } else {
            None
        };
        Self::new(weight, bias)
    }
}

/// Construct a Linear layer with bias from a VarBuilder.
///
/// Loads `"weight"` `[out_features, in_features]` and `"bias"` `[out_features]`.
/// **Bias is required** — returns an error if `"bias"` is missing from the
/// VarBuilder. Use [`linear_no_bias()`] for layers without bias, or
/// [`Linear::load()`] for auto-detection.
///
/// Matches candle-nn's `linear()` free function for find-and-replace migration.
pub fn linear(
    in_features: usize,
    out_features: usize,
    vb: impl AsRef<VarBuilder>,
) -> Result<Linear> {
    let vb = vb.as_ref();
    let weight = vb.get(&[out_features, in_features], "weight")?;
    let bias = vb.get(&[out_features], "bias")?;
    Linear::new(weight, Some(bias))
}

/// Construct a Linear layer without bias from a VarBuilder.
///
/// Loads only `"weight"` `[out_features, in_features]`.
/// Matches candle-nn's `linear_no_bias()` free function for find-and-replace migration.
pub fn linear_no_bias(
    in_features: usize,
    out_features: usize,
    vb: impl AsRef<VarBuilder>,
) -> Result<Linear> {
    let vb = vb.as_ref();
    let weight = vb.get(&[out_features, in_features], "weight")?;
    Linear::new(weight, None)
}

// -- Conv1d -------------------------------------------------------------------

impl Conv1d {
    /// Load a Conv1d layer from a VarBuilder.
    ///
    /// Loads `"weight"` (required) and `"bias"` (optional).
    /// Weight shape: `[out_channels, in_channels/groups, kernel_size]`.
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        config: Conv1dConfig,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        validate_groups(in_channels, config.groups, "Conv1d::load")?;
        let weight = vb.get(
            &[out_channels, in_channels / config.groups, kernel_size],
            "weight",
        )?;
        let bias = if vb.contains_tensor("bias") {
            Some(vb.get(&[out_channels], "bias")?)
        } else {
            None
        };
        Self::new(weight, bias, config)
    }
}

/// Construct a Conv1d layer with bias from a VarBuilder.
///
/// Loads `"weight"` and `"bias"` (both required).
/// Matches candle-nn's `conv1d()` free function for find-and-replace migration.
pub fn conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: Conv1dConfig,
    vb: impl AsRef<VarBuilder>,
) -> Result<Conv1d> {
    let vb = vb.as_ref();
    validate_groups(in_channels, config.groups, "conv1d")?;
    let weight = vb.get(
        &[out_channels, in_channels / config.groups, kernel_size],
        "weight",
    )?;
    let bias = vb.get(&[out_channels], "bias")?;
    Conv1d::new(weight, Some(bias), config)
}

/// Construct a Conv1d layer without bias from a VarBuilder.
///
/// Loads only `"weight"`.
/// Matches candle-nn's `conv1d_no_bias()` free function for find-and-replace migration.
pub fn conv1d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: Conv1dConfig,
    vb: impl AsRef<VarBuilder>,
) -> Result<Conv1d> {
    let vb = vb.as_ref();
    validate_groups(in_channels, config.groups, "conv1d_no_bias")?;
    let weight = vb.get(
        &[out_channels, in_channels / config.groups, kernel_size],
        "weight",
    )?;
    Conv1d::new(weight, None, config)
}

// -- Conv2d -------------------------------------------------------------------
fn conv2d_weight_shape(
    out_channels: usize,
    in_channels: usize,
    groups: usize,
    kernel_size: usize,
) -> [usize; 4] {
    [out_channels, in_channels / groups, kernel_size, kernel_size]
}
impl Conv2d {
    /// Load Conv2d: `"weight"` `[out, in/groups, kH, kW]`, optional `"bias"` `[out]`.
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        config: Conv2dConfig,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        validate_groups(in_channels, config.groups, "Conv2d::load")?;
        let weight = vb.get(
            &conv2d_weight_shape(out_channels, in_channels, config.groups, kernel_size),
            "weight",
        )?;
        let bias = if vb.contains_tensor("bias") {
            Some(vb.get(&[out_channels], "bias")?)
        } else {
            None
        };
        Self::new(weight, bias, config)
    }
}
/// Construct a Conv2d with bias. Matches candle-nn's `conv2d()`.
pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: Conv2dConfig,
    vb: impl AsRef<VarBuilder>,
) -> Result<Conv2d> {
    let vb = vb.as_ref();
    validate_groups(in_channels, config.groups, "conv2d")?;
    let weight = vb.get(
        &conv2d_weight_shape(out_channels, in_channels, config.groups, kernel_size),
        "weight",
    )?;
    let bias = vb.get(&[out_channels], "bias")?;
    Conv2d::new(weight, Some(bias), config)
}

/// Construct a Conv2d without bias. Matches candle-nn's `conv2d_no_bias()`.
pub fn conv2d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: Conv2dConfig,
    vb: impl AsRef<VarBuilder>,
) -> Result<Conv2d> {
    let vb = vb.as_ref();
    validate_groups(in_channels, config.groups, "conv2d_no_bias")?;
    let weight = vb.get(
        &conv2d_weight_shape(out_channels, in_channels, config.groups, kernel_size),
        "weight",
    )?;
    Conv2d::new(weight, None, config)
}

// -- Conv3d -------------------------------------------------------------------

impl Conv3d {
    /// Load a Conv3d layer from a VarBuilder.
    ///
    /// Loads `"weight"` (required) and `"bias"` (optional).
    /// Weight shape: `[out_channels, in_channels/groups, kD, kH, kW]`.
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: [usize; 3],
        config: Conv3dConfig,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        validate_groups(in_channels, config.groups, "Conv3d::load")?;
        let weight = vb.get(
            &[
                out_channels,
                in_channels / config.groups,
                kernel_size[0],
                kernel_size[1],
                kernel_size[2],
            ],
            "weight",
        )?;
        let bias = if vb.contains_tensor("bias") {
            Some(vb.get(&[out_channels], "bias")?)
        } else {
            None
        };
        Self::new(weight, bias, config)
    }
}

/// Construct a Conv3d layer with bias from a VarBuilder.
///
/// Loads `"weight"` and `"bias"` (both required).
pub fn conv3d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: [usize; 3],
    config: Conv3dConfig,
    vb: impl AsRef<VarBuilder>,
) -> Result<Conv3d> {
    let vb = vb.as_ref();
    validate_groups(in_channels, config.groups, "conv3d")?;
    let weight = vb.get(
        &[
            out_channels,
            in_channels / config.groups,
            kernel_size[0],
            kernel_size[1],
            kernel_size[2],
        ],
        "weight",
    )?;
    let bias = vb.get(&[out_channels], "bias")?;
    Conv3d::new(weight, Some(bias), config)
}

/// Construct a Conv3d layer without bias from a VarBuilder.
///
/// Loads only `"weight"`.
pub fn conv3d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: [usize; 3],
    config: Conv3dConfig,
    vb: impl AsRef<VarBuilder>,
) -> Result<Conv3d> {
    let vb = vb.as_ref();
    validate_groups(in_channels, config.groups, "conv3d_no_bias")?;
    let weight = vb.get(
        &[
            out_channels,
            in_channels / config.groups,
            kernel_size[0],
            kernel_size[1],
            kernel_size[2],
        ],
        "weight",
    )?;
    Conv3d::new(weight, None, config)
}

// -- ConvTranspose1d ----------------------------------------------------------

impl ConvTranspose1d {
    /// Load a ConvTranspose1d layer from a VarBuilder.
    ///
    /// Loads `"weight"` (required) and `"bias"` (optional).
    /// Weight shape: `[in_channels, out_channels/groups, kernel_size]`.
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        config: ConvTranspose1dConfig,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        validate_groups(out_channels, config.groups, "ConvTranspose1d::load")?;
        let weight = vb.get(
            &[in_channels, out_channels / config.groups, kernel_size],
            "weight",
        )?;
        let bias = if vb.contains_tensor("bias") {
            Some(vb.get(&[out_channels], "bias")?)
        } else {
            None
        };
        Self::new(weight, bias, config)
    }
}

/// Construct a ConvTranspose1d layer with bias from a VarBuilder.
///
/// Loads `"weight"` and `"bias"` (both required).
/// Matches candle-nn's `conv_transpose1d()` free function.
pub fn conv_transpose1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: ConvTranspose1dConfig,
    vb: impl AsRef<VarBuilder>,
) -> Result<ConvTranspose1d> {
    let vb = vb.as_ref();
    validate_groups(out_channels, config.groups, "conv_transpose1d")?;
    let weight = vb.get(
        &[in_channels, out_channels / config.groups, kernel_size],
        "weight",
    )?;
    let bias = vb.get(&[out_channels], "bias")?;
    ConvTranspose1d::new(weight, Some(bias), config)
}

/// Construct a ConvTranspose1d layer without bias from a VarBuilder.
///
/// Loads only `"weight"`.
/// Matches candle-nn's `conv_transpose1d_no_bias()` free function.
pub fn conv_transpose1d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: ConvTranspose1dConfig,
    vb: impl AsRef<VarBuilder>,
) -> Result<ConvTranspose1d> {
    let vb = vb.as_ref();
    validate_groups(out_channels, config.groups, "conv_transpose1d_no_bias")?;
    let weight = vb.get(
        &[in_channels, out_channels / config.groups, kernel_size],
        "weight",
    )?;
    ConvTranspose1d::new(weight, None, config)
}

// -- ConvTranspose2d ----------------------------------------------------------

impl ConvTranspose2d {
    /// Load a ConvTranspose2d layer from a VarBuilder.
    ///
    /// Loads `"weight"` (required) and `"bias"` (optional).
    /// Weight shape: `[in_channels, out_channels/groups, kH, kW]`.
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        config: ConvTranspose2dConfig,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        validate_groups(out_channels, config.groups, "ConvTranspose2d::load")?;
        let weight = vb.get(
            &[
                in_channels,
                out_channels / config.groups,
                kernel_size,
                kernel_size,
            ],
            "weight",
        )?;
        let bias = if vb.contains_tensor("bias") {
            Some(vb.get(&[out_channels], "bias")?)
        } else {
            None
        };
        Self::new(weight, bias, config)
    }
}

/// Construct a ConvTranspose2d layer with bias from a VarBuilder.
///
/// Loads `"weight"` and `"bias"` (both required).
pub fn conv_transpose2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: ConvTranspose2dConfig,
    vb: impl AsRef<VarBuilder>,
) -> Result<ConvTranspose2d> {
    let vb = vb.as_ref();
    validate_groups(out_channels, config.groups, "conv_transpose2d")?;
    let weight = vb.get(
        &[
            in_channels,
            out_channels / config.groups,
            kernel_size,
            kernel_size,
        ],
        "weight",
    )?;
    let bias = vb.get(&[out_channels], "bias")?;
    ConvTranspose2d::new(weight, Some(bias), config)
}

/// Construct a ConvTranspose2d layer without bias from a VarBuilder.
///
/// Loads only `"weight"`.
pub fn conv_transpose2d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: ConvTranspose2dConfig,
    vb: impl AsRef<VarBuilder>,
) -> Result<ConvTranspose2d> {
    let vb = vb.as_ref();
    validate_groups(out_channels, config.groups, "conv_transpose2d_no_bias")?;
    let weight = vb.get(
        &[
            in_channels,
            out_channels / config.groups,
            kernel_size,
            kernel_size,
        ],
        "weight",
    )?;
    ConvTranspose2d::new(weight, None, config)
}

// -- Norm + Embedding + LSTM loaders (extracted to var_builder_loaders_norm.rs) -
#[path = "var_builder_loaders_norm.rs"]
mod norm;
pub use norm::{batch_norm, embedding, group_norm, layer_norm, lstm, rms_norm, LayerNormConfig};

#[cfg(test)]
#[path = "var_builder_tests.rs"]
mod tests;

#[cfg(kani)]
#[path = "var_builder_loaders_kani.rs"]
mod kani_proofs;
