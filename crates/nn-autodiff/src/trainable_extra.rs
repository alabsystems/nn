// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional trainable layer wrappers: Embedding, Conv2d.
//!
//! Normalization wrappers (LayerNorm, RmsNorm, GroupNorm, BatchNorm,
//! InstanceNorm) are in `trainable_extra_norm.rs`.
//! Extracted from `trainable.rs` to keep file sizes under 500 lines.

use crate::error::Result;
use crate::tracked::TrackedTensor;
use crate::var::Var;
use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;
use std::sync::Arc;

use super::TrainableModule;

#[path = "trainable_extra_norm.rs"]
mod norm;
pub use norm::{
    TrainableBatchNorm, TrainableGroupNorm, TrainableInstanceNorm, TrainableLayerNorm,
    TrainableRmsNorm,
};

#[path = "trainable_lstm.rs"]
mod lstm;
pub use lstm::{TrackedLstmState, TrainableLstm};

#[path = "trainable_mha.rs"]
mod mha;
pub use mha::TrainableMultiHeadAttention;

#[path = "trainable_swiglu.rs"]
mod swiglu;
pub use swiglu::TrainableSwiGlu;

#[cfg(test)]
#[path = "trainable_extra_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "trainable_advanced_tests.rs"]
mod advanced_tests;

/// An embedding layer with a trainable weight matrix.
///
/// Performs lookup: `y = weight[indices]` where `weight` is `[vocab_size, embed_dim]`.
/// The weight `Var` receives gradients via `Op::Embedding` backward rule (scatter_add).
///
/// Matches `layers::Embedding` semantics for training.
#[derive(Debug, Clone)]
pub struct TrainableEmbedding {
    weight: Var, // [vocab_size, embed_dim]
}

impl TrainableEmbedding {
    /// Create a new embedding with standard normal initialization.
    ///
    /// Matches PyTorch's `nn.Embedding` default: N(0, 1).
    /// Weight shape: `[vocab_size, embed_dim]`.
    pub fn new(vocab_size: usize, embed_dim: usize) -> Result<Self> {
        let weight = Var::randn(&[vocab_size, embed_dim], 0.0, 1.0, &Device::Cpu)?;
        Ok(Self { weight })
    }

    /// Create from an existing `Var`.
    ///
    /// Weight must be 2D `[vocab_size, embed_dim]`.
    pub fn from_var(weight: Var) -> Self {
        Self { weight }
    }

    /// Create from a `DynTensor` (wraps in a new `Var`).
    pub fn from_tensor(weight: DynTensor) -> Self {
        Self {
            weight: Var::new(weight),
        }
    }

    /// Reference to the weight `Var`.
    #[must_use]
    pub fn weight(&self) -> &Var {
        &self.weight
    }

    /// Forward pass: look up embeddings by indices.
    ///
    /// `indices` is a tracked tensor of integer indices (will be converted internally).
    /// Returns `[..indices_shape, embed_dim]`.
    pub fn forward_indices(&self, indices: &Arc<TrackedTensor>) -> Result<Arc<TrackedTensor>> {
        let w = Arc::new(TrackedTensor::from_var(&self.weight)?);
        TrackedTensor::embedding(&w, indices)
    }
}

impl TrainableModule for TrainableEmbedding {
    fn forward(&self, x: &Arc<TrackedTensor>) -> Result<Arc<TrackedTensor>> {
        self.forward_indices(x)
    }

    fn vars(&self) -> Vec<&Var> {
        vec![&self.weight]
    }
}

/// A 1-D transposed convolution (deconvolution) layer with trainable `Var` weights.
///
/// Weight layout: `[in_channels, out_channels/groups, kernel_size]`.
/// Forward: `y = conv_transpose1d(x, weight, padding, stride, dilation, groups, output_padding) + bias`
///
/// Matches `layers::ConvTranspose1d` semantics for training.
/// Required for HTDemucs decoder upsampling.
#[derive(Debug, Clone)]
pub struct TrainableConvTranspose1d {
    weight: Var, // [in_channels, out_channels/groups, kernel_size]
    bias: Option<Var>,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    output_padding: usize,
}

impl TrainableConvTranspose1d {
    /// Create from existing `Var`s and convolution parameters.
    ///
    /// Weight must be 3D `[in_channels, out_channels/groups, kernel_size]`.
    /// Bias, if present, must be `[out_channels]`.
    pub fn from_vars(
        weight: Var,
        bias: Option<Var>,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        output_padding: usize,
    ) -> Self {
        Self {
            weight,
            bias,
            padding,
            stride,
            dilation,
            groups,
            output_padding,
        }
    }

    /// Create from `DynTensor` weight and optional bias (wraps each in a new `Var`).
    pub fn from_tensors(
        weight: DynTensor,
        bias: Option<DynTensor>,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        output_padding: usize,
    ) -> Self {
        Self {
            weight: Var::new(weight),
            bias: bias.map(Var::new),
            padding,
            stride,
            dilation,
            groups,
            output_padding,
        }
    }

    /// Reference to the weight `Var`.
    #[must_use]
    pub fn weight(&self) -> &Var {
        &self.weight
    }

    /// Reference to the bias `Var`, if present.
    #[must_use]
    pub fn bias(&self) -> Option<&Var> {
        self.bias.as_ref()
    }
}

impl TrainableModule for TrainableConvTranspose1d {
    fn forward(&self, x: &Arc<TrackedTensor>) -> Result<Arc<TrackedTensor>> {
        let w = Arc::new(TrackedTensor::from_var(&self.weight)?);
        let y = x.conv_transpose1d(
            &w,
            self.padding,
            self.stride,
            self.dilation,
            self.groups,
            self.output_padding,
        )?;
        match &self.bias {
            Some(b) => {
                // Bias shape [out_channels] needs broadcast to [batch, out_channels, length].
                // Reshape to [1, out_channels, 1] for broadcasting.
                let b_tracked = Arc::new(TrackedTensor::from_var(b)?);
                let out_ch = b.dims()?[0];
                let b_reshaped = b_tracked.reshape(&[1, out_ch, 1])?;
                y.add(&b_reshaped)
            }
            None => Ok(y),
        }
    }

    fn vars(&self) -> Vec<&Var> {
        let mut v = vec![&self.weight];
        if let Some(b) = &self.bias {
            v.push(b);
        }
        v
    }
}

/// A 2-D convolution layer with trainable `Var` weights.
///
/// Weight layout: `[out_channels, in_channels/groups, kH, kW]`.
/// Forward: `y = conv2d(x, weight, padding, stride, dilation, groups) + bias`
///
/// Matches `layers::Conv2d` semantics for training.
#[derive(Debug, Clone)]
pub struct TrainableConv2d {
    weight: Var, // [out_channels, in_channels/groups, kH, kW]
    bias: Option<Var>,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
}

impl TrainableConv2d {
    /// Create from existing `Var`s and convolution parameters.
    ///
    /// Weight must be 4D `[out_channels, in_channels/groups, kH, kW]`.
    /// Bias, if present, must be `[out_channels]`.
    pub fn from_vars(
        weight: Var,
        bias: Option<Var>,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Self {
        Self {
            weight,
            bias,
            padding,
            stride,
            dilation,
            groups,
        }
    }

    /// Create from `DynTensor` weight and optional bias (wraps each in a new `Var`).
    pub fn from_tensors(
        weight: DynTensor,
        bias: Option<DynTensor>,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Self {
        Self {
            weight: Var::new(weight),
            bias: bias.map(Var::new),
            padding,
            stride,
            dilation,
            groups,
        }
    }

    /// Reference to the weight `Var`.
    #[must_use]
    pub fn weight(&self) -> &Var {
        &self.weight
    }

    /// Reference to the bias `Var`, if present.
    #[must_use]
    pub fn bias(&self) -> Option<&Var> {
        self.bias.as_ref()
    }
}

impl TrainableModule for TrainableConv2d {
    fn forward(&self, x: &Arc<TrackedTensor>) -> Result<Arc<TrackedTensor>> {
        let w = Arc::new(TrackedTensor::from_var(&self.weight)?);
        let y = x.conv2d(&w, self.padding, self.stride, self.dilation, self.groups)?;
        match &self.bias {
            Some(b) => {
                // Bias shape [out_channels] needs broadcast to [batch, out_channels, H, W].
                // Reshape to [1, out_channels, 1, 1] for broadcasting.
                let b_tracked = Arc::new(TrackedTensor::from_var(b)?);
                let b_reshaped = b_tracked.reshape(&[1, b.dims()?[0], 1, 1])?;
                y.add(&b_reshaped)
            }
            None => Ok(y),
        }
    }

    fn vars(&self) -> Vec<&Var> {
        let mut v = vec![&self.weight];
        if let Some(b) = &self.bias {
            v.push(b);
        }
        v
    }
}
