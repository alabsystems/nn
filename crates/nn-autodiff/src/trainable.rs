// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Trainable layer implementations that own [`Var`] weights.
//!
//! These types bridge [`nn_core::layers`] layers (inference, `DynTensor`) with the
//! autodiff system (`TrackedTensor` + gradient tape). Each layer owns its weight
//! [`Var`]s directly, so gradients accumulate during `backward()`.
//!
//! # Example
//!
//! ```no_run
//! use nn_autodiff::trainable::{TrainableLinear, TrainableModule};
//! use nn_autodiff::{TrackedTensor, Var, backward};
//! use nn_core::{DType, Device, DynTensor};
//! use std::sync::Arc;
//!
//! let layer = TrainableLinear::new(4, 3, true).expect("valid dims");
//! let x = DynTensor::from_vec(vec![1.0; 8], &[2, 4], &Device::Cpu).expect("valid");
//! let x_tracked = Arc::new(TrackedTensor::from_tensor(x));
//! let y = layer.forward(&x_tracked).expect("forward");
//! // y is [2, 3], tracked on the gradient tape
//! ```

use crate::error::Result;
use crate::tracked::TrackedTensor;
use crate::var::Var;
use crate::var_init::Init;
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use std::sync::Arc;

#[path = "trainable_extra.rs"]
mod extra;
pub use extra::{
    TrackedLstmState, TrainableBatchNorm, TrainableConv2d, TrainableConvTranspose1d,
    TrainableEmbedding, TrainableGroupNorm, TrainableInstanceNorm, TrainableLayerNorm,
    TrainableLstm, TrainableMultiHeadAttention, TrainableRmsNorm, TrainableSwiGlu,
};

/// A trainable module that can be used in a gradient-tracked forward pass.
///
/// Analogous to PyTorch's `nn.Module` with `requires_grad=True` parameters.
/// Provides both the forward pass (returning `Arc<TrackedTensor>` for autodiff)
/// and access to trainable parameters for optimizer registration.
pub trait TrainableModule {
    /// Run the forward pass with gradient tracking.
    fn forward(&self, x: &Arc<TrackedTensor>) -> Result<Arc<TrackedTensor>>;

    /// Return references to all trainable [`Var`]s in this module.
    ///
    /// Used by optimizers to register parameters for gradient updates.
    fn vars(&self) -> Vec<&Var>;
}

/// A linear (fully-connected) layer that owns trainable [`Var`] weights.
///
/// Computes `y = x @ weight^T + bias` with operations tracked on the
/// gradient tape. The `weight` and optional `bias` are [`Var`]s whose
/// gradients accumulate during `backward()`.
///
/// Matches `layers::Linear` semantics: weight is `[out_features, in_features]`,
/// bias is `[out_features]`.
#[derive(Debug, Clone)]
pub struct TrainableLinear {
    weight: Var,
    bias: Option<Var>,
}

impl TrainableLinear {
    /// Create a new linear layer with Kaiming uniform weight initialization.
    ///
    /// Matches PyTorch's `nn.Linear` default: weight is initialized with
    /// Kaiming uniform (He init, fan_in), bias is zero-initialized.
    ///
    /// Weight shape: `[out_features, in_features]`.
    /// Bias shape: `[out_features]` (if enabled).
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Result<Self> {
        let weight = Var::kaiming_uniform(&[out_features, in_features], &Device::Cpu)?;
        let bias_var = if bias {
            Some(Var::zeros(&[out_features], DType::F32, &Device::Cpu)?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias: bias_var,
        })
    }

    /// Create a new linear layer with explicit weight initialization.
    ///
    /// Bias is always zero-initialized (matching PyTorch convention).
    pub fn new_with_init(
        in_features: usize,
        out_features: usize,
        bias: bool,
        init: Init,
    ) -> Result<Self> {
        let weight = Var::from_init(init, &[out_features, in_features], &Device::Cpu)?;
        let bias_var = if bias {
            Some(Var::zeros(&[out_features], DType::F32, &Device::Cpu)?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias: bias_var,
        })
    }

    /// Create a linear layer from existing [`Var`]s.
    ///
    /// Useful when loading pre-trained weights or sharing parameters.
    pub fn from_vars(weight: Var, bias: Option<Var>) -> Self {
        Self { weight, bias }
    }

    /// Create from a `DynTensor` weight and optional bias (wraps each in a new `Var`).
    pub fn from_tensors(weight: DynTensor, bias: Option<DynTensor>) -> Self {
        Self {
            weight: Var::new(weight),
            bias: bias.map(Var::new),
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

impl TrainableModule for TrainableLinear {
    fn forward(&self, x: &Arc<TrackedTensor>) -> Result<Arc<TrackedTensor>> {
        let w = Arc::new(TrackedTensor::from_var(&self.weight)?);
        let wt = w.transpose(0, 1)?;
        let y = x.matmul(&wt)?;
        match &self.bias {
            Some(b) => {
                let b_tracked = Arc::new(TrackedTensor::from_var(b)?);
                y.add(&b_tracked)
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

/// A 1-D convolution layer with trainable `Var` weights.
///
/// Mirrors `layers::Conv1d` but operates on `Arc<TrackedTensor>` for training.
/// Weight layout: `[out_channels, in_channels/groups, kernel_size]`.
///
/// Forward: `y = conv1d(x, weight, padding, stride, dilation, groups) + bias`
#[derive(Debug, Clone)]
pub struct TrainableConv1d {
    weight: Var, // [out_channels, in_channels/groups, kernel_size]
    bias: Option<Var>,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
}

impl TrainableConv1d {
    /// Create from existing `Var`s and convolution parameters.
    ///
    /// Weight must be 3D `[out_channels, in_channels/groups, kernel_size]`.
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

impl TrainableModule for TrainableConv1d {
    fn forward(&self, x: &Arc<TrackedTensor>) -> Result<Arc<TrackedTensor>> {
        let w = Arc::new(TrackedTensor::from_var(&self.weight)?);
        let y = x.conv1d(&w, self.padding, self.stride, self.dilation, self.groups)?;
        match &self.bias {
            Some(b) => {
                // Bias shape [out_channels] needs broadcast to [batch, out_channels, length].
                // Reshape to [1, out_channels, 1] for broadcasting.
                let b_tracked = Arc::new(TrackedTensor::from_var(b)?);
                let b_reshaped = b_tracked.reshape(&[1, b.dims()?[0], 1])?;
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

#[cfg(test)]
#[path = "trainable_tests.rs"]
mod tests;
