// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Convolution nn layers — Conv1d and re-exports.
//!
//! Conv2d extracted to `conv2d.rs`. Conv3d extracted to `conv3d.rs`.
//! ConvTranspose1d in `conv_transpose1d.rs`.
//! WeightNormConv1d in `weight_norm_conv1d.rs`.

use super::Module;
use crate::dyn_tensor::trace::TraceOp;
use crate::dyn_tensor::{gpu_backend_dispatch, DynTensor};
use crate::error::{Result, TensorError};

#[path = "conv_transpose1d.rs"]
mod conv_transpose1d_mod;
pub use conv_transpose1d_mod::{ConvTranspose1d, ConvTranspose1dConfig};

#[path = "weight_norm_conv1d.rs"]
mod weight_norm_conv1d_mod;
pub use weight_norm_conv1d_mod::WeightNormConv1d;

/// Configuration for a [`Conv1d`] layer.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Conv1dConfig {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Default for Conv1dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }
    }
}

impl Conv1dConfig {
    /// Create a config with the common `(padding, stride, dilation)` triple.
    ///
    /// Groups defaults to 1. Chain `.with_groups(n)` for grouped convolutions.
    #[must_use]
    pub fn new(padding: usize, stride: usize, dilation: usize) -> Self {
        Self {
            padding,
            stride,
            dilation,
            ..Default::default()
        }
    }

    /// Set padding.
    #[must_use]
    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// Set stride.
    #[must_use]
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Set dilation.
    #[must_use]
    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    /// Set groups.
    #[must_use]
    pub fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }
}

/// 1-D convolution layer.
///
/// Weight shape: `[out_channels, in_channels/groups, kernel_size]`.
/// Bias shape: `[out_channels]` (if present).
///
/// Matches candle-nn's `Conv1d`.
#[derive(Clone, Debug)]
pub struct Conv1d {
    weight: DynTensor,
    bias: Option<DynTensor>,
    config: Conv1dConfig,
}

impl Conv1d {
    /// Create from pre-loaded weight and optional bias.
    ///
    /// - `weight`: shape `[out_channels, in_channels/groups, kernel_size]` (must be 3D)
    /// - `config.groups` must divide `weight` dim 1 × groups == out_channels relationship
    ///
    /// Returns an error if `weight` is not 3D or `groups` is zero.
    pub fn new(weight: DynTensor, bias: Option<DynTensor>, config: Conv1dConfig) -> Result<Self> {
        if weight.rank() != 3 {
            return Err(TensorError::RankMismatch {
                expected: 3,
                actual: weight.rank(),
            });
        }
        if config.groups == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "groups",
                value: 0,
                reason: "must be > 0",
            });
        }
        let out_channels = weight.dims()[0];
        if let Some(ref b) = bias {
            if b.dims() != [out_channels] {
                return Err(TensorError::shape_mismatch(
                    vec![out_channels],
                    b.dims().to_vec(),
                ));
            }
        }
        Ok(Self {
            weight,
            bias,
            config,
        })
    }

    #[must_use]
    pub fn weight(&self) -> &DynTensor {
        &self.weight
    }

    #[must_use]
    pub fn bias(&self) -> Option<&DynTensor> {
        self.bias.as_ref()
    }

    #[must_use]
    pub fn config(&self) -> &Conv1dConfig {
        &self.config
    }

    fn conv1d_cpu(&self, x: &DynTensor) -> Result<DynTensor> {
        let y = x.conv1d(
            &self.weight,
            self.config.padding,
            self.config.stride,
            self.config.dilation,
            self.config.groups,
        )?;
        match &self.bias {
            Some(b) => {
                let b = b.reshape([1, b.numel(), 1])?;
                y.broadcast_add(&b)
            }
            None => Ok(y),
        }
    }
}

impl Module for Conv1d {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        super::traced_forward(
            &[x],
            || {
                Ok(TraceOp::Conv1d {
                    weight: self.weight.to_weight_ref()?,
                    bias: self
                        .bias
                        .as_ref()
                        .map(DynTensor::to_weight_ref)
                        .transpose()?,
                    padding: self.config.padding,
                    stride: self.config.stride,
                    dilation: self.config.dilation,
                    groups: self.config.groups,
                })
            },
            || {
                // GPU fused path: pass bias directly to GPU backend (1 dispatch, not 3).
                if x.device().is_gpu() && self.bias.is_some() {
                    if let Some(r) = gpu_backend_dispatch(|b| {
                        b.conv1d(
                            x,
                            &self.weight,
                            self.bias.as_ref(),
                            self.config.padding,
                            self.config.stride,
                            self.config.dilation,
                            self.config.groups,
                        )
                    }) {
                        return r;
                    }
                }
                self.conv1d_cpu(x)
            },
        )
    }
}

// Conv2d extracted to conv2d.rs
#[path = "conv2d.rs"]
mod conv2d_mod;
pub use conv2d_mod::{Conv2d, Conv2dConfig};

// Conv3d extracted to conv3d.rs
#[path = "conv3d.rs"]
mod conv3d_mod;
pub use conv3d_mod::{Conv3d, Conv3dConfig};

// ConvTranspose2d extracted to conv_transpose2d.rs
#[path = "conv_transpose2d.rs"]
mod conv_transpose2d_mod;
pub use conv_transpose2d_mod::{ConvTranspose2d, ConvTranspose2dConfig};

#[cfg(test)]
#[path = "conv_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "conv2d_tests.rs"]
mod conv2d_tests;

#[cfg(test)]
#[path = "conv_transpose2d_tests.rs"]
mod conv_transpose2d_tests;

#[cfg(test)]
#[path = "conv3d_tests.rs"]
mod conv3d_tests;
