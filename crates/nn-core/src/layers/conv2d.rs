// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! 2-D convolution layer.
//!
//! Extracted from `conv.rs` to keep files under 250 lines.

use super::Module;
use crate::dyn_tensor::trace::TraceOp;
use crate::dyn_tensor::{gpu_backend_dispatch, DynTensor};
use crate::error::{Result, TensorError};
use crate::layers::traced_forward;

/// Configuration for a [`Conv2d`] layer.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Conv2dConfig {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Default for Conv2dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }
    }
}

impl Conv2dConfig {
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

/// 2-D convolution layer.
///
/// Weight shape: `[out_channels, in_channels/groups, kH, kW]`.
/// Bias shape: `[out_channels]` (if present).
///
/// Matches candle-nn's `Conv2d`.
#[derive(Clone, Debug)]
pub struct Conv2d {
    weight: DynTensor,
    bias: Option<DynTensor>,
    config: Conv2dConfig,
}

impl Conv2d {
    /// Create from pre-loaded weight and optional bias.
    ///
    /// - `weight`: shape `[out_channels, in_channels/groups, kH, kW]` (must be 4D)
    /// - `config.groups` must be > 0
    ///
    /// Returns an error if `weight` is not 4D or `groups` is zero.
    pub fn new(weight: DynTensor, bias: Option<DynTensor>, config: Conv2dConfig) -> Result<Self> {
        if weight.rank() != 4 {
            return Err(TensorError::RankMismatch {
                expected: 4,
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
    pub fn config(&self) -> &Conv2dConfig {
        &self.config
    }

    fn conv2d_cpu(&self, x: &DynTensor) -> Result<DynTensor> {
        let y = x.conv2d(
            &self.weight,
            self.config.padding,
            self.config.stride,
            self.config.dilation,
            self.config.groups,
        )?;
        match &self.bias {
            Some(b) => {
                let b = b.reshape([1, b.numel(), 1, 1])?;
                y.broadcast_add(&b)
            }
            None => Ok(y),
        }
    }
}

impl Module for Conv2d {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        traced_forward(
            &[x],
            || {
                Ok(TraceOp::Conv2d {
                    weight: self.weight.to_weight_ref()?,
                    bias: self
                        .bias
                        .as_ref()
                        .map(DynTensor::to_weight_ref)
                        .transpose()?,
                    padding: [self.config.padding, self.config.padding],
                    stride: [self.config.stride, self.config.stride],
                    dilation: [self.config.dilation, self.config.dilation],
                    groups: self.config.groups,
                })
            },
            || {
                if x.device().is_gpu() && self.bias.is_some() {
                    if let Some(r) = gpu_backend_dispatch(|b| {
                        b.conv2d(
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
                self.conv2d_cpu(x)
            },
        )
    }
}
