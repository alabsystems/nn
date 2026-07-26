// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! 1-D transposed convolution layer.
//!
//! Extracted from `conv.rs` for 500-line compliance (#1280 Direction 3).

use crate::dyn_tensor::trace::TraceOp;
use crate::dyn_tensor::{gpu_backend_dispatch, DynTensor};
use crate::error::{Result, TensorError};
use crate::layers::Module;

/// Configuration for a [`ConvTranspose1d`] layer.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct ConvTranspose1dConfig {
    pub padding: usize,
    pub output_padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Default for ConvTranspose1dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            output_padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }
    }
}

impl ConvTranspose1dConfig {
    /// Create a config with the common `(padding, stride, dilation)` triple.
    ///
    /// Output padding and groups default to 0 and 1 respectively.
    /// Chain `.with_output_padding(n)` or `.with_groups(n)` as needed.
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

    /// Set output padding.
    #[must_use]
    pub fn with_output_padding(mut self, output_padding: usize) -> Self {
        self.output_padding = output_padding;
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

/// 1-D transposed convolution (deconvolution) layer.
///
/// Weight shape: `[in_channels, out_channels/groups, kernel_size]`.
/// Bias shape: `[out_channels]` (if present).
///
/// Matches candle-nn's `ConvTranspose1d`.
#[derive(Clone, Debug)]
pub struct ConvTranspose1d {
    weight: DynTensor,
    bias: Option<DynTensor>,
    config: ConvTranspose1dConfig,
}

impl ConvTranspose1d {
    /// Create from pre-loaded weight and optional bias.
    ///
    /// - `weight`: shape `[in_channels, out_channels/groups, kernel_size]` (must be 3D)
    /// - `config.groups` must be > 0
    ///
    /// Returns an error if `weight` is not 3D or `groups` is zero.
    pub fn new(
        weight: DynTensor,
        bias: Option<DynTensor>,
        config: ConvTranspose1dConfig,
    ) -> Result<Self> {
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
    pub fn config(&self) -> &ConvTranspose1dConfig {
        &self.config
    }

    fn conv_transpose1d_cpu(&self, x: &DynTensor) -> Result<DynTensor> {
        let y = x.conv_transpose1d(
            &self.weight,
            self.config.padding,
            self.config.output_padding,
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

impl Module for ConvTranspose1d {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        crate::layers::traced_forward(
            &[x],
            || {
                Ok(TraceOp::ConvTranspose1d {
                    weight: self.weight.to_weight_ref()?,
                    bias: self
                        .bias
                        .as_ref()
                        .map(DynTensor::to_weight_ref)
                        .transpose()?,
                    padding: self.config.padding,
                    output_padding: self.config.output_padding,
                    stride: self.config.stride,
                    dilation: self.config.dilation,
                    groups: self.config.groups,
                })
            },
            || {
                // GPU fused path: pass bias directly to GPU backend (1 dispatch, not 3).
                if x.device().is_gpu() && self.bias.is_some() {
                    if let Some(r) = gpu_backend_dispatch(|b| {
                        b.conv_transpose1d(
                            x,
                            &self.weight,
                            self.bias.as_ref(),
                            self.config.padding,
                            self.config.output_padding,
                            self.config.stride,
                            self.config.dilation,
                            self.config.groups,
                        )
                    }) {
                        return r;
                    }
                }
                self.conv_transpose1d_cpu(x)
            },
        )
    }
}
