// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! 2-D transposed convolution layer.
//!
//! Extracted from `conv.rs` to stay under the 500-line limit.
//! Wired via `#[path]` submodule in the parent module.

use super::super::Module;
use crate::dyn_tensor::trace::TraceOp;
use crate::dyn_tensor::DynTensor;
use crate::error::{Result, TensorError};

/// Configuration for a [`ConvTranspose2d`] layer.
///
/// Spatial parameters use `[usize; 2]` for `[height, width]` to support
/// non-square transposed convolutions (e.g., stride=[2,1]).
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct ConvTranspose2dConfig {
    pub padding: [usize; 2],
    pub output_padding: [usize; 2],
    pub stride: [usize; 2],
    pub dilation: [usize; 2],
    pub groups: usize,
}

impl ConvTranspose2dConfig {
    /// Create a config with symmetric `(padding, stride, dilation)`.
    ///
    /// Output padding and groups default to `[0, 0]` and 1 respectively.
    /// Chain `.with_output_padding(n)` or `.with_groups(n)` as needed.
    #[must_use]
    pub fn new(padding: usize, stride: usize, dilation: usize) -> Self {
        Self {
            padding: [padding, padding],
            stride: [stride, stride],
            dilation: [dilation, dilation],
            ..Default::default()
        }
    }

    /// Set symmetric padding.
    #[must_use]
    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = [padding, padding];
        self
    }

    /// Set output padding (symmetric).
    #[must_use]
    pub fn with_output_padding(mut self, output_padding: usize) -> Self {
        self.output_padding = [output_padding, output_padding];
        self
    }

    /// Set symmetric stride.
    #[must_use]
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = [stride, stride];
        self
    }

    /// Set symmetric dilation.
    #[must_use]
    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = [dilation, dilation];
        self
    }

    /// Set groups for grouped transposed convolution.
    #[must_use]
    pub fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }
}

impl Default for ConvTranspose2dConfig {
    fn default() -> Self {
        Self {
            padding: [0, 0],
            output_padding: [0, 0],
            stride: [1, 1],
            dilation: [1, 1],
            groups: 1,
        }
    }
}

/// 2-D transposed convolution (deconvolution) layer.
///
/// Weight shape: `[in_channels, out_channels/groups, kH, kW]`.
/// Bias shape: `[out_channels]` (if present).
///
/// Matches PyTorch's `nn.ConvTranspose2d`.
#[derive(Clone, Debug)]
pub struct ConvTranspose2d {
    weight: DynTensor,
    bias: Option<DynTensor>,
    config: ConvTranspose2dConfig,
}

impl ConvTranspose2d {
    /// Create from pre-loaded weight and optional bias.
    ///
    /// - `weight`: shape `[in_channels, out_channels/groups, kH, kW]` (must be 4D)
    /// - `config.groups` must be > 0
    ///
    /// Returns an error if `weight` is not 4D or `groups` is zero.
    pub fn new(
        weight: DynTensor,
        bias: Option<DynTensor>,
        config: ConvTranspose2dConfig,
    ) -> Result<Self> {
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
    pub fn config(&self) -> &ConvTranspose2dConfig {
        &self.config
    }

    fn conv_transpose2d_cpu(&self, x: &DynTensor) -> Result<DynTensor> {
        let y = x.conv_transpose2d(
            &self.weight,
            self.config.padding,
            self.config.output_padding,
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

impl Module for ConvTranspose2d {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        crate::layers::traced_forward(
            &[x],
            || {
                Ok(TraceOp::ConvTranspose2d {
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
            || self.conv_transpose2d_cpu(x),
        )
    }
}
