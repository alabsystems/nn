// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! 3-D convolution nn layer.
//!
//! Extracted to its own file following the Conv2d pattern from `conv2d.rs`.
//! Uses `DynTensor::conv3d` for the actual convolution computation.

use super::Module;
use crate::dyn_tensor::trace::TraceOp;
use crate::dyn_tensor::{gpu_backend_dispatch, DynTensor};
use crate::error::{Result, TensorError};
use crate::layers::traced_forward;

/// Configuration for a [`Conv3d`] layer.
///
/// All spatial parameters use `[usize; 3]` for `[depth, height, width]` to
/// support non-cubic convolutions (e.g., Qwen3-VL 3D patch embedding with
/// temporal + spatial dims).
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Conv3dConfig {
    /// Zero-padding added to both sides of each spatial dimension `[pad_d, pad_h, pad_w]`.
    pub padding: [usize; 3],
    /// Stride of the convolution `[stride_d, stride_h, stride_w]`.
    pub stride: [usize; 3],
    /// Spacing between kernel elements `[dil_d, dil_h, dil_w]`.
    pub dilation: [usize; 3],
    /// Number of blocked connections from input to output channels.
    pub groups: usize,
}

impl Default for Conv3dConfig {
    fn default() -> Self {
        Self {
            padding: [0, 0, 0],
            stride: [1, 1, 1],
            dilation: [1, 1, 1],
            groups: 1,
        }
    }
}

impl Conv3dConfig {
    /// Create a config with uniform `(padding, stride, dilation)` applied to all 3 dims.
    ///
    /// Groups defaults to 1. Chain `.with_groups(n)` for grouped convolutions.
    #[must_use]
    pub fn new(padding: usize, stride: usize, dilation: usize) -> Self {
        Self {
            padding: [padding, padding, padding],
            stride: [stride, stride, stride],
            dilation: [dilation, dilation, dilation],
            ..Default::default()
        }
    }

    /// Set padding (uniform across all 3 dims).
    #[must_use]
    pub fn with_padding(mut self, padding: [usize; 3]) -> Self {
        self.padding = padding;
        self
    }

    /// Set stride (uniform across all 3 dims).
    #[must_use]
    pub fn with_stride(mut self, stride: [usize; 3]) -> Self {
        self.stride = stride;
        self
    }

    /// Set dilation (uniform across all 3 dims).
    #[must_use]
    pub fn with_dilation(mut self, dilation: [usize; 3]) -> Self {
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

/// 3-D convolution layer.
///
/// Weight shape: `[out_channels, in_channels/groups, kD, kH, kW]`.
/// Bias shape: `[out_channels]` (if present).
///
/// Input shape: `[N, C_in, D, H, W]` (5D).
/// Output shape: `[N, C_out, D_out, H_out, W_out]`.
///
/// Used by Qwen3-VL and PaddleOCR-VL vision encoders for 3D patch embeddings.
#[derive(Clone, Debug)]
pub struct Conv3d {
    weight: DynTensor,
    bias: Option<DynTensor>,
    config: Conv3dConfig,
}

impl Conv3d {
    /// Create from pre-loaded weight and optional bias.
    ///
    /// - `weight`: shape `[out_channels, in_channels/groups, kD, kH, kW]` (must be 5D)
    /// - `bias`: shape `[out_channels]` (if present)
    /// - `config.groups` must be > 0
    ///
    /// Returns an error if `weight` is not 5D, `groups` is zero, or bias
    /// shape does not match `out_channels`.
    pub fn new(weight: DynTensor, bias: Option<DynTensor>, config: Conv3dConfig) -> Result<Self> {
        if weight.rank() != 5 {
            return Err(TensorError::RankMismatch {
                expected: 5,
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

    /// Reference to the weight tensor.
    #[must_use]
    pub fn weight(&self) -> &DynTensor {
        &self.weight
    }

    /// Reference to the optional bias tensor.
    #[must_use]
    pub fn bias(&self) -> Option<&DynTensor> {
        self.bias.as_ref()
    }

    /// Reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &Conv3dConfig {
        &self.config
    }

    fn conv3d_cpu(&self, x: &DynTensor) -> Result<DynTensor> {
        let y = x.conv3d(
            &self.weight,
            self.config.padding,
            self.config.stride,
            self.config.dilation,
            self.config.groups,
        )?;
        match &self.bias {
            Some(b) => {
                // bias shape [out_channels] -> [1, out_channels, 1, 1, 1] for broadcast
                let b = b.reshape([1, b.numel(), 1, 1, 1])?;
                y.broadcast_add(&b)
            }
            None => Ok(y),
        }
    }
}

impl Module for Conv3d {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        traced_forward(
            &[x],
            || {
                Ok(TraceOp::Conv3d {
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
                // GPU fused path: pass bias directly to GPU backend (1 dispatch).
                if x.device().is_gpu() && self.bias.is_some() {
                    if let Some(r) = gpu_backend_dispatch(|b| {
                        b.conv3d(
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
                self.conv3d_cpu(x)
            },
        )
    }
}
