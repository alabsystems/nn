// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PixelShuffle / PixelUnshuffle layers — sub-pixel convolution (Shi et al., 2016).
//!
//! Rearranges channels into spatial dimensions (and vice versa).
//! Parameter-free: no learnable weights, no VarBuilder needed.
//! Matches PyTorch `nn.PixelShuffle` / `nn.PixelUnshuffle`.

use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::DynTensor;
use crate::layers::Module;
use crate::{Result, TensorError};

/// Sub-pixel convolution layer: `[B, C*r², H, W] → [B, C, H*r, W*r]`.
///
/// Rearranges elements from the channel dimension into spatial dimensions.
/// Used by super-resolution models and vision encoders for spatial upsampling.
/// Matches PyTorch `nn.PixelShuffle(upscale_factor)`.
#[derive(Debug, Clone, Copy)]
pub struct PixelShuffle {
    upscale_factor: usize,
}

impl PixelShuffle {
    /// Create a new PixelShuffle layer with the given upscale factor.
    pub fn new(upscale_factor: usize) -> Result<Self> {
        if upscale_factor == 0 {
            return Err(TensorError::InvalidShape(
                "PixelShuffle: upscale_factor must be > 0".into(),
            ));
        }
        Ok(Self { upscale_factor })
    }

    /// The upscale factor.
    #[must_use]
    pub fn upscale_factor(&self) -> usize {
        self.upscale_factor
    }
}

impl Module for PixelShuffle {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let factor = self.upscale_factor;
        trace::traced_forward(
            &[x],
            || {
                Ok(TraceOp::PixelShuffle {
                    upscale_factor: factor,
                })
            },
            || x.pixel_shuffle(factor),
        )
    }
}

/// Inverse sub-pixel convolution layer: `[B, C, H*r, W*r] → [B, C*r², H, W]`.
///
/// Rearranges spatial elements into the channel dimension.
/// Inverse of [`PixelShuffle`]. Matches PyTorch `nn.PixelUnshuffle(downscale_factor)`.
#[derive(Debug, Clone, Copy)]
pub struct PixelUnshuffle {
    downscale_factor: usize,
}

impl PixelUnshuffle {
    /// Create a new PixelUnshuffle layer with the given downscale factor.
    pub fn new(downscale_factor: usize) -> Result<Self> {
        if downscale_factor == 0 {
            return Err(TensorError::InvalidShape(
                "PixelUnshuffle: downscale_factor must be > 0".into(),
            ));
        }
        Ok(Self { downscale_factor })
    }

    /// The downscale factor.
    #[must_use]
    pub fn downscale_factor(&self) -> usize {
        self.downscale_factor
    }
}

impl Module for PixelUnshuffle {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let factor = self.downscale_factor;
        trace::traced_forward(
            &[x],
            || {
                Ok(TraceOp::PixelUnshuffle {
                    downscale_factor: factor,
                })
            },
            || x.pixel_unshuffle(factor),
        )
    }
}

#[cfg(test)]
#[path = "pixel_shuffle_tests.rs"]
mod tests;
