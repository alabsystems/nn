// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! 2D upsampling layers — nearest-neighbor and bilinear interpolation.
//!
//! Matches PyTorch's `nn.Upsample` / `F.interpolate` for 2D spatial data.
//! Parameter-free: no learnable weights, no VarBuilder needed.
//!
//! Two layer types:
//! - [`Upsample2d`] — scale-factor-based (e.g., 2x upscale).
//! - [`Upsample2dToSize`] — absolute output size (e.g., upsample to 16x16).
//!   Used by FPN/PAN when matching feature map spatial dimensions.

use crate::dyn_tensor::trace::{self, TraceOp, TraceUpsampleMode};
use crate::dyn_tensor::DynTensor;
use crate::layers::Module;
use crate::{Result, TensorError};

/// Upsampling interpolation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum UpsampleMode {
    /// Nearest-neighbor: each pixel is replicated in a `scale_h x scale_w` block.
    Nearest,
    /// Bilinear interpolation with configurable corner alignment.
    Bilinear { align_corners: bool },
}

/// 2D upsampling layer — nearest-neighbor or bilinear interpolation.
///
/// Input: `[B, C, H, W]` (or any shape with >= 2 dims; last two are spatial).
/// Output: `[B, C, H * scale_h, W * scale_w]` (nearest) or
///         `[B, C, round(H * scale_h), round(W * scale_w)]` (bilinear).
///
/// Parameter-free: no learnable weights.
/// Matches PyTorch `nn.Upsample(scale_factor=..., mode=...)`.
#[derive(Debug, Clone, Copy)]
pub struct Upsample2d {
    scale_h: f64,
    scale_w: f64,
    mode: UpsampleMode,
}

impl Upsample2d {
    /// Create a new upsampling layer.
    ///
    /// - `scale_h`, `scale_w`: scale factors for height and width dimensions.
    ///   For nearest mode, must be positive integers (truncated to usize).
    ///   For bilinear mode, can be any positive finite float.
    /// - `mode`: interpolation mode.
    pub fn new(scale_h: f64, scale_w: f64, mode: UpsampleMode) -> Result<Self> {
        if !scale_h.is_finite() || !scale_w.is_finite() || scale_h <= 0.0 || scale_w <= 0.0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Upsample2d: scale factors must be finite and > 0",
            });
        }
        // Prevent OOM from huge scale factors (e.g., 1e18 as usize).
        const MAX_SCALE: f64 = 65536.0;
        if scale_h > MAX_SCALE || scale_w > MAX_SCALE {
            return Err(TensorError::ValueOutOfRange {
                description: "Upsample2d: scale factors exceed maximum 65536",
            });
        }
        if matches!(mode, UpsampleMode::Nearest) {
            let sh = scale_h as usize;
            let sw = scale_w as usize;
            if sh == 0 || sw == 0 {
                return Err(TensorError::ValueOutOfRange {
                    description: "Upsample2d: nearest mode requires integer scale factors >= 1",
                });
            }
        }
        Ok(Self {
            scale_h,
            scale_w,
            mode,
        })
    }

    /// Scale factor for the height dimension.
    #[must_use]
    pub fn scale_h(&self) -> f64 {
        self.scale_h
    }

    /// Scale factor for the width dimension.
    #[must_use]
    pub fn scale_w(&self) -> f64 {
        self.scale_w
    }

    /// Interpolation mode.
    #[must_use]
    pub fn mode(&self) -> UpsampleMode {
        self.mode
    }
}

impl Module for Upsample2d {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let scale_h = self.scale_h;
        let scale_w = self.scale_w;
        let mode = self.mode;
        trace::traced_forward(
            &[x],
            || {
                let trace_mode = match mode {
                    UpsampleMode::Nearest => TraceUpsampleMode::Nearest,
                    UpsampleMode::Bilinear { .. } => TraceUpsampleMode::Bilinear,
                };
                Ok(TraceOp::Upsample2d {
                    mode: trace_mode,
                    scale_h,
                    scale_w,
                })
            },
            || match mode {
                UpsampleMode::Nearest => x.upsample_nearest_2d(scale_h as usize, scale_w as usize),
                UpsampleMode::Bilinear { align_corners } => {
                    x.upsample_bilinear_2d(scale_h, scale_w, align_corners)
                }
            },
        )
    }
}

/// 2D upsampling layer to explicit output dimensions — bilinear interpolation.
///
/// Input: `[B, C, H, W]` (or any shape with >= 2 dims; last two are spatial).
/// Output: `[B, C, out_h, out_w]`.
///
/// Used by FPN/PAN necks in detection models where the target spatial size
/// is determined by the feature map to be concatenated with, not a fixed
/// scale factor. Matches PyTorch `F.interpolate(size=(...), mode='bilinear', ...)`.
///
/// Parameter-free: no learnable weights.
#[derive(Debug, Clone, Copy)]
pub struct Upsample2dToSize {
    out_h: usize,
    out_w: usize,
    align_corners: bool,
}

impl Upsample2dToSize {
    /// Create a bilinear upsample layer with fixed output dimensions.
    ///
    /// - `out_h`, `out_w`: target spatial dimensions.
    /// - `align_corners`: corner pixel alignment (see `upsample_bilinear_2d_to_size`).
    pub fn new(out_h: usize, out_w: usize, align_corners: bool) -> Result<Self> {
        if out_h == 0 || out_w == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Upsample2dToSize: output dimensions must be > 0",
            });
        }
        Ok(Self {
            out_h,
            out_w,
            align_corners,
        })
    }

    /// Target output height.
    #[must_use]
    pub fn out_h(&self) -> usize {
        self.out_h
    }

    /// Target output width.
    #[must_use]
    pub fn out_w(&self) -> usize {
        self.out_w
    }

    /// Whether corners are aligned.
    #[must_use]
    pub fn align_corners(&self) -> bool {
        self.align_corners
    }
}

impl Module for Upsample2dToSize {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        x.upsample_bilinear_2d_to_size(self.out_h, self.out_w, self.align_corners)
    }
}

#[cfg(test)]
#[path = "upsample_tests.rs"]
mod tests;
