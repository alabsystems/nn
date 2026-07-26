// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pooling nn layers: [`MaxPool1d`], [`MaxPool2d`], [`AvgPool2d`], [`AdaptiveAvgPool2d`].
//!
//! Parameter-free layers that match PyTorch/candle-nn API. These are Tier 2
//! layers (pure linear/elementwise on finite inputs — no per-layer finiteness check).

use super::Module;
use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::DynTensor;
use crate::error::{Result, TensorError};

/// Configuration for [`MaxPool1d`].
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Pool1dConfig {
    /// Pooling window size.
    pub kernel_size: usize,
    /// Stride between pooling windows. Defaults to `kernel_size`.
    pub stride: usize,
    /// Zero-padding on each side. Defaults to 0.
    pub padding: usize,
}

impl Pool1dConfig {
    /// Create config with stride defaulting to kernel_size.
    #[must_use]
    pub fn new(kernel_size: usize) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
            padding: 0,
        }
    }

    /// Set stride.
    #[must_use]
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Set padding.
    #[must_use]
    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    fn validate(&self, layer: &str) -> Result<()> {
        if self.kernel_size == 0 {
            return Err(TensorError::InvalidShape(format!(
                "{layer}: kernel_size must be > 0"
            )));
        }
        if self.stride == 0 {
            return Err(TensorError::InvalidShape(format!(
                "{layer}: stride must be > 0"
            )));
        }
        Ok(())
    }
}

/// 1-D max pooling layer.
///
/// Input shape: `[batch, channels, length]`
/// Output shape: `[batch, channels, out_length]`
///
/// Parameter-free (no learned weights).
#[derive(Clone, Debug)]
pub struct MaxPool1d {
    config: Pool1dConfig,
}

impl MaxPool1d {
    pub fn new(config: Pool1dConfig) -> Result<Self> {
        config.validate("MaxPool1d")?;
        Ok(Self { config })
    }

    #[must_use]
    pub fn config(&self) -> &Pool1dConfig {
        &self.config
    }
}

impl Module for MaxPool1d {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let k = self.config.kernel_size;
        let s = self.config.stride;
        let p = self.config.padding;
        trace::traced_forward(
            &[x],
            || {
                Ok(TraceOp::MaxPool1d {
                    kernel_size: k,
                    stride: s,
                    padding: p,
                })
            },
            || x.max_pool1d(k, s, p),
        )
    }
}

/// Configuration for [`MaxPool2d`] and [`AvgPool2d`].
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Pool2dConfig {
    /// Pooling window size (square).
    pub kernel_size: usize,
    /// Stride between pooling windows. Defaults to `kernel_size`.
    pub stride: usize,
    /// Zero-padding on each side. Defaults to 0.
    pub padding: usize,
}

impl Pool2dConfig {
    /// Create config with stride defaulting to kernel_size.
    #[must_use]
    pub fn new(kernel_size: usize) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
            padding: 0,
        }
    }

    /// Set stride.
    #[must_use]
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Set padding.
    #[must_use]
    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    fn validate(&self, layer: &str) -> Result<()> {
        if self.kernel_size == 0 {
            return Err(TensorError::InvalidShape(format!(
                "{layer}: kernel_size must be > 0"
            )));
        }
        if self.stride == 0 {
            return Err(TensorError::InvalidShape(format!(
                "{layer}: stride must be > 0"
            )));
        }
        Ok(())
    }
}

/// 2-D max pooling layer.
///
/// Input shape: `[batch, channels, height, width]`
/// Output shape: `[batch, channels, out_h, out_w]`
///
/// Parameter-free (no learned weights).
#[derive(Clone, Debug)]
pub struct MaxPool2d {
    config: Pool2dConfig,
}

impl MaxPool2d {
    pub fn new(config: Pool2dConfig) -> Result<Self> {
        config.validate("MaxPool2d")?;
        Ok(Self { config })
    }

    #[must_use]
    pub fn config(&self) -> &Pool2dConfig {
        &self.config
    }
}

impl Module for MaxPool2d {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let k = self.config.kernel_size;
        let s = self.config.stride;
        let p = self.config.padding;
        trace::traced_forward(
            &[x],
            || {
                Ok(TraceOp::MaxPool2d {
                    kernel_size: [k, k],
                    stride: [s, s],
                    padding: [p, p],
                })
            },
            || x.max_pool2d(k, s, p),
        )
    }
}

/// 2-D average pooling layer.
///
/// Input shape: `[batch, channels, height, width]`
/// Output shape: `[batch, channels, out_h, out_w]`
///
/// Parameter-free. Padding positions excluded from averaging count
/// (count_include_pad=false, matching PyTorch default).
#[derive(Clone, Debug)]
pub struct AvgPool2d {
    config: Pool2dConfig,
}

impl AvgPool2d {
    pub fn new(config: Pool2dConfig) -> Result<Self> {
        config.validate("AvgPool2d")?;
        Ok(Self { config })
    }

    #[must_use]
    pub fn config(&self) -> &Pool2dConfig {
        &self.config
    }
}

impl Module for AvgPool2d {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let k = self.config.kernel_size;
        let s = self.config.stride;
        let p = self.config.padding;
        trace::traced_forward(
            &[x],
            || {
                Ok(TraceOp::AvgPool2d {
                    kernel_size: [k, k],
                    stride: [s, s],
                    padding: [p, p],
                })
            },
            || x.avg_pool2d(k, s, p),
        )
    }
}

/// Adaptive 2-D average pooling layer.
///
/// Input shape: `[batch, channels, height, width]`
/// Output shape: `[batch, channels, out_h, out_w]`
///
/// Automatically selects kernel/stride to produce the target output dimensions.
/// Parameter-free.
#[derive(Clone, Debug)]
pub struct AdaptiveAvgPool2d {
    out_h: usize,
    out_w: usize,
}

impl AdaptiveAvgPool2d {
    pub fn new(out_h: usize, out_w: usize) -> Result<Self> {
        if out_h == 0 || out_w == 0 {
            return Err(TensorError::InvalidShape(
                "AdaptiveAvgPool2d: output dimensions must be > 0".into(),
            ));
        }
        Ok(Self { out_h, out_w })
    }

    #[must_use]
    pub fn output_size(&self) -> (usize, usize) {
        (self.out_h, self.out_w)
    }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let out_h = self.out_h;
        let out_w = self.out_w;
        trace::traced_forward(
            &[x],
            || {
                Ok(TraceOp::AdaptiveAvgPool2d {
                    output_size: [out_h, out_w],
                })
            },
            || x.adaptive_avg_pool2d(out_h, out_w),
        )
    }
}

#[cfg(kani)]
#[path = "kani_pool_proofs.rs"]
mod kani_pool_proofs;

#[cfg(test)]
#[path = "pool_tests.rs"]
mod tests;
