// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Adaptive Instance Normalization (AdaIN) for [`DynTensor`].
//!
//! Style-conditioned affine after instance normalization.
//! Extracted from `instance_norm.rs` to keep files under 450 lines (#2920).

use super::instance_norm::{InstanceNorm, InstanceNormPrecision};
use super::{check_output_finite, Linear, Module};
use crate::dyn_tensor::trace::{KokoroFusedOp, TraceOp};
use crate::dyn_tensor::{gpu_backend_dispatch, DynTensor};
use crate::var_builder::VarBuilder;
use crate::Result;

/// Adaptive Instance Normalization (AdaIN).
///
/// `y = gamma(style) * InstanceNorm(x) + beta(style)`
///
/// Used in Kokoro TTS ISTFTNet decoder for style conditioning.
/// `gamma` and `beta` are projected from a style vector via a Linear layer.
#[derive(Debug, Clone)]
pub struct AdaIn {
    norm: InstanceNorm,
    style_linear: Linear,
}

impl AdaIn {
    /// Create AdaIN with style projection.
    ///
    /// Uses [`InstanceNormPrecision::F64`] accumulation by default.
    ///
    /// - `style_linear`: Linear layer projecting style `[B, style_dim]` to `[B, 2 * num_channels]`.
    ///   The output is split: first half is gamma, second half is beta.
    /// - `eps`: epsilon for InstanceNorm.
    pub fn new(style_linear: Linear, eps: f64) -> Result<Self> {
        Ok(Self {
            norm: InstanceNorm::new(eps)?,
            style_linear,
        })
    }

    /// Create AdaIN with style projection and specified precision mode.
    ///
    /// Use [`InstanceNormPrecision::MatchPyTorchCpu`] when parity with PyTorch
    /// CPU is the metric and the model chains 20+ InstanceNorm operations.
    pub fn new_with_precision(
        style_linear: Linear,
        eps: f64,
        precision: InstanceNormPrecision,
    ) -> Result<Self> {
        Ok(Self {
            norm: InstanceNorm::with_precision(eps, precision)?,
            style_linear,
        })
    }

    /// Weight of the style projection Linear layer.
    ///
    /// Returns the `[2 * channels, style_dim]` projection weight tensor.
    /// Used by cross-crate callers to build `WeightRef` for fused trace ops (#2459).
    pub fn style_weight(&self) -> &DynTensor {
        self.style_linear.weight()
    }

    /// Bias of the style projection Linear layer.
    ///
    /// Returns the `[2 * channels]` projection bias, or `None` if absent.
    pub fn style_bias(&self) -> Option<&DynTensor> {
        self.style_linear.bias()
    }

    /// Epsilon used for the internal InstanceNorm.
    pub fn eps(&self) -> f64 {
        self.norm.eps()
    }

    /// Load from a [`VarBuilder`] using PyTorch-style weight names.
    ///
    /// Uses [`InstanceNormPrecision::F64`] accumulation by default.
    ///
    /// Loads the `style_linear` sub-module projecting `[style_dim]` to `[2 * channels]`.
    /// Weight name: `style_linear.weight` plus optional `style_linear.bias`.
    ///
    /// - `style_dim`: Dimension of the style vector input.
    /// - `channels`: Number of channels in the normalized tensor.
    ///   The linear projects to `2 * channels` (gamma + beta).
    /// - `eps`: Epsilon for InstanceNorm.
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        style_dim: usize,
        channels: usize,
        eps: f64,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let style_linear = Linear::load(vb.pp("style_linear"), style_dim, 2 * channels)?;
        Self::new(style_linear, eps)
    }

    /// Load from a [`VarBuilder`] with specified precision mode.
    ///
    /// Same as [`load`](Self::load) but with configurable InstanceNorm precision.
    pub fn load_with_precision(
        vb: impl AsRef<VarBuilder>,
        style_dim: usize,
        channels: usize,
        eps: f64,
        precision: InstanceNormPrecision,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let style_linear = Linear::load(vb.pp("style_linear"), style_dim, 2 * channels)?;
        Self::new_with_precision(style_linear, eps, precision)
    }

    /// Forward: normalize `x` and apply style-conditioned affine.
    ///
    /// - `x`: `[B, C, T]`
    /// - `style`: `[B, style_dim]`
    pub fn forward_style(&self, x: &DynTensor, style: &DynTensor) -> Result<DynTensor> {
        // Wrap norm in traced_forward so normed gets a trace_node_id (#2370).
        // Without this, GPU forward_norm returns trace_node_id: None, breaking
        // downstream broadcast_mul/broadcast_add trace chain.
        let eps = self.norm.eps();
        let normed = super::traced_forward(
            &[x],
            || Ok(TraceOp::InstanceNorm { eps }),
            || self.norm.forward_norm(x),
        )?;

        // Project style to gamma and beta: [B, 2*C]
        let projected = self.style_linear.forward(style)?;
        let channels = x.dim(1)?;
        let gamma = projected.narrow(1, 0, channels)?;
        let beta = projected.narrow(1, channels, channels)?;

        // Reshape gamma/beta to [B, C, 1] for broadcasting over spatial dims
        let rank = x.rank();
        let mut affine_shape = vec![1usize; rank];
        affine_shape[0] = x.dim(0)?;
        affine_shape[1] = channels;
        let gamma = gamma.reshape(&affine_shape)?;
        let beta = beta.reshape(&affine_shape)?;

        // y = (1 + gamma) * normed + beta
        let ones = DynTensor::full(&affine_shape, 1.0, x.dtype(), &x.device())?;
        let scale = ones.broadcast_add(&gamma)?;
        let result = normed.broadcast_mul(&scale)?.broadcast_add(&beta)?;
        check_output_finite(&result, "AdaIn")?;
        Ok(result)
    }

    /// Fused forward: AdaIN + Snake in a single traced op.
    ///
    /// Equivalent to `forward_style(x, style)` followed by `snake_tensor(alpha)`,
    /// but records a single `TraceOp::AdainSnake` and fuses the computation to
    /// eliminate intermediate buffers.
    ///
    /// - `x`: `[B, C, T]`
    /// - `style`: `[B, style_dim]`
    /// - `alpha`: `[1, C, 1]` per-channel Snake parameter
    pub fn forward_snake(
        &self,
        x: &DynTensor,
        style: &DynTensor,
        alpha: &DynTensor,
    ) -> Result<DynTensor> {
        // Project style to gamma and beta: [B, 2*C]
        // These ops are recorded as their own trace nodes (Linear, Narrow, Reshape).
        let projected = self.style_linear.forward(style)?;
        let channels = x.dim(1)?;
        let gamma = projected.narrow(1, 0, channels)?;
        let beta = projected.narrow(1, channels, channels)?;

        // Reshape gamma/beta to [B, C, 1] for broadcasting over spatial dims
        let rank = x.rank();
        let mut affine_shape = vec![1usize; rank];
        affine_shape[0] = x.dim(0)?;
        affine_shape[1] = channels;
        let gamma = gamma.reshape(&affine_shape)?;
        let beta = beta.reshape(&affine_shape)?;

        // Fused: InstanceNorm(x) -> affine(gamma, beta) -> Snake(alpha)
        let eps = self.norm.eps();
        let alpha_ref = alpha.to_weight_ref()?;
        super::traced_forward(
            &[x, &gamma, &beta],
            || {
                Ok(TraceOp::KokoroFused(KokoroFusedOp::AdainSnake {
                    alpha: alpha_ref.clone(),
                    eps,
                }))
            },
            || {
                // GPU fused path: single dispatch for norm+affine+snake (#2227)
                if x.device().is_gpu() {
                    if let Some(result) =
                        gpu_backend_dispatch(|b| b.adain_snake(x, &gamma, &beta, alpha, eps))
                    {
                        let r = result?;
                        check_output_finite(&r, "AdainSnake")?;
                        return Ok(r);
                    }
                }

                // CPU/fallback: decomposed ops
                let normed = self.norm.forward_norm(x)?;

                // Affine: (1 + gamma) * normed + beta
                let ones = DynTensor::full(&affine_shape, 1.0, x.dtype(), &x.device())?;
                let scale = ones.broadcast_add(&gamma)?;
                let adain_out = normed.broadcast_mul(&scale)?.broadcast_add(&beta)?;

                // Snake: y + (1/alpha) * sin²(alpha * y)
                let result = adain_out.snake_tensor(alpha)?;
                check_output_finite(&result, "AdainSnake")?;
                Ok(result)
            },
        )
    }
    /// Fused forward: AdaIN + LeakyRelu in a single traced op.
    ///
    /// Equivalent to `forward_style(x, style)` followed by `leaky_relu(slope)`,
    /// but records a single `TraceOp::AdainLeakyRelu` and fuses the computation
    /// to eliminate intermediate buffers.
    ///
    /// - `x`: `[B, C, T]`
    /// - `style`: `[B, style_dim]`
    /// - `slope`: LeakyRelu negative slope (e.g. 0.2)
    pub fn forward_leaky_relu(
        &self,
        x: &DynTensor,
        style: &DynTensor,
        slope: f64,
    ) -> Result<DynTensor> {
        // Project style to gamma and beta: [B, 2*C]
        // These ops are recorded as their own trace nodes (Linear, Narrow, Reshape).
        let projected = self.style_linear.forward(style)?;
        let channels = x.dim(1)?;
        let gamma = projected.narrow(1, 0, channels)?;
        let beta = projected.narrow(1, channels, channels)?;

        // Reshape gamma/beta to [B, C, 1] for broadcasting over spatial dims
        let rank = x.rank();
        let mut affine_shape = vec![1usize; rank];
        affine_shape[0] = x.dim(0)?;
        affine_shape[1] = channels;
        let gamma = gamma.reshape(&affine_shape)?;
        let beta = beta.reshape(&affine_shape)?;

        // Fused: InstanceNorm(x) -> affine(gamma, beta) -> LeakyRelu(slope)
        let eps = self.norm.eps();
        super::traced_forward(
            &[x, &gamma, &beta],
            || {
                Ok(TraceOp::KokoroFused(KokoroFusedOp::AdainLeakyRelu {
                    eps,
                    slope,
                }))
            },
            || {
                // GPU fused path: single dispatch for norm+affine+leaky_relu (#2472)
                if x.device().is_gpu() {
                    if let Some(result) =
                        gpu_backend_dispatch(|b| b.adain_leaky_relu(x, &gamma, &beta, eps, slope))
                    {
                        let r = result?;
                        check_output_finite(&r, "AdainLeakyRelu")?;
                        return Ok(r);
                    }
                }

                // CPU/fallback: decomposed ops
                let normed = self.norm.forward_norm(x)?;

                // Affine: (1 + gamma) * normed + beta
                let ones = DynTensor::full(&affine_shape, 1.0, x.dtype(), &x.device())?;
                let scale = ones.broadcast_add(&gamma)?;
                let adain_out = normed.broadcast_mul(&scale)?.broadcast_add(&beta)?;

                // LeakyRelu
                let result = adain_out.leaky_relu(slope)?;
                check_output_finite(&result, "AdainLeakyRelu")?;
                Ok(result)
            },
        )
    }
}

#[cfg(test)]
#[path = "adain_tests.rs"]
mod tests;
