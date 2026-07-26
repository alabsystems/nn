// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Normalization-family tracked composite operations: rms_norm, group_norm,
//! batch_norm, instance_norm.
//!
//! Extracted from `tracked_composite_ops.rs` for 500-line compliance.

use super::TrackedTensor;
use crate::error::Result;
use crate::op::Op;
use std::sync::Arc;

/// Delegate to the shared reshape helper in backward_rules.rs.
/// Reshape `[C]` → `[1, C, 1, 1, ...]` for left-aligned broadcast.
fn reshape_for_channel_broadcast(
    t: &nn_core::dyn_tensor::DynTensor,
    target_rank: usize,
) -> std::result::Result<nn_core::dyn_tensor::DynTensor, nn_core::TensorError> {
    crate::backward_rules::reshape_for_channel_broadcast(t, target_rank)
}

impl TrackedTensor {
    /// RMS normalization: `x / rms(x) * weight`, where `rms(x) = sqrt(mean(x^2) + eps)`.
    ///
    /// Used by Qwen3, LLaMA, and most modern LLMs. Unlike LayerNorm, RMSNorm
    /// has no bias term and normalizes by root-mean-square instead of variance.
    pub fn rms_norm(self: &Arc<Self>, weight: &Arc<Self>, eps: f64) -> Result<Arc<Self>> {
        let x = &self.data;
        if x.rank() < 1 {
            return Err(crate::AutodiffError::InvalidConfig {
                op: "rms_norm",
                reason: format!("input rank {} < minimum 1", x.rank()),
            });
        }
        let last_dim = x.rank() - 1;
        let rms_sq = x.sqr()?.mean_keepdim(last_dim)?;
        let inv_rms = rms_sq.add_scalar(eps)?.sqrt()?.recip()?;
        let normed = x.mul(&inv_rms)?;
        let data = normed.mul(weight.tensor())?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::RmsNorm {
                input: Arc::clone(self),
                weight: Arc::clone(weight),
                eps,
            },
        )))
    }

    /// Group normalization.
    ///
    /// Input: `[N, C, *]`. Divides C channels into `num_groups` groups and
    /// normalizes within each group. Used by Demucs, vision models, diffusion.
    pub fn group_norm(
        self: &Arc<Self>,
        weight: &Arc<Self>,
        bias: &Arc<Self>,
        num_groups: usize,
        eps: f64,
    ) -> Result<Arc<Self>> {
        // Decomposed: reshape [N,C,*] → [N,G,C/G,*], normalize over dims 2..,
        // reshape back, apply weight and bias.
        let x = &self.data;
        if x.rank() < 2 {
            return Err(crate::AutodiffError::InvalidConfig {
                op: "group_norm",
                reason: format!("input rank {} < minimum 2", x.rank()),
            });
        }
        let dims = x.dims().to_vec();
        if num_groups == 0 {
            return Err(crate::AutodiffError::InvalidConfig {
                op: "group_norm",
                reason: "num_groups must be > 0".into(),
            });
        }
        let n = dims[0];
        let c = dims[1];
        if !c.is_multiple_of(num_groups) {
            return Err(crate::AutodiffError::InvalidConfig {
                op: "group_norm",
                reason: format!("channels {c} not divisible by num_groups {num_groups}"),
            });
        }
        let channels_per_group = c / num_groups;
        // Reshape to [N, G, C/G, *spatial]
        let mut grouped = vec![n, num_groups, channels_per_group];
        grouped.extend_from_slice(&dims[2..]);
        let xr = x.reshape(&grouped)?;
        // Compute mean and variance over dims 2.. (C/G and spatial)
        let mut mean = xr.clone();
        for d in (2..grouped.len()).rev() {
            mean = mean.mean_keepdim(d)?;
        }
        let diff = xr.sub(&mean.expand(xr.dims())?)?;
        let mut var = diff.sqr()?;
        for d in (2..grouped.len()).rev() {
            var = var.mean_keepdim(d)?;
        }
        let inv_std = var.add_scalar(eps)?.sqrt()?.recip()?;
        let normed = diff.mul(&inv_std.expand(xr.dims())?)?;
        // Reshape back to [N, C, *]
        let normed_flat = normed.reshape(&dims)?;
        // Apply affine: weight [C] → [1, C, 1, ...] for left-aligned broadcast
        let w = reshape_for_channel_broadcast(weight.tensor(), x.rank())?;
        let b = reshape_for_channel_broadcast(bias.tensor(), x.rank())?;
        let data = normed_flat.mul(&w)?.add(&b)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::GroupNorm {
                input: Arc::clone(self),
                weight: Arc::clone(weight),
                bias: Arc::clone(bias),
                num_groups,
                eps,
            },
        )))
    }

    /// Batch normalization (training mode).
    ///
    /// Input: `[N, C, *]`. Computes mean and variance over batch and spatial dims.
    pub fn batch_norm(
        self: &Arc<Self>,
        weight: &Arc<Self>,
        bias: &Arc<Self>,
        eps: f64,
    ) -> Result<Arc<Self>> {
        // Decomposed: mean/var over all dims except dim=1 (channels).
        let x = &self.data;
        if x.rank() < 2 {
            return Err(crate::AutodiffError::InvalidConfig {
                op: "batch_norm",
                reason: format!("input rank {} < minimum 2", x.rank()),
            });
        }
        // Mean over batch (dim 0) and spatial (dims 2..)
        let mut mean = x.clone();
        for d in (0..x.rank()).rev() {
            if d != 1 {
                mean = mean.mean_keepdim(d)?;
            }
        }
        let diff = x.sub(&mean.expand(x.dims())?)?;
        let mut var = diff.sqr()?;
        for d in (0..x.rank()).rev() {
            if d != 1 {
                var = var.mean_keepdim(d)?;
            }
        }
        let inv_std = var.add_scalar(eps)?.sqrt()?.recip()?;
        let normed = diff.mul(&inv_std.expand(x.dims())?)?;
        // Apply affine: weight [C] → [1, C, 1, ...] for left-aligned broadcast
        let w = reshape_for_channel_broadcast(weight.tensor(), x.rank())?;
        let b = reshape_for_channel_broadcast(bias.tensor(), x.rank())?;
        let data = normed.mul(&w)?.add(&b)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::BatchNorm {
                input: Arc::clone(self),
                weight: Arc::clone(weight),
                bias: Arc::clone(bias),
                eps,
            },
        )))
    }

    /// Instance normalization.
    ///
    /// Input: `[N, C, *]`. Normalizes each (N, C) slice independently.
    pub fn instance_norm(
        self: &Arc<Self>,
        weight: &Arc<Self>,
        bias: &Arc<Self>,
        eps: f64,
    ) -> Result<Arc<Self>> {
        // Decomposed: mean/var over spatial dims (2..) per (N, C).
        let x = &self.data;
        if x.rank() < 3 {
            return Err(crate::AutodiffError::InvalidConfig {
                op: "instance_norm",
                reason: format!("input rank {} < minimum 3", x.rank()),
            });
        }
        let mut mean = x.clone();
        for d in (2..x.rank()).rev() {
            mean = mean.mean_keepdim(d)?;
        }
        let diff = x.sub(&mean.expand(x.dims())?)?;
        let mut var = diff.sqr()?;
        for d in (2..x.rank()).rev() {
            var = var.mean_keepdim(d)?;
        }
        let inv_std = var.add_scalar(eps)?.sqrt()?.recip()?;
        let normed = diff.mul(&inv_std.expand(x.dims())?)?;
        // Apply affine: weight [C] → [1, C, 1, ...] for left-aligned broadcast
        let w = reshape_for_channel_broadcast(weight.tensor(), x.rank())?;
        let b = reshape_for_channel_broadcast(bias.tensor(), x.rank())?;
        let data = normed.mul(&w)?.add(&b)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::InstanceNorm {
                input: Arc::clone(self),
                weight: Arc::clone(weight),
                bias: Arc::clone(bias),
                eps,
            },
        )))
    }
}
