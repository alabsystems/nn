// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Backward rules for normalization operations: RmsNorm, GroupNorm, BatchNorm, InstanceNorm.
//!
//! Extracted from backward_rules_special.rs to keep that file under 350 lines.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::tensor::checked_dim_product;

use crate::error::Result;
use crate::grad::GradStore;
use crate::tracked::TrackedTensor;

use super::accumulate;

/// Delegate to the shared reshape helper in backward_rules.rs.
/// Reshape `[C]` → `[1, C, 1, 1, ...]` for left-aligned broadcast.
fn reshape_channel(
    t: &DynTensor,
    rank: usize,
) -> std::result::Result<DynTensor, nn_core::TensorError> {
    crate::backward_rules::reshape_for_channel_broadcast(t, rank)
}

/// Backward rule for RMS normalization.
///
/// RMSNorm: y = x / rms(x) * weight, where rms(x) = sqrt(mean(x^2) + eps)
/// Gradients:
///   grad_weight = sum(grad * normed, all dims except last)
///   grad_input = weight/rms * (grad - normed * mean(grad * normed, last_dim))
pub(super) fn backward_rms_norm(
    input: &Arc<TrackedTensor>,
    weight: &Arc<TrackedTensor>,
    eps: f64,
    grad: &DynTensor,
    grads: &mut GradStore,
) -> Result<()> {
    let x = input.tensor();
    if x.rank() < 1 {
        return Err(crate::AutodiffError::InvalidConfig {
            op: "backward_rms_norm",
            reason: format!("input rank {} < minimum 1", x.rank()),
        });
    }
    let gamma = weight.tensor();
    let last_dim = x.rank() - 1;

    // Recompute forward intermediates
    let rms_sq = x.sqr()?.mean_keepdim(last_dim)?;
    let inv_rms = rms_sq.add_scalar(eps)?.sqrt()?.recip()?;
    let normed = x.mul(&inv_rms)?;

    // grad_weight = sum(grad * normed) over all dims except last
    let grad_weight = sum_all_but_last(&grad.mul(&normed)?)?;
    accumulate(weight, &grad_weight, grads)?;

    // grad_input = inv_rms * weight * (grad - normed * mean(grad * normed * weight, last_dim))
    let grad_normed = grad.mul(gamma)?;
    let proj = grad_normed.mul(&normed)?.mean_keepdim(last_dim)?;
    let grad_input = inv_rms.mul(&grad_normed.sub(&normed.mul(&proj.expand(grad.dims())?)?)?)?;
    accumulate(input, &grad_input, grads)?;
    Ok(())
}

/// Backward rule for group normalization.
///
/// Input: [N, C, *]. Groups C channels into G groups of C/G.
/// Same structure as LayerNorm but over group-shaped dimensions.
pub(super) fn backward_group_norm(
    input: &Arc<TrackedTensor>,
    weight: &Arc<TrackedTensor>,
    bias: &Arc<TrackedTensor>,
    num_groups: usize,
    eps: f64,
    grad: &DynTensor,
    grads: &mut GradStore,
) -> Result<()> {
    let x = input.tensor();
    if x.rank() < 2 {
        return Err(crate::AutodiffError::InvalidConfig {
            op: "backward_group_norm",
            reason: format!("input rank {} < minimum 2", x.rank()),
        });
    }
    let dims = x.dims().to_vec();
    let c = dims[1];
    // Defense-in-depth: forward path validates num_groups > 0, but
    // guard here to prevent division-by-zero panic if reached directly.
    if num_groups == 0 || !c.is_multiple_of(num_groups) {
        return Err(crate::AutodiffError::InvalidConfig {
            op: "group_norm backward",
            reason: format!("invalid num_groups={num_groups} for channels={c}"),
        });
    }
    let channels_per_group = c / num_groups;

    // Reshape to [N, G, C/G, *spatial]
    let mut grouped = vec![dims[0], num_groups, channels_per_group];
    grouped.extend_from_slice(&dims[2..]);
    let xr = x.reshape(&grouped)?;
    let gr = grad.reshape(&grouped)?;

    // Compute forward intermediates in grouped space
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
    let normed_grouped = diff.mul(&inv_std.expand(xr.dims())?)?;
    let normed_flat = normed_grouped.reshape(&dims)?;

    // grad_bias = sum(grad) over all dims except C (dim 1)
    let grad_bias = sum_all_except_dim1(grad)?;
    accumulate(bias, &grad_bias, grads)?;

    // grad_weight = sum(grad * normed) over all dims except C
    let grad_weight = sum_all_except_dim1(&grad.mul(&normed_flat)?)?;
    accumulate(weight, &grad_weight, grads)?;

    // grad_input: apply weight [C] → [1,C,1,...] in flat [N,C,*] space, reshape to grouped
    let w_bc = reshape_channel(weight.tensor(), grad.rank())?;
    let grad_gamma_flat = grad.mul(&w_bc)?;
    let grad_gamma = grad_gamma_flat.reshape(&grouped)?;
    let mean_gg = {
        let mut m = grad_gamma.clone();
        for d in (2..grouped.len()).rev() {
            m = m.mean_keepdim(d)?;
        }
        m
    };
    let mean_gg_norm = {
        let mut m = grad_gamma.mul(&normed_grouped)?;
        for d in (2..grouped.len()).rev() {
            m = m.mean_keepdim(d)?;
        }
        m
    };
    let gi = inv_std.expand(xr.dims())?.mul(
        &grad_gamma
            .sub(&mean_gg.expand(gr.dims())?)?
            .sub(&normed_grouped.mul(&mean_gg_norm.expand(gr.dims())?)?)?,
    )?;
    let grad_input = gi.reshape(&dims)?;
    accumulate(input, &grad_input, grads)?;
    Ok(())
}

/// Backward rule for batch normalization (training mode).
///
/// Input: [N, C, *]. Normalizes over batch (dim 0) and spatial dims (2..).
pub(super) fn backward_batch_norm(
    input: &Arc<TrackedTensor>,
    weight: &Arc<TrackedTensor>,
    bias: &Arc<TrackedTensor>,
    eps: f64,
    grad: &DynTensor,
    grads: &mut GradStore,
) -> Result<()> {
    let x = input.tensor();
    if x.rank() < 2 {
        return Err(crate::AutodiffError::InvalidConfig {
            op: "backward_batch_norm",
            reason: format!("input rank {} < minimum 2", x.rank()),
        });
    }

    // Compute forward intermediates: mean/var over all dims except 1
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

    // grad_bias = sum(grad) over all dims except C (dim 1)
    let grad_bias = sum_all_except_dim1(grad)?;
    accumulate(bias, &grad_bias, grads)?;

    // grad_weight = sum(grad * normed) over all dims except C
    let grad_weight = sum_all_except_dim1(&grad.mul(&normed)?)?;
    accumulate(weight, &grad_weight, grads)?;

    // grad_input: standard batch norm backward
    // Apply weight [C] → [1,C,1,...] for left-aligned broadcast
    let w_bc = reshape_channel(weight.tensor(), grad.rank())?;
    let grad_gamma = grad.mul(&w_bc)?;
    // mean over all dims except 1 (keepdim to preserve rank for expand)
    let mean_gg = mean_keepdim_except_dim1(&grad_gamma)?;
    let mean_gg_norm = mean_keepdim_except_dim1(&grad_gamma.mul(&normed)?)?;
    let grad_input = inv_std.expand(x.dims())?.mul(
        &grad_gamma
            .sub(&mean_gg.expand(grad.dims())?)?
            .sub(&normed.mul(&mean_gg_norm.expand(grad.dims())?)?)?,
    )?;
    accumulate(input, &grad_input, grads)?;
    Ok(())
}

/// Backward rule for instance normalization.
///
/// Input: [N, C, *]. Normalizes over spatial dims (2..) per (N, C) independently.
pub(super) fn backward_instance_norm(
    input: &Arc<TrackedTensor>,
    weight: &Arc<TrackedTensor>,
    bias: &Arc<TrackedTensor>,
    eps: f64,
    grad: &DynTensor,
    grads: &mut GradStore,
) -> Result<()> {
    let x = input.tensor();
    if x.rank() < 3 {
        return Err(crate::AutodiffError::InvalidConfig {
            op: "backward_instance_norm",
            reason: format!("input rank {} < minimum 3", x.rank()),
        });
    }

    // Compute forward intermediates: mean/var over spatial dims (2..)
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

    // grad_bias = sum(grad) over all dims except C (dim 1)
    let grad_bias = sum_all_except_dim1(grad)?;
    accumulate(bias, &grad_bias, grads)?;

    // grad_weight = sum(grad * normed) over all dims except C
    let grad_weight = sum_all_except_dim1(&grad.mul(&normed)?)?;
    accumulate(weight, &grad_weight, grads)?;

    // grad_input: normalize backward over spatial dims per (N,C)
    // Apply weight [C] → [1,C,1,...] for left-aligned broadcast
    let w_bc = reshape_channel(weight.tensor(), grad.rank())?;
    let grad_gamma = grad.mul(&w_bc)?;
    let mut mean_gg = grad_gamma.clone();
    for d in (2..x.rank()).rev() {
        mean_gg = mean_gg.mean_keepdim(d)?;
    }
    let mut mean_gg_norm = grad_gamma.mul(&normed)?;
    for d in (2..x.rank()).rev() {
        mean_gg_norm = mean_gg_norm.mean_keepdim(d)?;
    }
    let grad_input = inv_std.expand(x.dims())?.mul(
        &grad_gamma
            .sub(&mean_gg.expand(grad.dims())?)?
            .sub(&normed.mul(&mean_gg_norm.expand(grad.dims())?)?)?,
    )?;
    accumulate(input, &grad_input, grads)?;
    Ok(())
}

/// Sum a tensor over all dimensions except the last.
///
/// Uses reshape to flatten leading dims into one, then a single sum_keepdim+squeeze
/// instead of (rank-1) iterations × 2 intermediate allocations each.
pub(super) fn sum_all_but_last(t: &DynTensor) -> Result<DynTensor> {
    let rank = t.rank();
    if rank <= 1 {
        return Ok(t.clone());
    }
    // Reshape [D0, D1, ..., Dk] → [D0*D1*...*D(k-1), Dk], sum dim 0, squeeze.
    let last_size = t.dims()[rank - 1];
    let leading = checked_dim_product(&t.dims()[..rank - 1])?;
    Ok(t.reshape([leading, last_size])?
        .sum_keepdim(0)?
        .squeeze(0)?)
}

/// Sum a tensor over all dimensions except dim 1 (channels).
///
/// Reshapes to [N*spatial_product, C] then sums dim 0, avoiding (rank-1)
/// iterations × 2 intermediate allocations each.
pub(super) fn sum_all_except_dim1(t: &DynTensor) -> Result<DynTensor> {
    let rank = t.rank();
    if rank < 2 {
        return Ok(t.clone());
    }
    let c = t.dims()[1];
    // Reshape [N, C, *spatial] → transposed [C, N*spatial_product] via
    // reshape [N, C, flat] → transpose(0,1) → reshape [C, N*flat] → sum(1) → squeeze(1)
    let n = t.dims()[0];
    let spatial = if rank > 2 {
        checked_dim_product(&t.dims()[2..])?
    } else {
        1
    };
    let n_spatial =
        n.checked_mul(spatial)
            .ok_or_else(|| nn_core::TensorError::DimensionOverflow {
                dims: t.dims().to_vec(),
            })?;
    let flat = t.reshape([n, c, spatial])?.transpose(0, 1)?.contiguous()?;
    Ok(flat.reshape([c, n_spatial])?.sum_keepdim(1)?.squeeze(1)?)
}

/// Mean over all dims except dim 1, preserving rank (keepdim).
/// Produces shape `[1, C, 1, 1, ...]` from `[N, C, *spatial]`.
///
/// Reshapes to [C, N*spatial] for a single mean, then reshapes back to
/// [1, C, 1, 1, ...] instead of (rank-1) iterations × mean_keepdim calls.
fn mean_keepdim_except_dim1(t: &DynTensor) -> Result<DynTensor> {
    let rank = t.rank();
    if rank < 2 {
        return Ok(t.clone());
    }
    let c = t.dims()[1];
    let n = t.dims()[0];
    let spatial = if rank > 2 {
        checked_dim_product(&t.dims()[2..])?
    } else {
        1
    };
    let n_spatial =
        n.checked_mul(spatial)
            .ok_or_else(|| nn_core::TensorError::DimensionOverflow {
                dims: t.dims().to_vec(),
            })?;
    // [N, C, *spatial] → [N, C, flat] → transpose(0,1) → [C, N, flat] → [C, N*flat]
    let flat = t.reshape([n, c, spatial])?.transpose(0, 1)?.contiguous()?;
    // mean over dim 1 → [C, 1] → squeeze → [C]
    let mean_c = flat.reshape([c, n_spatial])?.mean_keepdim(1)?.squeeze(1)?;
    // Reshape back to [1, C, 1, 1, ...]
    let mut out_shape = vec![1usize; rank];
    out_shape[1] = c;
    Ok(mean_c.reshape(&out_shape)?)
}
