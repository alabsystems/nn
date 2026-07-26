// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Backward rules for composite operations: LayerNorm, Embedding, CrossEntropy.
//!
//! Extracted from backward_rules.rs to keep that file under 350 lines.
//! Normalization rules (RmsNorm, GroupNorm, BatchNorm, InstanceNorm) are in
//! `backward_rules_norm.rs`.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::DType;

use crate::error::Result;
use crate::grad::GradStore;
use crate::tracked::TrackedTensor;

use crate::op::Op;

/// Extract the scalar f64 value from a 0-d or 1-element gradient tensor.
///
/// Uses `to_scalar::<f32>()` which transfers only a single element from GPU,
/// avoiding `as_cpu_f32()` which would transfer the entire tensor.
fn scalar_grad_val(grad: &DynTensor) -> Result<f64> {
    Ok(f64::from(grad.to_scalar::<f32>()?))
}

use super::accumulate;

#[path = "backward_rules_norm.rs"]
mod norm;

/// Dispatch backward rule for composite/special operations.
pub(super) fn backward_special(op: &Op, grad: &DynTensor, grads: &mut GradStore) -> Result<()> {
    match op {
        Op::Softmax(x, dim) => {
            let s = x.tensor().softmax(*dim)?;
            let dot = grad.mul(&s)?.sum_keepdim(*dim)?;
            let grad_input = s.mul(&grad.sub(&dot.expand(grad.dims())?)?)?;
            accumulate(x, &grad_input, grads)
        }
        Op::LayerNorm {
            input,
            weight,
            bias,
            eps,
            normalized_shape,
        } => backward_layer_norm(input, weight, bias, *eps, *normalized_shape, grad, grads),
        Op::Embedding(weight, indices) => backward_embedding(weight, indices, grad, grads),
        Op::CrossEntropyLoss(input, targets, dim) => {
            backward_cross_entropy(input, targets, *dim, grad, grads)
        }
        Op::MseLoss(input, target) => backward_mse(input, target, grad, grads),
        Op::L1Loss(input, target) => backward_l1(input, target, grad, grads),
        Op::HuberLoss(input, target, delta) => backward_huber(input, target, *delta, grad, grads),
        Op::RmsNorm { input, weight, eps } => {
            norm::backward_rms_norm(input, weight, *eps, grad, grads)
        }
        Op::GroupNorm {
            input,
            weight,
            bias,
            num_groups,
            eps,
        } => norm::backward_group_norm(input, weight, bias, *num_groups, *eps, grad, grads),
        Op::BatchNorm {
            input,
            weight,
            bias,
            eps,
        } => norm::backward_batch_norm(input, weight, bias, *eps, grad, grads),
        Op::InstanceNorm {
            input,
            weight,
            bias,
            eps,
        } => norm::backward_instance_norm(input, weight, bias, *eps, grad, grads),
        other => Err(super::unsupported(other)),
    }
}

/// Backward rule for layer normalization.
///
/// Computes gradients for input, weight (gamma), and bias (beta).
/// Uses the standard LayerNorm backward formula:
///   grad_bias = sum(grad, all dims except last)
///   grad_weight = sum(grad * normalized_input, all dims except last)
///   grad_input = (1/std) * (grad*gamma - mean(grad*gamma) - norm*(grad*gamma*norm).mean())
pub(super) fn backward_layer_norm(
    input: &Arc<TrackedTensor>,
    weight: &Arc<TrackedTensor>,
    bias: &Arc<TrackedTensor>,
    eps: f64,
    _normalized_shape: usize,
    grad: &DynTensor,
    grads: &mut GradStore,
) -> Result<()> {
    let x = input.tensor();
    let gamma = weight.tensor();
    let last_dim = x.rank() - 1;

    // Recompute forward intermediates
    let mean = x.mean_keepdim(last_dim)?;
    let diff = x.sub(&mean)?;
    let var = diff.sqr()?.mean_keepdim(diff.rank() - 1)?;
    let inv_std = var.add_scalar(eps)?.sqrt()?.recip()?;
    let normalized = diff.mul(&inv_std)?;

    // grad_bias = sum(grad) over all dims except last
    let grad_bias = norm::sum_all_but_last(grad)?;
    accumulate(bias, &grad_bias, grads)?;

    // grad_weight = sum(grad * normalized) over all dims except last
    let grad_weight = norm::sum_all_but_last(&grad.mul(&normalized)?)?;
    accumulate(weight, &grad_weight, grads)?;

    // grad_input: (1/std) * (grad*gamma - mean(grad*gamma) - norm*mean(grad*gamma*norm))
    let grad_gamma = grad.mul(gamma)?;
    let mean_gg = grad_gamma.mean_keepdim(last_dim)?;
    let mean_gg_norm = grad_gamma.mul(&normalized)?.mean_keepdim(last_dim)?;
    let grad_input = inv_std.mul(
        &grad_gamma
            .sub(&mean_gg.expand(grad.dims())?)?
            .sub(&normalized.mul(&mean_gg_norm.expand(grad.dims())?)?)?,
    )?;
    accumulate(input, &grad_input, grads)?;
    Ok(())
}

/// Backward rule for embedding: scatter gradient into weight table rows.
///
/// For each position i in the flattened indices, grad_weight[indices[i]] += grad[i].
///
/// Uses `DynTensor::index_add` which is device-agnostic — GPU tensors
/// stay on device (with CPU round-trip fallback until native GPU kernel lands).
pub(super) fn backward_embedding(
    weight: &Arc<TrackedTensor>,
    indices: &Arc<TrackedTensor>,
    grad: &DynTensor,
    grads: &mut GradStore,
) -> Result<()> {
    let w_dims = weight.tensor().dims();
    if w_dims.len() < 2 || w_dims[1] == 0 {
        return Err(crate::AutodiffError::InvalidConfig {
            op: "embedding_backward",
            reason: format!(
                "weight must be 2D with embed_dim > 0, got shape {w_dims:?}",
            ),
        });
    }
    let vocab_size = w_dims[0];
    let embed_dim = w_dims[1];
    let device = weight.tensor().device();

    // Flatten indices to 1-D U32 for index_add.
    let num_tokens = grad.numel() / embed_dim;
    let idx_flat = indices
        .tensor()
        .reshape([num_tokens])?
        .to_dtype(DType::U32)?;

    // Reshape grad to [num_tokens, embed_dim] for row-wise accumulation.
    let grad_flat = grad.reshape([num_tokens, embed_dim])?;

    // Accumulate: grad_weight[idx_flat[i]] += grad_flat[i] for each token i.
    // index_add_into avoids cloning the zeros tensor (refcount == 1).
    let grad_weight = DynTensor::zeros(&[vocab_size, embed_dim], weight.tensor().dtype(), &device)?;
    let gw = grad_weight.index_add_into(0, &idx_flat, &grad_flat)?;
    accumulate(weight, &gw, grads)?;
    // No gradient for indices (discrete, non-differentiable)
    Ok(())
}

/// Backward rule for cross-entropy loss with fused softmax gradient.
///
/// For `loss = -mean(log_softmax(logits, dim)[targets])`:
///   `grad_logits = (softmax(logits) - one_hot(targets)) / N`
/// where N is the number of samples (total elements / num_classes).
///
/// Uses device-agnostic `scatter_add` for one-hot construction — GPU tensors
/// stay on device (with CPU round-trip fallback until native GPU scatter lands).
pub(super) fn backward_cross_entropy(
    input: &Arc<TrackedTensor>,
    targets: &Arc<TrackedTensor>,
    dim: usize,
    grad: &DynTensor,
    grads: &mut GradStore,
) -> Result<()> {
    let logits = input.tensor();
    let num_classes = logits.dims()[dim];
    // Guard: num_classes == 0 would cause division-by-zero below.
    if num_classes == 0 {
        return Err(crate::AutodiffError::InvalidConfig {
            op: "cross_entropy_loss backward",
            reason: format!("dim {dim} has size 0 (no classes)"),
        });
    }
    let n = logits.numel() / num_classes;

    // Empty batch: no elements to backpropagate through.
    if n == 0 {
        return Ok(());
    }

    // Compute softmax of the original logits (numerically stable)
    let softmax = logits.softmax(dim)?;

    // Build one-hot encoding using scatter_add (device-agnostic).
    // scatter_add(dim, index, src) requires:
    //   index: same rank as logits, with size 1 along `dim`, U32 dtype
    //   src: same shape as index (ones)
    let target_u32 = targets.tensor().to_dtype(DType::U32)?;
    // If targets rank < logits rank, unsqueeze to match.
    // Tests pass targets as [N, 1] (rank 2) for logits [N, C] (rank 2) — no unsqueeze needed.
    // User may also pass targets as [N] (rank 1) for logits [N, C] — needs unsqueeze.
    let idx = if target_u32.rank() < logits.rank() {
        target_u32.unsqueeze(dim)?
    } else {
        target_u32
    };
    let ones_src = DynTensor::ones(idx.dims(), logits.dtype(), &logits.device())?;
    // scatter_add_into avoids cloning the zeros tensor (refcount == 1).
    let zeros = DynTensor::zeros(logits.dims(), logits.dtype(), &logits.device())?;
    let one_hot = zeros.scatter_add_into(dim, &idx, &ones_src)?;

    // grad_logits = (softmax - one_hot) / N * upstream_grad
    let diff = softmax.sub(&one_hot)?;
    let scaled = diff.mul_scalar(1.0 / n as f64)?;
    // Upstream grad is scalar — multiply by its value to broadcast correctly.
    let grad_val = scalar_grad_val(grad)?;
    let grad_input = scaled.mul_scalar(grad_val)?;
    accumulate(input, &grad_input, grads)?;
    // No gradient for targets (discrete indices)
    Ok(())
}

/// Backward rule for MSE loss: grad_input = 2 * (input - target) / N.
fn backward_mse(
    input: &Arc<TrackedTensor>,
    target: &Arc<TrackedTensor>,
    grad: &DynTensor,
    grads: &mut GradStore,
) -> Result<()> {
    let n = input.tensor().numel().max(1) as f64;
    let grad_val = scalar_grad_val(grad)?;
    let diff = input.tensor().sub(target.tensor())?;
    let grad_input = diff.mul_scalar(2.0 * grad_val / n)?;
    accumulate(input, &grad_input, grads)?;
    let grad_target = diff.mul_scalar(-2.0 * grad_val / n)?;
    accumulate(target, &grad_target, grads)?;
    Ok(())
}

/// Backward rule for L1 loss: grad_input = sign(input - target) / N.
fn backward_l1(
    input: &Arc<TrackedTensor>,
    target: &Arc<TrackedTensor>,
    grad: &DynTensor,
    grads: &mut GradStore,
) -> Result<()> {
    let n = input.tensor().numel().max(1) as f64;
    let grad_val = scalar_grad_val(grad)?;
    let diff = input.tensor().sub(target.tensor())?;
    // sign(diff): +1 where diff > 0, -1 where diff < 0, 0 where diff == 0
    let pos = diff.gt(0.0)?;
    let neg = diff.lt(0.0)?;
    let ones = DynTensor::ones(diff.dims(), diff.dtype(), &diff.device())?;
    let zeros = DynTensor::zeros(diff.dims(), diff.dtype(), &diff.device())?;
    let sign = pos
        .where_cond(&ones, &zeros)?
        .sub(&neg.where_cond(&ones, &zeros)?)?;
    let grad_input = sign.mul_scalar(grad_val / n)?;
    accumulate(input, &grad_input, grads)?;
    accumulate(target, &grad_input.neg()?, grads)?;
    Ok(())
}

/// Backward rule for Huber loss.
///
/// Gradient: `diff / (N * delta)` where `|diff| < delta`, `sign(diff) / N` otherwise.
fn backward_huber(
    input: &Arc<TrackedTensor>,
    target: &Arc<TrackedTensor>,
    delta: f64,
    grad: &DynTensor,
    grads: &mut GradStore,
) -> Result<()> {
    let n = input.tensor().numel().max(1) as f64;
    let grad_val = scalar_grad_val(grad)?;
    let diff = input.tensor().sub(target.tensor())?;
    let abs_diff = diff.abs()?;
    // Quadratic region gradient: diff / delta
    let quad_grad = diff.mul_scalar(1.0 / delta)?;
    // Linear region gradient: sign(diff)
    let pos = diff.gt(0.0)?;
    let neg = diff.lt(0.0)?;
    let ones = DynTensor::ones(diff.dims(), diff.dtype(), &diff.device())?;
    let zeros = DynTensor::zeros(diff.dims(), diff.dtype(), &diff.device())?;
    let sign = pos
        .where_cond(&ones, &zeros)?
        .sub(&neg.where_cond(&ones, &zeros)?)?;
    // Select: where |diff| < delta use quad_grad, else sign
    let mask = abs_diff.lt(delta)?;
    let elem_grad = mask.where_cond(&quad_grad, &sign)?;
    let grad_input = elem_grad.mul_scalar(grad_val / n)?;
    accumulate(input, &grad_input, grads)?;
    accumulate(target, &grad_input.neg()?, grads)?;
    Ok(())
}
