// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Backward rules for each Op variant.
//!
//! Each rule computes input gradients from the output gradient and accumulates
//! them into the [`GradStore`]. Implements the chain rule for reverse-mode AD.
//!
//! Composite rules (LayerNorm, Embedding, CrossEntropy) are in `backward_rules_special.rs`.
//! Conv1d rules are in `backward_rules_conv.rs`.
//! Conv2d rules are in `backward_rules_conv2d.rs`.
//! Elementwise rules (activations + math) are in `backward_rules_elementwise.rs`.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::tensor::checked_dim_product;

use crate::error::{AutodiffError, Result};
use crate::grad::GradStore;
use crate::op::Op;
use crate::tracked::TrackedTensor;

#[path = "backward_rules_conv.rs"]
mod conv;
#[path = "backward_rules_conv2d.rs"]
mod conv2d;
#[path = "backward_rules_conv_transpose.rs"]
mod conv_transpose;
#[path = "backward_rules_elementwise.rs"]
mod elementwise;
#[path = "backward_rules_pool.rs"]
mod pool;
#[path = "backward_rules_special.rs"]
mod special;

/// Dispatch backward rule for an operation.
pub(crate) fn backward_op(op: &Op, grad: &DynTensor, grads: &mut GradStore) -> Result<()> {
    match op {
        Op::Add(..) | Op::Sub(..) | Op::Mul(..) | Op::Div(..) | Op::MatMul(..) => {
            backward_binary(op, grad, grads)
        }
        Op::Relu(..)
        | Op::Gelu(..)
        | Op::GeluErf(..)
        | Op::Silu(..)
        | Op::Tanh(..)
        | Op::Exp(..)
        | Op::Neg(..)
        | Op::Sigmoid(..)
        | Op::Log(..)
        | Op::Sqrt(..)
        | Op::Sqr(..)
        | Op::Abs(..) => elementwise::backward_activation(op, grad, grads),

        Op::Sin(..) | Op::Cos(..) | Op::Recip(..) | Op::Powf(..) | Op::Clamp(..) | Op::Elu(..) => {
            elementwise::backward_elementwise_math(op, grad, grads)
        }

        Op::HardSigmoid(..)
        | Op::HardSwish(..)
        | Op::Mish(..)
        | Op::Selu(..)
        | Op::Softplus(..)
        | Op::Celu(..) => elementwise::backward_new_activations(op, grad, grads),

        Op::MulScalar(x, s) => accumulate(x, &grad.affine(*s, 0.0)?, grads),
        Op::AddScalar(x, _) => accumulate(x, grad, grads),

        Op::Dropout(x, mask, scale) => {
            // Backward: grad_input = grad * mask * scale (same mask as forward)
            accumulate(x, &grad.mul(mask.tensor())?.mul_scalar(*scale)?, grads)
        }

        Op::SumKeepDim(..) | Op::MeanKeepDim(..) => backward_reduce(op, grad, grads),

        Op::Reshape(..)
        | Op::Transpose(..)
        | Op::Narrow(..)
        | Op::Unsqueeze(..)
        | Op::Squeeze(..)
        | Op::Broadcast(..)
        | Op::Permute(..)
        | Op::Unfold(..) => backward_shape(op, grad, grads),

        Op::Conv1d { .. } => conv::backward_conv1d(op, grad, grads),
        Op::Conv2d { .. } => conv2d::backward_conv2d(op, grad, grads),
        Op::ConvTranspose1d { .. } => conv_transpose::backward_conv_transpose1d(op, grad, grads),
        Op::Cat(..) => backward_cat(op, grad, grads),
        Op::Stack(..) => backward_stack(op, grad, grads),

        Op::Maximum(..) | Op::Minimum(..) => backward_minmax(op, grad, grads),
        Op::LogSoftmax(..) => backward_log_softmax(op, grad, grads),

        Op::MaxPool1d { .. }
        | Op::MaxPool2d { .. }
        | Op::AdaptiveAvgPool2d { .. }
        | Op::AvgPool2d { .. } => pool::backward_pool(op, grad, grads),

        Op::Softmax(..)
        | Op::LayerNorm { .. }
        | Op::RmsNorm { .. }
        | Op::GroupNorm { .. }
        | Op::BatchNorm { .. }
        | Op::InstanceNorm { .. }
        | Op::Embedding(..)
        | Op::CrossEntropyLoss(..)
        | Op::MseLoss(..)
        | Op::L1Loss(..)
        | Op::HuberLoss(..) => special::backward_special(op, grad, grads),
    }
}

/// Backward rules for binary arithmetic (Add, Sub, Mul, Div, MatMul).
fn backward_binary(op: &Op, grad: &DynTensor, grads: &mut GradStore) -> Result<()> {
    match op {
        Op::Add(a, b) => {
            accumulate(a, &reduce_to_shape(grad, a.tensor().dims())?, grads)?;
            accumulate(b, &reduce_to_shape(grad, b.tensor().dims())?, grads)
        }
        Op::Sub(a, b) => {
            accumulate(a, &reduce_to_shape(grad, a.tensor().dims())?, grads)?;
            accumulate(b, &reduce_to_shape(&grad.neg()?, b.tensor().dims())?, grads)
        }
        Op::Mul(a, b) => {
            accumulate(
                a,
                &reduce_to_shape(&grad.mul(b.tensor())?, a.tensor().dims())?,
                grads,
            )?;
            accumulate(
                b,
                &reduce_to_shape(&grad.mul(a.tensor())?, b.tensor().dims())?,
                grads,
            )
        }
        Op::Div(a, b) => {
            accumulate(
                a,
                &reduce_to_shape(&grad.div(b.tensor())?, a.tensor().dims())?,
                grads,
            )?;
            let neg_a = a.tensor().neg()?;
            let grad_b = grad.mul(&neg_a.div(&b.tensor().sqr()?)?)?;
            accumulate(b, &reduce_to_shape(&grad_b, b.tensor().dims())?, grads)
        }
        Op::MatMul(a, b) => {
            let r_b = b.tensor().rank();
            let r_a = a.tensor().rank();
            if r_b < 2 || r_a < 2 {
                return Err(AutodiffError::MatMulRankTooLow {
                    rank_a: r_a,
                    rank_b: r_b,
                });
            }
            let b_t = b.tensor().transpose(r_b - 2, r_b - 1)?;
            let grad_a = reduce_to_shape(&grad.matmul(&b_t)?, a.tensor().dims())?;
            accumulate(a, &grad_a, grads)?;
            let a_t = a.tensor().transpose(r_a - 2, r_a - 1)?;
            let grad_b = reduce_to_shape(&a_t.matmul(grad)?, b.tensor().dims())?;
            accumulate(b, &grad_b, grads)
        }
        other => Err(unsupported(other)),
    }
}

/// Backward rules for reduction operations (Sum, Mean).
fn backward_reduce(op: &Op, grad: &DynTensor, grads: &mut GradStore) -> Result<()> {
    match op {
        Op::SumKeepDim(x, _dim) => accumulate(x, &grad.expand(x.tensor().dims())?, grads),
        Op::MeanKeepDim(x, dim) => {
            let n = x.tensor().dims()[*dim] as f64;
            let scaled = grad.mul_scalar(1.0 / n)?;
            accumulate(x, &scaled.expand(x.tensor().dims())?, grads)
        }
        other => Err(unsupported(other)),
    }
}

/// Backward rules for shape operations (Reshape, Transpose, Narrow, etc).
fn backward_shape(op: &Op, grad: &DynTensor, grads: &mut GradStore) -> Result<()> {
    match op {
        Op::Reshape(x, orig) => accumulate(x, &grad.reshape(orig)?, grads),
        Op::Transpose(x, d1, d2) => accumulate(x, &grad.transpose(*d1, *d2)?, grads),
        Op::Narrow(x, dim, start, orig_dim_size) => {
            // Narrow sliced [start..start+len] from dim of size orig_dim_size.
            // Backward: place gradient into a zeros tensor at the original offset.
            // Uses zeros + slice_set (2 allocations, potentially in-place) instead
            // of the prior approach of up to 3 zero-pads + cat (4 allocations).
            let mut full_shape = grad.dims().to_vec();
            full_shape[*dim] = *orig_dim_size;
            let padded = DynTensor::zeros(&full_shape, grad.dtype(), &grad.device())?
                .slice_set_into(*dim, *start, grad)?;
            accumulate(x, &padded, grads)
        }
        Op::Unsqueeze(x, dim) => accumulate(x, &grad.squeeze(*dim)?, grads),
        Op::Squeeze(x, dim) => accumulate(x, &grad.unsqueeze(*dim)?, grads),
        Op::Broadcast(x, _) => accumulate(x, &reduce_to_shape(grad, x.tensor().dims())?, grads),
        // Unfold backward: scatter-add gradients from windows back to input positions.
        // Forward: input[..., T, ...] → output[..., n_windows, ..., size]
        //   where `size` is appended as the trailing dimension.
        // Backward: for each window w, extract grad[..., w, ..., :], move the
        //   trailing `size` dim back to position `dim`, then scatter-add into result.
        Op::Unfold(x, dim, size, step) => {
            let input_shape = x.tensor().dims();
            let grad_dims = grad.dims();
            let n_windows = grad_dims[*dim];
            let grad_rank = grad_dims.len(); // input_rank + 1
            let mut result = DynTensor::zeros(input_shape, grad.dtype(), &grad.device())?;
            for w in 0..n_windows {
                // Extract window w's gradient: narrow dim to 1, squeeze.
                // After squeeze: [..., d_{dim+1}, ..., dN, size] — size is at end.
                let window_grad = grad.narrow(*dim, w, 1)?.squeeze(*dim)?;
                // Move trailing `size` dim from position (grad_rank-2) back to `dim`.
                // Build permutation: [0, ..., dim-1, last, dim, ..., last-1]
                let squeezed_rank = grad_rank - 1; // = input_rank
                let window_grad = if *dim < squeezed_rank - 1 {
                    let mut perm: Vec<usize> = (0..squeezed_rank).collect();
                    // Remove last position and insert at dim.
                    let last = perm.pop().ok_or(AutodiffError::EmptySequence {
                        op: "unfold_backward",
                    })?;
                    perm.insert(*dim, last);
                    window_grad.permute(&perm)?
                } else {
                    // dim is the last input dim — size is already in correct position.
                    window_grad
                };
                // Scatter-add into result at position w*step along dim.
                let offset = w * step;
                let existing = result.narrow(*dim, offset, *size)?;
                let updated = existing.add(&window_grad)?;
                result = result.slice_set_into(*dim, offset, &updated)?;
            }
            accumulate(x, &result, grads)
        }
        // Permute backward: apply the inverse permutation (stored in the Op).
        Op::Permute(x, inv_perm) => accumulate(x, &grad.permute(inv_perm)?, grads),
        other => Err(unsupported(other)),
    }
}

/// Backward rule for concatenation: split gradient along the cat dimension.
fn backward_cat(op: &Op, grad: &DynTensor, grads: &mut GradStore) -> Result<()> {
    match op {
        Op::Cat(inputs, dim) => {
            let mut offset = 0;
            for input in inputs {
                let len = input.tensor().dims()[*dim];
                let g = grad.narrow(*dim, offset, len)?;
                accumulate(input, &g, grads)?;
                offset += len;
            }
            Ok(())
        }
        other => Err(unsupported(other)),
    }
}

/// Backward rule for stack: remove the new dimension and split.
fn backward_stack(op: &Op, grad: &DynTensor, grads: &mut GradStore) -> Result<()> {
    match op {
        Op::Stack(inputs, dim) => {
            for (i, input) in inputs.iter().enumerate() {
                // Select the i-th slice along the stacked dimension, then squeeze it.
                let g = grad.narrow(*dim, i, 1)?.squeeze(*dim)?;
                accumulate(input, &g, grads)?;
            }
            Ok(())
        }
        other => Err(unsupported(other)),
    }
}

/// Backward rule for element-wise maximum/minimum (subgradient).
///
/// NaN defense: when either operand is NaN, `diff = a - b` is NaN, and
/// IEEE 754 comparisons (`ge`, `lt`, `le`, `gt`) all return false. Both
/// masks become zero, silently dropping the gradient. We validate `diff`
/// for non-finite values before masking to fail-closed (#1999).
fn backward_minmax(op: &Op, grad: &DynTensor, grads: &mut GradStore) -> Result<()> {
    match op {
        Op::Maximum(a, b) => {
            // grad_a = grad where a >= b, 0 otherwise (subgradient: tie → a gets gradient)
            // Use (a - b) >= 0 as mask since tensor-to-tensor ge isn't available.
            let diff = a.tensor().sub(b.tensor())?;
            if diff.any_non_finite()? {
                return Err(AutodiffError::NonFiniteBackwardInput { op: "Maximum" });
            }
            let mask_a = diff.ge(0.0)?;
            let zeros = DynTensor::zeros(grad.dims(), grad.dtype(), &grad.device())?;
            accumulate(
                a,
                &reduce_to_shape(&mask_a.where_cond(grad, &zeros)?, a.tensor().dims())?,
                grads,
            )?;
            // grad_b = grad where b > a (diff < 0)
            let mask_b = diff.lt(0.0)?;
            accumulate(
                b,
                &reduce_to_shape(&mask_b.where_cond(grad, &zeros)?, b.tensor().dims())?,
                grads,
            )
        }
        Op::Minimum(a, b) => {
            // grad_a = grad where a <= b, 0 otherwise (subgradient: tie → a gets gradient)
            let diff = a.tensor().sub(b.tensor())?;
            if diff.any_non_finite()? {
                return Err(AutodiffError::NonFiniteBackwardInput { op: "Minimum" });
            }
            let mask_a = diff.le(0.0)?;
            let zeros = DynTensor::zeros(grad.dims(), grad.dtype(), &grad.device())?;
            accumulate(
                a,
                &reduce_to_shape(&mask_a.where_cond(grad, &zeros)?, a.tensor().dims())?,
                grads,
            )?;
            // grad_b = grad where b < a (diff > 0)
            let mask_b = diff.gt(0.0)?;
            accumulate(
                b,
                &reduce_to_shape(&mask_b.where_cond(grad, &zeros)?, b.tensor().dims())?,
                grads,
            )
        }
        other => Err(unsupported(other)),
    }
}

/// Backward rule for log_softmax.
fn backward_log_softmax(op: &Op, grad: &DynTensor, grads: &mut GradStore) -> Result<()> {
    match op {
        Op::LogSoftmax(x, dim) => {
            // log_softmax backward: grad_input = grad - softmax * sum(grad, dim)
            let softmax = x.tensor().softmax(*dim)?;
            let grad_sum = grad.sum_keepdim(*dim)?;
            let g = grad.sub(&softmax.mul(&grad_sum)?)?;
            accumulate(x, &g, grads)
        }
        other => Err(unsupported(other)),
    }
}

/// Construct an `UnsupportedBackward` error for an unrecognized `Op` variant.
///
/// Used by all backward rule dispatch functions as the catch-all `other =>` arm.
fn unsupported(op: &Op) -> AutodiffError {
    AutodiffError::UnsupportedBackward(format!("{op:?}"))
}

/// Accumulate gradient for a node.
fn accumulate(node: &Arc<TrackedTensor>, grad: &DynTensor, grads: &mut GradStore) -> Result<()> {
    grads.accumulate_node(node.node_id(), grad)?;
    Ok(())
}

/// Reshape a `[C]` tensor to `[1, C, 1, 1, ...]` for left-aligned broadcast
/// against `[N, C, *spatial]` tensors. NumPy-style right-aligned broadcasting
/// would incorrectly map `[C]` to the trailing (spatial) dimension.
///
/// Used by both backward rules (backward_rules_norm.rs) and forward
/// composite ops (tracked_composite_ops_norm.rs) for normalization layers.
pub(crate) fn reshape_for_channel_broadcast(
    t: &DynTensor,
    target_rank: usize,
) -> std::result::Result<DynTensor, nn_core::TensorError> {
    if target_rank < 2 {
        return Err(nn_core::TensorError::ValueOutOfRange {
            description: "reshape_for_channel_broadcast: target_rank must be >= 2",
        });
    }
    let c = t.dims()[0];
    let mut shape = vec![1usize; target_rank];
    shape[1] = c;
    t.reshape(&shape)
}

/// Reduce a tensor back to a target shape by summing over broadcast dimensions.
///
/// When rank differs, collapses leading dimensions in a single reshape+sum
/// instead of repeated sum_keepdim(0)/squeeze(0) which creates 2 intermediate
/// tensors per iteration.
fn reduce_to_shape(tensor: &DynTensor, target: &[usize]) -> Result<DynTensor> {
    if tensor.dims() == target {
        return Ok(tensor.clone());
    }
    let mut result = tensor.clone();
    // Collapse extra leading dims in one reshape+sum instead of repeated
    // sum_keepdim(0)+squeeze(0). For rank 6→rank 4, this does 1 reshape + 1 sum
    // instead of 2×(sum_keepdim + squeeze) = 4 intermediate allocations.
    let extra = result.rank().saturating_sub(target.len());
    if extra > 0 {
        let dims = result.dims();
        let leading_product = checked_dim_product(&dims[..extra])?;
        let mut new_shape = vec![leading_product];
        new_shape.extend_from_slice(&dims[extra..]);
        result = result.reshape(&new_shape)?.sum_keepdim(0)?.squeeze(0)?;
    }
    // Sum dims that were broadcast (target == 1 but result > 1).
    for (d, &t) in target.iter().enumerate() {
        if t == 1 && result.dim(d)? > 1 {
            result = result.sum_keepdim(d)?;
        }
    }
    Ok(result)
}
