// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Computation graph operations for automatic differentiation.
//!
//! Each [`Op`] variant records how a [`TrackedTensor`] was computed, enabling
//! the backward pass to apply the chain rule. Inputs are stored as `Arc<TrackedTensor>`
//! so the graph is a DAG of reference-counted nodes.

use crate::tracked::TrackedTensor;
use std::sync::Arc;

/// Operation that produced a `TrackedTensor`.
///
/// Each variant stores references to its inputs so the backward pass
/// can traverse the computation graph and compute gradients.
#[non_exhaustive]
pub enum Op {
    // -- Binary element-wise --
    /// a + b
    Add(Arc<TrackedTensor>, Arc<TrackedTensor>),
    /// a - b
    Sub(Arc<TrackedTensor>, Arc<TrackedTensor>),
    /// a * b (element-wise)
    Mul(Arc<TrackedTensor>, Arc<TrackedTensor>),
    /// a / b (element-wise)
    Div(Arc<TrackedTensor>, Arc<TrackedTensor>),
    /// Matrix multiply: a @ b
    MatMul(Arc<TrackedTensor>, Arc<TrackedTensor>),

    // -- Unary element-wise --
    /// ReLU(x) = max(0, x)
    Relu(Arc<TrackedTensor>),
    /// GELU (tanh approximation)
    Gelu(Arc<TrackedTensor>),
    /// GELU (exact erf-based): x * 0.5 * (1 + erf(x / sqrt(2)))
    GeluErf(Arc<TrackedTensor>),
    /// SiLU(x) = x * sigmoid(x)
    Silu(Arc<TrackedTensor>),
    /// tanh(x)
    Tanh(Arc<TrackedTensor>),
    /// sigmoid(x) = 1 / (1 + exp(-x))
    Sigmoid(Arc<TrackedTensor>),
    /// exp(x)
    Exp(Arc<TrackedTensor>),
    /// ln(x)
    Log(Arc<TrackedTensor>),
    /// sqrt(x)
    Sqrt(Arc<TrackedTensor>),
    /// x^2
    Sqr(Arc<TrackedTensor>),
    /// -x
    Neg(Arc<TrackedTensor>),
    /// |x|
    Abs(Arc<TrackedTensor>),

    // -- Reductions (keepdim = true) --
    /// Sum over a single axis, keeping dimension.
    SumKeepDim(Arc<TrackedTensor>, usize),
    /// Mean over a single axis, keeping dimension.
    MeanKeepDim(Arc<TrackedTensor>, usize),

    // -- Shape operations --
    /// Reshape to new dims. Stores the *original* dims for backward.
    Reshape(Arc<TrackedTensor>, Vec<usize>),
    /// Transpose two dimensions.
    Transpose(Arc<TrackedTensor>, usize, usize),
    /// Narrow (slice): dim, start, original_dim_size for backward zero-padding.
    Narrow(Arc<TrackedTensor>, usize, usize, usize),
    /// Broadcast expand. Stores *original* shape for backward reduction.
    Broadcast(Arc<TrackedTensor>, Vec<usize>),
    /// Unsqueeze: inserted dimension index.
    Unsqueeze(Arc<TrackedTensor>, usize),
    /// Squeeze: removed dimension index.
    Squeeze(Arc<TrackedTensor>, usize),
    /// Unfold (sliding window extraction): dim, size, step.
    /// Stores original shape for backward scatter-add.
    Unfold(Arc<TrackedTensor>, usize, usize, usize),

    // -- Convolution --
    /// 1D convolution: input, kernel, padding, stride, dilation, groups.
    Conv1d {
        input: Arc<TrackedTensor>,
        kernel: Arc<TrackedTensor>,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    },
    /// 2D convolution: input, kernel, padding, stride, dilation, groups.
    Conv2d {
        input: Arc<TrackedTensor>,
        kernel: Arc<TrackedTensor>,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    },

    // -- Concatenation --
    /// Concatenate along a dimension.
    Cat(Vec<Arc<TrackedTensor>>, usize),

    // -- Composite operations --
    /// Softmax over a dimension. Stores the dimension index.
    Softmax(Arc<TrackedTensor>, usize),
    /// Layer normalization. Stores input, weight (gamma), bias (beta), eps, normalized_shape len.
    LayerNorm {
        input: Arc<TrackedTensor>,
        weight: Arc<TrackedTensor>,
        bias: Arc<TrackedTensor>,
        eps: f64,
        normalized_shape: usize,
    },
    /// Embedding lookup. Stores weight table and input indices.
    Embedding(Arc<TrackedTensor>, Arc<TrackedTensor>),

    // -- Loss functions --
    /// Cross-entropy loss: -log_softmax(input, dim)[targets].mean()
    /// Stores input logits, target indices, and softmax dimension.
    CrossEntropyLoss(Arc<TrackedTensor>, Arc<TrackedTensor>, usize),
    /// Mean squared error loss: mean((input - target)^2).
    /// Stores input predictions and target values.
    MseLoss(Arc<TrackedTensor>, Arc<TrackedTensor>),
    /// L1 loss: mean(|input - target|).
    /// Stores input predictions and target values.
    L1Loss(Arc<TrackedTensor>, Arc<TrackedTensor>),
    /// Huber (smooth L1) loss with transition point delta.
    /// Quadratic for |x| < delta, linear for |x| >= delta.
    HuberLoss(Arc<TrackedTensor>, Arc<TrackedTensor>, f64),

    // -- Regularization --
    /// Dropout: stores input, binary mask (1=keep, 0=drop), and scale factor 1/(1-p).
    /// Backward: grad * mask * scale (same mask as forward).
    Dropout(Arc<TrackedTensor>, Arc<TrackedTensor>, f64),

    // -- Additional unary element-wise --
    /// sin(x)
    Sin(Arc<TrackedTensor>),
    /// cos(x)
    Cos(Arc<TrackedTensor>),
    /// 1/x
    Recip(Arc<TrackedTensor>),
    /// x^p (element-wise power with scalar exponent)
    Powf(Arc<TrackedTensor>, f64),
    /// clamp(x, min, max)
    Clamp(Arc<TrackedTensor>, f64, f64),

    // -- Shape operations (multi-axis) --
    /// Permute axes. Stores the *inverse* permutation for backward.
    Permute(Arc<TrackedTensor>, Vec<usize>),

    // -- Normalization --
    /// RMS normalization: x / rms(x) * weight. Stores input, weight, eps.
    RmsNorm {
        input: Arc<TrackedTensor>,
        weight: Arc<TrackedTensor>,
        eps: f64,
    },
    /// Group normalization. Stores input, weight (gamma), bias (beta), num_groups, eps.
    GroupNorm {
        input: Arc<TrackedTensor>,
        weight: Arc<TrackedTensor>,
        bias: Arc<TrackedTensor>,
        num_groups: usize,
        eps: f64,
    },
    /// Batch normalization (training mode). Stores input, weight, bias, eps.
    BatchNorm {
        input: Arc<TrackedTensor>,
        weight: Arc<TrackedTensor>,
        bias: Arc<TrackedTensor>,
        eps: f64,
    },
    /// Instance normalization. Stores input, weight, bias, eps.
    InstanceNorm {
        input: Arc<TrackedTensor>,
        weight: Arc<TrackedTensor>,
        bias: Arc<TrackedTensor>,
        eps: f64,
    },

    // -- Transposed Convolution --
    /// 1D transposed convolution: input, kernel, padding, stride, dilation, groups, output_padding.
    ConvTranspose1d {
        input: Arc<TrackedTensor>,
        kernel: Arc<TrackedTensor>,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        output_padding: usize,
    },

    // -- Additional activations --
    /// ELU(x, alpha) = x if x > 0, alpha * (exp(x) - 1) otherwise
    Elu(Arc<TrackedTensor>, f64),
    /// HardSigmoid(x) = max(0, min(1, x/6 + 0.5))
    HardSigmoid(Arc<TrackedTensor>),
    /// HardSwish(x) = x * HardSigmoid(x)
    HardSwish(Arc<TrackedTensor>),
    /// Mish(x) = x * tanh(softplus(x))
    Mish(Arc<TrackedTensor>),
    /// SELU(x) = lambda * (x if x >= 0, else alpha * (exp(x) - 1))
    Selu(Arc<TrackedTensor>),
    /// Softplus(x) = log(1 + exp(x))
    Softplus(Arc<TrackedTensor>),
    /// CELU(x, alpha) = max(0,x) + min(0, alpha*(exp(x/alpha)-1))
    Celu(Arc<TrackedTensor>, f64),
    /// log_softmax(x, dim) = x - log(sum(exp(x), dim))
    LogSoftmax(Arc<TrackedTensor>, usize),

    // -- Binary element-wise (comparator-based) --
    /// Element-wise maximum of two tensors.
    Maximum(Arc<TrackedTensor>, Arc<TrackedTensor>),
    /// Element-wise minimum of two tensors.
    Minimum(Arc<TrackedTensor>, Arc<TrackedTensor>),

    // -- Stack --
    /// Stack tensors along a new dimension.
    Stack(Vec<Arc<TrackedTensor>>, usize),

    // -- Scalar operations --
    /// x * scalar
    MulScalar(Arc<TrackedTensor>, f64),
    /// x + scalar
    AddScalar(Arc<TrackedTensor>, f64),

    // -- Pooling --
    /// 1-D max pooling. Stores input, argmax flat indices (u32), kernel_size, stride, padding.
    /// Argmax indices map each output position to the flat index of the max element in the input.
    MaxPool1d {
        input: Arc<TrackedTensor>,
        indices: nn_core::dyn_tensor::DynTensor,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    },
    /// 2-D max pooling. Stores input, argmax flat indices (u32), kernel_size, stride, padding.
    /// Argmax indices map each output position to the flat index of the max element in the input.
    MaxPool2d {
        input: Arc<TrackedTensor>,
        indices: nn_core::dyn_tensor::DynTensor,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    },
    /// Adaptive 2-D average pooling. Stores input and target output size.
    AdaptiveAvgPool2d {
        input: Arc<TrackedTensor>,
        output_h: usize,
        output_w: usize,
    },
    /// 2-D average pooling. Stores input, kernel_size, stride, padding.
    AvgPool2d {
        input: Arc<TrackedTensor>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    },
}

/// Debug format helper for pooling Op variants.
fn fmt_pool(op: &Op, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match op {
        Op::MaxPool1d {
            kernel_size: k,
            stride: s,
            padding: p,
            ..
        } => {
            write!(f, "MaxPool1d(k={k}, s={s}, p={p})")
        }
        Op::MaxPool2d {
            kernel_size: k,
            stride: s,
            padding: p,
            ..
        } => {
            write!(f, "MaxPool2d(k={k}, s={s}, p={p})")
        }
        Op::AvgPool2d {
            kernel_size: k,
            stride: s,
            padding: p,
            ..
        } => {
            write!(f, "AvgPool2d(k={k}, s={s}, p={p})")
        }
        Op::AdaptiveAvgPool2d {
            output_h, output_w, ..
        } => {
            write!(f, "AdaptiveAvgPool2d({output_h}, {output_w})")
        }
        _ => unreachable!(),
    }
}

impl std::fmt::Debug for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add(..) => write!(f, "Add"),
            Self::Sub(..) => write!(f, "Sub"),
            Self::Mul(..) => write!(f, "Mul"),
            Self::Div(..) => write!(f, "Div"),
            Self::MatMul(..) => write!(f, "MatMul"),
            Self::Relu(..) => write!(f, "Relu"),
            Self::Gelu(..) => write!(f, "Gelu"),
            Self::GeluErf(..) => write!(f, "GeluErf"),
            Self::Silu(..) => write!(f, "Silu"),
            Self::Tanh(..) => write!(f, "Tanh"),
            Self::Sigmoid(..) => write!(f, "Sigmoid"),
            Self::Exp(..) => write!(f, "Exp"),
            Self::Log(..) => write!(f, "Log"),
            Self::Sqrt(..) => write!(f, "Sqrt"),
            Self::Sqr(..) => write!(f, "Sqr"),
            Self::Neg(..) => write!(f, "Neg"),
            Self::Abs(..) => write!(f, "Abs"),
            Self::SumKeepDim(_, d) => write!(f, "SumKeepDim(dim={d})"),
            Self::MeanKeepDim(_, d) => write!(f, "MeanKeepDim(dim={d})"),
            Self::Reshape(_, s) => write!(f, "Reshape({s:?})"),
            Self::Transpose(_, d1, d2) => write!(f, "Transpose({d1}, {d2})"),
            Self::Narrow(_, d, s, _) => write!(f, "Narrow(dim={d}, start={s})"),
            Self::Broadcast(_, s) => write!(f, "Broadcast(orig={s:?})"),
            Self::Unsqueeze(_, d) => write!(f, "Unsqueeze({d})"),
            Self::Squeeze(_, d) => write!(f, "Squeeze({d})"),
            Self::Unfold(_, d, sz, st) => write!(f, "Unfold(dim={d}, size={sz}, step={st})"),
            Self::Conv1d { .. } => write!(f, "Conv1d"),
            Self::Conv2d { .. } => write!(f, "Conv2d"),
            Self::Cat(_, d) => write!(f, "Cat(dim={d})"),
            Self::Softmax(_, d) => write!(f, "Softmax(dim={d})"),
            Self::LayerNorm { .. } => write!(f, "LayerNorm"),
            Self::Embedding(..) => write!(f, "Embedding"),
            Self::CrossEntropyLoss(_, _, d) => write!(f, "CrossEntropyLoss(dim={d})"),
            Self::MseLoss(..) => write!(f, "MseLoss"),
            Self::L1Loss(..) => write!(f, "L1Loss"),
            Self::HuberLoss(_, _, d) => write!(f, "HuberLoss(delta={d})"),
            Self::Dropout(_, _, s) => write!(f, "Dropout(scale={s:.4})"),
            Self::Sin(..) => write!(f, "Sin"),
            Self::Cos(..) => write!(f, "Cos"),
            Self::Recip(..) => write!(f, "Recip"),
            Self::Powf(_, p) => write!(f, "Powf({p})"),
            Self::Clamp(_, lo, hi) => write!(f, "Clamp({lo}, {hi})"),
            Self::Permute(_, p) => write!(f, "Permute({p:?})"),
            Self::RmsNorm { eps, .. } => write!(f, "RmsNorm(eps={eps})"),
            Self::GroupNorm {
                num_groups, eps, ..
            } => {
                write!(f, "GroupNorm(groups={num_groups}, eps={eps})")
            }
            Self::BatchNorm { eps, .. } => write!(f, "BatchNorm(eps={eps})"),
            Self::InstanceNorm { eps, .. } => write!(f, "InstanceNorm(eps={eps})"),
            Self::ConvTranspose1d { .. } => write!(f, "ConvTranspose1d"),
            Self::Elu(_, a) => write!(f, "Elu(alpha={a})"),
            Self::HardSigmoid(..) => write!(f, "HardSigmoid"),
            Self::HardSwish(..) => write!(f, "HardSwish"),
            Self::Mish(..) => write!(f, "Mish"),
            Self::Selu(..) => write!(f, "Selu"),
            Self::Softplus(..) => write!(f, "Softplus"),
            Self::Celu(_, a) => write!(f, "Celu(alpha={a})"),
            Self::LogSoftmax(_, d) => write!(f, "LogSoftmax(dim={d})"),
            Self::Maximum(..) => write!(f, "Maximum"),
            Self::Minimum(..) => write!(f, "Minimum"),
            Self::Stack(_, d) => write!(f, "Stack(dim={d})"),
            Self::MulScalar(_, v) => write!(f, "MulScalar({v})"),
            Self::AddScalar(_, v) => write!(f, "AddScalar({v})"),
            Self::MaxPool1d { .. }
            | Self::MaxPool2d { .. }
            | Self::AvgPool2d { .. }
            | Self::AdaptiveAvgPool2d { .. } => fmt_pool(self, f),
        }
    }
}

#[cfg(test)]
#[path = "op_tests.rs"]
mod tests;
