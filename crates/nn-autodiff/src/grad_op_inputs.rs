// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extract input nodes from an autodiff `Op`.
//!
//! Extracted from grad.rs to keep it under the 500-line limit.

use std::sync::Arc;

use crate::tracked::TrackedTensor;

/// Extract input nodes from an Op.
pub(super) fn op_inputs(op: &crate::op::Op) -> Vec<Arc<TrackedTensor>> {
    use crate::op::Op;
    match op {
        Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::Div(a, b) | Op::MatMul(a, b) => {
            vec![Arc::clone(a), Arc::clone(b)]
        }
        Op::Relu(x)
        | Op::Gelu(x)
        | Op::GeluErf(x)
        | Op::Silu(x)
        | Op::Tanh(x)
        | Op::Sigmoid(x)
        | Op::Exp(x)
        | Op::Log(x)
        | Op::Sqrt(x)
        | Op::Sqr(x)
        | Op::Neg(x)
        | Op::Abs(x)
        | Op::Sin(x)
        | Op::Cos(x)
        | Op::Recip(x)
        | Op::Powf(x, _)
        | Op::Clamp(x, _, _)
        | Op::Elu(x, _)
        | Op::HardSigmoid(x)
        | Op::HardSwish(x)
        | Op::Mish(x)
        | Op::Selu(x)
        | Op::Softplus(x)
        | Op::Celu(x, _) => vec![Arc::clone(x)],

        Op::SumKeepDim(x, _)
        | Op::MeanKeepDim(x, _)
        | Op::Reshape(x, _)
        | Op::Transpose(x, _, _)
        | Op::Narrow(x, _, _, _)
        | Op::Broadcast(x, _)
        | Op::Unsqueeze(x, _)
        | Op::Squeeze(x, _)
        | Op::Unfold(x, _, _, _)
        | Op::Permute(x, _)
        | Op::Softmax(x, _)
        | Op::LogSoftmax(x, _) => vec![Arc::clone(x)],

        Op::Maximum(a, b) | Op::Minimum(a, b) => vec![Arc::clone(a), Arc::clone(b)],

        Op::Conv1d { input, kernel, .. }
        | Op::Conv2d { input, kernel, .. }
        | Op::ConvTranspose1d { input, kernel, .. } => {
            vec![Arc::clone(input), Arc::clone(kernel)]
        }

        Op::Cat(inputs, _) | Op::Stack(inputs, _) => inputs.iter().map(Arc::clone).collect(),

        Op::LayerNorm {
            input,
            weight,
            bias,
            ..
        }
        | Op::GroupNorm {
            input,
            weight,
            bias,
            ..
        }
        | Op::BatchNorm {
            input,
            weight,
            bias,
            ..
        }
        | Op::InstanceNorm {
            input,
            weight,
            bias,
            ..
        } => vec![Arc::clone(input), Arc::clone(weight), Arc::clone(bias)],
        Op::RmsNorm { input, weight, .. } => {
            vec![Arc::clone(input), Arc::clone(weight)]
        }
        Op::Embedding(weight, indices) => vec![Arc::clone(weight), Arc::clone(indices)],
        Op::CrossEntropyLoss(input, targets, _)
        | Op::MseLoss(input, targets)
        | Op::L1Loss(input, targets)
        | Op::HuberLoss(input, targets, _) => {
            vec![Arc::clone(input), Arc::clone(targets)]
        }
        Op::MulScalar(x, _) | Op::AddScalar(x, _) => vec![Arc::clone(x)],
        Op::Dropout(x, mask, _) => vec![Arc::clone(x), Arc::clone(mask)],
        Op::MaxPool1d { input, .. }
        | Op::MaxPool2d { input, .. }
        | Op::AdaptiveAvgPool2d { input, .. }
        | Op::AvgPool2d { input, .. } => vec![Arc::clone(input)],
    }
}
