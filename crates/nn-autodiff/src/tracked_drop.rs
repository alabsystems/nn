// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Iterative drop for [`TrackedTensor`] to prevent stack overflow on deep
//! computation graphs.
//!
//! Without this, dropping a chain of 10,000+ nodes causes recursive drop
//! through `TrackedTensor → Op → Arc<TrackedTensor> → TrackedTensor → ...`,
//! consuming one stack frame per node. The custom [`Drop`] impl collects child
//! `Arc` references iteratively using a heap-allocated `Vec` instead.

use super::TrackedTensor;
use crate::op::Op;
use std::sync::Arc;

impl Drop for TrackedTensor {
    fn drop(&mut self) {
        // Take ownership of the op to break the recursive drop chain.
        // If we don't have an op, there's nothing to worry about.
        let Some(op) = self.op.take() else {
            return;
        };

        // Collect all Arc<TrackedTensor> from this op into a work queue.
        let mut to_drop: Vec<Arc<Self>> = op_arc_extract(op);

        // Iteratively process the queue: for each Arc that we are the sole
        // owner of (strong_count == 1), take its op's children before it drops.
        while let Some(arc) = to_drop.pop() {
            // If there are other references, skip — the node won't be dropped
            // when we release our reference.
            if Arc::strong_count(&arc) > 1 {
                // Just drop our reference (decrement refcount). No recursion
                // because the node stays alive.
                drop(arc);
                continue;
            }

            // We hold the last Arc. When we drop it, the TrackedTensor will
            // be dropped. We need to extract its children FIRST to prevent
            // the automatic Drop from recursing.
            //
            // Arc::try_unwrap succeeds because strong_count == 1.
            if let Ok(mut tensor) = Arc::try_unwrap(arc) {
                if let Some(child_op) = tensor.op.take() {
                    to_drop.extend(op_arc_extract(child_op));
                }
                // tensor drops here with op=None, so no recursive drop
            }
            // If try_unwrap fails (race condition in multi-threaded context),
            // the Arc drops normally — acceptable since another thread holds it.
        }
    }
}

/// Extract all `Arc<TrackedTensor>` references from an `Op`.
///
/// This is separate from `op_inputs` in grad.rs because we need owned Arcs,
/// and we don't need the function to be generic over Op lifetimes.
fn op_arc_extract(op: Op) -> Vec<Arc<TrackedTensor>> {
    // No catch-all `_ =>` — exhaustive matching ensures new Op variants that
    // carry Arc<TrackedTensor> children will cause a compile error here,
    // preventing silent gradient propagation failures in the Drop cleanup.
    match op {
        Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::Div(a, b) | Op::MatMul(a, b) => {
            vec![a, b]
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
        | Op::Celu(x, _) => vec![x],

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
        | Op::LogSoftmax(x, _) => vec![x],

        Op::Maximum(a, b) | Op::Minimum(a, b) => vec![a, b],

        Op::Conv1d { input, kernel, .. }
        | Op::Conv2d { input, kernel, .. }
        | Op::ConvTranspose1d { input, kernel, .. } => {
            vec![input, kernel]
        }

        Op::Cat(inputs, _) | Op::Stack(inputs, _) => inputs,

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
        } => vec![input, weight, bias],
        Op::RmsNorm { input, weight, .. } => {
            vec![input, weight]
        }
        Op::Embedding(weight, indices) => vec![weight, indices],
        Op::CrossEntropyLoss(input, targets, _)
        | Op::MseLoss(input, targets)
        | Op::L1Loss(input, targets)
        | Op::HuberLoss(input, targets, _) => {
            vec![input, targets]
        }
        Op::MulScalar(x, _) | Op::AddScalar(x, _) => vec![x],
        Op::Dropout(x, mask, _) => vec![x, mask],
        Op::MaxPool1d { input, .. }
        | Op::MaxPool2d { input, .. }
        | Op::AdaptiveAvgPool2d { input, .. }
        | Op::AvgPool2d { input, .. } => vec![input],
    }
}
