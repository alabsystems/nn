// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Op classification for graph partition fusion.
//!
//! Classifies every `TraceOp` variant into a [`FusionCategory`] that
//! determines how the op participates in graph-global fusion. The partition
//! algorithm in [`super::partition`] uses these categories to decide which
//! ops can be fused into a single GPU dispatch.

use nn_core::dyn_tensor::trace::TraceOp;

/// Classification of a trace op for fusion decisions.
///
/// The partition algorithm uses these categories with the following rules:
/// - Elementwise + Elementwise → fuse (loop fusion)
/// - Elementwise + Reduction → fuse (reduce absorbs producers)
/// - Broadcast + anything → fuse (zero-cost metadata)
/// - Opaque + Elementwise → fuse (epilogue/prologue absorption)
/// - Opaque + Opaque → barrier (separate dispatches)
/// - Reduction + Reduction → barrier
/// - Native → barrier (already optimal)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum FusionCategory {
    /// Element-wise ops that can be loop-fused: unary math, binary arithmetic,
    /// activations. Shape-preserving (or broadcast-compatible).
    Elementwise,
    /// Shape-only ops: reshape, squeeze, unsqueeze, transpose, permute, expand,
    /// flip, narrow, slice. Zero-cost metadata operations.
    Broadcast,
    /// Reduction ops: sum, mean, max, min, softmax, log_softmax.
    /// Create mandatory kernel boundaries but absorb elementwise producers.
    Reduction,
    /// Opaque compute ops: linear, conv, matmul, LSTM, attention, embedding.
    /// Cannot fuse with each other but absorb elementwise epilogues/prologues.
    Opaque,
    /// Pre-fused native ops, segment boundaries, or ops that should not
    /// participate in fusion. Already optimal or not fusible.
    Native,
}

/// Classify a `TraceOp` for graph partition fusion decisions.
pub(crate) fn fusion_category(op: &TraceOp) -> FusionCategory {
    use FusionCategory::{Elementwise, Broadcast, Reduction, Opaque, Native};
    match op {
        // -- Elementwise (unary math) --
        TraceOp::Exp
        | TraceOp::Log
        | TraceOp::Sqrt
        | TraceOp::Sqr
        | TraceOp::Abs
        | TraceOp::Neg
        | TraceOp::Recip
        | TraceOp::Sin
        | TraceOp::Cos
        | TraceOp::Tan
        | TraceOp::Floor
        | TraceOp::Ceil
        | TraceOp::Round
        | TraceOp::Sign
        | TraceOp::Fract
        | TraceOp::Tanh => Elementwise,

        // -- Elementwise (activations) --
        TraceOp::Relu
        | TraceOp::Gelu
        | TraceOp::GeluErf
        | TraceOp::Sigmoid
        | TraceOp::Silu
        | TraceOp::Softplus
        | TraceOp::Selu
        | TraceOp::Mish
        | TraceOp::HardSigmoid
        | TraceOp::HardSwish
        | TraceOp::Softsign => Elementwise,

        // -- Elementwise (parameterized activations) --
        TraceOp::LeakyRelu { .. }
        | TraceOp::Elu { .. }
        | TraceOp::Celu { .. }
        | TraceOp::PRelu { .. } => Elementwise,

        // -- Elementwise (binary arithmetic) --
        TraceOp::Add
        | TraceOp::Sub
        | TraceOp::Mul
        | TraceOp::Div
        | TraceOp::Maximum
        | TraceOp::Minimum
        | TraceOp::Atan2 => Elementwise,

        // -- Elementwise (misc) --
        TraceOp::Clamp { .. }
        | TraceOp::Powf { .. }
        | TraceOp::Compare { .. }
        | TraceOp::CompareTensor { .. }
        | TraceOp::WhereCond => Elementwise,

        // -- Broadcast (shape-only, zero-cost) --
        TraceOp::Reshape { .. }
        | TraceOp::Transpose { .. }
        | TraceOp::Narrow { .. }
        | TraceOp::Unsqueeze { .. }
        | TraceOp::Squeeze { .. }
        | TraceOp::Permute { .. }
        | TraceOp::Expand { .. }
        | TraceOp::Flip { .. } => Broadcast,

        // -- Reduction --
        TraceOp::ReduceSum { .. }
        | TraceOp::ReduceMean { .. }
        | TraceOp::ReduceMax { .. }
        | TraceOp::ReduceMin { .. }
        | TraceOp::Softmax { .. }
        | TraceOp::LogSoftmax { .. }
        | TraceOp::Cumsum { .. } => Reduction,

        // -- Opaque compute (cannot fuse with each other) --
        TraceOp::Linear { .. }
        | TraceOp::Conv1d { .. }
        | TraceOp::Conv2d { .. }
        | TraceOp::Conv3d { .. }
        | TraceOp::ConvTranspose1d { .. }
        | TraceOp::ConvTranspose2d { .. }
        | TraceOp::MatMul
        | TraceOp::Sdpa { .. }
        | TraceOp::SdpaCausal { .. }
        | TraceOp::Embedding { .. }
        | TraceOp::Lstm { .. }
        | TraceOp::QLinear { .. }
        | TraceOp::MoeGating { .. } => Opaque,

        // -- Opaque (normalization — has internal reductions) --
        TraceOp::LayerNorm { .. }
        | TraceOp::RmsNorm { .. }
        | TraceOp::GroupNorm { .. }
        | TraceOp::InstanceNorm { .. }
        | TraceOp::BatchNorm { .. } => Opaque,

        // -- Opaque (pooling) --
        TraceOp::MaxPool1d { .. }
        | TraceOp::AvgPool2d { .. }
        | TraceOp::MaxPool2d { .. }
        | TraceOp::AdaptiveAvgPool2d { .. } => Opaque,

        // -- Opaque (vision spatial ops) --
        TraceOp::PixelShuffle { .. }
        | TraceOp::PixelUnshuffle { .. }
        | TraceOp::Upsample1d { .. }
        | TraceOp::Upsample2d { .. }
        | TraceOp::ResizeBilinear { .. }
        | TraceOp::GridSample { .. } => Opaque,

        // -- Opaque (selection/indexing — data-dependent) --
        TraceOp::Topk { .. }
        | TraceOp::Argmax { .. }
        | TraceOp::Argmin { .. }
        | TraceOp::ArgSort { .. }
        | TraceOp::Sort { .. }
        | TraceOp::IndexSelect { .. }
        | TraceOp::Gather { .. }
        | TraceOp::RepeatInterleave { .. }
        | TraceOp::Scatter { .. }
        | TraceOp::ScatterAdd { .. }
        | TraceOp::IndexAdd { .. }
        | TraceOp::IndexPut { .. }
        | TraceOp::SliceSet { .. }
        | TraceOp::Unfold { .. }
        | TraceOp::Roll { .. }
        | TraceOp::Triu { .. }
        | TraceOp::Tril { .. } => Opaque,

        // -- Opaque (padding — changes shape with data copy) --
        TraceOp::ReflectionPad1d { .. }
        | TraceOp::ReflectionPad2d { .. }
        | TraceOp::ConstantPadNd { .. } => Opaque,

        // -- Opaque (multi-head attention, concat, rotary, dtype) --
        TraceOp::MultiHeadAttention { .. }
        | TraceOp::RotaryEmbedding { .. }
        | TraceOp::Cat { .. }
        | TraceOp::ToDtype { .. }
        | TraceOp::Arange { .. } => Opaque,

        // -- Native / non-fusible --
        TraceOp::Input
        | TraceOp::ConstantWeight { .. }
        | TraceOp::Constant { .. }
        | TraceOp::Dropout
        | TraceOp::Activation { .. }
        | TraceOp::KokoroFused(_)
        | TraceOp::SwiGlu
        | TraceOp::SegmentBoundary { .. }
        | TraceOp::Custom { .. } => Native,

        // TraceOp is #[non_exhaustive] — treat unknown variants as Native
        // (safe default: no fusion, separate dispatch).
        _ => Native,
    }
}

/// Returns `true` if ops in category `producer` can fuse with a `consumer`.
pub(crate) fn can_fuse(producer: FusionCategory, consumer: FusionCategory) -> bool {
    use FusionCategory::{Elementwise, Reduction, Opaque, Broadcast};
    matches!(
        (producer, consumer),
        // Elementwise chains
        (Elementwise, Elementwise)
        // Reduction absorbs elementwise producers
        | (Elementwise, Reduction)
        // Opaque epilogue: matmul + bias + activation
        | (Opaque, Elementwise)
        // Opaque prologue: input transforms before compute
        | (Elementwise, Opaque)
        // Broadcast always fuses (zero-cost metadata)
        | (Broadcast, Elementwise)
        | (Broadcast, Reduction)
        | (Broadcast, Opaque)
        | (Broadcast, Broadcast)
        | (Elementwise, Broadcast)
        | (Reduction, Broadcast)
        | (Opaque, Broadcast)
        // Reduction epilogue: normalize after reduce
        | (Reduction, Elementwise)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elementwise_classification() {
        assert_eq!(fusion_category(&TraceOp::Relu), FusionCategory::Elementwise);
        assert_eq!(fusion_category(&TraceOp::Add), FusionCategory::Elementwise);
        assert_eq!(fusion_category(&TraceOp::Exp), FusionCategory::Elementwise);
        assert_eq!(
            fusion_category(&TraceOp::Clamp {
                min: Some(0.0),
                max: None
            }),
            FusionCategory::Elementwise
        );
    }

    #[test]
    fn test_broadcast_classification() {
        assert_eq!(
            fusion_category(&TraceOp::Reshape {
                target_shape: vec![1, 2, 3]
            }),
            FusionCategory::Broadcast
        );
        assert_eq!(
            fusion_category(&TraceOp::Transpose { dim0: 0, dim1: 1 }),
            FusionCategory::Broadcast
        );
    }

    #[test]
    fn test_reduction_classification() {
        assert_eq!(
            fusion_category(&TraceOp::ReduceSum {
                dim: 0,
                keepdim: false
            }),
            FusionCategory::Reduction
        );
        assert_eq!(
            fusion_category(&TraceOp::Softmax { dim: 1 }),
            FusionCategory::Reduction
        );
    }

    #[test]
    fn test_opaque_classification() {
        use nn_core::dyn_tensor::trace::WeightRef;
        let w = WeightRef::new(vec![1.0], vec![1]).unwrap();
        assert_eq!(
            fusion_category(&TraceOp::Linear {
                weight: w.clone(),
                bias: None
            }),
            FusionCategory::Opaque
        );
        assert_eq!(fusion_category(&TraceOp::MatMul), FusionCategory::Opaque);
        assert_eq!(
            fusion_category(&TraceOp::LayerNorm {
                eps: 1e-5,
                weight: w.clone(),
                bias: w,
            }),
            FusionCategory::Opaque
        );
    }

    #[test]
    fn test_native_classification() {
        assert_eq!(fusion_category(&TraceOp::Input), FusionCategory::Native);
        assert_eq!(fusion_category(&TraceOp::Dropout), FusionCategory::Native);
    }

    #[test]
    fn test_can_fuse_rules() {
        use FusionCategory::*;
        // Elementwise chains
        assert!(can_fuse(Elementwise, Elementwise));
        // Reduce absorbs elementwise
        assert!(can_fuse(Elementwise, Reduction));
        // Opaque epilogue
        assert!(can_fuse(Opaque, Elementwise));
        // Broadcast always fuses
        assert!(can_fuse(Broadcast, Elementwise));
        assert!(can_fuse(Broadcast, Opaque));
        // Barriers
        assert!(!can_fuse(Opaque, Opaque));
        assert!(!can_fuse(Reduction, Reduction));
        assert!(!can_fuse(Native, Elementwise));
        assert!(!can_fuse(Elementwise, Native));
    }
}
