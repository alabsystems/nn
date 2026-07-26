// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared `TraceOp` classification for compile and verify dispatchers.
//!
//! Both `trace_compile` (nn-dsl -> GPU execution) and `trace_to_graph`
//! (nn-verify -> NY verification) dispatch on `TraceOp`. This
//! module provides a shared classification so both dispatchers agree on
//! op categories and get compile-time coverage when new variants are added.
//!
//! See #2134 for motivation.

use super::{KokoroFusedOp, TraceOp};

/// Classification of a `TraceOp` for dispatch routing.
///
/// Both the compile path (nn-dsl) and verify path (nn-verify) use this
/// to decide how to handle each op. The enum is **not** `#[non_exhaustive]`
/// so adding a new class forces both dispatchers to handle it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TraceOpClass {
    /// Network input placeholder.
    Input,
    /// Identity ops (Dropout at inference).
    Identity,
    /// Binary element-wise: Add, Sub, Mul, Div, Maximum, Minimum.
    BinaryElementwise,
    /// Unary element-wise: Relu, Sigmoid, Exp, Sqrt, etc.
    UnaryElementwise,
    /// Matrix multiply (variable-variable).
    MatMul,
    /// Reduction along a dimension: Sum, Mean, Max, Min.
    Reduction,
    /// Shape manipulation without data movement: Reshape, Unsqueeze, Squeeze.
    ShapeOnly,
    /// Shape ops with data movement: Transpose, Permute, Narrow, Cat, Flip, Expand.
    ShapeDataMove,
    /// Normalization layers: LayerNorm, RmsNorm, GroupNorm, etc.
    Normalization,
    /// Linear / conv layers with weight parameters.
    WeightedLinear,
    /// Attention ops: Softmax, LogSoftmax, SDPA, RoPE, MHA.
    Attention,
    /// Embedding lookup.
    Embedding,
    /// Recurrent layers: LSTM.
    Recurrent,
    /// Pooling layers: AvgPool2d, MaxPool2d, AdaptiveAvgPool2d.
    Pooling,
    /// Selection / indexing: IndexSelect, Gather, WhereCond.
    Indexing,
    /// Scan / accumulation: Cumsum, RepeatInterleave, ScatterAdd, IndexAdd.
    ScanAccumulate,
    /// Type conversion: ToDtype.
    TypeConversion,
    /// Composite ops that decompose to primitives: SwiGlu.
    Composite,
    /// Vision-specific ops: PixelShuffle, Upsample2d, etc.
    Vision,
    /// Quantized ops: QLinear.
    Quantized,
    /// Named activation: Activation { name }.
    NamedActivation,
    /// Clamp: element-wise range restriction.
    Clamp,
    /// Power: element-wise exponentiation.
    Power,
    /// Constant value injection.
    ConstantValue,
    /// Pipeline segment boundary marker (data-dependent ops like length_regulate).
    SegmentBoundary,
    /// Custom / unknown ops.
    Custom,
}

impl TraceOp {
    /// Returns the classification of this operation.
    #[allow(unreachable_patterns)] // #[non_exhaustive] catch-all for future variants
    pub fn classification(&self) -> TraceOpClass {
        match self {
            Self::Input => TraceOpClass::Input,

            Self::Dropout => TraceOpClass::Identity,

            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::Maximum
            | Self::Minimum => TraceOpClass::BinaryElementwise,

            Self::Relu
            | Self::Gelu
            | Self::GeluErf
            | Self::Silu
            | Self::Tanh
            | Self::Sigmoid
            | Self::Exp
            | Self::Log
            | Self::Sqrt
            | Self::Sqr
            | Self::Abs
            | Self::Neg
            | Self::Recip
            | Self::Sin
            | Self::Cos
            | Self::Tan
            | Self::Floor
            | Self::Ceil
            | Self::Round
            | Self::Sign
            | Self::Fract => TraceOpClass::UnaryElementwise,

            Self::MatMul => TraceOpClass::MatMul,

            Self::ReduceSum { .. }
            | Self::ReduceMean { .. }
            | Self::ReduceMax { .. }
            | Self::ReduceMin { .. } => TraceOpClass::Reduction,

            Self::Reshape { .. } | Self::Unsqueeze { .. } | Self::Squeeze { .. } => {
                TraceOpClass::ShapeOnly
            }

            Self::Transpose { .. }
            | Self::Permute { .. }
            | Self::Narrow { .. }
            | Self::Cat { .. }
            | Self::Flip { .. }
            | Self::Roll { .. }
            | Self::Unfold { .. }
            | Self::Expand { .. }
            | Self::SliceSet { .. } => TraceOpClass::ShapeDataMove,

            Self::LayerNorm { .. }
            | Self::RmsNorm { .. }
            | Self::GroupNorm { .. }
            | Self::InstanceNorm { .. }
            | Self::BatchNorm { .. } => TraceOpClass::Normalization,

            Self::Linear { .. }
            | Self::Conv1d { .. }
            | Self::Conv2d { .. }
            | Self::Conv3d { .. }
            | Self::ConvTranspose1d { .. }
            | Self::ConvTranspose2d { .. } => TraceOpClass::WeightedLinear,

            Self::Softmax { .. }
            | Self::LogSoftmax { .. }
            | Self::Sdpa { .. }
            | Self::SdpaCausal { .. }
            | Self::RotaryEmbedding { .. }
            | Self::MultiHeadAttention { .. } => TraceOpClass::Attention,

            Self::Embedding { .. } => TraceOpClass::Embedding,

            Self::Lstm { .. } => TraceOpClass::Recurrent,

            Self::MaxPool1d { .. }
            | Self::AvgPool2d { .. }
            | Self::MaxPool2d { .. }
            | Self::AdaptiveAvgPool2d { .. }
            | Self::AvgPool1d { .. }
            | Self::AdaptiveAvgPool1d { .. }
            | Self::AdaptiveMaxPool2d { .. } => TraceOpClass::Pooling,

            Self::Topk { .. }
            | Self::Argmax { .. }
            | Self::Argmin { .. }
            | Self::ArgSort { .. }
            | Self::Sort { .. }
            | Self::IndexSelect { .. }
            | Self::Gather { .. }
            | Self::WhereCond
            | Self::Compare { .. }
            | Self::CompareTensor { .. } => TraceOpClass::Indexing,

            Self::Cumsum { .. }
            | Self::RepeatInterleave { .. }
            | Self::Scatter { .. }
            | Self::ScatterAdd { .. }
            | Self::IndexAdd { .. }
            | Self::IndexPut { .. } => TraceOpClass::ScanAccumulate,

            Self::ToDtype { .. } => TraceOpClass::TypeConversion,

            Self::SwiGlu | Self::MoeGating { .. } => TraceOpClass::Composite,

            Self::PixelShuffle { .. }
            | Self::PixelUnshuffle { .. }
            | Self::Upsample1d { .. }
            | Self::Upsample2d { .. }
            | Self::ResizeBilinear { .. }
            | Self::Triu { .. }
            | Self::Tril { .. }
            | Self::GridSample { .. } => TraceOpClass::Vision,

            Self::QLinear { .. } => TraceOpClass::Quantized,

            Self::Activation { .. }
            | Self::Elu { .. }
            | Self::LeakyRelu { .. }
            | Self::Softplus
            | Self::Selu
            | Self::Celu { .. }
            | Self::Mish
            | Self::HardSigmoid
            | Self::HardSwish
            | Self::Softsign
            | Self::PRelu { .. } => TraceOpClass::NamedActivation,

            Self::KokoroFused(ref fused) => match fused {
                KokoroFusedOp::SnakeTensor { .. }
                | KokoroFusedOp::AdainSnake { .. }
                | KokoroFusedOp::AdainLeakyRelu { .. } => TraceOpClass::NamedActivation,
                KokoroFusedOp::AdaLayerNorm { .. } => TraceOpClass::Normalization,
                KokoroFusedOp::FusedAdainResBlock { .. } => TraceOpClass::Composite,
                _ => TraceOpClass::Composite,
            },

            Self::Clamp { .. } => TraceOpClass::Clamp,

            Self::Powf { .. } => TraceOpClass::Power,

            Self::Constant { .. } | Self::ConstantWeight { .. } => {
                TraceOpClass::ConstantValue
            }

            Self::SegmentBoundary { .. } => TraceOpClass::SegmentBoundary,

            Self::Custom { .. } => TraceOpClass::Custom,

            // #[non_exhaustive] catch-all — new variants default to Custom
            // until explicitly classified. Both dispatchers will see Custom
            // and return an unsupported error, making the gap visible.
            _ => TraceOpClass::Custom,
        }
    }

    /// Expected number of tensor inputs for this operation.
    ///
    /// Returns `None` for unknown `#[non_exhaustive]` variants so callers
    /// can detect missing arity arms instead of silently using a wrong default.
    /// Variable-arity ops (Cat) return their specific count from their fields.
    #[allow(unreachable_patterns)] // #[non_exhaustive] catch-all for future variants
    pub fn expected_arity(&self) -> Option<usize> {
        match self {
            Self::Input | Self::Constant { .. } => Some(0),

            // Unary
            Self::Relu
            | Self::Gelu
            | Self::GeluErf
            | Self::Silu
            | Self::Tanh
            | Self::Sigmoid
            | Self::Exp
            | Self::Log
            | Self::Sqrt
            | Self::Sqr
            | Self::Abs
            | Self::Neg
            | Self::Recip
            | Self::Sin
            | Self::Cos
            | Self::Tan
            | Self::Floor
            | Self::Ceil
            | Self::Round
            | Self::Sign
            | Self::Fract
            | Self::Dropout
            | Self::Reshape { .. }
            | Self::Transpose { .. }
            | Self::Narrow { .. }
            | Self::Unsqueeze { .. }
            | Self::Squeeze { .. }
            | Self::Permute { .. }
            | Self::Flip { .. }
            | Self::Unfold { .. }
            | Self::Expand { .. }
            | Self::ReduceSum { .. }
            | Self::ReduceMean { .. }
            | Self::ReduceMax { .. }
            | Self::ReduceMin { .. }
            | Self::LayerNorm { .. }
            | Self::RmsNorm { .. }
            | Self::GroupNorm { .. }
            | Self::InstanceNorm { .. }
            | Self::BatchNorm { .. }
            | Self::Linear { .. }
            | Self::QLinear { .. }
            | Self::Embedding { .. }
            | Self::Softmax { .. }
            | Self::LogSoftmax { .. }
            | Self::Clamp { .. }
            | Self::Powf { .. }
            | Self::ToDtype { .. }
            | Self::SwiGlu
            | Self::MoeGating { .. }
            | Self::Activation { .. }
            | Self::Elu { .. }
            | Self::LeakyRelu { .. }
            | Self::Softplus
            | Self::Selu
            | Self::Celu { .. }
            | Self::Mish
            | Self::HardSigmoid
            | Self::HardSwish
            | Self::Softsign
            | Self::PRelu { .. }
            | Self::Conv1d { .. }
            | Self::Conv2d { .. }
            | Self::Conv3d { .. }
            | Self::ConvTranspose1d { .. }
            | Self::ConvTranspose2d { .. }
            | Self::MaxPool1d { .. }
            | Self::AvgPool2d { .. }
            | Self::MaxPool2d { .. }
            | Self::AdaptiveAvgPool2d { .. }
            | Self::AvgPool1d { .. }
            | Self::AdaptiveAvgPool1d { .. }
            | Self::AdaptiveMaxPool2d { .. }
            | Self::PixelShuffle { .. }
            | Self::PixelUnshuffle { .. }
            | Self::Upsample1d { .. }
            | Self::Upsample2d { .. }
            | Self::ResizeBilinear { .. }
            | Self::Cumsum { .. }
            | Self::Topk { .. }
            | Self::Argmax { .. }
            | Self::Argmin { .. }
            | Self::ArgSort { .. }
            | Self::Sort { .. }
            | Self::Roll { .. }
            | Self::Triu { .. }
            | Self::Tril { .. }
            | Self::Compare { .. } => Some(1),

            // Binary
            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::Maximum
            | Self::Minimum
            | Self::MatMul
            | Self::IndexSelect { .. }
            | Self::Gather { .. }
            | Self::RepeatInterleave { .. }
            | Self::GridSample { .. }
            | Self::SliceSet { .. }
            | Self::CompareTensor { .. } => Some(2),

            // Ternary
            Self::WhereCond
            | Self::Scatter { .. }
            | Self::ScatterAdd { .. }
            | Self::IndexAdd { .. }
            | Self::IndexPut { .. } => Some(3),

            Self::KokoroFused(ref fused) => match fused {
                KokoroFusedOp::SnakeTensor { .. } => Some(1),
                KokoroFusedOp::AdainSnake { .. }
                | KokoroFusedOp::AdainLeakyRelu { .. }
                | KokoroFusedOp::AdaLayerNorm { .. } => Some(3),
                KokoroFusedOp::FusedAdainResBlock { .. } => Some(2),
                _ => None,
            },

            // LSTM: input + h_state + c_state
            Self::Lstm { .. } => Some(3),

            // Attention: Q + K + V (+ optional mask) — variable arity (3 or 4)
            Self::Sdpa { .. } => None,
            // SdpaCausal: always exactly 3 inputs (Q, K, V), no mask tensor.
            Self::SdpaCausal { .. } => Some(3),
            Self::RotaryEmbedding { .. } => Some(1),
            Self::MultiHeadAttention { .. } => Some(3),

            // Variable arity
            Self::Cat { num_inputs, .. } => Some(*num_inputs),

            // Segment boundary: passthrough (1 input)
            Self::SegmentBoundary { .. } => Some(1),

            // Custom: unknown
            Self::Custom { .. } => Some(1),

            // #[non_exhaustive] catch-all — returns None so callers detect
            // missing arity arms instead of silently using a wrong default.
            _ => None,
        }
    }
}
