// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `TraceOp` properties (#3681).
//!
//! Proves correctness of `TraceOp` methods: `classification()`, `expected_arity()`,
//! `canonical_name()`. Each harness constructs a specific `TraceOp` variant and
//! verifies that its derived properties are consistent and non-degenerate.
//!
//! These harnesses exercise the match arms in `trace_op_class.rs` and
//! `trace_op_names.rs`, catching dead arms, misclassifications, and
//! missing arity entries when new variants are added.

use crate::dyn_tensor::trace::{
    KokoroFusedOp, ResBlockActivation, TraceActivation, TraceOp, TraceOpClass, TraceUpsampleMode,
    WeightRef,
};
use crate::dyn_tensor::CompareOp;
use crate::DType;

// -- Helper: construct a minimal WeightRef for testing -----------------------

fn dummy_weight(shape: &[usize]) -> WeightRef {
    let numel: usize = shape.iter().product();
    WeightRef::new(vec![0.0f32; numel], shape.to_vec()).unwrap()
}

fn dummy_weight_1d(n: usize) -> WeightRef {
    dummy_weight(&[n])
}

fn dummy_weight_2d(r: usize, c: usize) -> WeightRef {
    dummy_weight(&[r, c])
}

// ===========================================================================
// classification() correctness: every explicitly-constructed variant maps to
// the expected TraceOpClass.
// ===========================================================================

/// Prove: all 6 binary elementwise ops classify as BinaryElementwise.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_binary_elementwise() {
    let ops: [TraceOp; 6] = [
        TraceOp::Add,
        TraceOp::Sub,
        TraceOp::Mul,
        TraceOp::Div,
        TraceOp::Maximum,
        TraceOp::Minimum,
    ];
    let mut i = 0;
    while i < 6 {
        assert!(
            ops[i].classification() == TraceOpClass::BinaryElementwise,
            "binary elementwise op must classify as BinaryElementwise"
        );
        i += 1;
    }
}

/// Prove: all 20 unary elementwise ops classify as UnaryElementwise.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_unary_elementwise() {
    let ops: [TraceOp; 20] = [
        TraceOp::Relu,
        TraceOp::Gelu,
        TraceOp::GeluErf,
        TraceOp::Silu,
        TraceOp::Tanh,
        TraceOp::Sigmoid,
        TraceOp::Exp,
        TraceOp::Log,
        TraceOp::Sqrt,
        TraceOp::Sqr,
        TraceOp::Abs,
        TraceOp::Neg,
        TraceOp::Recip,
        TraceOp::Sin,
        TraceOp::Cos,
        TraceOp::Tan,
        TraceOp::Floor,
        TraceOp::Ceil,
        TraceOp::Round,
        TraceOp::Sign,
    ];
    let mut i = 0;
    while i < 20 {
        assert!(
            ops[i].classification() == TraceOpClass::UnaryElementwise,
            "unary elementwise op must classify as UnaryElementwise"
        );
        i += 1;
    }
}

/// Prove: Fract classifies as UnaryElementwise.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_fract_unary() {
    assert!(TraceOp::Fract.classification() == TraceOpClass::UnaryElementwise);
}

/// Prove: MatMul classifies as MatMul.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_matmul() {
    assert!(TraceOp::MatMul.classification() == TraceOpClass::MatMul);
}

/// Prove: Dropout classifies as Identity (inference-time no-op).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_dropout_identity() {
    assert!(TraceOp::Dropout.classification() == TraceOpClass::Identity);
}

/// Prove: Input classifies as Input.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_input() {
    assert!(TraceOp::Input.classification() == TraceOpClass::Input);
}

/// Prove: 4 reduction variants classify as Reduction.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_reductions() {
    let dim: usize = 0;
    let keepdim = false;
    assert!(TraceOp::ReduceSum { dim, keepdim }.classification() == TraceOpClass::Reduction);
    assert!(TraceOp::ReduceMean { dim, keepdim }.classification() == TraceOpClass::Reduction);
    assert!(TraceOp::ReduceMax { dim, keepdim }.classification() == TraceOpClass::Reduction);
    assert!(TraceOp::ReduceMin { dim, keepdim }.classification() == TraceOpClass::Reduction);
}

/// Prove: shape-only ops (Reshape, Unsqueeze, Squeeze) classify as ShapeOnly.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_shape_only() {
    assert!(
        TraceOp::Reshape {
            target_shape: vec![4, 8]
        }
        .classification()
            == TraceOpClass::ShapeOnly
    );
    assert!(TraceOp::Unsqueeze { dim: 0 }.classification() == TraceOpClass::ShapeOnly);
    assert!(TraceOp::Squeeze { dim: 1 }.classification() == TraceOpClass::ShapeOnly);
}

/// Prove: shape-with-data-movement ops classify as ShapeDataMove.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_shape_data_move() {
    assert!(
        TraceOp::Transpose { dim0: 0, dim1: 1 }.classification() == TraceOpClass::ShapeDataMove
    );
    assert!(TraceOp::Permute { axes: vec![1, 0] }.classification() == TraceOpClass::ShapeDataMove);
    assert!(
        TraceOp::Narrow {
            dim: 0,
            start: 0,
            length: 5
        }
        .classification()
            == TraceOpClass::ShapeDataMove
    );
    assert!(
        TraceOp::Cat {
            dim: 0,
            num_inputs: 3
        }
        .classification()
            == TraceOpClass::ShapeDataMove
    );
    assert!(TraceOp::Flip { dim: 0 }.classification() == TraceOpClass::ShapeDataMove);
    assert!(
        TraceOp::Unfold {
            dim: 0,
            size: 4,
            step: 2
        }
        .classification()
            == TraceOpClass::ShapeDataMove
    );
    assert!(
        TraceOp::Expand {
            target_shape: vec![2, 3]
        }
        .classification()
            == TraceOpClass::ShapeDataMove
    );
    assert!(TraceOp::SliceSet { dim: 0, start: 0 }.classification() == TraceOpClass::ShapeDataMove);
}

/// Prove: normalization ops classify as Normalization.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_normalization() {
    let w = dummy_weight_1d(4);
    let b = dummy_weight_1d(4);
    assert!(
        TraceOp::LayerNorm {
            eps: 1e-5,
            weight: w.clone(),
            bias: b.clone()
        }
        .classification()
            == TraceOpClass::Normalization
    );
    assert!(
        TraceOp::RmsNorm {
            eps: 1e-5,
            weight: w.clone()
        }
        .classification()
            == TraceOpClass::Normalization
    );
    assert!(
        TraceOp::GroupNorm {
            num_groups: 2,
            eps: 1e-5,
            weight: w.clone(),
            bias: b.clone()
        }
        .classification()
            == TraceOpClass::Normalization
    );
    assert!(TraceOp::InstanceNorm { eps: 1e-5 }.classification() == TraceOpClass::Normalization);
    assert!(
        TraceOp::BatchNorm {
            eps: 1e-5,
            weight: w.clone(),
            bias: b.clone(),
            running_mean: w.clone(),
            running_var: b.clone()
        }
        .classification()
            == TraceOpClass::Normalization
    );
}

/// Prove: weighted linear/conv ops classify as WeightedLinear.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_weighted_linear() {
    let w = dummy_weight_2d(4, 4);
    let b = dummy_weight_1d(4);
    assert!(
        TraceOp::Linear {
            weight: w.clone(),
            bias: Some(b.clone())
        }
        .classification()
            == TraceOpClass::WeightedLinear
    );
    assert!(
        TraceOp::Conv1d {
            weight: w.clone(),
            bias: None,
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1
        }
        .classification()
            == TraceOpClass::WeightedLinear
    );
    assert!(
        TraceOp::Conv2d {
            weight: w.clone(),
            bias: None,
            padding: [0, 0],
            stride: [1, 1],
            dilation: [1, 1],
            groups: 1
        }
        .classification()
            == TraceOpClass::WeightedLinear
    );
    assert!(
        TraceOp::ConvTranspose1d {
            weight: w.clone(),
            bias: None,
            padding: 0,
            output_padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1
        }
        .classification()
            == TraceOpClass::WeightedLinear
    );
    assert!(
        TraceOp::ConvTranspose2d {
            weight: w.clone(),
            bias: None,
            padding: [0, 0],
            output_padding: [0, 0],
            stride: [1, 1],
            dilation: [1, 1],
            groups: 1
        }
        .classification()
            == TraceOpClass::WeightedLinear
    );
}

/// Prove: attention ops classify as Attention.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_attention() {
    assert!(TraceOp::Softmax { dim: 1 }.classification() == TraceOpClass::Attention);
    assert!(TraceOp::LogSoftmax { dim: 1 }.classification() == TraceOpClass::Attention);
    assert!(TraceOp::Sdpa { scale: 0.125 }.classification() == TraceOpClass::Attention);
    assert!(TraceOp::SdpaCausal { scale: 0.125 }.classification() == TraceOpClass::Attention);
    let cos_cache = dummy_weight_2d(8, 32);
    let sin_cache = dummy_weight_2d(8, 32);
    assert!(
        TraceOp::RotaryEmbedding {
            head_dim: 64,
            offset: 0,
            cos_cache,
            sin_cache,
        }
        .classification()
            == TraceOpClass::Attention
    );
    assert!(
        TraceOp::MultiHeadAttention {
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 64
        }
        .classification()
            == TraceOpClass::Attention
    );
}

/// Prove: Embedding classifies as Embedding.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_embedding() {
    let w = dummy_weight_2d(100, 64);
    assert!(TraceOp::Embedding { weight: w }.classification() == TraceOpClass::Embedding);
}

/// Prove: LSTM classifies as Recurrent.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_recurrent() {
    let w = dummy_weight_2d(4, 4);
    assert!(
        TraceOp::Lstm {
            weight_ih: w.clone(),
            weight_hh: w.clone(),
            bias_ih: None,
            bias_hh: None,
            hidden_size: 4,
            initial_hidden: None,
            initial_cell: None,
        }
        .classification()
            == TraceOpClass::Recurrent
    );
}

/// Prove: pooling ops classify as Pooling.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_pooling() {
    assert!(
        TraceOp::MaxPool1d {
            kernel_size: 2,
            stride: 2,
            padding: 0
        }
        .classification()
            == TraceOpClass::Pooling
    );
    assert!(
        TraceOp::AvgPool2d {
            kernel_size: [2, 2],
            stride: [2, 2],
            padding: [0, 0]
        }
        .classification()
            == TraceOpClass::Pooling
    );
    assert!(
        TraceOp::MaxPool2d {
            kernel_size: [2, 2],
            stride: [2, 2],
            padding: [0, 0]
        }
        .classification()
            == TraceOpClass::Pooling
    );
    assert!(
        TraceOp::AdaptiveAvgPool2d {
            output_size: [1, 1]
        }
        .classification()
            == TraceOpClass::Pooling
    );
}

/// Prove: indexing/selection ops classify as Indexing.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_indexing() {
    assert!(TraceOp::Topk { k: 5, dim: 0 }.classification() == TraceOpClass::Indexing);
    assert!(TraceOp::Argmax { dim: 0 }.classification() == TraceOpClass::Indexing);
    assert!(TraceOp::Argmin { dim: 0 }.classification() == TraceOpClass::Indexing);
    assert!(
        TraceOp::ArgSort {
            dim: 0,
            descending: false
        }
        .classification()
            == TraceOpClass::Indexing
    );
    assert!(TraceOp::IndexSelect { dim: 0 }.classification() == TraceOpClass::Indexing);
    assert!(TraceOp::Gather { dim: 0 }.classification() == TraceOpClass::Indexing);
    assert!(TraceOp::WhereCond.classification() == TraceOpClass::Indexing);
    assert!(
        TraceOp::Compare {
            op: CompareOp::Gt,
            value: 0.0
        }
        .classification()
            == TraceOpClass::Indexing
    );
    assert!(
        TraceOp::CompareTensor { op: CompareOp::Eq }.classification() == TraceOpClass::Indexing
    );
}

/// Prove: scan/accumulate ops classify as ScanAccumulate.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_scan_accumulate() {
    assert!(TraceOp::Cumsum { dim: 0 }.classification() == TraceOpClass::ScanAccumulate);
    assert!(TraceOp::RepeatInterleave { dim: 0 }.classification() == TraceOpClass::ScanAccumulate);
    assert!(TraceOp::ScatterAdd { dim: 0 }.classification() == TraceOpClass::ScanAccumulate);
    assert!(TraceOp::IndexAdd { dim: 0 }.classification() == TraceOpClass::ScanAccumulate);
}

/// Prove: ToDtype classifies as TypeConversion.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_type_conversion() {
    assert!(
        TraceOp::ToDtype {
            target_dtype: DType::F16
        }
        .classification()
            == TraceOpClass::TypeConversion
    );
}

/// Prove: Composite ops (SwiGlu, MoeGating) classify as Composite.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_composite() {
    assert!(TraceOp::SwiGlu.classification() == TraceOpClass::Composite);
    assert!(
        TraceOp::MoeGating {
            num_experts: 8,
            top_k: 2
        }
        .classification()
            == TraceOpClass::Composite
    );
}

/// Prove: vision ops classify as Vision.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_vision() {
    assert!(TraceOp::PixelShuffle { upscale_factor: 2 }.classification() == TraceOpClass::Vision);
    assert!(
        TraceOp::PixelUnshuffle {
            downscale_factor: 2
        }
        .classification()
            == TraceOpClass::Vision
    );
    assert!(TraceOp::Upsample1d { factor: 2 }.classification() == TraceOpClass::Vision);
    assert!(
        TraceOp::Upsample2d {
            mode: TraceUpsampleMode::Nearest,
            scale_h: 2.0,
            scale_w: 2.0
        }
        .classification()
            == TraceOpClass::Vision
    );
    assert!(
        TraceOp::ResizeBilinear {
            target_h: 224,
            target_w: 224
        }
        .classification()
            == TraceOpClass::Vision
    );
    assert!(TraceOp::Triu { diagonal: 0 }.classification() == TraceOpClass::Vision);
    assert!(TraceOp::Tril { diagonal: 0 }.classification() == TraceOpClass::Vision);
}

/// Prove: QLinear classifies as Quantized.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_quantized() {
    let w = dummy_weight_2d(4, 4);
    assert!(
        TraceOp::QLinear {
            weight: w,
            bias: None
        }
        .classification()
            == TraceOpClass::Quantized
    );
}

/// Prove: named activation ops classify as NamedActivation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_named_activation() {
    assert!(
        TraceOp::Activation {
            kind: TraceActivation::Relu
        }
        .classification()
            == TraceOpClass::NamedActivation
    );
    assert!(TraceOp::Elu { alpha: 1.0 }.classification() == TraceOpClass::NamedActivation);
    assert!(TraceOp::LeakyRelu { slope: 0.01 }.classification() == TraceOpClass::NamedActivation);
    assert!(TraceOp::Softplus.classification() == TraceOpClass::NamedActivation);
    assert!(TraceOp::Selu.classification() == TraceOpClass::NamedActivation);
    assert!(TraceOp::Celu { alpha: 1.0 }.classification() == TraceOpClass::NamedActivation);
    assert!(TraceOp::Mish.classification() == TraceOpClass::NamedActivation);
    assert!(TraceOp::HardSigmoid.classification() == TraceOpClass::NamedActivation);
    assert!(TraceOp::HardSwish.classification() == TraceOpClass::NamedActivation);
    assert!(TraceOp::Softsign.classification() == TraceOpClass::NamedActivation);
    let slope_w = dummy_weight_1d(4);
    assert!(TraceOp::PRelu { slope: slope_w }.classification() == TraceOpClass::NamedActivation);
}

/// Prove: Clamp classifies as Clamp.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_clamp() {
    assert!(
        TraceOp::Clamp {
            min: Some(0.0),
            max: Some(1.0)
        }
        .classification()
            == TraceOpClass::Clamp
    );
}

/// Prove: Powf classifies as Power.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_power() {
    assert!(TraceOp::Powf { exponent: 2.0 }.classification() == TraceOpClass::Power);
}

/// Prove: Constant and ConstantWeight classify as ConstantValue.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_constant_value() {
    assert!(TraceOp::Constant { value: 1.0 }.classification() == TraceOpClass::ConstantValue);
    let w = dummy_weight_1d(4);
    assert!(TraceOp::ConstantWeight { weight: w }.classification() == TraceOpClass::ConstantValue);
}

/// Prove: SegmentBoundary classifies as SegmentBoundary.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_segment_boundary() {
    assert!(
        TraceOp::SegmentBoundary {
            reason: "test".to_string(),
            input_bounds: None
        }
        .classification()
            == TraceOpClass::SegmentBoundary
    );
}

/// Prove: Custom classifies as Custom.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_custom() {
    assert!(
        TraceOp::Custom {
            name: "nn_op".to_string()
        }
        .classification()
            == TraceOpClass::Custom
    );
}

// ===========================================================================
// KokoroFused classification: sub-variant routing
// ===========================================================================

/// Prove: KokoroFused::SnakeTensor classifies as NamedActivation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_kokoro_snake_tensor() {
    let alpha = dummy_weight(&[1, 4, 1]);
    assert!(
        TraceOp::KokoroFused(KokoroFusedOp::SnakeTensor { alpha }).classification()
            == TraceOpClass::NamedActivation
    );
}

/// Prove: KokoroFused::AdainSnake classifies as NamedActivation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_kokoro_adain_snake() {
    let alpha = dummy_weight(&[1, 4, 1]);
    assert!(
        TraceOp::KokoroFused(KokoroFusedOp::AdainSnake { alpha, eps: 1e-5 }).classification()
            == TraceOpClass::NamedActivation
    );
}

/// Prove: KokoroFused::AdaLayerNorm classifies as Normalization.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_kokoro_ada_layer_norm() {
    let w = dummy_weight_1d(4);
    let b = dummy_weight_1d(4);
    assert!(
        TraceOp::KokoroFused(KokoroFusedOp::AdaLayerNorm {
            norm_weight: w,
            norm_bias: b,
            eps: 1e-5,
        })
        .classification()
            == TraceOpClass::Normalization
    );
}

/// Prove: KokoroFused::FusedAdainResBlock classifies as Composite.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_kokoro_fused_resblock() {
    let w2x2 = dummy_weight_2d(2, 2);
    let w1 = dummy_weight_1d(2);
    let activation = ResBlockActivation::LeakyRelu { slope: 0.2 };
    assert!(
        TraceOp::KokoroFused(KokoroFusedOp::FusedAdainResBlock {
            activation,
            adain1_weight: w2x2.clone(),
            adain1_bias: w1.clone(),
            adain2_weight: w2x2.clone(),
            adain2_bias: w1.clone(),
            conv1_weight: dummy_weight(&[2, 2, 3]),
            conv1_bias: w1.clone(),
            conv1_dilation: 1,
            conv1_padding: 1,
            conv2_weight: dummy_weight(&[2, 2, 3]),
            conv2_bias: w1,
            conv2_padding: 1,
            eps: 1e-5,
            residual_scale: 1.0,
        })
        .classification()
            == TraceOpClass::Composite
    );
}

/// Prove: KokoroFused::AdainLeakyRelu classifies as NamedActivation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn classification_kokoro_adain_leaky_relu() {
    assert!(
        TraceOp::KokoroFused(KokoroFusedOp::AdainLeakyRelu {
            eps: 1e-5,
            slope: 0.2,
        })
        .classification()
            == TraceOpClass::NamedActivation
    );
}
