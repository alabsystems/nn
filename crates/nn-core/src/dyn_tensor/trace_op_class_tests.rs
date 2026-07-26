// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `TraceOpClass` classification, `expected_arity()`, and
//! `canonical_name()` methods on `TraceOp`.

use super::{
    KokoroFusedOp, ResBlockActivation, TraceActivation, TraceOp, TraceOpClass, TraceUpsampleMode,
    WeightRef,
};
use crate::DType;

// ---------------------------------------------------------------------------
// classification() — verify every category routes correctly
// ---------------------------------------------------------------------------

#[test]
fn test_classification_input() {
    assert_eq!(TraceOp::Input.classification(), TraceOpClass::Input);
}

#[test]
fn test_classification_identity() {
    assert_eq!(TraceOp::Dropout.classification(), TraceOpClass::Identity);
}

#[test]
fn test_classification_binary_elementwise() {
    for op in [
        TraceOp::Add,
        TraceOp::Sub,
        TraceOp::Mul,
        TraceOp::Div,
        TraceOp::Maximum,
        TraceOp::Minimum,
    ] {
        assert_eq!(
            op.classification(),
            TraceOpClass::BinaryElementwise,
            "wrong class for {op:?}"
        );
    }
}

#[test]
fn test_classification_unary_elementwise() {
    let unary_ops = [
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
        TraceOp::Floor,
        TraceOp::Round,
        TraceOp::Fract,
    ];
    for op in &unary_ops {
        assert_eq!(
            op.classification(),
            TraceOpClass::UnaryElementwise,
            "wrong class for {op:?}"
        );
    }
}

#[test]
fn test_classification_matmul() {
    assert_eq!(TraceOp::MatMul.classification(), TraceOpClass::MatMul);
}

#[test]
fn test_classification_reduction() {
    let ops = [
        TraceOp::ReduceSum {
            dim: 0,
            keepdim: true,
        },
        TraceOp::ReduceMean {
            dim: 1,
            keepdim: false,
        },
        TraceOp::ReduceMax {
            dim: 0,
            keepdim: false,
        },
        TraceOp::ReduceMin {
            dim: 0,
            keepdim: true,
        },
    ];
    for op in &ops {
        assert_eq!(
            op.classification(),
            TraceOpClass::Reduction,
            "wrong class for {op:?}"
        );
    }
}

#[test]
fn test_classification_shape_only() {
    let ops = [
        TraceOp::Reshape {
            target_shape: vec![2, 3],
        },
        TraceOp::Unsqueeze { dim: 0 },
        TraceOp::Squeeze { dim: 1 },
    ];
    for op in &ops {
        assert_eq!(
            op.classification(),
            TraceOpClass::ShapeOnly,
            "wrong class for {op:?}"
        );
    }
}

#[test]
fn test_classification_shape_data_move() {
    let ops = [
        TraceOp::Transpose { dim0: 0, dim1: 1 },
        TraceOp::Permute {
            axes: vec![0, 2, 1],
        },
        TraceOp::Narrow {
            dim: 0,
            start: 0,
            length: 1,
        },
        TraceOp::Cat {
            dim: 0,
            num_inputs: 2,
        },
        TraceOp::Flip { dim: 0 },
        TraceOp::Expand {
            target_shape: vec![2, 3],
        },
    ];
    for op in &ops {
        assert_eq!(
            op.classification(),
            TraceOpClass::ShapeDataMove,
            "wrong class for {op:?}"
        );
    }
}

#[test]
fn test_classification_normalization() {
    let w = WeightRef::new(vec![1.0; 4], vec![4]).unwrap();
    let b = WeightRef::new(vec![0.0; 4], vec![4]).unwrap();
    let ops = [
        TraceOp::LayerNorm {
            eps: 1e-5,
            weight: w.clone(),
            bias: b,
        },
        TraceOp::RmsNorm {
            eps: 1e-5,
            weight: w,
        },
        TraceOp::InstanceNorm { eps: 1e-5 },
    ];
    for op in &ops {
        assert_eq!(
            op.classification(),
            TraceOpClass::Normalization,
            "wrong class for {op:?}"
        );
    }
}

#[test]
fn test_classification_weighted_linear() {
    let w = WeightRef::new(vec![1.0; 4], vec![2, 2]).unwrap();
    assert_eq!(
        TraceOp::Linear {
            weight: w,
            bias: None
        }
        .classification(),
        TraceOpClass::WeightedLinear,
    );
}

#[test]
fn test_classification_attention() {
    assert_eq!(
        TraceOp::Softmax { dim: 1 }.classification(),
        TraceOpClass::Attention,
    );
    assert_eq!(
        TraceOp::LogSoftmax { dim: 1 }.classification(),
        TraceOpClass::Attention,
    );
}

#[test]
fn test_classification_indexing() {
    assert_eq!(
        TraceOp::IndexSelect { dim: 0 }.classification(),
        TraceOpClass::Indexing,
    );
    assert_eq!(
        TraceOp::Gather { dim: 0 }.classification(),
        TraceOpClass::Indexing,
    );
    assert_eq!(TraceOp::WhereCond.classification(), TraceOpClass::Indexing,);
}

#[test]
fn test_classification_composite() {
    assert_eq!(TraceOp::SwiGlu.classification(), TraceOpClass::Composite);
    let dummy = WeightRef::from_shape(&[1]);
    assert_eq!(
        TraceOp::KokoroFused(KokoroFusedOp::FusedAdainResBlock {
            activation: ResBlockActivation::Snake {
                alpha1: dummy.clone(),
                alpha2: dummy.clone(),
            },
            adain1_weight: dummy.clone(),
            adain1_bias: dummy.clone(),
            adain2_weight: dummy.clone(),
            adain2_bias: dummy.clone(),
            conv1_weight: dummy.clone(),
            conv1_bias: dummy.clone(),
            conv1_dilation: 1,
            conv1_padding: 0,
            conv2_weight: dummy.clone(),
            conv2_bias: dummy,
            conv2_padding: 0,
            eps: 1e-5,
            residual_scale: 1.0,
        })
        .classification(),
        TraceOpClass::Composite,
    );
}

#[test]
fn test_classification_type_conversion() {
    assert_eq!(
        TraceOp::ToDtype {
            target_dtype: DType::F16,
        }
        .classification(),
        TraceOpClass::TypeConversion,
    );
}

#[test]
fn test_classification_embedding() {
    let w = WeightRef::new(vec![1.0; 8], vec![4, 2]).unwrap();
    assert_eq!(
        TraceOp::Embedding { weight: w }.classification(),
        TraceOpClass::Embedding,
    );
}

#[test]
fn test_classification_recurrent() {
    let w = WeightRef::new(vec![1.0; 4], vec![4]).unwrap();
    let w2 = w.clone();
    assert_eq!(
        TraceOp::Lstm {
            weight_ih: w,
            weight_hh: w2,
            bias_ih: None,
            bias_hh: None,
            hidden_size: 2,
            initial_hidden: None,
            initial_cell: None,
        }
        .classification(),
        TraceOpClass::Recurrent,
    );
}

#[test]
fn test_classification_pooling() {
    assert_eq!(
        TraceOp::AvgPool2d {
            kernel_size: [2, 2],
            stride: [2, 2],
            padding: [0, 0],
        }
        .classification(),
        TraceOpClass::Pooling,
    );
    assert_eq!(
        TraceOp::AdaptiveAvgPool2d {
            output_size: [1, 1],
        }
        .classification(),
        TraceOpClass::Pooling,
    );
}

#[test]
fn test_classification_scan_accumulate() {
    assert_eq!(
        TraceOp::Cumsum { dim: 0 }.classification(),
        TraceOpClass::ScanAccumulate,
    );
    assert_eq!(
        TraceOp::RepeatInterleave { dim: 0 }.classification(),
        TraceOpClass::ScanAccumulate,
    );
}

#[test]
fn test_classification_vision() {
    assert_eq!(
        TraceOp::PixelShuffle { upscale_factor: 2 }.classification(),
        TraceOpClass::Vision,
    );
    assert_eq!(
        TraceOp::PixelUnshuffle {
            downscale_factor: 2
        }
        .classification(),
        TraceOpClass::Vision,
    );
    assert_eq!(
        TraceOp::Upsample2d {
            mode: TraceUpsampleMode::Nearest,
            scale_h: 2.0,
            scale_w: 2.0,
        }
        .classification(),
        TraceOpClass::Vision,
    );
}

#[test]
fn test_classification_quantized() {
    let w = WeightRef::new(vec![1.0; 4], vec![2, 2]).unwrap();
    assert_eq!(
        TraceOp::QLinear {
            weight: w,
            bias: None,
        }
        .classification(),
        TraceOpClass::Quantized,
    );
}

#[test]
fn test_classification_named_activation() {
    assert_eq!(
        TraceOp::Activation {
            kind: TraceActivation::Mish,
        }
        .classification(),
        TraceOpClass::NamedActivation,
    );
}

#[test]
fn test_classification_clamp() {
    assert_eq!(
        TraceOp::Clamp {
            min: Some(0.0),
            max: Some(1.0),
        }
        .classification(),
        TraceOpClass::Clamp,
    );
}

#[test]
fn test_classification_power() {
    assert_eq!(
        TraceOp::Powf { exponent: 2.0 }.classification(),
        TraceOpClass::Power,
    );
}

#[test]
fn test_classification_custom() {
    assert_eq!(
        TraceOp::Custom {
            name: "nn_op".into()
        }
        .classification(),
        TraceOpClass::Custom,
    );
}

// ---------------------------------------------------------------------------
// expected_arity() — verify input count for key ops
// ---------------------------------------------------------------------------

#[test]
fn test_arity_input_is_zero() {
    assert_eq!(TraceOp::Input.expected_arity(), Some(0));
}

#[test]
fn test_arity_unary_ops_are_one() {
    assert_eq!(TraceOp::Relu.expected_arity(), Some(1));
    assert_eq!(TraceOp::Softmax { dim: 1 }.expected_arity(), Some(1));
    assert_eq!(TraceOp::Dropout.expected_arity(), Some(1));
    assert_eq!(
        TraceOp::Reshape {
            target_shape: vec![2]
        }
        .expected_arity(),
        Some(1)
    );
}

#[test]
fn test_arity_binary_ops_are_two() {
    assert_eq!(TraceOp::Add.expected_arity(), Some(2));
    assert_eq!(TraceOp::MatMul.expected_arity(), Some(2));
    assert_eq!(TraceOp::IndexSelect { dim: 0 }.expected_arity(), Some(2));
}

#[test]
fn test_arity_ternary_ops_are_three() {
    assert_eq!(TraceOp::WhereCond.expected_arity(), Some(3));
    assert_eq!(TraceOp::ScatterAdd { dim: 0 }.expected_arity(), Some(3));
}

#[test]
fn test_arity_lstm_is_three() {
    let w = WeightRef::new(vec![1.0; 4], vec![4]).expect("valid weight");
    let w2 = w.clone();
    assert_eq!(
        TraceOp::Lstm {
            weight_ih: w,
            weight_hh: w2,
            bias_ih: None,
            bias_hh: None,
            hidden_size: 2,
            initial_hidden: None,
            initial_cell: None,
        }
        .expected_arity(),
        Some(3),
    );
}

#[test]
fn test_arity_cat_uses_num_inputs() {
    assert_eq!(
        TraceOp::Cat {
            dim: 0,
            num_inputs: 5
        }
        .expected_arity(),
        Some(5)
    );
    assert_eq!(
        TraceOp::Cat {
            dim: 1,
            num_inputs: 1
        }
        .expected_arity(),
        Some(1)
    );
}

// ---------------------------------------------------------------------------
// canonical_name() — verify names match op_prefix() for shared variants
// ---------------------------------------------------------------------------

#[test]
fn test_canonical_name_matches_op_prefix_for_core_ops() {
    assert_eq!(TraceOp::Input.canonical_name(), "input");
    assert_eq!(TraceOp::Add.canonical_name(), "add");
    assert_eq!(TraceOp::Relu.canonical_name(), "relu");
    assert_eq!(TraceOp::MatMul.canonical_name(), "matmul");
    assert_eq!(TraceOp::Softmax { dim: 1 }.canonical_name(), "softmax");
    assert_eq!(TraceOp::Dropout.canonical_name(), "dropout");
}

#[test]
fn test_canonical_name_gelu_variants() {
    assert_eq!(TraceOp::Gelu.canonical_name(), "gelu");
    assert_eq!(TraceOp::GeluErf.canonical_name(), "gelu");
}

#[test]
fn test_canonical_name_activation_returns_inner_name() {
    assert_eq!(
        TraceOp::Activation {
            kind: TraceActivation::Mish,
        }
        .canonical_name(),
        "mish",
    );
}

#[test]
fn test_canonical_name_custom_returns_inner_name() {
    assert_eq!(
        TraceOp::Custom {
            name: "nn_layer".into()
        }
        .canonical_name(),
        "nn_layer",
    );
}

#[test]
fn test_canonical_name_covers_all_explicit_variants() {
    // Verify that no explicitly-handled variant returns the catch-all name.
    // Only the #[non_exhaustive] catch-all returns "<unknown_trace_op>".
    let ops: Vec<TraceOp> = vec![
        TraceOp::Input,
        TraceOp::Add,
        TraceOp::Sub,
        TraceOp::Mul,
        TraceOp::Div,
        TraceOp::Maximum,
        TraceOp::Minimum,
        TraceOp::MatMul,
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
        TraceOp::Floor,
        TraceOp::Round,
        TraceOp::Fract,
        TraceOp::Dropout,
        TraceOp::SwiGlu,
        TraceOp::WhereCond,
        TraceOp::ToDtype {
            target_dtype: DType::F32,
        },
        TraceOp::Flip { dim: 0 },
        TraceOp::Powf { exponent: 2.0 },
        TraceOp::Clamp {
            min: None,
            max: None,
        },
        TraceOp::Expand {
            target_shape: vec![1],
        },
        TraceOp::IndexSelect { dim: 0 },
        TraceOp::Gather { dim: 0 },
        TraceOp::Cumsum { dim: 0 },
        TraceOp::RepeatInterleave { dim: 0 },
        TraceOp::ScatterAdd { dim: 0 },
        TraceOp::IndexAdd { dim: 0 },
        TraceOp::KokoroFused(KokoroFusedOp::FusedAdainResBlock {
            activation: ResBlockActivation::LeakyRelu { slope: 0.2 },
            adain1_weight: WeightRef::from_shape(&[1]),
            adain1_bias: WeightRef::from_shape(&[1]),
            adain2_weight: WeightRef::from_shape(&[1]),
            adain2_bias: WeightRef::from_shape(&[1]),
            conv1_weight: WeightRef::from_shape(&[1]),
            conv1_bias: WeightRef::from_shape(&[1]),
            conv1_dilation: 1,
            conv1_padding: 0,
            conv2_weight: WeightRef::from_shape(&[1]),
            conv2_bias: WeightRef::from_shape(&[1]),
            conv2_padding: 0,
            eps: 1e-5,
            residual_scale: 1.0,
        }),
    ];
    for op in &ops {
        assert_ne!(
            op.canonical_name(),
            "<unknown_trace_op>",
            "{op:?} should have a real canonical name"
        );
    }
}
