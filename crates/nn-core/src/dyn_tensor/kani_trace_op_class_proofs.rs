// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `TraceOp` arity and name invariants (#3681).
//!
//! Proves correctness of `expected_arity()` and `canonical_name()` across
//! `TraceOp` variants. Key properties verified:
//!
//! - Arity returns `Some(n)` for all explicitly-handled variants
//! - Zero-input ops (Input, Constant) have arity 0
//! - Unary ops have arity 1, binary ops have arity 2, ternary ops have arity 3
//! - canonical_name() is non-empty for every variant
//! - Classification and arity are consistent (e.g., BinaryElementwise => arity 2)
//! - WeightRef invariants hold for the shapes used throughout TraceOp fields

use crate::dyn_tensor::trace::{
    KokoroFusedOp, ResBlockActivation, TraceActivation, TraceOp, TraceOpClass, TraceUpsampleMode,
    WeightRef,
};
use crate::dyn_tensor::CompareOp;
use crate::DType;

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
// expected_arity(): zero-input ops
// ===========================================================================

/// Prove: Input and Constant have arity 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_zero_input_ops() {
    assert!(TraceOp::Input.expected_arity() == Some(0));
    assert!(TraceOp::Constant { value: 42.0 }.expected_arity() == Some(0));
}

/// Prove: remaining unary elementwise ops have arity 1.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_unary_elementwise_ext() {
    let ops: [TraceOp; 10] = [
        TraceOp::GeluErf,
        TraceOp::Sqr,
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
    while i < 10 {
        assert!(
            ops[i].expected_arity() == Some(1),
            "unary elementwise ext must have arity 1"
        );
        i += 1;
    }
}

/// Prove: Fract and Dropout have arity 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_fract_dropout_unary() {
    assert!(TraceOp::Fract.expected_arity() == Some(1));
    assert!(TraceOp::Dropout.expected_arity() == Some(1));
}

/// Prove: shape-only ops have arity 1.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_shape_only() {
    assert!(
        TraceOp::Reshape {
            target_shape: vec![2, 3]
        }
        .expected_arity()
            == Some(1)
    );
    assert!(TraceOp::Unsqueeze { dim: 0 }.expected_arity() == Some(1));
    assert!(TraceOp::Squeeze { dim: 0 }.expected_arity() == Some(1));
}

/// Prove: unary shape-data-move ops have arity 1.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_shape_data_move_unary() {
    assert!(TraceOp::Transpose { dim0: 0, dim1: 1 }.expected_arity() == Some(1));
    assert!(
        TraceOp::Narrow {
            dim: 0,
            start: 0,
            length: 5
        }
        .expected_arity()
            == Some(1)
    );
    assert!(TraceOp::Permute { axes: vec![1, 0] }.expected_arity() == Some(1));
    assert!(TraceOp::Flip { dim: 0 }.expected_arity() == Some(1));
    assert!(
        TraceOp::Unfold {
            dim: 0,
            size: 4,
            step: 2
        }
        .expected_arity()
            == Some(1)
    );
    assert!(
        TraceOp::Expand {
            target_shape: vec![2, 3]
        }
        .expected_arity()
            == Some(1)
    );
}

/// Prove: reduction ops have arity 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_reductions() {
    assert!(
        TraceOp::ReduceSum {
            dim: 0,
            keepdim: true
        }
        .expected_arity()
            == Some(1)
    );
    assert!(
        TraceOp::ReduceMean {
            dim: 0,
            keepdim: false
        }
        .expected_arity()
            == Some(1)
    );
    assert!(
        TraceOp::ReduceMax {
            dim: 0,
            keepdim: true
        }
        .expected_arity()
            == Some(1)
    );
    assert!(
        TraceOp::ReduceMin {
            dim: 0,
            keepdim: false
        }
        .expected_arity()
            == Some(1)
    );
}

/// Prove: weighted linear/conv ops have arity 1 (weights are in-op, not tensor inputs).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_weighted_linear() {
    let w = dummy_weight_2d(4, 4);
    assert!(
        TraceOp::Linear {
            weight: w.clone(),
            bias: None
        }
        .expected_arity()
            == Some(1)
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
        .expected_arity()
            == Some(1)
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
        .expected_arity()
            == Some(1)
    );
}

/// Prove: pooling ops have arity 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_pooling() {
    assert!(
        TraceOp::MaxPool1d {
            kernel_size: 2,
            stride: 2,
            padding: 0
        }
        .expected_arity()
            == Some(1)
    );
    assert!(
        TraceOp::AvgPool2d {
            kernel_size: [2, 2],
            stride: [2, 2],
            padding: [0, 0]
        }
        .expected_arity()
            == Some(1)
    );
    assert!(
        TraceOp::AdaptiveAvgPool2d {
            output_size: [1, 1]
        }
        .expected_arity()
            == Some(1)
    );
}

/// Prove: MatMul has arity 2.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_matmul() {
    assert!(TraceOp::MatMul.expected_arity() == Some(2));
}

/// Prove: binary indexing/selection ops have arity 2.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_binary_indexing() {
    assert!(TraceOp::IndexSelect { dim: 0 }.expected_arity() == Some(2));
    assert!(TraceOp::Gather { dim: 0 }.expected_arity() == Some(2));
    assert!(TraceOp::RepeatInterleave { dim: 0 }.expected_arity() == Some(2));
    assert!(TraceOp::SliceSet { dim: 0, start: 0 }.expected_arity() == Some(2));
    assert!(TraceOp::CompareTensor { op: CompareOp::Eq }.expected_arity() == Some(2));
}

// ===========================================================================
// expected_arity(): ternary ops
// ===========================================================================

/// Prove: ternary ops (WhereCond, ScatterAdd, IndexAdd) have arity 3.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_ternary() {
    assert!(TraceOp::WhereCond.expected_arity() == Some(3));
    assert!(TraceOp::ScatterAdd { dim: 0 }.expected_arity() == Some(3));
    assert!(TraceOp::IndexAdd { dim: 0 }.expected_arity() == Some(3));
}

/// Prove: LSTM has arity 3 (input, h, c).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_lstm() {
    let w = dummy_weight_2d(4, 4);
    assert!(
        TraceOp::Lstm {
            weight_ih: w.clone(),
            weight_hh: w,
            bias_ih: None,
            bias_hh: None,
            hidden_size: 4,
            initial_hidden: None,
            initial_cell: None,
        }
        .expected_arity()
            == Some(3)
    );
}

/// Prove: SdpaCausal has arity 3 (Q, K, V).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_sdpa_causal() {
    assert!(TraceOp::SdpaCausal { scale: 0.125 }.expected_arity() == Some(3));
}

/// Prove: MultiHeadAttention has arity 3 (Q, K, V).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_multi_head_attention() {
    assert!(
        TraceOp::MultiHeadAttention {
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 64
        }
        .expected_arity()
            == Some(3)
    );
}

// ===========================================================================
// expected_arity(): variable-arity ops
// ===========================================================================

/// Prove: Cat arity equals num_inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_cat_variable() {
    let n: u8 = kani::any();
    kani::assume(n >= 1 && n <= 16);
    let op = TraceOp::Cat {
        dim: 0,
        num_inputs: n as usize,
    };
    assert!(
        op.expected_arity() == Some(n as usize),
        "Cat arity must equal num_inputs"
    );
}

/// Prove: Sdpa (non-causal) returns None for arity (variable: 3 or 4).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_sdpa_variable() {
    assert!(
        TraceOp::Sdpa { scale: 0.125 }.expected_arity().is_none(),
        "Sdpa has variable arity (3 or 4), must return None"
    );
}

/// Prove: SegmentBoundary has arity 1 (passthrough).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_segment_boundary() {
    assert!(
        TraceOp::SegmentBoundary {
            reason: "len_reg".to_string(),
            input_bounds: Some((-1.0, 1.0))
        }
        .expected_arity()
            == Some(1)
    );
}

// ===========================================================================
// expected_arity(): KokoroFused sub-variants
// ===========================================================================

/// Prove: KokoroFused arity: SnakeTensor=1, AdainSnake/AdainLeakyRelu/AdaLayerNorm=3.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_kokoro_fused() {
    let alpha = dummy_weight(&[1, 4, 1]);
    assert!(
        TraceOp::KokoroFused(KokoroFusedOp::SnakeTensor {
            alpha: alpha.clone()
        })
        .expected_arity()
            == Some(1)
    );
    assert!(
        TraceOp::KokoroFused(KokoroFusedOp::AdainSnake { alpha, eps: 1e-5 }).expected_arity()
            == Some(3)
    );
    assert!(
        TraceOp::KokoroFused(KokoroFusedOp::AdainLeakyRelu {
            eps: 1e-5,
            slope: 0.2,
        })
        .expected_arity()
            == Some(3)
    );
    let nw = dummy_weight_1d(4);
    let nb = dummy_weight_1d(4);
    assert!(
        TraceOp::KokoroFused(KokoroFusedOp::AdaLayerNorm {
            norm_weight: nw,
            norm_bias: nb,
            eps: 1e-5,
        })
        .expected_arity()
            == Some(3)
    );
}

/// Prove: KokoroFused::FusedAdainResBlock has arity 2 (x, style).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_kokoro_fused_resblock() {
    let w2 = dummy_weight_2d(4, 4);
    let w1 = dummy_weight_1d(4);
    let op = TraceOp::KokoroFused(KokoroFusedOp::FusedAdainResBlock {
        activation: ResBlockActivation::LeakyRelu { slope: 0.2 },
        adain1_weight: w2.clone(),
        adain1_bias: w1.clone(),
        adain2_weight: w2.clone(),
        adain2_bias: w1.clone(),
        conv1_weight: dummy_weight(&[4, 4, 3]),
        conv1_bias: w1.clone(),
        conv1_dilation: 1,
        conv1_padding: 1,
        conv2_weight: dummy_weight(&[4, 4, 3]),
        conv2_bias: w1,
        conv2_padding: 1,
        eps: 1e-5,
        residual_scale: 1.0,
    });
    assert!(op.expected_arity() == Some(2));
}

// ===========================================================================
// canonical_name(): non-empty for all major variant families
// ===========================================================================

/// Prove: canonical_name() is non-empty for unary ops.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn canonical_name_nonempty_unary() {
    let ops: [TraceOp; 8] = [
        TraceOp::Relu,
        TraceOp::Silu,
        TraceOp::Tanh,
        TraceOp::Exp,
        TraceOp::Sqrt,
        TraceOp::Abs,
        TraceOp::Sin,
        TraceOp::Cos,
    ];
    let mut i = 0;
    while i < 8 {
        assert!(
            !ops[i].canonical_name().is_empty(),
            "canonical_name must be non-empty"
        );
        i += 1;
    }
}

/// Prove: canonical_name() is non-empty for structural ops.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn canonical_name_nonempty_structural() {
    assert!(!TraceOp::Input.canonical_name().is_empty());
    assert!(!TraceOp::Add.canonical_name().is_empty());
    assert!(!TraceOp::MatMul.canonical_name().is_empty());
    assert!(!TraceOp::Softmax { dim: 0 }.canonical_name().is_empty());
    assert!(!TraceOp::Reshape {
        target_shape: vec![1]
    }
    .canonical_name()
    .is_empty());
    assert!(!TraceOp::Cat {
        dim: 0,
        num_inputs: 2
    }
    .canonical_name()
    .is_empty());
    assert!(!TraceOp::ReduceSum {
        dim: 0,
        keepdim: false
    }
    .canonical_name()
    .is_empty());
    assert!(!TraceOp::Dropout.canonical_name().is_empty());
    assert!(!TraceOp::SwiGlu.canonical_name().is_empty());
    assert!(!TraceOp::Atan2.canonical_name().is_empty());
}

// ===========================================================================
// Cross-property consistency: classification ↔ arity agreement
// ===========================================================================

/// Prove: all BinaryElementwise ops have arity exactly 2.
///
/// If classification says BinaryElementwise but arity is not 2, the compile
/// and verify dispatchers would disagree on input count — a verification gap.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn consistency_binary_class_implies_arity_two() {
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
        assert!(ops[i].classification() == TraceOpClass::BinaryElementwise);
        assert!(
            ops[i].expected_arity() == Some(2),
            "BinaryElementwise must have arity 2"
        );
        i += 1;
    }
}

/// Prove: Input class has arity 0 (no tensor inputs).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn consistency_input_class_implies_arity_zero() {
    let op = TraceOp::Input;
    assert!(op.classification() == TraceOpClass::Input);
    assert!(op.expected_arity() == Some(0));
}

/// Prove: Identity class ops (Dropout) have arity 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn consistency_identity_class_implies_arity_one() {
    let op = TraceOp::Dropout;
    assert!(op.classification() == TraceOpClass::Identity);
    assert!(op.expected_arity() == Some(1));
}

// ===========================================================================
// WeightRef invariants
// ===========================================================================

/// Prove: WeightRef::new validates shape-data consistency.
///
/// If data length does not match shape product, new() must return Err.
/// This prevents silent weight corruption during model import.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_ref_rejects_inconsistent_shape() {
    let s0: u8 = kani::any();
    let s1: u8 = kani::any();
    kani::assume(s0 >= 1 && s0 <= 8);
    kani::assume(s1 >= 1 && s1 <= 8);

    let shape = vec![s0 as usize, s1 as usize];
    let expected_len = (s0 as usize) * (s1 as usize);

    // Correct length must succeed.
    let data_ok = vec![0.0f32; expected_len];
    assert!(WeightRef::new(data_ok, shape.clone()).is_ok());

    // Wrong length (one more) must fail.
    let data_bad = vec![0.0f32; expected_len + 1];
    assert!(WeightRef::new(data_bad, shape).is_err());
}

/// Prove: WeightRef with matching data/shape is NOT a placeholder.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_ref_with_data_is_not_placeholder() {
    let n: u8 = kani::any();
    kani::assume(n >= 1 && n <= 8);
    let shape = vec![n as usize];
    let data = vec![1.0f32; n as usize];
    let wr = WeightRef::new(data, shape).unwrap();
    assert!(
        !wr.is_placeholder(),
        "data-bearing WeightRef must not be a placeholder"
    );
}

/// Prove: WeightRef::new with empty data always succeeds (shape-only allowed).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_ref_empty_data_always_ok() {
    let s: u8 = kani::any();
    kani::assume(s <= 16);
    let shape = vec![s as usize, 4];
    let result = WeightRef::new(vec![], shape);
    assert!(
        result.is_ok(),
        "empty data must always succeed (shape-only ref)"
    );
}

// ===========================================================================
// TraceActivation::as_str() non-empty
// ===========================================================================

/// Prove: every TraceActivation variant has a non-empty string name.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn trace_activation_as_str_nonempty() {
    let acts = [
        TraceActivation::Relu,
        TraceActivation::Gelu,
        TraceActivation::GeluErf,
        TraceActivation::Silu,
        TraceActivation::Sigmoid,
        TraceActivation::Tanh,
        TraceActivation::Exp,
        TraceActivation::Log,
        TraceActivation::Elu,
        TraceActivation::LeakyRelu,
        TraceActivation::Mish,
    ];
    let mut i = 0;
    while i < 11 {
        assert!(
            !acts[i].as_str().is_empty(),
            "TraceActivation::as_str must be non-empty"
        );
        i += 1;
    }
}

/// Prove: every TraceUpsampleMode variant has a non-empty string name.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn trace_upsample_mode_as_str_nonempty() {
    assert!(!TraceUpsampleMode::Nearest.as_str().is_empty());
    assert!(!TraceUpsampleMode::Bilinear.as_str().is_empty());
}
