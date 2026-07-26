// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for TraceOp extended coverage and nn support types (#3799).
//!
//! Covers TraceOp classification and arity for op categories not yet verified:
//! NamedActivation, Vision, Composite, Clamp, Power, TypeConversion, Quantized,
//! ScanAccumulate, Indexing, and Recurrent.
//!
//! Also proves properties of nn support types: TraceActivation, TraceUpsampleMode,
//! WeightRef, and TraceOp canonical_name consistency.
//!
//! Properties proved:
//! - NamedActivation ops have arity 1 and correct classification
//! - Vision ops have arity 1 and correct classification
//! - Composite ops (SwiGlu, MoeGating) have arity 1 and correct classification
//! - Clamp has arity 1
//! - Powf has arity 1
//! - ToDtype has arity 1
//! - QLinear has arity 1
//! - Cumsum has arity 1
//! - Lstm has arity 3
//! - WhereCond has arity 3
//! - TraceActivation::as_str() returns non-empty strings
//! - TraceUpsampleMode::as_str() returns non-empty strings
//! - WeightRef::from_shape creates refs with empty data
//! - TraceOp canonical_name returns non-empty for all basic ops

#![cfg(kani)]

use crate::dyn_tensor::trace::{
    TraceActivation, TraceOp, TraceOpClass, TraceUpsampleMode, WeightRef,
};
use crate::DType;

// ---------------------------------------------------------------------------
// NamedActivation ops: all arity 1, classification == NamedActivation
// ---------------------------------------------------------------------------

/// Prove: Elu, LeakyRelu, Softplus, Selu, Celu, Mish, HardSigmoid,
/// HardSwish, Softsign all classify as NamedActivation with arity 1.
///
/// These activation functions are element-wise unary ops. The compile
/// and verify dispatchers both need arity == 1 to correctly build
/// single-input subgraphs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn named_activation_ops_arity_and_class() {
    // Elu
    let op = TraceOp::Elu { alpha: 1.0 };
    assert!(
        op.classification() == TraceOpClass::NamedActivation,
        "Elu must be NamedActivation"
    );
    assert!(op.expected_arity() == Some(1), "Elu must have arity 1");

    // LeakyRelu
    let op = TraceOp::LeakyRelu { slope: 0.01 };
    assert!(
        op.classification() == TraceOpClass::NamedActivation,
        "LeakyRelu must be NamedActivation"
    );
    assert!(
        op.expected_arity() == Some(1),
        "LeakyRelu must have arity 1"
    );

    // Softplus
    let op = TraceOp::Softplus;
    assert!(
        op.classification() == TraceOpClass::NamedActivation,
        "Softplus must be NamedActivation"
    );
    assert!(op.expected_arity() == Some(1), "Softplus must have arity 1");

    // Selu
    let op = TraceOp::Selu;
    assert!(
        op.classification() == TraceOpClass::NamedActivation,
        "Selu must be NamedActivation"
    );
    assert!(op.expected_arity() == Some(1), "Selu must have arity 1");
}

/// Prove: HardSigmoid, HardSwish, Softsign, Mish, Celu classification
/// and arity.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn named_activation_ops_hard_and_smooth() {
    let op = TraceOp::HardSigmoid;
    assert!(op.classification() == TraceOpClass::NamedActivation);
    assert!(op.expected_arity() == Some(1));

    let op = TraceOp::HardSwish;
    assert!(op.classification() == TraceOpClass::NamedActivation);
    assert!(op.expected_arity() == Some(1));

    let op = TraceOp::Softsign;
    assert!(op.classification() == TraceOpClass::NamedActivation);
    assert!(op.expected_arity() == Some(1));

    let op = TraceOp::Mish;
    assert!(op.classification() == TraceOpClass::NamedActivation);
    assert!(op.expected_arity() == Some(1));

    let op = TraceOp::Celu { alpha: 1.0 };
    assert!(op.classification() == TraceOpClass::NamedActivation);
    assert!(op.expected_arity() == Some(1));
}

// ---------------------------------------------------------------------------
// Vision ops: arity 1, classification == Vision
// ---------------------------------------------------------------------------

/// Prove: PixelShuffle, PixelUnshuffle, Upsample1d, Upsample2d,
/// ResizeBilinear, Triu, Tril all classify as Vision with arity 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn vision_ops_arity_and_class() {
    let op = TraceOp::PixelShuffle { upscale_factor: 2 };
    assert!(op.classification() == TraceOpClass::Vision);
    assert!(op.expected_arity() == Some(1));

    let op = TraceOp::PixelUnshuffle {
        downscale_factor: 2,
    };
    assert!(op.classification() == TraceOpClass::Vision);
    assert!(op.expected_arity() == Some(1));

    let op = TraceOp::Upsample1d { factor: 4 };
    assert!(op.classification() == TraceOpClass::Vision);
    assert!(op.expected_arity() == Some(1));

    let op = TraceOp::Upsample2d {
        mode: TraceUpsampleMode::Nearest,
        scale_h: 2.0,
        scale_w: 2.0,
    };
    assert!(op.classification() == TraceOpClass::Vision);
    assert!(op.expected_arity() == Some(1));

    let op = TraceOp::ResizeBilinear {
        target_h: 64,
        target_w: 64,
    };
    assert!(op.classification() == TraceOpClass::Vision);
    assert!(op.expected_arity() == Some(1));

    let op = TraceOp::Triu { diagonal: 0 };
    assert!(op.classification() == TraceOpClass::Vision);
    assert!(op.expected_arity() == Some(1));

    let op = TraceOp::Tril { diagonal: 0 };
    assert!(op.classification() == TraceOpClass::Vision);
    assert!(op.expected_arity() == Some(1));
}

// ---------------------------------------------------------------------------
// Composite, Clamp, Power, TypeConversion, Quantized
// ---------------------------------------------------------------------------

/// Prove: SwiGlu and MoeGating classify as Composite with arity 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn composite_ops_arity_and_class() {
    let op = TraceOp::SwiGlu;
    assert!(op.classification() == TraceOpClass::Composite);
    assert!(op.expected_arity() == Some(1));

    let op = TraceOp::MoeGating {
        num_experts: 8,
        top_k: 2,
    };
    assert!(op.classification() == TraceOpClass::Composite);
    assert!(op.expected_arity() == Some(1));
}

/// Prove: Clamp classifies as Clamp with arity 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn clamp_op_arity_and_class() {
    let op = TraceOp::Clamp {
        min: Some(-1.0),
        max: Some(1.0),
    };
    assert!(op.classification() == TraceOpClass::Clamp);
    assert!(op.expected_arity() == Some(1));

    // Clamp with only min
    let op = TraceOp::Clamp {
        min: Some(0.0),
        max: None,
    };
    assert!(op.classification() == TraceOpClass::Clamp);
    assert!(op.expected_arity() == Some(1));
}

/// Prove: Powf classifies as Power with arity 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn powf_op_arity_and_class() {
    let op = TraceOp::Powf { exponent: 2.0 };
    assert!(op.classification() == TraceOpClass::Power);
    assert!(op.expected_arity() == Some(1));
}

/// Prove: ToDtype classifies as TypeConversion with arity 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn todtype_op_arity_and_class() {
    let op = TraceOp::ToDtype {
        target_dtype: DType::F16,
    };
    assert!(op.classification() == TraceOpClass::TypeConversion);
    assert!(op.expected_arity() == Some(1));
}

/// Prove: QLinear classifies as Quantized with arity 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn qlinear_op_arity_and_class() {
    let w = WeightRef::from_shape(&[128, 64]);
    let op = TraceOp::QLinear {
        weight: w,
        bias: None,
    };
    assert!(op.classification() == TraceOpClass::Quantized);
    assert!(op.expected_arity() == Some(1));
}

// ---------------------------------------------------------------------------
// ScanAccumulate: Cumsum arity 1
// ---------------------------------------------------------------------------

/// Prove: Cumsum classifies as ScanAccumulate with arity 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn cumsum_op_arity_and_class() {
    let op = TraceOp::Cumsum { dim: 0 };
    assert!(op.classification() == TraceOpClass::ScanAccumulate);
    assert!(op.expected_arity() == Some(1));
}

// ---------------------------------------------------------------------------
// Recurrent: Lstm has arity 3 (input, h, c)
// ---------------------------------------------------------------------------

/// Prove: Lstm classifies as Recurrent with arity 3.
///
/// LSTM requires 3 tensor inputs: the data input, hidden state, and
/// cell state. Weight matrices are stored in the op's fields.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_op_arity_and_class() {
    let w_ih = WeightRef::from_shape(&[256, 64]);
    let w_hh = WeightRef::from_shape(&[256, 64]);
    let op = TraceOp::Lstm {
        weight_ih: w_ih,
        weight_hh: w_hh,
        bias_ih: None,
        bias_hh: None,
        hidden_size: 64,
        initial_hidden: None,
        initial_cell: None,
    };
    assert!(op.classification() == TraceOpClass::Recurrent);
    assert!(op.expected_arity() == Some(3));
}

// ---------------------------------------------------------------------------
// Ternary: WhereCond arity 3, ScatterAdd arity 3
// ---------------------------------------------------------------------------

/// Prove: WhereCond and ScatterAdd have arity 3.
///
/// WhereCond takes (condition, true_val, false_val).
/// ScatterAdd takes (self, index, src).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn ternary_ops_have_arity_three() {
    let op = TraceOp::WhereCond;
    assert!(op.classification() == TraceOpClass::Indexing);
    assert!(op.expected_arity() == Some(3));

    let op = TraceOp::ScatterAdd { dim: 0 };
    assert!(op.classification() == TraceOpClass::ScanAccumulate);
    assert!(op.expected_arity() == Some(3));

    let op = TraceOp::IndexAdd { dim: 0 };
    assert!(op.classification() == TraceOpClass::ScanAccumulate);
    assert!(op.expected_arity() == Some(3));
}

// ---------------------------------------------------------------------------
// TraceActivation::as_str() non-empty
// ---------------------------------------------------------------------------

/// Prove: every TraceActivation variant returns a non-empty string.
///
/// The compilation path uses as_str() for MSL function name generation.
/// An empty string would produce invalid MSL code.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn trace_activation_as_str_non_empty() {
    let activations = [
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
            !activations[i].as_str().is_empty(),
            "TraceActivation::as_str must return non-empty string"
        );
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// TraceUpsampleMode::as_str() non-empty
// ---------------------------------------------------------------------------

/// Prove: every TraceUpsampleMode variant returns a non-empty string.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn trace_upsample_mode_as_str_non_empty() {
    let nearest = TraceUpsampleMode::Nearest;
    let bilinear = TraceUpsampleMode::Bilinear;

    assert!(
        !nearest.as_str().is_empty(),
        "Nearest as_str must be non-empty"
    );
    assert!(
        !bilinear.as_str().is_empty(),
        "Bilinear as_str must be non-empty"
    );
}

// ---------------------------------------------------------------------------
// WeightRef::from_shape creates ref with empty data
// ---------------------------------------------------------------------------

/// Prove: WeightRef::from_shape produces a ref with empty data and
/// correct shape.
///
/// Shape-only weight refs are the fallback when data extraction fails.
/// The data must be empty (not zero-filled) so that consumers can
/// distinguish "no data available" from "data is all zeros".
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_ref_from_shape_empty_data() {
    let w = WeightRef::from_shape(&[8, 4, 3]);
    assert!(w.data().is_empty(), "from_shape must produce empty data");
    assert_eq!(w.shape().len(), 3, "shape must have 3 dims");
    assert_eq!(w.shape()[0], 8, "dim 0 must be 8");
    assert_eq!(w.shape()[1], 4, "dim 1 must be 4");
    assert_eq!(w.shape()[2], 3, "dim 2 must be 3");
}

// ---------------------------------------------------------------------------
// TraceOp canonical_name non-empty for basic ops
// ---------------------------------------------------------------------------

/// Prove: canonical_name() returns non-empty strings for all basic
/// unary and binary ops.
///
/// The trace recorder uses canonical_name() for node naming. An empty
/// name would produce undebuggable traces.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn canonical_name_non_empty_basic_ops() {
    let ops: [TraceOp; 10] = [
        TraceOp::Add,
        TraceOp::Sub,
        TraceOp::Mul,
        TraceOp::Div,
        TraceOp::Relu,
        TraceOp::Sigmoid,
        TraceOp::Tanh,
        TraceOp::Exp,
        TraceOp::Neg,
        TraceOp::Abs,
    ];

    let mut i = 0;
    while i < 10 {
        assert!(
            !ops[i].canonical_name().is_empty(),
            "canonical_name must be non-empty for basic ops"
        );
        i += 1;
    }
}

/// Prove: canonical_name() returns non-empty strings for shape and
/// normalization ops.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn canonical_name_non_empty_shape_and_norm_ops() {
    let w = WeightRef::from_shape(&[64]);
    let w2 = WeightRef::from_shape(&[64]);

    let ops: [TraceOp; 5] = [
        TraceOp::Reshape {
            target_shape: vec![4, 16],
        },
        TraceOp::Transpose { dim0: 0, dim1: 1 },
        TraceOp::Softmax { dim: 1 },
        TraceOp::LayerNorm {
            eps: 1e-5,
            weight: w,
            bias: w2,
        },
        TraceOp::ReduceSum {
            dim: 0,
            keepdim: false,
        },
    ];

    let mut i = 0;
    while i < 5 {
        assert!(
            !ops[i].canonical_name().is_empty(),
            "canonical_name must be non-empty for shape/norm ops"
        );
        i += 1;
    }
}
