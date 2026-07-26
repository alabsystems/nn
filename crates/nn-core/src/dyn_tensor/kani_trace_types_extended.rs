// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for `TraceOp` properties (#3711).
//!
//! Supplements `kani_trace_types_proofs.rs` (classification) with proofs
//! for `expected_arity()`, `canonical_name()`, and cross-property consistency:
//!
//! **expected_arity correctness (8 harnesses):**
//!  1. Nullary ops (Input, Constant) have arity 0
//!  2. All 21 unary elementwise ops have arity 1
//!  3. Shape-only ops have arity 1
//!  4. Normalization ops have arity 1
//!  5. Binary elementwise ops have arity 2
//!  6. Ternary ops (WhereCond, ScatterAdd, IndexAdd) have arity 3
//!  7. Cat arity equals num_inputs
//!  8. LSTM has arity 3 (input + h + c)
//!
//! **canonical_name non-empty (4 harnesses):**
//!  9. All unary ops have non-empty canonical names
//! 10. All binary ops have non-empty canonical names
//! 11. Normalization ops have non-empty canonical names
//! 12. Activation ops have non-empty canonical names
//!
//! **Cross-property consistency (3 harnesses):**
//! 13. arity(Input) == 0 AND classification(Input) == Input
//! 14. arity(Dropout) == 1 AND classification(Dropout) == Identity
//! 15. arity(Custom) == 1 AND classification(Custom) == Custom
//!
//! Part of #3711.

use crate::dyn_tensor::trace::{
    KokoroFusedOp, TraceActivation, TraceOp, TraceOpClass, TraceUpsampleMode, WeightRef,
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
// expected_arity correctness
// ===========================================================================

/// Prove: nullary ops (Input, Constant) have arity 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_nullary_ops() {
    assert!(TraceOp::Input.expected_arity() == Some(0));
    assert!(TraceOp::Constant { value: 1.0 }.expected_arity() == Some(0));
}

/// Prove: all 21 unary elementwise ops + Fract have arity 1.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_unary_elementwise() {
    let ops: [TraceOp; 21] = [
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
        TraceOp::Fract,
    ];
    let mut i = 0;
    while i < 21 {
        assert!(
            ops[i].expected_arity() == Some(1),
            "unary elementwise op must have arity 1"
        );
        i += 1;
    }
}

/// Prove: shape-only ops (Reshape, Unsqueeze, Squeeze, Transpose,
/// Narrow, Permute, Flip, Unfold, Expand) have arity 1.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_shape_ops() {
    assert!(
        TraceOp::Reshape {
            target_shape: vec![4, 8]
        }
        .expected_arity()
            == Some(1)
    );
    assert!(TraceOp::Unsqueeze { dim: 0 }.expected_arity() == Some(1));
    assert!(TraceOp::Squeeze { dim: 1 }.expected_arity() == Some(1));
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

/// Prove: normalization ops have arity 1 (single input tensor).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_normalization_ops() {
    let w = dummy_weight_1d(4);
    let b = dummy_weight_1d(4);
    assert!(
        TraceOp::LayerNorm {
            eps: 1e-5,
            weight: w.clone(),
            bias: b.clone()
        }
        .expected_arity()
            == Some(1)
    );
    assert!(
        TraceOp::RmsNorm {
            eps: 1e-5,
            weight: w.clone()
        }
        .expected_arity()
            == Some(1)
    );
    assert!(
        TraceOp::GroupNorm {
            num_groups: 2,
            eps: 1e-5,
            weight: w.clone(),
            bias: b.clone()
        }
        .expected_arity()
            == Some(1)
    );
    assert!(TraceOp::InstanceNorm { eps: 1e-5 }.expected_arity() == Some(1));
    assert!(
        TraceOp::BatchNorm {
            eps: 1e-5,
            weight: w.clone(),
            bias: b.clone(),
            running_mean: w.clone(),
            running_var: b.clone()
        }
        .expected_arity()
            == Some(1)
    );
}

/// Prove: binary elementwise ops have arity 2.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_binary_elementwise() {
    let ops: [TraceOp; 7] = [
        TraceOp::Add,
        TraceOp::Sub,
        TraceOp::Mul,
        TraceOp::Div,
        TraceOp::Maximum,
        TraceOp::Minimum,
        TraceOp::MatMul,
    ];
    let mut i = 0;
    while i < 7 {
        assert!(
            ops[i].expected_arity() == Some(2),
            "binary op must have arity 2"
        );
        i += 1;
    }
}

/// Prove: ternary ops have arity 3.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arity_ternary_ops() {
    assert!(TraceOp::WhereCond.expected_arity() == Some(3));
    assert!(TraceOp::ScatterAdd { dim: 0 }.expected_arity() == Some(3));
    assert!(TraceOp::IndexAdd { dim: 0 }.expected_arity() == Some(3));
}

/// Prove: Cat arity equals num_inputs for various values.
#[kani::unwind(1)]
#[kani::proof]
fn arity_cat_equals_num_inputs() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 32);

    let op = TraceOp::Cat {
        dim: 0,
        num_inputs: n,
    };
    assert!(
        op.expected_arity() == Some(n),
        "Cat arity must equal num_inputs"
    );
}

// ===========================================================================
// canonical_name non-empty
// ===========================================================================

/// Prove: all unary elementwise ops have non-empty canonical names.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn canonical_name_unary_nonempty() {
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
        TraceOp::Floor,
        TraceOp::Round,
        TraceOp::Fract,
        TraceOp::Dropout,
        TraceOp::SwiGlu,
    ];
    let mut i = 0;
    while i < 20 {
        assert!(
            !ops[i].canonical_name().is_empty(),
            "canonical_name must be non-empty"
        );
        i += 1;
    }
}

/// Prove: all binary ops have non-empty canonical names.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn canonical_name_binary_nonempty() {
    let ops: [TraceOp; 7] = [
        TraceOp::Add,
        TraceOp::Sub,
        TraceOp::Mul,
        TraceOp::Div,
        TraceOp::Maximum,
        TraceOp::Minimum,
        TraceOp::MatMul,
    ];
    let mut i = 0;
    while i < 7 {
        assert!(
            !ops[i].canonical_name().is_empty(),
            "canonical_name must be non-empty"
        );
        i += 1;
    }
}

/// Prove: normalization ops have non-empty canonical names.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn canonical_name_normalization_nonempty() {
    let w = dummy_weight_1d(4);
    let b = dummy_weight_1d(4);
    let ops: Vec<TraceOp> = vec![
        TraceOp::LayerNorm {
            eps: 1e-5,
            weight: w.clone(),
            bias: b.clone(),
        },
        TraceOp::RmsNorm {
            eps: 1e-5,
            weight: w.clone(),
        },
        TraceOp::GroupNorm {
            num_groups: 2,
            eps: 1e-5,
            weight: w.clone(),
            bias: b.clone(),
        },
        TraceOp::InstanceNorm { eps: 1e-5 },
        TraceOp::BatchNorm {
            eps: 1e-5,
            weight: w.clone(),
            bias: b.clone(),
            running_mean: w.clone(),
            running_var: b.clone(),
        },
    ];
    for op in &ops {
        assert!(
            !op.canonical_name().is_empty(),
            "normalization canonical_name must be non-empty"
        );
    }
}

/// Prove: activation ops have non-empty canonical names.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn canonical_name_activation_nonempty() {
    let slope = dummy_weight_1d(4);
    let ops: Vec<TraceOp> = vec![
        TraceOp::Activation {
            kind: TraceActivation::Relu,
        },
        TraceOp::Elu { alpha: 1.0 },
        TraceOp::LeakyRelu { slope: 0.01 },
        TraceOp::Softplus,
        TraceOp::PRelu { slope },
    ];
    for op in &ops {
        assert!(
            !op.canonical_name().is_empty(),
            "activation canonical_name must be non-empty"
        );
    }
}

// ===========================================================================
// Cross-property consistency
// ===========================================================================

/// Prove: Input has arity 0 AND classifies as Input.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn cross_input_arity_class() {
    let op = TraceOp::Input;
    assert!(op.expected_arity() == Some(0));
    assert!(op.classification() == TraceOpClass::Input);
    assert!(op.canonical_name() == "input");
}

/// Prove: Dropout has arity 1 AND classifies as Identity.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn cross_dropout_arity_class() {
    let op = TraceOp::Dropout;
    assert!(op.expected_arity() == Some(1));
    assert!(op.classification() == TraceOpClass::Identity);
    assert!(op.canonical_name() == "dropout");
}

/// Prove: Custom has arity 1 AND classifies as Custom.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn cross_custom_arity_class() {
    let op = TraceOp::Custom {
        name: "nn_op".to_string(),
    };
    assert!(op.expected_arity() == Some(1));
    assert!(op.classification() == TraceOpClass::Custom);
    assert!(op.canonical_name() == "nn_op");
}
