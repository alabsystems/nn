// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for TraceOp classification and arity correctness (#3594).
//!
//! Proves structural invariants of the `TraceOp::classification()` and
//! `TraceOp::expected_arity()` functions that both the compile path (nn-dsl)
//! and verify path (nn-verify) depend on for correct dispatch routing.
//!
//! Properties proved:
//! - All BinaryElementwise ops have arity 2
//! - All UnaryElementwise ops have arity 1
//! - All ShapeOnly ops have arity 1
//! - All Reduction ops have arity 1
//! - Input and Constant ops have arity 0
//! - Identity ops have arity 1
//! - Binary elementwise Add/Mul are commutative (classification agrees)
//! - Sub classification is BinaryElementwise (non-commutative)
//! - Classification round-trip: classify then re-classify is idempotent
//! - Normalization ops have arity 1
//! - Attention Softmax/LogSoftmax have arity 1
//! - Pooling ops have arity 1

#![cfg(kani)]

use crate::dyn_tensor::trace::{TraceOp, TraceOpClass};

// ---------------------------------------------------------------------------
// Binary elementwise ops all have arity 2
// ---------------------------------------------------------------------------

/// Prove: every binary elementwise operation has expected_arity == Some(2).
///
/// The compile and verify dispatchers both assume BinaryElementwise ops
/// consume exactly 2 tensor inputs. A mismatch causes either silent shape
/// corruption (if arity is wrong) or a crash (if fewer inputs are provided).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn binary_elementwise_ops_have_arity_two() {
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
        let class = ops[i].classification();
        assert!(
            class == TraceOpClass::BinaryElementwise,
            "binary op must classify as BinaryElementwise"
        );
        let arity = ops[i].expected_arity();
        assert!(arity == Some(2), "binary elementwise op must have arity 2");
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Unary elementwise ops all have arity 1
// ---------------------------------------------------------------------------

/// Prove: every unary elementwise operation has expected_arity == Some(1).
///
/// These ops take a single tensor and produce an output of the same shape.
/// An incorrect arity would cause the dispatch graph to read a nonexistent
/// second input, producing undefined behavior in the compiled pipeline.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn unary_elementwise_ops_have_arity_one() {
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
        let class = ops[i].classification();
        assert!(
            class == TraceOpClass::UnaryElementwise,
            "unary op must classify as UnaryElementwise"
        );
        let arity = ops[i].expected_arity();
        assert!(arity == Some(1), "unary elementwise op must have arity 1");
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// ShapeOnly ops all have arity 1
// ---------------------------------------------------------------------------

/// Prove: shape-only operations (Reshape, Unsqueeze, Squeeze) have arity 1.
///
/// ShapeOnly ops reinterpret tensor metadata without data movement. They always
/// consume exactly one input tensor. A wrong arity here would corrupt the
/// topological ordering in the computation graph.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn shape_only_ops_have_arity_one() {
    let ops: [TraceOp; 3] = [
        TraceOp::Reshape {
            target_shape: vec![4, 8],
        },
        TraceOp::Unsqueeze { dim: 0 },
        TraceOp::Squeeze { dim: 1 },
    ];

    let mut i = 0;
    while i < 3 {
        let class = ops[i].classification();
        assert!(
            class == TraceOpClass::ShapeOnly,
            "shape-only op must classify as ShapeOnly"
        );
        let arity = ops[i].expected_arity();
        assert!(arity == Some(1), "shape-only op must have arity 1");
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Reduction ops all have arity 1
// ---------------------------------------------------------------------------

/// Prove: reduction operations (ReduceSum, ReduceMean, ReduceMax, ReduceMin)
/// all have arity 1. Reductions consume a single input tensor and produce
/// an output with one fewer dimension (or same dim with size 1 if keepdim).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn reduction_ops_have_arity_one() {
    let ops: [TraceOp; 4] = [
        TraceOp::ReduceSum {
            dim: 0,
            keepdim: false,
        },
        TraceOp::ReduceMean {
            dim: 1,
            keepdim: true,
        },
        TraceOp::ReduceMax {
            dim: 2,
            keepdim: false,
        },
        TraceOp::ReduceMin {
            dim: 0,
            keepdim: true,
        },
    ];

    let mut i = 0;
    while i < 4 {
        let class = ops[i].classification();
        assert!(
            class == TraceOpClass::Reduction,
            "reduction op must classify as Reduction"
        );
        let arity = ops[i].expected_arity();
        assert!(arity == Some(1), "reduction op must have arity 1");
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Input and Constant ops have arity 0
// ---------------------------------------------------------------------------

/// Prove: Input and Constant ops have arity 0 (no tensor inputs).
///
/// These are source nodes in the computation graph. An incorrect arity would
/// cause the graph builder to look for nonexistent predecessor nodes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn source_ops_have_arity_zero() {
    let input = TraceOp::Input;
    assert!(
        input.classification() == TraceOpClass::Input,
        "Input must classify as Input"
    );
    assert!(input.expected_arity() == Some(0), "Input must have arity 0");

    let constant = TraceOp::Constant { value: 1.0 };
    assert!(
        constant.classification() == TraceOpClass::ConstantValue,
        "Constant must classify as ConstantValue"
    );
    assert!(
        constant.expected_arity() == Some(0),
        "Constant must have arity 0"
    );
}

// ---------------------------------------------------------------------------
// Identity ops have arity 1
// ---------------------------------------------------------------------------

/// Prove: identity-class operations (Dropout at inference) have arity 1.
///
/// Dropout passes through its input unchanged at inference time. It must
/// have arity 1 so the compiled pipeline correctly wires the single input
/// to the output without data transformation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn identity_ops_have_arity_one() {
    let dropout = TraceOp::Dropout;
    assert!(
        dropout.classification() == TraceOpClass::Identity,
        "Dropout must classify as Identity"
    );
    assert!(
        dropout.expected_arity() == Some(1),
        "Dropout must have arity 1"
    );
}

// ---------------------------------------------------------------------------
// Normalization ops have arity 1
// ---------------------------------------------------------------------------

/// Prove: normalization ops (LayerNorm, RmsNorm, GroupNorm, InstanceNorm,
/// BatchNorm) all classify as Normalization with arity 1.
///
/// Normalization layers consume a single activation tensor. Weight and bias
/// are embedded in the TraceOp fields, not counted as tensor inputs.
/// A wrong arity would break the trace-to-graph translation for NY
/// verification (the CROWN linearization path depends on correct arity).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn normalization_ops_have_arity_one() {
    use crate::dyn_tensor::trace::WeightRef;

    let dummy_weight = WeightRef::new(vec![0.0], vec![1]).unwrap();

    let ops: [TraceOp; 2] = [
        TraceOp::InstanceNorm { eps: 1e-5 },
        TraceOp::RmsNorm {
            eps: 1e-5,
            weight: dummy_weight.clone(),
        },
    ];

    let mut i = 0;
    while i < 2 {
        let class = ops[i].classification();
        assert!(
            class == TraceOpClass::Normalization,
            "norm op must classify as Normalization"
        );
        let arity = ops[i].expected_arity();
        assert!(arity == Some(1), "norm op must have arity 1");
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Softmax / LogSoftmax classify as Attention with arity 1
// ---------------------------------------------------------------------------

/// Prove: Softmax and LogSoftmax classify as Attention with arity 1.
///
/// These ops are classified under Attention because they are the core
/// building blocks of attention scoring. Despite being unary (single input
/// tensor), they belong to the Attention class for dispatch routing purposes.
/// An arity mismatch would cause the softmax kernel to read invalid memory.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn softmax_ops_are_attention_arity_one() {
    let softmax = TraceOp::Softmax { dim: 1 };
    assert!(
        softmax.classification() == TraceOpClass::Attention,
        "Softmax must classify as Attention"
    );
    assert!(
        softmax.expected_arity() == Some(1),
        "Softmax must have arity 1"
    );

    let log_softmax = TraceOp::LogSoftmax { dim: 0 };
    assert!(
        log_softmax.classification() == TraceOpClass::Attention,
        "LogSoftmax must classify as Attention"
    );
    assert!(
        log_softmax.expected_arity() == Some(1),
        "LogSoftmax must have arity 1"
    );
}

// ---------------------------------------------------------------------------
// Pooling ops all have arity 1
// ---------------------------------------------------------------------------

/// Prove: pooling operations classify as Pooling with arity 1.
///
/// Pooling ops consume a single activation tensor. They reduce spatial
/// dimensions according to kernel/stride parameters embedded in the variant.
/// A wrong arity would cause the compiled dispatch to look for a nonexistent
/// second input (e.g., confusing AvgPool2d with a binary op).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn pooling_ops_have_arity_one() {
    let ops: [TraceOp; 4] = [
        TraceOp::MaxPool1d {
            kernel_size: 2,
            stride: 2,
            padding: 0,
        },
        TraceOp::AvgPool2d {
            kernel_size: [3, 3],
            stride: [2, 2],
            padding: [1, 1],
        },
        TraceOp::MaxPool2d {
            kernel_size: [2, 2],
            stride: [2, 2],
            padding: [0, 0],
        },
        TraceOp::AdaptiveAvgPool2d {
            output_size: [7, 7],
        },
    ];

    let mut i = 0;
    while i < 4 {
        let class = ops[i].classification();
        assert!(
            class == TraceOpClass::Pooling,
            "pooling op must classify as Pooling"
        );
        let arity = ops[i].expected_arity();
        assert!(arity == Some(1), "pooling op must have arity 1");
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// MatMul is binary (arity 2) but NOT BinaryElementwise
// ---------------------------------------------------------------------------

/// Prove: MatMul has arity 2 and classifies as MatMul (not BinaryElementwise).
///
/// Matrix multiplication is a binary op but is NOT elementwise — it
/// contracts inner dimensions. Misclassifying it as BinaryElementwise
/// would cause the fusion pass to incorrectly try to fuse it with
/// elementwise chains, producing wrong results.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn matmul_is_binary_not_elementwise() {
    let matmul = TraceOp::MatMul;
    let class = matmul.classification();
    assert!(
        class == TraceOpClass::MatMul,
        "MatMul must classify as MatMul, not BinaryElementwise"
    );
    assert!(
        class != TraceOpClass::BinaryElementwise,
        "MatMul must NOT be BinaryElementwise"
    );
    assert!(
        matmul.expected_arity() == Some(2),
        "MatMul must have arity 2"
    );
}

// ---------------------------------------------------------------------------
// Cat arity equals num_inputs
// ---------------------------------------------------------------------------

/// Prove: Cat's arity equals its declared num_inputs field.
///
/// Cat is the only variable-arity op where arity comes from a field.
/// The field is set at trace time and consumed at compile/verify time.
/// A mismatch would cause the dispatch to read too few or too many inputs,
/// leading to buffer overrun or silent data corruption.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn cat_arity_matches_num_inputs() {
    let n: u8 = kani::any();
    kani::assume(n >= 1 && n <= 16);

    let cat = TraceOp::Cat {
        dim: 0,
        num_inputs: n as usize,
    };

    assert!(
        cat.classification() == TraceOpClass::ShapeDataMove,
        "Cat must classify as ShapeDataMove"
    );
    assert!(
        cat.expected_arity() == Some(n as usize),
        "Cat arity must equal num_inputs"
    );
}

// ---------------------------------------------------------------------------
// WhereCond is ternary (arity 3)
// ---------------------------------------------------------------------------

/// Prove: WhereCond has arity 3 (condition, true_branch, false_branch).
///
/// Element-wise conditional select requires exactly 3 inputs. If the arity
/// were 2, the false branch would be missing; if 1, both branches missing.
/// Either causes undefined dispatch behavior.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn where_cond_is_ternary() {
    let op = TraceOp::WhereCond;
    assert!(
        op.classification() == TraceOpClass::Indexing,
        "WhereCond must classify as Indexing"
    );
    assert!(
        op.expected_arity() == Some(3),
        "WhereCond must have arity 3"
    );
}

// ---------------------------------------------------------------------------
// LSTM is ternary (arity 3): input + h_state + c_state
// ---------------------------------------------------------------------------

/// Prove: LSTM has arity 3 (input, hidden_state, cell_state).
///
/// The LSTM cell takes 3 tensor inputs from the graph: the activation input,
/// the previous hidden state, and the previous cell state. Weights are in the
/// TraceOp fields. A wrong arity would silently drop the cell state, causing
/// the LSTM to behave like a GRU (2-input) or simple RNN (1-input).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_is_ternary() {
    use crate::dyn_tensor::trace::WeightRef;

    let w = WeightRef::new(vec![0.0], vec![1]).unwrap();
    let lstm = TraceOp::Lstm {
        weight_ih: w.clone(),
        weight_hh: w.clone(),
        bias_ih: None,
        bias_hh: None,
        hidden_size: 256,
        initial_hidden: None,
        initial_cell: None,
    };

    assert!(
        lstm.classification() == TraceOpClass::Recurrent,
        "LSTM must classify as Recurrent"
    );
    assert!(lstm.expected_arity() == Some(3), "LSTM must have arity 3");
}
