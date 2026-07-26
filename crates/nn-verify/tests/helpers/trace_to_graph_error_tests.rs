// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error-path integration tests for the trace-to-graph translator
//! (NY-owned via ny-trace-bridge).
//!
//! Tests exercise `trace_to_graph_model()` with programmatically constructed
//! `ComputationGraph` instances that trigger each defense-in-depth guard:
//!
//! - Empty graph
//! - Unsupported ops (MatMul, SwiGlu, Custom, Powf, AdaptiveAvgPool2d, etc.)
//! - Missing input node references
//! - Weight validation (empty data, non-finite values)
//! - Normalization eps validation (NaN, non-positive, f32 overflow)
//! - LSTM weight shape mismatches
//! - Unknown activation names
//! - Transpose dim out-of-range
//!
//! Part of #2146.

use nn_core::dyn_tensor::trace::{
    ComputationGraph, TraceActivation, TraceNode, TraceOp, WeightRef,
};
use nn_core::DType;
use nn_verify::{trace_to_graph_model, trace_to_graph_model_multi_input};

/// Helper: build a graph with one Input node and one op node consuming it.
fn graph_with_unary_op(op: TraceOp, output_shape: Vec<usize>) -> ComputationGraph {
    ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(1, "op_0".into(), op, vec![0], output_shape, DType::F32),
    ])
}

/// Helper: build a graph with two Input nodes and one binary op node.
fn graph_with_binary_op(op: TraceOp, output_shape: Vec<usize>) -> ComputationGraph {
    ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "input_1".into(),
            TraceOp::Input,
            vec![],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(2, "op_0".into(), op, vec![0, 1], output_shape, DType::F32),
    ])
}

// ---------------------------------------------------------------------------
// Top-level: empty graph
// ---------------------------------------------------------------------------

#[test]
fn test_empty_graph_rejected() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let err = trace_to_graph_model(&graph).expect_err("empty graph should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("empty"),
        "error should mention empty, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Unsupported ops
// ---------------------------------------------------------------------------

// MatMul is now supported via MatMulLayer (commit 01ece04).
// Full IBP round-trip test is in trace_to_graph_model_binary_ops.rs.
// Note: trace_to_graph_model rejects graphs with >1 variable input (#2425).
// Binary ops need multi_input mode to translate both operands as variables.
#[test]
fn test_raw_matmul_multi_input_accepted() {
    let graph = graph_with_binary_op(TraceOp::MatMul, vec![2, 4]);
    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("MatMul should be accepted (multi-input)")
        .graph;
    assert!(
        gn.num_nodes() >= 2,
        "MatMul graph should have at least 2 nodes"
    );
}

#[test]
fn test_swiglu_rejected() {
    let graph = graph_with_unary_op(TraceOp::SwiGlu, vec![2, 4]);
    let err = trace_to_graph_model(&graph).expect_err("SwiGlu should be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("SwiGlu") || msg.contains("not supported"),
        "error should mention SwiGlu, got: {msg}"
    );
}

/// An unknown `TraceOp::Custom` op is accepted as a conservative `OpaqueSkip`
/// layer rather than hard-failing (intended design, #4349).
///
/// SOUNDNESS: `OpaqueSkipLayer`'s IBP rule returns `[-inf, +inf]` (verified in
/// ny-propagate `skip_merge.rs`) — a sound *over-approximation* of an arbitrary
/// unknown op, NOT a silent identity passthrough (which would be unsound, since
/// a real custom op can change values it would not bound). We assert the output
/// is widened far beyond the finite input interval to prove the
/// over-approximation is in effect. Accepting unknown ops as OpaqueSkip lets a
/// model containing a single unknown op still verify soundly instead of being
/// rejected outright.
#[test]
fn test_custom_op_accepted_as_opaque_skip() {
    let graph = graph_with_unary_op(
        TraceOp::Custom {
            name: "nn_fancy_op".to_string(),
        },
        vec![2, 4],
    );
    let gn = trace_to_graph_model(&graph)
        .expect("Custom op should translate to a conservative OpaqueSkip layer")
        .graph;

    // Soundness: OpaqueSkip over-approximates rather than passing through. The
    // [2,4] input is bounded in [-1, 1]; the unknown op's output must be much
    // wider (a sound superset of any possible custom-op output).
    let input_bounds = super::common::uniform_bounds(&[2, 4], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP over OpaqueSkip");
    let (lo, hi) = output.lower_upper();
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l <= u, "OpaqueSkip lower <= upper: [{l}, {u}]");
        assert!(
            u - l > 1e3,
            "OpaqueSkip must over-approximate the unknown op (width >> input \
             width of 2), got [{l}, {u}] — a near-identity passthrough would be \
             UNSOUND for an arbitrary custom op"
        );
    }
}

#[test]
fn test_powf_general_accepted() {
    // Powf(3.0) is now supported via Exp(n*Log(Abs(x))) decomposition (#3557).
    let graph = graph_with_unary_op(TraceOp::Powf { exponent: 3.0 }, vec![2, 4]);
    trace_to_graph_model(&graph).expect("Powf(3.0) should be accepted");
}

#[test]
fn test_adaptive_avg_pool2d_rejected() {
    let graph = graph_with_unary_op(
        TraceOp::AdaptiveAvgPool2d {
            output_size: [1, 1],
        },
        vec![1, 3, 1, 1],
    );
    let err = trace_to_graph_model(&graph).expect_err("AdaptiveAvgPool2d should be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("AdaptiveAvgPool2d") || msg.contains("not supported"),
        "error should mention AdaptiveAvgPool2d, got: {msg}"
    );
}

#[test]
fn test_index_select_rejected() {
    let graph = graph_with_unary_op(TraceOp::IndexSelect { dim: 0 }, vec![2, 4]);
    let err = trace_to_graph_model(&graph).expect_err("IndexSelect should be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("IndexSelect") || msg.contains("not supported") || msg.contains("Gather"),
        "error should mention IndexSelect or Gather, got: {msg}"
    );
}

/// WhereCond fixture: 3 inputs (cond, true_branch, false_branch).
fn where_cond_graph() -> ComputationGraph {
    ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "cond".into(),
            TraceOp::Input,
            vec![],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "a".into(),
            TraceOp::Input,
            vec![],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "b".into(),
            TraceOp::Input,
            vec![],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            3,
            "where".into(),
            TraceOp::WhereCond,
            vec![0, 1, 2],
            vec![2, 4],
            DType::F32,
        ),
    ])
}

/// INC-FINAL soundness fix: WhereCond is REFUSED.
///
/// The deleted legacy `m·x + (1−m)·y` decomposition was UNSOUND whenever a
/// realizable mask value is not in {0, 1}: a constant 0.5 "mask" makes the
/// true output exactly `x`, but the decomposition yields the x/y midpoint —
/// bounds that can exclude the real output (a false "holds"). The bridge
/// classifies WhereCond `Unsupported` and refuses fail-closed.
#[test]
fn test_where_cond_refused_sound() {
    // WhereCond has 3 inputs (cond, true_branch, false_branch) — needs multi-input mode.
    let err = trace_to_graph_model_multi_input(&where_cond_graph())
        .expect_err("WhereCond must be refused (unsound legacy decomposition)");
    let msg = err.to_string();
    assert!(
        msg.contains("WhereCond") && msg.contains("not supported"),
        "refusal should name WhereCond, got: {msg}"
    );
}

#[test]
fn test_to_dtype_translated_as_identity() {
    // ToDtype downcasts (F16/BF16) use Clamp; upcasts (F32/F64) use CastLayer.
    // BF16 downcast: Clamp to representable range.
    let graph = graph_with_unary_op(
        TraceOp::ToDtype {
            target_dtype: DType::BF16,
        },
        vec![2, 4],
    );
    let gn = trace_to_graph_model(&graph)
        .expect("ToDtype should translate as identity")
        .graph;
    assert!(
        gn.num_nodes() >= 1,
        "ToDtype should produce at least one node"
    );
}

// ---------------------------------------------------------------------------
// dtype_cast_count integration (Part of #3023)
// ---------------------------------------------------------------------------

#[test]
fn test_trace_translate_result_f16_cast_count() {
    // Graph: Input -> ToDtype(F16) -> output
    // Should report dtype_cast_count == 1
    let graph = graph_with_unary_op(
        TraceOp::ToDtype {
            target_dtype: DType::F16,
        },
        vec![2, 4],
    );
    let result = trace_to_graph_model(&graph).expect("F16 cast should translate");
    assert_eq!(
        result.dtype_cast_count, 1,
        "F16 downcast should report cast_count=1"
    );
    assert!(result.graph.num_nodes() >= 1);
}

#[test]
fn test_trace_translate_result_f32_cast_count_zero() {
    // Graph: Input -> ToDtype(F32) -> output
    // Should report dtype_cast_count == 0 (upcast is identity)
    let graph = graph_with_unary_op(
        TraceOp::ToDtype {
            target_dtype: DType::F32,
        },
        vec![2, 4],
    );
    let result = trace_to_graph_model(&graph).expect("F32 upcast should translate");
    assert_eq!(
        result.dtype_cast_count, 0,
        "F32 upcast should report cast_count=0"
    );
}

#[test]
fn test_trace_translate_result_no_casts_count_zero() {
    // Graph: Input -> Relu -> output (no dtype casts at all)
    let graph = graph_with_unary_op(TraceOp::Relu, vec![2, 4]);
    let result = trace_to_graph_model(&graph).expect("Relu should translate");
    assert_eq!(
        result.dtype_cast_count, 0,
        "No casts should report cast_count=0"
    );
}

// ---------------------------------------------------------------------------
// Weight validation: empty data (shape-only capture)
// ---------------------------------------------------------------------------

#[test]
fn test_linear_empty_weight_rejected() {
    // WeightRef::from_shape creates a shape-only reference with no data.
    let graph = graph_with_unary_op(
        TraceOp::Linear {
            weight: WeightRef::from_shape(&[4, 4]),
            bias: None,
        },
        vec![2, 4],
    );
    let err = trace_to_graph_model(&graph).expect_err("empty weight should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("empty") || msg.contains("shape-only"),
        "error should mention empty/shape-only, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Weight validation: non-finite values
// ---------------------------------------------------------------------------

#[test]
fn test_linear_nan_weight_rejected() {
    let mut data = vec![1.0_f32; 16];
    data[7] = f32::NAN;
    let graph = graph_with_unary_op(
        TraceOp::Linear {
            weight: WeightRef::new(data, vec![4, 4]).expect("valid shape"),
            bias: None,
        },
        vec![2, 4],
    );
    let err = trace_to_graph_model(&graph).expect_err("NaN weight should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite"),
        "error should mention non-finite, got: {msg}"
    );
}

#[test]
fn test_linear_inf_weight_rejected() {
    let mut data = vec![0.5_f32; 16];
    data[0] = f32::INFINITY;
    let graph = graph_with_unary_op(
        TraceOp::Linear {
            weight: WeightRef::new(data, vec![4, 4]).expect("valid shape"),
            bias: None,
        },
        vec![2, 4],
    );
    let err = trace_to_graph_model(&graph).expect_err("Inf weight should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite"),
        "error should mention non-finite, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Normalization: eps validation
// ---------------------------------------------------------------------------

#[test]
fn test_layer_norm_nan_eps_rejected() {
    let weight = WeightRef::new(vec![1.0; 4], vec![4]).expect("valid weight");
    let bias = WeightRef::new(vec![0.0; 4], vec![4]).expect("valid bias");
    let graph = graph_with_unary_op(
        TraceOp::LayerNorm {
            eps: f64::NAN,
            weight,
            bias,
        },
        vec![2, 4],
    );
    let err = trace_to_graph_model(&graph).expect_err("NaN eps should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite") || msg.contains("eps"),
        "error should mention eps/non-finite, got: {msg}"
    );
}

#[test]
fn test_rms_norm_negative_eps_rejected() {
    let weight = WeightRef::new(vec![1.0; 4], vec![4]).expect("valid weight");
    let graph = graph_with_unary_op(TraceOp::RmsNorm { eps: -1e-5, weight }, vec![2, 4]);
    let err = trace_to_graph_model(&graph).expect_err("negative eps should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("positive") || msg.contains("eps"),
        "error should mention eps/positive, got: {msg}"
    );
}

#[test]
fn test_group_norm_non_divisible_channels_rejected() {
    // 4 channels / 3 groups is not divisible.
    let weight = WeightRef::new(vec![1.0; 4], vec![4]).expect("valid weight");
    let bias = WeightRef::new(vec![0.0; 4], vec![4]).expect("valid bias");
    let graph = graph_with_unary_op(
        TraceOp::GroupNorm {
            num_groups: 3,
            eps: 1e-5,
            weight,
            bias,
        },
        vec![1, 4, 8],
    );
    let err = trace_to_graph_model(&graph).expect_err("non-divisible channels should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("divisible") || msg.contains("num_groups"),
        "error should mention divisibility, got: {msg}"
    );
}

#[test]
fn test_instance_norm_scalar_output_rejected() {
    // Instance norm on a scalar (0D) output cannot infer num_channels.
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "inorm".into(),
            TraceOp::InstanceNorm { eps: 1e-5 },
            vec![0],
            vec![4],
            DType::F32,
        ),
    ]);
    let err = trace_to_graph_model(&graph).expect_err("1D InstanceNorm should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("cannot infer") || msg.contains("scalar") || msg.contains("1D"),
        "error should mention shape limitation, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// LSTM: weight shape mismatches
// ---------------------------------------------------------------------------

#[test]
fn test_lstm_weight_ih_wrong_rows_rejected() {
    let hidden_size = 3;
    let input_size = 4;
    let gate_size = 4 * hidden_size; // 12
                                     // weight_ih should be [gate_size, input_size] = [12, 4] = 48 elements.
                                     // Instead, give [8, 4] = 32 elements (wrong gate_size).
    let weight_ih = WeightRef::new(vec![1.0; 32], vec![8, input_size]).expect("test data");
    let weight_hh = WeightRef::new(
        vec![1.0; gate_size * hidden_size],
        vec![gate_size, hidden_size],
    )
    .expect("test data");

    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "x".into(),
            TraceOp::Input,
            vec![],
            vec![1, input_size],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "h".into(),
            TraceOp::Input,
            vec![],
            vec![1, hidden_size],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "c".into(),
            TraceOp::Input,
            vec![],
            vec![1, hidden_size],
            DType::F32,
        ),
        TraceNode::new(
            3,
            "lstm".into(),
            TraceOp::Lstm {
                weight_ih,
                weight_hh,
                bias_ih: None,
                bias_hh: None,
                hidden_size,
                initial_hidden: None,
                initial_cell: None,
            },
            vec![0, 1, 2],
            vec![1, hidden_size],
            DType::F32,
        ),
    ]);
    let err = trace_to_graph_model(&graph).expect_err("LSTM bad weight_ih should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("weight_ih") || msg.contains("4*hidden_size") || msg.contains("rows"),
        "error should mention weight_ih mismatch, got: {msg}"
    );
}

#[test]
fn test_lstm_bias_length_mismatch_rejected() {
    let hidden_size = 3;
    let input_size = 4;
    let gate_size = 4 * hidden_size;
    let weight_ih = WeightRef::new(
        vec![1.0; gate_size * input_size],
        vec![gate_size, input_size],
    )
    .expect("test data");
    let weight_hh = WeightRef::new(
        vec![1.0; gate_size * hidden_size],
        vec![gate_size, hidden_size],
    )
    .expect("test data");
    // bias_ih should have gate_size (12) elements, give 8 instead.
    let bias_ih = Some(WeightRef::new(vec![0.0; 8], vec![8]).expect("test data"));

    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "x".into(),
            TraceOp::Input,
            vec![],
            vec![1, input_size],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "h".into(),
            TraceOp::Input,
            vec![],
            vec![1, hidden_size],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "c".into(),
            TraceOp::Input,
            vec![],
            vec![1, hidden_size],
            DType::F32,
        ),
        TraceNode::new(
            3,
            "lstm".into(),
            TraceOp::Lstm {
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh: None,
                hidden_size,
                initial_hidden: None,
                initial_cell: None,
            },
            vec![0, 1, 2],
            vec![1, hidden_size],
            DType::F32,
        ),
    ]);
    let err = trace_to_graph_model(&graph).expect_err("LSTM bad bias_ih should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("bias") || msg.contains("length"),
        "error should mention bias mismatch, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Activation: generic-path Mish (INC-FINAL reconciliation)
// ---------------------------------------------------------------------------

/// `Activation { Mish }` is ACCEPTED through the generic path, routing to
/// the same Clip(-88, 88) + Mish lowering the dedicated `TraceOp::Mish`
/// variant uses. Mish is parameterless, so — unlike Elu/LeakyRelu, where the
/// generic path would silently default alpha/slope — nothing can be lost.
///
/// Historical note (INC-FINAL reconciliation): this replaced
/// `test_unknown_activation_rejected`, which used Mish as the "unknown"
/// kind. After the Mish arm landed, every `TraceActivation` kind is either
/// mapped or deliberately refused; the generic-path red-path contract is
/// covered by `test_named_activation_elu_rejected` /
/// `test_named_activation_leaky_relu_rejected` below.
#[test]
fn test_named_activation_mish_accepted() {
    let graph = graph_with_unary_op(
        TraceOp::Activation {
            kind: TraceActivation::Mish,
        },
        vec![2, 4],
    );
    let gn = trace_to_graph_model(&graph)
        .expect("named activation 'mish' should translate")
        .graph;
    // Clip(-88, 88) domain guard + Mish — at least two layers beyond input.
    assert!(
        gn.num_nodes() >= 2,
        "Mish lowering should include the domain clip, got {} nodes",
        gn.num_nodes()
    );
}

// ---------------------------------------------------------------------------
// Named activation: all 10 variants succeed through verify path
// Part of #2258: parity between compile and verify named activation dispatch.
// ---------------------------------------------------------------------------

#[test]
fn test_named_activation_gelu_accepted() {
    let graph = graph_with_unary_op(
        TraceOp::Activation {
            kind: TraceActivation::Gelu,
        },
        vec![2, 4],
    );
    trace_to_graph_model(&graph).expect("named activation 'gelu' should translate");
}

#[test]
fn test_named_activation_gelu_erf_accepted() {
    let graph = graph_with_unary_op(
        TraceOp::Activation {
            kind: TraceActivation::GeluErf,
        },
        vec![2, 4],
    );
    trace_to_graph_model(&graph).expect("named activation 'gelu_erf' should translate");
}

#[test]
fn test_named_activation_relu_accepted() {
    let graph = graph_with_unary_op(
        TraceOp::Activation {
            kind: TraceActivation::Relu,
        },
        vec![2, 4],
    );
    trace_to_graph_model(&graph).expect("named activation 'relu' should translate");
}

#[test]
fn test_named_activation_sigmoid_accepted() {
    let graph = graph_with_unary_op(
        TraceOp::Activation {
            kind: TraceActivation::Sigmoid,
        },
        vec![2, 4],
    );
    trace_to_graph_model(&graph).expect("named activation 'sigmoid' should translate");
}

#[test]
fn test_named_activation_silu_accepted() {
    let graph = graph_with_unary_op(
        TraceOp::Activation {
            kind: TraceActivation::Silu,
        },
        vec![2, 4],
    );
    trace_to_graph_model(&graph).expect("named activation 'silu' should translate");
}

#[test]
fn test_named_activation_tanh_accepted() {
    let graph = graph_with_unary_op(
        TraceOp::Activation {
            kind: TraceActivation::Tanh,
        },
        vec![2, 4],
    );
    trace_to_graph_model(&graph).expect("named activation 'tanh' should translate");
}

#[test]
fn test_named_activation_exp_accepted() {
    let graph = graph_with_unary_op(
        TraceOp::Activation {
            kind: TraceActivation::Exp,
        },
        vec![2, 4],
    );
    trace_to_graph_model(&graph).expect("named activation 'exp' should translate");
}

#[test]
fn test_named_activation_log_accepted() {
    let graph = graph_with_unary_op(
        TraceOp::Activation {
            kind: TraceActivation::Log,
        },
        vec![2, 4],
    );
    trace_to_graph_model(&graph).expect("named activation 'log' should translate");
}

#[test]
fn test_named_activation_elu_rejected() {
    // Generic Activation { Elu } is rejected because it uses hardcoded alpha=1.0.
    // Callers must use TraceOp::Elu { alpha } instead (#2267).
    let graph = graph_with_unary_op(
        TraceOp::Activation {
            kind: TraceActivation::Elu,
        },
        vec![2, 4],
    );
    let err = trace_to_graph_model(&graph).expect_err("generic Elu should be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("Elu") && msg.contains("rejected"),
        "error should mention Elu rejection, got: {msg}"
    );
}

#[test]
fn test_named_activation_leaky_relu_rejected() {
    // Generic Activation { LeakyRelu } is rejected because it uses a hardcoded
    // slope=0.01. Callers must use TraceOp::LeakyRelu { slope } instead (#2267).
    let graph = graph_with_unary_op(
        TraceOp::Activation {
            kind: TraceActivation::LeakyRelu,
        },
        vec![2, 4],
    );
    let err = trace_to_graph_model(&graph).expect_err("generic LeakyRelu should be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("LeakyRelu") && msg.contains("rejected"),
        "error should mention LeakyRelu rejection, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Shape: transpose dim out of range
// ---------------------------------------------------------------------------

#[test]
fn test_transpose_dim_out_of_range_rejected() {
    let graph = graph_with_unary_op(
        TraceOp::Transpose { dim0: 0, dim1: 5 }, // dim1=5 exceeds rank 2
        vec![4, 2],
    );
    let err = trace_to_graph_model(&graph).expect_err("transpose OOB dim should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("dim") || msg.contains("exceeds") || msg.contains("range"),
        "error should mention dim out of range, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Embedding: non-2D weight
// ---------------------------------------------------------------------------

#[test]
fn test_embedding_non_2d_weight_rejected() {
    // Embedding weight must be 2D [vocab_size, embed_dim].
    let graph = graph_with_unary_op(
        TraceOp::Embedding {
            weight: WeightRef::new(vec![1.0; 24], vec![2, 3, 4]).expect("valid shape"),
        },
        vec![2, 4],
    );
    let err = trace_to_graph_model(&graph).expect_err("non-2D embedding should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("2D") || msg.contains("Embedding"),
        "error should mention 2D requirement, got: {msg}"
    );
}

#[test]
fn test_embedding_empty_weight_rejected() {
    let graph = graph_with_unary_op(
        TraceOp::Embedding {
            weight: WeightRef::from_shape(&[10, 4]),
        },
        vec![2, 4],
    );
    let err = trace_to_graph_model(&graph).expect_err("empty embedding weight should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("empty") || msg.contains("Embedding"),
        "error should mention empty weight, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Sub: bounds-checked inputs
// ---------------------------------------------------------------------------

#[test]
fn test_sub_no_inputs_rejected() {
    let graph = ComputationGraph::from_nodes(vec![TraceNode::new(
        0,
        "sub".into(),
        TraceOp::Sub,
        vec![],
        vec![2, 4],
        DType::F32,
    )]);
    let err = trace_to_graph_model(&graph).expect_err("Sub with no inputs should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("Sub") || msg.contains("no inputs"),
        "error should mention Sub/no inputs, got: {msg}"
    );
}

#[test]
fn test_sub_one_input_rejected() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "sub".into(),
            TraceOp::Sub,
            vec![0],
            vec![2, 4],
            DType::F32,
        ),
    ]);
    let err = trace_to_graph_model(&graph).expect_err("Sub with one input should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("Sub") || msg.contains("two inputs"),
        "error should mention Sub/two inputs, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Sqr: bounds-checked input
// ---------------------------------------------------------------------------

#[test]
fn test_sqr_no_inputs_rejected() {
    let graph = ComputationGraph::from_nodes(vec![TraceNode::new(
        0,
        "sqr".into(),
        TraceOp::Sqr,
        vec![],
        vec![2, 4],
        DType::F32,
    )]);
    let err = trace_to_graph_model(&graph).expect_err("Sqr with no inputs should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("Sqr") || msg.contains("no inputs"),
        "error should mention Sqr/no inputs, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Dropout: no inputs
// ---------------------------------------------------------------------------

#[test]
fn test_dropout_no_inputs_rejected() {
    let graph = ComputationGraph::from_nodes(vec![TraceNode::new(
        0,
        "drop".into(),
        TraceOp::Dropout,
        vec![],
        vec![2, 4],
        DType::F32,
    )]);
    let err = trace_to_graph_model(&graph).expect_err("dropout with no inputs should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("input") || msg.contains("Dropout"),
        "error should mention missing input, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Topology validation: forward references
// ---------------------------------------------------------------------------

#[test]
fn test_forward_reference_rejected_by_topology_validation() {
    // Node 1 references node 2 which appears later — a forward reference.
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "add_forward_ref".into(),
            TraceOp::Add,
            vec![0, 2], // references node 2, not yet seen
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "relu_0".into(),
            TraceOp::Relu,
            vec![0],
            vec![2, 4],
            DType::F32,
        ),
    ]);
    let err = trace_to_graph_model(&graph).expect_err("forward reference should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("topology") || msg.contains("not appeared earlier"),
        "error should mention topology violation, got: {msg}"
    );
}

#[test]
fn test_unreachable_forward_ref_still_rejected() {
    // Node 1 has a forward reference to node 2, but node 1 is unreachable
    // from the output (node 3). Before the topology fix, this was silently
    // accepted because reachable_nodes() skipped node 1.  Now
    // validate_topology() runs before reachability filtering.
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "add_unreachable".into(),
            TraceOp::Add,
            vec![0, 2], // forward reference to node 2
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "relu_0".into(),
            TraceOp::Relu,
            vec![0],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            3,
            "sqrt_out".into(),
            TraceOp::Sqrt,
            vec![2], // output only depends on node 2, not node 1
            vec![2, 4],
            DType::F32,
        ),
    ]);
    let err = trace_to_graph_model(&graph)
        .expect_err("forward reference should be caught even if node is unreachable from output");
    let msg = err.to_string();
    assert!(
        msg.contains("topology") || msg.contains("not appeared earlier"),
        "error should mention topology violation, got: {msg}"
    );
}
