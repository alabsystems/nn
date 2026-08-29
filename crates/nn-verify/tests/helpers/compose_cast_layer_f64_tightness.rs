// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CastLayer compose tests: ToDtype translation policy through trace_to_graph.
//!
//! Posture change (ny hardening 25dee0c5e): the bridge's former lossy-cast
//! surrogates (Clamp for F16/BF16 downcasts, identity CastLayer for F64
//! upcasts) were UNSOUND — the rounding semantics of the non-F32 endpoint are
//! not modeled, so the emitted bounds could exclude the true cast output.
//! Today exactly one ToDtype combination lowers: the F32 -> F32 identity
//! (restored as an `Add(x, 0)` identity). Every other resolved combination is
//! refused fail-closed by `translate_to_dtype`
//! (`ToDtype {src} -> {tgt} is refused: rounding semantics ... not modeled`);
//! a well-formed trace whose cast node records its non-F32 output dtype is
//! refused even earlier by the graph-wide F32 gate ("the bridge only soundly
//! supports F32 tensors" — asserted in `certify_tests_inline`).
//!
//! The former F16/BF16/F64 clamp/roundtrip tightness measurements are
//! meaningless under refusal; one representative refusal test per dtype
//! remains so a future un-refusal is a conscious, test-visible event.
//!
//! Historical: #4316 (CastLayer for ToDtype verification + f64 evaluation).

use super::common::{assert_bounds_valid, uniform_bounds};
use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;
use nn_verify::{
    propagate_with_crown_fallback, trace_to_graph_model, PropMethod, VerifyError,
};
use ny_api::NyError;

// ---------------------------------------------------------------------------
// Graph construction helpers
// ---------------------------------------------------------------------------

/// Build a graph: Input -> op -> output.
fn graph_with_unary_op(op: TraceOp, shape: Vec<usize>) -> ComputationGraph {
    ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(1, "op_0".into(), op, vec![0], shape, DType::F32),
    ])
}

/// Build a graph: Input -> Relu -> ToDtype -> output.
fn graph_relu_then_cast(target_dtype: DType, shape: Vec<usize>) -> ComputationGraph {
    ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            1,
            "relu_0".into(),
            TraceOp::Relu,
            vec![0],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            2,
            "cast_0".into(),
            TraceOp::ToDtype { target_dtype },
            vec![1],
            shape,
            DType::F32,
        ),
    ])
}

/// Build a graph: Input -> ToDtype -> Relu -> output.
fn graph_cast_then_relu(target_dtype: DType, shape: Vec<usize>) -> ComputationGraph {
    ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            1,
            "cast_0".into(),
            TraceOp::ToDtype { target_dtype },
            vec![0],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            2,
            "relu_0".into(),
            TraceOp::Relu,
            vec![1],
            shape,
            DType::F32,
        ),
    ])
}

/// Assert that translating `graph` refuses fail-closed with the bridge's
/// resolved-source ToDtype refusal, naming both dtypes and the unmodeled
/// endpoint (`expect_dtype`, e.g. "F16").
///
/// Fixtures here record the cast node's output dtype as F32 (they predate the
/// hardening), so the graph-wide F32 gate passes and the refusal comes from
/// `translate_to_dtype`'s deliberately independent resolved-combination check
/// (ny 25dee0c5e: a later gate relaxation must not silently turn a lossy cast
/// into an identity).
fn assert_cast_refused(graph: &ComputationGraph, expect_dtype: &str) {
    let err = trace_to_graph_model(graph)
        .expect_err("non-F32 ToDtype must be refused fail-closed (ny 25dee0c5e)");
    match err {
        VerifyError::Ny(NyError::UnsupportedOp(msg)) => {
            assert!(
                msg.contains(&format!("ToDtype F32 -> {expect_dtype} is refused")),
                "refusal must name the resolved source and target dtypes, got: {msg}"
            );
            assert!(
                msg.contains(&format!(
                    "rounding semantics of the {expect_dtype} target are not modeled"
                )),
                "refusal must state the unmodeled-rounding rationale, got: {msg}"
            );
        }
        other => panic!("expected VerifyError::Ny(UnsupportedOp), got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 1. CastLayer (F32 identity) IBP propagation
// ---------------------------------------------------------------------------

/// F32 -> F32 identity cast (the one lowerable ToDtype) propagates IBP bounds
/// unchanged.
#[test]
fn test_castlayer_f32_ibp_propagation() {
    let shape = vec![2, 4];
    let graph = graph_with_unary_op(
        TraceOp::ToDtype {
            target_dtype: DType::F32,
        },
        shape.clone(),
    );

    let result = trace_to_graph_model(&graph).expect("F32 CastLayer translate");
    assert_eq!(result.dtype_cast_count, 0, "F32 identity = no cast count");

    let input = uniform_bounds(&shape, 2.0);
    let output = result
        .graph
        .propagate_ibp(&input)
        .expect("IBP through CastLayer");
    assert_bounds_valid(&output);

    let (in_lo, in_hi) = input.lower_upper();
    let (out_lo, out_hi) = output.lower_upper();
    let eps = 1e-5;
    for (i, ((&il, &ih), (&ol, &oh))) in in_lo
        .iter()
        .zip(in_hi.iter())
        .zip(out_lo.iter().zip(out_hi.iter()))
        .enumerate()
    {
        assert!(
            (il - ol).abs() < eps,
            "CastLayer lower[{i}]: input={il}, output={ol}"
        );
        assert!(
            (ih - oh).abs() < eps,
            "CastLayer upper[{i}]: input={ih}, output={oh}"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Non-F32 casts: one representative fail-closed refusal per dtype
// ---------------------------------------------------------------------------
//
// These replace the former F64 identity-CastLayer and F16/BF16 downcast-Clamp
// tightness tests (in-range clamp, out-of-range clamp, F16 roundtrip). Their
// bound-value assertions measured surrogates the bridge no longer emits.

/// ToDtype(F64) is refused fail-closed: F64 rounding semantics are unmodeled,
/// so the former identity-CastLayer lowering was unsound (ny 25dee0c5e).
#[test]
fn test_f64_cast_refused_fail_closed() {
    let graph = graph_with_unary_op(
        TraceOp::ToDtype {
            target_dtype: DType::F64,
        },
        vec![2, 4],
    );
    assert_cast_refused(&graph, "F64");
}

/// ToDtype(F16) is refused fail-closed: F16 rounding semantics are unmodeled,
/// so the former downcast-Clamp lowering was unsound (ny 25dee0c5e).
#[test]
fn test_f16_cast_refused_fail_closed() {
    let graph = graph_with_unary_op(
        TraceOp::ToDtype {
            target_dtype: DType::F16,
        },
        vec![2, 4],
    );
    assert_cast_refused(&graph, "F16");
}

/// ToDtype(BF16) is refused fail-closed: BF16 rounding semantics are
/// unmodeled, so the former downcast-Clamp lowering was unsound (ny
/// 25dee0c5e). (The bridge dtype enum spells it `Bf16`.)
#[test]
fn test_bf16_cast_refused_fail_closed() {
    let graph = graph_with_unary_op(
        TraceOp::ToDtype {
            target_dtype: DType::BF16,
        },
        vec![2, 4],
    );
    assert_cast_refused(&graph, "Bf16");
}

// ---------------------------------------------------------------------------
// 3. CastLayer in pipeline: Relu -> Cast(F32) -> output
// ---------------------------------------------------------------------------

/// CastLayer (F32) after Relu: output matches Relu-only bounds.
#[test]
fn test_relu_then_castlayer_f32_ibp() {
    let shape = vec![2, 3];

    let graph_with = graph_relu_then_cast(DType::F32, shape.clone());
    let result_with = trace_to_graph_model(&graph_with).expect("Relu+Cast translate");
    let input = uniform_bounds(&shape, 2.0);
    let out_with = result_with
        .graph
        .propagate_ibp(&input)
        .expect("IBP with cast");
    assert_bounds_valid(&out_with);

    let graph_no = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            1,
            "relu_0".into(),
            TraceOp::Relu,
            vec![0],
            shape,
            DType::F32,
        ),
    ]);
    let result_no = trace_to_graph_model(&graph_no).expect("Relu-only translate");
    let out_no = result_no
        .graph
        .propagate_ibp(&input)
        .expect("IBP without cast");
    assert_bounds_valid(&out_no);

    let (lo_with, hi_with) = out_with.lower_upper();
    let (lo_no, hi_no) = out_no.lower_upper();
    let eps = 1e-5;
    for (i, ((&lw, &ln), (&hw, &hn))) in lo_with
        .iter()
        .zip(lo_no.iter())
        .zip(hi_with.iter().zip(hi_no.iter()))
        .enumerate()
    {
        assert!(
            (lw - ln).abs() < eps,
            "Relu+Cast lower[{i}]={lw} vs Relu={ln}"
        );
        assert!(
            (hw - hn).abs() < eps,
            "Relu+Cast upper[{i}]={hw} vs Relu={hn}"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. CastLayer before activation: Cast(F32) -> Relu
// ---------------------------------------------------------------------------

/// CastLayer (F32) before Relu: transparent, bounds match Relu-only.
#[test]
fn test_castlayer_f32_then_relu_ibp() {
    let shape = vec![2, 3];
    let graph = graph_cast_then_relu(DType::F32, shape.clone());
    let result = trace_to_graph_model(&graph).expect("Cast+Relu translate");
    let input = uniform_bounds(&shape, 2.0);
    let output = result.graph.propagate_ibp(&input).expect("IBP Cast+Relu");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    let eps = 1e-4;
    for &v in lo.iter() {
        assert!(v >= -eps, "Cast+Relu lower >= 0, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 2.0 + eps, "Cast+Relu upper <= 2, got {v}");
    }
}

// ---------------------------------------------------------------------------
// 5. CROWN propagation through CastLayer
// ---------------------------------------------------------------------------

/// CROWN through CastLayer (F32) pipeline: coefficients pass unchanged.
#[test]
fn test_castlayer_crown_propagation() {
    let shape = vec![2, 3];
    let graph = graph_cast_then_relu(DType::F32, shape.clone());
    let result = trace_to_graph_model(&graph).expect("Cast+Relu translate");
    let input = uniform_bounds(&shape, 2.0);

    let ibp_output = result
        .graph
        .propagate_ibp(&input)
        .expect("IBP through Cast+Relu");
    assert_bounds_valid(&ibp_output);

    let (method, crown_output, _fallback) =
        propagate_with_crown_fallback(&result.graph, &input).expect("CROWN Cast+Relu");
    assert_bounds_valid(&crown_output);

    if matches!(method, PropMethod::Crown) {
        let (crown_lo, crown_hi) = crown_output.lower_upper();
        let (ibp_lo, ibp_hi) = ibp_output.lower_upper();
        let eps = 1e-4;
        for (i, (&cl, &il)) in crown_lo.iter().zip(ibp_lo.iter()).enumerate() {
            assert!(cl >= il - eps, "CROWN lower[{i}] {cl} >= IBP lower {il}");
        }
        for (i, (&cu, &iu)) in crown_hi.iter().zip(ibp_hi.iter()).enumerate() {
            assert!(cu <= iu + eps, "CROWN upper[{i}] {cu} <= IBP upper {iu}");
        }
    }
}
