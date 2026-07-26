// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Smoke test: `trace_to_graph_model` (and the segmented entry) route through
//! the NY-owned `ny-trace-bridge` translator — the unconditional translation
//! path since the legacy in-crate translator was deleted.
//!
//! Probes (each meaningful only if the bridge actually ran):
//!
//! * `dtype_cast_count` — wired from the bridge's `Translation` metadata
//!   (not hardcoded 0);
//! * the MoeGating soundness refusal — the bridge classifies MoeGating
//!   `Unsupported` and refuses fail-closed (`VerifyError::Ny`, UnsupportedOp).
//!   The deleted legacy translator lowered it to an unsound identity
//!   passthrough (a softmax-weighted mix of DIFFERENT elements can leave any
//!   single element's interval), so acceptance would mean a non-bridge
//!   translator ran;
//! * the canonical `VerifyError::MultipleVariableInputs` guard — nn's own
//!   guard runs before delegating, so the caller-visible refusal is nn's
//!   error contract, not the bridge's internal equivalent;
//! * segmented translation — per-segment bridge routing with per-segment
//!   `dtype_cast_count` metadata and the per-segment single-input guard
//!   (sound refusal of aliased multi-variable segments; the legacy segmented
//!   path skipped the guard and aliased independent inputs — a flagged
//!   soundness gap, closed at the cutover).
#![cfg(feature = "ny")]

use nn_core::dyn_tensor::trace::{
    record_input, trace_graph, ComputationGraph, TraceNode, TraceOp,
};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Linear, Module};
use nn_core::{DType, Device};
use nn_verify::{trace_to_graph_model, trace_to_graph_segmented, BoundedTensor, VerifyError};
use ndarray::{ArrayD, IxDyn};

fn cpu() -> Device {
    Device::Cpu
}

/// Small Linear → (F16 downcast → F32 upcast) → ReLU → Linear MLP trace.
///
/// The F16 round-trip puts exactly ONE downcast point in the trace, so
/// `dtype_cast_count == 1` — asserting the count proves the bridge's
/// `Translation` metadata is wired through (not hardcoded 0).
fn build_mlp_with_cast() -> ComputationGraph {
    let w1 = DynTensor::new(&[1.0, -1.0, 0.5, 0.5], &[2, 2], &cpu()).unwrap();
    let b1 = DynTensor::new(&[0.1, -0.2], &[2], &cpu()).unwrap();
    let layer1 = Linear::new(w1, Some(b1)).unwrap();

    let w2 = DynTensor::new(&[1.0, 1.0, -1.0, 1.0], &[2, 2], &cpu()).unwrap();
    let b2 = DynTensor::new(&[0.0, 0.0], &[2], &cpu()).unwrap();
    let layer2 = Linear::new(w2, Some(b2)).unwrap();

    let x = DynTensor::new(&[0.5, -0.5], &[1, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let h = layer1.forward(&x)?;
        let h = h.to_dtype(DType::F16)?; // downcast: the counted cast point
        let h = h.to_dtype(DType::F32)?; // upcast: identity, not counted
        let h = h.relu()?;
        let y = layer2.forward(&h)?;
        Ok(y)
    })
    .unwrap();

    graph
}

/// MoeGating routing probe: a PERMANENT deliberate soundness refusal in the
/// bridge (`Unsupported` in the coverage taxonomy), so it identifies the
/// bridge translator no matter how many op families exist.
fn build_moe_gating_graph() -> ComputationGraph {
    ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![1, 4, 32],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "moe".into(),
            TraceOp::MoeGating {
                num_experts: 4,
                top_k: 1,
            },
            vec![0],
            vec![1, 4, 32],
            DType::F32,
        ),
    ])
}

/// Two genuinely independent variable inputs — the #2425 guard case.
fn build_two_variable_input_graph() -> ComputationGraph {
    let a = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let b = DynTensor::new(&[4.0, 5.0, 6.0], &[1, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id_a = record_input(&[1, 3], DType::F32).unwrap();
        a.set_trace_id(id_a);

        let mut b = b.clone();
        let id_b = record_input(&[1, 3], DType::F32).unwrap();
        b.set_trace_id(id_b);

        let y = a.add(&b)?;
        Ok(y)
    })
    .unwrap();

    graph
}

fn input_box() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 2]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 2]), 1.0_f32),
    )
    .expect("valid input bounds")
}

#[test]
fn trace_to_graph_model_routes_through_bridge() {
    // (a) Bridge Translation metadata is wired through the public entry.
    let graph = build_mlp_with_cast();
    let result = trace_to_graph_model(&graph).expect("bridge-routed translation should succeed");
    assert_eq!(
        result.dtype_cast_count, 1,
        "bridge Translation metadata must be wired through (one F16 downcast)"
    );
    let out = result
        .graph
        .propagate_ibp(&input_box())
        .expect("IBP propagation should succeed");
    let (lo, hi) = out.lower_upper();
    assert!(!lo.is_empty() && lo.len() == hi.len(), "sane output");
    for (l, h) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && h.is_finite() && l <= h, "valid bounds");
    }

    // (b) Routing probe: the bridge's deliberate MoeGating soundness refusal
    //     shows through the public entry (NyError-typed UnsupportedOp).
    let err = trace_to_graph_model(&build_moe_gating_graph())
        .expect_err("bridge must refuse MoeGating (unsound legacy identity lowering)");
    match &err {
        VerifyError::Ny(inner) => {
            let msg = inner.to_string();
            assert!(
                msg.contains("MoeGating"),
                "refusal must name MoeGating, got: {msg}"
            );
        }
        other => panic!("MoeGating must refuse via VerifyError::Ny, got: {other:?}"),
    }

    // (c) Canonical multi-variable guard: nn's own error contract, on the
    //     bridge path.
    let err = trace_to_graph_model(&build_two_variable_input_graph())
        .expect_err("two variable inputs must be refused");
    assert!(
        matches!(err, VerifyError::MultipleVariableInputs { count: 2 }),
        "the guard must be the canonical MultipleVariableInputs, got: {err:?}"
    );
}

/// A two-segment graph split at a `SegmentBoundary`: Input → ReLU | boundary |
/// Input → ReLU. Each segment is single-input and translates independently.
fn build_segmented_graph() -> ComputationGraph {
    ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "in0".into(),
            TraceOp::Input,
            vec![],
            vec![2, 3],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "relu0".into(),
            TraceOp::Relu,
            vec![0],
            vec![2, 3],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "boundary".into(),
            TraceOp::SegmentBoundary {
                reason: "length_regulate".into(),
                input_bounds: Some((-1.0, 1.0)),
            },
            vec![1],
            vec![2, 3],
            DType::F32,
        ),
        TraceNode::new(
            3,
            "in1".into(),
            TraceOp::Input,
            vec![],
            vec![2, 3],
            DType::F32,
        ),
        TraceNode::new(
            4,
            "relu1".into(),
            TraceOp::Relu,
            vec![3],
            vec![2, 3],
            DType::F32,
        ),
    ])
}

#[test]
fn trace_to_graph_segmented_routes_through_bridge() {
    let result =
        trace_to_graph_segmented(&build_segmented_graph()).expect("segmented translation");
    assert_eq!(result.segments.len(), 2, "one segment per boundary side");

    // Boundary metadata carries over (attached to the segment PRECEDING the
    // boundary, matching split_at_segment_boundaries).
    assert_eq!(result.segments[0].segment_index, 0);
    assert_eq!(
        result.segments[0].boundary_reason.as_deref(),
        Some("length_regulate")
    );
    assert_eq!(result.segments[0].boundary_bounds, Some((-1.0, 1.0)));
    assert_eq!(result.segments[1].boundary_reason, None);

    // Per-segment bridge metadata: pure-F32 segments report zero casts, and
    // each segment's GraphNetwork is propagation-ready.
    for seg in &result.segments {
        assert_eq!(seg.result.dtype_cast_count, 0, "pure F32 segments");
        assert!(seg.result.graph.num_nodes() > 0, "non-empty segment graph");
    }
}

/// SOUND REFUSAL (cutover behavior change, deliberate): the deleted legacy
/// segmented path skipped the single-input guard, so a segment with two
/// independent variable inputs had BOTH aliased to the same network input —
/// bounds for one variable silently applied to the other (unsound; a false
/// "holds" is possible whenever the two variables' ranges differ). The
/// bridge-routed segmented path runs the guard PER SEGMENT and refuses with
/// nn's canonical error instead.
#[test]
fn trace_to_graph_segmented_refuses_aliased_multi_variable_segment() {
    // Single segment (no boundary) with two variable inputs feeding an Add.
    let graph = build_two_variable_input_graph();
    let err = trace_to_graph_segmented(&graph)
        .expect_err("aliased multi-variable segment must be refused, not aliased");
    assert!(
        matches!(err, VerifyError::MultipleVariableInputs { count: 2 }),
        "sound per-segment refusal must be the canonical MultipleVariableInputs, got: {err:?}"
    );
}
