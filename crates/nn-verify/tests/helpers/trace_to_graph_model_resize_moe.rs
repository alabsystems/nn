// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for ResizeBilinear and MoeGating trace-to-graph
//! translation via the `trace_to_graph_model` (LayerSpec → build_graph_network)
//! path.
//!
//! ResizeBilinear: decomposed into Tile + Slice + Reshape (conservative
//! bounds-preserving over-approximation of bilinear interpolation).
//!
//! MoeGating: REFUSED (the deleted legacy identity passthrough was unsound —
//! cross-element softmax-weighted mixing can leave any single element's
//! interval).
//!
//! Part of #3545; MoeGating reconciliation part of INC-FINAL.

use super::common::{assert_bounds_valid, uniform_bounds};
use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;
use nn_verify::trace_to_graph_model;

// -- ResizeBilinear: upscale IBP ------------------------------------------

/// Upscale: [1, 1, 2, 2] → [1, 1, 4, 4]. Bounds preserved.
#[test]
fn test_resize_bilinear_upscale_ibp() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![1, 1, 2, 2],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "resize".into(),
            TraceOp::ResizeBilinear {
                target_h: 4,
                target_w: 4,
            },
            vec![0],
            vec![1, 1, 4, 4],
            DType::F32,
        ),
    ]);

    let gn = trace_to_graph_model(&graph)
        .expect("ResizeBilinear upscale translation")
        .graph;
    let input_bounds = uniform_bounds(&[1, 1, 2, 2], 5.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    // Bilinear resize is bounds-preserving (convex combination).
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -5.0 - 1e-5, "resize upscale lo >= -5, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 5.0 + 1e-5, "resize upscale hi <= 5, got {v}");
    }
}

// -- ResizeBilinear: downscale IBP ----------------------------------------

/// Downscale: [1, 1, 4, 4] → [1, 1, 2, 2]. Bounds preserved.
#[test]
fn test_resize_bilinear_downscale_ibp() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![1, 1, 4, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "resize".into(),
            TraceOp::ResizeBilinear {
                target_h: 2,
                target_w: 2,
            },
            vec![0],
            vec![1, 1, 2, 2],
            DType::F32,
        ),
    ]);

    let gn = trace_to_graph_model(&graph)
        .expect("ResizeBilinear downscale translation")
        .graph;
    let input_bounds = uniform_bounds(&[1, 1, 4, 4], 10.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -10.0 - 1e-5, "resize downscale lo >= -10, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 10.0 + 1e-5, "resize downscale hi <= 10, got {v}");
    }
}

// -- ResizeBilinear: non-integer scale IBP --------------------------------

/// Non-integer scale: [1, 1, 3, 3] → [1, 1, 5, 5]. Tile factor is ceil(25/9)=3.
#[test]
fn test_resize_bilinear_non_integer_scale_ibp() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![1, 1, 3, 3],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "resize".into(),
            TraceOp::ResizeBilinear {
                target_h: 5,
                target_w: 5,
            },
            vec![0],
            vec![1, 1, 5, 5],
            DType::F32,
        ),
    ]);

    let gn = trace_to_graph_model(&graph)
        .expect("ResizeBilinear non-integer scale translation")
        .graph;
    let input_bounds = uniform_bounds(&[1, 1, 3, 3], 8.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -8.0 - 1e-5, "resize non-int lo >= -8, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 8.0 + 1e-5, "resize non-int hi <= 8, got {v}");
    }
}

// -- ResizeBilinear: identity (same dims) IBP -----------------------------

/// Identity resize: [1, 1, 4, 4] → [1, 1, 4, 4]. No tile/slice needed.
#[test]
fn test_resize_bilinear_identity_ibp() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![1, 1, 4, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "resize".into(),
            TraceOp::ResizeBilinear {
                target_h: 4,
                target_w: 4,
            },
            vec![0],
            vec![1, 1, 4, 4],
            DType::F32,
        ),
    ]);

    let gn = trace_to_graph_model(&graph)
        .expect("ResizeBilinear identity translation")
        .graph;
    let input_bounds = uniform_bounds(&[1, 1, 4, 4], 3.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -3.0 - 1e-5, "resize identity lo >= -3, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 3.0 + 1e-5, "resize identity hi <= 3, got {v}");
    }
}

// -- MoeGating ------------------------------------------------------------
//
// INC-FINAL soundness fix: MoeGating is REFUSED. The deleted legacy
// translator modeled it as an identity passthrough, which is NOT a sound
// over-approximation: a softmax-weighted mix of DIFFERENT elements
// (cross-element mixing over data-dependent top-k expert routing) can leave
// any single element's input interval.

/// Shared refusal assertion for the MoeGating fixtures.
fn assert_moe_gating_refused(graph: &ComputationGraph) {
    let err = trace_to_graph_model(graph)
        .expect_err("MoeGating must be refused (unsound identity lowering)");
    let msg = err.to_string();
    assert!(
        msg.contains("MoeGating") && msg.contains("not supported"),
        "refusal should name MoeGating, got: {msg}"
    );
}

/// MoeGating [1, 4, 64]: refused (sound; the deleted legacy passthrough was not).
#[test]
fn test_moe_gating_ibp() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![1, 4, 64],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "moe".into(),
            TraceOp::MoeGating {
                num_experts: 8,
                top_k: 2,
            },
            vec![0],
            vec![1, 4, 64],
            DType::F32,
        ),
    ]);

    assert_moe_gating_refused(&graph);
}

/// MoeGating with different expert config: [2, 8, 128], 16 experts, top-4.
/// Refused (sound).
#[test]
fn test_moe_gating_large_ibp() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![2, 8, 128],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "moe".into(),
            TraceOp::MoeGating {
                num_experts: 16,
                top_k: 4,
            },
            vec![0],
            vec![2, 8, 128],
            DType::F32,
        ),
    ]);

    assert_moe_gating_refused(&graph);
}

/// MoeGating: refused at graph-build time too (sound).
#[test]
fn test_moe_gating_graph_builds() {
    let graph = ComputationGraph::from_nodes(vec![
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
    ]);

    assert_moe_gating_refused(&graph);
}
