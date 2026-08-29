// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for ResizeBilinear and MoeGating trace-to-graph
//! translation via the `trace_to_graph_model` (LayerSpec → build_graph_network)
//! path.
//!
//! ResizeBilinear: LOWERED, and pinned against nn's own kernel. The op's
//! geometry is entirely trace-time-known (`target_h`/`target_w` are
//! serialized and the input extents come from the recorded shape) and nn's
//! `DynTensor::resize_bilinear` has exactly ONE coordinate convention
//! (half-pixel centers `src = (dst + 0.5) * in/out - 0.5`, clamped; there is
//! no `align_corners` knob on that path), so the op is a CONSTANT
//! row-stochastic linear map. The bridge lowers it to a separated pair of
//! constant `Linear` resampling matrices. The former Tile + Slice + Reshape
//! decomposition was refused (ny 25dee0c5e) because it preserved per-element
//! input intervals rather than the convex hull an interpolated value needs;
//! the `Linear` lowering is the exact affine map instead, so the refusal is
//! lifted — and these tests are the evidence, running nn's REAL kernel
//! against the translated graph's bounds rather than a transcription of it.
//!
//! MoeGating: REFUSED (the deleted legacy identity passthrough was unsound —
//! cross-element softmax-weighted mixing can leave any single element's
//! interval).
//!
//! Part of #3545; MoeGating reconciliation part of INC-FINAL; ResizeBilinear
//! refusal flip part of the 25dee0c5e hardening, un-flipped by the bilinear
//! lowering pass.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::{DType, Device, DynTensor};
use nn_verify::{trace_to_graph_model, BoundedTensor};

// -- ResizeBilinear: lowered, and pinned against nn's own kernel -----------
//
// The four scale relations (upscale, downscale, non-integer scale, identity)
// are exercised end to end: nn's real `DynTensor::resize_bilinear` output
// must land inside the translated graph's IBP interval at a point input, at
// every output cell.

/// The four scale relations, as `(input shape, target_h, target_w, output shape)`.
const RESIZE_CASES: [(&[usize], usize, usize, &[usize]); 4] = [
    (&[1, 1, 2, 2], 4, 4, &[1, 1, 4, 4]), // upscale
    (&[1, 1, 4, 4], 2, 2, &[1, 1, 2, 2]), // downscale
    (&[1, 1, 3, 3], 5, 5, &[1, 1, 5, 5]), // non-integer scale
    (&[1, 1, 4, 4], 4, 4, &[1, 1, 4, 4]), // identity
];

/// Deterministic pseudo-random f32 fill in `[-4, 4)`.
fn resize_fixture(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (((state >> 40) & 0xFFFF) as f32) / 8192.0 - 4.0
        })
        .collect()
}

/// ResizeBilinear's lowering must ENCLOSE nn's kernel for every scale
/// relation.
///
/// This is the assertion a coordinate-convention error fails: substituting
/// `align_corners` (`src = dst*(in-1)/(out-1)`) or the naive asymmetric map
/// (`src = dst*in/out`) shifts the sampled cells by up to half a pixel, which
/// on this data moves values far outside a rounding-width interval. Because
/// the reference here is nn's REAL kernel — the same code the traced model
/// executes — agreement is cross-repo evidence, not a restatement of the
/// bridge's own arithmetic.
#[test]
fn test_resize_bilinear_lowering_encloses_the_nn_kernel_at_all_scales() {
    for (case_index, (input_shape, target_h, target_w, output_shape)) in
        RESIZE_CASES.into_iter().enumerate()
    {
        let numel: usize = input_shape.iter().product();
        let values = resize_fixture(numel, 0x2026_0826 ^ (case_index as u64));

        // nn's own kernel is the oracle.
        let input = DynTensor::from_vec(values.clone(), input_shape, &Device::Cpu)
            .expect("input DynTensor");
        let expected = input
            .resize_bilinear(target_h, target_w)
            .expect("nn resize_bilinear")
            .to_f32_array()
            .expect("kernel output as f32");
        assert_eq!(expected.shape(), output_shape);

        let graph = ComputationGraph::from_nodes(vec![
            TraceNode::new(
                0,
                "input".into(),
                TraceOp::Input,
                vec![],
                input_shape.to_vec(),
                DType::F32,
            ),
            TraceNode::new(
                1,
                "resize".into(),
                TraceOp::ResizeBilinear { target_h, target_w },
                vec![0],
                output_shape.to_vec(),
                DType::F32,
            ),
        ]);
        let translated = trace_to_graph_model(&graph).unwrap_or_else(|e| {
            panic!(
                "{input_shape:?} -> {target_h}x{target_w}: ResizeBilinear must translate \
                 (its geometry is entirely trace-time-known), got {e:?}"
            )
        });

        let point = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(input_shape), values)
            .expect("point tensor");
        let bounds = BoundedTensor::new(point.clone(), point).expect("degenerate box");
        let out = translated
            .graph
            .propagate_ibp(&bounds)
            .expect("translated ResizeBilinear IBP");
        assert_eq!(out.shape(), output_shape);

        for (index, (want, (lo, hi))) in expected
            .iter()
            .zip(out.lower().iter().zip(out.upper().iter()))
            .enumerate()
        {
            assert!(
                lo <= want && want <= hi,
                "{input_shape:?} -> {target_h}x{target_w} cell {index}: nn kernel {want} \
                 outside the translated bound [{lo}, {hi}]"
            );
        }
    }
}

/// The enclosure must be TIGHT: a lowering that sampled the wrong cells but
/// returned a wide interval would still "enclose". Pin the point-input
/// interval width to a rounding scale, well below the spread of the data.
#[test]
fn test_resize_bilinear_lowering_is_rounding_tight() {
    for (case_index, (input_shape, target_h, target_w, output_shape)) in
        RESIZE_CASES.into_iter().enumerate()
    {
        let numel: usize = input_shape.iter().product();
        let values = resize_fixture(numel, 0xfeed_face ^ (case_index as u64));

        let graph = ComputationGraph::from_nodes(vec![
            TraceNode::new(
                0,
                "input".into(),
                TraceOp::Input,
                vec![],
                input_shape.to_vec(),
                DType::F32,
            ),
            TraceNode::new(
                1,
                "resize".into(),
                TraceOp::ResizeBilinear { target_h, target_w },
                vec![0],
                output_shape.to_vec(),
                DType::F32,
            ),
        ]);
        let translated = trace_to_graph_model(&graph).expect("translates");
        let point = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(input_shape), values)
            .expect("point tensor");
        let bounds = BoundedTensor::new(point.clone(), point).expect("degenerate box");
        let out = translated.graph.propagate_ibp(&bounds).expect("IBP");

        for (lo, hi) in out.lower().iter().zip(out.upper().iter()) {
            assert!(
                hi - lo < 1e-3,
                "{input_shape:?} -> {target_h}x{target_w}: point-input interval \
                 [{lo}, {hi}] is not rounding-tight"
            );
        }
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
