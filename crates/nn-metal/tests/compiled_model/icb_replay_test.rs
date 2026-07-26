// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for ICB (Indirect Command Buffer) replay.
//!
//! Tests the full ICB path: segment detection → pre-compile → first-pass
//! encoding → subsequent-pass replay. Exercises `build_icb_step_bindings`
//! (AC2), two-pass replay correctness (AC3), and `collect_icb_resources`
//! buffer dedup (AC5 — tested indirectly via execute_icb).
//!
//! Part of #3410.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceOp};
use nn_metal::compiled_model::CompiledModel;

use super::helpers::{binary_node, create_input_buffer, input_node, read_output_n, unary_node};

// ── Graph builders ──────────────────────────────────────────────────

/// Deep diamond DAG: intermediate nodes have fan-out > 1 → prevents fusion.
///
/// ```text
/// input [16]
///   ├── a = relu(input)       (fan-out=2: c, d)
///   └── b = exp(input)        (fan-out=2: c, d)
///         c = a + b           (fan-out=2: e, f)
///         d = a - b           (fan-out=2: e, f)
///         e = c + d → fuses with g (fan-out=1)
///         f = c - d           (fan-out=1, but g claimed by e's chain)
///         g = e + f           output
/// ```
///
/// Produces ≥ 4 consecutive Dispatch steps (6 dispatches total):
/// a, b, c, d, f, fused(e→g). All are elementwise → ICB-compatible.
fn build_diamond_dag_graph() -> ComputationGraph {
    ComputationGraph::from_nodes(vec![
        input_node(0, &[16]),
        unary_node(1, "relu_a", TraceOp::Relu, 0, &[16]),
        unary_node(2, "exp_b", TraceOp::Exp, 0, &[16]),
        binary_node(3, "add_c", TraceOp::Add, 1, 2, &[16]),
        binary_node(4, "sub_d", TraceOp::Sub, 1, 2, &[16]),
        binary_node(5, "add_e", TraceOp::Add, 3, 4, &[16]),
        binary_node(6, "sub_f", TraceOp::Sub, 3, 4, &[16]),
        binary_node(7, "add_g", TraceOp::Add, 5, 6, &[16]),
    ])
}

/// Reference computation for the diamond DAG.
fn diamond_dag_reference(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            let a = v.max(0.0); // relu
            let b = v.exp(); // exp
            let c = a + b;
            let d = a - b;
            let e = c + d; // = 2*a
            let f = c - d; // = 2*b
            e + f // = 2*a + 2*b = 2*(relu(x) + exp(x))
        })
        .collect()
}

// ── ICB segment detection ───────────────────────────────────────────

/// Verify ICB segments are detected for a non-autocast compiled model
/// with 4+ consecutive Dispatch steps.
#[test]
fn test_icb_segments_detected_without_autocast() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();
    let graph = build_diamond_dag_graph();
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    assert!(!compiled.is_autocast());
    assert!(!compiled.is_mixed_precision());

    // Diamond DAG should produce ≥ 4 IR dispatches (6 expected).
    let ir_dispatches = compiled.num_ir_dispatches();
    assert!(
        ir_dispatches >= 4,
        "expected >= 4 IR dispatches from diamond DAG, got {ir_dispatches}"
    );

    // ICB segments should be detected when segment_starts is wired (#3410).
    let segments = compiled.num_icb_segments();
    assert!(
        segments >= 1,
        "expected >= 1 ICB segment for {ir_dispatches} consecutive dispatches, got {segments}"
    );
}

/// Autocast models support ICB via static dtype flow analysis (#3426 D1).
/// Verify segments are detected (not zero) when the graph has enough
/// consecutive Dispatch steps.
#[test]
fn test_icb_segments_with_autocast() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();
    let graph = build_diamond_dag_graph();

    use nn_core::mixed_precision::MixedPrecisionPolicy;
    let policy = MixedPrecisionPolicy::apple_silicon_default();

    let compiled = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile with autocast");

    assert!(compiled.is_autocast());
    // With static dtype flow analysis, autocast models CAN have ICB segments.
    // The diamond DAG has enough consecutive dispatches to form at least 1 segment.
    let segments = compiled.num_icb_segments();
    assert!(
        segments >= 1,
        "autocast with dtype flow analysis should enable ICB, got {segments} segments"
    );
}

// ── ICB two-pass replay correctness (AC3) ───────────────────────────

/// Two forward passes through the same compiled model with SAME inputs.
/// First pass: encodes ICB. Second pass: replays ICB.
/// Both must produce identical output.
#[test]
fn test_icb_replay_two_passes_identical_output() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();
    let graph = build_diamond_dag_graph();
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    let input_data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.5).collect();
    let input_buf = create_input_buffer(&cache, &input_data);
    let expected = diamond_dag_reference(&input_data);

    // First pass (encodes ICB on first use).
    let out1 = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute pass 1");
    let r1 = read_output_n(&out1, 16);

    // Second pass (should replay ICB if segments were detected).
    let out2 = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute pass 2");
    let r2 = read_output_n(&out2, 16);

    // Verify numerical correctness against reference.
    for (i, ((&r, &e), &r2v)) in r1.iter().zip(expected.iter()).zip(r2.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-4,
            "pass 1[{i}]: gpu={r}, expected={e}, diff={}",
            (r - e).abs()
        );
        assert!(
            (r2v - e).abs() < 1e-4,
            "pass 2[{i}]: gpu={r2v}, expected={e}, diff={}",
            (r2v - e).abs()
        );
    }

    // Two passes must produce identical output (bit-exact for elementwise).
    assert_eq!(r1, r2, "pass 1 and pass 2 outputs must be identical");
}

/// Two forward passes with DIFFERENT inputs — verify ICB replay updates bindings.
#[test]
fn test_icb_replay_different_inputs() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();
    let graph = build_diamond_dag_graph();
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    // Pass 1: positive values.
    let input1: Vec<f32> = vec![1.0; 16];
    let buf1 = create_input_buffer(&cache, &input1);
    let out1 = compiled.execute(&cache, &[&buf1]).expect("execute pass 1");
    let r1 = read_output_n(&out1, 16);
    let expected1 = diamond_dag_reference(&input1);
    for (i, (&r, &e)) in r1.iter().zip(expected1.iter()).enumerate() {
        assert!((r - e).abs() < 1e-4, "pass 1[{i}]: gpu={r}, expected={e}");
    }

    // Pass 2: different values.
    let input2: Vec<f32> = vec![2.0; 16];
    let buf2 = create_input_buffer(&cache, &input2);
    let out2 = compiled.execute(&cache, &[&buf2]).expect("execute pass 2");
    let r2 = read_output_n(&out2, 16);
    let expected2 = diamond_dag_reference(&input2);
    for (i, (&r, &e)) in r2.iter().zip(expected2.iter()).enumerate() {
        assert!((r - e).abs() < 1e-4, "pass 2[{i}]: gpu={r}, expected={e}");
    }

    // Outputs must differ — proves ICB binding update worked.
    assert_ne!(r1, r2, "outputs must differ for different inputs");
}

// ── ICB segment_starts wiring verification ──────────────────────────

/// Verify that icb_segment_starts is populated (not empty) when segments exist.
/// This catches the regression where segment_starts was always empty (#3410).
#[test]
fn test_icb_segment_starts_populated() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();
    let graph = build_diamond_dag_graph();
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    if compiled.num_icb_segments() > 0 {
        // The first Dispatch step should be a segment start.
        let steps = compiled.steps();
        let first_dispatch_idx = steps
            .iter()
            .position(|s| matches!(s, nn_dsl::CompiledStep::Dispatch { .. }))
            .expect("should have Dispatch steps");

        assert!(
            compiled.icb_segment_starts_at(first_dispatch_idx),
            "ICB segment should start at first Dispatch step (idx {first_dispatch_idx})"
        );
    }
}

/// Verify non-autocast model produces correct output regardless of ICB activation.
/// This test works whether ICB is active or falls back to normal dispatch.
#[test]
fn test_icb_correctness_multiple_passes() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();
    let graph = build_diamond_dag_graph();
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    // Run 5 passes with different inputs to stress ICB encode/replay cycles.
    for pass in 0..5 {
        let val = (pass as f32 + 1.0) * 0.5;
        let input_data: Vec<f32> = vec![val; 16];
        let buf = create_input_buffer(&cache, &input_data);
        let out = compiled
            .execute(&cache, &[&buf])
            .unwrap_or_else(|e| panic!("execute pass {pass} failed: {e}"));
        let result = read_output_n(&out, 16);
        let expected = diamond_dag_reference(&input_data);

        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-4,
                "pass {pass}[{i}]: gpu={r}, expected={e}, diff={}",
                (r - e).abs()
            );
        }
    }
}
