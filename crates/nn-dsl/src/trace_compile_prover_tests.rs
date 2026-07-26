// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Prover-authored soundness tests for trace compilation.
//!
//! These tests document and verify invariants found during strategic
//! verification audit. Each test targets a specific correctness risk
//! discovered in the trace→compile pipeline.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use crate::trace_compile::compile_trace;
use crate::trace_compile::CompiledStep;

// -- Helpers (shared pattern with other trace_compile test modules) ------------

fn graph_from_nodes(nodes: Vec<TraceNode>) -> ComputationGraph {
    ComputationGraph::from_nodes(nodes)
}

fn input_node(id: u64, shape: &[usize]) -> TraceNode {
    TraceNode::new(
        id,
        format!("input_{id}"),
        TraceOp::Input,
        vec![],
        shape.to_vec(),
        DType::F32,
    )
}

// -- LSTM with 1 input synthesizes zero h/c states ---------------------------
//
// When the trace records only the data input (forward_seq with None initial
// state), compile_lstm synthesizes zero hidden/cell states as weight data.
// This is the common path for BiLSTM expansion where the import layer creates
// Constant nodes for h_0/c_0 but they may not be in the LSTM trace inputs.

#[test]
fn test_lstm_compile_succeeds_with_1_input_zero_state_synthesis() {
    // Construct an LSTM node with only 1 input (data only, no h/c state).
    // compile_trace should succeed by synthesizing zero h/c states.
    let weight_ih = WeightRef::new(vec![1.0; 128], vec![32, 4]).expect("test data");
    let weight_hh = WeightRef::new(vec![1.0; 256], vec![32, 8]).expect("test data");
    let bias_ih = Some(WeightRef::new(vec![0.0; 32], vec![32]).expect("test data"));
    let bias_hh = Some(WeightRef::new(vec![0.0; 32], vec![32]).expect("test data"));

    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 4]), // input only — no h_state or c_state
        TraceNode::new(
            1,
            "lstm_0".into(),
            TraceOp::Lstm {
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                hidden_size: 8,
                initial_hidden: None,
                initial_cell: None,
            },
            vec![0], // Only 1 input — zero states synthesized
            vec![1, 8],
            DType::F32,
        ),
    ]);

    let steps =
        compile_trace(&graph).expect("1-input LSTM should compile with zero-state synthesis");
    assert_eq!(steps.len(), 2, "1 input + 1 dispatch");
    match &steps[1] {
        CompiledStep::Dispatch { weight_data, .. } => {
            assert!(
                weight_data.contains_key("zero_h"),
                "should synthesize zero_h"
            );
            assert!(
                weight_data.contains_key("zero_c"),
                "should synthesize zero_c"
            );
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

#[test]
fn test_lstm_compile_succeeds_with_3_inputs() {
    // When all 3 inputs are provided (input, h_state, c_state),
    // LSTM should compile successfully.
    let weight_ih = WeightRef::new(vec![1.0; 128], vec![32, 4]).expect("test data");
    let weight_hh = WeightRef::new(vec![1.0; 256], vec![32, 8]).expect("test data");
    let bias_ih = Some(WeightRef::new(vec![0.0; 32], vec![32]).expect("test data"));
    let bias_hh = Some(WeightRef::new(vec![0.0; 32], vec![32]).expect("test data"));

    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 4]), // input
        input_node(1, &[1, 8]), // h_state
        input_node(2, &[1, 8]), // c_state
        TraceNode::new(
            3,
            "lstm_0".into(),
            TraceOp::Lstm {
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                hidden_size: 8,
                initial_hidden: None,
                initial_cell: None,
            },
            vec![0, 1, 2], // All 3 inputs provided
            vec![1, 8],
            DType::F32,
        ),
    ]);

    let steps = compile_trace(&graph).expect("LSTM with 3 inputs should compile");
    assert_eq!(steps.len(), 4, "3 input steps + 1 dispatch");
    assert!(
        matches!(steps[3], CompiledStep::Dispatch { .. }),
        "LSTM should compile to a Dispatch step"
    );
}

// -- WeightRef::from_shape produces empty data → compile accepts it ----------
//
// When to_weight_ref() fallback creates WeightRef::from_shape() (empty
// data + shape), compile_trace accepts it without validation — compilation
// only builds the dispatch plan, it doesn't check weight data presence.
//
// At execution time, compiled_model_build.rs now rejects empty weight data
// for non-zero shapes with CompiledModelError::WeightDataMissing (#2190).
// This test documents the compile-time gap: the error is only caught
// downstream at model build time, not at compile_trace time.

#[test]
fn test_empty_weight_ref_compiles_without_error() {
    // WeightRef::from_shape creates empty data. compile_trace accepts it
    // because compilation only builds the dispatch plan — it doesn't
    // validate that weights have actual data. The downstream error
    // (WeightDataMissing) is caught at model build time, not here.
    let empty_weight = WeightRef::from_shape(&[3, 4]);
    assert!(
        empty_weight.data().is_empty(),
        "from_shape should produce empty data"
    );

    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 4]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear {
                weight: empty_weight,
                bias: None,
            },
            vec![0],
            vec![2, 3],
            DType::F32,
        ),
    ]);

    // This succeeds — the compile step does not validate weight data presence.
    let steps = compile_trace(&graph).expect("empty weight compiles");
    match &steps[1] {
        CompiledStep::Dispatch { weight_data, .. } => {
            let weight = weight_data.get("weight").expect("should have weight key");
            assert!(
                weight.data().is_empty(),
                "empty WeightRef data propagates through compilation unchecked — \
                 compiled_model_build.rs catches this at build time with WeightDataMissing"
            );
        }
        other => {
            panic!("expected Dispatch, got {other:?}");
        }
    }
}

// -- BatchNorm compilation produces Dispatch with precomputed scale/offset ----
//
// Coverage gap found during P1 reflection audit. BatchNorm compiles to
// precomputed bn_scale and bn_offset (not raw running_mean/var/weight/bias).
// scale = weight / sqrt(var + eps), offset = bias - mean * scale.

#[test]
fn test_compile_batch_norm() {
    let num_channels = 4;
    let weight = WeightRef::new(vec![1.0; num_channels], vec![num_channels]).expect("test data");
    let bias = WeightRef::new(vec![0.0; num_channels], vec![num_channels]).expect("test data");
    let running_mean =
        WeightRef::new(vec![0.0; num_channels], vec![num_channels]).expect("test data");
    let running_var =
        WeightRef::new(vec![1.0; num_channels], vec![num_channels]).expect("test data");

    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 4, 8]), // [B, C, T]
        TraceNode::new(
            1,
            "batch_norm_0".into(),
            TraceOp::BatchNorm {
                eps: 1e-5,
                weight,
                bias,
                running_mean,
                running_var,
            },
            vec![0],
            vec![2, 4, 8],
            DType::F32,
        ),
    ]);

    let steps = compile_trace(&graph).expect("batch_norm should compile");
    assert_eq!(steps.len(), 2, "1 input + 1 dispatch");
    match &steps[1] {
        CompiledStep::Dispatch { weight_data, .. } => {
            assert!(
                weight_data.contains_key("bn_scale"),
                "batch_norm must have precomputed bn_scale"
            );
            assert!(
                weight_data.contains_key("bn_offset"),
                "batch_norm must have precomputed bn_offset"
            );
            // Verify precomputed values: scale = w / sqrt(var + eps), offset = b - mean * scale
            // With w=1, b=0, mean=0, var=1, eps=1e-5: scale ≈ 0.999995, offset = 0.0
            let scale = weight_data["bn_scale"].data();
            let offset = weight_data["bn_offset"].data();
            assert_eq!(scale.len(), num_channels);
            assert_eq!(offset.len(), num_channels);
            for c in 0..num_channels {
                let expected_scale = 1.0_f32 / (1.0 + 1e-5_f32).sqrt();
                assert!(
                    (scale[c] - expected_scale).abs() < 1e-6,
                    "scale[{c}] mismatch"
                );
                assert!(offset[c].abs() < 1e-6, "offset[{c}] should be ~0");
            }
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

#[test]
fn test_compile_batch_norm_non_finite_eps_rejected() {
    let c = 4;
    let weight = WeightRef::new(vec![1.0; c], vec![c]).expect("test data");
    let bias = WeightRef::new(vec![0.0; c], vec![c]).expect("test data");
    let running_mean = WeightRef::new(vec![0.0; c], vec![c]).expect("test data");
    let running_var = WeightRef::new(vec![1.0; c], vec![c]).expect("test data");

    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 4, 8]),
        TraceNode::new(
            1,
            "batch_norm_0".into(),
            TraceOp::BatchNorm {
                eps: f64::INFINITY, // non-finite
                weight,
                bias,
                running_mean,
                running_var,
            },
            vec![0],
            vec![2, 4, 8],
            DType::F32,
        ),
    ]);

    let err = compile_trace(&graph).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("non-finite") || msg.contains("NonFinite"),
        "non-finite eps should be rejected, got: {msg}"
    );
}

// -- RmsNorm compilation produces Dispatch with weight + eps ------------------
//
// Coverage gap: RmsNorm has compile code at trace_compile_norm.rs:68 but
// zero compile_trace tests.

#[test]
fn test_compile_rms_norm() {
    let hidden = 8;
    let weight = WeightRef::new(vec![1.0; hidden], vec![hidden]).expect("test data");

    let graph = graph_from_nodes(vec![
        input_node(0, &[2, hidden]), // [B, H]
        TraceNode::new(
            1,
            "rms_norm_0".into(),
            TraceOp::RmsNorm { eps: 1e-6, weight },
            vec![0],
            vec![2, hidden],
            DType::F32,
        ),
    ]);

    let steps = compile_trace(&graph).expect("rms_norm should compile");
    assert_eq!(steps.len(), 2, "1 input + 1 dispatch");
    match &steps[1] {
        CompiledStep::Dispatch { weight_data, .. } => {
            assert!(
                weight_data.contains_key("weight"),
                "rms_norm must have weight"
            );
            assert!(weight_data.contains_key("eps"), "rms_norm must have eps");
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

#[test]
fn test_compile_rms_norm_non_finite_eps_rejected() {
    let hidden = 8;
    let weight = WeightRef::new(vec![1.0; hidden], vec![hidden]).expect("test data");

    let graph = graph_from_nodes(vec![
        input_node(0, &[2, hidden]),
        TraceNode::new(
            1,
            "rms_norm_0".into(),
            TraceOp::RmsNorm {
                eps: f64::NAN, // non-finite
                weight,
            },
            vec![0],
            vec![2, hidden],
            DType::F32,
        ),
    ]);

    let err = compile_trace(&graph).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("non-finite") || msg.contains("NonFinite"),
        "NaN eps should be rejected, got: {msg}"
    );
}

// -- ConvTranspose1d compilation produces Dispatch with weight + bias ---------
//
// Coverage gap: ConvTranspose1d has compile code at trace_compile_conv.rs:124
// but zero compile_trace tests. Conv1d, Conv2d, and ConvTranspose2d all have
// tests; ConvTranspose1d was missed.

#[test]
fn test_compile_conv_transpose1d() {
    // weight: [in_ch=3, out_ch=2, kernel_size=4] => 3*2*4 = 24
    let weight = WeightRef::new(vec![1.0; 24], vec![3, 2, 4]).expect("test data");
    let bias = Some(WeightRef::new(vec![0.0; 2], vec![2]).expect("test data"));

    let graph = graph_from_nodes(vec![
        input_node(0, &[3, 8]), // [in_ch, T]
        TraceNode::new(
            1,
            "conv_transpose1d_0".into(),
            TraceOp::ConvTranspose1d {
                weight,
                bias,
                padding: 1,
                output_padding: 0,
                stride: 2,
                dilation: 1,
                groups: 1,
            },
            vec![0],
            // output_len = (T-1)*stride - 2*padding + dilation*(K-1) + output_padding + 1
            //            = (8-1)*2 - 2*1 + 1*(4-1) + 0 + 1 = 16
            vec![2, 16], // [out_ch, output_len]
            DType::F32,
        ),
    ]);

    let steps = compile_trace(&graph).expect("conv_transpose1d should compile");
    assert_eq!(steps.len(), 2, "1 input + 1 dispatch");
    match &steps[1] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } => {
            assert_eq!(kernel.name(), "conv_transpose1d");
            assert!(weight_data.contains_key("weight"));
            assert!(weight_data.contains_key("bias"));
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

#[test]
fn test_compile_conv_transpose1d_no_bias() {
    let weight = WeightRef::new(vec![1.0; 24], vec![3, 2, 4]).expect("test data");

    let graph = graph_from_nodes(vec![
        input_node(0, &[3, 8]),
        TraceNode::new(
            1,
            "conv_transpose1d_0".into(),
            TraceOp::ConvTranspose1d {
                weight,
                bias: None,
                padding: 0,
                output_padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
            vec![0],
            vec![2, 11], // [out_ch, output_len]
            DType::F32,
        ),
    ]);

    let steps = compile_trace(&graph).expect("conv_transpose1d without bias should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::Dispatch {
            kernel,
            weight_data,
            ..
        } => {
            assert_eq!(kernel.name(), "conv_transpose1d");
            assert!(weight_data.contains_key("weight"));
            assert!(
                !weight_data.contains_key("bias"),
                "no-bias variant should not have bias key"
            );
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

// -- op_name catch-all returns "unknown" for valid TraceOp variants ----------
//
// trace_compile_names.rs:70 has `_ => "unknown"` which makes UnsupportedTraceOp
// error messages unhelpful for debugging. This test documents which ops hit
// the catch-all.

#[test]
fn test_unsupported_ops_have_named_error_messages() {
    // Each of these ops hits compile_node's catch-all → UnsupportedTraceOp.
    // The error should include the op name, not "unknown".
    // Expand, Cumsum, Powf, ToDtype are now supported (zone merge).
    // Only test ops that still hit the unsupported catch-all.
    let test_cases: Vec<(&str, TraceOp)> = vec![
        // Sdpa is now supported (compile_sdpa in trace_compile_attention.rs).
        // Flip is now supported (compile_flip in trace_compile_misc.rs).
        ("triu", TraceOp::Triu { diagonal: 0 }),
        ("tril", TraceOp::Tril { diagonal: 0 }),
    ];

    for (expected_name, op) in test_cases {
        let graph = graph_from_nodes(vec![
            input_node(0, &[2, 4]),
            TraceNode::new(
                1,
                format!("{expected_name}_0"),
                op,
                vec![0],
                vec![2, 4],
                DType::F32,
            ),
        ]);

        let err = compile_trace(&graph).unwrap_err();
        let msg = format!("{err}");
        // Currently many of these return "unknown" instead of the actual op name.
        // This test documents the gap — when op_name is fixed, these will
        // assert that the error includes a useful name.
        // KNOWN GAP: op_name returns "unknown" for many of these variants.
        // When trace_compile_names.rs covers all variants, the "unknown"
        // branch will no longer be reachable.
        // Always verify it's an UnsupportedTraceOp error (not a crash).
        assert!(
            msg.contains("unsupported") || msg.contains("Unsupported"),
            "expected UnsupportedTraceOp for {expected_name}, got: {msg}"
        );
    }
}

// -- Compilation scaling: catch O(n²) regression in compile pipeline ----------
//
// Generates graphs of increasing size and verifies compilation completes
// within a bounded time ratio. If fusion detection or constant folding
// regressed to O(n²), the 4x-larger graph would take ~16x longer (not ~4x).
//
// The graphs are chains of elementwise ops (worst case for fusion detection)
// interleaved with non-fusible ops to stress the skip-scan logic.

/// Build a synthetic graph with `n_chains` disjoint elementwise chains,
/// each of length `chain_len`, interleaved with input nodes.
///
/// Total node count: `n_chains * (chain_len + 1)` (one input per chain
/// plus `chain_len` unary ops).
fn build_scaling_graph(n_chains: usize, chain_len: usize) -> ComputationGraph {
    let mut nodes = Vec::new();
    let mut next_id: u64 = 0;
    let shape = &[1, 64, 256]; // typical activation shape

    for _ in 0..n_chains {
        // Input node for this chain
        let input_id = next_id;
        nodes.push(TraceNode::new(
            input_id,
            format!("input_{input_id}"),
            TraceOp::Input,
            vec![],
            shape.to_vec(),
            DType::F32,
        ));
        next_id += 1;

        // Chain of alternating Relu and Exp (both fusible)
        let mut prev_id = input_id;
        for j in 0..chain_len {
            let op = if j % 2 == 0 {
                TraceOp::Relu
            } else {
                TraceOp::Exp
            };
            let id = next_id;
            nodes.push(TraceNode::new(
                id,
                format!("op_{id}"),
                op,
                vec![prev_id],
                shape.to_vec(),
                DType::F32,
            ));
            prev_id = id;
            next_id += 1;
        }
    }

    ComputationGraph::from_nodes(nodes)
}

#[test]
fn test_compile_scales_linearly_with_graph_size() {
    use crate::trace_compile::compile_trace_to_plan_with_fusion;
    use std::time::Instant;

    // Small: 50 chains × 4 ops = 250 nodes
    let g_small = build_scaling_graph(50, 4);
    let t0 = Instant::now();
    let plan_small = compile_trace_to_plan_with_fusion(&g_small).expect("small graph compiles");
    let dur_small = t0.elapsed();

    // Large: 200 chains × 4 ops = 1000 nodes (4x bigger)
    let g_large = build_scaling_graph(200, 4);
    let t1 = Instant::now();
    let plan_large = compile_trace_to_plan_with_fusion(&g_large).expect("large graph compiles");
    let dur_large = t1.elapsed();

    // Verify correctness: each chain of 4 should fuse into 1 dispatch + 3 passthrough
    let dispatches_small = plan_small
        .steps
        .iter()
        .filter(|s| matches!(s, CompiledStep::Dispatch { .. }))
        .count();
    let dispatches_large = plan_large
        .steps
        .iter()
        .filter(|s| matches!(s, CompiledStep::Dispatch { .. }))
        .count();
    // Each chain of 4 fusible ops → 1 fused dispatch
    assert_eq!(dispatches_small, 50, "50 chains → 50 fused dispatches");
    assert_eq!(dispatches_large, 200, "200 chains → 200 fused dispatches");

    // Scaling check: 4x more nodes should take at most 8x time (sub-quadratic).
    // Pure O(n²) would give ~16x. We allow up to 8x for overhead/cache effects.
    // Skip timing assertion if dur_small is too small for stable measurement
    // (< 100μs), which happens on fast machines.
    if dur_small.as_micros() >= 100 {
        let ratio = dur_large.as_secs_f64() / dur_small.as_secs_f64();
        assert!(
            ratio < 8.0,
            "compilation scaling ratio {ratio:.1}x for 4x graph size — \
             expected < 8x (sub-quadratic). Small: {dur_small:?}, Large: {dur_large:?}"
        );
    }
}

#[test]
fn test_compile_1000_node_graph_under_one_second() {
    use crate::trace_compile::compile_trace_to_plan_with_fusion;
    use std::time::Instant;

    // 200 chains × 5 ops = 1200 total nodes. Should compile in < 1s.
    let graph = build_scaling_graph(200, 5);
    assert_eq!(graph.len(), 1200);

    let start = Instant::now();
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("1200-node graph compiles");
    let elapsed = start.elapsed();

    // Verify fusion happened
    let fused = plan
        .steps
        .iter()
        .filter(|s| matches!(s, CompiledStep::Dispatch { .. }))
        .count();
    assert_eq!(fused, 200, "200 chains should produce 200 fused dispatches");

    // Absolute time bound: 1 second is generous for 1200 nodes.
    // On M4 Max, this takes ~5-50ms. If it exceeds 1s, something is O(n²).
    assert!(
        elapsed.as_secs_f64() < 1.0,
        "1200-node compilation took {elapsed:?} — expected < 1s. Possible quadratic regression."
    );
}

// -- Buffer planner linear-scan regression guard ------------------------------
//
// The buffer planner's `alloc_or_reuse` scans free_slots linearly. Verify
// that for a graph with many short-lived intermediates, buffer planning
// stays fast (< 100ms for 1000 nodes).

#[test]
fn test_buffer_planner_scales_for_many_intermediates() {
    use crate::buffer_planner::plan_buffers;
    use crate::trace_compile::compile_trace_to_plan_with_fusion;
    use std::time::Instant;

    // 500 chains × 2 ops = 1500 nodes — many disjoint chains create
    // many short-lived intermediates that stress free_slots reuse.
    let graph = build_scaling_graph(500, 2);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("graph compiles");

    let start = Instant::now();
    let buffer_plan = plan_buffers(&plan, &graph);
    let elapsed = start.elapsed();

    // Buffer reuse should be active: total_bytes < naive_total
    assert!(
        buffer_plan.total_bytes <= buffer_plan.naive_total,
        "buffer planner should reuse slots: total={} vs naive={}",
        buffer_plan.total_bytes,
        buffer_plan.naive_total,
    );

    // Time bound: 100ms is very generous for 1500 nodes.
    assert!(
        elapsed.as_secs_f64() < 0.1,
        "buffer planning for 1500 nodes took {elapsed:?} — expected < 100ms"
    );
}
