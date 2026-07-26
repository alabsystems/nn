// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end integration test for batched QKV projection (pass 10).
//!
//! Builds a trace graph with 3 parallel Linear projections sharing one input
//! (Q/K/V pattern), compiles via `CompiledModel::builder().build()`, executes
//! on GPU, and verifies outputs match CPU reference within tolerance.
//!
//! Tests both F32 and autocast F16 modes. Part of #3272.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::mixed_precision::MixedPrecisionPolicy;
use nn_core::DType;
use nn_metal::compiled_model::CompiledModel;

use crate::helpers::{assert_close, create_input_buffer, input_node, read_output_n};
use crate::test_utils;

/// Build a graph: Input [1, in_f] → 3 parallel Linear projections → Add chain → output.
///
/// Architecture: shared hidden → Q Linear(in_f, q_out) + K Linear(in_f, k_out) + V Linear(in_f, v_out)
/// For simplicity, all projections have the same out_features so we can Add them.
/// Graph: Input(0) → Q(1) → K(2) → V(3) → Add(Q,K)(4) → Add(4,V)(5)
///
/// Returns `(graph, q_weight, k_weight, v_weight, bias_q, bias_k, bias_v)`.
fn build_qkv_graph(
    in_features: usize,
    out_features: usize,
) -> (
    ComputationGraph,
    WeightRef,
    WeightRef,
    WeightRef,
    WeightRef,
    WeightRef,
    WeightRef,
) {
    let q_w = WeightRef::new(
        test_utils::rand_f32_vec(100, out_features * in_features, -0.3, 0.3),
        vec![out_features, in_features],
    )
    .unwrap();
    let k_w = WeightRef::new(
        test_utils::rand_f32_vec(200, out_features * in_features, -0.3, 0.3),
        vec![out_features, in_features],
    )
    .unwrap();
    let v_w = WeightRef::new(
        test_utils::rand_f32_vec(300, out_features * in_features, -0.3, 0.3),
        vec![out_features, in_features],
    )
    .unwrap();
    let q_b = WeightRef::new(
        test_utils::rand_f32_vec(400, out_features, -0.1, 0.1),
        vec![out_features],
    )
    .unwrap();
    let k_b = WeightRef::new(
        test_utils::rand_f32_vec(500, out_features, -0.1, 0.1),
        vec![out_features],
    )
    .unwrap();
    let v_b = WeightRef::new(
        test_utils::rand_f32_vec(600, out_features, -0.1, 0.1),
        vec![out_features],
    )
    .unwrap();

    let out_shape = vec![1, out_features];
    let nodes = vec![
        // 0: input
        input_node(0, &[1, in_features]),
        // 1: Q linear
        TraceNode::new(
            1,
            "q_proj".into(),
            TraceOp::Linear {
                weight: q_w.clone(),
                bias: Some(q_b.clone()),
            },
            vec![0],
            out_shape.clone(),
            DType::F32,
        ),
        // 2: K linear
        TraceNode::new(
            2,
            "k_proj".into(),
            TraceOp::Linear {
                weight: k_w.clone(),
                bias: Some(k_b.clone()),
            },
            vec![0],
            out_shape.clone(),
            DType::F32,
        ),
        // 3: V linear
        TraceNode::new(
            3,
            "v_proj".into(),
            TraceOp::Linear {
                weight: v_w.clone(),
                bias: Some(v_b.clone()),
            },
            vec![0],
            out_shape.clone(),
            DType::F32,
        ),
        // 4: Add(Q, K)
        TraceNode::new(
            4,
            "add_qk".into(),
            TraceOp::Add,
            vec![1, 2],
            out_shape.clone(),
            DType::F32,
        ),
        // 5: Add(QK, V) — final output
        TraceNode::new(
            5,
            "add_qkv".into(),
            TraceOp::Add,
            vec![4, 3],
            out_shape,
            DType::F32,
        ),
    ];

    let graph = ComputationGraph::from_nodes(nodes);
    (graph, q_w, k_w, v_w, q_b, k_b, v_b)
}

/// CPU reference: compute Q + K + V projections and sum them.
fn qkv_cpu_ref(
    input: &[f32],
    in_features: usize,
    out_features: usize,
    q_w: &WeightRef,
    k_w: &WeightRef,
    v_w: &WeightRef,
    q_b: &WeightRef,
    k_b: &WeightRef,
    v_b: &WeightRef,
) -> Vec<f32> {
    let q = test_utils::linear_ref(
        input,
        q_w.data(),
        Some(q_b.data()),
        1,
        in_features,
        out_features,
    );
    let k = test_utils::linear_ref(
        input,
        k_w.data(),
        Some(k_b.data()),
        1,
        in_features,
        out_features,
    );
    let v = test_utils::linear_ref(
        input,
        v_w.data(),
        Some(v_b.data()),
        1,
        in_features,
        out_features,
    );
    q.iter()
        .zip(k.iter())
        .zip(v.iter())
        .map(|((&a, &b), &c)| a + b + c)
        .collect()
}

// -- Tests --------------------------------------------------------------------

/// F32 mode: 3 parallel Linears sharing one input → batched QKV projection →
/// Add chain → GPU output matches CPU reference.
///
/// The peephole pass 12 should detect the 3 parallel Linears and batch them
/// into one BatchedLinearProjection + 2 ProjectionSlice. This test verifies
/// the end-to-end numerical correctness of that optimization.
#[test]
fn test_batched_qkv_e2e_f32() {
    test_utils::gpu_init();
    let cache = test_utils::metal_setup();

    let in_f = 16;
    let out_f = 8;
    let (graph, q_w, k_w, v_w, q_b, k_b, v_b) = build_qkv_graph(in_f, out_f);

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile QKV");
    assert_eq!(compiled.num_inputs(), 1);
    assert_eq!(compiled.output_shape(), &[1, out_f]);

    // Log dispatch count — batching should reduce dispatches.
    // Unbatched: 3 linear + 2 add = 5 dispatches (linear = matmul composite).
    // Batched: 1 batched_linear + 2 projection_slice + 2 add = 5 NativeOp steps,
    // but fewer GPU dispatches since projection_slice is a narrow (no kernel).
    let nd = compiled.num_dispatches();
    eprintln!(
        "[QKV-F32] steps={}, dispatches={nd} (unbatched=5)",
        compiled.num_steps()
    );

    let input = test_utils::rand_f32_vec(42, in_f, -1.0, 1.0);
    let buf = create_input_buffer(&cache, &input);
    let out = compiled.execute(&cache, &[&buf]).expect("execute QKV");

    let expected = qkv_cpu_ref(&input, in_f, out_f, &q_w, &k_w, &v_w, &q_b, &k_b, &v_b);
    let result = read_output_n(&out, expected.len());
    assert_close("qkv_f32", &result, &expected, 1e-4);
}

/// Autocast F16 mode: same graph compiled with autocast, verify output
/// matches F32 baseline within BF16/F16 tolerance.
#[test]
fn test_batched_qkv_e2e_autocast() {
    test_utils::gpu_init();
    let cache = test_utils::metal_setup();

    let in_f = 16;
    let out_f = 8;
    let (graph, q_w, k_w, v_w, q_b, k_b, v_b) = build_qkv_graph(in_f, out_f);

    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let compiled = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile QKV autocast");
    assert_eq!(compiled.num_inputs(), 1);
    assert_eq!(compiled.output_shape(), &[1, out_f]);
    assert!(compiled.is_autocast());

    let nd = compiled.num_dispatches();
    eprintln!(
        "[QKV-AC] steps={}, dispatches={nd}, f16_steps={}",
        compiled.num_steps(),
        compiled.num_autocast_f16_steps()
    );

    let input = test_utils::rand_f32_vec(42, in_f, -1.0, 1.0);
    let buf = create_input_buffer(&cache, &input);
    let out = compiled
        .execute(&cache, &[&buf])
        .expect("execute QKV autocast");

    // CPU reference in F32 — autocast should match within F16 precision.
    let expected = qkv_cpu_ref(&input, in_f, out_f, &q_w, &k_w, &v_w, &q_b, &k_b, &v_b);
    let result = read_output_n(&out, expected.len());
    // F16 tolerance: ~1e-2 for small matmuls with accumulation error.
    assert_close("qkv_autocast", &result, &expected, 5e-2);
}

/// GQA pattern: Q has larger out_features than K/V (768 vs 256).
/// Verifies that the batching pass handles heterogeneous projection sizes.
#[test]
fn test_batched_qkv_gqa_e2e_f32() {
    test_utils::gpu_init();
    let cache = test_utils::metal_setup();

    let in_f = 16;
    // Q: 16→12, K: 16→4, V: 16→4 (GQA-like ratio)
    let q_out = 12;
    let kv_out = 4;

    let q_w = WeightRef::new(
        test_utils::rand_f32_vec(100, q_out * in_f, -0.3, 0.3),
        vec![q_out, in_f],
    )
    .unwrap();
    let k_w = WeightRef::new(
        test_utils::rand_f32_vec(200, kv_out * in_f, -0.3, 0.3),
        vec![kv_out, in_f],
    )
    .unwrap();
    let v_w = WeightRef::new(
        test_utils::rand_f32_vec(300, kv_out * in_f, -0.3, 0.3),
        vec![kv_out, in_f],
    )
    .unwrap();
    let q_b = WeightRef::new(test_utils::rand_f32_vec(400, q_out, -0.1, 0.1), vec![q_out]).unwrap();
    let k_b = WeightRef::new(
        test_utils::rand_f32_vec(500, kv_out, -0.1, 0.1),
        vec![kv_out],
    )
    .unwrap();
    let v_b = WeightRef::new(
        test_utils::rand_f32_vec(600, kv_out, -0.1, 0.1),
        vec![kv_out],
    )
    .unwrap();

    // Graph: Input → Q(16→12), K(16→4), V(16→4) → mark all as outputs.
    let nodes = vec![
        input_node(0, &[1, in_f]),
        TraceNode::new(
            1,
            "q_proj".into(),
            TraceOp::Linear {
                weight: q_w.clone(),
                bias: Some(q_b.clone()),
            },
            vec![0],
            vec![1, q_out],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "k_proj".into(),
            TraceOp::Linear {
                weight: k_w.clone(),
                bias: Some(k_b.clone()),
            },
            vec![0],
            vec![1, kv_out],
            DType::F32,
        ),
        TraceNode::new(
            3,
            "v_proj".into(),
            TraceOp::Linear {
                weight: v_w.clone(),
                bias: Some(v_b.clone()),
            },
            vec![0],
            vec![1, kv_out],
            DType::F32,
        ),
    ];
    let mut graph = ComputationGraph::from_nodes(nodes);
    // Mark Q and K as additional outputs (V is auto-marked as last node).
    let _ = graph.mark_output(1);
    let _ = graph.mark_output(2);

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile GQA");
    assert_eq!(compiled.num_inputs(), 1);
    assert_eq!(compiled.num_outputs(), 3);

    let nd = compiled.num_dispatches();
    eprintln!(
        "[GQA-F32] steps={}, dispatches={nd}, outputs={}",
        compiled.num_steps(),
        compiled.num_outputs()
    );

    let input = test_utils::rand_f32_vec(42, in_f, -1.0, 1.0);
    let buf = create_input_buffer(&cache, &input);
    let outs = compiled
        .execute_outputs(&cache, &[&buf])
        .expect("execute GQA");
    assert_eq!(outs.len(), 3);

    // CPU reference for each projection.
    let q_exp = test_utils::linear_ref(&input, q_w.data(), Some(q_b.data()), 1, in_f, q_out);
    let k_exp = test_utils::linear_ref(&input, k_w.data(), Some(k_b.data()), 1, in_f, kv_out);
    let v_exp = test_utils::linear_ref(&input, v_w.data(), Some(v_b.data()), 1, in_f, kv_out);

    // Output order: V (from_nodes auto-mark=last), Q (mark_output(1)), K (mark_output(2))
    let v_result = read_output_n(&outs[0], v_exp.len());
    let q_result = read_output_n(&outs[1], q_exp.len());
    let k_result = read_output_n(&outs[2], k_exp.len());

    assert_close("gqa_q", &q_result, &q_exp, 1e-4);
    assert_close("gqa_k", &k_result, &k_exp, 1e-4);
    assert_close("gqa_v", &v_result, &v_exp, 1e-4);
}

/// GQA autocast: heterogeneous Q/K/V sizes under F16 execution.
/// Verifies that BatchedLinearProjection + ProjectionSlice handles
/// asymmetric projection sizes (Q: 12, K/V: 4) correctly when
/// autocast converts compute to F16. Part of #3277.
#[test]
fn test_batched_qkv_gqa_e2e_autocast() {
    test_utils::gpu_init();
    let cache = test_utils::metal_setup();

    let in_f = 16;
    let q_out = 12;
    let kv_out = 4;

    let q_w = WeightRef::new(
        test_utils::rand_f32_vec(100, q_out * in_f, -0.3, 0.3),
        vec![q_out, in_f],
    )
    .unwrap();
    let k_w = WeightRef::new(
        test_utils::rand_f32_vec(200, kv_out * in_f, -0.3, 0.3),
        vec![kv_out, in_f],
    )
    .unwrap();
    let v_w = WeightRef::new(
        test_utils::rand_f32_vec(300, kv_out * in_f, -0.3, 0.3),
        vec![kv_out, in_f],
    )
    .unwrap();
    let q_b = WeightRef::new(test_utils::rand_f32_vec(400, q_out, -0.1, 0.1), vec![q_out]).unwrap();
    let k_b = WeightRef::new(
        test_utils::rand_f32_vec(500, kv_out, -0.1, 0.1),
        vec![kv_out],
    )
    .unwrap();
    let v_b = WeightRef::new(
        test_utils::rand_f32_vec(600, kv_out, -0.1, 0.1),
        vec![kv_out],
    )
    .unwrap();

    let nodes = vec![
        input_node(0, &[1, in_f]),
        TraceNode::new(
            1,
            "q_proj".into(),
            TraceOp::Linear {
                weight: q_w.clone(),
                bias: Some(q_b.clone()),
            },
            vec![0],
            vec![1, q_out],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "k_proj".into(),
            TraceOp::Linear {
                weight: k_w.clone(),
                bias: Some(k_b.clone()),
            },
            vec![0],
            vec![1, kv_out],
            DType::F32,
        ),
        TraceNode::new(
            3,
            "v_proj".into(),
            TraceOp::Linear {
                weight: v_w.clone(),
                bias: Some(v_b.clone()),
            },
            vec![0],
            vec![1, kv_out],
            DType::F32,
        ),
    ];
    let mut graph = ComputationGraph::from_nodes(nodes);
    let _ = graph.mark_output(1);
    let _ = graph.mark_output(2);

    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let compiled = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile GQA autocast");
    assert_eq!(compiled.num_inputs(), 1);
    assert_eq!(compiled.num_outputs(), 3);
    assert!(compiled.is_autocast());
    assert!(
        compiled.num_autocast_f16_steps() > 0,
        "GQA projections should have F16 steps"
    );

    // Guard: BatchedLinearProjection classified via is_compute_native_op,
    // not extract_mixed_gemm_infos. DynTensor matmul at F16 routes to
    // simd_gemm_f16 (F32 accumulators) internally. #3277, #3281.
    assert_eq!(
        compiled.num_mixed_gemm_steps(),
        0,
        "BatchedLinearProjection must not be in mixed_gemm_infos (see #3277)"
    );

    // Verify pass 10 triggered: BatchedLinearProjection must exist.
    let (_, native_ops) = compiled.dispatch_breakdown();
    let batched_count = native_ops
        .iter()
        .find(|(name, _)| name == "BatchedLinearProjection")
        .map(|(_, c)| *c)
        .unwrap_or(0);
    assert!(
        batched_count > 0,
        "Pass 10 should trigger for GQA autocast, got native_ops: {native_ops:?}"
    );

    let input = test_utils::rand_f32_vec(42, in_f, -1.0, 1.0);
    let buf = create_input_buffer(&cache, &input);
    let outs = compiled
        .execute_outputs(&cache, &[&buf])
        .expect("execute GQA autocast");
    assert_eq!(outs.len(), 3);

    // CPU reference (F32 precision).
    let q_exp = test_utils::linear_ref(&input, q_w.data(), Some(q_b.data()), 1, in_f, q_out);
    let k_exp = test_utils::linear_ref(&input, k_w.data(), Some(k_b.data()), 1, in_f, kv_out);
    let v_exp = test_utils::linear_ref(&input, v_w.data(), Some(v_b.data()), 1, in_f, kv_out);

    // Output order: V (auto-mark=last), Q (mark_output(1)), K (mark_output(2))
    let v_result = read_output_n(&outs[0], v_exp.len());
    let q_result = read_output_n(&outs[1], q_exp.len());
    let k_result = read_output_n(&outs[2], k_exp.len());

    // F16 tolerance: wider than F32 due to half-precision rounding.
    assert_close("gqa_ac_q", &q_result, &q_exp, 5e-2);
    assert_close("gqa_ac_k", &k_result, &k_exp, 5e-2);
    assert_close("gqa_ac_v", &v_result, &v_exp, 5e-2);
}

/// Production-sized dims: input [512, 256] → 3 projections of 256 each.
///
/// Exercises `should_use_f16_simdgroup` = true:
///   m=512, k=256, n=768 → all %8, m*n=393216 >> 16384, k=256 >> 128,
///   tg_count = ceil(512/32)*ceil(768/32) = 16*24 = 384 >= F16_MIN_THREADGROUPS.
///
/// Before #3281, the mixed GEMM path read F16-cast input as F32 (garbage).
/// After the fix, DynTensor matmul at F16 routes to simd_gemm_f16 (F32
/// accumulators). Part of #3281.
#[test]
fn test_batched_qkv_production_dims_autocast() {
    test_utils::gpu_init();
    let cache = test_utils::metal_setup();

    let seq_len = 512;
    let in_f = 256;
    let out_f = 256;

    let q_w = WeightRef::new(
        test_utils::rand_f32_vec(100, out_f * in_f, -0.3, 0.3),
        vec![out_f, in_f],
    )
    .unwrap();
    let k_w = WeightRef::new(
        test_utils::rand_f32_vec(200, out_f * in_f, -0.3, 0.3),
        vec![out_f, in_f],
    )
    .unwrap();
    let v_w = WeightRef::new(
        test_utils::rand_f32_vec(300, out_f * in_f, -0.3, 0.3),
        vec![out_f, in_f],
    )
    .unwrap();
    let q_b = WeightRef::new(test_utils::rand_f32_vec(400, out_f, -0.1, 0.1), vec![out_f]).unwrap();
    let k_b = WeightRef::new(test_utils::rand_f32_vec(500, out_f, -0.1, 0.1), vec![out_f]).unwrap();
    let v_b = WeightRef::new(test_utils::rand_f32_vec(600, out_f, -0.1, 0.1), vec![out_f]).unwrap();

    let out_shape = vec![seq_len, out_f];
    let nodes = vec![
        input_node(0, &[seq_len, in_f]),
        TraceNode::new(
            1,
            "q_proj".into(),
            TraceOp::Linear {
                weight: q_w.clone(),
                bias: Some(q_b.clone()),
            },
            vec![0],
            out_shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            2,
            "k_proj".into(),
            TraceOp::Linear {
                weight: k_w.clone(),
                bias: Some(k_b.clone()),
            },
            vec![0],
            out_shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            3,
            "v_proj".into(),
            TraceOp::Linear {
                weight: v_w.clone(),
                bias: Some(v_b.clone()),
            },
            vec![0],
            out_shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            4,
            "add_qk".into(),
            TraceOp::Add,
            vec![1, 2],
            out_shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            5,
            "add_qkv".into(),
            TraceOp::Add,
            vec![4, 3],
            out_shape,
            DType::F32,
        ),
    ];
    let graph = ComputationGraph::from_nodes(nodes);

    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let compiled = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile production-sized QKV autocast");
    assert_eq!(compiled.num_inputs(), 1);
    assert!(compiled.is_autocast());

    // Guard: BatchedLinearProjection classified via is_compute_native_op,
    // not extract_mixed_gemm_infos. Uses DynTensor matmul (F32 accumulators
    // via simd_gemm_f16). #3272, #3281.
    assert_eq!(
        compiled.num_mixed_gemm_steps(),
        0,
        "BatchedLinearProjection must not be in mixed_gemm_infos"
    );

    // Verify dims qualify for simdgroup F16 (inline threshold check).
    // total_out = 3 * out_f = 768. m=512, k=256, n=768.
    let total_out = out_f * 3;
    assert!(seq_len % 8 == 0 && in_f % 8 == 0 && total_out % 8 == 0);
    assert!(
        seq_len * total_out >= 16_384,
        "m*n must qualify for simdgroup"
    );
    assert!(in_f >= 128, "k must qualify for simdgroup");
    let tg_count = seq_len.div_ceil(32) * total_out.div_ceil(32);
    assert!(
        tg_count >= 384,
        "tg_count={tg_count} must >= 384 for F16 simdgroup"
    );

    let input = test_utils::rand_f32_vec(42, seq_len * in_f, -0.5, 0.5);
    let buf = create_input_buffer(&cache, &input);
    let out = compiled
        .execute(&cache, &[&buf])
        .expect("execute production QKV autocast");

    // CPU reference in F32.
    let q = test_utils::linear_ref(&input, q_w.data(), Some(q_b.data()), seq_len, in_f, out_f);
    let k = test_utils::linear_ref(&input, k_w.data(), Some(k_b.data()), seq_len, in_f, out_f);
    let v = test_utils::linear_ref(&input, v_w.data(), Some(v_b.data()), seq_len, in_f, out_f);
    let expected: Vec<f32> = q
        .iter()
        .zip(k.iter())
        .zip(v.iter())
        .map(|((&a, &b), &c)| a + b + c)
        .collect();

    let result = read_output_n(&out, expected.len());
    // F16 tolerance with larger dims — accumulation error scales with K.
    assert_close("qkv_prod_ac", &result, &expected, 0.15);
}
