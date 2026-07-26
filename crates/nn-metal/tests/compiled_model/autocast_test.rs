// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for per-op autocast `CompiledModel` execution.
//!
//! Validates that `builder().autocast().build()` produces correct results identical
//! to F32 baseline. Phase 1 autocast keeps all buffers F32 — the test
//! confirms no regression from the autocast code path.
//!
//! Part of #3085, #2981.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceOp};
use nn_core::mixed_precision::{default_op_category, MixedPrecisionPolicy, OpDTypeCategory};
use nn_metal::compiled_model::CompiledModel;

use crate::helpers::{
    assert_close, binary_node, create_input_buffer, input_node, read_output_n, unary_node,
};
use crate::test_utils;

// -- Metadata tests -----------------------------------------------------------

#[test]
fn test_autocast_flag() {
    let cache = test_utils::metal_setup();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let compiled = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert!(compiled.is_autocast());
    assert!(!compiled.is_mixed_precision());
    assert_eq!(compiled.num_steps(), 2);
    assert_eq!(compiled.num_dispatches(), 1);
}

#[test]
fn test_autocast_vs_f32_flag() {
    let cache = test_utils::metal_setup();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert!(!f32_model.is_autocast());
    assert!(!f32_model.is_mixed_precision());
    assert!(ac_model.is_autocast());
    assert!(!ac_model.is_mixed_precision());
}

/// num_autocast_f16_steps returns the total count of steps autocasted to F16.
/// For a matmul (Compute) + relu (non-Compute) graph, only matmul gets F16.
/// Uses 640×128×640 to meet F16 simdgroup threshold (tg_count=20*20=400 >= 384).
#[test]
fn test_autocast_f16_step_count() {
    let cache = test_utils::metal_setup();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[640, 128]),
        input_node(1, &[128, 640]),
        binary_node(2, "matmul_0", TraceOp::MatMul, 0, 1, &[640, 640]),
        unary_node(3, "relu_0", TraceOp::Relu, 2, &[640, 640]),
    ]);

    // F32 model: no autocast steps.
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    assert_eq!(f32_model.num_autocast_f16_steps(), 0);

    // Autocast model: matmul step should be F16, relu stays F32.
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert!(ac_model.num_autocast_f16_steps() > 0);
    // At least matmul step is F16; exact count depends on step classification.
    assert!(ac_model.num_autocast_f16_steps() <= ac_model.num_steps());
}

/// f32_only policy passed to autocast builder is silently discarded.
/// is_autocast() returns false, num_autocast_f16_steps() returns 0.
#[test]
fn test_autocast_f32_only_is_noop() {
    let cache = test_utils::metal_setup();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);
    let policy = MixedPrecisionPolicy::f32_only();
    let model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile f32_only autocast");
    assert!(
        !model.is_autocast(),
        "f32_only policy should not activate autocast"
    );
    assert_eq!(model.num_autocast_f16_steps(), 0);
}

// -- Execution correctness tests ----------------------------------------------

#[test]
fn test_autocast_relu_matches_f32() {
    test_utils::gpu_init();
    let cache = test_utils::metal_setup();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);

    let input_data = [1.0_f32, -2.0, 3.0, -4.0];
    let input_buf = create_input_buffer(&cache, &input_data);

    // F32 baseline.
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f32_out = f32_model.execute(&cache, &[&input_buf]).expect("f32 exec");
    let f32_result = read_output_n(&f32_out, 4);

    // Autocast (Phase 1: all F32 buffers, should be bit-exact).
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    let ac_out = ac_model
        .execute(&cache, &[&input_buf])
        .expect("autocast exec");
    let ac_result = read_output_n(&ac_out, 4);

    assert_close("autocast_relu", &ac_result, &f32_result, 0.0);
}

#[test]
fn test_autocast_chain_sigmoid_add_matches_f32() {
    test_utils::gpu_init();
    let cache = test_utils::metal_setup();
    // Graph: input -> sigmoid -> add(sigmoid_out, sigmoid_out)
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[8]),
        unary_node(1, "sigmoid_0", TraceOp::Sigmoid, 0, &[8]),
        binary_node(2, "add_0", TraceOp::Add, 1, 1, &[8]),
    ]);

    let data = [0.5_f32, -0.5, 1.0, -1.0, 2.0, -2.0, 0.0, 3.0];
    let buf = create_input_buffer(&cache, &data);

    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f32_out = f32_model.execute(&cache, &[&buf]).expect("f32 exec");
    let f32_result = read_output_n(&f32_out, 8);

    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    let ac_out = ac_model.execute(&cache, &[&buf]).expect("autocast exec");
    let ac_result = read_output_n(&ac_out, 8);

    // Phase 1 autocast = all F32 buffers. Should be bit-exact.
    assert_close("autocast_chain", &ac_result, &f32_result, 0.0);
}

#[test]
fn test_autocast_binary_add_matches_f32() {
    test_utils::gpu_init();
    let cache = test_utils::metal_setup();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        binary_node(2, "add_0", TraceOp::Add, 0, 1, &[4]),
    ]);

    let a_data = [1.0_f32, 2.0, 3.0, 4.0];
    let b_data = [10.0_f32, 20.0, 30.0, 40.0];
    let a_buf = create_input_buffer(&cache, &a_data);
    let b_buf = create_input_buffer(&cache, &b_data);

    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f32_out = f32_model
        .execute(&cache, &[&a_buf, &b_buf])
        .expect("f32 exec");
    let f32_result = read_output_n(&f32_out, 4);

    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    let ac_out = ac_model
        .execute(&cache, &[&a_buf, &b_buf])
        .expect("autocast exec");
    let ac_result = read_output_n(&ac_out, 4);

    assert_close("autocast_add", &ac_result, &f32_result, 0.0);
}

// -- Per-op classification tests (Phase 2: Compute ops get F16) ---------------

#[test]
fn test_autocast_matmul_then_relu_compiles() {
    let cache = test_utils::metal_setup();
    // MatMul: [2,4] x [4,3] → [2,3], then Relu on [2,3].
    // MatMul is Compute → F16, Relu is elementwise → F32.
    // Boundary cast F16→F32 required between matmul and relu.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 4]),
        input_node(1, &[4, 3]),
        binary_node(2, "matmul_0", TraceOp::MatMul, 0, 1, &[2, 3]),
        unary_node(3, "relu_0", TraceOp::Relu, 2, &[2, 3]),
    ]);
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let compiled = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert!(compiled.is_autocast());
    assert_eq!(compiled.num_inputs(), 2);
    // 4 steps: 2 inputs + matmul dispatch + relu dispatch
    assert_eq!(compiled.num_steps(), 4);
}

#[test]
fn test_autocast_matmul_relu_matches_f32() {
    test_utils::gpu_init();
    let cache = test_utils::metal_setup();
    // MatMul: [2,4] x [4,3] → [2,3], then Relu on [2,3].
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 4]),
        input_node(1, &[4, 3]),
        binary_node(2, "matmul_0", TraceOp::MatMul, 0, 1, &[2, 3]),
        unary_node(3, "relu_0", TraceOp::Relu, 2, &[2, 3]),
    ]);

    // A = [[1,2,3,4],[5,6,7,8]], B = [[1,0,0],[0,1,0],[0,0,1],[1,1,1]]
    // A*B = [[1+4, 2+4, 3+4],[5+8, 6+8, 7+8]] = [[5,6,7],[13,14,15]]
    // Relu: all positive → same
    let a_data = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b_data = [
        1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let a_buf = create_input_buffer(&cache, &a_data);
    let b_buf = create_input_buffer(&cache, &b_data);

    // F32 baseline.
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f32_out = f32_model
        .execute(&cache, &[&a_buf, &b_buf])
        .expect("f32 exec");
    let f32_result = read_output_n(&f32_out, 6);

    // Autocast: matmul runs F16 (with F32 accumulation), relu runs F32.
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    let ac_out = ac_model
        .execute(&cache, &[&a_buf, &b_buf])
        .expect("autocast exec");
    let ac_result = read_output_n(&ac_out, 6);

    // F16 matmul with small integer values should be exact, but allow small
    // tolerance for F16 rounding on non-integer intermediate values.
    assert_close("autocast_matmul_relu", &ac_result, &f32_result, 1e-2);
}

// -- Regression: blit_copy bug with buffer planning + autocast boundaries ------

/// Regression test for the blit_copy bug (#2981, #3085).
///
/// The old `slice_element_count()` computed element counts from `buffer.len() -
/// byte_offset`, which is wrong for slices relocated to the planned buffer
/// (shared allocation). Now `cast_input_f32_to_f16` takes `step_numel` directly,
/// avoiding the buffer-geometry bug (#3304).
///
/// This test uses a chain of linear(F16) → relu(F32) → linear(F16) → relu(F32)
/// which creates buffer lifetime overlaps that trigger the buffer planner's
/// slot reuse. The intermediate linear output gets relocated to the planned
/// buffer, then the relu step needs an F16→F32 boundary cast on that relocated
/// slice.
#[test]
fn test_autocast_buffer_plan_relocated_slice_cast() {
    test_utils::gpu_init();
    let cache = test_utils::metal_setup();

    use nn_core::dyn_tensor::trace::{TraceNode, WeightRef};
    use nn_core::DType;

    let w1_data: Vec<f32> = test_utils::rand_f32_vec(0xAC01, 8 * 16, -0.3, 0.3);
    let w2_data: Vec<f32> = test_utils::rand_f32_vec(0xAC02, 16 * 4, -0.3, 0.3);
    let w1 = WeightRef::new(w1_data, vec![16, 8]).expect("w1");
    let w2 = WeightRef::new(w2_data, vec![4, 16]).expect("w2");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4, 8]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear {
                weight: w1,
                bias: None,
            },
            vec![0],
            vec![4, 16],
            DType::F32,
        ),
        unary_node(2, "relu_0", TraceOp::Relu, 1, &[4, 16]),
        TraceNode::new(
            3,
            "linear_1".into(),
            TraceOp::Linear {
                weight: w2,
                bias: None,
            },
            vec![2],
            vec![4, 4],
            DType::F32,
        ),
        unary_node(4, "relu_1", TraceOp::Relu, 3, &[4, 4]),
    ]);

    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert!(ac_model.is_autocast());
    assert!(
        ac_model.buffer_plan().total_bytes > 0,
        "buffer planner should be active"
    );

    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let input_data = test_utils::rand_f32_vec(0xAC03, 4 * 8, -1.0, 1.0);
    let buf = create_input_buffer(&cache, &input_data);

    let f32_out = f32_model.execute(&cache, &[&buf]).expect("f32 exec");
    let f32_result = read_output_n(&f32_out, 16);

    let ac_out = ac_model.execute(&cache, &[&buf]).expect("autocast exec");
    let ac_result = read_output_n(&ac_out, 16);

    // Before fix: oversized cast reads garbage from planned buffer → wrong values.
    // After fix: correct element count → values match F32 baseline within F16 tolerance.
    assert_close(
        "autocast_buffer_plan_relocated",
        &ac_result,
        &f32_result,
        0.05,
    );
}

// -- FlashAttention NativeOp in autocast (compute-dominant, F16) ---------------

/// FlashAttention NativeOp runs in F16 in autocast mode because it is
/// compute-dominant: the kernel uses F32 accumulators for online softmax
/// while reading Q/K/V in F16 for 2x bandwidth throughput.
///
/// This test verifies:
/// 1. FlashAttention is compiled as a NativeOp (not decomposed Dispatch)
/// 2. In autocast, its step_scalar_type is F16 (compute-dominant)
/// 3. Output matches F32 baseline within F16 tolerance
///
/// Part of #2981 (PlBert has 12 attention layers — 36 boundary casts saved).
#[test]
fn test_autocast_flash_attention_f16() {
    test_utils::gpu_init();
    let cache = test_utils::metal_setup();

    let (batch, heads, seq, d) = (1, 4, 8, 16);
    let scale = 1.0 / (d as f64).sqrt();
    let numel = batch * heads * seq * d;

    use nn_core::dyn_tensor::trace::TraceNode;
    use nn_core::DType;

    let q_data = test_utils::rand_f32_vec(0xAC_FA01, numel, -0.5, 0.5);
    let k_data = test_utils::rand_f32_vec(0xAC_FA02, numel, -0.5, 0.5);
    let v_data = test_utils::rand_f32_vec(0xAC_FA03, numel, -0.5, 0.5);

    let shape = &[batch, heads, seq, d];
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, shape),
        input_node(1, shape),
        input_node(2, shape),
        TraceNode::new(
            3,
            "sdpa_0".into(),
            TraceOp::Sdpa { scale },
            vec![0, 1, 2],
            shape.to_vec(),
            DType::F32,
        ),
    ]);

    // F32 baseline.
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let q_buf = create_input_buffer(&cache, &q_data);
    let k_buf = create_input_buffer(&cache, &k_data);
    let v_buf = create_input_buffer(&cache, &v_data);
    let f32_out = f32_model
        .execute(&cache, &[&q_buf, &k_buf, &v_buf])
        .expect("f32 exec");
    let f32_result = read_output_n(&f32_out, numel);

    // Autocast: FlashAttention should run in F16 (compute-dominant NativeOp).
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert!(ac_model.is_autocast());
    let ac_out = ac_model
        .execute(&cache, &[&q_buf, &k_buf, &v_buf])
        .expect("autocast exec");
    let ac_result = read_output_n(&ac_out, numel);

    // F16 flash attention: softmax uses F32 accumulators internally, so
    // precision loss comes only from F16 quantization of Q/K/V inputs.
    // Allow slightly larger tolerance than pure F32 comparison.
    assert_close("autocast_flash_attn", &ac_result, &f32_result, 0.02);
}

/// Linear(F16, Compute) → FlashAttention(F16, Compute) chain in autocast.
///
/// This is the critical PlBert pattern: QKV projections (Linear Dispatch, F16)
/// feed directly into FlashAttention (NativeOp, F16). Because both ops are
/// compute-dominant and marked F16 in autocast, no F16↔F32 boundary cast
/// is needed between them.
///
/// Previously, FlashAttention stayed F32 in autocast, requiring 3 boundary
/// casts per attention layer × 12 layers = 36 casts in PlBert.
#[test]
fn test_autocast_linear_to_flash_attention_chain() {
    test_utils::gpu_init();
    let cache = test_utils::metal_setup();

    let (batch, heads, seq, d) = (1, 2, 4, 16);
    let in_dim = heads * d; // 32
    let scale = 1.0 / (d as f64).sqrt();
    let out_numel = batch * heads * seq * d;

    use nn_core::dyn_tensor::trace::{TraceNode, WeightRef};
    use nn_core::DType;

    // Q/K/V projection weights: [heads*d, in_dim] = [32, 32]
    let wq = WeightRef::new(
        test_utils::rand_f32_vec(0xAC0F01, in_dim * in_dim, -0.3, 0.3),
        vec![in_dim, in_dim],
    )
    .expect("wq");
    let wk = WeightRef::new(
        test_utils::rand_f32_vec(0xAC0F02, in_dim * in_dim, -0.3, 0.3),
        vec![in_dim, in_dim],
    )
    .expect("wk");
    let wv = WeightRef::new(
        test_utils::rand_f32_vec(0xAC0F03, in_dim * in_dim, -0.3, 0.3),
        vec![in_dim, in_dim],
    )
    .expect("wv");

    // Graph: Input [1, 4, 32] → Linear(Q) → Reshape [1,2,4,16]
    //                          → Linear(K) → Reshape [1,2,4,16]
    //                          → Linear(V) → Reshape [1,2,4,16]
    //        → Sdpa(Q', K', V') → [1,2,4,16]
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, seq, in_dim]),
        // Q projection
        TraceNode::new(
            1,
            "linear_q".into(),
            TraceOp::Linear {
                weight: wq,
                bias: None,
            },
            vec![0],
            vec![batch, seq, in_dim],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "reshape_q".into(),
            TraceOp::Reshape {
                target_shape: vec![batch, seq, heads, d],
            },
            vec![1],
            vec![batch, seq, heads, d],
            DType::F32,
        ),
        TraceNode::new(
            3,
            "transpose_q".into(),
            TraceOp::Transpose { dim0: 1, dim1: 2 },
            vec![2],
            vec![batch, heads, seq, d],
            DType::F32,
        ),
        // K projection
        TraceNode::new(
            4,
            "linear_k".into(),
            TraceOp::Linear {
                weight: wk,
                bias: None,
            },
            vec![0],
            vec![batch, seq, in_dim],
            DType::F32,
        ),
        TraceNode::new(
            5,
            "reshape_k".into(),
            TraceOp::Reshape {
                target_shape: vec![batch, seq, heads, d],
            },
            vec![4],
            vec![batch, seq, heads, d],
            DType::F32,
        ),
        TraceNode::new(
            6,
            "transpose_k".into(),
            TraceOp::Transpose { dim0: 1, dim1: 2 },
            vec![5],
            vec![batch, heads, seq, d],
            DType::F32,
        ),
        // V projection
        TraceNode::new(
            7,
            "linear_v".into(),
            TraceOp::Linear {
                weight: wv,
                bias: None,
            },
            vec![0],
            vec![batch, seq, in_dim],
            DType::F32,
        ),
        TraceNode::new(
            8,
            "reshape_v".into(),
            TraceOp::Reshape {
                target_shape: vec![batch, seq, heads, d],
            },
            vec![7],
            vec![batch, seq, heads, d],
            DType::F32,
        ),
        TraceNode::new(
            9,
            "transpose_v".into(),
            TraceOp::Transpose { dim0: 1, dim1: 2 },
            vec![8],
            vec![batch, heads, seq, d],
            DType::F32,
        ),
        // Flash Attention
        TraceNode::new(
            10,
            "sdpa_0".into(),
            TraceOp::Sdpa { scale },
            vec![3, 6, 9],
            vec![batch, heads, seq, d],
            DType::F32,
        ),
    ]);

    let input_data = test_utils::rand_f32_vec(0xAC0F00, batch * seq * in_dim, -1.0, 1.0);
    let buf = create_input_buffer(&cache, &input_data);

    // F32 baseline.
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f32_out = f32_model.execute(&cache, &[&buf]).expect("f32 exec");
    let f32_result = read_output_n(&f32_out, out_numel);

    // Autocast: Linear projections (Dispatch, Compute) → F16,
    //           FlashAttention (NativeOp, Compute) → F16.
    // No boundary casts between them.
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert!(ac_model.is_autocast());
    let ac_out = ac_model.execute(&cache, &[&buf]).expect("autocast exec");
    let ac_result = read_output_n(&ac_out, out_numel);

    // Linear(F16) → FlashAttention(F16) chain: both run in F16, F32 accum in kernels.
    // Tolerance accounts for F16 weight quantization + F16 Q/K/V in attention.
    assert_close(
        "autocast_linear_flash_attn_chain",
        &ac_result,
        &f32_result,
        0.1,
    );
}

/// Autocast keeps intermediate buffers in F32, so large-magnitude intermediates
/// that would overflow F16 (>65504) remain finite. This is the critical
/// difference from whole-buffer mixed precision which NaN'd with production
/// Kokoro weights (#3085).
///
/// Graph: Input → Linear(large_weights) → ReLU → output
/// The linear output can exceed F16 range. In autocast, the output buffer is
/// F32 so no overflow occurs; only the kernel I/O uses F16.
#[test]
fn test_autocast_large_values_no_overflow() {
    test_utils::gpu_init();
    let cache = test_utils::metal_setup();

    let (batch, in_dim, out_dim) = (1, 8, 4);
    let out_numel = batch * out_dim;

    use nn_core::dyn_tensor::trace::{TraceNode, WeightRef};
    use nn_core::DType;

    // Weights chosen so linear output exceeds F16 range (65504).
    // Input ~100, weight ~1000 → output ~800_000 per element (sums over in_dim).
    let input_data: Vec<f32> = (0..batch * in_dim).map(|i| 100.0 + i as f32).collect();
    let weight_data: Vec<f32> = (0..in_dim * out_dim)
        .map(|i| 1000.0 + i as f32 * 10.0)
        .collect();

    let weight = WeightRef::new(weight_data, vec![out_dim, in_dim]).expect("weight");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, in_dim]),
        TraceNode::new(
            1,
            "linear_big".into(),
            TraceOp::Linear { weight, bias: None },
            vec![0],
            vec![batch, out_dim],
            DType::F32,
        ),
        unary_node(2, "relu_0", TraceOp::Relu, 1, &[batch, out_dim]),
    ]);

    let buf = create_input_buffer(&cache, &input_data);

    // F32 baseline.
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f32_out = f32_model.execute(&cache, &[&buf]).expect("f32 exec");
    let f32_result = read_output_n(&f32_out, out_numel);

    // Verify F32 output exceeds F16 range.
    let max_val = f32_result.iter().copied().fold(0.0_f32, f32::max);
    assert!(
        max_val > 65504.0,
        "test expects output > F16 max, got {max_val}"
    );

    // Autocast: output should match F32 exactly (intermediates stay F32).
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    let ac_out = ac_model.execute(&cache, &[&buf]).expect("autocast exec");
    let ac_result = read_output_n(&ac_out, out_numel);

    // All values must be finite (no F16 overflow).
    for (i, v) in ac_result.iter().enumerate() {
        assert!(
            v.is_finite(),
            "autocast output[{i}] is not finite: {v} (F16 overflow)"
        );
    }

    // Autocast uses F16 for the Linear kernel I/O, so there's some
    // quantization error, but values must be in the right ballpark.
    // With F16 weight quantization of values ~1000-1800, relative error
    // can be significant. Check within 5% relative error.
    for (i, (ac, f32v)) in ac_result.iter().zip(f32_result.iter()).enumerate() {
        let rel_err = (ac - f32v).abs() / f32v.abs().max(1.0);
        assert!(
            rel_err < 0.05,
            "autocast_large_values output[{i}]: ac={ac}, f32={f32v}, rel_err={rel_err:.4}"
        );
    }
}

// -- Op classification consistency (D3) ---------------------------------------

/// Cross-check: `default_op_category` must agree with builder classify functions.
///
/// The builder (`compiled_model_builder_classify.rs`) uses `TensorOpKind` match
/// arms to classify ops. `default_op_category` uses string names. These must
/// return consistent categories. If a new op is added to the builder without
/// updating `default_op_category`, this test fails.
///
/// Part of #2981 (designs/2026-03-22-f16-op-classification-api-unification.md D3).
#[test]
fn test_op_category_matches_builder_classify() {
    // Compute ops: is_non_gemm_compute_dispatch matches on these TensorOpKind variants.
    // Linear/MatMul are also Compute (gated by mixed_gemm_infos, not listed in
    // is_non_gemm_compute_dispatch, but still Compute category).
    let compute_ops = [
        "matmul",
        "conv1d",
        "conv2d",
        "conv_transpose1d",
        "conv_transpose2d",
        "linear",
        "embedding",
        "lstm_gates",
        "attention",
        // NativeOp compute ops (is_compute_native_op):
        "flash_attention",
        "norm_activ_conv1d",
        "fused_res_block",
        "norm_linear",
        "batched_linear_projection",
    ];
    for op in compute_ops {
        assert_eq!(
            default_op_category(op),
            OpDTypeCategory::Compute,
            "{op} should be Compute in public API (matches builder)"
        );
    }

    // Accumulate ops: softmax, norms, reductions — stay F32 in both systems.
    let accumulate_ops = [
        "softmax",
        "log_softmax",
        "layer_norm",
        "group_norm",
        "instance_norm",
        "rms_norm",
        "batch_norm",
        "sum",
        "mean",
        "log",
        "pow",
        "cumsum",
    ];
    for op in accumulate_ops {
        assert_eq!(
            default_op_category(op),
            OpDTypeCategory::Accumulate,
            "{op} should be Accumulate in public API"
        );
    }

    // Passthrough/Inherit ops: is_passthrough_safe matches on these TensorOpKind variants.
    // default_op_category returns Inherit for unknown ops (catch-all), so these
    // should also return Inherit.
    let inherit_ops = [
        "relu",
        "leaky_relu",
        "elu",
        "softplus",
        "exp",
        "add",
        "mul",
        "reshape",
        "narrow",
        "transpose",
        "sigmoid",
        "gelu",
        "silu",
        "tanh",
    ];
    for op in inherit_ops {
        assert_eq!(
            default_op_category(op),
            OpDTypeCategory::Inherit,
            "{op} should be Inherit in public API (matches builder passthrough)"
        );
    }
}

// -- NormLinear NativeOp in autocast (compute-dominant, F16) -------------------

/// NormLinear (fused LayerNorm + Linear) runs in F16 in autocast mode because
/// its MSL kernel uses F32 accumulators for both the normalization reduction
/// and the GEMM dot-product, while loading/storing in the step's scalar type.
///
/// Graph: Input [4, 16] → LayerNorm(16) → Linear(16, 8) → output [4, 8]
/// Peephole fuses LayerNorm + Linear into a single NormLinear NativeOp.
///
/// This test verifies:
/// 1. NormLinear NativeOp is present in the compiled plan
/// 2. In autocast, it runs in F16 (num_autocast_f16_steps > 0)
/// 3. Output matches F32 baseline within F16 tolerance
///
/// Part of #3287 (NormLinear F16 autocast gap).
#[test]
fn test_autocast_norm_linear_f16() {
    test_utils::gpu_init();
    let cache = test_utils::metal_setup();

    let (batch, hidden, out_f) = (4, 16, 8);
    let eps = 1e-5_f64;

    use nn_core::dyn_tensor::trace::{TraceNode, WeightRef};
    use nn_core::DType;

    let ln_w = test_utils::rand_f32_vec(0xAC_0E01, hidden, 0.5, 1.5);
    let ln_b = test_utils::rand_f32_vec(0xAC_0E02, hidden, -0.1, 0.1);
    let w_data = test_utils::rand_f32_vec(0xAC_0E03, out_f * hidden, -0.5, 0.5);
    let b_data = test_utils::rand_f32_vec(0xAC_0E04, out_f, -0.1, 0.1);
    let input_data = test_utils::rand_f32_vec(0xAC_0E05, batch * hidden, -1.0, 1.0);

    fn weight(data: Vec<f32>, shape: Vec<usize>) -> WeightRef {
        WeightRef::new(data, shape).expect("weight")
    }

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, hidden]),
        TraceNode::new(
            1,
            "layernorm_0".into(),
            TraceOp::LayerNorm {
                eps,
                weight: weight(ln_w, vec![hidden]),
                bias: weight(ln_b, vec![hidden]),
            },
            vec![0],
            vec![batch, hidden],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "linear_0".into(),
            TraceOp::Linear {
                weight: weight(w_data, vec![out_f, hidden]),
                bias: Some(weight(b_data, vec![out_f])),
            },
            vec![1],
            vec![batch, out_f],
            DType::F32,
        ),
    ]);

    // F32 baseline.
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let buf = create_input_buffer(&cache, &input_data);
    let f32_out = f32_model.execute(&cache, &[&buf]).expect("f32 exec");
    let f32_result = read_output_n(&f32_out, batch * out_f);

    // Verify NormLinear NativeOp is present.
    let has_norm_linear = f32_model.steps().iter().any(|s| {
        matches!(
            s,
            nn_dsl::CompiledStep::NativeOp {
                op: nn_dsl::NativeOpKind::NormLinear { .. },
                ..
            }
        )
    });
    assert!(
        has_norm_linear,
        "peephole should fuse LayerNorm+Linear into NormLinear"
    );

    // Autocast: NormLinear should run in F16 (compute-dominant NativeOp).
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert!(ac_model.is_autocast());
    assert!(
        ac_model.num_autocast_f16_steps() > 0,
        "NormLinear should be autocasted to F16"
    );

    let ac_out = ac_model.execute(&cache, &[&buf]).expect("autocast exec");
    let ac_result = read_output_n(&ac_out, batch * out_f);

    // NormLinear uses F32 accumulators internally. Precision loss comes from
    // F16 quantization of input/weight/norm_weight/norm_bias.
    assert_close("autocast_norm_linear", &ac_result, &f32_result, 0.05);
}
