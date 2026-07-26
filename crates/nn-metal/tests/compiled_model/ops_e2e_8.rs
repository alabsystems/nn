// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end compiled model tests: tiled shared-memory GEMM.
//!
//! Validates the tiled GEMM middle tier (M>=16, N>=16, K>=8) that was
//! previously handled by the naive one-thread-per-element kernel.
//! Shapes are chosen to fail simdgroup requirements (K<128 or M*N<16384)
//! but pass tiled requirements.
//!
//! Part of #3230 (Gap 1).

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use super::helpers::{assert_close, compile_and_run, create_input_buffer, input_node};

fn weight(data: Vec<f32>, shape: Vec<usize>) -> WeightRef {
    WeightRef::new(data, shape).expect("weight")
}

// -- Test 57: Tiled linear (attention-sized, K<128) ---------------------------

/// Linear [32, 64] -> [32, 32]: K=64 fails simdgroup (K<128), hits tiled path.
/// Validates shared-memory tiling produces correct output for aligned dims.
#[test]
fn test_tiled_linear_aligned() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, in_f, out_f) = (32, 64, 32);
    let w_data = super::test_utils::rand_f32_vec(0xD1ED_0001, out_f * in_f, -0.5, 0.5);
    let b_data = super::test_utils::rand_f32_vec(0xD1ED_0002, out_f, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0xD1ED_0003, batch * in_f, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, in_f]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear {
                weight: weight(w_data.clone(), vec![out_f, in_f]),
                bias: Some(weight(b_data.clone(), vec![out_f])),
            },
            vec![0],
            vec![batch, out_f],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * out_f,
    );

    let expected =
        super::test_utils::linear_ref(&input_data, &w_data, Some(&b_data), batch, in_f, out_f);
    assert_close("tiled_linear_aligned", &result, &expected, 1e-3);
}

// -- Test 58: Tiled matmul (attention QK^T shape) -----------------------------

/// MatMul [64, 64] x [64, 64]: M*N=4096 < 16384, K=64 < 128.
/// Fails both simdgroup requirements, routes to tiled GEMM.
/// This matches the dominant attention QK^T shape in HTDemucs.
#[test]
fn test_tiled_matmul_attention_shape() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (m, k, n) = (64, 64, 64);
    let left = super::test_utils::rand_f32_vec(0xD1ED_0010, m * k, -1.0, 1.0);
    let right = super::test_utils::rand_f32_vec(0xD1ED_0011, k * n, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[m, k]),
        input_node(1, &[k, n]),
        TraceNode::new(
            2,
            "matmul_0".into(),
            TraceOp::MatMul,
            vec![0, 1],
            vec![m, n],
            DType::F32,
        ),
    ]);

    let left_buf = create_input_buffer(&cache, &left);
    let right_buf = create_input_buffer(&cache, &right);
    let result = compile_and_run(&cache, graph, &[&left_buf, &right_buf], m * n);

    let expected = super::test_utils::matmul_ref(&left, &right, m, k, n, false, None);
    assert_close("tiled_matmul_attention", &result, &expected, 1e-3);
}

// -- Test 59: Tiled linear (non-tile-aligned dims) ----------------------------

/// Linear [20, 33] -> [20, 19]: non-power-of-2 dims that are NOT tile-aligned.
/// Tests boundary handling (M=20 not %16, K=33 not %16, N=19 not %16).
/// The tiled kernel must zero-pad partial tiles correctly.
#[test]
fn test_tiled_linear_non_aligned() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, in_f, out_f) = (20, 33, 19);
    let w_data = super::test_utils::rand_f32_vec(0xD1ED_0020, out_f * in_f, -0.5, 0.5);
    let b_data = super::test_utils::rand_f32_vec(0xD1ED_0021, out_f, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0xD1ED_0022, batch * in_f, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, in_f]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear {
                weight: weight(w_data.clone(), vec![out_f, in_f]),
                bias: Some(weight(b_data.clone(), vec![out_f])),
            },
            vec![0],
            vec![batch, out_f],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * out_f,
    );

    let expected =
        super::test_utils::linear_ref(&input_data, &w_data, Some(&b_data), batch, in_f, out_f);
    assert_close("tiled_linear_non_aligned", &result, &expected, 1e-3);
}

// -- Test 60: Tiled linear without bias ---------------------------------------

/// Linear [32, 64] -> [32, 32] without bias: same shape as Test 57 but no bias.
/// Validates the no-bias path through the tiled kernel.
#[test]
fn test_tiled_linear_no_bias() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, in_f, out_f) = (32, 64, 32);
    let w_data = super::test_utils::rand_f32_vec(0xD1ED_0030, out_f * in_f, -0.5, 0.5);
    let input_data = super::test_utils::rand_f32_vec(0xD1ED_0031, batch * in_f, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, in_f]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear {
                weight: weight(w_data.clone(), vec![out_f, in_f]),
                bias: None,
            },
            vec![0],
            vec![batch, out_f],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * out_f,
    );

    let expected = super::test_utils::linear_ref(&input_data, &w_data, None, batch, in_f, out_f);
    assert_close("tiled_linear_no_bias", &result, &expected, 1e-3);
}

// -- Test 61: Tiled matmul F16 autocast parity --------------------------------

/// MatMul [64, 64] x [64, 64] with F16 autocast: tiled GEMM path in F16.
/// Verifies F16 tiled output matches F32 reference within 1e-2.
#[test]
fn test_tiled_matmul_f16_parity() {
    use nn_core::mixed_precision::MixedPrecisionPolicy;
    use nn_metal::compiled_model::CompiledModel;

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (m, k, n) = (64, 64, 64);
    let left = super::test_utils::rand_f32_vec(0xD1ED_0040, m * k, -1.0, 1.0);
    let right = super::test_utils::rand_f32_vec(0xD1ED_0041, k * n, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[m, k]),
        input_node(1, &[k, n]),
        TraceNode::new(
            2,
            "matmul_0".into(),
            TraceOp::MatMul,
            vec![0, 1],
            vec![m, n],
            DType::F32,
        ),
    ]);

    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let compiled = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");

    let left_buf = create_input_buffer(&cache, &left);
    let right_buf = create_input_buffer(&cache, &right);
    let out_buf = compiled
        .execute(&cache, &[&left_buf, &right_buf])
        .expect("execute");
    let result = super::helpers::read_output_n(&out_buf, m * n);

    let expected = super::test_utils::matmul_ref(&left, &right, m, k, n, false, None);
    assert_close("tiled_matmul_f16_parity", &result, &expected, 1e-2);
}

// -- Test 62: Conv1d GEMM NativeOp (im2col + simdgroup) -----------------------

/// Conv1d(256, 256, 3, pad=1) with GEMM-qualifying shapes: exercises
/// `NativeOpKind::Conv1dGemm` through the full compiled model pipeline.
///
/// M=256, K=768, N=128 → 25.2M FLOPs — well above the 2M GEMM threshold.
/// The trace compiler routes this to `CompiledStep::NativeOp { Conv1dGemm }`
/// instead of `CompiledStep::Dispatch { Conv1d }`.
///
/// Input is 3D `[B, C_in, L_in]` — the GEMM routing requires 3D shapes.
///
/// Part of #3390.
#[test]
fn test_compiled_conv1d_gemm_nativeop() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, in_ch, out_ch, ks, in_len, pad) = (1, 256, 256, 3, 128, 1);
    let out_len = (in_len + 2 * pad - ks) + 1; // stride=1

    let w_data = super::test_utils::rand_f32_vec(0x6E11_0001, out_ch * in_ch * ks, -0.1, 0.1);
    let b_data = super::test_utils::rand_f32_vec(0x6E11_0002, out_ch, -0.05, 0.05);
    let input_data =
        super::test_utils::rand_f32_vec(0x6E11_0003, batch * in_ch * in_len, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, in_ch, in_len]),
        TraceNode::new(
            1,
            "conv1d_gemm".into(),
            TraceOp::Conv1d {
                weight: weight(w_data.clone(), vec![out_ch, in_ch, ks]),
                bias: Some(weight(b_data.clone(), vec![out_ch])),
                padding: pad,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
            vec![0],
            vec![batch, out_ch, out_len],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * out_ch * out_len,
    );

    // CPU reference: batch=1 so the flat layout matches [C_out, L_out].
    let expected = super::test_utils::conv1d_ref(
        &input_data,
        &w_data,
        Some(&b_data),
        in_ch,
        out_ch,
        ks,
        in_len,
        1,
        pad,
    );
    assert_close("conv1d_gemm_nativeop", &result, &expected, 5e-4);
}

// -- Test 63: Conv1d GEMM NativeOp without bias --------------------------------

/// Conv1d(128, 128, 3, pad=1) without bias: GEMM path, no-bias variant.
/// M=128, K=384, N=256 → 12.6M FLOPs — above threshold.
/// Input is 3D `[1, C_in, L_in]` for GEMM routing.
/// Part of #3390.
#[test]
fn test_compiled_conv1d_gemm_no_bias() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, in_ch, out_ch, ks, in_len, pad) = (1, 128, 128, 3, 256, 1);
    let out_len = (in_len + 2 * pad - ks) + 1;

    let w_data = super::test_utils::rand_f32_vec(0x6E11_0010, out_ch * in_ch * ks, -0.1, 0.1);
    let input_data =
        super::test_utils::rand_f32_vec(0x6E11_0011, batch * in_ch * in_len, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, in_ch, in_len]),
        TraceNode::new(
            1,
            "conv1d_gemm_nobias".into(),
            TraceOp::Conv1d {
                weight: weight(w_data.clone(), vec![out_ch, in_ch, ks]),
                bias: None,
                padding: pad,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
            vec![0],
            vec![batch, out_ch, out_len],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * out_ch * out_len,
    );

    // CPU reference: batch=1 so flat layout matches [C_out, L_out].
    let expected = super::test_utils::conv1d_ref(
        &input_data,
        &w_data,
        None,
        in_ch,
        out_ch,
        ks,
        in_len,
        1,
        pad,
    );
    assert_close("conv1d_gemm_no_bias", &result, &expected, 5e-4);
}
