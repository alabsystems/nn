// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end compiled model tests for NativeOpKind::FlashAttention.
//!
//! Exercises the full pipeline: build trace graph -> compile (NativeOpKind) ->
//! GPU execute via fused Flash Attention kernel -> verify against CPU reference.
//!
//! This is the ONLY NativeOpKind that was missing compiled-model parity tests.
//! The eager-path tests (`gpu_flash_attn.rs`) cover the kernel; these tests
//! verify the compiled pipeline bridge: buffer resolution, shape wiring,
//! `execute_native_flash_attention()` correctness through CompiledModel.
//!
//! Part of #2505 (FlashAttention compiled-model parity).

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use super::helpers::{assert_close, compile_and_run, create_input_buffer, input_node};

// -- CPU reference helpers ----------------------------------------------------

/// CPU reference for scaled dot-product attention (non-causal, no mask).
///
/// Q: `[B, H_q, S_q, D]`, K: `[B, H_kv, S_kv, D]`, V: `[B, H_kv, S_kv, D]`.
/// Output: `[B, H_q, S_q, D]`.
///
/// GQA: when H_q > H_kv, each KV head serves `H_q / H_kv` query heads.
fn cpu_sdpa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    h_q: usize,
    h_kv: usize,
    s_q: usize,
    s_kv: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    cpu_sdpa_inner(q, k, v, batch, h_q, h_kv, s_q, s_kv, d, scale, false)
}

/// CPU reference for causal scaled dot-product attention.
///
/// Same as `cpu_sdpa` but applies a causal mask: positions where col > row
/// are set to -inf before softmax.
fn cpu_sdpa_causal(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    h_q: usize,
    h_kv: usize,
    s: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    cpu_sdpa_inner(q, k, v, batch, h_q, h_kv, s, s, d, scale, true)
}

/// Inner implementation for CPU SDPA with optional causal masking.
#[allow(clippy::needless_range_loop)]
fn cpu_sdpa_inner(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    h_q: usize,
    h_kv: usize,
    s_q: usize,
    s_kv: usize,
    d: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let group_size = h_q / h_kv;
    let mut output = vec![0.0_f32; batch * h_q * s_q * d];

    for b in 0..batch {
        for h in 0..h_q {
            let kv_h = h / group_size;

            for sq in 0..s_q {
                // Compute scores: Q[b,h,sq,:] @ K[b,kv_h,:,:]^T * scale
                let mut scores = vec![0.0_f32; s_kv];
                let q_offset = ((b * h_q + h) * s_q + sq) * d;
                for skv in 0..s_kv {
                    let k_offset = ((b * h_kv + kv_h) * s_kv + skv) * d;
                    let mut dot = 0.0_f32;
                    for dd in 0..d {
                        dot += q[q_offset + dd] * k[k_offset + dd];
                    }
                    scores[skv] = dot * scale;

                    // Causal mask: mask positions where skv > sq.
                    if causal && skv > sq {
                        scores[skv] = f32::NEG_INFINITY;
                    }
                }

                // Softmax over scores.
                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0_f32;
                let mut attn = vec![0.0_f32; s_kv];
                for skv in 0..s_kv {
                    attn[skv] = (scores[skv] - max_score).exp();
                    sum_exp += attn[skv];
                }
                if sum_exp > 0.0 {
                    for a in attn.iter_mut() {
                        *a /= sum_exp;
                    }
                }

                // Output: attn @ V[b,kv_h,:,:] -> [D]
                let out_offset = ((b * h_q + h) * s_q + sq) * d;
                for dd in 0..d {
                    let mut val = 0.0_f32;
                    for skv in 0..s_kv {
                        let v_offset = ((b * h_kv + kv_h) * s_kv + skv) * d;
                        val += attn[skv] * v[v_offset + dd];
                    }
                    output[out_offset + dd] = val;
                }
            }
        }
    }
    output
}

// -- Test: FlashAttention (non-causal) through CompiledModel ------------------

/// B=1, H=4, S_q=32, S_kv=32, D=64: standard MHA through compiled pipeline.
///
/// Verifies NativeOpKind::FlashAttention { causal: false } executes correctly
/// through the full compiled model pipeline (trace -> compile -> GPU execute ->
/// CPU readback).
#[test]
fn test_compiled_flash_attn_noncausal() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, heads, s_q, s_kv, d) = (1, 4, 32, 32, 64);
    let scale = 1.0 / (d as f64).sqrt();
    let q_numel = batch * heads * s_q * d;
    let kv_numel = batch * heads * s_kv * d;
    let out_numel = batch * heads * s_q * d;

    let q_data = super::test_utils::rand_f32_vec(0xFA01_0001, q_numel, -1.0, 1.0);
    let k_data = super::test_utils::rand_f32_vec(0xFA01_0002, kv_numel, -1.0, 1.0);
    let v_data = super::test_utils::rand_f32_vec(0xFA01_0003, kv_numel, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, heads, s_q, d]),
        input_node(1, &[batch, heads, s_kv, d]),
        input_node(2, &[batch, heads, s_kv, d]),
        TraceNode::new(
            3,
            "sdpa_0".into(),
            TraceOp::Sdpa { scale },
            vec![0, 1, 2],
            vec![batch, heads, s_q, d],
            DType::F32,
        ),
    ]);

    let q_buf = create_input_buffer(&cache, &q_data);
    let k_buf = create_input_buffer(&cache, &k_data);
    let v_buf = create_input_buffer(&cache, &v_data);

    let result = compile_and_run(&cache, graph, &[&q_buf, &k_buf, &v_buf], out_numel);

    let expected = cpu_sdpa(
        &q_data,
        &k_data,
        &v_data,
        batch,
        heads,
        heads,
        s_q,
        s_kv,
        d,
        scale as f32,
    );
    assert_close("flash_attn_noncausal", &result, &expected, 1e-3);
}

/// B=2, H=8, S_q=64, S_kv=64, D=64: batched MHA through compiled pipeline.
#[test]
fn test_compiled_flash_attn_noncausal_batched() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, heads, s, d) = (2, 8, 64, 64);
    let scale = 1.0 / (d as f64).sqrt();
    let q_numel = batch * heads * s * d;
    let out_numel = q_numel;

    let q_data = super::test_utils::rand_f32_vec(0xFA01_0010, q_numel, -0.5, 0.5);
    let k_data = super::test_utils::rand_f32_vec(0xFA01_0011, q_numel, -0.5, 0.5);
    let v_data = super::test_utils::rand_f32_vec(0xFA01_0012, q_numel, -0.5, 0.5);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, heads, s, d]),
        input_node(1, &[batch, heads, s, d]),
        input_node(2, &[batch, heads, s, d]),
        TraceNode::new(
            3,
            "sdpa_0".into(),
            TraceOp::Sdpa { scale },
            vec![0, 1, 2],
            vec![batch, heads, s, d],
            DType::F32,
        ),
    ]);

    let q_buf = create_input_buffer(&cache, &q_data);
    let k_buf = create_input_buffer(&cache, &k_data);
    let v_buf = create_input_buffer(&cache, &v_data);

    let result = compile_and_run(&cache, graph, &[&q_buf, &k_buf, &v_buf], out_numel);

    let expected = cpu_sdpa(
        &q_data,
        &k_data,
        &v_data,
        batch,
        heads,
        heads,
        s,
        s,
        d,
        scale as f32,
    );
    assert_close("flash_attn_noncausal_batched", &result, &expected, 1e-3);
}

// -- Test: FlashAttention (causal) through CompiledModel ----------------------

/// B=1, H=4, S=32, D=64: causal attention through compiled pipeline.
///
/// Verifies NativeOpKind::FlashAttention { causal: true } with SdpaCausal.
#[test]
fn test_compiled_flash_attn_causal() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, heads, s, d) = (1, 4, 32, 64);
    let scale = 1.0 / (d as f64).sqrt();
    let numel = batch * heads * s * d;

    let q_data = super::test_utils::rand_f32_vec(0xFA02_0001, numel, -1.0, 1.0);
    let k_data = super::test_utils::rand_f32_vec(0xFA02_0002, numel, -1.0, 1.0);
    let v_data = super::test_utils::rand_f32_vec(0xFA02_0003, numel, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, heads, s, d]),
        input_node(1, &[batch, heads, s, d]),
        input_node(2, &[batch, heads, s, d]),
        TraceNode::new(
            3,
            "sdpa_causal_0".into(),
            TraceOp::SdpaCausal { scale },
            vec![0, 1, 2],
            vec![batch, heads, s, d],
            DType::F32,
        ),
    ]);

    let q_buf = create_input_buffer(&cache, &q_data);
    let k_buf = create_input_buffer(&cache, &k_data);
    let v_buf = create_input_buffer(&cache, &v_data);

    let result = compile_and_run(&cache, graph, &[&q_buf, &k_buf, &v_buf], numel);

    let expected = cpu_sdpa_causal(
        &q_data,
        &k_data,
        &v_data,
        batch,
        heads,
        heads,
        s,
        d,
        scale as f32,
    );
    assert_close("flash_attn_causal", &result, &expected, 1e-3);
}

/// B=2, H=8, S=64, D=64: batched causal attention through compiled pipeline.
#[test]
fn test_compiled_flash_attn_causal_batched() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, heads, s, d) = (2, 8, 64, 64);
    let scale = 1.0 / (d as f64).sqrt();
    let numel = batch * heads * s * d;

    let q_data = super::test_utils::rand_f32_vec(0xFA02_0010, numel, -0.5, 0.5);
    let k_data = super::test_utils::rand_f32_vec(0xFA02_0011, numel, -0.5, 0.5);
    let v_data = super::test_utils::rand_f32_vec(0xFA02_0012, numel, -0.5, 0.5);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, heads, s, d]),
        input_node(1, &[batch, heads, s, d]),
        input_node(2, &[batch, heads, s, d]),
        TraceNode::new(
            3,
            "sdpa_causal_0".into(),
            TraceOp::SdpaCausal { scale },
            vec![0, 1, 2],
            vec![batch, heads, s, d],
            DType::F32,
        ),
    ]);

    let q_buf = create_input_buffer(&cache, &q_data);
    let k_buf = create_input_buffer(&cache, &k_data);
    let v_buf = create_input_buffer(&cache, &v_data);

    let result = compile_and_run(&cache, graph, &[&q_buf, &k_buf, &v_buf], numel);

    let expected = cpu_sdpa_causal(
        &q_data,
        &k_data,
        &v_data,
        batch,
        heads,
        heads,
        s,
        d,
        scale as f32,
    );
    assert_close("flash_attn_causal_batched", &result, &expected, 1e-3);
}

// -- Test: FlashAttention (GQA) through CompiledModel -------------------------

/// B=1, H_q=8, H_kv=2 (group=4), S=32, D=64: GQA non-causal.
///
/// Verifies NativeOpKind::FlashAttention with grouped-query attention where
/// Q has more heads than K/V. The GPU kernel maps each Q head to its KV group.
#[test]
fn test_compiled_flash_attn_gqa_noncausal() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, h_q, h_kv, s, d) = (1, 8, 2, 32, 64);
    let scale = 1.0 / (d as f64).sqrt();
    let q_numel = batch * h_q * s * d;
    let kv_numel = batch * h_kv * s * d;
    let out_numel = q_numel;

    let q_data = super::test_utils::rand_f32_vec(0xFA03_0001, q_numel, -1.0, 1.0);
    let k_data = super::test_utils::rand_f32_vec(0xFA03_0002, kv_numel, -1.0, 1.0);
    let v_data = super::test_utils::rand_f32_vec(0xFA03_0003, kv_numel, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, h_q, s, d]),
        input_node(1, &[batch, h_kv, s, d]),
        input_node(2, &[batch, h_kv, s, d]),
        TraceNode::new(
            3,
            "sdpa_0".into(),
            TraceOp::Sdpa { scale },
            vec![0, 1, 2],
            vec![batch, h_q, s, d],
            DType::F32,
        ),
    ]);

    let q_buf = create_input_buffer(&cache, &q_data);
    let k_buf = create_input_buffer(&cache, &k_data);
    let v_buf = create_input_buffer(&cache, &v_data);

    let result = compile_and_run(&cache, graph, &[&q_buf, &k_buf, &v_buf], out_numel);

    let expected = cpu_sdpa(
        &q_data,
        &k_data,
        &v_data,
        batch,
        h_q,
        h_kv,
        s,
        s,
        d,
        scale as f32,
    );
    assert_close("flash_attn_gqa_noncausal", &result, &expected, 1e-3);
}

/// B=1, H_q=8, H_kv=2 (group=4), S=32, D=64: GQA causal.
///
/// Combines GQA head mapping with causal masking in the compiled pipeline.
#[test]
fn test_compiled_flash_attn_gqa_causal() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, h_q, h_kv, s, d) = (1, 8, 2, 32, 64);
    let scale = 1.0 / (d as f64).sqrt();
    let q_numel = batch * h_q * s * d;
    let kv_numel = batch * h_kv * s * d;
    let out_numel = q_numel;

    let q_data = super::test_utils::rand_f32_vec(0xFA04_0001, q_numel, -1.0, 1.0);
    let k_data = super::test_utils::rand_f32_vec(0xFA04_0002, kv_numel, -1.0, 1.0);
    let v_data = super::test_utils::rand_f32_vec(0xFA04_0003, kv_numel, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, h_q, s, d]),
        input_node(1, &[batch, h_kv, s, d]),
        input_node(2, &[batch, h_kv, s, d]),
        TraceNode::new(
            3,
            "sdpa_causal_0".into(),
            TraceOp::SdpaCausal { scale },
            vec![0, 1, 2],
            vec![batch, h_q, s, d],
            DType::F32,
        ),
    ]);

    let q_buf = create_input_buffer(&cache, &q_data);
    let k_buf = create_input_buffer(&cache, &k_data);
    let v_buf = create_input_buffer(&cache, &v_data);

    let result = compile_and_run(&cache, graph, &[&q_buf, &k_buf, &v_buf], out_numel);

    let expected = cpu_sdpa_causal(
        &q_data,
        &k_data,
        &v_data,
        batch,
        h_q,
        h_kv,
        s,
        d,
        scale as f32,
    );
    assert_close("flash_attn_gqa_causal", &result, &expected, 1e-3);
}

// -- Test: Cross-attention (S_q != S_kv) through CompiledModel ----------------

/// B=1, H=4, S_q=16, S_kv=64, D=64: cross-attention (non-causal).
///
/// Verifies the compiled pipeline handles asymmetric sequence lengths correctly
/// through the buffer resolution and shape wiring in execute_native_flash_attention.
#[test]
fn test_compiled_flash_attn_cross_attention() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, heads, s_q, s_kv, d) = (1, 4, 16, 64, 64);
    let scale = 1.0 / (d as f64).sqrt();
    let q_numel = batch * heads * s_q * d;
    let kv_numel = batch * heads * s_kv * d;
    let out_numel = q_numel;

    let q_data = super::test_utils::rand_f32_vec(0xFA05_0001, q_numel, -1.0, 1.0);
    let k_data = super::test_utils::rand_f32_vec(0xFA05_0002, kv_numel, -1.0, 1.0);
    let v_data = super::test_utils::rand_f32_vec(0xFA05_0003, kv_numel, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, heads, s_q, d]),
        input_node(1, &[batch, heads, s_kv, d]),
        input_node(2, &[batch, heads, s_kv, d]),
        TraceNode::new(
            3,
            "sdpa_0".into(),
            TraceOp::Sdpa { scale },
            vec![0, 1, 2],
            vec![batch, heads, s_q, d],
            DType::F32,
        ),
    ]);

    let q_buf = create_input_buffer(&cache, &q_data);
    let k_buf = create_input_buffer(&cache, &k_data);
    let v_buf = create_input_buffer(&cache, &v_data);

    let result = compile_and_run(&cache, graph, &[&q_buf, &k_buf, &v_buf], out_numel);

    let expected = cpu_sdpa(
        &q_data,
        &k_data,
        &v_data,
        batch,
        heads,
        heads,
        s_q,
        s_kv,
        d,
        scale as f32,
    );
    assert_close("flash_attn_cross_attention", &result, &expected, 1e-3);
}
