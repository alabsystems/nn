// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end compiled model tests for SeqFirst FlashAttention.
//!
//! Verifies the full pipeline: input in [B,S,H,D] → peephole pass 9 absorbs
//! Transpose(1,2) → SeqFirst FlashAttention kernel → output in [B,S,H,D].
//!
//! The trace graph simulates what multi-head attention produces:
//! ```text
//! Q [B,S,H,D] → Transpose(1,2) → [B,H,S,D] ─┐
//! K [B,S,H,D] → Transpose(1,2) → [B,H,S,D] ──┼─→ SDPA → [B,H,S,D]
//! V [B,S,H,D] → Transpose(1,2) → [B,H,S,D] ─┘         │
//!                                    Transpose(1,2) ←────┘
//!                                         → [B,S,H,D]
//! ```
//!
//! The peephole pass 9 (`absorb_attention_transposes`) replaces all 4 Transposes
//! with IdentityPassthrough and switches the SDPA to SeqFirst layout.
//!
//! Part of #3088 (attention transpose elimination) and #1815 (Tier 5 D1).

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use super::helpers::{assert_close, compile_and_run, create_input_buffer, input_node};

// -- CPU reference helpers ----------------------------------------------------

/// Transpose flat data from [B,S,H,D] to [B,H,S,D] layout.
fn transpose_bshd_to_bhsd(data: &[f32], b: usize, s: usize, h: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for bi in 0..b {
        for si in 0..s {
            for hi in 0..h {
                for di in 0..d {
                    let src = bi * s * h * d + si * h * d + hi * d + di;
                    let dst = bi * h * s * d + hi * s * d + si * d + di;
                    out[dst] = data[src];
                }
            }
        }
    }
    out
}

/// Transpose flat data from [B,H,S,D] to [B,S,H,D] layout.
fn transpose_bhsd_to_bshd(data: &[f32], b: usize, h: usize, s: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for bi in 0..b {
        for hi in 0..h {
            for si in 0..s {
                for di in 0..d {
                    let src = bi * h * s * d + hi * s * d + si * d + di;
                    let dst = bi * s * h * d + si * h * d + hi * d + di;
                    out[dst] = data[src];
                }
            }
        }
    }
    out
}

/// CPU reference SDPA on [B,H,S_q,D] layout data.
#[allow(clippy::needless_range_loop)]
fn cpu_sdpa_bhsd(
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
    let mut output = vec![0.0f32; batch * h_q * s_q * d];

    for b in 0..batch {
        for h in 0..h_q {
            let kv_h = h / group_size;
            for sq in 0..s_q {
                let q_offset = ((b * h_q + h) * s_q + sq) * d;
                let mut scores = vec![0.0f32; s_kv];
                for skv in 0..s_kv {
                    let k_offset = ((b * h_kv + kv_h) * s_kv + skv) * d;
                    let mut dot = 0.0f32;
                    for dd in 0..d {
                        dot += q[q_offset + dd] * k[k_offset + dd];
                    }
                    scores[skv] = dot * scale;
                    if causal && skv > sq {
                        scores[skv] = f32::NEG_INFINITY;
                    }
                }

                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                let mut attn = vec![0.0f32; s_kv];
                for skv in 0..s_kv {
                    attn[skv] = (scores[skv] - max_score).exp();
                    sum_exp += attn[skv];
                }
                if sum_exp > 0.0 {
                    for a in attn.iter_mut() {
                        *a /= sum_exp;
                    }
                }

                let out_offset = ((b * h_q + h) * s_q + sq) * d;
                for dd in 0..d {
                    let mut val = 0.0f32;
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

/// Full CPU reference: data in [B,S,H,D] → transpose → SDPA → transpose → [B,S,H,D].
fn cpu_sdpa_seq_first(
    q_bshd: &[f32],
    k_bshd: &[f32],
    v_bshd: &[f32],
    batch: usize,
    h_q: usize,
    h_kv: usize,
    s_q: usize,
    s_kv: usize,
    d: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let q_bhsd = transpose_bshd_to_bhsd(q_bshd, batch, s_q, h_q, d);
    let k_bhsd = transpose_bshd_to_bhsd(k_bshd, batch, s_kv, h_kv, d);
    let v_bhsd = transpose_bshd_to_bhsd(v_bshd, batch, s_kv, h_kv, d);
    let out_bhsd = cpu_sdpa_bhsd(
        &q_bhsd, &k_bhsd, &v_bhsd, batch, h_q, h_kv, s_q, s_kv, d, scale, causal,
    );
    transpose_bhsd_to_bshd(&out_bhsd, batch, h_q, s_q, d)
}

// -- Graph builder: Transpose(1,2) → SDPA → Transpose(1,2) ------------------

/// Build a trace graph that mimics standard multi-head attention layout:
///
/// Input Q/K/V in [B,S,H,D] → Transpose(1,2) → SDPA [B,H,S,D] → Transpose(1,2) → [B,S,H,D]
///
/// The peephole pass 9 should absorb all 4 transposes into a SeqFirst FlashAttention.
fn build_seq_first_sdpa_graph(
    batch: usize,
    s_q: usize,
    s_kv: usize,
    h_q: usize,
    h_kv: usize,
    d: usize,
    scale: f64,
    causal: bool,
) -> ComputationGraph {
    let sdpa_op = if causal {
        TraceOp::SdpaCausal { scale }
    } else {
        TraceOp::Sdpa { scale }
    };

    ComputationGraph::from_nodes(vec![
        // Inputs in [B, S, H, D] (sequence-first).
        input_node(0, &[batch, s_q, h_q, d]),
        input_node(1, &[batch, s_kv, h_kv, d]),
        input_node(2, &[batch, s_kv, h_kv, d]),
        // Transpose(1,2): [B,S,H,D] → [B,H,S,D].
        TraceNode::new(
            3,
            "q_transpose".into(),
            TraceOp::Transpose { dim0: 1, dim1: 2 },
            vec![0],
            vec![batch, h_q, s_q, d],
            DType::F32,
        ),
        TraceNode::new(
            4,
            "k_transpose".into(),
            TraceOp::Transpose { dim0: 1, dim1: 2 },
            vec![1],
            vec![batch, h_kv, s_kv, d],
            DType::F32,
        ),
        TraceNode::new(
            5,
            "v_transpose".into(),
            TraceOp::Transpose { dim0: 1, dim1: 2 },
            vec![2],
            vec![batch, h_kv, s_kv, d],
            DType::F32,
        ),
        // SDPA on [B,H,S,D] layout.
        TraceNode::new(
            6,
            "sdpa".into(),
            sdpa_op,
            vec![3, 4, 5],
            vec![batch, h_q, s_q, d],
            DType::F32,
        ),
        // Transpose(1,2) output: [B,H,S,D] → [B,S,H,D].
        TraceNode::new(
            7,
            "out_transpose".into(),
            TraceOp::Transpose { dim0: 1, dim1: 2 },
            vec![6],
            vec![batch, s_q, h_q, d],
            DType::F32,
        ),
    ])
}

// -- Tests --------------------------------------------------------------------

/// B=1, H=4, S=32, D=64: standard MHA with SeqFirst peephole absorption.
#[test]
fn test_seq_first_flash_attn_noncausal() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, h_q, s_q, d) = (1, 4, 32, 64);
    let h_kv = h_q;
    let s_kv = s_q;
    let scale = 1.0 / (d as f64).sqrt();
    let numel = batch * s_q * h_q * d;

    let q_data = super::test_utils::rand_f32_vec(0x5F01_0001, numel, -1.0, 1.0);
    let k_data = super::test_utils::rand_f32_vec(0x5F01_0002, numel, -1.0, 1.0);
    let v_data = super::test_utils::rand_f32_vec(0x5F01_0003, numel, -1.0, 1.0);

    let graph = build_seq_first_sdpa_graph(batch, s_q, s_kv, h_q, h_kv, d, scale, false);

    let q_buf = create_input_buffer(&cache, &q_data);
    let k_buf = create_input_buffer(&cache, &k_data);
    let v_buf = create_input_buffer(&cache, &v_data);

    let result = compile_and_run(&cache, graph, &[&q_buf, &k_buf, &v_buf], numel);

    let expected = cpu_sdpa_seq_first(
        &q_data,
        &k_data,
        &v_data,
        batch,
        h_q,
        h_kv,
        s_q,
        s_kv,
        d,
        scale as f32,
        false,
    );
    assert_close("seq_first_noncausal", &result, &expected, 1e-3);
}

/// B=2, H=8, S=64, D=64: batched MHA with SeqFirst.
#[test]
fn test_seq_first_flash_attn_batched() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, h_q, s, d) = (2, 8, 64, 64);
    let h_kv = h_q;
    let scale = 1.0 / (d as f64).sqrt();
    let numel = batch * s * h_q * d;

    let q_data = super::test_utils::rand_f32_vec(0x5F02_0001, numel, -0.5, 0.5);
    let k_data = super::test_utils::rand_f32_vec(0x5F02_0002, numel, -0.5, 0.5);
    let v_data = super::test_utils::rand_f32_vec(0x5F02_0003, numel, -0.5, 0.5);

    let graph = build_seq_first_sdpa_graph(batch, s, s, h_q, h_kv, d, scale, false);

    let q_buf = create_input_buffer(&cache, &q_data);
    let k_buf = create_input_buffer(&cache, &k_data);
    let v_buf = create_input_buffer(&cache, &v_data);

    let result = compile_and_run(&cache, graph, &[&q_buf, &k_buf, &v_buf], numel);

    let expected = cpu_sdpa_seq_first(
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
        false,
    );
    assert_close("seq_first_batched", &result, &expected, 1e-3);
}

/// B=1, H=4, S=32, D=64: causal attention with SeqFirst.
#[test]
fn test_seq_first_flash_attn_causal() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, h_q, s, d) = (1, 4, 32, 64);
    let h_kv = h_q;
    let scale = 1.0 / (d as f64).sqrt();
    let numel = batch * s * h_q * d;

    let q_data = super::test_utils::rand_f32_vec(0x5F03_0001, numel, -1.0, 1.0);
    let k_data = super::test_utils::rand_f32_vec(0x5F03_0002, numel, -1.0, 1.0);
    let v_data = super::test_utils::rand_f32_vec(0x5F03_0003, numel, -1.0, 1.0);

    let graph = build_seq_first_sdpa_graph(batch, s, s, h_q, h_kv, d, scale, true);

    let q_buf = create_input_buffer(&cache, &q_data);
    let k_buf = create_input_buffer(&cache, &k_data);
    let v_buf = create_input_buffer(&cache, &v_data);

    let result = compile_and_run(&cache, graph, &[&q_buf, &k_buf, &v_buf], numel);

    let expected = cpu_sdpa_seq_first(
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
        true,
    );
    assert_close("seq_first_causal", &result, &expected, 1e-3);
}

/// B=1, H_q=8, H_kv=2 (GQA group=4), S=32, D=64: GQA with SeqFirst.
#[test]
fn test_seq_first_flash_attn_gqa() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, h_q, h_kv, s, d) = (1, 8, 2, 32, 64);
    let scale = 1.0 / (d as f64).sqrt();
    let q_numel = batch * s * h_q * d;
    let kv_numel = batch * s * h_kv * d;

    let q_data = super::test_utils::rand_f32_vec(0x5F04_0001, q_numel, -1.0, 1.0);
    let k_data = super::test_utils::rand_f32_vec(0x5F04_0002, kv_numel, -1.0, 1.0);
    let v_data = super::test_utils::rand_f32_vec(0x5F04_0003, kv_numel, -1.0, 1.0);

    let graph = build_seq_first_sdpa_graph(batch, s, s, h_q, h_kv, d, scale, false);

    let q_buf = create_input_buffer(&cache, &q_data);
    let k_buf = create_input_buffer(&cache, &k_data);
    let v_buf = create_input_buffer(&cache, &v_data);

    let result = compile_and_run(&cache, graph, &[&q_buf, &k_buf, &v_buf], q_numel);

    let expected = cpu_sdpa_seq_first(
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
        false,
    );
    assert_close("seq_first_gqa", &result, &expected, 1e-3);
}

/// Non-power-of-2 sequence: B=1, H=4, S=100, D=64 with SeqFirst.
#[test]
fn test_seq_first_flash_attn_non_pow2_seq() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, h_q, s, d) = (1, 4, 100, 64);
    let h_kv = h_q;
    let scale = 1.0 / (d as f64).sqrt();
    let numel = batch * s * h_q * d;

    let q_data = super::test_utils::rand_f32_vec(0x5F05_0001, numel, -1.0, 1.0);
    let k_data = super::test_utils::rand_f32_vec(0x5F05_0002, numel, -1.0, 1.0);
    let v_data = super::test_utils::rand_f32_vec(0x5F05_0003, numel, -1.0, 1.0);

    let graph = build_seq_first_sdpa_graph(batch, s, s, h_q, h_kv, d, scale, false);

    let q_buf = create_input_buffer(&cache, &q_data);
    let k_buf = create_input_buffer(&cache, &k_data);
    let v_buf = create_input_buffer(&cache, &v_data);

    let result = compile_and_run(&cache, graph, &[&q_buf, &k_buf, &v_buf], numel);

    let expected = cpu_sdpa_seq_first(
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
        false,
    );
    assert_close("seq_first_non_pow2", &result, &expected, 1e-3);
}
