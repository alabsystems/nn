// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Attention block parity tests at real model dimensions.
//!
//! Tests composed attention patterns (Q*K^T/sqrt(d) -> softmax -> V) with
//! shapes from production transformers:
//! - Whisper: 12 heads, d_model=768, d_head=64
//! - Qwen3: 32 heads, d_model=2048, d_head=64
//! - Multi-head with long sequence
//!
//! Attention is the most composition-sensitive operation: matmul -> scale ->
//! softmax -> matmul chains amplify precision differences from each component.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{LayerNorm, Linear, Module};
use nn_core::test_prng::rand_f32_vec;
use nn_core::{DType, Device};

/// Tolerance for composed attention blocks.
/// Multiple matmul + softmax stages accumulate error.
const ATTN_TOL: f32 = 1e-3;

/// Run a scaled dot-product attention block on one device.
///
/// Implements: softmax(Q @ K^T / sqrt(d_head)) @ V
/// Q, K, V are split from a single 3D input for simplicity.
fn run_attention_block(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    heads: usize,
    seq: usize,
    d_head: usize,
    device: &Device,
) -> DynTensor {
    let q = DynTensor::new(q_data, &[heads, seq, d_head], device).unwrap();
    let k = DynTensor::new(k_data, &[heads, seq, d_head], device).unwrap();
    let v = DynTensor::new(v_data, &[heads, seq, d_head], device).unwrap();

    // Q @ K^T -> [heads, seq, seq]
    let kt = k.transpose(1, 2).unwrap();
    let scores = q.matmul(&kt).unwrap();

    // Scale by 1/sqrt(d_head)
    let scale = (d_head as f32).sqrt();
    let scale_tensor = DynTensor::full(&[1], f64::from(scale), DType::F32, device).unwrap();
    let scores = scores.broadcast_div(&scale_tensor).unwrap();

    // Softmax over last dim -> attention weights
    let attn = scores.softmax(2).unwrap();

    // Attention @ V -> [heads, seq, d_head]
    attn.matmul(&v).unwrap()
}

// -- Whisper attention (12 heads, d_head=64) ---------------------------------

/// Whisper self-attention: 12 heads, seq=128, d_head=64.
/// This is the exact shape used in every Whisper encoder layer.
#[test]
fn test_attention_whisper_self_attn() {
    gpu_init();
    let heads = 12;
    let seq = 128;
    let d_head = 64;

    let q = rand_f32_vec(7000, heads * seq * d_head, -1.0, 1.0);
    let k = rand_f32_vec(7001, heads * seq * d_head, -1.0, 1.0);
    let v = rand_f32_vec(7002, heads * seq * d_head, -1.0, 1.0);

    let cpu_out = run_attention_block(&q, &k, &v, heads, seq, d_head, &Device::Cpu);
    let gpu_out = run_attention_block(&q, &k, &v, heads, seq, d_head, &Device::metal());

    assert_eq!(gpu_out.dims(), &[heads, seq, d_head]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, ATTN_TOL, "whisper_self_attn");
}

/// Whisper short-sequence attention: 12 heads, seq=16, d_head=64.
/// Decoder-side attention with short context.
#[test]
fn test_attention_whisper_short_seq() {
    gpu_init();
    let heads = 12;
    let seq = 16;
    let d_head = 64;

    let q = rand_f32_vec(7010, heads * seq * d_head, -1.0, 1.0);
    let k = rand_f32_vec(7011, heads * seq * d_head, -1.0, 1.0);
    let v = rand_f32_vec(7012, heads * seq * d_head, -1.0, 1.0);

    let cpu_out = run_attention_block(&q, &k, &v, heads, seq, d_head, &Device::Cpu);
    let gpu_out = run_attention_block(&q, &k, &v, heads, seq, d_head, &Device::metal());

    assert_eq!(gpu_out.dims(), &[heads, seq, d_head]);
    assert_gpu_cpu_close(&gpu_out, &cpu_out, ATTN_TOL, "whisper_short_seq_attn");
}

// -- Qwen3 attention (32 heads, d_head=64) ----------------------------------

/// Qwen3 self-attention: 32 heads, seq=64, d_head=64.
#[test]
fn test_attention_qwen3_self_attn() {
    gpu_init();
    let heads = 32;
    let seq = 64;
    let d_head = 64;

    let q = rand_f32_vec(7020, heads * seq * d_head, -1.0, 1.0);
    let k = rand_f32_vec(7021, heads * seq * d_head, -1.0, 1.0);
    let v = rand_f32_vec(7022, heads * seq * d_head, -1.0, 1.0);

    let cpu_out = run_attention_block(&q, &k, &v, heads, seq, d_head, &Device::Cpu);
    let gpu_out = run_attention_block(&q, &k, &v, heads, seq, d_head, &Device::metal());

    assert_eq!(gpu_out.dims(), &[heads, seq, d_head]);
    assert_gpu_cpu_close(&gpu_out, &cpu_out, ATTN_TOL, "qwen3_self_attn");
}

// -- Long sequence attention -------------------------------------------------

/// Long sequence attention: 8 heads, seq=256, d_head=64.
/// Tests attention at longer sequences where softmax precision matters more.
#[test]
fn test_attention_long_sequence() {
    gpu_init();
    let heads = 8;
    let seq = 256;
    let d_head = 64;

    let q = rand_f32_vec(7030, heads * seq * d_head, -1.0, 1.0);
    let k = rand_f32_vec(7031, heads * seq * d_head, -1.0, 1.0);
    let v = rand_f32_vec(7032, heads * seq * d_head, -1.0, 1.0);

    let cpu_out = run_attention_block(&q, &k, &v, heads, seq, d_head, &Device::Cpu);
    let gpu_out = run_attention_block(&q, &k, &v, heads, seq, d_head, &Device::metal());

    assert_eq!(gpu_out.dims(), &[heads, seq, d_head]);
    assert_gpu_cpu_close(&gpu_out, &cpu_out, ATTN_TOL, "long_seq_attn_256");
}

// -- Transformer block: attention + FFN + residual + LayerNorm ---------------

/// Full transformer block: LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual.
/// Tests the complete composition at Whisper encoder dimensions.
#[test]
fn test_transformer_block_whisper_scale() {
    gpu_init();
    let batch = 1;
    let seq = 64;
    let d_model = 768;
    let d_head = 64;
    let heads = 12;
    let ffn_hidden = 3072;

    // Weights (deterministic random).
    let ln1_w = rand_f32_vec(7100, d_model, 0.8, 1.2);
    let ln1_b = rand_f32_vec(7101, d_model, -0.05, 0.05);
    let wq_data = rand_f32_vec(7102, d_model * d_model, -0.05, 0.05);
    let wk_data = rand_f32_vec(7103, d_model * d_model, -0.05, 0.05);
    let wv_data = rand_f32_vec(7104, d_model * d_model, -0.05, 0.05);
    let wo_data = rand_f32_vec(7105, d_model * d_model, -0.05, 0.05);
    let ln2_w = rand_f32_vec(7106, d_model, 0.8, 1.2);
    let ln2_b = rand_f32_vec(7107, d_model, -0.05, 0.05);
    let ff1_data = rand_f32_vec(7108, ffn_hidden * d_model, -0.02, 0.02);
    let ff1_b = rand_f32_vec(7109, ffn_hidden, -0.01, 0.01);
    let ff2_data = rand_f32_vec(7110, d_model * ffn_hidden, -0.02, 0.02);
    let ff2_b = rand_f32_vec(7111, d_model, -0.01, 0.01);
    let x_data = rand_f32_vec(7112, batch * seq * d_model, -1.0, 1.0);

    let run = |device: &Device| -> DynTensor {
        let x = DynTensor::new(&x_data, &[batch, seq, d_model], device).unwrap();

        // LayerNorm 1
        let lw1 = DynTensor::new(&ln1_w, &[d_model], device).unwrap();
        let lb1 = DynTensor::new(&ln1_b, &[d_model], device).unwrap();
        let ln1 = LayerNorm::new(lw1, lb1, 1e-5).unwrap();
        let normed = ln1.forward(&x).unwrap();

        // Q, K, V projections: [batch, seq, d_model] -> [batch, seq, d_model]
        let wq = DynTensor::new(&wq_data, &[d_model, d_model], device).unwrap();
        let wk = DynTensor::new(&wk_data, &[d_model, d_model], device).unwrap();
        let wv = DynTensor::new(&wv_data, &[d_model, d_model], device).unwrap();
        let lq = Linear::new(wq, None).unwrap();
        let lk = Linear::new(wk, None).unwrap();
        let lv = Linear::new(wv, None).unwrap();

        let q = lq.forward(&normed).unwrap(); // [B, S, D]
        let k = lk.forward(&normed).unwrap();
        let v = lv.forward(&normed).unwrap();

        // Reshape for multi-head: [B, S, D] -> [B, S, H, Dh] -> [B, H, S, Dh]
        let q = q
            .reshape([batch, seq, heads, d_head])
            .unwrap()
            .permute([0, 2, 1, 3])
            .unwrap();
        let k = k
            .reshape([batch, seq, heads, d_head])
            .unwrap()
            .permute([0, 2, 1, 3])
            .unwrap();
        let v = v
            .reshape([batch, seq, heads, d_head])
            .unwrap()
            .permute([0, 2, 1, 3])
            .unwrap();

        // Attention: softmax(Q @ K^T / sqrt(d_head)) @ V
        let kt = k.transpose(2, 3).unwrap();
        let scores = q.matmul(&kt).unwrap();
        let scale_val = (d_head as f64).sqrt();
        let scale = DynTensor::full(&[1], scale_val, DType::F32, device).unwrap();
        let scores = scores.broadcast_div(&scale).unwrap();
        let attn = scores.softmax(3).unwrap();
        let attn_out = attn.matmul(&v).unwrap(); // [B, H, S, Dh]

        // Reshape back: [B, H, S, Dh] -> [B, S, D]
        let attn_out = attn_out
            .permute([0, 2, 1, 3])
            .unwrap()
            .reshape([batch, seq, d_model])
            .unwrap();

        // Output projection
        let wo = DynTensor::new(&wo_data, &[d_model, d_model], device).unwrap();
        let lo = Linear::new(wo, None).unwrap();
        let attn_out = lo.forward(&attn_out).unwrap();

        // Residual connection
        let h = x.add(&attn_out).unwrap();

        // LayerNorm 2
        let lw2 = DynTensor::new(&ln2_w, &[d_model], device).unwrap();
        let lb2 = DynTensor::new(&ln2_b, &[d_model], device).unwrap();
        let ln2 = LayerNorm::new(lw2, lb2, 1e-5).unwrap();
        let normed2 = ln2.forward(&h).unwrap();

        // Feed-forward: Linear -> GELU -> Linear
        let w1 = DynTensor::new(&ff1_data, &[ffn_hidden, d_model], device).unwrap();
        let b1 = DynTensor::new(&ff1_b, &[ffn_hidden], device).unwrap();
        let w2 = DynTensor::new(&ff2_data, &[d_model, ffn_hidden], device).unwrap();
        let b2 = DynTensor::new(&ff2_b, &[d_model], device).unwrap();
        let ff1_layer = Linear::new(w1, Some(b1)).unwrap();
        let ff2_layer = Linear::new(w2, Some(b2)).unwrap();

        let ff_out = ff1_layer.forward(&normed2).unwrap();
        let ff_out = ff_out.gelu().unwrap();
        let ff_out = ff2_layer.forward(&ff_out).unwrap();

        // Residual connection
        h.add(&ff_out).unwrap()
    };

    let cpu_out = run(&Device::Cpu);
    let gpu_out = run(&Device::metal());

    assert_eq!(gpu_out.dims(), &[batch, seq, d_model]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    // Full transformer block: widest tolerance due to deep composition.
    assert_gpu_cpu_close(&gpu_out, &cpu_out, 2e-3, "transformer_block_whisper");
}

// -- Attention with output projection (end-to-end attention layer) -----------

/// End-to-end attention layer: projection -> attention -> output projection.
/// Tests the full attention layer pattern with random weights.
#[test]
fn test_attention_with_projections() {
    gpu_init();
    let batch = 1;
    let seq = 32;
    let d_model = 256;
    let d_head = 64;
    let heads = 4;

    let x_data = rand_f32_vec(7200, batch * seq * d_model, -1.0, 1.0);
    let wq = rand_f32_vec(7201, d_model * d_model, -0.1, 0.1);
    let wk = rand_f32_vec(7202, d_model * d_model, -0.1, 0.1);
    let wv = rand_f32_vec(7203, d_model * d_model, -0.1, 0.1);
    let wo = rand_f32_vec(7204, d_model * d_model, -0.1, 0.1);

    let run = |device: &Device| -> DynTensor {
        let x = DynTensor::new(&x_data, &[batch, seq, d_model], device).unwrap();

        let lq = Linear::new(
            DynTensor::new(&wq, &[d_model, d_model], device).unwrap(),
            None,
        )
        .unwrap();
        let lk = Linear::new(
            DynTensor::new(&wk, &[d_model, d_model], device).unwrap(),
            None,
        )
        .unwrap();
        let lv = Linear::new(
            DynTensor::new(&wv, &[d_model, d_model], device).unwrap(),
            None,
        )
        .unwrap();
        let lo = Linear::new(
            DynTensor::new(&wo, &[d_model, d_model], device).unwrap(),
            None,
        )
        .unwrap();

        let q = lq.forward(&x).unwrap();
        let k = lk.forward(&x).unwrap();
        let v = lv.forward(&x).unwrap();

        // Multi-head reshape
        let q = q
            .reshape([batch, seq, heads, d_head])
            .unwrap()
            .permute([0, 2, 1, 3])
            .unwrap();
        let k = k
            .reshape([batch, seq, heads, d_head])
            .unwrap()
            .permute([0, 2, 1, 3])
            .unwrap();
        let v = v
            .reshape([batch, seq, heads, d_head])
            .unwrap()
            .permute([0, 2, 1, 3])
            .unwrap();

        let kt = k.transpose(2, 3).unwrap();
        let scores = q.matmul(&kt).unwrap();
        let scale_val = (d_head as f64).sqrt();
        let scale = DynTensor::full(&[1], scale_val, DType::F32, device).unwrap();
        let scores = scores.broadcast_div(&scale).unwrap();
        let attn = scores.softmax(3).unwrap();
        let out = attn.matmul(&v).unwrap();

        // Back to [batch, seq, d_model]
        let out = out
            .permute([0, 2, 1, 3])
            .unwrap()
            .reshape([batch, seq, d_model])
            .unwrap();
        lo.forward(&out).unwrap()
    };

    let cpu_out = run(&Device::Cpu);
    let gpu_out = run(&Device::metal());

    assert_eq!(gpu_out.dims(), &[batch, seq, d_model]);
    assert_gpu_cpu_close(&gpu_out, &cpu_out, ATTN_TOL, "attn_with_projections");
}
