// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MatMul parity tests at real model dimensions.
//!
//! Tests matmul with shapes drawn from production models:
//! - Transformer projections (Whisper, Qwen3): [1, seq, 768] x [768, 768]
//! - Attention scores: [heads, seq, seq] shapes
//! - Feed-forward: [1, seq, 768] x [768, 3072] (4x expansion)
//! - LSTM gates: [1, 128] x [128, 512]
//!
//! These sizes exercise the simdgroup GEMM path on Metal and catch
//! precision issues that hide at small test dimensions.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_prng::rand_f32_vec;
use nn_core::Device;

/// Tolerance for matmul at production scale.
/// Large K (inner dimension) accumulates more floating-point error.
const TOL: f32 = 1e-3;

fn run_matmul_parity(seed: u64, m: usize, k: usize, n: usize, label: &str) {
    let a_data = rand_f32_vec(seed, m * k, -1.0, 1.0);
    let b_data = rand_f32_vec(seed + 1, k * n, -1.0, 1.0);

    let a_cpu = DynTensor::new(&a_data, &[m, k], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[k, n], &Device::Cpu).unwrap();
    let c_cpu = a_cpu.matmul(&b_cpu).unwrap();

    let a_gpu = DynTensor::new(&a_data, &[m, k], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[k, n], &Device::metal()).unwrap();
    let c_gpu = a_gpu.matmul(&b_gpu).unwrap();

    assert_eq!(c_gpu.dims(), &[m, n], "{label}: output shape mismatch");
    assert_eq!(
        c_gpu.dims(),
        c_cpu.dims(),
        "{label}: cpu/gpu shape mismatch"
    );
    assert_gpu_cpu_close(&c_gpu, &c_cpu, TOL, label);
}

fn run_batched_matmul_parity(seed: u64, batch: usize, m: usize, k: usize, n: usize, label: &str) {
    let a_data = rand_f32_vec(seed, batch * m * k, -1.0, 1.0);
    let b_data = rand_f32_vec(seed + 1, batch * k * n, -1.0, 1.0);

    let a_cpu = DynTensor::new(&a_data, &[batch, m, k], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[batch, k, n], &Device::Cpu).unwrap();
    let c_cpu = a_cpu.matmul(&b_cpu).unwrap();

    let a_gpu = DynTensor::new(&a_data, &[batch, m, k], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[batch, k, n], &Device::metal()).unwrap();
    let c_gpu = a_gpu.matmul(&b_gpu).unwrap();

    assert_eq!(
        c_gpu.dims(),
        &[batch, m, n],
        "{label}: output shape mismatch"
    );
    assert_gpu_cpu_close(&c_gpu, &c_cpu, TOL, label);
}

// -- Transformer projection shapes (Whisper/Qwen3) ---------------------------

/// Whisper encoder: self-attention Q/K/V projection.
/// Shape: [1, 128, 768] x [768, 768] -> [1, 128, 768]
#[test]
fn test_matmul_whisper_qkv_projection() {
    gpu_init();
    run_batched_matmul_parity(
        2000,
        1,   // batch
        128, // seq_len
        768, // d_model
        768, // d_model (Q, K, V each projected)
        "whisper_qkv_768x768",
    );
}

/// Whisper encoder: feed-forward expansion.
/// Shape: [1, 128, 768] x [768, 3072] -> [1, 128, 3072]
#[test]
fn test_matmul_whisper_ffn_expand() {
    gpu_init();
    run_batched_matmul_parity(
        2001,
        1,    // batch
        128,  // seq_len
        768,  // d_model
        3072, // 4x expansion
        "whisper_ffn_expand_768x3072",
    );
}

/// Whisper encoder: feed-forward contraction.
/// Shape: [1, 128, 3072] x [3072, 768] -> [1, 128, 768]
#[test]
fn test_matmul_whisper_ffn_contract() {
    gpu_init();
    run_batched_matmul_parity(
        2002,
        1,    // batch
        128,  // seq_len
        3072, // expanded hidden
        768,  // d_model
        "whisper_ffn_contract_3072x768",
    );
}

/// Qwen3: attention projection with larger model dimension.
/// Shape: [1, 64, 2048] x [2048, 2048] -> [1, 64, 2048]
#[test]
fn test_matmul_qwen3_attn_projection() {
    gpu_init();
    run_batched_matmul_parity(
        2003,
        1,    // batch
        64,   // seq_len
        2048, // d_model
        2048, // d_model
        "qwen3_attn_2048x2048",
    );
}

/// Qwen3: SwiGLU feed-forward gate projection.
/// Shape: [1, 64, 2048] x [2048, 5632] -> [1, 64, 5632]
#[test]
fn test_matmul_qwen3_swiglu_gate() {
    gpu_init();
    run_batched_matmul_parity(
        2004,
        1,    // batch
        64,   // seq_len
        2048, // d_model
        5632, // intermediate_size
        "qwen3_swiglu_2048x5632",
    );
}

// -- Attention score shapes ---------------------------------------------------

/// Multi-head attention scores: Q @ K^T.
/// Shape: [12, 128, 64] x [12, 64, 128] -> [12, 128, 128]
/// (12 heads, 128 seq_len, 64 d_head)
#[test]
fn test_matmul_attention_scores() {
    gpu_init();
    let heads = 12;
    let seq = 128;
    let d_head = 64;

    let q_data = rand_f32_vec(2010, heads * seq * d_head, -1.0, 1.0);
    let k_data = rand_f32_vec(2011, heads * d_head * seq, -1.0, 1.0);

    let q_cpu = DynTensor::new(&q_data, &[heads, seq, d_head], &Device::Cpu).unwrap();
    let k_cpu = DynTensor::new(&k_data, &[heads, d_head, seq], &Device::Cpu).unwrap();
    let scores_cpu = q_cpu.matmul(&k_cpu).unwrap();

    let q_gpu = DynTensor::new(&q_data, &[heads, seq, d_head], &Device::metal()).unwrap();
    let k_gpu = DynTensor::new(&k_data, &[heads, d_head, seq], &Device::metal()).unwrap();
    let scores_gpu = q_gpu.matmul(&k_gpu).unwrap();

    assert_eq!(scores_gpu.dims(), &[heads, seq, seq]);
    assert_gpu_cpu_close(&scores_gpu, &scores_cpu, TOL, "attn_scores_12h_128s");
}

/// GQA attention scores (grouped-query attention from Qwen3).
/// Shape: [4, 64, 128] x [4, 128, 64] -> [4, 64, 64]
/// (4 KV heads, 64 d_head, 128 seq_len)
#[test]
fn test_matmul_gqa_scores() {
    gpu_init();
    run_batched_matmul_parity(
        2012,
        4,   // kv_heads
        64,  // seq
        128, // d_head (wide)
        64,  // seq (attention output)
        "gqa_scores_4h_64s",
    );
}

// -- LSTM gate shapes (Silero VAD) -------------------------------------------

/// Silero VAD LSTM: input-hidden matmul.
/// Shape: [1, 128] x [128, 512] -> [1, 512]
/// (batch=1, input_size=128, 4*hidden_size=512)
#[test]
fn test_matmul_lstm_input_hidden() {
    gpu_init();
    run_matmul_parity(
        2020,
        1,   // batch
        128, // input_size (encoder output)
        512, // 4 * hidden_size (gates)
        "lstm_ih_128x512",
    );
}

/// Silero VAD LSTM: hidden-hidden matmul.
/// Shape: [1, 128] x [128, 512] -> [1, 512]
#[test]
fn test_matmul_lstm_hidden_hidden() {
    gpu_init();
    run_matmul_parity(
        2021,
        1,   // batch
        128, // hidden_size
        512, // 4 * hidden_size
        "lstm_hh_128x512",
    );
}

// -- Large square (common in embedding projections) --------------------------

/// Large square matmul exercising simdgroup path.
/// Shape: [512, 512] x [512, 512] -> [512, 512]
#[test]
fn test_matmul_large_square_512() {
    gpu_init();
    run_matmul_parity(2030, 512, 512, 512, "large_square_512");
}

/// Very large matmul: [1024, 768] x [768, 1024].
/// Tests sustained precision at scale.
#[test]
fn test_matmul_large_rectangular() {
    gpu_init();
    run_matmul_parity(2031, 1024, 768, 1024, "large_rect_1024x768x1024");
}
