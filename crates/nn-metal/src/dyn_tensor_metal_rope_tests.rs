#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU tests for RotaryEmbedding (RoPE) — verifies that RoPE rotation on
//! Metal GPU tensors produces results matching CPU within tolerance.
//!
//! Critical for dvoice-integration: autoregressive models (Qwen3, Whisper)
//! use RoPE in the attention layer hot path. Zero GPU coverage before this file.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::RotaryEmbedding;
use nn_core::Device;

use crate::test_common::{assert_close, init};

// -- GPU RoPE apply tests -----------------------------------------------------

/// RoPE apply on GPU 2D tensor [seq_len, head_dim] matches CPU.
#[test]
fn test_gpu_rope_apply_2d() {
    init();
    let head_dim = 4;
    let max_seq = 16;
    let seq_len = 3;
    let base = 10000.0;

    // CPU reference
    let cpu_rope = RotaryEmbedding::new(head_dim, max_seq, base, &Device::Cpu).unwrap();
    let cpu_x = DynTensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[seq_len, head_dim],
        &Device::Cpu,
    )
    .unwrap();
    let cpu_out = cpu_rope.apply(&cpu_x, 0).unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();

    // GPU
    let gpu_rope = RotaryEmbedding::new(head_dim, max_seq, base, &Device::metal()).unwrap();
    let gpu_x = cpu_x.to_device(&Device::metal()).unwrap();
    assert_eq!(gpu_x.device(), Device::metal());
    let gpu_out = gpu_rope.apply(&gpu_x, 0).unwrap();
    assert_eq!(
        gpu_out.device(),
        Device::metal(),
        "RoPE output should stay on GPU"
    );
    assert_eq!(gpu_out.dims(), cpu_out.dims(), "shape mismatch");

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-5, "rope_apply_2d");
}

/// RoPE apply on GPU 4D tensor [batch, heads, seq, head_dim] matches CPU.
#[test]
fn test_gpu_rope_apply_4d() {
    init();
    let head_dim = 4;
    let max_seq = 32;
    let batch = 1;
    let heads = 2;
    let seq_len = 3;
    let base = 10000.0;

    let n = batch * heads * seq_len * head_dim;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();

    let cpu_rope = RotaryEmbedding::new(head_dim, max_seq, base, &Device::Cpu).unwrap();
    let cpu_x =
        DynTensor::from_vec(data, &[batch, heads, seq_len, head_dim], &Device::Cpu).unwrap();
    let cpu_out = cpu_rope.apply(&cpu_x, 0).unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();

    let gpu_rope = RotaryEmbedding::new(head_dim, max_seq, base, &Device::metal()).unwrap();
    let gpu_x = cpu_x.to_device(&Device::metal()).unwrap();
    let gpu_out = gpu_rope.apply(&gpu_x, 0).unwrap();
    assert_eq!(gpu_out.device(), Device::metal());

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-5, "rope_apply_4d");
}

/// RoPE apply with non-zero offset on GPU matches CPU.
#[test]
fn test_gpu_rope_apply_with_offset() {
    init();
    let head_dim = 4;
    let max_seq = 32;
    let seq_len = 2;
    let offset = 5;
    let base = 10000.0;

    let data: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32 + 1.0).collect();

    let cpu_rope = RotaryEmbedding::new(head_dim, max_seq, base, &Device::Cpu).unwrap();
    let cpu_x = DynTensor::from_vec(data, &[seq_len, head_dim], &Device::Cpu).unwrap();
    let cpu_out = cpu_rope.apply(&cpu_x, offset).unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();

    let gpu_rope = RotaryEmbedding::new(head_dim, max_seq, base, &Device::metal()).unwrap();
    let gpu_x = cpu_x.to_device(&Device::metal()).unwrap();
    let gpu_out = gpu_rope.apply(&gpu_x, offset).unwrap();
    assert_eq!(gpu_out.device(), Device::metal());

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-5, "rope_apply_offset");
}

// -- GPU RoPE apply_pair tests ------------------------------------------------

/// RoPE apply_pair on GPU matches CPU for both Q and K tensors.
#[test]
fn test_gpu_rope_apply_pair() {
    init();
    let head_dim = 4;
    let max_seq = 32;
    let batch = 1;
    let num_heads = 2;
    let num_kv_heads = 1;
    let seq_len = 3;
    let base = 10000.0;

    let q_data: Vec<f32> = (0..batch * num_heads * seq_len * head_dim)
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let k_data: Vec<f32> = (0..batch * num_kv_heads * seq_len * head_dim)
        .map(|i| (i as f32 + 1.0) * 0.2)
        .collect();
    let positions: Vec<usize> = (0..seq_len).collect();

    // CPU reference
    let cpu_rope = RotaryEmbedding::new(head_dim, max_seq, base, &Device::Cpu).unwrap();
    let cpu_q =
        DynTensor::from_vec(q_data, &[batch, num_heads, seq_len, head_dim], &Device::Cpu).unwrap();
    let cpu_k = DynTensor::from_vec(
        k_data,
        &[batch, num_kv_heads, seq_len, head_dim],
        &Device::Cpu,
    )
    .unwrap();
    let (cpu_q_out, cpu_k_out) = cpu_rope.apply_pair(&cpu_q, &cpu_k, &positions).unwrap();

    // GPU
    let gpu_rope = RotaryEmbedding::new(head_dim, max_seq, base, &Device::metal()).unwrap();
    let gpu_q = cpu_q.to_device(&Device::metal()).unwrap();
    let gpu_k = cpu_k.to_device(&Device::metal()).unwrap();
    let (gpu_q_out, gpu_k_out) = gpu_rope.apply_pair(&gpu_q, &gpu_k, &positions).unwrap();

    assert_eq!(gpu_q_out.device(), Device::metal(), "Q output device");
    assert_eq!(gpu_k_out.device(), Device::metal(), "K output device");

    let gpu_q_vals = gpu_q_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let gpu_k_vals = gpu_k_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_q_vals = cpu_q_out.to_flat_vec::<f32>().unwrap();
    let cpu_k_vals = cpu_k_out.to_flat_vec::<f32>().unwrap();

    assert_close(&gpu_q_vals, &cpu_q_vals, 1e-5, "rope_pair_q");
    assert_close(&gpu_k_vals, &cpu_k_vals, 1e-5, "rope_pair_k");
}

// -- GPU RoPE norm preservation -----------------------------------------------

/// RoPE preserves L2 norm of each position vector on GPU.
/// This is a fundamental property of rotary embeddings.
#[test]
fn test_gpu_rope_norm_preservation() {
    init();
    let head_dim = 8;
    let max_seq = 16;
    let seq_len = 4;
    let base = 10000.0;

    let data: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i as f32 + 1.0) * 0.5)
        .collect();

    let gpu_rope = RotaryEmbedding::new(head_dim, max_seq, base, &Device::metal()).unwrap();
    let gpu_x = DynTensor::from_vec(data, &[seq_len, head_dim], &Device::metal()).unwrap();
    let gpu_out = gpu_rope.apply(&gpu_x, 0).unwrap();
    assert_eq!(gpu_out.device(), Device::metal());

    // Transfer to CPU and compare norms per position
    let in_vals = gpu_x
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let out_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    for pos in 0..seq_len {
        let start = pos * head_dim;
        let end = start + head_dim;
        let in_norm: f32 = in_vals[start..end]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        let out_norm: f32 = out_vals[start..end]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        assert!(
            (in_norm - out_norm).abs() < 1e-4,
            "Position {pos}: input norm {in_norm} vs output norm {out_norm} differ by {}",
            (in_norm - out_norm).abs()
        );
    }
}

// -- GPU HalfRotaryEmbedding tests --------------------------------------------

/// HalfRotaryEmbedding on GPU matches CPU (rotates first half, passes second half).
#[test]
fn test_gpu_half_rope_apply() {
    init();
    let head_dim = 8; // must be multiple of 4 for HalfRoPE
    let max_seq = 16;
    let seq_len = 3;
    let base = 1_000_000.0; // Qwen3-style base

    let data: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i as f32 + 1.0) * 0.3)
        .collect();

    let cpu_half_rope =
        nn_core::layers::HalfRotaryEmbedding::new(head_dim, max_seq, base, &Device::Cpu).unwrap();
    let cpu_x = DynTensor::from_vec(data, &[seq_len, head_dim], &Device::Cpu).unwrap();
    let cpu_out = cpu_half_rope.apply(&cpu_x, 0).unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();

    let gpu_half_rope =
        nn_core::layers::HalfRotaryEmbedding::new(head_dim, max_seq, base, &Device::metal()).unwrap();
    let gpu_x = cpu_x.to_device(&Device::metal()).unwrap();
    let gpu_out = gpu_half_rope.apply(&gpu_x, 0).unwrap();
    assert_eq!(gpu_out.device(), Device::metal());

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-5, "half_rope_apply");
}

// -- GPU rope() free function tests -------------------------------------------

/// Free function rope(t, cos, sin) on GPU matches CPU.
#[test]
fn test_gpu_rope_free_fn() {
    init();
    let head_dim = 4;
    let seq_len = 3;
    let half = head_dim / 2;

    let t_data: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i as f32 + 1.0) * 0.2)
        .collect();
    // cos/sin for 3 positions, half_dim=2 each
    let cos_data: Vec<f32> = vec![1.0, 0.9999, 0.5403, 0.9999, -0.4161, 0.9998];
    let sin_data: Vec<f32> = vec![0.0, 0.01, 0.8415, 0.01, 0.9093, 0.02];

    // CPU
    let cpu_t = DynTensor::from_vec(t_data, &[seq_len, head_dim], &Device::Cpu).unwrap();
    let cpu_cos = DynTensor::from_vec(cos_data, &[seq_len, half], &Device::Cpu).unwrap();
    let cpu_sin = DynTensor::from_vec(sin_data, &[seq_len, half], &Device::Cpu).unwrap();
    let cpu_out = nn_core::layers::rope(&cpu_t, &cpu_cos, &cpu_sin).unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();

    // GPU
    let gpu_t = cpu_t.to_device(&Device::metal()).unwrap();
    let gpu_cos = cpu_cos.to_device(&Device::metal()).unwrap();
    let gpu_sin = cpu_sin.to_device(&Device::metal()).unwrap();
    let gpu_out = nn_core::layers::rope(&gpu_t, &gpu_cos, &gpu_sin).unwrap();
    assert_eq!(gpu_out.device(), Device::metal());

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-5, "rope_free_fn");
}

// -- GPU RoPE decode step test (dvoice pattern) ----------------------------

/// Single decode step: apply RoPE at offset to length-1 sequence on GPU.
/// This is the hot path in autoregressive decoding.
#[test]
fn test_gpu_rope_decode_step() {
    init();
    let head_dim = 8;
    let max_seq = 128;
    let batch = 1;
    let heads = 4;
    let seq_len = 1; // single token decode
    let offset = 42; // mid-sequence decode position
    let base = 10000.0;

    let n = batch * heads * seq_len * head_dim;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.05).collect();

    let cpu_rope = RotaryEmbedding::new(head_dim, max_seq, base, &Device::Cpu).unwrap();
    let cpu_x =
        DynTensor::from_vec(data, &[batch, heads, seq_len, head_dim], &Device::Cpu).unwrap();
    let cpu_out = cpu_rope.apply(&cpu_x, offset).unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();

    let gpu_rope = RotaryEmbedding::new(head_dim, max_seq, base, &Device::metal()).unwrap();
    let gpu_x = cpu_x.to_device(&Device::metal()).unwrap();
    let gpu_out = gpu_rope.apply(&gpu_x, offset).unwrap();
    assert_eq!(gpu_out.device(), Device::metal());

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-5, "rope_decode_step");
}
