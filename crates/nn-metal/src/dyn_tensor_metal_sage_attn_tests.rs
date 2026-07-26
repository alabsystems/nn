// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for SageAttention GPU dispatch.
//!
//! Tests verify the CPU fallback path produces correct results when
//! dispatched through the GPU interface (GPU → CPU → SageAttention → GPU).

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::attention::{SageAttention, SageAttentionConfig};
use nn_core::Device;

use super::SageAttentionGpu;

/// Initialize Metal backend for tests.
fn gpu_init() {
    let _ = crate::MetalBackend::init();
    crate::register_metal_dyn_backend();
}

/// Create random tensor data using a simple deterministic PRNG.
fn rand_f32_vec(seed: u64, count: usize, lo: f32, hi: f32) -> Vec<f32> {
    nn_core::test_prng::rand_f32_vec(seed, count, lo, hi)
}

/// Create random Q, K, V tensors on the given device.
fn random_qkv(
    seed_base: u64,
    batch: usize,
    heads: usize,
    s_q: usize,
    s_kv: usize,
    head_dim: usize,
    device: &Device,
) -> (DynTensor, DynTensor, DynTensor) {
    let q_data = rand_f32_vec(seed_base, batch * heads * s_q * head_dim, -0.5, 0.5);
    let k_data = rand_f32_vec(seed_base + 1, batch * heads * s_kv * head_dim, -0.5, 0.5);
    let v_data = rand_f32_vec(seed_base + 2, batch * heads * s_kv * head_dim, -0.5, 0.5);

    let q = DynTensor::from_vec(q_data, &[batch, heads, s_q, head_dim], device).unwrap();
    let k = DynTensor::from_vec(k_data, &[batch, heads, s_kv, head_dim], device).unwrap();
    let v = DynTensor::from_vec(v_data, &[batch, heads, s_kv, head_dim], device).unwrap();

    (q, k, v)
}

/// Create random Q, K, V with different head counts (GQA).
fn random_qkv_gqa(
    seed_base: u64,
    batch: usize,
    h_q: usize,
    h_kv: usize,
    s_q: usize,
    s_kv: usize,
    head_dim: usize,
    device: &Device,
) -> (DynTensor, DynTensor, DynTensor) {
    let q_data = rand_f32_vec(seed_base, batch * h_q * s_q * head_dim, -0.5, 0.5);
    let k_data = rand_f32_vec(seed_base + 1, batch * h_kv * s_kv * head_dim, -0.5, 0.5);
    let v_data = rand_f32_vec(seed_base + 2, batch * h_kv * s_kv * head_dim, -0.5, 0.5);

    let q = DynTensor::from_vec(q_data, &[batch, h_q, s_q, head_dim], device).unwrap();
    let k = DynTensor::from_vec(k_data, &[batch, h_kv, s_kv, head_dim], device).unwrap();
    let v = DynTensor::from_vec(v_data, &[batch, h_kv, s_kv, head_dim], device).unwrap();

    (q, k, v)
}

/// Compare two DynTensors element-wise within tolerance.
fn assert_close(a: &DynTensor, b: &DynTensor, tol: f32, label: &str) {
    let a_cpu = a.to_device(&Device::Cpu).unwrap();
    let b_cpu = b.to_device(&Device::Cpu).unwrap();
    let a_vals = a_cpu.to_flat_vec::<f32>().unwrap();
    let b_vals = b_cpu.to_flat_vec::<f32>().unwrap();
    assert_eq!(
        a_vals.len(),
        b_vals.len(),
        "{label}: length mismatch (a={}, b={})",
        a_vals.len(),
        b_vals.len()
    );
    for (i, (av, bv)) in a_vals.iter().zip(b_vals.iter()).enumerate() {
        let diff = (av - bv).abs();
        assert!(
            diff <= tol,
            "{label}[{i}]: a={av} b={bv} diff={diff} > {tol}"
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_sage_gpu_basic_forward() {
    gpu_init();
    let device = Device::metal();
    let config = SageAttentionConfig {
        head_dim: 64,
        num_heads: 4,
        num_kv_heads: None,
        causal: false,
        smooth_k: false,
    };
    let sage_gpu = SageAttentionGpu::new(config).unwrap();
    let (q, k, v) = random_qkv(42, 1, 4, 16, 16, 64, &device);

    let output = sage_gpu.forward(&q, &k, &v).unwrap();

    // Verify output shape: [B=1, H=4, S_q=16, D=64]
    assert_eq!(output.dims(), &[1, 4, 16, 64]);
    assert!(output.device().is_gpu(), "output should be on GPU");
}

#[test]
fn test_sage_gpu_matches_cpu() {
    gpu_init();
    let device = Device::metal();
    let config = SageAttentionConfig {
        head_dim: 64,
        num_heads: 4,
        num_kv_heads: None,
        causal: false,
        smooth_k: false,
    };
    let sage_gpu = SageAttentionGpu::new(config).unwrap();
    let sage_cpu = SageAttention::new(config).unwrap();

    let (q_gpu, k_gpu, v_gpu) = random_qkv(100, 1, 4, 16, 16, 64, &device);
    let q_cpu = q_gpu.to_device(&Device::Cpu).unwrap();
    let k_cpu = k_gpu.to_device(&Device::Cpu).unwrap();
    let v_cpu = v_gpu.to_device(&Device::Cpu).unwrap();

    let gpu_result = sage_gpu.forward(&q_gpu, &k_gpu, &v_gpu).unwrap();
    let cpu_result = sage_cpu.forward(&q_cpu, &k_cpu, &v_cpu).unwrap();

    // CPU fallback should produce identical results (exact match within
    // float round-trip tolerance from GPU → CPU → GPU transfers).
    assert_close(&gpu_result, &cpu_result, 1e-5, "sage_gpu_vs_cpu");
}

#[test]
fn test_sage_gpu_causal() {
    gpu_init();
    let device = Device::metal();
    let config = SageAttentionConfig {
        head_dim: 64,
        num_heads: 4,
        num_kv_heads: None,
        causal: true,
        smooth_k: false,
    };
    let sage_gpu = SageAttentionGpu::new(config).unwrap();
    let sage_cpu = SageAttention::new(config).unwrap();

    let (q_gpu, k_gpu, v_gpu) = random_qkv(200, 1, 4, 32, 32, 64, &device);
    let q_cpu = q_gpu.to_device(&Device::Cpu).unwrap();
    let k_cpu = k_gpu.to_device(&Device::Cpu).unwrap();
    let v_cpu = v_gpu.to_device(&Device::Cpu).unwrap();

    let gpu_result = sage_gpu.forward(&q_gpu, &k_gpu, &v_gpu).unwrap();
    let cpu_result = sage_cpu.forward(&q_cpu, &k_cpu, &v_cpu).unwrap();

    assert_eq!(gpu_result.dims(), &[1, 4, 32, 64]);
    assert_close(&gpu_result, &cpu_result, 1e-5, "sage_gpu_causal");
}

#[test]
fn test_sage_gpu_gqa() {
    gpu_init();
    let device = Device::metal();
    // 8 query heads, 2 KV heads → group size 4
    let config = SageAttentionConfig {
        head_dim: 64,
        num_heads: 8,
        num_kv_heads: Some(2),
        causal: false,
        smooth_k: false,
    };
    let sage_gpu = SageAttentionGpu::new(config).unwrap();
    let sage_cpu = SageAttention::new(config).unwrap();

    let (q_gpu, k_gpu, v_gpu) = random_qkv_gqa(300, 1, 8, 2, 16, 16, 64, &device);
    let q_cpu = q_gpu.to_device(&Device::Cpu).unwrap();
    let k_cpu = k_gpu.to_device(&Device::Cpu).unwrap();
    let v_cpu = v_gpu.to_device(&Device::Cpu).unwrap();

    let gpu_result = sage_gpu.forward(&q_gpu, &k_gpu, &v_gpu).unwrap();
    let cpu_result = sage_cpu.forward(&q_cpu, &k_cpu, &v_cpu).unwrap();

    assert_eq!(gpu_result.dims(), &[1, 8, 16, 64]);
    assert_close(&gpu_result, &cpu_result, 1e-5, "sage_gpu_gqa");
}

#[test]
fn test_sage_gpu_long_sequence() {
    gpu_init();
    let device = Device::metal();
    let config = SageAttentionConfig {
        head_dim: 64,
        num_heads: 4,
        num_kv_heads: None,
        causal: false,
        smooth_k: false,
    };
    let sage_gpu = SageAttentionGpu::new(config).unwrap();
    let sage_cpu = SageAttention::new(config).unwrap();

    // seq_len=256: realistic for document VLM patch token sequences
    let (q_gpu, k_gpu, v_gpu) = random_qkv(400, 1, 4, 256, 256, 64, &device);
    let q_cpu = q_gpu.to_device(&Device::Cpu).unwrap();
    let k_cpu = k_gpu.to_device(&Device::Cpu).unwrap();
    let v_cpu = v_gpu.to_device(&Device::Cpu).unwrap();

    let gpu_result = sage_gpu.forward(&q_gpu, &k_gpu, &v_gpu).unwrap();
    let cpu_result = sage_cpu.forward(&q_cpu, &k_cpu, &v_cpu).unwrap();

    assert_eq!(gpu_result.dims(), &[1, 4, 256, 64]);
    assert_close(&gpu_result, &cpu_result, 1e-4, "sage_gpu_long_seq");
}

#[test]
fn test_sage_gpu_batch() {
    gpu_init();
    let device = Device::metal();
    let config = SageAttentionConfig {
        head_dim: 64,
        num_heads: 4,
        num_kv_heads: None,
        causal: false,
        smooth_k: false,
    };
    let sage_gpu = SageAttentionGpu::new(config).unwrap();
    let sage_cpu = SageAttention::new(config).unwrap();

    // batch_size=3
    let (q_gpu, k_gpu, v_gpu) = random_qkv(500, 3, 4, 16, 16, 64, &device);
    let q_cpu = q_gpu.to_device(&Device::Cpu).unwrap();
    let k_cpu = k_gpu.to_device(&Device::Cpu).unwrap();
    let v_cpu = v_gpu.to_device(&Device::Cpu).unwrap();

    let gpu_result = sage_gpu.forward(&q_gpu, &k_gpu, &v_gpu).unwrap();
    let cpu_result = sage_cpu.forward(&q_cpu, &k_cpu, &v_cpu).unwrap();

    assert_eq!(gpu_result.dims(), &[3, 4, 16, 64]);
    assert_close(&gpu_result, &cpu_result, 1e-5, "sage_gpu_batch");
}

#[test]
fn test_sage_gpu_smooth_k() {
    gpu_init();
    let device = Device::metal();
    let config = SageAttentionConfig {
        head_dim: 64,
        num_heads: 4,
        num_kv_heads: None,
        causal: false,
        smooth_k: true,
    };
    let sage_gpu = SageAttentionGpu::new(config).unwrap();
    let sage_cpu = SageAttention::new(config).unwrap();

    let (q_gpu, k_gpu, v_gpu) = random_qkv(600, 1, 4, 32, 32, 64, &device);
    let q_cpu = q_gpu.to_device(&Device::Cpu).unwrap();
    let k_cpu = k_gpu.to_device(&Device::Cpu).unwrap();
    let v_cpu = v_gpu.to_device(&Device::Cpu).unwrap();

    let gpu_result = sage_gpu.forward(&q_gpu, &k_gpu, &v_gpu).unwrap();
    let cpu_result = sage_cpu.forward(&q_cpu, &k_cpu, &v_cpu).unwrap();

    assert_eq!(gpu_result.dims(), &[1, 4, 32, 64]);
    assert_close(&gpu_result, &cpu_result, 1e-5, "sage_gpu_smooth_k");
}

#[test]
fn test_sage_gpu_dpdf_config() {
    gpu_init();
    let device = Device::metal();
    // Qwen3-VL-2B config: head_dim=128, 12 heads, 2 KV heads (GQA 6:1)
    let config = SageAttentionConfig {
        head_dim: 128,
        num_heads: 12,
        num_kv_heads: Some(2),
        causal: false,
        smooth_k: true,
    };
    let sage_gpu = SageAttentionGpu::new(config).unwrap();
    let sage_cpu = SageAttention::new(config).unwrap();

    // Realistic dpdf patch token count: 196 (14x14 patches from 224x224 image)
    let (q_gpu, k_gpu, v_gpu) = random_qkv_gqa(700, 1, 12, 2, 196, 196, 128, &device);
    let q_cpu = q_gpu.to_device(&Device::Cpu).unwrap();
    let k_cpu = k_gpu.to_device(&Device::Cpu).unwrap();
    let v_cpu = v_gpu.to_device(&Device::Cpu).unwrap();

    let gpu_result = sage_gpu.forward(&q_gpu, &k_gpu, &v_gpu).unwrap();
    let cpu_result = sage_cpu.forward(&q_cpu, &k_cpu, &v_cpu).unwrap();

    assert_eq!(gpu_result.dims(), &[1, 12, 196, 128]);
    assert_close(&gpu_result, &cpu_result, 1e-4, "sage_gpu_dpdf");
}
