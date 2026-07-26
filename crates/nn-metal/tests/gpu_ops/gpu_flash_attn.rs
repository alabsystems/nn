// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Differential test: Flash Attention GPU kernel vs decomposed SDPA.
//!
//! Verifies that the fused Flash Attention Metal kernel produces results
//! matching the decomposed SDPA path (Q@K^T*scale → softmax → @V) across
//! a range of shapes relevant to dpdf document understanding models.
//!
//! Tolerance is higher than exact match because online softmax accumulation
//! order differs from standard softmax (FA2 paper Section 3.1).
//!
//! Issue: #2434, Part of #2218.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::attention::{causal_mask, repeat_kv, sdpa, sdpa_causal};
use nn_core::{DType, Device};

/// Decomposed CPU SDPA as reference implementation.
fn cpu_sdpa(q: &DynTensor, k: &DynTensor, v: &DynTensor, scale: f64) -> DynTensor {
    let k_t = k.transpose(2, 3).unwrap();
    let scores = q.matmul(&k_t).unwrap().mul_scalar(scale).unwrap();
    let attn_weights = scores.softmax(scores.rank() - 1).unwrap();
    attn_weights.matmul(v).unwrap()
}

/// Decomposed CPU SDPA with causal mask as reference implementation.
fn cpu_sdpa_causal(q: &DynTensor, k: &DynTensor, v: &DynTensor, scale: f64) -> DynTensor {
    let seq_len = q.dims()[2];
    let mask = causal_mask(seq_len, &Device::Cpu).unwrap();
    let k_t = k.transpose(2, 3).unwrap();
    let scores = q.matmul(&k_t).unwrap().mul_scalar(scale).unwrap();
    let scores = scores.broadcast_add(&mask).unwrap();
    let attn_weights = scores.softmax(scores.rank() - 1).unwrap();
    attn_weights.matmul(v).unwrap()
}

/// Helper: create random Q, K, V tensors on the given device.
fn random_qkv(
    seed_base: u64,
    batch: usize,
    heads: usize,
    s_q: usize,
    s_kv: usize,
    head_dim: usize,
    device: &Device,
) -> (DynTensor, DynTensor, DynTensor) {
    let q_data =
        super::test_utils::rand_f32_vec(seed_base, batch * heads * s_q * head_dim, -1.0, 1.0);
    let k_data =
        super::test_utils::rand_f32_vec(seed_base + 1, batch * heads * s_kv * head_dim, -1.0, 1.0);
    let v_data =
        super::test_utils::rand_f32_vec(seed_base + 2, batch * heads * s_kv * head_dim, -1.0, 1.0);

    let q = DynTensor::from_vec(q_data, &[batch, heads, s_q, head_dim], device).unwrap();
    let k = DynTensor::from_vec(k_data, &[batch, heads, s_kv, head_dim], device).unwrap();
    let v = DynTensor::from_vec(v_data, &[batch, heads, s_kv, head_dim], device).unwrap();

    (q, k, v)
}

/// Helper: create random Q, K, V with different head counts (GQA).
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
    let q_data =
        super::test_utils::rand_f32_vec(seed_base, batch * h_q * s_q * head_dim, -1.0, 1.0);
    let k_data =
        super::test_utils::rand_f32_vec(seed_base + 1, batch * h_kv * s_kv * head_dim, -1.0, 1.0);
    let v_data =
        super::test_utils::rand_f32_vec(seed_base + 2, batch * h_kv * s_kv * head_dim, -1.0, 1.0);

    let q = DynTensor::from_vec(q_data, &[batch, h_q, s_q, head_dim], device).unwrap();
    let k = DynTensor::from_vec(k_data, &[batch, h_kv, s_kv, head_dim], device).unwrap();
    let v = DynTensor::from_vec(v_data, &[batch, h_kv, s_kv, head_dim], device).unwrap();

    (q, k, v)
}

/// Core parity check: GPU Flash Attention vs CPU decomposed SDPA.
fn check_flash_attn_parity(
    batch: usize,
    heads: usize,
    s_q: usize,
    s_kv: usize,
    head_dim: usize,
    tol: f32,
) {
    super::test_utils::gpu_init();

    let scale = 1.0 / (head_dim as f64).sqrt();
    let label = format!("B={batch} H={heads} Sq={s_q} Skv={s_kv} D={head_dim}");

    // CPU reference: use the decomposed path directly.
    let (q_cpu, k_cpu, v_cpu) = random_qkv(42, batch, heads, s_q, s_kv, head_dim, &Device::Cpu);
    let cpu_result = cpu_sdpa(&q_cpu, &k_cpu, &v_cpu, scale);

    // GPU path: the sdpa() function should route to Flash Attention.
    let q_gpu = q_cpu.to_device(&Device::metal()).unwrap();
    let k_gpu = k_cpu.to_device(&Device::metal()).unwrap();
    let v_gpu = v_cpu.to_device(&Device::metal()).unwrap();
    let gpu_result = sdpa(&q_gpu, &k_gpu, &v_gpu, None, scale).unwrap();

    compare_results(&gpu_result, &cpu_result, tol, &label);
}

/// Core parity check for GQA: GPU Flash Attention vs CPU repeat_kv + decomposed.
fn check_flash_attn_gqa_parity(
    batch: usize,
    h_q: usize,
    h_kv: usize,
    seq: usize,
    head_dim: usize,
    tol: f32,
) {
    super::test_utils::gpu_init();

    let scale = 1.0 / (head_dim as f64).sqrt();
    let label = format!("GQA B={batch} Hq={h_q} Hkv={h_kv} S={seq} D={head_dim}");

    // CPU reference: expand K/V heads via repeat_kv, then standard SDPA.
    let (q_cpu, k_cpu, v_cpu) =
        random_qkv_gqa(42, batch, h_q, h_kv, seq, seq, head_dim, &Device::Cpu);
    let num_rep = h_q / h_kv;
    let k_expanded = repeat_kv(&k_cpu, num_rep).unwrap();
    let v_expanded = repeat_kv(&v_cpu, num_rep).unwrap();
    let cpu_result = cpu_sdpa(&q_cpu, &k_expanded, &v_expanded, scale);

    // GPU path: Flash Attention handles GQA natively (no repeat_kv needed).
    let q_gpu = q_cpu.to_device(&Device::metal()).unwrap();
    let k_gpu = k_cpu.to_device(&Device::metal()).unwrap();
    let v_gpu = v_cpu.to_device(&Device::metal()).unwrap();
    let gpu_result = sdpa(&q_gpu, &k_gpu, &v_gpu, None, scale).unwrap();

    compare_results(&gpu_result, &cpu_result, tol, &label);
}

/// Core parity check for causal attention: GPU fused causal vs CPU mask + SDPA.
fn check_flash_attn_causal_parity(
    batch: usize,
    heads: usize,
    seq: usize,
    head_dim: usize,
    tol: f32,
) {
    super::test_utils::gpu_init();

    let scale = 1.0 / (head_dim as f64).sqrt();
    let label = format!("causal B={batch} H={heads} S={seq} D={head_dim}");

    // CPU reference: explicit causal mask + decomposed path.
    let (q_cpu, k_cpu, v_cpu) = random_qkv(42, batch, heads, seq, seq, head_dim, &Device::Cpu);
    let cpu_result = cpu_sdpa_causal(&q_cpu, &k_cpu, &v_cpu, scale);

    // GPU path: fused causal Flash Attention via sdpa_causal().
    let q_gpu = q_cpu.to_device(&Device::metal()).unwrap();
    let k_gpu = k_cpu.to_device(&Device::metal()).unwrap();
    let v_gpu = v_cpu.to_device(&Device::metal()).unwrap();
    let gpu_result = sdpa_causal(&q_gpu, &k_gpu, &v_gpu, scale).unwrap();

    compare_results(&gpu_result, &cpu_result, tol, &label);
}

/// Core parity check for GQA + causal combined.
fn check_flash_attn_gqa_causal_parity(
    batch: usize,
    h_q: usize,
    h_kv: usize,
    seq: usize,
    head_dim: usize,
    tol: f32,
) {
    super::test_utils::gpu_init();

    let scale = 1.0 / (head_dim as f64).sqrt();
    let label = format!("GQA+causal B={batch} Hq={h_q} Hkv={h_kv} S={seq} D={head_dim}");

    // CPU reference: expand K/V heads + causal mask + decomposed.
    let (q_cpu, k_cpu, v_cpu) =
        random_qkv_gqa(42, batch, h_q, h_kv, seq, seq, head_dim, &Device::Cpu);
    let num_rep = h_q / h_kv;
    let k_expanded = repeat_kv(&k_cpu, num_rep).unwrap();
    let v_expanded = repeat_kv(&v_cpu, num_rep).unwrap();
    let cpu_result = cpu_sdpa_causal(&q_cpu, &k_expanded, &v_expanded, scale);

    // GPU path: Flash Attention handles GQA + causal natively.
    let q_gpu = q_cpu.to_device(&Device::metal()).unwrap();
    let k_gpu = k_cpu.to_device(&Device::metal()).unwrap();
    let v_gpu = v_cpu.to_device(&Device::metal()).unwrap();
    let gpu_result = sdpa_causal(&q_gpu, &k_gpu, &v_gpu, scale).unwrap();

    compare_results(&gpu_result, &cpu_result, tol, &label);
}

/// Shared comparison logic.
fn compare_results(gpu_result: &DynTensor, cpu_result: &DynTensor, tol: f32, label: &str) {
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    assert_eq!(gpu_vals.len(), cpu_vals.len(), "{label}: length mismatch");

    let max_diff = gpu_vals
        .iter()
        .zip(cpu_vals.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0_f32, f32::max);

    let mean_diff: f32 = gpu_vals
        .iter()
        .zip(cpu_vals.iter())
        .map(|(g, c)| (g - c).abs())
        .sum::<f32>()
        / gpu_vals.len() as f32;

    eprintln!(
        "flash_attn parity [{label}]: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, \
         elements={}",
        gpu_vals.len()
    );

    assert!(
        max_diff < tol,
        "{label}: max_diff={max_diff:.6e} exceeds tol={tol:.6e}"
    );
}

// ==========================================================================
// Standard MHA tests (existing Wave 1)
// ==========================================================================

/// Small: batch=1, heads=1, seq=32, head_dim=32.
#[test]
fn flash_attn_b1_h1_s32_d32() {
    check_flash_attn_parity(1, 1, 32, 32, 32, 1e-3);
}

/// Medium: batch=2, heads=8, seq=64, head_dim=64.
#[test]
fn flash_attn_b2_h8_s64_d64() {
    check_flash_attn_parity(2, 8, 64, 64, 64, 1e-3);
}

/// Large head_dim: batch=1, heads=4, seq=64, head_dim=128.
#[test]
fn flash_attn_b1_h4_s64_d128() {
    check_flash_attn_parity(1, 4, 64, 64, 128, 1e-3);
}

/// Cross-attention: S_q != S_kv.
#[test]
fn flash_attn_cross_attention() {
    check_flash_attn_parity(1, 4, 32, 128, 64, 1e-3);
}

/// Larger seq: batch=1, heads=4, seq=128, head_dim=64.
#[test]
fn flash_attn_b1_h4_s128_d64() {
    check_flash_attn_parity(1, 4, 128, 128, 64, 1e-3);
}

/// Non-power-of-2 sequence: batch=1, heads=2, seq=100, head_dim=64.
#[test]
fn flash_attn_non_power_of_2_seq() {
    check_flash_attn_parity(1, 2, 100, 100, 64, 1e-3);
}

/// Single Q row (like autoregressive decoding): S_q=1, S_kv=64.
#[test]
fn flash_attn_single_query() {
    check_flash_attn_parity(1, 8, 1, 64, 64, 1e-3);
}

/// Verify the GPU path is actually used (not silently falling back to CPU).
#[test]
fn flash_attn_routes_to_gpu() {
    super::test_utils::gpu_init();

    let (q, k, v) = random_qkv(123, 1, 4, 32, 32, 64, &Device::metal());
    let scale = 1.0 / 64.0_f64.sqrt();

    let result = sdpa(&q, &k, &v, None, scale).unwrap();

    assert!(
        result.device().is_gpu(),
        "sdpa result should be on GPU (Flash Attention), got {:?}",
        result.device()
    );
}

// ==========================================================================
// GQA tests (Wave 3)
// ==========================================================================

/// GQA with group_size=4: 8 query heads, 2 KV heads.
#[test]
fn flash_attn_gqa_h8_hkv2() {
    check_flash_attn_gqa_parity(1, 8, 2, 64, 64, 1e-3);
}

/// GQA with group_size=2: 8 query heads, 4 KV heads.
#[test]
fn flash_attn_gqa_h8_hkv4() {
    check_flash_attn_gqa_parity(1, 8, 4, 64, 64, 1e-3);
}

/// MQA (multi-query attention): 8 query heads, 1 KV head.
#[test]
fn flash_attn_mqa_h8_hkv1() {
    check_flash_attn_gqa_parity(1, 8, 1, 64, 64, 1e-3);
}

/// GQA with larger batch and head_dim: B=2, Hq=16, Hkv=4, D=128.
#[test]
fn flash_attn_gqa_b2_h16_hkv4_d128() {
    check_flash_attn_gqa_parity(2, 16, 4, 64, 128, 1e-3);
}

/// GQA routes to GPU (output on Metal device).
#[test]
fn flash_attn_gqa_routes_to_gpu() {
    super::test_utils::gpu_init();

    let (q, k, v) = random_qkv_gqa(123, 1, 8, 2, 32, 32, 64, &Device::metal());
    let scale = 1.0 / 64.0_f64.sqrt();

    let result = sdpa(&q, &k, &v, None, scale).unwrap();

    assert!(
        result.device().is_gpu(),
        "GQA sdpa result should be on GPU, got {:?}",
        result.device()
    );
    assert_eq!(
        result.dims(),
        &[1, 8, 32, 64],
        "GQA output should have H_q heads"
    );
}

// ==========================================================================
// Causal masking tests (Wave 3)
// ==========================================================================

/// Causal: small, batch=1, heads=4, seq=32, head_dim=64.
#[test]
fn flash_attn_causal_b1_h4_s32_d64() {
    check_flash_attn_causal_parity(1, 4, 32, 64, 1e-3);
}

/// Causal: medium, batch=2, heads=8, seq=64, head_dim=64.
#[test]
fn flash_attn_causal_b2_h8_s64_d64() {
    check_flash_attn_causal_parity(2, 8, 64, 64, 1e-3);
}

/// Causal: larger seq, batch=1, heads=4, seq=128, head_dim=64.
#[test]
fn flash_attn_causal_b1_h4_s128_d64() {
    check_flash_attn_causal_parity(1, 4, 128, 64, 1e-3);
}

/// Causal: large head_dim, batch=1, heads=4, seq=64, head_dim=128.
#[test]
fn flash_attn_causal_b1_h4_s64_d128() {
    check_flash_attn_causal_parity(1, 4, 64, 128, 1e-3);
}

/// Causal: non-power-of-2 seq, batch=1, heads=2, seq=100, head_dim=64.
#[test]
fn flash_attn_causal_non_pow2() {
    check_flash_attn_causal_parity(1, 2, 100, 64, 1e-3);
}

/// Causal: verify output is on GPU.
#[test]
fn flash_attn_causal_routes_to_gpu() {
    super::test_utils::gpu_init();

    let (q, k, v) = random_qkv(123, 1, 4, 64, 64, 64, &Device::metal());
    let scale = 1.0 / 64.0_f64.sqrt();

    let result = sdpa_causal(&q, &k, &v, scale).unwrap();

    assert!(
        result.device().is_gpu(),
        "sdpa_causal result should be on GPU, got {:?}",
        result.device()
    );
}

// ==========================================================================
// GQA + Causal combined tests (Wave 3)
// ==========================================================================

/// GQA + causal: Hq=8, Hkv=2, seq=64 — like Qwen3-VL autoregressive.
#[test]
fn flash_attn_gqa_causal_h8_hkv2_s64() {
    check_flash_attn_gqa_causal_parity(1, 8, 2, 64, 64, 1e-3);
}

/// GQA + causal: larger config B=2, Hq=16, Hkv=4, seq=128, D=128.
#[test]
fn flash_attn_gqa_causal_b2_h16_hkv4_s128_d128() {
    check_flash_attn_gqa_causal_parity(2, 16, 4, 128, 128, 1e-3);
}

/// MQA + causal: Hq=8, Hkv=1, seq=64.
#[test]
fn flash_attn_mqa_causal_h8_hkv1_s64() {
    check_flash_attn_gqa_causal_parity(1, 8, 1, 64, 64, 1e-3);
}

// ==========================================================================
// Half-precision tests (Wave 3 — F16/BF16 support)
// ==========================================================================

/// Core parity check: GPU Flash Attention (half) vs CPU decomposed SDPA (f32).
///
/// Creates F32 reference tensors, casts to the target dtype for GPU dispatch,
/// then compares GPU (half → f32 readback) against CPU (f32) reference.
/// Tolerance is higher than f32 tests to account for half-precision rounding.
fn check_flash_attn_half_parity(
    batch: usize,
    heads: usize,
    s_q: usize,
    s_kv: usize,
    head_dim: usize,
    dtype: DType,
    tol: f32,
) {
    super::test_utils::gpu_init();

    let scale = 1.0 / (head_dim as f64).sqrt();
    let label = format!("{dtype:?} B={batch} H={heads} Sq={s_q} Skv={s_kv} D={head_dim}");

    // CPU reference in f32 (gold standard).
    let (q_cpu, k_cpu, v_cpu) = random_qkv(42, batch, heads, s_q, s_kv, head_dim, &Device::Cpu);
    let cpu_result = cpu_sdpa(&q_cpu, &k_cpu, &v_cpu, scale);

    // Cast to target dtype and move to GPU.
    let q_half = q_cpu
        .to_dtype(dtype)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let k_half = k_cpu
        .to_dtype(dtype)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let v_half = v_cpu
        .to_dtype(dtype)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let gpu_result = sdpa(&q_half, &k_half, &v_half, None, scale).unwrap();

    // Verify output dtype matches input.
    assert_eq!(
        gpu_result.dtype(),
        dtype,
        "{label}: output dtype should match input"
    );

    // Cast GPU result back to f32 for comparison.
    let gpu_f32 = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    compare_results(&gpu_f32, &cpu_result, tol, &label);
}

/// Half-precision GQA parity check.
fn check_flash_attn_gqa_half_parity(
    batch: usize,
    h_q: usize,
    h_kv: usize,
    seq: usize,
    head_dim: usize,
    dtype: DType,
    tol: f32,
) {
    super::test_utils::gpu_init();

    let scale = 1.0 / (head_dim as f64).sqrt();
    let label = format!("{dtype:?} GQA B={batch} Hq={h_q} Hkv={h_kv} S={seq} D={head_dim}");

    let (q_cpu, k_cpu, v_cpu) =
        random_qkv_gqa(42, batch, h_q, h_kv, seq, seq, head_dim, &Device::Cpu);
    let num_rep = h_q / h_kv;
    let k_expanded = repeat_kv(&k_cpu, num_rep).unwrap();
    let v_expanded = repeat_kv(&v_cpu, num_rep).unwrap();
    let cpu_result = cpu_sdpa(&q_cpu, &k_expanded, &v_expanded, scale);

    let q_half = q_cpu
        .to_dtype(dtype)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let k_half = k_cpu
        .to_dtype(dtype)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let v_half = v_cpu
        .to_dtype(dtype)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let gpu_result = sdpa(&q_half, &k_half, &v_half, None, scale).unwrap();

    assert_eq!(gpu_result.dtype(), dtype, "{label}: output dtype mismatch");

    let gpu_f32 = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    compare_results(&gpu_f32, &cpu_result, tol, &label);
}

/// Half-precision causal parity check.
fn check_flash_attn_causal_half_parity(
    batch: usize,
    heads: usize,
    seq: usize,
    head_dim: usize,
    dtype: DType,
    tol: f32,
) {
    super::test_utils::gpu_init();

    let scale = 1.0 / (head_dim as f64).sqrt();
    let label = format!("{dtype:?} causal B={batch} H={heads} S={seq} D={head_dim}");

    let (q_cpu, k_cpu, v_cpu) = random_qkv(42, batch, heads, seq, seq, head_dim, &Device::Cpu);
    let cpu_result = cpu_sdpa_causal(&q_cpu, &k_cpu, &v_cpu, scale);

    let q_half = q_cpu
        .to_dtype(dtype)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let k_half = k_cpu
        .to_dtype(dtype)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let v_half = v_cpu
        .to_dtype(dtype)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let gpu_result = sdpa_causal(&q_half, &k_half, &v_half, scale).unwrap();

    assert_eq!(gpu_result.dtype(), dtype, "{label}: output dtype mismatch");

    let gpu_f32 = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    compare_results(&gpu_f32, &cpu_result, tol, &label);
}

// -- F16 tests ----------------------------------------------------------------

/// F16: standard MHA, B=1, H=4, S=64, D=64.
#[test]
fn flash_attn_f16_b1_h4_s64_d64() {
    check_flash_attn_half_parity(1, 4, 64, 64, 64, DType::F16, 5e-2);
}

/// F16: larger config B=2, H=8, S=64, D=128.
#[test]
fn flash_attn_f16_b2_h8_s64_d128() {
    check_flash_attn_half_parity(2, 8, 64, 64, 128, DType::F16, 5e-2);
}

/// F16 + GQA: Hq=8, Hkv=2, S=64, D=64.
#[test]
fn flash_attn_f16_gqa_h8_hkv2() {
    check_flash_attn_gqa_half_parity(1, 8, 2, 64, 64, DType::F16, 5e-2);
}

/// F16 + causal: B=1, H=4, S=64, D=64.
#[test]
fn flash_attn_f16_causal_b1_h4_s64_d64() {
    check_flash_attn_causal_half_parity(1, 4, 64, 64, DType::F16, 5e-2);
}

/// F16: output dtype is F16 (not upcast to F32).
#[test]
fn flash_attn_f16_preserves_dtype() {
    super::test_utils::gpu_init();

    let (q, k, v) = random_qkv(42, 1, 4, 32, 32, 64, &Device::Cpu);
    let q_f16 = q
        .to_dtype(DType::F16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let k_f16 = k
        .to_dtype(DType::F16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let v_f16 = v
        .to_dtype(DType::F16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let scale = 1.0 / 64.0_f64.sqrt();

    let result = sdpa(&q_f16, &k_f16, &v_f16, None, scale).unwrap();

    assert_eq!(
        result.dtype(),
        DType::F16,
        "F16 Flash Attention should produce F16 output"
    );
    assert!(result.device().is_gpu(), "result should be on GPU");
}

// -- BF16 tests ---------------------------------------------------------------

/// BF16: standard MHA, B=1, H=4, S=64, D=64.
#[test]
fn flash_attn_bf16_b1_h4_s64_d64() {
    check_flash_attn_half_parity(1, 4, 64, 64, 64, DType::BF16, 5e-2);
}

/// BF16 + GQA: Hq=8, Hkv=2, S=64, D=64.
#[test]
fn flash_attn_bf16_gqa_h8_hkv2() {
    check_flash_attn_gqa_half_parity(1, 8, 2, 64, 64, DType::BF16, 5e-2);
}

/// BF16 + causal: B=1, H=4, S=64, D=64.
#[test]
fn flash_attn_bf16_causal_b1_h4_s64_d64() {
    check_flash_attn_causal_half_parity(1, 4, 64, 64, DType::BF16, 5e-2);
}

/// BF16: output dtype is BF16 (preserved through dispatch).
#[test]
fn flash_attn_bf16_preserves_dtype() {
    super::test_utils::gpu_init();

    let (q, k, v) = random_qkv(42, 1, 4, 32, 32, 64, &Device::Cpu);
    let q_bf16 = q
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let k_bf16 = k
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let v_bf16 = v
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let scale = 1.0 / 64.0_f64.sqrt();

    let result = sdpa(&q_bf16, &k_bf16, &v_bf16, None, scale).unwrap();

    assert_eq!(
        result.dtype(),
        DType::BF16,
        "BF16 Flash Attention should produce BF16 output"
    );
    assert!(result.device().is_gpu(), "result should be on GPU");
}

// ==========================================================================
// Edge case tests: non-power-of-2 head dimensions (Wave 4)
// ==========================================================================

/// Non-power-of-2 head_dim: D=48.
#[test]
fn flash_attn_head_dim_48() {
    check_flash_attn_parity(1, 4, 64, 64, 48, 1e-3);
}

/// Non-power-of-2 head_dim: D=80 (GPT-J style).
#[test]
fn flash_attn_head_dim_80() {
    check_flash_attn_parity(1, 4, 64, 64, 80, 1e-3);
}

/// Non-power-of-2 head_dim: D=96 (GPT-2 medium style).
#[test]
fn flash_attn_head_dim_96() {
    check_flash_attn_parity(2, 8, 64, 64, 96, 1e-3);
}

/// Non-power-of-2 head_dim with causal: D=48.
#[test]
fn flash_attn_causal_head_dim_48() {
    check_flash_attn_causal_parity(1, 4, 64, 48, 1e-3);
}

/// Non-power-of-2 head_dim with causal: D=96.
#[test]
fn flash_attn_causal_head_dim_96() {
    check_flash_attn_causal_parity(1, 8, 64, 96, 1e-3);
}

/// Non-power-of-2 head_dim with GQA: D=80, Hq=8, Hkv=2.
#[test]
fn flash_attn_gqa_head_dim_80() {
    check_flash_attn_gqa_parity(1, 8, 2, 64, 80, 1e-3);
}

/// Non-power-of-2 head_dim with GQA+causal: D=96, Hq=16, Hkv=4.
#[test]
fn flash_attn_gqa_causal_head_dim_96() {
    check_flash_attn_gqa_causal_parity(1, 16, 4, 64, 96, 1e-3);
}

/// Minimum head_dim: D=1 (extreme edge case).
#[test]
fn flash_attn_head_dim_1() {
    check_flash_attn_parity(1, 2, 32, 32, 1, 1e-3);
}

/// Larger sequence: S=256, D=64 (stresses tiling with 8 K/V blocks).
#[test]
fn flash_attn_large_seq_256() {
    check_flash_attn_parity(1, 4, 256, 256, 64, 1e-3);
}

/// Larger sequence causal: S=256, D=64.
#[test]
fn flash_attn_causal_large_seq_256() {
    check_flash_attn_causal_parity(1, 4, 256, 64, 1e-3);
}

/// Single query + causal (autoregressive decode step with mask).
/// S_q != S_kv not supported for causal, so this uses S_q=S_kv=1.
#[test]
fn flash_attn_causal_single_token() {
    check_flash_attn_causal_parity(1, 8, 1, 64, 1e-3);
}

/// D=256 (exceeds Flash Attention limit of 128): should silently fall back
/// to decomposed SDPA on GPU and still produce correct results.
#[test]
fn flash_attn_d256_fallback_to_decomposed() {
    super::test_utils::gpu_init();

    let batch = 1;
    let heads = 2;
    let seq = 32;
    let head_dim = 256;
    let scale = 1.0 / (head_dim as f64).sqrt();

    let (q_cpu, k_cpu, v_cpu) = random_qkv(42, batch, heads, seq, seq, head_dim, &Device::Cpu);
    let cpu_result = cpu_sdpa(&q_cpu, &k_cpu, &v_cpu, scale);

    // GPU path: should fall back to decomposed (D>128).
    let q_gpu = q_cpu.to_device(&Device::metal()).unwrap();
    let k_gpu = k_cpu.to_device(&Device::metal()).unwrap();
    let v_gpu = v_cpu.to_device(&Device::metal()).unwrap();
    let gpu_result = sdpa(&q_gpu, &k_gpu, &v_gpu, None, scale).unwrap();

    compare_results(
        &gpu_result.to_device(&Device::Cpu).unwrap(),
        &cpu_result,
        1e-3,
        "D=256 fallback",
    );
}

// ==========================================================================
// Performance comparison: Flash Attention vs decomposed SDPA
// ==========================================================================

/// Benchmark: measure wall-clock speedup of Flash Attention over decomposed.
///
/// Runs both paths multiple times for a representative shape (B=1, H=8, S=128, D=64)
/// and reports timing. This is a differential benchmark, not a hard pass/fail —
/// the key metric is the speedup ratio.
#[test]
fn flash_attn_performance_vs_decomposed() {
    super::test_utils::gpu_init();

    let batch = 1;
    let heads = 8;
    let seq = 128;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let warmup = 3;
    let iters = 10;

    // Create GPU tensors.
    let (q_cpu, k_cpu, v_cpu) = random_qkv(42, batch, heads, seq, seq, head_dim, &Device::Cpu);
    let q_gpu = q_cpu.to_device(&Device::metal()).unwrap();
    let k_gpu = k_cpu.to_device(&Device::metal()).unwrap();
    let v_gpu = v_cpu.to_device(&Device::metal()).unwrap();

    // Warmup: Flash Attention (fused).
    for _ in 0..warmup {
        let _ = sdpa(&q_gpu, &k_gpu, &v_gpu, None, scale).unwrap();
    }

    // Timed: Flash Attention.
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = sdpa(&q_gpu, &k_gpu, &v_gpu, None, scale).unwrap();
    }
    let flash_elapsed = start.elapsed();

    // Decomposed path: build Q@K^T, softmax, @V on GPU.
    // Force decomposed by using the CPU function on GPU tensors (via explicit ops).
    let decomposed_fn = |q: &DynTensor, k: &DynTensor, v: &DynTensor| -> DynTensor {
        let k_t = k.transpose(2, 3).unwrap();
        let scores = q.matmul(&k_t).unwrap().mul_scalar(scale).unwrap();
        let attn_weights = scores.softmax(scores.rank() - 1).unwrap();
        attn_weights.matmul(v).unwrap()
    };

    // Warmup: decomposed.
    for _ in 0..warmup {
        let _ = decomposed_fn(&q_gpu, &k_gpu, &v_gpu);
    }

    // Timed: decomposed.
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = decomposed_fn(&q_gpu, &k_gpu, &v_gpu);
    }
    let decomposed_elapsed = start.elapsed();

    let flash_us = flash_elapsed.as_micros() as f64 / f64::from(iters);
    let decomposed_us = decomposed_elapsed.as_micros() as f64 / f64::from(iters);
    let speedup = decomposed_us / flash_us;

    eprintln!(
        "flash_attn perf [B={batch} H={heads} S={seq} D={head_dim}]: \
         flash={flash_us:.0}us decomposed={decomposed_us:.0}us speedup={speedup:.2}x"
    );

    // Verify correctness (both paths produce same results).
    let flash_result = sdpa(&q_gpu, &k_gpu, &v_gpu, None, scale).unwrap();
    let decomposed_result = decomposed_fn(&q_gpu, &k_gpu, &v_gpu);
    let flash_cpu = flash_result.to_device(&Device::Cpu).unwrap();
    let decomposed_cpu = decomposed_result.to_device(&Device::Cpu).unwrap();
    compare_results(&flash_cpu, &decomposed_cpu, 1e-3, "perf correctness");
}

// ==========================================================================
// dpdf document understanding VLM shapes (1K-16K tokens) — Part of #3858
// ==========================================================================
//
// Document understanding VLMs (Granite-Docling, Qwen3-VL, InternVL) process
// document images at high resolution, producing 1K-16K visual tokens. The
// attention heads and head dimensions match common vision encoder configs:
//
//   SigLIP2-base: H=12, D=64 (768 hidden / 12 heads)
//   ViT-L/14:     H=16, D=64 (1024 hidden / 16 heads)
//   ViT-G/14:     H=16, D=88 (1408 hidden / 16 heads)
//
// Sequence lengths for square document images at patch_size=16:
//   512px  → (512/16)^2  = 1024 patches
//   768px  → (768/16)^2  = 2304 patches
//   1024px → (1024/16)^2 = 4096 patches
//   2048px → (2048/16)^2 = 16384 patches (extreme, table-heavy documents)
//
// These tests verify correctness at dpdf-relevant scales. For S >= 1024,
// the CPU decomposed reference is computed on GPU (decomposed matmul path)
// since the O(S^2) CPU path would be too slow. Flash Attention's O(S * D)
// memory advantage is the whole point — the GPU decomposed path materializes
// the full attention matrix while Flash Attention does not.

/// Helper: GPU-only correctness check. Uses decomposed GPU SDPA as reference
/// (allocates full N^2 attention matrix) and compares against fused Flash
/// Attention. Both paths run on Metal — this tests the kernel, not CPU parity.
fn check_flash_attn_gpu_parity(
    batch: usize,
    heads: usize,
    s_q: usize,
    s_kv: usize,
    head_dim: usize,
    tol: f32,
) {
    super::test_utils::gpu_init();

    let scale = 1.0 / (head_dim as f64).sqrt();
    let label = format!("gpu B={batch} H={heads} Sq={s_q} Skv={s_kv} D={head_dim}");

    // Create tensors directly on GPU.
    let (q_cpu, k_cpu, v_cpu) = random_qkv(42, batch, heads, s_q, s_kv, head_dim, &Device::Cpu);
    let q_gpu = q_cpu.to_device(&Device::metal()).unwrap();
    let k_gpu = k_cpu.to_device(&Device::metal()).unwrap();
    let v_gpu = v_cpu.to_device(&Device::metal()).unwrap();

    // Flash Attention (fused GPU path).
    let flash_result = sdpa(&q_gpu, &k_gpu, &v_gpu, None, scale).unwrap();

    // Decomposed GPU path: explicit matmul + softmax + matmul.
    let decomposed_result = {
        let k_t = k_gpu.transpose(2, 3).unwrap();
        let scores = q_gpu.matmul(&k_t).unwrap().mul_scalar(scale).unwrap();
        let attn_weights = scores.softmax(scores.rank() - 1).unwrap();
        attn_weights.matmul(&v_gpu).unwrap()
    };

    let flash_cpu = flash_result.to_device(&Device::Cpu).unwrap();
    let decomposed_cpu = decomposed_result.to_device(&Device::Cpu).unwrap();
    compare_results(&flash_cpu, &decomposed_cpu, tol, &label);
}

/// Helper: GPU-only causal correctness check.
fn check_flash_attn_gpu_causal_parity(
    batch: usize,
    heads: usize,
    seq: usize,
    head_dim: usize,
    tol: f32,
) {
    super::test_utils::gpu_init();

    let scale = 1.0 / (head_dim as f64).sqrt();
    let label = format!("gpu causal B={batch} H={heads} S={seq} D={head_dim}");

    let (q_cpu, k_cpu, v_cpu) = random_qkv(42, batch, heads, seq, seq, head_dim, &Device::Cpu);
    let q_gpu = q_cpu.to_device(&Device::metal()).unwrap();
    let k_gpu = k_cpu.to_device(&Device::metal()).unwrap();
    let v_gpu = v_cpu.to_device(&Device::metal()).unwrap();

    // Flash Attention causal (fused).
    let flash_result = sdpa_causal(&q_gpu, &k_gpu, &v_gpu, scale).unwrap();

    // Decomposed causal: explicit mask + matmul + softmax + matmul.
    let mask = causal_mask(seq, &Device::metal()).unwrap();
    let decomposed_result = {
        let k_t = k_gpu.transpose(2, 3).unwrap();
        let scores = q_gpu.matmul(&k_t).unwrap().mul_scalar(scale).unwrap();
        let scores = scores.broadcast_add(&mask).unwrap();
        let attn_weights = scores.softmax(scores.rank() - 1).unwrap();
        attn_weights.matmul(&v_gpu).unwrap()
    };

    let flash_cpu = flash_result.to_device(&Device::Cpu).unwrap();
    let decomposed_cpu = decomposed_result.to_device(&Device::Cpu).unwrap();
    compare_results(&flash_cpu, &decomposed_cpu, tol, &label);
}

/// Helper: GPU-only GQA correctness check.
fn check_flash_attn_gpu_gqa_parity(
    batch: usize,
    h_q: usize,
    h_kv: usize,
    seq: usize,
    head_dim: usize,
    tol: f32,
) {
    super::test_utils::gpu_init();

    let scale = 1.0 / (head_dim as f64).sqrt();
    let label = format!("gpu GQA B={batch} Hq={h_q} Hkv={h_kv} S={seq} D={head_dim}");

    let (q_cpu, k_cpu, v_cpu) =
        random_qkv_gqa(42, batch, h_q, h_kv, seq, seq, head_dim, &Device::Cpu);
    let q_gpu = q_cpu.to_device(&Device::metal()).unwrap();
    let k_gpu = k_cpu.to_device(&Device::metal()).unwrap();
    let v_gpu = v_cpu.to_device(&Device::metal()).unwrap();

    // Flash Attention handles GQA natively.
    let flash_result = sdpa(&q_gpu, &k_gpu, &v_gpu, None, scale).unwrap();

    // Decomposed: expand K/V heads via repeat_kv, then standard SDPA.
    let num_rep = h_q / h_kv;
    let k_expanded = repeat_kv(&k_gpu, num_rep).unwrap();
    let v_expanded = repeat_kv(&v_gpu, num_rep).unwrap();
    let decomposed_result = {
        let k_t = k_expanded.transpose(2, 3).unwrap();
        let scores = q_gpu.matmul(&k_t).unwrap().mul_scalar(scale).unwrap();
        let attn_weights = scores.softmax(scores.rank() - 1).unwrap();
        attn_weights.matmul(&v_expanded).unwrap()
    };

    let flash_cpu = flash_result.to_device(&Device::Cpu).unwrap();
    let decomposed_cpu = decomposed_result.to_device(&Device::Cpu).unwrap();
    compare_results(&flash_cpu, &decomposed_cpu, tol, &label);
}

// -- dpdf SigLIP2-base shapes: H=12, D=64 ------------------------------------
// SigLIP2-base-patch16 used by Granite-Docling-258M (dpdf Tier 1).

/// SigLIP2-base at 512px: 1024 patches, H=12, D=64.
/// This is the most common dpdf resolution for single-page documents.
#[test]
fn flash_attn_dpdf_siglip2_512px_s1024() {
    check_flash_attn_gpu_parity(1, 12, 1024, 1024, 64, 2e-3);
}

/// SigLIP2-base at 768px: 2304 patches, H=12, D=64.
/// High-resolution single-page scan.
#[test]
fn flash_attn_dpdf_siglip2_768px_s2304() {
    check_flash_attn_gpu_parity(1, 12, 2304, 2304, 64, 2e-3);
}

/// SigLIP2-base at 1024px: 4096 patches, H=12, D=64.
/// Maximum common resolution for high-detail document pages.
#[test]
fn flash_attn_dpdf_siglip2_1024px_s4096() {
    check_flash_attn_gpu_parity(1, 12, 4096, 4096, 64, 2e-3);
}

// -- dpdf ViT-L shapes: H=16, D=64 -------------------------------------------
// ViT-Large/14 used by InternVL and other VLMs.

/// ViT-L at 448px (patch_size=14): (448/14)^2 = 1024 patches.
#[test]
fn flash_attn_dpdf_vit_large_s1024() {
    check_flash_attn_gpu_parity(1, 16, 1024, 1024, 64, 2e-3);
}

/// ViT-L at 896px: (896/14)^2 = 4096 patches.
#[test]
fn flash_attn_dpdf_vit_large_s4096() {
    check_flash_attn_gpu_parity(1, 16, 4096, 4096, 64, 2e-3);
}

// -- dpdf VLM cross-attention: text queries attend to visual tokens -----------
// In VLM cross-attention, a small number of text/query tokens attend to the
// full set of visual tokens. The asymmetry (S_q << S_kv) is the typical
// dpdf use case: OCR text attending to document image patches.

/// VLM cross-attention: 128 text tokens querying 1024 visual tokens.
#[test]
fn flash_attn_dpdf_cross_attn_128q_1024kv() {
    check_flash_attn_gpu_parity(1, 12, 128, 1024, 64, 2e-3);
}

/// VLM cross-attention: 256 text tokens querying 4096 visual tokens.
#[test]
fn flash_attn_dpdf_cross_attn_256q_4096kv() {
    check_flash_attn_gpu_parity(1, 12, 256, 4096, 64, 2e-3);
}

/// VLM cross-attention: single decode token querying 4096 visual tokens.
/// Autoregressive decode step attending to full visual context.
#[test]
fn flash_attn_dpdf_cross_attn_1q_4096kv() {
    check_flash_attn_gpu_parity(1, 12, 1, 4096, 64, 2e-3);
}

// -- dpdf causal attention for VLM decoder ------------------------------------
// The LLM decoder side of VLMs uses causal attention on the combined
// (visual + text) sequence.

/// Causal attention at 1024 tokens, H=12, D=64 — VLM decoder.
#[test]
fn flash_attn_dpdf_causal_s1024() {
    check_flash_attn_gpu_causal_parity(1, 12, 1024, 64, 2e-3);
}

/// Causal attention at 2048 tokens — longer document + OCR text combined.
#[test]
fn flash_attn_dpdf_causal_s2048() {
    check_flash_attn_gpu_causal_parity(1, 12, 2048, 64, 2e-3);
}

// -- dpdf GQA at document scales ----------------------------------------------
// Modern VLM decoders (Qwen3-VL, Llama-3.2-Vision) use GQA.
// Qwen3-VL: H_q=16, H_kv=2 (group_size=8), D=128.

/// GQA at 1024 tokens: Qwen3-VL-like config.
#[test]
fn flash_attn_dpdf_gqa_s1024_h16_hkv2() {
    check_flash_attn_gpu_gqa_parity(1, 16, 2, 1024, 128, 2e-3);
}

/// GQA at 2048 tokens: extended document context.
#[test]
fn flash_attn_dpdf_gqa_s2048_h16_hkv2() {
    check_flash_attn_gpu_gqa_parity(1, 16, 2, 2048, 128, 2e-3);
}

// -- Extreme document understanding scales ------------------------------------
// These test the kernel at production scales for multi-page documents
// and high-resolution table extraction.

/// 8192 tokens: two A4 pages at 1024px in a multi-page document model.
/// Uses H=12, D=64 (SigLIP2-base family).
#[test]
// OOM GUARD (2026-06-15 incident): the decomposed-parity reference materializes
// the full O(S^2) attention matrix on GPU (~10GB+ live at S=8192). Run only
// serially via the nextest `heavy` test-group, or set NN_RUN_GPU=1 deliberately.
#[ignore = "OOM risk: O(S^2) decomposed reference; run via nextest heavy group / NN_RUN_GPU"]
fn flash_attn_dpdf_s8192() {
    check_flash_attn_gpu_parity(1, 12, 8192, 8192, 64, 3e-3);
}

/// 16384 tokens: four-page document or 2048px single page.
/// Maximum scale for dpdf production. Higher tolerance (3e-3) to account
/// for accumulated floating-point differences across 512 K/V tile iterations.
#[test]
// OOM GUARD (2026-06-15 incident): at S=16384 the decomposed-parity reference
// holds ~38-50GB of O(S^2) intermediates and was the dominant consumer in the
// kernel-panic snapshot (gpu_ops_all hit 98.5GB). Serialize via nextest `heavy`
// group or run deliberately with NN_RUN_GPU=1 under a memory cap.
#[ignore = "OOM risk: ~38-50GB O(S^2) decomposed reference (S=16384); run via nextest heavy group / NN_RUN_GPU"]
fn flash_attn_dpdf_s16384() {
    check_flash_attn_gpu_parity(1, 12, 16384, 16384, 64, 3e-3);
}

// -- dpdf performance benchmarks ----------------------------------------------

/// Benchmark: Flash Attention throughput at dpdf-relevant scales.
///
/// Reports wall-clock time for Flash Attention at S=1024, 2048, 4096
/// with SigLIP2-base config (H=12, D=64). Measures the fused kernel only
/// (not decomposed comparison) since the decomposed path OOMs at large S.
#[test]
fn flash_attn_dpdf_benchmark() {
    super::test_utils::gpu_init();

    let head_dim = 64;
    let heads = 12;
    let batch = 1;
    let warmup = 3;
    let iters = 5;

    for seq in [1024, 2048, 4096] {
        let scale = 1.0 / (head_dim as f64).sqrt();

        let (q_cpu, k_cpu, v_cpu) = random_qkv(42, batch, heads, seq, seq, head_dim, &Device::Cpu);
        let q_gpu = q_cpu.to_device(&Device::metal()).unwrap();
        let k_gpu = k_cpu.to_device(&Device::metal()).unwrap();
        let v_gpu = v_cpu.to_device(&Device::metal()).unwrap();

        // Warmup.
        for _ in 0..warmup {
            let _ = sdpa(&q_gpu, &k_gpu, &v_gpu, None, scale).unwrap();
        }

        // Timed.
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let _ = sdpa(&q_gpu, &k_gpu, &v_gpu, None, scale).unwrap();
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() as f64 / f64::from(iters);

        // Memory: Flash Attention uses O(S * D) per head vs O(S^2) decomposed.
        let flash_mem_per_head = seq * head_dim * 4; // f32 output
        let decomposed_mem_per_head = seq * seq * 4; // f32 attention matrix
        let savings = decomposed_mem_per_head as f64 / flash_mem_per_head as f64;

        eprintln!(
            "flash_attn dpdf [B={batch} H={heads} S={seq} D={head_dim}]: \
             {avg_us:.0}us, memory savings={savings:.0}x \
             (flash={flash_mem_per_head} bytes/head, decomposed={decomposed_mem_per_head} bytes/head)"
        );
    }

    // Verify output is on GPU and correct shape.
    let scale = 1.0 / (head_dim as f64).sqrt();
    let (q_cpu, k_cpu, v_cpu) = random_qkv(42, batch, heads, 1024, 1024, head_dim, &Device::Cpu);
    let q_gpu = q_cpu.to_device(&Device::metal()).unwrap();
    let k_gpu = k_cpu.to_device(&Device::metal()).unwrap();
    let v_gpu = v_cpu.to_device(&Device::metal()).unwrap();
    let result = sdpa(&q_gpu, &k_gpu, &v_gpu, None, scale).unwrap();
    assert!(result.device().is_gpu(), "benchmark result must be on GPU");
    assert_eq!(result.dims(), &[batch, heads, 1024, head_dim]);
}
