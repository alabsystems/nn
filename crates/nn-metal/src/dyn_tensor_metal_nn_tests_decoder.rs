#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Transformer decoder block GPU e2e tests — LLaMA-style decoder with
//! MultiHeadAttention + RmsNorm + SwiGlu + causal masking + residual.
//!
//! Validates DynTensor nn modules compose correctly on Metal GPU for a
//! realistic decoder block pattern used by LLaMA, Mistral, Qwen3, etc.
//!
//! GPU ops exercised (not covered by PlBert encoder tests):
//! - `MultiHeadAttention` via nn module (not hand-rolled)
//! - `RmsNorm` (pre-norm architecture, no LayerNorm)
//! - `SwiGlu` (gated FFN with SiLU, not GELU)
//! - `causal_mask` (attention masking with -inf)
//! - Residual connections (broadcast_add of residual stream)
//!
//! Issue: #1185 AC2

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{causal_mask, Linear, Module, MultiHeadAttention, RmsNorm, SwiGlu};
use nn_core::Device;

use crate::test_common::{assert_close, init};

// -- Test dimensions ----------------------------------------------------------
// Tiny model dimensions for fast tests: dim=8, heads=2, ff=16.

const DIM: usize = 8;
const NUM_HEADS: usize = 2;
const FF_DIM: usize = DIM * 2; // 16
const EPS: f64 = 1e-5;

/// Deterministic varied-value tensor (non-uniform values to break symmetry).
fn v(shape: &[usize], base: f32) -> DynTensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| base + 0.01 * (i as f32) - 0.005 * ((i % 3) as f32))
        .collect();
    DynTensor::from_vec(data, shape, &Device::Cpu).unwrap()
}

/// Build MultiHeadAttention on given device.
fn build_attention(dev: &Device) -> MultiHeadAttention {
    let q_proj = Linear::new(v(&[DIM, DIM], 0.05).to_device(dev).unwrap(), None).unwrap();
    let k_proj = Linear::new(v(&[DIM, DIM], 0.07).to_device(dev).unwrap(), None).unwrap();
    let v_proj = Linear::new(v(&[DIM, DIM], 0.09).to_device(dev).unwrap(), None).unwrap();
    let out_proj = Linear::new(v(&[DIM, DIM], 0.03).to_device(dev).unwrap(), None).unwrap();
    MultiHeadAttention::new(q_proj, k_proj, v_proj, out_proj, NUM_HEADS, NUM_HEADS).unwrap()
}

/// Build SwiGlu FFN on given device.
fn build_swiglu(dev: &Device) -> SwiGlu {
    let w_gate = Linear::new(v(&[FF_DIM, DIM], 0.04).to_device(dev).unwrap(), None).unwrap();
    let w_up = Linear::new(v(&[FF_DIM, DIM], 0.06).to_device(dev).unwrap(), None).unwrap();
    let w_down = Linear::new(v(&[DIM, FF_DIM], 0.02).to_device(dev).unwrap(), None).unwrap();
    SwiGlu::new(w_gate, w_up, w_down).unwrap()
}

/// Build RmsNorm on given device.
fn build_rms_norm(dev: &Device) -> RmsNorm {
    let weight = v(&[DIM], 1.0).to_device(dev).unwrap();
    RmsNorm::new(weight, EPS).unwrap()
}

/// LLaMA-style decoder block forward pass:
///   residual = x
///   x = rms_norm_attn(x)
///   x = attention(x, causal_mask)
///   x = residual + x
///   residual = x
///   x = rms_norm_ffn(x)
///   x = swiglu(x)
///   x = residual + x
fn decoder_block_forward(
    x: &DynTensor,
    attn: &MultiHeadAttention,
    ffn: &SwiGlu,
    norm_attn: &RmsNorm,
    norm_ffn: &RmsNorm,
    mask: &DynTensor,
) -> nn_core::Result<DynTensor> {
    // Pre-norm attention with residual
    let residual = x.clone();
    let h = norm_attn.forward(x)?;
    let h = attn.forward(&h, None, Some(mask), None, 0)?;
    let h = residual.broadcast_add(&h)?;

    // Pre-norm SwiGlu FFN with residual
    let residual = h.clone();
    let h = norm_ffn.forward(&h)?;
    let h = ffn.forward(&h)?;
    residual.broadcast_add(&h)
}

// -- Tests --------------------------------------------------------------------

#[test]
fn test_decoder_block_gpu_matches_cpu() {
    // Full decoder block: RmsNorm → MHA → residual → RmsNorm → SwiGlu → residual.
    // B=1, T=4, D=8, H=2, FF=16.
    init();

    let seq_len = 4;
    let batch = 1;

    // Build on CPU
    let cpu_attn = build_attention(&Device::Cpu);
    let cpu_ffn = build_swiglu(&Device::Cpu);
    let cpu_norm_attn = build_rms_norm(&Device::Cpu);
    let cpu_norm_ffn = build_rms_norm(&Device::Cpu);
    let cpu_mask = causal_mask(seq_len, &Device::Cpu).unwrap();
    let cpu_x = v(&[batch, seq_len, DIM], 0.5);

    let cpu_out = decoder_block_forward(
        &cpu_x,
        &cpu_attn,
        &cpu_ffn,
        &cpu_norm_attn,
        &cpu_norm_ffn,
        &cpu_mask,
    )
    .unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();

    // Build on GPU
    let gpu_dev = Device::metal();
    let gpu_attn = build_attention(&gpu_dev);
    let gpu_ffn = build_swiglu(&gpu_dev);
    let gpu_norm_attn = build_rms_norm(&gpu_dev);
    let gpu_norm_ffn = build_rms_norm(&gpu_dev);
    let gpu_mask = causal_mask(seq_len, &gpu_dev).unwrap();
    let gpu_x = v(&[batch, seq_len, DIM], 0.5).to_device(&gpu_dev).unwrap();

    let gpu_out = decoder_block_forward(
        &gpu_x,
        &gpu_attn,
        &gpu_ffn,
        &gpu_norm_attn,
        &gpu_norm_ffn,
        &gpu_mask,
    )
    .unwrap();

    assert_eq!(gpu_out.dims(), &[batch, seq_len, DIM]);
    assert_eq!(gpu_out.device(), Device::metal(), "output must stay on GPU");
    assert!(
        gpu_out
            .to_device(&Device::Cpu)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .all(|x| x.is_finite()),
        "GPU output must be finite"
    );

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 5e-3, "decoder_block_gpu");
}

#[test]
fn test_decoder_block_gpu_batched() {
    // Batched decoder: B=2, T=3. Both batch items must match CPU.
    init();

    let seq_len = 3;
    let batch = 2;

    let cpu_attn = build_attention(&Device::Cpu);
    let cpu_ffn = build_swiglu(&Device::Cpu);
    let cpu_norm_attn = build_rms_norm(&Device::Cpu);
    let cpu_norm_ffn = build_rms_norm(&Device::Cpu);
    let cpu_mask = causal_mask(seq_len, &Device::Cpu).unwrap();
    let cpu_x = v(&[batch, seq_len, DIM], 0.3);

    let cpu_out = decoder_block_forward(
        &cpu_x,
        &cpu_attn,
        &cpu_ffn,
        &cpu_norm_attn,
        &cpu_norm_ffn,
        &cpu_mask,
    )
    .unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();

    let gpu_dev = Device::metal();
    let gpu_attn = build_attention(&gpu_dev);
    let gpu_ffn = build_swiglu(&gpu_dev);
    let gpu_norm_attn = build_rms_norm(&gpu_dev);
    let gpu_norm_ffn = build_rms_norm(&gpu_dev);
    let gpu_mask = causal_mask(seq_len, &gpu_dev).unwrap();
    let gpu_x = v(&[batch, seq_len, DIM], 0.3).to_device(&gpu_dev).unwrap();

    let gpu_out = decoder_block_forward(
        &gpu_x,
        &gpu_attn,
        &gpu_ffn,
        &gpu_norm_attn,
        &gpu_norm_ffn,
        &gpu_mask,
    )
    .unwrap();

    assert_eq!(gpu_out.dims(), &[batch, seq_len, DIM]);
    assert_eq!(gpu_out.device(), Device::metal());

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        gpu_vals.iter().all(|x| x.is_finite()),
        "batched GPU output must be finite"
    );
    assert_close(&gpu_vals, &cpu_vals, 5e-3, "decoder_block_batched_gpu");
}

#[test]
fn test_decoder_block_gpu_output_device() {
    // Output must remain on Metal device, not silently CPU.
    init();

    let gpu_dev = Device::metal();
    let attn = build_attention(&gpu_dev);
    let ffn = build_swiglu(&gpu_dev);
    let norm_attn = build_rms_norm(&gpu_dev);
    let norm_ffn = build_rms_norm(&gpu_dev);
    let mask = causal_mask(3, &gpu_dev).unwrap();
    let x = v(&[1, 3, DIM], 0.5).to_device(&gpu_dev).unwrap();

    let out = decoder_block_forward(&x, &attn, &ffn, &norm_attn, &norm_ffn, &mask).unwrap();
    assert_eq!(out.device(), Device::metal());
}

#[test]
fn test_swiglu_gpu_matches_cpu() {
    // SwiGlu forward in isolation: gate=silu(w_gate(x)), up=w_up(x), out=w_down(gate*up).
    // Exercises: Linear (3 matmuls), SiLU, broadcast_mul.
    init();

    let cpu_ffn = build_swiglu(&Device::Cpu);
    let cpu_x = v(&[2, 4, DIM], 0.4);
    let cpu_vals = cpu_ffn
        .forward(&cpu_x)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let gpu_dev = Device::metal();
    let gpu_ffn = build_swiglu(&gpu_dev);
    let gpu_x = v(&[2, 4, DIM], 0.4).to_device(&gpu_dev).unwrap();
    let gpu_out = gpu_ffn.forward(&gpu_x).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[2, 4, DIM]);

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-3, "swiglu_gpu");
}

#[test]
fn test_rms_norm_gpu_matches_cpu() {
    // RmsNorm forward in isolation: x / sqrt(mean(x^2) + eps) * weight.
    // Exercises: sqr, mean_keepdim, broadcast_add, sqrt, broadcast_div, broadcast_mul.
    init();

    let cpu_norm = build_rms_norm(&Device::Cpu);
    let cpu_x = v(&[2, 4, DIM], 0.8);
    let cpu_vals = cpu_norm
        .forward(&cpu_x)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let gpu_dev = Device::metal();
    let gpu_norm = build_rms_norm(&gpu_dev);
    let gpu_x = v(&[2, 4, DIM], 0.8).to_device(&gpu_dev).unwrap();
    let gpu_out = gpu_norm.forward(&gpu_x).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[2, 4, DIM]);

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-4, "rms_norm_gpu");
}

#[test]
fn test_multi_head_attention_gpu_with_causal_mask() {
    // MHA with causal mask: Q/K/V projections + masked softmax + output projection.
    // Exercises: matmul, reshape, transpose, mul_scalar, broadcast_add(-inf),
    // softmax, matmul(attn_weights @ V).
    init();

    let seq_len = 5;
    let batch = 1;

    let cpu_attn = build_attention(&Device::Cpu);
    let cpu_mask = causal_mask(seq_len, &Device::Cpu).unwrap();
    let cpu_x = v(&[batch, seq_len, DIM], 0.6);
    let cpu_vals = cpu_attn
        .forward(&cpu_x, None, Some(&cpu_mask), None, 0)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let gpu_dev = Device::metal();
    let gpu_attn = build_attention(&gpu_dev);
    let gpu_mask = causal_mask(seq_len, &gpu_dev).unwrap();
    let gpu_x = v(&[batch, seq_len, DIM], 0.6).to_device(&gpu_dev).unwrap();
    let gpu_out = gpu_attn
        .forward(&gpu_x, None, Some(&gpu_mask), None, 0)
        .unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[batch, seq_len, DIM]);

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        gpu_vals.iter().all(|x| x.is_finite()),
        "MHA GPU output must be finite (no NaN from masked softmax)"
    );
    assert_close(&gpu_vals, &cpu_vals, 5e-3, "mha_causal_gpu");
}
