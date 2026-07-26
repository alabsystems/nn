// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Softmax parity tests at real attention dimensions.
//!
//! Tests softmax on tensor shapes drawn from production transformers:
//! - Whisper attention: [12, 128, 128] (12 heads, 128 seq)
//! - Qwen3 attention: [32, 64, 64] (32 heads, 64 seq)
//! - Large sequence attention: [8, 512, 512]
//! - Logits softmax: [1, seq, vocab_size]
//!
//! Softmax precision is critical: small numerical errors in the
//! max-subtraction or exp-sum steps propagate through attention weights.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_prng::rand_f32_vec;
use nn_core::Device;

/// Tolerance for softmax. The exp/sum chain is sensitive to precision.
const TOL: f32 = 1e-5;

/// Helper: verify softmax output sums to ~1.0 along the given axis.
fn verify_softmax_sums(tensor: &DynTensor, axis: usize, label: &str) {
    let summed = tensor.sum(axis).unwrap();
    let vals = summed
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-4,
            "{label}: softmax sum[{i}]={v}, expected ~1.0"
        );
    }
}

// -- Whisper attention shapes ------------------------------------------------

/// Whisper self-attention scores: softmax over [12, 128, 128].
/// 12 heads, 128 sequence length — softmax along last dim.
#[test]
fn test_softmax_whisper_attention() {
    gpu_init();
    let heads = 12;
    let seq = 128;

    let data = rand_f32_vec(4000, heads * seq * seq, -5.0, 5.0);

    let cpu = DynTensor::new(&data, &[heads, seq, seq], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[heads, seq, seq], &Device::metal()).unwrap();

    let cpu_out = cpu.softmax(2).unwrap();
    let gpu_out = gpu.softmax(2).unwrap();

    assert_eq!(gpu_out.dims(), &[heads, seq, seq]);
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "whisper_attn_softmax");
    verify_softmax_sums(&gpu_out, 2, "whisper_attn");
}

/// Whisper cross-attention: softmax over [12, 64, 128].
/// Decoder seq=64 attending to encoder seq=128.
#[test]
fn test_softmax_whisper_cross_attention() {
    gpu_init();
    let heads = 12;
    let dec_seq = 64;
    let enc_seq = 128;

    let data = rand_f32_vec(4001, heads * dec_seq * enc_seq, -5.0, 5.0);

    let cpu = DynTensor::new(&data, &[heads, dec_seq, enc_seq], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[heads, dec_seq, enc_seq], &Device::metal()).unwrap();

    let cpu_out = cpu.softmax(2).unwrap();
    let gpu_out = gpu.softmax(2).unwrap();

    assert_eq!(gpu_out.dims(), &[heads, dec_seq, enc_seq]);
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "whisper_cross_attn_softmax");
    verify_softmax_sums(&gpu_out, 2, "whisper_cross_attn");
}

// -- Qwen3 attention shapes -------------------------------------------------

/// Qwen3 self-attention: softmax over [32, 64, 64].
/// 32 heads (GQA expanded), 64 sequence length.
#[test]
fn test_softmax_qwen3_attention() {
    gpu_init();
    let heads = 32;
    let seq = 64;

    let data = rand_f32_vec(4010, heads * seq * seq, -5.0, 5.0);

    let cpu = DynTensor::new(&data, &[heads, seq, seq], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[heads, seq, seq], &Device::metal()).unwrap();

    let cpu_out = cpu.softmax(2).unwrap();
    let gpu_out = gpu.softmax(2).unwrap();

    assert_eq!(gpu_out.dims(), &[heads, seq, seq]);
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "qwen3_attn_softmax");
}

// -- Long sequence attention -------------------------------------------------

/// Large attention matrix: softmax over [8, 512, 512].
/// Exercises the softmax kernel at a scale where numerical stability matters.
#[test]
fn test_softmax_long_sequence() {
    gpu_init();
    let heads = 8;
    let seq = 512;

    let data = rand_f32_vec(4020, heads * seq * seq, -10.0, 10.0);

    let cpu = DynTensor::new(&data, &[heads, seq, seq], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[heads, seq, seq], &Device::metal()).unwrap();

    let cpu_out = cpu.softmax(2).unwrap();
    let gpu_out = gpu.softmax(2).unwrap();

    assert_eq!(gpu_out.dims(), &[heads, seq, seq]);
    // Wider tolerance for large softmax due to accumulation over 512 elements.
    assert_gpu_cpu_close(&gpu_out, &cpu_out, 5e-5, "long_seq_softmax_512");
    verify_softmax_sums(&gpu_out, 2, "long_seq_512");
}

// -- Logits softmax (vocabulary distribution) --------------------------------

/// Token logits softmax: [1, 16, 32000] (Qwen3-style vocabulary).
/// Large last dimension tests the softmax reduction path at vocabulary scale.
#[test]
fn test_softmax_logits_qwen3() {
    gpu_init();
    let batch = 1;
    let seq = 16;
    let vocab = 32000;

    let data = rand_f32_vec(4030, batch * seq * vocab, -5.0, 5.0);

    let cpu = DynTensor::new(&data, &[batch, seq, vocab], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[batch, seq, vocab], &Device::metal()).unwrap();

    let cpu_out = cpu.softmax(2).unwrap();
    let gpu_out = gpu.softmax(2).unwrap();

    assert_eq!(gpu_out.dims(), &[batch, seq, vocab]);
    // Large vocab dimension: tolerance slightly wider.
    assert_gpu_cpu_close(&gpu_out, &cpu_out, 1e-4, "logits_softmax_32k");
    verify_softmax_sums(&gpu_out, 2, "logits_32k");
}

/// Whisper logits softmax: [1, 8, 51865] (Whisper vocabulary).
#[test]
fn test_softmax_logits_whisper() {
    gpu_init();
    let batch = 1;
    let seq = 8;
    let vocab = 51865;

    let data = rand_f32_vec(4031, batch * seq * vocab, -5.0, 5.0);

    let cpu = DynTensor::new(&data, &[batch, seq, vocab], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[batch, seq, vocab], &Device::metal()).unwrap();

    let cpu_out = cpu.softmax(2).unwrap();
    let gpu_out = gpu.softmax(2).unwrap();

    assert_eq!(gpu_out.dims(), &[batch, seq, vocab]);
    assert_gpu_cpu_close(&gpu_out, &cpu_out, 1e-4, "logits_softmax_whisper");
}

// -- Log-softmax (used in cross-entropy loss) --------------------------------

/// Log-softmax at attention scale: [12, 128, 128].
#[test]
fn test_log_softmax_attention_scale() {
    gpu_init();
    let heads = 12;
    let seq = 128;

    let data = rand_f32_vec(4040, heads * seq * seq, -5.0, 5.0);

    let cpu = DynTensor::new(&data, &[heads, seq, seq], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[heads, seq, seq], &Device::metal()).unwrap();

    let cpu_out = cpu.log_softmax(2).unwrap();
    let gpu_out = gpu.log_softmax(2).unwrap();

    assert_eq!(gpu_out.dims(), &[heads, seq, seq]);
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "log_softmax_attn");

    // Log-softmax values should be <= 0.
    let vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(v <= 1e-6, "log_softmax[{i}]={v} should be <= 0");
    }
}

// -- Softmax with extreme values (numerical stability) -----------------------

/// Softmax with large magnitude inputs: tests max-subtraction stability.
/// Values in [-50, 50] are realistic for unscaled attention logits.
#[test]
fn test_softmax_extreme_values() {
    gpu_init();
    let shape = [4, 64, 64];
    let n: usize = shape.iter().product();

    let data = rand_f32_vec(4050, n, -50.0, 50.0);

    let cpu = DynTensor::new(&data, &shape, &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &shape, &Device::metal()).unwrap();

    let cpu_out = cpu.softmax(2).unwrap();
    let gpu_out = gpu.softmax(2).unwrap();

    assert_gpu_cpu_close(&gpu_out, &cpu_out, 1e-4, "softmax_extreme_values");

    // No NaN/Inf in output.
    let vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "softmax_extreme[{i}]={v} is non-finite");
        assert!(v >= 0.0, "softmax_extreme[{i}]={v} is negative");
    }
}
