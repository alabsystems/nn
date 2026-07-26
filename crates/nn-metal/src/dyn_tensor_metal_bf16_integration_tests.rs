// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![allow(deprecated)]

//! BF16 end-to-end integration tests (#1710).
//!
//! Tests exercise the **composition** of bf16 ops through model-like
//! pipelines on Metal GPU, comparing outputs against f32 baselines.
//! Individual bf16 op tests exist elsewhere; these catch composition
//! failures like the 5 P1 bugs from #1646.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{self, Module};
use nn_core::{DType, Device};

use crate::test_common::{assert_close, init};

// -- Helpers ------------------------------------------------------------------

/// Create f32 CPU tensor, convert to bf16, move to GPU.
fn bf16_gpu(data: &[f32], shape: &[usize]) -> DynTensor {
    let cpu = DynTensor::new(data, shape, &Device::Cpu).unwrap();
    let bf16 = cpu.to_dtype(DType::BF16).unwrap();
    bf16.to_device(&Device::metal()).unwrap()
}

/// Extract bf16 GPU tensor as f32 values for comparison.
fn to_f32_vals(t: &DynTensor) -> Vec<f32> {
    let cpu = t.to_device(&Device::Cpu).unwrap();
    let arr = cpu.to_f32_array().unwrap();
    arr.iter().copied().collect()
}

/// Assert all values are finite (no NaN or Inf).
fn assert_all_finite(vals: &[f32], label: &str) {
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "{label}[{i}] is not finite: {v}");
    }
}

// -- AC1: Composed nn layer pipeline with bf16 on GPU -------------------------
//
// Mirrors a simplified model forward: Conv1d → LayerNorm → Linear.
// Exercises matmul, normalization, and element-wise ops in bf16 composition.

#[test]
fn test_bf16_conv1d_layernorm_linear_pipeline() {
    init();
    let dev = Device::metal();

    // Build f32 baseline on CPU.
    let vb_f32 = nn_core::VarBuilder::zeros(DType::F32, &Device::Cpu);
    let conv = layers::conv1d(2, 4, 3, Default::default(), vb_f32.pp("conv")).unwrap();
    let ln = layers::layer_norm(4, Default::default(), vb_f32.pp("ln")).unwrap();
    let linear = layers::linear(4, 2, vb_f32.pp("linear")).unwrap();

    // f32 CPU forward.
    let input_f32 = DynTensor::ones(&[1, 2, 8], DType::F32, &Device::Cpu).unwrap();
    let cpu_out = conv.forward(&input_f32).unwrap();
    let cpu_out = ln.forward(&cpu_out.transpose(1, 2).unwrap()).unwrap();
    let cpu_out = linear.forward(&cpu_out).unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();

    // Build bf16 pipeline on GPU.
    let vb_bf16 = nn_core::VarBuilder::zeros(DType::BF16, &dev);
    let conv_gpu = layers::conv1d(2, 4, 3, Default::default(), vb_bf16.pp("conv")).unwrap();
    let ln_gpu = layers::layer_norm(4, Default::default(), vb_bf16.pp("ln")).unwrap();
    let linear_gpu = layers::linear(4, 2, vb_bf16.pp("linear")).unwrap();

    let input_bf16 = DynTensor::ones(&[1, 2, 8], DType::BF16, &dev).unwrap();
    let gpu_out = conv_gpu.forward(&input_bf16).unwrap();
    let gpu_out = ln_gpu.forward(&gpu_out.transpose(1, 2).unwrap()).unwrap();
    let gpu_out = linear_gpu.forward(&gpu_out).unwrap();

    let gpu_vals = to_f32_vals(&gpu_out);
    assert_all_finite(&gpu_vals, "bf16_pipeline");
    assert_eq!(gpu_vals.len(), cpu_vals.len(), "output length mismatch");
    // bf16 tolerance: 1e-2 (quantization noise accumulates through layers).
    assert_close(&gpu_vals, &cpu_vals, 1e-2, "bf16_pipeline");
}

// -- AC2: Whisper-like encoder pattern with bf16 on GPU -----------------------
//
// Tests Conv1d stem + activation + LayerNorm — the core Whisper encoder pattern.
// Does not require nn-whisper dependency, uses nn layers directly.

#[test]
fn test_bf16_whisper_encoder_pattern() {
    init();
    let dev = Device::metal();

    // Whisper encoder pattern: Conv1d(mel→d_model) → GELU → Conv1d → GELU → LayerNorm
    let vb = nn_core::VarBuilder::zeros(DType::BF16, &dev);
    let conv1 = layers::conv1d(4, 8, 3, Default::default(), vb.pp("conv1")).unwrap();
    let conv2 = layers::conv1d(8, 8, 3, Default::default(), vb.pp("conv2")).unwrap();
    let ln = layers::layer_norm(8, Default::default(), vb.pp("ln")).unwrap();

    // BF16 input: [batch=1, mel_bins=4, frames=16]
    let input = DynTensor::ones(&[1, 4, 16], DType::BF16, &dev).unwrap();

    let x = conv1.forward(&input).unwrap();
    assert_eq!(x.dtype(), DType::BF16, "conv1 output should be bf16");
    let x = x.gelu_erf().unwrap();
    let x = conv2.forward(&x).unwrap();
    let x = x.gelu_erf().unwrap();

    // Transpose to [batch, seq_len, channels] for LayerNorm.
    let x = x.transpose(1, 2).unwrap();
    let x = ln.forward(&x).unwrap();

    let vals = to_f32_vals(&x);
    assert_all_finite(&vals, "bf16_whisper_encoder");
    assert!(!vals.is_empty(), "encoder should produce output");
}

// -- AC3: KV cache decode step with bf16 --------------------------------------
//
// Exercises slice_set, narrow, softmax, argmax in composition — the ops that
// compose during autoregressive decode.

#[test]
fn test_bf16_kv_cache_decode_step() {
    init();

    // Simulate KV cache: [batch=1, heads=2, max_seq=8, head_dim=4]
    let cache = DynTensor::zeros(&[1, 2, 8, 4], DType::BF16, &Device::metal()).unwrap();

    // New KV entry for position 0: [1, 2, 1, 4]
    let new_kv = bf16_gpu(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &[1, 2, 1, 4]);

    // slice_set: write new KV into cache at position 0.
    let cache = cache.slice_set(2, 0, &new_kv).unwrap();

    // Narrow to retrieve filled portion: [1, 2, 1, 4]
    let filled = cache.narrow(2, 0, 1).unwrap();

    let filled_vals = to_f32_vals(&filled);
    assert_all_finite(&filled_vals, "kv_cache_filled");
    // Verify the data was written correctly (bf16 quantization tolerance).
    assert_close(
        &filled_vals,
        &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        1e-2,
        "kv_cache_written",
    );

    // Simulate attention: Q * K^T → softmax → argmax (simplified).
    let query = bf16_gpu(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &[1, 2, 1, 4]);
    let key = filled.transpose(2, 3).unwrap();
    let scores = query.matmul(&key).unwrap();
    assert_eq!(scores.dtype(), DType::BF16, "matmul should preserve bf16");

    // Softmax over seq_len dimension (last dim = 1, so softmax = 1.0).
    let probs = scores.softmax(3).unwrap();
    let prob_vals = to_f32_vals(&probs);
    assert_all_finite(&prob_vals, "softmax_probs");
    // With single token, softmax should be ~1.0.
    for &v in &prob_vals {
        assert!(
            (v - 1.0).abs() < 0.01,
            "single-token softmax should be ~1.0, got {v}"
        );
    }
}

// -- Additional composition: bf16 matmul → softmax → argmax chain -------------

#[test]
fn test_bf16_matmul_softmax_argmax_chain() {
    init();

    // Logits from linear layer: [batch=1, seq_len=4, vocab=8]
    let logits_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
    let logits = bf16_gpu(&logits_data, &[1, 4, 8]);
    assert_eq!(logits.dtype(), DType::BF16);

    // Softmax over vocab dimension.
    let probs = logits.softmax(2).unwrap();
    let prob_vals = to_f32_vals(&probs);
    assert_all_finite(&prob_vals, "logit_softmax");

    // Each row should sum to ~1.0.
    for row in 0..4 {
        let row_sum: f32 = prob_vals[row * 8..(row + 1) * 8].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 0.02,
            "row {row} sum: {row_sum} (expected ~1.0)"
        );
    }

    // Argmax over vocab dimension — returns U32 tensor.
    let ids = logits.argmax(2).unwrap();
    let id_vals = ids
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<u32>()
        .unwrap();
    assert_eq!(id_vals.len(), 4, "should have 4 token IDs");
    // Each row's max is at the last position (increasing values).
    for &id in &id_vals {
        assert_eq!(id, 7, "argmax of ascending values should be last index");
    }
}

// -- bf16 Linear → RmsNorm → Linear (LLM pattern) ----------------------------

#[test]
fn test_bf16_linear_rmsnorm_linear_llm_pattern() {
    init();
    let dev = Device::metal();

    let vb = nn_core::VarBuilder::zeros(DType::BF16, &dev);
    let linear1 = layers::linear(8, 16, vb.pp("l1")).unwrap();
    let rms = layers::rms_norm(16, 1e-5, vb.pp("rms")).unwrap();
    let linear2 = layers::linear(16, 8, vb.pp("l2")).unwrap();

    // Input: [batch=1, seq_len=4, hidden=8]
    let input = DynTensor::ones(&[1, 4, 8], DType::BF16, &dev).unwrap();

    let x = linear1.forward(&input).unwrap();
    assert_eq!(x.dtype(), DType::BF16);
    let x = rms.forward(&x).unwrap();
    let x = linear2.forward(&x).unwrap();

    let vals = to_f32_vals(&x);
    assert_all_finite(&vals, "bf16_llm_pattern");
    assert_eq!(vals.len(), 32); // batch=1 * seq_len=4 * hidden=8
}
