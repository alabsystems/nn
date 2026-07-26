// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LayerNorm parity tests at real model dimensions.
//!
//! Tests LayerNorm with hidden dimensions from production models:
//! - Whisper: d_model=768
//! - Qwen3: d_model=2048
//! - Large LLM: d_model=4096
//! - Silero VAD output: hidden=128
//!
//! LayerNorm precision matters because it normalizes activations before
//! every attention and FFN layer — error compounds through the model.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{LayerNorm, Module, RmsNorm};
use nn_core::test_prng::rand_f32_vec;
use nn_core::Device;

/// Tolerance for LayerNorm — variance computation is sensitive.
const TOL: f32 = 1e-4;

/// Helper: run LayerNorm on both CPU and GPU with the same data.
fn run_layernorm_parity(seed: u64, batch: usize, seq: usize, hidden: usize, label: &str) {
    let x_data = rand_f32_vec(seed, batch * seq * hidden, -2.0, 2.0);
    let w_data = rand_f32_vec(seed + 1, hidden, 0.8, 1.2);
    let b_data = rand_f32_vec(seed + 2, hidden, -0.05, 0.05);

    let run = |device: &Device| -> DynTensor {
        let x = DynTensor::new(&x_data, &[batch, seq, hidden], device).unwrap();
        let w = DynTensor::new(&w_data, &[hidden], device).unwrap();
        let b = DynTensor::new(&b_data, &[hidden], device).unwrap();
        let ln = LayerNorm::new(w, b, 1e-5).unwrap();
        ln.forward(&x).unwrap()
    };

    let cpu_out = run(&Device::Cpu);
    let gpu_out = run(&Device::metal());

    assert_eq!(gpu_out.dims(), &[batch, seq, hidden], "{label}: shape");
    assert_eq!(gpu_out.dims(), cpu_out.dims(), "{label}: shape mismatch");
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, label);
}

/// Helper: run RmsNorm on both CPU and GPU with the same data.
fn run_rmsnorm_parity(seed: u64, batch: usize, seq: usize, hidden: usize, label: &str) {
    let x_data = rand_f32_vec(seed, batch * seq * hidden, -2.0, 2.0);
    let w_data = rand_f32_vec(seed + 1, hidden, 0.8, 1.2);

    let run = |device: &Device| -> DynTensor {
        let x = DynTensor::new(&x_data, &[batch, seq, hidden], device).unwrap();
        let w = DynTensor::new(&w_data, &[hidden], device).unwrap();
        let rn = RmsNorm::new(w, 1e-5).unwrap();
        rn.forward(&x).unwrap()
    };

    let cpu_out = run(&Device::Cpu);
    let gpu_out = run(&Device::metal());

    assert_eq!(gpu_out.dims(), &[batch, seq, hidden], "{label}: shape");
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, label);
}

// -- Whisper dimensions (d_model=768) ----------------------------------------

/// Whisper encoder LayerNorm: [1, 128, 768].
#[test]
fn test_layer_norm_whisper_encoder() {
    gpu_init();
    run_layernorm_parity(5000, 1, 128, 768, "ln_whisper_enc_768");
}

/// Whisper decoder LayerNorm: [1, 64, 768].
#[test]
fn test_layer_norm_whisper_decoder() {
    gpu_init();
    run_layernorm_parity(5001, 1, 64, 768, "ln_whisper_dec_768");
}

// -- Qwen3 dimensions (d_model=2048) ----------------------------------------

/// Qwen3 self-attention pre-norm: [1, 64, 2048].
#[test]
fn test_layer_norm_qwen3() {
    gpu_init();
    run_layernorm_parity(5010, 1, 64, 2048, "ln_qwen3_2048");
}

/// Qwen3 RmsNorm (Qwen3 uses RMSNorm, not LayerNorm).
#[test]
fn test_rms_norm_qwen3() {
    gpu_init();
    run_rmsnorm_parity(5011, 1, 64, 2048, "rmsnorm_qwen3_2048");
}

// -- Large LLM dimensions (d_model=4096) ------------------------------------

/// Large LLM LayerNorm: [1, 32, 4096].
#[test]
fn test_layer_norm_large_llm() {
    gpu_init();
    run_layernorm_parity(5020, 1, 32, 4096, "ln_large_llm_4096");
}

/// Large LLM RmsNorm: [1, 32, 4096].
#[test]
fn test_rms_norm_large_llm() {
    gpu_init();
    run_rmsnorm_parity(5021, 1, 32, 4096, "rmsnorm_large_llm_4096");
}

// -- Silero VAD dimensions (hidden=128) -------------------------------------

/// Silero VAD LSTM output LayerNorm-like: [1, 128].
/// Tests 2D LayerNorm (no sequence dimension).
#[test]
fn test_layer_norm_silero_vad() {
    gpu_init();
    let hidden = 128;
    let batch = 1;

    let x_data = rand_f32_vec(5030, batch * hidden, -2.0, 2.0);
    let w_data = rand_f32_vec(5031, hidden, 0.8, 1.2);
    let b_data = rand_f32_vec(5032, hidden, -0.05, 0.05);

    let run = |device: &Device| -> DynTensor {
        let x = DynTensor::new(&x_data, &[batch, hidden], device).unwrap();
        let w = DynTensor::new(&w_data, &[hidden], device).unwrap();
        let b = DynTensor::new(&b_data, &[hidden], device).unwrap();
        let ln = LayerNorm::new(w, b, 1e-5).unwrap();
        ln.forward(&x).unwrap()
    };

    let cpu_out = run(&Device::Cpu);
    let gpu_out = run(&Device::metal());

    assert_eq!(gpu_out.dims(), &[batch, hidden]);
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "ln_silero_128");
}

// -- Batched inference -------------------------------------------------------

/// Batched LayerNorm: [4, 128, 768] (4-sample batch, Whisper scale).
#[test]
fn test_layer_norm_batched_whisper() {
    gpu_init();
    run_layernorm_parity(5040, 4, 128, 768, "ln_batched_4x128x768");
}

/// Batched RmsNorm: [4, 64, 2048] (4-sample batch, Qwen3 scale).
#[test]
fn test_rms_norm_batched_qwen3() {
    gpu_init();
    run_rmsnorm_parity(5041, 4, 64, 2048, "rmsnorm_batched_4x64x2048");
}

// -- Precision stress: narrow variance inputs --------------------------------

/// LayerNorm on near-constant input (low variance).
/// This is a precision stress test: when variance is very small, the
/// 1/sqrt(var+eps) computation is sensitive to floating-point error.
#[test]
fn test_layer_norm_low_variance() {
    gpu_init();
    let hidden = 768;
    let batch = 1;
    let seq = 32;

    // Near-constant input: base value 1.0, tiny variation.
    let x_data: Vec<f32> = rand_f32_vec(5050, batch * seq * hidden, -0.001, 0.001)
        .into_iter()
        .map(|v| 1.0 + v)
        .collect();
    let w_data = rand_f32_vec(5051, hidden, 0.8, 1.2);
    let b_data = rand_f32_vec(5052, hidden, -0.05, 0.05);

    let run = |device: &Device| -> DynTensor {
        let x = DynTensor::new(&x_data, &[batch, seq, hidden], device).unwrap();
        let w = DynTensor::new(&w_data, &[hidden], device).unwrap();
        let b = DynTensor::new(&b_data, &[hidden], device).unwrap();
        let ln = LayerNorm::new(w, b, 1e-5).unwrap();
        ln.forward(&x).unwrap()
    };

    let cpu_out = run(&Device::Cpu);
    let gpu_out = run(&Device::metal());

    // Wider tolerance for low-variance case due to amplified numerical sensitivity.
    assert_gpu_cpu_close(&gpu_out, &cpu_out, 5e-4, "ln_low_variance");
}
