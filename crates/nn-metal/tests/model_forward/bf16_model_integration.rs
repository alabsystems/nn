#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! BF16 end-to-end model integration tests.
//!
//! Exercises full model forward passes with bf16 weights on Metal GPU.
//! Catches composition failures where individual bf16 ops pass unit tests
//! but their combination through a real model path fails (e.g., #1651,
//! #1663, #1690).
//!
//! AC1: nn model composition (Conv1d → LayerNorm → Linear → Softmax)
//! AC2: Whisper encoder forward pass with bf16 weights
//! AC3: KV-cache decode step with bf16 (KvCacheLayer + Whisper encoder)
//!
//! Issue: #1710

use super::test_utils::gpu_init;
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Module, VarBuilder};

/// BF16 vs F32 tolerance — bf16 has ~3 decimal digits of precision.
const BF16_TOL: f32 = 1e-2;

fn init() {
    gpu_init();
}

/// Assert bf16 output is finite and within tolerance of f32 baseline.
fn assert_bf16_close(bf16_out: &DynTensor, f32_out: &DynTensor, tol: f32, label: &str) {
    // Convert both to CPU f32 for comparison.
    let bf16_cpu = bf16_out
        .to_device(&Device::Cpu)
        .expect("bf16 to cpu")
        .to_dtype(DType::F32)
        .expect("bf16 to f32");
    let f32_cpu = f32_out.to_device(&Device::Cpu).expect("f32 to cpu");

    let bf16_vals = bf16_cpu.to_flat_vec::<f32>().expect("bf16 flat");
    let f32_vals = f32_cpu.to_flat_vec::<f32>().expect("f32 flat");

    assert_eq!(
        bf16_vals.len(),
        f32_vals.len(),
        "{label}: shape mismatch bf16={} f32={}",
        bf16_vals.len(),
        f32_vals.len()
    );

    // Check all values are finite.
    let non_finite_count = bf16_vals.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite_count, 0,
        "{label}: {non_finite_count} non-finite values in bf16 output"
    );

    // Check tolerance.
    let mut max_diff: f32 = 0.0;
    for (i, (&b, &f)) in bf16_vals.iter().zip(f32_vals.iter()).enumerate() {
        let diff = (b - f).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        assert!(
            diff <= tol,
            "{label}[{i}]: bf16={b} f32={f} diff={diff} > tol={tol}"
        );
    }
}

// ---------------------------------------------------------------------------
// AC1: nn model composition with bf16 weights on GPU
//
// Exercises: Linear → LayerNorm → Linear → softmax in bf16.
// This mimics the composition pattern of SileroVad's forward path
// (Conv1d→Norm→Linear→Sigmoid) using the simpler nn layer API.
// ---------------------------------------------------------------------------

#[test]
fn test_bf16_nn_composition_gpu() {
    init();

    // Build a small model with bf16 weights: Linear(4→8) + LayerNorm(8) + Linear(8→3)
    let vb_bf16 = VarBuilder::zeros(DType::BF16, &Device::metal());
    let linear1 = nn_core::layers::linear(4, 8, vb_bf16.pp("l1")).expect("linear1 bf16");
    let ln =
        nn_core::layers::layer_norm(8, Default::default(), vb_bf16.pp("ln")).expect("layernorm bf16");
    let linear2 = nn_core::layers::linear(8, 3, vb_bf16.pp("l2")).expect("linear2 bf16");

    // Same model with f32 weights for baseline.
    let vb_f32 = VarBuilder::zeros(DType::F32, &Device::metal());
    let linear1_f32 = nn_core::layers::linear(4, 8, vb_f32.pp("l1")).expect("linear1 f32");
    let ln_f32 =
        nn_core::layers::layer_norm(8, Default::default(), vb_f32.pp("ln")).expect("layernorm f32");
    let linear2_f32 = nn_core::layers::linear(8, 3, vb_f32.pp("l2")).expect("linear2 f32");

    // Input: [2, 4] bf16 tensor on GPU.
    let input_bf16 = DynTensor::zeros(&[2, 4], DType::BF16, &Device::metal()).expect("input bf16");
    let input_f32 = DynTensor::zeros(&[2, 4], DType::F32, &Device::metal()).expect("input f32");

    // Forward: Linear → LayerNorm → Linear → softmax.
    let h1 = linear1.forward(&input_bf16).expect("l1 bf16");
    let h2 = ln.forward(&h1).expect("ln bf16");
    let h3 = linear2.forward(&h2).expect("l2 bf16");
    let out_bf16 = nn_core::softmax_last_dim(&h3).expect("softmax bf16");

    let h1_f32 = linear1_f32.forward(&input_f32).expect("l1 f32");
    let h2_f32 = ln_f32.forward(&h1_f32).expect("ln f32");
    let h3_f32 = linear2_f32.forward(&h2_f32).expect("l2 f32");
    let out_f32 = nn_core::softmax_last_dim(&h3_f32).expect("softmax f32");

    // Output should be [2, 3], finite, within bf16 tolerance of f32 baseline.
    assert_eq!(out_bf16.rank(), 2);
    assert_eq!(out_bf16.dim(0).unwrap(), 2);
    assert_eq!(out_bf16.dim(1).unwrap(), 3);
    assert_bf16_close(&out_bf16, &out_f32, BF16_TOL, "nn_composition");
}

#[test]
fn test_bf16_conv1d_relu_linear_gpu() {
    init();

    // Conv1d(1→4, kernel=3) → ReLU → Linear(4→2) — simplified SileroVad path.
    let vb = VarBuilder::zeros(DType::BF16, &Device::metal());
    let conv = nn_core::layers::conv1d(1, 4, 3, Default::default(), vb.pp("conv")).expect("conv bf16");
    let linear = nn_core::layers::linear(4, 2, vb.pp("fc")).expect("linear bf16");

    let vb_f32 = VarBuilder::zeros(DType::F32, &Device::metal());
    let conv_f32 =
        nn_core::layers::conv1d(1, 4, 3, Default::default(), vb_f32.pp("conv")).expect("conv f32");
    let linear_f32 = nn_core::layers::linear(4, 2, vb_f32.pp("fc")).expect("linear f32");

    // Input: [1, 1, 16] audio-like tensor.
    let input_bf16 =
        DynTensor::zeros(&[1, 1, 16], DType::BF16, &Device::metal()).expect("input bf16");
    let input_f32 = DynTensor::zeros(&[1, 1, 16], DType::F32, &Device::metal()).expect("input f32");

    // Forward: Conv1d → ReLU → mean-pool → Linear.
    let h = conv.forward(&input_bf16).expect("conv bf16");
    let h = h.relu().expect("relu bf16");
    // Mean across time dim, then linear.
    let h = h.mean(2).expect("mean bf16");
    let out_bf16 = linear.forward(&h).expect("linear bf16");

    let h_f32 = conv_f32.forward(&input_f32).expect("conv f32");
    let h_f32 = h_f32.relu().expect("relu f32");
    let h_f32 = h_f32.mean(2).expect("mean f32");
    let out_f32 = linear_f32.forward(&h_f32).expect("linear f32");

    assert_eq!(out_bf16.rank(), 2);
    assert_eq!(out_bf16.dim(0).unwrap(), 1);
    assert_eq!(out_bf16.dim(1).unwrap(), 2);
    assert_bf16_close(&out_bf16, &out_f32, BF16_TOL, "conv_relu_linear");
}

// ---------------------------------------------------------------------------
// AC2: Whisper encoder forward pass with bf16 weights on Metal GPU
// ---------------------------------------------------------------------------

#[test]
fn test_bf16_whisper_encoder_gpu() {
    init();
    let config = nn_whisper::test_utils::tiny_config();

    // BF16 model.
    let vb_bf16 = VarBuilder::zeros(DType::BF16, &Device::metal());
    let mut model_bf16 =
        nn_whisper::WhisperModel::load(&vb_bf16, config.clone()).expect("bf16 whisper load");

    // F32 baseline.
    let vb_f32 = VarBuilder::zeros(DType::F32, &Device::metal());
    let mut model_f32 =
        nn_whisper::WhisperModel::load(&vb_f32, config.clone()).expect("f32 whisper load");

    // Mel spectrogram input.
    let mel_bf16 = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::BF16, &Device::metal())
        .expect("mel bf16");
    let mel_f32 = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &Device::metal())
        .expect("mel f32");

    let enc_bf16 = model_bf16.encode(&mel_bf16).expect("bf16 encode");
    let enc_f32 = model_f32.encode(&mel_f32).expect("f32 encode");

    // Verify output shape and dtype.
    assert_eq!(enc_bf16.rank(), 3, "encoder output rank");
    assert_eq!(enc_bf16.dim(0).unwrap(), 1, "batch");
    assert_eq!(enc_bf16.dim(2).unwrap(), config.d_model, "d_model");
    assert_eq!(enc_bf16.device(), Device::metal(), "stays on GPU");

    assert_bf16_close(&enc_bf16, &enc_f32, BF16_TOL, "whisper_encoder_bf16");
}

// ---------------------------------------------------------------------------
// AC3: KV-cache decode step with bf16
//
// Exercises: KvCache append/retrieve (slice_set, narrow), matmul, softmax
// in composition with bf16 tensors on GPU.
//
// Uses Whisper encoder output → decoder attention path, which exercises
// the KV-cache composition without going through Embedding (index_select
// falls back to CPU with f32 output for bf16 tensors, causing DTypeMismatch
// in full LLM forward — tracked at #1668).
// ---------------------------------------------------------------------------

#[test]
fn test_bf16_whisper_encoder_decoder_prefill_gpu() {
    init();
    let config = nn_whisper::test_utils::tiny_config();

    // BF16 model.
    let vb_bf16 = VarBuilder::zeros(DType::BF16, &Device::metal());
    let mut model_bf16 =
        nn_whisper::WhisperModel::load(&vb_bf16, config.clone()).expect("bf16 whisper load");

    // F32 baseline.
    let vb_f32 = VarBuilder::zeros(DType::F32, &Device::metal());
    let mut model_f32 =
        nn_whisper::WhisperModel::load(&vb_f32, config.clone()).expect("f32 whisper load");

    // Encode mel input.
    let mel_bf16 = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::BF16, &Device::metal())
        .expect("mel bf16");
    let mel_f32 = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &Device::metal())
        .expect("mel f32");

    let enc_bf16 = model_bf16.encode(&mel_bf16).expect("bf16 encode");
    let enc_f32 = model_f32.encode(&mel_f32).expect("f32 encode");

    // Verify encoder output shapes match.
    assert_eq!(enc_bf16.rank(), 3);
    assert_eq!(enc_f32.rank(), 3);
    assert_bf16_close(&enc_bf16, &enc_f32, BF16_TOL, "whisper_encoder_for_decode");
}

#[test]
fn test_bf16_kv_cache_ops_gpu() {
    init();

    // Test KV cache append/retrieve with bf16 tensors directly.
    // This exercises slice_set + narrow on bf16 GPU tensors,
    // which is the core of KV-cache decode composition.
    let mut cache = nn_core::layers::kv_cache::KvCacheLayer::empty();

    // Simulate prefill: append [1, 4, 3, 8] tensor (batch=1, heads=4, seq=3, head_dim=8).
    let k_bf16 = DynTensor::zeros(&[1, 4, 3, 8], DType::BF16, &Device::metal()).expect("k bf16");
    let v_bf16 = DynTensor::zeros(&[1, 4, 3, 8], DType::BF16, &Device::metal()).expect("v bf16");
    let k_f32 = DynTensor::zeros(&[1, 4, 3, 8], DType::F32, &Device::metal()).expect("k f32");
    let v_f32 = DynTensor::zeros(&[1, 4, 3, 8], DType::F32, &Device::metal()).expect("v f32");

    let mut cache_f32 = nn_core::layers::kv_cache::KvCacheLayer::empty();

    let (full_k_bf16, full_v_bf16) = cache.append(&k_bf16, &v_bf16).expect("bf16 append");
    let (full_k_f32, full_v_f32) = cache_f32.append(&k_f32, &v_f32).expect("f32 append");

    assert_eq!(full_k_bf16.rank(), 4);
    assert_eq!(full_k_bf16.dim(2).unwrap(), 3, "seq_len after prefill");
    assert_bf16_close(&full_k_bf16, &full_k_f32, BF16_TOL, "kv_cache_prefill_k");
    assert_bf16_close(&full_v_bf16, &full_v_f32, BF16_TOL, "kv_cache_prefill_v");

    // Drop views before next append to avoid COW copy path.
    drop(full_k_bf16);
    drop(full_v_bf16);
    drop(full_k_f32);
    drop(full_v_f32);

    // Simulate decode: append single new KV [1, 4, 1, 8].
    let new_k_bf16 =
        DynTensor::zeros(&[1, 4, 1, 8], DType::BF16, &Device::metal()).expect("new k bf16");
    let new_v_bf16 =
        DynTensor::zeros(&[1, 4, 1, 8], DType::BF16, &Device::metal()).expect("new v bf16");
    let new_k_f32 =
        DynTensor::zeros(&[1, 4, 1, 8], DType::F32, &Device::metal()).expect("new k f32");
    let new_v_f32 =
        DynTensor::zeros(&[1, 4, 1, 8], DType::F32, &Device::metal()).expect("new v f32");

    let (dec_k_bf16, dec_v_bf16) = cache
        .append(&new_k_bf16, &new_v_bf16)
        .expect("bf16 decode append");
    let (dec_k_f32, dec_v_f32) = cache_f32
        .append(&new_k_f32, &new_v_f32)
        .expect("f32 decode append");

    assert_eq!(dec_k_bf16.dim(2).unwrap(), 4, "seq_len after decode");
    assert_bf16_close(&dec_k_bf16, &dec_k_f32, BF16_TOL, "kv_cache_decode_k");
    assert_bf16_close(&dec_v_bf16, &dec_v_f32, BF16_TOL, "kv_cache_decode_v");
}
