// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Whisper model GPU forward pass tests.
//!
//! Verifies the full Whisper encoder-decoder model runs on Metal GPU tensors.
//! Uses VarBuilder::zeros for deterministic weights and a tiny config
//! (1 encoder layer, 1 decoder layer, 2 heads, d_model=16) to keep tests fast.
//!
//! CPU vs GPU comparison validates the DynTensor->GpuBackend->Metal path
//! for all Whisper components: Conv1d stem, sinusoidal positional embeddings,
//! encoder transformer blocks (self-attention + FFN + LayerNorm), decoder
//! transformer blocks (self-attention + cross-attention + FFN + LayerNorm),
//! token embedding, learned positional embedding, and KV cache.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, VarBuilder};
use nn_whisper::{WhisperConfig, WhisperModel};

const TOL: f32 = 1e-3;

fn init() {
    gpu_init();
}

/// Minimal config for GPU tests: 1 encoder layer, 1 decoder layer, 2 heads.
fn tiny_gpu_config() -> WhisperConfig {
    nn_whisper::test_utils::tiny_config()
}

fn assert_close(gpu_result: &DynTensor, cpu_result: &DynTensor, label: &str) {
    assert_gpu_cpu_close(gpu_result, cpu_result, TOL, label);
}

// -- Whisper encoder forward pass completes on GPU ----------------------------

#[test]
fn test_whisper_encoder_forward_gpu() {
    init();
    let config = tiny_gpu_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::metal());
    let mut model = WhisperModel::load(&vb, config.clone()).expect("GPU model load");

    // Input: [1, num_mel_bins, 16] on GPU.
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &Device::metal())
        .expect("mel tensor");

    let result = model.encode(&mel);
    assert!(
        result.is_ok(),
        "Whisper encoder forward on GPU should succeed: {result:?}"
    );

    let out = result.unwrap();
    assert_eq!(out.rank(), 3, "encoder output should be rank 3");
    assert_eq!(out.dim(0).unwrap(), 1, "batch dim");
    assert_eq!(out.dim(2).unwrap(), config.d_model, "d_model dim");
    assert_eq!(
        out.device(),
        Device::metal(),
        "encoder output should stay on GPU"
    );
}

// -- Whisper decoder forward pass completes on GPU ----------------------------

#[test]
fn test_whisper_decoder_forward_gpu() {
    init();
    let config = tiny_gpu_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::metal());
    let mut model = WhisperModel::load(&vb, config.clone()).expect("GPU model load");

    // Fake encoder output on GPU: [1, 8, d_model].
    let encoder_output = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &Device::metal())
        .expect("encoder output tensor");

    // Token IDs on GPU: [1, 3] (3 tokens).
    let tokens = DynTensor::new(&[0.0, 1.0, 2.0], &[1, 3], &Device::metal()).expect("token tensor");

    let result = model.decode(&tokens, &encoder_output, true, 0);
    assert!(
        result.is_ok(),
        "Whisper decoder forward on GPU should succeed: {result:?}"
    );

    let logits = result.unwrap();
    assert_eq!(logits.rank(), 3, "logits should be rank 3");
    assert_eq!(logits.dim(0).unwrap(), 1, "batch dim");
    assert_eq!(logits.dim(1).unwrap(), 3, "seq_len dim");
    assert_eq!(logits.dim(2).unwrap(), config.vocab_size, "vocab_size dim");
    assert_eq!(
        logits.device(),
        Device::metal(),
        "logits should stay on GPU"
    );
}

// -- Encoder CPU vs GPU correctness comparison --------------------------------

#[test]
fn test_whisper_encoder_cpu_gpu_match() {
    init();
    let config = tiny_gpu_config();

    let mel_frames = 16;

    // CPU reference
    let vb_cpu = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let mut model_cpu = WhisperModel::load(&vb_cpu, config.clone()).expect("CPU model load");
    let mel_cpu = DynTensor::zeros(
        &[1, config.num_mel_bins, mel_frames],
        DType::F32,
        &Device::Cpu,
    )
    .expect("CPU mel tensor");
    let enc_cpu = model_cpu.encode(&mel_cpu).expect("CPU encode");

    // GPU
    let vb_gpu = VarBuilder::zeros(DType::F32, &Device::metal());
    let mut model_gpu = WhisperModel::load(&vb_gpu, config.clone()).expect("GPU model load");
    let mel_gpu = DynTensor::zeros(
        &[1, config.num_mel_bins, mel_frames],
        DType::F32,
        &Device::metal(),
    )
    .expect("GPU mel tensor");
    let enc_gpu = model_gpu.encode(&mel_gpu).expect("GPU encode");

    assert_close(&enc_gpu, &enc_cpu, "whisper_encoder_cpu_gpu");
}

// -- Decoder CPU vs GPU correctness comparison --------------------------------

#[test]
fn test_whisper_decoder_cpu_gpu_match() {
    init();
    let config = tiny_gpu_config();

    let token_data = [0.0f32, 1.0, 2.0];
    let token_shape = [1, 3];
    let enc_seq_len = 8;

    // CPU reference
    let vb_cpu = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let mut model_cpu = WhisperModel::load(&vb_cpu, config.clone()).expect("CPU model load");
    let enc_out_cpu = DynTensor::zeros(&[1, enc_seq_len, config.d_model], DType::F32, &Device::Cpu)
        .expect("CPU encoder output");
    let tokens_cpu = DynTensor::new(&token_data, &token_shape, &Device::Cpu).expect("CPU tokens");
    let logits_cpu = model_cpu
        .decode(&tokens_cpu, &enc_out_cpu, true, 0)
        .expect("CPU decode");

    // GPU
    let vb_gpu = VarBuilder::zeros(DType::F32, &Device::metal());
    let mut model_gpu = WhisperModel::load(&vb_gpu, config.clone()).expect("GPU model load");
    let enc_out_gpu = DynTensor::zeros(
        &[1, enc_seq_len, config.d_model],
        DType::F32,
        &Device::metal(),
    )
    .expect("GPU encoder output");
    let tokens_gpu =
        DynTensor::new(&token_data, &token_shape, &Device::metal()).expect("GPU tokens");
    let logits_gpu = model_gpu
        .decode(&tokens_gpu, &enc_out_gpu, true, 0)
        .expect("GPU decode");

    assert_close(&logits_gpu, &logits_cpu, "whisper_decoder_cpu_gpu");
}

// -- Full encode->decode round-trip on GPU ------------------------------------

#[test]
fn test_whisper_encode_decode_gpu_roundtrip() {
    init();
    let config = tiny_gpu_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::metal());
    let mut model = WhisperModel::load(&vb, config.clone()).expect("GPU model load");

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &Device::metal())
        .expect("mel tensor");

    let encoder_out = model.encode(&mel).expect("encoder forward");

    let tokens = DynTensor::new(&[0.0, 1.0], &[1, 2], &Device::metal()).expect("token tensor");

    let logits = model
        .decode(&tokens, &encoder_out, true, 0)
        .expect("decoder forward");

    assert_eq!(logits.dim(1).unwrap(), 2);
    assert_eq!(logits.dim(2).unwrap(), config.vocab_size);
    assert_eq!(logits.device(), Device::metal());
}

// -- Full encode->decode CPU vs GPU parity ------------------------------------

#[test]
fn test_whisper_encode_decode_cpu_gpu_parity() {
    init();
    let config = tiny_gpu_config();

    let mel_frames = 16;
    let token_data = [0.0f32, 1.0];
    let token_shape = [1, 2];

    // CPU pipeline
    let vb_cpu = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let mut model_cpu = WhisperModel::load(&vb_cpu, config.clone()).expect("CPU model load");
    let mel_cpu = DynTensor::zeros(
        &[1, config.num_mel_bins, mel_frames],
        DType::F32,
        &Device::Cpu,
    )
    .expect("CPU mel");
    let enc_cpu = model_cpu.encode(&mel_cpu).expect("CPU encode");
    let tokens_cpu = DynTensor::new(&token_data, &token_shape, &Device::Cpu).expect("CPU tokens");
    let logits_cpu = model_cpu
        .decode(&tokens_cpu, &enc_cpu, true, 0)
        .expect("CPU decode");

    // GPU pipeline
    let vb_gpu = VarBuilder::zeros(DType::F32, &Device::metal());
    let mut model_gpu = WhisperModel::load(&vb_gpu, config.clone()).expect("GPU model load");
    let mel_gpu = DynTensor::zeros(
        &[1, config.num_mel_bins, mel_frames],
        DType::F32,
        &Device::metal(),
    )
    .expect("GPU mel");
    let enc_gpu = model_gpu.encode(&mel_gpu).expect("GPU encode");
    let tokens_gpu =
        DynTensor::new(&token_data, &token_shape, &Device::metal()).expect("GPU tokens");
    let logits_gpu = model_gpu
        .decode(&tokens_gpu, &enc_gpu, true, 0)
        .expect("GPU decode");

    assert_close(&logits_gpu, &logits_cpu, "whisper_encode_decode_cpu_gpu");
}

// -- Single token decode on GPU -----------------------------------------------

#[test]
fn test_whisper_single_token_decode_gpu() {
    init();
    let config = tiny_gpu_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::metal());
    let mut model = WhisperModel::load(&vb, config.clone()).expect("GPU model load");

    let encoder_out = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &Device::metal())
        .expect("encoder output");

    let tokens = DynTensor::new(&[0.0], &[1, 1], &Device::metal()).expect("single token");

    let logits = model
        .decode(&tokens, &encoder_out, true, 0)
        .expect("single token decode");
    assert_eq!(logits.dims(), &[1, 1, config.vocab_size]);
}

// -- Varying mel input lengths ------------------------------------------------

#[test]
fn test_whisper_varying_mel_lengths_gpu() {
    init();
    let config = tiny_gpu_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::metal());

    // Conv1d stem has kernel_size=3 and stride=1 then stride=2, so mel frames
    // must be at least large enough for the convolution. Test several lengths.
    for mel_frames in [4, 8, 16, 32] {
        let mut model = WhisperModel::load(&vb, config.clone()).expect("GPU model load");

        let mel = DynTensor::zeros(
            &[1, config.num_mel_bins, mel_frames],
            DType::F32,
            &Device::metal(),
        )
        .expect("mel tensor");

        let result = model.encode(&mel);
        assert!(
            result.is_ok(),
            "Whisper encoder failed for mel_frames={mel_frames}: {result:?}"
        );

        let out = result.unwrap();
        assert_eq!(out.rank(), 3, "rank mismatch for mel_frames={mel_frames}");
        assert_eq!(out.dim(0).unwrap(), 1, "batch dim");
        assert_eq!(
            out.dim(2).unwrap(),
            config.d_model,
            "d_model dim for mel_frames={mel_frames}"
        );
    }
}

// -- KV cache on GPU ----------------------------------------------------------

#[test]
fn test_whisper_kv_cache_gpu() {
    init();
    let config = tiny_gpu_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::metal());
    let mut model = WhisperModel::load(&vb, config.clone()).expect("GPU model load");

    let encoder_out = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &Device::metal())
        .expect("encoder output");

    // First step: flush cache.
    let t1 = DynTensor::new(&[0.0], &[1, 1], &Device::metal()).expect("t1");
    let logits1 = model
        .decode(&t1, &encoder_out, true, 0)
        .expect("first decode step");

    // Second step: reuse cache.
    let t2 = DynTensor::new(&[1.0], &[1, 1], &Device::metal()).expect("t2");
    let logits2 = model
        .decode(&t2, &encoder_out, false, 1)
        .expect("second decode step (cached)");

    assert_eq!(logits1.dim(2).unwrap(), config.vocab_size);
    assert_eq!(logits2.dim(2).unwrap(), config.vocab_size);
    assert_eq!(logits1.device(), Device::metal());
    assert_eq!(logits2.device(), Device::metal());
}

// -- KV cache CPU vs GPU parity (autoregressive) ------------------------------

#[test]
fn test_whisper_kv_cache_cpu_gpu_parity() {
    init();
    let config = tiny_gpu_config();

    let enc_seq_len = 8;

    // CPU path
    let vb_cpu = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let mut model_cpu = WhisperModel::load(&vb_cpu, config.clone()).expect("CPU model load");
    let enc_out_cpu = DynTensor::zeros(&[1, enc_seq_len, config.d_model], DType::F32, &Device::Cpu)
        .expect("CPU encoder output");
    let t0_cpu = DynTensor::new(&[0.0], &[1, 1], &Device::Cpu).expect("CPU t0");
    let _cpu_0 = model_cpu
        .decode(&t0_cpu, &enc_out_cpu, true, 0)
        .expect("CPU decode step 0");
    let t1_cpu = DynTensor::new(&[1.0], &[1, 1], &Device::Cpu).expect("CPU t1");
    let cpu_1 = model_cpu
        .decode(&t1_cpu, &enc_out_cpu, false, 1)
        .expect("CPU decode step 1");

    // GPU path
    let vb_gpu = VarBuilder::zeros(DType::F32, &Device::metal());
    let mut model_gpu = WhisperModel::load(&vb_gpu, config.clone()).expect("GPU model load");
    let enc_out_gpu = DynTensor::zeros(
        &[1, enc_seq_len, config.d_model],
        DType::F32,
        &Device::metal(),
    )
    .expect("GPU encoder output");
    let t0_gpu = DynTensor::new(&[0.0], &[1, 1], &Device::metal()).expect("GPU t0");
    let _gpu_0 = model_gpu
        .decode(&t0_gpu, &enc_out_gpu, true, 0)
        .expect("GPU decode step 0");
    let t1_gpu = DynTensor::new(&[1.0], &[1, 1], &Device::metal()).expect("GPU t1");
    let gpu_1 = model_gpu
        .decode(&t1_gpu, &enc_out_gpu, false, 1)
        .expect("GPU decode step 1");

    assert_close(&gpu_1, &cpu_1, "whisper_kv_cache_step1_cpu_gpu");
}

// -- KV cache reset produces identical results --------------------------------

#[test]
fn test_whisper_kv_cache_reset_determinism_gpu() {
    init();
    let config = tiny_gpu_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::metal());
    let mut model = WhisperModel::load(&vb, config.clone()).expect("GPU model load");

    let encoder_out = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &Device::metal())
        .expect("encoder output");

    let tokens = DynTensor::new(&[0.0], &[1, 1], &Device::metal()).expect("token tensor");

    // First decode
    let logits1 = model
        .decode(&tokens, &encoder_out, true, 0)
        .expect("first decode");

    // Reset and decode again
    model.reset_kv_cache();
    let logits2 = model
        .decode(&tokens, &encoder_out, true, 0)
        .expect("second decode after reset");

    // Both runs should produce identical results
    assert_close(&logits1, &logits2, "whisper_kv_cache_reset_determinism");
}
