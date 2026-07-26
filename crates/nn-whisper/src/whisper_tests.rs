#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for nn-whisper.

use crate::config::WhisperConfig;
use crate::positional::{causal_mask, sinusoidal_embedding};
use crate::test_utils::tiny_config;
use crate::WhisperModel;
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};

// -- Config tests --

#[test]
fn test_large_v3_turbo_config() {
    let config = WhisperConfig::large_v3_turbo();
    assert_eq!(config.d_model, 1280);
    assert_eq!(config.encoder_layers, 32);
    assert_eq!(config.decoder_layers, 4);
    assert_eq!(config.vocab_size, 51866);
    assert_eq!(config.encoder_head_dim(), 64); // 1280 / 20
    assert_eq!(config.decoder_head_dim(), 64);
}

#[test]
fn test_default_config_is_turbo() {
    let config = WhisperConfig::default();
    assert_eq!(config.d_model, 1280);
    assert_eq!(config.decoder_layers, 4);
}

// -- Positional encoding tests --
// (Additional tests are in positional.rs inline tests)

#[test]
fn test_sinusoidal_embedding_shape_large() {
    let emb = sinusoidal_embedding(1500, 1280, DType::F32, &cpu()).unwrap();
    assert_eq!(emb.dims(), &[1500, 1280]);
}

#[test]
fn test_sinusoidal_embedding_values_bounded() {
    let emb = sinusoidal_embedding(100, 64, DType::F32, &cpu()).unwrap();
    let flat = emb.to_flat_vec::<f32>().unwrap();
    for &v in &flat {
        assert!(v.is_finite());
        assert!((-1.0..=1.0).contains(&v));
    }
}

#[test]
fn test_causal_mask_diagonal() {
    let mask = causal_mask(3, DType::F32, &cpu()).unwrap();
    let flat = mask.to_flat_vec::<f32>().unwrap();
    // Diagonal elements should be 0 (can attend to self).
    for i in 0..3 {
        assert_eq!(flat[i * 3 + i], 0.0);
    }
}

// -- Model construction tests with ZerosBackend --

#[test]
fn test_model_load_zeros_backend() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let model = WhisperModel::load(&vb, config);
    assert!(
        model.is_ok(),
        "model load should succeed with zeros backend"
    );
}

#[test]
fn test_encoder_forward_zeros() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    // Input: [1, num_mel_bins, 16] (enough for conv stride-2 to produce >0 frames).
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();

    let result = model.encode(&mel);
    assert!(result.is_ok(), "encoder forward should succeed: {result:?}");

    let out = result.unwrap();
    assert_eq!(out.rank(), 3); // [batch, seq_len, d_model]
    assert_eq!(out.dim(0).unwrap(), 1); // batch
    assert_eq!(out.dim(2).unwrap(), config.d_model); // d_model
}

#[test]
fn test_encoder_forward_bf16_model_f32_mel() {
    // BF16 model with F32 mel input — encode() should auto-convert (#1721).
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::BF16, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();
    assert_eq!(model.dtype(), DType::BF16);

    // Mel from whisper_mel_spectrogram() is always F32.
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();
    let result = model.encode(&mel);
    assert!(
        result.is_ok(),
        "bf16 encode with f32 mel should succeed: {result:?}"
    );
    let out = result.unwrap();
    assert_eq!(out.rank(), 3);
    assert_eq!(out.dim(0).unwrap(), 1);
    assert_eq!(out.dim(2).unwrap(), config.d_model);
}

#[test]
fn test_encoder_forward_bf16_model_bf16_mel() {
    // BF16 model with BF16 mel input — should work without conversion.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::BF16, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::BF16, &cpu()).unwrap();
    let result = model.encode(&mel);
    assert!(
        result.is_ok(),
        "bf16 encode with bf16 mel should succeed: {result:?}"
    );
}

#[test]
fn test_f32_model_encode_unchanged() {
    // F32 model — mel is not converted (regression test for #1721).
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();
    assert_eq!(model.dtype(), DType::F32);

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();
    let result = model.encode(&mel);
    assert!(result.is_ok(), "f32 encode should still work: {result:?}");
}

#[test]
fn test_decoder_forward_zeros() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    // Fake encoder output: [1, 8, d_model]
    let encoder_output = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).unwrap();

    // Token IDs: [1, 3] (3 tokens, f32-encoded indices 0, 1, 2).
    let tokens = DynTensor::new(&[0.0, 1.0, 2.0], &[1, 3], &cpu()).unwrap();

    let result = model.decode(&tokens, &encoder_output, true, 0);
    assert!(result.is_ok(), "decoder forward should succeed: {result:?}");

    let logits = result.unwrap();
    assert_eq!(logits.rank(), 3); // [batch, seq_len, vocab_size]
    assert_eq!(logits.dim(0).unwrap(), 1);
    assert_eq!(logits.dim(1).unwrap(), 3);
    assert_eq!(logits.dim(2).unwrap(), config.vocab_size);
}

#[test]
fn test_encode_decode_roundtrip() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();

    let encoder_out = model.encode(&mel).unwrap();

    let tokens = DynTensor::new(&[0.0, 1.0], &[1, 2], &cpu()).unwrap();

    let logits = model.decode(&tokens, &encoder_out, true, 0).unwrap();
    assert_eq!(logits.dim(1).unwrap(), 2); // seq_len matches input tokens
    assert_eq!(logits.dim(2).unwrap(), config.vocab_size);
}

#[test]
fn test_kv_cache_consistency() {
    // Cross-attention KV cache: second call without flush should reuse cache.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    let encoder_out = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).unwrap();

    let t1 = DynTensor::new(&[0.0], &[1, 1], &cpu()).unwrap();

    // First step: flush cache.
    let logits1 = model.decode(&t1, &encoder_out, true, 0).unwrap();

    // Second step: don't flush.
    let t2 = DynTensor::new(&[1.0], &[1, 1], &cpu()).unwrap();
    let logits2 = model.decode(&t2, &encoder_out, false, 1).unwrap();

    assert_eq!(logits1.dim(2).unwrap(), config.vocab_size);
    assert_eq!(logits2.dim(2).unwrap(), config.vocab_size);
}

#[test]
fn test_reset_kv_cache() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config).unwrap();
    // Should not panic.
    model.reset_kv_cache();
}

// -- Attention tests --

#[test]
fn test_attention_scale_factor() {
    // Whisper uses (head_dim)^{-0.25} scale on both Q and K.
    let head_dim = 64usize;
    let scale = (head_dim as f64).powf(-0.25);
    assert!((scale - 0.353_553_390_593_273_8).abs() < 1e-10);
    assert!(scale.is_finite());
}

#[test]
fn test_attention_scale_factor_small() {
    let head_dim = 8usize;
    let scale = (head_dim as f64).powf(-0.25);
    assert!(scale.is_finite());
    assert!(scale > 0.0);
}

// -- Structural tests --

#[test]
fn test_config_head_dim_division() {
    let config = WhisperConfig::large_v3_turbo();
    // Verify d_model is divisible by heads.
    assert_eq!(config.d_model % config.encoder_attention_heads, 0);
    assert_eq!(config.d_model % config.decoder_attention_heads, 0);
}

#[test]
fn test_audio_constants() {
    use crate::config::*;
    assert_eq!(N_SAMPLES, 480_000);
    assert_eq!(N_FRAMES, 3_000);
    assert_eq!(SAMPLE_RATE, 16_000);
}

// -- KV cache and encoder shape tests (extracted to whisper_tests_kv_cache.rs) --
#[path = "whisper_tests_kv_cache.rs"]
mod kv_cache;

// -- Config validation tests (extracted to whisper_tests_config_validation.rs) --
#[path = "whisper_tests_config_validation.rs"]
mod config_validation;

// -- Finiteness check tests (#1444) --

#[test]
fn test_encoder_catches_nan_in_output() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    // Inject NaN into mel input — propagates through zero-weight Conv1d (0 * NaN = NaN).
    let mut mel_data = vec![0.0f32; config.num_mel_bins * 16];
    mel_data[0] = f32::NAN;
    let mel = DynTensor::new(&mel_data, &[1, config.num_mel_bins, 16], &cpu()).unwrap();

    let result = model.encode(&mel);
    assert!(
        result.is_err(),
        "encoder forward should return error when input contains NaN"
    );
}

#[test]
fn test_decoder_catches_nan_in_output() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    // Inject NaN into encoder output — propagates through decoder cross-attention.
    let mut enc_data = vec![0.0f32; 8 * config.d_model];
    enc_data[0] = f32::NAN;
    let encoder_output = DynTensor::new(&enc_data, &[1, 8, config.d_model], &cpu()).unwrap();

    let tokens = DynTensor::new(&[0.0], &[1, 1], &cpu()).unwrap();
    let result = model.decode(&tokens, &encoder_output, true, 0);
    assert!(
        result.is_err(),
        "decoder forward should return error when encoder output contains NaN"
    );
}

// -- WhisperConfig with_* builder and preset tests --

#[test]
fn test_with_methods_chain() {
    let config = WhisperConfig::whisper_tiny()
        .with_d_model(512)
        .with_encoder_layers(6)
        .with_decoder_layers(2);
    assert_eq!(config.d_model, 512);
    assert_eq!(config.encoder_layers, 6);
    assert_eq!(config.decoder_layers, 2);
    // Unchanged fields retain preset values.
    assert_eq!(config.num_mel_bins, 80);
    assert_eq!(config.vocab_size, 51865);
}

#[test]
fn test_with_methods_validate() {
    let config = WhisperConfig::whisper_tiny()
        .with_d_model(512)
        .with_encoder_attention_heads(8)
        .with_decoder_attention_heads(8);
    config.validate().expect("chained config should be valid");
}

#[test]
fn test_preset_configs_valid() {
    WhisperConfig::whisper_tiny()
        .validate()
        .expect("tiny valid");
    WhisperConfig::whisper_base()
        .validate()
        .expect("base valid");
    WhisperConfig::whisper_small()
        .validate()
        .expect("small valid");
    WhisperConfig::whisper_medium()
        .validate()
        .expect("medium valid");
    WhisperConfig::whisper_large_v2()
        .validate()
        .expect("large-v2 valid");
    WhisperConfig::large_v3_turbo()
        .validate()
        .expect("turbo valid");
}

#[test]
fn test_preset_base_dimensions() {
    let c = WhisperConfig::whisper_base();
    assert_eq!(c.d_model, 512);
    assert_eq!(c.encoder_attention_heads, 8);
    assert_eq!(c.encoder_layers, 6);
    assert_eq!(c.encoder_ffn_dim, 2048);
}

#[test]
fn test_preset_small_dimensions() {
    let c = WhisperConfig::whisper_small();
    assert_eq!(c.d_model, 768);
    assert_eq!(c.encoder_attention_heads, 12);
    assert_eq!(c.encoder_layers, 12);
}

#[test]
fn test_preset_medium_dimensions() {
    let c = WhisperConfig::whisper_medium();
    assert_eq!(c.d_model, 1024);
    assert_eq!(c.encoder_attention_heads, 16);
    assert_eq!(c.encoder_layers, 24);
}

#[test]
fn test_preset_large_v2_dimensions() {
    let c = WhisperConfig::whisper_large_v2();
    assert_eq!(c.d_model, 1280);
    assert_eq!(c.encoder_attention_heads, 20);
    assert_eq!(c.encoder_layers, 32);
    assert_eq!(c.decoder_layers, 32); // Not distilled.
    assert_eq!(c.num_mel_bins, 128);
}

/// forward_no_cache matches cached forward for full-sequence decode.
#[test]
fn test_decoder_forward_no_cache_matches_cached() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    let encoder_output = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).unwrap();
    let tokens = DynTensor::new(&[0.0, 1.0, 2.0], &[1, 3], &cpu()).unwrap();

    // Cached forward: full sequence, flush cache, offset=0.
    let cached_logits = model.decode(&tokens, &encoder_output, true, 0).unwrap();
    let cached_data = cached_logits.to_flat_vec::<f32>().unwrap();

    // No-cache forward.
    let no_cache_logits = model
        .decoder()
        .forward_no_cache(&tokens, &encoder_output)
        .unwrap();
    let no_cache_data = no_cache_logits.to_flat_vec::<f32>().unwrap();

    assert_eq!(cached_logits.dims(), no_cache_logits.dims());

    let max_error = cached_data
        .iter()
        .zip(no_cache_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_error < 1e-5,
        "no_cache vs cached max error should be < 1e-5, got {max_error}"
    );
}

#[path = "whisper_tests_safetensors.rs"]
mod safetensors;

// -- Extended tests for config, decode, tokenizer, mel (#3820) --
#[path = "whisper_tests_extended.rs"]
mod extended;

// -- Architecture validation tests (#3942) --
#[path = "whisper_tests_arch_validation.rs"]
mod arch_validation;

// -- Expanded coverage tests (#4292) --
#[path = "whisper_tests_coverage.rs"]
mod coverage;
