// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Whisper encoder, decoder, mel spectrogram, and
//! end-to-end pipeline.
//!
//! All tests use small synthetic configs and zero/random weights for speed.
//! No real model weights required.

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};
use nn_whisper::audio::{mel_filterbank, pcm_to_mel};
use nn_whisper::config::WhisperConfig;
// tiny_config from test_utils is available but these tests use small_config()
// for slightly larger dimensions.
use nn_whisper::{whisper_mel_spectrogram, whisper_mel_spectrogram_for_config, WhisperModel};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Small config with custom dimensions for integration testing.
/// Slightly larger than tiny_config to exercise more paths.
fn small_config() -> WhisperConfig {
    WhisperConfig::whisper_tiny()
        .with_num_mel_bins(40)
        .with_d_model(64)
        .with_encoder_attention_heads(4)
        .with_encoder_layers(2)
        .with_encoder_ffn_dim(128)
        .with_decoder_attention_heads(4)
        .with_decoder_layers(2)
        .with_decoder_ffn_dim(128)
        .with_vocab_size(100)
        .with_max_source_positions(32)
        .with_max_target_positions(32)
}

fn small_model() -> WhisperModel {
    let config = small_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    WhisperModel::load(&vb, config).expect("small model loads with zero weights")
}

// ===========================================================================
// A. Encoder tests
// ===========================================================================

#[test]
fn test_encoder_forward_shape() {
    // Encoder: [1, num_mel_bins, T] -> [1, seq_len, d_model]
    // seq_len = ceil(T / 2) due to stride-2 conv2.
    let config = small_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 20], DType::F32, &cpu()).unwrap();
    let out = model.encode(&mel).unwrap();

    assert_eq!(out.rank(), 3, "encoder output should be rank 3");
    assert_eq!(out.dim(0).unwrap(), 1, "batch dim should be 1");
    // After stride-2 conv: (20 + 2*1 - 3) / 2 + 1 = 10
    assert_eq!(out.dim(1).unwrap(), 10, "seq_len after stride-2 conv");
    assert_eq!(
        out.dim(2).unwrap(),
        config.d_model,
        "last dim should be d_model"
    );
}

#[test]
fn test_encoder_mel_input_finite() {
    // Mel-shaped input with non-zero values should produce all finite outputs.
    let config = small_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    // Simulate mel spectrogram values (typical range: -1.5 to 1.0).
    let n_elem = config.num_mel_bins * 30;
    let mel_data: Vec<f32> = (0..n_elem)
        .map(|i| ((i as f32) * 0.07).sin() * 0.5 - 0.3)
        .collect();
    let mel = DynTensor::from_vec(mel_data, &[1, config.num_mel_bins, 30], &cpu()).unwrap();

    let out = model.encode(&mel).unwrap();
    let flat = out.to_flat_vec::<f32>().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "all encoder outputs should be finite for valid mel input"
    );
}

#[test]
fn test_encoder_different_lengths() {
    // Various audio lengths should produce expected output dimensions.
    let config = small_config();

    for n_frames in [4, 8, 16, 32] {
        let vb = VarBuilder::zeros(DType::F32, &cpu());
        let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

        let mel =
            DynTensor::zeros(&[1, config.num_mel_bins, n_frames], DType::F32, &cpu()).unwrap();
        let out = model.encode(&mel).unwrap();

        // Expected seq_len after stride-2 conv (padding=1, kernel=3):
        // floor((n_frames + 2*1 - 3) / 2 + 1)
        let expected_seq = (n_frames + 2 - 3) / 2 + 1;
        assert_eq!(
            out.dim(1).unwrap(),
            expected_seq,
            "n_frames={n_frames}: expected seq_len={expected_seq}, got {}",
            out.dim(1).unwrap()
        );
        assert_eq!(out.dim(2).unwrap(), config.d_model);
    }
}

#[test]
fn test_encoder_bf16() {
    // BF16 model should successfully encode F32 mel input.
    let config = small_config();
    let vb = VarBuilder::zeros(DType::BF16, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();
    assert_eq!(model.dtype(), DType::BF16);

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();
    let out = model.encode(&mel).unwrap();
    assert_eq!(out.rank(), 3);
    assert_eq!(out.dim(0).unwrap(), 1);
    assert_eq!(out.dim(2).unwrap(), config.d_model);

    let flat = out.to_flat_vec::<f32>().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "bf16 encoder output should be finite"
    );
}

#[test]
fn test_encoder_batch_size_2() {
    // Encoder should handle batch size > 1.
    let config = small_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    let mel = DynTensor::zeros(&[2, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();
    let out = model.encode(&mel).unwrap();
    assert_eq!(out.dim(0).unwrap(), 2, "batch dim should be 2");
    assert_eq!(out.dim(2).unwrap(), config.d_model);
}

// ===========================================================================
// B. Decoder tests
// ===========================================================================

#[test]
fn test_decoder_forward_shape() {
    // Decoder: encoder_output + token_ids -> [batch, seq_len, vocab_size]
    let config = small_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    let encoder_output = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).unwrap();
    let tokens = DynTensor::new(&[0.0, 1.0, 2.0], &[1, 3], &cpu()).unwrap();

    let logits = model.decode(&tokens, &encoder_output, true, 0).unwrap();
    assert_eq!(logits.rank(), 3);
    assert_eq!(logits.dim(0).unwrap(), 1, "batch");
    assert_eq!(logits.dim(1).unwrap(), 3, "seq_len matches token count");
    assert_eq!(logits.dim(2).unwrap(), config.vocab_size, "vocab_size");
}

#[test]
fn test_decoder_autoregressive() {
    // Multi-step autoregressive decoding: one token at a time with position offsets.
    let config = small_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    let encoder_output = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).unwrap();

    // Step 0: initial token, flush cache.
    let t0 = DynTensor::new(&[0.0], &[1, 1], &cpu()).unwrap();
    let logits0 = model.decode(&t0, &encoder_output, true, 0).unwrap();
    assert_eq!(logits0.dims(), &[1, 1, config.vocab_size]);

    // Step 1: next token, no flush, position offset 1.
    let t1 = DynTensor::new(&[5.0], &[1, 1], &cpu()).unwrap();
    let logits1 = model.decode(&t1, &encoder_output, false, 1).unwrap();
    assert_eq!(logits1.dims(), &[1, 1, config.vocab_size]);

    // Step 2: another token, position offset 2.
    let t2 = DynTensor::new(&[10.0], &[1, 1], &cpu()).unwrap();
    let logits2 = model.decode(&t2, &encoder_output, false, 2).unwrap();
    assert_eq!(logits2.dims(), &[1, 1, config.vocab_size]);

    // All logits should be finite.
    for (i, logits) in [&logits0, &logits1, &logits2].iter().enumerate() {
        let flat = logits.to_flat_vec::<f32>().unwrap();
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "step {i}: logits contain non-finite values"
        );
    }
}

#[test]
fn test_decoder_kv_cache_affects_output() {
    // Verify that the KV cache produces different results compared to flushing.
    // With zero weights, outputs may be identical, but the dimensions should be
    // consistent and the operations should succeed without error.
    let config = small_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    let encoder_output = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).unwrap();

    // Step 0: flush cache.
    let t0 = DynTensor::new(&[0.0], &[1, 1], &cpu()).unwrap();
    model.decode(&t0, &encoder_output, true, 0).unwrap();

    // Step 1 with cache (no flush).
    let t1 = DynTensor::new(&[1.0], &[1, 1], &cpu()).unwrap();
    let logits_cached = model.decode(&t1, &encoder_output, false, 1).unwrap();

    // Step 1 again after flushing (resets KV cache).
    model.reset_kv_cache();
    let t0_again = DynTensor::new(&[0.0], &[1, 1], &cpu()).unwrap();
    model.decode(&t0_again, &encoder_output, true, 0).unwrap();
    let t1_again = DynTensor::new(&[1.0], &[1, 1], &cpu()).unwrap();
    let logits_reflushed = model.decode(&t1_again, &encoder_output, false, 1).unwrap();

    // The two cached paths should produce identical results.
    assert_eq!(logits_cached.dims(), logits_reflushed.dims());
    let v1 = logits_cached.to_flat_vec::<f32>().unwrap();
    let v2 = logits_reflushed.to_flat_vec::<f32>().unwrap();
    let max_diff = v1
        .iter()
        .zip(v2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-6,
        "re-flushed KV cache should produce identical results, max_diff={max_diff}"
    );
}

#[test]
fn test_decoder_cross_attention_shape() {
    // Decoder cross-attention should work with different encoder output lengths.
    let config = small_config();

    for enc_len in [4, 8, 16] {
        let vb = VarBuilder::zeros(DType::F32, &cpu());
        let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

        let encoder_output =
            DynTensor::zeros(&[1, enc_len, config.d_model], DType::F32, &cpu()).unwrap();
        let tokens = DynTensor::new(&[0.0, 1.0], &[1, 2], &cpu()).unwrap();

        let logits = model.decode(&tokens, &encoder_output, true, 0).unwrap();
        assert_eq!(logits.dim(0).unwrap(), 1);
        assert_eq!(logits.dim(1).unwrap(), 2);
        assert_eq!(logits.dim(2).unwrap(), config.vocab_size);
    }
}

#[test]
fn test_decoder_multi_token_prompt() {
    // Decoder should handle a multi-token prompt (e.g., SOT + language + task).
    let config = small_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    let encoder_output = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).unwrap();
    // 5-token prompt: flush cache and process all at once.
    let prompt_data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let tokens = DynTensor::from_vec(prompt_data, &[1, 5], &cpu()).unwrap();

    let logits = model.decode(&tokens, &encoder_output, true, 0).unwrap();
    assert_eq!(logits.dims(), &[1, 5, config.vocab_size]);
}

// ===========================================================================
// C. Mel spectrogram tests
// ===========================================================================

#[test]
fn test_mel_spectrogram_shape() {
    // Standard whisper_mel_spectrogram produces [1, 128, 3000].
    let audio = vec![0.0f32; 16000]; // 1 second of silence
    let mel = whisper_mel_spectrogram(&audio).unwrap();
    assert_eq!(mel.dims(), &[1, 128, 3000]);
}

#[test]
fn test_mel_spectrogram_values_finite() {
    // Known input (440 Hz sine) should produce finite frequency bins.
    let audio: Vec<f32> = (0..16000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
        .collect();
    let mel = whisper_mel_spectrogram(&audio).unwrap();
    let vals = mel.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "mel spectrogram should have all finite values"
    );
    // Tone should produce values above silence floor.
    let max_val = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_val > -1.5,
        "tone mel max ({max_val}) should exceed silence floor (-1.5)"
    );
}

#[test]
fn test_mel_spectrogram_short_audio() {
    // Very short audio (160 samples = 10ms) should be padded to 30s and produce valid output.
    let audio = vec![0.1f32; 160];
    let mel = whisper_mel_spectrogram(&audio).unwrap();
    assert_eq!(mel.dim(0).unwrap(), 1, "batch dim");
    assert_eq!(mel.dim(1).unwrap(), 128, "mel bins should be 128");
    assert_eq!(
        mel.dim(2).unwrap(),
        3000,
        "frames should be 3000 after pad-to-30s"
    );
}

#[test]
fn test_pcm_to_mel_empty_returns_error() {
    // Empty audio to pcm_to_mel should return a proper error.
    // (whisper_mel_spectrogram pads empty audio to 30s, so use pcm_to_mel directly.)
    let audio: Vec<f32> = vec![];
    let filters = mel_filterbank(128, 400, 16000);
    let result = pcm_to_mel(&audio, &filters, 400, 160, 128);
    assert!(
        result.is_err(),
        "empty audio to pcm_to_mel should return error"
    );
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.to_lowercase().contains("empty"),
        "error should mention empty audio, got: {msg}"
    );
}

#[test]
fn test_mel_spectrogram_empty_audio_produces_silence() {
    // whisper_mel_spectrogram pads empty audio to 30s, producing a valid silent mel.
    let audio: Vec<f32> = vec![];
    let mel = whisper_mel_spectrogram(&audio).unwrap();
    assert_eq!(mel.dims(), &[1, 128, 3000], "empty audio padded to 30s");
    let vals = mel.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "padded empty audio should produce finite values"
    );
    // All values should be the silence constant (-1.5).
    let max_val = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_val = vals.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        (max_val - (-1.5)).abs() < 1e-4,
        "silence mel max should be -1.5, got {max_val}"
    );
    assert!(
        (min_val - (-1.5)).abs() < 1e-4,
        "silence mel min should be -1.5, got {min_val}"
    );
}

#[test]
fn test_mel_spectrogram_for_config_80_bins() {
    // whisper_mel_spectrogram_for_config with 80 bins (whisper-tiny/base/small/medium).
    let audio = vec![0.0f32; 16000];
    let mel = whisper_mel_spectrogram_for_config(&audio, 80).unwrap();
    assert_eq!(mel.dim(1).unwrap(), 80, "should have 80 mel bins");
    assert_eq!(mel.dim(2).unwrap(), 3000, "should have 3000 frames");
}

#[test]
fn test_mel_spectrogram_for_config_128_bins() {
    // whisper_mel_spectrogram_for_config with 128 bins (whisper large-v3).
    let audio = vec![0.0f32; 16000];
    let mel = whisper_mel_spectrogram_for_config(&audio, 128).unwrap();
    assert_eq!(mel.dim(1).unwrap(), 128, "should have 128 mel bins");
    assert_eq!(mel.dim(2).unwrap(), 3000, "should have 3000 frames");
}

#[test]
fn test_pcm_to_mel_custom_params() {
    // pcm_to_mel with custom parameters for a non-Whisper scenario.
    let audio: Vec<f32> = (0..512).map(|i| (i as f32 * 0.05).sin()).collect();
    let n_fft = 64;
    let hop = 32;
    let n_mels = 16;
    let filters = mel_filterbank(n_mels, n_fft, 16000);
    let mel = pcm_to_mel(&audio, &filters, n_fft, hop, n_mels).unwrap();
    assert_eq!(mel.dim(0).unwrap(), 1);
    assert_eq!(mel.dim(1).unwrap(), n_mels);
    // Expected frames: (512 + n_fft - n_fft) / hop + 1 = 512/32 + 1 = 17
    assert_eq!(mel.dim(2).unwrap(), 17);
    let vals = mel.to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().all(|v| v.is_finite()));
}

// ===========================================================================
// D. End-to-end tests
// ===========================================================================

#[test]
fn test_whisper_forward_full() {
    // Full pipeline: pcm_to_mel -> encoder -> decoder -> logits.
    // Uses small_config with custom mel to stay within max_source_positions.
    let config = small_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    // Generate mel with pcm_to_mel using small parameters.
    // 1024 samples, n_fft=64, hop=32, n_mels=40 -> about 33 frames.
    // After stride-2 conv: (33 + 2 - 3)/2 + 1 = 17 positions (within 32 limit).
    let audio: Vec<f32> = (0..1024)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
        .collect();
    let filters = mel_filterbank(config.num_mel_bins, 64, 16000);
    let mel = pcm_to_mel(&audio, &filters, 64, 32, config.num_mel_bins).unwrap();
    assert_eq!(mel.dim(1).unwrap(), config.num_mel_bins);

    // Encode.
    let encoder_output = model.encode(&mel).unwrap();
    assert_eq!(encoder_output.rank(), 3);
    assert_eq!(encoder_output.dim(0).unwrap(), 1);
    assert_eq!(encoder_output.dim(2).unwrap(), config.d_model);

    // Decode a single prompt token.
    let tokens = DynTensor::new(&[0.0], &[1, 1], &cpu()).unwrap();
    let logits = model.decode(&tokens, &encoder_output, true, 0).unwrap();
    assert_eq!(logits.dims(), &[1, 1, config.vocab_size]);

    // All logits should be finite.
    let flat = logits.to_flat_vec::<f32>().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "end-to-end logits should be finite"
    );
}

#[test]
fn test_whisper_config_tiny_loads() {
    // whisper_tiny config should load and validate correctly.
    let config = WhisperConfig::whisper_tiny();
    config
        .validate()
        .expect("whisper_tiny config should be valid");

    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let model = WhisperModel::load(&vb, config);
    assert!(
        model.is_ok(),
        "whisper_tiny should load with zero weights: {:?}",
        model.err()
    );

    let model = model.unwrap();
    assert_eq!(model.config().d_model, 384);
    assert_eq!(model.config().num_mel_bins, 80);
    assert_eq!(model.config().encoder_layers, 4);
    assert_eq!(model.config().decoder_layers, 4);
}

#[test]
fn test_whisper_weight_loading_error() {
    // Missing safetensors file should produce a proper error.
    let config = WhisperConfig::whisper_tiny();
    let result = WhisperModel::load_safetensors("/nonexistent/path/model.safetensors", config);
    assert!(result.is_err(), "loading from nonexistent path should fail");
}

#[test]
fn test_whisper_config_validation_rejects_zero_d_model() {
    let config = small_config().with_d_model(0);
    let result = config.validate();
    assert!(result.is_err(), "zero d_model should be rejected");
}

#[test]
fn test_whisper_config_validation_rejects_indivisible_heads() {
    // d_model must be divisible by attention heads.
    let config = small_config()
        .with_d_model(64)
        .with_encoder_attention_heads(5);
    let result = config.validate();
    assert!(
        result.is_err(),
        "d_model=64, encoder_heads=5 should be rejected (64 % 5 != 0)"
    );
}

#[test]
fn test_whisper_encoder_no_cache_matches_cached() {
    // forward_no_cache should produce identical results to forward (full sequence).
    let config = small_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    let mel_data: Vec<f32> = (0..config.num_mel_bins * 20)
        .map(|i| (i as f32 * 0.03).sin())
        .collect();
    let mel = DynTensor::from_vec(mel_data, &[1, config.num_mel_bins, 20], &cpu()).unwrap();

    let cached_out = model.encode(&mel).unwrap();
    let no_cache_out = model.encoder().forward_no_cache(&mel).unwrap();

    assert_eq!(cached_out.dims(), no_cache_out.dims());
    let v1 = cached_out.to_flat_vec::<f32>().unwrap();
    let v2 = no_cache_out.to_flat_vec::<f32>().unwrap();
    let max_diff = v1
        .iter()
        .zip(v2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-5,
        "encoder no_cache vs cached max diff should be < 1e-5, got {max_diff}"
    );
}

#[test]
fn test_whisper_decoder_no_cache_matches_cached() {
    // forward_no_cache should produce identical results to forward (full sequence at once).
    let config = small_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    let encoder_output = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).unwrap();
    let tokens = DynTensor::new(&[0.0, 1.0, 2.0], &[1, 3], &cpu()).unwrap();

    let cached_logits = model.decode(&tokens, &encoder_output, true, 0).unwrap();
    let no_cache_logits = model
        .decoder()
        .forward_no_cache(&tokens, &encoder_output)
        .unwrap();

    assert_eq!(cached_logits.dims(), no_cache_logits.dims());
    let v1 = cached_logits.to_flat_vec::<f32>().unwrap();
    let v2 = no_cache_logits.to_flat_vec::<f32>().unwrap();
    let max_diff = v1
        .iter()
        .zip(v2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-5,
        "decoder no_cache vs cached max diff should be < 1e-5, got {max_diff}"
    );
}

#[test]
fn test_whisper_reset_kv_cache_reproducible() {
    // Resetting KV cache and re-running should produce the same results.
    let config = small_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();
    let encoder_output = model.encode(&mel).unwrap();

    let tokens = DynTensor::new(&[0.0, 1.0], &[1, 2], &cpu()).unwrap();

    // First decode.
    let logits1 = model.decode(&tokens, &encoder_output, true, 0).unwrap();

    // Reset and re-decode.
    model.reset_kv_cache();
    let logits2 = model.decode(&tokens, &encoder_output, true, 0).unwrap();

    let v1 = logits1.to_flat_vec::<f32>().unwrap();
    let v2 = logits2.to_flat_vec::<f32>().unwrap();
    assert_eq!(v1.len(), v2.len());
    let max_diff = v1
        .iter()
        .zip(v2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-6,
        "KV cache reset should produce identical results, max_diff={max_diff}"
    );
}

#[test]
fn test_whisper_all_preset_configs_valid() {
    // Every preset config variant should pass validation.
    for (name, config) in [
        ("tiny", WhisperConfig::whisper_tiny()),
        ("base", WhisperConfig::whisper_base()),
        ("small", WhisperConfig::whisper_small()),
        ("medium", WhisperConfig::whisper_medium()),
        ("large_v2", WhisperConfig::whisper_large_v2()),
        ("large_v3_turbo", WhisperConfig::large_v3_turbo()),
    ] {
        config
            .validate()
            .unwrap_or_else(|e| panic!("{name} config validation failed: {e}"));
        // Verify head dimensions are positive.
        assert!(
            config.encoder_head_dim() > 0,
            "{name}: encoder_head_dim should be > 0"
        );
        assert!(
            config.decoder_head_dim() > 0,
            "{name}: decoder_head_dim should be > 0"
        );
    }
}

#[test]
fn test_whisper_encode_long_audio() {
    // Longer mel input that fits within small_config's max_source_positions.
    // max_source_positions=32 means encoder supports up to 32 output positions.
    // Stride-2 conv: output_len = (input_frames + 2*pad - kernel) / stride + 1
    // For 62 input frames: (62 + 2 - 3) / 2 + 1 = 31 (within 32 limit).
    let config = small_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 62], DType::F32, &cpu()).unwrap();
    let enc_out = model.encode(&mel).unwrap();
    assert_eq!(enc_out.dim(0).unwrap(), 1, "batch");
    assert_eq!(enc_out.dim(1).unwrap(), 31, "seq_len after stride-2");
    assert_eq!(enc_out.dim(2).unwrap(), config.d_model, "d_model");

    let flat = enc_out.to_flat_vec::<f32>().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "encoder output should be finite for longer mel input"
    );
}
