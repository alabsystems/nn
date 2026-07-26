// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Expanded test coverage for nn-whisper (#4292).
//!
//! Targets gaps: encoder shape propagation with varying mel lengths,
//! encoder forward_no_cache parity, convert_tensor_bytes paths,
//! WhisperError Display coverage, mel spectrogram edge cases,
//! tokenizer boundary conditions, BF16 positional embeddings,
//! batch > 1 encode, decoder multi-step consistency, and
//! quality metric edge cases.

use crate::test_utils::{tiny_config, tiny_encoder_output, tiny_model};
use crate::WhisperModel;
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};

// ==========================================================================
// Encoder shape propagation with varying mel lengths
// ==========================================================================

#[test]
fn test_encoder_shape_mel_len_12() {
    // mel_len=12: conv2 produces (12+2-3)/2+1=6 positions, within max_source_positions=8.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 12], DType::F32, &cpu()).unwrap();
    let out = model.encode(&mel).unwrap();
    assert_eq!(out.dim(1).unwrap(), 6);
    assert_eq!(out.dim(2).unwrap(), config.d_model);
}

#[test]
fn test_encoder_shape_mel_len_larger_config() {
    // Use a config with larger max_source_positions so longer mel works.
    let config = tiny_config().with_max_source_positions(64);
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 64], DType::F32, &cpu()).unwrap();
    let out = model.encode(&mel).unwrap();
    // Conv2 output: (64+2-3)/2+1 = 32.
    assert_eq!(out.dim(1).unwrap(), 32);
    assert_eq!(out.dim(2).unwrap(), config.d_model);
}

#[test]
fn test_encoder_shape_mel_len_4_minimum() {
    // Minimum viable mel length: conv2 stride-2 needs at least kernel_size=3.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 4], DType::F32, &cpu()).unwrap();
    let out = model.encode(&mel).unwrap();
    // Conv2 output: (4+2-3)/2+1 = 2.
    assert_eq!(out.dim(1).unwrap(), 2);
    assert_eq!(out.dim(2).unwrap(), config.d_model);
}

// ==========================================================================
// Encoder forward_no_cache parity
// ==========================================================================

#[test]
fn test_encoder_forward_no_cache_matches_cached() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();

    // Cached forward.
    let cached_out = model.encode(&mel).unwrap();
    let cached_data = cached_out.to_flat_vec::<f32>().unwrap();

    // No-cache forward.
    let no_cache_out = model.encoder().forward_no_cache(&mel).unwrap();
    let no_cache_data = no_cache_out.to_flat_vec::<f32>().unwrap();

    assert_eq!(cached_out.dims(), no_cache_out.dims());

    let max_error = cached_data
        .iter()
        .zip(no_cache_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_error < 1e-5,
        "encoder no_cache vs cached max error should be < 1e-5, got {max_error}"
    );
}

#[test]
fn test_encoder_forward_no_cache_shape_matches() {
    // Use max_source_positions=32 so mel_len=32 produces 16 positions (within limit).
    let config = tiny_config().with_max_source_positions(32);
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let model = WhisperModel::load(&vb, config.clone()).unwrap();
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 32], DType::F32, &cpu()).unwrap();

    let out = model.encoder().forward_no_cache(&mel).unwrap();
    assert_eq!(out.rank(), 3);
    assert_eq!(out.dim(0).unwrap(), 1);
    assert_eq!(out.dim(2).unwrap(), config.d_model);
}

// ==========================================================================
// convert_tensor_bytes paths (lib.rs)
// ==========================================================================

#[test]
fn test_convert_tensor_bytes_f32() {
    // Test the F32 conversion path in convert_tensor_bytes.
    let float_val: f32 = 1.5;
    let bytes = float_val.to_le_bytes();
    let result = crate::convert_tensor_bytes("test", &bytes, safetensors::Dtype::F32).unwrap();
    let data = result.unwrap();
    assert_eq!(data.len(), 1);
    assert!((data[0] - 1.5).abs() < 1e-6);
}

#[test]
fn test_convert_tensor_bytes_f32_multiple() {
    let values: Vec<f32> = vec![1.0, 2.0, 3.0];
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let result = crate::convert_tensor_bytes("test", &bytes, safetensors::Dtype::F32).unwrap();
    let data = result.unwrap();
    assert_eq!(data.len(), 3);
    assert!((data[0] - 1.0).abs() < 1e-6);
    assert!((data[1] - 2.0).abs() < 1e-6);
    assert!((data[2] - 3.0).abs() < 1e-6);
}

#[test]
fn test_convert_tensor_bytes_f32_bad_alignment() {
    // 3 bytes is not aligned to 4.
    let bytes = vec![0u8; 3];
    let result = crate::convert_tensor_bytes("bad_tensor", &bytes, safetensors::Dtype::F32);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("bad_tensor"), "error should name the tensor: {msg}");
    assert!(msg.contains("not aligned"), "error should mention alignment: {msg}");
}

#[test]
fn test_convert_tensor_bytes_bf16() {
    // BF16 for 1.0: sign=0, exp=01111111, mantissa=0000000 → 0x3F80.
    let bf16_bytes = [0x80u8, 0x3F]; // little-endian 0x3F80
    let result = crate::convert_tensor_bytes("test_bf16", &bf16_bytes, safetensors::Dtype::BF16).unwrap();
    let data = result.unwrap();
    assert_eq!(data.len(), 1);
    assert!((data[0] - 1.0).abs() < 0.01, "bf16(1.0) should convert to ~1.0, got {}", data[0]);
}

#[test]
fn test_convert_tensor_bytes_bf16_bad_alignment() {
    let bytes = vec![0u8; 3]; // Not aligned to 2.
    let result = crate::convert_tensor_bytes("bad_bf16", &bytes, safetensors::Dtype::BF16);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("not aligned"), "error should mention alignment: {msg}");
}

#[test]
fn test_convert_tensor_bytes_f16() {
    // F16 for 1.0: 0x3C00 in half-precision IEEE 754.
    let f16_bytes = [0x00u8, 0x3C]; // little-endian 0x3C00
    let result = crate::convert_tensor_bytes("test_f16", &f16_bytes, safetensors::Dtype::F16).unwrap();
    let data = result.unwrap();
    assert_eq!(data.len(), 1);
    assert!((data[0] - 1.0).abs() < 0.01, "f16(1.0) should convert to ~1.0, got {}", data[0]);
}

#[test]
fn test_convert_tensor_bytes_f16_bad_alignment() {
    let bytes = vec![0u8; 5]; // Not aligned to 2.
    let result = crate::convert_tensor_bytes("bad_f16", &bytes, safetensors::Dtype::F16);
    assert!(result.is_err());
}

#[test]
fn test_convert_tensor_bytes_non_float_returns_none() {
    // Non-float dtype should return Ok(None).
    let bytes = vec![0u8; 4];
    let result = crate::convert_tensor_bytes("test", &bytes, safetensors::Dtype::I32).unwrap();
    assert!(result.is_none(), "non-float dtype should return None");
}

// ==========================================================================
// WhisperError Display coverage
// ==========================================================================

#[test]
fn test_error_display_config_not_divisible() {
    use crate::WhisperError;
    let e = WhisperError::ConfigNotDivisible {
        a_name: "d_model",
        a_val: 100,
        b_name: "encoder_attention_heads",
        b_val: 3,
    };
    let msg = e.to_string();
    assert!(msg.contains("d_model"));
    assert!(msg.contains("100"));
    assert!(msg.contains("encoder_attention_heads"));
    assert!(msg.contains("3"));
    assert!(msg.contains("divisible"));
}

#[test]
fn test_error_display_non_finite_config_field() {
    use crate::WhisperError;
    let e = WhisperError::NonFiniteConfigField {
        field: "length_penalty",
        value: f64::NAN,
    };
    let msg = e.to_string();
    assert!(msg.contains("length_penalty"));
    assert!(msg.contains("finite"));
}

#[test]
fn test_error_display_empty_config_field() {
    use crate::WhisperError;
    let e = WhisperError::EmptyConfigField {
        field: "initial_tokens",
    };
    let msg = e.to_string();
    assert!(msg.contains("initial_tokens"));
    assert!(msg.contains("empty"));
}

#[test]
fn test_error_display_config_exceeds_limit() {
    use crate::WhisperError;
    let e = WhisperError::ConfigExceedsLimit {
        field: "max_length",
        value: 500,
        limit: 224,
    };
    let msg = e.to_string();
    assert!(msg.contains("max_length"));
    assert!(msg.contains("500"));
    assert!(msg.contains("224"));
    assert!(msg.contains("exceeds limit"));
}

#[test]
fn test_error_display_empty_audio() {
    use crate::WhisperError;
    let e = WhisperError::EmptyAudio {
        stage: "pcm_to_mel",
    };
    let msg = e.to_string();
    assert!(msg.contains("pcm_to_mel"));
    assert!(msg.contains("empty"));
}

#[test]
fn test_error_display_token_id_overflow() {
    use crate::WhisperError;
    let e = WhisperError::TokenIdOverflow {
        token_id: usize::MAX,
    };
    let msg = e.to_string();
    assert!(msg.contains("u32::MAX"));
}

#[test]
fn test_error_display_logit_too_small() {
    use crate::WhisperError;
    let e = WhisperError::LogitTooSmall {
        logit_len: 10,
        vocab_size: 50000,
    };
    let msg = e.to_string();
    assert!(msg.contains("10"));
    assert!(msg.contains("50000"));
}

#[test]
fn test_error_display_invalid_temperature() {
    use crate::WhisperError;
    let e = WhisperError::InvalidTemperature { temperature: -0.5 };
    let msg = e.to_string();
    assert!(msg.contains("temperature"));
    assert!(msg.contains("-0.5"));
}

#[test]
fn test_error_display_position_overflow() {
    use crate::WhisperError;
    let e = WhisperError::PositionOverflow {
        offset: usize::MAX,
        seq_len: 1,
    };
    let msg = e.to_string();
    assert!(msg.contains("overflows"));
}

#[test]
fn test_error_display_batch_mismatch() {
    use crate::WhisperError;
    let e = WhisperError::BatchMismatch {
        encoder_batch: 2,
        decoder_batch: 1,
    };
    let msg = e.to_string();
    assert!(msg.contains("encoder batch size (2)"));
    assert!(msg.contains("decoder batch size (1)"));
}

#[test]
fn test_error_display_cache_seq_mismatch() {
    use crate::WhisperError;
    let e = WhisperError::CacheSeqMismatch {
        cached_seq: 100,
        encoder_seq: 200,
    };
    let msg = e.to_string();
    assert!(msg.contains("100"));
    assert!(msg.contains("200"));
}

#[test]
fn test_error_display_byte_alignment() {
    use crate::WhisperError;
    let e = WhisperError::ByteAlignment {
        tensor_name: "conv1.weight".into(),
        byte_len: 7,
        alignment: 4,
    };
    let msg = e.to_string();
    assert!(msg.contains("conv1.weight"));
    assert!(msg.contains("7"));
    assert!(msg.contains("4"));
}

#[test]
fn test_error_display_safetensors_parse() {
    use crate::WhisperError;
    let e = WhisperError::SafetensorsParseError {
        detail: "invalid header".into(),
    };
    let msg = e.to_string();
    assert!(msg.contains("invalid header"));
}

#[test]
fn test_error_display_non_finite_weight() {
    use crate::WhisperError;
    let e = WhisperError::NonFiniteWeight {
        tensor_name: "encoder.layers.0.fc1.weight".into(),
        count: 5,
    };
    let msg = e.to_string();
    assert!(msg.contains("encoder.layers.0.fc1.weight"));
    assert!(msg.contains("5"));
    assert!(msg.contains("non-finite"));
}

#[test]
fn test_error_display_vocab_parse() {
    use crate::WhisperError;
    let e = WhisperError::VocabParseError {
        detail: "unexpected EOF".into(),
    };
    let msg = e.to_string();
    assert!(msg.contains("unexpected EOF"));
}

#[test]
fn test_error_display_utf8_decode() {
    use crate::WhisperError;
    let e = WhisperError::Utf8DecodeError {
        detail: "invalid sequence".into(),
    };
    let msg = e.to_string();
    assert!(msg.contains("invalid sequence"));
}

#[test]
fn test_error_display_language_token_range() {
    use crate::WhisperError;
    let e = WhisperError::LanguageTokenRange {
        start: 50259,
        end: 50359,
        vocab_size: 100,
    };
    let msg = e.to_string();
    assert!(msg.contains("50259"));
    assert!(msg.contains("50359"));
    assert!(msg.contains("100"));
}

// ==========================================================================
// WhisperError From<WhisperError> for TensorError conversion
// ==========================================================================

#[test]
fn test_whisper_error_to_tensor_error_roundtrip() {
    use crate::WhisperError;
    use nn_core::TensorError;
    let we = WhisperError::ZeroConfigField { field: "d_model" };
    let te: TensorError = we.into();
    let msg = te.to_string();
    assert!(msg.contains("d_model"), "TensorError should preserve message: {msg}");
}

// ==========================================================================
// Mel spectrogram edge cases
// ==========================================================================

#[test]
fn test_whisper_mel_spectrogram_for_config_80_bins() {
    // Test with 80 mel bins (whisper-tiny/base/small/medium).
    let audio = vec![0.1f32; 16000];
    let mel = crate::audio::whisper_mel_spectrogram_for_config(&audio, 80).unwrap();
    assert_eq!(mel.dim(0).unwrap(), 1);
    assert_eq!(mel.dim(1).unwrap(), 80);
    assert_eq!(mel.dim(2).unwrap(), 3000); // Padded to 30s, clipped.
}

#[test]
fn test_whisper_mel_spectrogram_for_config_128_bins() {
    let audio = vec![0.1f32; 8000];
    let mel = crate::audio::whisper_mel_spectrogram_for_config(&audio, 128).unwrap();
    assert_eq!(mel.dim(1).unwrap(), 128);
    assert_eq!(mel.dim(2).unwrap(), 3000);
}

#[test]
fn test_pcm_to_mel_very_short_audio() {
    // Audio shorter than n_fft -- should still work because of reflect-padding.
    use crate::audio::{mel_filterbank, pcm_to_mel};
    let n_fft = 400;
    let hop = 160;
    let n_mels = 128;
    let filters = mel_filterbank(n_mels, n_fft, 16000);
    let audio = vec![0.1f32; 10]; // Only 10 samples.
    let mel = pcm_to_mel(&audio, &filters, n_fft, hop, n_mels).unwrap();
    assert_eq!(mel.rank(), 3);
    assert_eq!(mel.dim(0).unwrap(), 1);
    assert_eq!(mel.dim(1).unwrap(), n_mels);
    assert!(mel.dim(2).unwrap() > 0, "should produce at least one frame");
    // All values should be finite.
    let flat = mel.to_flat_vec::<f32>().unwrap();
    for &v in &flat {
        assert!(v.is_finite(), "mel value must be finite: {v}");
    }
}

#[test]
fn test_pcm_to_mel_single_sample() {
    use crate::audio::{mel_filterbank, pcm_to_mel};
    let filters = mel_filterbank(4, 16, 16000);
    let audio = vec![0.5f32]; // Single sample.
    let mel = pcm_to_mel(&audio, &filters, 16, 4, 4).unwrap();
    assert_eq!(mel.rank(), 3);
    assert_eq!(mel.dim(0).unwrap(), 1);
    assert_eq!(mel.dim(1).unwrap(), 4);
    let flat = mel.to_flat_vec::<f32>().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()));
}

#[test]
fn test_mel_filterbank_single_bin() {
    use crate::audio::mel_filterbank;
    // Single mel bin should produce a valid filterbank.
    let filters = mel_filterbank(1, 400, 16000);
    assert_eq!(filters.len(), 201); // 1 * (400/2 + 1)
    assert!(filters.iter().all(|&v| v >= 0.0 && v.is_finite()));
}

#[test]
fn test_pcm_to_mel_rejects_inf_audio() {
    use crate::audio::{mel_filterbank, pcm_to_mel};
    let filters = mel_filterbank(4, 16, 16000);
    let mut audio = vec![0.0f32; 100];
    audio[0] = f32::INFINITY;
    let result = pcm_to_mel(&audio, &filters, 16, 4, 4);
    assert!(result.is_err());
}

// ==========================================================================
// Positional encoding BF16 dtype tests
// ==========================================================================

#[test]
fn test_sinusoidal_embedding_bf16_shape_and_dtype() {
    use crate::positional::sinusoidal_embedding;
    let emb = sinusoidal_embedding(10, 16, DType::BF16, &cpu()).unwrap();
    assert_eq!(emb.dims(), &[10, 16]);
    assert_eq!(emb.dtype(), DType::BF16);
}

#[test]
fn test_causal_mask_bf16_dtype() {
    use crate::positional::causal_mask;
    let mask = causal_mask(4, DType::BF16, &cpu()).unwrap();
    assert_eq!(mask.dims(), &[4, 4]);
    assert_eq!(mask.dtype(), DType::BF16);
}

// ==========================================================================
// Batch > 1 encoder
// ==========================================================================

#[test]
fn test_encoder_batch_2() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();
    let mel = DynTensor::zeros(&[2, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();
    let out = model.encode(&mel).unwrap();
    assert_eq!(out.dim(0).unwrap(), 2, "batch size should propagate");
    assert_eq!(out.dim(2).unwrap(), config.d_model);
}

// ==========================================================================
// Decoder: multi-step decode produces consistent shapes
// ==========================================================================

#[test]
fn test_decoder_multi_step_shapes() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();
    let enc_out = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).unwrap();

    // Step 0: initial prompt of 4 tokens.
    let t0 = DynTensor::new(&[0.0, 1.0, 2.0, 3.0], &[1, 4], &cpu()).unwrap();
    let logits0 = model.decode(&t0, &enc_out, true, 0).unwrap();
    assert_eq!(logits0.dims(), &[1, 4, config.vocab_size]);

    // Steps 1-5: single token each.
    for step in 0..5 {
        let t = DynTensor::new(&[(4 + step) as f32], &[1, 1], &cpu()).unwrap();
        let logits = model.decode(&t, &enc_out, false, 4 + step).unwrap();
        assert_eq!(logits.dims(), &[1, 1, config.vocab_size], "step {step}");
    }
}

// ==========================================================================
// Decoder forward_no_cache with different token lengths
// ==========================================================================

#[test]
fn test_decoder_forward_no_cache_single_token() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let model = WhisperModel::load(&vb, config.clone()).unwrap();
    let enc_out = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).unwrap();
    let tokens = DynTensor::new(&[0.0], &[1, 1], &cpu()).unwrap();
    let logits = model.decoder().forward_no_cache(&tokens, &enc_out).unwrap();
    assert_eq!(logits.dims(), &[1, 1, config.vocab_size]);
}

#[test]
fn test_decoder_forward_no_cache_max_target_positions() {
    let config = tiny_config(); // max_target_positions = 16
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let model = WhisperModel::load(&vb, config.clone()).unwrap();
    let enc_out = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).unwrap();
    // Use exactly max_target_positions tokens.
    let data: Vec<f32> = (0..config.max_target_positions).map(|i| i as f32).collect();
    let tokens = DynTensor::new(&data, &[1, config.max_target_positions], &cpu()).unwrap();
    let logits = model.decoder().forward_no_cache(&tokens, &enc_out).unwrap();
    assert_eq!(
        logits.dim(1).unwrap(),
        config.max_target_positions,
        "seq_len should match input"
    );
}

// ==========================================================================
// Tokenizer boundary conditions
// ==========================================================================

#[test]
fn test_tokenizer_is_special_boundary() {
    use crate::tokenizer::{WhisperTokenizer, EOT_TOKEN};
    let tok = WhisperTokenizer::from_vocab_str(&serde_json::json!({
        "a": 0, "<|endoftext|>": 50257,
    }).to_string()).unwrap();

    // Token 50256 is the last non-special token.
    assert!(!tok.is_special(50256));
    // Token 50257 (EOT) is special.
    assert!(tok.is_special(EOT_TOKEN));
    // Tokens above 50257 are special.
    assert!(tok.is_special(50258));
    assert!(tok.is_special(100_000));
}

#[test]
fn test_tokenizer_is_timestamp() {
    use crate::tokenizer::WhisperTokenizer;
    let tok = WhisperTokenizer::from_vocab_str(&serde_json::json!({
        "a": 0, "<|endoftext|>": 50257,
    }).to_string()).unwrap();

    assert!(!tok.is_timestamp(50364)); // notimestamps
    assert!(tok.is_timestamp(50365)); // <|0.00|>
    assert!(tok.is_timestamp(50366)); // <|0.02|>
    assert!(!tok.is_timestamp(0)); // regular token
}

#[test]
fn test_tokenizer_timestamp_value_at_begin() {
    use crate::tokenizer::WhisperTokenizer;
    let tok = WhisperTokenizer::from_vocab_str("{}").unwrap();
    let v = tok.timestamp_value(50365);
    assert_eq!(v, Some(0.0));
}

#[test]
fn test_tokenizer_timestamp_value_at_30s() {
    use crate::tokenizer::WhisperTokenizer;
    let tok = WhisperTokenizer::from_vocab_str("{}").unwrap();
    // 50365 + 1500 = 51865 → 1500 * 0.02 = 30.0s
    let v = tok.timestamp_value(51865);
    assert_eq!(v, Some(30.0));
}

#[test]
fn test_tokenizer_can_encode_false_by_default() {
    use crate::tokenizer::WhisperTokenizer;
    let tok = WhisperTokenizer::from_vocab_str(&serde_json::json!({
        "hello": 0, "world": 1,
    }).to_string()).unwrap();
    assert!(!tok.can_encode(), "decode-only tokenizer should not support encode");
}

#[test]
fn test_tokenizer_decode_empty_token_ids() {
    use crate::tokenizer::WhisperTokenizer;
    let tok = WhisperTokenizer::from_vocab_str(&serde_json::json!({
        "hello": 0,
    }).to_string()).unwrap();
    let text = tok.decode(&[]).unwrap();
    assert_eq!(text, "");
}

#[test]
fn test_tokenizer_decode_with_timestamps_paired() {
    use crate::tokenizer::WhisperTokenizer;
    let vocab = serde_json::json!({
        "hello": 0,
        "\u{0120}world": 1,
        "<|endoftext|>": 50257,
        "<|startoftranscript|>": 50258,
    });
    let tok = WhisperTokenizer::from_vocab_str(&vocab.to_string()).unwrap();

    // <|0.00|> hello <|0.20|> <|0.20|> world <|0.40|>
    let tokens = vec![50365, 0, 50375, 50375, 1, 50385];
    let segments = tok.decode_with_timestamps(&tokens).unwrap();

    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].text, "hello");
    assert!((segments[0].start.unwrap() - 0.0).abs() < 1e-10);
    assert!((segments[0].end.unwrap() - 0.2).abs() < 1e-10);
    // Second segment: " world"
    assert!((segments[1].start.unwrap() - 0.2).abs() < 1e-10);
    assert!((segments[1].end.unwrap() - 0.4).abs() < 1e-10);
}

#[test]
fn test_tokenizer_decode_with_timestamps_no_timestamps() {
    use crate::tokenizer::WhisperTokenizer;
    let vocab = serde_json::json!({
        "hello": 0,
    });
    let tok = WhisperTokenizer::from_vocab_str(&vocab.to_string()).unwrap();

    // Regular token, no timestamps.
    let segments = tok.decode_with_timestamps(&[0]).unwrap();
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].text, "hello");
    assert_eq!(segments[0].start, None);
    assert_eq!(segments[0].end, None);
}

// ==========================================================================
// Quality: word_error_rate edge cases
// ==========================================================================

#[test]
fn test_wer_single_word_correct() {
    let wer = crate::quality::word_error_rate("hello", "hello");
    assert!((wer - 0.0).abs() < 1e-6);
}

#[test]
fn test_wer_single_word_wrong() {
    let wer = crate::quality::word_error_rate("goodbye", "hello");
    assert!((wer - 1.0).abs() < 1e-6); // 1 sub / 1 ref = 1.0
}

#[test]
fn test_wer_multiple_insertions() {
    // hyp has 3 extra words vs ref of 2 words.
    let wer = crate::quality::word_error_rate("the big brown lazy fox", "the fox");
    // Optimal: keep "the" and "fox", delete "big brown lazy" → 3 ins / 2 ref = 1.5
    assert!((wer - 1.5).abs() < 1e-6);
}

#[test]
fn test_wer_all_deleted() {
    let wer = crate::quality::word_error_rate("", "one two three four");
    assert!((wer - 1.0).abs() < 1e-6); // 4 dels / 4 ref = 1.0
}

#[test]
fn test_wer_whitespace_normalization() {
    // Multiple spaces should be treated as single separator.
    let wer = crate::quality::word_error_rate("hello   world", "hello world");
    assert!((wer - 0.0).abs() < 1e-6);
}

// ==========================================================================
// DecodeConfig validate: empty initial tokens
// ==========================================================================

#[test]
fn test_decode_config_empty_initial_tokens_rejected() {
    use crate::decode::DecodeConfig;
    let config = DecodeConfig::default().with_initial_tokens(vec![]);
    let err = config.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("initial_tokens"),
        "should reject empty initial_tokens: {msg}"
    );
}

// ==========================================================================
// Beam search config defaults
// ==========================================================================

#[test]
fn test_beam_config_default_values() {
    use crate::decode::WhisperBeamConfig;
    let bc = WhisperBeamConfig::default();
    assert_eq!(bc.beam_width, 5);
    assert!((bc.length_penalty - 1.0).abs() < f64::EPSILON);
    assert!(bc.validate().is_ok());
}

#[test]
fn test_beam_config_zero_width_rejected() {
    use crate::decode::WhisperBeamConfig;
    let config = WhisperBeamConfig {
        beam_width: 0,
        length_penalty: 1.0,
    };
    let err = config.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("beam_width"), "should reject zero beam_width: {msg}");
}

#[test]
fn test_beam_config_nan_penalty_rejected() {
    use crate::decode::WhisperBeamConfig;
    let config = WhisperBeamConfig {
        beam_width: 5,
        length_penalty: f64::NAN,
    };
    assert!(config.validate().is_err());
}

// ==========================================================================
// Temperature fallback with empty temperatures rejected
// ==========================================================================

#[test]
fn test_temperature_fallback_empty_temperatures_rejected() {
    use crate::decode::{temperature_fallback_decode, DecodeConfig};
    let mut model = tiny_model();
    let enc = tiny_encoder_output();
    let config = DecodeConfig {
        max_length: 3,
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        ..Default::default()
    };
    let result = temperature_fallback_decode(&mut model, &enc, &config, &[]);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("empty"), "should reject empty temperatures: {msg}");
}

// ==========================================================================
// DecodingResult no_speech_prob propagation
// ==========================================================================

#[test]
fn test_decoding_result_no_speech_prob_field() {
    use crate::decode::DecodingResult;
    let r = DecodingResult::new(vec![], 0.0, 1.0, false, 0.0, 0.95);
    assert!((r.no_speech_prob - 0.95).abs() < f64::EPSILON);
}

// ==========================================================================
// Encoder cache reset between encodes
// ==========================================================================

#[test]
fn test_encoder_reset_between_encodes() {
    // Verify that consecutive encode() calls produce identical results
    // (since encode() resets caches internally).
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();

    let out1 = model.encode(&mel).unwrap();
    let out2 = model.encode(&mel).unwrap();

    let data1 = out1.to_flat_vec::<f32>().unwrap();
    let data2 = out2.to_flat_vec::<f32>().unwrap();
    assert_eq!(data1, data2, "consecutive encodes should produce identical results");
}

// ==========================================================================
// Model accessors
// ==========================================================================

#[test]
fn test_model_config_accessor() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let model = WhisperModel::load(&vb, config.clone()).unwrap();
    assert_eq!(model.config().d_model, config.d_model);
    assert_eq!(model.config().encoder_layers, config.encoder_layers);
}

#[test]
fn test_model_dtype_accessor() {
    let vb_f32 = VarBuilder::zeros(DType::F32, &cpu());
    let model_f32 = WhisperModel::load(&vb_f32, tiny_config()).unwrap();
    assert_eq!(model_f32.dtype(), DType::F32);

    let vb_bf16 = VarBuilder::zeros(DType::BF16, &cpu());
    let model_bf16 = WhisperModel::load(&vb_bf16, tiny_config()).unwrap();
    assert_eq!(model_bf16.dtype(), DType::BF16);
}

// ==========================================================================
// Compression ratio additional cases
// ==========================================================================

#[test]
fn test_compression_ratio_three_tokens_all_different() {
    use crate::decode::compression_ratio;
    // [1, 2, 3]: 2 bigram slots, 2 unique → 1.0.
    let cr = compression_ratio(&[1, 2, 3]);
    assert!((cr - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_compression_ratio_long_repeat() {
    use crate::decode::compression_ratio;
    // 100 identical tokens: 99 bigram slots, 1 unique → 99.0.
    let tokens: Vec<usize> = vec![42; 100];
    let cr = compression_ratio(&tokens);
    assert!((cr - 99.0).abs() < f64::EPSILON);
}

// ==========================================================================
// Causal mask properties
// ==========================================================================

#[test]
fn test_causal_mask_first_row_all_attend() {
    use crate::positional::causal_mask;
    let mask = causal_mask(5, DType::F32, &cpu()).unwrap();
    let flat = mask.to_flat_vec::<f32>().unwrap();
    // Row 0: only element [0][0] = 0.0, rest should be -inf.
    assert_eq!(flat[0], 0.0);
    for (j, &v) in flat.iter().enumerate().take(5).skip(1) {
        assert_eq!(v, f32::NEG_INFINITY, "mask[0][{j}] should be -inf");
    }
}

#[test]
fn test_causal_mask_last_row_all_attend() {
    use crate::positional::causal_mask;
    let mask = causal_mask(5, DType::F32, &cpu()).unwrap();
    let flat = mask.to_flat_vec::<f32>().unwrap();
    // Row 4: all positions [0..=4] should be 0.0 (can attend to all).
    for j in 0..5 {
        assert_eq!(flat[4 * 5 + j], 0.0, "mask[4][{j}] should be 0.0");
    }
}
