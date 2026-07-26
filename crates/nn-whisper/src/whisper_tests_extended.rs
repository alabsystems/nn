// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for nn-whisper: config, decode helpers, beam search,
//! mel config, tokenizer edge cases. Issue #3820.

use crate::config::WhisperConfig;
use crate::decode::{
    compression_ratio, passes_quality_check, DecodeConfig, DecodingResult, WhisperBeamConfig,
    DEFAULT_AVG_LOGPROB_THRESHOLD, DEFAULT_COMPRESSION_RATIO_THRESHOLD, DEFAULT_TEMPERATURES,
    MAX_DECODE_LENGTH,
};
use crate::tokenizer::{
    WhisperTokenizer, EOT_TOKEN, LANGUAGE_TOKEN_END, LANGUAGE_TOKEN_START,
    NO_SPEECH_TOKEN, SOT_TOKEN,
};

// ==========================================================================
// WhisperConfig tests
// ==========================================================================

#[test]
fn test_config_clone_eq() {
    let c1 = WhisperConfig::large_v3_turbo();
    let c2 = c1.clone();
    assert_eq!(c1, c2, "cloned config should be equal");
}

#[test]
fn test_config_ne_different_fields() {
    let c1 = WhisperConfig::whisper_tiny();
    let c2 = WhisperConfig::whisper_base();
    assert_ne!(c1, c2, "different presets should not be equal");
}

#[test]
fn test_config_with_num_mel_bins() {
    let c = WhisperConfig::whisper_tiny().with_num_mel_bins(64);
    assert_eq!(c.num_mel_bins, 64);
    // Other fields unchanged.
    assert_eq!(c.d_model, 384);
}

#[test]
fn test_config_with_max_source_positions() {
    let c = WhisperConfig::whisper_tiny().with_max_source_positions(750);
    assert_eq!(c.max_source_positions, 750);
}

#[test]
fn test_config_with_encoder_ffn_dim() {
    let c = WhisperConfig::whisper_tiny().with_encoder_ffn_dim(2048);
    assert_eq!(c.encoder_ffn_dim, 2048);
}

#[test]
fn test_config_with_decoder_ffn_dim() {
    let c = WhisperConfig::whisper_tiny().with_decoder_ffn_dim(2048);
    assert_eq!(c.decoder_ffn_dim, 2048);
}

#[test]
fn test_config_with_vocab_size() {
    let c = WhisperConfig::whisper_tiny().with_vocab_size(10000);
    assert_eq!(c.vocab_size, 10000);
}

#[test]
fn test_config_with_max_target_positions() {
    let c = WhisperConfig::whisper_tiny().with_max_target_positions(256);
    assert_eq!(c.max_target_positions, 256);
}

#[test]
fn test_config_encoder_head_dim_tiny() {
    let c = WhisperConfig::whisper_tiny();
    // d_model=384, encoder_attention_heads=6 → 64.
    assert_eq!(c.encoder_head_dim(), 64);
}

#[test]
fn test_config_decoder_head_dim_medium() {
    let c = WhisperConfig::whisper_medium();
    // d_model=1024, decoder_attention_heads=16 → 64.
    assert_eq!(c.decoder_head_dim(), 64);
}

#[test]
fn test_config_encoder_head_dim_base() {
    let c = WhisperConfig::whisper_base();
    // d_model=512, encoder_attention_heads=8 → 64.
    assert_eq!(c.encoder_head_dim(), 64);
}

#[test]
fn test_config_large_v2_not_distilled() {
    let c = WhisperConfig::whisper_large_v2();
    // large-v2 has 32 decoder layers (not distilled).
    assert_eq!(c.decoder_layers, 32);
    assert_eq!(c.encoder_layers, 32);
}

#[test]
fn test_config_turbo_is_distilled() {
    let c = WhisperConfig::large_v3_turbo();
    // turbo has 4 decoder layers (distilled), 32 encoder layers.
    assert_eq!(c.decoder_layers, 4);
    assert_eq!(c.encoder_layers, 32);
}

// ==========================================================================
// DecodeConfig builder & validation tests
// ==========================================================================

#[test]
fn test_decode_config_builder_chain() {
    let config = DecodeConfig::default()
        .with_max_length(100)
        .with_initial_tokens(vec![1, 2, 3])
        .with_suppress_tokens(vec![0, 5])
        .with_seed(Some(42))
        .with_compression_ratio_threshold(3.0)
        .with_avg_logprob_threshold(-0.5);

    assert_eq!(config.max_length, 100);
    assert_eq!(config.initial_tokens, vec![1, 2, 3]);
    assert_eq!(config.suppress_tokens, vec![0, 5]);
    assert_eq!(config.seed, Some(42));
    assert!((config.compression_ratio_threshold - 3.0).abs() < f64::EPSILON);
    assert!((config.avg_logprob_threshold - (-0.5)).abs() < f64::EPSILON);
    assert!(config.validate().is_ok());
}

#[test]
fn test_decode_config_validate_nan_avg_logprob() {
    let config = DecodeConfig::default()
        .with_avg_logprob_threshold(f64::NAN);
    let err = config.validate().unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("avg_logprob_threshold"),
        "should reject NaN avg_logprob_threshold: {msg}"
    );
}

#[test]
fn test_decode_config_validate_inf_compression_ratio() {
    let config = DecodeConfig::default()
        .with_compression_ratio_threshold(f64::INFINITY);
    let err = config.validate().unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("compression_ratio_threshold"),
        "should reject Inf compression_ratio_threshold: {msg}"
    );
}

#[test]
fn test_decode_config_seed_none() {
    let config = DecodeConfig::default().with_seed(None);
    assert!(config.seed.is_none());
    assert!(config.validate().is_ok());
}

#[test]
fn test_decode_config_constants() {
    assert_eq!(MAX_DECODE_LENGTH, 224);
    assert!((DEFAULT_COMPRESSION_RATIO_THRESHOLD - 2.4).abs() < f64::EPSILON);
    assert!((DEFAULT_AVG_LOGPROB_THRESHOLD - (-1.0)).abs() < f64::EPSILON);
    assert_eq!(DEFAULT_TEMPERATURES.len(), 6);
    assert!((DEFAULT_TEMPERATURES[0] - 0.0).abs() < f64::EPSILON);
    assert!((DEFAULT_TEMPERATURES[5] - 1.0).abs() < f64::EPSILON);
}

// ==========================================================================
// DecodingResult tests
// ==========================================================================

#[test]
fn test_decoding_result_new() {
    let r = DecodingResult::new(vec![1, 2, 3], -0.5, 1.5, true, 0.2, 0.1);
    assert_eq!(r.tokens, vec![1, 2, 3]);
    assert!((r.avg_logprob - (-0.5)).abs() < f64::EPSILON);
    assert!((r.compression_ratio - 1.5).abs() < f64::EPSILON);
    assert!(r.reached_eot);
    assert!((r.temperature - 0.2).abs() < f64::EPSILON);
    assert!((r.no_speech_prob - 0.1).abs() < f64::EPSILON);
}

#[test]
fn test_decoding_result_clone() {
    let r1 = DecodingResult::new(vec![10, 20], -1.0, 2.0, false, 0.0, 0.3);
    let r2 = r1.clone();
    assert_eq!(r1.tokens, r2.tokens);
    assert!((r1.avg_logprob - r2.avg_logprob).abs() < f64::EPSILON);
}

// ==========================================================================
// Quality check tests
// ==========================================================================

#[test]
fn test_quality_check_borderline_compression() {
    // Exactly at threshold should pass.
    let result = DecodingResult::new(vec![1, 2], -0.5, 2.4, true, 0.0, 0.0);
    let config = DecodeConfig::default();
    assert!(passes_quality_check(&result, &config));
}

#[test]
fn test_quality_check_borderline_logprob() {
    // Exactly at threshold should pass.
    let result = DecodingResult::new(vec![1, 2], -1.0, 1.0, true, 0.0, 0.0);
    let config = DecodeConfig::default();
    assert!(passes_quality_check(&result, &config));
}

#[test]
fn test_quality_check_both_fail() {
    let result = DecodingResult::new(vec![1, 2], -2.0, 3.0, true, 0.0, 0.0);
    let config = DecodeConfig::default();
    assert!(!passes_quality_check(&result, &config));
}

// ==========================================================================
// Compression ratio tests
// ==========================================================================

#[test]
fn test_compression_ratio_two_tokens() {
    let cr = compression_ratio(&[1, 2]);
    // 1 bigram slot / 1 unique bigram = 1.0.
    assert!((cr - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_compression_ratio_all_same() {
    // [5, 5, 5, 5] → 3 bigram slots, 1 unique bigram → 3.0.
    let cr = compression_ratio(&[5, 5, 5, 5]);
    assert!((cr - 3.0).abs() < f64::EPSILON);
}

#[test]
fn test_compression_ratio_alternating() {
    // [1, 2, 1, 2] → 3 bigram slots, 2 unique bigrams → 1.5.
    // Bigrams: (1,2), (2,1), (1,2). Unique: {(1,2), (2,1)} → 2.
    // ratio = 3 / 2 = 1.5.
    let cr = compression_ratio(&[1, 2, 1, 2]);
    assert!((cr - 1.5).abs() < f64::EPSILON);
}

// ==========================================================================
// Beam config tests
// ==========================================================================

#[test]
fn test_beam_config_valid_custom() {
    let config = WhisperBeamConfig {
        beam_width: 10,
        length_penalty: 0.5,
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_beam_config_neg_inf_penalty_rejected() {
    let config = WhisperBeamConfig {
        beam_width: 3,
        length_penalty: f64::NEG_INFINITY,
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_beam_config_zero_penalty_accepted() {
    // Zero penalty is valid (disables length normalization).
    let config = WhisperBeamConfig {
        beam_width: 5,
        length_penalty: 0.0,
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_beam_config_negative_penalty_accepted() {
    // Negative penalty is unusual but finite — should be accepted.
    let config = WhisperBeamConfig {
        beam_width: 2,
        length_penalty: -1.0,
    };
    assert!(config.validate().is_ok());
}

// ==========================================================================
// Tokenizer tests
// ==========================================================================

fn test_vocab_json() -> String {
    serde_json::json!({
        "hello": 0,
        "\u{0120}world": 1,
        "\u{0120}": 2,
        "the": 3,
        "\u{0120}quick": 4,
        "\u{0120}brown": 5,
        "\u{0120}fox": 6,
        "<|endoftext|>": 50257,
        "<|startoftranscript|>": 50258,
        "<|en|>": 50259,
        "<|fr|>": 50260,
        "<|transcribe|>": 50360,
        "<|notimestamps|>": 50364,
    })
    .to_string()
}

#[test]
fn test_tokenizer_token_str_out_of_range() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // ID far beyond vocab_size.
    assert_eq!(tok.token_str(999_999), None);
}

#[test]
fn test_tokenizer_token_id_not_found() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    assert_eq!(tok.token_id("nonexistent_token"), None);
}

#[test]
fn test_tokenizer_vocab_size_matches_max_id_plus_one() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // Max ID in the vocab is 50364 (notimestamps), so vocab_size = 50365.
    assert_eq!(tok.vocab_size(), 50365);
}

#[test]
fn test_tokenizer_decode_only_special_tokens() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // All tokens are special — result is empty string.
    let text = tok.decode(&[50257, 50258, 50259, 50360, 50364]).unwrap();
    assert_eq!(text, "");
}

#[test]
fn test_tokenizer_language_token_missing_lang() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // "zh" not in test vocab.
    assert_eq!(tok.language_token("zh"), None);
}

#[test]
fn test_tokenizer_is_special_boundary_below() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // 50256 is the last non-special token.
    assert!(!tok.is_special(50256));
}

#[test]
fn test_tokenizer_timestamp_value_below_begin() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // Token 50364 is below TIMESTAMP_BEGIN (50365) — not a timestamp.
    assert_eq!(tok.timestamp_value(50364), None);
}

#[test]
fn test_tokenizer_decode_with_timestamps_only_eot() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // Only EOT token — no segments.
    let segments = tok.decode_with_timestamps(&[EOT_TOKEN]).unwrap();
    assert!(segments.is_empty());
}

#[test]
fn test_tokenizer_decode_with_timestamps_trailing_text() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // Start timestamp but no end timestamp — text has start but no end.
    // <|0.00|> hello
    let segments = tok.decode_with_timestamps(&[50365, 0]).unwrap();
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].text, "hello");
    assert!((segments[0].start.unwrap() - 0.0).abs() < 1e-10);
    assert_eq!(segments[0].end, None);
}

// ==========================================================================
// Tokenizer special token constants
// ==========================================================================

#[test]
fn test_special_token_constants() {
    assert_eq!(EOT_TOKEN, 50257);
    assert_eq!(SOT_TOKEN, 50258);
    assert_eq!(NO_SPEECH_TOKEN, 50363);
    assert_eq!(LANGUAGE_TOKEN_START, 50259);
    assert_eq!(LANGUAGE_TOKEN_END, 50358);
    // 100 language tokens.
    assert_eq!(LANGUAGE_TOKEN_END - LANGUAGE_TOKEN_START + 1, 100);
}

// ==========================================================================
// Mel filterbank shape tests
// ==========================================================================

#[test]
fn test_mel_filterbank_shape_whisper_default() {
    use crate::audio::mel_filterbank;
    use crate::config::{N_FFT, SAMPLE_RATE};

    let filters = mel_filterbank(128, N_FFT, SAMPLE_RATE);
    let n_freqs = N_FFT / 2 + 1; // 201
    assert_eq!(filters.len(), 128 * n_freqs);
}

#[test]
fn test_mel_filterbank_shape_80_bins() {
    use crate::audio::mel_filterbank;
    use crate::config::{N_FFT, SAMPLE_RATE};

    let filters = mel_filterbank(80, N_FFT, SAMPLE_RATE);
    let n_freqs = N_FFT / 2 + 1;
    assert_eq!(filters.len(), 80 * n_freqs);
}

#[test]
fn test_mel_filterbank_non_negative() {
    use crate::audio::mel_filterbank;
    use crate::config::{N_FFT, SAMPLE_RATE};

    let filters = mel_filterbank(128, N_FFT, SAMPLE_RATE);
    for &v in &filters {
        assert!(v >= 0.0, "mel filter values must be non-negative, got {v}");
        assert!(v.is_finite(), "mel filter values must be finite");
    }
}

#[test]
fn test_mel_filterbank_each_bin_has_nonzero() {
    use crate::audio::mel_filterbank;
    use crate::config::{N_FFT, SAMPLE_RATE};

    let n_mels = 128;
    let n_freqs = N_FFT / 2 + 1;
    let filters = mel_filterbank(n_mels, N_FFT, SAMPLE_RATE);

    // Each mel bin should have at least one non-zero filter value.
    for m in 0..n_mels {
        let row = &filters[m * n_freqs..(m + 1) * n_freqs];
        let sum: f32 = row.iter().sum();
        assert!(
            sum > 0.0,
            "mel bin {m} has zero total filter weight"
        );
    }
}

// ==========================================================================
// pcm_to_mel validation tests
// ==========================================================================

#[test]
fn test_pcm_to_mel_rejects_empty_audio() {
    use crate::audio::{mel_filterbank, pcm_to_mel};
    let filters = mel_filterbank(4, 16, 16000);
    let result = pcm_to_mel(&[], &filters, 16, 4, 4);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("empty"), "expected empty audio error: {msg}");
}

#[test]
fn test_pcm_to_mel_rejects_zero_nfft() {
    use crate::audio::pcm_to_mel;
    let audio = vec![0.0f32; 100];
    let result = pcm_to_mel(&audio, &[], 0, 4, 4);
    assert!(result.is_err());
}

#[test]
fn test_pcm_to_mel_rejects_zero_hop() {
    use crate::audio::{mel_filterbank, pcm_to_mel};
    let filters = mel_filterbank(4, 16, 16000);
    let audio = vec![0.0f32; 100];
    let result = pcm_to_mel(&audio, &filters, 16, 0, 4);
    assert!(result.is_err());
}

#[test]
fn test_pcm_to_mel_rejects_nan_audio() {
    use crate::audio::{mel_filterbank, pcm_to_mel};
    let filters = mel_filterbank(4, 16, 16000);
    let mut audio = vec![0.0f32; 100];
    audio[50] = f32::NAN;
    let result = pcm_to_mel(&audio, &filters, 16, 4, 4);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    let msg_lower = msg.to_lowercase();
    assert!(
        msg_lower.contains("non-finite") || msg_lower.contains("nonfinite"),
        "expected non-finite error: {msg}"
    );
}

#[test]
fn test_pcm_to_mel_output_shape() {
    use crate::audio::{mel_filterbank, pcm_to_mel};
    let n_mels = 4;
    let n_fft = 16;
    let hop = 4;
    let filters = mel_filterbank(n_mels, n_fft, 16000);
    let audio = vec![0.1f32; 200];
    let mel = pcm_to_mel(&audio, &filters, n_fft, hop, n_mels).unwrap();
    assert_eq!(mel.rank(), 3);
    assert_eq!(mel.dim(0).unwrap(), 1);
    assert_eq!(mel.dim(1).unwrap(), n_mels);
    // n_frames depends on padded audio length, just verify it's > 0.
    assert!(mel.dim(2).unwrap() > 0);
}

#[test]
fn test_pcm_to_mel_values_finite() {
    use crate::audio::{mel_filterbank, pcm_to_mel};
    let n_mels = 4;
    let n_fft = 16;
    let hop = 4;
    let filters = mel_filterbank(n_mels, n_fft, 16000);
    let audio = vec![0.5f32; 200];
    let mel = pcm_to_mel(&audio, &filters, n_fft, hop, n_mels).unwrap();
    let flat = mel.to_flat_vec::<f32>().unwrap();
    for &v in &flat {
        assert!(v.is_finite(), "mel values should be finite, got {v}");
    }
}

// ==========================================================================
// Audio constants cross-check
// ==========================================================================

#[test]
fn test_audio_constants_derivation() {
    use crate::config::*;
    // N_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH
    assert_eq!(N_SAMPLES, SAMPLE_RATE * CHUNK_LENGTH);
    // N_FRAMES = N_SAMPLES / HOP_LENGTH
    assert_eq!(N_FRAMES, N_SAMPLES / HOP_LENGTH);
    // N_FFT = 400 (25ms window at 16kHz)
    assert_eq!(N_FFT, 400);
    // HOP_LENGTH = 160 (10ms hop at 16kHz)
    assert_eq!(HOP_LENGTH, 160);
}

// ==========================================================================
// WhisperError Display tests
// ==========================================================================

#[test]
fn test_error_display_zero_config_field() {
    use crate::WhisperError;
    let e = WhisperError::ZeroConfigField { field: "d_model" };
    let msg = e.to_string();
    assert!(msg.contains("d_model"));
    assert!(msg.contains("must be > 0"));
}

#[test]
fn test_error_display_token_out_of_range() {
    use crate::WhisperError;
    let e = WhisperError::TokenOutOfRange {
        id: 99999,
        vocab_size: 50000,
    };
    let msg = e.to_string();
    assert!(msg.contains("99999"));
    assert!(msg.contains("50000"));
}
