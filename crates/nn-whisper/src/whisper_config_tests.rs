// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive Whisper model configuration and inference tests.
//!
//! Covers config validation edge cases (each zero-field rejection independently),
//! mel spectrogram dimensions, attention head math, positional encoding lengths,
//! tokenizer special tokens, language token ranges, multitask head tokens, and
//! decode config integration. Part of #3525.

use crate::config::{WhisperConfig, CHUNK_LENGTH, HOP_LENGTH, N_FRAMES, N_SAMPLES, NUM_MEL_BINS};
use crate::tokenizer::{
    EOT_TOKEN, LANGUAGE_TOKEN_END, LANGUAGE_TOKEN_START, NO_SPEECH_TOKEN, SOT_TOKEN,
    TIMESTAMP_BEGIN,
};

// ============================================================================
// Config validation: each zero-field rejection independently
// ============================================================================

#[test]
fn test_validate_rejects_zero_d_model() {
    let err = WhisperConfig::whisper_tiny().with_d_model(0).validate().unwrap_err();
    assert!(err.to_string().contains("d_model"));
}

#[test]
fn test_validate_rejects_zero_encoder_attention_heads() {
    let err = WhisperConfig::whisper_tiny()
        .with_encoder_attention_heads(0)
        .validate()
        .unwrap_err();
    assert!(err.to_string().contains("encoder_attention_heads"));
}

#[test]
fn test_validate_rejects_zero_decoder_attention_heads() {
    let err = WhisperConfig::whisper_tiny()
        .with_decoder_attention_heads(0)
        .validate()
        .unwrap_err();
    assert!(err.to_string().contains("decoder_attention_heads"));
}

#[test]
fn test_validate_rejects_zero_vocab_size() {
    let err = WhisperConfig::whisper_tiny().with_vocab_size(0).validate().unwrap_err();
    assert!(err.to_string().contains("vocab_size"));
}

#[test]
fn test_validate_rejects_zero_num_mel_bins() {
    let err = WhisperConfig::whisper_tiny().with_num_mel_bins(0).validate().unwrap_err();
    assert!(err.to_string().contains("num_mel_bins"));
}

#[test]
fn test_validate_rejects_zero_encoder_ffn_dim() {
    let err = WhisperConfig::whisper_tiny().with_encoder_ffn_dim(0).validate().unwrap_err();
    assert!(err.to_string().contains("encoder_ffn_dim"));
}

#[test]
fn test_validate_rejects_zero_decoder_ffn_dim() {
    let err = WhisperConfig::whisper_tiny().with_decoder_ffn_dim(0).validate().unwrap_err();
    assert!(err.to_string().contains("decoder_ffn_dim"));
}

#[test]
fn test_validate_rejects_zero_max_source_positions() {
    let err = WhisperConfig::whisper_tiny()
        .with_max_source_positions(0)
        .validate()
        .unwrap_err();
    assert!(err.to_string().contains("max_source_positions"));
}

#[test]
fn test_validate_rejects_zero_max_target_positions() {
    let err = WhisperConfig::whisper_tiny()
        .with_max_target_positions(0)
        .validate()
        .unwrap_err();
    assert!(err.to_string().contains("max_target_positions"));
}

#[test]
fn test_validate_rejects_non_divisible_encoder_heads() {
    let err = WhisperConfig::whisper_tiny()
        .with_encoder_attention_heads(5) // 384 % 5 != 0
        .validate()
        .unwrap_err();
    assert!(err.to_string().contains("divisible"));
}

#[test]
fn test_validate_rejects_non_divisible_decoder_heads() {
    let err = WhisperConfig::whisper_tiny()
        .with_decoder_attention_heads(5) // 384 % 5 != 0
        .validate()
        .unwrap_err();
    assert!(err.to_string().contains("divisible"));
}

// ============================================================================
// Mel spectrogram dimensions: n_mels * n_audio_ctx
// ============================================================================

#[test]
fn test_mel_spectrogram_total_features_per_preset() {
    let presets: &[(&str, WhisperConfig)] = &[
        ("tiny", WhisperConfig::whisper_tiny()),
        ("base", WhisperConfig::whisper_base()),
        ("small", WhisperConfig::whisper_small()),
        ("medium", WhisperConfig::whisper_medium()),
        ("large-v2", WhisperConfig::whisper_large_v2()),
        ("turbo", WhisperConfig::large_v3_turbo()),
    ];
    for (name, config) in presets {
        let mel_features = config.num_mel_bins * N_FRAMES;
        let expected = if config.num_mel_bins == 80 { 240_000 } else { 384_000 };
        assert_eq!(mel_features, expected, "{name}: mel features mismatch");
    }
}

#[test]
fn test_mel_bins_times_max_source_positions_consistent() {
    for config in &[
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_large_v2(),
    ] {
        let product = config.num_mel_bins * config.max_source_positions;
        assert!(product == 120_000 || product == 192_000, "got {product}");
    }
}

// ============================================================================
// Attention head dimensions: d_model / n_head == head_dim
// ============================================================================

#[test]
fn test_head_dim_is_64_all_presets() {
    let presets = [
        ("tiny", WhisperConfig::whisper_tiny()),
        ("base", WhisperConfig::whisper_base()),
        ("small", WhisperConfig::whisper_small()),
        ("medium", WhisperConfig::whisper_medium()),
        ("large-v2", WhisperConfig::whisper_large_v2()),
        ("turbo", WhisperConfig::large_v3_turbo()),
    ];
    for (name, config) in &presets {
        assert_eq!(config.encoder_head_dim(), 64, "{name}: enc head_dim");
        assert_eq!(config.decoder_head_dim(), 64, "{name}: dec head_dim");
    }
}

#[test]
fn test_head_dim_reconstruction_from_parts() {
    for config in &[WhisperConfig::whisper_tiny(), WhisperConfig::large_v3_turbo()] {
        assert_eq!(config.encoder_head_dim() * config.encoder_attention_heads, config.d_model);
        assert_eq!(config.decoder_head_dim() * config.decoder_attention_heads, config.d_model);
    }
}

// ============================================================================
// Positional encoding length matches context size
// ============================================================================

#[test]
fn test_encoder_positional_length_equals_half_n_frames() {
    // Conv stem halves mel frames: N_FRAMES(3000) / 2 = 1500 = max_source_positions
    for config in &[WhisperConfig::whisper_tiny(), WhisperConfig::large_v3_turbo()] {
        assert_eq!(config.max_source_positions, N_FRAMES / 2);
        assert_eq!(config.max_source_positions, 1500);
    }
}

#[test]
fn test_decoder_positional_length_covers_max_decode_length() {
    let max_decode = crate::decode::MAX_DECODE_LENGTH;
    for config in &[WhisperConfig::whisper_tiny(), WhisperConfig::large_v3_turbo()] {
        assert!(
            config.max_target_positions >= max_decode,
            "max_target_positions ({}) < MAX_DECODE_LENGTH ({max_decode})",
            config.max_target_positions
        );
    }
}

#[test]
fn test_decoder_positional_length_is_448() {
    for config in &[
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_medium(),
        WhisperConfig::large_v3_turbo(),
    ] {
        assert_eq!(config.max_target_positions, 448);
    }
}

// ============================================================================
// Tokenizer special tokens: SOT, EOT, TRANSLATE, TRANSCRIBE, etc.
// ============================================================================

#[test]
fn test_special_token_exact_values() {
    assert_eq!(EOT_TOKEN, 50257);
    assert_eq!(SOT_TOKEN, 50258);
    assert_eq!(LANGUAGE_TOKEN_START, 50259);
    assert_eq!(LANGUAGE_TOKEN_END, 50358);
    assert_eq!(NO_SPEECH_TOKEN, 50363);
    assert_eq!(TIMESTAMP_BEGIN, 50365);
}

#[test]
fn test_special_token_ordering_complete() {
    // Full ordering: EOT < SOT < LANG_START .. LANG_END < translate(50359)
    // < transcribe(50360) < NO_SPEECH(50363) < NO_TIMESTAMPS(50364) < TIMESTAMP_BEGIN
    assert!(EOT_TOKEN < SOT_TOKEN);
    assert!(SOT_TOKEN < LANGUAGE_TOKEN_START);
    assert!(LANGUAGE_TOKEN_START < LANGUAGE_TOKEN_END);
    assert!(LANGUAGE_TOKEN_END < 50359); // translate
    assert!(50359 < 50360); // transcribe
    assert!(50360 < NO_SPEECH_TOKEN);
    assert!(NO_SPEECH_TOKEN < TIMESTAMP_BEGIN);
}

#[test]
fn test_eot_sot_adjacent() {
    assert_eq!(SOT_TOKEN, EOT_TOKEN + 1);
}

// ============================================================================
// Language tokens: valid ISO 639-1 codes
// ============================================================================

#[test]
fn test_language_token_count_is_100() {
    assert_eq!(LANGUAGE_TOKEN_END - LANGUAGE_TOKEN_START + 1, 100);
}

#[test]
fn test_language_token_start_immediately_follows_sot() {
    assert_eq!(LANGUAGE_TOKEN_START, SOT_TOKEN + 1);
}

#[test]
fn test_language_tokens_are_special() {
    assert!(LANGUAGE_TOKEN_START >= EOT_TOKEN);
    assert!(LANGUAGE_TOKEN_END >= EOT_TOKEN);
}

#[test]
fn test_language_tokens_do_not_overlap_task_tokens() {
    assert!(LANGUAGE_TOKEN_END < 50359, "must end before translate token");
}

// ============================================================================
// Multitask head: task-specific output projection
// ============================================================================

#[test]
fn test_task_token_values() {
    let translate: usize = 50359;
    let transcribe: usize = 50360;
    assert_eq!(translate, LANGUAGE_TOKEN_END + 1);
    assert_eq!(transcribe, translate + 1);
}

#[test]
fn test_default_decode_config_uses_transcribe_task() {
    let config = crate::DecodeConfig::default();
    assert_eq!(config.initial_tokens.len(), 4);
    assert_eq!(config.initial_tokens[0], SOT_TOKEN);           // 50258
    assert_eq!(config.initial_tokens[1], LANGUAGE_TOKEN_START); // 50259 (English)
    assert_eq!(config.initial_tokens[2], 50360);                // transcribe
    assert_eq!(config.initial_tokens[3], 50364);                // no_timestamps
}

#[test]
fn test_translate_task_token_in_decode_config() {
    let config =
        crate::DecodeConfig::default().with_initial_tokens(vec![50258, 50259, 50359, 50364]);
    assert_eq!(config.initial_tokens[2], 50359);
}

#[test]
fn test_task_tokens_within_vocab_range() {
    for config in &[WhisperConfig::whisper_tiny(), WhisperConfig::large_v3_turbo()] {
        assert!(50359 < config.vocab_size, "translate token out of range");
        assert!(50360 < config.vocab_size, "transcribe token out of range");
    }
}

// ============================================================================
// Timestamp token math
// ============================================================================

#[test]
fn test_timestamp_resolution_covers_30s() {
    let steps = (CHUNK_LENGTH as f64 / 0.02) as usize;
    assert_eq!(steps, 1500, "30s / 0.02s = 1500 timestamp steps");
    let time_at_last: f64 = 1500.0 * 0.02;
    assert!((time_at_last - 30.0).abs() < 1e-10);
}

#[test]
fn test_timestamp_begin_follows_no_timestamps_token() {
    assert_eq!(TIMESTAMP_BEGIN, 50364 + 1); // NO_TIMESTAMPS_TOKEN + 1
}

// ============================================================================
// Audio constants coherence
// ============================================================================

#[test]
fn test_audio_constants_derivation_chain() {
    assert_eq!(N_SAMPLES, 16_000 * 30);       // SAMPLE_RATE * CHUNK_LENGTH
    assert_eq!(N_FRAMES, N_SAMPLES / HOP_LENGTH); // 480_000 / 160 = 3000
    assert_eq!(NUM_MEL_BINS, 128);
    assert_eq!(WhisperConfig::large_v3_turbo().num_mel_bins, NUM_MEL_BINS);
}

// ============================================================================
// Config Debug and Clone
// ============================================================================

#[test]
fn test_config_clone_preserves_equality() {
    let original = WhisperConfig::whisper_small();
    assert_eq!(original, original.clone());
}

#[test]
fn test_config_debug_contains_key_fields() {
    let debug = format!("{:?}", WhisperConfig::whisper_tiny());
    assert!(debug.contains("d_model"));
    assert!(debug.contains("vocab_size"));
}

// ============================================================================
// Layer count structure
// ============================================================================

#[test]
fn test_encoder_layer_counts() {
    let cases: &[(&str, WhisperConfig, usize)] = &[
        ("tiny", WhisperConfig::whisper_tiny(), 4),
        ("base", WhisperConfig::whisper_base(), 6),
        ("small", WhisperConfig::whisper_small(), 12),
        ("medium", WhisperConfig::whisper_medium(), 24),
        ("large-v2", WhisperConfig::whisper_large_v2(), 32),
        ("turbo", WhisperConfig::large_v3_turbo(), 32),
    ];
    for (name, config, expected) in cases {
        assert_eq!(config.encoder_layers, *expected, "{name}");
    }
}

#[test]
fn test_decoder_layer_counts() {
    let cases: &[(&str, WhisperConfig, usize)] = &[
        ("tiny", WhisperConfig::whisper_tiny(), 4),
        ("base", WhisperConfig::whisper_base(), 6),
        ("small", WhisperConfig::whisper_small(), 12),
        ("medium", WhisperConfig::whisper_medium(), 24),
        ("large-v2", WhisperConfig::whisper_large_v2(), 32),
        ("turbo", WhisperConfig::large_v3_turbo(), 4),
    ];
    for (name, config, expected) in cases {
        assert_eq!(config.decoder_layers, *expected, "{name}");
    }
}

#[test]
fn test_symmetric_layers_except_turbo() {
    for config in &[
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
        WhisperConfig::whisper_large_v2(),
    ] {
        assert_eq!(config.encoder_layers, config.decoder_layers);
    }
    let turbo = WhisperConfig::large_v3_turbo();
    assert_ne!(turbo.encoder_layers, turbo.decoder_layers);
}

// ============================================================================
// Decode config validation
// ============================================================================

#[test]
fn test_decode_config_default_validates() {
    crate::DecodeConfig::default().validate().expect("default should validate");
}

#[test]
fn test_decode_config_rejects_zero_max_length() {
    assert!(crate::DecodeConfig::default().with_max_length(0).validate().is_err());
}

#[test]
fn test_decode_config_rejects_empty_initial_tokens() {
    assert!(crate::DecodeConfig::default().with_initial_tokens(vec![]).validate().is_err());
}

#[test]
fn test_decode_config_rejects_nan_compression_threshold() {
    assert!(crate::DecodeConfig::default()
        .with_compression_ratio_threshold(f64::NAN)
        .validate()
        .is_err());
}

#[test]
fn test_decode_config_rejects_inf_logprob_threshold() {
    assert!(crate::DecodeConfig::default()
        .with_avg_logprob_threshold(f64::INFINITY)
        .validate()
        .is_err());
}

#[test]
fn test_decode_config_max_length_limit() {
    let over = crate::decode::MAX_DECODE_LENGTH + 1;
    assert!(crate::DecodeConfig::default().with_max_length(over).validate().is_err());
}
