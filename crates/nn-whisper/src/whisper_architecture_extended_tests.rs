// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Whisper architecture and configuration tests.
//!
//! Covers architecture-level invariants not addressed by existing test files:
//! - FFN dimension ratios across all size variants
//! - Encoder/decoder attention head symmetry vs asymmetry
//! - KV cache size calculations per layer and total
//! - Position embedding dimension consistency
//! - Cross-attention configuration validation (encoder-decoder d_model match)
//! - Mel spectrogram parameter validation (n_fft, hop_length, n_mels relationships)
//! - Token configuration completeness (translate, transcribe, no_timestamps)
//! - Language token range boundary validation
//! - Builder method chain idempotency
//! - Config field isolation (changing one field does not affect others)

use crate::config::{
    WhisperConfig, CHUNK_LENGTH, HOP_LENGTH, N_FFT, N_FRAMES, N_SAMPLES, NUM_MEL_BINS,
    SAMPLE_RATE,
};
use crate::tokenizer::{
    EOT_TOKEN, LANGUAGE_TOKEN_END, LANGUAGE_TOKEN_START, NO_SPEECH_TOKEN, NO_TIMESTAMPS_TOKEN,
    SOT_TOKEN, TIMESTAMP_BEGIN,
};

// ============================================================================
// FFN dimension ratios across all size variants
// ============================================================================

#[test]
fn test_ffn_dim_is_4x_d_model_all_presets() {
    let presets: &[(&str, WhisperConfig)] = &[
        ("tiny", WhisperConfig::whisper_tiny()),
        ("base", WhisperConfig::whisper_base()),
        ("small", WhisperConfig::whisper_small()),
        ("medium", WhisperConfig::whisper_medium()),
        ("large-v2", WhisperConfig::whisper_large_v2()),
        ("turbo", WhisperConfig::large_v3_turbo()),
    ];
    for (name, config) in presets {
        assert_eq!(
            config.encoder_ffn_dim,
            4 * config.d_model,
            "{name}: encoder_ffn_dim should be 4x d_model"
        );
        assert_eq!(
            config.decoder_ffn_dim,
            4 * config.d_model,
            "{name}: decoder_ffn_dim should be 4x d_model"
        );
    }
}

#[test]
fn test_ffn_dim_encoder_decoder_match_all_presets() {
    let presets = [
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
        WhisperConfig::whisper_large_v2(),
        WhisperConfig::large_v3_turbo(),
    ];
    for config in &presets {
        assert_eq!(
            config.encoder_ffn_dim, config.decoder_ffn_dim,
            "encoder and decoder FFN dims should match for d_model={}",
            config.d_model
        );
    }
}

#[test]
fn test_ffn_dim_concrete_values() {
    let cases: &[(&str, WhisperConfig, usize)] = &[
        ("tiny", WhisperConfig::whisper_tiny(), 1536),
        ("base", WhisperConfig::whisper_base(), 2048),
        ("small", WhisperConfig::whisper_small(), 3072),
        ("medium", WhisperConfig::whisper_medium(), 4096),
        ("large-v2", WhisperConfig::whisper_large_v2(), 5120),
        ("turbo", WhisperConfig::large_v3_turbo(), 5120),
    ];
    for (name, config, expected) in cases {
        assert_eq!(config.encoder_ffn_dim, *expected, "{name}: encoder_ffn_dim");
        assert_eq!(config.decoder_ffn_dim, *expected, "{name}: decoder_ffn_dim");
    }
}

// ============================================================================
// Encoder/decoder attention head symmetry
// ============================================================================

#[test]
fn test_encoder_decoder_heads_match_all_presets() {
    let presets = [
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
        WhisperConfig::whisper_large_v2(),
        WhisperConfig::large_v3_turbo(),
    ];
    for config in &presets {
        assert_eq!(
            config.encoder_attention_heads, config.decoder_attention_heads,
            "Whisper encoder and decoder heads should match for d_model={}",
            config.d_model
        );
    }
}

#[test]
fn test_decoder_heads_concrete_values() {
    let cases: &[(&str, WhisperConfig, usize)] = &[
        ("tiny", WhisperConfig::whisper_tiny(), 6),
        ("base", WhisperConfig::whisper_base(), 8),
        ("small", WhisperConfig::whisper_small(), 12),
        ("medium", WhisperConfig::whisper_medium(), 16),
        ("large-v2", WhisperConfig::whisper_large_v2(), 20),
        ("turbo", WhisperConfig::large_v3_turbo(), 20),
    ];
    for (name, config, expected) in cases {
        assert_eq!(
            config.decoder_attention_heads, *expected,
            "{name}: decoder heads"
        );
    }
}

// ============================================================================
// KV cache size calculations per layer and total
// ============================================================================

/// KV cache entries per self-attention layer for a given sequence length.
/// Each layer caches K and V tensors: 2 * seq_len * d_model floats.
fn kv_cache_floats_per_layer(config: &WhisperConfig, seq_len: usize) -> usize {
    2 * seq_len * config.d_model
}

#[test]
fn test_encoder_kv_cache_per_layer_at_max_source() {
    let cases: &[(&str, WhisperConfig, usize)] = &[
        ("tiny", WhisperConfig::whisper_tiny(), 2 * 1500 * 384),
        ("base", WhisperConfig::whisper_base(), 2 * 1500 * 512),
        ("small", WhisperConfig::whisper_small(), 2 * 1500 * 768),
        ("medium", WhisperConfig::whisper_medium(), 2 * 1500 * 1024),
        ("large-v2", WhisperConfig::whisper_large_v2(), 2 * 1500 * 1280),
        ("turbo", WhisperConfig::large_v3_turbo(), 2 * 1500 * 1280),
    ];
    for (name, config, expected) in cases {
        let per_layer = kv_cache_floats_per_layer(config, config.max_source_positions);
        assert_eq!(per_layer, *expected, "{name}: encoder KV cache per layer");
    }
}

#[test]
fn test_decoder_self_attn_kv_cache_per_layer_at_max_target() {
    let cases: &[(&str, WhisperConfig, usize)] = &[
        ("tiny", WhisperConfig::whisper_tiny(), 2 * 448 * 384),
        ("base", WhisperConfig::whisper_base(), 2 * 448 * 512),
        ("small", WhisperConfig::whisper_small(), 2 * 448 * 768),
        ("medium", WhisperConfig::whisper_medium(), 2 * 448 * 1024),
        ("large-v2", WhisperConfig::whisper_large_v2(), 2 * 448 * 1280),
        ("turbo", WhisperConfig::large_v3_turbo(), 2 * 448 * 1280),
    ];
    for (name, config, expected) in cases {
        let per_layer = kv_cache_floats_per_layer(config, config.max_target_positions);
        assert_eq!(per_layer, *expected, "{name}: decoder self-attn KV per layer");
    }
}

#[test]
fn test_total_decoder_kv_cache_scales_with_layers() {
    let tiny = WhisperConfig::whisper_tiny();
    let large = WhisperConfig::whisper_large_v2();
    let turbo = WhisperConfig::large_v3_turbo();

    // Decoder self-attention KV cache: layers * 2 * max_target * d_model
    let tiny_total = tiny.decoder_layers * kv_cache_floats_per_layer(&tiny, tiny.max_target_positions);
    let large_total = large.decoder_layers * kv_cache_floats_per_layer(&large, large.max_target_positions);
    let turbo_total = turbo.decoder_layers * kv_cache_floats_per_layer(&turbo, turbo.max_target_positions);

    // large-v2 has 32 decoder layers, turbo has 4
    assert!(large_total > turbo_total, "large-v2 decoder KV > turbo decoder KV");
    assert!(large_total > tiny_total, "large-v2 decoder KV > tiny decoder KV");

    // Turbo has same d_model as large-v2 but 8x fewer decoder layers
    assert_eq!(large.decoder_layers / turbo.decoder_layers, 8);
    assert_eq!(large_total / turbo_total, 8);
}

#[test]
fn test_cross_attention_kv_cache_per_decoder_layer() {
    // Cross-attention caches encoder output: 2 * max_source_positions * d_model per layer.
    let config = WhisperConfig::large_v3_turbo();
    let cross_kv_per_layer = kv_cache_floats_per_layer(&config, config.max_source_positions);
    let total_cross_kv = config.decoder_layers * cross_kv_per_layer;

    // 4 layers * 2 * 1500 * 1280 = 15,360,000 floats
    assert_eq!(total_cross_kv, 4 * 2 * 1500 * 1280);
}

#[test]
fn test_kv_cache_bytes_at_f32_and_f16() {
    let config = WhisperConfig::large_v3_turbo();
    let self_attn_floats = config.decoder_layers
        * kv_cache_floats_per_layer(&config, config.max_target_positions);
    let cross_attn_floats = config.decoder_layers
        * kv_cache_floats_per_layer(&config, config.max_source_positions);
    let total_floats = self_attn_floats + cross_attn_floats;

    let f32_bytes = total_floats * 4;
    let f16_bytes = total_floats * 2;

    // Turbo: 4 decoder layers
    // self-attn: 4 * 2 * 448 * 1280 = 4,587,520
    // cross-attn: 4 * 2 * 1500 * 1280 = 15,360,000
    // total floats: 19,947,520
    assert_eq!(total_floats, 4_587_520 + 15_360_000);
    assert_eq!(f32_bytes, total_floats * 4);
    assert_eq!(f16_bytes, total_floats * 2);
    assert!(f16_bytes < f32_bytes);
}

// ============================================================================
// Position embedding dimension checks
// ============================================================================

#[test]
fn test_encoder_positional_embedding_dims() {
    // Encoder uses sinusoidal positional embeddings of shape [max_source_positions, d_model].
    let presets = [
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
        WhisperConfig::whisper_large_v2(),
        WhisperConfig::large_v3_turbo(),
    ];
    for config in &presets {
        let embed_size = config.max_source_positions * config.d_model;
        assert!(embed_size > 0, "encoder positional embedding must be non-zero");
        // All presets have max_source_positions = 1500
        assert_eq!(config.max_source_positions, 1500);
    }
}

#[test]
fn test_decoder_positional_embedding_dims() {
    // Decoder uses learned positional embeddings of shape [max_target_positions, d_model].
    let presets = [
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
        WhisperConfig::whisper_large_v2(),
        WhisperConfig::large_v3_turbo(),
    ];
    for config in &presets {
        let embed_size = config.max_target_positions * config.d_model;
        assert!(embed_size > 0, "decoder positional embedding must be non-zero");
        // All presets have max_target_positions = 448
        assert_eq!(config.max_target_positions, 448);
    }
}

#[test]
fn test_encoder_positional_larger_than_decoder_positional() {
    // Encoder context (1500) is much larger than decoder context (448)
    // because audio sequences are longer than text token sequences.
    for config in &[WhisperConfig::whisper_tiny(), WhisperConfig::large_v3_turbo()] {
        assert!(
            config.max_source_positions > config.max_target_positions,
            "encoder positional context should exceed decoder"
        );
        let ratio = config.max_source_positions as f64 / config.max_target_positions as f64;
        assert!(ratio > 3.0, "encoder/decoder context ratio should be > 3x");
    }
}

// ============================================================================
// Cross-attention configuration validation
// ============================================================================

#[test]
fn test_cross_attention_d_model_shared() {
    // Cross-attention requires encoder and decoder to share d_model.
    // In Whisper, there is a single d_model field for both.
    let presets = [
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
        WhisperConfig::whisper_large_v2(),
        WhisperConfig::large_v3_turbo(),
    ];
    for config in &presets {
        // encoder and decoder both use config.d_model
        let enc_qkv_dim = config.d_model;
        let dec_qkv_dim = config.d_model;
        assert_eq!(
            enc_qkv_dim, dec_qkv_dim,
            "cross-attention requires shared d_model"
        );
    }
}

#[test]
fn test_cross_attention_head_dim_consistency() {
    // Cross-attention K,V come from encoder (d_model), Q from decoder (d_model).
    // Both encoder and decoder head_dim must be equal for cross-attention to work.
    for config in &[
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::large_v3_turbo(),
    ] {
        assert_eq!(
            config.encoder_head_dim(),
            config.decoder_head_dim(),
            "cross-attention requires matching head dims"
        );
    }
}

// ============================================================================
// Mel spectrogram parameter validation
// ============================================================================

#[test]
fn test_n_fft_greater_than_hop_length() {
    // For valid STFT: n_fft > hop_length (overlapping windows).
    assert!(N_FFT > HOP_LENGTH, "N_FFT ({N_FFT}) must exceed HOP_LENGTH ({HOP_LENGTH})");
}

#[test]
fn test_n_fft_hop_length_overlap_ratio() {
    // Standard speech processing: overlap ratio is typically 50-75%.
    // N_FFT=400, HOP_LENGTH=160 => overlap = (400-160)/400 = 60%
    let overlap = (N_FFT - HOP_LENGTH) as f64 / N_FFT as f64;
    assert!(
        overlap > 0.5 && overlap < 0.8,
        "overlap ratio {overlap} should be 50-80%"
    );
}

#[test]
fn test_mel_bins_power_of_two_or_standard() {
    // Whisper uses 80 (tiny-medium) or 128 (large/turbo) mel bins.
    // 128 is a power of two, 80 is the traditional standard.
    let tiny_mels = WhisperConfig::whisper_tiny().num_mel_bins;
    let large_mels = WhisperConfig::large_v3_turbo().num_mel_bins;
    assert_eq!(tiny_mels, 80);
    assert_eq!(large_mels, 128);
    assert!(large_mels.is_power_of_two());
}

#[test]
fn test_fft_bins_count() {
    // For real-valued FFT, frequency bins = N_FFT/2 + 1
    let fft_bins = N_FFT / 2 + 1;
    assert_eq!(fft_bins, 201);
    // mel_bins < fft_bins (mel is a compressed representation)
    assert!(NUM_MEL_BINS < fft_bins, "mel bins should be < FFT bins");
    assert!(80 < fft_bins, "80 mel bins should be < FFT bins");
}

#[test]
fn test_mel_bins_differ_between_small_and_large_models() {
    // tiny/base/small/medium use 80 mel bins, large/turbo use 128
    let small_models = [
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
    ];
    let large_models = [
        WhisperConfig::whisper_large_v2(),
        WhisperConfig::large_v3_turbo(),
    ];
    for config in &small_models {
        assert_eq!(config.num_mel_bins, 80);
    }
    for config in &large_models {
        assert_eq!(config.num_mel_bins, 128);
    }
}

// ============================================================================
// Token configuration: translate, transcribe, no_timestamps
// ============================================================================

#[test]
fn test_translate_token_value() {
    let translate: usize = 50359;
    assert_eq!(translate, LANGUAGE_TOKEN_END + 1);
}

#[test]
fn test_transcribe_token_value() {
    let transcribe: usize = 50360;
    assert_eq!(transcribe, LANGUAGE_TOKEN_END + 2);
}

#[test]
fn test_no_timestamps_token_value() {
    assert_eq!(NO_TIMESTAMPS_TOKEN, 50364);
}

#[test]
fn test_no_speech_token_precedes_no_timestamps() {
    assert!(NO_SPEECH_TOKEN < NO_TIMESTAMPS_TOKEN);
    assert_eq!(NO_TIMESTAMPS_TOKEN - NO_SPEECH_TOKEN, 1);
}

#[test]
fn test_task_and_control_token_ordering() {
    // After language tokens: translate(50359), transcribe(50360),
    // startoflm(50361), startofprev(50362), nospeech(50363),
    // notimestamps(50364), timestamps(50365+)
    let translate = 50359_usize;
    let transcribe = 50360_usize;
    let startoflm = 50361_usize;
    let startofprev = 50362_usize;

    assert_eq!(translate, LANGUAGE_TOKEN_END + 1);
    assert_eq!(transcribe, translate + 1);
    assert_eq!(startoflm, transcribe + 1);
    assert_eq!(startofprev, startoflm + 1);
    assert_eq!(NO_SPEECH_TOKEN, startofprev + 1);
    assert_eq!(NO_TIMESTAMPS_TOKEN, NO_SPEECH_TOKEN + 1);
    assert_eq!(TIMESTAMP_BEGIN, NO_TIMESTAMPS_TOKEN + 1);
}

#[test]
fn test_all_special_tokens_within_vocab() {
    // Every special token must be within the vocabulary of ALL presets.
    let special_tokens = [
        EOT_TOKEN,
        SOT_TOKEN,
        LANGUAGE_TOKEN_START,
        LANGUAGE_TOKEN_END,
        50359, // translate
        50360, // transcribe
        50361, // startoflm
        50362, // startofprev
        NO_SPEECH_TOKEN,
        NO_TIMESTAMPS_TOKEN,
        TIMESTAMP_BEGIN,
    ];
    let min_vocab = WhisperConfig::whisper_tiny().vocab_size;
    for token in &special_tokens {
        assert!(
            *token < min_vocab,
            "token {token} must be < min vocab_size {min_vocab}"
        );
    }
}

// ============================================================================
// Language token range boundary validation
// ============================================================================

#[test]
fn test_language_token_range_boundaries() {
    assert_eq!(LANGUAGE_TOKEN_START, 50259);
    assert_eq!(LANGUAGE_TOKEN_END, 50358);
    assert_eq!(LANGUAGE_TOKEN_END - LANGUAGE_TOKEN_START + 1, 100);
}

#[test]
fn test_english_is_first_language_token() {
    // English is language index 0 => LANGUAGE_TOKEN_START
    assert_eq!(LANGUAGE_TOKEN_START, SOT_TOKEN + 1);
}

#[test]
fn test_language_tokens_contiguous_and_non_overlapping() {
    // No gap between SOT and first language token
    assert_eq!(LANGUAGE_TOKEN_START, SOT_TOKEN + 1);
    // No gap between last language token and translate
    let translate = 50359_usize;
    assert_eq!(translate, LANGUAGE_TOKEN_END + 1);
}

#[test]
fn test_language_token_for_index() {
    // Language at index i has token ID = LANGUAGE_TOKEN_START + i
    for i in 0..100 {
        let token = LANGUAGE_TOKEN_START + i;
        assert!(token >= LANGUAGE_TOKEN_START);
        assert!(token <= LANGUAGE_TOKEN_END);
    }
    // Index 100 would be out of range
    assert_eq!(LANGUAGE_TOKEN_START + 100, LANGUAGE_TOKEN_END + 1);
}

// ============================================================================
// Builder method chain idempotency and field isolation
// ============================================================================

#[test]
fn test_builder_set_then_reset_restores_original() {
    let original = WhisperConfig::whisper_tiny();
    let modified = original
        .clone()
        .with_d_model(999)
        .with_d_model(original.d_model);
    assert_eq!(modified, original);
}

#[test]
fn test_builder_field_isolation_d_model() {
    let base = WhisperConfig::whisper_tiny();
    let modified = base.clone().with_d_model(256);
    assert_eq!(modified.d_model, 256);
    // All other fields unchanged
    assert_eq!(modified.num_mel_bins, base.num_mel_bins);
    assert_eq!(modified.max_source_positions, base.max_source_positions);
    assert_eq!(modified.encoder_attention_heads, base.encoder_attention_heads);
    assert_eq!(modified.encoder_layers, base.encoder_layers);
    assert_eq!(modified.encoder_ffn_dim, base.encoder_ffn_dim);
    assert_eq!(modified.vocab_size, base.vocab_size);
    assert_eq!(modified.max_target_positions, base.max_target_positions);
    assert_eq!(modified.decoder_attention_heads, base.decoder_attention_heads);
    assert_eq!(modified.decoder_layers, base.decoder_layers);
    assert_eq!(modified.decoder_ffn_dim, base.decoder_ffn_dim);
}

#[test]
fn test_builder_field_isolation_vocab_size() {
    let base = WhisperConfig::whisper_tiny();
    let modified = base.clone().with_vocab_size(100_000);
    assert_eq!(modified.vocab_size, 100_000);
    assert_eq!(modified.d_model, base.d_model);
    assert_eq!(modified.encoder_layers, base.encoder_layers);
    assert_eq!(modified.decoder_layers, base.decoder_layers);
}

#[test]
fn test_builder_field_isolation_decoder_layers() {
    let base = WhisperConfig::whisper_large_v2();
    let modified = base.clone().with_decoder_layers(8);
    assert_eq!(modified.decoder_layers, 8);
    assert_eq!(modified.encoder_layers, base.encoder_layers);
    assert_eq!(modified.d_model, base.d_model);
}

#[test]
fn test_builder_all_fields_chainable() {
    let config = WhisperConfig::whisper_tiny()
        .with_num_mel_bins(64)
        .with_max_source_positions(750)
        .with_d_model(256)
        .with_encoder_attention_heads(4)
        .with_encoder_layers(2)
        .with_encoder_ffn_dim(1024)
        .with_vocab_size(32000)
        .with_max_target_positions(224)
        .with_decoder_attention_heads(4)
        .with_decoder_layers(2)
        .with_decoder_ffn_dim(1024);

    assert_eq!(config.num_mel_bins, 64);
    assert_eq!(config.max_source_positions, 750);
    assert_eq!(config.d_model, 256);
    assert_eq!(config.encoder_attention_heads, 4);
    assert_eq!(config.encoder_layers, 2);
    assert_eq!(config.encoder_ffn_dim, 1024);
    assert_eq!(config.vocab_size, 32000);
    assert_eq!(config.max_target_positions, 224);
    assert_eq!(config.decoder_attention_heads, 4);
    assert_eq!(config.decoder_layers, 2);
    assert_eq!(config.decoder_ffn_dim, 1024);
    config.validate().expect("custom config should validate");
}

// ============================================================================
// Vocab size differences across model variants
// ============================================================================

#[test]
fn test_vocab_size_values() {
    // tiny through large-v2 have vocab 51865; turbo has 51866
    let cases: &[(&str, WhisperConfig, usize)] = &[
        ("tiny", WhisperConfig::whisper_tiny(), 51865),
        ("base", WhisperConfig::whisper_base(), 51865),
        ("small", WhisperConfig::whisper_small(), 51865),
        ("medium", WhisperConfig::whisper_medium(), 51865),
        ("large-v2", WhisperConfig::whisper_large_v2(), 51865),
        ("turbo", WhisperConfig::large_v3_turbo(), 51866),
    ];
    for (name, config, expected) in cases {
        assert_eq!(config.vocab_size, *expected, "{name}: vocab_size");
    }
}

#[test]
fn test_vocab_accommodates_all_timestamp_tokens() {
    // Timestamp tokens: 50365 through 50365 + 1500 (30s / 0.02s)
    let last_timestamp = TIMESTAMP_BEGIN + 1500;
    for config in &[WhisperConfig::whisper_tiny(), WhisperConfig::large_v3_turbo()] {
        assert!(
            config.vocab_size > last_timestamp,
            "vocab_size ({}) must accommodate last timestamp token ({})",
            config.vocab_size,
            last_timestamp
        );
    }
}

// ============================================================================
// Head dimension calculation edge cases
// ============================================================================

#[test]
fn test_head_dim_custom_configs() {
    let config = WhisperConfig::whisper_tiny()
        .with_d_model(512)
        .with_encoder_attention_heads(8)
        .with_decoder_attention_heads(8);
    assert_eq!(config.encoder_head_dim(), 64);
    assert_eq!(config.decoder_head_dim(), 64);
}

#[test]
fn test_head_dim_non_64() {
    // Construct a valid config with head_dim != 64
    let config = WhisperConfig::whisper_tiny()
        .with_d_model(256)
        .with_encoder_attention_heads(4)
        .with_decoder_attention_heads(4);
    config.validate().unwrap();
    assert_eq!(config.encoder_head_dim(), 64);

    let config2 = WhisperConfig::whisper_tiny()
        .with_d_model(256)
        .with_encoder_attention_heads(8)
        .with_decoder_attention_heads(8);
    config2.validate().unwrap();
    assert_eq!(config2.encoder_head_dim(), 32);
    assert_eq!(config2.decoder_head_dim(), 32);
}

#[test]
fn test_head_dim_single_head() {
    let config = WhisperConfig::whisper_tiny()
        .with_d_model(128)
        .with_encoder_attention_heads(1)
        .with_decoder_attention_heads(1);
    config.validate().unwrap();
    assert_eq!(config.encoder_head_dim(), 128);
    assert_eq!(config.decoder_head_dim(), 128);
}

// ============================================================================
// Max source/target positions consistency
// ============================================================================

#[test]
fn test_max_source_positions_uniform_across_presets() {
    let presets = [
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
        WhisperConfig::whisper_large_v2(),
        WhisperConfig::large_v3_turbo(),
    ];
    for config in &presets {
        assert_eq!(
            config.max_source_positions, 1500,
            "all presets share max_source_positions=1500"
        );
    }
}

#[test]
fn test_max_target_positions_uniform_across_presets() {
    let presets = [
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
        WhisperConfig::whisper_large_v2(),
        WhisperConfig::large_v3_turbo(),
    ];
    for config in &presets {
        assert_eq!(
            config.max_target_positions, 448,
            "all presets share max_target_positions=448"
        );
    }
}

// ============================================================================
// Audio duration and frame calculations
// ============================================================================

#[test]
fn test_audio_duration_from_n_samples() {
    let duration_secs = N_SAMPLES as f64 / SAMPLE_RATE as f64;
    assert!((duration_secs - 30.0).abs() < 1e-10);
}

#[test]
fn test_frame_to_time_conversion() {
    // Each frame corresponds to HOP_LENGTH/SAMPLE_RATE seconds
    let frame_duration = HOP_LENGTH as f64 / SAMPLE_RATE as f64;
    assert!((frame_duration - 0.01).abs() < 1e-10, "each frame = 10ms");

    // Frame index * frame_duration gives time in seconds
    let frame_1000_time = 1000.0 * frame_duration;
    assert!((frame_1000_time - 10.0).abs() < 1e-10, "frame 1000 = 10s");
}

#[test]
fn test_n_frames_matches_30s_at_10ms_stride() {
    // 30 seconds / 0.01 seconds per frame = 3000 frames
    let expected_frames = (CHUNK_LENGTH as f64 / (HOP_LENGTH as f64 / SAMPLE_RATE as f64)) as usize;
    assert_eq!(expected_frames, N_FRAMES);
    assert_eq!(N_FRAMES, 3000);
}

// ============================================================================
// Encoder output sequence length relationship
// ============================================================================

#[test]
fn test_encoder_output_is_half_mel_frames() {
    // Conv stem: conv1(stride=1) -> conv2(stride=2) halves the sequence.
    // For standard 30s audio: 3000 frames -> 1500 encoder positions.
    assert_eq!(N_FRAMES / 2, 1500);
    let config = WhisperConfig::large_v3_turbo();
    assert_eq!(config.max_source_positions, N_FRAMES / 2);
}

#[test]
fn test_timestamp_token_to_encoder_frame_ratio() {
    // 1500 timestamp tokens map to 1500 encoder positions (1:1).
    // Each timestamp = 0.02s, encoder position = 0.02s (2 mel frames / 1 position).
    let timestamp_count = 1501_usize; // 0.00 through 30.00 inclusive
    let encoder_positions = 1500_usize;
    // Timestamp resolution: 30s / 1500 steps = 0.02s per step
    let ts_resolution = CHUNK_LENGTH as f64 / timestamp_count as f64;
    let enc_resolution = CHUNK_LENGTH as f64 / encoder_positions as f64;
    assert!((ts_resolution - 0.02).abs() < 0.001);
    assert!((enc_resolution - 0.02).abs() < 0.001);
}
