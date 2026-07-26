// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Model configuration tests: all preset configs, head_dim consistency,
//! vocab size constants, max sequence length, FFN dim ratios,
//! and builder method coverage. Part of #4186.

use crate::config::WhisperConfig;

// ============================================================================
// All preset configs: verify specific field values
// ============================================================================

#[test]
fn test_whisper_tiny_all_fields() {
    let c = WhisperConfig::whisper_tiny();
    assert_eq!(c.num_mel_bins, 80);
    assert_eq!(c.max_source_positions, 1500);
    assert_eq!(c.d_model, 384);
    assert_eq!(c.encoder_attention_heads, 6);
    assert_eq!(c.encoder_layers, 4);
    assert_eq!(c.encoder_ffn_dim, 1536);
    assert_eq!(c.vocab_size, 51865);
    assert_eq!(c.max_target_positions, 448);
    assert_eq!(c.decoder_attention_heads, 6);
    assert_eq!(c.decoder_layers, 4);
    assert_eq!(c.decoder_ffn_dim, 1536);
}

#[test]
fn test_whisper_base_all_fields() {
    let c = WhisperConfig::whisper_base();
    assert_eq!(c.num_mel_bins, 80);
    assert_eq!(c.max_source_positions, 1500);
    assert_eq!(c.d_model, 512);
    assert_eq!(c.encoder_attention_heads, 8);
    assert_eq!(c.encoder_layers, 6);
    assert_eq!(c.encoder_ffn_dim, 2048);
    assert_eq!(c.vocab_size, 51865);
    assert_eq!(c.max_target_positions, 448);
    assert_eq!(c.decoder_attention_heads, 8);
    assert_eq!(c.decoder_layers, 6);
    assert_eq!(c.decoder_ffn_dim, 2048);
}

#[test]
fn test_whisper_small_all_fields() {
    let c = WhisperConfig::whisper_small();
    assert_eq!(c.num_mel_bins, 80);
    assert_eq!(c.max_source_positions, 1500);
    assert_eq!(c.d_model, 768);
    assert_eq!(c.encoder_attention_heads, 12);
    assert_eq!(c.encoder_layers, 12);
    assert_eq!(c.encoder_ffn_dim, 3072);
    assert_eq!(c.vocab_size, 51865);
    assert_eq!(c.max_target_positions, 448);
    assert_eq!(c.decoder_attention_heads, 12);
    assert_eq!(c.decoder_layers, 12);
    assert_eq!(c.decoder_ffn_dim, 3072);
}

#[test]
fn test_whisper_medium_all_fields() {
    let c = WhisperConfig::whisper_medium();
    assert_eq!(c.num_mel_bins, 80);
    assert_eq!(c.max_source_positions, 1500);
    assert_eq!(c.d_model, 1024);
    assert_eq!(c.encoder_attention_heads, 16);
    assert_eq!(c.encoder_layers, 24);
    assert_eq!(c.encoder_ffn_dim, 4096);
    assert_eq!(c.vocab_size, 51865);
    assert_eq!(c.max_target_positions, 448);
    assert_eq!(c.decoder_attention_heads, 16);
    assert_eq!(c.decoder_layers, 24);
    assert_eq!(c.decoder_ffn_dim, 4096);
}

#[test]
fn test_whisper_large_v2_all_fields() {
    let c = WhisperConfig::whisper_large_v2();
    assert_eq!(c.num_mel_bins, 128);
    assert_eq!(c.max_source_positions, 1500);
    assert_eq!(c.d_model, 1280);
    assert_eq!(c.encoder_attention_heads, 20);
    assert_eq!(c.encoder_layers, 32);
    assert_eq!(c.encoder_ffn_dim, 5120);
    assert_eq!(c.vocab_size, 51865);
    assert_eq!(c.max_target_positions, 448);
    assert_eq!(c.decoder_attention_heads, 20);
    assert_eq!(c.decoder_layers, 32);
    assert_eq!(c.decoder_ffn_dim, 5120);
}

#[test]
fn test_large_v3_turbo_all_fields() {
    let c = WhisperConfig::large_v3_turbo();
    assert_eq!(c.num_mel_bins, 128);
    assert_eq!(c.max_source_positions, 1500);
    assert_eq!(c.d_model, 1280);
    assert_eq!(c.encoder_attention_heads, 20);
    assert_eq!(c.encoder_layers, 32);
    assert_eq!(c.encoder_ffn_dim, 5120);
    assert_eq!(c.vocab_size, 51866); // turbo has 51866, not 51865
    assert_eq!(c.max_target_positions, 448);
    assert_eq!(c.decoder_attention_heads, 20);
    assert_eq!(c.decoder_layers, 4); // distilled decoder
    assert_eq!(c.decoder_ffn_dim, 5120);
}

// ============================================================================
// Head dimension consistency
// ============================================================================

#[test]
fn test_head_dim_consistency_all_presets() {
    let presets: Vec<(&str, WhisperConfig)> = vec![
        ("tiny", WhisperConfig::whisper_tiny()),
        ("base", WhisperConfig::whisper_base()),
        ("small", WhisperConfig::whisper_small()),
        ("medium", WhisperConfig::whisper_medium()),
        ("large-v2", WhisperConfig::whisper_large_v2()),
        ("turbo", WhisperConfig::large_v3_turbo()),
    ];

    for (name, config) in &presets {
        let enc_head_dim = config.encoder_head_dim();
        let dec_head_dim = config.decoder_head_dim();

        // d_model must be exactly divisible.
        assert_eq!(
            config.d_model % config.encoder_attention_heads,
            0,
            "{name}: d_model ({}) not divisible by encoder_heads ({})",
            config.d_model,
            config.encoder_attention_heads
        );
        assert_eq!(
            config.d_model % config.decoder_attention_heads,
            0,
            "{name}: d_model ({}) not divisible by decoder_heads ({})",
            config.d_model,
            config.decoder_attention_heads
        );

        // head_dim * heads = d_model.
        assert_eq!(
            enc_head_dim * config.encoder_attention_heads,
            config.d_model,
            "{name}: encoder head_dim * heads != d_model"
        );
        assert_eq!(
            dec_head_dim * config.decoder_attention_heads,
            config.d_model,
            "{name}: decoder head_dim * heads != d_model"
        );

        // Whisper uses the same head_dim for encoder and decoder.
        assert_eq!(
            enc_head_dim, dec_head_dim,
            "{name}: encoder head_dim ({enc_head_dim}) != decoder head_dim ({dec_head_dim})"
        );
    }
}

#[test]
fn test_head_dim_values_are_power_of_two_compatible() {
    // All Whisper presets use head_dim=64 (which is important for
    // efficient GPU attention implementations).
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
            config.encoder_head_dim(),
            64,
            "expected head_dim=64 for d_model={}",
            config.d_model
        );
    }
}

// ============================================================================
// Vocab size constants
// ============================================================================

#[test]
fn test_vocab_size_standard_models() {
    // Whisper tiny/base/small/medium/large-v2 all use vocab_size=51865.
    for config in &[
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
        WhisperConfig::whisper_large_v2(),
    ] {
        assert_eq!(
            config.vocab_size, 51865,
            "standard Whisper models use vocab_size=51865, got {}",
            config.vocab_size
        );
    }
}

#[test]
fn test_vocab_size_turbo() {
    // large-v3-turbo has 51866 (one extra token).
    let config = WhisperConfig::large_v3_turbo();
    assert_eq!(config.vocab_size, 51866);
}

#[test]
fn test_vocab_size_sufficient_for_special_tokens() {
    // Whisper special tokens go up to timestamp tokens (50365+).
    // Verify all configs have vocab_size > 50365.
    let presets = [
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
        WhisperConfig::whisper_large_v2(),
        WhisperConfig::large_v3_turbo(),
    ];
    for config in &presets {
        assert!(
            config.vocab_size > 50365,
            "vocab_size ({}) must be > 50365 for timestamp tokens",
            config.vocab_size
        );
    }
}

// ============================================================================
// Max sequence length
// ============================================================================

#[test]
fn test_max_source_positions_all_presets() {
    // All Whisper models use max_source_positions=1500 (corresponds to 30s
    // of audio after Conv1d stride-2 downsampling: 3000/2 = 1500).
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
            "max_source_positions should be 1500"
        );
    }
}

#[test]
fn test_max_target_positions_all_presets() {
    // All Whisper models use max_target_positions=448.
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
            "max_target_positions should be 448"
        );
    }
}

// ============================================================================
// FFN dimension ratio
// ============================================================================

#[test]
fn test_ffn_dim_is_4x_d_model_all_presets() {
    // All Whisper models use encoder_ffn_dim = decoder_ffn_dim = 4 * d_model.
    let presets: Vec<(&str, WhisperConfig)> = vec![
        ("tiny", WhisperConfig::whisper_tiny()),
        ("base", WhisperConfig::whisper_base()),
        ("small", WhisperConfig::whisper_small()),
        ("medium", WhisperConfig::whisper_medium()),
        ("large-v2", WhisperConfig::whisper_large_v2()),
        ("turbo", WhisperConfig::large_v3_turbo()),
    ];

    for (name, config) in &presets {
        assert_eq!(
            config.encoder_ffn_dim,
            4 * config.d_model,
            "{name}: encoder_ffn_dim ({}) != 4 * d_model ({})",
            config.encoder_ffn_dim,
            4 * config.d_model
        );
        assert_eq!(
            config.decoder_ffn_dim,
            4 * config.d_model,
            "{name}: decoder_ffn_dim ({}) != 4 * d_model ({})",
            config.decoder_ffn_dim,
            4 * config.d_model
        );
    }
}

// ============================================================================
// Builder method coverage
// ============================================================================

#[test]
fn test_builder_with_num_mel_bins() {
    let c = WhisperConfig::whisper_tiny().with_num_mel_bins(128);
    assert_eq!(c.num_mel_bins, 128);
    // Other fields unchanged.
    assert_eq!(c.d_model, 384);
}

#[test]
fn test_builder_with_max_source_positions() {
    let c = WhisperConfig::whisper_tiny().with_max_source_positions(3000);
    assert_eq!(c.max_source_positions, 3000);
}

#[test]
fn test_builder_with_vocab_size() {
    let c = WhisperConfig::whisper_tiny().with_vocab_size(100000);
    assert_eq!(c.vocab_size, 100000);
}

#[test]
fn test_builder_with_encoder_ffn_dim() {
    let c = WhisperConfig::whisper_tiny().with_encoder_ffn_dim(2048);
    assert_eq!(c.encoder_ffn_dim, 2048);
}

#[test]
fn test_builder_with_decoder_ffn_dim() {
    let c = WhisperConfig::whisper_tiny().with_decoder_ffn_dim(4096);
    assert_eq!(c.decoder_ffn_dim, 4096);
}

#[test]
fn test_builder_chaining_all_with_methods() {
    let c = WhisperConfig::whisper_tiny()
        .with_num_mel_bins(128)
        .with_max_source_positions(2000)
        .with_d_model(512)
        .with_encoder_attention_heads(8)
        .with_encoder_layers(8)
        .with_encoder_ffn_dim(2048)
        .with_vocab_size(60000)
        .with_max_target_positions(512)
        .with_decoder_attention_heads(8)
        .with_decoder_layers(8)
        .with_decoder_ffn_dim(2048);

    assert_eq!(c.num_mel_bins, 128);
    assert_eq!(c.max_source_positions, 2000);
    assert_eq!(c.d_model, 512);
    assert_eq!(c.encoder_attention_heads, 8);
    assert_eq!(c.encoder_layers, 8);
    assert_eq!(c.encoder_ffn_dim, 2048);
    assert_eq!(c.vocab_size, 60000);
    assert_eq!(c.max_target_positions, 512);
    assert_eq!(c.decoder_attention_heads, 8);
    assert_eq!(c.decoder_layers, 8);
    assert_eq!(c.decoder_ffn_dim, 2048);

    c.validate().expect("chained config should validate");
}

// ============================================================================
// Mel bins: tiny/base/small/medium use 80, large uses 128
// ============================================================================

#[test]
fn test_mel_bins_80_for_smaller_models() {
    for config in &[
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
    ] {
        assert_eq!(
            config.num_mel_bins, 80,
            "smaller Whisper models use 80 mel bins"
        );
    }
}

#[test]
fn test_mel_bins_128_for_large_models() {
    for config in &[
        WhisperConfig::whisper_large_v2(),
        WhisperConfig::large_v3_turbo(),
    ] {
        assert_eq!(
            config.num_mel_bins, 128,
            "large Whisper models use 128 mel bins"
        );
    }
}
