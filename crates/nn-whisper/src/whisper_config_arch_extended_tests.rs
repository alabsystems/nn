// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for WhisperConfig construction, architecture shape math,
//! parameter count estimation, tokenizer-config consistency, and audio
//! constant coherence. Part of #4495.
//!
//! Focuses on properties NOT covered by existing test files:
//! - Full model parameter count estimation (encoder + decoder + conv + embeddings)
//! - Encoder conv stem stride-2 downsampling math
//! - Decoder weight shape derivation from config
//! - Audio constant coherence (N_SAMPLES, N_FRAMES, HOP_LENGTH)
//! - Turbo vs large-v2 structural diff
//! - Config equality semantics
//! - Encoder output sequence length formula for arbitrary mel lengths
//! - Multi-config model loading to verify shape plumbing

use crate::config::{
    WhisperConfig, CHUNK_LENGTH, HOP_LENGTH, N_FFT, N_FRAMES, N_SAMPLES, NUM_MEL_BINS,
    SAMPLE_RATE,
};
use crate::test_utils::tiny_config;
use crate::tokenizer::{
    EOT_TOKEN, LANGUAGE_TOKEN_END, LANGUAGE_TOKEN_START, NO_SPEECH_TOKEN, SOT_TOKEN,
    TIMESTAMP_BEGIN,
};
use crate::WhisperModel;
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};

// ============================================================================
// Audio constants coherence
// ============================================================================

#[test]
fn test_audio_constants_n_samples_equals_sample_rate_times_chunk_length() {
    assert_eq!(
        N_SAMPLES,
        SAMPLE_RATE * CHUNK_LENGTH,
        "N_SAMPLES should be SAMPLE_RATE * CHUNK_LENGTH"
    );
}

#[test]
fn test_audio_constants_n_frames_equals_n_samples_div_hop_length() {
    assert_eq!(
        N_FRAMES,
        N_SAMPLES / HOP_LENGTH,
        "N_FRAMES should be N_SAMPLES / HOP_LENGTH"
    );
}

#[test]
fn test_audio_constants_concrete_values() {
    assert_eq!(SAMPLE_RATE, 16_000);
    assert_eq!(N_FFT, 400);
    assert_eq!(HOP_LENGTH, 160);
    assert_eq!(CHUNK_LENGTH, 30);
    assert_eq!(N_SAMPLES, 480_000);
    assert_eq!(N_FRAMES, 3_000);
    assert_eq!(NUM_MEL_BINS, 128);
}

#[test]
fn test_n_fft_is_25ms_window() {
    // N_FFT = 400 at 16kHz = 25ms window, standard for speech processing.
    let window_ms = (N_FFT as f64 / SAMPLE_RATE as f64) * 1000.0;
    assert!((window_ms - 25.0).abs() < 0.01, "N_FFT should be a 25ms window");
}

#[test]
fn test_hop_length_is_10ms_stride() {
    // HOP_LENGTH = 160 at 16kHz = 10ms stride.
    let stride_ms = (HOP_LENGTH as f64 / SAMPLE_RATE as f64) * 1000.0;
    assert!(
        (stride_ms - 10.0).abs() < 0.01,
        "HOP_LENGTH should be a 10ms stride"
    );
}

// ============================================================================
// Encoder conv stem downsampling math
// ============================================================================

/// Compute the output length after a 1D convolution.
/// Formula: floor((input_len + 2*padding - kernel_size) / stride) + 1
fn conv1d_output_len(input_len: usize, kernel_size: usize, stride: usize, padding: usize) -> usize {
    (input_len + 2 * padding - kernel_size) / stride + 1
}

#[test]
fn test_encoder_conv_stem_output_formula_30s_large() {
    // For 30s audio at large config (128 mel, 3000 frames):
    // conv1: kernel=3, stride=1, pad=1 => output=3000
    // conv2: kernel=3, stride=2, pad=1 => output=1500
    let mel_frames = N_FRAMES; // 3000
    let after_conv1 = conv1d_output_len(mel_frames, 3, 1, 1);
    assert_eq!(after_conv1, 3000, "conv1 preserves length (stride=1, pad=1)");
    let after_conv2 = conv1d_output_len(after_conv1, 3, 2, 1);
    assert_eq!(after_conv2, 1500, "conv2 halves length (stride=2, pad=1)");

    // This matches max_source_positions for all configs.
    let config = WhisperConfig::large_v3_turbo();
    assert_eq!(after_conv2, config.max_source_positions);
}

#[test]
fn test_encoder_conv_stem_output_formula_arbitrary_lengths() {
    // Verify the formula for various mel frame lengths.
    let test_cases = [
        (4, 2),   // minimal
        (8, 4),
        (16, 8),
        (100, 50),
        (1000, 500),
        (3000, 1500),
    ];
    for (mel_len, expected_enc_seq) in test_cases {
        let after_conv1 = conv1d_output_len(mel_len, 3, 1, 1);
        assert_eq!(after_conv1, mel_len, "conv1 preserves length for mel_len={mel_len}");
        let after_conv2 = conv1d_output_len(after_conv1, 3, 2, 1);
        assert_eq!(
            after_conv2, expected_enc_seq,
            "conv2 halves for mel_len={mel_len}"
        );
    }
}

#[test]
fn test_encoder_conv_stem_odd_mel_length() {
    // Odd mel length: conv2 stride-2 floors the result.
    // mel_len=5: conv1 output=5, conv2 output=(5+2-3)/2+1=3
    let after_conv1 = conv1d_output_len(5, 3, 1, 1);
    assert_eq!(after_conv1, 5);
    let after_conv2 = conv1d_output_len(after_conv1, 3, 2, 1);
    assert_eq!(after_conv2, 3);
}

// ============================================================================
// Full model parameter count estimation
// ============================================================================

/// Estimate total Whisper model parameter count from config.
///
/// Components:
/// - Conv1d stem: conv1 [d, mel, 3] + bias [d] + conv2 [d, d, 3] + bias [d]
/// - Encoder positional embedding: sinusoidal (not learned, excluded)
/// - Encoder blocks (N_enc): self_attn (4 * d^2 + 4*d bias) + FFN (2 * d * ffn + d + ffn) + LN (4*d)
/// - Encoder final LN: 2*d
/// - Decoder token embedding: vocab * d
/// - Decoder positional embedding: max_target * d
/// - Decoder blocks (N_dec): self_attn + cross_attn (2 * (4*d^2 + 4*d)) + FFN + LN (6*d)
/// - Decoder final LN: 2*d
/// - Output projection: tied with token embedding (no extra params)
fn estimate_total_params(c: &WhisperConfig) -> usize {
    let d = c.d_model;
    let mel = c.num_mel_bins;

    // Conv stem
    let conv1_params = d * mel * 3 + d; // weight + bias
    let conv2_params = d * d * 3 + d;

    // Encoder per-layer: self_attn (Q,K,V,O projections + biases for Q,V,O; K has no bias in Whisper)
    // Actually Whisper K has no bias, so: 4 weight matrices + 3 biases
    let attn_weight_params = 4 * d * d; // Q, K, V, O each [d, d]
    let attn_bias_params = 3 * d; // Q, V, O biases (K has no bias)
    let enc_attn_params = attn_weight_params + attn_bias_params;
    let enc_ffn_params = d * c.encoder_ffn_dim + c.encoder_ffn_dim + c.encoder_ffn_dim * d + d;
    let enc_ln_params = 4 * d; // 2 layer norms * (weight + bias)
    let enc_per_layer = enc_attn_params + enc_ffn_params + enc_ln_params;
    let enc_layers_total = enc_per_layer * c.encoder_layers;

    // Encoder final LN
    let enc_final_ln = 2 * d;

    // Decoder token embedding
    let dec_token_emb = c.vocab_size * d;
    // Decoder positional embedding (learned)
    let dec_pos_emb = c.max_target_positions * d;

    // Decoder per-layer: self_attn + cross_attn + FFN + 3 layer norms
    let dec_attn_params = 2 * (attn_weight_params + attn_bias_params); // self + cross
    let dec_ffn_params = d * c.decoder_ffn_dim + c.decoder_ffn_dim + c.decoder_ffn_dim * d + d;
    let dec_ln_params = 6 * d; // 3 layer norms * (weight + bias)
    let dec_per_layer = dec_attn_params + dec_ffn_params + dec_ln_params;
    let dec_layers_total = dec_per_layer * c.decoder_layers;

    // Decoder final LN
    let dec_final_ln = 2 * d;

    conv1_params
        + conv2_params
        + enc_layers_total
        + enc_final_ln
        + dec_token_emb
        + dec_pos_emb
        + dec_layers_total
        + dec_final_ln
}

#[test]
fn test_param_count_tiny_within_expected_range() {
    // whisper-tiny: ~39M params total
    let c = WhisperConfig::whisper_tiny();
    let est = estimate_total_params(&c);
    assert!(
        est > 30_000_000 && est < 50_000_000,
        "whisper-tiny estimate {est} should be ~39M"
    );
}

#[test]
fn test_param_count_base_within_expected_range() {
    // whisper-base: ~74M params total
    let c = WhisperConfig::whisper_base();
    let est = estimate_total_params(&c);
    assert!(
        est > 60_000_000 && est < 90_000_000,
        "whisper-base estimate {est} should be ~74M"
    );
}

#[test]
fn test_param_count_small_within_expected_range() {
    // whisper-small: ~244M params total
    let c = WhisperConfig::whisper_small();
    let est = estimate_total_params(&c);
    assert!(
        est > 200_000_000 && est < 300_000_000,
        "whisper-small estimate {est} should be ~244M"
    );
}

#[test]
fn test_param_count_medium_within_expected_range() {
    // whisper-medium: ~769M params total
    let c = WhisperConfig::whisper_medium();
    let est = estimate_total_params(&c);
    assert!(
        est > 600_000_000 && est < 900_000_000,
        "whisper-medium estimate {est} should be ~769M"
    );
}

#[test]
fn test_param_count_large_v2_within_expected_range() {
    // whisper-large-v2: ~1550M params total
    let c = WhisperConfig::whisper_large_v2();
    let est = estimate_total_params(&c);
    assert!(
        est > 1_200_000_000 && est < 1_800_000_000,
        "whisper-large-v2 estimate {est} should be ~1550M"
    );
}

#[test]
fn test_param_count_turbo_less_than_large_v2() {
    // Turbo has same encoder but only 4 decoder layers vs 32.
    let turbo = estimate_total_params(&WhisperConfig::large_v3_turbo());
    let large = estimate_total_params(&WhisperConfig::whisper_large_v2());
    assert!(
        turbo < large,
        "turbo ({turbo}) should have fewer params than large-v2 ({large})"
    );
}

#[test]
fn test_param_count_monotonic_tiny_to_large() {
    let sizes = [
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
        WhisperConfig::whisper_large_v2(),
    ];
    let counts: Vec<usize> = sizes.iter().map(estimate_total_params).collect();
    for i in 1..counts.len() {
        assert!(
            counts[i] > counts[i - 1],
            "param count should increase: {} ({}) vs {} ({})",
            counts[i - 1],
            i - 1,
            counts[i],
            i
        );
    }
}

// ============================================================================
// Turbo vs large-v2 structural diff
// ============================================================================

#[test]
fn test_turbo_shares_encoder_with_large_v2() {
    let turbo = WhisperConfig::large_v3_turbo();
    let large = WhisperConfig::whisper_large_v2();

    // Same encoder architecture
    assert_eq!(turbo.d_model, large.d_model);
    assert_eq!(turbo.encoder_layers, large.encoder_layers);
    assert_eq!(turbo.encoder_attention_heads, large.encoder_attention_heads);
    assert_eq!(turbo.encoder_ffn_dim, large.encoder_ffn_dim);
    assert_eq!(turbo.num_mel_bins, large.num_mel_bins);
    assert_eq!(turbo.max_source_positions, large.max_source_positions);
}

#[test]
fn test_turbo_has_distilled_decoder() {
    let turbo = WhisperConfig::large_v3_turbo();
    let large = WhisperConfig::whisper_large_v2();

    // Distilled decoder: fewer layers
    assert_eq!(turbo.decoder_layers, 4);
    assert_eq!(large.decoder_layers, 32);
    assert!(turbo.decoder_layers < large.decoder_layers);

    // Same decoder width and attention config
    assert_eq!(turbo.decoder_attention_heads, large.decoder_attention_heads);
    assert_eq!(turbo.decoder_ffn_dim, large.decoder_ffn_dim);
    assert_eq!(turbo.max_target_positions, large.max_target_positions);
}

#[test]
fn test_turbo_has_one_extra_vocab_token() {
    let turbo = WhisperConfig::large_v3_turbo();
    let large = WhisperConfig::whisper_large_v2();
    assert_eq!(turbo.vocab_size, large.vocab_size + 1);
}

// ============================================================================
// Config equality semantics
// ============================================================================

#[test]
fn test_config_equality_same_preset() {
    let a = WhisperConfig::whisper_tiny();
    let b = WhisperConfig::whisper_tiny();
    assert_eq!(a, b, "same preset factory should produce equal configs");
}

#[test]
fn test_config_inequality_different_presets() {
    let tiny = WhisperConfig::whisper_tiny();
    let base = WhisperConfig::whisper_base();
    assert_ne!(tiny, base, "different presets should not be equal");
}

#[test]
fn test_config_equality_after_builder_reconstruction() {
    // Reconstruct turbo config using builder from tiny base.
    let turbo_direct = WhisperConfig::large_v3_turbo();
    let turbo_built = WhisperConfig::whisper_tiny()
        .with_num_mel_bins(128)
        .with_max_source_positions(1500)
        .with_d_model(1280)
        .with_encoder_attention_heads(20)
        .with_encoder_layers(32)
        .with_encoder_ffn_dim(5120)
        .with_vocab_size(51866)
        .with_max_target_positions(448)
        .with_decoder_attention_heads(20)
        .with_decoder_layers(4)
        .with_decoder_ffn_dim(5120);
    assert_eq!(turbo_direct, turbo_built);
}

#[test]
fn test_default_config_equality() {
    let default = WhisperConfig::default();
    let turbo = WhisperConfig::large_v3_turbo();
    assert_eq!(default, turbo, "Default should equal large_v3_turbo");
}

// ============================================================================
// Tokenizer special token range vs config vocab_size
// ============================================================================

#[test]
fn test_special_tokens_within_vocab_range_all_presets() {
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
            EOT_TOKEN < config.vocab_size,
            "EOT_TOKEN ({}) must be < vocab_size ({})",
            EOT_TOKEN,
            config.vocab_size
        );
        assert!(
            SOT_TOKEN < config.vocab_size,
            "SOT_TOKEN ({}) must be < vocab_size ({})",
            SOT_TOKEN,
            config.vocab_size
        );
        assert!(
            NO_SPEECH_TOKEN < config.vocab_size,
            "NO_SPEECH_TOKEN ({}) must be < vocab_size ({})",
            NO_SPEECH_TOKEN,
            config.vocab_size
        );
        assert!(
            LANGUAGE_TOKEN_START < config.vocab_size,
            "LANGUAGE_TOKEN_START ({}) must be < vocab_size ({})",
            LANGUAGE_TOKEN_START,
            config.vocab_size
        );
        assert!(
            LANGUAGE_TOKEN_END < config.vocab_size,
            "LANGUAGE_TOKEN_END ({}) must be < vocab_size ({})",
            LANGUAGE_TOKEN_END,
            config.vocab_size
        );
    }
}

#[test]
fn test_timestamp_begin_within_vocab_range() {
    // TIMESTAMP_BEGIN (50365) must be within vocab for timestamp tokens to work.
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
            TIMESTAMP_BEGIN < config.vocab_size,
            "TIMESTAMP_BEGIN ({}) must be < vocab_size ({})",
            TIMESTAMP_BEGIN,
            config.vocab_size
        );
    }
}

#[test]
fn test_language_token_range_is_100_languages() {
    // Whisper supports 100 languages: token IDs 50259 through 50358.
    assert_eq!(LANGUAGE_TOKEN_END - LANGUAGE_TOKEN_START + 1, 100);
}

#[test]
fn test_special_token_ordering() {
    // EOT < SOT < LANGUAGE_START < ... < LANGUAGE_END < NO_SPEECH < TIMESTAMP_BEGIN
    assert!(EOT_TOKEN < SOT_TOKEN);
    assert!(SOT_TOKEN < LANGUAGE_TOKEN_START);
    assert!(LANGUAGE_TOKEN_START <= LANGUAGE_TOKEN_END);
    assert!(LANGUAGE_TOKEN_END < NO_SPEECH_TOKEN);
    assert!(NO_SPEECH_TOKEN < TIMESTAMP_BEGIN);
}

// ============================================================================
// Encoder output shape for multiple config sizes (zero-weight model loading)
// ============================================================================

#[test]
fn test_encoder_output_shape_with_tiny_config() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    // mel_len=16: after conv1 (stride=1) -> 16, after conv2 (stride=2) -> 8
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();
    let out = model.encode(&mel).unwrap();

    let expected_seq = conv1d_output_len(conv1d_output_len(16, 3, 1, 1), 3, 2, 1);
    assert_eq!(out.rank(), 3);
    assert_eq!(out.dim(0).unwrap(), 1); // batch
    assert_eq!(out.dim(1).unwrap(), expected_seq); // seq_len
    assert_eq!(out.dim(2).unwrap(), config.d_model); // d_model
}

#[test]
fn test_encoder_output_batch_dim_propagation() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());

    for batch_size in [1, 2, 4] {
        let mut model = WhisperModel::load(&vb, config.clone()).unwrap();
        let mel = DynTensor::zeros(
            &[batch_size, config.num_mel_bins, 16],
            DType::F32,
            &cpu(),
        )
        .unwrap();
        let out = model.encode(&mel).unwrap();
        assert_eq!(
            out.dim(0).unwrap(),
            batch_size,
            "batch dim should propagate for batch_size={batch_size}"
        );
    }
}

// ============================================================================
// Decoder output shape for multiple config sizes
// ============================================================================

#[test]
fn test_decoder_output_shape_vocab_dim_matches_config() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    let enc_out = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).unwrap();
    let tokens = DynTensor::from_vec(vec![0.0_f32; 3], &[1, 3], &cpu()).unwrap();
    let logits = model.decode(&tokens, &enc_out, true, 0).unwrap();

    assert_eq!(logits.rank(), 3);
    assert_eq!(logits.dim(0).unwrap(), 1);
    assert_eq!(logits.dim(1).unwrap(), 3); // matches token seq_len
    assert_eq!(logits.dim(2).unwrap(), config.vocab_size); // vocab dim
}

#[test]
fn test_decoder_output_various_seq_lengths() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let enc_out = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).unwrap();

    for seq_len in [1, 2, 4, 8] {
        let mut model = WhisperModel::load(&vb, config.clone()).unwrap();
        let tokens =
            DynTensor::from_vec(vec![0.0_f32; seq_len], &[1, seq_len], &cpu()).unwrap();
        let logits = model.decode(&tokens, &enc_out, true, 0).unwrap();
        assert_eq!(
            logits.dim(1).unwrap(),
            seq_len,
            "decoder output seq_len should match input for seq_len={seq_len}"
        );
    }
}

// ============================================================================
// Config validation edge cases
// ============================================================================

#[test]
fn test_validate_d_model_not_multiple_of_encoder_heads() {
    let config = WhisperConfig::whisper_tiny().with_d_model(100).with_encoder_attention_heads(7);
    let err = config.validate();
    assert!(err.is_err(), "100 % 7 != 0 should fail validation");
}

#[test]
fn test_validate_d_model_not_multiple_of_decoder_heads() {
    let config = WhisperConfig::whisper_tiny()
        .with_d_model(100)
        .with_encoder_attention_heads(10)
        .with_decoder_attention_heads(7);
    let err = config.validate();
    assert!(err.is_err(), "100 % 7 != 0 should fail decoder validation");
}

#[test]
fn test_validate_all_presets_pass() {
    let presets = [
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
        WhisperConfig::whisper_large_v2(),
        WhisperConfig::large_v3_turbo(),
    ];
    for (i, config) in presets.iter().enumerate() {
        config
            .validate()
            .unwrap_or_else(|e| panic!("preset {i} should validate: {e}"));
    }
}

#[test]
fn test_validate_minimal_valid_config() {
    // Smallest config that passes validation.
    let config = WhisperConfig::whisper_tiny()
        .with_num_mel_bins(1)
        .with_max_source_positions(1)
        .with_d_model(1)
        .with_encoder_attention_heads(1)
        .with_encoder_layers(1)
        .with_encoder_ffn_dim(1)
        .with_vocab_size(1)
        .with_max_target_positions(1)
        .with_decoder_attention_heads(1)
        .with_decoder_layers(1)
        .with_decoder_ffn_dim(1);
    config.validate().expect("minimal config should validate");
}

// ============================================================================
// Encoder d_model width across configs
// ============================================================================

#[test]
fn test_d_model_scales_with_size() {
    let sizes = [
        ("tiny", WhisperConfig::whisper_tiny(), 384),
        ("base", WhisperConfig::whisper_base(), 512),
        ("small", WhisperConfig::whisper_small(), 768),
        ("medium", WhisperConfig::whisper_medium(), 1024),
        ("large-v2", WhisperConfig::whisper_large_v2(), 1280),
        ("turbo", WhisperConfig::large_v3_turbo(), 1280),
    ];
    for (name, config, expected_d) in &sizes {
        assert_eq!(
            config.d_model, *expected_d,
            "{name}: d_model should be {expected_d}"
        );
    }
}

#[test]
fn test_d_model_monotonic_tiny_to_large() {
    let d_models = [
        WhisperConfig::whisper_tiny().d_model,
        WhisperConfig::whisper_base().d_model,
        WhisperConfig::whisper_small().d_model,
        WhisperConfig::whisper_medium().d_model,
        WhisperConfig::whisper_large_v2().d_model,
    ];
    for i in 1..d_models.len() {
        assert!(
            d_models[i] > d_models[i - 1],
            "d_model should increase: {} vs {}",
            d_models[i - 1],
            d_models[i]
        );
    }
}

// ============================================================================
// Encoder attention heads scale with d_model
// ============================================================================

#[test]
fn test_encoder_heads_scale_values() {
    let sizes = [
        ("tiny", WhisperConfig::whisper_tiny(), 6),
        ("base", WhisperConfig::whisper_base(), 8),
        ("small", WhisperConfig::whisper_small(), 12),
        ("medium", WhisperConfig::whisper_medium(), 16),
        ("large-v2", WhisperConfig::whisper_large_v2(), 20),
        ("turbo", WhisperConfig::large_v3_turbo(), 20),
    ];
    for (name, config, expected) in &sizes {
        assert_eq!(
            config.encoder_attention_heads, *expected,
            "{name}: encoder heads should be {expected}"
        );
        // Verify d_model / heads = 64 for all
        assert_eq!(
            config.d_model / config.encoder_attention_heads,
            64,
            "{name}: head_dim should be 64"
        );
    }
}

// ============================================================================
// Model load with different preset configs (zero weights)
// ============================================================================

#[test]
fn test_model_load_with_tiny_config_succeeds() {
    let config = WhisperConfig::whisper_tiny();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    WhisperModel::load(&vb, config).expect("tiny config should load");
}

#[test]
fn test_model_load_with_base_config_succeeds() {
    let config = WhisperConfig::whisper_base();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    WhisperModel::load(&vb, config).expect("base config should load");
}

#[test]
fn test_model_load_with_small_config_succeeds() {
    let config = WhisperConfig::whisper_small();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    WhisperModel::load(&vb, config).expect("small config should load");
}

#[test]
fn test_model_load_preserves_config() {
    let config = WhisperConfig::whisper_tiny();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let model = WhisperModel::load(&vb, config.clone()).unwrap();
    assert_eq!(model.config().d_model, config.d_model);
    assert_eq!(model.config().encoder_layers, config.encoder_layers);
    assert_eq!(model.config().decoder_layers, config.decoder_layers);
    assert_eq!(model.config().vocab_size, config.vocab_size);
}

#[test]
fn test_model_dtype_matches_vb() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let model = WhisperModel::load(&vb, config).unwrap();
    assert_eq!(model.dtype(), DType::F32);
}

// ============================================================================
// Encoder-decoder end-to-end shape plumbing
// ============================================================================

#[test]
fn test_encode_then_decode_shapes_consistent() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    // Encode
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();
    let enc_out = model.encode(&mel).unwrap();

    // Verify encoder output shape is compatible with decoder input
    assert_eq!(enc_out.rank(), 3);
    assert_eq!(enc_out.dim(2).unwrap(), config.d_model);

    // Decode using encoder output
    let tokens = DynTensor::from_vec(vec![0.0_f32; 2], &[1, 2], &cpu()).unwrap();
    let logits = model.decode(&tokens, &enc_out, true, 0).unwrap();

    assert_eq!(logits.rank(), 3);
    assert_eq!(logits.dim(0).unwrap(), 1);
    assert_eq!(logits.dim(1).unwrap(), 2);
    assert_eq!(logits.dim(2).unwrap(), config.vocab_size);
}

#[test]
fn test_kv_cache_reset_allows_new_segment() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();
    let enc_out = model.encode(&mel).unwrap();
    let tokens = DynTensor::from_vec(vec![0.0_f32; 2], &[1, 2], &cpu()).unwrap();

    // First segment
    let logits1 = model.decode(&tokens, &enc_out, true, 0).unwrap();
    assert_eq!(logits1.dims(), &[1, 2, config.vocab_size]);

    // Reset and start new segment
    model.reset_kv_cache();
    let logits2 = model.decode(&tokens, &enc_out, true, 0).unwrap();
    assert_eq!(logits2.dims(), &[1, 2, config.vocab_size]);
}
