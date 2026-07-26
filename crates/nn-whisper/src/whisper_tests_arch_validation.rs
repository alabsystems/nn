// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Architecture validation tests for Whisper model (#3942).
//!
//! Validates structural invariants across all preset configs: layer counts,
//! parameter count estimates, dimension ratios, shape propagation, and
//! builder pattern correctness.

use crate::config::WhisperConfig;
use crate::test_utils::tiny_config;
use crate::WhisperModel;
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};

// ---------------------------------------------------------------------------
// Layer counts for each preset size
// ---------------------------------------------------------------------------

#[test]
fn test_whisper_tiny_layer_counts() {
    let c = WhisperConfig::whisper_tiny();
    assert_eq!(c.encoder_layers, 4, "whisper-tiny has 4 encoder layers");
    assert_eq!(c.decoder_layers, 4, "whisper-tiny has 4 decoder layers");
    assert_eq!(c.d_model, 384);
    assert_eq!(c.num_mel_bins, 80);
}

#[test]
fn test_whisper_base_layer_counts() {
    let c = WhisperConfig::whisper_base();
    assert_eq!(c.encoder_layers, 6, "whisper-base has 6 encoder layers");
    assert_eq!(c.decoder_layers, 6, "whisper-base has 6 decoder layers");
    assert_eq!(c.d_model, 512);
}

#[test]
fn test_whisper_small_layer_counts() {
    let c = WhisperConfig::whisper_small();
    assert_eq!(c.encoder_layers, 12, "whisper-small has 12 encoder layers");
    assert_eq!(c.decoder_layers, 12, "whisper-small has 12 decoder layers");
    assert_eq!(c.d_model, 768);
}

#[test]
fn test_whisper_medium_layer_counts() {
    let c = WhisperConfig::whisper_medium();
    assert_eq!(c.encoder_layers, 24, "whisper-medium has 24 encoder layers");
    assert_eq!(c.decoder_layers, 24, "whisper-medium has 24 decoder layers");
    assert_eq!(c.d_model, 1024);
}

#[test]
fn test_whisper_large_v2_layer_counts() {
    let c = WhisperConfig::whisper_large_v2();
    assert_eq!(c.encoder_layers, 32, "whisper-large-v2 has 32 encoder layers");
    assert_eq!(c.decoder_layers, 32, "whisper-large-v2 has 32 decoder layers");
    assert_eq!(c.d_model, 1280);
}

#[test]
fn test_whisper_large_v3_turbo_layer_counts() {
    let c = WhisperConfig::large_v3_turbo();
    assert_eq!(c.encoder_layers, 32, "turbo has 32 encoder layers");
    assert_eq!(c.decoder_layers, 4, "turbo has 4 decoder layers (distilled)");
    assert_eq!(c.d_model, 1280);
    assert_eq!(c.vocab_size, 51866, "turbo has 51866 vocab (one more than others)");
}

// ---------------------------------------------------------------------------
// FFN dimension ratio: all presets use 4x d_model
// ---------------------------------------------------------------------------

#[test]
fn test_preset_ffn_dim_is_4x_d_model() {
    let presets: Vec<(&str, WhisperConfig)> = vec![
        ("tiny", WhisperConfig::whisper_tiny()),
        ("base", WhisperConfig::whisper_base()),
        ("small", WhisperConfig::whisper_small()),
        ("medium", WhisperConfig::whisper_medium()),
        ("large-v2", WhisperConfig::whisper_large_v2()),
        ("turbo", WhisperConfig::large_v3_turbo()),
    ];
    for (name, c) in &presets {
        assert_eq!(
            c.encoder_ffn_dim,
            4 * c.d_model,
            "{name}: encoder_ffn_dim should be 4 * d_model"
        );
        assert_eq!(
            c.decoder_ffn_dim,
            4 * c.d_model,
            "{name}: decoder_ffn_dim should be 4 * d_model"
        );
    }
}

// ---------------------------------------------------------------------------
// Head dimension consistency across presets
// ---------------------------------------------------------------------------

#[test]
fn test_preset_encoder_head_dim_consistent() {
    let presets: Vec<(&str, WhisperConfig)> = vec![
        ("tiny", WhisperConfig::whisper_tiny()),
        ("base", WhisperConfig::whisper_base()),
        ("small", WhisperConfig::whisper_small()),
        ("medium", WhisperConfig::whisper_medium()),
        ("large-v2", WhisperConfig::whisper_large_v2()),
        ("turbo", WhisperConfig::large_v3_turbo()),
    ];
    for (name, c) in &presets {
        let head_dim = c.encoder_head_dim();
        assert_eq!(
            head_dim * c.encoder_attention_heads,
            c.d_model,
            "{name}: encoder_head_dim * encoder_heads should equal d_model"
        );
        let dec_head_dim = c.decoder_head_dim();
        assert_eq!(
            dec_head_dim * c.decoder_attention_heads,
            c.d_model,
            "{name}: decoder_head_dim * decoder_heads should equal d_model"
        );
    }
}

#[test]
fn test_all_presets_head_dim_is_64() {
    // Whisper uses 64-dim heads across all sizes.
    let presets = vec![
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
        WhisperConfig::whisper_large_v2(),
        WhisperConfig::large_v3_turbo(),
    ];
    for c in &presets {
        assert_eq!(c.encoder_head_dim(), 64, "all Whisper models use 64-dim heads");
        assert_eq!(c.decoder_head_dim(), 64, "all Whisper models use 64-dim heads");
    }
}

// ---------------------------------------------------------------------------
// Parameter count estimation (encoder attention weights only, as a sanity check)
// ---------------------------------------------------------------------------

/// Estimate the number of parameters in a Whisper encoder attention block.
/// Each attention block has: 4 projections (Q, K, V, O) each [d_model, d_model].
fn encoder_attn_param_count(c: &WhisperConfig) -> usize {
    4 * c.d_model * c.d_model
}

/// Estimate Whisper encoder parameter count (attention + FFN + layer norms).
/// Per-layer: 4*d^2 (attn) + 2*d*ffn (FFN up/down) + 4*d (layernorm bias+weight x 2)
fn estimate_encoder_params(c: &WhisperConfig) -> usize {
    let per_layer = encoder_attn_param_count(c)
        + 2 * c.d_model * c.encoder_ffn_dim // FFN W1 + W2
        + 2 * c.encoder_ffn_dim             // FFN biases
        + 4 * c.d_model;                    // layernorm weight+bias (2 norms * 2 each)
    per_layer * c.encoder_layers
}

#[test]
fn test_param_count_monotonic_across_sizes() {
    let sizes = [("tiny", WhisperConfig::whisper_tiny()),
        ("base", WhisperConfig::whisper_base()),
        ("small", WhisperConfig::whisper_small()),
        ("medium", WhisperConfig::whisper_medium()),
        ("large-v2", WhisperConfig::whisper_large_v2())];
    for i in 1..sizes.len() {
        let (prev_name, prev_cfg) = &sizes[i - 1];
        let (curr_name, curr_cfg) = &sizes[i];
        let prev_params = estimate_encoder_params(prev_cfg);
        let curr_params = estimate_encoder_params(curr_cfg);
        assert!(
            curr_params > prev_params,
            "encoder params should grow: {prev_name} ({prev_params}) >= {curr_name} ({curr_params})"
        );
    }
}

// ---------------------------------------------------------------------------
// Shape propagation: encoder forward
// ---------------------------------------------------------------------------

#[test]
fn test_encoder_output_shape_matches_config() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();
    let out = model.encode(&mel).unwrap();

    assert_eq!(out.rank(), 3, "encoder output rank should be 3");
    assert_eq!(out.dim(0).unwrap(), 1, "batch dim");
    assert_eq!(
        out.dim(2).unwrap(),
        config.d_model,
        "last dim should be d_model"
    );
}

#[test]
fn test_decoder_output_shape_matches_config() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    let enc_out = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).unwrap();
    let tokens = DynTensor::new(&[0.0, 1.0, 2.0, 3.0], &[1, 4], &cpu()).unwrap();
    let logits = model.decode(&tokens, &enc_out, true, 0).unwrap();

    assert_eq!(logits.rank(), 3, "decoder output rank should be 3");
    assert_eq!(logits.dim(0).unwrap(), 1, "batch dim");
    assert_eq!(logits.dim(1).unwrap(), 4, "seq_len matches input");
    assert_eq!(
        logits.dim(2).unwrap(),
        config.vocab_size,
        "last dim should be vocab_size"
    );
}

// ---------------------------------------------------------------------------
// Builder pattern: all with_* methods produce valid configs
// ---------------------------------------------------------------------------

#[test]
fn test_builder_all_with_methods() {
    let config = WhisperConfig::whisper_tiny()
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

    // Should produce the turbo config
    let turbo = WhisperConfig::large_v3_turbo();
    assert_eq!(config.d_model, turbo.d_model);
    assert_eq!(config.encoder_layers, turbo.encoder_layers);
    assert_eq!(config.decoder_layers, turbo.decoder_layers);
    assert_eq!(config.num_mel_bins, turbo.num_mel_bins);
    assert_eq!(config.vocab_size, turbo.vocab_size);
    config.validate().expect("builder-constructed config should be valid");
}

// ---------------------------------------------------------------------------
// Default config validation
// ---------------------------------------------------------------------------

#[test]
fn test_default_config_matches_large_v3_turbo() {
    let default = WhisperConfig::default();
    let turbo = WhisperConfig::large_v3_turbo();
    assert_eq!(default.d_model, turbo.d_model);
    assert_eq!(default.encoder_layers, turbo.encoder_layers);
    assert_eq!(default.decoder_layers, turbo.decoder_layers);
    assert_eq!(default.vocab_size, turbo.vocab_size);
}

// ---------------------------------------------------------------------------
// Config clone independence
// ---------------------------------------------------------------------------

#[test]
fn test_config_clone_independence() {
    let c1 = WhisperConfig::whisper_tiny();
    let mut c2 = c1.clone();
    c2.d_model = 9999;
    assert_eq!(c1.d_model, 384, "original should be unchanged after clone mutation");
    assert_eq!(c2.d_model, 9999);
}

// ---------------------------------------------------------------------------
// Config Debug format includes key fields
// ---------------------------------------------------------------------------

#[test]
fn test_config_debug_contains_key_fields() {
    let c = WhisperConfig::whisper_tiny();
    let debug = format!("{c:?}");
    assert!(debug.contains("d_model"), "Debug should contain d_model");
    assert!(debug.contains("encoder_layers"), "Debug should contain encoder_layers");
    assert!(debug.contains("decoder_layers"), "Debug should contain decoder_layers");
    assert!(debug.contains("vocab_size"), "Debug should contain vocab_size");
}

// ---------------------------------------------------------------------------
// Mel bins: tiny/base/small/medium use 80, large uses 128
// ---------------------------------------------------------------------------

#[test]
fn test_mel_bins_per_size() {
    assert_eq!(WhisperConfig::whisper_tiny().num_mel_bins, 80);
    assert_eq!(WhisperConfig::whisper_base().num_mel_bins, 80);
    assert_eq!(WhisperConfig::whisper_small().num_mel_bins, 80);
    assert_eq!(WhisperConfig::whisper_medium().num_mel_bins, 80);
    assert_eq!(WhisperConfig::whisper_large_v2().num_mel_bins, 128);
    assert_eq!(WhisperConfig::large_v3_turbo().num_mel_bins, 128);
}

// ---------------------------------------------------------------------------
// Encoder + decoder head counts scale with model size
// ---------------------------------------------------------------------------

#[test]
fn test_head_count_scales_with_d_model() {
    let presets = [WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
        WhisperConfig::whisper_large_v2()];
    for i in 1..presets.len() {
        assert!(
            presets[i].encoder_attention_heads >= presets[i - 1].encoder_attention_heads,
            "encoder heads should be non-decreasing with model size"
        );
    }
}
