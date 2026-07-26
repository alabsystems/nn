// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! WhisperConfig validation tests (#1366).
//!
//! Extracted from `whisper_tests.rs` to keep the parent under 500 lines.

use crate::test_utils::tiny_config;
use crate::WhisperModel;
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};

#[test]
fn test_validate_valid_config() {
    let config = tiny_config();
    assert!(config.validate().is_ok());
}

#[test]
fn test_validate_large_v3_turbo() {
    let config = crate::config::WhisperConfig::large_v3_turbo();
    assert!(config.validate().is_ok());
}

#[test]
fn test_validate_zero_d_model() {
    let mut config = tiny_config();
    config.d_model = 0;
    let err = config.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("d_model"),
        "error should mention d_model: {msg}"
    );
}

#[test]
fn test_validate_zero_encoder_heads() {
    let mut config = tiny_config();
    config.encoder_attention_heads = 0;
    let err = config.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("encoder_attention_heads"),
        "error should mention encoder_attention_heads: {msg}"
    );
}

#[test]
fn test_validate_zero_decoder_heads() {
    let mut config = tiny_config();
    config.decoder_attention_heads = 0;
    let err = config.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("decoder_attention_heads"),
        "error should mention decoder_attention_heads: {msg}"
    );
}

#[test]
fn test_validate_encoder_heads_not_divisible() {
    let mut config = tiny_config();
    config.encoder_attention_heads = 3; // 16 % 3 != 0
    let err = config.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("divisible"),
        "error should mention divisibility: {msg}"
    );
}

#[test]
fn test_validate_decoder_heads_not_divisible() {
    let mut config = tiny_config();
    config.decoder_attention_heads = 3; // 16 % 3 != 0
    let err = config.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("divisible"),
        "error should mention divisibility: {msg}"
    );
}

#[test]
fn test_validate_zero_vocab_size() {
    let mut config = tiny_config();
    config.vocab_size = 0;
    let err = config.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("vocab_size"),
        "error should mention vocab_size: {msg}"
    );
}

#[test]
fn test_validate_zero_num_mel_bins() {
    let mut config = tiny_config();
    config.num_mel_bins = 0;
    let err = config.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("num_mel_bins"),
        "error should mention num_mel_bins: {msg}"
    );
}

#[test]
fn test_validate_zero_encoder_ffn_dim() {
    let mut config = tiny_config();
    config.encoder_ffn_dim = 0;
    let err = config.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("encoder_ffn_dim"),
        "error should mention encoder_ffn_dim: {msg}"
    );
}

#[test]
fn test_validate_zero_decoder_ffn_dim() {
    let mut config = tiny_config();
    config.decoder_ffn_dim = 0;
    let err = config.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("decoder_ffn_dim"),
        "error should mention decoder_ffn_dim: {msg}"
    );
}

#[test]
fn test_validate_zero_max_source_positions() {
    let mut config = tiny_config();
    config.max_source_positions = 0;
    let err = config.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("max_source_positions"),
        "error should mention max_source_positions: {msg}"
    );
}

#[test]
fn test_validate_zero_max_target_positions() {
    let mut config = tiny_config();
    config.max_target_positions = 0;
    let err = config.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("max_target_positions"),
        "error should mention max_target_positions: {msg}"
    );
}

#[test]
fn test_load_rejects_invalid_config() {
    let mut config = tiny_config();
    config.encoder_attention_heads = 0;
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let result = WhisperModel::load(&vb, config);
    assert!(result.is_err(), "load should reject invalid config");
}
