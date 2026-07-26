// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the Whisper decode loop.
//!
//! Sampling and log-prob tests extracted to `decode_sampling_tests.rs`.
//! Boundary and finiteness validation tests extracted to
//! `decode_boundary_tests.rs` (#1420).

use crate::decode::*;
use crate::test_utils::{tiny_config, tiny_encoder_output, tiny_model};
use crate::tokenizer::{LANGUAGE_TOKEN_END, LANGUAGE_TOKEN_START};
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};

/// Model with vocab_size large enough for Whisper special tokens
/// (SOT, language tokens, NO_SPEECH_TOKEN). Required for detect_language tests.
fn large_vocab_model() -> WhisperModel {
    let mut config = tiny_config();
    config.vocab_size = 51865; // Standard Whisper vocab size
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    WhisperModel::load(&vb, config).unwrap()
}

// -- Compression ratio tests --

#[test]
fn test_compression_ratio_unique_tokens() {
    // All unique bigrams → ratio close to 1.0.
    let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let cr = compression_ratio(&tokens);
    assert!((cr - 1.0).abs() < 0.01, "unique tokens: cr={cr}");
}

#[test]
fn test_compression_ratio_repetitive_tokens() {
    // Highly repetitive → high ratio.
    let tokens = vec![1, 2, 1, 2, 1, 2, 1, 2];
    let cr = compression_ratio(&tokens);
    assert!(cr > 2.0, "repetitive tokens should have high cr: {cr}");
}

#[test]
fn test_compression_ratio_single_token() {
    let cr = compression_ratio(&[42]);
    assert!((cr - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_compression_ratio_empty() {
    let cr = compression_ratio(&[]);
    assert!((cr - 1.0).abs() < f64::EPSILON);
}

// -- Quality check tests --

#[test]
fn test_quality_check_passes() {
    let result = DecodingResult {
        tokens: vec![1, 2, 3],
        avg_logprob: -0.5,
        compression_ratio: 1.5,
        reached_eot: true,
        temperature: 0.0,
        no_speech_prob: 0.0,
    };
    let config = DecodeConfig::default();
    assert!(passes_quality_check(&result, &config));
}

#[test]
fn test_quality_check_fails_compression() {
    let result = DecodingResult {
        tokens: vec![1, 2, 3],
        avg_logprob: -0.5,
        compression_ratio: 3.0, // Exceeds 2.4 threshold
        reached_eot: true,
        temperature: 0.0,
        no_speech_prob: 0.0,
    };
    let config = DecodeConfig::default();
    assert!(!passes_quality_check(&result, &config));
}

#[test]
fn test_quality_check_fails_logprob() {
    let result = DecodingResult {
        tokens: vec![1, 2, 3],
        avg_logprob: -2.0, // Below -1.0 threshold
        compression_ratio: 1.0,
        reached_eot: true,
        temperature: 0.0,
        no_speech_prob: 0.0,
    };
    let config = DecodeConfig::default();
    assert!(!passes_quality_check(&result, &config));
}

// -- Token suppression tests --

#[test]
fn test_apply_suppression_empty() {
    let mut logits = vec![1.0, 2.0, 3.0, 4.0];
    apply_suppression_inplace(&mut logits, &[]);
    assert_eq!(logits, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_apply_suppression_single_token() {
    let mut logits = vec![1.0, 2.0, 3.0, 4.0];
    apply_suppression_inplace(&mut logits, &[2]);
    assert_eq!(logits[0], 1.0);
    assert_eq!(logits[1], 2.0);
    assert_eq!(logits[2], f32::NEG_INFINITY);
    assert_eq!(logits[3], 4.0);
}

#[test]
fn test_apply_suppression_multiple_tokens() {
    let mut logits = vec![1.0, 2.0, 3.0, 4.0];
    apply_suppression_inplace(&mut logits, &[0, 3]);
    assert_eq!(logits[0], f32::NEG_INFINITY);
    assert_eq!(logits[1], 2.0);
    assert_eq!(logits[2], 3.0);
    assert_eq!(logits[3], f32::NEG_INFINITY);
}

#[test]
fn test_apply_suppression_out_of_range() {
    // Suppressing a token beyond vocab_size should be a no-op.
    let mut logits = vec![1.0, 2.0, 3.0];
    apply_suppression_inplace(&mut logits, &[100]);
    assert_eq!(logits, vec![1.0, 2.0, 3.0]);
}

// -- Sampling and log-prob tests (extracted to decode_sampling_tests.rs) ------

#[path = "decode_sampling_tests.rs"]
mod sampling;

// -- Greedy decode integration test --

#[test]
fn test_greedy_decode_with_zeros_model() {
    let mut model = tiny_model();
    let encoder_output = tiny_encoder_output();

    // With zeros model, all logits are identical → argmax picks token 0.
    // Token 0 is never EOT (EOT=50257 > vocab_size=32), so decode runs to max_length.
    let config = DecodeConfig {
        max_length: 5,
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        ..Default::default()
    };

    let result = greedy_decode(&mut model, &encoder_output, &config).unwrap();
    assert!(
        result.tokens.len() <= 5,
        "should respect max_length: got {}",
        result.tokens.len()
    );
    assert!(!result.reached_eot, "zeros model should not produce EOT");
    assert!((result.temperature - 0.0).abs() < f64::EPSILON);
}

// -- Temperature fallback test --

#[test]
fn test_temperature_fallback_returns_result() {
    let mut model = tiny_model();
    let encoder_output = tiny_encoder_output();

    let config = DecodeConfig {
        max_length: 3,
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        ..Default::default()
    };

    let result =
        temperature_fallback_decode(&mut model, &encoder_output, &config, &DEFAULT_TEMPERATURES)
            .unwrap();

    // Should produce some result (may not pass quality checks with zeros model).
    assert!(result.temperature >= 0.0);
    assert!(result.tokens.len() <= 3);
}

// -- EOT termination test --

#[test]
fn test_eot_token_constant() {
    assert_eq!(EOT_TOKEN, 50257);
}

// -- Default config test --

#[test]
fn test_default_config_values() {
    let config = DecodeConfig::default();
    assert_eq!(config.max_length, 224);
    assert!((config.compression_ratio_threshold - 2.4).abs() < f64::EPSILON);
    assert!((config.avg_logprob_threshold - (-1.0)).abs() < f64::EPSILON);
    assert!(config.suppress_tokens.is_empty());
    assert_eq!(config.initial_tokens, vec![50258, 50259, 50360, 50364]);
    assert!(config.seed.is_none());
}

// -- Seeded decode tests --

#[test]
fn test_decode_with_seed_produces_sampled_tokens() {
    // With a seed and temperature > 0, decode should sample from the distribution.
    // A zeros model produces uniform logits, so sampling should produce diverse tokens.
    let mut model = tiny_model();
    let encoder_output = tiny_encoder_output();

    let config = DecodeConfig {
        max_length: 10,
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        seed: Some(42),
        ..Default::default()
    };

    let result = decode_with_temperature(&mut model, &encoder_output, &config, 1.0).unwrap();
    // With uniform logits and sampling, tokens should not all be identical.
    let unique: std::collections::HashSet<usize> = result.tokens.iter().copied().collect();
    assert!(
        unique.len() >= 2,
        "seeded sampling with uniform logits should produce diverse tokens, got {unique:?}"
    );
}

#[test]
fn test_decode_same_seed_reproducible() {
    let encoder_output = tiny_encoder_output();
    let config = DecodeConfig {
        max_length: 5,
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        seed: Some(77),
        ..Default::default()
    };

    let mut model1 = tiny_model();
    let r1 = decode_with_temperature(&mut model1, &encoder_output, &config, 0.8).unwrap();
    let mut model2 = tiny_model();
    let r2 = decode_with_temperature(&mut model2, &encoder_output, &config, 0.8).unwrap();
    assert_eq!(r1.tokens, r2.tokens, "same seed should produce same tokens");
}

// -- Decode with temperature test --

#[test]
fn test_decode_with_temperature_zero() {
    let mut model = tiny_model();
    let encoder_output = tiny_encoder_output();

    let config = DecodeConfig {
        max_length: 3,
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        ..Default::default()
    };

    let result = decode_with_temperature(&mut model, &encoder_output, &config, 0.0).unwrap();
    assert!((result.temperature - 0.0).abs() < f64::EPSILON);
}

// -- Token suppression integration test --

#[test]
fn test_greedy_decode_with_suppression() {
    let mut model = tiny_model();
    let encoder_output = tiny_encoder_output();

    // With zeros model, all logits are equal. Suppress token 0 →
    // argmax should pick the next unsuppressed token.
    let config = DecodeConfig {
        max_length: 3,
        initial_tokens: vec![0],
        suppress_tokens: vec![0],
        ..Default::default()
    };

    let result = greedy_decode(&mut model, &encoder_output, &config).unwrap();
    // With token 0 suppressed, decoded tokens should not contain 0.
    for &t in &result.tokens {
        assert_ne!(t, 0, "suppressed token should not appear in output");
    }
}

// -- Language detection tests (AC2) --

#[test]
fn test_detect_language_returns_valid_token_range() {
    let mut model = large_vocab_model();
    let encoder_output = tiny_encoder_output();
    let result = detect_language(&mut model, &encoder_output).unwrap();
    // Language token must be in the valid range.
    assert!(
        result.language_token >= LANGUAGE_TOKEN_START
            && result.language_token <= LANGUAGE_TOKEN_END,
        "language_token {} not in [{}, {}]",
        result.language_token,
        LANGUAGE_TOKEN_START,
        LANGUAGE_TOKEN_END,
    );
    assert!(result.probability >= 0.0 && result.probability <= 1.0);
}

#[test]
fn test_detect_language_no_speech_prob_range() {
    let mut model = large_vocab_model();
    let encoder_output = tiny_encoder_output();
    let result = detect_language(&mut model, &encoder_output).unwrap();
    assert!(
        result.no_speech_prob >= 0.0 && result.no_speech_prob <= 1.0,
        "no_speech_prob should be in [0, 1], got {}",
        result.no_speech_prob,
    );
}

// -- No-speech probability tests (AC3) --

#[test]
fn test_greedy_decode_populates_no_speech_prob() {
    let mut model = tiny_model();
    let encoder_output = tiny_encoder_output();
    let config = DecodeConfig {
        max_length: 3,
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        ..Default::default()
    };
    let result = greedy_decode(&mut model, &encoder_output, &config).unwrap();
    // no_speech_prob should be a valid probability.
    assert!(
        result.no_speech_prob >= 0.0 && result.no_speech_prob <= 1.0,
        "no_speech_prob should be in [0, 1], got {}",
        result.no_speech_prob,
    );
}

#[test]
fn test_transcription_result_includes_no_speech_prob() {
    let mut model = tiny_model();
    let encoder_output = tiny_encoder_output();
    let config = DecodeConfig {
        max_length: 3,
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        ..Default::default()
    };
    let result = greedy_decode(&mut model, &encoder_output, &config).unwrap();
    // Verify no_speech_prob is populated in the result struct.
    assert!(result.no_speech_prob.is_finite());
}

// Beam search tests extracted to decode_beam_tests.rs (#1618).
#[path = "decode_beam_tests.rs"]
mod beam_tests;

// Boundary and finiteness validation tests extracted to
// decode_boundary_tests.rs (#1420).
#[path = "decode_boundary_tests.rs"]
mod boundary_tests;
