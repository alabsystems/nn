// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Whisper decode loop.
//!
//! Covers temperature_fallback_decode edge cases, decode + quality check
//! interaction, and structural assertions not covered by inline tests.

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};
use nn_whisper::test_utils::{tiny_config, tiny_encoder_output, tiny_model};
use nn_whisper::{
    compression_ratio, decode_with_temperature, greedy_decode, passes_quality_check,
    temperature_fallback_decode, DecodeConfig, DecodingResult, WhisperModel,
    DEFAULT_AVG_LOGPROB_THRESHOLD, DEFAULT_COMPRESSION_RATIO_THRESHOLD, DEFAULT_TEMPERATURES,
    EOT_TOKEN, MAX_DECODE_LENGTH,
};

fn short_config() -> DecodeConfig {
    DecodeConfig::default()
        .with_max_length(3)
        .with_initial_tokens(vec![0])
        .with_suppress_tokens(Vec::new())
}

// ---------------------------------------------------------------------------
// temperature_fallback_decode tests
// ---------------------------------------------------------------------------

#[test]
fn test_fallback_empty_temperatures_returns_error() {
    // Empty temperatures should be rejected with an error.
    let mut model = tiny_model();
    let enc = tiny_encoder_output();
    let config = short_config();

    let result = temperature_fallback_decode(&mut model, &enc, &config, &[]);
    assert!(result.is_err(), "empty temperatures should return error");
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("empty"), "error should mention empty: {msg}");
}

#[test]
fn test_fallback_single_temperature() {
    let mut model = tiny_model();
    let enc = tiny_encoder_output();
    let config = short_config();

    let result = temperature_fallback_decode(&mut model, &enc, &config, &[0.0]).unwrap();
    // Should produce a result with temperature 0.0.
    assert!((result.temperature - 0.0).abs() < f64::EPSILON);
    assert!(result.tokens.len() <= 3);
}

#[test]
fn test_fallback_tries_all_temperatures_on_failure() {
    // With a zeros model and tight quality thresholds that always fail,
    // the fallback should try all temperatures and return the last result.
    let mut model = tiny_model();
    let enc = tiny_encoder_output();

    // Set impossible quality thresholds so no temperature passes.
    let config = DecodeConfig::default()
        .with_max_length(3)
        .with_initial_tokens(vec![0])
        .with_suppress_tokens(Vec::new())
        .with_seed(Some(42))
        .with_compression_ratio_threshold(0.0) // Impossible: CR >= 1.0 always.
        .with_avg_logprob_threshold(100.0); // Impossible: avg_logprob always negative.

    let temps = [0.0, 0.5, 1.0];
    let result = temperature_fallback_decode(&mut model, &enc, &config, &temps).unwrap();
    // Last temperature attempted should be 1.0.
    assert!(
        (result.temperature - 1.0).abs() < f64::EPSILON,
        "should return last temperature's result, got {}",
        result.temperature
    );
}

#[test]
fn test_fallback_returns_first_passing_temperature() {
    // With very lenient quality thresholds, the first temperature should pass.
    let mut model = tiny_model();
    let enc = tiny_encoder_output();

    // Use extremely lenient thresholds so even zeros-model output passes.
    let config = DecodeConfig::default()
        .with_max_length(3)
        .with_initial_tokens(vec![0])
        .with_suppress_tokens(Vec::new())
        .with_compression_ratio_threshold(100.0) // Always passes.
        .with_avg_logprob_threshold(-100.0); // Always passes.

    let result =
        temperature_fallback_decode(&mut model, &enc, &config, &DEFAULT_TEMPERATURES).unwrap();
    // First temperature (0.0) should pass with lenient thresholds.
    assert!(
        (result.temperature - 0.0).abs() < f64::EPSILON,
        "first passing temperature should be 0.0, got {}",
        result.temperature
    );
}

#[test]
fn test_fallback_result_is_finite() {
    let mut model = tiny_model();
    let enc = tiny_encoder_output();
    let config = short_config();

    let result = temperature_fallback_decode(&mut model, &enc, &config, &[0.0, 0.5, 1.0]).unwrap();
    assert!(result.compression_ratio.is_finite());
    // avg_logprob may be 0.0 (if no decoded tokens) but should not be NaN.
    assert!(!result.avg_logprob.is_nan());
}

// ---------------------------------------------------------------------------
// decode_with_temperature structural tests
// ---------------------------------------------------------------------------

#[test]
fn test_decode_temperature_recorded_in_result() {
    let enc = tiny_encoder_output();
    let config = short_config();

    for &temp in &[0.0, 0.5, 1.0] {
        let mut model = tiny_model();
        let result = decode_with_temperature(&mut model, &enc, &config, temp).unwrap();
        assert!(
            (result.temperature - temp).abs() < f64::EPSILON,
            "temperature {temp} not recorded, got {}",
            result.temperature
        );
    }
}

#[test]
fn test_decode_max_length_respected() {
    for max_len in [1, 2, 5, 10] {
        let mut model = tiny_model();
        let enc = tiny_encoder_output();
        let config = DecodeConfig::default()
            .with_max_length(max_len)
            .with_initial_tokens(vec![0])
            .with_suppress_tokens(Vec::new());

        let result = greedy_decode(&mut model, &enc, &config).unwrap();
        assert!(
            result.tokens.len() <= max_len,
            "max_length={max_len} but got {} tokens",
            result.tokens.len()
        );
    }
}

#[test]
fn test_decode_greedy_deterministic() {
    // Greedy decode (temperature=0) should be deterministic.
    let enc = tiny_encoder_output();
    let config = short_config();

    let mut model1 = tiny_model();
    let r1 = greedy_decode(&mut model1, &enc, &config).unwrap();
    let mut model2 = tiny_model();
    let r2 = greedy_decode(&mut model2, &enc, &config).unwrap();

    assert_eq!(
        r1.tokens, r2.tokens,
        "greedy decode should be deterministic"
    );
    assert!(
        (r1.avg_logprob - r2.avg_logprob).abs() < 1e-6,
        "avg_logprob should match: {} vs {}",
        r1.avg_logprob,
        r2.avg_logprob
    );
}

#[test]
fn test_decode_with_all_tokens_suppressed() {
    // Suppress all tokens except one. Decode should always pick the unsuppressed token.
    let mut model = tiny_model();
    let enc = tiny_encoder_output();
    let vocab_size = tiny_config().vocab_size;

    // Suppress everything except token 5.
    let suppress: Vec<usize> = (0..vocab_size).filter(|&t| t != 5).collect();
    let config = DecodeConfig::default()
        .with_max_length(5)
        .with_initial_tokens(vec![0])
        .with_suppress_tokens(suppress);

    let result = greedy_decode(&mut model, &enc, &config).unwrap();
    for &t in &result.tokens {
        assert_eq!(t, 5, "only unsuppressed token (5) should appear, got {t}");
    }
}

// ---------------------------------------------------------------------------
// greedy_decode consecutive calls (KV cache independence)
// ---------------------------------------------------------------------------

#[test]
fn test_greedy_decode_consecutive_calls_independent() {
    // Consecutive greedy_decode calls on the same model should produce
    // identical results because each call resets the KV cache.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();
    let encoder_output = model.encode(&mel).unwrap();

    let decode_config = short_config();

    let r1 = greedy_decode(&mut model, &encoder_output, &decode_config).unwrap();
    let r2 = greedy_decode(&mut model, &encoder_output, &decode_config).unwrap();

    assert_eq!(
        r1.tokens, r2.tokens,
        "consecutive greedy_decode calls should produce identical tokens (KV cache reset)"
    );
    assert!(
        (r1.avg_logprob - r2.avg_logprob).abs() < 1e-6,
        "avg_logprob should match between consecutive calls: {} vs {}",
        r1.avg_logprob,
        r2.avg_logprob
    );
}

// ---------------------------------------------------------------------------
// compression_ratio edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_compression_ratio_two_tokens() {
    // Two tokens = 1 bigram, ratio = 1.0.
    let cr = compression_ratio(&[10, 20]);
    assert!((cr - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_compression_ratio_all_same_token() {
    // [A, A, A, A] → 3 bigram slots, 1 unique bigram (A,A) → ratio = 3.0.
    let cr = compression_ratio(&[7, 7, 7, 7]);
    assert!((cr - 3.0).abs() < f64::EPSILON, "expected 3.0, got {cr}");
}

#[test]
fn test_compression_ratio_alternating_pair() {
    // [A, B, A, B, A, B] → 5 bigram slots, 2 unique ((A,B), (B,A)) → ratio = 2.5.
    let cr = compression_ratio(&[1, 2, 1, 2, 1, 2]);
    assert!((cr - 2.5).abs() < f64::EPSILON, "expected 2.5, got {cr}");
}

// ---------------------------------------------------------------------------
// passes_quality_check boundary tests
// ---------------------------------------------------------------------------

#[test]
fn test_quality_check_exact_thresholds() {
    let config = DecodeConfig::default();

    // Exactly at both thresholds — should pass (<=, >=).
    let result = DecodingResult::new(
        vec![1],
        DEFAULT_AVG_LOGPROB_THRESHOLD,
        DEFAULT_COMPRESSION_RATIO_THRESHOLD,
        true,
        0.0,
        0.0,
    );
    assert!(
        passes_quality_check(&result, &config),
        "exact threshold values should pass"
    );
}

#[test]
fn test_quality_check_just_below_threshold() {
    let config = DecodeConfig::default();

    // Compression ratio just above threshold — should fail.
    let result = DecodingResult::new(
        vec![1],
        -0.5,
        DEFAULT_COMPRESSION_RATIO_THRESHOLD + 0.001,
        true,
        0.0,
        0.0,
    );
    assert!(!passes_quality_check(&result, &config));

    // Avg logprob just below threshold — should fail.
    let result = DecodingResult::new(
        vec![1],
        DEFAULT_AVG_LOGPROB_THRESHOLD - 0.001,
        1.0,
        true,
        0.0,
        0.0,
    );
    assert!(!passes_quality_check(&result, &config));
}

// ---------------------------------------------------------------------------
// Constants verification
// ---------------------------------------------------------------------------

#[test]
fn test_default_temperatures_ordered() {
    // Temperatures should be non-decreasing (greedy first, then increasing).
    for w in DEFAULT_TEMPERATURES.windows(2) {
        assert!(
            w[0] <= w[1],
            "temperatures should be non-decreasing: {} > {}",
            w[0],
            w[1]
        );
    }
    assert!(
        (DEFAULT_TEMPERATURES[0] - 0.0).abs() < f64::EPSILON,
        "first temperature should be 0.0 (greedy)"
    );
}

#[test]
fn test_eot_token_not_in_tiny_vocab() {
    // EOT_TOKEN (50257) is beyond tiny config vocab_size (32).
    // This means zeros-model tests never hit EOT — which is intentional.
    let config = tiny_config();
    assert!(
        EOT_TOKEN >= config.vocab_size,
        "EOT_TOKEN should be beyond tiny vocab for testing"
    );
}

#[test]
fn test_max_decode_length_constant() {
    assert_eq!(MAX_DECODE_LENGTH, 224);
}

// ---------------------------------------------------------------------------
// Seeded temperature fallback
// ---------------------------------------------------------------------------

#[test]
fn test_fallback_seeded_reproducible() {
    let enc = tiny_encoder_output();
    let config = DecodeConfig::default()
        .with_max_length(5)
        .with_initial_tokens(vec![0])
        .with_suppress_tokens(Vec::new())
        .with_seed(Some(99));

    let mut model1 = tiny_model();
    let r1 = temperature_fallback_decode(&mut model1, &enc, &config, &[0.0, 0.5, 1.0]).unwrap();
    let mut model2 = tiny_model();
    let r2 = temperature_fallback_decode(&mut model2, &enc, &config, &[0.0, 0.5, 1.0]).unwrap();

    assert_eq!(
        r1.tokens, r2.tokens,
        "seeded fallback should be reproducible"
    );
    assert_eq!(r1.temperature, r2.temperature);
}

// ---------------------------------------------------------------------------
// Encode-then-decode integration
// ---------------------------------------------------------------------------

#[test]
fn test_encode_then_decode_roundtrip() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = WhisperModel::load(&vb, config.clone()).unwrap();

    // Encode mel spectrogram.
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();
    let encoder_output = model.encode(&mel).unwrap();

    // Decode with the encode output.
    let decode_config = DecodeConfig::default()
        .with_max_length(3)
        .with_initial_tokens(vec![0])
        .with_suppress_tokens(Vec::new());

    let result = greedy_decode(&mut model, &encoder_output, &decode_config).unwrap();
    assert!(result.tokens.len() <= 3);
    assert!(!result.reached_eot);
    assert!(result.compression_ratio.is_finite());
}
