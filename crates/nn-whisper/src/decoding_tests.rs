// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Decoding tests: greedy decoding logic, beam search integration,
//! timestamp token handling, language detection tokens, end-of-text
//! detection, DecodeConfig validation, and sampling edge cases.
//! Part of #4186.

use crate::decode::{
    apply_suppression_inplace, compression_ratio, passes_quality_check, DecodeConfig,
    DecodingResult, DEFAULT_AVG_LOGPROB_THRESHOLD, DEFAULT_COMPRESSION_RATIO_THRESHOLD,
    DEFAULT_TEMPERATURES, MAX_DECODE_LENGTH,
};
use crate::tokenizer::{
    WhisperTokenizer, EOT_TOKEN, LANGUAGE_TOKEN_END, LANGUAGE_TOKEN_START, NO_SPEECH_TOKEN,
    SOT_TOKEN,
};

#[cfg(test)]
use crate::decode::{argmax_f32, compute_log_prob};

// ============================================================================
// Greedy decoding logic
// ============================================================================

#[test]
fn test_argmax_f32_basic() {
    assert_eq!(argmax_f32(&[1.0, 3.0, 2.0]), 1);
    assert_eq!(argmax_f32(&[5.0, 1.0, 2.0]), 0);
    assert_eq!(argmax_f32(&[0.0, 0.0, 1.0]), 2);
}

#[test]
fn test_argmax_f32_single_element() {
    assert_eq!(argmax_f32(&[42.0]), 0);
}

#[test]
fn test_argmax_f32_all_equal() {
    // When all values are equal, argmax returns a valid index (the specific
    // index depends on the iterator's max_by tie-breaking behavior with
    // total_cmp — it returns the last max, not the first).
    let logits = vec![1.0; 10];
    let idx = argmax_f32(&logits);
    assert!(
        idx < logits.len(),
        "equal logits should return a valid index, got {idx}"
    );
}

#[test]
fn test_argmax_f32_negative_values() {
    assert_eq!(argmax_f32(&[-5.0, -1.0, -3.0]), 1);
}

#[test]
fn test_argmax_f32_with_neg_infinity() {
    // Suppressed logits have NEG_INFINITY. argmax should pick the non-suppressed one.
    let logits = vec![f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY];
    assert_eq!(argmax_f32(&logits), 1);
}

#[test]
fn test_argmax_f32_empty_returns_zero() {
    assert_eq!(argmax_f32(&[]), 0);
}

#[test]
fn test_compute_log_prob_basic() {
    // For uniform logits [0, 0, 0], softmax = [1/3, 1/3, 1/3],
    // log-prob at any index = ln(1/3) ≈ -1.0986.
    let logits = vec![0.0, 0.0, 0.0];
    let lp = compute_log_prob(&logits, 0);
    let expected = (1.0f32 / 3.0).ln();
    assert!(
        (lp - expected).abs() < 1e-5,
        "log_prob = {lp}, expected {expected}"
    );
}

#[test]
fn test_compute_log_prob_dominant_logit() {
    // When one logit dominates, its log-prob should be close to 0.
    let logits = vec![-100.0, 100.0, -100.0];
    let lp = compute_log_prob(&logits, 1);
    assert!(
        lp > -0.001,
        "dominant logit should have log-prob near 0, got {lp}"
    );
}

#[test]
fn test_compute_log_prob_out_of_bounds() {
    let logits = vec![1.0, 2.0];
    let lp = compute_log_prob(&logits, 5);
    assert_eq!(lp, f32::NEG_INFINITY);
}

#[test]
fn test_compute_log_prob_empty() {
    let lp = compute_log_prob(&[], 0);
    assert_eq!(lp, f32::NEG_INFINITY);
}

// ============================================================================
// Greedy decode with suppression
// ============================================================================

#[test]
fn test_suppression_all_tokens_results_in_fallback() {
    // When all tokens are suppressed to NEG_INFINITY, argmax picks index 0
    // (or the first NEG_INFINITY entry, which is deterministic via total_cmp).
    let mut logits = vec![1.0, 2.0, 3.0, 4.0];
    apply_suppression_inplace(&mut logits, &[0, 1, 2, 3]);
    assert!(logits.iter().all(|&v| v == f32::NEG_INFINITY));
}

#[test]
fn test_suppression_preserves_unsuppressed() {
    let mut logits = vec![1.0, 5.0, 3.0, 2.0];
    apply_suppression_inplace(&mut logits, &[0, 2]);
    assert_eq!(logits[0], f32::NEG_INFINITY);
    assert_eq!(logits[1], 5.0);
    assert_eq!(logits[2], f32::NEG_INFINITY);
    assert_eq!(logits[3], 2.0);
}

// ============================================================================
// Timestamp token handling
// ============================================================================

#[test]
fn test_timestamp_value_at_start() {
    // Token 50365 = timestamp 0.00s.
    let tokenizer = make_test_tokenizer();
    let ts = tokenizer.timestamp_value(50365);
    assert_eq!(ts, Some(0.0));
}

#[test]
fn test_timestamp_value_resolution() {
    // Each token increments by 0.02s.
    let tokenizer = make_test_tokenizer();
    let ts_1 = tokenizer.timestamp_value(50366);
    assert_eq!(ts_1, Some(0.02));

    let ts_50 = tokenizer.timestamp_value(50415);
    assert!((ts_50.unwrap() - 1.0).abs() < 1e-10, "50 * 0.02 = 1.0s");
}

#[test]
fn test_timestamp_value_30_seconds() {
    // 30 seconds = 1500 * 0.02s → token 50365 + 1500 = 51865.
    let tokenizer = make_test_tokenizer();
    let ts_30s = tokenizer.timestamp_value(50365 + 1500);
    assert!(
        (ts_30s.unwrap() - 30.0).abs() < 1e-10,
        "expected 30.0s, got {ts_30s:?}"
    );
}

#[test]
fn test_timestamp_value_non_timestamp_returns_none() {
    // Regular text tokens should return None.
    let tokenizer = make_test_tokenizer();
    assert_eq!(tokenizer.timestamp_value(100), None);
    assert_eq!(tokenizer.timestamp_value(0), None);
}

#[test]
fn test_is_timestamp() {
    let tokenizer = make_test_tokenizer();
    assert!(!tokenizer.is_timestamp(50000));
    assert!(!tokenizer.is_timestamp(50364)); // NO_TIMESTAMPS, not a timestamp
    assert!(tokenizer.is_timestamp(50365));
    assert!(tokenizer.is_timestamp(51000));
}

// ============================================================================
// Language detection tokens
// ============================================================================

#[test]
fn test_language_token_range() {
    // Language tokens span 50259 (English) to 50358 (100 languages).
    assert_eq!(LANGUAGE_TOKEN_START, 50259);
    assert_eq!(LANGUAGE_TOKEN_END, 50358);
    assert_eq!(
        LANGUAGE_TOKEN_END - LANGUAGE_TOKEN_START + 1,
        100,
        "should have 100 language tokens"
    );
}

#[test]
fn test_is_special_for_language_tokens() {
    let tokenizer = make_test_tokenizer();
    // All language tokens are special (>= EOT_TOKEN=50257).
    for id in LANGUAGE_TOKEN_START..=LANGUAGE_TOKEN_END {
        assert!(
            tokenizer.is_special(id),
            "language token {id} should be special"
        );
    }
}

#[test]
fn test_sot_eot_nospeech_constants() {
    assert_eq!(SOT_TOKEN, 50258);
    assert_eq!(EOT_TOKEN, 50257);
    assert_eq!(NO_SPEECH_TOKEN, 50363);
}

#[test]
fn test_regular_tokens_not_special() {
    let tokenizer = make_test_tokenizer();
    // Tokens below 50257 are regular text tokens.
    for id in [0, 100, 1000, 10000, 50256] {
        assert!(
            !tokenizer.is_special(id),
            "token {id} should NOT be special"
        );
    }
}

// ============================================================================
// End-of-text detection
// ============================================================================

#[test]
fn test_eot_token_value() {
    assert_eq!(EOT_TOKEN, 50257, "EOT token must be 50257");
}

#[test]
fn test_eot_is_special() {
    let tokenizer = make_test_tokenizer();
    assert!(tokenizer.is_special(EOT_TOKEN));
}

#[test]
fn test_eot_not_timestamp() {
    let tokenizer = make_test_tokenizer();
    assert!(
        !tokenizer.is_timestamp(EOT_TOKEN),
        "EOT should not be a timestamp"
    );
}

// ============================================================================
// DecodeConfig validation
// ============================================================================

#[test]
fn test_decode_config_default_valid() {
    let config = DecodeConfig::default();
    config.validate().expect("default config should be valid");
}

#[test]
fn test_decode_config_max_length_zero_invalid() {
    let config = DecodeConfig::default().with_max_length(0);
    assert!(config.validate().is_err(), "max_length=0 should be invalid");
}

#[test]
fn test_decode_config_max_length_exceeds_limit() {
    let config = DecodeConfig::default().with_max_length(MAX_DECODE_LENGTH + 1);
    assert!(
        config.validate().is_err(),
        "max_length > MAX_DECODE_LENGTH should be invalid"
    );
}

#[test]
fn test_decode_config_nan_compression_threshold() {
    let config = DecodeConfig::default().with_compression_ratio_threshold(f64::NAN);
    assert!(
        config.validate().is_err(),
        "NaN compression threshold should be invalid"
    );
}

#[test]
fn test_decode_config_inf_logprob_threshold() {
    let config = DecodeConfig::default().with_avg_logprob_threshold(f64::INFINITY);
    assert!(
        config.validate().is_err(),
        "Inf logprob threshold should be invalid"
    );
}

#[test]
fn test_decode_config_empty_initial_tokens() {
    let config = DecodeConfig::default().with_initial_tokens(Vec::new());
    assert!(
        config.validate().is_err(),
        "empty initial_tokens should be invalid"
    );
}

// ============================================================================
// Quality check thresholds
// ============================================================================

#[test]
fn test_quality_check_at_exact_thresholds() {
    let config = DecodeConfig::default();
    // Exactly at thresholds should pass.
    let result = DecodingResult::new(
        vec![1, 2, 3],
        DEFAULT_AVG_LOGPROB_THRESHOLD,
        DEFAULT_COMPRESSION_RATIO_THRESHOLD,
        true,
        0.0,
        0.0,
    );
    assert!(passes_quality_check(&result, &config));
}

#[test]
fn test_quality_check_just_below_logprob_threshold() {
    let config = DecodeConfig::default();
    let result = DecodingResult::new(
        vec![1, 2, 3],
        DEFAULT_AVG_LOGPROB_THRESHOLD - 0.001,
        1.0,
        true,
        0.0,
        0.0,
    );
    assert!(!passes_quality_check(&result, &config));
}

#[test]
fn test_quality_check_just_above_compression_threshold() {
    let config = DecodeConfig::default();
    let result = DecodingResult::new(
        vec![1, 2, 3],
        0.0,
        DEFAULT_COMPRESSION_RATIO_THRESHOLD + 0.001,
        true,
        0.0,
        0.0,
    );
    assert!(!passes_quality_check(&result, &config));
}

// ============================================================================
// Compression ratio edge cases
// ============================================================================

#[test]
fn test_compression_ratio_two_tokens() {
    // Two tokens = one bigram. Ratio = 1/1 = 1.0.
    assert!((compression_ratio(&[1, 2]) - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_compression_ratio_all_same() {
    // [A, A, A, A] has 3 bigram slots, 1 unique bigram → ratio = 3.0.
    let cr = compression_ratio(&[42, 42, 42, 42]);
    assert!(
        (cr - 3.0).abs() < f64::EPSILON,
        "all-same tokens: cr={cr}, expected 3.0"
    );
}

#[test]
fn test_compression_ratio_increasing_sequence() {
    // Strictly increasing: all bigrams unique → ratio ≈ 1.0.
    let tokens: Vec<usize> = (0..100).collect();
    let cr = compression_ratio(&tokens);
    assert!(
        (cr - 1.0).abs() < 0.01,
        "increasing sequence: cr={cr}, expected ~1.0"
    );
}

// ============================================================================
// Temperature fallback constants
// ============================================================================

#[test]
fn test_default_temperatures_sequence() {
    assert_eq!(DEFAULT_TEMPERATURES.len(), 6);
    assert!((DEFAULT_TEMPERATURES[0] - 0.0).abs() < f64::EPSILON);
    assert!((DEFAULT_TEMPERATURES[5] - 1.0).abs() < f64::EPSILON);
    // Should be monotonically increasing.
    for i in 1..DEFAULT_TEMPERATURES.len() {
        assert!(
            DEFAULT_TEMPERATURES[i] > DEFAULT_TEMPERATURES[i - 1],
            "temperatures should be increasing"
        );
    }
}

#[test]
fn test_max_decode_length_constant() {
    assert_eq!(MAX_DECODE_LENGTH, 224);
}

// ============================================================================
// Tokenizer decode with timestamps
// ============================================================================

#[test]
fn test_decode_with_timestamps_no_timestamps() {
    let tokenizer = make_test_tokenizer();
    // Regular tokens only — should produce one segment with no times.
    let segments = tokenizer.decode_with_timestamps(&[100, 200, 300]).unwrap();
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start, None);
    assert_eq!(segments[0].end, None);
}

#[test]
fn test_decode_with_timestamps_single_pair() {
    let tokenizer = make_test_tokenizer();
    // [start_ts, token, token, end_ts]
    let start_ts = 50365; // 0.00s
    let end_ts = 50365 + 50; // 1.00s
    let segments = tokenizer
        .decode_with_timestamps(&[start_ts, 100, 200, end_ts])
        .unwrap();

    // Should have one segment with start=0.0 and end=1.0.
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start, Some(0.0));
    assert!((segments[0].end.unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_decode_with_timestamps_eot_skipped() {
    let tokenizer = make_test_tokenizer();
    // EOT should be skipped entirely.
    let segments = tokenizer
        .decode_with_timestamps(&[100, EOT_TOKEN, 200])
        .unwrap();
    assert_eq!(segments.len(), 1);
}

// ============================================================================
// Helper: create a minimal tokenizer for testing
// ============================================================================

fn make_test_tokenizer() -> WhisperTokenizer {
    // Build a minimal vocab JSON with enough entries to test special tokens.
    // We need entries from 0 to at least 51866 for turbo vocab.
    // For testing, a sparse vocab with key tokens is sufficient.
    let mut vocab = std::collections::HashMap::new();

    // A few regular tokens.
    vocab.insert("hello".to_string(), 100_usize);
    vocab.insert("world".to_string(), 200_usize);
    vocab.insert("test".to_string(), 300_usize);

    // Special tokens.
    vocab.insert("<|endoftext|>".to_string(), EOT_TOKEN);
    vocab.insert("<|startoftranscript|>".to_string(), SOT_TOKEN);
    vocab.insert("<|en|>".to_string(), LANGUAGE_TOKEN_START);
    vocab.insert("<|nospeech|>".to_string(), NO_SPEECH_TOKEN);
    vocab.insert("<|notimestamps|>".to_string(), 50364_usize);

    // Some timestamp tokens.
    vocab.insert("<|0.00|>".to_string(), 50365_usize);
    vocab.insert("<|1.00|>".to_string(), 50365 + 50);

    // A high-numbered entry to set vocab_size.
    vocab.insert("<|30.00|>".to_string(), 51865_usize);

    let json = serde_json::to_string(&vocab).unwrap();
    WhisperTokenizer::from_vocab_str(&json).unwrap()
}
