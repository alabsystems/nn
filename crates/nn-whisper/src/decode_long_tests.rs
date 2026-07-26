// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for long-form audio transcription.

use super::*;
use crate::decode::DecodingResult;

#[test]
fn test_extract_timestamp_advance_no_timestamps() {
    // Tokens with no timestamp tokens → None.
    let tokens = vec![0, 1, 2, 3];
    assert_eq!(extract_timestamp_advance(&tokens), None);
}

#[test]
fn test_extract_timestamp_advance_single_timestamp() {
    // TIMESTAMP_BEGIN = 50365 → 0.00s
    let tokens = vec![0, 1, TIMESTAMP_BEGIN];
    let advance = extract_timestamp_advance(&tokens).unwrap();
    assert!((advance - 0.0).abs() < 1e-10);
}

#[test]
fn test_extract_timestamp_advance_end_timestamp() {
    // 50465 = TIMESTAMP_BEGIN + 100 → 100 * 0.02 = 2.00s
    let tokens = vec![0, 1, TIMESTAMP_BEGIN, 0, 1, TIMESTAMP_BEGIN + 100];
    let advance = extract_timestamp_advance(&tokens).unwrap();
    assert!((advance - 2.0).abs() < 1e-10);
}

#[test]
fn test_extract_timestamp_advance_30s() {
    // TIMESTAMP_BEGIN + 1500 → 1500 * 0.02 = 30.0s (full chunk)
    let tokens = vec![TIMESTAMP_BEGIN, 0, TIMESTAMP_BEGIN + 1500];
    let advance = extract_timestamp_advance(&tokens).unwrap();
    assert!((advance - 30.0).abs() < 1e-10);
}

#[test]
fn test_extract_timestamp_advance_uses_last() {
    // Multiple timestamp tokens — uses the last one.
    let tokens = vec![TIMESTAMP_BEGIN + 50, 0, TIMESTAMP_BEGIN + 200];
    let advance = extract_timestamp_advance(&tokens).unwrap();
    // 200 * 0.02 = 4.0s
    assert!((advance - 4.0).abs() < 1e-10);
}

#[test]
fn test_long_form_config_default() {
    let config = LongFormConfig::default();
    assert!((config.no_speech_threshold - 0.6).abs() < 1e-10);
    assert_eq!(config.temperatures.len(), 6);
    assert!((config.temperatures[0] - 0.0).abs() < 1e-10);
}

#[test]
fn test_transcribe_long_empty_audio() {
    // Empty audio should return an error.
    use nn_core::{DType, Device, VarBuilder};
    let config = crate::WhisperConfig::default();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let mut model = WhisperModel::load(&vb, config).unwrap();
    let tokenizer = WhisperTokenizer::from_vocab_str("{}").unwrap();
    let long_config = LongFormConfig::default();

    let result = transcribe_long(&mut model, &[], &tokenizer, &long_config);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("empty"),
        "expected empty audio error, got: {err}"
    );
}

#[test]
fn test_long_form_segment_fields() {
    // Verify segment struct fields are accessible.
    let segment = LongFormSegment {
        text: "hello".into(),
        start: 0.0,
        end: 5.0,
        decode_result: DecodingResult::new(vec![0, 1], -0.5, 1.2, true, 0.0, 0.1),
    };
    assert_eq!(segment.text, "hello");
    assert!((segment.start - 0.0).abs() < 1e-10);
    assert!((segment.end - 5.0).abs() < 1e-10);
}

#[test]
fn test_long_form_result_fields() {
    let result = LongFormResult {
        text: "hello world".into(),
        segments: vec![],
    };
    assert_eq!(result.text, "hello world");
    assert!(result.segments.is_empty());
}

// -- Timestamp edge case tests --

#[test]
fn test_extract_timestamp_advance_empty_tokens() {
    let tokens: Vec<usize> = Vec::new();
    assert_eq!(extract_timestamp_advance(&tokens), None);
}

#[test]
fn test_extract_timestamp_advance_all_below_timestamp_begin() {
    // All tokens below TIMESTAMP_BEGIN are not timestamps.
    let tokens = vec![100, 200, 50000, TIMESTAMP_BEGIN - 1];
    assert_eq!(extract_timestamp_advance(&tokens), None);
}

#[test]
fn test_extract_timestamp_advance_beyond_30s() {
    // Token ID far beyond TIMESTAMP_BEGIN + 1500 (max valid 30.0s).
    // Model hallucination case: advance > 30s.
    let tokens = vec![TIMESTAMP_BEGIN + 3000]; // 3000 * 0.02 = 60.0s
    let advance = extract_timestamp_advance(&tokens).unwrap();
    assert!(
        (advance - 60.0).abs() < 1e-10,
        "should compute 60.0s: {advance}"
    );
    // This is accepted — transcribe_long clips segment_end to audio_duration
    // and the seek advances past the current chunk, which is safe.
}

#[test]
fn test_extract_timestamp_advance_mixed_tokens_last_wins() {
    // Non-timestamp tokens interleaved — only last timestamp matters.
    let tokens = vec![TIMESTAMP_BEGIN + 50, 100, 200, TIMESTAMP_BEGIN + 150, 300];
    // Last token (300) is below TIMESTAMP_BEGIN, so last timestamp is +150.
    let advance = extract_timestamp_advance(&tokens).unwrap();
    assert!(
        (advance - 3.0).abs() < 1e-10,
        "150 * 0.02 = 3.0s: {advance}"
    );
}

#[test]
fn test_extract_timestamp_advance_only_timestamp_tokens() {
    // All tokens are timestamps — last one wins.
    let tokens = vec![
        TIMESTAMP_BEGIN,
        TIMESTAMP_BEGIN + 250,
        TIMESTAMP_BEGIN + 500,
    ];
    let advance = extract_timestamp_advance(&tokens).unwrap();
    assert!(
        (advance - 10.0).abs() < 1e-10,
        "500 * 0.02 = 10.0s: {advance}"
    );
}

// -- Seek advance clamp tests (W3-93 / #1648) --

#[test]
fn test_advance_samples_clamped_to_n_samples() {
    // Corrupted timestamp token far beyond 30s should be clamped.
    // TIMESTAMP_BEGIN + 100_000 → 100_000 * 0.02 = 2000s → 2000 * 16_000 = 32M samples.
    // Without the .min(N_SAMPLES) clamp, this would overshoot audio.len() massively.
    // With the clamp, advance_samples <= N_SAMPLES (480_000).
    let tokens = vec![TIMESTAMP_BEGIN + 100_000]; // 2000s timestamp
    let advance_sec = extract_timestamp_advance(&tokens).unwrap();
    assert!(
        (advance_sec - 2000.0).abs() < 1e-6,
        "should compute 2000.0s: {advance_sec}"
    );
    // Verify the clamp logic matches transcribe_long:
    // advance_samples = (advance_sec * SAMPLE_RATE as f64) as usize → 32_000_000
    // .min(N_SAMPLES) → 480_000
    let advance_samples = ((advance_sec * SAMPLE_RATE as f64) as usize).min(N_SAMPLES);
    assert_eq!(
        advance_samples, N_SAMPLES,
        "corrupted timestamp advance should be clamped to N_SAMPLES"
    );
}

#[test]
fn test_advance_samples_normal_timestamp_not_clamped() {
    // Normal timestamp within 30s should not be clamped.
    let tokens = vec![TIMESTAMP_BEGIN + 500]; // 500 * 0.02 = 10.0s
    let advance_sec = extract_timestamp_advance(&tokens).unwrap();
    let advance_samples = ((advance_sec * SAMPLE_RATE as f64) as usize).min(N_SAMPLES);
    let expected = (10.0 * SAMPLE_RATE as f64) as usize; // 160_000
    assert_eq!(
        advance_samples, expected,
        "normal timestamp should not be clamped"
    );
}
