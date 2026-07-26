// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the streaming transcription API.

use super::*;
use crate::config::{N_SAMPLES, SAMPLE_RATE};

// ---------------------------------------------------------------------------
// Unit tests that do NOT require a loaded model (struct/config-level tests)
// ---------------------------------------------------------------------------

#[test]
fn test_streaming_config_default() {
    let config = StreamingConfig::default();
    assert_eq!(config.temperatures.len(), 6);
    assert!((config.no_speech_threshold - 0.6).abs() < 1e-9);
}

#[test]
fn test_streaming_config_builders() {
    let config = StreamingConfig::default()
        .with_temperatures(vec![0.0, 0.5])
        .with_no_speech_threshold(0.8);
    assert_eq!(config.temperatures, vec![0.0, 0.5]);
    assert!((config.no_speech_threshold - 0.8).abs() < 1e-9);
}

#[test]
fn test_streaming_segment_fields() {
    let seg = StreamingSegment::new(
        "hello world".into(),
        vec![1, 2, 3],
        0.0,
        30.0,
        -0.5,
        0.01,
    );
    assert_eq!(seg.text, "hello world");
    assert_eq!(seg.tokens, vec![1, 2, 3]);
    assert!((seg.start_time - 0.0).abs() < 1e-6);
    assert!((seg.end_time - 30.0).abs() < 1e-6);
    assert!((seg.avg_logprob - (-0.5)).abs() < 1e-6);
    assert!((seg.no_speech_prob - 0.01).abs() < 1e-6);
}

#[test]
fn test_extract_timestamp_advance_none() {
    // No timestamp tokens -> None.
    assert!(extract_timestamp_advance(&[100, 200, 300]).is_none());
    assert!(extract_timestamp_advance(&[]).is_none());
}

#[test]
fn test_extract_timestamp_advance_basic() {
    use crate::tokenizer::TIMESTAMP_BEGIN;
    // Token for 1.0 seconds = TIMESTAMP_BEGIN + 50.
    let tokens = vec![100, TIMESTAMP_BEGIN + 50];
    let advance = extract_timestamp_advance(&tokens).unwrap();
    assert!((advance - 1.0).abs() < 1e-9);
}

#[test]
fn test_extract_timestamp_advance_uses_last() {
    use crate::tokenizer::TIMESTAMP_BEGIN;
    // Multiple timestamps — should use the last one.
    let tokens = vec![TIMESTAMP_BEGIN + 25, 100, TIMESTAMP_BEGIN + 100];
    let advance = extract_timestamp_advance(&tokens).unwrap();
    assert!((advance - 2.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// Tests using a zero-weight model (verifies API contract without real weights)
// ---------------------------------------------------------------------------

fn make_zero_model() -> (WhisperModel, WhisperTokenizer) {
    use nn_core::{DType, Device, VarBuilder};

    let config = crate::WhisperConfig::whisper_tiny();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = WhisperModel::load(&vb, config).expect("load zero model");
    // Minimal vocab: empty JSON gives vocab_size=0 which makes decode() skip all tokens.
    let tokenizer = WhisperTokenizer::from_vocab_str("{}").expect("empty tokenizer");
    (model, tokenizer)
}

#[test]
fn test_streaming_transcriber_empty_audio() {
    let (mut model, tokenizer) = make_zero_model();
    let audio: Vec<f32> = vec![];
    let transcriber =
        StreamingTranscriber::new(&mut model, &audio, &tokenizer, StreamingConfig::default());

    assert!(transcriber.is_done());
    assert_eq!(transcriber.total_chunks(), 0);
    assert_eq!(transcriber.remaining_chunks(), 0);
    assert!((transcriber.current_time() - 0.0).abs() < 1e-6);
}

#[test]
fn test_streaming_transcriber_total_chunks_one() {
    let (mut model, tokenizer) = make_zero_model();
    // Exactly one chunk (30 seconds at 16 kHz).
    let audio = vec![0.0f32; N_SAMPLES];
    let transcriber =
        StreamingTranscriber::new(&mut model, &audio, &tokenizer, StreamingConfig::default());

    assert!(!transcriber.is_done());
    assert_eq!(transcriber.total_chunks(), 1);
    assert_eq!(transcriber.remaining_chunks(), 1);
}

#[test]
fn test_streaming_transcriber_total_chunks_partial() {
    let (mut model, tokenizer) = make_zero_model();
    // 1.5 chunks: should round up to 2 total chunks.
    let audio = vec![0.0f32; N_SAMPLES + N_SAMPLES / 2];
    let transcriber =
        StreamingTranscriber::new(&mut model, &audio, &tokenizer, StreamingConfig::default());

    assert!(!transcriber.is_done());
    assert_eq!(transcriber.total_chunks(), 2);
    assert_eq!(transcriber.remaining_chunks(), 2);
}

#[test]
fn test_streaming_transcriber_total_chunks_short_audio() {
    let (mut model, tokenizer) = make_zero_model();
    // 1 second of audio — still 1 chunk (padded to 30s internally).
    let audio = vec![0.0f32; SAMPLE_RATE];
    let transcriber =
        StreamingTranscriber::new(&mut model, &audio, &tokenizer, StreamingConfig::default());

    assert!(!transcriber.is_done());
    assert_eq!(transcriber.total_chunks(), 1);
    assert_eq!(transcriber.remaining_chunks(), 1);
}

#[test]
fn test_streaming_transcriber_current_offset() {
    let (mut model, tokenizer) = make_zero_model();
    let audio = vec![0.0f32; SAMPLE_RATE]; // 1s
    let transcriber =
        StreamingTranscriber::new(&mut model, &audio, &tokenizer, StreamingConfig::default());

    assert_eq!(transcriber.current_offset(), 0);
    assert!((transcriber.current_time() - 0.0).abs() < 1e-6);
}

#[test]
fn test_streaming_config_with_decode_config() {
    use crate::DecodeConfig;
    let dc = DecodeConfig::default().with_max_length(100);
    let config = StreamingConfig::default().with_decode_config(dc);
    assert_eq!(config.decode_config.max_length, 100);
}
