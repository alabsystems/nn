// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Zero-weight Silero VAD contract tests.
//!
//! Tests API contracts using zero-weight models (no real weights needed).
//! Real-weight end-to-end tests are in `silero_vad_e2e.rs`.
//!
//! Split from `silero_vad_e2e.rs` to stay under 500-line limit.

use nn_metal::{MetalBackend, PipelineCache, SileroVad, SileroVadWeights};

/// Number of new audio samples per chunk (32ms at 16kHz).
const CHUNK_SIZE: usize = 512;

/// Zero-weight model for API contract tests (no real weights needed).
fn zero_weight_model() -> SileroVad {
    let weights = SileroVadWeights::new(
        vec![0.0; 258 * 256],
        [
            vec![0.0; 128 * 129 * 3],
            vec![0.0; 64 * 128 * 3],
            vec![0.0; 64 * 64 * 3],
            vec![0.0; 128 * 64 * 3],
        ],
        [vec![0.0; 128], vec![0.0; 64], vec![0.0; 64], vec![0.0; 128]],
        vec![0.0; 512 * 128],
        vec![0.0; 512 * 128],
        vec![0.0; 512],
        vec![0.0; 512],
        vec![0.0; 128],
        vec![0.0; 1],
    );
    SileroVad::new(weights).expect("zero-weight model")
}

/// `get_probabilities()` API contract: chunk counting, trailing padding, empty.
/// Part of #839 — dvoice integration.
#[test]
fn e2e_get_probabilities_zero_weights() {
    let model = zero_weight_model();

    let backend = match MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return,
    };
    let cache = PipelineCache::new(backend.context().clone());

    // Exact multiple of CHUNK_SIZE (3 chunks).
    let audio_3 = vec![0.0f32; 3 * CHUNK_SIZE];
    let probs = model.get_probabilities(&cache, &audio_3).expect("3 chunks");
    assert_eq!(probs.len(), 3, "expected 3 probs, got {}", probs.len());
    for (i, &p) in probs.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&p),
            "chunk {i}: probability {p} outside [0, 1]",
        );
        // Zero weights → sigmoid(0) ≈ 0.5.
        assert!(
            (p - 0.5).abs() < 0.05,
            "chunk {i}: expected ~0.5 for zero weights, got {p}",
        );
    }

    // Non-multiple of CHUNK_SIZE (2 full chunks + 200 trailing samples).
    let audio_partial = vec![0.0f32; 2 * CHUNK_SIZE + 200];
    let probs = model
        .get_probabilities(&cache, &audio_partial)
        .expect("partial");
    assert_eq!(
        probs.len(),
        3,
        "expected 3 probs (2 full + 1 padded), got {}",
        probs.len(),
    );

    // Empty audio — should produce zero probabilities.
    let probs = model.get_probabilities(&cache, &[]).expect("empty");
    assert!(probs.is_empty(), "empty audio should produce 0 probs");

    // Single sample — should be padded to one chunk.
    let probs = model
        .get_probabilities(&cache, &[0.0f32])
        .expect("1 sample");
    assert_eq!(probs.len(), 1, "single sample should produce 1 prob");
}

/// `get_speech_segments()` API wiring: runs without error, valid segments.
/// Part of #839 — dvoice integration.
#[test]
fn e2e_get_speech_segments_zero_weights() {
    let model = zero_weight_model();

    let backend = match MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return,
    };
    let cache = PipelineCache::new(backend.context().clone());
    let config = nn_metal::SegmentConfig::default();

    // 20 chunks of silence — should return segments or not depending on
    // whether zero-weight probabilities are >= 0.5.
    let audio = vec![0.0f32; 20 * CHUNK_SIZE];
    let segments = model
        .get_speech_segments(&cache, &audio, &config, 16000)
        .expect("segment detection");
    // The specific number of segments depends on the exact probability
    // (may be slightly above or below 0.5 due to numerical precision).
    // The key assertion is that the API runs without error and returns
    // valid segments.
    for seg in &segments {
        assert!(
            seg.start_sample < seg.end_sample,
            "invalid segment: {seg:?}"
        );
        assert!(
            seg.end_sample <= audio.len(),
            "segment exceeds audio: {seg:?}"
        );
        assert!(seg.start_time >= 0.0);
        assert!(seg.end_time > seg.start_time);
    }

    // Empty audio — should return no segments.
    let segments = model
        .get_speech_segments(&cache, &[], &config, 16000)
        .expect("empty segments");
    assert!(segments.is_empty());
}
