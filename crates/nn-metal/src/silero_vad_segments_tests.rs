// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for speech segment detection from per-chunk VAD probabilities.

use super::*;

#[test]
fn test_empty_probs_returns_empty() {
    let result = segments_from_probs(&[], 0, &SegmentConfig::default(), 512, 16000);
    assert!(result.is_empty());
}

#[test]
fn test_all_silence_returns_empty() {
    let probs = vec![0.1f32; 100]; // 100 chunks of silence
    let audio_len = 100 * 512;
    let result = segments_from_probs(&probs, audio_len, &SegmentConfig::default(), 512, 16000);
    assert!(result.is_empty());
}

#[test]
fn test_continuous_speech_detected() {
    // 50 chunks of speech (1.6 seconds at 32ms/chunk)
    let probs = vec![0.9f32; 50];
    let audio_len = 50 * 512;
    let result = segments_from_probs(&probs, audio_len, &SegmentConfig::default(), 512, 16000);
    assert_eq!(result.len(), 1);
    assert!(result[0].duration() > 1.0);
}

#[test]
fn test_speech_segment_duration() {
    let seg = SpeechSegment {
        start_sample: 0,
        end_sample: 16000,
        start_time: 0.0,
        end_time: 1.0,
    };
    assert!((seg.duration() - 1.0).abs() < 1e-6);
}

#[test]
fn test_default_config() {
    let config = SegmentConfig::default();
    assert!((config.threshold - 0.5).abs() < 1e-6);
    assert_eq!(config.min_speech_duration_ms, 250);
    assert_eq!(config.min_silence_duration_ms, 300);
    assert_eq!(config.speech_pad_ms, 30);
    assert!((config.max_speech_duration_s).abs() < 1e-6);
}

#[test]
fn test_split_long_segments() {
    let sr = 16000.0f32;
    let mut segments = vec![SpeechSegment {
        start_sample: 0,
        end_sample: 48000, // 3 seconds
        start_time: 0.0,
        end_time: 3.0,
    }];
    split_long_segments(&mut segments, 1.0, sr);
    assert_eq!(segments.len(), 3);
    assert_eq!(segments[0].end_sample, 16000);
    assert_eq!(segments[1].start_sample, 16000);
    assert_eq!(segments[1].end_sample, 32000);
    assert_eq!(segments[2].start_sample, 32000);
    assert_eq!(segments[2].end_sample, 48000);
}

/// Speech surrounded by silence produces exactly one segment.
/// Pattern: 10 silence + 20 speech + 20 silence (50 chunks = 1.6s).
#[test]
fn test_speech_silence_speech_single_burst() {
    let mut probs = vec![0.1f32; 10]; // silence
    probs.extend(vec![0.9f32; 20]); // speech (640ms > min_speech 250ms)
    probs.extend(vec![0.1f32; 20]); // silence (640ms > min_silence 300ms)
    let audio_len = 50 * 512;
    let result = segments_from_probs(&probs, audio_len, &SegmentConfig::default(), 512, 16000);
    assert_eq!(result.len(), 1, "expected 1 segment, got {}", result.len());
    // Speech started at chunk 10 (sample 5120), ended within the silence region.
    assert!(result[0].start_sample <= 10 * 512 + 512);
    assert!(result[0].end_sample >= 20 * 512);
}

/// Two speech bursts separated by enough silence produce two segments.
#[test]
fn test_two_speech_bursts() {
    let mut probs = Vec::new();
    probs.extend(vec![0.9f32; 20]); // first speech burst (640ms)
    probs.extend(vec![0.1f32; 20]); // silence gap (640ms > min_silence 300ms)
    probs.extend(vec![0.9f32; 20]); // second speech burst (640ms)
    probs.extend(vec![0.1f32; 20]); // trailing silence
    let audio_len = 80 * 512;
    let result = segments_from_probs(&probs, audio_len, &SegmentConfig::default(), 512, 16000);
    assert_eq!(result.len(), 2, "expected 2 segments, got {result:?}");
    // First segment should end before second starts.
    assert!(
        result[0].end_sample <= result[1].start_sample,
        "segments overlap: {result:?}",
    );
}

/// Very short speech burst below min_speech_duration_ms is rejected.
#[test]
fn test_short_speech_rejected() {
    let mut probs = vec![0.1f32; 10]; // silence
    probs.extend(vec![0.9f32; 3]); // speech: 3 chunks = 96ms < 250ms min
    probs.extend(vec![0.1f32; 20]); // silence
    let audio_len = 33 * 512;
    let result = segments_from_probs(&probs, audio_len, &SegmentConfig::default(), 512, 16000);
    assert!(
        result.is_empty(),
        "short speech should be rejected, got {result:?}",
    );
}

/// Custom threshold: lower threshold detects quieter speech.
#[test]
fn test_custom_threshold() {
    // Probabilities at 0.35 — below default 0.5 but above custom 0.3.
    let mut probs = vec![0.1f32; 5];
    probs.extend(vec![0.35f32; 20]); // speech at custom threshold
    probs.extend(vec![0.1f32; 20]); // silence
    let audio_len = 45 * 512;

    // Default threshold (0.5): no speech detected.
    let default_result =
        segments_from_probs(&probs, audio_len, &SegmentConfig::default(), 512, 16000);
    assert!(
        default_result.is_empty(),
        "default threshold should miss 0.35 probs",
    );

    // Custom threshold (0.3): speech detected.
    let config = SegmentConfig {
        threshold: 0.3,
        ..SegmentConfig::default()
    };
    let custom_result = segments_from_probs(&probs, audio_len, &config, 512, 16000);
    assert_eq!(
        custom_result.len(),
        1,
        "custom threshold should detect 0.35 probs, got {custom_result:?}",
    );
}

/// max_speech_duration_s splits long segments through the public API path.
#[test]
fn test_max_speech_duration_integration() {
    // 100 chunks of speech = 3.2 seconds at 32ms/chunk.
    let probs = vec![0.9f32; 100];
    let audio_len = 100 * 512;
    let config = SegmentConfig {
        max_speech_duration_s: 1.0,
        ..SegmentConfig::default()
    };
    let result = segments_from_probs(&probs, audio_len, &config, 512, 16000);
    // 3.2s speech should be split into at least 3 segments (1s + 1s + 1.2s).
    assert!(
        result.len() >= 3,
        "expected >=3 segments from 3.2s speech with 1s max, got {}",
        result.len(),
    );
    for seg in &result {
        assert!(
            seg.duration() <= 1.1,
            "segment duration {:.2}s exceeds max 1.0s (with tolerance)",
            seg.duration(),
        );
    }
}

/// Extreme config values (u32::MAX) should not panic or produce usize::MAX.
#[test]
fn test_extreme_config_values_do_not_panic() {
    let probs = vec![0.9f32; 50];
    let audio_len = 50 * 512;
    let config = SegmentConfig {
        speech_pad_ms: u32::MAX,
        min_speech_duration_ms: u32::MAX,
        min_silence_duration_ms: u32::MAX,
        ..SegmentConfig::default()
    };
    // Should not panic — finiteness guards saturate to safe defaults.
    let result = segments_from_probs(&probs, audio_len, &config, 512, 16000);
    // With u32::MAX min_speech_duration, no segment can be long enough.
    assert!(
        result.is_empty(),
        "extreme min_speech should reject all segments, got {result:?}",
    );
}

/// Zero chunk_size returns empty (division by zero defense).
#[test]
fn test_zero_chunk_size_returns_empty() {
    let probs = vec![0.9f32; 10];
    let result = segments_from_probs(&probs, 5120, &SegmentConfig::default(), 0, 16000);
    assert!(result.is_empty());
}

/// Zero sample_rate returns empty (division by zero defense).
#[test]
fn test_zero_sample_rate_returns_empty() {
    let probs = vec![0.9f32; 10];
    let result = segments_from_probs(&probs, 5120, &SegmentConfig::default(), 512, 0);
    assert!(result.is_empty());
}

/// Adjacent segments near audio end: end_sample must be clamped to audio_len.
/// Regression test for #1612: `apply_padding` adjacent-gap branch was missing
/// `.min(audio_len)`, allowing `end_sample > audio_len`.
#[test]
fn test_adjacent_segments_near_end_clamped_to_audio_len() {
    let sr = 16000.0;
    // Build two adjacent segments whose gap is small enough to trigger
    // the `gap < 2 * pad` branch in apply_padding.
    // Place the second segment at the very end of the audio so that
    // padding the first segment's end could exceed audio_len.
    let mut segments = vec![
        SpeechSegment {
            start_sample: 14000,
            end_sample: 15900,
            start_time: 14000.0 / sr,
            end_time: 15900.0 / sr,
        },
        SpeechSegment {
            start_sample: 15950,
            end_sample: 16000,
            start_time: 15950.0 / sr,
            end_time: 16000.0 / sr,
        },
    ];
    // audio_len = 16000 (1 second). gap = 15950 - 15900 = 50.
    // pad = 480 (30ms * 16000 / 1000). gap (50) < 2*pad (960), so
    // the adjacent-gap branch fires: half = 50/2 = 25.
    // end_sample = 15900 + 25 = 15925, which is within audio_len.
    let pad = 480;
    let audio_len = 16000;
    apply_padding(&mut segments, pad, audio_len, sr);
    assert!(
        segments[0].end_sample <= audio_len,
        "first segment end_sample {} exceeds audio_len {}",
        segments[0].end_sample,
        audio_len,
    );
    assert!(
        segments[1].end_sample <= audio_len,
        "second segment end_sample {} exceeds audio_len {}",
        segments[1].end_sample,
        audio_len,
    );

    // Now test with end_sample very close to audio_len where half would push past.
    let mut segments2 = vec![
        SpeechSegment {
            start_sample: 15990,
            end_sample: 15996,
            start_time: 15990.0 / sr,
            end_time: 15996.0 / sr,
        },
        SpeechSegment {
            start_sample: 16000,
            end_sample: 16004,
            start_time: 16000.0 / sr,
            end_time: 16004.0 / sr,
        },
    ];
    // gap = 16000 - 15996 = 4. half = 2. end_sample = 15996 + 2 = 15998.
    // audio_len = 15999 — so 15998 is fine, but let's test with audio_len = 15997.
    let audio_len2 = 15997;
    apply_padding(&mut segments2, pad, audio_len2, sr);
    assert!(
        segments2[0].end_sample <= audio_len2,
        "end_sample {} exceeds audio_len {} after adjacent padding",
        segments2[0].end_sample,
        audio_len2,
    );
}
