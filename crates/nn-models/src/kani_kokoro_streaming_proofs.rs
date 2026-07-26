// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for kokoro_streaming_types: KokoroStreamConfig, AudioChunk.
//!
//! Proves that:
//! 1. Default KokoroStreamConfig has crossfade_samples == 960 (40ms at 24kHz).
//! 2. Default KokoroStreamConfig passes validation.
//! 3. Zero crossfade_samples fails validation.
//! 4. crossfade_duration_secs is consistent with crossfade_samples / sample_rate.
//! 5. AudioChunk::len() == pcm.len().
//! 6. AudioChunk::is_empty() iff pcm is empty.
//! 7. AudioChunk::duration_secs() is non-negative for valid chunks.
//! 8. concatenate_chunks preserves total sample count.
//!
//! Part of #3793, #3351.

use crate::kokoro_streaming::{concatenate_chunks, AudioChunk, KokoroStreamConfig};

/// Proof 1: Default crossfade is 960 samples (40ms at 24kHz).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_default_crossfade_is_960() {
    let config = KokoroStreamConfig::default();
    assert_eq!(config.crossfade_samples, 960);
}

/// Proof 2: Default config passes validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_default_stream_config_validates() {
    let config = KokoroStreamConfig::default();
    assert!(config.validate().is_ok());
}

/// Proof 3: Zero crossfade_samples fails validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_zero_crossfade_fails() {
    let result = KokoroStreamConfig::new(0);
    assert!(result.is_err(), "zero crossfade_samples must fail");
}

/// Proof 4: crossfade_duration_secs is consistent.
///
/// duration = crossfade_samples / KOKORO_SAMPLE_RATE (24000).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_crossfade_duration_consistent() {
    let samples: u16 = kani::any();
    kani::assume(samples >= 1);
    let config = KokoroStreamConfig::new(samples as usize).unwrap();
    let duration = config.crossfade_duration_secs();
    assert!(
        duration > 0.0,
        "duration must be positive for nonzero samples"
    );
    assert!(duration.is_finite(), "duration must be finite");
    let expected = samples as f64 / 24000.0;
    assert!(
        (duration - expected).abs() < 1e-12,
        "duration must equal samples / sample_rate"
    );
}

/// Proof 5: AudioChunk::len() == pcm.len().
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_audio_chunk_len() {
    let n: u8 = kani::any();
    kani::assume(n <= 16);
    let pcm = vec![0.0f32; n as usize];
    let expected_len = pcm.len();
    let chunk = AudioChunk::new(pcm, 1, 0, 0, 1, true);
    assert_eq!(chunk.len(), expected_len);
}

/// Proof 6: AudioChunk::is_empty() iff pcm is empty.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_audio_chunk_is_empty() {
    let empty_chunk = AudioChunk::new(vec![], 1, 0, 0, 1, true);
    assert!(empty_chunk.is_empty());

    let nonempty_chunk = AudioChunk::new(vec![0.5], 1, 0, 0, 1, true);
    assert!(!nonempty_chunk.is_empty());
}

/// Proof 7: AudioChunk::duration_secs() is non-negative.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_audio_chunk_duration_nonnegative() {
    let n: u8 = kani::any();
    kani::assume(n <= 16);
    let channels: u8 = kani::any();
    kani::assume(channels >= 1 && channels <= 2);
    let pcm = vec![0.0f32; n as usize];
    let chunk = AudioChunk::new(pcm, channels as usize, 0, 0, 1, true);
    let dur = chunk.duration_secs();
    assert!(dur >= 0.0, "duration must be non-negative");
    assert!(dur.is_finite(), "duration must be finite");
}

/// Proof 8: concatenate_chunks preserves total sample count.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn proof_concatenate_preserves_total() {
    let n1: u8 = kani::any();
    let n2: u8 = kani::any();
    kani::assume(n1 <= 8);
    kani::assume(n2 <= 8);
    let c1 = AudioChunk::new(vec![0.0; n1 as usize], 1, 0, 0, 2, false);
    let c2 = AudioChunk::new(vec![0.0; n2 as usize], 1, n1 as usize, 1, 2, true);
    let result = concatenate_chunks(&[c1, c2]);
    assert_eq!(
        result.len(),
        n1 as usize + n2 as usize,
        "concatenation must preserve total sample count"
    );
}
