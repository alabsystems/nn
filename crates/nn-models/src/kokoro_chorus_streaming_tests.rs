// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`StreamingChorusSession`].
//!
//! Verifies chunk-by-chunk streaming produces state-dependent output:
//! reverb tails, compressor envelopes, and filter memories carry across
//! chunk boundaries.

use super::*;
use crate::kokoro_chorus_pipeline::ChorusMasterConfig;

/// Generate a sine wave at the given frequency and sample count.
fn sine_wave(freq_hz: f32, n_samples: usize, sample_rate: f32) -> Vec<f32> {
    (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            (2.0 * std::f32::consts::PI * freq_hz * t).sin() * 0.5
        })
        .collect()
}

/// Basic: create a session, process one chunk, verify output is non-zero.
#[test]
fn test_streaming_session_single_chunk() {
    let config = ChorusMasterConfig::standard(4).expect("valid config");
    let mut session = StreamingChorusSession::new(config).expect("session created");

    let chunk_len = 512;
    let sr = 24000.0;
    let voices: Vec<Vec<f32>> = (0..4)
        .map(|i| sine_wave(220.0 + i as f32 * 55.0, chunk_len, sr))
        .collect();
    let voice_refs: Vec<&[f32]> = voices.iter().map(Vec::as_slice).collect();

    let mut left = vec![0.0f32; chunk_len];
    let mut right = vec![0.0f32; chunk_len];

    session
        .process_chunk(&voice_refs, &mut left, &mut right)
        .expect("process_chunk succeeded");

    // Output must be non-silent (we fed in sine waves with a standard config).
    let left_energy: f32 = left.iter().map(|s| s * s).sum();
    let right_energy: f32 = right.iter().map(|s| s * s).sum();
    assert!(
        left_energy > 1e-6,
        "left channel should have energy, got {left_energy}"
    );
    assert!(
        right_energy > 1e-6,
        "right channel should have energy, got {right_energy}"
    );

    assert_eq!(session.total_samples_processed(), chunk_len as u64);
}

/// State continuity: output of chunk N+1 depends on chunk N.
///
/// Process the same two chunks in sequence, then process just the second
/// chunk in isolation (fresh session). The outputs for chunk 2 must differ,
/// proving that state from chunk 1 affected chunk 2's processing.
#[test]
fn test_streaming_session_state_continuity() {
    let config = ChorusMasterConfig::singing_chorus(3).expect("valid config");

    let chunk_len = 512;
    let sr = 24000.0;
    let chunk1: Vec<Vec<f32>> = (0..3)
        .map(|i| sine_wave(440.0 + i as f32 * 30.0, chunk_len, sr))
        .collect();
    let chunk2: Vec<Vec<f32>> = (0..3)
        .map(|i| sine_wave(330.0 + i as f32 * 20.0, chunk_len, sr))
        .collect();

    // Path A: chunk1 then chunk2 (sequential, state carries over).
    let mut session_a = StreamingChorusSession::new(config.clone()).expect("session A");
    let refs1: Vec<&[f32]> = chunk1.iter().map(Vec::as_slice).collect();
    let refs2: Vec<&[f32]> = chunk2.iter().map(Vec::as_slice).collect();
    let mut a_l1 = vec![0.0f32; chunk_len];
    let mut a_r1 = vec![0.0f32; chunk_len];
    session_a
        .process_chunk(&refs1, &mut a_l1, &mut a_r1)
        .expect("A chunk1");
    let mut a_l2 = vec![0.0f32; chunk_len];
    let mut a_r2 = vec![0.0f32; chunk_len];
    session_a
        .process_chunk(&refs2, &mut a_l2, &mut a_r2)
        .expect("A chunk2");

    // Path B: only chunk2 (fresh session, no prior state).
    let mut session_b = StreamingChorusSession::new(config).expect("session B");
    let mut b_l2 = vec![0.0f32; chunk_len];
    let mut b_r2 = vec![0.0f32; chunk_len];
    session_b
        .process_chunk(&refs2, &mut b_l2, &mut b_r2)
        .expect("B chunk2");

    // The second chunk's output must differ between the two paths.
    // Compute MSE between path A and path B for the left channel.
    let mse: f32 = a_l2
        .iter()
        .zip(b_l2.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / chunk_len as f32;

    assert!(
        mse > 1e-10,
        "chunk 2 output should differ between sequential and fresh sessions (MSE={mse}), \
         proving state continuity"
    );

    assert_eq!(session_a.total_samples_processed(), 2 * chunk_len as u64);
}

/// Different chunk sizes work correctly.
#[test]
fn test_streaming_session_varying_chunk_sizes() {
    let config = ChorusMasterConfig::minimal(2).expect("valid config");
    let mut session = StreamingChorusSession::new(config).expect("session");

    let sr = 24000.0;
    let chunk_sizes = [256, 1024, 128, 512, 64];

    let mut total = 0u64;
    for &chunk_len in &chunk_sizes {
        let voices: Vec<Vec<f32>> = (0..2)
            .map(|i| sine_wave(300.0 + i as f32 * 50.0, chunk_len, sr))
            .collect();
        let voice_refs: Vec<&[f32]> = voices.iter().map(Vec::as_slice).collect();
        let mut left = vec![0.0f32; chunk_len];
        let mut right = vec![0.0f32; chunk_len];
        session
            .process_chunk(&voice_refs, &mut left, &mut right)
            .expect("varied chunk size");
        total += chunk_len as u64;
    }

    assert_eq!(session.total_samples_processed(), total);
}

/// Flush drains reverb/delay tails (non-silent output from silence input).
#[test]
fn test_streaming_session_flush() {
    // Use a config with reverb enabled to produce a non-trivial tail.
    let config = ChorusMasterConfig::cathedral(3).expect("valid config");
    let mut session = StreamingChorusSession::new(config).expect("session");

    // First feed in a loud chunk to prime the reverb.
    let chunk_len = 1024;
    let sr = 24000.0;
    let voices: Vec<Vec<f32>> = (0..3)
        .map(|i| sine_wave(440.0 + i as f32 * 55.0, chunk_len, sr))
        .collect();
    let voice_refs: Vec<&[f32]> = voices.iter().map(Vec::as_slice).collect();
    let mut left = vec![0.0f32; chunk_len];
    let mut right = vec![0.0f32; chunk_len];
    session
        .process_chunk(&voice_refs, &mut left, &mut right)
        .expect("prime reverb");

    // Flush: push silence through to drain the reverb tail.
    let (tail_l, tail_r) = session.flush().expect("flush succeeded");

    // The tail should contain non-zero energy from the reverb.
    let tail_energy: f32 = tail_l.iter().chain(tail_r.iter()).map(|s| s * s).sum();
    assert!(
        tail_energy > 1e-8,
        "flush tail should contain reverb energy, got {tail_energy}"
    );
}

/// Reset clears state so subsequent output matches a fresh session.
#[test]
fn test_streaming_session_reset() {
    let config = ChorusMasterConfig::standard(2).expect("valid config");

    let chunk_len = 512;
    let sr = 24000.0;
    let chunk: Vec<Vec<f32>> = (0..2)
        .map(|i| sine_wave(440.0 + i as f32 * 55.0, chunk_len, sr))
        .collect();
    let refs: Vec<&[f32]> = chunk.iter().map(Vec::as_slice).collect();

    // Session A: process chunk, reset, process same chunk again.
    let mut session_a = StreamingChorusSession::new(config.clone()).expect("session A");
    let mut tmp_l = vec![0.0f32; chunk_len];
    let mut tmp_r = vec![0.0f32; chunk_len];
    session_a
        .process_chunk(&refs, &mut tmp_l, &mut tmp_r)
        .expect("A first chunk");
    session_a.reset();
    let mut a_l = vec![0.0f32; chunk_len];
    let mut a_r = vec![0.0f32; chunk_len];
    session_a
        .process_chunk(&refs, &mut a_l, &mut a_r)
        .expect("A after reset");

    // Session B: fresh session, process same chunk.
    let mut session_b = StreamingChorusSession::new(config).expect("session B");
    let mut b_l = vec![0.0f32; chunk_len];
    let mut b_r = vec![0.0f32; chunk_len];
    session_b
        .process_chunk(&refs, &mut b_l, &mut b_r)
        .expect("B fresh");

    // After reset, output should be identical (or very close) to fresh.
    let mse: f32 = a_l
        .iter()
        .zip(b_l.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / chunk_len as f32;

    // Tolerance is 1e-4 rather than exact because some processors have
    // PRNG or float-accumulation state that does not reset to bit-identical
    // values. The state_continuity test above shows MSE >> 1e-4 when state
    // leaks, so this threshold distinguishes "reset worked" from "state leaked".
    assert!(
        mse < 1e-4,
        "after reset, output should match fresh session (MSE={mse})"
    );

    assert_eq!(session_a.total_samples_processed(), chunk_len as u64);
}

/// Error: wrong number of voices.
#[test]
fn test_streaming_session_wrong_voice_count() {
    let config = ChorusMasterConfig::minimal(3).expect("valid config");
    let mut session = StreamingChorusSession::new(config).expect("session");

    let voices: Vec<Vec<f32>> = vec![vec![0.0; 256]; 2]; // 2 voices, expected 3
    let refs: Vec<&[f32]> = voices.iter().map(Vec::as_slice).collect();
    let mut left = vec![0.0f32; 256];
    let mut right = vec![0.0f32; 256];

    let result = session.process_chunk(&refs, &mut left, &mut right);
    assert!(result.is_err(), "should error on wrong voice count");
}

/// Error: mismatched voice lengths.
#[test]
fn test_streaming_session_mismatched_voice_lengths() {
    let config = ChorusMasterConfig::minimal(2).expect("valid config");
    let mut session = StreamingChorusSession::new(config).expect("session");

    let v0 = vec![0.0f32; 256];
    let v1 = vec![0.0f32; 128]; // different length
    let refs: Vec<&[f32]> = vec![v0.as_slice(), v1.as_slice()];
    let mut left = vec![0.0f32; 256];
    let mut right = vec![0.0f32; 256];

    let result = session.process_chunk(&refs, &mut left, &mut right);
    assert!(result.is_err(), "should error on mismatched voice lengths");
}

/// Error: output buffer too short.
#[test]
fn test_streaming_session_output_buffer_too_short() {
    let config = ChorusMasterConfig::minimal(2).expect("valid config");
    let mut session = StreamingChorusSession::new(config).expect("session");

    let voices: Vec<Vec<f32>> = vec![vec![0.0; 256]; 2];
    let refs: Vec<&[f32]> = voices.iter().map(Vec::as_slice).collect();
    let mut left = vec![0.0f32; 128]; // too short
    let mut right = vec![0.0f32; 256];

    let result = session.process_chunk(&refs, &mut left, &mut right);
    assert!(result.is_err(), "should error on short left buffer");
}
