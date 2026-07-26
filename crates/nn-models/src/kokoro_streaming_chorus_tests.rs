// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for chorus streaming assembly (multi-voice + crossfade).
//!
//! Tests `assemble_streaming_chorus()` including mono and stereo chorus paths,
//! variable-length voices, compositional equivalence, and stereo interleaving.
//!
//! Extracted from `kokoro_streaming_tests.rs` (#3504, item 1) to comply
//! with the 500-line file limit.

use super::*;

/// Proof: chorus mixing with 32 voices (max) does not panic or overflow.
///
/// Equal-gain mixing with N=32 voices at gain=1/32 keeps mixed output
/// in [-1.0, 1.0]. This tests the upper bound of the voice count and
/// verifies no intermediate overflow in the accumulation loop.
#[test]
fn test_chorus_max_voices_no_overflow() {
    let cf = 5;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let n_voices = 32;
    let gain = 1.0 / n_voices as f32;
    let chorus_config = ChorusConfig::with_gains(vec![gain; n_voices])
        .unwrap()
        .with_clip(true);

    // All voices: constant +/-1.0 (worst case for accumulation).
    let per_voice: Vec<Vec<Vec<f32>>> = (0..n_voices)
        .map(|v| {
            vec![(0..200)
                .map(|i| if (i + v) % 2 == 0 { 1.0f32 } else { -1.0 })
                .collect()]
        })
        .collect();

    let chunks = assemble_streaming_chorus(&per_voice, &chorus_config, &config).unwrap();

    assert_eq!(chunks.len(), 1);
    // With clipping, all samples should be in [-1.0, 1.0].
    for &s in &chunks[0].pcm {
        assert!(
            (-1.0 - 1e-6..=1.0 + 1e-6).contains(&s),
            "sample {s} out of clipped range with 32 voices",
        );
    }
}

/// Proof: assemble_streaming_chorus mixing is equivalent to
/// mix_voices + assemble_streaming_chunks (compositional equivalence).
///
/// assemble_streaming_chorus delegates to mix_voices_with_config per chunk.
/// mix_voices in kokoro_chorus.rs is the standalone reference implementation.
/// This test proves the two paths produce identical output for multi-voice,
/// multi-chunk scenarios -- catching any future divergence if one implementation
/// is modified without the other.
#[test]
fn test_chorus_mixing_equivalent_to_mix_voices_compose() {
    use crate::kokoro_chorus::mix_voices;

    let cf = 10;
    let config = KokoroStreamConfig::new(cf).unwrap();

    // 3 voices, 3 chunks each -- with non-trivial, distinct values.
    let voice0 = vec![
        (0..120)
            .map(|i| (i as f32 * 0.01).sin())
            .collect::<Vec<_>>(),
        (0..100)
            .map(|i| (i as f32 * 0.02).cos())
            .collect::<Vec<_>>(),
        (0..80).map(|i| (i as f32 * 0.03).sin()).collect::<Vec<_>>(),
    ];
    let voice1 = vec![
        (0..120)
            .map(|i| -(i as f32 * 0.015).cos())
            .collect::<Vec<_>>(),
        (0..100)
            .map(|i| (i as f32 * 0.025).sin())
            .collect::<Vec<_>>(),
        (0..80)
            .map(|i| -(i as f32 * 0.035).cos())
            .collect::<Vec<_>>(),
    ];
    let voice2 = vec![
        (0..120)
            .map(|i| (i as f32 * 0.005) * 0.5)
            .collect::<Vec<_>>(),
        (0..100)
            .map(|i| 0.3 - (i as f32 * 0.01))
            .collect::<Vec<_>>(),
        (0..80)
            .map(|i| (i as f32 * 0.04).sin() * 0.7)
            .collect::<Vec<_>>(),
    ];

    let per_voice = vec![voice0.clone(), voice1, voice2];
    let gains = vec![0.4f32, 0.35, 0.25];
    let chorus_config = ChorusConfig::with_gains(gains.clone())
        .unwrap()
        .with_clip(true);

    // Path A: assemble_streaming_chorus (mix_voices_with_config + assembly).
    let chorus_chunks = assemble_streaming_chorus(&per_voice, &chorus_config, &config).unwrap();

    // Path B: mix_voices per chunk, then assemble_streaming_chunks.
    let n_chunks = voice0.len();
    let mut mixed_raw: Vec<Vec<f32>> = Vec::with_capacity(n_chunks);
    for chunk_idx in 0..n_chunks {
        let voice_audio: Vec<Vec<f32>> = per_voice.iter().map(|v| v[chunk_idx].clone()).collect();
        let mixed = mix_voices(&voice_audio, &gains, true).unwrap();
        mixed_raw.push(mixed);
    }
    let composed_chunks = assemble_streaming_chunks(&mixed_raw, &config).unwrap();

    // Verify: both paths produce identical output.
    assert_eq!(
        chorus_chunks.len(),
        composed_chunks.len(),
        "chunk count mismatch: chorus={}, composed={}",
        chorus_chunks.len(),
        composed_chunks.len(),
    );
    for (i, (cc, sc)) in chorus_chunks.iter().zip(composed_chunks.iter()).enumerate() {
        assert_eq!(
            cc.pcm.len(),
            sc.pcm.len(),
            "chunk {i} length mismatch: chorus={}, composed={}",
            cc.pcm.len(),
            sc.pcm.len(),
        );
        assert_eq!(
            cc.sample_offset, sc.sample_offset,
            "chunk {i} offset mismatch"
        );
        assert_eq!(cc.chunk_index, sc.chunk_index, "chunk {i} index mismatch");
        assert_eq!(cc.is_final, sc.is_final, "chunk {i} is_final mismatch");
        for (j, (&cv, &sv)) in cc.pcm.iter().zip(sc.pcm.iter()).enumerate() {
            assert!(
                (cv - sv).abs() < 1e-6,
                "chunk {i} sample {j}: chorus={cv}, composed={sv}, diff={}",
                (cv - sv).abs(),
            );
        }
    }
}

/// Proof: chorus with variable-length voice chunks (zero-padding) is safe.
///
/// When voices produce different-length audio for the same chunk index,
/// mix_voices zero-pads shorter voices. This test verifies no bounds panic
/// and correct mixed length.
#[test]
fn test_chorus_variable_length_voices_safe() {
    let cf = 5;
    let config = KokoroStreamConfig::new(cf).unwrap();

    // Voice 0: 100 samples. Voice 1: 150 samples (longer).
    let voice0 = vec![vec![1.0f32; 100]];
    let voice1 = vec![vec![0.5f32; 150]];

    let chorus_config = ChorusConfig::with_gains(vec![0.5, 0.5])
        .unwrap()
        .with_clip(false);
    let chunks = assemble_streaming_chorus(&[voice0, voice1], &chorus_config, &config).unwrap();

    assert_eq!(chunks.len(), 1);
    // Output length = max(100, 150) = 150.
    assert_eq!(chunks[0].pcm.len(), 150);

    // First 100 samples: mixed from both voices.
    for &s in &chunks[0].pcm[..100] {
        let expected = 1.0 * 0.5 + 0.5 * 0.5; // 0.75
        assert!(
            (s - expected).abs() < 1e-5,
            "overlap region: expected {expected}, got {s}",
        );
    }
    // Samples 100..150: only voice 1 contributes (voice 0 zero-padded).
    for &s in &chunks[0].pcm[100..] {
        let expected = 0.5 * 0.5; // 0.25
        assert!(
            (s - expected).abs() < 1e-5,
            "tail region: expected {expected}, got {s}",
        );
    }
}

// ---------------------------------------------------------------------------
// Stereo chorus streaming
// ---------------------------------------------------------------------------

/// Stereo chorus produces interleaved L/R output with channels=2.
///
/// Two voices panned hard left (-1.0) and hard right (1.0) should produce
/// interleaved stereo where voice0 appears only in L and voice1 only in R.
#[test]
fn test_stereo_chorus_produces_interleaved_output() {
    let cf = 5;
    let config = KokoroStreamConfig::new(cf).unwrap();

    // Voice 0: constant 1.0, panned hard left (-1.0).
    // Voice 1: constant 0.5, panned hard right (1.0).
    let voice0 = vec![vec![1.0f32; 50]];
    let voice1 = vec![vec![0.5f32; 50]];

    let chorus_config = ChorusConfig::with_stereo_pan(vec![1.0, 1.0], vec![-1.0, 1.0])
        .unwrap()
        .with_clip(false);
    let chunks = assemble_streaming_chorus(&[voice0, voice1], &chorus_config, &config).unwrap();

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].channels, 2, "stereo output must have channels=2");
    // Stereo interleaved: length = mono_samples * 2.
    assert_eq!(chunks[0].pcm.len(), 50 * 2);

    // Hard left pan: angle = 0, cos(0) = 1.0, sin(0) = 0.0.
    // Hard right pan: angle = pi/2, cos(pi/2) ~ 0, sin(pi/2) = 1.0.
    // So L = voice0 * 1.0 * 1.0 + voice1 * 1.0 * 0.0 = 1.0
    //    R = voice0 * 1.0 * 0.0 + voice1 * 1.0 * 1.0 = 0.5
    for i in 0..50 {
        let l = chunks[0].pcm[i * 2];
        let r = chunks[0].pcm[i * 2 + 1];
        assert!(
            (l - 1.0).abs() < 1e-4,
            "sample {i} L: expected ~1.0, got {l}",
        );
        assert!(
            (r - 0.5).abs() < 1e-4,
            "sample {i} R: expected ~0.5, got {r}",
        );
    }
}

/// Mono chorus keeps channels=1.
#[test]
fn test_mono_chorus_channels_field() {
    let cf = 5;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let voice0 = vec![vec![1.0f32; 50]];
    let chorus_config = ChorusConfig::with_gains(vec![1.0]).unwrap();
    let chunks = assemble_streaming_chorus(&[voice0], &chorus_config, &config).unwrap();
    assert_eq!(chunks[0].channels, 1);
}

/// Stereo crossfade covers the same wall-clock duration as mono.
///
/// For mono, crossfade_samples = N covers N/24000 seconds.
/// For stereo, the interleaved buffer has 2x floats per sample-pair.
/// The implementation doubles the effective crossfade to 2*N floats,
/// preserving the same time duration.
#[test]
fn test_stereo_crossfade_preserves_time_duration() {
    let cf = 10; // 10 mono samples crossfade
    let config = KokoroStreamConfig::new(cf).unwrap();

    // 2 voices, 2 chunks each. Same data for both mono and stereo.
    let voice0 = vec![vec![1.0f32; 100], vec![0.0f32; 100]];
    let voice1 = vec![vec![0.5f32; 100], vec![0.5f32; 100]];

    // Mono path.
    let mono_cfg = ChorusConfig::with_gains(vec![0.5, 0.5])
        .unwrap()
        .with_clip(false);
    let mono_chunks =
        assemble_streaming_chorus(&[voice0.clone(), voice1.clone()], &mono_cfg, &config).unwrap();

    // Stereo path (center pan -- both voices centered).
    let stereo_cfg = ChorusConfig::with_stereo_pan(vec![0.5, 0.5], vec![0.0, 0.0])
        .unwrap()
        .with_clip(false);
    let stereo_chunks = assemble_streaming_chorus(&[voice0, voice1], &stereo_cfg, &config).unwrap();

    assert_eq!(mono_chunks.len(), stereo_chunks.len());

    // For each chunk, stereo PCM length should be exactly 2x mono PCM length.
    // This means the crossfade consumed the same number of sample-pairs
    // (time duration) in both cases.
    for (i, (mc, sc)) in mono_chunks.iter().zip(stereo_chunks.iter()).enumerate() {
        assert_eq!(
            sc.pcm.len(),
            mc.pcm.len() * 2,
            "chunk {i}: stereo len {} != 2 * mono len {}",
            sc.pcm.len(),
            mc.pcm.len(),
        );
        assert_eq!(sc.channels, 2, "chunk {i}: stereo must have channels=2");
        assert_eq!(mc.channels, 1, "chunk {i}: mono must have channels=1");
    }

    // Verify total time is the same.
    let mono_total: f64 = mono_chunks.iter().map(AudioChunk::duration_secs).sum();
    let stereo_total: f64 = stereo_chunks.iter().map(AudioChunk::duration_secs).sum();
    assert!(
        (mono_total - stereo_total).abs() < 1e-6,
        "mono total {mono_total}s != stereo total {stereo_total}s",
    );
}

// ---------------------------------------------------------------------------
// Algorithm boundary: truncated crossfade for short middle chunks
// ---------------------------------------------------------------------------

/// Proof: 3-chunk assembly with a short middle chunk produces truncated crossfade.
///
/// When a non-first, non-last chunk has `raw.len() < 2*cf`, the crossfade loop
/// runs fewer than `cf` iterations (only `emit_len` iterations). This means:
///
/// - Only `min(cf, emit_len)` samples are blended at the chunk boundary
/// - The remaining `cf - emit_len` samples from the previous chunk's tail
///   never appear in ANY output -- they are silently lost
///
/// This test documents the behavior rather than asserting it's wrong -- in
/// production, chunks are always much larger than 2*cf (typical: 5000-50000
/// samples vs cf=240). But for algorithm correctness, this is a truncated
/// crossfade that reduces audio quality at the boundary.
///
/// See: `kokoro_streaming.rs` line 331 (`cf.min(emit_len)`).
#[test]
fn test_assemble_three_chunks_short_middle_truncated_crossfade() {
    let cf = 10;
    let config = KokoroStreamConfig::new(cf).unwrap();

    // 3 chunks: normal, short middle (cf+3=13 samples), normal last.
    // Middle chunk emit_len = 13 - 10 = 3. Only 3 of 10 crossfade samples blended.
    let raw = vec![
        vec![1.0f32; 100],    // chunk 0: constant 1.0
        vec![0.0f32; cf + 3], // chunk 1: constant 0.0, 13 samples -> emit_len=3
        vec![0.5f32; 100],    // chunk 2: constant 0.5
    ];

    let chunks = assemble_streaming_chunks(&raw, &config).unwrap();
    assert_eq!(chunks.len(), 3);

    // Chunk 0 (first, non-last): 100 - 10 = 90 emitted, no crossfade.
    assert_eq!(chunks[0].pcm.len(), 90);

    // Chunk 1 (middle): only 3 samples emitted -- truncated crossfade.
    assert_eq!(chunks[1].pcm.len(), 3);

    // Verify the 3 blended samples use the standard crossfade formula.
    // tail = raw0[90..100] = [1.0; 10], head = raw1[0..] = [0.0; ...].
    let inv = 1.0 / (cf - 1) as f32;
    for j in 0..3 {
        let alpha = j as f32 * inv;
        let expected = 1.0 * (1.0 - alpha) + 0.0 * alpha;
        assert!(
            (chunks[1].pcm[j] - expected).abs() < 1e-6,
            "chunk1 crossfade[{j}]: expected {expected}, got {}",
            chunks[1].pcm[j],
        );
    }

    // Document: samples raw0[93..100] (7 tail samples) are LOST.
    // They don't appear in chunk 1 output (only 3 blended samples)
    // and they don't appear in chunk 2 output (chunk 2 reads from raw1's tail).

    // Chunk 2 (last): full 100 samples. Crossfade uses raw1[3..13] as tail.
    assert_eq!(chunks[2].pcm.len(), 100);

    // Verify chunk 2's crossfade blends raw1's tail (all 0.0) with raw2's head (all 0.5).
    for j in 0..cf {
        let alpha = j as f32 * inv;
        let expected = 0.0 * (1.0 - alpha) + 0.5 * alpha;
        assert!(
            (chunks[2].pcm[j] - expected).abs() < 1e-6,
            "chunk2 crossfade[{j}]: expected {expected}, got {}",
            chunks[2].pcm[j],
        );
    }

    // Total output: 90 + 3 + 100 = 193.
    // Without truncation: 100 + 13 + 100 - 2*10 = 193 (same!).
    // The total sample count formula is preserved -- but the crossfade
    // quality at the chunk0->chunk1 boundary is degraded (3 samples
    // instead of 10 for the blend).
    let total: usize = chunks.iter().map(|c| c.pcm.len()).sum();
    assert_eq!(
        total, 193,
        "total output matches formula sum(L_i) - (N-1)*cf"
    );
}

/// Proof: middle chunk with emit_len=0 produces empty chunk but total output
/// formula still holds. Adjacent chunks' crossfade regions skip over it.
#[test]
fn test_assemble_three_chunks_degenerate_middle_emit_zero() {
    let cf = 10;
    let config = KokoroStreamConfig::new(cf).unwrap();

    // Middle chunk has exactly cf samples -> emit_len = 0.
    let raw = vec![
        vec![1.0f32; 100], // chunk 0
        vec![0.0f32; cf],  // chunk 1: exactly cf -> emit_len = 0
        vec![0.5f32; 100], // chunk 2
    ];

    let chunks = assemble_streaming_chunks(&raw, &config).unwrap();
    assert_eq!(chunks.len(), 3);

    assert_eq!(chunks[0].pcm.len(), 90);
    assert_eq!(
        chunks[1].pcm.len(),
        0,
        "degenerate middle chunk emits 0 samples"
    );
    assert_eq!(chunks[2].pcm.len(), 100);

    // Total: 90 + 0 + 100 = 190 = 100 + 10 + 100 - 2*10 = 190. Formula holds.
    let total: usize = chunks.iter().map(|c| c.pcm.len()).sum();
    assert_eq!(total, 190);

    // Chunk 2's crossfade reads from raw1[0..10] -- the entire middle chunk.
    let inv = 1.0 / (cf - 1) as f32;
    for j in 0..cf {
        let alpha = j as f32 * inv;
        let expected = 0.0 * (1.0 - alpha) + 0.5 * alpha;
        assert!(
            (chunks[2].pcm[j] - expected).abs() < 1e-6,
            "chunk2 crossfade[{j}]: expected {expected}, got {}",
            chunks[2].pcm[j],
        );
    }
}

/// Stereo center-panned output matches mono output values.
///
/// When all voices are panned to center (0.0), the equal-power pan law
/// gives L = cos(pi/4) * gain ~ 0.707 * gain, R = sin(pi/4) * gain ~ 0.707 * gain.
/// Both channels should have the same value for center-panned voices.
#[test]
fn test_stereo_center_pan_equal_channels() {
    let cf = 5;
    let config = KokoroStreamConfig::new(cf).unwrap();

    let voice0 = vec![vec![1.0f32; 50]];
    let stereo_cfg = ChorusConfig::with_stereo_pan(vec![1.0], vec![0.0])
        .unwrap()
        .with_clip(false);
    let chunks = assemble_streaming_chorus(&[voice0], &stereo_cfg, &config).unwrap();

    assert_eq!(chunks[0].channels, 2);
    // Center pan: L and R should be equal.
    for i in 0..50 {
        let l = chunks[0].pcm[i * 2];
        let r = chunks[0].pcm[i * 2 + 1];
        assert!(
            (l - r).abs() < 1e-6,
            "sample {i}: L={l} != R={r} for center pan",
        );
    }
}
