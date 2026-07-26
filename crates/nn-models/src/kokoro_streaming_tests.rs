// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the streaming synthesis API contract.
//!
//! These tests validate the crossfade logic and AudioChunk assembly without
//! requiring model weights or GPU — they operate on synthetic PCM data.
//!
//! Chorus-specific tests (multi-voice mixing, stereo, truncated crossfade)
//! are in `kokoro_streaming_chorus_tests.rs`.

use super::*;

// ---------------------------------------------------------------------------
// KokoroStreamConfig
// ---------------------------------------------------------------------------

#[test]
fn test_stream_config_default() {
    let config = KokoroStreamConfig::default();
    assert_eq!(config.crossfade_samples, 960);
    assert_eq!(config.crossfade_window, CrossfadeWindow::SqrtHann);
    assert!(config.validate().is_ok());
}

#[test]
fn test_stream_config_new_valid() {
    let config = KokoroStreamConfig::new(960).unwrap();
    assert_eq!(config.crossfade_samples, 960);
    assert_eq!(config.crossfade_window, CrossfadeWindow::SqrtHann);
    // 960 / 24000 = 0.04s
    assert!((config.crossfade_duration_secs() - 0.04).abs() < 1e-10);
}

#[test]
fn test_stream_config_new_auto_selects_linear_for_short_overlap() {
    let config = KokoroStreamConfig::new(480).unwrap();
    assert_eq!(config.crossfade_samples, 480);
    assert_eq!(config.crossfade_window, CrossfadeWindow::Linear);
}

#[test]
fn test_stream_config_new_auto_selects_sqrt_hann_for_long_overlap() {
    let config = KokoroStreamConfig::new(960).unwrap();
    assert_eq!(config.crossfade_window, CrossfadeWindow::SqrtHann);

    let config = KokoroStreamConfig::new(1920).unwrap();
    assert_eq!(config.crossfade_window, CrossfadeWindow::SqrtHann);
}

#[test]
fn test_stream_config_new_threshold_boundary() {
    // Just below threshold: Linear
    let config = KokoroStreamConfig::new(959).unwrap();
    assert_eq!(config.crossfade_window, CrossfadeWindow::Linear);

    // At threshold: SqrtHann
    let config = KokoroStreamConfig::new(960).unwrap();
    assert_eq!(config.crossfade_window, CrossfadeWindow::SqrtHann);
}

#[test]
fn test_stream_config_zero_crossfade_rejected() {
    let result = KokoroStreamConfig::new(0);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("crossfade_samples"));
}

// ---------------------------------------------------------------------------
// AudioChunk
// ---------------------------------------------------------------------------

#[test]
fn test_audio_chunk_duration() {
    let chunk = AudioChunk {
        pcm: vec![0.0; 24000], // 1 second
        channels: 1,
        sample_offset: 0,
        chunk_index: 0,
        total_chunks: 1,
        is_final: true,
    };
    assert!((chunk.duration_secs() - 1.0).abs() < 1e-10);
    assert_eq!(chunk.len(), 24000);
    assert!(!chunk.is_empty());
}

#[test]
fn test_audio_chunk_empty() {
    let chunk = AudioChunk {
        pcm: vec![],
        channels: 1,
        sample_offset: 0,
        chunk_index: 0,
        total_chunks: 1,
        is_final: true,
    };
    assert!(chunk.is_empty());
    assert_eq!(chunk.duration_secs(), 0.0);
}

// ---------------------------------------------------------------------------
// crossfade_chunks
// ---------------------------------------------------------------------------

#[test]
fn test_crossfade_basic_linear() {
    // Two constant-value chunks: prev=1.0, next=0.0
    // After crossfade, the overlap region should ramp from 1.0 to 0.0.
    let prev = vec![1.0f32; 100];
    let mut next = vec![0.0f32; 100];
    let cf = 10;

    crossfade_chunks(&prev, &mut next, cf).unwrap();

    // Check crossfade region: should linearly interpolate from 1.0 to 0.0.
    for i in 0..cf {
        let alpha = i as f32 / (cf - 1) as f32;
        let expected = 1.0 * (1.0 - alpha) + 0.0 * alpha;
        assert!(
            (next[i] - expected).abs() < 1e-6,
            "crossfade[{i}]: expected {expected}, got {}",
            next[i],
        );
    }
    // After crossfade region, values unchanged (0.0).
    for i in cf..next.len() {
        assert_eq!(next[i], 0.0, "post-crossfade[{i}] should be unchanged");
    }
}

#[test]
fn test_crossfade_identical_chunks() {
    // If both chunks have the same value, crossfade should preserve it.
    let prev = vec![0.5f32; 50];
    let mut next = vec![0.5f32; 50];

    crossfade_chunks(&prev, &mut next, 20).unwrap();

    for (i, &v) in next.iter().enumerate() {
        assert!(
            (v - 0.5).abs() < 1e-6,
            "crossfade[{i}]: expected 0.5, got {v}",
        );
    }
}

#[test]
fn test_crossfade_single_sample() {
    let prev = vec![1.0f32; 10];
    let mut next = vec![0.0f32; 10];

    crossfade_chunks(&prev, &mut next, 1).unwrap();

    // Single sample crossfade: average of boundary samples.
    assert!((next[0] - 0.5).abs() < 1e-6);
    // Rest unchanged.
    assert_eq!(next[1], 0.0);
}

#[test]
fn test_crossfade_prev_too_short() {
    let prev = vec![1.0f32; 5];
    let mut next = vec![0.0f32; 20];

    let result = crossfade_chunks(&prev, &mut next, 10);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("too short"));
}

#[test]
fn test_crossfade_next_too_short() {
    let prev = vec![1.0f32; 20];
    let mut next = vec![0.0f32; 5];

    let result = crossfade_chunks(&prev, &mut next, 10);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("too short"));
}

#[test]
fn test_crossfade_zero_is_noop() {
    let prev = vec![1.0f32; 10];
    let mut next = vec![0.0f32; 10];
    let original = next.clone();

    crossfade_chunks(&prev, &mut next, 0).unwrap();
    assert_eq!(next, original);
}

// ---------------------------------------------------------------------------
// assemble_streaming_chunks
// ---------------------------------------------------------------------------

#[test]
fn test_assemble_empty() {
    let config = KokoroStreamConfig::default();
    let chunks = assemble_streaming_chunks(&[], &config).unwrap();
    assert!(chunks.is_empty());
}

#[test]
fn test_assemble_single_chunk() {
    let config = KokoroStreamConfig::default();
    let raw = vec![vec![0.5f32; 1000]];
    let chunks = assemble_streaming_chunks(&raw, &config).unwrap();

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].chunk_index, 0);
    assert_eq!(chunks[0].total_chunks, 1);
    assert!(chunks[0].is_final);
    assert_eq!(chunks[0].sample_offset, 0);
    assert_eq!(chunks[0].pcm.len(), 1000);
}

#[test]
fn test_assemble_owned_single_chunk_reuses_allocation() {
    let config = KokoroStreamConfig::default();
    let raw0 = vec![0.5f32; 1000];
    let raw0_ptr = raw0.as_ptr();

    let chunks = assemble_streaming_chunks_owned(vec![raw0], &config).unwrap();

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].pcm.as_ptr(), raw0_ptr);
    assert_eq!(chunks[0].pcm.len(), 1000);
    assert_eq!(chunks[0].sample_offset, 0);
    assert!(chunks[0].is_final);
}

#[test]
fn test_assemble_owned_multi_chunk_reuses_allocations_across_boundaries() {
    let cf = 4;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let raw0 = vec![1.0f32; 8];
    let raw1 = vec![0.0f32; 8];
    let raw2 = vec![0.5f32; 6];
    let raw0_ptr = raw0.as_ptr();
    let raw1_ptr = raw1.as_ptr();
    let raw2_ptr = raw2.as_ptr();

    let chunks = assemble_streaming_chunks_owned(vec![raw0, raw1, raw2], &config).unwrap();

    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].pcm.as_ptr(), raw0_ptr);
    assert_eq!(chunks[1].pcm.as_ptr(), raw1_ptr);
    assert_eq!(chunks[2].pcm.as_ptr(), raw2_ptr);
    assert_eq!(chunks[0].pcm.len(), 4);
    assert_eq!(chunks[1].pcm.len(), 4);
    assert_eq!(chunks[2].pcm.len(), 6);
    assert_eq!(chunks[0].sample_offset, 0);
    assert_eq!(chunks[1].sample_offset, 4);
    assert_eq!(chunks[2].sample_offset, 8);

    for i in 0..cf {
        let alpha = i as f32 / (cf - 1) as f32;
        let expected = 1.0 * (1.0 - alpha);
        assert!(
            (chunks[1].pcm[i] - expected).abs() < 1e-5,
            "chunk1 crossfade[{i}]: expected {expected}, got {}",
            chunks[1].pcm[i],
        );
    }
    for i in 0..cf {
        let alpha = i as f32 / (cf - 1) as f32;
        let expected = 0.5 * alpha;
        assert!(
            (chunks[2].pcm[i] - expected).abs() < 1e-5,
            "chunk2 crossfade[{i}]: expected {expected}, got {}",
            chunks[2].pcm[i],
        );
    }
    assert_eq!(&chunks[2].pcm[cf..], &[0.5, 0.5]);
}

#[test]
fn test_assemble_two_chunks_crossfade() {
    let cf = 10;
    let config = KokoroStreamConfig::new(cf).unwrap();

    // Two constant chunks: chunk0=1.0, chunk1=0.0
    let raw = vec![vec![1.0f32; 100], vec![0.0f32; 100]];
    let chunks = assemble_streaming_chunks(&raw, &config).unwrap();

    assert_eq!(chunks.len(), 2);

    // Chunk 0: 100 - 10 = 90 samples (tail reserved for crossfade).
    assert_eq!(chunks[0].pcm.len(), 90);
    assert_eq!(chunks[0].sample_offset, 0);
    assert_eq!(chunks[0].chunk_index, 0);
    assert!(!chunks[0].is_final);

    // Chunk 1: full 100 samples (with crossfade applied to first 10).
    assert_eq!(chunks[1].pcm.len(), 100);
    assert_eq!(chunks[1].sample_offset, 90);
    assert_eq!(chunks[1].chunk_index, 1);
    assert!(chunks[1].is_final);

    // Verify crossfade region in chunk 1: ramps from 1.0 to 0.0.
    for i in 0..cf {
        let alpha = i as f32 / (cf - 1) as f32;
        let expected = 1.0 * (1.0 - alpha);
        assert!(
            (chunks[1].pcm[i] - expected).abs() < 1e-5,
            "chunk1 crossfade[{i}]: expected {expected}, got {}",
            chunks[1].pcm[i],
        );
    }
    // Post-crossfade region unchanged.
    for i in cf..100 {
        assert_eq!(chunks[1].pcm[i], 0.0);
    }
}

#[test]
fn test_assemble_three_chunks_offsets() {
    let cf = 5;
    let config = KokoroStreamConfig::new(cf).unwrap();

    let raw = vec![vec![0.0f32; 50], vec![0.0f32; 60], vec![0.0f32; 40]];
    let chunks = assemble_streaming_chunks(&raw, &config).unwrap();

    assert_eq!(chunks.len(), 3);

    // Chunk 0: 50 - 5 = 45 emitted.
    assert_eq!(chunks[0].pcm.len(), 45);
    assert_eq!(chunks[0].sample_offset, 0);
    assert!(!chunks[0].is_final);

    // Chunk 1: 60 - 5 = 55 emitted.
    assert_eq!(chunks[1].pcm.len(), 55);
    assert_eq!(chunks[1].sample_offset, 45);
    assert!(!chunks[1].is_final);

    // Chunk 2 (last): full 40 emitted.
    assert_eq!(chunks[2].pcm.len(), 40);
    assert_eq!(chunks[2].sample_offset, 100); // 45 + 55
    assert!(chunks[2].is_final);

    // Total samples: 45 + 55 + 40 = 140 (vs raw: 50+60+40=150, minus 2*5=10 overlap).
    let total: usize = chunks.iter().map(|c| c.pcm.len()).sum();
    assert_eq!(total, 140);
}

#[test]
fn test_assemble_chunk_too_short_for_crossfade() {
    let config = KokoroStreamConfig::new(100).unwrap();
    let raw = vec![vec![0.0f32; 50], vec![0.0f32; 50]];

    let result = assemble_streaming_chunks(&raw, &config);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// concatenate_chunks
// ---------------------------------------------------------------------------

#[test]
fn test_concatenate_chunks() {
    let chunks = vec![
        AudioChunk {
            pcm: vec![1.0, 2.0, 3.0],
            channels: 1,
            sample_offset: 0,
            chunk_index: 0,
            total_chunks: 2,
            is_final: false,
        },
        AudioChunk {
            pcm: vec![4.0, 5.0],
            channels: 1,
            sample_offset: 3,
            chunk_index: 1,
            total_chunks: 2,
            is_final: true,
        },
    ];

    let pcm = concatenate_chunks(&chunks);
    assert_eq!(pcm, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_concatenate_empty() {
    let pcm = concatenate_chunks(&[]);
    assert!(pcm.is_empty());
}

// ---------------------------------------------------------------------------
// Crossfade continuity property: no click at boundary
// ---------------------------------------------------------------------------

#[test]
fn test_crossfade_no_click_at_boundary() {
    // Simulate a boundary between a rising ramp and a falling ramp.
    // Without crossfade, there's a large discontinuity at the boundary.
    // With crossfade, the transition should be smooth.
    let n = 200;
    let cf = 40;
    let config = KokoroStreamConfig::new(cf).unwrap();

    // Chunk 0: ramp from 0.0 to 1.0
    let raw0: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
    // Chunk 1: ramp from -0.5 to 0.5
    let raw1: Vec<f32> = (0..n).map(|i| -0.5 + i as f32 / (n - 1) as f32).collect();

    let chunks = assemble_streaming_chunks(&[raw0, raw1], &config).unwrap();
    let pcm = concatenate_chunks(&chunks);

    // Check that the maximum sample-to-sample jump is small.
    let mut max_diff: f32 = 0.0;
    for i in 1..pcm.len() {
        let diff = (pcm[i] - pcm[i - 1]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    // Without crossfade, boundary jump would be ~1.5 (1.0 to -0.5).
    // With crossfade over 40 samples, max step should be much smaller.
    assert!(
        max_diff < 0.1,
        "max sample-to-sample diff {max_diff} exceeds click threshold 0.1",
    );
}

// ---------------------------------------------------------------------------
// assemble_streaming_chorus (basic tests; detailed in chorus_tests)
// ---------------------------------------------------------------------------

#[test]
fn test_chorus_single_voice_equals_streaming() {
    let cf = 10;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let chorus_config = ChorusConfig::with_gains(vec![1.0])
        .unwrap()
        .with_clip(false);

    let raw = vec![vec![1.0f32; 100], vec![0.5f32; 80]];

    // Single voice with gain 1.0 should match plain streaming assembly.
    let chorus_chunks =
        assemble_streaming_chorus(std::slice::from_ref(&raw), &chorus_config, &config).unwrap();
    let stream_chunks = assemble_streaming_chunks(&raw, &config).unwrap();

    assert_eq!(chorus_chunks.len(), stream_chunks.len());
    for (c, s) in chorus_chunks.iter().zip(stream_chunks.iter()) {
        assert_eq!(c.pcm.len(), s.pcm.len());
        for (i, (&cv, &sv)) in c.pcm.iter().zip(s.pcm.iter()).enumerate() {
            assert!(
                (cv - sv).abs() < 1e-6,
                "mismatch at chunk {} sample {i}: chorus={cv}, stream={sv}",
                c.chunk_index,
            );
        }
    }
}

#[test]
fn test_chorus_two_voices_mixed() {
    let cf = 5;
    let config = KokoroStreamConfig::new(cf).unwrap();

    // Voice 0: constant 1.0, Voice 1: constant -0.5
    // Gains: [0.6, 0.4] -> mixed = 1.0*0.6 + (-0.5)*0.4 = 0.4
    let voice0 = vec![vec![1.0f32; 50]];
    let voice1 = vec![vec![-0.5f32; 50]];

    let chorus_config = ChorusConfig::with_gains(vec![0.6, 0.4])
        .unwrap()
        .with_clip(false);
    let chunks = assemble_streaming_chorus(&[voice0, voice1], &chorus_config, &config).unwrap();

    assert_eq!(chunks.len(), 1);
    let expected = 1.0 * 0.6 + (-0.5) * 0.4;
    for &v in &chunks[0].pcm {
        assert!((v - expected).abs() < 1e-5, "expected ~{expected}, got {v}");
    }
}

#[test]
fn test_chorus_clipping() {
    let cf = 5;
    let config = KokoroStreamConfig::new(cf).unwrap();

    // Two voices at full gain that sum > 1.0
    let voice0 = vec![vec![0.8f32; 50]];
    let voice1 = vec![vec![0.8f32; 50]];

    // Without clip
    let no_clip_cfg = ChorusConfig::with_gains(vec![1.0, 1.0])
        .unwrap()
        .with_clip(false);
    let no_clip =
        assemble_streaming_chorus(&[voice0.clone(), voice1.clone()], &no_clip_cfg, &config)
            .unwrap();
    assert!(no_clip[0].pcm.iter().any(|&v| v > 1.0));

    // With clip
    let clip_cfg = ChorusConfig::with_gains(vec![1.0, 1.0])
        .unwrap()
        .with_clip(true);
    let clipped = assemble_streaming_chorus(&[voice0, voice1], &clip_cfg, &config).unwrap();
    assert!(clipped[0].pcm.iter().all(|&v| v <= 1.0));
}

#[test]
fn test_chorus_mismatched_voices_rejected() {
    let config = KokoroStreamConfig::default();
    let voice0 = vec![vec![1.0f32; 50]];

    // 1 voice data but ChorusConfig expects 2 voices
    let chorus_config = ChorusConfig::with_gains(vec![0.5, 0.5])
        .unwrap()
        .with_clip(false);
    let result = assemble_streaming_chorus(&[voice0], &chorus_config, &config);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("n_voices"));
}

#[test]
fn test_chorus_mismatched_chunk_count_rejected() {
    let config = KokoroStreamConfig::default();
    let voice0 = vec![vec![1.0f32; 50], vec![1.0f32; 50]]; // 2 chunks
    let voice1 = vec![vec![1.0f32; 50]]; // 1 chunk

    let chorus_config = ChorusConfig::with_gains(vec![0.5, 0.5])
        .unwrap()
        .with_clip(false);
    let result = assemble_streaming_chorus(&[voice0, voice1], &chorus_config, &config);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("chunks"));
}

#[test]
fn test_chorus_empty_chunks() {
    let config = KokoroStreamConfig::default();
    // 1 voice with 0 chunks -> empty output.
    let per_voice: Vec<Vec<Vec<f32>>> = vec![vec![]];
    let chorus_config = ChorusConfig::with_gains(vec![1.0]).unwrap();
    let result = assemble_streaming_chorus(&per_voice, &chorus_config, &config).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_chorus_empty_per_voice_returns_empty() {
    let config = KokoroStreamConfig::default();
    let chunks: Vec<Vec<Vec<f32>>> = vec![];
    // Empty per_voice_chunks short-circuits to Ok(empty) via the
    // `per_voice_chunks.is_empty()` early return, before the n_voices check.
    let chorus_config = ChorusConfig::with_gains(vec![1.0]).unwrap();
    let result = assemble_streaming_chorus(&chunks, &chorus_config, &config);
    assert!(result.unwrap().is_empty());
}

#[test]
fn test_chorus_two_chunks_crossfaded() {
    let cf = 10;
    let config = KokoroStreamConfig::new(cf).unwrap();

    // 2 voices, 2 chunks each. Check that crossfade and mixing both work.
    let voice0 = vec![vec![1.0f32; 100], vec![0.0f32; 100]];
    let voice1 = vec![vec![0.0f32; 100], vec![1.0f32; 100]];

    let chorus_config = ChorusConfig::with_gains(vec![0.5, 0.5])
        .unwrap()
        .with_clip(false);
    let chunks = assemble_streaming_chorus(&[voice0, voice1], &chorus_config, &config).unwrap();

    assert_eq!(chunks.len(), 2);
    // Chunk 0: mixed = 1.0*0.5 + 0.0*0.5 = 0.5, emitted 90 samples (100-10).
    assert_eq!(chunks[0].pcm.len(), 90);
    // Chunk 1: mixed = 0.0*0.5 + 1.0*0.5 = 0.5 (+ crossfade at boundary).
    assert_eq!(chunks[1].pcm.len(), 100);
    // Both chunks should have values around 0.5 (constant voices).
    for &v in &chunks[0].pcm {
        assert!((v - 0.5).abs() < 1e-5, "chunk0: expected ~0.5, got {v}");
    }
    // Chunk 1 post-crossfade region should also be 0.5.
    for &v in &chunks[1].pcm[cf..] {
        assert!(
            (v - 0.5).abs() < 1e-5,
            "chunk1 post-cf: expected ~0.5, got {v}"
        );
    }
}
