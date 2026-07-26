// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for [`StreamingAssembler`] crossfade assembly.
//!
//! Tests the assembler from the nn-metal consumer perspective, covering
//! mono/multi-channel assembly, crossfade overlap correctness, edge cases,
//! session reset, and sample rate preservation.
//!
//! All tests are CPU-only -- no GPU or Kokoro weights required.

use nn_models::kokoro_streaming::{
    assemble_streaming_chunks, AudioChunk, KokoroStreamConfig, StreamingAssembler,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a ramp signal of `len` samples starting at `start`.
fn ramp(start: f32, len: usize) -> Vec<f32> {
    (0..len).map(|i| start + i as f32 * 0.001).collect()
}

/// Create a constant signal of `len` samples at `val`.
fn constant(val: f32, len: usize) -> Vec<f32> {
    vec![val; len]
}

/// Verify that the crossfade region between two constant-valued chunks
/// produces a linear blend from `val_a` to `val_b` over `cf` samples.
fn assert_linear_crossfade(pcm: &[f32], val_a: f32, val_b: f32, cf: usize) {
    assert!(
        pcm.len() >= cf,
        "pcm too short ({}) for crossfade check (cf={})",
        pcm.len(),
        cf,
    );
    let inv = if cf > 1 { 1.0 / (cf - 1) as f32 } else { 1.0 };
    for j in 0..cf {
        let alpha = j as f32 * inv;
        let expected = val_a * (1.0 - alpha) + val_b * alpha;
        assert!(
            (pcm[j] - expected).abs() < 1e-5,
            "crossfade[{j}]: expected {expected}, got {}",
            pcm[j],
        );
    }
}

// ---------------------------------------------------------------------------
// Basic chunk assembly with crossfade
// ---------------------------------------------------------------------------

#[test]
fn test_two_chunks_crossfade_values() {
    let cf = 10;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let mut asm = StreamingAssembler::new(config, 2).unwrap();

    let chunk0 = asm.push_raw(constant(1.0, 100)).unwrap();
    let chunk1 = asm.push_raw(constant(0.0, 100)).unwrap();

    // First chunk emits 90 samples (100 - cf tail held back).
    assert_eq!(chunk0.pcm.len(), 90);
    // All of chunk0's emitted samples should be 1.0 (no crossfade on first).
    for &v in &chunk0.pcm {
        assert!((v - 1.0).abs() < 1e-6, "chunk0 sample should be 1.0, got {v}");
    }

    // Second chunk gets all 100 samples (it's the last chunk).
    assert_eq!(chunk1.pcm.len(), 100);
    // First cf samples should be linearly blended from 1.0 -> 0.0.
    assert_linear_crossfade(&chunk1.pcm, 1.0, 0.0, cf);
    // Remaining samples should be 0.0.
    for &v in &chunk1.pcm[cf..] {
        assert!((v - 0.0).abs() < 1e-6, "chunk1 post-crossfade sample should be 0.0, got {v}");
    }
}

#[test]
fn test_three_chunks_crossfade_chain() {
    let cf = 8;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let mut asm = StreamingAssembler::new(config, 3).unwrap();

    let c0 = asm.push_raw(constant(1.0, 80)).unwrap();
    let c1 = asm.push_raw(constant(0.5, 80)).unwrap();
    let c2 = asm.push_raw(constant(0.0, 80)).unwrap();

    // c0: 80 - 8 = 72 emitted, no crossfade.
    assert_eq!(c0.pcm.len(), 72);
    assert_eq!(c0.chunk_index, 0);
    assert!(!c0.is_final);

    // c1: 80 - 8 = 72 emitted, crossfade from 1.0 -> 0.5.
    assert_eq!(c1.pcm.len(), 72);
    assert_eq!(c1.chunk_index, 1);
    assert!(!c1.is_final);
    assert_linear_crossfade(&c1.pcm, 1.0, 0.5, cf);

    // c2: 80 emitted (last), crossfade from 0.5 -> 0.0.
    assert_eq!(c2.pcm.len(), 80);
    assert_eq!(c2.chunk_index, 2);
    assert!(c2.is_final);
    assert_linear_crossfade(&c2.pcm, 0.5, 0.0, cf);
}

// ---------------------------------------------------------------------------
// Multi-channel assembly (stereo, chorus)
// ---------------------------------------------------------------------------

#[test]
fn test_stereo_two_channels_crossfade() {
    let cf = 4; // 4 time-domain samples => 8 floats interleaved
    let config = KokoroStreamConfig::new(cf).unwrap();
    let channels = 2;
    let mut asm = StreamingAssembler::new_with_channels(config, 2, channels).unwrap();

    // Interleaved stereo: [L, R, L, R, ...]. 40 floats = 20 time-domain samples.
    let raw0: Vec<f32> = (0..40).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let raw1: Vec<f32> = (0..40).map(|i| if i % 2 == 0 { 0.0 } else { 0.0 }).collect();

    let c0 = asm.push_raw(raw0).unwrap();
    let c1 = asm.push_raw(raw1).unwrap();

    // Effective crossfade floats = cf * channels = 8.
    // c0 emits 40 - 8 = 32 floats.
    assert_eq!(c0.pcm.len(), 32);
    assert_eq!(c0.channels, 2);

    // c1 emits all 40 floats (last chunk).
    assert_eq!(c1.pcm.len(), 40);
    assert_eq!(c1.channels, 2);
    assert!(c1.is_final);

    // Verify crossfade was applied to the first 8 floats of c1.
    // The interleaved crossfade blends L-with-L and R-with-R via linear alpha.
    let effective_cf = cf * channels;
    let inv = 1.0 / (effective_cf - 1) as f32;
    for j in 0..effective_cf {
        let alpha = j as f32 * inv;
        // raw0 tail: last 8 floats = alternating [1.0, -1.0, ...]
        let tail_val = if (40 - effective_cf + j) % 2 == 0 { 1.0 } else { -1.0 };
        let head_val = 0.0; // raw1 is all zeros
        let expected = tail_val * (1.0 - alpha) + head_val * alpha;
        assert!(
            (c1.pcm[j] - expected).abs() < 1e-5,
            "stereo crossfade[{j}]: expected {expected}, got {}",
            c1.pcm[j],
        );
    }
}

#[test]
fn test_four_channel_assembly() {
    let cf = 2;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let channels = 4;
    let mut asm = StreamingAssembler::new_with_channels(config, 2, channels).unwrap();

    // 4 channels, cf=2 => effective 8 floats crossfade.
    let raw0 = constant(1.0, 40);
    let raw1 = constant(0.0, 40);

    let c0 = asm.push_raw(raw0).unwrap();
    let c1 = asm.push_raw(raw1).unwrap();

    assert_eq!(c0.pcm.len(), 40 - 8); // 32
    assert_eq!(c1.pcm.len(), 40);
    assert_eq!(c0.channels, 4);
    assert_eq!(c1.channels, 4);
}

#[test]
fn test_zero_channels_rejected() {
    let config = KokoroStreamConfig::new(10).unwrap();
    let result = StreamingAssembler::new_with_channels(config, 1, 0);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("channels"), "expected channels error, got: {msg}");
}

// ---------------------------------------------------------------------------
// Crossfade overlap handling
// ---------------------------------------------------------------------------

#[test]
fn test_crossfade_with_cf_equal_to_one() {
    let config = KokoroStreamConfig::new(1).unwrap();
    let mut asm = StreamingAssembler::new(config, 2).unwrap();

    let c0 = asm.push_raw(constant(1.0, 50)).unwrap();
    let c1 = asm.push_raw(constant(0.0, 50)).unwrap();

    // With cf=1, only 1 sample is held back from c0.
    assert_eq!(c0.pcm.len(), 49);
    assert_eq!(c1.pcm.len(), 50);

    // When cf=1, alpha = 0/(1-1) = 0/0. For cf=1, the single blend sample
    // has alpha=0 so it equals the tail value entirely... but let's just verify
    // the length is correct and no panic occurs.
    assert!(c1.is_final);
}

#[test]
fn test_crossfade_region_is_convex_combination() {
    // Verify: for all crossfade samples, output is between min(prev, next) and max(prev, next).
    let cf = 20;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let mut asm = StreamingAssembler::new(config, 2).unwrap();

    let _ = asm.push_raw(constant(0.8, 200)).unwrap();
    let c1 = asm.push_raw(constant(0.2, 200)).unwrap();

    for j in 0..cf {
        let val = c1.pcm[j];
        assert!(
            (0.2 - 1e-6..=0.8 + 1e-6).contains(&val),
            "crossfade[{j}]={val} outside [0.2, 0.8]",
        );
    }
}

#[test]
fn test_crossfade_with_ramp_signals() {
    let cf = 10;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let mut asm = StreamingAssembler::new(config, 2).unwrap();

    let raw0 = ramp(0.0, 100);
    let raw1 = ramp(1.0, 100);

    let _c0 = asm.push_raw(raw0.clone()).unwrap();
    let c1 = asm.push_raw(raw1.clone()).unwrap();

    // Verify crossfade region blends correctly.
    let inv = 1.0 / (cf - 1) as f32;
    let tail_start = raw0.len() - cf;
    for j in 0..cf {
        let alpha = j as f32 * inv;
        let tail_val = raw0[tail_start + j];
        let head_val = raw1[j];
        let expected = tail_val * (1.0 - alpha) + head_val * alpha;
        assert!(
            (c1.pcm[j] - expected).abs() < 1e-5,
            "ramp crossfade[{j}]: expected {expected}, got {}",
            c1.pcm[j],
        );
    }
}

// ---------------------------------------------------------------------------
// Empty chunk handling / error cases
// ---------------------------------------------------------------------------

#[test]
fn test_zero_total_chunks_rejected() {
    let config = KokoroStreamConfig::new(10).unwrap();
    let result = StreamingAssembler::new(config, 0);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("total_chunks"), "expected total_chunks error, got: {msg}");
}

#[test]
fn test_push_after_complete_errors() {
    let config = KokoroStreamConfig::new(5).unwrap();
    let mut asm = StreamingAssembler::new(config, 1).unwrap();
    let _ = asm.push_raw(constant(1.0, 50)).unwrap();
    assert!(asm.is_complete());

    let result = asm.push_raw(constant(1.0, 50));
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("already pushed"), "expected 'already pushed' error, got: {msg}");
}

#[test]
fn test_empty_raw_pcm_on_non_last_chunk_errors() {
    let cf = 10;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let mut asm = StreamingAssembler::new(config, 2).unwrap();

    // First (non-last) chunk with 0 samples should error.
    let result = asm.push_raw(Vec::new());
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("too short"), "expected 'too short' error, got: {msg}");
}

#[test]
fn test_empty_raw_pcm_on_single_chunk_ok() {
    let config = KokoroStreamConfig::new(10).unwrap();
    let mut asm = StreamingAssembler::new(config, 1).unwrap();

    // Single chunk with 0 samples is allowed (it's both first and last).
    let chunk = asm.push_raw(Vec::new()).unwrap();
    assert!(chunk.is_final);
    assert!(chunk.pcm.is_empty());
    assert!(asm.is_complete());
}

#[test]
fn test_crossfade_samples_zero_rejected_at_config() {
    let result = KokoroStreamConfig::new(0);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("crossfade_samples"),
        "expected crossfade_samples error, got: {msg}",
    );
}

// ---------------------------------------------------------------------------
// Single chunk (no crossfade needed)
// ---------------------------------------------------------------------------

#[test]
fn test_single_chunk_no_crossfade() {
    let config = KokoroStreamConfig::new(100).unwrap();
    let mut asm = StreamingAssembler::new(config, 1).unwrap();

    let data = ramp(0.0, 500);
    let chunk = asm.push_raw(data.clone()).unwrap();

    assert!(chunk.is_final);
    assert_eq!(chunk.chunk_index, 0);
    assert_eq!(chunk.total_chunks, 1);
    assert_eq!(chunk.sample_offset, 0);
    assert_eq!(chunk.channels, 1);
    // Single chunk: all samples emitted, no tail held back.
    assert_eq!(chunk.pcm.len(), 500);
    // Data should be unchanged.
    for (i, (&got, &exp)) in chunk.pcm.iter().zip(data.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "single chunk sample[{i}]: expected {exp}, got {got}",
        );
    }
}

#[test]
fn test_single_chunk_shorter_than_crossfade_ok() {
    let cf = 500;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let mut asm = StreamingAssembler::new(config, 1).unwrap();

    // Single chunk can be shorter than cf -- no crossfade needed.
    let chunk = asm.push_raw(constant(0.42, 10)).unwrap();
    assert!(chunk.is_final);
    assert_eq!(chunk.pcm.len(), 10);
}

// ---------------------------------------------------------------------------
// Chunk size mismatch edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_first_chunk_shorter_than_crossfade_two_chunks_errors() {
    let cf = 50;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let mut asm = StreamingAssembler::new(config, 2).unwrap();

    // First (non-last) chunk shorter than cf should error.
    let result = asm.push_raw(constant(1.0, 10));
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("too short"), "expected 'too short' error, got: {msg}");
}

#[test]
fn test_second_chunk_exactly_crossfade_length() {
    let cf = 20;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let mut asm = StreamingAssembler::new(config, 2).unwrap();

    let _ = asm.push_raw(constant(1.0, 100)).unwrap();
    // Second (last) chunk with exactly cf samples should work.
    let c1 = asm.push_raw(constant(0.0, cf)).unwrap();
    assert!(c1.is_final);
    assert_eq!(c1.pcm.len(), cf);
    assert!(asm.is_complete());
}

#[test]
fn test_varied_chunk_sizes() {
    let cf = 5;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let mut asm = StreamingAssembler::new(config, 4).unwrap();

    let c0 = asm.push_raw(constant(1.0, 200)).unwrap();
    let c1 = asm.push_raw(constant(0.7, 50)).unwrap();
    let c2 = asm.push_raw(constant(0.3, 1000)).unwrap();
    let c3 = asm.push_raw(constant(0.0, 10)).unwrap();

    assert_eq!(c0.pcm.len(), 195); // 200 - 5
    assert_eq!(c1.pcm.len(), 45);  // 50 - 5
    assert_eq!(c2.pcm.len(), 995); // 1000 - 5
    assert_eq!(c3.pcm.len(), 10);  // last chunk, no tail held back
    assert!(c3.is_final);
    assert!(asm.is_complete());
}

#[test]
fn test_chunk_exactly_at_crossfade_boundary_non_last() {
    let cf = 50;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let mut asm = StreamingAssembler::new(config, 3).unwrap();

    // First chunk exactly cf samples (non-last) -- emits 0 samples + saves tail.
    let c0 = asm.push_raw(constant(1.0, cf)).unwrap();
    assert_eq!(c0.pcm.len(), 0); // All held back as tail.
    assert!(!c0.is_final);

    // Middle chunk also exactly cf (non-last) -- crossfade uses full chunk.
    let c1 = asm.push_raw(constant(0.5, cf)).unwrap();
    assert_eq!(c1.pcm.len(), 0); // All crossfaded then held back.
    assert!(!c1.is_final);

    // Last chunk.
    let c2 = asm.push_raw(constant(0.0, cf)).unwrap();
    assert_eq!(c2.pcm.len(), cf); // Last chunk emits everything.
    assert!(c2.is_final);
    assert!(asm.is_complete());
}

// ---------------------------------------------------------------------------
// Assembly state reset between sessions
// ---------------------------------------------------------------------------

#[test]
fn test_reset_clears_state() {
    let cf = 10;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let mut asm = StreamingAssembler::new(config, 2).unwrap();

    let _ = asm.push_raw(constant(1.0, 100)).unwrap();
    let _ = asm.push_raw(constant(0.0, 100)).unwrap();
    assert!(asm.is_complete());
    assert_eq!(asm.remaining(), 0);

    asm.reset();

    assert!(!asm.is_complete());
    assert_eq!(asm.remaining(), 2);
    assert_eq!(asm.next_index(), 0);
    assert_eq!(asm.sample_offset(), 0);
}

#[test]
fn test_reset_produces_identical_output() {
    let cf = 10;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let mut asm = StreamingAssembler::new(config, 2).unwrap();

    let raw0 = ramp(0.0, 100);
    let raw1 = ramp(1.0, 100);

    // First pass.
    let c0_a = asm.push_raw(raw0.clone()).unwrap();
    let c1_a = asm.push_raw(raw1.clone()).unwrap();

    asm.reset();

    // Second pass after reset -- should produce identical output.
    let c0_b = asm.push_raw(raw0).unwrap();
    let c1_b = asm.push_raw(raw1).unwrap();

    assert_eq!(c0_a.pcm.len(), c0_b.pcm.len());
    assert_eq!(c1_a.pcm.len(), c1_b.pcm.len());
    assert_eq!(c0_a.sample_offset, c0_b.sample_offset);
    assert_eq!(c1_a.sample_offset, c1_b.sample_offset);

    for (i, (&a, &b)) in c0_a.pcm.iter().zip(c0_b.pcm.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "c0[{i}]: pass1={a}, pass2={b}");
    }
    for (i, (&a, &b)) in c1_a.pcm.iter().zip(c1_b.pcm.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "c1[{i}]: pass1={a}, pass2={b}");
    }
}

#[test]
fn test_reset_mid_session() {
    let cf = 5;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let mut asm = StreamingAssembler::new(config, 3).unwrap();

    // Push only 1 of 3 chunks, then reset.
    let _ = asm.push_raw(constant(1.0, 50)).unwrap();
    assert_eq!(asm.next_index(), 1);
    assert_eq!(asm.remaining(), 2);

    asm.reset();
    assert_eq!(asm.next_index(), 0);
    assert_eq!(asm.remaining(), 3);
    assert_eq!(asm.sample_offset(), 0);
    assert!(!asm.is_complete());
}

// ---------------------------------------------------------------------------
// Sample offset preservation
// ---------------------------------------------------------------------------

#[test]
fn test_sample_offsets_are_cumulative() {
    let cf = 10;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let mut asm = StreamingAssembler::new(config, 3).unwrap();

    let c0 = asm.push_raw(constant(1.0, 100)).unwrap();
    let c1 = asm.push_raw(constant(0.5, 80)).unwrap();
    let c2 = asm.push_raw(constant(0.0, 60)).unwrap();

    assert_eq!(c0.sample_offset, 0);
    assert_eq!(c1.sample_offset, c0.pcm.len());
    assert_eq!(c2.sample_offset, c0.pcm.len() + c1.pcm.len());

    // Total emitted matches running offset.
    assert_eq!(
        asm.sample_offset(),
        c0.pcm.len() + c1.pcm.len() + c2.pcm.len(),
    );
}

#[test]
fn test_sample_offsets_stereo() {
    let cf = 4;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let channels = 2;
    let mut asm = StreamingAssembler::new_with_channels(config, 2, channels).unwrap();

    let c0 = asm.push_raw(constant(1.0, 40)).unwrap();
    let c1 = asm.push_raw(constant(0.0, 40)).unwrap();

    // Offsets are in floats (not time-domain samples), matching pcm.len().
    assert_eq!(c0.sample_offset, 0);
    assert_eq!(c1.sample_offset, c0.pcm.len());
}

// ---------------------------------------------------------------------------
// AudioChunk metadata
// ---------------------------------------------------------------------------

#[test]
fn test_chunk_metadata_fields() {
    let cf = 10;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let mut asm = StreamingAssembler::new(config, 3).unwrap();

    let c0 = asm.push_raw(constant(1.0, 100)).unwrap();
    let c1 = asm.push_raw(constant(0.5, 100)).unwrap();
    let c2 = asm.push_raw(constant(0.0, 100)).unwrap();

    // chunk_index.
    assert_eq!(c0.chunk_index, 0);
    assert_eq!(c1.chunk_index, 1);
    assert_eq!(c2.chunk_index, 2);

    // total_chunks.
    assert_eq!(c0.total_chunks, 3);
    assert_eq!(c1.total_chunks, 3);
    assert_eq!(c2.total_chunks, 3);

    // is_final.
    assert!(!c0.is_final);
    assert!(!c1.is_final);
    assert!(c2.is_final);

    // channels (mono default).
    assert_eq!(c0.channels, 1);
    assert_eq!(c1.channels, 1);
    assert_eq!(c2.channels, 1);
}

#[test]
fn test_audio_chunk_duration() {
    let chunk = AudioChunk::new(
        constant(0.0, 24_000), // 1 second at 24kHz mono
        1,
        0,
        0,
        1,
        true,
    );
    let dur = chunk.duration_secs();
    assert!(
        (dur - 1.0).abs() < 1e-6,
        "expected ~1.0s duration, got {dur}",
    );

    // Stereo: 24000 floats = 0.5 seconds (12000 time-domain samples).
    let stereo_chunk = AudioChunk::new(
        constant(0.0, 24_000),
        2,
        0,
        0,
        1,
        true,
    );
    let stereo_dur = stereo_chunk.duration_secs();
    assert!(
        (stereo_dur - 0.5).abs() < 1e-6,
        "expected ~0.5s stereo duration, got {stereo_dur}",
    );
}

#[test]
fn test_audio_chunk_len_is_empty() {
    let chunk = AudioChunk::new(constant(0.0, 100), 1, 0, 0, 1, true);
    assert_eq!(chunk.len(), 100);
    assert!(!chunk.is_empty());

    let empty = AudioChunk::new(Vec::new(), 1, 0, 0, 1, true);
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());
}

// ---------------------------------------------------------------------------
// Incremental vs batch equivalence
// ---------------------------------------------------------------------------

#[test]
fn test_incremental_matches_batch_five_chunks() {
    let cf = 12;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let n = 5;

    let raws: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let len = 100 + i * 20;
            (0..len).map(|j| ((i * 1000 + j) as f32 * 0.0007).sin()).collect()
        })
        .collect();

    // Batch path.
    let batch = assemble_streaming_chunks(&raws, &config).unwrap();

    // Incremental path.
    let mut asm = StreamingAssembler::new(config, n).unwrap();
    for (i, raw) in raws.iter().enumerate() {
        let inc = asm.push_raw(raw.clone()).unwrap();
        assert_eq!(inc.pcm.len(), batch[i].pcm.len(), "chunk {i} len mismatch");
        assert_eq!(inc.sample_offset, batch[i].sample_offset, "chunk {i} offset");
        assert_eq!(inc.chunk_index, batch[i].chunk_index, "chunk {i} index");
        assert_eq!(inc.is_final, batch[i].is_final, "chunk {i} is_final");
        for (j, (&a, &b)) in inc.pcm.iter().zip(batch[i].pcm.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "chunk {i} sample {j}: inc={a}, batch={b}",
            );
        }
    }
    assert!(asm.is_complete());
}

// ---------------------------------------------------------------------------
// Accessor methods
// ---------------------------------------------------------------------------

#[test]
fn test_remaining_and_next_index() {
    let config = KokoroStreamConfig::new(5).unwrap();
    let mut asm = StreamingAssembler::new(config, 4).unwrap();

    assert_eq!(asm.remaining(), 4);
    assert_eq!(asm.next_index(), 0);
    assert_eq!(asm.total_chunks(), 4);
    assert!(!asm.is_complete());

    let _ = asm.push_raw(constant(1.0, 50)).unwrap();
    assert_eq!(asm.remaining(), 3);
    assert_eq!(asm.next_index(), 1);

    let _ = asm.push_raw(constant(1.0, 50)).unwrap();
    assert_eq!(asm.remaining(), 2);
    assert_eq!(asm.next_index(), 2);

    let _ = asm.push_raw(constant(1.0, 50)).unwrap();
    assert_eq!(asm.remaining(), 1);
    assert_eq!(asm.next_index(), 3);

    let _ = asm.push_raw(constant(1.0, 50)).unwrap();
    assert_eq!(asm.remaining(), 0);
    assert_eq!(asm.next_index(), 4);
    assert!(asm.is_complete());
}

// ---------------------------------------------------------------------------
// KokoroStreamConfig
// ---------------------------------------------------------------------------

#[test]
fn test_stream_config_default() {
    let config = KokoroStreamConfig::default();
    assert_eq!(config.crossfade_samples, 960);
    // 960 / 24000 = 0.04s
    let dur = config.crossfade_duration_secs();
    assert!(
        (dur - 0.04).abs() < 1e-6,
        "expected 0.04s crossfade duration, got {dur}",
    );
}

#[test]
fn test_stream_config_validate() {
    let good = KokoroStreamConfig::new(100);
    assert!(good.is_ok());

    let bad = KokoroStreamConfig::new(0);
    assert!(bad.is_err());
}

// ---------------------------------------------------------------------------
// Large chunk count stress test
// ---------------------------------------------------------------------------

#[test]
fn test_many_chunks_stress() {
    let cf = 10;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let n = 100;
    let mut asm = StreamingAssembler::new(config, n).unwrap();

    let mut total_emitted = 0usize;
    for i in 0..n {
        let len = 50 + (i % 20) * 10;
        let chunk = asm.push_raw(constant(i as f32 * 0.01, len)).unwrap();

        assert_eq!(chunk.chunk_index, i);
        assert_eq!(chunk.total_chunks, n);
        assert_eq!(chunk.is_final, i == n - 1);
        assert_eq!(chunk.sample_offset, total_emitted);

        total_emitted += chunk.pcm.len();
    }
    assert!(asm.is_complete());
    assert_eq!(asm.sample_offset(), total_emitted);
}
