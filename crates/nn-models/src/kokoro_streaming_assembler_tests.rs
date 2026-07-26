// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`StreamingAssembler`] incremental sub-utterance delivery.
//!
//! Extracted from `kokoro_streaming_tests.rs` (#3504, item 1) to match the
//! production code split (`kokoro_streaming_assembler.rs`).

use super::*;

#[test]
fn test_assembler_single_chunk() {
    let config = KokoroStreamConfig::new(10).unwrap();
    let mut asm = StreamingAssembler::new(config, 1).unwrap();
    assert!(!asm.is_complete());
    assert_eq!(asm.remaining(), 1);

    let chunk = asm.push_raw(vec![0.5f32; 100]).unwrap();

    assert!(chunk.is_final);
    assert_eq!(chunk.chunk_index, 0);
    assert_eq!(chunk.total_chunks, 1);
    assert_eq!(chunk.sample_offset, 0);
    assert_eq!(chunk.pcm.len(), 100);
    assert!(asm.is_complete());
    assert_eq!(asm.remaining(), 0);
}

#[test]
fn test_assembler_two_chunks_matches_batch() {
    let cf = 10;
    let config = KokoroStreamConfig::new(cf).unwrap();

    let raw0 = vec![1.0f32; 100];
    let raw1 = vec![0.0f32; 100];

    // Incremental path.
    let mut asm = StreamingAssembler::new(config.clone(), 2).unwrap();
    let inc_chunk0 = asm.push_raw(raw0.clone()).unwrap();
    let inc_chunk1 = asm.push_raw(raw1.clone()).unwrap();

    // Batch path.
    let batch = assemble_streaming_chunks(&[raw0, raw1], &config).unwrap();

    // Verify identical output.
    assert_eq!(inc_chunk0.pcm.len(), batch[0].pcm.len());
    assert_eq!(inc_chunk1.pcm.len(), batch[1].pcm.len());
    for (i, (&a, &b)) in inc_chunk0.pcm.iter().zip(batch[0].pcm.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "chunk0[{i}]: incremental={a}, batch={b}",
        );
    }
    for (i, (&a, &b)) in inc_chunk1.pcm.iter().zip(batch[1].pcm.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "chunk1[{i}]: incremental={a}, batch={b}",
        );
    }
    assert!(asm.is_complete());
}

#[test]
fn test_assembler_three_chunks_offsets_match_batch() {
    let cf = 5;
    let config = KokoroStreamConfig::new(cf).unwrap();

    let raws = vec![vec![0.3f32; 50], vec![0.6f32; 60], vec![0.9f32; 40]];

    // Incremental.
    let mut asm = StreamingAssembler::new(config.clone(), 3).unwrap();
    let inc: Vec<AudioChunk> = raws
        .iter()
        .map(|r| asm.push_raw(r.clone()).unwrap())
        .collect();

    // Batch.
    let batch = assemble_streaming_chunks(&raws, &config).unwrap();

    assert_eq!(inc.len(), batch.len());
    for (i, (a, b)) in inc.iter().zip(batch.iter()).enumerate() {
        assert_eq!(a.pcm.len(), b.pcm.len(), "chunk {i} len mismatch");
        assert_eq!(
            a.sample_offset, b.sample_offset,
            "chunk {i} offset mismatch"
        );
        assert_eq!(a.chunk_index, b.chunk_index, "chunk {i} index mismatch");
        assert_eq!(a.is_final, b.is_final, "chunk {i} is_final mismatch");
        for (j, (&av, &bv)) in a.pcm.iter().zip(b.pcm.iter()).enumerate() {
            assert!(
                (av - bv).abs() < 1e-6,
                "chunk {i} sample {j}: inc={av}, batch={bv}",
            );
        }
    }
}

#[test]
fn test_assembler_push_after_complete_errors() {
    let config = KokoroStreamConfig::new(5).unwrap();
    let mut asm = StreamingAssembler::new(config, 1).unwrap();
    let _ = asm.push_raw(vec![1.0f32; 50]).unwrap();
    assert!(asm.is_complete());

    let result = asm.push_raw(vec![1.0f32; 50]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("already pushed"));
}

#[test]
fn test_assembler_zero_chunks_rejected() {
    let config = KokoroStreamConfig::new(10).unwrap();
    let result = StreamingAssembler::new(config, 0);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("total_chunks"));
}

#[test]
fn test_assembler_chunk_too_short_for_crossfade() {
    let cf = 50;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let mut asm = StreamingAssembler::new(config, 2).unwrap();

    // First chunk is fine.
    let _ = asm.push_raw(vec![1.0f32; 100]).unwrap();

    // Second chunk too short for crossfade.
    let result = asm.push_raw(vec![0.0f32; 10]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("too short"));
}

/// Regression test for #3478: push_raw panics when the first chunk is shorter
/// than crossfade_samples. The second push would index out-of-bounds in
/// crossfade_blend_into because prev_tail.len() < cf.
#[test]
fn test_assembler_first_chunk_shorter_than_crossfade_returns_error() {
    let cf = 50;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let mut asm = StreamingAssembler::new(config, 2).unwrap();

    // First chunk is shorter than crossfade_samples — should return Err, not
    // silently save a short tail that causes a panic on the next push.
    let result = asm.push_raw(vec![1.0f32; 10]);
    assert!(result.is_err(), "expected error for short first chunk");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("too short"),
        "error should mention 'too short', got: {msg}",
    );
}

/// Variant of #3478: middle chunk (non-first, non-last) shorter than cf
/// should also error immediately rather than saving a short tail.
#[test]
fn test_assembler_middle_chunk_shorter_than_crossfade_returns_error() {
    let cf = 50;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let mut asm = StreamingAssembler::new(config, 3).unwrap();

    // First chunk is fine.
    let _ = asm.push_raw(vec![1.0f32; 200]).unwrap();

    // Middle chunk too short for crossfade.
    let result = asm.push_raw(vec![0.5f32; 10]);
    assert!(result.is_err(), "expected error for short middle chunk");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("too short"),
        "error should mention 'too short', got: {msg}",
    );
}

/// Single-chunk assembler: short chunks are fine (no crossfade needed).
#[test]
fn test_assembler_single_chunk_shorter_than_crossfade_ok() {
    let cf = 50;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let mut asm = StreamingAssembler::new(config, 1).unwrap();

    // Single chunk can be shorter than cf — it's both first and last.
    let chunk = asm.push_raw(vec![1.0f32; 10]).unwrap();
    assert!(chunk.is_final);
    assert_eq!(chunk.pcm.len(), 10);
}

#[test]
fn test_assembler_memory_efficiency() {
    // Verify only crossfade tail is retained, not entire previous chunk.
    let cf = 20;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let mut asm = StreamingAssembler::new(config, 3).unwrap();

    // Push a large first chunk.
    let _ = asm.push_raw(vec![1.0f32; 10_000]).unwrap();

    // prev_tail should be exactly cf samples, not 10_000.
    // We can't inspect private fields, but we can verify the assembler
    // works correctly which implies correct tail management.
    let chunk1 = asm.push_raw(vec![0.5f32; 8_000]).unwrap();
    let chunk2 = asm.push_raw(vec![0.0f32; 6_000]).unwrap();

    // Verify crossfade was applied (first cf samples of chunk1 are blended).
    // If tail was wrong, the blend would produce incorrect values.
    let inv = 1.0 / (cf - 1) as f32;
    for j in 0..cf {
        let alpha = j as f32 * inv;
        let expected = 1.0 * (1.0 - alpha) + 0.5 * alpha;
        assert!(
            (chunk1.pcm[j] - expected).abs() < 1e-5,
            "chunk1 crossfade[{j}]: expected {expected}, got {}",
            chunk1.pcm[j],
        );
    }
    assert!(chunk2.is_final);
}

/// Proof: StreamingAssembler produces identical output to assemble_streaming_chunks
/// across many chunks with varied data (ramp patterns).
#[test]
fn test_assembler_equivalence_many_chunks() {
    let cf = 15;
    let config = KokoroStreamConfig::new(cf).unwrap();
    let n = 20;

    let raws: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let len = 200 + i * 30; // varied lengths
            (0..len)
                .map(|j| ((i * len + j) as f32 * 0.001).sin())
                .collect()
        })
        .collect();

    // Batch.
    let batch = assemble_streaming_chunks(&raws, &config).unwrap();

    // Incremental.
    let mut asm = StreamingAssembler::new(config, n).unwrap();
    for (i, raw) in raws.iter().enumerate() {
        let inc = asm.push_raw(raw.clone()).unwrap();
        assert_eq!(inc.pcm.len(), batch[i].pcm.len(), "chunk {i} len");
        assert_eq!(
            inc.sample_offset, batch[i].sample_offset,
            "chunk {i} offset"
        );
        for (j, (&a, &b)) in inc.pcm.iter().zip(batch[i].pcm.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "chunk {i} sample {j}: inc={a}, batch={b}",
            );
        }
    }
    assert!(asm.is_complete());
}
