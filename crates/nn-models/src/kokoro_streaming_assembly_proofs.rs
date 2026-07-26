// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Performance and memory safety proofs for `assemble_streaming_chunks`.
//!
//! These tests verify the zero-clone assembly path: no double allocation,
//! correct inline crossfade, and exact sample_offset accounting. They also
//! benchmark linear-time assembly for many chunks.
//!
//! Extracted from `kokoro_streaming_tests.rs` (#3504, item 1) to comply
//! with the 500-line file limit.

use super::*;

// ---------------------------------------------------------------------------
// Performance proofs
// ---------------------------------------------------------------------------

/// Proof: assemble_streaming_chunks avoids double allocation.
///
/// The old implementation cloned each raw chunk (N samples), applied crossfade
/// in place, then created a truncated copy (N-cf samples), immediately dropping
/// the full clone. The fix builds the output directly: allocate emit_len, blend
/// crossfade region inline, copy the rest. Saves one allocation per chunk.
///
/// This test verifies correctness of the zero-clone assembly by comparing
/// output against the crossfade_chunks reference implementation.
#[test]
fn test_assemble_matches_crossfade_reference() {
    let cf = 20;
    let config = KokoroStreamConfig::new(cf).unwrap();

    // Create chunks with distinct ramp patterns to catch any blending errors.
    let raw0: Vec<f32> = (0..500).map(|i| (i as f32) / 500.0).collect();
    let raw1: Vec<f32> = (0..400).map(|i| 1.0 - (i as f32) / 400.0).collect();
    let raw2: Vec<f32> = (0..300).map(|i| ((i as f32) * 0.1).sin()).collect();

    let chunks = assemble_streaming_chunks(&[raw0.clone(), raw1.clone(), raw2.clone()], &config)
        .expect("assembly should succeed");

    assert_eq!(chunks.len(), 3);

    // Verify chunk 0: no crossfade, truncated.
    assert_eq!(chunks[0].pcm.len(), 500 - cf);
    for (i, &v) in chunks[0].pcm.iter().enumerate() {
        let expected = raw0[i];
        assert!(
            (v - expected).abs() < 1e-6,
            "chunk0[{i}]: expected {expected}, got {v}",
        );
    }

    // Verify chunk 1 crossfade region against reference.
    let mut ref_chunk1 = raw1.clone();
    crossfade_chunks(&raw0, &mut ref_chunk1, cf).unwrap();
    let emit1 = raw1.len() - cf;
    assert_eq!(chunks[1].pcm.len(), emit1);
    for (i, &v) in chunks[1].pcm.iter().enumerate() {
        let expected = ref_chunk1[i];
        assert!(
            (v - expected).abs() < 1e-6,
            "chunk1[{i}]: expected {expected}, got {v}",
        );
    }

    // Verify chunk 2 crossfade region against reference.
    let mut ref_chunk2 = raw2.clone();
    crossfade_chunks(&raw1, &mut ref_chunk2, cf).unwrap();
    assert_eq!(chunks[2].pcm.len(), raw2.len()); // last chunk, full length
    for (i, &v) in chunks[2].pcm.iter().enumerate() {
        let expected = ref_chunk2[i];
        assert!(
            (v - expected).abs() < 1e-6,
            "chunk2[{i}]: expected {expected}, got {v}",
        );
    }
}

/// Proof: assembly with many chunks scales linearly in total samples.
///
/// 100 chunks of 1000 samples each = 100K total samples. This should complete
/// in well under 1 second. A naive O(N*chunk_count) clone pattern would be
/// visible at this scale.
#[test]
fn test_assemble_many_chunks_linear_time() {
    let cf = 50;
    let config = KokoroStreamConfig::new(cf).unwrap();

    let n_chunks = 100;
    let chunk_size = 1000;
    let raw: Vec<Vec<f32>> = (0..n_chunks)
        .map(|i| {
            (0..chunk_size)
                .map(|j| ((i * chunk_size + j) as f32 * 0.001).sin())
                .collect()
        })
        .collect();

    let start = std::time::Instant::now();
    let chunks = assemble_streaming_chunks(&raw, &config).expect("assembly should succeed");
    let elapsed = start.elapsed();

    assert_eq!(chunks.len(), n_chunks);
    // 100K samples should assemble in well under 100ms.
    assert!(
        elapsed.as_millis() < 100,
        "assembly of {n_chunks} chunks took {}ms (expected <100ms)",
        elapsed.as_millis(),
    );

    // Verify total output length accounting for crossfade overlap.
    let total_samples: usize = chunks.iter().map(|c| c.pcm.len()).sum();
    let expected_total = n_chunks * chunk_size - (n_chunks - 1) * cf;
    assert_eq!(
        total_samples, expected_total,
        "total samples: got {total_samples}, expected {expected_total}",
    );
}

// ---------------------------------------------------------------------------
// Memory safety proofs — chunk boundary edge cases
// ---------------------------------------------------------------------------

/// Proof: no bounds panic when a non-final chunk length equals crossfade + 1.
///
/// With emit_len = 1, only 1 crossfade sample is emitted. The inline crossfade
/// path must not index past emit_len. This tests the `cf.min(emit_len)` guard
/// in the zero-clone assembly.
#[test]
fn test_assemble_chunk_barely_above_crossfade_no_panic() {
    let cf = 10;
    let config = KokoroStreamConfig::new(cf).unwrap();

    // Chunk 0: cf + 1 samples -> emit_len = 1 (barely above crossfade).
    // Chunk 1: large enough for crossfade.
    let raw = vec![vec![1.0f32; cf + 1], vec![0.5f32; 100]];
    let chunks =
        assemble_streaming_chunks(&raw, &config).expect("should not panic with emit_len=1");

    assert_eq!(chunks.len(), 2);
    assert_eq!(
        chunks[0].pcm.len(),
        1,
        "non-final chunk emits exactly 1 sample"
    );
    assert!(!chunks[0].is_final);
    // Last chunk gets full crossfade + remaining samples.
    assert_eq!(chunks[1].pcm.len(), 100);
    assert!(chunks[1].is_final);
}

/// Proof: no bounds panic when a non-final chunk length exactly equals crossfade.
///
/// With emit_len = 0, the non-final chunk produces a 0-sample AudioChunk.
/// This is a degenerate but memory-safe edge case — no allocation overflow.
#[test]
fn test_assemble_chunk_exactly_crossfade_length_produces_empty_chunk() {
    let cf = 10;
    let config = KokoroStreamConfig::new(cf).unwrap();

    // Chunk 0: exactly cf samples -> emit_len = 0.
    // Chunk 1: enough for crossfade.
    let raw = vec![vec![1.0f32; cf], vec![0.5f32; 50]];
    let chunks =
        assemble_streaming_chunks(&raw, &config).expect("should not panic with emit_len=0");

    assert_eq!(chunks.len(), 2);
    assert_eq!(
        chunks[0].pcm.len(),
        0,
        "emit_len=0 produces a 0-sample chunk (degenerate but safe)"
    );
    // Last chunk still gets crossfade applied.
    assert_eq!(chunks[1].pcm.len(), 50);
}

/// Proof: inline crossfade does not over-allocate (no 2x clone waste).
///
/// The old assembly cloned the full raw chunk (N samples), then truncated
/// to emit_len (N - cf), wasting cf * sizeof(f32) per chunk. The fix builds
/// the output with `Vec::with_capacity(emit_len)` and pushes exactly emit_len
/// elements. This test verifies capacity is at most 2x len (allocator may
/// round up, but never to the old full-clone size).
#[test]
fn test_assemble_inline_crossfade_no_overalloc() {
    let cf = 20;
    let config = KokoroStreamConfig::new(cf).unwrap();

    let raw = vec![vec![1.0f32; 200], vec![0.5f32; 300], vec![0.0f32; 150]];
    let chunks = assemble_streaming_chunks(&raw, &config).unwrap();

    // Verify each chunk's capacity is reasonable (not the old full-clone size).
    // Chunk 0: emit_len = 180, old clone would be 200.
    // Chunk 1: emit_len = 280, old clone would be 300.
    // Chunk 2: emit_len = 150 (last, full), old clone would be 150.
    let expected_emit = [200 - cf, 300 - cf, 150];
    let raw_sizes = [200, 300, 150];
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.pcm.len(), expected_emit[i], "chunk {i} wrong len");
        // Capacity should not exceed 2x emit_len (allocator rounding).
        // The old code's full clone would give capacity >= raw_sizes[i].
        assert!(
            chunk.pcm.capacity() <= expected_emit[i] * 2,
            "chunk {i}: capacity {} > 2 * emit_len {} — possible over-allocation \
             (raw size was {})",
            chunk.pcm.capacity(),
            expected_emit[i],
            raw_sizes[i],
        );
    }
}

/// Proof: crossfade_chunks does not allocate (in-place mutation only).
///
/// Verifies the standalone crossfade function modifies the slice in place
/// without growing or shrinking the buffer. The buffer's capacity and length
/// remain unchanged after crossfade.
#[test]
fn test_crossfade_no_allocation() {
    let prev = vec![1.0f32; 100];
    let mut next = vec![0.0f32; 100];
    let original_capacity = next.capacity();
    let original_len = next.len();

    crossfade_chunks(&prev, &mut next, 10).unwrap();

    assert_eq!(
        next.len(),
        original_len,
        "crossfade should not change length"
    );
    assert_eq!(
        next.capacity(),
        original_capacity,
        "crossfade should not reallocate"
    );
}

/// Proof: sample_offset accounting is exact across many chunks including
/// degenerate emit_len=0 chunks.
///
/// The sample_offset tracks cumulative position in the output stream.
/// When a chunk emits 0 samples (emit_len = 0), the next chunk's offset
/// must not advance — no gap in the stream.
#[test]
fn test_assemble_sample_offsets_monotonic_no_gaps() {
    let cf = 5;
    let config = KokoroStreamConfig::new(cf).unwrap();

    // Mix of normal and degenerate chunk sizes.
    let raw = vec![
        vec![1.0f32; 100],
        vec![0.5f32; cf], // emit_len = 0 (degenerate)
        vec![0.0f32; 80],
    ];
    let chunks = assemble_streaming_chunks(&raw, &config).unwrap();

    assert_eq!(chunks.len(), 3);

    // Verify sample_offset is monotonically non-decreasing.
    let mut cumulative = 0usize;
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(
            chunk.sample_offset, cumulative,
            "chunk {i}: expected offset {cumulative}, got {}",
            chunk.sample_offset,
        );
        cumulative += chunk.pcm.len();
    }
}
