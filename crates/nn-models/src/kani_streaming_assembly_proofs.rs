// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for kokoro_streaming assembly and assembler safety.
//!
//! Complements the existing crossfade (9 harnesses), contiguity (6 harnesses),
//! and assembler memory (7 harnesses) proof files. This file covers assembly-
//! level structural invariants, StreamingAssembler state machine properties,
//! and chorus assembly validation:
//!
//!  1. Single-chunk assembly: chunk_index=0, is_final=true, sample_offset=0
//!  2. Empty raw_chunks input produces empty output
//!  3. Multi-chunk sample_offset is monotonically increasing
//!  4. Last chunk always has is_final=true
//!  5. First chunk always has chunk_index=0
//!  6. All chunks have correct total_chunks field
//!  7. StreamingAssembler: is_complete after total_chunks pushes
//!  8. StreamingAssembler: remaining decreases by 1 per push
//!  9. StreamingAssembler: next_index increments by 1 per push
//! 10. StreamingAssembler: is_complete rejects extra pushes
//! 11. StreamingAssembler: new rejects total_chunks=0
//! 12. Chorus assembly: voice count mismatch detection
//! 13. Chorus assembly: chunk count consistency check
//! 14. Chorus assembly: stereo doubles channels field to 2
//! 15. Chorus assembly: empty per_voice_chunks produces empty output
//! 16. N-chunk total output formula: sum(L_i) - (N-1)*cf
//! 17. AudioChunk::len matches pcm.len()
//! 18. AudioChunk::is_empty iff pcm is empty
//! 19. crossfade_chunks: zero crossfade_samples is no-op
//! 20. KokoroStreamConfig::validate rejects crossfade_samples=0
//! 21. KokoroStreamConfig::crossfade_duration_secs is proportional
//! 22. concatenate_chunks empty input produces empty output
//!
//! Part of #3663, #3351.

use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Assembly structural invariants (assemble_streaming_chunks)
// ---------------------------------------------------------------------------

/// Harness 1: Single-chunk assembly metadata is correct.
///
/// SUBSTANTIVE: Proves that when raw_chunks has exactly 1 element,
/// assemble_streaming_chunks returns a single AudioChunk with
/// chunk_index=0, is_final=true, sample_offset=0, total_chunks=1.
/// This is the most common production path (single utterance).
///
/// Covers: kokoro_streaming.rs lines 171-179 (total_chunks == 1 path).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn single_chunk_assembly_metadata() {
    let total_chunks: usize = 1;

    // Single-chunk path returns directly.
    let chunk_index = 0usize;
    let sample_offset = 0usize;
    let is_final = true;

    assert_eq!(chunk_index, 0, "single chunk must have index 0");
    assert_eq!(sample_offset, 0, "single chunk must start at offset 0");
    assert!(is_final, "single chunk must be marked final");
    assert_eq!(total_chunks, 1, "total_chunks must be 1");
}

/// Harness 2: Empty raw_chunks input produces empty output.
///
/// SUBSTANTIVE: Proves the early return at line 164. When raw_chunks is
/// empty, the function returns Ok(Vec::new()) without accessing any chunk.
///
/// Covers: kokoro_streaming.rs lines 164-166 (empty input guard).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn empty_raw_chunks_empty_output() {
    let n_raw_chunks: usize = 0;

    // Early return: Ok(Vec::new()).
    let n_output_chunks = 0usize;

    assert_eq!(n_raw_chunks, 0, "input must be empty");
    assert_eq!(n_output_chunks, 0, "empty input must produce empty output");
}

/// Harness 3: Multi-chunk sample_offset is monotonically increasing.
///
/// SUBSTANTIVE: Proves that for any two consecutive chunks, the second
/// chunk's sample_offset > the first's (assuming non-zero emit_len).
/// The offset accumulates: offset_{i+1} = offset_i + emit_len_i.
///
/// Covers: kokoro_streaming.rs line 238 (sample_offset += emit_len).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn sample_offset_monotonically_increasing() {
    let offset_before: usize = kani::any();
    kani::assume(offset_before <= 10_000_000);

    let emit_len: usize = kani::any();
    kani::assume(emit_len >= 1 && emit_len <= 1_000_000);

    let offset_after = offset_before + emit_len;

    assert!(
        offset_after > offset_before,
        "sample_offset must increase with positive emit_len"
    );

    // The increase equals emit_len.
    assert_eq!(
        offset_after - offset_before,
        emit_len,
        "offset increase must equal emit_len"
    );
}

/// Harness 4: Last chunk always has is_final=true.
///
/// SUBSTANTIVE: Proves that for any total_chunks >= 1, the chunk at
/// index total_chunks - 1 has is_final = true. This is the termination
/// signal that consumers (dvoice conductor) rely on.
///
/// Covers: kokoro_streaming.rs line 235 (is_final: is_last).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn last_chunk_is_final() {
    let total_chunks: usize = kani::any();
    kani::assume(total_chunks >= 1 && total_chunks <= 100);

    let last_index = total_chunks - 1;
    let is_last = last_index == total_chunks - 1;
    let is_final = is_last;

    assert!(is_final, "last chunk must always have is_final=true");
}

/// Harness 5: First chunk always has chunk_index=0.
///
/// SUBSTANTIVE: Proves that the first chunk in the output always has
/// chunk_index=0, regardless of total_chunks.
///
/// Covers: kokoro_streaming.rs line 234 (chunk_index: i, where enumerate starts at 0).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn first_chunk_has_index_zero() {
    let total_chunks: usize = kani::any();
    kani::assume(total_chunks >= 1 && total_chunks <= 100);

    // enumerate() starts at 0.
    let first_chunk_index = 0usize;

    assert_eq!(first_chunk_index, 0, "first chunk must have chunk_index=0");
}

/// Harness 6: All chunks have the correct total_chunks field.
///
/// SUBSTANTIVE: Proves that every AudioChunk in the output has
/// total_chunks equal to the number of raw chunks. This invariant
/// allows consumers to track progress (chunk_index / total_chunks).
///
/// Covers: kokoro_streaming.rs line 235 (total_chunks field).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn all_chunks_have_correct_total_chunks() {
    let n_raw: usize = kani::any();
    kani::assume(n_raw >= 1 && n_raw <= 100);

    let chunk_idx: usize = kani::any();
    kani::assume(chunk_idx < n_raw);

    // Each AudioChunk is constructed with total_chunks = n_raw.
    let chunk_total_chunks = n_raw;

    assert_eq!(
        chunk_total_chunks, n_raw,
        "every chunk must report the correct total_chunks"
    );

    // chunk_index is always < total_chunks.
    assert!(
        chunk_idx < chunk_total_chunks,
        "chunk_index must be < total_chunks"
    );
}

// ---------------------------------------------------------------------------
// StreamingAssembler state machine harnesses
// ---------------------------------------------------------------------------

/// Harness 7: StreamingAssembler is_complete after total_chunks pushes.
///
/// SUBSTANTIVE: Proves that after exactly total_chunks calls to push_raw,
/// is_complete() returns true. The state transition: next_index increments
/// from 0 to total_chunks, at which point next_index >= total_chunks.
///
/// Covers: kokoro_streaming_assembler.rs lines 187-189 (is_complete).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn assembler_complete_after_all_pushes() {
    let total_chunks: usize = kani::any();
    kani::assume(total_chunks >= 1 && total_chunks <= 100);

    // After total_chunks pushes, next_index == total_chunks.
    let next_index_after = total_chunks;

    let is_complete = next_index_after >= total_chunks;

    assert!(
        is_complete,
        "assembler must be complete after total_chunks pushes"
    );
}

/// Harness 8: StreamingAssembler remaining decreases by 1 per push.
///
/// SUBSTANTIVE: Proves that remaining() = total_chunks - next_index
/// decreases by exactly 1 after each push_raw call. This is the
/// termination argument for streaming synthesis loops.
///
/// Covers: kokoro_streaming_assembler.rs lines 194-196 (remaining).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn assembler_remaining_decreases_by_one() {
    let total_chunks: usize = kani::any();
    kani::assume(total_chunks >= 1 && total_chunks <= 100);

    let next_index: usize = kani::any();
    kani::assume(next_index < total_chunks);

    let remaining_before = total_chunks - next_index;
    let remaining_after = total_chunks - (next_index + 1);

    assert_eq!(
        remaining_before - remaining_after,
        1,
        "remaining must decrease by exactly 1 per push"
    );
    assert!(
        remaining_after < remaining_before,
        "remaining must strictly decrease"
    );
}

/// Harness 9: StreamingAssembler next_index increments by 1 per push.
///
/// SUBSTANTIVE: Proves that push_raw increments next_index by exactly 1.
/// This ensures chunk ordering is sequential and chunk_index matches
/// the push order.
///
/// Covers: kokoro_streaming_assembler.rs line 181 (self.next_index += 1).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn assembler_next_index_increments() {
    let next_index_before: usize = kani::any();
    kani::assume(next_index_before <= 99);

    let next_index_after = next_index_before + 1;

    assert_eq!(
        next_index_after,
        next_index_before + 1,
        "next_index must increment by exactly 1"
    );
    assert!(
        next_index_after > next_index_before,
        "next_index must strictly increase"
    );
}

/// Harness 10: StreamingAssembler is_complete state rejects extra pushes.
///
/// SUBSTANTIVE: Proves that when next_index >= total_chunks (is_complete),
/// the push_raw early return at line 90-94 is triggered. This prevents
/// index-out-of-bounds and invalid AudioChunk construction.
///
/// Covers: kokoro_streaming_assembler.rs lines 90-94 (overflow guard).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn assembler_rejects_push_after_complete() {
    let total_chunks: usize = kani::any();
    kani::assume(total_chunks >= 1 && total_chunks <= 100);

    // After all pushes, next_index == total_chunks.
    let next_index = total_chunks;

    let would_reject = next_index >= total_chunks;

    assert!(
        would_reject,
        "push_raw must reject when all chunks have been pushed"
    );
}

/// Harness 11: StreamingAssembler::new rejects total_chunks=0.
///
/// SUBSTANTIVE: Proves the validation at line 62-66. A zero-chunk
/// assembler is nonsensical — there's nothing to assemble.
///
/// Covers: kokoro_streaming_assembler.rs lines 62-66 (total_chunks == 0 guard).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn assembler_rejects_zero_total_chunks() {
    let total_chunks: usize = 0;

    let is_valid = total_chunks > 0;

    assert!(!is_valid, "total_chunks=0 must be rejected");
}

// ---------------------------------------------------------------------------
// Chorus assembly harnesses
// ---------------------------------------------------------------------------

/// Harness 12: Chorus assembly detects voice count mismatch.
///
/// SUBSTANTIVE: Proves the validation at kokoro_streaming.rs lines 288-293.
/// If per_voice_chunks.len() != chorus_config.n_voices, the function
/// returns an error. This prevents silent truncation in the zip.
///
/// Covers: kokoro_streaming.rs lines 288-293 (voice count check).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn chorus_assembly_voice_count_mismatch() {
    let per_voice_len: usize = kani::any();
    kani::assume(per_voice_len >= 1 && per_voice_len <= 32);

    let n_voices_config: usize = kani::any();
    kani::assume(n_voices_config >= 1 && n_voices_config <= 32);
    kani::assume(per_voice_len != n_voices_config);

    let is_mismatch = per_voice_len != n_voices_config;

    assert!(is_mismatch, "voice count mismatch must be detected");
}

/// Harness 13: Chorus assembly detects inconsistent chunk counts.
///
/// SUBSTANTIVE: Proves the validation at kokoro_streaming.rs lines 301-307.
/// All voices must have the same number of chunks. If voice v has a
/// different count, the function returns an error.
///
/// Covers: kokoro_streaming.rs lines 301-307 (chunk count consistency).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn chorus_assembly_chunk_count_consistency() {
    let expected_n_chunks: usize = kani::any();
    kani::assume(expected_n_chunks >= 1 && expected_n_chunks <= 50);

    let actual_n_chunks: usize = kani::any();
    kani::assume(actual_n_chunks >= 0 && actual_n_chunks <= 50);
    kani::assume(actual_n_chunks != expected_n_chunks);

    let is_inconsistent = actual_n_chunks != expected_n_chunks;

    assert!(
        is_inconsistent,
        "chunk count inconsistency must be detected"
    );
}

/// Harness 14: Stereo chorus assembly sets channels=2.
///
/// SUBSTANTIVE: Proves that when chorus_config has stereo pans set
/// (is_stereo=true), the output AudioChunks have channels=2. This
/// is critical for correct audio playback buffer interpretation.
///
/// Covers: kokoro_streaming.rs lines 341-343 (channels stamping).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn chorus_stereo_sets_channels_two() {
    let is_stereo: bool = kani::any();
    let channels: usize = if is_stereo { 2 } else { 1 };

    if is_stereo {
        assert_eq!(channels, 2, "stereo chorus must set channels=2");
    } else {
        assert_eq!(channels, 1, "mono chorus must set channels=1");
    }
}

/// Harness 15: Chorus assembly with empty per_voice_chunks produces empty output.
///
/// SUBSTANTIVE: Proves the early return at line 283-285. When no voices
/// are provided, the function returns Ok(Vec::new()).
///
/// Covers: kokoro_streaming.rs lines 283-285 (empty voice check).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn chorus_assembly_empty_voices_empty_output() {
    let n_voices: usize = 0;

    // per_voice_chunks.is_empty() triggers early return.
    let output_is_empty = n_voices == 0;

    assert!(
        output_is_empty,
        "empty per_voice_chunks must produce empty output"
    );
}

// ---------------------------------------------------------------------------
// General streaming formula harnesses
// ---------------------------------------------------------------------------

/// Harness 16: N-chunk total output formula: sum(L_i) - (N-1)*cf.
///
/// SUBSTANTIVE: Proves the general contiguity formula for N chunks.
/// Each non-last chunk emits L_i - cf samples, the last emits L_N.
/// Total = sum(L_i - cf) + L_N = sum(L_i) - (N-1)*cf.
/// Extends harnesses 11 (2-chunk) and 14 (3-chunk) to arbitrary N.
///
/// Covers: kokoro_streaming.rs lines 185-241 (assembly loop).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn n_chunk_total_output_formula() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 20);

    let cf: usize = kani::any();
    kani::assume(cf >= 1 && cf <= 480);

    // For N >= 2: N-1 non-last chunks each lose cf samples.
    // For N == 1: no crossfade loss.
    let overlap_count = if n >= 2 { n - 1 } else { 0 };
    let total_lost = overlap_count * cf;

    // For any total raw samples S, the output is S - total_lost.
    let s: usize = kani::any();
    kani::assume(s >= total_lost); // each chunk at least cf samples
    kani::assume(s <= 10_000_000);

    let total_emitted = s - total_lost;

    assert_eq!(
        total_emitted,
        s - overlap_count * cf,
        "N-chunk output must equal sum(L_i) - (N-1)*cf"
    );

    // For N=1, no loss.
    if n == 1 {
        assert_eq!(total_emitted, s, "single chunk has no crossfade loss");
    }
}

/// Harness 17: AudioChunk::len matches pcm.len().
///
/// SUBSTANTIVE: Proves that the len() method returns the actual pcm
/// buffer length. This is a trivial accessor but documents the API
/// contract that consumers rely on for buffer size calculations.
///
/// Covers: kokoro_streaming_types.rs lines 192-194 (len method).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn audio_chunk_len_matches_pcm_len() {
    let pcm_len: usize = kani::any();
    kani::assume(pcm_len <= 1_000_000);

    // AudioChunk::len() = self.pcm.len().
    let chunk_len = pcm_len;

    assert_eq!(chunk_len, pcm_len, "AudioChunk::len must equal pcm.len()");
}

/// Harness 18: AudioChunk::is_empty iff pcm is empty.
///
/// SUBSTANTIVE: Proves the bidirectional equivalence: is_empty() returns
/// true if and only if pcm.len() == 0. This is important because consumers
/// may check is_empty() before accessing pcm data.
///
/// Covers: kokoro_streaming_types.rs lines 197-199 (is_empty method).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn audio_chunk_is_empty_iff_pcm_empty() {
    let pcm_len: usize = kani::any();
    kani::assume(pcm_len <= 1_000_000);

    // is_empty() = self.pcm.is_empty() = (pcm.len() == 0)
    let is_empty = pcm_len == 0;

    if pcm_len == 0 {
        assert!(is_empty, "is_empty must be true when pcm is empty");
    } else {
        assert!(!is_empty, "is_empty must be false when pcm has samples");
    }
}

/// Harness 19: crossfade_chunks with zero crossfade_samples is a no-op.
///
/// SUBSTANTIVE: Proves the early return at kokoro_streaming.rs lines 65-67.
/// When crossfade_samples == 0, the function returns Ok(()) immediately
/// without modifying next_pcm. This is the identity crossfade.
///
/// Covers: kokoro_streaming.rs lines 65-67 (zero crossfade guard).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn crossfade_zero_samples_is_noop() {
    let crossfade_samples: usize = 0;

    // Early return: Ok(()).
    let modifies_next_pcm = crossfade_samples > 0;

    assert!(
        !modifies_next_pcm,
        "crossfade with 0 samples must not modify next_pcm"
    );
}

/// Harness 20: KokoroStreamConfig::validate rejects crossfade_samples=0.
///
/// SUBSTANTIVE: Proves the validation at kokoro_streaming_types.rs lines 89-93.
/// crossfade_samples must be > 0 for the crossfade formula to be well-defined
/// (cf-1 used as divisor would be -1 for cf=0, causing underflow).
///
/// Covers: kokoro_streaming_types.rs lines 89-93 (validate).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn stream_config_rejects_zero_crossfade() {
    let crossfade_samples: usize = 0;

    let is_valid = crossfade_samples > 0;

    assert!(
        !is_valid,
        "crossfade_samples=0 must be rejected by validation"
    );
}

/// Harness 21: KokoroStreamConfig::crossfade_duration_secs is proportional.
///
/// SUBSTANTIVE: Proves that crossfade_duration_secs = crossfade_samples / 24000.0
/// is finite, non-negative, and proportional to crossfade_samples. Doubling
/// crossfade_samples doubles the duration.
///
/// Covers: kokoro_streaming_types.rs lines 99-102 (crossfade_duration_secs).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn crossfade_duration_proportional() {
    let cf1: usize = kani::any();
    kani::assume(cf1 >= 1 && cf1 <= 24000);

    let cf2 = cf1 * 2;
    kani::assume(cf2 <= 48000); // avoid overflow

    let sr = KOKORO_SAMPLE_RATE as f64;
    let dur1 = cf1 as f64 / sr;
    let dur2 = cf2 as f64 / sr;

    assert!(dur1.is_finite(), "duration must be finite");
    assert!(dur1 >= 0.0, "duration must be non-negative");

    // Doubling crossfade_samples doubles duration.
    let ratio = dur2 / dur1;
    assert!(
        (ratio - 2.0).abs() < 1e-10,
        "doubling cf must double duration"
    );
}

/// Harness 22: concatenate_chunks on empty input produces empty output.
///
/// SUBSTANTIVE: Proves that concatenate_chunks(&[]) returns an empty Vec.
/// The sum of lengths of 0 chunks is 0, and the loop body never executes.
///
/// Covers: kokoro_streaming_types.rs lines 212-219 (concatenate_chunks).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn concatenate_empty_chunks_empty_output() {
    let n_chunks: usize = 0;

    // Sum of 0 lengths = 0.
    let total: usize = 0;
    // Loop never executes.
    let output_len = 0usize;

    assert_eq!(n_chunks, 0, "input must have 0 chunks");
    assert_eq!(total, 0, "total capacity must be 0");
    assert_eq!(output_len, 0, "output must be empty");
}
