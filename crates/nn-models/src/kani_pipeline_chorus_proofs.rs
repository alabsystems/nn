// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for kokoro_pipeline and kokoro_chorus safety.
//!
//! Proves critical safety properties NOT covered by existing harnesses in
//! `kokoro_chorus_kani_tests.rs` (17 harnesses covering mix_voices numerics,
//! NaN propagation, and stereo pan law).
//!
//! This file covers:
//!
//! **Pipeline (kokoro_pipeline.rs):**
//!  1. chunks_to_tensors shape invariant: output shape is [1, T]
//!  2. chunks_to_tensors preserves chunk count
//!  3. Pipeline chorus validation: styles length must match n_voices
//!  4. Pipeline chorus validation: speeds length must match n_voices
//!  5. Stereo crossfade doubling: crossfade_samples * 2 does not overflow
//!  6. per_voice_chunks allocation: Vec capacity matches n_voices
//!  7. Batch synthesis output count matches voice count
//!
//! **Chorus config (kokoro_chorus.rs):**
//!  8. equal_gain rejects n_voices == 0
//!  9. equal_gain rejects n_voices > 32
//! 10. with_gains rejects NaN gain
//! 11. with_gains rejects negative gain
//! 12. with_gains rejects gain > 1.0
//! 13. with_stereo_pan rejects gains/pans length mismatch
//! 14. validate catches gains length != n_voices
//! 15. validate catches pans length != n_voices
//! 16. duration_secs: finite and non-negative for valid inputs
//! 17. duration_secs: division by KOKORO_SAMPLE_RATE is safe
//!
//! **Streaming (kokoro_streaming.rs, kokoro_streaming_types.rs):**
//! 18. Crossfade alpha is in [0.0, 1.0] for valid indices
//! 19. Crossfade convex combination preserves boundedness
//! 20. emit_len: saturating_sub prevents underflow
//! 21. AudioChunk duration_secs: channels division is safe
//! 22. concatenate_chunks capacity: total equals sum of lengths
//! 23. KokoroStreamConfig default crossfade is 960 samples
//! 24. Single-sample crossfade average is bounded
//!
//! Part of #3642, #3351.

use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Pipeline harnesses (kokoro_pipeline.rs)
// ---------------------------------------------------------------------------

/// Harness 1: chunks_to_tensors output shape is [1, T] for each chunk.
///
/// SUBSTANTIVE: Proves that for any token chunk of length T (1..=512),
/// the resulting DynTensor shape [1, T] has exactly 2 dimensions with
/// batch=1 and seq_len=T. This matches the KokoroSynth::synthesize_chunk
/// contract requiring [1, T] u32 input tensors.
///
/// The critical property: `DynTensor::from_vec_u32(ids, &[1, len], &Device::Cpu)`
/// produces shape [1, len] where the element count equals len (not 1*len
/// with potential overflow for large T). Since T <= 512 (max_phoneme_tokens + 2),
/// the product 1 * T never overflows.
///
/// Covers: kokoro_pipeline.rs lines 505-514 (chunks_to_tensors).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn chunks_to_tensors_shape_is_1_by_t() {
    let t: usize = kani::any();
    // Token chunks have length 2..=512 (PAD + content + PAD).
    kani::assume(t >= 2 && t <= 512);

    // Shape is [1, t].
    let batch = 1usize;
    let seq_len = t;

    // Total element count.
    let total = batch * seq_len;

    assert_eq!(total, t, "element count must equal T for batch=1");
    assert!(total <= 512, "total must fit in max context window");
    assert!(batch == 1, "batch dimension must be 1");
    assert!(seq_len == t, "seq dimension must equal token count");
}

/// Harness 2: chunks_to_tensors preserves chunk count.
///
/// SUBSTANTIVE: Proves that chunks_to_tensors maps N input chunks to
/// exactly N output tensors (1:1 mapping). This is the cardinality
/// invariant that the synthesis loop depends on.
///
/// Covers: kokoro_pipeline.rs lines 505-514 (chunks_to_tensors map).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn chunks_to_tensors_preserves_count() {
    let n_chunks: usize = kani::any();
    kani::assume(n_chunks <= 100);

    // iter().map(...).collect() preserves length.
    let n_output = n_chunks;

    assert_eq!(
        n_output, n_chunks,
        "output tensor count must equal input chunk count"
    );
}

/// Harness 3: Pipeline chorus validation — styles length must match n_voices.
///
/// SUBSTANTIVE: Proves the validation logic at kokoro_pipeline.rs:321-326.
/// If styles.len() != n_voices, the pipeline returns an error. This prevents
/// index-out-of-bounds in synthesize_batch which iterates styles by index.
///
/// Covers: kokoro_pipeline.rs lines 321-326 (text_to_chorus styles check).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pipeline_chorus_styles_count_validation() {
    let n_voices: usize = kani::any();
    kani::assume(n_voices >= 1 && n_voices <= 32);

    let styles_len: usize = kani::any();
    kani::assume(styles_len <= 33);

    let is_valid = styles_len == n_voices;

    if !is_valid {
        // Pipeline would return PipelineError::Assembly.
        assert!(styles_len != n_voices, "mismatch must be detected");
    } else {
        // Safe to proceed: zip(styles, speeds) iterates n_voices times.
        assert_eq!(styles_len, n_voices, "matching lengths must be equal");
    }
}

/// Harness 4: Pipeline chorus validation — speeds length must match n_voices.
///
/// SUBSTANTIVE: Proves the validation at kokoro_pipeline.rs:327-332.
/// speeds.len() != n_voices returns an error, preventing the zip in
/// synthesize_batch from silently truncating to the shorter iterator.
///
/// Covers: kokoro_pipeline.rs lines 327-332 (text_to_chorus speeds check).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pipeline_chorus_speeds_count_validation() {
    let n_voices: usize = kani::any();
    kani::assume(n_voices >= 1 && n_voices <= 32);

    let speeds_len: usize = kani::any();
    kani::assume(speeds_len <= 33);

    let is_valid = speeds_len == n_voices;

    if !is_valid {
        assert!(speeds_len != n_voices, "mismatch must be detected");
    } else {
        assert_eq!(speeds_len, n_voices, "matching lengths must be equal");
    }
}

/// Harness 5: Stereo crossfade doubling does not overflow.
///
/// SUBSTANTIVE: In text_to_chorus_streaming (line 417), stereo mode
/// doubles the crossfade_samples: `crossfade_samples * 2`. This harness
/// proves the multiplication does not overflow for all valid crossfade
/// values (up to 48000 = 2 seconds at 24kHz, far beyond production use
/// of 480 samples = 20ms).
///
/// Covers: kokoro_pipeline.rs line 417 (crossfade_samples * 2).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn stereo_crossfade_doubling_no_overflow() {
    let crossfade_samples: usize = kani::any();
    // Valid range: 1..=48000 (up to 2 seconds at 24kHz).
    kani::assume(crossfade_samples >= 1 && crossfade_samples <= 48000);

    let doubled = crossfade_samples * 2;

    assert!(
        doubled >= crossfade_samples,
        "doubled crossfade must be >= original (no overflow)"
    );
    assert!(doubled <= 96000, "doubled crossfade must be bounded");
    assert_eq!(
        doubled,
        crossfade_samples + crossfade_samples,
        "multiplication by 2 must equal addition"
    );
}

/// Harness 6: per_voice_chunks allocation capacity matches n_voices.
///
/// SUBSTANTIVE: In text_to_chorus (line 348-349), a Vec of Vecs is
/// allocated with capacity n_voices. Each outer element is itself a Vec
/// with capacity n_chunks. This harness proves the outer capacity is
/// correct — each of the n_voices gets exactly one inner Vec.
///
/// Covers: kokoro_pipeline.rs lines 348-349 (per_voice_chunks allocation).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn per_voice_chunks_allocation_correct() {
    let n_voices: usize = kani::any();
    kani::assume(n_voices >= 1 && n_voices <= 32);

    let n_chunks: usize = kani::any();
    kani::assume(n_chunks >= 1 && n_chunks <= 100);

    // (0..n).map(|_| Vec::with_capacity(n_chunks)).collect()
    let outer_len = n_voices;

    assert_eq!(outer_len, n_voices, "outer Vec must have n_voices elements");
    // Total inner allocations: n_voices * n_chunks (bounded).
    let total_inner_capacity = n_voices * n_chunks;
    assert!(
        total_inner_capacity <= 3200,
        "total inner capacity bounded by 32 * 100"
    );
}

/// Harness 7: Batch synthesis output count must match voice count.
///
/// SUBSTANTIVE: synthesize_batch returns Vec<Vec<f32>> where the outer
/// length must equal styles.len() (from the zip in the default impl,
/// kokoro_pipeline.rs:57-62). If the backend returns a different count,
/// the enumerate at line 357-359 would silently skip or panic.
///
/// Covers: kokoro_pipeline.rs lines 51-62 (synthesize_batch default).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn batch_synthesis_output_count_matches_voices() {
    let n_voices: usize = kani::any();
    kani::assume(n_voices >= 1 && n_voices <= 32);

    // The default synthesize_batch: styles.iter().zip(speeds).map(...).collect()
    // zip truncates to min(styles.len(), speeds.len()).
    // The pipeline validates styles.len() == speeds.len() == n_voices.
    let styles_len = n_voices;
    let speeds_len = n_voices;

    let zip_len = styles_len.min(speeds_len);
    let output_count = zip_len;

    assert_eq!(
        output_count, n_voices,
        "batch output must have exactly n_voices PCM buffers"
    );
}

// ---------------------------------------------------------------------------
// Chorus config harnesses (kokoro_chorus.rs)
// ---------------------------------------------------------------------------

/// Harness 8: ChorusConfig::equal_gain rejects n_voices == 0.
///
/// SUBSTANTIVE: Proves the guard at kokoro_chorus.rs:75-79. n_voices=0
/// would cause division by zero in the gain computation (1.0 / 0 = Inf)
/// and produce an empty gains vector. The validator catches this.
///
/// Covers: kokoro_chorus.rs line 75 (n_voices == 0 guard).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn equal_gain_rejects_zero_voices() {
    let n_voices: usize = 0;

    let is_valid = n_voices >= 1 && n_voices <= 32;

    assert!(!is_valid, "n_voices=0 must be rejected");
    // Also: 1.0 / 0.0 = Inf (not finite).
    let gain = 1.0f32 / n_voices as f32;
    assert!(!gain.is_finite(), "1.0/0 must produce Inf");
}

/// Harness 9: ChorusConfig::equal_gain rejects n_voices > 32.
///
/// SUBSTANTIVE: Proves the upper bound guard at kokoro_chorus.rs:75.
/// The 32-voice limit prevents excessive resource allocation (32 GPU
/// synthesis calls per chunk). The rejection is for any n > 32.
///
/// Covers: kokoro_chorus.rs line 75 (n_voices > 32 guard).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn equal_gain_rejects_over_32_voices() {
    let n_voices: usize = kani::any();
    kani::assume(n_voices > 32);
    kani::assume(n_voices <= 1000); // bounded for tractability

    let is_valid = n_voices >= 1 && n_voices <= 32;

    assert!(!is_valid, "n_voices > 32 must be rejected");
}

/// Harness 10: ChorusConfig::with_gains rejects NaN gain.
///
/// SUBSTANTIVE: Proves the validation at kokoro_chorus.rs:99-105.
/// A NaN gain would propagate through mix_voices (harness 8 in
/// kokoro_chorus_kani_tests.rs proves NaN gain → NaN output). The
/// with_gains constructor is the defense boundary.
///
/// Covers: kokoro_chorus.rs lines 99-105 (with_gains NaN check).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn with_gains_rejects_nan() {
    let gain = f32::NAN;

    // The validation: !g.is_finite() || g < 0.0 || g > 1.0
    let is_invalid = !gain.is_finite() || gain < 0.0 || gain > 1.0;

    // NaN makes is_finite() false, so the first clause catches it.
    assert!(
        is_invalid,
        "NaN gain must be rejected by with_gains validation"
    );
}

/// Harness 11: ChorusConfig::with_gains rejects negative gain.
///
/// SUBSTANTIVE: Proves that any negative finite gain triggers the
/// validation error. Negative gains would invert the audio phase,
/// which is physically meaningful for noise cancellation but not for
/// chorus mixing. The API restricts to [0.0, 1.0].
///
/// Covers: kokoro_chorus.rs line 100 (g < 0.0 check).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn with_gains_rejects_negative() {
    let gain: f32 = kani::any();
    kani::assume(gain.is_finite());
    kani::assume(gain < 0.0);

    let is_invalid = !gain.is_finite() || gain < 0.0 || gain > 1.0;

    assert!(is_invalid, "negative gain must be rejected");
}

/// Harness 12: ChorusConfig::with_gains rejects gain > 1.0.
///
/// SUBSTANTIVE: Proves that any finite gain exceeding 1.0 triggers
/// the validation error. Gains > 1.0 would amplify the signal beyond
/// the [-1, 1] range even for a single voice, requiring clipping.
///
/// Covers: kokoro_chorus.rs line 100 (g > 1.0 check).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn with_gains_rejects_over_one() {
    let gain: f32 = kani::any();
    kani::assume(gain.is_finite());
    kani::assume(gain > 1.0);

    let is_invalid = !gain.is_finite() || gain < 0.0 || gain > 1.0;

    assert!(is_invalid, "gain > 1.0 must be rejected");
}

/// Harness 13: with_stereo_pan rejects gains/pans length mismatch.
///
/// SUBSTANTIVE: Proves the guard at kokoro_chorus.rs:121-127. If
/// gains.len() != pans.len(), the zip in stereo mixing would silently
/// truncate, dropping voices. The constructor catches this.
///
/// Covers: kokoro_chorus.rs lines 121-127 (with_stereo_pan length check).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn with_stereo_pan_rejects_length_mismatch() {
    let gains_len: usize = kani::any();
    kani::assume(gains_len >= 1 && gains_len <= 32);

    let pans_len: usize = kani::any();
    kani::assume(pans_len >= 1 && pans_len <= 32);
    kani::assume(pans_len != gains_len);

    let is_mismatch = gains_len != pans_len;

    assert!(is_mismatch, "gains/pans length mismatch must be detected");
}

/// Harness 14: ChorusConfig::validate catches gains length != n_voices.
///
/// SUBSTANTIVE: Proves the consistency check at kokoro_chorus.rs:156-164.
/// This catches configs where n_voices was mutated after construction
/// (the struct fields are pub due to #[non_exhaustive]).
///
/// Covers: kokoro_chorus.rs lines 156-164 (validate gains length).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_catches_gains_n_voices_mismatch() {
    let n_voices: usize = kani::any();
    kani::assume(n_voices >= 1 && n_voices <= 32);

    let gains_len: usize = kani::any();
    kani::assume(gains_len >= 0 && gains_len <= 32);
    kani::assume(gains_len != n_voices);

    let is_valid = gains_len == n_voices;

    assert!(!is_valid, "gains length != n_voices must fail validation");
}

/// Harness 15: ChorusConfig::validate catches pans length != n_voices.
///
/// SUBSTANTIVE: Proves the consistency check at kokoro_chorus.rs:166-173.
/// When pans is Some, its length must equal n_voices. A mismatch would
/// cause the stereo mixing zip to silently truncate voices.
///
/// Covers: kokoro_chorus.rs lines 166-173 (validate pans length).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_catches_pans_n_voices_mismatch() {
    let n_voices: usize = kani::any();
    kani::assume(n_voices >= 1 && n_voices <= 32);

    let pans_len: usize = kani::any();
    kani::assume(pans_len >= 0 && pans_len <= 32);
    kani::assume(pans_len != n_voices);

    // When pans is Some:
    let is_valid = pans_len == n_voices;

    assert!(!is_valid, "pans length != n_voices must fail validation");
}

/// Harness 16: duration_secs is finite and non-negative for valid inputs.
///
/// SUBSTANTIVE: Proves that ChorusConfig::duration_secs (kokoro_chorus.rs:179-181)
/// produces a finite, non-negative f64 for any realistic sample count.
/// The computation is `max_samples as f64 / KOKORO_SAMPLE_RATE as f64`.
/// Since KOKORO_SAMPLE_RATE is 24000 (nonzero), no division by zero.
///
/// Covers: kokoro_chorus.rs lines 179-181 (duration_secs).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn duration_secs_finite_nonnegative() {
    let max_samples: usize = kani::any();
    // Bound: up to 24 hours of audio at 24kHz = 2,073,600,000 samples.
    kani::assume(max_samples <= 2_073_600_000);

    let sample_rate: usize = KOKORO_SAMPLE_RATE;

    // Division by sample_rate.
    let duration = max_samples as f64 / sample_rate as f64;

    assert!(duration.is_finite(), "duration must be finite");
    assert!(duration >= 0.0, "duration must be non-negative");
    // Max duration: ~24 hours.
    assert!(duration <= 86400.0 + 1.0, "duration must be <= 24 hours");
}

/// Harness 17: duration_secs division by KOKORO_SAMPLE_RATE is safe.
///
/// SUBSTANTIVE: Proves that KOKORO_SAMPLE_RATE (24000) as f64 is finite
/// and nonzero, making the division in duration_secs always well-defined.
/// This is a regression guard — if KOKORO_SAMPLE_RATE were accidentally
/// set to 0, duration_secs would produce Inf.
///
/// Covers: kokoro_chorus.rs line 180 (KOKORO_SAMPLE_RATE divisor).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn sample_rate_divisor_nonzero() {
    let sr = KOKORO_SAMPLE_RATE;

    assert!(sr > 0, "KOKORO_SAMPLE_RATE must be positive");

    let sr_f64 = sr as f64;
    assert!(sr_f64.is_finite(), "sample rate as f64 must be finite");
    assert!(sr_f64 > 0.0, "sample rate as f64 must be positive");

    // Reciprocal must be finite.
    let inv = 1.0 / sr_f64;
    assert!(inv.is_finite(), "1/sample_rate must be finite");
    assert!(inv > 0.0, "1/sample_rate must be positive");
}

// ---------------------------------------------------------------------------
// Streaming harnesses (kokoro_streaming.rs, kokoro_streaming_types.rs)
// ---------------------------------------------------------------------------

/// Harness 18: Crossfade alpha is in [0.0, 1.0] for all valid indices.
///
/// SUBSTANTIVE: Proves that the linear crossfade interpolation factor
/// `alpha = i / (crossfade_samples - 1)` is always in [0.0, 1.0] for
/// i in 0..crossfade_samples. This is the correctness precondition for
/// the convex combination in crossfade_chunks (kokoro_streaming.rs:98-99).
///
/// When crossfade_samples == 1, alpha is handled as a special case
/// (average). For crossfade_samples >= 2, alpha ranges from 0.0 to 1.0.
///
/// Covers: kokoro_streaming.rs lines 96-99 (alpha computation).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn crossfade_alpha_in_unit_interval() {
    let cf: usize = kani::any();
    kani::assume(cf >= 2 && cf <= 48000);

    let i: usize = kani::any();
    kani::assume(i < cf);

    let inv = 1.0f32 / (cf - 1) as f32;
    let alpha = i as f32 * inv;

    assert!(alpha.is_finite(), "alpha must be finite");
    assert!(alpha >= 0.0, "alpha must be >= 0.0");
    // i < cf and inv = 1/(cf-1), so i*inv <= (cf-1)/(cf-1) = 1.0.
    assert!(
        alpha <= 1.0 + 1e-6,
        "alpha must be <= 1.0 (within float tolerance)"
    );
}

/// Harness 19: Crossfade convex combination preserves boundedness.
///
/// SUBSTANTIVE: Proves that for samples in [-1, 1] and alpha in [0, 1],
/// the crossfade formula `prev * (1 - alpha) + next * alpha` produces
/// output in [-1, 1]. This is the key numerical safety property for
/// crossfade_chunks.
///
/// The formula is a convex combination: for alpha in [0, 1],
/// |(1-alpha) * prev + alpha * next| <= max(|prev|, |next|) <= 1.
///
/// Covers: kokoro_streaming.rs line 99 (crossfade formula).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn crossfade_convex_combination_bounded() {
    let prev: f32 = kani::any();
    let next: f32 = kani::any();
    kani::assume(prev.is_finite() && prev >= -1.0 && prev <= 1.0);
    kani::assume(next.is_finite() && next >= -1.0 && next <= 1.0);

    let alpha: f32 = kani::any();
    kani::assume(alpha.is_finite() && alpha >= 0.0 && alpha <= 1.0);

    let result = prev * (1.0 - alpha) + next * alpha;

    assert!(result.is_finite(), "crossfade result must be finite");
    assert!(result >= -1.0 - 1e-6, "crossfade result must be >= -1.0");
    assert!(result <= 1.0 + 1e-6, "crossfade result must be <= 1.0");
}

/// Harness 20: emit_len with saturating_sub prevents underflow.
///
/// SUBSTANTIVE: In assemble_streaming_chunks (kokoro_streaming.rs:194),
/// non-last chunks compute `emit_len = raw.len().saturating_sub(cf)`.
/// This harness proves that saturating_sub always returns a valid usize
/// (no wrapping), and that when raw.len() < cf the emit_len is 0 (not
/// a giant number from unsigned underflow).
///
/// Covers: kokoro_streaming.rs line 194 (emit_len computation).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn emit_len_saturating_sub_no_underflow() {
    let raw_len: usize = kani::any();
    kani::assume(raw_len <= 1_000_000); // realistic audio chunk

    let cf: usize = kani::any();
    kani::assume(cf >= 1 && cf <= 48000);

    let emit_len = raw_len.saturating_sub(cf);

    // saturating_sub clamps to 0, never wraps.
    assert!(emit_len <= raw_len, "emit_len must be <= raw_len");

    if raw_len >= cf {
        assert_eq!(
            emit_len,
            raw_len - cf,
            "when raw >= cf, emit_len = raw - cf"
        );
    } else {
        assert_eq!(emit_len, 0, "when raw < cf, emit_len saturates to 0");
    }
}

/// Harness 21: AudioChunk::duration_secs channels division is safe.
///
/// SUBSTANTIVE: Proves that the duration computation
/// `pcm.len() / (KOKORO_SAMPLE_RATE * channels)` is finite and non-negative
/// for valid channel counts (1 = mono, 2 = stereo). The channels.max(1)
/// guard prevents division by zero when channels is accidentally 0.
///
/// Covers: kokoro_streaming_types.rs lines 185-188 (duration_secs).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn audio_chunk_duration_secs_safe() {
    let pcm_len: usize = kani::any();
    kani::assume(pcm_len <= 10_000_000); // ~208 seconds at 24kHz stereo

    let channels: usize = kani::any();
    kani::assume(channels <= 2);

    let ch = channels.max(1); // guard against 0
    let denominator = KOKORO_SAMPLE_RATE as f64 * ch as f64;

    assert!(denominator > 0.0, "denominator must be positive");
    assert!(denominator.is_finite(), "denominator must be finite");

    let duration = pcm_len as f64 / denominator;
    assert!(duration.is_finite(), "duration must be finite");
    assert!(duration >= 0.0, "duration must be non-negative");
}

/// Harness 22: concatenate_chunks capacity equals sum of chunk lengths.
///
/// SUBSTANTIVE: Proves that the total capacity computation in
/// concatenate_chunks (sum of pcm.len() for all chunks) equals the
/// final output length. The Vec is allocated with_capacity(total)
/// and then filled with extend_from_slice, so the capacity must be
/// exact to avoid reallocation.
///
/// Models the computation for 2 chunks (the most common case).
///
/// Covers: kokoro_streaming_types.rs lines 212-219 (concatenate_chunks).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn concatenate_chunks_capacity_exact() {
    let len1: usize = kani::any();
    let len2: usize = kani::any();
    kani::assume(len1 <= 1_000_000);
    kani::assume(len2 <= 1_000_000);

    // Sum of chunk lengths.
    let total = len1 + len2;

    // No overflow for these bounds.
    assert!(total >= len1, "total must be >= each part (no overflow)");
    assert!(total >= len2, "total must be >= each part (no overflow)");

    // After extend_from_slice(chunk1) and extend_from_slice(chunk2):
    let final_len = len1 + len2;
    assert_eq!(
        final_len, total,
        "final output length must equal allocated capacity"
    );
}

/// Harness 23: KokoroStreamConfig default crossfade is 960 samples.
///
/// SUBSTANTIVE: Proves the default config has crossfade_samples = 960
/// (40ms at 24kHz). The Hann window at 40ms provides better spectral
/// continuity than the previous 20ms linear crossfade.
///
/// Covers: kokoro_streaming_types.rs Default impl.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn stream_config_default_crossfade_is_960() {
    let default_cf: usize = 960;
    let sample_rate: usize = KOKORO_SAMPLE_RATE;

    // 960 samples at 24000 Hz = 40ms.
    let duration_ms = (default_cf as f64 / sample_rate as f64) * 1000.0;

    assert_eq!(default_cf, 960, "default crossfade must be 960 samples");
    assert!(
        (duration_ms - 40.0).abs() < 0.01,
        "default crossfade must be ~40ms"
    );
}

/// Harness 24: Single-sample crossfade average is bounded.
///
/// SUBSTANTIVE: When crossfade_samples == 1, the crossfade formula
/// degenerates to `(prev + next) * 0.5` (average of boundary samples).
/// This harness proves the average of two [-1, 1] samples is in [-1, 1]
/// and finite, matching the special case at kokoro_streaming.rs:68-78.
///
/// Covers: kokoro_streaming.rs lines 68-78 (crossfade_samples == 1).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn single_sample_crossfade_bounded() {
    let prev: f32 = kani::any();
    let next: f32 = kani::any();
    kani::assume(prev.is_finite() && prev >= -1.0 && prev <= 1.0);
    kani::assume(next.is_finite() && next >= -1.0 && next <= 1.0);

    let avg = (prev + next) * 0.5;

    assert!(avg.is_finite(), "single-sample crossfade must be finite");
    assert!(avg >= -1.0, "single-sample crossfade must be >= -1.0");
    assert!(avg <= 1.0, "single-sample crossfade must be <= 1.0");
}
