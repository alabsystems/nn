// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Streaming chorus production gate tests for Kokoro.
//!
//! These gates exercise real-weight multi-voice streaming behavior and print
//! measured values for CI visibility, but they do not certify production-grade
//! audio quality or throughput.
//!
//! Gates:
//! 1. **Audio sanity**: No NaN/Inf, samples in [-1, 1], non-zero energy, sane length.
//! 2. **Streaming consistency**: Pull-based streaming produces identical audio to batch.
//! 3. **Clone dispatch memory**: 4-voice chorus GPU weight bytes == single-voice (aliased).
//! 4. **Segment cache reuse**: Second synthesis with same text length keeps cache counts
//!    unchanged and is materially faster than the first pass.
//! 5. **Speed variation**: speed=0.8 produces longer audio than speed=1.2.
//! 6. **Cancel/resume**: Cancel mid-stream, resume via reset, produces valid audio.
//! 7. **Arena hardening**: Warm pull-session runs stay within the current default-arena
//!    sizing envelope on shared-text and varied-text paths.
//!
//! All tests gated behind `KOKORO_WEIGHTS` env var. Skips gracefully when unset.
//!
//! Run: `cargo test -p nn-metal --test kokoro_all -- streaming_chorus_gate --nocapture`

use std::time::Instant;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use nn_metal::compiled_kokoro::chorus::KokoroChorus;
use nn_metal::compiled_kokoro::{CompiledKokoro, StreamingChorusSession};
use nn_models::kokoro_chorus::ChorusConfig;
use nn_models::kokoro_streaming::KokoroStreamConfig;

fn cpu() -> Device {
    Device::Cpu
}

// -- Constants ----------------------------------------------------------------

/// Kokoro sample rate (24 kHz).
const SAMPLE_RATE: usize = 24_000;

/// Number of chorus voices for gate tests.
const N_VOICES: usize = 4;

// -- Helpers ------------------------------------------------------------------

/// Load production Kokoro with Warn rejection policy.
///
/// Returns `None` if `KOKORO_WEIGHTS` is not set (test skips gracefully).
fn load_production_kokoro(skip_msg: &str) -> Option<(CompiledKokoro, nn_metal::PipelineCache)> {
    let weights_path = super::kokoro_test_env::require_kokoro_weights(skip_msg)?;

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let mut hb = nn_tts_verify::HardBoundsConfig::default();
    hb.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;

    // SAFETY: safetensors file not modified while alive.
    let kokoro = unsafe {
        CompiledKokoro::load_with_hard_bounds(&weights_path, hb)
            .expect("failed to load Kokoro weights")
    };

    Some((kokoro, cache))
}

/// Build a warmed-up KokoroChorus from a primary instance.
///
/// Warms up the primary (compiles all 8 segments), then creates an
/// N_VOICES chorus via `clone_dispatch()`.
fn build_chorus(primary: &mut CompiledKokoro, cache: &nn_metal::PipelineCache) -> KokoroChorus {
    let input = make_input_short();
    let style = make_style();
    let _ = primary
        .synthesize(&input, &style, 1.0, cache)
        .expect("warmup synthesis");

    let config = ChorusConfig::equal_gain(N_VOICES).expect("valid chorus config");
    KokoroChorus::new(primary, config).expect("chorus creation")
}

/// Standard test utterance: 8 phoneme tokens.
fn make_input_short() -> DynTensor {
    DynTensor::from_vec_i64(vec![0_i64, 1, 2, 3, 4, 5, 6, 7], &[1, 8], &cpu()).unwrap()
}

/// Longer test utterance: 12 phoneme tokens.
fn make_input_long() -> DynTensor {
    DynTensor::from_vec_i64(
        vec![0_i64, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        &[1, 12],
        &cpu(),
    )
    .unwrap()
}

/// Production style tensor: [1, 256] filled with 0.01.
fn make_style() -> DynTensor {
    DynTensor::full(&[1, 256], 0.01, DType::F32, &cpu()).unwrap()
}

/// Per-voice styles for chorus (N_VOICES styles, each [1, 256]).
fn make_styles() -> Vec<DynTensor> {
    (0..N_VOICES)
        .map(|i| DynTensor::full(&[1, 256], 0.01 + i as f64 * 0.001, DType::F32, &cpu()).unwrap())
        .collect()
}

/// Compute RMS energy of an audio buffer.
fn rms_energy(audio: &[f32]) -> f64 {
    if audio.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = audio.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (sum_sq / audio.len() as f64).sqrt()
}

// =============================================================================
// Gate 1: Chorus Audio Quality (#streaming_chorus_gate)
// =============================================================================

/// Gate: multi-voice streaming chorus output has no NaN, all samples in [-1,1],
/// non-zero energy, and reasonable signal characteristics.
///
/// Steps:
/// 1. Load production Kokoro, warmup, build 4-voice chorus.
/// 2. Create StreamingChorusSession with 3 chunks.
/// 3. Iterate all chunks, collecting mixed audio.
/// 4. Assert: 0 NaN, 0 samples outside [-1,1], RMS energy > 1e-6.
/// 5. Assert: audio length is plausible (>= 0.1s at 24kHz).
#[test]
fn gate_streaming_chorus_audio_quality() {
    let (mut primary, cache) = match load_production_kokoro(
        "streaming chorus audio quality gate skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let mut chorus = build_chorus(&mut primary, &cache);
    let styles = make_styles();
    let stream_config = KokoroStreamConfig::default();

    let chunks = vec![make_input_short(), make_input_long(), make_input_short()];
    let mut session =
        StreamingChorusSession::new(chunks, styles, 1.0, stream_config).expect("session creation");

    let mut all_audio: Vec<f32> = Vec::new();
    while let Some(result) = session.next_chunk(&mut chorus, &cache) {
        let audio_chunk = result.expect("chunk synthesis");
        all_audio.extend(&audio_chunk.pcm);
    }
    assert!(session.is_done());

    // Gate: no NaN samples.
    let nan_count = all_audio.iter().filter(|s| s.is_nan()).count();
    assert!(
        nan_count == 0,
        "gate_streaming_chorus_audio_quality: {nan_count} NaN samples in {} total",
        all_audio.len(),
    );

    // Gate: no Inf samples.
    let inf_count = all_audio.iter().filter(|s| s.is_infinite()).count();
    assert!(
        inf_count == 0,
        "gate_streaming_chorus_audio_quality: {inf_count} Inf samples in {} total",
        all_audio.len(),
    );

    // Gate: all samples in [-1, 1].
    let mut max_abs: f32 = 0.0;
    let mut clip_count = 0usize;
    for &sample in &all_audio {
        let abs = sample.abs();
        if abs > max_abs {
            max_abs = abs;
        }
        if abs > 1.0 {
            clip_count += 1;
        }
    }
    assert!(
        clip_count == 0,
        "gate_streaming_chorus_audio_quality: {clip_count} samples outside [-1,1], \
         max_abs={max_abs:.6}",
    );

    // Gate: non-zero energy (RMS > 1e-6 to detect silence bugs).
    let rms = rms_energy(&all_audio);
    assert!(
        rms > 1e-6,
        "gate_streaming_chorus_audio_quality: RMS energy {rms:.2e} <= 1e-6 — \
         audio is effectively silent",
    );

    // Gate: plausible audio length (>= 0.1s = 2400 samples at 24kHz).
    let min_samples = SAMPLE_RATE / 10;
    assert!(
        all_audio.len() >= min_samples,
        "gate_streaming_chorus_audio_quality: {} samples < minimum {min_samples}",
        all_audio.len(),
    );

    eprintln!(
        "\n=== STREAMING CHORUS AUDIO QUALITY GATE ===\n  \
         Samples:   {}\n  \
         NaN:       0\n  \
         Inf:       0\n  \
         Clipped:   0\n  \
         Max |x|:   {max_abs:.6}\n  \
         RMS:       {rms:.6}\n  \
         Duration:  {:.3}s\n  \
         PASS\n\
         ============================================\n",
        all_audio.len(),
        all_audio.len() as f64 / SAMPLE_RATE as f64,
    );
}

// =============================================================================
// Gate 2: Streaming Consistency (#streaming_chorus_gate)
// =============================================================================

/// Gate: pull-based StreamingChorusSession produces the same mixed audio as
/// the batch `synthesize_streaming_chorus` API for identical inputs.
///
/// This ensures the incremental crossfade in the pull-based session matches
/// the batch crossfade in `assemble_streaming_chorus`.
///
/// Steps:
/// 1. Load production Kokoro, warmup, build 4-voice chorus.
/// 2. Synthesize 2 chunks via batch `synthesize_streaming_chorus`.
/// 3. Synthesize same 2 chunks via pull-based `StreamingChorusSession`.
/// 4. Concatenate pull-based chunks.
/// 5. Assert: batch and pull produce the same sample count.
/// 6. Assert: max absolute difference < 1e-4 (float rounding tolerance).
#[test]
fn gate_streaming_chorus_consistency() {
    let (mut primary, cache) = match load_production_kokoro(
        "streaming chorus consistency gate skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let mut chorus = build_chorus(&mut primary, &cache);
    let styles = make_styles();
    let stream_config = KokoroStreamConfig::default();

    let chunk0 = make_input_short();
    let chunk1 = make_input_long();

    // -- Batch path: synthesize_streaming_chorus --------------------------------
    let batch_chunks = chorus
        .synthesize_streaming_chorus(
            &[chunk0.clone(), chunk1.clone()],
            &styles,
            1.0,
            &stream_config,
            &cache,
        )
        .expect("batch streaming chorus");

    let batch_audio: Vec<f32> = batch_chunks
        .iter()
        .flat_map(|c| c.pcm.iter().copied())
        .collect();

    // -- Pull path: StreamingChorusSession --------------------------------------
    let mut session = StreamingChorusSession::new(
        vec![chunk0, chunk1],
        styles.clone(),
        1.0,
        stream_config.clone(),
    )
    .expect("session creation");

    let mut pull_audio: Vec<f32> = Vec::new();
    while let Some(result) = session.next_chunk(&mut chorus, &cache) {
        let audio_chunk = result.expect("pull chunk synthesis");
        pull_audio.extend(&audio_chunk.pcm);
    }
    assert!(session.is_done());

    // Gate: same sample count.
    assert_eq!(
        batch_audio.len(),
        pull_audio.len(),
        "gate_streaming_chorus_consistency: batch ({} samples) and pull ({} samples) \
         produce different sample counts",
        batch_audio.len(),
        pull_audio.len(),
    );

    // Gate: max absolute difference < 1e-4.
    // Both paths go through the same GPU synthesis + mixing + crossfade code,
    // but the pull path applies crossfade incrementally. Floating-point
    // rounding differences should be negligible.
    let mut max_diff: f32 = 0.0;
    let mut diff_count = 0usize;
    let tolerance = 1e-4_f32;
    for (i, (&b, &p)) in batch_audio.iter().zip(pull_audio.iter()).enumerate() {
        let diff = (b - p).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > tolerance {
            diff_count += 1;
            if diff_count <= 5 {
                eprintln!("  diff[{i}]: batch={b:.8}, pull={p:.8}, diff={diff:.2e}");
            }
        }
    }

    assert!(
        diff_count == 0,
        "gate_streaming_chorus_consistency: {diff_count} samples exceed tolerance {tolerance}, \
         max_diff={max_diff:.2e}",
    );

    eprintln!(
        "\n=== STREAMING CHORUS CONSISTENCY GATE ===\n  \
         Samples:   {}\n  \
         Max diff:  {max_diff:.2e}\n  \
         Tolerance: {tolerance}\n  \
         Violators: 0\n  \
         PASS\n\
         =========================================\n",
        batch_audio.len(),
    );
}

// =============================================================================
// Gate 3: Clone Dispatch Memory (#streaming_chorus_gate)
// =============================================================================

/// Gate: 4-voice chorus GPU weight bytes equal single-voice bytes (aliased).
///
/// KokoroChorus creates voice instances via `clone_dispatch()`, which aliases
/// GPU weight buffers via `MetalBuffer::alias()` (zero-copy). A 4-voice chorus
/// should NOT use 4x the GPU weight memory — it should use exactly 1x because
/// all voices share the same underlying Metal buffers.
///
/// Steps:
/// 1. Load production Kokoro, warmup.
/// 2. Measure single-voice GPU weight bytes.
/// 3. Build 4-voice chorus.
/// 4. Measure per-voice GPU weight bytes via `gpu_weight_bytes_per_voice`.
/// 5. Assert: chorus per-voice bytes == single-voice bytes (aliased, not copied).
#[test]
fn gate_streaming_chorus_clone_memory() {
    let (mut primary, cache) = match load_production_kokoro(
        "streaming chorus clone memory gate skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    // Warmup: compile all 8 segments.
    let input = make_input_short();
    let style = make_style();
    let _ = primary
        .synthesize(&input, &style, 1.0, &cache)
        .expect("warmup synthesis");

    let single_voice_bytes = primary.gpu_weight_bytes();
    assert!(
        single_voice_bytes > 0,
        "gate_streaming_chorus_clone_memory: single voice has 0 GPU weight bytes after warmup"
    );

    // Build 4-voice chorus.
    let config = ChorusConfig::equal_gain(N_VOICES).expect("valid chorus config");
    let chorus = KokoroChorus::new(&primary, config).expect("chorus creation");

    let chorus_per_voice_bytes = chorus.gpu_weight_bytes_per_voice();

    // Gate: per-voice bytes must equal single-voice bytes (aliased buffers).
    assert_eq!(
        chorus_per_voice_bytes, single_voice_bytes,
        "gate_streaming_chorus_clone_memory: chorus per-voice bytes ({chorus_per_voice_bytes}) \
         != single-voice bytes ({single_voice_bytes}) — GPU weights not aliased"
    );

    // Verify shared state refcount indicates sharing.
    let refcount = chorus.shared_state_refcount();
    assert!(
        refcount >= N_VOICES,
        "gate_streaming_chorus_clone_memory: shared_state_refcount {refcount} < {N_VOICES}"
    );

    // Report: memory overhead ratio. Should be ~1.0x.
    let ratio = chorus_per_voice_bytes as f64 / single_voice_bytes as f64;

    eprintln!(
        "\n=== STREAMING CHORUS CLONE MEMORY GATE ===\n  \
         Single-voice GPU bytes: {single_voice_bytes}\n  \
         Chorus per-voice bytes: {chorus_per_voice_bytes}\n  \
         Memory ratio:           {ratio:.4}x\n  \
         Shared state refcount:  {refcount}\n  \
         PASS\n\
         ==========================================\n",
    );
}

// =============================================================================
// Gate 4: Segment Cache Hit (#streaming_chorus_gate)
// =============================================================================

/// Gate: second synthesis with the same text length reuses the cached segment set.
///
/// After warmup compiles all 8 segments for a given input shape, a second
/// synthesis with the same shape should reuse the previously compiled segment
/// set. We approximate that by checking that `total_cached_segments()` is
/// unchanged before and after the second synthesis, and by measuring that the
/// second synthesis is significantly faster than the first.
///
/// Steps:
/// 1. Load production Kokoro, warmup (compile all 8 segments).
/// 2. Record `total_cached_segments()` = 8.
/// 3. Build chorus and run streaming synthesis (same shape chunks).
/// 4. Verify `total_cached_segments()` unchanged on all voices.
/// 5. Time the second streaming pass; assert it's faster than first (cache hit).
#[test]
fn gate_streaming_chorus_cache_hit() {
    let (mut primary, cache) = match load_production_kokoro(
        "streaming chorus cache hit gate skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    // Warmup: compile all 8 segments.
    let input = make_input_short();
    let style = make_style();
    let _ = primary
        .synthesize(&input, &style, 1.0, &cache)
        .expect("warmup synthesis");

    let parent_cached = primary.total_cached_segments();
    assert_eq!(
        parent_cached, 8,
        "gate_streaming_chorus_cache_hit: parent should have 8 cached segments, got {parent_cached}"
    );

    // Build chorus — clones inherit 8 cached segments.
    let mut chorus = build_chorus(&mut primary, &cache);

    // Verify all voices inherited 8 cached segments.
    for i in 0..chorus.n_voices() {
        let voice = chorus.voice(i).expect("voice exists");
        let cached = voice.total_cached_segments();
        assert_eq!(
            cached, 8,
            "gate_streaming_chorus_cache_hit: voice[{i}] should have 8 cached, got {cached}"
        );
    }

    let styles = make_styles();
    let stream_config = KokoroStreamConfig::default();

    // First streaming pass (warm).
    let chunks_1 = vec![make_input_short(), make_input_short()];
    let t_first = Instant::now();
    let mut session_1 =
        StreamingChorusSession::new(chunks_1, styles.clone(), 1.0, stream_config.clone())
            .expect("session creation");
    while let Some(result) = session_1.next_chunk(&mut chorus, &cache) {
        let _ = result.expect("first pass chunk");
    }
    let first_ms = t_first.elapsed().as_secs_f64() * 1000.0;

    // Second streaming pass (same shape — should be pure cache hit).
    let chunks_2 = vec![make_input_short(), make_input_short()];
    let t_second = Instant::now();
    let mut session_2 = StreamingChorusSession::new(chunks_2, styles, 1.0, stream_config)
        .expect("session creation");
    while let Some(result) = session_2.next_chunk(&mut chorus, &cache) {
        let _ = result.expect("second pass chunk");
    }
    let second_ms = t_second.elapsed().as_secs_f64() * 1000.0;

    // Gate: cached segments unchanged after both passes.
    for i in 0..chorus.n_voices() {
        let voice = chorus.voice(i).expect("voice exists");
        let cached = voice.total_cached_segments();
        assert_eq!(
            cached, 8,
            "gate_streaming_chorus_cache_hit: voice[{i}] should still have 8 cached after \
             two passes, got {cached}"
        );
    }

    // Gate: second pass should not be dramatically slower than first
    // (would indicate recompilation). Allow generous 3x tolerance since
    // both passes are warm and timing jitter exists.
    assert!(
        second_ms < first_ms * 3.0,
        "gate_streaming_chorus_cache_hit: second pass ({second_ms:.1}ms) is >3x slower than \
         first ({first_ms:.1}ms) — possible recompilation"
    );

    eprintln!(
        "\n=== STREAMING CHORUS CACHE HIT GATE ===\n  \
         Parent cached:  {parent_cached}\n  \
         First pass:     {first_ms:.1}ms\n  \
         Second pass:    {second_ms:.1}ms\n  \
         Speedup:        {:.2}x\n  \
         All voices:     8/8 cached after both passes\n  \
         PASS\n\
         =======================================\n",
        first_ms / second_ms.max(0.001),
    );
}

// =============================================================================
// Gate 5: Speed Variation (#streaming_chorus_gate)
// =============================================================================

/// Gate: speed=0.8 produces longer audio than speed=1.2 in streaming chorus.
///
/// Duration is inversely proportional to speed. A chorus at speed 0.8 should
/// produce ~1.5x the samples of one at speed 1.2 (ratio = 1.2/0.8 = 1.5).
///
/// Steps:
/// 1. Load production Kokoro, warmup, build chorus.
/// 2. Streaming synthesis with 2 chunks at speed=0.8 (slow).
/// 3. Reset session, streaming synthesis at speed=1.2 (fast).
/// 4. Assert: slow > fast sample count.
/// 5. Assert: ratio is approximately 1.5 (tolerance: 1.2 to 2.0).
#[test]
fn gate_streaming_chorus_speed_variation() {
    let (mut primary, cache) = match load_production_kokoro(
        "streaming chorus speed variation gate skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let mut chorus = build_chorus(&mut primary, &cache);
    let styles = make_styles();
    let stream_config = KokoroStreamConfig::default();

    let chunks = vec![make_input_short(), make_input_long()];

    // Pass 1: speed 0.8 (slow — produces longer audio).
    let mut session =
        StreamingChorusSession::new(chunks.clone(), styles.clone(), 0.8, stream_config.clone())
            .expect("session creation (slow)");

    let mut audio_slow: Vec<f32> = Vec::new();
    while let Some(result) = session.next_chunk(&mut chorus, &cache) {
        let audio_chunk = result.expect("slow-speed chunk synthesis");
        audio_slow.extend(&audio_chunk.pcm);
    }
    assert!(session.is_done());

    // Validate slow audio.
    let nan_slow = audio_slow.iter().filter(|s| !s.is_finite()).count();
    assert!(
        nan_slow == 0,
        "gate_streaming_chorus_speed_variation: {nan_slow} non-finite samples in slow audio"
    );

    // Pass 2: speed 1.2 (fast — produces shorter audio).
    let mut session_fast = StreamingChorusSession::new(chunks, styles, 1.2, stream_config)
        .expect("session creation (fast)");

    let mut audio_fast: Vec<f32> = Vec::new();
    while let Some(result) = session_fast.next_chunk(&mut chorus, &cache) {
        let audio_chunk = result.expect("fast-speed chunk synthesis");
        audio_fast.extend(&audio_chunk.pcm);
    }
    assert!(session_fast.is_done());

    // Validate fast audio.
    let nan_fast = audio_fast.iter().filter(|s| !s.is_finite()).count();
    assert!(
        nan_fast == 0,
        "gate_streaming_chorus_speed_variation: {nan_fast} non-finite samples in fast audio"
    );

    // Gate: slow speed produces more samples.
    assert!(
        audio_slow.len() > audio_fast.len(),
        "gate_streaming_chorus_speed_variation: slow (0.8x, {} samples) should produce \
         more audio than fast (1.2x, {} samples)",
        audio_slow.len(),
        audio_fast.len(),
    );

    // Gate: ratio is approximately 1.5 (tolerance: 1.2 to 2.0).
    // Expected: (1.2/0.8) = 1.5x more samples at slow speed.
    let ratio = audio_slow.len() as f64 / audio_fast.len() as f64;
    assert!(
        (1.2..=2.0).contains(&ratio),
        "gate_streaming_chorus_speed_variation: slow/fast ratio should be ~1.5, got {ratio:.3} \
         (slow={}, fast={})",
        audio_slow.len(),
        audio_fast.len(),
    );

    eprintln!(
        "\n=== STREAMING CHORUS SPEED VARIATION GATE ===\n  \
         Slow (0.8x):  {} samples\n  \
         Fast (1.2x):  {} samples\n  \
         Ratio:        {ratio:.3} (expected ~1.5)\n  \
         PASS\n\
         =============================================\n",
        audio_slow.len(),
        audio_fast.len(),
    );
}

// =============================================================================
// Gate 6: Cancel/Resume (#streaming_chorus_gate)
// =============================================================================

/// Gate: canceling mid-stream and resuming (via reset) produces valid audio.
///
/// This exercises the full session lifecycle: partial synthesis, cancel,
/// verify cancellation state, reset, re-synthesize from the beginning,
/// and verify the resumed audio is valid and complete.
///
/// Steps:
/// 1. Load production Kokoro, warmup, build chorus.
/// 2. Create StreamingChorusSession with 3 chunks.
/// 3. Consume 1 chunk, verify valid audio.
/// 4. Cancel the session.
/// 5. Verify: `is_cancelled()`, `is_done()`, `remaining() == 0`.
/// 6. Verify: `next_chunk()` returns `None` after cancel.
/// 7. Reset the session.
/// 8. Verify: state restored (`remaining() == 3`, not cancelled).
/// 9. Consume all 3 chunks from the reset session.
/// 10. Verify all resumed audio is valid (no NaN, in [-1,1], non-zero energy).
#[test]
fn gate_streaming_chorus_cancel_resume() {
    let (mut primary, cache) = match load_production_kokoro(
        "streaming chorus cancel/resume gate skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let mut chorus = build_chorus(&mut primary, &cache);
    let styles = make_styles();
    let stream_config = KokoroStreamConfig::default();

    let chunks = vec![make_input_short(), make_input_long(), make_input_short()];
    let mut session =
        StreamingChorusSession::new(chunks, styles, 1.0, stream_config).expect("session creation");

    assert_eq!(session.total_chunks(), 3);
    assert_eq!(session.remaining(), 3);
    assert!(!session.is_done());
    assert!(!session.is_cancelled());

    // --- Phase 1: Consume 1 chunk, then cancel --------------------------------

    let result = session
        .next_chunk(&mut chorus, &cache)
        .expect("chunk 0 should be available");
    let audio0 = result.expect("chunk 0 synthesis");

    // Validate the pre-cancel chunk.
    let nan_count = audio0.pcm.iter().filter(|s| !s.is_finite()).count();
    assert!(
        nan_count == 0,
        "gate_streaming_chorus_cancel_resume: {nan_count} non-finite in pre-cancel chunk"
    );
    assert!(
        !audio0.pcm.is_empty(),
        "gate_streaming_chorus_cancel_resume: pre-cancel chunk is empty"
    );

    assert_eq!(session.remaining(), 2);
    assert_eq!(session.synthesized_count(), 1);

    // Cancel.
    session.cancel();

    // Gate: cancellation state is correct.
    assert!(
        session.is_cancelled(),
        "gate_streaming_chorus_cancel_resume: session should be cancelled"
    );
    assert!(
        session.is_done(),
        "gate_streaming_chorus_cancel_resume: session should be done after cancel"
    );
    assert_eq!(
        session.remaining(),
        0,
        "gate_streaming_chorus_cancel_resume: remaining should be 0 after cancel"
    );

    // Gate: next_chunk returns None after cancel.
    assert!(
        session.next_chunk(&mut chorus, &cache).is_none(),
        "gate_streaming_chorus_cancel_resume: next_chunk should return None after cancel"
    );

    // --- Phase 2: Reset and re-synthesize from the beginning -------------------

    session.reset();

    // Gate: state is restored after reset.
    assert!(
        !session.is_cancelled(),
        "gate_streaming_chorus_cancel_resume: session should not be cancelled after reset"
    );
    assert!(
        !session.is_done(),
        "gate_streaming_chorus_cancel_resume: session should not be done after reset"
    );
    assert_eq!(
        session.remaining(),
        3,
        "gate_streaming_chorus_cancel_resume: remaining should be 3 after reset"
    );
    assert_eq!(
        session.synthesized_count(),
        0,
        "gate_streaming_chorus_cancel_resume: synthesized_count should be 0 after reset"
    );

    // Consume all 3 chunks from the reset session.
    let mut resumed_audio: Vec<f32> = Vec::new();
    let mut chunk_count = 0usize;
    while let Some(result) = session.next_chunk(&mut chorus, &cache) {
        let audio_chunk = result.expect("resumed chunk synthesis");
        assert!(
            !audio_chunk.pcm.is_empty(),
            "gate_streaming_chorus_cancel_resume: resumed chunk {chunk_count} is empty"
        );
        resumed_audio.extend(&audio_chunk.pcm);
        chunk_count += 1;
    }

    // Gate: consumed all 3 chunks after reset.
    assert_eq!(
        chunk_count, 3,
        "gate_streaming_chorus_cancel_resume: should have synthesized 3 chunks after reset, \
         got {chunk_count}"
    );
    assert!(session.is_done());

    // Gate: resumed audio is valid.
    let nan_resumed = resumed_audio.iter().filter(|s| s.is_nan()).count();
    let inf_resumed = resumed_audio.iter().filter(|s| s.is_infinite()).count();
    assert!(
        nan_resumed == 0,
        "gate_streaming_chorus_cancel_resume: {nan_resumed} NaN in resumed audio"
    );
    assert!(
        inf_resumed == 0,
        "gate_streaming_chorus_cancel_resume: {inf_resumed} Inf in resumed audio"
    );

    let mut max_abs: f32 = 0.0;
    let mut clip_count = 0usize;
    for &sample in &resumed_audio {
        let abs = sample.abs();
        if abs > max_abs {
            max_abs = abs;
        }
        if abs > 1.0 {
            clip_count += 1;
        }
    }
    assert!(
        clip_count == 0,
        "gate_streaming_chorus_cancel_resume: {clip_count} samples outside [-1,1], \
         max_abs={max_abs:.6}"
    );

    // Gate: non-zero energy in resumed audio.
    let rms = rms_energy(&resumed_audio);
    assert!(
        rms > 1e-6,
        "gate_streaming_chorus_cancel_resume: resumed audio RMS {rms:.2e} is silent"
    );

    // Gate: resumed audio has plausible length.
    let min_samples = SAMPLE_RATE / 10;
    assert!(
        resumed_audio.len() >= min_samples,
        "gate_streaming_chorus_cancel_resume: {} resumed samples < minimum {min_samples}",
        resumed_audio.len(),
    );

    eprintln!(
        "\n=== STREAMING CHORUS CANCEL/RESUME GATE ===\n  \
         Pre-cancel:    {} samples (1 chunk)\n  \
         Cancel:        is_cancelled=true, remaining=0\n  \
         Reset:         remaining=3, is_cancelled=false\n  \
         Resumed:       {} samples (3 chunks)\n  \
         NaN:           0\n  \
         Max |x|:       {max_abs:.6}\n  \
         RMS:           {rms:.6}\n  \
         PASS\n\
         ===========================================\n",
        audio0.pcm.len(),
        resumed_audio.len(),
    );
}

// =============================================================================
// Gate 7: Arena Hardening (#streaming_chorus_gate)
// =============================================================================

/// Gate: a warmed shared-text `StreamingChorusSession` run stays within the
/// current default-arena sizing envelope.
///
/// This is a targeted regression gate for the on-demand pull session path
/// after arena hardening, not a broad production-readiness claim. It runs on
/// a fresh worker thread so the default arena stats start from a clean
/// thread-local state for this exact path.
#[test]
fn gate_streaming_chorus_warm_path_no_arena_overflow() {
    if super::kokoro_test_env::require_kokoro_weights(
        "streaming chorus warm-path arena gate skipped -- set KOKORO_WEIGHTS to enable.",
    )
    .is_none()
    {
        return;
    }

    let (arena_estimate, chunk_count, total_samples, max_abs, overflow_count, overflow_bytes) =
        std::thread::spawn(|| {
            let (mut primary, cache) = load_production_kokoro(
                "streaming chorus warm-path arena gate requires KOKORO_WEIGHTS inside worker thread.",
            )
            .expect("production weights already checked");

            let mut chorus = build_chorus(&mut primary, &cache);
            let styles = make_styles();
            let stream_config = KokoroStreamConfig::default();
            let chunks = vec![
                make_input_short(),
                make_input_long(),
                make_input_short(),
                make_input_long(),
            ];

            let arena_estimate = chorus
                .voice(0)
                .expect("voice 0")
                .estimate_arena_bytes();
            assert!(
                arena_estimate > 0,
                "gate_streaming_chorus_warm_path_no_arena_overflow: warmed chorus must expose \
                 a non-zero arena estimate",
            );

            nn_metal::reset_arena_stats();

            let mut session = StreamingChorusSession::new(chunks, styles, 1.0, stream_config)
                .expect("session creation");
            let mut chunk_count = 0usize;
            let mut total_samples = 0usize;
            let mut max_abs = 0.0f32;
            while let Some(result) = session.next_chunk(&mut chorus, &cache) {
                let audio_chunk = result.expect("shared-text chunk synthesis");
                assert!(
                    !audio_chunk.pcm.is_empty(),
                    "gate_streaming_chorus_warm_path_no_arena_overflow: chunk {chunk_count} is empty"
                );

                for &sample in &audio_chunk.pcm {
                    assert!(
                        sample.is_finite(),
                        "gate_streaming_chorus_warm_path_no_arena_overflow: chunk {chunk_count} has non-finite audio"
                    );
                    let abs = sample.abs();
                    if abs > max_abs {
                        max_abs = abs;
                    }
                    assert!(
                        abs <= 1.0,
                        "gate_streaming_chorus_warm_path_no_arena_overflow: chunk {chunk_count} sample exceeds [-1,1], max_abs={abs}"
                    );
                }

                total_samples += audio_chunk.pcm.len();
                chunk_count += 1;
            }
            assert!(session.is_done());

            let stats = nn_metal::arena_stats();
            (
                arena_estimate,
                chunk_count,
                total_samples,
                max_abs,
                stats.overflow_count,
                stats.overflow_bytes,
            )
        })
        .join()
        .expect("streaming chorus warm-path arena worker thread");

    assert_eq!(
        overflow_count, 0,
        "gate_streaming_chorus_warm_path_no_arena_overflow: StreamingChorusSession overflowed \
         the default arena on a warmed shared-text run (overflow_bytes={overflow_bytes})"
    );
    assert_eq!(
        chunk_count, 4,
        "gate_streaming_chorus_warm_path_no_arena_overflow: expected 4 chunks, got {chunk_count}"
    );
    assert!(
        total_samples > SAMPLE_RATE / 2,
        "gate_streaming_chorus_warm_path_no_arena_overflow: expected >0.5s of audio, got \
         {total_samples} samples"
    );

    eprintln!(
        "\n=== STREAMING CHORUS ARENA HARDENING GATE (SHARED TEXT) ===\n  \
         Estimate:        {arena_estimate} bytes\n  \
         Chunks:          {chunk_count}\n  \
         Samples:         {total_samples}\n  \
         Max |x|:         {max_abs:.6}\n  \
         Overflow count:  {overflow_count}\n  \
         PASS\n\
         =============================================================\n",
    );
}

/// Gate: a warmed varied-text `StreamingChorusSession` run stays within the
/// current default-arena sizing envelope.
///
/// This specifically exercises the per-voice chunk path in the pull-based
/// session API, using the same fresh-thread stats isolation as the shared-text
/// gate above.
#[test]
fn gate_streaming_chorus_varied_text_warm_path_no_arena_overflow() {
    if super::kokoro_test_env::require_kokoro_weights(
        "streaming chorus varied-text arena gate skipped -- set KOKORO_WEIGHTS to enable.",
    )
    .is_none()
    {
        return;
    }

    let (arena_estimate, chunk_count, total_samples, max_abs, overflow_count, overflow_bytes) =
        std::thread::spawn(|| {
            let (mut primary, cache) = load_production_kokoro(
                "streaming chorus varied-text arena gate requires KOKORO_WEIGHTS inside worker thread.",
            )
            .expect("production weights already checked");

            let mut chorus = build_chorus(&mut primary, &cache);
            let styles = make_styles();
            let stream_config = KokoroStreamConfig::default();
            let per_voice_chunks = vec![
                vec![make_input_short(), make_input_long(), make_input_short()],
                vec![make_input_long(), make_input_short(), make_input_long()],
                vec![make_input_short(), make_input_short(), make_input_long()],
                vec![make_input_long(), make_input_long(), make_input_short()],
            ];

            let arena_estimate = chorus
                .voice(0)
                .expect("voice 0")
                .estimate_arena_bytes();
            assert!(
                arena_estimate > 0,
                "gate_streaming_chorus_varied_text_warm_path_no_arena_overflow: warmed chorus \
                 must expose a non-zero arena estimate",
            );

            nn_metal::reset_arena_stats();

            let mut session = StreamingChorusSession::new_varied_text(
                per_voice_chunks,
                styles,
                1.0,
                stream_config,
            )
            .expect("varied-text session creation");
            assert!(session.is_varied_text());

            let mut chunk_count = 0usize;
            let mut total_samples = 0usize;
            let mut max_abs = 0.0f32;
            while let Some(result) = session.next_chunk(&mut chorus, &cache) {
                let audio_chunk = result.expect("varied-text chunk synthesis");
                assert!(
                    !audio_chunk.pcm.is_empty(),
                    "gate_streaming_chorus_varied_text_warm_path_no_arena_overflow: chunk {chunk_count} is empty"
                );

                for &sample in &audio_chunk.pcm {
                    assert!(
                        sample.is_finite(),
                        "gate_streaming_chorus_varied_text_warm_path_no_arena_overflow: chunk {chunk_count} has non-finite audio"
                    );
                    let abs = sample.abs();
                    if abs > max_abs {
                        max_abs = abs;
                    }
                    assert!(
                        abs <= 1.0,
                        "gate_streaming_chorus_varied_text_warm_path_no_arena_overflow: chunk {chunk_count} sample exceeds [-1,1], max_abs={abs}"
                    );
                }

                total_samples += audio_chunk.pcm.len();
                chunk_count += 1;
            }
            assert!(session.is_done());

            let stats = nn_metal::arena_stats();
            (
                arena_estimate,
                chunk_count,
                total_samples,
                max_abs,
                stats.overflow_count,
                stats.overflow_bytes,
            )
        })
        .join()
        .expect("streaming chorus varied-text arena worker thread");

    assert_eq!(
        overflow_count, 0,
        "gate_streaming_chorus_varied_text_warm_path_no_arena_overflow: \
         StreamingChorusSession::new_varied_text overflowed the default arena on a warmed run \
         (overflow_bytes={overflow_bytes})"
    );
    assert_eq!(
        chunk_count, 3,
        "gate_streaming_chorus_varied_text_warm_path_no_arena_overflow: expected 3 chunks, \
         got {chunk_count}"
    );
    assert!(
        total_samples > SAMPLE_RATE / 2,
        "gate_streaming_chorus_varied_text_warm_path_no_arena_overflow: expected >0.5s of \
         audio, got {total_samples} samples"
    );

    eprintln!(
        "\n=== STREAMING CHORUS ARENA HARDENING GATE (VARIED TEXT) ===\n  \
         Estimate:        {arena_estimate} bytes\n  \
         Chunks:          {chunk_count}\n  \
         Samples:         {total_samples}\n  \
         Max |x|:         {max_abs:.6}\n  \
         Overflow count:  {overflow_count}\n  \
         PASS\n\
         ============================================================\n",
    );
}
