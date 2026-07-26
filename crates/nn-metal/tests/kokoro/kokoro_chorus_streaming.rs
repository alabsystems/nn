// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for Kokoro chorus synthesis using [`StreamingKokoroSession`]
//! and [`clone_dispatch()`].
//!
//! Exercises the pull-based streaming API with multi-voice chorus:
//! 1. Two voices with interleaved `next_chunk()` calls produce valid audio.
//! 2. Session state machine (remaining, is_done, reset) works correctly.
//! 3. `clone_dispatch()` shares compiled segments with no recompilation.
//!
//! Part of #4265.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use nn_metal::compiled_kokoro::{CompiledKokoro, StreamingKokoroSession};

use super::kokoro_test_weights as kw;

fn cpu() -> Device {
    Device::Cpu
}

const STYLE_DIM: usize = 4; // Must match kw::mini_test_config().style_dim

// -- Helpers ------------------------------------------------------------------

fn build_kokoro() -> (CompiledKokoro, nn_metal::PipelineCache) {
    kw::build_kokoro_mini()
}

fn make_style(seed: usize) -> DynTensor {
    DynTensor::new(
        &super::test_utils::rand_f32_vec(400 + seed as u64, 2 * STYLE_DIM, -0.1, 0.1),
        &[1, 2 * STYLE_DIM],
        &cpu(),
    )
    .unwrap()
}

fn make_input(len: usize) -> DynTensor {
    let vals: Vec<f32> = (1..=len).map(|v| v as f32).collect();
    DynTensor::from_vec(vals, &[1, len], &cpu()).unwrap()
}

// -- Tests --------------------------------------------------------------------

/// Two voices with interleaved `next_chunk()` calls produce valid audio.
///
/// Steps:
/// 1. Load CompiledKokoro (mini weights), synthesize once to warm up.
/// 2. `clone_dispatch()` to get a second voice.
/// 3. Create two `StreamingKokoroSession` instances with identical chunks
///    but different speeds (1.0 and 0.9).
/// 4. Interleave `next_chunk()` calls between voices (voice0 chunk0,
///    voice1 chunk0, voice0 chunk1, voice1 chunk1).
/// 5. Verify both produce valid audio (no NaN, finite).
/// 6. Verify both complete (`is_done()`).
#[test]
fn test_chorus_streaming_two_voices() {
    super::test_utils::gpu_init();
    let (mut primary, cache) = build_kokoro();

    let input = make_input(3);
    let style = make_style(0);

    // Warmup: compile all segments.
    let _ = primary
        .synthesize(&input, &style, 1.0, &cache)
        .expect("warmup");

    // Clone to get a second voice sharing weights.
    let mut voice1 = primary.clone_dispatch();

    // Prepare chunks for streaming sessions (2 chunks each).
    let chunk0_ids = make_input(3);
    let chunk1_ids = make_input(4);
    let style_a = make_style(10);
    let style_b = make_style(20);

    let chunks_voice0 = vec![
        (chunk0_ids.clone(), style_a.clone()),
        (chunk1_ids.clone(), style_a),
    ];
    let chunks_voice1 = vec![
        (chunk0_ids, style_b.clone()),
        (chunk1_ids, style_b),
    ];

    let mut session0 = StreamingKokoroSession::new(chunks_voice0, 1.0);
    let mut session1 = StreamingKokoroSession::new(chunks_voice1, 0.9);

    assert_eq!(session0.remaining(), 2);
    assert_eq!(session1.remaining(), 2);
    assert!(!session0.is_done());
    assert!(!session1.is_done());

    // Interleave: voice0 chunk0, voice1 chunk0, voice0 chunk1, voice1 chunk1.
    let mut all_audio_v0: Vec<f32> = Vec::new();
    let mut all_audio_v1: Vec<f32> = Vec::new();

    // Voice 0 - chunk 0
    let result = session0
        .next_chunk(&mut primary, &cache)
        .expect("session0 should have chunk 0");
    let (audio, _cert) = result.expect("session0 chunk 0 synthesis");
    let vals = audio.to_flat_vec::<f32>().unwrap();
    all_audio_v0.extend(&vals);

    // Voice 1 - chunk 0
    let result = session1
        .next_chunk(&mut voice1, &cache)
        .expect("session1 should have chunk 0");
    let (audio, _cert) = result.expect("session1 chunk 0 synthesis");
    let vals = audio.to_flat_vec::<f32>().unwrap();
    all_audio_v1.extend(&vals);

    assert_eq!(session0.remaining(), 1);
    assert_eq!(session1.remaining(), 1);

    // Voice 0 - chunk 1
    let result = session0
        .next_chunk(&mut primary, &cache)
        .expect("session0 should have chunk 1");
    let (audio, _cert) = result.expect("session0 chunk 1 synthesis");
    let vals = audio.to_flat_vec::<f32>().unwrap();
    all_audio_v0.extend(&vals);

    // Voice 1 - chunk 1
    let result = session1
        .next_chunk(&mut voice1, &cache)
        .expect("session1 should have chunk 1");
    let (audio, _cert) = result.expect("session1 chunk 1 synthesis");
    let vals = audio.to_flat_vec::<f32>().unwrap();
    all_audio_v1.extend(&vals);

    // Both sessions should be done.
    assert!(session0.is_done());
    assert!(session1.is_done());
    assert_eq!(session0.remaining(), 0);
    assert_eq!(session1.remaining(), 0);

    // next_chunk() after completion returns None.
    assert!(session0.next_chunk(&mut primary, &cache).is_none());
    assert!(session1.next_chunk(&mut voice1, &cache).is_none());

    // Validate audio: no NaN, all finite.
    assert!(
        !all_audio_v0.is_empty(),
        "voice 0 should produce non-empty audio"
    );
    assert!(
        !all_audio_v1.is_empty(),
        "voice 1 should produce non-empty audio"
    );

    for (vi, audio) in [("voice0", &all_audio_v0), ("voice1", &all_audio_v1)] {
        let nan_count = audio.iter().filter(|s| s.is_nan()).count();
        let inf_count = audio.iter().filter(|s| s.is_infinite()).count();
        assert!(nan_count == 0, "{vi}: {nan_count} NaN samples");
        assert!(inf_count == 0, "{vi}: {inf_count} Inf samples");
    }

    eprintln!(
        "test_chorus_streaming_two_voices: voice0={} samples, voice1={} samples",
        all_audio_v0.len(),
        all_audio_v1.len(),
    );
}

/// Session state machine: remaining counts down, is_done transitions, reset works.
///
/// Uses the mini model to drive real `next_chunk()` calls through 3 chunks,
/// verifying that `remaining()`, `is_done()`, `synthesized_count()`, and
/// `reset()` all transition correctly.
#[test]
fn test_streaming_session_state_machine() {
    super::test_utils::gpu_init();
    let (mut kokoro, cache) = build_kokoro();

    // Warmup to populate segment caches.
    let input = make_input(3);
    let style = make_style(0);
    let _ = kokoro
        .synthesize(&input, &style, 1.0, &cache)
        .expect("warmup");

    // Build 3 chunks for the session.
    let style_a = make_style(10);
    let chunks = vec![
        (make_input(3), style_a.clone()),
        (make_input(4), style_a.clone()),
        (make_input(3), style_a),
    ];
    let mut session = StreamingKokoroSession::new(chunks, 1.0);

    // Initial state: 3 chunks, not done.
    assert_eq!(session.total_chunks(), 3);
    assert_eq!(session.remaining(), 3);
    assert_eq!(session.synthesized_count(), 0);
    assert!(!session.is_done());
    assert!((session.speed() - 1.0).abs() < f32::EPSILON);

    // Consume chunk 0.
    let r = session.next_chunk(&mut kokoro, &cache);
    assert!(r.is_some(), "chunk 0 should be available");
    let _ = r.unwrap().expect("chunk 0 synthesis");
    assert_eq!(session.remaining(), 2);
    assert_eq!(session.synthesized_count(), 1);
    assert!(!session.is_done());

    // Consume chunk 1.
    let r = session.next_chunk(&mut kokoro, &cache);
    assert!(r.is_some(), "chunk 1 should be available");
    let _ = r.unwrap().expect("chunk 1 synthesis");
    assert_eq!(session.remaining(), 1);
    assert_eq!(session.synthesized_count(), 2);
    assert!(!session.is_done());

    // Consume chunk 2.
    let r = session.next_chunk(&mut kokoro, &cache);
    assert!(r.is_some(), "chunk 2 should be available");
    let _ = r.unwrap().expect("chunk 2 synthesis");
    assert_eq!(session.remaining(), 0);
    assert_eq!(session.synthesized_count(), 3);
    assert!(session.is_done());

    // No more chunks.
    assert!(session.next_chunk(&mut kokoro, &cache).is_none());

    // Reset brings us back to the beginning.
    session.reset();
    assert_eq!(session.remaining(), 3);
    assert_eq!(session.synthesized_count(), 0);
    assert!(!session.is_done());

    // Speed can be updated.
    session.set_speed(0.8);
    assert!((session.speed() - 0.8).abs() < f32::EPSILON);

    // After reset, we can consume again.
    let r = session.next_chunk(&mut kokoro, &cache);
    assert!(r.is_some(), "chunk 0 should be available after reset");
    let _ = r.unwrap().expect("chunk 0 synthesis after reset");
    assert_eq!(session.remaining(), 2);
    assert_eq!(session.synthesized_count(), 1);

    eprintln!("test_streaming_session_state_machine: all state transitions verified");
}

/// `clone_dispatch()` shares compiled segments -- no recompilation needed.
///
/// Steps:
/// 1. Load + warmup parent, verify `total_cached_segments() == 8`.
/// 2. `clone_dispatch()`, verify clone `total_cached_segments() == 8` immediately.
/// 3. Synthesize on both parent and clone, verify both succeed.
///
/// Uses miniaturized model (no production weights needed).
#[test]
fn test_chorus_shared_segments_no_recompile() {
    super::test_utils::gpu_init();
    let (mut parent, cache) = build_kokoro();

    let input = make_input(3);
    let style = make_style(0);

    // Warmup: compile all 8 segments in the parent.
    let (audio_parent, _cert) = parent
        .synthesize(&input, &style, 1.0, &cache)
        .expect("parent warmup");

    // After warmup, parent has 8 cached segments.
    let parent_cached = parent.total_cached_segments();
    assert_eq!(
        parent_cached, 8,
        "warmed parent should have 8 cached segments, got {parent_cached}"
    );

    // Clone: shares Arc-wrapped compiled segments.
    let mut clone = parent.clone_dispatch();

    // Clone should have 8 cached segments immediately -- no recompilation.
    let clone_cached = clone.total_cached_segments();
    assert_eq!(
        clone_cached, 8,
        "clone should have 8 cached segments immediately after clone_dispatch, got {clone_cached}"
    );

    // Synthesize on both -- should be cache hits, no new compilations.
    let (audio_clone, _cert) = clone
        .synthesize(&input, &style, 1.0, &cache)
        .expect("clone synthesize");

    let (audio_parent2, _cert) = parent
        .synthesize(&input, &style, 1.0, &cache)
        .expect("parent second synthesize");

    // Both should still have 8 segments (no new compilations).
    assert_eq!(
        parent.total_cached_segments(),
        8,
        "parent should still have 8 cached segments after second synthesis"
    );
    assert_eq!(
        clone.total_cached_segments(),
        8,
        "clone should still have 8 cached segments after synthesis"
    );

    // Both audio outputs should be valid and have the same shape.
    assert_eq!(
        audio_parent.dims(),
        audio_clone.dims(),
        "parent and clone audio shapes should match"
    );
    assert_eq!(
        audio_parent.dims(),
        audio_parent2.dims(),
        "parent repeated synthesis should produce same shape"
    );

    // Validate clone audio: no NaN, all finite.
    let clone_vals = audio_clone.to_flat_vec::<f32>().unwrap();
    assert!(!clone_vals.is_empty(), "clone audio must not be empty");
    let nan_count = clone_vals.iter().filter(|s| s.is_nan()).count();
    let inf_count = clone_vals.iter().filter(|s| s.is_infinite()).count();
    assert!(nan_count == 0, "clone audio has {nan_count} NaN samples");
    assert!(inf_count == 0, "clone audio has {inf_count} Inf samples");

    eprintln!(
        "test_chorus_shared_segments_no_recompile: parent_cached={parent_cached}, \
         clone_cached={clone_cached}, audio_shape={:?}, clone_samples={}",
        audio_clone.dims(),
        clone_vals.len(),
    );
}

// -- Production-weight tests (require KOKORO_WEIGHTS env var) -----------------

/// Production weights: two voices with interleaved streaming, different speeds.
///
/// Steps:
/// 1. Load production CompiledKokoro, synthesize once to warm up.
/// 2. `clone_dispatch()` to get a second voice.
/// 3. Create two `StreamingKokoroSession` instances with identical chunks
///    but different speeds (1.0 and 0.85).
/// 4. Interleave `next_chunk()` calls between voices.
/// 5. Verify both produce valid audio (no NaN, in range [-1, 1]).
/// 6. Verify both complete (`is_done()`).
///
/// Requires `KOKORO_WEIGHTS` env var.
/// Part of #4265.
#[test]
fn test_production_chorus_streaming_two_voices() {
    let weights_path = match super::kokoro_test_env::require_kokoro_weights(
        "production chorus streaming test skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(path) => path,
        None => return,
    };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Use Warn policy: test tokens produce click artifacts with production
    // weights that fail the no_clicks hard bound. Part of #4262.
    let mut hb = nn_tts_verify::HardBoundsConfig::default();
    hb.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;

    // SAFETY: safetensors file not modified while alive.
    let mut parent = unsafe {
        CompiledKokoro::load_with_hard_bounds(&weights_path, hb)
            .expect("failed to load Kokoro weights")
    };

    // Standard test utterance: 8 phoneme tokens.
    let input_ids =
        DynTensor::from_vec_i64(vec![0_i64, 1, 2, 3, 4, 5, 6, 7], &[1, 8], &cpu()).unwrap();
    let style = DynTensor::full(&[1, 256], 0.01, DType::F32, &cpu()).unwrap();

    // Warmup.
    let _ = parent
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("parent warmup");

    // Clone for second voice.
    let mut voice1 = parent.clone_dispatch();

    // Prepare chunks (2 chunks, same tokens, same style for simplicity).
    let chunks_v0 = vec![
        (input_ids.clone(), style.clone()),
        (input_ids.clone(), style.clone()),
    ];
    let chunks_v1 = vec![
        (input_ids.clone(), style.clone()),
        (input_ids.clone(), style.clone()),
    ];

    let mut session0 = StreamingKokoroSession::new(chunks_v0, 1.0);
    let mut session1 = StreamingKokoroSession::new(chunks_v1, 0.85);

    // Interleave next_chunk() calls.
    let mut all_audio_v0: Vec<f32> = Vec::new();
    let mut all_audio_v1: Vec<f32> = Vec::new();

    while !session0.is_done() || !session1.is_done() {
        if let Some(result) = session0.next_chunk(&mut parent, &cache) {
            let (audio, _cert) = result.expect("voice0 chunk synthesis");
            all_audio_v0.extend(audio.to_flat_vec::<f32>().unwrap());
        }
        if let Some(result) = session1.next_chunk(&mut voice1, &cache) {
            let (audio, _cert) = result.expect("voice1 chunk synthesis");
            all_audio_v1.extend(audio.to_flat_vec::<f32>().unwrap());
        }
    }

    assert!(session0.is_done());
    assert!(session1.is_done());

    // Validate audio: no NaN, all samples in [-1, 1].
    for (vi, audio) in [("voice0", &all_audio_v0), ("voice1", &all_audio_v1)] {
        assert!(!audio.is_empty(), "{vi} should produce non-empty audio");
        let mut nan_count = 0usize;
        let mut out_of_range = 0usize;
        let mut max_abs = 0.0f32;
        for &sample in audio.iter() {
            if sample.is_nan() {
                nan_count += 1;
            } else {
                let abs = sample.abs();
                if abs > max_abs {
                    max_abs = abs;
                }
                if abs > 1.0 {
                    out_of_range += 1;
                }
            }
        }
        assert!(nan_count == 0, "{vi}: {nan_count} NaN samples");
        assert!(
            out_of_range == 0,
            "{vi}: {out_of_range} samples outside [-1,1], max_abs={max_abs}"
        );
    }

    eprintln!(
        "test_production_chorus_streaming_two_voices: v0={} samples, v1={} samples",
        all_audio_v0.len(),
        all_audio_v1.len(),
    );
}

/// Production weights: `clone_dispatch()` shares all 8 cached segments immediately.
///
/// Steps:
/// 1. Load + warmup parent, assert `total_cached_segments() == 8`.
/// 2. `clone_dispatch()`, assert clone `total_cached_segments() == 8` (immediate).
/// 3. Synthesize on both, verify both succeed and produce valid audio.
///
/// Requires `KOKORO_WEIGHTS` env var.
/// Part of #4265.
#[test]
fn test_production_shared_segments_no_recompile() {
    let weights_path = match super::kokoro_test_env::require_kokoro_weights(
        "production shared segments test skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(path) => path,
        None => return,
    };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let mut hb = nn_tts_verify::HardBoundsConfig::default();
    hb.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;

    // SAFETY: safetensors file not modified while alive.
    let mut parent = unsafe {
        CompiledKokoro::load_with_hard_bounds(&weights_path, hb)
            .expect("failed to load Kokoro weights")
    };

    let input_ids =
        DynTensor::from_vec_i64(vec![0_i64, 1, 2, 3, 4, 5, 6, 7], &[1, 8], &cpu()).unwrap();
    let style = DynTensor::full(&[1, 256], 0.01, DType::F32, &cpu()).unwrap();

    // Warmup: compile all 8 segments.
    let _ = parent
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("parent warmup");

    let parent_cached = parent.total_cached_segments();
    assert_eq!(
        parent_cached, 8,
        "warmed parent should have 8 cached segments, got {parent_cached}"
    );

    // Clone.
    let mut clone = parent.clone_dispatch();

    let clone_cached = clone.total_cached_segments();
    assert_eq!(
        clone_cached, 8,
        "clone should have 8 cached segments immediately, got {clone_cached}"
    );

    // Synthesize on both.
    let (audio_parent, _) = parent
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("parent second synthesis");
    let (audio_clone, _) = clone
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("clone synthesis");

    // Shape must match.
    assert_eq!(
        audio_parent.dims(),
        audio_clone.dims(),
        "parent and clone audio shapes must match"
    );

    // Validate clone audio.
    let clone_vals = audio_clone.to_flat_vec::<f32>().unwrap();
    assert!(!clone_vals.is_empty(), "clone audio must not be empty");

    let nan_count = clone_vals.iter().filter(|s| s.is_nan()).count();
    assert!(nan_count == 0, "clone audio has {nan_count} NaN samples");

    let max_abs = clone_vals.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs <= 1.0 + 1e-6,
        "clone audio should be in [-1,1]: max_abs={max_abs}"
    );

    // Segments unchanged after synthesis.
    assert_eq!(parent.total_cached_segments(), 8);
    assert_eq!(clone.total_cached_segments(), 8);

    eprintln!(
        "test_production_shared_segments_no_recompile: parent_cached={parent_cached}, \
         clone_cached={clone_cached}, audio_shape={:?}, max_abs={max_abs:.6}",
        audio_clone.dims(),
    );
}
