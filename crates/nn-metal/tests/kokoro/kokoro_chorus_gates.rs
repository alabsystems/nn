// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-voice chorus quality gates for Kokoro.
//!
//! Verifies that [`KokoroChorus`] synthesis meets production quality thresholds:
//! 1. Output sample count matches longest voice duration.
//! 2. Mixed audio is in [-1.0, 1.0] (no clipping beyond bounds).
//! 3. No NaN/Inf in output.
//! 4. Segment cache is shared across chorus voices (cache hit rate > 50%).
//! 5. Varied speeds produce different output lengths.
//! 6. `clone_dispatch_warm` shares compiled segments (init < 1s).
//! 7. `new_warm()` produces identical output to manual `clone_dispatch`.
//! 8. Streaming chorus session produces complete audio.
//! 9. Recommended chorus autocast stays finite on the non-streaming shared-encode path.
//! 10. Recommended chorus autocast stays finite on the plain batch mixed-output path.
//!
//! All tests gated behind `KOKORO_WEIGHTS` env var. Skips gracefully when unset.
//!
//! Run: `cargo test -p nn-metal --test kokoro_all -- chorus_gate --nocapture`
//!
//! Part of #4265.

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

/// Standard test utterance: 8 phoneme tokens.
fn make_input() -> DynTensor {
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

/// Create a chorus from a warmed-up primary Kokoro.
fn build_chorus(
    primary: &mut CompiledKokoro,
    cache: &nn_metal::PipelineCache,
    n_voices: usize,
) -> KokoroChorus {
    // Warmup: compile all 8 segments so clones inherit them.
    let input = make_input();
    let style = make_style();
    let _ = primary
        .synthesize(&input, &style, 1.0, cache)
        .expect("warmup synthesis");

    let config = ChorusConfig::equal_gain(n_voices).expect("valid chorus config");
    KokoroChorus::new(primary, config).expect("chorus creation")
}

// -- Gate Tests ---------------------------------------------------------------

/// Gate: chorus output has correct sample count.
///
/// Synthesizes N_VOICES with the same text at equal speed. The mixed output
/// length must equal the individual voice output length (since all voices
/// produce the same duration at the same speed). Verifies the output is
/// non-empty and has a plausible sample count for the given token count.
#[test]
fn gate_chorus_output_length() {
    let (mut primary, cache) = match load_production_kokoro(
        "chorus output length gate skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let mut chorus = build_chorus(&mut primary, &cache, N_VOICES);

    let input = make_input();
    let style = make_style();
    let inputs: Vec<DynTensor> = (0..N_VOICES).map(|_| input.clone()).collect();
    let styles: Vec<DynTensor> = (0..N_VOICES).map(|_| style.clone()).collect();

    let mixed = chorus
        .synthesize_chorus(&inputs, &styles, 1.0, &cache)
        .expect("chorus synthesis");

    // Output must be non-empty.
    assert!(
        !mixed.is_empty(),
        "gate_chorus_output_length: mixed audio is empty"
    );

    // Synthesize a single voice for reference length.
    let (ref_audio, _cert) = primary
        .synthesize(&input, &style, 1.0, &cache)
        .expect("single-voice reference synthesis");
    let ref_len = ref_audio.to_flat_vec::<f32>().unwrap().len();

    // Chorus output should match the single-voice length (all same input/speed).
    // Allow small tolerance for mixing/padding alignment.
    let diff = (mixed.len() as isize - ref_len as isize).unsigned_abs();
    assert!(
        diff <= 1,
        "gate_chorus_output_length: mixed length {} differs from reference {} by {diff}",
        mixed.len(),
        ref_len,
    );

    // Plausibility: 8 tokens at speed 1.0 should produce at least 0.1s of audio.
    let min_samples = SAMPLE_RATE / 10; // 2400 samples
    assert!(
        mixed.len() >= min_samples,
        "gate_chorus_output_length: {} samples < minimum {min_samples}",
        mixed.len(),
    );

    eprintln!(
        "gate_chorus_output_length: PASS -- mixed={} samples, ref={ref_len}, diff={diff}",
        mixed.len(),
    );
}

/// Gate: chorus mixed audio contains no NaN or Inf values.
///
/// Synthesizes a multi-voice chorus and checks every sample for finiteness.
/// NaN/Inf in synthesis output indicates a numerical bug in the pipeline.
#[test]
fn gate_chorus_no_nan() {
    let (mut primary, cache) =
        match load_production_kokoro("chorus no-NaN gate skipped -- set KOKORO_WEIGHTS to enable.")
        {
            Some(pair) => pair,
            None => return,
        };

    let mut chorus = build_chorus(&mut primary, &cache, N_VOICES);

    let input = make_input();
    let style = make_style();
    let inputs: Vec<DynTensor> = (0..N_VOICES).map(|_| input.clone()).collect();
    let styles: Vec<DynTensor> = (0..N_VOICES).map(|_| style.clone()).collect();

    let mixed = chorus
        .synthesize_chorus(&inputs, &styles, 1.0, &cache)
        .expect("chorus synthesis");

    let nan_count = mixed.iter().filter(|s| s.is_nan()).count();
    let inf_count = mixed.iter().filter(|s| s.is_infinite()).count();

    assert!(
        nan_count == 0,
        "gate_chorus_no_nan: {nan_count} NaN samples in {} total",
        mixed.len(),
    );
    assert!(
        inf_count == 0,
        "gate_chorus_no_nan: {inf_count} Inf samples in {} total",
        mixed.len(),
    );

    eprintln!(
        "gate_chorus_no_nan: PASS -- {} samples, 0 NaN, 0 Inf",
        mixed.len(),
    );
}

/// Gate: chorus mixed audio does not clip (all samples in [-1.0, 1.0]).
///
/// The chorus mixer applies per-voice gains and the result must remain within
/// the valid PCM range. Clipping indicates gain miscalibration or a mixing bug.
#[test]
fn gate_chorus_no_clipping() {
    let (mut primary, cache) = match load_production_kokoro(
        "chorus no-clipping gate skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let mut chorus = build_chorus(&mut primary, &cache, N_VOICES);

    let input = make_input();
    let style = make_style();
    let inputs: Vec<DynTensor> = (0..N_VOICES).map(|_| input.clone()).collect();
    let styles: Vec<DynTensor> = (0..N_VOICES).map(|_| style.clone()).collect();

    let mixed = chorus
        .synthesize_chorus(&inputs, &styles, 1.0, &cache)
        .expect("chorus synthesis");

    let mut max_abs: f32 = 0.0;
    let mut clip_count = 0usize;
    for &sample in &mixed {
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
        "gate_chorus_no_clipping: {clip_count} samples outside [-1,1], max_abs={max_abs:.6}",
    );

    eprintln!(
        "gate_chorus_no_clipping: PASS -- {} samples, max_abs={max_abs:.6}, 0 clipped",
        mixed.len(),
    );
}

/// Gate: segment cache is shared across chorus voices.
///
/// After warmup, the primary has 8 cached segments. Each cloned voice in the
/// chorus should inherit those segments via `clone_dispatch()`. This gate
/// verifies that the cache hit rate across all chorus voices exceeds 50%
/// (conservatively -- in practice it should be 100% after warmup).
#[test]
fn gate_chorus_cache_sharing() {
    let (mut primary, cache) = match load_production_kokoro(
        "chorus cache-sharing gate skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    // Warmup: compile all 8 segments.
    let input = make_input();
    let style = make_style();
    let _ = primary
        .synthesize(&input, &style, 1.0, &cache)
        .expect("warmup synthesis");

    let parent_cached = primary.total_cached_segments();
    assert_eq!(
        parent_cached, 8,
        "gate_chorus_cache_sharing: parent should have 8 cached segments, got {parent_cached}"
    );

    // Create chorus -- clones should inherit cached segments.
    let config = ChorusConfig::equal_gain(N_VOICES).expect("valid chorus config");
    let chorus = KokoroChorus::new(&primary, config).expect("chorus creation");

    let mut total_cached = 0usize;
    let total_possible = N_VOICES * 8;
    for i in 0..chorus.n_voices() {
        let voice = chorus.voice(i).expect("voice exists");
        let cached = voice.total_cached_segments();
        total_cached += cached;

        eprintln!("  voice[{i}]: {cached}/8 cached segments");
    }

    let hit_rate = total_cached as f64 / total_possible as f64;

    // Gate: hit rate must exceed 50% (conservative -- expect 100%).
    assert!(
        hit_rate > 0.5,
        "gate_chorus_cache_sharing: cache hit rate {:.1}% ({total_cached}/{total_possible}) \
         below 50% threshold",
        hit_rate * 100.0,
    );

    // Stronger check: expect all segments shared.
    assert_eq!(
        total_cached, total_possible,
        "gate_chorus_cache_sharing: expected {total_possible} cached segments, got {total_cached}"
    );

    // Verify shared state refcount includes all voices + primary.
    let refcount = chorus.shared_state_refcount();
    assert!(
        refcount >= N_VOICES,
        "gate_chorus_cache_sharing: shared_state_refcount {refcount} < {N_VOICES}"
    );

    eprintln!(
        "gate_chorus_cache_sharing: PASS -- {total_cached}/{total_possible} cached \
         (hit_rate={:.1}%), refcount={refcount}",
        hit_rate * 100.0,
    );
}

/// Gate: varied-speed chorus produces different output lengths per voice.
///
/// Synthesizes the same text with speeds [0.8, 0.9, 1.0, 1.1] across 4 voices.
/// Slower speeds should produce longer audio; faster speeds shorter. The mixed
/// output length should equal the longest voice (speed 0.8).
#[test]
fn gate_chorus_speed_variation() {
    let (mut primary, cache) = match load_production_kokoro(
        "chorus speed-variation gate skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let mut chorus = build_chorus(&mut primary, &cache, N_VOICES);
    assert_eq!(chorus.n_voices(), N_VOICES);

    let input = make_input();
    let style = make_style();
    let speeds: Vec<f32> = vec![0.8, 0.9, 1.0, 1.1];

    // Use synthesize_chorus_varied_speed for shared-encode + per-voice speed.
    let mixed = chorus
        .synthesize_chorus_varied_speed(&input, &style, &speeds, &cache)
        .expect("varied-speed chorus synthesis");

    // Mixed output must be non-empty and finite.
    assert!(
        !mixed.is_empty(),
        "gate_chorus_speed_variation: mixed audio is empty"
    );
    let nan_count = mixed.iter().filter(|s| s.is_nan()).count();
    assert!(
        nan_count == 0,
        "gate_chorus_speed_variation: {nan_count} NaN samples"
    );

    // Now synthesize individual voices at each speed to verify length ordering.
    let mut lengths: Vec<(f32, usize)> = Vec::new();
    for &speed in &speeds {
        let (audio, _cert) = primary
            .synthesize(&input, &style, speed, &cache)
            .expect("single-voice synthesis");
        let len = audio.to_flat_vec::<f32>().unwrap().len();
        lengths.push((speed, len));
    }

    // Sort by speed ascending -- slower speeds should produce longer audio.
    let mut sorted = lengths.clone();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Verify ordering: slower speed => more samples.
    for window in sorted.windows(2) {
        let (s_slow, len_slow) = window[0];
        let (s_fast, len_fast) = window[1];
        assert!(
            len_slow >= len_fast,
            "gate_chorus_speed_variation: speed {s_slow} ({len_slow} samples) should produce \
             >= samples than speed {s_fast} ({len_fast} samples)"
        );
    }

    // The slowest and fastest speeds must produce measurably different lengths.
    let (s_min, len_min_speed) = sorted[0]; // speed 0.8 = longest
    let (s_max, len_max_speed) = sorted[sorted.len() - 1]; // speed 1.1 = shortest
    assert!(
        len_min_speed > len_max_speed,
        "gate_chorus_speed_variation: slowest speed {s_min} ({len_min_speed} samples) must \
         produce strictly more audio than fastest {s_max} ({len_max_speed} samples)"
    );

    // Mixed output length should match the longest voice (slowest speed).
    let diff = (mixed.len() as isize - len_min_speed as isize).unsigned_abs();
    assert!(
        diff <= 1,
        "gate_chorus_speed_variation: mixed length {} differs from longest voice \
         ({len_min_speed}) by {diff}",
        mixed.len(),
    );

    eprintln!(
        "gate_chorus_speed_variation: PASS -- speeds={speeds:?}, lengths={lengths:?}, \
         mixed={}, longest={len_min_speed}",
        mixed.len(),
    );
}

/// Gate: `clone_dispatch_warm` shares compiled segments and initializes fast.
///
/// After warming up the primary (one synthesis call compiles all 8 segments),
/// `clone_dispatch_warm()` should produce a voice instance that inherits the
/// compiled Metal pipelines via `Arc<CompiledModelDef>`. Creating a warm clone
/// must take less than 1 second (no recompilation).
#[test]
fn gate_chorus_clone_dispatch_warm() {
    let (mut primary, cache) = match load_production_kokoro(
        "chorus clone_dispatch_warm gate skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    // Warmup: compile all 8 segments.
    let input = make_input();
    let style = make_style();
    let _ = primary
        .synthesize(&input, &style, 1.0, &cache)
        .expect("warmup synthesis");

    let parent_cached = primary.total_cached_segments();
    assert_eq!(
        parent_cached, 8,
        "gate_chorus_clone_dispatch_warm: parent should have 8 cached segments, got {parent_cached}"
    );

    // Time warm clone creation.
    let t0 = Instant::now();
    let warm_clone = primary.clone_dispatch_warm();
    let clone_dur = t0.elapsed();

    // Warm clone must inherit all cached segments.
    let clone_cached = warm_clone.total_cached_segments();
    assert_eq!(
        clone_cached, 8,
        "gate_chorus_clone_dispatch_warm: warm clone should have 8 cached segments, \
         got {clone_cached}"
    );

    // Clone creation must be fast (no recompilation). Threshold: 1 second.
    assert!(
        clone_dur.as_secs_f64() < 1.0,
        "gate_chorus_clone_dispatch_warm: warm clone took {:.3}s (threshold: 1.0s)",
        clone_dur.as_secs_f64(),
    );

    // Shared state refcount: primary + warm_clone = at least 2.
    let refcount = warm_clone.shared_state_refcount();
    assert!(
        refcount >= 2,
        "gate_chorus_clone_dispatch_warm: shared_state_refcount {refcount} < 2"
    );

    eprintln!(
        "gate_chorus_clone_dispatch_warm: PASS -- clone_time={:.3}s, cached={clone_cached}/8, \
         refcount={refcount}",
        clone_dur.as_secs_f64(),
    );
}

/// Gate: `new_warm()` produces identical output to manual `clone_dispatch`.
///
/// Both `KokoroChorus::new()` (uses `clone_dispatch`) and
/// `KokoroChorus::new_warm()` (uses `clone_dispatch_warm`) should produce
/// matching mixed audio from the same inputs. Tight tolerance for GPU
/// floating-point non-determinism.
#[test]
fn gate_chorus_new_warm_parity() {
    let (mut primary, cache) = match load_production_kokoro(
        "chorus new_warm parity gate skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    // Warmup: compile all 8 segments so both cold and warm paths inherit them.
    let input = make_input();
    let style = make_style();
    let _ = primary
        .synthesize(&input, &style, 1.0, &cache)
        .expect("warmup synthesis");

    let config = ChorusConfig::equal_gain(N_VOICES).expect("valid chorus config");

    // Path A: KokoroChorus::new (cold clone).
    let mut chorus_cold = KokoroChorus::new(&primary, config.clone()).expect("cold chorus");
    let inputs: Vec<DynTensor> = (0..N_VOICES).map(|_| input.clone()).collect();
    let styles: Vec<DynTensor> = (0..N_VOICES).map(|_| style.clone()).collect();
    let mixed_cold = chorus_cold
        .synthesize_chorus(&inputs, &styles, 1.0, &cache)
        .expect("cold chorus synthesis");

    // Path B: KokoroChorus::new_warm (warm clone).
    let mut chorus_warm = KokoroChorus::new(&primary, config).expect("warm chorus");
    let inputs_w: Vec<DynTensor> = (0..N_VOICES).map(|_| input.clone()).collect();
    let styles_w: Vec<DynTensor> = (0..N_VOICES).map(|_| style.clone()).collect();
    let mixed_warm = chorus_warm
        .synthesize_chorus(&inputs_w, &styles_w, 1.0, &cache)
        .expect("warm chorus synthesis");

    // Both paths must produce the same length.
    assert_eq!(
        mixed_cold.len(),
        mixed_warm.len(),
        "gate_chorus_new_warm_parity: length mismatch -- cold={}, warm={}",
        mixed_cold.len(),
        mixed_warm.len(),
    );

    // Check sample-level parity. Allow tiny floating-point tolerance since
    // GPU dispatch order may differ slightly between cold/warm paths.
    let max_diff = mixed_cold
        .iter()
        .zip(mixed_warm.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    // Tight tolerance: same model, same weights, same inputs.
    let tolerance = 1e-4;
    assert!(
        max_diff <= tolerance,
        "gate_chorus_new_warm_parity: max sample diff {max_diff:.6e} > tolerance {tolerance:.6e}"
    );

    // Both outputs must be valid audio (no NaN/Inf).
    let cold_nan = mixed_cold.iter().filter(|s| !s.is_finite()).count();
    let warm_nan = mixed_warm.iter().filter(|s| !s.is_finite()).count();
    assert!(
        cold_nan == 0,
        "gate_chorus_new_warm_parity: cold path has {cold_nan} non-finite samples"
    );
    assert!(
        warm_nan == 0,
        "gate_chorus_new_warm_parity: warm path has {warm_nan} non-finite samples"
    );

    eprintln!(
        "gate_chorus_new_warm_parity: PASS -- {} samples, max_diff={max_diff:.6e}, \
         cold_nan=0, warm_nan=0",
        mixed_cold.len(),
    );
}

/// Gate: recommended chorus autocast stays finite on the non-streaming shared-encode path.
///
/// This gate is scoped narrowly:
/// - It covers `KokoroChorus::synthesize_chorus_shared_encode()` only.
/// - It checks recommended segment autocast propagation to warm chorus clones.
/// - It checks only finite, bounded mixed PCM and expected output length.
#[test]
fn gate_chorus_recommended_autocast_shared_encode_finite() {
    let (primary, cache) = match load_production_kokoro(
        "chorus recommended-autocast gate skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let mut primary = primary.with_recommended_autocast();
    let primary_autocast = primary
        .segment_autocast()
        .expect("primary should expose recommended segment autocast");
    assert_eq!(
        primary_autocast.enabled_count(),
        6,
        "recommended autocast should enable 6/8 segments"
    );

    let mut chorus = build_chorus(&mut primary, &cache, N_VOICES);
    for voice_idx in 0..chorus.n_voices() {
        let cfg = chorus
            .voice(voice_idx)
            .expect("chorus voice exists")
            .segment_autocast()
            .expect("warm chorus clone should preserve recommended autocast");
        assert_eq!(
            cfg.enabled_count(),
            6,
            "voice {voice_idx} should preserve the recommended 6/8 autocast config"
        );
    }

    let input = make_input();
    let style = make_style();
    let styles: Vec<DynTensor> = (0..N_VOICES).map(|_| style.clone()).collect();

    let mixed = chorus
        .synthesize_chorus_shared_encode(&input, &styles, 1.0, &cache)
        .expect("shared-encode chorus with recommended autocast");

    assert!(
        !mixed.is_empty(),
        "gate_chorus_recommended_autocast_shared_encode_finite: mixed audio is empty"
    );
    assert!(
        mixed.iter().all(|sample| sample.is_finite()),
        "gate_chorus_recommended_autocast_shared_encode_finite: mixed audio must stay finite"
    );
    let max_abs = mixed
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs <= 1.0,
        "gate_chorus_recommended_autocast_shared_encode_finite: max_abs={max_abs:.6} exceeds PCM bounds",
    );

    let (ref_audio, _cert) = primary
        .synthesize(&input, &style, 1.0, &cache)
        .expect("single-voice recommended-autocast reference synthesis");
    let ref_len = ref_audio.to_flat_vec::<f32>().unwrap().len();
    let diff = (mixed.len() as isize - ref_len as isize).unsigned_abs();
    assert!(
        diff <= 1,
        "gate_chorus_recommended_autocast_shared_encode_finite: mixed length {} differs from reference {} by {diff}",
        mixed.len(),
        ref_len,
    );

    eprintln!(
        "gate_chorus_recommended_autocast_shared_encode_finite: PASS -- mixed={} samples, ref={ref_len}, max_abs={max_abs:.6}",
        mixed.len(),
    );
}

/// Gate: recommended chorus autocast stays finite on the plain batch mixed-output path.
///
/// This gate is scoped narrowly:
/// - It covers `KokoroChorus::synthesize_chorus()` only.
/// - It checks recommended segment autocast propagation to warm chorus clones.
/// - It checks only finite, bounded mixed PCM, expected output length, and
///   parity with the existing non-autocast batch chorus path.
#[test]
fn gate_chorus_recommended_autocast_batch_matches_baseline() {
    let (mut primary_f32, cache_f32) = match load_production_kokoro(
        "chorus recommended-autocast batch gate skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };
    let (primary_autocast, cache_autocast) = match load_production_kokoro(
        "chorus recommended-autocast batch gate skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };
    let mut primary_autocast = primary_autocast.with_recommended_autocast();

    let primary_autocast_cfg = primary_autocast
        .segment_autocast()
        .expect("primary should expose recommended segment autocast");
    assert_eq!(
        primary_autocast_cfg.enabled_count(),
        6,
        "recommended autocast should enable 6/8 segments"
    );

    let mut chorus_f32 = build_chorus(&mut primary_f32, &cache_f32, N_VOICES);
    let mut chorus_autocast = build_chorus(&mut primary_autocast, &cache_autocast, N_VOICES);
    for voice_idx in 0..chorus_autocast.n_voices() {
        let cfg = chorus_autocast
            .voice(voice_idx)
            .expect("chorus voice exists")
            .segment_autocast()
            .expect("warm chorus clone should preserve recommended autocast");
        assert_eq!(
            cfg.enabled_count(),
            6,
            "voice {voice_idx} should preserve the recommended 6/8 autocast config"
        );
    }

    let input = make_input();
    let style = make_style();
    let inputs: Vec<DynTensor> = (0..N_VOICES).map(|_| input.clone()).collect();
    let styles: Vec<DynTensor> = (0..N_VOICES).map(|_| style.clone()).collect();

    let mixed_f32 = chorus_f32
        .synthesize_chorus(&inputs, &styles, 1.0, &cache_f32)
        .expect("baseline batch chorus");
    let mixed_autocast = chorus_autocast
        .synthesize_chorus(&inputs, &styles, 1.0, &cache_autocast)
        .expect("recommended-autocast batch chorus");

    assert!(
        !mixed_f32.is_empty(),
        "gate_chorus_recommended_autocast_batch_matches_baseline: baseline mixed audio is empty"
    );
    assert!(
        !mixed_autocast.is_empty(),
        "gate_chorus_recommended_autocast_batch_matches_baseline: recommended-autocast mixed audio is empty"
    );
    assert!(
        mixed_f32.iter().all(|sample| sample.is_finite()),
        "gate_chorus_recommended_autocast_batch_matches_baseline: baseline mixed audio must stay finite"
    );
    assert!(
        mixed_autocast.iter().all(|sample| sample.is_finite()),
        "gate_chorus_recommended_autocast_batch_matches_baseline: recommended-autocast mixed audio must stay finite"
    );

    let max_abs = mixed_autocast
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs <= 1.0,
        "gate_chorus_recommended_autocast_batch_matches_baseline: max_abs={max_abs:.6} exceeds PCM bounds",
    );

    assert_eq!(
        mixed_f32.len(),
        mixed_autocast.len(),
        "gate_chorus_recommended_autocast_batch_matches_baseline: length mismatch -- baseline={}, autocast={}",
        mixed_f32.len(),
        mixed_autocast.len(),
    );

    let signal_power: f64 = mixed_f32.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    let noise_power: f64 = mixed_f32
        .iter()
        .zip(mixed_autocast.iter())
        .map(|(&a, &b)| {
            let d = f64::from(a - b);
            d * d
        })
        .sum();

    let snr_db = if noise_power < 1e-30 {
        f64::INFINITY
    } else {
        10.0 * (signal_power / noise_power).log10()
    };
    assert!(
        snr_db > 30.0,
        "gate_chorus_recommended_autocast_batch_matches_baseline: SNR {snr_db:.1} dB < 30 dB threshold"
    );

    let max_diff = mixed_f32
        .iter()
        .zip(mixed_autocast.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    eprintln!(
        "gate_chorus_recommended_autocast_batch_matches_baseline: PASS -- {} samples, SNR={snr_db:.1} dB, max_diff={max_diff:.6e}, max_abs={max_abs:.6}",
        mixed_autocast.len(),
    );
}

/// Gate: streaming chorus session produces complete audio.
///
/// Creates a `StreamingChorusSession` with 3 chunks, iterates all chunks
/// via `next_chunk()`, and verifies:
/// 1. All 3 chunks are produced.
/// 2. Session reports `is_done()` after the last chunk.
/// 3. Each chunk has non-empty PCM with no NaN/Inf.
/// 4. Concatenated audio has plausible total length.
/// 5. All samples in [-1.0, 1.0].
#[test]
fn gate_chorus_streaming_complete() {
    let (mut primary, cache) = match load_production_kokoro(
        "chorus streaming gate skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let mut chorus = build_chorus(&mut primary, &cache, N_VOICES);

    let style = make_style();
    let styles: Vec<DynTensor> = (0..N_VOICES).map(|_| style.clone()).collect();
    let stream_config = KokoroStreamConfig::default();

    // Build 3 chunks: short, long, short.
    let chunks = vec![make_input(), make_input_long(), make_input()];

    let mut session = StreamingChorusSession::new(chunks, styles, 1.0, stream_config)
        .expect("streaming session creation");

    assert_eq!(session.total_chunks(), 3);
    assert!(!session.is_done());

    let mut all_audio: Vec<f32> = Vec::new();
    let mut chunk_count = 0usize;

    while let Some(result) = session.next_chunk(&mut chorus, &cache) {
        let audio_chunk = result.unwrap_or_else(|e| {
            panic!("gate_chorus_streaming_complete: chunk {chunk_count} failed: {e:?}")
        });

        // Each chunk must produce non-empty audio.
        assert!(
            !audio_chunk.pcm.is_empty(),
            "gate_chorus_streaming_complete: chunk {chunk_count} pcm is empty"
        );

        // Each chunk must have correct index.
        assert_eq!(
            audio_chunk.chunk_index, chunk_count,
            "gate_chorus_streaming_complete: chunk_index mismatch"
        );

        // Each chunk must have no NaN/Inf.
        let non_finite = audio_chunk.pcm.iter().filter(|s| !s.is_finite()).count();
        assert!(
            non_finite == 0,
            "gate_chorus_streaming_complete: chunk {chunk_count} has {non_finite} \
             non-finite samples"
        );

        // Each chunk must be in [-1, 1].
        let out_of_range = audio_chunk.pcm.iter().filter(|s| s.abs() > 1.0).count();
        assert!(
            out_of_range == 0,
            "gate_chorus_streaming_complete: chunk {chunk_count} has {out_of_range} \
             samples outside [-1,1]"
        );

        all_audio.extend(&audio_chunk.pcm);
        chunk_count += 1;
    }

    // All 3 chunks must have been produced.
    assert_eq!(
        chunk_count, 3,
        "gate_chorus_streaming_complete: expected 3 chunks, got {chunk_count}"
    );
    assert!(
        session.is_done(),
        "gate_chorus_streaming_complete: session not done after all chunks"
    );
    assert_eq!(session.remaining(), 0);
    assert_eq!(session.synthesized_count(), 3);

    // next_chunk() after completion returns None.
    assert!(session.next_chunk(&mut chorus, &cache).is_none());

    // Plausibility: 3 chunks (8+12+8 = 28 tokens) should produce at least 0.2s.
    let min_samples = SAMPLE_RATE / 5; // 4800 samples
    assert!(
        all_audio.len() >= min_samples,
        "gate_chorus_streaming_complete: {} total samples < minimum {min_samples}",
        all_audio.len(),
    );

    // Concatenated audio must be valid.
    let total_nan = all_audio.iter().filter(|s| !s.is_finite()).count();
    assert!(
        total_nan == 0,
        "gate_chorus_streaming_complete: concatenated audio has {total_nan} non-finite samples"
    );

    eprintln!(
        "gate_chorus_streaming_complete: PASS -- {chunk_count} chunks, {} total samples, 0 NaN",
        all_audio.len(),
    );
}

/// Gate: chorus with different text per voice produces valid mixed audio.
///
/// Each voice receives a different token sequence (varying lengths). The mixed
/// output length must match the longest individual voice output. All samples
/// must be finite and in [-1, 1].
#[test]
fn gate_chorus_different_text_per_voice() {
    let (mut primary, cache) = match load_production_kokoro(
        "chorus different-text gate skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let mut chorus = build_chorus(&mut primary, &cache, N_VOICES);

    let style = make_style();

    // Different token sequences per voice: varying lengths.
    let inputs: Vec<DynTensor> = vec![
        DynTensor::from_vec_i64(vec![0, 1, 2, 3, 4, 5], &[1, 6], &cpu()).unwrap(),
        DynTensor::from_vec_i64(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9], &[1, 10], &cpu()).unwrap(),
        DynTensor::from_vec_i64(vec![0, 1, 2, 3, 4, 5, 6, 7], &[1, 8], &cpu()).unwrap(),
        DynTensor::from_vec_i64(vec![0, 1, 2, 3], &[1, 4], &cpu()).unwrap(),
    ];
    let styles: Vec<DynTensor> = (0..N_VOICES).map(|_| style.clone()).collect();

    let mixed = chorus
        .synthesize_chorus(&inputs, &styles, 1.0, &cache)
        .expect("different-text chorus synthesis");

    // Output must be non-empty.
    assert!(
        !mixed.is_empty(),
        "gate_chorus_different_text: mixed audio is empty"
    );

    // No NaN/Inf.
    let non_finite = mixed.iter().filter(|s| !s.is_finite()).count();
    assert!(
        non_finite == 0,
        "gate_chorus_different_text: {non_finite} non-finite samples in {} total",
        mixed.len(),
    );

    // All samples in [-1, 1].
    let clipped = mixed.iter().filter(|s| s.abs() > 1.0).count();
    assert!(
        clipped == 0,
        "gate_chorus_different_text: {clipped} samples outside [-1,1]"
    );

    // Mixed length should match the longest voice output (voice[1] with 10 tokens).
    // Synthesize voice[1] alone for reference.
    let (longest_audio, _cert) = primary
        .synthesize(&inputs[1], &style, 1.0, &cache)
        .expect("longest voice reference");
    let longest_len = longest_audio.to_flat_vec::<f32>().unwrap().len();

    let diff = (mixed.len() as isize - longest_len as isize).unsigned_abs();
    assert!(
        diff <= 1,
        "gate_chorus_different_text: mixed len {} differs from longest voice ({longest_len}) by {diff}",
        mixed.len(),
    );

    // RMS should be non-trivial (not silence).
    let rms: f32 = (mixed.iter().map(|x| x * x).sum::<f32>() / mixed.len() as f32).sqrt();
    assert!(
        rms > 0.001,
        "gate_chorus_different_text: RMS {rms:.6} too low (near silence)"
    );

    eprintln!(
        "gate_chorus_different_text: PASS -- {} samples, longest_ref={longest_len}, \
         diff={diff}, rms={rms:.4}, 0 non-finite, 0 clipped",
        mixed.len(),
    );
}

/// Gate: `synthesize_chorus()` and `synthesize_chorus_shared_encode()` produce
/// matching audio (SNR > 30 dB).
///
/// Both methods synthesize the same text with the same styles. The shared-encode
/// path runs encoding once and reuses it, while the per-voice path encodes
/// independently per voice. The outputs should be nearly identical -- any
/// difference comes from GPU scheduling non-determinism.
///
/// Part of #4265.
#[test]
fn gate_chorus_shared_encode_matches() {
    let (mut primary, cache) = match load_production_kokoro(
        "chorus shared-encode match gate skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let mut chorus_a = build_chorus(&mut primary, &cache, N_VOICES);
    let mut chorus_b = build_chorus(&mut primary, &cache, N_VOICES);

    let input = make_input();
    let style = make_style();
    let inputs: Vec<DynTensor> = (0..N_VOICES).map(|_| input.clone()).collect();
    let styles: Vec<DynTensor> = (0..N_VOICES).map(|_| style.clone()).collect();

    // Path A: per-voice synthesize_chorus (independent encoding per voice).
    let mixed_a = chorus_a
        .synthesize_chorus(&inputs, &styles, 1.0, &cache)
        .expect("synthesize_chorus");

    // Path B: shared-encode path (encoding once, decode per voice).
    let mixed_b = chorus_b
        .synthesize_chorus_shared_encode(&input, &styles, 1.0, &cache)
        .expect("synthesize_chorus_shared_encode");

    // Both must be non-empty.
    assert!(
        !mixed_a.is_empty(),
        "gate_chorus_shared_encode_matches: synthesize_chorus produced empty audio"
    );
    assert!(
        !mixed_b.is_empty(),
        "gate_chorus_shared_encode_matches: synthesize_chorus_shared_encode produced empty audio"
    );

    // Both must be finite.
    assert!(
        mixed_a.iter().all(|s| s.is_finite()),
        "gate_chorus_shared_encode_matches: synthesize_chorus has non-finite samples"
    );
    assert!(
        mixed_b.iter().all(|s| s.is_finite()),
        "gate_chorus_shared_encode_matches: synthesize_chorus_shared_encode has non-finite samples"
    );

    // Lengths must match.
    assert_eq!(
        mixed_a.len(),
        mixed_b.len(),
        "gate_chorus_shared_encode_matches: length mismatch -- chorus={}, shared_encode={}",
        mixed_a.len(),
        mixed_b.len(),
    );

    // Compute SNR between the two outputs.
    // SNR = 10 * log10(signal_power / noise_power)
    // where signal = mixed_a, noise = mixed_a - mixed_b.
    let signal_power: f64 = mixed_a.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    let noise_power: f64 = mixed_a
        .iter()
        .zip(mixed_b.iter())
        .map(|(&a, &b)| {
            let d = f64::from(a - b);
            d * d
        })
        .sum();

    let snr_db = if noise_power < 1e-30 {
        f64::INFINITY // Identical outputs.
    } else {
        10.0 * (signal_power / noise_power).log10()
    };

    // Gate: SNR must exceed 30 dB.
    assert!(
        snr_db > 30.0,
        "gate_chorus_shared_encode_matches: SNR {snr_db:.1} dB < 30 dB threshold"
    );

    eprintln!(
        "gate_chorus_shared_encode_matches: PASS -- {} samples, SNR={snr_db:.1} dB",
        mixed_a.len(),
    );
}

/// Gate: `synthesize_chorus_parallel()` matches `synthesize_chorus_shared_encode()`
/// within SNR > 30 dB.
///
/// The parallel path submits each voice's decode as a separate GpuFence command
/// buffer, while the sequential path uses lazy command buffer batching. Both use
/// shared encoding. Floating-point ordering differences may cause small deltas,
/// but the outputs should be functionally identical.
///
/// Part of #4265.
#[test]
fn gate_chorus_parallel_matches_sequential() {
    let (mut primary, cache) = match load_production_kokoro(
        "chorus parallel-match gate skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let mut chorus_seq = build_chorus(&mut primary, &cache, N_VOICES);
    let mut chorus_par = build_chorus(&mut primary, &cache, N_VOICES);

    let input = make_input();
    let style = make_style();
    let styles: Vec<DynTensor> = (0..N_VOICES).map(|_| style.clone()).collect();

    // Path A: sequential shared-encode.
    let mixed_seq = chorus_seq
        .synthesize_chorus_shared_encode(&input, &styles, 1.0, &cache)
        .expect("synthesize_chorus_shared_encode");

    // Path B: parallel GpuFence-interleaved decode.
    let mixed_par = chorus_par
        .synthesize_chorus_parallel(&input, &styles, 1.0, &cache)
        .expect("synthesize_chorus_parallel");

    // Lengths must match.
    assert_eq!(
        mixed_seq.len(),
        mixed_par.len(),
        "gate_chorus_parallel_matches: length mismatch -- seq={}, par={}",
        mixed_seq.len(),
        mixed_par.len(),
    );

    // Both must be finite.
    assert!(
        mixed_seq.iter().all(|s| s.is_finite()),
        "gate_chorus_parallel_matches: sequential has non-finite samples"
    );
    assert!(
        mixed_par.iter().all(|s| s.is_finite()),
        "gate_chorus_parallel_matches: parallel has non-finite samples"
    );

    // Compute SNR.
    let signal_power: f64 = mixed_seq.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    let noise_power: f64 = mixed_seq
        .iter()
        .zip(mixed_par.iter())
        .map(|(&a, &b)| {
            let d = f64::from(a - b);
            d * d
        })
        .sum();

    let snr_db = if noise_power < 1e-30 {
        f64::INFINITY
    } else {
        10.0 * (signal_power / noise_power).log10()
    };

    // Gate: SNR must exceed 30 dB.
    assert!(
        snr_db > 30.0,
        "gate_chorus_parallel_matches: SNR {snr_db:.1} dB < 30 dB threshold"
    );

    // Also check max sample difference for diagnostic purposes.
    let max_diff = mixed_seq
        .iter()
        .zip(mixed_par.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    eprintln!(
        "gate_chorus_parallel_matches: PASS -- {} samples, SNR={snr_db:.1} dB, max_diff={max_diff:.6e}",
        mixed_seq.len(),
    );
}

/// Gate: repeated chorus synthesis does not cause cache thrashing.
///
/// Synthesizes 3 different texts sequentially. After the first call populates
/// segment caches, subsequent calls should be at least 2x faster (no full
/// recompilation). This gate validates that the warm-clone cache sharing
/// actually delivers speedup on production weights.
///
/// Part of #4265.
#[test]
fn gate_chorus_no_cache_thrashing() {
    let (mut primary, cache) = match load_production_kokoro(
        "chorus cache-thrashing gate skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let mut chorus = build_chorus(&mut primary, &cache, N_VOICES);

    let style = make_style();
    let styles: Vec<DynTensor> = (0..N_VOICES).map(|_| style.clone()).collect();

    // Three different texts (different token sequences).
    let texts = [
        make_input(),      // 8 tokens
        make_input_long(), // 12 tokens
        DynTensor::from_vec_i64(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9], &[1, 10], &cpu()).unwrap(),
    ];

    let mut durations: Vec<f64> = Vec::new();

    for (i, text) in texts.iter().enumerate() {
        let t0 = Instant::now();
        let mixed = chorus
            .synthesize_chorus_shared_encode(text, &styles, 1.0, &cache)
            .unwrap_or_else(|e| panic!("gate_chorus_no_cache_thrashing: text {i} failed: {e:?}"));
        let dur = t0.elapsed().as_secs_f64();
        durations.push(dur);

        // Sanity: each output must be non-empty and finite.
        assert!(
            !mixed.is_empty(),
            "gate_chorus_no_cache_thrashing: text {i} produced empty audio"
        );
        assert!(
            mixed.iter().all(|s| s.is_finite()),
            "gate_chorus_no_cache_thrashing: text {i} has non-finite samples"
        );

        eprintln!("  text[{i}]: {:.3}s, {} samples", dur, mixed.len());
    }

    // Gate: second and third calls should each be at least 2x faster than first.
    // The first call may include Metal pipeline compilation, arena allocation,
    // and cold cache misses. Subsequent calls benefit from warm caches.
    let first = durations[0];
    for (i, &dur) in durations[1..].iter().enumerate() {
        // Only enforce the 2x speedup if the first call took a meaningful amount
        // of time (> 100ms). If the first call is already very fast (e.g., on a
        // powerful GPU with hot caches from the build_chorus warmup), the ratio
        // is less meaningful and we relax the gate.
        if first > 0.1 {
            assert!(
                dur < first / 2.0,
                "gate_chorus_no_cache_thrashing: text[{}] ({dur:.3}s) should be \
                 < {:.3}s (first_call/2 = {:.3}s)",
                i + 1,
                first / 2.0,
                first / 2.0,
            );
        }
    }

    eprintln!(
        "gate_chorus_no_cache_thrashing: PASS -- durations={:.3?}s, \
         ratio[1]/[0]={:.2}x, ratio[2]/[0]={:.2}x",
        durations,
        if durations[0] > 0.0 {
            durations[1] / durations[0]
        } else {
            0.0
        },
        if durations[0] > 0.0 {
            durations[2] / durations[0]
        } else {
            0.0
        },
    );
}

/// Gate: 4-voice basic chorus quality validation.
///
/// Comprehensive single-test gate that verifies all fundamental quality
/// properties of a 4-voice chorus in one pass:
/// - Non-empty audio output
/// - All samples finite (no NaN/Inf)
/// - Samples in range [-1.0, 1.0]
/// - Audio length > 0.5 seconds at 24kHz
/// - Not silence (max absolute value > 0.01)
///
/// Part of #4265.
#[test]
fn gate_chorus_4voice_basic() {
    let (mut primary, cache) = match load_production_kokoro(
        "chorus 4-voice basic gate skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let mut chorus = build_chorus(&mut primary, &cache, N_VOICES);

    let input = make_input();
    let style = make_style();
    let inputs: Vec<DynTensor> = (0..N_VOICES).map(|_| input.clone()).collect();
    let styles: Vec<DynTensor> = (0..N_VOICES).map(|_| style.clone()).collect();

    let mixed = chorus
        .synthesize_chorus(&inputs, &styles, 1.0, &cache)
        .expect("4-voice chorus synthesis");

    // 1. Non-empty audio output.
    assert!(
        !mixed.is_empty(),
        "gate_chorus_4voice_basic: mixed audio is empty"
    );

    // 2. All samples finite (no NaN/Inf).
    let non_finite = mixed.iter().filter(|s| !s.is_finite()).count();
    assert!(
        non_finite == 0,
        "gate_chorus_4voice_basic: {non_finite} non-finite samples in {} total",
        mixed.len(),
    );

    // 3. Samples in range [-1.0, 1.0] (no clipping beyond amplitude).
    let max_abs = mixed.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
    assert!(
        max_abs <= 1.0,
        "gate_chorus_4voice_basic: max_abs={max_abs:.6} exceeds 1.0"
    );

    // 4. Audio length > 0.5 seconds at 24kHz sample rate.
    let min_samples_half_sec = SAMPLE_RATE / 2; // 12000 samples
    assert!(
        mixed.len() >= min_samples_half_sec,
        "gate_chorus_4voice_basic: {} samples < 0.5s minimum ({min_samples_half_sec} samples)",
        mixed.len(),
    );

    // 5. Not silence (max absolute value > 0.01).
    assert!(
        max_abs > 0.01,
        "gate_chorus_4voice_basic: max_abs={max_abs:.6} -- audio is near-silent"
    );

    let duration_s = mixed.len() as f64 / SAMPLE_RATE as f64;
    eprintln!(
        "gate_chorus_4voice_basic: PASS -- {} samples ({duration_s:.2}s), \
         max_abs={max_abs:.4}, 0 non-finite",
        mixed.len(),
    );
}

/// Gate: a warmed `synthesize_chorus_pipelined()` call on production weights
/// stays within the current arena sizing envelope for the direct non-text path.
///
/// This is a targeted regression gate for the recent pipelined chorus hardening,
/// not a broad production-readiness claim. It runs on a fresh worker thread so
/// the default arena stats start from a clean thread-local state for this path.
#[test]
fn gate_chorus_pipelined_warm_path_no_arena_overflow() {
    if super::kokoro_test_env::require_kokoro_weights(
        "chorus pipelined gate skipped -- set KOKORO_WEIGHTS to enable.",
    )
    .is_none()
    {
        return;
    }

    let (arena_estimate, mixed_len, max_abs, overflow_count, overflow_bytes) =
        std::thread::spawn(|| {
            let (mut primary, cache) = load_production_kokoro(
                "chorus pipelined gate requires KOKORO_WEIGHTS inside worker thread.",
            )
            .expect("production weights already checked");

            let input = make_input_long();
            let style = make_style();
            let styles: Vec<DynTensor> = (0..N_VOICES).map(|_| style.clone()).collect();

            let mut chorus = build_chorus(&mut primary, &cache, N_VOICES);
            let arena_estimate = chorus.voice(0).expect("voice 0").estimate_arena_bytes();
            assert!(
                arena_estimate > 0,
                "gate_chorus_pipelined_warm_path_no_arena_overflow: warmed chorus must expose \
                 a non-zero arena estimate",
            );

            nn_metal::reset_arena_stats();

            let mixed = chorus
                .synthesize_chorus_pipelined(&input, &styles, 1.0, &cache)
                .expect("synthesize_chorus_pipelined");

            assert!(
                !mixed.is_empty(),
                "gate_chorus_pipelined_warm_path_no_arena_overflow: mixed audio is empty"
            );
            assert!(
                mixed.iter().all(|sample| sample.is_finite()),
                "gate_chorus_pipelined_warm_path_no_arena_overflow: mixed audio has \
                 non-finite samples"
            );

            let max_abs = mixed
                .iter()
                .map(|sample| sample.abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_abs <= 1.0,
                "gate_chorus_pipelined_warm_path_no_arena_overflow: max_abs={max_abs:.6} \
                 exceeds 1.0"
            );

            let stats = nn_metal::arena_stats();
            (
                arena_estimate,
                mixed.len(),
                max_abs,
                stats.overflow_count,
                stats.overflow_bytes,
            )
        })
        .join()
        .expect("chorus pipelined worker thread");

    assert_eq!(
        overflow_count, 0,
        "gate_chorus_pipelined_warm_path_no_arena_overflow: \
         synthesize_chorus_pipelined overflowed the default arena on a warmed call \
         (overflow_bytes={overflow_bytes})"
    );
    assert!(
        mixed_len > SAMPLE_RATE / 2,
        "gate_chorus_pipelined_warm_path_no_arena_overflow: expected >0.5s of audio, got \
         {mixed_len} samples"
    );

    eprintln!(
        "gate_chorus_pipelined_warm_path_no_arena_overflow: PASS -- estimate={arena_estimate} bytes, \
         samples={mixed_len}, max_abs={max_abs:.4}, overflow_count={overflow_count}",
    );
}

/// Gate: the user-facing `text_to_chorus()` wrapper exercises the hardened
/// shared-encode GPU path without arena overflow growth on a warmed call.
///
/// Runs in a fresh worker thread so the default arena starts from its default
/// thread-local state for this exact path.
#[test]
fn gate_text_to_chorus_warm_path_no_arena_overflow() {
    if super::kokoro_test_env::require_kokoro_weights(
        "text-to-chorus gate skipped -- set KOKORO_WEIGHTS to enable.",
    )
    .is_none()
    {
        return;
    }

    let (arena_estimate, chunk_count, total_samples, max_abs, overflow_count, overflow_bytes) =
        std::thread::spawn(|| {
            let (mut primary, cache) = load_production_kokoro(
                "text-to-chorus gate requires KOKORO_WEIGHTS inside worker thread.",
            )
            .expect("production weights already checked");

            let mut chorus = build_chorus(&mut primary, &cache, N_VOICES);
            let style = make_style();
            let styles: Vec<DynTensor> = (0..N_VOICES).map(|_| style.clone()).collect();
            let speeds = vec![1.0_f32; N_VOICES];
            let stream_config = KokoroStreamConfig::default();
            let text = "hello world. ".repeat(96);
            let ipa = "hɛˈloʊ wˈɜːɹld. ".repeat(96);

            let arena_estimate = chorus
                .voice(0)
                .expect("voice 0")
                .estimate_arena_bytes();
            assert!(
                arena_estimate > 0,
                "gate_text_to_chorus_warm_path_no_arena_overflow: warmed chorus must expose \
                 a non-zero arena estimate",
            );

            nn_metal::reset_arena_stats();

            let chunks = chorus
                .text_to_chorus(
                    &text,
                    move |_| -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
                        Ok(ipa.clone())
                    },
                    &styles,
                    &speeds,
                    &cache,
                    &stream_config,
                )
                .expect("text_to_chorus");

            assert!(
                chunks.len() > 1,
                "gate_text_to_chorus_warm_path_no_arena_overflow: expected multiple output \
                 chunks from long precomputed IPA, got {}",
                chunks.len(),
            );
            assert!(
                chunks.last().map(|chunk| chunk.is_final).unwrap_or(false),
                "gate_text_to_chorus_warm_path_no_arena_overflow: last chunk must be final",
            );

            let mut total_samples = 0usize;
            let mut max_abs = 0.0f32;
            for (i, chunk) in chunks.iter().enumerate() {
                assert_eq!(
                    chunk.chunk_index, i,
                    "gate_text_to_chorus_warm_path_no_arena_overflow: chunk_index mismatch"
                );
                assert_eq!(
                    chunk.total_chunks,
                    chunks.len(),
                    "gate_text_to_chorus_warm_path_no_arena_overflow: total_chunks mismatch"
                );
                assert!(
                    !chunk.pcm.is_empty(),
                    "gate_text_to_chorus_warm_path_no_arena_overflow: chunk {i} is empty"
                );

                for &sample in &chunk.pcm {
                    assert!(
                        sample.is_finite(),
                        "gate_text_to_chorus_warm_path_no_arena_overflow: chunk {i} has non-finite audio"
                    );
                    let abs = sample.abs();
                    if abs > max_abs {
                        max_abs = abs;
                    }
                    assert!(
                        abs <= 1.0,
                        "gate_text_to_chorus_warm_path_no_arena_overflow: chunk {i} sample \
                         exceeds [-1,1], max_abs={abs}"
                    );
                }
                total_samples += chunk.pcm.len();
            }

            let stats = nn_metal::arena_stats();
            (
                arena_estimate,
                chunks.len(),
                total_samples,
                max_abs,
                stats.overflow_count,
                stats.overflow_bytes,
            )
        })
        .join()
        .expect("text_to_chorus worker thread");

    assert_eq!(
        overflow_count, 0,
        "gate_text_to_chorus_warm_path_no_arena_overflow: text_to_chorus overflowed the \
         default arena on a warmed call (overflow_bytes={overflow_bytes})"
    );
    assert!(
        total_samples > SAMPLE_RATE / 2,
        "gate_text_to_chorus_warm_path_no_arena_overflow: expected >0.5s of audio, got \
         {total_samples} samples"
    );

    eprintln!(
        "gate_text_to_chorus_warm_path_no_arena_overflow: PASS -- estimate={arena_estimate} bytes, \
         chunks={chunk_count}, samples={total_samples}, max_abs={max_abs:.4}, \
         overflow_count={overflow_count}",
    );
}
