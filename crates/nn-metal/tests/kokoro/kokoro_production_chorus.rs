// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Production-weight end-to-end tests for Kokoro chorus synthesis using
//! [`StreamingKokoroSession`] and [`clone_dispatch()`].
//!
//! Exercises the pull-based streaming API with multi-voice chorus on real
//! production weights:
//! 1. Two voices with interleaved `next_chunk()` calls produce valid audio.
//! 2. `clone_dispatch()` shares compiled segments so the warm clone run avoids
//!    the parent's cold-start compile cost in this harness.
//! 3. Two voices at different speeds produce different-length audio.
//! 4. Session state machine (`remaining()`, `is_done()`, `reset()`) works
//!    correctly with real synthesis.
//! 5. Interleaved voice sessions stay aligned with sequential per-voice
//!    baselines.
//!
//! This file covers the clone-dispatch + `StreamingKokoroSession` chorus
//! harness only. It does not certify mixed `KokoroChorus` output quality or
//! production throughput.
//!
//! All tests gated behind `KOKORO_WEIGHTS` env var.
//!
//! Part of #4105, #4265.

use std::fs;
use std::path::Path;
use std::time::Instant;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use nn_metal::compiled_kokoro::{CompiledKokoro, StreamingKokoroSession};
use serde::{Deserialize, Serialize};

fn cpu() -> Device {
    Device::Cpu
}

/// Stable schema version for the chorus production scorecard output.
const CHORUS_SCORECARD_SCHEMA_VERSION: u32 = 1;

/// Optional artifact path for the chorus production scorecard.
const CHORUS_SCORECARD_OUT_ENV: &str = "KOKORO_PRODUCTION_CHORUS_SCORECARD_OUT";

/// Stable stderr prefix for compact JSON chorus scorecards.
const CHORUS_SCORECARD_STDERR_PREFIX: &str = "KOKORO_PRODUCTION_CHORUS_SCORECARD_JSON=";

/// Tolerance used by the sequential vs interleaved parity check.
const PCM_TOLERANCE: f32 = 1e-5;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ChorusProductionHarness {
    CloneDispatchStreamingSession,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ChorusMeasuredSurface {
    PerVoiceStreamingSessions,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ChorusTextMode {
    SharedText,
    PerVoicePrograms,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ChorusWarmState {
    ColdStart,
    WarmParentCache,
    WarmCloneSharedCache,
    MixedColdParentAndWarmClone,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ChorusCheckedProperty {
    FiniteAudioSamples,
    SamplesWithinUnitRange,
    CloneStartsWithSharedSegmentCache,
    WarmCloneBeatsColdParentInHarness,
    InterleavedStreamingCompletes,
    InterleavedMatchesSequentialBaselineWithinTolerance,
    SlowerSpeedProducesLongerOutput,
    SessionStateTransitions,
    ResetAtAdjustedSpeedProducesLongerOutput,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ChorusScorecardLimitation {
    NotMixedKokoroChorusSurface,
    NotThroughputBenchmark,
    NotAudioQualityCertification,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ChorusMeasurementScope {
    name: String,
    harness: ChorusProductionHarness,
    voice_count: usize,
    text_mode: ChorusTextMode,
    warm_state: ChorusWarmState,
    checked_properties: Vec<ChorusCheckedProperty>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ChorusRuntimeDispatchScorecard {
    compute_encodings: usize,
    blits: usize,
    total_metal_command_encodings: usize,
    flushes: usize,
    submits: usize,
    blits_eliminated: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ChorusCompiledDispatchSegments {
    plbert: usize,
    text_encoder: usize,
    prosody: usize,
    regulate: usize,
    f0_energy: usize,
    sinegen_pre: usize,
    sinegen_post: usize,
    generator: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ChorusCompiledDispatchScorecard {
    logical_total: usize,
    estimated_metal_total: usize,
    estimated_encoding_events_total: usize,
    expected_submit_count: usize,
    segments: ChorusCompiledDispatchSegments,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct TwoVoiceInterleavedStreamingMeasurement {
    scope: ChorusMeasurementScope,
    voice_sample_counts: Vec<usize>,
    runtime_dispatch: ChorusRuntimeDispatchScorecard,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct SharedCompilationMeasurement {
    scope: ChorusMeasurementScope,
    cold_wall_ms: f64,
    warm_wall_ms: f64,
    warm_to_cold_ratio: f64,
    parent_cached_segments: usize,
    clone_cached_segments_before: usize,
    clone_cached_segments_after: usize,
    cold_num_samples: usize,
    clone_num_samples: usize,
    cold_runtime_dispatch: ChorusRuntimeDispatchScorecard,
    warm_runtime_dispatch: ChorusRuntimeDispatchScorecard,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct InterleavedParityMeasurement {
    scope: ChorusMeasurementScope,
    sequential_voice_sample_counts: Vec<usize>,
    interleaved_voice_sample_counts: Vec<usize>,
    max_abs_diff_per_voice: Vec<f32>,
    pcm_tolerance: f32,
    cached_segments_per_voice: Vec<usize>,
    sequential_runtime_dispatch: ChorusRuntimeDispatchScorecard,
    interleaved_runtime_dispatch: ChorusRuntimeDispatchScorecard,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct SpeedVariationMeasurement {
    scope: ChorusMeasurementScope,
    fast_speed: f32,
    slow_speed: f32,
    fast_num_samples: usize,
    slow_num_samples: usize,
    slow_to_fast_ratio: f64,
    runtime_dispatch: ChorusRuntimeDispatchScorecard,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct SessionStateMachineMeasurement {
    scope: ChorusMeasurementScope,
    initial_remaining: usize,
    remaining_after_first_chunk: usize,
    remaining_after_second_chunk: usize,
    remaining_after_reset: usize,
    reset_speed: f32,
    original_chunk0_samples: usize,
    chunk1_samples: usize,
    chunk2_samples: usize,
    reset_chunk0_samples: usize,
    runtime_dispatch: ChorusRuntimeDispatchScorecard,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "measurement_kind", rename_all = "snake_case")]
enum ChorusProductionMeasurement {
    TwoVoiceInterleavedStreaming(TwoVoiceInterleavedStreamingMeasurement),
    SharedCompilation(SharedCompilationMeasurement),
    InterleavedParity(InterleavedParityMeasurement),
    SpeedVariation(SpeedVariationMeasurement),
    SessionStateMachine(SessionStateMachineMeasurement),
}

impl ChorusProductionMeasurement {
    fn scope(&self) -> &ChorusMeasurementScope {
        match self {
            Self::TwoVoiceInterleavedStreaming(measurement) => &measurement.scope,
            Self::SharedCompilation(measurement) => &measurement.scope,
            Self::InterleavedParity(measurement) => &measurement.scope,
            Self::SpeedVariation(measurement) => &measurement.scope,
            Self::SessionStateMachine(measurement) => &measurement.scope,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ChorusProductionScorecard {
    schema_version: u32,
    suite_name: String,
    build_profile: String,
    measured_surface: ChorusMeasuredSurface,
    measured_harness: ChorusProductionHarness,
    /// Whether this scorecard's measured chorus harness ran with
    /// `CompiledKokoro::with_recommended_autocast()`.
    ///
    /// This is harness-configuration metadata only. It does not imply any
    /// quality or throughput claim.
    recommended_autocast_enabled: bool,
    limitations: Vec<ChorusScorecardLimitation>,
    compiled_dispatch: ChorusCompiledDispatchScorecard,
    measurements: Vec<ChorusProductionMeasurement>,
}

impl ChorusProductionScorecard {
    fn to_compact_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    fn to_pretty_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    fn write_json<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        let json = self.to_pretty_json().map_err(std::io::Error::other)?;
        fs::write(path, json)
    }

    fn measurements_match_declared_harness(&self) -> bool {
        self.measurements
            .iter()
            .all(|measurement| measurement.scope().harness == self.measured_harness)
    }
}

fn build_profile_name() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

// -- Helpers ------------------------------------------------------------------

/// Load production Kokoro with Warn rejection policy.
///
/// Returns `None` if `KOKORO_WEIGHTS` is not set (test skips gracefully).
fn load_production_kokoro(skip_msg: &str) -> Option<(CompiledKokoro, nn_metal::PipelineCache)> {
    let weights_path = super::kokoro_test_env::require_kokoro_weights(skip_msg)?;

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Use Warn policy: test tokens [0..7] produce click artifacts with
    // production weights that fail the no_clicks hard bound. Part of #4262.
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
fn make_production_input() -> DynTensor {
    DynTensor::from_vec_i64(vec![0_i64, 1, 2, 3, 4, 5, 6, 7], &[1, 8], &cpu()).unwrap()
}

/// A second test utterance: 12 phoneme tokens (longer text).
fn make_production_input_long() -> DynTensor {
    DynTensor::from_vec_i64(
        vec![0_i64, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        &[1, 12],
        &cpu(),
    )
    .unwrap()
}

/// Production style tensor: [1, 256] filled with 0.01.
fn make_production_style() -> DynTensor {
    DynTensor::full(&[1, 256], 0.01, DType::F32, &cpu()).unwrap()
}

/// Validate that audio samples are non-empty, all finite, and in [-1, 1].
fn validate_audio(audio: &[f32], label: &str) {
    assert!(!audio.is_empty(), "{label}: audio must not be empty");

    let mut nan_count = 0usize;
    let mut out_of_range_count = 0usize;
    let mut max_abs = 0.0f32;
    for &sample in audio {
        if sample.is_nan() {
            nan_count += 1;
        } else {
            let abs = sample.abs();
            if abs > max_abs {
                max_abs = abs;
            }
            if abs > 1.0 {
                out_of_range_count += 1;
            }
        }
    }
    assert!(nan_count == 0, "{label}: {nan_count} NaN samples detected");
    assert!(
        out_of_range_count == 0,
        "{label}: {out_of_range_count} samples outside [-1,1], max_abs={max_abs}"
    );
}

/// Drain a streaming session into a flat PCM buffer and validate the result.
fn collect_session_audio(
    session: &mut StreamingKokoroSession,
    kokoro: &mut CompiledKokoro,
    cache: &nn_metal::PipelineCache,
    label: &str,
) -> Vec<f32> {
    let mut audio = Vec::new();

    while let Some(result) = session.next_chunk(kokoro, cache) {
        let (chunk, _cert) = result.expect(label);
        audio.extend(chunk.to_flat_vec::<f32>().unwrap());
    }

    assert!(session.is_done(), "{label}: session should be done");
    validate_audio(&audio, label);
    audio
}

/// Compare two PCM buffers, returning the maximum per-sample absolute error.
fn assert_audio_close(reference: &[f32], actual: &[f32], label: &str, tolerance: f32) -> f32 {
    assert_eq!(
        actual.len(),
        reference.len(),
        "{label}: sample count mismatch, reference={} actual={}",
        reference.len(),
        actual.len(),
    );

    let max_diff = reference
        .iter()
        .zip(actual.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_diff <= tolerance,
        "{label}: max_diff={max_diff:.8} exceeds tolerance={tolerance:.8}",
    );

    max_diff
}

fn runtime_dispatch_scorecard(stats: nn_metal::DispatchStats) -> ChorusRuntimeDispatchScorecard {
    ChorusRuntimeDispatchScorecard {
        compute_encodings: stats.compute_encodings,
        blits: stats.blits,
        total_metal_command_encodings: stats.compute_encodings + stats.blits,
        flushes: stats.flushes,
        submits: stats.submits,
        blits_eliminated: stats.blits_eliminated,
    }
}

fn compiled_dispatch_scorecard(kokoro: &CompiledKokoro) -> ChorusCompiledDispatchScorecard {
    let summary = kokoro.dispatch_summary();
    ChorusCompiledDispatchScorecard {
        logical_total: summary.total(),
        estimated_metal_total: kokoro.total_metal_dispatches(),
        estimated_encoding_events_total: kokoro.total_encoding_events(),
        expected_submit_count: summary.expected_submit_count(),
        segments: ChorusCompiledDispatchSegments {
            plbert: summary.plbert,
            text_encoder: summary.text_encoder,
            prosody: summary.prosody,
            regulate: summary.regulate,
            f0_energy: summary.f0_energy,
            sinegen_pre: summary.sinegen_pre,
            sinegen_post: summary.sinegen_post,
            generator: summary.generator,
        },
    }
}

fn capture_dispatch_stats<T>(op: impl FnOnce() -> T) -> (T, ChorusRuntimeDispatchScorecard) {
    nn_metal::reset_counters();
    let value = op();
    let stats = nn_metal::dispatch_stats();
    (value, runtime_dispatch_scorecard(stats))
}

fn timed_with_dispatch_stats<T>(
    op: impl FnOnce() -> T,
) -> (T, f64, ChorusRuntimeDispatchScorecard) {
    nn_metal::reset_counters();
    let started = Instant::now();
    let value = op();
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let stats = nn_metal::dispatch_stats();
    (value, elapsed_ms, runtime_dispatch_scorecard(stats))
}

fn chorus_scorecard_stderr_line(report: &ChorusProductionScorecard) -> Option<String> {
    report
        .to_compact_json()
        .ok()
        .map(|compact_json| format!("{CHORUS_SCORECARD_STDERR_PREFIX}{compact_json}"))
}

fn configured_chorus_scorecard_artifact_path(env_var: &str) -> Option<String> {
    std::env::var(env_var)
        .ok()
        .map(|path| path.trim().to_string())
        .filter(|path| !path.is_empty())
}

fn emit_chorus_scorecard_with_artifact_path(
    report: &ChorusProductionScorecard,
    artifact_path: Option<&str>,
) -> Option<String> {
    let stderr_line = chorus_scorecard_stderr_line(report);
    if let Some(line) = stderr_line.as_deref() {
        eprintln!("{line}");
    }

    if let Some(path) = artifact_path {
        report
            .write_json(path)
            .unwrap_or_else(|e| panic!("failed to write chorus scorecard artifact {path}: {e}"));
        eprintln!("  Chorus scorecard artifact: {path}");
    }

    stderr_line
}

fn emit_chorus_scorecard(report: &ChorusProductionScorecard) {
    let artifact_path = configured_chorus_scorecard_artifact_path(CHORUS_SCORECARD_OUT_ENV);
    let _ = emit_chorus_scorecard_with_artifact_path(report, artifact_path.as_deref());
}

fn sample_chorus_production_scorecard() -> ChorusProductionScorecard {
    ChorusProductionScorecard {
        schema_version: CHORUS_SCORECARD_SCHEMA_VERSION,
        suite_name: "kokoro_production_chorus".to_string(),
        build_profile: "release".to_string(),
        measured_surface: ChorusMeasuredSurface::PerVoiceStreamingSessions,
        measured_harness: ChorusProductionHarness::CloneDispatchStreamingSession,
        recommended_autocast_enabled: false,
        limitations: vec![
            ChorusScorecardLimitation::NotMixedKokoroChorusSurface,
            ChorusScorecardLimitation::NotThroughputBenchmark,
            ChorusScorecardLimitation::NotAudioQualityCertification,
        ],
        compiled_dispatch: ChorusCompiledDispatchScorecard {
            logical_total: 476,
            estimated_metal_total: 674,
            estimated_encoding_events_total: 701,
            expected_submit_count: 6,
            segments: ChorusCompiledDispatchSegments {
                plbert: 50,
                text_encoder: 20,
                prosody: 43,
                regulate: 16,
                f0_energy: 32,
                sinegen_pre: 5,
                sinegen_post: 28,
                generator: 282,
            },
        },
        measurements: vec![
            ChorusProductionMeasurement::SharedCompilation(SharedCompilationMeasurement {
                scope: ChorusMeasurementScope {
                    name: "shared_compilation".to_string(),
                    harness: ChorusProductionHarness::CloneDispatchStreamingSession,
                    voice_count: 2,
                    text_mode: ChorusTextMode::SharedText,
                    warm_state: ChorusWarmState::MixedColdParentAndWarmClone,
                    checked_properties: vec![
                        ChorusCheckedProperty::CloneStartsWithSharedSegmentCache,
                        ChorusCheckedProperty::WarmCloneBeatsColdParentInHarness,
                    ],
                },
                cold_wall_ms: 120.0,
                warm_wall_ms: 30.0,
                warm_to_cold_ratio: 0.25,
                parent_cached_segments: 8,
                clone_cached_segments_before: 8,
                clone_cached_segments_after: 8,
                cold_num_samples: 24_000,
                clone_num_samples: 24_000,
                cold_runtime_dispatch: ChorusRuntimeDispatchScorecard {
                    compute_encodings: 445,
                    blits: 12,
                    total_metal_command_encodings: 457,
                    flushes: 3,
                    submits: 2,
                    blits_eliminated: 7,
                },
                warm_runtime_dispatch: ChorusRuntimeDispatchScorecard {
                    compute_encodings: 445,
                    blits: 12,
                    total_metal_command_encodings: 457,
                    flushes: 3,
                    submits: 2,
                    blits_eliminated: 7,
                },
            }),
            ChorusProductionMeasurement::InterleavedParity(InterleavedParityMeasurement {
                scope: ChorusMeasurementScope {
                    name: "interleaved_matches_sequential_baselines".to_string(),
                    harness: ChorusProductionHarness::CloneDispatchStreamingSession,
                    voice_count: 2,
                    text_mode: ChorusTextMode::PerVoicePrograms,
                    warm_state: ChorusWarmState::WarmParentCache,
                    checked_properties: vec![
                        ChorusCheckedProperty::InterleavedMatchesSequentialBaselineWithinTolerance,
                    ],
                },
                sequential_voice_sample_counts: vec![30_000, 32_000],
                interleaved_voice_sample_counts: vec![30_000, 32_000],
                max_abs_diff_per_voice: vec![0.0, 0.0],
                pcm_tolerance: PCM_TOLERANCE,
                cached_segments_per_voice: vec![8, 8],
                sequential_runtime_dispatch: ChorusRuntimeDispatchScorecard {
                    compute_encodings: 900,
                    blits: 24,
                    total_metal_command_encodings: 924,
                    flushes: 6,
                    submits: 4,
                    blits_eliminated: 14,
                },
                interleaved_runtime_dispatch: ChorusRuntimeDispatchScorecard {
                    compute_encodings: 900,
                    blits: 24,
                    total_metal_command_encodings: 924,
                    flushes: 6,
                    submits: 4,
                    blits_eliminated: 14,
                },
            }),
            ChorusProductionMeasurement::SpeedVariation(SpeedVariationMeasurement {
                scope: ChorusMeasurementScope {
                    name: "different_speeds".to_string(),
                    harness: ChorusProductionHarness::CloneDispatchStreamingSession,
                    voice_count: 2,
                    text_mode: ChorusTextMode::SharedText,
                    warm_state: ChorusWarmState::WarmParentCache,
                    checked_properties: vec![
                        ChorusCheckedProperty::SlowerSpeedProducesLongerOutput,
                    ],
                },
                fast_speed: 1.0,
                slow_speed: 0.8,
                fast_num_samples: 10_000,
                slow_num_samples: 12_500,
                slow_to_fast_ratio: 1.25,
                runtime_dispatch: ChorusRuntimeDispatchScorecard {
                    compute_encodings: 0,
                    blits: 0,
                    total_metal_command_encodings: 0,
                    flushes: 0,
                    submits: 0,
                    blits_eliminated: 0,
                },
            }),
        ],
    }
}

#[cfg(test)]
static CHORUS_SCORECARD_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn measure_two_voice_interleaved_streaming(
    parent: &mut CompiledKokoro,
    cache: &nn_metal::PipelineCache,
    input_ids: &DynTensor,
    input_ids_long: &DynTensor,
    style: &DynTensor,
) -> TwoVoiceInterleavedStreamingMeasurement {
    let mut voice1 = parent.clone_dispatch();
    let chunks_v0 = vec![
        (input_ids.clone(), style.clone()),
        (input_ids_long.clone(), style.clone()),
        (input_ids.clone(), style.clone()),
    ];
    let chunks_v1 = vec![
        (input_ids_long.clone(), style.clone()),
        (input_ids.clone(), style.clone()),
        (input_ids_long.clone(), style.clone()),
    ];

    let mut session0 = StreamingKokoroSession::new(chunks_v0, 1.0);
    let mut session1 = StreamingKokoroSession::new(chunks_v1, 1.0);

    let ((all_audio_v0, all_audio_v1), runtime_dispatch) = capture_dispatch_stats(|| {
        let mut all_audio_v0 = Vec::new();
        let mut all_audio_v1 = Vec::new();
        while !session0.is_done() || !session1.is_done() {
            if let Some(result) = session0.next_chunk(parent, cache) {
                let (audio, _cert) = result.expect("scorecard voice0 chunk synthesis");
                all_audio_v0.extend(audio.to_flat_vec::<f32>().unwrap());
            }
            if let Some(result) = session1.next_chunk(&mut voice1, cache) {
                let (audio, _cert) = result.expect("scorecard voice1 chunk synthesis");
                all_audio_v1.extend(audio.to_flat_vec::<f32>().unwrap());
            }
        }
        (all_audio_v0, all_audio_v1)
    });

    assert!(session0.is_done());
    assert!(session1.is_done());
    validate_audio(&all_audio_v0, "scorecard voice0");
    validate_audio(&all_audio_v1, "scorecard voice1");

    TwoVoiceInterleavedStreamingMeasurement {
        scope: ChorusMeasurementScope {
            name: "two_voice_interleaved_streaming".to_string(),
            harness: ChorusProductionHarness::CloneDispatchStreamingSession,
            voice_count: 2,
            text_mode: ChorusTextMode::PerVoicePrograms,
            warm_state: ChorusWarmState::WarmParentCache,
            checked_properties: vec![
                ChorusCheckedProperty::FiniteAudioSamples,
                ChorusCheckedProperty::SamplesWithinUnitRange,
                ChorusCheckedProperty::InterleavedStreamingCompletes,
            ],
        },
        voice_sample_counts: vec![all_audio_v0.len(), all_audio_v1.len()],
        runtime_dispatch,
    }
}

fn measure_shared_compilation(
    parent: &mut CompiledKokoro,
    cache: &nn_metal::PipelineCache,
    input_ids: &DynTensor,
    style: &DynTensor,
) -> SharedCompilationMeasurement {
    let ((audio_parent, _cert), cold_wall_ms, cold_runtime_dispatch) =
        timed_with_dispatch_stats(|| {
            parent
                .synthesize(input_ids, style, 1.0, cache)
                .expect("scorecard parent cold synthesis")
        });
    let cold_vals = audio_parent.to_flat_vec::<f32>().unwrap();
    validate_audio(&cold_vals, "scorecard parent cold");

    let parent_cached_segments = parent.total_cached_segments();
    assert_eq!(
        parent_cached_segments, 8,
        "scorecard warmed parent should have 8 cached segments, got {parent_cached_segments}"
    );

    let mut clone = parent.clone_dispatch();
    let clone_cached_segments_before = clone.total_cached_segments();
    assert_eq!(
        clone_cached_segments_before, 8,
        "scorecard clone should have 8 cached segments immediately, got {clone_cached_segments_before}"
    );

    let ((audio_clone, _cert), warm_wall_ms, warm_runtime_dispatch) =
        timed_with_dispatch_stats(|| {
            clone
                .synthesize(input_ids, style, 1.0, cache)
                .expect("scorecard clone warm synthesis")
        });
    let clone_vals = audio_clone.to_flat_vec::<f32>().unwrap();
    validate_audio(&clone_vals, "scorecard clone warm");

    assert!(
        warm_wall_ms < cold_wall_ms,
        "scorecard clone warm synthesis ({warm_wall_ms:.1}ms) should be faster than parent \
         cold start ({cold_wall_ms:.1}ms)"
    );

    let clone_cached_segments_after = clone.total_cached_segments();

    SharedCompilationMeasurement {
        scope: ChorusMeasurementScope {
            name: "shared_compilation".to_string(),
            harness: ChorusProductionHarness::CloneDispatchStreamingSession,
            voice_count: 2,
            text_mode: ChorusTextMode::SharedText,
            warm_state: ChorusWarmState::MixedColdParentAndWarmClone,
            checked_properties: vec![
                ChorusCheckedProperty::FiniteAudioSamples,
                ChorusCheckedProperty::SamplesWithinUnitRange,
                ChorusCheckedProperty::CloneStartsWithSharedSegmentCache,
                ChorusCheckedProperty::WarmCloneBeatsColdParentInHarness,
            ],
        },
        cold_wall_ms,
        warm_wall_ms,
        warm_to_cold_ratio: warm_wall_ms / cold_wall_ms,
        parent_cached_segments,
        clone_cached_segments_before,
        clone_cached_segments_after,
        cold_num_samples: cold_vals.len(),
        clone_num_samples: clone_vals.len(),
        cold_runtime_dispatch,
        warm_runtime_dispatch,
    }
}

fn measure_interleaved_parity(
    parent: &mut CompiledKokoro,
    cache: &nn_metal::PipelineCache,
    input_ids: &DynTensor,
    input_ids_long: &DynTensor,
    style: &DynTensor,
) -> InterleavedParityMeasurement {
    let chunks_v0 = vec![
        (input_ids.clone(), style.clone()),
        (input_ids_long.clone(), style.clone()),
        (input_ids.clone(), style.clone()),
    ];
    let chunks_v1 = vec![
        (input_ids_long.clone(), style.clone()),
        (input_ids.clone(), style.clone()),
        (input_ids_long.clone(), style.clone()),
    ];

    let mut seq_voice0 = parent.clone_dispatch();
    let mut seq_voice1 = parent.clone_dispatch();
    let mut seq_session0 = StreamingKokoroSession::new(chunks_v0.clone(), 1.0);
    let mut seq_session1 = StreamingKokoroSession::new(chunks_v1.clone(), 1.0);
    let ((baseline_v0, baseline_v1), sequential_runtime_dispatch) = capture_dispatch_stats(|| {
        (
            collect_session_audio(
                &mut seq_session0,
                &mut seq_voice0,
                cache,
                "scorecard seq v0",
            ),
            collect_session_audio(
                &mut seq_session1,
                &mut seq_voice1,
                cache,
                "scorecard seq v1",
            ),
        )
    });

    let mut interleaved_voice0 = parent.clone_dispatch();
    let mut interleaved_voice1 = parent.clone_dispatch();
    let mut interleaved_session0 = StreamingKokoroSession::new(chunks_v0, 1.0);
    let mut interleaved_session1 = StreamingKokoroSession::new(chunks_v1, 1.0);
    let ((interleaved_v0, interleaved_v1), interleaved_runtime_dispatch) =
        capture_dispatch_stats(|| {
            let mut interleaved_v0 = Vec::new();
            let mut interleaved_v1 = Vec::new();
            while !interleaved_session0.is_done() || !interleaved_session1.is_done() {
                if let Some(result) =
                    interleaved_session0.next_chunk(&mut interleaved_voice0, cache)
                {
                    let (chunk, _cert) = result.expect("scorecard interleaved v0");
                    interleaved_v0.extend(chunk.to_flat_vec::<f32>().unwrap());
                }
                if let Some(result) =
                    interleaved_session1.next_chunk(&mut interleaved_voice1, cache)
                {
                    let (chunk, _cert) = result.expect("scorecard interleaved v1");
                    interleaved_v1.extend(chunk.to_flat_vec::<f32>().unwrap());
                }
            }
            (interleaved_v0, interleaved_v1)
        });

    validate_audio(&interleaved_v0, "scorecard interleaved v0");
    validate_audio(&interleaved_v1, "scorecard interleaved v1");

    let v0_max_diff = assert_audio_close(
        &baseline_v0,
        &interleaved_v0,
        "scorecard voice0 sequential parity",
        PCM_TOLERANCE,
    );
    let v1_max_diff = assert_audio_close(
        &baseline_v1,
        &interleaved_v1,
        "scorecard voice1 sequential parity",
        PCM_TOLERANCE,
    );

    InterleavedParityMeasurement {
        scope: ChorusMeasurementScope {
            name: "interleaved_matches_sequential_baselines".to_string(),
            harness: ChorusProductionHarness::CloneDispatchStreamingSession,
            voice_count: 2,
            text_mode: ChorusTextMode::PerVoicePrograms,
            warm_state: ChorusWarmState::WarmParentCache,
            checked_properties: vec![
                ChorusCheckedProperty::FiniteAudioSamples,
                ChorusCheckedProperty::SamplesWithinUnitRange,
                ChorusCheckedProperty::InterleavedMatchesSequentialBaselineWithinTolerance,
            ],
        },
        sequential_voice_sample_counts: vec![baseline_v0.len(), baseline_v1.len()],
        interleaved_voice_sample_counts: vec![interleaved_v0.len(), interleaved_v1.len()],
        max_abs_diff_per_voice: vec![v0_max_diff, v1_max_diff],
        pcm_tolerance: PCM_TOLERANCE,
        cached_segments_per_voice: vec![
            interleaved_voice0.total_cached_segments(),
            interleaved_voice1.total_cached_segments(),
        ],
        sequential_runtime_dispatch,
        interleaved_runtime_dispatch,
    }
}

fn measure_speed_variation(
    parent: &mut CompiledKokoro,
    cache: &nn_metal::PipelineCache,
    input_ids: &DynTensor,
    style: &DynTensor,
) -> SpeedVariationMeasurement {
    let mut voice_slow = parent.clone_dispatch();
    let chunks_fast = vec![
        (input_ids.clone(), style.clone()),
        (input_ids.clone(), style.clone()),
    ];
    let chunks_slow = vec![
        (input_ids.clone(), style.clone()),
        (input_ids.clone(), style.clone()),
    ];

    let mut session_fast = StreamingKokoroSession::new(chunks_fast, 1.0);
    let mut session_slow = StreamingKokoroSession::new(chunks_slow, 0.8);
    let ((audio_fast, audio_slow), runtime_dispatch) = capture_dispatch_stats(|| {
        let mut audio_fast = Vec::new();
        let mut audio_slow = Vec::new();
        while !session_fast.is_done() || !session_slow.is_done() {
            if let Some(result) = session_fast.next_chunk(parent, cache) {
                let (audio, _cert) = result.expect("scorecard fast voice chunk synthesis");
                audio_fast.extend(audio.to_flat_vec::<f32>().unwrap());
            }
            if let Some(result) = session_slow.next_chunk(&mut voice_slow, cache) {
                let (audio, _cert) = result.expect("scorecard slow voice chunk synthesis");
                audio_slow.extend(audio.to_flat_vec::<f32>().unwrap());
            }
        }
        (audio_fast, audio_slow)
    });

    validate_audio(&audio_fast, "scorecard fast (1.0)");
    validate_audio(&audio_slow, "scorecard slow (0.8)");
    assert!(
        audio_slow.len() > audio_fast.len(),
        "scorecard slow voice (0.8x, {} samples) should produce more audio than fast voice \
         (1.0x, {} samples)",
        audio_slow.len(),
        audio_fast.len(),
    );

    let slow_to_fast_ratio = audio_slow.len() as f64 / audio_fast.len() as f64;
    assert!(
        (1.1..=1.5).contains(&slow_to_fast_ratio),
        "scorecard slow/fast audio length ratio should be ~1.25, got {slow_to_fast_ratio:.3} \
         (slow={}, fast={})",
        audio_slow.len(),
        audio_fast.len(),
    );

    SpeedVariationMeasurement {
        scope: ChorusMeasurementScope {
            name: "different_speeds".to_string(),
            harness: ChorusProductionHarness::CloneDispatchStreamingSession,
            voice_count: 2,
            text_mode: ChorusTextMode::SharedText,
            warm_state: ChorusWarmState::WarmParentCache,
            checked_properties: vec![
                ChorusCheckedProperty::FiniteAudioSamples,
                ChorusCheckedProperty::SamplesWithinUnitRange,
                ChorusCheckedProperty::SlowerSpeedProducesLongerOutput,
            ],
        },
        fast_speed: 1.0,
        slow_speed: 0.8,
        fast_num_samples: audio_fast.len(),
        slow_num_samples: audio_slow.len(),
        slow_to_fast_ratio,
        runtime_dispatch,
    }
}

fn measure_session_state_machine(
    parent: &mut CompiledKokoro,
    cache: &nn_metal::PipelineCache,
    input_ids: &DynTensor,
    input_ids_long: &DynTensor,
    style: &DynTensor,
) -> SessionStateMachineMeasurement {
    let chunks = vec![
        (input_ids.clone(), style.clone()),
        (input_ids_long.clone(), style.clone()),
        (input_ids.clone(), style.clone()),
    ];
    let mut session = StreamingKokoroSession::new(chunks, 1.0);
    let initial_remaining = session.remaining();

    let (
        (
            original_chunk0_samples,
            chunk1_samples,
            chunk2_samples,
            remaining_after_first_chunk,
            remaining_after_second_chunk,
            remaining_after_reset,
            reset_chunk0_samples,
        ),
        runtime_dispatch,
    ) = capture_dispatch_stats(|| {
        let first = session.next_chunk(parent, cache);
        assert!(first.is_some(), "scorecard chunk 0 should be available");
        let (audio0, _cert) = first.unwrap().expect("scorecard chunk 0 synthesis");
        let audio0_vals = audio0.to_flat_vec::<f32>().unwrap();
        validate_audio(&audio0_vals, "scorecard chunk 0");
        let remaining_after_first_chunk = session.remaining();

        let second = session.next_chunk(parent, cache);
        assert!(second.is_some(), "scorecard chunk 1 should be available");
        let (audio1, _cert) = second.unwrap().expect("scorecard chunk 1 synthesis");
        let audio1_vals = audio1.to_flat_vec::<f32>().unwrap();
        validate_audio(&audio1_vals, "scorecard chunk 1");
        let remaining_after_second_chunk = session.remaining();

        let third = session.next_chunk(parent, cache);
        assert!(third.is_some(), "scorecard chunk 2 should be available");
        let (audio2, _cert) = third.unwrap().expect("scorecard chunk 2 synthesis");
        let audio2_vals = audio2.to_flat_vec::<f32>().unwrap();
        validate_audio(&audio2_vals, "scorecard chunk 2");
        assert!(session.is_done());
        assert!(session.next_chunk(parent, cache).is_none());

        session.reset();
        let remaining_after_reset = session.remaining();
        session.set_speed(0.9);

        let replay = session.next_chunk(parent, cache);
        assert!(
            replay.is_some(),
            "scorecard chunk 0 should be available after reset"
        );
        let (audio_reset, _cert) = replay
            .unwrap()
            .expect("scorecard chunk 0 synthesis after reset");
        let audio_reset_vals = audio_reset.to_flat_vec::<f32>().unwrap();
        validate_audio(&audio_reset_vals, "scorecard chunk 0 after reset");

        (
            audio0_vals.len(),
            audio1_vals.len(),
            audio2_vals.len(),
            remaining_after_first_chunk,
            remaining_after_second_chunk,
            remaining_after_reset,
            audio_reset_vals.len(),
        )
    });

    assert_eq!(initial_remaining, 3);
    assert_eq!(remaining_after_first_chunk, 2);
    assert_eq!(remaining_after_second_chunk, 1);
    assert_eq!(remaining_after_reset, 3);
    assert!(
        chunk1_samples > original_chunk0_samples,
        "scorecard longer chunk should produce more audio: chunk1={chunk1_samples} chunk0={original_chunk0_samples}",
    );
    assert!(
        reset_chunk0_samples > original_chunk0_samples,
        "scorecard reset chunk at speed 0.9 ({reset_chunk0_samples} samples) should be longer than original \
         chunk 0 at speed 1.0 ({original_chunk0_samples} samples)",
    );

    SessionStateMachineMeasurement {
        scope: ChorusMeasurementScope {
            name: "session_state_machine".to_string(),
            harness: ChorusProductionHarness::CloneDispatchStreamingSession,
            voice_count: 1,
            text_mode: ChorusTextMode::SharedText,
            warm_state: ChorusWarmState::WarmParentCache,
            checked_properties: vec![
                ChorusCheckedProperty::FiniteAudioSamples,
                ChorusCheckedProperty::SamplesWithinUnitRange,
                ChorusCheckedProperty::SessionStateTransitions,
                ChorusCheckedProperty::ResetAtAdjustedSpeedProducesLongerOutput,
            ],
        },
        initial_remaining,
        remaining_after_first_chunk,
        remaining_after_second_chunk,
        remaining_after_reset,
        reset_speed: 0.9,
        original_chunk0_samples,
        chunk1_samples,
        chunk2_samples,
        reset_chunk0_samples,
        runtime_dispatch,
    }
}

// -- Tests --------------------------------------------------------------------

/// Production weights: two voices via `clone_dispatch()` with interleaved
/// `StreamingKokoroSession::next_chunk()` calls produce valid audio.
///
/// Steps:
/// 1. Load production Kokoro, warmup.
/// 2. `clone_dispatch()` for second voice.
/// 3. Create two `StreamingKokoroSession` instances with 3 chunks each.
/// 4. Interleave `next_chunk()` calls between voices.
/// 5. Verify both produce valid audio (no NaN, all in [-1, 1]).
/// 6. Verify both complete (`is_done()`).
///
/// Part of #4105, #4265.
#[test]
fn test_production_two_voice_chorus_streaming() {
    let (mut parent, cache) = match load_production_kokoro(
        "production two-voice chorus streaming test skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let input_ids = make_production_input();
    let input_ids_long = make_production_input_long();
    let style = make_production_style();

    // Warmup: compile all 8 segments.
    let _ = parent
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("parent warmup");

    // Clone for second voice.
    let mut voice1 = parent.clone_dispatch();

    // Prepare 3 chunks per voice (mixed short + long inputs).
    let chunks_v0 = vec![
        (input_ids.clone(), style.clone()),
        (input_ids_long.clone(), style.clone()),
        (input_ids.clone(), style.clone()),
    ];
    let chunks_v1 = vec![
        (input_ids_long.clone(), style.clone()),
        (input_ids.clone(), style.clone()),
        (input_ids_long, style.clone()),
    ];

    let mut session0 = StreamingKokoroSession::new(chunks_v0, 1.0);
    let mut session1 = StreamingKokoroSession::new(chunks_v1, 1.0);

    assert_eq!(session0.remaining(), 3);
    assert_eq!(session1.remaining(), 3);

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

    // next_chunk() after completion returns None.
    assert!(session0.next_chunk(&mut parent, &cache).is_none());
    assert!(session1.next_chunk(&mut voice1, &cache).is_none());

    // Validate audio for both voices.
    validate_audio(&all_audio_v0, "voice0");
    validate_audio(&all_audio_v1, "voice1");

    eprintln!(
        "test_production_two_voice_chorus_streaming: v0={} samples, v1={} samples",
        all_audio_v0.len(),
        all_audio_v1.len(),
    );
}

/// Production weights: `clone_dispatch()` preserves the warm segment cache, so
/// the first clone synthesis should avoid the parent's cold-start compile cost.
///
/// Steps:
/// 1. Load production Kokoro.
/// 2. Time the first synthesis (cold: compiles all 8 segments).
/// 3. `clone_dispatch()`.
/// 4. Verify clone has 8 cached segments immediately.
/// 5. Time the clone's first synthesis (warm: all cache hits in this harness).
/// 6. Assert the warm clone run is faster than the parent's cold-start run.
///
/// Part of #4105, #4265.
#[test]
fn test_production_chorus_shared_compilation() {
    let (mut parent, cache) = match load_production_kokoro(
        "production chorus shared compilation test skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let input_ids = make_production_input();
    let style = make_production_style();

    // Cold start: compile all 8 segments.
    let t_cold = Instant::now();
    let _ = parent
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("parent cold synthesis");
    let cold_ms = t_cold.elapsed().as_secs_f64() * 1000.0;

    // After warmup, parent has 8 cached segments.
    let parent_cached = parent.total_cached_segments();
    assert_eq!(
        parent_cached, 8,
        "warmed parent should have 8 cached segments, got {parent_cached}"
    );

    // Clone: shares Arc-wrapped compiled segments.
    let mut clone = parent.clone_dispatch();

    // Clone should have 8 cached segments immediately.
    let clone_cached = clone.total_cached_segments();
    assert_eq!(
        clone_cached, 8,
        "clone should have 8 cached segments immediately, got {clone_cached}"
    );

    // Warm synthesis: clone uses shared segments (all cache hits).
    let t_warm = Instant::now();
    let (audio_clone, _cert) = clone
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("clone warm synthesis");
    let warm_ms = t_warm.elapsed().as_secs_f64() * 1000.0;

    // Validate clone audio.
    let clone_vals = audio_clone.to_flat_vec::<f32>().unwrap();
    validate_audio(&clone_vals, "clone");

    // Warm clone synthesis should beat the cold-start run in this harness.
    assert!(
        warm_ms < cold_ms,
        "clone warm synthesis ({warm_ms:.1}ms) should be faster than parent cold start \
         ({cold_ms:.1}ms)"
    );

    // Segments unchanged after clone synthesis.
    assert_eq!(clone.total_cached_segments(), 8);

    eprintln!(
        "test_production_chorus_shared_compilation: cold={cold_ms:.1}ms, warm={warm_ms:.1}ms, \
         speedup={:.1}x, parent_cached={parent_cached}, clone_cached={clone_cached}, \
         clone_samples={}",
        cold_ms / warm_ms,
        clone_vals.len(),
    );
}

/// Production weights: interleaving two clone-dispatch voices should not
/// perturb per-voice outputs relative to running the same chunk programs
/// sequentially on independent warmed clones.
///
/// Steps:
/// 1. Load production Kokoro, warm up all 8 segments once.
/// 2. Run voice 0 and voice 1 chunk programs sequentially on separate clones.
/// 3. Re-run the same chunk programs with interleaved `next_chunk()` calls.
/// 4. Verify each interleaved voice has the same sample count as its sequential
///    baseline and matches within a small floating-point tolerance.
///
/// Part of #4105, #4265.
#[test]
fn test_production_chorus_interleaved_matches_sequential_baselines() {
    let (mut parent, cache) = match load_production_kokoro(
        "production chorus sequential/interleaved parity test skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let input_ids = make_production_input();
    let input_ids_long = make_production_input_long();
    let style = make_production_style();

    // Warm once so every subsequent voice starts from the same 8-segment cache.
    let _ = parent
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("parent warmup");
    assert_eq!(parent.total_cached_segments(), 8);

    let chunks_v0 = vec![
        (input_ids.clone(), style.clone()),
        (input_ids_long.clone(), style.clone()),
        (input_ids.clone(), style.clone()),
    ];
    let chunks_v1 = vec![
        (input_ids_long.clone(), style.clone()),
        (input_ids.clone(), style.clone()),
        (input_ids_long, style.clone()),
    ];

    let mut seq_voice0 = parent.clone_dispatch();
    let mut seq_voice1 = parent.clone_dispatch();
    let mut seq_session0 = StreamingKokoroSession::new(chunks_v0.clone(), 1.0);
    let mut seq_session1 = StreamingKokoroSession::new(chunks_v1.clone(), 1.0);
    let baseline_v0 = collect_session_audio(&mut seq_session0, &mut seq_voice0, &cache, "seq v0");
    let baseline_v1 = collect_session_audio(&mut seq_session1, &mut seq_voice1, &cache, "seq v1");

    let mut interleaved_voice0 = parent.clone_dispatch();
    let mut interleaved_voice1 = parent.clone_dispatch();
    let mut interleaved_session0 = StreamingKokoroSession::new(chunks_v0, 1.0);
    let mut interleaved_session1 = StreamingKokoroSession::new(chunks_v1, 1.0);
    let mut interleaved_v0 = Vec::new();
    let mut interleaved_v1 = Vec::new();

    while !interleaved_session0.is_done() || !interleaved_session1.is_done() {
        if let Some(result) = interleaved_session0.next_chunk(&mut interleaved_voice0, &cache) {
            let (chunk, _cert) = result.expect("interleaved v0");
            interleaved_v0.extend(chunk.to_flat_vec::<f32>().unwrap());
        }
        if let Some(result) = interleaved_session1.next_chunk(&mut interleaved_voice1, &cache) {
            let (chunk, _cert) = result.expect("interleaved v1");
            interleaved_v1.extend(chunk.to_flat_vec::<f32>().unwrap());
        }
    }

    validate_audio(&interleaved_v0, "interleaved v0");
    validate_audio(&interleaved_v1, "interleaved v1");
    assert_eq!(interleaved_voice0.total_cached_segments(), 8);
    assert_eq!(interleaved_voice1.total_cached_segments(), 8);

    const PCM_TOLERANCE: f32 = 1e-5;
    let v0_max_diff = assert_audio_close(
        &baseline_v0,
        &interleaved_v0,
        "voice0 sequential parity",
        PCM_TOLERANCE,
    );
    let v1_max_diff = assert_audio_close(
        &baseline_v1,
        &interleaved_v1,
        "voice1 sequential parity",
        PCM_TOLERANCE,
    );

    eprintln!(
        "test_production_chorus_interleaved_matches_sequential_baselines: \
         v0_samples={} v1_samples={} v0_max_diff={v0_max_diff:.8} v1_max_diff={v1_max_diff:.8}",
        interleaved_v0.len(),
        interleaved_v1.len(),
    );
}

/// Production weights: two voices at different speeds (1.0 and 0.8) produce
/// different-length audio from identical input tokens.
///
/// Steps:
/// 1. Load production Kokoro, warmup.
/// 2. `clone_dispatch()` for second voice.
/// 3. Create two `StreamingKokoroSession` instances: speed 1.0 and speed 0.8.
/// 4. Consume all chunks from both sessions.
/// 5. Verify both produce valid audio.
/// 6. Verify the slower voice (0.8) produces more audio samples.
///
/// Part of #4105, #4265.
#[test]
fn test_production_chorus_different_speeds() {
    let (mut parent, cache) = match load_production_kokoro(
        "production chorus different speeds test skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let input_ids = make_production_input();
    let style = make_production_style();

    // Warmup.
    let _ = parent
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("parent warmup");

    // Clone for second voice.
    let mut voice_slow = parent.clone_dispatch();

    // Prepare 2 chunks per voice (same input tokens).
    let chunks_fast = vec![
        (input_ids.clone(), style.clone()),
        (input_ids.clone(), style.clone()),
    ];
    let chunks_slow = vec![
        (input_ids.clone(), style.clone()),
        (input_ids.clone(), style.clone()),
    ];

    let mut session_fast = StreamingKokoroSession::new(chunks_fast, 1.0);
    let mut session_slow = StreamingKokoroSession::new(chunks_slow, 0.8);

    // Consume all chunks from both sessions.
    let mut audio_fast: Vec<f32> = Vec::new();
    let mut audio_slow: Vec<f32> = Vec::new();

    while !session_fast.is_done() || !session_slow.is_done() {
        if let Some(result) = session_fast.next_chunk(&mut parent, &cache) {
            let (audio, _cert) = result.expect("fast voice chunk synthesis");
            audio_fast.extend(audio.to_flat_vec::<f32>().unwrap());
        }
        if let Some(result) = session_slow.next_chunk(&mut voice_slow, &cache) {
            let (audio, _cert) = result.expect("slow voice chunk synthesis");
            audio_slow.extend(audio.to_flat_vec::<f32>().unwrap());
        }
    }

    assert!(session_fast.is_done());
    assert!(session_slow.is_done());

    // Validate audio for both voices.
    validate_audio(&audio_fast, "fast (1.0)");
    validate_audio(&audio_slow, "slow (0.8)");

    // Slower speed (0.8) should produce longer audio than normal speed (1.0).
    // Duration is inversely proportional to speed: 0.8x speed => ~1.25x duration.
    assert!(
        audio_slow.len() > audio_fast.len(),
        "slow voice (0.8x speed, {} samples) should produce more audio than \
         fast voice (1.0x speed, {} samples)",
        audio_slow.len(),
        audio_fast.len(),
    );

    // Verify the ratio is approximately 1.25x (tolerance: 1.1x to 1.5x).
    let ratio = audio_slow.len() as f64 / audio_fast.len() as f64;
    assert!(
        (1.1..=1.5).contains(&ratio),
        "slow/fast audio length ratio should be ~1.25, got {ratio:.3} \
         (slow={}, fast={})",
        audio_slow.len(),
        audio_fast.len(),
    );

    eprintln!(
        "test_production_chorus_different_speeds: fast={} samples, slow={} samples, \
         ratio={ratio:.3}",
        audio_fast.len(),
        audio_slow.len(),
    );
}

/// Production weights: session state machine (`remaining()`, `is_done()`,
/// `reset()`) works correctly with real synthesis through full lifecycle.
///
/// Steps:
/// 1. Load production Kokoro, warmup.
/// 2. Create a 3-chunk `StreamingKokoroSession`.
/// 3. Consume all 3 chunks, verifying state at each step.
/// 4. Verify `is_done()`, `remaining() == 0`.
/// 5. `reset()` and consume first chunk again.
/// 6. Verify audio from the reset pass is valid.
///
/// Part of #4105, #4265.
#[test]
fn test_production_chorus_session_state_machine() {
    let (mut parent, cache) = match load_production_kokoro(
        "production chorus session state machine test skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let input_ids = make_production_input();
    let input_ids_long = make_production_input_long();
    let style = make_production_style();

    // Warmup.
    let _ = parent
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("parent warmup");

    // Build 3 chunks for the session.
    let chunks = vec![
        (input_ids.clone(), style.clone()),
        (input_ids_long, style.clone()),
        (input_ids.clone(), style.clone()),
    ];
    let mut session = StreamingKokoroSession::new(chunks, 1.0);

    // Initial state: 3 chunks, not done.
    assert_eq!(session.total_chunks(), 3);
    assert_eq!(session.remaining(), 3);
    assert_eq!(session.synthesized_count(), 0);
    assert!(!session.is_done());
    assert!((session.speed() - 1.0).abs() < f32::EPSILON);

    // Consume chunk 0.
    let r = session.next_chunk(&mut parent, &cache);
    assert!(r.is_some(), "chunk 0 should be available");
    let (audio0, _cert) = r.unwrap().expect("chunk 0 synthesis");
    let audio0_vals = audio0.to_flat_vec::<f32>().unwrap();
    validate_audio(&audio0_vals, "chunk 0");
    assert_eq!(session.remaining(), 2);
    assert_eq!(session.synthesized_count(), 1);
    assert!(!session.is_done());

    // Consume chunk 1 (longer input).
    let r = session.next_chunk(&mut parent, &cache);
    assert!(r.is_some(), "chunk 1 should be available");
    let (audio1, _cert) = r.unwrap().expect("chunk 1 synthesis");
    let audio1_vals = audio1.to_flat_vec::<f32>().unwrap();
    validate_audio(&audio1_vals, "chunk 1");
    assert_eq!(session.remaining(), 1);
    assert_eq!(session.synthesized_count(), 2);
    assert!(!session.is_done());

    // Longer input should produce more audio.
    assert!(
        audio1_vals.len() > audio0_vals.len(),
        "chunk 1 (12 tokens, {} samples) should produce more audio than \
         chunk 0 (8 tokens, {} samples)",
        audio1_vals.len(),
        audio0_vals.len(),
    );

    // Consume chunk 2.
    let r = session.next_chunk(&mut parent, &cache);
    assert!(r.is_some(), "chunk 2 should be available");
    let (audio2, _cert) = r.unwrap().expect("chunk 2 synthesis");
    let audio2_vals = audio2.to_flat_vec::<f32>().unwrap();
    validate_audio(&audio2_vals, "chunk 2");
    assert_eq!(session.remaining(), 0);
    assert_eq!(session.synthesized_count(), 3);
    assert!(session.is_done());

    // No more chunks.
    assert!(session.next_chunk(&mut parent, &cache).is_none());

    // Reset brings us back to the beginning.
    session.reset();
    assert_eq!(session.remaining(), 3);
    assert_eq!(session.synthesized_count(), 0);
    assert!(!session.is_done());

    // Change speed and consume first chunk again.
    session.set_speed(0.9);
    assert!((session.speed() - 0.9).abs() < f32::EPSILON);

    let r = session.next_chunk(&mut parent, &cache);
    assert!(r.is_some(), "chunk 0 should be available after reset");
    let (audio_reset, _cert) = r.unwrap().expect("chunk 0 synthesis after reset");
    let audio_reset_vals = audio_reset.to_flat_vec::<f32>().unwrap();
    validate_audio(&audio_reset_vals, "chunk 0 after reset");
    assert_eq!(session.remaining(), 2);
    assert_eq!(session.synthesized_count(), 1);

    // Audio after reset at slower speed should be longer than original.
    assert!(
        audio_reset_vals.len() > audio0_vals.len(),
        "chunk 0 at speed 0.9 ({} samples) should produce more audio than \
         at speed 1.0 ({} samples)",
        audio_reset_vals.len(),
        audio0_vals.len(),
    );

    eprintln!(
        "test_production_chorus_session_state_machine: chunk0={} chunk1={} chunk2={} \
         reset_chunk0={} samples, all state transitions verified",
        audio0_vals.len(),
        audio1_vals.len(),
        audio2_vals.len(),
        audio_reset_vals.len(),
    );
}

/// Production weights: emit a structured, machine-readable scorecard for the
/// real chorus production test surface.
///
/// This scorecard is intentionally explicit about scope:
/// - `measured_surface` is per-voice `StreamingKokoroSession` output, not mixed `KokoroChorus`.
/// - `measured_harness` is the clone-dispatch + `StreamingKokoroSession` harness in this file.
/// - `recommended_autocast_enabled` records harness configuration only; this file currently
///   measures the non-recommended-autocast variant of that harness.
/// - It does not claim mixed `KokoroChorus` output quality.
/// - It does not claim production throughput targets.
/// - It reports only the measurements and parity checks actually exercised here.
#[test]
fn test_production_chorus_scorecard() {
    let (mut parent, cache) = match load_production_kokoro(
        "production chorus scorecard skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let input_ids = make_production_input();
    let input_ids_long = make_production_input_long();
    let style = make_production_style();

    let shared_compilation = measure_shared_compilation(&mut parent, &cache, &input_ids, &style);
    let interleaved_streaming = measure_two_voice_interleaved_streaming(
        &mut parent,
        &cache,
        &input_ids,
        &input_ids_long,
        &style,
    );
    let interleaved_parity =
        measure_interleaved_parity(&mut parent, &cache, &input_ids, &input_ids_long, &style);
    let speed_variation = measure_speed_variation(&mut parent, &cache, &input_ids, &style);
    let session_state_machine =
        measure_session_state_machine(&mut parent, &cache, &input_ids, &input_ids_long, &style);

    let scorecard = ChorusProductionScorecard {
        schema_version: CHORUS_SCORECARD_SCHEMA_VERSION,
        suite_name: "kokoro_production_chorus".to_string(),
        build_profile: build_profile_name().to_string(),
        measured_surface: ChorusMeasuredSurface::PerVoiceStreamingSessions,
        measured_harness: ChorusProductionHarness::CloneDispatchStreamingSession,
        recommended_autocast_enabled: false,
        limitations: vec![
            ChorusScorecardLimitation::NotMixedKokoroChorusSurface,
            ChorusScorecardLimitation::NotThroughputBenchmark,
            ChorusScorecardLimitation::NotAudioQualityCertification,
        ],
        compiled_dispatch: compiled_dispatch_scorecard(&parent),
        measurements: vec![
            ChorusProductionMeasurement::SharedCompilation(shared_compilation),
            ChorusProductionMeasurement::TwoVoiceInterleavedStreaming(interleaved_streaming),
            ChorusProductionMeasurement::InterleavedParity(interleaved_parity),
            ChorusProductionMeasurement::SpeedVariation(speed_variation),
            ChorusProductionMeasurement::SessionStateMachine(session_state_machine),
        ],
    };
    assert!(
        scorecard.measurements_match_declared_harness(),
        "production chorus scorecard measurements should match the declared harness"
    );

    emit_chorus_scorecard(&scorecard);

    eprintln!(
        "test_production_chorus_scorecard: measurements={}, logical_dispatches={}, \
         estimated_metal_dispatches={}",
        scorecard.measurements.len(),
        scorecard.compiled_dispatch.logical_total,
        scorecard.compiled_dispatch.estimated_metal_total,
    );
}

#[test]
fn test_production_chorus_scorecard_json_round_trip() {
    let scorecard = sample_chorus_production_scorecard();

    let compact_json = scorecard
        .to_compact_json()
        .expect("serialize compact chorus scorecard");
    let compact_parsed: ChorusProductionScorecard =
        serde_json::from_str(&compact_json).expect("deserialize compact chorus scorecard");

    let pretty_json = scorecard
        .to_pretty_json()
        .expect("serialize pretty chorus scorecard");
    let pretty_parsed: ChorusProductionScorecard =
        serde_json::from_str(&pretty_json).expect("deserialize pretty chorus scorecard");

    assert_eq!(compact_parsed, scorecard);
    assert_eq!(pretty_parsed, scorecard);
    assert_eq!(compact_parsed.measurements.len(), 3);
    assert_eq!(
        compact_parsed.measured_surface,
        ChorusMeasuredSurface::PerVoiceStreamingSessions
    );
    assert_eq!(
        compact_parsed.measured_harness,
        ChorusProductionHarness::CloneDispatchStreamingSession
    );
    assert!(
        !compact_parsed.recommended_autocast_enabled,
        "sample chorus scorecard should declare the current non-autocast harness"
    );
    assert!(
        compact_parsed.measurements_match_declared_harness(),
        "sample chorus scorecard measurements should match the declared harness"
    );
    assert_eq!(
        compact_parsed.compiled_dispatch.expected_submit_count, 6,
        "expected submit count should survive round-trip"
    );
    assert_eq!(compact_parsed.limitations, scorecard.limitations);
}

#[test]
fn test_production_chorus_scorecard_write_json_artifact() {
    let scorecard = sample_chorus_production_scorecard();

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    let out_dir = std::env::temp_dir().join(format!(
        "nn-kokoro-production-chorus-scorecard-{}-{unique}",
        std::process::id()
    ));
    let out_path = out_dir.join("scorecards/production-chorus.json");

    scorecard
        .write_json(&out_path)
        .expect("write chorus scorecard artifact");

    let persisted = fs::read_to_string(&out_path).expect("read persisted chorus scorecard");
    assert!(
        persisted.contains('\n'),
        "artifact writer should emit pretty JSON"
    );
    let parsed: ChorusProductionScorecard =
        serde_json::from_str(&persisted).expect("deserialize persisted chorus scorecard");

    assert_eq!(parsed.schema_version, CHORUS_SCORECARD_SCHEMA_VERSION);
    assert_eq!(parsed.measurements.len(), 3);
    assert_eq!(
        parsed.measured_surface,
        ChorusMeasuredSurface::PerVoiceStreamingSessions
    );
    assert_eq!(
        parsed.measured_harness,
        ChorusProductionHarness::CloneDispatchStreamingSession
    );
    assert!(
        !parsed.recommended_autocast_enabled,
        "persisted chorus scorecard should declare the current non-autocast harness"
    );
    assert!(
        parsed.measurements_match_declared_harness(),
        "persisted chorus scorecard measurements should match the declared harness"
    );
    assert_eq!(parsed.compiled_dispatch.expected_submit_count, 6);
    assert_eq!(parsed.limitations, scorecard.limitations);

    let _ = fs::remove_file(&out_path);
    let _ = fs::remove_dir_all(&out_dir);
}

#[test]
fn test_production_chorus_scorecard_stderr_prefix_contract() {
    let scorecard = sample_chorus_production_scorecard();
    let stderr_line =
        chorus_scorecard_stderr_line(&scorecard).expect("compact chorus scorecard stderr line");

    assert!(
        stderr_line.starts_with(CHORUS_SCORECARD_STDERR_PREFIX),
        "stderr line must start with the stable chorus scorecard prefix"
    );

    let json_payload = stderr_line
        .strip_prefix(CHORUS_SCORECARD_STDERR_PREFIX)
        .expect("stderr line should contain the stable chorus scorecard prefix");
    let parsed: ChorusProductionScorecard =
        serde_json::from_str(json_payload).expect("deserialize chorus scorecard stderr payload");

    assert_eq!(parsed, scorecard);
    assert_eq!(
        parsed.measured_surface,
        ChorusMeasuredSurface::PerVoiceStreamingSessions
    );
    assert_eq!(
        parsed.measured_harness,
        ChorusProductionHarness::CloneDispatchStreamingSession
    );
    assert!(
        !parsed.recommended_autocast_enabled,
        "stderr chorus scorecard should declare the current non-autocast harness"
    );
    assert!(
        parsed.measurements_match_declared_harness(),
        "stderr chorus scorecard measurements should match the declared harness"
    );
}

#[test]
fn test_production_chorus_scorecard_emit_honors_env_artifact_path() {
    let _guard = CHORUS_SCORECARD_ENV_LOCK
        .lock()
        .expect("lock env for chorus scorecard test");
    let scorecard = sample_chorus_production_scorecard();

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    let out_dir = std::env::temp_dir().join(format!(
        "nn-kokoro-production-chorus-scorecard-env-{}-{unique}",
        std::process::id()
    ));
    let out_path = out_dir.join("scorecards/production-chorus-from-env.json");
    let out_path_string = out_path.to_string_lossy().into_owned();

    let previous = std::env::var_os(CHORUS_SCORECARD_OUT_ENV);
    std::env::set_var(CHORUS_SCORECARD_OUT_ENV, &out_path_string);

    let configured_path = configured_chorus_scorecard_artifact_path(CHORUS_SCORECARD_OUT_ENV)
        .expect("configured chorus scorecard artifact path");
    assert_eq!(configured_path, out_path_string);

    emit_chorus_scorecard(&scorecard);

    let persisted =
        fs::read_to_string(&out_path).expect("read env-driven chorus scorecard artifact");
    let parsed: ChorusProductionScorecard =
        serde_json::from_str(&persisted).expect("deserialize env-driven chorus scorecard");
    assert_eq!(parsed, scorecard);
    assert_eq!(
        parsed.measured_surface,
        ChorusMeasuredSurface::PerVoiceStreamingSessions
    );
    assert_eq!(
        parsed.measured_harness,
        ChorusProductionHarness::CloneDispatchStreamingSession
    );
    assert!(
        !parsed.recommended_autocast_enabled,
        "env-driven chorus scorecard should declare the current non-autocast harness"
    );
    assert!(
        parsed.measurements_match_declared_harness(),
        "env-driven chorus scorecard measurements should match the declared harness"
    );

    match previous {
        Some(value) => std::env::set_var(CHORUS_SCORECARD_OUT_ENV, value),
        None => std::env::remove_var(CHORUS_SCORECARD_OUT_ENV),
    }

    let _ = fs::remove_file(&out_path);
    let _ = fs::remove_dir_all(&out_dir);
}
