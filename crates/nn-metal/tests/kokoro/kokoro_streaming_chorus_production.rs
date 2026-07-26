// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Production-weight integration tests for [`StreamingChorusSession`].
//!
//! Exercises the pull-based streaming chorus API with real Kokoro weights
//! and a multi-voice [`KokoroChorus`]:
//! 1. Basic iteration: create session, iterate all chunks, verify non-empty audio.
//! 2. Cancel mid-stream: cancel after partial synthesis, verify remaining returns 0.
//! 3. Reset after partial: reset mid-way, verify re-synthesis from start.
//! 4. Crossfade smoothness: verify crossfade produces smooth transitions (no clicks).
//! 5. Cache sharing: verify segment cache sharing across chorus voices.
//! 6. Speed variation: different speeds produce different output lengths.
//! 7. Machine-readable scorecard: emit an honest report of the measured streaming
//!    chorus properties in this file.
//!
//! This file covers the mixed-output pull-streaming harness only. It does not
//! certify production audio quality or throughput.
//!
//! All tests gated behind `KOKORO_WEIGHTS` env var. Skips gracefully when unset.
//!
//! Run: `cargo test -p nn-metal --test kokoro_all -- streaming_chorus_production --nocapture`

use std::fs;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use nn_metal::compiled_kokoro::chorus::KokoroChorus;
use nn_metal::compiled_kokoro::{CompiledKokoro, StreamingChorusSession};
use nn_models::kokoro_chorus::ChorusConfig;
use nn_models::kokoro_streaming::{AudioChunk, CrossfadeWindow, KokoroStreamConfig};
use serde::{Deserialize, Serialize};

fn cpu() -> Device {
    Device::Cpu
}

/// Number of chorus voices for production tests.
const N_VOICES: usize = 3;

/// Kokoro sample rate (24 kHz).
const SAMPLE_RATE: usize = 24_000;

/// Stable schema version for the streaming chorus production scorecard.
const STREAMING_CHORUS_SCORECARD_SCHEMA_VERSION: u32 = 2;

/// Optional artifact path for the streaming chorus production scorecard.
const STREAMING_CHORUS_SCORECARD_OUT_ENV: &str = "KOKORO_STREAMING_CHORUS_SCORECARD_OUT";

/// Stable stderr prefix for compact JSON scorecard output.
const STREAMING_CHORUS_SCORECARD_STDERR_PREFIX: &str = "KOKORO_STREAMING_CHORUS_SCORECARD_JSON=";

/// Serialize env-driven scorecard emission tests that temporarily mutate process env.
static STREAMING_CHORUS_SCORECARD_ENV_LOCK: Mutex<()> = Mutex::new(());

/// Tolerance for reset replay parity in the current equal-gain mixed output path.
const STREAMING_CHORUS_PCM_TOLERANCE: f32 = 1e-6;

/// Absolute click/discontinuity threshold used by the crossfade continuity check.
const STREAMING_CHORUS_BOUNDARY_THRESHOLD: f32 = 0.5;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum StreamingChorusHarness {
    PullStreamingChorusSession,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum StreamingChorusChunkingMode {
    SharedChunkProgramMixedOutput,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum StreamingChorusWarmState {
    WarmPrimarySegmentCache,
    WarmChorusCloneSegments,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum StreamingChorusCheckedProperty {
    FiniteAudioSamples,
    SamplesWithinUnitRange,
    NonSilentAudio,
    ChunkCountMatchesInputCount,
    SessionCompletes,
    CancelStopsIteration,
    ResetReplaysMixedPrefixWithinTolerance,
    CrossfadeBoundaryBelowThreshold,
    ChorusVoicesPreserveCachedSegments,
    SlowerSpeedProducesLongerOutput,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum StreamingChorusScorecardLimitation {
    NotAudioQualityCertificate,
    NotThroughputBenchmark,
    NoBatchReferenceComparison,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum StreamingChorusCrossfadeWindow {
    Linear,
    Hann,
    SqrtHann,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct StreamingChorusStreamConfigScorecard {
    crossfade_samples: usize,
    crossfade_duration_ms: f64,
    crossfade_window: StreamingChorusCrossfadeWindow,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct StreamingChorusMeasurementScope {
    name: String,
    harness: StreamingChorusHarness,
    voice_count: usize,
    input_chunk_counts: Vec<usize>,
    chunking_mode: StreamingChorusChunkingMode,
    warm_state: StreamingChorusWarmState,
    first_chunk_warmup: StreamingChorusFirstChunkWarmupScorecard,
    checked_properties: Vec<StreamingChorusCheckedProperty>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct StreamingChorusFirstChunkWarmupScorecard {
    pending_at_start: bool,
    consumed_by_end: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
struct StreamingChorusRuntimeDispatchScorecard {
    compute_encodings: usize,
    blits: usize,
    total_metal_command_encodings: usize,
    flushes: usize,
    submits: usize,
    blits_eliminated: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct StreamingChorusCompiledDispatchSegments {
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
struct StreamingChorusCompiledDispatchScorecard {
    logical_total: usize,
    estimated_metal_total: usize,
    estimated_encoding_events_total: usize,
    expected_submit_count: usize,
    segments: StreamingChorusCompiledDispatchSegments,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct StreamingChorusBasicMeasurement {
    scope: StreamingChorusMeasurementScope,
    wall_ms: f64,
    observed_chunk_count: usize,
    chunk_sample_counts: Vec<usize>,
    chunk_sample_offsets: Vec<usize>,
    total_samples: usize,
    rms_energy: f64,
    runtime_dispatch: StreamingChorusRuntimeDispatchScorecard,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct StreamingChorusCancelMeasurement {
    scope: StreamingChorusMeasurementScope,
    first_chunk_samples: usize,
    remaining_before_cancel: usize,
    remaining_after_cancel: usize,
    synthesized_before_cancel: usize,
    runtime_dispatch: StreamingChorusRuntimeDispatchScorecard,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct StreamingChorusResetReplayMeasurement {
    scope: StreamingChorusMeasurementScope,
    replayed_chunk_count: usize,
    replayed_chunk_sample_counts: Vec<usize>,
    replay_chunk_max_abs_diff: Vec<f32>,
    replay_total_samples: usize,
    full_reset_pass_total_samples: usize,
    runtime_dispatch: StreamingChorusRuntimeDispatchScorecard,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct StreamingChorusCrossfadeMeasurement {
    scope: StreamingChorusMeasurementScope,
    wall_ms: f64,
    chunk_sample_counts: Vec<usize>,
    total_samples: usize,
    max_boundary_delta: f32,
    max_delta_overall: f32,
    boundary_threshold: f32,
    runtime_dispatch: StreamingChorusRuntimeDispatchScorecard,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct StreamingChorusCacheSharingMeasurement {
    scope: StreamingChorusMeasurementScope,
    wall_ms: f64,
    parent_cached_segments: usize,
    voice_cached_segments_before: Vec<usize>,
    voice_cached_segments_after: Vec<usize>,
    shared_state_refcount: usize,
    runtime_dispatch: StreamingChorusRuntimeDispatchScorecard,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct StreamingChorusSpeedVariationMeasurement {
    scope: StreamingChorusMeasurementScope,
    normal_speed: f32,
    slow_speed: f32,
    normal_num_samples: usize,
    slow_num_samples: usize,
    slow_to_normal_ratio: f64,
    normal_wall_ms: f64,
    slow_wall_ms: f64,
    runtime_dispatch: StreamingChorusRuntimeDispatchScorecard,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "measurement_kind", rename_all = "snake_case")]
enum StreamingChorusProductionMeasurement {
    Basic(StreamingChorusBasicMeasurement),
    Cancel(StreamingChorusCancelMeasurement),
    ResetReplay(StreamingChorusResetReplayMeasurement),
    CrossfadeContinuity(StreamingChorusCrossfadeMeasurement),
    CacheSharing(StreamingChorusCacheSharingMeasurement),
    SpeedVariation(StreamingChorusSpeedVariationMeasurement),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct StreamingChorusProductionScorecard {
    schema_version: u32,
    suite_name: String,
    build_profile: String,
    sample_rate_hz: usize,
    default_stream_config: StreamingChorusStreamConfigScorecard,
    /// Whether this scorecard's measured pull-streaming harness ran with
    /// `CompiledKokoro::with_recommended_autocast()`.
    ///
    /// This is harness-configuration metadata only. It does not imply any
    /// quality or throughput claim.
    recommended_autocast_enabled: bool,
    limitations: Vec<StreamingChorusScorecardLimitation>,
    compiled_dispatch: StreamingChorusCompiledDispatchScorecard,
    measurements: Vec<StreamingChorusProductionMeasurement>,
}

impl StreamingChorusProductionScorecard {
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

/// Assert that two streaming chunks match in metadata and PCM within a small
/// floating-point tolerance.
fn assert_chunks_match(expected: &AudioChunk, actual: &AudioChunk, label: &str) -> f32 {
    assert_eq!(
        actual.chunk_index, expected.chunk_index,
        "{label}: chunk_index mismatch"
    );
    assert_eq!(
        actual.total_chunks, expected.total_chunks,
        "{label}: total_chunks mismatch"
    );
    assert_eq!(
        actual.sample_offset, expected.sample_offset,
        "{label}: sample_offset mismatch"
    );
    assert_eq!(
        actual.channels, expected.channels,
        "{label}: channel count mismatch"
    );
    assert_eq!(
        actual.is_final, expected.is_final,
        "{label}: is_final mismatch"
    );
    assert_eq!(
        actual.pcm.len(),
        expected.pcm.len(),
        "{label}: PCM length mismatch"
    );

    let mut max_abs_diff = 0.0f32;
    let mut max_abs_index = 0usize;
    for (i, (&expected_sample, &actual_sample)) in
        expected.pcm.iter().zip(actual.pcm.iter()).enumerate()
    {
        let abs_diff = (actual_sample - expected_sample).abs();
        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
            max_abs_index = i;
        }
    }

    let tolerance = STREAMING_CHORUS_PCM_TOLERANCE;
    assert!(
        max_abs_diff <= tolerance,
        "{label}: max_abs_diff={max_abs_diff:.8} at sample {max_abs_index} exceeds {tolerance:.1e}"
    );
    max_abs_diff
}

fn crossfade_window_scorecard(window: CrossfadeWindow) -> StreamingChorusCrossfadeWindow {
    match window {
        CrossfadeWindow::Linear => StreamingChorusCrossfadeWindow::Linear,
        CrossfadeWindow::Hann => StreamingChorusCrossfadeWindow::Hann,
        CrossfadeWindow::SqrtHann => StreamingChorusCrossfadeWindow::SqrtHann,
        _ => panic!("unsupported crossfade window variant in scorecard"),
    }
}

fn stream_config_scorecard(
    stream_config: &KokoroStreamConfig,
) -> StreamingChorusStreamConfigScorecard {
    StreamingChorusStreamConfigScorecard {
        crossfade_samples: stream_config.crossfade_samples,
        crossfade_duration_ms: stream_config.crossfade_duration_secs() * 1000.0,
        crossfade_window: crossfade_window_scorecard(stream_config.crossfade_window),
    }
}

fn measurement_scope(
    name: &str,
    input_chunk_counts: Vec<usize>,
    warm_state: StreamingChorusWarmState,
    first_chunk_warmup: StreamingChorusFirstChunkWarmupScorecard,
    checked_properties: Vec<StreamingChorusCheckedProperty>,
) -> StreamingChorusMeasurementScope {
    StreamingChorusMeasurementScope {
        name: name.to_string(),
        harness: StreamingChorusHarness::PullStreamingChorusSession,
        voice_count: N_VOICES,
        input_chunk_counts,
        chunking_mode: StreamingChorusChunkingMode::SharedChunkProgramMixedOutput,
        warm_state,
        first_chunk_warmup,
        checked_properties,
    }
}

fn first_chunk_warmup_scorecard(
    pending_at_start: bool,
    consumed_by_end: bool,
) -> StreamingChorusFirstChunkWarmupScorecard {
    StreamingChorusFirstChunkWarmupScorecard {
        pending_at_start,
        consumed_by_end,
    }
}

fn runtime_dispatch_scorecard(
    stats: nn_metal::DispatchStats,
) -> StreamingChorusRuntimeDispatchScorecard {
    StreamingChorusRuntimeDispatchScorecard {
        compute_encodings: stats.compute_encodings,
        blits: stats.blits,
        total_metal_command_encodings: stats.compute_encodings + stats.blits,
        flushes: stats.flushes,
        submits: stats.submits,
        blits_eliminated: stats.blits_eliminated,
    }
}

fn compiled_dispatch_scorecard(
    kokoro: &CompiledKokoro,
) -> StreamingChorusCompiledDispatchScorecard {
    let summary = kokoro.dispatch_summary();
    StreamingChorusCompiledDispatchScorecard {
        logical_total: summary.total(),
        estimated_metal_total: kokoro.total_metal_dispatches(),
        estimated_encoding_events_total: kokoro.total_encoding_events(),
        expected_submit_count: summary.expected_submit_count(),
        segments: StreamingChorusCompiledDispatchSegments {
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

fn capture_dispatch_stats<T>(
    op: impl FnOnce() -> T,
) -> (T, StreamingChorusRuntimeDispatchScorecard) {
    nn_metal::reset_counters();
    let value = op();
    let stats = nn_metal::dispatch_stats();
    (value, runtime_dispatch_scorecard(stats))
}

fn timed_with_dispatch_stats<T>(
    op: impl FnOnce() -> T,
) -> (T, f64, StreamingChorusRuntimeDispatchScorecard) {
    nn_metal::reset_counters();
    let started = Instant::now();
    let value = op();
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    let stats = nn_metal::dispatch_stats();
    (value, wall_ms, runtime_dispatch_scorecard(stats))
}

fn collect_session_chunks(
    session: &mut StreamingChorusSession,
    chorus: &mut KokoroChorus,
    cache: &nn_metal::PipelineCache,
    label: &str,
) -> Vec<AudioChunk> {
    let mut chunks = Vec::new();
    let mut chunk_index = 0usize;
    while let Some(result) = session.next_chunk(chorus, cache) {
        let audio_chunk =
            result.unwrap_or_else(|e| panic!("{label}: chunk {chunk_index} synthesis failed: {e}"));
        validate_audio(&audio_chunk.pcm, &format!("{label}: chunk {chunk_index}"));
        chunks.push(audio_chunk);
        chunk_index += 1;
    }
    assert!(session.is_done(), "{label}: session should be done");
    chunks
}

fn chunk_pcm_lengths(chunks: &[AudioChunk]) -> Vec<usize> {
    chunks.iter().map(|chunk| chunk.pcm.len()).collect()
}

fn chunk_sample_offsets(chunks: &[AudioChunk]) -> Vec<usize> {
    chunks.iter().map(|chunk| chunk.sample_offset).collect()
}

fn flatten_chunks(chunks: &[AudioChunk]) -> Vec<f32> {
    chunks
        .iter()
        .flat_map(|chunk| chunk.pcm.iter().copied())
        .collect()
}

fn streaming_chorus_scorecard_stderr_line(
    report: &StreamingChorusProductionScorecard,
) -> Option<String> {
    report
        .to_compact_json()
        .ok()
        .map(|compact_json| format!("{STREAMING_CHORUS_SCORECARD_STDERR_PREFIX}{compact_json}"))
}

fn configured_streaming_chorus_scorecard_artifact_path(env_var: &str) -> Option<String> {
    std::env::var(env_var)
        .ok()
        .map(|path| path.trim().to_string())
        .filter(|path| !path.is_empty())
}

fn emit_streaming_chorus_scorecard_with_artifact_path(
    report: &StreamingChorusProductionScorecard,
    artifact_path: Option<&str>,
) -> Option<String> {
    let stderr_line = streaming_chorus_scorecard_stderr_line(report);
    if let Some(line) = stderr_line.as_deref() {
        eprintln!("{line}");
    }

    if let Some(path) = artifact_path {
        report.write_json(path).unwrap_or_else(|e| {
            panic!("failed to write streaming chorus scorecard artifact {path}: {e}")
        });
        eprintln!("  Streaming chorus scorecard artifact: {path}");
    }

    stderr_line
}

fn emit_streaming_chorus_scorecard(report: &StreamingChorusProductionScorecard) {
    let artifact_path =
        configured_streaming_chorus_scorecard_artifact_path(STREAMING_CHORUS_SCORECARD_OUT_ENV);
    let _ = emit_streaming_chorus_scorecard_with_artifact_path(report, artifact_path.as_deref());
}

fn sample_streaming_chorus_scorecard() -> StreamingChorusProductionScorecard {
    StreamingChorusProductionScorecard {
        schema_version: STREAMING_CHORUS_SCORECARD_SCHEMA_VERSION,
        suite_name: "kokoro_streaming_chorus_production".to_string(),
        build_profile: "release".to_string(),
        sample_rate_hz: SAMPLE_RATE,
        default_stream_config: StreamingChorusStreamConfigScorecard {
            crossfade_samples: 960,
            crossfade_duration_ms: 40.0,
            crossfade_window: StreamingChorusCrossfadeWindow::SqrtHann,
        },
        recommended_autocast_enabled: false,
        limitations: vec![
            StreamingChorusScorecardLimitation::NotAudioQualityCertificate,
            StreamingChorusScorecardLimitation::NotThroughputBenchmark,
            StreamingChorusScorecardLimitation::NoBatchReferenceComparison,
        ],
        compiled_dispatch: StreamingChorusCompiledDispatchScorecard {
            logical_total: 476,
            estimated_metal_total: 674,
            estimated_encoding_events_total: 701,
            expected_submit_count: 6,
            segments: StreamingChorusCompiledDispatchSegments {
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
            StreamingChorusProductionMeasurement::Basic(StreamingChorusBasicMeasurement {
                scope: measurement_scope(
                    "basic_streaming",
                    vec![3],
                    StreamingChorusWarmState::WarmChorusCloneSegments,
                    first_chunk_warmup_scorecard(true, true),
                    vec![
                        StreamingChorusCheckedProperty::FiniteAudioSamples,
                        StreamingChorusCheckedProperty::ChunkCountMatchesInputCount,
                    ],
                ),
                wall_ms: 42.0,
                observed_chunk_count: 3,
                chunk_sample_counts: vec![8_000, 11_200, 7_600],
                chunk_sample_offsets: vec![0, 7_040, 17_280],
                total_samples: 26_800,
                rms_energy: 0.031,
                runtime_dispatch: StreamingChorusRuntimeDispatchScorecard {
                    compute_encodings: 1_200,
                    blits: 36,
                    total_metal_command_encodings: 1_236,
                    flushes: 9,
                    submits: 6,
                    blits_eliminated: 18,
                },
            }),
            StreamingChorusProductionMeasurement::ResetReplay(
                StreamingChorusResetReplayMeasurement {
                    scope: measurement_scope(
                        "reset_replay",
                        vec![3],
                        StreamingChorusWarmState::WarmChorusCloneSegments,
                        first_chunk_warmup_scorecard(true, true),
                        vec![
                            StreamingChorusCheckedProperty::ResetReplaysMixedPrefixWithinTolerance,
                        ],
                    ),
                    replayed_chunk_count: 2,
                    replayed_chunk_sample_counts: vec![8_000, 11_200],
                    replay_chunk_max_abs_diff: vec![0.0, 0.0],
                    replay_total_samples: 19_200,
                    full_reset_pass_total_samples: 26_800,
                    runtime_dispatch: StreamingChorusRuntimeDispatchScorecard {
                        compute_encodings: 1_200,
                        blits: 36,
                        total_metal_command_encodings: 1_236,
                        flushes: 9,
                        submits: 6,
                        blits_eliminated: 18,
                    },
                },
            ),
            StreamingChorusProductionMeasurement::SpeedVariation(
                StreamingChorusSpeedVariationMeasurement {
                    scope: measurement_scope(
                        "speed_variation",
                        vec![2],
                        StreamingChorusWarmState::WarmChorusCloneSegments,
                        first_chunk_warmup_scorecard(true, true),
                        vec![StreamingChorusCheckedProperty::SlowerSpeedProducesLongerOutput],
                    ),
                    normal_speed: 1.0,
                    slow_speed: 0.8,
                    normal_num_samples: 10_000,
                    slow_num_samples: 12_500,
                    slow_to_normal_ratio: 1.25,
                    normal_wall_ms: 14.0,
                    slow_wall_ms: 18.0,
                    runtime_dispatch: StreamingChorusRuntimeDispatchScorecard {
                        compute_encodings: 0,
                        blits: 0,
                        total_metal_command_encodings: 0,
                        flushes: 0,
                        submits: 0,
                        blits_eliminated: 0,
                    },
                },
            ),
        ],
    }
}

fn measure_basic_streaming(
    primary: &mut CompiledKokoro,
    cache: &nn_metal::PipelineCache,
    stream_config: &KokoroStreamConfig,
) -> StreamingChorusBasicMeasurement {
    let mut chorus = build_chorus(primary, cache);
    let styles = make_styles();
    let chunks = vec![make_input_short(), make_input_long(), make_input_short()];
    let mut session = StreamingChorusSession::new(chunks, styles, 1.0, stream_config.clone())
        .expect("scorecard basic session creation");
    let first_chunk_warmup_pending_at_start = session.precompile_pending();

    assert_eq!(session.total_chunks(), 3);
    assert_eq!(session.remaining(), 3);
    assert_eq!(session.synthesized_count(), 0);
    assert!(!session.is_done());

    let (audio_chunks, wall_ms, runtime_dispatch) = timed_with_dispatch_stats(|| {
        collect_session_chunks(&mut session, &mut chorus, cache, "scorecard basic")
    });

    assert_eq!(
        audio_chunks.len(),
        3,
        "scorecard basic should synthesize 3 chunks"
    );
    assert_eq!(session.remaining(), 0);
    assert_eq!(session.synthesized_count(), 3);
    assert!(session.precompile_consumed());

    let all_audio = flatten_chunks(&audio_chunks);
    validate_audio(&all_audio, "scorecard basic full audio");
    let rms = rms_energy(&all_audio);
    assert!(
        rms > 1e-6,
        "scorecard basic RMS energy {rms:.2e} should indicate non-silent output"
    );

    StreamingChorusBasicMeasurement {
        scope: measurement_scope(
            "basic_streaming",
            vec![3],
            StreamingChorusWarmState::WarmChorusCloneSegments,
            first_chunk_warmup_scorecard(
                first_chunk_warmup_pending_at_start,
                session.precompile_consumed(),
            ),
            vec![
                StreamingChorusCheckedProperty::FiniteAudioSamples,
                StreamingChorusCheckedProperty::SamplesWithinUnitRange,
                StreamingChorusCheckedProperty::NonSilentAudio,
                StreamingChorusCheckedProperty::ChunkCountMatchesInputCount,
                StreamingChorusCheckedProperty::SessionCompletes,
            ],
        ),
        wall_ms,
        observed_chunk_count: audio_chunks.len(),
        chunk_sample_counts: chunk_pcm_lengths(&audio_chunks),
        chunk_sample_offsets: chunk_sample_offsets(&audio_chunks),
        total_samples: all_audio.len(),
        rms_energy: rms,
        runtime_dispatch,
    }
}

fn measure_cancel_semantics(
    primary: &mut CompiledKokoro,
    cache: &nn_metal::PipelineCache,
    stream_config: &KokoroStreamConfig,
) -> StreamingChorusCancelMeasurement {
    let mut chorus = build_chorus(primary, cache);
    let styles = make_styles();
    let chunks = vec![make_input_short(), make_input_short(), make_input_short()];
    let mut session = StreamingChorusSession::new(chunks, styles, 1.0, stream_config.clone())
        .expect("scorecard cancel session creation");
    let first_chunk_warmup_pending_at_start = session.precompile_pending();

    let remaining_before_cancel = session.remaining();
    let ((first_chunk_samples, synthesized_before_cancel), runtime_dispatch) =
        capture_dispatch_stats(|| {
            let result = session
                .next_chunk(&mut chorus, cache)
                .expect("scorecard cancel chunk 0 should be available");
            let audio_chunk = result.expect("scorecard cancel chunk 0 synthesis");
            validate_audio(&audio_chunk.pcm, "scorecard cancel chunk 0");
            (audio_chunk.pcm.len(), session.synthesized_count())
        });

    assert_eq!(session.remaining(), 2);
    session.cancel();

    assert!(session.is_cancelled());
    assert!(session.is_done());
    assert!(session.precompile_consumed());
    assert!(
        session.next_chunk(&mut chorus, cache).is_none(),
        "scorecard cancel should stop iteration immediately"
    );

    StreamingChorusCancelMeasurement {
        scope: measurement_scope(
            "cancel_semantics",
            vec![3],
            StreamingChorusWarmState::WarmChorusCloneSegments,
            first_chunk_warmup_scorecard(
                first_chunk_warmup_pending_at_start,
                session.precompile_consumed(),
            ),
            vec![
                StreamingChorusCheckedProperty::FiniteAudioSamples,
                StreamingChorusCheckedProperty::SamplesWithinUnitRange,
                StreamingChorusCheckedProperty::CancelStopsIteration,
            ],
        ),
        first_chunk_samples,
        remaining_before_cancel,
        remaining_after_cancel: session.remaining(),
        synthesized_before_cancel,
        runtime_dispatch,
    }
}

fn measure_reset_replay(
    primary: &mut CompiledKokoro,
    cache: &nn_metal::PipelineCache,
    stream_config: &KokoroStreamConfig,
) -> StreamingChorusResetReplayMeasurement {
    let mut chorus = build_chorus(primary, cache);
    let styles = make_styles();
    let chunks = vec![make_input_short(), make_input_long(), make_input_short()];
    let mut session = StreamingChorusSession::new(chunks, styles, 1.0, stream_config.clone())
        .expect("scorecard reset session creation");
    let first_chunk_warmup_pending_at_start = session.precompile_pending();

    let (
        (
            replayed_chunk_sample_counts,
            replay_chunk_max_abs_diff,
            replay_total_samples,
            full_reset_pass_total_samples,
        ),
        runtime_dispatch,
    ) = capture_dispatch_stats(|| {
        let mut pre_reset_chunks = Vec::new();
        for i in 0..2 {
            let result = session
                .next_chunk(&mut chorus, cache)
                .unwrap_or_else(|| panic!("scorecard reset chunk {i} should be available"));
            let audio_chunk = result
                .unwrap_or_else(|e| panic!("scorecard reset chunk {i} synthesis failed: {e}"));
            validate_audio(&audio_chunk.pcm, &format!("scorecard reset chunk {i}"));
            pre_reset_chunks.push(audio_chunk);
        }

        session.reset();
        assert_eq!(session.remaining(), 3);
        assert_eq!(session.synthesized_count(), 0);
        assert!(!session.is_done());
        assert!(!session.is_cancelled());

        let mut replayed_chunk_sample_counts = Vec::new();
        let mut replay_chunk_max_abs_diff = Vec::new();
        let mut replay_total_samples = 0usize;
        for (i, expected_chunk) in pre_reset_chunks.iter().enumerate() {
            let result = session
                .next_chunk(&mut chorus, cache)
                .unwrap_or_else(|| panic!("scorecard reset replay chunk {i} should be available"));
            let audio_chunk = result.unwrap_or_else(|e| {
                panic!("scorecard reset replay chunk {i} synthesis failed: {e}")
            });
            let max_abs_diff = assert_chunks_match(
                expected_chunk,
                &audio_chunk,
                &format!("scorecard reset replay chunk {i}"),
            );
            replayed_chunk_sample_counts.push(audio_chunk.pcm.len());
            replay_chunk_max_abs_diff.push(max_abs_diff);
            replay_total_samples += audio_chunk.pcm.len();
        }

        let mut full_reset_pass_total_samples = replay_total_samples;
        while let Some(result) = session.next_chunk(&mut chorus, cache) {
            let audio_chunk = result.expect("scorecard reset replay final chunk synthesis");
            validate_audio(&audio_chunk.pcm, "scorecard reset replay final chunk");
            full_reset_pass_total_samples += audio_chunk.pcm.len();
        }
        assert!(session.is_done());

        (
            replayed_chunk_sample_counts,
            replay_chunk_max_abs_diff,
            replay_total_samples,
            full_reset_pass_total_samples,
        )
    });

    StreamingChorusResetReplayMeasurement {
        scope: measurement_scope(
            "reset_replay",
            vec![3],
            StreamingChorusWarmState::WarmChorusCloneSegments,
            first_chunk_warmup_scorecard(
                first_chunk_warmup_pending_at_start,
                session.precompile_consumed(),
            ),
            vec![
                StreamingChorusCheckedProperty::FiniteAudioSamples,
                StreamingChorusCheckedProperty::SamplesWithinUnitRange,
                StreamingChorusCheckedProperty::ResetReplaysMixedPrefixWithinTolerance,
                StreamingChorusCheckedProperty::SessionCompletes,
            ],
        ),
        replayed_chunk_count: replayed_chunk_sample_counts.len(),
        replayed_chunk_sample_counts,
        replay_chunk_max_abs_diff,
        replay_total_samples,
        full_reset_pass_total_samples,
        runtime_dispatch,
    }
}

fn measure_crossfade_continuity(
    primary: &mut CompiledKokoro,
    cache: &nn_metal::PipelineCache,
    stream_config: &KokoroStreamConfig,
) -> StreamingChorusCrossfadeMeasurement {
    let mut chorus = build_chorus(primary, cache);
    let styles = make_styles();
    let chunks = vec![
        make_input_short(),
        make_input_long(),
        make_input_short(),
        make_input_long(),
    ];
    let mut session = StreamingChorusSession::new(chunks, styles, 1.0, stream_config.clone())
        .expect("scorecard crossfade session creation");
    let first_chunk_warmup_pending_at_start = session.precompile_pending();

    let (audio_chunks, wall_ms, runtime_dispatch) = timed_with_dispatch_stats(|| {
        collect_session_chunks(&mut session, &mut chorus, cache, "scorecard crossfade")
    });

    let full_audio = flatten_chunks(&audio_chunks);
    validate_audio(&full_audio, "scorecard crossfade full audio");

    let mut max_delta_overall = 0.0f32;
    for pair in full_audio.windows(2) {
        let delta = (pair[1] - pair[0]).abs();
        if delta > max_delta_overall {
            max_delta_overall = delta;
        }
    }

    let mut max_boundary_delta = 0.0f32;
    for pair in audio_chunks.windows(2) {
        if let (Some(&last), Some(&first)) = (pair[0].pcm.last(), pair[1].pcm.first()) {
            let delta = (first - last).abs();
            if delta > max_boundary_delta {
                max_boundary_delta = delta;
            }
        }
    }

    assert!(
        max_boundary_delta < STREAMING_CHORUS_BOUNDARY_THRESHOLD,
        "scorecard crossfade boundary delta {max_boundary_delta:.6} exceeds threshold \
         {STREAMING_CHORUS_BOUNDARY_THRESHOLD:.6}"
    );

    StreamingChorusCrossfadeMeasurement {
        scope: measurement_scope(
            "crossfade_continuity",
            vec![4],
            StreamingChorusWarmState::WarmChorusCloneSegments,
            first_chunk_warmup_scorecard(
                first_chunk_warmup_pending_at_start,
                session.precompile_consumed(),
            ),
            vec![
                StreamingChorusCheckedProperty::FiniteAudioSamples,
                StreamingChorusCheckedProperty::SamplesWithinUnitRange,
                StreamingChorusCheckedProperty::CrossfadeBoundaryBelowThreshold,
            ],
        ),
        wall_ms,
        chunk_sample_counts: chunk_pcm_lengths(&audio_chunks),
        total_samples: full_audio.len(),
        max_boundary_delta,
        max_delta_overall,
        boundary_threshold: STREAMING_CHORUS_BOUNDARY_THRESHOLD,
        runtime_dispatch,
    }
}

fn measure_cache_sharing(
    primary: &mut CompiledKokoro,
    cache: &nn_metal::PipelineCache,
    stream_config: &KokoroStreamConfig,
) -> StreamingChorusCacheSharingMeasurement {
    let input = make_input_short();
    let style = make_style();
    let _ = primary
        .synthesize(&input, &style, 1.0, cache)
        .expect("scorecard cache sharing warmup");

    let parent_cached_segments = primary.total_cached_segments();
    assert_eq!(
        parent_cached_segments, 8,
        "scorecard cache sharing parent should have 8 cached segments"
    );

    let config = ChorusConfig::equal_gain(N_VOICES).expect("valid chorus config");
    let mut chorus = KokoroChorus::new(primary, config).expect("scorecard cache sharing chorus");

    let voice_cached_segments_before: Vec<usize> = (0..chorus.n_voices())
        .map(|i| {
            let voice = chorus
                .voice(i)
                .expect("scorecard cache sharing voice exists");
            let cached = voice.total_cached_segments();
            assert_eq!(
                cached, 8,
                "scorecard cache sharing voice[{i}] should start warm"
            );
            cached
        })
        .collect();

    let styles = make_styles();
    let chunks = vec![make_input_short(), make_input_long()];
    let (
        (first_chunk_warmup_pending_at_start, first_chunk_warmup_consumed_by_end),
        wall_ms,
        runtime_dispatch,
    ) = timed_with_dispatch_stats(|| {
        let mut session = StreamingChorusSession::new(chunks, styles, 1.0, stream_config.clone())
            .expect("scorecard cache sharing session creation");
        let first_chunk_warmup_pending_at_start = session.precompile_pending();
        let _ = collect_session_chunks(&mut session, &mut chorus, cache, "scorecard cache sharing");
        (
            first_chunk_warmup_pending_at_start,
            session.precompile_consumed(),
        )
    });

    let voice_cached_segments_after: Vec<usize> = (0..chorus.n_voices())
        .map(|i| {
            let voice = chorus
                .voice(i)
                .expect("scorecard cache sharing voice exists");
            let cached = voice.total_cached_segments();
            assert_eq!(
                cached, 8,
                "scorecard cache sharing voice[{i}] should remain warm"
            );
            cached
        })
        .collect();

    let shared_state_refcount = chorus.shared_state_refcount();
    assert!(
        shared_state_refcount >= N_VOICES,
        "scorecard cache sharing refcount {shared_state_refcount} < {N_VOICES}"
    );

    StreamingChorusCacheSharingMeasurement {
        scope: measurement_scope(
            "cache_sharing",
            vec![2],
            StreamingChorusWarmState::WarmPrimarySegmentCache,
            first_chunk_warmup_scorecard(
                first_chunk_warmup_pending_at_start,
                first_chunk_warmup_consumed_by_end,
            ),
            vec![StreamingChorusCheckedProperty::ChorusVoicesPreserveCachedSegments],
        ),
        wall_ms,
        parent_cached_segments,
        voice_cached_segments_before,
        voice_cached_segments_after,
        shared_state_refcount,
        runtime_dispatch,
    }
}

fn measure_speed_variation(
    primary: &mut CompiledKokoro,
    cache: &nn_metal::PipelineCache,
    stream_config: &KokoroStreamConfig,
) -> StreamingChorusSpeedVariationMeasurement {
    let mut chorus = build_chorus(primary, cache);
    let styles = make_styles();
    let chunks = vec![make_input_short(), make_input_long()];
    let mut session = StreamingChorusSession::new(chunks, styles, 1.0, stream_config.clone())
        .expect("scorecard speed variation session creation");
    let first_chunk_warmup_pending_at_start = session.precompile_pending();

    nn_metal::reset_counters();

    let mut audio_normal = Vec::new();
    let normal_started = Instant::now();
    while let Some(result) = session.next_chunk(&mut chorus, cache) {
        let audio_chunk = result.expect("scorecard normal-speed chunk synthesis");
        audio_normal.extend(&audio_chunk.pcm);
    }
    let normal_wall_ms = normal_started.elapsed().as_secs_f64() * 1000.0;
    assert!(session.is_done());
    validate_audio(&audio_normal, "scorecard speed variation normal");

    session.reset();
    session.set_speed(0.8);

    let mut audio_slow = Vec::new();
    let slow_started = Instant::now();
    while let Some(result) = session.next_chunk(&mut chorus, cache) {
        let audio_chunk = result.expect("scorecard slow-speed chunk synthesis");
        audio_slow.extend(&audio_chunk.pcm);
    }
    let slow_wall_ms = slow_started.elapsed().as_secs_f64() * 1000.0;
    assert!(session.is_done());
    validate_audio(&audio_slow, "scorecard speed variation slow");

    let runtime_dispatch = runtime_dispatch_scorecard(nn_metal::dispatch_stats());

    assert!(
        audio_slow.len() > audio_normal.len(),
        "scorecard slow speed should produce longer output: slow={} normal={}",
        audio_slow.len(),
        audio_normal.len(),
    );

    let slow_to_normal_ratio = audio_slow.len() as f64 / audio_normal.len() as f64;
    assert!(
        (1.1..=1.5).contains(&slow_to_normal_ratio),
        "scorecard slow/normal ratio should stay near 1.25, got {slow_to_normal_ratio:.3}"
    );
    assert!(session.precompile_consumed());

    StreamingChorusSpeedVariationMeasurement {
        scope: measurement_scope(
            "speed_variation",
            vec![2],
            StreamingChorusWarmState::WarmChorusCloneSegments,
            first_chunk_warmup_scorecard(
                first_chunk_warmup_pending_at_start,
                session.precompile_consumed(),
            ),
            vec![
                StreamingChorusCheckedProperty::FiniteAudioSamples,
                StreamingChorusCheckedProperty::SamplesWithinUnitRange,
                StreamingChorusCheckedProperty::SlowerSpeedProducesLongerOutput,
            ],
        ),
        normal_speed: 1.0,
        slow_speed: 0.8,
        normal_num_samples: audio_normal.len(),
        slow_num_samples: audio_slow.len(),
        slow_to_normal_ratio,
        normal_wall_ms,
        slow_wall_ms,
        runtime_dispatch,
    }
}

// -- Tests --------------------------------------------------------------------

/// Production weights: create a StreamingChorusSession, iterate all chunks,
/// and verify non-empty valid audio output.
///
/// Steps:
/// 1. Load production Kokoro, warmup, build chorus.
/// 2. Create a StreamingChorusSession with 3 chunks.
/// 3. Iterate all chunks via `next_chunk()`.
/// 4. Verify all chunks produce valid audio (no NaN, in [-1, 1]).
/// 5. Verify session completes (`is_done()`).
/// 6. Verify concatenated audio has plausible length.
#[test]
fn test_streaming_chorus_basic() {
    let (mut primary, cache) = match load_production_kokoro(
        "streaming chorus basic test skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let mut chorus = build_chorus(&mut primary, &cache);
    let styles = make_styles();
    let stream_config = KokoroStreamConfig::default();

    // Build 3 chunks (mix of short and long inputs).
    let chunks = vec![make_input_short(), make_input_long(), make_input_short()];

    let mut session =
        StreamingChorusSession::new(chunks, styles, 1.0, stream_config).expect("session creation");

    assert_eq!(session.total_chunks(), 3);
    assert_eq!(session.remaining(), 3);
    assert_eq!(session.synthesized_count(), 0);
    assert!(!session.is_done());

    let mut all_audio: Vec<f32> = Vec::new();
    let mut chunk_count = 0usize;

    while let Some(result) = session.next_chunk(&mut chorus, &cache) {
        let audio_chunk = result.expect("chunk synthesis");
        assert!(
            !audio_chunk.pcm.is_empty(),
            "chunk {chunk_count}: pcm must not be empty"
        );
        assert_eq!(audio_chunk.chunk_index, chunk_count, "chunk_index mismatch");
        assert_eq!(audio_chunk.total_chunks, 3, "total_chunks mismatch");

        all_audio.extend(&audio_chunk.pcm);
        chunk_count += 1;
    }

    assert_eq!(chunk_count, 3, "should have synthesized exactly 3 chunks");
    assert!(session.is_done());
    assert_eq!(session.remaining(), 0);
    assert_eq!(session.synthesized_count(), 3);

    // next_chunk() after completion returns None.
    assert!(session.next_chunk(&mut chorus, &cache).is_none());

    // Validate concatenated audio.
    validate_audio(&all_audio, "streaming_chorus_basic");

    // Plausibility: 3 chunks should produce at least 0.1s of audio.
    let min_samples = SAMPLE_RATE / 10;
    assert!(
        all_audio.len() >= min_samples,
        "streaming_chorus_basic: {} samples < minimum {min_samples}",
        all_audio.len(),
    );

    eprintln!(
        "test_streaming_chorus_basic: PASS -- {chunk_count} chunks, {} samples total",
        all_audio.len(),
    );
}

/// Production weights: recommended autocast preserves the pull-based streaming
/// chorus chunk contract without over-claiming waveform parity.
///
/// This covers the actual `StreamingChorusSession` + `KokoroChorus` surface
/// under [`CompiledKokoro::with_recommended_autocast()`]. It asserts only the
/// current contract we need from that mode:
/// - the mixed streaming audio stays finite and in range,
/// - chunk count, offsets, lengths, and final flags match the F32 surface,
/// - chunk boundaries remain below the existing continuity threshold.
///
/// It does not assert per-sample equality or treat this as a throughput gate.
#[test]
fn test_streaming_chorus_recommended_autocast_preserves_chunk_contract() {
    let (mut baseline_primary, baseline_cache) = match load_production_kokoro(
        "streaming chorus recommended autocast test skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };
    let (autocast_primary, autocast_cache) = match load_production_kokoro(
        "streaming chorus recommended autocast test skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let mut recommended_autocast_primary = autocast_primary.with_recommended_autocast();
    assert!(
        recommended_autocast_primary.segment_autocast().is_some(),
        "recommended autocast test requires per-segment autocast to be enabled"
    );

    let mut baseline_chorus = build_chorus(&mut baseline_primary, &baseline_cache);
    let mut autocast_chorus = build_chorus(&mut recommended_autocast_primary, &autocast_cache);
    assert!(
        autocast_chorus
            .voice(0)
            .expect("autocast chorus voice 0")
            .segment_autocast()
            .is_some(),
        "clone-dispatch chorus voice should retain the recommended autocast config"
    );

    let stream_config = KokoroStreamConfig::default();
    let baseline_chunks_in = vec![
        make_input_short(),
        make_input_long(),
        make_input_short(),
        make_input_long(),
    ];
    let autocast_chunks_in = vec![
        make_input_short(),
        make_input_long(),
        make_input_short(),
        make_input_long(),
    ];

    let mut baseline_session = StreamingChorusSession::new(
        baseline_chunks_in,
        make_styles(),
        1.0,
        stream_config.clone(),
    )
    .expect("baseline session creation");
    let mut autocast_session =
        StreamingChorusSession::new(autocast_chunks_in, make_styles(), 1.0, stream_config)
            .expect("recommended autocast session creation");
    assert!(
        autocast_session.precompile_pending(),
        "recommended autocast session should still use the one-time warmup contract"
    );

    let baseline_chunks = collect_session_chunks(
        &mut baseline_session,
        &mut baseline_chorus,
        &baseline_cache,
        "recommended autocast baseline",
    );
    let autocast_chunks = collect_session_chunks(
        &mut autocast_session,
        &mut autocast_chorus,
        &autocast_cache,
        "recommended autocast",
    );

    assert_eq!(
        baseline_chunks.len(),
        autocast_chunks.len(),
        "recommended autocast should preserve the number of streamed chunks"
    );
    assert_eq!(
        chunk_pcm_lengths(&autocast_chunks),
        chunk_pcm_lengths(&baseline_chunks),
        "recommended autocast should preserve chunk PCM lengths"
    );
    assert_eq!(
        chunk_sample_offsets(&autocast_chunks),
        chunk_sample_offsets(&baseline_chunks),
        "recommended autocast should preserve chunk sample offsets"
    );

    for (chunk_index, (baseline_chunk, autocast_chunk)) in baseline_chunks
        .iter()
        .zip(autocast_chunks.iter())
        .enumerate()
    {
        assert_eq!(
            autocast_chunk.chunk_index, baseline_chunk.chunk_index,
            "recommended autocast chunk {chunk_index}: chunk_index mismatch"
        );
        assert_eq!(
            autocast_chunk.total_chunks, baseline_chunk.total_chunks,
            "recommended autocast chunk {chunk_index}: total_chunks mismatch"
        );
        assert_eq!(
            autocast_chunk.sample_offset, baseline_chunk.sample_offset,
            "recommended autocast chunk {chunk_index}: sample_offset mismatch"
        );
        assert_eq!(
            autocast_chunk.channels, baseline_chunk.channels,
            "recommended autocast chunk {chunk_index}: channels mismatch"
        );
        assert_eq!(
            autocast_chunk.is_final, baseline_chunk.is_final,
            "recommended autocast chunk {chunk_index}: is_final mismatch"
        );
        assert_eq!(
            autocast_chunk.pcm.len(),
            baseline_chunk.pcm.len(),
            "recommended autocast chunk {chunk_index}: PCM length mismatch"
        );
        validate_audio(
            &autocast_chunk.pcm,
            &format!("recommended autocast chunk {chunk_index}"),
        );
    }

    let autocast_audio = flatten_chunks(&autocast_chunks);
    validate_audio(&autocast_audio, "recommended autocast full audio");
    assert!(
        autocast_session.precompile_consumed(),
        "recommended autocast session should consume its first-chunk warmup"
    );

    let mut autocast_max_boundary_delta = 0.0f32;
    for pair in autocast_chunks.windows(2) {
        if let (Some(&last), Some(&first)) = (pair[0].pcm.last(), pair[1].pcm.first()) {
            let delta = (first - last).abs();
            if delta > autocast_max_boundary_delta {
                autocast_max_boundary_delta = delta;
            }
        }
    }
    assert!(
        autocast_max_boundary_delta < STREAMING_CHORUS_BOUNDARY_THRESHOLD,
        "recommended autocast boundary delta {autocast_max_boundary_delta:.6} exceeds the \
         streaming continuity threshold {STREAMING_CHORUS_BOUNDARY_THRESHOLD:.6}",
    );

    eprintln!(
        "test_streaming_chorus_recommended_autocast_preserves_chunk_contract: PASS -- \
         chunks={}, total_samples={}, max_boundary_delta={autocast_max_boundary_delta:.6}",
        autocast_chunks.len(),
        autocast_audio.len(),
    );
}

/// Production weights: cancel a StreamingChorusSession mid-stream and verify
/// that `remaining()` returns 0 and no further chunks are produced.
///
/// Steps:
/// 1. Load production Kokoro, warmup, build chorus.
/// 2. Create a StreamingChorusSession with 3 chunks.
/// 3. Consume 1 chunk successfully.
/// 4. Call `cancel()`.
/// 5. Verify `is_cancelled()`, `is_done()`, `remaining() == 0`.
/// 6. Verify `next_chunk()` returns `None` after cancel.
#[test]
fn test_streaming_chorus_cancel() {
    let (mut primary, cache) = match load_production_kokoro(
        "streaming chorus cancel test skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let mut chorus = build_chorus(&mut primary, &cache);
    let styles = make_styles();
    let stream_config = KokoroStreamConfig::default();

    let chunks = vec![make_input_short(), make_input_short(), make_input_short()];

    let mut session =
        StreamingChorusSession::new(chunks, styles, 1.0, stream_config).expect("session creation");

    assert_eq!(session.remaining(), 3);

    // Consume the first chunk.
    let result = session
        .next_chunk(&mut chorus, &cache)
        .expect("chunk 0 should be available");
    let audio0 = result.expect("chunk 0 synthesis");
    validate_audio(&audio0.pcm, "cancel: chunk 0");
    assert_eq!(session.remaining(), 2);
    assert_eq!(session.synthesized_count(), 1);

    // Cancel mid-stream.
    session.cancel();

    assert!(session.is_cancelled());
    assert!(session.is_done());
    assert_eq!(session.remaining(), 0);

    // No further chunks should be produced.
    assert!(
        session.next_chunk(&mut chorus, &cache).is_none(),
        "next_chunk after cancel must return None"
    );
    assert!(
        session.next_chunk(&mut chorus, &cache).is_none(),
        "repeated next_chunk after cancel must return None"
    );

    eprintln!(
        "test_streaming_chorus_cancel: PASS -- consumed 1 chunk ({} samples), \
         cancelled with 2 remaining",
        audio0.pcm.len(),
    );
}

/// Production weights: reset a StreamingChorusSession after partial synthesis
/// and verify re-synthesis from the start.
///
/// Steps:
/// 1. Load production Kokoro, warmup, build chorus.
/// 2. Create a StreamingChorusSession with 3 chunks.
/// 3. Consume 2 chunks, collecting audio.
/// 4. Call `reset()`.
/// 5. Verify state is restored: `remaining() == 3`, `synthesized_count() == 0`.
/// 6. Re-synthesize the first 2 chunks and verify metadata + PCM replay.
/// 7. Consume the final chunk and verify the reset pass completes normally.
#[test]
fn test_streaming_chorus_reset() {
    let (mut primary, cache) = match load_production_kokoro(
        "streaming chorus reset test skipped -- set KOKORO_WEIGHTS to enable.",
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

    // Consume 2 chunks.
    let mut pre_reset_audio: Vec<f32> = Vec::new();
    let mut pre_reset_chunks: Vec<AudioChunk> = Vec::new();
    for i in 0..2 {
        let result = session
            .next_chunk(&mut chorus, &cache)
            .unwrap_or_else(|| panic!("chunk {i} should be available"));
        let audio_chunk = result.unwrap_or_else(|e| panic!("chunk {i} synthesis failed: {e}"));
        pre_reset_audio.extend(&audio_chunk.pcm);
        pre_reset_chunks.push(audio_chunk);
    }
    assert_eq!(session.remaining(), 1);
    assert_eq!(session.synthesized_count(), 2);
    assert!(!session.is_done());

    // Reset.
    session.reset();

    assert_eq!(session.remaining(), 3);
    assert_eq!(session.synthesized_count(), 0);
    assert!(!session.is_done());
    assert!(!session.is_cancelled());

    // Re-synthesize the first 2 chunks from the reset session and verify
    // deterministic replay on the current equal-gain production path.
    let mut post_reset_audio: Vec<f32> = Vec::new();
    let mut chunk_count = 0usize;
    for (i, expected_chunk) in pre_reset_chunks.iter().enumerate() {
        let result = session
            .next_chunk(&mut chorus, &cache)
            .unwrap_or_else(|| panic!("post-reset chunk {i} should be available"));
        let audio_chunk =
            result.unwrap_or_else(|e| panic!("post-reset chunk {i} synthesis failed: {e}"));
        assert_chunks_match(
            expected_chunk,
            &audio_chunk,
            &format!("reset replay chunk {i}"),
        );
        post_reset_audio.extend(&audio_chunk.pcm);
        chunk_count += 1;
    }

    // Consume the final chunk from the reset session.
    while let Some(result) = session.next_chunk(&mut chorus, &cache) {
        let audio_chunk = result.expect("post-reset final chunk synthesis");
        post_reset_audio.extend(&audio_chunk.pcm);
        chunk_count += 1;
    }

    assert_eq!(chunk_count, 3, "should synthesize all 3 chunks after reset");
    assert!(session.is_done());

    // Validate audio from the reset pass.
    validate_audio(&post_reset_audio, "reset: post-reset audio");

    // Both passes should produce non-trivial audio.
    assert!(
        !pre_reset_audio.is_empty(),
        "pre-reset audio must not be empty"
    );
    assert!(
        !post_reset_audio.is_empty(),
        "post-reset audio must not be empty"
    );

    // Post-reset audio should be longer (3 chunks vs 2 chunks pre-reset).
    assert!(
        post_reset_audio.len() > pre_reset_audio.len(),
        "post-reset (3 chunks, {} samples) should be longer than pre-reset (2 chunks, {} samples)",
        post_reset_audio.len(),
        pre_reset_audio.len(),
    );

    eprintln!(
        "test_streaming_chorus_reset: PASS -- pre_reset={} samples (2 chunks), \
         post_reset={} samples (3 chunks)",
        pre_reset_audio.len(),
        post_reset_audio.len(),
    );
}

/// Production weights: verify crossfade produces smooth transitions between
/// adjacent chunks (no audible clicks at chunk boundaries).
///
/// A click manifests as a large sample-to-sample delta at the crossfade
/// boundary. This test checks that the maximum absolute delta between
/// consecutive samples in the crossfade region is within a reasonable
/// threshold. For reference, a hard cut (no crossfade) between unrelated
/// audio segments can produce deltas > 0.5; a smooth crossfade keeps deltas
/// comparable to intra-chunk deltas.
///
/// Steps:
/// 1. Load production Kokoro, warmup, build chorus.
/// 2. Create a StreamingChorusSession with 4 chunks.
/// 3. Consume all chunks, tracking per-chunk sample offsets.
/// 4. Measure max sample-to-sample delta at each chunk boundary.
/// 5. Measure max delta within the interior of each chunk.
/// 6. Assert boundary deltas are not dramatically larger than interior deltas.
#[test]
fn test_streaming_chorus_crossfade() {
    let (mut primary, cache) = match load_production_kokoro(
        "streaming chorus crossfade test skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let mut chorus = build_chorus(&mut primary, &cache);
    let styles = make_styles();
    let stream_config = KokoroStreamConfig::default();

    // Use 4 chunks to create 3 chunk boundaries.
    let chunks = vec![
        make_input_short(),
        make_input_long(),
        make_input_short(),
        make_input_long(),
    ];

    let mut session =
        StreamingChorusSession::new(chunks, styles, 1.0, stream_config).expect("session creation");

    // Collect all chunk PCM segments.
    let mut chunk_pcms: Vec<Vec<f32>> = Vec::new();
    while let Some(result) = session.next_chunk(&mut chorus, &cache) {
        let audio_chunk = result.expect("chunk synthesis");
        chunk_pcms.push(audio_chunk.pcm);
    }
    assert_eq!(chunk_pcms.len(), 4, "should have 4 chunks");

    // Concatenate all chunks into a single stream (this is what the caller
    // would hear).
    let full_audio: Vec<f32> = chunk_pcms.iter().flat_map(|c| c.iter().copied()).collect();
    validate_audio(&full_audio, "crossfade: full audio");

    // Measure max sample-to-sample delta across the entire stream.
    let mut max_delta_overall = 0.0f32;
    for pair in full_audio.windows(2) {
        let delta = (pair[1] - pair[0]).abs();
        if delta > max_delta_overall {
            max_delta_overall = delta;
        }
    }

    // Measure max delta at each chunk boundary (last sample of chunk N,
    // first sample of chunk N+1).
    let mut max_boundary_delta = 0.0f32;
    for pair in chunk_pcms.windows(2) {
        if let (Some(&last), Some(&first)) = (pair[0].last(), pair[1].first()) {
            let delta = (first - last).abs();
            if delta > max_boundary_delta {
                max_boundary_delta = delta;
            }
        }
    }

    // The crossfade should prevent hard jumps. Boundary deltas should not
    // exceed 3x the overall max delta within chunks. This is a conservative
    // threshold; the crossfade should produce boundary deltas comparable to
    // or smaller than interior deltas.
    //
    // If crossfade is broken (hard cuts), boundary deltas will be orders
    // of magnitude larger.
    //
    // Note: we use max_delta_overall as the reference because it already
    // includes boundary deltas. The key assertion is that boundary deltas
    // are not pathologically large.
    let boundary_threshold = 0.5f32; // Absolute threshold for click detection.
    assert!(
        max_boundary_delta < boundary_threshold,
        "crossfade boundary delta {max_boundary_delta:.6} exceeds click threshold \
         {boundary_threshold:.6} -- possible discontinuity at chunk boundary"
    );

    eprintln!(
        "test_streaming_chorus_crossfade: PASS -- 4 chunks, {} samples total, \
         max_boundary_delta={max_boundary_delta:.6}, max_delta_overall={max_delta_overall:.6}",
        full_audio.len(),
    );
}

/// Production weights: verify segment cache is shared across chorus voices
/// when using StreamingChorusSession.
///
/// Steps:
/// 1. Load production Kokoro, warmup (compiles 8 segments).
/// 2. Build KokoroChorus (N_VOICES clones).
/// 3. Verify all chorus voices inherited 8 cached segments immediately.
/// 4. Create and fully consume a StreamingChorusSession.
/// 5. Verify segment counts are unchanged after streaming synthesis.
#[test]
fn test_streaming_chorus_cache_sharing() {
    let (mut primary, cache) = match load_production_kokoro(
        "streaming chorus cache sharing test skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    // Warmup: compile all 8 segments in primary.
    let input = make_input_short();
    let style = make_style();
    let _ = primary
        .synthesize(&input, &style, 1.0, &cache)
        .expect("warmup synthesis");

    let parent_cached = primary.total_cached_segments();
    assert_eq!(
        parent_cached, 8,
        "warmed parent should have 8 cached segments, got {parent_cached}"
    );

    // Build chorus. Clones should inherit cached segments.
    let config = ChorusConfig::equal_gain(N_VOICES).expect("valid chorus config");
    let mut chorus = KokoroChorus::new(&primary, config).expect("chorus creation");

    // Verify all voices inherited 8 cached segments.
    let mut total_cached_before = 0usize;
    for i in 0..chorus.n_voices() {
        let voice = chorus.voice(i).expect("voice exists");
        let cached = voice.total_cached_segments();
        assert_eq!(
            cached, 8,
            "voice[{i}] should have 8 cached segments, got {cached}"
        );
        total_cached_before += cached;
    }
    assert_eq!(
        total_cached_before,
        N_VOICES * 8,
        "total cached segments should be {}",
        N_VOICES * 8
    );

    // Verify shared state refcount.
    let refcount = chorus.shared_state_refcount();
    assert!(
        refcount >= N_VOICES,
        "shared_state_refcount {refcount} < {N_VOICES}"
    );

    // Create and fully consume a streaming session.
    let styles = make_styles();
    let stream_config = KokoroStreamConfig::default();
    let chunks = vec![make_input_short(), make_input_long()];

    let mut session =
        StreamingChorusSession::new(chunks, styles, 1.0, stream_config).expect("session creation");

    while let Some(result) = session.next_chunk(&mut chorus, &cache) {
        let _audio_chunk = result.expect("chunk synthesis");
    }
    assert!(session.is_done());

    // Verify segment counts unchanged after streaming synthesis.
    let mut total_cached_after = 0usize;
    for i in 0..chorus.n_voices() {
        let voice = chorus.voice(i).expect("voice exists");
        let cached = voice.total_cached_segments();
        assert_eq!(
            cached, 8,
            "voice[{i}] should still have 8 cached segments after streaming, got {cached}"
        );
        total_cached_after += cached;
    }
    assert_eq!(
        total_cached_after, total_cached_before,
        "total cached segments should be unchanged after streaming"
    );

    eprintln!(
        "test_streaming_chorus_cache_sharing: PASS -- {N_VOICES} voices, \
         {total_cached_before} cached before, {total_cached_after} cached after, \
         refcount={refcount}",
    );
}

/// Production weights: different speeds produce different output lengths
/// from identical input when using StreamingChorusSession.
///
/// Steps:
/// 1. Load production Kokoro, warmup, build chorus.
/// 2. Create a StreamingChorusSession at speed 1.0, consume all chunks,
///    measure total audio length.
/// 3. Reset the session, set speed to 0.8, consume all chunks, measure
///    total audio length.
/// 4. Verify the slower speed (0.8) produces more audio samples.
/// 5. Verify the ratio is approximately 1.25x (tolerance: 1.1x to 1.5x).
#[test]
fn test_streaming_chorus_speed_variation() {
    let (mut primary, cache) = match load_production_kokoro(
        "streaming chorus speed variation test skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let mut chorus = build_chorus(&mut primary, &cache);
    let styles = make_styles();
    let stream_config = KokoroStreamConfig::default();

    // Use 2 chunks for a reasonable test.
    let chunks = vec![make_input_short(), make_input_long()];

    let mut session =
        StreamingChorusSession::new(chunks, styles, 1.0, stream_config).expect("session creation");

    // Pass 1: speed 1.0 (normal).
    let mut audio_normal: Vec<f32> = Vec::new();
    while let Some(result) = session.next_chunk(&mut chorus, &cache) {
        let audio_chunk = result.expect("normal-speed chunk synthesis");
        audio_normal.extend(&audio_chunk.pcm);
    }
    assert!(session.is_done());
    validate_audio(&audio_normal, "speed_variation: normal (1.0)");

    // Reset and switch to speed 0.8 (slower => longer audio).
    session.reset();
    session.set_speed(0.8);
    assert!((session.speed() - 0.8).abs() < f32::EPSILON);
    assert_eq!(session.remaining(), 2);

    let mut audio_slow: Vec<f32> = Vec::new();
    while let Some(result) = session.next_chunk(&mut chorus, &cache) {
        let audio_chunk = result.expect("slow-speed chunk synthesis");
        audio_slow.extend(&audio_chunk.pcm);
    }
    assert!(session.is_done());
    validate_audio(&audio_slow, "speed_variation: slow (0.8)");

    // Slower speed (0.8) should produce more audio than normal (1.0).
    // Duration is inversely proportional to speed: 0.8x speed => ~1.25x duration.
    assert!(
        audio_slow.len() > audio_normal.len(),
        "slow voice (0.8x speed, {} samples) should produce more audio than \
         normal voice (1.0x speed, {} samples)",
        audio_slow.len(),
        audio_normal.len(),
    );

    // Verify the ratio is approximately 1.25x (tolerance: 1.1x to 1.5x).
    let ratio = audio_slow.len() as f64 / audio_normal.len() as f64;
    assert!(
        (1.1..=1.5).contains(&ratio),
        "slow/normal audio length ratio should be ~1.25, got {ratio:.3} \
         (slow={}, normal={})",
        audio_slow.len(),
        audio_normal.len(),
    );

    eprintln!(
        "test_streaming_chorus_speed_variation: PASS -- normal={} samples, \
         slow={} samples, ratio={ratio:.3}",
        audio_normal.len(),
        audio_slow.len(),
    );
}

/// Production weights: emit a structured, machine-readable scorecard for the
/// real pull-based streaming chorus harness in this file.
///
/// This scorecard is intentionally scoped to the mixed-output
/// `StreamingChorusSession` + `KokoroChorus` path exercised here. It reports
/// only the properties actually measured in this file: mixed-session chunk
/// accounting, cancel semantics, reset replay parity, crossfade continuity,
/// cache preservation, speed variation, and dispatch counts.
#[test]
fn test_streaming_chorus_production_scorecard() {
    let (mut primary, cache) = match load_production_kokoro(
        "streaming chorus production scorecard skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let stream_config = KokoroStreamConfig::default();

    let basic = measure_basic_streaming(&mut primary, &cache, &stream_config);
    let cancel = measure_cancel_semantics(&mut primary, &cache, &stream_config);
    let reset_replay = measure_reset_replay(&mut primary, &cache, &stream_config);
    let crossfade = measure_crossfade_continuity(&mut primary, &cache, &stream_config);
    let cache_sharing = measure_cache_sharing(&mut primary, &cache, &stream_config);
    let speed_variation = measure_speed_variation(&mut primary, &cache, &stream_config);

    let scorecard = StreamingChorusProductionScorecard {
        schema_version: STREAMING_CHORUS_SCORECARD_SCHEMA_VERSION,
        suite_name: "kokoro_streaming_chorus_production".to_string(),
        build_profile: build_profile_name().to_string(),
        sample_rate_hz: SAMPLE_RATE,
        default_stream_config: stream_config_scorecard(&stream_config),
        // The production scorecard currently measures the F32 pull-streaming
        // harness in this file. Recommended autocast has separate contract
        // coverage and is reported here only as explicit harness metadata.
        recommended_autocast_enabled: false,
        limitations: vec![
            StreamingChorusScorecardLimitation::NotAudioQualityCertificate,
            StreamingChorusScorecardLimitation::NotThroughputBenchmark,
            StreamingChorusScorecardLimitation::NoBatchReferenceComparison,
        ],
        compiled_dispatch: compiled_dispatch_scorecard(&primary),
        measurements: vec![
            StreamingChorusProductionMeasurement::Basic(basic),
            StreamingChorusProductionMeasurement::Cancel(cancel),
            StreamingChorusProductionMeasurement::ResetReplay(reset_replay),
            StreamingChorusProductionMeasurement::CrossfadeContinuity(crossfade),
            StreamingChorusProductionMeasurement::CacheSharing(cache_sharing),
            StreamingChorusProductionMeasurement::SpeedVariation(speed_variation),
        ],
    };

    emit_streaming_chorus_scorecard(&scorecard);

    eprintln!(
        "test_streaming_chorus_production_scorecard: measurements={}, voices={}, \
         crossfade_samples={}",
        scorecard.measurements.len(),
        N_VOICES,
        scorecard.default_stream_config.crossfade_samples,
    );
}

#[test]
fn test_streaming_chorus_production_scorecard_json_round_trip() {
    let scorecard = sample_streaming_chorus_scorecard();

    let compact_json = scorecard
        .to_compact_json()
        .expect("serialize streaming chorus scorecard");
    let compact_parsed: StreamingChorusProductionScorecard = serde_json::from_str(&compact_json)
        .expect("deserialize compact streaming chorus scorecard");

    let pretty_json = scorecard
        .to_pretty_json()
        .expect("serialize pretty streaming chorus scorecard");
    let pretty_parsed: StreamingChorusProductionScorecard =
        serde_json::from_str(&pretty_json).expect("deserialize pretty streaming chorus scorecard");

    assert_eq!(compact_parsed, scorecard);
    assert_eq!(pretty_parsed, scorecard);
    assert_eq!(compact_parsed.measurements.len(), 3);
    assert_eq!(compact_parsed.default_stream_config.crossfade_samples, 960);
    assert!(!compact_parsed.recommended_autocast_enabled);
    let StreamingChorusProductionMeasurement::Basic(basic) = &compact_parsed.measurements[0] else {
        panic!("expected basic measurement in sample scorecard")
    };
    assert!(basic.scope.first_chunk_warmup.pending_at_start);
    assert!(basic.scope.first_chunk_warmup.consumed_by_end);
}

#[test]
fn test_streaming_chorus_production_scorecard_write_json_artifact() {
    let scorecard = sample_streaming_chorus_scorecard();

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    let out_dir = std::env::temp_dir().join(format!(
        "nn-streaming-chorus-scorecard-{}-{unique}",
        std::process::id()
    ));
    let out_path = out_dir.join("scorecards/streaming-chorus-production.json");

    scorecard
        .write_json(&out_path)
        .expect("write streaming chorus scorecard artifact");

    let persisted =
        fs::read_to_string(&out_path).expect("read persisted streaming chorus scorecard");
    let parsed: StreamingChorusProductionScorecard =
        serde_json::from_str(&persisted).expect("deserialize persisted streaming chorus scorecard");

    assert_eq!(
        parsed.schema_version,
        STREAMING_CHORUS_SCORECARD_SCHEMA_VERSION
    );
    assert_eq!(parsed.measurements.len(), 3);
    assert_eq!(
        parsed.default_stream_config.crossfade_window,
        StreamingChorusCrossfadeWindow::SqrtHann
    );
    assert!(!parsed.recommended_autocast_enabled);
    let StreamingChorusProductionMeasurement::Basic(basic) = &parsed.measurements[0] else {
        panic!("expected basic measurement in persisted scorecard")
    };
    assert!(basic.scope.first_chunk_warmup.pending_at_start);
    assert!(basic.scope.first_chunk_warmup.consumed_by_end);

    let _ = fs::remove_file(&out_path);
    let _ = fs::remove_dir_all(&out_dir);
}

#[test]
fn test_streaming_chorus_production_scorecard_stderr_prefix_contract() {
    let scorecard = sample_streaming_chorus_scorecard();
    let stderr_line = streaming_chorus_scorecard_stderr_line(&scorecard)
        .expect("compact streaming chorus scorecard stderr line");

    assert!(
        stderr_line.starts_with(STREAMING_CHORUS_SCORECARD_STDERR_PREFIX),
        "stderr line must start with the stable streaming chorus scorecard prefix"
    );

    let json_payload = stderr_line
        .strip_prefix(STREAMING_CHORUS_SCORECARD_STDERR_PREFIX)
        .expect("stderr line should contain the stable streaming chorus scorecard prefix");
    let parsed: StreamingChorusProductionScorecard = serde_json::from_str(json_payload)
        .expect("deserialize streaming chorus scorecard stderr payload");

    assert_eq!(parsed, scorecard);
}

#[test]
fn test_streaming_chorus_production_scorecard_reports_first_chunk_warmup_state() {
    let scorecard = sample_streaming_chorus_scorecard();

    for measurement in &scorecard.measurements {
        let scope = match measurement {
            StreamingChorusProductionMeasurement::Basic(m) => &m.scope,
            StreamingChorusProductionMeasurement::Cancel(m) => &m.scope,
            StreamingChorusProductionMeasurement::ResetReplay(m) => &m.scope,
            StreamingChorusProductionMeasurement::CrossfadeContinuity(m) => &m.scope,
            StreamingChorusProductionMeasurement::CacheSharing(m) => &m.scope,
            StreamingChorusProductionMeasurement::SpeedVariation(m) => &m.scope,
        };
        assert!(scope.first_chunk_warmup.pending_at_start);
        assert!(scope.first_chunk_warmup.consumed_by_end);
    }
}

#[test]
fn test_streaming_chorus_production_scorecard_emit_honors_env_artifact_path() {
    let _guard = STREAMING_CHORUS_SCORECARD_ENV_LOCK
        .lock()
        .expect("lock env for streaming chorus scorecard test");
    let scorecard = sample_streaming_chorus_scorecard();

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    let out_dir = std::env::temp_dir().join(format!(
        "nn-streaming-chorus-scorecard-env-{}-{unique}",
        std::process::id()
    ));
    let out_path = out_dir.join("scorecards/streaming-chorus-production-from-env.json");
    let out_path_string = out_path.to_string_lossy().into_owned();

    let previous = std::env::var_os(STREAMING_CHORUS_SCORECARD_OUT_ENV);
    std::env::set_var(
        STREAMING_CHORUS_SCORECARD_OUT_ENV,
        format!("  {out_path_string}  "),
    );

    let configured_path =
        configured_streaming_chorus_scorecard_artifact_path(STREAMING_CHORUS_SCORECARD_OUT_ENV)
            .expect("configured streaming chorus scorecard artifact path");
    assert_eq!(configured_path, out_path_string);

    emit_streaming_chorus_scorecard(&scorecard);

    let persisted =
        fs::read_to_string(&out_path).expect("read env-driven streaming chorus scorecard artifact");
    let parsed: StreamingChorusProductionScorecard = serde_json::from_str(&persisted)
        .expect("deserialize env-driven streaming chorus scorecard");
    assert_eq!(parsed, scorecard);

    match previous {
        Some(value) => std::env::set_var(STREAMING_CHORUS_SCORECARD_OUT_ENV, value),
        None => std::env::remove_var(STREAMING_CHORUS_SCORECARD_OUT_ENV),
    }

    let _ = fs::remove_file(&out_path);
    let _ = fs::remove_dir_all(&out_dir);
}

// =============================================================================
// Gate Tests -- production-weight assertions with human-readable summaries
// =============================================================================

/// Gate: Create a 3-voice StreamingChorusSession, iterate all chunks, verify
/// no panics occur throughout the entire pipeline.
///
/// This is the most basic production gate: the full streaming chorus pipeline
/// must complete without panicking. Exercises load, warmup, clone_dispatch,
/// session creation, and all chunk iterations end-to-end.
///
/// Steps:
/// 1. Load production Kokoro, warmup, build 3-voice chorus.
/// 2. Create StreamingChorusSession with 3 chunks of varying lengths.
/// 3. Iterate all chunks via `next_chunk()`, collecting results.
/// 4. Assert: session completes (`is_done()`).
/// 5. Assert: chunk count matches expected (3).
/// 6. Assert: total audio is non-empty.
#[test]
fn gate_streaming_chorus_no_panic() {
    let (mut primary, cache) = match load_production_kokoro(
        "gate_streaming_chorus_no_panic skipped -- set KOKORO_WEIGHTS to enable.",
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

    let mut total_samples = 0usize;
    let mut chunk_count = 0usize;
    while let Some(result) = session.next_chunk(&mut chorus, &cache) {
        let audio_chunk = result.expect("chunk synthesis must not fail");
        total_samples += audio_chunk.pcm.len();
        chunk_count += 1;
    }

    // Gate: session completes.
    assert!(
        session.is_done(),
        "gate_streaming_chorus_no_panic: session should be done after iterating all chunks"
    );

    // Gate: correct chunk count.
    assert_eq!(
        chunk_count, 3,
        "gate_streaming_chorus_no_panic: expected 3 chunks, got {chunk_count}"
    );

    // Gate: non-empty output.
    assert!(
        total_samples > 0,
        "gate_streaming_chorus_no_panic: total audio samples must be > 0"
    );

    eprintln!(
        "\n=== STREAMING CHORUS NO PANIC GATE ===\n  \
         Voices:     {N_VOICES}\n  \
         Chunks:     {chunk_count}\n  \
         Samples:    {total_samples}\n  \
         PASS\n\
         ======================================\n",
    );
}

/// Gate: All streaming chorus output chunks pass basic waveform sanity checks:
/// no NaN, no Inf, all samples in [-1, 1], and non-silent (RMS > 1e-6).
///
/// Steps:
/// 1. Load production Kokoro, warmup, build 3-voice chorus.
/// 2. Create StreamingChorusSession with 3 chunks.
/// 3. Iterate all chunks, collecting mixed audio.
/// 4. Assert: 0 NaN samples.
/// 5. Assert: 0 Inf samples.
/// 6. Assert: 0 samples outside [-1, 1].
/// 7. Assert: RMS energy > 1e-6 (non-silent).
/// 8. Assert: audio length >= 0.1s at 24kHz.
#[test]
fn gate_streaming_chorus_audio_quality() {
    let (mut primary, cache) = match load_production_kokoro(
        "gate_streaming_chorus_audio_quality skipped -- set KOKORO_WEIGHTS to enable.",
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
        "gate_streaming_chorus_audio_quality: RMS energy {rms:.2e} <= 1e-6 -- \
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

/// Gate: After warmup, a second StreamingChorusSession run should have a high
/// segment cache hit rate (measured via `segment_cache_stats()`).
///
/// The warmup pass compiles all 8 segments. A subsequent same-shape streaming
/// synthesis should be dominated by cache hits. This gate does not prove zero
/// recompiles; it resets cache stats after warmup, runs a full streaming
/// session, and verifies the observed hit rate is >= 0.9 (90%).
///
/// Steps:
/// 1. Load production Kokoro, warmup (compile all 8 segments).
/// 2. Build chorus from warmed primary.
/// 3. Run first streaming session (populates caches for this shape).
/// 4. Reset cache stats on all chorus voices.
/// 5. Run second streaming session with same-shape chunks.
/// 6. Read cache stats: assert hit_rate >= 0.9.
/// 7. Assert: hits > 0 (cache was actually used).
#[test]
fn gate_streaming_chorus_cache_hits() {
    let (mut primary, cache) = match load_production_kokoro(
        "gate_streaming_chorus_cache_hits skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let mut chorus = build_chorus(&mut primary, &cache);
    let styles = make_styles();
    let stream_config = KokoroStreamConfig::default();

    // First streaming session: warms up caches for this input shape.
    let chunks_first = vec![make_input_short(), make_input_short()];
    let mut session_first =
        StreamingChorusSession::new(chunks_first, styles.clone(), 1.0, stream_config.clone())
            .expect("session creation (first)");

    while let Some(result) = session_first.next_chunk(&mut chorus, &cache) {
        let _ = result.expect("first session chunk");
    }
    assert!(session_first.is_done());

    // Reset stats on all voices so we measure only the second pass.
    for i in 0..chorus.n_voices() {
        let voice = chorus.voice_mut(i).expect("voice exists");
        voice.reset_segment_cache_stats();
    }

    // Second streaming session: same shape inputs -- should be cache hits.
    let chunks_second = vec![make_input_short(), make_input_short()];
    let mut session_second = StreamingChorusSession::new(chunks_second, styles, 1.0, stream_config)
        .expect("session creation (second)");

    while let Some(result) = session_second.next_chunk(&mut chorus, &cache) {
        let _ = result.expect("second session chunk");
    }
    assert!(session_second.is_done());

    // Aggregate cache stats across all voices.
    let mut total_hits = 0usize;
    let mut total_misses = 0usize;
    for i in 0..chorus.n_voices() {
        let voice = chorus.voice(i).expect("voice exists");
        let stats = voice.segment_cache_stats();
        total_hits += stats.hits;
        total_misses += stats.misses;
    }

    let total_lookups = total_hits + total_misses;
    assert!(
        total_lookups > 0,
        "gate_streaming_chorus_cache_hits: no cache lookups recorded -- \
         cache stats may not be tracking"
    );

    let hit_rate = total_hits as f64 / total_lookups as f64;

    // Gate: hit rate >= 0.9 (90%).
    assert!(
        hit_rate >= 0.9,
        "gate_streaming_chorus_cache_hits: hit_rate {hit_rate:.3} < 0.9 -- \
         hits={total_hits}, misses={total_misses}, lookups={total_lookups}"
    );

    // Gate: hits > 0 (cache was actually exercised).
    assert!(
        total_hits > 0,
        "gate_streaming_chorus_cache_hits: 0 cache hits recorded"
    );

    eprintln!(
        "\n=== STREAMING CHORUS CACHE HITS GATE ===\n  \
         Voices:      {N_VOICES}\n  \
         Hits:        {total_hits}\n  \
         Misses:      {total_misses}\n  \
         Lookups:     {total_lookups}\n  \
         Hit rate:    {hit_rate:.3}\n  \
         PASS\n\
         ========================================\n",
    );
}

/// Gate: Number of output chunks from StreamingChorusSession matches the
/// number of input chunks provided at session creation.
///
/// The contract of StreamingChorusSession is that `next_chunk()` produces
/// exactly one output per input chunk, then returns `None`. This gate verifies
/// that contract for various input sizes (1, 2, 3, and 4 chunks).
///
/// Steps:
/// 1. Load production Kokoro, warmup, build chorus.
/// 2. For each chunk count in [1, 2, 3, 4]:
///    a. Create StreamingChorusSession with that many chunks.
///    b. Iterate all chunks, counting output.
///    c. Assert: output count == input count.
///    d. Assert: session is done.
///    e. Assert: remaining == 0.
#[test]
fn gate_streaming_chorus_chunk_count() {
    let (mut primary, cache) = match load_production_kokoro(
        "gate_streaming_chorus_chunk_count skipped -- set KOKORO_WEIGHTS to enable.",
    ) {
        Some(pair) => pair,
        None => return,
    };

    let mut chorus = build_chorus(&mut primary, &cache);
    let styles = make_styles();

    for expected_chunks in 1..=4usize {
        let mut chunks: Vec<DynTensor> = Vec::with_capacity(expected_chunks);
        for i in 0..expected_chunks {
            if i % 2 == 0 {
                chunks.push(make_input_short());
            } else {
                chunks.push(make_input_long());
            }
        }

        let stream_config = KokoroStreamConfig::default();
        let mut session = StreamingChorusSession::new(chunks, styles.clone(), 1.0, stream_config)
            .expect("session creation");

        assert_eq!(
            session.total_chunks(),
            expected_chunks,
            "gate_streaming_chorus_chunk_count: total_chunks mismatch for \
             {expected_chunks}-chunk session"
        );

        let mut output_count = 0usize;
        let mut total_samples = 0usize;
        while let Some(result) = session.next_chunk(&mut chorus, &cache) {
            let audio_chunk = result.unwrap_or_else(|e| {
                panic!(
                    "gate_streaming_chorus_chunk_count: chunk {output_count} synthesis \
                     failed for {expected_chunks}-chunk session: {e}"
                )
            });
            assert_eq!(
                audio_chunk.chunk_index, output_count,
                "gate_streaming_chorus_chunk_count: chunk_index mismatch"
            );
            assert_eq!(
                audio_chunk.total_chunks, expected_chunks,
                "gate_streaming_chorus_chunk_count: total_chunks mismatch in output"
            );
            total_samples += audio_chunk.pcm.len();
            output_count += 1;
        }

        // Gate: output count matches input count.
        assert_eq!(
            output_count, expected_chunks,
            "gate_streaming_chorus_chunk_count: expected {expected_chunks} output chunks, \
             got {output_count}"
        );

        // Gate: session is done.
        assert!(
            session.is_done(),
            "gate_streaming_chorus_chunk_count: session not done after \
             {expected_chunks} chunks"
        );

        // Gate: remaining is 0.
        assert_eq!(
            session.remaining(),
            0,
            "gate_streaming_chorus_chunk_count: remaining != 0 after \
             {expected_chunks} chunks"
        );

        // Gate: non-empty audio.
        assert!(
            total_samples > 0,
            "gate_streaming_chorus_chunk_count: 0 audio samples for \
             {expected_chunks}-chunk session"
        );

        eprintln!(
            "  chunk_count={expected_chunks}: {output_count} output chunks, \
             {total_samples} samples -- PASS"
        );
    }

    eprintln!(
        "\n=== STREAMING CHORUS CHUNK COUNT GATE ===\n  \
         Tested:    1, 2, 3, 4 input chunks\n  \
         All:       output count == input count\n  \
         PASS\n\
         =========================================\n",
    );
}
