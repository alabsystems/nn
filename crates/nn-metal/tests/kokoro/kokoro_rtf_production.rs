// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Production RTF (Real-Time Factor) benchmark for Kokoro TTS.
//!
//! Measures end-to-end synthesis performance with production weights across
//! three text lengths (short ~20 tokens, medium ~80 tokens, long ~200 tokens),
//! using the recommended per-segment F16 autocast path.
//!
//! RTF = synthesis_wall_time / audio_duration. Lower is better; < 1.0 means
//! faster than real-time.
//!
//! This benchmark covers a single-voice timing surface with synthetic token
//! inputs. It does not certify audio quality or general production readiness.
//!
//! Reports:
//! - Per-length wall-clock timing breakdown (encode, prosody, regulate,
//!   f0_energy, harmonic, generate, istft, verify)
//! - Total synthesis time vs. audio duration
//! - Per-length and average RTF
//! - Compiled dispatch counts plus per-utterance runtime flush/encoding counts
//! - A machine-readable JSON scorecard surface for issue/docs artifacts
//!
//! Gates:
//! - Debug:   RTF < 0.5 (lenient, tighten as perf improves)
//! - Release: RTF < 0.2
//!
//! Requires `KOKORO_WEIGHTS` env var pointing to kokoro_v1_0.safetensors.
//!
//! Run:
//!   KOKORO_WEIGHTS=path/to/kokoro_v1_0.safetensors \
//!   cargo test -p nn-metal --test kokoro_all kokoro_rtf_production -- --nocapture
//!
//! Optional artifact output:
//!   KOKORO_RTF_SCORECARD_OUT=/tmp/kokoro-rtf-scorecard.json
//!
//! Part of #3828 (self-optimizing compiler).

use std::fs;
use std::path::Path;
use std::time::Duration;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use nn_metal::PrecompileShapes;
use serde::{Deserialize, Serialize};

fn cpu() -> Device {
    Device::Cpu
}

/// Sample rate for Kokoro TTS output (24 kHz).
const SAMPLE_RATE: f64 = 24_000.0;

/// Number of measured iterations per text input for stable timing.
const BENCH_ITERS: usize = 3;

/// Debug-mode RTF gate. Lenient because debug builds are materially slower.
const RTF_GATE_DEBUG: f64 = 0.5;

/// Release-mode RTF gate for optimized builds.
const RTF_GATE_RELEASE: f64 = 0.2;

/// Stable schema version for the machine-readable scorecard output.
const SCORECARD_SCHEMA_VERSION: u32 = 1;

/// Optional artifact path for the machine-readable benchmark scorecard.
const SCORECARD_OUT_ENV: &str = "KOKORO_RTF_SCORECARD_OUT";

/// Stable stderr prefix for compact JSON scorecard logs.
const SCORECARD_STDERR_PREFIX: &str = "KOKORO_RTF_SCORECARD_JSON=";

/// Test utterances of varying length to measure RTF scaling behavior.
/// Each entry is (label, token IDs). Token IDs are synthetic (modulo Kokoro
/// vocab size 178) but representative of production token distributions.
///
/// Three lengths exercise different pipeline characteristics:
/// - Short (~20 tokens): dominated by pipeline overhead and JIT latency
/// - Medium (~80 tokens): typical sentence length, balanced compute
/// - Long (~200 tokens): paragraph-length, dominated by generator/iSTFT
fn test_utterances() -> Vec<(&'static str, Vec<i64>)> {
    vec![
        // Short: greeting or short phrase (~20 phonemes)
        ("short_20tok", (0..20).map(|i| i64::from(i % 178)).collect()),
        // Medium: typical sentence (~80 phonemes)
        ("medium_80tok", (0..80).map(|i| i64::from(i % 178)).collect()),
        // Long: paragraph-length (~200 phonemes)
        ("long_200tok", (0..200).map(|i| i64::from(i % 178)).collect()),
    ]
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BenchmarkWorkloadKind {
    SingleVoiceProduction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BenchmarkMeasuredSurface {
    SingleVoiceCompiledKokoroOutput,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BenchmarkMeasuredHarness {
    PrewarmedCompiledKokoroSynthesizeWithDiagnostics,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct BenchmarkWorkload {
    kind: BenchmarkWorkloadKind,
    voice_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ProductionRtfScorecardLimitation {
    SingleVoiceOnly,
    SyntheticTokenInputs,
    TimingDependsOnBuildAndHardware,
    NotAudioQualityCertificate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct StageTimingMs {
    encode: f64,
    prosody: f64,
    regulate: f64,
    f0_energy: f64,
    harmonic: f64,
    generate: f64,
    istft: f64,
    verify: f64,
    total: f64,
}

impl StageTimingMs {
    fn stage_pairs(&self) -> [(&'static str, f64); 8] {
        [
            ("encode", self.encode),
            ("prosody", self.prosody),
            ("regulate", self.regulate),
            ("f0_energy", self.f0_energy),
            ("harmonic", self.harmonic),
            ("generate", self.generate),
            ("istft", self.istft),
            ("verify", self.verify),
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct RuntimeDispatchScorecard {
    avg_compute_encodings: f64,
    avg_blits: f64,
    avg_total_metal_command_encodings: f64,
    avg_flushes: f64,
    avg_submits: f64,
    avg_blits_eliminated: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CompiledDispatchSegments {
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
struct CompiledDispatchScorecard {
    logical_total: usize,
    estimated_metal_total: usize,
    estimated_encoding_events_total: usize,
    expected_submit_count: usize,
    segments: CompiledDispatchSegments,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct UtteranceScorecard {
    label: String,
    token_count: usize,
    num_samples: usize,
    avg_wall_ms: f64,
    avg_audio_ms: f64,
    rtf: f64,
    avg_cache_misses: f64,
    stage_avg_ms: StageTimingMs,
    runtime_dispatch: RuntimeDispatchScorecard,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ProductionRtfScorecard {
    schema_version: u32,
    benchmark_name: String,
    build_profile: String,
    measured_surface: BenchmarkMeasuredSurface,
    measured_harness: BenchmarkMeasuredHarness,
    autocast_mode: String,
    workload: BenchmarkWorkload,
    limitations: Vec<ProductionRtfScorecardLimitation>,
    iterations_per_utterance: usize,
    warmup_precompile_shapes_compiled: usize,
    rtf_gate: f64,
    utterances: Vec<UtteranceScorecard>,
    overall_rtf: f64,
    total_wall_ms: f64,
    total_audio_ms: f64,
    compiled_dispatch: CompiledDispatchScorecard,
}

impl ProductionRtfScorecard {
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

    fn declares_expected_surface(&self) -> bool {
        self.measured_surface == BenchmarkMeasuredSurface::SingleVoiceCompiledKokoroOutput
            && self.measured_harness
                == BenchmarkMeasuredHarness::PrewarmedCompiledKokoroSynthesizeWithDiagnostics
            && self.workload.kind == BenchmarkWorkloadKind::SingleVoiceProduction
            && self.workload.voice_count == 1
    }
}

fn build_profile_name() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

fn dispatch_scorecard(
    kokoro: &nn_metal::compiled_kokoro::CompiledKokoro,
) -> CompiledDispatchScorecard {
    let summary = kokoro.dispatch_summary();
    CompiledDispatchScorecard {
        logical_total: summary.total(),
        estimated_metal_total: kokoro.total_metal_dispatches(),
        estimated_encoding_events_total: kokoro.total_encoding_events(),
        expected_submit_count: summary.expected_submit_count(),
        segments: CompiledDispatchSegments {
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

fn averaged_stage_timings(
    stage_accum: &[Duration; 8],
    total_wall_secs: f64,
    n: f64,
) -> StageTimingMs {
    StageTimingMs {
        encode: stage_accum[0].as_secs_f64() * 1000.0 / n,
        prosody: stage_accum[1].as_secs_f64() * 1000.0 / n,
        regulate: stage_accum[2].as_secs_f64() * 1000.0 / n,
        f0_energy: stage_accum[3].as_secs_f64() * 1000.0 / n,
        harmonic: stage_accum[4].as_secs_f64() * 1000.0 / n,
        generate: stage_accum[5].as_secs_f64() * 1000.0 / n,
        istft: stage_accum[6].as_secs_f64() * 1000.0 / n,
        verify: stage_accum[7].as_secs_f64() * 1000.0 / n,
        total: total_wall_secs * 1000.0 / n,
    }
}

fn averaged_runtime_dispatch(
    compute_encodings: usize,
    blits: usize,
    flushes: usize,
    submits: usize,
    blits_eliminated: usize,
    n: f64,
) -> RuntimeDispatchScorecard {
    let avg_compute_encodings = compute_encodings as f64 / n;
    let avg_blits = blits as f64 / n;
    RuntimeDispatchScorecard {
        avg_compute_encodings,
        avg_blits,
        avg_total_metal_command_encodings: avg_compute_encodings + avg_blits,
        avg_flushes: flushes as f64 / n,
        avg_submits: submits as f64 / n,
        avg_blits_eliminated: blits_eliminated as f64 / n,
    }
}

fn print_report(report: &ProductionRtfScorecard) {
    let build_mode = report.build_profile.to_ascii_uppercase();

    eprintln!("\n{}", "=".repeat(120));
    eprintln!(
        "  KOKORO RTF PRODUCTION BENCHMARK (recommended F16 autocast, {build_mode}, single voice)"
    );
    eprintln!(
        "  PrecompileShapes warmup: {} shapes, Bench: {} iters per utterance",
        report.warmup_precompile_shapes_compiled, report.iterations_per_utterance
    );
    eprintln!("  Gate: RTF < {} ({build_mode})", report.rtf_gate);
    eprintln!("{}", "=".repeat(120));

    eprintln!(
        "\n  {:<16} {:>6} {:>8} {:>10} {:>10} {:>8} {:>8} {:>8}",
        "Utterance", "Tokens", "Samples", "Wall (ms)", "Audio (ms)", "RTF", "CmpEnc", "Flush",
    );
    eprintln!("  {}", "-".repeat(94));
    for u in &report.utterances {
        eprintln!(
            "  {:<16} {:>6} {:>8} {:>10.2} {:>10.2} {:>8.4} {:>8.1} {:>8.1}",
            u.label,
            u.token_count,
            u.num_samples,
            u.avg_wall_ms,
            u.avg_audio_ms,
            u.rtf,
            u.runtime_dispatch.avg_compute_encodings,
            u.runtime_dispatch.avg_flushes,
        );
    }
    eprintln!("  {}", "-".repeat(94));
    eprintln!(
        "  {:<16} {:>6} {:>8} {:>10.2} {:>10.2} {:>8.4} {:>8} {:>8}",
        "OVERALL",
        "-",
        "-",
        report.total_wall_ms,
        report.total_audio_ms,
        report.overall_rtf,
        "-",
        "-",
    );

    if let Some(longest) = report.utterances.last() {
        eprintln!(
            "\n  Per-stage breakdown (avg, {} = {} tokens):",
            longest.label, longest.token_count
        );
        let total_stage_ms: f64 = longest
            .stage_avg_ms
            .stage_pairs()
            .iter()
            .map(|(_, ms)| *ms)
            .sum();
        for (name, ms) in longest.stage_avg_ms.stage_pairs() {
            let pct = if total_stage_ms > 0.0 {
                ms / total_stage_ms * 100.0
            } else {
                0.0
            };
            eprintln!("    {name:<14} {ms:>8.2} ms  ({pct:>5.1}%)");
        }
        eprintln!("    {}", "-".repeat(36));
        eprintln!("    {:<14} {:>8.2} ms", "total", longest.stage_avg_ms.total);
    }

    eprintln!("\n  Compiled dispatch metrics:");
    eprintln!(
        "    Logical dispatches:          {}",
        report.compiled_dispatch.logical_total
    );
    eprintln!(
        "    Estimated Metal dispatches:  {}",
        report.compiled_dispatch.estimated_metal_total
    );
    eprintln!(
        "    Estimated encoding events:   {}",
        report.compiled_dispatch.estimated_encoding_events_total
    );
    eprintln!(
        "    Expected submits / call:     {}",
        report.compiled_dispatch.expected_submit_count
    );

    eprintln!(
        "\n  RTF: {:.4} (gate: < {}, {build_mode})",
        report.overall_rtf, report.rtf_gate
    );
    if report.overall_rtf < report.rtf_gate {
        eprintln!("  PASS");
    } else {
        eprintln!("  FAIL");
    }
    eprintln!("{}\n", "=".repeat(120));
}

fn scorecard_stderr_line(report: &ProductionRtfScorecard) -> Option<String> {
    report
        .to_compact_json()
        .ok()
        .map(|compact_json| format!("{SCORECARD_STDERR_PREFIX}{compact_json}"))
}

fn configured_scorecard_artifact_path(env_var: &str) -> Option<String> {
    std::env::var(env_var)
        .ok()
        .map(|path| path.trim().to_string())
        .filter(|path| !path.is_empty())
}

fn emit_scorecard_with_artifact_path(
    report: &ProductionRtfScorecard,
    artifact_path: Option<&str>,
) -> Option<String> {
    let stderr_line = scorecard_stderr_line(report);
    if let Some(line) = stderr_line.as_deref() {
        eprintln!("{line}");
    }

    if let Some(path) = artifact_path {
        report
            .write_json(path)
            .unwrap_or_else(|e| panic!("failed to write scorecard artifact {path}: {e}"));
        eprintln!("  Scorecard artifact: {path}");
    }

    stderr_line
}

fn emit_scorecard(report: &ProductionRtfScorecard) {
    let artifact_path = configured_scorecard_artifact_path(SCORECARD_OUT_ENV);
    let _ = emit_scorecard_with_artifact_path(report, artifact_path.as_deref());
}

fn sample_production_rtf_scorecard() -> ProductionRtfScorecard {
    ProductionRtfScorecard {
        schema_version: SCORECARD_SCHEMA_VERSION,
        benchmark_name: "kokoro_rtf_production".to_string(),
        build_profile: "release".to_string(),
        measured_surface: BenchmarkMeasuredSurface::SingleVoiceCompiledKokoroOutput,
        measured_harness:
            BenchmarkMeasuredHarness::PrewarmedCompiledKokoroSynthesizeWithDiagnostics,
        autocast_mode: "recommended_f16".to_string(),
        workload: BenchmarkWorkload {
            kind: BenchmarkWorkloadKind::SingleVoiceProduction,
            voice_count: 1,
        },
        limitations: vec![
            ProductionRtfScorecardLimitation::SingleVoiceOnly,
            ProductionRtfScorecardLimitation::SyntheticTokenInputs,
            ProductionRtfScorecardLimitation::TimingDependsOnBuildAndHardware,
            ProductionRtfScorecardLimitation::NotAudioQualityCertificate,
        ],
        iterations_per_utterance: 3,
        warmup_precompile_shapes_compiled: 12,
        rtf_gate: RTF_GATE_RELEASE,
        utterances: vec![UtteranceScorecard {
            label: "medium_80tok".to_string(),
            token_count: 80,
            num_samples: 24_000,
            avg_wall_ms: 96.5,
            avg_audio_ms: 1000.0,
            rtf: 0.0965,
            avg_cache_misses: 0.0,
            stage_avg_ms: StageTimingMs {
                encode: 4.0,
                prosody: 6.0,
                regulate: 8.0,
                f0_energy: 10.0,
                harmonic: 12.0,
                generate: 40.0,
                istft: 12.5,
                verify: 4.0,
                total: 96.5,
            },
            runtime_dispatch: RuntimeDispatchScorecard {
                avg_compute_encodings: 445.0,
                avg_blits: 12.0,
                avg_total_metal_command_encodings: 457.0,
                avg_flushes: 3.0,
                avg_submits: 2.0,
                avg_blits_eliminated: 7.0,
            },
        }],
        overall_rtf: 0.0965,
        total_wall_ms: 96.5,
        total_audio_ms: 1000.0,
        compiled_dispatch: CompiledDispatchScorecard {
            logical_total: 476,
            estimated_metal_total: 674,
            estimated_encoding_events_total: 701,
            expected_submit_count: 6,
            segments: CompiledDispatchSegments {
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
    }
}

#[cfg(test)]
static RTF_SCORECARD_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Production RTF benchmark: end-to-end Kokoro synthesis with real weights.
///
/// Measures wall-clock time and audio duration across short (~20 tokens),
/// medium (~80 tokens), and long (~200 tokens) utterances using recommended
/// per-segment F16 autocast on this timing surface.
///
/// Uses `PrecompileShapes::default()` for warmup to match production startup.
///
/// Reports per-length and average RTF, per-stage timing, and dispatch counts.
/// The machine-readable scorecard explicitly marks this as warmed
/// single-voice `CompiledKokoro::synthesize_with_diagnostics()` output.
///
/// Gates:
/// - Debug:   RTF < 0.5 (lenient for unoptimized builds)
/// - Release: RTF < 0.2
///
/// Part of #3828.
#[test]
fn kokoro_rtf_production_benchmark() {
    let weights_path = match super::kokoro_test_env::require_kokoro_weights(
        "Production RTF benchmark not run. Set KOKORO_WEIGHTS to enable.",
    ) {
        Some(path) => path,
        None => return,
    };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Use Warn policy: benchmark measures timing, not audio quality.
    // Synthetic token IDs may produce click artifacts. Part of #4262.
    let mut hb = nn_tts_verify::HardBoundsConfig::default();
    hb.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;

    // Load with the recommended per-segment F16 autocast path.
    // SAFETY: safetensors file not modified while alive.
    let mut kokoro = unsafe {
        nn_metal::compiled_kokoro::CompiledKokoro::load_with_hard_bounds(&weights_path, hb)
            .expect("failed to load Kokoro weights")
    }
    .with_recommended_autocast();

    let style = DynTensor::full(&[1, 256], 0.01, DType::F32, &cpu()).unwrap();
    let speed = 1.0;
    let rtf_gate = if cfg!(debug_assertions) {
        RTF_GATE_DEBUG
    } else {
        RTF_GATE_RELEASE
    };

    let utterances = test_utterances();

    // Warmup using PrecompileShapes::default() to match production startup.
    // This pre-compiles segment caches for common input sizes.
    let warmup_count = kokoro
        .warmup(&PrecompileShapes::default(), &cache)
        .expect("PrecompileShapes warmup failed");
    eprintln!("  Warmup: compiled {warmup_count} segment shapes via PrecompileShapes::default()");

    // Also warm up each utterance length once (segment cache may differ by shape).
    for (_label, tokens) in &utterances {
        let ids = DynTensor::from_vec_i64(tokens.clone(), &[1, tokens.len()], &cpu()).unwrap();
        let _ = kokoro
            .synthesize(&ids, &style, speed, &cache)
            .expect("per-length warmup failed");
    }

    // Benchmark each utterance.
    let mut results = Vec::new();
    let mut grand_wall_secs = 0.0_f64;
    let mut grand_audio_secs = 0.0_f64;

    for (label, tokens) in &utterances {
        let ids = DynTensor::from_vec_i64(tokens.clone(), &[1, tokens.len()], &cpu()).unwrap();

        let mut total_wall_secs = 0.0_f64;
        let mut total_audio_secs = 0.0_f64;
        let mut stage_accum = [Duration::ZERO; 8];
        let mut cache_miss_accum = 0usize;
        let mut compute_encodings_accum = 0usize;
        let mut blits_accum = 0usize;
        let mut flushes_accum = 0usize;
        let mut submits_accum = 0usize;
        let mut blits_eliminated_accum = 0usize;
        let mut last_num_samples = 0usize;

        for _ in 0..BENCH_ITERS {
            let (audio, _cert, diagnostics) = kokoro
                .synthesize_with_diagnostics(&ids, &style, speed, &cache)
                .expect("benchmark synthesis failed");
            let timing = diagnostics.timing;
            let stats = diagnostics.stats;

            let num_samples = *audio.dims().last().expect("audio dim");
            assert!(num_samples > 0, "synthesis produced 0 audio samples");
            last_num_samples = num_samples;
            let audio_secs = num_samples as f64 / SAMPLE_RATE;

            total_wall_secs += timing.total.as_secs_f64();
            total_audio_secs += audio_secs;

            stage_accum[0] += timing.encode;
            stage_accum[1] += timing.prosody;
            stage_accum[2] += timing.regulate;
            stage_accum[3] += timing.f0_energy;
            stage_accum[4] += timing.harmonic;
            stage_accum[5] += timing.generate;
            stage_accum[6] += timing.istft;
            stage_accum[7] += timing.verify;
            cache_miss_accum += timing.cache_misses;
            compute_encodings_accum += stats.compute_encodings;
            blits_accum += stats.blits;
            flushes_accum += stats.flushes;
            submits_accum += stats.submits;
            blits_eliminated_accum += stats.blits_eliminated;
        }

        let n = BENCH_ITERS as f64;
        let avg_wall_ms = total_wall_secs * 1000.0 / n;
        let avg_audio_ms = total_audio_secs * 1000.0 / n;
        let rtf = total_wall_secs / total_audio_secs;

        grand_wall_secs += total_wall_secs;
        grand_audio_secs += total_audio_secs;

        results.push(UtteranceScorecard {
            label: (*label).to_string(),
            token_count: tokens.len(),
            num_samples: last_num_samples,
            avg_wall_ms,
            avg_audio_ms,
            rtf,
            avg_cache_misses: cache_miss_accum as f64 / n,
            stage_avg_ms: averaged_stage_timings(&stage_accum, total_wall_secs, n),
            runtime_dispatch: averaged_runtime_dispatch(
                compute_encodings_accum,
                blits_accum,
                flushes_accum,
                submits_accum,
                blits_eliminated_accum,
                n,
            ),
        });
    }

    let overall_rtf = grand_wall_secs / grand_audio_secs;
    let overall_n = (utterances.len() * BENCH_ITERS) as f64;
    let total_wall_ms = grand_wall_secs * 1000.0 / overall_n;
    let total_audio_ms = grand_audio_secs * 1000.0 / overall_n;

    let report = ProductionRtfScorecard {
        schema_version: SCORECARD_SCHEMA_VERSION,
        benchmark_name: "kokoro_rtf_production".to_string(),
        build_profile: build_profile_name().to_string(),
        measured_surface: BenchmarkMeasuredSurface::SingleVoiceCompiledKokoroOutput,
        measured_harness:
            BenchmarkMeasuredHarness::PrewarmedCompiledKokoroSynthesizeWithDiagnostics,
        autocast_mode: "recommended_f16".to_string(),
        workload: BenchmarkWorkload {
            kind: BenchmarkWorkloadKind::SingleVoiceProduction,
            voice_count: 1,
        },
        limitations: vec![
            ProductionRtfScorecardLimitation::SingleVoiceOnly,
            ProductionRtfScorecardLimitation::SyntheticTokenInputs,
            ProductionRtfScorecardLimitation::TimingDependsOnBuildAndHardware,
            ProductionRtfScorecardLimitation::NotAudioQualityCertificate,
        ],
        iterations_per_utterance: BENCH_ITERS,
        warmup_precompile_shapes_compiled: warmup_count,
        rtf_gate,
        utterances: results,
        overall_rtf,
        total_wall_ms,
        total_audio_ms,
        compiled_dispatch: dispatch_scorecard(&kokoro),
    };
    assert!(
        report.declares_expected_surface(),
        "production RTF scorecard should declare the measured single-voice harness explicitly"
    );

    print_report(&report);
    emit_scorecard(&report);

    // Per-length RTF reporting (informational).
    for u in &report.utterances {
        eprintln!(
            "  RTF({}, {} tokens, {} samples): {:.4}",
            u.label, u.token_count, u.num_samples, u.rtf
        );
    }

    // Gate assertion: varies by build mode.
    assert!(
        report.overall_rtf < rtf_gate,
        "Kokoro production RTF {:.4} exceeds gate {rtf_gate} ({}) on this timing surface. \
         Performance regression detected.",
        report.overall_rtf,
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );

    // In release mode, also assert the tighter bound explicitly.
    if cfg!(not(debug_assertions)) {
        assert!(
            report.overall_rtf < RTF_GATE_RELEASE,
            "Kokoro release RTF {:.4} exceeds release gate {RTF_GATE_RELEASE}.",
            report.overall_rtf
        );
    }
}

#[test]
fn kokoro_rtf_scorecard_json_round_trip() {
    let report = sample_production_rtf_scorecard();

    let compact_json = report.to_compact_json().expect("serialize compact json");
    let compact_parsed: ProductionRtfScorecard =
        serde_json::from_str(&compact_json).expect("deserialize compact scorecard json");

    let pretty_json = report.to_pretty_json().expect("serialize pretty json");
    let pretty_parsed: ProductionRtfScorecard =
        serde_json::from_str(&pretty_json).expect("deserialize pretty scorecard json");

    assert_eq!(compact_parsed, report);
    assert_eq!(pretty_parsed, report);
    assert_eq!(
        compact_parsed.measured_surface,
        BenchmarkMeasuredSurface::SingleVoiceCompiledKokoroOutput
    );
    assert_eq!(
        compact_parsed.measured_harness,
        BenchmarkMeasuredHarness::PrewarmedCompiledKokoroSynthesizeWithDiagnostics
    );
    assert!(
        compact_parsed.declares_expected_surface(),
        "scorecard surface/harness declaration should survive round-trip"
    );
    assert_eq!(compact_parsed.workload.voice_count, 1);
    assert_eq!(
        compact_parsed.workload.kind,
        BenchmarkWorkloadKind::SingleVoiceProduction
    );
    assert_eq!(
        compact_parsed.utterances[0].runtime_dispatch.avg_flushes, 3.0,
        "runtime flush count should survive round-trip"
    );
    assert_eq!(
        compact_parsed.limitations, report.limitations,
        "scorecard limitations should survive round-trip"
    );
}

#[test]
fn kokoro_rtf_scorecard_write_json_artifact() {
    let report = sample_production_rtf_scorecard();

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    let out_dir = std::env::temp_dir().join(format!(
        "nn-kokoro-rtf-scorecard-{}-{unique}",
        std::process::id()
    ));
    let out_path = out_dir.join("scorecards/production.json");

    report
        .write_json(&out_path)
        .expect("write scorecard artifact");

    let persisted = fs::read_to_string(&out_path).expect("read persisted scorecard");
    assert!(
        persisted.contains('\n'),
        "artifact writer should emit pretty JSON"
    );
    let parsed: ProductionRtfScorecard =
        serde_json::from_str(&persisted).expect("deserialize persisted scorecard");

    assert_eq!(parsed.schema_version, SCORECARD_SCHEMA_VERSION);
    assert_eq!(
        parsed.measured_surface,
        BenchmarkMeasuredSurface::SingleVoiceCompiledKokoroOutput
    );
    assert_eq!(
        parsed.measured_harness,
        BenchmarkMeasuredHarness::PrewarmedCompiledKokoroSynthesizeWithDiagnostics
    );
    assert!(
        parsed.declares_expected_surface(),
        "persisted scorecard should keep the explicit measured surface/harness"
    );
    assert_eq!(parsed.workload.voice_count, 1);
    assert_eq!(parsed.compiled_dispatch.expected_submit_count, 6);
    assert_eq!(parsed.limitations, report.limitations);

    let _ = fs::remove_file(&out_path);
    let _ = fs::remove_dir_all(&out_dir);
}

#[test]
fn kokoro_rtf_scorecard_stderr_prefix_contract() {
    let report = sample_production_rtf_scorecard();
    let stderr_line = scorecard_stderr_line(&report).expect("compact scorecard stderr line");

    assert!(
        stderr_line.starts_with(SCORECARD_STDERR_PREFIX),
        "stderr line must start with the stable scorecard prefix"
    );

    let json_payload = stderr_line
        .strip_prefix(SCORECARD_STDERR_PREFIX)
        .expect("stderr line should contain the stable prefix");
    let parsed: ProductionRtfScorecard =
        serde_json::from_str(json_payload).expect("deserialize scorecard stderr payload");

    assert_eq!(parsed, report);
    assert_eq!(
        parsed.measured_surface,
        BenchmarkMeasuredSurface::SingleVoiceCompiledKokoroOutput
    );
    assert_eq!(
        parsed.measured_harness,
        BenchmarkMeasuredHarness::PrewarmedCompiledKokoroSynthesizeWithDiagnostics
    );
    assert!(
        parsed.declares_expected_surface(),
        "stderr scorecard should expose the explicit measured surface/harness"
    );
}

#[test]
fn kokoro_rtf_scorecard_emit_honors_env_artifact_path() {
    let _guard = RTF_SCORECARD_ENV_LOCK
        .lock()
        .expect("lock env for scorecard test");
    let report = sample_production_rtf_scorecard();

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    let out_dir = std::env::temp_dir().join(format!(
        "nn-kokoro-rtf-scorecard-env-{}-{unique}",
        std::process::id()
    ));
    let out_path = out_dir.join("scorecards/production-from-env.json");
    let out_path_string = out_path.to_string_lossy().into_owned();

    let previous = std::env::var_os(SCORECARD_OUT_ENV);
    std::env::set_var(SCORECARD_OUT_ENV, &out_path_string);

    let configured_path =
        configured_scorecard_artifact_path(SCORECARD_OUT_ENV).expect("configured artifact path");
    assert_eq!(configured_path, out_path_string);

    emit_scorecard(&report);

    let persisted = fs::read_to_string(&out_path).expect("read env-driven scorecard artifact");
    let parsed: ProductionRtfScorecard =
        serde_json::from_str(&persisted).expect("deserialize env-driven scorecard artifact");
    assert_eq!(parsed, report);
    assert_eq!(
        parsed.measured_surface,
        BenchmarkMeasuredSurface::SingleVoiceCompiledKokoroOutput
    );
    assert_eq!(
        parsed.measured_harness,
        BenchmarkMeasuredHarness::PrewarmedCompiledKokoroSynthesizeWithDiagnostics
    );
    assert!(
        parsed.declares_expected_surface(),
        "env-driven scorecard artifact should keep the explicit measured surface/harness"
    );

    match previous {
        Some(value) => std::env::set_var(SCORECARD_OUT_ENV, value),
        None => std::env::remove_var(SCORECARD_OUT_ENV),
    }

    let _ = fs::remove_file(&out_path);
    let _ = fs::remove_dir_all(&out_dir);
}
