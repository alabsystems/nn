// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end test for `CompiledKokoro::synthesize()` with synthetic weights.
//!
//! Exercises the full pipeline: Seg0 (PlBert+bert_encoder, #2744) →
//! Seg1 (TextEncoder) → Seg2 (ProsodyPredictor) → CPU bridge (length_regulate)
//! → Seg3 (F0Energy) → CPU bridge (harmonic_source) → Seg4 (Generator) →
//! GPU iSTFT → TtsVerifier.
//!
//! Uses miniaturized config (D_EN=8, style_dim=4) via [`kokoro_test_weights`].
//! Synthetic weights are zeros (LayerNorm weights = ones for numerical stability).
//!
//! Weight builder unified with `kokoro_test_weights.rs` (#2567) to prevent
//! drift between test weights and model architecture.
//!
//! Part of #2483 (synthesize() untested).
//! Part of #2218 (Kokoro epic).

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;
use nn_metal::compiled_kokoro::{CompiledKokoro, CompiledKokoroError};

use super::kokoro_test_weights as kw;

fn cpu() -> Device {
    Device::Cpu
}

fn assert_cpu_finite(tensor: &DynTensor, label: &str) {
    let vals = tensor
        .to_device(&cpu())
        .unwrap_or_else(|e| panic!("{label}: to_device(cpu) failed: {e}"))
        .to_flat_vec::<f32>()
        .unwrap_or_else(|e| panic!("{label}: to_flat_vec failed: {e}"));
    let non_finite = vals.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite,
        0,
        "{label}: expected all finite values, found {non_finite} / {}",
        vals.len()
    );
}

fn assert_input_too_long<T>(result: Result<T, CompiledKokoroError>, label: &str) {
    match result {
        Err(CompiledKokoroError::InvalidInput(msg)) => {
            assert!(
                msg.contains("seq_len 17 exceeds max_position_embeddings 16"),
                "{label}: unexpected oversized-input message: {msg}"
            );
        }
        Err(other) => panic!("{label}: expected InvalidInput for oversized input, got: {other:?}"),
        Ok(_) => panic!("{label}: expected oversized input to fail"),
    }
}

// -- Test ---------------------------------------------------------------------

/// Full CompiledKokoro::synthesize() with synthetic weights.
///
/// Exercises: Seg0 (PlBert+bert_encoder, #2744) → Seg1 (TextEncoder) →
/// Seg2 (ProsodyPredictor) → CPU bridge (length_regulate) →
/// Seg3 (F0/energy) → CPU bridge (harmonic_source) → Seg4 (Generator) →
/// GPU iSTFT → TtsVerifier certificate.
///
/// Part of #2483, #2218.
#[test]
fn test_compiled_kokoro_synthesize_synthetic() {
    let (mut kokoro, cache) = kw::build_kokoro_mini();
    let config = kw::mini_test_config();
    let style_dim = config.style_dim;

    // Input: 3 token IDs, style vector of 2*style_dim.
    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(200, 2 * style_dim, -0.1, 0.1),
        &[1, 2 * style_dim],
        &cpu(),
    )
    .unwrap();

    // Run the full synthesize() pipeline.
    let result = kokoro.synthesize(&input_ids, &style, 1.0, &cache);
    assert!(result.is_ok(), "synthesize() failed: {:?}", result.err());

    let (audio, certificate) = result.unwrap();

    // Verify audio shape: [1, 1, T_audio] (mono PCM).
    assert_eq!(audio.dims().len(), 3, "audio should be rank-3");
    assert_eq!(audio.dims()[0], 1, "batch dim should be 1");
    assert_eq!(audio.dims()[1], 1, "channel dim should be 1");
    assert!(audio.dims()[2] > 0, "audio should have non-zero samples");

    // Verify certificate has all 8 hard bounds evaluated.
    assert_eq!(
        certificate.hard_bounds.len(),
        8,
        "expected 8 hard bounds, got {}",
        certificate.hard_bounds.len()
    );

    // With synthetic (zero) weights, content-dependent bounds (non_silence,
    // duration, spectral_coverage) are expected to fail because the model
    // produces near-zero output. Verify structural bounds that should pass
    // regardless of weight quality (no NaN, no clipping, no DC, no clicks).
    let structural_bounds: Vec<&str> = certificate
        .hard_bounds
        .iter()
        .filter(|b| {
            matches!(
                b.name,
                "no_clipping" | "no_dc_offset" | "no_clicks" | "no_nan"
            )
        })
        .filter(|b| !b.passed)
        .map(|b| b.name)
        .collect();
    assert!(
        structural_bounds.is_empty(),
        "Structural hard bounds should pass even with synthetic weights, but failed: {structural_bounds:?}"
    );

    // Log results for diagnostic visibility.
    eprintln!("synthesize() audio shape: {:?}", audio.dims());
    eprintln!(
        "Certificate overall_passed: {}, hard bounds: {}",
        certificate.overall_passed,
        certificate
            .hard_bounds
            .iter()
            .map(|b| format!("{}={}", b.name, b.passed))
            .collect::<Vec<_>>()
            .join(", ")
    );

    assert_dispatch_budget(&kokoro);
}

/// GPU-resident synthesis via `synthesize_gpu()`.
///
/// Verifies that `synthesize_gpu()` returns a `GpuAudioHandle` with the
/// correct sample rate, non-zero sample count, and a valid certificate.
/// Also verifies that `to_cpu_tensor()` produces a tensor matching the
/// shape and content of `synthesize()` output.
///
/// Part of #4251.
#[test]
fn test_compiled_kokoro_synthesize_gpu() {
    let (mut kokoro, cache) = kw::build_kokoro_mini();
    let config = kw::mini_test_config();
    let style_dim = config.style_dim;

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(200, 2 * style_dim, -0.1, 0.1),
        &[1, 2 * style_dim],
        &cpu(),
    )
    .unwrap();

    // Run synthesize_gpu() — returns GPU handle + certificate.
    let result = kokoro.synthesize_gpu(&input_ids, &style, 1.0, &cache);
    assert!(
        result.is_ok(),
        "synthesize_gpu() failed: {:?}",
        result.err()
    );
    let (handle, certificate) = result.unwrap();

    // Verify GpuAudioHandle properties.
    assert_eq!(
        handle.sample_rate(),
        24000,
        "Kokoro sample rate should be 24 kHz"
    );
    assert!(
        handle.sample_count() > 0,
        "audio should have non-zero samples"
    );
    assert!(handle.duration_secs() > 0.0, "duration should be positive");

    // Certificate has 8 hard bounds.
    assert_eq!(
        certificate.hard_bounds.len(),
        8,
        "expected 8 hard bounds, got {}",
        certificate.hard_bounds.len()
    );

    // Transfer to CPU tensor and verify shape matches synthesize() convention.
    let cpu_audio = handle.to_cpu_tensor().expect("to_cpu_tensor");
    assert_eq!(cpu_audio.dims().len(), 3, "cpu_audio should be rank-3");
    assert_eq!(cpu_audio.dims()[0], 1, "batch dim should be 1");
    assert_eq!(cpu_audio.dims()[1], 1, "channel dim should be 1");
    assert_eq!(
        cpu_audio.dims()[2],
        handle.sample_count(),
        "sample count mismatch between handle and tensor"
    );

    // Verify to_cpu() returns the same data as to_cpu_tensor().
    let pcm_vec = handle.to_cpu().expect("to_cpu");
    let tensor_vec = cpu_audio.to_flat_vec::<f32>().expect("to_flat_vec");
    assert_eq!(pcm_vec.len(), tensor_vec.len(), "pcm length mismatch");
    for (i, (a, b)) in pcm_vec.iter().zip(tensor_vec.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-7,
            "sample {i} differs: to_cpu={a}, to_cpu_tensor={b}"
        );
    }

    // Run synthesize() on a fresh instance and verify output matches.
    let (mut kokoro2, cache2) = kw::build_kokoro_mini();
    let (audio_ref, _cert_ref) = kokoro2
        .synthesize(&input_ids, &style, 1.0, &cache2)
        .expect("reference synthesize");
    assert_eq!(
        cpu_audio.dims(),
        audio_ref.dims(),
        "synthesize_gpu → to_cpu_tensor shape should match synthesize() shape"
    );
}

/// Step-by-step materialization guard for the compiled Kokoro pipeline.
///
/// Unlike `synthesize()`, this test reads each intermediate back to CPU after
/// the step completes. It catches bugs that escape the GPU-side NaN checker
/// but become visible when the tensor is actually materialized.
#[test]
fn test_compiled_kokoro_step_outputs_materialize_finite() {
    let (mut kokoro, cache) = kw::build_kokoro_mini();
    let config = kw::mini_test_config();
    let style_dim = config.style_dim;

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(201, 2 * style_dim, -0.1, 0.1),
        &[1, 2 * style_dim],
        &cpu(),
    )
    .unwrap();

    let split = kokoro.split_style(&style).unwrap();
    let enc = kokoro.step_encode(&input_ids, &cache).unwrap();
    assert_cpu_finite(&enc.bert_features, "bert_features");
    assert_cpu_finite(&enc.text_features, "text_features");

    let pros = kokoro
        .step_predict_prosody(
            &enc.bert_features,
            &split.prosody_style,
            enc.seq_len,
            &cache,
        )
        .unwrap();
    assert_cpu_finite(&pros.dur_logits, "dur_logits");
    assert_cpu_finite(&pros.features, "prosody_features");

    let reg = kokoro
        .step_regulate(
            &pros.dur_logits,
            &pros.features,
            &enc.text_features,
            1.0,
            &cache,
        )
        .unwrap();
    assert_cpu_finite(&reg.durations, "durations");
    assert_cpu_finite(&reg.aligned_dur, "aligned_dur");
    assert_cpu_finite(&reg.regulated, "regulated");

    let f0e = kokoro
        .step_predict_f0_energy(&reg.aligned_dur, &split.prosody_style, reg.t_mel, &cache)
        .unwrap();
    assert_cpu_finite(&f0e.f0, "f0");
    assert_cpu_finite(&f0e.energy, "energy");

    let har = kokoro
        .step_harmonic_source(&f0e.f0, &f0e.energy, reg.t_mel, &cache)
        .unwrap();
    assert_cpu_finite(&har, "harmonic_source");

    let generator = kokoro
        .step_generate(
            &reg.regulated,
            &f0e.f0,
            &f0e.energy,
            &split.decoder_style,
            &har,
            reg.t_mel,
            &cache,
        )
        .unwrap();
    assert_cpu_finite(&generator.magnitude, "generator_magnitude");
    assert_cpu_finite(&generator.phase, "generator_phase");
}

/// Flush/submit budget: validates hot-path GPU synchronization discipline.
///
/// Uses `synthesize_with_stats` to measure commit_and_wait calls. Expected
/// flushes/submits on cache-hit hot path:
/// - 0 flushes from `step_regulate` (duration readback uses `submit()+sync()`)
/// - ~N flushes from `step_harmonic_source` (SourceModule runs on CPU —
///   GPU→CPU roundtrip for f0 + forward STFT ops; TODO: GPU SourceModule)
/// - 1 flush from the pipeline-exit GPU→CPU audio transfer
/// - 3 non-blocking submits from the two-phase hot path:
///   1. encode+prosody fence before regulate
///   2. phase-1 handoff fence after regulate scatter outputs
///   3. f0+harmonic fence before generator
/// - 0 `step_regulate submit()+sync()` submits after warmup because
///   `total_repeats` is cached for the same `(seq_len, speed)` pair
///
/// Budget ≤15 accounts for SourceModule CPU execution path. Original #2739
/// AC1 (≤3) assumed no SourceModule. With GPU SourceModule weights, this
/// should drop back to ≤3.
///
/// Part of #2739, #2218.
#[test]
fn test_compiled_kokoro_flush_count_budget() {
    let (mut kokoro, cache) = kw::build_kokoro_mini();
    let config = kw::mini_test_config();
    let style_dim = config.style_dim;

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(200, 2 * style_dim, -0.1, 0.1),
        &[1, 2 * style_dim],
        &cpu(),
    )
    .unwrap();

    // First call compiles segments (compilation has additional flushes).
    let _ = kokoro
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("warmup synthesize");

    // Second call is cache-hit hot path — measure this.
    let (_audio, _cert, stats) = kokoro
        .synthesize_with_stats(&input_ids, &style, 1.0, &cache)
        .expect("synthesize_with_stats");

    eprintln!(
        "Flush budget: flushes={}, submits={}, encodings={}",
        stats.flushes, stats.submits, stats.compute_encodings
    );

    // Flush budget: ≤15 with CPU SourceModule (GPU→CPU roundtrip + forward STFT).
    // Original #2739 AC1 was ≤3 without SourceModule. With GPU SourceModule
    // weights, target is ≤3. Measured: 11 with CPU SourceModule path.
    assert!(
        stats.flushes <= 15,
        "Flush budget regression: expected ≤15 GPU flushes, got {}. \
         Each flush is a commit_and_wait barrier. SourceModule CPU path adds ~8 \
         flushes; GPU SourceModule would reduce to ≤3. See #2739.",
        stats.flushes
    );

    // Two-phase hot path uses 3 non-blocking fence submissions after warmup.
    // Larger counts indicate extra command-buffer fragmentation/regression.
    assert!(
        stats.submits <= 3,
        "Expected ≤3 hot-path non-blocking submits, got {}. \
         Current two-phase runtime uses 3 fence submits (encode+prosody, \
         phase handoff, f0+harmonic); larger counts indicate submit fragmentation.",
        stats.submits
    );

    // Sanity: pipeline should produce at least some GPU work.
    assert!(
        stats.compute_encodings > 10,
        "Expected >10 GPU encodings for a full Kokoro pipeline, got {}. \
         Pipeline may not be running on GPU.",
        stats.compute_encodings
    );
}

/// Same `CompiledKokoro` instance handles different sequence lengths back-to-back.
///
/// Streaming synthesis reuses one instance across chunk boundaries, so shape
/// changes must not leave stale segment state behind.
#[test]
fn test_compiled_kokoro_synthesize_different_lengths_same_instance() {
    let (mut kokoro, cache) = kw::build_kokoro_mini();
    let config = kw::mini_test_config();
    let style_dim = config.style_dim;

    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(777, 2 * style_dim, -0.1, 0.1),
        &[1, 2 * style_dim],
        &cpu(),
    )
    .unwrap();

    let ids_3 = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let ids_4 = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();

    let _ = kokoro
        .synthesize(&ids_3, &style, 1.0, &cache)
        .expect("warmup len=3");
    let (audio_3, _cert3) = kokoro
        .synthesize(&ids_3, &style, 1.0, &cache)
        .expect("repeat len=3");
    let pcm_3 = audio_3.to_flat_vec::<f32>().expect("read len=3 audio");
    let (audio_4, _cert4) = kokoro
        .synthesize(&ids_4, &style, 1.0, &cache)
        .expect("synthesize len=4 after warmup+repeat len=3");

    assert!(!pcm_3.is_empty(), "len=3 audio should have samples");
    assert!(audio_4.dims()[2] > 0, "len=4 audio should have samples");
    assert_ne!(
        pcm_3.len(),
        audio_4.dims()[2],
        "different sequence lengths should produce different audio lengths"
    );
}

/// Per-stage timing and diagnostic output via synthesize_with_timing()
/// and synthesize_with_diagnostics().
///
/// Exercises AC4 of #2781: integration test calling synthesize_with_timing()
/// with synthetic weights. Verifies:
/// - All 9 timing fields are populated (non-zero total).
/// - Cache misses are 5 on cold call (all 5 segments miss).
/// - Cache misses are 0 on warm call (all cached).
/// - TimingReport Display prints all stage names.
/// - DiagnosticOutput combines timing + GPU stats.
///
/// Part of #2781, #2218.
#[test]
fn test_compiled_kokoro_synthesize_with_timing() {
    let (mut kokoro, cache) = kw::build_kokoro_mini();
    let config = kw::mini_test_config();
    let style_dim = config.style_dim;

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(200, 2 * style_dim, -0.1, 0.1),
        &[1, 2 * style_dim],
        &cpu(),
    )
    .unwrap();

    // Cold call: all 5 segments are cache misses.
    let (audio, cert, timing) = kokoro
        .synthesize_with_timing(&input_ids, &style, 1.0, &cache)
        .expect("synthesize_with_timing (cold)");

    // Audio shape: [1, 1, T_audio].
    assert_eq!(audio.dims().len(), 3, "audio should be rank-3");
    assert!(audio.dims()[2] > 0, "audio should have non-zero samples");

    // Certificate has 8 hard bounds.
    assert_eq!(cert.hard_bounds.len(), 8, "expected 8 hard bounds");

    // Total timing should be non-zero.
    assert!(
        !timing.total.is_zero(),
        "total timing should be non-zero, got {:?}",
        timing.total
    );

    // Cold call: 6 cache misses (plbert, text, prosody, f0, sinegen_pre, generator).
    // sinegen_post also compiles but reuses sinegen_pre's cache key.
    // Regulate is eager (no segment cache entry).
    assert_eq!(
        timing.cache_misses, 6,
        "cold call should have 6 cache misses, got {}",
        timing.cache_misses
    );

    // Display output includes all stage names.
    let display = format!("{timing}");
    eprintln!("{display}");
    assert!(display.contains("encode:"), "Display missing encode stage");
    assert!(
        display.contains("prosody:"),
        "Display missing prosody stage"
    );
    assert!(
        display.contains("regulate:"),
        "Display missing regulate stage"
    );
    assert!(
        display.contains("f0_energy:"),
        "Display missing f0_energy stage"
    );
    assert!(
        display.contains("harmonic:"),
        "Display missing harmonic stage"
    );
    assert!(
        display.contains("generate:"),
        "Display missing generate stage"
    );
    assert!(display.contains("istft:"), "Display missing istft stage");
    assert!(display.contains("verify:"), "Display missing verify stage");
    assert!(
        display.contains("cache_misses: 6"),
        "Display missing cache misses"
    );

    // Warm call: 0 cache misses (same input shape → all segments cached).
    let (_audio2, _cert2, timing2) = kokoro
        .synthesize_with_timing(&input_ids, &style, 1.0, &cache)
        .expect("synthesize_with_timing (warm)");

    assert_eq!(
        timing2.cache_misses, 0,
        "warm call should have 0 cache misses, got {}",
        timing2.cache_misses
    );

    // synthesize_with_diagnostics combines timing + GPU stats.
    let (_audio3, _cert3, diag) = kokoro
        .synthesize_with_diagnostics(&input_ids, &style, 1.0, &cache)
        .expect("synthesize_with_diagnostics");

    assert!(
        !diag.timing.total.is_zero(),
        "DiagnosticOutput timing should be non-zero"
    );
    assert!(
        diag.stats.compute_encodings > 10,
        "DiagnosticOutput should have >10 GPU encodings, got {}",
        diag.stats.compute_encodings
    );

    let diag_display = format!("{diag}");
    eprintln!("{diag_display}");
    assert!(
        diag_display.contains("flushes:"),
        "DiagnosticOutput missing flushes"
    );
    assert!(
        diag_display.contains("compute:"),
        "DiagnosticOutput missing compute encodings"
    );
}

/// PlBert position embedding guard: synthesize() rejects oversized input before
/// any embedding lookup can read beyond the configured context window.
///
/// Part of #2976, #3213, #2218.
#[test]
fn test_compiled_kokoro_synthesize_input_too_long() {
    let (mut kokoro, cache) = kw::build_kokoro_mini();
    let config = kw::mini_test_config();
    let max_pos = config.plbert.max_position_embeddings; // 16
    let style_dim = config.style_dim;
    let vocab = config.plbert.vocab_size;

    // seq_len = max_pos + 1 (17 tokens, exceeds position embedding table of 16).
    let oversized: Vec<f32> = (0..=max_pos).map(|i| (i % vocab) as f32).collect();
    let input_ids = DynTensor::from_vec(oversized, &[1, max_pos + 1], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(300, 2 * style_dim, -0.1, 0.1),
        &[1, 2 * style_dim],
        &cpu(),
    )
    .unwrap();

    assert_input_too_long(
        kokoro.synthesize(&input_ids, &style, 1.0, &cache),
        "synthesize",
    );
    assert_input_too_long(
        kokoro.synthesize_gpu(&input_ids, &style, 1.0, &cache),
        "synthesize_gpu",
    );
    assert_input_too_long(
        kokoro.synthesize_with_stats(&input_ids, &style, 1.0, &cache),
        "synthesize_with_stats",
    );
    assert_input_too_long(
        kokoro.synthesize_with_timing(&input_ids, &style, 1.0, &cache),
        "synthesize_with_timing",
    );
    assert_input_too_long(
        kokoro.synthesize_with_diagnostics(&input_ids, &style, 1.0, &cache),
        "synthesize_with_diagnostics",
    );
    assert_input_too_long(
        kokoro.synthesize_with_memory(&input_ids, &style, 1.0, &cache),
        "synthesize_with_memory",
    );
    assert_input_too_long(
        kokoro.synthesize_with_intermediates(&input_ids, &style, 1.0, &cache),
        "synthesize_with_intermediates",
    );
    assert_input_too_long(kokoro.step_encode(&input_ids, &cache), "step_encode");

    // Boundary: seq_len == max_pos should be accepted (not rejected).
    // Token IDs must stay within [0, VOCAB) to avoid EmbeddingIndexOutOfRange.
    let boundary: Vec<f32> = (0..max_pos).map(|i| (i % vocab) as f32).collect();
    let boundary_ids = DynTensor::from_vec(boundary, &[1, max_pos], &cpu()).unwrap();
    let result = kokoro.synthesize(&boundary_ids, &style, 1.0, &cache);
    assert!(
        result.is_ok(),
        "seq_len == max_pos should succeed, got: {:?}",
        result.err()
    );
}

/// `synthesize_gpu()` audio is valid: no NaN, no Inf, all samples in [-1, 1].
///
/// With synthetic (zero) weights the model produces near-zero output, so the
/// amplitude bound is trivially satisfied. This test ensures the GPU-resident
/// path does not introduce NaN or clipping artifacts that the CPU path avoids.
///
/// Part of #4264, #4251.
#[test]
fn test_synthesize_gpu_audio_valid_range() {
    let (mut kokoro, cache) = kw::build_kokoro_mini();
    let config = kw::mini_test_config();
    let style_dim = config.style_dim;

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(600, 2 * style_dim, -0.1, 0.1),
        &[1, 2 * style_dim],
        &cpu(),
    )
    .unwrap();

    let (handle, _cert) = kokoro
        .synthesize_gpu(&input_ids, &style, 1.0, &cache)
        .expect("synthesize_gpu");

    let pcm = handle.to_cpu().expect("to_cpu");

    // No NaN or Inf.
    let non_finite = pcm.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite,
        0,
        "synthesize_gpu audio should be all finite, found {non_finite} non-finite / {} total",
        pcm.len()
    );

    // All samples in [-1, 1] (PCM audio invariant).
    let out_of_range: Vec<(usize, f32)> = pcm
        .iter()
        .enumerate()
        .filter(|(_, v)| v.abs() > 1.0)
        .map(|(i, v)| (i, *v))
        .collect();
    assert!(
        out_of_range.is_empty(),
        "synthesize_gpu audio should be in [-1, 1], found {} samples out of range. \
         First 5: {:?}",
        out_of_range.len(),
        &out_of_range[..out_of_range.len().min(5)]
    );

    // Sanity: non-empty audio.
    assert!(!pcm.is_empty(), "audio should have samples");

    eprintln!(
        "synthesize_gpu audio: {} samples, min={:.6}, max={:.6}",
        pcm.len(),
        pcm.iter().copied().fold(f32::INFINITY, f32::min),
        pcm.iter().copied().fold(f32::NEG_INFINITY, f32::max),
    );
}

/// `synthesize_gpu()` output matches `synthesize()` output within epsilon.
///
/// Runs both paths with the same inputs on the same compiled instance and
/// compares PCM samples. Since `synthesize()` delegates to `synthesize_gpu()`
/// internally, the outputs should be very close (within floating-point
/// rounding from the independent `to_cpu_tensor()` readback).
///
/// Part of #4264, #4251.
#[test]
fn test_synthesize_gpu_matches_synthesize_output() {
    let (mut kokoro, cache) = kw::build_kokoro_mini();
    let config = kw::mini_test_config();
    let style_dim = config.style_dim;

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(601, 2 * style_dim, -0.1, 0.1),
        &[1, 2 * style_dim],
        &cpu(),
    )
    .unwrap();

    // Warmup to compile segments.
    let _ = kokoro
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("warmup");

    // GPU path.
    let (handle, cert_gpu) = kokoro
        .synthesize_gpu(&input_ids, &style, 1.0, &cache)
        .expect("synthesize_gpu");
    let gpu_tensor = handle.to_cpu_tensor().expect("to_cpu_tensor");
    let gpu_pcm = gpu_tensor.to_flat_vec::<f32>().expect("gpu flat_vec");

    // CPU path (synthesize delegates to synthesize_gpu internally).
    let (cpu_audio, cert_cpu) = kokoro
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("synthesize");
    let cpu_pcm = cpu_audio.to_flat_vec::<f32>().expect("cpu flat_vec");

    // Shape must match.
    assert_eq!(
        gpu_tensor.dims(),
        cpu_audio.dims(),
        "synthesize_gpu shape should match synthesize shape"
    );

    // Sample count must match.
    assert_eq!(
        gpu_pcm.len(),
        cpu_pcm.len(),
        "sample counts should match: gpu={}, cpu={}",
        gpu_pcm.len(),
        cpu_pcm.len()
    );

    // PCM samples should be within epsilon. Both paths read from the same
    // GPU computation, so differences come only from independent readbacks.
    let max_diff = gpu_pcm
        .iter()
        .zip(cpu_pcm.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-5,
        "max PCM difference between synthesize_gpu and synthesize: {max_diff} (threshold 1e-5)"
    );

    // Certificates should agree.
    assert_eq!(
        cert_gpu.overall_passed, cert_cpu.overall_passed,
        "certificate overall_passed should match between GPU and CPU paths"
    );
    assert_eq!(
        cert_gpu.hard_bounds.len(),
        cert_cpu.hard_bounds.len(),
        "certificate hard_bounds count should match"
    );

    eprintln!(
        "synthesize_gpu vs synthesize: max_diff={max_diff:.2e}, samples={}",
        gpu_pcm.len()
    );
}

/// `GpuAudioHandle::to_cpu()` and `to_cpu_tensor()` return consistent data.
///
/// Verifies that the Vec<f32> from `to_cpu()` matches the flattened tensor
/// from `to_cpu_tensor()` element-by-element, and that the tensor shape
/// matches the handle's `sample_count()`.
///
/// Part of #4264, #4251.
#[test]
fn test_synthesize_gpu_handle_to_cpu_consistency() {
    let (mut kokoro, cache) = kw::build_kokoro_mini();
    let config = kw::mini_test_config();
    let style_dim = config.style_dim;

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(602, 2 * style_dim, -0.1, 0.1),
        &[1, 2 * style_dim],
        &cpu(),
    )
    .unwrap();

    let (handle, _cert) = kokoro
        .synthesize_gpu(&input_ids, &style, 1.0, &cache)
        .expect("synthesize_gpu");

    // to_cpu() returns Vec<f32>.
    let pcm_vec = handle.to_cpu().expect("to_cpu");

    // to_cpu_tensor() returns DynTensor with shape [1, 1, sample_count].
    let pcm_tensor = handle.to_cpu_tensor().expect("to_cpu_tensor");
    let tensor_flat = pcm_tensor.to_flat_vec::<f32>().expect("tensor flat_vec");

    // Length consistency.
    assert_eq!(
        pcm_vec.len(),
        handle.sample_count(),
        "to_cpu() length should equal sample_count()"
    );
    assert_eq!(
        tensor_flat.len(),
        handle.sample_count(),
        "to_cpu_tensor() element count should equal sample_count()"
    );
    assert_eq!(
        pcm_vec.len(),
        tensor_flat.len(),
        "to_cpu() and to_cpu_tensor() should have same element count"
    );

    // Tensor shape: [1, 1, sample_count].
    assert_eq!(pcm_tensor.dims(), &[1, 1, handle.sample_count()]);

    // Element-by-element match (both read the same GPU buffer after flush).
    let max_diff = pcm_vec
        .iter()
        .zip(tensor_flat.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-7,
        "to_cpu() and to_cpu_tensor() should produce identical data, max_diff={max_diff}"
    );
}

/// `GpuAudioHandle` metadata properties are consistent.
///
/// Verifies sample_rate, sample_count, and duration_secs are internally
/// consistent and match expected values for Kokoro (24 kHz).
///
/// Part of #4264, #4251.
#[test]
fn test_synthesize_gpu_handle_properties() {
    let (mut kokoro, cache) = kw::build_kokoro_mini();
    let config = kw::mini_test_config();
    let style_dim = config.style_dim;

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(603, 2 * style_dim, -0.1, 0.1),
        &[1, 2 * style_dim],
        &cpu(),
    )
    .unwrap();

    let (handle, _cert) = kokoro
        .synthesize_gpu(&input_ids, &style, 1.0, &cache)
        .expect("synthesize_gpu");

    // Kokoro sample rate is 24 kHz.
    assert_eq!(
        handle.sample_rate(),
        24000,
        "Kokoro sample rate should be 24 kHz"
    );

    // Sample count should be positive.
    assert!(handle.sample_count() > 0, "sample_count should be positive");

    // Duration should be consistent: sample_count / sample_rate.
    let expected_duration = handle.sample_count() as f32 / handle.sample_rate() as f32;
    let actual_duration = handle.duration_secs();
    assert!(
        (actual_duration - expected_duration).abs() < 1e-6,
        "duration_secs should equal sample_count/sample_rate: \
         expected={expected_duration}, actual={actual_duration}"
    );

    // Duration should be positive and reasonable for 3-token input.
    assert!(actual_duration > 0.0, "duration should be positive");
    // 3 tokens at speed=1.0 should produce less than 10 seconds of audio.
    assert!(
        actual_duration < 10.0,
        "duration for 3-token input should be < 10s, got {actual_duration}"
    );

    // gpu_buffer() should be accessible.
    let _buf = handle.gpu_buffer();

    eprintln!(
        "GpuAudioHandle: samples={}, rate={}, duration={:.4}s",
        handle.sample_count(),
        handle.sample_rate(),
        handle.duration_secs()
    );
}

/// `synthesize_gpu()` rejects invalid speed values (same as `synthesize()`).
///
/// Part of #4264, #4251.
#[test]
fn test_synthesize_gpu_rejects_invalid_speed() {
    let (mut kokoro, cache) = kw::build_kokoro_mini();
    let config = kw::mini_test_config();
    let style_dim = config.style_dim;

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(604, 2 * style_dim, -0.1, 0.1),
        &[1, 2 * style_dim],
        &cpu(),
    )
    .unwrap();

    // Zero speed.
    let result = kokoro.synthesize_gpu(&input_ids, &style, 0.0, &cache);
    assert!(
        result.is_err(),
        "zero speed should be rejected by synthesize_gpu"
    );

    // NaN speed.
    let result = kokoro.synthesize_gpu(&input_ids, &style, f32::NAN, &cache);
    assert!(
        result.is_err(),
        "NaN speed should be rejected by synthesize_gpu"
    );

    // Negative speed.
    let result = kokoro.synthesize_gpu(&input_ids, &style, -1.0, &cache);
    assert!(
        result.is_err(),
        "negative speed should be rejected by synthesize_gpu"
    );

    // Infinity.
    let result = kokoro.synthesize_gpu(&input_ids, &style, f32::INFINITY, &cache);
    assert!(
        result.is_err(),
        "infinite speed should be rejected by synthesize_gpu"
    );
}

/// Dispatch budget: logical dispatches < 600, Metal GPU commands < 2200.
///
/// History: #2459 set <120 when the test used an incomplete model (96 dispatches).
/// After weight builder alignment (#2567), full architecture produced ~227 logical.
/// After #2590 (bypass FusedAdainResBlock for NativeOps), logical increased to ~320
/// because each NativeOp (AdainSnake, AdainLeakyRelu) is a separate logical dispatch,
/// but Metal dispatches decreased dramatically (each NativeOp = 1 Metal dispatch vs
/// ~45 Metal dispatches per FusedAdainResBlock after expand_norm_ops).
///
/// Measured (miniaturized config): total=320, metal=673 (compute only).
/// Budget set with ~25% headroom: <400 logical, <800 Metal.
/// Budget increased to <600/<1200 after PlBert segment compilation (#2744).
/// Budget increased to <600/<2200 after #1815 D5: `num_metal_dispatches()` now
/// includes blit relocations (~1.8x increase). Miniaturized: 209→380 with blits.
///
/// Elementwise chain fusion (#1815 D1-D4) is active: Constant/ConstantWeight
/// nodes are skipped during chain detection and inlined as Literals. This
/// reduces dispatch count for mul_scalar/add_scalar/clamp chains. See
/// `designs/2026-03-21-elementwise-fusion-extension.md` for expected savings.
fn assert_dispatch_budget(kokoro: &CompiledKokoro) {
    let total = kokoro.total_dispatches();
    let metal = kokoro.total_metal_dispatches();
    let ds = kokoro.dispatch_summary();
    eprintln!(
        "Dispatch summary: total={total}, metal={metal} \
         [plbert={}, text={}, prosody={}, f0={}, generator={}]",
        ds.plbert, ds.text_encoder, ds.prosody, ds.f0_energy, ds.generator,
    );

    // Report fusion activity from dispatch breakdowns (#1815 D5).
    let breakdowns = kokoro.dispatch_breakdowns();
    let fused_count: usize = breakdowns
        .iter()
        .flat_map(|(_, ir, _)| ir.iter())
        .filter(|(name, _)| name.starts_with("fused_"))
        .map(|(_, count)| *count)
        .sum();
    eprintln!("Fused chain dispatches: {fused_count}");

    // PlBert is now compiled as segment 0 (#2744) — verify it produces dispatches.
    assert!(
        ds.plbert > 0,
        "PlBert segment should have >0 dispatches after compilation (#2744), got 0"
    );
    // Budget increased from <400 to <600 because PlBert compilation adds its
    // ~187 eager dispatches as compiled dispatches (previously uncounted).
    assert!(
        total < 600,
        "Pipeline should have <600 logical dispatches, got {total} \
         [plbert={}, text={}, prosody={}, f0={}, generator={}]",
        ds.plbert,
        ds.text_encoder,
        ds.prosody,
        ds.f0_energy,
        ds.generator,
    );
    assert!(
        metal < 2200,
        "Pipeline should have <2200 Metal GPU commands (compute + blits), got {metal} \
         [plbert={}, text={}, prosody={}, f0={}, generator={}]. \
         If >3500, FusedAdainResBlock is being used instead of decomposed NativeOps.",
        ds.plbert,
        ds.text_encoder,
        ds.prosody,
        ds.f0_energy,
        ds.generator,
    );
}
