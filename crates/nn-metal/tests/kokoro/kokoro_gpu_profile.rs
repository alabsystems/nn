// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU execution time profiler for the Kokoro synthesis pipeline.
//!
//! Uses [`synthesize_with_gpu_timing`] which flushes the GPU command buffer
//! after each pipeline step to measure actual GPU execution time per segment.
//! This reveals which segments are GPU-bound vs. encoding-bound.
//!
//! Requires `KOKORO_WEIGHTS` env var pointing to kokoro_v1_0.safetensors.
//!
//! Run:
//!   KOKORO_WEIGHTS=path/to/kokoro_v1_0.safetensors \
//!   cargo test -p nn-metal --test kokoro_all kokoro_gpu_profile -- --nocapture
//!
//! Part of #4264.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

fn cpu() -> Device {
    Device::Cpu
}

/// GPU execution time profile: short (20 token) and medium (80 token) inputs.
///
/// Prints per-step GPU timing breakdown showing actual GPU execution time
/// per pipeline segment (encode, prosody, regulate, f0_energy, harmonic,
/// generate, istft, verify).
///
/// This is a diagnostic tool — the per-step GPU flushes add overhead, so
/// total time will be higher than production `synthesize()`.
#[test]
fn kokoro_gpu_timing_profile() {
    let weights_path = match super::kokoro_test_env::require_kokoro_weights(
        "GPU timing profile not run. Set KOKORO_WEIGHTS to enable.",
    ) {
        Some(path) => path,
        None => return,
    };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let mut hb = nn_tts_verify::HardBoundsConfig::default();
    hb.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;

    // SAFETY: safetensors file not modified while alive.
    let mut kokoro = unsafe {
        nn_metal::compiled_kokoro::CompiledKokoro::load_with_hard_bounds(&weights_path, hb)
            .expect("failed to load Kokoro weights")
    }
    .with_recommended_autocast();

    let style = DynTensor::full(&[1, 256], 0.01, DType::F32, &cpu()).unwrap();

    let test_inputs: Vec<(&str, usize)> = vec![("short", 20), ("medium", 80)];

    for (label, token_count) in &test_inputs {
        let tokens: Vec<i64> = (0..*token_count).map(|i| (i % 178) as i64).collect();
        let ids = DynTensor::from_vec_i64(tokens, &[1, *token_count], &cpu()).unwrap();

        // Warmup call (triggers compilation).
        let _ = kokoro
            .synthesize(&ids, &style, 1.0, &cache)
            .expect("warmup failed");

        // Profiling call with per-step GPU flushes.
        let (audio, _cert, gpu_timing) = kokoro
            .synthesize_with_gpu_timing(&ids, &style, 1.0, &cache)
            .expect("GPU timing synthesis failed");

        let num_samples = audio.numel();
        let audio_secs = num_samples as f64 / 24_000.0;
        let total_secs = gpu_timing.total.as_secs_f64();

        eprintln!("\n=== GPU Timing Profile: {label} ({token_count} tokens) ===");
        eprintln!("{gpu_timing}");
        eprintln!(
            "  audio:      {:>8.2} ms  ({} samples)",
            audio_secs * 1000.0,
            num_samples,
        );
        eprintln!(
            "  RTF (profiled): {:.4}  (overhead from per-step flushes)",
            total_secs / audio_secs,
        );
        eprintln!();

        // Sanity: all step durations should be non-zero (GPU actually ran).
        assert!(
            gpu_timing.encode.as_nanos() > 0,
            "encode GPU time should be > 0",
        );
        assert!(
            gpu_timing.generate.as_nanos() > 0,
            "generate GPU time should be > 0",
        );
        assert!(
            gpu_timing.istft.as_nanos() > 0,
            "istft GPU time should be > 0",
        );

        // Sanity: total should be >= sum of individual steps.
        let sum = gpu_timing.encode
            + gpu_timing.prosody
            + gpu_timing.regulate
            + gpu_timing.f0_energy
            + gpu_timing.harmonic
            + gpu_timing.generate
            + gpu_timing.istft
            + gpu_timing.verify;
        assert!(
            gpu_timing.total >= sum,
            "total ({:?}) should be >= sum of steps ({:?})",
            gpu_timing.total,
            sum,
        );

        // Cache misses should be 0 on the profiling call (warmup already compiled).
        assert_eq!(
            gpu_timing.cache_misses, 0,
            "cache misses should be 0 after warmup",
        );
    }
}
