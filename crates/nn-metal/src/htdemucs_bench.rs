// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HTDemucs GPU vs CPU inference benchmark (#1385, Epic: #1370 AC1).
//!
//! Measures wall-clock latency for `forward()` (CPU dispatch) vs `forward_gpu()`
//! (Metal buffer-to-buffer dispatch) using synthetic weights. Reports average
//! latency and GPU/CPU speedup ratio.
//!
//! Run with `cargo test -p nn-metal test_htdemucs_bench -- --nocapture`
//! to see timing output.

use std::time::Instant;

use super::*;
use crate::demucs_test_common::make_htdemucs_weights;
use crate::test_common::make_cache;

/// Deterministic pseudo-random audio data (no `rand` crate dependency).
fn synthetic_audio(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i + seed) % 97) as f32) * 0.01 - 0.5)
        .collect()
}

/// Benchmark HTDemucs forward() vs forward_gpu() at a given audio_t.
///
/// Returns `(cpu_ms, gpu_ms, speedup)` averaged over `iters` timed runs.
fn bench_forward(audio_t: usize, warmup: usize, iters: usize) -> (f64, f64, f64) {
    let weights = make_htdemucs_weights();
    let model = HTDemucs::new(weights, audio_t).expect("valid weights");
    let cache = make_cache().expect("Metal backend required");

    // Stereo audio: [2, T] flattened.
    let audio = synthetic_audio(2 * audio_t, 42);

    // Warmup: compile Metal pipelines + warm CPU caches.
    for _ in 0..warmup {
        let _ = model.forward(&cache, &audio).expect("CPU warmup");
        let _ = model.forward_gpu(&cache, &audio).expect("GPU warmup");
    }

    let iters_f = iters as f64;

    // Benchmark CPU path.
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = model.forward(&cache, &audio).expect("CPU forward");
    }
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters_f;

    // Benchmark GPU path.
    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = model.forward_gpu(&cache, &audio).expect("GPU forward");
    }
    let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0 / iters_f;

    // Correctness: GPU and CPU outputs must be close.
    let cpu_out = model.forward(&cache, &audio).expect("CPU correctness");
    let gpu_out = model.forward_gpu(&cache, &audio).expect("GPU correctness");
    assert_eq!(cpu_out.len(), gpu_out.len(), "output length mismatch");
    let max_err = cpu_out
        .iter()
        .zip(gpu_out.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);
    // GPU and CPU paths must agree within tolerance. With synthetic constant
    // weights (0.01) the outputs are bit-identical; use 0.1 as a generous
    // regression guard matching matmul_bench convention.
    assert!(
        max_err < 0.1,
        "GPU vs CPU max error {max_err} exceeds tolerance 0.1 at audio_t={audio_t}"
    );

    let speedup = if gpu_ms > 0.0 { cpu_ms / gpu_ms } else { 0.0 };

    eprintln!(
        "  audio_t={audio_t:>6} | CPU: {cpu_ms:>8.2}ms | GPU: {gpu_ms:>8.2}ms | \
         speedup: {speedup:.2}x | max_err: {max_err:.6}"
    );

    (cpu_ms, gpu_ms, speedup)
}

#[test]
fn test_htdemucs_bench_small() {
    eprintln!("\n=== HTDemucs GPU vs CPU benchmark (synthetic weights) ===");

    // Small: audio_t=256 (dispatch overhead dominates).
    let (cpu_ms, gpu_ms, _speedup) = bench_forward(256, 1, 1);
    assert!(cpu_ms > 0.0, "CPU timing must be positive");
    assert!(gpu_ms > 0.0, "GPU timing must be positive");
}

#[test]
fn test_htdemucs_bench_medium() {
    eprintln!("\n=== HTDemucs GPU vs CPU benchmark (medium) ===");

    // Medium: audio_t=2048 (~0.13s of audio at 16kHz).
    let (cpu_ms, gpu_ms, _speedup) = bench_forward(2048, 1, 1);
    assert!(cpu_ms > 0.0, "CPU timing must be positive");
    assert!(gpu_ms > 0.0, "GPU timing must be positive");
}

#[test]
fn test_htdemucs_bench_production() {
    eprintln!("\n=== HTDemucs GPU vs CPU benchmark (production-length) ===");

    // Production: audio_t=16384 (~1s of audio at 16kHz).
    // Using fewer iterations to keep test time reasonable with synthetic weights.
    let (cpu_ms, gpu_ms, _speedup) = bench_forward(16384, 1, 1);
    assert!(cpu_ms > 0.0, "CPU timing must be positive");
    assert!(gpu_ms > 0.0, "GPU timing must be positive");
}
