// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU vs CPU iSTFT benchmark (#1393 AC4, Stage 5 of #1370).
//!
//! Measures wall-clock latency for GPU iSTFT vs CPU iSTFT at three scales:
//! - Small (n_fft=64, 8 frames): dispatch overhead dominates
//! - Medium (n_fft=256, 32 frames): moderate workload
//! - Production (n_fft=4096, 16 frames): HTDemucs actual parameters
//!
//! Run with `cargo test -p nn-metal --lib -- test_istft_bench -- --nocapture`

use std::time::Instant;

use crate::istft::{IstftBasis, IstftParams};
use crate::istft_gpu::IstftGpuBasis;
use crate::test_common::{assert_close, init, make_cache};

/// Generate deterministic STFT data with realistic amplitude distribution.
fn generate_bench_stft(n_bins: usize, n_frames: usize) -> (Vec<f32>, Vec<f32>) {
    let mut real = Vec::with_capacity(n_bins * n_frames);
    let mut imag = Vec::with_capacity(n_bins * n_frames);
    for f in 0..n_bins {
        for t in 0..n_frames {
            // Pseudo-random but deterministic, with frequency-dependent decay.
            let decay = 1.0 / (1.0 + f as f32 * 0.01);
            let phase = ((f * 97 + t * 31) % 1000) as f32 * 0.001;
            real.push(phase * decay);
            imag.push((1.0 - phase) * decay * 0.5);
        }
    }
    (real, imag)
}

/// Benchmark GPU vs CPU iSTFT at a given configuration.
///
/// Returns `(cpu_ms, gpu_ms, speedup)` averaged over `iters` timed runs.
fn bench_istft(n_fft: usize, n_frames: usize, warmup: usize, iters: usize) -> (f64, f64, f64) {
    let hop = n_fft / 4;
    let n_bins = n_fft / 2 + 1;

    let params = IstftParams::new(n_fft, hop, true, true).expect("valid params");
    let basis = IstftBasis::new(params).expect("valid params");
    let cache = make_cache().expect("Metal backend required");
    let gpu_basis = IstftGpuBasis::from_basis(&basis).expect("GPU basis upload");

    let (real, imag) = generate_bench_stft(n_bins, n_frames);
    let output_len = n_fft + (n_frames.saturating_sub(1)) * hop;

    // Warmup: compile Metal pipelines + warm CPU caches.
    for _ in 0..warmup {
        let _ = basis
            .istft(&real, &imag, n_frames, output_len)
            .expect("CPU warmup");
        let _ = gpu_basis
            .gpu_istft_from_cpu(&cache, &real, &imag, n_frames, output_len)
            .expect("GPU warmup");
    }

    let iters_f = iters as f64;

    // Benchmark CPU path.
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = basis
            .istft(&real, &imag, n_frames, output_len)
            .expect("CPU iSTFT");
    }
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters_f;

    // Benchmark GPU path.
    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = gpu_basis
            .gpu_istft_from_cpu(&cache, &real, &imag, n_frames, output_len)
            .expect("GPU iSTFT");
    }
    let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0 / iters_f;

    // Correctness check: GPU and CPU outputs must be close.
    let cpu_out = basis
        .istft(&real, &imag, n_frames, output_len)
        .expect("CPU correctness");
    let gpu_out = gpu_basis
        .gpu_istft_from_cpu(&cache, &real, &imag, n_frames, output_len)
        .expect("GPU correctness");
    // Use relaxed tolerance for large n_fft (4096 accumulates more floating-point error).
    let tol = if n_fft >= 1024 { 5e-3 } else { 1e-3 };
    assert_close(&gpu_out, &cpu_out, tol, &format!("bench_nfft{n_fft}"));

    let speedup = if gpu_ms > 0.0 { cpu_ms / gpu_ms } else { 0.0 };

    eprintln!(
        "  n_fft={n_fft:>5} frames={n_frames:>3} | CPU: {cpu_ms:>10.3}ms | \
         GPU: {gpu_ms:>10.3}ms | speedup: {speedup:.2}x | output_len: {output_len}"
    );

    (cpu_ms, gpu_ms, speedup)
}

#[test]
fn test_istft_bench_small() {
    init();
    if make_cache().is_none() {
        return; // Skip on non-Metal platforms.
    }

    eprintln!("\n=== iSTFT GPU vs CPU benchmark ===");

    // Small: dispatch overhead dominates.
    let (cpu_ms, gpu_ms, _) = bench_istft(64, 8, 1, 2);
    assert!(cpu_ms > 0.0, "CPU timing must be positive");
    assert!(gpu_ms > 0.0, "GPU timing must be positive");
}

#[test]
fn test_istft_bench_medium() {
    init();
    if make_cache().is_none() {
        return;
    }

    eprintln!("\n=== iSTFT GPU vs CPU benchmark (medium) ===");

    // Medium: moderate workload.
    let (cpu_ms, gpu_ms, _) = bench_istft(256, 32, 1, 2);
    assert!(cpu_ms > 0.0, "CPU timing must be positive");
    assert!(gpu_ms > 0.0, "GPU timing must be positive");
}

#[test]
fn test_istft_bench_production() {
    init();
    if make_cache().is_none() {
        return;
    }

    eprintln!("\n=== iSTFT GPU vs CPU benchmark (production: HTDemucs n_fft=4096) ===");

    // Production HTDemucs parameters: n_fft=4096, hop=1024, 16 frames.
    // This is the exact configuration used in the spectral decoder.
    let (cpu_ms, gpu_ms, _) = bench_istft(4096, 16, 1, 1);
    assert!(cpu_ms > 0.0, "CPU timing must be positive");
    assert!(gpu_ms > 0.0, "GPU timing must be positive");
}
