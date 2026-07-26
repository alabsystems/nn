// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmarks for GPU matmul dispatch.
//!
//! Measures DynTensor matmul throughput at sizes typical for transformer
//! inference (Kokoro, Whisper, Qwen3). Covers both the simdgroup-tiled
//! path and the naive fallback path.
//!
//! Run: `cargo bench -p nn-metal --bench matmul --no-default-features`
//!
//! Part of #3218.

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

fn gpu() -> Device {
    Device::Metal { device_id: 0 }
}

/// Initialize Metal backend once per process.
fn ensure_gpu() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = nn_metal::MetalBackend::init().expect("Metal device required for bench");
        nn_metal::register_metal_dyn_backend();
    });
}

/// Create a deterministic GPU tensor with shape `[m, k]`.
fn gpu_matrix(m: usize, k: usize, seed: f32) -> DynTensor {
    let data: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * seed).sin() * 0.1)
        .collect();
    DynTensor::new(&data, &[m, k], &Device::Cpu)
        .unwrap()
        .to_device(&gpu())
        .unwrap()
}

/// Create a deterministic F16 GPU tensor with shape `[m, k]`.
/// Pattern: CPU F32 -> CPU F16 -> GPU (avoids cross-byte-width guard on GPU).
fn gpu_matrix_f16(m: usize, k: usize, seed: f32) -> DynTensor {
    let data: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * seed).sin() * 0.1)
        .collect();
    DynTensor::new(&data, &[m, k], &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap()
        .to_device(&gpu())
        .unwrap()
}

/// Benchmark matmul at a given (M, K, N) size.
fn bench_matmul(c: &mut Criterion, label: &str, m: usize, k: usize, n: usize) {
    ensure_gpu();
    let a = gpu_matrix(m, k, 0.01);
    let b = gpu_matrix(k, n, 0.02);

    c.bench_with_input(
        BenchmarkId::new(label, format!("{m}x{k}x{n}")),
        &(m, k, n),
        |bencher, _| {
            bencher.iter(|| {
                let result = a.matmul(black_box(&b)).unwrap();
                // Flush GPU to measure actual completion, not just dispatch.
                nn_metal::flush().unwrap();
                black_box(result);
            });
        },
    );
}

/// Benchmark F16 matmul at a given (M, K, N) size.
fn bench_matmul_f16(c: &mut Criterion, label: &str, m: usize, k: usize, n: usize) {
    ensure_gpu();
    let a = gpu_matrix_f16(m, k, 0.01);
    let b = gpu_matrix_f16(k, n, 0.02);

    c.bench_with_input(
        BenchmarkId::new(label, format!("{m}x{k}x{n}")),
        &(m, k, n),
        |bencher, _| {
            bencher.iter(|| {
                let result = a.matmul(black_box(&b)).unwrap();
                nn_metal::flush().unwrap();
                black_box(result);
            });
        },
    );
}

fn matmul_benchmarks(c: &mut Criterion) {
    // Simdgroup path: all dims % 8 == 0, M*N >= 16384, K >= 128.
    bench_matmul(c, "simdgroup", 256, 768, 3072); // Kokoro FFN up-project
    bench_matmul(c, "simdgroup", 256, 3072, 768); // Kokoro FFN down-project
    bench_matmul(c, "simdgroup", 128, 512, 512); // Whisper attention

    // Naive fallback: small or non-aligned.
    bench_matmul(c, "naive", 32, 64, 32); // Small matmul
    bench_matmul(c, "naive", 33, 65, 35); // Non-aligned dims

    // F16 simdgroup: same sizes as F32 for direct comparison.
    // Measured M4 Max: 1.03-1.35x. simdgroup_multiply_accumulate uses float
    // accumulators (half operands, float accumulation), so the speedup comes
    // from reduced memory bandwidth (half the bytes), not 2x compute.
    bench_matmul_f16(c, "simdgroup_f16", 256, 768, 3072);
    bench_matmul_f16(c, "simdgroup_f16", 256, 3072, 768);
    bench_matmul_f16(c, "simdgroup_f16", 128, 512, 512);
}

criterion_group!(benches, matmul_benchmarks);
criterion_main!(benches);
