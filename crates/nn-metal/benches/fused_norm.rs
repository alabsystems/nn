// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmarks for fused norm GPU kernels at Kokoro shapes.
//!
//! Measures GroupNorm, RmsNorm, and Snake at the specific tensor shapes
//! used in Kokoro's pipeline to determine if fused single-dispatch kernels
//! are slower than the old multi-dispatch decomposition for small channels.
//!
//! Run: `cargo bench -p nn-metal --bench fused_norm`
//!
//! Part of #3348 (D2: fused MSL kernel shape-specific benchmark).

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::Module;
use nn_core::{DType, Device};

fn gpu() -> Device {
    Device::Metal { device_id: 0 }
}

fn ensure_gpu() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = nn_metal::MetalBackend::init().expect("Metal device required for bench");
        nn_metal::register_metal_dyn_backend();
    });
}

fn gpu_tensor(shape: &[usize], seed: f32) -> DynTensor {
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * seed).sin() * 0.5)
        .collect();
    DynTensor::new(&data, shape, &Device::Cpu)
        .unwrap()
        .to_device(&gpu())
        .unwrap()
}

fn gpu_ones(shape: &[usize]) -> DynTensor {
    DynTensor::ones(shape, DType::F32, &Device::Cpu)
        .unwrap()
        .to_device(&gpu())
        .unwrap()
}

fn gpu_zeros(shape: &[usize]) -> DynTensor {
    DynTensor::zeros(shape, DType::F32, &Device::Cpu)
        .unwrap()
        .to_device(&gpu())
        .unwrap()
}

/// Kokoro GroupNorm shapes: [1, channels, T] with num_groups = channels.
/// channels ∈ {48, 96, 192, 384, 512}, T ∈ {50, 200}.
fn bench_group_norm(c: &mut Criterion) {
    ensure_gpu();
    let mut group = c.benchmark_group("group_norm");
    group.sample_size(50);

    for &channels in &[48usize, 96, 192, 384, 512] {
        for &t in &[50usize, 200] {
            let shape = [1, channels, t];
            let label = format!("ch{channels}_t{t}");

            let x = gpu_tensor(&shape, 0.7);
            let weight = gpu_ones(&[channels]);
            let bias = gpu_zeros(&[channels]);
            let eps = 1e-5;
            let num_groups = channels;

            let gn = nn_core::layers::GroupNorm::new(num_groups, channels, weight, bias, eps).unwrap();

            group.bench_with_input(BenchmarkId::new("fused", &label), &(), |b, ()| {
                b.iter(|| {
                    black_box(gn.forward(&x).unwrap());
                });
            });
        }
    }
    group.finish();
}

/// Kokoro RmsNorm shapes: [1, seq_len, dim] with dim ∈ {256, 512}.
fn bench_rms_norm(c: &mut Criterion) {
    ensure_gpu();
    let mut group = c.benchmark_group("rms_norm");
    group.sample_size(50);

    for &dim in &[256usize, 512] {
        for &seq in &[8usize, 32, 128] {
            let shape = [1, seq, dim];
            let label = format!("d{dim}_s{seq}");

            let x = gpu_tensor(&shape, 0.3);
            let weight = gpu_ones(&[dim]);
            let eps = 1e-5;

            let rn = nn_core::layers::RmsNorm::new(weight, eps).unwrap();

            group.bench_with_input(BenchmarkId::new("fused", &label), &(), |b, ()| {
                b.iter(|| {
                    black_box(rn.forward(&x).unwrap());
                });
            });
        }
    }
    group.finish();
}

/// Kokoro Snake shapes: [1, channels, T] with alpha shape [channels].
fn bench_snake(c: &mut Criterion) {
    ensure_gpu();
    let mut group = c.benchmark_group("snake");
    group.sample_size(50);

    for &channels in &[48usize, 96, 192, 384, 512] {
        for &t in &[50usize, 200] {
            let shape = [1, channels, t];
            let label = format!("ch{channels}_t{t}");

            let x = gpu_tensor(&shape, 0.5);
            let alpha = gpu_ones(&[channels]);

            group.bench_with_input(BenchmarkId::new("fused", &label), &(), |b, ()| {
                b.iter(|| {
                    black_box(x.snake_tensor(&alpha).unwrap());
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_group_norm, bench_rms_norm, bench_snake);
criterion_main!(benches);
