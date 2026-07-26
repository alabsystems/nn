// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmarks for Conv1d GPU dispatch — F32 vs F16.
//!
//! Measures DynTensor conv1d throughput at HTDemucs encoder shapes.
//! All four production shapes exceed `MIN_GEMM_FLOPS` (2M) and route
//! through im2col + simdgroup GEMM (not the naive per-element kernel).
//! F16 provides 1.08-1.30x speedup via `simdgroup_multiply_accumulate`
//! 2x ALU throughput.
//!
//! Run: `cargo bench -p nn-metal --bench conv1d --no-default-features`
//!
//! Part of #2981.

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nn_core::dyn_tensor::DynTensor;
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

/// Create a deterministic F32 GPU tensor with given shape.
fn gpu_tensor(shape: &[usize], seed: f32) -> DynTensor {
    let len: usize = shape.iter().product();
    let data: Vec<f32> = (0..len).map(|i| ((i as f32) * seed).sin() * 0.1).collect();
    DynTensor::new(&data, shape, &Device::Cpu)
        .unwrap()
        .to_device(&gpu())
        .unwrap()
}

/// Create a deterministic F16 GPU tensor with given shape.
fn gpu_tensor_f16(shape: &[usize], seed: f32) -> DynTensor {
    let len: usize = shape.iter().product();
    let data: Vec<f32> = (0..len).map(|i| ((i as f32) * seed).sin() * 0.1).collect();
    DynTensor::new(&data, shape, &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap()
        .to_device(&gpu())
        .unwrap()
}

/// Benchmark Conv1d at a specific configuration, both F32 and F16.
fn bench_conv1d(
    c: &mut Criterion,
    label: &str,
    batch: usize,
    in_ch: usize,
    out_ch: usize,
    in_len: usize,
    kernel_size: usize,
    stride: usize,
) {
    ensure_gpu();
    let config = format!("b{batch}_c{in_ch}to{out_ch}_k{kernel_size}_s{stride}_l{in_len}");
    let padding = 0;
    let dilation = 1;
    let groups = 1;

    // F32
    let input_f32 = gpu_tensor(&[batch, in_ch, in_len], 0.01);
    let kernel_f32 = gpu_tensor(&[out_ch, in_ch / groups, kernel_size], 0.02);
    c.bench_with_input(
        BenchmarkId::new(format!("{label}_f32"), &config),
        &(),
        |bencher, ()| {
            bencher.iter(|| {
                let result = black_box(&input_f32)
                    .conv1d(black_box(&kernel_f32), padding, stride, dilation, groups)
                    .unwrap();
                nn_metal::flush().unwrap();
                black_box(result);
            });
        },
    );

    // F16
    let input_f16 = gpu_tensor_f16(&[batch, in_ch, in_len], 0.01);
    let kernel_f16 = gpu_tensor_f16(&[out_ch, in_ch / groups, kernel_size], 0.02);
    c.bench_with_input(
        BenchmarkId::new(format!("{label}_f16"), &config),
        &(),
        |bencher, ()| {
            bencher.iter(|| {
                let result = black_box(&input_f16)
                    .conv1d(black_box(&kernel_f16), padding, stride, dilation, groups)
                    .unwrap();
                nn_metal::flush().unwrap();
                black_box(result);
            });
        },
    );
}

fn conv1d_benchmarks(c: &mut Criterion) {
    // HTDemucs encoder layers (batch=1, audio ~66000 samples at 22050 Hz):
    // Layer 1: 1 -> 48, k=8, s=4
    bench_conv1d(c, "htdemucs_enc1", 1, 1, 48, 66000, 8, 4);
    // Layer 2: 48 -> 96, k=8, s=4 (input length = ~16500)
    bench_conv1d(c, "htdemucs_enc2", 1, 48, 96, 16500, 8, 4);
    // Layer 3: 96 -> 192, k=8, s=4 (input length = ~4125)
    bench_conv1d(c, "htdemucs_enc3", 1, 96, 192, 4125, 8, 4);
    // Layer 4: 192 -> 384, k=8, s=4 (input length = ~1031)
    bench_conv1d(c, "htdemucs_enc4", 1, 192, 384, 1031, 8, 4);
}

criterion_group!(benches, conv1d_benchmarks);
criterion_main!(benches);
