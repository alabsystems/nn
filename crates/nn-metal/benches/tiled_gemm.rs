// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmarks for tiled GEMM through the CompiledModel pipeline.
//!
//! Measures throughput for shapes in the tiled tier (between simdgroup and
//! naive) with both F32 and F16 autocast. These shapes are common in
//! attention heads (QK^T, score*V) where K < 128 and M*N < 16384.
//!
//! Run: `cargo bench -p nn-metal --bench tiled_gemm --no-default-features`
//!
//! Part of #2981, #3230. Per design:
//! designs/2026-03-22-f16-gap-analysis-benchmark-methodology.md (D1).

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::mixed_precision::MixedPrecisionPolicy;
use nn_core::{DType, Device};
use nn_metal::compiled_model::CompiledModel;
use nn_metal::PipelineCache;

fn cpu() -> Device {
    Device::Cpu
}

fn ensure_gpu() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = nn_metal::MetalBackend::init().expect("Metal device required for bench");
        nn_metal::register_metal_dyn_backend();
    });
}

fn make_cache() -> PipelineCache {
    PipelineCache::new_global().expect("Metal global cache")
}

/// Deterministic f32 data for weight/input generation.
fn det_f32(len: usize, seed: f32) -> Vec<f32> {
    (0..len).map(|i| ((i as f32) * seed).sin() * 0.1).collect()
}

/// Input trace node helper.
fn input_node(id: u64, shape: &[usize]) -> TraceNode {
    TraceNode::new(
        id,
        format!("input_{id}"),
        TraceOp::Input,
        vec![],
        shape.to_vec(),
        DType::F32,
    )
}

/// Build a compiled single-linear model: [batch, in_f] -> [batch, out_f].
fn build_linear_model(
    cache: &PipelineCache,
    batch: usize,
    in_f: usize,
    out_f: usize,
    autocast: bool,
) -> CompiledModel {
    let w = WeightRef::new(det_f32(out_f * in_f, 0.02), vec![out_f, in_f]).unwrap();
    let b = WeightRef::new(det_f32(out_f, 0.003), vec![out_f]).unwrap();

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, in_f]),
        TraceNode::new(
            1,
            "linear".into(),
            TraceOp::Linear {
                weight: w,
                bias: Some(b),
            },
            vec![0],
            vec![batch, out_f],
            DType::F32,
        ),
    ]);

    if autocast {
        let policy = MixedPrecisionPolicy::apple_silicon_default();
        CompiledModel::builder(&graph, cache)
            .autocast(policy)
            .build()
            .expect("compile autocast")
    } else {
        CompiledModel::builder(&graph, cache)
            .build()
            .expect("compile f32")
    }
}

/// Build a compiled single-matmul model: [m, k] x [k, n] -> [m, n].
fn build_matmul_model(
    cache: &PipelineCache,
    m: usize,
    k: usize,
    n: usize,
    autocast: bool,
) -> CompiledModel {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[m, k]),
        input_node(1, &[k, n]),
        TraceNode::new(
            2,
            "matmul".into(),
            TraceOp::MatMul,
            vec![0, 1],
            vec![m, n],
            DType::F32,
        ),
    ]);

    if autocast {
        let policy = MixedPrecisionPolicy::apple_silicon_default();
        CompiledModel::builder(&graph, cache)
            .autocast(policy)
            .build()
            .expect("compile autocast")
    } else {
        CompiledModel::builder(&graph, cache)
            .build()
            .expect("compile f32")
    }
}

/// Create a GPU DynTensor from deterministic f32 data.
fn gpu_input(shape: &[usize], seed: f32) -> nn_core::dyn_tensor::DynTensor {
    let len: usize = shape.iter().product();
    let data = det_f32(len, seed);
    nn_core::dyn_tensor::DynTensor::new(&data, shape, &cpu())
        .unwrap()
        .to_device(&Device::Metal { device_id: 0 })
        .unwrap()
}

fn bench_tiled_gemm(c: &mut Criterion) {
    ensure_gpu();
    let cache = make_cache();

    // -----------------------------------------------------------------------
    // Tiled-tier shapes from production models (HTDemucs, Whisper, Qwen3).
    // Routing: M>=16, N>=16, K>=8, but below simdgroup thresholds.
    // -----------------------------------------------------------------------

    // Shape 1: HTDemucs attention QK^T (64x64x64).
    // K=64 < 128, M*N=4096 < 16384 → tiled.
    {
        let (m, k, n) = (64, 64, 64);
        let label = format!("matmul_{m}x{k}x{n}");

        let f32_model = build_matmul_model(&cache, m, k, n, false);
        let left = gpu_input(&[m, k], 0.11);
        let right = gpu_input(&[k, n], 0.22);
        c.bench_with_input(BenchmarkId::new(&label, "f32"), &(), |b, ()| {
            b.iter(|| {
                let r = f32_model
                    .execute_dyn(&cache, black_box(&[&left, &right]))
                    .unwrap();
                nn_metal::flush().unwrap();
                black_box(r);
            });
        });

        let f16_model = build_matmul_model(&cache, m, k, n, true);
        c.bench_with_input(BenchmarkId::new(&label, "f16_autocast"), &(), |b, ()| {
            b.iter(|| {
                let r = f16_model
                    .execute_dyn(&cache, black_box(&[&left, &right]))
                    .unwrap();
                nn_metal::flush().unwrap();
                black_box(r);
            });
        });
    }

    // Shape 2: Small attention head (32x64x32).
    {
        let (m, k, n) = (32, 64, 32);
        let label = format!("matmul_{m}x{k}x{n}");

        let f32_model = build_matmul_model(&cache, m, k, n, false);
        let left = gpu_input(&[m, k], 0.33);
        let right = gpu_input(&[k, n], 0.44);
        c.bench_with_input(BenchmarkId::new(&label, "f32"), &(), |b, ()| {
            b.iter(|| {
                let r = f32_model
                    .execute_dyn(&cache, black_box(&[&left, &right]))
                    .unwrap();
                nn_metal::flush().unwrap();
                black_box(r);
            });
        });

        let f16_model = build_matmul_model(&cache, m, k, n, true);
        c.bench_with_input(BenchmarkId::new(&label, "f16_autocast"), &(), |b, ()| {
            b.iter(|| {
                let r = f16_model
                    .execute_dyn(&cache, black_box(&[&left, &right]))
                    .unwrap();
                nn_metal::flush().unwrap();
                black_box(r);
            });
        });
    }

    // Shape 3: Whisper cross-attention (128x64x128). K=64 < 128 → tiled.
    {
        let (m, k, n) = (128, 64, 128);
        let label = format!("matmul_{m}x{k}x{n}");

        let f32_model = build_matmul_model(&cache, m, k, n, false);
        let left = gpu_input(&[m, k], 0.55);
        let right = gpu_input(&[k, n], 0.66);
        c.bench_with_input(BenchmarkId::new(&label, "f32"), &(), |b, ()| {
            b.iter(|| {
                let r = f32_model
                    .execute_dyn(&cache, black_box(&[&left, &right]))
                    .unwrap();
                nn_metal::flush().unwrap();
                black_box(r);
            });
        });

        let f16_model = build_matmul_model(&cache, m, k, n, true);
        c.bench_with_input(BenchmarkId::new(&label, "f16_autocast"), &(), |b, ()| {
            b.iter(|| {
                let r = f16_model
                    .execute_dyn(&cache, black_box(&[&left, &right]))
                    .unwrap();
                nn_metal::flush().unwrap();
                black_box(r);
            });
        });
    }

    // Shape 4: Non-aligned worst case (20x33x19). Exercises tile boundary guards.
    {
        let (m, k, n) = (20, 33, 19);
        let label = format!("matmul_{m}x{k}x{n}");

        let f32_model = build_matmul_model(&cache, m, k, n, false);
        let left = gpu_input(&[m, k], 0.77);
        let right = gpu_input(&[k, n], 0.88);
        c.bench_with_input(BenchmarkId::new(&label, "f32"), &(), |b, ()| {
            b.iter(|| {
                let r = f32_model
                    .execute_dyn(&cache, black_box(&[&left, &right]))
                    .unwrap();
                nn_metal::flush().unwrap();
                black_box(r);
            });
        });

        let f16_model = build_matmul_model(&cache, m, k, n, true);
        c.bench_with_input(BenchmarkId::new(&label, "f16_autocast"), &(), |b, ()| {
            b.iter(|| {
                let r = f16_model
                    .execute_dyn(&cache, black_box(&[&left, &right]))
                    .unwrap();
                nn_metal::flush().unwrap();
                black_box(r);
            });
        });
    }

    // -----------------------------------------------------------------------
    // Naive: too small for tiled (8x8x8). M<16 → naive fallback.
    // Expected: F16 ~1.0x (latency-bound, not bandwidth-bound).
    // -----------------------------------------------------------------------
    {
        let (m, k, n) = (8, 8, 8);
        let label = format!("matmul_{m}x{k}x{n}");

        let f32_model = build_matmul_model(&cache, m, k, n, false);
        let left = gpu_input(&[m, k], 1.11);
        let right = gpu_input(&[k, n], 1.22);
        c.bench_with_input(BenchmarkId::new(&label, "f32"), &(), |b, ()| {
            b.iter(|| {
                let r = f32_model
                    .execute_dyn(&cache, black_box(&[&left, &right]))
                    .unwrap();
                nn_metal::flush().unwrap();
                black_box(r);
            });
        });

        let f16_model = build_matmul_model(&cache, m, k, n, true);
        c.bench_with_input(BenchmarkId::new(&label, "f16_autocast"), &(), |b, ()| {
            b.iter(|| {
                let r = f16_model
                    .execute_dyn(&cache, black_box(&[&left, &right]))
                    .unwrap();
                nn_metal::flush().unwrap();
                black_box(r);
            });
        });
    }

    // -----------------------------------------------------------------------
    // Control: simdgroup-tier shape (256x768x3072) for comparison.
    // All dims % 8 == 0, M*N=786432 > 16384, K=768 > 128 → simdgroup.
    // -----------------------------------------------------------------------
    {
        let (batch, in_f, out_f) = (256, 768, 3072);
        let f32_model = build_linear_model(&cache, batch, in_f, out_f, false);
        let input = gpu_input(&[batch, in_f], 0.99);
        c.bench_with_input(
            BenchmarkId::new("linear_simdgroup", "256x768x3072_f32"),
            &(),
            |b, ()| {
                b.iter(|| {
                    let r = f32_model.execute_dyn(&cache, black_box(&[&input])).unwrap();
                    nn_metal::flush().unwrap();
                    black_box(r);
                });
            },
        );

        let f16_model = build_linear_model(&cache, batch, in_f, out_f, true);
        c.bench_with_input(
            BenchmarkId::new("linear_simdgroup", "256x768x3072_f16"),
            &(),
            |b, ()| {
                b.iter(|| {
                    let r = f16_model.execute_dyn(&cache, black_box(&[&input])).unwrap();
                    nn_metal::flush().unwrap();
                    black_box(r);
                });
            },
        );
    }

    // -----------------------------------------------------------------------
    // Tiled-tier linear: [32, 100] -> [32, 64]. K=100 not %8 → tiled.
    // -----------------------------------------------------------------------
    {
        let (batch, in_f, out_f) = (32, 100, 64);
        let f32_model = build_linear_model(&cache, batch, in_f, out_f, false);
        let input = gpu_input(&[batch, in_f], 1.01);
        c.bench_with_input(
            BenchmarkId::new("linear_tiled", "32x100x64_f32"),
            &(),
            |b, ()| {
                b.iter(|| {
                    let r = f32_model.execute_dyn(&cache, black_box(&[&input])).unwrap();
                    nn_metal::flush().unwrap();
                    black_box(r);
                });
            },
        );

        let f16_model = build_linear_model(&cache, batch, in_f, out_f, true);
        c.bench_with_input(
            BenchmarkId::new("linear_tiled", "32x100x64_f16"),
            &(),
            |b, ()| {
                b.iter(|| {
                    let r = f16_model.execute_dyn(&cache, black_box(&[&input])).unwrap();
                    nn_metal::flush().unwrap();
                    black_box(r);
                });
            },
        );
    }
}

criterion_group!(benches, bench_tiled_gemm);
criterion_main!(benches);
