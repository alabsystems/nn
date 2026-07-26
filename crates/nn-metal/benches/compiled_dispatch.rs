// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmarks for CompiledModel dispatch.
//!
//! Traces a small MLP (Linear -> ReLU -> Linear), compiles to a
//! CompiledModel, then benchmarks the GPU execute path. Measures
//! end-to-end compiled dispatch latency including buffer allocation,
//! kernel dispatch, and flush.
//!
//! Run: `cargo bench -p nn-metal --bench compiled_dispatch --no-default-features`
//!
//! Part of #3218.

use std::collections::HashMap;

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::mixed_precision::MixedPrecisionPolicy;
use nn_core::layers::{Linear, Module};
use nn_core::{DType, Device, VarBuilder};
use nn_dsl::trace_compile::compile_trace_to_plan_with_fusion;
use nn_metal::compiled_model::CompiledModel;
use nn_metal::PipelineCache;

fn cpu() -> Device {
    Device::Cpu
}

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

fn make_cache() -> PipelineCache {
    PipelineCache::new_global().expect("Metal global cache")
}

/// Build a Linear layer from deterministic data.
fn build_linear(rows: usize, cols: usize, seed: f64) -> Linear {
    let mut m = HashMap::new();
    m.insert(
        "weight".to_string(),
        DynTensor::full(&[rows, cols], seed, DType::F32, &cpu()).unwrap(),
    );
    m.insert(
        "bias".to_string(),
        DynTensor::full(&[rows], seed * 0.1, DType::F32, &cpu()).unwrap(),
    );
    let vb = VarBuilder::from_tensors(m, DType::F32, &cpu());
    let w = vb.get(&[rows, cols], "weight").unwrap();
    let b = vb.get(&[rows], "bias").unwrap();
    Linear::new(w, Some(b)).unwrap()
}

/// Build a compiled MLP: Linear(d_in, d_hidden) -> ReLU -> Linear(d_hidden, d_out).
fn build_compiled_mlp(
    cache: &PipelineCache,
    batch: usize,
    d_in: usize,
    d_hidden: usize,
    d_out: usize,
) -> (CompiledModel, DynTensor) {
    let l1 = build_linear(d_hidden, d_in, 0.02);
    let l2 = build_linear(d_out, d_hidden, 0.03);

    let input = DynTensor::full(&[batch, d_in], 0.5, DType::F32, &cpu()).unwrap();

    let (_traced, graph) = trace_graph(|| {
        let mut x = input.clone();
        x.set_trace_id(record_input(x.dims(), DType::F32).unwrap());
        let h = l1.forward(&x)?;
        let h = h.relu()?;
        l2.forward(&h)
    })
    .expect("trace_graph");

    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    let model = CompiledModel::from_plan(&plan, &graph, cache).expect("from_plan");
    let gpu_input = input.to_device(&gpu()).unwrap();
    (model, gpu_input)
}

/// Build autocast compiled MLP using builder API with per-op autocast.
fn build_compiled_mlp_autocast(
    cache: &PipelineCache,
    batch: usize,
    d_in: usize,
    d_hidden: usize,
    d_out: usize,
) -> (CompiledModel, DynTensor) {
    let l1 = build_linear(d_hidden, d_in, 0.02);
    let l2 = build_linear(d_out, d_hidden, 0.03);
    let input = DynTensor::full(&[batch, d_in], 0.5, DType::F32, &cpu()).unwrap();

    let (_traced, graph) = trace_graph(|| {
        let mut x = input.clone();
        x.set_trace_id(record_input(x.dims(), DType::F32).unwrap());
        let h = l1.forward(&x)?;
        let h = h.relu()?;
        l2.forward(&h)
    })
    .expect("trace_graph");

    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let model = CompiledModel::builder(&graph, cache)
        .autocast(policy)
        .build()
        .expect("builder autocast");
    let gpu_input = input.to_device(&gpu()).unwrap();
    (model, gpu_input)
}

fn bench_compiled_dispatch(c: &mut Criterion) {
    ensure_gpu();
    let cache = make_cache();

    // Small MLP: representative of a single transformer sub-block.
    let (model_small, input_small) = build_compiled_mlp(&cache, 32, 128, 512, 128);
    c.bench_with_input(
        BenchmarkId::new("compiled_mlp", "32x128_512_128"),
        &(),
        |bencher, ()| {
            bencher.iter(|| {
                let result = model_small
                    .execute_dyn(&cache, black_box(&[&input_small]))
                    .unwrap();
                nn_metal::flush().unwrap();
                black_box(result);
            });
        },
    );

    // Larger MLP: closer to Kokoro decoder block dimensions.
    let (model_large, input_large) = build_compiled_mlp(&cache, 256, 512, 2048, 512);
    c.bench_with_input(
        BenchmarkId::new("compiled_mlp", "256x512_2048_512"),
        &(),
        |bencher, ()| {
            bencher.iter(|| {
                let result = model_large
                    .execute_dyn(&cache, black_box(&[&input_large]))
                    .unwrap();
                nn_metal::flush().unwrap();
                black_box(result);
            });
        },
    );

    // Autocast MLP: same dimensions as large F32 for direct comparison.
    // Measured M4 Max: ~1.34x (625µs autocast vs 837µs F32). Hybrid GEMM
    // (bc1b959) uses half×half MACs with float accumulator — gets 2x ALU
    // throughput from Apple Silicon mixed-precision path. Previous mixed GEMM
    // promoted F16→F32 before MACs, giving 0.92-0.97x (no speedup).
    let (model_ac, input_ac) = build_compiled_mlp_autocast(&cache, 256, 512, 2048, 512);
    c.bench_with_input(
        BenchmarkId::new("compiled_mlp_autocast", "256x512_2048_512"),
        &(),
        |bencher, ()| {
            bencher.iter(|| {
                let result = model_ac
                    .execute_dyn(&cache, black_box(&[&input_ac]))
                    .unwrap();
                nn_metal::flush().unwrap();
                black_box(result);
            });
        },
    );
}

criterion_group!(benches, bench_compiled_dispatch);
criterion_main!(benches);
