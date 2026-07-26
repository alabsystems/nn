// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Performance benchmarks with throughput reporting for core DynTensor operations.
//!
//! Run with: `cargo test -p nn-core --test performance_benchmarks -- --nocapture`
//!
//! Unlike the criterion benchmarks (`cargo bench`), these report mean/min/max
//! times and GFLOPS throughput in a human-readable table. Useful for quick
//! one-shot performance snapshots without criterion's statistical machinery.
//!
//! All benchmarks use synthetic random data at tensor shapes drawn from real
//! production models (Whisper, Qwen3, standard transformer blocks).

use std::time::{Duration, Instant};

use nn_core::layers::{Embedding, LayerNorm, Module};
use nn_core::{DType, Device, DynTensor, D};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a deterministic tensor (sin-based, no rand dependency needed).
fn det_tensor(dims: &[usize], seed: f32) -> DynTensor {
    let numel: usize = dims.iter().product();
    let data: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * seed).sin() * 0.5)
        .collect();
    DynTensor::from_vec(data, dims, &Device::Cpu).unwrap()
}

struct BenchResult {
    name: String,
    iterations: usize,
    mean: Duration,
    min: Duration,
    max: Duration,
    gflops: Option<f64>,
}

fn run_bench<F: FnMut()>(
    name: &str,
    iterations: usize,
    flops: Option<u64>,
    mut f: F,
) -> BenchResult {
    // Warmup: 5 iterations
    for _ in 0..5 {
        f();
    }

    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        times.push(start.elapsed());
    }

    let total: Duration = times.iter().sum();
    let mean = total / iterations as u32;
    let min = *times.iter().min().unwrap();
    let max = *times.iter().max().unwrap();

    let gflops = flops.map(|f| f as f64 / mean.as_secs_f64() / 1e9);

    BenchResult {
        name: name.to_string(),
        iterations,
        mean,
        min,
        max,
        gflops,
    }
}

fn print_table(results: &[BenchResult]) {
    println!();
    println!(
        "{:<45} {:>8} {:>12} {:>12} {:>12} {:>10}",
        "Benchmark", "Iters", "Mean", "Min", "Max", "GFLOPS"
    );
    println!("{}", "-".repeat(105));
    for r in results {
        let gflops_str = match r.gflops {
            Some(g) => format!("{g:.2}"),
            None => "-".to_string(),
        };
        println!(
            "{:<45} {:>8} {:>12} {:>12} {:>12} {:>10}",
            r.name,
            r.iterations,
            format_duration(r.mean),
            format_duration(r.min),
            format_duration(r.max),
            gflops_str,
        );
    }
    println!();
}

fn format_duration(d: Duration) -> String {
    let us = d.as_micros();
    if us < 1_000 {
        format!("{us} us")
    } else if us < 1_000_000 {
        format!("{:.2} ms", us as f64 / 1_000.0)
    } else {
        format!("{:.3} s", d.as_secs_f64())
    }
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

const ITERS: usize = 100;

#[test]
fn bench_matmul_whisper() {
    // Whisper encoder: [1, 384, 512] x [512, 512]
    // FLOPs = 2 * B * M * K * N = 2 * 1 * 384 * 512 * 512
    let a = det_tensor(&[1, 384, 512], 0.01);
    let b = det_tensor(&[512, 512], 0.02);
    let flops: u64 = 2 * 384 * 512 * 512;

    let r = run_bench(
        "matmul_whisper [1,384,512]x[512,512]",
        ITERS,
        Some(flops),
        || {
            let _ = black_box(a.matmul(&b).unwrap());
        },
    );
    print_table(&[r]);
}

#[test]
fn bench_matmul_qwen3() {
    // Qwen3 attention decode: [1, 1, 2048] x [2048, 2048]
    // FLOPs = 2 * 1 * 1 * 2048 * 2048
    let a = det_tensor(&[1, 1, 2048], 0.01);
    let b = det_tensor(&[2048, 2048], 0.02);
    let flops: u64 = 2 * 2048 * 2048;

    let r = run_bench(
        "matmul_qwen3 [1,1,2048]x[2048,2048]",
        ITERS,
        Some(flops),
        || {
            let _ = black_box(a.matmul(&b).unwrap());
        },
    );
    print_table(&[r]);
}

#[test]
fn bench_softmax_attention() {
    // Multi-head attention scores: [1, 8, 128, 128]
    // FLOPs ~= 5 * numel (exp + sub + div + max + sum)
    let x = det_tensor(&[1, 8, 128, 128], 0.03);
    let numel: u64 = 8 * 128 * 128;
    let flops = 5 * numel;

    let r = run_bench(
        "softmax_attn [1,8,128,128] dim=-1",
        ITERS,
        Some(flops),
        || {
            let _ = black_box(x.softmax(D::Minus1).unwrap());
        },
    );
    print_table(&[r]);
}

#[test]
fn bench_layer_norm() {
    // Standard transformer: [1, 128, 512]
    let x = det_tensor(&[1, 128, 512], 0.04);
    let dev = Device::Cpu;
    let w = DynTensor::ones(&[512], DType::F32, &dev).unwrap();
    let b = DynTensor::zeros(&[512], DType::F32, &dev).unwrap();
    let ln = LayerNorm::new(w, b, 1e-5).unwrap();

    // FLOPs ~= 5 * numel (mean + var + sub + div + scale+shift)
    let numel: u64 = 128 * 512;
    let flops = 5 * numel;

    let r = run_bench("layer_norm [1,128,512]", ITERS, Some(flops), || {
        let _ = black_box(ln.forward(&x).unwrap());
    });
    print_table(&[r]);
}

#[test]
fn bench_gelu() {
    // FFN intermediate: [1, 128, 2048]
    let x = det_tensor(&[1, 128, 2048], 0.05);
    // GELU FLOPs ~= 8 per element (erf approximation)
    let numel: u64 = 128 * 2048;
    let flops = 8 * numel;

    let r = run_bench("gelu [1,128,2048]", ITERS, Some(flops), || {
        let _ = black_box(x.gelu().unwrap());
    });
    print_table(&[r]);
}

#[test]
fn bench_embedding_lookup() {
    // Whisper: vocab_size=51865, embed_dim=512
    let w = det_tensor(&[51865, 512], 0.001);
    let emb = Embedding::new(w).unwrap();
    let ids: Vec<u32> = (0..384).map(|i| (i * 137) % 51865).collect();
    let ids_t = DynTensor::from_vec_u32(ids, &[1, 384], &Device::Cpu).unwrap();

    // Embedding: seq_len * embed_dim memory reads (not compute-bound)
    let r = run_bench("embedding_whisper [51865,512] seq=384", ITERS, None, || {
        let _ = black_box(emb.forward(&ids_t).unwrap());
    });
    print_table(&[r]);
}

#[test]
fn bench_all_ops_summary() {
    let dev = Device::Cpu;
    let mut results = Vec::new();

    // MatMul: Whisper encoder
    let a = det_tensor(&[1, 384, 512], 0.01);
    let b = det_tensor(&[512, 512], 0.02);
    results.push(run_bench(
        "matmul_whisper [1,384,512]x[512,512]",
        ITERS,
        Some(2 * 384 * 512 * 512),
        || {
            let _ = black_box(a.matmul(&b).unwrap());
        },
    ));

    // MatMul: Qwen3 decode
    let a2 = det_tensor(&[1, 1, 2048], 0.01);
    let b2 = det_tensor(&[2048, 2048], 0.02);
    results.push(run_bench(
        "matmul_qwen3 [1,1,2048]x[2048,2048]",
        ITERS,
        Some(2 * 2048 * 2048),
        || {
            let _ = black_box(a2.matmul(&b2).unwrap());
        },
    ));

    // MatMul: Qwen3 prefill
    let a3 = det_tensor(&[1, 128, 2048], 0.01);
    let b3 = det_tensor(&[2048, 2048], 0.02);
    results.push(run_bench(
        "matmul_qwen3_prefill [1,128,2048]x[2048,2048]",
        ITERS,
        Some(2 * 128 * 2048 * 2048),
        || {
            let _ = black_box(a3.matmul(&b3).unwrap());
        },
    ));

    // Softmax: attention scores
    let sx = det_tensor(&[1, 8, 128, 128], 0.03);
    results.push(run_bench(
        "softmax_attn [1,8,128,128]",
        ITERS,
        Some(5 * 8 * 128 * 128),
        || {
            let _ = black_box(sx.softmax(D::Minus1).unwrap());
        },
    ));

    // LayerNorm
    let lx = det_tensor(&[1, 128, 512], 0.04);
    let w = DynTensor::ones(&[512], DType::F32, &dev).unwrap();
    let bi = DynTensor::zeros(&[512], DType::F32, &dev).unwrap();
    let ln = LayerNorm::new(w, bi, 1e-5).unwrap();
    results.push(run_bench(
        "layer_norm [1,128,512]",
        ITERS,
        Some(5 * 128 * 512),
        || {
            let _ = black_box(ln.forward(&lx).unwrap());
        },
    ));

    // GELU
    let gx = det_tensor(&[1, 128, 2048], 0.05);
    results.push(run_bench(
        "gelu [1,128,2048]",
        ITERS,
        Some(8 * 128 * 2048),
        || {
            let _ = black_box(gx.gelu().unwrap());
        },
    ));

    // Embedding: Whisper
    let ew = det_tensor(&[51865, 512], 0.001);
    let emb = Embedding::new(ew).unwrap();
    let ids: Vec<u32> = (0..384).map(|i| (i * 137) % 51865).collect();
    let ids_t = DynTensor::from_vec_u32(ids, &[1, 384], &dev).unwrap();
    results.push(run_bench(
        "embedding_whisper [51865,512] seq=384",
        ITERS,
        None,
        || {
            let _ = black_box(emb.forward(&ids_t).unwrap());
        },
    ));

    // SiLU (Qwen3 activation)
    let sx2 = det_tensor(&[1, 128, 5504], 0.06);
    results.push(run_bench(
        "silu_qwen3_ffn [1,128,5504]",
        ITERS,
        Some(4 * 128 * 5504),
        || {
            let _ = black_box(sx2.silu().unwrap());
        },
    ));

    println!("\n=== nn-core CPU Performance Summary (real model shapes) ===");
    print_table(&results);
}

/// Prevent compiler from optimizing away the result.
#[inline(never)]
fn black_box<T>(x: T) -> T {
    // Delegate to the standard optimization barrier. The previous hand-rolled
    // `read_volatile(&raw const x)` bit-copied the value while the original was
    // still dropped at scope end — a double-free for any non-Copy `T` (e.g.
    // DynTensor), which aborted these benches with SIGABRT/SIGTRAP.
    std::hint::black_box(x)
}
