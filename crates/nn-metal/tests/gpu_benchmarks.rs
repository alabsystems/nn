// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU performance benchmarks for Metal backend operations.
//!
//! Run with: `cargo test -p nn-metal --test gpu_benchmarks -- --nocapture`
//!
//! Measures GPU dispatch + execution time for core operations at real-model
//! tensor shapes. Each benchmark flushes the GPU command buffer to measure
//! actual completion time (not just dispatch latency).
//!
//! Reports mean/min/max time and GFLOPS throughput where applicable.

use std::time::{Duration, Instant};

use nn_core::layers::{LayerNorm, Module};
use nn_core::{DType, Device, DynTensor, D};

// ---------------------------------------------------------------------------
// GPU helpers
// ---------------------------------------------------------------------------

fn gpu() -> Device {
    Device::Metal { device_id: 0 }
}

fn ensure_gpu() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = nn_metal::MetalBackend::init().expect("Metal device required for GPU benchmarks");
        nn_metal::register_metal_dyn_backend();
    });
}

/// Create a deterministic GPU tensor with given shape.
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

// ---------------------------------------------------------------------------
// Benchmark infrastructure
// ---------------------------------------------------------------------------

struct BenchResult {
    name: String,
    iterations: usize,
    mean: Duration,
    min: Duration,
    max: Duration,
    gflops: Option<f64>,
}

fn run_gpu_bench<F: FnMut()>(
    name: &str,
    iterations: usize,
    flops: Option<u64>,
    mut f: F,
) -> BenchResult {
    // Warmup: 10 iterations (GPU pipelines need more warmup)
    for _ in 0..10 {
        f();
        nn_metal::flush().unwrap();
    }

    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        nn_metal::flush().unwrap();
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
        "{:<50} {:>6} {:>12} {:>12} {:>12} {:>10}",
        "GPU Benchmark", "Iters", "Mean", "Min", "Max", "GFLOPS"
    );
    println!("{}", "-".repeat(108));
    for r in results {
        let gflops_str = match r.gflops {
            Some(g) => format!("{g:.2}"),
            None => "-".to_string(),
        };
        println!(
            "{:<50} {:>6} {:>12} {:>12} {:>12} {:>10}",
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

/// Prevent compiler from optimizing away the result.
#[inline(never)]
fn black_box<T>(x: T) -> T {
    let ptr = &raw const x;
    unsafe { std::ptr::read_volatile(ptr) }
}

const GPU_ITERS: usize = 100;

// ---------------------------------------------------------------------------
// GPU MatMul benchmarks
// ---------------------------------------------------------------------------

#[test]
fn gpu_bench_matmul() {
    ensure_gpu();
    let mut results = Vec::new();

    // Whisper encoder: [1, 384, 512] x [512, 512]
    let a = gpu_tensor(&[1, 384, 512], 0.01);
    let b = gpu_tensor(&[512, 512], 0.02);
    results.push(run_gpu_bench(
        "matmul_whisper [1,384,512]x[512,512]",
        GPU_ITERS,
        Some(2 * 384 * 512 * 512),
        || {
            let _ = black_box(a.matmul(&b).unwrap());
        },
    ));

    // Qwen3 decode: [1, 1, 2048] x [2048, 2048]
    let a2 = gpu_tensor(&[1, 1, 2048], 0.01);
    let b2 = gpu_tensor(&[2048, 2048], 0.02);
    results.push(run_gpu_bench(
        "matmul_qwen3_decode [1,1,2048]x[2048,2048]",
        GPU_ITERS,
        Some(2 * 2048 * 2048),
        || {
            let _ = black_box(a2.matmul(&b2).unwrap());
        },
    ));

    // Qwen3 prefill: [1, 128, 2048] x [2048, 2048]
    let a3 = gpu_tensor(&[1, 128, 2048], 0.01);
    let b3 = gpu_tensor(&[2048, 2048], 0.02);
    results.push(run_gpu_bench(
        "matmul_qwen3_prefill [1,128,2048]x[2048,2048]",
        GPU_ITERS,
        Some(2 * 128 * 2048 * 2048),
        || {
            let _ = black_box(a3.matmul(&b3).unwrap());
        },
    ));

    // Kokoro FFN: [1, 256, 768] x [768, 3072]
    let a4 = gpu_tensor(&[1, 256, 768], 0.01);
    let b4 = gpu_tensor(&[768, 3072], 0.02);
    results.push(run_gpu_bench(
        "matmul_kokoro_ffn [1,256,768]x[768,3072]",
        GPU_ITERS,
        Some(2 * 256 * 768 * 3072),
        || {
            let _ = black_box(a4.matmul(&b4).unwrap());
        },
    ));

    println!("\n=== GPU MatMul Benchmarks ===");
    print_table(&results);
}

// ---------------------------------------------------------------------------
// GPU Softmax benchmarks
// ---------------------------------------------------------------------------

#[test]
fn gpu_bench_softmax() {
    ensure_gpu();
    let mut results = Vec::new();

    // 8-head attention: [1, 8, 128, 128]
    let x1 = gpu_tensor(&[1, 8, 128, 128], 0.03);
    results.push(run_gpu_bench(
        "softmax_8h_seq128 [1,8,128,128]",
        GPU_ITERS,
        Some(5 * 8 * 128 * 128),
        || {
            let _ = black_box(x1.softmax(D::Minus1).unwrap());
        },
    ));

    // Whisper 6-head: [1, 6, 384, 384]
    let x2 = gpu_tensor(&[1, 6, 384, 384], 0.03);
    results.push(run_gpu_bench(
        "softmax_whisper_6h [1,6,384,384]",
        GPU_ITERS,
        Some(5 * 6 * 384 * 384),
        || {
            let _ = black_box(x2.softmax(D::Minus1).unwrap());
        },
    ));

    // Qwen3 GQA decode: [1, 32, 1, 128]
    let x3 = gpu_tensor(&[1, 32, 1, 128], 0.03);
    results.push(run_gpu_bench(
        "softmax_qwen3_gqa [1,32,1,128]",
        GPU_ITERS,
        Some((5 * 32) * 128),
        || {
            let _ = black_box(x3.softmax(D::Minus1).unwrap());
        },
    ));

    println!("\n=== GPU Softmax Benchmarks ===");
    print_table(&results);
}

// ---------------------------------------------------------------------------
// GPU LayerNorm benchmarks
// ---------------------------------------------------------------------------

#[test]
fn gpu_bench_layer_norm() {
    ensure_gpu();
    let dev = Device::Cpu;
    let mut results = Vec::new();

    // Standard: [1, 128, 512]
    let x1 = gpu_tensor(&[1, 128, 512], 0.04);
    let w1 = DynTensor::ones(&[512], DType::F32, &dev)
        .unwrap()
        .to_device(&gpu())
        .unwrap();
    let b1 = DynTensor::zeros(&[512], DType::F32, &dev)
        .unwrap()
        .to_device(&gpu())
        .unwrap();
    let ln1 = LayerNorm::new(w1, b1, 1e-5).unwrap();
    results.push(run_gpu_bench(
        "layer_norm [1,128,512]",
        GPU_ITERS,
        Some(5 * 128 * 512),
        || {
            let _ = black_box(ln1.forward(&x1).unwrap());
        },
    ));

    // Qwen3: [1, 128, 2048]
    let x2 = gpu_tensor(&[1, 128, 2048], 0.04);
    let w2 = DynTensor::ones(&[2048], DType::F32, &dev)
        .unwrap()
        .to_device(&gpu())
        .unwrap();
    let b2 = DynTensor::zeros(&[2048], DType::F32, &dev)
        .unwrap()
        .to_device(&gpu())
        .unwrap();
    let ln2 = LayerNorm::new(w2, b2, 1e-5).unwrap();
    results.push(run_gpu_bench(
        "layer_norm_qwen3 [1,128,2048]",
        GPU_ITERS,
        Some(5 * 128 * 2048),
        || {
            let _ = black_box(ln2.forward(&x2).unwrap());
        },
    ));

    // Whisper: [1, 384, 512]
    let x3 = gpu_tensor(&[1, 384, 512], 0.04);
    let w3 = DynTensor::ones(&[512], DType::F32, &dev)
        .unwrap()
        .to_device(&gpu())
        .unwrap();
    let b3 = DynTensor::zeros(&[512], DType::F32, &dev)
        .unwrap()
        .to_device(&gpu())
        .unwrap();
    let ln3 = LayerNorm::new(w3, b3, 1e-5).unwrap();
    results.push(run_gpu_bench(
        "layer_norm_whisper [1,384,512]",
        GPU_ITERS,
        Some(5 * 384 * 512),
        || {
            let _ = black_box(ln3.forward(&x3).unwrap());
        },
    ));

    println!("\n=== GPU LayerNorm Benchmarks ===");
    print_table(&results);
}

// ---------------------------------------------------------------------------
// CPU-to-GPU transfer overhead
// ---------------------------------------------------------------------------

#[test]
fn gpu_bench_transfer_overhead() {
    ensure_gpu();
    let mut results = Vec::new();

    // Small tensor: [1, 128, 512] = 65536 floats = 256 KB
    let small = DynTensor::new(&vec![0.1f32; 128 * 512], &[1, 128, 512], &Device::Cpu).unwrap();
    results.push(run_gpu_bench(
        "transfer_cpu_to_gpu 256KB [1,128,512]",
        GPU_ITERS,
        None,
        || {
            let _ = black_box(small.to_device(&gpu()).unwrap());
        },
    ));

    // Medium tensor: [1, 384, 512] = 196608 floats = 768 KB
    let medium =
        DynTensor::new(&vec![0.1f32; 384 * 512], &[1, 384, 512], &Device::Cpu).unwrap();
    results.push(run_gpu_bench(
        "transfer_cpu_to_gpu 768KB [1,384,512]",
        GPU_ITERS,
        None,
        || {
            let _ = black_box(medium.to_device(&gpu()).unwrap());
        },
    ));

    // Large tensor: [1, 128, 2048] = 262144 floats = 1 MB
    let large =
        DynTensor::new(&vec![0.1f32; 128 * 2048], &[1, 128, 2048], &Device::Cpu).unwrap();
    results.push(run_gpu_bench(
        "transfer_cpu_to_gpu 1MB [1,128,2048]",
        GPU_ITERS,
        None,
        || {
            let _ = black_box(large.to_device(&gpu()).unwrap());
        },
    ));

    // Weight matrix: [2048, 2048] = 4194304 floats = 16 MB
    let weight = DynTensor::new(&vec![0.1f32; 2048 * 2048], &[2048, 2048], &Device::Cpu).unwrap();
    results.push(run_gpu_bench(
        "transfer_cpu_to_gpu 16MB [2048,2048]",
        GPU_ITERS,
        None,
        || {
            let _ = black_box(weight.to_device(&gpu()).unwrap());
        },
    ));

    // GPU-to-CPU readback: [1, 128, 512]
    let on_gpu = small.to_device(&gpu()).unwrap();
    results.push(run_gpu_bench(
        "transfer_gpu_to_cpu 256KB [1,128,512]",
        GPU_ITERS,
        None,
        || {
            nn_metal::flush().unwrap();
            let _ = black_box(on_gpu.to_device(&Device::Cpu).unwrap());
        },
    ));

    println!("\n=== CPU<->GPU Transfer Overhead ===");
    print_table(&results);
}

// ---------------------------------------------------------------------------
// Full GPU summary
// ---------------------------------------------------------------------------

#[test]
fn gpu_bench_all_summary() {
    ensure_gpu();
    let dev = Device::Cpu;
    let mut results = Vec::new();

    // MatMul: Whisper
    let a = gpu_tensor(&[1, 384, 512], 0.01);
    let b = gpu_tensor(&[512, 512], 0.02);
    results.push(run_gpu_bench(
        "matmul_whisper [1,384,512]x[512,512]",
        GPU_ITERS,
        Some(2 * 384 * 512 * 512),
        || {
            let _ = black_box(a.matmul(&b).unwrap());
        },
    ));

    // MatMul: Qwen3 prefill
    let a2 = gpu_tensor(&[1, 128, 2048], 0.01);
    let b2 = gpu_tensor(&[2048, 2048], 0.02);
    results.push(run_gpu_bench(
        "matmul_qwen3_prefill [1,128,2048]x[2048,2048]",
        GPU_ITERS,
        Some(2 * 128 * 2048 * 2048),
        || {
            let _ = black_box(a2.matmul(&b2).unwrap());
        },
    ));

    // Softmax: attention
    let sx = gpu_tensor(&[1, 8, 128, 128], 0.03);
    results.push(run_gpu_bench(
        "softmax_attn [1,8,128,128]",
        GPU_ITERS,
        Some(5 * 8 * 128 * 128),
        || {
            let _ = black_box(sx.softmax(D::Minus1).unwrap());
        },
    ));

    // LayerNorm
    let lx = gpu_tensor(&[1, 128, 512], 0.04);
    let w = DynTensor::ones(&[512], DType::F32, &dev)
        .unwrap()
        .to_device(&gpu())
        .unwrap();
    let bi = DynTensor::zeros(&[512], DType::F32, &dev)
        .unwrap()
        .to_device(&gpu())
        .unwrap();
    let ln = LayerNorm::new(w, bi, 1e-5).unwrap();
    results.push(run_gpu_bench(
        "layer_norm [1,128,512]",
        GPU_ITERS,
        Some(5 * 128 * 512),
        || {
            let _ = black_box(ln.forward(&lx).unwrap());
        },
    ));

    // GELU
    let gx = gpu_tensor(&[1, 128, 2048], 0.05);
    results.push(run_gpu_bench(
        "gelu [1,128,2048]",
        GPU_ITERS,
        Some(8 * 128 * 2048),
        || {
            let _ = black_box(gx.gelu().unwrap());
        },
    ));

    // Transfer: CPU -> GPU 256KB
    let small = DynTensor::new(&vec![0.1f32; 128 * 512], &[1, 128, 512], &Device::Cpu).unwrap();
    results.push(run_gpu_bench(
        "transfer_cpu_to_gpu 256KB",
        GPU_ITERS,
        None,
        || {
            let _ = black_box(small.to_device(&gpu()).unwrap());
        },
    ));

    println!("\n=== Metal GPU Performance Summary (real model shapes) ===");
    print_table(&results);
}
