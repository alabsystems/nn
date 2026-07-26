#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU vs CPU matmul benchmark for #1136.
//!
//! Measures Metal GPU matmul throughput vs CPU (ndarray/Accelerate BLAS) for
//! representative matrix sizes. Reports speedup ratios for comparison against
//! candle-metal's measured 1.8x on M4 Max.
//!
//! Matrix sizes per #1136 AC1:
//! - 128×128 (small — dispatch overhead dominates)
//! - 512×512 (medium — typical intermediate)
//! - 2048×2048 (large — compute-bound)
//! - 512×768 × 768×3072 (transformer FFN — production shape)

#![allow(clippy::cast_lossless)] // benchmark iteration counts: usize→f64 is safe

use std::time::Instant;

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::init;

/// Deterministic data vector with varied values to avoid degenerate cases.
fn data_vec(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + seed) % 97) as f32) * 0.01 - 0.5)
        .collect()
}

/// Benchmark GPU vs CPU matmul for given dimensions.
///
/// Returns `(gpu_ms, cpu_ms, speedup)` averaged over `iters` runs.
/// Includes a warmup pass for GPU pipeline compilation.
fn bench_gpu_vs_cpu(m: usize, k: usize, n: usize, iters: usize) -> (f64, f64, f64) {
    let a_data = data_vec(m * k, 0);
    let b_data = data_vec(k * n, 42);

    let a_gpu =
        DynTensor::from_vec(a_data.clone(), &[m, k], &Device::metal()).expect("GPU tensor A");
    let b_gpu =
        DynTensor::from_vec(b_data.clone(), &[k, n], &Device::metal()).expect("GPU tensor B");
    let a_cpu = DynTensor::from_vec(a_data, &[m, k], &Device::Cpu).expect("CPU tensor A");
    let b_cpu = DynTensor::from_vec(b_data, &[k, n], &Device::Cpu).expect("CPU tensor B");

    // Warmup: compile GPU pipeline + warm CPU caches.
    let _ = a_gpu.matmul(&b_gpu).expect("GPU warmup matmul");
    let _ = a_cpu.matmul(&b_cpu).expect("CPU warmup matmul");

    let iters_f = iters as f64;

    // Benchmark GPU
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = a_gpu.matmul(&b_gpu).expect("GPU matmul");
    }
    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters_f;

    // Benchmark CPU
    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = a_cpu.matmul(&b_cpu).expect("CPU matmul");
    }
    let cpu_ms = t1.elapsed().as_secs_f64() * 1000.0 / iters_f;

    // Correctness: GPU and CPU must agree within tolerance.
    let gpu_out = a_gpu.matmul(&b_gpu).expect("GPU correctness matmul");
    let cpu_out = a_cpu.matmul(&b_cpu).expect("CPU correctness matmul");
    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .expect("GPU→CPU transfer")
        .to_flat_vec::<f32>()
        .expect("GPU flat vec");
    let cpu_vals = cpu_out.to_flat_vec::<f32>().expect("CPU flat vec");
    let max_err = gpu_vals
        .iter()
        .zip(cpu_vals.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    // Tolerance: O(K * epsilon * max_product). Inputs in [-0.5, 0.5], products ≤ 0.25.
    // For K=2048: ~2048 * 1.19e-7 * 0.25 ≈ 6e-5. Use 0.1 with generous safety margin.
    assert!(
        max_err < 0.1,
        "[{m},{k}]×[{k},{n}] GPU vs CPU max error {max_err} (tol 0.1)"
    );

    let speedup = cpu_ms / gpu_ms;
    (gpu_ms, cpu_ms, speedup)
}

/// #1136 AC1: Benchmark GPU vs CPU matmul for representative matrix sizes.
///
/// Reports wall-clock time and GPU/CPU speedup ratio. The CPU baseline uses
/// ndarray's `dot()` which delegates to Accelerate BLAS on macOS.
///
/// #1136 AC2: Compare against candle-metal's measured 1.8x on M4 Max.
/// The IR-based naive kernel should match or exceed candle's throughput
/// for compute-bound sizes (≥512).
#[test]
fn test_matmul_gpu_vs_cpu_benchmark() {
    init();

    let iters = 3;

    let (g1, _c1, s1) = bench_gpu_vs_cpu(128, 128, 128, iters);
    let (g2, _c2, s2) = bench_gpu_vs_cpu(512, 512, 512, iters);
    let (g3, c3, s3) = bench_gpu_vs_cpu(2048, 2048, 2048, iters);
    let (g4, c4, s4) = bench_gpu_vs_cpu(512, 768, 3072, iters);
    let (g5, _c5, s5) = bench_gpu_vs_cpu(1, 768, 768, iters);

    // Print results for manual inspection via `--nocapture`.
    eprintln!("[#1136] 128×128: gpu={g1:.3}ms s={s1:.2}x | 512×512: gpu={g2:.3}ms s={s2:.2}x");
    eprintln!(
        "[#1136] 2048×2048: gpu={g3:.3}ms cpu={c3:.3}ms s={s3:.2}x | FFN: gpu={g4:.3}ms cpu={c4:.3}ms s={s4:.2}x"
    );
    eprintln!("[#1136] decode [1,768]: gpu={g5:.3}ms s={s5:.2}x");

    // Regression guards: detect catastrophic regressions.
    assert!(s1 > 0.05, "[128] GPU catastrophically slow: {s1:.3}x");
    assert!(s2 > 0.05, "[512] GPU catastrophically slow: {s2:.3}x");

    // For large matrices (≥2048), GPU should be faster than CPU.
    // Claimed 3.6-7.3x; guard at 1.5x catches regressions without CI flakiness.
    assert!(
        s3 > 1.5,
        "[2048] GPU should be faster than CPU: {s3:.2}x (expected ≥1.5x)"
    );
    assert!(
        s4 > 1.5,
        "[512,768×3072] GPU should be faster than CPU: {s4:.2}x (expected ≥1.5x)"
    );
}

/// Batched matmul benchmark: [B,M,K] × [B,K,N] — measures batch overhead.
#[test]
fn test_matmul_batched_gpu_vs_cpu() {
    init();

    let batch = 8;
    let m = 64;
    let k = 64;
    let n = 64;
    let iters = 2;
    let iters_f = iters as f64;

    let a_data = data_vec(batch * m * k, 7);
    let b_data = data_vec(batch * k * n, 13);

    let a_gpu =
        DynTensor::from_vec(a_data.clone(), &[batch, m, k], &Device::metal()).expect("GPU batch A");
    let b_gpu =
        DynTensor::from_vec(b_data.clone(), &[batch, k, n], &Device::metal()).expect("GPU batch B");
    let a_cpu = DynTensor::from_vec(a_data, &[batch, m, k], &Device::Cpu).expect("CPU batch A");
    let b_cpu = DynTensor::from_vec(b_data, &[batch, k, n], &Device::Cpu).expect("CPU batch B");

    let _ = a_gpu.matmul(&b_gpu).expect("warmup");
    let _ = a_cpu.matmul(&b_cpu).expect("warmup");

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = a_gpu.matmul(&b_gpu).expect("GPU batched matmul");
    }
    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters_f;

    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = a_cpu.matmul(&b_cpu).expect("CPU batched matmul");
    }
    let cpu_ms = t1.elapsed().as_secs_f64() * 1000.0 / iters_f;

    let speedup = cpu_ms / gpu_ms;
    eprintln!("[#1136] batched [8,64,64]: gpu={gpu_ms:.3}ms cpu={cpu_ms:.3}ms s={speedup:.2}x");

    // Correctness
    let gpu_out = a_gpu.matmul(&b_gpu).expect("correctness");
    let cpu_out = a_cpu.matmul(&b_cpu).expect("correctness");
    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .expect("transfer")
        .to_flat_vec::<f32>()
        .expect("flat vec");
    let cpu_vals = cpu_out.to_flat_vec::<f32>().expect("flat vec");
    let max_err = gpu_vals
        .iter()
        .zip(cpu_vals.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    // Tolerance: K=64, products ≤ 0.25 → ~64 * 1.19e-7 * 0.25 ≈ 2e-6. Use 0.01.
    assert!(
        max_err < 0.01,
        "Batched matmul GPU vs CPU max error {max_err} (tol 0.01)"
    );
}

// -- Simdgroup vs Naive benchmarks (#1518) -------------------------------------

use super::MetalDynBackend;

/// Benchmark: simdgroup vs naive for AC3 target shape [512, 768] × [768, 3072].
///
/// Run with: `cargo test -p nn-metal --lib --release -- test_matmul_simdgroup_vs_naive_bench -- --nocapture`
#[test]
fn test_matmul_simdgroup_vs_naive_bench() {
    init();
    let (m, k, n) = (512, 768, 3072);
    let a_data = data_vec(m * k, 0);
    let b_data = data_vec(k * n, 42);

    let a_gpu = DynTensor::from_vec(a_data.clone(), &[m, k], &Device::metal()).unwrap();
    let b_gpu = DynTensor::from_vec(b_data.clone(), &[k, n], &Device::metal()).unwrap();

    // Warmup both paths
    let _ = MetalDynBackend::gpu_matmul_naive(&a_gpu, &b_gpu).unwrap();
    let _ = MetalDynBackend::gpu_matmul_simdgroup(&a_gpu, &b_gpu).unwrap();

    let iters = 3;

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = MetalDynBackend::gpu_matmul_naive(&a_gpu, &b_gpu).unwrap();
    }
    let naive_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = MetalDynBackend::gpu_matmul_simdgroup(&a_gpu, &b_gpu).unwrap();
    }
    let simd_ms = t1.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let a_cpu = DynTensor::from_vec(a_data, &[m, k], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::from_vec(b_data, &[k, n], &Device::Cpu).unwrap();
    let t2 = Instant::now();
    for _ in 0..2 {
        let _ = a_cpu.matmul(&b_cpu).unwrap();
    }
    let cpu_ms = t2.elapsed().as_secs_f64() * 1000.0 / 2.0;

    let speedup = naive_ms / simd_ms;
    eprintln!(
        "[#1518] FFN [512,768]×[768,3072]: naive={naive_ms:.3}ms simd={simd_ms:.3}ms \
         cpu={cpu_ms:.3}ms simd/naive={speedup:.2}x"
    );

    // Correctness: simdgroup vs naive must agree within tolerance
    let naive_out = MetalDynBackend::gpu_matmul_naive(&a_gpu, &b_gpu)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let simd_out = MetalDynBackend::gpu_matmul_simdgroup(&a_gpu, &b_gpu)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let max_err = naive_out
        .iter()
        .zip(simd_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 0.1,
        "simdgroup vs naive max error {max_err} (tol 0.1)"
    );
}

/// Broadcast matmul benchmark: [B,M,K] × [K,N] — weight sharing pattern.
///
/// This is the `Linear::forward()` pattern where the weight matrix is
/// shared across the batch dimension. Tests broadcast_rhs kernel path.
#[test]
fn test_matmul_broadcast_gpu_vs_cpu() {
    init();

    let batch = 4;
    let m = 256;
    let k = 768;
    let n = 3072;
    let iters = 2;
    let iters_f = iters as f64;

    let a_data = data_vec(batch * m * k, 0);
    let b_data = data_vec(k * n, 42);

    let a_gpu = DynTensor::from_vec(a_data.clone(), &[batch, m, k], &Device::metal())
        .expect("GPU broadcast A");
    let b_gpu =
        DynTensor::from_vec(b_data.clone(), &[k, n], &Device::metal()).expect("GPU broadcast B");
    let a_cpu = DynTensor::from_vec(a_data, &[batch, m, k], &Device::Cpu).expect("CPU broadcast A");
    let b_cpu = DynTensor::from_vec(b_data, &[k, n], &Device::Cpu).expect("CPU broadcast B");

    let _ = a_gpu.matmul(&b_gpu).expect("warmup");
    let _ = a_cpu.matmul(&b_cpu).expect("warmup");

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = a_gpu.matmul(&b_gpu).expect("GPU broadcast matmul");
    }
    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters_f;

    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = a_cpu.matmul(&b_cpu).expect("CPU broadcast matmul");
    }
    let cpu_ms = t1.elapsed().as_secs_f64() * 1000.0 / iters_f;

    let speedup = cpu_ms / gpu_ms;
    eprintln!(
        "[#1136] broadcast [4,256,768]×[768,3072]: gpu={gpu_ms:.2}ms cpu={cpu_ms:.2}ms s={speedup:.2}x"
    );

    // Correctness
    let gpu_out = a_gpu.matmul(&b_gpu).expect("correctness");
    let cpu_out = a_cpu.matmul(&b_cpu).expect("correctness");
    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .expect("transfer")
        .to_flat_vec::<f32>()
        .expect("flat vec");
    let cpu_vals = cpu_out.to_flat_vec::<f32>().expect("flat vec");
    let max_err = gpu_vals
        .iter()
        .zip(cpu_vals.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    // Tolerance: K=768, products ≤ 0.25 → ~768 * 1.19e-7 * 0.25 ≈ 2e-5. Use 0.1.
    assert!(
        max_err < 0.1,
        "Broadcast matmul GPU vs CPU max error {max_err} (tol 0.1)"
    );
}
