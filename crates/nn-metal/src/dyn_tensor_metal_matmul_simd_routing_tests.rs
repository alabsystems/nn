#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Simdgroup kernel correctness and routing tests (#1567).
//!
//! Extracted from `dyn_tensor_metal_matmul_simd_tests.rs` for file-size compliance.
//! Tests cover simdgroup routing predicate, direct kernel correctness at
//! aligned/transformer-scale/broadcast dimensions.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::dyn_tensor_metal::MetalDynBackend;
use crate::dyn_tensor_metal::{
    select_tile_config, should_use_f16_simdgroup, should_use_simdgroup, GemmTileConfig,
};
use crate::test_common::init;

/// Verify should_use_simdgroup routing predicate.
#[test]
fn test_should_use_simdgroup_routing() {
    // Large aligned: should route to simdgroup
    assert!(should_use_simdgroup(128, 128, 128));
    assert!(should_use_simdgroup(256, 768, 3072));
    assert!(should_use_simdgroup(512, 512, 512));
    assert!(should_use_simdgroup(64, 512, 2048));

    // Non-aligned: should NOT route to simdgroup
    assert!(!should_use_simdgroup(33, 64, 35));
    assert!(!should_use_simdgroup(19, 23, 17));

    // Too small M×N: should NOT route to simdgroup
    assert!(!should_use_simdgroup(8, 256, 8)); // M*N=64

    // K too small: should NOT route to simdgroup
    assert!(!should_use_simdgroup(128, 64, 128)); // K=64 < 128
}

/// Verify should_use_f16_simdgroup occupancy-aware routing predicate (#2981).
///
/// F16 only benefits at high threadgroup counts where ALU throughput is the
/// bottleneck. At low TG counts the GPU is under-saturated and F16 regresses.
#[test]
fn test_should_use_f16_simdgroup_routing() {
    // 256x768x3072 → SMALL, TGs = 8*96 = 768, threshold = 384 → ABOVE → true
    assert!(should_use_f16_simdgroup(256, 768, 3072, 1));

    // 256x3072x768 → SMALL, TGs = 8*24 = 192, threshold = 384 → BELOW → false
    assert!(!should_use_f16_simdgroup(256, 3072, 768, 1));

    // 128x512x512 → SMALL, TGs = 4*16 = 64, threshold = 384 → BELOW → false
    assert!(!should_use_f16_simdgroup(128, 512, 512, 1));

    // Batch multiplies TGs: 128x512x512 batch=8 → SMALL, TGs = 512 → ABOVE 384 → true
    assert!(should_use_f16_simdgroup(128, 512, 512, 8));

    // Not simdgroup-eligible (dims not % 8) → always false
    assert!(!should_use_f16_simdgroup(33, 64, 35, 1));

    // Simdgroup-eligible but M*N too small → false (should_use_simdgroup fails)
    assert!(!should_use_f16_simdgroup(8, 256, 8, 1));

    // batch=0 treated as batch=1
    assert!(should_use_f16_simdgroup(256, 768, 3072, 0));
}

/// Direct simdgroup kernel correctness: [128, 128] × [128, 128].
/// This shape hits the simdgroup path (M*N=16384, K=128, all % 8 == 0).
#[test]
fn test_simdgroup_correctness_128x128() {
    init();
    let (m, k, n) = (128, 128, 128);
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| ((i % 89) as f32 - 44.0) * 0.01)
        .collect();

    let a_cpu = DynTensor::from_vec(a_data.clone(), &[m, k], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::from_vec(b_data.clone(), &[k, n], &Device::Cpu).unwrap();
    let cpu_out = a_cpu.matmul(&b_cpu).unwrap().to_flat_vec::<f32>().unwrap();

    // Call simdgroup path directly
    let a_gpu = DynTensor::from_vec(a_data, &[m, k], &Device::metal()).unwrap();
    let b_gpu = DynTensor::from_vec(b_data, &[k, n], &Device::metal()).unwrap();
    let simd_out = MetalDynBackend::gpu_matmul_simdgroup(&a_gpu, &b_gpu)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    assert_eq!(simd_out.len(), cpu_out.len());
    let max_err = simd_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(&g, &c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 0.01,
        "simdgroup 128x128 max error {max_err} (tol 0.01)"
    );
}

/// Direct simdgroup kernel correctness: AC3 target shape [512, 768] × [768, 3072].
#[test]
fn test_simdgroup_correctness_transformer_ffn() {
    init();
    let (m, k, n) = (512, 768, 3072);
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| ((i % 89) as f32 - 44.0) * 0.001)
        .collect();

    let a_cpu = DynTensor::from_vec(a_data.clone(), &[m, k], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::from_vec(b_data.clone(), &[k, n], &Device::Cpu).unwrap();
    let cpu_out = a_cpu.matmul(&b_cpu).unwrap().to_flat_vec::<f32>().unwrap();

    let a_gpu = DynTensor::from_vec(a_data, &[m, k], &Device::metal()).unwrap();
    let b_gpu = DynTensor::from_vec(b_data, &[k, n], &Device::metal()).unwrap();
    let simd_out = MetalDynBackend::gpu_matmul_simdgroup(&a_gpu, &b_gpu)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    assert_eq!(simd_out.len(), cpu_out.len());
    let max_err = simd_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(&g, &c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 0.1, "simdgroup FFN max error {max_err} (tol 0.1)");
}

/// Simdgroup kernel correctness with broadcast RHS: [4, 64, 512] × [512, 2048].
#[test]
fn test_simdgroup_correctness_broadcast_rhs() {
    init();
    let batch = 4;
    let m = 64;
    let k = 512;
    let n = 2048;
    let a_data: Vec<f32> = (0..batch * m * k)
        .map(|i| ((i % 71) as f32 - 35.0) * 0.001)
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| ((i % 53) as f32 - 26.0) * 0.001)
        .collect();

    let a_cpu = DynTensor::from_vec(a_data.clone(), &[batch, m, k], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::from_vec(b_data.clone(), &[k, n], &Device::Cpu).unwrap();
    let cpu_out = a_cpu.matmul(&b_cpu).unwrap().to_flat_vec::<f32>().unwrap();

    let a_gpu = DynTensor::from_vec(a_data, &[batch, m, k], &Device::metal()).unwrap();
    let b_gpu = DynTensor::from_vec(b_data, &[k, n], &Device::metal()).unwrap();
    let simd_out = MetalDynBackend::gpu_matmul_simdgroup(&a_gpu, &b_gpu)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    assert_eq!(simd_out.len(), cpu_out.len());
    let max_err = simd_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(&g, &c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 0.1,
        "simdgroup broadcast max error {max_err} (tol 0.1)"
    );
}

// -- BF16 simdgroup GEMM tests (#1670) ----------------------------------------

/// Helper: create bf16 GPU tensor from f32 data.
/// Uses the production cpu_to_gpu path (D7/D8) which creates f16 Metal buffers.
fn make_bf16_gpu(data: &[f32], shape: &[usize]) -> DynTensor {
    let cpu = DynTensor::new(data, shape, &Device::Cpu).unwrap();
    let bf16_cpu = cpu.to_dtype(nn_core::DType::BF16).unwrap();
    bf16_cpu.to_device(&Device::metal()).unwrap()
}

/// BF16 simdgroup kernel correctness: [128, 128] × [128, 128].
/// Mixed-precision: half inputs, float accumulators, half output.
/// Tolerance is wider than f32 (0.05 vs 0.01) due to half-precision rounding.
#[test]
fn test_simdgroup_bf16_correctness_128x128() {
    init();
    let (m, k, n) = (128, 128, 128);
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| ((i % 89) as f32 - 44.0) * 0.01)
        .collect();

    // CPU reference (f32 precision)
    let a_cpu = DynTensor::from_vec(a_data.clone(), &[m, k], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::from_vec(b_data.clone(), &[k, n], &Device::Cpu).unwrap();
    let cpu_out = a_cpu.matmul(&b_cpu).unwrap().to_flat_vec::<f32>().unwrap();

    // BF16 GPU simdgroup path
    let a_gpu = make_bf16_gpu(&a_data, &[m, k]);
    let b_gpu = make_bf16_gpu(&b_data, &[k, n]);
    let gpu_result = MetalDynBackend::gpu_matmul_simdgroup(&a_gpu, &b_gpu).unwrap();
    assert_eq!(gpu_result.dtype(), nn_core::DType::BF16);

    let gpu_out = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(nn_core::DType::F32)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    assert_eq!(gpu_out.len(), cpu_out.len());
    let max_err = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(&g, &c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 0.05,
        "bf16 simdgroup 128x128 max error {max_err} (tol 0.05)"
    );
}

/// BF16 simdgroup kernel at transformer FFN scale: [512, 768] × [768, 3072].
/// This is the primary performance target shape for dvoice (Qwen3, Whisper).
#[test]
fn test_simdgroup_bf16_correctness_transformer_ffn() {
    init();
    let (m, k, n) = (512, 768, 3072);
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| ((i % 89) as f32 - 44.0) * 0.001)
        .collect();

    let a_cpu = DynTensor::from_vec(a_data.clone(), &[m, k], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::from_vec(b_data.clone(), &[k, n], &Device::Cpu).unwrap();
    let cpu_out = a_cpu.matmul(&b_cpu).unwrap().to_flat_vec::<f32>().unwrap();

    let a_gpu = make_bf16_gpu(&a_data, &[m, k]);
    let b_gpu = make_bf16_gpu(&b_data, &[k, n]);
    let gpu_result = MetalDynBackend::gpu_matmul_simdgroup(&a_gpu, &b_gpu).unwrap();
    assert_eq!(gpu_result.dtype(), nn_core::DType::BF16);

    let gpu_out = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(nn_core::DType::F32)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    assert_eq!(gpu_out.len(), cpu_out.len());
    let max_err = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(&g, &c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 0.5,
        "bf16 simdgroup FFN max error {max_err} (tol 0.5)"
    );
}

/// BF16 matmul routes through simdgroup for qualifying shapes.
/// Uses the public matmul() API (not direct kernel call) to verify routing.
#[test]
fn test_bf16_matmul_routes_to_simdgroup() {
    init();
    let (m, k, n) = (128, 256, 128);
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| ((i % 67) as f32 - 33.0) * 0.01)
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| ((i % 59) as f32 - 29.0) * 0.01)
        .collect();

    // CPU reference
    let a_cpu = DynTensor::from_vec(a_data.clone(), &[m, k], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::from_vec(b_data.clone(), &[k, n], &Device::Cpu).unwrap();
    let cpu_out = a_cpu.matmul(&b_cpu).unwrap().to_flat_vec::<f32>().unwrap();

    // BF16 GPU via public matmul() API — should route to simdgroup
    let a_gpu = make_bf16_gpu(&a_data, &[m, k]);
    let b_gpu = make_bf16_gpu(&b_data, &[k, n]);
    let gpu_result = a_gpu.matmul(&b_gpu).unwrap();
    assert_eq!(gpu_result.dtype(), nn_core::DType::BF16);

    let gpu_out = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(nn_core::DType::F32)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    assert_eq!(gpu_out.len(), cpu_out.len());
    let max_err = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(&g, &c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 0.1,
        "bf16 matmul (routed) max error {max_err} (tol 0.1)"
    );
}

/// BF16 matmul with broadcast RHS: [4, 64, 512] × [512, 2048].
#[test]
fn test_simdgroup_bf16_broadcast_rhs() {
    init();
    let batch = 4;
    let m = 64;
    let k = 512;
    let n = 2048;
    let a_data: Vec<f32> = (0..batch * m * k)
        .map(|i| ((i % 71) as f32 - 35.0) * 0.001)
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| ((i % 53) as f32 - 26.0) * 0.001)
        .collect();

    let a_cpu = DynTensor::from_vec(a_data.clone(), &[batch, m, k], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::from_vec(b_data.clone(), &[k, n], &Device::Cpu).unwrap();
    let cpu_out = a_cpu.matmul(&b_cpu).unwrap().to_flat_vec::<f32>().unwrap();

    let a_gpu = make_bf16_gpu(&a_data, &[batch, m, k]);
    let b_gpu = make_bf16_gpu(&b_data, &[k, n]);
    let gpu_result = MetalDynBackend::gpu_matmul_simdgroup(&a_gpu, &b_gpu).unwrap();
    assert_eq!(gpu_result.dtype(), nn_core::DType::BF16);

    let gpu_out = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(nn_core::DType::F32)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    assert_eq!(gpu_out.len(), cpu_out.len());
    let max_err = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(&g, &c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 0.5,
        "bf16 simdgroup broadcast max error {max_err} (tol 0.5)"
    );
}

// -- BF16 vs F32 simdgroup performance comparison (#1670 AC3) ------------------

/// AC3: bf16 simdgroup GEMM must be within 1.5x of f32 on the same shape.
///
/// Run with: `cargo test -p nn-metal --lib --release -- test_simdgroup_bf16_vs_f32_perf --nocapture`
#[test]
fn test_simdgroup_bf16_vs_f32_perf() {
    init();
    let (m, k, n) = (512, 768, 3072);
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| ((i % 89) as f32 - 44.0) * 0.001)
        .collect();

    // F32 path
    let a_f32 = DynTensor::from_vec(a_data.clone(), &[m, k], &Device::metal()).unwrap();
    let b_f32 = DynTensor::from_vec(b_data.clone(), &[k, n], &Device::metal()).unwrap();

    // BF16 path
    let a_bf16 = make_bf16_gpu(&a_data, &[m, k]);
    let b_bf16 = make_bf16_gpu(&b_data, &[k, n]);

    // Warmup both paths
    let _ = MetalDynBackend::gpu_matmul_simdgroup(&a_f32, &b_f32).unwrap();
    let _ = MetalDynBackend::gpu_matmul_simdgroup(&a_bf16, &b_bf16).unwrap();

    let iters: u32 = 3;

    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let _ = MetalDynBackend::gpu_matmul_simdgroup(&a_f32, &b_f32).unwrap();
    }
    let f32_ms = t0.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);

    let t1 = std::time::Instant::now();
    for _ in 0..iters {
        let _ = MetalDynBackend::gpu_matmul_simdgroup(&a_bf16, &b_bf16).unwrap();
    }
    let bf16_ms = t1.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);

    let ratio = bf16_ms / f32_ms;
    eprintln!(
        "[#1670 AC3] FFN [512,768]×[768,3072]: f32={f32_ms:.3}ms bf16={bf16_ms:.3}ms ratio={ratio:.2}x"
    );

    // AC3: bf16 must be within 1.5x of f32 performance.
    assert!(
        ratio < 1.5,
        "bf16 simdgroup is {ratio:.2}x slower than f32 (AC3 requires <1.5x)"
    );
}

// -- Adaptive tile selection tests (#3479) ------------------------------------

/// Verify select_tile_config routing (#3479).
///
/// LARGE (64×64, BK=32) selected when M ≥ 64, N ≥ 64, and ≥32 threadgroups.
/// SMALL (32×32, BK=32) for small shapes or insufficient TG count.
#[test]
fn test_select_tile_config_routing() {
    use crate::dyn_tensor_metal::{select_tile_config, GemmTileConfig};

    // -- LARGE: M ≥ 64, N ≥ 64, TGs ≥ 32 --
    // 512×768×3072: TGs = 8×48 = 384
    assert_eq!(select_tile_config(512, 768, 3072, 1), GemmTileConfig::LARGE);
    // 256×512×2048: TGs = 4×32 = 128
    assert_eq!(select_tile_config(256, 512, 2048, 1), GemmTileConfig::LARGE);
    // 128×256×512: TGs = 2×8 = 16 → below threshold
    assert_eq!(select_tile_config(128, 256, 512, 1), GemmTileConfig::SMALL);
    // 256×512×512: TGs = 4×8 = 32 → exactly at threshold
    assert_eq!(select_tile_config(256, 512, 512, 1), GemmTileConfig::LARGE);

    // -- SMALL: M < 64 or N < 64 --
    assert_eq!(select_tile_config(32, 256, 128, 1), GemmTileConfig::SMALL);
    assert_eq!(select_tile_config(128, 256, 32, 1), GemmTileConfig::SMALL);

    // -- SMALL: insufficient TG count --
    // 256×512×256: TGs = 4×4 = 16
    assert_eq!(select_tile_config(256, 512, 256, 1), GemmTileConfig::SMALL);
    // 64×256×64: TGs = 1×1 = 1
    assert_eq!(select_tile_config(64, 256, 64, 1), GemmTileConfig::SMALL);
    // Batch doesn't affect tile selection (only per-slice dimensions).
    assert_eq!(select_tile_config(64, 256, 64, 32), GemmTileConfig::SMALL);
    assert_eq!(select_tile_config(64, 256, 64, 256), GemmTileConfig::SMALL);
}

/// Correctness test for 64×64 BK=32 tile kernel via forced dispatch (#3479).
///
/// Uses `gpu_matmul_simdgroup_forced` to explicitly exercise the LARGE kernel.
/// This shape also routes to LARGE via auto-selection, but forced dispatch
/// isolates the kernel path for debugging.
#[test]
fn test_simdgroup_64x64_correctness_transformer_ffn() {
    use crate::dyn_tensor_metal::GemmTileConfig;
    init();
    let (m, k, n) = (512, 768, 3072);

    let a_data: Vec<f32> = (0..m * k)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| ((i % 89) as f32 - 44.0) * 0.001)
        .collect();

    let a_cpu = DynTensor::from_vec(a_data.clone(), &[m, k], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::from_vec(b_data.clone(), &[k, n], &Device::Cpu).unwrap();
    let cpu_out = a_cpu.matmul(&b_cpu).unwrap().to_flat_vec::<f32>().unwrap();

    let a_gpu = DynTensor::from_vec(a_data, &[m, k], &Device::metal()).unwrap();
    let b_gpu = DynTensor::from_vec(b_data, &[k, n], &Device::metal()).unwrap();
    let simd_out =
        MetalDynBackend::gpu_matmul_simdgroup_forced(&a_gpu, &b_gpu, GemmTileConfig::LARGE)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap();

    assert_eq!(simd_out.len(), cpu_out.len());
    let max_err = simd_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(&g, &c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 0.1, "64x64 FFN max error {max_err} (tol 0.1)");
}

/// Correctness test for 64×64 tile with broadcast RHS (#3479).
#[test]
fn test_simdgroup_64x64_correctness_broadcast_rhs() {
    use crate::dyn_tensor_metal::GemmTileConfig;
    init();
    let (batch, m, k, n) = (4, 128, 512, 2048);

    let a_data: Vec<f32> = (0..batch * m * k)
        .map(|i| ((i % 71) as f32 - 35.0) * 0.001)
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| ((i % 53) as f32 - 26.0) * 0.001)
        .collect();

    let a_cpu = DynTensor::from_vec(a_data.clone(), &[batch, m, k], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::from_vec(b_data.clone(), &[k, n], &Device::Cpu).unwrap();
    let cpu_out = a_cpu.matmul(&b_cpu).unwrap().to_flat_vec::<f32>().unwrap();

    let a_gpu = DynTensor::from_vec(a_data, &[batch, m, k], &Device::metal()).unwrap();
    let b_gpu = DynTensor::from_vec(b_data, &[k, n], &Device::metal()).unwrap();
    let simd_out =
        MetalDynBackend::gpu_matmul_simdgroup_forced(&a_gpu, &b_gpu, GemmTileConfig::LARGE)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap();

    assert_eq!(simd_out.len(), cpu_out.len());
    let max_err = simd_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(&g, &c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 0.1,
        "64x64 broadcast max error {max_err} (tol 0.1)"
    );
}

/// Correctness test for 64×64 tile with edge tiles (M not a multiple of 64).
///
/// M=200 → 4 row tiles, last tile covers rows 192-199 (8 of 64 valid).
/// Exercises the 2-pass edge write path in simd_gemm_64_f32.
/// Routes to LARGE via auto-selection: TGs = ceil(200/64) * ceil(2048/64) = 128.
#[test]
fn test_simdgroup_64x64_correctness_edge_tiles() {
    init();
    let (m, k, n) = (200, 256, 2048);

    let a_data: Vec<f32> = (0..m * k)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| ((i % 89) as f32 - 44.0) * 0.001)
        .collect();

    let a_cpu = DynTensor::from_vec(a_data.clone(), &[m, k], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::from_vec(b_data.clone(), &[k, n], &Device::Cpu).unwrap();
    let cpu_out = a_cpu.matmul(&b_cpu).unwrap().to_flat_vec::<f32>().unwrap();

    // Verify routing selects LARGE for this shape.
    assert_eq!(select_tile_config(m, 0, n, 1), GemmTileConfig::LARGE);

    let a_gpu = DynTensor::from_vec(a_data, &[m, k], &Device::metal()).unwrap();
    let b_gpu = DynTensor::from_vec(b_data, &[k, n], &Device::metal()).unwrap();
    let simd_out = MetalDynBackend::gpu_matmul_simdgroup(&a_gpu, &b_gpu)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    assert_eq!(simd_out.len(), cpu_out.len());
    let max_err = simd_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(&g, &c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 0.1,
        "64x64 edge tile max error {max_err} (tol 0.1)"
    );
}

// -- 64×64 vs 32×32 benchmark (#3479 D2) --------------------------------------

/// A/B performance comparison: 64×64 tile vs 32×32 tile at Kokoro GEMM shapes.
///
/// Tests 3 representative shapes from the Kokoro model:
/// 1. Transformer FFN: 512×768×3072 (largest GEMM, >1M output elements)
/// 2. LSTM hidden projection: 512×512×2048 (LSTM forward gate computation)
/// 3. Linear output projection: 256×768×768 (typical decoder projection)
///
/// Each shape is tested with both tile configs. GPU execution is flushed per
/// iteration to measure actual GPU compute time (not command buffer encoding).
///
/// Run with: `cargo test -p nn-metal --lib --release -- test_simdgroup_64x64_vs_32x32_perf --nocapture`
#[test]
fn test_simdgroup_64x64_vs_32x32_perf() {
    use crate::dyn_tensor_metal::GemmTileConfig;
    init();

    let shapes: &[(usize, usize, usize, &str)] = &[
        (512, 768, 3072, "transformer_ffn"),
        (512, 512, 2048, "lstm_gate_projection"),
        (256, 768, 768, "decoder_output"),
    ];

    let iters: u32 = 5;

    for &(m, k, n, label) in shapes {
        let a_data: Vec<f32> = (0..m * k)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
            .collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| ((i % 89) as f32 - 44.0) * 0.001)
            .collect();

        let a_gpu = DynTensor::from_vec(a_data.clone(), &[m, k], &Device::metal()).unwrap();
        let b_gpu = DynTensor::from_vec(b_data.clone(), &[k, n], &Device::metal()).unwrap();

        // Warmup both tile configs (compile MSL, fill caches).
        let _ = MetalDynBackend::gpu_matmul_simdgroup_forced(&a_gpu, &b_gpu, GemmTileConfig::SMALL)
            .unwrap();
        crate::flush().unwrap();
        let _ = MetalDynBackend::gpu_matmul_simdgroup_forced(&a_gpu, &b_gpu, GemmTileConfig::LARGE)
            .unwrap();
        crate::flush().unwrap();

        // Benchmark 32×32 (flush each iteration to measure GPU compute).
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ =
                MetalDynBackend::gpu_matmul_simdgroup_forced(&a_gpu, &b_gpu, GemmTileConfig::SMALL)
                    .unwrap();
            crate::flush().unwrap();
        }
        let small_ms = t0.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);

        // Benchmark 64×64 (flush each iteration).
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let _ =
                MetalDynBackend::gpu_matmul_simdgroup_forced(&a_gpu, &b_gpu, GemmTileConfig::LARGE)
                    .unwrap();
            crate::flush().unwrap();
        }
        let large_ms = t1.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);

        let speedup = small_ms / large_ms;
        let tg_32 = m.div_ceil(32) * n.div_ceil(32);
        let tg_64 = m.div_ceil(64) * n.div_ceil(64);
        eprintln!(
            "[#3479 D2] {label} [{m},{k}]×[{k},{n}]: 32×32={small_ms:.3}ms ({tg_32} TGs) \
             64×64={large_ms:.3}ms ({tg_64} TGs) speedup={speedup:.2}x"
        );
    }
}
