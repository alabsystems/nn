#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended matmul GPU correctness tests at various matrix sizes.
//!
//! Extracted from `dyn_tensor_metal_matmul_tests.rs` (#1377).
//! Covers GPU-vs-CPU matmul correctness at tile-aligned, non-aligned,
//! large, batched (3D), multi-head (4D), and transformer-scale dimensions.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::init;

// -- Matmul correctness tests at various sizes (#1289) ------------------------

#[test]
fn test_matmul_32x32() {
    // 32×32 matmul — verifies GPU-vs-CPU correctness at aligned dimensions.
    init();
    let m = 32;
    let k = 32;
    let n = 32;
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();

    let a_gpu = DynTensor::from_vec(a_data.clone(), &[m, k], &Device::metal()).unwrap();
    let b_gpu = DynTensor::from_vec(b_data.clone(), &[k, n], &Device::metal()).unwrap();
    let gpu_out = a_gpu.matmul(&b_gpu).unwrap();
    assert_eq!(gpu_out.dims(), &[m, n]);

    let a_cpu = DynTensor::from_vec(a_data, &[m, k], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::from_vec(b_data, &[k, n], &Device::Cpu).unwrap();
    let cpu_out = a_cpu.matmul(&b_cpu).unwrap();

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();
    assert_eq!(gpu_vals.len(), cpu_vals.len());
    for (i, (g, c)) in gpu_vals.iter().zip(cpu_vals.iter()).enumerate() {
        assert!(
            (g - c).abs() < 1e-2,
            "32x32 mismatch at [{i}]: gpu={g}, cpu={c}"
        );
    }
}

#[test]
fn test_matmul_non_aligned_19x17() {
    // 19×23 @ 23×17 — non-aligned dimensions exercise boundary handling.
    init();
    let m = 19;
    let k = 23;
    let n = 17;
    let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) * 0.1).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32) * 0.1).collect();

    let a_gpu = DynTensor::from_vec(a_data.clone(), &[m, k], &Device::metal()).unwrap();
    let b_gpu = DynTensor::from_vec(b_data.clone(), &[k, n], &Device::metal()).unwrap();
    let gpu_out = a_gpu.matmul(&b_gpu).unwrap();
    assert_eq!(gpu_out.dims(), &[m, n]);

    let a_cpu = DynTensor::from_vec(a_data, &[m, k], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::from_vec(b_data, &[k, n], &Device::Cpu).unwrap();
    let cpu_out = a_cpu.matmul(&b_cpu).unwrap();

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();
    assert_eq!(gpu_vals.len(), cpu_vals.len());
    for (i, (g, c)) in gpu_vals.iter().zip(cpu_vals.iter()).enumerate() {
        assert!(
            (g - c).abs() < 1e-2,
            "19x17 mismatch at [{i}]: gpu={g}, cpu={c}"
        );
    }
}

#[test]
fn test_matmul_large_128x64() {
    // 128×64 @ 64×128 — representative model-scale matmul (e.g. Linear(64, 128)).
    init();
    let m = 128;
    let k = 64;
    let n = 128;
    let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32) * 0.01).collect();

    let a_gpu = DynTensor::from_vec(a_data.clone(), &[m, k], &Device::metal()).unwrap();
    let b_gpu = DynTensor::from_vec(b_data.clone(), &[k, n], &Device::metal()).unwrap();
    let gpu_out = a_gpu.matmul(&b_gpu).unwrap();
    assert_eq!(gpu_out.dims(), &[m, n]);

    let a_cpu = DynTensor::from_vec(a_data, &[m, k], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::from_vec(b_data, &[k, n], &Device::Cpu).unwrap();
    let cpu_out = a_cpu.matmul(&b_cpu).unwrap();

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();
    assert_eq!(gpu_vals.len(), cpu_vals.len());
    let max_err = gpu_vals
        .iter()
        .zip(cpu_vals.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 0.05,
        "128x64 max error {max_err} exceeds tolerance"
    );
}

#[test]
fn test_matmul_batched_3d() {
    // [4, 16, 32] @ [4, 32, 16] — batched matmul.
    init();
    let batch = 4;
    let m = 16;
    let k = 32;
    let n = 16;
    let a_data: Vec<f32> = (0..batch * m * k).map(|i| ((i % 9) as f32) * 0.1).collect();
    let b_data: Vec<f32> = (0..batch * k * n).map(|i| ((i % 7) as f32) * 0.1).collect();

    let a_gpu = DynTensor::from_vec(a_data.clone(), &[batch, m, k], &Device::metal()).unwrap();
    let b_gpu = DynTensor::from_vec(b_data.clone(), &[batch, k, n], &Device::metal()).unwrap();
    let gpu_out = a_gpu.matmul(&b_gpu).unwrap();
    assert_eq!(gpu_out.dims(), &[batch, m, n]);

    let a_cpu = DynTensor::from_vec(a_data, &[batch, m, k], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::from_vec(b_data, &[batch, k, n], &Device::Cpu).unwrap();
    let cpu_out = a_cpu.matmul(&b_cpu).unwrap();

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();
    assert_eq!(gpu_vals.len(), cpu_vals.len());
    let max_err = gpu_vals
        .iter()
        .zip(cpu_vals.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 0.05,
        "batched 3D max error {max_err} exceeds tolerance"
    );
}

#[test]
fn test_matmul_4d_multihead() {
    // [2, 8, 16, 32] @ [2, 8, 32, 16] — 4D multi-head attention style matmul.
    init();
    let b = 2;
    let h = 8;
    let m = 16;
    let k = 32;
    let n = 16;
    let total_a = b * h * m * k;
    let total_b = b * h * k * n;
    let a_data: Vec<f32> = (0..total_a).map(|i| ((i % 11) as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..total_b).map(|i| ((i % 13) as f32) * 0.01).collect();

    let a_gpu = DynTensor::from_vec(a_data.clone(), &[b, h, m, k], &Device::metal()).unwrap();
    let b_gpu = DynTensor::from_vec(b_data.clone(), &[b, h, k, n], &Device::metal()).unwrap();
    let gpu_out = a_gpu.matmul(&b_gpu).unwrap();
    assert_eq!(gpu_out.dims(), &[b, h, m, n]);

    let a_cpu = DynTensor::from_vec(a_data, &[b, h, m, k], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::from_vec(b_data, &[b, h, k, n], &Device::Cpu).unwrap();
    let cpu_out = a_cpu.matmul(&b_cpu).unwrap();

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();
    assert_eq!(gpu_vals.len(), cpu_vals.len());
    let max_err = gpu_vals
        .iter()
        .zip(cpu_vals.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 0.05,
        "4D multihead max error {max_err} exceeds tolerance"
    );
}

// -- Large-dimension matmul correctness ----------------------------------------

/// GPU matmul correctness at exactly-32 boundary dimensions.
/// Verifies against CPU reference.
#[test]
fn test_gpu_matmul_boundary_32x32() {
    init();
    let (m, k, n) = (32, 32, 32);
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| ((i % 89) as f32 - 44.0) * 0.01)
        .collect();

    let a_cpu = DynTensor::from_vec(a_data.clone(), &[m, k], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::from_vec(b_data.clone(), &[k, n], &Device::Cpu).unwrap();
    let cpu_out = a_cpu.matmul(&b_cpu).unwrap().to_flat_vec::<f32>().unwrap();

    let a_gpu = DynTensor::from_vec(a_data, &[m, k], &Device::metal()).unwrap();
    let b_gpu = DynTensor::from_vec(b_data, &[k, n], &Device::metal()).unwrap();
    let gpu_out = a_gpu
        .matmul(&b_gpu)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    assert_eq!(gpu_out.len(), cpu_out.len());
    let max_err = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-4,
        "32x32 GPU vs CPU max error {max_err} (tol 1e-4)"
    );
}

/// GPU matmul with non-multiple-of-32 dimensions (M=33, N=35).
/// Tests handling of non-aligned dimension sizes.
#[test]
fn test_gpu_matmul_non_multiple_33x35() {
    init();
    let (m, k, n) = (33, 64, 35);
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| ((i % 71) as f32 - 35.0) * 0.01)
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| ((i % 53) as f32 - 26.0) * 0.01)
        .collect();

    let a_cpu = DynTensor::from_vec(a_data.clone(), &[m, k], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::from_vec(b_data.clone(), &[k, n], &Device::Cpu).unwrap();
    let cpu_out = a_cpu.matmul(&b_cpu).unwrap().to_flat_vec::<f32>().unwrap();

    let a_gpu = DynTensor::from_vec(a_data, &[m, k], &Device::metal()).unwrap();
    let b_gpu = DynTensor::from_vec(b_data, &[k, n], &Device::metal()).unwrap();
    let gpu_out = a_gpu
        .matmul(&b_gpu)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    assert_eq!(gpu_out.len(), cpu_out.len());
    let max_err = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-3,
        "33x35 GPU vs CPU max error {max_err} (tol 1e-3)"
    );
}

/// GPU matmul correctness at transformer-scale (M=256, K=768, N=3072).
/// Verifies against CPU reference.
#[test]
fn test_gpu_matmul_transformer_scale_correctness() {
    init();
    let (m, k, n) = (256, 768, 3072);
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
    let gpu_out = a_gpu
        .matmul(&b_gpu)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    assert_eq!(gpu_out.len(), cpu_out.len());
    let max_err = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    // K=768 with inputs in [-0.048, 0.048]: products ≤ ~0.002, sum of 768 terms ≤ ~1.5.
    // f32 accumulation error: ~768 * 2e-7 * 0.002 ≈ 3e-7 per element. Use 0.01 generously.
    assert!(
        max_err < 0.01,
        "Transformer-scale GPU vs CPU max error {max_err} (tol 0.01)"
    );
}

// -- Simdgroup kernel correctness + routing tests extracted to
// dyn_tensor_metal_matmul_simd_routing_tests.rs (#1567) --
#[path = "dyn_tensor_metal_matmul_simd_routing_tests.rs"]
mod simd_routing;

// Simdgroup vs Naive benchmark tests are in dyn_tensor_metal_matmul_bench.rs.
