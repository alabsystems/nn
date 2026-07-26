#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Basic matmul GPU correctness tests.
//!
//! Covers naive matmul, batched/broadcast matmul, NaN propagation, and
//! edge-case dimensions. Extended size/dimension tests live in
//! `dyn_tensor_metal_matmul_simd_tests.rs` (#1377).

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::init;

// -- Matmul tests -------------------------------------------------------------

#[test]
fn test_gpu_matmul() {
    init();
    // [2,3] @ [3,2] = [2,2]
    let a = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::metal()).unwrap();
    let b = DynTensor::new(
        &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        &[3, 2],
        &Device::metal(),
    )
    .unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    let result = c
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    // [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    // = [[58, 64], [139, 154]]
    assert_eq!(result, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_gpu_matmul_batched_broadcast_right() {
    // [2, 3, 4] @ [4, 5] → [2, 3, 5]
    // Right tensor has no batch dims — must broadcast (all batches share same right).
    // Regression test for #1134: MSL kernel was using batch_idx * right_batch_stride,
    // reading past the end of the right buffer for batch_idx > 0.
    init();
    let l_data: Vec<f32> = (0..24).map(|i| (i + 1) as f32).collect();
    let r_data: Vec<f32> = (0..20).map(|i| (i + 1) as f32).collect();

    let l_gpu = DynTensor::from_vec(l_data.clone(), &[2, 3, 4], &Device::metal()).unwrap();
    let r_gpu = DynTensor::from_vec(r_data.clone(), &[4, 5], &Device::metal()).unwrap();
    let gpu_out = l_gpu.matmul(&r_gpu).unwrap();
    assert_eq!(gpu_out.dims(), &[2, 3, 5]);

    let l_cpu = DynTensor::from_vec(l_data, &[2, 3, 4], &Device::Cpu).unwrap();
    let r_cpu = DynTensor::from_vec(r_data, &[4, 5], &Device::Cpu).unwrap();
    let cpu_out = l_cpu.matmul(&r_cpu).unwrap();

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();
    assert_eq!(gpu_vals.len(), cpu_vals.len());
    for (i, (g, c)) in gpu_vals.iter().zip(cpu_vals.iter()).enumerate() {
        assert!((g - c).abs() < 1e-3, "mismatch at [{i}]: gpu={g}, cpu={c}");
    }
}

#[test]
fn test_gpu_matmul_batched_broadcast_right_b3() {
    // [3, 4, 5] @ [5, 2] → [3, 4, 2]
    // B=3 regression test for #1134 AC3: verifies all 3 batch items match CPU.
    init();
    let l_data: Vec<f32> = (0..60).map(|i| (i as f32) * 0.1).collect();
    let r_data: Vec<f32> = (0..10).map(|i| (i as f32) * 0.1).collect();

    let l_gpu = DynTensor::from_vec(l_data.clone(), &[3, 4, 5], &Device::metal()).unwrap();
    let r_gpu = DynTensor::from_vec(r_data.clone(), &[5, 2], &Device::metal()).unwrap();
    let gpu_out = l_gpu.matmul(&r_gpu).unwrap();
    assert_eq!(gpu_out.dims(), &[3, 4, 2]);

    let l_cpu = DynTensor::from_vec(l_data, &[3, 4, 5], &Device::Cpu).unwrap();
    let r_cpu = DynTensor::from_vec(r_data, &[5, 2], &Device::Cpu).unwrap();
    let cpu_out = l_cpu.matmul(&r_cpu).unwrap();

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();
    assert_eq!(gpu_vals.len(), cpu_vals.len());
    for (i, (g, c)) in gpu_vals.iter().zip(cpu_vals.iter()).enumerate() {
        assert!((g - c).abs() < 1e-3, "mismatch at [{i}]: gpu={g}, cpu={c}");
    }
}

// -- NaN propagation behaviour (#1312, updated after #1375) --------------------

/// After #1375 switched production dispatch to `gpu_matmul_naive` (IR-based),
/// the debug-only NaN guard that lived in `dispatch_tiled_gemm` is no longer on
/// the production path. NaN inputs therefore propagate through the kernel and
/// appear in the output on both debug and release builds.
///
/// This test verifies the observable behaviour: matmul with NaN input succeeds
/// and the output contains NaN values. Model-level finiteness guards (#941) are
/// responsible for catching NaN at stage boundaries.
#[test]
fn test_matmul_nan_input_detected_debug() {
    init();
    let mut a_data = vec![1.0f32; 16];
    a_data[5] = f32::NAN; // inject NaN
    let b_data = vec![1.0f32; 16];

    let a_gpu = DynTensor::from_vec(a_data, &[4, 4], &Device::metal()).unwrap();
    let b_gpu = DynTensor::from_vec(b_data, &[4, 4], &Device::metal()).unwrap();

    // Naive matmul has no output finiteness guard — NaN propagates through.
    let out = a_gpu.matmul(&b_gpu).unwrap();
    let vals = out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(vals.iter().any(|v| v.is_nan()), "Expected NaN in output");
}

// -- Matmul edge-case tests (proof_coverage) ----------------------------------

/// K=1 matmul: outer product [M,1] × [1,N] → [M,N].
/// Exercises the tiled GEMM's inner loop at the K=1 boundary.
#[test]
fn test_gpu_matmul_k1_outer_product() {
    init();
    let a_data: Vec<f32> = (1..=4).map(|x| x as f32).collect(); // [4,1]
    let b_data: Vec<f32> = (1..=3).map(|x| x as f32).collect(); // [1,3]

    let a_cpu = DynTensor::from_vec(a_data.clone(), &[4, 1], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::from_vec(b_data.clone(), &[1, 3], &Device::Cpu).unwrap();
    let cpu_out = a_cpu.matmul(&b_cpu).unwrap();
    let expected = cpu_out.to_flat_vec::<f32>().unwrap();

    let a_gpu = DynTensor::from_vec(a_data, &[4, 1], &Device::metal()).unwrap();
    let b_gpu = DynTensor::from_vec(b_data, &[1, 3], &Device::metal()).unwrap();
    let gpu_out = a_gpu.matmul(&b_gpu).unwrap();

    assert_eq!(gpu_out.dims(), &[4, 3]);
    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    for (i, (g, c)) in gpu_vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - c).abs() < 1e-5,
            "K=1 mismatch at [{i}]: gpu={g}, cpu={c}"
        );
    }
}

/// M=1 matmul: vector-matrix [1,K] × [K,N] → [1,N].
/// Falls to tiled path (M < 32), exercises single-row GEMM.
#[test]
fn test_gpu_matmul_m1_vector_matrix() {
    init();
    let a_data: Vec<f32> = (0..8).map(|x| x as f32 * 0.5).collect(); // [1,8]
    let b_data: Vec<f32> = (0..40).map(|x| (x as f32 * 0.1).sin()).collect(); // [8,5]

    let a_cpu = DynTensor::from_vec(a_data.clone(), &[1, 8], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::from_vec(b_data.clone(), &[8, 5], &Device::Cpu).unwrap();
    let cpu_out = a_cpu.matmul(&b_cpu).unwrap();
    let expected = cpu_out.to_flat_vec::<f32>().unwrap();

    let a_gpu = DynTensor::from_vec(a_data, &[1, 8], &Device::metal()).unwrap();
    let b_gpu = DynTensor::from_vec(b_data, &[8, 5], &Device::metal()).unwrap();
    let gpu_out = a_gpu.matmul(&b_gpu).unwrap();

    assert_eq!(gpu_out.dims(), &[1, 5]);
    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    for (i, (g, c)) in gpu_vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - c).abs() < 1e-4,
            "M=1 mismatch at [{i}]: gpu={g}, cpu={c}"
        );
    }
}
