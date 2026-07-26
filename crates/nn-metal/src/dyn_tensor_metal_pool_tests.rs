#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU parity tests for 2-D pooling operations.
//!
//! Validates that `max_pool2d`, `avg_pool2d`, and `adaptive_avg_pool2d` produce
//! identical results on GPU (via CPU round-trip) as on CPU directly.
//! Part of #4323.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

use crate::test_common::{assert_gpu_vals, init};

// -- Helpers ------------------------------------------------------------------

/// Run a pool op on both CPU and GPU, assert parity within tolerance.
fn assert_pool_parity_max(data: &[f32], shape: &[usize], k: usize, s: usize, p: usize, tol: f32) {
    let cpu = DynTensor::new(data, shape, &Device::Cpu).unwrap();
    let cpu_result = cpu.max_pool2d(k, s, p).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(data, shape, &Device::metal()).unwrap();
    let gpu_result = gpu.max_pool2d(k, s, p).unwrap();
    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_gpu_vals(
        &gpu_result,
        &expected,
        tol,
        &format!("max_pool2d k={k} s={s} p={p}"),
    );
}

fn assert_pool_parity_avg(data: &[f32], shape: &[usize], k: usize, s: usize, p: usize, tol: f32) {
    let cpu = DynTensor::new(data, shape, &Device::Cpu).unwrap();
    let cpu_result = cpu.avg_pool2d(k, s, p).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(data, shape, &Device::metal()).unwrap();
    let gpu_result = gpu.avg_pool2d(k, s, p).unwrap();
    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_gpu_vals(
        &gpu_result,
        &expected,
        tol,
        &format!("avg_pool2d k={k} s={s} p={p}"),
    );
}

fn assert_adaptive_parity(data: &[f32], shape: &[usize], out_h: usize, out_w: usize, tol: f32) {
    let cpu = DynTensor::new(data, shape, &Device::Cpu).unwrap();
    let cpu_result = cpu.adaptive_avg_pool2d(out_h, out_w).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(data, shape, &Device::metal()).unwrap();
    let gpu_result = gpu.adaptive_avg_pool2d(out_h, out_w).unwrap();
    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_gpu_vals(
        &gpu_result,
        &expected,
        tol,
        &format!("adaptive_avg_pool2d {out_h}x{out_w}"),
    );
}

// == max_pool2d GPU parity ====================================================

#[test]
fn test_gpu_max_pool2d_basic() {
    init();
    // [1, 1, 4, 4] -> kernel=2, stride=2, padding=0 -> [1, 1, 2, 2]
    #[rustfmt::skip]
    let data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    assert_pool_parity_max(&data, &[1, 1, 4, 4], 2, 2, 0, 1e-6);
}

#[test]
fn test_gpu_max_pool2d_multi_channel() {
    init();
    // [1, 2, 4, 4] -> kernel=2, stride=2 -> [1, 2, 2, 2]
    let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
    assert_pool_parity_max(&data, &[1, 2, 4, 4], 2, 2, 0, 1e-6);
}

#[test]
fn test_gpu_max_pool2d_batched() {
    init();
    // [2, 2, 4, 4] -> kernel=2, stride=2 -> [2, 2, 2, 2]
    let data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
    assert_pool_parity_max(&data, &[2, 2, 4, 4], 2, 2, 0, 1e-6);
}

#[test]
fn test_gpu_max_pool2d_with_padding() {
    init();
    // [1, 1, 4, 4] -> kernel=3, stride=1, padding=1 -> [1, 1, 4, 4]
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    assert_pool_parity_max(&data, &[1, 1, 4, 4], 3, 1, 1, 1e-6);
}

#[test]
fn test_gpu_max_pool2d_stride_ne_kernel() {
    init();
    // [1, 1, 6, 6] -> kernel=3, stride=2, padding=0 -> [1, 1, 2, 2]
    let data: Vec<f32> = (0..36).map(|i| i as f32).collect();
    assert_pool_parity_max(&data, &[1, 1, 6, 6], 3, 2, 0, 1e-6);
}

#[test]
fn test_gpu_max_pool2d_kernel1_identity() {
    init();
    // kernel_size=1, stride=1 should be identity
    let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
    assert_pool_parity_max(&data, &[1, 1, 4, 4], 1, 1, 0, 1e-6);
}

#[test]
fn test_gpu_max_pool2d_negative_values() {
    init();
    // All negative -- max pool should still work
    let data: Vec<f32> = (0..16).map(|i| -(i as f32) - 1.0).collect();
    assert_pool_parity_max(&data, &[1, 1, 4, 4], 2, 2, 0, 1e-6);
}

#[test]
fn test_gpu_max_pool2d_large() {
    init();
    // [2, 3, 16, 16] -> kernel=4, stride=4 -> [2, 3, 4, 4]
    let n = 2 * 3 * 16 * 16;
    let data: Vec<f32> = (0..n).map(|i| ((i * 17 + 3) % 100) as f32 * 0.01).collect();
    assert_pool_parity_max(&data, &[2, 3, 16, 16], 4, 4, 0, 1e-5);
}

// == avg_pool2d GPU parity ====================================================

#[test]
fn test_gpu_avg_pool2d_basic() {
    init();
    #[rustfmt::skip]
    let data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    assert_pool_parity_avg(&data, &[1, 1, 4, 4], 2, 2, 0, 1e-6);
}

#[test]
fn test_gpu_avg_pool2d_with_padding() {
    init();
    // [1, 1, 2, 2] -> kernel=2, stride=1, padding=1 -> [1, 1, 3, 3]
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    assert_pool_parity_avg(&data, &[1, 1, 2, 2], 2, 1, 1, 1e-6);
}

#[test]
fn test_gpu_avg_pool2d_multi_channel_batched() {
    init();
    // [2, 3, 4, 4] -> kernel=2, stride=2 -> [2, 3, 2, 2]
    let n = 2 * 3 * 4 * 4;
    let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    assert_pool_parity_avg(&data, &[2, 3, 4, 4], 2, 2, 0, 1e-5);
}

#[test]
fn test_gpu_avg_pool2d_stride1_padding1() {
    init();
    // [1, 1, 4, 4] -> kernel=3, stride=1, padding=1 -> [1, 1, 4, 4]
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    assert_pool_parity_avg(&data, &[1, 1, 4, 4], 3, 1, 1, 1e-5);
}

#[test]
fn test_gpu_avg_pool2d_kernel1_identity() {
    init();
    let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
    assert_pool_parity_avg(&data, &[1, 1, 4, 4], 1, 1, 0, 1e-6);
}

#[test]
fn test_gpu_avg_pool2d_large() {
    init();
    // [2, 4, 8, 8] -> kernel=2, stride=2 -> [2, 4, 4, 4]
    let n = 2 * 4 * 8 * 8;
    let data: Vec<f32> = (0..n).map(|i| ((i * 13 + 7) % 50) as f32 * 0.02).collect();
    assert_pool_parity_avg(&data, &[2, 4, 8, 8], 2, 2, 0, 1e-5);
}

// == adaptive_avg_pool2d GPU parity ===========================================

#[test]
fn test_gpu_adaptive_avg_pool2d_to_1x1() {
    init();
    #[rustfmt::skip]
    let data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    assert_adaptive_parity(&data, &[1, 1, 4, 4], 1, 1, 1e-5);
}

#[test]
fn test_gpu_adaptive_avg_pool2d_downscale() {
    init();
    #[rustfmt::skip]
    let data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    assert_adaptive_parity(&data, &[1, 1, 4, 4], 2, 2, 1e-5);
}

#[test]
fn test_gpu_adaptive_avg_pool2d_identity() {
    init();
    // Output same size as input -- should be identity.
    let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
    assert_adaptive_parity(&data, &[1, 1, 4, 4], 4, 4, 1e-6);
}

#[test]
fn test_gpu_adaptive_avg_pool2d_non_divisible() {
    init();
    // [1, 1, 7, 7] -> [1, 1, 3, 3] -- non-divisible dimensions
    let data: Vec<f32> = (0..49).map(|i| i as f32 * 0.1).collect();
    assert_adaptive_parity(&data, &[1, 1, 7, 7], 3, 3, 1e-5);
}

#[test]
fn test_gpu_adaptive_avg_pool2d_batched_multi_channel() {
    init();
    // [2, 3, 6, 6] -> [2, 3, 2, 2]
    let n = 2 * 3 * 6 * 6;
    let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    assert_adaptive_parity(&data, &[2, 3, 6, 6], 2, 2, 1e-5);
}

#[test]
fn test_gpu_adaptive_avg_pool2d_asymmetric_output() {
    init();
    // [1, 1, 8, 6] -> [1, 1, 3, 2] -- asymmetric output size
    let n = 1 * 1 * 8 * 6;
    let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    assert_adaptive_parity(&data, &[1, 1, 8, 6], 3, 2, 1e-5);
}

#[test]
fn test_gpu_adaptive_avg_pool2d_large() {
    init();
    // [2, 4, 14, 14] -> [2, 4, 7, 7]
    let n = 2 * 4 * 14 * 14;
    let data: Vec<f32> = (0..n).map(|i| ((i * 11 + 5) % 100) as f32 * 0.01).collect();
    assert_adaptive_parity(&data, &[2, 4, 14, 14], 7, 7, 1e-5);
}

// == F16 dtype parity =========================================================

#[test]
fn test_gpu_max_pool2d_f16() {
    init();
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let cpu_f32 = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let cpu_result = cpu_f32.max_pool2d(2, 2, 0).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_f16 = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let gpu_result = gpu_f16.max_pool2d(2, 2, 0).unwrap();
    assert_eq!(gpu_result.dims(), cpu_result.dims());
    // F16 has lower precision; use wider tolerance.
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    for (i, (g, e)) in gpu_vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= 0.1,
            "max_pool2d f16 [{i}]: gpu={g}, expected={e}",
        );
    }
}

#[test]
fn test_gpu_avg_pool2d_f16() {
    init();
    let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    let cpu_f32 = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let cpu_result = cpu_f32.avg_pool2d(2, 2, 0).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_f16 = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let gpu_result = gpu_f16.avg_pool2d(2, 2, 0).unwrap();
    assert_eq!(gpu_result.dims(), cpu_result.dims());
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    for (i, (g, e)) in gpu_vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= 0.05,
            "avg_pool2d f16 [{i}]: gpu={g}, expected={e}",
        );
    }
}

#[test]
fn test_gpu_adaptive_avg_pool2d_f16() {
    init();
    let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    let cpu_f32 = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let cpu_result = cpu_f32.adaptive_avg_pool2d(2, 2).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_f16 = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let gpu_result = gpu_f16.adaptive_avg_pool2d(2, 2).unwrap();
    assert_eq!(gpu_result.dims(), cpu_result.dims());
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    for (i, (g, e)) in gpu_vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= 0.05,
            "adaptive_avg_pool2d f16 [{i}]: gpu={g}, expected={e}",
        );
    }
}

// == BF16 dtype parity ========================================================

#[test]
fn test_gpu_max_pool2d_bf16() {
    init();
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let cpu_f32 = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let cpu_result = cpu_f32.max_pool2d(2, 2, 0).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_bf16 = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let gpu_result = gpu_bf16.max_pool2d(2, 2, 0).unwrap();
    assert_eq!(gpu_result.dims(), cpu_result.dims());
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    for (i, (g, e)) in gpu_vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= 0.2,
            "max_pool2d bf16 [{i}]: gpu={g}, expected={e}",
        );
    }
}

#[test]
fn test_gpu_avg_pool2d_bf16() {
    init();
    let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    let cpu_f32 = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let cpu_result = cpu_f32.avg_pool2d(2, 2, 0).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_bf16 = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let gpu_result = gpu_bf16.avg_pool2d(2, 2, 0).unwrap();
    assert_eq!(gpu_result.dims(), cpu_result.dims());
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    for (i, (g, e)) in gpu_vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= 0.1,
            "avg_pool2d bf16 [{i}]: gpu={g}, expected={e}",
        );
    }
}

// == Error/edge case tests ====================================================

#[test]
fn test_gpu_max_pool2d_rank_error() {
    init();
    // Rank 3 should fail on GPU (rank 4 required).
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gpu = DynTensor::new(&data, &[1, 2, 3], &Device::metal()).unwrap();
    let result = gpu.max_pool2d(2, 1, 0);
    assert!(result.is_err(), "should reject rank-3 input");
}

#[test]
fn test_gpu_avg_pool2d_rank_error() {
    init();
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gpu = DynTensor::new(&data, &[1, 2, 3], &Device::metal()).unwrap();
    let result = gpu.avg_pool2d(2, 1, 0);
    assert!(result.is_err(), "should reject rank-3 input");
}

#[test]
fn test_gpu_adaptive_avg_pool2d_rank_error() {
    init();
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gpu = DynTensor::new(&data, &[1, 2, 3], &Device::metal()).unwrap();
    let result = gpu.adaptive_avg_pool2d(1, 1);
    assert!(result.is_err(), "should reject rank-3 input");
}
