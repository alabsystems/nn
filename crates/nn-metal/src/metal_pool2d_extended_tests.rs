// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for Metal GPU pool2d kernel dispatch configuration.
//!
//! Covers MaxPool2d, AvgPool2d, AdaptiveAvgPool2d dispatch parameters,
//! output size calculations, stride/padding/dilation handling, edge cases
//! (1x1 kernels, global pooling, non-square kernels), batch dimension
//! handling, and DType compatibility (F32, F16, BF16).
//!
//! These tests validate dispatch configuration and parameter correctness
//! without requiring a live GPU device (structure/config tests), plus
//! CPU-vs-GPU parity tests for Metal-equipped machines.
//!
//! Part of #4323.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use nn_dsl::ir::ScalarType;

use crate::test_common::{assert_gpu_vals, init};

// ═══════════════════════════════════════════════════════════════════════
// 1. MaxPool2d kernel dispatch configuration
// ═══════════════════════════════════════════════════════════════════════

/// MaxPool2d basic 2x2 kernel with stride=2 on [1, 1, 4, 4].
/// Output should be [1, 1, 2, 2] with correct max values.
#[test]
fn max_pool2d_dispatch_basic_2x2() {
    init();
    #[rustfmt::skip]
    let data: Vec<f32> = vec![
        1.0,  3.0,  2.0,  4.0,
        5.0,  7.0,  6.0,  8.0,
        9.0, 11.0, 10.0, 12.0,
       13.0, 15.0, 14.0, 16.0,
    ];
    let cpu = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let cpu_result = cpu.max_pool2d(2, 2, 0).unwrap();
    assert_eq!(cpu_result.dims(), &[1, 1, 2, 2]);
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 4, 4], &Device::metal()).unwrap();
    let gpu_result = gpu.max_pool2d(2, 2, 0).unwrap();
    assert_eq!(gpu_result.dims(), &[1, 1, 2, 2]);
    assert_gpu_vals(&gpu_result, &expected, 1e-6, "max_pool2d basic 2x2");
}

/// MaxPool2d with 3x3 kernel, stride=1, padding=0 on [1, 1, 5, 5].
/// Output should be [1, 1, 3, 3].
#[test]
fn max_pool2d_dispatch_3x3_stride1() {
    init();
    let data: Vec<f32> = (0..25).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 5, 5], &Device::Cpu).unwrap();
    let cpu_result = cpu.max_pool2d(3, 1, 0).unwrap();
    assert_eq!(cpu_result.dims(), &[1, 1, 3, 3]);
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 5, 5], &Device::metal()).unwrap();
    let gpu_result = gpu.max_pool2d(3, 1, 0).unwrap();
    assert_eq!(gpu_result.dims(), &[1, 1, 3, 3]);
    assert_gpu_vals(&gpu_result, &expected, 1e-6, "max_pool2d 3x3 stride=1");
}

/// MaxPool2d with large kernel covering entire spatial extent (global max pooling).
#[test]
fn max_pool2d_dispatch_global_pooling() {
    init();
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let cpu_result = cpu.max_pool2d(4, 1, 0).unwrap();
    assert_eq!(cpu_result.dims(), &[1, 1, 1, 1]);
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();
    // Global max of 0..15 is 15.0
    assert!((expected[0] - 15.0).abs() < 1e-6);

    let gpu = DynTensor::new(&data, &[1, 1, 4, 4], &Device::metal()).unwrap();
    let gpu_result = gpu.max_pool2d(4, 1, 0).unwrap();
    assert_eq!(gpu_result.dims(), &[1, 1, 1, 1]);
    assert_gpu_vals(&gpu_result, &expected, 1e-6, "max_pool2d global");
}

/// MaxPool2d preserves correct output when all values are identical.
#[test]
fn max_pool2d_dispatch_uniform_values() {
    init();
    let data: Vec<f32> = vec![3.14; 16];
    let cpu = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let cpu_result = cpu.max_pool2d(2, 2, 0).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 4, 4], &Device::metal()).unwrap();
    let gpu_result = gpu.max_pool2d(2, 2, 0).unwrap();
    assert_gpu_vals(&gpu_result, &expected, 1e-6, "max_pool2d uniform");
}

// ═══════════════════════════════════════════════════════════════════════
// 2. AvgPool2d kernel dispatch configuration
// ═══════════════════════════════════════════════════════════════════════

/// AvgPool2d basic 2x2 kernel with stride=2 on [1, 1, 4, 4].
#[test]
fn avg_pool2d_dispatch_basic_2x2() {
    init();
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let cpu_result = cpu.avg_pool2d(2, 2, 0).unwrap();
    assert_eq!(cpu_result.dims(), &[1, 1, 2, 2]);
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 4, 4], &Device::metal()).unwrap();
    let gpu_result = gpu.avg_pool2d(2, 2, 0).unwrap();
    assert_eq!(gpu_result.dims(), &[1, 1, 2, 2]);
    assert_gpu_vals(&gpu_result, &expected, 1e-5, "avg_pool2d basic 2x2");
}

/// AvgPool2d with 3x3 kernel, stride=1, padding=0 on [1, 1, 5, 5].
#[test]
fn avg_pool2d_dispatch_3x3_stride1() {
    init();
    let data: Vec<f32> = (0..25).map(|i| i as f32 * 0.1).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 5, 5], &Device::Cpu).unwrap();
    let cpu_result = cpu.avg_pool2d(3, 1, 0).unwrap();
    assert_eq!(cpu_result.dims(), &[1, 1, 3, 3]);
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 5, 5], &Device::metal()).unwrap();
    let gpu_result = gpu.avg_pool2d(3, 1, 0).unwrap();
    assert_eq!(gpu_result.dims(), &[1, 1, 3, 3]);
    assert_gpu_vals(&gpu_result, &expected, 1e-5, "avg_pool2d 3x3 stride=1");
}

/// AvgPool2d global pooling: kernel covers entire spatial dims.
#[test]
fn avg_pool2d_dispatch_global_pooling() {
    init();
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let cpu_result = cpu.avg_pool2d(4, 1, 0).unwrap();
    assert_eq!(cpu_result.dims(), &[1, 1, 1, 1]);
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();
    // Mean of 0..15 = 7.5
    assert!((expected[0] - 7.5).abs() < 1e-5);

    let gpu = DynTensor::new(&data, &[1, 1, 4, 4], &Device::metal()).unwrap();
    let gpu_result = gpu.avg_pool2d(4, 1, 0).unwrap();
    assert_eq!(gpu_result.dims(), &[1, 1, 1, 1]);
    assert_gpu_vals(&gpu_result, &expected, 1e-5, "avg_pool2d global");
}

/// AvgPool2d with uniform values should produce the same uniform value.
#[test]
fn avg_pool2d_dispatch_uniform_values() {
    init();
    let val = 2.5f32;
    let data: Vec<f32> = vec![val; 36];
    let cpu = DynTensor::new(&data, &[1, 1, 6, 6], &Device::Cpu).unwrap();
    let cpu_result = cpu.avg_pool2d(3, 3, 0).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();
    for e in &expected {
        assert!((e - val).abs() < 1e-6, "uniform avg should be {val}, got {e}");
    }

    let gpu = DynTensor::new(&data, &[1, 1, 6, 6], &Device::metal()).unwrap();
    let gpu_result = gpu.avg_pool2d(3, 3, 0).unwrap();
    assert_gpu_vals(&gpu_result, &expected, 1e-5, "avg_pool2d uniform");
}

// ═══════════════════════════════════════════════════════════════════════
// 3. AdaptiveAvgPool2d output size calculations
// ═══════════════════════════════════════════════════════════════════════

/// AdaptiveAvgPool2d to 1x1 (global average pooling).
#[test]
fn adaptive_avg_pool2d_output_1x1() {
    init();
    let data: Vec<f32> = (0..36).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 6, 6], &Device::Cpu).unwrap();
    let cpu_result = cpu.adaptive_avg_pool2d(1, 1).unwrap();
    assert_eq!(cpu_result.dims(), &[1, 1, 1, 1]);
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();
    // Mean of 0..35 = 17.5
    assert!((expected[0] - 17.5).abs() < 1e-4);

    let gpu = DynTensor::new(&data, &[1, 1, 6, 6], &Device::metal()).unwrap();
    let gpu_result = gpu.adaptive_avg_pool2d(1, 1).unwrap();
    assert_eq!(gpu_result.dims(), &[1, 1, 1, 1]);
    assert_gpu_vals(&gpu_result, &expected, 1e-4, "adaptive 1x1");
}

/// AdaptiveAvgPool2d with output size equal to input (identity).
#[test]
fn adaptive_avg_pool2d_output_identity() {
    init();
    let data: Vec<f32> = (0..25).map(|i| i as f32 * 0.1).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 5, 5], &Device::Cpu).unwrap();
    let cpu_result = cpu.adaptive_avg_pool2d(5, 5).unwrap();
    assert_eq!(cpu_result.dims(), &[1, 1, 5, 5]);
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();
    // Identity: each output equals the corresponding input.
    for (i, (&e, &d)) in expected.iter().zip(data.iter()).enumerate() {
        assert!(
            (e - d).abs() < 1e-6,
            "identity adaptive [{i}]: expected {d}, got {e}",
        );
    }

    let gpu = DynTensor::new(&data, &[1, 1, 5, 5], &Device::metal()).unwrap();
    let gpu_result = gpu.adaptive_avg_pool2d(5, 5).unwrap();
    assert_eq!(gpu_result.dims(), &[1, 1, 5, 5]);
    assert_gpu_vals(&gpu_result, &expected, 1e-6, "adaptive identity");
}

/// AdaptiveAvgPool2d with non-divisible output sizes.
#[test]
fn adaptive_avg_pool2d_output_non_divisible() {
    init();
    // [1, 1, 7, 7] -> [1, 1, 3, 3] (7 not divisible by 3)
    let data: Vec<f32> = (0..49).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 7, 7], &Device::Cpu).unwrap();
    let cpu_result = cpu.adaptive_avg_pool2d(3, 3).unwrap();
    assert_eq!(cpu_result.dims(), &[1, 1, 3, 3]);
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 7, 7], &Device::metal()).unwrap();
    let gpu_result = gpu.adaptive_avg_pool2d(3, 3).unwrap();
    assert_eq!(gpu_result.dims(), &[1, 1, 3, 3]);
    assert_gpu_vals(&gpu_result, &expected, 1e-4, "adaptive non-divisible");
}

/// AdaptiveAvgPool2d with asymmetric output (out_h != out_w).
#[test]
fn adaptive_avg_pool2d_output_asymmetric() {
    init();
    // [1, 1, 8, 6] -> [1, 1, 2, 3]
    let data: Vec<f32> = (0..48).map(|i| i as f32 * 0.1).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 8, 6], &Device::Cpu).unwrap();
    let cpu_result = cpu.adaptive_avg_pool2d(2, 3).unwrap();
    assert_eq!(cpu_result.dims(), &[1, 1, 2, 3]);
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 8, 6], &Device::metal()).unwrap();
    let gpu_result = gpu.adaptive_avg_pool2d(2, 3).unwrap();
    assert_eq!(gpu_result.dims(), &[1, 1, 2, 3]);
    assert_gpu_vals(&gpu_result, &expected, 1e-4, "adaptive asymmetric");
}

/// AdaptiveAvgPool2d rejects zero output dimensions.
#[test]
fn adaptive_avg_pool2d_zero_output_error() {
    init();
    let data: Vec<f32> = vec![1.0; 16];
    let t = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    assert!(t.adaptive_avg_pool2d(0, 1).is_err(), "out_h=0 should error");
    assert!(t.adaptive_avg_pool2d(1, 0).is_err(), "out_w=0 should error");
}

// ═══════════════════════════════════════════════════════════════════════
// 4. Pool2d stride/padding parameter handling
// ═══════════════════════════════════════════════════════════════════════

/// MaxPool2d with stride != kernel_size (overlapping windows).
#[test]
fn pool2d_stride_less_than_kernel() {
    init();
    // kernel=3, stride=1, padding=0 on [1, 1, 5, 5] -> [1, 1, 3, 3]
    let data: Vec<f32> = (0..25).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 5, 5], &Device::Cpu).unwrap();
    let cpu_result = cpu.max_pool2d(3, 1, 0).unwrap();
    assert_eq!(cpu_result.dims(), &[1, 1, 3, 3]);
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 5, 5], &Device::metal()).unwrap();
    let gpu_result = gpu.max_pool2d(3, 1, 0).unwrap();
    assert_gpu_vals(&gpu_result, &expected, 1e-6, "max_pool stride<kernel");
}

/// MaxPool2d with stride > kernel_size (non-overlapping, gaps between windows).
#[test]
fn pool2d_stride_greater_than_kernel() {
    init();
    // kernel=2, stride=3, padding=0 on [1, 1, 6, 6] -> [1, 1, 2, 2]
    let data: Vec<f32> = (0..36).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 6, 6], &Device::Cpu).unwrap();
    let cpu_result = cpu.max_pool2d(2, 3, 0).unwrap();
    assert_eq!(cpu_result.dims(), &[1, 1, 2, 2]);
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 6, 6], &Device::metal()).unwrap();
    let gpu_result = gpu.max_pool2d(2, 3, 0).unwrap();
    assert_gpu_vals(&gpu_result, &expected, 1e-6, "max_pool stride>kernel");
}

/// AvgPool2d with padding=1 on small input.
#[test]
fn pool2d_padding_avg_pool() {
    init();
    // [1, 1, 3, 3] kernel=3, stride=1, padding=1 -> [1, 1, 3, 3]
    let data: Vec<f32> = (1..=9).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 3, 3], &Device::Cpu).unwrap();
    let cpu_result = cpu.avg_pool2d(3, 1, 1).unwrap();
    assert_eq!(cpu_result.dims(), &[1, 1, 3, 3]);
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 3, 3], &Device::metal()).unwrap();
    let gpu_result = gpu.avg_pool2d(3, 1, 1).unwrap();
    assert_eq!(gpu_result.dims(), &[1, 1, 3, 3]);
    assert_gpu_vals(&gpu_result, &expected, 1e-5, "avg_pool2d padding=1");
}

/// MaxPool2d with large padding increases output spatial dims.
#[test]
fn pool2d_large_padding_max_pool() {
    init();
    // [1, 1, 4, 4] kernel=3, stride=1, padding=2 -> [1, 1, 6, 6]
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let cpu_result = cpu.max_pool2d(3, 1, 2).unwrap();
    assert_eq!(cpu_result.dims(), &[1, 1, 6, 6]);
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 4, 4], &Device::metal()).unwrap();
    let gpu_result = gpu.max_pool2d(3, 1, 2).unwrap();
    assert_eq!(gpu_result.dims(), &[1, 1, 6, 6]);
    assert_gpu_vals(&gpu_result, &expected, 1e-6, "max_pool large padding");
}

/// Pool2d with zero stride should return an error.
#[test]
fn pool2d_zero_stride_error() {
    let data: Vec<f32> = vec![1.0; 16];
    let t = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    assert!(t.max_pool2d(2, 0, 0).is_err(), "stride=0 should error");
    assert!(t.avg_pool2d(2, 0, 0).is_err(), "stride=0 should error");
}

/// Pool2d with zero kernel_size should return an error.
#[test]
fn pool2d_zero_kernel_error() {
    let data: Vec<f32> = vec![1.0; 16];
    let t = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    assert!(
        t.max_pool2d(0, 1, 0).is_err(),
        "kernel_size=0 should error"
    );
    assert!(
        t.avg_pool2d(0, 1, 0).is_err(),
        "kernel_size=0 should error"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 5. Edge cases: 1x1 kernels, global pooling, non-square kernels
// ═══════════════════════════════════════════════════════════════════════

/// 1x1 kernel with stride=1 is identity for both max and avg pooling.
#[test]
fn pool2d_1x1_kernel_identity_max() {
    init();
    let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.5).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 4, 6], &Device::Cpu).unwrap();
    let cpu_result = cpu.max_pool2d(1, 1, 0).unwrap();
    assert_eq!(cpu_result.dims(), &[1, 1, 4, 6]);
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();
    // 1x1 max pool = identity
    for (i, (&e, &d)) in expected.iter().zip(data.iter()).enumerate() {
        assert!((e - d).abs() < 1e-6, "1x1 max [{i}]: {e} != {d}");
    }

    let gpu = DynTensor::new(&data, &[1, 1, 4, 6], &Device::metal()).unwrap();
    let gpu_result = gpu.max_pool2d(1, 1, 0).unwrap();
    assert_gpu_vals(&gpu_result, &expected, 1e-6, "max_pool 1x1 identity");
}

/// 1x1 kernel with stride=1 is identity for avg pooling.
#[test]
fn pool2d_1x1_kernel_identity_avg() {
    init();
    let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.5).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 4, 6], &Device::Cpu).unwrap();
    let cpu_result = cpu.avg_pool2d(1, 1, 0).unwrap();
    assert_eq!(cpu_result.dims(), &[1, 1, 4, 6]);
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 4, 6], &Device::metal()).unwrap();
    let gpu_result = gpu.avg_pool2d(1, 1, 0).unwrap();
    assert_gpu_vals(&gpu_result, &expected, 1e-6, "avg_pool 1x1 identity");
}

/// 1x1 kernel with stride=2 produces spatial downsampling (subsampling).
#[test]
fn pool2d_1x1_kernel_stride2_subsampling() {
    init();
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let cpu_result = cpu.max_pool2d(1, 2, 0).unwrap();
    assert_eq!(cpu_result.dims(), &[1, 1, 2, 2]);
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 4, 4], &Device::metal()).unwrap();
    let gpu_result = gpu.max_pool2d(1, 2, 0).unwrap();
    assert_gpu_vals(&gpu_result, &expected, 1e-6, "max_pool 1x1 stride=2");
}

/// Non-square input spatial dimensions: [1, 1, 3, 7].
#[test]
fn pool2d_non_square_input() {
    init();
    let data: Vec<f32> = (0..21).map(|i| i as f32 * 0.1).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 3, 7], &Device::Cpu).unwrap();
    let cpu_result = cpu.max_pool2d(2, 1, 0).unwrap();
    assert_eq!(cpu_result.dims(), &[1, 1, 2, 6]);
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 3, 7], &Device::metal()).unwrap();
    let gpu_result = gpu.max_pool2d(2, 1, 0).unwrap();
    assert_eq!(gpu_result.dims(), &[1, 1, 2, 6]);
    assert_gpu_vals(&gpu_result, &expected, 1e-5, "max_pool non-square");
}

/// Minimum spatial: [1, 1, 1, 1] with kernel=1.
#[test]
fn pool2d_minimum_spatial() {
    init();
    let data = vec![42.0f32];
    let cpu = DynTensor::new(&data, &[1, 1, 1, 1], &Device::Cpu).unwrap();
    let cpu_max = cpu.max_pool2d(1, 1, 0).unwrap();
    let cpu_avg = cpu.avg_pool2d(1, 1, 0).unwrap();
    assert_eq!(cpu_max.dims(), &[1, 1, 1, 1]);
    assert_eq!(cpu_avg.dims(), &[1, 1, 1, 1]);
    assert!((cpu_max.to_flat_vec::<f32>().unwrap()[0] - 42.0).abs() < 1e-6);
    assert!((cpu_avg.to_flat_vec::<f32>().unwrap()[0] - 42.0).abs() < 1e-6);

    let gpu = DynTensor::new(&data, &[1, 1, 1, 1], &Device::metal()).unwrap();
    let gpu_max = gpu.max_pool2d(1, 1, 0).unwrap();
    let gpu_avg = gpu.avg_pool2d(1, 1, 0).unwrap();
    assert_gpu_vals(&gpu_max, &[42.0], 1e-6, "max_pool 1x1 spatial");
    assert_gpu_vals(&gpu_avg, &[42.0], 1e-6, "avg_pool 1x1 spatial");
}

// ═══════════════════════════════════════════════════════════════════════
// 6. Batch dimension handling
// ═══════════════════════════════════════════════════════════════════════

/// MaxPool2d on batched input [4, 2, 6, 6].
#[test]
fn pool2d_batch_max_pool_multi_batch() {
    init();
    let n = 4 * 2 * 6 * 6;
    let data: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 100) as f32 * 0.01).collect();
    let cpu = DynTensor::new(&data, &[4, 2, 6, 6], &Device::Cpu).unwrap();
    let cpu_result = cpu.max_pool2d(2, 2, 0).unwrap();
    assert_eq!(cpu_result.dims(), &[4, 2, 3, 3]);
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[4, 2, 6, 6], &Device::metal()).unwrap();
    let gpu_result = gpu.max_pool2d(2, 2, 0).unwrap();
    assert_eq!(gpu_result.dims(), &[4, 2, 3, 3]);
    assert_gpu_vals(&gpu_result, &expected, 1e-5, "max_pool batched");
}

/// AvgPool2d on batched input [3, 4, 8, 8].
#[test]
fn pool2d_batch_avg_pool_multi_batch() {
    init();
    let n = 3 * 4 * 8 * 8;
    let data: Vec<f32> = (0..n).map(|i| ((i * 11 + 5) % 100) as f32 * 0.02).collect();
    let cpu = DynTensor::new(&data, &[3, 4, 8, 8], &Device::Cpu).unwrap();
    let cpu_result = cpu.avg_pool2d(2, 2, 0).unwrap();
    assert_eq!(cpu_result.dims(), &[3, 4, 4, 4]);
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[3, 4, 8, 8], &Device::metal()).unwrap();
    let gpu_result = gpu.avg_pool2d(2, 2, 0).unwrap();
    assert_eq!(gpu_result.dims(), &[3, 4, 4, 4]);
    assert_gpu_vals(&gpu_result, &expected, 1e-5, "avg_pool batched");
}

/// AdaptiveAvgPool2d on batched input [2, 3, 10, 10] -> [2, 3, 3, 3].
#[test]
fn pool2d_batch_adaptive_multi_batch() {
    init();
    let n = 2 * 3 * 10 * 10;
    let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    let cpu = DynTensor::new(&data, &[2, 3, 10, 10], &Device::Cpu).unwrap();
    let cpu_result = cpu.adaptive_avg_pool2d(3, 3).unwrap();
    assert_eq!(cpu_result.dims(), &[2, 3, 3, 3]);
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[2, 3, 10, 10], &Device::metal()).unwrap();
    let gpu_result = gpu.adaptive_avg_pool2d(3, 3).unwrap();
    assert_eq!(gpu_result.dims(), &[2, 3, 3, 3]);
    assert_gpu_vals(&gpu_result, &expected, 1e-4, "adaptive batched");
}

/// Single batch, many channels: [1, 16, 4, 4].
#[test]
fn pool2d_batch_many_channels() {
    init();
    let n = 1 * 16 * 4 * 4;
    let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let cpu = DynTensor::new(&data, &[1, 16, 4, 4], &Device::Cpu).unwrap();
    let cpu_result = cpu.max_pool2d(2, 2, 0).unwrap();
    assert_eq!(cpu_result.dims(), &[1, 16, 2, 2]);
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 16, 4, 4], &Device::metal()).unwrap();
    let gpu_result = gpu.max_pool2d(2, 2, 0).unwrap();
    assert_eq!(gpu_result.dims(), &[1, 16, 2, 2]);
    assert_gpu_vals(&gpu_result, &expected, 1e-5, "max_pool many channels");
}

/// Rank-3 input should be rejected by all pool2d ops.
#[test]
fn pool2d_batch_rank3_error() {
    init();
    let data: Vec<f32> = vec![1.0; 24];
    let gpu = DynTensor::new(&data, &[2, 3, 4], &Device::metal()).unwrap();
    assert!(gpu.max_pool2d(2, 2, 0).is_err(), "rank 3 should error for max_pool2d");
    assert!(gpu.avg_pool2d(2, 2, 0).is_err(), "rank 3 should error for avg_pool2d");
    assert!(
        gpu.adaptive_avg_pool2d(1, 1).is_err(),
        "rank 3 should error for adaptive_avg_pool2d"
    );
}

/// Rank-5 input should be rejected by all pool2d ops.
#[test]
fn pool2d_batch_rank5_error() {
    init();
    let data: Vec<f32> = vec![1.0; 32];
    let gpu = DynTensor::new(&data, &[1, 1, 2, 4, 4], &Device::metal()).unwrap();
    assert!(gpu.max_pool2d(2, 2, 0).is_err(), "rank 5 should error for max_pool2d");
    assert!(gpu.avg_pool2d(2, 2, 0).is_err(), "rank 5 should error for avg_pool2d");
    assert!(
        gpu.adaptive_avg_pool2d(1, 1).is_err(),
        "rank 5 should error for adaptive_avg_pool2d"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 7. DType compatibility (F32, F16, BF16)
// ═══════════════════════════════════════════════════════════════════════

/// MaxPool2d F32 GPU parity with small specific values.
#[test]
fn pool2d_dtype_max_pool_f32() {
    init();
    let data: Vec<f32> = vec![
        -1.0, 2.0, 0.5, 3.0,
         4.0, -2.0, 1.5, 0.0,
         0.0, 5.0, -3.0, 1.0,
         2.0, 1.0, 4.0, -1.0,
    ];
    let cpu = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let cpu_result = cpu.max_pool2d(2, 2, 0).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 4, 4], &Device::metal()).unwrap();
    let gpu_result = gpu.max_pool2d(2, 2, 0).unwrap();
    assert_gpu_vals(&gpu_result, &expected, 1e-6, "max_pool f32");
}

/// AvgPool2d F16 GPU dispatch: verify output matches F32 CPU within F16 tolerance.
#[test]
fn pool2d_dtype_avg_pool_f16() {
    init();
    let data: Vec<f32> = (0..36).map(|i| i as f32 * 0.1).collect();
    let cpu_f32 = DynTensor::new(&data, &[1, 1, 6, 6], &Device::Cpu).unwrap();
    let cpu_result = cpu_f32.avg_pool2d(2, 2, 0).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_f16 = DynTensor::new(&data, &[1, 1, 6, 6], &Device::Cpu)
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
            (g - e).abs() <= 0.1,
            "avg_pool2d f16 [{i}]: gpu={g}, expected={e}",
        );
    }
}

/// MaxPool2d BF16 GPU dispatch.
#[test]
fn pool2d_dtype_max_pool_bf16() {
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
            (g - e).abs() <= 0.5,
            "max_pool2d bf16 [{i}]: gpu={g}, expected={e}",
        );
    }
}

/// AdaptiveAvgPool2d F16 GPU dispatch.
#[test]
fn pool2d_dtype_adaptive_f16() {
    init();
    let data: Vec<f32> = (0..36).map(|i| i as f32 * 0.1).collect();
    let cpu_f32 = DynTensor::new(&data, &[1, 1, 6, 6], &Device::Cpu).unwrap();
    let cpu_result = cpu_f32.adaptive_avg_pool2d(2, 2).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_f16 = DynTensor::new(&data, &[1, 1, 6, 6], &Device::Cpu)
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
            (g - e).abs() <= 0.1,
            "adaptive_avg_pool2d f16 [{i}]: gpu={g}, expected={e}",
        );
    }
}

/// AdaptiveAvgPool2d BF16 GPU dispatch.
#[test]
fn pool2d_dtype_adaptive_bf16() {
    init();
    let data: Vec<f32> = (0..36).map(|i| i as f32 * 0.1).collect();
    let cpu_f32 = DynTensor::new(&data, &[1, 1, 6, 6], &Device::Cpu).unwrap();
    let cpu_result = cpu_f32.adaptive_avg_pool2d(3, 3).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_bf16 = DynTensor::new(&data, &[1, 1, 6, 6], &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let gpu_result = gpu_bf16.adaptive_avg_pool2d(3, 3).unwrap();
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
            "adaptive_avg_pool2d bf16 [{i}]: gpu={g}, expected={e}",
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 8. MSL kernel generation and ScalarType mapping for pool ops
// ═══════════════════════════════════════════════════════════════════════

/// ScalarType for pool dispatch dtypes maps correctly.
#[test]
fn pool2d_scalar_type_mapping() {
    let f32_st = ScalarType::try_from(DType::F32).unwrap();
    assert_eq!(f32_st.msl_str(), "float");
    assert_eq!(f32_st.byte_size(), 4);

    let f16_st = ScalarType::try_from(DType::F16).unwrap();
    assert_eq!(f16_st.msl_str(), "half");
    assert_eq!(f16_st.byte_size(), 2);

    let bf16_st = ScalarType::try_from(DType::BF16).unwrap();
    assert_eq!(bf16_st.msl_str(), "half");
    assert_eq!(bf16_st.byte_size(), 2);
}

/// dtype_to_msl for pool-compatible types returns valid MSL type + size.
#[test]
fn pool2d_dtype_to_msl_valid() {
    let (msl, size) = crate::dtype_to_msl(DType::F32).unwrap();
    assert_eq!(msl, "float");
    assert_eq!(size, 4);

    let (msl, size) = crate::dtype_to_msl(DType::F16).unwrap();
    assert_eq!(msl, "half");
    assert_eq!(size, 2);

    let (msl, size) = crate::dtype_to_msl(DType::BF16).unwrap();
    assert_eq!(msl, "half");
    assert_eq!(size, 2);
}

/// Integer types should fail dtype_to_msl (pool ops are float-only).
#[test]
fn pool2d_dtype_to_msl_integer_rejected() {
    assert!(crate::dtype_to_msl(DType::I32).is_err());
    assert!(crate::dtype_to_msl(DType::I64).is_err());
    assert!(crate::dtype_to_msl(DType::U32).is_err());
    assert!(crate::dtype_to_msl(DType::U8).is_err());
}

// ═══════════════════════════════════════════════════════════════════════
// 9. Pool2d output dimension formula verification
// ═══════════════════════════════════════════════════════════════════════

/// Verify output shape formula: out = (input + 2*padding - kernel) / stride + 1.
#[test]
fn pool2d_output_dim_formula_basic() {
    // (4 + 0 - 2) / 2 + 1 = 2
    let t = DynTensor::new(&[0.0f32; 16], &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let result = t.max_pool2d(2, 2, 0).unwrap();
    assert_eq!(result.dims(), &[1, 1, 2, 2]);
}

#[test]
fn pool2d_output_dim_formula_with_padding() {
    // (4 + 2*1 - 3) / 1 + 1 = 4
    let t = DynTensor::new(&[0.0f32; 16], &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let result = t.max_pool2d(3, 1, 1).unwrap();
    assert_eq!(result.dims(), &[1, 1, 4, 4]);
}

#[test]
fn pool2d_output_dim_formula_large_stride() {
    // (8 + 0 - 2) / 3 + 1 = 3
    let t = DynTensor::new(&vec![0.0f32; 64], &[1, 1, 8, 8], &Device::Cpu).unwrap();
    let result = t.avg_pool2d(2, 3, 0).unwrap();
    assert_eq!(result.dims(), &[1, 1, 3, 3]);
}

/// Adaptive output dimension is exactly the requested size.
#[test]
fn pool2d_output_dim_adaptive_exact() {
    let sizes = [(1, 1), (2, 2), (3, 3), (5, 7), (1, 4)];
    for (oh, ow) in sizes {
        let t = DynTensor::new(
            &vec![0.0f32; 2 * 3 * 14 * 14],
            &[2, 3, 14, 14],
            &Device::Cpu,
        )
        .unwrap();
        let result = t.adaptive_avg_pool2d(oh, ow).unwrap();
        assert_eq!(
            result.dims(),
            &[2, 3, oh, ow],
            "adaptive output should be [{oh}, {ow}]"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 10. MaxPool2d with negative values
// ═══════════════════════════════════════════════════════════════════════

/// MaxPool2d correctly handles all-negative inputs.
#[test]
fn pool2d_max_pool_all_negative() {
    init();
    let data: Vec<f32> = (0..16).map(|i| -(i as f32) - 1.0).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let cpu_result = cpu.max_pool2d(2, 2, 0).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();
    // All values are negative, but the max of each window is the least negative.
    for e in &expected {
        assert!(*e < 0.0, "all outputs should be negative, got {e}");
    }

    let gpu = DynTensor::new(&data, &[1, 1, 4, 4], &Device::metal()).unwrap();
    let gpu_result = gpu.max_pool2d(2, 2, 0).unwrap();
    assert_gpu_vals(&gpu_result, &expected, 1e-6, "max_pool all negative");
}

/// MaxPool2d with mixed positive/negative values including zeros.
#[test]
fn pool2d_max_pool_mixed_sign() {
    init();
    #[rustfmt::skip]
    let data: Vec<f32> = vec![
        -5.0, 0.0,  3.0, -1.0,
         2.0, -4.0, 0.0,  7.0,
         0.0,  1.0, -2.0, 0.0,
        -3.0,  6.0,  0.0, -8.0,
    ];
    let cpu = DynTensor::new(&data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let cpu_result = cpu.max_pool2d(2, 2, 0).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[1, 1, 4, 4], &Device::metal()).unwrap();
    let gpu_result = gpu.max_pool2d(2, 2, 0).unwrap();
    assert_gpu_vals(&gpu_result, &expected, 1e-6, "max_pool mixed sign");
}

/// Larger pooling scenario: [2, 8, 16, 16] -> kernel=4, stride=4, padding=0.
#[test]
fn pool2d_large_multi_channel_batched() {
    init();
    let n = 2 * 8 * 16 * 16;
    let data: Vec<f32> = (0..n)
        .map(|i| ((i * 17 + 3) % 200) as f32 * 0.01 - 1.0)
        .collect();

    let cpu = DynTensor::new(&data, &[2, 8, 16, 16], &Device::Cpu).unwrap();
    let cpu_max = cpu.max_pool2d(4, 4, 0).unwrap();
    let cpu_avg = cpu.avg_pool2d(4, 4, 0).unwrap();
    assert_eq!(cpu_max.dims(), &[2, 8, 4, 4]);
    assert_eq!(cpu_avg.dims(), &[2, 8, 4, 4]);

    let expected_max = cpu_max.to_flat_vec::<f32>().unwrap();
    let expected_avg = cpu_avg.to_flat_vec::<f32>().unwrap();

    let gpu = DynTensor::new(&data, &[2, 8, 16, 16], &Device::metal()).unwrap();
    let gpu_max = gpu.max_pool2d(4, 4, 0).unwrap();
    let gpu_avg = gpu.avg_pool2d(4, 4, 0).unwrap();
    assert_gpu_vals(&gpu_max, &expected_max, 1e-5, "large max_pool");
    assert_gpu_vals(&gpu_avg, &expected_avg, 1e-5, "large avg_pool");
}
