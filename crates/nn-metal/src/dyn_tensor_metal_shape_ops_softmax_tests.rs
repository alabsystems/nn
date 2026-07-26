#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU softmax and log_softmax integration tests.
//!
//! Extracted from `dyn_tensor_metal_shape_ops_tests.rs` (#1341).
//! Tests Metal GPU softmax paths including NaN guards (#1326) and
//! +inf divergence fixes (#1339).

use nn_core::dyn_tensor::{softmax_last_dim, DynTensor};
use nn_core::Device;

use crate::test_common::{assert_close, init};

// -- Softmax tests ------------------------------------------------------------

#[test]
fn test_gpu_softmax() {
    init();
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0];
    let cpu = DynTensor::new(&data, &[2, 3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let gpu_result = softmax_last_dim(&gpu).unwrap();
    assert_eq!(gpu_result.dims(), &[2, 3]);
    assert_eq!(gpu_result.device(), Device::metal());

    let cpu_result = softmax_last_dim(&cpu).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-5, "softmax");
}

// -- Log-softmax tests --------------------------------------------------------

#[test]
fn test_gpu_log_softmax() {
    init();
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0];
    let cpu = DynTensor::new(&data, &[2, 3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    // log_softmax along last dimension
    let gpu_result = gpu.log_softmax(nn_core::dyn_tensor::D::Minus1).unwrap();
    assert_eq!(gpu_result.dims(), &[2, 3]);
    assert_eq!(gpu_result.device(), Device::metal());

    let cpu_result = cpu.log_softmax(nn_core::dyn_tensor::D::Minus1).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-5, "log_softmax");
}

#[test]
fn test_gpu_log_softmax_dim0() {
    init();
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let cpu = DynTensor::new(&data, &[2, 3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    // log_softmax along dim 0
    let gpu_result = gpu.log_softmax(0usize).unwrap();
    assert_eq!(gpu_result.dims(), &[2, 3]);

    let cpu_result = cpu.log_softmax(0usize).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-5, "log_softmax_dim0");
}

// -- GPU softmax/log_softmax NaN guard tests (#1326) --------------------------

/// AC1+AC2: GPU softmax on all-neg-inf input produces zeros (not NaN).
/// Tests both the Metal native kernel and decomposed fallback paths.
#[test]
fn test_gpu_softmax_all_neg_inf_produces_zeros() {
    init();
    let data = vec![
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    let cpu = DynTensor::new(&data, &[2, 3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    // softmax_last_dim — hits Metal native kernel path
    let gpu_result = softmax_last_dim(&gpu).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        gpu_vals.iter().all(|v| *v == 0.0),
        "GPU softmax on all-neg-inf should produce zeros, got: {gpu_vals:?}"
    );

    // Also test via DynTensor::softmax method
    let gpu_result2 = gpu.softmax(1usize).unwrap();
    let gpu_vals2 = gpu_result2
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        gpu_vals2.iter().all(|v| *v == 0.0),
        "GPU softmax(dim=1) on all-neg-inf should produce zeros, got: {gpu_vals2:?}"
    );
}

/// AC5: CPU/GPU softmax behavioral parity on masked attention input.
/// Mixed rows: row 0 all-neg-inf (fully masked), row 1 normal.
#[test]
fn test_gpu_softmax_mixed_neg_inf_parity() {
    init();
    let data = vec![
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY, // row 0: all masked
        1.0,
        2.0,
        3.0, // row 1: normal
    ];
    let cpu = DynTensor::new(&data, &[2, 3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let cpu_result = softmax_last_dim(&cpu).unwrap();
    let gpu_result = softmax_last_dim(&gpu).unwrap();

    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Row 0: both should produce zeros
    assert!(
        cpu_vals[..3].iter().all(|v| *v == 0.0),
        "CPU masked row should be zeros: {:?}",
        &cpu_vals[..3]
    );
    assert!(
        gpu_vals[..3].iter().all(|v| *v == 0.0),
        "GPU masked row should be zeros: {:?}",
        &gpu_vals[..3]
    );

    // Row 1: GPU should match CPU
    assert_close(
        &gpu_vals[3..6],
        &cpu_vals[3..6],
        1e-5,
        "softmax_mixed_parity",
    );
}

/// AC3+AC4: GPU log_softmax on all-neg-inf produces -inf (not NaN).
#[test]
fn test_gpu_log_softmax_all_neg_inf_produces_neg_inf() {
    init();
    let data = vec![
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    let cpu = DynTensor::new(&data, &[2, 3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    // log_softmax along last dim — hits Metal native (softmax kernel + log)
    let gpu_result = gpu.log_softmax(nn_core::dyn_tensor::D::Minus1).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        gpu_vals.iter().all(|v| *v == f32::NEG_INFINITY),
        "GPU log_softmax on all-neg-inf should produce -inf, got: {gpu_vals:?}"
    );
}

/// AC5: CPU/GPU log_softmax parity on mixed masked attention input.
#[test]
fn test_gpu_log_softmax_mixed_neg_inf_parity() {
    init();
    let data = vec![
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY, // row 0: all masked
        1.0,
        2.0,
        3.0, // row 1: normal
    ];
    let cpu = DynTensor::new(&data, &[2, 3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let cpu_result = cpu.log_softmax(nn_core::dyn_tensor::D::Minus1).unwrap();
    let gpu_result = gpu.log_softmax(nn_core::dyn_tensor::D::Minus1).unwrap();

    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Row 0: both should produce -inf
    assert!(
        cpu_vals[..3].iter().all(|v| *v == f32::NEG_INFINITY),
        "CPU masked row should be -inf: {:?}",
        &cpu_vals[..3]
    );
    assert!(
        gpu_vals[..3].iter().all(|v| *v == f32::NEG_INFINITY),
        "GPU masked row should be -inf: {:?}",
        &gpu_vals[..3]
    );

    // Row 1: GPU should match CPU
    assert_close(
        &gpu_vals[3..6],
        &cpu_vals[3..6],
        1e-5,
        "log_softmax_mixed_parity",
    );
}

// -- GPU +inf softmax tests (#1339) -------------------------------------------
// Verify CPU/GPU parity for +inf inputs. +inf positions get uniform share of
// probability mass; non-inf positions get 0.

/// GPU softmax with single +inf element should produce 1.0 at that position, 0.0 elsewhere.
#[test]
fn test_gpu_softmax_single_pos_inf() {
    init();
    let data = vec![f32::INFINITY, 1.0, -1.0];
    let cpu = DynTensor::new(&data, &[1, 3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let cpu_result = softmax_last_dim(&cpu).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    // CPU expected: [1.0, 0.0, 0.0]
    assert!(
        (cpu_vals[0] - 1.0).abs() < 1e-6,
        "CPU softmax +inf position should be 1.0, got {}",
        cpu_vals[0]
    );
    assert!(
        cpu_vals[1].abs() < 1e-6 && cpu_vals[2].abs() < 1e-6,
        "CPU non-inf positions should be 0.0, got {:?}",
        &cpu_vals[1..3]
    );

    let gpu_result = softmax_last_dim(&gpu).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        (gpu_vals[0] - 1.0).abs() < 1e-6,
        "GPU softmax +inf position should be 1.0, got {}",
        gpu_vals[0]
    );
    assert!(
        gpu_vals[1].abs() < 1e-6 && gpu_vals[2].abs() < 1e-6,
        "GPU non-inf positions should be 0.0, got {:?}",
        &gpu_vals[1..3]
    );
}

/// GPU softmax with multiple +inf elements should split probability uniformly.
#[test]
fn test_gpu_softmax_multiple_pos_inf() {
    init();
    let data = vec![f32::INFINITY, f32::INFINITY, 1.0];
    let cpu = DynTensor::new(&data, &[1, 3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let cpu_result = softmax_last_dim(&cpu).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    // CPU expected: [0.5, 0.5, 0.0]
    assert!(
        (cpu_vals[0] - 0.5).abs() < 1e-6,
        "CPU softmax two +inf should be 0.5 each, got {}",
        cpu_vals[0]
    );

    let gpu_result = softmax_last_dim(&gpu).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        (gpu_vals[0] - 0.5).abs() < 1e-6,
        "GPU softmax two +inf should be 0.5 each, got {}",
        gpu_vals[0]
    );
    assert!(
        (gpu_vals[1] - 0.5).abs() < 1e-6,
        "GPU softmax second +inf should be 0.5, got {}",
        gpu_vals[1]
    );
    assert!(
        gpu_vals[2].abs() < 1e-6,
        "GPU non-inf position should be 0.0, got {}",
        gpu_vals[2]
    );
}

/// GPU softmax with all +inf should produce uniform distribution.
#[test]
fn test_gpu_softmax_all_pos_inf() {
    init();
    let data = vec![f32::INFINITY; 3];
    let cpu = DynTensor::new(&data, &[1, 3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let cpu_result = softmax_last_dim(&cpu).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    let expected = 1.0 / 3.0;
    for (i, v) in cpu_vals.iter().enumerate() {
        assert!(
            (v - expected).abs() < 1e-6,
            "CPU all-inf softmax[{i}] should be {expected}, got {v}"
        );
    }

    let gpu_result = softmax_last_dim(&gpu).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    for (i, v) in gpu_vals.iter().enumerate() {
        assert!(
            (v - expected).abs() < 1e-6,
            "GPU all-inf softmax[{i}] should be {expected}, got {v}"
        );
    }
}

// BF16/F16 auto-upcast tests (#1813) and +inf log_softmax tests extracted
// to dyn_tensor_metal_shape_ops_softmax_tests_dtype.rs via #[path] submodule.
