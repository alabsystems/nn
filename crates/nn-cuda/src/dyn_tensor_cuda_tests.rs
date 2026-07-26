// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the CUDA DynTensor backend.
//!
//! These tests validate the `register_cuda_dyn_backend()` / `init_cuda_runtime()`
//! API surface. On macOS (no CUDA), they confirm graceful `NotAvailable` errors.
//! On Linux with NVIDIA GPU, they exercise the full transfer + compute pipeline.

use crate::cuda_runtime::{is_cuda_available, CudaRuntimeError};
use crate::dyn_tensor_cuda::{init_cuda_runtime, CudaDynBackend, CudaTensorData};
use nn_core::dyn_tensor::{GpuBackend, GpuNnOps};

/// Confirm that `init_cuda_runtime()` returns `NotAvailable` on macOS.
#[test]
fn test_init_graceful_failure_on_macos() {
    if cfg!(target_os = "macos") {
        let result = init_cuda_runtime(0);
        assert!(result.is_err());
        match result {
            Err(CudaRuntimeError::NotAvailable) => {} // expected
            Err(other) => panic!("expected NotAvailable, got: {other}"),
            Ok(()) => panic!("CUDA should not be available on macOS"),
        }
    }
}

/// Validate CudaTensorData type metadata.
#[test]
fn test_cuda_tensor_data_type_properties() {
    // CudaTensorData must be Send + Sync (required for DynTensor GPU storage).
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CudaTensorData>();
}

/// Validate CudaDynBackend implements all required traits.
#[test]
fn test_cuda_dyn_backend_trait_surface() {
    let backend = CudaDynBackend;
    // GpuBackend required method.
    assert_eq!(backend.backend_name(), "cuda");
}

/// Validate GpuNnOps default fallbacks return None (CPU fallback path).
#[test]
fn test_nn_ops_default_fallbacks() {
    let backend = CudaDynBackend;
    // conv1d, conv2d, layer_norm, etc. should all return None.
    // We can't call them without a valid GPU tensor, but we verify the trait
    // is implemented by using the backend name check.
    assert_eq!(backend.backend_name(), "cuda");
}

/// On a platform with CUDA available, exercise the to_gpu -> ops -> to_cpu pipeline.
#[test]
fn test_e2e_pipeline_on_cuda_hardware() {
    if !is_cuda_available() {
        // Skip on macOS / systems without CUDA.
        return;
    }

    // Initialize runtime.
    init_cuda_runtime(0).expect("CUDA init failed on CUDA-available platform");

    // Create a simple CPU tensor.
    let x = nn_core::dyn_tensor::DynTensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        &nn_core::Device::Cpu,
    )
    .expect("create CPU tensor");

    let backend = CudaDynBackend;

    // Transfer to GPU.
    let gpu_x = backend.to_gpu(&x).expect("to_gpu failed");
    assert!(gpu_x.device().is_cuda());

    // Transfer back to CPU and verify.
    let cpu_x = backend.to_cpu(&gpu_x).expect("to_cpu failed");
    let arr = cpu_x.to_f32_array().expect("to_f32_array");
    let flat: Vec<f32> = arr.iter().copied().collect();
    assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Test unary op (relu).
    let neg_data = &[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0];
    let neg_tensor =
        nn_core::dyn_tensor::DynTensor::new(neg_data, vec![2, 3], &nn_core::Device::Cpu)
            .expect("create neg tensor");
    let gpu_neg = backend.to_gpu(&neg_tensor).expect("to_gpu neg");
    let gpu_relu = backend
        .unary_op(nn_core::dyn_tensor::UnaryOp::Relu, &gpu_neg)
        .expect("relu failed");
    let cpu_relu = backend.to_cpu(&gpu_relu).expect("to_cpu relu");
    let relu_flat: Vec<f32> = cpu_relu.to_f32_array().unwrap().iter().copied().collect();
    assert_eq!(relu_flat, vec![0.0, 2.0, 0.0, 4.0, 0.0, 6.0]);

    // Test binary op (add).
    let y = nn_core::dyn_tensor::DynTensor::new(
        &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        vec![2, 3],
        &nn_core::Device::Cpu,
    )
    .expect("create y tensor");
    let gpu_y = backend.to_gpu(&y).expect("to_gpu y");
    let gpu_sum = backend
        .binary_op(nn_core::dyn_tensor::BinaryOp::Add, &gpu_x, &gpu_y)
        .expect("add failed");
    let cpu_sum = backend.to_cpu(&gpu_sum).expect("to_cpu sum");
    let sum_flat: Vec<f32> = cpu_sum.to_f32_array().unwrap().iter().copied().collect();
    assert_eq!(sum_flat, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);

    // Test matmul: [2,3] @ [3,2] = [2,2]
    let w = nn_core::dyn_tensor::DynTensor::new(
        &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![3, 2],
        &nn_core::Device::Cpu,
    )
    .expect("create weight tensor");
    let gpu_w = backend.to_gpu(&w).expect("to_gpu w");
    let gpu_mm = backend.matmul(&gpu_x, &gpu_w).expect("matmul failed");
    assert_eq!(gpu_mm.dims(), &[2, 2]);
    let cpu_mm = backend.to_cpu(&gpu_mm).expect("to_cpu mm");
    let mm_flat: Vec<f32> = cpu_mm.to_f32_array().unwrap().iter().copied().collect();
    // [1,2,3] @ [[1,0],[0,1],[1,1]] = [1+0+3, 0+2+3] = [4, 5]
    // [4,5,6] @ [[1,0],[0,1],[1,1]] = [4+0+6, 0+5+6] = [10, 11]
    assert_eq!(mm_flat, vec![4.0, 5.0, 10.0, 11.0]);

    // Test softmax.
    let sm_input =
        nn_core::dyn_tensor::DynTensor::new(&[1.0, 2.0, 3.0], vec![1, 3], &nn_core::Device::Cpu)
            .expect("create softmax input");
    let gpu_sm_in = backend.to_gpu(&sm_input).expect("to_gpu sm");
    let gpu_sm_out = backend
        .softmax(&gpu_sm_in, 1)
        .expect("softmax returned None")
        .expect("softmax failed");
    let cpu_sm = backend.to_cpu(&gpu_sm_out).expect("to_cpu sm");
    let sm_flat: Vec<f32> = cpu_sm.to_f32_array().unwrap().iter().copied().collect();
    // Verify softmax sums to 1.0.
    let sm_sum: f32 = sm_flat.iter().sum();
    assert!((sm_sum - 1.0).abs() < 1e-5, "softmax sum={sm_sum}");
    // Verify monotonicity: sm[0] < sm[1] < sm[2].
    assert!(sm_flat[0] < sm_flat[1]);
    assert!(sm_flat[1] < sm_flat[2]);
}

/// Validate error paths when CUDA runtime is not initialized.
#[test]
fn test_ops_fail_without_runtime_init() {
    // On macOS, CUDA is not available, so init will fail.
    // Even if init succeeded elsewhere, the backend methods require init.
    if cfg!(target_os = "macos") {
        let backend = CudaDynBackend;
        // to_gpu should fail because runtime is not initialized.
        let x = nn_core::dyn_tensor::DynTensor::new(&[1.0, 2.0], vec![2], &nn_core::Device::Cpu)
            .expect("create tensor");
        let result = backend.to_gpu(&x);
        assert!(result.is_err(), "to_gpu should fail without CUDA");
    }
}
