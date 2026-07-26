// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`LoraGpuOverlay`] GPU-resident LoRA adapter application.

#![allow(deprecated)]

use crate::lora_overlay::LoraGpuOverlay;
use crate::test_common::{assert_close, init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

/// Create a CPU tensor from a flat f32 vec and shape, then upload to GPU.
fn gpu_tensor(data: Vec<f32>, dims: &[usize]) -> DynTensor {
    let cpu = DynTensor::from_vec(data, dims, &Device::Cpu).unwrap();
    cpu.to_device(&Device::metal()).unwrap()
}

/// Create a CPU tensor from a flat f32 vec and shape.
fn cpu_tensor(data: Vec<f32>, dims: &[usize]) -> DynTensor {
    DynTensor::from_vec(data, dims, &Device::Cpu).unwrap()
}

#[test]
fn test_apply_rank1_known_values() {
    init();
    // A = [[1, 2]]  shape [1, 2]  (rank=1, in=2)
    // B = [[3], [4]] shape [2, 1]  (out=2, rank=1)
    // B @ A = [[3,6],[4,8]]  shape [2, 2]
    // scaling = 0.5
    // scaled = [[1.5, 3.0], [2.0, 4.0]]
    // W = [[10, 20], [30, 40]]
    // W_eff = [[11.5, 23.0], [32.0, 44.0]]
    let a = gpu_tensor(vec![1.0, 2.0], &[1, 2]);
    let b = gpu_tensor(vec![3.0, 4.0], &[2, 1]);
    let w = gpu_tensor(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]);

    let overlay = LoraGpuOverlay::from_tensors(a, b, 0.5).unwrap();
    assert_eq!(overlay.rank(), 1);
    assert_eq!(overlay.scaling(), 0.5);

    let w_eff = overlay.apply(&w).unwrap();
    assert_eq!(w_eff.dims(), &[2, 2]);

    let vals = w_eff
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&vals, &[11.5, 23.0, 32.0, 44.0], 1e-5, "rank1_known");
}

#[test]
fn test_apply_rank4_identity_base() {
    init();
    // rank=4, out=4, in=4
    // A = identity [4, 4]
    // B = identity [4, 4]
    // B @ A = identity
    // scaling = 1.0
    // W = zeros
    // W_eff = identity
    let eye = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let a = gpu_tensor(eye.clone(), &[4, 4]);
    let b = gpu_tensor(eye.clone(), &[4, 4]);
    let w = gpu_tensor(vec![0.0; 16], &[4, 4]);

    let overlay = LoraGpuOverlay::from_tensors(a, b, 1.0).unwrap();
    let w_eff = overlay.apply(&w).unwrap();
    let vals = w_eff
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&vals, &eye, 1e-5, "rank4_identity");
}

#[test]
fn test_swap_produces_different_effective_weight() {
    init();
    let a1 = gpu_tensor(vec![1.0, 0.0], &[1, 2]);
    let b1 = gpu_tensor(vec![1.0, 0.0], &[2, 1]);
    let a2 = gpu_tensor(vec![0.0, 1.0], &[1, 2]);
    let b2 = gpu_tensor(vec![0.0, 1.0], &[2, 1]);
    let w = gpu_tensor(vec![0.0; 4], &[2, 2]);

    let overlay1 = LoraGpuOverlay::from_tensors(a1, b1, 1.0).unwrap();
    let overlay2 = LoraGpuOverlay::from_tensors(a2, b2, 1.0).unwrap();

    let w1 = overlay1.apply(&w).unwrap();
    let w2 = LoraGpuOverlay::swap(&overlay2, &w).unwrap();

    let v1 = w1
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let v2 = w2
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // overlay1 produces [[1,0],[0,0]], overlay2 produces [[0,0],[0,1]]
    assert_close(&v1, &[1.0, 0.0, 0.0, 0.0], 1e-5, "swap_v1");
    assert_close(&v2, &[0.0, 0.0, 0.0, 1.0], 1e-5, "swap_v2");
}

#[test]
fn test_validation_rank0_rejected() {
    init();
    // A [0, 4] and B [4, 0] — rank 0 should be rejected
    let a = cpu_tensor(vec![], &[0, 4]);
    let b = cpu_tensor(vec![], &[4, 0]);
    let result = LoraGpuOverlay::from_tensors(a, b, 1.0);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("rank must be > 0"), "got: {msg}");
}

#[test]
fn test_validation_nan_scaling_rejected() {
    init();
    let a = cpu_tensor(vec![1.0, 2.0], &[1, 2]);
    let b = cpu_tensor(vec![3.0, 4.0], &[2, 1]);
    let result = LoraGpuOverlay::from_tensors(a, b, f32::NAN);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("finite"), "got: {msg}");
}

#[test]
fn test_validation_inf_scaling_rejected() {
    init();
    let a = cpu_tensor(vec![1.0, 2.0], &[1, 2]);
    let b = cpu_tensor(vec![3.0, 4.0], &[2, 1]);
    let result = LoraGpuOverlay::from_tensors(a, b, f32::INFINITY);
    assert!(result.is_err());
}

#[test]
fn test_validation_shape_mismatch_rejected() {
    init();
    // A [2, 4], B [4, 3] — inner dims 2 != 3
    let a = cpu_tensor(vec![0.0; 8], &[2, 4]);
    let b = cpu_tensor(vec![0.0; 12], &[4, 3]);
    let result = LoraGpuOverlay::from_tensors(a, b, 1.0);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("inner rank mismatch"), "got: {msg}");
}

#[test]
fn test_validation_non_2d_rejected() {
    init();
    let a = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
    let b = cpu_tensor(vec![1.0, 2.0], &[2, 1]);
    let result = LoraGpuOverlay::from_tensors(a, b, 1.0);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("2D"), "got: {msg}");
}

#[test]
fn test_apply_base_weight_mismatch_rejected() {
    init();
    // A [2, 4], B [3, 2] — overlay expects W to be [3, 4]
    let a = gpu_tensor(vec![0.0; 8], &[2, 4]);
    let b = gpu_tensor(vec![0.0; 6], &[3, 2]);
    let w = gpu_tensor(vec![0.0; 20], &[4, 5]); // wrong shape

    let overlay = LoraGpuOverlay::from_tensors(a, b, 1.0).unwrap();
    let result = overlay.apply(&w);
    assert!(result.is_err());
}

#[test]
fn test_gpu_cpu_parity() {
    init();
    // Compare GPU LoraGpuOverlay::apply with CPU matmul equivalent
    let a_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let b_data = vec![0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0];
    let w_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
    let scaling = 0.5_f32;

    // CPU path: manual computation
    let a_cpu = cpu_tensor(a_data.clone(), &[2, 4]);
    let b_cpu = cpu_tensor(b_data.clone(), &[6, 2]);
    let w_cpu = cpu_tensor(w_data.clone(), &[6, 4]);
    let ba_cpu = b_cpu.matmul(&a_cpu).unwrap();
    let scaled_cpu = ba_cpu.mul_scalar(f64::from(scaling)).unwrap();
    let expected = w_cpu.add(&scaled_cpu).unwrap();
    let expected_vals = expected.to_flat_vec::<f32>().unwrap();

    // GPU path: LoraGpuOverlay
    let a_gpu = gpu_tensor(a_data, &[2, 4]);
    let b_gpu = gpu_tensor(b_data, &[6, 2]);
    let w_gpu = gpu_tensor(w_data, &[6, 4]);
    let overlay = LoraGpuOverlay::from_tensors(a_gpu, b_gpu, scaling).unwrap();
    let w_eff = overlay.apply(&w_gpu).unwrap();

    let gpu_vals = w_eff
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &expected_vals, 1e-4, "gpu_cpu_parity");
}

#[test]
fn test_large_matrix_simdgroup() {
    init();
    // [512, 768] base weight with rank-4 overlay
    // This should trigger simdgroup matmul (M×N = 512*768 = 393K >> 16K threshold)
    let rank = 4;
    let out = 512;
    let inp = 768;

    let a_data: Vec<f32> = (0..rank * inp).map(|i| (i as f32 * 0.001) % 1.0).collect();
    let b_data: Vec<f32> = (0..out * rank).map(|i| (i as f32 * 0.001) % 1.0).collect();
    let w_data: Vec<f32> = (0..out * inp).map(|i| (i as f32 * 0.0001) % 1.0).collect();

    let a_gpu = gpu_tensor(a_data, &[rank, inp]);
    let b_gpu = gpu_tensor(b_data, &[out, rank]);
    let w_gpu = gpu_tensor(w_data, &[out, inp]);

    let overlay = LoraGpuOverlay::from_tensors(a_gpu, b_gpu, 0.25).unwrap();
    let w_eff = overlay.apply(&w_gpu).unwrap();

    assert_eq!(w_eff.dims(), &[out, inp]);
    assert_eq!(w_eff.device(), Device::metal());

    // Verify non-NaN — detailed numeric check would require CPU reference
    let cpu_vals = w_eff
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        cpu_vals.iter().all(|v| v.is_finite()),
        "large matrix overlay produced NaN/Inf"
    );
}
