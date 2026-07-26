// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for fused per-channel Snake GPU kernel (#2226).
//!
//! Verifies that the fused GPU dispatch (`x + (1/alpha) * sin²(alpha * x)`)
//! produces results within tolerance of the CPU decomposed path.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::{assert_close, init};

/// Run DynTensor::snake_tensor on both CPU and GPU, compare results.
fn assert_snake_tensor_gpu_matches_cpu(channels: usize, time: usize, tol: f32, label: &str) {
    init();

    // Build alpha [1, C, 1] — per-channel Snake parameter (typical range 1..10)
    let alpha_data: Vec<f32> = (0..channels).map(|i| 1.0 + (i as f32) * 0.5).collect();

    // Build input [1, C, T]
    let x_data: Vec<f32> = (0..channels * time)
        .map(|i| ((i as f32) * 0.1 - 1.0).sin())
        .collect();

    // --- CPU path ---
    let cpu = Device::Cpu;
    let cpu_x = DynTensor::new(&x_data, &[1, channels, time], &cpu).unwrap();
    let cpu_alpha = DynTensor::new(&alpha_data, &[1, channels, 1], &cpu).unwrap();
    let cpu_out = cpu_x.snake_tensor(&cpu_alpha).unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();

    // --- GPU path ---
    let gpu = Device::metal();
    let gpu_x = DynTensor::new(&x_data, &[1, channels, time], &gpu).unwrap();
    let gpu_alpha = DynTensor::new(&alpha_data, &[1, channels, 1], &gpu).unwrap();
    let gpu_out = gpu_x.snake_tensor(&gpu_alpha).unwrap();

    assert_eq!(
        gpu_out.device(),
        Device::metal(),
        "{label}: output stays on GPU"
    );
    assert_eq!(gpu_out.dims(), cpu_out.dims(), "{label}: shape mismatch");

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, tol, label);
}

#[test]
fn test_fused_snake_tensor_basic() {
    // Minimal: 4 channels, 8 time steps
    assert_snake_tensor_gpu_matches_cpu(4, 8, 1e-5, "snake_tensor_basic");
}

#[test]
fn test_fused_snake_tensor_kokoro_like() {
    // Kokoro-like: 64 channels, 128 time steps
    assert_snake_tensor_gpu_matches_cpu(64, 128, 1e-4, "snake_tensor_kokoro_like");
}

#[test]
fn test_fused_snake_tensor_single_channel() {
    // Edge case: 1 channel
    assert_snake_tensor_gpu_matches_cpu(1, 16, 1e-5, "snake_tensor_single_channel");
}
