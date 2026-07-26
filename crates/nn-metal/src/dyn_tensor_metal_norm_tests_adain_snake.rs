// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for fused AdaIN+Snake GPU kernel (#2227).
//!
//! Verifies that the fused GPU dispatch (InstanceNorm → affine → Snake)
//! produces results within tolerance of the CPU decomposed path.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{AdaIn, Linear};
use nn_core::Device;

use crate::test_common::{assert_close, init};

/// Run AdaIn::forward_snake on both CPU and GPU, compare results.
fn assert_adain_snake_gpu_matches_cpu(
    channels: usize,
    time: usize,
    style_dim: usize,
    tol: f32,
    label: &str,
) {
    init();

    // Build weight/bias for style_linear: [2*C, style_dim]
    let w_data: Vec<f32> = (0..2 * channels * style_dim)
        .map(|i| ((i as f32) * 0.01 - 0.5).sin() * 0.1)
        .collect();
    let b_data: Vec<f32> = (0..2 * channels)
        .map(|i| (i as f32) * 0.005 - 0.05)
        .collect();

    // Build alpha [1, C, 1] — per-channel Snake parameter (typical range 1..10)
    let alpha_data: Vec<f32> = (0..channels).map(|i| 1.0 + (i as f32) * 0.5).collect();

    // Build input [1, C, T]
    let x_data: Vec<f32> = (0..channels * time)
        .map(|i| ((i as f32) * 0.1 - 1.0).sin())
        .collect();

    // Build style [1, style_dim]
    let style_data: Vec<f32> = (0..style_dim).map(|i| (i as f32) * 0.05 - 0.25).collect();

    let eps = 1e-5;

    // --- CPU path ---
    let cpu = Device::Cpu;
    let cpu_w = DynTensor::new(&w_data, &[2 * channels, style_dim], &cpu).unwrap();
    let cpu_b = DynTensor::new(&b_data, &[2 * channels], &cpu).unwrap();
    let cpu_linear = Linear::new(cpu_w, Some(cpu_b)).unwrap();
    let cpu_adain = AdaIn::new(cpu_linear, eps).unwrap();
    let cpu_x = DynTensor::new(&x_data, &[1, channels, time], &cpu).unwrap();
    let cpu_style = DynTensor::new(&style_data, &[1, style_dim], &cpu).unwrap();
    let cpu_alpha = DynTensor::new(&alpha_data, &[1, channels, 1], &cpu).unwrap();
    let cpu_out = cpu_adain
        .forward_snake(&cpu_x, &cpu_style, &cpu_alpha)
        .unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();

    // --- GPU path ---
    let gpu = Device::metal();
    let gpu_w = DynTensor::new(&w_data, &[2 * channels, style_dim], &gpu).unwrap();
    let gpu_b = DynTensor::new(&b_data, &[2 * channels], &gpu).unwrap();
    let gpu_linear = Linear::new(gpu_w, Some(gpu_b)).unwrap();
    let gpu_adain = AdaIn::new(gpu_linear, eps).unwrap();
    let gpu_x = DynTensor::new(&x_data, &[1, channels, time], &gpu).unwrap();
    let gpu_style = DynTensor::new(&style_data, &[1, style_dim], &gpu).unwrap();
    let gpu_alpha = DynTensor::new(&alpha_data, &[1, channels, 1], &gpu).unwrap();
    let gpu_out = gpu_adain
        .forward_snake(&gpu_x, &gpu_style, &gpu_alpha)
        .unwrap();

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
fn test_fused_adain_snake_basic() {
    // Minimal: 4 channels, 8 time steps, style_dim=8
    assert_adain_snake_gpu_matches_cpu(4, 8, 8, 1e-4, "adain_snake_basic");
}

#[test]
fn test_fused_adain_snake_kokoro_like() {
    // Kokoro-like: 64 channels, 128 time steps, style_dim=128
    assert_adain_snake_gpu_matches_cpu(64, 128, 128, 1e-3, "adain_snake_kokoro_like");
}

#[test]
fn test_fused_adain_snake_single_channel() {
    // Edge case: 1 channel
    assert_adain_snake_gpu_matches_cpu(1, 16, 4, 1e-4, "adain_snake_single_channel");
}

/// Run AdaIn::forward_snake with custom alpha on GPU, verify all outputs are finite.
fn assert_adain_snake_gpu_finite_with_alpha(
    alpha_data: &[f32],
    channels: usize,
    time: usize,
    style_dim: usize,
    label: &str,
) {
    init();

    let w_data: Vec<f32> = (0..2 * channels * style_dim)
        .map(|i| ((i as f32) * 0.01 - 0.5).sin() * 0.1)
        .collect();
    let b_data: Vec<f32> = (0..2 * channels)
        .map(|i| (i as f32) * 0.005 - 0.05)
        .collect();
    let x_data: Vec<f32> = (0..channels * time)
        .map(|i| ((i as f32) * 0.1 - 1.0).sin())
        .collect();
    let style_data: Vec<f32> = (0..style_dim).map(|i| (i as f32) * 0.05 - 0.25).collect();
    let eps = 1e-5;

    let gpu = Device::metal();
    let gpu_w = DynTensor::new(&w_data, &[2 * channels, style_dim], &gpu).unwrap();
    let gpu_b = DynTensor::new(&b_data, &[2 * channels], &gpu).unwrap();
    let gpu_linear = Linear::new(gpu_w, Some(gpu_b)).unwrap();
    let gpu_adain = AdaIn::new(gpu_linear, eps).unwrap();
    let gpu_x = DynTensor::new(&x_data, &[1, channels, time], &gpu).unwrap();
    let gpu_style = DynTensor::new(&style_data, &[1, style_dim], &gpu).unwrap();
    let gpu_alpha = DynTensor::new(alpha_data, &[1, channels, 1], &gpu).unwrap();
    let gpu_out = gpu_adain
        .forward_snake(&gpu_x, &gpu_style, &gpu_alpha)
        .unwrap();

    let vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "{label}: output[{i}] = {v} is not finite");
    }
}

#[test]
fn test_fused_adain_snake_alpha_zero_produces_finite() {
    // All-zero alpha must produce finite output (clamp prevents division by zero).
    let alpha_data = vec![0.0_f32; 4];
    assert_adain_snake_gpu_finite_with_alpha(&alpha_data, 4, 8, 8, "alpha_zero");
}

#[test]
fn test_fused_adain_snake_near_zero_alpha_matches_cpu() {
    // Near-zero alpha: GPU fused path should match CPU path (both clamp).
    let channels = 4;
    let time = 8;
    let style_dim = 8;
    let alpha_data: Vec<f32> = vec![1e-10, 0.0, 1e-12, -1e-5];

    init();

    let w_data: Vec<f32> = (0..2 * channels * style_dim)
        .map(|i| ((i as f32) * 0.01 - 0.5).sin() * 0.1)
        .collect();
    let b_data: Vec<f32> = (0..2 * channels)
        .map(|i| (i as f32) * 0.005 - 0.05)
        .collect();
    let x_data: Vec<f32> = (0..channels * time)
        .map(|i| ((i as f32) * 0.1 - 1.0).sin())
        .collect();
    let style_data: Vec<f32> = (0..style_dim).map(|i| (i as f32) * 0.05 - 0.25).collect();
    let eps = 1e-5;

    // CPU
    let cpu = Device::Cpu;
    let cpu_w = DynTensor::new(&w_data, &[2 * channels, style_dim], &cpu).unwrap();
    let cpu_b = DynTensor::new(&b_data, &[2 * channels], &cpu).unwrap();
    let cpu_linear = Linear::new(cpu_w, Some(cpu_b)).unwrap();
    let cpu_adain = AdaIn::new(cpu_linear, eps).unwrap();
    let cpu_x = DynTensor::new(&x_data, &[1, channels, time], &cpu).unwrap();
    let cpu_style = DynTensor::new(&style_data, &[1, style_dim], &cpu).unwrap();
    let cpu_alpha = DynTensor::new(&alpha_data, &[1, channels, 1], &cpu).unwrap();
    let cpu_out = cpu_adain
        .forward_snake(&cpu_x, &cpu_style, &cpu_alpha)
        .unwrap();
    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();

    // GPU
    let gpu = Device::metal();
    let gpu_w = DynTensor::new(&w_data, &[2 * channels, style_dim], &gpu).unwrap();
    let gpu_b = DynTensor::new(&b_data, &[2 * channels], &gpu).unwrap();
    let gpu_linear = Linear::new(gpu_w, Some(gpu_b)).unwrap();
    let gpu_adain = AdaIn::new(gpu_linear, eps).unwrap();
    let gpu_x = DynTensor::new(&x_data, &[1, channels, time], &gpu).unwrap();
    let gpu_style = DynTensor::new(&style_data, &[1, style_dim], &gpu).unwrap();
    let gpu_alpha = DynTensor::new(&alpha_data, &[1, channels, 1], &gpu).unwrap();
    let gpu_out = gpu_adain
        .forward_snake(&gpu_x, &gpu_style, &gpu_alpha)
        .unwrap();

    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-4, "near_zero_alpha_cpu_gpu");
}
