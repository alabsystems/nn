// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fused-vs-decomposed parity tests for NormActivConv1d GPU kernel (#2218 F13).
//!
//! The fused kernel (`gpu_norm_activ_conv1d`) computes InstanceNorm + affine +
//! activation + Conv1d in 2 Metal dispatches. These tests verify that the fused
//! GPU output matches the **CPU decomposed** sequence:
//!
//!   1. `InstanceNorm::forward()` (CPU)
//!   2. `(1 + gamma) * normed + beta` (CPU broadcast)
//!   3. `leaky_relu()` or `snake_tensor()` (CPU)
//!   4. `conv1d()` + bias add (CPU)
//!
//! CPU decomposed is the mathematical ground truth. The fused MSL kernel must
//! reproduce it within tolerance. This catches both MSL codegen bugs and
//! numerical divergence from operation fusion.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{InstanceNorm, Module};
use nn_core::Device;

use crate::test_common::{assert_close, init};

/// Deterministic pseudo-random f32 vector (xorshift32).
fn rand_f32_vec(mut seed: u32, count: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..count)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let u = f64::from(seed) / f64::from(u32::MAX);
            lo + (hi - lo) * u as f32
        })
        .collect()
}

/// CPU decomposed reference for LeakyRelu variant.
fn cpu_decomposed_leaky_relu(
    x: &DynTensor,
    gamma: &DynTensor,
    beta: &DynTensor,
    weight: &DynTensor,
    bias: &DynTensor,
    eps: f64,
    slope: f64,
    padding: usize,
    dilation: usize,
) -> DynTensor {
    let normed = InstanceNorm::new(eps).unwrap().forward(x).unwrap();
    let one_plus_g = gamma.add_scalar(1.0).unwrap();
    let affine = normed.mul(&one_plus_g).unwrap().add(beta).unwrap();
    let activated = affine.leaky_relu(slope).unwrap();
    let conv_out = activated.conv1d(weight, padding, 1, dilation, 1).unwrap();
    let c_out = weight.dims()[0];
    let bias_bc = bias.reshape([1, c_out, 1]).unwrap();
    conv_out.add(&bias_bc).unwrap()
}

/// CPU decomposed reference for Snake variant.
fn cpu_decomposed_snake(
    x: &DynTensor,
    gamma: &DynTensor,
    beta: &DynTensor,
    alpha: &DynTensor,
    weight: &DynTensor,
    bias: &DynTensor,
    eps: f64,
    padding: usize,
    dilation: usize,
) -> DynTensor {
    let normed = InstanceNorm::new(eps).unwrap().forward(x).unwrap();
    let one_plus_g = gamma.add_scalar(1.0).unwrap();
    let affine = normed.mul(&one_plus_g).unwrap().add(beta).unwrap();
    let activated = affine.snake_tensor(alpha).unwrap();
    let conv_out = activated.conv1d(weight, padding, 1, dilation, 1).unwrap();
    let c_out = weight.dims()[0];
    let bias_bc = bias.reshape([1, c_out, 1]).unwrap();
    conv_out.add(&bias_bc).unwrap()
}

/// Compare fused NormActivConv1d (LeakyRelu) against CPU decomposed reference.
fn assert_fused_matches_cpu_leaky_relu(
    batch: usize,
    c_in: usize,
    c_out: usize,
    time: usize,
    kernel_size: usize,
    padding: usize,
    dilation: usize,
    slope: f64,
    tol: f32,
    label: &str,
) {
    init();
    let gpu = Device::metal();
    let cpu = Device::Cpu;
    let eps = 1e-5_f64;
    let seed_base: u32 = 0xF13A_0000 + (batch as u32 * 1000) + (c_in as u32 * 100) + (time as u32);

    let x_data = rand_f32_vec(seed_base, batch * c_in * time, -1.0, 1.0);
    let gamma_data = rand_f32_vec(seed_base + 1, batch * c_in, -0.3, 0.3);
    let beta_data = rand_f32_vec(seed_base + 2, batch * c_in, -0.2, 0.2);
    let w_data = rand_f32_vec(seed_base + 3, c_out * c_in * kernel_size, -0.5, 0.5);
    let b_data = rand_f32_vec(seed_base + 4, c_out, -0.1, 0.1);

    // Fused GPU path: 2 Metal dispatches.
    let gx = DynTensor::new(&x_data, &[batch, c_in, time], &gpu).unwrap();
    let gg = DynTensor::new(&gamma_data, &[batch, c_in, 1], &gpu).unwrap();
    let gb = DynTensor::new(&beta_data, &[batch, c_in, 1], &gpu).unwrap();
    let gw = DynTensor::new(&w_data, &[c_out, c_in, kernel_size], &gpu).unwrap();
    let gbias = DynTensor::new(&b_data, &[c_out], &gpu).unwrap();

    let fused = super::super::MetalDynBackend::gpu_norm_activ_conv1d(
        &gx, &gg, &gb, &gw, &gbias, eps, slope, padding, dilation, None,
    )
    .unwrap();
    let fused_vals = fused.to_device(&cpu).unwrap().to_flat_vec::<f32>().unwrap();

    // CPU decomposed reference.
    let cx = DynTensor::new(&x_data, &[batch, c_in, time], &cpu).unwrap();
    let cg = DynTensor::new(&gamma_data, &[batch, c_in, 1], &cpu).unwrap();
    let cb = DynTensor::new(&beta_data, &[batch, c_in, 1], &cpu).unwrap();
    let cw = DynTensor::new(&w_data, &[c_out, c_in, kernel_size], &cpu).unwrap();
    let cbias = DynTensor::new(&b_data, &[c_out], &cpu).unwrap();

    let reference =
        cpu_decomposed_leaky_relu(&cx, &cg, &cb, &cw, &cbias, eps, slope, padding, dilation);
    let ref_vals = reference.to_flat_vec::<f32>().unwrap();

    assert_eq!(fused.dims(), reference.dims(), "{label}: shape mismatch");
    assert_close(&fused_vals, &ref_vals, tol, label);
}

/// Compare fused NormActivConv1d with residual (LeakyRelu) against CPU decomposed.
///
/// The residual path is used by FusedResBlock phase 2: the fused kernel adds
/// the residual tensor and scales the result by `scale` (typically `1/sqrt(2)`).
/// This exercises the `has_residual=1` MSL branch.
fn assert_fused_residual_matches_cpu(
    batch: usize,
    c_in: usize,
    c_out: usize,
    time: usize,
    kernel_size: usize,
    padding: usize,
    dilation: usize,
    slope: f64,
    scale: f32,
    tol: f32,
    label: &str,
) {
    init();
    let gpu = Device::metal();
    let cpu = Device::Cpu;
    let eps = 1e-5_f64;
    let t_out = time + 2 * padding - dilation * (kernel_size - 1);
    let seed_base: u32 = 0xF13C_0000 + (batch as u32 * 1000) + (c_in as u32 * 100) + (time as u32);

    let x_data = rand_f32_vec(seed_base, batch * c_in * time, -1.0, 1.0);
    let gamma_data = rand_f32_vec(seed_base + 1, batch * c_in, -0.3, 0.3);
    let beta_data = rand_f32_vec(seed_base + 2, batch * c_in, -0.2, 0.2);
    let w_data = rand_f32_vec(seed_base + 3, c_out * c_in * kernel_size, -0.5, 0.5);
    let b_data = rand_f32_vec(seed_base + 4, c_out, -0.1, 0.1);
    let residual_data = rand_f32_vec(seed_base + 5, batch * c_out * t_out, -0.5, 0.5);

    // Fused GPU path with residual.
    let gx = DynTensor::new(&x_data, &[batch, c_in, time], &gpu).unwrap();
    let gg = DynTensor::new(&gamma_data, &[batch, c_in, 1], &gpu).unwrap();
    let gb = DynTensor::new(&beta_data, &[batch, c_in, 1], &gpu).unwrap();
    let gw = DynTensor::new(&w_data, &[c_out, c_in, kernel_size], &gpu).unwrap();
    let gbias = DynTensor::new(&b_data, &[c_out], &gpu).unwrap();
    let gres = DynTensor::new(&residual_data, &[batch, c_out, t_out], &gpu).unwrap();

    let residual_params = super::ResidualParams {
        residual: &gres,
        scale,
    };
    let fused = super::super::MetalDynBackend::gpu_norm_activ_conv1d(
        &gx,
        &gg,
        &gb,
        &gw,
        &gbias,
        eps,
        slope,
        padding,
        dilation,
        Some(residual_params),
    )
    .unwrap();
    let fused_vals = fused.to_device(&cpu).unwrap().to_flat_vec::<f32>().unwrap();

    // CPU decomposed reference: conv_out + residual, then scale.
    let cx = DynTensor::new(&x_data, &[batch, c_in, time], &cpu).unwrap();
    let cg = DynTensor::new(&gamma_data, &[batch, c_in, 1], &cpu).unwrap();
    let cb = DynTensor::new(&beta_data, &[batch, c_in, 1], &cpu).unwrap();
    let cw = DynTensor::new(&w_data, &[c_out, c_in, kernel_size], &cpu).unwrap();
    let cbias = DynTensor::new(&b_data, &[c_out], &cpu).unwrap();
    let cres = DynTensor::new(&residual_data, &[batch, c_out, t_out], &cpu).unwrap();

    let conv_out =
        cpu_decomposed_leaky_relu(&cx, &cg, &cb, &cw, &cbias, eps, slope, padding, dilation);
    let reference = conv_out
        .add(&cres)
        .unwrap()
        .mul_scalar(f64::from(scale))
        .unwrap();
    let ref_vals = reference.to_flat_vec::<f32>().unwrap();

    assert_eq!(fused.dims(), reference.dims(), "{label}: shape mismatch");
    assert_close(&fused_vals, &ref_vals, tol, label);
}

/// Compare fused NormActivConv1d (Snake) against CPU decomposed reference.
fn assert_fused_matches_cpu_snake(
    batch: usize,
    c_in: usize,
    c_out: usize,
    time: usize,
    kernel_size: usize,
    padding: usize,
    dilation: usize,
    tol: f32,
    label: &str,
) {
    init();
    let gpu = Device::metal();
    let cpu = Device::Cpu;
    let eps = 1e-5_f64;
    let seed_base: u32 = 0xF13B_0000 + (batch as u32 * 1000) + (c_in as u32 * 100) + (time as u32);

    let x_data = rand_f32_vec(seed_base, batch * c_in * time, -1.0, 1.0);
    let gamma_data = rand_f32_vec(seed_base + 1, batch * c_in, -0.3, 0.3);
    let beta_data = rand_f32_vec(seed_base + 2, batch * c_in, -0.2, 0.2);
    let alpha_data: Vec<f32> = (0..c_in).map(|i| 1.0 + (i as f32) * 0.5).collect();
    let w_data = rand_f32_vec(seed_base + 3, c_out * c_in * kernel_size, -0.5, 0.5);
    let b_data = rand_f32_vec(seed_base + 4, c_out, -0.1, 0.1);

    // Fused GPU path.
    let gx = DynTensor::new(&x_data, &[batch, c_in, time], &gpu).unwrap();
    let gg = DynTensor::new(&gamma_data, &[batch, c_in, 1], &gpu).unwrap();
    let gb = DynTensor::new(&beta_data, &[batch, c_in, 1], &gpu).unwrap();
    let ga = DynTensor::new(&alpha_data, &[1, c_in, 1], &gpu).unwrap();
    let gw = DynTensor::new(&w_data, &[c_out, c_in, kernel_size], &gpu).unwrap();
    let gbias = DynTensor::new(&b_data, &[c_out], &gpu).unwrap();

    let fused = super::super::MetalDynBackend::gpu_norm_activ_conv1d_snake(
        &gx, &gg, &gb, &ga, &gw, &gbias, eps, padding, dilation, None,
    )
    .unwrap();
    let fused_vals = fused.to_device(&cpu).unwrap().to_flat_vec::<f32>().unwrap();

    // CPU decomposed reference.
    let cx = DynTensor::new(&x_data, &[batch, c_in, time], &cpu).unwrap();
    let cg = DynTensor::new(&gamma_data, &[batch, c_in, 1], &cpu).unwrap();
    let cb = DynTensor::new(&beta_data, &[batch, c_in, 1], &cpu).unwrap();
    let ca = DynTensor::new(&alpha_data, &[1, c_in, 1], &cpu).unwrap();
    let cw = DynTensor::new(&w_data, &[c_out, c_in, kernel_size], &cpu).unwrap();
    let cbias = DynTensor::new(&b_data, &[c_out], &cpu).unwrap();

    let reference = cpu_decomposed_snake(&cx, &cg, &cb, &ca, &cw, &cbias, eps, padding, dilation);
    let ref_vals = reference.to_flat_vec::<f32>().unwrap();

    assert_eq!(fused.dims(), reference.dims(), "{label}: shape mismatch");
    assert_close(&fused_vals, &ref_vals, tol, label);
}

// -- LeakyRelu fused-vs-CPU tests ---------------------------------------------

#[test]
fn test_fused_vs_cpu_norm_conv_leaky_relu_basic() {
    // Small: [1, 4, 16], kernel=3, pad=1
    assert_fused_matches_cpu_leaky_relu(1, 4, 4, 16, 3, 1, 1, 0.2, 5e-4, "lr_basic");
}

#[test]
fn test_fused_vs_cpu_norm_conv_leaky_relu_batched() {
    // Batched: [2, 8, 32], kernel=3, pad=1
    assert_fused_matches_cpu_leaky_relu(2, 8, 8, 32, 3, 1, 1, 0.2, 5e-4, "lr_batched");
}

#[test]
fn test_fused_vs_cpu_norm_conv_leaky_relu_dilated() {
    // Dilated: [1, 4, 16], kernel=3, pad=2, dilation=2
    assert_fused_matches_cpu_leaky_relu(1, 4, 4, 16, 3, 2, 2, 0.2, 5e-4, "lr_dilated");
}

#[test]
fn test_fused_vs_cpu_norm_conv_leaky_relu_asymmetric() {
    // C_out != C_in: [1, 4, 32], 4→8 channels, kernel=5, pad=2
    assert_fused_matches_cpu_leaky_relu(1, 4, 8, 32, 5, 2, 1, 0.2, 5e-4, "lr_asym");
}

#[test]
fn test_fused_vs_cpu_norm_conv_leaky_relu_kokoro_like() {
    // Kokoro-scale: [1, 64, 128], kernel=3, pad=1
    assert_fused_matches_cpu_leaky_relu(1, 64, 64, 128, 3, 1, 1, 0.2, 1e-3, "lr_kokoro");
}

#[test]
fn test_fused_vs_cpu_norm_conv_leaky_relu_large_batch() {
    // Large batch: [4, 16, 64], kernel=3, pad=1
    assert_fused_matches_cpu_leaky_relu(4, 16, 16, 64, 3, 1, 1, 0.2, 1e-3, "lr_large_batch");
}

// -- Snake fused-vs-CPU tests -------------------------------------------------

#[test]
fn test_fused_vs_cpu_norm_conv_snake_basic() {
    // Small: [1, 4, 16], kernel=3, pad=1
    assert_fused_matches_cpu_snake(1, 4, 4, 16, 3, 1, 1, 1e-3, "snake_basic");
}

#[test]
fn test_fused_vs_cpu_norm_conv_snake_batched() {
    // Batched: [2, 8, 32], kernel=3, pad=1
    assert_fused_matches_cpu_snake(2, 8, 8, 32, 3, 1, 1, 1e-3, "snake_batched");
}

#[test]
fn test_fused_vs_cpu_norm_conv_snake_dilated() {
    // Dilated: [1, 4, 16], kernel=3, pad=2, dilation=2
    assert_fused_matches_cpu_snake(1, 4, 4, 16, 3, 2, 2, 1e-3, "snake_dilated");
}

#[test]
fn test_fused_vs_cpu_norm_conv_snake_kokoro_like() {
    // Kokoro-scale: [1, 64, 128], kernel=3, pad=1
    assert_fused_matches_cpu_snake(1, 64, 64, 128, 3, 1, 1, 2e-3, "snake_kokoro");
}

// -- Residual path tests (FusedResBlock phase 2) ------------------------------

#[test]
fn test_fused_residual_basic() {
    // [1, 4, 16] → fused conv + residual, scale=1/sqrt(2)
    let inv_sqrt2 = 1.0_f32 / 2.0_f32.sqrt();
    assert_fused_residual_matches_cpu(1, 4, 4, 16, 3, 1, 1, 0.2, inv_sqrt2, 5e-4, "residual_basic");
}

#[test]
fn test_fused_residual_batched() {
    // [2, 8, 32] → fused conv + residual with batch
    let inv_sqrt2 = 1.0_f32 / 2.0_f32.sqrt();
    assert_fused_residual_matches_cpu(
        2,
        8,
        8,
        32,
        3,
        1,
        1,
        0.2,
        inv_sqrt2,
        1e-3,
        "residual_batched",
    );
}

#[test]
fn test_fused_residual_scale_one() {
    // scale=1.0 (identity scale — no division)
    assert_fused_residual_matches_cpu(1, 4, 4, 16, 3, 1, 1, 0.2, 1.0, 5e-4, "residual_scale_one");
}
