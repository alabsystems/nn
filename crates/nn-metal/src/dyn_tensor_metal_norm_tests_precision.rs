// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Chained precision comparison: fused (Welford+Kahan) vs decomposed (two-pass).
//!
//! These tests validate that the fused GPU norm kernels with Kahan-compensated
//! Welford (#2696) produce results matching the decomposed TensorBlockBuilder
//! path used by `PrecisionTier::Strict`. If they pass, the Strict fallback in
//! CompiledKokoro segments 3+4 can be safely removed (#2704), eliminating ~842
//! extra GPU dispatches.
//!
//! Part of #2700, #2704.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{AdaIn, Linear};
use nn_core::Device;

use crate::test_common::init;

/// Shared assertion: compare fused vs decomposed RMS and max element-wise diff.
fn assert_precision_parity(
    label: &str,
    fused: &DynTensor,
    decomposed: &DynTensor,
    rms_tol: f32,
    max_rel_tol: f32,
) {
    let fused_vals = fused.to_flat_vec::<f32>().unwrap();
    let decomposed_vals = decomposed.to_flat_vec::<f32>().unwrap();
    assert_eq!(fused_vals.len(), decomposed_vals.len());

    let rms = |v: &[f32]| -> f32 { (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt() };
    let fused_rms = rms(&fused_vals);
    let decomposed_rms = rms(&decomposed_vals);

    let max_diff: f32 = fused_vals
        .iter()
        .zip(decomposed_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let rms_ratio = if decomposed_rms > 1e-10 {
        fused_rms / decomposed_rms
    } else {
        1.0
    };
    let max_rel = if decomposed_rms > 1e-10 {
        max_diff / decomposed_rms
    } else {
        max_diff
    };

    assert!(
        (1.0 - rms_tol..=1.0 + rms_tol).contains(&rms_ratio),
        "{label}: fused/decomposed RMS ratio {rms_ratio:.4} \
         (fused={fused_rms:.6}, decomposed={decomposed_rms:.6}) exceeds {rms_tol} tolerance."
    );
    assert!(
        max_rel < max_rel_tol,
        "{label}: max relative diff {max_rel:.4} (max_diff={max_diff:.6}) exceeds {max_rel_tol}."
    );
}

/// 48-chained AdaIN+Snake: fused (Welford+Kahan) vs decomposed (two-pass).
///
/// Prerequisite for #2704: validates fused+Kahan matches decomposed precision.
/// Part of #2700, #2704.
#[test]
fn test_adain_snake_fused_vs_decomposed_chained_48() {
    init();
    let (batch, channels, spatial, eps) = (1, 8, 200, 1e-5);

    let data: Vec<f32> = (0..batch * channels * spatial)
        .map(|i| ((i as f32) * 0.013).sin() * 2.0 + ((i as f32) * 0.007).cos())
        .collect();
    let gamma_data: Vec<f32> = (0..batch * channels)
        .map(|i| ((i as f32) * 0.31).sin() * 0.3)
        .collect();
    let beta_data: Vec<f32> = (0..batch * channels)
        .map(|i| ((i as f32) * 0.47).cos() * 0.1)
        .collect();
    let alpha_data: Vec<f32> = (0..channels).map(|i| 1.0 + (i as f32) * 0.5).collect();

    let dev = &Device::metal();
    let gamma = DynTensor::new(&gamma_data, &[batch, channels, 1], dev).unwrap();
    let beta = DynTensor::new(&beta_data, &[batch, channels, 1], dev).unwrap();
    let alpha_flat = DynTensor::new(&alpha_data, &[channels], dev).unwrap();
    let alpha_ranked = DynTensor::new(&alpha_data, &[1, channels, 1], dev).unwrap();

    let mut fused_x = DynTensor::new(&data, &[batch, channels, spatial], dev).unwrap();
    let mut decomposed_x = fused_x.clone();

    for _ in 0..48 {
        fused_x = crate::dyn_tensor_metal::native_adain_snake(
            &fused_x,
            &gamma,
            &beta,
            &alpha_flat,
            eps,
            true,
        )
        .unwrap();
        decomposed_x = crate::dyn_tensor_metal::native_adain_snake_precise(
            &decomposed_x,
            &gamma,
            &beta,
            &alpha_ranked,
            eps,
            true, // Kokoro residual gamma convention
        )
        .unwrap();
    }

    // 5% RMS tolerance, 10% max relative difference.
    assert_precision_parity("AdaIN+Snake 48-chain", &fused_x, &decomposed_x, 0.05, 0.10);
}

/// 48-chained AdaIN+LeakyRelu: fused vs decomposed precision.
///
/// AdaIN+LeakyRelu is used by Stage1ResBlk in FullDecoder.
/// Part of #2700, #2704.
#[test]
fn test_adain_leaky_relu_fused_vs_decomposed_chained_48() {
    init();
    let (batch, channels, spatial, eps) = (1, 8, 200, 1e-5);
    let slope = 0.2; // Kokoro default LeakyReLU slope

    let data: Vec<f32> = (0..batch * channels * spatial)
        .map(|i| ((i as f32) * 0.019).sin() * 1.5 + ((i as f32) * 0.011).cos() * 0.5)
        .collect();
    let gamma_data: Vec<f32> = (0..batch * channels)
        .map(|i| ((i as f32) * 0.41).sin() * 0.2)
        .collect();
    let beta_data: Vec<f32> = (0..batch * channels)
        .map(|i| ((i as f32) * 0.53).cos() * 0.15)
        .collect();

    let dev = &Device::metal();
    let gamma = DynTensor::new(&gamma_data, &[batch, channels, 1], dev).unwrap();
    let beta = DynTensor::new(&beta_data, &[batch, channels, 1], dev).unwrap();

    let mut fused_x = DynTensor::new(&data, &[batch, channels, spatial], dev).unwrap();
    let mut decomposed_x = fused_x.clone();

    for _ in 0..48 {
        fused_x =
            crate::dyn_tensor_metal::native_adain_leaky_relu(&fused_x, &gamma, &beta, eps, slope)
                .unwrap();
        // Decomposed: precise InstanceNorm + manual affine + LeakyRelu.
        // Matches compiled_model_execute_native_fused.rs Strict path.
        let normed =
            crate::dyn_tensor_metal::native_instance_norm_precise(&decomposed_x, eps).unwrap();
        let gamma_normed = normed.mul(&gamma).unwrap();
        let affined = normed.add(&gamma_normed).unwrap().add(&beta).unwrap();
        decomposed_x = affined.leaky_relu(slope).unwrap();
    }

    assert_precision_parity(
        "AdaIN+LeakyRelu 48-chain",
        &fused_x,
        &decomposed_x,
        0.05,
        0.10,
    );
}

// -- CPU-vs-GPU chained precision regression tests (#2700) --------------------

/// Build AdaIn + style tensor on a given device with deterministic data.
fn build_adain_on_device(channels: usize, style_dim: usize, device: &Device) -> (AdaIn, DynTensor) {
    let w_data: Vec<f32> = (0..2 * channels * style_dim)
        .map(|i| ((i as f32) * 0.013 - 0.5).sin() * 0.1)
        .collect();
    let b_data: Vec<f32> = (0..2 * channels)
        .map(|i| (i as f32) * 0.004 - 0.04)
        .collect();
    let style_data: Vec<f32> = (0..style_dim)
        .map(|i| ((i as f32) * 0.07 - 0.35).cos() * 0.3)
        .collect();

    let w = DynTensor::new(&w_data, &[2 * channels, style_dim], device).unwrap();
    let b = DynTensor::new(&b_data, &[2 * channels], device).unwrap();
    let linear = Linear::new(w, Some(b)).unwrap();
    let adain = AdaIn::new(linear, 1e-5).unwrap();
    let style = DynTensor::new(&style_data, &[1, style_dim], device).unwrap();
    (adain, style)
}

/// Compare GPU vs CPU RMS amplitude ratio, assert within `tol` fraction.
fn assert_gpu_cpu_rms_ratio(label: &str, cpu_t: &DynTensor, gpu_t: &DynTensor, tol: f32) {
    let cpu_vals = cpu_t.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_t.to_flat_vec::<f32>().unwrap();
    assert_eq!(cpu_vals.len(), gpu_vals.len());

    let rms = |v: &[f32]| -> f32 { (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt() };
    let cpu_rms = rms(&cpu_vals);
    let gpu_rms = rms(&gpu_vals);

    let ratio = if cpu_rms > 1e-10 {
        gpu_rms / cpu_rms
    } else {
        1.0
    };

    assert!(
        (1.0 - tol..=1.0 + tol).contains(&ratio),
        "{label}: GPU/CPU amplitude ratio {ratio:.4} \
         (GPU RMS={gpu_rms:.6}, CPU RMS={cpu_rms:.6}) exceeds {tol} drift threshold."
    );
}

/// 36-chained AdaIN+Snake: CPU vs GPU precision regression.
///
/// Kokoro Generator chains 36 AdaIN+Snake calls per forward pass
/// (2 per ResBlock × 3 dilations × 3 blocks/stage × 2 stages).
/// This test verifies GPU fused kernel does not drift from CPU over 36 iterations.
///
/// Part of #2700.
#[test]
fn test_adain_snake_chained_36_cpu_vs_gpu() {
    init();
    let (channels, spatial, style_dim) = (8, 256, 16);

    let data: Vec<f32> = (0..channels * spatial)
        .map(|i| ((i as f32) * 0.017).sin() * 0.5)
        .collect();
    let alpha_data: Vec<f32> = (0..channels).map(|i| 1.0 + (i as f32) * 0.5).collect();

    let cpu_dev = Device::Cpu;
    let gpu_dev = Device::metal();

    let (cpu_adain, cpu_style) = build_adain_on_device(channels, style_dim, &cpu_dev);
    let (gpu_adain, gpu_style) = build_adain_on_device(channels, style_dim, &gpu_dev);

    let cpu_alpha = DynTensor::new(&alpha_data, &[1, channels, 1], &cpu_dev).unwrap();
    let gpu_alpha = DynTensor::new(&alpha_data, &[1, channels, 1], &gpu_dev).unwrap();

    let mut cpu_x = DynTensor::new(&data, &[1, channels, spatial], &cpu_dev).unwrap();
    let mut gpu_x = DynTensor::new(&data, &[1, channels, spatial], &gpu_dev).unwrap();

    for _ in 0..36 {
        cpu_x = cpu_adain
            .forward_snake(&cpu_x, &cpu_style, &cpu_alpha)
            .unwrap();
        gpu_x = gpu_adain
            .forward_snake(&gpu_x, &gpu_style, &gpu_alpha)
            .unwrap();
    }

    // 5% tolerance — wider than plain InstanceNorm (2%) because fused
    // affine+Snake activation compounds additional precision drift.
    assert_gpu_cpu_rms_ratio("AdaIN+Snake 36-chain", &cpu_x, &gpu_x, 0.05);
}

/// 10-chained AdaIN+LeakyRelu: CPU vs GPU precision regression.
///
/// Kokoro FullDecoder Stage1ResBlk chains 10 AdaIN+LeakyRelu calls per forward
/// pass (2 per Stage1ResBlk × 5 blocks). This test verifies GPU fused kernel
/// does not drift from CPU over 10 iterations.
///
/// Part of #2700.
#[test]
fn test_adain_leaky_relu_chained_10_cpu_vs_gpu() {
    init();
    let (channels, spatial, style_dim) = (8, 256, 16);
    let slope = 0.2; // Kokoro default LeakyReLU slope

    let data: Vec<f32> = (0..channels * spatial)
        .map(|i| ((i as f32) * 0.019).sin() * 1.5 + ((i as f32) * 0.011).cos() * 0.5)
        .collect();

    let cpu_dev = Device::Cpu;
    let gpu_dev = Device::metal();

    let (cpu_adain, cpu_style) = build_adain_on_device(channels, style_dim, &cpu_dev);
    let (gpu_adain, gpu_style) = build_adain_on_device(channels, style_dim, &gpu_dev);

    let mut cpu_x = DynTensor::new(&data, &[1, channels, spatial], &cpu_dev).unwrap();
    let mut gpu_x = DynTensor::new(&data, &[1, channels, spatial], &gpu_dev).unwrap();

    for _ in 0..10 {
        cpu_x = cpu_adain
            .forward_leaky_relu(&cpu_x, &cpu_style, slope)
            .unwrap();
        gpu_x = gpu_adain
            .forward_leaky_relu(&gpu_x, &gpu_style, slope)
            .unwrap();
    }

    // 5% tolerance as specified in #2700 acceptance criteria.
    assert_gpu_cpu_rms_ratio("AdaIN+LeakyRelu 10-chain", &cpu_x, &gpu_x, 0.05);
}
