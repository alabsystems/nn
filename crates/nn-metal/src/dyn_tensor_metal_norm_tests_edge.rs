#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Edge-case and NaN/Inf GPU normalization tests.
//!
//! Extracted from `dyn_tensor_metal_norm_tests.rs` — tests constant input
//! (var=0), single-element reduction, large values, and NaN/Inf propagation
//! through fused GPU norm kernels.
//!
//! Part of #1290.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{GroupNorm, InstanceNorm, LayerNorm, Module, RmsNorm};
use nn_core::{DType, Device};

use crate::test_common::{assert_gpu_matches_cpu, init};

// -- Edge-case GPU norm tests (proof_coverage) --------------------------------

/// LayerNorm with all-identical input (variance=0, exercises rsqrt(eps) path).
/// When all elements are the same value, (x - mean) = 0 for all x, so the
/// numerically-stable result is bias (or 0 when bias=0).
#[test]
fn test_fused_layer_norm_constant_input() {
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::ones(&[4], DType::F32, dev).unwrap();
            let b = DynTensor::zeros(&[4], DType::F32, dev).unwrap();
            let layer = LayerNorm::new(w, b, 1e-5).unwrap();
            let x = DynTensor::new(&[5.0, 5.0, 5.0, 5.0], &[1, 4], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-5,
        "LayerNorm constant input (var=0)",
    );
}

/// RmsNorm with all-identical input (exercises rsqrt(mean_sq + eps) with
/// mean_sq = val^2 for all elements).
#[test]
fn test_fused_rms_norm_constant_input() {
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::ones(&[4], DType::F32, dev).unwrap();
            let layer = RmsNorm::new(w, 1e-5).unwrap();
            let x = DynTensor::new(&[3.0, 3.0, 3.0, 3.0], &[1, 4], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-5,
        "RmsNorm constant input",
    );
}

/// GroupNorm with all-identical input per channel (variance=0 within groups).
#[test]
fn test_fused_group_norm_constant_input() {
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::ones(&[4], DType::F32, dev).unwrap();
            let b = DynTensor::zeros(&[4], DType::F32, dev).unwrap();
            let layer = GroupNorm::new(2, 4, w, b, 1e-5).unwrap();
            // Each channel is constant but different across channels
            let x =
                DynTensor::new(&[5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0], &[1, 4, 2], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-5,
        "GroupNorm constant input (var=0)",
    );
}

/// LayerNorm with hidden_dim=1 (single-element reduction, threadgroup edge case).
#[test]
fn test_fused_layer_norm_dim1() {
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::ones(&[1], DType::F32, dev).unwrap();
            let b = DynTensor::new(&[2.0], &[1], dev).unwrap();
            let layer = LayerNorm::new(w, b, 1e-5).unwrap();
            let x = DynTensor::new(&[7.0, -3.0, 0.0], &[3, 1], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-5,
        "LayerNorm hidden_dim=1",
    );
}

/// RmsNorm with hidden_dim=1 (single-element mean-square).
#[test]
fn test_fused_rms_norm_dim1() {
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::new(&[0.5], &[1], dev).unwrap();
            let layer = RmsNorm::new(w, 1e-5).unwrap();
            let x = DynTensor::new(&[4.0, -2.0, 0.1], &[3, 1], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-5,
        "RmsNorm hidden_dim=1",
    );
}

/// LayerNorm with large values to detect GPU float accumulation overflow.
#[test]
fn test_fused_layer_norm_large_values() {
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::ones(&[4], DType::F32, dev).unwrap();
            let b = DynTensor::zeros(&[4], DType::F32, dev).unwrap();
            let layer = LayerNorm::new(w, b, 1e-5).unwrap();
            let x = DynTensor::new(&[1e6, -1e6, 5e5, -5e5], &[1, 4], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-3,
        "LayerNorm large values",
    );
}

/// RmsNorm with large values — GPU accumulation of x^2 can overflow differently.
#[test]
fn test_fused_rms_norm_large_values() {
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::ones(&[4], DType::F32, dev).unwrap();
            let layer = RmsNorm::new(w, 1e-5).unwrap();
            let x = DynTensor::new(&[1e4, -1e4, 5e3, -5e3], &[1, 4], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-3,
        "RmsNorm large values",
    );
}

// -- NaN/Inf GPU behavior tests -----------------------------------------------
//
// GPU fused norm kernels silently propagate NaN/Inf through the MSL dispatch
// (no per-element guards in Metal). These tests verify GPU propagation behavior.

/// LayerNorm with NaN input: GPU silently produces NaN output.
#[test]
fn test_fused_layer_norm_nan_propagates_gpu() {
    init();
    let w = DynTensor::ones(&[4], DType::F32, &Device::metal()).unwrap();
    let b = DynTensor::zeros(&[4], DType::F32, &Device::metal()).unwrap();
    let layer = LayerNorm::new(w, b, 1e-5).unwrap();
    let x = DynTensor::new(&[1.0, f32::NAN, 3.0, 4.0], &[1, 4], &Device::metal()).unwrap();
    let out = layer.forward(&x).unwrap();
    let vals = out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    // NaN in mean/var computation poisons all outputs.
    assert!(
        vals.iter().all(|v| v.is_nan()),
        "all outputs should be NaN: {vals:?}"
    );
}

/// RmsNorm with +Inf input: GPU output contains non-finite values.
#[test]
fn test_fused_rms_norm_inf_propagates_gpu() {
    init();
    let w = DynTensor::ones(&[4], DType::F32, &Device::metal()).unwrap();
    let layer = RmsNorm::new(w, 1e-5).unwrap();
    let x = DynTensor::new(&[1.0, f32::INFINITY, 3.0, 4.0], &[1, 4], &Device::metal()).unwrap();
    let out = layer.forward(&x).unwrap();
    let vals = out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    // rms includes Inf^2 = Inf, so rsqrt(Inf) = 0, and x * 0 = 0 for finite,
    // but Inf * 0 = NaN for the Inf element.
    let has_non_finite = vals.iter().any(|v| !v.is_finite());
    assert!(
        has_non_finite,
        "GPU output should contain non-finite values: {vals:?}"
    );
}

/// GroupNorm with all-zero input: variance=0, output should be bias only.
#[test]
fn test_fused_group_norm_all_zeros() {
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::new(&[2.0, 3.0, 0.5, 1.5], &[4], dev).unwrap();
            let b = DynTensor::new(&[1.0, -1.0, 0.5, 2.0], &[4], dev).unwrap();
            let layer = GroupNorm::new(2, 4, w, b, 1e-5).unwrap();
            let x = DynTensor::zeros(&[1, 4, 8], DType::F32, dev).unwrap();
            (Box::new(layer), x)
        },
        1e-5,
        "GroupNorm all-zeros",
    );
}

/// GroupNorm with single spatial element (T=1): degenerate reduction.
#[test]
fn test_fused_group_norm_spatial_dim1() {
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::ones(&[4], DType::F32, dev).unwrap();
            let b = DynTensor::zeros(&[4], DType::F32, dev).unwrap();
            let layer = GroupNorm::new(2, 4, w, b, 1e-5).unwrap();
            let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-5,
        "GroupNorm spatial_dim=1",
    );
}

/// GroupNorm with large values — GPU float accumulation on group reduction.
#[test]
fn test_fused_group_norm_large_values() {
    assert_gpu_matches_cpu(
        |dev| {
            let w = DynTensor::ones(&[4], DType::F32, dev).unwrap();
            let b = DynTensor::zeros(&[4], DType::F32, dev).unwrap();
            let layer = GroupNorm::new(2, 4, w, b, 1e-5).unwrap();
            let x = DynTensor::new(
                &[1e5, -1e5, 5e4, -5e4, 1e4, -1e4, 5e3, -5e3],
                &[1, 4, 2],
                dev,
            )
            .unwrap();
            (Box::new(layer), x)
        },
        1e-3,
        "GroupNorm large values",
    );
}

// -- InstanceNorm edge-case GPU tests (#2040) ---------------------------------

/// InstanceNorm with all-identical input per channel (variance=0).
/// (x - mean) = 0 for all x, result should be all zeros.
#[test]
fn test_fused_instance_norm_constant_input() {
    assert_gpu_matches_cpu(
        |dev| {
            let layer = InstanceNorm::new(1e-5).unwrap();
            // 2 channels, each constant but different across channels
            let x =
                DynTensor::new(&[5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0], &[1, 2, 4], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-5,
        "InstanceNorm constant input (var=0)",
    );
}

/// InstanceNorm with spatial_dim=1: degenerate reduction (single element per channel).
#[test]
fn test_fused_instance_norm_spatial_dim1() {
    assert_gpu_matches_cpu(
        |dev| {
            let layer = InstanceNorm::new(1e-5).unwrap();
            let x = DynTensor::new(&[3.0, 7.0, -2.0, 0.5], &[2, 2, 1], dev).unwrap();
            (Box::new(layer), x)
        },
        1e-5,
        "InstanceNorm spatial_dim=1",
    );
}

/// InstanceNorm with large values — tests GPU float accumulation in variance.
#[test]
fn test_fused_instance_norm_large_values() {
    assert_gpu_matches_cpu(
        |dev| {
            let layer = InstanceNorm::new(1e-5).unwrap();
            let x = DynTensor::new(
                &[1e5, -1e5, 5e4, -5e4, 1e4, -1e4, 5e3, -5e3],
                &[1, 2, 4],
                dev,
            )
            .unwrap();
            (Box::new(layer), x)
        },
        1e-3,
        "InstanceNorm large values",
    );
}

/// InstanceNorm with all-zero input: variance=0, output should be all zeros.
#[test]
fn test_fused_instance_norm_all_zeros() {
    assert_gpu_matches_cpu(
        |dev| {
            let layer = InstanceNorm::new(1e-5).unwrap();
            let x = DynTensor::zeros(&[1, 3, 8], DType::F32, dev).unwrap();
            (Box::new(layer), x)
        },
        1e-5,
        "InstanceNorm all-zeros",
    );
}

/// InstanceNorm with NaN input: GPU fused path produces NaN, caught by
/// `check_output_finite` (Tier 1 layer). Forward returns `NonFiniteData` error.
#[test]
fn test_fused_instance_norm_nan_propagates_gpu() {
    init();
    let layer = InstanceNorm::new(1e-5).unwrap();
    let x = DynTensor::new(
        &[1.0, f32::NAN, 3.0, 4.0, 5.0, 6.0],
        &[1, 2, 3],
        &Device::metal(),
    )
    .unwrap();
    let err = layer.forward(&x).unwrap_err();
    let msg = format!("{err}").to_lowercase();
    assert!(
        msg.contains("instancenorm") && msg.contains("non-finite"),
        "expected NonFiniteData error for InstanceNorm, got: {err}"
    );
}

// -- Precision regression: chained InstanceNorm (#2685) -----------------------

/// 48 chained InstanceNorm layers: verify GPU amplitude drift < 2%.
///
/// Prior to Welford (#2685), naive f32 summation caused 4.4x (340%)
/// amplitude blowup. Kahan-compensated reduction (#2696) keeps drift under 2%.
/// This test guards against precision regressions in the fused GPU kernel.
#[test]
fn test_fused_instance_norm_chained_48_precision() {
    init();
    let layer = InstanceNorm::new(1e-5).unwrap();

    // Input: [1, 8, 256] — 8 channels, spatial dim 256 (realistic).
    let data: Vec<f32> = (0..8 * 256)
        .map(|i| ((i as f32) * 0.017).sin() * 0.5)
        .collect();

    let cpu_dev = &Device::Cpu;
    let gpu_dev = &Device::metal();

    let mut cpu_x = DynTensor::new(&data, &[1, 8, 256], cpu_dev).unwrap();
    let mut gpu_x = DynTensor::new(&data, &[1, 8, 256], gpu_dev).unwrap();

    // Chain 48 InstanceNorm applications.
    for _ in 0..48 {
        cpu_x = layer.forward(&cpu_x).unwrap();
        gpu_x = layer.forward(&gpu_x).unwrap();
    }

    // Compare RMS amplitude between CPU and GPU.
    let cpu_vals = cpu_x.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_x.to_flat_vec::<f32>().unwrap();

    let cpu_rms: f32 = (cpu_vals.iter().map(|v| v * v).sum::<f32>() / cpu_vals.len() as f32).sqrt();
    let gpu_rms: f32 = (gpu_vals.iter().map(|v| v * v).sum::<f32>() / gpu_vals.len() as f32).sqrt();

    // InstanceNorm should keep amplitude near 1.0 regardless of chain length.
    // Allow 2% drift between CPU and GPU.
    // Pre-Welford (#2685): 340%+ drift. Kahan-compensated (#2696): <2%.
    let ratio = if cpu_rms > 1e-10 {
        gpu_rms / cpu_rms
    } else {
        1.0 // Both near zero = no drift
    };

    assert!(
        (0.98..=1.02).contains(&ratio),
        "48-chained InstanceNorm GPU/CPU amplitude ratio {ratio:.4} \
         (GPU RMS={gpu_rms:.6}, CPU RMS={cpu_rms:.6}) exceeds 2% drift. \
         Regression in Kahan-compensated reduction precision (#2696)."
    );
}
