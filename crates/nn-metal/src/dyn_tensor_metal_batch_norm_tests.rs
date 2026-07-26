// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Differential tests: Metal GPU BatchNorm vs CPU reference.
//!
//! Part of #4324.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{BatchNorm, BatchNorm2d, Module};
use nn_core::Device;

use crate::test_common::{assert_close, init};

/// Helper: run BatchNorm on CPU, then on GPU, assert results match.
fn assert_batch_norm_cpu_gpu_match(
    running_mean: &[f32],
    running_var: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f64,
    x_data: &[f32],
    x_shape: &[usize],
    tol: f32,
    label: &str,
) {
    let channels = running_mean.len();

    // CPU reference.
    let rm_cpu = DynTensor::from_vec(running_mean.to_vec(), &[channels], &Device::Cpu).unwrap();
    let rv_cpu = DynTensor::from_vec(running_var.to_vec(), &[channels], &Device::Cpu).unwrap();
    let w_cpu = weight.map(|w| DynTensor::from_vec(w.to_vec(), &[channels], &Device::Cpu).unwrap());
    let b_cpu = bias.map(|b| DynTensor::from_vec(b.to_vec(), &[channels], &Device::Cpu).unwrap());
    let bn_cpu = BatchNorm::new(rm_cpu, rv_cpu, w_cpu, b_cpu, eps).unwrap();
    let x_cpu = DynTensor::from_vec(x_data.to_vec(), x_shape, &Device::Cpu).unwrap();
    let y_cpu = bn_cpu.forward(&x_cpu).unwrap();
    let cpu_vals = y_cpu.to_flat_vec::<f32>().unwrap();

    // GPU path.
    let rm_gpu = DynTensor::from_vec(running_mean.to_vec(), &[channels], &Device::metal()).unwrap();
    let rv_gpu = DynTensor::from_vec(running_var.to_vec(), &[channels], &Device::metal()).unwrap();
    let w_gpu =
        weight.map(|w| DynTensor::from_vec(w.to_vec(), &[channels], &Device::metal()).unwrap());
    let b_gpu =
        bias.map(|b| DynTensor::from_vec(b.to_vec(), &[channels], &Device::metal()).unwrap());
    let bn_gpu = BatchNorm::new(rm_gpu, rv_gpu, w_gpu, b_gpu, eps).unwrap();
    let x_gpu = DynTensor::from_vec(x_data.to_vec(), x_shape, &Device::metal()).unwrap();
    let y_gpu = bn_gpu.forward(&x_gpu).unwrap();
    let gpu_vals = y_gpu.to_flat_vec::<f32>().unwrap();

    assert_eq!(
        cpu_vals.len(),
        gpu_vals.len(),
        "{label}: output length mismatch"
    );
    assert_close(&gpu_vals, &cpu_vals, tol, label);
}

// -- Basic tests ---------------------------------------------------------------

#[test]
fn test_gpu_batch_norm_identity() {
    init();
    // mean=0, var=1, no affine, eps=1e-5 -> nearly identity
    assert_batch_norm_cpu_gpu_match(
        &[0.0, 0.0],
        &[1.0, 1.0],
        None,
        None,
        1e-5,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[1, 2, 3],
        1e-4,
        "identity_no_affine",
    );
}

#[test]
fn test_gpu_batch_norm_with_affine() {
    init();
    // Non-trivial running stats + affine
    assert_batch_norm_cpu_gpu_match(
        &[0.5, -0.5, 1.0],
        &[2.0, 0.5, 4.0],
        Some(&[1.0, 2.0, 0.5]),
        Some(&[0.0, 1.0, -1.0]),
        1e-5,
        &[1.5, 0.5, 3.0],
        &[1, 3, 1],
        1e-4,
        "affine_3ch",
    );
}

#[test]
fn test_gpu_batch_norm_2d_basic() {
    init();
    // 4D input [B=1, C=2, H=1, W=3]
    assert_batch_norm_cpu_gpu_match(
        &[2.0, 5.0],
        &[1.0, 4.0],
        None,
        None,
        0.0001,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[1, 2, 1, 3],
        1e-3,
        "bn2d_basic",
    );
}

#[test]
fn test_gpu_batch_norm_2d_with_affine() {
    init();
    // Full affine BatchNorm2d: mean=0, var=1, weight=2, bias=10
    assert_batch_norm_cpu_gpu_match(
        &[0.0],
        &[1.0],
        Some(&[2.0]),
        Some(&[10.0]),
        1e-5,
        &[1.0],
        &[1, 1, 1, 1],
        1e-4,
        "bn2d_affine",
    );
}

#[test]
fn test_gpu_batch_norm_batched() {
    init();
    // Batched: [B=2, C=2, H=2, W=2]
    let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    assert_batch_norm_cpu_gpu_match(
        &[0.5, -0.3],
        &[2.0, 1.5],
        Some(&[1.5, 0.8]),
        Some(&[-0.1, 0.2]),
        1e-5,
        &data,
        &[2, 2, 2, 2],
        1e-4,
        "batched_2x2x2x2",
    );
}

#[test]
fn test_gpu_batch_norm_pytorch_parity() {
    init();
    // PyTorch reference from batch_norm.rs tests
    assert_batch_norm_cpu_gpu_match(
        &[0.5, -0.5, 1.0],
        &[2.0, 0.5, 4.0],
        Some(&[1.0, 2.0, 0.5]),
        Some(&[0.0, 1.0, -1.0]),
        1e-5,
        &[1.5, 0.5, 3.0],
        &[1, 3, 1, 1],
        1e-4,
        "pytorch_parity",
    );
}

#[test]
fn test_gpu_batch_norm_2d_module() {
    init();
    // Test via the BatchNorm2d module directly on GPU
    let channels = 3;
    let rm = DynTensor::from_vec(vec![0.5, -0.5, 1.0], &[channels], &Device::metal()).unwrap();
    let rv = DynTensor::from_vec(vec![2.0, 0.5, 4.0], &[channels], &Device::metal()).unwrap();
    let w = DynTensor::from_vec(vec![1.0, 2.0, 0.5], &[channels], &Device::metal()).unwrap();
    let b = DynTensor::from_vec(vec![0.0, 1.0, -1.0], &[channels], &Device::metal()).unwrap();
    let bn = BatchNorm2d::new(channels, rm, rv, Some(w), Some(b), 1e-5).unwrap();

    let x = DynTensor::from_vec(vec![1.5, 0.5, 3.0], &[1, 3, 1, 1], &Device::metal()).unwrap();
    let y = bn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 3, 1, 1]);

    let vals = y.to_flat_vec::<f32>().unwrap();
    // ch0: (1.5-0.5)/sqrt(2+1e-5) * 1.0 + 0.0
    let expected_ch0 = 1.0 / (2.0_f32 + 1e-5).sqrt() * 1.0 + 0.0;
    assert!(
        (vals[0] - expected_ch0).abs() < 1e-4,
        "ch0: expected {expected_ch0}, got {}",
        vals[0]
    );
}

#[test]
fn test_gpu_batch_norm_larger_spatial() {
    init();
    // Larger spatial: [1, 4, 8, 8] = 256 elements
    let data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 0.5).collect();
    assert_batch_norm_cpu_gpu_match(
        &[0.1, -0.2, 0.3, -0.4],
        &[1.0, 2.0, 0.5, 3.0],
        Some(&[1.0, 0.5, 2.0, 1.5]),
        Some(&[0.0, 0.1, -0.1, 0.2]),
        1e-5,
        &data,
        &[1, 4, 8, 8],
        1e-3,
        "larger_spatial_4x8x8",
    );
}

// -- Edge case tests (#4324) --------------------------------------------------

#[test]
fn test_gpu_batch_norm_single_channel() {
    init();
    // Single channel [B=1, C=1, T=5] -- minimal channel count.
    assert_batch_norm_cpu_gpu_match(
        &[2.0],
        &[0.5],
        Some(&[3.0]),
        Some(&[-1.0]),
        1e-5,
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        &[1, 1, 5],
        1e-4,
        "single_channel",
    );
}

#[test]
fn test_gpu_batch_norm_rank2_no_spatial() {
    init();
    // Rank-2 input [B=3, C=2] -- no spatial dims (spatial_size=1).
    // This is the edge case where dims.len() == 2 exactly.
    assert_batch_norm_cpu_gpu_match(
        &[1.0, -1.0],
        &[4.0, 2.0],
        Some(&[0.5, 2.0]),
        Some(&[0.1, -0.1]),
        1e-5,
        &[3.0, 5.0, 7.0, 1.0, -1.0, 0.5],
        &[3, 2],
        1e-4,
        "rank2_no_spatial",
    );
}

#[test]
fn test_gpu_batch_norm_weight_only_no_bias() {
    init();
    // Weight without bias -- tests has_weight=1, has_bias=0 path.
    assert_batch_norm_cpu_gpu_match(
        &[0.0, 0.0],
        &[1.0, 1.0],
        Some(&[2.0, 3.0]),
        None,
        1e-5,
        &[1.0, 2.0, 3.0, 4.0],
        &[1, 2, 2],
        1e-4,
        "weight_only_no_bias",
    );
}

#[test]
fn test_gpu_batch_norm_bias_only_no_weight() {
    init();
    // Bias without weight -- tests has_weight=0, has_bias=1 path.
    assert_batch_norm_cpu_gpu_match(
        &[0.0, 0.0],
        &[1.0, 1.0],
        None,
        Some(&[10.0, 20.0]),
        1e-5,
        &[1.0, 2.0, 3.0, 4.0],
        &[1, 2, 2],
        1e-4,
        "bias_only_no_weight",
    );
}

#[test]
fn test_gpu_batch_norm_large_variance() {
    init();
    // Large variance values -- tests numerical stability.
    assert_batch_norm_cpu_gpu_match(
        &[100.0, -100.0],
        &[1e6, 1e6],
        Some(&[1.0, 1.0]),
        Some(&[0.0, 0.0]),
        1e-5,
        &[100.5, -100.5, 101.0, -101.0],
        &[1, 2, 2],
        1e-3,
        "large_variance",
    );
}

#[test]
fn test_gpu_batch_norm_small_variance() {
    init();
    // Very small variance -- tests rsqrt stability (eps prevents div-by-zero).
    assert_batch_norm_cpu_gpu_match(
        &[0.0],
        &[1e-8],
        None,
        None,
        1e-5,
        &[0.001],
        &[1, 1, 1],
        1e-2,
        "small_variance",
    );
}

#[test]
fn test_gpu_batch_norm_zero_variance() {
    init();
    // Zero variance -- eps prevents division by zero.
    assert_batch_norm_cpu_gpu_match(
        &[5.0],
        &[0.0],
        Some(&[1.0]),
        Some(&[0.0]),
        1e-5,
        &[5.0, 5.1, 4.9],
        &[1, 1, 3],
        1e-3,
        "zero_variance",
    );
}

#[test]
fn test_gpu_batch_norm_5d_input() {
    init();
    // 5D input [B=1, C=2, D=2, H=2, W=2] -- higher-rank spatial dims.
    let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
    assert_batch_norm_cpu_gpu_match(
        &[1.0, 2.0],
        &[1.0, 1.0],
        Some(&[0.5, 2.0]),
        Some(&[0.0, -1.0]),
        1e-5,
        &data,
        &[1, 2, 2, 2, 2],
        1e-3,
        "5d_input",
    );
}

#[test]
fn test_gpu_batch_norm_many_channels() {
    init();
    // 64 channels [B=1, C=64, T=4] -- typical ResNet channel count.
    let channels = 64;
    let spatial = 4;
    let total = channels * spatial;
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01 - 1.0).collect();
    let mean: Vec<f32> = (0..channels).map(|i| i as f32 * 0.1).collect();
    let var: Vec<f32> = (0..channels).map(|i| 1.0 + i as f32 * 0.05).collect();
    let weight: Vec<f32> = (0..channels).map(|i| 0.5 + i as f32 * 0.01).collect();
    let bias: Vec<f32> = (0..channels).map(|i| -0.5 + i as f32 * 0.005).collect();
    assert_batch_norm_cpu_gpu_match(
        &mean,
        &var,
        Some(&weight),
        Some(&bias),
        1e-5,
        &data,
        &[1, channels, spatial],
        1e-3,
        "64_channels",
    );
}

#[test]
fn test_gpu_batch_norm_large_spatial() {
    init();
    // Large spatial: [B=1, C=3, H=32, W=32] = 3072 elements.
    // Tests dispatch with >256 threads (TG_SIZE) per channel.
    let channels = 3;
    let h = 32;
    let w = 32;
    let total = channels * h * w;
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.001 - 1.5).collect();
    assert_batch_norm_cpu_gpu_match(
        &[0.0, 0.5, -0.5],
        &[1.0, 2.0, 0.5],
        Some(&[1.0, 0.5, 2.0]),
        Some(&[0.1, -0.1, 0.0]),
        1e-5,
        &data,
        &[1, channels, h, w],
        1e-3,
        "large_spatial_32x32",
    );
}

#[test]
fn test_gpu_batch_norm_negative_mean_positive_var() {
    init();
    // Negative running mean with positive variance.
    assert_batch_norm_cpu_gpu_match(
        &[-5.0, -10.0],
        &[3.0, 7.0],
        Some(&[1.0, 1.0]),
        Some(&[0.0, 0.0]),
        1e-5,
        &[-4.0, -9.0, -6.0, -11.0],
        &[1, 2, 2],
        1e-4,
        "negative_mean",
    );
}

#[test]
fn test_gpu_batch_norm_large_eps() {
    init();
    // Large epsilon -- should dominate variance in normalization.
    assert_batch_norm_cpu_gpu_match(
        &[0.0, 0.0],
        &[0.01, 0.01],
        Some(&[1.0, 1.0]),
        Some(&[0.0, 0.0]),
        1.0, // Very large eps
        &[1.0, 2.0, 3.0, 4.0],
        &[1, 2, 2],
        1e-4,
        "large_eps",
    );
}

#[test]
fn test_gpu_batch_norm_output_stays_on_gpu() {
    init();
    // Verify the output tensor stays on the GPU device.
    let channels = 2;
    let dev = Device::metal();
    let rm = DynTensor::from_vec(vec![0.0, 0.0], &[channels], &dev).unwrap();
    let rv = DynTensor::from_vec(vec![1.0, 1.0], &[channels], &dev).unwrap();
    let bn = BatchNorm::new(rm, rv, None, None, 1e-5).unwrap();
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2], &dev).unwrap();
    let y = bn.forward(&x).unwrap();
    assert_eq!(y.device(), dev, "output must stay on GPU");
    assert_eq!(y.dims(), &[1, 2, 2], "output shape must match input");
}

#[test]
fn test_gpu_batch_norm_multi_batch() {
    init();
    // Multiple batch items [B=4, C=2, H=3, W=3] = 72 elements.
    // Ensures channel indexing is correct across batch boundaries.
    let data: Vec<f32> = (0..72).map(|i| (i as f32) * 0.05 - 1.8).collect();
    assert_batch_norm_cpu_gpu_match(
        &[0.5, -0.3],
        &[2.0, 1.5],
        Some(&[1.0, 0.5]),
        Some(&[0.0, 0.1]),
        1e-5,
        &data,
        &[4, 2, 3, 3],
        1e-3,
        "multi_batch_4x2x3x3",
    );
}

// -- Validation tests ----------------------------------------------------------

#[test]
fn test_gpu_batch_norm_zero_elements_returns_zeros() {
    init();
    // Zero spatial dimensions: [B=1, C=2, T=0] -> empty tensor.
    let rm = DynTensor::from_vec(vec![0.0, 0.0], &[2], &Device::metal()).unwrap();
    let rv = DynTensor::from_vec(vec![1.0, 1.0], &[2], &Device::metal()).unwrap();
    let bn = BatchNorm::new(rm, rv, None, None, 1e-5).unwrap();
    let x = DynTensor::zeros(&[1, 2, 0], nn_core::DType::F32, &Device::metal()).unwrap();
    let y = bn.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 2, 0]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!(vals.is_empty(), "empty tensor should produce empty output");
}
