#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv2d GPU dispatch parity tests (#1291).
//!
//! Validates that Metal GPU conv2d produces results matching CPU conv2d
//! across padding, stride, and multi-channel configurations.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::{assert_gpu_vals, init};

#[test]
fn test_gpu_conv2d_basic() {
    init();
    // Simple 1-channel, 1-filter, 3x3 kernel, no padding.
    // Input: [1, 1, 4, 4], Kernel: [1, 1, 3, 3]
    let input_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let kernel_data = vec![1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0];

    let cpu_input = DynTensor::new(&input_data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let cpu_kernel = DynTensor::new(&kernel_data, &[1, 1, 3, 3], &Device::Cpu).unwrap();
    let cpu_result = cpu_input.conv2d(&cpu_kernel, 0, 1, 1, 1).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_input = DynTensor::new(&input_data, &[1, 1, 4, 4], &Device::metal()).unwrap();
    let gpu_kernel = DynTensor::new(&kernel_data, &[1, 1, 3, 3], &Device::metal()).unwrap();
    let gpu_result = gpu_input.conv2d(&gpu_kernel, 0, 1, 1, 1).unwrap();

    assert_eq!(
        gpu_result.device(),
        Device::metal(),
        "conv2d must stay on GPU"
    );
    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_gpu_vals(&gpu_result, &expected, 1e-4, "conv2d basic");
}

#[test]
fn test_gpu_conv2d_with_padding() {
    init();
    // [1, 1, 3, 3] input, [1, 1, 3, 3] kernel, padding=1 → same spatial size.
    let input_data: Vec<f32> = (1..=9).map(|x| x as f32).collect();
    let kernel_data = vec![1.0; 9]; // all-ones 3x3 kernel = sum of neighborhood

    let cpu_input = DynTensor::new(&input_data, &[1, 1, 3, 3], &Device::Cpu).unwrap();
    let cpu_kernel = DynTensor::new(&kernel_data, &[1, 1, 3, 3], &Device::Cpu).unwrap();
    let cpu_result = cpu_input.conv2d(&cpu_kernel, 1, 1, 1, 1).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_input = DynTensor::new(&input_data, &[1, 1, 3, 3], &Device::metal()).unwrap();
    let gpu_kernel = DynTensor::new(&kernel_data, &[1, 1, 3, 3], &Device::metal()).unwrap();
    let gpu_result = gpu_input.conv2d(&gpu_kernel, 1, 1, 1, 1).unwrap();

    assert_eq!(
        gpu_result.dims(),
        cpu_result.dims(),
        "padding=1 should preserve spatial dims"
    );
    assert_gpu_vals(&gpu_result, &expected, 1e-4, "conv2d padding=1");
}

#[test]
fn test_gpu_conv2d_stride2() {
    init();
    // [1, 1, 6, 6] input, [1, 1, 3, 3] kernel, stride=2.
    let input_data: Vec<f32> = (1..=36).map(|x| x as f32).collect();
    let kernel_data = vec![1.0; 9];

    let cpu_input = DynTensor::new(&input_data, &[1, 1, 6, 6], &Device::Cpu).unwrap();
    let cpu_kernel = DynTensor::new(&kernel_data, &[1, 1, 3, 3], &Device::Cpu).unwrap();
    let cpu_result = cpu_input.conv2d(&cpu_kernel, 0, 2, 1, 1).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_input = DynTensor::new(&input_data, &[1, 1, 6, 6], &Device::metal()).unwrap();
    let gpu_kernel = DynTensor::new(&kernel_data, &[1, 1, 3, 3], &Device::metal()).unwrap();
    let gpu_result = gpu_input.conv2d(&gpu_kernel, 0, 2, 1, 1).unwrap();

    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_gpu_vals(&gpu_result, &expected, 1e-4, "conv2d stride=2");
}

#[test]
fn test_gpu_conv2d_multi_channel() {
    init();
    // [1, 3, 4, 4] input, [2, 3, 3, 3] kernel → [1, 2, 2, 2] output.
    let input_data: Vec<f32> = (0..48).map(|x| (x as f32) * 0.1).collect();
    let kernel_data: Vec<f32> = (0..54).map(|x| (x as f32) * 0.01).collect();

    let cpu_input = DynTensor::new(&input_data, &[1, 3, 4, 4], &Device::Cpu).unwrap();
    let cpu_kernel = DynTensor::new(&kernel_data, &[2, 3, 3, 3], &Device::Cpu).unwrap();
    let cpu_result = cpu_input.conv2d(&cpu_kernel, 0, 1, 1, 1).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_input = DynTensor::new(&input_data, &[1, 3, 4, 4], &Device::metal()).unwrap();
    let gpu_kernel = DynTensor::new(&kernel_data, &[2, 3, 3, 3], &Device::metal()).unwrap();
    let gpu_result = gpu_input.conv2d(&gpu_kernel, 0, 1, 1, 1).unwrap();

    assert_eq!(gpu_result.dims(), &[1, 2, 2, 2]);
    assert_gpu_vals(&gpu_result, &expected, 1e-3, "conv2d multi-channel");
}

// -- Conv2d batch > 1 tests (regression for GPU batch indexing bug) -----------

#[test]
fn test_gpu_conv2d_batch2() {
    init();
    // [2, 1, 4, 4] input, [1, 1, 3, 3] kernel → [2, 1, 2, 2] output.
    let input_data: Vec<f32> = (0..32).map(|x| x as f32 * 0.1).collect();
    let kernel_data = vec![1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0];

    let cpu_input = DynTensor::new(&input_data, &[2, 1, 4, 4], &Device::Cpu).unwrap();
    let cpu_kernel = DynTensor::new(&kernel_data, &[1, 1, 3, 3], &Device::Cpu).unwrap();
    let cpu_result = cpu_input.conv2d(&cpu_kernel, 0, 1, 1, 1).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_input = DynTensor::new(&input_data, &[2, 1, 4, 4], &Device::metal()).unwrap();
    let gpu_kernel = DynTensor::new(&kernel_data, &[1, 1, 3, 3], &Device::metal()).unwrap();
    let gpu_result = gpu_input.conv2d(&gpu_kernel, 0, 1, 1, 1).unwrap();

    assert_eq!(gpu_result.dims(), &[2, 1, 2, 2]);
    assert_eq!(gpu_result.dims()[0], 2, "output batch must be 2");
    assert_gpu_vals(&gpu_result, &expected, 1e-4, "conv2d batch=2");
}

#[test]
fn test_gpu_conv2d_batch3_multichannel() {
    init();
    // [3, 2, 4, 4] input, [4, 2, 3, 3] kernel → [3, 4, 2, 2] output.
    let input_data: Vec<f32> = (0..96).map(|x| (x as f32 * 0.05).sin()).collect();
    let kernel_data: Vec<f32> = (0..72).map(|x| (x as f32 - 36.0) * 0.01).collect();

    let cpu_input = DynTensor::new(&input_data, &[3, 2, 4, 4], &Device::Cpu).unwrap();
    let cpu_kernel = DynTensor::new(&kernel_data, &[4, 2, 3, 3], &Device::Cpu).unwrap();
    let cpu_result = cpu_input.conv2d(&cpu_kernel, 0, 1, 1, 1).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_input = DynTensor::new(&input_data, &[3, 2, 4, 4], &Device::metal()).unwrap();
    let gpu_kernel = DynTensor::new(&kernel_data, &[4, 2, 3, 3], &Device::metal()).unwrap();
    let gpu_result = gpu_input.conv2d(&gpu_kernel, 0, 1, 1, 1).unwrap();

    assert_eq!(gpu_result.dims(), &[3, 4, 2, 2]);
    assert_eq!(gpu_result.dims()[0], 3, "output batch must be 3");
    assert_gpu_vals(&gpu_result, &expected, 1e-3, "conv2d batch=3 multichannel");
}

// -- Conv2d dilation and groups GPU parity tests (#1339 proof_coverage) --------

/// GPU conv2d with dilation=2: [1, 1, 7, 7] input, [1, 1, 3, 3] kernel.
/// Effective kernel size is 5x5 (dilation expands 3x3 kernel to 5x5 footprint).
/// Output: [1, 1, 3, 3].
#[test]
fn test_gpu_conv2d_dilation2() {
    init();
    let input_data: Vec<f32> = (0..49).map(|x| (x as f32) * 0.1).collect();
    let kernel_data = vec![1.0, 0.0, -1.0, 0.5, 0.0, -0.5, 0.25, 0.0, -0.25];

    let cpu_input = DynTensor::new(&input_data, &[1, 1, 7, 7], &Device::Cpu).unwrap();
    let cpu_kernel = DynTensor::new(&kernel_data, &[1, 1, 3, 3], &Device::Cpu).unwrap();
    // conv2d(padding=0, stride=1, dilation=2, groups=1)
    let cpu_result = cpu_input.conv2d(&cpu_kernel, 0, 1, 2, 1).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_input = DynTensor::new(&input_data, &[1, 1, 7, 7], &Device::metal()).unwrap();
    let gpu_kernel = DynTensor::new(&kernel_data, &[1, 1, 3, 3], &Device::metal()).unwrap();
    let gpu_result = gpu_input.conv2d(&gpu_kernel, 0, 1, 2, 1).unwrap();

    assert_eq!(
        gpu_result.dims(),
        cpu_result.dims(),
        "dilation=2 shape mismatch"
    );
    assert_gpu_vals(&gpu_result, &expected, 1e-4, "conv2d dilation=2");
}

/// GPU conv2d with non-depthwise groups: groups=2 with 4 input channels, 6 output channels.
/// Each group processes 2 input channels → 3 output channels.
/// Input: [1, 4, 5, 5], Kernel: [6, 2, 3, 3], groups=2.
#[test]
fn test_gpu_conv2d_grouped_non_depthwise() {
    init();
    let input_data: Vec<f32> = (0..100).map(|x| (x as f32 * 0.07).sin()).collect();
    let kernel_data: Vec<f32> = (0..108).map(|x| (x as f32 - 54.0) * 0.01).collect();

    let cpu_input = DynTensor::new(&input_data, &[1, 4, 5, 5], &Device::Cpu).unwrap();
    let cpu_kernel = DynTensor::new(&kernel_data, &[6, 2, 3, 3], &Device::Cpu).unwrap();
    // groups=2: channels 0-1 → filters 0-2, channels 2-3 → filters 3-5
    let cpu_result = cpu_input.conv2d(&cpu_kernel, 0, 1, 1, 2).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_input = DynTensor::new(&input_data, &[1, 4, 5, 5], &Device::metal()).unwrap();
    let gpu_kernel = DynTensor::new(&kernel_data, &[6, 2, 3, 3], &Device::metal()).unwrap();
    let gpu_result = gpu_input.conv2d(&gpu_kernel, 0, 1, 1, 2).unwrap();

    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_gpu_vals(&gpu_result, &expected, 1e-3, "conv2d grouped non-depthwise");
}

// -- ViT / vision-model Conv2d GPU parity tests (#4320) -----------------------

/// GPU conv2d with ViT patch embedding config: kernel=16, stride=16.
/// Input: [1, 3, 32, 32] (tiny image), Kernel: [64, 3, 16, 16] → [1, 64, 2, 2].
/// This is the standard ViT patch embedding: non-overlapping 16x16 patches
/// projected to hidden_dim. Large kernel + large stride exercises a different
/// MSL dispatch shape than the 3x3 tests above.
#[test]
fn test_gpu_conv2d_vit_patch_embed() {
    init();
    let batch = 1;
    let in_ch = 3;
    let h = 32;
    let w = 32;
    let out_ch = 64;
    let k = 16;
    let stride = 16;

    let input_data: Vec<f32> = (0..(batch * in_ch * h * w))
        .map(|i| ((i as f32) * 0.001).sin())
        .collect();
    // Pseudo-random kernel weights (deterministic via index).
    let kernel_data: Vec<f32> = (0..(out_ch * in_ch * k * k))
        .map(|i| ((i as f32) * 0.0037).cos() * 0.02)
        .collect();

    let cpu_input = DynTensor::new(&input_data, &[batch, in_ch, h, w], &Device::Cpu).unwrap();
    let cpu_kernel =
        DynTensor::new(&kernel_data, &[out_ch, in_ch, k, k], &Device::Cpu).unwrap();
    let cpu_result = cpu_input.conv2d(&cpu_kernel, 0, stride, 1, 1).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_input =
        DynTensor::new(&input_data, &[batch, in_ch, h, w], &Device::metal()).unwrap();
    let gpu_kernel =
        DynTensor::new(&kernel_data, &[out_ch, in_ch, k, k], &Device::metal()).unwrap();
    let gpu_result = gpu_input.conv2d(&gpu_kernel, 0, stride, 1, 1).unwrap();

    assert_eq!(gpu_result.dims(), &[batch, out_ch, 2, 2]);
    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_gpu_vals(&gpu_result, &expected, 1e-3, "conv2d vit patch embed 16x16");
}

/// GPU conv2d with bias via layers::Conv2d layer — exercises the full GPU dispatch
/// path through `MetalDynBackend::conv2d` with `bias: Some(...)`.
/// This is critical for vision models where Conv2d patch embedding always has bias.
#[test]
fn test_gpu_conv2d_with_bias() {
    use nn_core::layers::{Conv2d, Conv2dConfig, Module};

    init();
    let in_ch = 3;
    let out_ch = 8;
    let k = 3;

    let weight_data: Vec<f32> = (0..(out_ch * in_ch * k * k))
        .map(|i| ((i as f32) * 0.013).sin() * 0.1)
        .collect();
    let bias_data: Vec<f32> = (0..out_ch).map(|i| (i as f32) * 0.1 - 0.4).collect();
    let input_data: Vec<f32> = (0..48).map(|i| (i as f32) * 0.05).collect();

    // CPU reference with bias.
    let cpu_w = DynTensor::new(&weight_data, &[out_ch, in_ch, k, k], &Device::Cpu).unwrap();
    let cpu_b = DynTensor::new(&bias_data, &[out_ch], &Device::Cpu).unwrap();
    let cpu_layer =
        Conv2d::new(cpu_w, Some(cpu_b), Conv2dConfig::new(1, 1, 1)).unwrap();
    let cpu_input = DynTensor::new(&input_data, &[1, in_ch, 4, 4], &Device::Cpu).unwrap();
    let cpu_result = cpu_layer.forward(&cpu_input).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    // GPU with bias.
    let gpu_w =
        DynTensor::new(&weight_data, &[out_ch, in_ch, k, k], &Device::metal()).unwrap();
    let gpu_b = DynTensor::new(&bias_data, &[out_ch], &Device::metal()).unwrap();
    let gpu_layer =
        Conv2d::new(gpu_w, Some(gpu_b), Conv2dConfig::new(1, 1, 1)).unwrap();
    let gpu_input = DynTensor::new(&input_data, &[1, in_ch, 4, 4], &Device::metal()).unwrap();
    let gpu_result = gpu_layer.forward(&gpu_input).unwrap();

    assert_eq!(gpu_result.device(), Device::metal(), "conv2d+bias stays on GPU");
    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_gpu_vals(&gpu_result, &expected, 1e-3, "conv2d with bias");
}

/// GPU conv2d ViT patch embedding with bias via layers::Conv2d.
/// Input: [1, 3, 224, 224] (standard ViT input), Kernel: [768, 3, 16, 16], stride=16.
/// Output: [1, 768, 14, 14] → 196 patches projected to hidden_dim=768.
/// Tests the exact shape used by ViT-Base/ViT-Large in production dpdf models.
#[test]
fn test_gpu_conv2d_vit_patch_embed_224() {
    use nn_core::layers::{Conv2d, Conv2dConfig, Module};

    init();
    let batch = 1;
    let in_ch = 3;
    let h = 224;
    let w = 224;
    let hidden = 768;
    let k = 16;
    let stride = 16;

    // Use small deterministic values to avoid fp32 accumulation divergence.
    let input_data: Vec<f32> = (0..(batch * in_ch * h * w))
        .map(|i| ((i as f32) * 0.00007).sin() * 0.5)
        .collect();
    let weight_data: Vec<f32> = (0..(hidden * in_ch * k * k))
        .map(|i| ((i as f32) * 0.000013).cos() * 0.01)
        .collect();
    let bias_data: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.0001).collect();

    let cpu_w =
        DynTensor::new(&weight_data, &[hidden, in_ch, k, k], &Device::Cpu).unwrap();
    let cpu_b = DynTensor::new(&bias_data, &[hidden], &Device::Cpu).unwrap();
    let cfg = Conv2dConfig::default().with_stride(stride);
    let cpu_layer = Conv2d::new(cpu_w, Some(cpu_b), cfg).unwrap();
    let cpu_input = DynTensor::new(&input_data, &[batch, in_ch, h, w], &Device::Cpu).unwrap();
    let cpu_result = cpu_layer.forward(&cpu_input).unwrap();
    let expected = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_w =
        DynTensor::new(&weight_data, &[hidden, in_ch, k, k], &Device::metal()).unwrap();
    let gpu_b = DynTensor::new(&bias_data, &[hidden], &Device::metal()).unwrap();
    let gpu_layer = Conv2d::new(gpu_w, Some(gpu_b), cfg).unwrap();
    let gpu_input =
        DynTensor::new(&input_data, &[batch, in_ch, h, w], &Device::metal()).unwrap();
    let gpu_result = gpu_layer.forward(&gpu_input).unwrap();

    assert_eq!(gpu_result.device(), Device::metal());
    assert_eq!(gpu_result.dims(), &[batch, hidden, 14, 14]);
    assert_eq!(gpu_result.dims(), cpu_result.dims());
    // Larger tolerance for 768-channel dot product accumulation.
    assert_gpu_vals(
        &gpu_result,
        &expected,
        5e-3,
        "conv2d vit-base patch embed 224x224",
    );
}
