#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU convolution shape-op tests (conv1d, conv_transpose1d).
//!
//! Extracted from `dyn_tensor_metal_shape_ops_tests.rs` (#1299).
//! Verifies GPU conv dispatches produce correct results vs CPU reference.
//!
//! ConvTranspose1d dilation/groups tests extracted to
//! `dyn_tensor_metal_shape_ops_conv_transpose_ext_tests.rs` (#1402).

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

use crate::test_common::{assert_close, init};

// -- Conv1d tests -------------------------------------------------------------

#[test]
fn test_gpu_conv1d_basic() {
    init();
    // Input: [1, 2, 8] (batch=1, channels=2, length=8)
    let input_data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    let cpu_input = DynTensor::new(&input_data, &[1, 2, 8], &Device::Cpu).unwrap();

    // Kernel: [3, 2, 3] (out_ch=3, in_ch=2, kernel_size=3)
    let kernel_data: Vec<f32> = (0..18).map(|i| (i as f32 - 9.0) * 0.1).collect();
    let cpu_kernel = DynTensor::new(&kernel_data, &[3, 2, 3], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();

    // stride=1, padding=0, dilation=1, groups=1
    let gpu_result = gpu_input.conv1d(&gpu_kernel, 0, 1, 1, 1).unwrap();
    let cpu_result = cpu_input.conv1d(&cpu_kernel, 0, 1, 1, 1).unwrap();

    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_eq!(gpu_result.device(), Device::metal());

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-4, "conv1d_basic");
}

#[test]
fn test_gpu_conv1d_with_padding() {
    init();
    let input_data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let cpu_input = DynTensor::new(&input_data, &[1, 3, 4], &Device::Cpu).unwrap();

    let kernel_data: Vec<f32> = vec![1.0; 18]; // [2, 3, 3]
    let cpu_kernel = DynTensor::new(&kernel_data, &[2, 3, 3], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();

    // padding=1
    let gpu_result = gpu_input.conv1d(&gpu_kernel, 1, 1, 1, 1).unwrap();
    let cpu_result = cpu_input.conv1d(&cpu_kernel, 1, 1, 1, 1).unwrap();

    assert_eq!(gpu_result.dims(), cpu_result.dims());
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-4, "conv1d_padding");
}

// -- Conv1d batch > 1 tests (regression for GPU batch indexing bug) -----------

#[test]
fn test_gpu_conv1d_batch2() {
    init();
    // Input: [2, 2, 8] (batch=2, channels=2, length=8)
    let input_data: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let cpu_input = DynTensor::new(&input_data, &[2, 2, 8], &Device::Cpu).unwrap();

    // Kernel: [3, 2, 3] (out_ch=3, in_ch=2, kernel_size=3)
    let kernel_data: Vec<f32> = (0..18).map(|i| (i as f32 - 9.0) * 0.1).collect();
    let cpu_kernel = DynTensor::new(&kernel_data, &[3, 2, 3], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();

    let gpu_result = gpu_input.conv1d(&gpu_kernel, 0, 1, 1, 1).unwrap();
    let cpu_result = cpu_input.conv1d(&cpu_kernel, 0, 1, 1, 1).unwrap();

    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_eq!(gpu_result.dims()[0], 2, "output batch must be 2");
    assert_eq!(gpu_result.device(), Device::metal());

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-4, "conv1d_batch2");
}

#[test]
fn test_gpu_conv1d_batch4_groups() {
    init();
    // Input: [4, 4, 6] (batch=4, channels=4, length=6)
    let input_data: Vec<f32> = (0..96).map(|i| (i as f32 * 0.05).sin()).collect();
    let cpu_input = DynTensor::new(&input_data, &[4, 4, 6], &Device::Cpu).unwrap();

    // Kernel: [4, 2, 3] (out_ch=4, in_ch_per_group=2, kernel_size=3), groups=2
    let kernel_data: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 0.05).collect();
    let cpu_kernel = DynTensor::new(&kernel_data, &[4, 2, 3], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();

    // groups=2
    let gpu_result = gpu_input.conv1d(&gpu_kernel, 1, 1, 1, 2).unwrap();
    let cpu_result = cpu_input.conv1d(&cpu_kernel, 1, 1, 1, 2).unwrap();

    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_eq!(gpu_result.dims()[0], 4, "output batch must be 4");

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-4, "conv1d_batch4_groups");
}

// -- ConvTranspose1d GPU tests -----------------------------------------------

#[test]
fn test_gpu_conv_transpose1d_basic() {
    init();
    // Input: [1, 2, 4] (batch=1, in_ch=2, length=4)
    let input_data: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
    let cpu_input = DynTensor::new(&input_data, &[1, 2, 4], &Device::Cpu).unwrap();

    // Kernel: [2, 3, 3] (in_ch=2, out_ch=3, kernel_size=3)
    let kernel_data: Vec<f32> = (0..18).map(|i| (i as f32 - 9.0) * 0.05).collect();
    let cpu_kernel = DynTensor::new(&kernel_data, &[2, 3, 3], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();

    // stride=2, padding=1, output_padding=0, dilation=1, groups=1
    // GPU path handles this via native dispatch.
    let gpu_result = gpu_input
        .conv_transpose1d(&gpu_kernel, 1, 0, 2, 1, 1)
        .unwrap();
    let cpu_result = cpu_input
        .conv_transpose1d(&cpu_kernel, 1, 0, 2, 1, 1)
        .unwrap();

    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_eq!(gpu_result.device(), Device::metal());

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-4, "conv_transpose1d_basic");
}

#[test]
fn test_gpu_conv_transpose1d_stride1() {
    init();
    // Input: [1, 1, 6] (batch=1, in_ch=1, length=6)
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let cpu_input = DynTensor::new(&input_data, &[1, 1, 6], &Device::Cpu).unwrap();

    // Kernel: [1, 1, 3] (in_ch=1, out_ch=1, kernel_size=3)
    let kernel_data: Vec<f32> = vec![1.0, 0.5, 0.25];
    let cpu_kernel = DynTensor::new(&kernel_data, &[1, 1, 3], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();

    // stride=1, padding=0
    let gpu_result = gpu_input
        .conv_transpose1d(&gpu_kernel, 0, 0, 1, 1, 1)
        .unwrap();
    let cpu_result = cpu_input
        .conv_transpose1d(&cpu_kernel, 0, 0, 1, 1, 1)
        .unwrap();

    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_eq!(gpu_result.device(), Device::metal());

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-4, "conv_transpose1d_stride1");
}

#[test]
fn test_gpu_conv_transpose1d_demucs_params() {
    init();
    // Demucs-like params: stride=4, padding=2, kernel_size=8
    // Input: [1, 96, 16] (batch=1, in_ch=96, length=16)
    let input_data: Vec<f32> = (0..1536).map(|i| (i as f32).sin() * 0.1).collect();
    let cpu_input = DynTensor::new(&input_data, &[1, 96, 16], &Device::Cpu).unwrap();

    // Kernel: [96, 48, 8] (in_ch=96, out_ch=48, kernel_size=8)
    let kernel_data: Vec<f32> = (0..36864)
        .map(|i| (i as f32 * 0.001).cos() * 0.01)
        .collect();
    let cpu_kernel = DynTensor::new(&kernel_data, &[96, 48, 8], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();

    // stride=4, padding=2 (Demucs decoder config)
    let gpu_result = gpu_input
        .conv_transpose1d(&gpu_kernel, 2, 0, 4, 1, 1)
        .unwrap();
    let cpu_result = cpu_input
        .conv_transpose1d(&cpu_kernel, 2, 0, 4, 1, 1)
        .unwrap();

    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_eq!(gpu_result.device(), Device::metal());

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-3, "conv_transpose1d_demucs");
}

// -- ConvTranspose1d batch > 1 tests (regression for GPU batch indexing bug) --

#[test]
fn test_gpu_conv_transpose1d_batch2() {
    init();
    // Input: [2, 2, 4] (batch=2, in_ch=2, length=4)
    let input_data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    let cpu_input = DynTensor::new(&input_data, &[2, 2, 4], &Device::Cpu).unwrap();

    // Kernel: [2, 3, 3] (in_ch=2, out_ch=3, kernel_size=3)
    let kernel_data: Vec<f32> = (0..18).map(|i| (i as f32 - 9.0) * 0.05).collect();
    let cpu_kernel = DynTensor::new(&kernel_data, &[2, 3, 3], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();

    // stride=2, padding=1
    let gpu_result = gpu_input
        .conv_transpose1d(&gpu_kernel, 1, 0, 2, 1, 1)
        .unwrap();
    let cpu_result = cpu_input
        .conv_transpose1d(&cpu_kernel, 1, 0, 2, 1, 1)
        .unwrap();

    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_eq!(gpu_result.dims()[0], 2, "output batch must be 2");
    assert_eq!(gpu_result.device(), Device::metal());

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-4, "conv_transpose1d_batch2");
}

/// Passing a 2D tensor to conv1d on GPU should return an error, not panic (#1658).
#[test]
fn test_gpu_conv1d_rejects_2d_input() {
    init();
    let input_2d = DynTensor::new(&[1.0f32; 8], &[2, 4], &Device::Cpu)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let kernel_3d = DynTensor::new(&[1.0f32; 6], &[1, 2, 3], &Device::Cpu)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let err = input_2d.conv1d(&kernel_3d, 0, 1, 1, 1).unwrap_err();
    let msg = err.to_string();
    let msg_lower = msg.to_lowercase();
    assert!(
        msg_lower.contains("3d")
            || msg_lower.contains("rank")
            || msg_lower.contains("dimension")
            || msg_lower.contains("gpu_conv1d"),
        "error should mention rank requirement, got: {msg}"
    );
}

/// Passing a 2D kernel to conv1d on GPU should return an error, not panic (#1658).
#[test]
fn test_gpu_conv1d_rejects_2d_kernel() {
    init();
    let input_3d = DynTensor::new(&[1.0f32; 12], &[1, 2, 6], &Device::Cpu)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let kernel_2d = DynTensor::new(&[1.0f32; 6], &[2, 3], &Device::Cpu)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let err = input_3d.conv1d(&kernel_2d, 0, 1, 1, 1).unwrap_err();
    let msg = err.to_string();
    let msg_lower = msg.to_lowercase();
    assert!(
        msg_lower.contains("3d")
            || msg_lower.contains("rank")
            || msg_lower.contains("dimension")
            || msg_lower.contains("gpu_conv1d"),
        "error should mention rank requirement, got: {msg}"
    );
}

// -- Conv1d im2col + GEMM path tests (#3002) ----------------------------------

/// Parity test: im2col + simdgroup GEMM path vs CPU for Kokoro-like shapes.
///
/// Uses shapes large enough to trigger `should_use_conv1d_gemm` (MIN_GEMM_FLOPS).
/// c_out=256, c_in=256, k=3, l_out=124 → 256 * 768 * 124 ≈ 24M FLOPs.
#[test]
fn test_gpu_conv1d_gemm_kokoro_like() {
    init();
    let c_in = 256;
    let c_out = 256;
    let k = 3;
    let l_in = 126;

    let total_in = c_in * l_in;
    let total_k = c_out * c_in * k;
    let input_data = nn_core::test_prng::rand_f32_vec(42, total_in, -1.0, 1.0);
    let kernel_data = nn_core::test_prng::rand_f32_vec(99, total_k, -0.1, 0.1);

    let cpu_input = DynTensor::new(&input_data, &[1, c_in, l_in], &Device::Cpu).unwrap();
    let cpu_kernel = DynTensor::new(&kernel_data, &[c_out, c_in, k], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();

    // padding=1, stride=1, dilation=1, groups=1
    let gpu_result = gpu_input.conv1d(&gpu_kernel, 1, 1, 1, 1).unwrap();
    let cpu_result = cpu_input.conv1d(&cpu_kernel, 1, 1, 1, 1).unwrap();

    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_eq!(gpu_result.device(), Device::metal());

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    // Tolerance relaxed for GEMM path (F32 accumulation, tiled shared memory).
    assert_close(&gpu_vals, &cpu_vals, 1e-3, "conv1d_gemm_kokoro_like");
}

/// GEMM path with dilation (K=7, dilation=3).
#[test]
fn test_gpu_conv1d_gemm_dilated() {
    init();
    let c_in = 256;
    let c_out = 256;
    let k = 7;
    let l_in = 126;

    let total_in = c_in * l_in;
    let total_k = c_out * c_in * k;
    let input_data = nn_core::test_prng::rand_f32_vec(55, total_in, -1.0, 1.0);
    let kernel_data = nn_core::test_prng::rand_f32_vec(77, total_k, -0.1, 0.1);

    let cpu_input = DynTensor::new(&input_data, &[1, c_in, l_in], &Device::Cpu).unwrap();
    let cpu_kernel = DynTensor::new(&kernel_data, &[c_out, c_in, k], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();

    // padding=9 (=(k-1)*dilation/2 for same), stride=1, dilation=3, groups=1
    let gpu_result = gpu_input.conv1d(&gpu_kernel, 9, 1, 3, 1).unwrap();
    let cpu_result = cpu_input.conv1d(&cpu_kernel, 9, 1, 3, 1).unwrap();

    assert_eq!(gpu_result.dims(), cpu_result.dims());

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    assert_close(&gpu_vals, &cpu_vals, 1e-3, "conv1d_gemm_dilated");
}

/// GEMM path: bias is added via broadcast after GEMM (matches `gpu_conv1d_gemm` step 3).
///
/// The DynTensor `conv1d` API does not take a bias parameter, so we test
/// conv1d + manual bias add. The GPU backend receives bias through
/// `layers::Conv1d::forward()`, which is covered by compiled model tests.
#[test]
fn test_gpu_conv1d_gemm_with_manual_bias() {
    init();
    let c_in = 256;
    let c_out = 128;
    let k = 3;
    let l_in = 64;

    let total_in = c_in * l_in;
    let total_k = c_out * c_in * k;
    let input_data = nn_core::test_prng::rand_f32_vec(10, total_in, -1.0, 1.0);
    let kernel_data = nn_core::test_prng::rand_f32_vec(20, total_k, -0.1, 0.1);
    let bias_data = nn_core::test_prng::rand_f32_vec(30, c_out, -0.5, 0.5);

    let cpu_input = DynTensor::new(&input_data, &[1, c_in, l_in], &Device::Cpu).unwrap();
    let cpu_kernel = DynTensor::new(&kernel_data, &[c_out, c_in, k], &Device::Cpu).unwrap();
    let cpu_bias = DynTensor::new(&bias_data, &[c_out], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();
    let gpu_bias = cpu_bias.to_device(&Device::metal()).unwrap();

    // Conv1d via GEMM path (shapes exceed MIN_GEMM_FLOPS).
    let gpu_conv = gpu_input.conv1d(&gpu_kernel, 1, 1, 1, 1).unwrap();
    let cpu_conv = cpu_input.conv1d(&cpu_kernel, 1, 1, 1, 1).unwrap();

    // Bias: reshape [c_out] → [1, c_out, 1] then broadcast add.
    let gpu_bias_3d = gpu_bias.reshape([1, c_out, 1]).unwrap();
    let cpu_bias_3d = cpu_bias.reshape([1, c_out, 1]).unwrap();
    let gpu_result = gpu_conv.add(&gpu_bias_3d).unwrap();
    let cpu_result = cpu_conv.add(&cpu_bias_3d).unwrap();

    assert_eq!(gpu_result.dims(), cpu_result.dims());

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    assert_close(&gpu_vals, &cpu_vals, 1e-3, "conv1d_gemm_bias");
}

/// BF16 GEMM path: im2col_1d_f16 kernel + simdgroup_matrix f16 GEMM.
///
/// Creates Kokoro-like shapes in BF16 to verify the F16 im2col wiring
/// produces results matching the F32 CPU reference within half-precision tolerance.
#[test]
fn test_gpu_conv1d_gemm_bf16() {
    init();
    let c_in = 256;
    let c_out = 256;
    let k = 3;
    let l_in = 126;

    let total_in = c_in * l_in;
    let total_k = c_out * c_in * k;
    let input_data = nn_core::test_prng::rand_f32_vec(42, total_in, -1.0, 1.0);
    let kernel_data = nn_core::test_prng::rand_f32_vec(99, total_k, -0.1, 0.1);

    // F32 CPU reference
    let cpu_input = DynTensor::new(&input_data, &[1, c_in, l_in], &Device::Cpu).unwrap();
    let cpu_kernel = DynTensor::new(&kernel_data, &[c_out, c_in, k], &Device::Cpu).unwrap();
    let cpu_result = cpu_input.conv1d(&cpu_kernel, 1, 1, 1, 1).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    // BF16 GPU via im2col + simdgroup GEMM
    let bf16_input = cpu_input
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();
    let bf16_kernel = cpu_kernel
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();

    let gpu_result = bf16_input.conv1d(&bf16_kernel, 1, 1, 1, 1).unwrap();
    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_eq!(gpu_result.dtype(), DType::BF16);

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // BF16 tolerance wider than F32 (0.05 vs 0.001) due to half-precision
    // rounding in both im2col and GEMM.
    assert_close(&gpu_vals, &cpu_vals, 0.05, "conv1d_gemm_bf16");
}

// -- Conv1d GEMM F16 occupancy gate (#3315) ------------------------------------

/// Verify that F16 conv1d falls back to naive path when GEMM threadgroup
/// occupancy is too low. Shape chosen to pass MIN_GEMM_FLOPS but produce
/// < 384 threadgroups (the F16 simdgroup regression threshold).
#[test]
fn test_conv1d_gemm_f16_low_occupancy_falls_back() {
    // c_out=32, c_in=32, k=8 → col_rows=256, l_out=8000
    // FLOPs: 32 * 256 * 8000 = 65.5M (passes MIN_GEMM_FLOPS=2M)
    // TGs: ceil(32/32) * ceil(8000/32) * 1 = 1 * 250 = 250 (< 384)
    let in_shape = [1, 32, 8007]; // l_out = (8007 + 0 - 8) / 1 + 1 = 8000
    let k_shape = [32, 32, 8];

    // F32 should still use GEMM — occupancy gate is F16-only.
    assert!(
        super::MetalDynBackend::should_use_conv1d_gemm(&in_shape, &k_shape, 8000, 1, DType::F32),
        "F32 should use GEMM (no occupancy gate)"
    );

    // F16 should fall back — 250 TGs < 384 threshold.
    assert!(
        !super::MetalDynBackend::should_use_conv1d_gemm(&in_shape, &k_shape, 8000, 1, DType::F16),
        "F16 should fall back: 250 TGs < 384"
    );

    // BF16 same behavior as F16.
    assert!(
        !super::MetalDynBackend::should_use_conv1d_gemm(&in_shape, &k_shape, 8000, 1, DType::BF16),
        "BF16 should fall back: 250 TGs < 384"
    );

    // F16 with enough batch to exceed threshold: batch=2 → 500 TGs >= 384.
    let in_shape_b2 = [2, 32, 8007];
    assert!(
        super::MetalDynBackend::should_use_conv1d_gemm(&in_shape_b2, &k_shape, 8000, 1, DType::F16),
        "F16 batch=2 should use GEMM: 500 TGs >= 384"
    );
}

/// Conv1d GEMM with shape that passes MIN_GEMM_FLOPS but fails
/// `should_use_simdgroup` (M*N < 16384). Exercises the naive fallback
/// added in #3315 — verifies GPU↔CPU parity through the naive path.
#[test]
fn test_gpu_conv1d_gemm_low_occupancy_routes_to_naive() {
    init();
    // c_out=128, c_in=16, k=8, l_in=134 → l_out = 134 - 8 + 1 = 127 (pad=0)
    // GEMM FLOPs: 128*128*127 ≈ 2.08M (>= MIN_GEMM_FLOPS=2M, enters GEMM path)
    // M*N: 128*127 = 16,256 (< 16,384, fails should_use_simdgroup → naive)
    let c_in = 16;
    let c_out = 128;
    let k = 8;
    let l_in = 134;

    let total_in = c_in * l_in;
    let total_k = c_out * c_in * k;
    let input_data = nn_core::test_prng::rand_f32_vec(77, total_in, -1.0, 1.0);
    let kernel_data = nn_core::test_prng::rand_f32_vec(88, total_k, -0.1, 0.1);

    let cpu_input = DynTensor::new(&input_data, &[1, c_in, l_in], &Device::Cpu).unwrap();
    let cpu_kernel = DynTensor::new(&kernel_data, &[c_out, c_in, k], &Device::Cpu).unwrap();

    let gpu_input = cpu_input.to_device(&Device::metal()).unwrap();
    let gpu_kernel = cpu_kernel.to_device(&Device::metal()).unwrap();

    let gpu_result = gpu_input.conv1d(&gpu_kernel, 0, 1, 1, 1).unwrap();
    let cpu_result = cpu_input.conv1d(&cpu_kernel, 0, 1, 1, 1).unwrap();

    assert_eq!(gpu_result.dims(), cpu_result.dims());
    assert_eq!(gpu_result.dims(), &[1, c_out, 127]);

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-3, "conv1d_gemm_low_occupancy");
}

// ConvTranspose1d dilation/groups tests extracted to
// dyn_tensor_metal_shape_ops_conv_transpose_ext_tests.rs (#1402).
#[path = "dyn_tensor_metal_shape_ops_conv_transpose_ext_tests.rs"]
mod conv_transpose_ext;
