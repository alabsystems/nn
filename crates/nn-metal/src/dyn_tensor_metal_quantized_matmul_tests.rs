// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for quantized matmul GPU dispatch (INT4/INT8 W4A16/W8A16).
//!
//! Tests validate CPU-fallback-path correctness by comparing GPU dispatch
//! results against direct CPU `quantized_matmul`. Phase 2 native MSL kernel
//! tests will verify GPU-only path parity.
//!
//! Part of #3869

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::quantized::{
    weight_quantize_per_group, quantized_matmul, QuantizationConfig,
};
use nn_core::Device;

use crate::dyn_tensor_metal::MetalDynBackend;
use crate::test_common::init;

/// Helper: create a QuantizedTensor from a flat weight slice.
fn make_quantized_weight(
    data: &[f32],
    out_features: usize,
    in_features: usize,
    config: &QuantizationConfig,
) -> nn_core::layers::quantized::QuantizedTensor {
    let weight_cpu =
        DynTensor::from_vec(data.to_vec(), &[out_features, in_features], &Device::Cpu).unwrap();
    weight_quantize_per_group(&weight_cpu, config).unwrap()
}

// -- INT4 basic ---------------------------------------------------------------

#[test]
fn test_quantized_matmul_gpu_int4_basic() {
    init();
    let config = QuantizationConfig::int4(4);

    // Weight [2, 4], input [1, 4] → output [1, 2]
    let weight_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let qt = make_quantized_weight(&weight_data, 2, 4, &config);

    let input_data = vec![1.0, 1.0, 1.0, 1.0];
    let input_gpu = DynTensor::new(&input_data, &[1, 4], &Device::metal()).unwrap();

    let result = MetalDynBackend::gpu_quantized_matmul(&input_gpu, &qt).unwrap();

    // Result should be on GPU
    assert!(result.device().is_gpu(), "result should be on GPU");
    assert_eq!(result.dims(), &[1, 2]);

    // Compare against CPU reference
    let input_cpu = DynTensor::from_vec(input_data, &[1, 4], &Device::Cpu).unwrap();
    let ref_result = quantized_matmul(&input_cpu, &qt).unwrap();
    let ref_vals = ref_result.to_flat_vec::<f32>().unwrap();

    let gpu_vals = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    for (i, (g, r)) in gpu_vals.iter().zip(ref_vals.iter()).enumerate() {
        assert!(
            (g - r).abs() < 1e-4,
            "INT4 basic [{i}]: gpu={g}, ref={r}, diff={}",
            (g - r).abs()
        );
    }
}

// -- INT8 basic ---------------------------------------------------------------

#[test]
fn test_quantized_matmul_gpu_int8_basic() {
    init();
    let config = QuantizationConfig::int8(4);

    // Weight [2, 4], input [1, 4] → output [1, 2]
    let weight_data: Vec<f32> = vec![0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0];
    let qt = make_quantized_weight(&weight_data, 2, 4, &config);

    let input_data = vec![1.0, 2.0, 3.0, 4.0];
    let input_gpu = DynTensor::new(&input_data, &[1, 4], &Device::metal()).unwrap();

    let result = MetalDynBackend::gpu_quantized_matmul(&input_gpu, &qt).unwrap();

    assert!(result.device().is_gpu());
    assert_eq!(result.dims(), &[1, 2]);

    // Compare against CPU
    let input_cpu = DynTensor::from_vec(input_data, &[1, 4], &Device::Cpu).unwrap();
    let ref_result = quantized_matmul(&input_cpu, &qt).unwrap();
    let ref_vals = ref_result.to_flat_vec::<f32>().unwrap();
    let gpu_vals = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    for (i, (g, r)) in gpu_vals.iter().zip(ref_vals.iter()).enumerate() {
        assert!(
            (g - r).abs() < 1e-4,
            "INT8 basic [{i}]: gpu={g}, ref={r}, diff={}",
            (g - r).abs()
        );
    }
}

// -- GPU matches CPU ----------------------------------------------------------

#[test]
fn test_quantized_matmul_gpu_matches_cpu() {
    init();
    let config = QuantizationConfig::int4(8);

    // Random-ish weight [4, 16]
    let weight_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.2).collect();
    let qt = make_quantized_weight(&weight_data, 4, 16, &config);

    // Random-ish input [3, 16]
    let input_data: Vec<f32> = (0..48).map(|i| (i as f32) * 0.05 - 1.2).collect();

    let input_cpu = DynTensor::from_vec(input_data.clone(), &[3, 16], &Device::Cpu).unwrap();
    let input_gpu = DynTensor::new(&input_data, &[3, 16], &Device::metal()).unwrap();

    let cpu_result = quantized_matmul(&input_cpu, &qt).unwrap();
    let gpu_result = MetalDynBackend::gpu_quantized_matmul(&input_gpu, &qt).unwrap();

    assert_eq!(cpu_result.dims(), gpu_result.dims());

    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    for (i, (g, c)) in gpu_vals.iter().zip(cpu_vals.iter()).enumerate() {
        assert!(
            (g - c).abs() < 1e-4,
            "GPU vs CPU [{i}]: gpu={g}, cpu={c}, diff={}",
            (g - c).abs()
        );
    }
}

// -- Group size 128 (standard VLM group) --------------------------------------

#[test]
fn test_quantized_matmul_gpu_group_size_128() {
    init();
    let config = QuantizationConfig::int4(128);

    // Weight [8, 256] (in_features divisible by 128)
    let weight_data: Vec<f32> = (0..2048)
        .map(|i| (i as f32) * 0.01 - 10.24)
        .collect();
    let qt = make_quantized_weight(&weight_data, 8, 256, &config);

    // Input [2, 256]
    let input_data: Vec<f32> = (0..512).map(|i| (i as f32) * 0.02 - 5.12).collect();

    let input_cpu = DynTensor::from_vec(input_data.clone(), &[2, 256], &Device::Cpu).unwrap();
    let input_gpu = DynTensor::new(&input_data, &[2, 256], &Device::metal()).unwrap();

    let cpu_result = quantized_matmul(&input_cpu, &qt).unwrap();
    let gpu_result = MetalDynBackend::gpu_quantized_matmul(&input_gpu, &qt).unwrap();

    assert_eq!(cpu_result.dims(), &[2, 8]);
    assert_eq!(gpu_result.dims(), &[2, 8]);

    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    for (i, (g, c)) in gpu_vals.iter().zip(cpu_vals.iter()).enumerate() {
        assert!(
            (g - c).abs() < 1e-3,
            "group_size_128 [{i}]: gpu={g}, cpu={c}, diff={}",
            (g - c).abs()
        );
    }
}

// -- Large matrix (realistic VLM dimensions) ----------------------------------

#[test]
fn test_quantized_matmul_gpu_large_matrix() {
    init();
    let config = QuantizationConfig::int4(128);

    // Weight [512, 512] — smaller than true VLM (4096x4096) but still realistic
    let n = 512;
    let k = 512;
    let weight_data: Vec<f32> = (0..(n * k))
        .map(|i| (i % 997) as f32 * 0.001 - 0.5)
        .collect();
    let qt = make_quantized_weight(&weight_data, n, k, &config);

    // Input [4, 512]
    let m = 4;
    let input_data: Vec<f32> = (0..(m * k))
        .map(|i| (i % 503) as f32 * 0.002 - 0.5)
        .collect();

    let input_cpu = DynTensor::from_vec(input_data.clone(), &[m, k], &Device::Cpu).unwrap();
    let input_gpu = DynTensor::new(&input_data, &[m, k], &Device::metal()).unwrap();

    let cpu_result = quantized_matmul(&input_cpu, &qt).unwrap();
    let gpu_result = MetalDynBackend::gpu_quantized_matmul(&input_gpu, &qt).unwrap();

    assert_eq!(cpu_result.dims(), &[m, n]);
    assert_eq!(gpu_result.dims(), &[m, n]);

    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Larger tolerance for accumulated error in 512-wide dot products
    for (i, (g, c)) in gpu_vals.iter().zip(cpu_vals.iter()).enumerate() {
        assert!(
            (g - c).abs() < 0.1,
            "large_matrix [{i}]: gpu={g}, cpu={c}, diff={}",
            (g - c).abs()
        );
    }
}

// -- Batched input [B, M, K] --------------------------------------------------

#[test]
fn test_quantized_matmul_gpu_batched() {
    init();
    let config = QuantizationConfig::int4(8);

    // Weight [4, 16]
    let weight_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.2).collect();
    let qt = make_quantized_weight(&weight_data, 4, 16, &config);

    // Input [2, 3, 16] (batch=2, seq=3, features=16)
    let input_data: Vec<f32> = (0..96).map(|i| (i as f32) * 0.05 - 2.4).collect();

    let input_cpu = DynTensor::from_vec(input_data.clone(), &[2, 3, 16], &Device::Cpu).unwrap();
    let input_gpu = DynTensor::new(&input_data, &[2, 3, 16], &Device::metal()).unwrap();

    let cpu_result = quantized_matmul(&input_cpu, &qt).unwrap();
    let gpu_result = MetalDynBackend::gpu_quantized_matmul(&input_gpu, &qt).unwrap();

    assert_eq!(cpu_result.dims(), &[2, 3, 4]);
    assert_eq!(gpu_result.dims(), &[2, 3, 4]);

    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    for (i, (g, c)) in gpu_vals.iter().zip(cpu_vals.iter()).enumerate() {
        assert!(
            (g - c).abs() < 1e-3,
            "batched [{i}]: gpu={g}, cpu={c}, diff={}",
            (g - c).abs()
        );
    }
}

// -- Error cases --------------------------------------------------------------

#[test]
fn test_quantized_matmul_gpu_shape_mismatch() {
    init();
    let config = QuantizationConfig::int4(4);

    let weight_data: Vec<f32> = vec![1.0; 16];
    let qt = make_quantized_weight(&weight_data, 4, 4, &config);

    // Input has wrong last dimension (8 instead of 4)
    let input_gpu = DynTensor::new(&[1.0; 8], &[1, 8], &Device::metal()).unwrap();

    let result = MetalDynBackend::gpu_quantized_matmul(&input_gpu, &qt);
    assert!(result.is_err(), "should error on shape mismatch");
}

#[test]
fn test_quantized_matmul_gpu_empty_input() {
    init();
    let config = QuantizationConfig::int4(4);

    let weight_data: Vec<f32> = vec![1.0; 16];
    let qt = make_quantized_weight(&weight_data, 4, 4, &config);

    // Rank-0 tensor
    let input_gpu = DynTensor::new(&[1.0], &[], &Device::metal()).unwrap();

    let result = MetalDynBackend::gpu_quantized_matmul(&input_gpu, &qt);
    assert!(result.is_err(), "should error on rank-0 input");
}

// -- INT8 group_size 128 ------------------------------------------------------

#[test]
fn test_quantized_matmul_gpu_int8_group_size_128() {
    init();
    let config = QuantizationConfig::int8(128);

    // Weight [4, 256]
    let weight_data: Vec<f32> = (0..1024)
        .map(|i| (i as f32) * 0.01 - 5.12)
        .collect();
    let qt = make_quantized_weight(&weight_data, 4, 256, &config);

    // Input [2, 256]
    let input_data: Vec<f32> = (0..512).map(|i| (i as f32) * 0.02 - 5.12).collect();

    let input_cpu = DynTensor::from_vec(input_data.clone(), &[2, 256], &Device::Cpu).unwrap();
    let input_gpu = DynTensor::new(&input_data, &[2, 256], &Device::metal()).unwrap();

    let cpu_result = quantized_matmul(&input_cpu, &qt).unwrap();
    let gpu_result = MetalDynBackend::gpu_quantized_matmul(&input_gpu, &qt).unwrap();

    assert_eq!(cpu_result.dims(), &[2, 4]);
    assert_eq!(gpu_result.dims(), &[2, 4]);

    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    for (i, (g, c)) in gpu_vals.iter().zip(cpu_vals.iter()).enumerate() {
        assert!(
            (g - c).abs() < 1e-3,
            "int8_group128 [{i}]: gpu={g}, cpu={c}, diff={}",
            (g - c).abs()
        );
    }
}
