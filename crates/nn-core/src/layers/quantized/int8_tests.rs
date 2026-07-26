// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::dyn_tensor::DynTensor;
use crate::layers::quantized::Int8Linear;
use crate::layers::{Linear, Module};
use crate::{DType, Device};

// -- Symmetric quantization round-trip ----------------------------------------

#[test]
fn test_symmetric_quantize_dequantize_zeros() {
    let weights = DynTensor::zeros(&[4, 8], DType::F32, &Device::Cpu).unwrap();
    let (quantized, params) = quantize_per_channel(&weights, Int8Mode::Symmetric).unwrap();

    assert_eq!(quantized.dims(), &[4, 8]);
    assert_eq!(quantized.dtype(), DType::U8);
    assert_eq!(params.scale.len(), 4);
    assert_eq!(params.zero_point.len(), 4);

    // All scales should be 0 for zero weights
    for &s in &params.scale {
        assert_eq!(s, 0.0);
    }
    // All zero points should be 0 for symmetric
    for &zp in &params.zero_point {
        assert_eq!(zp, 0);
    }

    let dequantized = dequantize_per_channel(&quantized, &params).unwrap();
    let deq_data = dequantized.to_flat_vec::<f32>().unwrap();
    for &v in &deq_data {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn test_symmetric_quantize_dequantize_uniform() {
    // Weights in [-1, 1] range
    let mut data = vec![0.0_f32; 4 * 16];
    for (i, v) in data.iter_mut().enumerate() {
        *v = (i as f32 / 31.0) * 2.0 - 1.0;
    }
    let weights = DynTensor::from_vec(data.clone(), &[4, 16], &Device::Cpu).unwrap();
    let (quantized, params) = quantize_per_channel(&weights, Int8Mode::Symmetric).unwrap();

    assert_eq!(quantized.dims(), &[4, 16]);

    let dequantized = dequantize_per_channel(&quantized, &params).unwrap();
    let deq_data = dequantized.to_flat_vec::<f32>().unwrap();

    let max_err = max_quantization_error(&data, &deq_data);
    // INT8 symmetric: max error should be <= scale/2 per channel
    // scale = abs_max / 127, so max error ~ abs_max / 254
    // For [-1, 1] range: max error ~ 1/127 ~ 0.008
    assert!(
        max_err < 0.02,
        "symmetric quantize round-trip max error: {max_err}"
    );
}

#[test]
fn test_symmetric_quantize_dequantize_positive_only() {
    // All-positive weights (common for ReLU-activated layers)
    let mut data = vec![0.0_f32; 4 * 16];
    for (i, v) in data.iter_mut().enumerate() {
        *v = (i as f32 / 63.0) * 0.5;
    }
    let weights = DynTensor::from_vec(data.clone(), &[4, 16], &Device::Cpu).unwrap();
    let (quantized, params) = quantize_per_channel(&weights, Int8Mode::Symmetric).unwrap();

    // Symmetric: zero_point should be 0 even for all-positive data
    for &zp in &params.zero_point {
        assert_eq!(zp, 0);
    }

    let dequantized = dequantize_per_channel(&quantized, &params).unwrap();
    let deq_data = dequantized.to_flat_vec::<f32>().unwrap();

    let max_err = max_quantization_error(&data, &deq_data);
    assert!(
        max_err < 0.01,
        "symmetric positive-only max error: {max_err}"
    );
}

#[test]
fn test_symmetric_quantize_constant_channel() {
    // Channel where all values are identical
    let data = vec![0.5_f32; 3 * 8];
    let weights = DynTensor::from_vec(data.clone(), &[3, 8], &Device::Cpu).unwrap();
    let (quantized, params) = quantize_per_channel(&weights, Int8Mode::Symmetric).unwrap();

    let dequantized = dequantize_per_channel(&quantized, &params).unwrap();
    let deq_data = dequantized.to_flat_vec::<f32>().unwrap();

    let max_err = max_quantization_error(&data, &deq_data);
    assert!(
        max_err < 0.01,
        "constant channel round-trip max error: {max_err}"
    );
}

// -- Asymmetric quantization -------------------------------------------------

#[test]
fn test_asymmetric_quantize_dequantize() {
    let mut data = vec![0.0_f32; 4 * 16];
    for (i, v) in data.iter_mut().enumerate() {
        *v = (i as f32 / 63.0) * 2.0 - 1.0;
    }
    let weights = DynTensor::from_vec(data.clone(), &[4, 16], &Device::Cpu).unwrap();
    let (quantized, params) = quantize_per_channel(&weights, Int8Mode::Asymmetric).unwrap();

    assert_eq!(quantized.dims(), &[4, 16]);
    assert_eq!(params.scale.len(), 4);

    let dequantized = dequantize_per_channel(&quantized, &params).unwrap();
    let deq_data = dequantized.to_flat_vec::<f32>().unwrap();

    let max_err = max_quantization_error(&data, &deq_data);
    assert!(
        max_err < 0.02,
        "asymmetric quantize round-trip max error: {max_err}"
    );
}

// -- Error cases --------------------------------------------------------------

#[test]
fn test_quantize_nan_returns_error() {
    let data = vec![0.0, 1.0, f32::NAN, 0.5, 0.0, 1.0, 0.0, 0.5];
    let weights = DynTensor::from_vec(data, &[2, 4], &Device::Cpu).unwrap();
    let result = quantize_per_channel(&weights, Int8Mode::Symmetric);
    assert!(result.is_err(), "quantize with NaN should fail");
}

#[test]
fn test_quantize_inf_returns_error() {
    let data = vec![0.0, f32::INFINITY, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5];
    let weights = DynTensor::from_vec(data, &[2, 4], &Device::Cpu).unwrap();
    let result = quantize_per_channel(&weights, Int8Mode::Symmetric);
    assert!(result.is_err(), "quantize with Inf should fail");
}

#[test]
fn test_dequantize_mismatched_params_error() {
    let data = vec![0u8; 2 * 4];
    let quantized = DynTensor::from_vec_u8(data, &[2, 4], &Device::Cpu).unwrap();
    let bad_params = Int8QuantParams {
        scale: vec![1.0], // wrong length: should be 2
        zero_point: vec![0],
    };
    let result = dequantize_per_channel(&quantized, &bad_params);
    assert!(result.is_err(), "mismatched params length should fail");
}

// -- Int8Linear forward -------------------------------------------------------

#[test]
fn test_int8linear_from_linear_symmetric() {
    let weight = DynTensor::ones(&[4, 8], DType::F32, &Device::Cpu).unwrap();
    let linear = Linear::new(weight, None).unwrap();
    let int8_linear = Int8Linear::from_linear(&linear, Int8Mode::Symmetric).unwrap();

    assert_eq!(int8_linear.out_features(), 4);
    assert_eq!(int8_linear.in_features(), 8);
    assert!(int8_linear.bias().is_none());
}

#[test]
fn test_int8linear_forward_close_to_linear() {
    // Create a linear layer with known weights
    let mut weight_data = vec![0.0_f32; 4 * 16];
    for (i, v) in weight_data.iter_mut().enumerate() {
        *v = ((i % 17) as f32 / 16.0) * 0.3 - 0.15;
    }
    let weight = DynTensor::from_vec(weight_data, &[4, 16], &Device::Cpu).unwrap();
    let linear = Linear::new(weight, None).unwrap();

    // Get reference output
    let input = DynTensor::ones(&[2, 16], DType::F32, &Device::Cpu).unwrap();
    let ref_output = linear.forward(&input).unwrap();

    // Quantize and get INT8 output
    let int8_linear = Int8Linear::from_linear(&linear, Int8Mode::Symmetric).unwrap();
    let int8_output = int8_linear.forward(&input).unwrap();

    assert_eq!(ref_output.dims(), int8_output.dims());

    let ref_data = ref_output.to_flat_vec::<f32>().unwrap();
    let int8_data = int8_output.to_flat_vec::<f32>().unwrap();

    let max_err = ref_data
        .iter()
        .zip(int8_data.iter())
        .map(|(a, b): (&f32, &f32)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    // INT8 quantization error accumulates over K=16 elements in matmul:
    // per-element error ~ scale/2 ~ 0.15/127/2 ~ 0.0006
    // accumulated over K=16: ~ 0.01
    assert!(
        max_err < 0.1,
        "Int8Linear forward vs Linear max error: {max_err}"
    );
}

#[test]
fn test_int8linear_forward_with_bias() {
    let mut weight_data = vec![0.0_f32; 4 * 16];
    for (i, v) in weight_data.iter_mut().enumerate() {
        *v = ((i % 13) as f32 / 12.0) * 0.2 - 0.1;
    }
    let weight = DynTensor::from_vec(weight_data, &[4, 16], &Device::Cpu).unwrap();
    let bias = DynTensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[4], &Device::Cpu).unwrap();
    let linear = Linear::new(weight, Some(bias)).unwrap();

    let int8_linear = Int8Linear::from_linear(&linear, Int8Mode::Symmetric).unwrap();
    assert!(int8_linear.bias().is_some());

    let input = DynTensor::ones(&[1, 16], DType::F32, &Device::Cpu).unwrap();
    let output = int8_linear.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 4]);

    // Output should have non-zero values from bias
    let data = output.to_flat_vec::<f32>().unwrap();
    assert!(
        data.iter().any(|&v: &f32| v.abs() > 0.01),
        "output should be non-zero with bias"
    );
}

#[test]
fn test_int8linear_forward_3d_input() {
    let weight = DynTensor::ones(&[4, 8], DType::F32, &Device::Cpu).unwrap();
    let linear = Linear::new(weight, None).unwrap();
    let int8_linear = Int8Linear::from_linear(&linear, Int8Mode::Symmetric).unwrap();

    // 3D input [batch, seq, features]
    let input = DynTensor::ones(&[2, 3, 8], DType::F32, &Device::Cpu).unwrap();
    let output = int8_linear.forward(&input).unwrap();
    assert_eq!(output.dims(), &[2, 3, 4]);
}

#[test]
fn test_int8linear_dequantize_round_trip() {
    let mut weight_data = vec![0.0_f32; 4 * 16];
    for (i, v) in weight_data.iter_mut().enumerate() {
        *v = ((i % 11) as f32 / 10.0) * 0.4 - 0.2;
    }
    let weight = DynTensor::from_vec(weight_data.clone(), &[4, 16], &Device::Cpu).unwrap();
    let linear = Linear::new(weight, None).unwrap();

    let int8_linear = Int8Linear::from_linear(&linear, Int8Mode::Symmetric).unwrap();
    let recovered = int8_linear.dequantize().unwrap();

    let orig_data = linear.weight().to_flat_vec::<f32>().unwrap();
    let recov_data = recovered.weight().to_flat_vec::<f32>().unwrap();

    let max_err = max_quantization_error(&orig_data, &recov_data);
    assert!(
        max_err < 0.01,
        "Int8Linear dequantize round-trip max error: {max_err}"
    );
}

// -- Memory footprint ---------------------------------------------------------

#[test]
fn test_int8linear_memory_savings() {
    let weight = DynTensor::ones(&[768, 768], DType::F32, &Device::Cpu).unwrap();
    let linear = Linear::new(weight, None).unwrap();
    let int8_linear = Int8Linear::from_linear(&linear, Int8Mode::Symmetric).unwrap();

    let int8_bytes = int8_linear.memory_bytes();
    let f32_bytes = int8_linear.f32_memory_bytes();

    // INT8: 768*768*1 + 768*4 + 768*1 = 589,824 + 3,072 + 768 = 593,664
    // F32:  768*768*4 = 2,359,296
    assert!(
        int8_bytes < f32_bytes,
        "INT8 ({int8_bytes}) should be smaller than F32 ({f32_bytes})"
    );

    let ratio = int8_linear.compression_ratio();
    assert!(
        ratio > 3.5,
        "compression ratio should be ~4x, got {ratio:.2}"
    );
}

// -- Asymmetric Int8Linear ----------------------------------------------------

#[test]
fn test_int8linear_asymmetric_forward() {
    let mut weight_data = vec![0.0_f32; 4 * 16];
    for (i, v) in weight_data.iter_mut().enumerate() {
        *v = ((i % 17) as f32 / 16.0) * 0.3 - 0.15;
    }
    let weight = DynTensor::from_vec(weight_data, &[4, 16], &Device::Cpu).unwrap();
    let linear = Linear::new(weight, None).unwrap();

    let ref_output = {
        let input = DynTensor::ones(&[1, 16], DType::F32, &Device::Cpu).unwrap();
        linear.forward(&input).unwrap()
    };

    let int8_linear = Int8Linear::from_linear(&linear, Int8Mode::Asymmetric).unwrap();
    let input = DynTensor::ones(&[1, 16], DType::F32, &Device::Cpu).unwrap();
    let int8_output = int8_linear.forward(&input).unwrap();

    assert_eq!(ref_output.dims(), int8_output.dims());

    let ref_data = ref_output.to_flat_vec::<f32>().unwrap();
    let int8_data = int8_output.to_flat_vec::<f32>().unwrap();

    let max_err = ref_data
        .iter()
        .zip(int8_data.iter())
        .map(|(a, b): (&f32, &f32)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_err < 0.1,
        "asymmetric Int8Linear forward max error: {max_err}"
    );
}

// -- Edge cases ---------------------------------------------------------------

#[test]
fn test_int8linear_wrong_input_dim_error() {
    let weight = DynTensor::ones(&[4, 8], DType::F32, &Device::Cpu).unwrap();
    let linear = Linear::new(weight, None).unwrap();
    let int8_linear = Int8Linear::from_linear(&linear, Int8Mode::Symmetric).unwrap();

    // Input with wrong last dimension
    let input = DynTensor::ones(&[1, 16], DType::F32, &Device::Cpu).unwrap();
    let result = int8_linear.forward(&input);
    assert!(result.is_err(), "wrong input dim should fail");
}

#[test]
fn test_int8linear_new_validation() {
    let quantized = DynTensor::from_vec_u8(vec![0u8; 4 * 8], &[4, 8], &Device::Cpu).unwrap();

    // Wrong scale length
    let bad_params = Int8QuantParams {
        scale: vec![1.0, 2.0], // should be 4
        zero_point: vec![0; 4],
    };
    let result = Int8Linear::new(quantized.clone(), bad_params, None);
    assert!(result.is_err());

    // Wrong zero_point length
    let bad_params2 = Int8QuantParams {
        scale: vec![1.0; 4],
        zero_point: vec![0; 2], // should be 4
    };
    let result = Int8Linear::new(quantized.clone(), bad_params2, None);
    assert!(result.is_err());

    // Wrong bias shape
    let good_params = Int8QuantParams {
        scale: vec![1.0; 4],
        zero_point: vec![0; 4],
    };
    let bad_bias = DynTensor::zeros(&[8], DType::F32, &Device::Cpu).unwrap(); // should be [4]
    let result = Int8Linear::new(quantized, good_params, Some(bad_bias));
    assert!(result.is_err());
}
