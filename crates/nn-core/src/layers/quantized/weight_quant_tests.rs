// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::dyn_tensor::DynTensor;
use crate::layers::quantized::int8::max_quantization_error;
use crate::{DType, Device};

// ---------------------------------------------------------------------------
// INT4 symmetric round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_int4_symmetric_roundtrip_zeros() {
    let weights = DynTensor::zeros(&[4, 128], DType::F32, &Device::Cpu).unwrap();
    let config = QuantizationConfig::int4(128);
    let qt = quantize_per_group(&weights, &config).unwrap();

    assert_eq!(qt.shape, [4, 128]);
    assert_eq!(qt.scales.len(), 4); // 4 rows * 1 group/row
    assert_eq!(qt.zero_points.len(), 4);

    // All scales should be 0 for zero weights
    for &s in &qt.scales {
        assert_eq!(s, 0.0);
    }

    let deq = dequantize(&qt).unwrap();
    let deq_data = deq.to_flat_vec::<f32>().unwrap();
    for &v in &deq_data {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn test_int4_symmetric_roundtrip_uniform() {
    let n = 4 * 128;
    let mut data = vec![0.0_f32; n];
    for (i, v) in data.iter_mut().enumerate() {
        *v = (i as f32 / (n - 1) as f32) * 2.0 - 1.0; // range [-1, 1]
    }
    let weights = DynTensor::from_vec(data.clone(), &[4, 128], &Device::Cpu).unwrap();
    let config = QuantizationConfig::int4(128);
    let qt = quantize_per_group(&weights, &config).unwrap();

    let deq = dequantize(&qt).unwrap();
    let deq_data = deq.to_flat_vec::<f32>().unwrap();

    let max_err = max_quantization_error(&data, &deq_data);
    // INT4 symmetric: scale = abs_max / 7 = 1.0/7 = 0.143, max error ~ scale/2 = 0.071
    assert!(
        max_err < 0.10,
        "INT4 symmetric round-trip max error: {max_err}"
    );
}

#[test]
fn test_int4_symmetric_roundtrip_small_groups() {
    // group_size = 32 (smaller groups, better accuracy)
    let n = 4 * 128;
    let mut data = vec![0.0_f32; n];
    for (i, v) in data.iter_mut().enumerate() {
        *v = (i as f32 / (n - 1) as f32) * 2.0 - 1.0; // range [-1, 1]
    }
    let weights = DynTensor::from_vec(data.clone(), &[4, 128], &Device::Cpu).unwrap();
    let config = QuantizationConfig::int4(32);
    let qt = quantize_per_group(&weights, &config).unwrap();

    assert_eq!(qt.scales.len(), 4 * 4); // 4 rows * (128/32) groups/row = 16

    let deq = dequantize(&qt).unwrap();
    let deq_data = deq.to_flat_vec::<f32>().unwrap();

    let max_err = max_quantization_error(&data, &deq_data);
    // INT4 symmetric: scale = abs_max / 7, max error ~ scale / 2
    assert!(
        max_err < 0.10,
        "INT4 group_size=32 round-trip max error: {max_err}"
    );
}

// ---------------------------------------------------------------------------
// INT8 symmetric round-trip (via weight_quant API)
// ---------------------------------------------------------------------------

#[test]
fn test_int8_group_roundtrip_uniform() {
    let n = 4 * 128;
    let mut data = vec![0.0_f32; n];
    for (i, v) in data.iter_mut().enumerate() {
        *v = (i as f32 / (n - 1) as f32) * 2.0 - 1.0; // range [-1, 1]
    }
    let weights = DynTensor::from_vec(data.clone(), &[4, 128], &Device::Cpu).unwrap();
    let config = QuantizationConfig::int8(128);
    let qt = quantize_per_group(&weights, &config).unwrap();

    let deq = dequantize(&qt).unwrap();
    let deq_data = deq.to_flat_vec::<f32>().unwrap();

    let max_err = max_quantization_error(&data, &deq_data);
    // INT8 with 255 levels: max error ~ abs_max / 127 / 2 ~ 0.004
    assert!(max_err < 0.02, "INT8 group round-trip max error: {max_err}");
}

#[test]
fn test_int8_group_better_than_int4() {
    // INT8 should always have lower max error than INT4 with same group_size
    let mut data = vec![0.0_f32; 4 * 128];
    for (i, v) in data.iter_mut().enumerate() {
        *v = ((i as f32).sin() * 0.5).clamp(-0.5, 0.5);
    }
    let weights = DynTensor::from_vec(data.clone(), &[4, 128], &Device::Cpu).unwrap();

    let int4_qt = quantize_per_group(&weights, &QuantizationConfig::int4(128)).unwrap();
    let int8_qt = quantize_per_group(&weights, &QuantizationConfig::int8(128)).unwrap();

    let int4_deq = dequantize(&int4_qt).unwrap().to_flat_vec::<f32>().unwrap();
    let int8_deq = dequantize(&int8_qt).unwrap().to_flat_vec::<f32>().unwrap();

    let int4_err = max_quantization_error(&data, &int4_deq);
    let int8_err = max_quantization_error(&data, &int8_deq);

    assert!(
        int8_err <= int4_err,
        "INT8 error ({int8_err}) should be <= INT4 error ({int4_err})"
    );
}

// ---------------------------------------------------------------------------
// Asymmetric quantization
// ---------------------------------------------------------------------------

#[test]
fn test_int4_asymmetric_roundtrip() {
    let mut data = vec![0.0_f32; 4 * 128];
    for (i, v) in data.iter_mut().enumerate() {
        *v = (i as f32 / 511.0) * 2.0 - 1.0;
    }
    let weights = DynTensor::from_vec(data.clone(), &[4, 128], &Device::Cpu).unwrap();

    let config = QuantizationConfig {
        dtype: QuantDtype::Int4,
        group_size: 128,
        symmetric: false,
    };
    let qt = quantize_per_group(&weights, &config).unwrap();

    let deq = dequantize(&qt).unwrap();
    let deq_data = deq.to_flat_vec::<f32>().unwrap();

    let max_err = max_quantization_error(&data, &deq_data);
    assert!(
        max_err < 0.2,
        "INT4 asymmetric round-trip max error: {max_err}"
    );
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

#[test]
fn test_quantize_nan_returns_error() {
    let data = vec![0.0; 126];
    let mut full = data;
    full.push(f32::NAN);
    full.push(0.5);
    let weights = DynTensor::from_vec(full, &[1, 128], &Device::Cpu).unwrap();
    let config = QuantizationConfig::int4(128);
    let result = quantize_per_group(&weights, &config);
    assert!(result.is_err(), "quantize with NaN should fail");
}

#[test]
fn test_quantize_inf_returns_error() {
    let mut data = vec![0.0_f32; 128];
    data[5] = f32::INFINITY;
    let weights = DynTensor::from_vec(data, &[1, 128], &Device::Cpu).unwrap();
    let config = QuantizationConfig::int4(128);
    let result = quantize_per_group(&weights, &config);
    assert!(result.is_err(), "quantize with Inf should fail");
}

#[test]
fn test_quantize_indivisible_group_size_error() {
    let weights = DynTensor::zeros(&[4, 100], DType::F32, &Device::Cpu).unwrap();
    let config = QuantizationConfig::int4(128); // 100 not divisible by 128
    let result = quantize_per_group(&weights, &config);
    assert!(result.is_err(), "indivisible group_size should fail");
}

#[test]
fn test_quantize_zero_group_size_error() {
    let weights = DynTensor::zeros(&[4, 128], DType::F32, &Device::Cpu).unwrap();
    let config = QuantizationConfig {
        dtype: QuantDtype::Int4,
        group_size: 0,
        symmetric: true,
    };
    let result = quantize_per_group(&weights, &config);
    assert!(result.is_err(), "zero group_size should fail");
}

// ---------------------------------------------------------------------------
// quantized_matmul
// ---------------------------------------------------------------------------

#[test]
fn test_quantized_matmul_int4_close_to_float() {
    let mut weight_data = vec![0.0_f32; 4 * 128];
    for (i, v) in weight_data.iter_mut().enumerate() {
        *v = ((i % 17) as f32 / 16.0) * 0.3 - 0.15;
    }
    let weight_tensor = DynTensor::from_vec(weight_data.clone(), &[4, 128], &Device::Cpu).unwrap();

    // Reference: full-precision matmul
    let input = DynTensor::ones(&[2, 128], DType::F32, &Device::Cpu).unwrap();
    let ref_output = input.matmul(&weight_tensor.t().unwrap()).unwrap();

    // Quantized matmul
    let config = QuantizationConfig::int4(128);
    let qt = quantize_per_group(
        &DynTensor::from_vec(weight_data, &[4, 128], &Device::Cpu).unwrap(),
        &config,
    )
    .unwrap();
    let q_output = quantized_matmul(&input, &qt).unwrap();

    assert_eq!(ref_output.dims(), q_output.dims());

    let ref_data = ref_output.to_flat_vec::<f32>().unwrap();
    let q_data = q_output.to_flat_vec::<f32>().unwrap();

    let max_err = ref_data
        .iter()
        .zip(q_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    // INT4 quantization error accumulates over K=128 elements
    // Per-element error ~ 0.15/7/2 ~ 0.01, accumulated ~ 1.3
    assert!(
        max_err < 2.0,
        "INT4 quantized_matmul vs float max error: {max_err}"
    );
}

#[test]
fn test_quantized_matmul_int8_close_to_float() {
    let mut weight_data = vec![0.0_f32; 4 * 128];
    for (i, v) in weight_data.iter_mut().enumerate() {
        *v = ((i % 17) as f32 / 16.0) * 0.3 - 0.15;
    }
    let weight_tensor = DynTensor::from_vec(weight_data.clone(), &[4, 128], &Device::Cpu).unwrap();

    let input = DynTensor::ones(&[2, 128], DType::F32, &Device::Cpu).unwrap();
    let ref_output = input.matmul(&weight_tensor.t().unwrap()).unwrap();

    let config = QuantizationConfig::int8(128);
    let qt = quantize_per_group(
        &DynTensor::from_vec(weight_data, &[4, 128], &Device::Cpu).unwrap(),
        &config,
    )
    .unwrap();
    let q_output = quantized_matmul(&input, &qt).unwrap();

    assert_eq!(ref_output.dims(), q_output.dims());

    let ref_data = ref_output.to_flat_vec::<f32>().unwrap();
    let q_data = q_output.to_flat_vec::<f32>().unwrap();

    let max_err = ref_data
        .iter()
        .zip(q_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    // INT8 should be much more accurate than INT4
    assert!(
        max_err < 0.5,
        "INT8 quantized_matmul vs float max error: {max_err}"
    );
}

#[test]
fn test_quantized_matmul_wrong_dim_error() {
    let weight_data = vec![0.0_f32; 4 * 128];
    let weights = DynTensor::from_vec(weight_data, &[4, 128], &Device::Cpu).unwrap();
    let config = QuantizationConfig::int4(128);
    let qt = quantize_per_group(&weights, &config).unwrap();

    // Input with wrong last dimension
    let input = DynTensor::ones(&[2, 64], DType::F32, &Device::Cpu).unwrap();
    let result = quantized_matmul(&input, &qt);
    assert!(result.is_err(), "wrong input dim should fail");
}

#[test]
fn test_quantized_matmul_3d_input() {
    let weight_data = vec![0.1_f32; 4 * 128];
    let weights = DynTensor::from_vec(weight_data, &[4, 128], &Device::Cpu).unwrap();
    let config = QuantizationConfig::int4(128);
    let qt = quantize_per_group(&weights, &config).unwrap();

    let input = DynTensor::ones(&[2, 3, 128], DType::F32, &Device::Cpu).unwrap();
    let output = quantized_matmul(&input, &qt).unwrap();
    assert_eq!(output.dims(), &[2, 3, 4]);
}

// ---------------------------------------------------------------------------
// Memory footprint
// ---------------------------------------------------------------------------

#[test]
fn test_int4_compression_ratio() {
    let weight_data = vec![0.1_f32; 768 * 768];
    let weights = DynTensor::from_vec(weight_data, &[768, 768], &Device::Cpu).unwrap();
    let config = QuantizationConfig::int4(128);
    let qt = quantize_per_group(&weights, &config).unwrap();

    let ratio = qt.compression_ratio();
    // INT4: data = 768*768/2 = 294912, scales = 768*6*4 = 18432, zps = 768*6*4 = 18432
    // Total INT4 = 294912 + 18432 + 18432 = 331776
    // F32 = 768*768*4 = 2359296
    // Ratio ~ 7.1x
    assert!(
        ratio > 5.0,
        "INT4 compression ratio should be > 5x, got {ratio:.2}"
    );
}

#[test]
fn test_int8_compression_ratio() {
    let weight_data = vec![0.1_f32; 768 * 768];
    let weights = DynTensor::from_vec(weight_data, &[768, 768], &Device::Cpu).unwrap();
    let config = QuantizationConfig::int8(128);
    let qt = quantize_per_group(&weights, &config).unwrap();

    let ratio = qt.compression_ratio();
    // INT8: data = 768*768 = 589824, scales = 768*6*4 = 18432, zps = 768*6*4 = 18432
    // Total INT8 = 589824 + 18432 + 18432 = 626688
    // F32 = 768*768*4 = 2359296
    // Ratio ~ 3.76x
    assert!(
        ratio > 3.0,
        "INT8 compression ratio should be > 3x, got {ratio:.2}"
    );
}

#[test]
fn test_int4_uses_less_memory_than_int8() {
    let weight_data = vec![0.1_f32; 768 * 768];
    let weights = DynTensor::from_vec(weight_data, &[768, 768], &Device::Cpu).unwrap();

    let int4_qt = quantize_per_group(&weights, &QuantizationConfig::int4(128)).unwrap();
    let int8_qt = quantize_per_group(&weights, &QuantizationConfig::int8(128)).unwrap();

    assert!(
        int4_qt.memory_bytes() < int8_qt.memory_bytes(),
        "INT4 ({}) should use less memory than INT8 ({})",
        int4_qt.memory_bytes(),
        int8_qt.memory_bytes()
    );
}

// ---------------------------------------------------------------------------
// Constant channel / edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_quantize_constant_group() {
    let data = vec![0.5_f32; 4 * 128];
    let weights = DynTensor::from_vec(data.clone(), &[4, 128], &Device::Cpu).unwrap();
    let config = QuantizationConfig::int4(128);
    let qt = quantize_per_group(&weights, &config).unwrap();

    let deq = dequantize(&qt).unwrap();
    let deq_data = deq.to_flat_vec::<f32>().unwrap();

    let max_err = max_quantization_error(&data, &deq_data);
    assert!(
        max_err < 0.01,
        "constant group round-trip max error: {max_err}"
    );
}

#[test]
fn test_quantize_negative_only() {
    let mut data = vec![0.0_f32; 4 * 128];
    for (i, v) in data.iter_mut().enumerate() {
        *v = -(i as f32 / 511.0) * 0.5;
    }
    let weights = DynTensor::from_vec(data.clone(), &[4, 128], &Device::Cpu).unwrap();
    let config = QuantizationConfig::int4(128);
    let qt = quantize_per_group(&weights, &config).unwrap();

    let deq = dequantize(&qt).unwrap();
    let deq_data = deq.to_flat_vec::<f32>().unwrap();

    let max_err = max_quantization_error(&data, &deq_data);
    assert!(
        max_err < 0.2,
        "negative-only round-trip max error: {max_err}"
    );
}

// ---------------------------------------------------------------------------
// QuantDtype properties
// ---------------------------------------------------------------------------

#[test]
fn test_quant_dtype_bits() {
    assert_eq!(QuantDtype::Int4.bits(), 4);
    assert_eq!(QuantDtype::Int8.bits(), 8);
}

#[test]
fn test_default_config() {
    let config = QuantizationConfig::default();
    assert_eq!(config.dtype, QuantDtype::Int4);
    assert_eq!(config.group_size, 128);
    assert!(config.symmetric);
}
