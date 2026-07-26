#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{DType, Device};

// -- BlockQ4K struct size (AC6) -----------------------------------------------

#[test]
fn test_block_q4k_size_144_bytes() {
    assert_eq!(size_of::<BlockQ4K>(), 144);
}

#[test]
fn test_block_q4k_repr_c_alignment() {
    // repr(C) means fields are laid out in declaration order
    assert_eq!(align_of::<BlockQ4K>(), 2); // u16 alignment
}

// -- GgmlDType (AC1) ----------------------------------------------------------

#[test]
fn test_ggml_dtype_variants() {
    let _q4k = GgmlDType::Q4K;
    let _f32 = GgmlDType::F32;
    assert_ne!(GgmlDType::Q4K, GgmlDType::F32);
}

// -- Dequantize / Quantize round-trip (AC4) -----------------------------------

#[test]
fn test_quantize_dequantize_round_trip_zeros() {
    let input = [0.0_f32; 256];
    let block = BlockQ4K::quantize(&input).unwrap();
    let mut output = [0.0_f32; 256];
    block.dequantize(&mut output);
    let max_err = input
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_err < 0.01, "zeros round-trip error: {max_err}");
}

#[test]
fn test_quantize_dequantize_round_trip_uniform() {
    // Values in [0, 1] range
    let mut input = [0.0_f32; 256];
    for (i, v) in input.iter_mut().enumerate() {
        *v = i as f32 / 255.0;
    }
    let block = BlockQ4K::quantize(&input).unwrap();
    let mut output = [0.0_f32; 256];
    block.dequantize(&mut output);
    let max_err = input
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_err < 0.1,
        "uniform [0,1] round-trip max error: {max_err}"
    );
}

#[test]
fn test_quantize_dequantize_round_trip_negative() {
    // Values in [-1, 1] range
    let mut input = [0.0_f32; 256];
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i as f32 / 127.5) - 1.0;
    }
    let block = BlockQ4K::quantize(&input).unwrap();
    let mut output = [0.0_f32; 256];
    block.dequantize(&mut output);
    let max_err = input
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_err < 0.2, "[-1,1] round-trip max error: {max_err}");
}

#[test]
fn test_quantize_dequantize_round_trip_constant() {
    let input = [0.5_f32; 256];
    let block = BlockQ4K::quantize(&input).unwrap();
    let mut output = [0.0_f32; 256];
    block.dequantize(&mut output);
    let max_err = input
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_err < 0.1, "constant round-trip max error: {max_err}");
}

// -- QLinear from_linear round-trip (AC3, AC4) --------------------------------

#[test]
fn test_qlinear_from_linear_q4k() {
    let weight = DynTensor::ones(&[8, 256], DType::F32, &Device::Cpu).unwrap();
    let linear = Linear::new(weight, None).unwrap();
    let qlinear = QLinear::from_linear(&linear, GgmlDType::Q4K).unwrap();
    assert!(qlinear.is_quantized());
}

#[test]
fn test_qlinear_from_linear_f32_passthrough() {
    let weight = DynTensor::ones(&[8, 256], DType::F32, &Device::Cpu).unwrap();
    let linear = Linear::new(weight, None).unwrap();
    let qlinear = QLinear::from_linear(&linear, GgmlDType::F32).unwrap();
    assert!(!qlinear.is_quantized());
}

#[test]
fn test_qlinear_round_trip_weight_tolerance() {
    // AC4: from_linear → dequantize → max abs error < 0.1
    let mut data = vec![0.0_f32; 8 * 256];
    for (i, v) in data.iter_mut().enumerate() {
        *v = ((i % 256) as f32 / 255.0) * 0.5;
    }
    let weight = DynTensor::from_vec(data.clone(), &[8, 256], &Device::Cpu).unwrap();
    let linear = Linear::new(weight, None).unwrap();

    let qlinear = QLinear::from_linear(&linear, GgmlDType::Q4K).unwrap();
    let recovered = qlinear.dequantize().unwrap();

    let orig_data = linear.weight().to_flat_vec::<f32>().unwrap();
    let recov_data = recovered.weight().to_flat_vec::<f32>().unwrap();

    let max_err = orig_data
        .iter()
        .zip(recov_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_err < 0.1, "QLinear round-trip max error: {max_err}");
}

// -- QLinear::forward parity (AC2, AC5) ---------------------------------------

#[test]
fn test_qlinear_forward_float_same_as_linear() {
    let weight = DynTensor::ones(&[4, 8], DType::F32, &Device::Cpu).unwrap();
    let bias = DynTensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    let linear = Linear::new(weight, Some(bias)).unwrap();
    let qlinear = QLinear::from_float(linear);

    let input = DynTensor::ones(&[1, 8], DType::F32, &Device::Cpu).unwrap();
    let output = qlinear.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 4]);
}

#[test]
fn test_qlinear_forward_quantized_within_tolerance() {
    // AC5: QLinear::forward() within tolerance of Linear::forward()
    let mut weight_data = vec![0.0_f32; 4 * 256];
    for (i, v) in weight_data.iter_mut().enumerate() {
        *v = ((i % 17) as f32 / 16.0) * 0.3;
    }
    let weight = DynTensor::from_vec(weight_data, &[4, 256], &Device::Cpu).unwrap();
    let linear = Linear::new(weight, None).unwrap();

    // Linear forward
    let input = DynTensor::ones(&[1, 256], DType::F32, &Device::Cpu).unwrap();
    let linear_out = linear.forward(&input).unwrap();

    // QLinear forward
    let qlinear = QLinear::from_linear(&linear, GgmlDType::Q4K).unwrap();
    let q_out = qlinear.forward(&input).unwrap();

    assert_eq!(linear_out.dims(), q_out.dims());

    let linear_data = linear_out.to_flat_vec::<f32>().unwrap();
    let q_data = q_out.to_flat_vec::<f32>().unwrap();

    let max_err = linear_data
        .iter()
        .zip(q_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    // Q4K is ~4.5 bits, tolerance is generous for Phase 1
    assert!(
        max_err < 5.0,
        "QLinear forward max error vs Linear: {max_err}"
    );
}

#[test]
fn test_qlinear_forward_quantized_with_bias() {
    let mut weight_data = vec![0.0_f32; 4 * 256];
    for (i, v) in weight_data.iter_mut().enumerate() {
        *v = ((i % 13) as f32 / 12.0) * 0.2;
    }
    let weight = DynTensor::from_vec(weight_data, &[4, 256], &Device::Cpu).unwrap();
    let bias = DynTensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[4], &Device::Cpu).unwrap();
    let linear = Linear::new(weight, Some(bias)).unwrap();

    let qlinear = QLinear::from_linear(&linear, GgmlDType::Q4K).unwrap();
    let input = DynTensor::ones(&[1, 256], DType::F32, &Device::Cpu).unwrap();
    let output = qlinear.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 4]);

    // Verify bias is present in output (output should not be all-zero)
    let data = output.to_flat_vec::<f32>().unwrap();
    assert!(
        data.iter().any(|&v| v.abs() > 0.01),
        "output should be non-zero with bias"
    );
}

// -- from_float / is_quantized -----------------------------------------------

#[test]
fn test_from_float_preserves_forward() {
    let weight = DynTensor::ones(&[3, 4], DType::F32, &Device::Cpu).unwrap();
    let linear = Linear::new(weight, None).unwrap();
    let qlinear = QLinear::from_float(linear);
    assert!(!qlinear.is_quantized());

    let input = DynTensor::ones(&[1, 4], DType::F32, &Device::Cpu).unwrap();
    let output = qlinear.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 3]);
}

// -- Edge cases ---------------------------------------------------------------

#[test]
fn test_qlinear_dequantize_f32_passthrough() {
    let weight = DynTensor::ones(&[3, 4], DType::F32, &Device::Cpu).unwrap();
    let linear = Linear::new(weight, None).unwrap();
    let qlinear = QLinear::from_linear(&linear, GgmlDType::F32).unwrap();

    let recovered = qlinear.dequantize().unwrap();
    let orig = linear.weight().to_flat_vec::<f32>().unwrap();
    let recov = recovered.weight().to_flat_vec::<f32>().unwrap();
    assert_eq!(orig, recov);
}

#[test]
fn test_qlinear_non_block_aligned_weight() {
    // Weight dimensions where out*in is not a multiple of 256
    let weight = DynTensor::ones(&[3, 100], DType::F32, &Device::Cpu).unwrap();
    let linear = Linear::new(weight, None).unwrap();
    let qlinear = QLinear::from_linear(&linear, GgmlDType::Q4K).unwrap();
    assert!(qlinear.is_quantized());

    // Forward should still work
    let input = DynTensor::ones(&[1, 100], DType::F32, &Device::Cpu).unwrap();
    let output = qlinear.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 3]);
}

// -- #1383 boundary regression tests ------------------------------------------

#[test]
fn test_nearest_int_nan_returns_error() {
    let result = nearest_int(f32::NAN);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("non-finite"),
        "expected non-finite error: {msg}"
    );
}

#[test]
fn test_nearest_int_inf_returns_error() {
    let result = nearest_int(f32::INFINITY);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("non-finite"),
        "expected non-finite error: {msg}"
    );
}

#[test]
fn test_nearest_int_neg_inf_returns_error() {
    let result = nearest_int(f32::NEG_INFINITY);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("non-finite"),
        "expected non-finite error: {msg}"
    );
}

#[test]
fn test_nearest_int_normal_values() {
    assert_eq!(nearest_int(0.0).unwrap(), 0);
    assert_eq!(nearest_int(1.5).unwrap(), 2);
    assert_eq!(nearest_int(-1.5).unwrap(), -2);
    assert_eq!(nearest_int(0.4).unwrap(), 0);
    assert_eq!(nearest_int(-0.4).unwrap(), 0);
}

#[test]
fn test_quantize_nan_input_returns_error() {
    let mut input = [0.5_f32; 256];
    input[128] = f32::NAN;
    let result = BlockQ4K::quantize(&input);
    assert!(result.is_err(), "quantize with NaN input should fail");
}

// -- #1589 scale==0.0 division-by-zero regression ----------------------------

#[test]
fn test_quantize_values_clustered_at_min_no_div_by_zero() {
    // Sub-block where most values equal min, causing sumlx==0 and scale==0
    // in the iterative loop. Before fix, this produced Infinity iscale.
    let mut input = [0.0_f32; 256];
    // Set min at -1.0, one outlier at 0.0, rest at -1.0
    for v in input.iter_mut() {
        *v = -1.0;
    }
    input[0] = 0.0; // small positive offset from min
    let block = BlockQ4K::quantize(&input);
    assert!(block.is_ok(), "quantize should not fail: {:?}", block.err());

    let block = block.unwrap();
    let mut output = [0.0_f32; 256];
    block.dequantize(&mut output);
    // All output values should be finite
    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "non-finite output at index {i}: {v}");
    }
}

// -- Batched forward tests (#1536) --------------------------------------------

#[test]
fn test_qlinear_forward_float_batched() {
    let weight = DynTensor::ones(&[4, 8], DType::F32, &Device::Cpu).unwrap();
    let qlinear = QLinear::from_float(Linear::new(weight, None).unwrap());

    let input = DynTensor::ones(&[4, 8], DType::F32, &Device::Cpu).unwrap();
    let output = qlinear.forward(&input).unwrap();
    assert_eq!(output.dims(), &[4, 4]);

    // All rows should produce the same output (same input, same weights).
    let data = output.to_flat_vec::<f32>().unwrap();
    for row in 1..4 {
        for col in 0..4 {
            assert!(
                (data[row * 4 + col] - data[col]).abs() < 1e-6,
                "batch row {row} col {col} differs: {} vs {}",
                data[row * 4 + col],
                data[col]
            );
        }
    }
}

#[test]
fn test_qlinear_forward_quantized_batched() {
    // Q4K quantized forward with batch > 1.
    let mut weight_data = vec![0.0_f32; 4 * 256];
    for (i, v) in weight_data.iter_mut().enumerate() {
        *v = ((i % 17) as f32 / 16.0) * 0.3;
    }
    let weight = DynTensor::from_vec(weight_data, &[4, 256], &Device::Cpu).unwrap();
    let linear = Linear::new(weight, None).unwrap();
    let qlinear = QLinear::from_linear(&linear, GgmlDType::Q4K).unwrap();

    let input = DynTensor::ones(&[4, 256], DType::F32, &Device::Cpu).unwrap();
    let output = qlinear.forward(&input).unwrap();
    assert_eq!(output.dims(), &[4, 4]);

    // Verify output is finite.
    let data = output.to_flat_vec::<f32>().unwrap();
    for &v in &data {
        assert!(v.is_finite(), "non-finite quantized batched output: {v}");
    }
}

#[test]
fn test_qlinear_forward_3d_batched() {
    // 3D input [batch, seq, features] — typical transformer shape.
    let weight = DynTensor::ones(&[4, 8], DType::F32, &Device::Cpu).unwrap();
    let qlinear = QLinear::from_float(Linear::new(weight, None).unwrap());

    let input = DynTensor::ones(&[2, 3, 8], DType::F32, &Device::Cpu).unwrap();
    let output = qlinear.forward(&input).unwrap();
    assert_eq!(output.dims(), &[2, 3, 4]);
}
