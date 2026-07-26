// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for quantized matrix multiplication (Q4_0, Q8_0).

use crate::dyn_tensor::quantized::QuantType;
use crate::dyn_tensor::quantized_matmul::{
    quantized_linear, quantized_matmul_q4_0, quantized_matmul_q8_0,
};
use crate::{DType, Device, DynTensor};
use ndarray::ArrayD;

// =========================================================================
// Test helpers: build quantized weight blocks
// =========================================================================

/// Build a Q4_0 block (18 bytes = 32 elements).
/// Layout: [f16 scale][16 bytes: 32 x 4-bit signed, packed 2/byte]
/// Dequant: val = scale * (nibble - 8)
fn make_q4_0_block(scale: f32, nibbles: &[u8; 32]) -> Vec<u8> {
    let scale_f16 = half::f16::from_f32(scale);
    let mut block = Vec::with_capacity(18);
    block.extend_from_slice(&scale_f16.to_le_bytes());
    for i in 0..16 {
        let lo = nibbles[2 * i] & 0x0F;
        let hi = nibbles[2 * i + 1] & 0x0F;
        block.push(lo | (hi << 4));
    }
    assert_eq!(block.len(), 18);
    block
}

/// Build a Q8_0 block (34 bytes = 32 elements).
/// Layout: [f16 scale][32 i8 values]
/// Dequant: val = scale * q
fn make_q8_0_block(scale: f32, values: &[i8; 32]) -> Vec<u8> {
    let scale_f16 = half::f16::from_f32(scale);
    let mut block = Vec::with_capacity(34);
    block.extend_from_slice(&scale_f16.to_le_bytes());
    for &v in values {
        block.push(v as u8);
    }
    assert_eq!(block.len(), 34);
    block
}

/// Build a Q4_0 weight tensor of shape [out_features, in_features] where
/// each row has the same scale and nibbles pattern.
fn make_q4_0_weight(
    out_features: usize,
    in_features: usize,
    scale: f32,
    nibbles: &[u8; 32],
) -> DynTensor {
    assert_eq!(in_features % 32, 0);
    let blocks_per_row = in_features / 32;
    let block = make_q4_0_block(scale, nibbles);
    let mut data = Vec::with_capacity(out_features * blocks_per_row * 18);
    for _ in 0..out_features * blocks_per_row {
        data.extend_from_slice(&block);
    }
    DynTensor::from_quantized(&data, QuantType::Q4_0, &[out_features, in_features]).unwrap()
}

/// Build a Q8_0 weight tensor of shape [out_features, in_features] where
/// each row has the same scale and values pattern.
fn make_q8_0_weight(
    out_features: usize,
    in_features: usize,
    scale: f32,
    values: &[i8; 32],
) -> DynTensor {
    assert_eq!(in_features % 32, 0);
    let blocks_per_row = in_features / 32;
    let block = make_q8_0_block(scale, values);
    let mut data = Vec::with_capacity(out_features * blocks_per_row * 34);
    for _ in 0..out_features * blocks_per_row {
        data.extend_from_slice(&block);
    }
    DynTensor::from_quantized(&data, QuantType::Q8_0, &[out_features, in_features]).unwrap()
}

// =========================================================================
// Q4_0 matmul correctness
// =========================================================================

#[test]
fn test_quantized_matmul_q4_0_known_values() {
    // Weight: [2, 32] Q4_0, scale=1.0, all nibbles=9 -> dequant = 1.0*(9-8) = 1.0
    // Each weight row is 32 ones.
    // Input: [1, 32] all ones.
    // Result: [1, 2] where each element = dot(ones_32, ones_32) = 32.0
    let weight = make_q4_0_weight(2, 32, 1.0, &[9u8; 32]);
    let input = DynTensor::ones(&[1, 32], DType::F32, &Device::Cpu).unwrap();

    let result = quantized_matmul_q4_0(&input, &weight).unwrap();
    assert_eq!(result.dims(), &[1, 2]);

    let vals = result.to_f32_array().unwrap();
    // Each output should be ~32.0 (32 elements * 1.0 * 1.0)
    for &v in vals.iter() {
        assert!((v - 32.0).abs() < 0.5, "expected ~32.0, got {v}");
    }
}

#[test]
fn test_quantized_matmul_q4_0_zero_weight() {
    // Weight: all nibbles=8 -> dequant = scale*(8-8) = 0.0
    // Result should be all zeros regardless of input.
    let weight = make_q4_0_weight(4, 32, 2.0, &[8u8; 32]);
    let input_arr = ArrayD::from_elem(vec![3, 32], 42.0_f32);
    let input = DynTensor::from_cpu_f32(input_arr).unwrap();

    let result = quantized_matmul_q4_0(&input, &weight).unwrap();
    assert_eq!(result.dims(), &[3, 4]);

    let vals = result.to_f32_array().unwrap();
    for &v in vals.iter() {
        assert!(
            v.abs() < 1e-3,
            "expected ~0.0 for zero-weight matmul, got {v}"
        );
    }
}

#[test]
fn test_quantized_matmul_q4_0_negative_values() {
    // Weight: scale=0.5, nibbles=4 -> dequant = 0.5*(4-8) = -2.0
    // Input: [1, 32] all 1.0
    // dot = 32 * 1.0 * (-2.0) = -64.0
    let weight = make_q4_0_weight(1, 32, 0.5, &[4u8; 32]);
    let input = DynTensor::ones(&[1, 32], DType::F32, &Device::Cpu).unwrap();

    let result = quantized_matmul_q4_0(&input, &weight).unwrap();
    assert_eq!(result.dims(), &[1, 1]);

    let vals = result.to_f32_array().unwrap();
    assert!(
        (vals[[0, 0]] - (-64.0)).abs() < 1.0,
        "expected ~-64.0, got {}",
        vals[[0, 0]]
    );
}

// =========================================================================
// Q8_0 matmul correctness
// =========================================================================

#[test]
fn test_quantized_matmul_q8_0_known_values() {
    // Weight: [2, 32] Q8_0, scale=0.5, all q=2 -> dequant = 0.5*2 = 1.0
    // Input: [1, 32] all ones.
    // dot = 32 * 1.0 * 1.0 = 32.0
    let weight = make_q8_0_weight(2, 32, 0.5, &[2i8; 32]);
    let input = DynTensor::ones(&[1, 32], DType::F32, &Device::Cpu).unwrap();

    let result = quantized_matmul_q8_0(&input, &weight).unwrap();
    assert_eq!(result.dims(), &[1, 2]);

    let vals = result.to_f32_array().unwrap();
    for &v in vals.iter() {
        assert!((v - 32.0).abs() < 0.5, "expected ~32.0, got {v}");
    }
}

#[test]
fn test_quantized_matmul_q8_0_identity_like() {
    // Weight: scale=1.0, q values = 1 for all elements
    // Input: [2, 32] with known values
    // Each output = sum of input row * 1.0
    let weight = make_q8_0_weight(1, 32, 1.0, &[1i8; 32]);

    let mut input_data = vec![0.0_f32; 2 * 32];
    for i in 0..32 {
        input_data[i] = 1.0; // first row: all ones
        input_data[32 + i] = 2.0; // second row: all twos
    }
    let input_arr = ArrayD::from_shape_vec(vec![2, 32], input_data).unwrap();
    let input = DynTensor::from_cpu_f32(input_arr).unwrap();

    let result = quantized_matmul_q8_0(&input, &weight).unwrap();
    assert_eq!(result.dims(), &[2, 1]);

    let vals = result.to_f32_array().unwrap();
    // Row 0: sum(1.0 * 1.0) * 32 = 32.0
    assert!(
        (vals[[0, 0]] - 32.0).abs() < 0.5,
        "row 0: expected ~32.0, got {}",
        vals[[0, 0]]
    );
    // Row 1: sum(2.0 * 1.0) * 32 = 64.0
    assert!(
        (vals[[1, 0]] - 64.0).abs() < 1.0,
        "row 1: expected ~64.0, got {}",
        vals[[1, 0]]
    );
}

// =========================================================================
// Shape validation (incompatible dims)
// =========================================================================

#[test]
fn test_quantized_matmul_q4_0_shape_mismatch() {
    // Weight: [2, 32], input: [1, 64] -> K mismatch (64 != 32)
    let weight = make_q4_0_weight(2, 32, 1.0, &[8u8; 32]);
    let input = DynTensor::ones(&[1, 64], DType::F32, &Device::Cpu).unwrap();

    let result = quantized_matmul_q4_0(&input, &weight);
    assert!(result.is_err(), "should fail on K dimension mismatch");
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("mismatch") || err.contains("64") || err.contains("32"),
        "error should describe shape mismatch: {err}"
    );
}

#[test]
fn test_quantized_matmul_q8_0_shape_mismatch() {
    // Weight: [3, 64], input: [2, 32] -> K mismatch (32 != 64)
    let weight = make_q8_0_weight(3, 64, 1.0, &[1i8; 32]);
    let input = DynTensor::ones(&[2, 32], DType::F32, &Device::Cpu).unwrap();

    let result = quantized_matmul_q8_0(&input, &weight);
    assert!(result.is_err(), "should fail on K dimension mismatch");
}

#[test]
fn test_quantized_matmul_rejects_non_quantized_weight() {
    // Passing a regular float tensor as weight should fail.
    let weight = DynTensor::ones(&[2, 32], DType::F32, &Device::Cpu).unwrap();
    let input = DynTensor::ones(&[1, 32], DType::F32, &Device::Cpu).unwrap();

    let result = quantized_matmul_q4_0(&input, &weight);
    assert!(result.is_err(), "should reject non-quantized weight");
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("not quantized"),
        "error should mention not quantized: {err}"
    );
}

#[test]
fn test_quantized_matmul_rejects_wrong_quant_type() {
    // Create a Q8_0 weight but call the Q4_0 matmul.
    let weight = make_q8_0_weight(2, 32, 1.0, &[1i8; 32]);
    let input = DynTensor::ones(&[1, 32], DType::F32, &Device::Cpu).unwrap();

    let result = quantized_matmul_q4_0(&input, &weight);
    assert!(result.is_err(), "should reject wrong quant type");
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("mismatch") || err.contains("Q4_0") || err.contains("Q8_0"),
        "error should describe quant type mismatch: {err}"
    );
}

#[test]
fn test_quantized_matmul_rejects_1d_input() {
    // Input must be at least 2-D.
    let weight = make_q4_0_weight(2, 32, 1.0, &[8u8; 32]);
    let input = DynTensor::ones(&[32], DType::F32, &Device::Cpu).unwrap();

    let result = quantized_matmul_q4_0(&input, &weight);
    assert!(result.is_err(), "should reject 1-D input");
}

// =========================================================================
// Batch dimension support
// =========================================================================

#[test]
fn test_quantized_matmul_q4_0_batch_3d() {
    // Input: [2, 3, 32] (batch=2, M=3, K=32)
    // Weight: [4, 32] (out_features=4, in_features=32)
    // Output: [2, 3, 4]
    let weight = make_q4_0_weight(4, 32, 1.0, &[9u8; 32]); // dequant = 1.0

    let input_arr = ArrayD::from_elem(vec![2, 3, 32], 1.0_f32);
    let input = DynTensor::from_cpu_f32(input_arr).unwrap();

    let result = quantized_matmul_q4_0(&input, &weight).unwrap();
    assert_eq!(result.dims(), &[2, 3, 4]);

    let vals = result.to_f32_array().unwrap();
    // Each element should be dot(ones_32, ones_32) = 32.0
    for &v in vals.iter() {
        assert!(
            (v - 32.0).abs() < 0.5,
            "expected ~32.0 in batched result, got {v}"
        );
    }
}

#[test]
fn test_quantized_matmul_q8_0_batch_4d() {
    // Input: [2, 2, 1, 32] (batch dims=[2,2], M=1, K=32)
    // Weight: [3, 32]
    // Output: [2, 2, 1, 3]
    let weight = make_q8_0_weight(3, 32, 1.0, &[1i8; 32]); // dequant = 1.0

    let input_arr = ArrayD::from_elem(vec![2, 2, 1, 32], 2.0_f32);
    let input = DynTensor::from_cpu_f32(input_arr).unwrap();

    let result = quantized_matmul_q8_0(&input, &weight).unwrap();
    assert_eq!(result.dims(), &[2, 2, 1, 3]);

    let vals = result.to_f32_array().unwrap();
    // Each element: dot(2.0 * 32, 1.0 * 32) = 64.0
    for &v in vals.iter() {
        assert!(
            (v - 64.0).abs() < 1.0,
            "expected ~64.0 in 4D batch, got {v}"
        );
    }
}

// =========================================================================
// Linear layer with bias
// =========================================================================

#[test]
fn test_quantized_linear_with_bias() {
    // Weight: [2, 32] Q8_0, scale=1.0, q=1 -> dequant = 1.0
    // Input: [1, 32] all ones.
    // matmul result: [1, 2] = [32.0, 32.0]
    // Bias: [2] = [10.0, 20.0]
    // Final: [42.0, 52.0]
    let weight = make_q8_0_weight(2, 32, 1.0, &[1i8; 32]);
    let input = DynTensor::ones(&[1, 32], DType::F32, &Device::Cpu).unwrap();

    let bias_arr = ArrayD::from_shape_vec(vec![2], vec![10.0_f32, 20.0]).unwrap();
    let bias = DynTensor::from_cpu_f32(bias_arr).unwrap();

    let result = quantized_linear(&input, &weight, Some(&bias)).unwrap();
    assert_eq!(result.dims(), &[1, 2]);

    let vals = result.to_f32_array().unwrap();
    assert!(
        (vals[[0, 0]] - 42.0).abs() < 1.0,
        "expected ~42.0, got {}",
        vals[[0, 0]]
    );
    assert!(
        (vals[[0, 1]] - 52.0).abs() < 1.0,
        "expected ~52.0, got {}",
        vals[[0, 1]]
    );
}

#[test]
fn test_quantized_linear_without_bias() {
    // No bias should produce the same result as raw quantized matmul.
    let weight = make_q4_0_weight(2, 32, 1.0, &[9u8; 32]); // dequant = 1.0
    let input = DynTensor::ones(&[1, 32], DType::F32, &Device::Cpu).unwrap();

    let result = quantized_linear(&input, &weight, None).unwrap();
    assert_eq!(result.dims(), &[1, 2]);

    let vals = result.to_f32_array().unwrap();
    for &v in vals.iter() {
        assert!(
            (v - 32.0).abs() < 0.5,
            "expected ~32.0 without bias, got {v}"
        );
    }
}

#[test]
fn test_quantized_linear_bias_shape_mismatch() {
    // Weight out_features=2, bias length=3 -> error.
    let weight = make_q8_0_weight(2, 32, 1.0, &[1i8; 32]);
    let input = DynTensor::ones(&[1, 32], DType::F32, &Device::Cpu).unwrap();

    let bias_arr = ArrayD::from_shape_vec(vec![3], vec![1.0_f32, 2.0, 3.0]).unwrap();
    let bias = DynTensor::from_cpu_f32(bias_arr).unwrap();

    let result = quantized_linear(&input, &weight, Some(&bias));
    assert!(result.is_err(), "should reject mismatched bias shape");
}

#[test]
fn test_quantized_linear_batched_with_bias() {
    // Input: [2, 1, 32], Weight: [3, 32], Bias: [3]
    // Output: [2, 1, 3] = matmul + bias broadcast
    let weight = make_q8_0_weight(3, 32, 1.0, &[1i8; 32]); // dequant = 1.0
    let input_arr = ArrayD::from_elem(vec![2, 1, 32], 0.5_f32);
    let input = DynTensor::from_cpu_f32(input_arr).unwrap();

    let bias_arr = ArrayD::from_shape_vec(vec![3], vec![100.0_f32, 200.0, 300.0]).unwrap();
    let bias = DynTensor::from_cpu_f32(bias_arr).unwrap();

    let result = quantized_linear(&input, &weight, Some(&bias)).unwrap();
    assert_eq!(result.dims(), &[2, 1, 3]);

    let vals = result.to_f32_array().unwrap();
    // matmul: dot(0.5*32, 1.0*32) = 16.0 per element, then +bias
    assert!(
        (vals[[0, 0, 0]] - 116.0).abs() < 1.0,
        "expected ~116.0, got {}",
        vals[[0, 0, 0]]
    );
    assert!(
        (vals[[0, 0, 1]] - 216.0).abs() < 1.0,
        "expected ~216.0, got {}",
        vals[[0, 0, 1]]
    );
    assert!(
        (vals[[0, 0, 2]] - 316.0).abs() < 1.0,
        "expected ~316.0, got {}",
        vals[[0, 0, 2]]
    );
}

// =========================================================================
// Comparison with full-precision matmul (dequant then matmul should match)
// =========================================================================

#[test]
fn test_quantized_matmul_q4_0_matches_dequant_then_matmul() {
    // The fused quantized matmul should produce the same result (within
    // floating-point tolerance) as: dequantize weight -> transpose -> matmul.
    let weight = make_q4_0_weight(4, 32, 0.5, &[10u8; 32]); // dequant = 0.5*(10-8) = 1.0

    let mut input_data = vec![0.0_f32; 3 * 32];
    for (i, v) in input_data.iter_mut().enumerate() {
        *v = (i as f32) * 0.01;
    }
    let input_arr = ArrayD::from_shape_vec(vec![3, 32], input_data).unwrap();
    let input = DynTensor::from_cpu_f32(input_arr).unwrap();

    // Fused path.
    let fused = quantized_matmul_q4_0(&input, &weight).unwrap();

    // Reference path: dequantize -> transpose -> standard matmul.
    let weight_deq = weight.dequantize().unwrap();
    let weight_t = weight_deq.t().unwrap();
    let reference = input.matmul(&weight_t).unwrap();

    assert_eq!(fused.dims(), reference.dims());
    let fused_vals = fused.to_f32_array().unwrap();
    let ref_vals = reference.to_f32_array().unwrap();

    let max_diff = fused_vals
        .iter()
        .zip(ref_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_diff < 1e-3,
        "fused Q4_0 matmul differs from reference by {max_diff} (tolerance 1e-3)"
    );
}

#[test]
fn test_quantized_matmul_q8_0_matches_dequant_then_matmul() {
    // Same comparison for Q8_0.
    let weight = make_q8_0_weight(4, 64, 0.25, &[4i8; 32]); // dequant = 0.25*4 = 1.0

    let mut input_data = vec![0.0_f32; 2 * 64];
    for (i, v) in input_data.iter_mut().enumerate() {
        *v = ((i % 64) as f32) * 0.1 - 3.0;
    }
    let input_arr = ArrayD::from_shape_vec(vec![2, 64], input_data).unwrap();
    let input = DynTensor::from_cpu_f32(input_arr).unwrap();

    // Fused path.
    let fused = quantized_matmul_q8_0(&input, &weight).unwrap();

    // Reference path.
    let weight_deq = weight.dequantize().unwrap();
    let weight_t = weight_deq.t().unwrap();
    let reference = input.matmul(&weight_t).unwrap();

    assert_eq!(fused.dims(), reference.dims());
    let fused_vals = fused.to_f32_array().unwrap();
    let ref_vals = reference.to_f32_array().unwrap();

    let max_diff = fused_vals
        .iter()
        .zip(ref_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_diff < 1e-3,
        "fused Q8_0 matmul differs from reference by {max_diff} (tolerance 1e-3)"
    );
}

#[test]
fn test_quantized_matmul_q4_0_batched_matches_reference() {
    // Batched case: input [2, 3, 32], weight [4, 32]
    let weight = make_q4_0_weight(4, 32, 1.0, &[9u8; 32]); // dequant = 1.0

    let input_arr = ArrayD::from_elem(vec![2, 3, 32], 0.5_f32);
    let input = DynTensor::from_cpu_f32(input_arr).unwrap();

    let fused = quantized_matmul_q4_0(&input, &weight).unwrap();

    let weight_deq = weight.dequantize().unwrap();
    let weight_t = weight_deq.t().unwrap();
    // Standard matmul handles 3D x 2D broadcast.
    let reference = input.matmul(&weight_t).unwrap();

    assert_eq!(fused.dims(), reference.dims());
    let fused_vals = fused.to_f32_array().unwrap();
    let ref_vals = reference.to_f32_array().unwrap();

    let max_diff = fused_vals
        .iter()
        .zip(ref_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_diff < 1e-3,
        "batched Q4_0 matmul differs from reference by {max_diff}"
    );
}

// =========================================================================
// Multi-block weight (in_features > 32)
// =========================================================================

#[test]
fn test_quantized_matmul_q8_0_multi_block() {
    // Weight: [2, 64] (2 blocks of 32 per row)
    // Input: [1, 64]
    let weight = make_q8_0_weight(2, 64, 1.0, &[1i8; 32]); // dequant = 1.0
    let input = DynTensor::ones(&[1, 64], DType::F32, &Device::Cpu).unwrap();

    let result = quantized_matmul_q8_0(&input, &weight).unwrap();
    assert_eq!(result.dims(), &[1, 2]);

    let vals = result.to_f32_array().unwrap();
    // dot(ones_64, ones_64) = 64.0
    for &v in vals.iter() {
        assert!(
            (v - 64.0).abs() < 1.0,
            "expected ~64.0 for 64-wide weight, got {v}"
        );
    }
}

#[test]
fn test_quantized_matmul_q4_0_multi_block() {
    // Weight: [1, 96] (3 blocks of 32 per row)
    // Scale=1.0, nibbles=9 -> dequant=1.0
    let weight = make_q4_0_weight(1, 96, 1.0, &[9u8; 32]);
    let input = DynTensor::ones(&[2, 96], DType::F32, &Device::Cpu).unwrap();

    let result = quantized_matmul_q4_0(&input, &weight).unwrap();
    assert_eq!(result.dims(), &[2, 1]);

    let vals = result.to_f32_array().unwrap();
    for &v in vals.iter() {
        assert!(
            (v - 96.0).abs() < 1.0,
            "expected ~96.0 for 96-wide weight, got {v}"
        );
    }
}

// =========================================================================
// Output finiteness
// =========================================================================

#[test]
fn test_quantized_matmul_output_finite() {
    // Ensure all outputs are finite (no NaN/Inf) with varied input.
    let weight = make_q4_0_weight(8, 32, 0.3, &[12u8; 32]);

    let mut input_data = vec![0.0_f32; 5 * 32];
    for (i, v) in input_data.iter_mut().enumerate() {
        *v = ((i as f32) * 0.7).sin();
    }
    let input_arr = ArrayD::from_shape_vec(vec![5, 32], input_data).unwrap();
    let input = DynTensor::from_cpu_f32(input_arr).unwrap();

    let result = quantized_matmul_q4_0(&input, &weight).unwrap();
    let vals = result.to_f32_array().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "output element {i} is not finite: {v}");
    }
}
