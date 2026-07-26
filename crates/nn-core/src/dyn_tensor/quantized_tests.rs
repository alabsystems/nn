// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for quantized tensor storage and dequantization.

use crate::dyn_tensor::quantized::{QuantType, QuantizedStorage};
use crate::{DType, DynTensor};

/// Build a minimal Q4_0 block (18 bytes = 32 elements).
///
/// Layout: [f16 scale][16 bytes of 4-bit pairs]
/// Each nibble stores (value + 8), so nibble 8 → 0.0, nibble 0 → -8*scale, nibble 15 → 7*scale.
fn make_q4_0_block(scale: f32, nibbles: &[u8; 32]) -> Vec<u8> {
    let scale_f16 = half::f16::from_f32(scale);
    let mut block = Vec::with_capacity(18);
    block.extend_from_slice(&scale_f16.to_le_bytes());
    // Pack 32 nibbles into 16 bytes (low nibble first per byte).
    for i in 0..16 {
        let lo = nibbles[2 * i] & 0x0F;
        let hi = nibbles[2 * i + 1] & 0x0F;
        block.push(lo | (hi << 4));
    }
    assert_eq!(block.len(), 18);
    block
}

/// Build a minimal Q4_1 block (20 bytes = 32 elements).
///
/// Layout: [f16 d][f16 m][16 bytes of 4-bit pairs]
/// Dequant: val = d * nibble + m
fn make_q4_1_block(d: f32, m: f32, nibbles: &[u8; 32]) -> Vec<u8> {
    let d_f16 = half::f16::from_f32(d);
    let m_f16 = half::f16::from_f32(m);
    let mut block = Vec::with_capacity(20);
    block.extend_from_slice(&d_f16.to_le_bytes());
    block.extend_from_slice(&m_f16.to_le_bytes());
    for i in 0..16 {
        let lo = nibbles[2 * i] & 0x0F;
        let hi = nibbles[2 * i + 1] & 0x0F;
        block.push(lo | (hi << 4));
    }
    assert_eq!(block.len(), 20);
    block
}

/// Build a minimal Q8_0 block (34 bytes = 32 elements).
///
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

// -- QuantType tests ----------------------------------------------------------

#[test]
fn test_quant_type_block_size() {
    assert_eq!(QuantType::Q4_0.block_size(), 32);
    assert_eq!(QuantType::Q4_1.block_size(), 32);
    assert_eq!(QuantType::Q8_0.block_size(), 32);
}

#[test]
fn test_quant_type_block_bytes() {
    assert_eq!(QuantType::Q4_0.block_bytes(), 18);
    assert_eq!(QuantType::Q4_1.block_bytes(), 20);
    assert_eq!(QuantType::Q8_0.block_bytes(), 34);
}

#[test]
fn test_quant_type_expected_bytes() {
    assert_eq!(QuantType::Q4_0.expected_bytes(32), Some(18));
    assert_eq!(QuantType::Q4_0.expected_bytes(64), Some(36));
    assert_eq!(QuantType::Q4_0.expected_bytes(0), Some(0));
    // Not a multiple of 32 → None.
    assert_eq!(QuantType::Q4_0.expected_bytes(31), None);
    assert_eq!(QuantType::Q4_0.expected_bytes(33), None);

    assert_eq!(QuantType::Q8_0.expected_bytes(32), Some(34));
    assert_eq!(QuantType::Q8_0.expected_bytes(64), Some(68));
}

#[test]
fn test_quant_type_display() {
    assert_eq!(format!("{}", QuantType::Q4_0), "Q4_0");
    assert_eq!(format!("{}", QuantType::Q4_1), "Q4_1");
    assert_eq!(format!("{}", QuantType::Q8_0), "Q8_0");
}

// -- QuantizedStorage tests ---------------------------------------------------

#[test]
fn test_quantized_storage_new_validates_size() {
    // Correct size for 32 Q4_0 elements.
    let data = vec![0u8; 18];
    let qs = QuantizedStorage::new(data, &[32], QuantType::Q4_0);
    assert!(qs.is_ok());

    // Wrong size → error.
    let data = vec![0u8; 17];
    let qs = QuantizedStorage::new(data, &[32], QuantType::Q4_0);
    assert!(qs.is_err());
}

#[test]
fn test_quantized_storage_rejects_non_block_aligned() {
    // 33 elements not aligned to block size 32.
    let data = vec![0u8; 100];
    let qs = QuantizedStorage::new(data, &[33], QuantType::Q4_0);
    assert!(qs.is_err());
}

#[test]
fn test_quantized_storage_shape() {
    let data = vec![0u8; 36]; // 2 blocks of Q4_0 (2*18)
    let qs = QuantizedStorage::new(data, &[2, 32], QuantType::Q4_0).unwrap();
    assert_eq!(qs.shape(), &[2, 32]);
    assert_eq!(qs.quant_type(), QuantType::Q4_0);
}

// -- DynTensor::from_quantized tests ------------------------------------------

#[test]
fn test_from_quantized_shape() {
    // All-zero Q4_0 data (scale=0, all nibbles zero).
    let data = vec![0u8; 18];
    let t = DynTensor::from_quantized(&data, QuantType::Q4_0, &[32]).unwrap();
    assert_eq!(t.dims(), &[32]);
    assert_eq!(t.dtype(), DType::F32);
    assert!(t.is_quantized());
}

#[test]
fn test_from_quantized_2d_shape() {
    // 2 blocks = 64 elements, shape [2, 32].
    let data = vec![0u8; 36];
    let t = DynTensor::from_quantized(&data, QuantType::Q4_0, &[2, 32]).unwrap();
    assert_eq!(t.dims(), &[2, 32]);
    assert_eq!(t.rank(), 2);
}

#[test]
fn test_from_quantized_rejects_bad_size() {
    let data = vec![0u8; 10]; // Wrong size for 32 Q4_0 elements.
    let result = DynTensor::from_quantized(&data, QuantType::Q4_0, &[32]);
    assert!(result.is_err());
}

// -- DynTensor::is_quantized --------------------------------------------------

#[test]
fn test_is_quantized_true_for_quantized() {
    let data = vec![0u8; 18];
    let t = DynTensor::from_quantized(&data, QuantType::Q4_0, &[32]).unwrap();
    assert!(t.is_quantized());
}

#[test]
fn test_is_quantized_false_for_float() {
    let t = DynTensor::zeros(&[32], DType::F32, &crate::Device::Cpu).unwrap();
    assert!(!t.is_quantized());
}

#[test]
fn test_is_quantized_false_for_integer() {
    let t = DynTensor::zeros(&[32], DType::U32, &crate::Device::Cpu).unwrap();
    assert!(!t.is_quantized());
}

// -- DynTensor::dequantize tests ----------------------------------------------

#[test]
fn test_dequantize_q4_0_produces_f32() {
    // Scale = 1.0, all nibbles = 8 → dequant = 1.0 * (8 - 8) = 0.0
    let mut nibbles = [8u8; 32];
    // Set first nibble to 10 → dequant = 1.0 * (10 - 8) = 2.0
    nibbles[0] = 10;
    let data = make_q4_0_block(1.0, &nibbles);

    let t = DynTensor::from_quantized(&data, QuantType::Q4_0, &[32]).unwrap();
    let deq = t.dequantize().unwrap();

    assert_eq!(deq.dtype(), DType::F32);
    assert_eq!(deq.dims(), &[32]);
    assert!(!deq.is_quantized());

    let vals = deq.to_f32_array().unwrap();
    assert!(
        (vals[[0]] - 2.0).abs() < 1e-3,
        "first element should be ~2.0, got {}",
        vals[[0]]
    );
    assert!(
        (vals[[1]] - 0.0).abs() < 1e-3,
        "second element should be ~0.0, got {}",
        vals[[1]]
    );
}

#[test]
fn test_dequantize_q4_1_correctness() {
    // d=2.0, m=1.0, all nibbles = 0 → dequant = 2.0 * 0 + 1.0 = 1.0
    let nibbles = [0u8; 32];
    let data = make_q4_1_block(2.0, 1.0, &nibbles);

    let t = DynTensor::from_quantized(&data, QuantType::Q4_1, &[32]).unwrap();
    let deq = t.dequantize().unwrap();

    assert_eq!(deq.dtype(), DType::F32);
    assert_eq!(deq.dims(), &[32]);

    let vals = deq.to_f32_array().unwrap();
    for &v in vals.iter() {
        assert!((v - 1.0).abs() < 1e-2, "expected ~1.0, got {v}");
    }
}

#[test]
fn test_dequantize_q8_0_correctness() {
    // Scale = 0.5, all values = 4 → dequant = 0.5 * 4 = 2.0
    let values = [4i8; 32];
    let data = make_q8_0_block(0.5, &values);

    let t = DynTensor::from_quantized(&data, QuantType::Q8_0, &[32]).unwrap();
    let deq = t.dequantize().unwrap();

    assert_eq!(deq.dtype(), DType::F32);
    let vals = deq.to_f32_array().unwrap();
    for &v in vals.iter() {
        assert!((v - 2.0).abs() < 1e-2, "expected ~2.0, got {v}");
    }
}

#[test]
fn test_dequantize_preserves_shape_2d() {
    let data = vec![0u8; 36]; // 2 blocks Q4_0
    let t = DynTensor::from_quantized(&data, QuantType::Q4_0, &[2, 32]).unwrap();
    let deq = t.dequantize().unwrap();
    assert_eq!(deq.dims(), &[2, 32]);
}

#[test]
fn test_dequantize_noop_on_float() {
    let t = DynTensor::zeros(&[4], DType::F32, &crate::Device::Cpu).unwrap();
    let deq = t.dequantize().unwrap();
    assert_eq!(deq.dims(), &[4]);
    assert_eq!(deq.dtype(), DType::F32);
    assert!(!deq.is_quantized());
}

// -- Auto-dequantize on operations --------------------------------------------

#[test]
fn test_auto_dequantize_on_add() {
    // Create a quantized tensor with all zeros.
    let data = vec![0u8; 18];
    let q = DynTensor::from_quantized(&data, QuantType::Q4_0, &[32]).unwrap();

    // Create a float tensor with all ones.
    let ones = DynTensor::ones(&[32], DType::F32, &crate::Device::Cpu).unwrap();

    // Adding quantized + float should auto-dequantize.
    let result = q.add(&ones).unwrap();
    assert_eq!(result.dims(), &[32]);
    assert!(!result.is_quantized());

    let vals = result.to_f32_array().unwrap();
    for &v in vals.iter() {
        assert!((v - 1.0).abs() < 1e-3, "expected ~1.0, got {v}");
    }
}

#[test]
fn test_auto_dequantize_on_matmul() {
    // Create two quantized tensors shaped [32, 32] and [32, 32].
    // All zeros → matmul = zeros.
    let data = vec![0u8; 18 * 32]; // 32 blocks of Q4_0 = 32*32 elements
    let a = DynTensor::from_quantized(&data, QuantType::Q4_0, &[32, 32]).unwrap();
    let b = DynTensor::from_quantized(&data, QuantType::Q4_0, &[32, 32]).unwrap();

    let result = a.matmul(&b).unwrap();
    assert_eq!(result.dims(), &[32, 32]);
    assert!(!result.is_quantized());
}

#[test]
fn test_to_f32_array_auto_dequantizes() {
    let values = [2i8; 32];
    let data = make_q8_0_block(1.0, &values);

    let t = DynTensor::from_quantized(&data, QuantType::Q8_0, &[32]).unwrap();

    // to_f32_array should auto-dequantize.
    let arr = t.to_f32_array().unwrap();
    assert_eq!(arr.len(), 32);
    for &v in arr.iter() {
        assert!((v - 2.0).abs() < 1e-2, "expected ~2.0, got {v}");
    }
}

#[test]
fn test_quantized_tensor_device_is_cpu() {
    let data = vec![0u8; 18];
    let t = DynTensor::from_quantized(&data, QuantType::Q4_0, &[32]).unwrap();
    assert_eq!(t.device(), crate::Device::Cpu);
}

#[test]
fn test_quantized_storage_accessor() {
    let data = vec![0u8; 34];
    let t = DynTensor::from_quantized(&data, QuantType::Q8_0, &[32]).unwrap();
    let qs = t.quantized_storage().unwrap();
    assert_eq!(qs.quant_type(), QuantType::Q8_0);
    assert_eq!(qs.shape(), &[32]);
    assert_eq!(qs.raw_data().len(), 34);
}

#[test]
fn test_non_quantized_has_no_quantized_storage() {
    let t = DynTensor::zeros(&[32], DType::F32, &crate::Device::Cpu).unwrap();
    assert!(t.quantized_storage().is_none());
}

// =========================================================================
// Dequantization roundtrip accuracy tests
// =========================================================================

// -- Q4_0 roundtrip: known f32 -> quantized block -> dequantize -> check error --

#[test]
fn test_q4_0_roundtrip_positive_values() {
    // Q4_0 dequant: val = scale * (nibble - 8).
    // Representable values with scale=0.5: {-4.0, -3.5, ..., 0.0, ..., 3.0, 3.5}
    // Use scale=0.5, nibbles [0..15, 0..15] to cover full range.
    let scale = 0.5_f32;
    let mut nibbles = [0u8; 32];
    for i in 0..16 {
        nibbles[i] = i as u8; // 0..15
        nibbles[16 + i] = 15 - i as u8; // 15..0
    }
    let data = make_q4_0_block(scale, &nibbles);
    let t = DynTensor::from_quantized(&data, QuantType::Q4_0, &[32]).unwrap();
    let deq = t.dequantize().unwrap();
    let vals = deq.to_f32_array().unwrap();

    for i in 0..16 {
        let expected = scale * (f32::from(nibbles[i]) - 8.0);
        assert!(
            (vals[[i]] - expected).abs() < 1e-3,
            "element {i}: expected {expected}, got {}",
            vals[[i]]
        );
    }
    for i in 16..32 {
        let expected = scale * (f32::from(nibbles[i]) - 8.0);
        assert!(
            (vals[[i]] - expected).abs() < 1e-3,
            "element {i}: expected {expected}, got {}",
            vals[[i]]
        );
    }
}

#[test]
fn test_q4_0_roundtrip_accuracy_bound() {
    // Q4_0 has 4-bit signed quantization (16 levels spanning [-8, 7] * scale).
    // For a given scale, max quantization error is scale/2 (half a step).
    // Use scale=1.0: step size = 1.0, max error = 0.5.
    let scale = 1.0_f32;
    // Set nibbles to represent the positive range: 8, 9, 10, ..., 15, 8, 9, ...
    let mut nibbles = [0u8; 32];
    for i in 0..32 {
        nibbles[i] = 8 + (i as u8 % 8); // nibbles 8-15 -> values 0-7
    }
    let data = make_q4_0_block(scale, &nibbles);
    let t = DynTensor::from_quantized(&data, QuantType::Q4_0, &[32]).unwrap();
    let deq = t.dequantize().unwrap();
    let vals = deq.to_f32_array().unwrap();

    for i in 0..32 {
        let expected = scale * (f32::from(nibbles[i]) - 8.0);
        let error = (vals[[i]] - expected).abs();
        // f16 scale introduces small rounding; allow 1e-3
        assert!(
            error < 1e-3,
            "element {i}: error {error} exceeds tolerance for Q4_0"
        );
    }
}

#[test]
fn test_q4_0_roundtrip_negative_values() {
    // All nibbles = 0 -> value = scale * (0 - 8) = -8 * scale
    let scale = 0.25_f32;
    let nibbles = [0u8; 32];
    let data = make_q4_0_block(scale, &nibbles);
    let t = DynTensor::from_quantized(&data, QuantType::Q4_0, &[32]).unwrap();
    let deq = t.dequantize().unwrap();
    let vals = deq.to_f32_array().unwrap();

    let expected = scale * -8.0;
    for i in 0..32 {
        assert!(
            (vals[[i]] - expected).abs() < 1e-3,
            "element {i}: expected {expected}, got {}",
            vals[[i]]
        );
    }
}

// -- Q4_1 roundtrip: asymmetric quantization with offset --

#[test]
fn test_q4_1_roundtrip_with_offset() {
    // Q4_1 dequant: val = d * nibble + m.
    // d=1.0, m=5.0, nibble=3 -> 1.0*3 + 5.0 = 8.0
    let nibbles = [3u8; 32];
    let data = make_q4_1_block(1.0, 5.0, &nibbles);
    let t = DynTensor::from_quantized(&data, QuantType::Q4_1, &[32]).unwrap();
    let deq = t.dequantize().unwrap();
    let vals = deq.to_f32_array().unwrap();

    for i in 0..32 {
        assert!(
            (vals[[i]] - 8.0).abs() < 1e-2,
            "element {i}: expected ~8.0, got {}",
            vals[[i]]
        );
    }
}

#[test]
fn test_q4_1_roundtrip_full_nibble_range() {
    // d=0.5, m=0.0, nibbles 0..15 repeated -> values 0.0, 0.5, ..., 7.5
    let d = 0.5_f32;
    let m = 0.0_f32;
    let mut nibbles = [0u8; 32];
    for i in 0..32 {
        nibbles[i] = (i as u8) % 16;
    }
    let data = make_q4_1_block(d, m, &nibbles);
    let t = DynTensor::from_quantized(&data, QuantType::Q4_1, &[32]).unwrap();
    let deq = t.dequantize().unwrap();
    let vals = deq.to_f32_array().unwrap();

    for i in 0..32 {
        let expected = d * f32::from(nibbles[i]) + m;
        assert!(
            (vals[[i]] - expected).abs() < 1e-2,
            "element {i}: expected {expected}, got {}",
            vals[[i]]
        );
    }
}

#[test]
fn test_q4_1_roundtrip_large_offset() {
    // d=0.0, m=100.0 -> all values = 100.0 regardless of nibble
    let nibbles = [15u8; 32]; // max nibble, but d=0 so irrelevant
    let data = make_q4_1_block(0.0, 100.0, &nibbles);
    let t = DynTensor::from_quantized(&data, QuantType::Q4_1, &[32]).unwrap();
    let deq = t.dequantize().unwrap();
    let vals = deq.to_f32_array().unwrap();

    for i in 0..32 {
        assert!(
            (vals[[i]] - 100.0).abs() < 0.5,
            "element {i}: expected ~100.0, got {}",
            vals[[i]]
        );
    }
}

// -- Q8_0 roundtrip: 8-bit precision --

#[test]
fn test_q8_0_roundtrip_signed_range() {
    // Q8_0 dequant: val = scale * q (q is i8).
    // scale=0.1, values span [-128, 127].
    let scale = 0.1_f32;
    let mut values = [0i8; 32];
    for i in 0..32 {
        values[i] = -16 + i as i8; // -16 to 15
    }
    let data = make_q8_0_block(scale, &values);
    let t = DynTensor::from_quantized(&data, QuantType::Q8_0, &[32]).unwrap();
    let deq = t.dequantize().unwrap();
    let vals = deq.to_f32_array().unwrap();

    for i in 0..32 {
        let expected = scale * f32::from(values[i]);
        // f16 scale can introduce small error; use tolerance relative to scale
        assert!(
            (vals[[i]] - expected).abs() < 0.02,
            "element {i}: expected {expected}, got {}",
            vals[[i]]
        );
    }
}

#[test]
fn test_q8_0_roundtrip_extreme_values() {
    // scale=1.0, i8::MIN (-128) and i8::MAX (127)
    let scale = 1.0_f32;
    let mut values = [0i8; 32];
    values[0] = i8::MIN; // -128
    values[1] = i8::MAX; // 127
    values[2] = 0;
    values[3] = 1;
    values[4] = -1;
    for i in 5..32 {
        values[i] = (i as i8) - 16;
    }
    let data = make_q8_0_block(scale, &values);
    let t = DynTensor::from_quantized(&data, QuantType::Q8_0, &[32]).unwrap();
    let deq = t.dequantize().unwrap();
    let vals = deq.to_f32_array().unwrap();

    assert!(
        (vals[[0]] - (-128.0)).abs() < 0.5,
        "i8::MIN: got {}",
        vals[[0]]
    );
    assert!(
        (vals[[1]] - 127.0).abs() < 0.5,
        "i8::MAX: got {}",
        vals[[1]]
    );
    assert!((vals[[2]] - 0.0).abs() < 1e-3, "zero: got {}", vals[[2]]);
    assert!((vals[[3]] - 1.0).abs() < 1e-3, "+1: got {}", vals[[3]]);
    assert!((vals[[4]] - (-1.0)).abs() < 1e-3, "-1: got {}", vals[[4]]);
}

#[test]
fn test_q8_0_roundtrip_accuracy_bound() {
    // Q8_0 with scale=0.01: max representable = 1.27, min = -1.28.
    // Step size = 0.01, max quantization error = 0.005.
    let scale = 0.01_f32;
    let mut values = [0i8; 32];
    for i in 0..32 {
        values[i] = (i as i8) * 4 - 64; // -64, -60, ..., 60
    }
    let data = make_q8_0_block(scale, &values);
    let t = DynTensor::from_quantized(&data, QuantType::Q8_0, &[32]).unwrap();
    let deq = t.dequantize().unwrap();
    let vals = deq.to_f32_array().unwrap();

    for i in 0..32 {
        let expected = scale * f32::from(values[i]);
        let error = (vals[[i]] - expected).abs();
        // f16 introduces up to ~1e-3 relative error on the scale
        assert!(
            error < 0.01,
            "element {i}: error {error} exceeds Q8_0 tolerance"
        );
    }
}

// =========================================================================
// Multi-block roundtrip tests
// =========================================================================

#[test]
fn test_q4_0_multi_block_varying_scales() {
    // 2 blocks with different scales: block0 scale=1.0, block1 scale=2.0
    // Block0: nibbles all 10 -> val = 1.0*(10-8) = 2.0
    // Block1: nibbles all 5  -> val = 2.0*(5-8) = -6.0
    let block0 = make_q4_0_block(1.0, &[10u8; 32]);
    let block1 = make_q4_0_block(2.0, &[5u8; 32]);
    let mut data = block0;
    data.extend_from_slice(&block1);

    let t = DynTensor::from_quantized(&data, QuantType::Q4_0, &[64]).unwrap();
    let deq = t.dequantize().unwrap();
    let vals = deq.to_f32_array().unwrap();

    for i in 0..32 {
        assert!(
            (vals[[i]] - 2.0).abs() < 1e-3,
            "block0 element {i}: expected 2.0, got {}",
            vals[[i]]
        );
    }
    for i in 32..64 {
        assert!(
            (vals[[i]] - (-6.0)).abs() < 1e-2,
            "block1 element {i}: expected -6.0, got {}",
            vals[[i]]
        );
    }
}

#[test]
fn test_q8_0_multi_block_3d_shape() {
    // 4 blocks = 128 elements, shape [2, 2, 32]
    let block = make_q8_0_block(0.5, &[10i8; 32]);
    let mut data = Vec::new();
    for _ in 0..4 {
        data.extend_from_slice(&block);
    }

    let t = DynTensor::from_quantized(&data, QuantType::Q8_0, &[2, 2, 32]).unwrap();
    assert_eq!(t.dims(), &[2, 2, 32]);
    let deq = t.dequantize().unwrap();
    assert_eq!(deq.dims(), &[2, 2, 32]);

    let vals = deq.to_f32_array().unwrap();
    let expected = 0.5 * 10.0;
    for &v in vals.iter() {
        assert!((v - expected).abs() < 0.1, "expected ~{expected}, got {v}");
    }
}

#[test]
fn test_q4_1_multi_block_different_offsets() {
    // Block0: d=1.0, m=0.0, nibbles=7 -> val = 7.0
    // Block1: d=1.0, m=10.0, nibbles=3 -> val = 13.0
    let block0 = make_q4_1_block(1.0, 0.0, &[7u8; 32]);
    let block1 = make_q4_1_block(1.0, 10.0, &[3u8; 32]);
    let mut data = block0;
    data.extend_from_slice(&block1);

    let t = DynTensor::from_quantized(&data, QuantType::Q4_1, &[64]).unwrap();
    let deq = t.dequantize().unwrap();
    let vals = deq.to_f32_array().unwrap();

    for i in 0..32 {
        assert!(
            (vals[[i]] - 7.0).abs() < 1e-2,
            "block0 element {i}: expected 7.0, got {}",
            vals[[i]]
        );
    }
    for i in 32..64 {
        assert!(
            (vals[[i]] - 13.0).abs() < 0.5,
            "block1 element {i}: expected 13.0, got {}",
            vals[[i]]
        );
    }
}

// =========================================================================
// Empty and zero-element tensor handling
// =========================================================================

#[test]
fn test_quantized_storage_empty_tensor() {
    // 0 elements should produce a valid storage with 0 bytes.
    let data = vec![];
    let qs = QuantizedStorage::new(data, &[0], QuantType::Q4_0);
    assert!(qs.is_ok());
    let qs = qs.unwrap();
    assert_eq!(qs.shape(), &[0]);
    assert_eq!(qs.raw_data().len(), 0);
}

#[test]
fn test_quantized_storage_empty_dequantize() {
    let data = vec![];
    let qs = QuantizedStorage::new(data, &[0], QuantType::Q8_0).unwrap();
    let arr = qs.dequantize().unwrap();
    assert_eq!(arr.len(), 0);
    assert_eq!(arr.shape(), &[0]);
}

#[test]
fn test_dyntensor_from_quantized_empty() {
    let data = vec![];
    let t = DynTensor::from_quantized(&data, QuantType::Q4_0, &[0]).unwrap();
    assert_eq!(t.dims(), &[0]);
    assert!(t.is_quantized());
    let deq = t.dequantize().unwrap();
    assert_eq!(deq.dims(), &[0]);
    assert!(!deq.is_quantized());
}

#[test]
fn test_quantized_storage_multidim_with_zero() {
    // Shape [0, 32] -> 0 total elements.
    let data = vec![];
    let qs = QuantizedStorage::new(data, &[0, 32], QuantType::Q4_0);
    assert!(qs.is_ok());
    assert_eq!(qs.unwrap().shape(), &[0, 32]);
}

// =========================================================================
// Error path tests
// =========================================================================

#[test]
fn test_quantized_storage_data_too_short() {
    // 32 Q4_0 elements need 18 bytes; provide only 16.
    let data = vec![0u8; 16];
    let result = QuantizedStorage::new(data, &[32], QuantType::Q4_0);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("18") || err.contains("16") || err.contains("mismatch"),
        "error should mention expected/actual size: {err}"
    );
}

#[test]
fn test_quantized_storage_data_too_long() {
    // 32 Q4_0 elements need 18 bytes; provide 20.
    let data = vec![0u8; 20];
    let result = QuantizedStorage::new(data, &[32], QuantType::Q4_0);
    assert!(result.is_err());
}

#[test]
fn test_quantized_storage_not_block_aligned_q8_0() {
    // 31 elements not aligned to block size 32 for Q8_0.
    let data = vec![0u8; 100];
    let result = QuantizedStorage::new(data, &[31], QuantType::Q8_0);
    assert!(result.is_err());
}

#[test]
fn test_quantized_storage_not_block_aligned_q4_1() {
    let data = vec![0u8; 100];
    let result = QuantizedStorage::new(data, &[17], QuantType::Q4_1);
    assert!(result.is_err());
}

#[test]
fn test_from_quantized_wrong_bytes_for_q8_0() {
    // 32 Q8_0 elements need 34 bytes; provide 32.
    let data = vec![0u8; 32];
    let result = DynTensor::from_quantized(&data, QuantType::Q8_0, &[32]);
    assert!(result.is_err());
}

#[test]
fn test_from_quantized_wrong_bytes_for_q4_1() {
    // 32 Q4_1 elements need 20 bytes; provide 18.
    let data = vec![0u8; 18];
    let result = DynTensor::from_quantized(&data, QuantType::Q4_1, &[32]);
    assert!(result.is_err());
}

// =========================================================================
// Finiteness and NaN safety
// =========================================================================

#[test]
fn test_q4_0_dequant_all_outputs_finite() {
    // Arbitrary non-zero data should produce finite outputs.
    let mut nibbles = [0u8; 32];
    for i in 0..32 {
        nibbles[i] = ((i * 7 + 3) % 16) as u8;
    }
    let data = make_q4_0_block(3.14, &nibbles);
    let t = DynTensor::from_quantized(&data, QuantType::Q4_0, &[32]).unwrap();
    let deq = t.dequantize().unwrap();
    let vals = deq.to_f32_array().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "Q4_0 element {i} not finite: {v}");
    }
}

#[test]
fn test_q4_1_dequant_all_outputs_finite() {
    let mut nibbles = [0u8; 32];
    for i in 0..32 {
        nibbles[i] = ((i * 11 + 5) % 16) as u8;
    }
    let data = make_q4_1_block(2.71, 0.5, &nibbles);
    let t = DynTensor::from_quantized(&data, QuantType::Q4_1, &[32]).unwrap();
    let deq = t.dequantize().unwrap();
    let vals = deq.to_f32_array().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "Q4_1 element {i} not finite: {v}");
    }
}

#[test]
fn test_q8_0_dequant_all_outputs_finite() {
    let mut values = [0i8; 32];
    for i in 0..32 {
        values[i] = ((i as i32 * 13 - 64) % 128) as i8;
    }
    let data = make_q8_0_block(1.5, &values);
    let t = DynTensor::from_quantized(&data, QuantType::Q8_0, &[32]).unwrap();
    let deq = t.dequantize().unwrap();
    let vals = deq.to_f32_array().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "Q8_0 element {i} not finite: {v}");
    }
}

// =========================================================================
// Zero-scale edge case
// =========================================================================

#[test]
fn test_q4_0_zero_scale_all_zeros() {
    // scale=0.0 means all outputs must be 0.0 regardless of nibble values.
    let nibbles = [15u8; 32]; // max nibble, but scale is zero
    let data = make_q4_0_block(0.0, &nibbles);
    let t = DynTensor::from_quantized(&data, QuantType::Q4_0, &[32]).unwrap();
    let deq = t.dequantize().unwrap();
    let vals = deq.to_f32_array().unwrap();
    for i in 0..32 {
        assert_eq!(
            vals[[i]],
            0.0,
            "zero scale should produce 0.0, got {}",
            vals[[i]]
        );
    }
}

#[test]
fn test_q8_0_zero_scale_all_zeros() {
    let values = [i8::MAX; 32];
    let data = make_q8_0_block(0.0, &values);
    let t = DynTensor::from_quantized(&data, QuantType::Q8_0, &[32]).unwrap();
    let deq = t.dequantize().unwrap();
    let vals = deq.to_f32_array().unwrap();
    for i in 0..32 {
        assert_eq!(
            vals[[i]],
            0.0,
            "zero scale should produce 0.0, got {}",
            vals[[i]]
        );
    }
}

#[test]
fn test_q4_1_zero_d_uses_m_only() {
    // d=0.0, m=42.0 -> all outputs = 42.0
    let nibbles = [8u8; 32]; // arbitrary; d=0 nullifies them
    let data = make_q4_1_block(0.0, 42.0, &nibbles);
    let t = DynTensor::from_quantized(&data, QuantType::Q4_1, &[32]).unwrap();
    let deq = t.dequantize().unwrap();
    let vals = deq.to_f32_array().unwrap();
    for i in 0..32 {
        assert!(
            (vals[[i]] - 42.0).abs() < 0.5,
            "element {i}: expected ~42.0, got {}",
            vals[[i]]
        );
    }
}

// =========================================================================
// QuantType expected_bytes edge cases
// =========================================================================

#[test]
fn test_quant_type_expected_bytes_all_types() {
    // Q4_1: 20 bytes per 32 elements
    assert_eq!(QuantType::Q4_1.expected_bytes(32), Some(20));
    assert_eq!(QuantType::Q4_1.expected_bytes(64), Some(40));
    assert_eq!(QuantType::Q4_1.expected_bytes(0), Some(0));
    assert_eq!(QuantType::Q4_1.expected_bytes(31), None);

    // Q8_0: 34 bytes per 32 elements
    assert_eq!(QuantType::Q8_0.expected_bytes(32), Some(34));
    assert_eq!(QuantType::Q8_0.expected_bytes(96), Some(102));
    assert_eq!(QuantType::Q8_0.expected_bytes(0), Some(0));
    assert_eq!(QuantType::Q8_0.expected_bytes(1), None);
}

// =========================================================================
// Cross-type comparison: Q8_0 should be more accurate than Q4_0
// =========================================================================

#[test]
fn test_q8_0_more_accurate_than_q4_0_for_same_range() {
    // Both formats quantize the same "target" float values.
    // Q8_0 (256 levels) should have smaller max error than Q4_0 (16 levels).
    //
    // Target value: 3.0 with scale=1.0.
    // Q4_0: nibble=11 -> 1.0*(11-8) = 3.0 (exact)
    // Q8_0: q=3 -> 1.0*3 = 3.0 (exact)
    // For a value like 2.7:
    // Q4_0: closest nibble=11 -> 3.0, error = 0.3
    // Q8_0: scale=0.1, q=27 -> 2.7, error = 0.0 (with f16 rounding)

    // Build Q4_0: scale=1.0, nibbles represent integer values
    let q4_nibbles = [11u8; 32]; // all -> 3.0
    let q4_data = make_q4_0_block(1.0, &q4_nibbles);
    let q4_t = DynTensor::from_quantized(&q4_data, QuantType::Q4_0, &[32]).unwrap();
    let q4_vals = q4_t.dequantize().unwrap().to_f32_array().unwrap();

    // Build Q8_0: scale=0.1, q=27 -> 2.7
    let q8_values = [27i8; 32];
    let q8_data = make_q8_0_block(0.1, &q8_values);
    let q8_t = DynTensor::from_quantized(&q8_data, QuantType::Q8_0, &[32]).unwrap();
    let q8_vals = q8_t.dequantize().unwrap().to_f32_array().unwrap();

    // Q4_0 represents 3.0 exactly; Q8_0 represents 2.7 with ~f16 error
    // The key insight: Q8_0 can represent 2.7 while Q4_0 cannot (nearest is 3.0).
    let target_2_7 = 2.7_f32;
    let q4_error = (q4_vals[[0]] - target_2_7).abs();
    let q8_error = (q8_vals[[0]] - target_2_7).abs();

    assert!(
        q8_error < q4_error,
        "Q8_0 error ({q8_error}) should be less than Q4_0 error ({q4_error}) for value 2.7"
    );
}

// =========================================================================
// Dequantize idempotency: dequantizing a non-quantized tensor is a no-op
// =========================================================================

#[test]
fn test_dequantize_idempotent() {
    let values = [5i8; 32];
    let data = make_q8_0_block(2.0, &values);
    let t = DynTensor::from_quantized(&data, QuantType::Q8_0, &[32]).unwrap();

    let deq1 = t.dequantize().unwrap();
    assert!(!deq1.is_quantized());

    // Second dequantize on already-dequantized tensor should be a no-op.
    let deq2 = deq1.dequantize().unwrap();
    assert!(!deq2.is_quantized());
    assert_eq!(deq2.dims(), deq1.dims());

    let v1 = deq1.to_f32_array().unwrap();
    let v2 = deq2.to_f32_array().unwrap();
    for i in 0..32 {
        assert_eq!(
            v1[[i]],
            v2[[i]],
            "element {i} differs after double-dequantize"
        );
    }
}
