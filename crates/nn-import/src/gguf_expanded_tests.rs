// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![allow(deprecated)]

//! Expanded GGUF quantization tests covering header parsing, quant type
//! identification, dequantization correctness, block size calculations,
//! tensor metadata extraction, ModelArchitecture detection, edge cases,
//! and byte alignment validation.
//!
//! These tests complement `quantization_tests.rs` (which covers safetensors
//! detection, Q4_0/Q4_1/Q8_0/Q4_K dequant, and basic GgufDType lookups) by
//! adding coverage for:
//! - Q5_0, Q5_1 dequantization correctness
//! - Q2_K, Q3_K, Q5_K, Q6_K dequantization correctness
//! - GGUF header parsing edge cases (truncated data, bad magic, bad version)
//! - GgufFile stream-based integration with quantized tensors
//! - GgufTensorInfo byte_size for all quant types
//! - Cross-validation between nn-gguf and nn-core dequant routines
//! - Metadata value type accessors
//! - Byte alignment validation

use std::collections::HashMap;
use std::io::Cursor;

use nn_gguf::{
    dequantize_q2_k, dequantize_q3_k, dequantize_q4_0, dequantize_q4_1, dequantize_q5_0,
    dequantize_q5_1, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0, GgufDType, GgufFile,
    GgufHeader, GgufMetadata, GgufMetadataValue, GgufTensorInfo, ModelArchitecture,
};

// --- Header Parsing ----------------------------------------------------------

const GGUF_MAGIC: u32 = 0x4647_5547;

/// Build a valid GGUF v3 header for `tensor_count` tensors and `kv_count`
/// metadata entries.
fn build_header(tensor_count: u64, kv_count: u64) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&tensor_count.to_le_bytes());
    data.extend_from_slice(&kv_count.to_le_bytes());
    data
}

#[test]
fn test_header_valid_zero_tensors() {
    let data = build_header(0, 0);
    let header = GgufHeader::read_from(&mut &data[..]).unwrap();
    assert_eq!(header.version, 3);
    assert_eq!(header.tensor_count, 0);
    assert_eq!(header.metadata_kv_count, 0);
}

#[test]
fn test_header_large_counts() {
    // GgufHeader enforces MAX_TENSOR_COUNT/MAX_METADATA_KV_COUNT = 100_000 as a
    // security bound, so "large counts" means large-but-valid (<= the cap).
    let data = build_header(100_000, 50_000);
    let header = GgufHeader::read_from(&mut &data[..]).unwrap();
    assert_eq!(header.tensor_count, 100_000);
    assert_eq!(header.metadata_kv_count, 50_000);
}

#[test]
fn test_header_truncated_at_magic() {
    // Only 3 bytes -- can't even read the magic u32.
    let data = &GGUF_MAGIC.to_le_bytes()[..3];
    let result = GgufHeader::read_from(&mut &data[..]);
    assert!(result.is_err());
}

#[test]
fn test_header_truncated_after_version() {
    // Magic + version but no tensor_count/kv_count.
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    let result = GgufHeader::read_from(&mut &data[..]);
    assert!(result.is_err());
}

#[test]
fn test_header_wrong_magic() {
    let mut data = Vec::new();
    data.extend_from_slice(&0x12345678u32.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("invalid magic"), "unexpected error: {msg}");
}

#[test]
fn test_header_version_2_rejected() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("version"), "unexpected error: {msg}");
}

#[test]
fn test_header_version_4_rejected() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("version"), "unexpected error: {msg}");
}

// --- GgufDType identification ------------------------------------------------

#[test]
fn test_dtype_from_u32_gap_values_return_none() {
    // Type IDs 4 and 5 are not assigned in the GGUF spec.
    assert!(GgufDType::from_u32(4).is_none());
    assert!(GgufDType::from_u32(5).is_none());
    // 15 is also a gap.
    assert!(GgufDType::from_u32(15).is_none());
    // Very large.
    assert!(GgufDType::from_u32(999).is_none());
    assert!(GgufDType::from_u32(u32::MAX).is_none());
}

#[test]
fn test_dtype_from_u32_boundary_values() {
    // Confirm all standard types round-trip.
    let expected: Vec<(u32, GgufDType)> = vec![
        (0, GgufDType::F32),
        (1, GgufDType::F16),
        (2, GgufDType::Q4_0),
        (3, GgufDType::Q4_1),
        (6, GgufDType::Q5_0),
        (7, GgufDType::Q5_1),
        (8, GgufDType::Q8_0),
        (9, GgufDType::Q8_1),
        (10, GgufDType::Q2K),
        (11, GgufDType::Q3K),
        (12, GgufDType::Q4K),
        (13, GgufDType::Q5K),
        (14, GgufDType::Q6K),
        (30, GgufDType::BF16),
    ];
    for (id, dtype) in expected {
        assert_eq!(
            GgufDType::from_u32(id),
            Some(dtype),
            "from_u32({id}) should be {dtype:?}"
        );
    }
}

// --- Block size and type_size for all types ----------------------------------

#[test]
fn test_block_size_standard_quant() {
    assert_eq!(GgufDType::Q4_0.block_size(), 32);
    assert_eq!(GgufDType::Q4_1.block_size(), 32);
    assert_eq!(GgufDType::Q5_0.block_size(), 32);
    assert_eq!(GgufDType::Q5_1.block_size(), 32);
    assert_eq!(GgufDType::Q8_0.block_size(), 32);
    assert_eq!(GgufDType::Q8_1.block_size(), 32);
}

#[test]
fn test_block_size_k_quant() {
    assert_eq!(GgufDType::Q2K.block_size(), 256);
    assert_eq!(GgufDType::Q3K.block_size(), 256);
    assert_eq!(GgufDType::Q4K.block_size(), 256);
    assert_eq!(GgufDType::Q5K.block_size(), 256);
    assert_eq!(GgufDType::Q6K.block_size(), 256);
}

#[test]
fn test_block_size_non_quantized_is_1() {
    assert_eq!(GgufDType::F32.block_size(), 1);
    assert_eq!(GgufDType::F16.block_size(), 1);
    assert_eq!(GgufDType::BF16.block_size(), 1);
}

#[test]
fn test_type_size_standard_quant() {
    assert_eq!(GgufDType::Q4_0.type_size(), 18);
    assert_eq!(GgufDType::Q4_1.type_size(), 20);
    assert_eq!(GgufDType::Q5_0.type_size(), 22);
    assert_eq!(GgufDType::Q5_1.type_size(), 24);
    assert_eq!(GgufDType::Q8_0.type_size(), 34);
    assert_eq!(GgufDType::Q8_1.type_size(), 40);
}

#[test]
fn test_type_size_k_quant() {
    assert_eq!(GgufDType::Q2K.type_size(), 84);
    assert_eq!(GgufDType::Q3K.type_size(), 110);
    assert_eq!(GgufDType::Q4K.type_size(), 144);
    assert_eq!(GgufDType::Q5K.type_size(), 176);
    assert_eq!(GgufDType::Q6K.type_size(), 210);
}

#[test]
fn test_type_size_float_types() {
    assert_eq!(GgufDType::F32.type_size(), 4);
    assert_eq!(GgufDType::F16.type_size(), 2);
    assert_eq!(GgufDType::BF16.type_size(), 2);
    assert_eq!(GgufDType::F64.type_size(), 8);
}

#[test]
fn test_type_size_integer_types() {
    assert_eq!(GgufDType::I8.type_size(), 1);
    assert_eq!(GgufDType::I16.type_size(), 2);
    assert_eq!(GgufDType::I32.type_size(), 4);
    assert_eq!(GgufDType::I64.type_size(), 8);
}

// --- Q5_0 dequantization -----------------------------------------------------

/// Build one Q5_0 block (22 bytes per 32 elements).
///
/// Layout: [f16 scale][4 bytes high bits (u32 LE)][16 bytes: 32 x 4-bit low]
/// Dequant: q = (lo4 | (hi1 << 4)), val = scale * (q - 16)
fn build_q5_0_block(scale_f32: f32, high_bits: u32, low_nibbles: &[u8; 16]) -> Vec<u8> {
    let mut block = Vec::with_capacity(22);
    block.extend_from_slice(&half::f16::from_f32(scale_f32).to_le_bytes());
    block.extend_from_slice(&high_bits.to_le_bytes());
    block.extend_from_slice(low_nibbles);
    block
}

#[test]
fn test_q5_0_dequant_all_zeros() {
    let block = build_q5_0_block(0.0, 0, &[0u8; 16]);
    let result = dequantize_q5_0(&block, 32);
    assert_eq!(result.len(), 32);
    // scale=0 => all output should be 0 regardless of quant values.
    for (i, &v) in result.iter().enumerate() {
        assert_eq!(v, 0.0, "index {i}");
    }
}

#[test]
fn test_q5_0_dequant_known_values() {
    // Scale=1.0, all high bits=0, all low nibbles=0 => q=0, val=1.0*(0-16)=-16.0
    let block = build_q5_0_block(1.0, 0, &[0u8; 16]);
    let result = dequantize_q5_0(&block, 32);
    assert_eq!(result.len(), 32);
    for &v in &result {
        assert!((v - (-16.0)).abs() < 0.01, "expected -16.0, got {v}");
    }
}

#[test]
fn test_q5_0_dequant_with_high_bits() {
    // Scale=1.0, all high bits=1, all low nibbles=0
    // => q = 0 | (1<<4) = 16, val = 1.0*(16-16) = 0.0
    let block = build_q5_0_block(1.0, 0xFFFFFFFF, &[0u8; 16]);
    let result = dequantize_q5_0(&block, 32);
    assert_eq!(result.len(), 32);
    for &v in &result {
        assert!(v.abs() < 0.01, "expected 0.0, got {v}");
    }
}

#[test]
fn test_q5_0_dequant_max_q_value() {
    // Scale=1.0, all high bits=1, all low nibbles=0xF
    // => q = 15 | (1<<4) = 31, val = 1.0*(31-16) = 15.0
    let block = build_q5_0_block(1.0, 0xFFFFFFFF, &[0xFF; 16]);
    let result = dequantize_q5_0(&block, 32);
    assert_eq!(result.len(), 32);
    for &v in &result {
        assert!((v - 15.0).abs() < 0.01, "expected 15.0, got {v}");
    }
}

#[test]
fn test_q5_0_output_length_multi_block() {
    // 2 blocks = 64 elements = 44 bytes.
    let mut data = build_q5_0_block(1.0, 0, &[0u8; 16]);
    data.extend(build_q5_0_block(2.0, 0, &[0u8; 16]));
    let result = dequantize_q5_0(&data, 64);
    assert_eq!(result.len(), 64);
}

#[test]
fn test_q5_0_all_outputs_finite() {
    let block = build_q5_0_block(0.5, 0xAAAAAAAA, &[0x55; 16]);
    let result = dequantize_q5_0(&block, 32);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "index {i} is not finite: {v}");
    }
}

// --- Q5_1 dequantization -----------------------------------------------------

/// Build one Q5_1 block (24 bytes per 32 elements).
///
/// Layout: [f16 d][f16 m][4 bytes high bits (u32 LE)][16 bytes: 32 x 4-bit low]
/// Dequant: q = (lo4 | (hi1 << 4)), val = d * q + m
fn build_q5_1_block(d: f32, m: f32, high_bits: u32, low_nibbles: &[u8; 16]) -> Vec<u8> {
    let mut block = Vec::with_capacity(24);
    block.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
    block.extend_from_slice(&half::f16::from_f32(m).to_le_bytes());
    block.extend_from_slice(&high_bits.to_le_bytes());
    block.extend_from_slice(low_nibbles);
    block
}

#[test]
fn test_q5_1_dequant_offset_only() {
    // d=0, m=5.0 => all outputs are 5.0.
    let block = build_q5_1_block(0.0, 5.0, 0, &[0u8; 16]);
    let result = dequantize_q5_1(&block, 32);
    assert_eq!(result.len(), 32);
    for &v in &result {
        assert!((v - 5.0).abs() < 0.01, "expected 5.0, got {v}");
    }
}

#[test]
fn test_q5_1_dequant_known_values() {
    // d=1.0, m=0.0, all high bits=0, all low nibbles=0
    // q = 0, val = 1.0*0 + 0.0 = 0.0
    let block = build_q5_1_block(1.0, 0.0, 0, &[0u8; 16]);
    let result = dequantize_q5_1(&block, 32);
    for &v in &result {
        assert!(v.abs() < 0.01, "expected 0.0, got {v}");
    }
}

#[test]
fn test_q5_1_dequant_max_q() {
    // d=1.0, m=0.0, all high bits=1, all low nibbles=0xF
    // q = 15 | (1<<4) = 31, val = 1.0*31 + 0.0 = 31.0
    let block = build_q5_1_block(1.0, 0.0, 0xFFFFFFFF, &[0xFF; 16]);
    let result = dequantize_q5_1(&block, 32);
    for &v in &result {
        assert!((v - 31.0).abs() < 0.01, "expected 31.0, got {v}");
    }
}

#[test]
fn test_q5_1_all_outputs_non_negative_with_positive_m() {
    // d=0.5, m=1.0 => min possible value is 0.5*0 + 1.0 = 1.0.
    let block = build_q5_1_block(0.5, 1.0, 0, &[0u8; 16]);
    let result = dequantize_q5_1(&block, 32);
    for &v in &result {
        assert!(v >= 0.99, "expected >= 1.0, got {v}");
    }
}

// --- K-quant dequantization (Q2_K, Q3_K, Q5_K, Q6_K) ------------------------

#[test]
fn test_q2_k_all_zeros_block() {
    let data = vec![0u8; 84]; // One Q2_K block.
    let result = dequantize_q2_k(&data, 256);
    assert_eq!(result.len(), 256);
    for &v in &result {
        assert_eq!(v, 0.0, "expected 0.0 for all-zeros block");
    }
}

#[test]
fn test_q2_k_output_length() {
    // 2 blocks = 512 elements.
    let data = vec![0u8; 84 * 2];
    let result = dequantize_q2_k(&data, 512);
    assert_eq!(result.len(), 512);
}

#[test]
fn test_q2_k_all_outputs_finite() {
    // Fill with non-trivial pattern.
    let mut data = vec![0u8; 84];
    for (i, b) in data.iter_mut().enumerate() {
        *b = (i % 256) as u8;
    }
    let result = dequantize_q2_k(&data, 256);
    assert_eq!(result.len(), 256);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "index {i}: not finite: {v}");
    }
}

#[test]
fn test_q3_k_all_zeros_block() {
    let data = vec![0u8; 110]; // One Q3_K block.
    let result = dequantize_q3_k(&data, 256);
    assert_eq!(result.len(), 256);
    for &v in &result {
        assert_eq!(v, 0.0, "expected 0.0 for all-zeros block");
    }
}

#[test]
fn test_q3_k_output_length() {
    let data = vec![0u8; 110 * 3];
    let result = dequantize_q3_k(&data, 768);
    assert_eq!(result.len(), 768);
}

#[test]
fn test_q3_k_all_outputs_finite() {
    let mut data = vec![0u8; 110];
    for (i, b) in data.iter_mut().enumerate() {
        *b = ((i * 7) % 256) as u8;
    }
    let result = dequantize_q3_k(&data, 256);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "index {i}: not finite: {v}");
    }
}

#[test]
fn test_q5_k_all_zeros_block() {
    let data = vec![0u8; 176]; // One Q5_K block.
    let result = dequantize_q5_k(&data, 256);
    assert_eq!(result.len(), 256);
    for &v in &result {
        assert_eq!(v, 0.0, "expected 0.0 for all-zeros block");
    }
}

#[test]
fn test_q5_k_output_length() {
    let data = vec![0u8; 176 * 2];
    let result = dequantize_q5_k(&data, 512);
    assert_eq!(result.len(), 512);
}

#[test]
fn test_q5_k_all_outputs_finite() {
    let mut data = vec![0u8; 176];
    for (i, b) in data.iter_mut().enumerate() {
        *b = ((i * 13) % 256) as u8;
    }
    let result = dequantize_q5_k(&data, 256);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "index {i}: not finite: {v}");
    }
}

#[test]
fn test_q6_k_all_zeros_block() {
    let data = vec![0u8; 210]; // One Q6_K block.
    let result = dequantize_q6_k(&data, 256);
    assert_eq!(result.len(), 256);
    for &v in &result {
        assert_eq!(v, 0.0, "expected 0.0 for all-zeros block");
    }
}

#[test]
fn test_q6_k_output_length() {
    let data = vec![0u8; 210 * 4];
    let result = dequantize_q6_k(&data, 1024);
    assert_eq!(result.len(), 1024);
}

#[test]
fn test_q6_k_all_outputs_finite() {
    let mut data = vec![0u8; 210];
    for (i, b) in data.iter_mut().enumerate() {
        *b = ((i * 17) % 256) as u8;
    }
    let result = dequantize_q6_k(&data, 256);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "index {i}: not finite: {v}");
    }
}

// --- GgufTensorInfo byte_size for all quant types ----------------------------

fn make_tensor_info(shape: &[u64], dtype: GgufDType) -> GgufTensorInfo {
    GgufTensorInfo {
        name: "test".into(),
        n_dims: shape.len() as u32,
        shape: shape.to_vec(),
        dtype,
        offset: 0,
    }
}

#[test]
fn test_tensor_info_byte_size_q5_0() {
    // 256 elements / 32 per block = 8 blocks * 22 bytes = 176
    let info = make_tensor_info(&[256], GgufDType::Q5_0);
    assert_eq!(info.byte_size(), 176);
}

#[test]
fn test_tensor_info_byte_size_q5_1() {
    // 256 elements / 32 per block = 8 blocks * 24 bytes = 192
    let info = make_tensor_info(&[256], GgufDType::Q5_1);
    assert_eq!(info.byte_size(), 192);
}

#[test]
fn test_tensor_info_byte_size_q2_k() {
    // 256 elements / 256 per block = 1 block * 84 bytes = 84
    let info = make_tensor_info(&[256], GgufDType::Q2K);
    assert_eq!(info.byte_size(), 84);
}

#[test]
fn test_tensor_info_byte_size_q3_k() {
    // 512 elements / 256 per block = 2 blocks * 110 bytes = 220
    let info = make_tensor_info(&[512], GgufDType::Q3K);
    assert_eq!(info.byte_size(), 220);
}

#[test]
fn test_tensor_info_byte_size_q5_k() {
    // 256 elements / 256 per block = 1 block * 176 bytes = 176
    let info = make_tensor_info(&[256], GgufDType::Q5K);
    assert_eq!(info.byte_size(), 176);
}

#[test]
fn test_tensor_info_byte_size_q6_k() {
    // 512 elements / 256 per block = 2 blocks * 210 bytes = 420
    let info = make_tensor_info(&[512], GgufDType::Q6K);
    assert_eq!(info.byte_size(), 420);
}

#[test]
fn test_tensor_info_byte_size_multi_dim() {
    // [4, 8, 32] = 1024 elements, Q4_0
    // 1024 / 32 = 32 blocks * 18 = 576
    let info = make_tensor_info(&[4, 8, 32], GgufDType::Q4_0);
    assert_eq!(info.num_elements(), 1024);
    assert_eq!(info.byte_size(), 576);
}

#[test]
fn test_tensor_info_byte_size_f16() {
    let info = make_tensor_info(&[100], GgufDType::F16);
    assert_eq!(info.byte_size(), 200); // 100 * 2 bytes
}

#[test]
fn test_tensor_info_byte_size_bf16() {
    let info = make_tensor_info(&[100], GgufDType::BF16);
    assert_eq!(info.byte_size(), 200); // 100 * 2 bytes
}

#[test]
fn test_tensor_info_num_elements_empty_shape() {
    // Empty shape => num_elements is max(product, 1) = 1.
    let info = make_tensor_info(&[], GgufDType::F32);
    assert_eq!(info.num_elements(), 1);
}

#[test]
fn test_tensor_info_num_elements_scalar() {
    let info = make_tensor_info(&[1], GgufDType::F32);
    assert_eq!(info.num_elements(), 1);
    assert_eq!(info.byte_size(), 4);
}

// --- Metadata value accessors ------------------------------------------------

#[test]
fn test_metadata_get_u64_widens_u32() {
    let mut entries = HashMap::new();
    entries.insert("count".to_string(), GgufMetadataValue::U32(42));
    let meta = GgufMetadata { entries };
    assert_eq!(meta.get_u64("count"), Some(42));
}

#[test]
fn test_metadata_get_u64_native() {
    let mut entries = HashMap::new();
    entries.insert(
        "count".to_string(),
        GgufMetadataValue::U64(1_000_000_000_000),
    );
    let meta = GgufMetadata { entries };
    assert_eq!(meta.get_u64("count"), Some(1_000_000_000_000));
}

#[test]
fn test_metadata_get_f64_widens_f32() {
    let mut entries = HashMap::new();
    entries.insert("freq".to_string(), GgufMetadataValue::F32(10000.0));
    let meta = GgufMetadata { entries };
    let val = meta.get_f64("freq").unwrap();
    assert!((val - 10000.0).abs() < 0.01);
}

#[test]
fn test_metadata_get_str_wrong_type_returns_none() {
    let mut entries = HashMap::new();
    entries.insert("name".to_string(), GgufMetadataValue::U32(123));
    let meta = GgufMetadata { entries };
    assert_eq!(meta.get_str("name"), None);
}

#[test]
fn test_metadata_get_u32_wrong_type_returns_none() {
    let mut entries = HashMap::new();
    entries.insert("val".to_string(), GgufMetadataValue::String("hello".into()));
    let meta = GgufMetadata { entries };
    assert_eq!(meta.get_u32("val"), None);
}

#[test]
fn test_metadata_get_missing_key_returns_none() {
    let meta = GgufMetadata {
        entries: HashMap::new(),
    };
    assert_eq!(meta.get_str("anything"), None);
    assert_eq!(meta.get_u32("anything"), None);
    assert_eq!(meta.get_u64("anything"), None);
    assert_eq!(meta.get_f64("anything"), None);
}

// --- ModelArchitecture detection ---------------------------------------------

fn build_metadata(entries: Vec<(&str, GgufMetadataValue)>) -> GgufMetadata {
    let map: HashMap<String, GgufMetadataValue> = entries
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    GgufMetadata { entries: map }
}

#[test]
fn test_model_architecture_gemma() {
    let meta = build_metadata(vec![
        (
            "general.architecture",
            GgufMetadataValue::String("gemma".into()),
        ),
        ("gemma.context_length", GgufMetadataValue::U32(8192)),
        ("gemma.embedding_length", GgufMetadataValue::U32(3072)),
        ("gemma.block_count", GgufMetadataValue::U32(28)),
        ("gemma.attention.head_count", GgufMetadataValue::U32(16)),
        ("gemma.attention.head_count_kv", GgufMetadataValue::U32(1)),
    ]);
    let arch = ModelArchitecture::from_metadata(&meta);
    assert_eq!(arch.architecture, "gemma");
    assert_eq!(arch.context_length, Some(8192));
    assert_eq!(arch.embedding_length, Some(3072));
    assert_eq!(arch.block_count, Some(28));
    assert_eq!(arch.head_count, Some(16));
    assert_eq!(arch.head_count_kv, Some(1));
    assert_eq!(arch.vocab_size, None);
    assert_eq!(arch.rope_freq_base, None);
}

#[test]
fn test_model_architecture_display_includes_architecture_name() {
    let meta = build_metadata(vec![(
        "general.architecture",
        GgufMetadataValue::String("mistral".into()),
    )]);
    let arch = ModelArchitecture::from_metadata(&meta);
    let display = format!("{arch}");
    assert!(display.contains("Architecture: mistral"));
}

#[test]
fn test_model_architecture_display_omits_none_fields() {
    let meta = build_metadata(vec![
        (
            "general.architecture",
            GgufMetadataValue::String("tiny".into()),
        ),
        ("tiny.block_count", GgufMetadataValue::U32(6)),
    ]);
    let arch = ModelArchitecture::from_metadata(&meta);
    let display = format!("{arch}");
    assert!(display.contains("Block count:       6"));
    assert!(!display.contains("Context length"));
    assert!(!display.contains("Embedding dim"));
}

#[test]
fn test_model_architecture_u64_large_context() {
    let meta = build_metadata(vec![
        (
            "general.architecture",
            GgufMetadataValue::String("llama".into()),
        ),
        ("llama.context_length", GgufMetadataValue::U64(1_048_576)),
    ]);
    let arch = ModelArchitecture::from_metadata(&meta);
    assert_eq!(arch.context_length, Some(1_048_576));
}

// --- GgufFile stream integration ---------------------------------------------

/// Build a minimal valid GGUF file with 0 tensors and 0 metadata.
fn minimal_gguf_bytes() -> Vec<u8> {
    let mut data = build_header(0, 0);
    // Pad to 32-byte alignment.
    while !data.len().is_multiple_of(32) {
        data.push(0);
    }
    data
}

/// Build a GGUF file with one Q4_0 tensor (32 elements).
fn gguf_with_q4_0_tensor() -> Vec<u8> {
    let mut data = Vec::new();

    // Header: 1 tensor, 0 metadata.
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count

    // Tensor info: "weight" [32] Q4_0, offset=0.
    let name = b"weight";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name);
    data.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
    data.extend_from_slice(&32u64.to_le_bytes()); // dim 0
    data.extend_from_slice(&(GgufDType::Q4_0 as u32).to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // offset = 0

    // Pad to 32-byte alignment.
    while data.len() % 32 != 0 {
        data.push(0);
    }

    // Q4_0 tensor data: 1 block = 18 bytes.
    // Scale = 1.0 as f16, all nibbles = 0x88 (lo=8, hi=8 => q-8=0).
    let scale_f16 = half::f16::from_f32(1.0);
    data.extend_from_slice(&scale_f16.to_le_bytes());
    data.extend_from_slice(&[0x88u8; 16]); // All nibbles = 8 => val = scale*(8-8) = 0

    data
}

#[test]
fn test_gguf_file_read_minimal() {
    let data = minimal_gguf_bytes();
    let mut cursor = Cursor::new(data);
    let file = GgufFile::read_from(&mut cursor).unwrap();
    assert_eq!(file.header.version, 3);
    assert_eq!(file.header.tensor_count, 0);
    assert!(file.tensors.is_empty());
    assert!(file.architecture().is_none());
    assert!(file.tensor_names().is_empty());
}

#[test]
fn test_gguf_file_read_q4_0_tensor() {
    let data = gguf_with_q4_0_tensor();
    let mut cursor = Cursor::new(data);
    let file = GgufFile::read_from(&mut cursor).unwrap();

    assert_eq!(file.header.tensor_count, 1);
    let names = file.tensor_names();
    assert_eq!(names.len(), 1);
    assert!(names.contains(&"weight"));

    let info = file.tensors.get("weight").unwrap();
    assert_eq!(info.dtype, GgufDType::Q4_0);
    assert_eq!(info.shape, vec![32]);
    assert_eq!(info.num_elements(), 32);
    assert_eq!(info.byte_size(), 18); // 1 block * 18 bytes
}

#[test]
fn test_gguf_file_read_tensor_f32_q4_0() {
    let data = gguf_with_q4_0_tensor();
    let mut cursor = Cursor::new(data);
    let file = GgufFile::read_from(&mut cursor).unwrap();

    let (values, shape) = file.read_tensor_f32(&mut cursor, "weight").unwrap();
    assert_eq!(shape, vec![32]);
    assert_eq!(values.len(), 32);
    // All nibbles were 0x88 => q=8, val = 1.0*(8-8) = 0.0
    for (i, &v) in values.iter().enumerate() {
        assert!(v.abs() < 0.01, "index {i}: expected 0.0, got {v}");
    }
}

#[test]
fn test_gguf_file_missing_tensor() {
    let data = minimal_gguf_bytes();
    let mut cursor = Cursor::new(data);
    let file = GgufFile::read_from(&mut cursor).unwrap();

    let result = file.read_tensor_f32(&mut cursor, "nonexistent");
    assert!(result.is_err());
}

// --- Byte alignment validation -----------------------------------------------

#[test]
fn test_data_offset_is_32_byte_aligned() {
    let data = gguf_with_q4_0_tensor();
    let mut cursor = Cursor::new(data);
    let file = GgufFile::read_from(&mut cursor).unwrap();
    assert_eq!(
        file.data_offset % 32,
        0,
        "data_offset should be 32-byte aligned"
    );
}

#[test]
fn test_data_offset_is_32_byte_aligned_minimal() {
    let data = minimal_gguf_bytes();
    let mut cursor = Cursor::new(data);
    let file = GgufFile::read_from(&mut cursor).unwrap();
    assert_eq!(
        file.data_offset % 32,
        0,
        "data_offset should be 32-byte aligned"
    );
}

// --- Cross-validation: nn-gguf vs nn-core dequant --------------------------

#[test]
fn test_q4_0_cross_crate_dequant_match() {
    use nn_core::dyn_tensor::quantized::QuantType;
    use nn_core::DynTensor;

    // Build one Q4_0 block with scale=2.0 and known nibbles.
    let scale = half::f16::from_f32(2.0);
    let mut block = Vec::new();
    block.extend_from_slice(&scale.to_le_bytes());
    // Nibble pattern: low=3, high=12 for each byte.
    block.extend_from_slice(&[0xC3u8; 16]);

    // nn-gguf dequantize.
    let gguf_result = dequantize_q4_0(&block, 32);

    // nn-core DynTensor dequantize.
    let tensor = DynTensor::from_quantized(&block, QuantType::Q4_0, &[32]).unwrap();
    assert!(tensor.is_quantized());
    let deq = tensor.dequantize().unwrap();
    let core_result = deq.to_flat_vec::<f32>().unwrap();

    assert_eq!(gguf_result.len(), core_result.len());
    for i in 0..32 {
        assert!(
            (gguf_result[i] - core_result[i]).abs() < 1e-5,
            "index {i}: gguf={} core={}",
            gguf_result[i],
            core_result[i]
        );
    }
}

#[test]
fn test_q4_1_cross_crate_dequant_match() {
    use nn_core::dyn_tensor::quantized::QuantType;
    use nn_core::DynTensor;

    // Build one Q4_1 block with d=0.5, m=1.0.
    let d = half::f16::from_f32(0.5);
    let m = half::f16::from_f32(1.0);
    let mut block = Vec::new();
    block.extend_from_slice(&d.to_le_bytes());
    block.extend_from_slice(&m.to_le_bytes());
    block.extend_from_slice(&[0xA5u8; 16]); // lo=5, hi=10

    let gguf_result = dequantize_q4_1(&block, 32);

    let tensor = DynTensor::from_quantized(&block, QuantType::Q4_1, &[32]).unwrap();
    let deq = tensor.dequantize().unwrap();
    let core_result = deq.to_flat_vec::<f32>().unwrap();

    assert_eq!(gguf_result.len(), core_result.len());
    for i in 0..32 {
        assert!(
            (gguf_result[i] - core_result[i]).abs() < 1e-5,
            "index {i}: gguf={} core={}",
            gguf_result[i],
            core_result[i]
        );
    }
}

#[test]
fn test_q8_0_cross_crate_dequant_match() {
    use nn_core::dyn_tensor::quantized::QuantType;
    use nn_core::DynTensor;

    // Build one Q8_0 block with scale=0.25, values = [-10..22].
    let scale = half::f16::from_f32(0.25);
    let mut block = Vec::new();
    block.extend_from_slice(&scale.to_le_bytes());
    for i in 0..32i8 {
        block.push((i - 10) as u8);
    }

    let gguf_result = dequantize_q8_0(&block, 32);

    let tensor = DynTensor::from_quantized(&block, QuantType::Q8_0, &[32]).unwrap();
    let deq = tensor.dequantize().unwrap();
    let core_result = deq.to_flat_vec::<f32>().unwrap();

    assert_eq!(gguf_result.len(), core_result.len());
    for i in 0..32 {
        assert!(
            (gguf_result[i] - core_result[i]).abs() < 1e-5,
            "index {i}: gguf={} core={}",
            gguf_result[i],
            core_result[i]
        );
    }
}

// --- DynTensor quantized storage edge cases ----------------------------------

#[test]
fn test_dyn_tensor_quantized_wrong_byte_length() {
    use nn_core::dyn_tensor::quantized::QuantType;
    use nn_core::DynTensor;

    // Q4_0 expects 18 bytes for 32 elements, give 17.
    let result = DynTensor::from_quantized(&[0u8; 17], QuantType::Q4_0, &[32]);
    assert!(result.is_err());
}

#[test]
fn test_dyn_tensor_quantized_non_block_aligned_elements() {
    use nn_core::dyn_tensor::quantized::QuantType;
    use nn_core::DynTensor;

    // 33 elements is not a multiple of block size 32.
    let result = DynTensor::from_quantized(&[0u8; 100], QuantType::Q4_0, &[33]);
    assert!(result.is_err());
}

#[test]
fn test_dyn_tensor_quantized_storage_accessors() {
    use nn_core::dyn_tensor::quantized::QuantType;
    use nn_core::DynTensor;

    let block = vec![0u8; 18]; // One Q4_0 block.
    let tensor = DynTensor::from_quantized(&block, QuantType::Q4_0, &[32]).unwrap();

    assert!(tensor.is_quantized());
    let qs = tensor.quantized_storage().unwrap();
    assert_eq!(qs.shape(), &[32]);
    assert_eq!(qs.quant_type(), QuantType::Q4_0);
    assert_eq!(qs.raw_data().len(), 18);
}

#[test]
fn test_dyn_tensor_dequantize_non_quantized_is_clone() {
    use nn_core::{DType, Device, DynTensor};

    let tensor = DynTensor::zeros(&[4, 4], DType::F32, &Device::Cpu).unwrap();
    assert!(!tensor.is_quantized());

    let deq = tensor.dequantize().unwrap();
    assert!(!deq.is_quantized());
    let vals = deq.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 16);
    for &v in &vals {
        assert_eq!(v, 0.0);
    }
}

// --- QuantType metadata ------------------------------------------------------

#[test]
fn test_quant_type_expected_bytes() {
    use nn_core::dyn_tensor::quantized::QuantType;

    assert_eq!(QuantType::Q4_0.expected_bytes(32), Some(18));
    assert_eq!(QuantType::Q4_0.expected_bytes(64), Some(36));
    assert_eq!(QuantType::Q4_0.expected_bytes(0), Some(0));
    // Not a multiple of 32:
    assert_eq!(QuantType::Q4_0.expected_bytes(31), None);
    assert_eq!(QuantType::Q4_0.expected_bytes(1), None);

    assert_eq!(QuantType::Q4_1.expected_bytes(32), Some(20));
    assert_eq!(QuantType::Q8_0.expected_bytes(32), Some(34));
    assert_eq!(QuantType::Q8_0.expected_bytes(64), Some(68));
}

#[test]
fn test_quant_type_display() {
    use nn_core::dyn_tensor::quantized::QuantType;

    assert_eq!(format!("{}", QuantType::Q4_0), "Q4_0");
    assert_eq!(format!("{}", QuantType::Q4_1), "Q4_1");
    assert_eq!(format!("{}", QuantType::Q8_0), "Q8_0");
}

#[test]
fn test_quant_type_block_size_all_same() {
    use nn_core::dyn_tensor::quantized::QuantType;

    // All currently supported QuantTypes have block size 32.
    assert_eq!(QuantType::Q4_0.block_size(), 32);
    assert_eq!(QuantType::Q4_1.block_size(), 32);
    assert_eq!(QuantType::Q8_0.block_size(), 32);
}

#[test]
fn test_quant_type_block_bytes() {
    use nn_core::dyn_tensor::quantized::QuantType;

    assert_eq!(QuantType::Q4_0.block_bytes(), 18);
    assert_eq!(QuantType::Q4_1.block_bytes(), 20);
    assert_eq!(QuantType::Q8_0.block_bytes(), 34);
}

// --- Multi-block dequantization consistency -----------------------------------

#[test]
fn test_q4_0_multi_block_length_matches() {
    // Build 4 blocks = 128 elements.
    let scale = half::f16::from_f32(1.0);
    let mut data = Vec::new();
    for _ in 0..4 {
        data.extend_from_slice(&scale.to_le_bytes());
        data.extend_from_slice(&[0x88u8; 16]);
    }
    let result = dequantize_q4_0(&data, 128);
    assert_eq!(result.len(), 128);
}

#[test]
fn test_q8_0_multi_block_different_scales() {
    // 2 blocks with different scales.
    let scale1 = half::f16::from_f32(1.0);
    let scale2 = half::f16::from_f32(2.0);

    let mut data = Vec::new();
    // Block 1: scale=1.0, all q=1 => val=1.0
    data.extend_from_slice(&scale1.to_le_bytes());
    data.extend_from_slice(&[1u8; 32]);
    // Block 2: scale=2.0, all q=1 => val=2.0
    data.extend_from_slice(&scale2.to_le_bytes());
    data.extend_from_slice(&[1u8; 32]);

    let result = dequantize_q8_0(&data, 64);
    assert_eq!(result.len(), 64);

    // First 32 should be ~1.0.
    for &v in &result[..32] {
        assert!((v - 1.0).abs() < 0.01, "block1: expected 1.0, got {v}");
    }
    // Next 32 should be ~2.0.
    for &v in &result[32..] {
        assert!((v - 2.0).abs() < 0.01, "block2: expected 2.0, got {v}");
    }
}

// --- GGUF file with metadata parsing -----------------------------------------

/// Build a GGUF file with multiple metadata entries and no tensors.
fn gguf_with_metadata() -> Vec<u8> {
    let mut data = Vec::new();

    // Header: 0 tensors, 3 metadata.
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&3u64.to_le_bytes()); // metadata_kv_count

    // Metadata 1: "general.architecture" = "llama" (STRING, type=8)
    let key1 = b"general.architecture";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1);
    data.extend_from_slice(&8u32.to_le_bytes()); // STRING type
    let val1 = b"llama";
    data.extend_from_slice(&(val1.len() as u64).to_le_bytes());
    data.extend_from_slice(val1);

    // Metadata 2: "llama.context_length" = 4096 (UINT32, type=4)
    let key2 = b"llama.context_length";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2);
    data.extend_from_slice(&4u32.to_le_bytes()); // UINT32 type
    data.extend_from_slice(&4096u32.to_le_bytes());

    // Metadata 3: "general.quantization_version" = 2 (UINT32, type=4)
    let key3 = b"general.quantization_version";
    data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
    data.extend_from_slice(key3);
    data.extend_from_slice(&4u32.to_le_bytes()); // UINT32 type
    data.extend_from_slice(&2u32.to_le_bytes());

    // Pad to 32-byte alignment.
    while data.len() % 32 != 0 {
        data.push(0);
    }

    data
}

#[test]
fn test_gguf_file_metadata_parsing() {
    let data = gguf_with_metadata();
    let mut cursor = Cursor::new(data);
    let file = GgufFile::read_from(&mut cursor).unwrap();

    assert_eq!(file.architecture(), Some("llama"));
    assert_eq!(file.metadata.get_str("general.architecture"), Some("llama"));
    assert_eq!(file.metadata.get_u32("llama.context_length"), Some(4096));
    assert_eq!(
        file.metadata.get_u32("general.quantization_version"),
        Some(2)
    );
}

#[test]
fn test_gguf_file_model_architecture_from_file() {
    let data = gguf_with_metadata();
    let mut cursor = Cursor::new(data);
    let file = GgufFile::read_from(&mut cursor).unwrap();

    let arch = ModelArchitecture::from_metadata(&file.metadata);
    assert_eq!(arch.architecture, "llama");
    assert_eq!(arch.context_length, Some(4096));
    assert_eq!(arch.quantization_version, Some(2));
}

// --- Safetensors detection smoke tests (nn-import API) ----------------------

#[test]
fn test_detect_quantization_returns_report() {
    use crate::quantization::detect_quantization_from_bytes;

    // Build a small safetensors file with one F32 tensor.
    let w_data = vec![0u8; 16]; // 4 elements * 4 bytes
    let tensors: Vec<(String, safetensors::tensor::TensorView<'_>)> = vec![(
        "w".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![4], &w_data).unwrap(),
    )];
    let bytes = safetensors::tensor::serialize(tensors, None).unwrap();

    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert_eq!(report.total_tensors, 1);
    assert_eq!(report.total_parameters, 4);
}
