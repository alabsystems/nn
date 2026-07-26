// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended security hardening tests for GGUF parser.
#![allow(deprecated)]
//!
//! These tests supplement `security_tests.rs` with deeper edge-case coverage:
//! - Boundary-value analysis for every security limit
//! - checked_byte_size vs byte_size divergence on crafted shapes
//! - Multi-dimensional overflow combinations
//! - Quantized-type-specific overflow paths
//! - Metadata value type bomb patterns
//! - Truncation attacks at every parse stage

use crate::dequant::GgufDType;
use crate::error::GgufError;
use crate::header::{
    GgufHeader, GGUF_MAGIC, MAX_ARRAY_LENGTH, MAX_DIMENSIONS, MAX_METADATA_KV_COUNT,
    MAX_STRING_LENGTH, MAX_TENSOR_BYTE_SIZE, MAX_TENSOR_COUNT,
};
use crate::reader::GgufFile;
use crate::tensor_info::GgufTensorInfo;

// ---------------------------------------------------------------
// Helper: build a crafted GGUF header as raw bytes
// ---------------------------------------------------------------

fn craft_header(magic: u32, version: u32, tensor_count: u64, metadata_kv_count: u64) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&magic.to_le_bytes());
    data.extend_from_slice(&version.to_le_bytes());
    data.extend_from_slice(&tensor_count.to_le_bytes());
    data.extend_from_slice(&metadata_kv_count.to_le_bytes());
    data
}

fn valid_header(tensor_count: u64, metadata_kv_count: u64) -> Vec<u8> {
    craft_header(GGUF_MAGIC, 3, tensor_count, metadata_kv_count)
}

/// Helper to build a GgufTensorInfo directly (bypassing parse).
fn make_tensor(name: &str, shape: &[u64], dtype: GgufDType) -> GgufTensorInfo {
    GgufTensorInfo {
        name: name.to_string(),
        n_dims: shape.len() as u32,
        shape: shape.to_vec(),
        dtype,
        offset: 0,
    }
}

/// Append a metadata array KV with given element type and count.
fn append_array_kv(buf: &mut Vec<u8>, key: &str, elem_type: u32, count: u64) {
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    buf.extend_from_slice(&9u32.to_le_bytes()); // ARRAY type
    buf.extend_from_slice(&elem_type.to_le_bytes());
    buf.extend_from_slice(&count.to_le_bytes());
}

// ===================================================================
// 1. Header overflow tests: boundary values around MAX_TENSOR_COUNT
// ===================================================================

#[test]
fn test_header_tensor_count_at_exact_limit_accepted() {
    let data = valid_header(MAX_TENSOR_COUNT, 0);
    let header = GgufHeader::read_from(&mut &data[..]).unwrap();
    assert_eq!(header.tensor_count, MAX_TENSOR_COUNT);
}

#[test]
fn test_header_tensor_count_one_over_limit_rejected() {
    let data = valid_header(MAX_TENSOR_COUNT + 1, 0);
    let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
    assert!(
        matches!(err, GgufError::TensorCountExceeded { count, max } if count == MAX_TENSOR_COUNT + 1 && max == MAX_TENSOR_COUNT),
        "expected TensorCountExceeded at boundary+1, got: {err}"
    );
}

#[test]
fn test_header_tensor_count_u64_max_rejected() {
    let data = valid_header(u64::MAX, 0);
    let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
    assert!(matches!(err, GgufError::TensorCountExceeded { .. }));
}

#[test]
fn test_header_tensor_count_u64_max_minus_one_rejected() {
    let data = valid_header(u64::MAX - 1, 0);
    let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
    assert!(matches!(err, GgufError::TensorCountExceeded { .. }));
}

// ===================================================================
// 2. Metadata count overflow: boundary values
// ===================================================================

#[test]
fn test_header_metadata_count_at_exact_limit_accepted() {
    let data = valid_header(0, MAX_METADATA_KV_COUNT);
    let header = GgufHeader::read_from(&mut &data[..]).unwrap();
    assert_eq!(header.metadata_kv_count, MAX_METADATA_KV_COUNT);
}

#[test]
fn test_header_metadata_count_one_over_limit_rejected() {
    let data = valid_header(0, MAX_METADATA_KV_COUNT + 1);
    let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
    assert!(
        matches!(err, GgufError::MetadataCountExceeded { count, max } if count == MAX_METADATA_KV_COUNT + 1 && max == MAX_METADATA_KV_COUNT),
        "expected MetadataCountExceeded at boundary+1, got: {err}"
    );
}

#[test]
fn test_header_both_counts_at_limit_accepted() {
    let data = valid_header(MAX_TENSOR_COUNT, MAX_METADATA_KV_COUNT);
    let header = GgufHeader::read_from(&mut &data[..]).unwrap();
    assert_eq!(header.tensor_count, MAX_TENSOR_COUNT);
    assert_eq!(header.metadata_kv_count, MAX_METADATA_KV_COUNT);
}

#[test]
fn test_header_both_counts_over_limit_tensor_checked_first() {
    // When both counts exceed limits, tensor_count is checked first in
    // GgufHeader::read_from.
    let data = valid_header(MAX_TENSOR_COUNT + 1, MAX_METADATA_KV_COUNT + 1);
    let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
    assert!(
        matches!(err, GgufError::TensorCountExceeded { .. }),
        "tensor count should be validated before metadata count"
    );
}

// ===================================================================
// 3. String length bomb: boundary values around MAX_STRING_LENGTH
// ===================================================================

#[test]
fn test_string_length_at_exact_limit_accepted_if_data_available() {
    // String at exactly MAX_STRING_LENGTH would need 16 MiB of data.
    // We verify the length check passes by constructing the read_string
    // path: the string length is accepted but the read_exact will fail
    // due to insufficient data (EOF), which is fine -- the length check
    // itself passed.
    let mut data = valid_header(0, 1);
    data.extend_from_slice(&(MAX_STRING_LENGTH).to_le_bytes());
    // Don't provide the actual bytes -- we just test the length is accepted.
    let err = GgufFile::read_from(&mut std::io::Cursor::new(data)).unwrap_err();
    // Should be an IO error (UnexpectedEof), NOT StringLengthExceeded.
    assert!(
        matches!(err, GgufError::Io(_)),
        "MAX_STRING_LENGTH should be accepted; got: {err}"
    );
}

#[test]
fn test_string_length_one_over_limit_rejected() {
    let mut data = valid_header(0, 1);
    data.extend_from_slice(&(MAX_STRING_LENGTH + 1).to_le_bytes());
    let err = GgufFile::read_from(&mut std::io::Cursor::new(data)).unwrap_err();
    assert!(
        matches!(err, GgufError::StringLengthExceeded { len, max } if len == MAX_STRING_LENGTH + 1 && max == MAX_STRING_LENGTH),
        "expected StringLengthExceeded at boundary+1, got: {err}"
    );
}

#[test]
fn test_string_length_in_metadata_value_position_bomb() {
    // String bomb in the value position (not the key).
    let mut data = valid_header(0, 1);
    let key = b"ok_key";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    data.extend_from_slice(&8u32.to_le_bytes()); // STRING type
    data.extend_from_slice(&(MAX_STRING_LENGTH + 1).to_le_bytes()); // bomb in value
    let err = GgufFile::read_from(&mut std::io::Cursor::new(data)).unwrap_err();
    assert!(matches!(err, GgufError::StringLengthExceeded { .. }));
}

#[test]
fn test_string_length_in_tensor_name_bomb() {
    let mut data = valid_header(1, 0);
    // Tensor name length = MAX + 1
    data.extend_from_slice(&(MAX_STRING_LENGTH + 1).to_le_bytes());
    let err = GgufFile::read_from(&mut std::io::Cursor::new(data)).unwrap_err();
    assert!(matches!(err, GgufError::StringLengthExceeded { .. }));
}

// ===================================================================
// 4. Array length bomb: boundary values around MAX_ARRAY_LENGTH
// ===================================================================

#[test]
fn test_array_length_at_exact_limit_accepted_if_data_available() {
    let mut data = valid_header(0, 1);
    append_array_kv(&mut data, "arr", 0, MAX_ARRAY_LENGTH); // U8 elements
                                                            // Don't provide actual array data; should fail on IO, not on length check.
    let err = GgufFile::read_from(&mut std::io::Cursor::new(data)).unwrap_err();
    assert!(
        matches!(err, GgufError::Io(_)),
        "MAX_ARRAY_LENGTH should be accepted; got: {err}"
    );
}

#[test]
fn test_array_length_one_over_limit_rejected() {
    let mut data = valid_header(0, 1);
    append_array_kv(&mut data, "arr", 0, MAX_ARRAY_LENGTH + 1);
    let err = GgufFile::read_from(&mut std::io::Cursor::new(data)).unwrap_err();
    assert!(
        matches!(err, GgufError::ArrayLengthExceeded { count, max } if count == MAX_ARRAY_LENGTH + 1 && max == MAX_ARRAY_LENGTH),
        "expected ArrayLengthExceeded at boundary+1, got: {err}"
    );
}

#[test]
fn test_array_length_u64_max_rejected() {
    let mut data = valid_header(0, 1);
    append_array_kv(&mut data, "arr", 0, u64::MAX);
    let err = GgufFile::read_from(&mut std::io::Cursor::new(data)).unwrap_err();
    assert!(matches!(err, GgufError::ArrayLengthExceeded { .. }));
}

// ===================================================================
// 5. Tensor dimension bomb: n_dims > MAX_DIMENSIONS
// ===================================================================

#[test]
fn test_dimension_count_at_exact_limit_accepted() {
    let mut data = valid_header(1, 0);
    let name = b"t";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name);
    data.extend_from_slice(&(MAX_DIMENSIONS).to_le_bytes()); // n_dims = MAX
    for _ in 0..MAX_DIMENSIONS {
        data.extend_from_slice(&1u64.to_le_bytes()); // each dim = 1
    }
    data.extend_from_slice(&(GgufDType::F32 as u32).to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // offset
                                                 // Pad to alignment
    while !data.len().is_multiple_of(32) {
        data.push(0);
    }
    data.extend_from_slice(&[0u8; 4]); // 1 f32

    let mut cursor = std::io::Cursor::new(&data);
    let file = GgufFile::read_from(&mut cursor).unwrap();
    assert_eq!(file.tensors.len(), 1);
}

#[test]
fn test_dimension_count_one_over_limit_rejected() {
    let mut data = valid_header(1, 0);
    let name = b"t";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name);
    data.extend_from_slice(&(MAX_DIMENSIONS + 1).to_le_bytes()); // n_dims = MAX + 1
    let err = GgufFile::read_from(&mut std::io::Cursor::new(data)).unwrap_err();
    assert!(
        matches!(err, GgufError::DimensionCountExceeded { n_dims, max } if n_dims == MAX_DIMENSIONS + 1 && max == MAX_DIMENSIONS),
        "expected DimensionCountExceeded, got: {err}"
    );
}

// ===================================================================
// 6. Zero-dimension tensor edge cases
// ===================================================================

#[test]
fn test_zero_dimension_first_of_three_rejected() {
    let mut data = valid_header(1, 0);
    let name = b"zd3";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name);
    data.extend_from_slice(&3u32.to_le_bytes()); // n_dims = 3
    data.extend_from_slice(&0u64.to_le_bytes()); // dim 0 = 0 INVALID
    data.extend_from_slice(&4u64.to_le_bytes());
    data.extend_from_slice(&8u64.to_le_bytes());
    let err = GgufFile::read_from(&mut std::io::Cursor::new(data)).unwrap_err();
    assert!(matches!(err, GgufError::ZeroDimension { ref name, dim_index: 0 } if name == "zd3"));
}

#[test]
fn test_zero_dimension_middle_of_three_rejected() {
    let mut data = valid_header(1, 0);
    let name = b"zdm";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name);
    data.extend_from_slice(&3u32.to_le_bytes()); // n_dims = 3
    data.extend_from_slice(&4u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // dim 1 = 0 INVALID
    data.extend_from_slice(&8u64.to_le_bytes());
    let err = GgufFile::read_from(&mut std::io::Cursor::new(data)).unwrap_err();
    assert!(matches!(err, GgufError::ZeroDimension { dim_index: 1, .. }));
}

#[test]
fn test_zero_dimension_last_of_three_rejected() {
    let mut data = valid_header(1, 0);
    let name = b"zdl";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name);
    data.extend_from_slice(&3u32.to_le_bytes()); // n_dims = 3
    data.extend_from_slice(&4u64.to_le_bytes());
    data.extend_from_slice(&8u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // dim 2 = 0 INVALID
    let err = GgufFile::read_from(&mut std::io::Cursor::new(data)).unwrap_err();
    assert!(matches!(err, GgufError::ZeroDimension { dim_index: 2, .. }));
}

// ===================================================================
// 7. Byte size overflow: multi-dimensional shape products
// ===================================================================

#[test]
fn test_byte_size_overflow_two_large_dims_f32() {
    // Two dimensions whose product overflows u64.
    let info = make_tensor("ov2", &[u64::MAX / 2 + 1, 3], GgufDType::F32);
    let err = info.checked_byte_size().unwrap_err();
    assert!(matches!(
        err,
        GgufError::ElementCountOverflow { .. } | GgufError::ByteSizeOverflow { .. }
    ));
}

#[test]
fn test_byte_size_overflow_three_moderate_dims() {
    // Three moderately large dims whose product overflows: ~2^21 * 2^21 * 2^22
    let info = make_tensor("ov3", &[2_097_152, 2_097_152, 4_194_304], GgufDType::F32);
    let err = info.checked_byte_size().unwrap_err();
    assert!(matches!(
        err,
        GgufError::ElementCountOverflow { .. } | GgufError::ByteSizeOverflow { .. }
    ));
}

#[test]
fn test_byte_size_overflow_type_size_multiplication() {
    // Element count fits in u64, but element_count * type_size overflows.
    // For F32 (type_size=4), we need elements > u64::MAX / 4.
    let elements = u64::MAX / 4 + 1; // 4611686018427387905
    let info = make_tensor("ovts", &[elements], GgufDType::F32);
    let err = info.checked_byte_size().unwrap_err();
    assert!(
        matches!(
            err,
            GgufError::ByteSizeOverflow { .. } | GgufError::TensorTooLarge { .. }
        ),
        "expected overflow or too-large, got: {err}"
    );
}

#[test]
fn test_byte_size_overflow_quantized_q4k() {
    // Q4K: block_size=256, type_size=144. Craft shape where
    // (elements / 256) * 144 overflows.
    let blocks = u64::MAX / 144 + 1;
    let elements = blocks.saturating_mul(256);
    let info = make_tensor("ovq4k", &[elements], GgufDType::Q4K);
    let err = info.checked_byte_size().unwrap_err();
    assert!(matches!(
        err,
        GgufError::ElementCountOverflow { .. }
            | GgufError::ByteSizeOverflow { .. }
            | GgufError::TensorTooLarge { .. }
    ));
}

// ===================================================================
// 8. Version mismatch: various invalid versions
// ===================================================================

#[test]
fn test_version_0_rejected() {
    let data = craft_header(GGUF_MAGIC, 0, 0, 0);
    let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
    assert!(matches!(err, GgufError::UnsupportedVersion { version: 0 }));
}

#[test]
fn test_version_1_rejected() {
    let data = craft_header(GGUF_MAGIC, 1, 0, 0);
    let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
    assert!(matches!(err, GgufError::UnsupportedVersion { version: 1 }));
}

#[test]
fn test_version_2_rejected() {
    let data = craft_header(GGUF_MAGIC, 2, 0, 0);
    let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
    assert!(matches!(err, GgufError::UnsupportedVersion { version: 2 }));
}

#[test]
fn test_version_3_accepted() {
    let data = craft_header(GGUF_MAGIC, 3, 0, 0);
    let header = GgufHeader::read_from(&mut &data[..]).unwrap();
    assert_eq!(header.version, 3);
}

#[test]
fn test_version_4_rejected() {
    let data = craft_header(GGUF_MAGIC, 4, 0, 0);
    let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
    assert!(matches!(err, GgufError::UnsupportedVersion { version: 4 }));
}

#[test]
fn test_version_u32_max_rejected() {
    let data = craft_header(GGUF_MAGIC, u32::MAX, 0, 0);
    let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
    assert!(matches!(
        err,
        GgufError::UnsupportedVersion { version } if version == u32::MAX
    ));
}

// ===================================================================
// 9. Magic number mismatch: various wrong magic values
// ===================================================================

#[test]
fn test_magic_all_zeros_rejected() {
    let data = craft_header(0x0000_0000, 3, 0, 0);
    let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
    assert!(matches!(err, GgufError::InvalidMagic { found: 0 }));
}

#[test]
fn test_magic_ggml_instead_of_gguf_rejected() {
    // "GGML" in LE = 0x4C4D4747
    let data = craft_header(0x4C4D_4747, 3, 0, 0);
    let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
    assert!(matches!(err, GgufError::InvalidMagic { .. }));
}

#[test]
fn test_magic_one_bit_flipped_rejected() {
    // Flip one bit in the valid magic (0x46475547).
    let flipped = GGUF_MAGIC ^ 1;
    let data = craft_header(flipped, 3, 0, 0);
    let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
    assert!(matches!(err, GgufError::InvalidMagic { .. }));
}

#[test]
fn test_magic_big_endian_swapped_rejected() {
    // "GGUF" in big-endian byte order.
    let be_magic = GGUF_MAGIC.swap_bytes();
    assert_ne!(be_magic, GGUF_MAGIC, "magic should differ in BE");
    let data = craft_header(be_magic, 3, 0, 0);
    let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
    assert!(matches!(err, GgufError::InvalidMagic { .. }));
}

#[test]
fn test_magic_u32_max_rejected() {
    let data = craft_header(u32::MAX, 3, 0, 0);
    let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
    assert!(matches!(err, GgufError::InvalidMagic { .. }));
}

// ===================================================================
// 10. Nested array depth: recursive array declarations
// ===================================================================

#[test]
fn test_deeply_nested_empty_arrays_no_crash() {
    // Array of arrays of arrays ... with 0 elements at the leaves.
    // This tests that the parser does not stack-overflow on deeply nested
    // (but ultimately empty) arrays.
    let mut data = valid_header(0, 1);
    let key = b"deep";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    data.extend_from_slice(&9u32.to_le_bytes()); // ARRAY type
    data.extend_from_slice(&9u32.to_le_bytes()); // element type = ARRAY
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 sub-array

    // Level 2: another array of arrays
    data.extend_from_slice(&9u32.to_le_bytes()); // element type = ARRAY
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 sub-sub-array

    // Level 3: array of U8 with 0 elements
    data.extend_from_slice(&0u32.to_le_bytes()); // U8
    data.extend_from_slice(&0u64.to_le_bytes()); // 0 elements

    let result = GgufFile::read_from(&mut std::io::Cursor::new(data));
    assert!(result.is_ok(), "deeply nested empty arrays should parse OK");
}

#[test]
fn test_array_of_strings_with_bomb_length() {
    // Array of strings where the first string has a bomb length.
    let mut data = valid_header(0, 1);
    let key = b"strarr";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    data.extend_from_slice(&9u32.to_le_bytes()); // ARRAY type
    data.extend_from_slice(&8u32.to_le_bytes()); // element type = STRING
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 element
                                                 // String element with bomb length.
    data.extend_from_slice(&(MAX_STRING_LENGTH + 1).to_le_bytes());
    let err = GgufFile::read_from(&mut std::io::Cursor::new(data)).unwrap_err();
    assert!(matches!(err, GgufError::StringLengthExceeded { .. }));
}

// ===================================================================
// 11. checked_byte_size vs byte_size: divergence on edge cases
// ===================================================================

#[test]
fn test_checked_vs_unchecked_normal_tensor_agree() {
    let info = make_tensor("norm", &[128, 256], GgufDType::F32);
    let checked = info.checked_byte_size().unwrap();
    let unchecked = info.byte_size();
    assert_eq!(checked, unchecked);
}

#[test]
fn test_checked_vs_unchecked_quantized_tensor_agree() {
    // Q4_0: block_size=32, type_size=18.
    // 1024 elements / 32 = 32 blocks * 18 = 576 bytes.
    let info = make_tensor("q40", &[1024], GgufDType::Q4_0);
    let checked = info.checked_byte_size().unwrap();
    let unchecked = info.byte_size();
    assert_eq!(checked, unchecked);
    assert_eq!(checked, 576);
}

#[test]
fn test_checked_byte_size_catches_overflow_that_unchecked_cannot() {
    // Construct a shape where the byte size overflows u64.
    // checked_byte_size returns a proper error; the unchecked byte_size
    // would panic in debug mode or wrap in release mode.
    let info = make_tensor("wrap", &[u64::MAX / 4 + 1], GgufDType::F32);

    // checked_byte_size must return an error.
    let result = info.checked_byte_size();
    assert!(result.is_err(), "checked_byte_size must catch the overflow");

    // Verify the overflow: the true byte count cannot be represented in u64.
    let expected_elements = u64::MAX / 4 + 1;
    assert!(
        expected_elements.checked_mul(4).is_none(),
        "the multiplication should overflow u64, proving checked_byte_size is necessary"
    );
}

#[test]
fn test_checked_byte_size_catches_element_overflow() {
    let info = make_tensor("elov", &[u64::MAX, 2], GgufDType::F32);
    assert!(info.checked_byte_size().is_err());
    assert!(info.checked_num_elements().is_err());
}

#[test]
fn test_checked_num_elements_single_element_ok() {
    let info = make_tensor("one", &[1], GgufDType::F32);
    assert_eq!(info.checked_num_elements().unwrap(), 1);
}

#[test]
fn test_checked_num_elements_empty_shape_returns_one() {
    // Zero dimensions = scalar. checked_num_elements should return >= 1.
    let info = GgufTensorInfo {
        name: "scalar".into(),
        n_dims: 0,
        shape: vec![],
        dtype: GgufDType::F32,
        offset: 0,
    };
    assert_eq!(info.checked_num_elements().unwrap(), 1);
}

// ===================================================================
// 12. MAX_TENSOR_BYTE_SIZE enforcement: 8 GiB cap
// ===================================================================

#[test]
fn test_tensor_byte_size_exactly_8gib_accepted() {
    // 8 GiB in bytes = 8 * 1024^3 = 8589934592. For F32 (4 bytes):
    // elements = 8589934592 / 4 = 2147483648
    let elements = MAX_TENSOR_BYTE_SIZE / 4;
    let info = make_tensor("at_cap", &[elements], GgufDType::F32);
    let byte_size = info.checked_byte_size().unwrap();
    assert_eq!(byte_size, MAX_TENSOR_BYTE_SIZE);
}

#[test]
fn test_tensor_byte_size_one_element_over_8gib_rejected() {
    let elements = MAX_TENSOR_BYTE_SIZE / 4 + 1;
    let info = make_tensor("over_cap", &[elements], GgufDType::F32);
    let err = info.checked_byte_size().unwrap_err();
    assert!(
        matches!(err, GgufError::TensorTooLarge { byte_size, max, .. } if byte_size == elements * 4 && max == MAX_TENSOR_BYTE_SIZE),
        "expected TensorTooLarge, got: {err}"
    );
}

#[test]
fn test_tensor_byte_size_cap_with_f16() {
    // F16 (2 bytes): elements = 8 GiB / 2 = 4294967296
    let elements = MAX_TENSOR_BYTE_SIZE / 2;
    let info = make_tensor("f16_at_cap", &[elements], GgufDType::F16);
    let byte_size = info.checked_byte_size().unwrap();
    assert_eq!(byte_size, MAX_TENSOR_BYTE_SIZE);
}

#[test]
fn test_tensor_byte_size_cap_with_f16_over() {
    let elements = MAX_TENSOR_BYTE_SIZE / 2 + 1;
    let info = make_tensor("f16_over", &[elements], GgufDType::F16);
    let err = info.checked_byte_size().unwrap_err();
    assert!(matches!(err, GgufError::TensorTooLarge { .. }));
}

#[test]
fn test_tensor_byte_size_cap_with_i8() {
    // I8 (1 byte): elements = MAX_TENSOR_BYTE_SIZE
    let info = make_tensor("i8_at_cap", &[MAX_TENSOR_BYTE_SIZE], GgufDType::I8);
    let byte_size = info.checked_byte_size().unwrap();
    assert_eq!(byte_size, MAX_TENSOR_BYTE_SIZE);
}

#[test]
fn test_tensor_byte_size_cap_with_i8_over() {
    let info = make_tensor("i8_over", &[MAX_TENSOR_BYTE_SIZE + 1], GgufDType::I8);
    let err = info.checked_byte_size().unwrap_err();
    assert!(matches!(err, GgufError::TensorTooLarge { .. }));
}

#[test]
fn test_tensor_byte_size_cap_with_f64() {
    // F64 (8 bytes): elements = MAX_TENSOR_BYTE_SIZE / 8
    let elements = MAX_TENSOR_BYTE_SIZE / 8;
    let info = make_tensor("f64_at_cap", &[elements], GgufDType::F64);
    let byte_size = info.checked_byte_size().unwrap();
    assert_eq!(byte_size, MAX_TENSOR_BYTE_SIZE);
}

#[test]
fn test_tensor_byte_size_cap_multidim_under() {
    // 2D tensor: [65536, 32768] * 4 bytes = 8 GiB exactly
    let info = make_tensor("md_at", &[65536, 32768], GgufDType::F32);
    let byte_size = info.checked_byte_size().unwrap();
    assert_eq!(byte_size, 65536u64 * 32768 * 4);
    assert_eq!(byte_size, MAX_TENSOR_BYTE_SIZE);
}

#[test]
fn test_tensor_byte_size_cap_multidim_over() {
    // 2D tensor: [65536, 32769] * 4 bytes = just over 8 GiB
    let info = make_tensor("md_over", &[65536, 32769], GgufDType::F32);
    let err = info.checked_byte_size().unwrap_err();
    assert!(matches!(err, GgufError::TensorTooLarge { .. }));
}

// ===================================================================
// Additional: Quantized type byte-size edge cases
// ===================================================================

#[test]
fn test_q8_0_byte_size_correct() {
    // Q8_0: block_size=32, type_size=34
    // 256 elements / 32 = 8 blocks * 34 = 272 bytes
    let info = make_tensor("q80", &[256], GgufDType::Q8_0);
    assert_eq!(info.checked_byte_size().unwrap(), 272);
}

#[test]
fn test_q6k_byte_size_correct() {
    // Q6K: block_size=256, type_size=210
    // 512 elements / 256 = 2 blocks * 210 = 420 bytes
    let info = make_tensor("q6k", &[512], GgufDType::Q6K);
    assert_eq!(info.checked_byte_size().unwrap(), 420);
}

#[test]
fn test_q2k_large_tensor_under_cap() {
    // Q2K: block_size=256, type_size=84
    // Largest Q2K tensor under 8 GiB:
    // 8 GiB / 84 = 102261126.095 blocks => 102261126 * 256 elements
    let blocks = MAX_TENSOR_BYTE_SIZE / 84;
    let elements = blocks * 256;
    let info = make_tensor("q2k_large", &[elements], GgufDType::Q2K);
    let byte_size = info.checked_byte_size().unwrap();
    assert!(byte_size <= MAX_TENSOR_BYTE_SIZE);
}

// ===================================================================
// Additional: truncated input at various parse stages
// ===================================================================

#[test]
fn test_truncated_after_magic() {
    let data = GGUF_MAGIC.to_le_bytes();
    let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
    assert!(matches!(err, GgufError::Io(_)));
}

#[test]
fn test_truncated_after_version() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
    assert!(matches!(err, GgufError::Io(_)));
}

#[test]
fn test_truncated_after_tensor_count() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    // Missing metadata_kv_count
    let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
    assert!(matches!(err, GgufError::Io(_)));
}

#[test]
fn test_truncated_mid_tensor_info() {
    let mut data = valid_header(1, 0);
    let name = b"t";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name);
    data.extend_from_slice(&2u32.to_le_bytes()); // n_dims = 2
    data.extend_from_slice(&4u64.to_le_bytes()); // dim 0 = 4
                                                 // Missing dim 1, dtype, offset
    let err = GgufFile::read_from(&mut std::io::Cursor::new(data)).unwrap_err();
    assert!(matches!(err, GgufError::Io(_)));
}

#[test]
fn test_truncated_mid_metadata_value() {
    let mut data = valid_header(0, 1);
    let key = b"k";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    data.extend_from_slice(&6u32.to_le_bytes()); // FLOAT32 type
                                                 // Missing the actual f32 value bytes
    let err = GgufFile::read_from(&mut std::io::Cursor::new(data)).unwrap_err();
    assert!(matches!(err, GgufError::Io(_)));
}

// ===================================================================
// Additional: metadata type ID edge cases
// ===================================================================

#[test]
fn test_metadata_type_id_just_past_valid_range_rejected() {
    // Valid type IDs are 0-12 and some gaps. 13 is invalid.
    let mut data = valid_header(0, 1);
    let key = b"k";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    data.extend_from_slice(&13u32.to_le_bytes()); // Invalid type
    let err = GgufFile::read_from(&mut std::io::Cursor::new(data)).unwrap_err();
    assert!(matches!(
        err,
        GgufError::UnknownMetadataType { type_id: 13 }
    ));
}

#[test]
fn test_metadata_type_id_u32_max_rejected() {
    let mut data = valid_header(0, 1);
    let key = b"k";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    data.extend_from_slice(&u32::MAX.to_le_bytes());
    let err = GgufFile::read_from(&mut std::io::Cursor::new(data)).unwrap_err();
    assert!(matches!(
        err,
        GgufError::UnknownMetadataType { type_id } if type_id == u32::MAX
    ));
}

// ===================================================================
// Additional: dtype_from_u32 edge cases for tensor dtype
// ===================================================================

#[test]
fn test_tensor_dtype_gap_values_rejected() {
    // GgufDType has gaps: 4, 5, 15 are not valid dtype IDs.
    for invalid_id in [4u32, 5, 15, 31, 100, 999, u32::MAX] {
        let mut data = valid_header(1, 0);
        let name = b"t";
        data.extend_from_slice(&(name.len() as u64).to_le_bytes());
        data.extend_from_slice(name);
        data.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
        data.extend_from_slice(&4u64.to_le_bytes()); // dim = 4
        data.extend_from_slice(&invalid_id.to_le_bytes()); // invalid dtype
        let err = GgufFile::read_from(&mut std::io::Cursor::new(data)).unwrap_err();
        assert!(
            matches!(err, GgufError::UnknownMetadataType { .. }),
            "dtype {invalid_id} should be rejected, got: {err}"
        );
    }
}

// ===================================================================
// Additional: empty file and minimal inputs
// ===================================================================

#[test]
fn test_zero_byte_input_rejected() {
    let data: &[u8] = &[];
    let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
    assert!(matches!(err, GgufError::Io(_)));
}

#[test]
fn test_three_byte_input_rejected() {
    let data: &[u8] = &[0x47, 0x55, 0x46]; // partial "GUF"
    let err = GgufHeader::read_from(&mut &data[..]).unwrap_err();
    assert!(matches!(err, GgufError::Io(_)));
}

#[test]
fn test_valid_minimal_gguf_zero_tensors_zero_metadata() {
    let data = valid_header(0, 0);
    let file = GgufFile::read_from(&mut std::io::Cursor::new(data)).unwrap();
    assert_eq!(file.header.version, 3);
    assert_eq!(file.header.tensor_count, 0);
    assert_eq!(file.header.metadata_kv_count, 0);
    assert!(file.tensors.is_empty());
}
