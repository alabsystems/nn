// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Security hardening tests for GGUF parser against malicious/crafted inputs.
//!
//! These tests verify that crafted GGUF files cannot trigger:
//! - Uncontrolled memory allocation (allocation bombs)
//! - Integer overflow in size computations
//! - Out-of-bounds reads from the data section
//! - Panics from unchecked arithmetic

use crate::dequant::GgufDType;
use crate::error::GgufError;
use crate::header::GGUF_MAGIC;
use crate::reader::GgufFile;
use crate::tensor_info::GgufTensorInfo;

/// Helper: build a minimal GGUF v3 header with given tensor/metadata counts.
fn gguf_header(tensor_count: u64, metadata_kv_count: u64) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&tensor_count.to_le_bytes());
    data.extend_from_slice(&metadata_kv_count.to_le_bytes());
    data
}

/// Helper: append a tensor info entry to a byte buffer.
fn append_tensor_info(buf: &mut Vec<u8>, name: &str, shape: &[u64], dtype: GgufDType, offset: u64) {
    // Name: u64 length + bytes.
    buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
    buf.extend_from_slice(name.as_bytes());
    // n_dims: u32.
    buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    // Dimensions: u64 each.
    for &dim in shape {
        buf.extend_from_slice(&dim.to_le_bytes());
    }
    // dtype: u32.
    buf.extend_from_slice(&(dtype as u32).to_le_bytes());
    // offset: u64.
    buf.extend_from_slice(&offset.to_le_bytes());
}

/// Helper: append a string metadata KV entry.
fn append_string_metadata(buf: &mut Vec<u8>, key: &str, val: &str) {
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes()); // STRING type
    buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
    buf.extend_from_slice(val.as_bytes());
}

/// Helper: pad buffer to 32-byte alignment.
fn pad_to_alignment(buf: &mut Vec<u8>) {
    while !buf.len().is_multiple_of(32) {
        buf.push(0);
    }
}

// ---------------------------------------------------------------
// Tensor count / metadata count caps
// ---------------------------------------------------------------

#[test]
fn test_huge_tensor_count_rejected() {
    let data = gguf_header(u64::MAX, 0);
    let mut cursor = std::io::Cursor::new(data);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    assert!(
        matches!(err, GgufError::TensorCountExceeded { .. }),
        "expected TensorCountExceeded, got: {err}"
    );
}

#[test]
fn test_huge_metadata_count_rejected() {
    let data = gguf_header(0, u64::MAX);
    let mut cursor = std::io::Cursor::new(data);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    assert!(
        matches!(err, GgufError::MetadataCountExceeded { .. }),
        "expected MetadataCountExceeded, got: {err}"
    );
}

#[test]
fn test_tensor_count_just_over_limit_rejected() {
    let data = gguf_header(100_001, 0);
    let mut cursor = std::io::Cursor::new(data);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    assert!(matches!(err, GgufError::TensorCountExceeded { .. }));
}

#[test]
fn test_metadata_count_just_over_limit_rejected() {
    let data = gguf_header(0, 100_001);
    let mut cursor = std::io::Cursor::new(data);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    assert!(matches!(err, GgufError::MetadataCountExceeded { .. }));
}

// ---------------------------------------------------------------
// String length caps
// ---------------------------------------------------------------

#[test]
fn test_huge_string_length_in_metadata_key_rejected() {
    let mut data = gguf_header(0, 1);
    // Metadata key with absurd length.
    data.extend_from_slice(&(u64::MAX).to_le_bytes());
    let mut cursor = std::io::Cursor::new(data);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    assert!(
        matches!(err, GgufError::StringLengthExceeded { .. }),
        "expected StringLengthExceeded, got: {err}"
    );
}

#[test]
fn test_huge_string_length_in_tensor_name_rejected() {
    let mut data = gguf_header(1, 0);
    // Tensor name with absurd length.
    data.extend_from_slice(&(u64::MAX).to_le_bytes());
    let mut cursor = std::io::Cursor::new(data);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    assert!(
        matches!(err, GgufError::StringLengthExceeded { .. }),
        "expected StringLengthExceeded, got: {err}"
    );
}

#[test]
fn test_string_length_just_over_limit_rejected() {
    let max_str_len: u64 = 16 * 1024 * 1024;
    let mut data = gguf_header(0, 1);
    // Key with length = MAX + 1.
    data.extend_from_slice(&(max_str_len + 1).to_le_bytes());
    let mut cursor = std::io::Cursor::new(data);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    assert!(matches!(err, GgufError::StringLengthExceeded { .. }));
}

// ---------------------------------------------------------------
// Dimension count caps
// ---------------------------------------------------------------

#[test]
fn test_huge_dimension_count_rejected() {
    let mut data = gguf_header(1, 0);
    // Tensor name.
    let name = b"evil";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name);
    // n_dims = u32::MAX.
    data.extend_from_slice(&u32::MAX.to_le_bytes());
    let mut cursor = std::io::Cursor::new(data);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    assert!(
        matches!(err, GgufError::DimensionCountExceeded { .. }),
        "expected DimensionCountExceeded, got: {err}"
    );
}

#[test]
fn test_dimension_count_9_rejected() {
    let mut data = gguf_header(1, 0);
    let name = b"evil";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name);
    data.extend_from_slice(&9u32.to_le_bytes()); // n_dims = 9 (max is 8)
    let mut cursor = std::io::Cursor::new(data);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    assert!(matches!(err, GgufError::DimensionCountExceeded { .. }));
}

// ---------------------------------------------------------------
// Zero dimension validation
// ---------------------------------------------------------------

#[test]
fn test_zero_dimension_rejected() {
    let mut data = gguf_header(1, 0);
    let name = b"evil_zero_dim";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name);
    data.extend_from_slice(&2u32.to_le_bytes()); // n_dims = 2
    data.extend_from_slice(&128u64.to_le_bytes()); // dim 0 = 128
    data.extend_from_slice(&0u64.to_le_bytes()); // dim 1 = 0 (INVALID)
    let mut cursor = std::io::Cursor::new(data);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    assert!(
        matches!(err, GgufError::ZeroDimension { .. }),
        "expected ZeroDimension, got: {err}"
    );
}

#[test]
fn test_first_dimension_zero_rejected() {
    let mut data = gguf_header(1, 0);
    let name = b"zd";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name);
    data.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
    data.extend_from_slice(&0u64.to_le_bytes()); // dim 0 = 0 (INVALID)
    let mut cursor = std::io::Cursor::new(data);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    assert!(matches!(err, GgufError::ZeroDimension { .. }));
}

// ---------------------------------------------------------------
// Element count overflow
// ---------------------------------------------------------------

#[test]
fn test_element_count_overflow_rejected() {
    let info = GgufTensorInfo {
        name: "overflow_tensor".into(),
        n_dims: 3,
        shape: vec![u64::MAX / 2, 3, 2], // product overflows u64
        dtype: GgufDType::F32,
        offset: 0,
    };
    let err = info.checked_num_elements().unwrap_err();
    assert!(
        matches!(err, GgufError::ElementCountOverflow { .. }),
        "expected ElementCountOverflow, got: {err}"
    );
}

#[test]
fn test_byte_size_overflow_via_element_count() {
    let info = GgufTensorInfo {
        name: "overflow_tensor".into(),
        n_dims: 2,
        shape: vec![u64::MAX / 4, 5], // product overflows
        dtype: GgufDType::F32,
        offset: 0,
    };
    let err = info.checked_byte_size().unwrap_err();
    assert!(
        matches!(
            err,
            GgufError::ElementCountOverflow { .. } | GgufError::ByteSizeOverflow { .. }
        ),
        "expected overflow error, got: {err}"
    );
}

// ---------------------------------------------------------------
// Tensor byte size cap
// ---------------------------------------------------------------

#[test]
fn test_tensor_byte_size_cap_enforced() {
    // 8 GiB + 1 byte worth of F32 data = (8*1024*1024*1024 / 4) + 1 elements.
    let elements = (8u64 * 1024 * 1024 * 1024) / 4 + 1;
    let info = GgufTensorInfo {
        name: "too_large".into(),
        n_dims: 1,
        shape: vec![elements],
        dtype: GgufDType::F32,
        offset: 0,
    };
    let err = info.checked_byte_size().unwrap_err();
    assert!(
        matches!(err, GgufError::TensorTooLarge { .. }),
        "expected TensorTooLarge, got: {err}"
    );
}

#[test]
fn test_tensor_byte_size_at_limit_passes() {
    // Exactly 8 GiB of F32 data = 8*1024*1024*1024 / 4 elements.
    let elements = 8u64 * 1024 * 1024 * 1024 / 4;
    let info = GgufTensorInfo {
        name: "at_limit".into(),
        n_dims: 1,
        shape: vec![elements],
        dtype: GgufDType::F32,
        offset: 0,
    };
    let byte_size = info.checked_byte_size().unwrap();
    assert_eq!(byte_size, 8 * 1024 * 1024 * 1024);
}

// ---------------------------------------------------------------
// Data offset overflow
// ---------------------------------------------------------------

#[test]
fn test_data_offset_overflow_in_tensor_data() {
    // Craft a GGUF where a tensor's offset would overflow when added to data_offset.
    let mut data = gguf_header(1, 0);
    append_tensor_info(&mut data, "evil", &[4], GgufDType::F32, u64::MAX);
    pad_to_alignment(&mut data);
    // Append minimal tensor data.
    data.extend_from_slice(&[0u8; 16]);

    let path = std::env::temp_dir().join("nn_gguf_sec_offset_overflow.gguf");
    std::fs::write(&path, &data).unwrap();

    let result = GgufFile::open(&path);
    // Should fail: data_offset + tensor.offset overflows.
    assert!(
        result.is_err(),
        "expected error for offset overflow, got Ok"
    );
    let err = result.unwrap_err();
    assert!(
        matches!(
            err,
            GgufError::DataOffsetOverflow { .. } | GgufError::DataOutOfBounds { .. }
        ),
        "expected DataOffsetOverflow or DataOutOfBounds, got: {err}"
    );

    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------
// Data region out of file bounds
// ---------------------------------------------------------------

#[test]
fn test_tensor_data_beyond_file_bounds_mmap() {
    // Create a GGUF with a tensor that claims a large offset, but the
    // file doesn't actually contain that much data.
    let mut data = gguf_header(1, 0);
    // Tensor with shape [1024], F32, offset=0. Needs 4096 bytes of data.
    append_tensor_info(&mut data, "big", &[1024], GgufDType::F32, 0);
    pad_to_alignment(&mut data);
    // Only provide 64 bytes of data (not enough for 4096).
    data.extend_from_slice(&[0u8; 64]);

    let path = std::env::temp_dir().join("nn_gguf_sec_oob_mmap.gguf");
    std::fs::write(&path, &data).unwrap();

    let result = GgufFile::open(&path);
    assert!(result.is_err(), "expected out-of-bounds error, got Ok");
    let err = result.unwrap_err();
    assert!(
        matches!(err, GgufError::DataOutOfBounds { .. }),
        "expected DataOutOfBounds, got: {err}"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_tensor_data_beyond_file_bounds_read() {
    // Same test but via read_tensor_f32 (stream path).
    let mut data = gguf_header(1, 0);
    append_tensor_info(&mut data, "big", &[1024], GgufDType::F32, 0);
    pad_to_alignment(&mut data);
    data.extend_from_slice(&[0u8; 64]); // insufficient data

    let mut cursor = std::io::Cursor::new(&data);
    let file = GgufFile::read_from(&mut cursor).unwrap();

    let result = file.read_tensor_f32(&mut cursor, "big");
    assert!(result.is_err(), "expected out-of-bounds error, got Ok");
    let err = result.unwrap_err();
    assert!(
        matches!(err, GgufError::DataOutOfBounds { .. }),
        "expected DataOutOfBounds, got: {err}"
    );
}

// ---------------------------------------------------------------
// Array length caps in metadata
// ---------------------------------------------------------------

#[test]
fn test_huge_metadata_array_rejected() {
    let mut data = gguf_header(0, 1);
    // Key.
    let key = b"evil_array";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    // ARRAY type (9).
    data.extend_from_slice(&9u32.to_le_bytes());
    // Element type: U8 (0).
    data.extend_from_slice(&0u32.to_le_bytes());
    // Array count: absurd.
    data.extend_from_slice(&u64::MAX.to_le_bytes());

    let mut cursor = std::io::Cursor::new(data);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    assert!(
        matches!(err, GgufError::ArrayLengthExceeded { .. }),
        "expected ArrayLengthExceeded, got: {err}"
    );
}

// ---------------------------------------------------------------
// Non-UTF-8 string in metadata key
// ---------------------------------------------------------------

#[test]
fn test_invalid_utf8_string_rejected() {
    let mut data = gguf_header(0, 1);
    // Key with invalid UTF-8 bytes.
    let invalid_bytes: &[u8] = &[0xFF, 0xFE, 0x80, 0x81];
    data.extend_from_slice(&(invalid_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(invalid_bytes);

    let mut cursor = std::io::Cursor::new(data);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    assert!(
        matches!(err, GgufError::InvalidUtf8 { .. }),
        "expected InvalidUtf8, got: {err}"
    );
}

// ---------------------------------------------------------------
// Truncated file (not enough bytes for header)
// ---------------------------------------------------------------

#[test]
fn test_truncated_header_rejected() {
    // Only 8 bytes (magic + partial version).
    let data = GGUF_MAGIC.to_le_bytes();
    let mut cursor = std::io::Cursor::new(data.to_vec());
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    assert!(
        matches!(err, GgufError::Io(_)),
        "expected Io error for truncated header, got: {err}"
    );
}

#[test]
fn test_empty_file_rejected() {
    let data: Vec<u8> = Vec::new();
    let mut cursor = std::io::Cursor::new(data);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    assert!(matches!(
        err,
        GgufError::Io(_) | GgufError::InvalidMagic { .. }
    ));
}

// ---------------------------------------------------------------
// Unknown dtype
// ---------------------------------------------------------------

#[test]
fn test_unknown_tensor_dtype_rejected() {
    let mut data = gguf_header(1, 0);
    let name = b"evil_dtype";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name);
    data.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
    data.extend_from_slice(&8u64.to_le_bytes()); // dim 0 = 8
    data.extend_from_slice(&999u32.to_le_bytes()); // dtype = 999 (invalid)

    let mut cursor = std::io::Cursor::new(data);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    assert!(
        matches!(err, GgufError::UnknownMetadataType { type_id: 999 }),
        "expected UnknownMetadataType, got: {err}"
    );
}

// ---------------------------------------------------------------
// Integration: valid minimal file still works
// ---------------------------------------------------------------

#[test]
fn test_valid_file_with_tensor_still_works() {
    let mut data = gguf_header(1, 1);
    append_string_metadata(&mut data, "general.architecture", "llama");
    append_tensor_info(&mut data, "weight", &[4, 3], GgufDType::F32, 0);
    pad_to_alignment(&mut data);
    // 12 f32 values.
    for i in 0..12 {
        data.extend_from_slice(&(i as f32).to_le_bytes());
    }

    let mut cursor = std::io::Cursor::new(&data);
    let file = GgufFile::read_from(&mut cursor).unwrap();
    assert_eq!(file.header.tensor_count, 1);
    assert_eq!(file.architecture(), Some("llama"));

    let (values, shape) = file.read_tensor_f32(&mut cursor, "weight").unwrap();
    assert_eq!(shape, vec![4, 3]);
    assert_eq!(values.len(), 12);
    for i in 0..12 {
        assert!((values[i] - i as f32).abs() < 1e-6);
    }
}

#[test]
fn test_valid_file_mmap_with_tensor_still_works() {
    let mut data = gguf_header(1, 1);
    append_string_metadata(&mut data, "general.architecture", "llama");
    append_tensor_info(&mut data, "weight", &[4, 3], GgufDType::F32, 0);
    pad_to_alignment(&mut data);
    for i in 0..12 {
        data.extend_from_slice(&(i as f32).to_le_bytes());
    }

    let path = std::env::temp_dir().join("nn_gguf_sec_valid.gguf");
    std::fs::write(&path, &data).unwrap();

    let file = GgufFile::open(&path).unwrap();
    assert_eq!(file.header.tensor_count, 1);

    let raw = file.tensor_data("weight").unwrap().unwrap();
    assert_eq!(raw.len(), 48); // 12 * 4 bytes

    let (values, shape) = file.dequantize_tensor("weight").unwrap();
    assert_eq!(shape, vec![4, 3]);
    assert_eq!(values.len(), 12);

    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------
// Multiple tensors with overlapping data regions
// ---------------------------------------------------------------

#[test]
fn test_overlapping_tensor_offsets_accepted() {
    // GGUF allows tensors to overlap (e.g., shared weights). This should
    // still parse successfully. The security concern is out-of-bounds, not
    // overlap.
    let mut data = gguf_header(2, 0);
    // Both tensors reference the same data region.
    append_tensor_info(&mut data, "a", &[4], GgufDType::F32, 0);
    append_tensor_info(&mut data, "b", &[4], GgufDType::F32, 0);
    pad_to_alignment(&mut data);
    data.extend_from_slice(&[0u8; 16]); // 4 f32s

    let path = std::env::temp_dir().join("nn_gguf_sec_overlap.gguf");
    std::fs::write(&path, &data).unwrap();

    let file = GgufFile::open(&path).unwrap();
    assert_eq!(file.tensors.len(), 2);
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------
// Enormous single dimension
// ---------------------------------------------------------------

#[test]
fn test_enormous_single_dimension_byte_size_cap() {
    // Single dimension of 4 billion F32 = 16 GiB > 8 GiB cap.
    let info = GgufTensorInfo {
        name: "enormous".into(),
        n_dims: 1,
        shape: vec![4_000_000_000],
        dtype: GgufDType::F32,
        offset: 0,
    };
    let err = info.checked_byte_size().unwrap_err();
    assert!(
        matches!(err, GgufError::TensorTooLarge { .. }),
        "expected TensorTooLarge, got: {err}"
    );
}

// ---------------------------------------------------------------
// Large but valid tensor passes
// ---------------------------------------------------------------

#[test]
fn test_large_valid_tensor_passes_byte_size_check() {
    // Llama-405B embedding: 128k * 16384 = 2 billion elements * 2 bytes (F16)
    // = 4 GiB, under the 8 GiB cap.
    let info = GgufTensorInfo {
        name: "token_embd.weight".into(),
        n_dims: 2,
        shape: vec![128_000, 16_384],
        dtype: GgufDType::F16,
        offset: 0,
    };
    let byte_size = info.checked_byte_size().unwrap();
    assert_eq!(byte_size, 128_000 * 16_384 * 2);
    assert!(byte_size < 8 * 1024 * 1024 * 1024);
}

// ---------------------------------------------------------------
// Metadata: unknown type
// ---------------------------------------------------------------

#[test]
fn test_unknown_metadata_value_type_rejected() {
    let mut data = gguf_header(0, 1);
    let key = b"evil_key";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    data.extend_from_slice(&255u32.to_le_bytes()); // Unknown type 255
    let mut cursor = std::io::Cursor::new(data);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    assert!(
        matches!(err, GgufError::UnknownMetadataType { type_id: 255 }),
        "expected UnknownMetadataType, got: {err}"
    );
}

// ---------------------------------------------------------------
// Nested array depth (arrays of arrays)
// ---------------------------------------------------------------

#[test]
fn test_nested_array_in_metadata_not_crash() {
    // GGUF arrays have a single element type, so a type_id=9 (ARRAY)
    // would create recursive arrays. This should be handled by hitting
    // the array length cap or EOF before stack overflow.
    let mut data = gguf_header(0, 1);
    let key = b"nested";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    data.extend_from_slice(&9u32.to_le_bytes()); // ARRAY type
    data.extend_from_slice(&9u32.to_le_bytes()); // Element type = ARRAY
    data.extend_from_slice(&2u64.to_le_bytes()); // 2 elements

    // Each sub-array: element type + count.
    // Sub-array 0:
    data.extend_from_slice(&0u32.to_le_bytes()); // U8
    data.extend_from_slice(&0u64.to_le_bytes()); // 0 elements
                                                 // Sub-array 1:
    data.extend_from_slice(&0u32.to_le_bytes()); // U8
    data.extend_from_slice(&0u64.to_le_bytes()); // 0 elements

    let mut cursor = std::io::Cursor::new(data);
    // This should parse without panic (arrays of arrays with small counts).
    let result = GgufFile::read_from(&mut cursor);
    assert!(result.is_ok(), "nested arrays should parse without panic");
}

// ---------------------------------------------------------------
// Bad magic
// ---------------------------------------------------------------

#[test]
fn test_wrong_magic_rejected() {
    let mut data = Vec::new();
    data.extend_from_slice(&0xDEADBEEFu32.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    let mut cursor = std::io::Cursor::new(data);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    assert!(matches!(err, GgufError::InvalidMagic { .. }));
}

// ---------------------------------------------------------------
// Wrong version
// ---------------------------------------------------------------

#[test]
fn test_version_2_rejected() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    let mut cursor = std::io::Cursor::new(data);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    assert!(matches!(err, GgufError::UnsupportedVersion { version: 2 }));
}

#[test]
fn test_version_4_rejected() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&4u32.to_le_bytes());
    let mut cursor = std::io::Cursor::new(data);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    assert!(matches!(err, GgufError::UnsupportedVersion { version: 4 }));
}

// ---------------------------------------------------------------
// Dequantization buffer bounds safety
// ---------------------------------------------------------------

/// Verify that dequantize functions do not panic when given a buffer
/// shorter than what num_elements would require. The validate_dequant_input
/// helper should clamp the block count to available data.
#[test]
fn test_dequant_q4_0_short_buffer_no_panic() {
    use crate::dequant::dequantize_q4_0;
    // Q4_0: 32 elements per block, 18 bytes per block.
    // Request 64 elements (2 blocks = 36 bytes), but only provide 10 bytes.
    let data = vec![0u8; 10];
    let result = dequantize_q4_0(&data, 64);
    // Should produce 0 elements (not enough data for even 1 block).
    assert_eq!(result.len(), 0);
}

#[test]
fn test_dequant_q8_0_short_buffer_no_panic() {
    use crate::dequant::dequantize_q8_0;
    // Q8_0: 32 elements per block, 34 bytes per block.
    let data = vec![0u8; 20];
    let result = dequantize_q8_0(&data, 32);
    assert_eq!(result.len(), 0);
}

#[test]
fn test_dequant_q4_k_short_buffer_no_panic() {
    use crate::dequant::dequantize_q4_k;
    // Q4_K: 256 elements per block, 144 bytes per block.
    let data = vec![0u8; 100];
    let result = dequantize_q4_k(&data, 256);
    assert_eq!(result.len(), 0);
}

#[test]
fn test_dequant_q6_k_short_buffer_no_panic() {
    use crate::dequant::dequantize_q6_k;
    let data = vec![0u8; 100];
    let result = dequantize_q6_k(&data, 256);
    assert_eq!(result.len(), 0);
}

#[test]
fn test_dequant_q2_k_short_buffer_no_panic() {
    use crate::dequant::dequantize_q2_k;
    let data = vec![0u8; 50];
    let result = dequantize_q2_k(&data, 256);
    assert_eq!(result.len(), 0);
}

#[test]
fn test_dequant_q3_k_short_buffer_no_panic() {
    use crate::dequant::dequantize_q3_k;
    let data = vec![0u8; 50];
    let result = dequantize_q3_k(&data, 256);
    assert_eq!(result.len(), 0);
}

#[test]
fn test_dequant_q5_k_short_buffer_no_panic() {
    use crate::dequant::dequantize_q5_k;
    let data = vec![0u8; 100];
    let result = dequantize_q5_k(&data, 256);
    assert_eq!(result.len(), 0);
}

#[test]
fn test_dequant_q4_1_short_buffer_no_panic() {
    use crate::dequant::dequantize_q4_1;
    let data = vec![0u8; 10];
    let result = dequantize_q4_1(&data, 32);
    assert_eq!(result.len(), 0);
}

#[test]
fn test_dequant_q5_0_short_buffer_no_panic() {
    use crate::dequant::dequantize_q5_0;
    let data = vec![0u8; 10];
    let result = dequantize_q5_0(&data, 32);
    assert_eq!(result.len(), 0);
}

#[test]
fn test_dequant_q5_1_short_buffer_no_panic() {
    use crate::dequant::dequantize_q5_1;
    let data = vec![0u8; 10];
    let result = dequantize_q5_1(&data, 32);
    assert_eq!(result.len(), 0);
}

#[test]
fn test_dequant_empty_buffer_no_panic() {
    use crate::dequant::*;
    // All dequant functions should handle empty buffers gracefully.
    assert_eq!(dequantize_q4_0(&[], 32).len(), 0);
    assert_eq!(dequantize_q4_1(&[], 32).len(), 0);
    assert_eq!(dequantize_q5_0(&[], 32).len(), 0);
    assert_eq!(dequantize_q5_1(&[], 32).len(), 0);
    assert_eq!(dequantize_q8_0(&[], 32).len(), 0);
    assert_eq!(dequantize_q2_k(&[], 256).len(), 0);
    assert_eq!(dequantize_q3_k(&[], 256).len(), 0);
    assert_eq!(dequantize_q4_k(&[], 256).len(), 0);
    assert_eq!(dequantize_q5_k(&[], 256).len(), 0);
    assert_eq!(dequantize_q6_k(&[], 256).len(), 0);
}

#[test]
fn test_dequant_zero_elements_no_panic() {
    use crate::dequant::*;
    // Requesting 0 elements should return empty vec.
    let data = vec![0u8; 100];
    assert_eq!(dequantize_q4_0(&data, 0).len(), 0);
    assert_eq!(dequantize_q8_0(&data, 0).len(), 0);
    assert_eq!(dequantize_q4_k(&data, 0).len(), 0);
    assert_eq!(dequantize_q6_k(&data, 0).len(), 0);
}

#[test]
fn test_dequant_q4_0_exact_buffer_still_works() {
    use crate::dequant::dequantize_q4_0;
    // Exact buffer: 1 block = 18 bytes for 32 elements.
    let mut block = vec![0u8; 18];
    let scale_bytes = half::f16::from_f32(1.0).to_le_bytes();
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];
    // Fill with 0x88 -> lo=8, hi=8 -> dequant to 0.0
    for i in 0..16 {
        block[2 + i] = 0x88;
    }
    let result = dequantize_q4_0(&block, 32);
    assert_eq!(result.len(), 32);
    for &v in &result {
        assert!((v - 0.0).abs() < 1e-6);
    }
}

/// Test that the mmap path rejects a tensor whose data region is truncated.
#[test]
fn test_mmap_truncated_tensor_data_rejected() {
    let mut data = gguf_header(1, 0);
    // Tensor with shape [256], F32, offset=0. Needs 1024 bytes.
    append_tensor_info(&mut data, "trunc", &[256], GgufDType::F32, 0);
    pad_to_alignment(&mut data);
    // Only provide 512 bytes (half of what's needed).
    data.extend_from_slice(&[0u8; 512]);

    let path = std::env::temp_dir().join("nn_gguf_sec_trunc_data.gguf");
    std::fs::write(&path, &data).unwrap();

    let result = GgufFile::open(&path);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, GgufError::DataOutOfBounds { .. }),
        "expected DataOutOfBounds for truncated tensor data, got: {err}"
    );

    let _ = std::fs::remove_file(&path);
}

/// Test that `read_tensor_f32` validates the stream length before reading.
#[test]
fn test_read_tensor_f32_validates_stream_length() {
    let mut data = gguf_header(1, 0);
    append_tensor_info(&mut data, "t", &[128], GgufDType::F32, 0);
    pad_to_alignment(&mut data);
    // Only 64 bytes of data, but tensor needs 512 bytes.
    data.extend_from_slice(&[0u8; 64]);

    let mut cursor = std::io::Cursor::new(&data);
    let file = GgufFile::read_from(&mut cursor).unwrap();
    let result = file.read_tensor_f32(&mut cursor, "t");
    assert!(result.is_err());
    assert!(
        matches!(result.unwrap_err(), GgufError::DataOutOfBounds { .. }),
        "should reject tensor whose data extends past stream"
    );
}
