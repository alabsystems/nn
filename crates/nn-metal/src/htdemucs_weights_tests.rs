// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `ParsedSafetensors`, `TensorSource`, and `extract()`.
//!
//! Verifies memory safety invariants: bounds validation on parsed tensor
//! offsets, correct f32 extraction, byte alignment checks.

use super::*;

/// Construct a `ParsedSafetensors` with known-good offsets and verify
/// that `tensor_bytes` returns the correct subslice.
#[test]
fn test_parsed_safetensors_tensor_bytes_valid_offset() {
    let data = vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let mut tensors = std::collections::HashMap::new();
    tensors.insert("a".to_string(), (2, 4)); // bytes [2..6]
    let ps = ParsedSafetensors { data, tensors };
    let bytes = ps.tensor_bytes("a").expect("valid tensor");
    assert_eq!(bytes, &[2, 3, 4, 5]);
}

/// Verify that `tensor_bytes` returns `MissingTensor` for unknown names.
#[test]
fn test_parsed_safetensors_missing_tensor() {
    let ps = ParsedSafetensors {
        data: vec![0; 10],
        tensors: std::collections::HashMap::new(),
    };
    let err = ps.tensor_bytes("nonexistent").unwrap_err();
    assert!(matches!(err, WeightLoadError::MissingTensor { .. }));
}

/// Verify that `extract` rejects non-f32-aligned byte lengths.
#[test]
fn test_extract_byte_alignment_error() {
    struct BadSource;
    impl TensorSource for BadSource {
        fn tensor_bytes(&self, _: &str) -> Result<&[u8], WeightLoadError> {
            // 5 bytes is not a multiple of 4 (f32 size)
            Ok(&[1, 2, 3, 4, 5])
        }
    }
    let err = extract(&BadSource, "test").unwrap_err();
    assert!(matches!(err, WeightLoadError::ByteAlignment { .. }));
}

/// Verify that `extract` correctly converts aligned bytes to f32.
#[test]
fn test_extract_f32_roundtrip() {
    let val: f32 = 3.125; // exact in f32, avoids clippy::approx_constant
    let bytes = val.to_le_bytes(); // extract() uses from_le_bytes
    struct F32Source([u8; 4]);
    impl TensorSource for F32Source {
        fn tensor_bytes(&self, _: &str) -> Result<&[u8], WeightLoadError> {
            Ok(&self.0)
        }
    }
    let result = extract(&F32Source(bytes), "test").expect("valid f32");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], 3.125);
}

/// Verify that `tensor_bytes` correctly handles a tensor at offset 0.
#[test]
fn test_parsed_safetensors_offset_zero() {
    let data = vec![10u8, 20, 30, 40];
    let mut tensors = std::collections::HashMap::new();
    tensors.insert("first".to_string(), (0, 4));
    let ps = ParsedSafetensors { data, tensors };
    let bytes = ps.tensor_bytes("first").expect("valid tensor");
    assert_eq!(bytes, &[10, 20, 30, 40]);
}

/// Verify that `tensor_bytes` handles a tensor at the end of the data buffer.
#[test]
fn test_parsed_safetensors_tensor_at_end() {
    let data = vec![0u8; 100];
    let mut tensors = std::collections::HashMap::new();
    tensors.insert("tail".to_string(), (96, 4)); // last 4 bytes
    let ps = ParsedSafetensors { data, tensors };
    let bytes = ps.tensor_bytes("tail").expect("valid tensor");
    assert_eq!(bytes.len(), 4);
}

/// Verify that multiple tensors in the same ParsedSafetensors are
/// independently addressable.
#[test]
fn test_parsed_safetensors_multiple_tensors() {
    let data: Vec<u8> = (0..20).collect();
    let mut tensors = std::collections::HashMap::new();
    tensors.insert("a".to_string(), (0, 8));
    tensors.insert("b".to_string(), (8, 4));
    tensors.insert("c".to_string(), (12, 8));
    let ps = ParsedSafetensors { data, tensors };

    assert_eq!(ps.tensor_bytes("a").unwrap(), &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(ps.tensor_bytes("b").unwrap(), &[8, 9, 10, 11]);
    assert_eq!(
        ps.tensor_bytes("c").unwrap(),
        &[12, 13, 14, 15, 16, 17, 18, 19]
    );
}

/// Verify that `extract` handles multi-element f32 data.
#[test]
fn test_extract_multiple_f32() {
    let vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    struct MultiF32(Vec<u8>);
    impl TensorSource for MultiF32 {
        fn tensor_bytes(&self, _: &str) -> Result<&[u8], WeightLoadError> {
            Ok(&self.0)
        }
    }
    let result = extract(&MultiF32(bytes), "test").expect("valid f32s");
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
}

/// AC3 (#940): Verify `extract()` handles data at misaligned memory addresses.
///
/// The old `bytemuck::cast_slice::<u8, f32>` panics when the byte slice pointer
/// is not 4-byte aligned. The `f32::from_le_bytes` approach is alignment-agnostic.
/// This test creates a buffer where the f32 data starts at offset 1 (odd address).
#[test]
fn test_extract_misaligned_bytes_no_panic() {
    let val: f32 = 42.5;
    let le_bytes = val.to_le_bytes();

    // Put f32 LE bytes at offset 1 so the slice starts at a non-4-byte-aligned address.
    let mut buf = vec![0xFFu8; 5];
    buf[1] = le_bytes[0];
    buf[2] = le_bytes[1];
    buf[3] = le_bytes[2];
    buf[4] = le_bytes[3];

    struct MisalignedSource(Vec<u8>);
    impl TensorSource for MisalignedSource {
        fn tensor_bytes(&self, _: &str) -> Result<&[u8], WeightLoadError> {
            Ok(&self.0[1..]) // starts at odd offset
        }
    }

    let result =
        extract(&MisalignedSource(buf), "test").expect("should not panic on misaligned data");
    assert_eq!(result.len(), 1);
    assert!((result[0] - 42.5).abs() < f32::EPSILON);
}

/// AC3 (#940): Multi-element misaligned extraction preserves all values.
#[test]
fn test_extract_misaligned_multiple_f32() {
    let vals: Vec<f32> = vec![1.0, -2.75, 0.0, f32::MAX];
    // Build buffer with 3-byte prefix to ensure misalignment (offset 3 mod 4 = 3).
    let mut buf = vec![0xAAu8; 3];
    for v in &vals {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    struct MisalignedMulti(Vec<u8>);
    impl TensorSource for MisalignedMulti {
        fn tensor_bytes(&self, _: &str) -> Result<&[u8], WeightLoadError> {
            Ok(&self.0[3..]) // offset 3 — not 4-byte aligned
        }
    }

    let result = extract(&MisalignedMulti(buf), "test").expect("should not panic");
    assert_eq!(result.len(), 4);
    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], -2.75);
    assert_eq!(result[2], 0.0);
    assert_eq!(result[3], f32::MAX);
}

/// AC4 (#940): Extract returns `ByteAlignment` error (not panic) for 1-byte input.
#[test]
fn test_extract_single_byte_returns_error() {
    struct SingleByte;
    impl TensorSource for SingleByte {
        fn tensor_bytes(&self, _: &str) -> Result<&[u8], WeightLoadError> {
            Ok(&[0x42])
        }
    }
    let err = extract(&SingleByte, "w").unwrap_err();
    match err {
        WeightLoadError::ByteAlignment { ref name, len } => {
            assert_eq!(name, "w");
            assert_eq!(len, 1);
        }
        other => panic!("expected ByteAlignment, got: {other}"),
    }
}

/// AC4 (#940): Extract returns `ByteAlignment` error for 3-byte input.
#[test]
fn test_extract_three_bytes_returns_error() {
    struct ThreeBytes;
    impl TensorSource for ThreeBytes {
        fn tensor_bytes(&self, _: &str) -> Result<&[u8], WeightLoadError> {
            Ok(&[1, 2, 3])
        }
    }
    let err = extract(&ThreeBytes, "bias").unwrap_err();
    match err {
        WeightLoadError::ByteAlignment { ref name, len } => {
            assert_eq!(name, "bias");
            assert_eq!(len, 3);
        }
        other => panic!("expected ByteAlignment, got: {other}"),
    }
}

/// AC4 (#940): Extract succeeds on empty byte slice (0 floats).
#[test]
fn test_extract_empty_bytes_returns_empty_vec() {
    struct EmptySource;
    impl TensorSource for EmptySource {
        fn tensor_bytes(&self, _: &str) -> Result<&[u8], WeightLoadError> {
            Ok(&[])
        }
    }
    let result = extract(&EmptySource, "test").expect("empty is valid");
    assert!(result.is_empty());
}

/// AC4 (#940): `ParsedSafetensors` with out-of-bounds tensor offset returns
/// an `OutOfBounds` error instead of panicking.
#[test]
fn test_parsed_safetensors_oob_offset_returns_error() {
    let data = vec![0u8; 4];
    let mut tensors = std::collections::HashMap::new();
    // Offset 2 + length 4 = 6 > data.len() (4) — out of bounds.
    tensors.insert("oob".to_string(), (2, 4));
    let ps = ParsedSafetensors { data, tensors };
    let err = ps.tensor_bytes("oob").unwrap_err();
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("exceeds buffer length"),
        "expected OutOfBounds error, got: {err_msg}"
    );
}

/// AC5 (#943): extract() rejects NaN values in weight tensor.
#[test]
fn test_extract_nan_weight_rejected() {
    let vals: Vec<f32> = vec![1.0, f32::NAN, 3.0];
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    struct NanSource(Vec<u8>);
    impl TensorSource for NanSource {
        fn tensor_bytes(&self, _: &str) -> Result<&[u8], WeightLoadError> {
            Ok(&self.0)
        }
    }
    let err = extract(&NanSource(bytes), "weight_with_nan").unwrap_err();
    match err {
        WeightLoadError::NonFiniteWeight { ref name, count } => {
            assert_eq!(name, "weight_with_nan");
            assert_eq!(count, 1);
        }
        other => panic!("expected NonFiniteWeight, got: {other}"),
    }
}

/// AC5 (#943): extract() rejects Inf values in weight tensor.
#[test]
fn test_extract_inf_weight_rejected() {
    let vals: Vec<f32> = vec![f32::INFINITY, 2.0, f32::NEG_INFINITY];
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    struct InfSource(Vec<u8>);
    impl TensorSource for InfSource {
        fn tensor_bytes(&self, _: &str) -> Result<&[u8], WeightLoadError> {
            Ok(&self.0)
        }
    }
    let err = extract(&InfSource(bytes), "bias_with_inf").unwrap_err();
    match err {
        WeightLoadError::NonFiniteWeight { ref name, count } => {
            assert_eq!(name, "bias_with_inf");
            assert_eq!(count, 2);
        }
        other => panic!("expected NonFiniteWeight, got: {other}"),
    }
}

/// AC5 (#943): extract() accepts all-finite weight tensor.
#[test]
fn test_extract_finite_weight_accepted() {
    let vals: Vec<f32> = vec![0.0, -1.0, f32::MAX, f32::MIN, f32::MIN_POSITIVE];
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    struct FiniteSource(Vec<u8>);
    impl TensorSource for FiniteSource {
        fn tensor_bytes(&self, _: &str) -> Result<&[u8], WeightLoadError> {
            Ok(&self.0)
        }
    }
    let result =
        extract(&FiniteSource(bytes), "good_weight").expect("should accept finite weights");
    assert_eq!(result.len(), 5);
}
