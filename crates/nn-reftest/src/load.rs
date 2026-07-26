// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Load reference tensors from safetensors files.

use std::path::Path;

use crate::error::ReftestError;
use crate::trace::{NamedTensor, ReferenceTrace};

/// Load a reference trace from a safetensors file.
///
/// Each tensor in the file becomes a checkpoint in the trace. Tensor names
/// are sorted alphabetically for deterministic ordering (safetensors stores
/// tensors in an unordered map).
///
/// Supported dtypes for automatic f32 conversion: F32, F16, BF16, F64.
#[must_use = "returns a Result that may contain an error"]
pub fn load_safetensors(path: impl AsRef<Path>) -> Result<ReferenceTrace, ReftestError> {
    let bytes = std::fs::read(path.as_ref())?;
    load_safetensors_from_bytes(&bytes)
}

/// Load a reference trace from in-memory safetensors data.
///
/// Same as [`load_safetensors`] but operates on a byte slice instead of a file.
#[must_use = "returns a Result that may contain an error"]
pub fn load_safetensors_from_bytes(data: &[u8]) -> Result<ReferenceTrace, ReftestError> {
    let tensors = safetensors::SafeTensors::deserialize(data)?;

    // Collect and sort tensor names for deterministic ordering.
    let mut names: Vec<String> = tensors.names().into_iter().map(String::from).collect();
    names.sort();

    let mut checkpoints = Vec::with_capacity(names.len());

    for name in &names {
        let view = tensors.tensor(name)?;
        let shape: Vec<usize> = view.shape().to_vec();
        let raw = view.data();

        let f32_data = convert_to_f32(raw, view.dtype(), &shape, name)?;
        checkpoints.push(NamedTensor::new(name.clone(), shape, f32_data)?);
    }

    Ok(ReferenceTrace::from_checkpoints(checkpoints))
}

/// Convert raw tensor bytes to f32 based on dtype.
pub(crate) fn convert_to_f32(
    raw: &[u8],
    dtype: safetensors::Dtype,
    shape: &[usize],
    name: &str,
) -> Result<Vec<f32>, ReftestError> {
    let numel: usize = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| ReftestError::ShapeProductOverflow(shape.to_vec()))?;

    // Use safe byte-level decoding (from_le_bytes) instead of unsafe pointer
    // casts. Safetensors does not guarantee alignment of tensor data within
    // the buffer, so raw pointer casts could be UB.
    let checked_byte_count =
        |numel: usize, bytes_per_element: usize| -> Result<usize, ReftestError> {
            numel
                .checked_mul(bytes_per_element)
                .ok_or(ReftestError::ByteCountOverflow {
                    numel,
                    bytes_per_element,
                })
        };

    match dtype {
        safetensors::Dtype::F32 => {
            let expected_bytes = checked_byte_count(numel, 4)?;
            if raw.len() != expected_bytes {
                return Err(ReftestError::DataLengthMismatch {
                    expected: expected_bytes,
                    actual: raw.len(),
                });
            }
            Ok(raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }

        safetensors::Dtype::F64 => {
            let expected_bytes = checked_byte_count(numel, 8)?;
            if raw.len() != expected_bytes {
                return Err(ReftestError::DataLengthMismatch {
                    expected: expected_bytes,
                    actual: raw.len(),
                });
            }
            raw.chunks_exact(8)
                .enumerate()
                .map(|(i, b)| {
                    let v = f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
                    if !v.is_finite() || v.abs() > f64::from(f32::MAX) {
                        return Err(ReftestError::F64OutOfF32Range { value: v, index: i });
                    }
                    Ok(v as f32)
                })
                .collect()
        }

        safetensors::Dtype::F16 => {
            let expected_bytes = checked_byte_count(numel, 2)?;
            if raw.len() != expected_bytes {
                return Err(ReftestError::DataLengthMismatch {
                    expected: expected_bytes,
                    actual: raw.len(),
                });
            }
            Ok(raw
                .chunks_exact(2)
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect())
        }

        safetensors::Dtype::BF16 => {
            let expected_bytes = checked_byte_count(numel, 2)?;
            if raw.len() != expected_bytes {
                return Err(ReftestError::DataLengthMismatch {
                    expected: expected_bytes,
                    actual: raw.len(),
                });
            }
            Ok(raw
                .chunks_exact(2)
                .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect())
        }

        other => Err(ReftestError::UnsupportedDtype(format!(
            "tensor '{name}' has dtype {other:?}, only F32/F64/F16/BF16 are supported"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Convert f32 slice to little-endian bytes (safe alternative to unsafe pointer cast).
    fn f32_to_le_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(size_of_val(values));
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    /// Build a minimal safetensors byte buffer from f32 tensors.
    fn build_safetensors(tensors: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
        let byte_bufs: Vec<Vec<u8>> = tensors
            .iter()
            .map(|&(_, _, data)| f32_to_le_bytes(data))
            .collect();
        let mut tensor_map: Vec<(String, safetensors::tensor::TensorView<'_>)> = Vec::new();

        for (i, &(name, shape, _)) in tensors.iter().enumerate() {
            let view = safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                shape.to_vec(),
                &byte_bufs[i],
            )
            .expect("valid tensor view");
            tensor_map.push((name.to_string(), view));
        }

        safetensors::tensor::serialize(tensor_map, None).expect("serialization should succeed")
    }

    #[test]
    fn test_load_safetensors_roundtrip() {
        let data_a: Vec<f32> = vec![1.0, 2.0, 3.0];
        let data_b: Vec<f32> = vec![4.0, 5.0];

        let bytes = build_safetensors(&[("layer_b", &[2], &data_b), ("layer_a", &[3], &data_a)]);

        let trace = load_safetensors_from_bytes(&bytes).expect("loading should succeed");

        // Should be sorted alphabetically.
        assert_eq!(trace.len(), 2);
        assert_eq!(trace.get(0).expect("exists").name, "layer_a");
        assert_eq!(trace.get(1).expect("exists").name, "layer_b");
        assert_eq!(trace.get(0).expect("exists").data, vec![1.0, 2.0, 3.0]);
        assert_eq!(trace.get(1).expect("exists").data, vec![4.0, 5.0]);
    }

    #[test]
    fn test_load_empty_safetensors() {
        let bytes = build_safetensors(&[]);
        let trace = load_safetensors_from_bytes(&bytes).expect("loading should succeed");
        assert!(trace.is_empty());
    }

    #[test]
    fn test_convert_to_f32_byte_count_overflow_returns_error() {
        // numel that doesn't overflow usize but numel*8 does (F64 path).
        let numel = (usize::MAX / 4) + 1; // numel*4 overflows, numel*8 overflows
        let shape = &[numel];
        let result = convert_to_f32(&[], safetensors::Dtype::F32, shape, "overflow_test");
        assert!(
            matches!(result, Err(ReftestError::ByteCountOverflow { .. })),
            "expected ByteCountOverflow, got {result:?}",
        );
    }

    #[test]
    fn test_convert_to_f32_f64_byte_count_overflow() {
        // numel where numel*8 overflows but numel itself doesn't.
        let numel = (usize::MAX / 8) + 1;
        let shape = &[numel];
        let result = convert_to_f32(&[], safetensors::Dtype::F64, shape, "overflow_f64");
        assert!(
            matches!(result, Err(ReftestError::ByteCountOverflow { .. })),
            "expected ByteCountOverflow, got {result:?}",
        );
    }

    #[test]
    fn test_convert_f64_rejects_infinity() {
        let inf_bytes: Vec<u8> = f64::INFINITY.to_le_bytes().to_vec();
        let result = convert_to_f32(&inf_bytes, safetensors::Dtype::F64, &[1], "inf_test");
        assert!(
            matches!(result, Err(ReftestError::F64OutOfF32Range { .. })),
            "expected F64OutOfF32Range for +inf, got {result:?}",
        );
    }

    #[test]
    fn test_convert_f64_rejects_nan() {
        let nan_bytes: Vec<u8> = f64::NAN.to_le_bytes().to_vec();
        let result = convert_to_f32(&nan_bytes, safetensors::Dtype::F64, &[1], "nan_test");
        assert!(
            matches!(result, Err(ReftestError::F64OutOfF32Range { .. })),
            "expected F64OutOfF32Range for NaN, got {result:?}",
        );
    }

    #[test]
    fn test_convert_f64_rejects_out_of_f32_range() {
        // Value just above f32::MAX (~3.4e38).
        let big: f64 = f64::from(f32::MAX) * 2.0;
        let big_bytes: Vec<u8> = big.to_le_bytes().to_vec();
        let result = convert_to_f32(&big_bytes, safetensors::Dtype::F64, &[1], "big_test");
        assert!(
            matches!(result, Err(ReftestError::F64OutOfF32Range { .. })),
            "expected F64OutOfF32Range for value > f32::MAX, got {result:?}",
        );
    }

    #[test]
    fn test_convert_f64_accepts_valid_values() {
        let values: Vec<f64> = vec![1.0, -1.0, 0.0, 2.5];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = convert_to_f32(&bytes, safetensors::Dtype::F64, &[4], "valid_test");
        let data = result.expect("valid f64 values should convert to f32");
        assert_eq!(data.len(), 4);
        assert!((data[0] - 1.0).abs() < f32::EPSILON);
        assert!((data[3] - 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_convert_f16_roundtrip() {
        let f16_vals = [half::f16::from_f32(1.0),
            half::f16::from_f32(-0.5),
            half::f16::from_f32(0.0)];
        let bytes: Vec<u8> = f16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = convert_to_f32(&bytes, safetensors::Dtype::F16, &[3], "f16_test");
        let data = result.expect("f16 conversion should succeed");
        assert_eq!(data.len(), 3);
        assert!((data[0] - 1.0).abs() < 0.01);
        assert!((data[1] - (-0.5)).abs() < 0.01);
        assert!((data[2] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_convert_bf16_roundtrip() {
        let bf16_vals = [half::bf16::from_f32(2.0),
            half::bf16::from_f32(-3.5),
            half::bf16::from_f32(0.125)];
        let bytes: Vec<u8> = bf16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = convert_to_f32(&bytes, safetensors::Dtype::BF16, &[3], "bf16_test");
        let data = result.expect("bf16 conversion should succeed");
        assert_eq!(data.len(), 3);
        assert!((data[0] - 2.0).abs() < 0.1);
        assert!((data[1] - (-3.5)).abs() < 0.1);
        assert!((data[2] - 0.125).abs() < 0.01);
    }

    #[test]
    fn test_convert_f32_data_length_mismatch() {
        // 3 elements need 12 bytes, provide only 8.
        let bytes: Vec<u8> = vec![0u8; 8];
        let result = convert_to_f32(&bytes, safetensors::Dtype::F32, &[3], "f32_short");
        assert!(
            matches!(
                result,
                Err(ReftestError::DataLengthMismatch {
                    expected: 12,
                    actual: 8
                })
            ),
            "expected DataLengthMismatch, got {result:?}",
        );
    }

    #[test]
    fn test_convert_f64_data_length_mismatch() {
        // 2 elements need 16 bytes, provide only 8.
        let bytes: Vec<u8> = vec![0u8; 8];
        let result = convert_to_f32(&bytes, safetensors::Dtype::F64, &[2], "f64_short");
        assert!(
            matches!(
                result,
                Err(ReftestError::DataLengthMismatch {
                    expected: 16,
                    actual: 8
                })
            ),
            "expected DataLengthMismatch, got {result:?}",
        );
    }

    #[test]
    fn test_convert_f16_data_length_mismatch() {
        // 3 elements need 6 bytes, provide only 4.
        let bytes: Vec<u8> = vec![0u8; 4];
        let result = convert_to_f32(&bytes, safetensors::Dtype::F16, &[3], "f16_short");
        assert!(
            matches!(
                result,
                Err(ReftestError::DataLengthMismatch {
                    expected: 6,
                    actual: 4
                })
            ),
            "expected DataLengthMismatch, got {result:?}",
        );
    }

    #[test]
    fn test_convert_bf16_data_length_mismatch() {
        let bytes: Vec<u8> = vec![0u8; 2];
        let result = convert_to_f32(&bytes, safetensors::Dtype::BF16, &[3], "bf16_short");
        assert!(
            matches!(
                result,
                Err(ReftestError::DataLengthMismatch {
                    expected: 6,
                    actual: 2
                })
            ),
            "expected DataLengthMismatch, got {result:?}",
        );
    }

    #[test]
    fn test_convert_unsupported_dtype_returns_error() {
        let result = convert_to_f32(&[], safetensors::Dtype::BOOL, &[1], "bool_test");
        assert!(
            matches!(result, Err(ReftestError::UnsupportedDtype(_))),
            "expected UnsupportedDtype, got {result:?}",
        );
    }

    #[test]
    fn test_convert_f64_rejects_neg_infinity() {
        let bytes: Vec<u8> = f64::NEG_INFINITY.to_le_bytes().to_vec();
        let result = convert_to_f32(&bytes, safetensors::Dtype::F64, &[1], "neginf_test");
        assert!(
            matches!(result, Err(ReftestError::F64OutOfF32Range { .. })),
            "expected F64OutOfF32Range for -inf, got {result:?}",
        );
    }

    #[test]
    fn test_convert_shape_product_overflow() {
        let result = convert_to_f32(
            &[],
            safetensors::Dtype::F32,
            &[usize::MAX, 2],
            "overflow_test",
        );
        assert!(
            matches!(result, Err(ReftestError::ShapeProductOverflow(_))),
            "expected ShapeProductOverflow, got {result:?}",
        );
    }

    #[test]
    fn test_convert_empty_tensor_succeeds() {
        let result = convert_to_f32(&[], safetensors::Dtype::F32, &[0], "empty");
        let data = result.expect("empty tensor should succeed");
        assert!(data.is_empty());
    }

    #[test]
    fn test_load_safetensors_from_bytes_invalid_data() {
        let result = load_safetensors_from_bytes(b"not a valid safetensors file");
        assert!(
            matches!(result, Err(ReftestError::Safetensors(_))),
            "expected Safetensors parse error, got {result:?}",
        );
    }

    #[test]
    fn test_load_safetensors_from_bytes_empty_data() {
        let result = load_safetensors_from_bytes(&[]);
        assert!(
            matches!(result, Err(ReftestError::Safetensors(_))),
            "expected Safetensors parse error for empty data, got {result:?}",
        );
    }

    #[test]
    fn test_load_safetensors_nonexistent_file() {
        let result = load_safetensors("/nonexistent/path/to/file.safetensors");
        assert!(
            matches!(result, Err(ReftestError::Io(_))),
            "expected Io error, got {result:?}",
        );
    }

    #[test]
    fn test_load_safetensors_multidimensional() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let byte_data = f32_to_le_bytes(&data);
        let view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            vec![2, 3, 4],
            &byte_data,
        )
        .expect("valid view");
        let serialized = safetensors::tensor::serialize(vec![("weights".to_string(), view)], None)
            .expect("serialization should succeed");

        let trace = load_safetensors_from_bytes(&serialized).expect("load should succeed");
        assert_eq!(trace.len(), 1);
        let t = trace.get(0).expect("exists");
        assert_eq!(t.name, "weights");
        assert_eq!(t.shape, vec![2, 3, 4]);
        assert_eq!(t.numel(), 24);
    }

    #[test]
    fn test_load_safetensors_sorts_alphabetically() {
        let bytes = build_safetensors(&[
            ("z_layer", &[1], &[3.0]),
            ("a_layer", &[1], &[1.0]),
            ("m_layer", &[1], &[2.0]),
        ]);

        let trace = load_safetensors_from_bytes(&bytes).expect("load should succeed");
        let names: Vec<&str> = trace.names().collect();
        assert_eq!(names, vec!["a_layer", "m_layer", "z_layer"]);
    }
}
