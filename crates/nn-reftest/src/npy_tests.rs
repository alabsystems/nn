// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the NPY loader module.

use super::*;

/// Build a minimal NPY v1.0 byte buffer.
fn build_npy_v1(dtype: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
    let shape_str = if shape.is_empty() {
        "()".to_string()
    } else if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
        format!("({})", dims.join(", "))
    };

    let header = format!(
        "{{'descr': '{dtype}', 'fortran_order': False, 'shape': {shape_str}, }}",
    );

    // Pad header to 64-byte alignment (magic + version + header_len + header).
    let prefix_len = 10; // 6 magic + 2 version + 2 header_len
    let total_header = header.len() + 1; // +1 for newline
    let padded_len = (prefix_len + total_header).div_ceil(64) * 64 - prefix_len;
    let padding = padded_len - header.len() - 1;

    let mut buf = Vec::new();
    buf.extend_from_slice(NPY_MAGIC);
    buf.push(1); // major
    buf.push(0); // minor
    let header_len = (padded_len) as u16;
    buf.extend_from_slice(&header_len.to_le_bytes());
    buf.extend_from_slice(header.as_bytes());
    buf.extend(std::iter::repeat_n(b' ', padding));
    buf.push(b'\n');
    buf.extend_from_slice(data);
    buf
}

#[test]
fn test_parse_npy_f32_le() {
    let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v1("<f4", &[2, 3], &raw);

    let tensor = parse_npy(&npy, "test".into()).expect("parse should succeed");
    assert_eq!(tensor.name, "test");
    assert_eq!(tensor.shape, vec![2, 3]);
    assert_eq!(tensor.data, values);
}

#[test]
fn test_parse_npy_f64_le() {
    let values: Vec<f64> = vec![1.5, -2.5];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v1("<f8", &[2], &raw);

    let tensor = parse_npy(&npy, "f64test".into()).expect("parse should succeed");
    assert_eq!(tensor.shape, vec![2]);
    assert!((tensor.data[0] - 1.5).abs() < f32::EPSILON);
    assert!((tensor.data[1] - (-2.5)).abs() < f32::EPSILON);
}

#[test]
fn test_parse_npy_f16_le() {
    let values: Vec<half::f16> = vec![half::f16::from_f32(1.0), half::f16::from_f32(0.5)];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v1("<f2", &[2], &raw);

    let tensor = parse_npy(&npy, "f16test".into()).expect("parse should succeed");
    assert_eq!(tensor.shape, vec![2]);
    assert!((tensor.data[0] - 1.0).abs() < 0.01);
    assert!((tensor.data[1] - 0.5).abs() < 0.01);
}

#[test]
fn test_parse_npy_scalar() {
    let raw: Vec<u8> = 42.0f32.to_le_bytes().to_vec();
    let npy = build_npy_v1("<f4", &[], &raw);

    let tensor = parse_npy(&npy, "scalar".into()).expect("parse should succeed");
    assert!(tensor.shape.is_empty());
    assert_eq!(tensor.data.len(), 1);
    assert!((tensor.data[0] - 42.0).abs() < f32::EPSILON);
}

#[test]
fn test_parse_npy_1d() {
    let values: Vec<f32> = vec![10.0, 20.0, 30.0];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v1("<f4", &[3], &raw);

    let tensor = parse_npy(&npy, "1d".into()).expect("parse should succeed");
    assert_eq!(tensor.shape, vec![3]);
    assert_eq!(tensor.data, values);
}

#[test]
fn test_parse_npy_i32_le() {
    let values: Vec<i32> = vec![1, -2, 3];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v1("<i4", &[3], &raw);

    let tensor = parse_npy(&npy, "i32test".into()).expect("parse should succeed");
    assert_eq!(tensor.data, vec![1.0, -2.0, 3.0]);
}

#[test]
fn test_parse_npy_bad_magic() {
    let result = parse_npy(b"BADDATA", "bad".into());
    assert!(matches!(result, Err(ReftestError::NpyBadMagic)));
}

#[test]
fn test_parse_npy_unsupported_version() {
    let mut data = NPY_MAGIC.to_vec();
    data.push(3); // major
    data.push(0); // minor
    data.extend_from_slice(&[0, 0]); // header_len
    let result = parse_npy(&data, "bad".into());
    assert!(matches!(
        result,
        Err(ReftestError::NpyUnsupportedVersion { major: 3, minor: 0 })
    ));
}

#[test]
fn test_parse_npy_fortran_order_rejected() {
    // Build header with fortran_order: True
    let header = "{'descr': '<f4', 'fortran_order': True, 'shape': (2,), }";
    let prefix_len = 10;
    let total_header = header.len() + 1;
    let padded_len = (prefix_len + total_header).div_ceil(64) * 64 - prefix_len;
    let padding = padded_len - header.len() - 1;

    let mut buf = Vec::new();
    buf.extend_from_slice(NPY_MAGIC);
    buf.push(1);
    buf.push(0);
    let header_len = padded_len as u16;
    buf.extend_from_slice(&header_len.to_le_bytes());
    buf.extend_from_slice(header.as_bytes());
    buf.extend(std::iter::repeat_n(b' ', padding));
    buf.push(b'\n');
    buf.extend_from_slice(&1.0f32.to_le_bytes());
    buf.extend_from_slice(&2.0f32.to_le_bytes());

    let result = parse_npy(&buf, "fortran".into());
    assert!(matches!(result, Err(ReftestError::NpyFortranOrder)));
}

#[test]
fn test_parse_npy_f32_be() {
    let values: Vec<f32> = vec![1.0, 2.0];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();
    let npy = build_npy_v1(">f4", &[2], &raw);

    let tensor = parse_npy(&npy, "be".into()).expect("parse should succeed");
    assert_eq!(tensor.data, values);
}

#[test]
fn test_parse_npy_u8() {
    let raw: Vec<u8> = vec![0, 128, 255];
    let npy = build_npy_v1("|u1", &[3], &raw);

    let tensor = parse_npy(&npy, "u8test".into()).expect("parse should succeed");
    assert_eq!(tensor.data, vec![0.0, 128.0, 255.0]);
}

#[test]
fn test_load_npy_from_bytes() {
    let values: Vec<f32> = vec![1.0, 2.0, 3.0];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v1("<f4", &[3], &raw);

    let trace = load_npy_from_bytes(&npy, "nn_tensor").expect("load should succeed");
    assert_eq!(trace.len(), 1);
    assert_eq!(trace.get(0).expect("exists").name, "nn_tensor");
    assert_eq!(trace.get(0).expect("exists").data, values);
}

#[test]
fn test_load_npy_dir_sorting() {
    let dir = std::env::temp_dir().join(format!("nn_reftest_npy_dir_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create dir");

    // Write files out of alphabetical order to test sorting.
    for (name, val) in &[("c_layer", 3.0f32), ("a_layer", 1.0), ("b_layer", 2.0)] {
        let raw: Vec<u8> = val.to_le_bytes().to_vec();
        let npy = build_npy_v1("<f4", &[1], &raw);
        std::fs::write(dir.join(format!("{name}.npy")), npy).expect("write");
    }

    // Add a non-.npy file that should be skipped.
    std::fs::write(dir.join("readme.txt"), "not a tensor").expect("write");

    let trace = load_npy_dir(&dir).expect("load dir should succeed");
    assert_eq!(trace.len(), 3);

    let names: Vec<&str> = trace.names().collect();
    assert_eq!(names, vec!["a_layer", "b_layer", "c_layer"]);

    // Verify values are correct.
    assert!((trace.get(0).expect("a").data[0] - 1.0).abs() < f32::EPSILON);
    assert!((trace.get(1).expect("b").data[0] - 2.0).abs() < f32::EPSILON);
    assert!((trace.get(2).expect("c").data[0] - 3.0).abs() < f32::EPSILON);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_convert_i32_precision_loss_rejected() {
    // 2^24 + 1 = 16,777,217 — cannot be represented exactly in f32.
    let values: Vec<i32> = vec![1, 16_777_217];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v1("<i4", &[2], &raw);

    let result = parse_npy(&npy, "i32_big".into());
    assert!(
        matches!(
            result,
            Err(ReftestError::IntPrecisionLoss {
                value: 16_777_217,
                index: 1
            })
        ),
        "i32 value > 2^24 should be rejected, got: {result:?}"
    );
}

#[test]
fn test_convert_i64_precision_loss_rejected() {
    // Large i64 value that cannot be exactly represented in f32.
    let values: Vec<i64> = vec![100, 20_000_000];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v1("<i8", &[2], &raw);

    let result = parse_npy(&npy, "i64_big".into());
    assert!(
        matches!(
            result,
            Err(ReftestError::IntPrecisionLoss {
                value: 20_000_000,
                index: 1
            })
        ),
        "i64 value > 2^24 should be rejected, got: {result:?}"
    );
}

#[test]
fn test_convert_i32_within_precision_limit_ok() {
    // Values within ±2^24 should convert exactly.
    let values: Vec<i32> = vec![0, -1, 16_777_216, -16_777_216];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v1("<i4", &[4], &raw);

    let tensor = parse_npy(&npy, "i32_ok".into()).expect("values within 2^24 should succeed");
    assert_eq!(tensor.data, vec![0.0, -1.0, 16_777_216.0, -16_777_216.0]);
}

#[test]
fn test_convert_i64_within_precision_limit_ok() {
    let values: Vec<i64> = vec![0, -1, 16_777_216, -16_777_216];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v1("<i8", &[4], &raw);

    let tensor = parse_npy(&npy, "i64_ok".into()).expect("values within 2^24 should succeed");
    assert_eq!(tensor.data, vec![0.0, -1.0, 16_777_216.0, -16_777_216.0]);
}

// ---- NPY v2.0 format ----

/// Build a minimal NPY v2.0 byte buffer (4-byte header length field).
fn build_npy_v2(dtype: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
    let shape_str = if shape.is_empty() {
        "()".to_string()
    } else if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
        format!("({})", dims.join(", "))
    };

    let header = format!(
        "{{'descr': '{dtype}', 'fortran_order': False, 'shape': {shape_str}, }}",
    );

    // Pad header to 64-byte alignment (magic + version + header_len + header).
    let prefix_len = 12; // 6 magic + 2 version + 4 header_len (v2)
    let total_header = header.len() + 1; // +1 for newline
    let padded_len = (prefix_len + total_header).div_ceil(64) * 64 - prefix_len;
    let padding = padded_len - header.len() - 1;

    let mut buf = Vec::new();
    buf.extend_from_slice(NPY_MAGIC);
    buf.push(2); // major
    buf.push(0); // minor
    let header_len = padded_len as u32;
    buf.extend_from_slice(&header_len.to_le_bytes());
    buf.extend_from_slice(header.as_bytes());
    buf.extend(std::iter::repeat_n(b' ', padding));
    buf.push(b'\n');
    buf.extend_from_slice(data);
    buf
}

#[test]
fn test_parse_npy_v2_f32() {
    let values: Vec<f32> = vec![10.0, 20.0, 30.0];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v2("<f4", &[3], &raw);

    let tensor = parse_npy(&npy, "v2_test".into()).expect("v2 parse should succeed");
    assert_eq!(tensor.name, "v2_test");
    assert_eq!(tensor.shape, vec![3]);
    assert_eq!(tensor.data, values);
}

#[test]
fn test_parse_npy_v2_2d() {
    let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v2("<f4", &[2, 3], &raw);

    let tensor = parse_npy(&npy, "v2_2d".into()).expect("v2 2d parse should succeed");
    assert_eq!(tensor.shape, vec![2, 3]);
    assert_eq!(tensor.data, values);
}

// ---- Header parsing edge cases ----

#[test]
fn test_extract_string_value_double_quotes() {
    let header = "'descr': \"<f4\", 'fortran_order': False, 'shape': (3,)";
    let result = extract_string_value(header, "descr");
    assert_eq!(result, Some("<f4".to_string()));
}

#[test]
fn test_extract_string_value_missing_key() {
    let header = "'fortran_order': False, 'shape': (3,)";
    let result = extract_string_value(header, "descr");
    assert!(result.is_none());
}

#[test]
fn test_extract_bool_value_true() {
    let header = "'fortran_order': True";
    let result = extract_bool_value(header, "fortran_order");
    assert_eq!(result, Some(true));
}

#[test]
fn test_extract_bool_value_false() {
    let header = "'fortran_order': False";
    let result = extract_bool_value(header, "fortran_order");
    assert_eq!(result, Some(false));
}

#[test]
fn test_extract_bool_value_missing() {
    let header = "'descr': '<f4'";
    let result = extract_bool_value(header, "fortran_order");
    assert!(result.is_none());
}

#[test]
fn test_extract_shape_scalar() {
    let header = "'descr': '<f4', 'fortran_order': False, 'shape': ()";
    let result = extract_shape(header);
    assert_eq!(result, Some(vec![]));
}

#[test]
fn test_extract_shape_1d() {
    let header = "'descr': '<f4', 'fortran_order': False, 'shape': (5,)";
    let result = extract_shape(header);
    assert_eq!(result, Some(vec![5]));
}

#[test]
fn test_extract_shape_3d() {
    let header = "'descr': '<f4', 'fortran_order': False, 'shape': (2, 3, 4)";
    let result = extract_shape(header);
    assert_eq!(result, Some(vec![2, 3, 4]));
}

#[test]
fn test_extract_shape_missing() {
    let header = "'descr': '<f4', 'fortran_order': False";
    let result = extract_shape(header);
    assert!(result.is_none());
}

#[test]
fn test_parse_npy_header_full() {
    let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }";
    let (dtype, shape, fortran) = parse_npy_header(header).expect("parse should succeed");
    assert_eq!(dtype, "<f4");
    assert_eq!(shape, vec![2, 3]);
    assert!(!fortran);
}

#[test]
fn test_parse_npy_header_fortran_true() {
    let header = "{'descr': '<f4', 'fortran_order': True, 'shape': (4,), }";
    let (_, _, fortran) = parse_npy_header(header).expect("parse should succeed");
    assert!(fortran);
}

#[test]
fn test_parse_npy_header_missing_descr() {
    let header = "{'fortran_order': False, 'shape': (2,), }";
    let result = parse_npy_header(header);
    assert!(matches!(result, Err(ReftestError::NpyHeaderParse(_))));
}

#[test]
fn test_parse_npy_header_missing_shape() {
    let header = "{'descr': '<f4', 'fortran_order': False, }";
    let result = parse_npy_header(header);
    assert!(matches!(result, Err(ReftestError::NpyHeaderParse(_))));
}

// ---- Truncated file handling ----

#[test]
fn test_parse_npy_truncated_before_header() {
    // Only magic + version, no header length or data.
    let mut buf = NPY_MAGIC.to_vec();
    buf.push(1);
    buf.push(0);
    // Missing header_len bytes.
    let result = parse_npy(&buf, "trunc".into());
    // Should fail due to truncation (header_len claims more than available).
    assert!(result.is_err());
}

#[test]
fn test_parse_npy_v2_truncated_header_len() {
    // v2 needs 12 bytes minimum but we provide only 10.
    let mut buf = NPY_MAGIC.to_vec();
    buf.push(2);
    buf.push(0);
    buf.extend_from_slice(&[0, 0]); // only 2 of 4 header_len bytes
    let result = parse_npy(&buf, "trunc_v2".into());
    assert!(matches!(result, Err(ReftestError::NpyBadMagic)));
}

#[test]
fn test_parse_npy_too_short() {
    // Less than 10 bytes total.
    let result = parse_npy(b"\x93NUMPY", "short".into());
    assert!(matches!(result, Err(ReftestError::NpyBadMagic)));
}

// ---- Additional dtype conversions ----

#[test]
fn test_parse_npy_i16_le() {
    let values: Vec<i16> = vec![100, -200, 32767, -32768];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v1("<i2", &[4], &raw);

    let tensor = parse_npy(&npy, "i16_test".into()).expect("i16 parse should succeed");
    assert_eq!(tensor.data, vec![100.0, -200.0, 32767.0, -32768.0]);
}

#[test]
fn test_parse_npy_i8() {
    let raw: Vec<u8> = vec![0u8, 127, 128, 255]; // as i8: 0, 127, -128, -1
    let npy = build_npy_v1("|i1", &[4], &raw);

    let tensor = parse_npy(&npy, "i8_test".into()).expect("i8 parse should succeed");
    assert_eq!(tensor.data, vec![0.0, 127.0, -128.0, -1.0]);
}

#[test]
fn test_parse_npy_f64_be() {
    let values: Vec<f64> = vec![1.25, -3.75];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();
    let npy = build_npy_v1(">f8", &[2], &raw);

    let tensor = parse_npy(&npy, "f64_be".into()).expect("f64 BE parse should succeed");
    assert!((tensor.data[0] - 1.25).abs() < f32::EPSILON);
    assert!((tensor.data[1] - (-3.75)).abs() < f32::EPSILON);
}

#[test]
fn test_parse_npy_f16_be() {
    let values: Vec<half::f16> = vec![half::f16::from_f32(2.0), half::f16::from_f32(-1.0)];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();
    let npy = build_npy_v1(">f2", &[2], &raw);

    let tensor = parse_npy(&npy, "f16_be".into()).expect("f16 BE parse should succeed");
    assert!((tensor.data[0] - 2.0).abs() < 0.01);
    assert!((tensor.data[1] - (-1.0)).abs() < 0.01);
}

#[test]
fn test_parse_npy_unsupported_dtype() {
    let npy = build_npy_v1("<c8", &[1], &[0u8; 8]); // complex64 not supported
    let result = parse_npy(&npy, "complex".into());
    assert!(
        matches!(result, Err(ReftestError::NpyUnsupportedDtype(_))),
        "expected NpyUnsupportedDtype, got {result:?}",
    );
}

#[test]
fn test_parse_npy_data_truncated() {
    // Shape says 3 f32 elements (12 bytes) but provide only 8.
    let raw: Vec<u8> = vec![0u8; 8];
    let npy = build_npy_v1("<f4", &[3], &raw);

    let result = parse_npy(&npy, "trunc_data".into());
    assert!(
        matches!(result, Err(ReftestError::DataLengthMismatch { .. })),
        "expected DataLengthMismatch for truncated data, got {result:?}",
    );
}

#[test]
fn test_parse_npy_shape_overflow() {
    // Shape with values that overflow when multiplied.
    let npy = build_npy_v1("<f4", &[usize::MAX, 2], &[]);
    let result = parse_npy(&npy, "shape_overflow".into());
    assert!(
        matches!(result, Err(ReftestError::ShapeProductOverflow(_))),
        "expected ShapeProductOverflow, got {result:?}",
    );
}

#[test]
fn test_load_npy_nonexistent_file() {
    let result = load_npy("/nonexistent/path/to/file.npy");
    assert!(
        matches!(result, Err(ReftestError::Io(_))),
        "expected Io error, got {result:?}",
    );
}

#[test]
fn test_load_npy_dir_nonexistent() {
    let result = load_npy_dir("/nonexistent/dir/");
    assert!(
        matches!(result, Err(ReftestError::Io(_))),
        "expected Io error for nonexistent dir, got {result:?}",
    );
}

#[test]
fn test_load_npy_dir_empty() {
    let dir = std::env::temp_dir().join(format!("nn_reftest_empty_dir_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create dir");

    let trace = load_npy_dir(&dir).expect("empty dir should produce empty trace");
    assert!(trace.is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_load_npy_from_bytes_preserves_name() {
    let values: Vec<f32> = vec![42.0];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v1("<f4", &[1], &raw);

    let trace = load_npy_from_bytes(&npy, "custom_name").expect("load should succeed");
    assert_eq!(trace.get(0).expect("exists").name, "custom_name");
}

// ===========================================================================
// NpyTensor / NpyDType / write_npy / read_npy tests
// ===========================================================================

use super::{
    read_npy, read_npy_from_bytes, write_npy, write_npy_to_bytes, NpyDType, NpyError, NpyTensor,
};

// ---- NpyDType ----

#[test]
fn test_npy_dtype_from_descr_f32() {
    assert_eq!(NpyDType::from_descr("<f4"), Some(NpyDType::F32));
    assert_eq!(NpyDType::from_descr(">f4"), Some(NpyDType::F32));
}

#[test]
fn test_npy_dtype_from_descr_f64() {
    assert_eq!(NpyDType::from_descr("<f8"), Some(NpyDType::F64));
    assert_eq!(NpyDType::from_descr(">f8"), Some(NpyDType::F64));
}

#[test]
fn test_npy_dtype_from_descr_f16() {
    assert_eq!(NpyDType::from_descr("<f2"), Some(NpyDType::F16));
    assert_eq!(NpyDType::from_descr(">f2"), Some(NpyDType::F16));
}

#[test]
fn test_npy_dtype_from_descr_i32() {
    assert_eq!(NpyDType::from_descr("<i4"), Some(NpyDType::I32));
    assert_eq!(NpyDType::from_descr("=i4"), Some(NpyDType::I32));
}

#[test]
fn test_npy_dtype_from_descr_i64() {
    assert_eq!(NpyDType::from_descr("<i8"), Some(NpyDType::I64));
}

#[test]
fn test_npy_dtype_from_descr_u8() {
    assert_eq!(NpyDType::from_descr("|u1"), Some(NpyDType::U8));
    assert_eq!(NpyDType::from_descr("<u1"), Some(NpyDType::U8));
}

#[test]
fn test_npy_dtype_from_descr_unknown() {
    assert_eq!(NpyDType::from_descr("<c8"), None);
    assert_eq!(NpyDType::from_descr("xyz"), None);
}

#[test]
fn test_npy_dtype_to_descr_roundtrip() {
    for dtype in [
        NpyDType::F16,
        NpyDType::F32,
        NpyDType::F64,
        NpyDType::I32,
        NpyDType::I64,
        NpyDType::U8,
    ] {
        let descr = dtype.to_descr();
        assert_eq!(
            NpyDType::from_descr(descr),
            Some(dtype),
            "roundtrip failed for {descr}"
        );
    }
}

#[test]
fn test_npy_dtype_display() {
    assert_eq!(format!("{}", NpyDType::F32), "<f4");
    assert_eq!(format!("{}", NpyDType::F64), "<f8");
    assert_eq!(format!("{}", NpyDType::U8), "|u1");
}

// ---- write_npy_to_bytes ----

#[test]
fn test_write_npy_to_bytes_1d() {
    let data = vec![1.0f32, 2.0, 3.0];
    let bytes = write_npy_to_bytes(&data, &[3]).expect("write should succeed");
    // Verify the output starts with NPY magic.
    assert_eq!(&bytes[..6], NPY_MAGIC);
    // Version 1.0.
    assert_eq!(bytes[6], 1);
    assert_eq!(bytes[7], 0);
}

#[test]
fn test_write_npy_to_bytes_scalar() {
    let data = vec![42.0f32];
    let bytes = write_npy_to_bytes(&data, &[]).expect("write should succeed");
    assert_eq!(&bytes[..6], NPY_MAGIC);
}

#[test]
fn test_write_npy_to_bytes_2d() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bytes = write_npy_to_bytes(&data, &[2, 3]).expect("write should succeed");
    assert_eq!(&bytes[..6], NPY_MAGIC);
}

#[test]
fn test_write_npy_to_bytes_3d() {
    let data = vec![0.0f32; 2 * 3 * 4];
    let bytes = write_npy_to_bytes(&data, &[2, 3, 4]).expect("write should succeed");
    assert_eq!(&bytes[..6], NPY_MAGIC);
}

#[test]
fn test_write_npy_to_bytes_shape_mismatch() {
    let data = vec![1.0f32, 2.0];
    let result = write_npy_to_bytes(&data, &[3]);
    assert!(
        matches!(
            result,
            Err(NpyError::DataLengthMismatch {
                expected: 3,
                actual: 2,
                ..
            })
        ),
        "expected DataLengthMismatch, got {result:?}",
    );
}

#[test]
fn test_write_npy_to_bytes_shape_overflow() {
    let data = vec![];
    let result = write_npy_to_bytes(&data, &[usize::MAX, 2]);
    assert!(
        matches!(result, Err(NpyError::ShapeOverflow(_))),
        "expected ShapeOverflow, got {result:?}",
    );
}

#[test]
fn test_write_npy_to_bytes_empty_tensor() {
    // Shape [0] means 0 elements.
    let data: Vec<f32> = vec![];
    let bytes = write_npy_to_bytes(&data, &[0]).expect("write should succeed");
    assert_eq!(&bytes[..6], NPY_MAGIC);
}

// ---- Round-trip: write_npy_to_bytes -> read_npy_from_bytes ----

#[test]
fn test_roundtrip_1d() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let shape = vec![5];
    let bytes = write_npy_to_bytes(&data, &shape).expect("write");
    let tensor = read_npy_from_bytes(&bytes).expect("read");
    assert_eq!(tensor.data, data);
    assert_eq!(tensor.shape, shape);
    assert_eq!(tensor.dtype, NpyDType::F32);
}

#[test]
fn test_roundtrip_2d() {
    let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
    let shape = vec![3, 4];
    let bytes = write_npy_to_bytes(&data, &shape).expect("write");
    let tensor = read_npy_from_bytes(&bytes).expect("read");
    assert_eq!(tensor.data, data);
    assert_eq!(tensor.shape, shape);
}

#[test]
fn test_roundtrip_3d() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let shape = vec![2, 3, 4];
    let bytes = write_npy_to_bytes(&data, &shape).expect("write");
    let tensor = read_npy_from_bytes(&bytes).expect("read");
    assert_eq!(tensor.data, data);
    assert_eq!(tensor.shape, shape);
}

#[test]
fn test_roundtrip_scalar() {
    let data = vec![42.0f32];
    let shape: Vec<usize> = vec![];
    let bytes = write_npy_to_bytes(&data, &shape).expect("write");
    let tensor = read_npy_from_bytes(&bytes).expect("read");
    assert_eq!(tensor.data, data);
    assert!(tensor.shape.is_empty());
}

#[test]
fn test_roundtrip_special_values() {
    let data = vec![0.0f32, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
    let shape = vec![5];
    let bytes = write_npy_to_bytes(&data, &shape).expect("write");
    let tensor = read_npy_from_bytes(&bytes).expect("read");
    // Compare element-by-element because NaN != NaN.
    assert_eq!(tensor.shape, shape);
    assert_eq!(tensor.data.len(), 5);
    assert_eq!(tensor.data[0], 0.0);
    assert!(tensor.data[1].to_bits() == (-0.0f32).to_bits()); // negative zero
    assert_eq!(tensor.data[2], f32::INFINITY);
    assert_eq!(tensor.data[3], f32::NEG_INFINITY);
    assert!(tensor.data[4].is_nan());
}

#[test]
fn test_roundtrip_large_shape() {
    // Large but not overflow: [100, 100] = 10,000 elements.
    let data: Vec<f32> = (0..10_000).map(|i| i as f32).collect();
    let shape = vec![100, 100];
    let bytes = write_npy_to_bytes(&data, &shape).expect("write");
    let tensor = read_npy_from_bytes(&bytes).expect("read");
    assert_eq!(tensor.data.len(), 10_000);
    assert_eq!(tensor.shape, shape);
    assert_eq!(tensor.data[0], 0.0);
    assert_eq!(tensor.data[9999], 9999.0);
}

// ---- Round-trip through files ----

#[test]
fn test_roundtrip_file() {
    let dir = std::env::temp_dir().join(format!("nn_reftest_npy_write_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create dir");

    let path = dir.join("test_tensor.npy");
    let data = vec![1.5f32, -2.5, 3.0, 0.0, 100.0, -100.0];
    let shape = vec![2, 3];

    write_npy(&path, &data, &shape).expect("write_npy should succeed");
    let tensor = read_npy(&path).expect("read_npy should succeed");

    assert_eq!(tensor.data, data);
    assert_eq!(tensor.shape, shape);
    assert_eq!(tensor.dtype, NpyDType::F32);

    let _ = std::fs::remove_dir_all(&dir);
}

// ---- read_npy_from_bytes error cases ----

#[test]
fn test_read_npy_from_bytes_bad_magic() {
    let result = read_npy_from_bytes(b"NOTANPY!");
    assert!(matches!(result, Err(NpyError::BadMagic)));
}

#[test]
fn test_read_npy_from_bytes_too_short() {
    let result = read_npy_from_bytes(b"\x93NUM");
    assert!(matches!(result, Err(NpyError::BadMagic)));
}

#[test]
fn test_read_npy_from_bytes_unsupported_version() {
    let mut data = NPY_MAGIC.to_vec();
    data.push(3); // major
    data.push(0); // minor
    data.extend_from_slice(&[0, 0]); // header_len
    let result = read_npy_from_bytes(&data);
    assert!(
        matches!(
            result,
            Err(NpyError::UnsupportedVersion { major: 3, minor: 0 })
        ),
        "expected UnsupportedVersion, got {result:?}",
    );
}

#[test]
fn test_read_npy_from_bytes_fortran_order() {
    let header = "{'descr': '<f4', 'fortran_order': True, 'shape': (2,), }";
    let prefix_len = 10;
    let total_header = header.len() + 1;
    let padded_len = (prefix_len + total_header).div_ceil(64) * 64 - prefix_len;
    let padding = padded_len - header.len() - 1;

    let mut buf = Vec::new();
    buf.extend_from_slice(NPY_MAGIC);
    buf.push(1);
    buf.push(0);
    let header_len = padded_len as u16;
    buf.extend_from_slice(&header_len.to_le_bytes());
    buf.extend_from_slice(header.as_bytes());
    buf.extend(std::iter::repeat_n(b' ', padding));
    buf.push(b'\n');
    buf.extend_from_slice(&1.0f32.to_le_bytes());
    buf.extend_from_slice(&2.0f32.to_le_bytes());

    let result = read_npy_from_bytes(&buf);
    assert!(matches!(result, Err(NpyError::FortranOrder)));
}

#[test]
fn test_read_npy_from_bytes_v2_format() {
    let values: Vec<f32> = vec![10.0, 20.0, 30.0];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v2("<f4", &[3], &raw);

    let tensor = read_npy_from_bytes(&npy).expect("v2 read should succeed");
    assert_eq!(tensor.data, values);
    assert_eq!(tensor.shape, vec![3]);
    assert_eq!(tensor.dtype, NpyDType::F32);
}

#[test]
fn test_read_npy_from_bytes_f64_dtype() {
    let values: Vec<f64> = vec![1.5, -2.5];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v1("<f8", &[2], &raw);

    let tensor = read_npy_from_bytes(&npy).expect("f64 read should succeed");
    assert_eq!(tensor.dtype, NpyDType::F64);
    assert!((tensor.data[0] - 1.5).abs() < f32::EPSILON);
    assert!((tensor.data[1] - (-2.5)).abs() < f32::EPSILON);
}

#[test]
fn test_read_npy_from_bytes_f16_dtype() {
    let values: Vec<half::f16> = vec![half::f16::from_f32(1.0), half::f16::from_f32(0.5)];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v1("<f2", &[2], &raw);

    let tensor = read_npy_from_bytes(&npy).expect("f16 read should succeed");
    assert_eq!(tensor.dtype, NpyDType::F16);
    assert!((tensor.data[0] - 1.0).abs() < 0.01);
    assert!((tensor.data[1] - 0.5).abs() < 0.01);
}

// ---- NpyTensor methods ----

#[test]
fn test_npy_tensor_numel() {
    let tensor = NpyTensor {
        data: vec![1.0, 2.0, 3.0],
        shape: vec![3],
        dtype: NpyDType::F32,
    };
    assert_eq!(tensor.numel(), 3);
}

#[test]
fn test_npy_tensor_numel_scalar() {
    let tensor = NpyTensor {
        data: vec![42.0],
        shape: vec![],
        dtype: NpyDType::F32,
    };
    assert_eq!(tensor.numel(), 1);
}

// ---- read_npy file error ----

#[test]
fn test_read_npy_nonexistent() {
    let result = read_npy("/nonexistent/path/test.npy");
    assert!(matches!(result, Err(NpyError::Io(_))));
}

// ---- Compatibility: read_npy_from_bytes reads what write_npy_to_bytes produces for parse_npy too ----

#[test]
fn test_write_then_parse_npy_compat() {
    let data = vec![1.0f32, 2.0, 3.0];
    let shape = vec![3];
    let bytes = write_npy_to_bytes(&data, &shape).expect("write");
    // The existing parse_npy (used by load_npy) should also read it.
    let named = parse_npy(&bytes, "compat".into()).expect("parse_npy should read written data");
    assert_eq!(named.data, data);
    assert_eq!(named.shape, shape);
}
