// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for nn-reftest reference tensor comparison infrastructure.
//!
//! Covers: safetensors loading (multi-dtype), NPY loading, tolerance assertions,
//! shape/dtype mismatch detection, trace comparison, and statistics reporting.

use nn_reftest::{
    assert_traces_match, compare_tensors, compare_traces, load_npy_from_bytes,
    load_safetensors_from_bytes, ComparisonConfig, NamedTensor, ReferenceTrace, ReftestError,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert f32 slice to little-endian bytes.
fn f32_to_le_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Build a minimal safetensors byte buffer from typed tensor descriptors.
fn build_safetensors_typed(tensors: &[(&str, safetensors::Dtype, &[usize], &[u8])]) -> Vec<u8> {
    let mut tensor_map: Vec<(String, safetensors::tensor::TensorView<'_>)> = Vec::new();
    for &(name, dtype, shape, data) in tensors {
        let view = safetensors::tensor::TensorView::new(dtype, shape.to_vec(), data)
            .expect("valid tensor view");
        tensor_map.push((name.to_string(), view));
    }
    safetensors::tensor::serialize(tensor_map, None).expect("serialization should succeed")
}

/// Build a minimal safetensors byte buffer from f32 tensors.
fn build_safetensors_f32(tensors: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
    let byte_bufs: Vec<Vec<u8>> = tensors
        .iter()
        .map(|&(_, _, data)| f32_to_le_bytes(data))
        .collect();
    let entries: Vec<(&str, safetensors::Dtype, &[usize], &[u8])> = tensors
        .iter()
        .enumerate()
        .map(|(i, &(name, shape, _))| {
            (
                name,
                safetensors::Dtype::F32,
                shape,
                byte_bufs[i].as_slice(),
            )
        })
        .collect();
    build_safetensors_typed(&entries)
}

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

    let prefix_len = 10;
    let total_header = header.len() + 1;
    let padded_len = (prefix_len + total_header).div_ceil(64) * 64 - prefix_len;
    let padding = padded_len - header.len() - 1;

    let mut buf = Vec::new();
    buf.extend_from_slice(b"\x93NUMPY");
    buf.push(1); // major
    buf.push(0); // minor
    let header_len = padded_len as u16;
    buf.extend_from_slice(&header_len.to_le_bytes());
    buf.extend_from_slice(header.as_bytes());
    buf.extend(std::iter::repeat_n(b' ', padding));
    buf.push(b'\n');
    buf.extend_from_slice(data);
    buf
}

fn make_tensor(name: &str, shape: Vec<usize>, data: Vec<f32>) -> NamedTensor {
    NamedTensor::new(name, shape, data).expect("valid test tensor")
}

// ===========================================================================
// A. Safetensors loading: shapes, dtypes, multi-tensor files
// ===========================================================================

#[test]
fn test_safetensors_load_verifies_shapes() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bytes = build_safetensors_f32(&[("weights", &[2, 3], &data)]);
    let trace = load_safetensors_from_bytes(&bytes).expect("load should succeed");

    assert_eq!(trace.len(), 1);
    let tensor = trace.get(0).expect("exists");
    assert_eq!(tensor.name, "weights");
    assert_eq!(tensor.shape, vec![2, 3]);
    assert_eq!(tensor.numel(), 6);
    assert_eq!(tensor.data, data);
}

#[test]
fn test_safetensors_load_f16_dtype() {
    let f16_values = [
        half::f16::from_f32(1.0),
        half::f16::from_f32(-0.5),
        half::f16::from_f32(3.14),
    ];
    let raw: Vec<u8> = f16_values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bytes = build_safetensors_typed(&[("f16_tensor", safetensors::Dtype::F16, &[3], &raw)]);
    let trace = load_safetensors_from_bytes(&bytes).expect("load should succeed");

    let tensor = trace.get(0).expect("exists");
    assert_eq!(tensor.shape, vec![3]);
    assert!((tensor.data[0] - 1.0).abs() < 0.01);
    assert!((tensor.data[1] - (-0.5)).abs() < 0.01);
    assert!((tensor.data[2] - 3.14).abs() < 0.02); // f16 limited precision
}

#[test]
fn test_safetensors_load_bf16_dtype() {
    let bf16_values = [
        half::bf16::from_f32(2.0),
        half::bf16::from_f32(-1.5),
        half::bf16::from_f32(0.0),
        half::bf16::from_f32(100.0),
    ];
    let raw: Vec<u8> = bf16_values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bytes =
        build_safetensors_typed(&[("bf16_tensor", safetensors::Dtype::BF16, &[2, 2], &raw)]);
    let trace = load_safetensors_from_bytes(&bytes).expect("load should succeed");

    let tensor = trace.get(0).expect("exists");
    assert_eq!(tensor.shape, vec![2, 2]);
    assert!((tensor.data[0] - 2.0).abs() < 0.1);
    assert!((tensor.data[1] - (-1.5)).abs() < 0.1);
    assert!((tensor.data[2] - 0.0).abs() < f32::EPSILON);
    assert!((tensor.data[3] - 100.0).abs() < 1.0);
}

#[test]
fn test_safetensors_load_f64_dtype() {
    let f64_values: Vec<f64> = vec![1.5, -2.5, 0.0, 42.0];
    let raw: Vec<u8> = f64_values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bytes = build_safetensors_typed(&[("f64_tensor", safetensors::Dtype::F64, &[4], &raw)]);
    let trace = load_safetensors_from_bytes(&bytes).expect("load should succeed");

    let tensor = trace.get(0).expect("exists");
    assert_eq!(tensor.shape, vec![4]);
    assert!((tensor.data[0] - 1.5).abs() < f32::EPSILON);
    assert!((tensor.data[1] - (-2.5)).abs() < f32::EPSILON);
    assert!((tensor.data[2] - 0.0).abs() < f32::EPSILON);
    assert!((tensor.data[3] - 42.0).abs() < f32::EPSILON);
}

#[test]
fn test_safetensors_load_multiple_tensors_sorted() {
    let data_z: Vec<f32> = vec![9.0, 8.0];
    let data_a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let data_m: Vec<f32> = vec![4.0, 5.0, 6.0, 7.0];

    // Insert out of order -- names should be sorted alphabetically on load.
    let bytes = build_safetensors_f32(&[
        ("z_output", &[2], &data_z),
        ("a_input", &[3], &data_a),
        ("m_hidden", &[2, 2], &data_m),
    ]);
    let trace = load_safetensors_from_bytes(&bytes).expect("load should succeed");

    assert_eq!(trace.len(), 3);
    let names: Vec<&str> = trace.names().collect();
    assert_eq!(names, vec!["a_input", "m_hidden", "z_output"]);

    assert_eq!(trace.get_by_name("a_input").expect("exists").shape, vec![3]);
    assert_eq!(
        trace.get_by_name("m_hidden").expect("exists").shape,
        vec![2, 2]
    );
    assert_eq!(
        trace.get_by_name("z_output").expect("exists").shape,
        vec![2]
    );
}

#[test]
fn test_safetensors_load_scalar_shape() {
    let data: Vec<f32> = vec![42.0];
    let bytes = build_safetensors_f32(&[("loss", &[1], &data)]);
    let trace = load_safetensors_from_bytes(&bytes).expect("load should succeed");

    let tensor = trace.get(0).expect("exists");
    assert_eq!(tensor.shape, vec![1]);
    assert_eq!(tensor.data, vec![42.0]);
}

#[test]
fn test_safetensors_load_high_rank_tensor() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let bytes = build_safetensors_f32(&[("high_rank", &[2, 3, 4], &data)]);
    let trace = load_safetensors_from_bytes(&bytes).expect("load should succeed");

    let tensor = trace.get(0).expect("exists");
    assert_eq!(tensor.shape, vec![2, 3, 4]);
    assert_eq!(tensor.numel(), 24);
}

// ===========================================================================
// B. NPY loading
// ===========================================================================

#[test]
fn test_npy_load_f32_from_bytes() {
    let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v1("<f4", &[2, 2], &raw);

    let trace = load_npy_from_bytes(&npy, "test_tensor").expect("load should succeed");
    assert_eq!(trace.len(), 1);

    let tensor = trace.get(0).expect("exists");
    assert_eq!(tensor.name, "test_tensor");
    assert_eq!(tensor.shape, vec![2, 2]);
    assert_eq!(tensor.data, values);
}

#[test]
fn test_npy_load_f64_converts_to_f32() {
    let values: Vec<f64> = vec![1.5, -2.5, 0.0];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v1("<f8", &[3], &raw);

    let trace = load_npy_from_bytes(&npy, "f64_test").expect("load should succeed");
    let tensor = trace.get(0).expect("exists");
    assert_eq!(tensor.shape, vec![3]);
    assert!((tensor.data[0] - 1.5).abs() < f32::EPSILON);
    assert!((tensor.data[1] - (-2.5)).abs() < f32::EPSILON);
    assert!((tensor.data[2] - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_npy_load_f16_converts_to_f32() {
    let f16_values = [half::f16::from_f32(1.0), half::f16::from_f32(0.5)];
    let raw: Vec<u8> = f16_values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v1("<f2", &[2], &raw);

    let trace = load_npy_from_bytes(&npy, "f16_test").expect("load should succeed");
    let tensor = trace.get(0).expect("exists");
    assert!((tensor.data[0] - 1.0).abs() < 0.01);
    assert!((tensor.data[1] - 0.5).abs() < 0.01);
}

#[test]
fn test_npy_load_big_endian() {
    let values: Vec<f32> = vec![1.0, 2.0];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();
    let npy = build_npy_v1(">f4", &[2], &raw);

    let trace = load_npy_from_bytes(&npy, "be_test").expect("load should succeed");
    let tensor = trace.get(0).expect("exists");
    assert_eq!(tensor.data, values);
}

#[test]
fn test_npy_load_integer_converts_to_f32() {
    let values: Vec<i32> = vec![1, -2, 3, 0];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = build_npy_v1("<i4", &[4], &raw);

    let trace = load_npy_from_bytes(&npy, "i32_test").expect("load should succeed");
    let tensor = trace.get(0).expect("exists");
    assert_eq!(tensor.data, vec![1.0, -2.0, 3.0, 0.0]);
}

#[test]
fn test_npy_rejects_bad_magic() {
    let result = load_npy_from_bytes(b"BADDATA", "bad");
    assert!(result.is_err(), "should reject invalid NPY magic bytes");
}

// ===========================================================================
// C. Tolerance assertions
// ===========================================================================

#[test]
fn test_tolerance_exact_match_passes() {
    let a = make_tensor("x", vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let b = make_tensor("x", vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let result = compare_tensors(&a, &b, &ComparisonConfig::default()).expect("should succeed");
    assert!(result.passed);
    assert_eq!(result.max_abs_diff, 0.0);
    assert_eq!(result.max_rel_diff, 0.0);
}

#[test]
fn test_tolerance_within_abs_tolerance_passes() {
    let a = make_tensor("x", vec![3], vec![1.0, 2.0, 3.0]);
    let b = make_tensor("x", vec![3], vec![1.0 + 1e-6, 2.0 - 1e-6, 3.0 + 1e-6]);
    let config = ComparisonConfig::new(1e-5, 1e-4, 0.9999);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(result.passed, "within abs tolerance should pass");
}

#[test]
fn test_tolerance_exceeds_abs_tolerance_fails() {
    let a = make_tensor("x", vec![3], vec![1.0, 2.0, 3.0]);
    let b = make_tensor("x", vec![3], vec![1.0, 2.0, 3.1]);
    let config = ComparisonConfig::new(0.01, 1.0, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(!result.passed, "exceeding abs tolerance should fail");
}

#[test]
fn test_tolerance_exceeds_rel_tolerance_fails() {
    let a = make_tensor("x", vec![2], vec![1.0, 2.0]);
    let b = make_tensor("x", vec![2], vec![1.05, 2.0]); // 5% rel error
    let config = ComparisonConfig::new(0.1, 0.01, 0.0); // 1% rel tolerance
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        !result.passed,
        "exceeding rel tolerance should fail, max_rel={:.4e}",
        result.max_rel_diff
    );
}

#[test]
fn test_tolerance_cosine_threshold_fails() {
    // Orthogonal vectors have cosine similarity ~0.
    let a = make_tensor("x", vec![2], vec![1.0, 0.0]);
    let b = make_tensor("x", vec![2], vec![0.0, 1.0]);
    let config = ComparisonConfig::new(f32::MAX, f32::MAX, 0.99);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        !result.passed,
        "orthogonal vectors should fail cosine threshold"
    );
    assert!(result.cosine_similarity.abs() < 1e-6);
}

#[test]
fn test_relaxed_config_passes_larger_differences() {
    let a = make_tensor("x", vec![3], vec![1.0, 2.0, 3.0]);
    let b = make_tensor("x", vec![3], vec![1.005, 2.005, 3.005]);
    let config = ComparisonConfig::relaxed();
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        result.passed,
        "relaxed config should pass small differences"
    );
}

#[test]
fn test_strict_config_rejects_small_differences() {
    let a = make_tensor("x", vec![3], vec![1.0, 2.0, 3.0]);
    let b = make_tensor("x", vec![3], vec![1.0 + 5e-6, 2.0, 3.0]);
    let config = ComparisonConfig::strict();
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        !result.passed,
        "strict config should reject 5e-6 difference"
    );
}

#[test]
fn test_tolerance_rms_gate() {
    let a = make_tensor("x", vec![4], vec![0.0, 0.0, 0.0, 0.0]);
    let b = make_tensor("x", vec![4], vec![0.1, 0.1, 0.1, 0.1]);
    // RMS = sqrt(mean(0.01)) = 0.1
    let config = ComparisonConfig::new(1.0, 1.0, 0.0).with_rms_tolerance(0.05);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        !result.passed,
        "rms_diff={:.4e} should exceed rms_tolerance=0.05",
        result.rms_diff
    );
}

#[test]
fn test_tolerance_peak_amplitude_gate() {
    let a = make_tensor("x", vec![3], vec![0.0, 0.0, 0.0]);
    let b = make_tensor("x", vec![3], vec![0.0, 0.0, 200.0]);
    let config = ComparisonConfig::new(1000.0, 1000.0, 0.0).with_peak_amplitude_limit(100.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        !result.passed,
        "peak_amplitude={:.1} should exceed limit=100.0",
        result.peak_amplitude
    );
    assert_eq!(result.peak_amplitude, 200.0);
}

// ===========================================================================
// D. Shape mismatch detection
// ===========================================================================

#[test]
fn test_shape_mismatch_different_dims_produces_error() {
    let a = make_tensor("conv_out", vec![6], vec![0.0; 6]);
    let b = NamedTensor::new("conv_out", vec![2, 3], vec![0.0; 6]).expect("valid");

    let err = compare_tensors(&a, &b, &ComparisonConfig::default())
        .expect_err("shape mismatch should return error");

    match err {
        ReftestError::ShapeMismatch {
            name,
            expected,
            actual,
        } => {
            assert_eq!(name, "conv_out");
            assert_eq!(expected, vec![6]);
            assert_eq!(actual, vec![2, 3]);
        }
        other => panic!("expected ShapeMismatch, got: {other:?}"),
    }
}

#[test]
fn test_shape_mismatch_different_sizes_same_rank() {
    let a = NamedTensor::new("hidden", vec![2, 4], vec![0.0; 8]).expect("valid");
    let b = NamedTensor::new("hidden", vec![4, 2], vec![0.0; 8]).expect("valid");

    let err = compare_tensors(&a, &b, &ComparisonConfig::default())
        .expect_err("shape mismatch should return error");

    match err {
        ReftestError::ShapeMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, vec![2, 4]);
            assert_eq!(actual, vec![4, 2]);
        }
        other => panic!("expected ShapeMismatch, got: {other:?}"),
    }
}

#[test]
fn test_shape_mismatch_error_message_contains_details() {
    let a = NamedTensor::new("encoder.fc", vec![3, 5], vec![0.0; 15]).expect("valid");
    let b = NamedTensor::new("encoder.fc", vec![5, 3], vec![0.0; 15]).expect("valid");

    let err = compare_tensors(&a, &b, &ComparisonConfig::default())
        .expect_err("shape mismatch should return error");

    let msg = format!("{err}");
    assert!(
        msg.contains("encoder.fc"),
        "error message should contain tensor name, got: {msg}"
    );
    assert!(
        msg.contains("[3, 5]"),
        "error message should contain expected shape, got: {msg}"
    );
    assert!(
        msg.contains("[5, 3]"),
        "error message should contain actual shape, got: {msg}"
    );
}

#[test]
fn test_named_tensor_data_shape_mismatch() {
    let result = NamedTensor::new("bad", vec![2, 3], vec![1.0, 2.0]);
    match result {
        Err(ReftestError::ElementCountMismatch {
            name,
            expected,
            actual,
            ..
        }) => {
            assert_eq!(name, "bad");
            assert_eq!(expected, 6);
            assert_eq!(actual, 2);
        }
        other => panic!("expected ElementCountMismatch, got: {other:?}"),
    }
}

// ===========================================================================
// E. Dtype mismatch detection (unsupported dtypes)
// ===========================================================================

#[test]
fn test_safetensors_unsupported_dtype_i32_returns_error() {
    // Build safetensors with I32 dtype -- not supported by the loader.
    let values: Vec<i32> = vec![1, 2, 3, 4];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bytes = build_safetensors_typed(&[("int_tensor", safetensors::Dtype::I32, &[4], &raw)]);

    let result = load_safetensors_from_bytes(&bytes);
    assert!(
        result.is_err(),
        "I32 dtype should not be supported in safetensors loader"
    );
    let err = result.unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("unsupported dtype") || msg.contains("Unsupported"),
        "error should mention unsupported dtype, got: {msg}"
    );
}

#[test]
fn test_safetensors_unsupported_dtype_bool_returns_error() {
    let raw: Vec<u8> = vec![1, 0, 1, 1];
    let bytes = build_safetensors_typed(&[("bool_tensor", safetensors::Dtype::BOOL, &[4], &raw)]);

    let result = load_safetensors_from_bytes(&bytes);
    assert!(
        result.is_err(),
        "BOOL dtype should not be supported in safetensors loader"
    );
}

#[test]
fn test_npy_unsupported_dtype_returns_error() {
    // Build NPY with complex dtype which is not supported.
    let raw: Vec<u8> = vec![0; 16]; // dummy data
    let npy = build_npy_v1("<c8", &[2], &raw);

    let result = load_npy_from_bytes(&npy, "complex_test");
    assert!(
        result.is_err(),
        "complex dtype should not be supported in NPY loader"
    );
}

// ===========================================================================
// F. Trace comparison
// ===========================================================================

#[test]
fn test_trace_comparison_matching_multi_layer() {
    let mut reference = ReferenceTrace::new();
    reference
        .checkpoint("encoder.conv1", &[1.0, 2.0, 3.0], &[3])
        .expect("valid");
    reference
        .checkpoint("encoder.bn1", &[0.5, 1.5, 2.5], &[3])
        .expect("valid");
    reference
        .checkpoint("encoder.relu", &[0.5, 1.5, 2.5], &[3])
        .expect("valid");
    reference
        .checkpoint("decoder.linear", &[4.0, 5.0], &[2])
        .expect("valid");

    let candidate = reference.clone();

    let report = compare_traces(&reference, &candidate, &ComparisonConfig::default())
        .expect("should succeed");

    assert!(report.all_passed);
    assert_eq!(report.layers.len(), 4);
    assert!(report.first_failure.is_none());
    for layer in &report.layers {
        assert!(layer.passed);
        assert_eq!(layer.max_abs_diff, 0.0);
    }
}

#[test]
fn test_trace_comparison_divergence_at_specific_layer() {
    let mut reference = ReferenceTrace::new();
    reference
        .checkpoint("layer0", &[1.0, 2.0], &[2])
        .expect("valid");
    reference
        .checkpoint("layer1", &[3.0, 4.0], &[2])
        .expect("valid");
    reference
        .checkpoint("layer2", &[5.0, 6.0], &[2])
        .expect("valid");

    let mut candidate = ReferenceTrace::new();
    candidate
        .checkpoint("layer0", &[1.0, 2.0], &[2])
        .expect("valid");
    candidate
        .checkpoint("layer1", &[3.0, 4.0], &[2])
        .expect("valid");
    candidate
        .checkpoint("layer2", &[5.0, 999.0], &[2])
        .expect("valid"); // large divergence in last layer

    let report = compare_traces(&reference, &candidate, &ComparisonConfig::default())
        .expect("should succeed");

    assert!(!report.all_passed);
    assert_eq!(report.first_failure, Some(2));
    assert!(report.layers[0].passed);
    assert!(report.layers[1].passed);
    assert!(!report.layers[2].passed);
}

#[test]
fn test_trace_comparison_length_mismatch() {
    let mut reference = ReferenceTrace::new();
    reference.checkpoint("a", &[1.0], &[1]).expect("valid");
    reference.checkpoint("b", &[2.0], &[1]).expect("valid");

    let mut candidate = ReferenceTrace::new();
    candidate.checkpoint("a", &[1.0], &[1]).expect("valid");

    let err = compare_traces(&reference, &candidate, &ComparisonConfig::default())
        .expect_err("should fail on length mismatch");

    match err {
        ReftestError::TraceLengthMismatch {
            reference: ref_len,
            candidate: cand_len,
        } => {
            assert_eq!(ref_len, 2);
            assert_eq!(cand_len, 1);
        }
        other => panic!("expected TraceLengthMismatch, got: {other:?}"),
    }
}

#[test]
fn test_trace_comparison_shape_mismatch_in_layer() {
    let mut reference = ReferenceTrace::new();
    reference
        .checkpoint("fc", &[1.0, 2.0, 3.0], &[3])
        .expect("valid");

    let mut candidate = ReferenceTrace::new();
    candidate
        .checkpoint("fc", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .expect("valid");

    let err = compare_traces(&reference, &candidate, &ComparisonConfig::default())
        .expect_err("should fail on shape mismatch");

    assert!(matches!(err, ReftestError::ShapeMismatch { .. }));
}

#[test]
fn test_trace_get_by_name() {
    let mut trace = ReferenceTrace::new();
    trace
        .checkpoint("encoder.conv1", &[1.0, 2.0], &[2])
        .expect("valid");
    trace
        .checkpoint("encoder.bn1", &[3.0, 4.0], &[2])
        .expect("valid");
    trace.checkpoint("decoder.fc", &[5.0], &[1]).expect("valid");

    assert!(trace.get_by_name("encoder.conv1").is_some());
    assert!(trace.get_by_name("decoder.fc").is_some());
    assert!(trace.get_by_name("nonexistent").is_none());

    assert_eq!(
        trace.get_by_name("encoder.bn1").unwrap().data,
        vec![3.0, 4.0]
    );
}

#[test]
fn test_trace_from_safetensors_compared_with_manual() {
    // Load from safetensors.
    let data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let bytes = build_safetensors_f32(&[("layer", &[3], &data)]);
    let loaded = load_safetensors_from_bytes(&bytes).expect("load should succeed");

    // Build manually.
    let mut manual = ReferenceTrace::new();
    manual
        .checkpoint("layer", &[1.0, 2.0, 3.0], &[3])
        .expect("valid");

    assert_traces_match!(loaded, manual);
}

#[test]
fn test_trace_capture_utility() {
    let (trace, result) = ReferenceTrace::capture(|capture| {
        capture
            .checkpoint("step1", &[1.0, 2.0], &[2])
            .expect("valid");
        capture
            .checkpoint("step2", &[3.0, 4.0, 5.0], &[3])
            .expect("valid");
        "done"
    });

    assert_eq!(result, "done");
    assert_eq!(trace.len(), 2);
    assert_eq!(trace.get(0).expect("exists").name, "step1");
    assert_eq!(trace.get(1).expect("exists").name, "step2");
}

// ===========================================================================
// G. Statistics reporting: mean absolute error, max absolute error, cosine,
//    rms, peak amplitude
// ===========================================================================

#[test]
fn test_statistics_max_abs_diff() {
    let a = make_tensor("x", vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let b = make_tensor("x", vec![4], vec![1.0, 2.0, 3.5, 4.0]);
    let config = ComparisonConfig::new(1.0, 1.0, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        (result.max_abs_diff - 0.5).abs() < 1e-7,
        "max_abs_diff should be 0.5, got {}",
        result.max_abs_diff
    );
}

#[test]
fn test_statistics_mean_abs_diff() {
    // diffs: [0.0, 0.0, 0.5, 0.0] -> mean = 0.125
    let a = make_tensor("x", vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let b = make_tensor("x", vec![4], vec![1.0, 2.0, 3.5, 4.0]);
    let config = ComparisonConfig::new(1.0, 1.0, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        (result.mean_abs_diff - 0.125).abs() < 1e-6,
        "mean_abs_diff should be 0.125, got {}",
        result.mean_abs_diff
    );
}

#[test]
fn test_statistics_rms_diff() {
    // diffs: [0.1, 0.0, 0.0, 0.0] -> sum_sq = 0.01, mean_sq = 0.0025, rms = 0.05
    let a = make_tensor("x", vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let b = make_tensor("x", vec![4], vec![1.1, 2.0, 3.0, 4.0]);
    let config = ComparisonConfig::new(1.0, 1.0, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");

    let expected_rms = (0.01_f64 / 4.0).sqrt() as f32;
    assert!(
        (result.rms_diff - expected_rms).abs() < 1e-7,
        "rms_diff should be {expected_rms:.6e}, got {:.6e}",
        result.rms_diff
    );
}

#[test]
fn test_statistics_cosine_similarity_identical() {
    let a = make_tensor("x", vec![3], vec![1.0, 2.0, 3.0]);
    let b = make_tensor("x", vec![3], vec![1.0, 2.0, 3.0]);
    let config = ComparisonConfig::new(1.0, 1.0, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        (result.cosine_similarity - 1.0).abs() < 1e-6,
        "identical vectors should have cosine ~1.0, got {}",
        result.cosine_similarity
    );
}

#[test]
fn test_statistics_cosine_similarity_scaled() {
    // Scaling a vector doesn't change direction -- cosine should be ~1.0.
    let a = make_tensor("x", vec![3], vec![1.0, 2.0, 3.0]);
    let b = make_tensor("x", vec![3], vec![10.0, 20.0, 30.0]);
    let config = ComparisonConfig::new(f32::MAX, f32::MAX, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        (result.cosine_similarity - 1.0).abs() < 1e-6,
        "scaled vectors should have cosine ~1.0, got {}",
        result.cosine_similarity
    );
}

#[test]
fn test_statistics_cosine_similarity_antiparallel() {
    let a = make_tensor("x", vec![3], vec![1.0, 2.0, 3.0]);
    let b = make_tensor("x", vec![3], vec![-1.0, -2.0, -3.0]);
    let config = ComparisonConfig::new(f32::MAX, f32::MAX, -2.0); // accept any cosine
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        (result.cosine_similarity - (-1.0)).abs() < 1e-6,
        "antiparallel vectors should have cosine ~-1.0, got {}",
        result.cosine_similarity
    );
}

#[test]
fn test_statistics_cosine_zero_vectors() {
    let a = make_tensor("x", vec![3], vec![0.0, 0.0, 0.0]);
    let b = make_tensor("x", vec![3], vec![0.0, 0.0, 0.0]);
    let config = ComparisonConfig::new(1.0, 1.0, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        (result.cosine_similarity - 1.0).abs() < 1e-6,
        "two zero vectors should have cosine=1.0, got {}",
        result.cosine_similarity
    );
}

#[test]
fn test_statistics_cosine_one_zero_vector() {
    let a = make_tensor("x", vec![3], vec![0.0, 0.0, 0.0]);
    let b = make_tensor("x", vec![3], vec![1.0, 2.0, 3.0]);
    let config = ComparisonConfig::new(f32::MAX, f32::MAX, -2.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        result.cosine_similarity.abs() < 1e-6,
        "zero vs non-zero should have cosine=0.0, got {}",
        result.cosine_similarity
    );
}

#[test]
fn test_statistics_peak_amplitude() {
    let a = make_tensor("x", vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let b = make_tensor("x", vec![4], vec![-7.0, 2.0, 3.0, 4.0]);
    let config = ComparisonConfig::new(100.0, 100.0, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert_eq!(
        result.peak_amplitude, 7.0,
        "peak amplitude should be abs(-7.0) = 7.0"
    );
}

#[test]
fn test_statistics_num_elements() {
    let a = make_tensor("x", vec![5], vec![0.0; 5]);
    let b = make_tensor("x", vec![5], vec![0.0; 5]);
    let config = ComparisonConfig::new(1.0, 1.0, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert_eq!(result.num_elements, 5);
}

#[test]
fn test_statistics_max_rel_diff() {
    // ref=100.0, cand=110.0 -> abs=10.0, rel=10/110 ~= 0.0909
    let a = make_tensor("x", vec![2], vec![100.0, 200.0]);
    let b = make_tensor("x", vec![2], vec![110.0, 200.0]);
    let config = ComparisonConfig::new(100.0, 1.0, 0.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    // rel = 10 / max(100, 110) = 10/110 ~= 0.0909
    assert!(
        (result.max_rel_diff - 10.0 / 110.0).abs() < 1e-5,
        "max_rel_diff should be ~0.0909, got {}",
        result.max_rel_diff
    );
}

// ===========================================================================
// H. NaN/Inf handling in statistics
// ===========================================================================

#[test]
fn test_statistics_nan_produces_infinite_metrics() {
    let a = make_tensor("x", vec![3], vec![1.0, 2.0, 3.0]);
    let b = make_tensor("x", vec![3], vec![1.0, f32::NAN, 3.0]);
    let config = ComparisonConfig::new(f32::MAX, f32::MAX, -2.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");

    assert!(result.max_abs_diff.is_infinite());
    assert!(result.max_rel_diff.is_infinite());
    assert!(!result.passed);
}

#[test]
fn test_statistics_all_nan_cosine_is_nan() {
    let a = make_tensor("x", vec![2], vec![f32::NAN, f32::NAN]);
    let b = make_tensor("x", vec![2], vec![f32::NAN, f32::NAN]);
    let config = ComparisonConfig::new(f32::MAX, f32::MAX, -2.0);
    let result = compare_tensors(&a, &b, &config).expect("should succeed");
    assert!(
        result.cosine_similarity.is_nan(),
        "all-NaN should produce NaN cosine, not {}",
        result.cosine_similarity
    );
}

// ===========================================================================
// I. DivergenceReport summary formatting
// ===========================================================================

#[test]
fn test_divergence_report_summary_all_passed() {
    let mut reference = ReferenceTrace::new();
    reference.checkpoint("layer1", &[1.0], &[1]).expect("valid");
    reference.checkpoint("layer2", &[2.0], &[1]).expect("valid");

    let candidate = reference.clone();
    let report = compare_traces(&reference, &candidate, &ComparisonConfig::default())
        .expect("should succeed");

    let summary = report.summary();
    assert!(summary.contains("All 2 layers passed"));
    assert!(summary.contains("PASS"));
    assert!(!summary.contains("FAIL"));
}

#[test]
fn test_divergence_report_summary_with_failure() {
    let mut reference = ReferenceTrace::new();
    reference.checkpoint("conv", &[1.0], &[1]).expect("valid");
    reference.checkpoint("relu", &[2.0], &[1]).expect("valid");

    let mut candidate = ReferenceTrace::new();
    candidate.checkpoint("conv", &[1.0], &[1]).expect("valid");
    candidate.checkpoint("relu", &[999.0], &[1]).expect("valid");

    let report = compare_traces(&reference, &candidate, &ComparisonConfig::default())
        .expect("should succeed");

    let summary = report.summary();
    assert!(summary.contains("FAIL"));
    assert!(summary.contains("relu"));
    assert!(summary.contains("First failure"));
}

// ===========================================================================
// J. assert_traces_match! macro variants
// ===========================================================================

#[test]
fn test_assert_macro_epsilon_with_cosine() {
    let mut reference = ReferenceTrace::new();
    reference
        .checkpoint("x", &[1.0, 2.0, 3.0], &[3])
        .expect("valid");

    let mut candidate = ReferenceTrace::new();
    candidate
        .checkpoint("x", &[1.0001, 2.0001, 3.0001], &[3])
        .expect("valid");

    assert_traces_match!(candidate, reference, epsilon = 0.001, cos = 0.999);
}

#[test]
fn test_assert_macro_panics_with_detailed_message() {
    let mut reference = ReferenceTrace::new();
    reference
        .checkpoint("encoder.out", &[1.0, 2.0], &[2])
        .expect("valid");

    let mut candidate = ReferenceTrace::new();
    candidate
        .checkpoint("encoder.out", &[1.0, 999.0], &[2])
        .expect("valid");

    let result = std::panic::catch_unwind(|| {
        assert_traces_match!(candidate, reference);
    });

    let err = result.expect_err("should panic");
    let msg = err
        .downcast_ref::<String>()
        .map(String::as_str)
        .unwrap_or_else(|| err.downcast_ref::<&str>().copied().unwrap_or("unknown"));

    assert!(msg.contains("encoder.out"), "should name the failing layer");
    assert!(msg.contains("max_abs_diff"), "should report max_abs_diff");
    assert!(msg.contains("cosine_similarity"), "should report cosine");
}

// ===========================================================================
// K. Cross-format comparison: safetensors vs NPY vs manual trace
// ===========================================================================

#[test]
fn test_safetensors_vs_npy_vs_manual_all_match() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    // From safetensors.
    let st_bytes = build_safetensors_f32(&[("tensor", &[4], &data)]);
    let st_trace = load_safetensors_from_bytes(&st_bytes).expect("safetensors load");

    // From NPY.
    let npy_raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy_bytes = build_npy_v1("<f4", &[4], &npy_raw);
    let npy_trace = load_npy_from_bytes(&npy_bytes, "tensor").expect("npy load");

    // Manual.
    let mut manual = ReferenceTrace::new();
    manual.checkpoint("tensor", &data, &[4]).expect("valid");

    assert_traces_match!(st_trace, npy_trace);
    assert_traces_match!(st_trace, manual);
    assert_traces_match!(npy_trace, manual);
}
