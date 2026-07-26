// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `external_bounds` module.

use std::collections::BTreeMap;

use super::*;

/// Convert f32 slice to little-endian bytes.
fn f32_to_le_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Build a safetensors buffer from tensors with metadata.
pub(super) fn build_safetensors_with_metadata(
    tensors: &[(&str, &[usize], &[f32])],
    metadata: Option<&std::collections::HashMap<String, String>>,
) -> Vec<u8> {
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

    safetensors::tensor::serialize(tensor_map, metadata.cloned()).expect("serialization")
}

#[test]
fn test_load_minimal_bounds() {
    let lower = [0.1_f32, 0.2];
    let upper = [0.9_f32, 0.8];
    let bytes = build_safetensors_with_metadata(
        &[
            ("output/lower", &[2], &lower),
            ("output/upper", &[2], &upper),
        ],
        None,
    );

    let result = load_external_bounds_from_bytes(&bytes).expect("load");
    assert_eq!(result.output_lower, vec![0.1, 0.2]);
    assert_eq!(result.output_upper, vec![0.9, 0.8]);
    assert_eq!(result.output_shape, vec![2]);
    assert!(result.layer_bounds.is_empty());
    assert_eq!(result.source.engine, "unknown");
}

#[test]
fn test_load_with_metadata() {
    let mut meta = std::collections::HashMap::new();
    meta.insert("method".to_string(), "CROWN-Optimized".to_string());
    meta.insert("engine".to_string(), "auto_LiRPA".to_string());
    meta.insert("eps".to_string(), "0.01".to_string());
    meta.insert("input_shape".to_string(), "[1, 80, 256]".to_string());

    let bytes = build_safetensors_with_metadata(
        &[
            ("output/lower", &[1], &[-1.0]),
            ("output/upper", &[1], &[1.0]),
        ],
        Some(&meta),
    );

    let result = load_external_bounds_from_bytes(&bytes).expect("load");
    assert_eq!(result.source.method, "CROWN-Optimized");
    assert_eq!(result.source.engine, "auto_LiRPA");
    assert!((result.source.eps - 0.01).abs() < f64::EPSILON);
    assert_eq!(result.source.input_shape, vec![1, 80, 256]);
}

#[test]
fn test_load_with_layer_bounds() {
    let bytes = build_safetensors_with_metadata(
        &[
            ("output/lower", &[2], &[0.1, 0.2]),
            ("output/upper", &[2], &[0.9, 0.8]),
            ("layer/relu_0/lower", &[3], &[0.0, 0.0, 0.1]),
            ("layer/relu_0/upper", &[3], &[1.0, 0.5, 0.3]),
            ("layer/linear_1/lower", &[2], &[-0.5, -0.3]),
            ("layer/linear_1/upper", &[2], &[0.5, 0.3]),
        ],
        None,
    );

    let result = load_external_bounds_from_bytes(&bytes).expect("load");
    assert_eq!(result.layer_bounds.len(), 2);
    assert!(result.layer_bounds.contains_key("relu_0"));
    assert!(result.layer_bounds.contains_key("linear_1"));

    let relu = &result.layer_bounds["relu_0"];
    assert_eq!(relu.shape, vec![3]);
    assert_eq!(relu.lower, vec![0.0, 0.0, 0.1]);
}

#[test]
fn test_missing_output_lower() {
    let bytes = build_safetensors_with_metadata(&[("output/upper", &[1], &[1.0])], None);
    let err = load_external_bounds_from_bytes(&bytes).unwrap_err();
    assert!(err.to_string().contains("output/lower"));
}

#[test]
fn test_non_finite_rejected() {
    let bytes = build_safetensors_with_metadata(
        &[
            ("output/lower", &[2], &[0.0, f32::NAN]),
            ("output/upper", &[2], &[1.0, 1.0]),
        ],
        None,
    );
    let err = load_external_bounds_from_bytes(&bytes).unwrap_err();
    assert!(err.to_string().contains("non-finite"));
}

#[test]
fn test_verification_from_external() {
    let bounds = ExternalBounds {
        source: ExternalBoundsSource::new(
            "CROWN-Optimized".to_string(),
            "auto_LiRPA".to_string(),
            0.01,
            vec![1, 80],
        ),
        output_lower: vec![-0.5, 0.1],
        output_upper: vec![0.8, 1.2],
        output_shape: vec![2],
        layer_bounds: BTreeMap::new(),
    };

    let v = verification_from_external(&bounds, "kokoro_decoder");
    assert_eq!(v.kernel_name, "kokoro_decoder");
    assert_eq!(v.method, PropMethod::Crown);
    assert!((v.output_lower - (-0.5)).abs() < f32::EPSILON);
    assert!((v.output_upper - 1.2).abs() < f32::EPSILON);
    assert!(v.is_finite);
    assert_eq!(v.soundness_mode, VerificationSoundnessMode::Heuristic);

    let tensor = v.output_tensor.as_ref().expect("should have output tensor");
    assert_eq!(tensor.lower, vec![-0.5, 0.1]);
    assert_eq!(tensor.upper, vec![0.8, 1.2]);
    assert_eq!(tensor.shape, vec![2]);
    assert_eq!(tensor.finite_mask, vec![true, true]);
}

#[test]
fn test_length_mismatch_rejected() {
    // output/lower has 2 elements, output/upper has 3.
    let bytes = build_safetensors_with_metadata(
        &[
            ("output/lower", &[2], &[0.0, 1.0]),
            ("output/upper", &[3], &[0.5, 0.8, 1.0]),
        ],
        None,
    );
    let err = load_external_bounds_from_bytes(&bytes).unwrap_err();
    assert!(err.to_string().contains("mismatch"));
}

#[test]
fn test_inverted_output_bounds_rejected() {
    // lower > upper at index 1: 0.9 > 0.2
    let bytes = build_safetensors_with_metadata(
        &[
            ("output/lower", &[2], &[0.1, 0.9]),
            ("output/upper", &[2], &[0.8, 0.2]),
        ],
        None,
    );
    let err = load_external_bounds_from_bytes(&bytes).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("inverted bounds"),
        "expected inverted bounds error, got: {msg}"
    );
    assert!(msg.contains("index 1"), "should report index 1, got: {msg}");
}

#[test]
fn test_inverted_layer_bounds_rejected() {
    // Output bounds valid, but layer bounds inverted at index 0.
    let bytes = build_safetensors_with_metadata(
        &[
            ("output/lower", &[1], &[0.0]),
            ("output/upper", &[1], &[1.0]),
            ("layer/relu_0/lower", &[2], &[0.5, 0.1]),
            ("layer/relu_0/upper", &[2], &[0.3, 0.9]),
        ],
        None,
    );
    let err = load_external_bounds_from_bytes(&bytes).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("inverted bounds"),
        "expected inverted bounds error, got: {msg}"
    );
    assert!(
        msg.contains("relu_0"),
        "should mention layer name, got: {msg}"
    );
}

#[test]
fn test_layer_bounds_length_mismatch_rejected() {
    // Per-layer bounds with mismatched lower (3 elements) vs upper (2 elements).
    // Before the fix, validate_ordering's zip would silently truncate to
    // the shorter length, leaving the 3rd lower element unchecked.
    let bytes = build_safetensors_with_metadata(
        &[
            ("output/lower", &[1], &[0.0]),
            ("output/upper", &[1], &[1.0]),
            ("layer/linear_0/lower", &[3], &[0.1, 0.2, 0.3]),
            ("layer/linear_0/upper", &[2], &[0.9, 0.8]),
        ],
        None,
    );
    let err = load_external_bounds_from_bytes(&bytes).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("length mismatch"),
        "expected length mismatch error, got: {msg}"
    );
    assert!(
        msg.contains("linear_0"),
        "should mention layer name, got: {msg}"
    );
}

#[test]
fn test_equal_bounds_accepted() {
    // lower == upper is valid (point interval).
    let bytes = build_safetensors_with_metadata(
        &[
            ("output/lower", &[2], &[0.5, 0.5]),
            ("output/upper", &[2], &[0.5, 0.5]),
        ],
        None,
    );
    let result = load_external_bounds_from_bytes(&bytes).expect("equal bounds should be valid");
    assert_eq!(result.output_lower, result.output_upper);
}

#[test]
fn test_verify_and_record_external() {
    use crate::status::VerifyStatus;

    let mut meta = std::collections::HashMap::new();
    meta.insert("method".to_string(), "CROWN-Optimized".to_string());
    meta.insert("engine".to_string(), "auto_LiRPA".to_string());
    meta.insert("eps".to_string(), "0.03".to_string());
    meta.insert("input_shape".to_string(), "[1, 80]".to_string());

    let bytes = build_safetensors_with_metadata(
        &[
            ("output/lower", &[2], &[-0.5, 0.1]),
            ("output/upper", &[2], &[0.8, 1.2]),
        ],
        Some(&meta),
    );

    let external = load_external_bounds_from_bytes(&bytes).expect("load");
    let mut status = VerifyStatus::default();

    let (recorded_ext, verification) =
        verify_and_record_external_from_loaded(external, &mut status, "silero_vad")
            .expect("pipeline");

    // Verification result checks.
    assert_eq!(verification.kernel_name, "silero_vad");
    assert!((verification.output_lower - (-0.5)).abs() < f32::EPSILON);
    assert!((verification.output_upper - 1.2).abs() < f32::EPSILON);
    assert!(verification.is_finite);

    // Status recording checks.
    let entry = status.kernel("silero_vad").expect("recorded");
    assert_eq!(entry.status, crate::status::VerifyOutcome::Verified);
    assert_eq!(entry.method, PropMethod::Crown);
    assert_eq!(entry.soundness_mode, VerificationSoundnessMode::Heuristic,);
    assert!(entry.output_bounds.tensor_lower.is_some());
    assert!(entry.output_bounds.tensor_upper.is_some());
    assert_eq!(entry.output_bounds.shape, Some(vec![2]));

    // Source metadata preserved.
    assert_eq!(recorded_ext.source.engine, "auto_LiRPA");
    assert!((recorded_ext.source.eps - 0.03).abs() < f64::EPSILON);
}
