// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for overlay composition verification API.

use std::collections::HashMap;

use super::*;
use ndarray::ArrayD;

/// Build a simple Linear model (y = W*x + b) for testing.
fn linear_model() -> TensorKernelDef {
    use nn_dsl::tensor_block_builder::TensorBlockBuilder;

    let mut builder = TensorBlockBuilder::new("linear_test");
    let x = builder.add_input("x", &[1, 4]);
    let w = builder.add_input("weight", &[4, 4]);
    let b = builder.add_input("bias", &[4]);
    let y = builder.add_matmul(x, w, false, None, &[1, 4]);
    let b_bc = builder.add_broadcast(b, &[1, 4]);
    let out = builder.add_binary_add(y, b_bc, &[1, 4]);
    builder.build(out).expect("valid graph")
}

/// Create input bounds for a [1, 4] tensor in [-1, 1].
fn make_input_bounds() -> BoundedTensor {
    let lower = ArrayD::from_elem(vec![1, 4], -1.0_f32);
    let upper = ArrayD::from_elem(vec![1, 4], 1.0_f32);
    BoundedTensor::new(lower, upper).expect("valid bounds")
}

/// Helper to create a small delta tensor.
fn small_delta(shape: &[usize], value: f32) -> ArrayD<f32> {
    ArrayD::from_elem(shape.to_vec(), value)
}

#[test]
fn test_verified_overlay_new_computes_norms() {
    let mut deltas = HashMap::new();
    deltas.insert(1, ArrayD::from_elem(vec![4, 4], 0.01_f32));

    let overlay = VerifiedOverlay::new("test".into(), deltas).expect("valid overlay");
    assert_eq!(overlay.name, "test");
    assert!(overlay.target_params.contains(&1));

    // Frobenius norm of a 4x4 matrix filled with 0.01 = 0.01 * sqrt(16) = 0.04
    let norm = overlay.delta_norms[&1];
    assert!(
        (norm - 0.04).abs() < 1e-6,
        "expected norm ~0.04, got {norm}"
    );
}

#[test]
fn test_verified_overlay_new_rejects_empty_deltas() {
    let deltas = HashMap::new();
    let err = VerifiedOverlay::new("empty".into(), deltas).unwrap_err();
    assert!(
        err.to_string().contains("at least one"),
        "expected error about empty deltas, got: {err}",
    );
}

#[test]
fn test_verified_overlay_new_rejects_non_finite_delta() {
    let mut deltas = HashMap::new();
    let mut arr = ArrayD::from_elem(vec![2, 2], 1.0_f32);
    arr[[0, 0]] = f32::NAN;
    deltas.insert(0, arr);

    let err = VerifiedOverlay::new("nan_overlay".into(), deltas).unwrap_err();
    assert!(
        err.to_string().contains("non-finite"),
        "expected non-finite error, got: {err}",
    );
}

#[test]
fn test_overlay_interaction_matrix_disjoint() {
    let o1 = VerifiedOverlay::new(
        "o1".into(),
        [(1, small_delta(&[4, 4], 0.01))].into_iter().collect(),
    )
    .unwrap();
    let o2 = VerifiedOverlay::new(
        "o2".into(),
        [(2, small_delta(&[4], 0.01))].into_iter().collect(),
    )
    .unwrap();

    let matrix = overlay_interaction_matrix(&[o1, o2]);
    // Each param is targeted by exactly one overlay.
    assert_eq!(matrix[&1].len(), 1);
    assert_eq!(matrix[&2].len(), 1);
}

#[test]
fn test_overlay_interaction_matrix_overlapping() {
    let o1 = VerifiedOverlay::new(
        "o1".into(),
        [(1, small_delta(&[4, 4], 0.01))].into_iter().collect(),
    )
    .unwrap();
    let o2 = VerifiedOverlay::new(
        "o2".into(),
        [(1, small_delta(&[4, 4], 0.02))].into_iter().collect(),
    )
    .unwrap();

    let matrix = overlay_interaction_matrix(&[o1, o2]);
    // Both overlays target param 1.
    assert_eq!(matrix[&1].len(), 2);
    assert_eq!(matrix[&1], vec![0, 1]);
}

#[test]
fn test_verify_composition_disjoint_overlays() {
    let model = linear_model();
    let input_bounds = make_input_bounds();

    let mut original_weights = HashMap::new();
    original_weights.insert(1, ArrayD::from_elem(vec![4, 4], 0.5_f32));
    original_weights.insert(2, ArrayD::from_elem(vec![4], 0.1_f32));

    // Two overlays targeting different parameters.
    let o1 = VerifiedOverlay::new(
        "weight_edit".into(),
        [(1, small_delta(&[4, 4], 0.001))].into_iter().collect(),
    )
    .unwrap();
    let o2 = VerifiedOverlay::new(
        "bias_edit".into(),
        [(2, small_delta(&[4], 0.001))].into_iter().collect(),
    )
    .unwrap();

    // Generous epsilon: IBP independent propagation produces wide diff bounds.
    let specs = vec![BoundSpec::new(10.0, "output preservation")];

    let cert =
        verify_overlay_composition(&model, &original_weights, &[o1, o2], &input_bounds, &specs)
            .expect("composition should verify");

    assert_eq!(cert.overlay_names, vec!["weight_edit", "bias_edit"]);
    assert!(cert.overlapping_params.is_empty(), "should be disjoint");
    assert_eq!(cert.disjoint_params.len(), 2);
    assert!(
        cert.verification.is_some(),
        "should have verification result"
    );
    let v = cert.verification.as_ref().unwrap();
    assert!(v.max_abs_diff.is_finite(), "diff should be finite");
}

#[test]
fn test_verify_composition_overlapping_overlays() {
    let model = linear_model();
    let input_bounds = make_input_bounds();

    let mut original_weights = HashMap::new();
    original_weights.insert(1, ArrayD::from_elem(vec![4, 4], 0.5_f32));
    original_weights.insert(2, ArrayD::from_elem(vec![4], 0.1_f32));

    // Two overlays targeting the same weight parameter.
    let o1 = VerifiedOverlay::new(
        "style_a".into(),
        [(1, small_delta(&[4, 4], 0.001))].into_iter().collect(),
    )
    .unwrap();
    let o2 = VerifiedOverlay::new(
        "style_b".into(),
        [(1, small_delta(&[4, 4], 0.001))].into_iter().collect(),
    )
    .unwrap();

    // Generous epsilon: IBP independent propagation produces wide diff bounds.
    let specs = vec![BoundSpec::new(10.0, "output preservation")];

    let cert =
        verify_overlay_composition(&model, &original_weights, &[o1, o2], &input_bounds, &specs)
            .expect("composition should verify");

    assert!(
        cert.overlapping_params.contains(&1),
        "param 1 should be overlapping",
    );
    assert!(
        cert.accumulated_norms.contains_key(&1),
        "should have accumulated norm for overlapping param",
    );

    // Combined delta = 0.001 + 0.001 = 0.002 per element, norm = 0.002 * sqrt(16) = 0.008
    let norm = cert.accumulated_norms[&1];
    assert!(
        (norm - 0.008).abs() < 1e-5,
        "expected accumulated norm ~0.008, got {norm}",
    );

    // Verify propagation produced finite results.
    assert!(
        cert.verification.is_some(),
        "should have verification result"
    );
    let v = cert.verification.as_ref().unwrap();
    assert!(v.max_abs_diff.is_finite(), "diff should be finite");
}

#[test]
fn test_verify_composition_rejects_empty_overlays() {
    let model = linear_model();
    let input_bounds = make_input_bounds();

    let err =
        verify_overlay_composition(&model, &HashMap::new(), &[], &input_bounds, &[]).unwrap_err();
    assert!(
        err.to_string().contains("at least one overlay"),
        "expected empty overlay error, got: {err}",
    );
}

#[test]
fn test_verify_composition_rejects_unknown_target_param() {
    let model = linear_model();
    let input_bounds = make_input_bounds();

    let original_weights = HashMap::new(); // No weights provided

    let o1 = VerifiedOverlay::new(
        "bad_target".into(),
        [(99, small_delta(&[4, 4], 0.01))].into_iter().collect(),
    )
    .unwrap();

    let err = verify_overlay_composition(
        &model,
        &original_weights,
        &[o1],
        &input_bounds,
        &[BoundSpec::new(0.1, "test")],
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("not in original_weights"),
        "expected unknown param error, got: {err}",
    );
}

#[test]
fn test_verify_composition_rejects_shape_mismatch() {
    let model = linear_model();
    let input_bounds = make_input_bounds();

    let mut original_weights = HashMap::new();
    original_weights.insert(1, ArrayD::from_elem(vec![4, 4], 0.5_f32));

    // Overlay with wrong shape delta.
    let o1 = VerifiedOverlay::new(
        "wrong_shape".into(),
        [(1, small_delta(&[4, 3], 0.01))].into_iter().collect(), // 4x3 != 4x4
    )
    .unwrap();

    let err = verify_overlay_composition(
        &model,
        &original_weights,
        &[o1],
        &input_bounds,
        &[BoundSpec::new(0.1, "test")],
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("shape"),
        "expected shape mismatch error, got: {err}",
    );
}

#[test]
fn test_composition_certificate_serde_roundtrip() {
    let cert = CompositionCertificate {
        overlay_names: vec!["o1".into(), "o2".into()],
        overlapping_params: vec![1],
        disjoint_params: vec![2],
        accumulated_norms: [(1, 0.005)].into_iter().collect(),
        verification: None,
        all_specs_pass: true,
        spec_results: vec![SpecResult {
            description: "test".into(),
            epsilon: 0.1,
            passed: true,
            max_abs_diff: Some(0.005),
        }],
        method: PropMethod::Crown,
        soundness_mode: VerificationSoundnessMode::Sound,
    };

    let json = serde_json::to_string_pretty(&cert).expect("serialize");
    let cert2: CompositionCertificate = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(cert.overlay_names, cert2.overlay_names);
    assert_eq!(cert.all_specs_pass, cert2.all_specs_pass);
    assert_eq!(cert.method, cert2.method);
}

#[test]
fn test_bound_spec_new() {
    let spec = BoundSpec::new(0.01, "intelligibility");
    assert_eq!(spec.epsilon, 0.01);
    assert_eq!(spec.description, "intelligibility");
}
