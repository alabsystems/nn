// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for edit verification via dual-path propagation.

use std::collections::HashMap;

use super::*;
use crate::verify::PropMethod;

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
    use ndarray::ArrayD;
    let lower = ArrayD::from_elem(vec![1, 4], -1.0_f32);
    let upper = ArrayD::from_elem(vec![1, 4], 1.0_f32);
    BoundedTensor::new(lower, upper).expect("valid bounds")
}

#[test]
fn test_edit_verify_identical_weights_produces_finite_bounds() {
    let model = linear_model();
    let input_bounds = make_input_bounds();

    // Original and edited are identical — true diff is 0.
    // Independent propagation (IBP) produces vacuously wide diff bounds because
    // two separate propagations lose input correlation. This is the expected
    // behavior documented in `is_conclusive()`: only CROWN diamond results
    // would be tight. We verify the pipeline succeeds and produces finite bounds.
    let weight = ArrayD::from_elem(vec![4, 4], 0.5_f32);
    let bias = ArrayD::from_elem(vec![4], 0.1_f32);

    let mut original = HashMap::new();
    original.insert(1, weight); // param index 1 = weight
    original.insert(2, bias); // param index 2 = bias

    let spec = EditVerificationSpec {
        model,
        original_weights: original.clone(),
        edited_weights: original,
        input_bounds,
        epsilon: 10.0, // Generous: IBP diff bounds are wide for identical weights
    };

    let result = verify_edit(&spec).expect("verification should succeed");
    assert!(result.diff_lower.is_finite(), "diff_lower should be finite");
    assert!(result.diff_upper.is_finite(), "diff_upper should be finite");
    assert!(
        result.max_abs_diff.is_finite(),
        "max_abs_diff should be finite"
    );
    // Symmetric bounds for identical weights.
    assert!(
        (result.diff_lower.abs() - result.diff_upper.abs()).abs() < 1e-4,
        "symmetric bounds expected for identical weights: lower={}, upper={}",
        result.diff_lower,
        result.diff_upper,
    );
}

#[test]
fn test_edit_verify_small_perturbation() {
    let model = linear_model();
    let input_bounds = make_input_bounds();

    let weight = ArrayD::from_elem(vec![4, 4], 0.5_f32);
    let bias = ArrayD::from_elem(vec![4], 0.1_f32);

    let mut original = HashMap::new();
    original.insert(1, weight);
    original.insert(2, bias);

    // Small perturbation to bias only.
    let mut edited = original.clone();
    let perturbed_bias = ArrayD::from_elem(vec![4], 0.1001_f32);
    edited.insert(2, perturbed_bias);

    let spec = EditVerificationSpec {
        model,
        original_weights: original,
        edited_weights: edited,
        input_bounds,
        epsilon: 10.0, // Generous: IBP independent propagation produces wide diff bounds
    };

    let result = verify_edit(&spec).expect("verification should succeed");
    // Independent propagation diff bounds are wider than the true perturbation.
    // Verify pipeline succeeds and produces finite, bounded output.
    assert!(
        result.max_abs_diff.is_finite(),
        "max_abs_diff should be finite"
    );
    assert!(
        result.max_abs_diff > 0.0,
        "diff should be positive (non-trivial bounds)",
    );
}

#[test]
fn test_edit_verify_invalid_epsilon_nan() {
    let model = linear_model();
    let input_bounds = make_input_bounds();

    let spec = EditVerificationSpec {
        model,
        original_weights: HashMap::new(),
        edited_weights: HashMap::new(),
        input_bounds,
        epsilon: f32::NAN,
    };

    let err = verify_edit(&spec).unwrap_err();
    assert!(
        err.to_string().contains("threshold"),
        "expected InvalidThreshold, got: {err}",
    );
}

#[test]
fn test_edit_verify_invalid_epsilon_negative() {
    let model = linear_model();
    let input_bounds = make_input_bounds();

    let spec = EditVerificationSpec {
        model,
        original_weights: HashMap::new(),
        edited_weights: HashMap::new(),
        input_bounds,
        epsilon: -1.0,
    };

    let err = verify_edit(&spec).unwrap_err();
    assert!(
        err.to_string().contains("threshold"),
        "expected InvalidThreshold, got: {err}",
    );
}

#[test]
fn test_edit_verify_mismatched_weight_keys() {
    let model = linear_model();
    let input_bounds = make_input_bounds();

    let weight = ArrayD::from_elem(vec![4, 4], 0.5_f32);
    let mut original = HashMap::new();
    original.insert(1, weight.clone());

    let mut edited = HashMap::new();
    edited.insert(2, weight); // Different key!

    let spec = EditVerificationSpec {
        model,
        original_weights: original,
        edited_weights: edited,
        input_bounds,
        epsilon: 0.1,
    };

    let err = verify_edit(&spec).unwrap_err();
    assert!(
        err.to_string().contains("same parameter indices"),
        "expected key mismatch error, got: {err}",
    );
}

#[test]
fn test_edit_verify_mismatched_weight_shapes() {
    let model = linear_model();
    let input_bounds = make_input_bounds();

    let mut original = HashMap::new();
    original.insert(1, ArrayD::from_elem(vec![4, 4], 0.5_f32));

    let mut edited = HashMap::new();
    edited.insert(1, ArrayD::from_elem(vec![4, 3], 0.5_f32)); // Wrong shape

    let spec = EditVerificationSpec {
        model,
        original_weights: original,
        edited_weights: edited,
        input_bounds,
        epsilon: 0.1,
    };

    let err = verify_edit(&spec).unwrap_err();
    assert!(
        err.to_string().contains("shape mismatch"),
        "expected shape mismatch error, got: {err}",
    );
}

#[test]
fn test_edit_verification_is_conclusive_crown() {
    let v = EditVerification {
        diff_lower: -0.01,
        diff_upper: 0.01,
        max_abs_diff: 0.01,
        within_epsilon: true,
        epsilon: 0.05,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        dead_neuron_proof: None,
    };
    assert!(v.is_conclusive());
}

#[test]
fn test_edit_verification_not_conclusive_ibp() {
    let v = EditVerification {
        diff_lower: -0.01,
        diff_upper: 0.01,
        max_abs_diff: 0.01,
        within_epsilon: true,
        epsilon: 0.05,
        method: PropMethod::Ibp,
        crown_fallback_reason: Some("test".into()),
        soundness_mode: VerificationSoundnessMode::Heuristic,
        dead_neuron_proof: None,
    };
    assert!(!v.is_conclusive());
}

// F9: is_conclusive must accept all CROWN-family methods.
#[test]
fn test_edit_verification_is_conclusive_alpha_crown() {
    let v = EditVerification {
        diff_lower: -0.01,
        diff_upper: 0.01,
        max_abs_diff: 0.01,
        within_epsilon: true,
        epsilon: 0.05,
        method: PropMethod::AlphaCrown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        dead_neuron_proof: None,
    };
    assert!(v.is_conclusive(), "AlphaCrown should be conclusive");
}

#[test]
fn test_edit_verification_is_conclusive_beta_crown() {
    let v = EditVerification {
        diff_lower: -0.01,
        diff_upper: 0.01,
        max_abs_diff: 0.01,
        within_epsilon: true,
        epsilon: 0.05,
        method: PropMethod::BetaCrown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        dead_neuron_proof: None,
    };
    assert!(v.is_conclusive(), "BetaCrown should be conclusive");
}

#[test]
fn test_edit_verification_is_conclusive_analytical() {
    let v = EditVerification {
        diff_lower: -0.01,
        diff_upper: 0.01,
        max_abs_diff: 0.01,
        within_epsilon: true,
        epsilon: 0.05,
        method: PropMethod::Analytical,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        dead_neuron_proof: None,
    };
    assert!(v.is_conclusive(), "Analytical should be conclusive");
}

#[test]
fn test_edit_verification_serde_roundtrip() {
    let v = EditVerification {
        diff_lower: -0.005,
        diff_upper: 0.003,
        max_abs_diff: 0.005,
        within_epsilon: true,
        epsilon: 0.01,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        dead_neuron_proof: None,
    };
    let json = serde_json::to_string(&v).expect("serialize");
    let v2: EditVerification = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(v, v2);
}
