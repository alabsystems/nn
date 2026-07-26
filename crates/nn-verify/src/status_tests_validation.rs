// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Input/output validation tests: non-finite input metadata rejection (#132),
//! non-finite output_width clamping (#173), non-finite tensor bounds
//! sanitization (#198), inverted bounds rejection, degenerate interval
//! acceptance.

use super::*;
use crate::verify_input::ScalarInputBounds;

// --- Non-finite input metadata rejection tests (#132) ---

fn make_finite_verification(name: &str) -> KernelVerification {
    KernelVerification {
        kernel_name: name.to_string(),
        method: PropMethod::Ibp,
        output_lower: -1.0,
        output_upper: 1.0,
        output_width: 2.0,
        is_finite: true,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: None,
    }
}

#[test]
fn test_record_rejects_nan_lower_bound() {
    let err = ScalarInputBounds::new(f32::NAN, 1.0).unwrap_err();
    assert!(
        matches!(err, VerifyError::InvalidInputBounds { .. }),
        "expected InvalidInputBounds, got {err:?}"
    );
}

#[test]
fn test_record_rejects_nan_upper_bound() {
    let err = ScalarInputBounds::new(-1.0, f32::NAN).unwrap_err();
    assert!(matches!(err, VerifyError::InvalidInputBounds { .. }));
}

#[test]
fn test_record_rejects_inf_lower_bound() {
    let err = ScalarInputBounds::new(f32::NEG_INFINITY, 1.0).unwrap_err();
    assert!(matches!(err, VerifyError::InvalidInputBounds { .. }));
}

#[test]
fn test_record_rejects_inf_upper_bound() {
    let err = ScalarInputBounds::new(-1.0, f32::INFINITY).unwrap_err();
    assert!(matches!(err, VerifyError::InvalidInputBounds { .. }));
}

#[test]
fn test_record_rejects_nan_constant_param() {
    let mut status = VerifyStatus::default();
    let result = make_finite_verification("nan_const");
    let err = status
        .record(
            &result,
            ScalarInputBounds::new(-1.0, 1.0).expect("valid test bounds"),
            &[f32::NAN],
            None,
        )
        .unwrap_err();
    assert!(matches!(err, VerifyError::NonFiniteInputMetadata { .. }));
}

#[test]
fn test_record_rejects_inf_constant_param() {
    let mut status = VerifyStatus::default();
    let result = make_finite_verification("inf_const");
    let err = status
        .record(
            &result,
            ScalarInputBounds::new(-1.0, 1.0).expect("valid test bounds"),
            &[1.0, f32::INFINITY],
            None,
        )
        .unwrap_err();
    assert!(matches!(err, VerifyError::NonFiniteInputMetadata { .. }));
}

#[test]
fn test_record_failure_rejects_nan_input() {
    let status = VerifyStatus::default();
    let err = ScalarInputBounds::new(f32::NAN, 1.0).unwrap_err();
    assert!(matches!(err, VerifyError::InvalidInputBounds { .. }));
    assert!(status.kernel_count() == 0);
}

#[test]
fn test_record_failure_rejects_inf_constant() {
    let mut status = VerifyStatus::default();
    let err = status
        .record_failure(
            "bad",
            PropMethod::Ibp,
            ScalarInputBounds::new(-1.0, 1.0).expect("valid test bounds"),
            &[f32::NEG_INFINITY],
        )
        .unwrap_err();
    assert!(matches!(err, VerifyError::NonFiniteInputMetadata { .. }));
}

#[test]
fn test_record_with_variable_inputs_rejects_nan() {
    let mut status = VerifyStatus::default();
    let result = make_finite_verification("var_nan");
    let variable_inputs = vec![
        ParamInputRecord {
            param_index: 0,
            lower: -1.0,
            upper: 1.0,
        },
        ParamInputRecord {
            param_index: 1,
            lower: f32::NAN,
            upper: 2.0,
        },
    ];
    let err = status
        .record_with_variable_inputs(&result, &variable_inputs, &[], None, None)
        .unwrap_err();
    assert!(
        matches!(err, VerifyError::NonFiniteInputMetadata { .. }),
        "expected NonFiniteInputMetadata for NaN in variable_inputs[1].lower, got {err:?}"
    );
}

#[test]
fn test_record_failure_with_variable_inputs_rejects_inf() {
    let mut status = VerifyStatus::default();
    let variable_inputs = vec![ParamInputRecord {
        param_index: 0,
        lower: -1.0,
        upper: f32::INFINITY,
    }];
    let err = status
        .record_failure_with_variable_inputs("bad", PropMethod::Ibp, &variable_inputs, &[], None)
        .unwrap_err();
    assert!(matches!(err, VerifyError::NonFiniteInputMetadata { .. }));
}

#[test]
fn test_record_accepts_finite_values() {
    let mut status = VerifyStatus::default();
    let result = make_finite_verification("finite_ok");
    status
        .record(
            &result,
            ScalarInputBounds::new(-1e30, 1e30).expect("valid test bounds"),
            &[0.0, -0.0, 1e-38],
            None,
        )
        .expect("finite values should be accepted");
    assert_eq!(status.kernel_count(), 1);
}

#[test]
fn test_record_rejection_does_not_mutate_state() {
    let mut status = VerifyStatus::default();
    let good = make_finite_verification("good_kernel");
    status
        .record(
            &good,
            ScalarInputBounds::new(-1.0, 1.0).expect("valid test bounds"),
            &[],
            None,
        )
        .expect("record good");
    assert_eq!(status.kernel_count(), 1);
    assert_eq!(status.run_count("good_kernel"), 1);

    // NaN bounds are now caught at ScalarInputBounds construction, so record() never called.
    let _ = ScalarInputBounds::new(f32::NAN, 1.0);

    assert_eq!(
        status.kernel_count(),
        1,
        "rejected record must not add entry"
    );
    assert!(!status.has_kernel("bad_kernel"));
    assert_eq!(
        status.run_count("good_kernel"),
        1,
        "good kernel history unchanged"
    );
}

// --- Non-finite output_width serialization guard (#173) ---

#[test]
fn test_non_finite_output_width_clamped_to_max_for_serialization() {
    let mut status = VerifyStatus::default();
    let result = KernelVerification {
        kernel_name: "inf_width".to_string(),
        method: PropMethod::Ibp,
        output_lower: f32::NEG_INFINITY,
        output_upper: f32::INFINITY,
        output_width: f32::INFINITY,
        is_finite: false,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: None,
    };

    status
        .record(
            &result,
            ScalarInputBounds::new(-1.0, 1.0).expect("valid test bounds"),
            &[],
            None,
        )
        .expect("record");

    let k = &status.kernels["inf_width"];
    assert!(
        k.output_width.is_finite(),
        "output_width must be finite for serde_json, got {}",
        k.output_width
    );
    assert_eq!(
        k.output_width,
        f32::MAX,
        "non-finite width should use f32::MAX sentinel"
    );

    // Verify serialization succeeds (would panic on NaN/Infinity).
    let json = serde_json::to_string_pretty(&status)
        .expect("serialize must succeed with guarded output_width");
    assert!(json.contains("inf_width"));
}

#[test]
fn test_nan_output_width_clamped_to_max_for_serialization() {
    let mut status = VerifyStatus::default();
    let result = KernelVerification {
        kernel_name: "nan_width".to_string(),
        method: PropMethod::Ibp,
        output_lower: f32::NAN,
        output_upper: f32::NAN,
        output_width: f32::NAN,
        is_finite: false,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: None,
    };

    status
        .record(
            &result,
            ScalarInputBounds::new(-1.0, 1.0).expect("valid test bounds"),
            &[],
            None,
        )
        .expect("record");

    let k = &status.kernels["nan_width"];
    assert!(
        k.output_width.is_finite(),
        "NaN output_width must be clamped, got {}",
        k.output_width
    );
    assert_eq!(k.output_width, f32::MAX);
}

/// Regression: non-finite tensor bounds must be sanitized to 0.0 sentinels
/// by `OutputBoundsRecord::from_verification`. Guards added in 94efb7e.
#[test]
fn test_from_verification_sanitizes_non_finite_tensor_bounds() {
    use crate::verify_types::OutputTensorBounds;

    let result = KernelVerification {
        kernel_name: "nan_tensor".to_string(),
        method: PropMethod::Ibp,
        output_lower: 0.0,
        output_upper: 1.0,
        output_width: 1.0,
        is_finite: true,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: Some(OutputTensorBounds {
            lower: vec![-1.0, f32::NAN, f32::NEG_INFINITY],
            upper: vec![1.0, f32::INFINITY, f32::NAN],
            shape: vec![3],
            finite_mask: vec![],
        }),
    };

    let record = OutputBoundsRecord::from_verification(&result);

    // tensor_lower: NaN → 0.0, -Inf → 0.0
    let tl = record.tensor_lower.expect("should have tensor_lower");
    assert!(tl[0].is_finite(), "finite value preserved");
    assert_eq!(tl[1], 0.0, "NaN in tensor_lower must be sanitized to 0.0");
    assert_eq!(tl[2], 0.0, "-Inf in tensor_lower must be sanitized to 0.0");

    // tensor_upper: Inf → 0.0, NaN → 0.0
    let tu = record.tensor_upper.expect("should have tensor_upper");
    assert!(tu[0].is_finite(), "finite value preserved");
    assert_eq!(tu[1], 0.0, "Inf in tensor_upper must be sanitized to 0.0");
    assert_eq!(tu[2], 0.0, "NaN in tensor_upper must be sanitized to 0.0");
}

// --- Non-finite tensor bounds serialization roundtrip (#198) ---

#[test]
fn test_non_finite_tensor_bounds_sanitized_for_serialization() {
    use crate::verify_types::OutputTensorBounds;

    let mut status = VerifyStatus::default();
    let result = KernelVerification {
        kernel_name: "nan_tensor_roundtrip".to_string(),
        method: PropMethod::Ibp,
        output_lower: -1.0,
        output_upper: 1.0,
        output_width: 2.0,
        is_finite: true,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: Some(OutputTensorBounds {
            lower: vec![-1.0, f32::NAN, f32::NEG_INFINITY, -0.5],
            upper: vec![1.0, f32::INFINITY, f32::NAN, 0.5],
            shape: vec![4],
            finite_mask: vec![],
        }),
    };

    status
        .record(
            &result,
            ScalarInputBounds::new(-1.0, 1.0).expect("valid test bounds"),
            &[],
            None,
        )
        .expect("record");

    let k = &status.kernels["nan_tensor_roundtrip"];
    let tl = k.output_bounds.tensor_lower.as_ref().expect("tensor_lower");
    let tu = k.output_bounds.tensor_upper.as_ref().expect("tensor_upper");

    // All elements must be finite (NaN/Inf replaced with 0.0 sentinel).
    for (i, &v) in tl.iter().enumerate() {
        assert!(v.is_finite(), "tensor_lower[{i}] = {v} is not finite");
    }
    for (i, &v) in tu.iter().enumerate() {
        assert!(v.is_finite(), "tensor_upper[{i}] = {v} is not finite");
    }

    // Finite values preserved, non-finite replaced with 0.0.
    assert_eq!(tl, &[-1.0, 0.0, 0.0, -0.5]);
    assert_eq!(tu, &[1.0, 0.0, 0.0, 0.5]);

    // Serialization must succeed (serde_json rejects NaN/Infinity).
    let json = serde_json::to_string_pretty(&status)
        .expect("serialize must succeed with sanitized tensor bounds");
    assert!(json.contains("nan_tensor_roundtrip"));
    assert!(json.contains("tensor_lower"));

    // Roundtrip preserves sanitized values.
    let deserialized: VerifyStatus = serde_json::from_str(&json).expect("deserialize");
    let k2 = &deserialized.kernels["nan_tensor_roundtrip"];
    assert_eq!(
        k2.output_bounds
            .tensor_lower
            .as_ref()
            .expect("tensor_lower"),
        &[-1.0, 0.0, 0.0, -0.5]
    );
    assert_eq!(
        k2.output_bounds
            .tensor_upper
            .as_ref()
            .expect("tensor_upper"),
        &[1.0, 0.0, 0.0, 0.5]
    );
}

#[test]
fn test_record_rejects_inverted_variable_input_bounds() {
    let mut status = VerifyStatus::default();
    let result = KernelVerification {
        kernel_name: "inverted_bounds_kernel".to_string(),
        method: PropMethod::Ibp,
        output_lower: -1.0,
        output_upper: 1.0,
        output_width: 2.0,
        is_finite: true,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: None,
    };
    let variable_inputs = [ParamInputRecord {
        param_index: 0,
        lower: 10.0, // inverted: lower > upper
        upper: -10.0,
    }];
    let err = status
        .record_with_variable_inputs(&result, &variable_inputs, &[], None, None)
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("lower") && msg.contains("upper"),
        "inverted bounds should be rejected with a clear message, got: {msg}"
    );
}

/// Degenerate single-point interval (lower == upper) is valid and must NOT be rejected.
/// Guards against a `>` → `>=` regression in validate_input_metadata.
#[test]
fn test_record_accepts_degenerate_single_point_interval() {
    let mut status = VerifyStatus::default();
    let result = KernelVerification {
        kernel_name: "degenerate_interval_kernel".to_string(),
        method: PropMethod::Ibp,
        output_lower: 0.0,
        output_upper: 0.0,
        output_width: 0.0,
        is_finite: true,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: None,
    };
    let variable_inputs = [ParamInputRecord {
        param_index: 0,
        lower: 5.0,
        upper: 5.0, // degenerate but valid: lower == upper
    }];
    status
        .record_with_variable_inputs(&result, &variable_inputs, &[], None, None)
        .expect("lower == upper (single-point interval) should be accepted");
}
