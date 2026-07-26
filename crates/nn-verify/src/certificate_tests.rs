// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for proof certificate format — v1 core behavior and pipeline integration.
//!
//! v2-specific tests (builder methods, v2 validation, v1 backward compat,
//! SHA-256 fingerprinting, serde roundtrips) are in `certificate_v2_tests.rs`.

use super::certificate_test_helpers::*;
use super::*;

// ---------------------------------------------------------------------------
// v1 tests (existing behavior preserved)
// ---------------------------------------------------------------------------

#[test]
fn test_certificate_from_verification() {
    let result = sample_verification();
    let input_spec = sample_input_spec();
    let cert = ProofCertificate::from_verification(&result, input_spec.clone());

    assert_eq!(cert.version, CERTIFICATE_VERSION);
    assert_eq!(cert.kernel_name, "snake");
    assert_eq!(cert.method, PropMethod::Ibp);
    assert!(cert.is_finite);
    assert_eq!(cert.soundness_mode, VerificationSoundnessMode::Sound);
    assert_eq!(cert.input_spec, input_spec);
    assert!(cert.output_tensor.is_some());
    assert!(cert.smt_outcome.is_none());
    assert!(!cert.generated_at.is_empty());
    // v2 fields default to None
    assert!(cert.layer_bounds.is_none());
    assert!(cert.kani_status.is_none());
    assert!(cert.weight_hash.is_none());
    assert!(cert.source_hash.is_none());
    assert!(cert.verifier_version.is_none());
}

#[test]
fn test_certificate_with_smt_outcome() {
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_smt_outcome("unexecuted");
    assert_eq!(cert.smt_outcome.as_deref(), Some("unexecuted"));
}

#[test]
fn test_certificate_validate_ok() {
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec());
    assert!(cert.validate().is_ok());
}

#[test]
fn test_certificate_validate_empty_name() {
    let mut result = sample_verification();
    result.kernel_name = String::new();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec());
    let err = cert.validate().unwrap_err();
    assert!(matches!(err, CertificateError::EmptyKernelName));
}

#[test]
fn test_certificate_validate_unsupported_version() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec());
    cert.version = 999;
    let err = cert.validate().unwrap_err();
    assert!(matches!(
        err,
        CertificateError::UnsupportedVersion { version: 999, .. }
    ));
}

#[test]
fn test_certificate_validate_finite_flag_mismatch() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec());
    // is_finite=true but set non-finite bounds
    cert.output_bounds.lower = f32::NEG_INFINITY;
    let err = cert.validate().unwrap_err();
    assert!(matches!(err, CertificateError::FiniteFlagMismatch { .. }));
}

#[test]
fn test_certificate_validate_inverted_bounds() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec());
    cert.output_bounds.lower = 10.0;
    cert.output_bounds.upper = -10.0;
    let err = cert.validate().unwrap_err();
    assert!(matches!(err, CertificateError::InvertedBounds { .. }));
}

#[test]
fn test_certificate_validate_non_finite_ok_when_not_finite() {
    let mut result = sample_verification();
    result.is_finite = false;
    result.output_lower = 0.0;
    result.output_upper = 0.0;
    result.output_width = 0.0;
    let cert = ProofCertificate::from_verification(&result, sample_input_spec());
    // Validation should pass — non-finite output is allowed when is_finite=false
    assert!(cert.validate().is_ok());
}

#[test]
fn test_certificate_json_roundtrip() {
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec());
    let json = cert.to_json().expect("serialize");
    let parsed: ProofCertificate = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(cert, parsed);
}

#[test]
fn test_bundle_new() {
    let bundle = CertificateBundle::new("test_model");
    assert_eq!(bundle.version, CERTIFICATE_VERSION);
    assert_eq!(bundle.model_name, "test_model");
    assert!(bundle.is_empty());
    assert_eq!(bundle.len(), 0);
    assert_eq!(bundle.verified_count(), 0);
    assert_eq!(bundle.sound_count(), 0);
}

#[test]
fn test_bundle_with_certificate() {
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec());
    let bundle = CertificateBundle::new("model").with_certificate(cert);
    assert_eq!(bundle.len(), 1);
    assert_eq!(bundle.verified_count(), 1);
    assert_eq!(bundle.sound_count(), 1);
}

#[test]
fn test_bundle_push() {
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec());
    let mut bundle = CertificateBundle::new("model");
    bundle.push(cert);
    assert_eq!(bundle.len(), 1);
}

#[test]
fn test_bundle_validate_all_ok() {
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec());
    let bundle = CertificateBundle::new("model").with_certificate(cert);
    assert!(bundle.validate_all().is_ok());
}

#[test]
fn test_bundle_validate_catches_invalid() {
    let mut result = sample_verification();
    result.kernel_name = String::new();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec());
    let bundle = CertificateBundle::new("model").with_certificate(cert);
    let (idx, err) = bundle.validate_all().unwrap_err();
    assert_eq!(idx, 0);
    assert!(matches!(err, CertificateError::EmptyKernelName));
}

#[test]
fn test_bundle_counts_mixed() {
    let sound_result = sample_verification();
    let mut heuristic_result = sample_verification();
    heuristic_result.kernel_name = "silu_mul".to_string();
    heuristic_result.soundness_mode = VerificationSoundnessMode::Heuristic;
    let mut non_finite_result = sample_verification();
    non_finite_result.kernel_name = "broken".to_string();
    non_finite_result.is_finite = false;
    non_finite_result.output_lower = 0.0;
    non_finite_result.output_upper = 0.0;

    let bundle = CertificateBundle::new("mixed")
        .with_certificate(ProofCertificate::from_verification(
            &sound_result,
            sample_input_spec(),
        ))
        .with_certificate(ProofCertificate::from_verification(
            &heuristic_result,
            sample_input_spec(),
        ))
        .with_certificate(ProofCertificate::from_verification(
            &non_finite_result,
            sample_input_spec(),
        ));

    assert_eq!(bundle.len(), 3);
    assert_eq!(bundle.verified_count(), 2); // 2 with is_finite=true
    assert_eq!(bundle.sound_count(), 2); // sound + non_finite_result defaults to Sound
}

#[test]
fn test_bundle_save_load_roundtrip() {
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_smt_outcome("unexecuted");
    let bundle = CertificateBundle::new("roundtrip_model").with_certificate(cert);

    let dir = std::env::temp_dir().join(format!("nn_cert_test_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test.proof.json");

    // Clean up any previous test artifact.
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(path.with_extension("proof.json.tmp"));

    bundle.save(&path).expect("save");
    let loaded = CertificateBundle::load(&path).expect("load");
    assert_eq!(bundle, loaded);

    // Clean up.
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

// ---------------------------------------------------------------------------
// output_width consistency validation
// ---------------------------------------------------------------------------

#[test]
fn test_certificate_validate_output_width_consistent() {
    // Normal case: width matches bounds difference.
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec());
    // output_width=20.0, bounds=(-9.704, 10.296), diff=20.0
    assert!(cert.validate().is_ok());
}

#[test]
fn test_certificate_validate_output_width_mismatch_detected() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec());
    // Tamper: set width to something clearly wrong.
    cert.output_width = 99.0;
    let err = cert.validate().unwrap_err();
    assert!(
        matches!(err, CertificateError::OutputWidthMismatch { .. }),
        "expected OutputWidthMismatch, got {err:?}"
    );
}

#[test]
fn test_certificate_validate_output_width_nan_with_finite_bounds() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec());
    cert.output_width = f32::NAN;
    let err = cert.validate().unwrap_err();
    assert!(
        matches!(err, CertificateError::NonFiniteOutputWidth { .. }),
        "expected NonFiniteOutputWidth, got {err:?}"
    );
}

#[test]
fn test_certificate_validate_output_width_inf_with_finite_bounds() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec());
    cert.output_width = f32::INFINITY;
    let err = cert.validate().unwrap_err();
    assert!(
        matches!(err, CertificateError::NonFiniteOutputWidth { .. }),
        "expected NonFiniteOutputWidth, got {err:?}"
    );
}

#[test]
fn test_certificate_validate_output_width_skipped_for_non_finite_bounds() {
    // When bounds are non-finite, output_width check is skipped.
    let mut result = sample_verification();
    result.is_finite = false;
    result.output_lower = 0.0;
    result.output_upper = 0.0;
    result.output_width = f32::MAX;
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec());
    cert.is_finite = false;
    cert.output_bounds.lower = f32::NEG_INFINITY;
    cert.output_bounds.upper = f32::INFINITY;
    // Width doesn't match bounds, but bounds are non-finite so check is skipped.
    cert.output_width = 42.0;
    assert!(cert.validate().is_ok());
}

#[test]
fn test_certificate_validate_output_width_small_rounding_ok() {
    // Allow small floating-point rounding differences.
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec());
    // Add tiny rounding error (well within tolerance).
    let expected = cert.output_bounds.upper - cert.output_bounds.lower;
    cert.output_width = expected + 1e-7;
    assert!(cert.validate().is_ok());
}

// ---------------------------------------------------------------------------
// certificate_from_pipeline tests
// ---------------------------------------------------------------------------

#[test]
fn test_certificate_from_pipeline_single_variable() {
    let result = sample_verification();
    let variable_inputs = vec![ParamInputRecord {
        param_index: 0,
        lower: -10.0,
        upper: 10.0,
    }];
    let constant_params = vec![1.0];

    let cert = certificate_from_pipeline(&result, &variable_inputs, &constant_params, None);
    assert_eq!(cert.kernel_name, "snake");
    assert_eq!(cert.input_spec.variable_inputs.len(), 1);
    assert_eq!(cert.input_spec.constant_params, vec![1.0]);
    assert_eq!(cert.input_spec.input_range, Some((-10.0, 10.0)));
    assert!(cert.smt_outcome.is_none());
    cert.validate()
        .expect("single-variable pipeline cert should pass validate()");
}

#[test]
fn test_certificate_from_pipeline_with_smt() {
    let result = sample_verification();
    let variable_inputs = vec![ParamInputRecord {
        param_index: 0,
        lower: -10.0,
        upper: 10.0,
    }];
    let cert = certificate_from_pipeline(&result, &variable_inputs, &[1.0], Some("proven"));
    assert_eq!(cert.smt_outcome.as_deref(), Some("proven"));
    cert.validate()
        .expect("smt pipeline cert should pass validate()");
}

#[test]
fn test_certificate_from_pipeline_multi_variable() {
    let mut result = sample_verification();
    result.kernel_name = "rope_cos".to_string();
    let variable_inputs = vec![
        ParamInputRecord {
            param_index: 0,
            lower: -1.0,
            upper: 1.0,
        },
        ParamInputRecord {
            param_index: 1,
            lower: 0.0,
            upper: 6.3,
        },
    ];

    let cert = certificate_from_pipeline(&result, &variable_inputs, &[], None);
    assert_eq!(cert.input_spec.variable_inputs.len(), 2);
    assert!(cert.input_spec.input_range.is_none()); // Multi-variable: no legacy range
    assert_eq!(cert.input_spec.input_shape, Some(vec![2]));
    cert.validate()
        .expect("multi-variable pipeline cert should pass validate()");
}

// ---------------------------------------------------------------------------
// Edge case validation tests
// ---------------------------------------------------------------------------

#[test]
fn test_certificate_validate_nan_bounds_not_finite() {
    // Regression: NaN bounds with is_finite=true should fail validate().
    // IEEE 754: NaN > NaN is false, which could bypass inverted-bounds check.
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec());
    cert.output_bounds.lower = f32::NAN;
    cert.is_finite = true;
    let err = cert.validate().unwrap_err();
    assert!(
        matches!(err, CertificateError::FiniteFlagMismatch { .. }),
        "NaN bounds with is_finite=true should be caught: {err:?}"
    );
}

#[test]
fn test_certificate_validate_version_zero_rejected() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec());
    cert.version = 0;
    let err = cert.validate().unwrap_err();
    assert!(matches!(
        err,
        CertificateError::UnsupportedVersion { version: 0, .. }
    ));
}

#[test]
fn test_certificate_validate_both_bounds_nan() {
    // IEEE 754: NaN > NaN is false, NaN < NaN is false.
    // Both bounds NaN with is_finite=false should pass (non-finite is allowed).
    let mut result = sample_verification();
    result.is_finite = false;
    result.output_lower = 0.0;
    result.output_upper = 0.0;
    result.output_width = 0.0;
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec());
    cert.output_bounds.lower = f32::NAN;
    cert.output_bounds.upper = f32::NAN;
    cert.output_width = f32::NAN;
    // Both NaN with is_finite=false: validate skips width check (non-finite bounds),
    // and inverted-bounds check returns false (NaN > NaN is false).
    assert!(cert.validate().is_ok());
}

#[test]
fn test_certificate_validate_both_bounds_nan_with_finite_flag() {
    // Both NaN with is_finite=true should be caught.
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec());
    cert.output_bounds.lower = f32::NAN;
    cert.output_bounds.upper = f32::NAN;
    let err = cert.validate().unwrap_err();
    assert!(matches!(err, CertificateError::FiniteFlagMismatch { .. }));
}

#[test]
fn test_certificate_validate_neg_inf_lower_inf_upper() {
    // Full range (-inf, +inf) with is_finite=false is valid.
    let mut result = sample_verification();
    result.is_finite = false;
    result.output_lower = 0.0;
    result.output_upper = 0.0;
    result.output_width = 0.0;
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec());
    cert.output_bounds.lower = f32::NEG_INFINITY;
    cert.output_bounds.upper = f32::INFINITY;
    cert.output_width = f32::INFINITY;
    assert!(cert.validate().is_ok());
}

#[test]
fn test_certificate_validate_equal_bounds_zero_width() {
    // Degenerate case: bounds are equal → width must be 0.
    let mut result = sample_verification();
    result.output_lower = 5.0;
    result.output_upper = 5.0;
    result.output_width = 0.0;
    let cert = ProofCertificate::from_verification(&result, sample_input_spec());
    assert!(cert.validate().is_ok());
}

// ---------------------------------------------------------------------------
// Pipeline edge cases (proof_coverage gap)
// ---------------------------------------------------------------------------

/// `certificate_from_pipeline` with 0 variable inputs produces
/// `input_shape=None` and `input_range=None`.
#[test]
fn test_certificate_from_pipeline_zero_variables() {
    let result = sample_verification();
    let cert = certificate_from_pipeline(&result, &[], &[1.0, 2.0], None);
    assert!(cert.input_spec.variable_inputs.is_empty());
    assert_eq!(cert.input_spec.constant_params, vec![1.0, 2.0]);
    assert!(
        cert.input_spec.input_shape.is_none(),
        "0 variables should produce input_shape=None"
    );
    assert!(
        cert.input_spec.input_range.is_none(),
        "0 variables should produce input_range=None"
    );
    cert.validate()
        .expect("zero-variable pipeline cert should pass validate()");
}

/// Negative output_width with valid finite bounds should fail validation.
#[test]
fn test_certificate_validate_negative_output_width() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec());
    cert.output_width = -5.0;
    let err = cert.validate().unwrap_err();
    assert!(
        matches!(err, CertificateError::OutputWidthMismatch { .. }),
        "negative output_width should be caught: {err:?}"
    );
}

/// Width overflow: both bounds finite but (upper - lower) overflows to Infinity.
/// IEEE 754: Inf > Inf is false. Without the overflow guard, the comparison
/// `abs_diff > rel_threshold` evaluates to `Inf > Inf → false`, silently
/// accepting any finite output_width. Part of #2441.
#[test]
fn test_certificate_validate_output_width_overflow_detected() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec());
    // Both bounds finite, but subtraction overflows: 2e38 - (-2e38) = 4e38 > f32::MAX.
    cert.output_bounds.lower = -2.0e38;
    cert.output_bounds.upper = 2.0e38;
    cert.output_width = 1.0e30; // Finite (within f32 range), but wrong — real width overflows
    cert.is_finite = true;
    let err = cert.validate().unwrap_err();
    assert!(
        matches!(err, CertificateError::OutputWidthMismatch { .. }),
        "width overflow must be detected, got: {err:?}"
    );
}
