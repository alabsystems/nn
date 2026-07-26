// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani verification harnesses for `certificate.rs`.
//!
//! Proves structural and correctness properties of:
//! - `ProofCertificate::validate`: version, name, bounds, width consistency
//! - `ProofCertificate::from_verification`: field mapping from KernelVerification
//! - Builder pattern: with_smt_outcome, with_layer_bounds, with_kani_status, etc.
//! - IEEE 754: NaN/Inf bounds handling, inverted bounds detection
//! - v2 validations: layer index ordering, hash format
//! - v4 validations: content_hash, hmac_signature format
//! - `CERTIFICATE_VERSION` correctness
//!
//! Part of #3708.

use super::{CertificateError, LayerBoundRecord, ProofCertificate, CERTIFICATE_VERSION};
use crate::certificate_types::{KaniOutcome, KaniProofRecord, PrecisionModel};
use crate::soundness_compat::VerificationSoundnessMode;
use crate::status::{InputBoundsRecord, OutputBoundsRecord, ParamInputRecord, SmtProofVerdict};
use crate::verify_types::{KernelVerification, PropMethod};

/// A valid 64-char hex SHA-256 hash for test fixtures.
const VALID_HASH: &str = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2";

/// Helper: create a minimal valid `ProofCertificate` for testing.
fn valid_cert() -> ProofCertificate {
    let kv = KernelVerification::new(
        "test_kernel".to_string(),
        PropMethod::Crown,
        -1.0,
        1.0,
        2.0,
        true,
    );
    let input_spec = InputBoundsRecord::new(&[ParamInputRecord::new(0, -1.0, 1.0)], &[]);
    ProofCertificate::from_verification(&kv, input_spec)
}

// ===========================================================================
// CERTIFICATE_VERSION
// ===========================================================================

// ---------------------------------------------------------------------------
// 1. CERTIFICATE_VERSION is 5
// ---------------------------------------------------------------------------

/// Prove: `CERTIFICATE_VERSION` is 5 (current version).
#[kani::unwind(1)]
#[kani::proof]
fn certificate_version_is_5() {
    assert_eq!(CERTIFICATE_VERSION, 5, "current version must be 5");
}

// ---------------------------------------------------------------------------
// 3. from_verification preserves kernel_name
// ---------------------------------------------------------------------------

/// Prove: `from_verification` preserves the kernel name.
#[kani::unwind(64)]
#[kani::proof]
fn from_verification_preserves_kernel_name() {
    let cert = valid_cert();
    assert_eq!(cert.kernel_name, "test_kernel");
}

// ---------------------------------------------------------------------------
// 4. from_verification preserves method
// ---------------------------------------------------------------------------

/// Prove: `from_verification` preserves the propagation method.
#[kani::unwind(64)]
#[kani::proof]
fn from_verification_preserves_method() {
    let cert = valid_cert();
    assert_eq!(cert.method, PropMethod::Crown);
}

// ---------------------------------------------------------------------------
// 5. from_verification preserves output_width
// ---------------------------------------------------------------------------

/// Prove: `from_verification` preserves the output width.
#[kani::unwind(64)]
#[kani::proof]
fn from_verification_preserves_output_width() {
    let cert = valid_cert();
    assert_eq!(cert.output_width, 2.0_f32);
}

// ---------------------------------------------------------------------------
// 6. from_verification preserves is_finite
// ---------------------------------------------------------------------------

/// Prove: `from_verification` preserves the is_finite flag.
#[kani::unwind(64)]
#[kani::proof]
fn from_verification_preserves_is_finite() {
    let cert = valid_cert();
    assert!(cert.is_finite, "is_finite must be true for finite bounds");
}

// ---------------------------------------------------------------------------
// 7. from_verification preserves soundness_mode
// ---------------------------------------------------------------------------

/// Prove: `from_verification` preserves the soundness mode.
#[kani::unwind(64)]
#[kani::proof]
fn from_verification_preserves_soundness_mode() {
    let cert = valid_cert();
    assert_eq!(cert.soundness_mode, VerificationSoundnessMode::Sound);
}

// ---------------------------------------------------------------------------
// 8. from_verification: optional v2 fields default to None
// ---------------------------------------------------------------------------

/// Prove: all v2 optional fields default to `None`.
#[kani::unwind(64)]
#[kani::proof]
fn from_verification_v2_fields_none() {
    let cert = valid_cert();
    assert!(cert.layer_bounds.is_none());
    assert!(cert.kani_status.is_none());
    assert!(cert.weight_hash.is_none());
    assert!(cert.source_hash.is_none());
    assert!(cert.verifier_version.is_none());
    assert!(cert.crown_coverage.is_none());
    assert!(cert.ibp_fallback_count.is_none());
}

// ---------------------------------------------------------------------------
// 9. from_verification: optional v3 fields default to None
// ---------------------------------------------------------------------------

/// Prove: all v3 optional fields default to `None`.
#[kani::unwind(64)]
#[kani::proof]
fn from_verification_v3_fields_none() {
    let cert = valid_cert();
    assert!(cert.smt_proof_alethe.is_none());
    assert!(cert.smt_proof_verdict.is_none());
}

// ---------------------------------------------------------------------------
// 10. from_verification: optional v4 fields default to None
// ---------------------------------------------------------------------------

/// Prove: all v4 optional fields default to `None`.
#[kani::unwind(64)]
#[kani::proof]
fn from_verification_v4_fields_none() {
    let cert = valid_cert();
    assert!(cert.content_hash.is_none());
    assert!(cert.hmac_signature.is_none());
}

// ---------------------------------------------------------------------------
// 11. from_verification: optional v5 fields default to None
// ---------------------------------------------------------------------------

/// Prove: v5 `precision_model` defaults to `None`.
#[kani::unwind(64)]
#[kani::proof]
fn from_verification_v5_fields_none() {
    let cert = valid_cert();
    assert!(cert.precision_model.is_none());
}

// ===========================================================================
// ProofCertificate::validate: version
// ===========================================================================

// ---------------------------------------------------------------------------
// 12. validate: version 0 rejected
// ---------------------------------------------------------------------------

/// Prove: version 0 is rejected by validate().
#[kani::unwind(64)]
#[kani::proof]
fn validate_version_0_rejected() {
    let mut cert = valid_cert();
    cert.version = 0;
    assert!(cert.validate().is_err(), "version 0 must be rejected");
}

// ---------------------------------------------------------------------------
// 14. validate: current version accepted
// ---------------------------------------------------------------------------

/// Prove: current version passes validation.
#[kani::unwind(64)]
#[kani::proof]
fn validate_current_version_accepted() {
    let cert = valid_cert();
    assert!(cert.validate().is_ok(), "current version must pass");
}

// ---------------------------------------------------------------------------
// 15. validate: version 1 accepted (backward compatibility)
// ---------------------------------------------------------------------------

/// Prove: version 1 passes validation.
#[kani::unwind(64)]
#[kani::proof]
fn validate_version_1_accepted() {
    let mut cert = valid_cert();
    cert.version = 1;
    assert!(cert.validate().is_ok(), "version 1 must be accepted");
}

// ---------------------------------------------------------------------------
// 18. validate: NaN bounds with is_finite=true rejected
// ---------------------------------------------------------------------------

/// Prove: NaN lower bound with is_finite=true is rejected.
#[kani::unwind(64)]
#[kani::proof]
fn validate_nan_lower_finite_flag_rejected() {
    let mut cert = valid_cert();
    cert.output_bounds.lower = f32::NAN;
    cert.is_finite = true;
    assert!(
        cert.validate().is_err(),
        "NaN with is_finite=true must be rejected"
    );
}

// ---------------------------------------------------------------------------
// 19. validate: Inf upper bound with is_finite=true rejected
// ---------------------------------------------------------------------------

/// Prove: +Inf upper bound with is_finite=true is rejected.
#[kani::unwind(64)]
#[kani::proof]
fn validate_inf_upper_finite_flag_rejected() {
    let mut cert = valid_cert();
    cert.output_bounds.upper = f32::INFINITY;
    cert.is_finite = true;
    assert!(
        cert.validate().is_err(),
        "Inf with is_finite=true must be rejected"
    );
}

// ---------------------------------------------------------------------------
// 20. validate: equal bounds (point interval) accepted
// ---------------------------------------------------------------------------

/// Prove: lower == upper (point interval) passes validation.
#[kani::unwind(64)]
#[kani::proof]
fn validate_equal_bounds_accepted() {
    let kv = KernelVerification::new(
        "point".to_string(),
        PropMethod::Analytical,
        3.0,
        3.0,
        0.0,
        true,
    );
    let input_spec = InputBoundsRecord::new(&[ParamInputRecord::new(0, 0.0, 1.0)], &[]);
    let cert = ProofCertificate::from_verification(&kv, input_spec);
    assert!(cert.validate().is_ok(), "equal bounds must pass");
}

// ===========================================================================
// ProofCertificate::validate: output_width consistency
// ===========================================================================

// ---------------------------------------------------------------------------
// 21. validate: output_width matches upper - lower for valid cert
// ---------------------------------------------------------------------------

/// Prove: valid certificate output_width matches bounds difference.
#[kani::unwind(64)]
#[kani::proof]
fn validate_width_matches_bounds() {
    let cert = valid_cert();
    let expected_width = cert.output_bounds.upper - cert.output_bounds.lower;
    let diff = (cert.output_width - expected_width).abs();
    assert!(diff < 1e-6, "width must match upper - lower");
    assert!(cert.validate().is_ok());
}

// ---------------------------------------------------------------------------
// 22. validate: mismatched output_width rejected
// ---------------------------------------------------------------------------

/// Prove: output_width that doesn't match bounds is rejected.
#[kani::unwind(64)]
#[kani::proof]
fn validate_mismatched_width_rejected() {
    let mut cert = valid_cert();
    // cert has bounds [-1, 1] and width 2.0. Change width to something wrong.
    cert.output_width = 999.0;
    assert!(
        cert.validate().is_err(),
        "mismatched width must be rejected"
    );
}

// ===========================================================================
// ProofCertificate::validate: v2 layer_bounds
// ===========================================================================

// ---------------------------------------------------------------------------
// 23. validate: empty layer_bounds rejected
// ---------------------------------------------------------------------------

/// Prove: `Some(vec![])` layer_bounds is rejected.
#[kani::unwind(128)]
#[kani::proof]
fn validate_empty_layer_bounds_rejected() {
    let mut cert = valid_cert();
    cert.layer_bounds = Some(vec![]);
    assert!(
        cert.validate().is_err(),
        "empty layer_bounds must be rejected"
    );
}

// ---------------------------------------------------------------------------
// 24. validate: layer_bounds with correct indices accepted
// ---------------------------------------------------------------------------

/// Prove: layer_bounds with sequential indices passes.
#[kani::unwind(128)]
#[kani::proof]
fn validate_sequential_layer_bounds_accepted() {
    let mut cert = valid_cert();
    cert.layer_bounds = Some(vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-1.0, 1.0)],
            output_bounds: vec![(-2.0, 2.0)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: None,
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-2.0, 2.0)],
            output_bounds: vec![(0.0, 2.0)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: None,
        },
    ]);
    assert!(cert.validate().is_ok(), "sequential layer bounds must pass");
}

// ---------------------------------------------------------------------------
// 25. validate: layer_bounds with wrong index rejected
// ---------------------------------------------------------------------------

/// Prove: layer_bounds with non-sequential index is rejected.
#[kani::unwind(128)]
#[kani::proof]
fn validate_wrong_layer_index_rejected() {
    let mut cert = valid_cert();
    cert.layer_bounds = Some(vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-1.0, 1.0)],
            output_bounds: vec![(-2.0, 2.0)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: None,
        },
        LayerBoundRecord {
            layer_index: 5, // wrong: should be 1
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-2.0, 2.0)],
            output_bounds: vec![(0.0, 2.0)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: None,
        },
    ]);
    assert!(
        cert.validate().is_err(),
        "wrong layer index must be rejected"
    );
}

// ===========================================================================
// ProofCertificate::validate: v2 hash format
// ===========================================================================

// ---------------------------------------------------------------------------
// 26. validate: valid weight_hash accepted
// ---------------------------------------------------------------------------

/// Prove: valid SHA-256 weight_hash passes validation.
#[kani::unwind(64)]
#[kani::proof]
fn validate_valid_weight_hash_accepted() {
    let mut cert = valid_cert();
    cert.weight_hash = Some(VALID_HASH.to_string());
    assert!(cert.validate().is_ok());
}

// ---------------------------------------------------------------------------
// 27. validate: invalid weight_hash rejected
// ---------------------------------------------------------------------------

/// Prove: non-hex weight_hash is rejected.
#[kani::unwind(64)]
#[kani::proof]
fn validate_invalid_weight_hash_rejected() {
    let mut cert = valid_cert();
    cert.weight_hash = Some("not_a_valid_hash".to_string());
    assert!(
        cert.validate().is_err(),
        "invalid weight_hash must be rejected"
    );
}

// ---------------------------------------------------------------------------
// 28. validate: valid source_hash accepted
// ---------------------------------------------------------------------------

/// Prove: valid SHA-256 source_hash passes validation.
#[kani::unwind(64)]
#[kani::proof]
fn validate_valid_source_hash_accepted() {
    let mut cert = valid_cert();
    cert.source_hash = Some(VALID_HASH.to_string());
    assert!(cert.validate().is_ok());
}

// ---------------------------------------------------------------------------
// 29. validate: invalid source_hash rejected
// ---------------------------------------------------------------------------

/// Prove: short source_hash is rejected.
#[kani::unwind(64)]
#[kani::proof]
fn validate_invalid_source_hash_rejected() {
    let mut cert = valid_cert();
    cert.source_hash = Some("abc123".to_string());
    assert!(
        cert.validate().is_err(),
        "short source_hash must be rejected"
    );
}

// ===========================================================================
// ProofCertificate::validate: v4 integrity fields
// ===========================================================================

// ---------------------------------------------------------------------------
// 30. validate: valid content_hash accepted
// ---------------------------------------------------------------------------

/// Prove: valid SHA-256 content_hash passes validation.
#[kani::unwind(64)]
#[kani::proof]
fn validate_valid_content_hash_accepted() {
    let mut cert = valid_cert();
    cert.content_hash = Some(VALID_HASH.to_string());
    assert!(cert.validate().is_ok());
}

// ---------------------------------------------------------------------------
// 31. validate: invalid content_hash rejected
// ---------------------------------------------------------------------------

/// Prove: non-hex content_hash is rejected.
#[kani::unwind(64)]
#[kani::proof]
fn validate_invalid_content_hash_rejected() {
    let mut cert = valid_cert();
    cert.content_hash =
        Some("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ".to_string());
    assert!(
        cert.validate().is_err(),
        "non-hex content_hash must be rejected"
    );
}

// ---------------------------------------------------------------------------
// 32. validate: invalid hmac_signature rejected
// ---------------------------------------------------------------------------

/// Prove: short hmac_signature is rejected.
#[kani::unwind(64)]
#[kani::proof]
fn validate_invalid_hmac_rejected() {
    let mut cert = valid_cert();
    cert.hmac_signature = Some("tooshort".to_string());
    assert!(cert.validate().is_err(), "short hmac must be rejected");
}

// ===========================================================================
// Builder methods: with_smt_outcome, with_layer_bounds, etc.
// ===========================================================================

// ---------------------------------------------------------------------------
// 33. with_smt_outcome sets smt_outcome
// ---------------------------------------------------------------------------

/// Prove: `with_smt_outcome` sets the field correctly.
#[kani::unwind(64)]
#[kani::proof]
fn with_smt_outcome_sets_field() {
    let cert = valid_cert().with_smt_outcome("Proven");
    assert_eq!(cert.smt_outcome, Some("Proven".to_string()));
}

// ---------------------------------------------------------------------------
// 34. with_layer_bounds computes crown_coverage
// ---------------------------------------------------------------------------

/// Prove: `with_layer_bounds` computes `crown_coverage` from layer methods.
#[kani::unwind(128)]
#[kani::proof]
fn with_layer_bounds_computes_coverage() {
    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-1.0, 1.0)],
            output_bounds: vec![(-2.0, 2.0)],
            method: PropMethod::Crown, // tight
            node_name: None,
            input_sources: None,
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-2.0, 2.0)],
            output_bounds: vec![(0.0, 2.0)],
            method: PropMethod::Ibp, // not tight
            node_name: None,
            input_sources: None,
        },
    ];
    let cert = valid_cert().with_layer_bounds(bounds);
    // 1 tight out of 2 total = 0.5
    assert_eq!(cert.crown_coverage, Some(0.5));
    assert_eq!(cert.ibp_fallback_count, Some(1));
}

// ---------------------------------------------------------------------------
// 35. with_kani_status sets kani_status
// ---------------------------------------------------------------------------

/// Prove: `with_kani_status` sets the field.
#[kani::unwind(128)]
#[kani::proof]
fn with_kani_status_sets_field() {
    let record = KaniProofRecord {
        harness_count: 10,
        status: KaniOutcome::Passed,
        properties: vec!["no_overflow".to_string()],
        cbmc_version: None,
    };
    let cert = valid_cert().with_kani_status(record);
    assert!(cert.kani_status.is_some());
    let ks = cert.kani_status.unwrap();
    assert_eq!(ks.harness_count, 10);
}

// ---------------------------------------------------------------------------
// 36. with_weight_hash sets weight_hash
// ---------------------------------------------------------------------------

/// Prove: `with_weight_hash` sets the field.
#[kani::unwind(128)]
#[kani::proof]
fn with_weight_hash_sets_field() {
    let cert = valid_cert().with_weight_hash(VALID_HASH.to_string());
    assert_eq!(cert.weight_hash, Some(VALID_HASH.to_string()));
}

// ---------------------------------------------------------------------------
// 37. with_source_hash sets source_hash
// ---------------------------------------------------------------------------

/// Prove: `with_source_hash` sets the field.
#[kani::unwind(128)]
#[kani::proof]
fn with_source_hash_sets_field() {
    let cert = valid_cert().with_source_hash(VALID_HASH.to_string());
    assert_eq!(cert.source_hash, Some(VALID_HASH.to_string()));
}

// ---------------------------------------------------------------------------
// 38. with_verifier_version sets verifier_version
// ---------------------------------------------------------------------------

/// Prove: `with_verifier_version` sets the field.
#[kani::unwind(128)]
#[kani::proof]
fn with_verifier_version_sets_field() {
    let cert = valid_cert().with_verifier_version("1.2.3".to_string());
    assert_eq!(cert.verifier_version, Some("1.2.3".to_string()));
}

// ---------------------------------------------------------------------------
// 39. with_smt_proof sets both proof fields
// ---------------------------------------------------------------------------

/// Prove: `with_smt_proof` sets both `smt_proof_alethe` and `smt_proof_verdict`.
#[kani::unwind(128)]
#[kani::proof]
fn with_smt_proof_sets_both_fields() {
    let cert = valid_cert().with_smt_proof("(proof ...)".to_string(), SmtProofVerdict::Verified);
    assert_eq!(cert.smt_proof_alethe, Some("(proof ...)".to_string()));
    assert_eq!(cert.smt_proof_verdict, Some(SmtProofVerdict::Verified));
}

// ---------------------------------------------------------------------------
// 40. with_precision_model sets precision_model
// ---------------------------------------------------------------------------

/// Prove: `with_precision_model` sets the field.
#[kani::unwind(64)]
#[kani::proof]
fn with_precision_model_sets_field() {
    let cert = valid_cert().with_precision_model(PrecisionModel::F16Aware {
        cast_count: 3,
        total_epsilon: 0.001,
    });
    assert!(cert.precision_model.is_some());
    match cert.precision_model.unwrap() {
        PrecisionModel::F16Aware { cast_count, .. } => {
            assert_eq!(cast_count, 3);
        }
        _ => panic!("must be F16Aware"),
    }
}

// ===========================================================================
// IEEE 754 NaN bypass: inverted bounds with NaN
// ===========================================================================

// ---------------------------------------------------------------------------
// 41. validate: NaN lower does not bypass inverted bounds check
// ---------------------------------------------------------------------------

/// Prove: NaN lower bound doesn't silently pass as non-inverted.
/// With is_finite=false: NaN bounds are structurally valid (not finite).
/// The key invariant: when is_finite=true and bounds contain NaN, validate fails.
#[kani::unwind(64)]
#[kani::proof]
fn validate_nan_lower_finite_true_caught() {
    let mut cert = valid_cert();
    cert.output_bounds.lower = f32::NAN;
    cert.output_bounds.upper = 1.0;
    cert.is_finite = true;
    // Must catch the finite flag mismatch (NaN is not finite).
    assert!(cert.validate().is_err());
}

// ---------------------------------------------------------------------------
// 42. validate: non-finite output_width with finite bounds rejected
// ---------------------------------------------------------------------------

/// Prove: non-finite output_width with finite bounds is rejected.
#[kani::unwind(64)]
#[kani::proof]
fn validate_non_finite_width_rejected() {
    let mut cert = valid_cert();
    cert.output_width = f32::INFINITY;
    assert!(
        cert.validate().is_err(),
        "non-finite width with finite bounds must be rejected"
    );
}
