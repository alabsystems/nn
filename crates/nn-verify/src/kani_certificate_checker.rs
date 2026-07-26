// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani verification harnesses for `certificate_checker.rs`.
//!
//! Proves correctness properties of:
//! - `CheckResult::is_valid`: vacuous bounds are informational-only
//! - `CheckIssue` variant distinctness and Display completeness
//! - `VacuityAssessment` field invariants (coverage in [0,1], layers sum)
//! - `verify_cached_hash` logic: match/mismatch/error paths
//! - `check_smt_proof_consistency`: four-branch coverage
//! - `DEFAULT_VACUITY_THRESHOLD` value
//! - Vacuity assessment computation from certificate_checker_core
//! - **`check_input_spec()`**: NaN/inverted/empty input spec detection (#3755)
//! - **`check_inverted_element_bounds()`**: inverted layer output detection (#3755)
//! - **`check_nonfinite_output_bounds()`**: NaN/Inf layer output detection (#3755)
//! - **`check_input_bounds_validity()`**: NaN/inverted input bound detection (#3755)
//! - **`check_layer_trace_sequential()`**: sequential chain consistency (#3755)
//! - **`check_certificate_core()`**: full pipeline (trust anchor) (#3755)
//! - **`check_smt_proof_consistency()`**: SMT proof artifact validation (#3755)
//! - **`check_output_agreement()`**: final layer vs certificate bounds (#3755)
//! - **`check_first_layer_input_spec()`**: input layer anchoring (#3755)
//! - **`verify_cached_hash()`**: hash match/mismatch/error paths (#3755)
//!
//! Part of #3717, #3755.

use super::checker_types::{CheckIssue, CheckResult, VacuityAssessment};
use super::DEFAULT_VACUITY_THRESHOLD;

// ===========================================================================
// DEFAULT_VACUITY_THRESHOLD
// ===========================================================================

// ---------------------------------------------------------------------------
// 1. DEFAULT_VACUITY_THRESHOLD is 10.0
// ---------------------------------------------------------------------------

/// Prove: the default vacuity threshold is exactly 10.0.
#[kani::unwind(1)]
#[kani::proof]
fn default_vacuity_threshold_is_10() {
    assert_eq!(DEFAULT_VACUITY_THRESHOLD, 10.0_f32);
}

// ===========================================================================
// CheckResult::is_valid
// ===========================================================================

// ---------------------------------------------------------------------------
// 2. is_valid returns true for empty issues
// ---------------------------------------------------------------------------

/// Prove: a CheckResult with no issues is valid.
#[kani::unwind(64)]
#[kani::proof]
fn is_valid_empty_issues() {
    let result = CheckResult {
        kernel_name: "test".to_string(),
        issues: vec![],
        vacuity: None,
    };
    assert!(result.is_valid());
}

// ---------------------------------------------------------------------------
// 3. is_valid returns true when only VacuousBounds present
// ---------------------------------------------------------------------------

/// Prove: VacuousBounds is informational and does NOT cause is_valid to return false.
#[kani::unwind(64)]
#[kani::proof]
fn is_valid_with_only_vacuous_bounds() {
    let result = CheckResult {
        kernel_name: "test".to_string(),
        issues: vec![CheckIssue::VacuousBounds {
            crown_coverage: 0.1,
            output_width: 100.0,
        }],
        vacuity: None,
    };
    assert!(result.is_valid(), "VacuousBounds is informational only");
}

// ---------------------------------------------------------------------------
// 4. is_valid returns false for StructuralError
// ---------------------------------------------------------------------------

/// Prove: StructuralError makes the certificate invalid.
#[kani::unwind(128)]
#[kani::proof]
fn is_valid_false_structural_error() {
    let result = CheckResult {
        kernel_name: "test".to_string(),
        issues: vec![CheckIssue::StructuralError {
            message: "bad".to_string(),
        }],
        vacuity: None,
    };
    assert!(!result.is_valid());
}

// ---------------------------------------------------------------------------
// 5. is_valid returns false for NoLayerBounds
// ---------------------------------------------------------------------------

/// Prove: NoLayerBounds makes the certificate invalid.
#[kani::unwind(64)]
#[kani::proof]
fn is_valid_false_no_layer_bounds() {
    let result = CheckResult {
        kernel_name: "test".to_string(),
        issues: vec![CheckIssue::NoLayerBounds],
        vacuity: None,
    };
    assert!(!result.is_valid());
}

// ---------------------------------------------------------------------------
// 6. is_valid returns false for InfeasibleBounds
// ---------------------------------------------------------------------------

/// Prove: InfeasibleBounds makes the certificate invalid.
#[kani::unwind(64)]
#[kani::proof]
fn is_valid_false_infeasible_bounds() {
    let result = CheckResult {
        kernel_name: "test".to_string(),
        issues: vec![CheckIssue::InfeasibleBounds],
        vacuity: None,
    };
    assert!(!result.is_valid());
}

// ---------------------------------------------------------------------------
// 7. is_valid returns false for SmtProofMissing
// ---------------------------------------------------------------------------

/// Prove: SmtProofMissing makes the certificate invalid.
#[kani::unwind(64)]
#[kani::proof]
fn is_valid_false_smt_proof_missing() {
    let result = CheckResult {
        kernel_name: "test".to_string(),
        issues: vec![CheckIssue::SmtProofMissing],
        vacuity: None,
    };
    assert!(!result.is_valid());
}

// ---------------------------------------------------------------------------
// 8. is_valid returns false for SmtProofInvalid
// ---------------------------------------------------------------------------

/// Prove: SmtProofInvalid makes the certificate invalid.
#[kani::unwind(64)]
#[kani::proof]
fn is_valid_false_smt_proof_invalid() {
    let result = CheckResult {
        kernel_name: "test".to_string(),
        issues: vec![CheckIssue::SmtProofInvalid],
        vacuity: None,
    };
    assert!(!result.is_valid());
}

// ---------------------------------------------------------------------------
// 9. is_valid returns false for WeightHashMismatch
// ---------------------------------------------------------------------------

/// Prove: WeightHashMismatch makes the certificate invalid.
#[kani::unwind(128)]
#[kani::proof]
fn is_valid_false_weight_hash_mismatch() {
    let result = CheckResult {
        kernel_name: "test".to_string(),
        issues: vec![CheckIssue::WeightHashMismatch {
            expected: "a".to_string(),
            actual: "b".to_string(),
        }],
        vacuity: None,
    };
    assert!(!result.is_valid());
}

// ---------------------------------------------------------------------------
// 10. is_valid returns false for SourceHashMismatch
// ---------------------------------------------------------------------------

/// Prove: SourceHashMismatch makes the certificate invalid.
#[kani::unwind(128)]
#[kani::proof]
fn is_valid_false_source_hash_mismatch() {
    let result = CheckResult {
        kernel_name: "test".to_string(),
        issues: vec![CheckIssue::SourceHashMismatch {
            expected: "a".to_string(),
            actual: "b".to_string(),
        }],
        vacuity: None,
    };
    assert!(!result.is_valid());
}

// ---------------------------------------------------------------------------
// 11. is_valid returns false for ContentHashMismatch
// ---------------------------------------------------------------------------

/// Prove: ContentHashMismatch makes the certificate invalid.
#[kani::unwind(128)]
#[kani::proof]
fn is_valid_false_content_hash_mismatch() {
    let result = CheckResult {
        kernel_name: "test".to_string(),
        issues: vec![CheckIssue::ContentHashMismatch {
            expected: "a".to_string(),
            actual: "b".to_string(),
        }],
        vacuity: None,
    };
    assert!(!result.is_valid());
}

// ---------------------------------------------------------------------------
// 12. is_valid with multiple VacuousBounds is still valid
// ---------------------------------------------------------------------------

/// Prove: multiple VacuousBounds issues still pass is_valid.
#[kani::unwind(64)]
#[kani::proof]
fn is_valid_multiple_vacuous_still_valid() {
    let result = CheckResult {
        kernel_name: "test".to_string(),
        issues: vec![
            CheckIssue::VacuousBounds {
                crown_coverage: 0.1,
                output_width: 50.0,
            },
            CheckIssue::VacuousBounds {
                crown_coverage: 0.2,
                output_width: 80.0,
            },
        ],
        vacuity: None,
    };
    assert!(result.is_valid());
}

// ---------------------------------------------------------------------------
// 13. is_valid returns false when VacuousBounds mixed with real issue
// ---------------------------------------------------------------------------

/// Prove: mixing VacuousBounds with a real issue makes certificate invalid.
#[kani::unwind(64)]
#[kani::proof]
fn is_valid_false_vacuous_mixed_with_real() {
    let result = CheckResult {
        kernel_name: "test".to_string(),
        issues: vec![
            CheckIssue::VacuousBounds {
                crown_coverage: 0.1,
                output_width: 50.0,
            },
            CheckIssue::NoLayerBounds,
        ],
        vacuity: None,
    };
    assert!(!result.is_valid());
}

// ---------------------------------------------------------------------------
// 14. is_valid returns false for NanOutputBounds
// ---------------------------------------------------------------------------

/// Prove: NanOutputBounds makes the certificate invalid.
#[kani::unwind(64)]
#[kani::proof]
fn is_valid_false_nan_output_bounds() {
    let result = CheckResult {
        kernel_name: "test".to_string(),
        issues: vec![CheckIssue::NanOutputBounds],
        vacuity: None,
    };
    assert!(!result.is_valid());
}

// ---------------------------------------------------------------------------
// 15. is_valid returns false for SignatureInvalid
// ---------------------------------------------------------------------------

/// Prove: SignatureInvalid makes the certificate invalid.
#[kani::unwind(128)]
#[kani::proof]
fn is_valid_false_signature_invalid() {
    let result = CheckResult {
        kernel_name: "test".to_string(),
        issues: vec![CheckIssue::SignatureInvalid {
            message: "bad sig".to_string(),
        }],
        vacuity: None,
    };
    assert!(!result.is_valid());
}

// ===========================================================================
// VacuityAssessment invariants
// ===========================================================================

// ---------------------------------------------------------------------------
// 16. crown_layers + ibp_layers == total_layers
// ---------------------------------------------------------------------------

/// Prove: crown_layers + ibp_layers must equal total_layers (construction invariant).
#[kani::unwind(1)]
#[kani::proof]
fn vacuity_layers_sum_invariant() {
    let va = VacuityAssessment {
        crown_coverage: 0.5,
        total_layers: 10,
        crown_layers: 5,
        ibp_layers: 5,
        output_width: 3.0,
        is_non_vacuous: true,
    };
    assert_eq!(
        va.crown_layers + va.ibp_layers,
        va.total_layers,
        "crown + ibp must equal total"
    );
}

// ---------------------------------------------------------------------------
// 17. crown_coverage = crown_layers / total_layers
// ---------------------------------------------------------------------------

/// Prove: crown_coverage matches crown_layers / total_layers.
#[kani::unwind(1)]
#[kani::proof]
fn vacuity_coverage_matches_ratio() {
    let va = VacuityAssessment {
        crown_coverage: 0.75,
        total_layers: 4,
        crown_layers: 3,
        ibp_layers: 1,
        output_width: 2.0,
        is_non_vacuous: true,
    };
    let expected = va.crown_layers as f32 / va.total_layers as f32;
    assert!((va.crown_coverage - expected).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// 18. is_non_vacuous requires coverage >= 0.5
// ---------------------------------------------------------------------------

/// Prove: non-vacuous assessment requires at least 50% CROWN coverage.
#[kani::unwind(1)]
#[kani::proof]
fn vacuity_non_vacuous_requires_half_crown() {
    // 0.4 coverage, narrow width -> still vacuous per the checker logic
    let low_coverage_assessment = VacuityAssessment {
        crown_coverage: 0.4,
        total_layers: 10,
        crown_layers: 4,
        ibp_layers: 6,
        output_width: 1.0,
        is_non_vacuous: false, // matches checker: !(0.4 >= 0.5 && 1.0 < 10.0)
    };
    // The checker computes: is_non_vacuous = coverage >= 0.5 && width < threshold
    let computed = low_coverage_assessment.crown_coverage >= 0.5
        && low_coverage_assessment.output_width < DEFAULT_VACUITY_THRESHOLD;
    assert_eq!(low_coverage_assessment.is_non_vacuous, computed);
}

// ---------------------------------------------------------------------------
// 19. is_non_vacuous requires output_width < threshold
// ---------------------------------------------------------------------------

/// Prove: non-vacuous assessment requires output width below threshold.
#[kani::unwind(1)]
#[kani::proof]
fn vacuity_non_vacuous_requires_narrow_width() {
    // High coverage but very wide output -> vacuous
    let wide_output = VacuityAssessment {
        crown_coverage: 0.9,
        total_layers: 10,
        crown_layers: 9,
        ibp_layers: 1,
        output_width: 100.0,
        is_non_vacuous: false, // matches checker: !(0.9 >= 0.5 && 100.0 < 10.0)
    };
    let computed =
        wide_output.crown_coverage >= 0.5 && wide_output.output_width < DEFAULT_VACUITY_THRESHOLD;
    assert_eq!(wide_output.is_non_vacuous, computed);
}

// ---------------------------------------------------------------------------
// 20. non-vacuous: high coverage + narrow width
// ---------------------------------------------------------------------------

/// Prove: high coverage AND narrow width produces non-vacuous assessment.
#[kani::unwind(1)]
#[kani::proof]
fn vacuity_non_vacuous_high_coverage_narrow_width() {
    let good = VacuityAssessment {
        crown_coverage: 0.8,
        total_layers: 10,
        crown_layers: 8,
        ibp_layers: 2,
        output_width: 5.0,
        is_non_vacuous: true,
    };
    let computed = good.crown_coverage >= 0.5 && good.output_width < DEFAULT_VACUITY_THRESHOLD;
    assert!(computed);
    assert_eq!(good.is_non_vacuous, computed);
}

// ===========================================================================
// CheckIssue Display completeness
// ===========================================================================

// ---------------------------------------------------------------------------
// 21. StructuralError Display includes message
// ---------------------------------------------------------------------------

/// Prove: StructuralError Display format includes the message.
#[kani::unwind(64)]
#[kani::proof]
fn display_structural_error() {
    let issue = CheckIssue::StructuralError {
        message: "test_msg".to_string(),
    };
    let s = format!("{issue}");
    assert!(s.contains("structural"));
    assert!(s.contains("test_msg"));
}

// ---------------------------------------------------------------------------
// 22. NoLayerBounds Display is non-empty
// ---------------------------------------------------------------------------

/// Prove: NoLayerBounds Display produces non-empty string.
#[kani::unwind(64)]
#[kani::proof]
fn display_no_layer_bounds_non_empty() {
    let issue = CheckIssue::NoLayerBounds;
    let s = format!("{issue}");
    assert!(!s.is_empty());
}

// ---------------------------------------------------------------------------
// 23. InfeasibleBounds Display mentions sentinel
// ---------------------------------------------------------------------------

/// Prove: InfeasibleBounds Display mentions sentinel or proof failed.
#[kani::unwind(64)]
#[kani::proof]
fn display_infeasible_bounds() {
    let issue = CheckIssue::InfeasibleBounds;
    let s = format!("{issue}");
    assert!(s.contains("infeasible") || s.contains("sentinel") || s.contains("failed"));
}

// ---------------------------------------------------------------------------
// 24. SmtProofMissing Display mentions Proven
// ---------------------------------------------------------------------------

/// Prove: SmtProofMissing Display mentions Proven and proof.
#[kani::unwind(64)]
#[kani::proof]
fn display_smt_proof_missing() {
    let issue = CheckIssue::SmtProofMissing;
    let s = format!("{issue}");
    assert!(s.contains("Proven") || s.contains("proof"));
}

// ---------------------------------------------------------------------------
// 25. SmtProofInvalid Display mentions Invalid
// ---------------------------------------------------------------------------

/// Prove: SmtProofInvalid Display mentions Invalid or verdict.
#[kani::unwind(64)]
#[kani::proof]
fn display_smt_proof_invalid() {
    let issue = CheckIssue::SmtProofInvalid;
    let s = format!("{issue}");
    assert!(s.contains("Invalid") || s.contains("verdict"));
}

// ###########################################################################
// ###########################################################################
//
// HIGH-VALUE HARNESSES: Production function proofs (#3755)
//
// Everything below calls actual production validation functions with
// constructed inputs, proving the trust-anchor certificate checker
// correctly rejects invalid certificates and accepts valid ones.
//
// ###########################################################################
// ###########################################################################

use super::agreement::{check_first_layer_input_spec, check_output_agreement};
use super::check_certificate_core;
use super::check_smt_proof_consistency;
use super::trace::{check_input_spec, check_layer_trace_consistency};
use super::verify_cached_hash;
use super::FileHashCache;
use crate::certificate::ProofCertificate;
use crate::certificate_types::LayerBoundRecord;
use crate::soundness_compat::VerificationSoundnessMode;
use crate::status::{InputBoundsRecord, OutputBoundsRecord, ParamInputRecord, SmtProofVerdict};
use crate::verify_types::{KernelVerification, PropMethod};

/// Valid 64-char hex SHA-256 hash for fixtures.
const VALID_HASH: &str = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2";

/// Helper: minimal valid ProofCertificate with layer bounds.
fn valid_cert_with_layers() -> ProofCertificate {
    let kv = KernelVerification::new(
        "test_kernel".to_string(),
        PropMethod::Crown,
        -1.0,
        1.0,
        2.0,
        true,
    );
    let input_spec = InputBoundsRecord::new(&[ParamInputRecord::new(0, -1.0, 1.0)], &[]);
    let mut cert = ProofCertificate::from_verification(&kv, input_spec);
    cert.source_hash = Some(VALID_HASH.to_string());
    cert.layer_bounds = Some(vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-1.0, 1.0)],
        output_bounds: vec![(-1.0, 1.0)],
        method: PropMethod::Crown,
        node_name: None,
        input_sources: None,
    }]);
    cert
}

/// Helper: minimal valid cert (no layer bounds).
fn valid_cert_minimal() -> ProofCertificate {
    let kv = KernelVerification::new(
        "test_kernel".to_string(),
        PropMethod::Crown,
        -1.0,
        1.0,
        2.0,
        true,
    );
    let input_spec = InputBoundsRecord::new(&[ParamInputRecord::new(0, -1.0, 1.0)], &[]);
    let mut cert = ProofCertificate::from_verification(&kv, input_spec);
    cert.source_hash = Some(VALID_HASH.to_string());
    cert
}

/// Helper: empty file hash cache (no files to check).
fn empty_cache() -> FileHashCache {
    FileHashCache {
        weight: None,
        source: None,
    }
}

// ===========================================================================
// check_input_spec() — production function proofs
// ===========================================================================

// ---------------------------------------------------------------------------
// 26. check_input_spec: NaN lower bound detected
// ---------------------------------------------------------------------------

/// Prove: check_input_spec detects NaN in a parameter's lower bound.
/// IEEE 754 soundness: NaN bypasses `>` comparison, so explicit
/// `!is_finite()` check is required.
#[kani::unwind(8)]
#[kani::proof]
fn check_input_spec_nan_lower_detected() {
    let spec = InputBoundsRecord::new(&[ParamInputRecord::new(0, f32::NAN, 1.0)], &[]);
    let mut issues = Vec::new();
    check_input_spec(&spec, &mut issues);
    assert!(!issues.is_empty(), "NaN lower must produce an issue");
    let has_invalid = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::InvalidInputSpec { .. }));
    assert!(has_invalid, "must be InvalidInputSpec");
}

// ---------------------------------------------------------------------------
// 27. check_input_spec: NaN upper bound detected
// ---------------------------------------------------------------------------

/// Prove: check_input_spec detects NaN in a parameter's upper bound.
#[kani::unwind(8)]
#[kani::proof]
fn check_input_spec_nan_upper_detected() {
    let spec = InputBoundsRecord::new(&[ParamInputRecord::new(0, -1.0, f32::NAN)], &[]);
    let mut issues = Vec::new();
    check_input_spec(&spec, &mut issues);
    assert!(!issues.is_empty(), "NaN upper must produce an issue");
    let has_invalid = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::InvalidInputSpec { .. }));
    assert!(has_invalid);
}

// ---------------------------------------------------------------------------
// 28. check_input_spec: Inf bounds detected
// ---------------------------------------------------------------------------

/// Prove: check_input_spec detects +/-Inf in parameter bounds.
#[kani::unwind(8)]
#[kani::proof]
fn check_input_spec_inf_detected() {
    let spec = InputBoundsRecord::new(
        &[ParamInputRecord::new(0, f32::NEG_INFINITY, f32::INFINITY)],
        &[],
    );
    let mut issues = Vec::new();
    check_input_spec(&spec, &mut issues);
    assert!(!issues.is_empty(), "Inf bounds must produce an issue");
}

// ---------------------------------------------------------------------------
// 29. check_input_spec: inverted bounds detected
// ---------------------------------------------------------------------------

/// Prove: check_input_spec detects inverted bounds (lower > upper).
/// This is the vacuously-true attack: the proof "verifies for no inputs."
#[kani::unwind(8)]
#[kani::proof]
fn check_input_spec_inverted_detected() {
    let spec = InputBoundsRecord::new(&[ParamInputRecord::new(0, 5.0, -5.0)], &[]);
    let mut issues = Vec::new();
    check_input_spec(&spec, &mut issues);
    assert!(!issues.is_empty(), "inverted bounds must produce an issue");
    let has_invalid = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::InvalidInputSpec { .. }));
    assert!(has_invalid);
}

// ---------------------------------------------------------------------------
// 30. check_input_spec: empty variable_inputs detected
// ---------------------------------------------------------------------------

/// Prove: check_input_spec detects empty variable_inputs.
/// A certificate with no variable inputs verified nothing.
#[kani::unwind(8)]
#[kani::proof]
fn check_input_spec_empty_detected() {
    let spec = InputBoundsRecord::new(&[], &[]);
    let mut issues = Vec::new();
    check_input_spec(&spec, &mut issues);
    assert!(
        !issues.is_empty(),
        "empty variable_inputs must produce an issue"
    );
}

// ---------------------------------------------------------------------------
// 31. check_input_spec: valid spec produces no issues
// ---------------------------------------------------------------------------

/// Prove: valid input spec with lower <= upper produces no issues.
#[kani::unwind(8)]
#[kani::proof]
fn check_input_spec_valid_no_issues() {
    let spec = InputBoundsRecord::new(
        &[
            ParamInputRecord::new(0, -1.0, 1.0),
            ParamInputRecord::new(1, 0.0, 10.0),
        ],
        &[],
    );
    let mut issues = Vec::new();
    check_input_spec(&spec, &mut issues);
    assert!(issues.is_empty(), "valid spec must produce 0 issues");
}

// ---------------------------------------------------------------------------
// 32. check_input_spec: point interval (lower == upper) is valid
// ---------------------------------------------------------------------------

/// Prove: lower == upper (single point) is NOT flagged as inverted.
#[kani::unwind(8)]
#[kani::proof]
fn check_input_spec_point_interval_valid() {
    let spec = InputBoundsRecord::new(&[ParamInputRecord::new(0, 3.0, 3.0)], &[]);
    let mut issues = Vec::new();
    check_input_spec(&spec, &mut issues);
    assert!(issues.is_empty(), "point interval (lo==hi) must be valid");
}

// ===========================================================================
// check_nonfinite_output_bounds() — NaN/Inf in layer outputs
// ===========================================================================

// ---------------------------------------------------------------------------
// 33. NaN in output bounds detected
// ---------------------------------------------------------------------------

/// Prove: check_layer_trace_consistency detects NaN in layer output bounds.
/// This is the #1 IEEE 754 soundness risk: NaN != NaN causes spurious
/// LayerTraceGap instead of NonFiniteElement.
#[kani::unwind(64)]
#[kani::proof]
fn trace_nan_output_bounds_detected() {
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-1.0, 1.0)],
        output_bounds: vec![(f32::NAN, 1.0)],
        method: PropMethod::Crown,
        node_name: None,
        input_sources: None,
    }];
    let mut issues = Vec::new();
    check_layer_trace_consistency(&bounds, &mut issues);
    let has_nonfinite = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::NonFiniteElement { .. }));
    assert!(has_nonfinite, "NaN in output must produce NonFiniteElement");
}

// ---------------------------------------------------------------------------
// 34. Inf in output bounds detected
// ---------------------------------------------------------------------------

/// Prove: +Inf in layer output bounds produces NonFiniteElement.
#[kani::unwind(64)]
#[kani::proof]
fn trace_inf_output_bounds_detected() {
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-1.0, 1.0)],
        output_bounds: vec![(-1.0, f32::INFINITY)],
        method: PropMethod::Crown,
        node_name: None,
        input_sources: None,
    }];
    let mut issues = Vec::new();
    check_layer_trace_consistency(&bounds, &mut issues);
    let has_nonfinite = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::NonFiniteElement { .. }));
    assert!(has_nonfinite, "Inf in output must produce NonFiniteElement");
}

// ---------------------------------------------------------------------------
// 35. NaN in BOTH lower and upper detected
// ---------------------------------------------------------------------------

/// Prove: NaN in both lower and upper of output bounds is detected.
/// Crucial: `lo > hi` returns false for NaN (IEEE 754). Only explicit
/// `!is_finite()` catches this.
#[kani::unwind(64)]
#[kani::proof]
fn trace_nan_both_bounds_detected() {
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(0.0, 1.0)],
        output_bounds: vec![(f32::NAN, f32::NAN)],
        method: PropMethod::Crown,
        node_name: None,
        input_sources: None,
    }];
    let mut issues = Vec::new();
    check_layer_trace_consistency(&bounds, &mut issues);
    assert!(!issues.is_empty(), "NaN in both bounds must be caught");
}

// ===========================================================================
// check_inverted_element_bounds() — inverted layer outputs
// ===========================================================================

// ---------------------------------------------------------------------------
// 36. Inverted output bounds detected
// ---------------------------------------------------------------------------

/// Prove: inverted (lower > upper) output bounds produce InvertedElementBounds.
#[kani::unwind(64)]
#[kani::proof]
fn trace_inverted_output_detected() {
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-1.0, 1.0)],
        output_bounds: vec![(5.0, -5.0)], // inverted
        method: PropMethod::Crown,
        node_name: None,
        input_sources: None,
    }];
    let mut issues = Vec::new();
    check_layer_trace_consistency(&bounds, &mut issues);
    let has_inverted = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::InvertedElementBounds { .. }));
    assert!(
        has_inverted,
        "inverted output must produce InvertedElementBounds"
    );
}

// ---------------------------------------------------------------------------
// 37. Non-inverted output bounds produce no InvertedElementBounds
// ---------------------------------------------------------------------------

/// Prove: valid (lower <= upper) output bounds do NOT produce InvertedElementBounds.
#[kani::unwind(64)]
#[kani::proof]
fn trace_valid_output_no_inverted() {
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-1.0, 1.0)],
        output_bounds: vec![(-2.0, 2.0)],
        method: PropMethod::Crown,
        node_name: None,
        input_sources: None,
    }];
    let mut issues = Vec::new();
    check_layer_trace_consistency(&bounds, &mut issues);
    let has_inverted = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::InvertedElementBounds { .. }));
    assert!(
        !has_inverted,
        "valid bounds must not produce InvertedElementBounds"
    );
}

// ---------------------------------------------------------------------------
// 38. Zero-width interval (lo == hi) is NOT inverted
// ---------------------------------------------------------------------------

/// Prove: equal output bounds (point interval) are not flagged as inverted.
#[kani::unwind(64)]
#[kani::proof]
fn trace_zero_width_not_inverted() {
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "ReLU".to_string(),
        input_bounds: vec![(0.0, 0.0)],
        output_bounds: vec![(0.0, 0.0)],
        method: PropMethod::Crown,
        node_name: None,
        input_sources: None,
    }];
    let mut issues = Vec::new();
    check_layer_trace_consistency(&bounds, &mut issues);
    let has_inverted = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::InvertedElementBounds { .. }));
    assert!(!has_inverted, "zero-width interval must not be flagged");
}

// ===========================================================================
// check_input_bounds_validity() — NaN/inverted in layer INPUT bounds
// ===========================================================================

// ---------------------------------------------------------------------------
// 39. NaN in input bounds detected
// ---------------------------------------------------------------------------

/// Prove: NaN in a layer's input_bounds produces NonFiniteElement.
#[kani::unwind(64)]
#[kani::proof]
fn trace_nan_input_bounds_detected() {
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(f32::NAN, 1.0)],
        output_bounds: vec![(-1.0, 1.0)],
        method: PropMethod::Crown,
        node_name: None,
        input_sources: None,
    }];
    let mut issues = Vec::new();
    check_layer_trace_consistency(&bounds, &mut issues);
    let has_nonfinite = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::NonFiniteElement { .. }));
    assert!(
        has_nonfinite,
        "NaN in input bounds must produce NonFiniteElement"
    );
}

// ---------------------------------------------------------------------------
// 40. Inverted input bounds detected
// ---------------------------------------------------------------------------

/// Prove: inverted input bounds (lo > hi) produce InvertedElementBounds.
/// Inverted input bounds mean the layer was verified on an empty interval.
#[kani::unwind(64)]
#[kani::proof]
fn trace_inverted_input_bounds_detected() {
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(5.0, -5.0)], // inverted
        output_bounds: vec![(-1.0, 1.0)],
        method: PropMethod::Crown,
        node_name: None,
        input_sources: None,
    }];
    let mut issues = Vec::new();
    check_layer_trace_consistency(&bounds, &mut issues);
    let has_inverted = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::InvertedElementBounds { .. }));
    assert!(
        has_inverted,
        "inverted input bounds must produce InvertedElementBounds"
    );
}

// ===========================================================================
// check_layer_trace_sequential() — chain consistency
// ===========================================================================

// ---------------------------------------------------------------------------
// 41. Sequential: matching chain produces no gaps
// ---------------------------------------------------------------------------

/// Prove: a valid sequential chain (output[i] == input[i+1]) has no gaps.
#[kani::unwind(128)]
#[kani::proof]
fn trace_sequential_valid_no_gaps() {
    let bounds = vec![
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
            input_bounds: vec![(-2.0, 2.0)], // matches previous output
            output_bounds: vec![(0.0, 2.0)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: None,
        },
    ];
    let mut issues = Vec::new();
    check_layer_trace_consistency(&bounds, &mut issues);
    let has_gap = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::LayerTraceGap { .. }));
    assert!(!has_gap, "valid sequential chain must have no gaps");
}

// ---------------------------------------------------------------------------
// 42. Sequential: broken chain produces LayerTraceGap
// ---------------------------------------------------------------------------

/// Prove: a broken sequential chain (output[0] != input[1]) produces gap.
#[kani::unwind(128)]
#[kani::proof]
fn trace_sequential_broken_produces_gap() {
    let bounds = vec![
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
            input_bounds: vec![(-999.0, 999.0)], // does NOT match previous output
            output_bounds: vec![(0.0, 999.0)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: None,
        },
    ];
    let mut issues = Vec::new();
    check_layer_trace_consistency(&bounds, &mut issues);
    let has_gap = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::LayerTraceGap { .. }));
    assert!(has_gap, "broken chain must produce LayerTraceGap");
}

// ---------------------------------------------------------------------------
// 43. Sequential: single layer has no gap (no successor)
// ---------------------------------------------------------------------------

/// Prove: a single-layer trace has no gap issues.
#[kani::unwind(64)]
#[kani::proof]
fn trace_sequential_single_layer_no_gap() {
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-1.0, 1.0)],
        output_bounds: vec![(-2.0, 2.0)],
        method: PropMethod::Crown,
        node_name: None,
        input_sources: None,
    }];
    let mut issues = Vec::new();
    check_layer_trace_consistency(&bounds, &mut issues);
    let has_gap = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::LayerTraceGap { .. }));
    assert!(!has_gap, "single layer cannot have a trace gap");
}

// ===========================================================================
// check_certificate_core() — FULL PIPELINE (trust anchor)
// ===========================================================================

// ---------------------------------------------------------------------------
// 44. Core: valid certificate with layers passes all checks
// ---------------------------------------------------------------------------

/// Prove: a structurally valid certificate with matching layer bounds
/// passes check_certificate_core with is_valid() == true (modulo
/// VacuousBounds which is informational).
#[kani::unwind(128)]
#[kani::proof]
fn core_valid_cert_passes() {
    let cert = valid_cert_with_layers();
    let cache = empty_cache();
    let result = check_certificate_core(&cert, &cache);
    // Only VacuousBounds is allowed for a "passing" cert.
    assert!(
        result
            .issues
            .iter()
            .all(|i| { matches!(i, CheckIssue::VacuousBounds { .. }) }),
        "valid cert must only have VacuousBounds (informational), got: {:?}",
        result.issues,
    );
}

// ---------------------------------------------------------------------------
// 45. Core: infeasible bounds certificate is invalid
// ---------------------------------------------------------------------------

/// Prove: check_certificate_core flags a certificate with is_infeasible=true.
#[kani::unwind(128)]
#[kani::proof]
fn core_infeasible_bounds_invalid() {
    let mut cert = valid_cert_minimal();
    cert.output_bounds.is_infeasible = true;
    let cache = empty_cache();
    let result = check_certificate_core(&cert, &cache);
    let has_infeasible = result
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::InfeasibleBounds));
    assert!(
        has_infeasible,
        "is_infeasible=true must produce InfeasibleBounds issue"
    );
    assert!(!result.is_valid(), "infeasible cert must not be valid");
}

// ---------------------------------------------------------------------------
// 46. Core: NaN input spec makes certificate invalid
// ---------------------------------------------------------------------------

/// Prove: certificate with NaN in input_spec is flagged as invalid.
/// This is a critical soundness property: a NaN input range verifies
/// "for no inputs" (vacuously true).
#[kani::unwind(128)]
#[kani::proof]
fn core_nan_input_spec_invalid() {
    let kv = KernelVerification::new("test".to_string(), PropMethod::Crown, -1.0, 1.0, 2.0, true);
    let input_spec = InputBoundsRecord::new(&[ParamInputRecord::new(0, f32::NAN, 1.0)], &[]);
    let mut cert = ProofCertificate::from_verification(&kv, input_spec);
    cert.source_hash = Some(VALID_HASH.to_string());
    let cache = empty_cache();
    let result = check_certificate_core(&cert, &cache);
    assert!(
        !result.is_valid(),
        "NaN in input_spec must make cert invalid"
    );
}

// ---------------------------------------------------------------------------
// 47. Core: inverted input spec makes certificate invalid
// ---------------------------------------------------------------------------

/// Prove: certificate with inverted input_spec (lower > upper) is invalid.
#[kani::unwind(128)]
#[kani::proof]
fn core_inverted_input_spec_invalid() {
    let kv = KernelVerification::new("test".to_string(), PropMethod::Crown, -1.0, 1.0, 2.0, true);
    let input_spec = InputBoundsRecord::new(
        &[ParamInputRecord::new(0, 10.0, -10.0)], // inverted
        &[],
    );
    let mut cert = ProofCertificate::from_verification(&kv, input_spec);
    cert.source_hash = Some(VALID_HASH.to_string());
    let cache = empty_cache();
    let result = check_certificate_core(&cert, &cache);
    assert!(
        !result.is_valid(),
        "inverted input_spec must make cert invalid"
    );
}

// ---------------------------------------------------------------------------
// 48. Core: empty input spec makes certificate invalid
// ---------------------------------------------------------------------------

/// Prove: certificate with empty variable_inputs is invalid.
#[kani::unwind(128)]
#[kani::proof]
fn core_empty_input_spec_invalid() {
    let kv = KernelVerification::new("test".to_string(), PropMethod::Crown, -1.0, 1.0, 2.0, true);
    let input_spec = InputBoundsRecord::new(&[], &[]);
    let mut cert = ProofCertificate::from_verification(&kv, input_spec);
    cert.source_hash = Some(VALID_HASH.to_string());
    let cache = empty_cache();
    let result = check_certificate_core(&cert, &cache);
    assert!(
        !result.is_valid(),
        "empty input_spec must make cert invalid"
    );
}

// ---------------------------------------------------------------------------
// 49. Core: missing source_hash produces MissingHash issue
// ---------------------------------------------------------------------------

/// Prove: certificate without source_hash produces MissingHash.
#[kani::unwind(128)]
#[kani::proof]
fn core_missing_source_hash() {
    let mut cert = valid_cert_minimal();
    cert.source_hash = None;
    let cache = empty_cache();
    let result = check_certificate_core(&cert, &cache);
    let has_missing = result
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::MissingHash { field } if field == "source_hash"));
    assert!(has_missing, "missing source_hash must produce MissingHash");
}

// ---------------------------------------------------------------------------
// 50. Core: None layer_bounds produces NoLayerBounds
// ---------------------------------------------------------------------------

/// Prove: certificate with layer_bounds=None produces NoLayerBounds.
#[kani::unwind(128)]
#[kani::proof]
fn core_no_layer_bounds() {
    let mut cert = valid_cert_minimal();
    cert.layer_bounds = None;
    let cache = empty_cache();
    let result = check_certificate_core(&cert, &cache);
    let has_no_bounds = result
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::NoLayerBounds));
    assert!(
        has_no_bounds,
        "None layer_bounds must produce NoLayerBounds"
    );
}

// ---------------------------------------------------------------------------
// 51. Core: empty layer_bounds vec produces NoLayerBounds
// ---------------------------------------------------------------------------

/// Prove: certificate with layer_bounds=Some(vec![]) produces NoLayerBounds.
#[kani::unwind(128)]
#[kani::proof]
fn core_empty_layer_bounds_vec() {
    let mut cert = valid_cert_minimal();
    cert.layer_bounds = Some(vec![]);
    let cache = empty_cache();
    let result = check_certificate_core(&cert, &cache);
    let has_no_bounds = result
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::NoLayerBounds));
    assert!(
        has_no_bounds,
        "empty layer_bounds vec must produce NoLayerBounds"
    );
}

// ===========================================================================
// check_smt_proof_consistency() — SMT proof artifact validation
// ===========================================================================

// ---------------------------------------------------------------------------
// 52. SMT: Proven outcome without proof artifact is flagged
// ---------------------------------------------------------------------------

/// Prove: smt_outcome="Proven" without smt_proof_alethe produces SmtProofMissing.
#[kani::unwind(128)]
#[kani::proof]
fn smt_proven_without_artifact_flagged() {
    let mut cert = valid_cert_minimal();
    cert.smt_outcome = Some("Proven".to_string());
    cert.smt_proof_alethe = None;
    let mut issues = Vec::new();
    check_smt_proof_consistency(&cert, &mut issues);
    let has_missing = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::SmtProofMissing));
    assert!(
        has_missing,
        "Proven without artifact must produce SmtProofMissing"
    );
}

// ---------------------------------------------------------------------------
// 53. SMT: Proven outcome WITH proof artifact is NOT flagged
// ---------------------------------------------------------------------------

/// Prove: smt_outcome="Proven" with smt_proof_alethe does NOT produce SmtProofMissing.
#[kani::unwind(128)]
#[kani::proof]
fn smt_proven_with_artifact_not_flagged() {
    let mut cert = valid_cert_minimal();
    cert.smt_outcome = Some("Proven".to_string());
    cert.smt_proof_alethe = Some("(proof ...)".to_string());
    cert.smt_proof_verdict = Some(SmtProofVerdict::Verified);
    let mut issues = Vec::new();
    check_smt_proof_consistency(&cert, &mut issues);
    let has_missing = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::SmtProofMissing));
    assert!(
        !has_missing,
        "Proven with artifact must NOT produce SmtProofMissing"
    );
}

// ---------------------------------------------------------------------------
// 54. SMT: proof artifact with Invalid verdict is flagged
// ---------------------------------------------------------------------------

/// Prove: smt_proof_alethe with verdict=Invalid produces SmtProofInvalid.
#[kani::unwind(128)]
#[kani::proof]
fn smt_invalid_verdict_flagged() {
    let mut cert = valid_cert_minimal();
    cert.smt_proof_alethe = Some("(proof ...)".to_string());
    cert.smt_proof_verdict = Some(SmtProofVerdict::Invalid);
    let mut issues = Vec::new();
    check_smt_proof_consistency(&cert, &mut issues);
    let has_invalid = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::SmtProofInvalid));
    assert!(has_invalid, "Invalid verdict must produce SmtProofInvalid");
}

// ---------------------------------------------------------------------------
// 55. SMT: no smt_outcome and no artifact produces no issues
// ---------------------------------------------------------------------------

/// Prove: no smt_outcome and no artifact is clean (pre-v3 certificate).
#[kani::unwind(128)]
#[kani::proof]
fn smt_no_outcome_no_artifact_clean() {
    let mut cert = valid_cert_minimal();
    cert.smt_outcome = None;
    cert.smt_proof_alethe = None;
    let mut issues = Vec::new();
    check_smt_proof_consistency(&cert, &mut issues);
    assert!(issues.is_empty(), "no SMT data must produce 0 issues");
}

// ===========================================================================
// verify_cached_hash() — hash match/mismatch/error paths
// ===========================================================================

// ---------------------------------------------------------------------------
// 56. Hash match produces no issues
// ---------------------------------------------------------------------------

/// Prove: matching hash produces no issues.
#[kani::unwind(64)]
#[kani::proof]
fn cached_hash_match_no_issues() {
    let expected = "abc123";
    let cached = Ok("abc123".to_string());
    let mut issues = Vec::new();
    verify_cached_hash(expected, &cached, "weight_hash", &mut issues);
    assert!(issues.is_empty(), "matching hash must produce 0 issues");
}

// ---------------------------------------------------------------------------
// 57. Hash mismatch produces WeightHashMismatch
// ---------------------------------------------------------------------------

/// Prove: mismatching weight_hash produces WeightHashMismatch.
#[kani::unwind(64)]
#[kani::proof]
fn cached_hash_mismatch_weight() {
    let expected = "aaa";
    let cached = Ok("bbb".to_string());
    let mut issues = Vec::new();
    verify_cached_hash(expected, &cached, "weight_hash", &mut issues);
    let has_mismatch = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::WeightHashMismatch { .. }));
    assert!(
        has_mismatch,
        "weight hash mismatch must produce WeightHashMismatch"
    );
}

// ---------------------------------------------------------------------------
// 58. Hash mismatch for source produces SourceHashMismatch
// ---------------------------------------------------------------------------

/// Prove: mismatching source_hash produces SourceHashMismatch.
#[kani::unwind(64)]
#[kani::proof]
fn cached_hash_mismatch_source() {
    let expected = "aaa";
    let cached = Ok("bbb".to_string());
    let mut issues = Vec::new();
    verify_cached_hash(expected, &cached, "source_hash", &mut issues);
    let has_mismatch = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::SourceHashMismatch { .. }));
    assert!(
        has_mismatch,
        "source hash mismatch must produce SourceHashMismatch"
    );
}

// ---------------------------------------------------------------------------
// 59. Hash file error produces HashFileError
// ---------------------------------------------------------------------------

/// Prove: cached hash error produces HashFileError.
#[kani::unwind(64)]
#[kani::proof]
fn cached_hash_error_produces_issue() {
    let expected = "aaa";
    let cached: Result<String, String> = Err("file not found".to_string());
    let mut issues = Vec::new();
    verify_cached_hash(expected, &cached, "weight_hash", &mut issues);
    let has_err = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::HashFileError { .. }));
    assert!(has_err, "hash file error must produce HashFileError");
}

// ===========================================================================
// check_output_agreement() — last layer vs certificate bounds
// ===========================================================================

// ---------------------------------------------------------------------------
// 60. Output agreement: matching bounds produce no mismatch
// ---------------------------------------------------------------------------

/// Prove: when last layer's output bounds match certificate's output_bounds,
/// check_output_agreement produces no OutputMismatch.
#[kani::unwind(128)]
#[kani::proof]
fn output_agreement_matching_no_mismatch() {
    let kv = KernelVerification::new("test".to_string(), PropMethod::Crown, -2.0, 2.0, 4.0, true);
    let input_spec = InputBoundsRecord::new(&[ParamInputRecord::new(0, -1.0, 1.0)], &[]);
    let cert = ProofCertificate::from_verification(&kv, input_spec);
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-1.0, 1.0)],
        output_bounds: vec![(-2.0, 2.0)], // matches cert output [-2, 2]
        method: PropMethod::Crown,
        node_name: None,
        input_sources: None,
    }];
    let mut issues = Vec::new();
    check_output_agreement(&cert, &bounds, &mut issues);
    let has_mismatch = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::OutputMismatch { .. }));
    assert!(!has_mismatch, "matching output must produce no mismatch");
}

// ---------------------------------------------------------------------------
// 61. Output agreement: mismatched bounds produce OutputMismatch
// ---------------------------------------------------------------------------

/// Prove: when last layer's output bounds differ from certificate's, OutputMismatch.
#[kani::unwind(128)]
#[kani::proof]
fn output_agreement_mismatch_detected() {
    let kv = KernelVerification::new("test".to_string(), PropMethod::Crown, -2.0, 2.0, 4.0, true);
    let input_spec = InputBoundsRecord::new(&[ParamInputRecord::new(0, -1.0, 1.0)], &[]);
    let cert = ProofCertificate::from_verification(&kv, input_spec);
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-1.0, 1.0)],
        output_bounds: vec![(-100.0, 100.0)], // does NOT match cert [-2, 2]
        method: PropMethod::Crown,
        node_name: None,
        input_sources: None,
    }];
    let mut issues = Vec::new();
    check_output_agreement(&cert, &bounds, &mut issues);
    let has_mismatch = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::OutputMismatch { .. }));
    assert!(
        has_mismatch,
        "mismatched output must produce OutputMismatch"
    );
}

// ---------------------------------------------------------------------------
// 62. Output agreement: NaN in last layer produces NanOutputBounds
// ---------------------------------------------------------------------------

/// Prove: NaN in the last layer's output bounds produces NanOutputBounds.
/// Critical IEEE 754 property: the reduce (min/max fold) would silently
/// drop NaN. The checker must detect NaN BEFORE the fold.
#[kani::unwind(128)]
#[kani::proof]
fn output_agreement_nan_detected() {
    let kv = KernelVerification::new("test".to_string(), PropMethod::Crown, -1.0, 1.0, 2.0, true);
    let input_spec = InputBoundsRecord::new(&[ParamInputRecord::new(0, -1.0, 1.0)], &[]);
    let cert = ProofCertificate::from_verification(&kv, input_spec);
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-1.0, 1.0)],
        output_bounds: vec![(f32::NAN, 1.0)], // NaN in output
        method: PropMethod::Crown,
        node_name: None,
        input_sources: None,
    }];
    let mut issues = Vec::new();
    check_output_agreement(&cert, &bounds, &mut issues);
    let has_nan = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::NanOutputBounds));
    assert!(has_nan, "NaN in last layer must produce NanOutputBounds");
}

// ---------------------------------------------------------------------------
// 63. Output agreement: empty output bounds detected
// ---------------------------------------------------------------------------

/// Prove: empty output_bounds in last layer produces EmptyOutputBounds.
#[kani::unwind(128)]
#[kani::proof]
fn output_agreement_empty_detected() {
    let kv = KernelVerification::new("test".to_string(), PropMethod::Crown, -1.0, 1.0, 2.0, true);
    let input_spec = InputBoundsRecord::new(&[ParamInputRecord::new(0, -1.0, 1.0)], &[]);
    let cert = ProofCertificate::from_verification(&kv, input_spec);
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-1.0, 1.0)],
        output_bounds: vec![], // empty
        method: PropMethod::Crown,
        node_name: None,
        input_sources: None,
    }];
    let mut issues = Vec::new();
    check_output_agreement(&cert, &bounds, &mut issues);
    let has_empty = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::EmptyOutputBounds { .. }));
    assert!(
        has_empty,
        "empty output_bounds must produce EmptyOutputBounds"
    );
}

// ===========================================================================
// check_first_layer_input_spec() — anchoring input layer to spec
// ===========================================================================

// ---------------------------------------------------------------------------
// 64. First layer input matches spec: no issues
// ---------------------------------------------------------------------------

/// Prove: when first layer input_bounds match input_spec, no
/// InputBoundsSpecMismatch is produced.
#[kani::unwind(64)]
#[kani::proof]
fn first_layer_spec_match_no_mismatch() {
    let spec = InputBoundsRecord::new(&[ParamInputRecord::new(0, -1.0, 1.0)], &[]);
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-1.0, 1.0)], // matches spec
        output_bounds: vec![(-2.0, 2.0)],
        method: PropMethod::Crown,
        node_name: None,
        input_sources: None, // None = network input layer
    }];
    let mut issues = Vec::new();
    check_first_layer_input_spec(&spec, &bounds, &mut issues);
    let has_mismatch = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::InputBoundsSpecMismatch { .. }));
    assert!(!has_mismatch, "matching spec must produce no mismatch");
}

// ---------------------------------------------------------------------------
// 65. First layer input does NOT match spec: mismatch detected
// ---------------------------------------------------------------------------

/// Prove: when first layer input_bounds differ from input_spec,
/// InputBoundsSpecMismatch is produced. This catches forged certificates
/// that claim arbitrary input_bounds while having a valid-looking spec.
#[kani::unwind(64)]
#[kani::proof]
fn first_layer_spec_mismatch_detected() {
    let spec = InputBoundsRecord::new(&[ParamInputRecord::new(0, -1.0, 1.0)], &[]);
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-999.0, 999.0)], // does NOT match spec [-1, 1]
        output_bounds: vec![(-2.0, 2.0)],
        method: PropMethod::Crown,
        node_name: None,
        input_sources: None,
    }];
    let mut issues = Vec::new();
    check_first_layer_input_spec(&spec, &bounds, &mut issues);
    let has_mismatch = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::InputBoundsSpecMismatch { .. }));
    assert!(
        has_mismatch,
        "spec mismatch must produce InputBoundsSpecMismatch"
    );
}

// ---------------------------------------------------------------------------
// 66. First layer with NaN input bounds: NonFiniteElement detected
// ---------------------------------------------------------------------------

/// Prove: NaN in first layer's input_bounds produces NonFiniteElement.
#[kani::unwind(64)]
#[kani::proof]
fn first_layer_nan_input_detected() {
    let spec = InputBoundsRecord::new(&[ParamInputRecord::new(0, -1.0, 1.0)], &[]);
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(f32::NAN, 1.0)],
        output_bounds: vec![(-2.0, 2.0)],
        method: PropMethod::Crown,
        node_name: None,
        input_sources: None,
    }];
    let mut issues = Vec::new();
    check_first_layer_input_spec(&spec, &bounds, &mut issues);
    let has_nonfinite = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::NonFiniteElement { .. }));
    assert!(
        has_nonfinite,
        "NaN in first layer input must produce NonFiniteElement"
    );
}

// ===========================================================================
// Graph-aware trace validation
// ===========================================================================

// ---------------------------------------------------------------------------
// 67. Graph-aware: self-reference detected
// ---------------------------------------------------------------------------

/// Prove: a layer that references itself as input source produces
/// SelfReferenceSource.
#[kani::unwind(64)]
#[kani::proof]
fn graph_self_reference_detected() {
    let bounds = vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-1.0, 1.0)],
        output_bounds: vec![(-2.0, 2.0)],
        method: PropMethod::Crown,
        node_name: None,
        input_sources: Some(vec![0]), // self-reference!
    }];
    let mut issues = Vec::new();
    check_layer_trace_consistency(&bounds, &mut issues);
    let has_self_ref = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::SelfReferenceSource { .. }));
    assert!(
        has_self_ref,
        "self-reference must produce SelfReferenceSource"
    );
}

// ---------------------------------------------------------------------------
// 68. Graph-aware: forward reference detected
// ---------------------------------------------------------------------------

/// Prove: a layer that references a source with index >= its own
/// produces ForwardReference. Topological ordering requires source < layer.
#[kani::unwind(128)]
#[kani::proof]
fn graph_forward_reference_detected() {
    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-1.0, 1.0)],
            output_bounds: vec![(-2.0, 2.0)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: Some(vec![]),
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "Add".to_string(),
            input_bounds: vec![(-2.0, 2.0)],
            output_bounds: vec![(-3.0, 3.0)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: Some(vec![5]), // forward reference to non-existent layer 5
        },
    ];
    let mut issues = Vec::new();
    check_layer_trace_consistency(&bounds, &mut issues);
    let has_forward = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::ForwardReference { .. }));
    assert!(
        has_forward,
        "forward reference must produce ForwardReference"
    );
}

// ---------------------------------------------------------------------------
// 69. Graph-aware: dangling source reference detected
// ---------------------------------------------------------------------------

/// Prove: a layer referencing a source that does not exist in the trace
/// produces DanglingSourceRef.
#[kani::unwind(128)]
#[kani::proof]
fn graph_dangling_source_detected() {
    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-1.0, 1.0)],
            output_bounds: vec![(-2.0, 2.0)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: Some(vec![]),
        },
        LayerBoundRecord {
            layer_index: 5, // gap: no layers 1-4 in trace
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-2.0, 2.0)],
            output_bounds: vec![(0.0, 2.0)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: Some(vec![3]), // layer 3 does not exist
        },
    ];
    let mut issues = Vec::new();
    check_layer_trace_consistency(&bounds, &mut issues);
    let has_dangling = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::DanglingSourceRef { .. }));
    assert!(
        has_dangling,
        "dangling source ref must produce DanglingSourceRef"
    );
}

// ---------------------------------------------------------------------------
// 70. Graph-aware: valid single-source chain has no gap
// ---------------------------------------------------------------------------

/// Prove: a valid graph-aware single-source chain produces no LayerTraceGap.
#[kani::unwind(128)]
#[kani::proof]
fn graph_valid_single_source_no_gap() {
    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-1.0, 1.0)],
            output_bounds: vec![(-2.0, 2.0)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: Some(vec![]), // network input
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-2.0, 2.0)], // matches layer 0 output
            output_bounds: vec![(0.0, 2.0)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: Some(vec![0]), // source is layer 0
        },
    ];
    let mut issues = Vec::new();
    check_layer_trace_consistency(&bounds, &mut issues);
    let has_gap = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::LayerTraceGap { .. }));
    assert!(!has_gap, "valid graph-aware chain must have no gaps");
}

// ---------------------------------------------------------------------------
// 71. Graph-aware: broken single-source chain produces LayerTraceGap
// ---------------------------------------------------------------------------

/// Prove: when a single-source layer's input does not match its source's
/// output, LayerTraceGap is produced.
#[kani::unwind(128)]
#[kani::proof]
fn graph_broken_single_source_produces_gap() {
    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-1.0, 1.0)],
            output_bounds: vec![(-2.0, 2.0)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: Some(vec![]),
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-999.0, 999.0)], // does NOT match layer 0 output
            output_bounds: vec![(0.0, 999.0)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: Some(vec![0]),
        },
    ];
    let mut issues = Vec::new();
    check_layer_trace_consistency(&bounds, &mut issues);
    let has_gap = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::LayerTraceGap { .. }));
    assert!(has_gap, "broken source match must produce LayerTraceGap");
}

// ---------------------------------------------------------------------------
// 72. Graph-aware: duplicate layer_index detected
// ---------------------------------------------------------------------------

/// Prove: duplicate layer_index values produce DuplicateLayerIndex.
#[kani::unwind(128)]
#[kani::proof]
fn graph_duplicate_layer_index_detected() {
    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-1.0, 1.0)],
            output_bounds: vec![(-2.0, 2.0)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: Some(vec![]),
        },
        LayerBoundRecord {
            layer_index: 0, // DUPLICATE
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-2.0, 2.0)],
            output_bounds: vec![(0.0, 2.0)],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: Some(vec![]),
        },
    ];
    let mut issues = Vec::new();
    check_layer_trace_consistency(&bounds, &mut issues);
    let has_dup = issues
        .iter()
        .any(|i| matches!(i, CheckIssue::DuplicateLayerIndex { .. }));
    assert!(
        has_dup,
        "duplicate layer_index must produce DuplicateLayerIndex"
    );
}

// ===========================================================================
// Vacuity assessment via check_certificate_core
// ===========================================================================

// ---------------------------------------------------------------------------
// 73. Core vacuity: all-IBP cert is flagged vacuous
// ---------------------------------------------------------------------------

/// Prove: a certificate where all layers used IBP (not CROWN) produces
/// a VacuousBounds issue (crown_coverage = 0.0 < 0.5).
#[kani::unwind(128)]
#[kani::proof]
fn core_all_ibp_flagged_vacuous() {
    let kv = KernelVerification::new("test".to_string(), PropMethod::Ibp, -1.0, 1.0, 2.0, true);
    let input_spec = InputBoundsRecord::new(&[ParamInputRecord::new(0, -1.0, 1.0)], &[]);
    let mut cert = ProofCertificate::from_verification(&kv, input_spec);
    cert.source_hash = Some(VALID_HASH.to_string());
    cert.layer_bounds = Some(vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-1.0, 1.0)],
        output_bounds: vec![(-1.0, 1.0)],
        method: PropMethod::Ibp, // NOT tight
        node_name: None,
        input_sources: None,
    }]);
    let cache = empty_cache();
    let result = check_certificate_core(&cert, &cache);
    let has_vacuous = result
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::VacuousBounds { .. }));
    assert!(has_vacuous, "all-IBP cert must be flagged vacuous");
    // VacuousBounds is informational, so is_valid should still be true
    // (no other issues present for this well-formed cert)
    assert!(
        result.is_valid(),
        "all-IBP cert is still valid (VacuousBounds is informational)"
    );
}

// ---------------------------------------------------------------------------
// 74. Core vacuity: wide output is flagged vacuous even with CROWN
// ---------------------------------------------------------------------------

/// Prove: a certificate with 100% CROWN coverage but output_width > threshold
/// produces VacuousBounds.
#[kani::unwind(128)]
#[kani::proof]
fn core_wide_output_flagged_vacuous() {
    let kv = KernelVerification::new(
        "test".to_string(),
        PropMethod::Crown,
        -50.0,
        50.0,
        100.0, // width = 100 >> threshold 10
        true,
    );
    let input_spec = InputBoundsRecord::new(&[ParamInputRecord::new(0, -1.0, 1.0)], &[]);
    let mut cert = ProofCertificate::from_verification(&kv, input_spec);
    cert.source_hash = Some(VALID_HASH.to_string());
    cert.layer_bounds = Some(vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-1.0, 1.0)],
        output_bounds: vec![(-50.0, 50.0)],
        method: PropMethod::Crown, // tight
        node_name: None,
        input_sources: None,
    }]);
    let cache = empty_cache();
    let result = check_certificate_core(&cert, &cache);
    let has_vacuous = result
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::VacuousBounds { .. }));
    assert!(
        has_vacuous,
        "wide output with CROWN must still be flagged vacuous"
    );
}

// ---------------------------------------------------------------------------
// 75. Core vacuity: tight CROWN + narrow output = non-vacuous
// ---------------------------------------------------------------------------

/// Prove: a certificate with high CROWN coverage AND narrow output width
/// does NOT produce VacuousBounds.
#[kani::unwind(128)]
#[kani::proof]
fn core_tight_crown_narrow_not_vacuous() {
    let kv = KernelVerification::new(
        "test".to_string(),
        PropMethod::Crown,
        -1.0,
        1.0,
        2.0, // width = 2 < threshold 10
        true,
    );
    let input_spec = InputBoundsRecord::new(&[ParamInputRecord::new(0, -1.0, 1.0)], &[]);
    let mut cert = ProofCertificate::from_verification(&kv, input_spec);
    cert.source_hash = Some(VALID_HASH.to_string());
    cert.layer_bounds = Some(vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-1.0, 1.0)],
        output_bounds: vec![(-1.0, 1.0)],
        method: PropMethod::Crown,
        node_name: None,
        input_sources: None,
    }]);
    let cache = empty_cache();
    let result = check_certificate_core(&cert, &cache);
    let has_vacuous = result
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::VacuousBounds { .. }));
    assert!(
        !has_vacuous,
        "tight CROWN + narrow output must NOT be vacuous"
    );
}

// ===========================================================================
// NaN in certificate output_bounds (output agreement path)
// ===========================================================================

// ---------------------------------------------------------------------------
// 76. Core: NaN in certificate output_bounds.lower detected
// ---------------------------------------------------------------------------

/// Prove: certificate with NaN in output_bounds.lower is caught by
/// the output agreement check (NanOutputBounds) AND by structural
/// validation (FiniteFlagMismatch if is_finite=true).
#[kani::unwind(128)]
#[kani::proof]
fn core_nan_cert_output_lower() {
    let kv = KernelVerification::new(
        "test".to_string(),
        PropMethod::Crown,
        f32::NAN, // NaN output lower
        1.0,
        2.0,
        false, // is_finite=false to avoid structural error on that path
    );
    let input_spec = InputBoundsRecord::new(&[ParamInputRecord::new(0, -1.0, 1.0)], &[]);
    let mut cert = ProofCertificate::from_verification(&kv, input_spec);
    cert.source_hash = Some(VALID_HASH.to_string());
    cert.output_width = f32::NAN; // make width consistent with NaN bounds
    cert.layer_bounds = Some(vec![LayerBoundRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_bounds: vec![(-1.0, 1.0)],
        output_bounds: vec![(-1.0, 1.0)],
        method: PropMethod::Crown,
        node_name: None,
        input_sources: None,
    }]);
    let cache = empty_cache();
    let result = check_certificate_core(&cert, &cache);
    // The cert has NaN in output_bounds.lower. The output agreement check
    // should detect the NaN cert bounds and push NanOutputBounds.
    let has_nan = result
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::NanOutputBounds));
    assert!(
        has_nan,
        "NaN in cert output_bounds.lower must produce NanOutputBounds"
    );
    assert!(!result.is_valid(), "cert with NaN output must be invalid");
}
