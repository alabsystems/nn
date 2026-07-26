// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SMT proof consistency checking in the certificate checker.
//!
//! Covers `check_smt_proof_consistency` (certificate_checker.rs): #3095.
//! Four branches:
//! 1. smt_outcome == "Proven" but no proof artifact → SmtProofMissing
//! 2. proof artifact present with verdict Invalid → SmtProofInvalid
//! 3. proof artifact present with verdict Verified → no issue
//! 4. No smt_outcome or no proof → no issue (pre-v3 certificates)

use super::checker_test_shared::{
    consistent_layer_bounds_with_method, sample_input_spec, sample_verification_with_bounds,
};
use super::*;
use crate::certificate::ProofCertificate;
use crate::certificate_types::LayerBoundRecord;
use crate::status::SmtProofVerdict;
use crate::verify_types::{KernelVerification, PropMethod};

fn sample_verification() -> KernelVerification {
    let mut v = sample_verification_with_bounds(-5.0, 5.0);
    v.method = PropMethod::Crown;
    v
}

fn consistent_layer_bounds() -> Vec<LayerBoundRecord> {
    // SMT proof tests use Crown-only 2-layer trace (no final Linear layer).
    let mut bounds = consistent_layer_bounds_with_method(PropMethod::Crown);
    // Override ReLU output to match this test's convention: (-5.0, 5.0) not (0.0, 5.0)
    bounds[1].output_bounds = vec![(-5.0, 5.0)];
    bounds.truncate(2);
    bounds
}

/// Build a fully valid base certificate (passes all non-SMT checks).
fn valid_base_cert() -> ProofCertificate {
    ProofCertificate::from_verification(&sample_verification(), sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds())
        .with_source_hash("a".repeat(64))
}

// ---------------------------------------------------------------------------
// Branch 1: smt_outcome == "Proven" but no proof artifact → SmtProofMissing
// ---------------------------------------------------------------------------

/// When smt_outcome is "Proven" but no Alethe proof is attached,
/// the checker must report SmtProofMissing.
#[test]
fn test_smt_proven_without_proof_artifact_reports_missing() {
    let cert = valid_base_cert().with_smt_outcome("Proven");
    // No smt_proof attached — smt_proof_alethe is None.

    let check = check_certificate(&cert, None, None);
    let has_missing = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::SmtProofMissing));
    assert!(
        has_missing,
        "Proven outcome without proof artifact must report SmtProofMissing: {:?}",
        check.issues
    );
    // SmtProofMissing is NOT informational — is_valid() returns false (#3221).
    assert!(
        !check.is_valid(),
        "SmtProofMissing must fail is_valid() — Proven without proof artifact is unverifiable: {:?}",
        check.issues
    );
}

// ---------------------------------------------------------------------------
// Branch 2: Proof artifact present with verdict Invalid → SmtProofInvalid
// ---------------------------------------------------------------------------

/// When a proof artifact is present but verdict is Invalid,
/// the checker must report SmtProofInvalid (validity failure).
#[test]
fn test_smt_proof_with_invalid_verdict_reports_invalid() {
    let cert = valid_base_cert()
        .with_smt_outcome("Proven")
        .with_smt_proof("(proof ...)".to_string(), SmtProofVerdict::Invalid);

    let check = check_certificate(&cert, None, None);
    let has_invalid = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::SmtProofInvalid));
    assert!(
        has_invalid,
        "proof with Invalid verdict must report SmtProofInvalid: {:?}",
        check.issues
    );
    // SmtProofInvalid is a real validity failure.
    assert!(
        !check.is_valid(),
        "SmtProofInvalid should cause is_valid() to return false"
    );
}

// ---------------------------------------------------------------------------
// Branch 3: Proof artifact with Verified verdict → clean
// ---------------------------------------------------------------------------

/// A certificate with Proven outcome, proof artifact, and Verified verdict
/// should not produce any SMT-related issues.
#[test]
fn test_smt_proven_with_verified_proof_clean() {
    let cert = valid_base_cert()
        .with_smt_outcome("Proven")
        .with_smt_proof("(proof (step ...))".to_string(), SmtProofVerdict::Verified);

    let check = check_certificate(&cert, None, None);
    let smt_issues: Vec<_> = check
        .issues
        .iter()
        .filter(|i| matches!(i, CheckIssue::SmtProofMissing | CheckIssue::SmtProofInvalid))
        .collect();
    assert!(
        smt_issues.is_empty(),
        "Proven + Verified proof should have no SMT issues: {smt_issues:?}"
    );
    assert!(check.is_valid(), "should be valid: {:?}", check.issues);
}

// ---------------------------------------------------------------------------
// Branch 4: No smt_outcome → no SMT issues (pre-v3 compatibility)
// ---------------------------------------------------------------------------

/// Pre-v3 certificates with no smt_outcome should not produce SMT issues.
#[test]
fn test_no_smt_outcome_no_smt_issues() {
    let cert = valid_base_cert();
    // No smt_outcome, no proof — pre-v3 certificate.

    let check = check_certificate(&cert, None, None);
    let smt_issues: Vec<_> = check
        .issues
        .iter()
        .filter(|i| matches!(i, CheckIssue::SmtProofMissing | CheckIssue::SmtProofInvalid))
        .collect();
    assert!(
        smt_issues.is_empty(),
        "pre-v3 cert should have no SMT issues: {smt_issues:?}"
    );
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

/// smt_outcome is some non-"Proven" value (e.g. "Unknown") — no SmtProofMissing.
#[test]
fn test_smt_outcome_not_proven_no_missing_issue() {
    let cert = valid_base_cert().with_smt_outcome("Unknown");

    let check = check_certificate(&cert, None, None);
    assert!(
        !check
            .issues
            .iter()
            .any(|i| matches!(i, CheckIssue::SmtProofMissing)),
        "non-Proven outcome should not trigger SmtProofMissing: {:?}",
        check.issues
    );
}

/// Proof artifact present with Unchecked verdict — no SmtProofInvalid.
/// Unchecked is legitimate (proof not yet validated).
#[test]
fn test_smt_proof_unchecked_verdict_no_invalid_issue() {
    let cert = valid_base_cert()
        .with_smt_outcome("Proven")
        .with_smt_proof("(proof ...)".to_string(), SmtProofVerdict::Unchecked);

    let check = check_certificate(&cert, None, None);
    assert!(
        !check
            .issues
            .iter()
            .any(|i| matches!(i, CheckIssue::SmtProofInvalid)),
        "Unchecked verdict should not trigger SmtProofInvalid: {:?}",
        check.issues
    );
    // Also no SmtProofMissing — proof IS present.
    assert!(
        !check
            .issues
            .iter()
            .any(|i| matches!(i, CheckIssue::SmtProofMissing)),
        "proof is present, should not trigger SmtProofMissing: {:?}",
        check.issues
    );
}

/// Proof artifact present without smt_outcome — no issues.
/// This tests the case where a proof was attached but the outcome field
/// is missing (e.g. manually constructed certificate).
#[test]
fn test_proof_artifact_without_outcome_no_issues() {
    let mut cert = valid_base_cert();
    cert.smt_proof_alethe = Some("(proof ...)".to_string());
    cert.smt_proof_verdict = Some(SmtProofVerdict::Verified);
    // smt_outcome is None — not claimed Proven.

    let check = check_certificate(&cert, None, None);
    let smt_issues: Vec<_> = check
        .issues
        .iter()
        .filter(|i| matches!(i, CheckIssue::SmtProofMissing | CheckIssue::SmtProofInvalid))
        .collect();
    assert!(
        smt_issues.is_empty(),
        "proof without outcome should have no SMT issues: {smt_issues:?}"
    );
}

/// smt_outcome == "Counterexample" with no proof → no SmtProofMissing.
/// Only "Proven" triggers the missing proof check.
#[test]
fn test_counterexample_outcome_no_missing_proof() {
    let cert = valid_base_cert().with_smt_outcome("Counterexample");

    let check = check_certificate(&cert, None, None);
    assert!(
        !check
            .issues
            .iter()
            .any(|i| matches!(i, CheckIssue::SmtProofMissing)),
        "Counterexample should not trigger SmtProofMissing"
    );
}
