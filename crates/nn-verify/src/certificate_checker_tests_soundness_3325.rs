// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Security model tests for #3325: HMAC signature verification in checker.
//!
//! The certificate checker has two modes:
//! - `check_certificate()` — backward-compatible, no key, validates content_hash only.
//! - `check_certificate_with_key()` — keyed mode, validates HMAC signature (#3325 fix).
//!
//! ## Attack model
//!
//! Without a key (`check_certificate`):
//! - **Recompute attack:** Modify fields → recompute content_hash → passes (by design).
//! - **Strip attack:** Remove content_hash + hmac_signature → passes (by design).
//!
//! With a key (`check_certificate_with_key`):
//! - Both attacks are caught because the attacker cannot forge the HMAC.
//!
//! These tests validate both the backward-compatible (keyless) behavior and
//! the keyed defense against sophisticated tampering attacks.

use super::checker_test_shared::{consistent_layer_bounds, sample_input_spec, sample_verification};
use super::*;
use crate::certificate::ProofCertificate;

// ---------------------------------------------------------------------------
// Attack 1: Recompute content_hash after tampering
// ---------------------------------------------------------------------------

/// Keyless mode: recomputed content_hash bypasses keyless checker (by design).
///
/// `check_certificate()` delegates to `check_certificate_with_key(..., None)`,
/// so HMAC verification is skipped. An attacker who recomputes content_hash
/// after tampering passes the keyless check. This is correct backward-compatible
/// behavior for pre-v4 unsigned certificates.
#[test]
fn test_gap_3325_recomputed_content_hash_bypasses_checker() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds())
        .with_source_hash("a".repeat(64));

    // Step 1: Sign the certificate with a key (creates valid content_hash + hmac_signature).
    let key = b"test-gap-3325-signing-key-32byte";
    crate::certificate::integrity::sign_certificate(&mut cert, key).unwrap();

    // Baseline: signed cert passes the checker.
    let baseline = check_certificate(&cert, None, None);
    assert!(
        baseline.is_valid(),
        "baseline signed cert should pass checker: {:?}",
        baseline.issues
    );

    // Step 2: Tamper — change kernel_name (simulates rebinding cert to different kernel).
    cert.kernel_name = "attacker_kernel".to_string();

    // Step 3: Recompute content_hash (no key needed — public algorithm).
    // The attacker uses the same canonical_json + SHA-256 that the signer uses.
    let recomputed_hash = crate::certificate::integrity::compute_content_hash(&cert).unwrap();
    cert.content_hash = Some(recomputed_hash);
    // hmac_signature is still the OLD value (computed over the original content_hash).
    // The attacker cannot recompute it without the key.

    // Step 4: GAP — checker passes because content_hash matches.
    let tampered_check = check_certificate(&cert, None, None);
    assert!(
        tampered_check.is_valid(),
        "GAP #3325: tampered cert with recomputed content_hash passes checker. \
         Issues: {:?}",
        tampered_check.issues
    );
    // Verify no ContentHashMismatch or SignatureInvalid reported.
    assert!(
        !tampered_check.issues.iter().any(|i| matches!(
            i,
            CheckIssue::ContentHashMismatch { .. } | CheckIssue::SignatureInvalid { .. }
        )),
        "checker should report no integrity issues for recomputed hash: {:?}",
        tampered_check.issues
    );

    // Step 5: verify_signature() DOES catch the tampering.
    let sig_result = verify_signature(&cert, key);
    assert!(
        sig_result.is_err(),
        "verify_signature must catch tampered cert with recomputed content_hash"
    );
}

// ---------------------------------------------------------------------------
// Attack 2: Strip integrity fields entirely
// ---------------------------------------------------------------------------

/// Keyless mode: stripped integrity fields bypass keyless checker (by design).
///
/// `check_content_hash_integrity()` only runs when `content_hash` is `Some`.
/// If both integrity fields are `None`, the keyless checker accepts the
/// unsigned state (backward compatibility for pre-v4 certificates).
#[test]
fn test_gap_3325_stripped_integrity_fields_bypass_checker() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds())
        .with_source_hash("a".repeat(64));

    // Sign the certificate.
    let key = b"test-gap-3325-signing-key-32byte";
    crate::certificate::integrity::sign_certificate(&mut cert, key).unwrap();

    // Baseline: signed cert passes.
    let baseline = check_certificate(&cert, None, None);
    assert!(baseline.is_valid());

    // Tamper: change output bounds to make false claims.
    cert.kernel_name = "attacker_kernel".to_string();

    // Strip: remove all integrity evidence.
    cert.content_hash = None;
    cert.hmac_signature = None;

    // GAP: checker passes — integrity check is a no-op without content_hash.
    let stripped_check = check_certificate(&cert, None, None);
    assert!(
        stripped_check.is_valid(),
        "GAP #3325: stripped integrity fields bypass checker. Issues: {:?}",
        stripped_check.issues
    );
    // No integrity-related issues at all.
    assert!(
        !stripped_check.issues.iter().any(|i| matches!(
            i,
            CheckIssue::ContentHashMismatch { .. } | CheckIssue::SignatureInvalid { .. }
        )),
        "no integrity issues reported for stripped cert: {:?}",
        stripped_check.issues
    );

    // verify_signature catches the stripping.
    let sig_result = verify_signature(&cert, key);
    assert!(
        sig_result.is_err(),
        "verify_signature must reject stripped cert"
    );
}

// ---------------------------------------------------------------------------
// Contrast: naive tampering WITHOUT recomputing hash IS detected
// ---------------------------------------------------------------------------

/// Contrast test: tampering without recomputing content_hash IS detected.
///
/// This confirms the checker's content_hash validation works for the naive
/// case. The gap is specifically about sophisticated attackers who recompute
/// the hash (Attack 1) or strip it (Attack 2).
#[test]
fn test_naive_tampering_without_recompute_is_detected() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds())
        .with_source_hash("a".repeat(64));

    let key = b"test-gap-3325-signing-key-32byte";
    crate::certificate::integrity::sign_certificate(&mut cert, key).unwrap();

    // Tamper without recomputing content_hash.
    cert.kernel_name = "attacker_kernel".to_string();
    // content_hash still holds hash of original content — now mismatches.

    let check = check_certificate(&cert, None, None);
    assert!(
        check
            .issues
            .iter()
            .any(|i| matches!(i, CheckIssue::ContentHashMismatch { .. })),
        "naive tampering (no hash recompute) must produce ContentHashMismatch: {:?}",
        check.issues
    );
    assert!(!check.is_valid(), "naive tampering must fail is_valid()");
}

// ---------------------------------------------------------------------------
// Keyed mode: check_certificate_with_key catches both attacks (#3325 fix)
// ---------------------------------------------------------------------------

/// Keyed mode catches recompute attack: tampered cert with recomputed
/// content_hash is detected when a signing key is provided.
#[test]
fn test_keyed_mode_catches_recompute_attack() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds())
        .with_source_hash("a".repeat(64));

    let key = b"test-gap-3325-signing-key-32byte";
    crate::certificate::integrity::sign_certificate(&mut cert, key).unwrap();

    // Tamper + recompute content_hash (same as keyless test).
    cert.kernel_name = "attacker_kernel".to_string();
    let recomputed_hash = crate::certificate::integrity::compute_content_hash(&cert).unwrap();
    cert.content_hash = Some(recomputed_hash);

    // Keyed mode catches it: HMAC over new content_hash doesn't match old signature.
    let check = check_certificate_with_key(&cert, None, None, Some(key));
    let has_sig_issue = check.issues.iter().any(|i| {
        matches!(
            i,
            CheckIssue::SignatureInvalid { .. } | CheckIssue::SignatureKeyError { .. }
        )
    });
    assert!(
        has_sig_issue,
        "keyed mode must catch recompute attack: {:?}",
        check.issues
    );
}

/// Keyed mode catches strip attack: stripped integrity fields produce
/// SignatureKeyError when a signing key is provided.
#[test]
fn test_keyed_mode_catches_strip_attack() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds())
        .with_source_hash("a".repeat(64));

    let key = b"test-gap-3325-signing-key-32byte";
    crate::certificate::integrity::sign_certificate(&mut cert, key).unwrap();

    // Tamper + strip (same as keyless test).
    cert.kernel_name = "attacker_kernel".to_string();
    cert.content_hash = None;
    cert.hmac_signature = None;

    // Keyed mode catches it: missing content_hash triggers SignatureKeyError.
    let check = check_certificate_with_key(&cert, None, None, Some(key));
    let has_key_error = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::SignatureKeyError { .. }));
    assert!(
        has_key_error,
        "keyed mode must catch strip attack with SignatureKeyError: {:?}",
        check.issues
    );
}
