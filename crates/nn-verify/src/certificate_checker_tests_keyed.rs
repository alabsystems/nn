// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `check_certificate_with_key` and `check_bundle_with_key` (#3325).

use super::checker_test_shared::{consistent_layer_bounds, sample_input_spec, sample_verification};
use super::*;

/// Signed cert with correct key: no signature issues.
#[test]
fn test_check_certificate_with_key_valid_signature() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds());

    let key = b"test-keyed-check-key";
    crate::certificate::integrity::sign_certificate(&mut cert, key).unwrap();

    let check = check_certificate_with_key(&cert, None, None, Some(key));
    let sig_issues: Vec<_> = check
        .issues
        .iter()
        .filter(|i| {
            matches!(
                i,
                CheckIssue::SignatureInvalid { .. } | CheckIssue::SignatureKeyError { .. }
            )
        })
        .collect();
    assert!(
        sig_issues.is_empty(),
        "valid signed cert should have no signature issues: {sig_issues:?}"
    );
}

/// Signed cert with wrong key: SignatureInvalid.
#[test]
fn test_check_certificate_with_key_wrong_key() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds());

    let key = b"correct-key";
    crate::certificate::integrity::sign_certificate(&mut cert, key).unwrap();

    let check = check_certificate_with_key(&cert, None, None, Some(b"wrong-key"));
    let sig_invalid = check
        .issues
        .iter()
        .any(|i| matches!(i, CheckIssue::SignatureInvalid { .. }));
    assert!(
        sig_invalid,
        "wrong key should produce SignatureInvalid, got: {:?}",
        check.issues
    );
}

/// Signed cert checked without key (None): no HMAC check, backward compatible.
#[test]
fn test_check_certificate_with_key_none_skips_hmac() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds());

    let key = b"some-key";
    crate::certificate::integrity::sign_certificate(&mut cert, key).unwrap();

    let check = check_certificate_with_key(&cert, None, None, None);
    let sig_issues: Vec<_> = check
        .issues
        .iter()
        .filter(|i| {
            matches!(
                i,
                CheckIssue::SignatureInvalid { .. } | CheckIssue::SignatureKeyError { .. }
            )
        })
        .collect();
    assert!(
        sig_issues.is_empty(),
        "None key should skip HMAC: {sig_issues:?}"
    );
}

/// Unsigned cert checked with key: SignatureKeyError (missing fields).
#[test]
fn test_check_certificate_with_key_unsigned_cert() {
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds());

    let check = check_certificate_with_key(&cert, None, None, Some(b"any-key"));
    let key_errors: Vec<_> = check
        .issues
        .iter()
        .filter(|i| matches!(i, CheckIssue::SignatureKeyError { .. }))
        .collect();
    assert!(
        !key_errors.is_empty(),
        "unsigned cert with key should produce SignatureKeyError, got: {:?}",
        check.issues
    );
}

/// check_certificate delegates to check_certificate_with_key(None).
#[test]
fn test_check_certificate_delegates_to_with_key() {
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds());

    let check_old = check_certificate(&cert, None, None);
    let check_new = check_certificate_with_key(&cert, None, None, None);

    assert_eq!(check_old.issues.len(), check_new.issues.len());
    assert_eq!(check_old.is_valid(), check_new.is_valid());
}
