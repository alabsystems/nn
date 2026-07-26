// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for certificate cryptographic integrity (HMAC-SHA256).
//!
//! Part of #3222.

use super::*;
use crate::certificate::certificate_test_helpers::{sample_input_spec, sample_verification};
use crate::certificate::ProofCertificate;

fn make_cert() -> ProofCertificate {
    ProofCertificate::from_verification(&sample_verification(), sample_input_spec())
}

#[test]
fn test_compute_content_hash_deterministic() {
    let cert = make_cert();
    let h1 = compute_content_hash(&cert).unwrap();
    let h2 = compute_content_hash(&cert).unwrap();
    assert_eq!(h1, h2, "content hash must be deterministic");
    assert_eq!(h1.len(), 64, "SHA-256 hex is 64 chars");
}

#[test]
fn test_content_hash_changes_on_modification() {
    let cert = make_cert();
    let h1 = compute_content_hash(&cert).unwrap();

    let mut modified = cert;
    modified.output_width = 999.0;
    let h2 = compute_content_hash(&modified).unwrap();
    assert_ne!(h1, h2, "different content must produce different hash");
}

#[test]
fn test_content_hash_ignores_integrity_fields() {
    let cert = make_cert();
    let h1 = compute_content_hash(&cert).unwrap();

    let mut with_hash = cert.clone();
    with_hash.content_hash = Some("deadbeef".repeat(8));
    let h2 = compute_content_hash(&with_hash).unwrap();
    assert_eq!(h1, h2, "content_hash field must be excluded from hash");

    let mut with_sig = cert;
    with_sig.hmac_signature = Some("cafebabe".repeat(8));
    let h3 = compute_content_hash(&with_sig).unwrap();
    assert_eq!(h1, h3, "hmac_signature field must be excluded from hash");
}

#[test]
fn test_sign_and_verify_roundtrip() {
    let mut cert = make_cert();
    let key = b"test-signing-key-for-nn-certificates";

    sign_certificate(&mut cert, key).unwrap();
    assert!(cert.content_hash.is_some());
    assert!(cert.hmac_signature.is_some());

    // Verify should pass.
    verify_signature(&cert, key).unwrap();
    verify_content_hash(&cert).unwrap();
}

#[test]
fn test_verify_detects_tampered_bounds() {
    let mut cert = make_cert();
    let key = b"test-key";

    sign_certificate(&mut cert, key).unwrap();

    // Tamper with bounds.
    cert.output_width = 0.001;

    // Content hash should fail.
    let result = verify_content_hash(&cert);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        IntegrityError::ContentHashMismatch { .. }
    ));
}

#[test]
fn test_verify_detects_wrong_key() {
    let mut cert = make_cert();
    let key = b"correct-key";
    let wrong_key = b"wrong-key";

    sign_certificate(&mut cert, key).unwrap();

    // Signature should fail with wrong key.
    let result = verify_signature(&cert, wrong_key);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        IntegrityError::SignatureInvalid
    ));
}

#[test]
fn test_verify_missing_content_hash() {
    let cert = make_cert();
    // No integrity fields — should report missing.
    let result = verify_content_hash(&cert);
    assert!(matches!(
        result.unwrap_err(),
        IntegrityError::MissingContentHash
    ));
}

#[test]
fn test_verify_missing_signature() {
    let mut cert = make_cert();
    let key = b"test-key";

    // Set content hash but not signature.
    cert.content_hash = Some(compute_content_hash(&cert).unwrap());

    let result = verify_signature(&cert, key);
    assert!(matches!(
        result.unwrap_err(),
        IntegrityError::MissingSignature
    ));
}

#[test]
fn test_sign_bundle_roundtrip() {
    let cert1 = make_cert();
    let mut cert2 = make_cert();
    cert2.kernel_name = "elu".to_string();

    let mut bundle = CertificateBundle::new("test_model")
        .with_certificate(cert1)
        .with_certificate(cert2);

    let key = b"bundle-key";
    sign_bundle(&mut bundle, key).unwrap();

    // All certs should be signed.
    for cert in &bundle.certificates {
        assert!(cert.content_hash.is_some());
        assert!(cert.hmac_signature.is_some());
    }

    // Bundle verification should pass.
    verify_bundle_signatures(&bundle, key).unwrap();
}

#[test]
fn test_bundle_verification_detects_tampered_cert() {
    let cert1 = make_cert();
    let mut cert2 = make_cert();
    cert2.kernel_name = "elu".to_string();

    let mut bundle = CertificateBundle::new("test_model")
        .with_certificate(cert1)
        .with_certificate(cert2);

    let key = b"bundle-key";
    sign_bundle(&mut bundle, key).unwrap();

    // Tamper with second certificate.
    bundle.certificates[1].is_finite = false;

    let result = verify_bundle_signatures(&bundle, key);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.certificate_index, 1);
    assert_eq!(err.kernel_name, "elu");
}

#[test]
fn test_bundle_skips_unsigned_certs() {
    let cert = make_cert();
    let bundle = CertificateBundle::new("test_model").with_certificate(cert);

    // No signatures — should pass (pre-v4 certificates are skipped).
    let key = b"any-key";
    verify_bundle_signatures(&bundle, key).unwrap();
}

#[test]
fn test_checker_validates_content_hash() {
    use crate::certificate_checker::check_certificate;

    let mut cert = make_cert();
    let key = b"checker-test-key";
    sign_certificate(&mut cert, key).unwrap();

    // Valid signature — checker should find no content hash issues.
    let result = check_certificate(&cert, None, None);
    assert!(
        !result.issues.iter().any(|i| matches!(
            i,
            crate::certificate_checker::CheckIssue::ContentHashMismatch { .. }
        )),
        "signed cert should have no content hash issues"
    );

    // Tamper and re-check.
    cert.output_width = 0.001;
    let result = check_certificate(&cert, None, None);
    assert!(
        result.issues.iter().any(|i| matches!(
            i,
            crate::certificate_checker::CheckIssue::ContentHashMismatch { .. }
        )),
        "tampered cert should have ContentHashMismatch"
    );
    assert!(!result.is_valid(), "tampered cert should not be valid");
}

#[test]
fn test_validate_accepts_v4_hash_fields() {
    let mut cert = make_cert();
    let key = b"test-key";
    sign_certificate(&mut cert, key).unwrap();

    // validate() should accept well-formed SHA-256 hex in integrity fields.
    assert!(cert.validate().is_ok());
}

#[test]
fn test_validate_rejects_malformed_content_hash() {
    let mut cert = make_cert();
    cert.content_hash = Some("not-a-valid-hash".to_string());

    let result = cert.validate();
    assert!(result.is_err());
}

#[test]
fn test_validate_rejects_malformed_hmac_signature() {
    let mut cert = make_cert();
    cert.hmac_signature = Some("too-short".to_string());

    let result = cert.validate();
    assert!(result.is_err());
}

#[test]
fn test_signature_stripping_attack_detected() {
    // Verify that stripping integrity fields from a signed certificate
    // is detected by verify_bundle_signatures_strict.
    let mut cert = make_cert();
    let key = b"signing-key";
    sign_certificate(&mut cert, key).unwrap();

    // Attacker strips both integrity fields.
    cert.content_hash = None;
    cert.hmac_signature = None;

    let bundle = CertificateBundle::new("test_model").with_certificate(cert);

    // Lenient mode: unsigned certs silently skipped (pre-v4 compat).
    let result = verify_bundle_signatures(&bundle, key);
    assert!(result.is_ok(), "lenient mode skips unsigned certs");

    // Strict mode: rejects the stripped cert.
    let result = verify_bundle_signatures_strict(&bundle, key);
    assert!(result.is_err(), "strict mode rejects stripped signatures");
    let err = result.unwrap_err();
    assert_eq!(err.certificate_index, 0);
    assert!(matches!(err.error, IntegrityError::MissingContentHash));
}

#[test]
fn test_strict_verification_passes_with_signatures() {
    let mut cert = make_cert();
    let key = b"strict-test-key";
    sign_certificate(&mut cert, key).unwrap();

    let bundle = CertificateBundle::new("test_model").with_certificate(cert);

    // Strict mode passes when all certs are properly signed.
    verify_bundle_signatures_strict(&bundle, key).unwrap();
}

#[test]
fn test_partial_signature_strip_detected() {
    // Stripping only hmac_signature (keeping content_hash) must fail.
    let mut cert = make_cert();
    let key = b"signing-key";
    sign_certificate(&mut cert, key).unwrap();

    // Attacker removes only the signature, keeping content_hash.
    cert.hmac_signature = None;

    let bundle = CertificateBundle::new("test_model").with_certificate(cert);
    let result = verify_bundle_signatures(&bundle, key);
    assert!(result.is_err(), "partial strip must be detected");
    assert!(matches!(
        result.unwrap_err().error,
        IntegrityError::MissingSignature,
    ));
}

#[test]
fn test_verify_signature_uses_constant_time_comparison() {
    // Verify that signing then verifying with the correct key succeeds
    // (exercises the constant-time Mac::verify_slice path).
    let mut cert = make_cert();
    let key = b"constant-time-test-key-32-bytes!";
    sign_certificate(&mut cert, key).unwrap();

    // This exercises the hex_decode → verify_slice path.
    verify_signature(&cert, key).expect("constant-time verify should pass");

    // Wrong key still fails.
    let result = verify_signature(&cert, b"wrong-key");
    assert!(matches!(
        result.unwrap_err(),
        IntegrityError::SignatureInvalid,
    ));
}

#[test]
fn test_verify_rejects_corrupted_hex_signature() {
    // If the hex in hmac_signature is corrupted, verification should fail
    // (exercises the hex_decode error path in the new constant-time verify).
    let mut cert = make_cert();
    let key = b"test-key";
    sign_certificate(&mut cert, key).unwrap();

    // Corrupt the signature with non-hex chars.
    cert.hmac_signature = Some("zzzz".to_string());
    let result = verify_signature(&cert, key);
    assert!(matches!(
        result.unwrap_err(),
        IntegrityError::SignatureInvalid,
    ));
}

#[test]
fn test_partial_strip_content_hash_only() {
    // Stripping only content_hash (keeping hmac_signature) must fail.
    // Different error path from test_partial_signature_strip_detected.
    let mut cert = make_cert();
    let key = b"signing-key";
    sign_certificate(&mut cert, key).unwrap();

    // Attacker removes only the content hash, keeping signature.
    cert.content_hash = None;

    let bundle = CertificateBundle::new("test_model").with_certificate(cert);

    // Lenient mode: cert has hmac_signature so it's NOT skipped as pre-v4.
    // verify_signature → verify_content_hash → MissingContentHash.
    let result = verify_bundle_signatures(&bundle, key);
    assert!(
        result.is_err(),
        "partial strip of content_hash must be detected"
    );
    assert!(matches!(
        result.unwrap_err().error,
        IntegrityError::MissingContentHash,
    ));
}

#[test]
fn test_mixed_bundle_lenient_skips_unsigned() {
    // Bundle with one signed and one unsigned cert.
    // Lenient mode: unsigned skipped, signed verified.
    let mut signed = make_cert();
    signed.kernel_name = "signed_kernel".to_string();
    let key = b"mixed-bundle-key";
    sign_certificate(&mut signed, key).unwrap();

    let mut unsigned = make_cert();
    unsigned.kernel_name = "unsigned_kernel".to_string();

    let bundle = CertificateBundle::new("test_model")
        .with_certificate(signed)
        .with_certificate(unsigned);

    // Lenient: passes (unsigned cert skipped).
    verify_bundle_signatures(&bundle, key).unwrap();
}

#[test]
fn test_mixed_bundle_strict_rejects_unsigned() {
    // Same mixed bundle, but strict mode rejects the unsigned cert.
    let mut signed = make_cert();
    signed.kernel_name = "signed_kernel".to_string();
    let key = b"mixed-bundle-key";
    sign_certificate(&mut signed, key).unwrap();

    let mut unsigned = make_cert();
    unsigned.kernel_name = "unsigned_kernel".to_string();

    let bundle = CertificateBundle::new("test_model")
        .with_certificate(signed)
        .with_certificate(unsigned);

    // Strict: rejects at the unsigned cert (index 1).
    let result = verify_bundle_signatures_strict(&bundle, key);
    assert!(
        result.is_err(),
        "strict mode rejects unsigned cert in mixed bundle"
    );
    let err = result.unwrap_err();
    assert_eq!(err.certificate_index, 1);
    assert_eq!(err.kernel_name, "unsigned_kernel");
    assert!(matches!(err.error, IntegrityError::MissingContentHash));
}

#[test]
fn test_empty_bundle_passes_both_modes() {
    let bundle = CertificateBundle::new("empty_model");
    let key = b"any-key";

    verify_bundle_signatures(&bundle, key).unwrap();
    verify_bundle_signatures_strict(&bundle, key).unwrap();
}

#[test]
fn test_strict_bundle_detects_tampered_cert() {
    // Strict mode also catches tampering, not just missing signatures.
    let mut cert1 = make_cert();
    cert1.kernel_name = "clean".to_string();
    let mut cert2 = make_cert();
    cert2.kernel_name = "tampered".to_string();
    let key = b"strict-tamper-key";
    sign_certificate(&mut cert1, key).unwrap();
    sign_certificate(&mut cert2, key).unwrap();

    // Tamper with cert2 after signing.
    cert2.output_width = 0.001;

    let bundle = CertificateBundle::new("test_model")
        .with_certificate(cert1)
        .with_certificate(cert2);

    let result = verify_bundle_signatures_strict(&bundle, key);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.certificate_index, 1);
    assert_eq!(err.kernel_name, "tampered");
    assert!(matches!(
        err.error,
        IntegrityError::ContentHashMismatch { .. }
    ));
}

#[test]
fn test_bundle_wrong_key_rejected() {
    // Bundle-level wrong key verification (complements unit-level test).
    let mut cert = make_cert();
    let sign_key = b"correct-key";
    let verify_key = b"attacker-key";
    sign_certificate(&mut cert, sign_key).unwrap();

    let bundle = CertificateBundle::new("test_model").with_certificate(cert);

    let result = verify_bundle_signatures(&bundle, verify_key);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err().error,
        IntegrityError::SignatureInvalid,
    ));

    // Same for strict mode.
    let mut cert2 = make_cert();
    sign_certificate(&mut cert2, sign_key).unwrap();
    let bundle2 = CertificateBundle::new("test_model").with_certificate(cert2);
    let result2 = verify_bundle_signatures_strict(&bundle2, verify_key);
    assert!(result2.is_err());
    assert!(matches!(
        result2.unwrap_err().error,
        IntegrityError::SignatureInvalid,
    ));
}

#[test]
fn test_resign_same_key_idempotent() {
    // Signing an already-signed certificate with the same key must produce
    // identical content_hash and hmac_signature (deterministic + canonical_json
    // excludes integrity fields).
    let mut cert = make_cert();
    let key = b"idempotent-key";

    sign_certificate(&mut cert, key).unwrap();
    let hash1 = cert.content_hash.clone().unwrap();
    let sig1 = cert.hmac_signature.clone().unwrap();

    // Re-sign with the same key.
    sign_certificate(&mut cert, key).unwrap();
    let hash2 = cert.content_hash.clone().unwrap();
    let sig2 = cert.hmac_signature.clone().unwrap();

    assert_eq!(
        hash1, hash2,
        "re-signing must produce identical content_hash"
    );
    assert_eq!(
        sig1, sig2,
        "re-signing with same key must produce identical signature"
    );

    // Verification must still pass.
    verify_signature(&cert, key).unwrap();
}

#[test]
fn test_resign_different_key_changes_signature() {
    // Re-signing with a different key must produce the same content_hash
    // (content unchanged) but a different hmac_signature.
    let mut cert = make_cert();
    let key_a = b"first-signing-key";
    let key_b = b"second-signing-key";

    sign_certificate(&mut cert, key_a).unwrap();
    let hash_a = cert.content_hash.clone().unwrap();
    let sig_a = cert.hmac_signature.clone().unwrap();

    // Re-sign with different key.
    sign_certificate(&mut cert, key_b).unwrap();
    let hash_b = cert.content_hash.clone().unwrap();
    let sig_b = cert.hmac_signature.clone().unwrap();

    assert_eq!(
        hash_a, hash_b,
        "content_hash must be identical (content unchanged)"
    );
    assert_ne!(
        sig_a, sig_b,
        "different keys must produce different signatures"
    );

    // Verification with the new key must pass.
    verify_signature(&cert, key_b).unwrap();

    // Verification with the old key must fail.
    let result = verify_signature(&cert, key_a);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        IntegrityError::SignatureInvalid,
    ));
}

#[test]
fn test_canonical_json_keys_are_sorted() {
    // Defense-in-depth: verify that canonical JSON keys are alphabetically
    // sorted at all nesting levels. If serde_json ever uses IndexMap
    // (preserve_order feature) instead of BTreeMap, key order changes and
    // ALL existing certificate content_hash values become invalid.
    //
    // This test catches the breakage at CI time. See #3297.
    let cert = make_cert();
    let json_str = canonical_json(&cert).unwrap();
    let value: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    fn assert_keys_sorted(val: &serde_json::Value, path: &str) {
        if let Some(obj) = val.as_object() {
            let keys: Vec<&String> = obj.keys().collect();
            for w in keys.windows(2) {
                assert!(
                    w[0] <= w[1],
                    "canonical JSON keys not sorted at {path}: \"{k1}\" > \"{k2}\".\n\
                     This likely means serde_json `preserve_order` feature was enabled.\n\
                     All existing certificate content_hash values will be invalidated!",
                    k1 = w[0],
                    k2 = w[1],
                );
            }
            for (k, v) in obj {
                assert_keys_sorted(v, &format!("{path}.{k}"));
            }
        } else if let Some(arr) = val.as_array() {
            for (i, v) in arr.iter().enumerate() {
                assert_keys_sorted(v, &format!("{path}[{i}]"));
            }
        }
    }

    assert_keys_sorted(&value, "root");
}

#[test]
fn test_canonical_json_excludes_integrity_fields() {
    // Verify that canonical_json removes both content_hash and hmac_signature.
    // This is critical: if either leaks into the canonical form, the hash
    // becomes self-referential (hash of string containing the hash).
    let mut cert = make_cert();
    let key = b"test-key";
    sign_certificate(&mut cert, key).unwrap();

    // After signing, cert has content_hash and hmac_signature set.
    assert!(cert.content_hash.is_some());
    assert!(cert.hmac_signature.is_some());

    let json_str = canonical_json(&cert).unwrap();
    let value: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    let obj = value.as_object().unwrap();

    assert!(
        !obj.contains_key("content_hash"),
        "canonical JSON must exclude content_hash"
    );
    assert!(
        !obj.contains_key("hmac_signature"),
        "canonical JSON must exclude hmac_signature"
    );
}

#[test]
fn test_content_hash_stable_across_signing_states() {
    // A certificate with content_hash/hmac_signature set to Some vs None
    // must produce the same canonical hash (both are excluded).
    // This tests the real-world scenario: signing an unsigned cert, then
    // re-signing should produce the same content_hash.
    let cert = make_cert();
    let h_unsigned = compute_content_hash(&cert).unwrap();

    let mut signed = cert;
    let key = b"any-key";
    sign_certificate(&mut signed, key).unwrap();
    let h_signed = compute_content_hash(&signed).unwrap();

    assert_eq!(
        h_unsigned, h_signed,
        "content hash must be identical regardless of integrity field state"
    );
}

// ---------------------------------------------------------------------------
// EditCertificate signing tests
// ---------------------------------------------------------------------------

use crate::edit_certificate::{EditCertificate, EditType, EditedWeight};
use crate::fusion_certificate::FusionEquivalenceCertificate;
use crate::fusion_spec::FusionVerification;
use crate::soundness_compat::VerificationSoundnessMode;
use crate::verify_types::PropMethod;

fn make_edit_cert() -> EditCertificate {
    EditCertificate::new("a".repeat(64), "b".repeat(64), PropMethod::Ibp).with_edited_weight(
        EditedWeight {
            layer_name: "transformer.h.4.mlp.c_proj".to_string(),
            edit_type: EditType::Rank1Update,
            delta_norm: 0.042,
            delta_rank: Some(1),
        },
    )
}

fn make_fusion_cert() -> FusionEquivalenceCertificate {
    let verification = FusionVerification {
        fused_kernel_name: "adain_snake_fused".to_string(),
        diff_lower: -0.001,
        diff_upper: 0.001,
        max_abs_diff: 0.001,
        within_epsilon: true,
        epsilon: 0.01,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    FusionEquivalenceCertificate::from_verification(
        &verification,
        "adain",
        "snake",
        512,
        &[(-1.0, 1.0)],
    )
}

#[test]
fn test_edit_cert_sign_verify_roundtrip() {
    let mut cert = make_edit_cert();
    let key = b"edit-cert-signing-key";

    sign_certificate(&mut cert, key).unwrap();
    assert!(cert.content_hash.is_some());
    assert!(cert.hmac_signature.is_some());

    verify_signature(&cert, key).unwrap();
    verify_content_hash(&cert).unwrap();
}

#[test]
fn test_edit_cert_detects_tampered_weight() {
    let mut cert = make_edit_cert();
    let key = b"edit-key";

    sign_certificate(&mut cert, key).unwrap();

    // Tamper: change delta_norm.
    cert.edited_weights[0].delta_norm = 999.0;

    let result = verify_content_hash(&cert);
    assert!(matches!(
        result.unwrap_err(),
        IntegrityError::ContentHashMismatch { .. }
    ));
}

#[test]
fn test_edit_cert_wrong_key_rejected() {
    let mut cert = make_edit_cert();
    let key = b"correct-key";

    sign_certificate(&mut cert, key).unwrap();

    let result = verify_signature(&cert, b"wrong-key");
    assert!(matches!(
        result.unwrap_err(),
        IntegrityError::SignatureInvalid
    ));
}

#[test]
fn test_edit_cert_content_hash_deterministic() {
    let cert = make_edit_cert();
    let h1 = compute_content_hash(&cert).unwrap();
    let h2 = compute_content_hash(&cert).unwrap();
    assert_eq!(h1, h2, "edit cert hash must be deterministic");
    assert_eq!(h1.len(), 64, "SHA-256 hex is 64 chars");
}

// ---------------------------------------------------------------------------
// FusionEquivalenceCertificate signing tests
// ---------------------------------------------------------------------------

#[test]
fn test_fusion_cert_sign_verify_roundtrip() {
    let mut cert = make_fusion_cert();
    let key = b"fusion-cert-signing-key";

    sign_certificate(&mut cert, key).unwrap();
    assert!(cert.content_hash.is_some());
    assert!(cert.hmac_signature.is_some());

    verify_signature(&cert, key).unwrap();
    verify_content_hash(&cert).unwrap();
}

#[test]
fn test_fusion_cert_detects_tampered_epsilon() {
    let mut cert = make_fusion_cert();
    let key = b"fusion-key";

    sign_certificate(&mut cert, key).unwrap();

    // Tamper: change epsilon.
    cert.epsilon = 1.0;

    let result = verify_content_hash(&cert);
    assert!(matches!(
        result.unwrap_err(),
        IntegrityError::ContentHashMismatch { .. }
    ));
}

#[test]
fn test_fusion_cert_wrong_key_rejected() {
    let mut cert = make_fusion_cert();
    let key = b"correct-key";

    sign_certificate(&mut cert, key).unwrap();

    let result = verify_signature(&cert, b"wrong-key");
    assert!(matches!(
        result.unwrap_err(),
        IntegrityError::SignatureInvalid
    ));
}

#[test]
fn test_fusion_cert_content_hash_deterministic() {
    let cert = make_fusion_cert();
    let h1 = compute_content_hash(&cert).unwrap();
    let h2 = compute_content_hash(&cert).unwrap();
    assert_eq!(h1, h2, "fusion cert hash must be deterministic");
    assert_eq!(h1.len(), 64, "SHA-256 hex is 64 chars");
}

#[test]
fn test_resign_edit_cert_idempotent() {
    let mut cert = make_edit_cert();
    let key = b"idempotent-edit-key";

    sign_certificate(&mut cert, key).unwrap();
    let hash1 = cert.content_hash.clone().unwrap();
    let sig1 = cert.hmac_signature.clone().unwrap();

    sign_certificate(&mut cert, key).unwrap();
    let hash2 = cert.content_hash.clone().unwrap();
    let sig2 = cert.hmac_signature.clone().unwrap();

    assert_eq!(hash1, hash2, "re-signing edit cert must be idempotent");
    assert_eq!(sig1, sig2, "re-signing edit cert with same key must match");
}
