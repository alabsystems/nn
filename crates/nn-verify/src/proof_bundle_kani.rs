// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani verification harnesses for `ProofBundle` and `BoundCertificate`
//! validation safety.
//!
//! Proves that the validation logic in `proof_bundle.rs` correctly detects:
//! - Non-finite (NaN, Inf) bounds per IEEE 754 rules
//! - Inverted bounds (lower > upper)
//! - Inconsistent Kani and CROWN summary counts
//! - Empty model names and kernel names
//! - Invalid SHA-256 hash format
//!
//! Part of #3614 (Kani harnesses for nn-verify certify + proof_bundle safety).

use super::*;

// ---------------------------------------------------------------------------
// SHA-256 hash validation (validate_sha256_hex)
// ---------------------------------------------------------------------------

/// Prove: `validate_sha256_hex` accepts all strings of exactly 64 hex chars.
///
/// Bounded to 64 chars. Kani verifies the function's length and char-class
/// checks are mutually consistent: a 64-char all-hex string always passes.
#[kani::unwind(8)]
#[kani::proof]
fn validate_sha256_hex_accepts_valid_hex() {
    // Construct a 64-char hex string from arbitrary nibbles.
    let hex_chars: [u8; 16] = *b"0123456789abcdef";
    let mut buf = [0u8; 64];
    for i in 0..64 {
        let idx: usize = kani::any();
        kani::assume(idx < 16);
        buf[i] = hex_chars[idx];
    }
    let s = core::str::from_utf8(&buf).unwrap();
    assert!(
        crate::certificate_types::validate_sha256_hex(s).is_ok(),
        "64-char hex string must be accepted"
    );
}

/// Prove: `validate_sha256_hex` rejects any string that is not exactly 64 chars.
///
/// Tests a string of length != 64 (bounded 0..=128 for tractability).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(130)]
fn validate_sha256_hex_rejects_wrong_length() {
    let len: usize = kani::any();
    kani::assume(len <= 128 && len != 64);
    // Content doesn't matter — wrong length alone must reject.
    let s: String = (0..len).map(|_| 'a').collect();
    assert!(
        crate::certificate_types::validate_sha256_hex(&s).is_err(),
        "non-64-length string must be rejected"
    );
}

// ---------------------------------------------------------------------------
// BoundCertificate finiteness: NaN detection
// ---------------------------------------------------------------------------

/// Prove: validate() detects NaN in input_bounds.0 for any certificate.
///
/// IEEE 754: NaN is not finite. The finiteness check MUST fire before the
/// relational comparison (lower > upper), because NaN comparisons return
/// false and would silently pass the inverted-bounds check.
#[kani::unwind(128)]
#[kani::proof]
fn validate_catches_nan_input_lower() {
    let bundle = ProofBundle {
        model_hash: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2".to_string(),
        model_name: "test".to_string(),
        bound_certificates: vec![BoundCertificate {
            kernel_name: "k".to_string(),
            input_bounds: (f32::NAN, 1.0),
            output_bounds: (-1.0, 1.0),
            method: VerificationMethod::Ibp,
            is_tight: false,
        }],
        kani_summary: None,
        gamma_crown_summary: None,
        created_at: String::new(),
        nn_version: String::new(),
    };
    assert!(
        bundle.validate().is_err(),
        "NaN input lower must be rejected"
    );
}

/// Prove: validate() detects NaN in input_bounds.1.
#[kani::unwind(128)]
#[kani::proof]
fn validate_catches_nan_input_upper() {
    let bundle = ProofBundle {
        model_hash: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2".to_string(),
        model_name: "test".to_string(),
        bound_certificates: vec![BoundCertificate {
            kernel_name: "k".to_string(),
            input_bounds: (-1.0, f32::NAN),
            output_bounds: (-1.0, 1.0),
            method: VerificationMethod::Ibp,
            is_tight: false,
        }],
        kani_summary: None,
        gamma_crown_summary: None,
        created_at: String::new(),
        nn_version: String::new(),
    };
    assert!(
        bundle.validate().is_err(),
        "NaN input upper must be rejected"
    );
}

/// Prove: validate() detects NaN in output_bounds.0.
#[kani::unwind(128)]
#[kani::proof]
fn validate_catches_nan_output_lower() {
    let bundle = ProofBundle {
        model_hash: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2".to_string(),
        model_name: "test".to_string(),
        bound_certificates: vec![BoundCertificate {
            kernel_name: "k".to_string(),
            input_bounds: (-1.0, 1.0),
            output_bounds: (f32::NAN, 1.0),
            method: VerificationMethod::Crown,
            is_tight: true,
        }],
        kani_summary: None,
        gamma_crown_summary: None,
        created_at: String::new(),
        nn_version: String::new(),
    };
    assert!(
        bundle.validate().is_err(),
        "NaN output lower must be rejected"
    );
}

/// Prove: validate() detects NaN in output_bounds.1.
#[kani::unwind(128)]
#[kani::proof]
fn validate_catches_nan_output_upper() {
    let bundle = ProofBundle {
        model_hash: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2".to_string(),
        model_name: "test".to_string(),
        bound_certificates: vec![BoundCertificate {
            kernel_name: "k".to_string(),
            input_bounds: (-1.0, 1.0),
            output_bounds: (-1.0, f32::NAN),
            method: VerificationMethod::Crown,
            is_tight: true,
        }],
        kani_summary: None,
        gamma_crown_summary: None,
        created_at: String::new(),
        nn_version: String::new(),
    };
    assert!(
        bundle.validate().is_err(),
        "NaN output upper must be rejected"
    );
}

// ---------------------------------------------------------------------------
// BoundCertificate finiteness: Inf detection
// ---------------------------------------------------------------------------

/// Prove: validate() detects +Inf in input bounds.
#[kani::unwind(128)]
#[kani::proof]
fn validate_catches_pos_inf_input() {
    let bundle = ProofBundle {
        model_hash: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2".to_string(),
        model_name: "test".to_string(),
        bound_certificates: vec![BoundCertificate {
            kernel_name: "k".to_string(),
            input_bounds: (-1.0, f32::INFINITY),
            output_bounds: (-1.0, 1.0),
            method: VerificationMethod::Ibp,
            is_tight: false,
        }],
        kani_summary: None,
        gamma_crown_summary: None,
        created_at: String::new(),
        nn_version: String::new(),
    };
    assert!(bundle.validate().is_err(), "+Inf input must be rejected");
}

/// Prove: validate() detects -Inf in output bounds.
#[kani::unwind(128)]
#[kani::proof]
fn validate_catches_neg_inf_output() {
    let bundle = ProofBundle {
        model_hash: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2".to_string(),
        model_name: "test".to_string(),
        bound_certificates: vec![BoundCertificate {
            kernel_name: "k".to_string(),
            input_bounds: (-1.0, 1.0),
            output_bounds: (f32::NEG_INFINITY, 1.0),
            method: VerificationMethod::Crown,
            is_tight: true,
        }],
        kani_summary: None,
        gamma_crown_summary: None,
        created_at: String::new(),
        nn_version: String::new(),
    };
    assert!(bundle.validate().is_err(), "-Inf output must be rejected");
}

// ---------------------------------------------------------------------------
// Inverted bounds detection
// ---------------------------------------------------------------------------

/// Prove: for any two finite f32 values where lower > upper, validate()
/// detects the inversion in input_bounds.
///
/// This is the critical IEEE 754 safety property: the finiteness check
/// MUST precede the comparison, otherwise NaN would bypass it.
#[kani::unwind(128)]
#[kani::proof]
fn validate_catches_inverted_input_bounds() {
    let lower: f32 = kani::any();
    let upper: f32 = kani::any();
    kani::assume(lower.is_finite() && upper.is_finite());
    kani::assume(lower > upper);

    let bundle = ProofBundle {
        model_hash: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2".to_string(),
        model_name: "test".to_string(),
        bound_certificates: vec![BoundCertificate {
            kernel_name: "k".to_string(),
            input_bounds: (lower, upper),
            output_bounds: (-1.0, 1.0),
            method: VerificationMethod::Crown,
            is_tight: true,
        }],
        kani_summary: None,
        gamma_crown_summary: None,
        created_at: String::new(),
        nn_version: String::new(),
    };
    assert!(
        bundle.validate().is_err(),
        "inverted input bounds must be rejected"
    );
}

/// Prove: for any two finite f32 values where lower > upper, validate()
/// detects the inversion in output_bounds.
#[kani::unwind(128)]
#[kani::proof]
fn validate_catches_inverted_output_bounds() {
    let lower: f32 = kani::any();
    let upper: f32 = kani::any();
    kani::assume(lower.is_finite() && upper.is_finite());
    kani::assume(lower > upper);

    let bundle = ProofBundle {
        model_hash: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2".to_string(),
        model_name: "test".to_string(),
        bound_certificates: vec![BoundCertificate {
            kernel_name: "k".to_string(),
            input_bounds: (-1.0, 1.0),
            output_bounds: (lower, upper),
            method: VerificationMethod::Crown,
            is_tight: true,
        }],
        kani_summary: None,
        gamma_crown_summary: None,
        created_at: String::new(),
        nn_version: String::new(),
    };
    assert!(
        bundle.validate().is_err(),
        "inverted output bounds must be rejected"
    );
}

// ---------------------------------------------------------------------------
// Valid bounds pass validation
// ---------------------------------------------------------------------------

/// Prove: for any two finite f32 values where lower <= upper, a well-formed
/// bundle with those input bounds passes validation.
#[kani::unwind(128)]
#[kani::proof]
fn validate_accepts_valid_finite_bounds() {
    let lower: f32 = kani::any();
    let upper: f32 = kani::any();
    kani::assume(lower.is_finite() && upper.is_finite());
    kani::assume(lower <= upper);

    let bundle = ProofBundle {
        model_hash: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2".to_string(),
        model_name: "test".to_string(),
        bound_certificates: vec![BoundCertificate {
            kernel_name: "k".to_string(),
            input_bounds: (lower, upper),
            output_bounds: (lower, upper),
            method: VerificationMethod::Crown,
            is_tight: true,
        }],
        kani_summary: None,
        gamma_crown_summary: None,
        created_at: String::new(),
        nn_version: String::new(),
    };
    assert!(
        bundle.validate().is_ok(),
        "valid finite non-inverted bounds must pass validation"
    );
}

// ---------------------------------------------------------------------------
// KaniSummary consistency
// ---------------------------------------------------------------------------

/// Prove: for any four usize values where passed + failed + timeout > total,
/// validate() rejects the bundle.
#[kani::unwind(128)]
#[kani::proof]
fn validate_catches_inconsistent_kani_summary() {
    let total: usize = kani::any();
    let passed: usize = kani::any();
    let failed: usize = kani::any();
    let timeout: usize = kani::any();

    // Avoid overflow in the sum.
    kani::assume(passed <= 1000 && failed <= 1000 && timeout <= 1000);
    let sum = passed + failed + timeout;
    kani::assume(sum > total);

    let bundle = ProofBundle {
        model_hash: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2".to_string(),
        model_name: "test".to_string(),
        bound_certificates: vec![BoundCertificate {
            kernel_name: "k".to_string(),
            input_bounds: (-1.0, 1.0),
            output_bounds: (-1.0, 1.0),
            method: VerificationMethod::Crown,
            is_tight: true,
        }],
        kani_summary: Some(KaniSummary {
            total_harnesses: total,
            passed,
            failed,
            timeout,
        }),
        gamma_crown_summary: None,
        created_at: String::new(),
        nn_version: String::new(),
    };
    assert!(
        bundle.validate().is_err(),
        "inconsistent kani summary must be rejected"
    );
}

/// Prove: for any four usize values where passed + failed + timeout <= total,
/// the Kani summary check passes (no spurious rejection).
#[kani::unwind(128)]
#[kani::proof]
fn validate_accepts_consistent_kani_summary() {
    let total: usize = kani::any();
    let passed: usize = kani::any();
    let failed: usize = kani::any();
    let timeout: usize = kani::any();

    kani::assume(passed <= 1000 && failed <= 1000 && timeout <= 1000 && total <= 3000);
    let sum = passed + failed + timeout;
    kani::assume(sum <= total);

    let bundle = ProofBundle {
        model_hash: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2".to_string(),
        model_name: "test".to_string(),
        bound_certificates: vec![BoundCertificate {
            kernel_name: "k".to_string(),
            input_bounds: (-1.0, 1.0),
            output_bounds: (-1.0, 1.0),
            method: VerificationMethod::Crown,
            is_tight: true,
        }],
        kani_summary: Some(KaniSummary {
            total_harnesses: total,
            passed,
            failed,
            timeout,
        }),
        gamma_crown_summary: None,
        created_at: String::new(),
        nn_version: String::new(),
    };
    assert!(
        bundle.validate().is_ok(),
        "consistent kani summary must pass validation"
    );
}

// ---------------------------------------------------------------------------
// CrownSummary consistency
// ---------------------------------------------------------------------------

/// Prove: for any four usize values where sound + heuristic + vacuous > total,
/// validate() rejects the bundle.
#[kani::unwind(128)]
#[kani::proof]
fn validate_catches_inconsistent_crown_summary() {
    let total: usize = kani::any();
    let sound: usize = kani::any();
    let heuristic: usize = kani::any();
    let vacuous: usize = kani::any();

    kani::assume(sound <= 1000 && heuristic <= 1000 && vacuous <= 1000);
    let sum = sound + heuristic + vacuous;
    kani::assume(sum > total);

    let bundle = ProofBundle {
        model_hash: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2".to_string(),
        model_name: "test".to_string(),
        bound_certificates: vec![BoundCertificate {
            kernel_name: "k".to_string(),
            input_bounds: (-1.0, 1.0),
            output_bounds: (-1.0, 1.0),
            method: VerificationMethod::Crown,
            is_tight: true,
        }],
        kani_summary: None,
        gamma_crown_summary: Some(CrownSummary {
            total_entries: total,
            sound,
            heuristic,
            vacuous,
        }),
        created_at: String::new(),
        nn_version: String::new(),
    };
    assert!(
        bundle.validate().is_err(),
        "inconsistent crown summary must be rejected"
    );
}

/// Prove: for any four usize values where sound + heuristic + vacuous <= total,
/// the CROWN summary check passes.
#[kani::unwind(128)]
#[kani::proof]
fn validate_accepts_consistent_crown_summary() {
    let total: usize = kani::any();
    let sound: usize = kani::any();
    let heuristic: usize = kani::any();
    let vacuous: usize = kani::any();

    kani::assume(sound <= 1000 && heuristic <= 1000 && vacuous <= 1000 && total <= 3000);
    let sum = sound + heuristic + vacuous;
    kani::assume(sum <= total);

    let bundle = ProofBundle {
        model_hash: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2".to_string(),
        model_name: "test".to_string(),
        bound_certificates: vec![BoundCertificate {
            kernel_name: "k".to_string(),
            input_bounds: (-1.0, 1.0),
            output_bounds: (-1.0, 1.0),
            method: VerificationMethod::Crown,
            is_tight: true,
        }],
        kani_summary: None,
        gamma_crown_summary: Some(CrownSummary {
            total_entries: total,
            sound,
            heuristic,
            vacuous,
        }),
        created_at: String::new(),
        nn_version: String::new(),
    };
    assert!(
        bundle.validate().is_ok(),
        "consistent crown summary must pass validation"
    );
}

// ---------------------------------------------------------------------------
// Empty names
// ---------------------------------------------------------------------------

/// Prove: validate() rejects bundles with an empty model name.
#[kani::unwind(128)]
#[kani::proof]
fn validate_catches_empty_model_name() {
    let bundle = ProofBundle {
        model_hash: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2".to_string(),
        model_name: String::new(),
        bound_certificates: vec![BoundCertificate {
            kernel_name: "k".to_string(),
            input_bounds: (-1.0, 1.0),
            output_bounds: (-1.0, 1.0),
            method: VerificationMethod::Ibp,
            is_tight: false,
        }],
        kani_summary: None,
        gamma_crown_summary: None,
        created_at: String::new(),
        nn_version: String::new(),
    };
    assert!(
        bundle.validate().is_err(),
        "empty model name must be rejected"
    );
}

/// Prove: validate() rejects bundles where a certificate has an empty kernel name.
#[kani::unwind(128)]
#[kani::proof]
fn validate_catches_empty_kernel_name() {
    let bundle = ProofBundle {
        model_hash: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2".to_string(),
        model_name: "model".to_string(),
        bound_certificates: vec![BoundCertificate {
            kernel_name: String::new(),
            input_bounds: (-1.0, 1.0),
            output_bounds: (-1.0, 1.0),
            method: VerificationMethod::Crown,
            is_tight: true,
        }],
        kani_summary: None,
        gamma_crown_summary: None,
        created_at: String::new(),
        nn_version: String::new(),
    };
    assert!(
        bundle.validate().is_err(),
        "empty kernel name must be rejected"
    );
}

// ---------------------------------------------------------------------------
// No certificates
// ---------------------------------------------------------------------------

/// Prove: validate() rejects bundles with zero bound certificates.
#[kani::unwind(128)]
#[kani::proof]
fn validate_catches_no_certificates() {
    let bundle = ProofBundle {
        model_hash: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2".to_string(),
        model_name: "model".to_string(),
        bound_certificates: vec![],
        kani_summary: None,
        gamma_crown_summary: None,
        created_at: String::new(),
        nn_version: String::new(),
    };
    assert!(
        bundle.validate().is_err(),
        "empty certificates must be rejected"
    );
}

// ---------------------------------------------------------------------------
// tight_count / certificate_count consistency
// ---------------------------------------------------------------------------

/// Prove: tight_count() <= certificate_count() for any bundle.
///
/// This is a structural invariant: tight certificates are a subset of all
/// certificates, so the count of tight ones cannot exceed the total.
#[kani::unwind(128)]
#[kani::proof]
fn tight_count_le_certificate_count() {
    // Build a bundle with 0..=3 certificates, arbitrary is_tight flags.
    let n: usize = kani::any();
    kani::assume(n <= 3);

    let mut certs = Vec::new();
    for _ in 0..n {
        let is_tight: bool = kani::any();
        certs.push(BoundCertificate {
            kernel_name: "k".to_string(),
            input_bounds: (-1.0, 1.0),
            output_bounds: (-1.0, 1.0),
            method: VerificationMethod::Crown,
            is_tight,
        });
    }

    let bundle = ProofBundle {
        model_hash: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2".to_string(),
        model_name: "m".to_string(),
        bound_certificates: certs,
        kani_summary: None,
        gamma_crown_summary: None,
        created_at: String::new(),
        nn_version: String::new(),
    };

    assert!(
        bundle.tight_count() <= bundle.certificate_count(),
        "tight_count must not exceed certificate_count"
    );
}
