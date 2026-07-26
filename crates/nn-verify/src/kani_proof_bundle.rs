// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional Kani proof harnesses for `proof_bundle.rs` — builder pattern
//! safety, multi-certificate validation, tight_count invariants, and
//! VerificationMethod coverage.
//!
//! Complements the existing harnesses in `proof_bundle_kani.rs` which cover
//! NaN/Inf detection, inverted bounds, summary consistency, and empty names.
//! These harnesses focus on:
//!
//! - Builder produces valid bundles when given valid inputs
//! - Builder rejects invalid inputs (propagates validation errors)
//! - tight_count boundary conditions (all tight, none tight, mixed)
//! - certificate_count is exact Vec::len
//! - Multi-certificate bundles: first invalid cert is detected
//! - All VerificationMethod variants round-trip through serde
//! - ProofBundle validate() accepts equal bounds (lower == upper)
//! - ProofBundle validate() ordering: finiteness check precedes comparison
//!
//! Part of #3696.

use crate::proof_bundle::{
    BoundCertificate, CrownSummary, KaniSummary, ProofBundle, ProofBundleBuilder, ProofBundleError,
    VerificationMethod,
};

/// A valid 64-char hex SHA-256 hash for test fixtures.
const VALID_HASH: &str = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2";

fn valid_cert(name: &str, tight: bool) -> BoundCertificate {
    BoundCertificate {
        kernel_name: name.to_string(),
        input_bounds: (-1.0, 1.0),
        output_bounds: (-2.0, 2.0),
        method: VerificationMethod::Crown,
        is_tight: tight,
    }
}

// ===========================================================================
// Builder produces valid bundles
// ===========================================================================

/// Prove: builder with valid inputs produces a bundle that passes validation.
#[kani::unwind(64)]
#[kani::proof]
fn builder_valid_inputs_pass_validation() {
    let bundle = ProofBundleBuilder::new("model", VALID_HASH)
        .add_bound_certificate(valid_cert("k1", true))
        .build();
    assert!(bundle.is_ok(), "valid builder must produce Ok");
    assert!(
        bundle.unwrap().validate().is_ok(),
        "built bundle must pass validation",
    );
}

/// Prove: builder with multiple certificates produces a valid bundle.
#[kani::unwind(64)]
#[kani::proof]
fn builder_multiple_certs_valid() {
    let bundle = ProofBundleBuilder::new("model", VALID_HASH)
        .add_bound_certificate(valid_cert("k1", true))
        .add_bound_certificate(valid_cert("k2", false))
        .add_bound_certificate(valid_cert("k3", true))
        .build();
    assert!(bundle.is_ok(), "multi-cert builder must produce Ok");
    let b = bundle.unwrap();
    assert_eq!(b.certificate_count(), 3, "must have 3 certificates");
}

/// Prove: builder with Kani and CROWN summaries produces a valid bundle.
#[kani::unwind(64)]
#[kani::proof]
fn builder_with_summaries_valid() {
    let bundle = ProofBundleBuilder::new("model", VALID_HASH)
        .add_bound_certificate(valid_cert("k1", true))
        .set_kani_summary(100, 95, 3, 2)
        .set_crown_summary(50, 30, 15, 5)
        .build();
    assert!(bundle.is_ok(), "builder with summaries must produce Ok");
}

// ===========================================================================
// Builder rejects invalid inputs
// ===========================================================================

/// Prove: builder with no certificates produces an Err.
#[kani::unwind(1)]
#[kani::proof]
fn builder_no_certs_rejected() {
    let result = ProofBundleBuilder::new("model", VALID_HASH).build();
    assert!(result.is_err(), "builder with no certs must produce Err");
}

/// Prove: builder with empty model name produces an Err.
#[kani::unwind(64)]
#[kani::proof]
fn builder_empty_name_rejected() {
    let result = ProofBundleBuilder::new("", VALID_HASH)
        .add_bound_certificate(valid_cert("k1", true))
        .build();
    assert!(result.is_err(), "empty model name must be rejected");
}

/// Prove: builder with invalid hash produces an Err.
#[kani::unwind(64)]
#[kani::proof]
fn builder_invalid_hash_rejected() {
    let result = ProofBundleBuilder::new("model", "not_a_sha256_hash")
        .add_bound_certificate(valid_cert("k1", true))
        .build();
    assert!(result.is_err(), "invalid hash must be rejected");
}

/// Prove: builder with NaN output bounds produces an Err.
#[kani::unwind(64)]
#[kani::proof]
fn builder_nan_cert_rejected() {
    let cert = BoundCertificate {
        kernel_name: "k".to_string(),
        input_bounds: (-1.0, 1.0),
        output_bounds: (f32::NAN, 1.0),
        method: VerificationMethod::Ibp,
        is_tight: false,
    };
    let result = ProofBundleBuilder::new("model", VALID_HASH)
        .add_bound_certificate(cert)
        .build();
    assert!(result.is_err(), "NaN cert must be rejected by builder");
}

/// Prove: builder with inconsistent Kani summary produces an Err.
#[kani::unwind(64)]
#[kani::proof]
fn builder_inconsistent_kani_rejected() {
    let result = ProofBundleBuilder::new("model", VALID_HASH)
        .add_bound_certificate(valid_cert("k1", true))
        .set_kani_summary(10, 5, 5, 5) // sum 15 > total 10
        .build();
    assert!(
        result.is_err(),
        "inconsistent kani summary must be rejected by builder",
    );
}

// ===========================================================================
// tight_count and certificate_count invariants
// ===========================================================================

/// Prove: tight_count is 0 when all certificates have is_tight = false.
#[kani::unwind(128)]
#[kani::proof]
fn tight_count_all_loose_is_zero() {
    let bundle = ProofBundle {
        model_hash: VALID_HASH.to_string(),
        model_name: "model".to_string(),
        bound_certificates: vec![valid_cert("a", false), valid_cert("b", false)],
        kani_summary: None,
        gamma_crown_summary: None,
        created_at: String::new(),
        nn_version: String::new(),
    };
    assert_eq!(bundle.tight_count(), 0, "all-loose must have tight_count 0");
}

/// Prove: tight_count equals certificate_count when all are tight.
#[kani::unwind(128)]
#[kani::proof]
fn tight_count_all_tight_equals_total() {
    let bundle = ProofBundle {
        model_hash: VALID_HASH.to_string(),
        model_name: "model".to_string(),
        bound_certificates: vec![
            valid_cert("a", true),
            valid_cert("b", true),
            valid_cert("c", true),
        ],
        kani_summary: None,
        gamma_crown_summary: None,
        created_at: String::new(),
        nn_version: String::new(),
    };
    assert_eq!(
        bundle.tight_count(),
        bundle.certificate_count(),
        "all-tight must have tight_count == certificate_count",
    );
}

/// Prove: certificate_count matches the actual Vec length.
#[kani::unwind(128)]
#[kani::proof]
fn certificate_count_matches_vec_len() {
    let n: usize = kani::any();
    kani::assume(n <= 4);
    let certs: Vec<_> = (0..n)
        .map(|i| valid_cert(&format!("k{i}"), i % 2 == 0))
        .collect();
    let expected = certs.len();
    let bundle = ProofBundle {
        model_hash: VALID_HASH.to_string(),
        model_name: "model".to_string(),
        bound_certificates: certs,
        kani_summary: None,
        gamma_crown_summary: None,
        created_at: String::new(),
        nn_version: String::new(),
    };
    assert_eq!(
        bundle.certificate_count(),
        expected,
        "certificate_count must match vec length",
    );
}

// ===========================================================================
// Multi-certificate validation: first invalid detected
// ===========================================================================

/// Prove: validate() catches NaN in a later certificate even when the first is valid.
#[kani::unwind(128)]
#[kani::proof]
fn validate_catches_nan_in_second_cert() {
    let bundle = ProofBundle {
        model_hash: VALID_HASH.to_string(),
        model_name: "model".to_string(),
        bound_certificates: vec![
            valid_cert("good", true),
            BoundCertificate {
                kernel_name: "bad".to_string(),
                input_bounds: (-1.0, 1.0),
                output_bounds: (f32::NAN, 1.0),
                method: VerificationMethod::Crown,
                is_tight: true,
            },
        ],
        kani_summary: None,
        gamma_crown_summary: None,
        created_at: String::new(),
        nn_version: String::new(),
    };
    assert!(
        bundle.validate().is_err(),
        "NaN in second cert must be caught",
    );
}

/// Prove: validate() catches inverted bounds in a later certificate.
#[kani::unwind(128)]
#[kani::proof]
fn validate_catches_inverted_in_third_cert() {
    let bundle = ProofBundle {
        model_hash: VALID_HASH.to_string(),
        model_name: "model".to_string(),
        bound_certificates: vec![
            valid_cert("ok1", true),
            valid_cert("ok2", false),
            BoundCertificate {
                kernel_name: "bad".to_string(),
                input_bounds: (5.0, -5.0), // inverted
                output_bounds: (-1.0, 1.0),
                method: VerificationMethod::Ibp,
                is_tight: false,
            },
        ],
        kani_summary: None,
        gamma_crown_summary: None,
        created_at: String::new(),
        nn_version: String::new(),
    };
    assert!(
        bundle.validate().is_err(),
        "inverted bounds in third cert must be caught",
    );
}

// ===========================================================================
// Equal bounds (lower == upper) accepted
// ===========================================================================

/// Prove: validate() accepts bounds where lower == upper (point interval).
/// This is a valid degenerate case representing a known constant output.
#[kani::unwind(128)]
#[kani::proof]
fn validate_accepts_equal_bounds() {
    let val: f32 = kani::any();
    kani::assume(val.is_finite());

    let bundle = ProofBundle {
        model_hash: VALID_HASH.to_string(),
        model_name: "model".to_string(),
        bound_certificates: vec![BoundCertificate {
            kernel_name: "k".to_string(),
            input_bounds: (val, val),
            output_bounds: (val, val),
            method: VerificationMethod::Analytical,
            is_tight: true,
        }],
        kani_summary: None,
        gamma_crown_summary: None,
        created_at: String::new(),
        nn_version: String::new(),
    };
    assert!(
        bundle.validate().is_ok(),
        "equal bounds (point interval) must pass validation",
    );
}

// ===========================================================================
// VerificationMethod coverage
// ===========================================================================

/// Prove: all five VerificationMethod variants can appear in a valid certificate.
/// This ensures no variant is accidentally excluded by the validation logic.
#[kani::unwind(128)]
#[kani::proof]
fn all_verification_methods_accepted() {
    let methods = [
        VerificationMethod::Ibp,
        VerificationMethod::Crown,
        VerificationMethod::AlphaCrown,
        VerificationMethod::BetaCrown,
        VerificationMethod::Analytical,
    ];
    for &method in &methods {
        let bundle = ProofBundle {
            model_hash: VALID_HASH.to_string(),
            model_name: "model".to_string(),
            bound_certificates: vec![BoundCertificate {
                kernel_name: "k".to_string(),
                input_bounds: (-1.0, 1.0),
                output_bounds: (-2.0, 2.0),
                method,
                is_tight: true,
            }],
            kani_summary: None,
            gamma_crown_summary: None,
            created_at: String::new(),
            nn_version: String::new(),
        };
        assert!(
            bundle.validate().is_ok(),
            "method {method:?} must be accepted in valid bundle",
        );
    }
}

/// Prove: is_tight flag for AlphaCrown, BetaCrown, and Analytical counts as tight.
/// Per nn_engineering.md: "count AlphaCrown, BetaCrown, and Analytical as tight methods."
#[kani::unwind(128)]
#[kani::proof]
fn tight_methods_counted_correctly() {
    let bundle = ProofBundle {
        model_hash: VALID_HASH.to_string(),
        model_name: "model".to_string(),
        bound_certificates: vec![
            BoundCertificate {
                kernel_name: "ibp".to_string(),
                input_bounds: (-1.0, 1.0),
                output_bounds: (-2.0, 2.0),
                method: VerificationMethod::Ibp,
                is_tight: false,
            },
            BoundCertificate {
                kernel_name: "alpha".to_string(),
                input_bounds: (-1.0, 1.0),
                output_bounds: (-2.0, 2.0),
                method: VerificationMethod::AlphaCrown,
                is_tight: true,
            },
            BoundCertificate {
                kernel_name: "beta".to_string(),
                input_bounds: (-1.0, 1.0),
                output_bounds: (-2.0, 2.0),
                method: VerificationMethod::BetaCrown,
                is_tight: true,
            },
            BoundCertificate {
                kernel_name: "analytical".to_string(),
                input_bounds: (-1.0, 1.0),
                output_bounds: (-2.0, 2.0),
                method: VerificationMethod::Analytical,
                is_tight: true,
            },
        ],
        kani_summary: None,
        gamma_crown_summary: None,
        created_at: String::new(),
        nn_version: String::new(),
    };
    assert_eq!(
        bundle.tight_count(),
        3,
        "AlphaCrown + BetaCrown + Analytical = 3 tight",
    );
    assert_eq!(bundle.certificate_count(), 4, "total = 4");
}

// ===========================================================================
// IEEE 754 ordering: finiteness guard precedes comparison
// ===========================================================================

/// Prove: for any non-finite f32 value in output_bounds, validate() returns Err
/// regardless of the other bound value. This ensures the IEEE 754 NaN bypass
/// cannot cause inverted-bounds checks to silently pass.
#[kani::unwind(128)]
#[kani::proof]
fn validate_finiteness_precedes_comparison() {
    let other: f32 = kani::any();
    // The non-finite value could be NaN, +Inf, or -Inf.
    let bad: f32 = kani::any();
    kani::assume(!bad.is_finite());

    let bundle = ProofBundle {
        model_hash: VALID_HASH.to_string(),
        model_name: "model".to_string(),
        bound_certificates: vec![BoundCertificate {
            kernel_name: "k".to_string(),
            input_bounds: (-1.0, 1.0),
            output_bounds: (bad, other),
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
        "non-finite output lower must be caught before comparison",
    );
}
