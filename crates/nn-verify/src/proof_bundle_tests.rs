// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for ProofBundle serialization, builder, and validation.
//!
//! Part of #3561 (Proof certificate serialization).

use super::*;

/// A valid SHA-256 hex digest for test fixtures.
const TEST_HASH: &str = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2";

fn sample_certificate() -> BoundCertificate {
    BoundCertificate {
        kernel_name: "snake".to_string(),
        input_bounds: (-1.0, 1.0),
        output_bounds: (-3.87, 3.87),
        method: VerificationMethod::Crown,
        is_tight: true,
    }
}

fn sample_bundle() -> ProofBundle {
    ProofBundleBuilder::new("test_model", TEST_HASH)
        .add_bound_certificate(sample_certificate())
        .set_kani_summary(754, 754, 0, 0)
        .set_crown_summary(51, 26, 14, 11)
        .build()
        .expect("valid bundle")
}

// ---------------------------------------------------------------------------
// Round-trip serialize / deserialize
// ---------------------------------------------------------------------------

#[test]
fn test_round_trip_json() {
    let bundle = sample_bundle();
    let json = bundle.to_json().expect("serialize");
    let restored = ProofBundle::from_json(&json).expect("deserialize");
    assert_eq!(bundle.model_hash, restored.model_hash);
    assert_eq!(bundle.model_name, restored.model_name);
    assert_eq!(bundle.bound_certificates, restored.bound_certificates);
    assert_eq!(bundle.kani_summary, restored.kani_summary);
    assert_eq!(bundle.gamma_crown_summary, restored.gamma_crown_summary);
    assert_eq!(bundle.nn_version, restored.nn_version);
}

#[test]
fn test_round_trip_preserves_all_methods() {
    let methods = [
        VerificationMethod::Ibp,
        VerificationMethod::Crown,
        VerificationMethod::AlphaCrown,
        VerificationMethod::BetaCrown,
        VerificationMethod::Analytical,
    ];
    for method in &methods {
        let cert = BoundCertificate {
            kernel_name: format!("test_{method:?}"),
            input_bounds: (-1.0, 1.0),
            output_bounds: (-2.0, 2.0),
            method: *method,
            is_tight: method != &VerificationMethod::Ibp,
        };
        let json = serde_json::to_string(&cert).expect("serialize");
        let restored: BoundCertificate = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cert, restored, "round-trip failed for {method:?}");
    }
}

#[test]
fn test_round_trip_file_save_load() {
    let bundle = sample_bundle();
    let dir = std::env::temp_dir().join("nn_proof_bundle_test");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test_bundle.proof.json");

    bundle.save(&path).expect("save");
    let loaded = ProofBundle::load(&path).expect("load");

    assert_eq!(bundle.model_hash, loaded.model_hash);
    assert_eq!(bundle.model_name, loaded.model_name);
    assert_eq!(
        bundle.bound_certificates.len(),
        loaded.bound_certificates.len()
    );
    assert_eq!(bundle.kani_summary, loaded.kani_summary);
    assert_eq!(bundle.gamma_crown_summary, loaded.gamma_crown_summary);

    // Cleanup
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn test_round_trip_without_optional_summaries() {
    let bundle = ProofBundleBuilder::new("minimal_model", TEST_HASH)
        .add_bound_certificate(sample_certificate())
        .build()
        .expect("valid bundle");

    assert!(bundle.kani_summary.is_none());
    assert!(bundle.gamma_crown_summary.is_none());

    let json = bundle.to_json().expect("serialize");
    let restored = ProofBundle::from_json(&json).expect("deserialize");
    assert!(restored.kani_summary.is_none());
    assert!(restored.gamma_crown_summary.is_none());
}

// ---------------------------------------------------------------------------
// Builder pattern
// ---------------------------------------------------------------------------

#[test]
fn test_builder_basic() {
    let bundle = ProofBundleBuilder::new("kokoro_v1", TEST_HASH)
        .add_bound_certificate(BoundCertificate {
            kernel_name: "snake".to_string(),
            input_bounds: (-10.0, 10.0),
            output_bounds: (-15.0, 15.0),
            method: VerificationMethod::Crown,
            is_tight: true,
        })
        .add_bound_certificate(BoundCertificate {
            kernel_name: "instance_norm".to_string(),
            input_bounds: (-1.0, 1.0),
            output_bounds: (-3.87, 3.87),
            method: VerificationMethod::AlphaCrown,
            is_tight: true,
        })
        .set_kani_summary(100, 95, 3, 2)
        .set_crown_summary(40, 30, 8, 2)
        .build()
        .expect("valid bundle");

    assert_eq!(bundle.model_name, "kokoro_v1");
    assert_eq!(bundle.model_hash, TEST_HASH);
    assert_eq!(bundle.certificate_count(), 2);
    assert_eq!(bundle.tight_count(), 2);
    assert!(bundle.kani_summary.is_some());
    assert!(bundle.gamma_crown_summary.is_some());
}

#[test]
fn test_builder_custom_version() {
    let bundle = ProofBundleBuilder::new("test", TEST_HASH)
        .add_bound_certificate(sample_certificate())
        .with_nn_version("0.2.0-custom")
        .build()
        .expect("valid");

    assert_eq!(bundle.nn_version, "0.2.0-custom");
}

#[test]
fn test_builder_fails_without_certificates() {
    let result = ProofBundleBuilder::new("test", TEST_HASH).build();
    assert!(result.is_err());
    match result.unwrap_err() {
        ProofBundleError::NoCertificates => {}
        other => panic!("expected NoCertificates, got {other:?}"),
    }
}

#[test]
fn test_builder_fails_with_invalid_hash() {
    let result = ProofBundleBuilder::new("test", "not_a_valid_hash")
        .add_bound_certificate(sample_certificate())
        .build();
    assert!(result.is_err());
    match result.unwrap_err() {
        ProofBundleError::InvalidModelHash { hash } => {
            assert_eq!(hash, "not_a_valid_hash");
        }
        other => panic!("expected InvalidModelHash, got {other:?}"),
    }
}

#[test]
fn test_builder_fails_with_empty_model_name() {
    let result = ProofBundleBuilder::new("", TEST_HASH)
        .add_bound_certificate(sample_certificate())
        .build();
    assert!(result.is_err());
    match result.unwrap_err() {
        ProofBundleError::EmptyModelName => {}
        other => panic!("expected EmptyModelName, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Validation edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_validate_nan_input_bounds() {
    let result = ProofBundleBuilder::new("test", TEST_HASH)
        .add_bound_certificate(BoundCertificate {
            kernel_name: "bad_kernel".to_string(),
            input_bounds: (f32::NAN, 1.0),
            output_bounds: (-1.0, 1.0),
            method: VerificationMethod::Ibp,
            is_tight: false,
        })
        .build();

    match result.unwrap_err() {
        ProofBundleError::NonFiniteBounds {
            field, kernel_name, ..
        } => {
            assert_eq!(field, "input_bounds");
            assert_eq!(kernel_name, "bad_kernel");
        }
        other => panic!("expected NonFiniteBounds, got {other:?}"),
    }
}

#[test]
fn test_validate_nan_output_bounds() {
    let result = ProofBundleBuilder::new("test", TEST_HASH)
        .add_bound_certificate(BoundCertificate {
            kernel_name: "bad_kernel".to_string(),
            input_bounds: (-1.0, 1.0),
            output_bounds: (-1.0, f32::NAN),
            method: VerificationMethod::Crown,
            is_tight: true,
        })
        .build();

    match result.unwrap_err() {
        ProofBundleError::NonFiniteBounds {
            field, kernel_name, ..
        } => {
            assert_eq!(field, "output_bounds");
            assert_eq!(kernel_name, "bad_kernel");
        }
        other => panic!("expected NonFiniteBounds, got {other:?}"),
    }
}

#[test]
fn test_validate_inf_bounds() {
    let result = ProofBundleBuilder::new("test", TEST_HASH)
        .add_bound_certificate(BoundCertificate {
            kernel_name: "inf_kernel".to_string(),
            input_bounds: (f32::NEG_INFINITY, 1.0),
            output_bounds: (-1.0, 1.0),
            method: VerificationMethod::Ibp,
            is_tight: false,
        })
        .build();

    match result.unwrap_err() {
        ProofBundleError::NonFiniteBounds { field, .. } => {
            assert_eq!(field, "input_bounds");
        }
        other => panic!("expected NonFiniteBounds, got {other:?}"),
    }
}

#[test]
fn test_validate_inverted_input_bounds() {
    let result = ProofBundleBuilder::new("test", TEST_HASH)
        .add_bound_certificate(BoundCertificate {
            kernel_name: "inverted".to_string(),
            input_bounds: (5.0, -5.0), // lower > upper
            output_bounds: (-1.0, 1.0),
            method: VerificationMethod::Crown,
            is_tight: true,
        })
        .build();

    match result.unwrap_err() {
        ProofBundleError::InvertedBounds {
            field, kernel_name, ..
        } => {
            assert_eq!(field, "input_bounds");
            assert_eq!(kernel_name, "inverted");
        }
        other => panic!("expected InvertedBounds, got {other:?}"),
    }
}

#[test]
fn test_validate_inverted_output_bounds() {
    let result = ProofBundleBuilder::new("test", TEST_HASH)
        .add_bound_certificate(BoundCertificate {
            kernel_name: "inverted_out".to_string(),
            input_bounds: (-1.0, 1.0),
            output_bounds: (3.0, -3.0), // lower > upper
            method: VerificationMethod::Crown,
            is_tight: true,
        })
        .build();

    match result.unwrap_err() {
        ProofBundleError::InvertedBounds { field, .. } => {
            assert_eq!(field, "output_bounds");
        }
        other => panic!("expected InvertedBounds, got {other:?}"),
    }
}

#[test]
fn test_validate_empty_kernel_name() {
    let result = ProofBundleBuilder::new("test", TEST_HASH)
        .add_bound_certificate(BoundCertificate {
            kernel_name: String::new(),
            input_bounds: (-1.0, 1.0),
            output_bounds: (-2.0, 2.0),
            method: VerificationMethod::Ibp,
            is_tight: false,
        })
        .build();

    match result.unwrap_err() {
        ProofBundleError::EmptyKernelName {
            certificate_index, ..
        } => {
            assert_eq!(certificate_index, 0);
        }
        other => panic!("expected EmptyKernelName, got {other:?}"),
    }
}

#[test]
fn test_validate_inconsistent_kani_summary() {
    let result = ProofBundleBuilder::new("test", TEST_HASH)
        .add_bound_certificate(sample_certificate())
        .set_kani_summary(10, 8, 5, 3) // 8+5+3=16 > 10
        .build();

    match result.unwrap_err() {
        ProofBundleError::InconsistentKaniSummary { total, sum } => {
            assert_eq!(total, 10);
            assert_eq!(sum, 16);
        }
        other => panic!("expected InconsistentKaniSummary, got {other:?}"),
    }
}

#[test]
fn test_validate_inconsistent_crown_summary() {
    let result = ProofBundleBuilder::new("test", TEST_HASH)
        .add_bound_certificate(sample_certificate())
        .set_crown_summary(5, 3, 2, 2) // 3+2+2=7 > 5
        .build();

    match result.unwrap_err() {
        ProofBundleError::InconsistentCrownSummary { total, sum } => {
            assert_eq!(total, 5);
            assert_eq!(sum, 7);
        }
        other => panic!("expected InconsistentCrownSummary, got {other:?}"),
    }
}

#[test]
fn test_validate_equal_bounds_ok() {
    // Equal lower==upper (point bounds) should be valid
    let bundle = ProofBundleBuilder::new("test", TEST_HASH)
        .add_bound_certificate(BoundCertificate {
            kernel_name: "point_bounds".to_string(),
            input_bounds: (0.0, 0.0),
            output_bounds: (1.0, 1.0),
            method: VerificationMethod::Analytical,
            is_tight: true,
        })
        .build();
    assert!(bundle.is_ok());
}

#[test]
fn test_validate_kani_exact_sum_ok() {
    // passed + failed + timeout == total should be valid
    let bundle = ProofBundleBuilder::new("test", TEST_HASH)
        .add_bound_certificate(sample_certificate())
        .set_kani_summary(10, 7, 2, 1) // 7+2+1=10 == 10
        .build();
    assert!(bundle.is_ok());
}

#[test]
fn test_validate_kani_under_sum_ok() {
    // passed + failed + timeout < total is OK (some harnesses may be pending)
    let bundle = ProofBundleBuilder::new("test", TEST_HASH)
        .add_bound_certificate(sample_certificate())
        .set_kani_summary(10, 5, 2, 1) // 5+2+1=8 < 10
        .build();
    assert!(bundle.is_ok());
}

#[test]
fn test_validate_hash_too_short() {
    let result = ProofBundleBuilder::new("test", "abcdef")
        .add_bound_certificate(sample_certificate())
        .build();
    match result.unwrap_err() {
        ProofBundleError::InvalidModelHash { hash } => {
            assert_eq!(hash, "abcdef");
        }
        other => panic!("expected InvalidModelHash, got {other:?}"),
    }
}

#[test]
fn test_validate_hash_non_hex() {
    // 64 chars but not hex
    let bad_hash = "g1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2";
    let result = ProofBundleBuilder::new("test", bad_hash)
        .add_bound_certificate(sample_certificate())
        .build();
    match result.unwrap_err() {
        ProofBundleError::InvalidModelHash { .. } => {}
        other => panic!("expected InvalidModelHash, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Helper methods
// ---------------------------------------------------------------------------

#[test]
fn test_certificate_count() {
    let bundle = ProofBundleBuilder::new("test", TEST_HASH)
        .add_bound_certificate(sample_certificate())
        .add_bound_certificate(BoundCertificate {
            kernel_name: "relu".to_string(),
            input_bounds: (-1.0, 1.0),
            output_bounds: (0.0, 1.0),
            method: VerificationMethod::Ibp,
            is_tight: false,
        })
        .build()
        .expect("valid");

    assert_eq!(bundle.certificate_count(), 2);
}

#[test]
fn test_tight_count() {
    let bundle = ProofBundleBuilder::new("test", TEST_HASH)
        .add_bound_certificate(BoundCertificate {
            kernel_name: "crown_kernel".to_string(),
            input_bounds: (-1.0, 1.0),
            output_bounds: (-2.0, 2.0),
            method: VerificationMethod::Crown,
            is_tight: true,
        })
        .add_bound_certificate(BoundCertificate {
            kernel_name: "ibp_kernel".to_string(),
            input_bounds: (-1.0, 1.0),
            output_bounds: (-5.0, 5.0),
            method: VerificationMethod::Ibp,
            is_tight: false,
        })
        .add_bound_certificate(BoundCertificate {
            kernel_name: "alpha_kernel".to_string(),
            input_bounds: (-1.0, 1.0),
            output_bounds: (-1.5, 1.5),
            method: VerificationMethod::AlphaCrown,
            is_tight: true,
        })
        .build()
        .expect("valid");

    assert_eq!(bundle.tight_count(), 2);
    assert_eq!(bundle.certificate_count(), 3);
}

// ---------------------------------------------------------------------------
// JSON structure verification
// ---------------------------------------------------------------------------

#[test]
fn test_json_contains_expected_fields() {
    let bundle = sample_bundle();
    let json = bundle.to_json().expect("serialize");

    // Verify expected top-level fields are present
    assert!(json.contains("\"model_hash\""));
    assert!(json.contains("\"model_name\""));
    assert!(json.contains("\"bound_certificates\""));
    assert!(json.contains("\"kani_summary\""));
    assert!(json.contains("\"gamma_crown_summary\""));
    assert!(json.contains("\"created_at\""));
    assert!(json.contains("\"nn_version\""));

    // Verify nested fields
    assert!(json.contains("\"kernel_name\""));
    assert!(json.contains("\"input_bounds\""));
    assert!(json.contains("\"output_bounds\""));
    assert!(json.contains("\"method\""));
    assert!(json.contains("\"is_tight\""));
    assert!(json.contains("\"total_harnesses\""));
    assert!(json.contains("\"total_entries\""));
    assert!(json.contains("\"sound\""));
    assert!(json.contains("\"heuristic\""));
    assert!(json.contains("\"vacuous\""));
}

#[test]
fn test_deserialized_bundle_validates() {
    let bundle = sample_bundle();
    let json = bundle.to_json().expect("serialize");
    let restored = ProofBundle::from_json(&json).expect("deserialize");
    assert!(restored.validate().is_ok());
}
