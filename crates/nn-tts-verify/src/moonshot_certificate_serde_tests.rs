// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use crate::moonshot::*;

// --- Round-trip tests ---

#[test]
fn test_serde_round_trip_from_status() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test-model", "English text", "abc123");

    let json = cert.to_json_serde().expect("serialization should succeed");
    let deserialized =
        MoonshotCertificate::from_json(&json).expect("deserialization should succeed");

    assert_eq!(deserialized.model_name, cert.model_name);
    assert_eq!(deserialized.input_specification, cert.input_specification);
    assert_eq!(deserialized.source_hash, cert.source_hash);
    assert_eq!(deserialized.verification_date, cert.verification_date);
    assert_eq!(deserialized.verification_dim, cert.verification_dim);
    assert_eq!(deserialized.all_at_least_partial, cert.all_at_least_partial);
    assert_eq!(deserialized.all_proven, cert.all_proven);
    assert_eq!(deserialized.properties.len(), cert.properties.len());

    for (i, (orig, deser)) in cert
        .properties
        .iter()
        .zip(&deserialized.properties)
        .enumerate()
    {
        assert_eq!(
            deser.property_index, orig.property_index,
            "index mismatch at {i}"
        );
        assert_eq!(
            deser.property_name, orig.property_name,
            "name mismatch at {i}"
        );
        assert_eq!(deser.level, orig.level, "level mismatch at {i}");
        assert_eq!(
            deser.proof_artifacts, orig.proof_artifacts,
            "artifacts mismatch at {i}"
        );
        assert_eq!(
            deser.assumptions, orig.assumptions,
            "assumptions mismatch at {i}"
        );
        assert_eq!(
            deser.bound_value, orig.bound_value,
            "bound_value mismatch at {i}"
        );
        assert_eq!(deser.threshold, orig.threshold, "threshold mismatch at {i}");
    }
}

#[test]
fn test_serde_round_trip_with_kani_evidence() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test-model", "English text", "abc123");

    let kani_evidence = KaniVerificationEvidence {
        harnesses_passed: 475,
        harnesses_total: 475,
        harness_files: vec![
            "crates/nn-core/src/kani_bounds.rs".to_string(),
            "crates/nn-autodiff/src/kani_backward_proofs.rs".to_string(),
        ],
        all_passed: true,
    };

    let enriched = cert.with_kani_results(&kani_evidence);
    let json = enriched
        .to_json_serde()
        .expect("serialization should succeed");
    let deserialized =
        MoonshotCertificate::from_json(&json).expect("deserialization should succeed");

    assert_eq!(
        deserialized.properties[6].level,
        VerificationLevel::KaniProven
    );
    assert_eq!(deserialized.properties[6].bound_value, Some(475.0));
    assert_eq!(deserialized.properties[6].threshold, Some(475.0));
    assert_eq!(deserialized.properties[6].proof_artifacts.len(), 2);
}

#[test]
fn test_serde_round_trip_with_smt_evidence() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test-model", "English text", "abc123");

    let smt_evidence = SmtVerificationEvidence {
        kernels_proven: 20,
        kernels_total: 20,
        proven_kernel_names: vec!["snake".to_string(), "silu_mul".to_string()],
        all_proven: true,
    };

    let enriched = cert.with_smt_results(&smt_evidence);
    let json = enriched
        .to_json_serde()
        .expect("serialization should succeed");
    let deserialized =
        MoonshotCertificate::from_json(&json).expect("deserialization should succeed");

    assert_eq!(
        deserialized.properties[7].level,
        VerificationLevel::SmtProven
    );
    assert_eq!(deserialized.properties[7].bound_value, Some(20.0));
}

#[test]
fn test_serde_round_trip_with_verification_dim() {
    let status = MoonshotStatus::from_repo();
    let mut cert =
        MoonshotCertificate::from_status(&status, "test-model", "English text", "abc123");
    cert.verification_dim = Some(256);

    let json = cert.to_json_serde().expect("serialization should succeed");
    let deserialized =
        MoonshotCertificate::from_json(&json).expect("deserialization should succeed");

    assert_eq!(deserialized.verification_dim, Some(256));
}

#[test]
fn test_serde_round_trip_all_proven() {
    let status = MoonshotStatus::from_repo();
    let mut cert =
        MoonshotCertificate::from_status(&status, "test-model", "English text", "abc123");

    // Set all properties to proven levels
    for (i, prop) in cert.properties.iter_mut().enumerate() {
        prop.level = if i < 6 {
            VerificationLevel::CrownProven
        } else if i == 6 {
            VerificationLevel::KaniProven
        } else {
            VerificationLevel::SmtProven
        };
        prop.bound_value = Some(1.0);
        prop.threshold = Some(0.5);
    }
    cert.all_at_least_partial = true;
    cert.all_proven = true;

    let json = cert.to_json_serde().expect("serialization should succeed");
    let deserialized =
        MoonshotCertificate::from_json(&json).expect("deserialization should succeed");

    assert!(deserialized.all_proven);
    assert!(deserialized.all_at_least_partial);
    for (i, prop) in deserialized.properties.iter().enumerate() {
        assert!(prop.bound_value.is_some(), "bound_value missing at {i}");
        assert!(prop.threshold.is_some(), "threshold missing at {i}");
    }
}

// --- Verification level serde tests ---

#[test]
fn test_verification_level_serde_round_trip() {
    let levels = [
        VerificationLevel::None,
        VerificationLevel::Empirical,
        VerificationLevel::CrownPartial,
        VerificationLevel::CrownProven,
        VerificationLevel::KaniProven,
        VerificationLevel::SmtProven,
    ];

    for level in &levels {
        let json = serde_json::to_string(level).expect("serialize");
        let deserialized: VerificationLevel = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(*level, deserialized, "round-trip failed for {level}");
    }
}

#[test]
fn test_kani_evidence_serde_round_trip() {
    let evidence = KaniVerificationEvidence {
        harnesses_passed: 300,
        harnesses_total: 475,
        harness_files: vec![
            "crates/nn-core/src/kani_bounds.rs".to_string(),
            "crates/nn-autodiff/src/kani_backward_proofs.rs".to_string(),
        ],
        all_passed: false,
    };

    let json = serde_json::to_string_pretty(&evidence).expect("serialize");
    let deserialized: KaniVerificationEvidence = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(deserialized.harnesses_passed, 300);
    assert_eq!(deserialized.harnesses_total, 475);
    assert_eq!(deserialized.harness_files.len(), 2);
    assert!(!deserialized.all_passed);
}

#[test]
fn test_smt_evidence_serde_round_trip() {
    let evidence = SmtVerificationEvidence {
        kernels_proven: 10,
        kernels_total: 20,
        proven_kernel_names: vec!["snake".to_string(), "silu_mul".to_string()],
        all_proven: false,
    };

    let json = serde_json::to_string_pretty(&evidence).expect("serialize");
    let deserialized: SmtVerificationEvidence = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(deserialized.kernels_proven, 10);
    assert_eq!(deserialized.kernels_total, 20);
    assert_eq!(deserialized.proven_kernel_names.len(), 2);
    assert!(!deserialized.all_proven);
}

// --- Error path tests ---

#[test]
fn test_from_json_invalid_json() {
    let result = MoonshotCertificate::from_json("not valid json");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        format!("{err}").contains("JSON parse error"),
        "expected JSON parse error, got: {err}"
    );
}

#[test]
fn test_from_json_property_name_mismatch() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test-model", "English text", "abc123");

    let mut json = cert.to_json_serde().expect("serialize");
    // Corrupt a property name
    json = json.replace("Non-silent (RMS > 0.01)", "WRONG NAME");

    let result = MoonshotCertificate::from_json(&json);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        format!("{err}").contains("property name mismatch"),
        "expected name mismatch error, got: {err}"
    );
}

#[test]
fn test_from_json_property_index_out_of_range() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test-model", "English text", "abc123");

    let mut json = cert.to_json_serde().expect("serialize");
    // Change property_index 0 to 99
    json = json.replacen("\"property_index\": 0", "\"property_index\": 99", 1);

    let result = MoonshotCertificate::from_json(&json);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        format!("{err}").contains("out of range"),
        "expected out of range error, got: {err}"
    );
}

// --- Cross-compatibility with hand-rolled to_json() ---

#[test]
fn test_serde_json_contains_same_fields_as_hand_rolled() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test-model", "English text", "abc123");

    let hand_rolled = cert.to_json();
    let serde_json = cert.to_json_serde().expect("serialize");

    // Both should contain the same key fields
    for field in &[
        "model_name",
        "input_specification",
        "source_hash",
        "verification_date",
        "all_at_least_partial",
        "all_proven",
        "properties",
    ] {
        assert!(
            hand_rolled.contains(field),
            "hand-rolled JSON missing field: {field}"
        );
        assert!(
            serde_json.contains(field),
            "serde JSON missing field: {field}"
        );
    }
}

// --- Double round-trip ---

#[test]
fn test_double_round_trip() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test-model", "English text", "abc123");

    // First round-trip
    let json1 = cert.to_json_serde().expect("serialize 1");
    let cert2 = MoonshotCertificate::from_json(&json1).expect("deserialize 1");

    // Second round-trip
    let json2 = cert2.to_json_serde().expect("serialize 2");
    let cert3 = MoonshotCertificate::from_json(&json2).expect("deserialize 2");

    // JSON outputs should be identical
    assert_eq!(json1, json2, "double round-trip JSON should be identical");

    // Final certificate should match
    assert_eq!(cert3.model_name, cert.model_name);
    assert_eq!(cert3.properties.len(), cert.properties.len());
}

// Serde+validate integration tests and SubCondition round-trip tests
// extracted to moonshot_certificate_serde_validate_tests.rs via #[path]
// submodule.

// Shared helpers used by lifecycle_tests submodule via `super::`.
fn test_kani_evidence() -> KaniVerificationEvidence {
    KaniVerificationEvidence {
        harnesses_passed: 459,
        harnesses_total: 459,
        harness_files: vec!["crates/nn-core/src/kani_bounds.rs".to_string()],
        all_passed: true,
    }
}

fn test_smt_evidence() -> SmtVerificationEvidence {
    SmtVerificationEvidence {
        kernels_proven: 20,
        kernels_total: 20,
        proven_kernel_names: vec!["snake".to_string(), "silu_mul".to_string()],
        all_proven: true,
    }
}

#[path = "moonshot_certificate_serde_lifecycle_tests.rs"]
mod lifecycle_tests;
