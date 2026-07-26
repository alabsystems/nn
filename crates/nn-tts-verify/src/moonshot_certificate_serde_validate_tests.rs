// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Serde + validate integration tests and SubCondition round-trip tests
//! for MoonshotCertificate.
//!
//! Extracted from `moonshot_certificate_serde_tests.rs` for the 500-line limit.

use super::*;

/// Helper: round-trip a certificate through serde and assert no structural errors.
fn assert_serde_validate_no_structural_errors(cert: &MoonshotCertificate) -> MoonshotCertificate {
    let json = cert.to_json_serde().expect("serialize");
    let de = MoonshotCertificate::from_json(&json).expect("deserialize");
    let result = validate_certificate(&de, std::path::Path::new("/nonexistent"));
    let errors: Vec<_> = result
        .findings
        .iter()
        .filter(|f| f.severity == FindingSeverity::Error && !f.message.contains("artifact"))
        .collect();
    assert!(
        errors.is_empty(),
        "structural errors after round-trip: {errors:?}"
    );
    de
}

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

#[test]
fn test_serde_round_trip_then_validate_structural() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test-model", "spec", "abc123");
    assert_serde_validate_no_structural_errors(&cert);
}

#[test]
fn test_schema_version_backward_compat_v1_json() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test-model", "spec", "abc123");
    let json = cert.to_json_serde().expect("serialize");
    // Remove schema_version line to simulate v1 JSON
    let v1_json: String = json
        .lines()
        .filter(|l| !l.contains("schema_version"))
        .collect::<Vec<_>>()
        .join("\n");
    let de = MoonshotCertificate::from_json(&v1_json).expect("v1 JSON should parse");
    assert_eq!(
        de.schema_version, 1,
        "missing schema_version should default to 1"
    );
}

#[test]
fn test_schema_version_v2_preserved_through_round_trip() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test-model", "spec", "abc123");
    assert_eq!(cert.schema_version, CERTIFICATE_SCHEMA_VERSION);
    let json = cert.to_json_serde().expect("serialize");
    let de = MoonshotCertificate::from_json(&json).expect("deserialize");
    assert_eq!(
        de.schema_version, CERTIFICATE_SCHEMA_VERSION,
        "schema_version should survive"
    );
}

#[test]
fn test_enriched_kani_smt_serde_then_validate() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test-model", "spec", "hash123");
    let enriched = cert
        .with_kani_results(&test_kani_evidence())
        .with_smt_results(&test_smt_evidence());
    let de = assert_serde_validate_no_structural_errors(&enriched);
    assert_eq!(de.properties[6].level, VerificationLevel::KaniProven);
    assert_eq!(de.properties[7].level, VerificationLevel::SmtProven);
}

#[test]
fn test_builder_kani_smt_serde_then_validate() {
    let cert = FullCertificateBuilder::new("builder-model", "English text", "hash456")
        .kani(&test_kani_evidence())
        .smt(&test_smt_evidence())
        .build();
    let de = assert_serde_validate_no_structural_errors(&cert);
    assert_eq!(de.model_name, "builder-model");
    assert_eq!(de.schema_version, CERTIFICATE_SCHEMA_VERSION);
    assert_eq!(de.properties[6].level, VerificationLevel::KaniProven);
    assert_eq!(de.properties[7].level, VerificationLevel::SmtProven);
}

#[test]
fn test_aggregate_flags_consistent_after_round_trip() {
    let status = MoonshotStatus::from_repo();
    let mut cert = MoonshotCertificate::from_status(&status, "test-model", "spec", "hash789");
    for (i, prop) in cert.properties.iter_mut().enumerate() {
        prop.level = match i {
            6 => VerificationLevel::KaniProven,
            7 => VerificationLevel::SmtProven,
            _ => VerificationLevel::CrownProven,
        };
    }
    cert.recompute_aggregate_flags();
    assert!(cert.all_at_least_partial);
    assert!(cert.all_proven);

    let de = assert_serde_validate_no_structural_errors(&cert);
    assert!(de.all_at_least_partial);
    assert!(de.all_proven);

    let result = validate_certificate(&de, std::path::Path::new("/nonexistent"));
    assert!(
        !result
            .findings
            .iter()
            .any(|f| f.message.contains("all_at_least_partial")
                || f.message.contains("all_proven")),
        "consistent aggregate flags should not be flagged"
    );
}

// --- SubCondition sub_results round-trip (#1925) ---

#[test]
fn test_serde_round_trip_with_sub_results() {
    use super::SubCondition;

    let status = MoonshotStatus::from_repo();
    let mut cert = MoonshotCertificate::from_status(&status, "test-model", "spec", "hash_sub");

    // Inject sub_results into property 4 (temporal+memory collision target).
    cert.properties[4].sub_results = vec![SubCondition {
        name: "memory_boundedness".to_string(),
        bound_value: 1_000_000_000.0,
        threshold: 2_000_000_000.0,
        proven: true,
        explanation: "WITHIN BOUND: 1000000000 <= 2000000000 bytes".to_string(),
    }];

    let json = cert.to_json_serde().expect("serialize with sub_results");
    assert!(
        json.contains("sub_results"),
        "JSON should contain sub_results when non-empty"
    );
    assert!(
        json.contains("memory_boundedness"),
        "JSON should contain the sub-condition name"
    );

    let de = MoonshotCertificate::from_json(&json).expect("deserialize with sub_results");
    assert_eq!(de.properties[4].sub_results.len(), 1);
    let sub = &de.properties[4].sub_results[0];
    assert_eq!(sub.name, "memory_boundedness");
    assert!((sub.bound_value - 1_000_000_000.0).abs() < 1.0);
    assert!((sub.threshold - 2_000_000_000.0).abs() < 1.0);
    assert!(sub.proven);
    assert!(sub.explanation.contains("WITHIN BOUND"));

    // Verify other properties have empty sub_results.
    for (i, prop) in de.properties.iter().enumerate() {
        if i != 4 {
            assert!(
                prop.sub_results.is_empty(),
                "property {i} should have empty sub_results"
            );
        }
    }
}

#[test]
fn test_serde_v2_json_missing_sub_results_defaults_empty() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test-model", "spec", "hash_v2");
    let json = cert.to_json_serde().expect("serialize");

    // v2 JSON should NOT contain sub_results (all empty, skip_serializing_if).
    assert!(
        !json.contains("sub_results"),
        "empty sub_results should be omitted"
    );

    // Parsing this v2 JSON should default sub_results to empty vec.
    let de = MoonshotCertificate::from_json(&json).expect("parse v2 JSON");
    for (i, prop) in de.properties.iter().enumerate() {
        assert!(
            prop.sub_results.is_empty(),
            "property {i} sub_results should default to empty for v2 JSON"
        );
    }
}
