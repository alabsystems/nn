// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for certificate validation.

use super::*;
use crate::moonshot::{
    compute_workspace_source_hash, is_valid_sha256_hex, MoonshotCertificate, MoonshotStatus,
    PropertyCertificate, VerificationLevel,
};

/// Helper: build a valid certificate from repo state.
fn valid_certificate() -> MoonshotCertificate {
    let status = MoonshotStatus::from_repo();
    MoonshotCertificate::from_status(
        &status,
        "test-model",
        "English text, ≤50 words",
        "abc123hash",
    )
}

#[test]
fn test_valid_certificate_passes_structural_validation() {
    let cert = valid_certificate();
    // Use a non-existent root so artifact checks fail but structural checks pass
    let result = validate_certificate(&cert, std::path::Path::new("/nonexistent"));
    // Structural checks should pass — no Error findings for structure
    let structural_errors: Vec<_> = result
        .findings
        .iter()
        .filter(|f| f.severity == FindingSeverity::Error)
        .collect();
    assert!(
        structural_errors.is_empty(),
        "Valid certificate should have no structural errors, got: {structural_errors:?}"
    );
}

#[test]
fn test_wrong_property_count_is_error() {
    let mut cert = valid_certificate();
    cert.properties.pop(); // Remove one property → 7 instead of 8
    let result = validate_certificate(&cert, std::path::Path::new("/nonexistent"));
    assert!(result.has_errors());
    assert!(
        result
            .findings
            .iter()
            .any(|f| f.message.contains("Expected 8 properties")),
        "Should flag wrong property count"
    );
}

#[test]
fn test_mismatched_property_name_is_error() {
    let mut cert = valid_certificate();
    // Corrupt property name at index 0
    cert.properties[0] = PropertyCertificate {
        property_index: 0,
        property_name: "WRONG NAME",
        level: VerificationLevel::None,
        proof_artifacts: vec![],
        assumptions: vec![],
        bound_value: None,
        threshold: None,
        sub_results: vec![],
        constructive_proof_lean4: None,
    };
    let result = validate_certificate(&cert, std::path::Path::new("/nonexistent"));
    assert!(result.has_errors());
    assert!(
        result
            .findings
            .iter()
            .any(|f| f.message.contains("Property name mismatch")),
        "Should flag name mismatch"
    );
}

#[test]
fn test_wrong_property_index_is_error() {
    let mut cert = valid_certificate();
    cert.properties[2].property_index = 5; // Index at position 2 says 5
    let result = validate_certificate(&cert, std::path::Path::new("/nonexistent"));
    assert!(result.has_errors());
    assert!(
        result
            .findings
            .iter()
            .any(|f| f.message.contains("has index 5 (expected 2)")),
        "Should flag wrong index"
    );
}

#[test]
fn test_aggregate_flag_mismatch_all_partial() {
    let mut cert = valid_certificate();
    // Force all_at_least_partial to true when it shouldn't be
    cert.all_at_least_partial = true;
    // Ensure at least one property is below CrownPartial
    let has_below_partial = cert
        .properties
        .iter()
        .any(|p| p.level < VerificationLevel::CrownPartial);
    if has_below_partial {
        let result = validate_certificate(&cert, std::path::Path::new("/nonexistent"));
        assert!(
            result
                .findings
                .iter()
                .any(|f| f.message.contains("all_at_least_partial")),
            "Should flag all_at_least_partial mismatch"
        );
    }
}

#[test]
fn test_aggregate_flag_mismatch_all_proven() {
    let mut cert = valid_certificate();
    // Force all_proven to true when some properties are at lower levels
    cert.all_proven = true;
    let has_unproven = cert.properties.iter().any(|p| {
        !matches!(
            p.level,
            VerificationLevel::CrownProven
                | VerificationLevel::KaniProven
                | VerificationLevel::SmtProven
        )
    });
    if has_unproven {
        let result = validate_certificate(&cert, std::path::Path::new("/nonexistent"));
        assert!(
            result
                .findings
                .iter()
                .any(|f| f.message.contains("all_proven")),
            "Should flag all_proven mismatch"
        );
    }
}

#[test]
fn test_missing_artifact_is_warning() {
    let mut cert = valid_certificate();
    cert.properties[0].proof_artifacts = vec!["nonexistent/file.rs".to_string()];
    // Use a guaranteed-nonexistent root to avoid matching real dirs under /tmp
    // (e.g. /tmp/crates/ exists on macOS from cargo build artifacts).
    let result = validate_certificate(&cert, std::path::Path::new("/nonexistent_validate_root"));
    assert!(
        result
            .findings
            .iter()
            .any(|f| f.message.contains("Proof artifact not found")),
        "Should warn about missing artifact"
    );
    assert_eq!(result.artifacts_found, 0);
    // At least 1 artifact is referenced (the one we added)
    assert!(result.artifacts_total >= 1);
}

#[test]
fn test_empty_model_name_is_error() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "", "input spec", "hash123");
    let result = validate_certificate(&cert, std::path::Path::new("/nonexistent"));
    assert!(
        result
            .findings
            .iter()
            .any(|f| f.message.contains("model_name is empty")),
        "Should flag empty model_name"
    );
}

#[test]
fn test_empty_source_hash_is_warning() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "model", "input spec", "");
    let result = validate_certificate(&cert, std::path::Path::new("/nonexistent"));
    assert!(
        result
            .findings
            .iter()
            .any(|f| f.message.contains("source_hash is empty")),
        "Should warn about empty source_hash"
    );
}

#[test]
fn test_non_finite_bound_value_is_error() {
    let mut cert = valid_certificate();
    cert.properties[0].bound_value = Some(f64::NAN);
    cert.properties[1].bound_value = Some(f64::INFINITY);
    let result = validate_certificate(&cert, std::path::Path::new("/nonexistent"));
    let nan_findings: Vec<_> = result
        .findings
        .iter()
        .filter(|f| f.message.contains("non-finite"))
        .collect();
    assert_eq!(
        nan_findings.len(),
        2,
        "Should flag both NaN and Inf bound_values, got: {nan_findings:?}"
    );
}

#[test]
fn test_non_finite_threshold_is_error() {
    let mut cert = valid_certificate();
    cert.properties[3].threshold = Some(f64::NEG_INFINITY);
    let result = validate_certificate(&cert, std::path::Path::new("/nonexistent"));
    assert!(
        result
            .findings
            .iter()
            .any(|f| f.message.contains("threshold is non-finite")),
        "Should flag non-finite threshold"
    );
}

#[test]
fn test_proven_without_artifacts_is_warning() {
    let mut cert = valid_certificate();
    cert.properties[0].level = VerificationLevel::CrownProven;
    cert.properties[0].proof_artifacts.clear();
    // Must also fix aggregate flags
    cert.recompute_aggregate_flags();
    let result = validate_certificate(&cert, std::path::Path::new("/nonexistent"));
    assert!(
        result
            .findings
            .iter()
            .any(|f| f.message.contains("no proof artifacts")),
        "Should warn about proven without artifacts"
    );
}

#[test]
fn test_serde_roundtrip_then_validate() {
    let cert = valid_certificate();
    let json = cert.to_json_serde().expect("serialize");
    let deserialized = MoonshotCertificate::from_json(&json).expect("deserialize");
    // Structural validation should pass on round-tripped certificate
    let result = validate_certificate(&deserialized, std::path::Path::new("/nonexistent"));
    let structural_errors: Vec<_> = result
        .findings
        .iter()
        .filter(|f| {
            f.severity == FindingSeverity::Error && !f.message.contains("non-finite")
            // Not about NaN
        })
        .collect();
    assert!(
        structural_errors.is_empty(),
        "Round-tripped certificate should have no structural errors, got: {structural_errors:?}"
    );
}

#[test]
fn test_validation_display_format() {
    let cert = valid_certificate();
    let result = validate_certificate(&cert, std::path::Path::new("/nonexistent"));
    let display = format!("{result}");
    assert!(
        display.contains("Certificate"),
        "Display should mention 'Certificate', got: {display}"
    );
}

#[test]
fn test_kani_scan_real_workspace() {
    // This test validates against the real crates/ directory
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent());

    if let Some(root) = repo_root {
        let crates_dir = root.join("crates");
        if crates_dir.is_dir() {
            let mut cert = valid_certificate();
            // Set P7 to KaniProven with a count
            cert.properties[6].level = VerificationLevel::KaniProven;
            cert.properties[6].bound_value = Some(400.0); // approximate
            cert.properties[6].threshold = Some(400.0);
            cert.recompute_aggregate_flags();

            let result = validate_certificate(&cert, root);
            // Should have scanned and found harnesses
            assert!(
                result.kani_scan_count.is_some(),
                "Should have performed Kani scan"
            );
            let count = result.kani_scan_count.unwrap();
            assert!(
                count > 100,
                "Expected >100 Kani harnesses in workspace, found {count}"
            );
        }
    }
}

// --- Source hash + bound direction tests (extracted) -------------------------

#[path = "moonshot_certificate_validate_hash_tests.rs"]
mod hash_tests;
