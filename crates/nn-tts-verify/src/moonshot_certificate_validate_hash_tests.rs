// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Source-hash and bound-direction validation tests.
//!
//! Extracted from `moonshot_certificate_validate_tests.rs` (Phase 37 of #1741).

use super::*;

// ---- Source hash tests ----

#[test]
fn test_is_valid_sha256_hex_correct() {
    let valid = "a".repeat(64);
    assert!(is_valid_sha256_hex(&valid));

    let valid_mixed = "0123456789abcdef".repeat(4);
    assert!(is_valid_sha256_hex(&valid_mixed));
}

#[test]
fn test_is_valid_sha256_hex_wrong_length() {
    assert!(!is_valid_sha256_hex("abc123"));
    assert!(!is_valid_sha256_hex(""));
    assert!(!is_valid_sha256_hex(&"a".repeat(63)));
    assert!(!is_valid_sha256_hex(&"a".repeat(65)));
}

#[test]
fn test_is_valid_sha256_hex_non_hex_chars() {
    let mut invalid = "a".repeat(64);
    invalid.replace_range(0..1, "g");
    assert!(!is_valid_sha256_hex(&invalid));
}

#[test]
fn test_compute_workspace_source_hash_real_crates() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent());

    if let Some(root) = repo_root {
        let result = compute_workspace_source_hash(root);
        assert!(result.is_ok(), "Should compute hash: {:?}", result.err());
        let hash = result.unwrap();
        assert!(
            is_valid_sha256_hex(&hash),
            "Hash should be valid SHA-256 hex, got: {hash}"
        );
    }
}

#[test]
fn test_compute_workspace_source_hash_deterministic() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent());

    if let Some(root) = repo_root {
        let hash1 = compute_workspace_source_hash(root).unwrap();
        let hash2 = compute_workspace_source_hash(root).unwrap();
        assert_eq!(hash1, hash2, "Hash should be deterministic across calls");
    }
}

#[test]
fn test_compute_workspace_source_hash_missing_crates_dir() {
    let result = compute_workspace_source_hash(std::path::Path::new("/nonexistent"));
    assert!(
        result.is_err(),
        "Should error when crates/ directory is missing"
    );
}

#[test]
fn test_validate_source_hash_match() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent());

    if let Some(root) = repo_root {
        let hash = compute_workspace_source_hash(root).unwrap();
        let status = MoonshotStatus::from_repo();
        let cert = MoonshotCertificate::from_status(&status, "test", "spec", &hash);

        let result = validate_certificate(&cert, root);
        assert_eq!(
            result.source_hash_match,
            Some(true),
            "Certificate with current hash should match, findings: {:?}",
            result.findings
        );
    }
}

#[test]
fn test_validate_source_hash_mismatch() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent());

    if let Some(root) = repo_root {
        let fake_hash = "0".repeat(64);
        let status = MoonshotStatus::from_repo();
        let cert = MoonshotCertificate::from_status(&status, "test", "spec", &fake_hash);

        let result = validate_certificate(&cert, root);
        assert_eq!(
            result.source_hash_match,
            Some(false),
            "Certificate with wrong hash should mismatch"
        );
        assert!(
            result
                .findings
                .iter()
                .any(|f| f.message.contains("source_hash mismatch")),
            "Should report source_hash mismatch finding"
        );
    }
}

#[test]
fn test_validate_source_hash_invalid_format() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test", "spec", "not-a-valid-hex");

    let result = validate_certificate(&cert, std::path::Path::new("/tmp"));
    assert_eq!(
        result.source_hash_match, None,
        "Invalid-format hash should skip validation"
    );
    assert!(
        result
            .findings
            .iter()
            .any(|f| f.message.contains("not a valid SHA-256 hex digest")),
        "Should warn about invalid hash format"
    );
}

#[test]
fn test_validate_source_hash_empty_skipped() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test", "spec", "");

    let result = validate_certificate(&cert, std::path::Path::new("/nonexistent"));
    assert_eq!(
        result.source_hash_match, None,
        "Empty hash should skip source hash validation"
    );
}

#[test]
fn test_validate_display_shows_source_hash_status() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent());

    if let Some(root) = repo_root {
        let hash = compute_workspace_source_hash(root).unwrap();
        let status = MoonshotStatus::from_repo();
        let cert = MoonshotCertificate::from_status(&status, "test", "spec", &hash);

        let result = validate_certificate(&cert, root);
        let display = format!("{result}");
        assert!(
            display.contains("source_hash: MATCH"),
            "Display should show source_hash status, got: {display}"
        );
    }
}

// ---- Bound-threshold direction tests ----

#[test]
fn test_bound_direction_p1_violation() {
    // P1 (Non-silent): bound > threshold required.  Violate: bound <= threshold.
    let mut cert = valid_certificate();
    cert.properties[0].level = VerificationLevel::CrownProven;
    cert.properties[0].bound_value = Some(0.005); // below threshold
    cert.properties[0].threshold = Some(0.01);
    cert.properties[0].proof_artifacts = vec!["fake.rs".to_string()];
    cert.recompute_aggregate_flags();
    let result = validate_certificate(&cert, std::path::Path::new("/nonexistent"));
    assert!(
        result
            .findings
            .iter()
            .any(|f| f.message.contains("bound/threshold direction violated")
                && f.message.contains("bound > threshold")
                && f.property_index == Some(0)),
        "P1 with bound <= threshold should be flagged, findings: {:?}",
        result.findings
    );
}

#[test]
fn test_bound_direction_p2_violation() {
    // P2 (Non-clipping): bound <= threshold required.  Violate: bound > threshold.
    let mut cert = valid_certificate();
    cert.properties[1].level = VerificationLevel::CrownProven;
    cert.properties[1].bound_value = Some(1.5); // above threshold
    cert.properties[1].threshold = Some(1.0);
    cert.properties[1].proof_artifacts = vec!["fake.rs".to_string()];
    cert.recompute_aggregate_flags();
    let result = validate_certificate(&cert, std::path::Path::new("/nonexistent"));
    assert!(
        result
            .findings
            .iter()
            .any(|f| f.message.contains("bound/threshold direction violated")
                && f.property_index == Some(1)),
        "P2 with bound > threshold should be flagged"
    );
}

#[test]
fn test_bound_direction_p7_violation() {
    // P7 (Memory-safe): bound >= threshold required.  Violate: bound < threshold.
    let mut cert = valid_certificate();
    cert.properties[6].level = VerificationLevel::KaniProven;
    cert.properties[6].bound_value = Some(400.0); // below threshold
    cert.properties[6].threshold = Some(500.0);
    cert.properties[6].proof_artifacts = vec!["fake.rs".to_string()];
    cert.recompute_aggregate_flags();
    let result = validate_certificate(&cert, std::path::Path::new("/nonexistent"));
    assert!(
        result
            .findings
            .iter()
            .any(|f| f.message.contains("bound/threshold direction violated")
                && f.property_index == Some(6)),
        "P7 with bound < threshold should be flagged"
    );
}

#[test]
fn test_bound_direction_valid_passes() {
    // P7 with bound >= threshold should NOT produce a direction finding.
    let mut cert = valid_certificate();
    cert.properties[6].level = VerificationLevel::KaniProven;
    cert.properties[6].bound_value = Some(500.0);
    cert.properties[6].threshold = Some(500.0);
    cert.properties[6].proof_artifacts = vec!["fake.rs".to_string()];
    cert.recompute_aggregate_flags();
    let result = validate_certificate(&cert, std::path::Path::new("/nonexistent"));
    assert!(
        !result
            .findings
            .iter()
            .any(|f| f.message.contains("bound/threshold direction violated")
                && f.property_index == Some(6)),
        "P7 with bound == threshold should not be flagged"
    );
}

#[test]
fn test_bound_direction_skips_low_level() {
    // Properties below CrownPartial should not be checked.
    let mut cert = valid_certificate();
    cert.properties[0].level = VerificationLevel::Empirical;
    cert.properties[0].bound_value = Some(0.001); // would violate P1 direction
    cert.properties[0].threshold = Some(0.01);
    let result = validate_certificate(&cert, std::path::Path::new("/nonexistent"));
    assert!(
        !result
            .findings
            .iter()
            .any(|f| f.message.contains("bound/threshold direction violated")
                && f.property_index == Some(0)),
        "Empirical-level property should not trigger direction check"
    );
}
