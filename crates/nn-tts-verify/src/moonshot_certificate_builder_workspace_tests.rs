// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`build_certificate_from_workspace`].
//!
//! These tests use temp directories with mock `kani_status.json` files
//! and workspace scan fallback to exercise the workspace-driven certificate
//! builder path.
//!
//! Extracted from `moonshot_certificate_builder_tests.rs` (Phase 37 of #1741).

use super::*;
use crate::moonshot::VerificationLevel;

#[test]
fn test_from_workspace_all_kani_passed() {
    use std::path::Path;

    let tmp = std::env::temp_dir().join(format!("cert_from_ws_all_pass_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("create tmp dir");

    let json = r#"{
        "harnesses": {
            "nn_core::kani_bounds::prove_add": {"status": "passed"},
            "nn_core::kani_bounds::prove_mul": {"status": "passed"},
            "nn_dsl::snake::prove_snake": {"status": "passed"}
        }
    }"#;
    let status_path = tmp.join("kani_status.json");
    std::fs::write(&status_path, json).expect("write kani_status.json");

    let cert = build_certificate_from_workspace(
        "test-model",
        "test input",
        "sha256",
        &status_path,
        Path::new("/nonexistent_crates"),
        false,
        None,
        None,
        None,
    );

    assert_eq!(cert.model_name, "test-model");
    assert_eq!(cert.properties.len(), 8);
    assert_eq!(
        cert.properties[6].level,
        VerificationLevel::KaniProven,
        "P7 should be KaniProven when all harnesses pass"
    );
    assert_eq!(
        cert.properties[6].bound_value,
        Some(3.0),
        "3 harnesses passed"
    );
    assert_eq!(cert.properties[6].threshold, Some(3.0), "3 harnesses total");

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn test_from_workspace_mixed_kani_results() {
    use std::path::Path;

    let tmp = std::env::temp_dir().join(format!("cert_from_ws_mixed_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("create tmp dir");

    let json = r#"{
        "harnesses": {
            "h1": {"status": "passed"},
            "h2": {"status": "failed"},
            "h3": {"status": "passed"},
            "h4": {"status": "timeout"}
        }
    }"#;
    let status_path = tmp.join("kani_status.json");
    std::fs::write(&status_path, json).expect("write");

    let cert = build_certificate_from_workspace(
        "model",
        "input",
        "hash",
        &status_path,
        Path::new("/nonexistent"),
        false,
        None,
        None,
        None,
    );

    // 2/4 passed → Empirical (not KaniProven).
    assert_eq!(
        cert.properties[6].level,
        VerificationLevel::Empirical,
        "P7 should be Empirical with partial Kani pass"
    );
    assert_eq!(cert.properties[6].bound_value, Some(2.0));
    assert_eq!(cert.properties[6].threshold, Some(4.0));

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn test_from_workspace_fallback_to_scan() {
    let tmp = std::env::temp_dir().join(format!("cert_from_ws_fallback_{}", std::process::id()));
    let crates_sub = tmp.join("crates/nn-core/src");
    std::fs::create_dir_all(&crates_sub).expect("create mock crate dir");

    // Write a mock .rs file with 2 harnesses.
    std::fs::write(
        crates_sub.join("kani_proofs.rs"),
        "#[kani::proof]\nfn prove_a() {}\n\n#[kani::proof]\nfn prove_b() {}\n",
    )
    .expect("write mock harness file");

    // Use a non-existent kani_status.json path → falls back to workspace scan.
    let nonexistent_status = tmp.join("no_such_kani_status.json");
    let crates_dir = tmp.join("crates");

    let cert = build_certificate_from_workspace(
        "fallback-model",
        "input",
        "hash",
        &nonexistent_status,
        &crates_dir,
        true, // assume_all_pass_fallback
        None,
        None,
        None,
    );

    assert_eq!(
        cert.properties[6].level,
        VerificationLevel::KaniProven,
        "P7 should be KaniProven with assume_all_pass fallback"
    );
    assert_eq!(cert.properties[6].bound_value, Some(2.0), "2 harnesses");
    assert_eq!(cert.properties[6].threshold, Some(2.0));

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn test_from_workspace_with_crown_and_smt() {
    use std::path::Path;

    let dim = 64;
    let (_pipeline_cert, bundle) = test_crown_bundle(dim);
    let smt = test_smt_evidence(14, 14);

    let tmp = std::env::temp_dir().join(format!("cert_from_ws_full_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("create tmp dir");

    let json = r#"{
        "harnesses": {
            "h1": {"status": "passed"},
            "h2": {"status": "passed"}
        }
    }"#;
    let status_path = tmp.join("kani_status.json");
    std::fs::write(&status_path, json).expect("write");

    let cert = build_certificate_from_workspace(
        "full-model",
        "English text",
        "sha256",
        &status_path,
        Path::new("/nonexistent"),
        false,
        Some(&bundle),
        Some(&smt),
        None,
    );

    assert_eq!(cert.model_name, "full-model");
    assert_eq!(cert.verification_dim, Some(dim));
    assert_eq!(
        cert.properties[6].level,
        VerificationLevel::KaniProven,
        "P7 with all Kani passed"
    );
    assert_eq!(
        cert.properties[7].level,
        VerificationLevel::SmtProven,
        "P8 with all SMT proven"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn test_from_workspace_matches_manual_builder() {
    use std::path::Path;

    let tmp = std::env::temp_dir().join(format!("cert_from_ws_match_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("create tmp dir");

    let json = r#"{
        "harnesses": {
            "h1": {"status": "passed"},
            "h2": {"status": "passed"},
            "h3": {"status": "passed"}
        }
    }"#;
    let status_path = tmp.join("kani_status.json");
    std::fs::write(&status_path, json).expect("write");

    // build_certificate_from_workspace path.
    let from_ws = build_certificate_from_workspace(
        "model",
        "input",
        "hash",
        &status_path,
        Path::new("/nonexistent"),
        false,
        None,
        None,
        None,
    );

    // Manual construction: read evidence then pass to builder.
    let kani_evidence =
        KaniVerificationEvidence::from_kani_status_file(&status_path, Path::new("/nonexistent"))
            .expect("should parse");

    let manual = FullCertificateBuilder::new("model", "input", "hash")
        .kani(&kani_evidence)
        .build();

    // Same P7 results.
    assert_eq!(from_ws.properties[6].level, manual.properties[6].level);
    assert_eq!(
        from_ws.properties[6].bound_value,
        manual.properties[6].bound_value
    );
    assert_eq!(
        from_ws.properties[6].threshold,
        manual.properties[6].threshold
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
