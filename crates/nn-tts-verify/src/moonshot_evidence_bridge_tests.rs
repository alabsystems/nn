// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the moonshot evidence bridge functions.
//!
//! P7 (Kani) tests use a temp directory with mock `.rs` files.
//! P8 (ay SMT) tests use `serde_json::from_str` to construct synthetic
//! `VerifyStatus` objects (required because `KernelStatus` and
//! `SmtStatusRecord` are `#[non_exhaustive]`).

use super::*;

// ---------------------------------------------------------------------------
// P7: KaniVerificationEvidence::from_workspace_scan tests
// ---------------------------------------------------------------------------

/// Empty directory → zero harnesses.
#[test]
fn test_kani_scan_empty_dir() {
    let tmp = std::env::temp_dir().join(format!("kani_scan_empty_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);

    let evidence = KaniVerificationEvidence::from_workspace_scan(&tmp, true);
    assert_eq!(evidence.harnesses_total, 0);
    assert_eq!(evidence.harnesses_passed, 0);
    assert!(evidence.harness_files.is_empty());
    assert!(!evidence.all_passed, "zero harnesses → not all_passed");

    let _ = std::fs::remove_dir_all(&tmp);
}

/// Non-existent directory → zero harnesses (graceful, not panic).
#[test]
fn test_kani_scan_nonexistent_dir() {
    let evidence = KaniVerificationEvidence::from_workspace_scan(
        Path::new("/tmp/nonexistent_kani_scan_dir_31415926"),
        true,
    );
    assert_eq!(evidence.harnesses_total, 0);
    assert!(!evidence.all_passed);
}

/// Temp directory with known `.rs` files containing `#[kani::proof]`.
#[test]
fn test_kani_scan_with_mock_harnesses() {
    let tmp = std::env::temp_dir().join(format!("kani_scan_mock_{}", std::process::id()));
    let sub = tmp.join("nn-core/src");
    std::fs::create_dir_all(&sub).expect("create mock crate dir");

    // File with 2 harnesses.
    std::fs::write(
        sub.join("kani_proofs.rs"),
        r#"
#[cfg(kani)]
mod proofs {
    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_add_bounds() { /* ... */ }

    #[kani::unwind(1)]
    #[kani::proof]
    fn prove_mul_bounds() { /* ... */ }
}
"#,
    )
    .expect("write mock harness file");

    // File with 1 harness.
    std::fs::write(
        sub.join("kani_extra.rs"),
        "#[kani::proof]\nfn prove_something() {}\n",
    )
    .expect("write second mock harness file");

    // File with zero harnesses.
    std::fs::write(sub.join("lib.rs"), "pub mod kani_proofs;\n").expect("write non-harness file");

    let evidence = KaniVerificationEvidence::from_workspace_scan(&tmp, true);
    assert_eq!(evidence.harnesses_total, 3, "expected 3 harnesses");
    assert_eq!(evidence.harnesses_passed, 3, "assume_all_pass = true");
    assert!(evidence.all_passed);
    assert_eq!(evidence.harness_files.len(), 2, "2 files contain harnesses");

    let _ = std::fs::remove_dir_all(&tmp);
}

/// `assume_all_pass = false` → harnesses_passed = 0.
#[test]
fn test_kani_scan_assume_all_pass_false() {
    let tmp = std::env::temp_dir().join(format!("kani_scan_no_assume_{}", std::process::id()));
    let sub = tmp.join("src");
    std::fs::create_dir_all(&sub).expect("create dir");
    std::fs::write(sub.join("proofs.rs"), "#[kani::proof]\nfn prove_x() {}\n").expect("write");

    let evidence = KaniVerificationEvidence::from_workspace_scan(&tmp, false);
    assert_eq!(evidence.harnesses_total, 1);
    assert_eq!(evidence.harnesses_passed, 0, "assume_all_pass = false");
    assert!(!evidence.all_passed);

    let _ = std::fs::remove_dir_all(&tmp);
}

/// Scan the real workspace `crates/` directory — should find >400 harnesses.
#[test]
fn test_kani_scan_real_workspace() {
    // Resolve workspace root from this file's location:
    // crates/nn-tts-verify/src/ → ../../.. → workspace root
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent() // crates/
        .and_then(|p| p.parent()); // workspace root

    let Some(root) = workspace_root else {
        // If we can't resolve the workspace root, skip gracefully.
        return;
    };
    let crates_dir = root.join("crates");
    if !crates_dir.is_dir() {
        return;
    }

    let evidence = KaniVerificationEvidence::from_workspace_scan(&crates_dir, true);

    // Workspace has 560+ Kani harnesses as of 2026-03-19.
    // Use a threshold that catches regressions without being fragile.
    assert!(
        evidence.harnesses_total >= 500,
        "expected >=500 Kani harnesses in workspace, found {}",
        evidence.harnesses_total,
    );
    assert!(evidence.all_passed, "assume_all_pass = true");
    assert!(
        evidence.harness_files.len() >= 10,
        "expected >=10 files with Kani harnesses, found {}",
        evidence.harness_files.len(),
    );
}

// ---------------------------------------------------------------------------
// P7: KaniVerificationEvidence::from_kani_status_file tests
// ---------------------------------------------------------------------------

/// All harnesses passed → all_passed = true.
#[test]
fn test_kani_status_file_all_passed() {
    let tmp = std::env::temp_dir().join(format!("kani_status_all_pass_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("create tmp dir");

    let json = r#"{
        "harnesses": {
            "nn_core::kani_bounds::prove_add_bounds": {
                "status": "passed",
                "duration_sec": 12.3,
                "commit": "abc123"
            },
            "nn_core::kani_bounds::prove_mul_bounds": {
                "status": "passed",
                "duration_sec": 8.1,
                "commit": "abc123"
            },
            "nn_dsl::snake::prove_snake_no_overflow": {
                "status": "passed",
                "duration_sec": 45.0,
                "commit": "abc123"
            }
        }
    }"#;

    let status_path = tmp.join("kani_status.json");
    std::fs::write(&status_path, json).expect("write kani_status.json");

    // Use a non-existent crates_dir — harness_files will be empty.
    let crates_dir = tmp.join("nonexistent_crates");
    let evidence = KaniVerificationEvidence::from_kani_status_file(&status_path, &crates_dir);
    let evidence = evidence.expect("should parse valid kani_status.json");

    assert_eq!(evidence.harnesses_total, 3);
    assert_eq!(evidence.harnesses_passed, 3);
    assert!(evidence.all_passed);
    assert!(
        evidence.harness_files.is_empty(),
        "no crates_dir → empty files"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// Mixed results: some passed, some failed/timeout → not all_passed.
#[test]
fn test_kani_status_file_mixed_results() {
    let tmp = std::env::temp_dir().join(format!("kani_status_mixed_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("create tmp dir");

    let json = r#"{
        "harnesses": {
            "harness_a": {"status": "passed", "duration_sec": 1.0},
            "harness_b": {"status": "failed", "duration_sec": 2.0},
            "harness_c": {"status": "timeout", "duration_sec": 300.0},
            "harness_d": {"status": "passed", "duration_sec": 1.5}
        }
    }"#;

    let status_path = tmp.join("kani_status.json");
    std::fs::write(&status_path, json).expect("write");

    let evidence =
        KaniVerificationEvidence::from_kani_status_file(&status_path, Path::new("/nonexistent"))
            .expect("should parse");

    assert_eq!(evidence.harnesses_total, 4);
    assert_eq!(evidence.harnesses_passed, 2);
    assert!(!evidence.all_passed);

    let _ = std::fs::remove_dir_all(&tmp);
}

/// Empty harnesses map → total 0, not all_passed.
#[test]
fn test_kani_status_file_empty_harnesses() {
    let tmp = std::env::temp_dir().join(format!("kani_status_empty_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("create tmp dir");

    let json = r#"{"harnesses": {}}"#;
    let status_path = tmp.join("kani_status.json");
    std::fs::write(&status_path, json).expect("write");

    let evidence =
        KaniVerificationEvidence::from_kani_status_file(&status_path, Path::new("/nonexistent"))
            .expect("should parse empty harnesses");

    assert_eq!(evidence.harnesses_total, 0);
    assert_eq!(evidence.harnesses_passed, 0);
    assert!(!evidence.all_passed, "zero harnesses → not all_passed");

    let _ = std::fs::remove_dir_all(&tmp);
}

/// Non-existent file → None.
#[test]
fn test_kani_status_file_nonexistent() {
    let evidence = KaniVerificationEvidence::from_kani_status_file(
        Path::new("/tmp/nonexistent_kani_status_file_31415926.json"),
        Path::new("/nonexistent"),
    );
    assert!(evidence.is_none(), "non-existent file → None");
}

/// Malformed JSON → None.
#[test]
fn test_kani_status_file_malformed_json() {
    let tmp = std::env::temp_dir().join(format!("kani_status_bad_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("create tmp dir");

    let status_path = tmp.join("kani_status.json");
    std::fs::write(&status_path, "not valid json {{{").expect("write");

    let evidence =
        KaniVerificationEvidence::from_kani_status_file(&status_path, Path::new("/nonexistent"));
    assert!(evidence.is_none(), "malformed JSON → None");

    let _ = std::fs::remove_dir_all(&tmp);
}

/// Only "not_run" harnesses → total > 0, passed = 0, not all_passed.
#[test]
fn test_kani_status_file_all_not_run() {
    let tmp = std::env::temp_dir().join(format!("kani_status_notrun_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("create tmp dir");

    let json = r#"{
        "harnesses": {
            "harness_x": {"status": "not_run"},
            "harness_y": {"status": "not_run"}
        }
    }"#;
    let status_path = tmp.join("kani_status.json");
    std::fs::write(&status_path, json).expect("write");

    let evidence =
        KaniVerificationEvidence::from_kani_status_file(&status_path, Path::new("/nonexistent"))
            .expect("should parse");

    assert_eq!(evidence.harnesses_total, 2);
    assert_eq!(evidence.harnesses_passed, 0);
    assert!(!evidence.all_passed);

    let _ = std::fs::remove_dir_all(&tmp);
}

// ---------------------------------------------------------------------------
// P7: KaniVerificationEvidence::from_kani_status_or_scan tests
// ---------------------------------------------------------------------------

/// When kani_status.json exists, uses it (not workspace scan).
#[test]
fn test_kani_status_or_scan_prefers_status_file() {
    let tmp = std::env::temp_dir().join(format!("kani_or_scan_pref_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("create tmp dir");

    // Write a status file with 2 passed, 1 failed.
    let json = r#"{
        "harnesses": {
            "h1": {"status": "passed"},
            "h2": {"status": "passed"},
            "h3": {"status": "failed"}
        }
    }"#;
    let status_path = tmp.join("kani_status.json");
    std::fs::write(&status_path, json).expect("write");

    let evidence = KaniVerificationEvidence::from_kani_status_or_scan(
        &status_path,
        Path::new("/nonexistent_crates"),
        true, // this would give all_passed if scan fallback were used
    );

    // Should reflect status file: 2/3 passed, NOT all_passed.
    assert_eq!(evidence.harnesses_total, 3);
    assert_eq!(evidence.harnesses_passed, 2);
    assert!(
        !evidence.all_passed,
        "status file shows 2/3, not all passed"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// When kani_status.json is missing, falls back to workspace scan.
#[test]
fn test_kani_status_or_scan_fallback_to_scan() {
    let tmp = std::env::temp_dir().join(format!("kani_or_scan_fb_{}", std::process::id()));
    let sub = tmp.join("src");
    std::fs::create_dir_all(&sub).expect("create dir");

    // Write a mock harness file.
    std::fs::write(sub.join("proofs.rs"), "#[kani::proof]\nfn prove_x() {}\n").expect("write");

    // Point to a non-existent status file.
    let nonexistent_status = tmp.join("nonexistent_kani_status.json");

    let evidence = KaniVerificationEvidence::from_kani_status_or_scan(
        &nonexistent_status,
        &tmp,
        false, // assume_all_pass = false
    );

    // Should reflect workspace scan: 1 harness found, 0 passed.
    assert_eq!(evidence.harnesses_total, 1);
    assert_eq!(
        evidence.harnesses_passed, 0,
        "fallback with assume_all_pass=false"
    );
    assert!(!evidence.all_passed);

    let _ = std::fs::remove_dir_all(&tmp);
}

/// Fallback with assume_all_pass = true gives all_passed.
#[test]
fn test_kani_status_or_scan_fallback_assume_true() {
    let tmp = std::env::temp_dir().join(format!("kani_or_scan_at_{}", std::process::id()));
    let sub = tmp.join("src");
    std::fs::create_dir_all(&sub).expect("create dir");

    std::fs::write(
        sub.join("proofs.rs"),
        "#[kani::proof]\nfn prove_a() {}\n#[kani::proof]\nfn prove_b() {}\n",
    )
    .expect("write");

    let nonexistent_status = tmp.join("nonexistent_kani_status.json");

    let evidence = KaniVerificationEvidence::from_kani_status_or_scan(
        &nonexistent_status,
        &tmp,
        true, // assume_all_pass = true
    );

    assert_eq!(evidence.harnesses_total, 2);
    assert_eq!(
        evidence.harnesses_passed, 2,
        "fallback with assume_all_pass=true"
    );
    assert!(evidence.all_passed);

    let _ = std::fs::remove_dir_all(&tmp);
}

// --- P8: SMT bridge tests (extracted) ----------------------------------------

#[cfg(feature = "ny")]
#[path = "moonshot_evidence_bridge_smt_tests.rs"]
mod smt_bridge_tests;
