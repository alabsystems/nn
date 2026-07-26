// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Kokoro deployment certificate generation, serialization, and verification.
//!
//! Part of #3874, #4254.

use super::*;

// ---------------------------------------------------------------------------
// Helper: build a minimal VerifyStatus for testing
// ---------------------------------------------------------------------------

fn test_status_with_entries(entries: Vec<(&str, &str, &str)>) -> VerifyStatus {
    // Build a VerifyStatus with the given (name, soundness, method) triples
    // by serializing/deserializing JSON (VerifyStatus fields are private).
    let mut kernels = serde_json::Map::new();
    for (name, soundness, method) in entries {
        let entry = serde_json::json!({
            "status": "verified",
            "method": method,
            "input_bounds": {
                "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}],
                "constant_params": [],
                "input_shape": [4, 8],
                "input_range": [-1.0, 1.0]
            },
            "output_bounds": {
                "lower": -2.0,
                "upper": 2.0
            },
            "output_width": 4.0,
            "soundness_mode": soundness
        });
        kernels.insert(name.to_string(), entry);
    }
    let status_json = serde_json::json!({ "kernels": kernels });
    serde_json::from_value(status_json).expect("test status JSON is valid")
}

fn test_config() -> CertificateConfig {
    CertificateConfig {
        model_hash: "a".repeat(64),
        status_path: std::path::PathBuf::from("/nonexistent"),
        gamma_crown_rev: "532203c188bef9eb00fed44ef0ac6466f258af35".to_string(),
        include_stale: false,
    }
}

// ---------------------------------------------------------------------------
// Certificate generation tests
// ---------------------------------------------------------------------------

#[test]
fn test_generate_from_status_basic() {
    let status = test_status_with_entries(vec![
        ("kernel_a", "sound", "IBP"),
        ("kernel_b", "sound", "CROWN"),
        ("kernel_c", "heuristic", "IBP"),
    ]);
    let config = test_config();
    let cert = generate_from_status(&status, &config).expect("generate succeeds");

    assert_eq!(cert.schema_version, KOKORO_CERTIFICATE_VERSION);
    assert_eq!(cert.model_hash, "a".repeat(64));
    assert_eq!(cert.entries.len(), 3);
    assert_eq!(cert.summary.total_entries, 3);
    assert_eq!(cert.summary.active_entries, 3);
    assert_eq!(cert.summary.sound_count, 2);
    assert_eq!(cert.summary.heuristic_count, 1);
    assert_eq!(cert.junction_bounds.len(), 6);
    assert!(cert.content_hash.is_some());
}

#[test]
fn test_generate_empty_status() {
    let status = test_status_with_entries(vec![]);
    let config = test_config();
    let cert = generate_from_status(&status, &config).expect("generate succeeds");

    assert_eq!(cert.entries.len(), 0);
    assert_eq!(cert.summary.total_entries, 0);
    assert_eq!(cert.summary.sound_count, 0);
}

#[test]
fn test_generate_junction_bounds() {
    let status = test_status_with_entries(vec![("k", "sound", "IBP")]);
    let config = test_config();
    let cert = generate_from_status(&status, &config).expect("generate succeeds");

    let names: Vec<&str> = cert
        .junction_bounds
        .iter()
        .map(|j| j.name.as_str())
        .collect();
    assert!(names.contains(&"J2_F0"));
    assert!(names.contains(&"J2_ENERGY"));
    assert!(names.contains(&"J3_MAGNITUDE"));
    assert!(names.contains(&"J3B_PHASE"));
    assert!(names.contains(&"J4_BF16"));
    assert!(names.contains(&"J5_AUDIO"));

    // Verify specific bound values
    let j5 = cert
        .junction_bounds
        .iter()
        .find(|j| j.name == "J5_AUDIO")
        .unwrap();
    assert!((j5.lower - (-1.0)).abs() < f64::EPSILON);
    assert!((j5.upper - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_entry_soundness_and_method_formatting() {
    let status = test_status_with_entries(vec![
        ("ibp_sound", "sound", "IBP"),
        ("crown_heuristic", "heuristic", "CROWN"),
    ]);
    let config = test_config();
    let cert = generate_from_status(&status, &config).expect("generate succeeds");

    let ibp_entry = cert
        .entries
        .iter()
        .find(|e| e.kernel_name == "ibp_sound")
        .unwrap();
    assert_eq!(ibp_entry.soundness_mode, "sound");
    assert_eq!(ibp_entry.method, "IBP");
    assert_eq!(ibp_entry.status, "verified");

    let crown_entry = cert
        .entries
        .iter()
        .find(|e| e.kernel_name == "crown_heuristic")
        .unwrap();
    assert_eq!(crown_entry.soundness_mode, "heuristic");
    assert_eq!(crown_entry.method, "CROWN");
}

// ---------------------------------------------------------------------------
// Serialization round-trip tests
// ---------------------------------------------------------------------------

#[test]
fn test_certificate_json_roundtrip() {
    let status = test_status_with_entries(vec![
        ("kernel_a", "sound", "IBP"),
        ("kernel_b", "sound", "CROWN"),
    ]);
    let config = test_config();
    let cert = generate_from_status(&status, &config).expect("generate succeeds");

    let json = cert.to_json().expect("serialize succeeds");
    let loaded = KokoroCertificate::from_json(&json).expect("deserialize succeeds");

    assert_eq!(cert.schema_version, loaded.schema_version);
    assert_eq!(cert.model_hash, loaded.model_hash);
    assert_eq!(cert.gamma_crown_rev, loaded.gamma_crown_rev);
    assert_eq!(cert.entries.len(), loaded.entries.len());
    assert_eq!(cert.junction_bounds.len(), loaded.junction_bounds.len());
    assert_eq!(cert.summary, loaded.summary);
    assert_eq!(cert.content_hash, loaded.content_hash);
}

#[test]
fn test_certificate_file_roundtrip() {
    let status = test_status_with_entries(vec![("k1", "sound", "IBP")]);
    let config = test_config();
    let cert = generate_from_status(&status, &config).expect("generate succeeds");

    let dir = std::env::temp_dir().join("nn_kokoro_cert_test");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test_kokoro.proof.json");

    cert.save(&path).expect("save succeeds");
    let loaded = KokoroCertificate::load(&path).expect("load succeeds");

    assert_eq!(cert.schema_version, loaded.schema_version);
    assert_eq!(cert.model_hash, loaded.model_hash);
    assert_eq!(cert.entries.len(), loaded.entries.len());
    assert_eq!(cert.content_hash, loaded.content_hash);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

// ---------------------------------------------------------------------------
// Certificate verification tests
// ---------------------------------------------------------------------------

#[test]
fn test_verify_valid_certificate() {
    let status = test_status_with_entries(vec![
        ("kernel_a", "sound", "IBP"),
        ("kernel_b", "sound", "CROWN"),
    ]);
    let config = test_config();
    let cert = generate_from_status(&status, &config).expect("generate succeeds");

    let verdict = verify_certificate(&cert, &"a".repeat(64));
    assert!(verdict.is_valid());
    assert!(!verdict.has_errors());
}

#[test]
fn test_verify_model_hash_mismatch() {
    let status = test_status_with_entries(vec![("k", "sound", "IBP")]);
    let config = test_config();
    let cert = generate_from_status(&status, &config).expect("generate succeeds");

    let verdict = verify_certificate(&cert, &"b".repeat(64));
    assert!(!verdict.is_valid());
    assert!(verdict.has_errors());
    assert!(verdict
        .findings
        .iter()
        .any(|f| f.message.contains("model_hash mismatch")));
}

#[test]
fn test_verify_tampered_content_hash() {
    let status = test_status_with_entries(vec![("k", "sound", "IBP")]);
    let config = test_config();
    let mut cert = generate_from_status(&status, &config).expect("generate succeeds");

    // Tamper with an entry — content hash becomes stale
    cert.entries[0].output_width = 999.0;

    let verdict = verify_certificate(&cert, &"a".repeat(64));
    assert!(!verdict.is_valid());
    assert!(verdict
        .findings
        .iter()
        .any(|f| f.message.contains("content_hash mismatch")));
}

#[test]
fn test_verify_no_content_hash() {
    let status = test_status_with_entries(vec![("k", "sound", "IBP")]);
    let config = test_config();
    let mut cert = generate_from_status(&status, &config).expect("generate succeeds");
    cert.content_hash = None;

    let verdict = verify_certificate(&cert, &"a".repeat(64));
    // Should still be valid (content hash is optional), but with an info finding
    assert!(verdict.is_valid());
    assert!(verdict
        .findings
        .iter()
        .any(|f| f.message.contains("unsigned certificate")));
}

#[test]
fn test_verify_empty_certificate() {
    let status = test_status_with_entries(vec![]);
    let config = test_config();
    let cert = generate_from_status(&status, &config).expect("generate succeeds");

    let verdict = verify_certificate(&cert, &"a".repeat(64));
    assert!(!verdict.is_valid());
    assert!(verdict
        .findings
        .iter()
        .any(|f| f.message.contains("no verification entries")));
    assert!(verdict
        .findings
        .iter()
        .any(|f| f.message.contains("no sound entries")));
}

#[test]
fn test_verify_bad_schema_version() {
    let status = test_status_with_entries(vec![("k", "sound", "IBP")]);
    let config = test_config();
    let mut cert = generate_from_status(&status, &config).expect("generate succeeds");
    cert.schema_version = 99;
    // Recompute content hash after modification
    cert.content_hash = Some(compute_certificate_content_hash(&cert));

    let verdict = verify_certificate(&cert, &"a".repeat(64));
    assert!(!verdict.is_valid());
    assert!(verdict
        .findings
        .iter()
        .any(|f| f.message.contains("unsupported schema version")));
}

#[test]
fn test_verify_missing_junction_bound() {
    let status = test_status_with_entries(vec![("k", "sound", "IBP")]);
    let config = test_config();
    let mut cert = generate_from_status(&status, &config).expect("generate succeeds");
    cert.junction_bounds.retain(|j| j.name != "J5_AUDIO");
    cert.content_hash = Some(compute_certificate_content_hash(&cert));

    let verdict = verify_certificate(&cert, &"a".repeat(64));
    // Missing junction is a warning, not an error
    assert!(verdict.is_valid());
    assert!(verdict
        .findings
        .iter()
        .any(|f| f.message.contains("missing junction bound: J5_AUDIO")));
}

// ---------------------------------------------------------------------------
// Integration: load from real status file (if present)
// ---------------------------------------------------------------------------

#[test]
fn test_load_real_kokoro_status_file() {
    let status_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("nn_verify_status_kokoro.json");

    if !status_path.exists() {
        // Skip gracefully if status file not present (CI environments)
        return;
    }

    let mut config = test_config();
    config.status_path = status_path;

    let cert = generate_kokoro_certificate(&config).expect("generate from real file succeeds");
    assert!(
        cert.entries.len() > 10,
        "expected many entries, got {}",
        cert.entries.len()
    );
    assert!(cert.summary.sound_count > 0);
    assert_eq!(cert.junction_bounds.len(), 6);
    assert!(cert.content_hash.is_some());

    // Verify the generated certificate
    let verdict = verify_certificate(&cert, &"a".repeat(64));
    assert!(verdict.is_valid(), "verdict: {verdict}");
}

// ---------------------------------------------------------------------------
// Helpers: format functions
// ---------------------------------------------------------------------------

#[test]
fn test_format_method() {
    assert_eq!(format_method(PropMethod::Ibp), "IBP");
    assert_eq!(format_method(PropMethod::Crown), "CROWN");
    assert_eq!(format_method(PropMethod::AlphaCrown), "AlphaCrown");
    assert_eq!(format_method(PropMethod::BetaCrown), "BetaCrown");
    assert_eq!(format_method(PropMethod::Analytical), "Analytical");
}

#[test]
fn test_format_proof_strength() {
    assert_eq!(
        format_proof_strength(ProofStrength::SoundCrown),
        "sound_crown"
    );
    assert_eq!(format_proof_strength(ProofStrength::SoundIbp), "sound_ibp");
    assert_eq!(format_proof_strength(ProofStrength::Vacuous), "vacuous");
    assert_eq!(format_proof_strength(ProofStrength::Heuristic), "heuristic");
}

#[test]
fn test_default_gamma_crown_rev() {
    let rev = default_gamma_crown_rev();
    assert_eq!(rev.len(), 40, "git rev should be 40 hex chars");
    assert!(rev.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn test_kokoro_junction_bounds_count() {
    let bounds = kokoro_junction_bounds();
    assert_eq!(bounds.len(), 6);
}

#[test]
fn test_certificate_verdict_display() {
    let verdict = CertificateVerdict {
        valid: true,
        findings: vec![],
    };
    let s = format!("{verdict}");
    assert!(s.contains("VALID"));

    let verdict = CertificateVerdict {
        valid: false,
        findings: vec![CertificateFinding {
            severity: FindingSeverity::Error,
            message: "test error".to_string(),
        }],
    };
    let s = format!("{verdict}");
    assert!(s.contains("INVALID"));
    assert!(s.contains("[ERROR] test error"));
}
