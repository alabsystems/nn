// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Kokoro CROWN deployment certificate generation.
//!
//! Part of #4254.

use super::*;

// ---------------------------------------------------------------------------
// Helpers: build test VerifyStatus instances
// ---------------------------------------------------------------------------

/// Build a VerifyStatus with the given (name, soundness, method) triples.
fn test_status_with_entries(entries: Vec<(&str, &str, &str)>) -> VerifyStatus {
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

/// Build a status matching the Kokoro pipeline stage keys for coverage testing.
fn test_status_with_pipeline_stages() -> VerifyStatus {
    let stages = kokoro_pipeline_stages();
    let mut kernels = serde_json::Map::new();
    for (i, stage) in stages.iter().enumerate() {
        let method = if i % 3 == 0 { "CROWN" } else { "IBP" };
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
            "soundness_mode": "sound"
        });
        kernels.insert(stage.status_key.to_string(), entry);
    }
    let status_json = serde_json::json!({ "kernels": kernels });
    serde_json::from_value(status_json).expect("test status JSON is valid")
}

fn test_deploy_config() -> DeploymentConfig {
    DeploymentConfig {
        model_hash: "a".repeat(64),
        status_path: std::path::PathBuf::from("/nonexistent"),
        gamma_crown_rev: "532203c188bef9eb00fed44ef0ac6466f258af35".to_string(),
        min_sound_ratio: 0.90,
        min_crown_stages: 3,
        max_vacuous: 0,
        max_gaps: 0,
    }
}

// ---------------------------------------------------------------------------
// Stage coverage tests
// ---------------------------------------------------------------------------

#[test]
fn test_stage_coverage_with_pipeline_stages() {
    let status = test_status_with_pipeline_stages();
    let coverage = compute_stage_coverage(&status);

    // Should have one entry per pipeline stage
    let stages = kokoro_pipeline_stages();
    assert_eq!(coverage.len(), stages.len());

    // All stages should have primary bounds
    for cov in &coverage {
        assert!(
            cov.has_primary,
            "stage '{}' should have primary bounds",
            cov.stage_name
        );
    }

    // At least some should have CROWN
    let crown_count = coverage.iter().filter(|c| c.has_crown).count();
    assert!(crown_count > 0, "should have at least one CROWN stage");
}

#[test]
fn test_stage_coverage_empty_status() {
    let status = test_status_with_entries(vec![]);
    let coverage = compute_stage_coverage(&status);

    // All stages should be uncovered
    for cov in &coverage {
        assert!(
            !cov.has_primary,
            "stage '{}' should not have primary bounds",
            cov.stage_name
        );
        assert!(
            !cov.has_crown,
            "stage '{}' should not have CROWN",
            cov.stage_name
        );
    }
}

#[test]
fn test_stage_coverage_crown_suffix_detection() {
    // Create a status with both primary and _crown suffix entries
    let mut kernels = serde_json::Map::new();
    let entry = serde_json::json!({
        "status": "verified",
        "method": "IBP",
        "input_bounds": {
            "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}],
            "constant_params": [],
            "input_shape": [4, 8],
            "input_range": [-1.0, 1.0]
        },
        "output_bounds": {"lower": -2.0, "upper": 2.0},
        "output_width": 4.0,
        "soundness_mode": "sound"
    });
    let crown_entry = serde_json::json!({
        "status": "verified",
        "method": "CROWN",
        "input_bounds": {
            "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}],
            "constant_params": [],
            "input_shape": [4, 8],
            "input_range": [-1.0, 1.0]
        },
        "output_bounds": {"lower": -1.5, "upper": 1.5},
        "output_width": 3.0,
        "soundness_mode": "sound"
    });
    // Use a real pipeline stage key
    kernels.insert("kokoro_production_text_encoder".to_string(), entry);
    kernels.insert(
        "kokoro_production_text_encoder_crown".to_string(),
        crown_entry,
    );
    let status_json = serde_json::json!({ "kernels": kernels });
    let status: VerifyStatus =
        serde_json::from_value(status_json).expect("test status JSON is valid");

    let coverage = compute_stage_coverage(&status);
    let text_enc = coverage
        .iter()
        .find(|c| c.status_key == "kokoro_production_text_encoder")
        .expect("should find text encoder stage");

    assert!(text_enc.has_primary);
    assert!(text_enc.has_crown, "should detect _crown suffix entry");
}

// ---------------------------------------------------------------------------
// Deployment gate tests
// ---------------------------------------------------------------------------

#[test]
fn test_gate_all_pass() {
    let status = test_status_with_pipeline_stages();
    let coverage = compute_stage_coverage(&status);
    let config = test_deploy_config();
    let gate = evaluate_gate(&status, &coverage, &config);

    assert!(gate.is_deployable(), "gate should pass: {:?}", gate.details);
    assert!(gate.sound_ratio_pass);
    assert!(gate.crown_stage_pass);
    assert!(gate.vacuous_pass);
    assert!(gate.gap_pass);
    assert_eq!(gate.vacuous_count, 0);
    assert_eq!(gate.gap_count, 0);
}

#[test]
fn test_gate_insufficient_sound_ratio() {
    // 1 sound + 9 heuristic = 10% sound ratio
    let mut entries = Vec::new();
    entries.push(("kokoro_production_generator", "sound", "CROWN"));
    for i in 0..9 {
        // Leak strings - use a static array of names
        let names = [
            "kokoro_h1",
            "kokoro_h2",
            "kokoro_h3",
            "kokoro_h4",
            "kokoro_h5",
            "kokoro_h6",
            "kokoro_h7",
            "kokoro_h8",
            "kokoro_h9",
        ];
        entries.push((names[i], "heuristic", "IBP"));
    }
    let status = test_status_with_entries(entries);
    let coverage = compute_stage_coverage(&status);
    let config = test_deploy_config();
    let gate = evaluate_gate(&status, &coverage, &config);

    assert!(!gate.sound_ratio_pass);
    assert!(!gate.is_deployable());
    assert!(gate.sound_ratio < 0.90);
}

#[test]
fn test_gate_too_many_gaps() {
    // Empty status = all stages are gaps
    let status = test_status_with_entries(vec![]);
    let coverage = compute_stage_coverage(&status);
    let config = test_deploy_config();
    let gate = evaluate_gate(&status, &coverage, &config);

    assert!(!gate.gap_pass);
    assert!(!gate.is_deployable());
    assert!(gate.gap_count > 0);
}

#[test]
fn test_gate_details_contain_pass_fail() {
    let status = test_status_with_pipeline_stages();
    let coverage = compute_stage_coverage(&status);
    let config = test_deploy_config();
    let gate = evaluate_gate(&status, &coverage, &config);

    assert!(!gate.details.is_empty());
    // Each detail should contain either PASS or FAIL
    for detail in &gate.details {
        assert!(
            detail.contains("PASS") || detail.contains("FAIL"),
            "detail should contain PASS or FAIL: {detail}"
        );
    }
}

// ---------------------------------------------------------------------------
// End-to-end certificate generation tests
// ---------------------------------------------------------------------------

#[test]
fn test_generate_deployment_from_status_basic() {
    let status = test_status_with_pipeline_stages();
    let config = test_deploy_config();
    let cert = generate_deployment_from_status(&status, &config).expect("generation succeeds");

    // Base certificate populated
    assert!(!cert.base_certificate.entries.is_empty());
    assert_eq!(cert.base_certificate.junction_bounds.len(), 6);

    // Stage coverage populated
    assert!(!cert.stage_coverage.is_empty());

    // Gate evaluated
    assert!(cert.gate.is_deployable());

    // Content hash present
    assert!(cert.content_hash.is_some());
}

#[test]
fn test_deployment_certificate_json_roundtrip() {
    let status = test_status_with_pipeline_stages();
    let config = test_deploy_config();
    let cert = generate_deployment_from_status(&status, &config).expect("generation succeeds");

    let json = cert.to_json().expect("serialize succeeds");
    let loaded = KokoroCrownCertificate::from_json(&json).expect("deserialize succeeds");

    assert_eq!(cert.gate.deployable, loaded.gate.deployable);
    assert_eq!(cert.stage_coverage.len(), loaded.stage_coverage.len());
    assert_eq!(
        cert.base_certificate.entries.len(),
        loaded.base_certificate.entries.len()
    );
    assert_eq!(cert.content_hash, loaded.content_hash);
}

#[test]
fn test_deployment_certificate_file_roundtrip() {
    let status = test_status_with_pipeline_stages();
    let config = test_deploy_config();
    let cert = generate_deployment_from_status(&status, &config).expect("generation succeeds");

    let dir = std::env::temp_dir().join("nn_kokoro_crown_cert_test");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test_deploy.proof.json");

    cert.save(&path).expect("save succeeds");
    let loaded = KokoroCrownCertificate::load(&path).expect("load succeeds");

    assert_eq!(cert.gate.deployable, loaded.gate.deployable);
    assert_eq!(cert.content_hash, loaded.content_hash);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn test_deployment_certificate_integrity() {
    let status = test_status_with_pipeline_stages();
    let config = test_deploy_config();
    let cert = generate_deployment_from_status(&status, &config).expect("generation succeeds");

    assert!(cert.verify_integrity(), "integrity should pass");

    // Tamper with the gate
    let mut tampered = cert;
    tampered.gate.vacuous_count = 999;
    assert!(
        !tampered.verify_integrity(),
        "integrity should fail after tampering"
    );
}

#[test]
fn test_crown_covered_count() {
    let status = test_status_with_pipeline_stages();
    let config = test_deploy_config();
    let cert = generate_deployment_from_status(&status, &config).expect("generation succeeds");

    let count = cert.crown_covered_count();
    assert!(count > 0, "should have some CROWN-covered stages");
}

#[test]
fn test_uncovered_stages_empty_when_full_coverage() {
    let status = test_status_with_pipeline_stages();
    let config = test_deploy_config();
    let cert = generate_deployment_from_status(&status, &config).expect("generation succeeds");

    let uncovered = cert.uncovered_stages();
    assert!(
        uncovered.is_empty(),
        "should have no gaps when all stages are covered, got: {uncovered:?}"
    );
}

#[test]
fn test_uncovered_stages_when_partial_coverage() {
    let status = test_status_with_entries(vec![("kokoro_production_generator", "sound", "IBP")]);
    let config = test_deploy_config();
    let cert = generate_deployment_from_status(&status, &config).expect("generation succeeds");

    let uncovered = cert.uncovered_stages();
    assert!(
        !uncovered.is_empty(),
        "should have gaps with partial coverage"
    );
}

// ---------------------------------------------------------------------------
// Per-entry constructive proof tests (#4254)
// ---------------------------------------------------------------------------

#[test]
fn test_entry_proofs_populated_from_pipeline_stages() {
    let status = test_status_with_pipeline_stages();
    let config = test_deploy_config();
    let cert = generate_deployment_from_status(&status, &config).expect("generation succeeds");

    // entry_proofs should be populated (not None)
    assert!(
        cert.entry_proofs.is_some(),
        "entry_proofs should be populated for non-empty status"
    );
    let proofs = cert.entry_proofs.as_ref().unwrap();
    assert!(!proofs.is_empty(), "should have at least one entry proof");

    // Entry proof count method should agree
    assert_eq!(cert.entry_proof_count(), proofs.len());
}

#[test]
fn test_entry_proofs_all_valid() {
    let status = test_status_with_pipeline_stages();
    let config = test_deploy_config();
    let cert = generate_deployment_from_status(&status, &config).expect("generation succeeds");

    assert!(
        cert.all_proofs_valid(),
        "all entry proofs should pass structural validation"
    );
}

#[test]
fn test_entry_proofs_crown_and_sound_counts() {
    let status = test_status_with_pipeline_stages();
    let config = test_deploy_config();
    let cert = generate_deployment_from_status(&status, &config).expect("generation succeeds");

    let total = cert.entry_proof_count();
    let crown = cert.crown_proof_count();
    let sound = cert.sound_proof_count();

    // All entries in test_status_with_pipeline_stages are sound
    assert_eq!(sound, total, "all test entries are sound");

    // At least some should be CROWN (every 3rd stage uses CROWN method)
    assert!(crown > 0, "should have some CROWN proofs");
    assert!(crown <= total, "CROWN count <= total");
}

#[test]
fn test_proof_summary() {
    let status = test_status_with_pipeline_stages();
    let config = test_deploy_config();
    let cert = generate_deployment_from_status(&status, &config).expect("generation succeeds");

    let summary = cert.proof_summary();
    assert_eq!(summary.total_proofs, cert.entry_proof_count());
    assert_eq!(summary.crown_proofs, cert.crown_proof_count());
    assert_eq!(summary.sound_proofs, cert.sound_proof_count());
    assert_eq!(
        summary.ibp_proofs,
        summary.total_proofs - summary.crown_proofs
    );
    assert!(
        summary.machine_checkable > 0,
        "should have machine-checkable proofs"
    );
}

#[test]
fn test_entry_proof_for_lookup() {
    let status = test_status_with_pipeline_stages();
    let config = test_deploy_config();
    let cert = generate_deployment_from_status(&status, &config).expect("generation succeeds");

    // Look up a known pipeline stage key
    let stages = kokoro_pipeline_stages();
    let key = stages[0].status_key;
    let proof = cert.entry_proof_for(key);
    assert!(proof.is_some(), "should find entry proof for '{key}'");
    let ep = proof.unwrap();
    assert_eq!(ep.kernel_name, key);
    assert!(ep.is_sound, "test entries are sound");

    // Non-existent key returns None
    assert!(cert.entry_proof_for("nonexistent_kernel_xyz").is_none());
}

#[test]
fn test_entry_proofs_none_for_empty_status() {
    let status = test_status_with_entries(vec![]);
    let config = test_deploy_config();
    let cert = generate_deployment_from_status(&status, &config).expect("generation succeeds");

    // Empty status → no entry proofs → None (not Some([]))
    assert!(
        cert.entry_proofs.is_none(),
        "entry_proofs should be None for empty status"
    );
    assert_eq!(cert.entry_proof_count(), 0);
    assert_eq!(cert.crown_proof_count(), 0);
    assert_eq!(cert.sound_proof_count(), 0);
    assert!(cert.all_proofs_valid(), "vacuously valid for no proofs");
}

#[test]
fn test_entry_proofs_json_roundtrip() {
    let status = test_status_with_pipeline_stages();
    let config = test_deploy_config();
    let cert = generate_deployment_from_status(&status, &config).expect("generation succeeds");

    let json = cert.to_json().expect("serialize succeeds");
    let loaded = KokoroCrownCertificate::from_json(&json).expect("deserialize succeeds");

    assert_eq!(
        cert.entry_proof_count(),
        loaded.entry_proof_count(),
        "entry proof count should survive roundtrip"
    );
    assert_eq!(
        cert.crown_proof_count(),
        loaded.crown_proof_count(),
        "crown proof count should survive roundtrip"
    );
    // Each entry proof should match
    if let (Some(orig), Some(loaded_proofs)) = (&cert.entry_proofs, &loaded.entry_proofs) {
        assert_eq!(orig.len(), loaded_proofs.len());
        for (o, l) in orig.iter().zip(loaded_proofs.iter()) {
            assert_eq!(o.kernel_name, l.kernel_name);
            assert_eq!(o.is_crown, l.is_crown);
            assert_eq!(o.is_sound, l.is_sound);
        }
    }
}

#[test]
fn test_entry_proofs_backward_compat_no_proofs_field() {
    // Simulate a certificate from before #4254 (no entry_proofs in JSON)
    let status = test_status_with_pipeline_stages();
    let config = test_deploy_config();
    let cert = generate_deployment_from_status(&status, &config).expect("generation succeeds");

    // Strip entry_proofs from the JSON by parsing → removing → re-serializing
    let json = cert.to_json().expect("serialize");
    let mut val: serde_json::Value = serde_json::from_str(&json).expect("parse");
    val.as_object_mut().unwrap().remove("entry_proofs");
    let modified_json = serde_json::to_string_pretty(&val).expect("re-serialize");

    let loaded = KokoroCrownCertificate::from_json(&modified_json)
        .expect("should deserialize without entry_proofs");
    assert!(
        loaded.entry_proofs.is_none(),
        "missing field should default to None"
    );
    assert_eq!(loaded.entry_proof_count(), 0);
    assert!(loaded.all_proofs_valid());
}

#[test]
fn test_entry_proof_for_specific_kokoro_stage() {
    // Test that a specific Kokoro stage (text_encoder) produces a valid
    // constructive proof — satisfies the "at least one Kokoro stage" requirement.
    let status = test_status_with_entries(vec![
        ("kokoro_production_text_encoder", "sound", "CROWN"),
        ("kokoro_production_generator", "sound", "IBP"),
    ]);
    let config = test_deploy_config();
    let cert = generate_deployment_from_status(&status, &config).expect("generation succeeds");

    // Text encoder should have a CROWN proof
    let text_enc = cert
        .entry_proof_for("kokoro_production_text_encoder")
        .expect("should find text encoder proof");
    assert!(text_enc.is_crown, "text encoder uses CROWN");
    assert!(text_enc.is_sound, "text encoder is sound");
    assert!(
        text_enc.proof.validate().is_ok(),
        "text encoder proof should validate"
    );

    // Generator should have an IBP proof
    let gen_entry = cert
        .entry_proof_for("kokoro_production_generator")
        .expect("should find generator proof");
    assert!(!gen_entry.is_crown, "generator uses IBP (not CROWN)");
    assert!(gen_entry.is_sound, "generator is sound");
    assert!(
        gen_entry.proof.validate().is_ok(),
        "generator proof should validate"
    );
}

// ---------------------------------------------------------------------------
// Integration: load from real status file (if present)
// ---------------------------------------------------------------------------

#[test]
fn test_load_real_kokoro_deployment_certificate() {
    let status_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("nn_verify_status_kokoro.json");

    if !status_path.exists() {
        return;
    }

    let status = VerifyStatus::load(&status_path).expect("load status succeeds");
    let config = DeploymentConfig {
        model_hash: "a".repeat(64),
        status_path,
        gamma_crown_rev: default_gamma_crown_rev().to_string(),
        min_sound_ratio: 0.50, // Relaxed for test — real threshold is 0.90
        min_crown_stages: 1,
        max_vacuous: 5,
        max_gaps: 10, // Relaxed — not all stages may be in the real file
    };

    let cert = generate_deployment_from_status(&status, &config).expect("generation succeeds");

    // Real file should have many entries
    assert!(
        cert.base_certificate.entries.len() > 10,
        "expected many entries, got {}",
        cert.base_certificate.entries.len()
    );

    // Should have some CROWN coverage
    let crown_count = cert.crown_covered_count();
    eprintln!(
        "Real status: {} entries, {} CROWN stages, gate={}",
        cert.base_certificate.entries.len(),
        crown_count,
        cert.gate.deployable
    );

    // Content hash should be present
    assert!(cert.content_hash.is_some());

    // Integrity should pass
    assert!(cert.verify_integrity());

    // Gate details should be non-empty
    assert!(!cert.gate.details.is_empty());
    for detail in &cert.gate.details {
        eprintln!("  {detail}");
    }
}
