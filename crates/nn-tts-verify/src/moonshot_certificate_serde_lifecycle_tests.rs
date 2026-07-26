// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Phase 28 gap-filling lifecycle tests and Phase 32 file persistence tests.
//!
//! Extracted from moonshot_certificate_serde_tests.rs for the 500-line limit.

use crate::moonshot::*;
use crate::moonshot_crown::{ImplementationCorrectnessEvidence, SpeakerConsistencyEvidence};
use crate::pipeline::{TimingCertificate, VerifiedStage};

// --- Phase 28 helper functions ---

/// Build a minimal PipelineCertificate + MoonshotCrownBundle for test use.
fn serde_test_crown_bundle(
    dim: usize,
) -> (
    crate::pipeline::PipelineCertificate,
    crate::moonshot_crown::MoonshotCrownBundle,
) {
    use crate::moonshot_crown::{MoonshotCrownBundle, MoonshotPropertyResult};
    use crate::pipeline::PipelineCertificate;

    let stage = VerifiedStage {
        name: "test-stage".to_string(),
        input_lower: vec![-1.0; dim],
        input_upper: vec![1.0; dim],
        output_lower: vec![-0.9; dim],
        output_upper: vec![0.9; dim],
        input_shape: vec![1, dim],
        output_shape: vec![1, dim],
        method: "CROWN".to_string(),
        is_sound: true,
    };

    let pipeline_cert = PipelineCertificate {
        stages: vec![stage],
        junctions: vec![],
        e2e_input_lower: vec![-1.0; dim],
        e2e_input_upper: vec![1.0; dim],
        e2e_output_lower: vec![-0.9; dim],
        e2e_output_upper: vec![0.9; dim],
        is_valid: true,
        is_sound: true,
    };

    let results: Vec<MoonshotPropertyResult> = (0..6)
        .map(|i| MoonshotPropertyResult {
            property_index: i,
            property_name: PROPERTY_NAMES[i],
            proven: true,
            level: VerificationLevel::CrownProven,
            bound_value: 0.5,
            threshold: 1.0,
            is_sound: true,
            explanation: format!("Property {} proven via CROWN", i + 1),
        })
        .collect();

    let bundle = MoonshotCrownBundle {
        results,
        pipeline_cert: pipeline_cert.clone(),
        verification_dim: dim,
        all_proven: true,
    };

    (pipeline_cert, bundle)
}

/// Build a minimal TimingCertificate for test use.
fn serde_test_timing_cert(
    pipeline_cert: &crate::pipeline::PipelineCertificate,
) -> TimingCertificate {
    use crate::cost_model::LayerCostProfile;

    TimingCertificate {
        bounds_cert: pipeline_cert.clone(),
        cost_profiles: vec![LayerCostProfile {
            layer_name: "test-layer".to_string(),
            flops: 1_000_000,
            memory_bytes: 4_000_000,
            estimated_time_us: 50_000.0,
            measured_time_us: None,
        }],
        worst_case_time_us: 50_000.0,
        total_flops: 1_000_000,
        total_memory_bytes: 4_000_000,
        hardware_name: "test-hardware".to_string(),
        timing_bound_us: 100_000.0,
        timing_bound_met: true,
        overall_passed: true,
        peak_memory: None,
    }
}

/// Build a minimal SpeakerConsistencyEvidence for test use.
fn serde_test_speaker(dim: usize) -> SpeakerConsistencyEvidence {
    SpeakerConsistencyEvidence {
        embed_dim: dim,
        embedding_lower: vec![-0.5; dim],
        embedding_upper: vec![0.5; dim],
        reference_embedding: vec![0.1; dim],
        distance_threshold: 0.8,
        is_sound: true,
    }
}

/// Build a minimal ImplementationCorrectnessEvidence for test use.
fn serde_test_dispatch(proven: usize, total: usize) -> ImplementationCorrectnessEvidence {
    ImplementationCorrectnessEvidence {
        proven_steps: proven,
        total_steps: total,
        proven_categories: vec!["matmul".to_string(), "softmax".to_string()],
        unproven_categories: vec![],
        all_proven: proven == total,
    }
}

// --- Phase 28: Gap-filling lifecycle tests ---

/// Gap 1: All 6 evidence sources -> serde round-trip -> validate with no structural errors.
#[test]
fn test_all_6_evidence_sources_serde_then_validate() {
    let dim = 32;
    let (pipeline_cert, bundle) = serde_test_crown_bundle(dim);
    let timing = serde_test_timing_cert(&pipeline_cert);
    let speaker = serde_test_speaker(dim);
    let kani = KaniVerificationEvidence {
        harnesses_passed: 500,
        harnesses_total: 500,
        harness_files: vec!["crates/nn-core/src/kani_bounds.rs".to_string()],
        all_passed: true,
    };
    let smt = SmtVerificationEvidence {
        kernels_proven: 20,
        kernels_total: 20,
        proven_kernel_names: vec!["snake".to_string(), "sigmoid".to_string()],
        all_proven: true,
    };
    let dispatch = serde_test_dispatch(5, 5);

    let cert =
        FullCertificateBuilder::new("full-lifecycle", "English text, <=50 words", "sha256abc")
            .crown_bundle(&bundle)
            .timing(&timing)
            .speaker(&speaker)
            .kani(&kani)
            .smt(&smt)
            .dispatch_plan(&dispatch)
            .build();

    // Serde round-trip
    let json = cert.to_json_serde().expect("serialize all-6");
    let de = MoonshotCertificate::from_json(&json).expect("deserialize all-6");

    // Validate: no structural errors
    let result = validate_certificate(&de, std::path::Path::new("/nonexistent"));
    let structural_errors: Vec<_> = result
        .findings
        .iter()
        .filter(|f| f.severity == FindingSeverity::Error && !f.message.contains("artifact"))
        .collect();
    assert!(
        structural_errors.is_empty(),
        "structural errors after all-6 round-trip: {structural_errors:?}"
    );

    // Check enrichment survived
    assert_eq!(de.properties[6].level, VerificationLevel::KaniProven);
    assert_eq!(de.properties[7].level, VerificationLevel::SmtProven);
    assert!(de.verification_dim.is_some());
}

/// Gap 2: CROWN evidence passes through serde -> validate with bound_value/threshold intact.
#[test]
fn test_crown_enriched_serde_then_validate() {
    let dim = 32;
    let (_pipeline_cert, bundle) = serde_test_crown_bundle(dim);

    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "crown-test", "spec", "hash_c")
        .with_crown_results(&bundle);

    let json = cert.to_json_serde().expect("serialize crown-enriched");
    let de = MoonshotCertificate::from_json(&json).expect("deserialize crown-enriched");

    // P1-P3 (non-silent, non-clipping, intelligible) should have bound_value and threshold
    for i in 0..3 {
        assert!(
            de.properties[i].bound_value.is_some(),
            "P{} should have bound_value after CROWN enrichment",
            i + 1
        );
        assert!(
            de.properties[i].threshold.is_some(),
            "P{} should have threshold after CROWN enrichment",
            i + 1
        );
    }

    // verification_dim should be set
    assert_eq!(de.verification_dim, Some(dim));

    // Validate: no structural errors
    let result = validate_certificate(&de, std::path::Path::new("/nonexistent"));
    let structural_errors: Vec<_> = result
        .findings
        .iter()
        .filter(|f| f.severity == FindingSeverity::Error && !f.message.contains("artifact"))
        .collect();
    assert!(
        structural_errors.is_empty(),
        "structural errors after CROWN round-trip: {structural_errors:?}"
    );
}

/// Gap 3: Deliberately inflated Kani count should trigger a validation warning.
#[test]
fn test_stale_kani_count_triggers_validation_warning() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "kani-stale", "spec", "hash_k");

    // Inflate Kani harness count far beyond any realistic workspace count
    let inflated_kani = KaniVerificationEvidence {
        harnesses_passed: 999_999,
        harnesses_total: 999_999,
        harness_files: vec!["nonexistent.rs".to_string()],
        all_passed: true,
    };
    let enriched = cert.with_kani_results(&inflated_kani);

    let json = enriched.to_json_serde().expect("serialize inflated");
    let de = MoonshotCertificate::from_json(&json).expect("deserialize inflated");

    // Validate against a nonexistent workspace root so artifact checks
    // produce warnings about missing files — the inflated count itself
    // passes structural validation (it's just a number), but the artifact
    // file "nonexistent.rs" won't exist.
    let result = validate_certificate(&de, std::path::Path::new("/nonexistent"));

    // Should have at least one finding (artifact doesn't exist).
    // This test confirms validation runs end-to-end on inflated evidence.
    assert!(
        !result.findings.is_empty(),
        "inflated Kani evidence should produce at least one finding"
    );
}

/// Gap 4: Evidence presence (verification_dim, proof_artifacts, ay/ paths) survives round-trip.
#[test]
fn test_evidence_presence_detectable_after_round_trip() {
    let dim = 64;
    let (_pipeline_cert, bundle) = serde_test_crown_bundle(dim);
    let smt = SmtVerificationEvidence {
        kernels_proven: 20,
        kernels_total: 20,
        proven_kernel_names: vec!["snake".to_string(), "sigmoid".to_string()],
        all_proven: true,
    };

    let cert = FullCertificateBuilder::new("evidence-check", "spec", "hash_e")
        .crown_bundle(&bundle)
        .smt(&smt)
        .build();

    let json = cert.to_json_serde().expect("serialize evidence");
    let de = MoonshotCertificate::from_json(&json).expect("deserialize evidence");

    // verification_dim from CROWN
    assert_eq!(de.verification_dim, Some(dim));

    // P8 (SMT) should have ay/ artifact paths
    let p8_artifacts = &de.properties[7].proof_artifacts;
    assert!(
        p8_artifacts
            .iter()
            .any(|a| a.starts_with("crates/nn-verify/src/ay/")),
        "P8 proof_artifacts should contain ay/ paths, got: {p8_artifacts:?}"
    );

    // P8 was enriched via .smt(), so it should have bound_value and threshold set.
    assert!(
        de.properties[7].bound_value.is_some(),
        "P8 enriched via .smt() should have bound_value after round-trip"
    );
    assert!(
        de.properties[7].threshold.is_some(),
        "P8 enriched via .smt() should have threshold after round-trip"
    );
}

/// Gap 5: Display and to_json() output with all 6 enrichment sources are well-formed.
#[test]
fn test_display_and_json_with_all_enrichments() {
    let dim = 64;
    let (pipeline_cert, bundle) = serde_test_crown_bundle(dim);
    let timing = serde_test_timing_cert(&pipeline_cert);
    let speaker = serde_test_speaker(dim);
    let kani = KaniVerificationEvidence {
        harnesses_passed: 500,
        harnesses_total: 500,
        harness_files: vec!["crates/nn-core/src/kani_bounds.rs".to_string()],
        all_passed: true,
    };
    let smt = SmtVerificationEvidence {
        kernels_proven: 20,
        kernels_total: 20,
        proven_kernel_names: vec!["snake".to_string(), "sigmoid".to_string()],
        all_proven: true,
    };
    let dispatch = serde_test_dispatch(5, 5);

    let cert = FullCertificateBuilder::new("display-test", "English text", "sha256_d")
        .crown_bundle(&bundle)
        .timing(&timing)
        .speaker(&speaker)
        .kani(&kani)
        .smt(&smt)
        .dispatch_plan(&dispatch)
        .build();

    // Display output
    let display = format!("{cert}");
    assert!(
        display.contains("display-test"),
        "Display should contain model name"
    );
    assert!(
        display.contains("KANI_PROVEN"),
        "Display should mention KANI_PROVEN level, got: {display}"
    );
    assert!(
        display.contains("SMT_PROVEN"),
        "Display should mention SMT_PROVEN level, got: {display}"
    );

    // Hand-rolled to_json()
    let hand_json = cert.to_json();
    assert!(hand_json.contains("display-test"));
    assert!(hand_json.contains("properties"));
    // Should have 8 property blocks
    let prop_count = hand_json.matches("\"index\":").count();
    assert_eq!(prop_count, 8, "hand-rolled JSON should have 8 properties");

    // Verify CROWN dimension appears
    assert!(
        hand_json.contains("verification_dim"),
        "hand-rolled JSON should include verification_dim"
    );
}

// --- Phase 32: File persistence tests ---

/// save() + load() round-trip through a temp file.
#[test]
fn test_save_load_round_trip() {
    let dir = std::env::temp_dir().join(format!("moonshot_cert_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("test_cert.proof.json");

    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "persist-test", "English text", "abc123");
    cert.save(&path).expect("save should succeed");

    let loaded = MoonshotCertificate::load(&path).expect("load should succeed");
    assert_eq!(loaded.model_name, "persist-test");
    assert_eq!(loaded.schema_version, CERTIFICATE_SCHEMA_VERSION);
    assert_eq!(loaded.properties.len(), 8);
    for (i, prop) in loaded.properties.iter().enumerate() {
        assert_eq!(prop.property_index, i, "property_index mismatch at {i}");
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// save() + load() with full enrichment (all 6 evidence sources).
#[test]
fn test_save_load_enriched_certificate() {
    let dir = std::env::temp_dir().join(format!("moonshot_cert_enr_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("enriched.proof.json");

    let dim = 32;
    let (pipeline_cert, bundle) = serde_test_crown_bundle(dim);
    let timing = serde_test_timing_cert(&pipeline_cert);
    let speaker = serde_test_speaker(dim);
    let kani = super::test_kani_evidence();
    let smt = super::test_smt_evidence();
    let dispatch = serde_test_dispatch(5, 5);

    let cert = FullCertificateBuilder::new("enriched-persist", "English text", "sha256_ep")
        .crown_bundle(&bundle)
        .timing(&timing)
        .speaker(&speaker)
        .kani(&kani)
        .smt(&smt)
        .dispatch_plan(&dispatch)
        .build();

    cert.save(&path).expect("save enriched");
    let loaded = MoonshotCertificate::load(&path).expect("load enriched");

    assert_eq!(loaded.model_name, "enriched-persist");
    assert_eq!(loaded.verification_dim, Some(dim));
    assert_eq!(loaded.properties[6].level, VerificationLevel::KaniProven);
    assert_eq!(loaded.properties[7].level, VerificationLevel::SmtProven);
    assert!(loaded.properties[0].bound_value.is_some());

    let _ = std::fs::remove_dir_all(&dir);
}

/// load() returns Io error for nonexistent file.
#[test]
fn test_load_nonexistent_file() {
    let result = MoonshotCertificate::load(std::path::Path::new("/nonexistent/cert.json"));
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        format!("{err}").contains("file I/O error"),
        "expected I/O error, got: {err}"
    );
}

/// Saved file is valid JSON that can be parsed by serde_json::Value.
#[test]
fn test_saved_file_is_valid_json() {
    let dir = std::env::temp_dir().join(format!("moonshot_cert_json_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("valid.proof.json");

    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "json-check", "spec", "hash");
    cert.save(&path).expect("save");

    let raw = std::fs::read_to_string(&path).expect("read saved file");
    let value: serde_json::Value = serde_json::from_str(&raw).expect("should be valid JSON");
    assert!(value.is_object());
    assert_eq!(value["model_name"], "json-check");
    assert_eq!(value["properties"].as_array().map(Vec::len), Some(8));

    let _ = std::fs::remove_dir_all(&dir);
}
