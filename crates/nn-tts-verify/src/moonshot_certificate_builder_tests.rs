// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`FullCertificateBuilder`] and convenience functions.

use super::*;
use crate::moonshot::{MoonshotStatus, VerificationLevel};
use crate::moonshot_crown::{
    ImplementationCorrectnessEvidence, MoonshotCrownBundle, SpeakerConsistencyEvidence,
};
use crate::pipeline::{PipelineCertificate, TimingCertificate, VerifiedStage};

// --- Test helpers ----------------------------------------------------------

/// Build a minimal CROWN bundle for testing.
fn test_crown_bundle(dim: usize) -> (PipelineCertificate, MoonshotCrownBundle) {
    let stages = vec![
        VerifiedStage {
            name: "encoder".to_string(),
            input_lower: vec![-1.0; dim],
            input_upper: vec![1.0; dim],
            output_lower: vec![-0.8; dim],
            output_upper: vec![0.8; dim],
            input_shape: vec![1, dim],
            output_shape: vec![1, dim],
            method: "CROWN".to_string(),
            is_sound: true,
        },
        VerifiedStage {
            name: "decoder".to_string(),
            input_lower: vec![-1.0; dim],
            input_upper: vec![1.0; dim],
            output_lower: vec![-0.3; dim],
            output_upper: vec![0.3; dim],
            input_shape: vec![1, dim],
            output_shape: vec![1, dim],
            method: "CROWN".to_string(),
            is_sound: true,
        },
    ];

    let pipeline_cert = crate::pipeline::verify_pipeline(&stages).expect("pipeline must compose");
    let bundle = crate::moonshot_crown::verify_properties_from_pipeline(&pipeline_cert, dim);
    (pipeline_cert, bundle)
}

/// Build a timing certificate for testing.
fn test_timing_cert(pipeline_cert: &PipelineCertificate, dim: usize) -> TimingCertificate {
    TimingCertificate {
        bounds_cert: pipeline_cert.clone(),
        cost_profiles: vec![crate::cost_model::LayerCostProfile {
            layer_name: "layer".to_string(),
            flops: 1_000_000,
            memory_bytes: 4 * dim as u64,
            estimated_time_us: 30_000.0,
            measured_time_us: None,
        }],
        worst_case_time_us: 30_000.0,
        total_flops: 1_000_000,
        total_memory_bytes: 4 * dim as u64,
        hardware_name: "M4 Max".to_string(),
        timing_bound_us: 100_000.0,
        timing_bound_met: true,
        overall_passed: true,
        peak_memory: None,
    }
}

fn test_speaker_evidence(dim: usize) -> SpeakerConsistencyEvidence {
    let norm_val = 1.0 / (dim as f64).sqrt();
    SpeakerConsistencyEvidence {
        embed_dim: dim,
        embedding_lower: vec![norm_val - 0.01; dim],
        embedding_upper: vec![norm_val + 0.01; dim],
        reference_embedding: vec![norm_val; dim],
        distance_threshold: 0.3,
        is_sound: true,
    }
}

fn test_kani_evidence(passed: usize, total: usize) -> KaniVerificationEvidence {
    KaniVerificationEvidence {
        harnesses_passed: passed,
        harnesses_total: total,
        harness_files: vec![
            "crates/nn-core/src/kani_bounds.rs".to_string(),
            "crates/nn-autodiff/src/kani_backward_proofs.rs".to_string(),
        ],
        all_passed: passed == total && total > 0,
    }
}

fn test_smt_evidence(proven: usize, total: usize) -> SmtVerificationEvidence {
    SmtVerificationEvidence {
        kernels_proven: proven,
        kernels_total: total,
        proven_kernel_names: vec!["snake".to_string(), "sigmoid".to_string()],
        all_proven: proven == total && total > 0,
    }
}

fn test_dispatch_evidence(proven: usize, total: usize) -> ImplementationCorrectnessEvidence {
    ImplementationCorrectnessEvidence {
        total_steps: total,
        proven_steps: proven,
        proven_categories: vec!["sigmoid".to_string(), "relu".to_string()],
        unproven_categories: if proven < total {
            vec!["linear".to_string()]
        } else {
            vec![]
        },
        all_proven: proven == total && total > 0,
    }
}

// --- Builder tests ---------------------------------------------------------

#[test]
fn test_builder_with_all_evidence_all_proven() {
    let dim = 64;
    let (pipeline_cert, bundle) = test_crown_bundle(dim);
    let timing = test_timing_cert(&pipeline_cert, dim);
    let speaker = test_speaker_evidence(dim);
    let kani = test_kani_evidence(500, 500);
    let smt = test_smt_evidence(14, 14);
    let dispatch = test_dispatch_evidence(5, 5);

    let cert = FullCertificateBuilder::new("test-model", "test input", "sha256hash")
        .crown_bundle(&bundle)
        .timing(&timing)
        .speaker(&speaker)
        .kani(&kani)
        .smt(&smt)
        .dispatch_plan(&dispatch)
        .build();

    assert_eq!(cert.model_name, "test-model");
    assert_eq!(cert.input_specification, "test input");
    assert_eq!(cert.source_hash, "sha256hash");
    assert_eq!(cert.properties.len(), 8);

    // P7 should be KaniProven.
    assert_eq!(
        cert.properties[6].level,
        VerificationLevel::KaniProven,
        "P7 must be KaniProven"
    );

    // P8 should be SmtProven (dispatch only upgrades, SMT is higher).
    assert_eq!(
        cert.properties[7].level,
        VerificationLevel::SmtProven,
        "P8 must be SmtProven"
    );

    // All properties should be at least Empirical (evidence was provided for all).
    for (i, prop) in cert.properties.iter().enumerate() {
        assert!(
            prop.level >= VerificationLevel::Empirical,
            "P{} should be at least Empirical with all evidence, got {:?}",
            i + 1,
            prop.level
        );
    }
}

#[test]
fn test_builder_crown_only() {
    let dim = 64;
    let (_pipeline_cert, bundle) = test_crown_bundle(dim);

    let cert = FullCertificateBuilder::new("model", "input", "hash")
        .crown_bundle(&bundle)
        .build();

    // CROWN bundle sets P1-P3, P6; P7/P8 retain base levels from artifact registry.
    // The artifact registry may already set P7/P8 to KaniProven/SmtProven, so we
    // just verify the builder didn't change them from the from_repo() baseline.
    let baseline = FullCertificateBuilder::new("model", "input", "hash").build();
    assert_eq!(
        cert.properties[6].level, baseline.properties[6].level,
        "P7 should match baseline (no Kani evidence provided)"
    );
    assert_eq!(
        cert.properties[7].level, baseline.properties[7].level,
        "P8 should match baseline (no SMT evidence provided)"
    );
    assert_eq!(cert.verification_dim, Some(dim));
}

#[test]
fn test_builder_kani_only() {
    let kani = test_kani_evidence(500, 500);

    let cert = FullCertificateBuilder::new("model", "input", "hash")
        .kani(&kani)
        .build();

    assert_eq!(
        cert.properties[6].level,
        VerificationLevel::KaniProven,
        "P7 with all Kani passed should be KaniProven"
    );
    // No CROWN bundle → P1-P6 at base level.
    assert_eq!(cert.verification_dim, None);
}

#[test]
fn test_builder_smt_only() {
    let smt = test_smt_evidence(14, 14);

    let cert = FullCertificateBuilder::new("model", "input", "hash")
        .smt(&smt)
        .build();

    assert_eq!(
        cert.properties[7].level,
        VerificationLevel::SmtProven,
        "P8 with all kernels proven should be SmtProven"
    );
}

#[test]
fn test_builder_no_evidence() {
    let cert = FullCertificateBuilder::new("empty-model", "input", "hash").build();

    // The builder with no evidence produces a certificate with only the base
    // levels from MoonshotStatus::from_repo(). The artifact registry may already
    // set some properties to high levels, so we just verify metadata is correct.
    assert_eq!(cert.model_name, "empty-model");
    assert_eq!(cert.verification_dim, None);
    assert_eq!(cert.properties.len(), 8);
}

#[test]
fn test_builder_dispatch_plan_upgrades_p8() {
    let dispatch = test_dispatch_evidence(5, 5);

    let cert = FullCertificateBuilder::new("model", "input", "hash")
        .dispatch_plan(&dispatch)
        .build();

    // Dispatch plan with all proven → SmtProven for P8 (upgrade from base).
    assert!(
        cert.properties[7].level >= VerificationLevel::CrownPartial,
        "dispatch all-proven should upgrade P8: got {:?}",
        cert.properties[7].level
    );
}

#[test]
fn test_builder_dispatch_plan_does_not_downgrade_smt() {
    let smt = test_smt_evidence(14, 14);
    // Partial dispatch evidence (3/5 proven → CrownPartial).
    let dispatch = test_dispatch_evidence(3, 10);

    let cert = FullCertificateBuilder::new("model", "input", "hash")
        .smt(&smt)
        .dispatch_plan(&dispatch)
        .build();

    // SMT all-proven → SmtProven, dispatch partial should NOT downgrade.
    assert_eq!(
        cert.properties[7].level,
        VerificationLevel::SmtProven,
        "dispatch must not downgrade P8 below SMT level"
    );
}

// --- Evidence summary tests ------------------------------------------------

#[test]
fn test_evidence_summary_all_sources() {
    let dim = 64;
    let (pipeline_cert, bundle) = test_crown_bundle(dim);
    let timing = test_timing_cert(&pipeline_cert, dim);
    let speaker = test_speaker_evidence(dim);
    let kani = test_kani_evidence(500, 500);
    let smt = test_smt_evidence(14, 14);
    let dispatch = test_dispatch_evidence(5, 5);

    let summary = FullCertificateBuilder::new("model", "input", "hash")
        .crown_bundle(&bundle)
        .timing(&timing)
        .speaker(&speaker)
        .kani(&kani)
        .smt(&smt)
        .dispatch_plan(&dispatch)
        .evidence_summary();

    assert!(summary.has_crown);
    assert!(summary.has_timing);
    assert!(summary.has_speaker);
    assert!(summary.has_kani);
    assert!(summary.has_smt);
    assert!(summary.has_dispatch_plan);
    assert_eq!(summary.total_sources, 6);
}

#[test]
fn test_evidence_summary_partial() {
    let kani = test_kani_evidence(500, 500);

    let summary = FullCertificateBuilder::new("model", "input", "hash")
        .kani(&kani)
        .evidence_summary();

    assert!(!summary.has_crown);
    assert!(!summary.has_timing);
    assert!(!summary.has_speaker);
    assert!(summary.has_kani);
    assert!(!summary.has_smt);
    assert!(!summary.has_dispatch_plan);
    assert_eq!(summary.total_sources, 1);
}

#[test]
fn test_evidence_summary_empty() {
    let summary = FullCertificateBuilder::new("model", "input", "hash").evidence_summary();

    assert_eq!(summary.total_sources, 0);
    assert!(!summary.has_crown);
    assert!(!summary.has_kani);
    assert!(!summary.has_smt);
}

#[test]
fn test_evidence_summary_display() {
    let kani = test_kani_evidence(500, 500);
    let smt = test_smt_evidence(14, 14);

    let summary = FullCertificateBuilder::new("model", "input", "hash")
        .kani(&kani)
        .smt(&smt)
        .evidence_summary();

    let s = format!("{summary}");
    assert!(s.contains("2/7 sources"));
    assert!(s.contains("Kani="));
    assert!(s.contains("SMT="));
}

// --- Convenience function tests --------------------------------------------

#[test]
fn test_build_full_certificate_convenience() {
    let dim = 64;
    let (_pipeline_cert, bundle) = test_crown_bundle(dim);
    let kani = test_kani_evidence(500, 500);
    let smt = test_smt_evidence(14, 14);

    let cert = build_full_certificate("model", "input", "hash", &bundle, &kani, &smt, None);

    assert_eq!(cert.model_name, "model");
    assert_eq!(cert.properties.len(), 8);
    assert_eq!(cert.properties[6].level, VerificationLevel::KaniProven,);
    assert_eq!(cert.properties[7].level, VerificationLevel::SmtProven,);
}

#[test]
fn test_build_full_certificate_with_dispatch() {
    let dim = 64;
    let (_pipeline_cert, bundle) = test_crown_bundle(dim);
    let kani = test_kani_evidence(500, 500);
    let smt = test_smt_evidence(14, 14);
    let dispatch = test_dispatch_evidence(5, 5);

    let cert = build_full_certificate(
        "model",
        "input",
        "hash",
        &bundle,
        &kani,
        &smt,
        Some(&dispatch),
    );

    // SMT all-proven + dispatch all-proven → SmtProven holds.
    assert_eq!(cert.properties[7].level, VerificationLevel::SmtProven,);
}

#[test]
fn test_build_full_certificate_with_all_evidence_convenience() {
    let dim = 64;
    let (pipeline_cert, bundle) = test_crown_bundle(dim);
    let timing = test_timing_cert(&pipeline_cert, dim);
    let speaker = test_speaker_evidence(dim);
    let kani = test_kani_evidence(500, 500);
    let smt = test_smt_evidence(14, 14);
    let dispatch = test_dispatch_evidence(5, 5);

    let cert = build_full_certificate_with_all_evidence(
        "full-model",
        "English text",
        "sha256",
        &bundle,
        &timing,
        &speaker,
        &kani,
        &smt,
        &dispatch,
    );

    assert_eq!(cert.model_name, "full-model");
    // all_proven depends on whether synthetic CROWN bounds produce CrownProven
    // for P1-P6; P7 and P8 should be KaniProven and SmtProven respectively.
    assert_eq!(cert.properties[6].level, VerificationLevel::KaniProven,);
    assert_eq!(cert.properties[7].level, VerificationLevel::SmtProven,);
}

// --- Builder matches manual chaining test ----------------------------------

#[test]
fn test_builder_matches_manual_chaining() {
    let dim = 64;
    let (_pipeline_cert, bundle) = test_crown_bundle(dim);
    let kani = test_kani_evidence(500, 500);
    let smt = test_smt_evidence(14, 14);

    // Manual chaining (old pattern).
    let status = MoonshotStatus::from_repo();
    let manual = MoonshotCertificate::from_status(&status, "model", "input", "hash");
    let manual = manual.with_crown_results(&bundle);
    let manual = manual.with_kani_results(&kani);
    let manual = manual.with_smt_results(&smt);

    // Builder (new pattern).
    let built = FullCertificateBuilder::new("model", "input", "hash")
        .crown_bundle(&bundle)
        .kani(&kani)
        .smt(&smt)
        .build();

    // Same property levels.
    for i in 0..8 {
        assert_eq!(
            manual.properties[i].level,
            built.properties[i].level,
            "P{} level mismatch: manual={:?}, built={:?}",
            i + 1,
            manual.properties[i].level,
            built.properties[i].level,
        );
    }
    assert_eq!(manual.all_proven, built.all_proven);
    assert_eq!(manual.all_at_least_partial, built.all_at_least_partial);
    assert_eq!(manual.verification_dim, built.verification_dim);
}

// --- build_certificate_from_workspace tests (extracted) --------------------

#[path = "moonshot_certificate_builder_workspace_tests.rs"]
mod workspace_tests;
