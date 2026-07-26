// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_moonshot_status_constructs() {
    let status = MoonshotStatus::from_repo();
    assert_eq!(status.properties.len(), 8);
    for prop in &status.properties {
        assert!(!prop.name.is_empty());
    }
}

#[test]
fn test_property_7_memory_safety_at_kani_proven() {
    let status = MoonshotStatus::from_repo();
    // Property 7 (index 6): Memory safety — 491 Kani harnesses exist.
    assert!(
        status.properties[6].verified >= VerificationLevel::KaniProven,
        "Memory safety should be KaniProven, got {:?}",
        status.properties[6].verified
    );
}

#[test]
fn test_property_8_correct_impl_at_smt_proven() {
    let status = MoonshotStatus::from_repo();
    // Property 8 (index 7): Correct implementation — ay SMT 15/15 linear kernels.
    assert!(
        status.properties[7].verified >= VerificationLevel::SmtProven,
        "Correct implementation should be SmtProven, got {:?}",
        status.properties[7].verified
    );
}

#[test]
fn test_properties_1_2_at_least_empirical() {
    let status = MoonshotStatus::from_repo();
    // Properties 1+2 (non-silence, non-clipping) have hard bounds + CROWN partial.
    assert!(
        status.properties[0].verified >= VerificationLevel::Empirical,
        "Non-silence should be at least Empirical"
    );
    assert!(
        status.properties[1].verified >= VerificationLevel::Empirical,
        "Non-clipping should be at least Empirical"
    );
}

#[test]
fn test_property_3_intelligibility_at_crown_partial() {
    let status = MoonshotStatus::from_repo();
    // Duration positivity exists — CrownPartial at minimum.
    assert!(
        status.properties[2].verified >= VerificationLevel::CrownPartial,
        "Intelligibility should be at least CrownPartial"
    );
}

/// All 8 moonshot properties now have at least CrownPartial evidence.
///
/// Speaker (P4) reached CrownPartial via moonshot_crown_speaker.rs and
/// pipeline.rs cross-cutting artifact. Temporal (P5) reached CrownPartial
/// via pipeline_hybrid.rs CROWN-coupled timing and moonshot_crown.rs bridge.
/// Streaming (P6) reached CrownPartial via moonshot_crown.rs bounded
/// crossfade discontinuity.
#[test]
fn test_all_properties_at_least_crown_partial() {
    let status = MoonshotStatus::from_repo();
    for (i, prop) in status.properties.iter().enumerate() {
        assert!(
            prop.verified >= VerificationLevel::CrownPartial,
            "Property {} ({}) should be at least CrownPartial, got {:?}",
            i + 1,
            prop.name,
            prop.verified
        );
    }
    assert!(
        status.all_at_least_crown_partial(),
        "All 8 properties should now be CrownPartial+"
    );
}

#[test]
fn test_all_properties_have_known_gaps() {
    let status = MoonshotStatus::from_repo();
    for (i, prop) in status.properties.iter().enumerate() {
        assert!(
            !prop.gaps.is_empty(),
            "Property {} ({}) should have at least one known gap",
            i + 1,
            prop.name
        );
    }
}

#[test]
fn test_artifact_registry_covers_all_properties() {
    let artifacts = artifact_registry();
    let mut covered = [false; 8];
    for artifact in &artifacts {
        for &idx in artifact.properties {
            if idx < 8 {
                covered[idx] = true;
            }
        }
    }
    for (i, c) in covered.iter().enumerate() {
        assert!(c, "Property {} should have at least one artifact", i + 1);
    }
}

#[test]
fn test_report_non_empty() {
    let status = MoonshotStatus::from_repo();
    let report = status.report();
    assert!(report.contains("Moonshot Status"));
    assert!(report.contains("Non-silent"));
    assert!(report.contains("Memory-safe"));
    assert!(report.contains("Summary:"));
}

#[test]
fn test_display_impl() {
    let status = MoonshotStatus::from_repo();
    let display = format!("{status}");
    assert!(display.contains("P1:"));
    assert!(display.contains("P8:"));
}

#[test]
fn test_verification_level_ordering() {
    assert!(VerificationLevel::None < VerificationLevel::Empirical);
    assert!(VerificationLevel::Empirical < VerificationLevel::CrownPartial);
    assert!(VerificationLevel::CrownPartial < VerificationLevel::CrownProven);
    assert!(VerificationLevel::CrownProven < VerificationLevel::KaniProven);
    assert!(VerificationLevel::KaniProven < VerificationLevel::SmtProven);
}

#[test]
fn test_level_counts_sum_to_8() {
    let status = MoonshotStatus::from_repo();
    let counts = status.level_counts();
    let total: usize = counts.iter().map(|(_, c)| c).sum();
    assert_eq!(total, 8, "Level counts should sum to 8 properties");
}

#[test]
fn test_streaming_at_least_empirical() {
    let status = MoonshotStatus::from_repo();
    // Property 6 (index 5): Streaming safety has empirical tests.
    assert!(
        status.properties[5].verified >= VerificationLevel::Empirical,
        "Streaming safety should be at least Empirical"
    );
}

// --- MoonshotCertificate tests ---

#[test]
fn test_certificate_from_status() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(
        &status,
        "dvoice-kokoro-v1",
        "English text, ≤50 words",
        "abc123",
    );
    assert_eq!(cert.model_name, "dvoice-kokoro-v1");
    assert_eq!(cert.properties.len(), 8);
    assert_eq!(cert.source_hash, "abc123");
    assert!(cert.all_proven); // All 8 properties now CrownProven+ after #1741 enrichment
}

#[test]
fn test_certificate_property_names_match() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test", "test", "hash");
    for (i, prop) in cert.properties.iter().enumerate() {
        assert_eq!(prop.property_index, i);
        assert_eq!(prop.property_name, status.properties[i].name);
    }
}

#[test]
fn test_certificate_to_json_valid() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test-model", "English text", "deadbeef");
    let json = cert.to_json();
    assert!(json.contains("\"model_name\": \"test-model\""));
    assert!(json.contains("\"source_hash\": \"deadbeef\""));
    assert!(json.contains("\"properties\": ["));
    assert!(json.contains("\"name\""));
    assert!(json.contains("\"level\""));
    assert!(json.contains("\"proof_artifacts\""));
    assert!(json.contains("\"assumptions\""));
}

#[test]
fn test_certificate_json_has_all_8_properties() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test", "test", "hash");
    let json = cert.to_json();
    // Count property blocks by counting "index" occurrences.
    let count = json.matches("\"index\":").count();
    assert_eq!(count, 8, "JSON should have 8 property entries");
}

#[test]
fn test_certificate_display() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(
        &status,
        "dvoice-kokoro-v1",
        "English text, ≤50 words",
        "abc123",
    );
    let s = format!("{cert}");
    assert!(s.contains("Moonshot Certificate: dvoice-kokoro-v1"));
    assert!(s.contains("P1:"));
    assert!(s.contains("P8:"));
    assert!(s.contains("all_proven=true"));
}

#[test]
fn test_certificate_with_crown_results() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test", "test", "hash");

    // Create a mock CROWN bundle.
    let bundle = super::super::moonshot_crown::MoonshotCrownBundle {
        results: vec![
            super::super::moonshot_crown::MoonshotPropertyResult {
                property_index: 0,
                property_name: "Non-silent (RMS > 0.01)",
                proven: true,
                level: VerificationLevel::CrownProven,
                bound_value: 0.8,
                threshold: 0.01,
                is_sound: true,
                explanation: "proven".to_string(),
            },
            super::super::moonshot_crown::MoonshotPropertyResult {
                property_index: 1,
                property_name: "Non-clipping (samples in [-1, 1])",
                proven: true,
                level: VerificationLevel::CrownProven,
                bound_value: 0.9,
                threshold: 1.0,
                is_sound: true,
                explanation: "proven".to_string(),
            },
        ],
        pipeline_cert: crate::pipeline::PipelineCertificate {
            e2e_input_lower: vec![-1.0; 8],
            e2e_input_upper: vec![1.0; 8],
            e2e_output_lower: vec![-0.5; 8],
            e2e_output_upper: vec![0.5; 8],
            junctions: vec![],
            stages: vec![],
            is_valid: true,
            is_sound: true,
        },
        verification_dim: 64,
        all_proven: true,
    };

    let enriched = cert.with_crown_results(&bundle);
    assert_eq!(enriched.verification_dim, Some(64));
    assert_eq!(enriched.properties[0].level, VerificationLevel::CrownProven);
    assert_eq!(enriched.properties[0].bound_value, Some(0.8));
    assert_eq!(enriched.properties[0].threshold, Some(0.01));
    // Properties 2-7 unchanged — still from status.
    assert_eq!(enriched.properties[2].level, status.properties[2].verified);
}

#[test]
fn test_certificate_json_with_crown_has_bound_values() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test", "test", "hash");

    let bundle = super::super::moonshot_crown::MoonshotCrownBundle {
        results: vec![super::super::moonshot_crown::MoonshotPropertyResult {
            property_index: 0,
            property_name: "Non-silent (RMS > 0.01)",
            proven: true,
            level: VerificationLevel::CrownProven,
            bound_value: 0.8,
            threshold: 0.01,
            is_sound: true,
            explanation: "proven".to_string(),
        }],
        pipeline_cert: crate::pipeline::PipelineCertificate {
            e2e_input_lower: vec![-1.0; 4],
            e2e_input_upper: vec![1.0; 4],
            e2e_output_lower: vec![-0.5; 4],
            e2e_output_upper: vec![0.5; 4],
            junctions: vec![],
            stages: vec![],
            is_valid: true,
            is_sound: true,
        },
        verification_dim: 64,
        all_proven: true,
    };

    let enriched = cert.with_crown_results(&bundle);
    let json = enriched.to_json();
    assert!(json.contains("\"bound_value\":"));
    assert!(json.contains("\"threshold\":"));
    assert!(json.contains("\"verification_dim\": 64"));
}

#[test]
fn test_certificate_with_timing_results() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test", "test", "hash");

    // Create a mock TimingCertificate with sound CROWN bounds and passing timing.
    let timing_cert = crate::pipeline::TimingCertificate {
        bounds_cert: crate::pipeline::PipelineCertificate {
            e2e_input_lower: vec![-1.0; 8],
            e2e_input_upper: vec![1.0; 8],
            e2e_output_lower: vec![-0.5; 8],
            e2e_output_upper: vec![0.5; 8],
            junctions: vec![],
            stages: vec![],
            is_valid: true,
            is_sound: true,
        },
        cost_profiles: vec![crate::cost_model::LayerCostProfile {
            layer_name: "kokoro_decoder".to_string(),
            flops: 1_000_000,
            memory_bytes: 500_000,
            estimated_time_us: 50_000.0,
            measured_time_us: None,
        }],
        worst_case_time_us: 50_000.0,
        total_flops: 1_000_000,
        total_memory_bytes: 500_000,
        hardware_name: "Apple M4 Max".to_string(),
        timing_bound_us: 100_000.0,
        timing_bound_met: true,
        overall_passed: true,
        peak_memory: None,
    };

    let enriched = cert.with_timing_results(&timing_cert);

    // Property 5 (index 4) should be enriched with timing data.
    assert_eq!(enriched.properties[4].level, VerificationLevel::CrownProven);
    assert_eq!(enriched.properties[4].bound_value, Some(50_000.0));
    assert_eq!(enriched.properties[4].threshold, Some(100_000.0));
    assert!(
        enriched.properties[4]
            .assumptions
            .iter()
            .any(|a| a.contains("CROWN-coupled timing")),
        "Sound timing should have CROWN-coupled assumption"
    );
    assert!(
        enriched.properties[4]
            .assumptions
            .iter()
            .any(|a| a.contains("Apple M4 Max")),
        "Timing assumptions should include hardware name"
    );
}

#[test]
fn test_certificate_with_timing_results_ibp_fallback() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test", "test", "hash");

    // IBP fallback: is_sound = false.
    let timing_cert = crate::pipeline::TimingCertificate {
        bounds_cert: crate::pipeline::PipelineCertificate {
            e2e_input_lower: vec![-1.0; 8],
            e2e_input_upper: vec![1.0; 8],
            e2e_output_lower: vec![-10.0; 8],
            e2e_output_upper: vec![10.0; 8],
            junctions: vec![],
            stages: vec![],
            is_valid: true,
            is_sound: false,
        },
        cost_profiles: vec![],
        worst_case_time_us: 200_000.0,
        total_flops: 2_000_000,
        total_memory_bytes: 1_000_000,
        hardware_name: "Generic GPU".to_string(),
        timing_bound_us: 100_000.0,
        timing_bound_met: false,
        overall_passed: false,
        peak_memory: None,
    };

    let enriched = cert.with_timing_results(&timing_cert);

    // timing_bound_met=false → proven=false → Empirical level.
    assert_eq!(enriched.properties[4].level, VerificationLevel::Empirical);
    assert!(
        enriched.properties[4]
            .assumptions
            .iter()
            .any(|a| a.contains("IBP fallback")),
        "IBP fallback should be noted in assumptions"
    );
}
