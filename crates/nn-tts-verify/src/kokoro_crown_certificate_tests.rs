// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Kokoro CROWN certificate integration.
//!
//! Part of #4254.

use crate::kokoro_contracts::{all_contracts, JunctionContract, VerifiedJunctionContract};
use crate::kokoro_crown_certificate::{
    CertificateError, JunctionContractEntry, KokoroCrownCertificate, PropertyCrownEntry,
    KOKORO_CERTIFICATE_VERSION,
};
use crate::moonshot::{MoonshotCertificate, MoonshotStatus, VerificationLevel};
use crate::moonshot_crown::{MoonshotCrownBundle, MoonshotPropertyResult};
use crate::pipeline::PipelineCertificate;

// ============================================================================
// Test helpers
// ============================================================================

fn make_property_result(
    index: usize,
    name: &'static str,
    proven: bool,
    bound: f64,
    threshold: f64,
) -> MoonshotPropertyResult {
    MoonshotPropertyResult {
        property_index: index,
        property_name: name,
        proven,
        level: if proven {
            VerificationLevel::CrownProven
        } else {
            VerificationLevel::Empirical
        },
        bound_value: bound,
        threshold,
        is_sound: proven,
        explanation: format!(
            "P{}: {} (bound={bound:.4}, threshold={threshold:.4})",
            index + 1,
            if proven { "proven" } else { "unproven" }
        ),
    }
}

fn make_pipeline_cert(out_lo: f64, out_hi: f64, is_sound: bool) -> PipelineCertificate {
    PipelineCertificate {
        e2e_input_lower: vec![-1.0; 8],
        e2e_input_upper: vec![1.0; 8],
        e2e_output_lower: vec![out_lo; 8],
        e2e_output_upper: vec![out_hi; 8],
        junctions: vec![],
        stages: vec![],
        is_valid: true,
        is_sound,
    }
}

fn make_crown_bundle(all_proven: bool, is_sound: bool, dim: usize) -> MoonshotCrownBundle {
    let results = vec![
        make_property_result(0, "Non-silent", all_proven, 0.05, 0.01),
        make_property_result(1, "Non-clipping", all_proven, 0.95, 1.0),
        make_property_result(2, "Intelligible", all_proven, 0.9, 0.5),
        make_property_result(3, "Speaker-consistent", all_proven, 0.02, 0.1),
    ];
    MoonshotCrownBundle {
        results,
        pipeline_cert: make_pipeline_cert(-0.5, 0.5, is_sound),
        verification_dim: dim,
        all_proven,
    }
}

fn make_verified_junctions(all_verified: bool) -> Vec<VerifiedJunctionContract> {
    all_contracts()
        .into_iter()
        .map(|c| {
            let mut vjc = VerifiedJunctionContract::new(c);
            if all_verified {
                vjc.bounds_verified = true;
            }
            vjc
        })
        .collect()
}

fn make_verified_junctions_with_proofs() -> Vec<VerifiedJunctionContract> {
    all_contracts()
        .into_iter()
        .map(|c| {
            VerifiedJunctionContract::new(c).with_composition_proof(
                "-- Lean4 placeholder".to_string(),
                "crown_composition_sound".to_string(),
            )
        })
        .collect()
}

// ============================================================================
// 1. Certificate creation tests
// ============================================================================

#[test]
fn test_certificate_creation_from_components() {
    let bundle = make_crown_bundle(true, true, 192);
    let junctions = make_verified_junctions(true);

    let cert = KokoroCrownCertificate::from_components("kokoro-v1", &bundle, &junctions);

    assert_eq!(cert.version, KOKORO_CERTIFICATE_VERSION);
    assert_eq!(cert.model_name, "kokoro-v1");
    assert_eq!(cert.properties.len(), 4);
    assert_eq!(cert.junctions.len(), 6);
    assert_eq!(cert.verification_dim, 192);
    assert!(cert.all_properties_proven);
    assert!(cert.all_junctions_verified);
    assert!(cert.pipeline_is_sound);
    assert_eq!(cert.proven_count, 4);
    assert_eq!(cert.total_count, 4);
    assert_eq!(cert.junctions_verified_count, 6);
    assert_eq!(cert.junctions_total_count, 6);
}

#[test]
fn test_certificate_creation_partial_proven() {
    let mut bundle = make_crown_bundle(false, true, 128);
    // Only first two properties proven.
    bundle.results[0] = make_property_result(0, "Non-silent", true, 0.05, 0.01);
    bundle.results[1] = make_property_result(1, "Non-clipping", true, 0.95, 1.0);
    bundle.results[2] = make_property_result(2, "Intelligible", false, 0.3, 0.5);
    bundle.results[3] = make_property_result(3, "Speaker-consistent", false, 0.15, 0.1);

    let junctions = make_verified_junctions(false);

    let cert = KokoroCrownCertificate::from_components("kokoro-partial", &bundle, &junctions);

    assert!(!cert.all_properties_proven);
    assert!(!cert.all_junctions_verified);
    assert_eq!(cert.proven_count, 2);
    assert_eq!(cert.total_count, 4);
    assert_eq!(cert.junctions_verified_count, 0);
}

#[test]
fn test_certificate_from_moonshot_and_bundle() {
    let status = MoonshotStatus::from_repo();
    let moonshot =
        MoonshotCertificate::from_status(&status, "dvoice-kokoro-v1", "English text", "abc123");
    let bundle = make_crown_bundle(true, true, 192);

    let cert = KokoroCrownCertificate::from_moonshot_and_bundle(&moonshot, &bundle);

    assert_eq!(cert.model_name, "dvoice-kokoro-v1");
    // Default junctions (6) should be present.
    assert_eq!(cert.junctions.len(), 6);
    // Default junctions are unverified.
    assert_eq!(cert.junctions_verified_count, 0);
}

#[test]
fn test_certificate_with_composition_proofs() {
    let bundle = make_crown_bundle(true, true, 256);
    let junctions = make_verified_junctions_with_proofs();

    let cert = KokoroCrownCertificate::from_components("kokoro-full-proofs", &bundle, &junctions);

    assert_eq!(cert.composition_proof_count, 6);
    for j in &cert.junctions {
        assert!(j.has_composition_proof);
        assert!(j.composition_theorem_name.is_some());
    }
}

// ============================================================================
// 2. JSON serialization roundtrip tests
// ============================================================================

#[test]
fn test_json_serialization_roundtrip() {
    let bundle = make_crown_bundle(true, true, 192);
    let junctions = make_verified_junctions(true);
    let cert = KokoroCrownCertificate::from_components("kokoro-serial", &bundle, &junctions);

    let json = cert.to_json().expect("serialization should succeed");
    let deserialized =
        KokoroCrownCertificate::from_json(&json).expect("deserialization should succeed");

    assert_eq!(deserialized.version, cert.version);
    assert_eq!(deserialized.model_name, cert.model_name);
    assert_eq!(deserialized.properties.len(), cert.properties.len());
    assert_eq!(deserialized.junctions.len(), cert.junctions.len());
    assert_eq!(deserialized.verification_dim, cert.verification_dim);
    assert_eq!(
        deserialized.all_properties_proven,
        cert.all_properties_proven
    );
    assert_eq!(
        deserialized.all_junctions_verified,
        cert.all_junctions_verified
    );
    assert_eq!(deserialized.proven_count, cert.proven_count);
    assert_eq!(deserialized.total_count, cert.total_count);
}

#[test]
fn test_json_contains_all_fields() {
    let bundle = make_crown_bundle(true, true, 192);
    let junctions = make_verified_junctions(true);
    let cert = KokoroCrownCertificate::from_components("kokoro-fields", &bundle, &junctions);

    let json = cert.to_json().expect("serialization should succeed");

    assert!(json.contains("\"version\""));
    assert!(json.contains("\"model_name\""));
    assert!(json.contains("\"generated_at\""));
    assert!(json.contains("\"properties\""));
    assert!(json.contains("\"junctions\""));
    assert!(json.contains("\"verification_dim\""));
    assert!(json.contains("\"all_properties_proven\""));
    assert!(json.contains("\"all_junctions_verified\""));
    assert!(json.contains("\"pipeline_is_sound\""));
    assert!(json.contains("\"proven_count\""));
    assert!(json.contains("\"composition_proof_count\""));
    assert!(json.contains("kokoro-fields"));
}

#[test]
fn test_json_roundtrip_preserves_property_details() {
    let bundle = make_crown_bundle(false, false, 64);
    let junctions = make_verified_junctions(false);
    let cert = KokoroCrownCertificate::from_components("detail-test", &bundle, &junctions);

    let json = cert.to_json().expect("serialization should succeed");
    let rt = KokoroCrownCertificate::from_json(&json).expect("deserialization should succeed");

    for (original, roundtripped) in cert.properties.iter().zip(rt.properties.iter()) {
        assert_eq!(original.property_index, roundtripped.property_index);
        assert_eq!(original.property_name, roundtripped.property_name);
        assert_eq!(original.proven, roundtripped.proven);
        assert!(
            (original.bound_value - roundtripped.bound_value).abs() < 1e-10,
            "bound_value mismatch for P{}",
            original.property_index + 1
        );
        assert!(
            (original.threshold - roundtripped.threshold).abs() < 1e-10,
            "threshold mismatch for P{}",
            original.property_index + 1
        );
    }
}

#[test]
fn test_json_deserialization_of_invalid_json() {
    let result = KokoroCrownCertificate::from_json("not valid json");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, CertificateError::Serialization(_)),
        "should be Serialization error"
    );
}

// ============================================================================
// 3. Certificate validation tests
// ============================================================================

#[test]
fn test_validation_passes_for_valid_certificate() {
    let bundle = make_crown_bundle(true, true, 192);
    let junctions = make_verified_junctions(true);
    let cert = KokoroCrownCertificate::from_components("valid-cert", &bundle, &junctions);

    assert!(cert.validate().is_ok());
}

#[test]
fn test_validation_fails_for_nan_bound_value() {
    let bundle = make_crown_bundle(true, true, 192);
    let junctions = make_verified_junctions(true);
    let mut cert = KokoroCrownCertificate::from_components("nan-cert", &bundle, &junctions);

    cert.properties[0].bound_value = f64::NAN;

    let result = cert.validate();
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("non-finite bound_value"));
}

#[test]
fn test_validation_fails_for_inf_threshold() {
    let bundle = make_crown_bundle(true, true, 192);
    let junctions = make_verified_junctions(true);
    let mut cert = KokoroCrownCertificate::from_components("inf-cert", &bundle, &junctions);

    cert.properties[1].threshold = f64::INFINITY;

    let result = cert.validate();
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("non-finite threshold"));
}

#[test]
fn test_validation_fails_for_inverted_junction_bounds() {
    let bundle = make_crown_bundle(true, true, 192);
    let junctions = make_verified_junctions(true);
    let mut cert = KokoroCrownCertificate::from_components("inverted-cert", &bundle, &junctions);

    // Invert bounds: lower > upper.
    cert.junctions[0].lower = 100.0;
    cert.junctions[0].upper = -100.0;

    let result = cert.validate();
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("inverted bounds"));
}

#[test]
fn test_validation_fails_for_nan_junction_bounds() {
    let bundle = make_crown_bundle(true, true, 192);
    let junctions = make_verified_junctions(true);
    let mut cert = KokoroCrownCertificate::from_components("nan-junc-cert", &bundle, &junctions);

    cert.junctions[2].upper = f64::NAN;

    let result = cert.validate();
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("non-finite bounds"));
}

#[test]
fn test_validation_fails_for_inconsistent_proven_count() {
    let bundle = make_crown_bundle(true, true, 192);
    let junctions = make_verified_junctions(true);
    let mut cert = KokoroCrownCertificate::from_components("count-cert", &bundle, &junctions);

    // Tamper with the proven count.
    cert.proven_count = 99;

    let result = cert.validate();
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("proven_count mismatch"));
}

#[test]
fn test_validation_fails_for_inconsistent_junction_count() {
    let bundle = make_crown_bundle(true, true, 192);
    let junctions = make_verified_junctions(true);
    let mut cert = KokoroCrownCertificate::from_components("jcount-cert", &bundle, &junctions);

    cert.junctions_total_count = 99;

    let result = cert.validate();
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("junctions_total_count mismatch"));
}

#[test]
fn test_validation_fails_for_out_of_range_property_index() {
    let bundle = make_crown_bundle(true, true, 192);
    let junctions = make_verified_junctions(true);
    let mut cert = KokoroCrownCertificate::from_components("idx-cert", &bundle, &junctions);

    cert.properties[0].property_index = 99;

    let result = cert.validate();
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("out of range"));
}

// ============================================================================
// 4. Integration with junction contracts
// ============================================================================

#[test]
fn test_junction_contract_entries_match_all_contracts() {
    let contracts = all_contracts();
    let junctions: Vec<JunctionContractEntry> = contracts
        .iter()
        .map(JunctionContractEntry::from_contract)
        .collect();

    assert_eq!(junctions.len(), 6);
    assert_eq!(junctions[0].name, "J2_F0");
    assert_eq!(junctions[1].name, "J2_ENERGY");
    assert_eq!(junctions[2].name, "J3_MAGNITUDE");
    assert_eq!(junctions[3].name, "J3B_PHASE");
    assert_eq!(junctions[4].name, "J4_BF16");
    assert_eq!(junctions[5].name, "J5_AUDIO");

    // All should be unverified by default.
    for j in &junctions {
        assert!(!j.bounds_verified);
        assert!(!j.has_composition_proof);
        assert!(j.composition_theorem_name.is_none());
    }
}

#[test]
fn test_verified_junction_contract_entry_with_proof() {
    let contract = JunctionContract::new("J5_AUDIO", "iSTFT output", -1.0, 1.0);
    let vjc = VerifiedJunctionContract::new(contract).with_composition_proof(
        "theorem audio_bound : ...".to_string(),
        "audio_bound_sound".to_string(),
    );

    let entry = JunctionContractEntry::from_verified(&vjc);

    assert_eq!(entry.name, "J5_AUDIO");
    assert!(entry.bounds_verified);
    assert!(entry.has_composition_proof);
    assert_eq!(
        entry.composition_theorem_name.as_deref(),
        Some("audio_bound_sound")
    );
}

#[test]
fn test_is_fully_certified_requires_all_conditions() {
    // All proven, all verified, sound pipeline.
    let bundle = make_crown_bundle(true, true, 192);
    let junctions = make_verified_junctions(true);
    let cert = KokoroCrownCertificate::from_components("full", &bundle, &junctions);
    assert!(cert.is_fully_certified());

    // Not all proven.
    let bundle2 = make_crown_bundle(false, true, 192);
    let cert2 = KokoroCrownCertificate::from_components("partial", &bundle2, &junctions);
    assert!(!cert2.is_fully_certified());

    // Unsound pipeline.
    let bundle3 = make_crown_bundle(true, false, 192);
    let cert3 = KokoroCrownCertificate::from_components("unsound", &bundle3, &junctions);
    assert!(!cert3.is_fully_certified());

    // Unverified junctions.
    let junctions_unverified = make_verified_junctions(false);
    let bundle4 = make_crown_bundle(true, true, 192);
    let cert4 =
        KokoroCrownCertificate::from_components("unverified-junc", &bundle4, &junctions_unverified);
    assert!(!cert4.is_fully_certified());
}

#[test]
fn test_summary_report_contains_key_sections() {
    let bundle = make_crown_bundle(true, true, 192);
    let junctions = make_verified_junctions_with_proofs();
    let cert = KokoroCrownCertificate::from_components("summary-test", &bundle, &junctions);

    let summary = cert.summary();

    assert!(summary.contains("Kokoro CROWN Certificate"));
    assert!(summary.contains("summary-test"));
    assert!(summary.contains("Properties:"));
    assert!(summary.contains("Junctions:"));
    assert!(summary.contains("PROVEN"));
    assert!(summary.contains("VERIFIED"));
    assert!(summary.contains("[Lean4]"));
    assert!(summary.contains("Fully certified:"));
}

#[test]
fn test_property_crown_entry_from_result() {
    let result = make_property_result(1, "Non-clipping", true, 0.95, 1.0);
    let entry = PropertyCrownEntry::from_result(&result);

    assert_eq!(entry.property_index, 1);
    assert_eq!(entry.property_name, "Non-clipping");
    assert!(entry.proven);
    assert!((entry.bound_value - 0.95).abs() < 1e-10);
    assert!((entry.threshold - 1.0).abs() < 1e-10);
    assert!(entry.is_sound);
}

#[test]
fn test_certificate_with_empty_results() {
    let bundle = MoonshotCrownBundle {
        results: vec![],
        pipeline_cert: make_pipeline_cert(-0.5, 0.5, true),
        verification_dim: 64,
        all_proven: true,
    };
    let cert = KokoroCrownCertificate::from_components("empty", &bundle, &[]);

    assert_eq!(cert.properties.len(), 0);
    assert_eq!(cert.junctions.len(), 0);
    assert_eq!(cert.proven_count, 0);
    assert_eq!(cert.total_count, 0);
    assert!(cert.validate().is_ok());
}

#[test]
fn test_json_roundtrip_with_composition_proofs() {
    let bundle = make_crown_bundle(true, true, 256);
    let junctions = make_verified_junctions_with_proofs();
    let cert = KokoroCrownCertificate::from_components("lean4-test", &bundle, &junctions);

    let json = cert.to_json().expect("serialization should succeed");
    let rt = KokoroCrownCertificate::from_json(&json).expect("deserialization should succeed");

    assert_eq!(rt.composition_proof_count, 6);
    for j in &rt.junctions {
        assert!(j.has_composition_proof);
        assert_eq!(
            j.composition_theorem_name.as_deref(),
            Some("crown_composition_sound")
        );
    }
}
