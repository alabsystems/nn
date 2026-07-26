// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for gap_detector.rs, certify.rs, and certificate.rs types
//! and validation logic.
//!
//! Part of #3815.

use nn_verify::certificate::{CertificateBundle, ProofCertificate, CERTIFICATE_VERSION};
use nn_verify::gap_detector::{
    detect_gaps, format_gap_report, kokoro_pipeline_stages, PipelineStage, StageGapResult,
};
use nn_verify::{
    CertificateError, InputBoundsRecord, KaniOutcome, KaniProofRecord, KernelVerification,
    LayerBoundRecord, OutputTensorBounds, ParamInputRecord, PrecisionModel, PropMethod,
    SmtProofVerdict, VerificationSoundnessMode,
};

// ---------------------------------------------------------------------------
// Test helpers (using constructors for #[non_exhaustive] types)
// ---------------------------------------------------------------------------

fn make_verification() -> KernelVerification {
    make_verification_with_bounds(-9.704, 10.296)
}

fn make_verification_with_bounds(lower: f32, upper: f32) -> KernelVerification {
    let mut v = KernelVerification::new(
        "test_kernel".to_string(),
        PropMethod::Ibp,
        lower,
        upper,
        upper - lower,
        true,
    );
    v.output_tensor = Some(OutputTensorBounds::new(vec![lower], vec![upper], vec![1]));
    v
}

fn make_input_spec() -> InputBoundsRecord {
    InputBoundsRecord::new(&[ParamInputRecord::new(0, -10.0, 10.0)], &[1.0])
}

fn make_status_entry(
    status: &str,
    method: &str,
    width: f64,
    proof_strength: &str,
) -> serde_json::Value {
    serde_json::json!({
        "status": status,
        "method": method,
        "output_width": width,
        "proof_strength": proof_strength,
        "soundness_mode": if proof_strength == "sound" { "sound" } else { "heuristic" }
    })
}

// ===========================================================================
// Gap detector tests
// ===========================================================================

/// All pipeline stages have unique status_key prefixes.
#[test]
fn test_pipeline_stages_unique_status_keys() {
    let stages = kokoro_pipeline_stages();
    let mut keys: Vec<&str> = stages.iter().map(|s| s.status_key).collect();
    let original_len = keys.len();
    keys.sort_unstable();
    keys.dedup();
    assert_eq!(
        keys.len(),
        original_len,
        "duplicate status_key found in pipeline stages"
    );
}

/// All pipeline stages have unique names.
#[test]
fn test_pipeline_stages_unique_names() {
    let stages = kokoro_pipeline_stages();
    let mut names: Vec<&str> = stages.iter().map(|s| s.name).collect();
    let original_len = names.len();
    names.sort_unstable();
    names.dedup();
    assert_eq!(
        names.len(),
        original_len,
        "duplicate name found in pipeline stages"
    );
}

/// StageGapResult::has_any_bounds is false when all bound flags are false.
#[test]
fn test_stage_gap_result_has_any_bounds_all_false() {
    let result = StageGapResult {
        stage: PipelineStage {
            name: "test",
            status_key: "test_key",
            is_compiled_segment: false,
            is_bridge: true,
            source_file: "test.rs",
            cpu_bridges: &[],
        },
        has_ibp_bounds: false,
        has_crown_bounds: false,
        has_analytical_bounds: false,
        is_vacuous: false,
        bound_width: None,
        proof_strength: None,
        soundness_mode: None,
        has_constructive_certificate: false,
    };
    assert!(!result.has_any_bounds());
}

/// StageGapResult::has_any_bounds is true for IBP-only.
#[test]
fn test_stage_gap_result_has_any_bounds_ibp_only() {
    let result = StageGapResult {
        stage: PipelineStage {
            name: "test",
            status_key: "test_key",
            is_compiled_segment: true,
            is_bridge: false,
            source_file: "test.rs",
            cpu_bridges: &[],
        },
        has_ibp_bounds: true,
        has_crown_bounds: false,
        has_analytical_bounds: false,
        is_vacuous: false,
        bound_width: Some(5.0),
        proof_strength: Some("sound".to_string()),
        soundness_mode: Some("sound".to_string()),
        has_constructive_certificate: false,
    };
    assert!(result.has_any_bounds());
}

/// StageGapResult::has_any_bounds is true for analytical-only.
#[test]
fn test_stage_gap_result_has_any_bounds_analytical_only() {
    let result = StageGapResult {
        stage: PipelineStage {
            name: "test",
            status_key: "test_key",
            is_compiled_segment: false,
            is_bridge: true,
            source_file: "test.rs",
            cpu_bridges: &[],
        },
        has_ibp_bounds: false,
        has_crown_bounds: false,
        has_analytical_bounds: true,
        is_vacuous: false,
        bound_width: Some(3.0),
        proof_strength: Some("sound".to_string()),
        soundness_mode: Some("sound".to_string()),
        has_constructive_certificate: false,
    };
    assert!(result.has_any_bounds());
}

/// Gap report with no kernels object in the status JSON treats all stages as gaps.
#[test]
fn test_detect_gaps_no_kernels_key() {
    let status = serde_json::json!({});
    let report = detect_gaps(&status);
    assert_eq!(
        report.total_gaps, 8,
        "all stages should be gaps when no 'kernels' key exists"
    );
    assert_eq!(report.vacuous_count, 0);
}

/// Mixed methods: some CROWN, some IBP, some ANALYTICAL.
#[test]
fn test_detect_gaps_mixed_methods() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_status_entry("verified", "IBP", 200.0, "heuristic"),
            "kokoro_production_bert_encoder_crown": make_status_entry("verified", "CROWN", 1.4, "sound"),
            "kokoro_production_text_encoder": make_status_entry("verified", "IBP", 1.4, "sound"),
            "kokoro_production_prosody_predictor": make_status_entry("verified", "IBP", 2.0, "sound"),
            "kokoro_production_f0_predictor": make_status_entry("verified", "IBP", 2.0, "heuristic"),
            "kokoro_production_generator": make_status_entry("verified", "IBP", 2.0, "heuristic"),
            "kokoro_production_length_regulate": make_status_entry("verified", "ANALYTICAL", 10.0, "sound"),
            "kokoro_production_harmonic_source": make_status_entry("verified", "ANALYTICAL", 5.0, "sound"),
            "kokoro_production_istft": make_status_entry("verified", "CROWN", 2.0, "sound"),
        }
    });

    let report = detect_gaps(&status);
    assert_eq!(report.total_gaps, 0, "all stages should have bounds");

    // Check that each method type is detected correctly.
    let bert = report
        .stages
        .iter()
        .find(|r| r.stage.name == "PlBert + bert_encoder")
        .unwrap();
    assert!(bert.has_ibp_bounds);
    assert!(bert.has_crown_bounds);

    let regulate = report
        .stages
        .iter()
        .find(|r| r.stage.name.contains("length_regulate"))
        .unwrap();
    assert!(regulate.has_analytical_bounds);
    assert!(!regulate.has_ibp_bounds);
    assert!(!regulate.has_crown_bounds);
}

/// Alpha-CROWN variant is detected as CROWN.
#[test]
fn test_detect_gaps_alpha_crown_detected() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_status_entry("verified", "IBP", 200.0, "heuristic"),
            "kokoro_production_bert_encoder_crown": make_status_entry("verified", "AlphaCROWN", 1.0, "sound"),
        }
    });

    let report = detect_gaps(&status);
    let bert = report
        .stages
        .iter()
        .find(|r| r.stage.name == "PlBert + bert_encoder")
        .unwrap();
    assert!(
        bert.has_crown_bounds,
        "AlphaCROWN should be detected as CROWN"
    );
}

/// Beta-CROWN variant is detected as CROWN.
#[test]
fn test_detect_gaps_beta_crown_detected() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_text_encoder_crown": make_status_entry("verified", "BetaCROWN", 0.8, "sound"),
        }
    });

    let report = detect_gaps(&status);
    let te = report
        .stages
        .iter()
        .find(|r| r.stage.name == "TextEncoder")
        .unwrap();
    assert!(te.has_crown_bounds, "BetaCROWN should be detected as CROWN");
}

/// Mixed-IBP-CROWN is detected as CROWN.
#[test]
fn test_detect_gaps_mixed_ibp_crown_detected() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_generator_crown": make_status_entry("verified", "Mixed-IBP-CROWN", 3.0, "sound"),
        }
    });

    let report = detect_gaps(&status);
    let generator = report
        .stages
        .iter()
        .find(|r| r.stage.name == "Generator")
        .unwrap();
    assert!(
        generator.has_crown_bounds,
        "Mixed-IBP-CROWN should be detected as CROWN"
    );
}

/// Entries with status "failed" are not treated as verified.
#[test]
fn test_detect_gaps_failed_status_not_counted() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_status_entry("failed", "IBP", 300.0, "heuristic"),
        }
    });

    let report = detect_gaps(&status);
    let bert = report
        .stages
        .iter()
        .find(|r| r.stage.name == "PlBert + bert_encoder")
        .unwrap();
    assert!(
        !bert.has_ibp_bounds,
        "failed entry should not provide bounds"
    );
    assert!(!bert.has_any_bounds());
}

/// Format report shows [~~] for vacuous stages.
#[test]
fn test_format_gap_report_vacuous_marker() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_status_entry("verified", "IBP", 5000.0, "heuristic"),
        }
    });

    let report = detect_gaps(&status);
    let formatted = format_gap_report(&report);
    assert!(
        formatted.contains("[~~]"),
        "should have VACUOUS marker for width > 1000"
    );
}

/// Format report summary line includes correct counts.
#[test]
fn test_format_gap_report_summary_counts() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_status_entry("verified", "CROWN", 1.5, "sound"),
            "kokoro_production_text_encoder": make_status_entry("verified", "IBP", 1.4, "sound"),
        }
    });

    let report = detect_gaps(&status);
    let formatted = format_gap_report(&report);

    // 6 gaps (8 total - 2 verified)
    assert!(
        formatted.contains("6 gaps"),
        "should report 6 gaps in summary: {formatted}"
    );
    assert!(
        formatted.contains("8 total stages"),
        "should report 8 total stages: {formatted}"
    );
}

/// Proof strength "vacuous" label makes entry vacuous regardless of width.
#[test]
fn test_detect_gaps_vacuous_by_label_not_width() {
    let status = serde_json::json!({
        "kernels": {
            // Width is small (2.0) but proof_strength says "vacuous"
            "kokoro_production_bert_encoder": make_status_entry("verified", "IBP", 2.0, "vacuous"),
        }
    });

    let report = detect_gaps(&status);
    let bert = report
        .stages
        .iter()
        .find(|r| r.stage.name == "PlBert + bert_encoder")
        .unwrap();
    assert!(
        bert.is_vacuous,
        "proof_strength='vacuous' should override width-based check"
    );
    assert_eq!(report.vacuous_count, 1);
}

/// iSTFT stage has a CPU bridge documented.
#[test]
fn test_istft_stage_has_cpu_bridge() {
    let stages = kokoro_pipeline_stages();
    let istft = stages
        .iter()
        .find(|s| s.name.contains("iSTFT"))
        .expect("iSTFT stage should exist");
    assert!(
        !istft.cpu_bridges.is_empty(),
        "iSTFT should have istft_terminal_readback CPU bridge"
    );
    assert!(istft.cpu_bridges[0].contains("istft_terminal_readback"));
}

/// detect_gaps report length matches kokoro_pipeline_stages length.
#[test]
fn test_detect_gaps_report_length_matches_stages() {
    let status = serde_json::json!({ "kernels": {} });
    let report = detect_gaps(&status);
    let stages = kokoro_pipeline_stages();
    assert_eq!(report.stages.len(), stages.len());
}

// ===========================================================================
// Certificate validation tests
// ===========================================================================

/// Certificate with version 1 through CERTIFICATE_VERSION all pass validation.
#[test]
fn test_certificate_valid_versions() {
    let result = make_verification();
    for v in 1..=CERTIFICATE_VERSION {
        let mut cert = ProofCertificate::from_verification(&result, make_input_spec());
        cert.version = v;
        assert!(cert.validate().is_ok(), "version {v} should be valid");
    }
}

/// Certificate with layer_bounds where layer_index is misordered fails.
#[test]
fn test_certificate_validate_layer_index_misordered() {
    let result = make_verification();
    let mut cert = ProofCertificate::from_verification(&result, make_input_spec());
    cert.layer_bounds = Some(vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-1.0, 1.0)],
            output_bounds: vec![(-0.5, 0.5)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: None,
        },
        LayerBoundRecord {
            layer_index: 5, // Should be 1
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-0.5, 0.5)],
            output_bounds: vec![(0.0, 0.5)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: None,
        },
    ]);
    let err = cert.validate().unwrap_err();
    assert!(
        matches!(
            err,
            CertificateError::LayerIndexMismatch {
                expected: 1,
                actual: 5
            }
        ),
        "expected LayerIndexMismatch, got {err:?}"
    );
}

/// Certificate with empty layer_bounds Some(vec![]) fails.
#[test]
fn test_certificate_validate_empty_layer_bounds() {
    let result = make_verification();
    let mut cert = ProofCertificate::from_verification(&result, make_input_spec());
    cert.layer_bounds = Some(vec![]);
    let err = cert.validate().unwrap_err();
    assert!(matches!(err, CertificateError::EmptyLayerBounds));
}

/// Certificate with invalid weight_hash format fails.
#[test]
fn test_certificate_validate_invalid_weight_hash() {
    let result = make_verification();
    let mut cert = ProofCertificate::from_verification(&result, make_input_spec());
    cert.weight_hash = Some("not_a_valid_sha256".to_string());
    let err = cert.validate().unwrap_err();
    assert!(
        matches!(err, CertificateError::InvalidHash { ref field, .. } if field == "weight_hash"),
        "expected InvalidHash for weight_hash, got {err:?}"
    );
}

/// Certificate with invalid source_hash format fails.
#[test]
fn test_certificate_validate_invalid_source_hash() {
    let result = make_verification();
    let mut cert = ProofCertificate::from_verification(&result, make_input_spec());
    cert.source_hash = Some("abcdefg".to_string()); // Too short
    let err = cert.validate().unwrap_err();
    assert!(
        matches!(err, CertificateError::InvalidHash { ref field, .. } if field == "source_hash"),
        "expected InvalidHash for source_hash, got {err:?}"
    );
}

/// Certificate with valid 64-char hex hash passes.
#[test]
fn test_certificate_validate_valid_hashes() {
    let result = make_verification();
    let mut cert = ProofCertificate::from_verification(&result, make_input_spec());
    let valid_hash = "a".repeat(64);
    cert.weight_hash = Some(valid_hash.clone());
    cert.source_hash = Some(valid_hash);
    assert!(cert.validate().is_ok());
}

/// Certificate with invalid content_hash format fails validation.
#[test]
fn test_certificate_validate_invalid_content_hash() {
    let result = make_verification();
    let mut cert = ProofCertificate::from_verification(&result, make_input_spec());
    cert.content_hash = Some("short".to_string());
    let err = cert.validate().unwrap_err();
    assert!(
        matches!(err, CertificateError::InvalidHash { ref field, .. } if field == "content_hash"),
        "expected InvalidHash for content_hash, got {err:?}"
    );
}

/// Certificate with invalid hmac_signature format fails validation.
#[test]
fn test_certificate_validate_invalid_hmac_signature() {
    let result = make_verification();
    let mut cert = ProofCertificate::from_verification(&result, make_input_spec());
    cert.hmac_signature = Some("not_hex_64_chars".to_string());
    let err = cert.validate().unwrap_err();
    assert!(
        matches!(err, CertificateError::InvalidHash { ref field, .. } if field == "hmac_signature"),
        "expected InvalidHash for hmac_signature, got {err:?}"
    );
}

/// Certificate with_smt_proof populates both fields.
#[test]
fn test_certificate_with_smt_proof() {
    let result = make_verification();
    let cert = ProofCertificate::from_verification(&result, make_input_spec())
        .with_smt_proof("(proof ...)".to_string(), SmtProofVerdict::Verified);
    assert_eq!(cert.smt_proof_alethe.as_deref(), Some("(proof ...)"));
    assert_eq!(cert.smt_proof_verdict, Some(SmtProofVerdict::Verified));
}

/// Certificate with_verifier_version populates the field.
#[test]
fn test_certificate_with_verifier_version() {
    let result = make_verification();
    let cert = ProofCertificate::from_verification(&result, make_input_spec())
        .with_verifier_version("NY-1.2.3".to_string());
    assert_eq!(cert.verifier_version.as_deref(), Some("NY-1.2.3"));
}

/// with_layer_bounds computes crown_coverage and ibp_fallback_count.
#[test]
fn test_certificate_with_layer_bounds_coverage() {
    let result = make_verification();
    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-1.0, 1.0)],
            output_bounds: vec![(-0.5, 0.5)],
            method: PropMethod::Crown, // tight
            node_name: None,
            input_sources: None,
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-0.5, 0.5)],
            output_bounds: vec![(0.0, 0.5)],
            method: PropMethod::Ibp, // not tight
            node_name: None,
            input_sources: None,
        },
        LayerBoundRecord {
            layer_index: 2,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(0.0, 0.5)],
            output_bounds: vec![(-0.3, 0.3)],
            method: PropMethod::AlphaCrown, // tight
            node_name: None,
            input_sources: None,
        },
    ];

    let cert =
        ProofCertificate::from_verification(&result, make_input_spec()).with_layer_bounds(bounds);

    // 2 tight out of 3 total -> coverage = 2/3
    assert!((cert.crown_coverage.unwrap() - 2.0 / 3.0).abs() < 1e-6);
    assert_eq!(cert.ibp_fallback_count, Some(1));
}

/// with_precision_model sets the field.
#[test]
fn test_certificate_with_precision_model() {
    let result = make_verification();
    let cert = ProofCertificate::from_verification(&result, make_input_spec())
        .with_precision_model(PrecisionModel::F16Aware {
            cast_count: 3,
            total_epsilon: 1e-4,
        });
    match cert.precision_model {
        Some(PrecisionModel::F16Aware {
            cast_count,
            total_epsilon,
        }) => {
            assert_eq!(cast_count, 3);
            assert!((total_epsilon - 1e-4).abs() < 1e-10);
        }
        other => panic!("expected F16Aware, got {other:?}"),
    }
}

/// with_kani_status populates the kani_status field.
#[test]
fn test_certificate_with_kani_status() {
    let result = make_verification();
    let kani = KaniProofRecord {
        harness_count: 5,
        status: KaniOutcome::Passed,
        properties: vec!["no_overflow".to_string(), "no_nan".to_string()],
        cbmc_version: Some("6.1.0".to_string()),
    };
    let cert =
        ProofCertificate::from_verification(&result, make_input_spec()).with_kani_status(kani);
    let ks = cert.kani_status.as_ref().unwrap();
    assert_eq!(ks.harness_count, 5);
    assert_eq!(ks.status, KaniOutcome::Passed);
    assert_eq!(ks.properties.len(), 2);
}

// ===========================================================================
// CertificateBundle tests
// ===========================================================================

/// Bundle filter_by_names returns correct subset.
#[test]
fn test_bundle_filter_by_names() {
    let mut v1 = make_verification();
    v1.kernel_name = "snake".to_string();
    let mut v2 = make_verification();
    v2.kernel_name = "silu_mul".to_string();
    let mut v3 = make_verification();
    v3.kernel_name = "relu".to_string();

    let bundle = CertificateBundle::new("test")
        .with_certificate(ProofCertificate::from_verification(&v1, make_input_spec()))
        .with_certificate(ProofCertificate::from_verification(&v2, make_input_spec()))
        .with_certificate(ProofCertificate::from_verification(&v3, make_input_spec()));

    let filtered = bundle.filter_by_names("subset", &["snake", "relu"]);
    assert_eq!(filtered.len(), 2);
    assert_eq!(filtered.model_name, "subset");
    assert_eq!(filtered.certificates[0].kernel_name, "snake");
    assert_eq!(filtered.certificates[1].kernel_name, "relu");
}

/// Bundle filter_by_names with empty names list returns empty bundle.
#[test]
fn test_bundle_filter_by_names_empty() {
    let v = make_verification();
    let bundle = CertificateBundle::new("test")
        .with_certificate(ProofCertificate::from_verification(&v, make_input_spec()));

    let filtered = bundle.filter_by_names("empty", &[]);
    assert!(filtered.is_empty());
}

/// Bundle all_sound returns true when all certificates are Sound.
#[test]
fn test_bundle_all_sound_true() {
    let v = make_verification();
    let bundle = CertificateBundle::new("test")
        .with_certificate(ProofCertificate::from_verification(&v, make_input_spec()))
        .with_certificate(ProofCertificate::from_verification(&v, make_input_spec()));
    assert!(bundle.all_sound());
}

/// Bundle all_sound returns false when one certificate is Heuristic.
#[test]
fn test_bundle_all_sound_false_with_heuristic() {
    let v1 = make_verification();
    let mut v2 = make_verification();
    v2.soundness_mode = VerificationSoundnessMode::Heuristic;

    let bundle = CertificateBundle::new("test")
        .with_certificate(ProofCertificate::from_verification(&v1, make_input_spec()))
        .with_certificate(ProofCertificate::from_verification(&v2, make_input_spec()));
    assert!(!bundle.all_sound());
}

/// Bundle all_have_source_hash returns false when any cert lacks source_hash.
#[test]
fn test_bundle_all_have_source_hash_false() {
    let v = make_verification();
    let bundle = CertificateBundle::new("test")
        .with_certificate(ProofCertificate::from_verification(&v, make_input_spec()));
    assert!(
        !bundle.all_have_source_hash(),
        "default certificates have no source_hash"
    );
}

/// Bundle all_have_source_hash returns true when all certs have source_hash.
#[test]
fn test_bundle_all_have_source_hash_true() {
    let v = make_verification();
    let valid_hash = "b".repeat(64);
    let cert =
        ProofCertificate::from_verification(&v, make_input_spec()).with_source_hash(valid_hash);
    let bundle = CertificateBundle::new("test").with_certificate(cert);
    assert!(bundle.all_have_source_hash());
}

/// certificate_from_pipeline produces valid certificates for single variable.
#[cfg(feature = "ny")]
#[test]
fn test_certificate_from_pipeline_validates() {
    use nn_verify::certificate::certificate_from_pipeline;
    let result = make_verification();
    let variable_inputs = vec![ParamInputRecord::new(0, -10.0, 10.0)];
    let cert = certificate_from_pipeline(&result, &variable_inputs, &[1.0], None);
    assert!(
        cert.validate().is_ok(),
        "certificate_from_pipeline should produce valid certificates"
    );
}

/// Certificate JSON roundtrip preserves all v2-v5 fields.
#[test]
fn test_certificate_json_roundtrip_all_fields() {
    let v = make_verification();
    let valid_hash = "c".repeat(64);
    let cert = ProofCertificate::from_verification(&v, make_input_spec())
        .with_smt_outcome("proven")
        .with_weight_hash(valid_hash.clone())
        .with_source_hash(valid_hash)
        .with_verifier_version("gc-2.0".to_string())
        .with_kani_status(KaniProofRecord {
            harness_count: 2,
            status: KaniOutcome::Passed,
            properties: vec!["no_nan".to_string()],
            cbmc_version: None,
        })
        .with_precision_model(PrecisionModel::F32Only)
        .with_layer_bounds(vec![LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Crown,
            node_name: Some("n0".to_string()),
            input_sources: Some(vec![]),
        }]);

    let json = cert.to_json().expect("serialize");
    let parsed: ProofCertificate = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(cert, parsed);
}

// ===========================================================================
// CertifyConfig tests
// ===========================================================================

#[cfg(feature = "ny")]
mod certify_config_tests {
    use nn_verify::certify::CertifyConfig;
    use nn_verify::SigningKey;

    #[test]
    fn test_certify_config_defaults() {
        let config = CertifyConfig::new("model");
        assert_eq!(config.model_name, "model");
        assert!((config.fusion_epsilon - 1e-5).abs() < 1e-10);
        assert_eq!(config.production_dim, 256);
        assert!(config.signing_key.is_none());
        assert!(config.enrichment.is_none());
    }

    #[test]
    fn test_certify_config_signing_key_raw() {
        let mut config = CertifyConfig::new("signed_model");
        config.signing_key = SigningKey::Raw(vec![42; 32]);
        assert!(!config.signing_key.is_none());
        assert_eq!(config.signing_key.as_bytes().unwrap().len(), 32);
    }
}

// ===========================================================================
// KaniOutcome and KaniProofRecord tests
// ===========================================================================

/// KaniOutcome serde roundtrip.
#[test]
fn test_kani_outcome_serde_roundtrip() {
    for outcome in [
        KaniOutcome::Passed,
        KaniOutcome::Failed,
        KaniOutcome::NotRun,
        KaniOutcome::Timeout,
    ] {
        let json = serde_json::to_string(&outcome).expect("serialize");
        let parsed: KaniOutcome = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(outcome, parsed);
    }
}

/// KaniProofRecord serde roundtrip.
#[test]
fn test_kani_proof_record_serde_roundtrip() {
    let record = KaniProofRecord {
        harness_count: 10,
        status: KaniOutcome::Passed,
        properties: vec!["no_overflow".to_string(), "bounds_preservation".to_string()],
        cbmc_version: Some("6.2.0".to_string()),
    };
    let json = serde_json::to_string_pretty(&record).expect("serialize");
    let parsed: KaniProofRecord = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(record, parsed);
}

// ===========================================================================
// PrecisionModel tests
// ===========================================================================

/// PrecisionModel default is F32Only.
#[test]
fn test_precision_model_default() {
    let model = PrecisionModel::default();
    assert_eq!(model, PrecisionModel::F32Only);
}

/// PrecisionModel serde roundtrip for both variants.
#[test]
fn test_precision_model_serde_roundtrip() {
    let models = [
        PrecisionModel::F32Only,
        PrecisionModel::F16Aware {
            cast_count: 4,
            total_epsilon: 0.001,
        },
    ];
    for model in &models {
        let json = serde_json::to_string(model).expect("serialize");
        let parsed: PrecisionModel = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(model, &parsed);
    }
}

// ===========================================================================
// PropMethod tests
// ===========================================================================

/// PropMethod::is_tight correctly classifies methods.
#[test]
fn test_prop_method_is_tight() {
    assert!(PropMethod::Crown.is_tight());
    assert!(PropMethod::AlphaCrown.is_tight());
    assert!(PropMethod::BetaCrown.is_tight());
    assert!(PropMethod::Analytical.is_tight());
    assert!(!PropMethod::Ibp.is_tight());
    assert!(!PropMethod::MixedIbpCrown.is_tight());
}

/// PropMethod serde roundtrip for all variants.
#[test]
fn test_prop_method_serde_roundtrip() {
    let methods = [
        PropMethod::Ibp,
        PropMethod::Crown,
        PropMethod::AlphaCrown,
        PropMethod::BetaCrown,
        PropMethod::Analytical,
        PropMethod::MixedIbpCrown,
    ];
    for method in &methods {
        let json = serde_json::to_string(method).expect("serialize");
        let parsed: PropMethod = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(method, &parsed);
    }
}

// ===========================================================================
// Certificate integrity tests (signing)
// ===========================================================================

/// Sign and verify a certificate roundtrip.
#[test]
fn test_sign_verify_certificate_roundtrip() {
    let v = make_verification();
    let mut cert = ProofCertificate::from_verification(&v, make_input_spec());
    let key: Vec<u8> = (0..32).collect();

    nn_verify::sign_certificate(&mut cert, &key).expect("sign");
    assert!(cert.content_hash.is_some());
    assert!(cert.hmac_signature.is_some());

    nn_verify::verify_signature(&cert, &key).expect("verify");
}

/// Verify with wrong key fails.
#[test]
fn test_verify_signature_wrong_key_fails() {
    let v = make_verification();
    let mut cert = ProofCertificate::from_verification(&v, make_input_spec());
    let sign_key: Vec<u8> = (0..32).collect();
    let wrong_key: Vec<u8> = (32..64).collect();

    nn_verify::sign_certificate(&mut cert, &sign_key).expect("sign");
    let err = nn_verify::verify_signature(&cert, &wrong_key);
    assert!(err.is_err(), "verification with wrong key should fail");
}

/// Content hash verification detects tampering.
#[test]
fn test_content_hash_detects_tampering() {
    let v = make_verification();
    let mut cert = ProofCertificate::from_verification(&v, make_input_spec());
    let key: Vec<u8> = (0..32).collect();

    nn_verify::sign_certificate(&mut cert, &key).expect("sign");

    // Tamper with the kernel_name
    cert.kernel_name = "tampered".to_string();

    let err = nn_verify::verify_content_hash(&cert);
    assert!(err.is_err(), "content hash should detect tampering");
}

/// Bundle signing signs all certificates.
#[test]
fn test_bundle_sign_and_verify() {
    let v = make_verification();
    let mut bundle = CertificateBundle::new("test")
        .with_certificate(ProofCertificate::from_verification(&v, make_input_spec()));
    let key: Vec<u8> = (0..32).collect();

    nn_verify::sign_bundle(&mut bundle, &key).expect("sign bundle");
    assert!(bundle.certificates[0].content_hash.is_some());

    nn_verify::verify_bundle_signatures(&bundle, &key).expect("verify bundle");
}

/// Strict bundle verification rejects unsigned certificates.
#[test]
fn test_bundle_strict_verify_rejects_unsigned() {
    let v = make_verification();
    let bundle = CertificateBundle::new("test")
        .with_certificate(ProofCertificate::from_verification(&v, make_input_spec()));
    let key: Vec<u8> = (0..32).collect();

    let err = nn_verify::verify_bundle_signatures_strict(&bundle, &key);
    assert!(
        err.is_err(),
        "strict verification should reject unsigned certificates"
    );
}

/// Non-strict bundle verification skips unsigned certificates.
#[test]
fn test_bundle_nonstrict_verify_skips_unsigned() {
    let v = make_verification();
    let bundle = CertificateBundle::new("test")
        .with_certificate(ProofCertificate::from_verification(&v, make_input_spec()));
    let key: Vec<u8> = (0..32).collect();

    // Non-strict should pass (unsigned certs are silently skipped)
    nn_verify::verify_bundle_signatures(&bundle, &key)
        .expect("non-strict should skip unsigned certs");
}

/// Bundle save/load roundtrip via temp file.
#[test]
fn test_bundle_save_load_roundtrip_integration() {
    let v = make_verification();
    let cert = ProofCertificate::from_verification(&v, make_input_spec())
        .with_smt_outcome("unexecuted")
        .with_verifier_version("gc-test".to_string());
    let bundle = CertificateBundle::new("roundtrip_integration").with_certificate(cert);

    let dir = std::env::temp_dir().join(format!("nn_cert_integ_test_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test_roundtrip.proof.json");
    let _ = std::fs::remove_file(&path);

    bundle.save(&path).expect("save");
    let loaded = CertificateBundle::load(&path).expect("load");
    assert_eq!(bundle, loaded);

    // Clean up
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}
