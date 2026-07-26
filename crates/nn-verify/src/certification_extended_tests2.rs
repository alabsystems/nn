// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for certification, gap detection, and compose verification
//! infrastructure.
//!
//! Covers: CertificateBuilder API validation, certificate field accessors,
//! VerificationLevel variants and ordering, GapDetector configuration,
//! gap report generation for empty/trivial cases, proof strength classification,
//! and soundness mode comparisons.

use crate::certificate::{
    CertificateBundle, CertificateError, ProofCertificate, CERTIFICATE_VERSION,
};
use crate::certificate_types::{
    compute_bytes_hash, KaniOutcome, KaniProofRecord, LayerBoundRecord, PrecisionModel,
};
use crate::gap_detector::{
    classify_entry, count_gaps_and_vacuous, detect_gaps, format_gap_report, kokoro_pipeline_stages,
    method_is_crown, StageGapResult,
};
use crate::soundness_compat::VerificationSoundnessMode;
use crate::status::{
    compute_proof_strength, InputBoundsRecord, ParamInputRecord, ProofStrength, SmtProofVerdict,
    VerifyOutcome, VACUOUS_WIDTH_THRESHOLD,
};
use crate::verify_types::{KernelVerification, OutputTensorBounds, PropMethod, VerifyConfig};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_verification(name: &str, lower: f32, upper: f32) -> KernelVerification {
    KernelVerification::new(
        name.to_string(),
        PropMethod::Ibp,
        lower,
        upper,
        upper - lower,
        lower.is_finite() && upper.is_finite(),
    )
}

fn make_input_spec(lower: f32, upper: f32) -> InputBoundsRecord {
    InputBoundsRecord {
        variable_inputs: vec![ParamInputRecord {
            param_index: 0,
            lower,
            upper,
        }],
        constant_params: vec![1.0],
        input_shape: Some(vec![1]),
        input_range: Some((lower, upper)),
    }
}

fn valid_sha256() -> String {
    "a".repeat(64)
}

// ===========================================================================
// 1. ProofCertificate builder API validation
// ===========================================================================

#[test]
fn test_certificate_builder_chain() {
    let result = make_verification("test_kernel", -5.0, 5.0);
    let input = make_input_spec(-10.0, 10.0);
    let cert = ProofCertificate::from_verification(&result, input)
        .with_smt_outcome("Proven")
        .with_layer_bounds(vec![LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Crown,
            node_name: Some("n0".to_string()),
            input_sources: Some(vec![]),
        }])
        .with_kani_status(KaniProofRecord {
            harness_count: 2,
            status: KaniOutcome::Passed,
            properties: vec!["no_overflow".to_string()],
            cbmc_version: None,
        })
        .with_weight_hash(valid_sha256())
        .with_source_hash(valid_sha256())
        .with_verifier_version("NY 0.9.0".to_string())
        .with_smt_proof("(proof ...)".to_string(), SmtProofVerdict::Verified)
        .with_precision_model(PrecisionModel::F16Aware {
            cast_count: 3,
            total_epsilon: 0.001,
        });

    assert!(
        cert.validate().is_ok(),
        "fully-populated certificate should validate"
    );
    assert_eq!(cert.smt_outcome.as_deref(), Some("Proven"));
    assert_eq!(cert.layer_bounds.as_ref().unwrap().len(), 1);
    assert_eq!(cert.kani_status.as_ref().unwrap().harness_count, 2);
    assert_eq!(cert.weight_hash.as_deref(), Some(valid_sha256().as_str()));
    assert_eq!(cert.source_hash.as_deref(), Some(valid_sha256().as_str()));
    assert_eq!(cert.verifier_version.as_deref(), Some("NY 0.9.0"));
    assert_eq!(cert.smt_proof_verdict, Some(SmtProofVerdict::Verified));
    assert!(cert.smt_proof_alethe.is_some());
    assert!(matches!(
        cert.precision_model,
        Some(PrecisionModel::F16Aware { cast_count: 3, .. })
    ));
}

#[test]
fn test_certificate_builder_defaults_are_none() {
    let result = make_verification("kernel_a", -1.0, 1.0);
    let cert = ProofCertificate::from_verification(&result, make_input_spec(-1.0, 1.0));

    assert!(cert.smt_outcome.is_none());
    assert!(cert.layer_bounds.is_none());
    assert!(cert.kani_status.is_none());
    assert!(cert.weight_hash.is_none());
    assert!(cert.source_hash.is_none());
    assert!(cert.verifier_version.is_none());
    assert!(cert.smt_proof_alethe.is_none());
    assert!(cert.smt_proof_verdict.is_none());
    assert!(cert.content_hash.is_none());
    assert!(cert.hmac_signature.is_none());
    assert!(cert.precision_model.is_none());
}

// ===========================================================================
// 2. Certificate field accessors and validation edge cases
// ===========================================================================

#[test]
fn test_certificate_version_is_current() {
    let result = make_verification("v_check", -1.0, 1.0);
    let cert = ProofCertificate::from_verification(&result, make_input_spec(-1.0, 1.0));
    assert_eq!(cert.version, CERTIFICATE_VERSION);
    assert!(cert.version >= 5, "current version should be at least 5");
}

#[test]
fn test_certificate_validate_version_zero_rejected() {
    let result = make_verification("zero_ver", -1.0, 1.0);
    let mut cert = ProofCertificate::from_verification(&result, make_input_spec(-1.0, 1.0));
    cert.version = 0;
    let err = cert.validate().unwrap_err();
    assert!(matches!(
        err,
        CertificateError::UnsupportedVersion { version: 0, .. }
    ));
}

#[test]
fn test_certificate_validate_future_version_rejected() {
    let result = make_verification("future_ver", -1.0, 1.0);
    let mut cert = ProofCertificate::from_verification(&result, make_input_spec(-1.0, 1.0));
    cert.version = CERTIFICATE_VERSION + 1;
    assert!(cert.validate().is_err());
}

#[test]
fn test_certificate_validate_invalid_weight_hash() {
    let result = make_verification("bad_hash", -1.0, 1.0);
    let cert = ProofCertificate::from_verification(&result, make_input_spec(-1.0, 1.0))
        .with_weight_hash("not_a_valid_sha256".to_string());
    let err = cert.validate().unwrap_err();
    assert!(
        matches!(err, CertificateError::InvalidHash { ref field, .. } if field == "weight_hash")
    );
}

#[test]
fn test_certificate_validate_invalid_source_hash() {
    let result = make_verification("bad_src", -1.0, 1.0);
    let cert = ProofCertificate::from_verification(&result, make_input_spec(-1.0, 1.0))
        .with_source_hash("aa".repeat(32)); // 64 chars of valid hex ('a') -> validates OK
    // Use genuinely invalid hex chars ('g' is not in [0-9a-f]):
    let cert2 = ProofCertificate::from_verification(
        &make_verification("bad_src2", -1.0, 1.0),
        make_input_spec(-1.0, 1.0),
    )
    .with_source_hash("g".repeat(64));
    let err = cert2.validate().unwrap_err();
    assert!(
        matches!(err, CertificateError::InvalidHash { ref field, .. } if field == "source_hash")
    );

    // Valid hex 64-char hash should pass.
    assert!(cert.validate().is_ok());
}

#[test]
fn test_certificate_validate_empty_layer_bounds() {
    let result = make_verification("empty_lb", -1.0, 1.0);
    let cert = ProofCertificate::from_verification(&result, make_input_spec(-1.0, 1.0))
        .with_layer_bounds(vec![]);
    let err = cert.validate().unwrap_err();
    assert!(matches!(err, CertificateError::EmptyLayerBounds));
}

#[test]
fn test_certificate_validate_layer_index_mismatch() {
    let result = make_verification("idx_mismatch", -1.0, 1.0);
    let cert = ProofCertificate::from_verification(&result, make_input_spec(-1.0, 1.0))
        .with_layer_bounds(vec![LayerBoundRecord {
            layer_index: 5, // should be 0
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-1.0, 1.0)],
            output_bounds: vec![(-1.0, 1.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: None,
        }]);
    let err = cert.validate().unwrap_err();
    assert!(matches!(
        err,
        CertificateError::LayerIndexMismatch {
            expected: 0,
            actual: 5
        }
    ));
}

#[test]
fn test_certificate_with_layer_bounds_computes_crown_coverage() {
    let result = make_verification("cov", -1.0, 1.0);
    let layers = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-1.0, 1.0)],
            output_bounds: vec![(-0.5, 0.5)],
            method: PropMethod::Crown, // tight
            node_name: None,
            input_sources: Some(vec![]),
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_bounds: vec![(-0.5, 0.5)],
            output_bounds: vec![(0.0, 0.5)],
            method: PropMethod::Ibp, // not tight
            node_name: None,
            input_sources: Some(vec![0]),
        },
    ];
    let cert = ProofCertificate::from_verification(&result, make_input_spec(-1.0, 1.0))
        .with_layer_bounds(layers);

    // 1 out of 2 layers is Crown (tight).
    assert_eq!(cert.crown_coverage, Some(0.5));
    assert_eq!(cert.ibp_fallback_count, Some(1));
}

#[test]
fn test_certificate_output_width_consistency() {
    let result = make_verification("width", -3.0, 7.0);
    let cert = ProofCertificate::from_verification(&result, make_input_spec(-10.0, 10.0));
    assert!((cert.output_width - 10.0).abs() < 1e-6);
    assert!(cert.validate().is_ok());
}

#[test]
fn test_certificate_output_width_mismatch_rejected() {
    let result = make_verification("width_bad", -3.0, 7.0);
    let mut cert = ProofCertificate::from_verification(&result, make_input_spec(-10.0, 10.0));
    cert.output_width = 999.0; // inconsistent with bounds
    let err = cert.validate().unwrap_err();
    assert!(matches!(err, CertificateError::OutputWidthMismatch { .. }));
}

#[test]
fn test_certificate_json_roundtrip() {
    let result = make_verification("roundtrip", -2.0, 3.0);
    let cert = ProofCertificate::from_verification(&result, make_input_spec(-5.0, 5.0))
        .with_smt_outcome("Unexecuted")
        .with_weight_hash(valid_sha256());

    let json = cert.to_json().expect("serialize");
    let deserialized: ProofCertificate = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.kernel_name, "roundtrip");
    assert_eq!(deserialized.smt_outcome.as_deref(), Some("Unexecuted"));
    assert_eq!(deserialized.weight_hash, cert.weight_hash);
    assert_eq!(deserialized.version, CERTIFICATE_VERSION);
}

// ===========================================================================
// 3. CertificateBundle API
// ===========================================================================

#[test]
fn test_bundle_new_is_empty() {
    let bundle = CertificateBundle::new("test_model");
    assert!(bundle.is_empty());
    assert_eq!(bundle.len(), 0);
    assert_eq!(bundle.model_name, "test_model");
    assert_eq!(bundle.version, CERTIFICATE_VERSION);
    assert!(!bundle.generated_at.is_empty());
}

#[test]
fn test_bundle_with_certificate_builder() {
    let cert1 = ProofCertificate::from_verification(
        &make_verification("k1", -1.0, 1.0),
        make_input_spec(-1.0, 1.0),
    );
    let cert2 = ProofCertificate::from_verification(
        &make_verification("k2", -2.0, 2.0),
        make_input_spec(-2.0, 2.0),
    );

    let bundle = CertificateBundle::new("model")
        .with_certificate(cert1)
        .with_certificate(cert2);

    assert_eq!(bundle.len(), 2);
    assert!(!bundle.is_empty());
    assert_eq!(bundle.certificates[0].kernel_name, "k1");
    assert_eq!(bundle.certificates[1].kernel_name, "k2");
}

#[test]
fn test_bundle_push() {
    let mut bundle = CertificateBundle::new("model");
    let cert = ProofCertificate::from_verification(
        &make_verification("pushed", -1.0, 1.0),
        make_input_spec(-1.0, 1.0),
    );
    bundle.push(cert);
    assert_eq!(bundle.len(), 1);
    assert_eq!(bundle.certificates[0].kernel_name, "pushed");
}

#[test]
fn test_bundle_verified_count() {
    let finite_result = make_verification("finite", -1.0, 1.0);
    let mut non_finite_result = make_verification("non_finite", -1.0, 1.0);
    non_finite_result.is_finite = false;

    let bundle = CertificateBundle::new("model")
        .with_certificate(ProofCertificate::from_verification(
            &finite_result,
            make_input_spec(-1.0, 1.0),
        ))
        .with_certificate(ProofCertificate::from_verification(
            &non_finite_result,
            make_input_spec(-1.0, 1.0),
        ));

    assert_eq!(bundle.verified_count(), 1);
}

#[test]
fn test_bundle_sound_count() {
    let sound_result = make_verification("sound", -1.0, 1.0);
    let heuristic_result = make_verification("heuristic", -1.0, 1.0)
        .with_soundness_mode(VerificationSoundnessMode::Heuristic);

    let bundle = CertificateBundle::new("model")
        .with_certificate(ProofCertificate::from_verification(
            &sound_result,
            make_input_spec(-1.0, 1.0),
        ))
        .with_certificate(ProofCertificate::from_verification(
            &heuristic_result,
            make_input_spec(-1.0, 1.0),
        ));

    assert_eq!(bundle.sound_count(), 1);
}

#[test]
fn test_bundle_validate_all_ok() {
    let cert = ProofCertificate::from_verification(
        &make_verification("ok", -1.0, 1.0),
        make_input_spec(-1.0, 1.0),
    );
    let bundle = CertificateBundle::new("model").with_certificate(cert);
    assert!(bundle.validate_all().is_ok());
}

#[test]
fn test_bundle_validate_all_reports_index() {
    let mut bad_cert = ProofCertificate::from_verification(
        &make_verification("bad", -1.0, 1.0),
        make_input_spec(-1.0, 1.0),
    );
    bad_cert.version = 0; // invalid

    let good_cert = ProofCertificate::from_verification(
        &make_verification("good", -1.0, 1.0),
        make_input_spec(-1.0, 1.0),
    );

    let bundle = CertificateBundle::new("model")
        .with_certificate(good_cert)
        .with_certificate(bad_cert);

    let err = bundle.validate_all().unwrap_err();
    assert_eq!(err.0, 1, "error should be at index 1 (second certificate)");
}

#[test]
fn test_bundle_filter_by_names() {
    let c1 = ProofCertificate::from_verification(
        &make_verification("snake", -1.0, 1.0),
        make_input_spec(-1.0, 1.0),
    );
    let c2 = ProofCertificate::from_verification(
        &make_verification("relu", -1.0, 1.0),
        make_input_spec(-1.0, 1.0),
    );
    let c3 = ProofCertificate::from_verification(
        &make_verification("softmax", -1.0, 1.0),
        make_input_spec(-1.0, 1.0),
    );

    let bundle = CertificateBundle::new("full")
        .with_certificate(c1)
        .with_certificate(c2)
        .with_certificate(c3);

    let filtered = bundle.filter_by_names("sub", &["snake", "softmax"]);
    assert_eq!(filtered.len(), 2);
    assert_eq!(filtered.model_name, "sub");
    assert_eq!(filtered.certificates[0].kernel_name, "snake");
    assert_eq!(filtered.certificates[1].kernel_name, "softmax");
}

#[test]
fn test_bundle_all_sound() {
    let sound = ProofCertificate::from_verification(
        &make_verification("s", -1.0, 1.0),
        make_input_spec(-1.0, 1.0),
    );
    let heuristic = ProofCertificate::from_verification(
        &make_verification("h", -1.0, 1.0)
            .with_soundness_mode(VerificationSoundnessMode::Heuristic),
        make_input_spec(-1.0, 1.0),
    );

    let all_sound = CertificateBundle::new("m").with_certificate(sound.clone());
    assert!(all_sound.all_sound());

    let mixed = CertificateBundle::new("m")
        .with_certificate(sound)
        .with_certificate(heuristic);
    assert!(!mixed.all_sound());
}

// ===========================================================================
// 4. PropMethod variants and is_tight ordering
// ===========================================================================

#[test]
fn test_prop_method_is_tight() {
    // Tight methods
    assert!(PropMethod::Crown.is_tight());
    assert!(PropMethod::AlphaCrown.is_tight());
    assert!(PropMethod::BetaCrown.is_tight());
    assert!(PropMethod::Analytical.is_tight());

    // Loose methods
    assert!(!PropMethod::Ibp.is_tight());
    assert!(!PropMethod::MixedIbpCrown.is_tight());
}

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
        let back: PropMethod = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(*method, back, "roundtrip failed for {method:?}");
    }
}

// ===========================================================================
// 5. VerifyConfig API
// ===========================================================================

#[test]
fn test_verify_config_default() {
    let config = VerifyConfig::default();
    assert!(!config.require_sound());
    assert!(!config.collect_layer_bounds());
    assert!(config.escalation_threshold().is_finite());
    assert!(config.escalation_threshold() > 0.0);
}

#[test]
fn test_verify_config_invalid_threshold() {
    assert!(VerifyConfig::with_threshold(f32::NAN).is_err());
    assert!(VerifyConfig::with_threshold(f32::INFINITY).is_err());
    assert!(VerifyConfig::with_threshold(-1.0).is_err());
    assert!(VerifyConfig::with_threshold(0.0).is_ok());
    assert!(VerifyConfig::with_threshold(100.0).is_ok());
}

#[test]
fn test_verify_config_builder_chain() {
    let config = VerifyConfig::with_threshold(500.0)
        .unwrap()
        .with_require_sound(true)
        .with_collect_layer_bounds(true);
    assert_eq!(config.escalation_threshold(), 500.0);
    assert!(config.require_sound());
    assert!(config.collect_layer_bounds());
}

// ===========================================================================
// 6. GapDetector classify_entry
// ===========================================================================

#[test]
fn test_classify_entry_ibp_only() {
    let (has_ibp, has_crown, has_analytical, is_vacuous, has_any) =
        classify_entry(true, false, "IBP", "", None, Some("sound"));
    assert!(has_ibp);
    assert!(!has_crown);
    assert!(!has_analytical);
    assert!(!is_vacuous);
    assert!(has_any);
}

#[test]
fn test_classify_entry_crown_primary() {
    let (has_ibp, has_crown, has_analytical, is_vacuous, has_any) =
        classify_entry(true, false, "CROWN", "", None, Some("sound"));
    assert!(!has_ibp);
    assert!(has_crown);
    assert!(!has_analytical);
    assert!(!is_vacuous);
    assert!(has_any);
}

#[test]
fn test_classify_entry_crown_suffix_with_ibp_is_not_crown() {
    // Crown suffix entry exists but uses IBP method.
    let (has_ibp, has_crown, _, _, _) = classify_entry(true, true, "IBP", "IBP", None, None);
    assert!(has_ibp);
    assert!(!has_crown);
}

#[test]
fn test_classify_entry_analytical() {
    let (has_ibp, has_crown, has_analytical, _, has_any) =
        classify_entry(true, false, "ANALYTICAL", "", None, Some("sound"));
    assert!(!has_ibp);
    assert!(!has_crown);
    assert!(has_analytical);
    assert!(has_any);
}

#[test]
fn test_classify_entry_vacuous_by_width() {
    let (_, _, _, is_vacuous, _) =
        classify_entry(true, false, "IBP", "", Some(5000.0), Some("heuristic"));
    assert!(is_vacuous);
}

#[test]
fn test_classify_entry_vacuous_by_label() {
    let (_, _, _, is_vacuous, _) =
        classify_entry(true, false, "IBP", "", Some(50.0), Some("vacuous"));
    assert!(
        is_vacuous,
        "proof_strength='vacuous' should mark as vacuous regardless of width"
    );
}

#[test]
fn test_classify_entry_not_vacuous_at_threshold() {
    // Width exactly at 1000.0 is NOT vacuous (> not >=).
    let (_, _, _, is_vacuous, _) =
        classify_entry(true, false, "IBP", "", Some(1000.0), Some("heuristic"));
    assert!(!is_vacuous);
}

#[test]
fn test_classify_entry_no_entries_is_gap() {
    let (has_ibp, has_crown, has_analytical, is_vacuous, has_any) =
        classify_entry(false, false, "", "", None, None);
    assert!(!has_ibp);
    assert!(!has_crown);
    assert!(!has_analytical);
    assert!(!is_vacuous);
    assert!(!has_any);
}

#[test]
fn test_classify_entry_alpha_crown_variants() {
    let variants = ["AlphaCROWN", "ALPHA-CROWN", "BetaCROWN", "BETA-CROWN"];
    for variant in &variants {
        let (_, has_crown, _, _, _) = classify_entry(false, true, "", variant, None, Some("sound"));
        assert!(
            has_crown,
            "CROWN variant '{variant}' should be detected as CROWN"
        );
    }
}

#[test]
fn test_classify_entry_mixed_ibp_crown() {
    let (_, has_crown, _, _, _) =
        classify_entry(true, false, "MIXED_IBP_CROWN", "", None, Some("sound"));
    assert!(has_crown, "MIXED_IBP_CROWN should count as CROWN");

    let (_, has_crown2, _, _, _) =
        classify_entry(true, false, "MIXED-IBP-CROWN", "", None, Some("sound"));
    assert!(has_crown2, "MIXED-IBP-CROWN should count as CROWN");
}

// ===========================================================================
// 7. method_is_crown
// ===========================================================================

#[test]
fn test_method_is_crown_positive_cases() {
    assert!(method_is_crown("CROWN"));
    assert!(method_is_crown("AlphaCROWN"));
    assert!(method_is_crown("ALPHA-CROWN"));
    assert!(method_is_crown("BetaCROWN"));
    assert!(method_is_crown("BETA-CROWN"));
    assert!(method_is_crown("MIXED_IBP_CROWN"));
    assert!(method_is_crown("MIXED-IBP-CROWN"));
}

#[test]
fn test_method_is_crown_negative_cases() {
    assert!(!method_is_crown("IBP"));
    assert!(!method_is_crown("ANALYTICAL"));
    assert!(!method_is_crown(""));
    assert!(!method_is_crown("crown_ish"));
    assert!(!method_is_crown("NotCROWN"));
}

#[test]
fn test_method_is_crown_case_insensitive() {
    assert!(method_is_crown("crown"));
    assert!(method_is_crown("Crown"));
    assert!(method_is_crown("alphacrown"));
}

#[test]
fn test_method_is_crown_trims_whitespace() {
    assert!(method_is_crown("  CROWN  "));
    assert!(method_is_crown("\tAlphaCROWN\n"));
}

// ===========================================================================
// 8. count_gaps_and_vacuous
// ===========================================================================

#[test]
fn test_count_gaps_empty_slice() {
    let (gaps, vacuous) = count_gaps_and_vacuous(&[]);
    assert_eq!(gaps, 0);
    assert_eq!(vacuous, 0);
}

#[test]
fn test_count_gaps_all_verified() {
    let stages = kokoro_pipeline_stages();
    let results: Vec<StageGapResult> = stages
        .into_iter()
        .map(|stage| StageGapResult {
            stage,
            has_ibp_bounds: true,
            has_crown_bounds: false,
            has_analytical_bounds: false,
            is_vacuous: false,
            bound_width: Some(5.0),
            proof_strength: Some("sound".to_string()),
            soundness_mode: Some("sound".to_string()),
            has_constructive_certificate: false,
        })
        .collect();
    let (gaps, vacuous) = count_gaps_and_vacuous(&results);
    assert_eq!(gaps, 0);
    assert_eq!(vacuous, 0);
}

// ===========================================================================
// 9. Gap report generation for empty/trivial cases
// ===========================================================================

#[test]
fn test_detect_gaps_empty_json() {
    let status = serde_json::json!({});
    let report = detect_gaps(&status);
    assert_eq!(report.stages.len(), 8);
    assert_eq!(report.total_gaps, 8);
    assert_eq!(report.vacuous_count, 0);
}

#[test]
fn test_detect_gaps_null_kernels() {
    let status = serde_json::json!({ "kernels": null });
    let report = detect_gaps(&status);
    assert_eq!(report.total_gaps, 8);
}

#[test]
fn test_detect_gaps_empty_kernels() {
    let status = serde_json::json!({ "kernels": {} });
    let report = detect_gaps(&status);
    assert_eq!(report.total_gaps, 8);
    assert_eq!(report.vacuous_count, 0);
}

#[test]
fn test_format_gap_report_empty_status() {
    let status = serde_json::json!({ "kernels": {} });
    let report = detect_gaps(&status);
    let formatted = format_gap_report(&report);

    assert!(formatted.contains("Summary:"));
    assert!(formatted.contains("8 gaps"));
    assert!(
        formatted.contains("[!!]"),
        "all stages should show GAP marker"
    );
}

#[test]
fn test_format_gap_report_contains_stage_names() {
    let status = serde_json::json!({ "kernels": {} });
    let report = detect_gaps(&status);
    let formatted = format_gap_report(&report);

    // Every pipeline stage name should appear in the report.
    let stages = kokoro_pipeline_stages();
    for stage in &stages {
        assert!(
            formatted.contains(stage.name),
            "report should contain stage name: {}",
            stage.name
        );
    }
}

// ===========================================================================
// 10. Proof strength classification
// ===========================================================================

#[test]
fn test_proof_strength_sound_crown() {
    let ps = compute_proof_strength(VerificationSoundnessMode::Sound, PropMethod::Crown, 5.0);
    assert_eq!(ps, ProofStrength::SoundCrown);
}

#[test]
fn test_proof_strength_sound_alpha_crown() {
    let ps = compute_proof_strength(
        VerificationSoundnessMode::Sound,
        PropMethod::AlphaCrown,
        5.0,
    );
    assert_eq!(ps, ProofStrength::SoundCrown);
}

#[test]
fn test_proof_strength_sound_beta_crown() {
    let ps = compute_proof_strength(VerificationSoundnessMode::Sound, PropMethod::BetaCrown, 5.0);
    assert_eq!(ps, ProofStrength::SoundCrown);
}

#[test]
fn test_proof_strength_sound_analytical() {
    let ps = compute_proof_strength(
        VerificationSoundnessMode::Sound,
        PropMethod::Analytical,
        5.0,
    );
    assert_eq!(ps, ProofStrength::SoundCrown);
}

#[test]
fn test_proof_strength_sound_ibp() {
    let ps = compute_proof_strength(VerificationSoundnessMode::Sound, PropMethod::Ibp, 5.0);
    assert_eq!(ps, ProofStrength::SoundIbp);
}

#[test]
fn test_proof_strength_sound_mixed() {
    let ps = compute_proof_strength(
        VerificationSoundnessMode::Sound,
        PropMethod::MixedIbpCrown,
        5.0,
    );
    assert_eq!(ps, ProofStrength::SoundMixed);
}

#[test]
fn test_proof_strength_heuristic() {
    let ps = compute_proof_strength(VerificationSoundnessMode::Heuristic, PropMethod::Crown, 5.0);
    assert_eq!(ps, ProofStrength::Heuristic);
}

#[test]
fn test_proof_strength_vacuous_overrides_sound() {
    let ps = compute_proof_strength(
        VerificationSoundnessMode::Sound,
        PropMethod::Crown,
        VACUOUS_WIDTH_THRESHOLD + 1.0,
    );
    assert_eq!(ps, ProofStrength::Vacuous);
}

#[test]
fn test_proof_strength_vacuous_overrides_heuristic() {
    let ps = compute_proof_strength(
        VerificationSoundnessMode::Heuristic,
        PropMethod::Ibp,
        VACUOUS_WIDTH_THRESHOLD + 1.0,
    );
    assert_eq!(ps, ProofStrength::Vacuous);
}

#[test]
fn test_proof_strength_at_vacuous_threshold_is_not_vacuous() {
    let ps = compute_proof_strength(
        VerificationSoundnessMode::Sound,
        PropMethod::Crown,
        VACUOUS_WIDTH_THRESHOLD,
    );
    assert_eq!(
        ps,
        ProofStrength::SoundCrown,
        "width exactly at threshold should not be vacuous (> not >=)"
    );
}

#[test]
fn test_proof_strength_serde_roundtrip() {
    let strengths = [
        ProofStrength::SoundCrown,
        ProofStrength::SoundIbp,
        ProofStrength::Heuristic,
        ProofStrength::Vacuous,
        ProofStrength::SoundMixed,
    ];
    for ps in &strengths {
        let json = serde_json::to_string(ps).expect("serialize");
        let back: ProofStrength = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(*ps, back, "roundtrip failed for {ps:?}");
    }
}

// ===========================================================================
// 11. Soundness mode comparisons
// ===========================================================================

#[test]
fn test_soundness_mode_sound_vs_heuristic() {
    assert_ne!(
        VerificationSoundnessMode::Sound,
        VerificationSoundnessMode::Heuristic
    );
}

#[test]
fn test_soundness_mode_equality() {
    assert_eq!(
        VerificationSoundnessMode::Sound,
        VerificationSoundnessMode::Sound
    );
    assert_eq!(
        VerificationSoundnessMode::Heuristic,
        VerificationSoundnessMode::Heuristic
    );
}

#[test]
fn test_soundness_mode_serde_roundtrip() {
    for mode in &[
        VerificationSoundnessMode::Sound,
        VerificationSoundnessMode::Heuristic,
    ] {
        let json = serde_json::to_string(mode).expect("serialize");
        let back: VerificationSoundnessMode = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(*mode, back);
    }
}

#[test]
fn test_default_soundness_mode_is_heuristic() {
    // default_soundness_mode is fail-closed: old results without the field
    // are treated as Heuristic.
    let mode = crate::soundness_compat::default_soundness_mode();
    assert_eq!(mode, VerificationSoundnessMode::Heuristic);
}

// ===========================================================================
// 12. KaniOutcome and KaniProofRecord
// ===========================================================================

#[test]
fn test_kani_outcome_variants() {
    assert_ne!(KaniOutcome::Passed, KaniOutcome::Failed);
    assert_ne!(KaniOutcome::Passed, KaniOutcome::NotRun);
    assert_ne!(KaniOutcome::Passed, KaniOutcome::Timeout);
}

#[test]
fn test_kani_proof_record_serde_roundtrip() {
    let record = KaniProofRecord {
        harness_count: 42,
        status: KaniOutcome::Passed,
        properties: vec!["no_overflow".to_string(), "bounds_preservation".to_string()],
        cbmc_version: Some("6.1.0".to_string()),
    };
    let json = serde_json::to_string(&record).expect("serialize");
    let back: KaniProofRecord = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.harness_count, 42);
    assert_eq!(back.status, KaniOutcome::Passed);
    assert_eq!(back.properties.len(), 2);
    assert_eq!(back.cbmc_version.as_deref(), Some("6.1.0"));
}

// ===========================================================================
// 13. PrecisionModel
// ===========================================================================

#[test]
fn test_precision_model_default_is_f32_only() {
    let pm = PrecisionModel::default();
    assert_eq!(pm, PrecisionModel::F32Only);
}

#[test]
fn test_precision_model_f16_aware() {
    let pm = PrecisionModel::F16Aware {
        cast_count: 5,
        total_epsilon: 0.01,
    };
    assert!(matches!(
        pm,
        PrecisionModel::F16Aware {
            cast_count: 5,
            total_epsilon: e
        } if (e - 0.01).abs() < 1e-6
    ));
}

#[test]
fn test_precision_model_serde_roundtrip() {
    let models = [
        PrecisionModel::F32Only,
        PrecisionModel::F16Aware {
            cast_count: 3,
            total_epsilon: 0.005,
        },
    ];
    for pm in &models {
        let json = serde_json::to_string(pm).expect("serialize");
        let back: PrecisionModel = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(*pm, back, "roundtrip failed for {pm:?}");
    }
}

// ===========================================================================
// 14. OutputTensorBounds
// ===========================================================================

#[test]
fn test_output_tensor_bounds_new() {
    let otb = OutputTensorBounds::new(vec![-1.0, 0.0], vec![1.0, 2.0], vec![2]);
    assert_eq!(otb.lower, vec![-1.0, 0.0]);
    assert_eq!(otb.upper, vec![1.0, 2.0]);
    assert_eq!(otb.shape, vec![2]);
    assert!(
        otb.finite_mask.is_empty(),
        "new() should not set finite_mask"
    );
}

// ===========================================================================
// 15. VerifyOutcome serde
// ===========================================================================

#[test]
fn test_verify_outcome_serde_roundtrip() {
    let outcomes = [
        VerifyOutcome::Verified,
        VerifyOutcome::BoundsComputed,
        VerifyOutcome::IbpFallback,
        VerifyOutcome::Failed,
        VerifyOutcome::SmtContradiction,
    ];
    for outcome in &outcomes {
        let json = serde_json::to_string(outcome).expect("serialize");
        let back: VerifyOutcome = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(*outcome, back, "roundtrip failed for {outcome:?}");
    }
}

// ===========================================================================
// 16. compute_bytes_hash
// ===========================================================================

#[test]
fn test_compute_bytes_hash_deterministic() {
    let h1 = compute_bytes_hash(b"hello world");
    let h2 = compute_bytes_hash(b"hello world");
    assert_eq!(h1, h2);
    assert_eq!(h1.len(), 64, "SHA-256 hex digest should be 64 chars");
    assert!(h1.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn test_compute_bytes_hash_different_inputs() {
    let h1 = compute_bytes_hash(b"input_a");
    let h2 = compute_bytes_hash(b"input_b");
    assert_ne!(h1, h2);
}

#[test]
fn test_compute_bytes_hash_empty_input() {
    let h = compute_bytes_hash(b"");
    // SHA-256 of empty string is well-known.
    assert_eq!(
        h,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

// ===========================================================================
// 17. KernelVerification builder
// ===========================================================================

#[test]
fn test_kernel_verification_new_defaults() {
    let kv = KernelVerification::new("test".to_string(), PropMethod::Ibp, -1.0, 1.0, 2.0, true);
    assert_eq!(kv.kernel_name, "test");
    assert_eq!(kv.method, PropMethod::Ibp);
    assert_eq!(kv.output_lower, -1.0);
    assert_eq!(kv.output_upper, 1.0);
    assert_eq!(kv.output_width, 2.0);
    assert!(kv.is_finite);
    assert!(kv.crown_fallback_reason.is_none());
    assert_eq!(kv.soundness_mode, VerificationSoundnessMode::Sound);
    assert!(kv.output_tensor.is_none());
}

#[test]
fn test_kernel_verification_with_crown_fallback() {
    let kv = KernelVerification::new("test".to_string(), PropMethod::Ibp, -1.0, 1.0, 2.0, true)
        .with_crown_fallback_reason(Some("matrix too large".to_string()));
    assert_eq!(
        kv.crown_fallback_reason.as_deref(),
        Some("matrix too large")
    );
}

#[test]
fn test_kernel_verification_with_soundness_mode() {
    let kv = KernelVerification::new("test".to_string(), PropMethod::Ibp, -1.0, 1.0, 2.0, true)
        .with_soundness_mode(VerificationSoundnessMode::Heuristic);
    assert_eq!(kv.soundness_mode, VerificationSoundnessMode::Heuristic);
}

// ===========================================================================
// 18. Pipeline stage registry
// ===========================================================================

#[test]
fn test_pipeline_stages_have_unique_status_keys() {
    let stages = kokoro_pipeline_stages();
    let mut keys: Vec<&str> = stages.iter().map(|s| s.status_key).collect();
    let original_len = keys.len();
    keys.sort_unstable();
    keys.dedup();
    assert_eq!(
        keys.len(),
        original_len,
        "all status_key values must be unique"
    );
}

#[test]
fn test_pipeline_stages_have_unique_names() {
    let stages = kokoro_pipeline_stages();
    let mut names: Vec<&str> = stages.iter().map(|s| s.name).collect();
    let original_len = names.len();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), original_len, "all stage names must be unique");
}

#[test]
fn test_pipeline_stages_no_overlap_compiled_bridge() {
    let stages = kokoro_pipeline_stages();
    for stage in &stages {
        assert!(
            !(stage.is_compiled_segment && stage.is_bridge),
            "stage '{}' is both compiled and bridge",
            stage.name
        );
    }
}

// ===========================================================================
// 19. SmtProofVerdict
// ===========================================================================

#[test]
fn test_smt_proof_verdict_serde_roundtrip() {
    let verdicts = [SmtProofVerdict::Verified, SmtProofVerdict::Unchecked];
    for v in &verdicts {
        let json = serde_json::to_string(v).expect("serialize");
        let back: SmtProofVerdict = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(*v, back, "roundtrip failed for {v:?}");
    }
}
