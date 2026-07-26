// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for verification infrastructure: gap detector, certification
//! pipeline, verification status, soundness classification, and proof strength.
//!
//! 70+ tests covering:
//! - Gap detector configuration and classify_entry decision logic
//! - Pipeline stage registry invariants
//! - Certification pipeline stages (ProofCertificate, CertificateBundle)
//! - Verification status JSON parsing and VerifyStatus API
//! - Soundness mode classification
//! - Proof strength calculation edge cases
//! - StatusReport aggregation
//! - SmtStatusRecord and SmtOutcome semantics
//! - ConstructiveProofData validation and replay
//! - Certificate checker types

use crate::certificate::{
    CertificateBundle, CertificateError, ProofCertificate,
};
use crate::certificate_types::{
    compute_bytes_hash, ConstructiveLayerRecord, ConstructiveProofData, ConstructiveProofMethod, LayerBoundRecord,
};
use crate::gap_detector::{
    classify_entry, count_gaps_and_vacuous, detect_gaps, format_gap_report, kokoro_pipeline_stages,
    method_is_crown, PipelineStage, StageGapResult,
};
use crate::soundness_compat::VerificationSoundnessMode;
use crate::status::{
    compute_proof_strength, InputBoundsRecord, KernelStatus, OutputBoundsRecord, ParamInputRecord,
    ProofStrength, VerifyOutcome, VerifyStatus, VACUOUS_WIDTH_THRESHOLD,
};
use crate::status_report::{GapSummary, StatusReport, Trend, VerificationBreakdown};
use crate::status_smt::{
    BoundsSource, SmtEncodingKind, SmtOutcome, SmtStatusRecord,
};
use crate::verify_types::{
    KernelVerification, NormBoundsMode, PropMethod,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_verification(name: &str, method: PropMethod, lower: f32, upper: f32) -> KernelVerification {
    KernelVerification::new(
        name.to_string(),
        method,
        lower,
        upper,
        upper - lower,
        lower.is_finite() && upper.is_finite(),
    )
}

fn make_input_spec(lower: f32, upper: f32) -> InputBoundsRecord {
    InputBoundsRecord::new(&[ParamInputRecord::new(0, lower, upper)], &[1.0])
}

fn make_kernel_status(
    soundness: VerificationSoundnessMode,
    method: PropMethod,
    output_width: f32,
) -> KernelStatus {
    KernelStatus::new(
        VerifyOutcome::Verified,
        method,
        make_input_spec(-1.0, 1.0),
        OutputBoundsRecord::new(-output_width / 2.0, output_width / 2.0),
        output_width,
        soundness,
    )
}

fn make_gap_json_entry(
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
// 1. Gap detector — classify_entry decision matrix
// ===========================================================================

#[test]
fn test_classify_entry_both_primary_and_crown_ibp_method_counts_ibp() {
    // Both entries exist and both use IBP — should count as IBP, not CROWN.
    let (has_ibp, has_crown, _, _, has_any) =
        classify_entry(true, true, "IBP", "IBP", None, Some("sound"));
    assert!(has_ibp);
    assert!(!has_crown);
    assert!(has_any);
}

#[test]
fn test_classify_entry_empty_method_primary_valid() {
    // Primary valid with empty method string — treated as IBP.
    let (has_ibp, has_crown, has_analytical, _, has_any) =
        classify_entry(true, false, "", "", None, None);
    assert!(has_ibp, "empty method on valid primary should count as IBP");
    assert!(!has_crown);
    assert!(!has_analytical);
    assert!(has_any);
}

#[test]
fn test_classify_entry_crown_in_crown_suffix_entry() {
    // Crown suffix entry has CROWN method — should be detected.
    let (_, has_crown, _, _, _) = classify_entry(false, true, "", "CROWN", None, Some("sound"));
    assert!(has_crown);
}

#[test]
fn test_classify_entry_width_none_not_vacuous() {
    // No width data — should not be vacuous.
    let (_, _, _, is_vacuous, _) = classify_entry(true, false, "IBP", "", None, None);
    assert!(!is_vacuous);
}

#[test]
fn test_classify_entry_vacuous_proof_strength_overrides_narrow_width() {
    // proof_strength="vacuous" should make it vacuous even with narrow width.
    let (_, _, _, is_vacuous, _) =
        classify_entry(true, false, "IBP", "", Some(1.0), Some("vacuous"));
    assert!(is_vacuous);
}

#[test]
fn test_classify_entry_analytical_in_crown_suffix() {
    // ANALYTICAL method in the crown suffix entry.
    let (_, _, has_analytical, _, has_any) =
        classify_entry(false, true, "", "ANALYTICAL", None, Some("sound"));
    assert!(has_analytical);
    assert!(has_any);
}

#[test]
fn test_classify_entry_neither_entry_valid() {
    // Both entries invalid.
    let (has_ibp, has_crown, has_analytical, is_vacuous, has_any) =
        classify_entry(false, false, "CROWN", "CROWN", Some(1.0), Some("sound"));
    assert!(!has_ibp);
    assert!(!has_crown);
    assert!(!has_analytical);
    assert!(!is_vacuous);
    assert!(!has_any);
}

// ===========================================================================
// 2. method_is_crown — comprehensive coverage
// ===========================================================================

#[test]
fn test_method_is_crown_mixed_variants() {
    // Underscore and hyphen variants.
    assert!(method_is_crown("MIXED_IBP_CROWN"));
    assert!(method_is_crown("MIXED-IBP-CROWN"));
    assert!(method_is_crown("mixed_ibp_crown")); // lowercase
    assert!(method_is_crown("Mixed-IBP-Crown")); // mixed case
}

#[test]
fn test_method_is_crown_partial_match_rejected() {
    // These should NOT match even though they contain "CROWN".
    assert!(!method_is_crown("NN_CROWN"));
    assert!(!method_is_crown("CROWN_V2"));
    assert!(!method_is_crown("NOTCROWN"));
    assert!(!method_is_crown("PRE-CROWN"));
}

// ===========================================================================
// 3. count_gaps_and_vacuous
// ===========================================================================

#[test]
fn test_count_gaps_mixed_results() {
    let stages = kokoro_pipeline_stages();
    let mut results = Vec::new();

    // First 3 stages: verified with bounds, one vacuous.
    for (i, stage) in stages.into_iter().enumerate() {
        results.push(StageGapResult {
            stage,
            has_ibp_bounds: i < 3,
            has_crown_bounds: false,
            has_analytical_bounds: false,
            is_vacuous: i == 1,
            bound_width: if i < 3 { Some(50.0) } else { None },
            proof_strength: None,
            soundness_mode: None,
            has_constructive_certificate: false,
        });
    }

    let (gaps, vacuous) = count_gaps_and_vacuous(&results);
    assert_eq!(gaps, 5, "8 stages - 3 with bounds = 5 gaps");
    assert_eq!(vacuous, 1);
}

// ===========================================================================
// 4. Pipeline stage registry
// ===========================================================================

#[test]
fn test_pipeline_stages_status_keys_start_with_kokoro() {
    let stages = kokoro_pipeline_stages();
    for stage in &stages {
        assert!(
            stage.status_key.starts_with("kokoro_production_"),
            "stage '{}' has status_key '{}' not starting with kokoro_production_",
            stage.name,
            stage.status_key
        );
    }
}

#[test]
fn test_pipeline_stages_source_files_non_empty() {
    let stages = kokoro_pipeline_stages();
    for stage in &stages {
        assert!(
            !stage.source_file.is_empty(),
            "stage '{}' has empty source_file",
            stage.name
        );
    }
}

#[test]
fn test_pipeline_stage_equality() {
    let stages1 = kokoro_pipeline_stages();
    let stages2 = kokoro_pipeline_stages();
    assert_eq!(stages1, stages2, "pipeline stages should be deterministic");
}

// ===========================================================================
// 5. detect_gaps — JSON parsing edge cases
// ===========================================================================

#[test]
fn test_detect_gaps_single_verified_stage() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_gap_json_entry("verified", "IBP", 5.0, "sound"),
        }
    });
    let report = detect_gaps(&status);
    assert_eq!(report.total_gaps, 7, "7 of 8 stages missing");
    assert_eq!(report.vacuous_count, 0);

    let bert = report
        .stages
        .iter()
        .find(|r| r.stage.name.contains("PlBert"))
        .unwrap();
    assert!(bert.has_ibp_bounds);
    assert!(bert.has_any_bounds());
}

#[test]
fn test_detect_gaps_failed_status_is_gap() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": { "status": "failed", "method": "IBP" },
        }
    });
    let report = detect_gaps(&status);
    let bert = report
        .stages
        .iter()
        .find(|r| r.stage.name.contains("PlBert"))
        .unwrap();
    assert!(
        !bert.has_any_bounds(),
        "failed status should not count as bounds"
    );
}

#[test]
fn test_detect_gaps_extra_unknown_kernels_ignored() {
    // Extra kernels not in the pipeline registry should be ignored.
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_gap_json_entry("verified", "IBP", 5.0, "sound"),
            "some_random_kernel_not_in_registry": make_gap_json_entry("verified", "CROWN", 1.0, "sound"),
        }
    });
    let report = detect_gaps(&status);
    assert_eq!(report.stages.len(), 8, "only 8 pipeline stages");
}

#[test]
fn test_detect_gaps_proof_strength_and_soundness_mode_captured() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": {
                "status": "verified",
                "method": "IBP",
                "output_width": 300.0,
                "proof_strength": "heuristic",
                "soundness_mode": "heuristic"
            },
        }
    });
    let report = detect_gaps(&status);
    let bert = report
        .stages
        .iter()
        .find(|r| r.stage.name.contains("PlBert"))
        .unwrap();
    assert_eq!(bert.proof_strength.as_deref(), Some("heuristic"));
    assert_eq!(bert.soundness_mode.as_deref(), Some("heuristic"));
}

// ===========================================================================
// 6. format_gap_report
// ===========================================================================

#[test]
fn test_format_gap_report_verified_non_vacuous_count() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_gap_json_entry("verified", "IBP", 5.0, "sound"),
            "kokoro_production_text_encoder": make_gap_json_entry("verified", "IBP", 1500.0, "vacuous"),
        }
    });
    let report = detect_gaps(&status);
    let formatted = format_gap_report(&report);
    // 1 verified non-vacuous, 1 vacuous, 6 gaps.
    assert!(formatted.contains("Verified (non-vacuous): 1/8"));
}

#[test]
fn test_format_gap_report_cpu_bridges_listed() {
    let status = serde_json::json!({ "kernels": {} });
    let report = detect_gaps(&status);
    let formatted = format_gap_report(&report);
    // iSTFT has a CPU bridge.
    assert!(
        formatted.contains("istft_terminal_readback"),
        "CPU bridge should be listed in report"
    );
}

// ===========================================================================
// 7. Proof strength — edge cases and combinations
// ===========================================================================

#[test]
fn test_proof_strength_sound_analytical_is_sound_crown() {
    // Analytical is tight → SoundCrown.
    let ps = compute_proof_strength(
        VerificationSoundnessMode::Sound,
        PropMethod::Analytical,
        50.0,
    );
    assert_eq!(ps, ProofStrength::SoundCrown);
}

#[test]
fn test_proof_strength_heuristic_ibp_narrow_is_heuristic() {
    let ps = compute_proof_strength(VerificationSoundnessMode::Heuristic, PropMethod::Ibp, 5.0);
    assert_eq!(ps, ProofStrength::Heuristic);
}

#[test]
fn test_proof_strength_heuristic_crown_narrow_is_heuristic() {
    let ps = compute_proof_strength(VerificationSoundnessMode::Heuristic, PropMethod::Crown, 5.0);
    assert_eq!(ps, ProofStrength::Heuristic);
}

#[test]
fn test_proof_strength_vacuous_threshold_is_100() {
    assert_eq!(VACUOUS_WIDTH_THRESHOLD, 100.0);
}

#[test]
fn test_proof_strength_width_just_above_threshold() {
    let ps = compute_proof_strength(VerificationSoundnessMode::Sound, PropMethod::Crown, 100.001);
    assert_eq!(ps, ProofStrength::Vacuous);
}

#[test]
fn test_proof_strength_width_exactly_at_threshold() {
    let ps = compute_proof_strength(VerificationSoundnessMode::Sound, PropMethod::Crown, 100.0);
    // Threshold is > not >=, so 100.0 is NOT vacuous.
    assert_eq!(ps, ProofStrength::SoundCrown);
}

#[test]
fn test_proof_strength_zero_width() {
    let ps = compute_proof_strength(VerificationSoundnessMode::Sound, PropMethod::Crown, 0.0);
    assert_eq!(ps, ProofStrength::SoundCrown);
}

#[test]
fn test_proof_strength_negative_width() {
    // Negative width (should not happen in practice but should not panic).
    let ps = compute_proof_strength(VerificationSoundnessMode::Sound, PropMethod::Crown, -5.0);
    assert_eq!(ps, ProofStrength::SoundCrown);
}

// ===========================================================================
// 8. ProofCertificate — validation edge cases
// ===========================================================================

#[test]
fn test_certificate_validate_inverted_bounds_rejected() {
    let result = make_verification("inv", PropMethod::Ibp, 5.0, -5.0);
    let mut cert = ProofCertificate::from_verification(&result, make_input_spec(-10.0, 10.0));
    // Force inverted bounds.
    cert.output_bounds.lower = 5.0;
    cert.output_bounds.upper = -5.0;
    cert.output_width = -10.0;
    let err = cert.validate().unwrap_err();
    assert!(matches!(err, CertificateError::InvertedBounds { .. }));
}

#[test]
fn test_certificate_validate_nan_bounds_finite_flag_mismatch() {
    let result = make_verification("nan", PropMethod::Ibp, 0.0, 1.0);
    let mut cert = ProofCertificate::from_verification(&result, make_input_spec(-1.0, 1.0));
    cert.is_finite = true;
    cert.output_bounds.lower = f32::NAN;
    let err = cert.validate().unwrap_err();
    assert!(matches!(err, CertificateError::FiniteFlagMismatch { .. }));
}

#[test]
fn test_certificate_validate_empty_kernel_name() {
    let result = make_verification("", PropMethod::Ibp, -1.0, 1.0);
    let cert = ProofCertificate::from_verification(&result, make_input_spec(-1.0, 1.0));
    let err = cert.validate().unwrap_err();
    assert!(matches!(err, CertificateError::EmptyKernelName));
}

#[test]
fn test_certificate_validate_content_hash_invalid() {
    let result = make_verification("ch", PropMethod::Ibp, -1.0, 1.0);
    let mut cert = ProofCertificate::from_verification(&result, make_input_spec(-1.0, 1.0));
    cert.content_hash = Some("not64chars".to_string());
    let err = cert.validate().unwrap_err();
    assert!(
        matches!(err, CertificateError::InvalidHash { ref field, .. } if field == "content_hash")
    );
}

#[test]
fn test_certificate_validate_hmac_signature_invalid() {
    let result = make_verification("hmac", PropMethod::Ibp, -1.0, 1.0);
    let mut cert = ProofCertificate::from_verification(&result, make_input_spec(-1.0, 1.0));
    cert.hmac_signature = Some("xyz".to_string());
    let err = cert.validate().unwrap_err();
    assert!(
        matches!(err, CertificateError::InvalidHash { ref field, .. } if field == "hmac_signature")
    );
}

#[test]
fn test_certificate_non_finite_output_width() {
    let result = make_verification("nfw", PropMethod::Ibp, -1.0, 1.0);
    let mut cert = ProofCertificate::from_verification(&result, make_input_spec(-1.0, 1.0));
    cert.output_width = f32::INFINITY;
    let err = cert.validate().unwrap_err();
    assert!(matches!(err, CertificateError::NonFiniteOutputWidth { .. }));
}

#[test]
fn test_certificate_layer_bounds_correct_indices_passes() {
    let result = make_verification("lb_ok", PropMethod::Ibp, -1.0, 1.0);
    let cert = ProofCertificate::from_verification(&result, make_input_spec(-1.0, 1.0))
        .with_layer_bounds(vec![
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
                layer_index: 1,
                layer_type: "ReLU".to_string(),
                input_bounds: vec![(-0.5, 0.5)],
                output_bounds: vec![(0.0, 0.5)],
                method: PropMethod::Crown,
                node_name: None,
                input_sources: None,
            },
        ]);
    assert!(cert.validate().is_ok());
}

// ===========================================================================
// 9. CertificateBundle — additional API coverage
// ===========================================================================

#[test]
fn test_bundle_json_roundtrip() {
    let cert = ProofCertificate::from_verification(
        &make_verification("rt", PropMethod::Crown, -1.0, 1.0),
        make_input_spec(-1.0, 1.0),
    );
    let bundle = CertificateBundle::new("test_model").with_certificate(cert);
    let json = serde_json::to_string_pretty(&bundle).expect("serialize bundle");
    let back: CertificateBundle = serde_json::from_str(&json).expect("deserialize bundle");
    assert_eq!(back.model_name, "test_model");
    assert_eq!(back.len(), 1);
    assert_eq!(back.certificates[0].kernel_name, "rt");
}

#[test]
fn test_bundle_empty_is_valid() {
    let bundle = CertificateBundle::new("empty");
    assert!(bundle.validate_all().is_ok());
    assert_eq!(bundle.verified_count(), 0);
    assert_eq!(bundle.sound_count(), 0);
    assert!(bundle.all_sound()); // vacuously true
}

#[test]
fn test_bundle_filter_by_names_not_found() {
    let cert = ProofCertificate::from_verification(
        &make_verification("a", PropMethod::Ibp, -1.0, 1.0),
        make_input_spec(-1.0, 1.0),
    );
    let bundle = CertificateBundle::new("m").with_certificate(cert);
    let filtered = bundle.filter_by_names("sub", &["nonexistent"]);
    assert_eq!(filtered.len(), 0);
}

// ===========================================================================
// 10. VerifyStatus — in-memory construction and queries
// ===========================================================================

#[test]
fn test_verify_status_default_is_empty() {
    let status = VerifyStatus::default();
    assert_eq!(status.kernel_count(), 0);
    assert!(!status.has_kernel("any"));
    assert!(status.kernels().is_empty());
    assert!(status.history().is_empty());
}

#[test]
fn test_verify_status_json_roundtrip() {
    let ks = make_kernel_status(VerificationSoundnessMode::Sound, PropMethod::Crown, 3.0);
    let json = serde_json::json!({
        "kernels": {
            "test_k": serde_json::to_value(&ks).unwrap(),
        }
    });
    let status: VerifyStatus = serde_json::from_value(json).expect("deserialize");
    assert_eq!(status.kernel_count(), 1);
    assert!(status.has_kernel("test_k"));
    let k = status.kernel("test_k").unwrap();
    assert_eq!(k.method, PropMethod::Crown);
    assert_eq!(k.soundness_mode, VerificationSoundnessMode::Sound);
}

#[test]
fn test_verify_status_soundness_counts_exclude_stale() {
    let sound_ks = make_kernel_status(VerificationSoundnessMode::Sound, PropMethod::Crown, 3.0);
    let heuristic_ks =
        make_kernel_status(VerificationSoundnessMode::Heuristic, PropMethod::Ibp, 5.0);
    let mut stale_ks = make_kernel_status(VerificationSoundnessMode::Sound, PropMethod::Crown, 2.0);
    stale_ks.stale = true;

    let json = serde_json::json!({
        "kernels": {
            "k1": serde_json::to_value(&sound_ks).unwrap(),
            "k2": serde_json::to_value(&heuristic_ks).unwrap(),
            "k3": serde_json::to_value(&stale_ks).unwrap(),
        }
    });
    let status: VerifyStatus = serde_json::from_value(json).expect("deserialize");
    let (sound, heuristic) = status.soundness_counts();
    assert_eq!(sound, 1, "stale sound entry should be excluded");
    assert_eq!(heuristic, 1);
}

#[test]
fn test_verify_status_proof_strength_counts() {
    let sound_crown = make_kernel_status(VerificationSoundnessMode::Sound, PropMethod::Crown, 3.0);
    let sound_ibp = make_kernel_status(VerificationSoundnessMode::Sound, PropMethod::Ibp, 5.0);
    let heuristic = make_kernel_status(VerificationSoundnessMode::Heuristic, PropMethod::Ibp, 5.0);
    let vacuous = make_kernel_status(VerificationSoundnessMode::Sound, PropMethod::Crown, 200.0);

    let json = serde_json::json!({
        "kernels": {
            "sc": serde_json::to_value(&sound_crown).unwrap(),
            "si": serde_json::to_value(&sound_ibp).unwrap(),
            "h": serde_json::to_value(&heuristic).unwrap(),
            "v": serde_json::to_value(&vacuous).unwrap(),
        }
    });
    let status: VerifyStatus = serde_json::from_value(json).expect("deserialize");
    let (sc, si, h, v) = status.proof_strength_counts();
    assert_eq!(sc, 1, "one sound crown");
    assert_eq!(si, 1, "one sound ibp");
    assert_eq!(h, 1, "one heuristic");
    assert_eq!(v, 1, "one vacuous");
}

// ===========================================================================
// 11. KernelStatus constructor
// ===========================================================================

#[test]
fn test_kernel_status_new_computes_proof_strength() {
    let ks = KernelStatus::new(
        VerifyOutcome::Verified,
        PropMethod::Crown,
        make_input_spec(-1.0, 1.0),
        OutputBoundsRecord::new(-1.0, 1.0),
        2.0,
        VerificationSoundnessMode::Sound,
    );
    assert_eq!(ks.proof_strength, Some(ProofStrength::SoundCrown));
    assert!(ks.crown_error.is_none());
    assert!(ks.smt.is_none());
    assert!(!ks.stale);
}

#[test]
fn test_kernel_status_new_vacuous_proof_strength() {
    let ks = KernelStatus::new(
        VerifyOutcome::Verified,
        PropMethod::Ibp,
        make_input_spec(-1.0, 1.0),
        OutputBoundsRecord::new(-200.0, 200.0),
        400.0,
        VerificationSoundnessMode::Heuristic,
    );
    assert_eq!(ks.proof_strength, Some(ProofStrength::Vacuous));
}

// ===========================================================================
// 12. VerifyOutcome — coverage
// ===========================================================================

#[test]
fn test_verify_outcome_all_variants_distinct() {
    let variants = [
        VerifyOutcome::Verified,
        VerifyOutcome::BoundsComputed,
        VerifyOutcome::IbpFallback,
        VerifyOutcome::Failed,
        VerifyOutcome::SmtContradiction,
    ];
    for i in 0..variants.len() {
        for j in (i + 1)..variants.len() {
            assert_ne!(variants[i], variants[j]);
        }
    }
}

// ===========================================================================
// 13. SmtStatusRecord and SmtOutcome
// ===========================================================================

#[test]
fn test_smt_status_record_new_defaults() {
    let rec = SmtStatusRecord::new(
        "ay".to_string(),
        SmtEncodingKind::Exact,
        "output_finite".to_string(),
        SmtOutcome::Proven,
    );
    assert_eq!(rec.solver, "ay");
    assert_eq!(rec.encoding, SmtEncodingKind::Exact);
    assert_eq!(rec.outcome, SmtOutcome::Proven);
    assert_eq!(rec.bounds_source, BoundsSource::Heuristic);
    assert!(rec.detail.is_none());
    assert!(rec.expected_bounds.is_none());
    assert!(rec.proof_alethe.is_none());
    assert!(rec.proof_verdict.is_none());
}

#[test]
fn test_smt_status_record_execution_failed() {
    let rec = SmtStatusRecord::execution_failed("solver crashed");
    assert_eq!(rec.outcome, SmtOutcome::ExecutionFailed);
    assert_eq!(rec.detail.as_deref(), Some("solver crashed"));
    assert_eq!(rec.property, "pipeline_failure");
}

#[test]
fn test_smt_outcome_all_variants_serde_roundtrip() {
    let outcomes = [
        SmtOutcome::Proven,
        SmtOutcome::Counterexample,
        SmtOutcome::Unknown,
        SmtOutcome::Unexecuted,
        SmtOutcome::ExecutionFailed,
    ];
    for outcome in &outcomes {
        let json = serde_json::to_string(outcome).expect("serialize");
        let back: SmtOutcome = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(*outcome, back);
    }
}

#[test]
fn test_smt_encoding_kind_serde_roundtrip() {
    let kinds = [SmtEncodingKind::Exact, SmtEncodingKind::UfApprox];
    for kind in &kinds {
        let json = serde_json::to_string(kind).expect("serialize");
        let back: SmtEncodingKind = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(*kind, back);
    }
}

#[test]
fn test_bounds_source_serde_roundtrip() {
    let sources = [
        BoundsSource::Analytical,
        BoundsSource::Heuristic,
        BoundsSource::CallerProvided,
    ];
    for src in &sources {
        let json = serde_json::to_string(src).expect("serialize");
        let back: BoundsSource = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(*src, back);
    }
}

// ===========================================================================
// 14. ConstructiveProofData
// ===========================================================================

#[test]
fn test_constructive_proof_data_new_defaults() {
    let cpd = ConstructiveProofData::new(
        ConstructiveProofMethod::Ibp,
        vec![-1.0],
        vec![1.0],
        vec![-2.0],
        vec![2.0],
        3,
        true,
    );
    assert_eq!(cpd.method, ConstructiveProofMethod::Ibp);
    assert!(cpd.verified);
    assert_eq!(cpd.num_layers, 3);
    assert!(cpd.lean4_export.is_none());
    assert!(cpd.layer_proofs.is_none());
    assert!(cpd.composition_lean4_source.is_none());
}

#[test]
fn test_constructive_proof_data_validate_ok() {
    let cpd = ConstructiveProofData::new(
        ConstructiveProofMethod::Crown,
        vec![-0.5, -0.3],
        vec![0.5, 0.3],
        vec![-1.0, -1.0],
        vec![1.0, 1.0],
        2,
        true,
    );
    assert!(cpd.validate().is_ok());
}

#[test]
fn test_constructive_proof_data_validate_mismatched_output_lengths() {
    let cpd = ConstructiveProofData::new(
        ConstructiveProofMethod::Ibp,
        vec![-1.0],
        vec![1.0, 2.0], // mismatched lengths
        vec![-1.0],
        vec![1.0],
        1,
        true,
    );
    let err = cpd.validate().unwrap_err();
    assert!(err.contains("output bounds length mismatch"));
}

#[test]
fn test_constructive_proof_data_validate_mismatched_input_lengths() {
    let cpd = ConstructiveProofData::new(
        ConstructiveProofMethod::Ibp,
        vec![-1.0],
        vec![1.0],
        vec![-1.0, -2.0], // mismatched
        vec![1.0],
        1,
        true,
    );
    let err = cpd.validate().unwrap_err();
    assert!(err.contains("input bounds length mismatch"));
}

#[test]
fn test_constructive_proof_data_validate_non_finite_input() {
    let cpd = ConstructiveProofData::new(
        ConstructiveProofMethod::Ibp,
        vec![-1.0],
        vec![1.0],
        vec![f32::NAN],
        vec![1.0],
        1,
        true,
    );
    let err = cpd.validate().unwrap_err();
    assert!(err.contains("non-finite input bound"));
}

#[test]
fn test_constructive_proof_data_validate_inverted_output() {
    let cpd = ConstructiveProofData::new(
        ConstructiveProofMethod::Ibp,
        vec![1.0], // lower > upper
        vec![-1.0],
        vec![-1.0],
        vec![1.0],
        1,
        true,
    );
    let err = cpd.validate().unwrap_err();
    assert!(err.contains("inverted output bound"));
}

#[test]
fn test_constructive_proof_data_is_machine_checkable() {
    // Verified + non-empty output bounds = machine checkable.
    let cpd = ConstructiveProofData::new(
        ConstructiveProofMethod::Ibp,
        vec![-1.0],
        vec![1.0],
        vec![-2.0],
        vec![2.0],
        1,
        true,
    );
    assert!(cpd.is_machine_checkable());

    // Not verified = not machine checkable.
    let cpd_unverified = ConstructiveProofData::new(
        ConstructiveProofMethod::Ibp,
        vec![-1.0],
        vec![1.0],
        vec![-2.0],
        vec![2.0],
        1,
        false,
    );
    assert!(!cpd_unverified.is_machine_checkable());
}

#[test]
fn test_constructive_proof_data_replay_verify_valid_chain() {
    let cpd = ConstructiveProofData::new(
        ConstructiveProofMethod::CrownComposition,
        vec![-0.5],
        vec![0.5],
        vec![-1.0],
        vec![1.0],
        2,
        true,
    )
    .with_layer_proofs(vec![
        ConstructiveLayerRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_lower: vec![-1.0],
            input_upper: vec![1.0],
            output_lower: vec![-0.8],
            output_upper: vec![0.8],
        },
        ConstructiveLayerRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_lower: vec![-0.8],
            input_upper: vec![0.8],
            output_lower: vec![-0.5],
            output_upper: vec![0.5],
        },
    ]);
    assert!(cpd.replay_verify());
}

#[test]
fn test_constructive_proof_data_replay_verify_broken_chain() {
    let cpd = ConstructiveProofData::new(
        ConstructiveProofMethod::CrownComposition,
        vec![-0.5],
        vec![0.5],
        vec![-1.0],
        vec![1.0],
        2,
        true,
    )
    .with_layer_proofs(vec![
        ConstructiveLayerRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_lower: vec![-1.0],
            input_upper: vec![1.0],
            output_lower: vec![-0.3],
            output_upper: vec![0.3],
        },
        ConstructiveLayerRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            // Input exceeds previous layer's output — broken chain.
            input_lower: vec![-0.5],
            input_upper: vec![0.5],
            output_lower: vec![-0.5],
            output_upper: vec![0.5],
        },
    ]);
    assert!(!cpd.replay_verify());
}

#[test]
fn test_constructive_proof_data_json_roundtrip() {
    let cpd = ConstructiveProofData::new(
        ConstructiveProofMethod::Crown,
        vec![-0.5, -0.3],
        vec![0.5, 0.3],
        vec![-1.0, -1.0],
        vec![1.0, 1.0],
        2,
        true,
    )
    .with_lean4_export("-- lean4 proof".to_string());

    let json = cpd.to_json().expect("serialize");
    let back = ConstructiveProofData::from_json(&json).expect("deserialize");
    assert_eq!(back.method, ConstructiveProofMethod::Crown);
    assert_eq!(back.output_lower, vec![-0.5, -0.3]);
    assert!(back.verified);
    assert_eq!(back.lean4_export.as_deref(), Some("-- lean4 proof"));
}

#[test]
fn test_constructive_proof_method_serde_roundtrip() {
    let methods = [
        ConstructiveProofMethod::Ibp,
        ConstructiveProofMethod::Crown,
        ConstructiveProofMethod::IbpComposition,
        ConstructiveProofMethod::CrownComposition,
    ];
    for m in &methods {
        let json = serde_json::to_string(m).expect("serialize");
        let back: ConstructiveProofMethod = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(*m, back);
    }
}

// ===========================================================================
// 15. VerificationBreakdown
// ===========================================================================

#[test]
fn test_verification_breakdown_sound_fraction_all_sound() {
    let entries: Vec<KernelStatus> = (0..5)
        .map(|_| make_kernel_status(VerificationSoundnessMode::Sound, PropMethod::Crown, 3.0))
        .collect();
    let refs: Vec<&KernelStatus> = entries.iter().collect();
    let b = VerificationBreakdown::from_entries(&refs);
    assert!((b.sound_fraction() - 1.0).abs() < f64::EPSILON);
    assert_eq!(b.total, 5);
    assert_eq!(b.sound, 5);
    assert_eq!(b.heuristic, 0);
}

#[test]
fn test_verification_breakdown_sound_fraction_none_sound() {
    let entries: Vec<KernelStatus> = (0..3)
        .map(|_| make_kernel_status(VerificationSoundnessMode::Heuristic, PropMethod::Ibp, 5.0))
        .collect();
    let refs: Vec<&KernelStatus> = entries.iter().collect();
    let b = VerificationBreakdown::from_entries(&refs);
    assert!((b.sound_fraction() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_verification_breakdown_serde_roundtrip() {
    let b = VerificationBreakdown {
        total: 10,
        sound: 7,
        heuristic: 3,
        stale: 2,
        sound_crown: 4,
        sound_ibp: 2,
        sound_mixed: 1,
        heuristic_non_vacuous: 2,
        vacuous: 1,
    };
    let json = serde_json::to_string(&b).expect("serialize");
    let back: VerificationBreakdown = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(b, back);
}

// ===========================================================================
// 16. StatusReport
// ===========================================================================

#[test]
fn test_status_report_from_verify_status_single_model() {
    let ks = make_kernel_status(VerificationSoundnessMode::Sound, PropMethod::Crown, 3.0);
    let json = serde_json::json!({
        "kernels": {
            "test_kernel": serde_json::to_value(&ks).unwrap(),
        }
    });
    let status: VerifyStatus = serde_json::from_value(json).expect("deserialize");
    let report = StatusReport::from_verify_status("test_model", &status);
    assert_eq!(report.total_entries(), 1);
    assert_eq!(report.total_stale(), 0);
    assert_eq!(report.per_model_summary().len(), 1);
    assert_eq!(report.per_model_summary()[0].model, "test_model");
}

#[test]
fn test_status_report_to_text_contains_sections() {
    let report = StatusReport::from_status_files(std::path::Path::new("/nonexistent/42"))
        .expect("empty report")
        .with_kani_count(500)
        .with_gap_summary(GapSummary {
            stages_checked: 8,
            gaps: 2,
            vacuous: 1,
        });
    let text = report.to_text();
    assert!(text.contains("nn Verification Status Report"));
    assert!(text.contains("Kani harnesses: 500"));
    assert!(text.contains("Gaps: 2"));
    assert!(text.contains("Vacuous: 1"));
}

#[test]
fn test_status_report_display_equals_to_text() {
    let report = StatusReport::from_status_files(std::path::Path::new("/nonexistent/42"))
        .expect("empty report");
    let display = format!("{report}");
    let text = report.to_text();
    assert_eq!(display, text);
}

// ===========================================================================
// 17. Trend
// ===========================================================================

#[test]
fn test_trend_default_all_none() {
    let t = Trend::default();
    assert!(t.prev_kani_harnesses.is_none());
    assert!(t.current_kani_harnesses.is_none());
    assert!(t.kani_delta.is_none());
    assert!(t.prev_total_sound.is_none());
    assert!(t.current_total_sound.is_none());
    assert!(t.sound_delta.is_none());
}

#[test]
fn test_trend_serde_roundtrip() {
    let t = Trend {
        prev_kani_harnesses: Some(1000),
        current_kani_harnesses: Some(1100),
        kani_delta: Some(100),
        prev_total_sound: Some(50),
        current_total_sound: Some(55),
        sound_delta: Some(5),
    };
    let json = serde_json::to_string(&t).expect("serialize");
    let back: Trend = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(t, back);
}

// ===========================================================================
// 18. GapSummary
// ===========================================================================

#[test]
fn test_gap_summary_serde_roundtrip() {
    let g = GapSummary {
        stages_checked: 8,
        gaps: 3,
        vacuous: 1,
    };
    let json = serde_json::to_string(&g).expect("serialize");
    let back: GapSummary = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(g, back);
}

// ===========================================================================
// 19. NormBoundsMode
// ===========================================================================

#[test]
fn test_norm_bounds_mode_forward_mode_flag() {
    assert!(!NormBoundsMode::Conservative.forward_mode());
    assert!(NormBoundsMode::ForwardMode.forward_mode());
    assert!(NormBoundsMode::CrownSampling.forward_mode());
}

// ===========================================================================
// 20. model_for_kernel classification
// ===========================================================================

#[test]
fn test_model_for_kernel_shared_fallback() {
    use crate::status::model_for_kernel;
    assert_eq!(model_for_kernel("snake_alpha"), "shared");
    assert_eq!(model_for_kernel("relu_v2"), "shared");
    assert_eq!(model_for_kernel("custom_unknown"), "shared");
    assert_eq!(model_for_kernel(""), "shared");
}

#[test]
fn test_model_for_kernel_all_categories() {
    use crate::status::model_for_kernel;
    assert_eq!(model_for_kernel("kokoro_x"), "kokoro");
    assert_eq!(model_for_kernel("demucs_x"), "demucs");
    assert_eq!(model_for_kernel("htdemucs_x"), "demucs");
    assert_eq!(model_for_kernel("silero_x"), "silero");
    assert_eq!(model_for_kernel("whisper_x"), "whisper");
    assert_eq!(model_for_kernel("qwen3_x"), "qwen3");
    assert_eq!(model_for_kernel("glm5_x"), "glm5");
    assert_eq!(model_for_kernel("glm_x"), "glm");
    assert_eq!(model_for_kernel("gptoss_x"), "gptoss");
    assert_eq!(model_for_kernel("gpt_oss_x"), "gptoss");
    assert_eq!(model_for_kernel("moe_dispatch_x"), "gptoss");
}

// ===========================================================================
// 21. MODEL_CATEGORIES constant
// ===========================================================================

#[test]
fn test_model_categories_contains_expected() {
    use crate::status::MODEL_CATEGORIES;
    assert!(MODEL_CATEGORIES.contains(&"kokoro"));
    assert!(MODEL_CATEGORIES.contains(&"demucs"));
    assert!(MODEL_CATEGORIES.contains(&"silero"));
    assert!(MODEL_CATEGORIES.contains(&"whisper"));
    assert!(MODEL_CATEGORIES.contains(&"qwen3"));
    assert!(MODEL_CATEGORIES.contains(&"glm5"));
    assert!(MODEL_CATEGORIES.contains(&"shared"));
}

// ===========================================================================
// 22. OutputBoundsRecord
// ===========================================================================

#[test]
fn test_output_bounds_record_zero() {
    let obr = OutputBoundsRecord::zero();
    assert_eq!(obr.lower, 0.0);
    assert_eq!(obr.upper, 0.0);
    assert!(!obr.is_infeasible);
}

#[test]
fn test_output_bounds_record_with_shape() {
    let obr = OutputBoundsRecord::with_shape(-5.0, 5.0, vec![1, 10]);
    assert_eq!(obr.lower, -5.0);
    assert_eq!(obr.upper, 5.0);
    assert_eq!(obr.shape, Some(vec![1, 10]));
    assert!(!obr.is_infeasible);
}

// ===========================================================================
// 23. compute_bytes_hash — SHA-256
// ===========================================================================

#[test]
fn test_compute_bytes_hash_known_value() {
    // SHA-256 of "test" is well-known.
    let h = compute_bytes_hash(b"test");
    assert_eq!(
        h,
        "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    );
}

#[test]
fn test_compute_bytes_hash_always_64_hex_chars() {
    for input in [b"a".as_ref(), b"hello world", b"", &[0u8; 1024]] {
        let h = compute_bytes_hash(input);
        assert_eq!(h.len(), 64);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }
}

// ===========================================================================
// 24. PipelineStage Debug/Clone/Eq
// ===========================================================================

#[test]
fn test_pipeline_stage_clone_eq() {
    let stage = PipelineStage {
        name: "test",
        status_key: "test_key",
        is_compiled_segment: true,
        is_bridge: false,
        source_file: "test.rs",
        cpu_bridges: &[],
    };
    let cloned = stage.clone();
    assert_eq!(stage, cloned);
}

#[test]
fn test_pipeline_stage_hash() {
    use std::collections::HashSet;
    let stages = kokoro_pipeline_stages();
    let set: HashSet<_> = stages.iter().collect();
    assert_eq!(
        set.len(),
        stages.len(),
        "all stages should be unique when hashed"
    );
}
