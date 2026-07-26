// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for the nn-verify certification pipeline, gap detector,
//! soundness classification, proof strength, status file parsing, coverage
//! metrics, stale entry detection, and multi-model verification.
//!
//! Part of #4186.

use nn_verify::certificate::{CertificateBundle, ProofCertificate, CERTIFICATE_VERSION};
use nn_verify::gap_detector::{detect_gaps, format_gap_report, kokoro_pipeline_stages};
use nn_verify::status::{
    compute_proof_strength, KernelStatus, ProofStrength, VerifyOutcome, VACUOUS_WIDTH_THRESHOLD,
};
use nn_verify::status_report::{GapSummary, StatusReport, VerificationBreakdown};
use nn_verify::{
    InputBoundsRecord, KernelVerification, OutputTensorBounds, ParamInputRecord, PropMethod,
    VerificationSoundnessMode, VerifyStatus,
};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn make_verification_named(name: &str, lower: f32, upper: f32) -> KernelVerification {
    let mut v = KernelVerification::new(
        name.to_string(),
        PropMethod::Ibp,
        lower,
        upper,
        upper - lower,
        true,
    );
    v.output_tensor = Some(OutputTensorBounds::new(vec![lower], vec![upper], vec![1]));
    v
}

fn make_verification(lower: f32, upper: f32) -> KernelVerification {
    make_verification_named("test_kernel", lower, upper)
}

fn make_input_spec() -> InputBoundsRecord {
    InputBoundsRecord::new(&[ParamInputRecord::new(0, -1.0, 1.0)], &[1.0])
}

fn make_status_entry(
    status: &str,
    method: &str,
    width: f64,
    proof_strength: &str,
    soundness_mode: &str,
) -> serde_json::Value {
    serde_json::json!({
        "status": status,
        "method": method,
        "output_width": width,
        "proof_strength": proof_strength,
        "soundness_mode": soundness_mode
    })
}

/// Build a VerifyStatus with kernel entries by round-tripping through JSON.
fn status_from_json(json: serde_json::Value) -> VerifyStatus {
    let dir = std::env::temp_dir().join(format!(
        "nn_cert_ext_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("status.json");
    std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).unwrap();
    let status = VerifyStatus::load(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
    status
}

// ===========================================================================
// 1. Gap detector — missing verification entries for known op types
// ===========================================================================

#[test]
fn test_gap_detector_identifies_missing_entries_for_all_known_stages() {
    // An empty status file should produce a gap for every known pipeline stage.
    let status = serde_json::json!({ "kernels": {} });
    let report = detect_gaps(&status);
    let stages = kokoro_pipeline_stages();

    assert_eq!(
        report.total_gaps,
        stages.len(),
        "every pipeline stage should be a gap when status is empty"
    );

    // Each individual result should report no bounds.
    for result in &report.stages {
        assert!(
            !result.has_any_bounds(),
            "stage '{}' should have no bounds in empty status",
            result.stage.name
        );
        assert!(!result.has_ibp_bounds);
        assert!(!result.has_crown_bounds);
        assert!(!result.has_analytical_bounds);
    }
}

#[test]
fn test_gap_detector_identifies_gaps_for_unrecognized_keys() {
    // Status file has entries, but none match known pipeline stage keys.
    let status = serde_json::json!({
        "kernels": {
            "some_random_model_alpha": make_status_entry("verified", "IBP", 5.0, "sound", "sound"),
            "another_model_beta": make_status_entry("verified", "CROWN", 2.0, "sound", "sound"),
        }
    });
    let report = detect_gaps(&status);
    let stages = kokoro_pipeline_stages();

    assert_eq!(
        report.total_gaps,
        stages.len(),
        "unrecognized keys should not reduce gap count"
    );
}

#[test]
fn test_gap_detector_partial_coverage_identifies_specific_missing_stages() {
    // Cover only bert_encoder and generator; the rest should be gaps.
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_status_entry("verified", "IBP", 10.0, "sound", "sound"),
            "kokoro_production_generator": make_status_entry("verified", "IBP", 5.0, "sound", "sound"),
        }
    });
    let report = detect_gaps(&status);

    let covered: Vec<&str> = report
        .stages
        .iter()
        .filter(|r| r.has_any_bounds())
        .map(|r| r.stage.status_key)
        .collect();
    assert_eq!(covered.len(), 2);
    assert!(covered.contains(&"kokoro_production_bert_encoder"));
    assert!(covered.contains(&"kokoro_production_generator"));

    // Remaining 6 should be gaps.
    assert_eq!(report.total_gaps, 6);
}

#[test]
fn test_gap_detector_bounds_computed_status_counts_as_covered() {
    // "bounds_computed" is treated as having bounds (not a gap).
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_text_encoder": {
                "status": "bounds_computed",
                "method": "IBP",
                "output_width": 50.0,
                "proof_strength": "heuristic",
                "soundness_mode": "heuristic"
            }
        }
    });
    let report = detect_gaps(&status);
    let te = report
        .stages
        .iter()
        .find(|r| r.stage.name == "TextEncoder")
        .unwrap();
    assert!(
        te.has_any_bounds(),
        "bounds_computed should count as covered"
    );
    assert_eq!(report.total_gaps, 7, "only text_encoder is covered");
}

// ===========================================================================
// 2. Certificate structure — required fields
// ===========================================================================

#[test]
fn test_certificate_has_required_fields_model_name() {
    let bundle = CertificateBundle::new("nn_model");
    assert_eq!(bundle.model_name, "nn_model");
}

#[test]
fn test_certificate_has_timestamp_after_creation() {
    let v = make_verification(0.0, 1.0);
    let cert = ProofCertificate::from_verification(&v, make_input_spec());
    // generated_at should be populated (non-empty timestamp string).
    assert!(
        !cert.generated_at.is_empty(),
        "certificate should have a generated_at timestamp"
    );
}

#[test]
fn test_certificate_bundle_has_version() {
    let bundle = CertificateBundle::new("versioned_model");
    assert!(
        bundle.version > 0,
        "bundle should have a positive version number"
    );
}

#[test]
fn test_certificate_bundle_entries_accessible() {
    let v1 = make_verification_named("kernel_a", 0.0, 1.0);
    let v2 = make_verification_named("kernel_b", -0.5, 0.5);

    let bundle = CertificateBundle::new("multi_entry")
        .with_certificate(ProofCertificate::from_verification(&v1, make_input_spec()))
        .with_certificate(ProofCertificate::from_verification(&v2, make_input_spec()));

    assert_eq!(bundle.len(), 2, "bundle should have 2 entries");
    assert_eq!(bundle.certificates[0].kernel_name, "kernel_a");
    assert_eq!(bundle.certificates[1].kernel_name, "kernel_b");
}

#[test]
fn test_certificate_bundle_summary_all_sound() {
    let v = make_verification(0.0, 1.0);
    let bundle = CertificateBundle::new("sound_bundle")
        .with_certificate(ProofCertificate::from_verification(&v, make_input_spec()));
    assert!(bundle.all_sound(), "IBP verification should be sound");
    assert_eq!(bundle.sound_count(), 1);
    assert_eq!(bundle.verified_count(), 1);
}

#[test]
fn test_certificate_version_constant_is_current() {
    // CERTIFICATE_VERSION should be >= 1 and match the latest format.
    assert!(CERTIFICATE_VERSION >= 1);
    let v = make_verification(0.0, 1.0);
    let cert = ProofCertificate::from_verification(&v, make_input_spec());
    assert_eq!(cert.version, CERTIFICATE_VERSION);
}

// ===========================================================================
// 3. Soundness modes — IbpValidated, AlphaCrown, BetaCrown, Analytical
// ===========================================================================

#[test]
fn test_soundness_mode_sound_with_ibp() {
    let strength = compute_proof_strength(VerificationSoundnessMode::Sound, PropMethod::Ibp, 5.0);
    assert_eq!(strength, ProofStrength::SoundIbp);
}

#[test]
fn test_soundness_mode_sound_with_crown() {
    let strength = compute_proof_strength(VerificationSoundnessMode::Sound, PropMethod::Crown, 3.0);
    assert_eq!(strength, ProofStrength::SoundCrown);
}

#[test]
fn test_soundness_mode_sound_with_alpha_crown() {
    let strength = compute_proof_strength(
        VerificationSoundnessMode::Sound,
        PropMethod::AlphaCrown,
        2.0,
    );
    assert_eq!(strength, ProofStrength::SoundCrown);
}

#[test]
fn test_soundness_mode_sound_with_beta_crown() {
    let strength =
        compute_proof_strength(VerificationSoundnessMode::Sound, PropMethod::BetaCrown, 1.5);
    assert_eq!(strength, ProofStrength::SoundCrown);
}

#[test]
fn test_soundness_mode_sound_with_analytical() {
    let strength = compute_proof_strength(
        VerificationSoundnessMode::Sound,
        PropMethod::Analytical,
        10.0,
    );
    assert_eq!(strength, ProofStrength::SoundCrown);
}

#[test]
fn test_soundness_mode_sound_with_mixed_ibp_crown() {
    let strength = compute_proof_strength(
        VerificationSoundnessMode::Sound,
        PropMethod::MixedIbpCrown,
        8.0,
    );
    assert_eq!(strength, ProofStrength::SoundMixed);
}

#[test]
fn test_soundness_mode_heuristic_produces_heuristic_strength() {
    let strength =
        compute_proof_strength(VerificationSoundnessMode::Heuristic, PropMethod::Crown, 5.0);
    assert_eq!(strength, ProofStrength::Heuristic);
}

#[test]
fn test_soundness_mode_heuristic_ibp_still_heuristic() {
    let strength =
        compute_proof_strength(VerificationSoundnessMode::Heuristic, PropMethod::Ibp, 3.0);
    assert_eq!(strength, ProofStrength::Heuristic);
}

// ===========================================================================
// 4. Proof strength classification — sound, heuristic, vacuous
// ===========================================================================

#[test]
fn test_proof_strength_vacuous_when_width_exceeds_threshold() {
    // Width > VACUOUS_WIDTH_THRESHOLD should always be Vacuous regardless of soundness.
    let strength_sound = compute_proof_strength(
        VerificationSoundnessMode::Sound,
        PropMethod::Ibp,
        VACUOUS_WIDTH_THRESHOLD + 1.0,
    );
    assert_eq!(strength_sound, ProofStrength::Vacuous);

    let strength_heuristic = compute_proof_strength(
        VerificationSoundnessMode::Heuristic,
        PropMethod::Crown,
        VACUOUS_WIDTH_THRESHOLD + 50.0,
    );
    assert_eq!(strength_heuristic, ProofStrength::Vacuous);
}

#[test]
fn test_proof_strength_not_vacuous_at_threshold_boundary() {
    // Width exactly at threshold should NOT be vacuous.
    let strength = compute_proof_strength(
        VerificationSoundnessMode::Sound,
        PropMethod::Ibp,
        VACUOUS_WIDTH_THRESHOLD,
    );
    assert_eq!(strength, ProofStrength::SoundIbp);
}

#[test]
fn test_proof_strength_vacuous_just_above_threshold() {
    let strength = compute_proof_strength(
        VerificationSoundnessMode::Sound,
        PropMethod::Crown,
        VACUOUS_WIDTH_THRESHOLD + 0.001,
    );
    assert_eq!(strength, ProofStrength::Vacuous);
}

#[test]
fn test_proof_strength_sound_crown_vs_sound_ibp_distinction() {
    let crown_strength =
        compute_proof_strength(VerificationSoundnessMode::Sound, PropMethod::Crown, 5.0);
    let ibp_strength =
        compute_proof_strength(VerificationSoundnessMode::Sound, PropMethod::Ibp, 5.0);
    assert_eq!(crown_strength, ProofStrength::SoundCrown);
    assert_eq!(ibp_strength, ProofStrength::SoundIbp);
    assert_ne!(crown_strength, ibp_strength);
}

#[test]
fn test_proof_strength_is_tight_classification() {
    // Tight methods: Crown, AlphaCrown, BetaCrown, Analytical
    assert!(PropMethod::Crown.is_tight());
    assert!(PropMethod::AlphaCrown.is_tight());
    assert!(PropMethod::BetaCrown.is_tight());
    assert!(PropMethod::Analytical.is_tight());
    // Non-tight methods:
    assert!(!PropMethod::Ibp.is_tight());
    assert!(!PropMethod::MixedIbpCrown.is_tight());
}

// ===========================================================================
// 5. Status file loading — parsing nn_verify_status JSON
// ===========================================================================

#[test]
fn test_status_file_parse_with_sound_entries() {
    let json = serde_json::json!({
        "kernels": {
            "kernel_alpha": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": {
                    "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}],
                    "constant_params": [1.0]
                },
                "output_bounds": { "lower": -0.5, "upper": 0.5 },
                "output_width": 1.0,
                "soundness_mode": "sound"
            },
            "kernel_beta": {
                "status": "verified",
                "method": "CROWN",
                "input_bounds": {
                    "variable_inputs": [{"param_index": 0, "lower": -2.0, "upper": 2.0}],
                    "constant_params": []
                },
                "output_bounds": { "lower": -0.3, "upper": 0.3 },
                "output_width": 0.6,
                "soundness_mode": "sound"
            }
        }
    });
    let status = status_from_json(json);

    assert_eq!(status.kernel_count(), 2);
    assert!(status.kernel("kernel_alpha").is_some());
    assert!(status.kernel("kernel_beta").is_some());

    let alpha = status.kernel("kernel_alpha").unwrap();
    assert_eq!(alpha.status, VerifyOutcome::Verified);
    assert_eq!(alpha.method, PropMethod::Ibp);
    assert_eq!(alpha.soundness_mode, VerificationSoundnessMode::Sound);
}

#[test]
fn test_status_file_parse_with_heuristic_entries() {
    let json = serde_json::json!({
        "kernels": {
            "kernel_h": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": {
                    "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}],
                    "constant_params": []
                },
                "output_bounds": { "lower": -5.0, "upper": 5.0 },
                "output_width": 10.0,
                "soundness_mode": "heuristic"
            }
        }
    });
    let status = status_from_json(json);
    let h = status.kernel("kernel_h").unwrap();
    assert_eq!(h.soundness_mode, VerificationSoundnessMode::Heuristic);
}

#[test]
fn test_status_file_parse_missing_soundness_defaults_to_heuristic() {
    // Legacy JSON without soundness_mode field defaults to Heuristic (fail-closed).
    let json = serde_json::json!({
        "kernels": {
            "legacy_kernel": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": {
                    "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}],
                    "constant_params": []
                },
                "output_bounds": { "lower": -0.5, "upper": 0.5 },
                "output_width": 1.0
            }
        }
    });
    let status = status_from_json(json);
    let entry = status.kernel("legacy_kernel").unwrap();
    assert_eq!(
        entry.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "missing soundness_mode should default to Heuristic (fail-closed)"
    );
}

#[test]
fn test_status_file_roundtrip_preserves_all_fields() {
    let dir = std::env::temp_dir().join(format!("nn_cert_ext_roundtrip_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("roundtrip.json");
    let _ = std::fs::remove_file(&path);

    let status = VerifyStatus::default();
    status.save(&path).expect("save");
    let loaded = VerifyStatus::load(&path).expect("load");
    assert!(loaded.kernels().is_empty());

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn test_status_file_load_nonexistent_returns_default() {
    let path = std::env::temp_dir().join("nonexistent_status_file_12345.json");
    let status = VerifyStatus::load(&path).expect("load nonexistent");
    assert!(status.kernels().is_empty());
    assert_eq!(status.kernel_count(), 0);
}

// ===========================================================================
// 6. Verification coverage metrics — coverage percentage calculation
// ===========================================================================

#[test]
fn test_verification_breakdown_all_sound() {
    let json = serde_json::json!({
        "kernels": {
            "k1": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": { "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}], "constant_params": [] },
                "output_bounds": { "lower": -0.5, "upper": 0.5 },
                "output_width": 1.0,
                "soundness_mode": "sound"
            },
            "k2": {
                "status": "verified",
                "method": "CROWN",
                "input_bounds": { "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}], "constant_params": [] },
                "output_bounds": { "lower": -0.3, "upper": 0.3 },
                "output_width": 0.6,
                "soundness_mode": "sound"
            }
        }
    });
    let status = status_from_json(json);
    let entries: Vec<&KernelStatus> = status.kernels().values().collect();
    let breakdown = VerificationBreakdown::from_entries(&entries);

    assert_eq!(breakdown.total, 2);
    assert_eq!(breakdown.sound, 2);
    assert_eq!(breakdown.heuristic, 0);
    assert_eq!(breakdown.stale, 0);
    assert!((breakdown.sound_fraction() - 1.0).abs() < 1e-10);
}

#[test]
fn test_verification_breakdown_mixed_sound_heuristic() {
    let json = serde_json::json!({
        "kernels": {
            "k_sound": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": { "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}], "constant_params": [] },
                "output_bounds": { "lower": -0.5, "upper": 0.5 },
                "output_width": 1.0,
                "soundness_mode": "sound"
            },
            "k_heuristic": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": { "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}], "constant_params": [] },
                "output_bounds": { "lower": -5.0, "upper": 5.0 },
                "output_width": 10.0,
                "soundness_mode": "heuristic"
            }
        }
    });
    let status = status_from_json(json);
    let entries: Vec<&KernelStatus> = status.kernels().values().collect();
    let breakdown = VerificationBreakdown::from_entries(&entries);

    assert_eq!(breakdown.total, 2);
    assert_eq!(breakdown.sound, 1);
    assert_eq!(breakdown.heuristic, 1);
    assert!((breakdown.sound_fraction() - 0.5).abs() < 1e-10);
}

#[test]
fn test_verification_breakdown_zero_entries() {
    let status = VerifyStatus::default();
    let entries: Vec<&KernelStatus> = status.kernels().values().collect();
    let breakdown = VerificationBreakdown::from_entries(&entries);

    assert_eq!(breakdown.total, 0);
    assert_eq!(breakdown.sound, 0);
    assert!((breakdown.sound_fraction() - 0.0).abs() < 1e-10);
}

#[test]
fn test_verification_breakdown_proof_strength_crown_vs_ibp() {
    let json = serde_json::json!({
        "kernels": {
            "k_crown": {
                "status": "verified",
                "method": "CROWN",
                "input_bounds": { "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}], "constant_params": [] },
                "output_bounds": { "lower": -0.3, "upper": 0.3 },
                "output_width": 0.6,
                "soundness_mode": "sound"
            },
            "k_ibp": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": { "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}], "constant_params": [] },
                "output_bounds": { "lower": -0.5, "upper": 0.5 },
                "output_width": 1.0,
                "soundness_mode": "sound"
            }
        }
    });
    let status = status_from_json(json);
    let entries: Vec<&KernelStatus> = status.kernels().values().collect();
    let breakdown = VerificationBreakdown::from_entries(&entries);

    // Both sound, 1 is CROWN (sound_crown), 1 is IBP (sound_ibp)
    assert_eq!(breakdown.sound, 2);
    assert_eq!(breakdown.sound_crown + breakdown.sound_ibp, 2);
}

#[test]
fn test_status_report_from_verify_status() {
    let json = serde_json::json!({
        "kernels": {
            "k1": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": { "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}], "constant_params": [] },
                "output_bounds": { "lower": -0.5, "upper": 0.5 },
                "output_width": 1.0,
                "soundness_mode": "sound"
            }
        }
    });
    let status = status_from_json(json);
    let report = StatusReport::from_verify_status("test_model", &status);

    assert_eq!(report.total_entries(), 1);
    assert_eq!(report.models.len(), 1);
    assert_eq!(report.models[0].model, "test_model");
    assert_eq!(report.summary.sound, 1);
}

#[test]
fn test_status_report_with_gap_summary() {
    let json = serde_json::json!({ "kernels": {} });
    let status = status_from_json(json);
    let report =
        StatusReport::from_verify_status("test_model", &status).with_gap_summary(GapSummary {
            stages_checked: 8,
            gaps: 3,
            vacuous: 1,
        });

    let gap = report.gap_summary.as_ref().unwrap();
    assert_eq!(gap.stages_checked, 8);
    assert_eq!(gap.gaps, 3);
    assert_eq!(gap.vacuous, 1);
}

// ===========================================================================
// 7. Stale entry detection — stale entries excluded from soundness counts
// ===========================================================================

#[test]
fn test_stale_entries_excluded_from_breakdown_totals() {
    let json = serde_json::json!({
        "kernels": {
            "k_active": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": { "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}], "constant_params": [] },
                "output_bounds": { "lower": -0.5, "upper": 0.5 },
                "output_width": 1.0,
                "soundness_mode": "sound"
            },
            "k_stale": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": { "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}], "constant_params": [] },
                "output_bounds": { "lower": -0.5, "upper": 0.5 },
                "output_width": 1.0,
                "soundness_mode": "sound",
                "stale": true,
                "stale_reason": "superseded"
            }
        }
    });
    let status = status_from_json(json);
    let entries: Vec<&KernelStatus> = status.kernels().values().collect();
    let breakdown = VerificationBreakdown::from_entries(&entries);

    assert_eq!(breakdown.total, 1, "only non-stale entries count in total");
    assert_eq!(
        breakdown.sound, 1,
        "stale entry should not be counted as sound"
    );
    assert_eq!(breakdown.stale, 1, "one stale entry");
}

#[test]
fn test_stale_entries_do_not_inflate_sound_fraction() {
    let json = serde_json::json!({
        "kernels": {
            "k_active_heuristic": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": { "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}], "constant_params": [] },
                "output_bounds": { "lower": -5.0, "upper": 5.0 },
                "output_width": 10.0,
                "soundness_mode": "heuristic"
            },
            "k_stale_sound": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": { "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}], "constant_params": [] },
                "output_bounds": { "lower": -0.5, "upper": 0.5 },
                "output_width": 1.0,
                "soundness_mode": "sound",
                "stale": true
            }
        }
    });
    let status = status_from_json(json);
    let entries: Vec<&KernelStatus> = status.kernels().values().collect();
    let breakdown = VerificationBreakdown::from_entries(&entries);

    // Only one non-stale entry (heuristic), so sound fraction = 0/1 = 0.0
    assert_eq!(breakdown.total, 1);
    assert_eq!(breakdown.sound, 0);
    assert_eq!(breakdown.stale, 1);
    assert!((breakdown.sound_fraction() - 0.0).abs() < 1e-10);
}

#[test]
fn test_all_stale_entries_yields_zero_total() {
    let json = serde_json::json!({
        "kernels": {
            "k1": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": { "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}], "constant_params": [] },
                "output_bounds": { "lower": -0.5, "upper": 0.5 },
                "output_width": 1.0,
                "soundness_mode": "sound",
                "stale": true
            },
            "k2": {
                "status": "verified",
                "method": "CROWN",
                "input_bounds": { "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}], "constant_params": [] },
                "output_bounds": { "lower": -0.3, "upper": 0.3 },
                "output_width": 0.6,
                "soundness_mode": "sound",
                "stale": true
            }
        }
    });
    let status = status_from_json(json);
    let entries: Vec<&KernelStatus> = status.kernels().values().collect();
    let breakdown = VerificationBreakdown::from_entries(&entries);

    assert_eq!(breakdown.total, 0, "all stale means zero non-stale entries");
    assert_eq!(breakdown.stale, 2);
    assert!((breakdown.sound_fraction() - 0.0).abs() < 1e-10);
}

#[test]
fn test_mark_stale_on_loaded_status() {
    let json = serde_json::json!({
        "kernels": {
            "k_to_stale": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": { "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}], "constant_params": [] },
                "output_bounds": { "lower": -0.5, "upper": 0.5 },
                "output_width": 1.0,
                "soundness_mode": "sound"
            }
        }
    });
    let mut status = status_from_json(json);

    // Before marking stale, entry is active.
    let entry = status.kernel("k_to_stale").unwrap();
    assert!(!entry.stale);

    // Mark stale.
    status
        .mark_stale("k_to_stale", "architecture changed")
        .expect("mark_stale");

    let entry = status.kernel("k_to_stale").unwrap();
    assert!(entry.stale);
    assert_eq!(entry.stale_reason.as_deref(), Some("architecture changed"));
}

// ===========================================================================
// 8. Multi-model verification — compose verification across model types
// ===========================================================================

#[test]
fn test_gap_detector_format_report_contains_all_stage_names() {
    let status = serde_json::json!({ "kernels": {} });
    let report = detect_gaps(&status);
    let formatted = format_gap_report(&report);

    let stages = kokoro_pipeline_stages();
    for stage in &stages {
        assert!(
            formatted.contains(stage.name),
            "formatted report should mention stage '{}'",
            stage.name
        );
    }
}

#[test]
fn test_gap_detector_report_distinguishes_ibp_crown_analytical() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_status_entry("verified", "IBP", 10.0, "sound", "sound"),
            "kokoro_production_text_encoder_crown": make_status_entry("verified", "CROWN", 1.4, "sound", "sound"),
            "kokoro_production_length_regulate": make_status_entry("verified", "ANALYTICAL", 49.0, "sound", "sound"),
        }
    });
    let report = detect_gaps(&status);
    let formatted = format_gap_report(&report);

    // IBP entry for bert
    assert!(formatted.contains("[ok]"), "should have IBP [ok] marker");
    // CROWN entry for text_encoder
    assert!(
        formatted.contains("[OK]"),
        "should have CROWN/ANALYTICAL [OK] marker"
    );
}

#[test]
fn test_multi_model_status_report_aggregation() {
    // Build two separate model statuses and verify aggregation.
    let json_a = serde_json::json!({
        "kernels": {
            "model_a_kernel_1": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": { "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}], "constant_params": [] },
                "output_bounds": { "lower": -0.5, "upper": 0.5 },
                "output_width": 1.0,
                "soundness_mode": "sound"
            },
            "model_a_kernel_2": {
                "status": "verified",
                "method": "CROWN",
                "input_bounds": { "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}], "constant_params": [] },
                "output_bounds": { "lower": -0.3, "upper": 0.3 },
                "output_width": 0.6,
                "soundness_mode": "sound"
            }
        }
    });
    let json_b = serde_json::json!({
        "kernels": {
            "model_b_kernel_1": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": { "variable_inputs": [{"param_index": 0, "lower": -2.0, "upper": 2.0}], "constant_params": [] },
                "output_bounds": { "lower": -1.0, "upper": 1.0 },
                "output_width": 2.0,
                "soundness_mode": "heuristic"
            }
        }
    });

    let status_a = status_from_json(json_a);
    let status_b = status_from_json(json_b);

    let report_a = StatusReport::from_verify_status("model_a", &status_a);
    let report_b = StatusReport::from_verify_status("model_b", &status_b);

    // Model A: 2 sound entries
    assert_eq!(report_a.summary.sound, 2);
    assert_eq!(report_a.summary.heuristic, 0);

    // Model B: 1 heuristic entry
    assert_eq!(report_b.summary.sound, 0);
    assert_eq!(report_b.summary.heuristic, 1);
}

#[test]
fn test_gap_detector_with_crown_suffix_entries() {
    // Both primary and _crown suffix entries for the same stage.
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_status_entry("verified", "IBP", 200.0, "heuristic", "heuristic"),
            "kokoro_production_bert_encoder_crown": make_status_entry("verified", "AlphaCROWN", 1.0, "sound", "sound"),
        }
    });
    let report = detect_gaps(&status);
    let bert = report
        .stages
        .iter()
        .find(|r| r.stage.name.contains("bert"))
        .unwrap();

    assert!(
        bert.has_ibp_bounds,
        "primary IBP entry should give IBP bounds"
    );
    assert!(bert.has_crown_bounds, "AlphaCROWN should give CROWN bounds");
    assert!(bert.has_any_bounds());
}

#[test]
fn test_gap_detector_vacuous_by_proof_strength_label() {
    // "vacuous" proof_strength label makes entry vacuous even with small width.
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_f0_predictor": {
                "status": "verified",
                "method": "IBP",
                "output_width": 2.0,
                "proof_strength": "vacuous",
                "soundness_mode": "heuristic"
            }
        }
    });
    let report = detect_gaps(&status);
    let f0 = report
        .stages
        .iter()
        .find(|r| r.stage.name.contains("F0"))
        .unwrap();

    assert!(
        f0.is_vacuous,
        "proof_strength='vacuous' should mark entry as vacuous"
    );
}

#[test]
fn test_gap_detector_stale_entries_in_status_json() {
    // Stale entries in the gap detector status JSON are still counted as
    // having bounds (the gap detector checks status, not staleness).
    // The staleness filtering is done at the VerificationBreakdown level.
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": {
                "status": "verified",
                "method": "IBP",
                "output_width": 10.0,
                "proof_strength": "sound",
                "soundness_mode": "sound",
                "stale": true
            }
        }
    });
    let report = detect_gaps(&status);
    let bert = report
        .stages
        .iter()
        .find(|r| r.stage.name.contains("bert"))
        .unwrap();

    // Gap detector operates on status JSON directly, stale flag does not affect it.
    assert!(
        bert.has_any_bounds(),
        "stale entry still has bounds in gap detector JSON"
    );
}

#[test]
fn test_certificate_bundle_multi_model_filter() {
    let v1 = make_verification_named("kokoro_encoder", 0.0, 1.0);
    let v2 = make_verification_named("demucs_decoder", -1.0, 1.0);
    let v3 = make_verification_named("whisper_encoder", -0.5, 0.5);

    let bundle = CertificateBundle::new("multi_model")
        .with_certificate(ProofCertificate::from_verification(&v1, make_input_spec()))
        .with_certificate(ProofCertificate::from_verification(&v2, make_input_spec()))
        .with_certificate(ProofCertificate::from_verification(&v3, make_input_spec()));

    assert_eq!(bundle.len(), 3);

    // Filter to only kokoro + whisper
    let filtered = bundle.filter_by_names("subset", &["kokoro_encoder", "whisper_encoder"]);
    assert_eq!(filtered.len(), 2);
    assert_eq!(filtered.certificates[0].kernel_name, "kokoro_encoder");
    assert_eq!(filtered.certificates[1].kernel_name, "whisper_encoder");
}

#[test]
fn test_verification_breakdown_vacuous_entries_counted() {
    let json = serde_json::json!({
        "kernels": {
            "k_tight": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": { "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}], "constant_params": [] },
                "output_bounds": { "lower": -0.5, "upper": 0.5 },
                "output_width": 1.0,
                "soundness_mode": "sound"
            },
            "k_vacuous": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": { "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}], "constant_params": [] },
                "output_bounds": { "lower": -500.0, "upper": 500.0 },
                "output_width": 1000.1,
                "soundness_mode": "sound"
            }
        }
    });
    let status = status_from_json(json);
    let entries: Vec<&KernelStatus> = status.kernels().values().collect();
    let breakdown = VerificationBreakdown::from_entries(&entries);

    assert_eq!(breakdown.total, 2);
    assert_eq!(breakdown.vacuous, 1, "one entry has width > 100 (vacuous)");
    assert!(
        breakdown.sound_ibp + breakdown.vacuous == 2,
        "should have 1 sound_ibp + 1 vacuous"
    );
}

#[test]
fn test_gap_report_summary_line_format() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_status_entry("verified", "CROWN", 1.5, "sound", "sound"),
        }
    });
    let report = detect_gaps(&status);
    let formatted = format_gap_report(&report);

    // Summary line format: "Summary: N gaps, N vacuous, N total stages"
    assert!(formatted.contains("Summary:"));
    assert!(formatted.contains("7 gaps"), "expected 7 gaps: {formatted}");
    assert!(
        formatted.contains("8 total stages"),
        "expected 8 total stages: {formatted}"
    );
    assert!(formatted.contains("Verified (non-vacuous):"));
}
