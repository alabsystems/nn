// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Gap detector pipeline tests.
//!
//! Covers: no-gap models, missing entries, vacuous entries, stale entries,
//! multi-model scenarios, report format, sound/heuristic counting, trend
//! comparison, kokoro status file format, classify_entry, method_is_crown,
//! and count_gaps_and_vacuous.
//!
//! Part of #2930, #3351.

use nn_verify::gap_detector::{
    detect_gaps, format_gap_report, kokoro_pipeline_stages, StageGapResult,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_entry(status: &str, method: &str, width: f64, proof_strength: &str) -> serde_json::Value {
    serde_json::json!({
        "status": status,
        "method": method,
        "output_width": width,
        "proof_strength": proof_strength,
        "soundness_mode": if proof_strength == "sound" { "sound" } else { "heuristic" }
    })
}

fn make_stale_entry(
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
        "soundness_mode": if proof_strength == "sound" { "sound" } else { "heuristic" },
        "stale": true
    })
}

/// Build a fully verified status file with all 8 kokoro pipeline stages covered.
fn fully_verified_status() -> serde_json::Value {
    serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_entry("verified", "IBP", 10.0, "sound"),
            "kokoro_production_bert_encoder_crown": make_entry("verified", "CROWN", 1.4, "sound"),
            "kokoro_production_text_encoder": make_entry("verified", "IBP", 1.4, "sound"),
            "kokoro_production_text_encoder_crown": make_entry("verified", "CROWN", 1.4, "sound"),
            "kokoro_production_prosody_predictor": make_entry("verified", "IBP", 5.0, "sound"),
            "kokoro_production_f0_predictor": make_entry("verified", "IBP", 2.0, "sound"),
            "kokoro_production_generator": make_entry("verified", "IBP", 2.0, "sound"),
            "kokoro_production_length_regulate": make_entry("verified", "ANALYTICAL", 49.0, "sound"),
            "kokoro_production_harmonic_source": make_entry("verified", "ANALYTICAL", 12.6, "sound"),
            "kokoro_production_istft": make_entry("verified", "CROWN", 2.0, "sound"),
        }
    })
}

// ---------------------------------------------------------------------------
// B. Gap Detector Tests (10+)
// ---------------------------------------------------------------------------

/// Test 1: Fully covered model produces 0 gaps.
#[test]
fn test_gap_detector_no_gaps() {
    let status = fully_verified_status();
    let report = detect_gaps(&status);

    assert_eq!(
        report.total_gaps, 0,
        "fully verified model should have 0 gaps"
    );
    assert_eq!(report.stages.len(), 8);
    for stage in &report.stages {
        assert!(
            stage.has_any_bounds(),
            "stage {} should have bounds",
            stage.stage.name
        );
    }
}

/// Test 2: Uncovered stage reported as gap.
#[test]
fn test_gap_detector_missing_entry() {
    // Only verify 5 compiled segments, skip 3 bridges
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_entry("verified", "IBP", 10.0, "sound"),
            "kokoro_production_text_encoder": make_entry("verified", "IBP", 1.4, "sound"),
            "kokoro_production_prosody_predictor": make_entry("verified", "IBP", 5.0, "sound"),
            "kokoro_production_f0_predictor": make_entry("verified", "IBP", 2.0, "sound"),
            "kokoro_production_generator": make_entry("verified", "IBP", 2.0, "sound"),
        }
    });

    let report = detect_gaps(&status);
    assert_eq!(report.total_gaps, 3, "3 bridge stages should be missing");

    let gap_names: Vec<&str> = report
        .stages
        .iter()
        .filter(|r| !r.has_any_bounds())
        .map(|r| r.stage.name)
        .collect();
    assert!(gap_names.iter().any(|n| n.contains("length_regulate")));
    assert!(gap_names.iter().any(|n| n.contains("harmonic_source")));
    assert!(gap_names.iter().any(|n| n.contains("iSTFT")));
}

/// Test 3: Vacuous entries counted separately from gaps.
#[test]
fn test_gap_detector_vacuous_entry() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_entry("verified", "IBP", 10.0, "sound"),
            "kokoro_production_text_encoder": make_entry("verified", "IBP", 1.4, "sound"),
            "kokoro_production_prosody_predictor": make_entry("verified", "IBP", 345.2, "vacuous"),
            "kokoro_production_f0_predictor": make_entry("verified", "IBP", 2.0, "heuristic"),
            "kokoro_production_generator": make_entry("verified", "IBP", 2.0, "heuristic"),
            "kokoro_production_length_regulate": make_entry("verified", "IBP", 5.0, "sound"),
            "kokoro_production_harmonic_source": make_entry("verified", "IBP", 10.0, "sound"),
            "kokoro_production_istft": make_entry("verified", "CROWN", 2.0, "sound"),
        }
    });

    let report = detect_gaps(&status);
    assert_eq!(report.total_gaps, 0, "all stages are covered");
    assert_eq!(
        report.vacuous_count, 1,
        "prosody predictor has vacuous proof_strength"
    );

    let vacuous: Vec<&StageGapResult> = report.stages.iter().filter(|r| r.is_vacuous).collect();
    assert_eq!(vacuous.len(), 1);
    assert!(vacuous[0].stage.name.contains("Prosody"));
}

/// Test 4: Stale entries in status file data (stale field present).
#[test]
fn test_gap_detector_stale_entry() {
    // The gap detector operates on serde_json::Value and checks the "status" field,
    // not the "stale" field. Stale entries with "verified" status still count as
    // having bounds. This test documents this behavior.
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_stale_entry("verified", "IBP", 10.0, "sound"),
            "kokoro_production_text_encoder": make_entry("verified", "IBP", 1.4, "sound"),
            "kokoro_production_prosody_predictor": make_entry("verified", "IBP", 5.0, "sound"),
            "kokoro_production_f0_predictor": make_entry("verified", "IBP", 2.0, "heuristic"),
            "kokoro_production_generator": make_entry("verified", "IBP", 2.0, "heuristic"),
            "kokoro_production_length_regulate": make_entry("verified", "IBP", 5.0, "sound"),
            "kokoro_production_harmonic_source": make_entry("verified", "IBP", 10.0, "sound"),
            "kokoro_production_istft": make_entry("verified", "CROWN", 2.0, "sound"),
        }
    });

    let report = detect_gaps(&status);
    // Stale entries still count as having bounds in the gap detector
    // (staleness is a status_report concern, not a gap_detector concern)
    assert_eq!(report.total_gaps, 0);

    // Verify the bert encoder stage still has bounds
    let bert = report
        .stages
        .iter()
        .find(|r| r.stage.name.contains("PlBert"))
        .unwrap();
    assert!(bert.has_any_bounds());
}

/// Test 5: Multiple models — gap report is per-pipeline (Kokoro focused).
#[test]
fn test_gap_detector_multi_model() {
    // The gap detector is Kokoro-specific. Extra non-kokoro keys should not
    // affect the Kokoro gap report.
    let status = serde_json::json!({
        "kernels": {
            // Kokoro entries
            "kokoro_production_bert_encoder": make_entry("verified", "IBP", 10.0, "sound"),
            "kokoro_production_text_encoder": make_entry("verified", "IBP", 1.4, "sound"),
            "kokoro_production_prosody_predictor": make_entry("verified", "IBP", 5.0, "sound"),
            "kokoro_production_f0_predictor": make_entry("verified", "IBP", 2.0, "sound"),
            "kokoro_production_generator": make_entry("verified", "IBP", 2.0, "sound"),
            "kokoro_production_length_regulate": make_entry("verified", "IBP", 5.0, "sound"),
            "kokoro_production_harmonic_source": make_entry("verified", "IBP", 10.0, "sound"),
            "kokoro_production_istft": make_entry("verified", "CROWN", 2.0, "sound"),
            // Non-kokoro entries should be ignored
            "demucs_encoder_stage0": make_entry("verified", "IBP", 100.0, "heuristic"),
            "whisper_encoder_layer0": make_entry("verified", "CROWN", 5.0, "sound"),
        }
    });

    let report = detect_gaps(&status);
    assert_eq!(report.stages.len(), 8, "only Kokoro pipeline stages");
    assert_eq!(report.total_gaps, 0);
}

/// Test 6: Report format contains expected fields and sections.
#[test]
fn test_gap_detector_report_format() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_entry("verified", "IBP", 10.0, "sound"),
            "kokoro_production_istft": make_entry("verified", "CROWN", 2.0, "sound"),
        }
    });

    let report = detect_gaps(&status);
    let formatted = format_gap_report(&report);

    // Header
    assert!(formatted.contains("Kokoro Pipeline Bound Propagation Gap Report"));
    // Stage names
    assert!(formatted.contains("PlBert + bert_encoder"));
    assert!(formatted.contains("iSTFT"));
    // Status markers
    assert!(formatted.contains("[!!]"), "should have GAP markers");
    assert!(formatted.contains("[OK]"), "should have OK markers");
    // Source file references
    assert!(formatted.contains("source:"));
    assert!(formatted.contains("status_key:"));
    // Summary line
    assert!(formatted.contains("Summary:"));
    assert!(formatted.contains("gaps"));
    assert!(formatted.contains("vacuous"));
    assert!(formatted.contains("total stages"));
    // Verified count
    assert!(formatted.contains("Verified (non-vacuous):"));
}

/// Test 7: Sound entries are correctly counted in the report.
#[test]
fn test_gap_detector_counts_sound() {
    let status = fully_verified_status();
    let report = detect_gaps(&status);

    let sound_count = report
        .stages
        .iter()
        .filter(|r| r.proof_strength.as_deref() == Some("sound"))
        .count();
    // All entries in fully_verified_status are "sound"
    assert_eq!(
        sound_count, 8,
        "all 8 stages should have sound proof_strength"
    );
}

/// Test 8: Heuristic entries are correctly counted.
#[test]
fn test_gap_detector_counts_heuristic() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_entry("verified", "IBP", 300.0, "heuristic"),
            "kokoro_production_text_encoder": make_entry("verified", "IBP", 1.4, "sound"),
            "kokoro_production_prosody_predictor": make_entry("verified", "IBP", 345.2, "heuristic"),
            "kokoro_production_f0_predictor": make_entry("verified", "IBP", 2.0, "heuristic"),
            "kokoro_production_generator": make_entry("verified", "IBP", 2.0, "sound"),
            "kokoro_production_length_regulate": make_entry("verified", "IBP", 5.0, "sound"),
            "kokoro_production_harmonic_source": make_entry("verified", "IBP", 10.0, "sound"),
            "kokoro_production_istft": make_entry("verified", "CROWN", 2.0, "sound"),
        }
    });

    let report = detect_gaps(&status);

    let heuristic_count = report
        .stages
        .iter()
        .filter(|r| r.proof_strength.as_deref() == Some("heuristic"))
        .count();
    assert_eq!(
        heuristic_count, 3,
        "bert_encoder, prosody, f0 should be heuristic"
    );

    let sound_count = report
        .stages
        .iter()
        .filter(|r| r.proof_strength.as_deref() == Some("sound"))
        .count();
    assert_eq!(sound_count, 5);
}

/// Test 9: Trend — before/after comparison of gap counts.
#[test]
fn test_gap_detector_trend() {
    // Before: only compiled segments verified
    let before = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_entry("verified", "IBP", 300.0, "heuristic"),
            "kokoro_production_text_encoder": make_entry("verified", "IBP", 1.4, "sound"),
            "kokoro_production_prosody_predictor": make_entry("verified", "IBP", 345.2, "vacuous"),
            "kokoro_production_f0_predictor": make_entry("verified", "IBP", 2.0, "heuristic"),
            "kokoro_production_generator": make_entry("verified", "IBP", 2.0, "heuristic"),
        }
    });

    // After: bridges added
    let after = fully_verified_status();

    let before_report = detect_gaps(&before);
    let after_report = detect_gaps(&after);

    // Gaps should decrease
    assert!(
        after_report.total_gaps < before_report.total_gaps,
        "gaps should decrease: {} -> {}",
        before_report.total_gaps,
        after_report.total_gaps
    );
    assert_eq!(before_report.total_gaps, 3);
    assert_eq!(after_report.total_gaps, 0);

    // Vacuous should decrease when proof strengths improve
    assert!(
        after_report.vacuous_count <= before_report.vacuous_count,
        "vacuous count should not increase"
    );
}

/// Test 10: Status file format matches kokoro production format.
#[test]
fn test_gap_detector_kokoro_format() {
    // Verify that all pipeline stage status_key values match the expected
    // kokoro_production_* prefix pattern
    let stages = kokoro_pipeline_stages();
    for stage in &stages {
        assert!(
            stage.status_key.starts_with("kokoro_production_"),
            "stage {} has non-standard status_key: {}",
            stage.name,
            stage.status_key
        );
    }

    // Verify format_gap_report output contains all stage names
    let status = fully_verified_status();
    let report = detect_gaps(&status);
    let formatted = format_gap_report(&report);

    for stage in &stages {
        assert!(
            formatted.contains(stage.name),
            "report should contain stage name: {}",
            stage.name
        );
    }
}

/// Test 11: Vacuous detection by width threshold (> 1000.0).
#[test]
fn test_gap_detector_vacuous_by_width_threshold() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_entry("verified", "IBP", 999.9, "sound"),
            "kokoro_production_text_encoder": make_entry("verified", "IBP", 1000.1, "sound"),
        }
    });

    let report = detect_gaps(&status);

    let bert = report
        .stages
        .iter()
        .find(|r| r.stage.name.contains("PlBert"))
        .unwrap();
    assert!(
        !bert.is_vacuous,
        "width 999.9 should NOT be vacuous (threshold is 1000.0)"
    );

    let text = report
        .stages
        .iter()
        .find(|r| r.stage.name == "TextEncoder")
        .unwrap();
    assert!(
        text.is_vacuous,
        "width 1000.1 should be vacuous (exceeds threshold)"
    );
}

/// Test 12: Alpha-CROWN, Beta-CROWN, and Mixed-IBP-CROWN variants detected as CROWN.
#[test]
fn test_gap_detector_crown_variants() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder_crown": make_entry("verified", "ALPHACROWN", 1.0, "sound"),
            "kokoro_production_text_encoder_crown": make_entry("verified", "BETACROWN", 1.0, "sound"),
            "kokoro_production_prosody_predictor_crown": make_entry("verified", "MIXED_IBP_CROWN", 5.0, "sound"),
        }
    });

    let report = detect_gaps(&status);

    let bert = report
        .stages
        .iter()
        .find(|r| r.stage.name.contains("PlBert"))
        .unwrap();
    assert!(
        bert.has_crown_bounds,
        "ALPHACROWN should be detected as CROWN"
    );

    let text = report
        .stages
        .iter()
        .find(|r| r.stage.name == "TextEncoder")
        .unwrap();
    assert!(
        text.has_crown_bounds,
        "BETACROWN should be detected as CROWN"
    );

    let prosody = report
        .stages
        .iter()
        .find(|r| r.stage.name.contains("Prosody"))
        .unwrap();
    assert!(
        prosody.has_crown_bounds,
        "MIXED_IBP_CROWN should be detected as CROWN"
    );
}

/// Test 13: Empty kernels object produces all gaps with no vacuous.
#[test]
fn test_gap_detector_empty_kernels() {
    let status = serde_json::json!({ "kernels": {} });
    let report = detect_gaps(&status);

    assert_eq!(report.stages.len(), 8);
    assert_eq!(report.total_gaps, 8, "all stages should be gaps");
    assert_eq!(report.vacuous_count, 0, "no entries means no vacuous");

    // Verify no stage has any bounds
    for stage in &report.stages {
        assert!(!stage.has_any_bounds());
        assert!(!stage.has_ibp_bounds);
        assert!(!stage.has_crown_bounds);
        assert!(!stage.has_analytical_bounds);
        assert!(!stage.is_vacuous);
    }
}

/// Test 14: Missing top-level "kernels" key handled gracefully.
#[test]
fn test_gap_detector_missing_kernels_key() {
    let status = serde_json::json!({});
    let report = detect_gaps(&status);

    assert_eq!(report.stages.len(), 8);
    assert_eq!(report.total_gaps, 8);
    assert_eq!(report.vacuous_count, 0);
}

/// Test 15: Format report with vacuous entries shows [~~] marker.
#[test]
fn test_gap_detector_format_vacuous_marker() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_prosody_predictor": make_entry("verified", "IBP", 345.2, "vacuous"),
        }
    });

    let report = detect_gaps(&status);
    let formatted = format_gap_report(&report);

    assert!(
        formatted.contains("[~~]"),
        "vacuous entries should show [~~] marker: {formatted}"
    );
    assert!(
        formatted.contains("VACUOUS"),
        "vacuous entries should be labeled VACUOUS"
    );
}

/// Test 16: Bound width is reported in format output.
#[test]
fn test_gap_detector_format_bound_width() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_entry("verified", "IBP", 300.5, "heuristic"),
        }
    });

    let report = detect_gaps(&status);
    let formatted = format_gap_report(&report);

    assert!(
        formatted.contains("bound_width:"),
        "should contain bound_width"
    );
    assert!(
        formatted.contains("300.5"),
        "should contain the actual width value"
    );
}

/// Test 17: Soundness mode is captured from status entries.
#[test]
fn test_gap_detector_captures_soundness_mode() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_entry("verified", "IBP", 10.0, "sound"),
            "kokoro_production_text_encoder": make_entry("verified", "IBP", 1.4, "heuristic"),
        }
    });

    let report = detect_gaps(&status);

    let bert = report
        .stages
        .iter()
        .find(|r| r.stage.name.contains("PlBert"))
        .unwrap();
    assert_eq!(bert.soundness_mode.as_deref(), Some("sound"));

    let text = report
        .stages
        .iter()
        .find(|r| r.stage.name == "TextEncoder")
        .unwrap();
    assert_eq!(text.soundness_mode.as_deref(), Some("heuristic"));
}
