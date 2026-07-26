// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn make_entry(status: &str, method: &str, width: f64, proof_strength: &str) -> serde_json::Value {
    serde_json::json!({
        "status": status,
        "method": method,
        "output_width": width,
        "proof_strength": proof_strength,
        "soundness_mode": if proof_strength == "sound" { "sound" } else { "heuristic" }
    })
}

#[test]
fn test_detect_gaps_all_verified() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_entry("verified", "IBP", 300.0, "heuristic"),
            "kokoro_production_bert_encoder_crown": make_entry("verified", "CROWN", 1.4, "sound"),
            "kokoro_production_text_encoder": make_entry("verified", "IBP", 1.4, "sound"),
            "kokoro_production_text_encoder_crown": make_entry("verified", "CROWN", 1.4, "sound"),
            "kokoro_production_prosody_predictor": make_entry("verified", "IBP", 345.2, "vacuous"),
            "kokoro_production_f0_predictor": make_entry("verified", "IBP", 2.0, "heuristic"),
            "kokoro_production_generator": make_entry("verified", "IBP", 2.0, "heuristic"),
            "kokoro_production_length_regulate": make_entry("verified", "IBP", 5.0, "sound"),
            "kokoro_production_harmonic_source": make_entry("verified", "IBP", 10.0, "sound"),
            "kokoro_production_istft": make_entry("verified", "CROWN", 2.0, "sound"),
        }
    });

    let report = detect_gaps(&status);
    assert_eq!(report.total_gaps, 0);
    // ProsodyPredictor has proof_strength: "vacuous"
    assert_eq!(report.vacuous_count, 1);
}

#[test]
fn test_detect_gaps_bridge_gaps() {
    // Typical state: compiled segments verified, bridges missing
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_entry("verified", "IBP", 300.0, "vacuous"),
            "kokoro_production_bert_encoder_crown": make_entry("verified", "CROWN", 1.4, "sound"),
            "kokoro_production_text_encoder": make_entry("verified", "IBP", 1.4, "sound"),
            "kokoro_production_text_encoder_crown": make_entry("verified", "CROWN", 1.4, "sound"),
            "kokoro_production_prosody_predictor": make_entry("verified", "IBP", 345.2, "vacuous"),
            "kokoro_production_f0_predictor": make_entry("verified", "IBP", 2.0, "heuristic"),
            "kokoro_production_generator": make_entry("verified", "IBP", 2.0, "heuristic"),
        }
    });

    let report = detect_gaps(&status);
    assert_eq!(report.total_gaps, 3, "3 bridge stages should be gaps");

    let gap_names: Vec<&str> = report
        .stages
        .iter()
        .filter(|r| !r.has_any_bounds())
        .map(|r| r.stage.name)
        .collect();
    assert!(gap_names.contains(&"length_regulate (sigmoid+sum+floor+clamp+repeat_interleave)"));
    assert!(gap_names.contains(&"harmonic_source (SineGen + forward STFT)"));
    assert!(gap_names.contains(&"iSTFT (polar_to_rect + frequency-to-time)"));
}

#[test]
fn test_detect_gaps_empty_status() {
    let status = serde_json::json!({ "kernels": {} });
    let report = detect_gaps(&status);
    assert_eq!(report.total_gaps, 8, "all 8 stages should be gaps");
    assert_eq!(report.vacuous_count, 0, "no entries means no vacuous");
}

#[test]
fn test_detect_gaps_bounds_computed_status() {
    // Some entries use "bounds_computed" instead of "verified"
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_entry("bounds_computed", "IBP", 300.0, "heuristic"),
        }
    });

    let report = detect_gaps(&status);
    let bert = report
        .stages
        .iter()
        .find(|r| r.stage.name == "PlBert + bert_encoder")
        .unwrap();
    assert!(
        bert.has_ibp_bounds,
        "bounds_computed should count as having bounds"
    );
}

#[test]
fn test_detect_gaps_vacuous_by_width() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_entry("verified", "IBP", 5000.0, "heuristic"),
        }
    });

    let report = detect_gaps(&status);
    let bert = report
        .stages
        .iter()
        .find(|r| r.stage.name == "PlBert + bert_encoder")
        .unwrap();
    assert!(bert.is_vacuous, "width > 1000 should be vacuous");
    assert_eq!(report.vacuous_count, 1);
}

#[test]
fn test_detect_gaps_crown_detection() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_text_encoder": make_entry("verified", "IBP", 1.4, "sound"),
            "kokoro_production_text_encoder_crown": make_entry("verified", "CROWN", 1.4, "sound"),
        }
    });

    let report = detect_gaps(&status);
    let te = report
        .stages
        .iter()
        .find(|r| r.stage.name == "TextEncoder")
        .unwrap();
    assert!(te.has_ibp_bounds);
    assert!(te.has_crown_bounds);
}

#[test]
fn test_detect_gaps_crown_suffix_with_ibp_method_is_not_crown() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_f0_predictor": make_entry("verified", "IBP", 2.0, "heuristic"),
            "kokoro_production_f0_predictor_crown": make_entry("verified", "IBP", 2.0, "heuristic"),
        }
    });

    let report = detect_gaps(&status);
    let f0 = report
        .stages
        .iter()
        .find(|r| r.stage.name == "F0EnergyPredictor")
        .unwrap();
    assert!(
        f0.has_ibp_bounds,
        "IBP fallback entry still provides bounds"
    );
    assert!(
        !f0.has_crown_bounds,
        "an IBP fallback recorded under a _crown suffix must not count as CROWN coverage"
    );
}

/// CROWN-primary entries (no separate `_crown` suffix) are detected.
///
/// Some stages (e.g., iSTFT) use CROWN as the primary method because the
/// transform is linear — no IBP entry exists, only a CROWN one.
#[test]
fn test_detect_gaps_crown_primary_method() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_istft": make_entry("verified", "CROWN", 2.0, "sound"),
        }
    });

    let report = detect_gaps(&status);
    let istft = report
        .stages
        .iter()
        .find(|r| r.stage.name.contains("iSTFT"))
        .unwrap();
    assert!(
        istft.has_crown_bounds,
        "CROWN-primary entry should be detected"
    );
    assert!(!istft.is_vacuous);
    // Not an IBP entry — primary method is CROWN.
    assert!(!istft.has_ibp_bounds);
}

#[test]
fn test_pipeline_stages_registry_completeness() {
    let stages = kokoro_pipeline_stages();
    assert_eq!(stages.len(), 8, "5 compiled segments + 3 bridges");

    let compiled: Vec<_> = stages.iter().filter(|s| s.is_compiled_segment).collect();
    assert_eq!(compiled.len(), 5);

    let bridges: Vec<_> = stages.iter().filter(|s| s.is_bridge).collect();
    assert_eq!(bridges.len(), 3);

    // No stage should be both compiled and bridge
    for stage in &stages {
        assert!(
            !(stage.is_compiled_segment && stage.is_bridge),
            "{} is both compiled and bridge",
            stage.name
        );
    }

    // All stages have source file locations
    for stage in &stages {
        assert!(
            !stage.source_file.is_empty(),
            "{} has empty source_file",
            stage.name
        );
    }
}

/// CPU bridges are only declared for bridge stages, not compiled segments.
#[test]
fn test_cpu_bridges_only_on_bridge_stages() {
    let stages = kokoro_pipeline_stages();
    for stage in &stages {
        if stage.is_compiled_segment {
            assert!(
                stage.cpu_bridges.is_empty(),
                "{}: compiled segment should have no CPU bridges",
                stage.name
            );
        }
    }
}

/// harmonic_source has no CPU bridges after Kahan GPU cumsum (#2909).
#[test]
fn test_harmonic_source_has_sinegen_bridge() {
    let stages = kokoro_pipeline_stages();
    let hs = stages
        .iter()
        .find(|s| s.name.contains("harmonic_source"))
        .unwrap();
    assert!(
        hs.cpu_bridges.is_empty(),
        "SineGen cumsum is fully GPU-native (#2909)"
    );
}

/// Format report produces readable output with file:line locations.
#[test]
fn test_format_gap_report() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_bert_encoder": make_entry("verified", "IBP", 300.0, "heuristic"),
            "kokoro_production_istft": make_entry("verified", "CROWN", 2.0, "sound"),
        }
    });

    let report = detect_gaps(&status);
    let formatted = format_gap_report(&report);

    assert!(formatted.contains("PlBert + bert_encoder"));
    assert!(formatted.contains("compiled_kokoro_segments.rs"));
    assert!(formatted.contains("[!!]"), "should have GAP markers");
    assert!(
        formatted.contains("[OK]"),
        "should have CROWN marker for iSTFT"
    );
    assert!(formatted.contains("Summary:"));
    assert!(formatted.contains("Verified (non-vacuous):"));
    eprintln!("{formatted}");
}

/// Analytical bounds are detected and not labeled as gaps.
#[test]
fn test_detect_gaps_analytical_bounds() {
    let status = serde_json::json!({
        "kernels": {
            "kokoro_production_length_regulate": make_entry("verified", "ANALYTICAL", 49.0, "sound"),
            "kokoro_production_harmonic_source": make_entry("verified", "ANALYTICAL", 12.6, "sound"),
        }
    });

    let report = detect_gaps(&status);

    let regulate = report
        .stages
        .iter()
        .find(|r| r.stage.name.contains("length_regulate"))
        .unwrap();
    assert!(
        regulate.has_analytical_bounds,
        "ANALYTICAL method should set has_analytical_bounds"
    );
    assert!(!regulate.has_ibp_bounds, "ANALYTICAL is not IBP");
    assert!(!regulate.has_crown_bounds, "ANALYTICAL is not CROWN");
    assert!(
        regulate.has_any_bounds(),
        "analytical should count as having bounds"
    );

    let harmonic = report
        .stages
        .iter()
        .find(|r| r.stage.name.contains("harmonic_source"))
        .unwrap();
    assert!(harmonic.has_analytical_bounds);
    assert!(harmonic.has_any_bounds());

    // Analytical entries should NOT be counted as gaps.
    let gap_count = report.stages.iter().filter(|r| !r.has_any_bounds()).count();
    // 8 total stages - 2 analytical = 6 gaps
    assert_eq!(gap_count, 6, "only non-analytical stages should be gaps");

    // Format report should show [OK] ANALYTICAL, not [!!] GAP.
    let formatted = format_gap_report(&report);
    assert!(
        formatted.contains("(ANALYTICAL)"),
        "analytical stages should be labeled ANALYTICAL in report"
    );
}

// ---------------------------------------------------------------------------
// Gap severity classification tests
// ---------------------------------------------------------------------------

/// Stages with no bounds at all are the most severe gaps (unverified).
#[test]
fn test_gap_severity_unverified_is_worst() {
    let status = serde_json::json!({ "kernels": {} });
    let report = detect_gaps(&status);
    // Every stage should be a gap with no bounds of any kind.
    for result in &report.stages {
        assert!(
            !result.has_any_bounds(),
            "{} should have no bounds",
            result.stage.name
        );
        assert!(!result.is_vacuous, "unverified stage should not be vacuous");
    }
}

/// Vacuous bounds (width > 1000) are better than no bounds but still a problem.
#[test]
fn test_gap_severity_vacuous_has_bounds_but_useless() {
    let stages = kokoro_pipeline_stages();
    let key = stages[0].status_key;
    let status = serde_json::json!({
        "kernels": {
            key: make_entry("verified", "IBP", 5000.0, "heuristic"),
        }
    });
    let report = detect_gaps(&status);
    let result = report
        .stages
        .iter()
        .find(|r| r.stage.status_key == key)
        .unwrap();
    // Has bounds but vacuous.
    assert!(result.has_any_bounds(), "vacuous still has bounds");
    assert!(result.is_vacuous, "width 5000 should be vacuous");
    assert_eq!(
        report.total_gaps,
        stages.len() - 1,
        "vacuous does not count as gap"
    );
}

/// CROWN bounds are the highest quality (tight, non-vacuous).
#[test]
fn test_gap_severity_crown_is_best() {
    let stages = kokoro_pipeline_stages();
    let key = stages[0].status_key;
    let crown_key = format!("{key}_crown");
    let status = serde_json::json!({
        "kernels": {
            key: make_entry("verified", "IBP", 1.5, "sound"),
            crown_key: make_entry("verified", "CROWN", 0.8, "sound"),
        }
    });
    let report = detect_gaps(&status);
    let result = report
        .stages
        .iter()
        .find(|r| r.stage.status_key == key)
        .unwrap();
    assert!(result.has_ibp_bounds);
    assert!(result.has_crown_bounds);
    assert!(!result.is_vacuous);
    assert_eq!(result.proof_strength.as_deref(), Some("sound"));
}

/// Verify that the format report classifies severity labels correctly
/// in the output text: GAP > VACUOUS > IBP > CROWN/ANALYTICAL.
#[test]
fn test_format_gap_report_severity_labels() {
    let stages = kokoro_pipeline_stages();
    // Set up: first stage CROWN, second IBP, third vacuous, rest gaps.
    let key0 = stages[0].status_key;
    let crown_key0 = format!("{key0}_crown");
    let key1 = stages[1].status_key;
    let key2 = stages[2].status_key;
    let status = serde_json::json!({
        "kernels": {
            key0: make_entry("verified", "IBP", 1.5, "sound"),
            crown_key0: make_entry("verified", "CROWN", 0.8, "sound"),
            key1: make_entry("verified", "IBP", 50.0, "heuristic"),
            key2: make_entry("verified", "IBP", 2000.0, "vacuous"),
        }
    });
    let report = detect_gaps(&status);
    let formatted = format_gap_report(&report);
    // Should contain all severity labels.
    assert!(formatted.contains("(CROWN)"), "should have CROWN label");
    assert!(formatted.contains("(IBP)"), "should have IBP label");
    assert!(formatted.contains("(VACUOUS)"), "should have VACUOUS label");
    assert!(formatted.contains("(GAP)"), "should have GAP label");
}

// ---------------------------------------------------------------------------
// Additional edge cases
// ---------------------------------------------------------------------------

/// When primary entry exists but has unknown status (e.g., "pending"), it's a gap.
#[test]
fn test_detect_gaps_pending_status_is_gap() {
    let stages = kokoro_pipeline_stages();
    let key = stages[0].status_key;
    let status = serde_json::json!({
        "kernels": {
            key: { "status": "pending", "method": "IBP" },
        }
    });
    let report = detect_gaps(&status);
    let result = report
        .stages
        .iter()
        .find(|r| r.stage.status_key == key)
        .unwrap();
    assert!(
        !result.has_any_bounds(),
        "pending status should not count as having bounds"
    );
}

/// Width exactly at the vacuous threshold (1000.0) is NOT vacuous (threshold is >).
#[test]
fn test_detect_gaps_width_at_exact_threshold_not_vacuous() {
    let stages = kokoro_pipeline_stages();
    let key = stages[0].status_key;
    let status = serde_json::json!({
        "kernels": {
            key: make_entry("verified", "IBP", 1000.0, "heuristic"),
        }
    });
    let report = detect_gaps(&status);
    let result = report
        .stages
        .iter()
        .find(|r| r.stage.status_key == key)
        .unwrap();
    assert!(
        !result.is_vacuous,
        "width exactly at 1000.0 should NOT be vacuous (> not >=)"
    );
}

/// Width just above threshold (1000.01) IS vacuous.
#[test]
fn test_detect_gaps_width_just_above_threshold_is_vacuous() {
    let stages = kokoro_pipeline_stages();
    let key = stages[0].status_key;
    let status = serde_json::json!({
        "kernels": {
            key: make_entry("verified", "IBP", 1000.01, "heuristic"),
        }
    });
    let report = detect_gaps(&status);
    let result = report
        .stages
        .iter()
        .find(|r| r.stage.status_key == key)
        .unwrap();
    assert!(result.is_vacuous, "width 1000.01 should be vacuous");
}

/// Both primary and crown entries present with different methods.
#[test]
fn test_detect_gaps_both_ibp_and_crown() {
    let stages = kokoro_pipeline_stages();
    let key = stages[0].status_key;
    let crown_key = format!("{key}_crown");
    let status = serde_json::json!({
        "kernels": {
            key: make_entry("verified", "IBP", 300.0, "heuristic"),
            crown_key: make_entry("verified", "AlphaCROWN", 1.2, "sound"),
        }
    });
    let report = detect_gaps(&status);
    let result = report
        .stages
        .iter()
        .find(|r| r.stage.status_key == key)
        .unwrap();
    assert!(result.has_ibp_bounds, "should have IBP from primary");
    assert!(
        result.has_crown_bounds,
        "should have CROWN from crown suffix"
    );
    assert!(!result.has_analytical_bounds);
}

/// Integration test: load the actual nn_verify_status_kokoro.json and report gaps.
///
/// This test does NOT assert zero gaps — it documents current state.
/// The gap list changes as verification work progresses.
#[test]
fn test_detect_gaps_real_status_file() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = std::path::Path::new(manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let status_path = workspace_root.join("nn_verify_status_kokoro.json");

    if !status_path.exists() {
        eprintln!(
            "Skipping: nn_verify_status_kokoro.json not found at {}",
            status_path.display()
        );
        return;
    }

    let content = std::fs::read_to_string(&status_path).expect("read status file");
    let status: serde_json::Value = serde_json::from_str(&content).expect("parse JSON");

    let report = detect_gaps(&status);
    let formatted = format_gap_report(&report);
    eprintln!("\n{formatted}");

    // Document current state (not an assertion — gap count decreases over time).
    eprintln!(
        "Current pipeline verification: {gaps} gaps, {vacuous} vacuous out of {total} stages",
        gaps = report.total_gaps,
        vacuous = report.vacuous_count,
        total = report.stages.len(),
    );

    // Structural assertions that should always hold.
    assert_eq!(report.stages.len(), 8, "pipeline should have 8 stages");

    // Raw kernels object, used to distinguish two distinct no-bounds causes:
    //   (a) the stage entry is ABSENT from the parsed status — nothing was
    //       recorded for it this run (e.g. the deprecated standalone
    //       bert_encoder writer is feature-gated and writes no entry in a
    //       normal run). This is skippable, not a regression.
    //   (b) the stage entry is PRESENT but failed to verify — a real
    //       regression we must keep asserting on.
    // Mirrors the status_load_diagnostic skip idiom above (file-absent
    // early-return): eprintln a clear skip note and continue rather than fail.
    let kernels = status.get("kernels").and_then(|k| k.as_object());
    let stage_recorded = |result: &StageGapResult| -> bool {
        let key = result.stage.status_key;
        let crown_key = format!("{key}_crown");
        kernels.is_some_and(|k| k.contains_key(key) || k.contains_key(&crown_key))
    };

    // All compiled segments should have at least IBP bounds — but only when an
    // entry was actually recorded for the stage this run.
    for result in &report.stages {
        if result.stage.is_compiled_segment {
            if !stage_recorded(result) {
                eprintln!(
                    "Skipping: compiled segment {} has no recorded status entry \
                     this run (writer absent/feature-gated) — not a regression",
                    result.stage.name
                );
                continue;
            }
            assert!(
                result.has_ibp_bounds || result.has_crown_bounds,
                "compiled segment {} has no bounds — this is a regression",
                result.stage.name
            );
        }
    }

    // iSTFT should have CROWN bounds (we just added this) — but only when an
    // iSTFT entry was actually recorded this run. The presence guard also
    // removes the find().unwrap() panic when the iSTFT stage is absent.
    if let Some(istft) = report
        .stages
        .iter()
        .find(|r| r.stage.name.contains("iSTFT"))
    {
        if stage_recorded(istft) {
            assert!(
                istft.has_crown_bounds,
                "iSTFT should have CROWN bounds after #2916"
            );
        } else {
            eprintln!(
                "Skipping: iSTFT has no recorded status entry this run \
                 (writer absent/feature-gated) — not a regression"
            );
        }
    }
}
