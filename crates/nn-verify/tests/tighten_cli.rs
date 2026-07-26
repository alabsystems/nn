// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use nn_verify::bound_analysis::{analyze_layer_bounds, AnalysisConfig, BoundAnalysisReport};
use nn_verify::{LayerBoundRecord, PropMethod};

fn report(
    model_name: &str,
    output_width: f32,
    explosion_points: Vec<usize>,
    crown_coverage: f32,
    precision_drift_ratio: Option<f32>,
) -> BoundAnalysisReport {
    let mut report = analyze_layer_bounds(
        model_name,
        &[LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(0.0, output_width.max(1.0))],
            output_bounds: vec![(-output_width / 2.0, output_width / 2.0)],
            method: PropMethod::Ibp,
            node_name: Some("n0".to_string()),
            input_sources: Some(vec![]),
        }],
        &AnalysisConfig::default(),
    );
    report.explosion_points = explosion_points;
    report.output_width = output_width;
    report.output_is_finite = output_width.is_finite();
    report.crown_coverage = crown_coverage;
    report.precision_drift_ratio = precision_drift_ratio;
    report.drift_per_layer = None;
    report
}

fn write_report(path: &PathBuf, report: &BoundAnalysisReport) {
    let json = serde_json::to_string_pretty(report).expect("serialize report");
    fs::write(path, json).expect("write report");
}

fn temp_path(prefix: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{unique}-{}", std::process::id()))
}

#[test]
fn tighten_report_diff_json_emits_expected_diff() {
    let base_dir = temp_path("nn-verify-tighten");
    fs::create_dir_all(&base_dir).expect("create temp dir");

    let baseline_path = base_dir.join("baseline.json");
    let candidate_path = base_dir.join("candidate.json");
    write_report(
        &baseline_path,
        &report("kokoro", 10.0, vec![1, 2], 0.25, Some(0.90)),
    );
    write_report(
        &candidate_path,
        &report("kokoro", 8.0, vec![1], 0.50, Some(0.95)),
    );

    let output = Command::new(env!("CARGO_BIN_EXE_tighten"))
        .args([
            "kokoro",
            "report-diff",
            "--baseline",
            baseline_path.to_str().expect("baseline path"),
            "--candidate",
            candidate_path.to_str().expect("candidate path"),
            "--json",
        ])
        .output()
        .expect("run tighten binary");

    assert!(output.status.success(), "{output:?}");

    let json: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("parse tighten diff JSON");
    assert_eq!(json["outcome"], "Improved");
    assert_eq!(json["previous_model_name"], "kokoro");
    assert_eq!(json["current_model_name"], "kokoro");
    assert_eq!(json["output_width_before"], 10.0);
    assert_eq!(json["output_width_after"], 8.0);
    assert_eq!(json["explosion_points_before"], 2);
    assert_eq!(json["explosion_points_after"], 1);
}

#[test]
fn tighten_report_diff_multi_pass_json_emits_sequence_summary() {
    let base_dir = temp_path("nn-verify-tighten-sequence");
    fs::create_dir_all(&base_dir).expect("create temp dir");

    let baseline_path = base_dir.join("baseline.json");
    let improved_path = base_dir.join("pass1-improved.json");
    let stalled_path = base_dir.join("pass2-stalled.json");
    let regressed_path = base_dir.join("pass3-regressed.json");

    write_report(
        &baseline_path,
        &report("kokoro", 10.0, vec![1, 2], 0.25, Some(0.90)),
    );
    write_report(
        &improved_path,
        &report("kokoro", 8.0, vec![1], 0.50, Some(0.95)),
    );
    write_report(
        &stalled_path,
        &report("kokoro", 8.0, vec![1], 0.50, Some(0.95)),
    );
    write_report(
        &regressed_path,
        &report("kokoro", 9.0, vec![1, 3], 0.45, Some(0.92)),
    );

    let output = Command::new(env!("CARGO_BIN_EXE_tighten"))
        .args([
            "kokoro",
            "report-diff",
            "--baseline",
            baseline_path.to_str().expect("baseline path"),
            "--candidate",
            improved_path.to_str().expect("improved path"),
            "--candidate",
            stalled_path.to_str().expect("stalled path"),
            "--candidate",
            regressed_path.to_str().expect("regressed path"),
            "--json",
        ])
        .output()
        .expect("run tighten binary");

    assert!(output.status.success(), "{output:?}");

    let json: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("parse tighten sequence JSON");
    assert_eq!(json["overall_outcome"], "Improved");
    assert_eq!(json["terminal_outcome"], "Regressed");
    assert_eq!(json["transition_count"], 3);
    assert_eq!(json["improvement_count"], 1);
    assert_eq!(json["stall_count"], 1);
    assert_eq!(json["regression_count"], 1);
    assert_eq!(json["monotonic"], false);
    assert_eq!(
        json["transitions"].as_array().expect("transitions").len(),
        3
    );
    assert_eq!(json["transitions"][0]["outcome"], "Improved");
    assert_eq!(json["transitions"][1]["outcome"], "Stalled");
    assert_eq!(json["transitions"][2]["outcome"], "Regressed");
}

#[test]
fn tighten_report_diff_multi_pass_text_summarizes_loop() {
    let base_dir = temp_path("nn-verify-tighten-sequence-text");
    fs::create_dir_all(&base_dir).expect("create temp dir");

    let baseline_path = base_dir.join("baseline.json");
    let improved_path = base_dir.join("pass1-improved.json");
    let stalled_path = base_dir.join("pass2-stalled.json");

    write_report(
        &baseline_path,
        &report("kokoro", 10.0, vec![1, 2], 0.25, Some(0.90)),
    );
    write_report(
        &improved_path,
        &report("kokoro", 8.0, vec![1], 0.50, Some(0.95)),
    );
    write_report(
        &stalled_path,
        &report("kokoro", 8.0, vec![1], 0.50, Some(0.95)),
    );

    let output = Command::new(env!("CARGO_BIN_EXE_tighten"))
        .args([
            "kokoro",
            "report-diff",
            baseline_path.to_str().expect("baseline path"),
            improved_path.to_str().expect("improved path"),
            stalled_path.to_str().expect("stalled path"),
        ])
        .output()
        .expect("run tighten binary");

    assert!(output.status.success(), "{output:?}");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Overall outcome: improved"), "{stdout}");
    assert!(stdout.contains("Latest transition: stalled"), "{stdout}");
    assert!(
        stdout.contains("Transitions: 2 (improved 1, stalled 1, regressed 0)"),
        "{stdout}"
    );
    assert!(stdout.contains("Pass 1: improved"), "{stdout}");
    assert!(stdout.contains("Pass 2: stalled"), "{stdout}");
    assert!(stdout.contains("Model path: kokoro -> kokoro"), "{stdout}");
}

#[test]
fn tighten_report_diff_multi_pass_overall_outcome_tracks_baseline_to_final_result() {
    let base_dir = temp_path("nn-verify-tighten-sequence-net-result");
    fs::create_dir_all(&base_dir).expect("create temp dir");

    let baseline_path = base_dir.join("baseline.json");
    let regressed_mid_path = base_dir.join("pass1-regressed.json");
    let final_improved_path = base_dir.join("pass2-improved.json");

    write_report(
        &baseline_path,
        &report("kokoro", 10.0, vec![1, 2], 0.25, Some(0.90)),
    );
    write_report(
        &regressed_mid_path,
        &report("kokoro", 12.0, vec![1, 2, 3], 0.20, Some(0.88)),
    );
    write_report(
        &final_improved_path,
        &report("kokoro", 8.0, vec![1], 0.50, Some(0.95)),
    );

    let output = Command::new(env!("CARGO_BIN_EXE_tighten"))
        .args([
            "kokoro",
            "report-diff",
            "--baseline",
            baseline_path.to_str().expect("baseline path"),
            "--candidate",
            regressed_mid_path.to_str().expect("regressed path"),
            "--candidate",
            final_improved_path.to_str().expect("improved path"),
            "--json",
        ])
        .output()
        .expect("run tighten binary");

    assert!(output.status.success(), "{output:?}");

    let json: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("parse tighten sequence JSON");
    assert_eq!(json["overall_outcome"], "Improved");
    assert_eq!(json["terminal_outcome"], "Improved");
    assert_eq!(json["regression_count"], 1);
    assert_eq!(json["improvement_count"], 1);
    assert_eq!(json["monotonic"], false);
}

#[test]
fn tighten_report_diff_accepts_selective_crown_model_name_variants() {
    let base_dir = temp_path("nn-verify-tighten-selective-crown");
    fs::create_dir_all(&base_dir).expect("create temp dir");

    let baseline_path = base_dir.join("baseline.json");
    let candidate_path = base_dir.join("candidate.json");
    write_report(
        &baseline_path,
        &report("selective_crown", 10.0, vec![1], 0.25, Some(0.90)),
    );
    write_report(
        &candidate_path,
        &report("selective_crown_tightened", 8.0, vec![1], 0.50, Some(0.95)),
    );

    let output = Command::new(env!("CARGO_BIN_EXE_tighten"))
        .args([
            "kokoro",
            "report-diff",
            "--baseline",
            baseline_path.to_str().expect("baseline path"),
            "--candidate",
            candidate_path.to_str().expect("candidate path"),
            "--json",
        ])
        .output()
        .expect("run tighten binary");

    assert!(output.status.success(), "{output:?}");

    let json: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("parse tighten diff JSON");
    assert_eq!(json["previous_model_name"], "selective_crown");
    assert_eq!(json["current_model_name"], "selective_crown_tightened");
    assert_eq!(json["outcome"], "Improved");
}

#[test]
fn tighten_report_diff_rejects_structural_mismatch_even_with_model_alias() {
    let base_dir = temp_path("nn-verify-tighten-alias-mismatch");
    fs::create_dir_all(&base_dir).expect("create temp dir");

    let baseline_path = base_dir.join("baseline.json");
    let candidate_path = base_dir.join("candidate.json");
    write_report(
        &baseline_path,
        &report("kokoro", 10.0, vec![1], 0.25, Some(0.90)),
    );

    let mut candidate = report("kokoro_selective_crown", 8.0, vec![1], 0.50, Some(0.95));
    candidate.layers[0].layer_type = "Conv".to_string();
    write_report(&candidate_path, &candidate);

    let output = Command::new(env!("CARGO_BIN_EXE_tighten"))
        .args([
            "kokoro",
            "report-diff",
            "--baseline",
            baseline_path.to_str().expect("baseline path"),
            "--candidate",
            candidate_path.to_str().expect("candidate path"),
        ])
        .output()
        .expect("run tighten binary");

    assert!(!output.status.success(), "{output:?}");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("layers[0].layer_type"), "{stderr}");
    assert!(stderr.contains("Linear"), "{stderr}");
    assert!(stderr.contains("Conv"), "{stderr}");
    assert!(!stderr.contains("model_name differs"), "{stderr}");
}

#[test]
fn tighten_report_diff_rejects_incompatible_sequence_reports() {
    let base_dir = temp_path("nn-verify-tighten-incompatible-sequence");
    fs::create_dir_all(&base_dir).expect("create temp dir");

    let baseline_path = base_dir.join("baseline.json");
    let compatible_path = base_dir.join("pass1-compatible.json");
    let incompatible_path = base_dir.join("pass2-incompatible.json");

    write_report(
        &baseline_path,
        &report("kokoro", 10.0, vec![1, 2], 0.25, Some(0.90)),
    );
    write_report(
        &compatible_path,
        &report("kokoro", 8.0, vec![1], 0.50, Some(0.95)),
    );

    let mut incompatible = report("kokoro", 7.0, vec![1], 0.55, Some(0.96));
    incompatible.layers[0].layer_type = "Conv".to_string();
    write_report(&incompatible_path, &incompatible);

    let output = Command::new(env!("CARGO_BIN_EXE_tighten"))
        .args([
            "kokoro",
            "report-diff",
            "--baseline",
            baseline_path.to_str().expect("baseline path"),
            "--candidate",
            compatible_path.to_str().expect("compatible path"),
            "--candidate",
            incompatible_path.to_str().expect("incompatible path"),
        ])
        .output()
        .expect("run tighten binary");

    assert!(!output.status.success(), "{output:?}");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Failed to compare reports"), "{stderr}");
    assert!(stderr.contains("report-diff pass 2"), "{stderr}");
    assert!(stderr.contains("layers[0].layer_type"), "{stderr}");
    assert!(stderr.contains("Linear"), "{stderr}");
    assert!(stderr.contains("Conv"), "{stderr}");
}

#[test]
fn tighten_kokoro_run_stays_unavailable() {
    let output = Command::new(env!("CARGO_BIN_EXE_tighten"))
        .args(["kokoro", "run"])
        .output()
        .expect("run tighten binary");

    assert!(!output.status.success(), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("not wired"), "{stderr}");
    assert!(stderr.contains("report-diff"), "{stderr}");
}

#[test]
fn nn_verify_cert_rejects_tighten_subcommand() {
    let output = Command::new(env!("CARGO_BIN_EXE_nn_verify_cert"))
        .arg("tighten")
        .output()
        .expect("run nn_verify_cert binary");

    assert!(!output.status.success(), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Unknown command: tighten"), "{stderr}");
    assert!(stderr.contains("standalone `tighten` binary"), "{stderr}");
}
