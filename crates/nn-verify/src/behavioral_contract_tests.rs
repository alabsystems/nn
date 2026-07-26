// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for behavioral contract validation.
//!
//! Validates the contract ratchet mechanism: validate() detects regressions,
//! tighten() ratchets bounds forward. Also covers edge cases around
//! NaN/Inf thresholds and the f64→usize cast in check_explosion_points.
//!
//! Part of #2218, #3327.

use super::*;
use crate::bound_analysis::BoundAnalysisReport;

/// Helper: create a minimal BoundAnalysisReport for contract testing.
fn sample_report() -> BoundAnalysisReport {
    BoundAnalysisReport {
        model_name: "test_model".to_string(),
        total_layers: 5,
        layers: Vec::new(),
        explosion_points: vec![2],
        output_width: 3.5,
        output_is_finite: true,
        crown_coverage: 0.8,
        recommendations: Vec::new(),
        analyzed_at: "2026-01-01T00:00:00Z".to_string(),
        chained_norm_depth: 0,
        precision_drift_ratio: Some(0.95),
        drift_per_layer: None,
    }
}

// ---------------------------------------------------------------------------
// ContractValidation constructors
// ---------------------------------------------------------------------------

#[test]
fn test_contract_validation_passing_is_valid() {
    let cv = ContractValidation::passing();
    assert!(cv.violations.is_empty());
    assert!(cv.tightened.is_empty());
    assert!(cv.all_satisfied);
}

// ---------------------------------------------------------------------------
// from_bound_analysis
// ---------------------------------------------------------------------------

#[test]
fn test_from_bound_analysis_captures_properties() {
    let report = sample_report();
    let contract = BehavioralContract::from_bound_analysis(&report, 0.01, 0);

    assert_eq!(contract.model_name, "test_model");
    assert_eq!(contract.version, BehavioralContract::CURRENT_VERSION);
    assert_eq!(contract.source_iteration, 0);

    // Should have 5 properties: output_is_finite, output_width, crown_coverage,
    // explosion_points, precision_drift_ratio.
    assert_eq!(contract.properties.len(), 5);

    let names: Vec<&str> = contract
        .properties
        .iter()
        .map(|p| p.name.as_str())
        .collect();
    assert!(names.contains(&"output_is_finite"));
    assert!(names.contains(&"output_width"));
    assert!(names.contains(&"crown_coverage"));
    assert!(names.contains(&"explosion_points"));
    assert!(names.contains(&"precision_drift_ratio"));
}

#[test]
fn test_from_bound_analysis_skips_nan_output_width() {
    let mut report = sample_report();
    report.output_width = f32::NAN;

    let contract = BehavioralContract::from_bound_analysis(&report, 0.01, 0);
    let has_width = contract.properties.iter().any(|p| p.name == "output_width");
    assert!(
        !has_width,
        "NaN output_width should not be recorded as a property"
    );
}

#[test]
fn test_from_bound_analysis_skips_nan_precision_drift() {
    let mut report = sample_report();
    report.precision_drift_ratio = Some(f32::NAN);

    let contract = BehavioralContract::from_bound_analysis(&report, 0.01, 0);
    let has_drift = contract
        .properties
        .iter()
        .any(|p| p.name == "precision_drift_ratio");
    assert!(
        !has_drift,
        "NaN precision_drift_ratio should not be recorded"
    );
}

// ---------------------------------------------------------------------------
// validate() — regression detection
// ---------------------------------------------------------------------------

#[test]
fn test_validate_no_violations_on_same_report() {
    let report = sample_report();
    let contract = BehavioralContract::from_bound_analysis(&report, 0.01, 0);

    let validation = contract.validate(&report);
    assert!(
        validation.violations.is_empty(),
        "same report should produce no violations: {:?}",
        validation.violations
    );
    assert!(validation.all_satisfied);
}

#[test]
fn test_validate_detects_finiteness_regression() {
    let report = sample_report();
    let contract = BehavioralContract::from_bound_analysis(&report, 0.01, 0);

    let mut regressed = sample_report();
    regressed.output_is_finite = false;

    let validation = contract.validate(&regressed);
    assert!(
        !validation.violations.is_empty(),
        "non-finite output should be a violation"
    );
    assert!(
        validation
            .violations
            .iter()
            .any(|v| v.contains("output_is_finite")),
        "violation should mention output_is_finite: {:?}",
        validation.violations
    );
}

#[test]
fn test_validate_detects_output_width_regression() {
    let report = sample_report();
    let contract = BehavioralContract::from_bound_analysis(&report, 0.01, 0);

    // >10% wider = regression.
    let mut regressed = sample_report();
    regressed.output_width = 5.0; // 3.5 * 1.1 = 3.85, so 5.0 is a regression

    let validation = contract.validate(&regressed);
    assert!(
        validation
            .violations
            .iter()
            .any(|v| v.contains("output_width")),
        ">10% wider output should be a violation: {:?}",
        validation.violations
    );
}

#[test]
fn test_validate_detects_output_width_improvement() {
    let report = sample_report();
    let contract = BehavioralContract::from_bound_analysis(&report, 0.01, 0);

    // <99% of contract = tightened.
    let mut improved = sample_report();
    improved.output_width = 2.0; // 2.0 < 3.5 * 0.99

    let validation = contract.validate(&improved);
    assert!(
        validation
            .tightened
            .iter()
            .any(|t| t.contains("output_width")),
        "tighter output should be reported as tightened: {:?}",
        validation.tightened
    );
}

#[test]
fn test_validate_detects_explosion_point_regression() {
    let report = sample_report();
    let contract = BehavioralContract::from_bound_analysis(&report, 0.01, 0);

    // More explosion points = regression.
    let mut regressed = sample_report();
    regressed.explosion_points = vec![0, 1, 2];

    let validation = contract.validate(&regressed);
    assert!(
        validation
            .violations
            .iter()
            .any(|v| v.contains("explosion_points")),
        "more explosion points should be a violation: {:?}",
        validation.violations
    );
}

#[test]
fn test_validate_detects_explosion_point_improvement() {
    let report = sample_report();
    let contract = BehavioralContract::from_bound_analysis(&report, 0.01, 0);

    // Fewer explosion points = tightened.
    let mut improved = sample_report();
    improved.explosion_points = Vec::new();

    let validation = contract.validate(&improved);
    assert!(
        validation
            .tightened
            .iter()
            .any(|t| t.contains("explosion_points")),
        "fewer explosion points should be tightened: {:?}",
        validation.tightened
    );
}

#[test]
fn test_validate_detects_precision_drift_regression() {
    let report = sample_report();
    let contract = BehavioralContract::from_bound_analysis(&report, 0.01, 0);

    // Lower ratio = worse drift = regression.
    let mut regressed = sample_report();
    regressed.precision_drift_ratio = Some(0.90); // below 0.95 * 0.99

    let validation = contract.validate(&regressed);
    assert!(
        validation
            .violations
            .iter()
            .any(|v| v.contains("precision_drift_ratio")),
        "lower drift ratio should be a violation: {:?}",
        validation.violations
    );
}

// ---------------------------------------------------------------------------
// tighten() — ratchet mechanism
// ---------------------------------------------------------------------------

#[test]
fn test_tighten_ratchets_output_width() {
    let report = sample_report();
    let contract = BehavioralContract::from_bound_analysis(&report, 0.01, 0);

    let mut improved = sample_report();
    improved.output_width = 2.0;

    let tightened = contract.tighten(&improved, 1);
    let width_prop = tightened
        .properties
        .iter()
        .find(|p| p.name == "output_width")
        .unwrap();
    assert!(
        (width_prop.threshold - 2.0_f64).abs() < 1e-6,
        "tightened contract should have new (tighter) output_width threshold"
    );
    assert_eq!(tightened.source_iteration, 1);
}

#[test]
fn test_tighten_does_not_weaken_output_width() {
    let report = sample_report();
    let contract = BehavioralContract::from_bound_analysis(&report, 0.01, 0);

    let mut wider = sample_report();
    wider.output_width = 10.0;

    let tightened = contract.tighten(&wider, 1);
    let width_prop = tightened
        .properties
        .iter()
        .find(|p| p.name == "output_width")
        .unwrap();
    // Original was 3.5, 10.0 is wider — threshold should NOT increase.
    assert!(
        (width_prop.threshold - f64::from(3.5_f32)).abs() < 1e-6,
        "tighten must not weaken: threshold={}, expected=3.5",
        width_prop.threshold
    );
}

#[test]
fn test_tighten_ratchets_explosion_points_down() {
    let report = sample_report();
    let contract = BehavioralContract::from_bound_analysis(&report, 0.01, 0);

    let mut improved = sample_report();
    improved.explosion_points = Vec::new(); // 0 < 1

    let tightened = contract.tighten(&improved, 1);
    let ep_prop = tightened
        .properties
        .iter()
        .find(|p| p.name == "explosion_points")
        .unwrap();
    assert!(
        ep_prop.threshold < 0.5,
        "explosion_points threshold should ratchet to 0"
    );
}

#[test]
fn test_tighten_ratchets_crown_coverage_up() {
    let report = sample_report();
    let contract = BehavioralContract::from_bound_analysis(&report, 0.01, 0);

    let mut improved = sample_report();
    improved.crown_coverage = 0.95; // > 0.8

    let tightened = contract.tighten(&improved, 1);
    let cov_prop = tightened
        .properties
        .iter()
        .find(|p| p.name == "crown_coverage")
        .unwrap();
    assert!(
        (cov_prop.threshold - f64::from(0.95_f32)).abs() < 1e-6,
        "crown_coverage should ratchet up"
    );
}

// ---------------------------------------------------------------------------
// Edge cases: NaN/Inf in contract thresholds (deserialized contracts)
// ---------------------------------------------------------------------------

#[test]
fn test_validate_nan_output_width_in_report_no_crash() {
    let report = sample_report();
    let contract = BehavioralContract::from_bound_analysis(&report, 0.01, 0);

    let mut nan_report = sample_report();
    nan_report.output_width = f32::NAN;

    // Should not crash — NaN.is_finite() returns false, skipping the check.
    let validation = contract.validate(&nan_report);
    // NaN output_width should not trigger a violation (guarded by is_finite).
    assert!(
        !validation
            .violations
            .iter()
            .any(|v| v.contains("output_width")),
        "NaN output_width should be silently skipped, not flagged"
    );
}

#[test]
fn test_validate_inf_output_width_in_report_no_crash() {
    let report = sample_report();
    let contract = BehavioralContract::from_bound_analysis(&report, 0.01, 0);

    let mut inf_report = sample_report();
    inf_report.output_width = f32::INFINITY;

    let validation = contract.validate(&inf_report);
    // Inf.is_finite() is false, so the check is skipped.
    assert!(
        !validation
            .violations
            .iter()
            .any(|v| v.contains("output_width")),
        "Inf output_width should be silently skipped"
    );
}

#[test]
fn test_tighten_nan_output_width_no_ratchet() {
    let report = sample_report();
    let contract = BehavioralContract::from_bound_analysis(&report, 0.01, 0);

    let mut nan_report = sample_report();
    nan_report.output_width = f32::NAN;

    let tightened = contract.tighten(&nan_report, 1);
    let width_prop = tightened
        .properties
        .iter()
        .find(|p| p.name == "output_width")
        .unwrap();
    // NaN should not ratchet (is_finite guard).
    assert!(
        (width_prop.threshold - f64::from(3.5_f32)).abs() < 1e-6,
        "NaN should not ratchet output_width"
    );
}

// ---------------------------------------------------------------------------
// Edge cases: f64→usize cast guard in check_explosion_points (#3331)
// ---------------------------------------------------------------------------

#[test]
fn test_validate_nan_explosion_threshold_is_violation() {
    let report = sample_report();
    let mut contract = BehavioralContract::from_bound_analysis(&report, 0.01, 0);

    // Corrupt the threshold to NaN (simulates deserialized contract with bad data).
    if let Some(prop) = contract
        .properties
        .iter_mut()
        .find(|p| p.name == "explosion_points")
    {
        prop.threshold = f64::NAN;
    }

    let validation = contract.validate(&report);
    assert!(
        validation.violations.iter().any(|v| v.contains("invalid")),
        "NaN threshold should be flagged as invalid: {:?}",
        validation.violations
    );
}

#[test]
fn test_validate_negative_explosion_threshold_is_violation() {
    let report = sample_report();
    let mut contract = BehavioralContract::from_bound_analysis(&report, 0.01, 0);

    if let Some(prop) = contract
        .properties
        .iter_mut()
        .find(|p| p.name == "explosion_points")
    {
        prop.threshold = -1.0;
    }

    let validation = contract.validate(&report);
    assert!(
        validation.violations.iter().any(|v| v.contains("invalid")),
        "negative threshold should be flagged as invalid: {:?}",
        validation.violations
    );
}

// ---------------------------------------------------------------------------
// JSON round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_contract_json_round_trip() {
    let report = sample_report();
    let contract = BehavioralContract::from_bound_analysis(&report, 0.01, 0);

    let json = contract.to_json().unwrap();
    let deserialized: BehavioralContract = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.model_name, contract.model_name);
    assert_eq!(deserialized.version, contract.version);
    assert_eq!(deserialized.properties.len(), contract.properties.len());
}

#[test]
fn test_validation_json_round_trip() {
    let report = sample_report();
    let contract = BehavioralContract::from_bound_analysis(&report, 0.01, 0);

    let validation = contract.validate(&report);
    let json = serde_json::to_string(&validation).unwrap();
    let deserialized: ContractValidation = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.all_satisfied, validation.all_satisfied);
    assert_eq!(deserialized.violations.len(), validation.violations.len());
}
