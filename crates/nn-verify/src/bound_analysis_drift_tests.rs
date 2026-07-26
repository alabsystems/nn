// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for F32/F64 precision drift tracking in BoundAnalysisReport.
//!
//! Extracted from `bound_analysis_tests.rs` to keep it under 1000 lines.
//!
//! Part of #2705, Part of #2218.

use super::estimate_norm_chain_precision_drift;
use super::{
    analyze_layer_bounds, default_config, make_norm_chain, make_record, report_to_json,
    BoundAnalysisReport, PropMethod, TighteningRecommendation,
};

#[test]
fn test_chained_norm_depth_computed() {
    // 10 consecutive InstanceNorm layers.
    let records = make_norm_chain(10, 1.01);
    let report = analyze_layer_bounds("depth_test", &records, &default_config());
    assert_eq!(report.chained_norm_depth, 10);
}

#[test]
fn test_chained_norm_depth_with_gap() {
    // 4 norms, Linear, 7 norms — longest chain is 7.
    let mut records = make_norm_chain(4, 1.01);
    records.push(make_record(
        4,
        "Linear",
        vec![(-1.0, 1.0)],
        vec![(-1.0, 1.0)],
        PropMethod::Ibp,
        None,
    ));
    for i in 5..12 {
        records.push(make_record(
            i,
            "InstanceNorm",
            vec![(-1.0, 1.0)],
            vec![(-1.01, 1.01)],
            PropMethod::Ibp,
            None,
        ));
    }
    let report = analyze_layer_bounds("gap_test", &records, &default_config());
    assert_eq!(report.chained_norm_depth, 7);
}

#[test]
fn test_chained_norm_depth_no_norms() {
    let records = vec![make_record(
        0,
        "Linear",
        vec![(-1.0, 1.0)],
        vec![(-2.0, 2.0)],
        PropMethod::Ibp,
        None,
    )];
    let report = analyze_layer_bounds("no_norms", &records, &default_config());
    assert_eq!(report.chained_norm_depth, 0);
}

#[test]
fn test_precision_drift_not_set_by_default() {
    let records = make_norm_chain(10, 1.01);
    let report = analyze_layer_bounds("defaults", &records, &default_config());
    assert!(report.precision_drift_ratio.is_none());
    assert!(report.drift_per_layer.is_none());
}

#[test]
fn test_set_precision_drift_kokoro_realistic() {
    // Kokoro: 58 InstanceNorm layers, F32/F64 ratio of 0.84 (17% attenuation).
    let records = make_norm_chain(58, 1.05);
    let config = default_config();
    let mut report = analyze_layer_bounds("kokoro_drift", &records, &config);
    assert_eq!(report.chained_norm_depth, 58);

    report.set_precision_drift(0.84, &config);

    assert_eq!(report.precision_drift_ratio, Some(0.84));
    assert!(report.drift_per_layer.is_some());

    // drift_per_layer = 1.0 - 0.84^(1/58) ≈ 0.003
    let dpl = report.drift_per_layer.unwrap();
    assert!(dpl > 0.002 && dpl < 0.005, "drift_per_layer={dpl}");

    // Should trigger PrecisionRisk: depth=58 > 20, ratio=0.84 < 0.95.
    let prisk_recs: Vec<_> = report
        .recommendations
        .iter()
        .filter(|r| matches!(r, TighteningRecommendation::PrecisionRisk { .. }))
        .collect();
    assert_eq!(prisk_recs.len(), 1);

    match &prisk_recs[0] {
        TighteningRecommendation::PrecisionRisk {
            chained_norm_depth,
            precision_drift_ratio,
            drift_per_layer,
        } => {
            assert_eq!(*chained_norm_depth, 58);
            assert!((*precision_drift_ratio - 0.84).abs() < 1e-6);
            assert!(*drift_per_layer > 0.002 && *drift_per_layer < 0.005);
        }
        _ => unreachable!(),
    }
}

#[test]
fn test_set_precision_drift_healthy_no_flag() {
    // Healthy model: ratio 0.99 (1% attenuation) — should NOT flag PrecisionRisk.
    let records = make_norm_chain(58, 1.001);
    let config = default_config();
    let mut report = analyze_layer_bounds("healthy_drift", &records, &config);

    report.set_precision_drift(0.99, &config);

    assert_eq!(report.precision_drift_ratio, Some(0.99));
    assert!(report.drift_per_layer.is_some());

    let prisk_recs: Vec<_> = report
        .recommendations
        .iter()
        .filter(|r| matches!(r, TighteningRecommendation::PrecisionRisk { .. }))
        .collect();
    assert!(
        prisk_recs.is_empty(),
        "Healthy model (ratio=0.99) should NOT trigger PrecisionRisk"
    );
}

#[test]
fn test_set_precision_drift_short_chain_no_flag() {
    // Short chain (10 layers < 20 threshold), even with poor ratio.
    let records = make_norm_chain(10, 1.05);
    let config = default_config();
    let mut report = analyze_layer_bounds("short_drift", &records, &config);

    report.set_precision_drift(0.80, &config);

    // Depth=10 < 20, so no PrecisionRisk despite ratio < 0.95.
    let prisk_recs: Vec<_> = report
        .recommendations
        .iter()
        .filter(|r| matches!(r, TighteningRecommendation::PrecisionRisk { .. }))
        .collect();
    assert!(prisk_recs.is_empty());
}

#[test]
fn test_set_precision_drift_nan_ratio_ignored() {
    let records = make_norm_chain(30, 1.01);
    let config = default_config();
    let mut report = analyze_layer_bounds("nan_drift", &records, &config);

    report.set_precision_drift(f32::NAN, &config);

    assert!(report.precision_drift_ratio.is_none());
    assert!(report.drift_per_layer.is_none());
}

#[test]
fn test_precision_risk_json_round_trip() {
    let rec = TighteningRecommendation::PrecisionRisk {
        chained_norm_depth: 58,
        precision_drift_ratio: 0.84,
        drift_per_layer: 0.003,
    };
    let json = serde_json::to_string(&rec).expect("serialize");
    let parsed: TighteningRecommendation = serde_json::from_str(&json).expect("deserialize");

    match parsed {
        TighteningRecommendation::PrecisionRisk {
            chained_norm_depth,
            precision_drift_ratio,
            drift_per_layer,
        } => {
            assert_eq!(chained_norm_depth, 58);
            assert!((precision_drift_ratio - 0.84).abs() < 1e-6);
            assert!((drift_per_layer - 0.003).abs() < 1e-6);
        }
        _ => panic!("Wrong variant after round-trip"),
    }
}

#[test]
fn test_report_with_precision_drift_json_round_trip() {
    let records = make_norm_chain(30, 1.02);
    let config = default_config();
    let mut report = analyze_layer_bounds("json_drift", &records, &config);
    report.set_precision_drift(0.90, &config);

    let json = report_to_json(&report).expect("serialize");
    let parsed: BoundAnalysisReport = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(parsed.chained_norm_depth, 30);
    assert_eq!(parsed.precision_drift_ratio, Some(0.90));
    assert!(parsed.drift_per_layer.is_some());

    let prisk_recs: Vec<_> = parsed
        .recommendations
        .iter()
        .filter(|r| matches!(r, TighteningRecommendation::PrecisionRisk { .. }))
        .collect();
    assert_eq!(prisk_recs.len(), 1);
}

// --- estimate_norm_chain_precision_drift tests (#2705) ---

#[test]
fn test_estimate_drift_zero_depth_returns_one() {
    let ratio = estimate_norm_chain_precision_drift(0);
    assert!((ratio - 1.0).abs() < 1e-6, "depth=0 ratio={ratio}");
}

#[test]
fn test_estimate_drift_single_layer_near_one() {
    let ratio = estimate_norm_chain_precision_drift(1);
    // Single layer: drift is negligible, ratio ≈ 1.0 (may round to exactly 1.0).
    assert!(ratio > 0.99, "depth=1 ratio={ratio}");
}

#[test]
fn test_estimate_drift_58_layers_below_one() {
    // Kokoro's 58 chained InstanceNorm layers with per-channel affine:
    // Naive F32 should measurably drift from the F64 reference.
    let ratio = estimate_norm_chain_precision_drift(58);
    assert!(
        ratio < 1.0,
        "58 layers should show measurable drift: ratio={ratio}"
    );
    assert!(
        ratio > 0.5,
        "drift should not be catastrophic: ratio={ratio}"
    );
}

#[test]
fn test_estimate_drift_deeper_has_more_drift() {
    // At sufficient depth, drift should be non-decreasing.
    let r10 = estimate_norm_chain_precision_drift(10);
    let r58 = estimate_norm_chain_precision_drift(58);
    assert!(
        r10 >= r58,
        "deeper chain should have >= drift: r10={r10}, r58={r58}"
    );
}

#[test]
fn test_estimate_and_set_precision_drift_auto() {
    // Full pipeline: build report with 58-layer norm chain, auto-estimate drift.
    let records = make_norm_chain(58, 1.05);
    let config = default_config();
    let mut report = analyze_layer_bounds("kokoro_auto", &records, &config);
    assert_eq!(report.chained_norm_depth, 58);
    assert!(report.precision_drift_ratio.is_none());

    report.estimate_and_set_precision_drift(&config);

    assert!(
        report.precision_drift_ratio.is_some(),
        "precision_drift_ratio should be populated"
    );
    let ratio = report.precision_drift_ratio.unwrap();
    // Synthetic naive-F32 estimator gives ratio very close to 1.0 (~0.99999994).
    // The precision_drift_ratio is populated; it won't trigger PRECISION_RISK
    // since the synthetic estimate is above 0.95. Real model data via
    // set_precision_drift() can trigger the flag.
    assert!(
        ratio > 0.5,
        "ratio should not be catastrophic: ratio={ratio}"
    );
    assert!(ratio <= 1.0, "ratio should be <= 1.0: ratio={ratio}");
    assert!(report.drift_per_layer.is_some());
}

#[test]
fn test_estimate_and_set_no_norms_is_noop() {
    let records = vec![make_record(
        0,
        "Linear",
        vec![(-1.0, 1.0)],
        vec![(-2.0, 2.0)],
        PropMethod::Ibp,
        None,
    )];
    let config = default_config();
    let mut report = analyze_layer_bounds("no_norms_auto", &records, &config);

    report.estimate_and_set_precision_drift(&config);

    assert!(report.precision_drift_ratio.is_none());
    assert!(report.drift_per_layer.is_none());
}
