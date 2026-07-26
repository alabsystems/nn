// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the bound analysis module.

use super::*;
use crate::certificate_types::LayerBoundRecord;
use crate::verify_types::PropMethod;

#[path = "bound_analysis_drift_tests.rs"]
mod drift_tests;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub(super) fn make_record(
    index: usize,
    layer_type: &str,
    input: Vec<(f32, f32)>,
    output: Vec<(f32, f32)>,
    method: PropMethod,
    node_name: Option<&str>,
) -> LayerBoundRecord {
    LayerBoundRecord {
        layer_index: index,
        layer_type: layer_type.to_string(),
        input_bounds: input,
        output_bounds: output,
        method,
        node_name: node_name.map(ToString::to_string),
        input_sources: None,
    }
}

pub(super) fn default_config() -> AnalysisConfig {
    AnalysisConfig::default()
}

// ---------------------------------------------------------------------------
// Width computation tests
// ---------------------------------------------------------------------------

#[test]
fn test_width_computation_known_bounds() {
    let bounds = vec![(-1.0, 1.0), (-2.0, 3.0), (0.0, 0.5)];
    assert_eq!(max_width(&bounds), 5.0); // 3 - (-2) = 5
    let avg = avg_width(&bounds);
    // widths: 2.0, 5.0, 0.5 => avg = 7.5/3 = 2.5
    assert!((avg - 2.5).abs() < 1e-6);
}

#[test]
fn test_width_computation_empty() {
    assert_eq!(max_width(&[]), 0.0);
    assert_eq!(avg_width(&[]), 0.0);
}

#[test]
fn test_width_computation_single_element() {
    let bounds = vec![(1.0, 3.0)];
    assert_eq!(max_width(&bounds), 2.0);
    assert_eq!(avg_width(&bounds), 2.0);
}

// ---------------------------------------------------------------------------
// NaN/Inf handling
// ---------------------------------------------------------------------------

#[test]
fn test_nan_bounds_detected() {
    let bounds = vec![(f32::NAN, 1.0), (0.0, 1.0)];
    assert!(has_non_finite(&bounds));
    assert_eq!(max_width(&bounds), f32::INFINITY);
    assert_eq!(avg_width(&bounds), f32::INFINITY);
}

#[test]
fn test_inf_bounds_detected() {
    let bounds = vec![(f32::NEG_INFINITY, f32::INFINITY)];
    assert!(has_non_finite(&bounds));
    assert_eq!(max_width(&bounds), f32::INFINITY);
}

#[test]
fn test_overflow_width_detected() {
    // f32::MAX - (-f32::MAX) overflows to Inf.
    let bounds = vec![(-f32::MAX, f32::MAX)];
    assert_eq!(max_width(&bounds), f32::INFINITY);
}

#[test]
fn test_nan_layer_generates_recommendation() {
    let records = vec![make_record(
        0,
        "Linear",
        vec![(0.0, 1.0)],
        vec![(f32::NAN, 1.0)],
        PropMethod::Ibp,
        Some("n0"),
    )];
    let report = analyze_layer_bounds("test", &records, &default_config());

    assert!(report.layers[0].has_non_finite_bounds);
    assert!(!report.output_is_finite);
    assert!(report.layers[0].is_explosion_point);

    // Should generate a TightenLayer recommendation.
    assert!(!report.recommendations.is_empty());
    match &report.recommendations[0] {
        TighteningRecommendation::TightenLayer { suggestion, .. } => {
            assert!(suggestion.contains("Non-finite"));
        }
        other => panic!("Expected TightenLayer, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Expansion ratio
// ---------------------------------------------------------------------------

#[test]
fn test_expansion_ratio_normal() {
    let records = vec![
        make_record(
            0,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-5.0, 5.0)],
            PropMethod::Ibp,
            None,
        ),
        make_record(
            1,
            "ReLU",
            vec![(-5.0, 5.0)],
            vec![(0.0, 5.0)],
            PropMethod::Crown,
            None,
        ),
    ];
    let report = analyze_layer_bounds("test", &records, &default_config());

    // Layer 0: input width=2, output width=10, ratio=5
    assert!((report.layers[0].expansion_ratio - 5.0).abs() < 1e-6);
    assert!(!report.layers[0].is_explosion_point);

    // Layer 1: input width=10, output width=5, ratio=0.5
    assert!((report.layers[1].expansion_ratio - 0.5).abs() < 1e-6);
}

#[test]
fn test_expansion_ratio_zero_input() {
    let records = vec![make_record(
        0,
        "Embedding",
        vec![(1.0, 1.0)], // zero width input
        vec![(-0.5, 0.5)],
        PropMethod::Crown,
        None,
    )];
    let report = analyze_layer_bounds("test", &records, &default_config());

    // Zero input width with nonzero output → INFINITY ratio.
    assert_eq!(report.layers[0].expansion_ratio, f32::INFINITY);
    assert!(report.layers[0].is_explosion_point);
}

#[test]
fn test_expansion_ratio_both_zero() {
    let records = vec![make_record(
        0,
        "Identity",
        vec![(0.0, 0.0)],
        vec![(0.0, 0.0)],
        PropMethod::Ibp,
        None,
    )];
    let report = analyze_layer_bounds("test", &records, &default_config());

    // Both zero → ratio is 1.0 (identity).
    assert!((report.layers[0].expansion_ratio - 1.0).abs() < 1e-6);
    assert!(!report.layers[0].is_explosion_point);
}

// ---------------------------------------------------------------------------
// Explosion point detection
// ---------------------------------------------------------------------------

#[test]
fn test_explosion_point_detection() {
    let config = AnalysisConfig {
        explosion_threshold: 10.0,
        ..default_config()
    };
    let records = vec![
        make_record(
            0,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-5.0, 5.0)],
            PropMethod::Ibp,
            None,
        ),
        make_record(
            1,
            "Softmax",
            vec![(-5.0, 5.0)],
            vec![(-500.0, 500.0)],
            PropMethod::Ibp,
            None,
        ),
        make_record(
            2,
            "Linear",
            vec![(-500.0, 500.0)],
            vec![(-600.0, 600.0)],
            PropMethod::Crown,
            None,
        ),
    ];
    let report = analyze_layer_bounds("test", &records, &config);

    assert_eq!(report.explosion_points, vec![1]);
    assert!(report.layers[1].is_explosion_point);
    assert!(!report.layers[0].is_explosion_point);
    assert!(!report.layers[2].is_explosion_point);
}

// ---------------------------------------------------------------------------
// Recommendation generation
// ---------------------------------------------------------------------------

#[test]
fn test_norm_layer_explosion_generates_switch_norm() {
    let records = vec![make_record(
        0,
        "LayerNorm",
        vec![(-1.0, 1.0)],
        vec![(-100.0, 100.0)],
        PropMethod::Ibp,
        Some("norm_0"),
    )];
    let report = analyze_layer_bounds("test", &records, &default_config());

    let switch_recs: Vec<_> = report
        .recommendations
        .iter()
        .filter(|r| matches!(r, TighteningRecommendation::SwitchNormMode { .. }))
        .collect();
    assert_eq!(switch_recs.len(), 1);

    match &switch_recs[0] {
        TighteningRecommendation::SwitchNormMode {
            layer_index,
            node_name,
            suggested_mode,
            target,
            ..
        } => {
            assert_eq!(*layer_index, 0);
            assert_eq!(node_name.as_deref(), Some("norm_0"));
            assert_eq!(suggested_mode, "ForwardMode");
            assert_eq!(*target, TighteningTarget::Framework);
        }
        _ => unreachable!(),
    }
}

#[test]
fn test_ibp_wide_bounds_generates_escalate_to_crown() {
    let config = AnalysisConfig {
        crown_escalation_width: 100.0,
        ..default_config()
    };
    let records = vec![make_record(
        0,
        "Linear",
        vec![(-1.0, 1.0)],
        vec![(-200.0, 200.0)],
        PropMethod::Ibp,
        Some("linear_0"),
    )];
    let report = analyze_layer_bounds("test", &records, &config);

    let esc_recs: Vec<_> = report
        .recommendations
        .iter()
        .filter(|r| matches!(r, TighteningRecommendation::EscalateToCrown { .. }))
        .collect();
    assert_eq!(esc_recs.len(), 1);

    match &esc_recs[0] {
        TighteningRecommendation::EscalateToCrown {
            layer_index,
            ibp_width,
            ..
        } => {
            assert_eq!(*layer_index, 0);
            assert!(*ibp_width > 100.0);
        }
        _ => unreachable!(),
    }
}

#[test]
fn test_crown_layer_no_escalation() {
    let config = AnalysisConfig {
        crown_escalation_width: 100.0,
        ..default_config()
    };
    let records = vec![make_record(
        0,
        "Linear",
        vec![(-1.0, 1.0)],
        vec![(-200.0, 200.0)],
        PropMethod::Crown, // Already CROWN
        None,
    )];
    let report = analyze_layer_bounds("test", &records, &config);

    let esc_recs: Vec<_> = report
        .recommendations
        .iter()
        .filter(|r| matches!(r, TighteningRecommendation::EscalateToCrown { .. }))
        .collect();
    assert!(esc_recs.is_empty());
}

#[test]
fn test_exp_family_generates_annotate_constraint() {
    let config = AnalysisConfig {
        explosion_threshold: 5.0,
        ..default_config()
    };
    let records = vec![
        make_record(
            0,
            "Exp",
            vec![(-1.0, 1.0)],
            vec![(-100.0, 100.0)],
            PropMethod::Ibp,
            None,
        ),
        make_record(
            1,
            "Linear",
            vec![(-100.0, 100.0)],
            vec![(-5000.0, 5000.0)],
            PropMethod::Ibp,
            Some("post_exp"),
        ),
    ];
    let report = analyze_layer_bounds("test", &records, &config);

    let constraint_recs: Vec<_> = report
        .recommendations
        .iter()
        .filter(|r| matches!(r, TighteningRecommendation::AnnotateConstraint { .. }))
        .collect();
    assert_eq!(constraint_recs.len(), 1);

    match &constraint_recs[0] {
        TighteningRecommendation::AnnotateConstraint {
            layer_index,
            reason,
            ..
        } => {
            assert_eq!(*layer_index, 1);
            assert!(reason.contains("Exp"));
        }
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// SMT extraction
// ---------------------------------------------------------------------------

#[test]
fn test_smt_extraction_small_subgraph() {
    let config = AnalysisConfig {
        smt_max_elements: 10,
        ..default_config()
    };
    let records = vec![
        make_record(
            0,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-2.0, 2.0), (-1.0, 1.0)], // 2 elements
            PropMethod::Ibp,
            None,
        ),
        make_record(
            1,
            "ReLU",
            vec![(-2.0, 2.0), (-1.0, 1.0)],
            vec![(0.0, 2.0), (0.0, 1.0)], // 2 elements
            PropMethod::Ibp,
            None,
        ),
        make_record(
            2,
            "Linear",
            vec![(0.0, 2.0), (0.0, 1.0)],
            vec![(-1.0, 3.0)], // 1 element
            PropMethod::Ibp,
            None,
        ),
    ];
    let report = analyze_layer_bounds("test", &records, &config);

    let smt_recs: Vec<_> = report
        .recommendations
        .iter()
        .filter(|r| matches!(r, TighteningRecommendation::ExtractForSmt { .. }))
        .collect();
    assert_eq!(smt_recs.len(), 1);

    match &smt_recs[0] {
        TighteningRecommendation::ExtractForSmt {
            start_layer,
            end_layer,
            estimated_elements,
        } => {
            assert_eq!(*start_layer, 0);
            assert_eq!(*end_layer, 2);
            assert_eq!(*estimated_elements, 5); // 2 + 2 + 1
        }
        _ => unreachable!(),
    }
}

#[test]
fn test_smt_extraction_over_budget() {
    let config = AnalysisConfig {
        smt_max_elements: 2,
        ..default_config()
    };
    let records = vec![
        make_record(
            0,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-2.0, 2.0); 100], // 100 elements — over budget
            PropMethod::Ibp,
            None,
        ),
        make_record(
            1,
            "ReLU",
            vec![(0.0, 2.0); 100],
            vec![(0.0, 2.0); 100],
            PropMethod::Ibp,
            None,
        ),
    ];
    let report = analyze_layer_bounds("test", &records, &config);

    let smt_recs: Vec<_> = report
        .recommendations
        .iter()
        .filter(|r| matches!(r, TighteningRecommendation::ExtractForSmt { .. }))
        .collect();
    assert!(smt_recs.is_empty());
}

// ---------------------------------------------------------------------------
// Report structure
// ---------------------------------------------------------------------------

#[test]
fn test_crown_coverage() {
    let records = vec![
        make_record(
            0,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-2.0, 2.0)],
            PropMethod::Ibp,
            None,
        ),
        make_record(
            1,
            "ReLU",
            vec![(-2.0, 2.0)],
            vec![(0.0, 2.0)],
            PropMethod::Crown,
            None,
        ),
        make_record(
            2,
            "Linear",
            vec![(0.0, 2.0)],
            vec![(-1.0, 3.0)],
            PropMethod::Crown,
            None,
        ),
    ];
    let report = analyze_layer_bounds("test", &records, &default_config());

    // 2 CROWN out of 3 layers.
    assert!((report.crown_coverage - 2.0 / 3.0).abs() < 1e-6);
}

#[test]
fn test_output_width_from_last_layer() {
    let records = vec![
        make_record(
            0,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-100.0, 100.0)],
            PropMethod::Ibp,
            None,
        ),
        make_record(
            1,
            "Sigmoid",
            vec![(-100.0, 100.0)],
            vec![(0.0, 1.0)],
            PropMethod::Crown,
            None,
        ),
    ];
    let report = analyze_layer_bounds("test", &records, &default_config());

    assert!((report.output_width - 1.0).abs() < 1e-6);
    assert!(report.output_is_finite);
}

#[test]
fn test_empty_records() {
    let report = analyze_layer_bounds("empty", &[], &default_config());
    assert_eq!(report.total_layers, 0);
    assert!(report.layers.is_empty());
    assert!(report.explosion_points.is_empty());
    assert!(report.recommendations.is_empty());
    assert_eq!(report.output_width, 0.0);
    assert!(report.output_is_finite);
    assert_eq!(report.crown_coverage, 0.0);
}

// ---------------------------------------------------------------------------
// node_name handling
// ---------------------------------------------------------------------------

#[test]
fn test_node_name_present() {
    let records = vec![make_record(
        0,
        "Linear",
        vec![(-1.0, 1.0)],
        vec![(-2.0, 2.0)],
        PropMethod::Ibp,
        Some("trace_5"),
    )];
    let report = analyze_layer_bounds("test", &records, &default_config());

    assert_eq!(report.layers[0].node_name.as_deref(), Some("trace_5"));
}

#[test]
fn test_node_name_absent() {
    let records = vec![make_record(
        0,
        "Linear",
        vec![(-1.0, 1.0)],
        vec![(-2.0, 2.0)],
        PropMethod::Ibp,
        None,
    )];
    let report = analyze_layer_bounds("test", &records, &default_config());

    assert!(report.layers[0].node_name.is_none());
}

// ---------------------------------------------------------------------------
// JSON round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_report_json_round_trip() {
    let records = vec![
        make_record(
            0,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-5.0, 5.0)],
            PropMethod::Ibp,
            Some("n0"),
        ),
        make_record(
            1,
            "ReLU",
            vec![(-5.0, 5.0)],
            vec![(0.0, 5.0)],
            PropMethod::Crown,
            Some("n1"),
        ),
    ];
    let report = analyze_layer_bounds("round_trip_test", &records, &default_config());

    let json = report_to_json(&report).expect("serialize");
    let parsed: BoundAnalysisReport = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(parsed.model_name, "round_trip_test");
    assert_eq!(parsed.total_layers, 2);
    assert_eq!(parsed.layers.len(), 2);
    assert_eq!(parsed.layers[0].layer_type, "Linear");
    assert_eq!(parsed.layers[1].layer_type, "ReLU");
    assert_eq!(parsed.layers[0].node_name.as_deref(), Some("n0"));
    assert_eq!(parsed.layers[1].node_name.as_deref(), Some("n1"));
}

#[test]
fn test_recommendation_json_round_trip() {
    let rec = TighteningRecommendation::SwitchNormMode {
        layer_index: 3,
        node_name: Some("norm_3".to_string()),
        layer_type: "LayerNorm".to_string(),
        current_width: 500.0,
        suggested_mode: "ForwardMode".to_string(),
        target: TighteningTarget::Framework,
    };
    let json = serde_json::to_string(&rec).expect("serialize");
    let parsed: TighteningRecommendation = serde_json::from_str(&json).expect("deserialize");

    match parsed {
        TighteningRecommendation::SwitchNormMode {
            layer_index,
            node_name,
            suggested_mode,
            ..
        } => {
            assert_eq!(layer_index, 3);
            assert_eq!(node_name.as_deref(), Some("norm_3"));
            assert_eq!(suggested_mode, "ForwardMode");
        }
        _ => panic!("Wrong variant"),
    }
}

// ---------------------------------------------------------------------------
// Norm layer detection
// ---------------------------------------------------------------------------

#[test]
fn test_norm_layer_variants() {
    assert!(is_norm_layer("LayerNorm"));
    assert!(is_norm_layer("RMSNorm"));
    assert!(is_norm_layer("InstanceNorm"));
    assert!(is_norm_layer("BatchNorm"));
    assert!(is_norm_layer("GroupNorm"));
    assert!(!is_norm_layer("Linear"));
    assert!(!is_norm_layer("ReLU"));
    assert!(!is_norm_layer("Softmax"));
}

#[test]
fn test_exp_family_detection() {
    assert!(is_exp_family("Exp"));
    assert!(is_exp_family("Pow"));
    assert!(is_exp_family("Softmax"));
    assert!(is_exp_family("LogSoftmax"));
    assert!(!is_exp_family("Linear"));
    assert!(!is_exp_family("ReLU"));
}

// ---------------------------------------------------------------------------
// Norm chain explosion detection (#2708)
// ---------------------------------------------------------------------------

/// Build a chain of `n` InstanceNorm layers where each layer widens bounds
/// by `per_layer_factor`. Input of first layer is `(-1.0, 1.0)`.
pub(super) fn make_norm_chain(n: usize, per_layer_factor: f32) -> Vec<LayerBoundRecord> {
    let mut records = Vec::with_capacity(n);
    let mut lo = -1.0f32;
    let mut hi = 1.0f32;

    for i in 0..n {
        let in_lo = lo;
        let in_hi = hi;
        lo *= per_layer_factor;
        hi *= per_layer_factor;
        records.push(make_record(
            i,
            "InstanceNorm",
            vec![(in_lo, in_hi)],
            vec![(lo, hi)],
            PropMethod::Ibp,
            Some(&format!("norm_{i}")),
        ));
    }
    records
}

#[test]
fn test_norm_chain_explosion_detected_6_layers_20x() {
    // 6 layers, each 2x expansion => total 2^6 = 64x (well above 10x threshold).
    let records = make_norm_chain(6, 2.0);
    let config = AnalysisConfig {
        norm_chain_min_length: 5,
        norm_chain_explosion_ratio: 10.0,
        ..default_config()
    };
    let report = analyze_layer_bounds("kokoro_test", &records, &config);

    let chain_recs: Vec<_> = report
        .recommendations
        .iter()
        .filter(|r| matches!(r, TighteningRecommendation::NormChainExplosion { .. }))
        .collect();
    assert_eq!(chain_recs.len(), 1);

    match &chain_recs[0] {
        TighteningRecommendation::NormChainExplosion {
            start_layer,
            end_layer,
            chain_depth,
            total_expansion,
            per_layer_expansions,
            layer_types,
        } => {
            assert_eq!(*start_layer, 0);
            assert_eq!(*end_layer, 5);
            assert_eq!(*chain_depth, 6);
            // Total: 64x (2^6).
            assert!(*total_expansion > 60.0 && *total_expansion < 70.0);
            assert_eq!(per_layer_expansions.len(), 6);
            assert_eq!(layer_types.len(), 6);
            assert!(layer_types.iter().all(|t| t == "InstanceNorm"));
        }
        _ => unreachable!(),
    }
}

#[test]
fn test_norm_chain_below_threshold_not_flagged() {
    // 6 layers, each 1.1x expansion => total ~1.77x (below 10x threshold).
    let records = make_norm_chain(6, 1.1);
    let config = AnalysisConfig {
        norm_chain_min_length: 5,
        norm_chain_explosion_ratio: 10.0,
        ..default_config()
    };
    let report = analyze_layer_bounds("tight_model", &records, &config);

    let chain_recs: Vec<_> = report
        .recommendations
        .iter()
        .filter(|r| matches!(r, TighteningRecommendation::NormChainExplosion { .. }))
        .collect();
    assert!(chain_recs.is_empty());
}

#[test]
fn test_norm_chain_too_short_not_flagged() {
    // 4 layers (below min_length=5), each 3x => total 81x.
    let records = make_norm_chain(4, 3.0);
    let config = AnalysisConfig {
        norm_chain_min_length: 5,
        norm_chain_explosion_ratio: 10.0,
        ..default_config()
    };
    let report = analyze_layer_bounds("short_chain", &records, &config);

    let chain_recs: Vec<_> = report
        .recommendations
        .iter()
        .filter(|r| matches!(r, TighteningRecommendation::NormChainExplosion { .. }))
        .collect();
    assert!(chain_recs.is_empty());
}

#[test]
fn test_norm_chain_interrupted_by_linear() {
    // 3 norms, then Linear, then 3 norms — neither chain reaches min_length=5.
    let mut records = make_norm_chain(3, 2.0);
    records.push(make_record(
        3,
        "Linear",
        vec![(-8.0, 8.0)],
        vec![(-10.0, 10.0)],
        PropMethod::Crown,
        None,
    ));
    for i in 4..7 {
        records.push(make_record(
            i,
            "InstanceNorm",
            vec![(-10.0, 10.0)],
            vec![(-20.0, 20.0)],
            PropMethod::Ibp,
            Some(&format!("norm_{i}")),
        ));
    }
    let config = AnalysisConfig {
        norm_chain_min_length: 5,
        norm_chain_explosion_ratio: 10.0,
        ..default_config()
    };
    let report = analyze_layer_bounds("interrupted", &records, &config);

    let chain_recs: Vec<_> = report
        .recommendations
        .iter()
        .filter(|r| matches!(r, TighteningRecommendation::NormChainExplosion { .. }))
        .collect();
    assert!(chain_recs.is_empty());
}

#[test]
fn test_norm_chain_mixed_norm_types() {
    // Mix of InstanceNorm, GroupNorm, RMSNorm — all count as norm layers.
    let types = [
        "InstanceNorm",
        "GroupNorm",
        "RMSNorm",
        "InstanceNorm",
        "LayerNorm",
        "BatchNorm",
    ];
    let mut records = Vec::new();
    let mut lo = -1.0f32;
    let mut hi = 1.0f32;

    for (i, &ty) in types.iter().enumerate() {
        let in_lo = lo;
        let in_hi = hi;
        lo *= 2.0;
        hi *= 2.0;
        records.push(make_record(
            i,
            ty,
            vec![(in_lo, in_hi)],
            vec![(lo, hi)],
            PropMethod::Ibp,
            None,
        ));
    }
    let config = AnalysisConfig {
        norm_chain_min_length: 5,
        norm_chain_explosion_ratio: 10.0,
        ..default_config()
    };
    let report = analyze_layer_bounds("mixed_norms", &records, &config);

    let chain_recs: Vec<_> = report
        .recommendations
        .iter()
        .filter(|r| matches!(r, TighteningRecommendation::NormChainExplosion { .. }))
        .collect();
    assert_eq!(chain_recs.len(), 1);

    match &chain_recs[0] {
        TighteningRecommendation::NormChainExplosion {
            chain_depth,
            layer_types,
            ..
        } => {
            assert_eq!(*chain_depth, 6);
            assert_eq!(layer_types[0], "InstanceNorm");
            assert_eq!(layer_types[1], "GroupNorm");
            assert_eq!(layer_types[2], "RMSNorm");
            assert_eq!(layer_types[4], "LayerNorm");
            assert_eq!(layer_types[5], "BatchNorm");
        }
        _ => unreachable!(),
    }
}

#[test]
fn test_norm_chain_nonfinite_bounds_flagged() {
    // Chain of 6 norm layers where one output is Inf — total expansion is Inf.
    let mut records = make_norm_chain(6, 1.5);
    // Make layer 3 output non-finite.
    records[3].output_bounds = vec![(f32::NEG_INFINITY, f32::INFINITY)];
    // Also make downstream layers reflect the non-finite input.
    for rec in &mut records[4..6] {
        rec.input_bounds = vec![(f32::NEG_INFINITY, f32::INFINITY)];
        rec.output_bounds = vec![(f32::NEG_INFINITY, f32::INFINITY)];
    }

    let config = AnalysisConfig {
        norm_chain_min_length: 5,
        norm_chain_explosion_ratio: 10.0,
        ..default_config()
    };
    let report = analyze_layer_bounds("nonfinite_chain", &records, &config);

    let chain_recs: Vec<_> = report
        .recommendations
        .iter()
        .filter(|r| matches!(r, TighteningRecommendation::NormChainExplosion { .. }))
        .collect();
    assert_eq!(chain_recs.len(), 1);

    match &chain_recs[0] {
        TighteningRecommendation::NormChainExplosion {
            total_expansion, ..
        } => {
            assert!(total_expansion.is_infinite());
        }
        _ => unreachable!(),
    }
}

#[test]
fn test_norm_chain_explosion_json_round_trip() {
    let rec = TighteningRecommendation::NormChainExplosion {
        start_layer: 0,
        end_layer: 57,
        chain_depth: 58,
        total_expansion: 17.3,
        per_layer_expansions: vec![1.003; 58],
        layer_types: vec!["InstanceNorm".to_string(); 58],
    };
    let json = serde_json::to_string(&rec).expect("serialize");
    let parsed: TighteningRecommendation = serde_json::from_str(&json).expect("deserialize");

    match parsed {
        TighteningRecommendation::NormChainExplosion {
            start_layer,
            end_layer,
            chain_depth,
            total_expansion,
            ..
        } => {
            assert_eq!(start_layer, 0);
            assert_eq!(end_layer, 57);
            assert_eq!(chain_depth, 58);
            assert!((total_expansion - 17.3).abs() < 1e-4);
        }
        _ => panic!("Wrong variant after round-trip"),
    }
}

/// Kokoro Generator: 58 InstanceNorm, 1.05x/layer = 17.5x total. Part of #2708 AC3.
#[test]
fn test_kokoro_generator_58_layer_norm_chain_detected() {
    let records = make_norm_chain(58, 1.05);
    let config = AnalysisConfig::default(); // min_length=5, ratio=10.0
    let report = analyze_layer_bounds("kokoro_generator_synthetic", &records, &config);

    let chain_recs: Vec<_> = report
        .recommendations
        .iter()
        .filter(|r| matches!(r, TighteningRecommendation::NormChainExplosion { .. }))
        .collect();

    assert_eq!(
        chain_recs.len(),
        1,
        "Kokoro 58-layer chain should trigger exactly 1 NormChainExplosion"
    );
    match &chain_recs[0] {
        TighteningRecommendation::NormChainExplosion {
            chain_depth,
            total_expansion,
            ..
        } => {
            assert_eq!(
                *chain_depth, 58,
                "chain_depth must be 58 for Kokoro Generator"
            );
            // 1.05^58 ≈ 17.5
            assert!(
                *total_expansion > 10.0,
                "total_expansion={total_expansion} must exceed 10x"
            );
        }
        _ => unreachable!(),
    }
}

/// Regression: ExtractForSmt must use record.layer_index, not array position.
///
/// When layer_index values are sparse (graph-aware verification with
/// non-contiguous indices), array positions diverge from semantic layer IDs.
/// The bug was masked by prior tests using contiguous indices (0, 1, 2).
///
/// Algorithm audit finding, Part of #3020.
#[test]
fn test_smt_extraction_sparse_layer_indices() {
    let config = AnalysisConfig {
        smt_max_elements: 256,
        ..default_config()
    };
    // Sparse indices: 10, 20, 30 (array positions 0, 1, 2).
    let records = vec![
        make_record(
            10,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-2.0, 2.0), (0.0, 1.0)], // 2 elements
            PropMethod::Ibp,
            None,
        ),
        make_record(
            20,
            "ReLU",
            vec![(0.0, 2.0), (0.0, 1.0)],
            vec![(0.0, 2.0), (0.0, 1.0)], // 2 elements
            PropMethod::Ibp,
            None,
        ),
        make_record(
            30,
            "Linear",
            vec![(0.0, 2.0), (0.0, 1.0)],
            vec![(-1.0, 3.0)], // 1 element
            PropMethod::Ibp,
            None,
        ),
    ];
    let report = analyze_layer_bounds("test_sparse", &records, &config);

    let smt_recs: Vec<_> = report
        .recommendations
        .iter()
        .filter(|r| matches!(r, TighteningRecommendation::ExtractForSmt { .. }))
        .collect();
    assert_eq!(smt_recs.len(), 1);

    match &smt_recs[0] {
        TighteningRecommendation::ExtractForSmt {
            start_layer,
            end_layer,
            estimated_elements,
        } => {
            // Must be semantic layer indices (10, 30), NOT array positions (0, 2).
            assert_eq!(
                *start_layer, 10,
                "start_layer must be layer_index, not array position"
            );
            assert_eq!(
                *end_layer, 30,
                "end_layer must be layer_index, not array position"
            );
            assert_eq!(*estimated_elements, 5); // 2 + 2 + 1
        }
        _ => unreachable!(),
    }
}

/// Healthy Kokoro: 1.001x/layer = 1.06x total, no false positive. Part of #2708 AC3.
#[test]
fn test_kokoro_generator_58_layer_healthy_no_explosion() {
    let records = make_norm_chain(58, 1.001);
    let config = AnalysisConfig::default();
    let report = analyze_layer_bounds("kokoro_generator_healthy", &records, &config);

    let chain_recs: Vec<_> = report
        .recommendations
        .iter()
        .filter(|r| matches!(r, TighteningRecommendation::NormChainExplosion { .. }))
        .collect();

    assert!(
        chain_recs.is_empty(),
        "Healthy Kokoro (1.001x/layer = {:.4}x total) should NOT trigger explosion",
        1.001f32.powi(58)
    );
}

// ---------------------------------------------------------------------------
// layers_needing_crown() — Phase 2 selective CROWN (#2454)
// ---------------------------------------------------------------------------

#[test]
fn test_layers_needing_crown_empty_report() {
    let report = analyze_layer_bounds("empty", &[], &default_config());
    let crown_layers = layers_needing_crown(&report);
    assert!(crown_layers.is_empty());
}

#[test]
fn test_layers_needing_crown_no_escalation_when_tight() {
    // All CROWN layers — no EscalateToCrown recommendations.
    let records = vec![
        make_record(
            0,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-200.0, 200.0)],
            PropMethod::Crown,
            None,
        ),
        make_record(
            1,
            "ReLU",
            vec![(-200.0, 200.0)],
            vec![(0.0, 200.0)],
            PropMethod::Crown,
            None,
        ),
    ];
    let config = AnalysisConfig {
        crown_escalation_width: 100.0,
        ..default_config()
    };
    let report = analyze_layer_bounds("test", &records, &config);
    let crown_layers = layers_needing_crown(&report);
    assert!(
        crown_layers.is_empty(),
        "CROWN layers should not trigger EscalateToCrown"
    );
}

#[test]
fn test_layers_needing_crown_identifies_wide_ibp_layers() {
    let config = AnalysisConfig {
        crown_escalation_width: 100.0,
        ..default_config()
    };
    let records = vec![
        make_record(
            0,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-50.0, 50.0)],
            PropMethod::Ibp,
            None,
        ),
        make_record(
            1,
            "Linear",
            vec![(-50.0, 50.0)],
            vec![(-200.0, 200.0)],
            PropMethod::Ibp,
            Some("wide_linear"),
        ),
        make_record(
            2,
            "ReLU",
            vec![(-200.0, 200.0)],
            vec![(0.0, 200.0)],
            PropMethod::Ibp,
            None,
        ),
        make_record(
            3,
            "Linear",
            vec![(0.0, 200.0)],
            vec![(-500.0, 500.0)],
            PropMethod::Ibp,
            Some("wider_linear"),
        ),
    ];
    let report = analyze_layer_bounds("test", &records, &config);
    let crown_layers = layers_needing_crown(&report);

    // Layers 1, 2, 3 have width > 100 (200, 200, 1000). Layer 0 has width=100, not > 100.
    assert!(
        crown_layers.contains(&1),
        "Layer 1 (width=400) should need CROWN: got {crown_layers:?}"
    );
    assert!(
        crown_layers.contains(&3),
        "Layer 3 (width=1000) should need CROWN: got {crown_layers:?}"
    );
}

#[test]
fn test_layers_needing_crown_sorted_and_deduped() {
    let config = AnalysisConfig {
        crown_escalation_width: 50.0,
        ..default_config()
    };
    // Multiple layers that trigger EscalateToCrown. Verify sorted + deduped.
    let records = vec![
        make_record(
            5,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-100.0, 100.0)],
            PropMethod::Ibp,
            None,
        ),
        make_record(
            2,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-200.0, 200.0)],
            PropMethod::Ibp,
            None,
        ),
        make_record(
            8,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-300.0, 300.0)],
            PropMethod::Ibp,
            None,
        ),
    ];
    let report = analyze_layer_bounds("test", &records, &config);
    let crown_layers = layers_needing_crown(&report);

    // All three layers should appear, sorted.
    assert_eq!(crown_layers, vec![2, 5, 8]);
}

#[test]
fn test_layers_needing_crown_mixed_methods() {
    // Mix of IBP and CROWN layers. Only IBP layers with wide bounds get flagged.
    let config = AnalysisConfig {
        crown_escalation_width: 100.0,
        ..default_config()
    };
    let records = vec![
        make_record(
            0,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-500.0, 500.0)],
            PropMethod::Ibp,
            None,
        ),
        make_record(
            1,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-500.0, 500.0)],
            PropMethod::Crown,
            None,
        ),
        make_record(
            2,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-500.0, 500.0)],
            PropMethod::Ibp,
            None,
        ),
    ];
    let report = analyze_layer_bounds("test", &records, &config);
    let crown_layers = layers_needing_crown(&report);

    // Layer 0 and 2 are IBP with width=1000 > 100 → escalate.
    // Layer 1 is already CROWN → no escalation.
    assert_eq!(crown_layers, vec![0, 2]);
}

/// Demonstrates width reduction: a graph where selective CROWN on explosion
/// points would tighten bounds compared to IBP-only. Part of #2454 AC2.
///
/// Scenario: 5-layer graph where layer 2 is a Linear explosion point
/// (100x expansion under IBP) but surrounding layers are well-behaved.
/// `layers_needing_crown` identifies only layer 2 for CROWN tightening.
#[test]
fn test_selective_crown_width_reduction_scenario() {
    let config = AnalysisConfig {
        crown_escalation_width: 100.0,
        explosion_threshold: 50.0,
        ..default_config()
    };

    // Build a graph with one explosion point at layer 2.
    let records = vec![
        make_record(
            0,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-2.0, 2.0)],
            PropMethod::Ibp,
            Some("l0"),
        ),
        make_record(
            1,
            "ReLU",
            vec![(-2.0, 2.0)],
            vec![(0.0, 2.0)],
            PropMethod::Ibp,
            Some("l1"),
        ),
        make_record(
            2,
            "Linear",
            vec![(0.0, 2.0)],
            vec![(-1e5, 1e5)],
            PropMethod::Ibp,
            Some("l2_explosion"),
        ),
        make_record(
            3,
            "ReLU",
            vec![(-1e5, 1e5)],
            vec![(0.0, 1e5)],
            PropMethod::Ibp,
            Some("l3"),
        ),
        make_record(
            4,
            "Linear",
            vec![(0.0, 1e5)],
            vec![(-5e4, 5e4)],
            PropMethod::Ibp,
            Some("l4"),
        ),
    ];

    let report = analyze_layer_bounds("selective_crown_demo", &records, &config);

    // Verify layer 2 is the explosion point.
    assert!(
        report.explosion_points.contains(&2),
        "Layer 2 should be an explosion point: {:?}",
        report.explosion_points
    );

    // layers_needing_crown should identify layer 2 (and possibly others with width > 100).
    let crown_layers = layers_needing_crown(&report);
    assert!(
        crown_layers.contains(&2),
        "Layer 2 (width=2e5) must be in layers_needing_crown: {crown_layers:?}"
    );

    // Simulate CROWN tightening: if layer 2 were CROWN-tightened to width 50
    // (instead of 2e5), downstream widths would be dramatically reduced.
    let tightened_records = vec![
        make_record(
            0,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-2.0, 2.0)],
            PropMethod::Ibp,
            Some("l0"),
        ),
        make_record(
            1,
            "ReLU",
            vec![(-2.0, 2.0)],
            vec![(0.0, 2.0)],
            PropMethod::Ibp,
            Some("l1"),
        ),
        make_record(
            2,
            "Linear",
            vec![(0.0, 2.0)],
            vec![(-25.0, 25.0)],
            PropMethod::Crown,
            Some("l2_tightened"),
        ),
        make_record(
            3,
            "ReLU",
            vec![(-25.0, 25.0)],
            vec![(0.0, 25.0)],
            PropMethod::Ibp,
            Some("l3"),
        ),
        make_record(
            4,
            "Linear",
            vec![(0.0, 25.0)],
            vec![(-12.5, 12.5)],
            PropMethod::Ibp,
            Some("l4"),
        ),
    ];
    let tightened_report =
        analyze_layer_bounds("selective_crown_tightened", &tightened_records, &config);

    // After tightening: output width shrinks from 1e5 to 25.
    assert!(
        tightened_report.output_width < report.output_width,
        "Tightened output width ({}) should be < IBP output width ({})",
        tightened_report.output_width,
        report.output_width
    );

    // And the tightened report should have fewer (or no) EscalateToCrown recommendations
    // since the explosion point is now CROWN-verified.
    let tightened_crown_layers = layers_needing_crown(&tightened_report);
    assert!(
        tightened_crown_layers.len() <= crown_layers.len(),
        "Tightened report should have <= escalation recommendations: {} vs {}",
        tightened_crown_layers.len(),
        crown_layers.len()
    );
}
