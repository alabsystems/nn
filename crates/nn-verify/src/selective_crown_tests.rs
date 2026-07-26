// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for selective CROWN escalation (#2454).

use super::*;
use crate::bound_analysis::AnalysisConfig;
use crate::certificate_types::LayerBoundRecord;
use crate::verify_types::PropMethod;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_record(
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

// ---------------------------------------------------------------------------
// SelectiveCrownConfig tests
// ---------------------------------------------------------------------------

#[test]
fn test_default_config() {
    let config = SelectiveCrownConfig::default();
    assert!((config.min_width_to_tighten - 5.0).abs() < f32::EPSILON);
    assert_eq!(config.max_crown_layers, 10);
    assert_eq!(config.escalation_strategy, EscalationStrategy::WidestFirst);
}

#[test]
fn test_config_builder() {
    let config = SelectiveCrownConfig::default()
        .with_min_width(10.0)
        .with_max_crown_layers(3)
        .with_strategy(EscalationStrategy::AllAboveThreshold);
    assert!((config.min_width_to_tighten - 10.0).abs() < f32::EPSILON);
    assert_eq!(config.max_crown_layers, 3);
    assert_eq!(
        config.escalation_strategy,
        EscalationStrategy::AllAboveThreshold
    );
}

#[test]
fn test_to_analysis_config() {
    let config = SelectiveCrownConfig::default().with_min_width(42.0);
    let analysis = config.to_analysis_config();
    assert!((analysis.crown_escalation_width - 42.0).abs() < f32::EPSILON);
}

// ---------------------------------------------------------------------------
// select_crown_layers tests
// ---------------------------------------------------------------------------

#[test]
fn test_select_empty_report() {
    let report = analyze_layer_bounds("empty", &[], &AnalysisConfig::default());
    let config = SelectiveCrownConfig::default();
    let layers = select_crown_layers(&report, &config);
    assert!(layers.is_empty());
}

#[test]
fn test_select_no_wide_layers() {
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
            PropMethod::Ibp,
            None,
        ),
    ];
    let report = analyze_layer_bounds("test", &records, &AnalysisConfig::default());
    let config = SelectiveCrownConfig::default(); // threshold = 5.0
    let layers = select_crown_layers(&report, &config);
    // All widths <= 4.0, below threshold of 5.0.
    assert!(layers.is_empty());
}

#[test]
fn test_select_identifies_wide_ibp_layers() {
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
            "Linear",
            vec![(-2.0, 2.0)],
            vec![(-10.0, 10.0)],
            PropMethod::Ibp,
            Some("wide"),
        ),
        make_record(
            2,
            "ReLU",
            vec![(-10.0, 10.0)],
            vec![(0.0, 10.0)],
            PropMethod::Ibp,
            None,
        ),
    ];
    let report = analyze_layer_bounds("test", &records, &AnalysisConfig::default());
    let config = SelectiveCrownConfig::default().with_min_width(5.0);
    let layers = select_crown_layers(&report, &config);

    // Layer 1 has width=20 > 5, layer 2 has width=10 > 5. Layer 0 has width=4, below.
    assert!(
        layers.contains(&1),
        "Layer 1 should be selected: {layers:?}"
    );
    assert!(
        layers.contains(&2),
        "Layer 2 should be selected: {layers:?}"
    );
    assert!(
        !layers.contains(&0),
        "Layer 0 should not be selected: {layers:?}"
    );
}

#[test]
fn test_select_skips_already_crown_layers() {
    let records = vec![
        make_record(
            0,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-100.0, 100.0)],
            PropMethod::Crown, // Already CROWN
            None,
        ),
        make_record(
            1,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-100.0, 100.0)],
            PropMethod::Ibp,
            None,
        ),
    ];
    let report = analyze_layer_bounds("test", &records, &AnalysisConfig::default());
    let config = SelectiveCrownConfig::default().with_min_width(5.0);
    let layers = select_crown_layers(&report, &config);

    assert!(
        !layers.contains(&0),
        "CROWN layer should not be re-escalated"
    );
    assert!(
        layers.contains(&1),
        "IBP layer with wide bounds should be selected"
    );
}

#[test]
fn test_select_widest_first_caps_count() {
    let records = vec![
        make_record(
            0,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-50.0, 50.0)],
            PropMethod::Ibp,
            None,
        ), // width=100
        make_record(
            1,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-200.0, 200.0)],
            PropMethod::Ibp,
            None,
        ), // width=400
        make_record(
            2,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-10.0, 10.0)],
            PropMethod::Ibp,
            None,
        ), // width=20
        make_record(
            3,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-300.0, 300.0)],
            PropMethod::Ibp,
            None,
        ), // width=600
    ];
    let report = analyze_layer_bounds("test", &records, &AnalysisConfig::default());
    let config = SelectiveCrownConfig::default()
        .with_min_width(5.0)
        .with_max_crown_layers(2)
        .with_strategy(EscalationStrategy::WidestFirst);
    let layers = select_crown_layers(&report, &config);

    // All 4 are above threshold, but cap=2. Widest are layer 3 (600) and layer 1 (400).
    assert_eq!(layers.len(), 2, "Should cap at 2 layers: {layers:?}");
    assert!(
        layers.contains(&1),
        "Layer 1 (width=400) should be selected"
    );
    assert!(
        layers.contains(&3),
        "Layer 3 (width=600) should be selected"
    );
}

#[test]
fn test_select_all_above_threshold_no_cap() {
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
            vec![(-1.0, 1.0)],
            vec![(-200.0, 200.0)],
            PropMethod::Ibp,
            None,
        ),
        make_record(
            2,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-10.0, 10.0)],
            PropMethod::Ibp,
            None,
        ),
        make_record(
            3,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-300.0, 300.0)],
            PropMethod::Ibp,
            None,
        ),
    ];
    let report = analyze_layer_bounds("test", &records, &AnalysisConfig::default());
    let config = SelectiveCrownConfig::default()
        .with_min_width(5.0)
        .with_max_crown_layers(2)
        .with_strategy(EscalationStrategy::AllAboveThreshold);
    let layers = select_crown_layers(&report, &config);

    // AllAboveThreshold ignores max_crown_layers — all 4 qualify.
    assert_eq!(layers.len(), 4, "All layers above threshold: {layers:?}");
}

#[test]
fn test_select_sorted_and_deduped() {
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
    let report = analyze_layer_bounds("test", &records, &AnalysisConfig::default());
    let config = SelectiveCrownConfig::default().with_min_width(5.0);
    let layers = select_crown_layers(&report, &config);

    assert_eq!(layers, vec![2, 5, 8], "Should be sorted by layer index");
}

#[test]
fn test_select_skips_non_finite_width() {
    let records = vec![
        make_record(
            0,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(f32::NEG_INFINITY, f32::INFINITY)],
            PropMethod::Ibp,
            None,
        ),
        make_record(
            1,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-10.0, 10.0)],
            PropMethod::Ibp,
            None,
        ),
    ];
    let report = analyze_layer_bounds("test", &records, &AnalysisConfig::default());
    let config = SelectiveCrownConfig::default().with_min_width(5.0);
    let layers = select_crown_layers(&report, &config);

    // Layer 0 has non-finite width — should be skipped.
    // Layer 1 has width=20 > 5.
    assert!(
        !layers.contains(&0),
        "Non-finite layer should not be selected"
    );
    assert!(layers.contains(&1), "Finite wide layer should be selected");
}

// ---------------------------------------------------------------------------
// simulate_crown_tightening tests
// ---------------------------------------------------------------------------

#[test]
fn test_simulate_tightening_changes_method() {
    let records = vec![
        make_record(
            0,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-10.0, 10.0)],
            PropMethod::Ibp,
            None,
        ),
        make_record(
            1,
            "ReLU",
            vec![(-10.0, 10.0)],
            vec![(0.0, 10.0)],
            PropMethod::Ibp,
            None,
        ),
        make_record(
            2,
            "Linear",
            vec![(0.0, 10.0)],
            vec![(-5.0, 5.0)],
            PropMethod::Crown,
            None,
        ),
    ];

    let tightened = simulate_crown_tightening(&records, &[0, 1]);

    assert_eq!(tightened[0].method, PropMethod::Crown, "Layer 0 tightened");
    assert_eq!(tightened[1].method, PropMethod::Crown, "Layer 1 tightened");
    assert_eq!(
        tightened[2].method,
        PropMethod::Crown,
        "Layer 2 already Crown, unchanged"
    );
}

#[test]
fn test_simulate_tightening_preserves_bounds() {
    let records = vec![make_record(
        0,
        "Linear",
        vec![(-1.0, 1.0)],
        vec![(-10.0, 10.0)],
        PropMethod::Ibp,
        Some("n0"),
    )];

    let tightened = simulate_crown_tightening(&records, &[0]);

    // Bounds are preserved (simulation only changes method).
    assert_eq!(tightened[0].output_bounds, vec![(-10.0, 10.0)]);
    assert_eq!(tightened[0].node_name.as_deref(), Some("n0"));
}

#[test]
fn test_simulate_tightening_no_change_when_empty() {
    let records = vec![make_record(
        0,
        "Linear",
        vec![(-1.0, 1.0)],
        vec![(-10.0, 10.0)],
        PropMethod::Ibp,
        None,
    )];

    let tightened = simulate_crown_tightening(&records, &[]);
    assert_eq!(tightened[0].method, PropMethod::Ibp, "No change when empty");
}

// ---------------------------------------------------------------------------
// analyze_selective_crown tests
// ---------------------------------------------------------------------------

#[test]
fn test_analyze_selective_crown_no_escalation() {
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
            PropMethod::Ibp,
            None,
        ),
    ];

    let config = SelectiveCrownConfig::default().with_min_width(100.0);
    let result = analyze_selective_crown("test", &records, &config);

    assert!(result.crown_layer_indices.is_empty());
    assert_eq!(result.crown_tightened_count, 0);
}

#[test]
fn test_analyze_selective_crown_with_escalation() {
    let records = vec![
        make_record(
            0,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-2.0, 2.0)],
            PropMethod::Ibp,
            Some("tight_layer"),
        ),
        make_record(
            1,
            "Linear",
            vec![(-2.0, 2.0)],
            vec![(-1e5, 1e5)],
            PropMethod::Ibp,
            Some("explosion"),
        ),
        make_record(
            2,
            "ReLU",
            vec![(-1e5, 1e5)],
            vec![(0.0, 1e5)],
            PropMethod::Ibp,
            Some("post_explosion"),
        ),
    ];

    let config = SelectiveCrownConfig::default().with_min_width(5.0);
    let result = analyze_selective_crown("test", &records, &config);

    assert!(
        !result.crown_layer_indices.is_empty(),
        "Should identify layers for CROWN"
    );
    assert!(
        result.crown_layer_indices.contains(&1),
        "Layer 1 (width=2e5) should be escalated"
    );
    assert!(
        result.crown_tightened_count > 0,
        "Should have tightened count > 0"
    );

    // Verify CROWN records have updated method.
    for &idx in &result.crown_layer_indices {
        let crown_rec = result
            .crown_records
            .iter()
            .find(|r| r.layer_index == idx)
            .expect("crown record exists");
        assert_eq!(
            crown_rec.method,
            PropMethod::Crown,
            "Layer {idx} should be CROWN in tightened records"
        );
    }
}

#[test]
fn test_analyze_selective_crown_ibp_records_preserved() {
    let records = vec![make_record(
        0,
        "Linear",
        vec![(-1.0, 1.0)],
        vec![(-100.0, 100.0)],
        PropMethod::Ibp,
        None,
    )];

    let config = SelectiveCrownConfig::default().with_min_width(5.0);
    let result = analyze_selective_crown("test", &records, &config);

    // IBP records are preserved as-is.
    assert_eq!(result.ibp_records.len(), 1);
    assert_eq!(result.ibp_records[0].method, PropMethod::Ibp);
}

// ---------------------------------------------------------------------------
// analyze_and_select tests
// ---------------------------------------------------------------------------

#[test]
fn test_analyze_and_select_convenience() {
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
            "Linear",
            vec![(-2.0, 2.0)],
            vec![(-100.0, 100.0)],
            PropMethod::Ibp,
            None,
        ),
    ];

    let config = SelectiveCrownConfig::default().with_min_width(5.0);
    let (report, crown_layers) = analyze_and_select("test", &records, &config);

    assert_eq!(report.total_layers, 2);
    assert!(
        crown_layers.contains(&1),
        "Layer 1 (width=200) above threshold"
    );
}

// ---------------------------------------------------------------------------
// select_from_recommendations tests
// ---------------------------------------------------------------------------

#[test]
fn test_select_from_recommendations_caps() {
    let config_for_analysis = AnalysisConfig {
        crown_escalation_width: 5.0,
        ..AnalysisConfig::default()
    };
    let records = vec![
        make_record(
            0,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-10.0, 10.0)],
            PropMethod::Ibp,
            None,
        ),
        make_record(
            1,
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
            vec![(-50.0, 50.0)],
            PropMethod::Ibp,
            None,
        ),
    ];
    let report = analyze_layer_bounds("test", &records, &config_for_analysis);

    let crown_config = SelectiveCrownConfig::default()
        .with_min_width(5.0)
        .with_max_crown_layers(1)
        .with_strategy(EscalationStrategy::WidestFirst);

    let layers = select_from_recommendations(&report, &crown_config);

    // All 3 have EscalateToCrown, but cap=1. Widest is layer 1 (width=200).
    assert_eq!(layers.len(), 1);
    assert_eq!(layers[0], 1);
}

#[test]
fn test_select_from_recommendations_no_cap_when_under_limit() {
    let config_for_analysis = AnalysisConfig {
        crown_escalation_width: 5.0,
        ..AnalysisConfig::default()
    };
    let records = vec![
        make_record(
            0,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-10.0, 10.0)],
            PropMethod::Ibp,
            None,
        ),
        make_record(
            1,
            "Linear",
            vec![(-1.0, 1.0)],
            vec![(-20.0, 20.0)],
            PropMethod::Ibp,
            None,
        ),
    ];
    let report = analyze_layer_bounds("test", &records, &config_for_analysis);

    let crown_config = SelectiveCrownConfig::default()
        .with_min_width(5.0)
        .with_max_crown_layers(10);

    let layers = select_from_recommendations(&report, &crown_config);
    assert_eq!(layers, vec![0, 1]);
}

// ---------------------------------------------------------------------------
// EscalationStrategy serde round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_escalation_strategy_serde() {
    let strategies = vec![
        EscalationStrategy::WidestFirst,
        EscalationStrategy::AllAboveThreshold,
    ];
    for strategy in strategies {
        let json = serde_json::to_string(&strategy).expect("serialize");
        let parsed: EscalationStrategy = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(strategy, parsed);
    }
}
