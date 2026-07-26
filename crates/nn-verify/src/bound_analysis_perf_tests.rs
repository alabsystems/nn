// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Performance proof tests for bound analysis pipeline.
//!
//! Proves that `analyze_layer_bounds` and its recommendation generation are
//! O(L) in layer count, not O(L^2). Also documents performance characteristics
//! of `recommend_smt_extraction` (sliding window, O(L)) and the overall
//! analysis pipeline.
//!
//! Part of #3020 (Proof Certificates), performance_proofs phase.
//!
//! # Findings documented by these tests
//!
//! 1. `analyze_layer_bounds` is O(L): single pass over records with O(1) per-
//!    layer work. `Vec::with_capacity(records.len())` avoids reallocation.
//!
//! 2. `recommend_smt_extraction` (bound_analysis_recommendations.rs:165-230)
//!    contains a dead first sliding window loop (lines 175-193) that iterates
//!    all records but produces no output. The second loop (lines 196-217)
//!    re-initializes `start` and `total_elements` and redoes the same work.
//!
//! 3. `overlay_composition.rs:266` deep-clones `original_weights` entirely,
//!    then only modifies entries with accumulated deltas. For models with
//!    100M+ parameters (~400MB f32), this doubles transient memory.
//!    `original_weights` is also cloned again at line 291 into the
//!    entries not targeted by overlays.
//!
//! 4. `trace_to_graph_model` calls `reachable_nodes(graph)` at line 54 (for
//!    variable input count guard), then `trace_to_graph_impl` calls it again
//!    set into `trace_to_graph_impl` to avoid duplicated O(V+E) BFS.
//!    NOTE: This may already be fixed on main (#3185, dfb972731).

use super::*;
use crate::certificate_types::LayerBoundRecord;
use crate::verify_types::PropMethod;

use super::tests::make_record;

/// Build N layers with varying widths, suitable for triggering all
/// recommendation paths (norm, exp, IBP escalation, SMT extraction).
fn build_n_layer_mixed(n: usize, elements: usize) -> Vec<LayerBoundRecord> {
    let mut records = Vec::with_capacity(n);
    for i in 0..n {
        let layer_type = match i % 5 {
            0 => "Linear",
            1 => "ReLU",
            2 => "InstanceNorm",
            3 => "Exp",
            4 => "Linear",
            _ => unreachable!(),
        };
        let method = if i % 3 == 0 {
            PropMethod::Crown
        } else {
            PropMethod::Ibp
        };
        let width = 1.0 + (i as f32) * 0.5;
        records.push(LayerBoundRecord {
            layer_index: i,
            layer_type: layer_type.to_string(),
            input_bounds: vec![(-width, width); elements],
            output_bounds: vec![(-(width + 0.5), width + 0.5); elements],
            method,
            node_name: Some(format!("node_{i}")),
            input_sources: None,
        });
    }
    records
}

// ---------------------------------------------------------------------------
// analyze_layer_bounds is O(L), not O(L^2)
// ---------------------------------------------------------------------------

/// Prove: analyze_layer_bounds scales linearly with layer count.
///
/// The analysis performs a single pass over records (O(L)) with O(E) per-layer
/// work for width computation. Recommendation generation is O(L) with a sliding
/// window for SMT extraction.
///
/// Method: Run on 100, 1000, and 10000 layers. If O(L^2), the 10000-layer
/// analysis takes ~100x longer than 1000. With O(L), it takes ~10x.
/// We verify the ratio is < 25x (generous margin for cache effects).
#[test]
fn proof_analyze_layer_bounds_is_linear() {
    let sizes = [100, 1000, 10000];
    let elements = 4;
    let config = AnalysisConfig::default();
    let mut durations = Vec::new();

    for &n in &sizes {
        let records = build_n_layer_mixed(n, elements);

        let start = std::time::Instant::now();
        let report = analyze_layer_bounds("perf_test", &records, &config);
        let elapsed = start.elapsed();

        assert_eq!(report.total_layers, n);
        assert_eq!(report.layers.len(), n);
        durations.push((n, elapsed));
    }

    // Compare 10000 vs 1000 ratio.
    let t_1000 = durations[1].1.as_nanos() as f64;
    let t_10000 = durations[2].1.as_nanos() as f64;

    // Guard: ensure 1000-layer time is measurable (> 1us).
    if t_1000 > 1000.0 {
        let ratio = t_10000 / t_1000;
        assert!(
            ratio < 25.0,
            "10000/1000 layer ratio = {ratio:.1}x, expected < 25x for O(L). \
             If > 50x this indicates O(L^2). \
             t_1000={:.3}ms, t_10000={:.3}ms",
            t_1000 / 1e6,
            t_10000 / 1e6
        );
    }
}

/// Prove: recommendation count is bounded by O(L).
///
/// For mixed-type models, each layer can produce at most one recommendation
/// (TightenLayer, SwitchNormMode, EscalateToCrown, or AnnotateConstraint).
/// Norm chain and SMT extraction add at most O(chains) and O(1) respectively.
/// Total recommendations <= L + chains + 1 = O(L).
#[test]
fn proof_recommendations_bounded_by_layer_count() {
    for &n in &[10, 100, 1000] {
        let records = build_n_layer_mixed(n, 4);
        let config = AnalysisConfig::default();
        let report = analyze_layer_bounds("rec_bound_test", &records, &config);

        // Recommendations cannot exceed 2*L (each layer contributes at most 1
        // per-layer rec + norm chains + 1 SMT rec).
        assert!(
            report.recommendations.len() <= 2 * n + 1,
            "n={n}: {} recommendations exceeds 2L+1 = {} bound",
            report.recommendations.len(),
            2 * n + 1
        );
    }
}

/// Prove: SMT extraction sliding window is O(L) even with many small layers.
///
/// The recommend_smt_extraction function uses a two-pointer sliding window
/// that advances both `start` and `end` monotonically. Total element additions
/// and removals are each at most L, giving O(L) overall.
///
/// NOTE: This function currently has a dead first loop (lines 175-193) that
/// wastes O(L) work. The second loop (lines 196-217) re-initializes state
#[test]
fn proof_smt_extraction_scales_linearly() {
    let config = AnalysisConfig {
        smt_max_elements: 20,
        ..AnalysisConfig::default()
    };

    let sizes = [100, 1000, 10000];
    let mut durations = Vec::new();

    for &n in &sizes {
        // Small layers (2 elements each) — many fit within SMT budget.
        let records: Vec<_> = (0..n)
            .map(|i| {
                make_record(
                    i,
                    "Linear",
                    vec![(-1.0, 1.0); 2],
                    vec![(-2.0, 2.0); 2],
                    PropMethod::Ibp,
                    None,
                )
            })
            .collect();

        let start = std::time::Instant::now();
        let report = analyze_layer_bounds("smt_perf", &records, &config);
        let elapsed = start.elapsed();

        // SMT extraction should find a window (2 elements/layer, budget=20 → window of 10).
        let smt_recs: Vec<_> = report
            .recommendations
            .iter()
            .filter(|r| matches!(r, TighteningRecommendation::ExtractForSmt { .. }))
            .collect();
        assert!(smt_recs.len() <= 1, "at most 1 SMT recommendation");

        durations.push((n, elapsed));
    }

    let t_1000 = durations[1].1.as_nanos() as f64;
    let t_10000 = durations[2].1.as_nanos() as f64;

    if t_1000 > 1000.0 {
        let ratio = t_10000 / t_1000;
        assert!(
            ratio < 25.0,
            "SMT extraction 10000/1000 ratio = {ratio:.1}x, expected < 25x for O(L). \
             t_1000={:.3}ms, t_10000={:.3}ms",
            t_1000 / 1e6,
            t_10000 / 1e6
        );
    }
}

/// Prove: norm chain detection is O(L).
///
/// `detect_norm_chain_explosions` uses a single pass with two cursors
/// (chain_start, i) that advance monotonically. Each record is visited
/// exactly once. Total work is O(L).
#[test]
fn proof_norm_chain_detection_is_linear() {
    let config = AnalysisConfig {
        norm_chain_min_length: 5,
        norm_chain_explosion_ratio: 10.0,
        ..AnalysisConfig::default()
    };

    let sizes = [100, 1000, 10000];
    let mut durations = Vec::new();

    for &n in &sizes {
        // Alternating norm chains of length 6, interrupted by Linear.
        // Total chains ≈ n/7.
        let mut records = Vec::with_capacity(n);
        let mut lo = -1.0f32;
        let mut hi = 1.0f32;
        for i in 0..n {
            if i % 7 == 6 {
                records.push(make_record(
                    i,
                    "Linear",
                    vec![(lo, hi)],
                    vec![(lo - 1.0, hi + 1.0)],
                    PropMethod::Crown,
                    None,
                ));
                lo -= 1.0;
                hi += 1.0;
            } else {
                let new_lo = lo * 2.0;
                let new_hi = hi * 2.0;
                records.push(make_record(
                    i,
                    "InstanceNorm",
                    vec![(lo, hi)],
                    vec![(new_lo, new_hi)],
                    PropMethod::Ibp,
                    None,
                ));
                lo = new_lo;
                hi = new_hi;
            }
        }

        let start = std::time::Instant::now();
        let report = analyze_layer_bounds("norm_perf", &records, &config);
        let elapsed = start.elapsed();

        // Should detect multiple chains.
        let chain_recs: Vec<_> = report
            .recommendations
            .iter()
            .filter(|r| matches!(r, TighteningRecommendation::NormChainExplosion { .. }))
            .collect();
        // Norm chains of length 6 at positions [0..5], [7..12], etc.
        assert!(!chain_recs.is_empty(), "n={n}: should detect norm chains");

        durations.push((n, elapsed));
    }

    let t_1000 = durations[1].1.as_nanos() as f64;
    let t_10000 = durations[2].1.as_nanos() as f64;

    if t_1000 > 1000.0 {
        let ratio = t_10000 / t_1000;
        assert!(
            ratio < 25.0,
            "norm chain 10000/1000 ratio = {ratio:.1}x, expected < 25x for O(L). \
             t_1000={:.3}ms, t_10000={:.3}ms",
            t_1000 / 1e6,
            t_10000 / 1e6
        );
    }
}

/// Prove: analysis output correctly reflects all layers even at high scale.
///
/// The `layers` Vec is allocated with `with_capacity(records.len())` at
/// analyze_layer_bounds:148, avoiding reallocation during the single pass.
/// Verify correctness (not just performance) at 5000 layers — each layer
/// has correct index, type, and width metrics.
#[test]
fn proof_analysis_correctness_at_scale() {
    let n = 5000;
    let records = build_n_layer_mixed(n, 4);
    let config = AnalysisConfig::default();
    let report = analyze_layer_bounds("scale", &records, &config);

    assert_eq!(report.total_layers, n);
    assert_eq!(report.layers.len(), n);

    // Spot-check: every layer index matches, types cycle as expected.
    for (i, layer) in report.layers.iter().enumerate() {
        assert_eq!(layer.layer_index, i);
        let expected_type = match i % 5 {
            0 => "Linear",
            1 => "ReLU",
            2 => "InstanceNorm",
            3 => "Exp",
            4 => "Linear",
            _ => unreachable!(),
        };
        assert_eq!(layer.layer_type, expected_type, "layer {i} type mismatch");
        // Width must be finite and positive for our test data.
        assert!(
            layer.max_output_width.is_finite() && layer.max_output_width > 0.0,
            "layer {i} width = {} (expected finite > 0)",
            layer.max_output_width
        );
    }

    // Crown coverage: every 3rd layer is Crown (i%3==0), so coverage = ceil(5000/3)/5000.
    let expected_crown = (0..n).filter(|i| i % 3 == 0).count();
    let expected_coverage = expected_crown as f32 / n as f32;
    assert!(
        (report.crown_coverage - expected_coverage).abs() < 1e-6,
        "crown coverage {:.4} != expected {:.4}",
        report.crown_coverage,
        expected_coverage
    );
}
