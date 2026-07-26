// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani verification harnesses for the certification orchestrator.
//!
//! Proves safety properties of:
//! - `tightest_enclosing_interval`: finiteness, containment, monotonicity, tightness
//! - `derive_fusion_bounds`: fallback logic, soundness, NaN defense
//! - `classify_graph`: summary consistency, empty graph, exhaustive variant coverage
//!
//! Part of #3614 (Kani harnesses for nn-verify certify + proof_bundle safety).
//! Extended in #3749 with derive_fusion_bounds + classify_graph + harder tightest proofs.

use super::*;
use crate::certificate_types::LayerBoundRecord;
use crate::verify_types::PropMethod;
use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

/// Helper to construct a `LayerBoundRecord` with given output bounds.
fn make_record(output_bounds: Vec<(f32, f32)>) -> LayerBoundRecord {
    LayerBoundRecord {
        layer_index: 0,
        layer_type: "Test".to_string(),
        input_bounds: vec![],
        output_bounds,
        method: PropMethod::Ibp,
        node_name: None,
        input_sources: None,
    }
}

// ---------------------------------------------------------------------------
// tightest_enclosing_interval: finiteness filtering
// ---------------------------------------------------------------------------

/// Prove: `tightest_enclosing_interval` only includes finite bound pairs.
///
/// For any single bound pair where either element is non-finite (NaN, Inf),
/// the function must return None (no valid enclosing interval).
#[kani::unwind(128)]
#[kani::proof]
fn tightest_enclosing_interval_skips_non_finite_lower() {
    let upper: f32 = kani::any();
    kani::assume(upper.is_finite());

    // Test with NaN lower
    let records = vec![make_record(vec![(f32::NAN, upper)])];
    assert!(
        tightest_enclosing_interval(&records).is_none(),
        "NaN lower must be excluded"
    );
}

/// Prove: non-finite upper bound is excluded.
#[kani::unwind(128)]
#[kani::proof]
fn tightest_enclosing_interval_skips_non_finite_upper() {
    let lower: f32 = kani::any();
    kani::assume(lower.is_finite());

    let records = vec![make_record(vec![(lower, f32::INFINITY)])];
    assert!(
        tightest_enclosing_interval(&records).is_none(),
        "Inf upper must be excluded"
    );
}

/// Prove: when exactly one finite pair exists among non-finite pairs, the
/// result equals that finite pair.
#[kani::unwind(128)]
#[kani::proof]
fn tightest_enclosing_interval_finite_among_non_finite() {
    let lo: f32 = kani::any();
    let hi: f32 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());

    let records = vec![make_record(vec![
        (f32::NAN, 1.0),          // skipped: NaN lower
        (lo, hi),                 // the only finite pair
        (0.0, f32::NEG_INFINITY), // skipped: -Inf upper
    ])];

    let result = tightest_enclosing_interval(&records);
    assert!(result.is_some(), "must find the one finite pair");
    let (rlo, rhi) = result.unwrap();
    assert!(rlo == lo, "lower must match the single finite pair");
    assert!(rhi == hi, "upper must match the single finite pair");
}

/// Prove: result of `tightest_enclosing_interval` is always finite when Some.
///
/// For any two finite bound pairs, the enclosing interval's lo and hi
/// must both be finite (no overflow to Inf from f32::min/max).
#[kani::unwind(128)]
#[kani::proof]
fn tightest_enclosing_interval_result_is_finite() {
    let lo1: f32 = kani::any();
    let hi1: f32 = kani::any();
    let lo2: f32 = kani::any();
    let hi2: f32 = kani::any();
    kani::assume(lo1.is_finite() && hi1.is_finite());
    kani::assume(lo2.is_finite() && hi2.is_finite());

    let records = vec![make_record(vec![(lo1, hi1), (lo2, hi2)])];
    let result = tightest_enclosing_interval(&records);
    assert!(result.is_some());
    let (rlo, rhi) = result.unwrap();
    assert!(rlo.is_finite(), "enclosing lower must be finite");
    assert!(rhi.is_finite(), "enclosing upper must be finite");
}

// ---------------------------------------------------------------------------
// tightest_enclosing_interval: empty input
// ---------------------------------------------------------------------------

/// Prove: empty input always returns None.
#[kani::unwind(1)]
#[kani::proof]
fn tightest_enclosing_interval_empty_returns_none() {
    let result = tightest_enclosing_interval(&[]);
    assert!(result.is_none(), "empty input must return None");
}

/// Prove: records with only empty output_bounds return None.
#[kani::unwind(128)]
#[kani::proof]
fn tightest_enclosing_interval_empty_bounds_returns_none() {
    let records = vec![make_record(vec![])];
    assert!(
        tightest_enclosing_interval(&records).is_none(),
        "empty bounds must return None"
    );
}

// ---------------------------------------------------------------------------
// tightest_enclosing_interval: monotonicity
// ---------------------------------------------------------------------------

/// Prove: for any two finite bound pairs, the enclosing interval contains
/// both pairs. That is, result_lo <= min(lo1, lo2) and result_hi >= max(hi1, hi2).
#[kani::unwind(128)]
#[kani::proof]
fn tightest_enclosing_interval_contains_all_pairs() {
    let lo1: f32 = kani::any();
    let hi1: f32 = kani::any();
    let lo2: f32 = kani::any();
    let hi2: f32 = kani::any();
    kani::assume(lo1.is_finite() && hi1.is_finite());
    kani::assume(lo2.is_finite() && hi2.is_finite());

    let records = vec![make_record(vec![(lo1, hi1), (lo2, hi2)])];
    let (rlo, rhi) = tightest_enclosing_interval(&records).unwrap();

    // The enclosing lower must be <= both input lowers.
    assert!(rlo <= lo1, "enclosing lower must be <= lo1");
    assert!(rlo <= lo2, "enclosing lower must be <= lo2");
    // The enclosing upper must be >= both input uppers.
    assert!(rhi >= hi1, "enclosing upper must be >= hi1");
    assert!(rhi >= hi2, "enclosing upper must be >= hi2");
}

// ---------------------------------------------------------------------------
// tightest_enclosing_interval: cross-record containment (3 symbolic pairs)
// ---------------------------------------------------------------------------

/// Prove: containment holds across MULTIPLE LayerBoundRecords, not just
/// multiple pairs within one record. Three symbolic pairs split across
/// two records -- the enclosing interval must contain all three.
#[kani::unwind(128)]
#[kani::proof]
fn tightest_enclosing_interval_cross_record_containment() {
    let lo1: f32 = kani::any();
    let hi1: f32 = kani::any();
    let lo2: f32 = kani::any();
    let hi2: f32 = kani::any();
    let lo3: f32 = kani::any();
    let hi3: f32 = kani::any();
    kani::assume(lo1.is_finite() && hi1.is_finite());
    kani::assume(lo2.is_finite() && hi2.is_finite());
    kani::assume(lo3.is_finite() && hi3.is_finite());

    // Two separate records -- the function must merge across records.
    let records = vec![
        make_record(vec![(lo1, hi1)]),
        make_record(vec![(lo2, hi2), (lo3, hi3)]),
    ];
    let (rlo, rhi) = tightest_enclosing_interval(&records).unwrap();

    // Enclosing interval contains every input pair.
    assert!(rlo <= lo1, "must contain pair 1 lower");
    assert!(rlo <= lo2, "must contain pair 2 lower");
    assert!(rlo <= lo3, "must contain pair 3 lower");
    assert!(rhi >= hi1, "must contain pair 1 upper");
    assert!(rhi >= hi2, "must contain pair 2 upper");
    assert!(rhi >= hi3, "must contain pair 3 upper");
}

/// Prove: tightness -- the enclosing interval's lower EQUALS the minimum of
/// all input lowers, and the upper EQUALS the maximum of all input uppers.
/// This proves the interval is the TIGHTEST possible, not just sound.
#[kani::unwind(128)]
#[kani::proof]
fn tightest_enclosing_interval_is_tight() {
    let lo1: f32 = kani::any();
    let hi1: f32 = kani::any();
    let lo2: f32 = kani::any();
    let hi2: f32 = kani::any();
    kani::assume(lo1.is_finite() && hi1.is_finite());
    kani::assume(lo2.is_finite() && hi2.is_finite());

    let records = vec![make_record(vec![(lo1, hi1), (lo2, hi2)])];
    let (rlo, rhi) = tightest_enclosing_interval(&records).unwrap();

    // Tightness: result must be EXACTLY min/max, not wider.
    let expected_lo = lo1.min(lo2);
    let expected_hi = hi1.max(hi2);
    assert!(rlo == expected_lo, "lower must be exactly min of inputs");
    assert!(rhi == expected_hi, "upper must be exactly max of inputs");
}

/// Prove monotonicity: adding a wider pair to the input cannot SHRINK the
/// enclosing interval. If we compute with 1 pair then with 2 pairs, the
/// 2-pair interval must be at least as wide.
#[kani::unwind(128)]
#[kani::proof]
fn tightest_enclosing_interval_monotone_widening() {
    let lo1: f32 = kani::any();
    let hi1: f32 = kani::any();
    let lo2: f32 = kani::any();
    let hi2: f32 = kani::any();
    kani::assume(lo1.is_finite() && hi1.is_finite());
    kani::assume(lo2.is_finite() && hi2.is_finite());

    // Compute interval with just pair 1.
    let records_single = vec![make_record(vec![(lo1, hi1)])];
    let (single_lo, single_hi) = tightest_enclosing_interval(&records_single).unwrap();

    // Compute interval with both pairs.
    let records_both = vec![make_record(vec![(lo1, hi1), (lo2, hi2)])];
    let (both_lo, both_hi) = tightest_enclosing_interval(&records_both).unwrap();

    // Monotonicity: adding a pair cannot shrink the interval.
    assert!(
        both_lo <= single_lo,
        "adding pair must not raise lower bound"
    );
    assert!(
        both_hi >= single_hi,
        "adding pair must not lower upper bound"
    );
}

/// Prove: NaN in either position of a bound pair causes that pair to be
/// skipped. If ALL pairs contain NaN, the result is None.
/// This is the defense-in-depth property: NaN must never leak into the
/// enclosing interval.
#[kani::unwind(128)]
#[kani::proof]
fn tightest_enclosing_interval_nan_both_positions() {
    let val: f32 = kani::any();
    kani::assume(val.is_finite());

    // NaN in lower only
    let r1 = vec![make_record(vec![(f32::NAN, val)])];
    assert!(tightest_enclosing_interval(&r1).is_none());

    // NaN in upper only
    let r2 = vec![make_record(vec![(val, f32::NAN)])];
    assert!(tightest_enclosing_interval(&r2).is_none());

    // NaN in both positions
    let r3 = vec![make_record(vec![(f32::NAN, f32::NAN)])];
    assert!(tightest_enclosing_interval(&r3).is_none());
}

// ---------------------------------------------------------------------------
// derive_fusion_bounds: fallback logic and soundness
// ---------------------------------------------------------------------------

/// Prove: when layer_bounds is None, derive_fusion_bounds uses the fallback
/// path and the result interval always contains [-3.0, 3.0].
///
/// This tests the DANGEROUS hardcoded fallback: `lo.min(-3.0), hi.max(3.0)`.
/// The fallback guarantees the interval is at least [-3.0, 3.0] wide,
/// which could mask real narrower bounds. This harness proves the clamp
/// fires exactly when layer_bounds is absent.
#[kani::unwind(1)]
#[kani::proof]
fn derive_fusion_bounds_none_layer_bounds_always_includes_default_range() {
    // Use a concrete 1-element BoundedTensor (Kani can't model ndarray symbolically).
    // The symbolic part is that layer_bounds = None, testing the fallback path.
    let lower = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1]), -1.0f32);
    let upper = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1]), 1.0f32);
    let input_bounds = ny_api::BoundedTensor::new(lower, upper).unwrap();

    let result = derive_fusion_bounds(&None, &input_bounds);

    assert_eq!(
        result.len(),
        1,
        "fallback must produce exactly one interval"
    );
    let (lo, hi) = result[0];
    // Fallback clamps: lo.min(-3.0) = min(-1.0, -3.0) = -3.0
    //                  hi.max(3.0) = max(1.0, 3.0) = 3.0
    assert!(lo <= -3.0, "fallback lower must be <= -3.0");
    assert!(hi >= 3.0, "fallback upper must be >= 3.0");
    assert!(lo.is_finite(), "fallback lower must be finite");
    assert!(hi.is_finite(), "fallback upper must be finite");
}

/// Prove: when layer_bounds has valid finite bounds, the derive_fusion_bounds
/// output exactly matches tightest_enclosing_interval -- no fallback clamp.
///
/// This is the SOUNDNESS property: with real layer bounds, the result must
/// faithfully reflect the verified bounds, not the wider hardcoded range.
#[kani::unwind(128)]
#[kani::proof]
fn derive_fusion_bounds_with_valid_bounds_matches_tightest() {
    let lo: f32 = kani::any();
    let hi: f32 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());

    let lb = vec![make_record(vec![(lo, hi)])];
    let lower = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1]), -10.0f32);
    let upper = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1]), 10.0f32);
    let input_bounds = ny_api::BoundedTensor::new(lower, upper).unwrap();

    let result = derive_fusion_bounds(&Some(lb), &input_bounds);
    assert_eq!(result.len(), 1);

    // With valid layer bounds, derive_fusion_bounds must use tightest_enclosing_interval,
    // NOT the fallback. The result should be (lo, hi), not clamped to (-3, 3).
    let (rlo, rhi) = result[0];
    assert!(rlo == lo, "must use layer bound lower, not fallback");
    assert!(rhi == hi, "must use layer bound upper, not fallback");
}

/// Prove: when layer_bounds is Some but all bounds are non-finite,
/// derive_fusion_bounds falls back to the input bounds + clamp path.
/// This covers the tightest_enclosing_interval -> None -> fallback chain.
#[kani::unwind(128)]
#[kani::proof]
fn derive_fusion_bounds_all_non_finite_triggers_fallback() {
    let lb = vec![make_record(vec![
        (f32::NAN, 1.0),
        (f32::NEG_INFINITY, f32::INFINITY),
    ])];
    let lower = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1]), -5.0f32);
    let upper = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1]), 5.0f32);
    let input_bounds = ny_api::BoundedTensor::new(lower, upper).unwrap();

    let result = derive_fusion_bounds(&Some(lb), &input_bounds);
    assert_eq!(result.len(), 1);

    let (rlo, rhi) = result[0];
    // Fallback: lo.min(-3.0) = min(-5.0, -3.0) = -5.0
    //           hi.max(3.0) = max(5.0, 3.0) = 5.0
    assert!(rlo == -5.0, "fallback must use input lower (wider than -3)");
    assert!(rhi == 5.0, "fallback must use input upper (wider than 3)");
}

/// Prove: derive_fusion_bounds fallback with narrow input bounds.
/// When input range is narrower than [-3, 3], the clamp widens it.
/// This is the specific DANGEROUS behavior -- the function injects width
/// that is not justified by any verification.
#[kani::unwind(1)]
#[kani::proof]
fn derive_fusion_bounds_fallback_widens_narrow_inputs() {
    // Input bounds [-0.1, 0.1] -- very narrow.
    let lower = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1]), -0.1f32);
    let upper = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1]), 0.1f32);
    let input_bounds = ny_api::BoundedTensor::new(lower, upper).unwrap();

    let result = derive_fusion_bounds(&None, &input_bounds);
    let (rlo, rhi) = result[0];

    // The fallback WIDENS: lo.min(-3.0) = -3.0, hi.max(3.0) = 3.0
    // This is 30x wider than the actual input range!
    assert!(rlo == -3.0, "clamp must widen lower to -3.0");
    assert!(rhi == 3.0, "clamp must widen upper to 3.0");

    // Prove the widening ratio is at least 10x (a real bug detector).
    let input_width: f32 = 0.2; // 0.1 - (-0.1)
    let output_width: f32 = rhi - rlo;
    assert!(
        output_width > input_width * 10.0,
        "fallback must significantly widen narrow inputs"
    );
}

// ---------------------------------------------------------------------------
// classify_graph: summary consistency
// ---------------------------------------------------------------------------

/// Helper to make a simple TraceNode.
fn make_node(id: u64, name: &str, op: TraceOp) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        op,
        if id == 0 { vec![] } else { vec![id - 1] },
        vec![4],
        DType::F32,
    )
}

/// Prove: empty graph produces all-zero summary.
#[kani::unwind(8)]
#[kani::proof]
fn classify_graph_empty_produces_zero_summary() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let summary = classify_graph(&graph);

    assert_eq!(summary.verifiable, 0);
    assert_eq!(summary.bounded, 0);
    assert_eq!(summary.shape_only, 0);
    assert_eq!(summary.passthrough, 0);
    assert_eq!(summary.unverifiable_safe, 0);
    assert_eq!(summary.unverifiable_learned, 0);
    assert!(summary.unverifiable_learned_ops.is_empty());
    assert!(summary.is_fully_compilable());
}

/// Prove: for a graph with one op of each classification category, the
/// summary counts are consistent -- sum of all categories == total node count.
///
/// This catches the real bug where a new VerifiabilityClass variant is added
/// but the match arm in classify_graph falls through to the catch-all,
/// making the counts inconsistent.
#[kani::unwind(128)]
#[kani::proof]
fn classify_graph_summary_counts_are_consistent() {
    // Build a graph with one op per classification category.
    let nodes = vec![
        // ShapeOnly: Input
        make_node(0, "input", TraceOp::Input),
        // Verifiable: Relu
        make_node(1, "relu", TraceOp::Relu),
        // VerifiableBounded: LayerNorm
        make_node(
            2,
            "ln",
            TraceOp::LayerNorm {
                eps: 1e-5,
                weight: WeightRef::from_shape(&[4]),
                bias: WeightRef::from_shape(&[4]),
            },
        ),
        // Passthrough: Dropout
        make_node(3, "drop", TraceOp::Dropout),
        // UnverifiableSafe: Fract
        make_node(4, "fract", TraceOp::Fract),
        // UnverifiableLearned: Custom
        make_node(
            5,
            "custom",
            TraceOp::Custom {
                name: "mystery_op".to_string(),
            },
        ),
    ];

    let graph = ComputationGraph::from_nodes(nodes);
    let summary = classify_graph(&graph);

    let total = summary.verifiable
        + summary.bounded
        + summary.shape_only
        + summary.passthrough
        + summary.unverifiable_safe
        + summary.unverifiable_learned;

    assert_eq!(total, 6, "sum of all categories must equal node count");

    // Each category should have exactly 1.
    assert_eq!(summary.shape_only, 1, "Input -> ShapeOnly");
    assert_eq!(summary.verifiable, 1, "Relu -> Verifiable");
    assert_eq!(summary.bounded, 1, "LayerNorm -> VerifiableBounded");
    assert_eq!(summary.passthrough, 1, "Dropout -> Passthrough");
    assert_eq!(summary.unverifiable_safe, 1, "Fract -> UnverifiableSafe");
    assert_eq!(
        summary.unverifiable_learned, 1,
        "Custom -> UnverifiableLearned"
    );

    // UnverifiableLearned ops list should contain our custom op.
    assert_eq!(summary.unverifiable_learned_ops.len(), 1);
    assert!(
        summary.unverifiable_learned_ops[0] == "mystery_op",
        "must record the custom op name"
    );

    // Not fully compilable because of the unverifiable learned op.
    assert!(!summary.is_fully_compilable());
}

/// Prove: a graph with ONLY verifiable and shape-only ops is fully compilable.
/// This is the common case -- a valid model should pass the compilation gate.
#[kani::unwind(128)]
#[kani::proof]
fn classify_graph_verifiable_graph_is_compilable() {
    let nodes = vec![
        make_node(0, "input", TraceOp::Input),
        make_node(1, "relu", TraceOp::Relu),
        make_node(2, "sigmoid", TraceOp::Sigmoid),
        make_node(3, "add", TraceOp::Add),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let summary = classify_graph(&graph);

    assert!(
        summary.is_fully_compilable(),
        "all-verifiable graph must be compilable"
    );
    assert_eq!(summary.unverifiable_learned, 0);
    assert_eq!(summary.unverifiable_learned_ops.len(), 0);

    let total = summary.verifiable
        + summary.bounded
        + summary.shape_only
        + summary.passthrough
        + summary.unverifiable_safe
        + summary.unverifiable_learned;
    assert_eq!(total, 4, "total must match node count");
}

/// Prove: classify_graph deduplicates unverifiable_learned_ops names.
/// If two Custom ops share the same name, the list should contain it once.
#[kani::unwind(128)]
#[kani::proof]
fn classify_graph_deduplicates_learned_ops() {
    let nodes = vec![
        make_node(0, "input", TraceOp::Input),
        make_node(
            1,
            "custom_a",
            TraceOp::Custom {
                name: "shared_name".to_string(),
            },
        ),
        make_node(
            2,
            "custom_b",
            TraceOp::Custom {
                name: "shared_name".to_string(),
            },
        ),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let summary = classify_graph(&graph);

    assert_eq!(
        summary.unverifiable_learned, 2,
        "count is per-node, not per-name"
    );
    assert_eq!(
        summary.unverifiable_learned_ops.len(),
        1,
        "names must be deduplicated (sort+dedup)"
    );
    assert!(summary.unverifiable_learned_ops[0] == "shared_name");
}
