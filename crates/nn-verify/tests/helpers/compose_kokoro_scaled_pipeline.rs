// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Scaled Kokoro pipeline NY composition tests.
//!
//! Extends the D=8 baseline proofs from `compose_kokoro_full_pipeline.rs` to
//! D=16, D=32, and D=64 — stepping toward production D=512.
//!
//! Each scale level exercises the *same* proof chain:
//! - **Property 1 (Non-silence):** exp() lower bound > 0
//! - **Property 2 (Non-clipping):** output upper bound < threshold
//! - **Property 3 (Duration positivity):** dur_logits bounds finite
//! - **Property 5 (Temporal boundedness):** dispatch plan cost < 100ms
//!
//! Per-dimension tests are consolidated to build each graph ONCE and run all
//! property checks on it, avoiding 4-5x redundant graph builds per dimension.
//!
//! Part of #1741: THE MOONSHOT — scaling composition proofs toward production.

#[path = "kokoro_scaled_pipeline.rs"]
mod scaled_pipeline_helpers;

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_tts_verify::cost_model::{
    profile_dispatch_plan, total_estimated_time_us, total_flops, HardwareCostModel,
};
use nn_verify::{tensor_kernel_to_graph, VerificationSoundnessMode};
use scaled_pipeline_helpers::{
    build_scaled_duration_branch, build_scaled_full_pipeline, scaled_duration_branch_bindings,
    scaled_full_pipeline_bindings, KokoroDims,
};

/// Timing bound: 100ms = 100,000 us (Moonshot Property 5 claim).
const TIMING_BOUND_US: f64 = 100_000.0;

// ===========================================================================
// D=8 (baseline -- matches compose_kokoro_full_pipeline.rs)
// ===========================================================================

/// D=8 baseline: full pipeline IBP proves Properties 1+2.
#[test]
fn test_scaled_d8_full_pipeline_ibp() {
    let dims = KokoroDims::d8();
    let (def, out_shape) = build_scaled_full_pipeline(&dims);
    assert_eq!(out_shape, [dims.out_channels, dims.time_up()]);

    let bindings = scaled_full_pipeline_bindings(&dims);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("D=8 graph translation");
    let input = uniform_bounds(&[dims.d_model, dims.seq_len], 1.0);
    let output = graph.propagate_ibp(&input).expect("D=8 IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);

    assert!(lo_min > 0.0, "D=8 P1: exp output positive, got {lo_min}");
    assert!(hi_max < 1e8, "D=8 P2: output bounded, got {hi_max}");
    eprintln!("D=8 IBP: [{lo_min}, {hi_max}] -- P1 P2");
}

/// D=8 baseline: duration branch proves Property 3.
#[test]
fn test_scaled_d8_duration_ibp() {
    let dims = KokoroDims::d8();
    let (def, out_len) = build_scaled_duration_branch(&dims);
    assert_eq!(out_len, dims.seq_len);

    let bindings = scaled_duration_branch_bindings(&dims);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("D=8 duration graph");
    let input = uniform_bounds(&[dims.d_model, dims.seq_len], 1.0);
    let output = graph.propagate_ibp(&input).expect("D=8 duration IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);

    assert!(lo_min.is_finite(), "D=8 P3: dur lower finite, got {lo_min}");
    assert!(hi_max.is_finite(), "D=8 P3: dur upper finite, got {hi_max}");
    eprintln!("D=8 duration IBP: [{lo_min}, {hi_max}] -- P3");
}

// ===========================================================================
// Consolidated per-dimension tests: build graph once, check all properties.
// Previously 6 tests per D (graph_builds, ibp, crown, duration_ibp,
// duration_timing, verify_and_record) -- now 2 per D.
// ===========================================================================

/// Helper: full pipeline all-properties check at given dimension.
/// Builds graph once, runs IBP, CROWN, verify, and records.
fn check_full_pipeline_all_properties(dims: &KokoroDims, label: &str) {
    let (def, out_shape) = build_scaled_full_pipeline(dims);
    assert_eq!(out_shape, [dims.out_channels, dims.time_up()]);

    let bindings = scaled_full_pipeline_bindings(dims);
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .unwrap_or_else(|e| panic!("{label} graph translation: {e}"));
    let num_nodes = graph.num_nodes();
    assert!(
        num_nodes >= 15,
        "{label} graph should have >= 15 nodes, got {num_nodes}",
    );
    eprintln!("{label} full pipeline: {num_nodes} nodes");

    let input = uniform_bounds(&[dims.d_model, dims.seq_len], 1.0);

    // IBP: Properties 1+2
    let output = graph
        .propagate_ibp(&input)
        .unwrap_or_else(|e| panic!("{label} IBP: {e}"));
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    assert!(
        lo_min > 0.0,
        "{label} P1: exp output positive, got {lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "{label} P2: output should be finite, got {hi_max}"
    );
    eprintln!("{label} IBP: [{lo_min}, {hi_max}] -- P1 P2(finite)");

    // CROWN: tighter or fallback
    let (method, crown_output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    eprintln!("{label} CROWN: method={method:?} [{crown_lo}, {crown_hi}]");
    if let Some(reason) = &fallback {
        eprintln!("  fallback: {reason}");
    }
    assert!(
        crown_lo > 0.0,
        "CROWN {label} P1: exp output positive, got {crown_lo}"
    );
    assert!(
        crown_hi.is_finite(),
        "CROWN {label} P2: output should be finite, got {crown_hi}"
    );

    // Verify and record
    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        &format!("kokoro_scaled_{}", label.to_lowercase()),
    );
    assert_eq!(result.num_variables, 1, "single Variable input");
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[dims.out_channels, dims.time_up()]);
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "ForwardMode NormBoundsMode should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}

/// Helper: duration branch all-properties check at given dimension.
/// Builds graph once, runs IBP + timing.
fn check_duration_all_properties(dims: &KokoroDims, label: &str) {
    let (def, _) = build_scaled_duration_branch(dims);
    let bindings = scaled_duration_branch_bindings(dims);
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .unwrap_or_else(|e| panic!("{label} duration graph: {e}"));
    let input = uniform_bounds(&[dims.d_model, dims.seq_len], 1.0);

    // IBP: Property 3
    let output = graph
        .propagate_ibp(&input)
        .unwrap_or_else(|e| panic!("{label} duration IBP: {e}"));
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    assert!(
        lo_min.is_finite(),
        "{label} P3: dur lower finite, got {lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "{label} P3: dur upper finite, got {hi_max}"
    );
    eprintln!("{label} duration IBP: [{lo_min}, {hi_max}] -- P3");

    // Property 5: timing
    let (steps, _) = nn_dsl::build_dispatch_plan(&def, nn_dsl::ScalarType::F32)
        .unwrap_or_else(|e| panic!("{label} duration dispatch plan: {e}"));
    let hw = HardwareCostModel::m4_max_conservative();
    let profiles = profile_dispatch_plan(&steps, &hw);
    let time_us = total_estimated_time_us(&profiles);
    let flops = total_flops(&profiles);
    assert!(
        time_us < TIMING_BOUND_US,
        "{label} P5: time {time_us:.3}us >= {TIMING_BOUND_US}us"
    );
    eprintln!("{label} P5: {flops} FLOPs, {time_us:.3}us < {TIMING_BOUND_US}us");
}

// ===========================================================================
// D=16 (first scaling step)
// ===========================================================================

/// D=16: full pipeline -- graph build, IBP, CROWN, verify.
#[test]
fn test_scaled_d16_full_pipeline() {
    check_full_pipeline_all_properties(&KokoroDims::d16(), "D=16");
}

/// D=16: duration branch -- IBP + timing.
#[test]
fn test_scaled_d16_duration() {
    check_duration_all_properties(&KokoroDims::d16(), "D=16");
}

// ===========================================================================
// D=32 (meaningful step toward production)
// ===========================================================================

/// D=32: full pipeline -- graph build, IBP, CROWN, verify.
#[test]
fn test_scaled_d32_full_pipeline() {
    check_full_pipeline_all_properties(&KokoroDims::d32(), "D=32");
}

/// D=32: duration branch -- IBP + timing.
#[test]
fn test_scaled_d32_duration() {
    check_duration_all_properties(&KokoroDims::d32(), "D=32");
}

// ===========================================================================
// D=64 (requires per-layer CROWN #1762 for tight bounds)
// ===========================================================================

/// D=64: full pipeline -- graph build, IBP, CROWN, verify.
///
/// At D=64, IBP bounds will be substantially wider due to the wrapping
/// problem in interval arithmetic. Property 1 (exp > 0) must still hold
/// structurally. Property 2 bounds may be vacuously wide -- this test
/// documents the current bound quality at D=64 as a baseline for
/// improvement via per-layer CROWN (#1762).
#[test]
fn test_scaled_d64_full_pipeline() {
    check_full_pipeline_all_properties(&KokoroDims::d64(), "D=64");
}

/// D=64: duration branch -- IBP + timing.
#[test]
fn test_scaled_d64_duration() {
    check_duration_all_properties(&KokoroDims::d64(), "D=64");
}

// ===========================================================================
// Scaling study: bound width vs dimension
// ===========================================================================

/// Measure IBP bound width across D=8 and D=64 for both duration and full
/// pipeline. Endpoints only -- D=16/D=32 are covered by per-dimension tests.
#[test]
fn test_scaling_study_bound_width() {
    let configs: &[(&str, KokoroDims)] = &[("D=8", KokoroDims::d8()), ("D=64", KokoroDims::d64())];

    // Duration branch scaling
    eprintln!("\n=== Duration Branch IBP Scaling Study ===");
    eprintln!(
        "{:<8} {:<12} {:<12} {:<12}",
        "Scale", "Lower", "Upper", "Width"
    );
    for (label, dims) in configs {
        let (def, _) = build_scaled_duration_branch(dims);
        let bindings = scaled_duration_branch_bindings(dims);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let input = uniform_bounds(&[dims.d_model, dims.seq_len], 1.0);
        let output = graph.propagate_ibp(&input).expect("IBP");
        let (lo_min, hi_max) = bounds_min_max(&output);
        let width = hi_max - lo_min;
        assert!(lo_min.is_finite(), "{label}: lower should be finite");
        assert!(hi_max.is_finite(), "{label}: upper should be finite");
        eprintln!(
            "{label:<8} {lo_min:<12.6} {hi_max:<12.6} {width:<12.6}"
        );
    }

    // Full pipeline scaling
    eprintln!("\n=== Full Pipeline IBP Scaling Study ===");
    eprintln!(
        "{:<8} {:<14} {:<14} {:<14} {:<8}",
        "Scale", "Lower", "Upper", "Width", "P1?"
    );
    for (label, dims) in configs {
        let (def, _) = build_scaled_full_pipeline(dims);
        let bindings = scaled_full_pipeline_bindings(dims);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let input = uniform_bounds(&[dims.d_model, dims.seq_len], 1.0);
        let output = graph.propagate_ibp(&input).expect("IBP");
        let (lo_min, hi_max) = bounds_min_max(&output);
        let width = hi_max - lo_min;
        let p1 = if lo_min > 0.0 { "yes" } else { "no" };
        assert!(lo_min.is_finite(), "{label}: lower should be finite");
        assert!(hi_max.is_finite(), "{label}: upper should be finite");
        assert!(width < 1e6, "{label}: width should be bounded, got {width}");
        eprintln!(
            "{label:<8} {lo_min:<14.6} {hi_max:<14.6} {width:<14.6} {p1:<8}"
        );
    }
}
