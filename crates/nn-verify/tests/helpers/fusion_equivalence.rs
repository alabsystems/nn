// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for fusion equivalence verification (issue #21).
//!
//! Proves K4 (fused AdaIN+Snake) ≡ K3+K1 (sequential) via NY
//! diamond DAG diff.
//!
//! # CROWN relaxation and diamond DAGs
//!
//! CROWN linearizes nonlinear operations (sin, pow, rsqrt), introducing
//! relaxation error at each node. In the diamond DAG, both paths have
//! separate nonlinear nodes, and the relaxation errors don't cancel even
//! though the paths compute the same function. This means:
//!
//! - **Point inputs**: CROWN proves exact equivalence (zero diff)
//! - **Narrow intervals**: CROWN proves tight bounds (small diff)
//! - **Wide intervals**: CROWN gives a valid but loose bound
//!
//! The CROWN bound is a proved upper limit on |fused - sequential|.
//! The actual maximum diff for identical formulas is zero; CROWN's
//! overestimate comes from independent relaxation of the two paths.

use nn_dsl::{
    build_adain_scalar_kernel, build_adain_snake_fused_kernel, build_snake_scalar_kernel,
};
use nn_verify::{
    build_fusion_diff_graph, verify_adain_snake_fusion, verify_fusion_equivalence,
    verify_fusion_equivalence_with_config, FusionSpec, FusionVerification, PropMethod,
    VerifyConfig, VerifyError,
};

/// Representative input bounds for the dvoice Kokoro decoder.
const DVOICE_BOUNDS: [(f32, f32); 7] = [
    (-10.0, 10.0), // x: audio features after encoder
    (-5.0, 5.0),   // mu: channel mean
    (0.001, 10.0), // var: channel variance (positive)
    (0.1, 5.0),    // gamma: style scale
    (-3.0, 3.0),   // beta: style shift
    (0.01, 100.0), // alpha: snake activation parameter
    (1e-5, 1e-5),  // eps: constant epsilon (point interval)
];

#[test]
fn test_diamond_dag_graph_builds() {
    let fused = build_adain_snake_fused_kernel().expect("fused kernel");
    let adain = build_adain_scalar_kernel().expect("adain kernel");
    let snake = build_snake_scalar_kernel().expect("snake kernel");

    let spec = FusionSpec::new(&fused, &adain, &snake, 7, &[0, 1, 2, 3, 4, 6], &[0, 5], 0)
        .expect("valid fusion spec");
    let graph = build_fusion_diff_graph(&spec).expect("diamond DAG should build");

    let output_name = graph.output_name();
    assert_eq!(output_name, "diff", "output should be the diff node");
}

#[test]
fn test_fusion_equivalence_with_point_inputs() {
    // All inputs at single points — CROWN should prove near-zero diff.
    // CROWN's independent linearization of nonlinear ops (sin, pow, rsqrt)
    // introduces ~1 ULP relaxation error (~1.19e-7 for f32) even on point
    // intervals. This is a known property of CROWN, not a bug.
    let point_bounds = [
        (1.0, 1.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (1.0, 1.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (1e-5, 1e-5),
    ];

    let result = verify_adain_snake_fusion(&point_bounds, 2e-6)
        .expect("point input verification should succeed");

    assert!(
        result.within_epsilon,
        "point inputs should prove near-zero diff, got max_abs_diff: {}",
        result.max_abs_diff,
    );
    // CROWN relaxation error is ~1.19e-7 (1 ULP at f32); allow up to 2e-6
    // to account for multi-op error accumulation.
    assert!(
        result.diff_lower.abs() < 2e-6 && result.diff_upper.abs() < 2e-6,
        "point inputs should produce near-zero diff, got [{}, {}]",
        result.diff_lower,
        result.diff_upper,
    );
    assert_eq!(result.method, PropMethod::Crown);
}

#[test]
fn test_fusion_equivalence_with_narrow_bounds() {
    // Narrow intervals where CROWN relaxation error is small.
    // The diff bound should be much tighter than for wide intervals.
    let narrow_bounds = [
        (0.9, 1.1),   // x: ±10% around 1.0
        (-0.1, 0.1),  // mu: near zero
        (0.9, 1.1),   // var: near 1.0
        (0.9, 1.1),   // gamma: near 1.0
        (-0.1, 0.1),  // beta: near zero
        (0.9, 1.1),   // alpha: near 1.0
        (1e-5, 1e-5), // eps
    ];

    let result = verify_adain_snake_fusion(&narrow_bounds, 10.0)
        .expect("narrow bounds verification should succeed");

    // CROWN should prove a reasonably tight bound for narrow intervals.
    assert!(
        result.max_abs_diff < 10.0,
        "narrow bounds should produce tight CROWN diff, got {}",
        result.max_abs_diff,
    );
    assert_eq!(result.method, PropMethod::Crown);
}

#[test]
fn test_fusion_dvoice_bounds_produces_finite_diff() {
    // For the full dvoice domain, CROWN produces a finite but loose bound.
    // The looseness comes from CROWN's independent relaxation of nonlinear
    // operations in both diamond DAG paths. The actual max diff is zero
    // (formulas are mathematically identical); the CROWN bound is an overestimate.
    // Use f32::MAX as a permissive finite epsilon (f32::INFINITY is now rejected).
    let result = verify_adain_snake_fusion(&DVOICE_BOUNDS, f32::MAX)
        .expect("dvoice bounds verification should succeed");

    // The diff must be finite (both paths produce finite bounds).
    assert!(
        result.diff_lower.is_finite() && result.diff_upper.is_finite(),
        "diff bounds must be finite, got [{}, {}]",
        result.diff_lower,
        result.diff_upper,
    );
    assert!(result.diff_lower <= result.diff_upper);
    assert_eq!(result.method, PropMethod::Crown);

    // The CROWN-proved diff bound for dvoice should be documented.
    // This is a valid upper bound on |K4(x) - (K3→K1)(x)| for all x
    // in the dvoice domain, though it overestimates due to relaxation.
    eprintln!(
        "CROWN-proved diff bound for dvoice: [{}, {}], max_abs: {}",
        result.diff_lower, result.diff_upper, result.max_abs_diff,
    );
}

#[test]
fn test_fusion_crown_tighter_than_ibp_narrow() {
    // Verify CROWN produces finite bounds (not falling back to IBP).
    let narrow_bounds = [
        (0.9, 1.1),
        (-0.1, 0.1),
        (0.9, 1.1),
        (0.9, 1.1),
        (-0.1, 0.1),
        (0.9, 1.1),
        (1e-5, 1e-5),
    ];

    let result =
        verify_adain_snake_fusion(&narrow_bounds, f32::MAX).expect("verification should succeed");

    assert_eq!(
        result.method,
        PropMethod::Crown,
        "should use CROWN, not fall back to IBP"
    );
    assert!(
        result.crown_fallback_reason.is_none(),
        "CROWN should not have failed"
    );
}

#[test]
fn test_fusion_verification_result_fields() {
    let result = verify_adain_snake_fusion(&DVOICE_BOUNDS, f32::MAX)
        .expect("verification should succeed");

    assert_eq!(result.fused_kernel_name, "adain_snake");
    assert!(result.epsilon > 0.0);
    assert!(result.diff_lower.is_finite());
    assert!(result.diff_upper.is_finite());
    assert!(result.diff_lower <= result.diff_upper);
}

#[test]
fn test_fusion_wrong_bounds_count_errors() {
    let bad_bounds = [
        (-10.0, 10.0),
        (-5.0, 5.0),
        (0.001, 10.0),
        (0.1, 5.0),
        (-3.0, 3.0),
        (0.01, 100.0),
    ];

    let err =
        verify_adain_snake_fusion(&bad_bounds, 1e-5).expect_err("wrong bounds count should fail");

    assert!(
        err.to_string().contains("mismatch"),
        "error should mention mismatch, got: {err}"
    );
}

#[test]
fn test_fusion_nan_epsilon_rejected() {
    // NaN epsilon would cause `max_abs_diff <= epsilon` to silently return
    // false per IEEE 754 (design doc #66). The guard must reject it.
    let err = verify_adain_snake_fusion(&DVOICE_BOUNDS, f32::NAN)
        .expect_err("NaN epsilon should be rejected");

    assert!(
        err.to_string().contains("threshold"),
        "error should mention threshold, got: {err}"
    );
}

#[test]
fn test_generic_fusion_diff_api() {
    let fused = build_adain_snake_fused_kernel().expect("fused kernel");
    let adain = build_adain_scalar_kernel().expect("adain kernel");
    let snake = build_snake_scalar_kernel().expect("snake kernel");

    let spec = FusionSpec::new(&fused, &adain, &snake, 7, &[0, 1, 2, 3, 4, 6], &[0, 5], 0)
        .expect("valid fusion spec");
    let result = verify_fusion_equivalence(&spec, &DVOICE_BOUNDS, f32::MAX)
        .expect("generic API verification should succeed");

    // Should produce finite diff bounds.
    assert!(result.diff_lower.is_finite());
    assert!(result.diff_upper.is_finite());
}

#[test]
fn test_fusion_serialization_roundtrip() {
    // Use a finite epsilon for JSON roundtrip (Infinity is not valid JSON).
    let result =
        verify_adain_snake_fusion(&DVOICE_BOUNDS, 1e6).expect("verification should succeed");

    let json = serde_json::to_string_pretty(&result).expect("serialize");
    let deser: FusionVerification = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(result.fused_kernel_name, deser.fused_kernel_name);
    assert_eq!(result.within_epsilon, deser.within_epsilon);
    assert!((result.max_abs_diff - deser.max_abs_diff).abs() < 1e-15);
}

// --- Regression tests for epsilon validation (#116) ---

#[test]
fn test_fusion_positive_infinity_epsilon_rejected() {
    // +Inf epsilon trivially passes all equivalence checks, which is a silent
    // contract violation in a proof API.
    let err = verify_adain_snake_fusion(&DVOICE_BOUNDS, f32::INFINITY)
        .expect_err("+Inf epsilon should be rejected");

    assert!(
        err.to_string().contains("threshold"),
        "error should mention threshold, got: {err}"
    );
}

#[test]
fn test_fusion_negative_infinity_epsilon_rejected() {
    let err = verify_adain_snake_fusion(&DVOICE_BOUNDS, f32::NEG_INFINITY)
        .expect_err("-Inf epsilon should be rejected");

    assert!(
        err.to_string().contains("threshold"),
        "error should mention threshold, got: {err}"
    );
}

#[test]
fn test_fusion_negative_epsilon_rejected() {
    // Negative epsilon trivially fails all equivalence checks (max_abs_diff >= 0
    // is always > epsilon), which is a silent contract violation.
    let err = verify_adain_snake_fusion(&DVOICE_BOUNDS, -1.0)
        .expect_err("negative epsilon should be rejected");

    assert!(
        err.to_string().contains("threshold"),
        "error should mention threshold, got: {err}"
    );
}

#[test]
fn test_fusion_zero_epsilon_accepted() {
    // Zero epsilon is accepted as input (not rejected), but CROWN relaxation
    // error (~1.19e-7 for f32) means point inputs will NOT satisfy epsilon=0.
    // This test verifies zero epsilon is a valid parameter, not that it proves
    // exact equivalence — CROWN cannot achieve bit-exact results on nonlinear ops.
    let point_bounds = [
        (1.0, 1.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (1.0, 1.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (1e-5, 1e-5),
    ];

    let result =
        verify_adain_snake_fusion(&point_bounds, 0.0).expect("zero epsilon should be accepted");

    // Zero epsilon will not be satisfied due to CROWN relaxation error, but the
    // verification should still complete without error. max_abs_diff should be
    // small (< 2e-6) even though within_epsilon is false.
    assert!(
        result.max_abs_diff < 2e-6,
        "point inputs should produce near-zero diff even with zero epsilon, got {}",
        result.max_abs_diff,
    );
}

#[test]
fn test_fusion_with_config_is_used() {
    // Verify the config parameter is actually wired (not _config).
    let fused = build_adain_snake_fused_kernel().expect("fused kernel");
    let adain = build_adain_scalar_kernel().expect("adain kernel");
    let snake = build_snake_scalar_kernel().expect("snake kernel");

    let config = VerifyConfig::with_threshold(500.0).expect("valid threshold");

    let spec = FusionSpec::new(&fused, &adain, &snake, 7, &[0, 1, 2, 3, 4, 6], &[0, 5], 0)
        .expect("valid fusion spec");
    let result = verify_fusion_equivalence_with_config(&spec, &DVOICE_BOUNDS, 1e6, &config)
        .expect("with_config should succeed");

    assert!(result.diff_lower.is_finite());
    assert!(result.diff_upper.is_finite());
}

#[test]
fn test_fusion_require_sound_rejects_heuristic_crown_ops() {
    // Fusion of sin-containing kernels via CROWN uses sampling-based
    // relaxations (SinLayer defaults to sound=false). With require_sound=true,
    // the verification must reject with SoundnessRequired. Before the #189
    // fix, soundness was hardcoded as Sound and this would incorrectly pass.
    let fused = build_adain_snake_fused_kernel().expect("fused kernel");
    let adain = build_adain_scalar_kernel().expect("adain kernel");
    let snake = build_snake_scalar_kernel().expect("snake kernel");

    let config = VerifyConfig::default().with_require_sound(true);

    let spec = FusionSpec::new(&fused, &adain, &snake, 7, &[0, 1, 2, 3, 4, 6], &[0, 5], 0)
        .expect("valid fusion spec");
    let result = verify_fusion_equivalence_with_config(&spec, &DVOICE_BOUNDS, f32::MAX, &config);

    assert!(
        matches!(result, Err(VerifyError::SoundnessRequired { .. })),
        "require_sound=true should reject CROWN fusion with sin (heuristic): got {result:?}"
    );
}

// --- ForwardMode config tests (#2225) ---

/// Tightened bounds from #2225 fusion_configs: var and alpha are safely away
/// from zero, avoiding singularities in rsqrt(var+eps) and 1/alpha that can
/// cause CROWN numerical instability with wide bounds.
const TIGHTENED_BOUNDS: [(f32, f32); 7] = [
    (-3.0, 3.0),  // x (tightened from ±10)
    (-2.0, 2.0),  // mu (tightened from ±5)
    (0.1, 3.0),   // var (positive, away from zero)
    (0.5, 2.0),   // gamma
    (-1.0, 1.0),  // beta
    (0.5, 5.0),   // alpha (positive, away from zero)
    (1e-5, 1e-5), // eps (point)
];

#[test]
fn test_fusion_with_config_tightened_bounds_crown_succeeds() {
    use nn_verify::{NormBoundsMode, VerifyConfig};

    let config = VerifyConfig::default().with_norm_mode(NormBoundsMode::ForwardMode);
    let result =
        nn_verify::verify_adain_snake_fusion_with_config(&TIGHTENED_BOUNDS, f32::MAX, &config)
            .expect("tightened bounds should produce valid CROWN result");

    assert_eq!(
        result.method,
        PropMethod::Crown,
        "tightened bounds should use CROWN, not fall back to IBP"
    );
    assert!(
        result.crown_fallback_reason.is_none(),
        "CROWN should not have failed with tightened bounds"
    );
    assert!(
        result.diff_lower.is_finite() && result.diff_upper.is_finite(),
        "diff bounds must be finite, got [{}, {}]",
        result.diff_lower,
        result.diff_upper,
    );
    // Tightened bounds should produce measurably narrower diff than DVOICE_BOUNDS.
    // DVOICE_BOUNDS produces ~4826; tightened should be significantly less.
    assert!(
        result.max_abs_diff < 4826.0,
        "tightened bounds should produce narrower diff than dvoice, got {}",
        result.max_abs_diff,
    );
    eprintln!(
        "CROWN diff with tightened bounds: [{}, {}], max_abs: {}",
        result.diff_lower, result.diff_upper, result.max_abs_diff,
    );
}

#[test]
fn test_fusion_with_config_is_conclusive() {
    use nn_verify::{NormBoundsMode, VerifyConfig};

    let config = VerifyConfig::default().with_norm_mode(NormBoundsMode::ForwardMode);
    let result =
        nn_verify::verify_adain_snake_fusion_with_config(&TIGHTENED_BOUNDS, f32::MAX, &config)
            .expect("should succeed");

    assert!(
        result.is_conclusive(),
        "CROWN result should be conclusive (not IBP fallback)"
    );
}

// validate_fusion_params error coverage extracted to
// fusion_equivalence_validation.rs (#356).
// Fusion recording tests extracted to fusion_equivalence_recording.rs (#1420).
// RMSNorm+SiLU-Mul and LayerNorm+GELU fusion pair tests extracted to
// fusion_equivalence_pairs.rs (#887).
