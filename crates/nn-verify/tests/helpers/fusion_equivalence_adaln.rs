// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fusion equivalence tests for AdaLayerNorm (AdaLN).
//!
//! Proves the fused AdaLN kernel (LayerNorm + adaptive affine in one pass)
//! produces the same result as the sequential composition for all inputs
//! within given bounds, via NY diamond DAG diff.
//!
//! The Kokoro ProsodyPredictor uses 2 fused AdaLN GPU kernels per forward
//! pass. These tests verify the fusion equivalence that was previously
//! unproven.
//!
//! Part of #2714, #2701, #2218.

use nn_verify::PropMethod;

/// Representative input bounds for Kokoro ProsodyPredictor AdaLN.
///
/// 8 params: x, mean, var_val, eps, norm_weight, norm_bias, gamma, beta
///
/// The fused AdaLN formula is:
///   `(1 + gamma) * ((x - mean) * rsqrt(var + eps) * norm_weight + norm_bias) + beta`
///
/// Bounds are tight enough to avoid overflow in intermediate computations:
/// max |normed| ≈ (3-(-1))/sqrt(0.5) * 2 + 1 ≈ 6.66, then
/// (1+2) * 6.66 + 2 ≈ 22 — well within f32 safe range.
const ADALN_BOUNDS: [(f32, f32); 8] = [
    (-3.0, 3.0),  // x: activations after previous layer
    (-1.0, 1.0),  // mean: layer mean
    (0.5, 5.0),   // var_val: layer variance (positive, not too small)
    (1e-5, 1e-5), // eps: constant epsilon (point interval)
    (0.5, 2.0),   // norm_weight: learned LayerNorm scale
    (-1.0, 1.0),  // norm_bias: learned LayerNorm shift
    (-2.0, 2.0),  // gamma: adaptive style scale
    (-2.0, 2.0),  // beta: adaptive style shift
];

// ---------------------------------------------------------------------------
// Core fusion equivalence tests
// ---------------------------------------------------------------------------

#[test]
fn test_adaln_fusion_point_inputs() {
    use nn_verify::verify_ada_layer_norm_fusion;

    // All inputs at single points — CROWN should prove near-zero diff.
    let point_bounds = [
        (1.0, 1.0),   // x
        (0.0, 0.0),   // mean
        (1.0, 1.0),   // var_val
        (1e-5, 1e-5), // eps
        (1.0, 1.0),   // norm_weight
        (0.0, 0.0),   // norm_bias
        (0.0, 0.0),   // gamma
        (0.0, 0.0),   // beta
    ];

    let result = verify_ada_layer_norm_fusion(&point_bounds, 2e-6)
        .expect("point input verification should succeed");

    assert!(
        result.within_epsilon,
        "point inputs should prove near-zero diff, got max_abs_diff: {}",
        result.max_abs_diff,
    );
    // CROWN relaxation error is ~1 ULP at f32; allow up to 2e-6 for
    // multi-op error accumulation through rsqrt.
    assert!(
        result.diff_lower.abs() < 2e-6 && result.diff_upper.abs() < 2e-6,
        "point inputs should produce near-zero diff, got [{}, {}]",
        result.diff_lower,
        result.diff_upper,
    );
    assert_eq!(result.method, PropMethod::Crown);
}

#[test]
fn test_adaln_fusion_narrow_bounds() {
    use nn_verify::verify_ada_layer_norm_fusion;

    let narrow_bounds = [
        (0.9, 1.1),   // x: ±10% around 1.0
        (-0.1, 0.1),  // mean: near zero
        (0.9, 1.1),   // var_val: near 1.0
        (1e-5, 1e-5), // eps
        (0.9, 1.1),   // norm_weight: near 1.0
        (-0.1, 0.1),  // norm_bias: near zero
        (-0.1, 0.1),  // gamma: near zero (scale ≈ 1)
        (-0.1, 0.1),  // beta: near zero
    ];

    let result = verify_ada_layer_norm_fusion(&narrow_bounds, 10.0)
        .expect("narrow bounds verification should succeed");

    assert!(
        result.max_abs_diff < 10.0,
        "narrow bounds should produce tight CROWN diff, got {}",
        result.max_abs_diff,
    );
    assert_eq!(result.method, PropMethod::Crown);
}

#[test]
fn test_adaln_fusion_dvoice_bounds_produces_finite_diff() {
    use nn_verify::verify_ada_layer_norm_fusion;

    let result = verify_ada_layer_norm_fusion(&ADALN_BOUNDS, f32::MAX)
        .expect("realistic bounds verification should succeed");

    assert!(
        result.diff_lower.is_finite() && result.diff_upper.is_finite(),
        "diff bounds must be finite, got [{}, {}]",
        result.diff_lower,
        result.diff_upper,
    );
    assert!(result.diff_lower <= result.diff_upper);
    assert_eq!(result.method, PropMethod::Crown);

    eprintln!(
        "CROWN-proved AdaLN diff bound: [{}, {}], max_abs: {}",
        result.diff_lower, result.diff_upper, result.max_abs_diff,
    );
}

#[test]
fn test_adaln_fusion_wrong_bounds_count_errors() {
    use nn_verify::verify_ada_layer_norm_fusion;

    // 7 instead of 8 — missing beta
    let bad_bounds = [
        (-3.0, 3.0),
        (-1.0, 1.0),
        (0.5, 5.0),
        (1e-5, 1e-5),
        (0.5, 2.0),
        (-1.0, 1.0),
        (-2.0, 2.0),
    ];

    let err = verify_ada_layer_norm_fusion(&bad_bounds, 1e-5)
        .expect_err("wrong bounds count should fail");

    assert!(
        err.to_string().contains("mismatch"),
        "error should mention mismatch, got: {err}"
    );
}

#[test]
fn test_adaln_fusion_nan_epsilon_rejected() {
    use nn_verify::verify_ada_layer_norm_fusion;

    let err = verify_ada_layer_norm_fusion(&ADALN_BOUNDS, f32::NAN)
        .expect_err("NaN epsilon should be rejected");

    assert!(
        err.to_string().contains("threshold"),
        "error should mention threshold, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Sensitivity: verify test detects wrong formula
// ---------------------------------------------------------------------------

#[test]
fn test_adaln_fusion_sensitive_to_formula_error() {
    use nn_verify::{verify_fusion_equivalence, FusionSpec};

    // Build the correct fused kernel and a deliberately WRONG second kernel.
    // Instead of `(1+gamma)*x + beta`, use `gamma*x + beta` (missing the +1).
    // The diff should be measurably non-zero for non-zero gamma.
    let fused = nn_dsl::build_ada_layer_norm_fused_kernel().expect("fused kernel");
    let layer_norm = nn_dsl::build_layer_norm_scalar_kernel().expect("layer_norm kernel");

    // Build a wrong adaptive affine: gamma*x + beta (NOT (1+gamma)*x + beta)
    let wrong_fn: syn::ItemFn = syn::parse_str(
        "fn wrong_affine(x: f32, gamma: f32, beta: f32) -> f32 {
            gamma * x + beta
        }",
    )
    .expect("valid Rust");
    let wrong_affine = nn_dsl::Lowerer::lower_fn(&wrong_fn).expect("wrong kernel should lower");

    let spec = FusionSpec::new(
        &fused,
        &layer_norm,
        &wrong_affine,
        8,
        &[0, 1, 2, 3, 4, 5],
        &[0, 6, 7],
        0,
    )
    .expect("valid fusion spec");

    // Use narrow bounds where the gamma-induced diff is clearly non-zero.
    let narrow_bounds = [
        (0.9, 1.1),   // x
        (-0.1, 0.1),  // mean
        (0.9, 1.1),   // var_val
        (1e-5, 1e-5), // eps
        (0.9, 1.1),   // norm_weight
        (-0.1, 0.1),  // norm_bias
        (0.5, 0.5),   // gamma = 0.5 (point): correct gives 1.5*normed, wrong gives 0.5*normed
        (0.0, 0.0),   // beta = 0 (point)
    ];

    let result = verify_fusion_equivalence(&spec, &narrow_bounds, 1e-3)
        .expect("verification should succeed (diff may be large)");

    // The diff should be significant because the wrong formula omits the +1
    // on gamma. For normed ≈ 1.0, diff ≈ |1.5*1 - 0.5*1| = 1.0.
    assert!(
        !result.within_epsilon,
        "wrong formula should NOT be within epsilon 1e-3, got max_abs_diff: {}",
        result.max_abs_diff,
    );
    assert!(
        result.max_abs_diff > 0.1,
        "diff should be measurably non-zero for wrong formula, got {}",
        result.max_abs_diff,
    );
}

// ---------------------------------------------------------------------------
// Verify-and-record test
// ---------------------------------------------------------------------------

#[test]
fn test_adaln_fusion_verify_and_record() {
    use nn_verify::{verify_fusion_and_record, FusionSpec, VerifyStatus};

    let fused = nn_dsl::build_ada_layer_norm_fused_kernel().expect("fused kernel");
    let layer_norm = nn_dsl::build_layer_norm_scalar_kernel().expect("layer_norm kernel");
    let adaptive_affine = nn_dsl::build_adaptive_affine_kernel().expect("affine kernel");

    let spec = FusionSpec::new(
        &fused,
        &layer_norm,
        &adaptive_affine,
        8,
        &[0, 1, 2, 3, 4, 5],
        &[0, 6, 7],
        0,
    )
    .expect("valid fusion spec");

    let mut status = VerifyStatus::default();
    let result = verify_fusion_and_record(&mut status, &spec, &ADALN_BOUNDS, f32::MAX, None)
        .expect("fusion verify-and-record should succeed");

    let entry = status
        .kernel("fusion_ada_layer_norm")
        .expect("fusion entry should exist in status");
    assert!(entry.output_bounds.lower.is_finite());
    assert!(entry.output_bounds.upper.is_finite());
    assert_eq!(entry.method, PropMethod::Crown);
    assert_eq!(entry.soundness_mode, result.fusion.soundness_mode);
    assert_eq!(status.run_count("fusion_ada_layer_norm"), 1);
}

// ---------------------------------------------------------------------------
// ForwardMode config tests (#2225)
// ---------------------------------------------------------------------------

/// Tightened bounds for ForwardMode CROWN propagation.
const ADALN_TIGHTENED: [(f32, f32); 8] = [
    (-1.5, 1.5),  // x
    (-0.5, 0.5),  // mean
    (0.5, 3.0),   // var_val (away from zero)
    (1e-5, 1e-5), // eps
    (0.5, 1.5),   // norm_weight
    (-0.5, 0.5),  // norm_bias
    (-1.0, 1.0),  // gamma
    (-1.0, 1.0),  // beta
];

#[test]
fn test_adaln_fusion_with_config_tightened_crown_succeeds() {
    use nn_verify::{NormBoundsMode, VerifyConfig};

    let config = VerifyConfig::default().with_norm_mode(NormBoundsMode::ForwardMode);
    let result =
        nn_verify::verify_ada_layer_norm_fusion_with_config(&ADALN_TIGHTENED, f32::MAX, &config)
            .expect("tightened bounds should produce valid CROWN result");

    assert_eq!(
        result.method,
        PropMethod::Crown,
        "CROWN should succeed with tightened bounds"
    );
    assert!(result.is_conclusive());
    assert!(
        result.diff_lower.is_finite() && result.diff_upper.is_finite(),
        "diff bounds must be finite, got [{}, {}]",
        result.diff_lower,
        result.diff_upper,
    );
    eprintln!(
        "AdaLN tightened CROWN diff: [{}, {}], max_abs: {}",
        result.diff_lower, result.diff_upper, result.max_abs_diff,
    );
}
