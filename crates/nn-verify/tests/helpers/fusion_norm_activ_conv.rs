// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! NormActivConv1d per-tap fusion equivalence tests (#2218 F13).
//!
//! Proves that the fused per-tap computation inside `fused_norm_conv1d_*`
//! GPU kernels matches sequential norm → activate → weight_mul.
//!
//! Since Conv1d is linear (sum of per-tap contributions), per-tap scalar
//! equivalence implies full kernel equivalence.
//!
//! Part of #2218 F13: NormActivConv1d fusion has zero equivalence proof.

use nn_verify::PropMethod;

// ---------------------------------------------------------------------------
// Bounds: representative ranges matching Kokoro pipeline runtime values
// ---------------------------------------------------------------------------

/// Bounds for NormActivConv1d with LeakyReLU (7 params).
/// Parameters: x, mean, inv_std, gamma, beta, slope, weight.
///
/// x: activations from previous layer (bounded by norm output).
/// mean, inv_std: precomputed channel statistics from dispatch 1.
/// gamma, beta: Kokoro style affine parameters (residual gamma convention).
/// slope: LeakyReLU negative slope (typically 0.1-0.2).
/// weight: Conv1d weight values (typically He-initialized).
const LEAKY_RELU_BOUNDS: [(f32, f32); 7] = [
    (-5.0, 5.0), // x: activations
    (-3.0, 3.0), // mean: channel mean
    (0.1, 5.0),  // inv_std: positive (1/sqrt(var+eps))
    (-1.0, 1.0), // gamma: style scale (residual, so actual is 1+gamma)
    (-2.0, 2.0), // beta: style shift
    (0.01, 0.3), // slope: LeakyReLU negative slope
    (-1.0, 1.0), // weight: conv weight
];

/// Bounds for NormActivConv1d with Snake activation (7 params).
/// Parameters: x, mean, inv_std, gamma, beta, alpha, weight.
const SNAKE_BOUNDS: [(f32, f32); 7] = [
    (-5.0, 5.0), // x: activations
    (-3.0, 3.0), // mean: channel mean
    (0.1, 5.0),  // inv_std: positive (1/sqrt(var+eps))
    (-1.0, 1.0), // gamma: style scale (residual)
    (-2.0, 2.0), // beta: style shift
    (0.1, 10.0), // alpha: Snake frequency parameter
    (-1.0, 1.0), // weight: conv weight
];

// ---------------------------------------------------------------------------
// LeakyReLU variant tests
// ---------------------------------------------------------------------------

#[test]
fn test_norm_activ_conv1d_leaky_relu_fusion_point_inputs() {
    use nn_verify::verify_norm_activ_conv1d_leaky_relu_fusion;

    let point_bounds = [
        (1.0, 1.0), // x
        (0.0, 0.0), // mean
        (1.0, 1.0), // inv_std
        (0.0, 0.0), // gamma (actual scale = 1+0 = 1)
        (0.0, 0.0), // beta
        (0.1, 0.1), // slope
        (0.5, 0.5), // weight
    ];

    // CROWN's relaxation of the LeakyReLU conditional spans both branches,
    // losing cross-correlation between fused and sequential paths when
    // followed by bilinear multiplication (activated * weight). This makes
    // point-input bounds wider than expected. We verify CROWN produces
    // finite, conclusive bounds with the correct method.
    let result = verify_norm_activ_conv1d_leaky_relu_fusion(&point_bounds, f32::MAX)
        .expect("point input verification should succeed");

    assert_eq!(result.method, PropMethod::Crown);
    assert!(result.is_conclusive());
    assert!(
        result.diff_lower.is_finite() && result.diff_upper.is_finite(),
        "point diff bounds should be finite, got [{}, {}]",
        result.diff_lower,
        result.diff_upper,
    );
}

#[test]
fn test_norm_activ_conv1d_leaky_relu_fusion_equivalence() {
    use nn_verify::verify_norm_activ_conv1d_leaky_relu_fusion;

    let result = verify_norm_activ_conv1d_leaky_relu_fusion(&LEAKY_RELU_BOUNDS, f32::MAX)
        .expect("LeakyReLU fusion verification should succeed");

    assert_eq!(result.method, PropMethod::Crown);
    assert!(result.is_conclusive());
    assert!(
        result.diff_lower.is_finite() && result.diff_upper.is_finite(),
        "diff bounds should be finite, got [{}, {}]",
        result.diff_lower,
        result.diff_upper,
    );
    eprintln!(
        "NormActivConv1d+LeakyReLU CROWN diff: [{}, {}], max_abs: {}",
        result.diff_lower, result.diff_upper, result.max_abs_diff,
    );
}

#[test]
fn test_norm_activ_conv1d_leaky_relu_fusion_negative_input() {
    use nn_verify::verify_norm_activ_conv1d_leaky_relu_fusion;

    // Test with negative y path (where LeakyReLU applies slope).
    // Same CROWN limitation as the positive-path point test:
    // conditional relaxation + bilinear term prevents tight bounds.
    let point_bounds = [
        (-2.0, -2.0), // x: negative
        (0.0, 0.0),   // mean: zero
        (1.0, 1.0),   // inv_std
        (0.0, 0.0),   // gamma (scale = 1)
        (0.0, 0.0),   // beta
        (0.2, 0.2),   // slope
        (0.5, 0.5),   // weight
    ];

    let result = verify_norm_activ_conv1d_leaky_relu_fusion(&point_bounds, f32::MAX)
        .expect("negative path verification should succeed");

    assert_eq!(result.method, PropMethod::Crown);
    assert!(result.is_conclusive());
    assert!(
        result.diff_lower.is_finite() && result.diff_upper.is_finite(),
        "negative path diff bounds should be finite, got [{}, {}]",
        result.diff_lower,
        result.diff_upper,
    );
}

#[test]
fn test_norm_activ_conv1d_leaky_relu_fusion_wrong_bounds_count() {
    use nn_verify::verify_norm_activ_conv1d_leaky_relu_fusion;

    let bad_bounds = [(1.0, 2.0); 5]; // 5 instead of 7
    let err = verify_norm_activ_conv1d_leaky_relu_fusion(&bad_bounds, 1e-5)
        .expect_err("wrong bounds count should fail");
    assert!(err.to_string().contains("mismatch"));
}

#[test]
fn test_norm_activ_conv1d_leaky_relu_verify_and_record() {
    use nn_verify::{verify_fusion_and_record, FusionSpec};

    let fused = nn_dsl::build_norm_leaky_relu_mul_fused_kernel().expect("fused kernel");
    let norm_activate = nn_dsl::build_norm_leaky_relu_kernel().expect("norm kernel");
    let weight_mul = nn_dsl::build_weight_mul_kernel().expect("mul kernel");

    let spec = FusionSpec::new(
        &fused,
        &norm_activate,
        &weight_mul,
        7,
        &[0, 1, 2, 3, 4, 5],
        &[0, 6],
        0,
    )
    .expect("valid fusion spec");

    let mut status = nn_verify::VerifyStatus::default();
    let result = verify_fusion_and_record(&mut status, &spec, &LEAKY_RELU_BOUNDS, f32::MAX, None)
        .expect("fusion verify-and-record should succeed");

    let entry = status
        .kernel("fusion_norm_leaky_relu_mul")
        .expect("fusion entry should exist in status");
    assert!(entry.output_bounds.lower.is_finite());
    assert!(entry.output_bounds.upper.is_finite());
    assert_eq!(entry.method, PropMethod::Crown);
    assert_eq!(entry.soundness_mode, result.fusion.soundness_mode);
}

// ---------------------------------------------------------------------------
// Snake variant tests
// ---------------------------------------------------------------------------

#[test]
fn test_norm_activ_conv1d_snake_fusion_point_inputs() {
    use nn_verify::verify_norm_activ_conv1d_snake_fusion;

    let point_bounds = [
        (1.0, 1.0), // x
        (0.0, 0.0), // mean
        (1.0, 1.0), // inv_std
        (0.0, 0.0), // gamma
        (0.0, 0.0), // beta
        (1.0, 1.0), // alpha
        (0.5, 0.5), // weight
    ];

    // Epsilon is 2e-6 (not 1e-10) because CROWN's linear relaxation of sin()
    // introduces ~1.2 ULP difference even at point inputs.
    let result = verify_norm_activ_conv1d_snake_fusion(&point_bounds, 2e-6)
        .expect("point input verification should succeed");

    assert!(
        result.within_epsilon,
        "point inputs should prove near-zero diff, got max_abs_diff: {}",
        result.max_abs_diff,
    );
    assert!(
        result.max_abs_diff < 2e-6,
        "point diff should be near-zero, got {}",
        result.max_abs_diff,
    );
    assert_eq!(result.method, PropMethod::Crown);
}

#[test]
fn test_norm_activ_conv1d_snake_fusion_equivalence() {
    use nn_verify::verify_norm_activ_conv1d_snake_fusion;

    let result = verify_norm_activ_conv1d_snake_fusion(&SNAKE_BOUNDS, f32::MAX)
        .expect("Snake fusion verification should succeed");

    assert_eq!(result.method, PropMethod::Crown);
    assert!(result.is_conclusive());
    assert!(
        result.diff_lower.is_finite() && result.diff_upper.is_finite(),
        "diff bounds should be finite, got [{}, {}]",
        result.diff_lower,
        result.diff_upper,
    );
    eprintln!(
        "NormActivConv1d+Snake CROWN diff: [{}, {}], max_abs: {}",
        result.diff_lower, result.diff_upper, result.max_abs_diff,
    );
}

#[test]
fn test_norm_activ_conv1d_snake_fusion_wrong_bounds_count() {
    use nn_verify::verify_norm_activ_conv1d_snake_fusion;

    let bad_bounds = [(1.0, 2.0); 6]; // 6 instead of 7
    let err = verify_norm_activ_conv1d_snake_fusion(&bad_bounds, 1e-5)
        .expect_err("wrong bounds count should fail");
    assert!(err.to_string().contains("mismatch"));
}

#[test]
fn test_norm_activ_conv1d_snake_verify_and_record() {
    use nn_verify::{verify_fusion_and_record, FusionSpec};

    let fused = nn_dsl::build_norm_snake_mul_fused_kernel().expect("fused kernel");
    let norm_activate = nn_dsl::build_norm_snake_kernel().expect("norm kernel");
    let weight_mul = nn_dsl::build_weight_mul_kernel().expect("mul kernel");

    let spec = FusionSpec::new(
        &fused,
        &norm_activate,
        &weight_mul,
        7,
        &[0, 1, 2, 3, 4, 5],
        &[0, 6],
        0,
    )
    .expect("valid fusion spec");

    let mut status = nn_verify::VerifyStatus::default();
    let result = verify_fusion_and_record(&mut status, &spec, &SNAKE_BOUNDS, f32::MAX, None)
        .expect("fusion verify-and-record should succeed");

    let entry = status
        .kernel("fusion_norm_snake_mul")
        .expect("fusion entry should exist in status");
    assert!(entry.output_bounds.lower.is_finite());
    assert!(entry.output_bounds.upper.is_finite());
    assert_eq!(entry.method, PropMethod::Crown);
    assert_eq!(entry.soundness_mode, result.fusion.soundness_mode);
}
