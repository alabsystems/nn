// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fusion equivalence tests for additional fusion pairs.
//!
//! Split from `fusion_equivalence.rs` to keep both files under 500 lines.
//!
//! Covers:
//! - RMSNorm + SiLU-Mul (LLaMA/Mistral SwiGLU pattern) — #803 AC2
//! - LayerNorm + GELU (Transformer FFN pattern) — #803 AC2
//! - AdaIN + LeakyReLU (Kokoro decoder pattern) — #2931 AC
//!
//! Part of #887 — test code-health.

use nn_verify::PropMethod;

// ---------------------------------------------------------------------------
// RMSNorm + SiLU-Mul fusion tests (#803 AC2)
// ---------------------------------------------------------------------------

/// Representative bounds for RMSNorm+SiLU-Mul (LLaMA/Mistral SwiGLU pattern).
/// Tighter than raw ranges because normed = x * rms_inv * weight is the
/// argument to exp(-normed) inside sigmoid. NY rejects exp(>88)
/// as overflow. With these bounds: max |normed| = 5 * 3 * 3 = 45 < 88.
const RMS_SILU_BOUNDS: [(f32, f32); 4] = [
    (-5.0, 5.0), // x: hidden activations
    (0.1, 3.0),  // rms_inv: positive (1/sqrt(mean(x²)+eps))
    (-3.0, 3.0), // weight: learned scale
    (-5.0, 5.0), // up: gating branch
];

#[test]
fn test_rms_norm_silu_mul_fusion_point_inputs() {
    use nn_verify::verify_rms_norm_silu_mul_fusion;

    let point_bounds = [
        (1.0, 1.0), // x
        (0.5, 0.5), // rms_inv
        (1.0, 1.0), // weight
        (2.0, 2.0), // up
    ];

    // Note: epsilon is 1e-6 (not 1e-10) because CROWN's linear relaxation of
    // exp() introduces ~1 ULP difference even at point inputs when the fused
    // and sequential paths evaluate exp(-normed) through different IR node paths.
    let result = verify_rms_norm_silu_mul_fusion(&point_bounds, 1e-6)
        .expect("point input verification should succeed");

    assert!(
        result.within_epsilon,
        "point inputs should prove near-zero diff, got max_abs_diff: {}",
        result.max_abs_diff,
    );
    assert!(
        result.max_abs_diff < 1e-6,
        "point diff should be sub-microsecond, got {}",
        result.max_abs_diff,
    );
    assert_eq!(result.method, PropMethod::Crown);
}

#[test]
fn test_rms_norm_silu_mul_fusion_narrow_bounds() {
    use nn_verify::verify_rms_norm_silu_mul_fusion;

    let narrow_bounds = [
        (0.9, 1.1),   // x: ±10% around 1.0
        (0.45, 0.55), // rms_inv: near 0.5
        (0.9, 1.1),   // weight: near 1.0
        (1.9, 2.1),   // up: near 2.0
    ];

    let result = verify_rms_norm_silu_mul_fusion(&narrow_bounds, 10.0)
        .expect("narrow bounds verification should succeed");

    assert!(
        result.max_abs_diff < 10.0,
        "narrow bounds should produce tight CROWN diff, got {}",
        result.max_abs_diff,
    );
    assert_eq!(result.method, PropMethod::Crown);
}

#[test]
fn test_rms_norm_silu_mul_fusion_verify_and_record() {
    use nn_verify::{verify_fusion_and_record, FusionSpec, VerifyStatus};

    // Build the spec directly to pass to verify_fusion_and_record.
    let fused = nn_dsl::build_rms_norm_silu_mul_fused_kernel().expect("fused kernel");
    let rms_norm = nn_dsl::build_rms_norm_scalar_kernel().expect("rms_norm kernel");
    let silu_mul = nn_dsl::build_silu_mul_kernel().expect("silu_mul kernel");

    let spec = FusionSpec::new(&fused, &rms_norm, &silu_mul, 4, &[0, 1, 2], &[0, 3], 0)
        .expect("valid fusion spec");

    let mut status = VerifyStatus::default();
    let result = verify_fusion_and_record(&mut status, &spec, &RMS_SILU_BOUNDS, f32::MAX, None)
        .expect("fusion verify-and-record should succeed");

    // Result recorded under default key "fusion_rms_norm_silu_mul".
    let entry = status
        .kernel("fusion_rms_norm_silu_mul")
        .expect("fusion entry should exist in status");
    assert!(entry.output_bounds.lower.is_finite());
    assert!(entry.output_bounds.upper.is_finite());
    assert_eq!(entry.method, PropMethod::Crown);
    assert_eq!(entry.soundness_mode, result.fusion.soundness_mode);
    assert_eq!(status.run_count("fusion_rms_norm_silu_mul"), 1);
}

#[test]
fn test_rms_norm_silu_mul_fusion_wrong_bounds_count() {
    use nn_verify::verify_rms_norm_silu_mul_fusion;

    let bad_bounds = [(1.0, 2.0), (0.1, 1.0), (0.5, 1.5)]; // 3 instead of 4
    let err = verify_rms_norm_silu_mul_fusion(&bad_bounds, 1e-5)
        .expect_err("wrong bounds count should fail");
    assert!(err.to_string().contains("mismatch"));
}

// ---------------------------------------------------------------------------
// LayerNorm + GELU fusion tests (#803 AC2)
// ---------------------------------------------------------------------------

/// Representative bounds for LayerNorm+GELU (Transformer FFN pattern).
/// Tighter than raw ranges because the normalized value
/// `(x-mean)/sqrt(var+eps) * gamma + beta` feeds into GELU's exp(2*k*inner).
/// NY rejects exp(>88). With these bounds: max |normed| ≈
/// (3-(-1))/sqrt(0.5)*2+1 = 4/0.71*2+1 ≈ 12.3, and GELU inner ≈
/// 0.8*(12.3+0.044715*12.3³) ≈ 1500, which exceeds 88/2=44. Use even
/// tighter bounds so inner stays below ~44.
const LN_GELU_BOUNDS: [(f32, f32); 6] = [
    (-2.0, 2.0),  // x: activations
    (-1.0, 1.0),  // mean: layer mean
    (0.5, 5.0),   // var_val: layer variance (positive, not too small)
    (1e-5, 1e-5), // eps: constant epsilon
    (0.5, 2.0),   // gamma: learned scale
    (-1.0, 1.0),  // beta: learned shift
];

#[test]
fn test_layer_norm_gelu_fusion_point_inputs() {
    use nn_verify::verify_layer_norm_gelu_fusion;

    let point_bounds = [
        (1.0, 1.0),   // x
        (0.0, 0.0),   // mean
        (1.0, 1.0),   // var_val
        (1e-5, 1e-5), // eps
        (1.0, 1.0),   // gamma
        (0.0, 0.0),   // beta
    ];

    // Note: epsilon is 1e-6 (not 1e-10) because CROWN's linear relaxation of
    // rsqrt() and exp() introduces ~1 ULP difference even at point inputs when
    // the fused and sequential paths evaluate through different IR node paths.
    let result = verify_layer_norm_gelu_fusion(&point_bounds, 1e-6)
        .expect("point input verification should succeed");

    assert!(
        result.within_epsilon,
        "point inputs should prove near-zero diff, got max_abs_diff: {}",
        result.max_abs_diff,
    );
    assert!(
        result.max_abs_diff < 1e-6,
        "point diff should be sub-microsecond, got {}",
        result.max_abs_diff,
    );
    assert_eq!(result.method, PropMethod::Crown);
}

#[test]
fn test_layer_norm_gelu_fusion_narrow_bounds() {
    use nn_verify::verify_layer_norm_gelu_fusion;

    let narrow_bounds = [
        (0.9, 1.1),   // x: ±10% around 1.0
        (-0.1, 0.1),  // mean: near zero
        (0.9, 1.1),   // var_val: near 1.0
        (1e-5, 1e-5), // eps
        (0.9, 1.1),   // gamma: near 1.0
        (-0.1, 0.1),  // beta: near zero
    ];

    let result = verify_layer_norm_gelu_fusion(&narrow_bounds, 10.0)
        .expect("narrow bounds verification should succeed");

    assert!(
        result.max_abs_diff < 10.0,
        "narrow bounds should produce tight CROWN diff, got {}",
        result.max_abs_diff,
    );
    assert_eq!(result.method, PropMethod::Crown);
}

#[test]
fn test_layer_norm_gelu_fusion_verify_and_record() {
    use nn_verify::{verify_fusion_and_record, FusionSpec, VerifyStatus};

    let fused = nn_dsl::build_layer_norm_gelu_fused_kernel().expect("fused kernel");
    let layer_norm = nn_dsl::build_layer_norm_scalar_kernel().expect("layer_norm kernel");
    let gelu = nn_dsl::build_gelu_kernel().expect("gelu kernel");

    let spec = FusionSpec::new(&fused, &layer_norm, &gelu, 6, &[0, 1, 2, 3, 4, 5], &[0], 0)
        .expect("valid fusion spec");

    let mut status = VerifyStatus::default();
    let result = verify_fusion_and_record(&mut status, &spec, &LN_GELU_BOUNDS, f32::MAX, None)
        .expect("fusion verify-and-record should succeed");

    // Result recorded under default key "fusion_layer_norm_gelu".
    let entry = status
        .kernel("fusion_layer_norm_gelu")
        .expect("fusion entry should exist in status");
    assert!(entry.output_bounds.lower.is_finite());
    assert!(entry.output_bounds.upper.is_finite());
    assert_eq!(entry.method, PropMethod::Crown);
    assert_eq!(entry.soundness_mode, result.fusion.soundness_mode);
    assert_eq!(status.run_count("fusion_layer_norm_gelu"), 1);
}

#[test]
fn test_layer_norm_gelu_fusion_wrong_bounds_count() {
    use nn_verify::verify_layer_norm_gelu_fusion;

    let bad_bounds = [(1.0, 2.0); 5]; // 5 instead of 6
    let err = verify_layer_norm_gelu_fusion(&bad_bounds, 1e-5)
        .expect_err("wrong bounds count should fail");
    assert!(err.to_string().contains("mismatch"));
}

// ---------------------------------------------------------------------------
// AdaIN + LeakyReLU fusion tests (#2931 AC)
// ---------------------------------------------------------------------------

/// Representative bounds for AdaIN+LeakyReLU (Kokoro decoder pattern).
/// Parameters: x, mu, var_val, gamma, beta, slope, eps (7 shared inputs).
/// AdaIN maps to [0,1,2,3,4,6], LeakyReLU maps to [output_of_adain, 5].
const ADAIN_LEAKY_RELU_BOUNDS: [(f32, f32); 7] = [
    (-10.0, 10.0), // x: activations
    (-5.0, 5.0),   // mu: instance mean
    (0.001, 10.0), // var_val: instance variance (positive)
    (0.1, 5.0),    // gamma: style scale
    (-3.0, 3.0),   // beta: style shift
    (0.01, 0.5),   // slope: LeakyReLU negative slope
    (1e-5, 1e-5),  // eps: constant epsilon
];

#[test]
fn test_adain_leaky_relu_fusion_equivalence() {
    use nn_verify::verify_adain_leaky_relu_fusion;

    let result = verify_adain_leaky_relu_fusion(&ADAIN_LEAKY_RELU_BOUNDS, f32::MAX)
        .expect("AdaIN+LeakyReLU fusion verification should succeed");

    assert_eq!(result.method, PropMethod::Crown);
    assert!(result.is_conclusive());
    assert!(
        result.diff_lower.is_finite() && result.diff_upper.is_finite(),
        "diff bounds should be finite, got [{}, {}]",
        result.diff_lower,
        result.diff_upper,
    );
    eprintln!(
        "AdaIN+LeakyReLU CROWN diff: [{}, {}], max_abs: {}",
        result.diff_lower, result.diff_upper, result.max_abs_diff,
    );
}

#[test]
fn test_adain_leaky_relu_fusion_point_inputs() {
    use nn_verify::verify_adain_leaky_relu_fusion;

    let point_bounds = [
        (1.0, 1.0),   // x
        (0.0, 0.0),   // mu
        (1.0, 1.0),   // var_val
        (1.0, 1.0),   // gamma
        (0.0, 0.0),   // beta
        (0.1, 0.1),   // slope
        (1e-5, 1e-5), // eps
    ];

    // Note: epsilon is 2e-6 (not 1e-10) because CROWN's linear relaxation of
    // rsqrt() and the branch/select for LeakyReLU introduce ~1.2 ULP difference
    // even at point inputs when the fused and sequential paths evaluate through
    // different IR node paths.
    let result = verify_adain_leaky_relu_fusion(&point_bounds, 2e-6)
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
fn test_adain_leaky_relu_fusion_with_verify_wiring() {
    use nn_verify::verify_fusion_wiring;

    let fused = nn_dsl::build_adain_leaky_relu_fused_kernel().expect("fused kernel");
    let adain = nn_dsl::build_adain_scalar_kernel().expect("adain kernel");
    let leaky_relu = nn_dsl::build_leaky_relu_scalar_kernel().expect("leaky_relu kernel");

    // AdaIN params: x(0), mu(1), var_val(2), gamma(3), beta(4), eps(6)
    // LeakyReLU params: y(output of adain), slope(5)
    let result = verify_fusion_wiring(
        &fused,
        &adain,
        &leaky_relu,
        7,                   // num_shared_inputs
        &[0, 1, 2, 3, 4, 6], // first_param_indices (adain)
        &[0, 5],             // second_param_indices (leaky_relu: y=placeholder, slope=5)
        0,                   // second_input_from_first (leaky_relu's y)
        &ADAIN_LEAKY_RELU_BOUNDS,
        f32::MAX,
    )
    .expect("verify_fusion_wiring should succeed");

    assert_eq!(result.method, PropMethod::Crown);
    assert!(result.is_conclusive());
}

// ---------------------------------------------------------------------------
// ForwardMode config tests (#2225)
// ---------------------------------------------------------------------------

/// Tightened bounds from #2225: narrower ranges avoid exp overflow and reduce
/// CROWN relaxation error through the diamond DAG.
const RMS_TIGHTENED: [(f32, f32); 4] = [
    (-3.0, 3.0), // x
    (0.2, 2.0),  // rms_inv (away from zero)
    (-2.0, 2.0), // weight
    (-3.0, 3.0), // up
];

const LN_TIGHTENED: [(f32, f32); 6] = [
    (-1.5, 1.5),  // x
    (-0.5, 0.5),  // mean
    (0.5, 3.0),   // var_val (away from zero)
    (1e-5, 1e-5), // eps
    (0.5, 1.5),   // gamma
    (-0.5, 0.5),  // beta
];

#[test]
fn test_rms_norm_silu_mul_with_config_tightened_crown_succeeds() {
    use nn_verify::{verify_rms_norm_silu_mul_fusion_with_config, NormBoundsMode, VerifyConfig};

    let config = VerifyConfig::default().with_norm_mode(NormBoundsMode::ForwardMode);
    let result = verify_rms_norm_silu_mul_fusion_with_config(&RMS_TIGHTENED, f32::MAX, &config)
        .expect("tightened bounds should produce valid CROWN result");

    assert_eq!(
        result.method,
        PropMethod::Crown,
        "CROWN should succeed with tightened bounds"
    );
    assert!(result.is_conclusive());
    assert!(result.diff_lower.is_finite() && result.diff_upper.is_finite());
    // Tightened bounds should produce bounded diff. Current CROWN diff is ~72;
    // ceiling at 500 catches regressions without being brittle.
    assert!(
        result.max_abs_diff < 500.0,
        "tightened RMS+SiLU CROWN diff should be bounded, got {}",
        result.max_abs_diff,
    );
    eprintln!(
        "RMS+SiLU tightened CROWN diff: [{}, {}], max_abs: {}",
        result.diff_lower, result.diff_upper, result.max_abs_diff,
    );
}

#[test]
fn test_layer_norm_gelu_with_config_tightened_crown_succeeds() {
    use nn_verify::{verify_layer_norm_gelu_fusion_with_config, NormBoundsMode, VerifyConfig};

    let config = VerifyConfig::default().with_norm_mode(NormBoundsMode::ForwardMode);
    let result = verify_layer_norm_gelu_fusion_with_config(&LN_TIGHTENED, f32::MAX, &config)
        .expect("tightened bounds should produce valid CROWN result");

    assert_eq!(
        result.method,
        PropMethod::Crown,
        "CROWN should succeed with tightened bounds"
    );
    assert!(result.is_conclusive());
    assert!(result.diff_lower.is_finite() && result.diff_upper.is_finite());
    // Tightened bounds should produce bounded diff. Current CROWN diff is ~9.5;
    // ceiling at 100 catches regressions without being brittle.
    assert!(
        result.max_abs_diff < 100.0,
        "tightened LN+GELU CROWN diff should be bounded, got {}",
        result.max_abs_diff,
    );
    eprintln!(
        "LN+GELU tightened CROWN diff: [{}, {}], max_abs: {}",
        result.diff_lower, result.diff_upper, result.max_abs_diff,
    );
}
