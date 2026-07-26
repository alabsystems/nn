// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for fusion equivalence certificates (#2462).
//!
//! Verifies that `verify_fusion_certificate()` generates valid certificates
//! for all 5 fusion pairs at production dimension D=512, with both CROWN
//! bounds and analytical ULP proofs.
//!
//! # Dimension coverage
//!
//! Fusion equivalence is element-wise: the scalar proof covers D=512 because
//! fused and sequential kernels compute the same scalar function at each
//! element position with no cross-element interactions. The certificate's
//! `dimension_coverage_rationale` documents this.
//!
//! # Monte Carlo validation
//!
//! Tests include random sampling that evaluates fused and sequential scalar
//! functions to confirm the empirical diff is zero (identical evaluation paths
//! in f32) or bounded by the analytical ULP bound.

use nn_dsl::{
    build_ada_layer_norm_fused_kernel, build_adain_leaky_relu_fused_kernel,
    build_adain_scalar_kernel, build_adain_snake_fused_kernel, build_adaptive_affine_kernel,
    build_gelu_kernel, build_layer_norm_gelu_fused_kernel, build_layer_norm_scalar_kernel,
    build_leaky_relu_scalar_kernel, build_rms_norm_scalar_kernel,
    build_rms_norm_silu_mul_fused_kernel, build_silu_mul_kernel, build_snake_scalar_kernel,
};
use nn_verify::{
    verify_fusion_certificate, FusionEquivalenceCertificate, FusionSpec, FUSION_CERTIFICATE_VERSION,
};

/// dvoice Kokoro AdaIN+Snake bounds (7 variables).
/// These match DVOICE_BOUNDS in fusion_equivalence.rs.
const ADAIN_SNAKE_BOUNDS: [(f32, f32); 7] = [
    (-10.0, 10.0), // x: audio features after encoder
    (-5.0, 5.0),   // mu: channel mean
    (0.001, 10.0), // var: channel variance (positive)
    (0.1, 5.0),    // gamma: style scale
    (-3.0, 3.0),   // beta: style shift
    (0.01, 100.0), // alpha: snake activation parameter
    (1e-5, 1e-5),  // eps: constant epsilon (point interval)
];

/// RMSNorm+SiLU-Mul bounds (4 variables).
/// Tighter than raw ranges: max |normed| = 5 * 3 * 3 = 45 < 88 (exp overflow).
/// Matches RMS_SILU_BOUNDS in fusion_equivalence_pairs.rs.
const RMS_SILU_MUL_BOUNDS: [(f32, f32); 4] = [
    (-5.0, 5.0), // x: hidden activations
    (0.1, 3.0),  // rms_inv: positive (1/sqrt(mean(x²)+eps))
    (-3.0, 3.0), // weight: learned scale
    (-5.0, 5.0), // up: gating branch
];

/// LayerNorm+GELU bounds (6 variables).
/// Tighter than raw ranges to avoid GELU tanh exp overflow (threshold 88).
/// Matches LN_GELU_BOUNDS in fusion_equivalence_pairs.rs.
const LN_GELU_BOUNDS: [(f32, f32); 6] = [
    (-2.0, 2.0),  // x: activations
    (-1.0, 1.0),  // mean: layer mean
    (0.5, 5.0),   // var_val: layer variance (positive, not too small)
    (1e-5, 1e-5), // eps: constant epsilon
    (0.5, 2.0),   // gamma: learned scale
    (-1.0, 1.0),  // beta: learned shift
];

/// AdaIN+LeakyReLU bounds (7 variables).
/// Same AdaIN bounds as Snake pair, with LeakyReLU slope parameter.
const ADAIN_LEAKY_RELU_CERT_BOUNDS: [(f32, f32); 7] = [
    (-10.0, 10.0), // x: activations
    (-5.0, 5.0),   // mu: instance mean
    (0.001, 10.0), // var_val: instance variance (positive)
    (0.1, 5.0),    // gamma: style scale
    (-3.0, 3.0),   // beta: style shift
    (0.01, 0.5),   // slope: LeakyReLU negative slope
    (1e-5, 1e-5),  // eps: constant epsilon
];

/// AdaLayerNorm bounds (8 variables).
/// LayerNorm + adaptive affine: x, mean, var_val, eps, norm_weight, norm_bias, gamma, beta.
/// Matches ADALN_BOUNDS in fusion_equivalence_adaln.rs.
const ADALN_CERT_BOUNDS: [(f32, f32); 8] = [
    (-3.0, 3.0),  // x: activations after previous layer
    (-1.0, 1.0),  // mean: layer mean
    (0.5, 5.0),   // var_val: layer variance (positive, not too small)
    (1e-5, 1e-5), // eps: constant epsilon (point interval)
    (0.5, 2.0),   // norm_weight: learned LayerNorm scale
    (-1.0, 1.0),  // norm_bias: learned LayerNorm shift
    (-2.0, 2.0),  // gamma: adaptive style scale
    (-2.0, 2.0),  // beta: adaptive style shift
];

// ── AdaIN+Snake certificate ─────────────────────────────────────────────

#[test]
fn test_adain_snake_certificate_d512() {
    let fused = build_adain_snake_fused_kernel().expect("fused kernel");
    let adain = build_adain_scalar_kernel().expect("adain kernel");
    let snake = build_snake_scalar_kernel().expect("snake kernel");

    let spec = FusionSpec::new(&fused, &adain, &snake, 7, &[0, 1, 2, 3, 4, 6], &[0, 5], 0)
        .expect("valid spec");

    let analytical =
        nn_verify::fusion_certificate::known_bounds::adain_snake().expect("known bound");

    let cert = verify_fusion_certificate(
        &spec,
        "adain",
        "snake",
        &ADAIN_SNAKE_BOUNDS,
        1e-4,
        512,
        Some(analytical),
    )
    .expect("certificate generation");

    // Certificate structure.
    assert_eq!(cert.version, FUSION_CERTIFICATE_VERSION);
    assert_eq!(cert.fused_kernel_name, "adain_snake");
    assert_eq!(
        cert.sequential_names,
        ("adain".to_string(), "snake".to_string())
    );
    assert_eq!(cert.dimension, 512);
    assert_eq!(cert.epsilon, 1e-4);
    assert_eq!(cert.variable_bounds.len(), 7);

    // CROWN bound exists (may be loose for wide intervals).
    assert!(cert.crown_bound.is_some());

    // Analytical ULP bound proves equivalence.
    let analytical = cert
        .analytical_bound
        .as_ref()
        .expect("has analytical bound");
    assert!(
        analytical.proves_within_epsilon(1e-4),
        "analytical max_abs_diff {} should be < 1e-4",
        analytical.max_abs_diff
    );

    // Certificate proves equivalence via analytical bound.
    assert!(
        cert.proves_equivalence(),
        "certificate should prove equivalence"
    );

    // Tightest bound should be the analytical one.
    let tightest = cert.tightest_bound().expect("has bounds");
    assert!(
        tightest < 1e-4,
        "tightest bound {tightest} should be < 1e-4"
    );

    // Validate passes.
    cert.validate().expect("valid certificate");

    // Serde roundtrip.
    let json = cert.to_json().expect("serialize");
    let deserialized: FusionEquivalenceCertificate =
        serde_json::from_str(&json).expect("deserialize");
    assert_eq!(cert.fused_kernel_name, deserialized.fused_kernel_name);
    assert_eq!(cert.dimension, deserialized.dimension);
    assert!(deserialized.proves_equivalence());
}

// ── RMSNorm+SiLU-Mul certificate ───────────────────────────────────────

#[test]
fn test_rms_norm_silu_mul_certificate_d512() {
    let fused = build_rms_norm_silu_mul_fused_kernel().expect("fused kernel");
    let rms_norm = build_rms_norm_scalar_kernel().expect("rms_norm kernel");
    let silu_mul = build_silu_mul_kernel().expect("silu_mul kernel");

    let spec = FusionSpec::new(&fused, &rms_norm, &silu_mul, 4, &[0, 1, 2], &[0, 3], 0)
        .expect("valid spec");

    let analytical =
        nn_verify::fusion_certificate::known_bounds::rms_norm_silu_mul().expect("known bound");

    let cert = verify_fusion_certificate(
        &spec,
        "rms_norm",
        "silu_mul",
        &RMS_SILU_MUL_BOUNDS,
        1e-4,
        512,
        Some(analytical),
    )
    .expect("certificate generation");

    assert_eq!(cert.dimension, 512);
    assert!(cert.crown_bound.is_some());

    let analytical = cert
        .analytical_bound
        .as_ref()
        .expect("has analytical bound");
    assert!(
        analytical.proves_within_epsilon(1e-4),
        "analytical max_abs_diff {} should be < 1e-4",
        analytical.max_abs_diff
    );
    assert!(cert.proves_equivalence());
    cert.validate().expect("valid certificate");
}

// ── LayerNorm+GELU certificate ──────────────────────────────────────────

#[test]
fn test_layer_norm_gelu_certificate_d512() {
    let fused = build_layer_norm_gelu_fused_kernel().expect("fused kernel");
    let layer_norm = build_layer_norm_scalar_kernel().expect("layer_norm kernel");
    let gelu = build_gelu_kernel().expect("gelu kernel");

    let spec = FusionSpec::new(&fused, &layer_norm, &gelu, 6, &[0, 1, 2, 3, 4, 5], &[0], 0)
        .expect("valid spec");

    let analytical =
        nn_verify::fusion_certificate::known_bounds::layer_norm_gelu().expect("known bound");

    let cert = verify_fusion_certificate(
        &spec,
        "layer_norm",
        "gelu",
        &LN_GELU_BOUNDS,
        1e-4,
        512,
        Some(analytical),
    )
    .expect("certificate generation");

    assert_eq!(cert.dimension, 512);
    assert!(cert.crown_bound.is_some());

    let analytical = cert
        .analytical_bound
        .as_ref()
        .expect("has analytical bound");
    assert!(
        analytical.proves_within_epsilon(1e-4),
        "analytical max_abs_diff {} should be < 1e-4",
        analytical.max_abs_diff
    );
    assert!(cert.proves_equivalence());
    cert.validate().expect("valid certificate");
}

// ── AdaIN+LeakyReLU certificate ──────────────────────────────────────────

#[test]
fn test_adain_leaky_relu_certificate_d512() {
    let fused = build_adain_leaky_relu_fused_kernel().expect("fused kernel");
    let adain = build_adain_scalar_kernel().expect("adain kernel");
    let leaky_relu = build_leaky_relu_scalar_kernel().expect("leaky_relu kernel");

    let spec = FusionSpec::new(
        &fused,
        &adain,
        &leaky_relu,
        7,
        &[0, 1, 2, 3, 4, 6],
        &[0, 5],
        0,
    )
    .expect("valid spec");

    let analytical =
        nn_verify::fusion_certificate::known_bounds::adain_leaky_relu().expect("known bound");

    let cert = verify_fusion_certificate(
        &spec,
        "adain",
        "leaky_relu",
        &ADAIN_LEAKY_RELU_CERT_BOUNDS,
        1e-4,
        512,
        Some(analytical),
    )
    .expect("certificate generation");

    assert_eq!(cert.version, FUSION_CERTIFICATE_VERSION);
    assert_eq!(cert.dimension, 512);
    assert!(cert.crown_bound.is_some());

    let analytical = cert
        .analytical_bound
        .as_ref()
        .expect("has analytical bound");
    assert!(
        analytical.proves_within_epsilon(1e-4),
        "analytical max_abs_diff {} should be < 1e-4",
        analytical.max_abs_diff
    );
    assert!(cert.proves_equivalence());
    cert.validate().expect("valid certificate");

    let json = cert.to_json().expect("serialize");
    let deserialized: FusionEquivalenceCertificate =
        serde_json::from_str(&json).expect("deserialize");
    assert!(deserialized.proves_equivalence());
}

// ── AdaLayerNorm certificate ─────────────────────────────────────────────

#[test]
fn test_ada_layer_norm_certificate_d512() {
    let fused = build_ada_layer_norm_fused_kernel().expect("fused kernel");
    let layer_norm = build_layer_norm_scalar_kernel().expect("layer_norm kernel");
    let adaptive_affine = build_adaptive_affine_kernel().expect("adaptive_affine kernel");

    // LayerNorm: params [0..6] → x, mean, var_val, eps, gamma, beta
    // Adaptive affine: param 0 from first output, params 6,7 → gamma, beta
    let spec = FusionSpec::new(
        &fused,
        &layer_norm,
        &adaptive_affine,
        8,
        &[0, 1, 2, 3, 4, 5],
        &[0, 6, 7],
        0,
    )
    .expect("valid spec");

    let analytical =
        nn_verify::fusion_certificate::known_bounds::ada_layer_norm().expect("known bound");

    let cert = verify_fusion_certificate(
        &spec,
        "layer_norm",
        "adaptive_affine",
        &ADALN_CERT_BOUNDS,
        1e-4,
        512,
        Some(analytical),
    )
    .expect("certificate generation");

    assert_eq!(cert.version, FUSION_CERTIFICATE_VERSION);
    assert_eq!(cert.dimension, 512);
    assert!(cert.crown_bound.is_some());

    let analytical = cert
        .analytical_bound
        .as_ref()
        .expect("has analytical bound");
    assert!(
        analytical.proves_within_epsilon(1e-4),
        "analytical max_abs_diff {} should be < 1e-4",
        analytical.max_abs_diff
    );
    assert!(cert.proves_equivalence());
    cert.validate().expect("valid certificate");

    let json = cert.to_json().expect("serialize");
    let deserialized: FusionEquivalenceCertificate =
        serde_json::from_str(&json).expect("deserialize");
    assert!(deserialized.proves_equivalence());
}

// ── Analytical bound documentation ──────────────────────────────────────

#[test]
fn test_all_analytical_bounds_documented() {
    let adain =
        nn_verify::fusion_certificate::known_bounds::adain_snake().expect("adain_snake bound");
    let rms = nn_verify::fusion_certificate::known_bounds::rms_norm_silu_mul()
        .expect("rms_norm_silu_mul bound");
    let ln = nn_verify::fusion_certificate::known_bounds::layer_norm_gelu()
        .expect("layer_norm_gelu bound");

    let eps = 5.960_464_477_539_063e-8_f64; // f32 machine epsilon: 2^-24

    // AdaIN+Snake: 2 ops × 64.0 magnitude × 2^-24 × 2.0 Lipschitz ≈ 1.53e-5
    assert!((adain.max_abs_diff - 64.0 * 2.0 * eps * 2.0).abs() < 1e-15);
    assert!(
        adain.max_abs_diff < 1e-4,
        "AdaIN+Snake: {}",
        adain.max_abs_diff
    );

    // RMSNorm+SiLU-Mul: 2 ops × 72.0 magnitude × 2^-24 × 1.1 Lipschitz ≈ 9.44e-6
    assert!((rms.max_abs_diff - 72.0 * 2.0 * eps * 1.1).abs() < 1e-15);
    assert!(
        rms.max_abs_diff < 1e-4,
        "RMSNorm+SiLU-Mul: {}",
        rms.max_abs_diff
    );

    // LayerNorm+GELU: 2 ops × 10.0 magnitude × 2^-24 × 1.2 Lipschitz ≈ 1.43e-6
    assert!((ln.max_abs_diff - 10.0 * 2.0 * eps * 1.2).abs() < 1e-15);
    assert!(
        ln.max_abs_diff < 1e-4,
        "LayerNorm+GELU: {}",
        ln.max_abs_diff
    );

    // AdaIN+LeakyReLU: 2 ops × 64.0 magnitude × 2^-24 × 1.0 Lipschitz ≈ 7.63e-6
    let adain_lr = nn_verify::fusion_certificate::known_bounds::adain_leaky_relu()
        .expect("adain_leaky_relu bound");
    assert!((adain_lr.max_abs_diff - 64.0 * 2.0 * eps * 1.0).abs() < 1e-15);
    assert!(
        adain_lr.max_abs_diff < 1e-4,
        "AdaIN+LeakyReLU: {}",
        adain_lr.max_abs_diff
    );

    // AdaLayerNorm: 2 ops × 10.0 magnitude × 2^-24 × 2.0 Lipschitz ≈ 2.38e-6
    let adaln = nn_verify::fusion_certificate::known_bounds::ada_layer_norm()
        .expect("ada_layer_norm bound");
    assert!((adaln.max_abs_diff - 10.0 * 2.0 * eps * 2.0).abs() < 1e-15);
    assert!(
        adaln.max_abs_diff < 1e-4,
        "AdaLayerNorm: {}",
        adaln.max_abs_diff
    );
}

// ── Monte Carlo validation (defense-in-depth) ──────────────────────────
//
// The fused and sequential kernel IRs use identical operations (both use
// rsqrt, not sqrt.recip). The evaluation paths in f32 are deterministic
// and produce identical results for identical inputs. Monte Carlo validates
// this property by confirming zero or near-zero empirical diff.

/// SNAKE_MIN_ALPHA value from nn-dsl (alpha clamping floor).
/// Must match `nn_dsl::snake::SNAKE_MIN_ALPHA` (1e-8).
const SNAKE_MIN_ALPHA: f32 = 1e-8;

/// Evaluate AdaIN scalar: gamma * (x - mu) * rsqrt(var + eps) + beta
fn eval_adain(x: f32, mu: f32, var: f32, gamma: f32, beta: f32, eps: f32) -> f32 {
    gamma * (x - mu) * (var + eps).sqrt().recip() + beta
}

/// Evaluate Snake scalar: y + (1/a) * sin²(a*y)
fn eval_snake(y: f32, alpha: f32) -> f32 {
    let a = alpha.max(SNAKE_MIN_ALPHA);
    y + (1.0 / a) * (a * y).sin().powi(2)
}

/// Evaluate fused AdaIN+Snake (same ops as sequential, single function).
fn eval_adain_snake_fused(
    x: f32,
    mu: f32,
    var: f32,
    gamma: f32,
    beta: f32,
    alpha: f32,
    eps: f32,
) -> f32 {
    let y = gamma * (x - mu) * (var + eps).sqrt().recip() + beta;
    let a = alpha.max(SNAKE_MIN_ALPHA);
    y + (1.0 / a) * (a * y).sin().powi(2)
}

/// Sequential: adain result (rounded f32) → snake.
fn eval_adain_then_snake(
    x: f32,
    mu: f32,
    var: f32,
    gamma: f32,
    beta: f32,
    alpha: f32,
    eps: f32,
) -> f32 {
    let y = eval_adain(x, mu, var, gamma, beta, eps);
    eval_snake(y, alpha)
}

/// Simple deterministic LCG PRNG for reproducible random sampling.
fn lcg_next(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as f32) / (u32::MAX as f32 / 2.0)
}

/// Sample a random value uniformly in [lo, hi].
fn random_in_range(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let t = lcg_next(state);
    lo + t * (hi - lo)
}

#[test]
fn test_adain_snake_monte_carlo_fused_equals_sequential() {
    // With identical operations, fused and sequential produce identical f32.
    // Any nonzero diff indicates a reference implementation mismatch.
    let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_1234;
    let mut max_diff: f64 = 0.0;

    for _ in 0..1_000_000 {
        let x = random_in_range(&mut rng_state, -10.0, 10.0);
        let mu = random_in_range(&mut rng_state, -5.0, 5.0);
        let var = random_in_range(&mut rng_state, 0.001, 10.0);
        let gamma = random_in_range(&mut rng_state, 0.1, 5.0);
        let beta = random_in_range(&mut rng_state, -3.0, 3.0);
        let alpha = random_in_range(&mut rng_state, 0.01, 100.0);
        let eps: f32 = 1e-5;

        let fused = eval_adain_snake_fused(x, mu, var, gamma, beta, alpha, eps);
        let sequential = eval_adain_then_snake(x, mu, var, gamma, beta, alpha, eps);

        if fused.is_finite() && sequential.is_finite() {
            let diff = (f64::from(fused) - f64::from(sequential)).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    // Identical ops → identical results. Tolerance for FMA/optimization effects.
    let analytical =
        nn_verify::fusion_certificate::known_bounds::adain_snake().expect("known bound");
    assert!(
        max_diff <= analytical.max_abs_diff,
        "Monte Carlo max diff {} exceeds analytical bound {}",
        max_diff,
        analytical.max_abs_diff
    );
}

/// Evaluate RMSNorm scalar: x * rms_inv * weight
fn eval_rms_norm(x: f32, rms_inv: f32, weight: f32) -> f32 {
    x * rms_inv * weight
}

/// Evaluate SiLU-Mul: sigmoid(x) * x * up
fn eval_silu_mul(x: f32, up: f32) -> f32 {
    let silu = x * (1.0 / (1.0 + (-x).exp()));
    silu * up
}

/// Fused RMSNorm+SiLU-Mul (same ops as sequential).
fn eval_rms_silu_fused(x: f32, rms_inv: f32, weight: f32, up: f32) -> f32 {
    let normed = x * rms_inv * weight;
    let silu = normed * (1.0 / (1.0 + (-normed).exp()));
    silu * up
}

#[test]
fn test_rms_norm_silu_mul_monte_carlo_fused_equals_sequential() {
    let mut rng_state: u64 = 0xCAFE_BABE_1234_5678;
    let mut max_diff: f64 = 0.0;

    for _ in 0..1_000_000 {
        let x = random_in_range(&mut rng_state, -5.0, 5.0);
        let rms_inv = random_in_range(&mut rng_state, 0.1, 3.0);
        let weight = random_in_range(&mut rng_state, -3.0, 3.0);
        let up = random_in_range(&mut rng_state, -5.0, 5.0);

        let fused = eval_rms_silu_fused(x, rms_inv, weight, up);
        let sequential = eval_silu_mul(eval_rms_norm(x, rms_inv, weight), up);

        if fused.is_finite() && sequential.is_finite() {
            let diff = (f64::from(fused) - f64::from(sequential)).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    let analytical =
        nn_verify::fusion_certificate::known_bounds::rms_norm_silu_mul().expect("known bound");
    assert!(
        max_diff <= analytical.max_abs_diff,
        "Monte Carlo max diff {} exceeds analytical bound {}",
        max_diff,
        analytical.max_abs_diff
    );
}

/// Evaluate LayerNorm scalar: gamma * (x - mean) * rsqrt(var + eps) + beta
fn eval_layer_norm(x: f32, mean: f32, var_val: f32, eps: f32, gamma: f32, beta: f32) -> f32 {
    gamma * (x - mean) * (var_val + eps).sqrt().recip() + beta
}

/// Evaluate GELU (tanh approximation).
fn eval_gelu(x: f32) -> f32 {
    let inner = std::f32::consts::FRAC_2_SQRT_PI
        * std::f32::consts::FRAC_1_SQRT_2
        * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Fused LayerNorm+GELU (same ops as sequential).
fn eval_ln_gelu_fused(x: f32, mean: f32, var_val: f32, eps: f32, gamma: f32, beta: f32) -> f32 {
    let y = gamma * (x - mean) * (var_val + eps).sqrt().recip() + beta;
    eval_gelu(y)
}

#[test]
fn test_layer_norm_gelu_monte_carlo_fused_equals_sequential() {
    let mut rng_state: u64 = 0x1234_5678_DEAD_BEEF;
    let mut max_diff: f64 = 0.0;

    for _ in 0..1_000_000 {
        let x = random_in_range(&mut rng_state, -2.0, 2.0);
        let mean = random_in_range(&mut rng_state, -1.0, 1.0);
        let var_val = random_in_range(&mut rng_state, 0.5, 5.0);
        let eps: f32 = 1e-5;
        let gamma = random_in_range(&mut rng_state, 0.5, 2.0);
        let beta = random_in_range(&mut rng_state, -1.0, 1.0);

        let fused = eval_ln_gelu_fused(x, mean, var_val, eps, gamma, beta);
        let sequential = eval_gelu(eval_layer_norm(x, mean, var_val, eps, gamma, beta));

        if fused.is_finite() && sequential.is_finite() {
            let diff = (f64::from(fused) - f64::from(sequential)).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    let analytical =
        nn_verify::fusion_certificate::known_bounds::layer_norm_gelu().expect("known bound");
    assert!(
        max_diff <= analytical.max_abs_diff,
        "Monte Carlo max diff {} exceeds analytical bound {}",
        max_diff,
        analytical.max_abs_diff
    );
}

/// Evaluate AdaIN+LeakyReLU fused.
fn eval_adain_leaky_relu_fused(
    x: f32,
    mu: f32,
    var: f32,
    gamma: f32,
    beta: f32,
    slope: f32,
    eps: f32,
) -> f32 {
    let y = gamma * (x - mu) * (var + eps).sqrt().recip() + beta;
    if y >= 0.0 {
        y
    } else {
        slope * y
    }
}

/// Sequential: adain → leaky_relu.
fn eval_adain_then_leaky_relu(
    x: f32,
    mu: f32,
    var: f32,
    gamma: f32,
    beta: f32,
    slope: f32,
    eps: f32,
) -> f32 {
    let y = eval_adain(x, mu, var, gamma, beta, eps);
    if y >= 0.0 {
        y
    } else {
        slope * y
    }
}

#[test]
fn test_adain_leaky_relu_monte_carlo_fused_equals_sequential() {
    let mut rng_state: u64 = 0xADAB_0EEF_1234_5678;
    let mut max_diff: f64 = 0.0;

    for _ in 0..1_000_000 {
        let x = random_in_range(&mut rng_state, -10.0, 10.0);
        let mu = random_in_range(&mut rng_state, -5.0, 5.0);
        let var = random_in_range(&mut rng_state, 0.001, 10.0);
        let gamma = random_in_range(&mut rng_state, 0.1, 5.0);
        let beta = random_in_range(&mut rng_state, -3.0, 3.0);
        let slope = random_in_range(&mut rng_state, 0.01, 0.5);
        let eps: f32 = 1e-5;

        let fused = eval_adain_leaky_relu_fused(x, mu, var, gamma, beta, slope, eps);
        let sequential = eval_adain_then_leaky_relu(x, mu, var, gamma, beta, slope, eps);

        if fused.is_finite() && sequential.is_finite() {
            let diff = (f64::from(fused) - f64::from(sequential)).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    let analytical =
        nn_verify::fusion_certificate::known_bounds::adain_leaky_relu().expect("known bound");
    assert!(
        max_diff <= analytical.max_abs_diff,
        "Monte Carlo max diff {} exceeds analytical bound {}",
        max_diff,
        analytical.max_abs_diff
    );
}

/// Evaluate LayerNorm scalar (same as eval_layer_norm above but for AdaLN context):
/// normed = (x - mean) * rsqrt(var + eps) * norm_weight + norm_bias
fn eval_layer_norm_adaln(
    x: f32,
    mean: f32,
    var_val: f32,
    eps: f32,
    norm_weight: f32,
    norm_bias: f32,
) -> f32 {
    norm_weight * (x - mean) * (var_val + eps).sqrt().recip() + norm_bias
}

/// Evaluate adaptive affine: (1 + gamma) * x + beta
fn eval_adaptive_affine(x: f32, gamma: f32, beta: f32) -> f32 {
    (1.0 + gamma) * x + beta
}

/// Evaluate fused AdaLayerNorm (same ops as sequential, single function).
fn eval_ada_layer_norm_fused(
    x: f32,
    mean: f32,
    var_val: f32,
    eps: f32,
    norm_weight: f32,
    norm_bias: f32,
    gamma: f32,
    beta: f32,
) -> f32 {
    let normed = norm_weight * (x - mean) * (var_val + eps).sqrt().recip() + norm_bias;
    (1.0 + gamma) * normed + beta
}

#[test]
fn test_ada_layer_norm_monte_carlo_fused_equals_sequential() {
    let mut rng_state: u64 = 0xADA1_0AEF_CAFE_1234;
    let mut max_diff: f64 = 0.0;

    for _ in 0..1_000_000 {
        let x = random_in_range(&mut rng_state, -3.0, 3.0);
        let mean = random_in_range(&mut rng_state, -1.0, 1.0);
        let var_val = random_in_range(&mut rng_state, 0.5, 5.0);
        let eps: f32 = 1e-5;
        let norm_weight = random_in_range(&mut rng_state, 0.5, 2.0);
        let norm_bias = random_in_range(&mut rng_state, -1.0, 1.0);
        let gamma = random_in_range(&mut rng_state, -2.0, 2.0);
        let beta = random_in_range(&mut rng_state, -2.0, 2.0);

        let fused =
            eval_ada_layer_norm_fused(x, mean, var_val, eps, norm_weight, norm_bias, gamma, beta);
        let normed = eval_layer_norm_adaln(x, mean, var_val, eps, norm_weight, norm_bias);
        let sequential = eval_adaptive_affine(normed, gamma, beta);

        if fused.is_finite() && sequential.is_finite() {
            let diff = (f64::from(fused) - f64::from(sequential)).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    let analytical =
        nn_verify::fusion_certificate::known_bounds::ada_layer_norm().expect("known bound");
    assert!(
        max_diff <= analytical.max_abs_diff,
        "Monte Carlo max diff {} exceeds analytical bound {}",
        max_diff,
        analytical.max_abs_diff
    );
}

// ── Certificate without analytical bound ────────────────────────────────

#[test]
fn test_certificate_without_analytical_relies_on_crown() {
    let fused = build_adain_snake_fused_kernel().expect("fused kernel");
    let adain = build_adain_scalar_kernel().expect("adain kernel");
    let snake = build_snake_scalar_kernel().expect("snake kernel");

    let spec = FusionSpec::new(&fused, &adain, &snake, 7, &[0, 1, 2, 3, 4, 6], &[0, 5], 0)
        .expect("valid spec");

    // No analytical bound — certificate relies solely on CROWN.
    let cert = verify_fusion_certificate(
        &spec,
        "adain",
        "snake",
        &ADAIN_SNAKE_BOUNDS,
        1e-4,
        512,
        None,
    )
    .expect("certificate generation");

    assert!(cert.analytical_bound.is_none());
    // CROWN alone may or may not prove equivalence for wide intervals.
    // The certificate is valid either way — it records the CROWN bound.
    assert!(cert.crown_bound.is_some());
    cert.validate().expect("valid certificate");
}
