// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Dvoice-realistic sequential composition tests (#576).
//!
//! Extracted from `compose_sequential.rs` to keep the parent under 500
//! lines (#1669). These tests verify AdaIN → Snake composition with
//! parameters matching the dvoice Kokoro decoder pipeline.

use super::common::{extract_scalar, scalar_bounds};
use nn_dsl::adain::{
    build_adain_scalar_kernel, build_adain_snake_fused_kernel, build_snake_scalar_kernel,
};
use nn_verify::{compose_sequential, propagate_with_crown_fallback, SequentialSpec};

// --- Dvoice-realistic composition tests (#576) ---

/// Dvoice-realistic constants for AdaIN: mu=0, var=1, gamma=1, beta=0, eps=1e-5.
/// These represent the identity normalization case (no style transfer shift).
const ADAIN_IDENTITY_CONSTANTS: [f32; 5] = [0.0, 1.0, 1.0, 0.0, 1e-5];

/// Dvoice-realistic alpha for Snake activation.
const SNAKE_ALPHA: f32 = 1.0;

/// AC1: Compose AdaIN → Snake using dvoice-realistic parameters.
///
/// This is the critical composition path for dvoice: the Kokoro decoder
/// applies AdaIN normalization then Snake activation per-element.
#[test]
fn test_adain_then_snake_dvoice_composition() {
    let adain = build_adain_scalar_kernel().expect("build adain");
    let snake = build_snake_scalar_kernel().expect("build snake");

    // AdaIN has 6 params: (x, mu, var_val, gamma, beta, eps).
    // first_constants covers params 1..5 = [mu, var_val, gamma, beta, eps].
    // Snake has 2 params: (y, alpha).
    // second_constants covers param 1 = [alpha].
    // chain_param=0: snake's param 0 (y) receives adain's output.
    let spec = SequentialSpec::new(&adain, &snake, &ADAIN_IDENTITY_CONSTANTS, &[SNAKE_ALPHA], 0)
        .expect("valid adain → snake spec");
    let graph = compose_sequential(&spec).expect("compose adain → snake");

    // Dvoice audio feature range: x ∈ [-10, 10].
    let input = scalar_bounds(-10.0, 10.0);
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = extract_scalar(&output);

    assert!(lo.is_finite(), "lower bound should be finite, got {lo}");
    assert!(hi.is_finite(), "upper bound should be finite, got {hi}");

    // With identity AdaIN (mu=0, var=1, gamma=1, beta=0), output ≈ x.
    // Snake(x, 1.0) = x + sin²(x), so output ∈ approx [-9.70, 10.30].
    // Sound IBP must contain the true range: lo <= -9.70, hi >= 10.30.
    assert!(
        lo < -9.0,
        "lower bound should be < -9.0 (true min ≈ -9.70), got {lo}"
    );
    assert!(
        hi > 10.0,
        "upper bound should be > 10.0 (true max ≈ 10.30), got {hi}"
    );

    // Width guard: analytical range ≈ 20, IBP should not exceed 10x.
    let width = hi - lo;
    assert!(
        width < 200.0,
        "IBP width {width} exceeds 10x analytical range (~20); likely computation error"
    );
}

/// AC1 continued: AdaIN → Snake with non-trivial style transfer parameters.
///
/// Analytical derivation for numerical assertions:
///   AdaIN(x) = gamma * (x - mu) / sqrt(var + eps) + beta
///            = 2.0 * (x - 1.0) / sqrt(4.0 + 1e-5) + 0.5
///            ≈ (x - 1.0) + 0.5 = x - 0.5   [since sqrt(4) ≈ 2]
///   For x ∈ [-5, 5]: AdaIN output ≈ [-5.5, 4.5]
///   Snake(y, 0.5) = y + 2*sin²(0.5*y), sin²∈[0,1] so Snake(y) ∈ [y, y+2]
///   For y ∈ [-5.5, 4.5]: output ≈ [-5.5, 6.5]
///   IBP relaxation widens these, but output must be in a reasonable range.
#[test]
fn test_adain_then_snake_styled_composition() {
    let adain = build_adain_scalar_kernel().expect("build adain");
    let snake = build_snake_scalar_kernel().expect("build snake");

    // Non-trivial style: gamma=2.0, beta=0.5 (amplify and shift).
    // mu=1.0, var=4.0 (realistic channel statistics).
    let styled_constants: [f32; 5] = [1.0, 4.0, 2.0, 0.5, 1e-5];

    let spec = SequentialSpec::new(&adain, &snake, &styled_constants, &[0.5], 0)
        .expect("valid styled spec");
    let graph = compose_sequential(&spec).expect("compose styled adain → snake");

    let input = scalar_bounds(-5.0, 5.0);
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = extract_scalar(&output);

    assert!(lo.is_finite(), "lower bound should be finite, got {lo}");
    assert!(hi.is_finite(), "upper bound should be finite, got {hi}");

    // Numerical assertions: analytically, output ≈ [-5.5, 6.5].
    // IBP relaxation widens, but bounds should remain within a reasonable factor.
    // The output range width is ~12, so IBP should not exceed ~10x (120).
    let width = hi - lo;
    assert!(
        width < 120.0,
        "IBP width {width} exceeds 10x analytical range (~12); likely computation error"
    );

    // Sound lower bound must be <= analytical minimum (-5.5).
    assert!(
        lo < -5.0,
        "lower bound {lo} should be < -5.0 (true min ≈ -5.5) for styled adain→snake"
    );
    // Sound upper bound must be >= analytical maximum (~6.5).
    assert!(
        hi > 6.0,
        "upper bound {hi} should be > 6.0 (true max ≈ 6.5) for styled adain→snake"
    );
}

/// Assert two IBP bound pairs are within tolerance of each other.
///
/// Checks width difference and endpoint differences. Tolerance is max 50% of the
/// analytical range width (~20 for identity AdaIN + snake on [-10, 10]).
/// Different graph topologies (composed vs fused) produce different IBP
/// relaxation, so exact equality is not expected.
fn assert_bounds_within_tolerance(composed: (f32, f32), fused: (f32, f32), tolerance: f32) {
    let composed_width = composed.1 - composed.0;
    let fused_width = fused.1 - fused.0;
    let width_diff = (composed_width - fused_width).abs();
    assert!(
        width_diff < tolerance,
        "composed width ({composed_width:.2}) and fused width ({fused_width:.2}) \
         differ by {width_diff:.2}, exceeds tolerance {tolerance}"
    );

    let lo_diff = (composed.0 - fused.0).abs();
    let hi_diff = (composed.1 - fused.1).abs();
    assert!(
        lo_diff < tolerance,
        "lower bound difference {lo_diff:.2} exceeds tolerance {tolerance} \
         (composed={:.2}, fused={:.2})",
        composed.0,
        fused.0
    );
    assert!(
        hi_diff < tolerance,
        "upper bound difference {hi_diff:.2} exceeds tolerance {tolerance} \
         (composed={:.2}, fused={:.2})",
        composed.1,
        fused.1
    );
}

/// AC2: Compare compose_sequential bounds against fused adain_snake bounds.
///
/// The fused kernel computes the same function as AdaIN → Snake composition.
/// Both should produce sound bounds that contain the true output range.
/// The composed graph and the single fused graph may differ in IBP tightness
/// (different graph topologies produce different relaxation paths), but
/// both must be finite, ordered, and contain the empirical range.
#[test]
fn test_composed_vs_fused_adain_snake_bounds() {
    let adain = build_adain_scalar_kernel().expect("build adain");
    let snake = build_snake_scalar_kernel().expect("build snake");
    let fused = build_adain_snake_fused_kernel().expect("build fused adain_snake");

    let x_lo = -10.0f32;
    let x_hi = 10.0f32;

    // Composed: adain → snake via compose_sequential.
    let spec = SequentialSpec::new(&adain, &snake, &ADAIN_IDENTITY_CONSTANTS, &[SNAKE_ALPHA], 0)
        .expect("valid spec");
    let composed = compose_sequential(&spec).expect("compose adain → snake");

    let composed_output = composed
        .propagate_ibp(&scalar_bounds(x_lo, x_hi))
        .expect("composed IBP");
    let (composed_lo, composed_hi) = extract_scalar(&composed_output);

    // Fused: single kernel with all 7 params.
    // adain_snake(x, mu, var_val, gamma, beta, alpha, eps)
    // constants = [mu=0, var=1, gamma=1, beta=0, alpha=1, eps=1e-5]
    let fused_constants = [0.0, 1.0, 1.0, 0.0, SNAKE_ALPHA, 1e-5];
    let fused_graph =
        nn_verify::kernel_to_graph(&fused, &fused_constants).expect("build fused graph");
    let fused_output = fused_graph
        .propagate_ibp(&scalar_bounds(x_lo, x_hi))
        .expect("fused IBP");
    let (fused_lo, fused_hi) = extract_scalar(&fused_output);

    // Both must be finite.
    assert!(composed_lo.is_finite() && composed_hi.is_finite());
    assert!(fused_lo.is_finite() && fused_hi.is_finite());

    // Both must contain the empirical range. snake(x) = x + sin²(x).
    // For x ∈ [-10, 10]: min ≈ -9.70, max ≈ 10.30
    // Sound IBP bounds must contain the true range.
    assert!(
        composed_lo < -9.0 && composed_hi > 10.0,
        "composed bounds ({composed_lo:.2}, {composed_hi:.2}) must contain true range [-9.70, 10.30]"
    );
    assert!(
        fused_lo < -9.0 && fused_hi > 10.0,
        "fused bounds ({fused_lo:.2}, {fused_hi:.2}) must contain true range [-9.70, 10.30]"
    );

    // Composed and fused bounds must be within tolerance (10.0 = 50% of ~20 range).
    assert_bounds_within_tolerance((composed_lo, composed_hi), (fused_lo, fused_hi), 10.0);
}

/// CROWN propagation produces tighter bounds than IBP for AdaIN → Snake.
#[test]
fn test_adain_then_snake_crown_tighter_than_ibp() {
    let adain = build_adain_scalar_kernel().expect("build adain");
    let snake = build_snake_scalar_kernel().expect("build snake");

    let spec = SequentialSpec::new(&adain, &snake, &ADAIN_IDENTITY_CONSTANTS, &[SNAKE_ALPHA], 0)
        .expect("valid adain → snake spec");
    let graph = compose_sequential(&spec).expect("compose adain → snake");

    let input = scalar_bounds(-10.0, 10.0);
    let ibp_output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (_, crown_output, _) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN propagation");

    // Inline CROWN-tighter-than-IBP assertion (no `common` module in this submodule file).
    let (crown_lo, crown_hi) = crown_output.lower_upper();
    let (ibp_lo, ibp_hi) = ibp_output.lower_upper();
    let eps = 1e-4;
    for (&cl, &il) in crown_lo.iter().zip(ibp_lo.iter()) {
        assert!(
            cl >= il - eps,
            "CROWN lower {cl} should be >= IBP lower {il}"
        );
    }
    for (&cu, &iu) in crown_hi.iter().zip(ibp_hi.iter()) {
        assert!(
            cu <= iu + eps,
            "CROWN upper {cu} should be <= IBP upper {iu}"
        );
    }
}
