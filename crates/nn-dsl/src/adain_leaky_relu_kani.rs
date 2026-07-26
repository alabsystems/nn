// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for fused AdaIN+LeakyReLU scalar.
//!
//! The fused AdaIN+LeakyReLU kernel is used in Kokoro's decoder FusedResBlock
//! (3 blocks × 2 layers = 6 invocations per forward pass). LeakyReLU slope
//! is 0.2 in production.
//!
//! ```text
//! y = gamma * (x - mu) * rsqrt(var + eps) + beta   // AdaIN
//! output = if y >= 0 { y } else { slope * y }       // LeakyReLU
//! ```
//!
//! These harnesses prove:
//! 1. Fused/sequential equivalence for symbolic x and slope
//! 2. Fused/sequential equivalence with symbolic style params
//! 3. Full 7-variable safety with all params symbolic
//! 4. Guard completeness: non-finite input → Err, Ok → finite (2^7-1 cases)
//!
//! Mirrors `adain_kani.rs` (AdaIN+Snake) structure. Part of #2218.

use super::*;

/// Asserts safety invariants shared by fused and sequential AdaIN+LeakyReLU paths.
///
/// Unlike AdaIN+Snake (where output >= y due to sin²), LeakyReLU is piecewise
/// linear: output == y when y >= 0, output == slope * y when y < 0.
///
/// Invariants proved:
/// - AdaIN intermediate `y` is finite
/// - Both `fused` and `sequential` outputs are finite
/// - Both outputs agree (fused == sequential to the bit)
/// - For non-negative slope: |output| <= max(|y|, |slope * y|)
fn assert_adain_leaky_relu_safety_invariants(y: f32, fused: f32, sequential: f32, slope: f32) {
    assert!(y.is_finite(), "AdaIN output must remain finite");
    assert!(fused.is_finite(), "fused output must remain finite");
    assert!(
        sequential.is_finite(),
        "sequential output must remain finite"
    );

    // Fused and sequential compute identical operations: adain then leaky_relu.
    // No transcendental functions involved (unlike Snake's sin²), so bitwise
    // equality holds.
    assert!(
        fused.to_bits() == sequential.to_bits(),
        "fused and sequential must be bitwise equal"
    );

    // For non-negative slope (production range), LeakyReLU is non-expansive:
    // the output magnitude never exceeds the input magnitude.
    if slope >= 0.0 && slope <= 1.0 {
        assert!(
            fused.abs() <= y.abs() + f32::EPSILON,
            "LeakyReLU with slope in [0,1] is non-expansive"
        );
    }
}

/// Proves fused/sequential AdaIN+LeakyReLU equivalence for symbolic x and slope,
/// with other parameters fixed to representative values.
///
/// SUBSTANTIVE: proves that fused and sequential paths produce bitwise-identical
/// output for all x in [-10, 10] and slope in [0, 1] (production range).
/// Also proves finiteness and non-expansiveness.
///
/// Covers: `adain.rs` `adain_leaky_relu_fused_scalar` (lines 220-233).
#[kani::unwind(1)]
#[kani::proof]
fn adain_leaky_relu_safety_x_slope() {
    let x: f32 = kani::any();
    let slope: f32 = kani::any();

    kani::assume(x.is_finite());
    kani::assume(slope.is_finite());
    kani::assume(x >= -10.0 && x <= 10.0);
    kani::assume(slope >= 0.0 && slope <= 1.0);

    let mu = 0.0_f32;
    let var_val = 1.0_f32;
    let gamma = 1.0_f32;
    let beta = 0.0_f32;
    let eps = 1e-5_f32;

    let fused = adain_leaky_relu_fused_scalar(x, mu, var_val, gamma, beta, slope, eps)
        .expect("invariant: var_val + eps > 0 under kani::assume");
    let y = adain_scalar(x, mu, var_val, gamma, beta, eps)
        .expect("invariant: var_val + eps > 0 under kani::assume");
    let sequential =
        leaky_relu_scalar(y, slope).expect("invariant: finite y and slope under kani::assume");

    assert_adain_leaky_relu_safety_invariants(y, fused, sequential, slope);
}

/// Proves fused/sequential equivalence with symbolic style params.
///
/// Exercises the AdaIN normalization dimension: gamma/beta are the affine
/// transform that style-conditions the output. Fixed mu=0, var=1, eps=1e-5
/// for tractability.
///
/// SUBSTANTIVE: proves bitwise equivalence and finiteness for all combinations
/// of x, gamma, beta, slope in Kokoro-realistic ranges.
///
/// Covers: `adain.rs` `adain_leaky_relu_fused_scalar` (lines 220-233).
#[kani::unwind(1)]
#[kani::proof]
fn adain_leaky_relu_safety_style_params() {
    let x: f32 = kani::any();
    let gamma: f32 = kani::any();
    let beta: f32 = kani::any();
    let slope: f32 = kani::any();

    kani::assume(x.is_finite());
    kani::assume(gamma.is_finite());
    kani::assume(beta.is_finite());
    kani::assume(slope.is_finite());
    kani::assume(x >= -5.0 && x <= 5.0);
    kani::assume(gamma >= -3.0 && gamma <= 3.0);
    kani::assume(beta >= -3.0 && beta <= 3.0);
    kani::assume(slope >= 0.0 && slope <= 1.0);

    let mu = 0.0_f32;
    let var_val = 1.0_f32;
    let eps = 1e-5_f32;

    let fused = adain_leaky_relu_fused_scalar(x, mu, var_val, gamma, beta, slope, eps)
        .expect("invariant: var_val + eps > 0 under kani::assume");
    let y = adain_scalar(x, mu, var_val, gamma, beta, eps)
        .expect("invariant: var_val + eps > 0 under kani::assume");
    let sequential =
        leaky_relu_scalar(y, slope).expect("invariant: finite y and slope under kani::assume");

    assert_adain_leaky_relu_safety_invariants(y, fused, sequential, slope);
}

/// Full 7-variable fused/sequential safety proof: all params symbolic.
///
/// This checks the full Kokoro parameter domain while preserving the
/// var_val + eps > 0 requirement for AdaIN. Slope is bounded to [0, 1]
/// (production LeakyReLU always uses slope=0.2).
///
/// May require extended Kani timeout due to the 7-variable state space.
///
/// SUBSTANTIVE: proves bitwise equivalence and finiteness for all finite
/// input combinations in bounded Kokoro ranges.
///
/// Covers: `adain.rs` `adain_leaky_relu_fused_scalar` (lines 220-233).
#[kani::unwind(1)]
#[kani::proof]
fn adain_leaky_relu_safety_all_params() {
    let x: f32 = kani::any();
    let mu: f32 = kani::any();
    let var_val: f32 = kani::any();
    let gamma: f32 = kani::any();
    let beta: f32 = kani::any();
    let slope: f32 = kani::any();
    let eps: f32 = kani::any();

    kani::assume(x.is_finite());
    kani::assume(mu.is_finite());
    kani::assume(var_val.is_finite());
    kani::assume(gamma.is_finite());
    kani::assume(beta.is_finite());
    kani::assume(slope.is_finite());
    kani::assume(eps.is_finite());

    kani::assume(x >= -5.0 && x <= 5.0);
    kani::assume(mu >= -5.0 && mu <= 5.0);
    kani::assume(var_val >= 1e-4 && var_val <= 5.0);
    kani::assume(gamma >= -3.0 && gamma <= 3.0);
    kani::assume(beta >= -3.0 && beta <= 3.0);
    kani::assume(slope >= 0.0 && slope <= 1.0);
    kani::assume(eps >= 1e-8 && eps <= 1e-3);

    let fused = adain_leaky_relu_fused_scalar(x, mu, var_val, gamma, beta, slope, eps)
        .expect("invariant: var_val + eps > 0 under kani::assume");
    let y = adain_scalar(x, mu, var_val, gamma, beta, eps)
        .expect("invariant: var_val + eps > 0 under kani::assume");
    let sequential =
        leaky_relu_scalar(y, slope).expect("invariant: finite y and slope under kani::assume");

    assert_adain_leaky_relu_safety_invariants(y, fused, sequential, slope);
}

/// Proves guard completeness: non-finite input always rejected, Ok always finite.
///
/// With 7 parameters, there are `2^7 - 1 = 127` combinations containing at
/// least one non-finite value. Kani symbolically explores all of them.
///
/// SUBSTANTIVE: proves `validate_finite_inputs` has no gaps for the full
/// 7-parameter call. Also proves `checked_scalar_output` guarantee: every
/// `Ok(val)` is finite.
///
/// Note: CBMC models `f32::sqrt` nondeterministically, so finite inputs
/// may still produce `Err` in this harness (valid defense-in-depth). The
/// harness does NOT assert that finite inputs always succeed — only that
/// non-finite inputs always fail and Ok values are always finite.
///
/// Covers: `adain.rs` `adain_leaky_relu_fused_scalar` (lines 220-233)
///         and `leaky_relu_scalar` (lines 206-210).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)] // validate_finite_inputs loops over up to 7 elements
fn adain_leaky_relu_guard_all_params() {
    let x: f32 = kani::any();
    let mu: f32 = kani::any();
    let var_val: f32 = kani::any();
    let gamma: f32 = kani::any();
    let beta: f32 = kani::any();
    let slope: f32 = kani::any();
    let eps: f32 = kani::any();

    let result = adain_leaky_relu_fused_scalar(x, mu, var_val, gamma, beta, slope, eps);

    let all_finite = x.is_finite()
        && mu.is_finite()
        && var_val.is_finite()
        && gamma.is_finite()
        && beta.is_finite()
        && slope.is_finite()
        && eps.is_finite();

    // Non-finite input must always be rejected.
    if !all_finite {
        assert!(result.is_err(), "non-finite input must produce Err");
    }

    // Ok result must always be finite (checked_scalar_output guarantee).
    if let Ok(val) = &result {
        assert!(val.is_finite(), "Ok result must be finite");
    }
}
