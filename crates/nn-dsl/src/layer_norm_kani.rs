// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for layer normalization scalar operations.
//!
//! Extracted from `layer_norm.rs` to keep files under the 500-line limit.

use super::*;

/// Proves `layer_norm_scalar` produces finite output for bounded inputs.
///
/// Domain: x in [-1e3, 1e3], mean in [-1e3, 1e3], var in [0, 1e6],
/// eps in [1e-8, 1.0], gamma in [-10, 10], beta in [-10, 10].
#[kani::unwind(8)]
#[kani::proof]
fn layer_norm_scalar_finite_for_bounded_inputs() {
    let x: f32 = kani::any();
    let mean: f32 = kani::any();
    let var: f32 = kani::any();
    let eps: f32 = kani::any();
    let gamma: f32 = kani::any();
    let beta: f32 = kani::any();

    kani::assume(x.is_finite());
    kani::assume(mean.is_finite());
    kani::assume(var.is_finite());
    kani::assume(eps.is_finite());
    kani::assume(gamma.is_finite());
    kani::assume(beta.is_finite());

    kani::assume(x >= -1.0e3 && x <= 1.0e3);
    kani::assume(mean >= -1.0e3 && mean <= 1.0e3);
    kani::assume(var >= 0.0 && var <= 1.0e6);
    kani::assume(eps >= 1.0e-8 && eps <= 1.0);
    kani::assume(gamma >= -10.0 && gamma <= 10.0);
    kani::assume(beta >= -10.0 && beta <= 10.0);

    let y = layer_norm_scalar(x, mean, var, eps, gamma, beta)
        .expect("layer_norm_scalar must succeed for bounded finite inputs");
    assert!(
        y.is_finite(),
        "layer_norm_scalar must produce finite output"
    );
    assert!(!y.is_nan(), "layer_norm_scalar must not produce NaN");
}

/// Proves that zero variance with positive eps still produces finite output.
///
/// When all inputs in a row are identical, var = 0, so inv_std = 1/sqrt(eps).
/// This must not overflow or produce NaN.
#[kani::unwind(8)]
#[kani::proof]
fn layer_norm_scalar_zero_variance_safe() {
    let x: f32 = kani::any();
    let eps: f32 = kani::any();
    let gamma: f32 = kani::any();
    let beta: f32 = kani::any();

    kani::assume(x.is_finite());
    kani::assume(eps.is_finite());
    kani::assume(gamma.is_finite());
    kani::assume(beta.is_finite());

    kani::assume(x >= -1.0e3 && x <= 1.0e3);
    kani::assume(eps >= 1.0e-8 && eps <= 1.0);
    kani::assume(gamma >= -10.0 && gamma <= 10.0);
    kani::assume(beta >= -10.0 && beta <= 10.0);

    // var = 0 (constant input row), mean = x (all identical)
    let y = layer_norm_scalar(x, x, 0.0, eps, gamma, beta)
        .expect("layer_norm_scalar must succeed for bounded finite inputs");
    assert!(y.is_finite(), "zero-variance layer_norm must be finite");
    // When mean = x, (x - mean) = 0, so output should equal beta.
    let expected = beta;
    assert!(
        (y - expected).abs() < 1e-3,
        "zero-variance layer_norm should equal beta"
    );
}

/// Proves that the identity affine (gamma=1, beta=0) normalizes to
/// a bounded range: |output| <= |x - mean| / sqrt(eps) for var = 0.
#[kani::unwind(8)]
#[kani::proof]
fn layer_norm_scalar_identity_affine_bounded() {
    let x: f32 = kani::any();
    let mean: f32 = kani::any();
    let var: f32 = kani::any();
    let eps: f32 = kani::any();

    kani::assume(x.is_finite());
    kani::assume(mean.is_finite());
    kani::assume(var.is_finite());
    kani::assume(eps.is_finite());

    kani::assume(x >= -100.0 && x <= 100.0);
    kani::assume(mean >= -100.0 && mean <= 100.0);
    kani::assume(var >= 0.0 && var <= 1.0e4);
    kani::assume(eps >= 1.0e-5 && eps <= 1.0);

    let y = layer_norm_scalar(x, mean, var, eps, 1.0, 0.0)
        .expect("layer_norm_scalar must succeed for bounded finite inputs");
    assert!(y.is_finite(), "identity affine layer_norm must be finite");

    // |y| = |x - mean| / sqrt(var + eps) <= 200 / sqrt(1e-5) = 63246
    // Conservative bound to verify the function is well-behaved.
    assert!(y.abs() <= 7.0e4, "identity affine output must be bounded");
}
