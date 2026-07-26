// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::snake::snake_scalar;

/// Asserts safety invariants shared by fused and sequential AdaIN+Snake paths.
///
/// NOTE: This does NOT assert equivalence (fused == sequential). It asserts
/// that both paths independently satisfy safety properties (finiteness,
/// lower bound). A true fusion equivalence check would compare
/// `fused.to_bits() == sequential.to_bits()` or `(fused - sequential).abs() <= eps`.
fn assert_adain_snake_safety_invariants(y: f32, fused: f32, sequential: f32, alpha: f32) {
    let safe_alpha = alpha.max(SNAKE_MIN_ALPHA);

    assert!(
        safe_alpha >= SNAKE_MIN_ALPHA && safe_alpha.is_finite(),
        "alpha clamp must enforce a finite denominator floor"
    );
    assert!(y.is_finite(), "AdaIN output must remain finite");
    assert!(fused.is_finite(), "fused output must remain finite");
    assert!(
        sequential.is_finite(),
        "sequential output must remain finite"
    );

    // Snake adds sin²(a*y)/a with a > 0, so output should not drop below y.
    assert!(fused >= y, "fused output must be >= AdaIN output");
    assert!(sequential >= y, "sequential output must be >= AdaIN output");
}

/// Proves fused/sequential AdaIN+Snake safety invariants for symbolic x and alpha,
/// with other parameters fixed to representative values.
///
/// NOTE: Does not prove fusion equivalence (fused == sequential). Proves only
/// that both paths produce finite output >= the AdaIN intermediate. Renamed
/// from fusion_equivalence_x_alpha per #97 naming rule.
#[kani::unwind(1)]
#[kani::proof]
fn adain_snake_safety_x_alpha() {
    let x: f32 = kani::any();
    let alpha: f32 = kani::any();

    kani::assume(x.is_finite());
    kani::assume(alpha.is_finite());
    kani::assume(x >= -10.0 && x <= 10.0);
    kani::assume(alpha >= 0.0 && alpha <= 10.0);

    let mu = 0.0_f32;
    let var_val = 1.0_f32;
    let gamma = 1.0_f32;
    let beta = 0.0_f32;
    let eps = 1e-5_f32;

    let fused = adain_snake_fused_scalar(x, mu, var_val, gamma, beta, alpha, eps)
        .expect("invariant: var_val + eps > 0 under kani::assume");
    let y = adain_scalar(x, mu, var_val, gamma, beta, eps)
        .expect("invariant: var_val + eps > 0 under kani::assume");
    let sequential =
        snake_scalar(y, alpha).expect("invariant: finite y and alpha under kani::assume");

    assert_adain_snake_safety_invariants(y, fused, sequential, alpha);
}

/// Proves fused/sequential safety invariants with symbolic style params.
///
/// Exercises the AdaIN normalization dimension: gamma/beta are the affine
/// transform that style-conditions the output. Fixed mu=0, var=1, eps=1e-5
/// for tractability.
///
/// NOTE: Does not prove fusion equivalence. Renamed from
/// fusion_equivalence_style_params per #97 naming rule.
#[kani::unwind(1)]
#[kani::proof]
fn adain_snake_safety_style_params() {
    let x: f32 = kani::any();
    let gamma: f32 = kani::any();
    let beta: f32 = kani::any();
    let alpha: f32 = kani::any();

    kani::assume(x.is_finite());
    kani::assume(gamma.is_finite());
    kani::assume(beta.is_finite());
    kani::assume(alpha.is_finite());
    kani::assume(x >= -5.0 && x <= 5.0);
    kani::assume(gamma >= -3.0 && gamma <= 3.0);
    kani::assume(beta >= -3.0 && beta <= 3.0);
    kani::assume(alpha >= 0.0 && alpha <= 5.0);

    let mu = 0.0_f32;
    let var_val = 1.0_f32;
    let eps = 1e-5_f32;

    let fused = adain_snake_fused_scalar(x, mu, var_val, gamma, beta, alpha, eps)
        .expect("invariant: var_val + eps > 0 under kani::assume");
    let y = adain_scalar(x, mu, var_val, gamma, beta, eps)
        .expect("invariant: var_val + eps > 0 under kani::assume");
    let sequential =
        snake_scalar(y, alpha).expect("invariant: finite y and alpha under kani::assume");

    assert_adain_snake_safety_invariants(y, fused, sequential, alpha);
}

/// Full 7-variable fused/sequential safety proof: all params symbolic.
///
/// This checks the full dvoice parameter domain while preserving the
/// alpha-clamp safety guarantees introduced for #97.
/// May require extended Kani timeout due to the 7-variable state space.
///
/// NOTE: Does not prove fusion equivalence. Renamed from
/// fusion_equivalence_all_params per #97 naming rule.
#[kani::unwind(1)]
#[kani::proof]
fn adain_snake_safety_all_params() {
    let x: f32 = kani::any();
    let mu: f32 = kani::any();
    let var_val: f32 = kani::any();
    let gamma: f32 = kani::any();
    let beta: f32 = kani::any();
    let alpha: f32 = kani::any();
    let eps: f32 = kani::any();

    kani::assume(x.is_finite());
    kani::assume(mu.is_finite());
    kani::assume(var_val.is_finite());
    kani::assume(gamma.is_finite());
    kani::assume(beta.is_finite());
    kani::assume(alpha.is_finite());
    kani::assume(eps.is_finite());

    kani::assume(x >= -5.0 && x <= 5.0);
    kani::assume(mu >= -5.0 && mu <= 5.0);
    kani::assume(var_val >= 1e-4 && var_val <= 5.0);
    kani::assume(gamma >= -3.0 && gamma <= 3.0);
    kani::assume(beta >= -3.0 && beta <= 3.0);
    kani::assume(alpha >= 0.0 && alpha <= 5.0);
    kani::assume(eps >= 1e-8 && eps <= 1e-3);

    let fused = adain_snake_fused_scalar(x, mu, var_val, gamma, beta, alpha, eps)
        .expect("invariant: var_val + eps > 0 under kani::assume");
    let y = adain_scalar(x, mu, var_val, gamma, beta, eps)
        .expect("invariant: var_val + eps > 0 under kani::assume");
    let sequential =
        snake_scalar(y, alpha).expect("invariant: finite y and alpha under kani::assume");

    assert_adain_snake_safety_invariants(y, fused, sequential, alpha);
}
