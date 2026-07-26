// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for fused NormActivConv1d+Snake per-tap scalar.
//!
//! The fused NormActivConv1d+Snake kernel computes InstanceNorm + residual
//! affine + Snake activation + Conv1d weight multiply in a single GPU dispatch.
//!
//! ```text
//! normed = (x - mean) * inv_std
//! y = (1 + gamma) * normed + beta
//! a = max(alpha, SNAKE_MIN_ALPHA)
//! activated = y + (1/a) * sin^2(a * y)
//! contribution = activated * weight
//! ```
//!
//! NOTE: These harnesses prove SAFETY PROPERTIES ONLY, not bitwise equivalence.
//! Snake uses `sin()` which CBMC cannot model correctly (design doc #708).
//! CBMC models `f32::sin` nondeterministically, so fused and sequential paths
//! may produce different sin values even with identical inputs. Bitwise
//! equivalence for Snake variants relies on CROWN bounds (vacuous) or ULP
//! analysis (not applicable here due to weight multiply after activation).
//!
//! Safety properties proved:
//! 1. Both fused and sequential outputs are finite for valid inputs
//! 2. Guard completeness: non-finite input -> Err, Ok -> finite
//!
//! Mirrors `adain_kani.rs` (safety-only) structure. Part of #3020, #2218 F13.

use super::*;

/// Asserts safety invariants for fused and sequential Snake paths.
///
/// Unlike the LeakyReLU variant, this does NOT assert bitwise equivalence
/// because CBMC cannot model `sin()` deterministically. It proves only that
/// both paths produce finite results independently.
fn assert_norm_conv_snake_safety_invariants(fused: f32, sequential: f32) {
    assert!(fused.is_finite(), "fused output must remain finite");
    assert!(
        sequential.is_finite(),
        "sequential output must remain finite"
    );
}

/// Proves both fused and sequential paths produce finite output for symbolic
/// x and weight, with normalization and activation params fixed.
///
/// SUBSTANTIVE: proves finiteness for all x in [-10, 10], weight in [-3, 3],
/// with representative normalization parameters. Does NOT prove equivalence
/// (CBMC sin limitation).
///
/// Covers: `norm_activ_conv_kernels.rs` `norm_snake_mul_fused_scalar`.
#[kani::unwind(1)]
#[kani::proof]
fn norm_conv_snake_safety_x_weight() {
    let x: f32 = kani::any();
    let weight: f32 = kani::any();

    kani::assume(x.is_finite());
    kani::assume(weight.is_finite());
    kani::assume(x >= -10.0 && x <= 10.0);
    kani::assume(weight >= -3.0 && weight <= 3.0);

    let mean = 0.0_f32;
    let inv_std = 1.0_f32;
    let gamma = 0.0_f32;
    let beta = 0.0_f32;
    let alpha = 1.0_f32; // typical Kokoro alpha

    let fused = norm_snake_mul_fused_scalar(x, mean, inv_std, gamma, beta, alpha, weight)
        .expect("invariant: all inputs finite and bounded");
    let activated = norm_snake_scalar(x, mean, inv_std, gamma, beta, alpha)
        .expect("invariant: all inputs finite and bounded");
    let sequential =
        weight_mul_scalar(activated, weight).expect("invariant: activated finite, weight finite");

    assert_norm_conv_snake_safety_invariants(fused, sequential);
}

/// Full 7-variable safety proof: all params symbolic, finiteness only.
///
/// Alpha is bounded to [0.1, 100] (Kokoro range). The alpha clamping
/// (`alpha.max(SNAKE_MIN_ALPHA)`) prevents division by zero.
///
/// SUBSTANTIVE: proves finiteness for all finite input combinations
/// in bounded Kokoro ranges. Does NOT prove equivalence (CBMC sin limitation).
///
/// Covers: `norm_activ_conv_kernels.rs` `norm_snake_mul_fused_scalar`.
#[kani::unwind(1)]
#[kani::proof]
fn norm_conv_snake_safety_all_params() {
    let x: f32 = kani::any();
    let mean: f32 = kani::any();
    let inv_std: f32 = kani::any();
    let gamma: f32 = kani::any();
    let beta: f32 = kani::any();
    let alpha: f32 = kani::any();
    let weight: f32 = kani::any();

    kani::assume(x.is_finite());
    kani::assume(mean.is_finite());
    kani::assume(inv_std.is_finite());
    kani::assume(gamma.is_finite());
    kani::assume(beta.is_finite());
    kani::assume(alpha.is_finite());
    kani::assume(weight.is_finite());

    kani::assume(x >= -5.0 && x <= 5.0);
    kani::assume(mean >= -5.0 && mean <= 5.0);
    kani::assume(inv_std >= 0.1 && inv_std <= 10.0);
    kani::assume(gamma >= -1.0 && gamma <= 1.0);
    kani::assume(beta >= -3.0 && beta <= 3.0);
    kani::assume(alpha >= 0.1 && alpha <= 100.0);
    kani::assume(weight >= -3.0 && weight <= 3.0);

    let fused = norm_snake_mul_fused_scalar(x, mean, inv_std, gamma, beta, alpha, weight)
        .expect("invariant: all inputs finite and bounded");
    let activated = norm_snake_scalar(x, mean, inv_std, gamma, beta, alpha)
        .expect("invariant: all inputs finite and bounded");
    let sequential =
        weight_mul_scalar(activated, weight).expect("invariant: activated finite, weight finite");

    assert_norm_conv_snake_safety_invariants(fused, sequential);
}

/// Proves guard completeness: non-finite input always rejected, Ok always finite.
///
/// With 7 parameters, there are `2^7 - 1 = 127` combinations containing at
/// least one non-finite value. Kani symbolically explores all of them.
///
/// SUBSTANTIVE: proves `validate_finite_inputs` has no gaps for the full
/// 7-parameter fused call. Also proves `checked_scalar_output` guarantee:
/// every `Ok(val)` is finite.
///
/// Covers: `norm_activ_conv_kernels.rs` `norm_snake_mul_fused_scalar`
///         and `weight_mul_scalar`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)] // validate_finite_inputs loops over up to 7 elements
fn norm_conv_snake_guard_all_params() {
    let x: f32 = kani::any();
    let mean: f32 = kani::any();
    let inv_std: f32 = kani::any();
    let gamma: f32 = kani::any();
    let beta: f32 = kani::any();
    let alpha: f32 = kani::any();
    let weight: f32 = kani::any();

    let result = norm_snake_mul_fused_scalar(x, mean, inv_std, gamma, beta, alpha, weight);

    let all_finite = x.is_finite()
        && mean.is_finite()
        && inv_std.is_finite()
        && gamma.is_finite()
        && beta.is_finite()
        && alpha.is_finite()
        && weight.is_finite();

    // Non-finite input must always be rejected.
    if !all_finite {
        assert!(result.is_err(), "non-finite input must produce Err");
    }

    // Ok result must always be finite (checked_scalar_output guarantee).
    if let Ok(val) = &result {
        assert!(val.is_finite(), "Ok result must be finite");
    }
}
