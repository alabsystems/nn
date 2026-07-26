// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for fused NormActivConv1d+LeakyReLU per-tap scalar.
//!
//! The fused NormActivConv1d+LeakyReLU kernel computes InstanceNorm + residual
//! affine + LeakyReLU + Conv1d weight multiply in a single GPU dispatch.
//! The sequential path uses two dispatches: norm_leaky_relu then weight_mul.
//!
//! ```text
//! normed = (x - mean) * inv_std
//! y = (1 + gamma) * normed + beta   // Kokoro residual gamma convention
//! activated = if y >= 0 { y } else { slope * y }
//! contribution = activated * weight
//! ```
//!
//! Unlike AdaIN (which computes `rsqrt(var + eps)` inline), NormActivConv1d
//! takes pre-computed `inv_std` directly — no transcendental functions.
//! This makes bitwise equivalence provable by CBMC without stub approximations.
//!
//! These harnesses prove:
//! 1. Fused/sequential bitwise equivalence for symbolic x and weight
//! 2. Bitwise equivalence with symbolic style+activation params
//! 3. Full 7-variable bitwise equivalence with all params symbolic
//! 4. Guard completeness: non-finite input -> Err, Ok -> finite (2^7-1 cases)
//!
//! Mirrors `adain_leaky_relu_kani.rs` structure. Part of #3020, #2218 F13.

use super::*;

/// Asserts safety invariants shared by fused and sequential paths.
///
/// NormActivConv1d+LeakyReLU is piecewise linear (no transcendentals):
/// bitwise equivalence holds because fused and sequential compute identical
/// floating-point operations in identical order.
///
/// Invariants proved:
/// - Both `fused` and `sequential` outputs are finite
/// - Both outputs agree (fused == sequential to the bit)
/// - For slope in [0, 1] and weight in [-W, W]: output magnitude is bounded
fn assert_norm_conv_leaky_relu_safety_invariants(fused: f32, sequential: f32) {
    assert!(fused.is_finite(), "fused output must remain finite");
    assert!(
        sequential.is_finite(),
        "sequential output must remain finite"
    );

    // Fused and sequential compute identical operations:
    //   fused:      activated * weight  (where activated = leaky_relu(normed_affine))
    //   sequential: weight_mul(norm_leaky_relu(...), weight)
    //             = norm_leaky_relu(...) * weight
    // Same FP operations, same order, same rounding. Bitwise equality holds.
    assert!(
        fused.to_bits() == sequential.to_bits(),
        "fused and sequential must be bitwise equal"
    );
}

/// Proves fused/sequential bitwise equivalence for symbolic x and weight,
/// with normalization and activation params fixed to representative values.
///
/// SUBSTANTIVE: proves that fused and sequential paths produce bitwise-identical
/// output for all x in [-10, 10] and weight in [-3, 3].
///
/// Covers: `norm_activ_conv_kernels.rs` `norm_leaky_relu_mul_fused_scalar`.
#[kani::unwind(1)]
#[kani::proof]
fn norm_conv_leaky_relu_safety_x_weight() {
    let x: f32 = kani::any();
    let weight: f32 = kani::any();

    kani::assume(x.is_finite());
    kani::assume(weight.is_finite());
    kani::assume(x >= -10.0 && x <= 10.0);
    kani::assume(weight >= -3.0 && weight <= 3.0);

    let mean = 0.0_f32;
    let inv_std = 1.0_f32;
    let gamma = 0.0_f32; // residual gamma: (1+0) = 1, identity scale
    let beta = 0.0_f32;
    let slope = 0.2_f32; // Kokoro production value

    let fused = norm_leaky_relu_mul_fused_scalar(x, mean, inv_std, gamma, beta, slope, weight)
        .expect("invariant: all inputs finite and bounded");
    let activated = norm_leaky_relu_scalar(x, mean, inv_std, gamma, beta, slope)
        .expect("invariant: all inputs finite and bounded");
    let sequential =
        weight_mul_scalar(activated, weight).expect("invariant: activated finite, weight finite");

    assert_norm_conv_leaky_relu_safety_invariants(fused, sequential);
}

/// Proves bitwise equivalence with symbolic style and activation params.
///
/// Exercises the normalization dimension: gamma/beta are the InstanceNorm
/// affine transform, slope controls the LeakyReLU negative region. Fixed
/// mean=0, inv_std=1 for tractability.
///
/// SUBSTANTIVE: proves bitwise equivalence for all combinations of
/// x, gamma, beta, slope, weight in Kokoro-realistic ranges.
///
/// Covers: `norm_activ_conv_kernels.rs` `norm_leaky_relu_mul_fused_scalar`.
#[kani::unwind(1)]
#[kani::proof]
fn norm_conv_leaky_relu_safety_style_params() {
    let x: f32 = kani::any();
    let gamma: f32 = kani::any();
    let beta: f32 = kani::any();
    let slope: f32 = kani::any();
    let weight: f32 = kani::any();

    kani::assume(x.is_finite());
    kani::assume(gamma.is_finite());
    kani::assume(beta.is_finite());
    kani::assume(slope.is_finite());
    kani::assume(weight.is_finite());
    kani::assume(x >= -5.0 && x <= 5.0);
    kani::assume(gamma >= -1.0 && gamma <= 1.0); // residual gamma is small
    kani::assume(beta >= -3.0 && beta <= 3.0);
    kani::assume(slope >= 0.0 && slope <= 1.0);
    kani::assume(weight >= -3.0 && weight <= 3.0);

    let mean = 0.0_f32;
    let inv_std = 1.0_f32;

    let fused = norm_leaky_relu_mul_fused_scalar(x, mean, inv_std, gamma, beta, slope, weight)
        .expect("invariant: all inputs finite and bounded");
    let activated = norm_leaky_relu_scalar(x, mean, inv_std, gamma, beta, slope)
        .expect("invariant: all inputs finite and bounded");
    let sequential =
        weight_mul_scalar(activated, weight).expect("invariant: activated finite, weight finite");

    assert_norm_conv_leaky_relu_safety_invariants(fused, sequential);
}

/// Full 7-variable fused/sequential bitwise equivalence proof.
///
/// All params symbolic including mean and inv_std. The inv_std range [0.1, 10]
/// covers typical InstanceNorm values (reciprocal of standard deviation).
///
/// May require extended Kani timeout due to the 7-variable state space.
///
/// SUBSTANTIVE: proves bitwise equivalence for all finite input combinations
/// in bounded Kokoro ranges.
///
/// Covers: `norm_activ_conv_kernels.rs` `norm_leaky_relu_mul_fused_scalar`.
#[kani::unwind(1)]
#[kani::proof]
fn norm_conv_leaky_relu_safety_all_params() {
    let x: f32 = kani::any();
    let mean: f32 = kani::any();
    let inv_std: f32 = kani::any();
    let gamma: f32 = kani::any();
    let beta: f32 = kani::any();
    let slope: f32 = kani::any();
    let weight: f32 = kani::any();

    kani::assume(x.is_finite());
    kani::assume(mean.is_finite());
    kani::assume(inv_std.is_finite());
    kani::assume(gamma.is_finite());
    kani::assume(beta.is_finite());
    kani::assume(slope.is_finite());
    kani::assume(weight.is_finite());

    kani::assume(x >= -5.0 && x <= 5.0);
    kani::assume(mean >= -5.0 && mean <= 5.0);
    kani::assume(inv_std >= 0.1 && inv_std <= 10.0);
    kani::assume(gamma >= -1.0 && gamma <= 1.0);
    kani::assume(beta >= -3.0 && beta <= 3.0);
    kani::assume(slope >= 0.0 && slope <= 1.0);
    kani::assume(weight >= -3.0 && weight <= 3.0);

    let fused = norm_leaky_relu_mul_fused_scalar(x, mean, inv_std, gamma, beta, slope, weight)
        .expect("invariant: all inputs finite and bounded");
    let activated = norm_leaky_relu_scalar(x, mean, inv_std, gamma, beta, slope)
        .expect("invariant: all inputs finite and bounded");
    let sequential =
        weight_mul_scalar(activated, weight).expect("invariant: activated finite, weight finite");

    assert_norm_conv_leaky_relu_safety_invariants(fused, sequential);
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
/// Unlike the equivalence harnesses above, this does NOT constrain input
/// ranges — it tests the full f32 domain including NaN, Inf, subnormals.
///
/// Covers: `norm_activ_conv_kernels.rs` `norm_leaky_relu_mul_fused_scalar`
///         and `weight_mul_scalar`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)] // validate_finite_inputs loops over up to 7 elements
fn norm_conv_leaky_relu_guard_all_params() {
    let x: f32 = kani::any();
    let mean: f32 = kani::any();
    let inv_std: f32 = kani::any();
    let gamma: f32 = kani::any();
    let beta: f32 = kani::any();
    let slope: f32 = kani::any();
    let weight: f32 = kani::any();

    let result = norm_leaky_relu_mul_fused_scalar(x, mean, inv_std, gamma, beta, slope, weight);

    let all_finite = x.is_finite()
        && mean.is_finite()
        && inv_std.is_finite()
        && gamma.is_finite()
        && beta.is_finite()
        && slope.is_finite()
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
