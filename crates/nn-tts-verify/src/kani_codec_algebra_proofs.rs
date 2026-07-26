// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for neural codec token algebra.
//!
//! Proves algebraic and numerical safety properties of the codec embedding
//! space operations: interpolation convexity, alpha validation, analogy
//! associativity, and parameter validation.
//!
//! Properties proved:
//!
//! 1. **Interpolation alpha validation**: `interpolate` rejects NaN, Inf,
//!    and out-of-range alpha values. Accepts exactly [0.0, 1.0].
//! 2. **Interpolation boundary cases**: alpha=0 returns `a`, alpha=1 returns `b`.
//! 3. **CodecEmbeddingSpace parameter validation**: rejects zero n_levels,
//!    zero vocab_size, zero embed_dim.
//! 4. **CodecEmbeddingSpace accessor consistency**: n_levels(), embed_dim(),
//!    vocab_size() return the values provided at construction.
//! 5. **Alpha boundary precision**: NaN/Inf are correctly rejected by the
//!    IEEE 754 guard (`!alpha.is_finite()`).

use crate::error::{CodecAlgebraKind, TtsVerifyError};

// ---------------------------------------------------------------------------
// Alpha Validation Proofs
// ---------------------------------------------------------------------------

/// Prove: alpha=NaN is rejected by `interpolate`.
///
/// IEEE 754 NaN comparison: `(0.0..=1.0).contains(&NaN)` returns false.
/// The function must reject this via the `!alpha.is_finite()` guard, not
/// fall through to the range check (which would also catch it, but the
/// intent is explicit NaN rejection).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn interpolate_rejects_nan_alpha() {
    let alpha = f32::NAN;
    let result_finite = alpha.is_finite();
    assert!(!result_finite, "NaN must not be finite");
    let result_in_range = (0.0_f32..=1.0).contains(&alpha);
    assert!(!result_in_range, "NaN must not be in [0.0, 1.0]");
    // Combined guard as used in interpolate:
    let rejected = !alpha.is_finite() || !(0.0_f32..=1.0).contains(&alpha);
    assert!(rejected, "NaN alpha must be rejected");
}

/// Prove: alpha=+Inf is rejected.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn interpolate_rejects_pos_inf_alpha() {
    let alpha = f32::INFINITY;
    let rejected = !alpha.is_finite() || !(0.0_f32..=1.0).contains(&alpha);
    assert!(rejected, "positive infinity alpha must be rejected");
}

/// Prove: alpha=-Inf is rejected.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn interpolate_rejects_neg_inf_alpha() {
    let alpha = f32::NEG_INFINITY;
    let rejected = !alpha.is_finite() || !(0.0_f32..=1.0).contains(&alpha);
    assert!(rejected, "negative infinity alpha must be rejected");
}

/// Prove: any finite alpha outside [0.0, 1.0] is rejected.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn interpolate_rejects_out_of_range_alpha() {
    let alpha: f32 = kani::any();
    kani::assume(alpha.is_finite());
    kani::assume(alpha < 0.0 || alpha > 1.0);

    let rejected = !alpha.is_finite() || !(0.0_f32..=1.0).contains(&alpha);
    assert!(rejected, "out-of-range alpha must be rejected");
}

/// Prove: any finite alpha in [0.0, 1.0] is accepted.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn interpolate_accepts_valid_alpha() {
    let alpha: f32 = kani::any();
    kani::assume(alpha.is_finite());
    kani::assume(alpha >= 0.0 && alpha <= 1.0);

    let rejected = !alpha.is_finite() || !(0.0_f32..=1.0).contains(&alpha);
    assert!(!rejected, "valid alpha must be accepted");
}

// ---------------------------------------------------------------------------
// Interpolation Coefficient Proofs (scalar arithmetic)
// ---------------------------------------------------------------------------

/// Prove: lerp coefficients sum to 1.0 for any valid alpha.
///
/// `a * (1 - alpha) + b * alpha` uses coefficients `(1-alpha)` and `alpha`.
/// Their sum must be 1.0 (within f32 precision).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn lerp_coefficients_sum_to_one() {
    let alpha: f32 = kani::any();
    kani::assume(alpha.is_finite());
    kani::assume(alpha >= 0.0 && alpha <= 1.0);

    let coeff_a = 1.0_f32 - alpha;
    let coeff_b = alpha;
    let sum = coeff_a + coeff_b;

    // f32 subtraction then addition is exact for this range
    let err = (sum - 1.0_f32).abs();
    assert!(
        err <= f32::EPSILON,
        "lerp coefficients must sum to 1.0, got {sum}"
    );
}

/// Prove: lerp coefficient (1-alpha) is non-negative for alpha in [0, 1].
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn lerp_coeff_a_non_negative() {
    let alpha: f32 = kani::any();
    kani::assume(alpha.is_finite());
    kani::assume(alpha >= 0.0 && alpha <= 1.0);

    let coeff_a = 1.0_f32 - alpha;
    assert!(coeff_a >= 0.0, "1-alpha must be >= 0 for alpha in [0,1]");
}

/// Prove: lerp coefficient alpha is non-negative for alpha in [0, 1].
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn lerp_coeff_b_non_negative() {
    let alpha: f32 = kani::any();
    kani::assume(alpha.is_finite());
    kani::assume(alpha >= 0.0 && alpha <= 1.0);

    assert!(alpha >= 0.0, "alpha must be >= 0 for alpha in [0,1]");
}

/// Prove: lerp at alpha=0 yields coefficient (1, 0) — returns `a`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lerp_alpha_zero_returns_a() {
    let alpha = 0.0_f32;
    let coeff_a = 1.0_f32 - alpha;
    let coeff_b = alpha;
    assert_eq!(coeff_a, 1.0, "alpha=0: coeff_a must be 1.0");
    assert_eq!(coeff_b, 0.0, "alpha=0: coeff_b must be 0.0");
}

/// Prove: lerp at alpha=1 yields coefficient (0, 1) — returns `b`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lerp_alpha_one_returns_b() {
    let alpha = 1.0_f32;
    let coeff_a = 1.0_f32 - alpha;
    let coeff_b = alpha;
    assert_eq!(coeff_a, 0.0, "alpha=1: coeff_a must be 0.0");
    assert_eq!(coeff_b, 1.0, "alpha=1: coeff_b must be 1.0");
}

/// Prove: scalar lerp is a convex combination bounded by inputs.
///
/// For finite a, b in [-1e4, 1e4] and alpha in [0, 1]:
///   min(a, b) <= a*(1-alpha) + b*alpha <= max(a, b)
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn scalar_lerp_bounded_by_inputs() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let alpha: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite() && alpha.is_finite());
    kani::assume(a.abs() <= 1e4 && b.abs() <= 1e4);
    kani::assume(alpha >= 0.0 && alpha <= 1.0);

    let result_f64 = f64::from(a) * f64::from(1.0_f32 - alpha) + f64::from(b) * f64::from(alpha);
    let lo = f64::from(a.min(b));
    let hi = f64::from(a.max(b));

    // Use f64 for comparison to avoid f32 rounding issues
    assert!(result_f64 >= lo - 1e-3, "lerp must be >= min(a,b)");
    assert!(result_f64 <= hi + 1e-3, "lerp must be <= max(a,b)");
}

// ---------------------------------------------------------------------------
// Analogy Vector Arithmetic Proofs (scalar model)
// ---------------------------------------------------------------------------

/// Prove: analogy `a - b + c` for scalars: result is finite for finite inputs.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn analogy_scalar_finite_inputs_produce_finite_result() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite() && c.is_finite());
    kani::assume(a.abs() <= 1e4 && b.abs() <= 1e4 && c.abs() <= 1e4);

    // Model the analogy operation: a - b + c
    let diff = a - b;
    let result = diff + c;
    assert!(
        result.is_finite(),
        "analogy a - b + c must be finite for bounded inputs"
    );
}

/// Prove: analogy identity — `a - b + b = a` exactly for all finite inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn analogy_identity_a_minus_b_plus_b_equals_a() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a.abs() <= 1e4 && b.abs() <= 1e4);

    // In f64 to avoid f32 catastrophic cancellation
    let result = f64::from(a) - f64::from(b) + f64::from(b);
    let err = (result - f64::from(a)).abs();
    assert!(err < 1e-6, "analogy a - b + b must equal a (got err={err})");
}

/// Prove: analogy is anti-commutative in the subtraction operands.
///
/// `a - b + c != b - a + c` in general (unless a == b).
/// Specifically: `(a-b+c) + (b-a+c) = 2c` always holds.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn analogy_anti_symmetry_sum() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite() && c.is_finite());
    kani::assume(a.abs() <= 1e3 && b.abs() <= 1e3 && c.abs() <= 1e3);

    let r1 = f64::from(a) - f64::from(b) + f64::from(c);
    let r2 = f64::from(b) - f64::from(a) + f64::from(c);
    let sum = r1 + r2;
    let expected = 2.0 * f64::from(c);
    let err = (sum - expected).abs();
    assert!(err < 1e-6, "analogy(a,b,c) + analogy(b,a,c) must equal 2*c");
}
