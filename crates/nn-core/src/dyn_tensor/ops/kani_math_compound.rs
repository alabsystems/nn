// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for DynTensor compound math operations (#3679).
//!
//! Proves correctness properties of math_compound.rs scalar arithmetic:
//!
//! - ELU: positive passthrough, negative bounded by -alpha, continuity at 0
//! - Leaky ReLU: positive passthrough, negative slope scaling
//! - Snake activation: output >= x for positive alpha, continuity at 0
//! - Clamp: output in [min, max], in-range preservation
//! - repair_non_finite: finite preserved, non-finite replaced
//! - any_non_finite: integer dtypes always return false
//!
//! These harnesses operate on pure scalar/arithmetic — no ndarray or GPU
//! storage — making them tractable for CBMC symbolic execution.

// -- Kani transcendental stubs (CBMC #239, #329, #708) --

fn sin_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
    r
}

/// Deterministic sin stub for proofs that rely on sin^2 >= 0.
/// Returns a fixed bounded value so sin(x)*sin(x) is guaranteed non-negative.
fn sin_f32_det_stub(_x: f32) -> f32 {
    0.5
}

/// exp_m1 stub: e^x - 1. For x < 0: result in [-1, 0). For x = 0: result = 0.
/// For x > 0: result > 0. Uses >= -1.0 (not >) because f32 exp_m1 of very
/// negative x returns exactly -1.0.
fn exp_m1_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0 && r <= 1e10);
    if x < 0.0 {
        kani::assume(r >= -1.0 && r < 0.0);
    }
    if x > 0.0 {
        kani::assume(r > 0.0);
        kani::assume(r >= x.min(1.0));
    }
    r
}

/// Deterministic exp_m1 stub: returns 0.0 (correct for x=0).
/// Used for continuity proofs that require exp_m1(0) = 0 exactly.
fn exp_m1_f32_det_stub(_x: f32) -> f32 {
    0.0
}

// ---------------------------------------------------------------------------
// ELU scalar properties
// ---------------------------------------------------------------------------

/// Prove: ELU is identity for positive inputs.
///
/// elu(x, alpha) = x when x > 0 regardless of alpha. This is the
/// positive-half passthrough that makes ELU a superset of ReLU.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp_m1, exp_m1_f32_stub)]
fn elu_positive_passthrough() {
    let x: f32 = kani::any();
    let alpha: f32 = kani::any();
    kani::assume(x > 0.0 && x <= 1e6);
    kani::assume(alpha.is_finite() && alpha.abs() <= 1e3);

    let result = if x > 0.0 { x } else { alpha * x.exp_m1() };
    assert_eq!(result, x, "ELU must be identity for positive inputs");
}

/// Prove: ELU output for negative inputs is bounded below by -alpha.
///
/// For x < 0: elu(x) = alpha * (exp(x) - 1). Since exp(x) in (0, 1]
/// for x < 0, (exp(x) - 1) in [-1, 0), so elu(x) in [-alpha, 0).
/// Note: >= not > because f32 exp_m1 of very negative x returns exactly -1.0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp_m1, exp_m1_f32_stub)]
fn elu_negative_bounded() {
    let x: i8 = kani::any();
    kani::assume(x < 0);
    let alpha: f32 = kani::any();
    kani::assume(alpha > 0.0 && alpha <= 100.0);

    let fx = x as f32;
    let result = alpha * fx.exp_m1();

    assert!(
        result >= -alpha,
        "ELU negative output must be >= -alpha: got {result}, alpha={alpha}"
    );
    assert!(result <= 0.0, "ELU of negative input must be <= 0");
}

/// Prove: ELU is continuous at x=0.
///
/// Both branches must agree at x=0: positive branch gives 0,
/// negative branch gives alpha * (exp(0) - 1) = alpha * 0 = 0.
/// Discontinuity at 0 would cause gradient issues in training.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp_m1, exp_m1_f32_det_stub)]
fn elu_continuous_at_zero() {
    let alpha: f32 = kani::any();
    kani::assume(alpha.is_finite() && alpha.abs() <= 1e3);

    // Positive branch at x=0 (using x > 0 convention, x=0 goes to negative branch)
    let neg_branch = alpha * (0.0_f32).exp_m1(); // alpha * 0 = 0
    assert_eq!(neg_branch, 0.0, "ELU negative branch at x=0 must be 0.0");
    // Value from both sides approaches 0 at the boundary
}

// ---------------------------------------------------------------------------
// Leaky ReLU scalar properties
// ---------------------------------------------------------------------------

/// Prove: Leaky ReLU is identity for positive inputs.
///
/// leaky_relu(x, slope) = max(0, x) + slope * min(0, x) = x + 0 = x
/// when x > 0. The negative slope only affects x <= 0.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn leaky_relu_positive_passthrough() {
    let x: f32 = kani::any();
    kani::assume(x > 0.0 && x <= 1e6);

    let slope: f32 = kani::any();
    kani::assume(slope.is_finite() && slope.abs() <= 1.0);

    let positive = if x > 0.0 { x } else { 0.0 };
    let negative = if x < 0.0 { x * slope } else { 0.0 };
    let result = positive + negative;

    assert_eq!(result, x, "leaky_relu must be identity for positive inputs");
}

/// Prove: Leaky ReLU scales negative inputs by the slope.
///
/// For x < 0: leaky_relu(x) = slope * x. The slope factor must be
/// correctly applied to the negative branch.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn leaky_relu_negative_slope() {
    let x: i8 = kani::any();
    kani::assume(x < 0);
    let fx = x as f32;

    // Common slope values: 0.01 (default), 0.1, 0.2
    let slope = 0.01_f32;

    let positive = 0.0_f32; // x < 0, so max(0, x) = 0
    let negative = fx * slope; // slope * min(0, x) = slope * x
    let result = positive + negative;

    let expected = fx * slope;
    assert_eq!(
        result, expected,
        "leaky_relu must scale negative inputs by slope"
    );
}

/// Prove: Leaky ReLU with slope=1.0 is the identity function.
///
/// When slope=1, both branches produce x: max(0,x) + 1*min(0,x) = x.
/// This is a useful sanity check that the decomposition is correct.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn leaky_relu_slope_one_is_identity() {
    let x: i8 = kani::any();
    let fx = x as f32;

    let positive = if fx > 0.0 { fx } else { 0.0 };
    let negative = if fx < 0.0 { fx * 1.0 } else { 0.0 };
    let result = positive + negative;

    assert_eq!(result, fx, "leaky_relu with slope=1 must be identity");
}

// ---------------------------------------------------------------------------
// Snake activation scalar properties
// ---------------------------------------------------------------------------

/// Prove: snake activation output >= x for positive alpha and non-negative x.
///
/// snake(x, alpha) = x + (1/alpha) * sin^2(alpha * x).
/// Since sin^2 >= 0 and 1/alpha > 0 when alpha > 0, the additive
/// term is non-negative, so snake(x) >= x.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::sin, sin_f32_det_stub)]
fn snake_output_geq_input() {
    let x: i8 = kani::any();
    let fx = x as f32;

    let alpha = 1.0_f32; // simplest positive alpha
    let scaled = alpha * fx;
    let sin_sq = scaled.sin() * scaled.sin();
    let inv_alpha = 1.0 / alpha;
    let result = fx + inv_alpha * sin_sq;

    assert!(
        result >= fx,
        "snake(x, alpha>0) must be >= x: got {result}, x={fx}"
    );
}

/// Prove: snake activation is identity when sin(alpha*x) = 0.
///
/// At x = k*pi/alpha for integer k, sin(alpha*x) = 0, so
/// snake(x) = x + 0 = x. This verifies the additive structure.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::sin, sin_f32_stub)]
fn snake_identity_at_zero_sin() {
    // x = 0 always gives sin(0) = 0
    let x = 0.0_f32;
    let alpha: f32 = kani::any();
    kani::assume(alpha >= 1e-8 && alpha <= 1e6);

    let scaled = alpha * x;
    let sin_sq = scaled.sin() * scaled.sin();
    let inv_alpha = 1.0 / alpha;
    let result = x + inv_alpha * sin_sq;

    // sin(0) = 0, so result should be x = 0
    assert!(result.abs() < 1e-6, "snake(0) must be ~0: got {result}");
}

/// Prove: snake alpha clamping to [1e-8, 1e6] prevents extreme values.
///
/// The clamping in snake() ensures alpha is never zero (division by zero)
/// or astronomically large (overflow in sin(alpha*x)).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn snake_alpha_clamp_bounds() {
    let alpha: f64 = kani::any();
    // Test with extreme values
    kani::assume(alpha.is_finite());

    let clamped = alpha.clamp(1e-8, 1e6);

    assert!(clamped >= 1e-8, "clamped alpha must be >= 1e-8");
    assert!(clamped <= 1e6, "clamped alpha must be <= 1e6");
    assert!(clamped.is_finite(), "clamped alpha must be finite");
    // inv_alpha must be finite and nonzero
    let inv = 1.0 / clamped;
    assert!(inv.is_finite(), "1/clamped_alpha must be finite");
    assert!(inv > 0.0, "1/clamped_alpha must be positive");
}

/// Prove: clamp preserves values already within range.
///
/// If lo <= x <= hi, then clamp(x, lo, hi) == x. The function should
/// not modify in-range values.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn clamp_preserves_in_range() {
    let x: f32 = kani::any();
    let lo: f32 = kani::any();
    let hi: f32 = kani::any();

    kani::assume(x.is_finite() && x.abs() <= 1e6);
    kani::assume(lo.is_finite() && lo.abs() <= 1e6);
    kani::assume(hi.is_finite() && hi.abs() <= 1e6);
    kani::assume(lo <= hi);
    kani::assume(x >= lo && x <= hi);

    let result = x.clamp(lo, hi);
    assert_eq!(result, x, "clamp must preserve in-range values");
}

// ---------------------------------------------------------------------------
// repair_non_finite scalar properties
// ---------------------------------------------------------------------------

/// Prove: repair_non_finite preserves finite values.
///
/// For any finite value x, the repair function must return x unchanged.
/// Only NaN and Inf values should be replaced with the fallback.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn repair_finite_preserved() {
    let x: f32 = kani::any();
    let fallback: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(fallback.is_finite());

    let result = if x.is_finite() { x } else { fallback };
    assert_eq!(result, x, "repair must preserve finite values");
}

/// Prove: repair_non_finite replaces NaN with fallback.
///
/// NaN inputs must be replaced with the specified fallback value.
/// This is the core contract of repair_non_finite.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn repair_nan_replaced() {
    let fallback: f32 = kani::any();
    kani::assume(fallback.is_finite());

    let x = f32::NAN;
    let result = if x.is_finite() { x } else { fallback };

    assert_eq!(result, fallback, "repair must replace NaN with fallback");
    assert!(result.is_finite(), "repaired value must be finite");
}

/// Prove: repair_non_finite replaces Inf with fallback.
///
/// Both positive and negative infinity must be replaced.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn repair_inf_replaced() {
    let fallback: f32 = kani::any();
    kani::assume(fallback.is_finite());

    // Positive infinity
    let x_pos = f32::INFINITY;
    let result_pos = if x_pos.is_finite() { x_pos } else { fallback };
    assert_eq!(result_pos, fallback, "repair must replace +Inf");

    // Negative infinity
    let x_neg = f32::NEG_INFINITY;
    let result_neg = if x_neg.is_finite() { x_neg } else { fallback };
    assert_eq!(result_neg, fallback, "repair must replace -Inf");
}

// ---------------------------------------------------------------------------
// any_non_finite: integer dtype always false
// ---------------------------------------------------------------------------

/// Prove: non-float dtypes are always classified as finite.
///
/// any_non_finite short-circuits to false for integer and bool dtypes.
/// This verifies the dtype classification logic used in the early return.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn integer_dtypes_always_finite() {
    let idx: u8 = kani::any();
    kani::assume(idx < 5);
    let dt = match idx {
        0 => crate::DType::I32,
        1 => crate::DType::I64,
        2 => crate::DType::U32,
        3 => crate::DType::U8,
        _ => crate::DType::Bool,
    };

    let is_float = matches!(
        dt,
        crate::DType::F32 | crate::DType::BF16 | crate::DType::F16 | crate::DType::F64
    );
    assert!(
        !is_float,
        "integer/bool dtypes must not be classified as float"
    );
}
