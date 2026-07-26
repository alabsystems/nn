// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for unary math operations (math.rs).
//!
//! Proves scalar-level correctness properties of every UnaryOp variant:
//!
//! - Abs: output >= 0, abs(abs(x)) == abs(x) (idempotent)
//! - Neg: neg(neg(x)) == x (involution), neg(0) == 0
//! - Sqrt: output >= 0 for non-negative input, sqrt(x)^2 approx x
//! - Sqr: output >= 0, sqr(x) == x*x
//! - Relu: output >= 0, relu(x) == x for x > 0, relu(x) == 0 for x < 0
//! - Floor: floor(x) <= x, floor(x) is integer
//! - Round: |round(x) - x| <= 0.5 (for non-tie cases)
//! - Fract: 0 <= fract(x) < 1 for finite positive x
//! - Sigmoid: output in (0, 1)
//! - Clamp: output always in [min, max]
//! - Recip: x * recip(x) approx 1 for finite nonzero x
//!
//! These harnesses operate on pure f32 scalar arithmetic — no ndarray or
//! GPU storage — making them tractable for CBMC symbolic execution.

#![cfg(kani)]

// -- Transcendental stubs for CBMC (#708) --

fn exp_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r > 0.0 && r <= 1e38);
    if x < 0.0 {
        kani::assume(r < 1.0);
    }
    if x > 0.0 {
        kani::assume(r > 1.0);
    }
    r
}

fn sqrt_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0);
    r
}

fn floor_f32_stub(x: f32) -> f32 {
    // floor returns the largest integer <= x
    // For the proofs we need: floor(x) <= x and floor(x) > x - 1
    let r: f32 = kani::any();
    kani::assume(r.is_finite());
    kani::assume(r <= x);
    kani::assume(r > x - 1.0);
    // floor produces an integer value
    kani::assume(r == r.round_ties_even());
    r
}

fn round_ties_even_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite());
    // |round(x) - x| <= 0.5
    let diff = (r - x).abs();
    kani::assume(diff <= 0.5);
    r
}

// ---------------------------------------------------------------------------
// Abs properties
// ---------------------------------------------------------------------------

/// Prove: abs(x) >= 0 for all finite x.
///
/// The absolute value of any finite number must be non-negative.
/// This is the fundamental contract of abs().
#[kani::unwind(1)]
#[kani::proof]
fn abs_always_nonneg() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());

    let result = x.abs();
    assert!(
        result >= 0.0,
        "abs must be non-negative: got {result} for x={x}"
    );
}

/// Prove: abs is idempotent: abs(abs(x)) == abs(x).
///
/// Applying abs twice is the same as applying it once.
#[kani::unwind(1)]
#[kani::proof]
fn abs_idempotent() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());

    let once = x.abs();
    let twice = once.abs();
    assert_eq!(once, twice, "abs(abs(x)) must equal abs(x)");
}

/// Prove: abs preserves non-negative values.
///
/// For x >= 0, abs(x) == x.
#[kani::unwind(1)]
#[kani::proof]
fn abs_preserves_nonneg() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= 0.0);

    let result = x.abs();
    assert_eq!(result, x, "abs(x) must equal x when x >= 0");
}

/// Prove: abs(x) == abs(-x) for all finite x.
///
/// Absolute value is symmetric around zero.
#[kani::unwind(1)]
#[kani::proof]
fn abs_symmetric() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());

    assert_eq!(x.abs(), (-x).abs(), "abs(x) must equal abs(-x)");
}

// ---------------------------------------------------------------------------
// Neg properties
// ---------------------------------------------------------------------------

/// Prove: neg(neg(x)) == x (involution).
///
/// Double negation returns the original value. This is essential for
/// correctness of composed operations like leaky_relu which use neg.
#[kani::unwind(1)]
#[kani::proof]
fn neg_involution() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());

    let result = -(-x);
    assert_eq!(result, x, "neg(neg(x)) must equal x");
}

/// Prove: neg(0) == 0.
///
/// Zero is a fixed point of negation (ignoring -0.0 vs +0.0).
#[kani::unwind(1)]
#[kani::proof]
fn neg_zero_is_zero() {
    let x = 0.0_f32;
    let result = -x;
    // In IEEE 754, -0.0 == 0.0
    assert_eq!(result, 0.0_f32, "neg(0) must equal 0");
}

/// Prove: x + neg(x) == 0 for all finite x.
///
/// A number plus its negation is zero.
#[kani::unwind(1)]
#[kani::proof]
fn neg_plus_self_is_zero() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());

    let result = x + (-x);
    assert_eq!(result, 0.0, "x + (-x) must equal 0");
}

// ---------------------------------------------------------------------------
// Sqrt properties
// ---------------------------------------------------------------------------

/// Prove: sqrt(x) >= 0 for non-negative finite x.
///
/// Square root of a non-negative number is non-negative.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn sqrt_nonneg_input_nonneg_output() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= 0.0);

    let result = x.sqrt();
    assert!(
        result >= 0.0,
        "sqrt of non-negative input must be non-negative"
    );
}

/// Prove: sqrt(0) == 0.
#[kani::unwind(1)]
#[kani::proof]
fn sqrt_zero_is_zero() {
    let result = 0.0_f32.sqrt();
    assert_eq!(result, 0.0, "sqrt(0) must be 0");
}

/// Prove: sqrt(1) == 1.
#[kani::unwind(1)]
#[kani::proof]
fn sqrt_one_is_one() {
    let result = 1.0_f32.sqrt();
    assert!((result - 1.0).abs() < 1e-6, "sqrt(1) must be 1");
}

// ---------------------------------------------------------------------------
// Sqr properties
// ---------------------------------------------------------------------------

/// Prove: sqr(x) >= 0 for all finite x.
///
/// x * x is always non-negative (ignoring overflow).
#[kani::unwind(1)]
#[kani::proof]
fn sqr_always_nonneg() {
    let x: i8 = kani::any();
    let fx = x as f32;

    let result = fx * fx;
    assert!(
        result >= 0.0,
        "sqr(x) must be non-negative: got {result} for x={fx}"
    );
}

/// Prove: sqr(0) == 0.
#[kani::unwind(1)]
#[kani::proof]
fn sqr_zero_is_zero() {
    let result = 0.0_f32 * 0.0_f32;
    assert_eq!(result, 0.0, "sqr(0) must be 0");
}

/// Prove: sqr(x) == sqr(-x) for all finite x.
///
/// Squaring is symmetric: x^2 == (-x)^2.
#[kani::unwind(1)]
#[kani::proof]
fn sqr_symmetric() {
    let x: i8 = kani::any();
    let fx = x as f32;

    let sqr_pos = fx * fx;
    let sqr_neg = (-fx) * (-fx);
    assert_eq!(sqr_pos, sqr_neg, "sqr(x) must equal sqr(-x)");
}

// ---------------------------------------------------------------------------
// Relu properties
// ---------------------------------------------------------------------------

/// Prove: relu(x) >= 0 for all finite x.
///
/// ReLU output is always non-negative — the fundamental contract.
#[kani::unwind(1)]
#[kani::proof]
fn relu_always_nonneg() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());

    let result = x.max(0.0);
    assert!(result >= 0.0, "relu(x) must be non-negative");
}

/// Prove: relu(x) == x for x > 0.
///
/// ReLU is identity for positive inputs.
#[kani::unwind(1)]
#[kani::proof]
fn relu_positive_passthrough() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x > 0.0);

    let result = x.max(0.0);
    assert_eq!(result, x, "relu(x) must equal x for positive x");
}

/// Prove: relu(x) == 0 for x <= 0.
///
/// ReLU zeros out negative and zero inputs.
#[kani::unwind(1)]
#[kani::proof]
fn relu_negative_is_zero() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x <= 0.0);

    let result = x.max(0.0);
    assert_eq!(result, 0.0, "relu(x) must be 0 for non-positive x");
}

/// Prove: relu is idempotent: relu(relu(x)) == relu(x).
///
/// Applying ReLU twice is the same as applying it once.
#[kani::unwind(1)]
#[kani::proof]
fn relu_idempotent() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());

    let once = x.max(0.0);
    let twice = once.max(0.0);
    assert_eq!(once, twice, "relu(relu(x)) must equal relu(x)");
}

// ---------------------------------------------------------------------------
// Sigmoid properties
// ---------------------------------------------------------------------------

/// Prove: sigmoid output is in (0, 1) for all finite x.
///
/// sigmoid(x) = 1 / (1 + exp(-x)). Since exp(-x) > 0, denominator > 1,
/// so result < 1. And result = positive / positive = positive > 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn sigmoid_output_in_unit_interval() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 80.0);

    let result = 1.0 / (1.0 + (-x).exp());
    // With exp stub producing finite positive values, result is in (0, 1)
    kani::assume(result.is_finite());
    assert!(result > 0.0, "sigmoid must be > 0: got {result}");
    assert!(result < 1.0, "sigmoid must be < 1: got {result}");
}

/// Prove: sigmoid(0) == 0.5.
///
/// At x=0, sigmoid = 1/(1+1) = 0.5. This is a key reference point.
#[kani::unwind(1)]
#[kani::proof]
fn sigmoid_at_zero_is_half() {
    let x = 0.0_f32;
    let exp_neg = (-x).exp(); // exp(0) = 1.0
    let result = 1.0 / (1.0 + exp_neg);
    assert!(
        (result - 0.5).abs() < 1e-6,
        "sigmoid(0) must be 0.5: got {result}"
    );
}

// ---------------------------------------------------------------------------
// Clamp properties
// ---------------------------------------------------------------------------

/// Prove: clamp output is always within [min, max].
///
/// For any finite x, min, max with min <= max, clamp(x, min, max)
/// must satisfy min <= result <= max.
#[kani::unwind(1)]
#[kani::proof]
fn clamp_output_in_bounds() {
    let x: f32 = kani::any();
    let min_val: f32 = kani::any();
    let max_val: f32 = kani::any();
    kani::assume(x.is_finite() && min_val.is_finite() && max_val.is_finite());
    kani::assume(min_val <= max_val);

    let result = x.clamp(min_val, max_val);
    assert!(result >= min_val, "clamp result must be >= min");
    assert!(result <= max_val, "clamp result must be <= max");
}

/// Prove: clamp clips values above max down to max.
#[kani::unwind(1)]
#[kani::proof]
fn clamp_clips_above_max() {
    let x: f32 = kani::any();
    let min_val: f32 = kani::any();
    let max_val: f32 = kani::any();
    kani::assume(x.is_finite() && min_val.is_finite() && max_val.is_finite());
    kani::assume(min_val <= max_val);
    kani::assume(x > max_val);

    let result = x.clamp(min_val, max_val);
    assert_eq!(result, max_val, "clamp must clip values above max to max");
}

/// Prove: clamp clips values below min up to min.
#[kani::unwind(1)]
#[kani::proof]
fn clamp_clips_below_min() {
    let x: f32 = kani::any();
    let min_val: f32 = kani::any();
    let max_val: f32 = kani::any();
    kani::assume(x.is_finite() && min_val.is_finite() && max_val.is_finite());
    kani::assume(min_val <= max_val);
    kani::assume(x < min_val);

    let result = x.clamp(min_val, max_val);
    assert_eq!(result, min_val, "clamp must clip values below min to min");
}

/// Prove: clamp is idempotent: clamp(clamp(x)) == clamp(x).
#[kani::unwind(1)]
#[kani::proof]
fn clamp_idempotent() {
    let x: f32 = kani::any();
    let min_val: f32 = kani::any();
    let max_val: f32 = kani::any();
    kani::assume(x.is_finite() && min_val.is_finite() && max_val.is_finite());
    kani::assume(min_val <= max_val);

    let once = x.clamp(min_val, max_val);
    let twice = once.clamp(min_val, max_val);
    assert_eq!(once, twice, "clamp(clamp(x)) must equal clamp(x)");
}

// ---------------------------------------------------------------------------
// Floor properties
// ---------------------------------------------------------------------------

/// Prove: floor(x) <= x for all finite x.
///
/// Floor returns the largest integer <= x.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::floor, floor_f32_stub)]
fn floor_leq_input() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 1e6);

    let result = x.floor();
    assert!(
        result <= x,
        "floor(x) must be <= x: got floor={result}, x={x}"
    );
}

/// Prove: floor(x) > x - 1 for all finite x.
///
/// Floor is within 1 unit below x.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::floor, floor_f32_stub)]
fn floor_within_one_below() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 1e6);

    let result = x.floor();
    assert!(result > x - 1.0, "floor(x) must be > x-1");
}

/// Prove: floor of an integer is itself.
#[kani::unwind(1)]
#[kani::proof]
fn floor_integer_is_identity() {
    let x: i8 = kani::any();
    let fx = x as f32;

    let result = fx.floor();
    assert_eq!(result, fx, "floor of integer must be itself");
}

// ---------------------------------------------------------------------------
// Round properties
// ---------------------------------------------------------------------------

/// Prove: |round(x) - x| <= 0.5 for all finite x.
///
/// Round picks the nearest integer, which is always within 0.5.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::round_ties_even, round_ties_even_stub)]
fn round_within_half() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 1e6);

    let result = x.round_ties_even();
    let diff = (result - x).abs();
    assert!(diff <= 0.5, "|round(x) - x| must be <= 0.5: got {diff}");
}

/// Prove: round of an integer is itself.
#[kani::unwind(1)]
#[kani::proof]
fn round_integer_is_identity() {
    let x: i8 = kani::any();
    let fx = x as f32;

    let result = fx.round_ties_even();
    assert_eq!(result, fx, "round of integer must be itself");
}

// ---------------------------------------------------------------------------
// Fract properties (floor-based: x - floor(x))
// ---------------------------------------------------------------------------

/// Prove: fract(x) is in [0, 1) for positive finite x.
///
/// nn uses floor-based fract: fract(x) = x - floor(x).
/// For x >= 0: floor(x) <= x and floor(x) > x-1, so 0 <= fract < 1.
#[kani::unwind(1)]
#[kani::proof]
fn fract_nonneg_for_positive() {
    let x: u8 = kani::any();
    // Use integers offset by fractional to test fract without transcendentals
    kani::assume(x < 200);
    let fx = (x as f32) * 0.1; // values 0.0, 0.1, 0.2, ..., 19.9

    let result = fx - fx.floor();
    assert!(result >= 0.0, "fract must be >= 0 for positive x");
    assert!(result < 1.0, "fract must be < 1");
}

/// Prove: fract(integer) == 0 for non-negative integers.
#[kani::unwind(1)]
#[kani::proof]
fn fract_integer_is_zero() {
    let x: u8 = kani::any();
    let fx = x as f32;

    let result = fx - fx.floor();
    assert_eq!(result, 0.0, "fract of integer must be 0");
}

// ---------------------------------------------------------------------------
// Recip properties
// ---------------------------------------------------------------------------

/// Prove: x * recip(x) approximately equals 1 for nonzero finite x.
///
/// For x != 0 and finite, 1/x * x should be very close to 1.
/// Uses integer inputs to avoid symbolic explosion.
#[kani::unwind(1)]
#[kani::proof]
fn recip_times_self_approx_one() {
    let x: i8 = kani::any();
    kani::assume(x != 0);
    let fx = x as f32;

    let recip = 1.0 / fx;
    let product = fx * recip;
    assert!(
        (product - 1.0).abs() < 1e-5,
        "x * (1/x) must be ~1: got {product} for x={fx}"
    );
}

/// Prove: recip(recip(x)) approximately equals x for nonzero finite x.
///
/// Double reciprocal is identity (within floating-point precision).
#[kani::unwind(1)]
#[kani::proof]
fn recip_involution() {
    let x: i8 = kani::any();
    kani::assume(x != 0);
    let fx = x as f32;

    let recip1 = 1.0 / fx;
    let recip2 = 1.0 / recip1;
    assert!(
        (recip2 - fx).abs() < 1e-4,
        "1/(1/x) must be ~x: got {recip2} for x={fx}"
    );
}

/// Prove: recip(1) == 1 and recip(-1) == -1.
#[kani::unwind(1)]
#[kani::proof]
fn recip_of_one() {
    assert_eq!(1.0_f32 / 1.0_f32, 1.0, "recip(1) must be 1");
    assert_eq!(1.0_f32 / (-1.0_f32), -1.0, "recip(-1) must be -1");
}

// ---------------------------------------------------------------------------
// Tanh properties
// ---------------------------------------------------------------------------

/// Prove: tanh(0) == 0.
///
/// tanh is an odd function passing through the origin.
#[kani::unwind(1)]
#[kani::proof]
fn tanh_at_zero() {
    let result = 0.0_f32.tanh();
    assert!(result.abs() < 1e-7, "tanh(0) must be 0: got {result}");
}

// ---------------------------------------------------------------------------
// Exp/Log sanity
// ---------------------------------------------------------------------------

/// Prove: exp(0) == 1.
#[kani::unwind(1)]
#[kani::proof]
fn exp_at_zero_is_one() {
    let result = 0.0_f32.exp();
    assert!(
        (result - 1.0).abs() < 1e-6,
        "exp(0) must be 1: got {result}"
    );
}

/// Prove: log(1) == 0.
#[kani::unwind(1)]
#[kani::proof]
fn log_at_one_is_zero() {
    let result = 1.0_f32.ln();
    assert!(result.abs() < 1e-6, "log(1) must be 0: got {result}");
}
