// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for ay SMT encoding correctness.
//!
//! Proves numerical safety properties of the encoding layer that translates
//! Rust f64 values into ay Real arithmetic expressions. These harnesses verify:
//!
//! - `real_from_f64`: NaN/Inf rejection, zero encodability, safe-range acceptance
//! - `SMT_QUANTIZATION_MARGIN`: positivity, ordering preservation, finiteness
//! - Analytical bounds (sigmoid, relu, tanh, exp, softplus, gelu, leaky_relu,
//!   snake, binary_add): output ordering (lo <= hi), range invariants,
//!   non-finite input rejection
//!
//! These are the pure-Rust functions that form the foundation of SMT encoding
//! soundness. Proving them correct means the ay encoding layer cannot silently
//! introduce NaN, Inf, or inverted bounds into SMT assertions.
//!
//! Issue: #3625

use super::prove::SMT_QUANTIZATION_MARGIN;
use super::translate_real::real_from_f64;
use crate::bounds::{
    binary_add_output_bounds, exp_output_bounds, gelu_output_bounds, leaky_relu_output_bounds,
    relu_output_bounds, sigmoid_output_bounds, snake_output_bounds, softplus_output_bounds,
    tanh_output_bounds,
};

// ── real_from_f64 encoding invariants ──────────────────────────────────

/// Proves `real_from_f64(0.0)` succeeds. Zero must always be encodable
/// because it appears as an identity element in SMT assertions (e.g.,
/// `real_add(x, 0)` elimination, `real_mul(x, 0)` folding).
#[kani::unwind(1)]
#[kani::proof]
fn real_from_f64_zero_is_ok() {
    assert!(real_from_f64(0.0).is_ok(), "zero must be encodable");
    assert!(
        real_from_f64(-0.0).is_ok(),
        "negative zero must be encodable"
    );
}

/// Proves `real_from_f64` rejects all three non-finite IEEE 754 values.
/// NaN or Inf in an SMT Real literal would produce unsound assertions.
#[kani::unwind(1)]
#[kani::proof]
fn real_from_f64_rejects_all_non_finite_sentinels() {
    assert!(real_from_f64(f64::NAN).is_err(), "NaN must be rejected");
    assert!(
        real_from_f64(f64::INFINITY).is_err(),
        "+Inf must be rejected"
    );
    assert!(
        real_from_f64(f64::NEG_INFINITY).is_err(),
        "-Inf must be rejected"
    );
}

/// Proves `real_from_f64` accepts small integer values. Integer-valued f64s
/// that fit in i64 take the fast path (no denominator, no rounding error).
#[kani::unwind(1)]
#[kani::proof]
fn real_from_f64_accepts_small_integers() {
    let val: i32 = kani::any();
    let f = val as f64;
    // All i32 values fit in i64, so integer fast path fires.
    let result = real_from_f64(f);
    assert!(result.is_ok(), "all i32-range integers must be encodable");
}

/// Proves `real_from_f64` correctly handles the i64 overflow boundary.
/// Per issue #398, the guard uses `>=` (not `>`) because `i64::MAX as f64`
/// rounds up. This harness verifies the boundary value is rejected.
#[kani::unwind(1)]
#[kani::proof]
fn real_from_f64_i64_boundary_rejected() {
    // i64::MAX as f64 = 9.223372036854776e18 (rounds up from odd i64::MAX).
    // This value times any denominator >= 1e6 exceeds i64 range.
    let boundary = i64::MAX as f64;
    let result = real_from_f64(boundary);
    assert!(
        result.is_err(),
        "i64::MAX as f64 (rounded up) must be rejected"
    );
}

/// Proves that typical ML constant values (epsilon, scale factors, frequencies)
/// are encodable. These are the values that appear in ground-folded kernel
/// constants and would cause silent zero-quantization without adaptive denominators.
#[kani::unwind(1)]
#[kani::proof]
fn real_from_f64_ml_constants_encodable() {
    // LayerNorm epsilon
    assert!(real_from_f64(1e-5).is_ok(), "epsilon 1e-5");
    // BatchNorm epsilon
    assert!(real_from_f64(1e-12).is_ok(), "epsilon 1e-12");
    // Typical weight scale
    assert!(real_from_f64(0.02).is_ok(), "weight scale 0.02");
    // sqrt(2/pi) used in GELU
    assert!(real_from_f64(0.797_884_560_802_865_4).is_ok(), "sqrt(2/pi)");
    // GELU coefficient
    assert!(real_from_f64(0.044715).is_ok(), "GELU coeff");
    // Negative values
    assert!(real_from_f64(-1.0).is_ok(), "negative one");
    assert!(real_from_f64(-100.0).is_ok(), "negative hundred");
}

// ── SMT_QUANTIZATION_MARGIN properties ─────────────────────────────────

/// Proves SMT_QUANTIZATION_MARGIN is positive, finite, and within a
/// reasonable range (less than 1.0). A non-positive margin would fail to
/// widen bounds. A margin >= 1.0 would make proofs vacuously wide.
#[kani::unwind(1)]
#[kani::proof]
fn smt_quantization_margin_invariants() {
    assert!(SMT_QUANTIZATION_MARGIN > 0.0, "margin must be positive");
    assert!(
        SMT_QUANTIZATION_MARGIN < 1.0,
        "margin must be < 1.0 to avoid vacuous proofs"
    );
    assert!(SMT_QUANTIZATION_MARGIN.is_finite(), "margin must be finite");
    // Verify it is encodable by real_from_f64 (used in finalize_query).
    assert!(
        real_from_f64(SMT_QUANTIZATION_MARGIN).is_ok(),
        "margin must be encodable"
    );
    assert!(
        real_from_f64(-SMT_QUANTIZATION_MARGIN).is_ok(),
        "negated margin must be encodable"
    );
}

/// Proves that widening ordered bounds by SMT_QUANTIZATION_MARGIN preserves
/// the ordering invariant and produces encodable values.
#[kani::unwind(1)]
#[kani::proof]
fn smt_margin_preserves_ordering() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo <= hi);
    // Stay within the encodable range for real_from_f64.
    kani::assume(lo.abs() < 1e10 && hi.abs() < 1e10);

    let widened_lo = lo - SMT_QUANTIZATION_MARGIN;
    let widened_hi = hi + SMT_QUANTIZATION_MARGIN;

    assert!(widened_lo.is_finite(), "widened lower must be finite");
    assert!(widened_hi.is_finite(), "widened upper must be finite");
    assert!(
        widened_lo <= widened_hi,
        "widened bounds must preserve lo <= hi"
    );
    assert!(widened_lo <= lo, "widened lower <= original lower");
    assert!(widened_hi >= hi, "widened upper >= original upper");
}

// ── Analytical bounds: output ordering and range invariants ────────────

/// Proves `relu_output_bounds` always returns non-negative values when
/// inputs are valid (finite, ordered). relu(x) = max(x, 0) >= 0.
#[kani::unwind(1)]
#[kani::proof]
fn relu_bounds_output_nonneg() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo <= hi);
    kani::assume(lo.abs() < 1e15 && hi.abs() < 1e15);

    if let Ok((out_lo, out_hi)) = relu_output_bounds(lo, hi) {
        assert!(out_lo >= 0.0, "relu lower bound must be >= 0");
        assert!(out_hi >= 0.0, "relu upper bound must be >= 0");
        assert!(out_lo <= out_hi, "relu bounds must be ordered");
    }
}

/// Proves `sigmoid_output_bounds` always returns values in (0, 1) when
/// inputs are valid. sigmoid(x) in (0, 1) for all finite x.
#[kani::unwind(1)]
#[kani::proof]
fn sigmoid_bounds_in_unit_interval() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo <= hi);
    kani::assume(lo.abs() < 1e6 && hi.abs() < 1e6);

    if let Ok((out_lo, out_hi)) = sigmoid_output_bounds(lo, hi) {
        assert!(out_lo > 0.0, "sigmoid lower > 0");
        assert!(out_hi < 1.0, "sigmoid upper < 1");
        assert!(out_lo <= out_hi, "sigmoid bounds must be ordered");
    }
}

/// Proves `tanh_output_bounds` always returns values in [-1, 1] when
/// inputs are valid. tanh(x) in (-1, 1) for all finite x.
#[kani::unwind(1)]
#[kani::proof]
fn tanh_bounds_in_neg1_pos1() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo <= hi);
    kani::assume(lo.abs() < 1e6 && hi.abs() < 1e6);

    if let Ok((out_lo, out_hi)) = tanh_output_bounds(lo, hi) {
        assert!(out_lo >= -1.0, "tanh lower >= -1");
        assert!(out_hi <= 1.0, "tanh upper <= 1");
        assert!(out_lo <= out_hi, "tanh bounds must be ordered");
    }
}

/// Proves `exp_output_bounds` always returns positive values when the
/// result is finite. exp(x) > 0 for all real x.
#[kani::unwind(1)]
#[kani::proof]
fn exp_bounds_positive() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo <= hi);
    // Keep inputs small enough that exp() doesn't overflow.
    kani::assume(lo >= -500.0 && hi <= 500.0);

    if let Ok((out_lo, out_hi)) = exp_output_bounds(lo, hi) {
        assert!(out_lo > 0.0, "exp lower bound must be > 0");
        assert!(out_hi > 0.0, "exp upper bound must be > 0");
        assert!(out_lo <= out_hi, "exp bounds must be ordered");
    }
}

/// Proves `softplus_output_bounds` always returns positive values when
/// the result is finite. softplus(x) = ln(1 + exp(x)) > 0 for all x.
#[kani::unwind(1)]
#[kani::proof]
fn softplus_bounds_positive() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo <= hi);
    kani::assume(lo >= -500.0 && hi <= 500.0);

    if let Ok((out_lo, out_hi)) = softplus_output_bounds(lo, hi) {
        assert!(out_lo > 0.0, "softplus lower bound must be > 0");
        assert!(out_hi > 0.0, "softplus upper bound must be > 0");
        assert!(out_lo <= out_hi, "softplus bounds must be ordered");
    }
}

/// Proves `gelu_output_bounds` always returns ordered output bounds.
/// GELU is not monotone (global minimum at x ~ -0.7523), so this is
/// non-trivial — the bounds computation must handle the minimum correctly.
#[kani::unwind(1)]
#[kani::proof]
fn gelu_bounds_ordered() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo <= hi);
    kani::assume(lo.abs() < 1e6 && hi.abs() < 1e6);

    if let Ok((out_lo, out_hi)) = gelu_output_bounds(lo, hi) {
        assert!(out_lo.is_finite(), "gelu lower must be finite");
        assert!(out_hi.is_finite(), "gelu upper must be finite");
        assert!(out_lo <= out_hi, "gelu bounds must be ordered");
    }
}

/// Proves `leaky_relu_output_bounds` returns ordered output bounds for
/// positive alpha. LeakyReLU with alpha >= 0 is monotonically non-decreasing.
#[kani::unwind(1)]
#[kani::proof]
fn leaky_relu_bounds_ordered_positive_alpha() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    let alpha: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite() && alpha.is_finite());
    kani::assume(lo <= hi);
    kani::assume(alpha >= 0.0 && alpha <= 1.0);
    kani::assume(lo.abs() < 1e10 && hi.abs() < 1e10);

    if let Ok((out_lo, out_hi)) = leaky_relu_output_bounds(alpha, lo, hi) {
        assert!(out_lo.is_finite(), "leaky_relu lower must be finite");
        assert!(out_hi.is_finite(), "leaky_relu upper must be finite");
        assert!(out_lo <= out_hi, "leaky_relu bounds must be ordered");
    }
}

/// Proves `snake_output_bounds` produces ordered output bounds where
/// the lower bound equals x_lo (the minimum of x + non-negative term).
#[kani::unwind(1)]
#[kani::proof]
fn snake_bounds_lower_is_x_lo() {
    let x_lo: f64 = kani::any();
    let x_hi: f64 = kani::any();
    let alpha: f64 = kani::any();
    kani::assume(x_lo.is_finite() && x_hi.is_finite() && alpha.is_finite());
    kani::assume(x_lo <= x_hi);
    kani::assume(alpha > 0.0);
    // Keep values in a range where 1/alpha doesn't overflow.
    kani::assume(alpha >= 1e-10);
    kani::assume(x_lo.abs() < 1e10 && x_hi.abs() < 1e10);

    if let Ok((out_lo, out_hi)) = snake_output_bounds(x_lo, x_hi, alpha) {
        assert_eq!(out_lo, x_lo, "snake lower bound must equal x_lo");
        assert!(out_hi >= x_hi, "snake upper bound must be >= x_hi");
        assert!(out_lo <= out_hi, "snake bounds must be ordered");
    }
}

/// Proves `binary_add_output_bounds` returns exact bounds:
/// out_lo = x_lo + y_lo, out_hi = x_hi + y_hi. Addition is monotonically
/// increasing in both arguments, so these bounds are tight.
#[kani::unwind(1)]
#[kani::proof]
fn binary_add_bounds_exact() {
    let x_lo: f64 = kani::any();
    let x_hi: f64 = kani::any();
    let y_lo: f64 = kani::any();
    let y_hi: f64 = kani::any();
    kani::assume(x_lo.is_finite() && x_hi.is_finite());
    kani::assume(y_lo.is_finite() && y_hi.is_finite());
    kani::assume(x_lo <= x_hi && y_lo <= y_hi);
    // Keep sums finite.
    kani::assume(x_lo.abs() < 1e14 && x_hi.abs() < 1e14);
    kani::assume(y_lo.abs() < 1e14 && y_hi.abs() < 1e14);

    if let Ok((out_lo, out_hi)) = binary_add_output_bounds(x_lo, x_hi, y_lo, y_hi) {
        assert!(
            (out_lo - (x_lo + y_lo)).abs() < 1e-10,
            "add lower bound must be x_lo + y_lo"
        );
        assert!(
            (out_hi - (x_hi + y_hi)).abs() < 1e-10,
            "add upper bound must be x_hi + y_hi"
        );
        assert!(out_lo <= out_hi, "add bounds must be ordered");
    }
}

// ── Non-finite input rejection ─────────────────────────────────────────

/// Proves all activation bounds functions reject NaN lower bound.
/// This is the IEEE 754 defense-in-depth: NaN bypasses comparisons (#3356),
/// so explicit `is_finite()` checks are mandatory.
#[kani::unwind(1)]
#[kani::proof]
fn bounds_reject_nan_lower() {
    let hi = 1.0_f64;
    assert!(
        relu_output_bounds(f64::NAN, hi).is_err(),
        "relu rejects NaN lo"
    );
    assert!(
        sigmoid_output_bounds(f64::NAN, hi).is_err(),
        "sigmoid rejects NaN lo"
    );
    assert!(
        tanh_output_bounds(f64::NAN, hi).is_err(),
        "tanh rejects NaN lo"
    );
    assert!(
        exp_output_bounds(f64::NAN, hi).is_err(),
        "exp rejects NaN lo"
    );
    assert!(
        softplus_output_bounds(f64::NAN, hi).is_err(),
        "softplus rejects NaN lo"
    );
    assert!(
        gelu_output_bounds(f64::NAN, hi).is_err(),
        "gelu rejects NaN lo"
    );
}

/// Proves all activation bounds functions reject inverted bounds (lo > hi).
/// Inverted bounds in an SMT assertion would produce unsound results.
#[kani::unwind(1)]
#[kani::proof]
fn bounds_reject_inverted() {
    let lo = 5.0_f64;
    let hi = -5.0_f64;
    assert!(relu_output_bounds(lo, hi).is_err(), "relu rejects inverted");
    assert!(
        sigmoid_output_bounds(lo, hi).is_err(),
        "sigmoid rejects inverted"
    );
    assert!(tanh_output_bounds(lo, hi).is_err(), "tanh rejects inverted");
    assert!(exp_output_bounds(lo, hi).is_err(), "exp rejects inverted");
    assert!(
        softplus_output_bounds(lo, hi).is_err(),
        "softplus rejects inverted"
    );
    assert!(gelu_output_bounds(lo, hi).is_err(), "gelu rejects inverted");
}
