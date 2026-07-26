// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani verification harnesses for `bounds/snake.rs` — `snake_output_bounds`.
//!
//! Proves safety and correctness properties of the analytical Snake activation
//! output bound computation: `snake(x, alpha) = x + (1/alpha) * sin(alpha*x)^2`.
//!
//! Properties verified:
//! - Input validation: zero/negative alpha, non-finite inputs, inverted bounds
//! - Output correctness: lower == x_lo, upper == x_hi + 1/alpha
//! - Output finiteness for valid inputs
//! - Monotonicity: wider input → wider output
//! - Bound ordering: output_lower <= output_upper
//! - Alpha sensitivity: larger alpha → tighter upper bound
//! - Overflow detection: small alpha causes 1/alpha overflow
//!
//! Part of #3658.

use crate::bounds::snake_output_bounds;

// ===========================================================================
// Input validation harnesses
// ===========================================================================

// ---------------------------------------------------------------------------
// 1. Zero alpha rejected
// ---------------------------------------------------------------------------

/// Prove: `snake_output_bounds(*, *, 0.0)` always returns Err.
/// Division by zero in `1/alpha` is undefined.
#[kani::unwind(1)]
#[kani::proof]
fn snake_bounds_zero_alpha_rejected() {
    let x_lo: i8 = kani::any();
    let x_hi: i8 = kani::any();
    kani::assume(x_lo <= x_hi);
    let result = snake_output_bounds(x_lo as f64, x_hi as f64, 0.0);
    assert!(result.is_err(), "alpha=0 must be rejected");
}

// ---------------------------------------------------------------------------
// 2. Negative alpha rejected
// ---------------------------------------------------------------------------

/// Prove: `snake_output_bounds(*, *, negative)` always returns Err.
/// Snake activation requires alpha > 0 for well-definedness.
#[kani::unwind(1)]
#[kani::proof]
fn snake_bounds_negative_alpha_rejected() {
    let x_lo: i8 = kani::any();
    let x_hi: i8 = kani::any();
    kani::assume(x_lo <= x_hi);
    let alpha: u8 = kani::any();
    kani::assume(alpha >= 1);
    let neg_alpha = -(alpha as f64);

    let result = snake_output_bounds(x_lo as f64, x_hi as f64, neg_alpha);
    assert!(result.is_err(), "negative alpha must be rejected");
}

// ---------------------------------------------------------------------------
// 3. NaN alpha rejected
// ---------------------------------------------------------------------------

/// Prove: `snake_output_bounds(*, *, NaN)` always returns Err.
#[kani::unwind(1)]
#[kani::proof]
fn snake_bounds_nan_alpha_rejected() {
    let result = snake_output_bounds(-1.0, 1.0, f64::NAN);
    assert!(result.is_err(), "NaN alpha must be rejected");
}

// ---------------------------------------------------------------------------
// 4. Infinity alpha rejected
// ---------------------------------------------------------------------------

/// Prove: `snake_output_bounds(*, *, Inf)` always returns Err.
#[kani::unwind(1)]
#[kani::proof]
fn snake_bounds_inf_alpha_rejected() {
    let result = snake_output_bounds(-1.0, 1.0, f64::INFINITY);
    assert!(result.is_err(), "Inf alpha must be rejected");
}

// ---------------------------------------------------------------------------
// 5. NaN x_lo rejected
// ---------------------------------------------------------------------------

/// Prove: `snake_output_bounds(NaN, *, *)` always returns Err.
#[kani::unwind(1)]
#[kani::proof]
fn snake_bounds_nan_x_lo_rejected() {
    let result = snake_output_bounds(f64::NAN, 1.0, 1.0);
    assert!(result.is_err(), "NaN x_lo must be rejected");
}

// ---------------------------------------------------------------------------
// 6. NaN x_hi rejected
// ---------------------------------------------------------------------------

/// Prove: `snake_output_bounds(*, NaN, *)` always returns Err.
#[kani::unwind(1)]
#[kani::proof]
fn snake_bounds_nan_x_hi_rejected() {
    let result = snake_output_bounds(-1.0, f64::NAN, 1.0);
    assert!(result.is_err(), "NaN x_hi must be rejected");
}

// ---------------------------------------------------------------------------
// 7. Inverted x bounds rejected
// ---------------------------------------------------------------------------

/// Prove: `snake_output_bounds(hi, lo, *)` where hi > lo always returns Err.
#[kani::unwind(1)]
#[kani::proof]
fn snake_bounds_inverted_x_rejected() {
    let x_lo: i8 = kani::any();
    let x_hi: i8 = kani::any();
    kani::assume(x_lo > x_hi); // inverted
    let alpha: u8 = kani::any();
    kani::assume(alpha >= 1);

    let result = snake_output_bounds(x_lo as f64, x_hi as f64, alpha as f64);
    assert!(result.is_err(), "inverted x bounds must be rejected");
}

// ---------------------------------------------------------------------------
// 8. Small alpha triggers overflow detection
// ---------------------------------------------------------------------------

/// Prove: very small alpha (where 1/alpha overflows) is properly detected.
#[kani::unwind(1)]
#[kani::proof]
fn snake_bounds_small_alpha_overflow() {
    // 5e-324 is the smallest positive f64; 1.0/5e-324 = Inf.
    let result = snake_output_bounds(0.0, 1.0, 5e-324);
    assert!(
        result.is_err(),
        "overflow from small alpha must be detected"
    );
}

// ===========================================================================
// Output correctness harnesses
// ===========================================================================

// ---------------------------------------------------------------------------
// 9. Output lower bound equals x_lo
// ---------------------------------------------------------------------------

/// Prove: for all valid inputs, `output_lower == x_lo`.
/// Snake adds a non-negative term `(1/alpha) * sin^2(alpha*x)`, so
/// the minimum output is at the minimum input with the additive term = 0.
#[kani::unwind(1)]
#[kani::proof]
fn snake_bounds_lower_equals_x_lo() {
    let x_lo: i8 = kani::any();
    let x_hi: i8 = kani::any();
    kani::assume(x_lo <= x_hi);
    let alpha: u8 = kani::any();
    kani::assume(alpha >= 1 && alpha <= 100);

    if let Ok((lo, _hi)) = snake_output_bounds(x_lo as f64, x_hi as f64, alpha as f64) {
        assert_eq!(lo, x_lo as f64, "output lower must equal x_lo");
    }
}

// ---------------------------------------------------------------------------
// 10. Output upper bound equals x_hi + 1/alpha
// ---------------------------------------------------------------------------

/// Prove: for all valid inputs, `output_upper == x_hi + 1/alpha`.
/// The maximum additive term `(1/alpha) * sin^2 = 1/alpha * 1 = 1/alpha`.
#[kani::unwind(1)]
#[kani::proof]
fn snake_bounds_upper_equals_formula() {
    let x_lo: i8 = kani::any();
    let x_hi: i8 = kani::any();
    kani::assume(x_lo <= x_hi);
    let alpha: u8 = kani::any();
    kani::assume(alpha >= 1 && alpha <= 100);

    if let Ok((_lo, hi)) = snake_output_bounds(x_lo as f64, x_hi as f64, alpha as f64) {
        let expected = (x_hi as f64) + 1.0 / (alpha as f64);
        assert!(
            (hi - expected).abs() < 1e-10,
            "output upper must equal x_hi + 1/alpha"
        );
    }
}

// ---------------------------------------------------------------------------
// 11. Output bounds are finite for valid inputs
// ---------------------------------------------------------------------------

/// Prove: when inputs are valid, both output bounds are finite.
#[kani::unwind(1)]
#[kani::proof]
fn snake_bounds_output_finite() {
    let x_lo: i8 = kani::any();
    let x_hi: i8 = kani::any();
    kani::assume(x_lo <= x_hi);
    let alpha: u8 = kani::any();
    kani::assume(alpha >= 1 && alpha <= 100);

    if let Ok((lo, hi)) = snake_output_bounds(x_lo as f64, x_hi as f64, alpha as f64) {
        assert!(lo.is_finite(), "output lower must be finite");
        assert!(hi.is_finite(), "output upper must be finite");
    }
}

// ---------------------------------------------------------------------------
// 12. Output ordering: lower <= upper
// ---------------------------------------------------------------------------

/// Prove: for all valid inputs, `output_lower <= output_upper`.
/// Since output_lower = x_lo and output_upper = x_hi + 1/alpha,
/// and x_lo <= x_hi and 1/alpha > 0, this is always true.
#[kani::unwind(1)]
#[kani::proof]
fn snake_bounds_output_ordered() {
    let x_lo: i8 = kani::any();
    let x_hi: i8 = kani::any();
    kani::assume(x_lo <= x_hi);
    let alpha: u8 = kani::any();
    kani::assume(alpha >= 1 && alpha <= 100);

    if let Ok((lo, hi)) = snake_output_bounds(x_lo as f64, x_hi as f64, alpha as f64) {
        assert!(lo <= hi, "output_lower must be <= output_upper");
    }
}

// ===========================================================================
// Monotonicity harnesses
// ===========================================================================

// ---------------------------------------------------------------------------
// 13. Wider input produces wider output
// ---------------------------------------------------------------------------

/// Prove: if [a, b] is contained in [c, d], then snake output of [a, b] is
/// contained in snake output of [c, d]. That is, snake_bounds is monotone
/// with respect to interval inclusion.
#[kani::unwind(1)]
#[kani::proof]
fn snake_bounds_monotone_in_input_width() {
    let lo1: i8 = kani::any();
    let hi1: i8 = kani::any();
    let lo2: i8 = kani::any();
    let hi2: i8 = kani::any();
    kani::assume(lo1 <= hi1);
    kani::assume(lo2 <= hi2);
    kani::assume(lo2 <= lo1 && hi1 <= hi2); // [lo1, hi1] contained in [lo2, hi2]
    let alpha: u8 = kani::any();
    kani::assume(alpha >= 1 && alpha <= 50);

    if let (Ok((out_lo1, out_hi1)), Ok((out_lo2, out_hi2))) = (
        snake_output_bounds(lo1 as f64, hi1 as f64, alpha as f64),
        snake_output_bounds(lo2 as f64, hi2 as f64, alpha as f64),
    ) {
        assert!(
            out_lo2 <= out_lo1,
            "wider input lower ({out_lo2}) must be <= narrower ({out_lo1})"
        );
        assert!(
            out_hi1 <= out_hi2,
            "narrower upper ({out_hi1}) must be <= wider ({out_hi2})"
        );
    }
}

// ---------------------------------------------------------------------------
// 14. Larger alpha produces tighter upper bound
// ---------------------------------------------------------------------------

/// Prove: for alpha1 < alpha2, `x_hi + 1/alpha2 < x_hi + 1/alpha1`.
/// Larger alpha means smaller additive term, so tighter upper bound.
#[kani::unwind(1)]
#[kani::proof]
fn snake_bounds_larger_alpha_tighter_upper() {
    let x_lo: i8 = kani::any();
    let x_hi: i8 = kani::any();
    kani::assume(x_lo <= x_hi);
    let alpha1: u8 = kani::any();
    let alpha2: u8 = kani::any();
    kani::assume(alpha1 >= 1 && alpha1 <= 50);
    kani::assume(alpha2 > alpha1 && alpha2 <= 100);

    if let (Ok((_lo1, hi1)), Ok((_lo2, hi2))) = (
        snake_output_bounds(x_lo as f64, x_hi as f64, alpha1 as f64),
        snake_output_bounds(x_lo as f64, x_hi as f64, alpha2 as f64),
    ) {
        assert!(
            hi2 < hi1 + 1e-10,
            "larger alpha ({alpha2}) must give tighter upper ({hi2}) than ({alpha1}, {hi1})"
        );
    }
}

// ---------------------------------------------------------------------------
// 15. Same lower bound regardless of alpha
// ---------------------------------------------------------------------------

/// Prove: `output_lower` is independent of alpha (always equals x_lo).
#[kani::unwind(1)]
#[kani::proof]
fn snake_bounds_lower_independent_of_alpha() {
    let x_lo: i8 = kani::any();
    let x_hi: i8 = kani::any();
    kani::assume(x_lo <= x_hi);
    let alpha1: u8 = kani::any();
    let alpha2: u8 = kani::any();
    kani::assume(alpha1 >= 1 && alpha1 <= 50);
    kani::assume(alpha2 >= 1 && alpha2 <= 100);

    if let (Ok((lo1, _)), Ok((lo2, _))) = (
        snake_output_bounds(x_lo as f64, x_hi as f64, alpha1 as f64),
        snake_output_bounds(x_lo as f64, x_hi as f64, alpha2 as f64),
    ) {
        assert_eq!(lo1, lo2, "lower bound must be same for different alphas");
    }
}

// ===========================================================================
// Special cases
// ===========================================================================

// ---------------------------------------------------------------------------
// 16. Point interval: x_lo == x_hi
// ---------------------------------------------------------------------------

/// Prove: for a point interval [x, x], output is [x, x + 1/alpha].
#[kani::unwind(1)]
#[kani::proof]
fn snake_bounds_point_interval() {
    let x: i8 = kani::any();
    let alpha: u8 = kani::any();
    kani::assume(alpha >= 1 && alpha <= 100);

    if let Ok((lo, hi)) = snake_output_bounds(x as f64, x as f64, alpha as f64) {
        assert_eq!(lo, x as f64, "point lower must equal x");
        let expected_hi = (x as f64) + 1.0 / (alpha as f64);
        assert!(
            (hi - expected_hi).abs() < 1e-10,
            "point upper must equal x + 1/alpha"
        );
        // Output width is exactly 1/alpha.
        let width = hi - lo;
        let expected_width = 1.0 / (alpha as f64);
        assert!(
            (width - expected_width).abs() < 1e-10,
            "width must be 1/alpha"
        );
    }
}

// ---------------------------------------------------------------------------
// 17. Alpha = 1: output upper = x_hi + 1
// ---------------------------------------------------------------------------

/// Prove: with alpha=1, `output_upper = x_hi + 1.0`.
#[kani::unwind(1)]
#[kani::proof]
fn snake_bounds_alpha_one_upper() {
    let x_lo: i8 = kani::any();
    let x_hi: i8 = kani::any();
    kani::assume(x_lo <= x_hi);

    if let Ok((_lo, hi)) = snake_output_bounds(x_lo as f64, x_hi as f64, 1.0) {
        let expected = (x_hi as f64) + 1.0;
        assert!(
            (hi - expected).abs() < 1e-10,
            "alpha=1 upper must be x_hi + 1"
        );
    }
}

// ---------------------------------------------------------------------------
// 18. Output width is always x_hi - x_lo + 1/alpha
// ---------------------------------------------------------------------------

/// Prove: output interval width equals input width plus 1/alpha.
#[kani::unwind(1)]
#[kani::proof]
fn snake_bounds_width_formula() {
    let x_lo: i8 = kani::any();
    let x_hi: i8 = kani::any();
    kani::assume(x_lo <= x_hi);
    let alpha: u8 = kani::any();
    kani::assume(alpha >= 1 && alpha <= 100);

    if let Ok((lo, hi)) = snake_output_bounds(x_lo as f64, x_hi as f64, alpha as f64) {
        let out_width = hi - lo;
        let in_width = (x_hi as f64) - (x_lo as f64);
        let expected_expansion = 1.0 / (alpha as f64);
        assert!(
            (out_width - in_width - expected_expansion).abs() < 1e-10,
            "output width must be input width + 1/alpha"
        );
    }
}
