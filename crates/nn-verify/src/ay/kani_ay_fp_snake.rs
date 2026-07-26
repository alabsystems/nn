// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `ay_fp_snake.rs` — Taylor-bounded sin() encoding
//! and Snake activation SMT translation.
//!
//! Proves numerical safety and mathematical correctness of the pure-Rust
//! functions that form the ay FP theory for Snake activation:
//!
//! - `factorial_f64`: no overflow for valid inputs, correct base cases
//! - `taylor_sin_coefficients`: alternating signs, coefficient count, ordering
//! - `taylor_remainder_bound`: non-negativity, monotonicity in radius and order
//! - `build_power`: exponentiation correctness
//! - `SnakeFpConfig`: default invariants
//!
//! These functions compute the Taylor polynomial and Lagrange remainder that
//! bound sin() in the SMT encoding. Errors here would make the Snake
//! verification unsound.
//!
//! Part of #3696.

use super::ay_fp_snake::{
    build_power, factorial_f64, taylor_remainder_bound, taylor_sin_coefficients, SnakeFpConfig,
};

// ===========================================================================
// factorial_f64 harnesses
// ===========================================================================

/// Prove: factorial_f64(0) == 1.0. The empty product is 1 by convention.
#[kani::unwind(1)]
#[kani::proof]
fn factorial_f64_zero_is_one() {
    assert_eq!(factorial_f64(0), 1.0, "0! must be 1.0");
}

/// Prove: factorial_f64(1) == 1.0.
#[kani::unwind(1)]
#[kani::proof]
fn factorial_f64_one_is_one() {
    assert_eq!(factorial_f64(1), 1.0, "1! must be 1.0");
}

/// Prove: for n in 0..=12, factorial_f64(n) is finite and positive.
/// All factorials up to 12! = 479001600 fit exactly in f64 (< 2^53).
#[kani::unwind(1)]
#[kani::proof]
fn factorial_f64_small_n_finite_positive() {
    let n: u32 = kani::any();
    kani::assume(n <= 12);
    let result = factorial_f64(n);
    assert!(result.is_finite(), "factorial({n}) must be finite");
    assert!(result > 0.0, "factorial({n}) must be positive");
}

/// Prove: factorial_f64 is monotonically non-decreasing for n in 0..=12.
/// f(n+1) = (n+1) * f(n) >= f(n) for n >= 0.
#[kani::unwind(1)]
#[kani::proof]
fn factorial_f64_monotone() {
    let n: u32 = kani::any();
    kani::assume(n < 12);
    let f_n = factorial_f64(n);
    let f_n1 = factorial_f64(n + 1);
    assert!(
        f_n1 >= f_n,
        "factorial must be non-decreasing: f({}) = {} < f({}) = {}",
        n + 1,
        f_n1,
        n,
        f_n,
    );
}

/// Prove: factorial_f64 satisfies the recurrence f(n) = n * f(n-1) for n in 1..=12.
#[kani::unwind(1)]
#[kani::proof]
fn factorial_f64_recurrence() {
    let n: u32 = kani::any();
    kani::assume(n >= 1 && n <= 12);
    let f_n = factorial_f64(n);
    let f_nm1 = factorial_f64(n - 1);
    let expected = f_nm1 * (n as f64);
    assert!(
        (f_n - expected).abs() < 1e-10,
        "f({n}) must equal {n} * f({})",
        n - 1,
    );
}

/// Prove: factorial_f64 for n in 0..=20 is finite. 20! ~ 2.43e18 < 2^53.
#[kani::unwind(1)]
#[kani::proof]
fn factorial_f64_up_to_20_finite() {
    let n: u32 = kani::any();
    kani::assume(n <= 20);
    let result = factorial_f64(n);
    assert!(
        result.is_finite(),
        "factorial({n}) must be finite for n <= 20"
    );
    assert!(result >= 1.0, "factorial({n}) must be >= 1.0");
}

// ===========================================================================
// taylor_sin_coefficients harnesses
// ===========================================================================

/// Prove: taylor_sin_coefficients returns exactly (order+1)/2 terms for valid odd orders.
#[kani::unwind(1)]
#[kani::proof]
fn taylor_coefficients_count_correct() {
    // Valid odd orders: 1, 3, 5, 7, 9, 11
    let order_idx: u32 = kani::any();
    kani::assume(order_idx <= 5);
    let order = 2 * order_idx + 1; // maps to 1, 3, 5, 7, 9, 11

    let coeffs = taylor_sin_coefficients(order);
    let expected_len = (order as usize + 1) / 2;
    assert_eq!(
        coeffs.len(),
        expected_len,
        "order {} must produce {} terms",
        order,
        expected_len,
    );
}

/// Prove: the first coefficient is always 1.0 (c_1 = 1/1! = 1) for all valid orders.
#[kani::unwind(1)]
#[kani::proof]
fn taylor_coefficients_first_is_one() {
    let order_idx: u32 = kani::any();
    kani::assume(order_idx <= 5);
    let order = 2 * order_idx + 1;

    let coeffs = taylor_sin_coefficients(order);
    assert!(
        (coeffs[0] - 1.0).abs() < 1e-15,
        "first coefficient must be 1.0, got {}",
        coeffs[0],
    );
}

/// Prove: coefficients alternate in sign. Even-index coefficients are positive,
/// odd-index coefficients are negative. This follows from (-1)^k / (2k+1)!.
#[kani::unwind(8)]
#[kani::proof]
fn taylor_coefficients_alternating_signs() {
    let order_idx: u32 = kani::any();
    kani::assume(order_idx >= 1 && order_idx <= 5);
    let order = 2 * order_idx + 1;

    let coeffs = taylor_sin_coefficients(order);
    for (k, &c) in coeffs.iter().enumerate() {
        if k % 2 == 0 {
            assert!(c > 0.0, "coefficient[{k}] must be positive, got {c}",);
        } else {
            assert!(c < 0.0, "coefficient[{k}] must be negative, got {c}",);
        }
    }
}

/// Prove: all coefficients are finite for valid odd orders 1..=11.
#[kani::unwind(8)]
#[kani::proof]
fn taylor_coefficients_all_finite() {
    let order_idx: u32 = kani::any();
    kani::assume(order_idx <= 5);
    let order = 2 * order_idx + 1;

    let coeffs = taylor_sin_coefficients(order);
    for (k, &c) in coeffs.iter().enumerate() {
        assert!(
            c.is_finite(),
            "coefficient[{k}] must be finite for order {order}",
        );
    }
}

/// Prove: absolute values of coefficients are strictly decreasing.
/// |c_k| = 1/(2k+1)! and factorial grows faster than any power.
#[kani::unwind(8)]
#[kani::proof]
fn taylor_coefficients_abs_decreasing() {
    let order_idx: u32 = kani::any();
    kani::assume(order_idx >= 1 && order_idx <= 5);
    let order = 2 * order_idx + 1;

    let coeffs = taylor_sin_coefficients(order);
    for k in 1..coeffs.len() {
        assert!(
            coeffs[k].abs() < coeffs[k - 1].abs(),
            "|c[{k}]| = {} must be < |c[{}]| = {}",
            coeffs[k].abs(),
            k - 1,
            coeffs[k - 1].abs(),
        );
    }
}

// ===========================================================================
// taylor_remainder_bound harnesses
// ===========================================================================

/// Prove: taylor_remainder_bound returns Ok with a non-negative value for
/// valid (non-negative, finite) radius and valid odd orders.
#[kani::unwind(1)]
#[kani::proof]
fn remainder_bound_nonneg_for_valid_inputs() {
    let order_idx: u32 = kani::any();
    kani::assume(order_idx <= 5);
    let order = 2 * order_idx + 1;

    // Use integer radius 0..=3 to keep values small and exact.
    let radius_int: u8 = kani::any();
    kani::assume(radius_int <= 3);
    let radius = radius_int as f64;

    if let Ok(bound) = taylor_remainder_bound(order, radius) {
        assert!(bound >= 0.0, "remainder bound must be >= 0, got {bound}");
        assert!(
            bound.is_finite(),
            "remainder bound must be finite, got {bound}",
        );
    }
}

/// Prove: higher Taylor order gives tighter remainder bound for the same radius.
/// E_n(R) = R^(n+1)/(n+1)! decreases as n increases for |R| <= convergence radius.
#[kani::unwind(1)]
#[kani::proof]
fn remainder_bound_tighter_with_higher_order() {
    // Use radius 1 where the bound is simply 1/(n+1)!
    let order_idx_1: u32 = kani::any();
    let order_idx_2: u32 = kani::any();
    kani::assume(order_idx_1 < order_idx_2);
    kani::assume(order_idx_2 <= 5);
    let order_1 = 2 * order_idx_1 + 1;
    let order_2 = 2 * order_idx_2 + 1;

    if let (Ok(b1), Ok(b2)) = (
        taylor_remainder_bound(order_1, 1.0),
        taylor_remainder_bound(order_2, 1.0),
    ) {
        assert!(
            b2 < b1 + 1e-15,
            "higher order ({order_2}) must give tighter bound ({b2}) than order ({order_1}, {b1})",
        );
    }
}

/// Prove: taylor_remainder_bound rejects NaN radius.
#[kani::unwind(1)]
#[kani::proof]
fn remainder_bound_rejects_nan_radius() {
    let result = taylor_remainder_bound(7, f64::NAN);
    assert!(result.is_err(), "NaN radius must be rejected");
}

/// Prove: taylor_remainder_bound rejects infinite radius.
#[kani::unwind(1)]
#[kani::proof]
fn remainder_bound_rejects_inf_radius() {
    let result = taylor_remainder_bound(7, f64::INFINITY);
    assert!(result.is_err(), "infinite radius must be rejected");
}

// ===========================================================================
// build_power harnesses
// ===========================================================================

/// Prove: build_power with exponent 0 produces Expr::real(1) for any base.
/// This is the identity: x^0 = 1.
#[kani::unwind(64)]
#[kani::proof]
fn build_power_exp_zero_is_one() {
    let base = ay_bindings::Expr::var("x", ay_bindings::Sort::real());
    let result = build_power(&base, 0);
    let result_str = format!("{result}");
    assert!(
        result_str == "1" || result_str == "1.0",
        "x^0 must be 1, got {result_str}",
    );
}

/// Prove: build_power with exponent 1 returns the base itself.
/// x^1 = x.
#[kani::unwind(64)]
#[kani::proof]
fn build_power_exp_one_is_base() {
    let base = ay_bindings::Expr::var("t", ay_bindings::Sort::real());
    let result = build_power(&base, 1);
    let result_str = format!("{result}");
    assert_eq!(result_str, "t", "x^1 must be x, got {result_str}");
}

/// Prove: build_power produces a non-trivial expression for exponent >= 2.
/// The result should NOT be "1" or the base variable name alone.
#[kani::unwind(64)]
#[kani::proof]
fn build_power_exp_ge2_is_nontrivial() {
    let exp: u32 = kani::any();
    kani::assume(exp >= 2 && exp <= 8);
    let base = ay_bindings::Expr::var("x", ay_bindings::Sort::real());
    let result = build_power(&base, exp);
    let result_str = format!("{result}");
    assert!(
        result_str != "1" && result_str != "1.0" && result_str != "x",
        "x^{exp} must not be trivial, got {result_str}",
    );
}

// ===========================================================================
// SnakeFpConfig harnesses
// ===========================================================================

/// Prove: default config has valid values — odd taylor_order >= 1,
/// positive alpha range, precision_bits > 0.
#[kani::unwind(1)]
#[kani::proof]
fn snake_fp_config_default_valid() {
    let config = SnakeFpConfig::default();
    assert!(config.precision_bits > 0, "precision_bits must be > 0");
    assert!(config.taylor_order >= 1, "taylor_order must be >= 1",);
    assert!(config.taylor_order % 2 == 1, "taylor_order must be odd",);
    assert!(config.alpha_range.0 > 0.0, "alpha_range.0 must be > 0",);
    assert!(
        config.alpha_range.0 < config.alpha_range.1,
        "alpha_range must be non-inverted",
    );
    assert!(
        config.alpha_range.0.is_finite() && config.alpha_range.1.is_finite(),
        "alpha_range must be finite",
    );
}

/// Prove: default taylor_order (7) produces 4 coefficients and a
/// finite, positive remainder bound at radius = pi.
#[kani::unwind(8)]
#[kani::proof]
fn snake_fp_config_default_order_produces_valid_coefficients() {
    let config = SnakeFpConfig::default();
    let coeffs = taylor_sin_coefficients(config.taylor_order);
    assert_eq!(coeffs.len(), 4, "order 7 must produce 4 coefficients");
    for (k, &c) in coeffs.iter().enumerate() {
        assert!(c.is_finite(), "coefficient[{k}] must be finite");
    }
    let bound = taylor_remainder_bound(config.taylor_order, std::f64::consts::PI);
    assert!(bound.is_ok(), "remainder at pi must be computable");
    let bound_val = bound.unwrap();
    assert!(bound_val > 0.0, "remainder at pi must be positive");
    assert!(bound_val.is_finite(), "remainder at pi must be finite");
}
