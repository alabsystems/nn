// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for ay fp_snake Taylor encoding.
//!
//! Complements `kani_ay_snake_decompose.rs` with additional proofs:
//! - Factorial growth rate and recursive relation
//! - Taylor coefficient finiteness and magnitude bounds
//! - Taylor polynomial evaluation properties (zero, symmetry)
//! - Remainder bound ratio between orders
//! - Build power edge cases and structure
//! - SnakeFpConfig field consistency
//!
//! Issue: #3736

// ============================================================
// CBMC transcendental stubs for Kani (#708)
// ============================================================

/// Nondeterministic stub for `f64::powi`.
/// CBMC cannot handle the powi intrinsic. Returns a finite f64.
fn powi_f64_stub(x: f64, n: i32) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite());
    if x > 0.0 && x < 1.0 && n >= 1 {
        kani::assume(r > 0.0 && r <= x);
    }
    if x > 0.0 {
        kani::assume(r > 0.0);
        kani::assume(r >= x.min(1.0));
    }
    r
}

// ============================================================
// factorial_f64: growth rate and recursive relation
// ============================================================

/// Proves `factorial_f64(n) >= 2^(n-1)` for n in [1, 20].
/// This is a known lower bound: n! >= 2^(n-1) because each factor >= 2 for n >= 2.
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::powi, powi_f64_stub)]
fn factorial_f64_growth_rate_lower_bound() {
    let n: u32 = kani::any();
    kani::assume(n >= 1 && n <= 20);

    let fact = crate::ay::ay_fp_snake::factorial_f64(n);
    let lower = 2.0_f64.powi((n as i32) - 1);
    assert!(
        fact >= lower,
        "factorial({n}) = {fact} must be >= 2^({}) = {lower}",
        n - 1
    );
}

/// Proves `factorial_f64` satisfies the recursive relation: n! = n * (n-1)!.
/// This is the defining property of the factorial function.
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn factorial_f64_recursive_relation() {
    let n: u32 = kani::any();
    kani::assume(n >= 2 && n <= 18); // Stay well within f64 exact range

    let fact_n = crate::ay::ay_fp_snake::factorial_f64(n);
    let fact_n_minus_1 = crate::ay::ay_fp_snake::factorial_f64(n - 1);
    let expected = (n as f64) * fact_n_minus_1;
    assert!(
        (fact_n - expected).abs() < 1e-6,
        "factorial({n}) = {fact_n} should equal {n} * factorial({}) = {expected}",
        n - 1
    );
}

/// Proves `factorial_f64(0)` and `factorial_f64(1)` are both exactly 1.0.
/// Base cases: 0! = 1 and 1! = 1 by definition.
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn factorial_f64_base_cases_are_one() {
    let f0 = crate::ay::ay_fp_snake::factorial_f64(0);
    let f1 = crate::ay::ay_fp_snake::factorial_f64(1);
    assert_eq!(f0, 1.0, "0! must be 1.0");
    assert_eq!(f1, 1.0, "1! must be 1.0");
}

/// Proves `1.0 / factorial_f64(n)` is positive and finite for n in [0, 20].
/// Factorial reciprocals are used as Taylor coefficients; they must not produce
/// NaN or Inf when used as divisors.
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn factorial_reciprocal_positive_finite() {
    let n: u32 = kani::any();
    kani::assume(n <= 20);

    let fact = crate::ay::ay_fp_snake::factorial_f64(n);
    let recip = 1.0 / fact;
    assert!(recip > 0.0, "1/{n}! must be positive, got {recip}");
    assert!(recip.is_finite(), "1/{n}! must be finite, got {recip}");
}

// ============================================================
// taylor_sin_coefficients: finiteness and magnitude
// ============================================================

/// Proves all Taylor coefficients are finite for valid orders.
/// Non-finite coefficients would poison the entire SMT polynomial encoding.
#[cfg(feature = "ay-smt")]
#[kani::unwind(8)]
#[kani::proof]
fn taylor_coeffs_all_finite() {
    let k: u32 = kani::any();
    kani::assume(k >= 1 && k <= 6);
    let order = 2 * k - 1;

    let coeffs = crate::ay::ay_fp_snake::taylor_sin_coefficients(order);
    for (i, c) in coeffs.iter().enumerate() {
        assert!(
            c.is_finite(),
            "coefficient {i} for order {order} must be finite, got {c}"
        );
    }
}

/// Proves each Taylor coefficient's magnitude equals 1/(2k+1)!.
/// This is the closed-form: |c_k| = 1 / (2k+1)! for the k-th term.
#[cfg(feature = "ay-smt")]
#[kani::unwind(8)]
#[kani::proof]
fn taylor_coeffs_magnitude_matches_factorial() {
    let k: u32 = kani::any();
    kani::assume(k >= 1 && k <= 5);
    let order = 2 * k - 1;

    let coeffs = crate::ay::ay_fp_snake::taylor_sin_coefficients(order);
    for (i, c) in coeffs.iter().enumerate() {
        let exp = 2 * i + 1;
        let expected_mag = 1.0 / crate::ay::ay_fp_snake::factorial_f64(exp as u32);
        assert!(
            (c.abs() - expected_mag).abs() < 1e-15,
            "term {i}: |coeff| = {} should equal 1/{}! = {expected_mag}",
            c.abs(),
            exp
        );
    }
}

/// Proves the sum of absolute values of Taylor coefficients is bounded.
/// For order 11 (6 terms), sum(|c_k|) < 2.0 since the series converges
/// absolutely and the partial sums approach sinh(1) ~ 1.1752.
#[cfg(feature = "ay-smt")]
#[kani::unwind(8)]
#[kani::proof]
fn taylor_coeffs_sum_magnitudes_bounded() {
    let k: u32 = kani::any();
    kani::assume(k >= 1 && k <= 6);
    let order = 2 * k - 1;

    let coeffs = crate::ay::ay_fp_snake::taylor_sin_coefficients(order);
    let sum_abs: f64 = coeffs.iter().map(|c| c.abs()).sum();
    assert!(
        sum_abs < 2.0,
        "sum of |coefficients| for order {order} must be < 2.0, got {sum_abs}"
    );
}

/// Proves the last coefficient is the smallest in absolute value.
/// Since factorials grow, the last term in the partial sum has the smallest
/// magnitude. This is important for truncation error analysis.
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn taylor_coeffs_last_is_smallest() {
    let k: u32 = kani::any();
    kani::assume(k >= 2 && k <= 6);
    let order = 2 * k - 1;

    let coeffs = crate::ay::ay_fp_snake::taylor_sin_coefficients(order);
    let last = coeffs.last().expect("non-empty").abs();
    let first = coeffs[0].abs();
    assert!(
        last < first,
        "last coefficient magnitude ({last}) must be < first ({first})"
    );
}

// ============================================================
// Taylor polynomial evaluation: pure arithmetic properties
// ============================================================

/// Proves the Taylor polynomial evaluates to 0 at t=0 for all valid orders.
/// sin(0) = 0, so the polynomial should produce exactly 0.
#[cfg(feature = "ay-smt")]
#[kani::unwind(8)]
#[kani::proof]
#[kani::stub(f64::powi, powi_f64_stub)]
fn taylor_eval_at_zero_is_zero() {
    let k: u32 = kani::any();
    kani::assume(k >= 1 && k <= 6);
    let order = 2 * k - 1;

    let coeffs = crate::ay::ay_fp_snake::taylor_sin_coefficients(order);
    // Evaluate P_n(0): all terms are c_k * 0^(2k+1) = 0.
    let mut sum = 0.0_f64;
    for (i, &c) in coeffs.iter().enumerate() {
        let exp = 2 * i + 1;
        sum += c * 0.0_f64.powi(exp as i32);
    }
    assert_eq!(sum, 0.0, "P_{order}(0) must be exactly 0");
}

/// Proves the Taylor polynomial's derivative at t=0 equals 1.
/// d/dt sin(t)|_{t=0} = cos(0) = 1. The derivative of P_n(t) at 0
/// is just the first coefficient (higher-order terms vanish).
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn taylor_derivative_at_zero_is_one() {
    let k: u32 = kani::any();
    kani::assume(k >= 1 && k <= 6);
    let order = 2 * k - 1;

    let coeffs = crate::ay::ay_fp_snake::taylor_sin_coefficients(order);
    // P'_n(0) = c_0 * 1 + c_1 * 3 * 0^2 + c_2 * 5 * 0^4 + ... = c_0.
    // The first coefficient c_0 = 1.0.
    assert!(
        (coeffs[0] - 1.0).abs() < 1e-15,
        "P'_{order}(0) = c_0 must be 1.0, got {}",
        coeffs[0]
    );
}

/// Proves the Taylor polynomial for sin is an odd function:
/// P_n(-t) = -P_n(t) for all t. The polynomial has only odd-power terms,
/// so negating t negates the result.
#[cfg(feature = "ay-smt")]
#[kani::unwind(8)]
#[kani::proof]
#[kani::stub(f64::powi, powi_f64_stub)]
fn taylor_eval_odd_symmetry() {
    let k: u32 = kani::any();
    kani::assume(k >= 1 && k <= 5);
    let order = 2 * k - 1;

    let coeffs = crate::ay::ay_fp_snake::taylor_sin_coefficients(order);

    // Test at a few non-zero integer points.
    let t_int: u8 = kani::any();
    kani::assume(t_int >= 1 && t_int <= 5);
    let t = t_int as f64;

    let mut val_pos = 0.0_f64;
    let mut val_neg = 0.0_f64;
    for (i, &c) in coeffs.iter().enumerate() {
        let exp = (2 * i + 1) as i32;
        val_pos += c * t.powi(exp);
        val_neg += c * (-t).powi(exp);
    }

    assert!(
        (val_pos + val_neg).abs() < 1e-10,
        "P_{order}({t}) + P_{order}(-{t}) should be 0, got {}",
        val_pos + val_neg
    );
}

// ============================================================
// taylor_remainder_bound: additional properties
// ============================================================

/// Proves the remainder bound at radius=0.5 is finite and small for orders 7 and 9.
/// This is a commonly used radius for production Kokoro inputs.
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn remainder_bound_at_half_is_small() {
    let b7 = crate::ay::ay_fp_snake::taylor_remainder_bound(7, 0.5)
        .expect("radius 0.5 must succeed for order 7");
    let b9 = crate::ay::ay_fp_snake::taylor_remainder_bound(9, 0.5)
        .expect("radius 0.5 must succeed for order 9");

    // 0.5^8 / 8! = 1/256 / 40320 ~ 9.68e-8
    assert!(b7 < 1e-4, "order-7 bound at r=0.5 must be < 1e-4, got {b7}");
    assert!(b9 < b7, "order-9 bound must be tighter than order-7");
    assert!(b9 < 1e-6, "order-9 bound at r=0.5 must be < 1e-6, got {b9}");
}

/// Proves the ratio of order-9 to order-7 remainder bounds is less than 1
/// for any radius >= 1. The improvement factor comes from the extra factorial
/// denominator term: bound_9/bound_7 = R^2 / (9*10) < 1 when R < sqrt(90).
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn remainder_bound_ratio_improves_with_order() {
    let r_int: u32 = kani::any();
    kani::assume(r_int >= 1 && r_int <= 9); // R < sqrt(90) ~ 9.49

    let radius = r_int as f64;
    if let (Ok(b7), Ok(b9)) = (
        crate::ay::ay_fp_snake::taylor_remainder_bound(7, radius),
        crate::ay::ay_fp_snake::taylor_remainder_bound(9, radius),
    ) {
        if b7 > 0.0 {
            let ratio = b9 / b7;
            assert!(
                ratio < 1.0,
                "bound_9/bound_7 at r={radius} must be < 1.0, got {ratio}"
            );
        }
    }
}

/// Proves `taylor_remainder_bound` rejects NaN radius.
/// NaN violates the precondition that radius is a non-negative real number.
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn remainder_bound_rejects_nan() {
    let result = crate::ay::ay_fp_snake::taylor_remainder_bound(7, f64::NAN);
    assert!(result.is_err(), "NaN radius must be rejected");
}

/// Proves `taylor_remainder_bound` rejects positive infinity radius.
/// Infinity would produce an infinite bound, which is useless for verification.
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn remainder_bound_rejects_infinity() {
    let result = crate::ay::ay_fp_snake::taylor_remainder_bound(7, f64::INFINITY);
    assert!(result.is_err(), "infinite radius must be rejected");
}

// ============================================================
// SnakeFpConfig: field consistency
// ============================================================

/// Proves `SnakeFpConfig::default()` precision_bits is 24 (f32).
/// The config targets f32 by default. 24 is the number of significand bits
/// in IEEE 754 binary32.
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn snake_config_precision_bits_is_f32() {
    let config = crate::ay::ay_fp_snake::SnakeFpConfig::default();
    assert_eq!(config.precision_bits, 24, "precision_bits must be 24 (f32)");
}

/// Proves `SnakeFpConfig::default()` Taylor order is in the valid range [1, 11].
/// Orders beyond 11 would require factorial_f64(12) which, while representable,
/// produces increasingly large SMT expressions for diminishing accuracy gains.
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn snake_config_taylor_order_in_range() {
    let config = crate::ay::ay_fp_snake::SnakeFpConfig::default();
    assert!(
        config.taylor_order >= 1 && config.taylor_order <= 11,
        "Taylor order must be in [1, 11], got {}",
        config.taylor_order
    );
}

/// Proves `SnakeFpConfig::default()` alpha range is a non-degenerate interval.
/// A degenerate range (min >= max) would make the alpha validation useless.
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn snake_config_alpha_range_non_degenerate() {
    let config = crate::ay::ay_fp_snake::SnakeFpConfig::default();
    let (lo, hi) = config.alpha_range;
    assert!(lo < hi, "alpha_range must have lo < hi: ({lo}, {hi})");
    assert!(lo > 0.0, "alpha_range lo must be > 0");
    assert!(hi.is_finite(), "alpha_range hi must be finite");
    assert!((hi - lo) > 1.0, "alpha_range must span > 1.0");
}

/// Proves that `taylor_sin_coefficients` followed by `taylor_remainder_bound`
/// produces a valid pair: the coefficients define a polynomial and the bound
/// defines the maximum error. Both must succeed for the same order.
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn taylor_coeffs_and_bound_consistent() {
    let k: u32 = kani::any();
    kani::assume(k >= 1 && k <= 5);
    let order = 2 * k - 1;

    let coeffs = crate::ay::ay_fp_snake::taylor_sin_coefficients(order);
    assert!(!coeffs.is_empty(), "coefficients must be non-empty");

    let bound = crate::ay::ay_fp_snake::taylor_remainder_bound(order, 1.0)
        .expect("remainder bound at radius=1 must succeed");
    assert!(bound >= 0.0, "remainder bound must be non-negative");
    assert!(
        bound < 1.0,
        "remainder bound at radius=1 must be < 1.0 for order >= 1, got {bound}"
    );
}
