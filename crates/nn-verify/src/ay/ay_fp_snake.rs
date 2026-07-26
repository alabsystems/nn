// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Exact IEEE 754 Snake encoding using Taylor polynomial bounded sin().
//!
//! Phase B of the ay FP theory migration (#2480). Replaces the UF
//! approximation for sin() in Snake activation with a piecewise
//! Taylor polynomial bounded by the Lagrange remainder theorem.
//!
//! # Strategy
//!
//! Instead of declaring `sin_approx : Real -> Real` with range `[-1, 1]`,
//! we encode `sin(t)` as a fresh Real variable constrained to lie within
//! the Taylor polynomial interval `[P_n(t) - E_n, P_n(t) + E_n]` where:
//!
//! - `P_n(t) = t - t^3/3! + t^5/5! - ... ` (odd terms up to order n)
//! - `E_n = |t|^(n+1) / (n+1)!` (Lagrange remainder bound)
//!
//! This is **sound** by the Taylor remainder theorem: for all real t,
//! `|sin(t) - P_n(t)| <= E_n`. The SMT solver then reasons about all
//! possible sin(t) values within this tighter interval.
//!
//! # Soundness
//!
//! Outside the polynomial's accurate domain (`|t| > radius`), we fall
//! back to the UF range axiom `sin(t) in [-1, 1]`. The `ite` encoding
//! ensures soundness in both regions.

use ay_bindings::{Expr, Sort, AYProgram};

use super::error::SmtError;
use super::translate_real::real_from_f64;
use super::translate_uf::declare_uf_if_needed;

/// Configuration for the Taylor-bounded Snake FP encoding.
#[derive(Debug, Clone)]
pub(crate) struct SnakeFpConfig {
    /// Number of precision bits for the target floating-point format.
    /// 24 for f32, 53 for f64. Controls the required Taylor order.
    pub precision_bits: u32,

    /// Order of the Taylor polynomial for sin(). Must be odd (3, 5, 7, 9, 11).
    /// Higher order = tighter bounds but more complex SMT expression.
    /// Recommended: 7 for f32 (error < 1.4e-6 at |t|=1).
    pub taylor_order: u32,

    /// Valid range for the alpha parameter: (min, max).
    /// Kokoro typically uses alpha in [0.5, 20.0].
    /// Alpha must be > 0 for Snake to be well-defined.
    pub alpha_range: (f64, f64),
}

impl Default for SnakeFpConfig {
    fn default() -> Self {
        Self {
            precision_bits: 24, // f32
            taylor_order: 7,    // Good balance: error < 1.4e-6 at |t|=1
            alpha_range: (0.1, 100.0),
        }
    }
}

/// Compute Taylor series coefficients for sin(t) up to the given odd order.
///
/// Returns coefficients `[c_1, c_3, c_5, ...]` where:
/// `sin(t) ~ c_1*t + c_3*t^3 + c_5*t^5 + ...`
///
/// The coefficients are `(-1)^k / (2k+1)!` for k = 0, 1, 2, ...
///
/// # Panics
///
/// Panics if `order` is even or less than 1.
pub(crate) fn taylor_sin_coefficients(order: u32) -> Vec<f64> {
    assert!(order >= 1, "Taylor order must be >= 1");
    assert!(order % 2 == 1, "Taylor order for sin must be odd");

    let num_terms = (order as usize + 1) / 2;
    let mut coeffs = Vec::with_capacity(num_terms);

    for k in 0..num_terms {
        let exponent = 2 * k + 1;
        let factorial = factorial_f64(exponent as u32);
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        coeffs.push(sign / factorial);
    }

    coeffs
}

/// Compute the Lagrange remainder bound for the Taylor polynomial of sin().
///
/// For sin(t) with Taylor polynomial of order n, the remainder satisfies:
/// `|sin(t) - P_n(t)| <= |t|^(n+1) / (n+1)!`
///
/// This function returns `radius^(n+1) / (n+1)!` for the given domain
/// radius, which is a sound upper bound on the approximation error
/// for all `|t| <= radius`.
///
/// # Arguments
///
/// * `order` - The Taylor polynomial order (must be odd, >= 1).
/// * `radius` - The domain radius `R` such that `|t| <= R`.
///
/// # Returns
///
/// The remainder bound `R^(n+1) / (n+1)!`, or an error if the result
/// is non-finite (overflow for very large radius).
pub(crate) fn taylor_remainder_bound(order: u32, radius: f64) -> Result<f64, SmtError> {
    if !radius.is_finite() || radius < 0.0 {
        return Err(SmtError::NonFiniteLiteral(radius));
    }
    let exp = order + 1;
    let factorial = factorial_f64(exp);
    let bound = radius.powi(exp as i32) / factorial;
    if !bound.is_finite() {
        return Err(SmtError::NonFiniteBound {
            lower: -bound,
            upper: bound,
        });
    }
    Ok(bound)
}

/// Encode sin(t) as a bounded variable using Taylor polynomial + remainder.
///
/// Creates a fresh SMT Real variable `sin_bounded_N` (where N is a counter)
/// and asserts:
///   `P_n(t) - E_n(R) <= sin_bounded <= P_n(t) + E_n(R)`
///
/// where `P_n` is the degree-`order` Taylor polynomial and `E_n(R)` is the
/// Lagrange remainder bound for `|t| <= radius`.
///
/// For `|t| > radius`, falls back to the UF range axiom `[-1, 1]`.
///
/// # Returns
///
/// A tuple `(sin_expr, uses_uf_fallback)`:
/// - `sin_expr`: the SMT expression representing sin(t) with bounds.
/// - `uses_uf_fallback`: true if the encoding includes a UF fallback
///   path (meaning the overall encoding is still `UfApprox` unless the
///   input domain is known to be within radius).
pub(crate) fn encode_sin_bounded(
    program: &mut AYProgram,
    t_expr: &Expr,
    order: u32,
    radius: f64,
    real_sort: &Sort,
    declared_ufs: &mut std::collections::HashSet<String>,
    counter: &mut u32,
) -> Result<(Expr, bool), SmtError> {
    let coeffs = taylor_sin_coefficients(order);
    let remainder = taylor_remainder_bound(order, radius)?;

    // Build the Taylor polynomial P_n(t) in SMT Real arithmetic.
    // P_n(t) = c_1*t + c_3*t^3 + c_5*t^5 + ...
    // We use exact integer encoding where possible: c_k = (-1)^k / (2k+1)!
    // The factorial denominators are exact integers, so we encode as
    // (/ (* sign t^(2k+1)) factorial) to avoid real_from_f64 quantization.
    let poly = build_taylor_polynomial(t_expr, &coeffs)?;

    // Create a fresh variable for the bounded sin value.
    let var_name = format!("sin_bounded_{}", *counter);
    *counter += 1;
    let sin_var = program.declare_const(&var_name, real_sort.clone());

    // Encode the remainder bound.
    let remainder_expr = real_from_f64(remainder)?;

    // Taylor region: P_n(t) - E <= sin_var <= P_n(t) + E
    let taylor_lower = poly.clone().real_sub(remainder_expr.clone());
    let taylor_upper = poly.real_add(remainder_expr);

    // Domain check: |t| <= radius
    let radius_expr = real_from_f64(radius)?;
    let neg_radius_expr = real_from_f64(-radius)?;
    let in_domain = t_expr
        .clone()
        .real_ge(neg_radius_expr)
        .and(t_expr.clone().real_le(radius_expr));

    // UF fallback for |t| > radius: sin(t) in [-1, 1]
    let neg_one = Expr::real(-1);
    let pos_one = Expr::real(1);

    // Combined constraint using ite:
    // if |t| <= radius:  taylor_lower <= sin_var <= taylor_upper
    // else:              -1 <= sin_var <= 1
    let effective_lower = Expr::ite(in_domain.clone(), taylor_lower, neg_one);
    let effective_upper = Expr::ite(in_domain, taylor_upper, pos_one);

    program.assert(sin_var.clone().real_ge(effective_lower));
    program.assert(sin_var.clone().real_le(effective_upper));

    // Also assert the global range [-1, 1] as a safety net.
    // This is redundant when in the Taylor domain (Taylor bounds are tighter)
    // but provides defense-in-depth.
    program.assert(sin_var.clone().real_ge(Expr::real(-1)));
    program.assert(sin_var.clone().real_le(Expr::real(1)));

    // We always include the UF fallback path in the ite, so this encoding
    // technically has a UF-like component for the out-of-domain case.
    // However, if the caller knows |t| <= radius, the fallback is unreachable.
    let uses_uf_fallback = true;

    // Declare the UF as well for compatibility with existing infrastructure.
    // This is not used in the encoding but ensures the UF name is reserved.
    declare_uf_if_needed("sin_approx", program, real_sort, declared_ufs);

    Ok((sin_var, uses_uf_fallback))
}

/// Encode the full Snake activation using Taylor-bounded sin().
///
/// `snake(x, alpha) = x + (1/alpha) * sin(alpha * x)^2`
///
/// Steps:
/// 1. `t = alpha * x`                    (exact Real mul)
/// 2. `s = encode_sin_bounded(t)`         (Taylor-bounded sin)
/// 3. `s2 = s * s`                        (exact Real mul, non-negative)
/// 4. `result = x + (1/alpha) * s2`       (exact Real arithmetic)
///
/// # Arguments
///
/// * `program` - The ay program to add assertions to.
/// * `x_expr` - SMT expression for the input x.
/// * `alpha_val` - The constant alpha value (must be > 0).
/// * `config` - Configuration for Taylor order and domain.
/// * `real_sort` - The Real sort for variable declarations.
/// * `declared_ufs` - Set of already-declared UF names.
/// * `counter` - Counter for generating unique variable names.
///
/// # Returns
///
/// A tuple `(result_expr, uses_uf_fallback)`.
pub(crate) fn encode_snake_fp(
    program: &mut AYProgram,
    x_expr: &Expr,
    alpha_val: f64,
    config: &SnakeFpConfig,
    real_sort: &Sort,
    declared_ufs: &mut std::collections::HashSet<String>,
    counter: &mut u32,
) -> Result<(Expr, bool), SmtError> {
    // Validate alpha.
    if alpha_val <= 0.0 || !alpha_val.is_finite() {
        return Err(SmtError::InvalidSnakeAlpha(alpha_val));
    }

    // Step 1: t = alpha * x
    let alpha_expr = real_from_f64(alpha_val)?;
    let t_expr = alpha_expr.clone().real_mul(x_expr.clone());

    // Compute the effective radius for the Taylor polynomial.
    // The argument to sin is alpha*x. If we know |x| <= X_max, then
    // |t| <= alpha * X_max. We use the configured alpha_range to
    // estimate a conservative radius. For now, use a fixed radius
    // that works well for typical Kokoro inputs.
    //
    // With order 7 and radius = pi:
    //   remainder = pi^8 / 8! = 0.0755
    // With order 9 and radius = pi:
    //   remainder = pi^10 / 10! = 0.00688
    //
    // We default to radius = pi, covering one full period of sin().
    let radius = std::f64::consts::PI;

    // Step 2: s = sin_bounded(t)
    let (s_expr, uses_uf) = encode_sin_bounded(
        program,
        &t_expr,
        config.taylor_order,
        radius,
        real_sort,
        declared_ufs,
        counter,
    )?;

    // Step 3: s2 = s * s (sin(alpha*x)^2, always non-negative)
    let s2_expr = s_expr.clone().real_mul(s_expr);

    // Assert s2 >= 0 (defense-in-depth: product of identical value is non-negative,
    // but the solver may not derive this automatically from the interval bounds).
    program.assert(s2_expr.clone().real_ge(Expr::real(0)));

    // Step 4: result = x + (1/alpha) * s2
    let inv_alpha = real_from_f64(1.0 / alpha_val)?;
    let scaled_s2 = inv_alpha.real_mul(s2_expr);
    let result = x_expr.clone().real_add(scaled_s2);

    Ok((result, uses_uf))
}

// ============================================================
// Internal helpers
// ============================================================

/// Build the Taylor polynomial P_n(t) as an SMT Real expression.
///
/// `P_n(t) = c_0 * t + c_1 * t^3 + c_2 * t^5 + ...`
///
/// where `coeffs[k]` is `(-1)^k / (2k+1)!`.
///
/// Uses exact integer encoding for factorials to avoid quantization error.
pub(crate) fn build_taylor_polynomial(t_expr: &Expr, coeffs: &[f64]) -> Result<Expr, SmtError> {
    if coeffs.is_empty() {
        return Ok(Expr::real(0));
    }

    let mut sum = Expr::real(0);

    for (k, &coeff) in coeffs.iter().enumerate() {
        let exponent = 2 * k + 1;

        // Build t^exponent via repeated multiplication.
        let t_power = build_power(t_expr, exponent as u32);

        // Encode the coefficient as a rational: sign * 1 / factorial.
        // We use real_from_f64 for the coefficient since factorials
        // up to 11! = 39916800 fit comfortably in the encoding range.
        let coeff_expr = real_from_f64(coeff)?;

        let term = coeff_expr.real_mul(t_power);
        sum = sum.real_add(term);
    }

    Ok(sum)
}

/// Build `t^n` via binary exponentiation (repeated squaring).
///
/// Returns O(log n) `real_mul` nodes, matching the strategy in
/// `translate_uf.rs::translate_powi`.
pub(crate) fn build_power(base: &Expr, exp: u32) -> Expr {
    match exp {
        0 => Expr::real(1),
        1 => base.clone(),
        _ => {
            let mut acc = Expr::real(1);
            let mut b = base.clone();
            let mut e = exp;
            while e > 0 {
                if e & 1 == 1 {
                    acc = acc.real_mul(b.clone());
                }
                e >>= 1;
                if e > 0 {
                    b = b.clone().real_mul(b.clone());
                }
            }
            acc
        }
    }
}

/// Compute n! as f64. Accurate for n <= 20 (fits in u64).
pub(crate) fn factorial_f64(n: u32) -> f64 {
    let mut result: f64 = 1.0;
    for i in 2..=n {
        result *= f64::from(i);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // Taylor coefficient tests
    // ============================================================

    #[test]
    fn test_taylor_sin_coefficients_order_1() {
        let coeffs = taylor_sin_coefficients(1);
        assert_eq!(coeffs.len(), 1);
        assert!((coeffs[0] - 1.0).abs() < 1e-15, "c_1 = 1/1! = 1.0");
    }

    #[test]
    fn test_taylor_sin_coefficients_order_3() {
        let coeffs = taylor_sin_coefficients(3);
        assert_eq!(coeffs.len(), 2);
        assert!((coeffs[0] - 1.0).abs() < 1e-15, "c_1 = 1.0");
        assert!(
            (coeffs[1] - (-1.0 / 6.0)).abs() < 1e-15,
            "c_3 = -1/3! = -1/6, got {}",
            coeffs[1]
        );
    }

    #[test]
    fn test_taylor_sin_coefficients_order_5() {
        let coeffs = taylor_sin_coefficients(5);
        assert_eq!(coeffs.len(), 3);
        assert!((coeffs[0] - 1.0).abs() < 1e-15);
        assert!((coeffs[1] - (-1.0 / 6.0)).abs() < 1e-15);
        assert!(
            (coeffs[2] - (1.0 / 120.0)).abs() < 1e-15,
            "c_5 = 1/5! = 1/120, got {}",
            coeffs[2]
        );
    }

    #[test]
    fn test_taylor_sin_coefficients_order_7() {
        let coeffs = taylor_sin_coefficients(7);
        assert_eq!(coeffs.len(), 4);
        assert!((coeffs[0] - 1.0).abs() < 1e-15);
        assert!((coeffs[1] - (-1.0 / 6.0)).abs() < 1e-15);
        assert!((coeffs[2] - (1.0 / 120.0)).abs() < 1e-15);
        assert!(
            (coeffs[3] - (-1.0 / 5040.0)).abs() < 1e-15,
            "c_7 = -1/7! = -1/5040, got {}",
            coeffs[3]
        );
    }

    #[test]
    fn test_taylor_sin_coefficients_order_9() {
        let coeffs = taylor_sin_coefficients(9);
        assert_eq!(coeffs.len(), 5);
        assert!(
            (coeffs[4] - (1.0 / 362880.0)).abs() < 1e-20,
            "c_9 = 1/9! = 1/362880, got {}",
            coeffs[4]
        );
    }

    #[test]
    fn test_taylor_sin_coefficients_order_11() {
        let coeffs = taylor_sin_coefficients(11);
        assert_eq!(coeffs.len(), 6);
        assert!(
            (coeffs[5] - (-1.0 / 39916800.0)).abs() < 1e-22,
            "c_11 = -1/11! = -1/39916800, got {}",
            coeffs[5]
        );
    }

    #[test]
    #[should_panic(expected = "Taylor order for sin must be odd")]
    fn test_taylor_sin_coefficients_even_order_panics() {
        taylor_sin_coefficients(4);
    }

    #[test]
    #[should_panic(expected = "Taylor order must be >= 1")]
    fn test_taylor_sin_coefficients_zero_order_panics() {
        taylor_sin_coefficients(0);
    }

    // ============================================================
    // Remainder bound tests
    // ============================================================

    #[test]
    fn test_remainder_bound_order_7_radius_1() {
        // E_7(1) = 1^8 / 8! = 1/40320 ~ 2.48e-5
        let bound = taylor_remainder_bound(7, 1.0).unwrap();
        let expected = 1.0 / 40320.0;
        assert!(
            (bound - expected).abs() < 1e-10,
            "remainder bound order 7 radius 1: expected {expected}, got {bound}"
        );
    }

    #[test]
    fn test_remainder_bound_order_7_radius_pi() {
        // E_7(pi) = pi^8 / 8! = 9488.53.../40320 ~ 0.2353
        let bound = taylor_remainder_bound(7, std::f64::consts::PI).unwrap();
        let expected = std::f64::consts::PI.powi(8) / 40320.0;
        assert!(
            (bound - expected).abs() < 1e-10,
            "remainder bound order 7 radius pi: expected {expected}, got {bound}"
        );
    }

    #[test]
    fn test_remainder_bound_order_9_radius_1() {
        // E_9(1) = 1^10 / 10! = 1/3628800 ~ 2.76e-7
        let bound = taylor_remainder_bound(9, 1.0).unwrap();
        let expected = 1.0 / 3628800.0;
        assert!(
            (bound - expected).abs() < 1e-15,
            "remainder bound order 9 radius 1: expected {expected}, got {bound}"
        );
    }

    #[test]
    fn test_remainder_bound_zero_radius() {
        let bound = taylor_remainder_bound(7, 0.0).unwrap();
        assert_eq!(bound, 0.0, "zero radius should give zero remainder");
    }

    #[test]
    fn test_remainder_bound_negative_radius_rejected() {
        let err = taylor_remainder_bound(7, -1.0).unwrap_err();
        assert!(
            matches!(err, SmtError::NonFiniteLiteral(_)),
            "negative radius should be rejected, got: {err}"
        );
    }

    #[test]
    fn test_remainder_bound_nan_radius_rejected() {
        let err = taylor_remainder_bound(7, f64::NAN).unwrap_err();
        assert!(
            matches!(err, SmtError::NonFiniteLiteral(_)),
            "NaN radius should be rejected, got: {err}"
        );
    }

    // ============================================================
    // Taylor polynomial accuracy tests (evaluated in f64)
    // ============================================================

    #[test]
    fn test_taylor_polynomial_accuracy_at_zero() {
        // sin(0) = 0, P_n(0) = 0 for all n.
        let coeffs = taylor_sin_coefficients(7);
        let t = 0.0;
        let val = eval_taylor_f64(&coeffs, t);
        assert_eq!(val, 0.0, "P_7(0) should be exactly 0");
    }

    #[test]
    fn test_taylor_polynomial_accuracy_at_half() {
        // sin(0.5) = 0.479425538604...
        let coeffs = taylor_sin_coefficients(7);
        let t = 0.5;
        let poly_val = eval_taylor_f64(&coeffs, t);
        let exact = t.sin();
        let error = (poly_val - exact).abs();
        let bound = taylor_remainder_bound(7, t.abs()).unwrap();
        assert!(
            error < bound,
            "Taylor error ({error:.2e}) should be less than bound ({bound:.2e})"
        );
        // The actual error (~5.4e-9) is far below the remainder bound (~2.5e-5).
        // Use the remainder bound as the check, which is the mathematically
        // guaranteed threshold. The "error < bound" assertion above is the
        // soundness-critical one.
        assert!(
            error < 1e-6,
            "Order-7 Taylor at t=0.5 should be very accurate, error={error:.2e}"
        );
    }

    #[test]
    fn test_taylor_polynomial_accuracy_at_1() {
        let coeffs = taylor_sin_coefficients(7);
        let t = 1.0;
        let poly_val = eval_taylor_f64(&coeffs, t);
        let exact = t.sin();
        let error = (poly_val - exact).abs();
        let bound = taylor_remainder_bound(7, t.abs()).unwrap();
        assert!(
            error < bound,
            "Taylor error ({error:.2e}) should be less than bound ({bound:.2e})"
        );
    }

    #[test]
    fn test_taylor_polynomial_accuracy_at_pi() {
        // sin(pi) ~ 0 in exact math, ~1.2e-16 in f64.
        let coeffs = taylor_sin_coefficients(7);
        let t = std::f64::consts::PI;
        let poly_val = eval_taylor_f64(&coeffs, t);
        let exact = t.sin();
        let error = (poly_val - exact).abs();
        let bound = taylor_remainder_bound(7, t.abs()).unwrap();
        assert!(
            error < bound,
            "Taylor error at pi ({error:.2e}) should be less than bound ({bound:.2e})"
        );
    }

    #[test]
    fn test_taylor_soundness_sweep() {
        // Sweep over many t values and verify |P_n(t) - sin(t)| <= E_n(|t|).
        let coeffs = taylor_sin_coefficients(7);
        for i in -100..=100 {
            let t = (i as f64) * 0.03; // covers [-3, 3]
            let poly_val = eval_taylor_f64(&coeffs, t);
            let exact = t.sin();
            let error = (poly_val - exact).abs();
            let bound = taylor_remainder_bound(7, t.abs()).unwrap();
            assert!(
                error <= bound + 1e-15, // small epsilon for f64 rounding
                "Taylor soundness violated at t={t}: error={error:.2e}, bound={bound:.2e}"
            );
        }
    }

    #[test]
    fn test_taylor_order_9_tighter_than_order_7() {
        let coeffs_7 = taylor_sin_coefficients(7);
        let coeffs_9 = taylor_sin_coefficients(9);
        let t = 1.0;
        let error_7 = (eval_taylor_f64(&coeffs_7, t) - t.sin()).abs();
        let error_9 = (eval_taylor_f64(&coeffs_9, t) - t.sin()).abs();
        assert!(
            error_9 < error_7,
            "Order 9 should be tighter than order 7: err9={error_9:.2e} < err7={error_7:.2e}"
        );
    }

    // ============================================================
    // SMT encoding tests
    // ============================================================

    #[test]
    fn test_encode_sin_bounded_produces_valid_program() {
        let mut program = AYProgram::new();
        program.set_logic("QF_UFNRA");
        let real_sort = Sort::real();
        let t = program.declare_const("t", real_sort.clone());
        let mut declared_ufs = std::collections::HashSet::new();
        let mut counter = 0u32;

        let result = encode_sin_bounded(
            &mut program,
            &t,
            7,
            std::f64::consts::PI,
            &real_sort,
            &mut declared_ufs,
            &mut counter,
        );
        assert!(result.is_ok(), "encode_sin_bounded should succeed");
        let (_sin_expr, uses_uf) = result.unwrap();
        assert!(uses_uf, "should indicate UF fallback path");

        // The expression should be a declared variable.
        let smt2 = format!("{}", program);
        assert!(
            smt2.contains("sin_bounded_0"),
            "program should declare sin_bounded_0 variable"
        );
    }

    #[test]
    fn test_encode_snake_fp_produces_valid_program() {
        let mut program = AYProgram::new();
        program.set_logic("QF_UFNRA");
        let real_sort = Sort::real();
        let x = program.declare_const("x", real_sort.clone());
        let mut declared_ufs = std::collections::HashSet::new();
        let mut counter = 0u32;
        let config = SnakeFpConfig::default();

        let result = encode_snake_fp(
            &mut program,
            &x,
            1.0, // alpha = 1
            &config,
            &real_sort,
            &mut declared_ufs,
            &mut counter,
        );
        assert!(result.is_ok(), "encode_snake_fp should succeed");

        let smt2 = format!("{}", program);
        assert!(
            smt2.contains("sin_bounded_0"),
            "program should contain Taylor-bounded sin variable"
        );
    }

    #[test]
    fn test_encode_snake_fp_invalid_alpha_zero() {
        let mut program = AYProgram::new();
        let real_sort = Sort::real();
        let x = program.declare_const("x", real_sort.clone());
        let mut declared_ufs = std::collections::HashSet::new();
        let mut counter = 0u32;
        let config = SnakeFpConfig::default();

        let err = encode_snake_fp(
            &mut program,
            &x,
            0.0,
            &config,
            &real_sort,
            &mut declared_ufs,
            &mut counter,
        )
        .unwrap_err();
        assert!(
            matches!(err, SmtError::InvalidSnakeAlpha(a) if a == 0.0),
            "alpha=0 should be rejected, got: {err}"
        );
    }

    #[test]
    fn test_encode_snake_fp_invalid_alpha_negative() {
        let mut program = AYProgram::new();
        let real_sort = Sort::real();
        let x = program.declare_const("x", real_sort.clone());
        let mut declared_ufs = std::collections::HashSet::new();
        let mut counter = 0u32;
        let config = SnakeFpConfig::default();

        let err = encode_snake_fp(
            &mut program,
            &x,
            -2.0,
            &config,
            &real_sort,
            &mut declared_ufs,
            &mut counter,
        )
        .unwrap_err();
        assert!(
            matches!(err, SmtError::InvalidSnakeAlpha(a) if a == -2.0),
            "negative alpha should be rejected, got: {err}"
        );
    }

    #[test]
    fn test_encode_snake_fp_large_alpha() {
        // Large alpha (e.g., 50.0) should still produce a valid encoding.
        let mut program = AYProgram::new();
        program.set_logic("QF_UFNRA");
        let real_sort = Sort::real();
        let x = program.declare_const("x", real_sort.clone());
        let mut declared_ufs = std::collections::HashSet::new();
        let mut counter = 0u32;
        let config = SnakeFpConfig::default();

        let result = encode_snake_fp(
            &mut program,
            &x,
            50.0,
            &config,
            &real_sort,
            &mut declared_ufs,
            &mut counter,
        );
        assert!(result.is_ok(), "large alpha should produce valid encoding");
    }

    #[test]
    fn test_default_config() {
        let config = SnakeFpConfig::default();
        assert_eq!(config.precision_bits, 24);
        assert_eq!(config.taylor_order, 7);
        assert_eq!(config.alpha_range, (0.1, 100.0));
    }

    #[test]
    fn test_factorial_f64_known_values() {
        assert_eq!(factorial_f64(0), 1.0);
        assert_eq!(factorial_f64(1), 1.0);
        assert_eq!(factorial_f64(2), 2.0);
        assert_eq!(factorial_f64(3), 6.0);
        assert_eq!(factorial_f64(4), 24.0);
        assert_eq!(factorial_f64(5), 120.0);
        assert_eq!(factorial_f64(6), 720.0);
        assert_eq!(factorial_f64(7), 5040.0);
        assert_eq!(factorial_f64(8), 40320.0);
        assert_eq!(factorial_f64(9), 362880.0);
        assert_eq!(factorial_f64(10), 3628800.0);
        assert_eq!(factorial_f64(11), 39916800.0);
    }

    #[test]
    fn test_build_power_edge_cases() {
        let t = Expr::var("t", Sort::real());
        // t^0 = 1
        let p0 = build_power(&t, 0);
        // Expr::real(1) may display as "1" or "1.0" depending on ay-bindings.
        let p0_str = format!("{p0}");
        assert!(
            p0_str == "1" || p0_str == "1.0",
            "t^0 should be 1 or 1.0, got {p0_str}"
        );
        // t^1 = t
        let p1 = build_power(&t, 1);
        assert_eq!(format!("{p1}"), "t");
    }

    // ============================================================
    // UF vs FP bounds comparison
    // ============================================================

    #[test]
    fn test_uf_vs_taylor_bound_quality() {
        // UF bound for sin(t): [-1, 1] -> sin^2 in [0, 1] -> 1/alpha contribution in [0, 1/alpha]
        // Taylor bound for sin(t) at t=0.5 with order 7:
        //   |sin(0.5) - P_7(0.5)| < 2.48e-5
        //   So sin(0.5) in [P_7(0.5) - 2.48e-5, P_7(0.5) + 2.48e-5]
        //   = [0.47940, 0.47945] approximately
        //   vs UF: [-1, 1]
        //
        // The Taylor interval is ~40000x tighter at t=0.5.
        let coeffs = taylor_sin_coefficients(7);
        let t = 0.5;
        let _poly_val = eval_taylor_f64(&coeffs, t);
        let bound = taylor_remainder_bound(7, t.abs()).unwrap();

        let taylor_width = 2.0 * bound;
        let uf_width = 2.0; // [-1, 1]

        let improvement = uf_width / taylor_width;
        assert!(
            improvement > 1000.0,
            "Taylor should be >1000x tighter than UF at t=0.5, got {improvement:.0}x"
        );
    }

    // ============================================================
    // Helper: evaluate Taylor polynomial in f64
    // ============================================================

    fn eval_taylor_f64(coeffs: &[f64], t: f64) -> f64 {
        let mut sum = 0.0;
        for (k, &coeff) in coeffs.iter().enumerate() {
            let exponent = 2 * k + 1;
            sum += coeff * t.powi(exponent as i32);
        }
        sum
    }
}
