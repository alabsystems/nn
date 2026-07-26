// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Uninterpreted function (UF) helpers and `powi` translation for the ay SMT path.
//!
//! Split from `translate.rs` to keep both files under 500 lines.
//! Contains: `translate_powi`, `apply_bounded_uf`, `apply_positive_uf`,
//! `apply_nonneg_uf`, `declare_uf_if_needed`.

use ay_bindings::{Expr, Sort, AYProgram};

use super::error::SmtError;
use super::translate_real::real_from_f64;

/// Maximum absolute exponent for exact binary exponentiation in SMT.
///
/// Exponents with `|exp| <= MAX_EXACT_POWI_EXP` are expanded to O(log n)
/// multiplications via binary exponentiation, matching MSL codegen behavior.
/// Beyond this limit, a UF approximation is used.
const MAX_EXACT_POWI_EXP: u32 = 32;

/// Translate `base.powi(exp)` into SMT.
///
/// Uses binary exponentiation (repeated squaring) for `|exp| <= 32`,
/// producing O(log n) `real_mul` nodes. This matches the MSL codegen
/// strategy from `codegen_msl.rs::powi_stmts` and extends exact SMT
/// coverage from the previous limit of 8 to 32.
///
/// For `|exp| > 32`, falls back to a UF approximation.
pub(super) fn translate_powi(
    base: Expr,
    exp: i32,
    program: &mut AYProgram,
    real_sort: &Sort,
    declared_ufs: &mut std::collections::HashSet<String>,
    uses_uf_approx: &mut bool,
) -> Result<Expr, SmtError> {
    let abs_exp = exp.unsigned_abs();

    // Compute the positive-exponent result.
    let pos_result = match abs_exp {
        0 => return Ok(Expr::real(1)),
        1 => base,
        _ if abs_exp <= MAX_EXACT_POWI_EXP => {
            // Binary exponentiation: O(log n) multiplications.
            let mut acc = Expr::real(1);
            let mut b = base;
            let mut e = abs_exp;
            while e > 0 {
                if e & 1 == 1 {
                    acc = acc.real_mul(b.clone());
                }
                e >>= 1;
                if e > 0 {
                    b = b.clone().real_mul(b);
                }
            }
            acc
        }
        _ => {
            // Very large exponent: UF approximation with range axioms.
            *uses_uf_approx = true;
            let uf_name = format!("powi_{}_approx", exp);
            declare_uf_if_needed(&uf_name, program, real_sort, declared_ufs);
            // Domain precondition: negative exponents require base != 0 (#388).
            // x^(-n) = 1/x^n is undefined at x = 0.
            if exp < 0 {
                program.assert(base.clone().ne(Expr::real(0)));
            }
            let result = Expr::func_app_with_sort(&uf_name, vec![base], real_sort.clone());
            // Even exponents: x^(2n) >= 0 for all real x.
            // Positive even: non-negative. Negative even (x^-2 = 1/x^2): positive.
            if abs_exp.is_multiple_of(2) {
                let zero = Expr::real(0);
                if exp > 0 {
                    program.assert(result.clone().real_ge(zero));
                } else {
                    program.assert(result.clone().real_gt(zero));
                }
            }
            return Ok(result);
        }
    };

    if exp < 0 {
        // Guard: base^n can be zero when base is zero (for n > 0).
        // Assert base != 0 so (/ 1 base^n) is well-defined in SMT.
        program.assert(pos_result.clone().ne(Expr::real(0)));
        Ok(Expr::real(1).real_div(pos_result))
    } else {
        Ok(pos_result)
    }
}

/// Apply a UF with bounded range axiom: `lo <= f(arg) <= hi`.
pub(super) fn apply_bounded_uf(
    name: &str,
    arg: Expr,
    program: &mut AYProgram,
    real_sort: &Sort,
    declared_ufs: &mut std::collections::HashSet<String>,
    lo: i64,
    hi: i64,
) -> Result<Expr, SmtError> {
    declare_uf_if_needed(name, program, real_sort, declared_ufs);
    let result = Expr::func_app_with_sort(name, vec![arg], real_sort.clone());

    // Assert range axiom: lo <= result <= hi
    let lower = Expr::real(lo);
    let upper = Expr::real(hi);
    program.assert(result.clone().real_ge(lower));
    program.assert(result.clone().real_le(upper));

    Ok(result)
}

/// Apply a UF with positive range axiom: `f(arg) > 0`.
pub(super) fn apply_positive_uf(
    name: &str,
    arg: Expr,
    program: &mut AYProgram,
    real_sort: &Sort,
    declared_ufs: &mut std::collections::HashSet<String>,
) -> Result<Expr, SmtError> {
    declare_uf_if_needed(name, program, real_sort, declared_ufs);
    let result = Expr::func_app_with_sort(name, vec![arg], real_sort.clone());

    let zero = Expr::real(0);
    program.assert(result.clone().real_gt(zero));

    Ok(result)
}

/// Apply a UF with non-negative range axiom: `f(arg) >= 0`.
pub(super) fn apply_nonneg_uf(
    name: &str,
    arg: Expr,
    program: &mut AYProgram,
    real_sort: &Sort,
    declared_ufs: &mut std::collections::HashSet<String>,
) -> Result<Expr, SmtError> {
    declare_uf_if_needed(name, program, real_sort, declared_ufs);
    let result = Expr::func_app_with_sort(name, vec![arg], real_sort.clone());

    let zero = Expr::real(0);
    program.assert(result.clone().real_ge(zero));

    Ok(result)
}

/// Declare an uninterpreted function `name: Real -> Real` if not already declared.
pub(super) fn declare_uf_if_needed(
    name: &str,
    program: &mut AYProgram,
    real_sort: &Sort,
    declared_ufs: &mut std::collections::HashSet<String>,
) {
    if declared_ufs.insert(name.to_string()) {
        program.declare_fun(name, vec![real_sort.clone()], real_sort.clone());
    }
}

/// F16 maximum representable value.
const F16_MAX: f64 = 65504.0;

/// F16 worst-case ULP (unit in last place) across the full representable range.
///
/// F16 has 10 mantissa bits + 1 implicit. At the maximum exponent (bias 15,
/// stored as 30), the ULP is 2^(15-10) = 32. This bound is **sound** for all
/// finite F16 values: |round_f16(x) - x| ≤ 32 for |x| ≤ 65504.
const F16_MAX_ULP: f64 = 32.0;

/// The BF16 maximum representable value as an exact `Real` literal.
///
/// BF16 shares f32's exponent range, so its maximum is
/// `(2 - 2^-7) * 2^127 = 2^128 - 2^120 = 255 * 2^120` (~3.389e38). That
/// magnitude overflows the i64 numerator of [`real_from_f64`], so it is built
/// here directly: `2^120 = (2^60)^2` and `2^60` fits in an `i64`, making the
/// whole constant a product of three `i64` literals — no `f64` round-trip and
/// no overflow.
fn bf16_max_real() -> Expr {
    let two_pow_60: i64 = 1 << 60; // 2^60 = 1_152_921_504_606_846_976 < i64::MAX
    Expr::real(255)
        .real_mul(Expr::real(two_pow_60))
        .real_mul(Expr::real(two_pow_60))
}

/// BF16 conservative ULP bound for typical NN intermediate values.
///
/// BF16 has 7 mantissa bits + 1 implicit. The ULP at value V is approximately
/// V * 2^(-7). For the full BF16 range (up to 3.389e38), the worst-case ULP
/// is ~2^120 — too large to be useful.
///
/// This bound (512) is sound for |x| ≤ 65536, which covers typical NN
/// intermediate values (matmul outputs, attention scores, activations).
/// For models with intermediate values exceeding 65536, use Approach A
/// (exact FP bit-blasting via ay `real_to_fp`, currently blocked).
const BF16_PRACTICAL_ULP: f64 = 512.0;

/// Encode an F16 downcast as a UF with sound range and error axioms.
///
/// Models `round_to_f16(x)` as an uninterpreted function with:
/// 1. **Range axiom:** result ∈ [-65504, 65504] (F16 representable range)
/// 2. **Error axiom:** |result - input| ≤ 32 (F16 worst-case ULP)
///
/// Sound for all finite F16 values. Uses QF_UFLRA (no multiplication of
/// symbolic variables). Part of #3023 Tier 2 (ay path).
pub(super) fn encode_f16_cast(
    input: Expr,
    program: &mut AYProgram,
    real_sort: &Sort,
    declared_ufs: &mut std::collections::HashSet<String>,
) -> Expr {
    declare_uf_if_needed("f16_cast", program, real_sort, declared_ufs);
    let result = Expr::func_app_with_sort("f16_cast", vec![input.clone()], real_sort.clone());

    // Range axiom: result in F16 representable range.
    let neg_f16_max = real_from_f64(-F16_MAX).expect("F16_MAX encoding");
    let pos_f16_max = real_from_f64(F16_MAX).expect("F16_MAX encoding");
    program.assert(result.clone().real_ge(neg_f16_max));
    program.assert(result.clone().real_le(pos_f16_max));

    // Error axiom: |result - input| ≤ F16 max ULP.
    let diff = result.clone().real_sub(input);
    let neg_ulp = real_from_f64(-F16_MAX_ULP).expect("F16_ULP encoding");
    let pos_ulp = real_from_f64(F16_MAX_ULP).expect("F16_ULP encoding");
    program.assert(diff.clone().real_ge(neg_ulp));
    program.assert(diff.real_le(pos_ulp));

    result
}

/// Encode a BF16 downcast as a UF with range and error axioms.
///
/// Models `round_to_bf16(x)` as an uninterpreted function with:
/// 1. **Range axiom:** result ∈ [-3.389e38, 3.389e38] (BF16 representable range)
/// 2. **Error axiom:** |result - input| ≤ 512 (BF16 ULP, sound for |x| ≤ 65536)
///
/// **Soundness caveat:** The error bound is sound for typical NN intermediate
/// values (|x| ≤ 65536). Models with larger intermediate values require
/// Approach A (exact FP bit-blasting via ay `real_to_fp`). Part of #3023 Tier 2.
pub(super) fn encode_bf16_cast(
    input: Expr,
    program: &mut AYProgram,
    real_sort: &Sort,
    declared_ufs: &mut std::collections::HashSet<String>,
) -> Expr {
    declare_uf_if_needed("bf16_cast", program, real_sort, declared_ufs);
    let result = Expr::func_app_with_sort("bf16_cast", vec![input.clone()], real_sort.clone());

    // Range axiom: result in BF16 representable range. BF16_MAX (~3.389e38)
    // overflows real_from_f64's i64 numerator, so build it exactly as 255*2^120.
    let pos_bf16_max = bf16_max_real();
    let neg_bf16_max = pos_bf16_max.clone().real_neg();
    program.assert(result.clone().real_ge(neg_bf16_max));
    program.assert(result.clone().real_le(pos_bf16_max));

    // Error axiom: |result - input| ≤ BF16 practical ULP.
    let diff = result.clone().real_sub(input);
    let neg_bf16_ulp = real_from_f64(-BF16_PRACTICAL_ULP).expect("BF16_ULP encoding");
    let pos_bf16_ulp = real_from_f64(BF16_PRACTICAL_ULP).expect("BF16_ULP encoding");
    program.assert(diff.clone().real_ge(neg_bf16_ulp));
    program.assert(diff.real_le(pos_bf16_ulp));

    result
}

#[cfg(test)]
#[path = "translate_uf_tests.rs"]
mod tests;
