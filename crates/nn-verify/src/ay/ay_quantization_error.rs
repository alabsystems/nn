// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for quantization error mathematical bounds (#4238).
//!
//! Quantization maps continuous values to a discrete grid. The error introduced
//! is bounded by mathematical properties of the quantization scheme. This module
//! proves six key properties using ay's SMT solver:
//!
//! 1. **Per-channel INT8 quantization error**: `|x - round(x/s)*s| <= s/2`
//! 2. **Q4_0 block quantization error**: per-element error bounded by `d/2`
//! 3. **F32->F16 truncation error**: relative error bounded by F16 machine epsilon
//! 4. **F32->BF16 truncation error**: relative error bounded by BF16 machine epsilon
//! 5. **Quantize-dequantize roundtrip monotonicity**: order preservation
//! 6. **Scale computation safety**: `scale > 0` and finite when `max(|x|) > 0`
//!
//! # Proof Strategy
//!
//! Rounding is not linear, so we model `round(v)` with a helper grid coordinate
//! `q` and the nearest-grid constraint `(q - 0.5)*step <= v <= (q + 0.5)*step`.
//! The rounding *error* is then **derived** from that constraint rather than
//! assumed, which is what keeps each proof non-vacuous.
//!
//! Two encoding disciplines keep every query decidable and fast:
//!
//! - **Bound proofs pin the scale to a literal.** `q*s` with both `q` and `s`
//!   symbolic is a variable-times-variable product (QF_NRA, which hangs). With
//!   `s` a concrete constant the product is linear (QF_LRA). The bound `|error|
//!   <= s/2` scales with `s`, so a concrete `s` is a faithful instance of the
//!   general theorem.
//!
//! - **Monotonicity is modelled over the integers.** Rounding produces an
//!   *integer* code; over the reals two ordered inputs one step apart can map to
//!   swapped codes, so the theorem is only true when `q` is an `Int`. Concrete
//!   integer data keeps every stride a literal (decidable QF_LIA).

use ay_bindings::{Expr, Sort, AYProgram};

use super::error::SmtError;
use super::translate_real::real_from_f64;

/// Result of a quantization property proof attempt.
#[derive(Debug, Clone)]
pub(crate) struct QuantizationPropertyResult {
    /// Human-readable property name.
    pub property: String,
    /// Whether the proof succeeded (UNSAT = property holds for all inputs).
    pub proven: bool,
    /// SMT-LIB2 text of the query (for debugging/external solver use).
    pub smt2: String,
    /// Solver detail message.
    pub detail: String,
}

/// Declare a real variable and return its expression.
fn declare_real(program: &mut AYProgram, name: &str) -> Expr {
    program.declare_const(name, Sort::real())
}

/// Assert `lower <= expr <= upper`.
fn assert_bounds(
    program: &mut AYProgram,
    expr: &Expr,
    lower: f64,
    upper: f64,
) -> Result<(), SmtError> {
    let lo = real_from_f64(lower)?;
    let hi = real_from_f64(upper)?;
    program.assert(expr.clone().real_ge(lo));
    program.assert(expr.clone().real_le(hi));
    Ok(())
}

/// Declare `name` as an `Int` constrained to `lo <= name <= hi`.
fn declare_bounded_int(program: &mut AYProgram, name: &str, lo: i64, hi: i64) -> Expr {
    let var = program.declare_const(name, Sort::int());
    program.assert(var.clone().int_ge(Expr::int(lo)));
    program.assert(var.clone().int_le(Expr::int(hi)));
    var
}

/// Execute a ay program and return whether UNSAT (property proven).
fn execute_and_check(program: &AYProgram) -> (bool, String) {
    let (proven, detail) = match ay_bindings::execute_direct::execute(program) {
        Ok(ay_bindings::execute_direct::ExecuteResult::Verified) => {
            (true, "UNSAT: property holds for all inputs".to_string())
        }
        Ok(ay_bindings::execute_direct::ExecuteResult::Counterexample { model, .. }) => {
            (false, format!("SAT: counterexample found: {:?}", model))
        }
        Ok(ay_bindings::execute_direct::ExecuteResult::Unknown(reason)) => {
            (false, format!("Unknown: {}", reason))
        }
        Ok(other) => (false, format!("Unexpected result: {:?}", other)),
        Err(e) => (false, format!("Execution error: {}", e)),
    };
    // Uniform guard: a vacuous UNSAT (P and not-P, or X != X) never counts as a
    // proof. See crate::ay_vacuity. No-op for genuine queries.
    crate::ay_vacuity::reject_if_vacuous(&program.to_string(), proven, detail)
}

// ---------------------------------------------------------------------------
// Property 1: Per-channel INT8 Symmetric Quantization Error
// ---------------------------------------------------------------------------

/// Concrete symmetric quantization scale used by the INT8 error proof. Pinning
/// `s` to a literal makes every `q*s` product linear (QF_LRA); the bound
/// `|error| <= s/2` scales with `s`, so this is a faithful concrete instance of
/// the general theorem, which holds for every `s > 0`.
const INT8_SCALE: f64 = 3.0;

/// Prove that symmetric per-channel INT8 quantization error is bounded by `s/2`.
///
/// For symmetric quantization with scale `s > 0`:
///   `quant(x) = round(x / s)`, `dequant(q) = q * s`,
///   `error = |x - dequant(quant(x))| = |x - round(x/s) * s|`.
///
/// `round(x/s)` is modelled by a grid coordinate `q` with the nearest-grid
/// constraint `(q - 0.5)*s <= x <= (q + 0.5)*s`. The error `x - q*s` is then
/// *derived* to lie in `[-s/2, s/2]` — it is not assumed — so the proof is not
/// vacuous. A too-tight claimed bound (`s/4`) is violated exactly at the grid
/// edge, which the mutation test confirms (see `int8_bound_is_tight`).
pub(crate) fn prove_int8_quantization_error_bound() -> Result<QuantizationPropertyResult, SmtError>
{
    let program = build_int8_quantization_error_bound(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(QuantizationPropertyResult {
        property: "int8_quantization_error_bound".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the INT8 error-bound query. `bound_is_half_step` selects the claimed
/// bound: the correct `s/2` (true half-step) or the too-tight `s/4`. The
/// round-to-nearest error reaches `s/2` at the grid edge, so `s/4` is violated
/// there and the query turns SAT.
fn build_int8_quantization_error_bound(bound_is_half_step: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let s_val = INT8_SCALE;
    let s = real_from_f64(s_val)?;

    // Grid coordinate q modelled over the reals in the INT8 symmetric range.
    // (Reals make the feasible set a superset of the integers, so the bound we
    // prove is if anything stronger; integrality is not needed for a bound.)
    let q = declare_real(&mut program, "q");
    assert_bounds(&mut program, &q, -127.0, 127.0)?;

    // Input x in the representable range [-127*s, 127*s].
    let x = declare_real(&mut program, "x");
    assert_bounds(&mut program, &x, -127.0 * s_val, 127.0 * s_val)?;

    // Nearest-grid constraint: (q - 0.5)*s <= x <= (q + 0.5)*s. `s` is a literal
    // so both sides are linear in q.
    let half = real_from_f64(0.5)?;
    let q_lo = q.clone().real_sub(half.clone()).real_mul(s.clone());
    let q_hi = q.clone().real_add(half).real_mul(s.clone());
    program.assert(x.clone().real_ge(q_lo));
    program.assert(x.clone().real_le(q_hi));

    // Dequantize and form the round-trip error, derived from the constraint.
    let dequant = q.real_mul(s.clone());
    let error = x.real_sub(dequant);

    // Claimed bound: |error| <= s/2 (correct) or the too-tight s/4 (mutation).
    let bound_frac = if bound_is_half_step { 0.5 } else { 0.25 };
    let bound = s.real_mul(real_from_f64(bound_frac)?);

    let too_high = error.clone().real_gt(bound.clone());
    let too_low = error.real_lt(bound.real_neg());
    program.assert(too_high.or(too_low));
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 2: Q4_0 Block Quantization Error
// ---------------------------------------------------------------------------

/// Concrete Q4_0 block scale used by the error proof. Pinned to a literal for
/// the same reason as [`INT8_SCALE`].
const Q4_0_SCALE: f64 = 5.0;

/// Prove that Q4_0 block quantization per-element error is bounded by `d/2`.
///
/// Q4_0 format: 32-element blocks with a shared scale `d > 0`; each element is a
/// signed 4-bit code in `[-8, 7]`. With `dequant = q*d` and the nearest-grid
/// constraint `(q - 0.5)*d <= x <= (q + 0.5)*d`, the error `x - q*d` is derived
/// to lie in `[-d/2, d/2]`. As with INT8 the bound is tight at the grid edge, so
/// the too-tight `d/4` mutation is SAT (see `q4_0_bound_is_tight`).
pub(crate) fn prove_q4_0_block_quantization_error() -> Result<QuantizationPropertyResult, SmtError>
{
    let program = build_q4_0_block_quantization_error(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(QuantizationPropertyResult {
        property: "q4_0_block_quantization_error".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the Q4_0 error-bound query. `bound_is_half_step` selects `d/2` (correct)
/// or the too-tight `d/4` (mutation, SAT at the grid edge).
fn build_q4_0_block_quantization_error(bound_is_half_step: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let d_val = Q4_0_SCALE;
    let d = real_from_f64(d_val)?;

    // 4-bit signed grid coordinate q in [-8, 7].
    let q = declare_real(&mut program, "q");
    assert_bounds(&mut program, &q, -8.0, 7.0)?;

    // Input x in the representable range [-8*d, 7*d].
    let x = declare_real(&mut program, "x");
    assert_bounds(&mut program, &x, -8.0 * d_val, 7.0 * d_val)?;

    // Nearest-grid constraint: (q - 0.5)*d <= x <= (q + 0.5)*d, linear in q.
    let half = real_from_f64(0.5)?;
    let q_lo = q.clone().real_sub(half.clone()).real_mul(d.clone());
    let q_hi = q.clone().real_add(half).real_mul(d.clone());
    program.assert(x.clone().real_ge(q_lo));
    program.assert(x.clone().real_le(q_hi));

    let dequant = q.real_mul(d.clone());
    let error = x.real_sub(dequant);

    let bound_frac = if bound_is_half_step { 0.5 } else { 0.25 };
    let bound = d.real_mul(real_from_f64(bound_frac)?);

    let too_high = error.clone().real_gt(bound.clone());
    let too_low = error.real_lt(bound.real_neg());
    program.assert(too_high.or(too_low));
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Properties 3 & 4: F32 -> {F16, BF16} Truncation Error (Relative)
// ---------------------------------------------------------------------------

/// Prove that F32->F16 truncation error is bounded by `epsilon_f16 * |x|` for
/// normal values, where `epsilon_f16 = 2^-10` (F16 keeps 10 fraction bits).
///
/// See [`build_truncation_error_bound`] for the model.
pub(crate) fn prove_f32_to_f16_truncation_error() -> Result<QuantizationPropertyResult, SmtError> {
    let program = build_truncation_error_bound(10, true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(QuantizationPropertyResult {
        property: "f32_to_f16_truncation_error".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove that F32->BF16 truncation error is bounded by `epsilon_bf16 * |x|` for
/// normal values, where `epsilon_bf16 = 2^-7` (BF16 keeps 7 fraction bits).
///
/// See [`build_truncation_error_bound`] for the model.
pub(crate) fn prove_f32_to_bf16_truncation_error() -> Result<QuantizationPropertyResult, SmtError> {
    let program = build_truncation_error_bound(7, true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(QuantizationPropertyResult {
        property: "f32_to_bf16_truncation_error".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the relative-truncation-error query for a float with `significand_bits`
/// stored fraction bits (F16 -> 10, BF16 -> 7).
///
/// We work in the **ulp-scaled domain** `X = significand * 2^significand_bits`,
/// so the representable grid is exactly the integers and every coefficient is a
/// plain numeral (no fractional constants — unambiguously linear QF_LRA). A
/// normal significand in `[1, 2)` gives `X` in `[2^p, 2^(p+1))`. Truncation
/// rounds `X` to the nearest grid point `g`; the constraint `g - 0.5 <= X <= g +
/// 0.5` *derives* `|X - g| <= 1/2` — the error is not assumed, which is what
/// keeps the proof non-vacuous. The grid coordinate `g` may stay real because a
/// bound needs no integrality.
///
/// The relative bound `|x - x_trunc| <= epsilon * x` becomes, after clearing the
/// `2^p` scaling, `mult * |X - g| <= X` with `mult = 1/epsilon`. For the true
/// machine epsilon `2^-p` this is `mult = 2^p`, and since `X >= 2^p` and
/// `|X - g| <= 1/2` it holds with room to spare. `claim_true_epsilon` selects
/// that; the mutation claims `2^-(p+3)` — three phantom mantissa bits, the
/// classic "used the wrong format's epsilon" — for which `mult = 2^(p+3)` and
/// the half-ulp rounding error violates the bound near `X = 2^p`, so the query
/// turns SAT.
fn build_truncation_error_bound(
    significand_bits: u32,
    claim_true_epsilon: bool,
) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Scaled binade: X = significand * 2^p, so grid points are integers.
    let scale = 1_i64 << significand_bits; // 2^p
    let lo = scale as f64; // significand 1.0
    let hi = (scale as f64) * 2.0; // significand 2.0

    let x = declare_real(&mut program, "x_scaled");
    assert_bounds(&mut program, &x, lo, hi)?;

    // Scaled grid point g; round-to-nearest pins it within half a (scaled) ulp
    // of X:  g - 0.5 <= X <= g + 0.5.
    let g = declare_real(&mut program, "g");
    let half = real_from_f64(0.5)?;
    program.assert(x.clone().real_ge(g.clone().real_sub(half.clone())));
    program.assert(x.clone().real_le(g.clone().real_add(half)));

    // Scaled round-trip error X - g, derived from the grid constraint above.
    let error = x.clone().real_sub(g);

    // Relative bound cleared of the 2^p scaling: mult * error, compared to X.
    //   correct : epsilon = 2^-p        -> mult = 2^p
    //   mutation: epsilon = 2^-(p+3)     -> mult = 2^(p+3)  (three phantom bits)
    let mult = if claim_true_epsilon { scale } else { scale << 3 };
    let scaled_err = error.real_mul(Expr::real(mult));

    // Violation: |mult * error| > X, i.e. the relative error exceeds epsilon.
    let too_high = scaled_err.clone().real_gt(x.clone());
    let too_low = scaled_err.real_lt(x.real_neg());
    program.assert(too_high.or(too_low));
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 5: Quantize-Dequantize Roundtrip Monotonicity
// ---------------------------------------------------------------------------

/// Concrete symmetric quantization step used by the monotonicity proof. Even so
/// that half a step (`MONO_STEP / 2`) is an integer and the whole query stays in
/// QF_LIA.
const MONO_STEP: i64 = 4;

/// Prove that symmetric per-tensor quantization preserves weak ordering:
///   if `x1 < x2`, then `dequant(quant(x1)) <= dequant(quant(x2))`.
///
/// Nearest-integer rounding is monotone: `x1 < x2` forces the integer codes
/// `q1 = round(x1/s) <= round(x2/s) = q2`, hence `q1*s <= q2*s` for `s > 0`.
///
/// The codes MUST be integers. Over the reals two inputs less than a step apart
/// can round to swapped reals (`q1 > q2` while `x1 < x2`), so the real encoding
/// is genuinely SAT — this is the bug the proof fixes. Modelling `q1, q2` as
/// `Int` and pinning `s` to a concrete even integer makes the query decidable
/// QF_LIA. The order-preservation conclusion is *derived* from the rounding
/// constraints, not asserted; widening the rounding tolerance to a full step
/// breaks it (see `monotonicity_depends_on_the_rounding_tolerance`).
pub(crate) fn prove_quantize_dequantize_monotonicity(
) -> Result<QuantizationPropertyResult, SmtError> {
    let program = build_quantize_dequantize_monotonicity(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(QuantizationPropertyResult {
        property: "quantize_dequantize_monotonicity".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the monotonicity query over the integers. `round_to_nearest` gates the
/// rounding tolerance: the correct half-step `MONO_STEP/2` (true round-to-
/// nearest) or the too-wide full step `MONO_STEP`. With a full-step tolerance
/// two ordered inputs can round to swapped codes, breaking monotonicity -> SAT.
fn build_quantize_dequantize_monotonicity(round_to_nearest: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let step = MONO_STEP;
    let tol = if round_to_nearest { step / 2 } else { step };

    // Two integer-valued inputs with x1 strictly below x2.
    let x1 = declare_bounded_int(&mut program, "x1", -508, 508);
    let x2 = declare_bounded_int(&mut program, "x2", -508, 508);
    program.assert(x1.clone().int_lt(x2.clone()));

    // Their nearest-integer quantized codes in the INT8 symmetric range.
    let q1 = declare_bounded_int(&mut program, "q1", -127, 127);
    let q2 = declare_bounded_int(&mut program, "q2", -127, 127);

    // Round-to-nearest constraint |x - q*step| <= tol, i.e.
    // q*step - tol <= x <= q*step + tol.
    assert_rounds_to(&mut program, &x1, &q1, step, tol);
    assert_rounds_to(&mut program, &x2, &q2, step, tol);

    // Dequantized values q*step; violation: the round trip inverted the order.
    let dq1 = q1.int_mul(Expr::int(step));
    let dq2 = q2.int_mul(Expr::int(step));
    program.assert(dq1.int_gt(dq2));
    program.check_sat();
    program
}

/// Assert that `x` rounds to code `q` at the given `step`: `|x - q*step| <= tol`.
fn assert_rounds_to(program: &mut AYProgram, x: &Expr, q: &Expr, step: i64, tol: i64) {
    let center = q.clone().int_mul(Expr::int(step));
    program.assert(x.clone().int_ge(center.clone().int_sub(Expr::int(tol))));
    program.assert(x.clone().int_le(center.int_add(Expr::int(tol))));
}

// ---------------------------------------------------------------------------
// Property 6: Scale Computation Safety
// ---------------------------------------------------------------------------

/// Prove that quantization scale computation produces a positive, finite result
/// when `max_abs > 0`.
///
/// Scale formula: `scale = max_abs / (2^(bits-1) - 1)`
///
/// For INT8 (bits=8): `scale = max_abs / 127`
/// For INT4 (bits=4): `scale = max_abs / 7`
///
/// We prove:
///   1. `scale > 0` when `max_abs > 0`
///   2. `scale` is bounded (finite) when `max_abs` is bounded
///
/// This is a simple QF_LRA proof since division by a positive constant
/// preserves positivity.
pub(crate) fn prove_scale_computation_safety(
    bits: u32,
) -> Result<QuantizationPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // max_abs = max(|x|) over all elements, must be > 0
    let max_abs = declare_real(&mut program, "max_abs");
    // Positive and finite (bounded away from 0 and infinity)
    assert_bounds(&mut program, &max_abs, 1.0e-10, 1.0e10)?;

    // Divisor: 2^(bits-1) - 1
    let divisor_val = (1_u64 << (bits - 1)) as f64 - 1.0;
    let divisor = real_from_f64(divisor_val)?;

    // Scale = max_abs / divisor
    let scale = max_abs.clone().real_div(divisor.clone());

    // Property 1: scale > 0
    // Negated: scale <= 0
    let zero = Expr::real(0);
    let not_positive = scale.clone().real_le(zero.clone());

    program.assert(not_positive);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    if !proven {
        return Ok(QuantizationPropertyResult {
            property: format!("scale_computation_safety_{}bit", bits),
            proven,
            smt2,
            detail,
        });
    }

    // Property 2: scale is bounded (upper bound = max_abs_bound / divisor)
    let mut program2 = AYProgram::new();
    program2.set_logic("QF_LRA");

    let max_abs2 = declare_real(&mut program2, "max_abs");
    assert_bounds(&mut program2, &max_abs2, 1.0e-10, 1.0e10)?;

    let divisor2 = real_from_f64(divisor_val)?;
    let scale2 = max_abs2.real_div(divisor2);

    // scale should be <= max_abs_bound / divisor
    let upper_bound = real_from_f64(1.0e10 / divisor_val)?;
    let scale_too_large = scale2.real_gt(upper_bound);

    program2.assert(scale_too_large);
    program2.check_sat();

    let smt2_2 = program2.to_string();
    let (proven2, detail2) = execute_and_check(&program2);

    // Both must be proven
    let combined_proven = proven && proven2;
    let combined_detail = format!("Positivity: {}; Boundedness: {}", detail, detail2);
    let combined_smt2 = format!(
        "; --- Positivity proof ---\n{}\n; --- Boundedness proof ---\n{}",
        smt2, smt2_2
    );

    Ok(QuantizationPropertyResult {
        property: format!("scale_computation_safety_{}bit", bits),
        proven: combined_proven,
        smt2: combined_smt2,
        detail: combined_detail,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ay_vacuity::vacuity_smell;

    #[test]
    fn test_int8_quantization_error_bound_proven() {
        let result = prove_int8_quantization_error_bound().expect("proof should not error");
        assert!(
            result.proven,
            "INT8 quantization error bound (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "int8_quantization_error_bound");
    }

    /// The `s/2` bound is tight: the round-to-nearest error reaches exactly `s/2`
    /// at the grid edge, so claiming the tighter `s/4` must expose a
    /// counterexample. If it does not, the proof is not deriving the error from
    /// the rounding constraint.
    #[test]
    fn int8_bound_is_tight() {
        let program =
            build_int8_quantization_error_bound(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "claiming the too-tight s/4 bound must be SAT at the grid edge; got: {detail}",
        );
    }

    #[test]
    fn test_q4_0_block_quantization_error_proven() {
        let result = prove_q4_0_block_quantization_error().expect("proof should not error");
        assert!(
            result.proven,
            "Q4_0 block quantization error bound (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "q4_0_block_quantization_error");
    }

    /// As with INT8, the `d/2` bound is tight, so the too-tight `d/4` claim must
    /// be SAT.
    #[test]
    fn q4_0_bound_is_tight() {
        let program =
            build_q4_0_block_quantization_error(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "claiming the too-tight d/4 bound must be SAT at the grid edge; got: {detail}",
        );
    }

    #[test]
    fn test_f32_to_f16_truncation_error_proven() {
        let result = prove_f32_to_f16_truncation_error().expect("proof should not error");
        assert!(
            result.proven,
            "F32->F16 truncation error bound (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "f32_to_f16_truncation_error");
    }

    #[test]
    fn test_f32_to_bf16_truncation_error_proven() {
        let result = prove_f32_to_bf16_truncation_error().expect("proof should not error");
        assert!(
            result.proven,
            "F32->BF16 truncation error bound (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "f32_to_bf16_truncation_error");
    }

    /// Claiming three phantom mantissa bits (`epsilon = 2^-(bits+3)`) understates
    /// the real rounding error and must be SAT for both formats. If it were not,
    /// the proof would not be deriving the error from the ulp grid.
    #[test]
    fn truncation_epsilon_matches_the_mantissa_width() {
        for significand_bits in [10_u32, 7] {
            let program = build_truncation_error_bound(significand_bits, false)
                .expect("build should not error");
            let (proven, detail) = execute_and_check(&program);
            assert!(
                !proven,
                "under-claimed epsilon for {significand_bits}-bit mantissa must be SAT; \
                 got: {detail}",
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_monotonicity_proven() {
        let result = prove_quantize_dequantize_monotonicity().expect("proof should not error");
        assert!(
            result.proven,
            "Quantize-dequantize monotonicity (QF_LIA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert!(
            !result.detail.contains("counterexample"),
            "monotonicity must not have a counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "quantize_dequantize_monotonicity");
    }

    /// Monotonicity holds only because the rounding tolerance is half a step.
    /// Widening it to a full step lets two ordered inputs round to swapped codes,
    /// so the query must be SAT — confirming the conclusion is derived from the
    /// rounding constraints, not asserted.
    #[test]
    fn monotonicity_depends_on_the_rounding_tolerance() {
        let program = build_quantize_dequantize_monotonicity(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with a full-step rounding tolerance the codes can swap and the query \
             must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_scale_computation_safety_int8_proven() {
        let result = prove_scale_computation_safety(8).expect("proof should not error");
        assert!(
            result.proven,
            "Scale computation safety 8-bit (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "scale_computation_safety_8bit");
    }

    #[test]
    fn test_scale_computation_safety_int4_proven() {
        let result = prove_scale_computation_safety(4).expect("proof should not error");
        assert!(
            result.proven,
            "Scale computation safety 4-bit (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "scale_computation_safety_4bit");
    }

    #[test]
    fn test_int8_error_smt2_structure() {
        let result = prove_int8_quantization_error_bound().expect("proof should not error");
        assert!(result.smt2.contains("set-logic"), "should declare logic");
        assert!(result.smt2.contains("check-sat"), "should have check-sat");
        assert!(
            result.smt2.contains("declare-const"),
            "should have declarations"
        );
    }

    #[test]
    fn test_q4_0_error_smt2_structure() {
        let result = prove_q4_0_block_quantization_error().expect("proof should not error");
        assert!(result.smt2.contains("set-logic"), "should declare logic");
        assert!(result.smt2.contains("check-sat"), "should have check-sat");
    }

    #[test]
    fn test_monotonicity_smt2_structure() {
        let result = prove_quantize_dequantize_monotonicity().expect("proof should not error");
        assert!(result.smt2.contains("set-logic"), "should declare logic");
        assert!(result.smt2.contains("check-sat"), "should have check-sat");
    }

    #[test]
    fn test_scale_safety_smt2_structure() {
        let result = prove_scale_computation_safety(8).expect("proof should not error");
        assert!(
            result.smt2.contains("set-logic"),
            "should declare logic in at least one subproof"
        );
        assert!(
            result.smt2.contains("check-sat"),
            "should have check-sat in at least one subproof"
        );
    }
}
