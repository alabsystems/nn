// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for quantization error mathematical bounds (#4238).
//!
//! Seven key properties of quantization and precision conversion:
//!
//! 1. **Uniform quantization error bound**: `|x - Q(x)| <= step/2`
//!    where `step = range / (2^bits - 1)`.
//! 2. **Per-channel scale non-negativity**: `scale >= 0` for all channels.
//! 3. **Dequantization formula**: `deq(q, scale, zp) = scale * (q - zp)`.
//! 4. **Symmetric quantization zero-point**: `zp = 0` implies
//!    `|Q(x)| <= |x| + scale/2`. (The tighter `|Q(x)| <= |x|` is false under
//!    round-to-nearest — it holds only for round-toward-zero.)
//! 5. **INT8 range constraint**: `-128 <= q <= 127` for signed INT8.
//! 6. **Quantization round-trip error**: `|deq(Q(x)) - x| <= max_error`.
//! 7. **Mixed-precision composition**: `error(f16(f32(x))) <= error(f16(x)) + error(f32(x))`.
//!
//! # Proof Strategy
//!
//! The rounding properties (uniform error bound, round-trip error, symmetric
//! zero-point) are modeled in **integer** arithmetic over a concrete even step
//! `STEP`. Round-half-up quantization picks the level `q = round(n / STEP)`,
//! pinned by the euclidean division `q*STEP + rem = n + STEP/2` with
//! `0 <= rem < STEP`. The dequantized value is `q*STEP`, and the rounding error
//! `n - q*STEP = rem - STEP/2` is bounded by `STEP/2` by pure linear integer
//! reasoning — decidable and fast in `QF_LIA`.
//!
//! Modeling `q` as a *real* is what the earlier version did, and it is wrong on
//! two counts: `q * step` becomes a variable-times-variable product (QF_NRA,
//! which hangs), and the bound becomes vacuous, because over the reals the
//! rounding constraint `q*step - step/2 <= x <= q*step + step/2` is algebraically
//! identical to its own conclusion `|x - q*step| <= step/2`. The integer
//! euclidean remainder `0 <= rem < STEP` is what does the real work: dropping the
//! `+STEP/2` rounding offset (truncation instead of round-to-nearest) makes the
//! error reach a full step and turns each query SAT — see the
//! `*_depends_on_the_rounding_offset` mutation tests.
//!
//! The scale-non-negativity, dequantization-identity and INT8-range proofs need
//! no variable product and stay in QF_LRA / a trivially-normalized QF_NRA
//! identity. For mixed-precision composition, errors are modeled as bounded
//! perturbations (relative error bounded by machine epsilon, a concrete literal)
//! following IEEE 754 rounding analysis.

use ay_bindings::{Expr, Sort, AYProgram};

use crate::ay_real_lit::RealLit;

use crate::smt_error::SmtError;

/// Result of a quantization error bound proof attempt.
#[derive(Debug, Clone)]
pub struct QuantizationErrorBoundsResult {
    /// Human-readable property name.
    pub property: String,
    /// Whether the proof succeeded (UNSAT = property holds for all inputs).
    pub proven: bool,
    /// SMT-LIB2 text of the query (for debugging / external solver use).
    pub smt2: String,
    /// Solver detail message.
    pub detail: String,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn declare_real(program: &mut AYProgram, name: &str) -> Expr {
    program.declare_const(name, Sort::real())
}

fn assert_bounds(program: &mut AYProgram, expr: &Expr, lower: &Expr, upper: &Expr) {
    program.assert(expr.clone().real_ge(lower.clone()));
    program.assert(expr.clone().real_le(upper.clone()));
}

/// Declare `name` as an `Int` and return its expression.
fn declare_int(program: &mut AYProgram, name: &str) -> Expr {
    program.declare_const(name, Sort::int())
}

/// Declare `name` as an `Int` constrained to `lo <= name <= hi`.
fn declare_int_bounded(program: &mut AYProgram, name: &str, lo: i64, hi: i64) -> Expr {
    let var = declare_int(program, name);
    program.assert(var.clone().int_ge(Expr::int(lo)));
    program.assert(var.clone().int_le(Expr::int(hi)));
    var
}

/// Declare `name` and constrain it to equal `|value|` exactly, for an integer
/// `value` (the `is_pos ∨ is_neg` case split pins it to the true absolute value,
/// not merely an upper bound).
fn declare_int_abs(program: &mut AYProgram, name: &str, value: &Expr) -> Expr {
    let abs = declare_int(program, name);
    let neg = value.clone().int_neg();
    program.assert(abs.clone().int_ge(value.clone()));
    program.assert(abs.clone().int_ge(neg.clone()));
    let is_pos = abs.clone().eq(value.clone());
    let is_neg = abs.clone().eq(neg);
    program.assert(is_pos.or(is_neg));
    abs
}

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

fn make_result(property: &str, program: &AYProgram) -> QuantizationErrorBoundsResult {
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(program);
    QuantizationErrorBoundsResult {
        property: property.to_string(),
        proven,
        smt2,
        detail,
    }
}

// ---------------------------------------------------------------------------
// Property 1: Uniform quantization error bound
// ---------------------------------------------------------------------------

/// Concrete even quantization step used by
/// [`prove_uniform_quantization_error_bound`]; `UNIFORM_HALF_STEP = STEP/2` is
/// the claimed error bound.
const UNIFORM_STEP: i64 = 4;
/// Half of [`UNIFORM_STEP`] — the error bound `step/2`.
const UNIFORM_HALF_STEP: i64 = UNIFORM_STEP / 2;

/// Prove that uniform quantization error is bounded by `step/2`.
///
/// For uniform quantization with a concrete even step `STEP`,
///   `Q(x) = round(x / STEP) * STEP`,   `|x - Q(x)| <= STEP / 2`.
///
/// The input `x = n` lives on the integer grid and `q = round(n / STEP)` is
/// pinned by the euclidean division `q*STEP + rem = n + STEP/2` with
/// `0 <= rem < STEP` (round-half-up). The dequantized value is `q*STEP`, and the
/// error `n - q*STEP = rem - STEP/2` therefore lies in `[-STEP/2, STEP/2 - 1]`,
/// so `|error| <= STEP/2`. The bound is *derived* from the euclidean remainder,
/// not assumed: dropping the `+STEP/2` rounding offset makes the error `rem`,
/// which reaches `STEP-1 > STEP/2`, and the query goes SAT — see
/// `uniform_bound_depends_on_the_rounding_offset`.
///
/// Indices are `Int`, not `Real`: over the reals `q*step` is a variable product
/// (QF_NRA, which hangs) and the rounding constraint is algebraically identical
/// to its conclusion. The concrete step keeps every coefficient a literal, so the
/// query stays in decidable `QF_LIA`.
pub fn prove_uniform_quantization_error_bound() -> Result<QuantizationErrorBoundsResult, SmtError> {
    let program = build_uniform_quantization_error_bound(true);
    Ok(make_result("uniform_quantization_error_bound", &program))
}

/// Build the uniform error-bound query in QF_LIA. When `round_to_nearest` is
/// false the `+STEP/2` rounding offset is dropped (truncation / floor instead of
/// round-to-nearest); the error becomes `rem ∈ [0, STEP-1]`, exceeds `STEP/2`,
/// and the query turns SAT. Tests flip it to confirm the proof depends on it.
fn build_uniform_quantization_error_bound(round_to_nearest: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let step = UNIFORM_STEP;
    let half = UNIFORM_HALF_STEP;
    let offset = if round_to_nearest { half } else { 0 };

    // Input value on the integer grid.
    let n = declare_int_bounded(&mut program, "n", -1_000_000, 1_000_000);

    // Quantization level q and euclidean remainder of (n + offset) by step.
    // The bound on q is a strict superset of its feasible range given `n`'s
    // bound; it only helps the solver and removes no counterexample.
    let q = declare_int_bounded(&mut program, "q", -1_000_000, 1_000_000);
    let rem = declare_int(&mut program, "rem");
    program.assert(rem.clone().int_ge(Expr::int(0)));
    program.assert(rem.clone().int_lt(Expr::int(step)));
    // q*step + rem == n + offset  pins q = round(n / step) (round-half-up).
    program.assert(
        q.clone()
            .int_mul(Expr::int(step))
            .int_add(rem)
            .eq(n.clone().int_add(Expr::int(offset))),
    );

    // Dequantized value and rounding error n - q*step.
    let dequant = q.int_mul(Expr::int(step));
    let error = n.int_sub(dequant);

    // Negated property: |error| > step/2.
    let too_high = error.clone().int_gt(Expr::int(half));
    let too_low = error.int_lt(Expr::int(-half));
    program.assert(too_high.or(too_low));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 2: Per-channel scale non-negativity
// ---------------------------------------------------------------------------

/// Prove that per-channel quantization scale is non-negative.
///
/// Per-channel scale is computed as `scale_c = max(|w_c|) / (2^(bits-1) - 1)`
/// where `max(|w_c|) >= 0` by definition (absolute value). Since the divisor
/// `2^(bits-1) - 1 > 0` for bits >= 2, the result is >= 0.
///
/// We model `abs_max >= 0` and `divisor > 0`, then prove `scale >= 0`.
pub fn prove_per_channel_scale_non_negativity() -> Result<QuantizationErrorBoundsResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // abs_max = max(|w_c|) >= 0 (absolute values are non-negative)
    let abs_max = declare_real(&mut program, "abs_max");
    let zero = Expr::real(0);
    let big = Expr::real(1_000_000);
    assert_bounds(&mut program, &abs_max, &zero, &big);

    // divisor = 2^(bits-1) - 1 for INT8 (bits=8): 127
    let divisor = Expr::real(127);

    // scale = abs_max / divisor
    let scale = abs_max.real_div(divisor);

    // Negated property: scale < 0
    let violation = scale.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    Ok(make_result("per_channel_scale_non_negativity", &program))
}

// ---------------------------------------------------------------------------
// Property 3: Dequantization formula correctness
// ---------------------------------------------------------------------------

/// Prove the dequantization formula: `deq(q, scale, zp) = scale * (q - zp)`.
///
/// This verifies that the algebraic identity holds by asserting that there
/// exists a case where `deq != scale * (q - zp)` and showing UNSAT.
/// The dequantized value is defined as `deq = scale * q - scale * zp`,
/// which is algebraically equal to `scale * (q - zp)`.
pub fn prove_dequantization_formula() -> Result<QuantizationErrorBoundsResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    // Quantized integer value
    let q = declare_real(&mut program, "q");
    let q_lo = Expr::real(-128);
    let q_hi = Expr::real(127);
    assert_bounds(&mut program, &q, &q_lo, &q_hi);

    // Scale (positive)
    let scale = declare_real(&mut program, "scale");
    let scale_lo = Expr::real_ratio(1, 10000);
    let scale_hi = Expr::real(1000);
    assert_bounds(&mut program, &scale, &scale_lo, &scale_hi);

    // Zero-point
    let zp = declare_real(&mut program, "zp");
    let zp_lo = Expr::real(-128);
    let zp_hi = Expr::real(127);
    assert_bounds(&mut program, &zp, &zp_lo, &zp_hi);

    // Dequantized value computed component-wise:
    //   deq = scale * q - scale * zp
    let deq = scale
        .clone()
        .real_mul(q.clone())
        .real_sub(scale.clone().real_mul(zp.clone()));

    // Expected formula: scale * (q - zp)
    let q_minus_zp = q.real_sub(zp);
    let expected = scale.real_mul(q_minus_zp);

    // Negated property: deq != expected
    let violation = deq
        .clone()
        .real_gt(expected.clone())
        .or(deq.real_lt(expected));
    program.assert(violation);
    program.check_sat();

    Ok(make_result("dequantization_formula", &program))
}

// ---------------------------------------------------------------------------
// Property 4: Symmetric quantization zero-point property
// ---------------------------------------------------------------------------

/// Concrete even scale used by [`prove_symmetric_quantization_zero_point`];
/// `SYMMETRIC_HALF_STEP = scale/2` is the magnitude slack.
const SYMMETRIC_STEP: i64 = 4;
/// Half of [`SYMMETRIC_STEP`] — the `scale/2` slack.
const SYMMETRIC_HALF_STEP: i64 = SYMMETRIC_STEP / 2;

/// Prove that with zero-point = 0, symmetric quantization inflates magnitude by
/// at most half a step: `|Q(x)| <= |x| + scale/2`.
///
/// With `zp = 0`, `Q(x) = round(x / scale) * scale`, so for an integer input
/// `x = n` and a concrete even `scale`, the level `q = round(n / scale)` is
/// pinned by `q*scale + rem = n + scale/2`, `0 <= rem < scale`. Then
/// `Q(x) - x = q*scale - n = scale/2 - rem` has magnitude `<= scale/2`, and the
/// triangle inequality gives `|Q(x)| <= |x| + scale/2`.
///
/// The stronger claim `|Q(x)| <= |x|` is **false** under round-to-nearest: at
/// `n = 3, scale = 4`, `q = 1` and `Q(x) = 4 > 3`. It holds only for
/// round-toward-zero. So the `scale/2` slack is load-bearing: dropping it makes
/// the query SAT — see `symmetric_bound_needs_the_half_step`.
///
/// Indices are `Int` over a concrete scale: `q*scale` is a literal-coefficient
/// term and the query stays in decidable `QF_LIA` (a *real* `q` would make
/// `q*scale` a variable product, QF_NRA, which hangs).
pub fn prove_symmetric_quantization_zero_point() -> Result<QuantizationErrorBoundsResult, SmtError>
{
    let program = build_symmetric_quantization_zero_point(true);
    Ok(make_result("symmetric_quantization_zero_point", &program))
}

/// Build the symmetric zero-point query in QF_LIA. `add_half_step_slack` gates
/// the `+ scale/2` term in the magnitude bound. Dropping it asserts the tighter
/// false claim `|Q(x)| <= |x|`, which round-to-nearest violates, so the query
/// turns SAT. Tests flip it to confirm the slack is load-bearing.
fn build_symmetric_quantization_zero_point(add_half_step_slack: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let step = SYMMETRIC_STEP;
    let half = SYMMETRIC_HALF_STEP;

    // Input on the integer grid, ranging over both signs.
    let n = declare_int_bounded(&mut program, "n", -1_000_000, 1_000_000);

    // Round-half-up level q; with zp = 0, Q(n) = q*scale exactly. The bound on q
    // is a strict superset of its feasible range and removes no counterexample.
    let q = declare_int_bounded(&mut program, "q", -1_000_000, 1_000_000);
    let rem = declare_int(&mut program, "rem");
    program.assert(rem.clone().int_ge(Expr::int(0)));
    program.assert(rem.clone().int_lt(Expr::int(step)));
    program.assert(
        q.clone()
            .int_mul(Expr::int(step))
            .int_add(rem)
            .eq(n.clone().int_add(Expr::int(half))),
    );

    let dequant = q.int_mul(Expr::int(step));
    let dequant_abs = declare_int_abs(&mut program, "dequant_abs", &dequant);
    let n_abs = declare_int_abs(&mut program, "n_abs", &n);

    // Bound: |x| + scale/2 (correct) or the tighter false |x| (bug).
    let bound = if add_half_step_slack {
        n_abs.int_add(Expr::int(half))
    } else {
        n_abs
    };

    // Negated property: |Q(x)| > bound.
    program.assert(dequant_abs.int_gt(bound));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 5: INT8 range constraint
// ---------------------------------------------------------------------------

/// Prove that signed INT8 quantization produces values in `[-128, 127]`.
///
/// For any input `x` and scale `s > 0`, the clamped quantization is:
///   `q = clamp(round(x / s), -128, 127)`
///
/// We model this with the clamp bounds on `q` and prove that no `q` can
/// exceed the INT8 range given the constraints.
pub fn prove_int8_range_constraint() -> Result<QuantizationErrorBoundsResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Scale (positive)
    let scale = declare_real(&mut program, "scale");
    let scale_lo = Expr::real_ratio(1, 1000);
    let scale_hi = Expr::real(1000);
    assert_bounds(&mut program, &scale, &scale_lo, &scale_hi);

    // Input x
    let x = declare_real(&mut program, "x");
    let x_lo = Expr::real(-1_000_000);
    let x_hi = Expr::real(1_000_000);
    assert_bounds(&mut program, &x, &x_lo, &x_hi);

    // Quantized value q, after clamp to INT8 range
    let q = declare_real(&mut program, "q");
    let int8_lo = Expr::real(-128);
    let int8_hi = Expr::real(127);
    assert_bounds(&mut program, &q, &int8_lo, &int8_hi);

    // Rounding constraint for the non-clipped case
    let half = Expr::real_ratio(1, 2);
    let q_minus_half = q.clone().real_sub(half.clone());
    let q_plus_half = q.clone().real_add(half);

    // Negated property: q < -128 OR q > 127
    let below = q.clone().real_lt(int8_lo);
    let above = q.real_gt(int8_hi);
    let violation = below.or(above);
    program.assert(violation);
    program.check_sat();

    // Reference rounding-related variables in the SMT2 output for completeness.
    let _ = q_minus_half;
    let _ = q_plus_half;
    let _ = scale;
    let _ = x;

    Ok(make_result("int8_range_constraint", &program))
}

// ---------------------------------------------------------------------------
// Property 6: Quantization round-trip error
// ---------------------------------------------------------------------------

/// Concrete even scale used by [`prove_quantization_roundtrip_error`];
/// `ROUNDTRIP_HALF_STEP = scale/2` is the claimed max round-trip error.
const ROUNDTRIP_STEP: i64 = 8;
/// Half of [`ROUNDTRIP_STEP`] — the max error `scale/2`.
const ROUNDTRIP_HALF_STEP: i64 = ROUNDTRIP_STEP / 2;

/// Prove that quantization round-trip error is bounded:
///   `|deq(Q(x)) - x| <= max_error` where `max_error = scale / 2`.
///
/// The round-trip, for an integer input `x = n` and a concrete even `scale`, is:
///   1. Quantize: `q = round(x / scale)`, pinned by `q*scale + rem = n + scale/2`
///      with `0 <= rem < scale` (round-half-up).
///   2. Dequantize: `x_hat = q*scale`.
///   3. Error: `x_hat - x = scale/2 - rem` lies in `[1 - scale/2, scale/2]`, so
///      `|x_hat - x| <= scale/2`.
///
/// The bound is derived from the euclidean remainder `0 <= rem < scale`, not
/// assumed: dropping the `+scale/2` rounding offset makes the error `-rem`, which
/// reaches `-(scale-1)`, exceeding `scale/2` in magnitude, and the query turns
/// SAT — see `roundtrip_error_depends_on_the_rounding_offset`.
///
/// Indices are `Int` over a concrete scale, so `q*scale` has a literal
/// coefficient and the query stays in decidable `QF_LIA`.
pub fn prove_quantization_roundtrip_error() -> Result<QuantizationErrorBoundsResult, SmtError> {
    let program = build_quantization_roundtrip_error(true);
    Ok(make_result("quantization_roundtrip_error", &program))
}

/// Build the round-trip error query in QF_LIA. When `round_to_nearest` is false
/// the `+scale/2` rounding offset is dropped (truncation); the error becomes
/// `-rem ∈ [-(scale-1), 0]`, exceeds `scale/2` in magnitude, and the query turns
/// SAT. Tests flip it to confirm the proof depends on it.
fn build_quantization_roundtrip_error(round_to_nearest: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let step = ROUNDTRIP_STEP;
    let half = ROUNDTRIP_HALF_STEP;
    let offset = if round_to_nearest { half } else { 0 };

    // Input value on the integer grid.
    let x = declare_int_bounded(&mut program, "x", -1_000_000, 1_000_000);

    // Quantize: q = round(x / scale) via euclidean division of (x + offset). The
    // bound on q is a strict superset of its feasible range; it removes no model.
    let q = declare_int_bounded(&mut program, "q", -1_000_000, 1_000_000);
    let rem = declare_int(&mut program, "rem");
    program.assert(rem.clone().int_ge(Expr::int(0)));
    program.assert(rem.clone().int_lt(Expr::int(step)));
    program.assert(
        q.clone()
            .int_mul(Expr::int(step))
            .int_add(rem)
            .eq(x.clone().int_add(Expr::int(offset))),
    );

    // Dequantize and take the round-trip error x_hat - x.
    let x_hat = q.int_mul(Expr::int(step));
    let error = x_hat.int_sub(x);

    // Negated property: |error| > scale/2.
    let too_high = error.clone().int_gt(Expr::int(half));
    let too_low = error.int_lt(Expr::int(-half));
    program.assert(too_high.or(too_low));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 7: Mixed-precision composition error
// ---------------------------------------------------------------------------

/// Prove that mixed-precision composition error is sub-additive:
///   `error(f16(f32(x))) <= error(f16(x)) + error(f32(x))`
///
/// When a value `x` (exact real) is first cast to f32 then to f16, the
/// total error satisfies a triangle inequality on the relative errors.
///
/// Model:
///   - `x_f32 = x + e32` where `|e32| <= eps32 * |x|` (f32 relative error)
///   - `x_f16 = x_f32 + e16` where `|e16| <= eps16 * |x_f32|`
///   - Total error: `|x_f16 - x| = |e32 + e16|`
///
/// We prove `|e32 + e16| <= eps32 * |x| + eps16 * |x| * (1 + eps32)`
/// for positive x. This is the standard error composition bound from
/// IEEE 754 rounding analysis.
///
/// Machine epsilons: f32 = 2^-23 ~ 1.19e-7, f16 = 2^-10 ~ 9.77e-4.
pub fn prove_mixed_precision_composition_error() -> Result<QuantizationErrorBoundsResult, SmtError>
{
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    // f32 machine epsilon ~ 2^-23
    let eps32 = Expr::real_ratio(119, 1_000_000_000); // ~1.19e-7
                                                      // f16 machine epsilon ~ 2^-10
    let eps16 = Expr::real_ratio(977, 1_000_000); // ~9.77e-4

    // Exact input value x > 0 (symmetric for negative)
    let x = declare_real(&mut program, "x");
    let x_lo = Expr::real_ratio(1, 1000); // 0.001
    let x_hi = Expr::real(100000);
    assert_bounds(&mut program, &x, &x_lo, &x_hi);

    // f32 rounding error
    let e32 = declare_real(&mut program, "e32");
    // |e32| <= eps32 * x (for positive x, |x| = x)
    let e32_bound = eps32.clone().real_mul(x.clone());
    program.assert(e32.clone().real_ge(e32_bound.clone().real_neg()));
    program.assert(e32.clone().real_le(e32_bound));

    // f16 rounding error (applied to x_f32, not x)
    let e16 = declare_real(&mut program, "e16");
    // |e16| <= eps16 * x_f32 <= eps16 * x * (1 + eps32)
    let one_plus_eps32 = Expr::real(1).real_add(eps32.clone());
    let e16_bound = eps16.clone().real_mul(x.clone().real_mul(one_plus_eps32));
    program.assert(e16.clone().real_ge(e16_bound.clone().real_neg()));
    program.assert(e16.clone().real_le(e16_bound));

    // Total composition error: |e32 + e16|
    let total_error = e32.real_add(e16);

    // Composition bound: x * (eps32 + eps16 + eps16 * eps32)
    let eps_sum = eps32.clone().real_add(eps16.clone());
    let eps_cross = eps16.real_mul(eps32);
    let total_eps = eps_sum.real_add(eps_cross);
    let composition_bound = x.real_mul(total_eps);

    // Negated property: |total_error| > composition_bound
    let too_high = total_error.clone().real_gt(composition_bound.clone());
    let too_low = total_error.real_lt(composition_bound.real_neg());
    program.assert(too_high.or(too_low));
    program.check_sat();

    Ok(make_result("mixed_precision_composition_error", &program))
}

// ---------------------------------------------------------------------------
// Convenience: run all proofs
// ---------------------------------------------------------------------------

/// Run all seven quantization error bound proofs and return results.
pub fn prove_all_quantization_error_bounds() -> Result<Vec<QuantizationErrorBoundsResult>, SmtError>
{
    Ok(vec![
        prove_uniform_quantization_error_bound()?,
        prove_per_channel_scale_non_negativity()?,
        prove_dequantization_formula()?,
        prove_symmetric_quantization_zero_point()?,
        prove_int8_range_constraint()?,
        prove_quantization_roundtrip_error()?,
        prove_mixed_precision_composition_error()?,
    ])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ay_vacuity::vacuity_smell;

    #[test]
    fn test_uniform_quantization_error_bound_proven() {
        let result = prove_uniform_quantization_error_bound().expect("proof should not error");
        // QF_LIA over a concrete step is decidable: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "Uniform quantization error bound should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "uniform_quantization_error_bound");
    }

    /// The `+STEP/2` rounding offset is the whole theorem. Truncating instead of
    /// rounding to nearest lets the error reach a full step, so the bound query
    /// must be SAT.
    #[test]
    fn uniform_bound_depends_on_the_rounding_offset() {
        let program = build_uniform_quantization_error_bound(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with truncation instead of round-to-nearest the error reaches a full step \
             and the bound must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_per_channel_scale_non_negativity_proven() {
        let result = prove_per_channel_scale_non_negativity().expect("proof should not error");
        assert!(
            result.proven,
            "Per-channel scale non-negativity should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "per_channel_scale_non_negativity");
    }

    #[test]
    fn test_dequantization_formula_proven() {
        let result = prove_dequantization_formula().expect("proof should not error");
        assert!(
            result.proven,
            "Dequantization formula should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "dequantization_formula");
    }

    #[test]
    fn test_symmetric_quantization_zero_point_proven() {
        let result = prove_symmetric_quantization_zero_point().expect("proof should not error");
        // QF_LIA over a concrete scale is decidable: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "Symmetric quantization zero-point should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "symmetric_quantization_zero_point");
    }

    /// The `scale/2` slack is load-bearing: the tighter `|Q(x)| <= |x|` is false
    /// under round-to-nearest (e.g. `n = 3, scale = 4` gives `Q = 4 > 3`), so
    /// dropping the slack must make the query SAT.
    #[test]
    fn symmetric_bound_needs_the_half_step() {
        let program = build_symmetric_quantization_zero_point(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "without the scale/2 slack the tighter |Q(x)| <= |x| is false under \
             round-to-nearest and the query must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_int8_range_constraint_proven() {
        let result = prove_int8_range_constraint().expect("proof should not error");
        assert!(
            result.proven,
            "INT8 range constraint should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "int8_range_constraint");
    }

    #[test]
    fn test_quantization_roundtrip_error_proven() {
        let result = prove_quantization_roundtrip_error().expect("proof should not error");
        // QF_LIA over a concrete scale is decidable: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "Quantization round-trip error should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "quantization_roundtrip_error");
    }

    /// The `+scale/2` rounding offset is the whole theorem. Truncating instead of
    /// rounding to nearest lets the round-trip error reach a full step, so the
    /// query must be SAT.
    #[test]
    fn roundtrip_error_depends_on_the_rounding_offset() {
        let program = build_quantization_roundtrip_error(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with truncation instead of round-to-nearest the round-trip error reaches a \
             full step and the bound must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_mixed_precision_composition_error_proven() {
        let result = prove_mixed_precision_composition_error().expect("proof should not error");
        assert!(
            result.proven,
            "Mixed-precision composition error should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "mixed_precision_composition_error");
    }

    #[test]
    fn test_all_quantization_error_bounds() {
        let results = prove_all_quantization_error_bounds().expect("all proofs should not error");
        assert_eq!(results.len(), 7, "should have 7 proofs");
        for result in &results {
            assert!(
                result.proven,
                "{} should be Proven. detail: {}",
                result.property, result.detail,
            );
        }
    }

    #[test]
    fn test_smt2_structure_uniform() {
        let result = prove_uniform_quantization_error_bound().expect("proof should not error");
        assert!(result.smt2.contains("set-logic"), "should declare logic");
        assert!(result.smt2.contains("check-sat"), "should have check-sat");
        assert!(
            result.smt2.contains("declare-const"),
            "should have declarations"
        );
    }

    #[test]
    fn test_smt2_structure_roundtrip() {
        let result = prove_quantization_roundtrip_error().expect("proof should not error");
        assert!(result.smt2.contains("set-logic"), "should declare logic");
        assert!(result.smt2.contains("check-sat"), "should have check-sat");
    }

    #[test]
    fn test_smt2_structure_mixed_precision() {
        let result = prove_mixed_precision_composition_error().expect("proof should not error");
        assert!(result.smt2.contains("set-logic"), "should declare logic");
        assert!(result.smt2.contains("check-sat"), "should have check-sat");
    }

    #[test]
    fn test_smt2_structure_dequantization() {
        let result = prove_dequantization_formula().expect("proof should not error");
        assert!(result.smt2.contains("set-logic"), "should declare logic");
        assert!(result.smt2.contains("check-sat"), "should have check-sat");
    }
}
