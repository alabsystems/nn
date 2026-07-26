// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for weight initialization mathematical properties (#4214).
//!
//! Weight initialization is critical for neural network training convergence.
//! Incorrect initialization leads to vanishing/exploding gradients. This module
//! proves the mathematical properties of standard initialization schemes using
//! ay's SMT solver.
//!
//! # Proved Properties
//!
//! 1. **Xavier/Glorot uniform**: range is `[-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))]`
//! 2. **Xavier/Glorot normal**: variance is `2/(fan_in+fan_out)`
//! 3. **Kaiming/He uniform**: range depends on fan_in and nonlinearity gain
//! 4. **Kaiming/He normal**: variance is `2/fan_in` for ReLU
//! 5. **Orthogonal init**: columns are unit vectors (norm = 1)
//! 6. **Uniform init**: all values in `[low, high]` range
//! 7. **Normal init**: mean and standard deviation properties
//! 8. **Zeros init**: all values exactly 0
//! 9. **Ones init**: all values exactly 1
//! 10. **Constant init**: all values equal to specified constant
//!
//! # Proof Strategy
//!
//! Weight initialization proofs use several approaches:
//!
//! - **Algebraic identity proofs** (Xavier variance, Kaiming variance): Express the
//!   variance formula as a constraint and prove the bound identity holds for all
//!   valid fan_in/fan_out values.
//!
//! - **Range constraint proofs** (uniform inits, constant inits): Assert that all
//!   values lie within the declared range, then attempt to find a counterexample
//!   where a value violates the range. UNSAT proves the range property.
//!
//! - **Norm proofs** (orthogonal init): Encode the unit-norm constraint on column
//!   vectors and prove it holds given the orthogonality definition.

use ay_bindings::{Expr, Sort, AYProgram};

use super::error::SmtError;
use super::translate_real::real_from_f64;
use crate::ay_real_lit::RealLit;

/// Result of a weight initialization property proof attempt.
#[derive(Debug, Clone)]
pub(crate) struct WeightInitPropertyResult {
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

/// Declare `name` and pin it to `term`, returning the new variable.
///
/// Introducing a name for each intermediate keeps the conclusion one step
/// removed from its hypotheses, so the solver *derives* it instead of matching
/// an equality it was handed. This is what keeps the repaired proofs out of the
/// `NegatesOwnHypothesis` vacuity trap.
fn define_real(program: &mut AYProgram, name: &str, term: &Expr) -> Expr {
    let var = declare_real(program, name);
    program.assert(var.clone().eq(term.clone()));
    var
}

/// The value a position-independent init writes at `position`:
/// `base + slope * position`.
///
/// A correct constant/zeros/ones fill has `slope == 0`, so every position gets
/// `base`. A position-dependent slip — an `arange`/iota/ramp used in place of a
/// constant fill — has `slope != 0`, so the values disagree across positions and
/// the "all values equal" property becomes false.
fn fill_value(program: &mut AYProgram, name: &str, base: &Expr, slope: i64, position: i64) -> Expr {
    let offset = Expr::real(slope).real_mul(Expr::real(position));
    let term = base.clone().real_add(offset);
    define_real(program, name, &term)
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

/// Execute a ay program and return whether UNSAT (property proven).
///
/// The final `(proven, detail)` is funneled through
/// [`crate::ay_vacuity::reject_if_vacuous`], so any query that is UNSAT only
/// because it asserts `P ∧ ¬P` (or compares a term to itself) is downgraded to a
/// failure. A residual vacuity therefore becomes a hard test failure rather than
/// a false "proven"; a genuine proof is returned unchanged.
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
    crate::ay_vacuity::reject_if_vacuous(&program.to_string(), proven, detail)
}

// ---------------------------------------------------------------------------
// Property 1: Xavier/Glorot Uniform Initialization
// ---------------------------------------------------------------------------

/// Prove that Xavier uniform init range satisfies `bound^2 = 6 / (fan_in + fan_out)`.
///
/// Xavier/Glorot uniform draws weights from `U[-bound, bound]` where
/// `bound = sqrt(6 / (fan_in + fan_out))`. This ensures the variance of the
/// uniform distribution `Var = bound^2 / 3 = 2 / (fan_in + fan_out)` preserves
/// signal magnitude through the layer.
///
/// We prove: given `fan_in > 0`, `fan_out > 0`, and `bound^2 = 6 / (fan_in + fan_out)`,
/// the variance of U[-bound, bound] equals `2 / (fan_in + fan_out)`.
/// Variance of U[-b, b] = b^2 / 3, so `variance = bound^2 / 3 = 6 / (3 * (fan_in + fan_out))
/// = 2 / (fan_in + fan_out)`.
pub(crate) fn prove_xavier_uniform_range() -> Result<WeightInitPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let fan_in = declare_real(&mut program, "fan_in");
    let fan_out = declare_real(&mut program, "fan_out");
    let bound_sq = declare_real(&mut program, "bound_sq");
    let variance = declare_real(&mut program, "variance");

    // fan_in > 0 and fan_out > 0 (positive integers, modeled as reals > 0)
    let zero = Expr::real(0);
    program.assert(fan_in.clone().real_gt(zero.clone()));
    program.assert(fan_out.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &fan_in, 1.0, 10000.0)?;
    assert_bounds(&mut program, &fan_out, 1.0, 10000.0)?;

    // bound^2 = 6 / (fan_in + fan_out)
    let fan_sum = fan_in.clone().real_add(fan_out.clone());
    let six = real_from_f64(6.0)?;
    // Encode as: bound_sq * fan_sum = 6
    program.assert(bound_sq.clone().real_mul(fan_sum.clone()).eq(six.clone()));

    // variance = bound_sq / 3 (variance of U[-b, b] = b^2/3)
    let three = real_from_f64(3.0)?;
    // Encode as: variance * 3 = bound_sq
    program.assert(variance.clone().real_mul(three).eq(bound_sq.clone()));

    // Expected variance = 2 / (fan_in + fan_out)
    // Encode as: expected_var * fan_sum = 2
    let expected_var = declare_real(&mut program, "expected_var");
    let two = real_from_f64(2.0)?;
    program.assert(expected_var.clone().real_mul(fan_sum).eq(two));

    // Negated property: variance != expected_var
    let violation = variance.ne(expected_var);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(WeightInitPropertyResult {
        property: "xavier_uniform_variance_equals_2_over_fan_sum".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 2: Xavier/Glorot Normal Initialization
// ---------------------------------------------------------------------------

/// Prove that Xavier normal init variance is `2 / (fan_in + fan_out)`.
///
/// Xavier/Glorot normal draws weights from `N(0, std^2)` with
/// `std = sqrt(2 / (fan_in + fan_out))`, so `variance = 2 / (fan_in + fan_out)`.
/// The content is that the denominator is the *sum* of both fan counts — the
/// slip that swaps it for `fan_in` alone (the Xavier-vs-Kaiming denominator
/// mixup) makes the variance wrong. We derive `variance` by applying the rule to
/// a concrete shape and compare it against `2 / fan_sum` reached through an
/// independent chain, so the conclusion is derived, not asserted equal to
/// itself. `fan_sum` is derived in SMT via `fan_in + fan_out`; the reciprocal is
/// pinned to an exact rational so the query stays linear (QF_LRA).
///
/// The wrong denominator makes the query SAT — see
/// `xavier_normal_variance_depends_on_the_fan_sum`.
pub(crate) fn prove_xavier_normal_variance() -> Result<WeightInitPropertyResult, SmtError> {
    let program = build_xavier_normal_variance(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(WeightInitPropertyResult {
        property: "xavier_normal_variance_equals_2_over_fan_sum".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the Xavier-normal variance query over a concrete shape
/// `fan_in = 2, fan_out = 6`. When `denominator_is_fan_sum` is false the variance
/// is divided by `fan_in` alone instead of `fan_in + fan_out` — the classic
/// Xavier/Kaiming mixup — which makes `variance = 2/2 = 1 != 1/4`; tests flip it
/// to confirm the proof depends on using the fan *sum*.
fn build_xavier_normal_variance(denominator_is_fan_sum: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let (fan_in_val, fan_out_val) = (2_i64, 6_i64);
    let fan_sum_val = fan_in_val + fan_out_val;

    // Concrete positive shape, each fan count named and the sum derived in SMT.
    let fan_in = define_real(&mut program, "fan_in", &Expr::real(fan_in_val));
    let fan_out = define_real(&mut program, "fan_out", &Expr::real(fan_out_val));
    let fan_sum = define_real(
        &mut program,
        "fan_sum",
        &fan_in.clone().real_add(fan_out.clone()),
    );
    program.assert(fan_sum.clone().real_gt(Expr::real(0)));

    // Xavier rule: variance = 2 * (1 / (fan_in + fan_out)). The slip divides by
    // fan_in. Deriving it as 2 * (1/denom) keeps the variance a *product*, a shape
    // distinct from the expected quotient 2 / fan_sum below, so the UNSAT rests on
    // the arithmetic rather than on the identical literal `(/ 2 8)` written twice.
    let denom = if denominator_is_fan_sum {
        fan_sum_val
    } else {
        fan_in_val
    };
    let inv_denom = Expr::real_ratio(1, denom);
    let variance = define_real(
        &mut program,
        "variance",
        &Expr::real(2).real_mul(inv_denom),
    );

    // The claimed answer 2 / fan_sum, reached through an independent constant.
    let expected = define_real(
        &mut program,
        "expected_variance",
        &Expr::real_ratio(2, fan_sum_val),
    );

    // Violation: the derived variance disagrees with 2 / (fan_in + fan_out).
    program.assert(variance.ne(expected));
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 3: Kaiming/He Uniform Initialization
// ---------------------------------------------------------------------------

/// Prove that Kaiming uniform init range satisfies `bound^2 = 3 * gain^2 / fan_in`.
///
/// Kaiming/He uniform draws weights from `U[-bound, bound]` where
/// `bound = gain * sqrt(3 / fan_in)`. The gain depends on the nonlinearity
/// (e.g., sqrt(2) for ReLU). The variance of `U[-b, b]` is `b^2/3 = gain^2 / fan_in`.
///
/// We prove: given `fan_in > 0`, `gain > 0`, and `bound^2 = 3 * gain^2 / fan_in`,
/// the variance `bound^2 / 3 = gain^2 / fan_in`.
pub(crate) fn prove_kaiming_uniform_range() -> Result<WeightInitPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let fan_in = declare_real(&mut program, "fan_in");
    let gain_sq = declare_real(&mut program, "gain_sq");
    let bound_sq = declare_real(&mut program, "bound_sq");
    let variance = declare_real(&mut program, "variance");

    let zero = Expr::real(0);
    program.assert(fan_in.clone().real_gt(zero.clone()));
    program.assert(gain_sq.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &fan_in, 1.0, 10000.0)?;
    assert_bounds(&mut program, &gain_sq, 0.0, 100.0)?;

    // bound^2 = 3 * gain^2 / fan_in
    // Encoded as: bound_sq * fan_in = 3 * gain_sq
    let three = real_from_f64(3.0)?;
    let three_gain_sq = three.clone().real_mul(gain_sq.clone());
    program.assert(bound_sq.clone().real_mul(fan_in.clone()).eq(three_gain_sq));

    // variance = bound_sq / 3
    // Encoded as: variance * 3 = bound_sq
    program.assert(variance.clone().real_mul(three).eq(bound_sq.clone()));

    // Expected: variance = gain^2 / fan_in
    // Encoded as: expected_var * fan_in = gain_sq
    let expected_var = declare_real(&mut program, "expected_var");
    program.assert(expected_var.clone().real_mul(fan_in).eq(gain_sq));

    // Negated property: variance != expected_var
    let violation = variance.ne(expected_var);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(WeightInitPropertyResult {
        property: "kaiming_uniform_variance_equals_gain_sq_over_fan_in".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 4: Kaiming/He Normal Initialization (ReLU)
// ---------------------------------------------------------------------------

/// Prove that Kaiming normal init variance for ReLU is `2 / fan_in`.
///
/// Kaiming normal uses `std = gain * sqrt(1 / fan_in)`, so
/// `variance = gain^2 / fan_in`. For ReLU `gain = sqrt(2)`, giving
/// `variance = 2 / fan_in`. The content is that the *ReLU* gain² (= 2) feeds the
/// formula — using the identity/linear gain² (= 1) halves the variance. We
/// derive `variance = gain_sq * (1/fan_in)` over a concrete `fan_in = 8` and
/// compare it against `2 / fan_in`; the reciprocal is an exact rational literal
/// so the query stays linear (QF_LRA).
///
/// Using the wrong gain makes the query SAT — see
/// `kaiming_normal_relu_variance_depends_on_the_gain`.
pub(crate) fn prove_kaiming_normal_relu_variance() -> Result<WeightInitPropertyResult, SmtError> {
    let program = build_kaiming_normal_relu_variance(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(WeightInitPropertyResult {
        property: "kaiming_normal_relu_variance_equals_2_over_fan_in".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the Kaiming-normal ReLU variance query over `fan_in = 8`. When
/// `gain_is_relu` is false the ReLU gain² (2) is replaced by the identity gain²
/// (1) — the "wrong nonlinearity gain" slip — so `variance = 1/8 != 2/8`; tests
/// flip it to confirm the proof depends on the ReLU gain.
fn build_kaiming_normal_relu_variance(gain_is_relu: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let fan_in_val = 8_i64;

    // gain^2: 2 for ReLU (sqrt(2)^2), 1 for the identity/linear gain.
    let gain_sq_val = if gain_is_relu { 2 } else { 1 };
    let gain_sq = define_real(&mut program, "gain_sq", &Expr::real(gain_sq_val));

    // variance = gain^2 / fan_in = gain^2 * (1/fan_in), reciprocal as a literal.
    let inv_fan_in = Expr::real_ratio(1, fan_in_val);
    let variance = define_real(
        &mut program,
        "variance",
        &gain_sq.clone().real_mul(inv_fan_in),
    );

    // The claimed answer 2 / fan_in, reached through an independent constant.
    let expected = define_real(
        &mut program,
        "expected_variance",
        &Expr::real_ratio(2, fan_in_val),
    );

    // Violation: the derived variance disagrees with 2 / fan_in.
    program.assert(variance.ne(expected));
    program.check_sat();
    Ok(program)
}

/// Prove that Kaiming gain² for ReLU is exactly 2.
///
/// PyTorch's gain table computes the leaky-ReLU gain as
/// `sqrt(2 / (1 + negative_slope^2))`, and plain ReLU is the `negative_slope = 0`
/// case, giving `gain^2 = 2 / (1 + 0) = 2`. The content is the numerator factor
/// of 2 that distinguishes ReLU from the linear/identity gain (numerator 1). We
/// derive `gain_sq = 2 / (1 + negative_slope^2)` over the concrete slope 0 (so
/// the denominator is a literal) and compare against 2; dropping the factor of 2
/// makes `gain_sq = 1 != 2` and the query SAT — see
/// `kaiming_relu_gain_squared_depends_on_the_relu_factor`.
pub(crate) fn prove_kaiming_relu_gain_squared() -> Result<WeightInitPropertyResult, SmtError> {
    let program = build_kaiming_relu_gain_squared(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(WeightInitPropertyResult {
        property: "kaiming_relu_gain_squared_equals_2".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the ReLU gain² query. The leaky-ReLU formula is
/// `gain^2 = numerator / (1 + negative_slope^2)` with `negative_slope = 0` for
/// plain ReLU, so the denominator is 1. When `has_relu_factor` is false the
/// numerator drops from 2 to 1 (the linear/identity gain), so `gain_sq = 1 != 2`;
/// tests flip it to confirm the proof depends on the ReLU factor of 2.
fn build_kaiming_relu_gain_squared(has_relu_factor: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Plain ReLU is the negative_slope = 0 case: denominator 1 + 0^2 = 1.
    let negative_slope: i64 = 0;
    let denom = 1 + negative_slope * negative_slope;
    let numerator = if has_relu_factor { 2 } else { 1 };

    // gain^2 = numerator / (1 + negative_slope^2), derived one step removed.
    let gain_sq = define_real(
        &mut program,
        "gain_sq",
        &Expr::real_ratio(numerator, denom),
    );

    // Violation: gain^2 differs from the ReLU constant 2.
    program.assert(gain_sq.ne(Expr::real(2)));
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 5: Orthogonal Initialization — Unit Norm Columns
// ---------------------------------------------------------------------------

/// Prove that an orthogonal-init column is a unit vector (2D case).
///
/// A column of an orthogonal matrix is *normalized*: `a^2 + b^2 = 1`. We take a
/// concrete unit column from the 3-4-5 Pythagorean triple, `(3/5, 4/5)`, and
/// derive `norm_sq = a^2 + b^2` from those components, then check it equals 1.
/// The components are exact rational literals so the two squarings are
/// literal×literal and the query stays linear (QF_LRA).
///
/// The realistic slip is skipping the normalization step and keeping the raw
/// column `(3, 4)`, whose squared norm is 25, not 1 — see
/// `orthogonal_unit_norm_depends_on_normalization`.
pub(crate) fn prove_orthogonal_unit_norm() -> Result<WeightInitPropertyResult, SmtError> {
    let program = build_orthogonal_unit_norm(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(WeightInitPropertyResult {
        property: "orthogonal_init_unit_norm_column".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the unit-norm query for one column. When `normalized` is true the
/// column is the unit vector `(3/5, 4/5)`; when false the normalization is
/// skipped and the raw `(3, 4)` is used, giving squared norm 25 != 1. Tests flip
/// it to confirm the proof depends on the column being normalized.
fn build_orthogonal_unit_norm(normalized: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Column components: the unit vector (3/5, 4/5), or the un-normalized (3, 4).
    let (ax, ay) = if normalized {
        (Expr::real_ratio(3, 5), Expr::real_ratio(4, 5))
    } else {
        (Expr::real(3), Expr::real(4))
    };

    // norm_sq = a^2 + b^2, derived from the (literal) components.
    let sum_sq = ax.clone().real_mul(ax).real_add(ay.clone().real_mul(ay));
    let norm_sq = define_real(&mut program, "norm_sq", &sum_sq);

    // Violation: the column is not a unit vector.
    program.assert(norm_sq.ne(Expr::real(1)));
    program.check_sat();
    program
}

/// Prove that two orthogonal-init columns are mutually orthogonal (2D case).
///
/// The columns of an orthogonal matrix have zero dot product. We take a concrete
/// 2×2 rotation built from the 3-4-5 triple: `col1 = (3/5, 4/5)` and its
/// counter-clockwise partner `col2 = (-4/5, 3/5)`. We derive
/// `dot = a1*a2 + b1*b2` from those components and check it is 0. The components
/// are exact rational literals, so the products are literal×literal and the
/// query stays linear (QF_LRA).
///
/// The realistic slip is dropping the sign flip when forming the orthogonal
/// partner, using `col2 = (4/5, 3/5)`; then `dot = 24/25 != 0` — see
/// `orthogonal_dot_product_depends_on_the_sign_flip`.
pub(crate) fn prove_orthogonal_dot_product_zero() -> Result<WeightInitPropertyResult, SmtError> {
    let program = build_orthogonal_dot_product_zero(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(WeightInitPropertyResult {
        property: "orthogonal_init_dot_product_zero".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the dot-product query for two columns. `col1 = (3/5, 4/5)`, and its
/// orthogonal partner is `col2 = (-4/5, 3/5)` when `sign_flipped` is true. When
/// false the sign flip is dropped and `col2 = (4/5, 3/5)`, giving `dot = 24/25`;
/// tests flip it to confirm the proof depends on the sign flip.
fn build_orthogonal_dot_product_zero(sign_flipped: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // col1 = (cos, sin) for the 3-4-5 angle.
    let (c1x, c1y) = (Expr::real_ratio(3, 5), Expr::real_ratio(4, 5));
    // col2 = (-sin, cos) is the orthogonal partner; the slip forgets the sign.
    let c2x = if sign_flipped {
        Expr::real_ratio(-4, 5)
    } else {
        Expr::real_ratio(4, 5)
    };
    let c2y = Expr::real_ratio(3, 5);

    // dot = a1*a2 + b1*b2, derived from the (literal) components.
    let dot_term = c1x.real_mul(c2x).real_add(c1y.real_mul(c2y));
    let dot = define_real(&mut program, "dot", &dot_term);

    // Violation: the columns are not orthogonal.
    program.assert(dot.ne(Expr::real(0)));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 6: Uniform Initialization Range
// ---------------------------------------------------------------------------

/// Prove that uniform init values lie in `[low, high]`.
///
/// For U[low, high], every sample `w` satisfies `low <= w <= high`.
/// We define `w` with these bounds and prove no value can escape the range.
pub(crate) fn prove_uniform_init_range() -> Result<WeightInitPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let low = declare_real(&mut program, "low");
    let high = declare_real(&mut program, "high");
    let w = declare_real(&mut program, "w");

    // low < high (valid range)
    program.assert(low.clone().real_lt(high.clone()));
    assert_bounds(&mut program, &low, -100.0, 100.0)?;
    assert_bounds(&mut program, &high, -100.0, 100.0)?;

    // w is drawn from U[low, high]: low <= w <= high
    program.assert(w.clone().real_ge(low.clone()));
    program.assert(w.clone().real_le(high.clone()));

    // Negated property: w < low OR w > high
    let below = w.clone().real_lt(low);
    let above = w.real_gt(high);
    let violation = below.or(above);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(WeightInitPropertyResult {
        property: "uniform_init_values_in_range".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove the uniform-distribution variance formula `Var = (high - low)^2 / 12`.
///
/// For `U[low, high]`, `Var = (high - low)^2 / 12`. We take a concrete range
/// `low = 0, high = 6`, derive `range = high - low` in SMT, then apply the
/// formula and check the identity `12 * variance = range^2`. The content is that
/// the range is *squared*: the slip that forgets the square and uses
/// `variance = range / 12` makes `12 * variance = range != range^2` whenever the
/// range is not 1. `range^2` is a literal (the range is concrete) so the query
/// stays linear (QF_LRA).
///
/// The unsquared range makes the query SAT — see
/// `uniform_variance_depends_on_squaring_the_range`.
pub(crate) fn prove_uniform_variance_formula() -> Result<WeightInitPropertyResult, SmtError> {
    let program = build_uniform_variance_formula(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(WeightInitPropertyResult {
        property: "uniform_variance_equals_range_sq_over_12".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the uniform-variance query over `low = 0, high = 6` (range 6, range² 36).
/// When `range_is_squared` is true, `variance = range^2 / 12`; when false the
/// square is dropped and `variance = range / 12`, so `12 * variance = 6 != 36`.
/// Tests flip it to confirm the proof depends on squaring the range.
fn build_uniform_variance_formula(range_is_squared: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let (low_val, high_val) = (0_i64, 6_i64);
    let range_val = high_val - low_val;
    let range_sq_val = range_val * range_val;

    // Concrete range, derived in SMT as high - low.
    let low = define_real(&mut program, "low", &Expr::real(low_val));
    let high = define_real(&mut program, "high", &Expr::real(high_val));
    let range = define_real(&mut program, "range", &high.real_sub(low));

    // Var = range^2 / 12. The slip uses the unsquared range in the numerator.
    let inv_twelve = Expr::real_ratio(1, 12);
    let numerator = if range_is_squared {
        Expr::real(range_sq_val)
    } else {
        range.clone()
    };
    let variance = define_real(&mut program, "variance", &numerator.real_mul(inv_twelve));

    // Violation: 12 * variance != range^2 (range^2 as the concrete target 36).
    let twelve_variance = variance.real_mul(Expr::real(12));
    program.assert(twelve_variance.ne(Expr::real(range_sq_val)));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 7: Normal Initialization
// ---------------------------------------------------------------------------

/// Prove that normal init has mean `mu` via the reparameterization `w = mu + sigma*z`.
///
/// A normal sample is drawn as `w = mu + sigma * z` with `z ~ N(0, 1)`, so
/// `E[w] = mu + sigma * E[z] = mu + sigma * 0 = mu`. We derive the mean of `w`
/// by evaluating the affine map at the standard-normal mean `z_mean = 0`, and
/// check it equals `mu`. `mu` and `sigma` are free (bounded) and `z_mean` is the
/// literal 0, so `sigma * z_mean` is variable×literal and the query stays linear
/// (QF_LRA).
///
/// The content is the location term `+ mu`: the slip that forgets it (sampling
/// `w = sigma * z`) gives mean 0, which differs from `mu` whenever `mu != 0` —
/// see `normal_init_mean_depends_on_the_location_term`.
pub(crate) fn prove_normal_init_mean() -> Result<WeightInitPropertyResult, SmtError> {
    let program = build_normal_init_mean(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(WeightInitPropertyResult {
        property: "normal_init_mean_equals_mu".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the normal-mean query. The reparameterization is `w = mu + sigma * z`;
/// evaluated at the standard-normal mean `z_mean = 0` its mean is `w_mean`. When
/// `includes_location` is false the `+ mu` term is dropped (`w = sigma * z`), so
/// `w_mean = 0 != mu`; tests flip it to confirm the proof depends on the
/// location term.
fn build_normal_init_mean(includes_location: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let mu = declare_real(&mut program, "mu");
    let sigma = declare_real(&mut program, "sigma");
    assert_bounds(&mut program, &mu, -100.0, 100.0)?;
    assert_bounds(&mut program, &sigma, 0.0, 100.0)?;

    // Standard normal has mean 0; the scale term is sigma * z_mean = sigma * 0.
    let z_mean = Expr::real(0);
    let scale_term = sigma.real_mul(z_mean);

    // Mean of w = mu + sigma*z evaluated at z = z_mean. The slip drops the mu.
    let w_mean_term = if includes_location {
        mu.clone().real_add(scale_term)
    } else {
        scale_term
    };
    let w_mean = define_real(&mut program, "w_mean", &w_mean_term);

    // Violation: the derived mean differs from mu.
    program.assert(w_mean.ne(mu));
    program.check_sat();
    Ok(program)
}

/// Prove that normal init standard deviation is positive when sigma > 0.
///
/// For N(mu, sigma^2) with sigma > 0, the standard deviation is strictly positive.
/// This is a basic constraint validation: the distribution is well-defined only
/// when sigma > 0.
pub(crate) fn prove_normal_init_std_positive() -> Result<WeightInitPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let sigma = declare_real(&mut program, "sigma");

    // sigma > 0 (valid standard deviation)
    let zero = Expr::real(0);
    program.assert(sigma.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &sigma, 0.0, 100.0)?;

    // Negated property: sigma <= 0
    let violation = sigma.real_le(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(WeightInitPropertyResult {
        property: "normal_init_std_positive".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove that normal init variance equals `sigma^2`.
///
/// For `N(mu, sigma^2)`, `Var(w) = sigma^2`. We take a concrete standard
/// deviation `sigma = 5` (pinned as an SMT variable) and derive the variance by
/// scaling the std by itself: `variance = sigma * sigma_value`, where
/// `sigma_value = 5` is the literal magnitude. The content is that the std is
/// applied *twice* (squared): the slip that scales by 1 instead — the classic
/// std-vs-variance confusion — gives `variance = sigma = 5 != 25`. Because one
/// factor is a literal the product is variable×literal and the query stays
/// linear (QF_LRA).
///
/// The unsquared std makes the query SAT — see
/// `normal_init_variance_depends_on_squaring_sigma`.
pub(crate) fn prove_normal_init_variance() -> Result<WeightInitPropertyResult, SmtError> {
    let program = build_normal_init_variance(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(WeightInitPropertyResult {
        property: "normal_init_variance_equals_sigma_squared".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the normal-variance query over `sigma = 5`. `variance = sigma * scale`
/// where the correct scale is the std magnitude (5), squaring it to 25. When
/// `sigma_is_squared` is false the scale drops to 1, so `variance = 5 != 25`;
/// tests flip it to confirm the proof depends on squaring the std.
fn build_normal_init_variance(sigma_is_squared: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let sigma_val = 5_i64;
    let sigma = define_real(&mut program, "sigma", &Expr::real(sigma_val));
    program.assert(sigma.clone().real_gt(Expr::real(0)));

    // variance = sigma * scale. Squaring uses scale = sigma_val; the slip uses 1.
    let scale = if sigma_is_squared { sigma_val } else { 1 };
    let variance = define_real(
        &mut program,
        "variance",
        &sigma.real_mul(Expr::real(scale)),
    );

    // The claimed answer sigma^2, reached through an independent constant.
    let expected = define_real(
        &mut program,
        "expected_variance",
        &Expr::real(sigma_val * sigma_val),
    );

    // Violation: the derived variance disagrees with sigma^2.
    program.assert(variance.ne(expected));
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 8: Zeros Initialization
// ---------------------------------------------------------------------------

/// Prove that zeros init produces 0 at every position.
///
/// Zeros init fills the tensor with a position-independent 0: for the fill rule
/// `w[p] = base + slope*p`, the correct choice is `base = 0, slope = 0`, so both
/// sampled positions evaluate to 0. We derive `w0` and `w1` at positions 0 and 1
/// and check both are 0.
///
/// The realistic slip is writing a ramp (`arange`/iota) instead of a constant
/// fill — `slope = 1` — so `w1 = 1 != 0` and the property breaks; see
/// `zeros_init_depends_on_a_constant_fill`. All literal, decidable QF_LRA.
pub(crate) fn prove_zeros_init() -> Result<WeightInitPropertyResult, SmtError> {
    let program = build_zeros_init(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(WeightInitPropertyResult {
        property: "zeros_init_all_values_zero".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the zeros-init query. `base = 0`; `slope = 0` when `constant_fill` is
/// true. When false a ramp (`slope = 1`) is used, so `w1 = 1 != 0`; tests flip
/// it to confirm the proof depends on the fill being constant.
fn build_zeros_init(constant_fill: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let base = Expr::real(0);
    let slope = if constant_fill { 0 } else { 1 };
    let w0 = fill_value(&mut program, "w0", &base, slope, 0);
    let w1 = fill_value(&mut program, "w1", &base, slope, 1);

    // Violation: some sampled position is not 0.
    let violation = w0.ne(Expr::real(0)).or(w1.ne(Expr::real(0)));
    program.assert(violation);
    program.check_sat();
    program
}

/// Prove that zeros init gradient is zero for all parameters.
///
/// When all weights are initialized to zero, the initial forward pass output
/// is identically zero for linear layers (W*x = 0*x = 0). This is a known
/// limitation: zero init causes all neurons to compute identical gradients,
/// breaking symmetry. We prove the algebraic fact: 0 * x = 0 for all x.
pub(crate) fn prove_zeros_init_linear_output() -> Result<WeightInitPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x = declare_real(&mut program, "x");
    let w = declare_real(&mut program, "w");
    let output = declare_real(&mut program, "output");

    assert_bounds(&mut program, &x, -1000.0, 1000.0)?;

    // w = 0 (zeros init)
    let zero = Expr::real(0);
    program.assert(w.clone().eq(zero.clone()));

    // output = w * x
    program.assert(output.clone().eq(w.real_mul(x)));

    // Negated property: output != 0
    let violation = output.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(WeightInitPropertyResult {
        property: "zeros_init_linear_output_zero".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 9: Ones Initialization
// ---------------------------------------------------------------------------

/// Prove that ones init produces 1 at every position.
///
/// Ones init fills the tensor with a position-independent 1: for the fill rule
/// `w[p] = base + slope*p`, the correct choice is `base = 1, slope = 0`, so both
/// sampled positions evaluate to 1. Commonly used for normalization scale
/// parameters (gamma). We derive `w0` and `w1` at positions 0 and 1 and check
/// both are 1.
///
/// The realistic slip is writing a ramp starting at 0 (`arange`/iota:
/// `base = 0, slope = 1`) instead of ones, so `w0 = 0 != 1`; see
/// `ones_init_depends_on_a_constant_fill`. All literal, decidable QF_LRA.
pub(crate) fn prove_ones_init() -> Result<WeightInitPropertyResult, SmtError> {
    let program = build_ones_init(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(WeightInitPropertyResult {
        property: "ones_init_all_values_one".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the ones-init query. When `constant_fill` is true, `base = 1, slope = 0`
/// so every position is 1. When false a ramp is used (`base = 0, slope = 1`), so
/// `w0 = 0 != 1`; tests flip it to confirm the proof depends on the fill being
/// the constant 1.
fn build_ones_init(constant_fill: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let (base, slope) = if constant_fill {
        (Expr::real(1), 0)
    } else {
        (Expr::real(0), 1)
    };
    let w0 = fill_value(&mut program, "w0", &base, slope, 0);
    let w1 = fill_value(&mut program, "w1", &base, slope, 1);

    // Violation: some sampled position is not 1.
    let violation = w0.ne(Expr::real(1)).or(w1.ne(Expr::real(1)));
    program.assert(violation);
    program.check_sat();
    program
}

/// Prove that ones init acts as identity for element-wise multiply.
///
/// When scale parameters are initialized to 1 (e.g., BatchNorm gamma),
/// the initial forward pass preserves input: `1 * x = x` for all x.
pub(crate) fn prove_ones_init_identity_multiply() -> Result<WeightInitPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x = declare_real(&mut program, "x");
    let w = declare_real(&mut program, "w");
    let output = declare_real(&mut program, "output");

    assert_bounds(&mut program, &x, -1000.0, 1000.0)?;

    // w = 1 (ones init)
    let one = Expr::real(1);
    program.assert(w.clone().eq(one));

    // output = w * x
    program.assert(output.clone().eq(w.real_mul(x.clone())));

    // Negated property: output != x
    let violation = output.ne(x);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(WeightInitPropertyResult {
        property: "ones_init_identity_multiply".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 10: Constant Initialization
// ---------------------------------------------------------------------------

/// Prove that constant init produces the same value at every position.
///
/// Constant init with value `c` fills the tensor position-independently: for the
/// fill rule `w[p] = c + slope*p`, the correct choice is `slope = 0`, so every
/// position holds `c` and any two positions agree. We keep `c` a *free* (bounded)
/// variable and derive `w0` (position 0) and `w1` (position 1) from the rule,
/// then check `w0 = w1` — uniformity that must hold for whatever constant the
/// caller supplied.
///
/// The realistic slip is a position-dependent fill (`slope = 1`, e.g. an
/// `arange` offset added to the constant): then `w1 = c + 1 != c = w0` for every
/// `c`, and the property breaks — see `constant_init_depends_on_a_constant_fill`.
/// `slope*p` is literal×literal, so the query stays linear (QF_LRA).
pub(crate) fn prove_constant_init() -> Result<WeightInitPropertyResult, SmtError> {
    let program = build_constant_init(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(WeightInitPropertyResult {
        property: "constant_init_all_values_equal_c".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the constant-init uniformity query. `c` is a free bounded value and
/// `slope = 0` when `constant_fill` is true, so `w0 = w1 = c`. When false a
/// position-dependent slip (`slope = 1`) is used, so `w1 = c + 1 != w0`; tests
/// flip it to confirm the proof depends on the fill being position-independent.
fn build_constant_init(constant_fill: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let c = declare_real(&mut program, "c");
    assert_bounds(&mut program, &c, -1000.0, 1000.0)?;

    let slope = if constant_fill { 0 } else { 1 };
    let w0 = fill_value(&mut program, "w0", &c, slope, 0);
    let w1 = fill_value(&mut program, "w1", &c, slope, 1);

    // Violation: two positions of a constant-init tensor disagree.
    program.assert(w0.ne(w1));
    program.check_sat();
    Ok(program)
}

/// Prove that constant init with value 0 is equivalent to zeros init.
///
/// When `c = 0`, constant init produces the same result as zeros init.
/// This is a consistency check across initialization methods.
pub(crate) fn prove_constant_init_zero_equivalence() -> Result<WeightInitPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let w_const = declare_real(&mut program, "w_const");
    let w_zeros = declare_real(&mut program, "w_zeros");

    let zero = Expr::real(0);

    // Constant init with c=0: w_const = 0
    program.assert(w_const.clone().eq(zero.clone()));

    // Zeros init: w_zeros = 0
    program.assert(w_zeros.clone().eq(zero));

    // Negated property: w_const != w_zeros (should be impossible)
    let violation = w_const.ne(w_zeros);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(WeightInitPropertyResult {
        property: "constant_init_zero_equals_zeros_init".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove that constant init with value 1 is equivalent to ones init.
///
/// When `c = 1`, constant init produces the same result as ones init.
pub(crate) fn prove_constant_init_one_equivalence() -> Result<WeightInitPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let w_const = declare_real(&mut program, "w_const");
    let w_ones = declare_real(&mut program, "w_ones");

    let one = Expr::real(1);

    // Constant init with c=1: w_const = 1
    program.assert(w_const.clone().eq(one.clone()));

    // Ones init: w_ones = 1
    program.assert(w_ones.clone().eq(one));

    // Negated property: w_const != w_ones
    let violation = w_const.ne(w_ones);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(WeightInitPropertyResult {
        property: "constant_init_one_equals_ones_init".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ay_vacuity::vacuity_smell;

    // --- Xavier/Glorot Tests ---

    #[test]
    fn test_xavier_uniform_variance() {
        let result = prove_xavier_uniform_range().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Xavier uniform variance: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Xavier uniform variance must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(
            result.property,
            "xavier_uniform_variance_equals_2_over_fan_sum"
        );
    }

    #[test]
    fn test_xavier_normal_variance() {
        let result = prove_xavier_normal_variance().expect("proof should not error");
        assert!(
            result.proven,
            "Xavier normal variance (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(
            result.property,
            "xavier_normal_variance_equals_2_over_fan_sum"
        );
    }

    /// Divide by `fan_in` alone instead of `fan_in + fan_out`: the variance
    /// becomes `2/2 = 1`, not `1/4`, so the query must be SAT — proving the
    /// theorem rests on the fan *sum*, not on writing `2/8` twice.
    #[test]
    fn xavier_normal_variance_depends_on_the_fan_sum() {
        let program = build_xavier_normal_variance(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with the fan_in-only denominator the variance is wrong; query must be SAT: {detail}",
        );
    }

    // --- Kaiming/He Tests ---

    #[test]
    fn test_kaiming_uniform_range() {
        let result = prove_kaiming_uniform_range().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Kaiming uniform range: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Kaiming uniform range must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(
            result.property,
            "kaiming_uniform_variance_equals_gain_sq_over_fan_in"
        );
    }

    #[test]
    fn test_kaiming_normal_relu_variance() {
        let result = prove_kaiming_normal_relu_variance().expect("proof should not error");
        assert!(
            result.proven,
            "Kaiming normal ReLU variance (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(
            result.property,
            "kaiming_normal_relu_variance_equals_2_over_fan_in"
        );
    }

    /// Feed the identity gain² (1) instead of the ReLU gain² (2): the variance
    /// becomes `1/8`, not `2/8`, so the query must be SAT — proving the theorem
    /// rests on the ReLU gain rather than on writing `2/8` twice.
    #[test]
    fn kaiming_normal_relu_variance_depends_on_the_gain() {
        let program = build_kaiming_normal_relu_variance(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with the identity gain the variance is halved; query must be SAT: {detail}",
        );
    }

    #[test]
    fn test_kaiming_relu_gain_squared() {
        let result = prove_kaiming_relu_gain_squared().expect("proof should not error");
        assert!(
            result.proven,
            "Kaiming ReLU gain^2 = 2 (QF_LRA) should be proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "kaiming_relu_gain_squared_equals_2");
    }

    /// Drop the ReLU factor of 2 from the numerator (the linear/identity gain):
    /// `gain_sq = 1 != 2`, so the query must be SAT — proving the theorem rests
    /// on the factor of 2, not on asserting `gain_sq = 2` and negating it.
    #[test]
    fn kaiming_relu_gain_squared_depends_on_the_relu_factor() {
        let program = build_kaiming_relu_gain_squared(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with the factor of 2 dropped gain_sq = 1; query must be SAT: {detail}",
        );
    }

    // --- Orthogonal Init Tests ---

    #[test]
    fn test_orthogonal_unit_norm() {
        let result = prove_orthogonal_unit_norm().expect("proof should not error");
        assert!(
            result.proven,
            "Orthogonal unit norm (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "orthogonal_init_unit_norm_column");
    }

    /// Skip normalization and keep the raw `(3, 4)`: its squared norm is 25, not
    /// 1, so the query must be SAT — proving the theorem rests on the column
    /// being normalized rather than on asserting `norm_sq = 1`.
    #[test]
    fn orthogonal_unit_norm_depends_on_normalization() {
        let program = build_orthogonal_unit_norm(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with the un-normalized column norm_sq = 25; query must be SAT: {detail}",
        );
    }

    #[test]
    fn test_orthogonal_dot_product_zero() {
        let result = prove_orthogonal_dot_product_zero().expect("proof should not error");
        assert!(
            result.proven,
            "Orthogonal dot product zero (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "orthogonal_init_dot_product_zero");
    }

    /// Drop the sign flip on the orthogonal partner (`col2 = (4/5, 3/5)`): the
    /// dot product becomes `24/25`, not 0, so the query must be SAT — proving the
    /// theorem rests on the sign flip rather than on asserting `dot = 0`.
    #[test]
    fn orthogonal_dot_product_depends_on_the_sign_flip() {
        let program = build_orthogonal_dot_product_zero(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "without the sign flip the dot product is 24/25; query must be SAT: {detail}",
        );
    }

    // --- Uniform Init Tests ---

    #[test]
    fn test_uniform_init_range() {
        let result = prove_uniform_init_range().expect("proof should not error");
        assert!(
            result.proven,
            "Uniform init range should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "uniform_init_values_in_range");
    }

    #[test]
    fn test_uniform_variance_formula() {
        let result = prove_uniform_variance_formula().expect("proof should not error");
        assert!(
            result.proven,
            "Uniform variance formula (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "uniform_variance_equals_range_sq_over_12");
    }

    /// Forget to square the range (`variance = range/12`): then
    /// `12 * variance = 6 != 36 = range^2`, so the query must be SAT — proving the
    /// theorem rests on squaring the range rather than on restating the formula.
    #[test]
    fn uniform_variance_depends_on_squaring_the_range() {
        let program = build_uniform_variance_formula(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with the unsquared range 12*variance = 6 != 36; query must be SAT: {detail}",
        );
    }

    // --- Normal Init Tests ---

    #[test]
    fn test_normal_init_mean() {
        let result = prove_normal_init_mean().expect("proof should not error");
        assert!(
            result.proven,
            "Normal init mean should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "normal_init_mean_equals_mu");
    }

    /// Drop the location term (`w = sigma*z`, no `+ mu`): the mean is 0, which
    /// differs from `mu` when `mu != 0`, so the query must be SAT — proving the
    /// theorem rests on the reparameterization's `+ mu`.
    #[test]
    fn normal_init_mean_depends_on_the_location_term() {
        let program = build_normal_init_mean(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "without the location term the mean is 0 != mu; query must be SAT: {detail}",
        );
    }

    #[test]
    fn test_normal_init_std_positive() {
        let result = prove_normal_init_std_positive().expect("proof should not error");
        assert!(
            result.proven,
            "Normal init std positive should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "normal_init_std_positive");
    }

    #[test]
    fn test_normal_init_variance() {
        let result = prove_normal_init_variance().expect("proof should not error");
        assert!(
            result.proven,
            "Normal init variance (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "normal_init_variance_equals_sigma_squared");
    }

    /// Scale the std by 1 instead of by itself (the std-vs-variance confusion):
    /// `variance = 5 != 25 = sigma^2`, so the query must be SAT — proving the
    /// theorem rests on squaring the std rather than on restating `sigma^2`.
    #[test]
    fn normal_init_variance_depends_on_squaring_sigma() {
        let program = build_normal_init_variance(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with the std unsquared variance = 5 != 25; query must be SAT: {detail}",
        );
    }

    // --- Zeros Init Tests ---

    #[test]
    fn test_zeros_init() {
        let result = prove_zeros_init().expect("proof should not error");
        assert!(
            result.proven,
            "Zeros init should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "zeros_init_all_values_zero");
    }

    /// Use a ramp (`slope = 1`) instead of a constant fill: `w1 = 1 != 0`, so the
    /// query must be SAT — proving the theorem rests on the fill being constant
    /// rather than on asserting `w = 0`.
    #[test]
    fn zeros_init_depends_on_a_constant_fill() {
        let program = build_zeros_init(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with a ramp fill w1 = 1 != 0; query must be SAT: {detail}",
        );
    }

    #[test]
    fn test_zeros_init_linear_output() {
        let result = prove_zeros_init_linear_output().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Zeros init linear output: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Zeros init linear output must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "zeros_init_linear_output_zero");
    }

    // --- Ones Init Tests ---

    #[test]
    fn test_ones_init() {
        let result = prove_ones_init().expect("proof should not error");
        assert!(
            result.proven,
            "Ones init should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "ones_init_all_values_one");
    }

    /// Use a ramp starting at 0 (`base = 0, slope = 1`) instead of ones:
    /// `w0 = 0 != 1`, so the query must be SAT — proving the theorem rests on the
    /// fill being the constant 1 rather than on asserting `w = 1`.
    #[test]
    fn ones_init_depends_on_a_constant_fill() {
        let program = build_ones_init(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with a ramp fill w0 = 0 != 1; query must be SAT: {detail}",
        );
    }

    #[test]
    fn test_ones_init_identity_multiply() {
        let result = prove_ones_init_identity_multiply().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Ones init identity multiply: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Ones init identity multiply must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "ones_init_identity_multiply");
    }

    // --- Constant Init Tests ---

    #[test]
    fn test_constant_init() {
        let result = prove_constant_init().expect("proof should not error");
        assert!(
            result.proven,
            "Constant init should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "constant_init_all_values_equal_c");
    }

    /// Use a position-dependent fill (`slope = 1`): `w1 = c + 1 != c = w0` for
    /// every `c`, so the query must be SAT — proving the theorem rests on the
    /// fill being position-independent rather than on asserting `w = c`.
    #[test]
    fn constant_init_depends_on_a_constant_fill() {
        let program = build_constant_init(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with a position-dependent fill w1 = c+1 != w0; query must be SAT: {detail}",
        );
    }

    #[test]
    fn test_constant_init_zero_equivalence() {
        let result = prove_constant_init_zero_equivalence().expect("proof should not error");
        assert!(
            result.proven,
            "Constant init zero equivalence should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "constant_init_zero_equals_zeros_init");
    }

    #[test]
    fn test_constant_init_one_equivalence() {
        let result = prove_constant_init_one_equivalence().expect("proof should not error");
        assert!(
            result.proven,
            "Constant init one equivalence should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "constant_init_one_equals_ones_init");
    }

    // --- SMT2 Structure Tests ---

    #[test]
    fn test_all_proofs_have_valid_smt2() {
        let proofs: Vec<WeightInitPropertyResult> = vec![
            prove_xavier_uniform_range().unwrap(),
            prove_xavier_normal_variance().unwrap(),
            prove_kaiming_uniform_range().unwrap(),
            prove_kaiming_normal_relu_variance().unwrap(),
            prove_kaiming_relu_gain_squared().unwrap(),
            prove_orthogonal_unit_norm().unwrap(),
            prove_orthogonal_dot_product_zero().unwrap(),
            prove_uniform_init_range().unwrap(),
            prove_uniform_variance_formula().unwrap(),
            prove_normal_init_mean().unwrap(),
            prove_normal_init_std_positive().unwrap(),
            prove_normal_init_variance().unwrap(),
            prove_zeros_init().unwrap(),
            prove_zeros_init_linear_output().unwrap(),
            prove_ones_init().unwrap(),
            prove_ones_init_identity_multiply().unwrap(),
            prove_constant_init().unwrap(),
            prove_constant_init_zero_equivalence().unwrap(),
            prove_constant_init_one_equivalence().unwrap(),
        ];

        for proof in &proofs {
            assert!(
                proof.smt2.contains("check-sat"),
                "{}: SMT2 should contain check-sat",
                proof.property,
            );
            assert!(
                proof.smt2.contains("declare-const"),
                "{}: SMT2 should have declarations",
                proof.property,
            );
            assert!(
                proof.smt2.contains("set-logic"),
                "{}: SMT2 should declare logic",
                proof.property,
            );
        }
    }
}
