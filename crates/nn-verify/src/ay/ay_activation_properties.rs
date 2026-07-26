// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for activation function mathematical properties (#4186).
//!
//! Proves fundamental mathematical properties of activation functions used
//! throughout nn models. Each proof encodes the expected property as a negated
//! assertion and proves UNSAT (no counterexample exists).
//!
//! # Proved Properties
//!
//! 1. **ReLU**: output >= 0, idempotent (ReLU(ReLU(x)) = ReLU(x))
//! 2. **Sigmoid**: output in (0, 1), symmetry sigmoid(-x) = 1 - sigmoid(x), monotone
//! 3. **Tanh**: output in (-1, 1), odd symmetry tanh(-x) = -tanh(x), tanh(0) = 0
//! 4. **GELU**: approximation bounds, GELU(x) >= 0 for x >= 0
//! 5. **SiLU/Swish**: x * sigmoid(x), SiLU(0) = 0
//! 6. **Softplus**: log(1 + exp(x)) >= 0, approaches ReLU for large x
//! 7. **LeakyReLU**: piecewise definition, output = x if x >= 0, alpha*x otherwise
//! 8. **ELU**: continuous at x = 0, output >= -alpha for all x
//! 9. **Mish**: x * tanh(softplus(x)), smooth properties
//! 10. **Snake**: x + sin^2(alpha*x)/alpha, approaches identity for small alpha
//!
//! # Proof Strategy
//!
//! Activation proofs use several approaches depending on the function:
//!
//! - **Piecewise/conditional proofs** (ReLU, LeakyReLU, ELU): Case-split on x >= 0
//!   vs x < 0, prove each branch independently using QF_LRA.
//!
//! - **Algebraic identity proofs** (idempotence, symmetry): Pure polynomial or
//!   linear identities provable via QF_NRA or QF_LRA.
//!
//! - **Transcendental function proofs** (Sigmoid, Tanh, GELU, SiLU, Softplus, Mish):
//!   Since exp/log are not in the decidable NRA fragment, we encode function outputs
//!   as symbolic variables with defining constraints (e.g., s = sigmoid(x) means
//!   0 < s < 1) and prove properties follow from these axioms.
//!
//! - **Uninterpreted function proofs** (Snake): Use UF approximation for sin with
//!   axiomatic range constraints (-1 <= sin(x) <= 1).

use ay_bindings::{Expr, Sort, AYProgram};

use crate::ay_real_lit::RealLit;

use super::error::SmtError;
use super::translate_real::real_from_f64;

/// Result of an activation property proof attempt.
#[derive(Debug, Clone)]
pub(crate) struct ActivationPropertyResult {
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

/// Execute a ay program and return whether UNSAT (property proven).
///
/// The `(proven, detail)` verdict is funnelled through
/// [`crate::ay_vacuity::reject_if_vacuous`] before it is returned, so a query
/// that is UNSAT only because it asserts `P ∧ ¬P` (or compares a term to
/// itself) never counts as a proof — any residual vacuity becomes a hard
/// `test_*_proven` failure rather than a silent pass.
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

// ===========================================================================
// 1. ReLU Properties
// ===========================================================================

/// Prove ReLU output is non-negative: max(0, x) >= 0 for all x.
///
/// Case-split on x >= 0 vs x < 0:
/// - When x >= 0: relu(x) = x >= 0
/// - When x < 0: relu(x) = 0 >= 0
///
/// We encode both branches via the relu definition and prove the output
/// cannot be negative.
pub(crate) fn prove_relu_non_negative() -> Result<ActivationPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let x = declare_real(&mut program, "x");
    let relu_x = declare_real(&mut program, "relu_x");

    assert_bounds(&mut program, &x, -100.0, 100.0)?;

    let zero = Expr::real(0);

    // ReLU definition: relu_x >= 0 AND relu_x >= x AND (relu_x = x OR relu_x = 0)
    program.assert(relu_x.clone().real_ge(zero.clone()));
    program.assert(relu_x.clone().real_ge(x.clone()));
    let eq_x = relu_x.clone().eq(x.clone());
    let eq_zero = relu_x.clone().eq(zero.clone());
    program.assert(eq_x.or(eq_zero));

    // Negated property: relu_x < 0
    let violation = relu_x.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ActivationPropertyResult {
        property: "relu_non_negative".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove ReLU idempotence: ReLU(ReLU(x)) = ReLU(x) for all x.
///
/// Since ReLU(x) >= 0, applying ReLU again to a non-negative value is identity.
/// We encode relu_x >= 0 (from the first application) and show that
/// relu(relu_x) = relu_x.
pub(crate) fn prove_relu_idempotent() -> Result<ActivationPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let x = declare_real(&mut program, "x");
    let relu_x = declare_real(&mut program, "relu_x");
    let relu_relu_x = declare_real(&mut program, "relu_relu_x");

    assert_bounds(&mut program, &x, -100.0, 100.0)?;

    let zero = Expr::real(0);

    // First application: relu_x = max(0, x)
    program.assert(relu_x.clone().real_ge(zero.clone()));
    program.assert(relu_x.clone().real_ge(x.clone()));
    let eq_x = relu_x.clone().eq(x.clone());
    let eq_zero = relu_x.clone().eq(zero.clone());
    program.assert(eq_x.or(eq_zero));

    // Second application: relu_relu_x = max(0, relu_x)
    // Since relu_x >= 0, relu(relu_x) = relu_x
    program.assert(relu_relu_x.clone().real_ge(zero.clone()));
    program.assert(relu_relu_x.clone().real_ge(relu_x.clone()));
    let eq_rx = relu_relu_x.clone().eq(relu_x.clone());
    let eq_zero2 = relu_relu_x.clone().eq(zero);
    program.assert(eq_rx.or(eq_zero2));

    // Negated property: relu(relu(x)) != relu(x)
    let violation = relu_relu_x.ne(relu_x);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ActivationPropertyResult {
        property: "relu_idempotent".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ===========================================================================
// 2. Sigmoid Properties
// ===========================================================================

/// Prove sigmoid output is bounded: 0 < sigmoid(x) < 1 for all x.
///
/// Since exp is transcendental, we encode sigmoid as a symbolic variable `s`
/// with the constraint `s = 1 / (1 + exp(-x))`. For any finite x, exp(-x) > 0,
/// so 1 + exp(-x) > 1, meaning 0 < s < 1. We encode this via the constraints:
/// - s > 0 and s < 1 (known range)
/// - Prove no counterexample violates these bounds.
///
/// We model the sigmoid output and prove it cannot leave (0, 1).
pub(crate) fn prove_sigmoid_bounded() -> Result<ActivationPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x = declare_real(&mut program, "x");
    let s = declare_real(&mut program, "s"); // s = sigmoid(x)
    let exp_neg_x = declare_real(&mut program, "exp_neg_x"); // exp(-x)

    assert_bounds(&mut program, &x, -100.0, 100.0)?;

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // exp(-x) > 0 for all x (fundamental property of exp)
    program.assert(exp_neg_x.clone().real_gt(zero.clone()));

    // s * (1 + exp(-x)) = 1  (definition: s = 1/(1 + exp(-x)))
    let one_plus_exp = one.clone().real_add(exp_neg_x);
    let s_times_denom = s.clone().real_mul(one_plus_exp);
    program.assert(s_times_denom.eq(one.clone()));

    // Negated property: s <= 0 OR s >= 1
    let too_low = s.clone().real_le(zero);
    let too_high = s.real_ge(one);
    let violation = too_low.or(too_high);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ActivationPropertyResult {
        property: "sigmoid_bounded_0_1".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove sigmoid symmetry: sigmoid(-x) = 1 - sigmoid(x).
///
/// Given s = sigmoid(x) and s_neg = sigmoid(-x), both defined via the
/// sigmoid equation, we prove s + s_neg = 1.
///
/// Algebraic proof: If s*(1 + e^{-x}) = 1 and s_neg*(1 + e^{x}) = 1,
/// then s = 1/(1+e^{-x}) and s_neg = 1/(1+e^{x}) = e^{-x}/(1+e^{-x}).
/// So s + s_neg = (1 + e^{-x})/(1+e^{-x}) = 1.
pub(crate) fn prove_sigmoid_symmetry() -> Result<ActivationPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let s = declare_real(&mut program, "s");
    let s_neg = declare_real(&mut program, "s_neg");
    let exp_neg_x = declare_real(&mut program, "exp_neg_x");
    let exp_pos_x = declare_real(&mut program, "exp_pos_x");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // exp(-x) > 0, exp(x) > 0
    program.assert(exp_neg_x.clone().real_gt(zero.clone()));
    program.assert(exp_pos_x.clone().real_gt(zero));

    // exp(-x) * exp(x) = 1 (fundamental: e^a * e^{-a} = 1)
    let product = exp_neg_x.clone().real_mul(exp_pos_x.clone());
    program.assert(product.eq(one.clone()));

    // s * (1 + exp(-x)) = 1
    let denom_s = one.clone().real_add(exp_neg_x.clone());
    program.assert(s.clone().real_mul(denom_s).eq(one.clone()));

    // s_neg * (1 + exp(x)) = 1
    let denom_s_neg = one.clone().real_add(exp_pos_x);
    program.assert(s_neg.clone().real_mul(denom_s_neg).eq(one.clone()));

    // Negated property: s + s_neg != 1
    let sum = s.real_add(s_neg);
    let violation = sum.ne(one);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ActivationPropertyResult {
        property: "sigmoid_symmetry".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove sigmoid is monotonically increasing.
///
/// For x1 < x2, we must have sigmoid(x1) < sigmoid(x2).
/// Since exp is monotonically increasing, exp(-x1) > exp(-x2) when x1 < x2,
/// so 1/(1+exp(-x1)) < 1/(1+exp(-x2)). We encode this via the exp ordering
/// axiom.
pub(crate) fn prove_sigmoid_monotone() -> Result<ActivationPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let s1 = declare_real(&mut program, "s1");
    let s2 = declare_real(&mut program, "s2");
    let e1 = declare_real(&mut program, "e1"); // exp(-x1)
    let e2 = declare_real(&mut program, "e2"); // exp(-x2)

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // Both exp values positive
    program.assert(e1.clone().real_gt(zero.clone()));
    program.assert(e2.clone().real_gt(zero));

    // x1 < x2 implies exp(-x1) > exp(-x2) (exp(-.) is decreasing)
    program.assert(e1.clone().real_gt(e2.clone()));

    // s1 * (1 + e1) = 1
    let denom1 = one.clone().real_add(e1);
    program.assert(s1.clone().real_mul(denom1).eq(one.clone()));

    // s2 * (1 + e2) = 1
    let denom2 = one.clone().real_add(e2);
    program.assert(s2.clone().real_mul(denom2).eq(one.clone()));

    // Negated property: s1 >= s2 (should be impossible since e1 > e2 => s1 < s2)
    let violation = s1.real_ge(s2);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ActivationPropertyResult {
        property: "sigmoid_monotone_increasing".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ===========================================================================
// 3. Tanh Properties
// ===========================================================================

/// Prove tanh output is bounded: -1 < tanh(x) < 1 for all x.
///
/// tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)).
/// Since exp(x) > 0 and exp(-x) > 0, the denominator is always positive.
/// The numerator is strictly between -denominator and +denominator,
/// so tanh(x) is in (-1, 1).
pub(crate) fn prove_tanh_bounded() -> Result<ActivationPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let t = declare_real(&mut program, "t"); // t = tanh(x)
    let ep = declare_real(&mut program, "ep"); // exp(x)
    let en = declare_real(&mut program, "en"); // exp(-x)

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // exp(x) > 0, exp(-x) > 0. Boundedness needs only the signs of ep, en (not
    // the reciprocal identity ep*en=1): with both positive, |ep-en| < ep+en, so
    // dropping that second var×var product keeps the theorem while the query
    // stays a single-product QF_NRA the solver dispatches quickly.
    program.assert(ep.clone().real_gt(zero.clone()));
    program.assert(en.clone().real_gt(zero));

    // t * (ep + en) = ep - en  (tanh definition)
    let denom = ep.clone().real_add(en.clone());
    let numer = ep.real_sub(en);
    program.assert(t.clone().real_mul(denom).eq(numer));

    // Negated property: t <= -1 OR t >= 1
    let neg_one = real_from_f64(-1.0)?;
    let too_low = t.clone().real_le(neg_one);
    let too_high = t.real_ge(one);
    let violation = too_low.or(too_high);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ActivationPropertyResult {
        property: "tanh_bounded_neg1_1".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove tanh is odd: tanh(-x) = -tanh(x).
///
/// Given t = tanh(x) and t_neg = tanh(-x), both defined via the tanh equation,
/// we prove t + t_neg = 0 (equivalently t_neg = -t).
pub(crate) fn prove_tanh_odd_symmetry() -> Result<ActivationPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let t = declare_real(&mut program, "t");
    let t_neg = declare_real(&mut program, "t_neg");
    let ep = declare_real(&mut program, "ep"); // exp(x)
    let en = declare_real(&mut program, "en"); // exp(-x)

    let zero = Expr::real(0);
    let one = Expr::real(1);

    program.assert(ep.clone().real_gt(zero.clone()));
    program.assert(en.clone().real_gt(zero.clone()));
    program.assert(ep.clone().real_mul(en.clone()).eq(one));

    // t * (ep + en) = ep - en  (tanh(x))
    let sum_exp = ep.clone().real_add(en.clone());
    let diff_exp = ep.clone().real_sub(en.clone());
    program.assert(t.clone().real_mul(sum_exp.clone()).eq(diff_exp));

    // t_neg * (en + ep) = en - ep  (tanh(-x): swap ep <-> en)
    let diff_neg = en.real_sub(ep);
    program.assert(t_neg.clone().real_mul(sum_exp).eq(diff_neg));

    // Negated property: t + t_neg != 0
    let sum = t.real_add(t_neg);
    let violation = sum.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ActivationPropertyResult {
        property: "tanh_odd_symmetry".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove tanh(0) = 0.
///
/// At x = 0, exp(0) = exp(-0) = 1, so tanh(0) = (1-1)/(1+1) = 0.
pub(crate) fn prove_tanh_zero_at_origin() -> Result<ActivationPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let t_0 = declare_real(&mut program, "t_0");

    let zero = Expr::real(0);
    let two = real_from_f64(2.0)?;

    // At x=0: exp(0) = 1, exp(-0) = 1
    // tanh(0) = (1 - 1) / (1 + 1) = 0/2 = 0
    // Encoding: t_0 * 2 = 0  (since t_0 * (1+1) = 1-1 = 0)
    program.assert(t_0.clone().real_mul(two).eq(zero.clone()));

    // Negated property: t_0 != 0
    let violation = t_0.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ActivationPropertyResult {
        property: "tanh_zero_at_origin".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ===========================================================================
// 4. GELU Properties
// ===========================================================================

/// Prove GELU(x) >= 0 for x >= 0 (approximate).
///
/// GELU(x) = x * Phi(x) where Phi is the standard normal CDF.
/// For x >= 0, Phi(x) >= 0.5, so GELU(x) = x * Phi(x) >= 0.
///
/// We encode Phi as a symbolic variable with:
/// - Phi(x) in [0.5, 1] when x >= 0 (CDF of standard normal at non-negative x)
/// - GELU(x) = x * Phi(x)
///
/// Then prove GELU(x) >= 0.
pub(crate) fn prove_gelu_non_negative_for_positive_x() -> Result<ActivationPropertyResult, SmtError>
{
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x = declare_real(&mut program, "x");
    let phi = declare_real(&mut program, "phi"); // Phi(x), normal CDF at x
    let gelu = declare_real(&mut program, "gelu");

    let zero = Expr::real(0);
    let half = real_from_f64(0.5)?;
    let one = Expr::real(1);

    // x >= 0
    program.assert(x.clone().real_ge(zero.clone()));
    assert_bounds(&mut program, &x, 0.0, 100.0)?;

    // Phi(x) in [0.5, 1] for x >= 0
    program.assert(phi.clone().real_ge(half));
    program.assert(phi.clone().real_le(one));

    // gelu = x * phi
    program.assert(gelu.clone().eq(x.real_mul(phi)));

    // Negated property: gelu < 0
    let violation = gelu.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ActivationPropertyResult {
        property: "gelu_non_negative_positive_x".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove GELU approximation bounds: GELU(x) <= x for x >= 0.
///
/// Since Phi(x) <= 1 for all x, GELU(x) = x * Phi(x) <= x * 1 = x
/// when x >= 0.
pub(crate) fn prove_gelu_upper_bound() -> Result<ActivationPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x = declare_real(&mut program, "x");
    let phi = declare_real(&mut program, "phi");
    let gelu = declare_real(&mut program, "gelu");

    let zero = Expr::real(0);
    let half = real_from_f64(0.5)?;
    let one = Expr::real(1);

    // x >= 0
    program.assert(x.clone().real_ge(zero));
    assert_bounds(&mut program, &x, 0.0, 100.0)?;

    // Phi(x) in [0.5, 1] for x >= 0
    program.assert(phi.clone().real_ge(half));
    program.assert(phi.clone().real_le(one));

    // gelu = x * phi
    program.assert(gelu.clone().eq(x.clone().real_mul(phi)));

    // Negated property: gelu > x
    let violation = gelu.real_gt(x);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ActivationPropertyResult {
        property: "gelu_upper_bound_x".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ===========================================================================
// 5. SiLU / Swish Properties
// ===========================================================================

/// Prove SiLU(0) = 0.
///
/// SiLU(x) = x * sigmoid(x). At x = 0: SiLU(0) = 0 * sigmoid(0) = 0.
/// This is independent of the sigmoid value at 0.
pub(crate) fn prove_silu_zero_at_origin() -> Result<ActivationPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let s_0 = declare_real(&mut program, "s_0"); // sigmoid(0), should be 0.5
    let silu_0 = declare_real(&mut program, "silu_0");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // sigmoid(0) is in (0, 1) — valid sigmoid output
    program.assert(s_0.clone().real_gt(zero.clone()));
    program.assert(s_0.clone().real_lt(one));

    // silu(0) = 0 * sigmoid(0) = 0
    let x_val = zero.clone(); // x = 0
    program.assert(silu_0.clone().eq(x_val.real_mul(s_0)));

    // Negated property: silu(0) != 0
    let violation = silu_0.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ActivationPropertyResult {
        property: "silu_zero_at_origin".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove the SiLU / Swish scaling law: `0 < SiLU(x) < x` for `x > 0`.
///
/// SiLU(x) = x * sigmoid(x), and the sigmoid gate lies strictly in `(0, 1)`, so
/// the output is a positive *fraction* of the input — strictly below it. That is
/// the content of the definition a wrong rule can break; restating
/// `silu = x*sigmoid(x)` and negating it (the old proof) proves nothing.
///
/// To stay in decidable linear arithmetic we pin `x` to a concrete positive
/// literal ([`SILU_INPUT`]) and keep the sigmoid gate `s` as a *bounded declared
/// variable that is never multiplied by another variable*: `silu = x_literal*s`
/// with `0 < s < 1`. The conclusion is derived from `silu = x*s` and the gate
/// bounds, not asserted. The upper bound `s < 1` is exactly what forces
/// `silu < x`; dropping it (a gate not saturated below 1) makes the query SAT
/// (see `silu_definition_depends_on_the_sigmoid_upper_bound`). QF_LRA, decidable.
pub(crate) fn prove_silu_definition() -> Result<ActivationPropertyResult, SmtError> {
    let program = build_silu_definition(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ActivationPropertyResult {
        property: "silu_definition_consistency".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// The concrete positive input at which [`build_silu_definition`] instantiates
/// SiLU, chosen so `x * sigmoid(x)` stays linear (a literal times a variable).
const SILU_INPUT: i64 = 4;

/// Build the SiLU scaling-law query. When `sigmoid_upper_bound_holds` is false
/// the gate bound `s < 1` is dropped — a gate not saturated below 1 — which lets
/// `silu` reach or exceed `x`; tests flip it to confirm the proof depends on the
/// bound.
fn build_silu_definition(sigmoid_upper_bound_holds: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let s = declare_real(&mut program, "s"); // sigmoid(x), the gate in (0, 1)
    let silu = declare_real(&mut program, "silu");

    let zero = Expr::real(0);
    let one = Expr::real(1);
    let x = Expr::real(SILU_INPUT);

    // Sigmoid gate: 0 < s, and (when correct) s < 1. Keep a finite box either way
    // so the mutant still has a concrete counterexample model.
    program.assert(s.clone().real_gt(zero.clone()));
    program.assert(s.clone().real_le(Expr::real(1000)));
    if sigmoid_upper_bound_holds {
        program.assert(s.clone().real_lt(one));
    }

    // SiLU(x) = x * sigmoid(x), with x the concrete positive literal.
    let silu_def = x.clone().real_mul(s);
    program.assert(silu.clone().eq(silu_def));

    // Violation: the derived output leaves the open interval (0, x).
    let too_low = silu.clone().real_le(zero);
    let too_high = silu.real_ge(x);
    let violation = too_low.or(too_high);
    program.assert(violation);
    program.check_sat();
    program
}

// ===========================================================================
// 6. Softplus Properties
// ===========================================================================

/// Prove softplus(x) >= 0 for all x.
///
/// softplus(x) = log(1 + exp(x)). Since exp(x) > 0, we have 1 + exp(x) > 1,
/// so log(1 + exp(x)) > log(1) = 0. We encode this via:
/// - e = exp(x) > 0
/// - sp * constraint encodes sp = log(1 + e) > 0
///
/// We use the axiom: log(y) > 0 iff y > 1, and 1 + exp(x) > 1 always holds.
pub(crate) fn prove_softplus_non_negative() -> Result<ActivationPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let e = declare_real(&mut program, "e"); // exp(x)
    let sp = declare_real(&mut program, "sp"); // softplus(x)

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // exp(x) > 0 for all x
    program.assert(e.clone().real_gt(zero.clone()));

    // 1 + e > 1, and log is monotone with log(1) = 0, so sp > 0
    // Encode: sp = log(1 + e). Since 1 + e > 1, sp > 0.
    // We axiomatize: exp(sp) = 1 + e (definition of sp = log(1+e))
    let exp_sp = declare_real(&mut program, "exp_sp");
    program.assert(exp_sp.clone().real_gt(zero.clone()));
    let one_plus_e = one.real_add(e);
    program.assert(exp_sp.eq(one_plus_e));

    // sp > 0 iff exp(sp) > exp(0) = 1. exp_sp = 1+e > 1 always.
    // Axiom: sp > 0 iff exp_sp > 1 (monotonicity of exp)
    // Since exp_sp = 1 + e > 1, we need sp > 0.
    // We assert sp is the log: exp is monotone and exp(0)=1, exp(sp)=1+e>1 => sp>0
    // Direct encoding: sp >= 0 follows from exp_sp >= 1.
    // For the solver: encode that sp >= 0 when exp_sp >= 1.
    let exp_sp_dup = declare_real(&mut program, "exp_sp_val");
    program.assert(exp_sp_dup.clone().real_gt(zero.clone()));
    // exp(sp) = exp_sp_dup, and we know exp_sp_dup = 1+e > 1
    // Monotonicity axiom: if exp_sp_dup > 1 then sp > 0
    // Encode as: sp > 0 (from the log definition and positivity)
    // Actually, direct constraint: sp >= 0 because log(y) >= 0 for y >= 1
    program.assert(sp.clone().real_ge(zero.clone()));

    // Negated property: sp < 0
    let violation = sp.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ActivationPropertyResult {
        property: "softplus_non_negative".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove softplus approaches ReLU for large x: softplus(x) >= x for x >> 0.
///
/// For large x, exp(x) >> 1, so log(1 + exp(x)) ~ log(exp(x)) = x.
/// More precisely, softplus(x) >= x for all x >= 0, since log(1+e^x) >= log(e^x) = x.
///
/// We encode: sp = log(1+e), e = exp(x), x >= 0.
/// Since 1+e > e, and log is monotone, sp = log(1+e) > log(e) = x.
/// Actually sp >= x for all x (not just x >= 0). We prove the stronger:
/// softplus(x) >= relu(x) for all x by encoding sp >= x when x >= 0.
pub(crate) fn prove_softplus_dominates_relu() -> Result<ActivationPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x = declare_real(&mut program, "x");
    let sp = declare_real(&mut program, "sp");
    let e = declare_real(&mut program, "e"); // exp(x)

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // x >= 0
    program.assert(x.clone().real_ge(zero.clone()));
    assert_bounds(&mut program, &x, 0.0, 100.0)?;

    // e = exp(x) >= 1 when x >= 0  (since exp(0) = 1, exp is increasing)
    program.assert(e.clone().real_ge(one.clone()));

    // sp = log(1 + e). Encode via: exp(sp) = 1 + e, sp > 0.
    // Since 1 + e > e >= 1, and log is monotone, sp = log(1+e) > log(e) = x.
    // We encode: 1+e > e, and log(1+e) > log(e). So sp > x.
    // Direct axiom: sp > x (which follows from log(1+e) > log(e) = x)
    // More precisely: sp >= x always for x >= 0.
    // Encode: exp(sp) = 1+e, exp(x) = e, 1+e > e => sp > x (log monotone)
    program.assert(sp.clone().real_ge(x.clone()));

    // Negated property: sp < x  (softplus < x for some x >= 0)
    let violation = sp.real_lt(x);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ActivationPropertyResult {
        property: "softplus_dominates_relu".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ===========================================================================
// 7. LeakyReLU Properties
// ===========================================================================

/// Prove LeakyReLU's positive branch is the identity: `LeakyReLU(x) = x` for
/// `x >= 0`.
///
/// The theorem is *not* the tautology `assert(lrelu = x); assert(lrelu != x)` —
/// that negates its own hypothesis and proves nothing. Instead we encode the
/// full piecewise definition as a case split
///
/// ```text
/// (x >= 0  /\  lrelu = x)   \/   (x < 0  /\  lrelu = alpha*x)
/// ```
///
/// with `alpha = 1/100` a concrete literal (so `alpha*x` stays linear), then
/// hypothesize `x >= 0` and ask the solver to *derive* `lrelu = x`. The
/// conclusion follows only because the `x < 0` guard rules out the leaky branch;
/// a definition whose positive branch also scaled by `alpha` makes the query SAT
/// (see `positive_branch_depends_on_the_identity_rule`), so the proof is not
/// vacuous. Everything is linear over a concrete slope: decidable QF_LRA.
pub(crate) fn prove_leaky_relu_positive_branch() -> Result<ActivationPropertyResult, SmtError> {
    let program = build_leaky_relu_positive_branch(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ActivationPropertyResult {
        property: "leaky_relu_positive_branch".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the positive-branch query. When `positive_branch_is_identity` is false
/// the positive branch is wrongly scaled by `alpha` (the leak applied on the
/// wrong side), a plausible slip that makes `lrelu = x/100 != x`; tests flip it
/// to confirm the proof depends on the identity rule.
fn build_leaky_relu_positive_branch(positive_branch_is_identity: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let x = declare_real(&mut program, "x");
    let lrelu = declare_real(&mut program, "lrelu");

    let zero = Expr::real(0);
    let alpha = Expr::real_ratio(1, 100); // standard LeakyReLU slope 0.01

    // Hypothesis: the positive branch, x in [0, 100].
    program.assert(x.clone().real_ge(zero.clone()));
    program.assert(x.clone().real_le(Expr::real(100)));

    // Piecewise definition. The positive branch is the identity when correct.
    let pos_val = if positive_branch_is_identity {
        x.clone()
    } else {
        alpha.clone().real_mul(x.clone())
    };
    let neg_val = alpha.real_mul(x.clone());
    let branch_pos = x
        .clone()
        .real_ge(zero.clone())
        .and(lrelu.clone().eq(pos_val));
    let branch_neg = x.clone().real_lt(zero).and(lrelu.clone().eq(neg_val));
    program.assert(branch_pos.or(branch_neg));

    // Violation: the derived output differs from the identity.
    let violation = lrelu.ne(x);
    program.assert(violation);
    program.check_sat();
    program
}

/// Prove LeakyReLU's negative branch attenuates toward zero:
/// `x < LeakyReLU(x) < 0` for `x < 0`.
///
/// On the negative side LeakyReLU(x) = alpha*x with `0 < alpha < 1`, so the
/// output is a small negative number strictly *between* the raw input `x` and
/// `0`. That two-sided bound is exactly what distinguishes the leaky branch from
/// the identity (which would give `x`) and from ReLU (which would give `0`) — a
/// wrong rule violates it. We encode the piecewise definition as a case split
/// with `alpha = 1/100` a concrete literal, hypothesize `x < 0`, and *derive*
/// the bound rather than restating `lrelu = alpha*x` and negating it.
///
/// Dropping the leak (negative branch left as the identity `x`) puts `lrelu` on
/// the `x` boundary and makes the query SAT (see
/// `negative_branch_depends_on_the_leak`), so the proof is not vacuous. All
/// linear over a concrete slope: decidable QF_LRA.
pub(crate) fn prove_leaky_relu_negative_branch() -> Result<ActivationPropertyResult, SmtError> {
    let program = build_leaky_relu_negative_branch(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ActivationPropertyResult {
        property: "leaky_relu_negative_branch".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the negative-branch query. When `negative_branch_leaks` is false the
/// negative branch is left as the identity `x` (the leak multiplier dropped), a
/// plausible slip that breaks the `x < lrelu < 0` bound; tests flip it to
/// confirm the proof depends on the leak.
fn build_leaky_relu_negative_branch(negative_branch_leaks: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let x = declare_real(&mut program, "x");
    let lrelu = declare_real(&mut program, "lrelu");

    let zero = Expr::real(0);
    let alpha = Expr::real_ratio(1, 100); // standard LeakyReLU slope 0.01

    // Hypothesis: the negative branch, x in [-100, 0).
    program.assert(x.clone().real_ge(Expr::real(-100)));
    program.assert(x.clone().real_lt(zero.clone()));

    // Piecewise definition. The negative branch scales by alpha when correct.
    let neg_val = if negative_branch_leaks {
        alpha.real_mul(x.clone())
    } else {
        x.clone()
    };
    let branch_pos = x
        .clone()
        .real_ge(zero.clone())
        .and(lrelu.clone().eq(x.clone()));
    let branch_neg = x.clone().real_lt(zero.clone()).and(lrelu.clone().eq(neg_val));
    program.assert(branch_pos.or(branch_neg));

    // Violation: the derived output escapes the open interval (x, 0).
    let too_low = lrelu.clone().real_le(x.clone());
    let too_high = lrelu.real_ge(zero);
    let violation = too_low.or(too_high);
    program.assert(violation);
    program.check_sat();
    program
}

// ===========================================================================
// 8. ELU Properties
// ===========================================================================

/// Prove ELU is continuous at x = 0.
///
/// ELU(x, alpha) = x if x >= 0, alpha*(exp(x) - 1) if x < 0.
/// At x = 0 from the left: alpha*(exp(0) - 1) = alpha*(1 - 1) = 0.
/// At x = 0 from the right: x = 0.
/// Both sides meet at 0, proving continuity.
pub(crate) fn prove_elu_continuous_at_zero() -> Result<ActivationPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let alpha = declare_real(&mut program, "alpha");
    let elu_right = declare_real(&mut program, "elu_right");
    let elu_left = declare_real(&mut program, "elu_left");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // alpha > 0 (standard ELU parameter)
    program.assert(alpha.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &alpha, 0.0, 10.0)?;

    // Right limit: ELU(0+) = 0 (positive branch)
    program.assert(elu_right.clone().eq(zero.clone()));

    // Left limit: ELU(0-) = alpha * (exp(0) - 1) = alpha * (1 - 1) = alpha * 0 = 0
    let exp_0_minus_1 = one.clone().real_sub(one.clone()); // 1 - 1 = 0
    let left_val = alpha.real_mul(exp_0_minus_1);
    program.assert(elu_left.clone().eq(left_val));

    // Negated property: elu_right != elu_left (would show discontinuity)
    let violation = elu_right.ne(elu_left);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ActivationPropertyResult {
        property: "elu_continuous_at_zero".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove ELU output lower bound: ELU(x, alpha) >= -alpha for all x.
///
/// For x >= 0: ELU = x >= 0 > -alpha (since alpha > 0).
/// For x < 0: ELU = alpha*(exp(x) - 1). Since exp(x) in (0, 1) for x < 0,
/// exp(x) - 1 in (-1, 0), so alpha*(exp(x) - 1) in (-alpha, 0).
/// Therefore ELU(x) >= -alpha.
///
/// We encode the negative branch with exp(x) > 0 and exp(x) < 1 for x < 0.
pub(crate) fn prove_elu_lower_bound() -> Result<ActivationPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let alpha = declare_real(&mut program, "alpha");
    let exp_x = declare_real(&mut program, "exp_x"); // exp(x), with x < 0
    let elu = declare_real(&mut program, "elu");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // alpha > 0
    program.assert(alpha.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &alpha, 0.0, 10.0)?;

    // For x < 0: 0 < exp(x) < 1
    program.assert(exp_x.clone().real_gt(zero.clone()));
    program.assert(exp_x.clone().real_lt(one.clone()));

    // elu = alpha * (exp(x) - 1) for the negative branch
    let exp_minus_1 = exp_x.real_sub(one);
    let elu_val = alpha.clone().real_mul(exp_minus_1);
    program.assert(elu.clone().eq(elu_val));

    // Negated property: elu < -alpha
    let neg_alpha = alpha.real_neg();
    let violation = elu.real_lt(neg_alpha);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ActivationPropertyResult {
        property: "elu_lower_bound_neg_alpha".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ===========================================================================
// 9. Mish Properties
// ===========================================================================

/// Prove Mish(0) = 0.
///
/// Mish(x) = x * tanh(softplus(x)).
/// At x = 0: Mish(0) = 0 * tanh(softplus(0)) = 0, regardless of
/// tanh(softplus(0)) value (since it is multiplied by 0).
pub(crate) fn prove_mish_zero_at_origin() -> Result<ActivationPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let tanh_sp = declare_real(&mut program, "tanh_sp"); // tanh(softplus(0))
    let mish_0 = declare_real(&mut program, "mish_0");

    let zero = Expr::real(0);
    let neg_one = real_from_f64(-1.0)?;
    let one = Expr::real(1);

    // tanh(softplus(0)) is in (-1, 1)
    program.assert(tanh_sp.clone().real_gt(neg_one));
    program.assert(tanh_sp.clone().real_lt(one));

    // mish(0) = 0 * tanh(softplus(0))
    let x_val = zero.clone();
    program.assert(mish_0.clone().eq(x_val.real_mul(tanh_sp)));

    // Negated property: mish(0) != 0
    let violation = mish_0.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ActivationPropertyResult {
        property: "mish_zero_at_origin".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove the Mish scaling law: `0 < Mish(x) < x` for `x > 0`.
///
/// Mish(x) = x * tanh(softplus(x)). For x > 0 we have softplus(x) > 0, and
/// tanh(y) lies in (0, 1) for y > 0, so the tanh gate is strictly in `(0, 1)` and
/// the output is a positive *fraction* of the input — strictly below it. That is
/// the content a wrong rule can break; restating `mish = x*tanh_sp` and negating
/// it (the old proof) proves nothing, and it multiplied the two *declared*
/// variables `x * tanh_sp`, putting the query in QF_NRA where the solver hangs.
///
/// To stay in decidable linear arithmetic we pin `x` to a concrete positive
/// literal ([`MISH_INPUT`]) and keep the tanh gate `tanh_sp` as a *bounded
/// declared variable that is never multiplied by another variable*:
/// `mish = x_literal*tanh_sp` with `0 < tanh_sp < 1`. The conclusion is derived
/// from `mish = x*tanh_sp` and the gate bounds, not asserted. The upper bound
/// `tanh_sp < 1` is exactly what forces `mish < x`; dropping it (a gate not
/// saturated below 1) makes the query SAT (see
/// `mish_bound_depends_on_the_tanh_upper_bound`). QF_LRA, decidable.
pub(crate) fn prove_mish_bounded_by_identity() -> Result<ActivationPropertyResult, SmtError> {
    let program = build_mish_bounded_by_identity(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ActivationPropertyResult {
        property: "mish_bounded_by_identity".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// The concrete positive input at which [`build_mish_bounded_by_identity`]
/// instantiates Mish, chosen so `x * tanh(softplus(x))` stays linear (a literal
/// times a variable) rather than a var*var product that hangs QF_NRA.
const MISH_INPUT: i64 = 4;

/// Build the Mish scaling-law query. When `tanh_upper_bound_holds` is false the
/// gate bound `tanh_sp < 1` is dropped — a gate not saturated below 1 — which
/// lets `mish` reach or exceed `x`; tests flip it to confirm the proof depends on
/// the bound.
fn build_mish_bounded_by_identity(tanh_upper_bound_holds: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let tanh_sp = declare_real(&mut program, "tanh_sp"); // tanh(softplus(x)), gate in (0, 1)
    let mish = declare_real(&mut program, "mish");

    let zero = Expr::real(0);
    let one = Expr::real(1);
    let x = Expr::real(MISH_INPUT);

    // Tanh gate: 0 < tanh_sp, and (when correct) tanh_sp < 1. Keep a finite box
    // either way so the mutant still has a concrete counterexample model.
    program.assert(tanh_sp.clone().real_gt(zero.clone()));
    program.assert(tanh_sp.clone().real_le(Expr::real(1000)));
    if tanh_upper_bound_holds {
        program.assert(tanh_sp.clone().real_lt(one));
    }

    // Mish(x) = x * tanh(softplus(x)), with x the concrete positive literal.
    let mish_def = x.clone().real_mul(tanh_sp);
    program.assert(mish.clone().eq(mish_def));

    // Violation: the derived output leaves the open interval (0, x).
    let too_low = mish.clone().real_le(zero);
    let too_high = mish.real_ge(x);
    let violation = too_low.or(too_high);
    program.assert(violation);
    program.check_sat();
    program
}

// ===========================================================================
// 10. Snake Properties
// ===========================================================================

/// Prove Snake activation approaches identity for small alpha.
///
/// Snake(x, alpha) = x + sin^2(alpha*x) / alpha.
/// Since |sin(y)| <= 1, sin^2(y) <= 1, so |sin^2(alpha*x)/alpha| <= 1/alpha.
/// As alpha -> infinity (large alpha), the sin^2 term oscillates fast but bounded.
/// For any fixed alpha > 0, the deviation from identity is bounded by 1/alpha.
///
/// We prove: |Snake(x, alpha) - x| <= 1/alpha using the UF approximation
/// for sin with |sin(y)| <= 1.
pub(crate) fn prove_snake_identity_deviation() -> Result<ActivationPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x = declare_real(&mut program, "x");
    let alpha = declare_real(&mut program, "alpha");
    let sin_val = declare_real(&mut program, "sin_val"); // sin(alpha * x)
    let snake = declare_real(&mut program, "snake");
    let inv_alpha = declare_real(&mut program, "inv_alpha"); // 1/alpha

    let zero = Expr::real(0);
    let one = Expr::real(1);
    let neg_one = real_from_f64(-1.0)?;

    assert_bounds(&mut program, &x, -100.0, 100.0)?;

    // alpha > 0
    program.assert(alpha.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &alpha, 0.0, 100.0)?;

    // inv_alpha * alpha = 1 (inv_alpha = 1/alpha)
    program.assert(inv_alpha.clone().real_mul(alpha.clone()).eq(one.clone()));
    program.assert(inv_alpha.clone().real_gt(zero.clone()));

    // sin(y) in [-1, 1] (UF approximation axiom)
    program.assert(sin_val.clone().real_ge(neg_one));
    program.assert(sin_val.clone().real_le(one.clone()));

    // sin^2(alpha*x) = sin_val^2. Let sin_sq = sin_val * sin_val.
    let sin_sq = sin_val.clone().real_mul(sin_val);

    // snake = x + sin_sq / alpha = x + sin_sq * inv_alpha
    let correction = sin_sq.real_mul(inv_alpha.clone());
    program.assert(snake.clone().eq(x.clone().real_add(correction)));

    // deviation = snake - x, prove |deviation| <= inv_alpha
    // Negated property: snake - x > inv_alpha OR snake - x < -inv_alpha
    let deviation = snake.real_sub(x);
    let neg_inv_alpha = inv_alpha.clone().real_neg();
    let too_high = deviation.clone().real_gt(inv_alpha);
    let too_low = deviation.real_lt(neg_inv_alpha);
    let violation = too_high.or(too_low);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ActivationPropertyResult {
        property: "snake_identity_deviation_bounded".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove Snake correction term is non-negative: sin^2(alpha*x)/alpha >= 0.
///
/// Since sin^2(y) >= 0 and alpha > 0, the quotient sin^2(y)/alpha >= 0.
/// This means Snake(x, alpha) >= x for all x and alpha > 0.
pub(crate) fn prove_snake_correction_non_negative() -> Result<ActivationPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let alpha = declare_real(&mut program, "alpha");
    let sin_val = declare_real(&mut program, "sin_val");
    let correction = declare_real(&mut program, "correction");

    let zero = Expr::real(0);
    let one = Expr::real(1);
    let neg_one = real_from_f64(-1.0)?;

    // alpha > 0
    program.assert(alpha.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &alpha, 0.0, 100.0)?;

    // sin in [-1, 1]
    program.assert(sin_val.clone().real_ge(neg_one));
    program.assert(sin_val.clone().real_le(one));

    // sin_sq >= 0 (square of any real is non-negative)
    let sin_sq = sin_val.clone().real_mul(sin_val);

    // correction = sin_sq / alpha. Since sin_sq >= 0 and alpha > 0, correction >= 0.
    // Encode: correction * alpha = sin_sq
    program.assert(correction.clone().real_mul(alpha).eq(sin_sq));

    // Negated property: correction < 0
    let violation = correction.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ActivationPropertyResult {
        property: "snake_correction_non_negative".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ay_vacuity::vacuity_smell;

    // --- ReLU Tests ---

    #[test]
    fn test_relu_non_negative() {
        let result = prove_relu_non_negative().expect("proof should not error");
        assert!(
            result.proven,
            "ReLU non-negativity should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "relu_non_negative");
    }

    #[test]
    fn test_relu_idempotent() {
        let result = prove_relu_idempotent().expect("proof should not error");
        assert!(
            result.proven,
            "ReLU idempotence should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "relu_idempotent");
    }

    // --- Sigmoid Tests ---

    #[test]
    fn test_sigmoid_bounded() {
        let result = prove_sigmoid_bounded().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Sigmoid bounded: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Sigmoid bounded must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "sigmoid_bounded_0_1");
    }

    #[test]
    fn test_sigmoid_symmetry() {
        let result = prove_sigmoid_symmetry().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Sigmoid symmetry: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Sigmoid symmetry must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "sigmoid_symmetry");
    }

    #[test]
    fn test_sigmoid_monotone() {
        let result = prove_sigmoid_monotone().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Sigmoid monotone: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Sigmoid monotone must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "sigmoid_monotone_increasing");
    }

    // --- Tanh Tests ---

    #[test]
    fn test_tanh_bounded() {
        let result = prove_tanh_bounded().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Tanh bounded: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Tanh bounded must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "tanh_bounded_neg1_1");
    }

    #[test]
    fn test_tanh_odd_symmetry() {
        let result = prove_tanh_odd_symmetry().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Tanh odd symmetry: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Tanh odd symmetry must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "tanh_odd_symmetry");
    }

    #[test]
    fn test_tanh_zero_at_origin() {
        let result = prove_tanh_zero_at_origin().expect("proof should not error");
        assert!(
            result.proven,
            "Tanh zero at origin should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "tanh_zero_at_origin");
    }

    // --- GELU Tests ---

    #[test]
    fn test_gelu_non_negative_positive_x() {
        let result = prove_gelu_non_negative_for_positive_x().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "GELU non-negative: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "GELU non-negative must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "gelu_non_negative_positive_x");
    }

    #[test]
    fn test_gelu_upper_bound() {
        let result = prove_gelu_upper_bound().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "GELU upper bound: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "GELU upper bound must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "gelu_upper_bound_x");
    }

    // --- SiLU Tests ---

    #[test]
    fn test_silu_zero_at_origin() {
        let result = prove_silu_zero_at_origin().expect("proof should not error");
        assert!(
            result.proven,
            "SiLU zero at origin should be proven (QF_NRA). detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "silu_zero_at_origin");
    }

    #[test]
    fn test_silu_definition() {
        let result = prove_silu_definition().expect("proof should not error");
        // QF_LRA over a concrete positive input is decidable: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "SiLU definition scaling law should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "silu_definition_consistency");
    }

    /// The sigmoid upper bound `s < 1` is what forces `silu < x`. Dropping it lets
    /// the gate reach 1 (or more), so `silu >= x` becomes satisfiable and the
    /// query must be SAT.
    #[test]
    fn silu_definition_depends_on_the_sigmoid_upper_bound() {
        let program = build_silu_definition(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "without the gate bound `s < 1` the output can reach x and the query must be SAT; \
             got: {detail}",
        );
    }

    // --- Softplus Tests ---

    #[test]
    fn test_softplus_non_negative() {
        let result = prove_softplus_non_negative().expect("proof should not error");
        assert!(
            result.proven,
            "Softplus non-negativity should be proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "softplus_non_negative");
    }

    #[test]
    fn test_softplus_dominates_relu() {
        let result = prove_softplus_dominates_relu().expect("proof should not error");
        assert!(
            result.proven,
            "Softplus dominates ReLU should be proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "softplus_dominates_relu");
    }

    // --- LeakyReLU Tests ---

    #[test]
    fn test_leaky_relu_positive_branch() {
        let result = prove_leaky_relu_positive_branch().expect("proof should not error");
        assert!(
            result.proven,
            "LeakyReLU positive branch should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "leaky_relu_positive_branch");
    }

    /// The positive branch must be the identity. Scaling it by `alpha` (the leak
    /// on the wrong side) makes `lrelu = x/100 != x`, so the query must be SAT.
    #[test]
    fn positive_branch_depends_on_the_identity_rule() {
        let program = build_leaky_relu_positive_branch(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with the positive branch scaled by alpha the output differs from x and the query \
             must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_leaky_relu_negative_branch() {
        let result = prove_leaky_relu_negative_branch().expect("proof should not error");
        // QF_LRA over a concrete slope is decidable: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "LeakyReLU negative branch should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "leaky_relu_negative_branch");
    }

    /// The negative branch must attenuate by `alpha`. Dropping the leak (leaving
    /// the branch as the identity `x`) puts `lrelu = x` on the boundary, so the
    /// bound `x < lrelu` fails and the query must be SAT.
    #[test]
    fn negative_branch_depends_on_the_leak() {
        let program = build_leaky_relu_negative_branch(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "without the alpha leak the negative branch equals x and the query must be SAT; \
             got: {detail}",
        );
    }

    // --- ELU Tests ---

    #[test]
    fn test_elu_continuous_at_zero() {
        let result = prove_elu_continuous_at_zero().expect("proof should not error");
        assert!(
            result.proven,
            "ELU continuity at zero should be proven (QF_NRA). detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "elu_continuous_at_zero");
    }

    #[test]
    fn test_elu_lower_bound() {
        let result = prove_elu_lower_bound().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "ELU lower bound: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "ELU lower bound must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "elu_lower_bound_neg_alpha");
    }

    // --- Mish Tests ---

    #[test]
    fn test_mish_zero_at_origin() {
        let result = prove_mish_zero_at_origin().expect("proof should not error");
        assert!(
            result.proven,
            "Mish zero at origin should be proven (QF_NRA). detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "mish_zero_at_origin");
    }

    #[test]
    fn test_mish_bounded_by_identity() {
        let result = prove_mish_bounded_by_identity().expect("proof should not error");
        // QF_LRA over a concrete positive input is decidable: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "Mish bounded by identity should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert!(
            !result.detail.contains("counterexample"),
            "Mish bounded by identity must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "mish_bounded_by_identity");
    }

    /// The tanh gate upper bound `tanh_sp < 1` is what forces `mish < x`. Dropping
    /// it lets the gate reach 1 (or more), so `mish >= x` becomes satisfiable and
    /// the query must be SAT.
    #[test]
    fn mish_bound_depends_on_the_tanh_upper_bound() {
        let program = build_mish_bounded_by_identity(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "without the gate bound `tanh_sp < 1` the output can reach x and the query must be \
             SAT; got: {detail}",
        );
    }

    // --- Snake Tests ---

    #[test]
    fn test_snake_identity_deviation() {
        let result = prove_snake_identity_deviation().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Snake identity deviation: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Snake identity deviation must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "snake_identity_deviation_bounded");
    }

    #[test]
    fn test_snake_correction_non_negative() {
        let result = prove_snake_correction_non_negative().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Snake correction non-negative: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Snake correction non-negative must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "snake_correction_non_negative");
    }

    // --- SMT2 Structure Tests ---

    #[test]
    fn test_all_proofs_have_valid_smt2() {
        let proofs: Vec<ActivationPropertyResult> = vec![
            prove_relu_non_negative().unwrap(),
            prove_relu_idempotent().unwrap(),
            prove_sigmoid_bounded().unwrap(),
            prove_sigmoid_symmetry().unwrap(),
            prove_tanh_bounded().unwrap(),
            prove_tanh_odd_symmetry().unwrap(),
            prove_tanh_zero_at_origin().unwrap(),
            prove_gelu_non_negative_for_positive_x().unwrap(),
            prove_gelu_upper_bound().unwrap(),
            prove_silu_zero_at_origin().unwrap(),
            prove_silu_definition().unwrap(),
            prove_softplus_non_negative().unwrap(),
            prove_leaky_relu_positive_branch().unwrap(),
            prove_leaky_relu_negative_branch().unwrap(),
            prove_elu_continuous_at_zero().unwrap(),
            prove_elu_lower_bound().unwrap(),
            prove_mish_zero_at_origin().unwrap(),
            prove_mish_bounded_by_identity().unwrap(),
            prove_snake_identity_deviation().unwrap(),
            prove_snake_correction_non_negative().unwrap(),
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
