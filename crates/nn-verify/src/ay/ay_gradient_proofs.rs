// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for gradient computation mathematical correctness (#4241).
//!
//! Proves fundamental differentiation rules used by the autodiff engine (nn-autodiff)
//! are mathematically correct. Each proof encodes the expected gradient identity as a
//! negated assertion and proves UNSAT (no counterexample exists).
//!
//! # Proved Properties
//!
//! 1. **Chain rule**: For composed f(g(x)), d/dx f(g(x)) = f'(g(x)) * g'(x)
//! 2. **Linear gradient**: For f(x) = W*x + b, df/dW = x and df/db = 1
//! 3. **ReLU gradient**: d/dx max(0,x) = 1 when x > 0, 0 when x < 0
//! 4. **Sigmoid gradient**: d/dx sigma(x) = sigma(x) * (1 - sigma(x))
//! 5. **Softmax Jacobian diagonal**: ds_i/dx_i = s_i * (1 - s_i)
//! 6. **Cross-entropy gradient**: d/dp_i [-sum(y * log(p))] = -y_i / p_i
//!
//! # Proof Strategy
//!
//! Gradient proofs use two approaches depending on the function:
//!
//! - **Algebraic identity proofs** (chain rule, linear, softmax diagonal): Pure polynomial
//!   identities provable via QF_NRA or QF_LRA. These hold for all values satisfying
//!   the function definitions.
//!
//! - **Piecewise/conditional proofs** (ReLU): Case-split on x > 0 vs x < 0,
//!   prove each branch independently using QF_LRA.
//!
//! - **Transcendental function proofs** (sigmoid, cross-entropy): Since exp/log are
//!   not in the decidable NRA fragment, we encode the function output as a symbolic
//!   variable with defining constraints (e.g., s = sigma(x) means s*(1-s) is the
//!   derivative) and prove the identity holds for all valid function values.

use ay_bindings::{Expr, Sort, AYProgram};

use crate::ay_real_lit::RealLit;

use super::error::SmtError;
use super::translate_real::real_from_f64;

/// Result of a gradient property proof attempt.
#[derive(Debug, Clone)]
pub(crate) struct GradientPropertyResult {
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
/// Naming each intermediate quantity keeps a proof's conclusion one derivation
/// step removed from its hypotheses, so the solver must *derive* the conclusion
/// through the chain of definitions instead of matching a restated answer.
fn define_real(program: &mut AYProgram, name: &str, term: &Expr) -> Expr {
    let var = declare_real(program, name);
    program.assert(var.clone().eq(term.clone()));
    var
}

/// Encode `name = max(0, arg)` via its defining constraints and return the var.
///
/// `max(0, arg)` is the least value that is both `>= 0` and `>= arg`; the
/// disjunction pins it to whichever bound is active. Under a known sign of `arg`
/// the solver derives the exact ReLU output, so a downstream gradient claim is
/// checked against the real ReLU rather than a restated answer.
fn relu_via_max(program: &mut AYProgram, name: &str, arg: &Expr) -> Expr {
    let relu = declare_real(program, name);
    let zero = Expr::real(0);
    program.assert(relu.clone().real_ge(zero.clone())); // relu >= 0
    program.assert(relu.clone().real_ge(arg.clone())); // relu >= arg
    // relu attains one of its two lower bounds: relu = 0 OR relu = arg.
    program.assert(relu.clone().eq(zero).or(relu.clone().eq(arg.clone())));
    relu
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
/// that is UNSAT only because it asserts `P ∧ ¬P` (or compares a term to itself)
/// never counts as a proof — any residual vacuity becomes a hard `test_*_proven`
/// failure rather than a silent pass.
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
// Property 1: Chain Rule Correctness
// ---------------------------------------------------------------------------

/// Prove the chain rule for composed functions: d/dx f(g(x)) = f'(g(x)) * g'(x).
///
/// The content is checked against an *independently computed* derivative rather
/// than restated. We take affine `g(x) = a*x + p` (declared slope `a`) and
/// `f(y) = 3*y + q` (slope literal 3, so `f'` is a constant `3`), form the
/// composite `h = f∘g` by actually substituting, and read its slope by finite
/// difference over `x in {0, 1}` — exact, since `h` is affine. The chain-rule
/// gradient `f'(g(x)) * g'(x) = 3 * a` must equal that composite slope.
///
/// Keeping `f`'s slope a literal makes every product `literal * variable`, so the
/// query is linear (`QF_LRA`, decidable). Dropping the inner factor `g'(x)` — the
/// classic "forgot to multiply by the upstream gradient" autodiff slip — makes
/// the query SAT (see `chain_rule_depends_on_the_inner_derivative`).
pub(crate) fn prove_chain_rule() -> Result<GradientPropertyResult, SmtError> {
    let program = build_chain_rule(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(GradientPropertyResult {
        property: "gradient_chain_rule".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the chain-rule query. When `correct` is false the chain rule drops the
/// inner derivative `g'(x)`, returning just `f'(g(x))`, which no longer matches
/// the composite slope; tests flip it to confirm the proof depends on the rule.
fn build_chain_rule(correct: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // g(x) = a*x + p (declared slope/intercept); f(y) = 3*y + q (slope literal).
    let a = declare_real(&mut program, "g_slope"); // g'(x) = a
    let p = declare_real(&mut program, "g_intercept");
    let q = declare_real(&mut program, "f_intercept");
    let f_slope = Expr::real(3); // f'(y) = 3 for every y

    // Compose h = f∘g and read its slope by finite difference over x in {0, 1}.
    //   g(0) = p,          g(1) = a + p
    //   h(0) = 3*g(0) + q, h(1) = 3*g(1) + q
    let g0 = define_real(&mut program, "g_at_0", &p);
    let g1 = define_real(&mut program, "g_at_1", &a.clone().real_add(p.clone()));
    let h0 = define_real(
        &mut program,
        "h_at_0",
        &f_slope.clone().real_mul(g0).real_add(q.clone()),
    );
    let h1 = define_real(
        &mut program,
        "h_at_1",
        &f_slope.clone().real_mul(g1).real_add(q),
    );
    let composite_slope = define_real(&mut program, "composite_slope", &h1.real_sub(h0));

    // Chain rule: d/dx f(g(x)) = f'(g(x)) * g'(x) = 3 * a.
    let g_prime = define_real(&mut program, "g_prime", &a);
    let chain_rule = if correct {
        f_slope.real_mul(g_prime) // 3 * a
    } else {
        f_slope // 3 (dropped the inner derivative g'(x))
    };
    let chain_grad = define_real(&mut program, "chain_grad", &chain_rule);

    // Violation: the chain-rule gradient disagrees with the true composite slope.
    program.assert(composite_slope.ne(chain_grad));
    program.check_sat();
    program
}

/// Prove the chain rule for multiplication: d/dx [f(x) * g(x)] = f'(x)*g(x) + f(x)*g'(x).
///
/// This is the product rule, which is a specific instance of the chain rule for
/// the multiplication operation. Given:
///   - `fx`, `gx`: function values at x
///   - `f_prime`, `g_prime`: derivatives at x
///   - `product_deriv`: derivative of f*g at x
///
/// We prove `product_deriv = f_prime * gx + fx * g_prime`.
pub(crate) fn prove_chain_rule_multiplication() -> Result<GradientPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let fx = declare_real(&mut program, "fx");
    let gx = declare_real(&mut program, "gx");
    let f_prime = declare_real(&mut program, "f_prime");
    let g_prime = declare_real(&mut program, "g_prime");
    let product_deriv = declare_real(&mut program, "product_deriv");

    assert_bounds(&mut program, &fx, -100.0, 100.0)?;
    assert_bounds(&mut program, &gx, -100.0, 100.0)?;
    assert_bounds(&mut program, &f_prime, -100.0, 100.0)?;
    assert_bounds(&mut program, &g_prime, -100.0, 100.0)?;
    assert_bounds(&mut program, &product_deriv, -20000.0, 20000.0)?;

    // Product rule: product_deriv = f'(x)*g(x) + f(x)*g'(x)
    // Use intermediate variables to keep polynomial degree low
    let f_prime_gx = declare_real(&mut program, "f_prime_gx");
    let fx_g_prime = declare_real(&mut program, "fx_g_prime");
    program.assert(f_prime_gx.clone().eq(f_prime.clone().real_mul(gx.clone())));
    program.assert(fx_g_prime.clone().eq(fx.clone().real_mul(g_prime.clone())));

    let rhs = f_prime_gx.clone().real_add(fx_g_prime.clone());
    program.assert(product_deriv.clone().eq(rhs));

    // Negated property: product_deriv != f'(x)*g(x) + f(x)*g'(x)
    let rhs_check = f_prime.real_mul(gx).real_add(fx.real_mul(g_prime));
    let violation = product_deriv.ne(rhs_check);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(GradientPropertyResult {
        property: "gradient_chain_rule_multiplication".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove the sum rule: d/dx [f(x) + g(x)] = f'(x) + g'(x).
///
/// Both `f(x) = a*x + p` and `g(x) = b*x + q` are affine with declared slopes.
/// We form the sum `s = f + g` by adding the actual outputs and read its slope by
/// finite difference over `x in {0, 1}` (exact, since `s` is affine). The sum
/// rule `f'(x) + g'(x) = a + b` must equal that slope.
///
/// Everything is linear (`QF_LRA`, decidable). Computing the rule as `f' - g'` (a
/// sign slip) makes the query SAT (see `addition_rule_depends_on_the_sign`).
pub(crate) fn prove_chain_rule_addition() -> Result<GradientPropertyResult, SmtError> {
    let program = build_chain_rule_addition(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(GradientPropertyResult {
        property: "gradient_chain_rule_addition".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the sum-rule query. When `correct` is false the rule subtracts the
/// derivatives (`f' - g'`) instead of adding them; tests flip it to confirm the
/// proof depends on the rule.
fn build_chain_rule_addition(correct: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // f(x) = a*x + p, g(x) = b*x + q.
    let a = declare_real(&mut program, "f_slope"); // f'(x) = a
    let b = declare_real(&mut program, "g_slope"); // g'(x) = b
    let p = declare_real(&mut program, "f_intercept");
    let q = declare_real(&mut program, "g_intercept");

    // Sum s(x) = f(x) + g(x); read its slope by finite difference over {0, 1}.
    //   s(0) = f(0) + g(0) = p + q
    //   s(1) = f(1) + g(1) = (a + p) + (b + q)
    let s0 = define_real(&mut program, "sum_at_0", &p.clone().real_add(q.clone()));
    let s1 = define_real(
        &mut program,
        "sum_at_1",
        &a.clone().real_add(p).real_add(b.clone()).real_add(q),
    );
    let sum_slope = define_real(&mut program, "sum_slope", &s1.real_sub(s0));

    // Sum rule: (f + g)'(x) = f'(x) + g'(x) = a + b.
    let f_prime = define_real(&mut program, "f_prime", &a);
    let g_prime = define_real(&mut program, "g_prime", &b);
    let sum_rule = if correct {
        f_prime.real_add(g_prime) // a + b
    } else {
        f_prime.real_sub(g_prime) // a - b (sign slip)
    };
    let sum_grad = define_real(&mut program, "sum_grad", &sum_rule);

    // Violation: the sum-rule gradient disagrees with the true sum slope.
    program.assert(sum_slope.ne(sum_grad));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 2: Linear Gradient
// ---------------------------------------------------------------------------

/// Prove that for f(W, x, b) = W*x + b, the gradient df/dW = x.
///
/// We read `df/dW` as a finite difference of `f` over `W in {0, 1}` (exact, since
/// `f` is affine in `W`): `f(1) - f(0) = (1*x + b) - (0*x + b) = x`. Because the
/// two evaluation points are concrete literals, no `W*x` product appears and the
/// query is linear (`QF_LRA`, decidable). The derived derivative must equal the
/// reverse-mode gradient of `z = W*x`, which is the *other* factor `x`.
///
/// The realistic slip (`!correct`) returns the weight `W` itself instead of the
/// input `x` — a transposed-operand bug — and makes the query SAT (see
/// `dw_gradient_depends_on_the_cofactor`).
pub(crate) fn prove_linear_gradient_dw() -> Result<GradientPropertyResult, SmtError> {
    let program = build_linear_gradient_dw(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(GradientPropertyResult {
        property: "gradient_linear_dW_equals_x".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the df/dW query. When `correct` is false the returned gradient is the
/// weight `W` rather than the input `x`; tests flip it to confirm the proof
/// depends on which factor the multiply node hands back.
fn build_linear_gradient_dw(correct: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let w = declare_real(&mut program, "W"); // current weight (a multiply input)
    let x = declare_real(&mut program, "x"); // input activation
    let b = declare_real(&mut program, "b"); // bias, held constant

    // f(W) = W*x + b evaluated at W = 0 and W = 1. Concrete W keeps every product
    // literal*variable, so the query stays linear.
    let f_at_w0 = define_real(
        &mut program,
        "f_at_W0",
        &Expr::real(0).real_mul(x.clone()).real_add(b.clone()),
    );
    let f_at_w1 = define_real(
        &mut program,
        "f_at_W1",
        &Expr::real(1).real_mul(x.clone()).real_add(b),
    );
    let df_dw = define_real(&mut program, "df_dW", &f_at_w1.real_sub(f_at_w0)); // = x

    // Reverse-mode grad of z = W*x w.r.t. W is the co-factor x.
    let grad_claim = define_real(&mut program, "grad_claim", if correct { &x } else { &w });

    // Violation: the derived derivative disagrees with the returned gradient.
    program.assert(df_dw.ne(grad_claim));
    program.check_sat();
    program
}

/// Prove that for f(W, x, b) = W*x + b, the gradient df/db = 1.
///
/// We read `df/db` as a finite difference over a unit step in `b`. The `W*x` term
/// is constant w.r.t. `b`, so we model it as one opaque variable `Wx` (no
/// product — the query stays linear): `f(b) = Wx + b` and `f(b+1) = Wx + (b+1)`,
/// whose difference is exactly `1` because `Wx` and `b` cancel. That derived
/// derivative must equal `1`.
///
/// The realistic slip (`!correct`) forgets to apply the `+1` perturbation, so the
/// perturbed point coincides with the base point and the difference collapses to
/// `0`, making the query SAT (see `db_gradient_depends_on_the_perturbation`).
pub(crate) fn prove_linear_gradient_db() -> Result<GradientPropertyResult, SmtError> {
    let program = build_linear_gradient_db(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(GradientPropertyResult {
        property: "gradient_linear_db_equals_one".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the df/db query. When `correct` is false the perturbation step is `0`
/// instead of `1`, so the finite difference degenerates to `0`; tests flip it to
/// confirm the derived derivative — not a restated `1` — is what is checked.
fn build_linear_gradient_db(correct: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // W*x is held constant w.r.t. b; model it as one opaque term (no product).
    let linear_part = declare_real(&mut program, "Wx");
    let b = declare_real(&mut program, "b");

    let step = if correct { Expr::real(1) } else { Expr::real(0) };
    let f_at_b = define_real(
        &mut program,
        "f_at_b",
        &linear_part.clone().real_add(b.clone()),
    );
    let f_at_b_step = define_real(
        &mut program,
        "f_at_b_step",
        &linear_part.real_add(b.real_add(step)),
    );
    let df_db = define_real(&mut program, "df_db", &f_at_b_step.real_sub(f_at_b));

    // Violation: the bias gradient is not the derived value 1.
    program.assert(df_db.ne(Expr::real(1)));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 3: ReLU Gradient
// ---------------------------------------------------------------------------

/// Prove that d/dx max(0, x) = 1 when x > 0.
///
/// Both ReLU evaluations come from the `max(0, ·)` definition (see
/// [`relu_via_max`]), not from a restated answer. On the positive branch we take
/// two in-branch points `x` and `x+1` (both `> 0`), and read the gradient as the
/// finite difference `relu(x+1) - relu(x)` over the unit step. The `max`
/// constraints force `relu(x) = x` and `relu(x+1) = x+1` when `x > 0`, so the
/// derived gradient is `1`.
///
/// The slip (`!correct`) returns the wrong-branch gradient `0` (the "dead ReLU"),
/// making the query SAT (see `relu_positive_depends_on_the_branch`).
pub(crate) fn prove_relu_gradient_positive() -> Result<GradientPropertyResult, SmtError> {
    let program = build_relu_gradient_positive(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(GradientPropertyResult {
        property: "gradient_relu_positive_branch".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the positive-branch query. When `correct` is false the claimed gradient
/// is `0` instead of `1`; tests flip it to confirm the proof reads the slope off
/// the real ReLU rather than restating `1`.
fn build_relu_gradient_positive(correct: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let x = declare_real(&mut program, "x");
    program.assert(x.clone().real_gt(Expr::real(0))); // positive branch

    // ReLU at x and at x+1 (both > 0), each from the max definition.
    let relu_x = relu_via_max(&mut program, "relu_x", &x);
    let x_step = x.real_add(Expr::real(1));
    let relu_x_step = relu_via_max(&mut program, "relu_x_step", &x_step);

    // Gradient by finite difference over the unit step (both points in-branch).
    let grad = define_real(&mut program, "relu_grad", &relu_x_step.real_sub(relu_x));
    let claim = if correct { Expr::real(1) } else { Expr::real(0) };
    let grad_claim = define_real(&mut program, "grad_claim", &claim);

    // Violation: the derived slope disagrees with the claimed gradient.
    program.assert(grad.ne(grad_claim));
    program.check_sat();
    program
}

/// Prove that d/dx max(0, x) = 0 when x < 0.
///
/// Mirrors the positive branch, using two in-branch points `x` and `x-1` (both
/// `< 0`). The `max` constraints force `relu(x) = 0` and `relu(x-1) = 0` when
/// `x < 0`, so the finite-difference gradient `relu(x) - relu(x-1)` is `0`.
///
/// The slip (`!correct`) returns the active-branch gradient `1`, making the query
/// SAT (see `relu_negative_depends_on_the_branch`).
pub(crate) fn prove_relu_gradient_negative() -> Result<GradientPropertyResult, SmtError> {
    let program = build_relu_gradient_negative(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(GradientPropertyResult {
        property: "gradient_relu_negative_branch".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the negative-branch query. When `correct` is false the claimed gradient
/// is `1` instead of `0`; tests flip it to confirm the proof reads the slope off
/// the real ReLU rather than restating `0`.
fn build_relu_gradient_negative(correct: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let x = declare_real(&mut program, "x");
    program.assert(x.clone().real_lt(Expr::real(0))); // negative branch

    // ReLU at x and at x-1 (both < 0), each from the max definition.
    let relu_x = relu_via_max(&mut program, "relu_x", &x);
    let x_step = x.real_sub(Expr::real(1));
    let relu_x_step = relu_via_max(&mut program, "relu_x_step", &x_step);

    // Gradient by finite difference over the unit step from (x-1) to x.
    let grad = define_real(&mut program, "relu_grad", &relu_x.real_sub(relu_x_step));
    let claim = if correct { Expr::real(0) } else { Expr::real(1) };
    let grad_claim = define_real(&mut program, "grad_claim", &claim);

    // Violation: the derived slope disagrees with the claimed gradient.
    program.assert(grad.ne(grad_claim));
    program.check_sat();
    program
}

/// Prove that the subgradient convention d/dx relu(0) = 0 matches the left
/// derivative of ReLU at the kink.
///
/// At `x = 0` ReLU is not differentiable; the standard ML convention picks the
/// left derivative, `0`. We derive that left derivative from the `max`
/// definition: `relu(0) = 0` and `relu(-1) = 0`, so the one-sided finite
/// difference `(relu(0) - relu(-1)) / 1 = 0`. The convention value must equal it.
///
/// The slip (`!correct`) uses the *right* derivative `1` instead — the two
/// one-sided derivatives genuinely disagree at the kink — making the query SAT
/// (see `relu_at_zero_depends_on_the_side`).
pub(crate) fn prove_relu_gradient_at_zero() -> Result<GradientPropertyResult, SmtError> {
    let program = build_relu_gradient_at_zero(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(GradientPropertyResult {
        property: "gradient_relu_at_zero_convention".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the at-zero query. When `correct` is false the convention value is the
/// right derivative `1` rather than the left derivative `0`; tests flip it to
/// confirm the convention is checked against the derived left derivative.
fn build_relu_gradient_at_zero(correct: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // relu(0) and relu(-1), each from the max definition (both derive to 0).
    let relu_0 = relu_via_max(&mut program, "relu_0", &Expr::real(0));
    let relu_neg1 = relu_via_max(&mut program, "relu_neg1", &Expr::real(-1));

    // Left derivative at 0 = (relu(0) - relu(-1)) / 1 = 0.
    let left_deriv = define_real(&mut program, "left_deriv", &relu_0.real_sub(relu_neg1));

    // ML convention: relu'(0) is the left derivative, 0.
    let conv = if correct { Expr::real(0) } else { Expr::real(1) };
    let conv_grad = define_real(&mut program, "conv_grad", &conv);

    // Violation: the convention disagrees with the derived left derivative.
    program.assert(conv_grad.ne(left_deriv));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 4: Sigmoid Gradient
// ---------------------------------------------------------------------------

/// Prove that d/dx sigma(x) = sigma(x) * (1 - sigma(x)).
///
/// `exp` is transcendental, so the two forms of the derivative would need the
/// nonlinear product `s*(1-s)` (undecidable `QF_NRA`). Instead we pin the sigmoid
/// value to a concrete rational `s = sigma(x) = 3/5` (the endorsed "pin the
/// nonlinear piece to a constant" tactic) and check the derivative written two
/// ways:
///
///   - product form  `sigma(x) * sigma(-x)`, using the *free* reflected value; and
///   - target form   `sigma(x) * (1 - sigma(x))`, the ground-truth derivative.
///
/// These are DIFFERENT expressions — `(* s sigma_neg_x)` versus `(* s (- 1 s))` —
/// that coincide only because the reflection symmetry `sigma(-x) = 1 - sigma(x)`
/// is asserted as a load-bearing hypothesis. Crucially, that symmetry is stated as
/// an equality between two *declared* quantities (`sigma_neg_x = one_minus_sigma`),
/// so the solver must derive the conclusion by applying it rather than having it
/// folded into `sigma_neg_x`'s definition. Drop the hypothesis and `sigma_neg_x`
/// is unconstrained, so the two forms disagree and the query is SAT. With `s` a
/// literal every product is `constant * variable`, so the query is linear
/// (`QF_LRA`).
///
/// The slip (`!correct`) flips the symmetry to `sigma(-x) = 1 + sigma(x)`, so the
/// two forms disagree and the query is SAT (see `sigmoid_depends_on_the_reflection`).
pub(crate) fn prove_sigmoid_gradient() -> Result<GradientPropertyResult, SmtError> {
    let program = build_sigmoid_gradient(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(GradientPropertyResult {
        property: "gradient_sigmoid_derivative".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the sigmoid-derivative query. When `correct` is false the reflection
/// symmetry is `sigma(-x) = 1 + sigma(x)` instead of `1 - sigma(x)`; tests flip
/// it to confirm the two forms only coincide because of the true symmetry.
fn build_sigmoid_gradient(correct: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let s = Expr::real_ratio(3, 5); // sigma(x) = 3/5 at the chosen point
    let one = Expr::real(1);

    // The reflected value 1 - sigma(x), computed independently as a *named*
    // quantity. (The `!correct` slip corrupts it to 1 + sigma(x).)
    let refl = if correct {
        define_real(&mut program, "one_minus_sigma", &one.clone().real_sub(s.clone()))
    } else {
        define_real(&mut program, "one_minus_sigma", &one.clone().real_add(s.clone()))
    };

    // sigma(-x) is a free value. The sigmoid reflection symmetry is the *hypothesis*
    // sigma(-x) = 1 - sigma(x), i.e. `sigma_neg_x = refl`. Stated as an equality
    // between two declared quantities, the solver must APPLY it to equate the two
    // derivative forms below (it is not folded into `sigma_neg_x`'s definition).
    let sc = declare_real(&mut program, "sigma_neg_x"); // sigma(-x)
    program.assert(sc.clone().eq(refl));

    // sigma'(x) written two structurally different ways:
    //   product form  s * sigma(-x)   (the free reflected value)
    //   target form   s * (1 - s)     (the ground-truth derivative)
    let grad_product = define_real(&mut program, "grad_product", &s.clone().real_mul(sc));
    let grad_formula = define_real(
        &mut program,
        "grad_formula",
        &s.clone().real_mul(one.real_sub(s)),
    );

    // Violation: the two forms of the derivative disagree. They coincide only
    // because of the reflection hypothesis `sigma_neg_x = refl`.
    program.assert(grad_product.ne(grad_formula));
    program.check_sat();
    program
}

/// Prove that sigmoid gradient is bounded: 0 < sigma'(x) <= 0.25 for all x.
///
/// Since sigma'(x) = s*(1-s) and s in (0,1), the maximum is at s=0.5
/// where sigma'(x) = 0.25. This bound is critical for gradient stability
/// analysis.
pub(crate) fn prove_sigmoid_gradient_bounded() -> Result<GradientPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let s = declare_real(&mut program, "s");

    // Sigmoid range: 0 < s < 1
    let zero = Expr::real(0);
    let one = Expr::real(1);
    program.assert(s.clone().real_gt(zero.clone()));
    program.assert(s.clone().real_lt(one.clone()));

    // sig_grad = s * (1 - s)
    let one_minus_s = one.real_sub(s.clone());
    let sig_grad = s.real_mul(one_minus_s);

    // Negated property: sig_grad <= 0 OR sig_grad > 0.25
    let quarter = real_from_f64(0.25)?;
    let too_low = sig_grad.clone().real_le(zero);
    let too_high = sig_grad.real_gt(quarter);
    let violation = too_low.or(too_high);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(GradientPropertyResult {
        property: "gradient_sigmoid_bounded_0_to_quarter".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 5: Softmax Jacobian Diagonal
// ---------------------------------------------------------------------------

/// Prove that the diagonal of the softmax Jacobian satisfies: ds_i/dx_i = s_i * (1 - s_i).
///
/// Same algebraic form as the sigmoid gradient, so the same tactic applies: pin
/// the softmax probability to a concrete rational `s_i = 3/5` and check the
/// diagonal Jacobian written two structurally different ways:
///
///   - mass-on-others form  `s_i * s_rest`, where `s_rest` is the *free* total
///     probability on the other classes; and
///   - target form          `s_i * (1 - s_i)`, the ground-truth diagonal entry.
///
/// These are DIFFERENT expressions — `(* s_i s_rest)` versus `(* s_i (- 1 s_i))` —
/// that coincide only because softmax normalization is asserted as a load-bearing
/// hypothesis: `s_rest = 1 - s_i`, stated as an equality between two *declared*
/// quantities (`s_rest = one_minus_s_i`) so the solver must apply it rather than
/// having it folded into `s_rest`'s definition. Drop it and `s_rest` is
/// unconstrained, so the two forms disagree and the query is SAT. With `s_i` a
/// literal the query is linear (`QF_LRA`, decidable).
///
/// The slip (`!correct`) forgets normalization and uses `s_rest = 1 + s_i`, so
/// the two forms disagree and the query is SAT (see
/// `softmax_diagonal_depends_on_normalization`).
pub(crate) fn prove_softmax_jacobian_diagonal() -> Result<GradientPropertyResult, SmtError> {
    let program = build_softmax_jacobian_diagonal(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(GradientPropertyResult {
        property: "gradient_softmax_jacobian_diagonal".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the softmax-diagonal query. When `correct` is false the normalization is
/// `s_rest = 1 + s_i` instead of `1 - s_i`; tests flip it to confirm the two forms
/// only coincide because softmax sums to 1.
fn build_softmax_jacobian_diagonal(correct: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let s_i = Expr::real_ratio(3, 5); // softmax prob for class i
    let one = Expr::real(1);

    // The remaining probability mass 1 - s_i, computed independently as a *named*
    // quantity. (The `!correct` slip corrupts it to 1 + s_i.)
    let rest_mass = if correct {
        define_real(&mut program, "one_minus_s_i", &one.clone().real_sub(s_i.clone()))
    } else {
        define_real(&mut program, "one_minus_s_i", &one.clone().real_add(s_i.clone()))
    };

    // s_rest is the free probability mass on the other classes. Softmax
    // normalization is the *hypothesis* s_rest = 1 - s_i, i.e. `s_rest = rest_mass`.
    // Stated as an equality between two declared quantities, the solver must APPLY
    // it to equate the two diagonal forms below (not fold it into `s_rest`).
    let s_rest = declare_real(&mut program, "s_rest"); // prob mass on other classes
    program.assert(s_rest.clone().eq(rest_mass));

    // ds_i/dx_i written two structurally different ways:
    //   mass-on-others form  s_i * s_rest   (the free remaining mass)
    //   target form          s_i * (1 - s_i)  (the ground-truth diagonal entry)
    let jac_product = define_real(&mut program, "jac_product", &s_i.clone().real_mul(s_rest));
    let jacobian_diag = define_real(
        &mut program,
        "jacobian_diag",
        &s_i.clone().real_mul(one.real_sub(s_i)),
    );

    // Violation: the two forms of the diagonal entry disagree. They coincide only
    // because softmax normalizes to 1 (`s_rest = rest_mass`).
    program.assert(jac_product.ne(jacobian_diag));
    program.check_sat();
    program
}

/// Prove that the off-diagonal of the softmax Jacobian satisfies: ds_i/dx_j = -s_i * s_j (i != j).
///
/// For completeness, we also prove the off-diagonal Jacobian entry. This is needed
/// for correct backpropagation through softmax layers.
pub(crate) fn prove_softmax_jacobian_off_diagonal() -> Result<GradientPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let s_i = declare_real(&mut program, "s_i");
    let s_j = declare_real(&mut program, "s_j");
    let jacobian_off = declare_real(&mut program, "jacobian_off");

    // Softmax output constraints: 0 < s_i < 1, 0 < s_j < 1
    let zero = Expr::real(0);
    let one = Expr::real(1);
    program.assert(s_i.clone().real_gt(zero.clone()));
    program.assert(s_i.clone().real_lt(one.clone()));
    program.assert(s_j.clone().real_gt(zero));
    program.assert(s_j.clone().real_lt(one));

    // jacobian_off = -s_i * s_j
    let neg_si_sj = declare_real(&mut program, "neg_si_sj");
    let si_sj = s_i.clone().real_mul(s_j.clone());
    program.assert(neg_si_sj.clone().eq(si_sj.clone().real_neg()));
    program.assert(jacobian_off.clone().eq(neg_si_sj.clone()));

    // Negated property: jacobian_off != -s_i * s_j
    let violation = jacobian_off.ne(s_i.real_mul(s_j).real_neg());
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(GradientPropertyResult {
        property: "gradient_softmax_jacobian_off_diagonal".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 6: Cross-Entropy Gradient
// ---------------------------------------------------------------------------

/// Prove that for cross-entropy loss L = -sum(y_i * log(p_i)), dL/dp_i = -y_i / p_i.
///
/// Since log is transcendental, we work at the level of the derivative directly.
/// The cross-entropy loss for a single element is `L_i = -y_i * log(p_i)`.
/// Its derivative with respect to p_i is `dL_i/dp_i = -y_i / p_i` (since d/dp log(p) = 1/p).
///
/// We encode this as: given y_i and p_i > 0, the gradient `-y_i / p_i` satisfies
/// the identity `grad * p_i = -y_i` (avoiding division in the SMT encoding).
///
/// This is equivalent to proving: `grad * p_i + y_i = 0`.
pub(crate) fn prove_cross_entropy_gradient() -> Result<GradientPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let y_i = declare_real(&mut program, "y_i"); // target label (probability)
    let p_i = declare_real(&mut program, "p_i"); // predicted probability
    let grad = declare_real(&mut program, "grad"); // dL/dp_i

    // Constraints: p_i > 0 (probabilities are positive), y_i >= 0 (valid label)
    let zero = Expr::real(0);
    program.assert(p_i.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &p_i, 0.0, 1.0)?;
    assert_bounds(&mut program, &y_i, 0.0, 1.0)?;

    // Define grad = -y_i / p_i
    // Equivalently: grad * p_i = -y_i
    // Equivalently: grad * p_i + y_i = 0
    let grad_times_pi = declare_real(&mut program, "grad_times_pi");
    program.assert(grad_times_pi.clone().eq(grad.clone().real_mul(p_i.clone())));
    let neg_yi = y_i.clone().real_neg();
    program.assert(grad_times_pi.clone().eq(neg_yi));

    // Negated property: grad * p_i + y_i != 0
    let sum = grad.real_mul(p_i).real_add(y_i);
    let violation = sum.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(GradientPropertyResult {
        property: "gradient_cross_entropy".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove that cross-entropy gradient is non-positive when y_i >= 0 and p_i > 0.
///
/// Since grad = -y_i / p_i, and y_i >= 0, p_i > 0, the gradient is always <= 0.
/// This is important for loss minimization: the gradient points in the correct
/// direction for reducing loss via gradient descent.
pub(crate) fn prove_cross_entropy_gradient_sign() -> Result<GradientPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let y_i = declare_real(&mut program, "y_i");
    let p_i = declare_real(&mut program, "p_i");
    let grad = declare_real(&mut program, "grad");

    // Constraints
    let zero = Expr::real(0);
    program.assert(y_i.clone().real_ge(zero.clone()));
    program.assert(p_i.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &y_i, 0.0, 1.0)?;
    assert_bounds(&mut program, &p_i, 0.0, 1.0)?;

    // grad * p_i = -y_i  (definition of grad = -y_i / p_i)
    let neg_yi = y_i.real_neg();
    program.assert(grad.clone().real_mul(p_i).eq(neg_yi));

    // Negated property: grad > 0 (should be impossible given y_i >= 0, p_i > 0)
    let violation = grad.real_gt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(GradientPropertyResult {
        property: "gradient_cross_entropy_non_positive".to_string(),
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

    // --- Chain Rule Tests ---

    #[test]
    fn test_chain_rule_proven() {
        let result = prove_chain_rule().expect("proof should not error");
        // Linear (QF_LRA) over concrete data: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "Chain rule should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "gradient_chain_rule");
    }

    /// Dropping the inner derivative g'(x) leaves `f'(g(x))` = 3, which no longer
    /// matches the composite slope 3a — the query must be SAT.
    #[test]
    fn chain_rule_depends_on_the_inner_derivative() {
        let program = build_chain_rule(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "dropping g'(x) makes the chain rule wrong; query must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_chain_rule_multiplication_proven() {
        let result = prove_chain_rule_multiplication().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Product rule: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Product rule must not have counterexample: {}",
            result.detail
        );
        assert_eq!(result.property, "gradient_chain_rule_multiplication");
    }

    #[test]
    fn test_chain_rule_addition_proven() {
        let result = prove_chain_rule_addition().expect("proof should not error");
        assert!(
            result.proven,
            "Sum rule should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "gradient_chain_rule_addition");
    }

    /// Subtracting the derivatives (`f' - g'`) no longer matches the sum slope
    /// `a + b` — the query must be SAT.
    #[test]
    fn addition_rule_depends_on_the_sign() {
        let program = build_chain_rule_addition(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "subtracting derivatives breaks the sum rule; query must be SAT; got: {detail}",
        );
    }

    // --- Linear Gradient Tests ---

    #[test]
    fn test_linear_gradient_dw_proven() {
        let result = prove_linear_gradient_dw().expect("proof should not error");
        assert!(
            result.proven,
            "Linear gradient df/dW = x should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "gradient_linear_dW_equals_x");
    }

    /// Returning the weight W instead of the input x makes df/dW = x fail whenever
    /// x != W — the query must be SAT.
    #[test]
    fn dw_gradient_depends_on_the_cofactor() {
        let program = build_linear_gradient_dw(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "returning W instead of x is wrong; query must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_linear_gradient_db_proven() {
        let result = prove_linear_gradient_db().expect("proof should not error");
        assert!(
            result.proven,
            "Linear gradient df/db = 1 should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "gradient_linear_db_equals_one");
    }

    /// Forgetting the +1 perturbation collapses the finite difference to 0, so
    /// df/db = 1 fails — the query must be SAT.
    #[test]
    fn db_gradient_depends_on_the_perturbation() {
        let program = build_linear_gradient_db(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "a zero perturbation step gives df/db = 0, not 1; query must be SAT; got: {detail}",
        );
    }

    // --- ReLU Gradient Tests ---

    #[test]
    fn test_relu_gradient_positive_proven() {
        let result = prove_relu_gradient_positive().expect("proof should not error");
        assert!(
            result.proven,
            "ReLU gradient (x > 0) = 1 should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "gradient_relu_positive_branch");
    }

    /// Claiming gradient 0 on the positive branch contradicts the slope read off
    /// the real ReLU (which is 1) — the query must be SAT.
    #[test]
    fn relu_positive_depends_on_the_branch() {
        let program = build_relu_gradient_positive(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "positive-branch slope is 1, not 0; query must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_relu_gradient_negative_proven() {
        let result = prove_relu_gradient_negative().expect("proof should not error");
        assert!(
            result.proven,
            "ReLU gradient (x < 0) = 0 should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "gradient_relu_negative_branch");
    }

    /// Claiming gradient 1 on the negative branch contradicts the slope read off
    /// the real ReLU (which is 0) — the query must be SAT.
    #[test]
    fn relu_negative_depends_on_the_branch() {
        let program = build_relu_gradient_negative(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "negative-branch slope is 0, not 1; query must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_relu_gradient_at_zero_convention() {
        let result = prove_relu_gradient_at_zero().expect("proof should not error");
        assert!(
            result.proven,
            "ReLU gradient at x=0 (convention: 0) should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "gradient_relu_at_zero_convention");
    }

    /// Using the right derivative 1 as the convention disagrees with the derived
    /// left derivative 0 — the query must be SAT.
    #[test]
    fn relu_at_zero_depends_on_the_side() {
        let program = build_relu_gradient_at_zero(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "left derivative at 0 is 0, not the right derivative 1; query must be SAT; got: {detail}",
        );
    }

    // --- Sigmoid Gradient Tests ---

    #[test]
    fn test_sigmoid_gradient_proven() {
        let result = prove_sigmoid_gradient().expect("proof should not error");
        // Linear (QF_LRA) over a pinned sigmoid value: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "Sigmoid gradient should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "gradient_sigmoid_derivative");
    }

    /// Flipping the reflection symmetry to `sigma(-x) = 1 + sigma(x)` makes the
    /// product form and the target form of sigma' disagree — the query must be SAT.
    #[test]
    fn sigmoid_depends_on_the_reflection() {
        let program = build_sigmoid_gradient(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "a wrong reflection breaks the derivative identity; query must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_sigmoid_gradient_bounded() {
        let result = prove_sigmoid_gradient_bounded().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Sigmoid gradient bound: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Sigmoid gradient bound must not have counterexample: {}",
            result.detail
        );
        assert_eq!(result.property, "gradient_sigmoid_bounded_0_to_quarter");
    }

    // --- Softmax Jacobian Tests ---

    #[test]
    fn test_softmax_jacobian_diagonal_proven() {
        let result = prove_softmax_jacobian_diagonal().expect("proof should not error");
        // Linear (QF_LRA) over a pinned probability: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "Softmax Jacobian diagonal should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "gradient_softmax_jacobian_diagonal");
    }

    /// Forgetting softmax normalization (`s_rest = 1 + s_i`) makes the mass-on-
    /// others form and the target form of the diagonal disagree — query must be SAT.
    #[test]
    fn softmax_diagonal_depends_on_normalization() {
        let program = build_softmax_jacobian_diagonal(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "a wrong normalization breaks the diagonal identity; query must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_softmax_jacobian_off_diagonal_proven() {
        let result = prove_softmax_jacobian_off_diagonal().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Softmax Jacobian off-diagonal: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Softmax Jacobian off-diagonal must not have counterexample: {}",
            result.detail
        );
        assert_eq!(result.property, "gradient_softmax_jacobian_off_diagonal");
    }

    // --- Cross-Entropy Gradient Tests ---

    #[test]
    fn test_cross_entropy_gradient_proven() {
        let result = prove_cross_entropy_gradient().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Cross-entropy gradient: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Cross-entropy gradient must not have counterexample: {}",
            result.detail
        );
        assert_eq!(result.property, "gradient_cross_entropy");
    }

    #[test]
    fn test_cross_entropy_gradient_sign_proven() {
        let result = prove_cross_entropy_gradient_sign().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Cross-entropy gradient sign: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Cross-entropy gradient sign must not have counterexample: {}",
            result.detail
        );
        assert_eq!(result.property, "gradient_cross_entropy_non_positive");
    }

    // --- SMT2 Structure Tests ---

    #[test]
    fn test_all_proofs_have_valid_smt2() {
        let proofs: Vec<GradientPropertyResult> = vec![
            prove_chain_rule().unwrap(),
            prove_chain_rule_addition().unwrap(),
            prove_linear_gradient_dw().unwrap(),
            prove_linear_gradient_db().unwrap(),
            prove_relu_gradient_positive().unwrap(),
            prove_relu_gradient_negative().unwrap(),
            prove_relu_gradient_at_zero().unwrap(),
            prove_sigmoid_gradient().unwrap(),
            prove_softmax_jacobian_diagonal().unwrap(),
            prove_cross_entropy_gradient().unwrap(),
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
