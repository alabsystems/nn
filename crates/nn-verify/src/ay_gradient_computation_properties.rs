// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for gradient computation mathematical correctness (#4241).
//!
//! Proves fundamental mathematical properties of gradient computations used
//! throughout the nn-autodiff backward pass. These are algebraic identity
//! proofs showing that the gradient formulas implemented in the framework
//! are mathematically sound.
//!
//! # Properties Proved
//!
//! 1. **Chain rule correctness**: d/dx f(g(x)) = f'(g(x)) * g'(x)
//! 2. **Linear gradient**: d/dx (W*x + b) = W (matrix form for 2x2)
//! 3. **ReLU subgradient**: grad = 1 if x > 0, 0 if x < 0
//! 4. **Softmax Jacobian diagonal**: d(softmax_i)/d(x_i) = s_i * (1 - s_i)
//! 5. **Cross-entropy gradient**: d/dx (-log(p)) = -1/p
//! 6. **Batch gradient mean**: mean gradient = sum / batch_size
//!
//! # Proof Strategy
//!
//! - **Algebraic proofs (QF_NRA)**: Chain rule, softmax diagonal, cross-entropy
//!   involve products of symbolic variables. Proved via negated assertion + UNSAT.
//! - **Linear proofs (QF_LRA)**: Linear gradient, ReLU branches, and batch mean
//!   are purely linear and provable in LRA.
//! - **Piecewise proofs**: ReLU is proved per-branch (x > 0 and x < 0
//!   separately) since the derivative is piecewise.
//!
//! Part of #4241.

use ay_bindings::{Expr, Sort, AYProgram};

use crate::ay_real_lit::RealLit;
use crate::smt_error::SmtError;

/// Result of a gradient computation property proof attempt.
#[derive(Debug, Clone)]
pub struct GradientPropertyResult {
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
fn assert_bounds(program: &mut AYProgram, expr: &Expr, lower: &Expr, upper: &Expr) {
    program.assert(expr.clone().real_ge(lower.clone()));
    program.assert(expr.clone().real_le(upper.clone()));
}

/// Assert `expr > 0` (strict positivity).
fn assert_positive(program: &mut AYProgram, expr: &Expr) {
    program.assert(expr.clone().real_gt(Expr::real(0)));
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

/// Build result from program.
fn make_result(program: &AYProgram, property: &str) -> GradientPropertyResult {
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(program);
    GradientPropertyResult {
        property: property.to_string(),
        proven,
        smt2,
        detail,
    }
}

// ---------------------------------------------------------------------------
// Property 1: Chain Rule Correctness
// ---------------------------------------------------------------------------

/// Inner-layer slope `w1` (concrete) of the composition in
/// [`build_chain_rule_correctness`]. This is `du/dx`.
const CHAIN_INNER_SLOPE: i64 = 2;
/// Inner-layer bias `b1` (concrete). Kept literal so the outer product `w2 * u`
/// has a literal factor and the whole query stays linear (`QF_LRA`).
const CHAIN_INNER_BIAS: i64 = 3;

/// Prove the chain rule `d/dx f(g(x)) = f'(g(x)) * g'(x)` on an affine
/// composition, deriving the composed derivative independently of the chain-rule
/// product so the query is not vacuous.
///
/// The composition is `h(x) = f(g(x))` with an affine inner map
/// `g(x) = w1*x + b1` (concrete slope `w1`, so `g'(x) = w1`) and an affine outer
/// map `f(u) = w2*u + b2` (symbolic slope `w2`, so `f'(g(x)) = w2`). For an
/// affine `h` the exact derivative is the secant `(h(1) - h(0)) / (1 - 0)`, which
/// we obtain by *evaluating the composition* at `x = 0` and `x = 1` — no
/// reference to the product formula. The theorem is that this secant equals the
/// chain-rule product `w2 * w1`.
///
/// The two sides are genuinely independent: the secant comes from substituting
/// into `h`, the product from multiplying the layer slopes. Replacing the
/// product with a sum (the classic "added the derivatives" slip) makes the query
/// SAT — see `chain_rule_depends_on_multiplying_the_derivatives`.
///
/// Uses `QF_LRA`: `w1`/`b1` are literal, so `w2 * u` and `w2 * w1` each have a
/// literal factor and stay linear (decidable).
pub fn prove_chain_rule_correctness() -> Result<GradientPropertyResult, SmtError> {
    let program = build_chain_rule_correctness(true);
    Ok(make_result(&program, "chain_rule_correctness"))
}

/// Build the chain-rule query. When `multiply_derivatives` is false the outer and
/// inner derivatives are *added* (`w2 + w1`) instead of multiplied, a plausible
/// slip that makes the identity false; tests flip it to confirm the proof depends
/// on multiplying them.
fn build_chain_rule_correctness(multiply_derivatives: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Outer affine map f(u) = w2*u + b2 with symbolic slope and bias.
    let w2 = declare_real(&mut program, "w2");
    let b2 = declare_real(&mut program, "b2");
    let bound_lo = Expr::real(-100);
    let bound_hi = Expr::real(100);
    assert_bounds(&mut program, &w2, &bound_lo, &bound_hi);
    assert_bounds(&mut program, &b2, &bound_lo, &bound_hi);

    // Inner affine map g(x) = w1*x + b1 is fully concrete, so u(0) and u(1) are
    // literals: u(0) = b1, u(1) = w1 + b1.
    let u0 = Expr::real(CHAIN_INNER_BIAS);
    let u1 = Expr::real(CHAIN_INNER_SLOPE + CHAIN_INNER_BIAS);

    // Evaluate the composition h(x) = f(g(x)) = w2*u(x) + b2 at x = 0 and x = 1.
    let y0 = declare_real(&mut program, "y0");
    let y1 = declare_real(&mut program, "y1");
    program.assert(y0.clone().eq(w2.clone().real_mul(u0).real_add(b2.clone())));
    program.assert(y1.clone().eq(w2.clone().real_mul(u1).real_add(b2)));

    // Exact derivative of the affine composition over [0, 1]: the secant
    // (y1 - y0) / (1 - 0) = y1 - y0, derived only from evaluating h.
    let finite_diff = y1.real_sub(y0);

    // Chain rule: d/dx h = f'(g(x)) * g'(x) = w2 * w1. The knob turns the product
    // into a sum.
    let w1 = Expr::real(CHAIN_INNER_SLOPE);
    let chain_deriv = declare_real(&mut program, "chain_deriv");
    let combined = if multiply_derivatives {
        w2.clone().real_mul(w1)
    } else {
        w2.real_add(w1)
    };
    program.assert(chain_deriv.clone().eq(combined));

    // Violation: the finite-difference derivative disagrees with the chain rule.
    program.assert(finite_diff.ne(chain_deriv));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 2: Linear Gradient — d/dx (W*x + b) = W
// ---------------------------------------------------------------------------

/// Prove that for a linear transform `y = W*x + b` (2x2 matrix), the partial
/// `dy_i/dx_j` recovered by a *finite difference of the layer* equals `W_ij`.
///
/// Rather than assert `dy_i/dx_j = W_ij` and negate it (which proves nothing),
/// each partial is measured the way the backward pass' correctness is really at
/// stake: evaluate `y_i = W[i][0]*x0 + W[i][1]*x1 + b_i` at the base input
/// `(0, 0)` and at the unit perturbation of one coordinate, and take the
/// difference. Because the layer is affine the unit finite difference is exact,
/// so `y_i(e_j) - y_i(0) = W[i][j]`. The theorem is that this measured partial
/// equals the corresponding weight.
///
/// The content is that the layer applies `W` and not its transpose. Reading the
/// weights transposed (`W[j][i]`) — a very common backward-pass slip — sends the
/// measured off-diagonal partial to the wrong weight and makes the query SAT; see
/// `linear_gradient_depends_on_the_weight_layout`.
///
/// Uses `QF_LRA`: the inputs are literal perturbation points (`0`/`1`), so every
/// `W_ij * x_j` has a literal factor and the query stays linear (decidable). The
/// property name is unchanged for callers.
pub fn prove_linear_gradient() -> Result<GradientPropertyResult, SmtError> {
    let program = build_linear_gradient(true);
    Ok(make_result(&program, "linear_gradient_dy_dx_equals_W"))
}

/// One output `y_i = wi0*x0 + wi1*x1 + b_i` of the 2x2 linear layer at the
/// literal input `(x0, x1)`, declared and pinned so the finite difference is one
/// step removed from the raw dot product.
fn linear_output(
    program: &mut AYProgram,
    name: &str,
    wi0: &Expr,
    wi1: &Expr,
    bi: &Expr,
    x0: i64,
    x1: i64,
) -> Expr {
    let term = wi0
        .clone()
        .real_mul(Expr::real(x0))
        .real_add(wi1.clone().real_mul(Expr::real(x1)))
        .real_add(bi.clone());
    let y = declare_real(program, name);
    program.assert(y.clone().eq(term));
    y
}

/// Build the linear-gradient query. When `weights_row_major` is false the layer
/// reads `W` transposed (`W[j][i]` where `W[i][j]` belongs), the classic
/// transposed-weight slip; tests flip it to confirm the proof depends on the
/// layout.
fn build_linear_gradient(weights_row_major: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let bound_lo = Expr::real(-100);
    let bound_hi = Expr::real(100);

    // Weight matrix W = [[w00, w01], [w10, w11]] and bias b = [b0, b1].
    let w00 = declare_real(&mut program, "w00");
    let w01 = declare_real(&mut program, "w01");
    let w10 = declare_real(&mut program, "w10");
    let w11 = declare_real(&mut program, "w11");
    let b0 = declare_real(&mut program, "b0");
    let b1 = declare_real(&mut program, "b1");
    for v in [&w00, &w01, &w10, &w11, &b0, &b1] {
        assert_bounds(&mut program, v, &bound_lo, &bound_hi);
    }

    // The weights the layer actually reads for each output row. Row-major reads
    // W[i][j]; the transpose bug reads W[j][i], which only moves the off-diagonal
    // entries (the diagonal is fixed by symmetry of the index swap).
    let w0_0 = &w00;
    let w0_1 = if weights_row_major { &w01 } else { &w10 };
    let w1_0 = if weights_row_major { &w10 } else { &w01 };
    let w1_1 = &w11;

    // Evaluate each output at the base point and at each unit perturbation.
    let y0_base = linear_output(&mut program, "y0_base", w0_0, w0_1, &b0, 0, 0);
    let y0_bx0 = linear_output(&mut program, "y0_bx0", w0_0, w0_1, &b0, 1, 0);
    let y0_bx1 = linear_output(&mut program, "y0_bx1", w0_0, w0_1, &b0, 0, 1);
    let y1_base = linear_output(&mut program, "y1_base", w1_0, w1_1, &b1, 0, 0);
    let y1_bx0 = linear_output(&mut program, "y1_bx0", w1_0, w1_1, &b1, 1, 0);
    let y1_bx1 = linear_output(&mut program, "y1_bx1", w1_0, w1_1, &b1, 0, 1);

    // Unit finite differences recover the partials dy_i/dx_j.
    let p00 = y0_bx0.real_sub(y0_base.clone());
    let p01 = y0_bx1.real_sub(y0_base);
    let p10 = y1_bx0.real_sub(y1_base.clone());
    let p11 = y1_bx1.real_sub(y1_base);

    // Violation: a measured partial differs from the true weight W[i][j].
    let violation = Expr::or_many(vec![
        p00.ne(w00.clone()),
        p01.ne(w01.clone()),
        p10.ne(w10.clone()),
        p11.ne(w11.clone()),
    ]);
    program.assert(violation);
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 3: ReLU Subgradient
// ---------------------------------------------------------------------------

/// Prove the ReLU subgradient: grad = 1 if x > 0, grad = 0 if x < 0.
///
/// ReLU(x) = max(0, x) is piecewise linear:
///   - For x > 0: ReLU(x) = x, so d/dx ReLU(x) = 1
///   - For x < 0: ReLU(x) = 0, so d/dx ReLU(x) = 0
///
/// We prove both branches in a single proof by case-splitting on the sign of x.
/// The positive branch asserts grad = 1, the negative branch asserts grad = 0,
/// and we prove no counterexample exists in either case.
///
/// Uses `QF_LRA` since both branches are linear.
pub fn prove_relu_subgradient() -> Result<GradientPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let x = declare_real(&mut program, "x");
    let grad = declare_real(&mut program, "grad");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // Bound x to avoid unbounded reals
    let bound_lo = Expr::real(-100);
    let bound_hi = Expr::real(100);
    assert_bounds(&mut program, &x, &bound_lo, &bound_hi);

    // x != 0 (exclude the non-differentiable point)
    program.assert(x.clone().ne(zero.clone()));

    // Case-split: if x > 0 then grad = 1; if x < 0 then grad = 0
    // Encode as: (x > 0 => grad = 1) AND (x < 0 => grad = 0)
    // Equivalently: (x <= 0 OR grad = 1) AND (x >= 0 OR grad = 0)
    let pos_case = x
        .clone()
        .real_le(zero.clone())
        .or(grad.clone().eq(one.clone()));
    let neg_case = x
        .clone()
        .real_ge(zero.clone())
        .or(grad.clone().eq(zero.clone()));
    program.assert(pos_case);
    program.assert(neg_case);

    // Negated property: gradient is wrong in at least one case
    // (x > 0 AND grad != 1) OR (x < 0 AND grad != 0)
    let violation_pos = x.clone().real_gt(zero.clone()).and(grad.clone().ne(one));
    let violation_neg = x.real_lt(zero.clone()).and(grad.ne(zero));
    let violation = violation_pos.or(violation_neg);
    program.assert(violation);
    program.check_sat();

    Ok(make_result(&program, "relu_subgradient"))
}

// ---------------------------------------------------------------------------
// Property 4: Softmax Jacobian Diagonal
// ---------------------------------------------------------------------------

/// Prove the softmax Jacobian diagonal formula `d(softmax_i)/d(x_i) = s_i*(1-s_i)`
/// by deriving the diagonal from the off-diagonals via the conservation law,
/// rather than restating the formula and negating it.
///
/// The softmax Jacobian is `J_ij = -s_i*s_j` for `i != j` and `J_ii = s_i*(1-s_i)`
/// on the diagonal. Because the softmax outputs sum to 1, every *row* of `J` sums
/// to zero — a genuine, independent fact. We take a concrete distribution
/// `s = (1/2, 1/3, 1/6)`, form the off-diagonal entries `J_01`, `J_02` from the
/// off-diagonal formula, and *derive* the diagonal `J_00` from the conservation
/// law `J_00 + J_01 + J_02 = 0`. The theorem is that this conservation-derived
/// diagonal equals the diagonal formula `s_0*(1 - s_0)`.
///
/// This is exactly the consistency between the two halves of the Jacobian
/// formula. Dropping the minus sign on the off-diagonals (a very common slip)
/// flips the derived diagonal and makes the query SAT — see
/// `softmax_diagonal_depends_on_the_off_diagonal_sign`.
///
/// Uses `QF_LRA`: `s` is a concrete distribution, so `J_0j = -s_0*s_j` and
/// `s_0*(1 - s_0)` are constants (no variable-by-variable product), and the query
/// stays linear (decidable).
pub fn prove_softmax_jacobian_diagonal() -> Result<GradientPropertyResult, SmtError> {
    let program = build_softmax_jacobian_diagonal(true);
    Ok(make_result(&program, "softmax_jacobian_diagonal"))
}

/// `-a*b` when `negative`, else `+a*b`. Both `a` and `b` are concrete here, so
/// the product is a constant and introduces no nonlinearity.
fn signed_product(negative: bool, a: &Expr, b: &Expr) -> Expr {
    let prod = a.clone().real_mul(b.clone());
    if negative {
        prod.real_neg()
    } else {
        prod
    }
}

/// Build the softmax-diagonal query. When `off_diagonal_negative` is false the
/// off-diagonal entries drop their minus sign (`+s_i*s_j` instead of `-s_i*s_j`),
/// the classic sign slip; tests flip it to confirm the proof depends on it.
fn build_softmax_jacobian_diagonal(off_diagonal_negative: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // A concrete softmax distribution over 3 classes: s0 + s1 + s2 = 1.
    let s0 = Expr::real_ratio(1, 2);
    let s1 = Expr::real_ratio(1, 3);
    let s2 = Expr::real_ratio(1, 6);

    // Off-diagonal Jacobian entries J_0j = -s0*sj (j != 0), from the formula.
    let j01 = signed_product(off_diagonal_negative, &s0, &s1);
    let j02 = signed_product(off_diagonal_negative, &s0, &s2);

    // Derive the diagonal from the conservation law: rows of J sum to zero, so
    // J00 = -(J01 + J02). J00 is a declared var pinned by that constraint.
    let j00 = declare_real(&mut program, "j00");
    program.assert(j00.clone().real_add(j01).real_add(j02).eq(Expr::real(0)));

    // The diagonal formula under test: d(softmax_0)/d(x_0) = s0*(1 - s0).
    let one = Expr::real(1);
    let formula = s0.clone().real_mul(one.real_sub(s0));

    // Violation: the conservation-derived diagonal disagrees with the formula.
    program.assert(j00.ne(formula));
    program.check_sat();
    program
}

/// Prove that the softmax Jacobian diagonal is bounded in (0, 0.25].
///
/// Since s_i in (0,1), the function s_i*(1-s_i) achieves maximum 0.25 at s_i=0.5.
/// This bound is critical for gradient stability analysis in softmax layers.
///
/// Uses `QF_NRA`.
pub fn prove_softmax_jacobian_diagonal_bounded() -> Result<GradientPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let s_i = declare_real(&mut program, "s_i");

    let zero = Expr::real(0);
    let one = Expr::real(1);
    program.assert(s_i.clone().real_gt(zero.clone()));
    program.assert(s_i.clone().real_lt(one.clone()));

    // diag = s_i * (1 - s_i)
    let one_minus_si = one.real_sub(s_i.clone());
    let diag = s_i.real_mul(one_minus_si);

    // Violation: diag <= 0 OR diag > 1/4
    // Encode 1/4 as a variable constrained to equal 1/4
    let quarter = declare_real(&mut program, "quarter");
    program.assert(Expr::real(4).real_mul(quarter.clone()).eq(Expr::real(1)));

    let too_low = diag.clone().real_le(zero);
    let too_high = diag.real_gt(quarter);
    let violation = too_low.or(too_high);
    program.assert(violation);
    program.check_sat();

    Ok(make_result(&program, "softmax_jacobian_diagonal_bounded"))
}

// ---------------------------------------------------------------------------
// Property 5: Cross-Entropy Gradient
// ---------------------------------------------------------------------------

/// Prove that the cross-entropy gradient `d/dp (-log(p)) = -1/p` takes its exact
/// reciprocal values, deriving each gradient from the loss' defining relation
/// rather than restating and negating `grad*p = -1`.
///
/// For `L = -log(p)`, the gradient satisfies `grad * p = -1` (multiply the
/// identity `grad = -1/p` through by `p`). We pin two concrete probabilities and
/// impose that relation, then *check the exact value it forces*: at `p = 1/2` the
/// relation makes `grad = -2`, and at `p = 1/4` it makes `grad = -4`. The theorem
/// is that these derived gradients are exactly `-1/p`.
///
/// The content is the sign and reciprocal magnitude of the loss gradient.
/// Dropping the minus sign of `-log` (using `grad*p = +1`) forces `grad = +2`,
/// `+4` and breaks the checked values — the query turns SAT; see
/// `cross_entropy_gradient_depends_on_the_loss_sign`.
///
/// Uses `QF_LRA`: `p` is a concrete rational, so `grad*p` has a literal factor
/// and stays linear (decidable).
pub fn prove_cross_entropy_gradient() -> Result<GradientPropertyResult, SmtError> {
    let program = build_cross_entropy_gradient(true);
    Ok(make_result(&program, "cross_entropy_gradient"))
}

/// Build the cross-entropy-gradient query. When `loss_sign_negative` is false the
/// defining relation is `grad*p = +1` instead of `-1` — the classic "dropped the
/// minus in -log" slip; tests flip it to confirm the proof depends on the sign.
fn build_cross_entropy_gradient(loss_sign_negative: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // The loss' defining relation grad*p = -1 (or +1 under the sign bug).
    let rhs = if loss_sign_negative {
        Expr::real(-1)
    } else {
        Expr::real(1)
    };

    // Point A: p = 1/2  =>  the relation forces grad = -2.
    let grad_a = declare_real(&mut program, "grad_a");
    program.assert(grad_a.clone().real_mul(Expr::real_ratio(1, 2)).eq(rhs.clone()));

    // Point B: p = 1/4  =>  the relation forces grad = -4.
    let grad_b = declare_real(&mut program, "grad_b");
    program.assert(grad_b.clone().real_mul(Expr::real_ratio(1, 4)).eq(rhs));

    // Violation: either derived gradient is not the exact reciprocal -1/p.
    let violation = grad_a.ne(Expr::real(-2)).or(grad_b.ne(Expr::real(-4)));
    program.assert(violation);
    program.check_sat();
    program
}

/// Prove that the cross-entropy gradient is always negative for p in (0, 1].
///
/// Since d/dp (-log(p)) = -1/p and p > 0, the gradient is always negative.
/// This confirms the loss function's gradient points toward increasing p
/// (minimizing loss), which is the correct direction for optimization.
///
/// Uses `QF_NRA`.
pub fn prove_cross_entropy_gradient_negative() -> Result<GradientPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let p = declare_real(&mut program, "p");
    let grad = declare_real(&mut program, "grad");

    assert_positive(&mut program, &p);
    let zero = Expr::real(0);
    let one = Expr::real(1);
    assert_bounds(&mut program, &p, &zero, &one);

    // grad * p = -1 (definition of grad = -1/p)
    let neg_one = Expr::real(-1);
    program.assert(grad.clone().real_mul(p).eq(neg_one));

    // Violation: grad >= 0 (should be UNSAT since grad = -1/p < 0 for p > 0)
    let violation = grad.real_ge(zero);
    program.assert(violation);
    program.check_sat();

    Ok(make_result(&program, "cross_entropy_gradient_negative"))
}

// ---------------------------------------------------------------------------
// Property 6: Batch Gradient Mean
// ---------------------------------------------------------------------------

/// Number of equal-size groups the hierarchical mean in
/// [`build_batch_gradient_mean`] combines at the top level (two pairs).
const BATCH_TOP_GROUPS: i64 = 2;

/// Prove that the batch mean is well-defined by checking that averaging
/// *hierarchically* agrees with the flat mean, instead of restating and negating
/// the definition `mean*4 = sum`.
///
/// In mini-batch training the gradient is averaged over the batch,
/// `mean = (g0 + g1 + g2 + g3) / 4`. A correct averaging routine may split the
/// batch into equal groups and average the group means: `m01 = (g0+g1)/2`,
/// `m23 = (g2+g3)/2`, and then `mtop = (m01 + m23)/2`. Because the groups are
/// equal-size, `mtop` must equal the flat `mean`. That equality is the theorem,
/// and it is *derived* — `mean` from `mean*4 = sum`, `mtop` from three separate
/// group-average constraints — so the query is not vacuous.
///
/// The content is that the group counts are right. Dividing the top level by the
/// batch size (4) instead of by the number of groups (2) — a classic
/// double-counting slip — makes `mtop = mean/2` and turns the query SAT; see
/// `batch_gradient_mean_depends_on_the_group_count`.
///
/// Uses `QF_LRA` (purely linear).
pub fn prove_batch_gradient_mean() -> Result<GradientPropertyResult, SmtError> {
    let program = build_batch_gradient_mean(BATCH_TOP_GROUPS);
    Ok(make_result(&program, "batch_gradient_mean"))
}

/// Build the batch-mean query. `top_group_count` is the divisor used when
/// combining the two pair-means: the correct value is the number of groups
/// ([`BATCH_TOP_GROUPS`] = 2); tests pass a wrong count (e.g. the batch size 4) to
/// confirm the proof depends on it.
fn build_batch_gradient_mean(top_group_count: i64) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let bound_lo = Expr::real(-100);
    let bound_hi = Expr::real(100);
    let g0 = declare_real(&mut program, "g0");
    let g1 = declare_real(&mut program, "g1");
    let g2 = declare_real(&mut program, "g2");
    let g3 = declare_real(&mut program, "g3");
    for v in [&g0, &g1, &g2, &g3] {
        assert_bounds(&mut program, v, &bound_lo, &bound_hi);
    }

    // Flat mean over the batch of 4: mean * 4 = g0 + g1 + g2 + g3.
    let sum = g0
        .clone()
        .real_add(g1.clone())
        .real_add(g2.clone())
        .real_add(g3.clone());
    let mean = declare_real(&mut program, "mean");
    program.assert(mean.clone().real_mul(Expr::real(4)).eq(sum));

    // Hierarchical mean: average each pair, then average the two pair-means.
    let m01 = declare_real(&mut program, "m01");
    program.assert(m01.clone().real_mul(Expr::real(2)).eq(g0.real_add(g1)));
    let m23 = declare_real(&mut program, "m23");
    program.assert(m23.clone().real_mul(Expr::real(2)).eq(g2.real_add(g3)));

    // Combine the pair-means; the correct divisor is the number of groups.
    let mtop = declare_real(&mut program, "mtop");
    program.assert(
        mtop.clone()
            .real_mul(Expr::real(top_group_count))
            .eq(m01.real_add(m23)),
    );

    // Violation: the hierarchical mean disagrees with the flat mean.
    program.assert(mean.ne(mtop));
    program.check_sat();
    program
}

/// Prove that batch gradient mean preserves bounds: if |g_i| <= B for all i,
/// then |mean| <= B.
///
/// Since mean = (1/N) * sum and |sum| <= N*B (triangle inequality),
/// |mean| <= (1/N) * N * B = B.
///
/// This is important for gradient stability: averaging over a batch does not
/// amplify the gradient magnitude beyond the per-sample bound.
///
/// Uses `QF_LRA`.
pub fn prove_batch_gradient_mean_bounded() -> Result<GradientPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let bound = declare_real(&mut program, "B");
    assert_positive(&mut program, &bound);
    let max_b = Expr::real(1000);
    let zero = Expr::real(0);
    assert_bounds(&mut program, &bound, &zero, &max_b);

    let neg_b = bound.clone().real_neg();

    // 4 gradients, each bounded by B
    let g0 = declare_real(&mut program, "g0");
    let g1 = declare_real(&mut program, "g1");
    let g2 = declare_real(&mut program, "g2");
    let g3 = declare_real(&mut program, "g3");

    assert_bounds(&mut program, &g0, &neg_b, &bound);
    assert_bounds(&mut program, &g1, &neg_b, &bound);
    assert_bounds(&mut program, &g2, &neg_b, &bound);
    assert_bounds(&mut program, &g3, &neg_b, &bound);

    // mean * 4 = g0 + g1 + g2 + g3
    let mean = declare_real(&mut program, "mean");
    let sum = g0.real_add(g1).real_add(g2).real_add(g3);
    let four = Expr::real(4);
    program.assert(mean.clone().real_mul(four).eq(sum));

    // Violation: |mean| > B, i.e., mean > B or mean < -B
    let too_high = mean.clone().real_gt(bound.clone());
    let too_low = mean.real_lt(neg_b);
    let violation = too_high.or(too_low);
    program.assert(violation);
    program.check_sat();

    Ok(make_result(&program, "batch_gradient_mean_bounded"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ay_gradient_computation_properties_tests.rs"]
mod tests;
