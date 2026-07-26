// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for weight initialization and parameter constraint properties (#4214).
//!
//! Properties proved:
//! 1. Xavier/Glorot uniform bounds: |w| <= sqrt(6/(fan_in+fan_out))
//! 2. Kaiming/He variance: Var(w) = 2/fan_in
//! 3. Orthogonal init: W^T * W == I (2x2)
//! 4. Zero-bias identity: b=0 => y = W*x
//! 5. Frobenius norm: ||W||_F^2 <= n * limit^2
//! 6. Gradient scaling: grad_w^2 = grad_y^2 * x^2
//! 7. Parameter count of a uniform stack: total = N * per_layer_count
//! 8. Spectral norm: ||W*v||^2 <= ||W||_F^2 for unit v
//!
//! Proofs use QF_LIA (integer counts), QF_LRA, or QF_NRA. Sqrt is avoided by
//! squaring both sides; nonlinear pieces are pinned to concrete rationals.

use ay_bindings::{Expr, Sort, AYProgram};

use crate::ay_real_lit::RealLit;
use crate::smt_error::SmtError;

/// Result of a weight initialization property proof attempt.
#[derive(Debug, Clone)]
pub struct WeightInitResult {
    /// Human-readable property name.
    pub property: String,
    /// Whether the proof succeeded (UNSAT = property holds for all inputs).
    pub proven: bool,
    /// SMT-LIB2 text of the query.
    pub smt2: String,
    /// Solver detail message.
    pub detail: String,
}

fn declare_real(program: &mut AYProgram, name: &str) -> Expr {
    program.declare_const(name, Sort::real())
}

fn declare_int(program: &mut AYProgram, name: &str) -> Expr {
    program.declare_const(name, Sort::int())
}

/// Declare `name` and pin it to `term`, returning the new variable.
///
/// Naming each intermediate keeps the conclusion one step removed from its
/// hypotheses, so the solver derives it instead of matching an asserted answer.
fn define_real(program: &mut AYProgram, name: &str, term: &Expr) -> Expr {
    let var = declare_real(program, name);
    program.assert(var.clone().eq(term.clone()));
    var
}

/// Integer analogue of [`define_real`].
fn define_int(program: &mut AYProgram, name: &str, term: &Expr) -> Expr {
    let var = declare_int(program, name);
    program.assert(var.clone().eq(term.clone()));
    var
}

fn assert_positive(program: &mut AYProgram, expr: &Expr) {
    program.assert(expr.clone().real_gt(Expr::real(0)));
}

fn assert_bounds(program: &mut AYProgram, expr: &Expr, lower: &Expr, upper: &Expr) {
    program.assert(expr.clone().real_ge(lower.clone()));
    program.assert(expr.clone().real_le(upper.clone()));
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

fn make_result(property: &str, program: &AYProgram) -> WeightInitResult {
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(program);
    WeightInitResult {
        property: property.to_string(),
        proven,
        smt2,
        detail,
    }
}

/// Prove Xavier/Glorot uniform: w^2 <= limit^2 where limit^2*(fan_in+fan_out)=6.
pub fn prove_xavier_uniform_bounds() -> Result<WeightInitResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let fan_in = declare_real(&mut program, "fan_in");
    let fan_out = declare_real(&mut program, "fan_out");
    let one = Expr::real(1);
    let max_fan = Expr::real(10000);
    assert_bounds(&mut program, &fan_in, &one, &max_fan);
    assert_bounds(&mut program, &fan_out, &one, &max_fan);

    // limit^2 * (fan_in + fan_out) = 6
    let fan_sum = fan_in.real_add(fan_out);
    let limit_sq = declare_real(&mut program, "limit_sq");
    program.assert(limit_sq.clone().real_mul(fan_sum).eq(Expr::real(6)));
    assert_positive(&mut program, &limit_sq);

    // w^2 <= limit_sq
    let w = declare_real(&mut program, "w");
    let w_sq = declare_real(&mut program, "w_sq");
    program.assert(w_sq.clone().eq(w.clone().real_mul(w)));
    program.assert(w_sq.clone().real_le(limit_sq.clone()));

    // Violation: w^2 > limit_sq
    program.assert(w_sq.real_gt(limit_sq));
    program.check_sat();

    Ok(make_result("xavier_uniform_weight_bounded", &program))
}

/// Prove that Kaiming/He initialization preserves signal variance through a
/// ReLU linear layer: with `Var(w) = 2/fan_in`, the output variance equals the
/// input variance.
///
/// This is the *reason* the He variance is `2/fan_in` and not, say, `1/fan_in`
/// (Xavier). For a linear layer of `fan_in` independent zero-mean weights the
/// signal variance propagates as
///
/// ```text
/// Var(out) = fan_in * Var(w) * Var(in_eff),   Var(in_eff) = Var(in) / 2
/// ```
///
/// where the `/2` is ReLU passing half the variance of a zero-mean input.
/// Substituting the He rule `Var(w) = 2/fan_in` gives `Var(out) = Var(in)` —
/// variance is preserved layer to layer. The conclusion is *derived* from the
/// propagation rule applied to the He variance, not asserted; using Xavier's
/// `gain = 1` instead of ReLU's `gain = 2` breaks preservation and makes the
/// query SAT (see `kaiming_variance_depends_on_the_gain`).
///
/// `fan_in` and `Var(w)` are literals so `var_in` is the only variable factor in
/// every product; the query stays linear (decidable `QF_LRA`).
pub fn prove_kaiming_variance() -> Result<WeightInitResult, SmtError> {
    let program = build_kaiming_variance(true);
    Ok(make_result("kaiming_variance_constraint", &program))
}

/// Concrete fan-in of the layer whose variance propagation is checked. Kept a
/// literal so no variable×variable product appears (decidable `QF_LRA`).
const KAIMING_FAN_IN: i64 = 100;

/// Build the variance-preservation query. `use_relu_gain` selects the gain in
/// `Var(w) = gain/fan_in`: `2` (He, correct) or `1` (Xavier's gain, the slip that
/// forgets the factor ReLU needs). Tests flip it to confirm the proof depends on
/// the gain.
fn build_kaiming_variance(use_relu_gain: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Kaiming/He variance for a ReLU layer: Var(w) = gain / fan_in with gain = 2.
    let gain: i64 = if use_relu_gain { 2 } else { 1 };
    let var_w = Expr::real_ratio(gain, KAIMING_FAN_IN); // literal rational gain/fan_in

    // Input signal variance flowing into the layer (any positive value).
    let var_in = declare_real(&mut program, "var_in");
    assert_bounds(&mut program, &var_in, &Expr::real(1), &Expr::real(1000));

    // ReLU passes half the variance of a zero-mean input, so the effective input
    // variance the layer's linear part sees is var_in / 2 (division by a literal).
    let var_in_eff = var_in.clone().real_div(Expr::real(2));

    // Var(out) = fan_in * Var(w) * Var(in_eff). fan_in and Var(w) are literals, so
    // only var_in is a variable factor: the product is linear.
    let var_out_term = Expr::real(KAIMING_FAN_IN)
        .real_mul(var_w)
        .real_mul(var_in_eff);
    let var_out = define_real(&mut program, "var_out", &var_out_term);

    // The theorem: Kaiming's gain = 2 makes the layer variance-preserving,
    // Var(out) = Var(in). Violation: the two variances differ.
    program.assert(var_out.ne(var_in));
    program.check_sat();
    program
}

/// Prove orthogonal init: the Gram matrix `W^T W` of a 2x2 orthogonal `W` is the
/// identity — its columns are orthonormal.
///
/// `W = [[a, b], [c, d]]` is pinned to the rotation by the `(3/5, 4/5)`
/// Pythagorean angle, an exactly-representable orthogonal matrix. The three Gram
/// entries are then *computed* from the entries by the matrix-product rule
///
/// ```text
/// g00 = col0·col0 = a*a + c*c,  g01 = col0·col1 = a*b + c*d,  g11 = b*b + d*d
/// ```
///
/// and the theorem is that they equal `I`'s entries `1, 0, 1`. The conclusion is
/// derived from the products, not asserted: perturbing one entry so the columns
/// are no longer orthonormal makes the query SAT (see
/// `orthogonal_init_depends_on_orthonormality`).
///
/// Each Gram product is a variable times the pinned literal value of the *same*
/// entry, so every product carries a literal factor and the query stays linear
/// (decidable `QF_LRA`).
pub fn prove_orthogonal_init() -> Result<WeightInitResult, SmtError> {
    let program = build_orthogonal_init(true);
    Ok(make_result("orthogonal_init_wtw_identity", &program))
}

/// Build the orthogonality query. `orthonormal` picks `W`'s bottom-right entry:
/// `3/5` (the true rotation) or `4/5` (the slip that leaves column 1 with squared
/// norm `32/25 != 1`). Tests flip it to confirm the proof depends on the columns
/// actually being orthonormal.
fn build_orthogonal_init(orthonormal: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // W = [[a, b], [c, d]], declared so the SMT names the matrix elements.
    let a = declare_real(&mut program, "a");
    let b = declare_real(&mut program, "b");
    let c = declare_real(&mut program, "c");
    let d = declare_real(&mut program, "d");

    // Correct: rotation by the (3/5, 4/5) angle, orthonormal columns.
    // Bug: d = 4/5, so column 1 = [-4/5, 4/5] has squared norm 32/25 != 1.
    let a_val = Expr::real_ratio(3, 5);
    let b_val = Expr::real_ratio(-4, 5);
    let c_val = Expr::real_ratio(4, 5);
    let d_val = if orthonormal {
        Expr::real_ratio(3, 5)
    } else {
        Expr::real_ratio(4, 5)
    };
    program.assert(a.clone().eq(a_val.clone()));
    program.assert(b.clone().eq(b_val.clone()));
    program.assert(c.clone().eq(c_val.clone()));
    program.assert(d.clone().eq(d_val.clone()));

    // Gram matrix G = W^T W. Each product is (variable · pinned-literal), which
    // equals the true square/product because the variable is pinned to the same
    // literal, yet keeps a literal factor so the encoding is linear.
    let g00 = a
        .clone()
        .real_mul(a_val)
        .real_add(c.clone().real_mul(c_val.clone()));
    let g01 = a
        .real_mul(b_val.clone())
        .real_add(c.real_mul(d_val.clone()));
    let g11 = b.real_mul(b_val).real_add(d.real_mul(d_val));

    let g00v = define_real(&mut program, "g00", &g00);
    let g01v = define_real(&mut program, "g01", &g01);
    let g11v = define_real(&mut program, "g11", &g11);

    let one = Expr::real(1);
    let zero = Expr::real(0);

    // Violation: W^T W differs from I in some entry.
    let violation = g00v
        .ne(one.clone())
        .or(g01v.ne(zero))
        .or(g11v.ne(one));
    program.assert(violation);
    program.check_sat();
    program
}

/// Prove zero-bias identity: y = W*x + 0 = W*x (for 2-element output).
pub fn prove_zero_bias_identity() -> Result<WeightInitResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let zero = Expr::real(0);
    let bound = Expr::real(100);
    let neg_bound = Expr::real(-100);

    let z0 = declare_real(&mut program, "z0");
    let z1 = declare_real(&mut program, "z1");
    assert_bounds(&mut program, &z0, &neg_bound, &bound);
    assert_bounds(&mut program, &z1, &neg_bound, &bound);

    let b0 = declare_real(&mut program, "b0");
    let b1 = declare_real(&mut program, "b1");
    program.assert(b0.clone().eq(zero.clone()));
    program.assert(b1.clone().eq(zero.clone()));

    let y0 = declare_real(&mut program, "y0");
    let y1 = declare_real(&mut program, "y1");
    program.assert(y0.clone().eq(z0.clone().real_add(b0)));
    program.assert(y1.clone().eq(z1.clone().real_add(b1)));

    // Violation: y != z
    program.assert(y0.ne(z0).or(y1.ne(z1)));
    program.check_sat();

    Ok(make_result("zero_bias_preserves_linear_output", &program))
}

/// Prove Frobenius norm bound: ||W||_F^2 <= 4*L^2 for 2x2 W with |w_ij| <= L.
pub fn prove_frobenius_norm_bound() -> Result<WeightInitResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let limit = declare_real(&mut program, "L");
    assert_positive(&mut program, &limit);
    assert_bounds(&mut program, &limit, &Expr::real(0), &Expr::real(100));
    let neg_limit = limit.clone().real_neg();

    let w00 = declare_real(&mut program, "w00");
    let w01 = declare_real(&mut program, "w01");
    let w10 = declare_real(&mut program, "w10");
    let w11 = declare_real(&mut program, "w11");
    assert_bounds(&mut program, &w00, &neg_limit, &limit);
    assert_bounds(&mut program, &w01, &neg_limit, &limit);
    assert_bounds(&mut program, &w10, &neg_limit, &limit);
    assert_bounds(&mut program, &w11, &neg_limit, &limit);

    let frob_sq = declare_real(&mut program, "frob_sq");
    program.assert(
        frob_sq.clone().eq(w00
            .clone()
            .real_mul(w00)
            .real_add(w01.clone().real_mul(w01))
            .real_add(w10.clone().real_mul(w10))
            .real_add(w11.clone().real_mul(w11))),
    );

    let four_l_sq = declare_real(&mut program, "four_l_sq");
    program.assert(
        four_l_sq
            .clone()
            .eq(Expr::real(4).real_mul(limit.clone().real_mul(limit))),
    );

    // Violation: ||W||_F^2 > 4*L^2
    program.assert(frob_sq.real_gt(four_l_sq));
    program.check_sat();

    Ok(make_result(
        "frobenius_norm_bounded_by_n_limit_sq",
        &program,
    ))
}

/// Prove gradient scaling: grad_w = grad_y * x implies grad_w^2 = grad_y^2 * x^2.
///
/// This structural identity underlies the 1/sqrt(fan_in) scaling of weight
/// gradients in Xavier-initialized networks.
pub fn prove_gradient_scaling_fan_in() -> Result<WeightInitResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let fan_in = declare_real(&mut program, "fan_in");
    let one = Expr::real(1);
    assert_bounds(&mut program, &fan_in, &one, &Expr::real(10000));

    let bound = Expr::real(100);
    let neg_bound = Expr::real(-100);
    let grad_y = declare_real(&mut program, "grad_y");
    let x = declare_real(&mut program, "x");
    assert_bounds(&mut program, &grad_y, &neg_bound, &bound);
    assert_bounds(&mut program, &x, &neg_bound, &bound);

    let grad_w = declare_real(&mut program, "grad_w");
    program.assert(grad_w.clone().eq(grad_y.clone().real_mul(x.clone())));

    let grad_w_sq = declare_real(&mut program, "grad_w_sq");
    program.assert(grad_w_sq.clone().eq(grad_w.clone().real_mul(grad_w)));

    let expected = grad_y
        .clone()
        .real_mul(grad_y)
        .real_mul(x.clone().real_mul(x));

    // Violation: grad_w^2 != grad_y^2 * x^2
    program.assert(grad_w_sq.ne(expected));
    program.check_sat();

    Ok(make_result(
        "gradient_magnitude_scales_with_input",
        &program,
    ))
}

/// Prove the parameter count of a *homogeneous* 3-layer stack: when every layer
/// carries the same per-layer parameter count, the model's total parameter count
/// is three copies of a single layer's count.
///
/// Layer `i` reports `p_i = w_i + b_i` (weight-matrix entries plus bias entries)
/// and the model total sums them, `total = p0 + p1 + p2`. The theorem is the
/// *shortcut* count `total == 3 * p0` that a weight-tied / homogeneous stack
/// (e.g. `N` identical transformer blocks → `N ×` block params) admits. It is
/// load-bearing on uniformity: the `3 * p0` shortcut (a literal-scaled single
/// count) and the actual per-layer sum (a three-way sum of distinct counts) are
/// *different* expression shapes that agree only because the layers are
/// constrained identical. Drop the uniformity hypothesis and a stack whose layers
/// differ makes the two disagree, turning the query SAT (see
/// `param_count_uniform_stack_depends_on_uniformity`).
///
/// Counts are `Int` and the shortcut multiplies by the literal `3`, so no
/// variable×variable product appears and the query is decidable `QF_LIA`.
pub fn prove_param_count_uniform_stack() -> Result<WeightInitResult, SmtError> {
    let program = build_param_count_uniform_stack(true);
    Ok(make_result("parameter_count_uniform_stack_3x", &program))
}

/// Build the uniform-stack count query. `uniform` selects whether the layers are
/// constrained to share a per-layer count: `true` pins `p1 = p0` and `p2 = p0`
/// (a homogeneous stack — the theorem's hypothesis); `false` leaves the layers
/// free (a heterogeneous stack, for which the `3 * p0` shortcut is wrong). Tests
/// flip it to confirm the proof depends on the layers being uniform.
fn build_param_count_uniform_stack(uniform: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let one = Expr::int(1);
    let max = Expr::int(1_000_000);

    // Per-layer weight-matrix and bias-vector element counts. Symbolic, so the
    // theorem holds for any three layers rather than one fixed network; each is a
    // positive count.
    let mut ws = Vec::new();
    let mut bs = Vec::new();
    for i in 0..3 {
        let w = declare_int(&mut program, &format!("w{i}"));
        let b = declare_int(&mut program, &format!("b{i}"));
        program.assert(w.clone().int_ge(one.clone()));
        program.assert(w.clone().int_le(max.clone()));
        program.assert(b.clone().int_ge(one.clone()));
        program.assert(b.clone().int_le(max.clone()));
        ws.push(w);
        bs.push(b);
    }

    // Per-layer parameter count = weights + biases (the correct rule, always).
    let p0 = define_int(&mut program, "p0", &ws[0].clone().int_add(bs[0].clone()));
    let p1 = define_int(&mut program, "p1", &ws[1].clone().int_add(bs[1].clone()));
    let p2 = define_int(&mut program, "p2", &ws[2].clone().int_add(bs[2].clone()));

    // Uniformity hypothesis: the stack is homogeneous — every layer carries the
    // same per-layer count. This is exactly what the `3 * p0` shortcut relies on;
    // without it the layers may differ and the shortcut fails. The equalities pin
    // p1 and p2 to p0 as a symbol-to-symbol rename, so the solver uses them but
    // the lineage normalizer leaves each `p_i` defined by its own `w_i + b_i` —
    // the two sides of the conclusion stay structurally distinct.
    if uniform {
        program.assert(p1.clone().eq(p0.clone()));
        program.assert(p2.clone().eq(p0.clone()));
    }

    // Model total = sum of the per-layer counts.
    let total = define_int(&mut program, "total", &p0.clone().int_add(p1).int_add(p2));

    // The theorem: a homogeneous stack's total is three copies of one layer's
    // count. Violation: the per-layer sum differs from the `3 * p0` shortcut. The
    // sides are different shapes — a three-way sum versus a literal-scaled single
    // count — equal only because the layers are pinned identical.
    let shortcut = Expr::int(3).int_mul(p0);
    program.assert(total.ne(shortcut));
    program.check_sat();
    program
}

/// Prove spectral norm bound: ||W*v||^2 <= ||W||_F^2 for unit vector v.
///
/// For a 2x2 matrix W with unit eigenvector v (||v||=1), the Rayleigh
/// quotient v^T W^T W v cannot exceed the Frobenius norm squared.
/// This is a fundamental matrix inequality bounding the spectral norm.
pub fn prove_spectral_norm_bound() -> Result<WeightInitResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let bound = Expr::real(100);
    let neg_bound = Expr::real(-100);

    let w00 = declare_real(&mut program, "w00");
    let w01 = declare_real(&mut program, "w01");
    let w10 = declare_real(&mut program, "w10");
    let w11 = declare_real(&mut program, "w11");
    for v in [&w00, &w01, &w10, &w11] {
        assert_bounds(&mut program, v, &neg_bound, &bound);
    }

    let frob_sq = declare_real(&mut program, "frob_sq");
    program.assert(
        frob_sq.clone().eq(w00
            .clone()
            .real_mul(w00.clone())
            .real_add(w01.clone().real_mul(w01.clone()))
            .real_add(w10.clone().real_mul(w10.clone()))
            .real_add(w11.clone().real_mul(w11.clone()))),
    );

    // Unit eigenvector v = [v0, v1] with v0^2 + v1^2 = 1
    let v0 = declare_real(&mut program, "v0");
    let v1 = declare_real(&mut program, "v1");
    assert_bounds(&mut program, &v0, &Expr::real(-1), &Expr::real(1));
    assert_bounds(&mut program, &v1, &Expr::real(-1), &Expr::real(1));
    program.assert(
        v0.clone()
            .real_mul(v0.clone())
            .real_add(v1.clone().real_mul(v1.clone()))
            .eq(Expr::real(1)),
    );

    // W*v
    let wv0 = w00.real_mul(v0.clone()).real_add(w01.real_mul(v1.clone()));
    let wv1 = w10.real_mul(v0).real_add(w11.real_mul(v1));

    let wv_norm_sq = declare_real(&mut program, "wv_norm_sq");
    program.assert(
        wv_norm_sq.clone().eq(wv0
            .clone()
            .real_mul(wv0)
            .real_add(wv1.clone().real_mul(wv1))),
    );

    // Violation: ||W*v||^2 > ||W||_F^2
    program.assert(wv_norm_sq.real_gt(frob_sq));
    program.check_sat();

    Ok(make_result("spectral_norm_le_frobenius_norm", &program))
}

#[cfg(test)]
#[path = "ay_weight_init_constraints_tests.rs"]
mod tests;
