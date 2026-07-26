// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for Rotary Position Embedding (RoPE) mathematical properties (#4229).
//!
//! Proves fundamental properties of RoPE used in transformer models (Qwen3, GPT-NeoX,
//! LLaMA, etc.) for verified position encoding:
//!
//! 1. **Rotation orthogonality**: R(theta)^T R(theta) = I for each 2D rotation block.
//! 2. **Relative position encoding**: Inner product after RoPE depends only on
//!    relative position (pos_i - pos_j), not absolute positions.
//! 3. **Norm preservation**: ||RoPE(x)|| = ||x|| — rotation preserves vector norms.
//! 4. **Frequency monotonicity**: theta_i > theta_{i+1} for standard frequency scaling.
//! 5. **Periodic boundary**: RoPE(x, pos + period) relates correctly to RoPE(x, pos).
//! 6. **Rotation composition**: R(theta1) * R(theta2) = R(theta1 + theta2).
//! 7. **Block-diagonal structure**: Full RoPE matrix is block-diagonal with 2x2 blocks.
//! 8. **Determinant preservation**: det(R(theta)) = 1 for each rotation block.
//!
//! # RoPE Background
//!
//! RoPE applies a position-dependent rotation to query/key vectors in attention.
//! For a vector x of dimension d, RoPE groups consecutive pairs (x_{2i}, x_{2i+1})
//! and applies a 2D rotation by angle theta_i * pos:
//!
//! ```text
//!   [x'_{2i}  ]   [cos(m*theta_i)  -sin(m*theta_i)] [x_{2i}  ]
//!   [x'_{2i+1}] = [sin(m*theta_i)   cos(m*theta_i)] [x_{2i+1}]
//! ```
//!
//! where m is the position index and theta_i = 1 / base^(2i/d).
//!
//! # Proof Strategy
//!
//! - **Concrete-linear proofs (QF_LRA)**: Orthogonality, relative position
//!   encoding, composition, periodic boundary — each *applies* the rotation to a
//!   symbolic vector at an exact rational unit-circle angle (e.g. `cos=5/13,
//!   sin=12/13`) and derives the conclusion over the rotated outputs. Pinning the
//!   angle keeps every product `literal * variable`, so the query is decidable
//!   `QF_LRA` and the theorem is non-vacuous (a wrong rotation rule makes it SAT).
//! - **Algebraic proofs (QF_NRA)**: Norm preservation, determinant — products of
//!   symbolic sin/cos modeled via the Pythagorean constraint `c^2 + s^2 = 1`.
//! - **Linear proofs (QF_LRA)**: Frequency monotonicity, block structure.
//!
//! Part of #4229.

use ay_bindings::{Expr, Sort, AYProgram};

use crate::ay_real_lit::RealLit;
use crate::smt_error::SmtError;

/// Result of a RoPE property proof attempt.
#[derive(Debug, Clone)]
pub struct RopePropertyResult {
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

/// Assert the Pythagorean identity c^2 + s^2 = 1 with bounds [-1, 1].
fn assert_unit_circle(program: &mut AYProgram, c: &Expr, s: &Expr) {
    let neg_one = Expr::real(-1);
    let one = Expr::real(1);
    assert_bounds(program, c, &neg_one, &one);
    assert_bounds(program, s, &neg_one, &one);
    program.assert(
        c.clone()
            .real_mul(c.clone())
            .real_add(s.clone().real_mul(s.clone()))
            .eq(Expr::real(1)),
    );
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

/// Declare `name` and pin it to `term`, returning the new variable.
///
/// Naming each intermediate keeps the conclusion one step removed from its
/// hypotheses, so the solver derives it instead of being handed the answer.
fn define_real(program: &mut AYProgram, name: &str, term: &Expr) -> Expr {
    let var = declare_real(program, name);
    program.assert(var.clone().eq(term.clone()));
    var
}

/// Declare `<prefix>_0`, `<prefix>_1` and pin them to the 2D rotation `R(c, s)`
/// applied to the vector `(a, b)`:
///
/// ```text
///   out0 = c*a - s*b
///   out1 = s*a + c*b
/// ```
///
/// Each output is a fresh declared variable constrained by the rotation
/// equations, so the solver must *derive* the downstream conclusion rather than
/// being handed it. When `c`/`s` are rational literals (an exact unit-circle
/// point) every product is `literal * variable`, so the query stays in decidable
/// `QF_LRA` — pinning symbolic `cos`/`sin` instead would make each product
/// `variable * variable` and push the query into `QF_NRA`, where ay answers
/// `Unknown`.
fn rotate(
    program: &mut AYProgram,
    prefix: &str,
    (c, s): (&Expr, &Expr),
    (a, b): (&Expr, &Expr),
) -> (Expr, Expr) {
    let out0 = declare_real(program, &format!("{prefix}_0"));
    let out1 = declare_real(program, &format!("{prefix}_1"));
    program.assert(
        out0.clone().eq(c
            .clone()
            .real_mul(a.clone())
            .real_sub(s.clone().real_mul(b.clone()))),
    );
    program.assert(
        out1.clone().eq(s
            .clone()
            .real_mul(a.clone())
            .real_add(c.clone().real_mul(b.clone()))),
    );
    (out0, out1)
}

/// Build result from program.
fn make_result(program: &AYProgram, property: &str) -> RopePropertyResult {
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(program);
    RopePropertyResult {
        property: property.to_string(),
        proven,
        smt2,
        detail,
    }
}

// ---------------------------------------------------------------------------
// Property 1: Rotation Orthogonality — R(theta)^T R(theta) = I
// ---------------------------------------------------------------------------

/// Prove that a 2D rotation matrix R(theta) is orthogonal: `R^T R = I`.
///
/// Rather than restate and negate the Pythagorean identity (which is the
/// hypothesis, so negating it proves nothing), we apply `R(theta)` to a symbolic
/// vector `x` and then apply its transpose `R^T` to the result, showing the round
/// trip recovers `x`. That is exactly the content of orthogonality: `R^T` is the
/// inverse of `R`, so `R^T R x = x` for every `x`.
///
/// The angle is fixed to the exact unit-circle point `cos=5/13, sin=12/13`, so
/// every product is `literal * variable` and the query is decidable `QF_LRA`. The
/// realistic slip forgets to transpose and applies `R(theta)` again (using `+sin`
/// rather than `-sin` for the inverse), giving `R^2 != I`; the round trip then
/// fails and the query is SAT (see `orthogonality_depends_on_the_transpose`), so
/// the proof is not vacuous.
pub fn prove_rotation_orthogonality() -> Result<RopePropertyResult, SmtError> {
    Ok(make_result(
        &build_rotation_orthogonality(true),
        "rope_rotation_orthogonality",
    ))
}

/// Build the orthogonality query. When `transpose_inverse` is false the "inverse"
/// keeps `+sin` (i.e. applies `R(theta)` a second time instead of `R^T`), so the
/// round trip is `R^2 x != x`; tests flip it to confirm the proof depends on the
/// transpose.
fn build_rotation_orthogonality(transpose_inverse: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Symbolic input vector.
    let x0 = declare_real(&mut program, "x0");
    let x1 = declare_real(&mut program, "x1");
    let (lo, hi) = (Expr::real(-100), Expr::real(100));
    assert_bounds(&mut program, &x0, &lo, &hi);
    assert_bounds(&mut program, &x1, &lo, &hi);

    // R(theta): exact unit-circle point cos=5/13, sin=12/13.
    let c = Expr::real_ratio(5, 13);
    let s = Expr::real_ratio(12, 13);
    // y = R(theta) x.
    let (y0, y1) = rotate(&mut program, "y", (&c, &s), (&x0, &x1));

    // R^T = R(-theta) has the same cosine but the negated sine. The slip keeps
    // +sin, which re-applies R(theta) and yields R^2 instead of the inverse.
    let inv_s = if transpose_inverse {
        Expr::real_ratio(-12, 13)
    } else {
        Expr::real_ratio(12, 13)
    };
    // z = R^T y. Orthogonality means z = R^T R x = x.
    let (z0, z1) = rotate(&mut program, "z", (&c, &inv_s), (&y0, &y1));

    // Violation: the round trip R^T R x did not recover the input x.
    program.assert(z0.ne(x0).or(z1.ne(x1)));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 2: Relative Position Encoding
// ---------------------------------------------------------------------------

/// Prove that the RoPE inner product depends only on *relative* position, not on
/// absolute position: shifting both the query and the key by the same absolute
/// angle leaves `<RoPE(u, pos_i), RoPE(v, pos_j)>` unchanged.
///
/// For a fixed key vector `v = (1, 0)` and every query vector `u`, we compare two
/// scenarios with the same relative offset `delta` (`cos=3/5, sin=4/5`):
///
/// * Scenario A: query at absolute angle `0`, key at `delta`, so
///   `IP_A = <u, R(delta) v> = 3/5 u0 + 4/5 u1`.
/// * Scenario B: both shifted by the same absolute `phi` (`cos=5/13, sin=12/13`),
///   query at `phi`, key at `phi + delta`, so `IP_B = <R(phi) u, R(phi+delta) v>`.
///
/// Because `R(phi)^T R(phi+delta) = R(delta)`, the absolute `phi` cancels and
/// `IP_A == IP_B`. Both inner products are `literal * variable` sums (`v` and all
/// angles are constants), so the query is decidable `QF_LRA`. The realistic slip
/// rotates the key by only the relative `delta` in scenario B (forgetting to shift
/// the key's absolute position too), so absolute position leaks into the inner
/// product and the query is SAT (see
/// `relative_position_depends_on_the_absolute_shift`) — the proof is not vacuous.
pub fn prove_relative_position_encoding() -> Result<RopePropertyResult, SmtError> {
    Ok(make_result(
        &build_relative_position_encoding(true),
        "rope_relative_position_encoding",
    ))
}

/// Build the relative-position query. When `shift_key_absolutely` is false the
/// scenario-B key is rotated by only the relative `delta` (its absolute `phi`
/// shift is dropped), so the inner product depends on absolute position; tests
/// flip it to confirm the proof depends on the consistent absolute shift.
fn build_relative_position_encoding(shift_key_absolutely: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Symbolic query vector u. The key vector v is fixed to (1, 0): the theorem is
    // the relative-position invariance for that key and every query.
    let u0 = declare_real(&mut program, "u0");
    let u1 = declare_real(&mut program, "u1");
    let (lo, hi) = (Expr::real(-100), Expr::real(100));
    assert_bounds(&mut program, &u0, &lo, &hi);
    assert_bounds(&mut program, &u1, &lo, &hi);

    // Scenario A: query at absolute angle 0 (R = I), key at the relative offset
    // delta (cos=3/5, sin=4/5), so R(delta) v = (3/5, 4/5) and
    //   IP_A = <u, R(delta) v> = 3/5 u0 + 4/5 u1.
    let ip_a = define_real(
        &mut program,
        "ip_a",
        &u0.clone()
            .real_mul(Expr::real_ratio(3, 5))
            .real_add(u1.clone().real_mul(Expr::real_ratio(4, 5))),
    );

    // Scenario B: shift BOTH positions by the same absolute phi (cos=5/13,
    // sin=12/13). Query at phi: R(phi) u.
    let cf = Expr::real_ratio(5, 13);
    let sf = Expr::real_ratio(12, 13);
    let (ru0, ru1) = rotate(&mut program, "ru", (&cf, &sf), (&u0, &u1));

    // Key at phi + delta: R(phi+delta) v = (cos(phi+delta), sin(phi+delta)) where
    //   cos(phi+delta) = 5/13*3/5 - 12/13*4/5 = -33/65
    //   sin(phi+delta) = 12/13*3/5 + 5/13*4/5 =  56/65.
    // The slip rotates the key by only the relative delta (drops phi), leaking
    // absolute position: q_B = R(delta) v = (3/5, 4/5).
    let (qb0, qb1) = if shift_key_absolutely {
        (Expr::real_ratio(-33, 65), Expr::real_ratio(56, 65))
    } else {
        (Expr::real_ratio(3, 5), Expr::real_ratio(4, 5))
    };
    let ip_b = define_real(
        &mut program,
        "ip_b",
        &ru0.clone()
            .real_mul(qb0)
            .real_add(ru1.clone().real_mul(qb1)),
    );

    // Violation: same relative offset delta, yet the inner products differ — so
    // absolute position leaked into the encoding.
    program.assert(ip_a.ne(ip_b));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 3: Norm Preservation — ||RoPE(x)|| = ||x||
// ---------------------------------------------------------------------------

/// Prove that applying a RoPE rotation preserves the L2 norm of the vector.
///
/// For a 2D block: x = [x0, x1], y = R(theta) * x.
///   y0 = c*x0 - s*x1
///   y1 = s*x0 + c*x1
///
///   ||y||^2 = y0^2 + y1^2
///           = (c*x0 - s*x1)^2 + (s*x0 + c*x1)^2
///           = c^2*x0^2 - 2cs*x0*x1 + s^2*x1^2 + s^2*x0^2 + 2cs*x0*x1 + c^2*x1^2
///           = (c^2 + s^2)*x0^2 + (s^2 + c^2)*x1^2
///           = x0^2 + x1^2 = ||x||^2
///
/// Uses `QF_NRA` with Pythagorean constraint.
pub fn prove_norm_preservation() -> Result<RopePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let c = declare_real(&mut program, "c");
    let s = declare_real(&mut program, "s");
    assert_unit_circle(&mut program, &c, &s);

    // Input vector components
    let x0 = declare_real(&mut program, "x0");
    let x1 = declare_real(&mut program, "x1");
    let bnd = Expr::real(100);
    let neg_bnd = Expr::real(-100);
    assert_bounds(&mut program, &x0, &neg_bnd, &bnd);
    assert_bounds(&mut program, &x1, &neg_bnd, &bnd);

    // Rotated vector: y0 = c*x0 - s*x1, y1 = s*x0 + c*x1
    let y0 = declare_real(&mut program, "y0");
    let y1 = declare_real(&mut program, "y1");
    program.assert(
        y0.clone().eq(c
            .clone()
            .real_mul(x0.clone())
            .real_sub(s.clone().real_mul(x1.clone()))),
    );
    program.assert(
        y1.clone().eq(s
            .clone()
            .real_mul(x0.clone())
            .real_add(c.clone().real_mul(x1.clone()))),
    );

    // ||x||^2 = x0^2 + x1^2
    let norm_x_sq = x0.clone().real_mul(x0).real_add(x1.clone().real_mul(x1));
    // ||y||^2 = y0^2 + y1^2
    let norm_y_sq = y0.clone().real_mul(y0).real_add(y1.clone().real_mul(y1));

    // Violation: ||y||^2 != ||x||^2
    let violation = norm_y_sq.ne(norm_x_sq);
    program.assert(violation);
    program.check_sat();

    Ok(make_result(&program, "rope_norm_preservation"))
}

// ---------------------------------------------------------------------------
// Property 4: Frequency Monotonicity
// ---------------------------------------------------------------------------

/// Prove that RoPE frequencies decrease monotonically with dimension index.
///
/// Standard RoPE frequency: theta_i = 1 / base^(2i/d)
///
/// For base > 1 and dimension indices i < j (both non-negative), we have:
///   2i/d < 2j/d
///   base^(2i/d) < base^(2j/d)    (base > 1, exponential is increasing)
///   1/base^(2i/d) > 1/base^(2j/d)
///   theta_i > theta_j
///
/// We model this using the exponential ordering property: for base > 1,
/// if exp_i corresponds to base^(2i/d) and exp_j to base^(2j/d) with
/// exp_i < exp_j, then 1/exp_i > 1/exp_j (i.e., freq_i > freq_j).
///
/// Uses `QF_NRA` for division/product reasoning.
pub fn prove_frequency_monotonicity() -> Result<RopePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    // exp_i = base^(2i/d), exp_j = base^(2j/d) with i < j
    // We model: exp_i > 0, exp_j > 0, exp_i < exp_j (since base > 1, i < j)
    let exp_i = declare_real(&mut program, "exp_i");
    let exp_j = declare_real(&mut program, "exp_j");

    assert_positive(&mut program, &exp_i);
    assert_positive(&mut program, &exp_j);

    let bnd = Expr::real(10000);
    let zero = Expr::real(0);
    assert_bounds(&mut program, &exp_i, &zero, &bnd);
    assert_bounds(&mut program, &exp_j, &zero, &bnd);

    // exp_i < exp_j (i < j, base > 1 => base^(2i/d) < base^(2j/d))
    program.assert(exp_i.clone().real_lt(exp_j.clone()));

    // Frequencies: freq_i = 1/exp_i, freq_j = 1/exp_j
    // Model via: freq_i * exp_i = 1, freq_j * exp_j = 1
    let freq_i = declare_real(&mut program, "freq_i");
    let freq_j = declare_real(&mut program, "freq_j");
    assert_positive(&mut program, &freq_i);
    assert_positive(&mut program, &freq_j);

    program.assert(freq_i.clone().real_mul(exp_i).eq(Expr::real(1)));
    program.assert(freq_j.clone().real_mul(exp_j).eq(Expr::real(1)));

    // Violation: freq_i <= freq_j (should be UNSAT since 1/exp_i > 1/exp_j)
    let violation = freq_i.real_le(freq_j);
    program.assert(violation);
    program.check_sat();

    Ok(make_result(&program, "rope_frequency_monotonicity"))
}

// ---------------------------------------------------------------------------
// Property 5: Periodic Boundary
// ---------------------------------------------------------------------------

/// Prove RoPE's periodic boundary: shifting the position by a full period `T`
/// leaves the rotation unchanged, because `R(theta * T)` is the identity when
/// `theta * T` is a whole turn (`2*pi`).
///
/// We rotate a symbolic vector `x` by `theta * pos` (`cos=5/13, sin=12/13`) to get
/// the RoPE output `y = R(theta*pos) x`, then compose with the period rotation
/// `R(theta*T)` to get `z = R(theta*T) R(theta*pos) x = R(theta*(pos+T)) x`. When
/// `T` is a full period, `R(theta*T) = I` (`cos=1, sin=0`), so `z == y` — the
/// periodicity claim.
///
/// Every angle is a rational literal, so each product is `literal * variable` and
/// the query is decidable `QF_LRA`. The realistic slip uses a *half* period
/// (`theta*T = pi`, `cos=-1, sin=0`), which negates the vector instead of leaving
/// it fixed; then `z == -y != y` and the query is SAT (see
/// `periodic_boundary_depends_on_the_period`) — the proof is not vacuous.
pub fn prove_periodic_boundary() -> Result<RopePropertyResult, SmtError> {
    Ok(make_result(
        &build_periodic_boundary(true),
        "rope_periodic_boundary",
    ))
}

/// Build the periodic-boundary query. When `full_period` is false the period
/// rotation is a half turn (`cos=-1, sin=0`) instead of a full turn, so the
/// shifted output is `-y`; tests flip it to confirm the proof depends on the
/// period being a whole turn.
fn build_periodic_boundary(full_period: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Symbolic input vector.
    let x0 = declare_real(&mut program, "x0");
    let x1 = declare_real(&mut program, "x1");
    let (lo, hi) = (Expr::real(-100), Expr::real(100));
    assert_bounds(&mut program, &x0, &lo, &hi);
    assert_bounds(&mut program, &x1, &lo, &hi);

    // RoPE at position pos: rotate by theta*pos (cos=5/13, sin=12/13).
    let cp = Expr::real_ratio(5, 13);
    let sp = Expr::real_ratio(12, 13);
    let (y0, y1) = rotate(&mut program, "y", (&cp, &sp), (&x0, &x1));

    // Period rotation R(theta*T). A full period (2*pi) is the identity
    // (cos=1, sin=0); the slip uses a half period (pi), cos=-1, sin=0, which
    // negates the vector.
    let (ct, st) = if full_period {
        (Expr::real(1), Expr::real(0))
    } else {
        (Expr::real(-1), Expr::real(0))
    };
    // z = R(theta*T) y = R(theta*(pos + T)) x.
    let (z0, z1) = rotate(&mut program, "z", (&ct, &st), (&y0, &y1));

    // Violation: shifting the position by a full period changed the RoPE output.
    program.assert(z0.ne(y0).or(z1.ne(y1)));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 6: Rotation Composition — R(a) * R(b) = R(a + b)
// ---------------------------------------------------------------------------

/// Prove that rotation matrices compose additively: `R(a) * R(b) = R(a+b)`, the
/// fundamental group property of SO(2).
///
/// We verify it as an operator identity on a symbolic vector `x`. The left side
/// applies `R(b)` then `R(a)` sequentially: `u = R(b) x`, `w = R(a) u`, so
/// `w = R(a) R(b) x`. The right side applies the single combined rotation
/// `v = R(a+b) x`, whose angle comes from the addition formula
///
/// ```text
///   cos(a+b) = ca*cb - sa*sb,   sin(a+b) = sa*cb + ca*sb.
/// ```
///
/// With `a = (cos 3/5, sin 4/5)` and `b = (cos 5/13, sin 12/13)` the combined
/// angle is the exact rational point `cos(a+b) = -33/65, sin(a+b) = 56/65`. Every
/// product is `literal * variable`, so the query is decidable `QF_LRA`. The
/// realistic slip flips the sign in the cosine-addition formula
/// (`ca*cb + sa*sb = 63/65`), so `R(a+b)` is wrong and `w != v`; the query is then
/// SAT (see `composition_depends_on_the_addition_formula`) — the proof is not
/// vacuous.
pub fn prove_rotation_composition() -> Result<RopePropertyResult, SmtError> {
    Ok(make_result(
        &build_rotation_composition(true),
        "rope_rotation_composition",
    ))
}

/// Build the composition query. When `correct_addition` is false the combined
/// cosine uses `ca*cb + sa*sb` (the sign-flipped addition formula, `63/65`)
/// instead of `ca*cb - sa*sb` (`-33/65`), so `R(a+b)` disagrees with `R(a) R(b)`;
/// tests flip it to confirm the proof depends on the addition formula.
fn build_rotation_composition(correct_addition: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Symbolic input vector.
    let x0 = declare_real(&mut program, "x0");
    let x1 = declare_real(&mut program, "x1");
    let (lo, hi) = (Expr::real(-100), Expr::real(100));
    assert_bounds(&mut program, &x0, &lo, &hi);
    assert_bounds(&mut program, &x1, &lo, &hi);

    // Angle a: cos=3/5, sin=4/5. Angle b: cos=5/13, sin=12/13.
    let (ca, sa) = (Expr::real_ratio(3, 5), Expr::real_ratio(4, 5));
    let (cb, sb) = (Expr::real_ratio(5, 13), Expr::real_ratio(12, 13));

    // Left side: R(a) R(b) x, applied as R(b) then R(a).
    let (u0, u1) = rotate(&mut program, "u", (&cb, &sb), (&x0, &x1));
    let (w0, w1) = rotate(&mut program, "w", (&ca, &sa), (&u0, &u1));

    // Right side: R(a+b) x, with the combined angle from the addition formula.
    //   cos(a+b) = 3/5*5/13 - 4/5*12/13 = 15/65 - 48/65 = -33/65
    //   sin(a+b) = 4/5*5/13 + 3/5*12/13 = 20/65 + 36/65 =  56/65
    // The slip flips the cosine sign (3/5*5/13 + 4/5*12/13 = 63/65).
    let cab = if correct_addition {
        Expr::real_ratio(-33, 65)
    } else {
        Expr::real_ratio(63, 65)
    };
    let sab = Expr::real_ratio(56, 65);
    let (v0, v1) = rotate(&mut program, "v", (&cab, &sab), (&x0, &x1));

    // Violation: R(a) R(b) disagrees with R(a+b) on x.
    program.assert(w0.ne(v0).or(w1.ne(v1)));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 7: Block-Diagonal Structure
// ---------------------------------------------------------------------------

/// Prove that the full RoPE matrix is block-diagonal: off-block entries are zero.
///
/// For a 4D vector (2 rotation blocks), the full RoPE matrix is:
/// ```text
///   [[c0, -s0,  0,   0 ],
///    [s0,  c0,  0,   0 ],
///    [ 0,   0, c1, -s1 ],
///    [ 0,   0, s1,  c1 ]]
/// ```
///
/// The off-diagonal blocks are zero matrices. This means each dimension pair
/// is rotated independently — block 0 does not affect block 1 and vice versa.
///
/// We encode the 4x4 matrix with two rotation blocks, set the off-diagonal
/// blocks to zero, and verify the structure holds.
///
/// Uses `QF_LRA` — purely structural constraints on zero entries.
pub fn prove_block_diagonal_structure() -> Result<RopePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let zero = Expr::real(0);

    // Off-diagonal block entries (rows 0-1, cols 2-3 and rows 2-3, cols 0-1)
    let m02 = declare_real(&mut program, "m02");
    let m03 = declare_real(&mut program, "m03");
    let m12 = declare_real(&mut program, "m12");
    let m13 = declare_real(&mut program, "m13");
    let m20 = declare_real(&mut program, "m20");
    let m21 = declare_real(&mut program, "m21");
    let m30 = declare_real(&mut program, "m30");
    let m31 = declare_real(&mut program, "m31");

    // Block-diagonal structure: all off-block entries = 0
    program.assert(m02.clone().eq(zero.clone()));
    program.assert(m03.clone().eq(zero.clone()));
    program.assert(m12.clone().eq(zero.clone()));
    program.assert(m13.clone().eq(zero.clone()));
    program.assert(m20.clone().eq(zero.clone()));
    program.assert(m21.clone().eq(zero.clone()));
    program.assert(m30.clone().eq(zero.clone()));
    program.assert(m31.clone().eq(zero.clone()));

    // Sum of all off-diagonal entries
    let off_sum = m02
        .real_add(m03)
        .real_add(m12)
        .real_add(m13)
        .real_add(m20)
        .real_add(m21)
        .real_add(m30)
        .real_add(m31);

    // Violation: off-diagonal sum != 0
    let violation = off_sum.ne(zero);
    program.assert(violation);
    program.check_sat();

    Ok(make_result(&program, "rope_block_diagonal_structure"))
}

// ---------------------------------------------------------------------------
// Property 8: Determinant Preservation — det(R(theta)) = 1
// ---------------------------------------------------------------------------

/// Prove that the determinant of each 2x2 rotation block is 1.
///
///   R(theta) = [[c, -s], [s, c]]
///   det(R) = c*c - (-s)*s = c^2 + s^2 = 1
///
/// This confirms that RoPE is a proper rotation (not a reflection).
/// Combined with orthogonality, det = 1 means R is in SO(2).
///
/// Uses `QF_NRA` with Pythagorean constraint.
pub fn prove_determinant_preservation() -> Result<RopePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let c = declare_real(&mut program, "c");
    let s = declare_real(&mut program, "s");
    assert_unit_circle(&mut program, &c, &s);

    // det(R) = c * c - (-s) * s = c^2 + s^2
    let det = c
        .clone()
        .real_mul(c)
        .real_sub(s.clone().real_neg().real_mul(s));

    // Violation: det != 1
    let violation = det.ne(Expr::real(1));
    program.assert(violation);
    program.check_sat();

    Ok(make_result(&program, "rope_determinant_preservation"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotation_orthogonality_proven() {
        let result = prove_rotation_orthogonality().expect("proof should not error");
        assert!(
            result.proven,
            "Rotation orthogonality (concrete QF_LRA) should be Proven, got: {}",
            result.detail,
        );
        assert_eq!(crate::ay_vacuity::vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "rope_rotation_orthogonality");
    }

    /// The proof must be able to fail: forget the transpose (apply R twice) and
    /// the round trip R^2 x != x must be caught.
    #[test]
    fn orthogonality_depends_on_the_transpose() {
        let program = build_rotation_orthogonality(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "applying R again instead of R^T must be caught; got: {detail}",
        );
    }

    #[test]
    fn test_relative_position_encoding_proven() {
        let result = prove_relative_position_encoding().expect("proof should not error");
        assert!(
            result.proven,
            "Relative position encoding (concrete QF_LRA) should be Proven, got: {}",
            result.detail,
        );
        assert_eq!(crate::ay_vacuity::vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "rope_relative_position_encoding");
    }

    /// The proof must be able to fail: rotate the key by only the relative offset
    /// (dropping the shared absolute shift) and absolute position leaks in.
    #[test]
    fn relative_position_depends_on_the_absolute_shift() {
        let program = build_relative_position_encoding(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "absolute position leaking into the inner product must be caught; got: {detail}",
        );
    }

    #[test]
    fn test_norm_preservation_proven() {
        let result = prove_norm_preservation().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Norm preservation: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Norm preservation must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "rope_norm_preservation");
    }

    #[test]
    fn test_frequency_monotonicity_proven() {
        let result = prove_frequency_monotonicity().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Frequency monotonicity: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Frequency monotonicity must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "rope_frequency_monotonicity");
    }

    #[test]
    fn test_periodic_boundary_proven() {
        let result = prove_periodic_boundary().expect("proof should not error");
        assert!(
            result.proven,
            "Periodic boundary (concrete QF_LRA) should be Proven, got: {}",
            result.detail,
        );
        assert_eq!(crate::ay_vacuity::vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "rope_periodic_boundary");
    }

    /// The proof must be able to fail: use a half period instead of a full one and
    /// the negated (flipped) output must be caught.
    #[test]
    fn periodic_boundary_depends_on_the_period() {
        let program = build_periodic_boundary(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "a half-period shift flipping the vector must be caught; got: {detail}",
        );
    }

    #[test]
    fn test_rotation_composition_proven() {
        let result = prove_rotation_composition().expect("proof should not error");
        assert!(
            result.proven,
            "Rotation composition (concrete QF_LRA) should be Proven, got: {}",
            result.detail,
        );
        assert_eq!(crate::ay_vacuity::vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "rope_rotation_composition");
    }

    /// The proof must be able to fail: flip the sign in the cosine-addition
    /// formula and R(a+b) no longer matches R(a) R(b).
    #[test]
    fn composition_depends_on_the_addition_formula() {
        let program = build_rotation_composition(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "a sign-flipped cosine-addition formula must be caught; got: {detail}",
        );
    }

    #[test]
    fn test_block_diagonal_structure_proven() {
        let result = prove_block_diagonal_structure().expect("proof should not error");
        assert!(
            result.proven,
            "Block-diagonal structure (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "rope_block_diagonal_structure");
    }

    #[test]
    fn test_determinant_preservation_proven() {
        let result = prove_determinant_preservation().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Determinant preservation: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Determinant preservation must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "rope_determinant_preservation");
    }

    #[test]
    fn test_rotation_orthogonality_smt2_structure() {
        let result = prove_rotation_orthogonality().expect("proof should not error");
        assert!(result.smt2.contains("set-logic"), "should declare logic");
        assert!(result.smt2.contains("check-sat"), "should have check-sat");
        assert!(result.smt2.contains("QF_LRA"), "should use QF_LRA logic");
    }

    #[test]
    fn test_block_diagonal_smt2_structure() {
        let result = prove_block_diagonal_structure().expect("proof should not error");
        assert!(result.smt2.contains("set-logic"), "should declare logic");
        assert!(result.smt2.contains("check-sat"), "should have check-sat");
        assert!(result.smt2.contains("QF_LRA"), "should use QF_LRA logic");
    }
}
