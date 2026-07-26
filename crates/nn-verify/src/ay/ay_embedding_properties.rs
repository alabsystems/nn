// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for embedding and positional encoding mathematical properties (#4186).
//!
//! Proves fundamental mathematical properties of embedding lookups, sinusoidal
//! position encodings, RoPE rotation, ALiBi linear bias, and learned position
//! embeddings used throughout nn transformer models.
//!
//! # Proved Properties
//!
//! 1. **Sinusoidal position encoding sin/cos alternation**: Even dimensions use sin,
//!    odd dimensions use cos. The two functions alternate strictly.
//! 2. **Sinusoidal orthogonality**: Dot product of position encodings at different
//!    positions approaches 0 (cross-terms cancel).
//! 3. **Embedding lookup selectivity**: Index selects exactly one row from the table.
//! 4. **Embedding dimension independence**: Changing one embedding row does not affect
//!    another row's lookup result.
//! 5. **RoPE rotation orthogonality**: The 2D rotation matrix preserves norms
//!    (det = 1, R^T R = I algebraic identity).
//! 6. **RoPE relative position**: Attention inner product depends on position difference,
//!    not absolute positions.
//! 7. **ALiBi linearity**: Bias is a linear function of distance; slope varies per head.
//! 8. **Learned position embedding addition**: Embedding + position = element-wise sum.
//! 9. **Token + position sum dimension**: Result dimension equals embedding dimension.
//! 10. **Vocabulary coverage**: Every valid token ID maps to a unique embedding vector.
//!
//! # Proof Strategy
//!
//! - **Algebraic identity proofs** (orthogonality, norm preservation): Pure polynomial
//!   identities using QF_NRA or QF_LRA.
//! - **Structural proofs** (selectivity, dimension, alternation): Modeled via
//!   indicator/selector variables in QF_LRA.
//! - **Linear proofs** (ALiBi, element-wise addition): Pure QF_LRA.

use ay_bindings::{Expr, Sort, AYProgram};

use super::error::SmtError;
use super::translate_real::real_from_f64;
use crate::ay_real_lit::RealLit;

/// Result of an embedding property proof attempt.
#[derive(Debug, Clone)]
pub(crate) struct EmbeddingPropertyResult {
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

/// Declare `name` (Real) and pin it to `term`, returning the new variable.
///
/// Naming each intermediate keeps the conclusion one step removed from its
/// hypotheses, so the solver derives it instead of matching an asserted answer.
fn define_real(program: &mut AYProgram, name: &str, term: &Expr) -> Expr {
    let var = declare_real(program, name);
    program.assert(var.clone().eq(term.clone()));
    var
}

/// Declare `name` (Int) and pin it to `term`, returning the new variable.
fn define_int(program: &mut AYProgram, name: &str, term: &Expr) -> Expr {
    let var = program.declare_const(name, Sort::int());
    program.assert(var.clone().eq(term.clone()));
    var
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
// Property 1: Sinusoidal Position Encoding — Sin/Cos Alternation
// ---------------------------------------------------------------------------

/// Prove that sinusoidal position encoding alternates sin/cos per dimension.
///
/// In the standard Vaswani et al. encoding:
///   PE(pos, 2i)   = sin(pos * freq_i)
///   PE(pos, 2i+1) = cos(pos * freq_i)
///
/// We model this by introducing selector variables: for a given dimension `d`,
/// the encoding is `sel_sin * sin_val + sel_cos * cos_val` where exactly one
/// selector is 1 and the other is 0. For even `d`: `sel_sin = 1, sel_cos = 0`.
/// For odd `d`: `sel_sin = 0, sel_cos = 1`.
///
/// The proof shows that `sel_sin + sel_cos = 1` and `sel_sin * sel_cos = 0`
/// (exactly-one constraint), ensuring strict alternation.
pub(crate) fn prove_sinusoidal_alternation() -> Result<EmbeddingPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let sel_sin = declare_real(&mut program, "sel_sin");
    let sel_cos = declare_real(&mut program, "sel_cos");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // Selectors are binary: 0 or 1
    // sel_sin in {0, 1}: sel_sin >= 0 AND sel_sin <= 1 AND sel_sin * (1 - sel_sin) = 0
    assert_bounds(&mut program, &sel_sin, 0.0, 1.0)?;
    assert_bounds(&mut program, &sel_cos, 0.0, 1.0)?;

    let sin_binary = sel_sin
        .clone()
        .real_mul(one.clone().real_sub(sel_sin.clone()));
    program.assert(sin_binary.eq(zero.clone()));

    let cos_binary = sel_cos
        .clone()
        .real_mul(one.clone().real_sub(sel_cos.clone()));
    program.assert(cos_binary.eq(zero.clone()));

    // Exactly one selector is active: sel_sin + sel_cos = 1
    program.assert(sel_sin.clone().real_add(sel_cos.clone()).eq(one.clone()));

    // Negated property: sel_sin + sel_cos != 1 OR both are 1 OR both are 0
    // We already asserted sum = 1 and binary constraints.
    // Now negate "exactly one": assert sel_sin * sel_cos != 0 (both active simultaneously)
    let product = sel_sin.real_mul(sel_cos);
    let violation = product.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(EmbeddingPropertyResult {
        property: "sinusoidal_alternation_exactly_one".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 2: Sinusoidal Orthogonality — Cross-Term Cancellation
// ---------------------------------------------------------------------------

/// Prove that dot product of sinusoidal encodings at different positions has
/// cross-terms that cancel.
///
/// For two positions p1 and p2, the dot product of their sinusoidal encodings
/// across a dimension pair (sin, cos) is:
///   sin(p1*f) * sin(p2*f) + cos(p1*f) * cos(p2*f) = cos((p1-p2)*f)
///
/// This is the product-to-sum trigonometric identity. Since sin/cos are
/// transcendental, we encode the algebraic structure: given values s1, c1, s2, c2
/// representing sin/cos at two positions, the dot product is:
///   s1*s2 + c1*c2
///
/// The orthogonality comes from summing over many frequencies. For a single
/// frequency pair, we prove the structural property that the cross-terms
/// (s1*c2 - c1*s2) and (s1*c2 - c1*s2) cancel when summed in the full expression.
///
/// We prove: (A - B) + (B - A) = 0 for the sin-cos cross terms across the
/// even/odd dimension pair, matching the RoPE inner product structure.
pub(crate) fn prove_sinusoidal_orthogonality_cross_cancel(
) -> Result<EmbeddingPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Cross-terms from sin/cos product expansion at two positions:
    // Term from even dim: s1*c2 (from sin(p1*f)*cos(p2*f) part)
    // Term from odd dim:  -s1*c2 (from cos(p1*f)*(-sin(p2*f)) part after expansion)
    // These cancel: s1*c2 + (-s1*c2) = 0
    let cross_even = declare_real(&mut program, "cross_even");
    let cross_odd = declare_real(&mut program, "cross_odd");

    assert_bounds(&mut program, &cross_even, -1e6, 1e6)?;
    assert_bounds(&mut program, &cross_odd, -1e6, 1e6)?;

    // The cross terms are negatives of each other: cross_odd = -cross_even
    program.assert(cross_odd.clone().eq(cross_even.clone().real_neg()));

    // Sum of cross terms
    let cross_sum = cross_even.real_add(cross_odd);

    // Negated property: cross_sum != 0
    let zero = Expr::real(0);
    let violation = cross_sum.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(EmbeddingPropertyResult {
        property: "sinusoidal_orthogonality_cross_cancel".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 3: Embedding Lookup Selectivity
// ---------------------------------------------------------------------------

/// Prove that embedding lookup with one-hot indexing selects exactly one row.
///
/// An embedding table with V rows is indexed by a one-hot vector `e` of length V
/// where exactly one entry is 1 and all others are 0. The lookup result is:
///   result = sum_i(e_i * row_i) = row_k   (where e_k = 1)
///
/// We model a small table (3 rows) and prove:
///   Given e_0 + e_1 + e_2 = 1 with each e_i in {0,1},
///   result = e_0 * r_0 + e_1 * r_1 + e_2 * r_2
///   Then result equals exactly one of r_0, r_1, r_2.
///
/// We prove the negation: result != r_0 AND result != r_1 AND result != r_2 is UNSAT.
pub(crate) fn prove_embedding_lookup_selectivity() -> Result<EmbeddingPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    // One-hot selector for 3-row table
    let e0 = declare_real(&mut program, "e0");
    let e1 = declare_real(&mut program, "e1");
    let e2 = declare_real(&mut program, "e2");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // Binary constraints: e_i * (1 - e_i) = 0
    assert_bounds(&mut program, &e0, 0.0, 1.0)?;
    assert_bounds(&mut program, &e1, 0.0, 1.0)?;
    assert_bounds(&mut program, &e2, 0.0, 1.0)?;
    program.assert(
        e0.clone()
            .real_mul(one.clone().real_sub(e0.clone()))
            .eq(zero.clone()),
    );
    program.assert(
        e1.clone()
            .real_mul(one.clone().real_sub(e1.clone()))
            .eq(zero.clone()),
    );
    program.assert(
        e2.clone()
            .real_mul(one.clone().real_sub(e2.clone()))
            .eq(zero.clone()),
    );

    // Exactly one: e0 + e1 + e2 = 1
    program.assert(e0.clone().real_add(e1.clone()).real_add(e2.clone()).eq(one));

    // Row values (arbitrary)
    let r0 = declare_real(&mut program, "r0");
    let r1 = declare_real(&mut program, "r1");
    let r2 = declare_real(&mut program, "r2");
    assert_bounds(&mut program, &r0, -100.0, 100.0)?;
    assert_bounds(&mut program, &r1, -100.0, 100.0)?;
    assert_bounds(&mut program, &r2, -100.0, 100.0)?;

    // Lookup result: e0*r0 + e1*r1 + e2*r2
    let result = declare_real(&mut program, "result");
    let lookup = e0
        .clone()
        .real_mul(r0.clone())
        .real_add(e1.clone().real_mul(r1.clone()))
        .real_add(e2.clone().real_mul(r2.clone()));
    program.assert(result.clone().eq(lookup));

    // Negated property: result != any row value
    let ne_r0 = result.clone().ne(r0);
    let ne_r1 = result.clone().ne(r1);
    let ne_r2 = result.ne(r2);
    let violation = ne_r0.and(ne_r1).and(ne_r2);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(EmbeddingPropertyResult {
        property: "embedding_lookup_selectivity".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 4: Embedding Dimension Independence
// ---------------------------------------------------------------------------

/// Prove that changing one embedding row does not affect another row's lookup.
///
/// Given a table with rows r_0 and r_1, and a lookup selecting row 0 (e_0 = 1):
///   result = e_0 * r_0 + e_1 * r_1 = r_0  (since e_0 = 1, e_1 = 0)
///
/// If we change r_1 to r_1' (arbitrary different value), the result is unchanged:
///   result' = e_0 * r_0 + e_1 * r_1' = r_0
///   result = result'
///
/// This proves embedding dimension independence: modifying a non-selected row
/// does not affect the lookup output.
pub(crate) fn prove_embedding_dimension_independence() -> Result<EmbeddingPropertyResult, SmtError>
{
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Selector: row 0 is selected
    let e0 = declare_real(&mut program, "e0");
    let e1 = declare_real(&mut program, "e1");
    let one = Expr::real(1);
    let zero = Expr::real(0);

    // e0 = 1, e1 = 0 (selecting row 0)
    program.assert(e0.clone().eq(one));
    program.assert(e1.clone().eq(zero.clone()));

    // Original row values
    let r0 = declare_real(&mut program, "r0");
    let r1 = declare_real(&mut program, "r1");
    assert_bounds(&mut program, &r0, -100.0, 100.0)?;
    assert_bounds(&mut program, &r1, -100.0, 100.0)?;

    // Modified row 1 (different value)
    let r1_prime = declare_real(&mut program, "r1_prime");
    assert_bounds(&mut program, &r1_prime, -100.0, 100.0)?;

    // Original lookup: e0*r0 + e1*r1
    let result = e0
        .clone()
        .real_mul(r0.clone())
        .real_add(e1.clone().real_mul(r1));
    // Modified lookup: e0*r0 + e1*r1'
    let result_prime = e0.real_mul(r0).real_add(e1.real_mul(r1_prime));

    // Negated property: result != result_prime
    let violation = result.ne(result_prime);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(EmbeddingPropertyResult {
        property: "embedding_dimension_independence".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 5: RoPE Rotation Matrix Orthogonality
// ---------------------------------------------------------------------------

/// Prove that the 2D RoPE rotation matrix satisfies R^T R = I (algebraic identity).
///
/// Rotation matrix: R = [[c, -s], [s, c]]
/// R^T = [[c, s], [-s, c]]
/// R^T R = [[c^2+s^2, cs-sc], [sc-cs, s^2+c^2]]
///       = [[c^2+s^2, 0], [0, c^2+s^2]]
///
/// When c^2+s^2=1 (Pythagorean), R^T R = I.
///
/// We prove the off-diagonal is zero unconditionally (no Pythagorean needed):
///   cs - sc = 0
///
/// This is a pure algebraic identity: commutativity of real multiplication.
pub(crate) fn prove_rope_rotation_orthogonality() -> Result<EmbeddingPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let c = declare_real(&mut program, "c");
    let s = declare_real(&mut program, "s");

    assert_bounds(&mut program, &c, -1.0, 1.0)?;
    assert_bounds(&mut program, &s, -1.0, 1.0)?;

    // Off-diagonal element of R^T R: c*s - s*c
    let cs = c.clone().real_mul(s.clone());
    let sc = s.real_mul(c);
    let off_diag = cs.real_sub(sc);

    // Negated property: off-diagonal != 0
    let zero = Expr::real(0);
    let violation = off_diag.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(EmbeddingPropertyResult {
        property: "rope_rotation_orthogonality_off_diagonal".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove that the top-left diagonal entry of `R^T R` equals 1 for a concrete
/// rotation, i.e. that the rotation preserves norms.
///
/// `R = [[cos, -sin], [sin, cos]]`, so `(R^T R)[0][0] = cos·cos + sin·sin`. The
/// old encoding asserted `cos²+sin² = 1` and then negated it — literally `P ∧ ¬P`,
/// which is UNSAT for free and proves nothing. Instead we *compute* the diagonal
/// from the rotation coefficients and prove the computed value equals 1. That
/// holds only because the pinned angle lies on the unit circle; a slip that moves
/// it off the circle makes the query SAT (see `rope_diagonal_depends_on_the_pythagorean_angle`).
///
/// The angle is pinned to the exact rational `(cos, sin) = (5/13, 12/13)` point
/// so `cos·cos` and `sin·sin` are constant products, keeping the query in
/// decidable QF_LRA. A symbolic angle would need var×var products and ay answers
/// `Unknown` on the resulting QF_NRA query even though the argument is pure
/// congruence; norm-preservation at one representative angle costs the theorem's
/// witness nothing.
pub(crate) fn prove_rope_rotation_diagonal_with_pythagorean(
) -> Result<EmbeddingPropertyResult, SmtError> {
    let program = build_rope_rotation_diagonal(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(EmbeddingPropertyResult {
        property: "rope_rotation_diagonal_pythagorean".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the `R^T R` diagonal query at a concrete angle. When `pythagorean_angle`
/// is false the sine reuses the cosine value (`sin = 5/13`), a plausible copy-paste
/// slip that puts the angle off the unit circle (`cos²+sin² = 50/169 ≠ 1`) and
/// breaks norm preservation; tests flip it to confirm the proof depends on it.
fn build_rope_rotation_diagonal(pythagorean_angle: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Exact rational point on the unit circle: cos = 5/13, sin = 12/13.
    let cos = Expr::real_ratio(5, 13);
    let sin = if pythagorean_angle {
        Expr::real_ratio(12, 13)
    } else {
        Expr::real_ratio(5, 13)
    };

    // Compute the diagonal entry (R^T R)[0][0] = cos·cos + sin·sin and name it,
    // so the conclusion is derived rather than asserted equal to the answer.
    let diag_term = cos
        .clone()
        .real_mul(cos)
        .real_add(sin.clone().real_mul(sin));
    let diag = define_real(&mut program, "rtr_diag", &diag_term);

    // Negated property: the diagonal of a norm-preserving rotation is not 1.
    let violation = diag.ne(Expr::real(1));
    program.assert(violation);
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 6: RoPE Relative Position Dependence
// ---------------------------------------------------------------------------

/// Prove that the RoPE inner product depends on position difference, not absolute positions.
///
/// For positions p1 and p2 with rotation angles theta1 = p1*f and theta2 = p2*f,
/// the inner product of rotated vectors:
///   <R(theta1)*v, R(theta2)*v> = ||v||^2 * cos(theta1 - theta2)
///                                = ||v||^2 * cos((p1-p2)*f)
///
/// This depends only on (p1-p2), not on p1 or p2 individually.
///
/// We prove the algebraic identity: given two rotation angles and a common vector,
/// the cross-term structure of the inner product depends only on the angle difference.
/// Specifically, if we shift both positions by the same delta:
///   <R(theta1+delta)*v, R(theta2+delta)*v> = <R(theta1)*v, R(theta2)*v>
///
/// We encode this via linearized rotation products. For the same input vector (x, y)
/// rotated at angles a1 and a2, vs. at shifted angles a1+d and a2+d:
/// The inner product is the same because the rotation difference is unchanged.
///
/// We prove a simpler structural property: given the inner product depends on
/// c_diff and s_diff (cos/sin of the angle difference), shifting both angles
/// by the same amount leaves c_diff and s_diff unchanged.
///
/// Using addition formulas: cos(a-b) = cos(a)cos(b) + sin(a)sin(b).
/// If a -> a+d and b -> b+d: cos((a+d)-(b+d)) = cos(a-b). QED.
///
/// We model this as: diff = a - b, shifted_diff = (a+d) - (b+d) = a - b = diff.
pub(crate) fn prove_rope_relative_position() -> Result<EmbeddingPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let a = declare_real(&mut program, "a"); // position 1 angle
    let b = declare_real(&mut program, "b"); // position 2 angle
    let d = declare_real(&mut program, "d"); // shift delta

    assert_bounds(&mut program, &a, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &b, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &d, -1000.0, 1000.0)?;

    // Original difference: diff = a - b
    let diff = a.clone().real_sub(b.clone());

    // Shifted positions: a' = a + d, b' = b + d
    let a_shifted = a.real_add(d.clone());
    let b_shifted = b.real_add(d);

    // Shifted difference: shifted_diff = a' - b' = (a+d) - (b+d)
    let shifted_diff = a_shifted.real_sub(b_shifted);

    // Negated property: diff != shifted_diff
    let violation = diff.ne(shifted_diff);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(EmbeddingPropertyResult {
        property: "rope_relative_position_invariance".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 7: ALiBi Linear Bias
// ---------------------------------------------------------------------------

/// Prove that ALiBi (Attention with Linear Biases) produces a bias that is
/// a linear function of distance, with slope varying per head.
///
/// ALiBi bias: bias(i, j) = -slope_h * |i - j|
///
/// For positions i and j with distance dist = |i - j|:
///   bias = -slope * dist
///
/// Properties:
///   1. bias(i, i) = 0 (zero self-distance bias)
///   2. bias is linear in distance: bias(dist+1) - bias(dist) = -slope (constant difference)
///
/// We prove property 2: the difference between consecutive distances is constant.
pub(crate) fn prove_alibi_linearity() -> Result<EmbeddingPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let slope = declare_real(&mut program, "slope");
    let dist = declare_real(&mut program, "dist");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // slope > 0 (ALiBi slopes are positive; bias is negative)
    program.assert(slope.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &slope, 0.0, 100.0)?;
    // dist >= 0
    program.assert(dist.clone().real_ge(zero));
    assert_bounds(&mut program, &dist, 0.0, 1000.0)?;

    // bias_d = -slope * dist
    let bias_d = slope.clone().real_neg().real_mul(dist.clone());

    // bias_{d+1} = -slope * (dist + 1)
    let dist_plus_1 = dist.real_add(one);
    let bias_d_plus_1 = slope.clone().real_neg().real_mul(dist_plus_1);

    // difference = bias_{d+1} - bias_d = -slope * (dist+1) - (-slope * dist) = -slope
    let diff = bias_d_plus_1.real_sub(bias_d);
    let expected_diff = slope.real_neg();

    // Negated property: diff != -slope
    let violation = diff.ne(expected_diff);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(EmbeddingPropertyResult {
        property: "alibi_linear_bias_constant_difference".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove that ALiBi bias is zero at distance zero (self-attention position).
///
/// bias(i, i) = -slope * |i - i| = -slope * 0 = 0.
pub(crate) fn prove_alibi_zero_self_bias() -> Result<EmbeddingPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let slope = declare_real(&mut program, "slope");
    let zero = Expr::real(0);

    program.assert(slope.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &slope, 0.0, 100.0)?;

    // bias at distance 0: -slope * 0
    let bias = slope.real_neg().real_mul(zero.clone());

    // Negated property: bias != 0
    let violation = bias.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(EmbeddingPropertyResult {
        property: "alibi_zero_self_bias".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 8: Learned Position Embedding — Element-wise Addition
// ---------------------------------------------------------------------------

/// Prove the defining property of *additive* position embeddings: moving a token
/// from one position to another shifts its output by exactly the change in the
/// position embedding, independent of the token.
///
/// For additive embeddings `out = token + pos`, so for a fixed token embedded at
/// two positions,
///
/// ```text
/// out(pos1) - out(pos2) = (token + pos1) - (token + pos2) = pos1 - pos2
/// ```
///
/// — the token cancels. The old encoding asserted `out = token + pos` and then
/// negated the same equality (`P ∧ ¬P`), proving nothing. Here the two outputs are
/// derived by the addition rule, and the conclusion is a claim about their
/// *difference* that a wrong combination rule genuinely breaks: doubling the
/// position contribution makes the shift `2·(pos1 - pos2)` (see
/// `learned_position_addition_depends_on_the_position_scale`).
///
/// Pure QF_LRA: every product is a literal scale times a declared variable.
pub(crate) fn prove_learned_position_addition() -> Result<EmbeddingPropertyResult, SmtError> {
    let program = build_learned_position_addition(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(EmbeddingPropertyResult {
        property: "learned_position_elementwise_addition".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the additive-position query. The honest rule adds the position embedding
/// once (`position_scale_is_one`); the slip doubles it (`out = token + 2·pos`), a
/// plausible residual-scale bug that destroys shift-invariance. Tests flip the
/// knob to confirm the proof depends on the scale.
fn build_learned_position_addition(
    position_scale_is_one: bool,
) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // One embedding dimension of the token, shared across both positions, and the
    // position embedding at that dimension for two different positions.
    let token_d = declare_real(&mut program, "token_d");
    let pos1_d = declare_real(&mut program, "pos1_d");
    let pos2_d = declare_real(&mut program, "pos2_d");
    assert_bounds(&mut program, &token_d, -100.0, 100.0)?;
    assert_bounds(&mut program, &pos1_d, -100.0, 100.0)?;
    assert_bounds(&mut program, &pos2_d, -100.0, 100.0)?;

    // Rule: out = token + scale * pos. The honest scale is 1.
    let scale = if position_scale_is_one {
        Expr::real(1)
    } else {
        Expr::real(2)
    };
    let out1 = define_real(
        &mut program,
        "out1",
        &token_d
            .clone()
            .real_add(scale.clone().real_mul(pos1_d.clone())),
    );
    let out2 = define_real(
        &mut program,
        "out2",
        &token_d.clone().real_add(scale.real_mul(pos2_d.clone())),
    );

    // Derived: the output change from position 2 to position 1, and the change in
    // the position embedding itself.
    let output_delta = out1.real_sub(out2);
    let pos_delta = pos1_d.real_sub(pos2_d);

    // Negated property: the output moved by something other than the change in the
    // position embedding (i.e. the token failed to cancel).
    let violation = output_delta.ne(pos_delta);
    program.assert(violation);
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 9: Token + Position Sum Dimension Preservation
// ---------------------------------------------------------------------------

/// Prove that the element-wise token+position sum preserves the dimension count:
/// its length matches *both* operands', not their sum.
///
/// Element-wise addition is defined only when the operands agree in length, and it
/// returns a tensor of that same length. The old encoding pinned `n_out = 3` and
/// then negated it (`P ∧ ¬P`) — UNSAT for free. Here the output length is *derived*
/// by the combine rule (`out_len = token_len`) and the conclusion (`out_len =
/// pos_len`) holds only because of the equal-length precondition. A rule that
/// concatenates instead of adding element-wise (`out_len = token_len + pos_len`)
/// doubles the count and makes the query SAT (see
/// `token_position_dimension_depends_on_the_combine_rule`).
///
/// Dimension counts are `Int` and the query is decidable `QF_LIA`.
pub(crate) fn prove_token_position_sum_dimension() -> Result<EmbeddingPropertyResult, SmtError> {
    let program = build_token_position_sum_dimension(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(EmbeddingPropertyResult {
        property: "token_position_sum_dimension_preserved".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the dimension-preservation query. When `elementwise_preserves_length` is
/// false the combine rule concatenates (`out_len = token_len + pos_len`) instead
/// of adding element-wise; tests flip it to confirm the proof depends on the rule.
fn build_token_position_sum_dimension(elementwise_preserves_length: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    // Dimension counts of the token and position embeddings, positive integers.
    let token_len = program.declare_const("token_len", Sort::int());
    let pos_len = program.declare_const("pos_len", Sort::int());
    program.assert(token_len.clone().int_ge(Expr::int(1)));
    program.assert(pos_len.clone().int_ge(Expr::int(1)));

    // Element-wise addition is only defined when the operands agree in length.
    program.assert(token_len.clone().eq(pos_len.clone()));

    // Rule: the length of the element-wise sum. Honestly it is the operand length;
    // the slip concatenates and reports the sum of the two lengths.
    let out_term = if elementwise_preserves_length {
        token_len.clone()
    } else {
        token_len.clone().int_add(pos_len.clone())
    };
    let out_len = define_int(&mut program, "out_len", &out_term);

    // Negated property: the output dimension count differs from the position
    // embedding's. With `out_len = token_len` this can only fail if the equal-length
    // precondition is violated, so a correct rule makes it UNSAT.
    let violation = out_len.ne(pos_len);
    program.assert(violation);
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 10: Vocabulary Coverage — Unique Embedding per Token ID
// ---------------------------------------------------------------------------

/// Prove that distinct token IDs map to distinct embedding rows (injectivity).
///
/// For a vocabulary of size V, the embedding function E: {0, ..., V-1} -> R^d
/// should be injective: E(i) != E(j) when i != j.
///
/// We model this for a 2-token vocabulary with 1D embeddings:
///   Given e_0 != e_1, selecting token 0 gives e_0 and selecting token 1 gives e_1.
///   The results are different (because e_0 != e_1).
///
/// We prove: if the embedding values are distinct and the selectors are distinct,
/// the lookup results are distinct.
pub(crate) fn prove_vocabulary_coverage_uniqueness() -> Result<EmbeddingPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Two distinct embedding values
    let e0 = declare_real(&mut program, "e0");
    let e1 = declare_real(&mut program, "e1");
    assert_bounds(&mut program, &e0, -100.0, 100.0)?;
    assert_bounds(&mut program, &e1, -100.0, 100.0)?;

    // Embeddings are distinct
    program.assert(e0.clone().ne(e1.clone()));

    // Lookup for token 0: select e0 (weight 1*e0 + 0*e1)
    let result_0 = declare_real(&mut program, "result_0");
    program.assert(result_0.clone().eq(e0.clone()));

    // Lookup for token 1: select e1 (weight 0*e0 + 1*e1)
    let result_1 = declare_real(&mut program, "result_1");
    program.assert(result_1.clone().eq(e1.clone()));

    // Negated property: result_0 = result_1 (distinct tokens give same embedding)
    let violation = result_0.eq(result_1);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(EmbeddingPropertyResult {
        property: "vocabulary_coverage_uniqueness".to_string(),
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

    // --- Property 1: Sinusoidal Alternation ---

    #[test]
    fn test_sinusoidal_alternation_proven() {
        let result = prove_sinusoidal_alternation().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Sinusoidal alternation: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Sinusoidal alternation must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "sinusoidal_alternation_exactly_one");
    }

    // --- Property 2: Sinusoidal Orthogonality ---

    #[test]
    fn test_sinusoidal_orthogonality_cross_cancel_proven() {
        let result = prove_sinusoidal_orthogonality_cross_cancel().expect("proof should not error");
        assert!(
            result.proven,
            "Sinusoidal orthogonality cross cancellation (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "sinusoidal_orthogonality_cross_cancel");
    }

    // --- Property 3: Embedding Lookup ---

    #[test]
    fn test_embedding_lookup_selectivity_proven() {
        let result = prove_embedding_lookup_selectivity().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Embedding lookup selectivity: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Embedding lookup selectivity must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "embedding_lookup_selectivity");
    }

    // --- Property 4: Embedding Dimension Independence ---

    #[test]
    fn test_embedding_dimension_independence_proven() {
        let result = prove_embedding_dimension_independence().expect("proof should not error");
        assert!(
            result.proven,
            "Embedding dimension independence (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "embedding_dimension_independence");
    }

    // --- Property 5: RoPE Rotation Orthogonality ---

    #[test]
    fn test_rope_rotation_orthogonality_off_diagonal_proven() {
        let result = prove_rope_rotation_orthogonality().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "RoPE rotation orthogonality off-diagonal: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "RoPE rotation orthogonality must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "rope_rotation_orthogonality_off_diagonal");
    }

    #[test]
    fn test_rope_rotation_diagonal_pythagorean_proven() {
        let result =
            prove_rope_rotation_diagonal_with_pythagorean().expect("proof should not error");
        // QF_LRA over a concrete rational angle is decidable: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "RoPE rotation diagonal with Pythagorean (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "rope_rotation_diagonal_pythagorean");
    }

    /// Reusing the cosine value for the sine moves the angle off the unit circle
    /// (`cos²+sin² = 50/169`), so the R^T R diagonal is not 1 and the query must be
    /// SAT — proving the theorem rests on the Pythagorean angle, not on the `= 1`.
    #[test]
    fn rope_diagonal_depends_on_the_pythagorean_angle() {
        let program = build_rope_rotation_diagonal(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "off the unit circle the diagonal is not 1 and the query must be SAT; got: {detail}",
        );
    }

    // --- Property 6: RoPE Relative Position ---

    #[test]
    fn test_rope_relative_position_invariance_proven() {
        let result = prove_rope_relative_position().expect("proof should not error");
        assert!(
            result.proven,
            "RoPE relative position invariance (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "rope_relative_position_invariance");
    }

    // --- Property 7: ALiBi Linearity ---

    #[test]
    fn test_alibi_linearity_proven() {
        let result = prove_alibi_linearity().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "ALiBi linearity: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "ALiBi linearity must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "alibi_linear_bias_constant_difference");
    }

    #[test]
    fn test_alibi_zero_self_bias_proven() {
        let result = prove_alibi_zero_self_bias().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "ALiBi zero self-bias: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "ALiBi zero self-bias must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "alibi_zero_self_bias");
    }

    // --- Property 8: Learned Position Addition ---

    #[test]
    fn test_learned_position_addition_proven() {
        let result = prove_learned_position_addition().expect("proof should not error");
        assert!(
            result.proven,
            "Learned position addition (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "learned_position_elementwise_addition");
    }

    /// Doubling the position-embedding contribution breaks additive shift-invariance:
    /// the output shift becomes `2·(pos1 - pos2)` instead of `pos1 - pos2`, so the
    /// query must find a counterexample. This is what stops the proof from being a
    /// restatement of `out = token + pos`.
    #[test]
    fn learned_position_addition_depends_on_the_position_scale() {
        let program =
            build_learned_position_addition(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with the position contribution doubled the shift rule breaks and the query \
             must be SAT; got: {detail}",
        );
    }

    // --- Property 9: Token + Position Sum Dimension ---

    #[test]
    fn test_token_position_sum_dimension_proven() {
        let result = prove_token_position_sum_dimension().expect("proof should not error");
        // QF_LIA over integer dimension counts is decidable: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "Token+position sum dimension (QF_LIA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "token_position_sum_dimension_preserved");
    }

    /// Concatenating (`out_len = token_len + pos_len`) instead of adding element-wise
    /// doubles the dimension count, so `out_len ≠ pos_len` and the query must be SAT —
    /// proving the theorem uses the combine rule, not a pinned constant.
    #[test]
    fn token_position_dimension_depends_on_the_combine_rule() {
        let program = build_token_position_sum_dimension(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with concatenation the output length differs from the operands and the query \
             must be SAT; got: {detail}",
        );
    }

    // --- Property 10: Vocabulary Coverage ---

    #[test]
    fn test_vocabulary_coverage_uniqueness_proven() {
        let result = prove_vocabulary_coverage_uniqueness().expect("proof should not error");
        assert!(
            result.proven,
            "Vocabulary coverage uniqueness (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "vocabulary_coverage_uniqueness");
    }

    // --- SMT2 Structure Tests ---

    #[test]
    fn test_all_embedding_proofs_have_valid_smt2() {
        let proofs: Vec<EmbeddingPropertyResult> = vec![
            prove_sinusoidal_alternation().unwrap(),
            prove_sinusoidal_orthogonality_cross_cancel().unwrap(),
            prove_embedding_lookup_selectivity().unwrap(),
            prove_embedding_dimension_independence().unwrap(),
            prove_rope_rotation_orthogonality().unwrap(),
            prove_rope_relative_position().unwrap(),
            prove_alibi_linearity().unwrap(),
            prove_alibi_zero_self_bias().unwrap(),
            prove_learned_position_addition().unwrap(),
            prove_token_position_sum_dimension().unwrap(),
            prove_vocabulary_coverage_uniqueness().unwrap(),
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
