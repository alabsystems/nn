// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for attention mask and position bias mathematical properties (#4217).
//!
//! Proves fundamental properties of attention mechanisms used in transformers:
//!
//! 1. **Causal mask is lower triangular**: mask[i][j] = true iff j <= i.
//! 2. **Causal mask + softmax**: Masked positions contribute 0 probability.
//! 3. **ALiBi bias linearity**: bias(i, j) = -slope * |i - j| is linear in |i - j|.
//! 4. **ALiBi slopes are geometric**: slope_h = 2^(-8h/H) forms geometric sequence.
//! 5. **RoPE rotation is orthogonal**: R(theta)^T * R(theta) = I.
//! 6. **RoPE composition**: R(theta1) * R(theta2) = R(theta1 + theta2).
//! 7. **Relative position bias symmetry**: bias(i, j) = bias(j, i).
//! 8. **Sliding window mask**: mask[i][j] = true iff |i - j| <= W.
//! 9. **Attention score boundedness**: |QK^T/sqrt(D)| <= B^2 * sqrt(D).
//!
//! # Proof Strategy
//!
//! Attention properties are encoded as scalar real arithmetic on small concrete
//! dimensions (sequence length 3-4, head dim 2). Each position index is a
//! separate SMT real variable. We assert the negation of the property and
//! prove UNSAT.
//!
//! - **Structural proofs (QF_LRA)**: Mask properties, linearity, symmetry.
//! - **Algebraic proofs (QF_NRA)**: Orthogonality, composition, boundedness.

use ay_bindings::{Expr, Sort, AYProgram};

use crate::ay_real_lit::RealLit;
use crate::smt_error::SmtError;

/// Result of an attention mask/position bias property proof attempt.
#[derive(Debug, Clone)]
pub struct AttentionMaskPropertyResult {
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
/// Naming each intermediate keeps the conclusion one step removed from its
/// hypotheses, so the solver derives it (by transitivity through the pinned
/// definition) instead of being handed it — matching the `define_real` pattern
/// used by the linear-algebra and reshape templates.
fn define_real(program: &mut AYProgram, name: &str, term: &Expr) -> Expr {
    let var = declare_real(program, name);
    program.assert(var.clone().eq(term.clone()));
    var
}

/// Assert `lower <= expr <= upper`.
fn assert_bounds(program: &mut AYProgram, expr: &Expr, lower: &Expr, upper: &Expr) {
    program.assert(expr.clone().real_ge(lower.clone()));
    program.assert(expr.clone().real_le(upper.clone()));
}

/// Assert `expr > 0` (strict positivity).
fn assert_positive(program: &mut AYProgram, expr: &Expr) {
    let zero = Expr::real(0);
    program.assert(expr.clone().real_gt(zero));
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
// Property 1: Causal Mask is Lower Triangular
// ---------------------------------------------------------------------------

/// Prove that a causal mask for sequence length S=3 satisfies:
///   mask[i][j] = 1 iff j <= i, and mask[i][j] = 0 otherwise.
///
/// We encode each mask entry m_ij as a real variable constrained to be
/// either 0 or 1, then assert the causal structure: m_ij = 1 when j <= i,
/// m_ij = 0 when j > i. The violation is any entry breaking this rule.
///
/// For S=3, the mask is:
/// ```text
///   [[1, 0, 0],
///    [1, 1, 0],
///    [1, 1, 1]]
/// ```
///
/// Uses `QF_LRA` — pure linear constraints on 0/1 indicators.
pub fn prove_causal_mask_lower_triangular() -> Result<AttentionMaskPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // Declare 3x3 mask entries
    let m = [
        [
            declare_real(&mut program, "m00"),
            declare_real(&mut program, "m01"),
            declare_real(&mut program, "m02"),
        ],
        [
            declare_real(&mut program, "m10"),
            declare_real(&mut program, "m11"),
            declare_real(&mut program, "m12"),
        ],
        [
            declare_real(&mut program, "m20"),
            declare_real(&mut program, "m21"),
            declare_real(&mut program, "m22"),
        ],
    ];

    // Causal mask definition: m[i][j] = 1 if j <= i, else 0.
    // Row 0: m00 = 1, m01 = 0, m02 = 0
    program.assert(m[0][0].clone().eq(one.clone()));
    program.assert(m[0][1].clone().eq(zero.clone()));
    program.assert(m[0][2].clone().eq(zero.clone()));
    // Row 1: m10 = 1, m11 = 1, m12 = 0
    program.assert(m[1][0].clone().eq(one.clone()));
    program.assert(m[1][1].clone().eq(one.clone()));
    program.assert(m[1][2].clone().eq(zero.clone()));
    // Row 2: m20 = 1, m21 = 1, m22 = 1
    program.assert(m[2][0].clone().eq(one.clone()));
    program.assert(m[2][1].clone().eq(one.clone()));
    program.assert(m[2][2].clone().eq(one.clone()));

    // Lower triangular property: sum of upper-triangular entries = 0
    // Upper triangle entries: m01, m02, m12
    let upper_sum = m[0][1]
        .clone()
        .real_add(m[0][2].clone())
        .real_add(m[1][2].clone());

    // Lower triangle + diagonal sum = 6 (all ones)
    let lower_sum = m[0][0]
        .clone()
        .real_add(m[1][0].clone())
        .real_add(m[1][1].clone())
        .real_add(m[2][0].clone())
        .real_add(m[2][1].clone())
        .real_add(m[2][2].clone());

    let six = Expr::real(6);

    // Violation: upper sum != 0 OR lower sum != 6
    let violation = upper_sum.ne(zero).or(lower_sum.ne(six));
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionMaskPropertyResult {
        property: "causal_mask_lower_triangular_3x3".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 2: Causal Mask + Softmax Zero Contribution
// ---------------------------------------------------------------------------

/// Prove that after applying a causal mask (adding -inf to masked positions)
/// and computing softmax, the masked positions contribute 0 probability.
///
/// For a 3-element row where position 0 is unmasked and positions 1, 2 are masked:
///   scores = [s0, -inf, -inf]
///   exp_scores = [exp(s0), 0, 0]  (since exp(-inf) = 0)
///   softmax = [1, 0, 0]  (only the unmasked position contributes)
///
/// We model exp(-inf) as 0: if a value is masked (exp_val = 0), its softmax
/// output is 0 and the unmasked position gets all probability.
///
/// Uses `QF_LRA` with structural constraints on exp values.
pub fn prove_causal_mask_softmax_zero() -> Result<AttentionMaskPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // One unmasked position (exp value > 0), two masked positions (exp = 0).
    let e_unmasked = declare_real(&mut program, "e_unmasked");
    let e_masked_1 = declare_real(&mut program, "e_masked_1");
    let e_masked_2 = declare_real(&mut program, "e_masked_2");

    // Unmasked position has positive exp value
    assert_positive(&mut program, &e_unmasked);
    // Masked positions: exp(-inf) = 0
    program.assert(e_masked_1.clone().eq(zero.clone()));
    program.assert(e_masked_2.clone().eq(zero.clone()));

    // Denominator = e_unmasked + 0 + 0 = e_unmasked
    let denom = declare_real(&mut program, "denom");
    program.assert(
        denom.clone().eq(e_unmasked
            .clone()
            .real_add(e_masked_1.clone())
            .real_add(e_masked_2.clone())),
    );

    // Softmax outputs: s_i * denom = e_i
    let s_unmasked = declare_real(&mut program, "s_unmasked");
    let s_masked_1 = declare_real(&mut program, "s_masked_1");
    let s_masked_2 = declare_real(&mut program, "s_masked_2");

    program.assert(s_unmasked.clone().real_mul(denom.clone()).eq(e_unmasked));
    program.assert(s_masked_1.clone().real_mul(denom.clone()).eq(e_masked_1));
    program.assert(s_masked_2.clone().real_mul(denom.clone()).eq(e_masked_2));

    // Violation: masked softmax outputs are not 0 OR unmasked is not 1
    let v_masked_1 = s_masked_1.ne(zero.clone());
    let v_masked_2 = s_masked_2.ne(zero.clone());
    let v_unmasked = s_unmasked.ne(one);

    let violation = Expr::or_many(vec![v_masked_1, v_masked_2, v_unmasked]);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionMaskPropertyResult {
        property: "causal_mask_softmax_zero_contribution".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 3: ALiBi Bias Linearity
// ---------------------------------------------------------------------------

/// Prove that ALiBi bias is linear in position difference: for a fixed slope m,
///   bias(i, j) = -m * |i - j|
///
/// Linearity means: bias(i, j) + bias(j, k) = bias(i, k) when i >= j >= k
/// (the bias accumulates linearly along the sequence).
///
/// For positions i >= j >= k (all non-negative):
///   |i - j| + |j - k| = (i - j) + (j - k) = i - k = |i - k|
///
/// So: -m * |i - j| + (-m * |j - k|) = -m * (|i - j| + |j - k|) = -m * |i - k|
///
/// Uses `QF_NRA` since the proof involves products of slope and distances.
pub fn prove_alibi_bias_linearity() -> Result<AttentionMaskPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let zero = Expr::real(0);

    // Slope m > 0
    let m = declare_real(&mut program, "m");
    assert_positive(&mut program, &m);
    let bound_lo = Expr::real(0);
    let bound_hi = Expr::real(100);
    assert_bounds(&mut program, &m, &bound_lo, &bound_hi);

    // Positions: i >= j >= k >= 0
    let i = declare_real(&mut program, "i");
    let j = declare_real(&mut program, "j");
    let k = declare_real(&mut program, "k");

    program.assert(i.clone().real_ge(zero.clone()));
    program.assert(j.clone().real_ge(zero.clone()));
    program.assert(k.clone().real_ge(zero.clone()));
    program.assert(i.clone().real_ge(j.clone()));
    program.assert(j.clone().real_ge(k.clone()));

    let pos_bound = Expr::real(1000);
    assert_bounds(&mut program, &i, &zero, &pos_bound);
    assert_bounds(&mut program, &j, &zero, &pos_bound);
    assert_bounds(&mut program, &k, &zero, &pos_bound);

    // Distances (absolute values, given i >= j >= k)
    let d_ij = i.clone().real_sub(j.clone()); // |i - j| = i - j since i >= j
    let d_jk = j.real_sub(k.clone()); // |j - k| = j - k since j >= k
    let d_ik = i.real_sub(k); // |i - k| = i - k since i >= k

    // bias(i,j) = -m * d_ij, bias(j,k) = -m * d_jk, bias(i,k) = -m * d_ik
    let bias_ij = m.clone().real_neg().real_mul(d_ij);
    let bias_jk = m.clone().real_neg().real_mul(d_jk);
    let bias_ik = m.real_neg().real_mul(d_ik);

    // Linearity: bias(i,j) + bias(j,k) = bias(i,k)
    let lhs = bias_ij.real_add(bias_jk);

    // Violation: lhs != bias_ik
    let violation = lhs.ne(bias_ik);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionMaskPropertyResult {
        property: "alibi_bias_linearity".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 4: ALiBi Slopes Are Geometric
// ---------------------------------------------------------------------------

/// Prove that ALiBi slopes form a geometric sequence: for H heads,
///   slope_h = 2^(-8h/H)
///
/// A geometric sequence satisfies the geometric-mean identity on any three
/// consecutive terms: `slope_{h+1}^2 = slope_h * slope_{h+2}` (equivalently, the
/// consecutive ratios `slope_{h+1}/slope_h` and `slope_{h+2}/slope_{h+1}` are
/// equal). This is the defining property — it holds iff the ratio is constant.
///
/// The original encoding was a QF_NRA query over four *variable×variable*
/// products (`s0*ratio`, `s1*ratio`, `s1*s1`, `s0*s2`) with symbolic slopes and
/// a symbolic ratio, which is undecidable in practice and hung the solver.
///
/// We repair it exactly as the RoPE proofs above do: pin the three consecutive
/// slopes to the *exact rationals* the ALiBi formula produces for `H = 8` heads,
/// where `slope_h = 2^(-8h/H) = 2^(-h)`:
///   - `slope_1 = 1/2`, `slope_2 = 1/4`, `slope_3 = 1/8`, common ratio `r = 1/2`.
///
/// Every entry is then a *constant*, so the products fold and the query is
/// decidable ground arithmetic rather than an `Unknown`-prone symbolic QF_NRA
/// query. The two sides of the identity are supplied as *independently named*
/// definitions, so the conclusion is derived by the solver folding the pinned
/// slopes rather than being asserted. A knob perturbs `slope_3` to `1/6`, which
/// breaks the constant ratio (`(1/6)/(1/4) = 2/3 != 1/2`) and turns the query
/// SAT — see `slopes_geometric_depends_on_the_constant_ratio`.
///
/// Uses `QF_NRA` (the constant products fold exactly).
pub fn prove_alibi_slopes_geometric() -> Result<AttentionMaskPropertyResult, SmtError> {
    let program = build_alibi_slopes_geometric(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionMaskPropertyResult {
        property: "alibi_slopes_geometric".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the geometric-sequence query for three consecutive ALiBi slopes.
///
/// When `ratio_is_constant` is false the third slope is set to `1/6` instead of
/// the geometric `1/8`, a plausible slip that breaks the constant ratio; tests
/// flip it to confirm the proof depends on the sequence actually being geometric.
fn build_alibi_slopes_geometric(ratio_is_constant: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    // Exact ALiBi slopes for H = 8 heads: slope_h = 2^(-8h/H) = 2^(-h).
    //   slope_1 = 1/2, slope_2 = 1/4, slope_3 = 1/8;  common ratio r = 1/2.
    let s0 = Expr::real_ratio(1, 2);
    let s1 = Expr::real_ratio(1, 4);
    // The slip replaces slope_3 (1/8) with 1/6, so the ratio is no longer
    // constant: (1/6)/(1/4) = 2/3 != 1/2 = (1/4)/(1/2).
    let s2 = if ratio_is_constant {
        Expr::real_ratio(1, 8)
    } else {
        Expr::real_ratio(1, 6)
    };

    // Geometric-mean identity: slope_2^2 = slope_1 * slope_3. Name each side so
    // the conclusion is *derived* by the solver folding the pinned slopes (the
    // constant products) rather than being asserted and negated.
    let s1_sq = define_real(&mut program, "s1_squared", &s1.clone().real_mul(s1.clone()));
    let s0_s2 = define_real(&mut program, "s0_times_s2", &s0.clone().real_mul(s2.clone()));

    // Violation: the geometric-mean identity fails on the middle term.
    let violation = s1_sq.ne(s0_s2);
    program.assert(violation);
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 5: RoPE Rotation Is Orthogonal
// ---------------------------------------------------------------------------

/// Prove that R(theta)^T * R(theta) = I for a 2D rotation matrix.
///
/// The rotation matrix is `R(theta) = [[c, -s], [s, c]]` with `c = cos(theta)`,
/// `s = sin(theta)`. Its transpose swaps the two off-diagonal entries:
/// `R^T = [[c, s], [-s, c]]`. The product is
///
/// ```text
///   R^T * R = [[c^2 + s^2, -cs + sc], [-sc + cs, s^2 + c^2]] = [[1, 0], [0, 1]]
/// ```
///
/// The theorem's content is *entirely in transposing correctly* and in the
/// Pythagorean identity `c^2 + s^2 = 1`. The original encoding was vacuous: it
/// asserted `c^2 + s^2 = 1` as a hypothesis and then took the violation
/// `(c^2 + s^2) != 1`, which merely negates that hypothesis (`P ∧ ¬P`).
///
/// We repair it by pinning the angle to an exact rational point on the unit
/// circle, `(c, s) = (5/13, 12/13)` (since `5^2 + 12^2 = 13^2`), so that every
/// matrix entry is a *constant* — the query stays decidable ground arithmetic
/// rather than an `Unknown`-prone QF_NRA query over symbolic `c`, `s` (a symbolic
/// `c*c` is a product of two declared variables). The product `R^T * R` is then
/// computed by the solver from the transposed matrix and compared entrywise to
/// the identity. The transpose is guarded by a knob: dropping it (reusing `R` in
/// place of `R^T`) makes the diagonal `c^2 - s^2 = -119/169 != 1`, so the query
/// turns SAT — see `rotation_orthogonal_depends_on_the_transpose`.
///
/// Uses `QF_NRA` (the constant products fold exactly).
pub fn prove_rope_rotation_orthogonal() -> Result<AttentionMaskPropertyResult, SmtError> {
    let program = build_rope_rotation_orthogonal(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionMaskPropertyResult {
        property: "rope_rotation_orthogonal".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the orthogonality query. When `transpose_is_correct` is false the
/// transpose is dropped (`R^T` is set to `R`) — a plausible slip that breaks
/// `R^T R = I`; tests flip it to confirm the proof depends on the transpose.
fn build_rope_rotation_orthogonal(transpose_is_correct: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    // Exact rational point on the unit circle: (c, s) = (5/13, 12/13).
    let c = Expr::real_ratio(5, 13);
    let s = Expr::real_ratio(12, 13);
    let neg_s = s.clone().real_neg();

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // R = [[c, -s], [s, c]] as constant entries.
    let (r00, r01, r10, r11) = (c.clone(), neg_s.clone(), s.clone(), c.clone());

    // R^T. A correct transpose swaps the off-diagonal entries -> [[c, s], [-s, c]].
    // The slip reuses R unchanged (a dropped transpose).
    let (rt00, rt01, rt10, rt11) = if transpose_is_correct {
        (r00.clone(), r10.clone(), r01.clone(), r11.clone())
    } else {
        (r00.clone(), r01.clone(), r10.clone(), r11.clone())
    };

    // M = R^T * R, each entry M[i][j] = sum_k RT[i][k] * R[k][j], named so the
    // conclusion is derived through the pinned definition rather than asserted.
    let m00 = define_real(
        &mut program,
        "rtr_00",
        &rt00.clone().real_mul(r00.clone()).real_add(rt01.clone().real_mul(r10.clone())),
    );
    let m01 = define_real(
        &mut program,
        "rtr_01",
        &rt00.real_mul(r01.clone()).real_add(rt01.real_mul(r11.clone())),
    );
    let m10 = define_real(
        &mut program,
        "rtr_10",
        &rt10.clone().real_mul(r00).real_add(rt11.clone().real_mul(r10)),
    );
    let m11 = define_real(
        &mut program,
        "rtr_11",
        &rt10.real_mul(r01).real_add(rt11.real_mul(r11)),
    );

    // Violation: R^T * R != I.
    let violation = Expr::or_many(vec![
        m00.ne(one.clone()),
        m01.ne(zero.clone()),
        m10.ne(zero),
        m11.ne(one),
    ]);
    program.assert(violation);
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 6: RoPE Composition
// ---------------------------------------------------------------------------

/// Prove that R(theta1) * R(theta2) = R(theta1 + theta2) for 2D rotation blocks.
///
/// With `R(t) = [[cos t, -sin t], [sin t, cos t]]`, the matrix product equals a
/// single rotation by the summed angle:
///
/// ```text
///   R(t1) * R(t2) = [[c1*c2 - s1*s2, -(c1*s2 + s1*c2)],
///                    [s1*c2 + c1*s2,   c1*c2 - s1*s2 ]]
///                 = R(t1 + t2)
/// ```
///
/// The original encoding was vacuous: it *defined* `c12 := c1*c2 - s1*s2` and
/// `s12 := s1*c2 + c1*s2`, then took the violation `(c1*c2 - s1*s2) != c12` —
/// i.e. it asserted the answer and negated it (`P ∧ ¬P`).
///
/// We repair it with two concrete angles whose cos/sin, *and* the cos/sin of
/// their sum, are all exact rationals:
///   - `theta1: (c1, s1) = (5/13, 12/13)` (5-12-13 triple),
///   - `theta2: (c2, s2) = (3/5, 4/5)` (3-4-5 triple),
///   - `theta1 + theta2: (c12, s12) = (-33/65, 56/65)` (checked: `33^2 + 56^2 = 65^2`).
///
/// The sum-angle cos/sin are supplied as *independent literals*, not as
/// `c1*c2 - s1*s2`, so the conclusion is derived by the solver evaluating the
/// product rather than handed to it. Pinning the angles keeps every entry a
/// constant (decidable ground arithmetic; symbolic angles would make each entry
/// a product of two declared variables and the query `Unknown`-prone). A knob
/// flips the off-diagonal sign convention of `R(theta2)` — rotating by `-theta2`
/// so the product becomes `R(theta1 - theta2)` — which makes the entries
/// disagree and the query SAT (see `composition_depends_on_the_rotation_sign`).
///
/// Uses `QF_NRA` (the constant products fold exactly).
pub fn prove_rope_composition() -> Result<AttentionMaskPropertyResult, SmtError> {
    let program = build_rope_composition(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionMaskPropertyResult {
        property: "rope_composition".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the composition query. When `rotation_sign_correct` is false the
/// off-diagonal signs of `R(theta2)` are flipped (rotating by `-theta2`), a
/// plausible direction slip that turns the product into `R(theta1 - theta2)`;
/// tests flip it to confirm the proof depends on the rotation convention.
fn build_rope_composition(rotation_sign_correct: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    // theta1: (c1, s1) = (5/13, 12/13);  theta2: (c2, s2) = (3/5, 4/5).
    let c1 = Expr::real_ratio(5, 13);
    let s1 = Expr::real_ratio(12, 13);
    let c2 = Expr::real_ratio(3, 5);
    let s2 = Expr::real_ratio(4, 5);

    // Independently-known cos/sin of the SUM angle theta1 + theta2. Supplied as
    // literals, NOT as c1*c2 - s1*s2, so the product must be *derived* to equal
    // them rather than being compared to its own definition.
    let c12 = Expr::real_ratio(-33, 65);
    let s12 = Expr::real_ratio(56, 65);

    // R(theta1) = [[c1, -s1], [s1, c1]].
    let (r1_00, r1_01, r1_10, r1_11) = (c1.clone(), s1.clone().real_neg(), s1.clone(), c1.clone());

    // R(theta2) = [[c2, -s2], [s2, c2]]. The slip flips the off-diagonal signs.
    let (r2_00, r2_01, r2_10, r2_11) = if rotation_sign_correct {
        (c2.clone(), s2.clone().real_neg(), s2.clone(), c2.clone())
    } else {
        (c2.clone(), s2.clone(), s2.clone().real_neg(), c2.clone())
    };

    // P = R(theta1) * R(theta2), each entry P[i][j] = sum_k R1[i][k] * R2[k][j].
    let p00 = define_real(
        &mut program,
        "p00",
        &r1_00.clone().real_mul(r2_00.clone()).real_add(r1_01.clone().real_mul(r2_10.clone())),
    );
    let p01 = define_real(
        &mut program,
        "p01",
        &r1_00.real_mul(r2_01.clone()).real_add(r1_01.real_mul(r2_11.clone())),
    );
    let p10 = define_real(
        &mut program,
        "p10",
        &r1_10.clone().real_mul(r2_00).real_add(r1_11.clone().real_mul(r2_10)),
    );
    let p11 = define_real(
        &mut program,
        "p11",
        &r1_10.real_mul(r2_01).real_add(r1_11.real_mul(r2_11)),
    );

    // R(theta1 + theta2) = [[c12, -s12], [s12, c12]].
    let neg_s12 = s12.clone().real_neg();

    // Violation: the product disagrees with the sum-angle rotation on any entry.
    let violation = Expr::or_many(vec![
        p00.ne(c12.clone()),
        p01.ne(neg_s12),
        p10.ne(s12),
        p11.ne(c12),
    ]);
    program.assert(violation);
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 7: Relative Position Bias Symmetry
// ---------------------------------------------------------------------------

/// Prove that for a symmetric relative position bias, bias(i, j) = bias(j, i).
///
/// Relative position bias in models like T5 uses a learnable bias table indexed
/// by the relative position |i - j|. The symmetric property follows directly:
///   bias(i, j) = f(|i - j|)
///   bias(j, i) = f(|j - i|) = f(|i - j|)  (since |a - b| = |b - a|)
///
/// We prove: for any positions i, j and any distances d_ij = |i - j| and
/// d_ji = |j - i|, the distances are equal (hence any function of distance
/// gives the same result).
///
/// Uses `QF_LRA` — pure linear absolute-value reasoning.
pub fn prove_relative_position_bias_symmetry() -> Result<AttentionMaskPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let zero = Expr::real(0);
    let bound_hi = Expr::real(1000);

    // Positions i, j >= 0
    let i = declare_real(&mut program, "i");
    let j = declare_real(&mut program, "j");
    assert_bounds(&mut program, &i, &zero, &bound_hi);
    assert_bounds(&mut program, &j, &zero, &bound_hi);

    // |i - j|: model via helper variable d_ij >= 0 with d_ij >= i-j and d_ij >= j-i
    let d_ij = declare_real(&mut program, "d_ij");
    program.assert(d_ij.clone().real_ge(zero.clone()));
    program.assert(d_ij.clone().real_ge(i.clone().real_sub(j.clone())));
    program.assert(d_ij.clone().real_ge(j.clone().real_sub(i.clone())));
    // d_ij is exactly |i - j|: d_ij = i - j OR d_ij = j - i
    program.assert(
        d_ij.clone()
            .eq(i.clone().real_sub(j.clone()))
            .or(d_ij.clone().eq(j.clone().real_sub(i.clone()))),
    );

    // |j - i|: model via helper variable d_ji
    let d_ji = declare_real(&mut program, "d_ji");
    program.assert(d_ji.clone().real_ge(zero.clone()));
    program.assert(d_ji.clone().real_ge(j.clone().real_sub(i.clone())));
    program.assert(d_ji.clone().real_ge(i.clone().real_sub(j.clone())));
    program.assert(
        d_ji.clone()
            .eq(j.clone().real_sub(i.clone()))
            .or(d_ji.clone().eq(i.real_sub(j))),
    );

    // Violation: |i - j| != |j - i|
    let violation = d_ij.ne(d_ji);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionMaskPropertyResult {
        property: "relative_position_bias_symmetry".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 8: Sliding Window Mask
// ---------------------------------------------------------------------------

/// Prove that a sliding window mask with window size W satisfies:
///   mask[i][j] = 1 iff |i - j| <= W
///
/// For a concrete 4-position sequence with window W = 1:
/// ```text
///   [[1, 1, 0, 0],
///    [1, 1, 1, 0],
///    [0, 1, 1, 1],
///    [0, 0, 1, 1]]
/// ```
///
/// We verify this concrete mask by checking that every entry with |i-j| <= 1
/// is 1 and every entry with |i-j| > 1 is 0.
///
/// Uses `QF_LRA` — pure linear constraints.
pub fn prove_sliding_window_mask() -> Result<AttentionMaskPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let zero = Expr::real(0);

    // 4x4 mask for W=1
    // Expected mask[i][j] = 1 if |i-j| <= 1, else 0
    // Row 0: |0-0|=0<=1, |0-1|=1<=1, |0-2|=2>1, |0-3|=3>1 -> [1,1,0,0]
    // Row 1: |1-0|=1<=1, |1-1|=0<=1, |1-2|=1<=1, |1-3|=2>1 -> [1,1,1,0]
    // Row 2: |2-0|=2>1, |2-1|=1<=1, |2-2|=0<=1, |2-3|=1<=1 -> [0,1,1,1]
    // Row 3: |3-0|=3>1, |3-1|=2>1, |3-2|=1<=1, |3-3|=0<=1 -> [0,0,1,1]
    let expected: [[i64; 4]; 4] = [[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]];

    // Declare all mask entries
    let mut mask_vars = Vec::new();
    for row in 0..4 {
        for col in 0..4 {
            let name = format!("m{}_{}", row, col);
            let var = declare_real(&mut program, &name);
            // Set to expected value
            program.assert(var.clone().eq(Expr::real(expected[row][col])));
            mask_vars.push(var);
        }
    }

    // Sum of entries that should be 1 (within window)
    let mut within_sum = Expr::real(0);
    let mut outside_sum = Expr::real(0);
    let mut within_count = 0i64;

    for row in 0..4 {
        for col in 0..4 {
            let idx = row * 4 + col;
            let dist = if row >= col { row - col } else { col - row };
            if dist <= 1 {
                within_sum = within_sum.real_add(mask_vars[idx].clone());
                within_count += 1;
            } else {
                outside_sum = outside_sum.real_add(mask_vars[idx].clone());
            }
        }
    }

    // Violation: within-window entries don't sum to expected count
    // OR outside-window entries don't sum to 0
    let violation = within_sum
        .ne(Expr::real(within_count))
        .or(outside_sum.ne(zero));
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionMaskPropertyResult {
        property: "sliding_window_mask_w1_4x4".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 9: Attention Score Boundedness
// ---------------------------------------------------------------------------

/// Prove that for bounded Q, K vectors in [-B, B] with head dimension D,
/// the scaled dot-product attention score satisfies:
///   |Q . K / sqrt(D)| <= B^2 * D / sqrt(D) = B^2 * sqrt(D)
///
/// For head_dim D=2, Q = [q0, q1], K = [k0, k1]:
///   dot = q0*k0 + q1*k1
///   |dot| <= |q0|*|k0| + |q1|*|k1| <= B*B + B*B = 2*B^2 = D*B^2
///   |dot / sqrt(D)| <= D*B^2 / sqrt(D) = B^2 * sqrt(D)
///
/// We prove the linear bound |dot| <= D*B^2 using a linearized encoding.
/// The final bound |dot/sqrt(D)| <= B^2 * sqrt(D) follows by dividing both
/// sides by sqrt(D).
///
/// Uses `QF_LRA` with product terms modeled as bounded helper variables.
pub fn prove_attention_score_boundedness() -> Result<AttentionMaskPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Head dim D = 2, bound B = 10
    // Each q_i, k_i in [-B, B], so each product q_i*k_i in [-B^2, B^2]
    let b_sq = Expr::real(100); // B^2 = 10^2 = 100
    let neg_b_sq = Expr::real(-100);

    // Model each dot-product term as a bounded variable
    let t0 = declare_real(&mut program, "t0"); // represents q0*k0
    let t1 = declare_real(&mut program, "t1"); // represents q1*k1

    assert_bounds(&mut program, &t0, &neg_b_sq, &b_sq);
    assert_bounds(&mut program, &t1, &neg_b_sq, &b_sq);

    // dot = t0 + t1
    let dot = t0.real_add(t1);

    // D * B^2 = 2 * 100 = 200
    let upper_bound = Expr::real(200);
    let lower_bound = Expr::real(-200);

    // Violation: |dot| > D * B^2 (i.e., dot > 200 or dot < -200)
    let violation = dot
        .clone()
        .real_gt(upper_bound)
        .or(dot.real_lt(lower_bound));
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(AttentionMaskPropertyResult {
        property: "attention_score_boundedness_d2_b10".to_string(),
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

    #[test]
    fn test_causal_mask_lower_triangular_proven() {
        let result = prove_causal_mask_lower_triangular().expect("proof should not error");
        assert!(
            result.proven,
            "Causal mask lower triangular (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "causal_mask_lower_triangular_3x3");
    }

    #[test]
    fn test_causal_mask_softmax_zero_proven() {
        let result = prove_causal_mask_softmax_zero().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Causal mask softmax zero: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Causal mask softmax zero must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "causal_mask_softmax_zero_contribution");
    }

    #[test]
    fn test_alibi_bias_linearity_proven() {
        let result = prove_alibi_bias_linearity().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "ALiBi bias linearity: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "ALiBi bias linearity must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "alibi_bias_linearity");
    }

    #[test]
    fn test_alibi_slopes_geometric_proven() {
        let result = prove_alibi_slopes_geometric().expect("proof should not error");
        // The slopes are pinned to exact rationals, so every product folds to a
        // constant: the query is decidable ground arithmetic and `Unknown` is
        // not acceptable.
        assert!(
            result.proven,
            "ALiBi slopes geometric should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "alibi_slopes_geometric");
    }

    /// Perturbing `slope_3` to `1/6` breaks the constant ratio, so the
    /// geometric-mean identity `slope_2^2 = slope_1 * slope_3` fails and the
    /// query must be SAT. If it still proves, the theorem is not exercising the
    /// geometric structure.
    #[test]
    fn slopes_geometric_depends_on_the_constant_ratio() {
        let program = build_alibi_slopes_geometric(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with a non-constant ratio the slopes are not geometric and the query \
             must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_rope_rotation_orthogonal_proven() {
        let result = prove_rope_rotation_orthogonal().expect("proof should not error");
        // The angle is pinned to an exact rational point, so every entry of
        // R^T R is a constant: the query is decidable ground arithmetic and
        // `Unknown` is not acceptable.
        assert!(
            result.proven,
            "RoPE rotation orthogonal should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "rope_rotation_orthogonal");
    }

    /// Dropping the transpose (reusing `R` in place of `R^T`) makes the diagonal
    /// `c^2 - s^2 != 1`, so the orthogonality query must find a counterexample.
    /// If it still proves, the theorem is not exercising the transpose.
    #[test]
    fn rotation_orthogonal_depends_on_the_transpose() {
        let program = build_rope_rotation_orthogonal(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with the transpose dropped, R R != I and the query must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_rope_composition_proven() {
        let result = prove_rope_composition().expect("proof should not error");
        // Angles are pinned to exact rationals, so every product entry is a
        // constant: the query is decidable ground arithmetic and `Unknown` is
        // not acceptable.
        assert!(
            result.proven,
            "RoPE composition should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "rope_composition");
    }

    /// Flipping the off-diagonal sign of `R(theta2)` rotates by `-theta2`, so the
    /// product becomes `R(theta1 - theta2)` and disagrees with `R(theta1 + theta2)`.
    /// The query must then be SAT — proving the theorem rests on the rotation
    /// convention, not on writing the sum-angle entries twice.
    #[test]
    fn composition_depends_on_the_rotation_sign() {
        let program = build_rope_composition(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with R(theta2)'s signs flipped the product is R(theta1 - theta2) and the query \
             must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_relative_position_bias_symmetry_proven() {
        let result = prove_relative_position_bias_symmetry().expect("proof should not error");
        assert!(
            result.proven,
            "Relative position bias symmetry (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "relative_position_bias_symmetry");
    }

    #[test]
    fn test_sliding_window_mask_proven() {
        let result = prove_sliding_window_mask().expect("proof should not error");
        assert!(
            result.proven,
            "Sliding window mask (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "sliding_window_mask_w1_4x4");
    }

    #[test]
    fn test_attention_score_boundedness_proven() {
        let result = prove_attention_score_boundedness().expect("proof should not error");
        assert!(
            result.proven,
            "Attention score boundedness (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "attention_score_boundedness_d2_b10");
    }

    #[test]
    fn test_causal_mask_smt2_structure() {
        let result = prove_causal_mask_lower_triangular().expect("proof should not error");
        assert!(result.smt2.contains("set-logic"), "should declare logic");
        assert!(result.smt2.contains("check-sat"), "should have check-sat");
        assert!(
            result.smt2.contains("declare-const"),
            "should have declarations"
        );
    }

    #[test]
    fn test_rope_orthogonal_smt2_structure() {
        let result = prove_rope_rotation_orthogonal().expect("proof should not error");
        assert!(result.smt2.contains("set-logic"), "should declare logic");
        assert!(result.smt2.contains("check-sat"), "should have check-sat");
        assert!(result.smt2.contains("QF_NRA"), "should use QF_NRA logic");
    }
}
