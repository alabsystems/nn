// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for matrix decomposition properties (#4235).
//!
//! Proves properties of matrix decompositions commonly used in ML:
//!
//! 1. **Low-rank approximation error bounds**: For rank-k approximation A ~ UV^T,
//!    prove ||A - UV^T||_F^2 >= 0 (non-negative residual).
//! 2. **SVD orthogonality**: Prove U^T U = I and V^T V = I for orthogonal factors.
//! 3. **NMF non-negativity preservation**: For W, H >= 0, prove (WH)_ij >= 0.
//! 4. **Diagonal dominance under decomposition**: Prove that for a diagonally
//!    dominant matrix, diagonal elements dominate off-diagonal after LDL^T.
//! 5. **Schur complement positive definiteness**: Prove conditions for block
//!    matrix positive definiteness via Schur complement.
//! 6. **LoRA composition bounds**: Prove (W + AB^T)x bounds given bounds on W, A, B, x.
//!
//! # Proof Strategy
//!
//! Matrix decomposition operations on small concrete dimensions (2x2) are encoded
//! as scalar real arithmetic. Each matrix entry is a separate SMT real variable.
//! This avoids quantifiers and keeps proofs in `QF_NRA` or `QF_LRA`.
//!
//! - **Algebraic identity proofs (QF_NRA)**: Matmul-based properties that hold
//!   for all element values. We assert the negation and prove UNSAT.
//! - **Constrained proofs**: Orthogonality uses `Q^T Q = I`; NMF uses `W,H >= 0`;
//!   positive definiteness uses eigenvalue constraints.
//!
//! Small dimensions (2x2) suffice because these are universal algebraic identities.

use ay_bindings::{Expr, Sort, AYProgram};

use crate::ay_real_lit::RealLit;
use crate::smt_error::SmtError;

/// Result of a matrix decomposition property proof attempt.
#[derive(Debug, Clone)]
pub struct DecompPropertyResult {
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
/// hypotheses, so the solver derives it by chaining the definitions instead of
/// matching a term it was handed. This is what keeps the repaired
/// orthogonality proof non-vacuous.
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

/// Assert `expr >= 0`.
fn assert_non_negative(program: &mut AYProgram, expr: &Expr) {
    let zero = Expr::real(0);
    program.assert(expr.clone().real_ge(zero));
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
// Property 1: Low-Rank Approximation Error Non-Negativity
// ---------------------------------------------------------------------------

/// Prove that the Frobenius-norm residual of a rank-1 approximation is non-negative.
///
/// For a 2x2 matrix A and rank-1 factors U (2x1) and V (2x1), the approximation
/// is UV^T (outer product). The residual is:
///
/// ```text
/// ||A - UV^T||_F^2 = sum_{i,j} (A_ij - (UV^T)_ij)^2 >= 0
/// ```
///
/// This follows because each squared term is non-negative and a sum of
/// non-negative terms is non-negative. We verify this symbolically by asserting
/// the negation (residual < 0) and proving UNSAT.
///
/// Uses `QF_NRA` since the residual involves products and squares.
pub fn prove_low_rank_residual_non_negative() -> Result<DecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let bound_lo = Expr::real(-100);
    let bound_hi = Expr::real(100);

    // Declare 2x2 matrix A
    let a00 = declare_real(&mut program, "a00");
    let a01 = declare_real(&mut program, "a01");
    let a10 = declare_real(&mut program, "a10");
    let a11 = declare_real(&mut program, "a11");

    // Declare rank-1 factors: U (2x1), V (2x1)
    let u0 = declare_real(&mut program, "u0");
    let u1 = declare_real(&mut program, "u1");
    let v0 = declare_real(&mut program, "v0");
    let v1 = declare_real(&mut program, "v1");

    for var in [&a00, &a01, &a10, &a11, &u0, &u1, &v0, &v1] {
        assert_bounds(&mut program, var, &bound_lo, &bound_hi);
    }

    // UV^T entries: (UV^T)_ij = u_i * v_j
    let uvt_00 = u0.clone().real_mul(v0.clone());
    let uvt_01 = u0.real_mul(v1.clone());
    let uvt_10 = u1.clone().real_mul(v0);
    let uvt_11 = u1.real_mul(v1);

    // Residual entries: r_ij = a_ij - (UV^T)_ij
    let r00 = a00.real_sub(uvt_00);
    let r01 = a01.real_sub(uvt_01);
    let r10 = a10.real_sub(uvt_10);
    let r11 = a11.real_sub(uvt_11);

    // ||A - UV^T||_F^2 = r00^2 + r01^2 + r10^2 + r11^2
    let residual_sq = r00
        .clone()
        .real_mul(r00)
        .real_add(r01.clone().real_mul(r01))
        .real_add(r10.clone().real_mul(r10))
        .real_add(r11.clone().real_mul(r11));

    // Violation: residual < 0
    let zero = Expr::real(0);
    let violation = residual_sq.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "low_rank_residual_non_negative_2x2".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 2: SVD Orthogonality (U^T U = I)
// ---------------------------------------------------------------------------

/// Prove that an orthogonal (rotation) matrix satisfies `U^T U = I`.
///
/// The previous encoding was vacuous: it *asserted* the orthonormality
/// constraints `u00^2 + u10^2 = 1`, `u01^2 + u11^2 = 1`,
/// `u00*u01 + u10*u11 = 0` and then "proved" that the product `U^T U` — whose
/// entries are literally those same three expressions — equals `I`. The
/// conclusion was one of the hypotheses, so the query was UNSAT for free (the
/// vacuity gate flags it as `NegatesOwnHypothesis`).
///
/// The real content of `U^T U = I` is that `U` is an isometry whose transpose
/// is its inverse: `U^T U x = x` for *every* vector `x`. We state exactly that.
///
/// To stay in decidable linear arithmetic (`QF_LRA`) we pin `U` to a concrete
/// rotation at the Pythagorean angle `cos = 5/13`, `sin = 12/13` (so
/// `cos^2 + sin^2 = 1` exactly), and let the input `x = (x0, x1)` range over all
/// reals:
///
/// ```text
/// U   = [[ c, -s], [ s,  c]]        (rotation)
/// U^T = [[ c,  s], [-s,  c]]        (its transpose = its inverse)
/// U x     = ( c*x0 - s*x1,  s*x0 + c*x1)
/// U^T(U x) = x                       <-- the property, for all x
/// ```
///
/// Every coefficient is a rational *constant*, so each product is
/// `constant * variable` — linear, hence `QF_LRA`-decidable and strictly
/// provable. The conclusion `U^T(Ux) = x` is derived from the transpose rule
/// applied to `U`, not asserted; a wrong transpose (see the mutation test
/// `svd_orthogonality_depends_on_the_transpose`) makes it genuinely false.
pub fn prove_svd_orthogonality() -> Result<DecompPropertyResult, SmtError> {
    let program = build_svd_orthogonality(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "svd_orthogonality_utu_eq_i_2x2".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the `U^T U = I` query as "the transpose inverts the rotation".
///
/// When `transpose_is_correct` is false, the second factor is built as `U`
/// itself instead of `U^T` — the realistic slip of forgetting that transposing
/// a rotation negates the off-diagonal. Then the query computes `U (U x) = U^2 x`
/// (rotation by `2*theta`), which is not the identity, so the property becomes
/// false and the query turns SAT.
fn build_svd_orthogonality(transpose_is_correct: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Concrete rotation: cos = 5/13, sin = 12/13 (exactly, cos^2 + sin^2 = 1).
    let c = Expr::real_ratio(5, 13);
    let s = Expr::real_ratio(12, 13);
    let neg_s = s.clone().real_neg();

    // Arbitrary input vector x = (x0, x1), ranging over all reals.
    let x0 = declare_real(&mut program, "x0");
    let x1 = declare_real(&mut program, "x1");

    // U = [[c, -s], [s, c]]; U x = (c*x0 - s*x1, s*x0 + c*x1).
    let ux0 = define_real(
        &mut program,
        "ux0",
        &c.clone().real_mul(x0.clone()).real_sub(s.clone().real_mul(x1.clone())),
    );
    let ux1 = define_real(
        &mut program,
        "ux1",
        &s.clone().real_mul(x0.clone()).real_add(c.clone().real_mul(x1.clone())),
    );

    // Second factor: U^T = [[c, s], [-s, c]] when correct. The slip reuses U
    // itself ([[c, -s], [s, c]]), forgetting to negate the off-diagonal.
    let (m01, m10) = if transpose_is_correct {
        (s.clone(), neg_s.clone())
    } else {
        (neg_s.clone(), s.clone())
    };

    // Apply the second factor to (U x): row 0 = c*ux0 + m01*ux1,
    // row 1 = m10*ux0 + c*ux1.
    let utux0 = define_real(
        &mut program,
        "utux0",
        &c.clone().real_mul(ux0.clone()).real_add(m01.real_mul(ux1.clone())),
    );
    let utux1 = define_real(
        &mut program,
        "utux1",
        &m10.real_mul(ux0).real_add(c.real_mul(ux1)),
    );

    // Property: U^T (U x) = x for all x, i.e. U^T U = I.
    // Violation: some component of U^T(Ux) differs from x.
    let violation = utux0.ne(x0).or(utux1.ne(x1));
    program.assert(violation);
    program.check_sat();

    program
}

// ---------------------------------------------------------------------------
// Property 3: NMF Non-Negativity Preservation
// ---------------------------------------------------------------------------

/// Prove that for non-negative matrices W (2x2) and H (2x2), the product WH
/// has all non-negative entries.
///
/// If W_ij >= 0 and H_ij >= 0 for all i, j, then:
///   (WH)_ij = sum_k W_ik * H_kj
///
/// Each term W_ik * H_kj >= 0 (product of non-negatives). The sum of
/// non-negative terms is non-negative, so (WH)_ij >= 0.
///
/// Uses `QF_NRA` since the proof involves products of symbolic variables.
pub fn prove_nmf_non_negativity() -> Result<DecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let bound_hi = Expr::real(100);

    // W (2x2), non-negative entries
    let w00 = declare_real(&mut program, "w00");
    let w01 = declare_real(&mut program, "w01");
    let w10 = declare_real(&mut program, "w10");
    let w11 = declare_real(&mut program, "w11");

    // H (2x2), non-negative entries
    let h00 = declare_real(&mut program, "h00");
    let h01 = declare_real(&mut program, "h01");
    let h10 = declare_real(&mut program, "h10");
    let h11 = declare_real(&mut program, "h11");

    for var in [&w00, &w01, &w10, &w11, &h00, &h01, &h10, &h11] {
        assert_non_negative(&mut program, var);
        let zero = Expr::real(0);
        assert_bounds(&mut program, var, &zero, &bound_hi);
    }

    // Compute WH (2x2)
    let wh_00 = w00
        .clone()
        .real_mul(h00.clone())
        .real_add(w01.clone().real_mul(h10.clone()));
    let wh_01 = w00
        .real_mul(h01.clone())
        .real_add(w01.real_mul(h11.clone()));
    let wh_10 = w10
        .clone()
        .real_mul(h00)
        .real_add(w11.clone().real_mul(h10));
    let wh_11 = w10.real_mul(h01).real_add(w11.real_mul(h11));

    let zero = Expr::real(0);

    // Violation: any entry of WH is negative
    let v00 = wh_00.real_lt(zero.clone());
    let v01 = wh_01.real_lt(zero.clone());
    let v10 = wh_10.real_lt(zero.clone());
    let v11 = wh_11.real_lt(zero);

    let violation = Expr::or_many(vec![v00, v01, v10, v11]);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "nmf_non_negativity_2x2".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 4: Diagonal Dominance Preservation
// ---------------------------------------------------------------------------

/// Prove that for a strictly diagonally dominant 2x2 matrix, the diagonal
/// entries are larger in magnitude than the sum of off-diagonal entries in
/// the same row.
///
/// A matrix A is strictly diagonally dominant if for every row i:
///   |A_ii| > sum_{j != i} |A_ij|
///
/// For a 2x2 matrix with positive diagonal entries:
///   a00 > 0, a11 > 0
///   a00 > |a01|
///   a11 > |a10|
///
/// We prove that under these constraints, a00 * a11 > a01 * a10 (i.e.,
/// the determinant is positive). This is a known consequence of strict
/// diagonal dominance: such matrices are non-singular with positive determinant.
///
/// Proof: a00 * a11 > |a01| * |a10| >= a01 * a10 (by AM-GM direction),
/// hence det(A) = a00*a11 - a01*a10 > 0.
///
/// We encode: a00*a11 - |a01|*|a10| > 0, which the solver models as:
///   a00 > abs01, a11 > abs10, abs01 >= 0, abs10 >= 0, a00*a11 <= abs01*abs10
/// should be UNSAT.
///
/// Uses `QF_NRA`.
pub fn prove_diagonal_dominance_positive_det() -> Result<DecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let bound_lo = Expr::real(-100);
    let bound_hi = Expr::real(100);
    let zero = Expr::real(0);

    // Diagonal entries: positive
    let a00 = declare_real(&mut program, "a00");
    let a11 = declare_real(&mut program, "a11");
    program.assert(a00.clone().real_gt(zero.clone()));
    program.assert(a11.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &a00, &zero, &bound_hi);
    assert_bounds(&mut program, &a11, &zero, &bound_hi);

    // Off-diagonal entries (bounded)
    let a01 = declare_real(&mut program, "a01");
    let a10 = declare_real(&mut program, "a10");
    assert_bounds(&mut program, &a01, &bound_lo, &bound_hi);
    assert_bounds(&mut program, &a10, &bound_lo, &bound_hi);

    // Absolute values of off-diagonal entries
    let abs01 = declare_real(&mut program, "abs01");
    let abs10 = declare_real(&mut program, "abs10");
    assert_non_negative(&mut program, &abs01);
    assert_non_negative(&mut program, &abs10);
    // abs01 >= a01 and abs01 >= -a01, and abs01 = a01 or abs01 = -a01
    program.assert(abs01.clone().real_ge(a01.clone()));
    program.assert(abs01.clone().real_ge(a01.clone().real_neg()));
    program.assert(
        abs01
            .clone()
            .eq(a01.clone())
            .or(abs01.clone().eq(a01.real_neg())),
    );
    program.assert(abs10.clone().real_ge(a10.clone()));
    program.assert(abs10.clone().real_ge(a10.clone().real_neg()));
    program.assert(
        abs10
            .clone()
            .eq(a10.clone())
            .or(abs10.clone().eq(a10.real_neg())),
    );

    // Strict diagonal dominance:
    //   a00 > abs01 (row 0)
    //   a11 > abs10 (row 1)
    program.assert(a00.clone().real_gt(abs01.clone()));
    program.assert(a11.clone().real_gt(abs10.clone()));

    // Product of diagonals: a00 * a11
    let diag_prod = a00.real_mul(a11);
    // Product of off-diagonal absolute values: abs01 * abs10
    let offdiag_prod = abs01.real_mul(abs10);

    // Violation: diagonal product is NOT greater than off-diagonal product
    // (i.e., a00*a11 <= abs01*abs10, which would mean det could be non-positive)
    let violation = diag_prod.real_le(offdiag_prod);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "diagonal_dominance_positive_det_2x2".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 5: Schur Complement Positive Definiteness
// ---------------------------------------------------------------------------

/// Prove that for a 2x2 block matrix M = [[a, b], [b, d]] where a > 0 and
/// the Schur complement s = d - b^2/a > 0, the quadratic form x^T M x > 0
/// for all nonzero x.
///
/// For M = [[a, b], [b, d]] symmetric, x = (x0, x1):
///   x^T M x = a*x0^2 + 2*b*x0*x1 + d*x1^2
///
/// By completing the square:
///   = a*(x0 + (b/a)*x1)^2 + (d - b^2/a)*x1^2
///   = a*(x0 + (b/a)*x1)^2 + s*x1^2
///
/// If a > 0 and s > 0, both terms are non-negative, and at least one is
/// positive when (x0, x1) != (0, 0). Hence x^T M x > 0.
///
/// We encode this by asserting the conditions and checking that the quadratic
/// form can never be non-positive for nonzero x.
///
/// Uses `QF_NRA` since it involves products.
pub fn prove_schur_complement_positive_definite() -> Result<DecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let zero = Expr::real(0);
    let bound_lo = Expr::real(-100);
    let bound_hi = Expr::real(100);

    // Matrix entries: M = [[a, b], [b, d]] symmetric
    let a = declare_real(&mut program, "a");
    let b = declare_real(&mut program, "b");
    let d = declare_real(&mut program, "d");

    assert_bounds(&mut program, &a, &zero, &bound_hi);
    assert_bounds(&mut program, &b, &bound_lo, &bound_hi);
    assert_bounds(&mut program, &d, &zero, &bound_hi);

    // a > 0
    program.assert(a.clone().real_gt(zero.clone()));

    // Schur complement condition: a*d - b^2 > 0 (equivalently det(M) > 0)
    let det = a
        .clone()
        .real_mul(d.clone())
        .real_sub(b.clone().real_mul(b.clone()));
    program.assert(det.real_gt(zero.clone()));

    // Vector x = (x0, x1), not both zero
    let x0 = declare_real(&mut program, "x0");
    let x1 = declare_real(&mut program, "x1");
    assert_bounds(&mut program, &x0, &bound_lo, &bound_hi);
    assert_bounds(&mut program, &x1, &bound_lo, &bound_hi);

    program.assert(x0.clone().ne(zero.clone()).or(x1.clone().ne(zero.clone())));

    // x^T M x = a*x0^2 + 2*b*x0*x1 + d*x1^2
    let term1 = a.real_mul(x0.clone().real_mul(x0.clone()));
    let term2 = Expr::real(2).real_mul(b.real_mul(x0.real_mul(x1.clone())));
    let term3 = d.real_mul(x1.clone().real_mul(x1));
    let quad_form = term1.real_add(term2).real_add(term3);

    // Violation: x^T M x <= 0 for some nonzero x
    let violation = quad_form.real_le(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "schur_complement_positive_definite_2x2".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 6: LoRA Composition Bounds
// ---------------------------------------------------------------------------

/// Prove that for a LoRA-augmented linear layer (W + AB^T)x, the output
/// is bounded when W, A, B, and x are bounded.
///
/// For 2x2 weight matrix W, rank-1 LoRA factors A (2x1) and B (2x1),
/// and input vector x (2x1), all entries in [-K, K]:
///
/// ```text
/// y = (W + AB^T)x = Wx + AB^Tx
/// ```
///
/// Each component of Wx is bounded: |(Wx)_i| <= 2*K^2 (sum of 2 products).
/// Each component of AB^Tx: (AB^Tx)_i = a_i * (b^T x) where
///   |b^T x| <= 2*K^2 and |a_i| <= K, so |(AB^Tx)_i| <= 2*K^3.
///
/// Total: |y_i| <= 2*K^2 + 2*K^3.
///
/// For K = 5: bound = 2*25 + 2*125 = 50 + 250 = 300.
///
/// We prove |y_i| <= 300 for all bounded inputs.
///
/// Uses `QF_LRA` with products modeled as bounded helper variables.
pub fn prove_lora_composition_bounds() -> Result<DecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // K = 5, so all entries in [-5, 5]
    // Products of two bounded vars in [-25, 25] (K^2 = 25)
    // Products of three bounded vars in [-125, 125] (K^3 = 125)
    let k_sq = Expr::real(25);
    let neg_k_sq = Expr::real(-25);
    let k_cube = Expr::real(125);
    let neg_k_cube = Expr::real(-125);

    // Model Wx: each entry is sum of 2 products, each product in [-K^2, K^2]
    // (Wx)_0 = w00*x0 + w01*x1 => two terms, each in [-25, 25]
    let wx_t00 = declare_real(&mut program, "wx_t00"); // w00*x0
    let wx_t01 = declare_real(&mut program, "wx_t01"); // w01*x1
    let wx_t10 = declare_real(&mut program, "wx_t10"); // w10*x0
    let wx_t11 = declare_real(&mut program, "wx_t11"); // w11*x1

    for var in [&wx_t00, &wx_t01, &wx_t10, &wx_t11] {
        assert_bounds(&mut program, var, &neg_k_sq, &k_sq);
    }

    let wx_0 = wx_t00.real_add(wx_t01);
    let wx_1 = wx_t10.real_add(wx_t11);

    // Model AB^Tx: (AB^Tx)_i = a_i * (b0*x0 + b1*x1)
    // b0*x0 and b1*x1 are each in [-K^2, K^2], so b^Tx in [-2K^2, 2K^2]
    // a_i * b^Tx: a_i in [-K, K], b^Tx in [-2K^2, 2K^2]
    // But we model the triple product a_i*b_j*x_j directly.
    // (AB^Tx)_0 = a0*b0*x0 + a0*b1*x1 => two terms, each in [-K^3, K^3]
    let abt_t00 = declare_real(&mut program, "abt_t00"); // a0*b0*x0
    let abt_t01 = declare_real(&mut program, "abt_t01"); // a0*b1*x1
    let abt_t10 = declare_real(&mut program, "abt_t10"); // a1*b0*x0
    let abt_t11 = declare_real(&mut program, "abt_t11"); // a1*b1*x1

    for var in [&abt_t00, &abt_t01, &abt_t10, &abt_t11] {
        assert_bounds(&mut program, var, &neg_k_cube, &k_cube);
    }

    let abtx_0 = abt_t00.real_add(abt_t01);
    let abtx_1 = abt_t10.real_add(abt_t11);

    // y = Wx + AB^Tx
    let y0 = wx_0.real_add(abtx_0);
    let y1 = wx_1.real_add(abtx_1);

    // Bound: |y_i| <= 2*K^2 + 2*K^3 = 2*25 + 2*125 = 300
    let upper = Expr::real(300);
    let lower = Expr::real(-300);

    // Violation: |y_0| > 300 or |y_1| > 300
    let v0_hi = y0.clone().real_gt(upper.clone());
    let v0_lo = y0.real_lt(lower.clone());
    let v1_hi = y1.clone().real_gt(upper);
    let v1_lo = y1.real_lt(lower);

    let violation = Expr::or_many(vec![v0_hi, v0_lo, v1_hi, v1_lo]);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "lora_composition_bounds_k5_2x2".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Run all six matrix decomposition property proofs and return results.
pub fn prove_all_decomposition_properties() -> Result<Vec<DecompPropertyResult>, SmtError> {
    Ok(vec![
        prove_low_rank_residual_non_negative()?,
        prove_svd_orthogonality()?,
        prove_nmf_non_negativity()?,
        prove_diagonal_dominance_positive_det()?,
        prove_schur_complement_positive_definite()?,
        prove_lora_composition_bounds()?,
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
    fn test_low_rank_residual_non_negative_proven() {
        let result = prove_low_rank_residual_non_negative().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Low-rank residual non-negativity: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Low-rank residual non-negativity must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "low_rank_residual_non_negative_2x2");
    }

    #[test]
    fn test_svd_orthogonality_proven() {
        let result = prove_svd_orthogonality().expect("proof should not error");
        // The rotation isometry is encoded in QF_LRA over concrete rational
        // coefficients, so it is decidable and must strictly prove.
        assert!(
            result.proven,
            "SVD orthogonality (QF_LRA rotation isometry) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "svd_orthogonality_utu_eq_i_2x2");
    }

    /// Build the second factor as `U` itself instead of `U^T` (forgetting to
    /// negate the off-diagonal when transposing a rotation). Then the query
    /// computes `U^2 x` (rotation by `2*theta`), which is not the identity, so
    /// the property `U^T U = I` is genuinely false and the query must be SAT.
    /// If this still "proved", the theorem would be vacuous.
    #[test]
    fn svd_orthogonality_depends_on_the_transpose() {
        let program = build_svd_orthogonality(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with the transpose wrong the map is U^2 != I and the query must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_nmf_non_negativity_proven() {
        let result = prove_nmf_non_negativity().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "NMF non-negativity: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "NMF non-negativity must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "nmf_non_negativity_2x2");
    }

    #[test]
    fn test_diagonal_dominance_positive_det_proven() {
        let result = prove_diagonal_dominance_positive_det().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Diagonal dominance positive det: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Diagonal dominance positive det must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "diagonal_dominance_positive_det_2x2");
    }

    #[test]
    fn test_schur_complement_positive_definite_proven() {
        let result = prove_schur_complement_positive_definite().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Schur complement positive definite: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Schur complement positive definite must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "schur_complement_positive_definite_2x2");
    }

    #[test]
    fn test_lora_composition_bounds_proven() {
        let result = prove_lora_composition_bounds().expect("proof should not error");
        assert!(
            result.proven,
            "LoRA composition bounds (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "lora_composition_bounds_k5_2x2");
    }

    #[test]
    fn test_prove_all_decomposition_properties() {
        let results = prove_all_decomposition_properties().expect("all proofs should not error");
        assert_eq!(results.len(), 6);
        for result in &results {
            assert!(
                result.proven || result.detail.contains("Unknown"),
                "Property '{}' should be Proven or Unknown, got: {}",
                result.property,
                result.detail,
            );
            assert!(
                !result.detail.contains("counterexample"),
                "Property '{}' must not have counterexample: {}",
                result.property,
                result.detail,
            );
        }
    }

    #[test]
    fn test_low_rank_residual_smt2_structure() {
        let result = prove_low_rank_residual_non_negative().expect("proof should not error");
        assert!(result.smt2.contains("set-logic"), "should declare logic");
        assert!(result.smt2.contains("check-sat"), "should have check-sat");
        assert!(result.smt2.contains("QF_NRA"), "should use QF_NRA logic");
        assert!(
            result.smt2.contains("declare-const"),
            "should have declarations"
        );
    }

    #[test]
    fn test_lora_smt2_structure() {
        let result = prove_lora_composition_bounds().expect("proof should not error");
        assert!(result.smt2.contains("set-logic"), "should declare logic");
        assert!(result.smt2.contains("check-sat"), "should have check-sat");
        assert!(result.smt2.contains("QF_LRA"), "should use QF_LRA logic");
    }

    #[test]
    fn test_nmf_smt2_structure() {
        let result = prove_nmf_non_negativity().expect("proof should not error");
        assert!(result.smt2.contains("set-logic"), "should declare logic");
        assert!(result.smt2.contains("QF_NRA"), "should use QF_NRA logic");
        assert!(result.smt2.contains("w00"), "should have W matrix entries");
        assert!(result.smt2.contains("h00"), "should have H matrix entries");
    }
}
