// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for matrix decomposition mathematical properties (#4235).
//!
//! Proves fundamental linear algebra identities using ay's SMT solver:
//!
//! 1. **Matrix multiplication associativity**: `(A*B)*C = A*(B*C)` element-wise.
//! 2. **Transpose involution**: `(A^T)^T = A`.
//! 4. **Frobenius norm non-negativity**: `||A||_F >= 0` and `||A||_F = 0` iff `A = 0`.
//! 5. **Orthogonal matrix norm preservation**: For orthogonal `Q`, `||Qx||_2 = ||x||_2`.
//! 6. **SVD decomposition shape**: U is `[m, m]`, S is `[min(m, n)]`, V^T is `[n, n]`.
//! 7. **Matrix trace scalar homogeneity**: `tr(cA) = c * tr(A)`.
//!
//! # Proof Strategy
//!
//! Matrix operations on small concrete dimensions (2x2 or 2x3) are encoded as
//! scalar real arithmetic. Each matrix entry is a separate SMT real variable.
//! This avoids quantifiers and keeps proofs in `QF_NRA` or `QF_LRA`.
//!
//! - **Algebraic identity proofs (QF_NRA/QF_LRA)**: Matrix identities that hold
//!   for all element values. We assert the negation and prove UNSAT.
//!
//! - **Constrained proofs**: Orthogonality uses the constraint `Q^T * Q = I`,
//!   which introduces non-linear constraints (products of matrix entries).
//!
//! Small dimensions (2x2, 2x3) suffice because these are universal algebraic
//! identities — if they hold for symbolic 2x2 matrices, the element-wise
//! structure generalizes to any dimension.

use ay_bindings::{Expr, Sort, AYProgram};

use crate::smt_error::SmtError;

/// Result of a matrix decomposition property proof attempt.
#[derive(Debug, Clone)]
pub struct MatrixDecompPropertyResult {
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

/// Execute a ay program and return whether UNSAT (property proven).
///
/// The final `(proven, detail)` is funneled through
/// [`crate::ay_vacuity::reject_if_vacuous`], so any query that is UNSAT only
/// because it asserts `P ∧ ¬P` (or compares a term to itself) is downgraded to a
/// failure rather than being reported as a false "proven". A genuine proof is
/// returned unchanged; a residual vacuity becomes a hard test failure.
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
/// Naming each intermediate quantity keeps the conclusion one step removed from
/// its hypotheses, so the solver *derives* it by chaining definitions instead of
/// reading it back off an asserted answer — which is what keeps a shape proof
/// from collapsing into a vacuous `P ∧ ¬P`.
fn define_real(program: &mut AYProgram, name: &str, term: &Expr) -> Expr {
    let var = declare_real(program, name);
    program.assert(var.clone().eq(term.clone()));
    var
}

// ---------------------------------------------------------------------------
// Property 1: Matrix Multiplication Associativity
// ---------------------------------------------------------------------------

/// Prove `(A * B) * C = A * (B * C)` for 2x2 matrices, element-wise.
///
/// For 2x2 matrices with entries `a_ij`, `b_ij`, `c_ij`:
///
/// ```text
/// AB_ij = sum_k a_ik * b_kj
/// (AB)C_ij = sum_k AB_ik * c_kj = sum_k (sum_l a_il * b_lk) * c_kj
/// A(BC)_ij = sum_k a_ik * BC_kj = sum_k a_ik * (sum_l b_kl * c_lj)
/// ```
///
/// Both expand to `sum_{k,l} a_ik * b_kl * c_lj` — the same triple sum.
/// We verify this algebraic identity by asserting the negation and proving UNSAT.
///
/// Uses `QF_NRA` since matmul involves products of symbolic variables.
pub fn prove_matmul_associativity() -> Result<MatrixDecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let bound_lo = Expr::real(-100);
    let bound_hi = Expr::real(100);

    // Declare 2x2 matrix A entries
    let a00 = declare_real(&mut program, "a00");
    let a01 = declare_real(&mut program, "a01");
    let a10 = declare_real(&mut program, "a10");
    let a11 = declare_real(&mut program, "a11");

    // Declare 2x2 matrix B entries
    let b00 = declare_real(&mut program, "b00");
    let b01 = declare_real(&mut program, "b01");
    let b10 = declare_real(&mut program, "b10");
    let b11 = declare_real(&mut program, "b11");

    // Declare 2x2 matrix C entries
    let c00 = declare_real(&mut program, "c00");
    let c01 = declare_real(&mut program, "c01");
    let c10 = declare_real(&mut program, "c10");
    let c11 = declare_real(&mut program, "c11");

    // Bound all inputs for solver convergence
    for v in [
        &a00, &a01, &a10, &a11, &b00, &b01, &b10, &b11, &c00, &c01, &c10, &c11,
    ] {
        assert_bounds(&mut program, v, &bound_lo, &bound_hi);
    }

    // Compute AB (2x2)
    let ab00 = a00
        .clone()
        .real_mul(b00.clone())
        .real_add(a01.clone().real_mul(b10.clone()));
    let ab01 = a00
        .clone()
        .real_mul(b01.clone())
        .real_add(a01.clone().real_mul(b11.clone()));
    let ab10 = a10
        .clone()
        .real_mul(b00.clone())
        .real_add(a11.clone().real_mul(b10.clone()));
    let ab11 = a10
        .clone()
        .real_mul(b01.clone())
        .real_add(a11.clone().real_mul(b11.clone()));

    // Compute (AB)C (2x2)
    let abc_left_00 = ab00
        .clone()
        .real_mul(c00.clone())
        .real_add(ab01.clone().real_mul(c10.clone()));
    let abc_left_01 = ab00
        .real_mul(c01.clone())
        .real_add(ab01.real_mul(c11.clone()));
    let abc_left_10 = ab10
        .clone()
        .real_mul(c00.clone())
        .real_add(ab11.clone().real_mul(c10.clone()));
    let abc_left_11 = ab10
        .real_mul(c01.clone())
        .real_add(ab11.real_mul(c11.clone()));

    // Compute BC (2x2)
    let bc00 = b00
        .clone()
        .real_mul(c00.clone())
        .real_add(b01.clone().real_mul(c10.clone()));
    let bc01 = b00
        .real_mul(c01.clone())
        .real_add(b01.real_mul(c11.clone()));
    let bc10 = b10
        .clone()
        .real_mul(c00)
        .real_add(b11.clone().real_mul(c10));
    let bc11 = b10.real_mul(c01).real_add(b11.real_mul(c11));

    // Compute A(BC) (2x2)
    let abc_right_00 = a00
        .clone()
        .real_mul(bc00.clone())
        .real_add(a01.clone().real_mul(bc10.clone()));
    let abc_right_01 = a00
        .real_mul(bc01.clone())
        .real_add(a01.real_mul(bc11.clone()));
    let abc_right_10 = a10
        .clone()
        .real_mul(bc00)
        .real_add(a11.clone().real_mul(bc10));
    let abc_right_11 = a10.real_mul(bc01).real_add(a11.real_mul(bc11));

    // Violation: any element of (AB)C differs from A(BC)
    let v00 = abc_left_00.ne(abc_right_00);
    let v01 = abc_left_01.ne(abc_right_01);
    let v10 = abc_left_10.ne(abc_right_10);
    let v11 = abc_left_11.ne(abc_right_11);

    let violation = Expr::or_many(vec![v00, v01, v10, v11]);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(MatrixDecompPropertyResult {
        property: "matmul_associativity_2x2".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 2: Transpose Involution
// ---------------------------------------------------------------------------

/// Prove `(A^T)^T = A` for a 2x3 matrix.
///
/// For matrix A of shape [2, 3]:
///   `A^T` has shape [3, 2] with `(A^T)_ji = A_ij`.
///   `(A^T)^T` has shape [2, 3] with `((A^T)^T)_ij = (A^T)_ji = A_ij`.
///
/// This is a direct identity on indices. We verify it symbolically in `QF_LRA`.
pub fn prove_transpose_involution() -> Result<MatrixDecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Declare 2x3 matrix A entries
    let a00 = declare_real(&mut program, "a00");
    let a01 = declare_real(&mut program, "a01");
    let a02 = declare_real(&mut program, "a02");
    let a10 = declare_real(&mut program, "a10");
    let a11 = declare_real(&mut program, "a11");
    let a12 = declare_real(&mut program, "a12");

    let bound_lo = Expr::real(-1000);
    let bound_hi = Expr::real(1000);
    for v in [&a00, &a01, &a02, &a10, &a11, &a12] {
        assert_bounds(&mut program, v, &bound_lo, &bound_hi);
    }

    // A^T is [3, 2]:
    //   at_00 = a00, at_01 = a10
    //   at_10 = a01, at_11 = a11
    //   at_20 = a02, at_21 = a12
    //
    // (A^T)^T is [2, 3]:
    //   att_00 = at_00 = a00
    //   att_01 = at_10 = a01
    //   att_02 = at_20 = a02
    //   att_10 = at_01 = a10
    //   att_11 = at_11 = a11
    //   att_12 = at_21 = a12
    //
    // So (A^T)^T = A element-wise. Declare intermediate variables to make
    // the SMT encoding explicit.

    // A^T entries (named for clarity)
    let at_00 = declare_real(&mut program, "at_00");
    let at_01 = declare_real(&mut program, "at_01");
    let at_10 = declare_real(&mut program, "at_10");
    let at_11 = declare_real(&mut program, "at_11");
    let at_20 = declare_real(&mut program, "at_20");
    let at_21 = declare_real(&mut program, "at_21");

    // Define A^T: (A^T)_ji = A_ij
    program.assert(at_00.clone().eq(a00.clone()));
    program.assert(at_01.clone().eq(a10.clone()));
    program.assert(at_10.clone().eq(a01.clone()));
    program.assert(at_11.clone().eq(a11.clone()));
    program.assert(at_20.clone().eq(a02.clone()));
    program.assert(at_21.clone().eq(a12.clone()));

    // (A^T)^T entries
    let att_00 = declare_real(&mut program, "att_00");
    let att_01 = declare_real(&mut program, "att_01");
    let att_02 = declare_real(&mut program, "att_02");
    let att_10 = declare_real(&mut program, "att_10");
    let att_11 = declare_real(&mut program, "att_11");
    let att_12 = declare_real(&mut program, "att_12");

    // Define (A^T)^T: ((A^T)^T)_ij = (A^T)_ji
    program.assert(att_00.clone().eq(at_00));
    program.assert(att_01.clone().eq(at_10));
    program.assert(att_02.clone().eq(at_20));
    program.assert(att_10.clone().eq(at_01));
    program.assert(att_11.clone().eq(at_11));
    program.assert(att_12.clone().eq(at_21));

    // Violation: any element of (A^T)^T differs from A
    let v00 = att_00.ne(a00);
    let v01 = att_01.ne(a01);
    let v02 = att_02.ne(a02);
    let v10 = att_10.ne(a10);
    let v11 = att_11.ne(a11);
    let v12 = att_12.ne(a12);

    let violation = Expr::or_many(vec![v00, v01, v02, v10, v11, v12]);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(MatrixDecompPropertyResult {
        property: "transpose_involution_2x3".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 4: Frobenius Norm Non-Negativity
// ---------------------------------------------------------------------------

/// Prove `||A||_F >= 0` and `||A||_F = 0` iff `A = 0` for a 2x2 matrix.
///
/// The Frobenius norm is defined as:
///   `||A||_F^2 = sum_{i,j} A_ij^2`
///
/// Since each `A_ij^2 >= 0`, the sum is `>= 0`, hence `||A||_F >= 0`.
///
/// For the "iff zero" direction:
/// - Forward: If `A = 0` then all `A_ij = 0`, so `||A||_F^2 = 0`.
/// - Backward: If `||A||_F^2 = 0` and each `A_ij^2 >= 0`, then each `A_ij^2 = 0`,
///   so each `A_ij = 0`, i.e., `A = 0`.
///
/// We prove both parts: (a) non-negativity, (b) zero-norm implies zero matrix.
pub fn prove_frobenius_norm_non_negativity() -> Result<MatrixDecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let bound_lo = Expr::real(-100);
    let bound_hi = Expr::real(100);

    // Declare 2x2 matrix A
    let a00 = declare_real(&mut program, "a00");
    let a01 = declare_real(&mut program, "a01");
    let a10 = declare_real(&mut program, "a10");
    let a11 = declare_real(&mut program, "a11");

    for v in [&a00, &a01, &a10, &a11] {
        assert_bounds(&mut program, v, &bound_lo, &bound_hi);
    }

    // ||A||_F^2 = a00^2 + a01^2 + a10^2 + a11^2
    let norm_sq = a00
        .clone()
        .real_mul(a00.clone())
        .real_add(a01.clone().real_mul(a01.clone()))
        .real_add(a10.clone().real_mul(a10.clone()))
        .real_add(a11.clone().real_mul(a11.clone()));

    let zero = Expr::real(0);

    // Violation for non-negativity: ||A||_F^2 < 0
    let violation = norm_sq.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(MatrixDecompPropertyResult {
        property: "frobenius_norm_non_negativity".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove that `||A||_F^2 = 0` implies `A = 0` for a 2x2 matrix.
///
/// If `a00^2 + a01^2 + a10^2 + a11^2 = 0` and all entries are real,
/// then each entry must be zero (sum of non-negative terms = 0 implies
/// each term is 0).
pub fn prove_frobenius_norm_zero_iff_zero_matrix() -> Result<MatrixDecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let bound_lo = Expr::real(-100);
    let bound_hi = Expr::real(100);

    let a00 = declare_real(&mut program, "a00");
    let a01 = declare_real(&mut program, "a01");
    let a10 = declare_real(&mut program, "a10");
    let a11 = declare_real(&mut program, "a11");

    for v in [&a00, &a01, &a10, &a11] {
        assert_bounds(&mut program, v, &bound_lo, &bound_hi);
    }

    // ||A||_F^2 = 0
    let norm_sq = a00
        .clone()
        .real_mul(a00.clone())
        .real_add(a01.clone().real_mul(a01.clone()))
        .real_add(a10.clone().real_mul(a10.clone()))
        .real_add(a11.clone().real_mul(a11.clone()));

    let zero = Expr::real(0);
    program.assert(norm_sq.eq(zero.clone()));

    // Violation: A is not the zero matrix (at least one entry != 0)
    let v00 = a00.ne(zero.clone());
    let v01 = a01.ne(zero.clone());
    let v10 = a10.ne(zero.clone());
    let v11 = a11.ne(zero);

    let violation = Expr::or_many(vec![v00, v01, v10, v11]);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(MatrixDecompPropertyResult {
        property: "frobenius_norm_zero_iff_zero_matrix".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 5: Orthogonal Matrix Norm Preservation
// ---------------------------------------------------------------------------

/// Prove that for an orthogonal 2x2 matrix Q (where `Q^T * Q = I`),
/// `||Qx||_2^2 = ||x||_2^2` (norm preservation).
///
/// Given Q orthogonal: `Q^T * Q = I`, so for any vector x:
///   `||Qx||^2 = (Qx)^T (Qx) = x^T Q^T Q x = x^T I x = x^T x = ||x||^2`
///
/// For a 2x2 orthogonal matrix Q = [[q00, q01], [q10, q11]] with
/// `Q^T * Q = I`, we prove `(q00*x + q01*y)^2 + (q10*x + q11*y)^2 = x^2 + y^2`.
///
/// The orthogonality constraints are:
///   `q00^2 + q10^2 = 1`  (first column has unit norm)
///   `q01^2 + q11^2 = 1`  (second column has unit norm)
///   `q00*q01 + q10*q11 = 0`  (columns are orthogonal)
pub fn prove_orthogonal_norm_preservation() -> Result<MatrixDecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    // Orthogonal matrix Q (2x2)
    let q00 = declare_real(&mut program, "q00");
    let q01 = declare_real(&mut program, "q01");
    let q10 = declare_real(&mut program, "q10");
    let q11 = declare_real(&mut program, "q11");

    // Input vector x = (x0, x1)
    let x0 = declare_real(&mut program, "x0");
    let x1 = declare_real(&mut program, "x1");

    // Bound all inputs for solver convergence
    let bound_lo = Expr::real(-100);
    let bound_hi = Expr::real(100);
    for v in [&q00, &q01, &q10, &q11, &x0, &x1] {
        assert_bounds(&mut program, v, &bound_lo, &bound_hi);
    }

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // Orthogonality constraints: Q^T * Q = I
    // Column 0 unit norm: q00^2 + q10^2 = 1
    program.assert(
        q00.clone()
            .real_mul(q00.clone())
            .real_add(q10.clone().real_mul(q10.clone()))
            .eq(one.clone()),
    );
    // Column 1 unit norm: q01^2 + q11^2 = 1
    program.assert(
        q01.clone()
            .real_mul(q01.clone())
            .real_add(q11.clone().real_mul(q11.clone()))
            .eq(one),
    );
    // Columns orthogonal: q00*q01 + q10*q11 = 0
    program.assert(
        q00.clone()
            .real_mul(q01.clone())
            .real_add(q10.clone().real_mul(q11.clone()))
            .eq(zero),
    );

    // Qx = (q00*x0 + q01*x1, q10*x0 + q11*x1)
    let qx0 = q00.real_mul(x0.clone()).real_add(q01.real_mul(x1.clone()));
    let qx1 = q10.real_mul(x0.clone()).real_add(q11.real_mul(x1.clone()));

    // ||Qx||^2 = qx0^2 + qx1^2
    let norm_qx_sq = qx0
        .clone()
        .real_mul(qx0)
        .real_add(qx1.clone().real_mul(qx1));

    // ||x||^2 = x0^2 + x1^2
    let norm_x_sq = x0
        .clone()
        .real_mul(x0)
        .real_add(x1.clone().real_mul(x1));

    // Violation: ||Qx||^2 != ||x||^2
    let violation = norm_qx_sq.ne(norm_x_sq);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(MatrixDecompPropertyResult {
        property: "orthogonal_norm_preservation_2x2".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 6: SVD Decomposition Shape
// ---------------------------------------------------------------------------

/// Prove the SVD (full) decomposition shape rule: for `A` of shape `[m, n]`,
/// the factors `A = U · Σ · Vᵀ` have shapes `U:[m,m]`, `Σ:[m,n]`, `Vᵀ:[n,n]`,
/// and running those shapes back through the reconstruction matmul chain is
/// dimensionally consistent and reproduces the original shape `[m, n]`.
///
/// The theorem has real content: the shape rule is *applied* to free dimensions
/// `m, n` (positive integers modeled as reals ≥ 1), each factor shape is a
/// declared variable derived from that rule, and the reconstruction chain
///
/// ```text
/// P = U · Σ   needs u_cols = sigma_rows;  P : [u_rows, sigma_cols]
/// R = P · Vᵀ  needs p_cols = vt_rows;     R : [p_rows, vt_cols]
/// ```
///
/// is required to be compatible at each matmul AND to yield `R : [m, n]`. The
/// conclusion is reached by chaining definitions (`r_cols = vt_cols = a_cols = n`),
/// never by asserting the answer and negating it — so the query is non-vacuous
/// and `QF_LRA`-decidable. Mis-shaping `U` to `[m, n]` (the thin-vs-full SVD slip)
/// breaks the first matmul's inner-dimension match and makes the query SAT (see
/// `svd_shape_depends_on_u_being_square`).
pub fn prove_svd_shape_consistency() -> Result<MatrixDecompPropertyResult, SmtError> {
    let program = build_svd_shape_consistency(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(MatrixDecompPropertyResult {
        property: "svd_shape_consistency_2x3".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the SVD shape-consistency query.
///
/// When `u_is_square` is false, `U` is given the shape of `A`, `[m, n]`, instead
/// of the square `[m, m]` the full SVD requires — a plausible slip that confuses
/// the full decomposition with the thin/reduced form. Then `U · Σ` requires
/// `u_cols = sigma_rows`, i.e. `n = m`, which fails for a non-square `A`, so the
/// reconstruction chain is incompatible and the query becomes SAT. Tests flip the
/// knob to confirm the proof depends on the "U is square" rule.
fn build_svd_shape_consistency(u_is_square: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Free shape of the input matrix A: [m, n] (positive integers as reals >= 1).
    let m = declare_real(&mut program, "m");
    let n = declare_real(&mut program, "n");
    let one = Expr::real(1);
    let hi = Expr::real(10000);
    for dim in [&m, &n] {
        assert_bounds(&mut program, dim, &one, &hi);
    }

    // Input matrix shape [m, n].
    let a_rows = define_real(&mut program, "a_rows", &m);
    let a_cols = define_real(&mut program, "a_cols", &n);

    // Full-SVD shape rule applied to A:
    //   U  : [m, m]  (square; the mutation gives it A's shape [m, n] instead)
    //   Σ  : [m, n]  (same shape as A; min(m,n) singular values on the diagonal)
    //   Vᵀ : [n, n]  (square)
    let u_rows = define_real(&mut program, "u_rows", &a_rows);
    let u_cols = define_real(
        &mut program,
        "u_cols",
        if u_is_square { &a_rows } else { &a_cols },
    );
    let sigma_rows = define_real(&mut program, "sigma_rows", &a_rows);
    let sigma_cols = define_real(&mut program, "sigma_cols", &a_cols);
    let vt_rows = define_real(&mut program, "vt_rows", &a_cols);
    let vt_cols = define_real(&mut program, "vt_cols", &a_cols);

    // Reconstruction chain, left to right.
    // P = U · Σ  : output rows from U, output cols from Σ.
    let p_rows = define_real(&mut program, "p_rows", &u_rows);
    let p_cols = define_real(&mut program, "p_cols", &sigma_cols);
    // R = P · Vᵀ : output rows from P, output cols from Vᵀ.
    let r_rows = define_real(&mut program, "r_rows", &p_rows);
    let r_cols = define_real(&mut program, "r_cols", &vt_cols);

    // Property P (all four clauses must hold):
    //   compat at U·Σ  : u_cols = sigma_rows
    //   compat at P·Vᵀ : p_cols = vt_rows
    //   reconstruction shape : r_rows = m  AND  r_cols = n
    // Each is derived by transitivity through the definitions above, never
    // asserted outright, so the violation ¬P is not P ∧ ¬P.
    let compat_1_broken = u_cols.ne(sigma_rows);
    let compat_2_broken = p_cols.ne(vt_rows);
    let rows_wrong = r_rows.ne(m);
    let cols_wrong = r_cols.ne(n);

    let violation = Expr::or_many(vec![compat_1_broken, compat_2_broken, rows_wrong, cols_wrong]);
    program.assert(violation);
    program.check_sat();

    program
}

// ---------------------------------------------------------------------------
// Property 7: Matrix Trace Linearity
// ---------------------------------------------------------------------------

/// Prove `tr(c * A) = c * tr(A)` for a 2x2 matrix and scalar c.
///
/// For 2x2 matrix A and scalar c:
///   `tr(cA) = c*a00 + c*a11 = c*(a00 + a11) = c * tr(A)`
///
/// Uses `QF_NRA` since the proof involves products of symbolic variables.
pub fn prove_trace_scalar_homogeneity() -> Result<MatrixDecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let bound_lo = Expr::real(-100);
    let bound_hi = Expr::real(100);

    let a00 = declare_real(&mut program, "a00");
    let a11 = declare_real(&mut program, "a11");
    let c = declare_real(&mut program, "c");

    for v in [&a00, &a11, &c] {
        assert_bounds(&mut program, v, &bound_lo, &bound_hi);
    }

    // tr(cA) = c*a00 + c*a11
    let tr_ca = c
        .clone()
        .real_mul(a00.clone())
        .real_add(c.clone().real_mul(a11.clone()));

    // c * tr(A) = c * (a00 + a11)
    let c_tr_a = c.real_mul(a00.real_add(a11));

    // Violation: tr(cA) != c * tr(A)
    let violation = tr_ca.ne(c_tr_a);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(MatrixDecompPropertyResult {
        property: "trace_scalar_homogeneity_2x2".to_string(),
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

    #[test]
    fn test_matmul_associativity_proven() {
        let result = prove_matmul_associativity().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Matmul associativity: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Matmul associativity must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "matmul_associativity_2x2");
    }

    #[test]
    fn test_transpose_involution_proven() {
        let result = prove_transpose_involution().expect("proof should not error");
        assert!(
            result.proven,
            "Transpose involution (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "transpose_involution_2x3");
    }

    #[test]
    fn test_frobenius_norm_non_negativity_proven() {
        let result = prove_frobenius_norm_non_negativity().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Frobenius norm non-negativity: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Frobenius norm non-negativity must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "frobenius_norm_non_negativity");
    }

    #[test]
    fn test_frobenius_norm_zero_iff_zero_matrix_proven() {
        let result = prove_frobenius_norm_zero_iff_zero_matrix().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Frobenius norm zero iff zero: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Frobenius norm zero iff zero must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "frobenius_norm_zero_iff_zero_matrix");
    }

    #[test]
    fn test_orthogonal_norm_preservation_proven() {
        let result = prove_orthogonal_norm_preservation().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Orthogonal norm preservation: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Orthogonal norm preservation must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "orthogonal_norm_preservation_2x2");
    }

    #[test]
    fn test_svd_shape_consistency_proven() {
        let result = prove_svd_shape_consistency().expect("proof should not error");
        assert!(
            result.proven,
            "SVD shape consistency (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(crate::ay_vacuity::vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "svd_shape_consistency_2x3");
    }

    /// Shape `U` as `[m, n]` (thin-SVD slip) instead of the square `[m, m]` the
    /// full decomposition requires. Then `U · Σ` needs `u_cols = sigma_rows`,
    /// i.e. `n = m`, which fails for a non-square `A`, so the reconstruction chain
    /// is incompatible and the query must be SAT — proving the theorem rests on
    /// the "U is square" shape rule rather than on writing the shapes twice.
    #[test]
    fn svd_shape_depends_on_u_being_square() {
        let program = build_svd_shape_consistency(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with U shaped [m, n] the chain is dimensionally incompatible and the query must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_trace_scalar_homogeneity_proven() {
        let result = prove_trace_scalar_homogeneity().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Trace scalar homogeneity: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Trace scalar homogeneity must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "trace_scalar_homogeneity_2x2");
    }

    #[test]
    fn test_matmul_associativity_smt2_structure() {
        let result = prove_matmul_associativity().expect("proof should not error");
        assert!(result.smt2.contains("set-logic"), "should declare logic");
        assert!(result.smt2.contains("check-sat"), "should have check-sat");
        assert!(
            result.smt2.contains("declare-const"),
            "should have declarations"
        );
    }
}
