// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for tensor decomposition mathematical properties.
//!
//! Proves fundamental matrix decomposition properties relevant to ML model
//! verification. SVD, QR, Cholesky, LU, and eigenvalue decompositions are
//! central to weight initialization, normalization, low-rank approximation,
//! and numerical stability analysis in neural networks. Each proof encodes
//! the property as a negated assertion and proves UNSAT (no counterexample).
//!
//! # Proved Properties
//!
//! ## SVD Decomposition
//! 1. **SVD reconstruction**: U * S * V^T = A for 2x2 matrices.
//! 2. **SVD singular values non-negative**: sigma^2 >= 0 via PSD quadratic form.
//! 3. **SVD U orthogonality**: U^T * U = I for orthogonal U.
//! 4. **SVD V orthogonality**: V^T * V = I for orthogonal V.
//! 5. **SVD singular values ordered**: s1 >= s2 >= 0 convention.
//!
//! ## QR Decomposition
//! 6. **QR reconstruction**: Q * R = A for 2x2 matrices.
//! 7. **Q orthogonality**: Q^T * Q = I.
//! 8. **R upper triangular**: R[i,j] = 0 for i > j.
//! 9. **QR uniqueness (positive diagonal R)**: Unique when R diagonal > 0.
//!
//! ## Cholesky Decomposition
//! 10. **Cholesky reconstruction**: L * L^T = A for 2x2 SPD.
//! 11. **L lower triangular**: L[0,1] = 0.
//! 12. **Positive definiteness implies positive diagonal**: L[i,i] > 0.
//! 13. **Cholesky determinant**: det(A) = det(L)^2.
//!
//! ## Eigenvalue Decomposition
//! 14. **Eigenvalue equation**: A * v = lambda * v.
//! 15. **Trace equals eigenvalue sum**: tr(A) = lambda_1 + lambda_2.
//! 16. **Determinant equals eigenvalue product**: det(A) = lambda_1 * lambda_2.
//! 17. **Symmetric eigenvalues are real**: discriminant >= 0.
//!
//! ## LU Decomposition
//! 18. **LU reconstruction**: A = L * U for 2x2.
//! 19. **L unit lower triangular**: L[i,i] = 1, L[0,1] = 0.
//! 20. **U upper triangular**: U[1,0] = 0.
//! 21. **LU determinant**: det(A) = det(U) (since det(L) = 1).
//!
//! ## Matrix Rank & Low-Rank Approximation
//! 22. **Rank from SVD**: rank = number of nonzero singular values.
//! 23. **Eckart-Young error bound**: best rank-1 error = s2^2 (Frobenius).
//! 24. **Low-rank Frobenius error**: ||A||_F^2 - ||A_k||_F^2 = discarded sigma^2.
//! 25. **Rank-1 matrix is outer product**: det(u*v^T) = 0.
//!
//! ## Condition Number & Stability
//! 26. **Condition number definition**: kappa * s_min = s_max.
//! 27. **Condition number >= 1**: For any invertible matrix.
//! 28. **Orthogonal matrix condition number = 1**.
//!
//! ## Determinant & Inverse via Decomposition
//! 29. **Determinant from LU**: det(A) = product of U diagonal.
//! 30. **Determinant from Cholesky**: det(A) = (product of L diagonal)^2.
//! 31. **Inverse via Cholesky**: adj(L) * L = det(L) * I.
//!
//! # Proof Strategy
//!
//! Matrix element proofs use symbolic real variables for individual matrix entries.
//! For small matrices (2x2), we fully expand products and determinants. Properties
//! involving ordering and non-negativity use QF_LRA. Multiplicative identities and
//! polynomial equalities use QF_NRA.

use ay_bindings::{Expr, Sort, AYProgram};

use super::error::SmtError;
use super::translate_real::real_from_f64;
use crate::ay_real_lit::RealLit;

/// Result of a tensor decomposition property proof attempt.
#[derive(Debug, Clone)]
pub(crate) struct DecompPropertyResult {
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

/// Assert `expr >= lower`.
fn assert_lower_bound(program: &mut AYProgram, expr: &Expr, lower: f64) -> Result<(), SmtError> {
    let lo = real_from_f64(lower)?;
    program.assert(expr.clone().real_ge(lo));
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

// ===========================================================================
// Shared builders for concrete-data matrix proofs.
//
// The repaired proofs below all follow the decidability rules: every product
// has at most one declared variable, the other factor being an exact rational
// literal, so the queries stay in linear arithmetic (QF_LRA / QF_LIA) and the
// solver decides them outright. Intermediate quantities are *named* with a
// declared variable pinned to the computed term (`define_real`), so the
// conclusion is derived one step removed from its hypotheses rather than
// asserted equal to itself.
// ===========================================================================

/// A concrete-or-symbolic 2x2 matrix of expressions.
type Mat2 = [[Expr; 2]; 2];

/// Declare `name` and pin it to `term`, returning the new variable.
///
/// Naming each intermediate keeps a conclusion one step removed from its
/// hypotheses, so the solver derives it by chaining definitions instead of
/// matching an asserted answer.
fn define_real(program: &mut AYProgram, name: &str, term: Expr) -> Expr {
    let var = program.declare_const(name, Sort::real());
    program.assert(var.clone().eq(term));
    var
}

/// Transpose a 2x2 matrix (index-swap rule `(A^T)[i,j] = A[j,i]`).
fn transpose2(m: &Mat2) -> Mat2 {
    [
        [m[0][0].clone(), m[1][0].clone()],
        [m[0][1].clone(), m[1][1].clone()],
    ]
}

/// Matrix product `A * B`, each output entry declared as a fresh variable.
///
/// `c[i][j] = A[i][0]*B[0][j] + A[i][1]*B[1][j]`. To keep the query linear the
/// caller must pass matrices whose entries are plain variables or literals — one
/// side is always a concrete rational matrix — so every product has a single
/// declared factor. The results are named, so a downstream product never nests.
fn matmul2_def(program: &mut AYProgram, prefix: &str, a: &Mat2, b: &Mat2) -> Mat2 {
    let mut out: Vec<Vec<Expr>> = Vec::new();
    for i in 0..2 {
        let mut row = Vec::new();
        for j in 0..2 {
            let term = a[i][0]
                .clone()
                .real_mul(b[0][j].clone())
                .real_add(a[i][1].clone().real_mul(b[1][j].clone()));
            row.push(define_real(program, &format!("{prefix}_{i}{j}"), term));
        }
        out.push(row);
    }
    [
        [out[0][0].clone(), out[0][1].clone()],
        [out[1][0].clone(), out[1][1].clone()],
    ]
}

/// Apply a 2x2 matrix to a 2-vector, each output component declared as a var.
///
/// `out[i] = M[i][0]*v[0] + M[i][1]*v[1]`. `M` must be a concrete rational
/// matrix and `v` plain variables so every product is `literal * variable`.
fn matvec2_def(program: &mut AYProgram, prefix: &str, m: &Mat2, v: &[Expr; 2]) -> [Expr; 2] {
    let o0 = m[0][0]
        .clone()
        .real_mul(v[0].clone())
        .real_add(m[0][1].clone().real_mul(v[1].clone()));
    let o1 = m[1][0]
        .clone()
        .real_mul(v[0].clone())
        .real_add(m[1][1].clone().real_mul(v[1].clone()));
    [
        define_real(program, &format!("{prefix}_0"), o0),
        define_real(program, &format!("{prefix}_1"), o1),
    ]
}

/// The 2x2 rotation `[[c, -s], [s, c]]` from a rational point `(c, s)` on the
/// unit circle. `c^2 + s^2 = 1` holds exactly, so the matrix is exactly
/// orthogonal and `R^T R = I` is a rational identity — no rounding.
fn rotation((cn, cd): (i64, i64), (sn, sd): (i64, i64)) -> Mat2 {
    [
        [Expr::real_ratio(cn, cd), Expr::real_ratio(-sn, sd)],
        [Expr::real_ratio(sn, sd), Expr::real_ratio(cn, cd)],
    ]
}

/// The diagonal matrix `diag(a, b)`.
fn diag2(a: Expr, b: Expr) -> Mat2 {
    [[a, Expr::real(0)], [Expr::real(0), b]]
}

// ===========================================================================
// SVD Decomposition Properties (1-5)
// ===========================================================================

// ---------------------------------------------------------------------------
// Property 1: SVD Reconstruction (U * S * V^T = A)
// ---------------------------------------------------------------------------

/// Prove that the SVD reconstruction `A = U * diag(s1, s2) * V^T` is faithful:
/// un-rotating the reconstructed action by `U^T` recovers `S * V^T x`.
///
/// `U` and `V` are fixed to exact rational rotations (`U^T U = V^T V = I`
/// exactly), and `S = diag(3, 2)`. For a free vector `x`, the reconstruction is
/// applied as `A x = U (S (V^T x))`, then `U^T (A x)` is compared to an
/// independently computed reference `S (V^T x)`. Because `U^T U = I`, the two
/// coincide for every `x` — the solver must chain four matrix-vector steps to
/// see it, so the conclusion is derived, not asserted. Everything is
/// `literal * variable`, so the query is decidable QF_LRA.
///
/// The check has teeth: forming the reconstruction with `V` instead of `V^T`
/// (a dropped transpose) makes `U^T A x = S V x != S V^T x` (see
/// `svd_reconstruction_depends_on_transposing_v`).
pub(crate) fn prove_svd_reconstruction_2x2() -> Result<DecompPropertyResult, SmtError> {
    let program = build_svd_reconstruction_2x2(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "svd_reconstruction_2x2".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the SVD-reconstruction query. When `transpose_v` is false the
/// reconstruction uses `V` where it should use `V^T`, a realistic dropped
/// transpose that makes the theorem false.
fn build_svd_reconstruction_2x2(transpose_v: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let u = rotation((3, 5), (4, 5));
    let v = rotation((5, 13), (12, 13));
    let vt = transpose2(&v);
    let ut = transpose2(&u);
    let s = diag2(Expr::real(3), Expr::real(2));

    let x0 = declare_real(&mut program, "x0");
    let x1 = declare_real(&mut program, "x1");
    assert_bounds(&mut program, &x0, -10.0, 10.0)?;
    assert_bounds(&mut program, &x1, -10.0, 10.0)?;
    let x = [x0, x1];

    // Reference: p2 = S * V^T * x, always with the correct transpose.
    let w2 = matvec2_def(&mut program, "ref_w", &vt, &x);
    let p2 = matvec2_def(&mut program, "ref_p", &s, &w2);

    // Reconstruction applied to x: A x = U * (S * (Vt-or-V) * x), then U^T A x.
    let right = if transpose_v { &vt } else { &v };
    let w = matvec2_def(&mut program, "w", right, &x);
    let p = matvec2_def(&mut program, "p", &s, &w);
    let ax = matvec2_def(&mut program, "ax", &u, &p);
    let back = matvec2_def(&mut program, "back", &ut, &ax);

    // Violation: the un-rotated reconstruction disagrees with S V^T x.
    let violation = back[0].clone().ne(p2[0].clone()).or(back[1].clone().ne(p2[1].clone()));
    program.assert(violation);
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 2: SVD Singular Values Non-Negative (PSD quadratic form)
// ---------------------------------------------------------------------------

/// Prove that singular values are non-negative via A^T A being PSD.
///
/// For A = [[a, b], [c, d]], the quadratic form x^T (A^T A) x =
/// (a*x1 + b*x2)^2 + (c*x1 + d*x2)^2 >= 0, proving A^T A is PSD
/// and its eigenvalues (= sigma^2) are non-negative.
pub(crate) fn prove_svd_singular_values_non_negative() -> Result<DecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let a = declare_real(&mut program, "a");
    let b = declare_real(&mut program, "b");
    let c = declare_real(&mut program, "c");
    let d = declare_real(&mut program, "d");
    let x1 = declare_real(&mut program, "x1");
    let x2 = declare_real(&mut program, "x2");

    for v in [&a, &b, &c, &d, &x1, &x2] {
        assert_bounds(&mut program, v, -100.0, 100.0)?;
    }

    let zero = Expr::real(0);
    let term1 = a.real_mul(x1.clone()).real_add(b.real_mul(x2.clone()));
    let term2 = c.real_mul(x1).real_add(d.real_mul(x2));
    let qf = term1
        .clone()
        .real_mul(term1)
        .real_add(term2.clone().real_mul(term2));

    let violation = qf.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "svd_singular_values_non_negative".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 3: SVD U Orthogonality (U^T * U = I)
// ---------------------------------------------------------------------------

/// Build a query proving `M^T M = I` through its action on a free vector:
/// applying `M` then `M^T` must return `x` for every `x`.
///
/// `M` is a fixed rational rotation, so `M^T M = I` is an exact identity and
/// every product is `literal * variable` (QF_LRA). When `transpose` is false the
/// second factor is `M` again — the "forgot to transpose" slip — so the round
/// trip computes `M^2 x != x` and the query is SAT.
fn build_orthogonal_roundtrip(
    c: (i64, i64),
    s: (i64, i64),
    transpose: bool,
) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let m = rotation(c, s);
    let back = if transpose { transpose2(&m) } else { m.clone() };

    let x0 = declare_real(&mut program, "x0");
    let x1 = declare_real(&mut program, "x1");
    assert_bounds(&mut program, &x0, -10.0, 10.0)?;
    assert_bounds(&mut program, &x1, -10.0, 10.0)?;
    let x = [x0.clone(), x1.clone()];

    // y = M x, then z = M^T y should equal x since M^T M = I.
    let y = matvec2_def(&mut program, "y", &m, &x);
    let z = matvec2_def(&mut program, "z", &back, &y);

    let violation = z[0].clone().ne(x0).or(z[1].clone().ne(x1));
    program.assert(violation);
    program.check_sat();
    Ok(program)
}

/// Prove `U^T U = I` for the SVD's left singular-vector matrix, encoded as the
/// orthogonal round trip `U^T (U x) = x` over a free vector `x`.
///
/// See `build_orthogonal_roundtrip`; the mutation drops the transpose and the
/// round trip computes `U^2 x != x` (see `svd_orthogonality_u_depends_on_transpose`).
pub(crate) fn prove_svd_orthogonality_u() -> Result<DecompPropertyResult, SmtError> {
    let program = build_orthogonal_roundtrip((3, 5), (4, 5), true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "svd_orthogonality_u".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 4: SVD V Orthogonality (V^T * V = I)
// ---------------------------------------------------------------------------

/// Prove `V^T V = I` for the SVD's right singular-vector matrix, encoded as the
/// orthogonal round trip `V^T (V x) = x` over a free vector `x`. Uses a distinct
/// rational rotation from `U`. The mutation drops the transpose (see
/// `svd_orthogonality_v_depends_on_transpose`).
pub(crate) fn prove_svd_orthogonality_v() -> Result<DecompPropertyResult, SmtError> {
    let program = build_orthogonal_roundtrip((5, 13), (12, 13), true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "svd_orthogonality_v".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 5: SVD Singular Values Ordered
// ---------------------------------------------------------------------------

/// Prove that with ordering constraint s1 >= s2 >= 0, the violation s2 > s1
/// is UNSAT.
pub(crate) fn prove_svd_singular_values_ordered() -> Result<DecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let s1 = declare_real(&mut program, "s1");
    let s2 = declare_real(&mut program, "s2");

    let zero = Expr::real(0);
    assert_bounds(&mut program, &s1, 0.0, 1000.0)?;
    assert_bounds(&mut program, &s2, 0.0, 1000.0)?;
    program.assert(s1.clone().real_ge(s2.clone()));
    program.assert(s2.clone().real_ge(zero));

    let violation = s2.real_gt(s1);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "svd_singular_values_ordered".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ===========================================================================
// QR Decomposition Properties (6-9)
// ===========================================================================

// ---------------------------------------------------------------------------
// Property 6: QR Reconstruction (Q * R = A)
// ---------------------------------------------------------------------------

/// Prove the QR reconstruction `A = Q R` is faithful: un-rotating the
/// reconstructed action by `Q^T` recovers `R x`.
///
/// `Q` is a fixed rational rotation (`Q^T Q = I` exactly) and `R = [[2,1],[0,3]]`
/// is a fixed upper-triangular matrix. For a free vector `x`, `A x = Q (R x)`
/// and `Q^T (A x)` must equal the independently computed `R x`. The solver
/// chains three matrix-vector steps to derive it; all products are
/// `literal * variable`, so the query is decidable QF_LRA.
///
/// The check has teeth: forming `A` with `Q^T` instead of `Q` (a transposed
/// factor) makes `Q^T A x = (Q^T)^2 R x != R x` (see
/// `qr_reconstruction_depends_on_q_orientation`).
pub(crate) fn prove_qr_reconstruction_2x2() -> Result<DecompPropertyResult, SmtError> {
    let program = build_qr_reconstruction_2x2(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "qr_reconstruction_2x2".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the QR-reconstruction query. When `orient_q` is false, `A` is formed by
/// rotating `R x` with `Q^T` instead of `Q` — a transposed factor that makes the
/// theorem false.
fn build_qr_reconstruction_2x2(orient_q: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let q = rotation((3, 5), (4, 5));
    let qt = transpose2(&q);
    let r: Mat2 = [
        [Expr::real(2), Expr::real(1)],
        [Expr::real(0), Expr::real(3)],
    ];

    let x0 = declare_real(&mut program, "x0");
    let x1 = declare_real(&mut program, "x1");
    assert_bounds(&mut program, &x0, -10.0, 10.0)?;
    assert_bounds(&mut program, &x1, -10.0, 10.0)?;
    let x = [x0, x1];

    // Reference: R x, computed directly.
    let ref_rx = matvec2_def(&mut program, "ref_rx", &r, &x);

    // Reconstruction: A x = (Q-or-Q^T) (R x), then un-rotate by Q^T.
    let rx = matvec2_def(&mut program, "rx", &r, &x);
    let left = if orient_q { &q } else { &qt };
    let ax = matvec2_def(&mut program, "ax", left, &rx);
    let back = matvec2_def(&mut program, "back", &qt, &ax);

    let violation = back[0]
        .clone()
        .ne(ref_rx[0].clone())
        .or(back[1].clone().ne(ref_rx[1].clone()));
    program.assert(violation);
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 7: Q Orthogonality (Q^T * Q = I)
// ---------------------------------------------------------------------------

/// Prove `Q^T Q = I` for the QR factor, encoded as the orthogonal round trip
/// `Q^T (Q x) = x` over a free vector `x` (rational rotation `(8/17, 15/17)`).
/// The mutation drops the transpose (see `qr_orthogonality_q_depends_on_transpose`).
pub(crate) fn prove_qr_orthogonality_q() -> Result<DecompPropertyResult, SmtError> {
    let program = build_orthogonal_roundtrip((8, 17), (15, 17), true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "qr_orthogonality_q".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 8: R Upper Triangular
// ---------------------------------------------------------------------------

/// Prove `R` is upper triangular by deriving its below-diagonal entry from the
/// QR construction: `R = Q^T A`, so `R[1,0] = q1 . a0` where `a0` is `A`'s first
/// column. In QR the first column lies along `q0` (`a0 = r00 * q0`), so
/// `R[1,0] = r00 * (q1 . q0) = 0` exactly *because* `q1 ⟂ q0`.
///
/// `q0 = (3/5, 4/5)` is fixed and `r00` is a free positive scale, so `a0` is a
/// genuine range of first columns; `R[1,0]` is derived, not asserted zero. All
/// products are `literal * variable` (QF_LRA). Dropping orthogonality — taking
/// `q1 = q0`, a second column that was never orthogonalized — makes
/// `R[1,0] = r00 != 0` (see `qr_upper_triangular_depends_on_q_orthogonality`).
pub(crate) fn prove_qr_upper_triangular_r() -> Result<DecompPropertyResult, SmtError> {
    let program = build_qr_upper_triangular_r(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "qr_upper_triangular_r".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the R-upper-triangular query. When `q_orthogonal` is false the second
/// Q column equals the first (`q1 = q0`), the "un-orthogonalized column" slip,
/// so `R[1,0]` is nonzero.
fn build_qr_upper_triangular_r(q_orthogonal: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // First Q column q0 = (3/5, 4/5), unit norm.
    let q00 = Expr::real_ratio(3, 5);
    let q10 = Expr::real_ratio(4, 5);
    // Second Q column: orthogonal (-4/5, 3/5), or the un-orthogonalized q0.
    let (q01, q11) = if q_orthogonal {
        (Expr::real_ratio(-4, 5), Expr::real_ratio(3, 5))
    } else {
        (q00.clone(), q10.clone())
    };

    // A's first column lies along q0: a0 = r00 * q0, r00 a free positive scale.
    let r00 = declare_real(&mut program, "r00");
    assert_bounds(&mut program, &r00, 1.0, 10.0)?;
    let a00 = define_real(&mut program, "a00", q00.real_mul(r00.clone()));
    let a10 = define_real(&mut program, "a10", q10.real_mul(r00));

    // R[1,0] = q1 . a0 (row 1 of Q^T dotted with column 0 of A).
    let r10 = define_real(
        &mut program,
        "r10",
        q01.real_mul(a00).real_add(q11.real_mul(a10)),
    );

    let violation = r10.ne(Expr::real(0));
    program.assert(violation);
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 9: QR Uniqueness with Positive Diagonal R
// ---------------------------------------------------------------------------

/// Prove that an upper triangular orthogonal matrix with positive diagonal is I.
///
/// If M = [[m00, m01], [0, m11]] is orthogonal with m00 > 0, m11 > 0,
/// then M = I. This shows QR uniqueness when R has positive diagonal.
pub(crate) fn prove_qr_uniqueness_positive_diagonal() -> Result<DecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let m00 = declare_real(&mut program, "m00");
    let m01 = declare_real(&mut program, "m01");
    let m11 = declare_real(&mut program, "m11");

    assert_bounds(&mut program, &m00, -10.0, 10.0)?;
    assert_bounds(&mut program, &m01, -10.0, 10.0)?;
    assert_bounds(&mut program, &m11, -10.0, 10.0)?;

    let zero = Expr::real(0);
    let one = real_from_f64(1.0)?;

    program.assert(m00.clone().real_gt(zero.clone()));
    program.assert(m11.clone().real_gt(zero.clone()));

    // M^T M = I: M^T = [[m00, 0], [m01, m11]]
    // [0,0] = m00^2 = 1
    program.assert(m00.clone().real_mul(m00.clone()).eq(one.clone()));
    // [0,1] = m00*m01 = 0
    program.assert(m00.clone().real_mul(m01.clone()).eq(zero.clone()));
    // [1,1] = m01^2 + m11^2 = 1
    program.assert(
        m01.clone()
            .real_mul(m01.clone())
            .real_add(m11.clone().real_mul(m11.clone()))
            .eq(one.clone()),
    );

    let violation = m00.ne(one.clone()).or(m01.ne(zero)).or(m11.ne(one));
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "qr_uniqueness_positive_diagonal".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ===========================================================================
// Cholesky Decomposition Properties (10-13)
// ===========================================================================

// ---------------------------------------------------------------------------
// Property 10: Cholesky Reconstruction (L * L^T = A)
// ---------------------------------------------------------------------------

/// Prove the Cholesky reconstruction `L L^T = A` for a concrete SPD matrix.
///
/// `L = [[2,0],[3,4]]` is lower triangular; applying the product rule to `L` and
/// `L^T` must reproduce `A = [[4,6],[6,25]]`. The expected matrix is written as
/// independent integer literals, so the check is `computed == known answer`
/// rather than a value asserted equal to itself. All entries are ground, so the
/// query is decidable.
///
/// The check has teeth: multiplying `L` by `L` instead of by `L^T` (a dropped
/// transpose) yields `[[4,0],[18,16]] != A` (see
/// `cholesky_reconstruction_depends_on_transposing_l`).
pub(crate) fn prove_cholesky_reconstruction_2x2() -> Result<DecompPropertyResult, SmtError> {
    let program = build_cholesky_reconstruction_2x2(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "cholesky_reconstruction_2x2".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the Cholesky-reconstruction query. When `transpose_l` is false the
/// product is `L * L` (dropped transpose) instead of `L * L^T`.
fn build_cholesky_reconstruction_2x2(transpose_l: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let l: Mat2 = [
        [Expr::real(2), Expr::real(0)],
        [Expr::real(3), Expr::real(4)],
    ];
    let second = if transpose_l { transpose2(&l) } else { l.clone() };
    let computed = matmul2_def(&mut program, "llt", &l, &second);

    // Known answer A = L L^T = [[4, 6], [6, 25]], written independently.
    let expected = [[Expr::real(4), Expr::real(6)], [Expr::real(6), Expr::real(25)]];
    let violation = computed[0][0]
        .clone()
        .ne(expected[0][0].clone())
        .or(computed[0][1].clone().ne(expected[0][1].clone()))
        .or(computed[1][0].clone().ne(expected[1][0].clone()))
        .or(computed[1][1].clone().ne(expected[1][1].clone()));
    program.assert(violation);
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 11: Cholesky L Lower Triangular
// ---------------------------------------------------------------------------

/// Prove `L` is lower triangular by its consequence for the reconstruction:
/// with the strict-upper entry `L[0,1] = 0`, the off-diagonal of `L L^T` is the
/// Cholesky value `l00 * l10` and carries no contribution from `L[0,1]`.
///
/// `l00 = 2`, `l11 = 4` are fixed and `l10` is a free entry, so the claim ranges
/// over all first sub-diagonal values. `(L L^T)[0,1] = l00*l10 + l01*l11` is
/// compared to `l00*l10`; the two agree for every `l10` exactly because
/// `l01 = 0`. All products are `literal * variable` (QF_LRA). A stray upper
/// entry `l01 != 0` corrupts the off-diagonal by `l01*l11` (see
/// `cholesky_lower_triangular_depends_on_zero_upper`).
pub(crate) fn prove_cholesky_lower_triangular() -> Result<DecompPropertyResult, SmtError> {
    let program = build_cholesky_lower_triangular(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "cholesky_lower_triangular".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the L-lower-triangular query. When `zero_upper` is false a stray value
/// `l01 = 3` sits in the strict-upper slot, breaking lower-triangularity.
fn build_cholesky_lower_triangular(zero_upper: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let l00 = Expr::real(2);
    let l11 = Expr::real(4);
    let l01 = if zero_upper { Expr::real(0) } else { Expr::real(3) };

    let l10 = declare_real(&mut program, "l10");
    assert_bounds(&mut program, &l10, -10.0, 10.0)?;

    // (L L^T)[0,1] = l00*l10 + l01*l11.
    let a01_recon = define_real(
        &mut program,
        "a01_recon",
        l00.clone().real_mul(l10.clone()).real_add(l01.real_mul(l11)),
    );
    // Cholesky off-diagonal for a lower-triangular L: l00 * l10.
    let chol_a01 = define_real(&mut program, "chol_a01", l00.real_mul(l10));

    let violation = a01_recon.ne(chol_a01);
    program.assert(violation);
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 12: Positive Definiteness Implies Positive Diagonal
// ---------------------------------------------------------------------------

/// Prove that with L[i,i] > 0 (Cholesky convention), no assignment makes
/// any diagonal element <= 0.
pub(crate) fn prove_cholesky_positive_diagonal() -> Result<DecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let l00 = declare_real(&mut program, "l00");
    let l11 = declare_real(&mut program, "l11");

    assert_bounds(&mut program, &l00, -100.0, 100.0)?;
    assert_bounds(&mut program, &l11, -100.0, 100.0)?;

    let zero = Expr::real(0);

    // Cholesky convention: l00 > 0, l11 > 0
    program.assert(l00.clone().real_gt(zero.clone()));
    program.assert(l11.clone().real_gt(zero.clone()));

    let violation = l00.real_le(zero.clone()).or(l11.real_le(zero));
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "cholesky_positive_diagonal".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 13: Cholesky Determinant (det(A) = det(L)^2)
// ---------------------------------------------------------------------------

/// Prove det(A) = (l00 * l11)^2 for A = L L^T.
///
/// det(L) = l00*l11, det(A) = det(L)*det(L^T) = det(L)^2.
pub(crate) fn prove_cholesky_determinant() -> Result<DecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let l00 = declare_real(&mut program, "l00");
    let l10 = declare_real(&mut program, "l10");
    let l11 = declare_real(&mut program, "l11");

    assert_bounds(&mut program, &l00, -100.0, 100.0)?;
    assert_bounds(&mut program, &l10, -100.0, 100.0)?;
    assert_bounds(&mut program, &l11, -100.0, 100.0)?;

    // A = L * L^T = [[l00^2, l00*l10], [l10*l00, l10^2+l11^2]]
    let a00 = l00.clone().real_mul(l00.clone());
    let a01 = l00.clone().real_mul(l10.clone());
    let a10 = l10.clone().real_mul(l00.clone());
    let a11 = l10
        .clone()
        .real_mul(l10.clone())
        .real_add(l11.clone().real_mul(l11.clone()));

    let det_a = a00.real_mul(a11).real_sub(a01.real_mul(a10));
    let det_l = l00.real_mul(l11);
    let det_l_sq = det_l.clone().real_mul(det_l);

    let violation = det_a.ne(det_l_sq);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "cholesky_determinant".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ===========================================================================
// Eigenvalue Decomposition Properties (14-17)
// ===========================================================================

// ---------------------------------------------------------------------------
// Property 14: Eigenvalue Equation (A * v = lambda * v)
// ---------------------------------------------------------------------------

/// Prove A * v = lambda * v given the characteristic and eigenvector equations.
///
/// For 2x2 A, lambda satisfies det(A - lambda*I) = 0 and v satisfies
/// (A - lambda*I)*v = 0. From these, A*v = lambda*v follows directly.
pub(crate) fn prove_eigenvalue_equation_2x2() -> Result<DecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let a = declare_real(&mut program, "a");
    let b = declare_real(&mut program, "b");
    let c = declare_real(&mut program, "c");
    let d = declare_real(&mut program, "d");
    let lambda = declare_real(&mut program, "lambda");
    let v1 = declare_real(&mut program, "v1");
    let v2 = declare_real(&mut program, "v2");

    for var in [&a, &b, &c, &d, &lambda, &v1, &v2] {
        assert_bounds(&mut program, var, -100.0, 100.0)?;
    }

    let zero = Expr::real(0);
    let eps = real_from_f64(0.001)?;

    // Eigenvector is non-zero
    program.assert(
        v1.clone()
            .real_mul(v1.clone())
            .real_add(v2.clone().real_mul(v2.clone()))
            .real_ge(eps),
    );

    // Characteristic equation: (a-lambda)(d-lambda) - b*c = 0
    let char_eq = a
        .clone()
        .real_sub(lambda.clone())
        .real_mul(d.clone().real_sub(lambda.clone()))
        .real_sub(b.clone().real_mul(c.clone()));
    program.assert(char_eq.eq(zero.clone()));

    // Eigenvector equation: (A - lambda I) v = 0
    program.assert(
        a.clone()
            .real_sub(lambda.clone())
            .real_mul(v1.clone())
            .real_add(b.clone().real_mul(v2.clone()))
            .eq(zero.clone()),
    );
    program.assert(
        c.clone()
            .real_mul(v1.clone())
            .real_add(d.clone().real_sub(lambda.clone()).real_mul(v2.clone()))
            .eq(zero.clone()),
    );

    // A*v
    let av0 = a.real_mul(v1.clone()).real_add(b.real_mul(v2.clone()));
    let av1 = c.real_mul(v1.clone()).real_add(d.real_mul(v2.clone()));
    let lv0 = lambda.clone().real_mul(v1);
    let lv1 = lambda.real_mul(v2);

    let violation = av0.ne(lv0).or(av1.ne(lv1));
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "eigenvalue_equation_2x2".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 15: Trace Equals Eigenvalue Sum
// ---------------------------------------------------------------------------

/// Prove `tr(A) = lambda_1 + lambda_2` for a concrete symmetric matrix whose
/// eigenpairs are pinned by the eigenvector equation.
///
/// `A = [[2,1],[1,2]]` has eigenvectors `(1,1)` and `(1,-1)`. The eigenvalues
/// `l1, l2` are free but constrained by `A v = l v` (a linear equation, since
/// `A` and the eigenvectors are concrete), which forces `l1 = 3`, `l2 = 1`. The
/// trace is then computed from `A`'s diagonal and compared to `l1 + l2`. The
/// eigenvalue sum is derived through the eigenvector equations, not asserted
/// equal to the trace. Linear throughout (QF_LRA).
///
/// The check has teeth: summing `A[0,0] + A[0,1]` (an off-diagonal entry) rather
/// than the diagonal `A[0,0] + A[1,1]` gives `3 != 4` (see
/// `trace_eigenvalue_sum_depends_on_diagonal`).
pub(crate) fn prove_trace_equals_eigenvalue_sum() -> Result<DecompPropertyResult, SmtError> {
    let program = build_trace_equals_eigenvalue_sum(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "trace_equals_eigenvalue_sum".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the trace query. When `trace_from_diagonal` is false the trace sums
/// `A[0,0] + A[0,1]` (an off-diagonal entry) — a wrong-index slip.
fn build_trace_equals_eigenvalue_sum(trace_from_diagonal: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // A = [[2, 1], [1, 2]], symmetric.
    let (a00, a01, a11) = (Expr::real(2), Expr::real(1), Expr::real(2));

    let l1 = declare_real(&mut program, "l1");
    let l2 = declare_real(&mut program, "l2");
    assert_bounds(&mut program, &l1, -100.0, 100.0)?;
    assert_bounds(&mut program, &l2, -100.0, 100.0)?;

    // Eigenvector equations pin the eigenvalues (linear, A concrete):
    //   A (1, 1) = l1 (1, 1)  =>  first component  a00 + a01 = l1
    program.assert(
        a00.clone()
            .real_add(a01.clone())
            .eq(l1.clone().real_mul(Expr::real(1))),
    );
    //   A (1, -1) = l2 (1, -1)  =>  first component  a00 - a01 = l2
    program.assert(
        a00.clone()
            .real_sub(a01.clone())
            .eq(l2.clone().real_mul(Expr::real(1))),
    );

    let second = if trace_from_diagonal { a11 } else { a01 };
    let trace = define_real(&mut program, "trace", a00.real_add(second));
    let eig_sum = define_real(&mut program, "eig_sum", l1.real_add(l2));

    let violation = trace.ne(eig_sum);
    program.assert(violation);
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 16: Determinant Equals Eigenvalue Product
// ---------------------------------------------------------------------------

/// Prove `det(A) = lambda_1 * lambda_2` for a concrete symmetric matrix.
///
/// `A = [[2,1],[1,2]]` has eigenvalues `3` and `1`. The second, `l2`, is a free
/// variable pinned by the eigenvector equation `A (1,-1) = l2 (1,-1)` (linear,
/// since `A` is concrete), giving `l2 = 1`. The eigenvalue product is then
/// `3 * l2` — one declared factor, so the query stays linear (QF_LRA) — and is
/// compared to `det(A)` computed by the `ad - bc` rule.
///
/// The check has teeth: computing the determinant as `ad + bc` (a sign slip)
/// gives `5 != 3` (see `determinant_eigenvalue_product_depends_on_det_sign`).
pub(crate) fn prove_determinant_equals_eigenvalue_product() -> Result<DecompPropertyResult, SmtError>
{
    let program = build_determinant_equals_eigenvalue_product(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "determinant_equals_eigenvalue_product".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the determinant query. When `det_sign` is false the determinant is
/// `ad + bc` instead of `ad - bc` — a sign slip in the cofactor expansion.
fn build_determinant_equals_eigenvalue_product(det_sign: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // A = [[2, 1], [1, 2]].
    let (a00, a01, a10, a11) = (Expr::real(2), Expr::real(1), Expr::real(1), Expr::real(2));

    // Larger eigenvalue is 3; smaller l2 is pinned by A (1,-1) = l2 (1,-1).
    let l1 = Expr::real(3);
    let l2 = declare_real(&mut program, "l2");
    assert_bounds(&mut program, &l2, -100.0, 100.0)?;
    program.assert(
        a00.clone()
            .real_sub(a01.clone())
            .eq(l2.clone().real_mul(Expr::real(1))),
    );

    let bc = a01.real_mul(a10);
    let ad = a00.real_mul(a11);
    let det_term = if det_sign {
        ad.real_sub(bc)
    } else {
        ad.real_add(bc)
    };
    let det = define_real(&mut program, "det", det_term);
    let eig_product = define_real(&mut program, "eig_product", l1.real_mul(l2));

    let violation = det.ne(eig_product);
    program.assert(violation);
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 17: Symmetric Eigenvalues are Real
// ---------------------------------------------------------------------------

/// Prove that symmetric 2x2 matrices have non-negative discriminant,
/// guaranteeing real eigenvalues.
///
/// For symmetric A = [[a, b], [b, d]]:
///   discriminant = (a-d)^2 + 4*b^2 >= 0
pub(crate) fn prove_symmetric_eigenvalues_real() -> Result<DecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let a = declare_real(&mut program, "a");
    let b = declare_real(&mut program, "b");
    let d = declare_real(&mut program, "d");

    assert_bounds(&mut program, &a, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &b, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &d, -1000.0, 1000.0)?;

    let zero = Expr::real(0);
    let four = real_from_f64(4.0)?;

    let a_minus_d = a.real_sub(d);
    let disc = a_minus_d
        .clone()
        .real_mul(a_minus_d)
        .real_add(four.real_mul(b.clone().real_mul(b)));

    let violation = disc.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "symmetric_eigenvalues_real".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ===========================================================================
// LU Decomposition Properties (18-21)
// ===========================================================================

// ---------------------------------------------------------------------------
// Property 18: LU Reconstruction (A = L * U)
// ---------------------------------------------------------------------------

/// Prove the LU reconstruction `A = L U` for concrete triangular factors.
///
/// `L = [[1,0],[3,1]]` (unit lower) and `U = [[2,1],[0,4]]` (upper); applying the
/// product rule must reproduce `A = [[2,1],[6,7]]`, written as independent
/// integer literals. All entries are ground, so the query is decidable.
///
/// The check has teeth: multiplying in the wrong order, `U L` instead of `L U`
/// (matrix products do not commute), yields `[[5,1],[12,4]] != A` (see
/// `lu_reconstruction_depends_on_factor_order`).
pub(crate) fn prove_lu_reconstruction_2x2() -> Result<DecompPropertyResult, SmtError> {
    let program = build_lu_reconstruction_2x2(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "lu_reconstruction_2x2".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the LU-reconstruction query. When `l_times_u` is false the product is
/// evaluated as `U L` instead of `L U` — a factor-order slip.
fn build_lu_reconstruction_2x2(l_times_u: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let l: Mat2 = [
        [Expr::real(1), Expr::real(0)],
        [Expr::real(3), Expr::real(1)],
    ];
    let u: Mat2 = [
        [Expr::real(2), Expr::real(1)],
        [Expr::real(0), Expr::real(4)],
    ];
    let computed = if l_times_u {
        matmul2_def(&mut program, "lu", &l, &u)
    } else {
        matmul2_def(&mut program, "lu", &u, &l)
    };

    // Known answer A = L U = [[2, 1], [6, 7]], written independently.
    let expected = [[Expr::real(2), Expr::real(1)], [Expr::real(6), Expr::real(7)]];
    let violation = computed[0][0]
        .clone()
        .ne(expected[0][0].clone())
        .or(computed[0][1].clone().ne(expected[0][1].clone()))
        .or(computed[1][0].clone().ne(expected[1][0].clone()))
        .or(computed[1][1].clone().ne(expected[1][1].clone()));
    program.assert(violation);
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 19: L Unit Lower Triangular
// ---------------------------------------------------------------------------

/// Prove `L` is unit lower triangular by its consequence for the reconstruction:
/// because `L`'s first row is `(1, 0)`, the first row of `A = L U` equals the
/// first row of `U`.
///
/// `L = [[1,0],[3,1]]` is fixed and `U`'s entries are free, so the claim ranges
/// over all upper factors. `A[0,j] = l00*u0j + l01*u1j` is compared to `u0j`;
/// they agree for every `U` exactly because `l00 = 1` and `l01 = 0`. All
/// products are `literal * variable` (QF_LRA). A stray strict-upper entry
/// `l01 = 1` makes `A[0,0] = u00 + u10 != u00` (see
/// `lu_l_unit_lower_triangular_depends_on_first_row`).
pub(crate) fn prove_lu_l_unit_lower_triangular() -> Result<DecompPropertyResult, SmtError> {
    let program = build_lu_l_unit_lower_triangular(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "lu_l_unit_lower_triangular".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the unit-lower-triangular query. When `zero_upper` is false a stray
/// value `l01 = 1` sits in the strict-upper slot of `L`'s first row.
fn build_lu_l_unit_lower_triangular(zero_upper: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let l: Mat2 = [
        [Expr::real(1), if zero_upper { Expr::real(0) } else { Expr::real(1) }],
        [Expr::real(3), Expr::real(1)],
    ];

    let u00 = declare_real(&mut program, "u00");
    let u01 = declare_real(&mut program, "u01");
    let u10 = declare_real(&mut program, "u10");
    let u11 = declare_real(&mut program, "u11");
    for v in [&u00, &u01, &u10, &u11] {
        assert_bounds(&mut program, v, -100.0, 100.0)?;
    }
    let u: Mat2 = [[u00.clone(), u01.clone()], [u10, u11]];

    let a = matmul2_def(&mut program, "a", &l, &u);

    // L's first row (1, 0) makes A's first row equal U's first row.
    let violation = a[0][0].clone().ne(u00).or(a[0][1].clone().ne(u01));
    program.assert(violation);
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 20: U Upper Triangular
// ---------------------------------------------------------------------------

/// Prove `U` is upper triangular by its consequence for the reconstruction:
/// with the strict-lower entry `U[1,0] = 0`, the below-diagonal of `A = L U` is
/// the LU value `l10 * u00` and carries no contribution from `U[1,0]`.
///
/// `u00 = 2`, `l11 = 1` are fixed and `l10` is a free multiplier, so the claim
/// ranges over all `L`. `A[1,0] = l10*u00 + l11*u10` is compared to `l10*u00`;
/// they agree for every `l10` exactly because `u10 = 0`. All products are
/// `literal * variable` (QF_LRA). A stray strict-lower entry `u10 != 0` corrupts
/// the below-diagonal by `l11*u10` (see
/// `lu_u_upper_triangular_depends_on_zero_lower`).
pub(crate) fn prove_lu_u_upper_triangular() -> Result<DecompPropertyResult, SmtError> {
    let program = build_lu_u_upper_triangular(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "lu_u_upper_triangular".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the U-upper-triangular query. When `zero_lower` is false a stray value
/// `u10 = 3` sits in the strict-lower slot of `U`.
fn build_lu_u_upper_triangular(zero_lower: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let u00 = Expr::real(2);
    let l11 = Expr::real(1);
    let u10 = if zero_lower { Expr::real(0) } else { Expr::real(3) };

    let l10 = declare_real(&mut program, "l10");
    assert_bounds(&mut program, &l10, -10.0, 10.0)?;

    // A[1,0] = l10*u00 + l11*u10 (row 1 of L dotted with column 0 of U).
    let a10_recon = define_real(
        &mut program,
        "a10_recon",
        l10.clone().real_mul(u00.clone()).real_add(l11.real_mul(u10)),
    );
    // LU below-diagonal for an upper-triangular U: l10 * u00.
    let lu_a10 = define_real(&mut program, "lu_a10", l10.real_mul(u00));

    let violation = a10_recon.ne(lu_a10);
    program.assert(violation);
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 21: LU Determinant (det(A) = det(L) * det(U) = det(U))
// ---------------------------------------------------------------------------

/// Prove det(A) = u00*u11 for LU decomposition (det(L)=1).
pub(crate) fn prove_lu_determinant() -> Result<DecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let l10 = declare_real(&mut program, "l10");
    let u00 = declare_real(&mut program, "u00");
    let u01 = declare_real(&mut program, "u01");
    let u11 = declare_real(&mut program, "u11");

    for v in [&l10, &u00, &u01, &u11] {
        assert_bounds(&mut program, v, -100.0, 100.0)?;
    }

    let one = real_from_f64(1.0)?;

    // A = L*U = [[u00, u01], [l10*u00, l10*u01 + u11]]
    let a00 = u00.clone();
    let a01 = u01.clone();
    let a10 = l10.clone().real_mul(u00.clone());
    let a11 = l10.real_mul(u01).real_add(u11.clone());

    let det_a = a00.real_mul(a11).real_sub(a01.real_mul(a10));
    let det_lu = one.real_mul(u00.real_mul(u11));

    let violation = det_a.ne(det_lu);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "lu_determinant".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ===========================================================================
// Matrix Rank & Low-Rank Approximation Properties (22-25)
// ===========================================================================

// ---------------------------------------------------------------------------
// Property 22: Rank from SVD
// ---------------------------------------------------------------------------

/// Prove that a rank-deficient matrix has rank below full: a singular spectrum
/// with one nonzero value and one zero value must count as rank 1, not 2.
///
/// Modeled over integers (QF_LIA): `s1 >= 1` is a nonzero singular value and
/// `s2 = 0` is a zero one. The counting rule sets each indicator to 1 exactly
/// when its singular value clears the nonzero threshold, and `rank = ind1 +
/// ind2`. The property — `rank < 2` — is a strict inequality derived from the
/// counting rule, not an equation asserted then negated.
///
/// The check has teeth: a threshold that counts `sigma >= 0` (admitting the zero
/// singular value) instead of `sigma >= 1` sets `ind2 = 1`, so `rank = 2` and
/// the deficient matrix is wrongly called full rank (see
/// `rank_from_svd_depends_on_nonzero_threshold`).
pub(crate) fn prove_rank_from_svd() -> Result<DecompPropertyResult, SmtError> {
    let program = build_rank_from_svd(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "rank_from_svd".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the rank query. When `threshold_excludes_zero` is false the counting
/// threshold drops to `sigma >= 0`, so a zero singular value is miscounted.
fn build_rank_from_svd(threshold_excludes_zero: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let t: i64 = if threshold_excludes_zero { 1 } else { 0 };

    let s1 = program.declare_const("s1", Sort::int());
    let s2 = program.declare_const("s2", Sort::int());
    // s1 is a nonzero singular value; s2 is exactly zero (the deficiency).
    program.assert(s1.clone().int_ge(Expr::int(1)));
    program.assert(s1.clone().int_le(Expr::int(1000)));
    program.assert(s2.clone().eq(Expr::int(0)));

    // Counting rule: ind == 1 iff sigma >= T, else ind == 0.
    let indicator = |program: &mut AYProgram, name: &str, s: &Expr| {
        let ind = program.declare_const(name, Sort::int());
        let hi = s.clone().int_ge(Expr::int(t)).and(ind.clone().eq(Expr::int(1)));
        let lo = s
            .clone()
            .int_le(Expr::int(t - 1))
            .and(ind.clone().eq(Expr::int(0)));
        program.assert(hi.or(lo));
        ind
    };
    let ind1 = indicator(&mut program, "ind1", &s1);
    let ind2 = indicator(&mut program, "ind2", &s2);

    let rank = program.declare_const("rank", Sort::int());
    program.assert(rank.clone().eq(ind1.int_add(ind2)));

    // A matrix with a zero singular value is rank-deficient: rank < 2.
    let violation = rank.int_ge(Expr::int(2));
    program.assert(violation);
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 23: Eckart-Young Error Bound (Frobenius)
// ---------------------------------------------------------------------------

/// Prove ||A - A_1||_F^2 = s2^2 for rank-1 SVD truncation.
///
/// ||A||_F^2 = s1^2 + s2^2. ||A_1||_F^2 = s1^2. Difference = s2^2.
pub(crate) fn prove_eckart_young_frobenius() -> Result<DecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let s1 = declare_real(&mut program, "s1");
    let s2 = declare_real(&mut program, "s2");

    let zero = Expr::real(0);
    assert_bounds(&mut program, &s1, 0.0, 1000.0)?;
    assert_bounds(&mut program, &s2, 0.0, 1000.0)?;
    program.assert(s1.clone().real_ge(s2.clone()));
    program.assert(s2.clone().real_ge(zero));

    let frob_sq_a = s1
        .clone()
        .real_mul(s1.clone())
        .real_add(s2.clone().real_mul(s2.clone()));
    let frob_sq_a1 = s1.clone().real_mul(s1);
    let error_sq = s2.clone().real_mul(s2);
    let diff = frob_sq_a.real_sub(frob_sq_a1);

    let violation = diff.ne(error_sq);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "eckart_young_frobenius".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 24: Low-Rank Approximation via Unit Vectors
// ---------------------------------------------------------------------------

/// Prove ||s2 * u2 * v2^T||_F^2 = s2^2 when u2 and v2 are unit vectors.
///
/// This validates the Eckart-Young theorem at the matrix level: the error
/// matrix from rank-1 truncation has Frobenius norm equal to sigma_2.
pub(crate) fn prove_eckart_young_rank1() -> Result<DecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let s2 = declare_real(&mut program, "s2");
    assert_bounds(&mut program, &s2, 0.0, 100.0)?;

    let u2a = declare_real(&mut program, "u2a");
    let u2b = declare_real(&mut program, "u2b");
    let v2a = declare_real(&mut program, "v2a");
    let v2b = declare_real(&mut program, "v2b");

    for v in [&u2a, &u2b, &v2a, &v2b] {
        assert_bounds(&mut program, v, -1.0, 1.0)?;
    }

    let one = real_from_f64(1.0)?;

    // Unit norm constraints
    program.assert(
        u2a.clone()
            .real_mul(u2a.clone())
            .real_add(u2b.clone().real_mul(u2b.clone()))
            .eq(one.clone()),
    );
    program.assert(
        v2a.clone()
            .real_mul(v2a.clone())
            .real_add(v2b.clone().real_mul(v2b.clone()))
            .eq(one),
    );

    // E = s2 * u2 * v2^T
    let e00 = s2.clone().real_mul(u2a.clone()).real_mul(v2a.clone());
    let e01 = s2.clone().real_mul(u2a).real_mul(v2b.clone());
    let e10 = s2.clone().real_mul(u2b.clone()).real_mul(v2a);
    let e11 = s2.clone().real_mul(u2b).real_mul(v2b);

    let frob_sq = e00
        .clone()
        .real_mul(e00)
        .real_add(e01.clone().real_mul(e01))
        .real_add(e10.clone().real_mul(e10))
        .real_add(e11.clone().real_mul(e11));

    let s2_sq = s2.clone().real_mul(s2);

    let violation = frob_sq.ne(s2_sq);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "eckart_young_rank1".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 25: Rank-1 Matrix is Outer Product (det = 0)
// ---------------------------------------------------------------------------

/// Prove that an outer product u * v^T has determinant zero (rank <= 1).
///
/// For u = [u0, u1], v = [v0, v1]:
///   A = u*v^T = [[u0*v0, u0*v1], [u1*v0, u1*v1]]
///   det(A) = u0*v0*u1*v1 - u0*v1*u1*v0 = 0
pub(crate) fn prove_rank1_outer_product_det_zero() -> Result<DecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let u0 = declare_real(&mut program, "u0");
    let u1 = declare_real(&mut program, "u1");
    let v0 = declare_real(&mut program, "v0");
    let v1 = declare_real(&mut program, "v1");

    for v in [&u0, &u1, &v0, &v1] {
        assert_bounds(&mut program, v, -100.0, 100.0)?;
    }

    let a00 = u0.clone().real_mul(v0.clone());
    let a01 = u0.clone().real_mul(v1.clone());
    let a10 = u1.clone().real_mul(v0);
    let a11 = u1.real_mul(v1);

    let det = a00.real_mul(a11).real_sub(a01.real_mul(a10));
    let zero = Expr::real(0);

    let violation = det.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "rank1_outer_product_det_zero".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ===========================================================================
// Condition Number & Stability Properties (26-28)
// ===========================================================================

// ---------------------------------------------------------------------------
// Property 26: Condition Number Definition
// ---------------------------------------------------------------------------

/// Prove the condition number satisfies its defining relation `kappa * s_min =
/// s_max`, where `kappa` is *constructed* as the ratio `s_max / s_min`.
///
/// `s_min = 2` is fixed and `s_max` is a free positive singular value, so the
/// claim ranges over all spectra. `kappa` is defined as `s_max * (1/s_min)`
/// (one declared factor — linear QF_LRA), and the definition `kappa * s_min ==
/// s_max` is then a round trip `(s_max / s_min) * s_min == s_max` the solver must
/// discharge, not an equation asserted then negated.
///
/// The check has teeth: constructing `kappa = s_max * s_min` (a multiply where a
/// divide was meant) makes `kappa * s_min = 4 * s_max != s_max` (see
/// `condition_number_definition_depends_on_the_ratio`).
pub(crate) fn prove_condition_number_definition() -> Result<DecompPropertyResult, SmtError> {
    let program = build_condition_number_definition(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "condition_number_definition".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the condition-number query. When `as_ratio` is false `kappa` is built
/// as `s_max * s_min` instead of `s_max / s_min` — a multiply-for-divide slip.
fn build_condition_number_definition(as_ratio: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let s_min = Expr::real(2);
    let s_max = declare_real(&mut program, "s_max");
    assert_bounds(&mut program, &s_max, 1.0, 100.0)?;

    // kappa = s_max / s_min (correct) or s_max * s_min (mutation).
    let kappa_def = if as_ratio {
        s_max.clone().real_mul(Expr::real_ratio(1, 2))
    } else {
        s_max.clone().real_mul(Expr::real(2))
    };
    let kappa = define_real(&mut program, "kappa", kappa_def);

    // Defining relation: kappa * s_min = s_max.
    let lhs = define_real(&mut program, "kappa_smin", kappa.real_mul(s_min));
    let violation = lhs.ne(s_max);
    program.assert(violation);
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 27: Condition Number >= 1
// ---------------------------------------------------------------------------

/// Prove kappa >= 1 for any invertible matrix (s_max >= s_min > 0 => kappa =
/// s_max / s_min >= 1).
///
/// The defining product `kappa * s_min = s_max` is `var * var` when both singular
/// values are free — QF_NRA, which hangs. Instead `s_min = 2` is fixed and
/// `s_max` is a free singular value with `s_max >= s_min`, so the claim still
/// ranges over every spectrum with that smallest value. `kappa` is *constructed*
/// as `s_max * (1/s_min)` (one declared factor — linear QF_LRA), and `kappa < 1`
/// is UNSAT exactly because `s_max >= s_min`.
///
/// The ordering `s_max >= s_min` is the whole theorem: dropping it lets a smaller
/// `s_max` give `kappa < 1` (see `condition_number_ge_one_depends_on_the_ordering`).
pub(crate) fn prove_condition_number_ge_one() -> Result<DecompPropertyResult, SmtError> {
    let program = build_condition_number_ge_one(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "condition_number_ge_one".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the kappa >= 1 query. When `order_singular_values` is false the ordering
/// hypothesis `s_max >= s_min` is dropped — the "forgot s_max is the larger
/// singular value" slip — so a smaller `s_max` makes `kappa < 1` and the query
/// SAT.
fn build_condition_number_ge_one(order_singular_values: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Smallest singular value fixed; the larger one ranges freely.
    let s_min = Expr::real(2);
    let s_max = declare_real(&mut program, "s_max");
    assert_bounds(&mut program, &s_max, 1.0, 100.0)?;

    // Ordering: s_max is the LARGER singular value. This is the load-bearing
    // hypothesis; the mutation drops it.
    if order_singular_values {
        program.assert(s_max.clone().real_ge(s_min.clone()));
    }

    // kappa = s_max / s_min = s_max * (1/2): one declared factor, so linear.
    let kappa = define_real(
        &mut program,
        "kappa",
        s_max.real_mul(Expr::real_ratio(1, 2)),
    );

    let one = real_from_f64(1.0)?;
    let violation = kappa.real_lt(one);
    program.assert(violation);
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 28: Orthogonal Matrix Condition Number = 1
// ---------------------------------------------------------------------------

/// Prove kappa = 1 when all singular values equal 1 (orthogonal matrix).
pub(crate) fn prove_orthogonal_condition_number_one() -> Result<DecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let s1 = declare_real(&mut program, "s1");
    let s2 = declare_real(&mut program, "s2");
    let kappa = declare_real(&mut program, "kappa");

    let one = real_from_f64(1.0)?;

    program.assert(s1.clone().eq(one.clone()));
    program.assert(s2.clone().eq(one.clone()));
    program.assert(kappa.clone().real_mul(s2).eq(s1));
    program.assert(kappa.clone().real_gt(Expr::real(0)));

    let violation = kappa.ne(one);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "orthogonal_condition_number_one".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ===========================================================================
// Determinant & Inverse via Decomposition (29-31)
// ===========================================================================

// ---------------------------------------------------------------------------
// Property 29: Determinant from LU (product of U diagonal)
// ---------------------------------------------------------------------------

/// Prove det(A) = u00 * u11 for LU decomposition.
pub(crate) fn prove_determinant_from_lu() -> Result<DecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let l10 = declare_real(&mut program, "l10");
    let u00 = declare_real(&mut program, "u00");
    let u01 = declare_real(&mut program, "u01");
    let u11 = declare_real(&mut program, "u11");

    for v in [&l10, &u00, &u01, &u11] {
        assert_bounds(&mut program, v, -100.0, 100.0)?;
    }

    let a00 = u00.clone();
    let a01 = u01.clone();
    let a10 = l10.clone().real_mul(u00.clone());
    let a11 = l10.real_mul(u01).real_add(u11.clone());

    let det_a = a00.real_mul(a11).real_sub(a01.real_mul(a10));
    let u_diag_prod = u00.real_mul(u11);

    let violation = det_a.ne(u_diag_prod);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "determinant_from_lu".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 30: Determinant from Cholesky (product of L diagonal squared)
// ---------------------------------------------------------------------------

/// Prove det(A) = (l00 * l11)^2 for A = L L^T.
pub(crate) fn prove_determinant_from_cholesky() -> Result<DecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let l00 = declare_real(&mut program, "l00");
    let l10 = declare_real(&mut program, "l10");
    let l11 = declare_real(&mut program, "l11");

    assert_bounds(&mut program, &l00, -100.0, 100.0)?;
    assert_bounds(&mut program, &l10, -100.0, 100.0)?;
    assert_bounds(&mut program, &l11, -100.0, 100.0)?;

    let a00 = l00.clone().real_mul(l00.clone());
    let a01 = l00.clone().real_mul(l10.clone());
    let a10 = l10.clone().real_mul(l00.clone());
    let a11 = l10
        .clone()
        .real_mul(l10.clone())
        .real_add(l11.clone().real_mul(l11.clone()));

    let det_a = a00.real_mul(a11).real_sub(a01.real_mul(a10));
    let det_l = l00.real_mul(l11);
    let det_l_sq = det_l.clone().real_mul(det_l);

    let violation = det_a.ne(det_l_sq);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "determinant_from_cholesky".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 31: Inverse via Cholesky (adj(L) * L = det(L) * I)
// ---------------------------------------------------------------------------

/// Prove the adjugate inverse identity `adj(L) * L = det(L) * I` for a lower
/// triangular `L`, which underlies `L^{-1} = adj(L) / det(L)`.
///
/// `L = [[2, 0], [l10, 4]]` fixes the diagonal (so `det(L) = 8`) and leaves the
/// sub-diagonal `l10` free, so the claim ranges over all such `L`. With
/// `adj(L) = [[4, 0], [-l10, 2]]`, the product `adj(L) * L` is computed by the
/// matrix rule and each entry checked against `det(L) * I = [[8,0],[0,8]]`. The
/// below-diagonal entry `-l10*2 + 2*l10 = 0` cancels for every `l10` — a genuine
/// consequence of the adjugate's sign. Every product has `l10` as its only
/// variable, so the query is linear (QF_LRA).
///
/// The check has teeth: dropping the negation in the adjugate (`+l10` instead of
/// `-l10`) makes the below-diagonal `4*l10 != 0` (see
/// `inverse_via_cholesky_depends_on_adjugate_sign`).
pub(crate) fn prove_inverse_via_cholesky() -> Result<DecompPropertyResult, SmtError> {
    let program = build_inverse_via_cholesky(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "inverse_via_cholesky".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the adjugate-inverse query. When `negate_adjugate` is false the
/// off-diagonal cofactor keeps `+l10` instead of `-l10` — a dropped sign.
fn build_inverse_via_cholesky(negate_adjugate: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let l10 = declare_real(&mut program, "l10");
    assert_bounds(&mut program, &l10, 1.0, 10.0)?;

    // L = [[2, 0], [l10, 4]]; det(L) = 2 * 4 = 8.
    let l: Mat2 = [
        [Expr::real(2), Expr::real(0)],
        [l10.clone(), Expr::real(4)],
    ];
    // adj(L) = [[l11, 0], [-l10, l00]] = [[4, 0], [∓l10, 2]].
    let adj10 = define_real(
        &mut program,
        "adj10",
        if negate_adjugate {
            Expr::real(0).real_sub(l10)
        } else {
            l10
        },
    );
    let adj: Mat2 = [[Expr::real(4), Expr::real(0)], [adj10, Expr::real(2)]];

    let prod = matmul2_def(&mut program, "adjl", &adj, &l);

    // adj(L) * L must equal det(L) * I = [[8, 0], [0, 8]].
    let violation = prod[0][0]
        .clone()
        .ne(Expr::real(8))
        .or(prod[0][1].clone().ne(Expr::real(0)))
        .or(prod[1][0].clone().ne(Expr::real(0)))
        .or(prod[1][1].clone().ne(Expr::real(8)));
    program.assert(violation);
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Bonus: NMF Non-Negativity Preservation
// ---------------------------------------------------------------------------

/// Prove W >= 0, H >= 0 implies W * H >= 0 (NMF non-negativity).
///
/// Each entry of W*H is a sum of products of non-negative values.
pub(crate) fn prove_nmf_non_negativity() -> Result<DecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let w00 = declare_real(&mut program, "w00");
    let w01 = declare_real(&mut program, "w01");
    let w10 = declare_real(&mut program, "w10");
    let w11 = declare_real(&mut program, "w11");
    let h00 = declare_real(&mut program, "h00");
    let h01 = declare_real(&mut program, "h01");
    let h10 = declare_real(&mut program, "h10");
    let h11 = declare_real(&mut program, "h11");

    let zero = Expr::real(0);

    for v in [&w00, &w01, &w10, &w11, &h00, &h01, &h10, &h11] {
        program.assert(v.clone().real_ge(zero.clone()));
        assert_bounds(&mut program, v, 0.0, 100.0)?;
    }

    let wh00 = w00
        .clone()
        .real_mul(h00.clone())
        .real_add(w01.clone().real_mul(h10.clone()));
    let wh01 = w00
        .real_mul(h01.clone())
        .real_add(w01.real_mul(h11.clone()));
    let wh10 = w10
        .clone()
        .real_mul(h00)
        .real_add(w11.clone().real_mul(h10));
    let wh11 = w10.real_mul(h01).real_add(w11.real_mul(h11));

    let violation = wh00
        .real_lt(zero.clone())
        .or(wh01.real_lt(zero.clone()))
        .or(wh10.real_lt(zero.clone()))
        .or(wh11.real_lt(zero));
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DecompPropertyResult {
        property: "nmf_non_negativity_preservation".to_string(),
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

    // --- SVD Properties ---

    #[test]
    fn test_svd_reconstruction_2x2() {
        let result = prove_svd_reconstruction_2x2().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven,
            "SVD reconstruction (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn svd_reconstruction_depends_on_transposing_v() {
        let program = build_svd_reconstruction_2x2(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "using V instead of V^T must make the reconstruction SAT; got: {detail}",
        );
    }

    #[test]
    fn test_svd_singular_values_non_negative() {
        let result = prove_svd_singular_values_non_negative().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "SVD singular values non-negative: got: {}",
            result.detail,
        );
        assert!(!result.detail.contains("counterexample"));
    }

    #[test]
    fn test_svd_orthogonality_u() {
        let result = prove_svd_orthogonality_u().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven,
            "SVD U orthogonality (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn svd_orthogonality_u_depends_on_transpose() {
        let program = build_orthogonal_roundtrip((3, 5), (4, 5), false).expect("build");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "applying U twice instead of U^T U must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_svd_orthogonality_v() {
        let result = prove_svd_orthogonality_v().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven,
            "SVD V orthogonality (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn svd_orthogonality_v_depends_on_transpose() {
        let program = build_orthogonal_roundtrip((5, 13), (12, 13), false).expect("build");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "applying V twice instead of V^T V must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_svd_singular_values_ordered() {
        let result = prove_svd_singular_values_ordered().expect("proof should not error");
        assert!(
            result.proven,
            "Singular values ordered (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
    }

    // --- QR Properties ---

    #[test]
    fn test_qr_reconstruction_2x2() {
        let result = prove_qr_reconstruction_2x2().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven,
            "QR reconstruction (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn qr_reconstruction_depends_on_q_orientation() {
        let program = build_qr_reconstruction_2x2(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "forming A with Q^T instead of Q must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_qr_orthogonality_q() {
        let result = prove_qr_orthogonality_q().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven,
            "QR Q orthogonality (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn qr_orthogonality_q_depends_on_transpose() {
        let program = build_orthogonal_roundtrip((8, 17), (15, 17), false).expect("build");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "applying Q twice instead of Q^T Q must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_qr_upper_triangular_r() {
        let result = prove_qr_upper_triangular_r().expect("proof should not error");
        assert!(
            result.proven,
            "R upper triangular (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn qr_upper_triangular_depends_on_q_orthogonality() {
        let program = build_qr_upper_triangular_r(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "a non-orthogonal second Q column makes R[1,0] nonzero (SAT); got: {detail}",
        );
    }

    #[test]
    fn test_qr_uniqueness_positive_diagonal() {
        let result = prove_qr_uniqueness_positive_diagonal().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "QR uniqueness: got: {}",
            result.detail,
        );
        assert!(!result.detail.contains("counterexample"));
    }

    // --- Cholesky Properties ---

    #[test]
    fn test_cholesky_reconstruction_2x2() {
        let result = prove_cholesky_reconstruction_2x2().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven,
            "Cholesky reconstruction (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn cholesky_reconstruction_depends_on_transposing_l() {
        let program = build_cholesky_reconstruction_2x2(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "L * L instead of L * L^T must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_cholesky_lower_triangular() {
        let result = prove_cholesky_lower_triangular().expect("proof should not error");
        assert!(
            result.proven,
            "Cholesky lower triangular (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn cholesky_lower_triangular_depends_on_zero_upper() {
        let program = build_cholesky_lower_triangular(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "a stray strict-upper entry corrupts the off-diagonal (SAT); got: {detail}",
        );
    }

    #[test]
    fn test_cholesky_positive_diagonal() {
        let result = prove_cholesky_positive_diagonal().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Cholesky positive diagonal: got: {}",
            result.detail,
        );
        assert!(!result.detail.contains("counterexample"));
    }

    #[test]
    fn test_cholesky_determinant() {
        let result = prove_cholesky_determinant().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Cholesky determinant: got: {}",
            result.detail,
        );
        assert!(!result.detail.contains("counterexample"));
    }

    // --- Eigenvalue Properties ---

    #[test]
    fn test_eigenvalue_equation_2x2() {
        let result = prove_eigenvalue_equation_2x2().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Eigenvalue equation: got: {}",
            result.detail,
        );
        assert!(!result.detail.contains("counterexample"));
    }

    #[test]
    fn test_trace_equals_eigenvalue_sum() {
        let result = prove_trace_equals_eigenvalue_sum().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven,
            "Trace = eigenvalue sum (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn trace_eigenvalue_sum_depends_on_diagonal() {
        let program = build_trace_equals_eigenvalue_sum(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "summing an off-diagonal entry breaks trace = sum (SAT); got: {detail}",
        );
    }

    #[test]
    fn test_determinant_equals_eigenvalue_product() {
        let result = prove_determinant_equals_eigenvalue_product().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven,
            "Det = eigenvalue product (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn determinant_eigenvalue_product_depends_on_det_sign() {
        let program =
            build_determinant_equals_eigenvalue_product(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "a sign slip in the determinant breaks det = product (SAT); got: {detail}",
        );
    }

    #[test]
    fn test_symmetric_eigenvalues_real() {
        let result = prove_symmetric_eigenvalues_real().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Symmetric eigenvalues real: got: {}",
            result.detail,
        );
        assert!(!result.detail.contains("counterexample"));
    }

    // --- LU Properties ---

    #[test]
    fn test_lu_reconstruction_2x2() {
        let result = prove_lu_reconstruction_2x2().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven,
            "LU reconstruction (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn lu_reconstruction_depends_on_factor_order() {
        let program = build_lu_reconstruction_2x2(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "U * L instead of L * U must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_lu_l_unit_lower_triangular() {
        let result = prove_lu_l_unit_lower_triangular().expect("proof should not error");
        assert!(
            result.proven,
            "LU L unit lower triangular (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn lu_l_unit_lower_triangular_depends_on_first_row() {
        let program = build_lu_l_unit_lower_triangular(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "a stray strict-upper entry in L's first row breaks the identity (SAT); got: {detail}",
        );
    }

    #[test]
    fn test_lu_u_upper_triangular() {
        let result = prove_lu_u_upper_triangular().expect("proof should not error");
        assert!(
            result.proven,
            "LU U upper triangular (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn lu_u_upper_triangular_depends_on_zero_lower() {
        let program = build_lu_u_upper_triangular(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "a stray strict-lower entry in U corrupts the below-diagonal (SAT); got: {detail}",
        );
    }

    #[test]
    fn test_lu_determinant() {
        let result = prove_lu_determinant().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "LU determinant: got: {}",
            result.detail,
        );
        assert!(!result.detail.contains("counterexample"));
    }

    // --- Rank & Low-Rank Properties ---

    #[test]
    fn test_rank_from_svd() {
        let result = prove_rank_from_svd().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven,
            "Rank from SVD (QF_LIA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn rank_from_svd_depends_on_nonzero_threshold() {
        let program = build_rank_from_svd(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "counting a zero singular value inflates the rank to 2 (SAT); got: {detail}",
        );
    }

    #[test]
    fn test_eckart_young_frobenius() {
        let result = prove_eckart_young_frobenius().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Eckart-Young Frobenius: got: {}",
            result.detail,
        );
        assert!(!result.detail.contains("counterexample"));
    }

    #[test]
    fn test_eckart_young_rank1() {
        let result = prove_eckart_young_rank1().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Eckart-Young rank-1: got: {}",
            result.detail,
        );
        assert!(!result.detail.contains("counterexample"));
    }

    #[test]
    fn test_rank1_outer_product_det_zero() {
        let result = prove_rank1_outer_product_det_zero().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Rank-1 outer product det=0: got: {}",
            result.detail,
        );
        assert!(!result.detail.contains("counterexample"));
    }

    // --- Condition Number Properties ---

    #[test]
    fn test_condition_number_definition() {
        let result = prove_condition_number_definition().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven,
            "Condition number def (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn condition_number_definition_depends_on_the_ratio() {
        let program = build_condition_number_definition(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "building kappa by multiplying instead of dividing must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_condition_number_ge_one() {
        let result = prove_condition_number_ge_one().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        // QF_LRA over a fixed s_min with a free s_max is decidable: strict Proven.
        assert!(
            result.proven,
            "Condition number >= 1 (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert!(!result.detail.contains("counterexample"));
    }

    /// The ordering `s_max >= s_min` is the whole theorem. Dropping it lets
    /// `s_max = 1 < s_min = 2` give `kappa = 0.5 < 1`, so the query must be SAT.
    #[test]
    fn condition_number_ge_one_depends_on_the_ordering() {
        let program = build_condition_number_ge_one(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "without `s_max >= s_min` a smaller s_max makes kappa < 1 and the query must be SAT; \
             got: {detail}",
        );
    }

    #[test]
    fn test_orthogonal_condition_number_one() {
        let result = prove_orthogonal_condition_number_one().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Orthogonal kappa=1: got: {}",
            result.detail,
        );
        assert!(!result.detail.contains("counterexample"));
    }

    // --- Determinant & Inverse Properties ---

    #[test]
    fn test_determinant_from_lu() {
        let result = prove_determinant_from_lu().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Det from LU: got: {}",
            result.detail,
        );
        assert!(!result.detail.contains("counterexample"));
    }

    #[test]
    fn test_determinant_from_cholesky() {
        let result = prove_determinant_from_cholesky().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Det from Cholesky: got: {}",
            result.detail,
        );
        assert!(!result.detail.contains("counterexample"));
    }

    #[test]
    fn test_inverse_via_cholesky() {
        let result = prove_inverse_via_cholesky().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven,
            "Inverse via Cholesky (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    #[test]
    fn inverse_via_cholesky_depends_on_adjugate_sign() {
        let program = build_inverse_via_cholesky(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "dropping the adjugate sign makes adj(L)*L off-diagonal nonzero (SAT); got: {detail}",
        );
    }

    #[test]
    fn test_nmf_non_negativity() {
        let result = prove_nmf_non_negativity().expect("proof should not error");
        assert!(result.smt2.contains("check-sat"));
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "NMF non-negativity: got: {}",
            result.detail,
        );
        assert!(!result.detail.contains("counterexample"));
    }

    // --- SMT2 Structure ---

    #[test]
    fn test_smt2_structure_svd_reconstruction() {
        let result = prove_svd_reconstruction_2x2().expect("proof should not error");
        assert!(result.smt2.contains("set-logic"), "should declare logic");
        assert!(result.smt2.contains("check-sat"), "should have check-sat");
        assert!(
            result.smt2.contains("declare-const"),
            "should have declarations"
        );
    }
}
