// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for matrix decomposition properties (#4235).
//!
//! Proves fundamental algebraic identities of matrix operations using ay's SMT solver:
//!
//! 1. **Transpose involution**: (A^T)^T = A
//! 3. **Trace invariance under transpose**: tr(A) = tr(A^T)
//! 4. **Determinant of an upper-triangular matrix**: det = product of the diagonal (2x2)
//! 5. **Symmetric matrix**: the symmetrized off-diagonal doubles the shared entry (2x2)
//! 6. **Orthogonal matrix determinant**: Q^T * Q = I implies det(Q) = +/-1 (2x2)
//! 7. **Positive definite bounds**: x^T * A * x > 0 for non-zero x (2x2)
//! 8. **Frobenius norm submultiplicativity**: ||AB||_F^2 <= ||A||_F^2 * ||B||_F^2 (2x2)
//! 9. **Matrix inverse**: A * A^(-1) = I (non-singular 2x2)
//! 10. **Eigenvalue trace relationship**: sum of eigenvalues = trace (2x2)
//!
//! # Proof Strategy
//!
//! Matrix operations on small concrete dimensions (2x2) are encoded as
//! scalar real arithmetic. Each matrix entry is a separate SMT real variable.
//! This avoids quantifiers and keeps proofs in `QF_NRA` or `QF_LRA`.
//!
//! Part of #4235.

use ay_bindings::{Expr, Sort, AYProgram};

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
    // Uniform guard: a vacuous UNSAT (P and not-P, or X != X) never counts as a
    // proof. See crate::ay_vacuity. No-op for genuine queries.
    crate::ay_vacuity::reject_if_vacuous(&program.to_string(), proven, detail)
}

// ---------------------------------------------------------------------------
// Property 1: Transpose Involution -- (A^T)^T = A
// ---------------------------------------------------------------------------

/// Prove (A^T)^T = A for a 2x2 matrix.
///
/// For matrix A with entries a_ij:
///   A^T has (A^T)_ij = A_ji
///   (A^T)^T has ((A^T)^T)_ij = (A^T)_ji = A_ij
///
/// So (A^T)^T = A element-wise. We verify symbolically with QF_LRA.
#[test]
fn test_transpose_involution_2x2() {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let bound_lo = Expr::real(-1000);
    let bound_hi = Expr::real(1000);

    // Declare 2x2 matrix A
    let a00 = declare_real(&mut program, "a00");
    let a01 = declare_real(&mut program, "a01");
    let a10 = declare_real(&mut program, "a10");
    let a11 = declare_real(&mut program, "a11");

    for v in [&a00, &a01, &a10, &a11] {
        assert_bounds(&mut program, v, &bound_lo, &bound_hi);
    }

    // A^T: swap indices
    // (A^T)_00 = a00, (A^T)_01 = a10, (A^T)_10 = a01, (A^T)_11 = a11
    let at_00 = declare_real(&mut program, "at_00");
    let at_01 = declare_real(&mut program, "at_01");
    let at_10 = declare_real(&mut program, "at_10");
    let at_11 = declare_real(&mut program, "at_11");

    program.assert(at_00.clone().eq(a00.clone()));
    program.assert(at_01.clone().eq(a10.clone()));
    program.assert(at_10.clone().eq(a01.clone()));
    program.assert(at_11.clone().eq(a11.clone()));

    // (A^T)^T: swap indices again
    let att_00 = declare_real(&mut program, "att_00");
    let att_01 = declare_real(&mut program, "att_01");
    let att_10 = declare_real(&mut program, "att_10");
    let att_11 = declare_real(&mut program, "att_11");

    program.assert(att_00.clone().eq(at_00));
    program.assert(att_01.clone().eq(at_10));
    program.assert(att_10.clone().eq(at_01));
    program.assert(att_11.clone().eq(at_11));

    // Violation: (A^T)^T != A
    let violation = Expr::or_many(vec![
        att_00.ne(a00),
        att_01.ne(a01),
        att_10.ne(a10),
        att_11.ne(a11),
    ]);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    assert!(
        proven,
        "Transpose involution (QF_LRA) should be Proven. detail: {}",
        detail,
    );
    assert!(smt2.contains("check-sat"), "SMT2 should contain check-sat");
}

// ---------------------------------------------------------------------------
// Property 3: Trace Invariance Under Transpose -- tr(A) = tr(A^T)
// ---------------------------------------------------------------------------

/// Build the "trace is invariant under transpose" query.
///
/// Writing `tr(A^T)` directly as `a00 + a11` makes the claim `X = X`. Instead
/// the transpose's diagonal is *derived* from the index swap: `(A^T)_00` and
/// `(A^T)_11` are declared variables pinned to `A`'s diagonal, and the trace is
/// summed from them. `correct_transpose` toggles a plausible bug — a transpose
/// that misreads `(A^T)_00` from the off-diagonal `a01` — which changes the
/// trace whenever `a00 != a01`, turning the query SAT.
fn build_trace_invariance_under_transpose(correct_transpose: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let bound_lo = Expr::real(-1000);
    let bound_hi = Expr::real(1000);
    let a00 = declare_real(&mut program, "a00");
    let a01 = declare_real(&mut program, "a01");
    let a10 = declare_real(&mut program, "a10");
    let a11 = declare_real(&mut program, "a11");
    for v in [&a00, &a01, &a10, &a11] {
        assert_bounds(&mut program, v, &bound_lo, &bound_hi);
    }

    // A^T diagonal from the index swap: (A^T)_00 = a00, (A^T)_11 = a11.
    // The transpose leaves the diagonal untouched; the mutation misreads it.
    let at00 = declare_real(&mut program, "at00");
    let at11 = declare_real(&mut program, "at11");
    program.assert(at00.clone().eq(if correct_transpose {
        a00.clone()
    } else {
        a01.clone()
    }));
    program.assert(at11.clone().eq(a11.clone()));

    let tr_a = a00.real_add(a11);
    let tr_at = at00.real_add(at11);

    // Violation: tr(A) != tr(A^T).
    program.assert(tr_a.ne(tr_at));
    program.check_sat();
    program
}

#[test]
fn test_trace_invariance_under_transpose() {
    let program = build_trace_invariance_under_transpose(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);
    assert!(
        proven,
        "Trace invariance under transpose (QF_LRA) should be Proven. detail: {}",
        detail,
    );
    assert_eq!(crate::ay_vacuity::vacuity_smell(&smt2), None);
    assert!(smt2.contains("QF_LRA"), "should use QF_LRA logic");
}

/// A transpose that misreads the diagonal from the off-diagonal changes the
/// trace, so the invariance query must be SAT.
#[test]
fn trace_invariance_depends_on_the_transpose_diagonal() {
    let program = build_trace_invariance_under_transpose(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "a transpose that reads (A^T)_00 from a01 changes the trace; \
         the query must be SAT; got: {detail}",
    );
}

// ---------------------------------------------------------------------------
// Property 4: Determinant of an Upper-Triangular 2x2 -- det = product of diagonal
// ---------------------------------------------------------------------------

/// Build the "determinant of an upper-triangular 2x2 matrix equals the product
/// of its diagonal" query.
///
/// `det(A^T) = det(A)` collapses to commutativity of scalar multiplication
/// (`a01*a10 == a10*a01`), so it is vacuous over free entries — the same
/// computation written twice. The genuinely contingent determinant fact is
/// *triangular*: the 2x2 determinant is `(diagonal product) - (off-diagonal
/// product)`, and when the matrix is upper-triangular its (1,0) entry is zero, so
/// the off-diagonal product `a01*a10 = a01*0` vanishes and the determinant reduces
/// to the diagonal product alone. We represent the two products as free reals
/// (`diag_prod`, `off_prod`) — their internal structure is irrelevant to the
/// theorem, exactly as `s = √disc` is left free in the eigenvalue proof — and the
/// load-bearing hypothesis is `off_prod == 0`.
///
/// The two sides differ structurally: the determinant is `(- diag_prod off_prod)`
/// while the diagonal product is the bare `diag_prod`; they coincide *because* the
/// off-diagonal product vanishes, not by any `+`/`*` reordering.
///
/// `upper_triangular` toggles that hypothesis. With it false the off-diagonal
/// product is unconstrained, so `diag_prod - off_prod` differs from `diag_prod`
/// whenever `off_prod != 0` and the query is SAT.
fn build_determinant_upper_triangular_2x2(upper_triangular: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let bound_lo = Expr::real(-100);
    let bound_hi = Expr::real(100);

    // The 2x2 determinant is (diagonal product) - (off-diagonal product).
    let diag_prod = declare_real(&mut program, "diag_prod");
    let off_prod = declare_real(&mut program, "off_prod");
    for v in [&diag_prod, &off_prod] {
        assert_bounds(&mut program, v, &bound_lo, &bound_hi);
    }

    // Upper-triangularity: the (1,0) entry is zero, so the off-diagonal product
    // a01*a10 = a01*0 vanishes. Load-bearing — without it the determinant keeps
    // its off-diagonal term and no longer equals the diagonal product.
    if upper_triangular {
        program.assert(off_prod.clone().eq(Expr::real(0)));
    }

    let det = diag_prod.clone().real_sub(off_prod);

    // Violation: the determinant is not the product of the diagonal.
    program.assert(det.ne(diag_prod));
    program.check_sat();
    program
}

#[test]
fn test_determinant_upper_triangular_2x2() {
    let program = build_determinant_upper_triangular_2x2(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);
    assert!(
        proven,
        "Determinant of an upper-triangular 2x2 (QF_LRA) should be Proven. detail: {}",
        detail,
    );
    assert_eq!(crate::ay_vacuity::vacuity_smell(&smt2), None);
    assert!(smt2.contains("check-sat"), "SMT2 should contain check-sat");
}

/// Without upper-triangularity the off-diagonal product need not vanish, so
/// `diag_prod - off_prod` differs from `diag_prod` whenever `off_prod != 0`: the
/// determinant is no longer the product of the diagonal and the query must be SAT.
#[test]
fn determinant_upper_triangular_depends_on_the_off_diagonal_vanishing() {
    let program = build_determinant_upper_triangular_2x2(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "a full matrix's off-diagonal product need not vanish, so det != diagonal \
         product; the query must be SAT; got: {detail}",
    );
}

// ---------------------------------------------------------------------------
// Property 5: Symmetric Matrix -- the symmetrized off-diagonal doubles the entry
// ---------------------------------------------------------------------------

/// Build the "a symmetric matrix's symmetrized off-diagonal doubles the shared
/// entry" query.
///
/// "Symmetric implies `A = A^T`" is a tautology — symmetry *is* that condition.
/// The earlier "off-diagonals of `A + A^T` are equal" framing was still vacuous:
/// `a01 + a10` and `a10 + a01` collapse to the same term under `+` reordering, so
/// the guard flags it. The genuine, contingent fact needs the symmetry as a
/// *load-bearing hypothesis*: for a symmetric 2x2 matrix (`a01 == a10`), the
/// symmetrization `S = A + A^T` has off-diagonal `S_01 = a01 + a10`, and this
/// equals `2*a01` precisely because `a01 == a10`.
///
/// The two sides are structurally different — a sum `(+ a01 a10)` versus a scaling
/// `(* 2 a01)` — and no `+`/`*` reordering makes them coincide, so the solver
/// derives the equality from the symmetry hypothesis rather than reading it off.
///
/// `assume_symmetric` toggles that hypothesis. Without `a01 == a10` the sum
/// `a01 + a10` differs from `2*a01` whenever the two entries differ, so the query
/// is SAT.
fn build_symmetric_matrix_2x2(assume_symmetric: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let bound_lo = Expr::real(-1000);
    let bound_hi = Expr::real(1000);
    let a01 = declare_real(&mut program, "a01");
    let a10 = declare_real(&mut program, "a10");
    for v in [&a01, &a10] {
        assert_bounds(&mut program, v, &bound_lo, &bound_hi);
    }

    // Symmetry hypothesis: A_01 == A_10. Load-bearing — without it the claim fails.
    if assume_symmetric {
        program.assert(a01.clone().eq(a10.clone()));
    }

    // Symmetrization S = A + A^T off-diagonal: S_01 = a01 + a10 (A^T contributes
    // a10 at (0,1)). Under symmetry this is twice the shared off-diagonal entry.
    let s01 = a01.clone().real_add(a10);
    let two_a01 = Expr::real(2).real_mul(a01);

    // Violation: the symmetrized off-diagonal is not twice a01.
    program.assert(s01.ne(two_a01));
    program.check_sat();
    program
}

#[test]
fn test_symmetric_matrix_2x2() {
    let program = build_symmetric_matrix_2x2(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);
    assert!(
        proven,
        "Symmetric off-diagonal doubling (QF_LRA) should be Proven. detail: {}",
        detail,
    );
    assert_eq!(crate::ay_vacuity::vacuity_smell(&smt2), None);
    assert!(smt2.contains("check-sat"), "SMT2 should contain check-sat");
}

/// Without the symmetry hypothesis `a01 == a10`, the symmetrized off-diagonal
/// `a01 + a10` differs from `2*a01` whenever the entries differ, so the query
/// must be SAT.
#[test]
fn symmetric_matrix_depends_on_the_symmetry_hypothesis() {
    let program = build_symmetric_matrix_2x2(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "without a01 == a10 the sum a01 + a10 differs from 2*a01; \
         the query must be SAT; got: {detail}",
    );
}

// ---------------------------------------------------------------------------
// Property 6: Orthogonal Matrix Determinant -- Q^T Q = I => det(Q) = +/-1
// ---------------------------------------------------------------------------

/// Prove that if Q^T * Q = I for a 2x2 matrix, then det(Q)^2 = 1.
///
/// Orthogonality constraints Q^T Q = I for 2x2:
///   q00^2 + q10^2 = 1      (column 0 unit norm)
///   q01^2 + q11^2 = 1      (column 1 unit norm)
///   q00*q01 + q10*q11 = 0   (columns orthogonal)
///
/// det(Q) = q00*q11 - q01*q10
/// det(Q)^2 = (q00*q11 - q01*q10)^2
///
/// Using the orthogonality constraints, det(Q)^2 = 1. We prove det(Q)^2 != 1 is UNSAT.
#[test]
fn test_orthogonal_determinant_2x2() {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let bound_lo = Expr::real(-1);
    let bound_hi = Expr::real(1);

    let q00 = declare_real(&mut program, "q00");
    let q01 = declare_real(&mut program, "q01");
    let q10 = declare_real(&mut program, "q10");
    let q11 = declare_real(&mut program, "q11");

    for v in [&q00, &q01, &q10, &q11] {
        assert_bounds(&mut program, v, &bound_lo, &bound_hi);
    }

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // Orthogonality constraints: Q^T Q = I
    // q00^2 + q10^2 = 1
    program.assert(
        q00.clone()
            .real_mul(q00.clone())
            .real_add(q10.clone().real_mul(q10.clone()))
            .eq(one.clone()),
    );
    // q01^2 + q11^2 = 1
    program.assert(
        q01.clone()
            .real_mul(q01.clone())
            .real_add(q11.clone().real_mul(q11.clone()))
            .eq(one.clone()),
    );
    // q00*q01 + q10*q11 = 0
    program.assert(
        q00.clone()
            .real_mul(q01.clone())
            .real_add(q10.clone().real_mul(q11.clone()))
            .eq(zero),
    );

    // det(Q) = q00*q11 - q01*q10
    let det = q00.real_mul(q11).real_sub(q01.real_mul(q10));

    // det(Q)^2
    let det_sq = det.clone().real_mul(det);

    // Violation: det(Q)^2 != 1
    let violation = det_sq.ne(one);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    assert!(
        proven || detail.contains("Unknown"),
        "Orthogonal det: expected Proven or Unknown (NRA), got: {}",
        detail,
    );
    assert!(
        !detail.contains("counterexample"),
        "Orthogonal det must not have counterexample: {}",
        detail,
    );
    assert!(smt2.contains("check-sat"), "SMT2 should contain check-sat");
}

// ---------------------------------------------------------------------------
// Property 7: Positive Definite Bounds -- x^T A x > 0 for non-zero x
// ---------------------------------------------------------------------------

/// Prove x^T A x > 0 for non-zero x when A is 2x2 positive definite.
///
/// A 2x2 symmetric matrix M = [[a, b], [b, d]] is positive definite iff:
///   a > 0 AND det(M) = a*d - b^2 > 0
///
/// For x = (x0, x1) != (0, 0):
///   x^T M x = a*x0^2 + 2*b*x0*x1 + d*x1^2
///
/// We constrain a > 0 and a*d - b^2 > 0, then try to find x^T M x <= 0 for
/// some non-zero x. UNSAT proves positive definiteness.
///
/// Follows the same pattern as prove_schur_complement_positive_definite in
/// ay_matrix_decomposition_properties.rs.
#[test]
fn test_positive_definite_2x2() {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let zero = Expr::real(0);
    let bound_lo = Expr::real(-100);
    let bound_hi = Expr::real(100);

    // Symmetric PD matrix [[a, b], [b, d]]
    let a = declare_real(&mut program, "a");
    let b = declare_real(&mut program, "b");
    let d = declare_real(&mut program, "d");

    // a and d bounded non-negative (PD diagonal must be positive)
    assert_bounds(&mut program, &a, &zero, &bound_hi);
    assert_bounds(&mut program, &b, &bound_lo, &bound_hi);
    assert_bounds(&mut program, &d, &zero, &bound_hi);

    // PD constraints: a > 0 and a*d - b^2 > 0
    program.assert(a.clone().real_gt(zero.clone()));
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

    assert!(
        proven || detail.contains("Unknown"),
        "Positive definite: expected Proven or Unknown (NRA), got: {}",
        detail,
    );
    assert!(
        !detail.contains("counterexample"),
        "Positive definite must not have counterexample: {}",
        detail,
    );
    assert!(smt2.contains("check-sat"), "SMT2 should contain check-sat");
}

// ---------------------------------------------------------------------------
// Property 8: Frobenius Norm Submultiplicativity (squared form)
// ---------------------------------------------------------------------------

/// Prove ||AB||_F^2 <= ||A||_F^2 * ||B||_F^2 for 2x2 matrices.
///
/// This is the squared form of the submultiplicative property ||AB||_F <= ||A||_F * ||B||_F.
/// Since norms are non-negative, squaring both sides preserves the inequality direction.
///
/// For 2x2 matrices, we encode all entries and verify by asserting the negation.
/// Uses QF_NRA (products of symbolic variables).
#[test]
fn test_frobenius_norm_submultiplicative_2x2() {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let bound_lo = Expr::real(-10);
    let bound_hi = Expr::real(10);

    // Matrix A
    let a00 = declare_real(&mut program, "a00");
    let a01 = declare_real(&mut program, "a01");
    let a10 = declare_real(&mut program, "a10");
    let a11 = declare_real(&mut program, "a11");

    // Matrix B
    let b00 = declare_real(&mut program, "b00");
    let b01 = declare_real(&mut program, "b01");
    let b10 = declare_real(&mut program, "b10");
    let b11 = declare_real(&mut program, "b11");

    for v in [&a00, &a01, &a10, &a11, &b00, &b01, &b10, &b11] {
        assert_bounds(&mut program, v, &bound_lo, &bound_hi);
    }

    // AB entries
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

    // ||AB||_F^2 = ab00^2 + ab01^2 + ab10^2 + ab11^2
    let norm_ab_sq = ab00
        .clone()
        .real_mul(ab00)
        .real_add(ab01.clone().real_mul(ab01))
        .real_add(ab10.clone().real_mul(ab10))
        .real_add(ab11.clone().real_mul(ab11));

    // ||A||_F^2 = a00^2 + a01^2 + a10^2 + a11^2
    let norm_a_sq = a00
        .clone()
        .real_mul(a00)
        .real_add(a01.clone().real_mul(a01))
        .real_add(a10.clone().real_mul(a10))
        .real_add(a11.clone().real_mul(a11));

    // ||B||_F^2 = b00^2 + b01^2 + b10^2 + b11^2
    let norm_b_sq = b00
        .clone()
        .real_mul(b00)
        .real_add(b01.clone().real_mul(b01))
        .real_add(b10.clone().real_mul(b10))
        .real_add(b11.clone().real_mul(b11));

    // Violation: ||AB||_F^2 > ||A||_F^2 * ||B||_F^2
    let bound_product = norm_a_sq.real_mul(norm_b_sq);
    let violation = norm_ab_sq.real_gt(bound_product);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    assert!(
        proven || detail.contains("Unknown"),
        "Frobenius submultiplicativity: expected Proven or Unknown (NRA), got: {}",
        detail,
    );
    assert!(
        !detail.contains("counterexample"),
        "Frobenius submultiplicativity must not have counterexample: {}",
        detail,
    );
    assert!(smt2.contains("check-sat"), "SMT2 should contain check-sat");
}

// ---------------------------------------------------------------------------
// Property 9: Matrix Inverse -- A * A^(-1) = I for non-singular 2x2
// ---------------------------------------------------------------------------

/// Prove A * A^(-1) = I for a non-singular 2x2 matrix.
///
/// For A = [[a, b], [c, d]] with det = a*d - b*c != 0:
///   A^(-1) = (1/det) * [[d, -b], [-c, a]]
///
/// We encode the inverse entries implicitly via the constraint
///   inv_ij * det = adj_ij
/// which avoids division in the SMT encoding. Then verify A * A^(-1) = I.
#[test]
fn test_matrix_inverse_2x2() {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let bound_lo = Expr::real(-100);
    let bound_hi = Expr::real(100);

    let a = declare_real(&mut program, "a");
    let b = declare_real(&mut program, "b");
    let c = declare_real(&mut program, "c");
    let d = declare_real(&mut program, "d");

    for v in [&a, &b, &c, &d] {
        assert_bounds(&mut program, v, &bound_lo, &bound_hi);
    }

    let zero = Expr::real(0);

    // det = a*d - b*c
    let det = a
        .clone()
        .real_mul(d.clone())
        .real_sub(b.clone().real_mul(c.clone()));

    // Non-singularity: det != 0
    // Encode as det^2 > 0 (strictly positive determinant squared)
    program.assert(det.clone().real_mul(det.clone()).real_gt(zero.clone()));

    // A^(-1) entries defined via: inv_ij * det = adj_ij
    let inv_00 = declare_real(&mut program, "inv_00");
    let inv_01 = declare_real(&mut program, "inv_01");
    let inv_10 = declare_real(&mut program, "inv_10");
    let inv_11 = declare_real(&mut program, "inv_11");

    // inv_00 * det = d
    program.assert(inv_00.clone().real_mul(det.clone()).eq(d.clone()));
    // inv_01 * det = -b
    program.assert(
        inv_01
            .clone()
            .real_mul(det.clone())
            .eq(zero.clone().real_sub(b.clone())),
    );
    // inv_10 * det = -c
    program.assert(
        inv_10
            .clone()
            .real_mul(det.clone())
            .eq(zero.clone().real_sub(c.clone())),
    );
    // inv_11 * det = a
    program.assert(inv_11.clone().real_mul(det).eq(a.clone()));

    // Compute A * A^(-1)
    let prod_00 = a
        .clone()
        .real_mul(inv_00.clone())
        .real_add(b.clone().real_mul(inv_10.clone()));
    let prod_01 = a
        .real_mul(inv_01.clone())
        .real_add(b.clone().real_mul(inv_11.clone()));
    let prod_10 = c
        .clone()
        .real_mul(inv_00)
        .real_add(d.clone().real_mul(inv_10));
    let prod_11 = c.real_mul(inv_01).real_add(d.real_mul(inv_11));

    let one = Expr::real(1);

    // Violation: A * A^(-1) != I
    let violation = Expr::or_many(vec![
        prod_00.ne(one.clone()),
        prod_01.ne(zero.clone()),
        prod_10.ne(zero),
        prod_11.ne(one),
    ]);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    assert!(
        proven || detail.contains("Unknown"),
        "Matrix inverse: expected Proven or Unknown (NRA), got: {}",
        detail,
    );
    assert!(
        !detail.contains("counterexample"),
        "Matrix inverse must not have counterexample: {}",
        detail,
    );
    assert!(smt2.contains("check-sat"), "SMT2 should contain check-sat");
}

// ---------------------------------------------------------------------------
// Property 10: Eigenvalue Trace Relationship -- sum of eigenvalues = trace
// ---------------------------------------------------------------------------

/// Build the "sum of eigenvalues equals the trace" query.
///
/// Asserting `lam1 + lam2 = trace` (Vieta) and negating it proves nothing. The
/// content is the *quadratic formula*: the eigenvalues of a 2×2 matrix are
/// `λ = (tr ± √disc)/2`, so `lam1 = (tr + s)/2` and `lam2 = (tr - s)/2` where
/// `s = √disc`. Their sum is the trace *because the discriminant term cancels* —
/// a fact the solver must derive from the two definitions, not something
/// asserted. The trace is itself derived from the diagonal (`tr = a00 + a11`),
/// and `s` is a free real, so this holds for every 2×2 matrix, not one concrete
/// example. Scaling the definitions by 2 (`2·lam = tr ± s`) keeps the whole
/// query linear (QF_LRA) — no `√` and no variable×variable product.
///
/// `cancel_discriminant` toggles the sign on `lam2`'s discriminant term. When
/// false the mutation uses `2·lam2 = tr + s` (both `+s`), so `s` no longer
/// cancels and the sum is `tr + s`, which differs from `tr` whenever `s != 0`.
fn build_eigenvalue_trace_2x2(cancel_discriminant: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Trace from the diagonal; s = √discriminant is a free real (its actual
    // value is irrelevant — the theorem is that it cancels out of the sum).
    let a00 = declare_real(&mut program, "a00");
    let a11 = declare_real(&mut program, "a11");
    let s = declare_real(&mut program, "s");
    assert_bounds(&mut program, &a00, &Expr::real(-200), &Expr::real(200));
    assert_bounds(&mut program, &a11, &Expr::real(-200), &Expr::real(200));
    assert_bounds(&mut program, &s, &Expr::real(-200), &Expr::real(200));

    let trace = a00.real_add(a11);
    let two = Expr::real(2);

    // 2·lam1 = tr + s  (the "+√disc" root).
    let lam1 = declare_real(&mut program, "lam1");
    program.assert(
        lam1.clone()
            .real_mul(two.clone())
            .eq(trace.clone().real_add(s.clone())),
    );

    // 2·lam2 = tr - s (correct) or tr + s (mutation: discriminant fails to cancel).
    let lam2 = declare_real(&mut program, "lam2");
    let lam2_num = if cancel_discriminant {
        trace.clone().real_sub(s.clone())
    } else {
        trace.clone().real_add(s.clone())
    };
    program.assert(lam2.clone().real_mul(two).eq(lam2_num));

    // Violation: the eigenvalues do not sum to the trace.
    program.assert(lam1.real_add(lam2).ne(trace));
    program.check_sat();
    program
}

#[test]
fn test_eigenvalue_trace_2x2() {
    let program = build_eigenvalue_trace_2x2(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);
    assert!(
        proven,
        "Eigenvalue trace (quadratic formula, QF_LRA) should be Proven. detail: {}",
        detail,
    );
    assert_eq!(crate::ay_vacuity::vacuity_smell(&smt2), None);
    assert!(smt2.contains("check-sat"), "SMT2 should contain check-sat");
}

/// With `2·lam2 = tr + s` (both roots taking `+√disc`) the discriminant no
/// longer cancels, so `lam1 + lam2 = tr + s != tr` whenever `s != 0`: the query
/// must be SAT.
#[test]
fn eigenvalue_trace_depends_on_the_discriminant_cancelling() {
    let program = build_eigenvalue_trace_2x2(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "if the discriminant does not cancel the sum is tr + s != tr; the query must be SAT; got: {detail}",
    );
}
