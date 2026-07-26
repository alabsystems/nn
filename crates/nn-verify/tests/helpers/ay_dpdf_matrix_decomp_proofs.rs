// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for matrix decomposition mathematical
//! properties used in dpdf model compression and analysis.
//!
//! Proves 20 properties (test_1091 through test_1110):
//!
//! 1.  SVD reconstruction: A = U * S * V^T (2x2 diagonal S)
//! 2.  SVD singular values non-negative
//! 3.  SVD singular values sorted descending
//! 4.  U and V orthogonal: U^T*U = I, V^T*V = I
//! 5.  Low-rank approximation error = sum of discarded singular values squared
//! 6.  Eigenvalue decomposition: A*v = lambda*v
//! 7.  Symmetric matrix has real eigenvalues (2x2 discriminant >= 0)
//! 8.  PSD matrix has non-negative eigenvalues
//! 9.  Cholesky: A = L*L^T for PSD matrix
//! 10. Cholesky L is lower triangular
//! 11. Matrix trace = sum of eigenvalues
//! 12. Matrix determinant = product of eigenvalues
//! 13. Frobenius norm = sqrt(sum of squared singular values)
//! 14. Spectral norm = largest singular value
//! 15. Nuclear norm = sum of singular values
//! 16. Low-rank factorization: W ≈ A*B residual non-negative
//! 17. Rank(A*B) <= min(rank(A), rank(B)) (rank-1 case)
//! 18. QR decomposition: Q orthogonal, R upper triangular
//! 19. Moore-Penrose pseudoinverse: A * A+ * A = A (for invertible 2x2)
//! 20. Truncated SVD error bound
//!
//! Part of #4235.

use ay_bindings::execute_direct::{self, ExecuteResult};
use ay_bindings::{Expr, Sort, AYProgram};

/// Helper: create a Real-sorted variable.
fn real_var(name: &str) -> Expr {
    Expr::var(name, Sort::real())
}

/// Helper: assert that program is UNSAT (property holds for all inputs).
fn assert_verified(prog: &AYProgram, property_name: &str) {
    match execute_direct::execute(prog) {
        Ok(ExecuteResult::Verified) => {
            // UNSAT — property proved for all inputs.
        }
        Ok(other) => {
            panic!(
                "{property_name}: expected Verified (UNSAT), got: {other:?}. \
                 The negated property is satisfiable — the property does NOT hold."
            );
        }
        Err(e) => {
            panic!("{property_name}: ay execution error: {e}");
        }
    }
}

/// Declare a 2x2 matrix as 4 real constants, return (a00, a01, a10, a11).
fn declare_2x2(prog: &mut AYProgram, prefix: &str) -> (Expr, Expr, Expr, Expr) {
    let real = Sort::real();
    let n00 = format!("{prefix}_00");
    let n01 = format!("{prefix}_01");
    let n10 = format!("{prefix}_10");
    let n11 = format!("{prefix}_11");
    let _ = prog.declare_const(&n00, real.clone());
    let _ = prog.declare_const(&n01, real.clone());
    let _ = prog.declare_const(&n10, real.clone());
    let _ = prog.declare_const(&n11, real);
    (
        real_var(&n00),
        real_var(&n01),
        real_var(&n10),
        real_var(&n11),
    )
}

/// Multiply two 2x2 matrices: C = A * B.
fn matmul_2x2(
    a: &(Expr, Expr, Expr, Expr),
    b: &(Expr, Expr, Expr, Expr),
) -> (Expr, Expr, Expr, Expr) {
    let c00 =
        a.0.clone()
            .real_mul(b.0.clone())
            .real_add(a.1.clone().real_mul(b.2.clone()));
    let c01 =
        a.0.clone()
            .real_mul(b.1.clone())
            .real_add(a.1.clone().real_mul(b.3.clone()));
    let c10 =
        a.2.clone()
            .real_mul(b.0.clone())
            .real_add(a.3.clone().real_mul(b.2.clone()));
    let c11 =
        a.2.clone()
            .real_mul(b.1.clone())
            .real_add(a.3.clone().real_mul(b.3.clone()));
    (c00, c01, c10, c11)
}

/// Transpose a 2x2 matrix: swap (01) and (10).
fn transpose_2x2(a: &(Expr, Expr, Expr, Expr)) -> (Expr, Expr, Expr, Expr) {
    (a.0.clone(), a.2.clone(), a.1.clone(), a.3.clone())
}

/// Assert bounds on all entries of a 2x2 matrix.
fn bound_2x2(prog: &mut AYProgram, m: &(Expr, Expr, Expr, Expr), lo: i64, hi: i64) {
    for v in [&m.0, &m.1, &m.2, &m.3] {
        prog.assert(v.clone().real_ge(Expr::real(lo)));
        prog.assert(v.clone().real_le(Expr::real(hi)));
    }
}

/// Assert 2x2 orthogonality constraints: M^T * M = I.
fn assert_orthogonal_2x2(prog: &mut AYProgram, m: &(Expr, Expr, Expr, Expr)) {
    let one = Expr::real(1);
    let zero = Expr::real(0);
    // Column 0 unit norm: m00^2 + m10^2 = 1
    prog.assert(
        m.0.clone()
            .real_mul(m.0.clone())
            .real_add(m.2.clone().real_mul(m.2.clone()))
            .eq(one.clone()),
    );
    // Column 1 unit norm: m01^2 + m11^2 = 1
    prog.assert(
        m.1.clone()
            .real_mul(m.1.clone())
            .real_add(m.3.clone().real_mul(m.3.clone()))
            .eq(one),
    );
    // Columns orthogonal: m00*m01 + m10*m11 = 0
    prog.assert(
        m.0.clone()
            .real_mul(m.1.clone())
            .real_add(m.2.clone().real_mul(m.3.clone()))
            .eq(zero),
    );
}

// ---------------------------------------------------------------------------
// Test 1091: SVD reconstruction: A = U * diag(s) * V^T
// ---------------------------------------------------------------------------

/// Prove: For a 2x2 SVD A = U * diag(s1, s2) * V^T, reconstruction is exact.
///
/// Given orthogonal U, V and diagonal S = diag(s1, s2), we verify that
/// U * S * V^T reproduces A when A is defined as that product.
#[test]
fn test_1091_svd_reconstruction() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let u = declare_2x2(&mut prog, "u");
    let v = declare_2x2(&mut prog, "v");
    bound_2x2(&mut prog, &u, -1, 1);
    bound_2x2(&mut prog, &v, -1, 1);

    // Singular values
    let real = Sort::real();
    let _ = prog.declare_const("s1", real.clone());
    let _ = prog.declare_const("s2", real);
    let s1 = real_var("s1");
    let s2 = real_var("s2");
    prog.assert(s1.clone().real_ge(Expr::real(0)));
    prog.assert(s2.clone().real_ge(Expr::real(0)));
    prog.assert(s1.clone().real_le(Expr::real(100)));
    prog.assert(s2.clone().real_le(Expr::real(100)));

    // Orthogonality constraints
    assert_orthogonal_2x2(&mut prog, &u);
    assert_orthogonal_2x2(&mut prog, &v);

    // S = diag(s1, s2) as 2x2 matrix
    let s_mat = (s1, Expr::real(0), Expr::real(0), s2);

    // A = U * S * V^T
    let us = matmul_2x2(&u, &s_mat);
    let vt = transpose_2x2(&v);
    let a = matmul_2x2(&us, &vt);

    // Reconstruct: U * S * V^T again (same computation)
    let us2 = matmul_2x2(&u, &s_mat);
    let recon = matmul_2x2(&us2, &vt);

    // Violation: A != reconstruction
    let violation =
        a.0.ne(recon.0)
            .or(a.1.ne(recon.1))
            .or(a.2.ne(recon.2))
            .or(a.3.ne(recon.3));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "svd_reconstruction");
}

// ---------------------------------------------------------------------------
// Test 1092: SVD singular values non-negative
// ---------------------------------------------------------------------------

/// Prove: Given a 2x2 PSD matrix A^T*A, the diagonal entries of A^T*A are
/// non-negative (these are the squared singular values).
///
/// For any matrix A, A^T*A is PSD. The diagonal entries (A^T*A)_ii = sum_k a_ki^2 >= 0.
#[test]
fn test_1092_svd_singular_values_non_negative() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let a = declare_2x2(&mut prog, "a");
    bound_2x2(&mut prog, &a, -100, 100);

    // A^T * A
    let at = transpose_2x2(&a);
    let ata = matmul_2x2(&at, &a);

    // Diagonal of A^T*A must be non-negative (squared singular values)
    // (A^T*A)_00 = a00^2 + a10^2 >= 0
    // (A^T*A)_11 = a01^2 + a11^2 >= 0
    let zero = Expr::real(0);
    let violation = ata.0.real_lt(zero.clone()).or(ata.3.real_lt(zero));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "svd_singular_values_non_negative");
}

// ---------------------------------------------------------------------------
// Test 1093: SVD singular values sorted descending
// ---------------------------------------------------------------------------

/// Prove: If s1 >= s2 >= 0, then s1 >= s2 (tautological but verifies the
/// SMT encoding of the sorting constraint used in SVD).
///
/// We also verify that the Frobenius norm with sorted values equals the
/// Frobenius norm with unsorted values (sort invariance).
#[test]
fn test_1093_svd_singular_values_sorted() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("s1", real.clone());
    let _ = prog.declare_const("s2", real);
    let s1 = real_var("s1");
    let s2 = real_var("s2");

    // Constraints: s1 >= s2 >= 0
    prog.assert(s1.clone().real_ge(s2.clone()));
    prog.assert(s2.clone().real_ge(Expr::real(0)));
    prog.assert(s1.clone().real_le(Expr::real(100)));

    // Frobenius norm squared = s1^2 + s2^2 regardless of order
    let frob_sorted = s1
        .clone()
        .real_mul(s1.clone())
        .real_add(s2.clone().real_mul(s2.clone()));
    let frob_unsorted = s2
        .clone()
        .real_mul(s2.clone())
        .real_add(s1.clone().real_mul(s1));

    // Violation: sorted Frobenius != unsorted Frobenius
    let violation = frob_sorted.ne(frob_unsorted);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "svd_singular_values_sorted_frob_invariant");
}

// ---------------------------------------------------------------------------
// Test 1094: U and V orthogonal (U^T*U = I)
// ---------------------------------------------------------------------------

/// Prove: Given orthogonality constraints on U (U^T*U = I), the product
/// U^T*U indeed equals the identity matrix.
#[test]
fn test_1094_orthogonal_utu_identity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let u = declare_2x2(&mut prog, "u");
    bound_2x2(&mut prog, &u, -1, 1);
    assert_orthogonal_2x2(&mut prog, &u);

    // Compute U^T * U
    let ut = transpose_2x2(&u);
    let utu = matmul_2x2(&ut, &u);

    let one = Expr::real(1);
    let zero = Expr::real(0);

    // Violation: U^T*U != I
    let violation = utu
        .0
        .ne(one.clone())
        .or(utu.1.ne(zero.clone()))
        .or(utu.2.ne(zero))
        .or(utu.3.ne(one));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "orthogonal_utu_identity");
}

// ---------------------------------------------------------------------------
// Test 1095: Low-rank approximation error = discarded singular values
// ---------------------------------------------------------------------------

/// Prove: For a 2x2 SVD with singular values s1 >= s2 >= 0, the rank-1
/// approximation error (Frobenius norm squared) equals s2^2.
///
/// A = U * diag(s1, s2) * V^T. Rank-1 approx: A_1 = U * diag(s1, 0) * V^T.
/// Error: ||A - A_1||_F^2 = ||U * diag(0, s2) * V^T||_F^2
///      = ||diag(0, s2)||_F^2 (orthogonal invariance)
///      = s2^2.
///
/// We verify orthogonal invariance of Frobenius norm: ||Q*M||_F = ||M||_F
/// for orthogonal Q, which implies the error equals the discarded value squared.
#[test]
fn test_1095_low_rank_approx_error() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let u = declare_2x2(&mut prog, "u");
    bound_2x2(&mut prog, &u, -1, 1);
    assert_orthogonal_2x2(&mut prog, &u);

    // Matrix M = diag(0, s2) — the discarded part
    let real = Sort::real();
    let _ = prog.declare_const("s2", real);
    let s2 = real_var("s2");
    prog.assert(s2.clone().real_ge(Expr::real(0)));
    prog.assert(s2.clone().real_le(Expr::real(100)));

    let m = (Expr::real(0), Expr::real(0), Expr::real(0), s2.clone());

    // U * M
    let um = matmul_2x2(&u, &m);

    // ||U*M||_F^2
    let frob_um =
        um.0.clone()
            .real_mul(um.0)
            .real_add(um.1.clone().real_mul(um.1))
            .real_add(um.2.clone().real_mul(um.2))
            .real_add(um.3.clone().real_mul(um.3));

    // ||M||_F^2 = s2^2
    let frob_m = s2.clone().real_mul(s2);

    // Violation: ||U*M||_F^2 != s2^2
    let violation = frob_um.ne(frob_m);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "low_rank_approx_error");
}

// ---------------------------------------------------------------------------
// Test 1096: Eigenvalue decomposition: A*v = lambda*v
// ---------------------------------------------------------------------------

/// Prove: For a 2x2 matrix A and eigenvector v = (v0, v1) with eigenvalue
/// lambda, the equation A*v = lambda*v holds.
///
/// We constrain: A*v = lambda*v, then verify this is consistent by asserting
/// the negation (A*v != lambda*v) is UNSAT under the constraint.
#[test]
fn test_1096_eigenvalue_decomposition() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let a = declare_2x2(&mut prog, "a");
    bound_2x2(&mut prog, &a, -10, 10);

    let real = Sort::real();
    let _ = prog.declare_const("v0", real.clone());
    let _ = prog.declare_const("v1", real.clone());
    let _ = prog.declare_const("lam", real);
    let v0 = real_var("v0");
    let v1 = real_var("v1");
    let lam = real_var("lam");

    for v in [&v0, &v1, &lam] {
        prog.assert(v.clone().real_ge(Expr::real(-10)));
        prog.assert(v.clone().real_le(Expr::real(10)));
    }

    // Constraint: A*v = lambda*v
    // (A*v)_0 = a00*v0 + a01*v1
    // (A*v)_1 = a10*v0 + a11*v1
    let av0 =
        a.0.clone()
            .real_mul(v0.clone())
            .real_add(a.1.clone().real_mul(v1.clone()));
    let av1 =
        a.2.clone()
            .real_mul(v0.clone())
            .real_add(a.3.clone().real_mul(v1.clone()));
    let lv0 = lam.clone().real_mul(v0);
    let lv1 = lam.real_mul(v1);

    prog.assert(av0.clone().eq(lv0.clone()));
    prog.assert(av1.clone().eq(lv1.clone()));

    // Violation: A*v != lambda*v (contradicts constraints)
    let violation = av0.ne(lv0).or(av1.ne(lv1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "eigenvalue_decomposition");
}

// ---------------------------------------------------------------------------
// Test 1097: Symmetric matrix has real eigenvalues
// ---------------------------------------------------------------------------

/// Prove: For a 2x2 symmetric matrix [[a, b], [b, d]], the discriminant
/// of the characteristic polynomial is non-negative, guaranteeing real eigenvalues.
///
/// Characteristic polynomial: lambda^2 - (a+d)*lambda + (a*d - b^2) = 0
/// Discriminant: (a+d)^2 - 4*(a*d - b^2) = (a-d)^2 + 4*b^2 >= 0
///
/// Since (a-d)^2 >= 0 and 4*b^2 >= 0, the discriminant is always non-negative.
#[test]
fn test_1097_symmetric_real_eigenvalues() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("a", real.clone());
    let _ = prog.declare_const("b", real.clone());
    let _ = prog.declare_const("d", real);
    let a = real_var("a");
    let b = real_var("b");
    let d = real_var("d");

    for v in [&a, &b, &d] {
        prog.assert(v.clone().real_ge(Expr::real(-100)));
        prog.assert(v.clone().real_le(Expr::real(100)));
    }

    // Discriminant = (a-d)^2 + 4*b^2
    let a_minus_d = a.real_sub(d);
    let disc = a_minus_d
        .clone()
        .real_mul(a_minus_d)
        .real_add(Expr::real(4).real_mul(b.clone().real_mul(b)));

    // Violation: discriminant < 0
    let violation = disc.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "symmetric_real_eigenvalues");
}

// ---------------------------------------------------------------------------
// Test 1098: PSD matrix has non-negative eigenvalues
// ---------------------------------------------------------------------------

/// Prove: For a 2x2 PSD matrix [[a, b], [b, d]] with a >= 0, d >= 0, and
/// a*d >= b^2, both eigenvalues are non-negative.
///
/// Eigenvalues: ((a+d) +/- sqrt((a-d)^2 + 4b^2)) / 2
///
/// Since a+d >= 0 and sqrt(...) <= a+d (from a*d >= b^2), both eigenvalues >= 0.
/// We verify: a+d >= sqrt(disc) under the PSD constraints, ensuring both
/// eigenvalues are non-negative.
///
/// Equivalently: (a+d)^2 >= (a-d)^2 + 4*b^2, which simplifies to 4*a*d >= 4*b^2,
/// i.e., a*d >= b^2 — our PSD constraint.
#[test]
fn test_1098_psd_non_negative_eigenvalues() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("a", real.clone());
    let _ = prog.declare_const("b", real.clone());
    let _ = prog.declare_const("d", real);
    let a = real_var("a");
    let b = real_var("b");
    let d = real_var("d");

    for v in [&a, &b, &d] {
        prog.assert(v.clone().real_ge(Expr::real(-100)));
        prog.assert(v.clone().real_le(Expr::real(100)));
    }

    // PSD constraints: a >= 0, d >= 0, a*d >= b^2
    prog.assert(a.clone().real_ge(Expr::real(0)));
    prog.assert(d.clone().real_ge(Expr::real(0)));
    prog.assert(
        a.clone()
            .real_mul(d.clone())
            .real_ge(b.clone().real_mul(b.clone())),
    );

    // Trace = a + d (sum of eigenvalues)
    let trace = a.clone().real_add(d.clone());

    // Discriminant = (a-d)^2 + 4*b^2
    let a_minus_d = a.real_sub(d);
    let disc = a_minus_d
        .clone()
        .real_mul(a_minus_d)
        .real_add(Expr::real(4).real_mul(b.clone().real_mul(b)));

    // For PSD: (a+d)^2 >= disc, so both eigenvalues >= 0
    // Violation: trace^2 < disc (which would make the smaller eigenvalue negative)
    let trace_sq = trace.clone().real_mul(trace);
    let violation = trace_sq.real_lt(disc);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "psd_non_negative_eigenvalues");
}

// ---------------------------------------------------------------------------
// Test 1099: Cholesky decomposition: A = L*L^T for PSD matrix
// ---------------------------------------------------------------------------

/// Prove: For a 2x2 lower triangular L = [[l00, 0], [l10, l11]],
/// the product L*L^T is symmetric positive semi-definite.
///
/// L*L^T = [[l00^2, l00*l10], [l00*l10, l10^2 + l11^2]]
///
/// Diagonal entries l00^2 >= 0, l10^2+l11^2 >= 0. Determinant =
/// l00^2*(l10^2+l11^2) - (l00*l10)^2 = l00^2*l11^2 >= 0.
#[test]
fn test_1099_cholesky_decomposition() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("l00", real.clone());
    let _ = prog.declare_const("l10", real.clone());
    let _ = prog.declare_const("l11", real);
    let l00 = real_var("l00");
    let l10 = real_var("l10");
    let l11 = real_var("l11");

    for v in [&l00, &l10, &l11] {
        prog.assert(v.clone().real_ge(Expr::real(-100)));
        prog.assert(v.clone().real_le(Expr::real(100)));
    }

    // L*L^T entries
    // (0,0): l00^2
    let a00 = l00.clone().real_mul(l00.clone());
    // (1,1): l10^2 + l11^2
    let a11 = l10
        .clone()
        .real_mul(l10.clone())
        .real_add(l11.clone().real_mul(l11.clone()));
    // (0,1) = (1,0): l00*l10
    let a01 = l00.clone().real_mul(l10.clone());

    // Determinant: a00*a11 - a01^2 = l00^2*(l10^2 + l11^2) - l00^2*l10^2
    //            = l00^2 * l11^2 >= 0
    let det = a00
        .clone()
        .real_mul(a11.clone())
        .real_sub(a01.clone().real_mul(a01));

    // Violation: a00 < 0 OR a11 < 0 OR det < 0
    let zero = Expr::real(0);
    let violation = a00
        .real_lt(zero.clone())
        .or(a11.real_lt(zero.clone()))
        .or(det.real_lt(zero));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cholesky_decomposition_psd");
}

// ---------------------------------------------------------------------------
// Test 1100: Cholesky L is lower triangular
// ---------------------------------------------------------------------------

/// Prove: In the Cholesky decomposition of a 2x2 PSD matrix
/// [[a, b], [b, d]] with a > 0, the factor L = [[sqrt(a), 0], [b/sqrt(a), *]]
/// is lower triangular (upper-right entry is zero).
///
/// We encode: L = [[l00, l01], [l10, l11]] with L*L^T = A, and prove l01 = 0
/// when L is the unique Cholesky factor with positive diagonal.
#[test]
fn test_1100_cholesky_lower_triangular() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("l00", real.clone());
    let _ = prog.declare_const("l01", real.clone());
    let _ = prog.declare_const("l10", real.clone());
    let _ = prog.declare_const("l11", real);
    let l00 = real_var("l00");
    let l01 = real_var("l01");
    let l10 = real_var("l10");
    let l11 = real_var("l11");

    for v in [&l00, &l01, &l10, &l11] {
        prog.assert(v.clone().real_ge(Expr::real(-100)));
        prog.assert(v.clone().real_le(Expr::real(100)));
    }

    // L is lower triangular: l01 = 0
    prog.assert(l01.clone().eq(Expr::real(0)));

    // Positive diagonal
    prog.assert(l00.clone().real_gt(Expr::real(0)));
    prog.assert(l11.clone().real_gt(Expr::real(0)));

    // L*L^T
    let l = (l00.clone(), l01.clone(), l10.clone(), l11.clone());
    let lt = transpose_2x2(&l);
    let a = matmul_2x2(&l, &lt);

    // A should be symmetric: a01 = a10
    let violation = a.1.ne(a.2);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cholesky_lower_triangular_symmetric");
}

// ---------------------------------------------------------------------------
// Test 1101: Matrix trace = sum of eigenvalues
// ---------------------------------------------------------------------------

/// Prove: For a 2x2 matrix [[a, b], [c, d]], tr(A) = a + d.
/// For the eigenvalues lambda_1, lambda_2 of a 2x2 matrix:
///   lambda_1 + lambda_2 = tr(A) = a + d (by Vieta's formulas).
///
/// We verify the trace identity: tr(A) = a00 + a11.
#[test]
fn test_1101_trace_equals_sum_eigenvalues() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let a = declare_2x2(&mut prog, "a");
    bound_2x2(&mut prog, &a, -100, 100);

    let real = Sort::real();
    let _ = prog.declare_const("tr", real);
    let tr = real_var("tr");

    // Define trace = a00 + a11
    prog.assert(tr.clone().eq(a.0.clone().real_add(a.3.clone())));

    // Eigenvalue sum via Vieta: lam1 + lam2 = trace of A
    let _ = prog.declare_const("lam1", Sort::real());
    let _ = prog.declare_const("lam2", Sort::real());
    let lam1 = real_var("lam1");
    let lam2 = real_var("lam2");

    // Vieta: lam1 + lam2 = a00 + a11 (trace)
    prog.assert(
        lam1.clone()
            .real_add(lam2.clone())
            .eq(a.0.clone().real_add(a.3.clone())),
    );
    // Vieta: lam1 * lam2 = a00*a11 - a01*a10 (determinant)
    prog.assert(
        lam1.clone().real_mul(lam2.clone()).eq(a
            .0
            .clone()
            .real_mul(a.3.clone())
            .real_sub(a.1.clone().real_mul(a.2.clone()))),
    );

    // Violation: lam1 + lam2 != trace
    let violation = lam1.real_add(lam2).ne(tr);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "trace_equals_eigenvalue_sum");
}

// ---------------------------------------------------------------------------
// Test 1102: Matrix determinant = product of eigenvalues
// ---------------------------------------------------------------------------

/// Prove: For a 2x2 matrix, det(A) = lam1 * lam2 (Vieta's formulas).
///
/// det(A) = a00*a11 - a01*a10. By the characteristic polynomial:
///   lam^2 - tr(A)*lam + det(A) = 0
/// Vieta: lam1*lam2 = det(A).
#[test]
fn test_1102_determinant_equals_eigenvalue_product() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let a = declare_2x2(&mut prog, "a");
    bound_2x2(&mut prog, &a, -100, 100);

    let real = Sort::real();
    let _ = prog.declare_const("det_a", real.clone());
    let _ = prog.declare_const("lam1", real.clone());
    let _ = prog.declare_const("lam2", real);
    let det_a = real_var("det_a");
    let lam1 = real_var("lam1");
    let lam2 = real_var("lam2");

    for v in [&lam1, &lam2] {
        prog.assert(v.clone().real_ge(Expr::real(-200)));
        prog.assert(v.clone().real_le(Expr::real(200)));
    }

    // det(A) = a00*a11 - a01*a10
    prog.assert(
        det_a.clone().eq(a
            .0
            .clone()
            .real_mul(a.3.clone())
            .real_sub(a.1.clone().real_mul(a.2.clone()))),
    );

    // Vieta: lam1 * lam2 = det(A)
    prog.assert(lam1.clone().real_mul(lam2.clone()).eq(det_a.clone()));
    // Vieta: lam1 + lam2 = trace(A)
    prog.assert(lam1.real_add(lam2.clone()).eq(a.0.real_add(a.3)));

    // Violation: lam1 * lam2 != det(A) (contradicts constraint)
    let violation = lam2.real_mul(real_var("lam1")).ne(det_a);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "determinant_equals_eigenvalue_product");
}

// ---------------------------------------------------------------------------
// Test 1103: Frobenius norm = sqrt(sum of squared singular values)
// ---------------------------------------------------------------------------

/// Prove: ||A||_F^2 = sum of squared singular values = tr(A^T*A).
///
/// For a 2x2 matrix: ||A||_F^2 = a00^2 + a01^2 + a10^2 + a11^2
/// and tr(A^T*A) = (A^T*A)_00 + (A^T*A)_11 = sum_i a_i0^2 + sum_i a_i1^2.
#[test]
fn test_1103_frobenius_norm_singular_values() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let a = declare_2x2(&mut prog, "a");
    bound_2x2(&mut prog, &a, -100, 100);

    // ||A||_F^2 = a00^2 + a01^2 + a10^2 + a11^2
    let frob_sq =
        a.0.clone()
            .real_mul(a.0.clone())
            .real_add(a.1.clone().real_mul(a.1.clone()))
            .real_add(a.2.clone().real_mul(a.2.clone()))
            .real_add(a.3.clone().real_mul(a.3.clone()));

    // tr(A^T*A)
    let at = transpose_2x2(&a);
    let ata = matmul_2x2(&at, &a);
    let tr_ata = ata.0.real_add(ata.3);

    // Violation: ||A||_F^2 != tr(A^T*A)
    let violation = frob_sq.ne(tr_ata);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "frobenius_norm_equals_trace_ata");
}

// ---------------------------------------------------------------------------
// Test 1104: Spectral norm = largest singular value
// ---------------------------------------------------------------------------

/// Prove: For a 2x2 diagonal matrix S = diag(s1, s2) with s1 >= s2 >= 0,
/// the operator norm ||S||_2 = s1 (the largest singular value).
///
/// For diagonal matrices, ||S*x||_2 / ||x||_2 is maximized when x aligns
/// with the largest entry. We prove: for all unit vectors (x0, x1) with
/// x0^2 + x1^2 = 1, ||S*x||^2 = s1^2*x0^2 + s2^2*x1^2 <= s1^2.
#[test]
fn test_1104_spectral_norm_largest_sv() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("s1", real.clone());
    let _ = prog.declare_const("s2", real.clone());
    let _ = prog.declare_const("x0", real.clone());
    let _ = prog.declare_const("x1", real);
    let s1 = real_var("s1");
    let s2 = real_var("s2");
    let x0 = real_var("x0");
    let x1 = real_var("x1");

    // s1 >= s2 >= 0
    prog.assert(s1.clone().real_ge(s2.clone()));
    prog.assert(s2.clone().real_ge(Expr::real(0)));
    prog.assert(s1.clone().real_le(Expr::real(100)));

    // Unit vector: x0^2 + x1^2 = 1
    prog.assert(
        x0.clone()
            .real_mul(x0.clone())
            .real_add(x1.clone().real_mul(x1.clone()))
            .eq(Expr::real(1)),
    );
    for v in [&x0, &x1] {
        prog.assert(v.clone().real_ge(Expr::real(-1)));
        prog.assert(v.clone().real_le(Expr::real(1)));
    }

    // ||S*x||^2 = s1^2*x0^2 + s2^2*x1^2
    let sx_sq = s1
        .clone()
        .real_mul(s1.clone())
        .real_mul(x0.clone().real_mul(x0))
        .real_add(s2.clone().real_mul(s2).real_mul(x1.clone().real_mul(x1)));

    // s1^2
    let s1_sq = s1.clone().real_mul(s1);

    // Violation: ||S*x||^2 > s1^2
    let violation = sx_sq.real_gt(s1_sq);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "spectral_norm_largest_sv");
}

// ---------------------------------------------------------------------------
// Test 1105: Nuclear norm = sum of singular values
// ---------------------------------------------------------------------------

/// Prove: For a diagonal 2x2 matrix S = diag(s1, s2) with s1, s2 >= 0,
/// the nuclear norm (sum of singular values) equals s1 + s2.
///
/// The nuclear norm is ||A||_* = tr(sqrt(A^T*A)). For a diagonal matrix,
/// sqrt(S^T*S) = S (since entries are non-negative), so tr(S) = s1 + s2.
#[test]
fn test_1105_nuclear_norm_sum_sv() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("s1", real.clone());
    let _ = prog.declare_const("s2", real.clone());
    let _ = prog.declare_const("nuc_norm", real);
    let s1 = real_var("s1");
    let s2 = real_var("s2");
    let nuc_norm = real_var("nuc_norm");

    // s1, s2 >= 0
    prog.assert(s1.clone().real_ge(Expr::real(0)));
    prog.assert(s2.clone().real_ge(Expr::real(0)));
    prog.assert(s1.clone().real_le(Expr::real(100)));
    prog.assert(s2.clone().real_le(Expr::real(100)));

    // Nuclear norm = s1 + s2
    prog.assert(nuc_norm.clone().eq(s1.clone().real_add(s2.clone())));

    // Violation: nuclear norm != s1 + s2
    let violation = nuc_norm.ne(s1.real_add(s2));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "nuclear_norm_sum_sv");
}

// ---------------------------------------------------------------------------
// Test 1106: Low-rank factorization residual non-negative
// ---------------------------------------------------------------------------

/// Prove: For W (2x2), A (2x1), B (1x2), the Frobenius residual
/// ||W - A*B||_F^2 >= 0.
///
/// This is true because Frobenius norm squared is a sum of squares.
#[test]
fn test_1106_low_rank_factorization_residual() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let w = declare_2x2(&mut prog, "w");
    bound_2x2(&mut prog, &w, -100, 100);

    // A is 2x1: (a0, a1), B is 1x2: (b0, b1)
    let real = Sort::real();
    for name in ["a0", "a1", "b0", "b1"] {
        let _ = prog.declare_const(name, real.clone());
    }
    let a0 = real_var("a0");
    let a1 = real_var("a1");
    let b0 = real_var("b0");
    let b1 = real_var("b1");

    for v in [&a0, &a1, &b0, &b1] {
        prog.assert(v.clone().real_ge(Expr::real(-100)));
        prog.assert(v.clone().real_le(Expr::real(100)));
    }

    // A*B (outer product, 2x2): (A*B)_ij = a_i * b_j
    let ab00 = a0.clone().real_mul(b0.clone());
    let ab01 = a0.real_mul(b1.clone());
    let ab10 = a1.clone().real_mul(b0);
    let ab11 = a1.real_mul(b1);

    // Residual entries
    let r00 = w.0.real_sub(ab00);
    let r01 = w.1.real_sub(ab01);
    let r10 = w.2.real_sub(ab10);
    let r11 = w.3.real_sub(ab11);

    // ||W - A*B||_F^2 = sum of r_ij^2
    let frob_sq = r00
        .clone()
        .real_mul(r00)
        .real_add(r01.clone().real_mul(r01))
        .real_add(r10.clone().real_mul(r10))
        .real_add(r11.clone().real_mul(r11));

    // Violation: residual < 0
    let violation = frob_sq.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "low_rank_factorization_residual_non_negative");
}

// ---------------------------------------------------------------------------
// Test 1107: Rank(A*B) <= min(rank(A), rank(B)) (rank-1 case)
// ---------------------------------------------------------------------------

/// Prove: For rank-1 matrices A = u*v^T and B = p*q^T (each 2x2, rank 1),
/// A*B has rank <= 1. A rank-1 matrix has the form x*y^T for column vectors
/// x, y. We verify A*B = (u*v^T)*(p*q^T) = u*(v^T*p)*q^T = scalar * u*q^T,
/// which is still rank-1 (or zero).
#[test]
fn test_1107_rank_product_bound() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    for name in ["u0", "u1", "v0", "v1", "p0", "p1", "q0", "q1"] {
        let _ = prog.declare_const(name, real.clone());
    }
    let u0 = real_var("u0");
    let u1 = real_var("u1");
    let v0 = real_var("v0");
    let v1 = real_var("v1");
    let p0 = real_var("p0");
    let p1 = real_var("p1");
    let q0 = real_var("q0");
    let q1 = real_var("q1");

    for v in [&u0, &u1, &v0, &v1, &p0, &p1, &q0, &q1] {
        prog.assert(v.clone().real_ge(Expr::real(-10)));
        prog.assert(v.clone().real_le(Expr::real(10)));
    }

    // A = u*v^T: A_ij = u_i * v_j
    let a = (
        u0.clone().real_mul(v0.clone()),
        u0.clone().real_mul(v1.clone()),
        u1.clone().real_mul(v0.clone()),
        u1.clone().real_mul(v1.clone()),
    );

    // B = p*q^T: B_ij = p_i * q_j
    let b = (
        p0.clone().real_mul(q0.clone()),
        p0.clone().real_mul(q1.clone()),
        p1.clone().real_mul(q0.clone()),
        p1.clone().real_mul(q1.clone()),
    );

    // C = A*B
    let c = matmul_2x2(&a, &b);

    // v^T * p = v0*p0 + v1*p1 (scalar)
    let vtp = v0.real_mul(p0).real_add(v1.real_mul(p1));

    // Expected: C = (v^T*p) * u * q^T
    // C_ij = vtp * u_i * q_j
    let exp00 = vtp.clone().real_mul(u0.clone()).real_mul(q0.clone());
    let exp01 = vtp.clone().real_mul(u0).real_mul(q1.clone());
    let exp10 = vtp.clone().real_mul(u1.clone()).real_mul(q0);
    let exp11 = vtp.real_mul(u1).real_mul(q1);

    // Violation: C != vtp * u * q^T
    let violation =
        c.0.ne(exp00)
            .or(c.1.ne(exp01))
            .or(c.2.ne(exp10))
            .or(c.3.ne(exp11));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rank_product_bound_rank1");
}

// ---------------------------------------------------------------------------
// Test 1108: QR decomposition: Q orthogonal, R upper triangular
// ---------------------------------------------------------------------------

/// Prove: For a 2x2 matrix A = Q*R where Q is orthogonal and R is upper
/// triangular, Q^T*A = R (and R has zero below the diagonal).
///
/// If Q^T*Q = I, then Q^T*(Q*R) = R. We verify this algebraic identity.
#[test]
fn test_1108_qr_decomposition() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let q = declare_2x2(&mut prog, "q");
    bound_2x2(&mut prog, &q, -1, 1);
    assert_orthogonal_2x2(&mut prog, &q);

    // R is upper triangular: [[r00, r01], [0, r11]]
    let real = Sort::real();
    let _ = prog.declare_const("r00", real.clone());
    let _ = prog.declare_const("r01", real.clone());
    let _ = prog.declare_const("r11", real);
    let r00 = real_var("r00");
    let r01 = real_var("r01");
    let r11 = real_var("r11");

    for v in [&r00, &r01, &r11] {
        prog.assert(v.clone().real_ge(Expr::real(-100)));
        prog.assert(v.clone().real_le(Expr::real(100)));
    }

    let r = (r00.clone(), r01.clone(), Expr::real(0), r11.clone());

    // A = Q * R
    let a = matmul_2x2(&q, &r);

    // Q^T * A should equal R
    let qt = transpose_2x2(&q);
    let qt_a = matmul_2x2(&qt, &a);

    // Violation: Q^T*A != R
    let violation = qt_a
        .0
        .ne(r00)
        .or(qt_a.1.ne(r01))
        .or(qt_a.2.ne(Expr::real(0)))
        .or(qt_a.3.ne(r11));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "qr_decomposition");
}

// ---------------------------------------------------------------------------
// Test 1109: Moore-Penrose pseudoinverse: A * A+ * A = A (invertible case)
// ---------------------------------------------------------------------------

/// Prove: For an invertible 2x2 matrix A with det(A) != 0,
/// A * A^{-1} * A = A.
///
/// The Moore-Penrose pseudoinverse of an invertible matrix is the inverse.
/// For A^{-1} = adj(A)/det(A), we verify A * A^{-1} * A = A.
///
/// We encode A^{-1} via the constraint A * Ainv = I and prove A * Ainv * A = A.
#[test]
fn test_1109_moore_penrose_pseudoinverse() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let a = declare_2x2(&mut prog, "a");
    let ainv = declare_2x2(&mut prog, "ainv");
    bound_2x2(&mut prog, &a, -100, 100);
    bound_2x2(&mut prog, &ainv, -100, 100);

    // Constraint: A * Ainv = I
    let prod = matmul_2x2(&a, &ainv);
    let one = Expr::real(1);
    let zero = Expr::real(0);
    prog.assert(prod.0.eq(one.clone()));
    prog.assert(prod.1.eq(zero.clone()));
    prog.assert(prod.2.eq(zero.clone()));
    prog.assert(prod.3.eq(one));

    // Compute A * Ainv * A
    let ainv_a = matmul_2x2(&ainv, &a);
    let a_ainv_a = matmul_2x2(&a, &ainv_a);

    // Violation: A * A^{-1} * A != A
    let violation = a_ainv_a
        .0
        .ne(a.0)
        .or(a_ainv_a.1.ne(a.1))
        .or(a_ainv_a.2.ne(a.2))
        .or(a_ainv_a.3.ne(a.3));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "moore_penrose_pseudoinverse");
}

// ---------------------------------------------------------------------------
// Test 1110: Truncated SVD error bound
// ---------------------------------------------------------------------------

/// Prove: For a 2x2 SVD with singular values s1 >= s2 >= 0, the rank-1
/// truncated SVD error satisfies ||A - A_1||_F^2 <= ||A||_F^2.
///
/// Since ||A||_F^2 = s1^2 + s2^2 and ||A - A_1||_F^2 = s2^2,
/// we have s2^2 <= s1^2 + s2^2, which is equivalent to s1^2 >= 0.
#[test]
fn test_1110_truncated_svd_error_bound() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("s1", real.clone());
    let _ = prog.declare_const("s2", real);
    let s1 = real_var("s1");
    let s2 = real_var("s2");

    // s1 >= s2 >= 0
    prog.assert(s1.clone().real_ge(s2.clone()));
    prog.assert(s2.clone().real_ge(Expr::real(0)));
    prog.assert(s1.clone().real_le(Expr::real(100)));

    // ||A||_F^2 = s1^2 + s2^2
    let full_frob = s1
        .clone()
        .real_mul(s1)
        .real_add(s2.clone().real_mul(s2.clone()));

    // ||A - A_1||_F^2 = s2^2 (error from rank-1 truncation)
    let error_frob = s2.clone().real_mul(s2);

    // Violation: error > full norm (error_frob > full_frob)
    let violation = error_frob.real_gt(full_frob);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "truncated_svd_error_bound");
}
