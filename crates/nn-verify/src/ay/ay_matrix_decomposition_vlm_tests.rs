// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for ay SMT proofs of matrix decomposition properties for dpdf VLMs.

use super::*;

#[test]
fn test_matmul_associativity_error_bound() {
    let result = prove_matmul_associativity_error_bound().expect("proof should not error");
    assert!(result.smt2.contains("check-sat"));
    assert!(
        result.proven,
        "Matmul associativity error bound (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(result.property, "matmul_associativity_error_bound");
}

#[test]
fn test_low_rank_error_bound() {
    let result = prove_low_rank_error_bound().expect("proof should not error");
    assert!(result.smt2.contains("check-sat"));
    // Re-encoded in the energy domain: QF_LRA over a finite box is decidable, so
    // `Unknown` is not acceptable and the bound must be strictly proven.
    assert!(
        result.proven,
        "Low-rank error bound (QF_LRA) should be Proven, got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "Low-rank error bound must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(
        crate::ay_vacuity::vacuity_smell(&result.smt2),
        None,
        "low-rank error-bound query must not be vacuous",
    );
    assert_eq!(result.property, "low_rank_error_bound");
}

/// The theorem rests on the rank-1 approximation keeping the *larger* singular
/// value. Keeping the smaller one discards the larger energy `e1`, which the
/// truncation threshold never bounds, so the error can exceed `tau^2` and the
/// query must be SAT. If it stayed UNSAT the bound would not depend on which
/// value is kept and the proof would be vacuous.
#[test]
fn low_rank_bound_depends_on_keeping_the_larger_value() {
    let program = build_low_rank_error_bound(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "keeping the smaller singular value discards e1 (unbounded by tau) and the \
         query must be SAT; got: {detail}",
    );
}

#[test]
fn test_svd_reconstruction_tolerance() {
    let result = prove_svd_reconstruction_tolerance().expect("proof should not error");
    assert!(result.smt2.contains("check-sat"));
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "SVD reconstruction tolerance: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "SVD reconstruction tolerance must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "svd_reconstruction_tolerance");
}

#[test]
fn test_cholesky_pd_preservation() {
    let result = prove_cholesky_pd_preservation().expect("proof should not error");
    assert!(result.smt2.contains("check-sat"));
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "Cholesky PD preservation: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "Cholesky PD preservation must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "cholesky_pd_preservation");
}

#[test]
fn test_qr_orthogonality_within_bounds() {
    let result = prove_qr_orthogonality_within_bounds().expect("proof should not error");
    assert!(result.smt2.contains("check-sat"));
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "QR orthogonality within bounds: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "QR orthogonality within bounds must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "qr_orthogonality_within_bounds");
}

#[test]
fn test_eigenvalue_gershgorin_bound() {
    let result = prove_eigenvalue_gershgorin_bound().expect("proof should not error");
    assert!(result.smt2.contains("check-sat"));
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "Eigenvalue Gershgorin bound: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "Eigenvalue Gershgorin bound must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "eigenvalue_gershgorin_bound");
}

// --- SMT2 Structure Tests ---

#[test]
fn test_smt2_structure_matmul_error() {
    let result = prove_matmul_associativity_error_bound().expect("proof should not error");
    assert!(result.smt2.contains("set-logic"), "should declare logic");
    assert!(result.smt2.contains("check-sat"), "should have check-sat");
    assert!(
        result.smt2.contains("declare-const"),
        "should have declarations"
    );
    assert!(
        result.smt2.contains("delta"),
        "should have delta error bound"
    );
}

#[test]
fn test_smt2_structure_gershgorin() {
    let result = prove_eigenvalue_gershgorin_bound().expect("proof should not error");
    assert!(result.smt2.contains("set-logic"), "should declare logic");
    assert!(result.smt2.contains("QF_NRA"), "should use QF_NRA logic");
    assert!(
        result.smt2.contains("lambda"),
        "should have eigenvalue variable"
    );
    assert!(
        result.smt2.contains("abs_b"),
        "should have absolute value variable"
    );
}

#[test]
fn test_smt2_structure_svd_tolerance() {
    let result = prove_svd_reconstruction_tolerance().expect("proof should not error");
    assert!(result.smt2.contains("set-logic"), "should declare logic");
    assert!(result.smt2.contains("check-sat"), "should have check-sat");
    assert!(result.smt2.contains("eps"), "should have epsilon variable");
    assert!(result.smt2.contains("du"), "should have perturbation du");
}
