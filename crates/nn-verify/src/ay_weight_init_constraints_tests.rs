// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for ay weight initialization and parameter constraint proofs (#4214).

use super::*;
use crate::ay_vacuity::vacuity_smell;

// --- Xavier/Glorot ---

#[test]
fn test_xavier_uniform_bounds_proven() {
    let result = prove_xavier_uniform_bounds().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "Xavier uniform bounds: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "Xavier uniform bounds must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "xavier_uniform_weight_bounded");
}

// --- Kaiming/He ---

#[test]
fn test_kaiming_variance_proven() {
    let result = prove_kaiming_variance().expect("proof should not error");
    // QF_LRA over a concrete fan-in is decidable: `Unknown` is not acceptable.
    assert!(
        result.proven,
        "Kaiming variance preservation should be proven, got: {}",
        result.detail,
    );
    assert_eq!(vacuity_smell(&result.smt2), None);
    assert_eq!(result.property, "kaiming_variance_constraint");
}

/// The gain of 2 is the whole theorem: with Xavier's gain of 1 the ReLU layer
/// only preserves half the variance, so `Var(out) = Var(in)` must fail and the
/// query must be SAT.
#[test]
fn kaiming_variance_depends_on_the_gain() {
    let program = build_kaiming_variance(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "with gain = 1 (Xavier) the output variance is halved and the query must be SAT; \
         got: {detail}",
    );
}

// --- Orthogonal Init ---

#[test]
fn test_orthogonal_init_proven() {
    let result = prove_orthogonal_init().expect("proof should not error");
    // QF_LRA over a concrete rotation is decidable: `Unknown` is not acceptable.
    assert!(
        result.proven,
        "Orthogonal init W^T W = I should be proven, got: {}",
        result.detail,
    );
    assert_eq!(vacuity_smell(&result.smt2), None);
    assert_eq!(result.property, "orthogonal_init_wtw_identity");
}

/// Orthonormality of the columns is the whole theorem. Setting `d = 4/5` leaves
/// column 1 with squared norm `32/25 != 1`, so `W^T W` is no longer the identity
/// and the query must be SAT.
#[test]
fn orthogonal_init_depends_on_orthonormality() {
    let program = build_orthogonal_init(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "with column 1 not unit-norm the Gram matrix differs from I and the query \
         must be SAT; got: {detail}",
    );
}

// --- Zero Bias ---

#[test]
fn test_zero_bias_identity_proven() {
    let result = prove_zero_bias_identity().expect("proof should not error");
    assert!(
        result.proven,
        "Zero bias identity (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(result.property, "zero_bias_preserves_linear_output");
}

// --- Frobenius Norm ---

#[test]
fn test_frobenius_norm_bound_proven() {
    let result = prove_frobenius_norm_bound().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "Frobenius norm bound: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "Frobenius norm bound must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "frobenius_norm_bounded_by_n_limit_sq");
}

// --- Gradient Scaling ---

#[test]
fn test_gradient_scaling_fan_in_proven() {
    let result = prove_gradient_scaling_fan_in().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "Gradient scaling fan_in: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "Gradient scaling fan_in must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "gradient_magnitude_scales_with_input");
}

// --- Parameter Count ---

#[test]
fn test_param_count_uniform_stack_proven() {
    let result = prove_param_count_uniform_stack().expect("proof should not error");
    // QF_LIA over integer counts is decidable: `Unknown` is not acceptable.
    assert!(
        result.proven,
        "Uniform-stack parameter count (QF_LIA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(vacuity_smell(&result.smt2), None);
    assert_eq!(result.property, "parameter_count_uniform_stack_3x");
}

/// Uniformity is load-bearing: without the hypothesis that every layer carries
/// the same per-layer count, a heterogeneous stack's total (the actual per-layer
/// sum) need not equal the `3 * p0` shortcut, so the query must be SAT.
#[test]
fn param_count_uniform_stack_depends_on_uniformity() {
    let program = build_param_count_uniform_stack(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "with the layers unconstrained the per-layer sum can differ from 3 * p0 and \
         the query must be SAT; got: {detail}",
    );
}

// --- Spectral Norm ---

#[test]
fn test_spectral_norm_bound_proven() {
    let result = prove_spectral_norm_bound().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "Spectral norm bound: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "Spectral norm bound must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "spectral_norm_le_frobenius_norm");
}

// --- SMT2 Structure ---

#[test]
fn test_all_proofs_have_valid_smt2() {
    let proofs: Vec<WeightInitResult> = vec![
        prove_xavier_uniform_bounds().unwrap(),
        prove_kaiming_variance().unwrap(),
        prove_orthogonal_init().unwrap(),
        prove_zero_bias_identity().unwrap(),
        prove_frobenius_norm_bound().unwrap(),
        prove_gradient_scaling_fan_in().unwrap(),
        prove_param_count_uniform_stack().unwrap(),
        prove_spectral_norm_bound().unwrap(),
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

#[test]
fn test_xavier_smt2_has_fan_variables() {
    let result = prove_xavier_uniform_bounds().expect("proof should not error");
    assert!(
        result.smt2.contains("fan_in"),
        "Xavier SMT2 should reference fan_in"
    );
    assert!(
        result.smt2.contains("fan_out"),
        "Xavier SMT2 should reference fan_out"
    );
}

#[test]
fn test_orthogonal_smt2_has_matrix_elements() {
    let result = prove_orthogonal_init().expect("proof should not error");
    for var in ["a", "b", "c", "d"] {
        assert!(
            result
                .smt2
                .contains(&format!("(declare-const {} Real)", var)),
            "Orthogonal init SMT2 should declare matrix element {}",
            var,
        );
    }
}

#[test]
fn test_spectral_norm_smt2_has_eigenvector() {
    let result = prove_spectral_norm_bound().expect("proof should not error");
    assert!(
        result.smt2.contains("v0"),
        "Spectral norm SMT2 should reference eigenvector component v0"
    );
    assert!(
        result.smt2.contains("v1"),
        "Spectral norm SMT2 should reference eigenvector component v1"
    );
}
