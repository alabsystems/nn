// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for ay gradient computation mathematical properties (#4241).

use super::*;
use crate::ay_vacuity::vacuity_smell;

// --- Property 1: Chain Rule ---

#[test]
fn test_chain_rule_correctness_proven() {
    let result = prove_chain_rule_correctness().expect("proof should not error");
    // QF_LRA over an affine composition is decidable: `Unknown` is not acceptable.
    assert!(
        result.proven,
        "Chain rule correctness should be proven, got: {}",
        result.detail,
    );
    assert_eq!(vacuity_smell(&result.smt2), None);
    assert!(
        !result.detail.contains("counterexample"),
        "Chain rule correctness must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "chain_rule_correctness");
}

/// The chain rule *multiplies* the outer and inner derivatives. Adding them
/// instead makes the secant `y1 - y0` disagree with the combined derivative, so
/// the query must be SAT.
#[test]
fn chain_rule_depends_on_multiplying_the_derivatives() {
    let program = build_chain_rule_correctness(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "with the derivatives added instead of multiplied the identity fails and \
         the query must be SAT; got: {detail}",
    );
}

// --- Property 2: Linear Gradient ---

#[test]
fn test_linear_gradient_proven() {
    let result = prove_linear_gradient().expect("proof should not error");
    // QF_LRA over literal perturbation points is decidable: `Unknown` is not
    // acceptable.
    assert!(
        result.proven,
        "Linear gradient should be proven, got: {}",
        result.detail,
    );
    assert_eq!(vacuity_smell(&result.smt2), None);
    assert!(
        !result.detail.contains("counterexample"),
        "Linear gradient must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "linear_gradient_dy_dx_equals_W");
}

/// The layer must apply `W`, not its transpose. Reading the weights transposed
/// sends the off-diagonal finite-difference partial to the wrong weight, so the
/// query must be SAT.
#[test]
fn linear_gradient_depends_on_the_weight_layout() {
    let program = build_linear_gradient(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "with the weights read transposed the off-diagonal partial is wrong and \
         the query must be SAT; got: {detail}",
    );
}

// --- Property 3: ReLU Subgradient ---

#[test]
fn test_relu_subgradient_proven() {
    let result = prove_relu_subgradient().expect("proof should not error");
    assert!(
        result.proven,
        "ReLU subgradient (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(result.property, "relu_subgradient");
}

// --- Property 4: Softmax Jacobian Diagonal ---

#[test]
fn test_softmax_jacobian_diagonal_proven() {
    let result = prove_softmax_jacobian_diagonal().expect("proof should not error");
    // QF_LRA over a concrete distribution is decidable: `Unknown` is not
    // acceptable.
    assert!(
        result.proven,
        "Softmax Jacobian diagonal should be proven, got: {}",
        result.detail,
    );
    assert_eq!(vacuity_smell(&result.smt2), None);
    assert!(
        !result.detail.contains("counterexample"),
        "Softmax Jacobian diagonal must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "softmax_jacobian_diagonal");
}

/// The diagonal is derived from the off-diagonals via the row-sum-zero
/// conservation law. Dropping the minus sign on the off-diagonals flips the
/// derived diagonal away from `s_0*(1 - s_0)`, so the query must be SAT.
#[test]
fn softmax_diagonal_depends_on_the_off_diagonal_sign() {
    let program = build_softmax_jacobian_diagonal(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "with the off-diagonal sign dropped the derived diagonal is wrong and \
         the query must be SAT; got: {detail}",
    );
}

#[test]
fn test_softmax_jacobian_diagonal_bounded_proven() {
    let result = prove_softmax_jacobian_diagonal_bounded().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "Softmax Jacobian diagonal bounded: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "Softmax Jacobian diagonal bounded must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "softmax_jacobian_diagonal_bounded");
}

// --- Property 5: Cross-Entropy Gradient ---

#[test]
fn test_cross_entropy_gradient_proven() {
    let result = prove_cross_entropy_gradient().expect("proof should not error");
    // QF_LRA over concrete probabilities is decidable: `Unknown` is not
    // acceptable.
    assert!(
        result.proven,
        "Cross-entropy gradient should be proven, got: {}",
        result.detail,
    );
    assert_eq!(vacuity_smell(&result.smt2), None);
    assert!(
        !result.detail.contains("counterexample"),
        "Cross-entropy gradient must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "cross_entropy_gradient");
}

/// The gradient of `-log(p)` is `-1/p`; the loss' minus sign is what makes it
/// negative. Using `grad*p = +1` forces the positive reciprocal, breaking the
/// checked values, so the query must be SAT.
#[test]
fn cross_entropy_gradient_depends_on_the_loss_sign() {
    let program = build_cross_entropy_gradient(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "with the loss sign dropped the gradients are +2/+4 not -2/-4 and \
         the query must be SAT; got: {detail}",
    );
}

#[test]
fn test_cross_entropy_gradient_negative_proven() {
    let result = prove_cross_entropy_gradient_negative().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "Cross-entropy gradient negative: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "Cross-entropy gradient negative must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "cross_entropy_gradient_negative");
}

// --- Property 6: Batch Gradient Mean ---

#[test]
fn test_batch_gradient_mean_proven() {
    let result = prove_batch_gradient_mean().expect("proof should not error");
    assert!(
        result.proven,
        "Batch gradient mean (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(vacuity_smell(&result.smt2), None);
    assert_eq!(result.property, "batch_gradient_mean");
}

/// The top-level combine must divide by the number of groups (2), not the batch
/// size (4). Dividing by 4 double-counts and makes the hierarchical mean half the
/// flat mean, so the query must be SAT.
#[test]
fn batch_gradient_mean_depends_on_the_group_count() {
    // 4 is the batch size, not the group count -- the double-counting slip.
    let program = build_batch_gradient_mean(4);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "dividing the top level by the batch size instead of the group count \
         halves the hierarchical mean and the query must be SAT; got: {detail}",
    );
}

#[test]
fn test_batch_gradient_mean_bounded_proven() {
    let result = prove_batch_gradient_mean_bounded().expect("proof should not error");
    assert!(
        result.proven,
        "Batch gradient mean bounded (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(result.property, "batch_gradient_mean_bounded");
}

// --- SMT2 Structure ---

#[test]
fn test_all_proofs_have_valid_smt2() {
    let proofs: Vec<GradientPropertyResult> = vec![
        prove_chain_rule_correctness().unwrap(),
        prove_linear_gradient().unwrap(),
        prove_relu_subgradient().unwrap(),
        prove_softmax_jacobian_diagonal().unwrap(),
        prove_softmax_jacobian_diagonal_bounded().unwrap(),
        prove_cross_entropy_gradient().unwrap(),
        prove_cross_entropy_gradient_negative().unwrap(),
        prove_batch_gradient_mean().unwrap(),
        prove_batch_gradient_mean_bounded().unwrap(),
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
fn test_nra_proofs_use_correct_logic() {
    // The remaining nonlinear proofs (a var*var product with no concrete pin)
    // still use NRA: the softmax diagonal bound and the cross-entropy sign proof.
    let nra_proofs = vec![
        prove_softmax_jacobian_diagonal_bounded().unwrap(),
        prove_cross_entropy_gradient_negative().unwrap(),
    ];

    for proof in &nra_proofs {
        assert!(
            proof.smt2.contains("QF_NRA"),
            "{}: should use QF_NRA logic",
            proof.property,
        );
    }
}

#[test]
fn test_lra_proofs_use_correct_logic() {
    // The rewritten gradient proofs are decidable linear encodings (concrete or
    // literal factors), so they, ReLU and batch mean all use LRA.
    let lra_proofs = vec![
        prove_chain_rule_correctness().unwrap(),
        prove_linear_gradient().unwrap(),
        prove_softmax_jacobian_diagonal().unwrap(),
        prove_cross_entropy_gradient().unwrap(),
        prove_relu_subgradient().unwrap(),
        prove_batch_gradient_mean().unwrap(),
    ];

    for proof in &lra_proofs {
        assert!(
            proof.smt2.contains("QF_LRA"),
            "{}: should use QF_LRA logic",
            proof.property,
        );
    }
}
