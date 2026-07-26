// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for ay gradient computation proofs (#4241).

use super::*;
use crate::ay_vacuity::vacuity_smell;

// --- Chain Rule Multi-Layer ---

#[test]
fn test_chain_rule_three_layers_proven() {
    let result = prove_chain_rule_three_layers().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "Three-layer chain rule: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "Three-layer chain rule must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "chain_rule_three_layer_composition");
}

// --- Gradient Accumulation ---

#[test]
fn test_gradient_accumulation_bounds_proven() {
    let result = prove_gradient_accumulation_bounds().expect("proof should not error");
    assert!(
        result.proven,
        "Gradient accumulation bounds (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(result.property, "gradient_accumulation_bounds_n3");
}

// --- Linear Backward ---

#[test]
fn test_linear_backward_grad_w_proven() {
    let result = prove_linear_backward_grad_w().expect("proof should not error");
    // QF_LRA over concrete input activations is decidable: `Unknown` is not
    // acceptable, so require a strict UNSAT.
    assert!(
        result.proven,
        "Linear backward grad_W (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(vacuity_smell(&result.smt2), None);
    assert!(
        !result.detail.contains("counterexample"),
        "Linear backward grad_W must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "linear_backward_grad_w_outer_product");
}

/// The outer product's operand order is the whole theorem. Swapping it to
/// `outer(x, grad_out)` transposes grad_W, so its off-diagonal entries disagree
/// with the adjoint reference (x0 != x1) and the query must be SAT.
#[test]
fn grad_w_depends_on_the_operand_order() {
    let program = build_linear_backward_grad_w(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "with the outer-product operands swapped grad_W is transposed and \
         disagrees with the adjoint; the query must be SAT; got: {detail}",
    );
}

#[test]
fn test_linear_backward_grad_x_proven() {
    let result = prove_linear_backward_grad_x().expect("proof should not error");
    // QF_LRA over a concrete weight matrix is decidable: `Unknown` is not
    // acceptable, so require a strict UNSAT.
    assert!(
        result.proven,
        "Linear backward grad_x (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(vacuity_smell(&result.smt2), None);
    assert!(
        !result.detail.contains("counterexample"),
        "Linear backward grad_x must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "linear_backward_grad_x_wt_times_grad_out");
}

/// The transpose is the whole theorem. Reusing `W` unchanged instead of `W^T`
/// makes grad_x[j] contract the wrong index, disagreeing with the adjoint
/// reference wherever `W` is asymmetric (W01 != W10), so the query must be SAT.
#[test]
fn grad_x_depends_on_the_transpose() {
    let program = build_linear_backward_grad_x(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "with W left untransposed grad_x disagrees with the adjoint; \
         the query must be SAT; got: {detail}",
    );
}

// --- Gradient Clipping ---

#[test]
fn test_gradient_clipping_preserves_direction_proven() {
    let result = prove_gradient_clipping_preserves_direction().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "Gradient clipping direction: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "Gradient clipping direction must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "gradient_clipping_preserves_direction");
}

#[test]
fn test_gradient_clipping_reduces_norm_proven() {
    let result = prove_gradient_clipping_reduces_norm().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "Gradient clipping norm reduction: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "Gradient clipping norm reduction must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "gradient_clipping_reduces_norm");
}

// --- Gradient Scaling ---

#[test]
fn test_gradient_scaling_relative_magnitudes_proven() {
    let result = prove_gradient_scaling_relative_magnitudes().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "Gradient scaling relative magnitudes: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "Gradient scaling relative magnitudes must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "gradient_scaling_relative_magnitudes");
}

#[test]
fn test_gradient_scaling_preserves_sign_proven() {
    let result = prove_gradient_scaling_preserves_sign().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "Gradient scaling sign preservation: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "Gradient scaling sign preservation must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "gradient_scaling_preserves_sign");
}

// --- Mixed-Precision ---

#[test]
fn test_mixed_precision_conversion_bounds_proven() {
    let result = prove_mixed_precision_conversion_bounds().expect("proof should not error");
    assert!(
        result.proven,
        "Mixed-precision conversion bounds (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(result.property, "mixed_precision_conversion_error_bounded");
}

#[test]
fn test_mixed_precision_accumulation_bound_proven() {
    let result = prove_mixed_precision_accumulation_bound().expect("proof should not error");
    assert!(
        result.proven,
        "Mixed-precision accumulation bound (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(result.property, "mixed_precision_accumulation_bound_n3");
}

// --- Gradient Checkpointing ---

#[test]
fn test_gradient_checkpointing_equivalence_proven() {
    let result = prove_gradient_checkpointing_equivalence().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "Gradient checkpointing equivalence: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "Gradient checkpointing equivalence must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "gradient_checkpointing_equivalence");
}

#[test]
fn test_gradient_checkpoint_multi_step_proven() {
    let result = prove_gradient_checkpoint_multi_step().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "Gradient checkpoint multi-step: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "Gradient checkpoint multi-step must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(
        result.property,
        "gradient_checkpoint_multi_step_equivalence"
    );
}

// --- SMT2 Structure ---

#[test]
fn test_all_proofs_have_valid_smt2() {
    let proofs: Vec<GradientComputationResult> = vec![
        prove_chain_rule_three_layers().unwrap(),
        prove_gradient_accumulation_bounds().unwrap(),
        prove_linear_backward_grad_w().unwrap(),
        prove_linear_backward_grad_x().unwrap(),
        prove_gradient_clipping_preserves_direction().unwrap(),
        prove_gradient_clipping_reduces_norm().unwrap(),
        prove_gradient_scaling_relative_magnitudes().unwrap(),
        prove_gradient_scaling_preserves_sign().unwrap(),
        prove_mixed_precision_conversion_bounds().unwrap(),
        prove_mixed_precision_accumulation_bound().unwrap(),
        prove_gradient_checkpointing_equivalence().unwrap(),
        prove_gradient_checkpoint_multi_step().unwrap(),
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
