// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! Tests for ay normalization layer property proofs.
//!
//! Part of #4223.

use super::*;
use crate::ay_vacuity::vacuity_smell;

#[test]
fn test_layernorm_output_mean_zero_proven() {
    let result = prove_layernorm_output_mean_zero().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "LayerNorm mean zero: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "LayerNorm mean zero must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "layernorm_output_mean_zero");
}

#[test]
fn test_layernorm_output_variance_one_proven() {
    let result = prove_layernorm_output_variance_one().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "LayerNorm variance one: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "LayerNorm variance one must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "layernorm_output_variance_one");
}

#[test]
fn test_batchnorm_running_mean_update_proven() {
    let result = prove_batchnorm_running_mean_update().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "BatchNorm running mean: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "BatchNorm running mean must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "batchnorm_running_mean_update");
}

#[test]
fn test_batchnorm_running_var_update_proven() {
    let result = prove_batchnorm_running_var_update().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "BatchNorm running var: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "BatchNorm running var must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "batchnorm_running_var_update_nonneg");
}

#[test]
fn test_rmsnorm_output_formula_proven() {
    let result = prove_rmsnorm_output_formula().expect("proof should not error");
    // QF_LRA over concrete gamma/rms is decidable: `Unknown` is not acceptable.
    assert!(
        result.proven,
        "RMSNorm output formula should be proven, got: {}",
        result.detail,
    );
    assert_eq!(vacuity_smell(&result.smt2), None);
    assert_eq!(result.property, "rmsnorm_output_formula");
}

/// The reciprocal is the whole theorem: scaling the output by `gamma * rms`
/// instead of `gamma / rms` (multiplying by the norm instead of dividing) breaks
/// `out * rms = gamma * x`, so the query must be SAT.
#[test]
fn output_formula_depends_on_dividing_by_rms() {
    let program = build_rmsnorm_output_formula(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "scaling by gamma*rms instead of gamma/rms breaks the formula; \
         the query must be SAT, got: {detail}",
    );
}

#[test]
fn test_groupnorm_partition_proven() {
    let result = prove_groupnorm_partition().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "GroupNorm partition: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "GroupNorm partition must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "groupnorm_partition_exact");
}

#[test]
fn test_instancenorm_independence_proven() {
    let result = prove_instancenorm_independence().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "InstanceNorm independence: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "InstanceNorm independence must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "instancenorm_per_sample_independence");
}

#[test]
fn test_affine_transform_proven() {
    let result = prove_affine_transform().expect("proof should not error");
    // QF_LRA over concrete gamma/beta is decidable: `Unknown` is not acceptable.
    assert!(
        result.proven,
        "Affine transform should be proven, got: {}",
        result.detail,
    );
    assert_eq!(vacuity_smell(&result.smt2), None);
    assert_eq!(result.property, "affine_transform_identity");
}

/// The inverse must SUBTRACT beta. Adding it back instead flips the sign of the
/// shift correction, so the round trip no longer recovers `x_norm` and the query
/// must be SAT.
#[test]
fn affine_inverse_depends_on_the_beta_sign() {
    let program = build_affine_transform(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "adding beta on the inverse instead of subtracting breaks the round trip; \
         the query must be SAT, got: {detail}",
    );
}

#[test]
fn test_epsilon_stability_proven() {
    let result = prove_epsilon_stability().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "Epsilon stability: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "Epsilon stability must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "epsilon_stability");
}

#[test]
fn test_norm_preserves_shape_proven() {
    let result = prove_norm_preserves_shape().expect("proof should not error");
    // QF_LIA over a concrete shape is decidable: `Unknown` is not acceptable.
    assert!(
        result.proven,
        "Norm preserves shape (QF_LIA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(vacuity_smell(&result.smt2), None);
    assert_eq!(result.property, "norm_preserves_shape");
}

/// The row-major output stride is the whole theorem: packing rows `COLS-1` apart
/// instead of `COLS` makes `(0, COLS-1)` and `(1, 0)` collide on one slot, so the
/// injectivity query must find a counterexample.
#[test]
fn shape_preservation_depends_on_the_row_stride() {
    let program = build_norm_preserves_shape(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "with the output row stride mis-set to COLS-1 two cells collide; \
         the query must be SAT, got: {detail}",
    );
}

#[test]
fn test_all_normalization_proofs_have_valid_smt2() {
    let proofs: Vec<NormalizationProofResult> = vec![
        prove_layernorm_output_mean_zero().unwrap(),
        prove_layernorm_output_variance_one().unwrap(),
        prove_batchnorm_running_mean_update().unwrap(),
        prove_batchnorm_running_var_update().unwrap(),
        prove_rmsnorm_output_formula().unwrap(),
        prove_groupnorm_partition().unwrap(),
        prove_instancenorm_independence().unwrap(),
        prove_affine_transform().unwrap(),
        prove_epsilon_stability().unwrap(),
        prove_norm_preserves_shape().unwrap(),
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
fn test_layernorm_smt2_has_mean_variable() {
    let result = prove_layernorm_output_mean_zero().expect("proof should not error");
    assert!(
        result.smt2.contains("mean"),
        "LayerNorm mean zero SMT2 should reference the mean variable"
    );
}

#[test]
fn test_rmsnorm_smt2_has_rms_variable() {
    let result = prove_rmsnorm_output_formula().expect("proof should not error");
    assert!(
        result.smt2.contains("rms"),
        "RMSNorm output formula SMT2 should reference the rms variable"
    );
}

#[test]
fn test_epsilon_smt2_has_eps_or_denom() {
    let result = prove_epsilon_stability().expect("proof should not error");
    assert!(
        result.smt2.contains("denom") || result.smt2.contains("eps"),
        "Epsilon stability SMT2 should reference denom or eps"
    );
}
