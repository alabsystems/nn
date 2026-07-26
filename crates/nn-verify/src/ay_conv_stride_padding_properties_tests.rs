// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `ay_conv_stride_padding_properties` module.
//!
//! Each test verifies that the corresponding proof function succeeds and
//! returns a valid SMT-LIB2 encoding. QF_LRA proofs should be deterministic
//! (always Proven); we assert `proven == true` for all properties.
//!
//! Part of #4226.

use super::*;
use crate::ay_vacuity::vacuity_smell;

#[test]
fn test_output_dimension_formula_proven() {
    let result = prove_output_dimension_formula().expect("proof should not error");
    assert!(
        result.proven,
        "Output dimension formula (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(result.property, "output_dimension_formula");
}

#[test]
fn test_transposed_conv_output_proven() {
    let result = prove_transposed_conv_output().expect("proof should not error");
    assert!(
        result.proven,
        "Transposed conv output (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(vacuity_smell(&result.smt2), None);
    assert!(
        !result.detail.contains("counterexample"),
        "Transposed conv output must not have a counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "transposed_conv_output");
}

/// The transpose recovers `N` only because the padding term is *subtracted* on
/// the way back. Adding it with the wrong sign yields `N + 4P`, which differs
/// from `N` whenever `P > 0`, so the query must be SAT.
#[test]
fn transposed_conv_output_depends_on_padding_sign() {
    let program = build_transposed_conv_output(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "with the padding term added back with the wrong sign the transpose no \
         longer recovers N and the query must be SAT; got: {detail}",
    );
}

#[test]
fn test_dilated_effective_kernel_proven() {
    let result = prove_dilated_effective_kernel().expect("proof should not error");
    assert!(
        result.proven,
        "Dilated effective kernel (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(result.property, "dilated_effective_kernel");
}

#[test]
fn test_same_padding_preserves_length_proven() {
    let result = prove_same_padding_preserves_length().expect("proof should not error");
    assert!(
        result.proven,
        "Same padding preserves length (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(result.property, "same_padding_preserves_length");
}

#[test]
fn test_valid_padding_shrinks_proven() {
    let result = prove_valid_padding_shrinks().expect("proof should not error");
    assert!(
        result.proven,
        "Valid padding shrinks (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(result.property, "valid_padding_shrinks");
}

#[test]
fn test_causal_padding_preserves_length_proven() {
    let result = prove_causal_padding_preserves_length().expect("proof should not error");
    assert!(
        result.proven,
        "Causal padding preserves length (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(result.property, "causal_padding_preserves_length");
}

#[test]
fn test_depthwise_conv_params_proven() {
    let result = prove_depthwise_conv_params().expect("proof should not error");
    assert!(
        result.proven,
        "Depthwise conv params (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(vacuity_smell(&result.smt2), None);
    assert!(
        !result.detail.contains("counterexample"),
        "Depthwise conv params must not have a counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "depthwise_conv_params");
}

/// The param count is `C_in * K` only because depthwise sets `groups = C_in`
/// (so `cpg = 1`). Leaving `groups = 1` makes `cpg = C_in` and the standard-conv
/// count `C_in^2 * K != C_in * K`, so the query must be SAT.
#[test]
fn depthwise_params_depend_on_the_group_count() {
    let program = build_depthwise_conv_params(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "with groups=1 (not depthwise) the count is C_in^2*K, not C_in*K, and \
         the query must be SAT; got: {detail}",
    );
}

#[test]
fn test_group_conv_weight_partition_proven() {
    let result = prove_group_conv_weight_partition().expect("proof should not error");
    assert!(
        result.proven,
        "Group conv weight partition (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(vacuity_smell(&result.smt2), None);
    assert!(
        !result.detail.contains("counterexample"),
        "Group conv weight partition must not have a counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "group_conv_weight_partition");
}

/// Grouped params stay within the standard count only because the partition
/// *divides* by `G` (`total * G = standard`). Multiplying instead
/// (`total = standard * G`) makes `total > standard`, so the query must be SAT.
#[test]
fn group_partition_depends_on_dividing_by_groups() {
    let program = build_group_conv_weight_partition(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "with the count multiplied by G instead of divided, total > standard and \
         the query must be SAT; got: {detail}",
    );
}

#[test]
fn test_all_conv_proofs_have_valid_smt2() {
    let proofs: Vec<ConvPropertyResult> = vec![
        prove_output_dimension_formula().unwrap(),
        prove_transposed_conv_output().unwrap(),
        prove_dilated_effective_kernel().unwrap(),
        prove_same_padding_preserves_length().unwrap(),
        prove_valid_padding_shrinks().unwrap(),
        prove_causal_padding_preserves_length().unwrap(),
        prove_depthwise_conv_params().unwrap(),
        prove_group_conv_weight_partition().unwrap(),
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
fn test_all_conv_proofs_use_qf_lra() {
    let proofs: Vec<ConvPropertyResult> = vec![
        prove_output_dimension_formula().unwrap(),
        prove_transposed_conv_output().unwrap(),
        prove_dilated_effective_kernel().unwrap(),
        prove_same_padding_preserves_length().unwrap(),
        prove_valid_padding_shrinks().unwrap(),
        prove_causal_padding_preserves_length().unwrap(),
        prove_depthwise_conv_params().unwrap(),
        prove_group_conv_weight_partition().unwrap(),
    ];

    for proof in &proofs {
        assert!(
            proof.smt2.contains("QF_LRA"),
            "{}: should use QF_LRA logic for deterministic solving",
            proof.property,
        );
    }
}
