// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for ay reshape and view property proofs (#4220).

use super::*;
use crate::ay_vacuity::vacuity_smell;

// --- Element Count Preservation ---

#[test]
fn test_element_count_preservation_proven() {
    let result = prove_element_count_preservation().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "Element count preservation: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "Element count preservation must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "element_count_preservation");
}

// --- Contiguous Stride Computation ---

#[test]
fn test_contiguous_stride_computation_proven() {
    let result = prove_contiguous_stride_computation().expect("proof should not error");
    // QF_LIA over a concrete shape is decidable: `Unknown` is not acceptable.
    assert!(
        result.proven,
        "Contiguous stride computation should be proven, got: {}",
        result.detail,
    );
    assert_eq!(vacuity_smell(&result.smt2), None);
    assert!(
        !result.detail.contains("counterexample"),
        "Contiguous stride computation must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "contiguous_stride_computation");
}

/// The row-major outer stride is the whole theorem. Using `d1` where `d1*d2`
/// belongs collapses two distinct coordinates onto one physical slot, so the
/// injectivity query must find a counterexample.
#[test]
fn contiguous_stride_depends_on_the_outer_stride() {
    let program = build_contiguous_stride_computation(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "with the outer stride mis-set to d1 the map collides and the query must be SAT; \
         got: {detail}",
    );
}

// --- Transpose Stride Swap ---

#[test]
fn test_transpose_stride_swap_proven() {
    let result = prove_transpose_stride_swap().expect("proof should not error");
    // QF_LIA over a concrete shape is decidable: `Unknown` is not acceptable.
    assert!(
        result.proven,
        "Transpose stride swap (QF_LIA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(vacuity_smell(&result.smt2), None);
    assert!(
        !result.detail.contains("counterexample"),
        "Transpose stride swap must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "transpose_stride_swap");
}

/// The stride *swap* is the whole theorem. Leaving the transposed strides
/// unswapped applies the big outer stride `s0` to the wide `d2` axis, so the
/// largest coordinate the transposed view names escapes the `[0, N)` buffer and
/// the query must be SAT.
#[test]
fn transpose_stride_swap_depends_on_the_swap() {
    let program = build_transpose_stride_swap(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "with the transposed strides left unswapped the biggest transposed index \
         escapes the buffer and the query must be SAT; got: {detail}",
    );
}

// --- Reshape Invertibility ---

#[test]
fn test_reshape_invertibility_proven() {
    let result = prove_reshape_invertibility().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "Reshape invertibility: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "Reshape invertibility must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "reshape_invertibility");
}

// --- View Offset Computation ---

#[test]
fn test_view_offset_computation_proven() {
    let result = prove_view_offset_computation().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "View offset computation: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "View offset computation must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "view_offset_computation");
}

// --- Broadcast Stride Zero ---

#[test]
fn test_broadcast_stride_zero_proven() {
    let result = prove_broadcast_stride_zero().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "Broadcast stride zero: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "Broadcast stride zero must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "broadcast_stride_zero");
}

// --- Flatten Element Count ---

#[test]
fn test_flatten_element_count_proven() {
    let result = prove_flatten_element_count().expect("proof should not error");
    // QF_LIA over a concrete shape is decidable: `Unknown` is not acceptable.
    assert!(
        result.proven,
        "Flatten element count should be proven, got: {}",
        result.detail,
    );
    assert_eq!(vacuity_smell(&result.smt2), None);
    assert!(
        !result.detail.contains("counterexample"),
        "Flatten element count must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "flatten_element_count");
}

/// The full product is the whole theorem. Undercounting the flatten length as
/// `d0*d1` (forgetting the last factor) lets the largest flat index run off the
/// end of the buffer, so the range query must find a counterexample.
#[test]
fn flatten_count_depends_on_all_dims() {
    let program = build_flatten_element_count(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "with the flatten length undercounted to d0*d1 the flat index escapes the \
         buffer and the query must be SAT; got: {detail}",
    );
}

// --- Unflatten Inverse ---

#[test]
fn test_unflatten_inverse_proven() {
    let result = prove_unflatten_inverse().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "Unflatten inverse: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "Unflatten inverse must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "unflatten_inverse");
}

// --- SMT2 Structure ---

#[test]
fn test_all_proofs_have_valid_smt2() {
    let proofs: Vec<ReshapeViewResult> = vec![
        prove_element_count_preservation().unwrap(),
        prove_contiguous_stride_computation().unwrap(),
        prove_transpose_stride_swap().unwrap(),
        prove_reshape_invertibility().unwrap(),
        prove_view_offset_computation().unwrap(),
        prove_broadcast_stride_zero().unwrap(),
        prove_flatten_element_count().unwrap(),
        prove_unflatten_inverse().unwrap(),
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
fn test_broadcast_smt2_has_stride_zero() {
    let result = prove_broadcast_stride_zero().expect("proof should not error");
    assert!(
        result.smt2.contains("offset_a") && result.smt2.contains("offset_b"),
        "Broadcast SMT2 should reference both offset variables"
    );
}

#[test]
fn test_view_offset_smt2_has_index_variables() {
    let result = prove_view_offset_computation().expect("proof should not error");
    assert!(
        result.smt2.contains("i0") && result.smt2.contains("i1") && result.smt2.contains("i2"),
        "View offset SMT2 should reference multi-index variables i0, i1, i2"
    );
}
