// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for ay SMT quantization error proofs for dpdf VLMs (#4238).

use super::*;

#[test]
fn test_f32_to_bf16_rounding_error_proven() {
    let result = prove_f32_to_bf16_rounding_error().expect("proof should not error");
    assert!(
        result.proven,
        "F32->BF16 rounding error bound (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(result.property, "f32_to_bf16_rounding_error_bounded");
}

#[test]
fn test_f32_to_f16_truncation_error_proven() {
    let result = prove_f32_to_f16_truncation_error().expect("proof should not error");
    assert!(
        result.proven,
        "F32->F16 truncation error bound (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(result.property, "f32_to_f16_truncation_error_bounded");
}

#[test]
fn test_symmetric_quantization_preserves_zero_proven() {
    let result = prove_symmetric_quantization_preserves_zero().expect("proof should not error");
    // QF_LIA over a concrete integer scale is decidable: `Unknown` is not acceptable.
    assert!(
        result.proven,
        "Symmetric quantization preserves zero (QF_LIA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(
        crate::ay_vacuity::vacuity_smell(&result.smt2),
        None,
        "symmetric-zero proof must not be vacuous",
    );
    assert_eq!(result.property, "symmetric_quantization_preserves_zero");
}

/// A biased (non-centered) rounding interval maps input 0 to code 1, so the
/// zero-preservation query must find a counterexample.
#[test]
fn symmetric_zero_depends_on_centered_rounding() {
    let program = build_symmetric_quantization_preserves_zero(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "with a biased rounding interval 0 quantizes to a nonzero code and the query \
         must be SAT; got: {detail}",
    );
}

#[test]
fn test_asymmetric_quantization_range_mapping_proven() {
    let result = prove_asymmetric_quantization_range_mapping().expect("proof should not error");
    // QF_LIA over concrete integer levels is decidable: `Unknown` is not acceptable.
    assert!(
        result.proven,
        "Asymmetric quantization range mapping (QF_LIA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(
        crate::ay_vacuity::vacuity_smell(&result.smt2),
        None,
        "asymmetric range-mapping proof must not be vacuous",
    );
    assert_eq!(result.property, "asymmetric_quantization_range_mapping");
}

/// Flipping the zero-point sign to `z = +n_min` leaves the min endpoint at
/// `2*n_min`, nonzero whenever `n_min != 0`, so the query must be SAT.
#[test]
fn range_mapping_depends_on_zero_point_sign() {
    let program = build_asymmetric_quantization_range_mapping(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "with the zero-point sign flipped the min endpoint misses code 0 and the query \
         must be SAT; got: {detail}",
    );
}

#[test]
fn test_dequantize_inverts_quantize_proven() {
    let result = prove_dequantize_inverts_quantize().expect("proof should not error");
    // QF_LIA over a concrete integer scale is decidable: `Unknown` is not acceptable.
    assert!(
        result.proven,
        "Dequantize inverts quantize (QF_LIA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(
        crate::ay_vacuity::vacuity_smell(&result.smt2),
        None,
        "dequantize-roundtrip proof must not be vacuous",
    );
    assert_eq!(result.property, "dequantize_inverts_quantize_within_error");
}

/// Truncation (floor) rounding admits an error of `s - 1 > s/2`, so the roundtrip
/// bound must fail and the query must be SAT.
#[test]
fn roundtrip_bound_depends_on_nearest_rounding() {
    let program = build_dequantize_inverts_quantize(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "truncation widens the roundtrip error past s/2 and the query must be SAT; got: {detail}",
    );
}

#[test]
fn test_quantized_matmul_error_accumulation_proven() {
    let result = prove_quantized_matmul_error_accumulation().expect("proof should not error");
    assert!(
        result.proven,
        "Quantized matmul error accumulation (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(result.property, "quantized_matmul_error_accumulation_d2");
}

#[test]
fn test_mixed_precision_chain_error_proven() {
    let result = prove_mixed_precision_chain_error().expect("proof should not error");
    assert!(
        result.proven,
        "Mixed-precision chain error (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(result.property, "mixed_precision_chain_error_bf16_f16");
}

// SMT2 structure tests

#[test]
fn test_bf16_rounding_smt2_structure() {
    let result = prove_f32_to_bf16_rounding_error().expect("proof should not error");
    assert!(result.smt2.contains("set-logic"), "should declare logic");
    assert!(result.smt2.contains("QF_LRA"), "should use QF_LRA");
    assert!(result.smt2.contains("check-sat"), "should have check-sat");
    assert!(
        result.smt2.contains("declare-const"),
        "should have declarations"
    );
}

#[test]
fn test_f16_truncation_smt2_structure() {
    let result = prove_f32_to_f16_truncation_error().expect("proof should not error");
    assert!(result.smt2.contains("set-logic"), "should declare logic");
    assert!(result.smt2.contains("QF_LRA"), "should use QF_LRA");
    assert!(result.smt2.contains("check-sat"), "should have check-sat");
}

#[test]
fn test_symmetric_zero_smt2_structure() {
    let result = prove_symmetric_quantization_preserves_zero().expect("proof should not error");
    assert!(result.smt2.contains("set-logic"), "should declare logic");
    assert!(result.smt2.contains("check-sat"), "should have check-sat");
}

#[test]
fn test_matmul_accumulation_smt2_structure() {
    let result = prove_quantized_matmul_error_accumulation().expect("proof should not error");
    assert!(result.smt2.contains("set-logic"), "should declare logic");
    assert!(result.smt2.contains("QF_LRA"), "should use QF_LRA");
    assert!(result.smt2.contains("check-sat"), "should have check-sat");
}

#[test]
fn test_chain_error_smt2_structure() {
    let result = prove_mixed_precision_chain_error().expect("proof should not error");
    assert!(result.smt2.contains("set-logic"), "should declare logic");
    assert!(result.smt2.contains("check-sat"), "should have check-sat");
}
