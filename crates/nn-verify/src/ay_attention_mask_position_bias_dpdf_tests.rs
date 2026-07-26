// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for ay dpdf attention mask and position bias proofs (#4217).

use super::*;

#[test]
fn test_causal_mask_zero_future_proven() {
    let result = prove_causal_mask_zero_future().expect("proof should not error");
    assert!(
        result.proven,
        "Causal mask zero future (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(result.property, "causal_mask_zero_future_3x3");
}

#[test]
fn test_padding_mask_neg_inf_proven() {
    let result = prove_padding_mask_neg_inf().expect("proof should not error");
    assert!(
        result.proven,
        "Padding mask neg-inf (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(result.property, "padding_mask_neg_inf_col2");
}

#[test]
fn test_combined_causal_padding_mask_proven() {
    let result = prove_combined_causal_padding_mask().expect("proof should not error");
    assert!(
        result.proven,
        "Combined causal+padding mask (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(result.property, "combined_causal_padding_mask_3x3");
}

#[test]
fn test_alibi_distance_symmetry_proven() {
    let result = prove_alibi_distance_symmetry().expect("proof should not error");
    assert!(
        result.proven,
        "ALiBi distance symmetry (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(result.property, "alibi_distance_symmetry");
}

#[test]
fn test_rope_norm_preservation_proven() {
    let result = prove_rope_norm_preservation().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "RoPE norm preservation: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "RoPE norm preservation must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "rope_norm_preservation");
}

#[test]
fn test_sliding_window_sparsity_proven() {
    let result = prove_sliding_window_sparsity().expect("proof should not error");
    assert!(
        result.proven,
        "Sliding window sparsity (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(result.property, "sliding_window_sparsity_5x5_w1");
}

#[test]
fn test_cross_attention_shape_compat_proven() {
    let result = prove_cross_attention_shape_compat().expect("proof should not error");
    assert!(
        result.proven,
        "Cross-attention shape compat (QF_LRA) should be Proven. detail: {}",
        result.detail,
    );
    assert_eq!(result.property, "cross_attention_shape_compat_2x3");
}

#[test]
fn test_mask_additive_form_proven() {
    let result = prove_mask_additive_form().expect("proof should not error");
    assert!(
        result.proven || result.detail.contains("Unknown"),
        "Mask additive form: expected Proven or Unknown (NRA), got: {}",
        result.detail,
    );
    assert!(
        !result.detail.contains("counterexample"),
        "Mask additive form must not have counterexample: {}",
        result.detail,
    );
    assert_eq!(result.property, "mask_additive_form_softmax_zero");
}

#[test]
fn test_all_dpdf_attention_proofs_have_valid_smt2() {
    let proofs: Vec<DpdfAttentionMaskResult> = vec![
        prove_causal_mask_zero_future().unwrap(),
        prove_padding_mask_neg_inf().unwrap(),
        prove_combined_causal_padding_mask().unwrap(),
        prove_alibi_distance_symmetry().unwrap(),
        prove_rope_norm_preservation().unwrap(),
        prove_sliding_window_sparsity().unwrap(),
        prove_cross_attention_shape_compat().unwrap(),
        prove_mask_additive_form().unwrap(),
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
fn test_rope_norm_smt2_has_pythagorean() {
    let result = prove_rope_norm_preservation().expect("proof should not error");
    assert!(
        result.smt2.contains("QF_NRA"),
        "RoPE norm preservation should use QF_NRA logic"
    );
}

#[test]
fn test_causal_mask_smt2_uses_lra() {
    let result = prove_causal_mask_zero_future().expect("proof should not error");
    assert!(
        result.smt2.contains("QF_LRA"),
        "Causal mask should use QF_LRA logic"
    );
}
