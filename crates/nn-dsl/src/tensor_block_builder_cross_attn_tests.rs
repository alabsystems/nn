// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for cross-attention builder (#779 Phase D).
//!
//! Covers `add_multi_head_cross_attention` and `add_cross_attention_transformer_block`.

use super::*;
use crate::tensor_block_builder::TensorBlockBuilder;
use crate::tensor_ir::{
    TensorIRError, TensorIRLayerError, TensorKernelDef, TensorNodeId, TensorOpKind,
};

// ===================================================================
// add_multi_head_cross_attention tests
// ===================================================================

/// Helper: build a cross-MHA graph with separate Q and KV inputs.
fn build_cross_mha(
    q_seq: usize,
    kv_seq: usize,
    model_dim: usize,
    num_heads: usize,
) -> Result<TensorKernelDef, TensorIRError> {
    let mut b = TensorBlockBuilder::new("cross_mha_test");
    let q_input = b.add_input("q_input", &[q_seq, model_dim]);
    let kv_input = b.add_input("kv_input", &[kv_seq, model_dim]);
    let q_w = b.add_input("q_weight", &[model_dim, model_dim]);
    let k_w = b.add_input("k_weight", &[model_dim, model_dim]);
    let v_w = b.add_input("v_weight", &[model_dim, model_dim]);
    let out_w = b.add_input("out_weight", &[model_dim, model_dim]);

    let out = b.add_multi_head_cross_attention(
        q_input,
        kv_input,
        q_w,
        k_w,
        v_w,
        out_w,
        num_heads,
        AttentionMask::Standard,
        &[q_seq, model_dim],
    )?;
    b.build(out)
}

#[test]
fn cross_mha_builds_successfully() {
    let def = build_cross_mha(4, 6, 8, 2).expect("valid cross-MHA");
    assert_eq!(def.name, "cross_mha_test");
    // Output shape matches Q sequence length, not KV
    assert_eq!(def.nodes[def.output.index()].shape, vec![4, 8]);
}

#[test]
fn cross_mha_different_seq_lengths() {
    // Q has 3 tokens, KV has 10 tokens — asymmetric cross-attention
    let def = build_cross_mha(3, 10, 16, 4).expect("valid");
    assert_eq!(def.nodes[def.output.index()].shape, vec![3, 16]);
}

#[test]
fn cross_mha_equal_seq_lengths() {
    // When Q and KV have same length, behaves like self-attention structurally
    let def = build_cross_mha(4, 4, 8, 2).expect("valid");
    assert_eq!(def.nodes[def.output.index()].shape, vec![4, 8]);
}

#[test]
fn cross_mha_rejects_zero_heads() {
    let result = build_cross_mha(4, 6, 8, 0);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        TensorIRError::Layer(TensorIRLayerError::MhaZeroHeads)
    ));
}

#[test]
fn cross_mha_rejects_indivisible_heads() {
    // model_dim=8, num_heads=3 → 8 % 3 != 0
    let result = build_cross_mha(4, 6, 8, 3);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        TensorIRError::Layer(TensorIRLayerError::MhaHeadDimNotDivisible { .. })
    ));
}

#[test]
fn cross_mha_rejects_wrong_q_rank() {
    let mut b = TensorBlockBuilder::new("rank3_q");
    let q = b.add_input("q", &[2, 4, 8]); // rank 3
    let kv = b.add_input("kv", &[6, 8]);
    let q_w = b.add_input("q_w", &[8, 8]);
    let k_w = b.add_input("k_w", &[8, 8]);
    let v_w = b.add_input("v_w", &[8, 8]);
    let out_w = b.add_input("out_w", &[8, 8]);

    let result = b.add_multi_head_cross_attention(
        q,
        kv,
        q_w,
        k_w,
        v_w,
        out_w,
        2,
        AttentionMask::Standard,
        &[4, 8],
    );
    assert!(matches!(
        result.unwrap_err(),
        TensorIRError::Layer(TensorIRLayerError::MhaInputRankInvalid { rank: 3 })
    ));
}

#[test]
fn cross_mha_rejects_wrong_kv_rank() {
    let mut b = TensorBlockBuilder::new("rank3_kv");
    let q = b.add_input("q", &[4, 8]);
    let kv = b.add_input("kv", &[2, 6, 8]); // rank 3
    let q_w = b.add_input("q_w", &[8, 8]);
    let k_w = b.add_input("k_w", &[8, 8]);
    let v_w = b.add_input("v_w", &[8, 8]);
    let out_w = b.add_input("out_w", &[8, 8]);

    let result = b.add_multi_head_cross_attention(
        q,
        kv,
        q_w,
        k_w,
        v_w,
        out_w,
        2,
        AttentionMask::Standard,
        &[4, 8],
    );
    assert!(matches!(
        result.unwrap_err(),
        TensorIRError::Layer(TensorIRLayerError::MhaInputRankInvalid { rank: 3 })
    ));
}

#[test]
fn cross_mha_rejects_dim_mismatch() {
    let mut b = TensorBlockBuilder::new("dim_mismatch");
    let q = b.add_input("q", &[4, 8]);
    let kv = b.add_input("kv", &[6, 16]); // model_dim=16 != 8
    let q_w = b.add_input("q_w", &[8, 8]);
    let k_w = b.add_input("k_w", &[8, 8]);
    let v_w = b.add_input("v_w", &[8, 8]);
    let out_w = b.add_input("out_w", &[8, 8]);

    let result = b.add_multi_head_cross_attention(
        q,
        kv,
        q_w,
        k_w,
        v_w,
        out_w,
        2,
        AttentionMask::Standard,
        &[4, 8],
    );
    assert!(result.is_err());
}

#[test]
fn cross_mha_contains_attention_node() {
    let def = build_cross_mha(4, 6, 8, 2).expect("valid");
    let attn_count = def
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, TensorOpKind::Attention { .. }))
        .count();
    assert_eq!(attn_count, 1, "cross-MHA has exactly 1 Attention node");
}

#[test]
fn cross_mha_attention_has_correct_q_kv_shapes() {
    // Q: [4, 8] → 2 heads → [2, 4, 4], KV: [6, 8] → [2, 6, 4]
    let def = build_cross_mha(4, 6, 8, 2).expect("valid");
    let attn_node = def
        .nodes
        .iter()
        .find(|n| matches!(n.kind, TensorOpKind::Attention { .. }))
        .expect("Attention exists");

    // Output of Attention is [H, T_q, head_dim] = [2, 4, 4]
    assert_eq!(attn_node.shape, vec![2, 4, 4]);
}

// ===================================================================
// add_cross_attention_transformer_block tests
// ===================================================================

/// Helper: build a cross-attention transformer block.
fn build_cross_attn_block(
    q_seq: usize,
    kv_seq: usize,
    model_dim: usize,
    num_heads: usize,
    ffn_hidden_dim: usize,
) -> Result<TensorKernelDef, TensorIRError> {
    let mut b = TensorBlockBuilder::new("cross_attn_block_test");
    let q_input = b.add_input("q_input", &[q_seq, model_dim]);
    let kv_input = b.add_input("kv_input", &[kv_seq, model_dim]);
    let eps = b.add_input("eps", &[1]);

    let ln1_w = b.add_input("ln1_weight", &[model_dim]);
    let ln1_b = b.add_input("ln1_bias", &[model_dim]);
    let ln2_w = b.add_input("ln2_weight", &[model_dim]);
    let ln2_b = b.add_input("ln2_bias", &[model_dim]);
    let ln3_w = b.add_input("ln3_weight", &[model_dim]);
    let ln3_b = b.add_input("ln3_bias", &[model_dim]);
    let ln_out_w = b.add_input("ln_out_weight", &[model_dim]);
    let ln_out_b = b.add_input("ln_out_bias", &[model_dim]);

    let q_w = b.add_input("q_weight", &[model_dim, model_dim]);
    let k_w = b.add_input("k_weight", &[model_dim, model_dim]);
    let v_w = b.add_input("v_weight", &[model_dim, model_dim]);
    let out_w = b.add_input("out_weight", &[model_dim, model_dim]);
    let ffn1_w = b.add_input("ffn1_weight", &[ffn_hidden_dim, model_dim]);
    let ffn2_w = b.add_input("ffn2_weight", &[model_dim, ffn_hidden_dim]);

    let weights = CrossAttentionBlockWeights {
        ln1_weight: ln1_w,
        ln1_bias: ln1_b,
        ln2_weight: ln2_w,
        ln2_bias: ln2_b,
        ln3_weight: ln3_w,
        ln3_bias: ln3_b,
        ln_out_weight: ln_out_w,
        ln_out_bias: ln_out_b,
        q_weight: q_w,
        k_weight: k_w,
        v_weight: v_w,
        out_weight: out_w,
        ffn1_weight: ffn1_w,
        ffn2_weight: ffn2_w,
        eps,
    };

    let config = CrossAttentionBlockConfig {
        num_heads,
        mask: AttentionMask::Standard,
        ffn_hidden_dim,
    };

    let out = b.add_cross_attention_transformer_block(q_input, kv_input, &weights, &config)?;
    b.build(out)
}

#[test]
fn cross_attn_block_builds_successfully() {
    let def = build_cross_attn_block(4, 6, 8, 2, 16).expect("valid");
    assert_eq!(def.name, "cross_attn_block_test");
    // Output shape matches Q input [T_q, D]
    assert_eq!(def.nodes[def.output.index()].shape, vec![4, 8]);
}

#[test]
fn cross_attn_block_output_follows_q_shape() {
    // Q: 3 tokens, KV: 10 tokens → output is [3, 16]
    let def = build_cross_attn_block(3, 10, 16, 4, 32).expect("valid");
    assert_eq!(def.nodes[def.output.index()].shape, vec![3, 16]);
}

#[test]
fn cross_attn_block_has_four_layer_norms() {
    let def = build_cross_attn_block(4, 6, 8, 2, 16).expect("valid");
    let ln_count = def
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, TensorOpKind::LayerNorm { .. }))
        .count();
    // LN1 (Q), LN2 (KV), LN3 (pre-FFN), LN_out
    assert_eq!(ln_count, 4, "cross-attention block needs 4 LayerNorms");
}

#[test]
fn cross_attn_block_has_two_residual_connections() {
    let def = build_cross_attn_block(4, 6, 8, 2, 16).expect("valid");
    let add_count = def
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, TensorOpKind::BinaryAdd { .. }))
        .count();
    assert_eq!(
        add_count, 2,
        "cross-attention block needs 2 residual BinaryAdds"
    );
}

#[test]
fn cross_attn_block_first_residual_wires_q_input() {
    let def = build_cross_attn_block(4, 6, 8, 2, 16).expect("valid");
    let first_add = def
        .nodes
        .iter()
        .find(|n| matches!(n.kind, TensorOpKind::BinaryAdd { .. }))
        .expect("at least one BinaryAdd");

    if let TensorOpKind::BinaryAdd { left, .. } = &first_add.kind {
        assert_eq!(
            *left,
            TensorNodeId::new(0),
            "first residual should connect to Q input (node 0)"
        );
    }
}

#[test]
fn cross_attn_block_contains_gelu() {
    let def = build_cross_attn_block(4, 6, 8, 2, 16).expect("valid");
    let gelu_count = def
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, TensorOpKind::Gelu { .. }))
        .count();
    assert_eq!(gelu_count, 1, "FFN needs exactly 1 GELU");
}

#[test]
fn cross_attn_block_output_is_layer_norm() {
    // The output should be the final LayerNorm (output normalization)
    let def = build_cross_attn_block(4, 6, 8, 2, 16).expect("valid");
    let output_node = &def.nodes[def.output.index()];
    assert!(
        matches!(output_node.kind, TensorOpKind::LayerNorm { .. }),
        "output should be the output LayerNorm, got {:?}",
        output_node.kind
    );
}

// ===================================================================
// Validation tests
// ===================================================================

#[test]
fn cross_attn_block_rejects_zero_heads() {
    let result = build_cross_attn_block(4, 6, 8, 0, 16);
    assert!(matches!(
        result.unwrap_err(),
        TensorIRError::Layer(TensorIRLayerError::TransformerZeroHeads)
    ));
}

#[test]
fn cross_attn_block_rejects_zero_ffn_dim() {
    let result = build_cross_attn_block(4, 6, 8, 2, 0);
    assert!(matches!(
        result.unwrap_err(),
        TensorIRError::Layer(TensorIRLayerError::TransformerZeroFfnDim)
    ));
}

#[test]
fn cross_attn_block_rejects_indivisible_heads() {
    let result = build_cross_attn_block(4, 6, 8, 3, 16);
    assert!(matches!(
        result.unwrap_err(),
        TensorIRError::Layer(TensorIRLayerError::MhaHeadDimNotDivisible { .. })
    ));
}

#[test]
fn cross_attn_block_validates_via_build() {
    let def = build_cross_attn_block(4, 6, 8, 2, 16).expect("valid");
    assert!(def.validate().is_ok(), "graph validation should pass");
}

#[test]
fn cross_attn_block_config_copy_and_debug() {
    let config = CrossAttentionBlockConfig {
        num_heads: 4,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: 64,
    };
    let copy = config;
    assert_eq!(copy.num_heads, 4);
    let debug_str = format!("{config:?}");
    assert!(debug_str.contains("num_heads"));
}
