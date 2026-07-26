// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for transformer block composite builder (#811).

use super::*;
use crate::tensor_block_builder::TensorBlockBuilder;
use crate::tensor_ir::{
    TensorIRError, TensorIRLayerError, TensorKernelDef, TensorNodeId, TensorOpKind,
};

/// Helper: build a standard transformer block with the given dimensions.
fn build_transformer_block(
    seq_len: usize,
    model_dim: usize,
    num_heads: usize,
    ffn_hidden_dim: usize,
) -> Result<TensorKernelDef, TensorIRError> {
    let mut b = TensorBlockBuilder::new("transformer_test");
    let input = b.add_input("x", &[seq_len, model_dim]);
    let eps = b.add_input("eps", &[1]);
    let ln1_w = b.add_input("ln1_weight", &[model_dim]);
    let ln1_b = b.add_input("ln1_bias", &[model_dim]);
    let ln2_w = b.add_input("ln2_weight", &[model_dim]);
    let ln2_b = b.add_input("ln2_bias", &[model_dim]);
    let q_w = b.add_input("q_weight", &[model_dim, model_dim]);
    let k_w = b.add_input("k_weight", &[model_dim, model_dim]);
    let v_w = b.add_input("v_weight", &[model_dim, model_dim]);
    let out_w = b.add_input("out_weight", &[model_dim, model_dim]);
    let ffn1_w = b.add_input("ffn1_weight", &[ffn_hidden_dim, model_dim]);
    let ffn2_w = b.add_input("ffn2_weight", &[model_dim, ffn_hidden_dim]);

    let weights = TransformerBlockWeights {
        ln1_weight: ln1_w,
        ln1_bias: ln1_b,
        ln2_weight: ln2_w,
        ln2_bias: ln2_b,
        q_weight: q_w,
        k_weight: k_w,
        v_weight: v_w,
        out_weight: out_w,
        ffn1_weight: ffn1_w,
        ffn2_weight: ffn2_w,
        eps,
    };

    let config = TransformerBlockConfig {
        num_heads,
        mask: AttentionMask::Standard,
        ffn_hidden_dim,
    };

    let out = b.add_transformer_block(input, &weights, &config)?;
    b.build(out)
}

// ===================================================================
// AC1: add_transformer_block() decomposes correctly
// ===================================================================

#[test]
fn transformer_block_builds_successfully() {
    let def = build_transformer_block(4, 8, 2, 16).expect("valid transformer block");
    assert_eq!(def.name, "transformer_test");
    // Output shape should match input shape [T, D]
    assert_eq!(def.nodes[def.output.index()].shape, vec![4, 8]);
}

#[test]
fn transformer_block_output_shape_preserved() {
    let def = build_transformer_block(8, 16, 4, 32).expect("valid");
    assert_eq!(def.nodes[def.output.index()].shape, vec![8, 16]);
}

// ===================================================================
// AC2: TransformerBlockConfig struct fields
// ===================================================================

#[test]
fn transformer_config_copy_and_debug() {
    let config = TransformerBlockConfig {
        num_heads: 4,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: 64,
    };
    // TransformerBlockConfig derives Copy — verify both copies are valid.
    let copy1 = config;
    let copy2 = config;
    assert_eq!(copy1.num_heads, 4);
    assert_eq!(copy2.ffn_hidden_dim, 64);
    let debug_str = format!("{config:?}");
    assert!(debug_str.contains("num_heads"));
}

// ===================================================================
// AC3: Validation — model_dim % num_heads, ffn_hidden_dim, input rank
// ===================================================================

#[test]
fn transformer_block_rejects_zero_heads() {
    let result = build_transformer_block(4, 8, 0, 16);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::TransformerZeroHeads)
        ),
        "expected TransformerZeroHeads, got {err:?}"
    );
}

#[test]
fn transformer_block_rejects_zero_ffn_dim() {
    let result = build_transformer_block(4, 8, 2, 0);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::TransformerZeroFfnDim)
        ),
        "expected TransformerZeroFfnDim, got {err:?}"
    );
}

#[test]
fn transformer_block_rejects_indivisible_heads() {
    // model_dim=8, num_heads=3 → 8 % 3 != 0
    let result = build_transformer_block(4, 8, 3, 16);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::MhaHeadDimNotDivisible { .. })
        ),
        "expected MhaHeadDimNotDivisible, got {err:?}"
    );
}

#[test]
fn transformer_block_rejects_wrong_input_rank() {
    let mut b = TensorBlockBuilder::new("rank3_test");
    // Input is [B, T, D] (rank 3) instead of [T, D] (rank 2)
    let input = b.add_input("x", &[2, 4, 8]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_w", &[8]);
    let ln_b = b.add_input("ln_b", &[8]);
    let q_w = b.add_input("q_w", &[8, 8]);
    let k_w = b.add_input("k_w", &[8, 8]);
    let v_w = b.add_input("v_w", &[8, 8]);
    let out_w = b.add_input("out_w", &[8, 8]);
    let ffn1_w = b.add_input("ffn1_w", &[16, 8]);
    let ffn2_w = b.add_input("ffn2_w", &[8, 16]);

    let weights = TransformerBlockWeights {
        ln1_weight: ln_w,
        ln1_bias: ln_b,
        ln2_weight: ln_w,
        ln2_bias: ln_b,
        q_weight: q_w,
        k_weight: k_w,
        v_weight: v_w,
        out_weight: out_w,
        ffn1_weight: ffn1_w,
        ffn2_weight: ffn2_w,
        eps,
    };

    let config = TransformerBlockConfig {
        num_heads: 2,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: 16,
    };

    let result = b.add_transformer_block(input, &weights, &config);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::TransformerInputRankInvalid { rank: 3 })
        ),
        "expected TransformerInputRankInvalid, got {err:?}"
    );
}

// ===================================================================
// AC6: Residual connections correctly wired
// ===================================================================

#[test]
fn transformer_block_has_two_residual_connections() {
    let def = build_transformer_block(4, 8, 2, 16).expect("valid");
    let binary_adds: Vec<_> = def
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, TensorOpKind::BinaryAdd { .. }))
        .collect();
    assert_eq!(
        binary_adds.len(),
        2,
        "transformer block needs exactly 2 residual BinaryAdd connections"
    );
}

#[test]
fn transformer_block_first_residual_wires_input() {
    let def = build_transformer_block(4, 8, 2, 16).expect("valid");
    // Find first BinaryAdd — should have input (node 0) as one operand
    let first_add = def
        .nodes
        .iter()
        .find(|n| matches!(n.kind, TensorOpKind::BinaryAdd { .. }))
        .expect("at least one BinaryAdd");

    if let TensorOpKind::BinaryAdd { left, .. } = &first_add.kind {
        assert_eq!(
            *left,
            TensorNodeId::new(0),
            "first residual should connect to input (node 0)"
        );
    }
}

#[test]
fn transformer_block_output_is_second_residual() {
    let def = build_transformer_block(4, 8, 2, 16).expect("valid");
    let output_node = &def.nodes[def.output.index()];
    assert!(
        matches!(output_node.kind, TensorOpKind::BinaryAdd { .. }),
        "output should be the second residual BinaryAdd, got {:?}",
        output_node.kind
    );
}

// ===================================================================
// Decomposition structure checks
// ===================================================================

#[test]
fn transformer_block_contains_two_layer_norms() {
    let def = build_transformer_block(4, 8, 2, 16).expect("valid");
    let ln_count = def
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, TensorOpKind::LayerNorm { .. }))
        .count();
    assert_eq!(ln_count, 2, "pre-norm architecture needs 2 LayerNorms");
}

#[test]
fn transformer_block_contains_gelu_activation() {
    let def = build_transformer_block(4, 8, 2, 16).expect("valid");
    let gelu_count = def
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, TensorOpKind::Gelu { .. }))
        .count();
    assert_eq!(gelu_count, 1, "FFN block needs exactly 1 GELU");
}

#[test]
fn transformer_block_ffn_linear_shapes() {
    // model_dim=8, ffn_hidden_dim=16 → first Linear [T,8]→[T,16], second [T,16]→[T,8]
    let def = build_transformer_block(4, 8, 2, 16).expect("valid");
    let gelu_node = def
        .nodes
        .iter()
        .find(|n| matches!(n.kind, TensorOpKind::Gelu { .. }))
        .expect("GELU exists");
    // GELU's input should be [4, 16] (FFN intermediate)
    assert_eq!(gelu_node.shape, vec![4, 16]);
}

#[test]
fn transformer_block_causal_mask() {
    // Causal mask should also work
    let mut b = TensorBlockBuilder::new("causal_test");
    let input = b.add_input("x", &[4, 8]);
    let eps = b.add_input("eps", &[1]);
    let ln1_w = b.add_input("ln1_w", &[8]);
    let ln1_b = b.add_input("ln1_b", &[8]);
    let ln2_w = b.add_input("ln2_w", &[8]);
    let ln2_b = b.add_input("ln2_b", &[8]);
    let q_w = b.add_input("q_w", &[8, 8]);
    let k_w = b.add_input("k_w", &[8, 8]);
    let v_w = b.add_input("v_w", &[8, 8]);
    let out_w = b.add_input("out_w", &[8, 8]);
    let ffn1_w = b.add_input("ffn1_w", &[16, 8]);
    let ffn2_w = b.add_input("ffn2_w", &[8, 16]);

    let weights = TransformerBlockWeights {
        ln1_weight: ln1_w,
        ln1_bias: ln1_b,
        ln2_weight: ln2_w,
        ln2_bias: ln2_b,
        q_weight: q_w,
        k_weight: k_w,
        v_weight: v_w,
        out_weight: out_w,
        ffn1_weight: ffn1_w,
        ffn2_weight: ffn2_w,
        eps,
    };

    let config = TransformerBlockConfig {
        num_heads: 2,
        mask: AttentionMask::Causal,
        ffn_hidden_dim: 16,
    };

    let out = b
        .add_transformer_block(input, &weights, &config)
        .expect("causal should build");
    let def = b.build(out).expect("valid graph");
    assert_eq!(def.nodes[def.output.index()].shape, vec![4, 8]);
}

// ===================================================================
// AC5: Multi-block composition
// ===================================================================

/// Add one block's weights to the builder and return the `TransformerBlockWeights`.
fn add_block_weights(
    b: &mut TensorBlockBuilder,
    prefix: &str,
    model_dim: usize,
    ffn_hidden_dim: usize,
    eps: TensorNodeId,
) -> TransformerBlockWeights {
    let ln1_w = b.add_input(&format!("{prefix}_ln1_w"), &[model_dim]);
    let ln1_b = b.add_input(&format!("{prefix}_ln1_b"), &[model_dim]);
    let ln2_w = b.add_input(&format!("{prefix}_ln2_w"), &[model_dim]);
    let ln2_b = b.add_input(&format!("{prefix}_ln2_b"), &[model_dim]);
    let q_w = b.add_input(&format!("{prefix}_q_w"), &[model_dim, model_dim]);
    let k_w = b.add_input(&format!("{prefix}_k_w"), &[model_dim, model_dim]);
    let v_w = b.add_input(&format!("{prefix}_v_w"), &[model_dim, model_dim]);
    let out_w = b.add_input(&format!("{prefix}_out_w"), &[model_dim, model_dim]);
    let ffn1_w = b.add_input(&format!("{prefix}_ffn1_w"), &[ffn_hidden_dim, model_dim]);
    let ffn2_w = b.add_input(&format!("{prefix}_ffn2_w"), &[model_dim, ffn_hidden_dim]);

    TransformerBlockWeights {
        ln1_weight: ln1_w,
        ln1_bias: ln1_b,
        ln2_weight: ln2_w,
        ln2_bias: ln2_b,
        q_weight: q_w,
        k_weight: k_w,
        v_weight: v_w,
        out_weight: out_w,
        ffn1_weight: ffn1_w,
        ffn2_weight: ffn2_w,
        eps,
    }
}

#[test]
fn transformer_two_block_stack() {
    let mut b = TensorBlockBuilder::new("two_block");
    let input = b.add_input("x", &[4, 8]);
    let eps = b.add_input("eps", &[1]);

    let config = TransformerBlockConfig {
        num_heads: 2,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: 16,
    };

    let w1 = add_block_weights(&mut b, "b1", 8, 16, eps);
    let block1 = b
        .add_transformer_block(input, &w1, &config)
        .expect("block 1");

    let w2 = add_block_weights(&mut b, "b2", 8, 16, eps);
    let block2 = b
        .add_transformer_block(block1, &w2, &config)
        .expect("block 2");
    let def = b.build(block2).expect("valid two-block graph");

    assert_eq!(def.nodes[def.output.index()].shape, vec![4, 8]);

    let ln_count = def
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, TensorOpKind::LayerNorm { .. }))
        .count();
    assert_eq!(ln_count, 4, "2 blocks × 2 LayerNorms each");

    let add_count = def
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, TensorOpKind::BinaryAdd { .. }))
        .count();
    assert_eq!(add_count, 4, "2 blocks × 2 residuals each");
}

#[test]
fn transformer_block_validates_via_build() {
    // The full graph passes TensorKernelDef::validate()
    let def = build_transformer_block(4, 8, 2, 16).expect("valid");
    assert!(def.validate().is_ok(), "graph validation should pass");
}
