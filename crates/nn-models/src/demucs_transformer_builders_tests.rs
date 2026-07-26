// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`super::demucs_transformer_builders`].

use super::*;
use crate::demucs_transformer_constants::{FFN_HIDDEN_DIM, TRANSFORMER_DIM};
use crate::demucs_transformer_weights::{
    CrossAttentionLayerWeights, LayerNormWeights, SelfAttentionLayerWeights,
    TransformerLayerWeights,
};

fn zero_ln() -> LayerNormWeights {
    LayerNormWeights {
        weight: vec![1.0; TRANSFORMER_DIM],
        bias: vec![0.0; TRANSFORMER_DIM],
    }
}

fn zero_self_attn() -> SelfAttentionLayerWeights {
    let d = TRANSFORMER_DIM;
    let ffn = FFN_HIDDEN_DIM;
    SelfAttentionLayerWeights {
        norm1: zero_ln(),
        norm2: zero_ln(),
        norm_out: zero_ln(),
        q_weight: vec![0.0; d * d],
        k_weight: vec![0.0; d * d],
        v_weight: vec![0.0; d * d],
        out_weight: vec![0.0; d * d],
        ffn_linear1_weight: vec![0.0; ffn * d],
        ffn_linear1_bias: vec![0.0; ffn],
        ffn_linear2_weight: vec![0.0; d * ffn],
        ffn_linear2_bias: vec![0.0; d],
        gamma_1: vec![1.0; d],
        gamma_2: vec![1.0; d],
    }
}

fn zero_cross_attn() -> CrossAttentionLayerWeights {
    let d = TRANSFORMER_DIM;
    let ffn = FFN_HIDDEN_DIM;
    CrossAttentionLayerWeights {
        norm1: zero_ln(),
        norm2: zero_ln(),
        norm3: zero_ln(),
        norm_out: zero_ln(),
        q_weight: vec![0.0; d * d],
        k_weight: vec![0.0; d * d],
        v_weight: vec![0.0; d * d],
        out_weight: vec![0.0; d * d],
        ffn_linear1_weight: vec![0.0; ffn * d],
        ffn_linear1_bias: vec![0.0; ffn],
        ffn_linear2_weight: vec![0.0; d * ffn],
        ffn_linear2_bias: vec![0.0; d],
        gamma_1: vec![1.0; d],
        gamma_2: vec![1.0; d],
    }
}

// -- build_channel_bridge_def -------------------------------------------------

#[test]
fn test_build_channel_bridge_def_succeeds() {
    let (def, _wmap) =
        build_channel_bridge_def("bridge_up", 384, 512, 16).expect("channel bridge should build");
    let output_shape = &def.nodes[def.output.index()].shape;
    assert_eq!(output_shape, &[512, 16]);
}

#[test]
fn test_build_channel_bridge_def_downsample() {
    let (def, _) =
        build_channel_bridge_def("bridge_down", 512, 384, 8).expect("channel bridge should build");
    let output_shape = &def.nodes[def.output.index()].shape;
    assert_eq!(output_shape, &[384, 8]);
}

// -- build_layer_norm_def -----------------------------------------------------

#[test]
fn test_build_layer_norm_def_succeeds() {
    let ln_w = zero_ln();
    let (def, wmap) =
        build_layer_norm_def("ln_test", 16, &ln_w).expect("layer norm def should build");
    let output_shape = &def.nodes[def.output.index()].shape;
    assert_eq!(output_shape, &[16, TRANSFORMER_DIM]);
    assert!(wmap.contains_key("ln_weight"));
    assert!(wmap.contains_key("eps"));
}

// -- build_self_attention_layer_def -------------------------------------------

#[test]
fn test_build_self_attention_layer_def_succeeds() {
    let weights = TransformerLayerWeights::SelfAttention(zero_self_attn());
    let (def, wmap) = build_self_attention_layer_def("self_attn_test", 16, &weights)
        .expect("self-attention layer should build");
    let output_shape = &def.nodes[def.output.index()].shape;
    assert_eq!(output_shape, &[16, TRANSFORMER_DIM]);
    // Weight map should have projections, FFN, LayerNorms, LayerScales
    assert!(wmap.contains_key("q_weight"));
    assert!(wmap.contains_key("gamma_1"));
    assert!(wmap.contains_key("ffn_linear1_weight"));
}

#[test]
fn test_build_self_attention_rejects_cross_weights() {
    let weights = TransformerLayerWeights::CrossAttention(zero_cross_attn());
    let result = build_self_attention_layer_def("bad", 16, &weights);
    assert!(
        result.is_err(),
        "self-attention should reject cross-attention weights"
    );
}

// -- build_cross_attention_layer_def ------------------------------------------

#[test]
fn test_build_cross_attention_layer_def_succeeds() {
    let weights = TransformerLayerWeights::CrossAttention(zero_cross_attn());
    let (def, wmap) = build_cross_attention_layer_def("cross_attn_test", 16, 32, &weights)
        .expect("cross-attention layer should build");
    let output_shape = &def.nodes[def.output.index()].shape;
    // Output uses q_seq_len
    assert_eq!(output_shape, &[16, TRANSFORMER_DIM]);
    assert!(wmap.contains_key("q_weight"));
}

#[test]
fn test_build_cross_attention_rejects_self_weights() {
    let weights = TransformerLayerWeights::SelfAttention(zero_self_attn());
    let result = build_cross_attention_layer_def("bad", 16, 32, &weights);
    assert!(
        result.is_err(),
        "cross-attention should reject self-attention weights"
    );
}

#[test]
fn test_build_cross_attention_asymmetric_seq_lens() {
    let weights = TransformerLayerWeights::CrossAttention(zero_cross_attn());
    // q_seq=8, kv_seq=64: temporal attends to spectral (different lengths)
    let (def, _) = build_cross_attention_layer_def("cross_asym", 8, 64, &weights)
        .expect("asymmetric cross-attention should build");
    let output_shape = &def.nodes[def.output.index()].shape;
    assert_eq!(output_shape[0], 8, "output uses q_seq_len");
}
