// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`super::demucs_transformer_validate`].

use super::*;
use crate::demucs_transformer_constants::{
    BOTTLENECK_DIM, FFN_HIDDEN_DIM, LAYER_NORM_EPS, NUM_LAYERS, TRANSFORMER_DIM,
};
use crate::demucs_transformer_weights::{
    CrossAttentionLayerWeights, DemucsTransformerWeights, LayerNormWeights,
    SelfAttentionLayerWeights, TransformerLayerWeights,
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

fn make_valid_weights() -> DemucsTransformerWeights {
    let d = TRANSFORMER_DIM;
    let b = BOTTLENECK_DIM;

    // HTDemucs alternates: self, cross, self, cross, self (5 layers)
    let mut temporal_layers = Vec::new();
    let mut spectral_layers = Vec::new();
    for i in 0..NUM_LAYERS {
        if i % 2 == 0 {
            temporal_layers.push(TransformerLayerWeights::SelfAttention(zero_self_attn()));
            spectral_layers.push(TransformerLayerWeights::SelfAttention(zero_self_attn()));
        } else {
            temporal_layers.push(TransformerLayerWeights::CrossAttention(zero_cross_attn()));
            spectral_layers.push(TransformerLayerWeights::CrossAttention(zero_cross_attn()));
        }
    }

    DemucsTransformerWeights {
        channel_upsampler_t_weight: vec![0.0; d * b],
        channel_upsampler_t_bias: vec![0.0; d],
        channel_upsampler_s_weight: vec![0.0; d * b],
        channel_upsampler_s_bias: vec![0.0; d],
        channel_downsampler_t_weight: vec![0.0; b * d],
        channel_downsampler_t_bias: vec![0.0; b],
        channel_downsampler_s_weight: vec![0.0; b * d],
        channel_downsampler_s_bias: vec![0.0; b],
        norm_in_t: zero_ln(),
        norm_in_s: zero_ln(),
        temporal_layers,
        spectral_layers,
    }
}

// -- validate_all_weights -----------------------------------------------------

#[test]
fn test_validate_all_weights_succeeds() {
    let weights = make_valid_weights();
    validate_all_weights(&weights).expect("valid weights should pass");
}

#[test]
fn test_validate_all_weights_rejects_wrong_upsampler_t() {
    let mut weights = make_valid_weights();
    weights.channel_upsampler_t_weight = vec![0.0; 1]; // wrong size
    let err = validate_all_weights(&weights).unwrap_err();
    assert!(
        matches!(err, TransformerBuildError::WeightSize { .. }),
        "err: {err}"
    );
}

#[test]
fn test_validate_all_weights_rejects_wrong_norm() {
    let mut weights = make_valid_weights();
    weights.norm_in_t.weight = vec![1.0; 1]; // wrong size
    let err = validate_all_weights(&weights).unwrap_err();
    assert!(
        matches!(err, TransformerBuildError::WeightSize { .. }),
        "err: {err}"
    );
}

#[test]
fn test_validate_all_weights_rejects_wrong_self_attn_projection() {
    let mut weights = make_valid_weights();
    if let TransformerLayerWeights::SelfAttention(ref mut sa) = weights.temporal_layers[0] {
        sa.q_weight = vec![0.0; 1]; // wrong size
    }
    let err = validate_all_weights(&weights).unwrap_err();
    assert!(
        matches!(err, TransformerBuildError::WeightSize { .. }),
        "err: {err}"
    );
}

// -- build_self_attention_weight_map ------------------------------------------

#[test]
fn test_build_self_attention_weight_map_key_count() {
    let w = zero_self_attn();
    let map = build_self_attention_weight_map(&w);

    // 9 LN keys (3 norms × 3 keys) + 4 projections + 4 FFN + 2 gammas = 19
    assert!(
        map.len() >= 19,
        "expected at least 19 keys, got {}",
        map.len()
    );
    assert!(map.contains_key("q_weight"));
    assert!(map.contains_key("ln1_eps"));
    assert!(map.contains_key("gamma_1"));
    assert!(map.contains_key("ffn_linear1_weight"));
}

#[test]
fn test_build_self_attention_weight_map_eps_value() {
    let w = zero_self_attn();
    let map = build_self_attention_weight_map(&w);
    let eps = &map["ln1_eps"];
    assert_eq!(eps.len(), 1);
    assert!((eps[0] - LAYER_NORM_EPS).abs() < 1e-10);
}

// -- build_cross_attention_weight_map -----------------------------------------

#[test]
fn test_build_cross_attention_weight_map_key_count() {
    let w = zero_cross_attn();
    let map = build_cross_attention_weight_map(&w);

    // 12 LN keys (4 norms × 3 keys) + 4 projections + 4 FFN + 2 gammas = 22
    assert!(
        map.len() >= 22,
        "expected at least 22 keys, got {}",
        map.len()
    );
    assert!(map.contains_key("ln3_weight"));
    assert!(map.contains_key("lnout_eps"));
}

#[test]
fn test_build_cross_attention_weight_map_has_all_ln3_keys() {
    let w = zero_cross_attn();
    let map = build_cross_attention_weight_map(&w);
    assert!(map.contains_key("ln3_eps"));
    assert!(map.contains_key("ln3_weight"));
    assert!(map.contains_key("ln3_bias"));
}
