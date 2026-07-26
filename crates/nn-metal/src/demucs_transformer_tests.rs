// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for DemucsTransformer.
//!
//! Tests cover: construction validation, weight validation, channel bridge
//! builder correctness, transformer layer def building, helper functions.
//!
//! GPU/CPU forward parity tests extracted to
//! `demucs_transformer_gpu_tests.rs` (#1420).
//!
//! Part of #779 — Phase D.

use super::builders;
use super::helpers::{add_sinusoidal_1d, transpose_ct_to_tc, transpose_tc_to_ct};
use super::*;
use crate::demucs_test_common::{
    make_cross_attention_weights, make_layer_norm_weights, make_self_attention_weights,
    make_transformer_weights,
};

// ---------------------------------------------------------------------------
// Construction tests
// ---------------------------------------------------------------------------

#[test]
fn test_construction_valid_weights() {
    let weights = make_transformer_weights();
    let result = DemucsTransformer::new(weights, 16, 32);
    assert!(
        result.is_ok(),
        "construction failed: {}",
        result.unwrap_err()
    );
    let model = result.unwrap();
    assert_eq!(model.temporal_layer_defs.len(), NUM_LAYERS);
    assert_eq!(model.spectral_layer_defs.len(), NUM_LAYERS);
    assert_eq!(model.temporal_seq_len, 16);
    assert_eq!(model.spectral_seq_len, 32);
}

#[test]
fn test_construction_wrong_temporal_layer_count() {
    let mut weights = make_transformer_weights();
    weights.temporal_layers.pop(); // remove last layer
    let result = DemucsTransformer::new(weights, 16, 32);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        format!("{err}").contains("temporal_layers.len"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_construction_wrong_spectral_layer_count() {
    let mut weights = make_transformer_weights();
    weights.spectral_layers.pop();
    let result = DemucsTransformer::new(weights, 16, 32);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        format!("{err}").contains("spectral_layers.len"),
        "unexpected error: {err}"
    );
}

// ---------------------------------------------------------------------------
// Weight validation tests
// ---------------------------------------------------------------------------

#[test]
fn test_validate_wrong_upsampler_weight_size() {
    let mut weights = make_transformer_weights();
    weights.channel_upsampler_t_weight = vec![0.01; 100]; // wrong size
    let result = DemucsTransformer::new(weights, 16, 32);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        format!("{err}").contains("channel_upsampler_t.weight"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_validate_wrong_layer_norm_size() {
    let mut weights = make_transformer_weights();
    weights.norm_in_t.weight = vec![1.0; 100]; // wrong dim
    let result = DemucsTransformer::new(weights, 16, 32);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        format!("{err}").contains("norm_in_t.weight"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_validate_wrong_mha_weight_size() {
    let mut weights = make_transformer_weights();
    if let TransformerLayerWeights::SelfAttention(ref mut sa) = weights.temporal_layers[0] {
        sa.q_weight = vec![0.01; 100]; // wrong size
    }
    let result = DemucsTransformer::new(weights, 16, 32);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        format!("{err}").contains("q_weight"),
        "unexpected error: {err}"
    );
}

// ---------------------------------------------------------------------------
// Channel bridge builder tests
// ---------------------------------------------------------------------------

#[test]
fn test_channel_bridge_def_shape() {
    let (def, _wmap) =
        builders::build_channel_bridge_def("test_bridge", BOTTLENECK_DIM, TRANSFORMER_DIM, 16)
            .expect("bridge build failed");

    // Output shape should be [TRANSFORMER_DIM, 16].
    let output_node = &def.nodes[def.nodes.len() - 1];
    assert_eq!(output_node.shape, vec![TRANSFORMER_DIM, 16]);
}

#[test]
fn test_channel_bridge_def_node_count() {
    let (def, _) = builders::build_channel_bridge_def("test", BOTTLENECK_DIM, TRANSFORMER_DIM, 8)
        .expect("bridge build failed");

    // 3 inputs (data, weight, bias) + 1 conv1d = 4 nodes.
    assert_eq!(def.nodes.len(), 4);
}

// ---------------------------------------------------------------------------
// LayerNorm builder tests
// ---------------------------------------------------------------------------

#[test]
fn test_layer_norm_def_shape() {
    let ln = make_layer_norm_weights(TRANSFORMER_DIM);
    let (def, wmap) =
        builders::build_layer_norm_def("test_ln", 16, &ln).expect("layer norm build failed");

    let output = &def.nodes[def.nodes.len() - 1];
    assert_eq!(output.shape, vec![16, TRANSFORMER_DIM]);

    assert!(wmap.contains_key("ln_weight"));
    assert!(wmap.contains_key("ln_bias"));
    assert!(wmap.contains_key("eps"));
}

// ---------------------------------------------------------------------------
// Self-attention layer builder tests
// ---------------------------------------------------------------------------

#[test]
fn test_self_attention_layer_def_builds() {
    let sa = make_self_attention_weights();
    let layer = TransformerLayerWeights::SelfAttention(sa);
    let result = builders::build_self_attention_layer_def("test_sa", 8, &layer);
    assert!(
        result.is_ok(),
        "self-attn build failed: {}",
        result.unwrap_err()
    );

    let (def, wmap) = result.unwrap();
    let output = &def.nodes[def.nodes.len() - 1];
    assert_eq!(output.shape, vec![8, TRANSFORMER_DIM]);

    // Check essential weight keys exist.
    assert!(wmap.contains_key("q_weight"));
    assert!(wmap.contains_key("k_weight"));
    assert!(wmap.contains_key("v_weight"));
    assert!(wmap.contains_key("out_weight"));
    assert!(wmap.contains_key("gamma_1"));
    assert!(wmap.contains_key("gamma_2"));
    assert!(wmap.contains_key("ffn_linear1_weight"));
}

#[test]
fn test_self_attention_layer_weight_map_completeness() {
    let sa = make_self_attention_weights();
    let layer = TransformerLayerWeights::SelfAttention(sa);
    let (_def, wmap) = builders::build_self_attention_layer_def("test", 4, &layer).unwrap();

    // 9 LayerNorm (3 norms × 3 params: eps, weight, bias) + 4 MHA + 4 FFN + 2 gamma = 19 total.
    // Actually: 3 eps + 3 weights + 3 biases = 9 LN params + 4 MHA + 4 FFN + 2 gamma = 19.
    assert_eq!(
        wmap.len(),
        19,
        "expected 19 weight entries, got {}",
        wmap.len()
    );
}

// ---------------------------------------------------------------------------
// Cross-attention layer builder tests
// ---------------------------------------------------------------------------

#[test]
fn test_cross_attention_layer_def_builds() {
    let ca = make_cross_attention_weights();
    let layer = TransformerLayerWeights::CrossAttention(ca);
    let result = builders::build_cross_attention_layer_def("test_ca", 8, 12, &layer);
    assert!(
        result.is_ok(),
        "cross-attn build failed: {}",
        result.unwrap_err()
    );

    let (def, wmap) = result.unwrap();
    let output = &def.nodes[def.nodes.len() - 1];
    assert_eq!(output.shape, vec![8, TRANSFORMER_DIM]);

    assert!(wmap.contains_key("q_weight"));
    assert!(wmap.contains_key("ln3_weight")); // cross-attention has extra norm3
}

#[test]
fn test_cross_attention_layer_weight_map_completeness() {
    let ca = make_cross_attention_weights();
    let layer = TransformerLayerWeights::CrossAttention(ca);
    let (_def, wmap) = builders::build_cross_attention_layer_def("test", 4, 6, &layer).unwrap();

    // 12 LayerNorm (4 norms × 3 params) + 4 MHA + 4 FFN + 2 gamma = 22 total.
    assert_eq!(
        wmap.len(),
        22,
        "expected 22 weight entries, got {}",
        wmap.len()
    );
}

#[test]
fn test_cross_attention_asymmetric_seq_lengths() {
    // Q from temporal (short), KV from spectral (long).
    let ca = make_cross_attention_weights();
    let layer = TransformerLayerWeights::CrossAttention(ca);
    let result = builders::build_cross_attention_layer_def("test_asym", 4, 16, &layer);
    assert!(
        result.is_ok(),
        "asymmetric cross-attn failed: {}",
        result.unwrap_err()
    );

    let (def, _) = result.unwrap();
    let output = &def.nodes[def.nodes.len() - 1];
    // Output shape matches Q sequence length.
    assert_eq!(output.shape, vec![4, TRANSFORMER_DIM]);
}

// ---------------------------------------------------------------------------
// Helper function tests
// ---------------------------------------------------------------------------

#[test]
fn test_transpose_ct_to_tc_roundtrip() {
    let channels = 3;
    let seq_len = 4;
    let original: Vec<f32> = (0..12).map(|x| x as f32).collect();

    let tc = transpose_ct_to_tc(&original, channels, seq_len);
    let ct = transpose_tc_to_ct(&tc, seq_len, channels);

    assert_eq!(ct, original);
}

#[test]
fn test_transpose_ct_to_tc_values() {
    // [C=2, T=3]: [[0,1,2],[3,4,5]] → [T=3, C=2]: [[0,3],[1,4],[2,5]]
    let ct = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let tc = transpose_ct_to_tc(&ct, 2, 3);
    assert_eq!(tc, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
}

#[test]
fn test_sinusoidal_1d_shape_preserved() {
    let seq = 8;
    let dim = 16;
    let mut data = vec![0.0f32; seq * dim];
    add_sinusoidal_1d(&mut data, seq, dim);
    assert_eq!(data.len(), seq * dim);

    // Position 0 should have cos(0)=1 for all frequencies.
    assert!(
        (data[0] - 1.0).abs() < 1e-5,
        "cos(0) should be ~1.0, got {}",
        data[0]
    );
}

#[test]
fn test_sinusoidal_1d_different_positions_differ() {
    let seq = 4;
    let dim = 8;
    let mut data = vec![0.0f32; seq * dim];
    add_sinusoidal_1d(&mut data, seq, dim);

    // Position 0 and position 1 should produce different embeddings.
    let row0: Vec<f32> = data[0..dim].to_vec();
    let row1: Vec<f32> = data[dim..2 * dim].to_vec();
    assert_ne!(
        row0, row1,
        "different positions should have different embeddings"
    );
}

// ---------------------------------------------------------------------------
// Debug formatting test
// ---------------------------------------------------------------------------

#[test]
fn test_debug_format() {
    let weights = make_transformer_weights();
    let model = DemucsTransformer::new(weights, 16, 32).unwrap();
    let debug = format!("{model:?}");
    assert!(debug.contains("DemucsTransformer"));
    assert!(debug.contains("temporal_seq_len: 16"));
    assert!(debug.contains("spectral_seq_len: 32"));
}

// GPU/CPU forward parity tests extracted to
// demucs_transformer_gpu_tests.rs (#1420).
#[path = "demucs_transformer_gpu_tests.rs"]
mod gpu_tests;
