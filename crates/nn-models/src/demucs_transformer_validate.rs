// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Weight validation and weight map builders for the Demucs transformer.
//!
//! Backend-agnostic — depends only on weight types and architecture constants.
//! Extracted from `nn-metal` as part of #860.

use std::collections::HashMap;

use crate::demucs_shared::validate_weight_size;
use crate::demucs_transformer_constants::{
    BOTTLENECK_DIM, FFN_HIDDEN_DIM, LAYER_NORM_EPS, TRANSFORMER_DIM,
};
use crate::demucs_transformer_weights::{
    CrossAttentionLayerWeights, DemucsTransformerWeights, LayerNormWeights,
    SelfAttentionLayerWeights, TransformerLayerWeights,
};

use super::TransformerBuildError;

// ---------------------------------------------------------------------------
// Weight validation
// ---------------------------------------------------------------------------

fn validate_weight(data: &[f32], name: &str, expected: usize) -> Result<(), TransformerBuildError> {
    Ok(validate_weight_size(data, name, expected)?)
}

fn validate_layer_norm(
    ln: &LayerNormWeights,
    prefix: &str,
    dim: usize,
) -> Result<(), TransformerBuildError> {
    validate_weight(&ln.weight, &format!("{prefix}.weight"), dim)?;
    validate_weight(&ln.bias, &format!("{prefix}.bias"), dim)?;
    Ok(())
}

/// Validate all weights for the full transformer.
pub fn validate_all_weights(
    weights: &DemucsTransformerWeights,
) -> Result<(), TransformerBuildError> {
    let d = TRANSFORMER_DIM;
    let b = BOTTLENECK_DIM;
    let ffn = FFN_HIDDEN_DIM;

    validate_weight(
        &weights.channel_upsampler_t_weight,
        "channel_upsampler_t.weight",
        d * b,
    )?;
    validate_weight(
        &weights.channel_upsampler_t_bias,
        "channel_upsampler_t.bias",
        d,
    )?;
    validate_weight(
        &weights.channel_upsampler_s_weight,
        "channel_upsampler_s.weight",
        d * b,
    )?;
    validate_weight(
        &weights.channel_upsampler_s_bias,
        "channel_upsampler_s.bias",
        d,
    )?;
    validate_weight(
        &weights.channel_downsampler_t_weight,
        "channel_downsampler_t.weight",
        b * d,
    )?;
    validate_weight(
        &weights.channel_downsampler_t_bias,
        "channel_downsampler_t.bias",
        b,
    )?;
    validate_weight(
        &weights.channel_downsampler_s_weight,
        "channel_downsampler_s.weight",
        b * d,
    )?;
    validate_weight(
        &weights.channel_downsampler_s_bias,
        "channel_downsampler_s.bias",
        b,
    )?;

    validate_layer_norm(&weights.norm_in_t, "norm_in_t", d)?;
    validate_layer_norm(&weights.norm_in_s, "norm_in_s", d)?;

    for (i, layer) in weights.temporal_layers.iter().enumerate() {
        validate_layer_weights(layer, &format!("temporal[{i}]"), d, ffn)?;
    }
    for (i, layer) in weights.spectral_layers.iter().enumerate() {
        validate_layer_weights(layer, &format!("spectral[{i}]"), d, ffn)?;
    }

    Ok(())
}

fn validate_layer_weights(
    layer: &TransformerLayerWeights,
    prefix: &str,
    d: usize,
    ffn: usize,
) -> Result<(), TransformerBuildError> {
    match layer {
        TransformerLayerWeights::SelfAttention(w) => {
            validate_layer_norm(&w.norm1, &format!("{prefix}.norm1"), d)?;
            validate_layer_norm(&w.norm2, &format!("{prefix}.norm2"), d)?;
            validate_layer_norm(&w.norm_out, &format!("{prefix}.norm_out"), d)?;
            validate_weight(&w.q_weight, &format!("{prefix}.q_weight"), d * d)?;
            validate_weight(&w.k_weight, &format!("{prefix}.k_weight"), d * d)?;
            validate_weight(&w.v_weight, &format!("{prefix}.v_weight"), d * d)?;
            validate_weight(&w.out_weight, &format!("{prefix}.out_weight"), d * d)?;
            validate_weight(
                &w.ffn_linear1_weight,
                &format!("{prefix}.ffn.linear1.weight"),
                ffn * d,
            )?;
            validate_weight(
                &w.ffn_linear1_bias,
                &format!("{prefix}.ffn.linear1.bias"),
                ffn,
            )?;
            validate_weight(
                &w.ffn_linear2_weight,
                &format!("{prefix}.ffn.linear2.weight"),
                d * ffn,
            )?;
            validate_weight(
                &w.ffn_linear2_bias,
                &format!("{prefix}.ffn.linear2.bias"),
                d,
            )?;
            validate_weight(&w.gamma_1, &format!("{prefix}.gamma_1"), d)?;
            validate_weight(&w.gamma_2, &format!("{prefix}.gamma_2"), d)?;
        }
        TransformerLayerWeights::CrossAttention(w) => {
            validate_layer_norm(&w.norm1, &format!("{prefix}.norm1"), d)?;
            validate_layer_norm(&w.norm2, &format!("{prefix}.norm2"), d)?;
            validate_layer_norm(&w.norm3, &format!("{prefix}.norm3"), d)?;
            validate_layer_norm(&w.norm_out, &format!("{prefix}.norm_out"), d)?;
            validate_weight(&w.q_weight, &format!("{prefix}.q_weight"), d * d)?;
            validate_weight(&w.k_weight, &format!("{prefix}.k_weight"), d * d)?;
            validate_weight(&w.v_weight, &format!("{prefix}.v_weight"), d * d)?;
            validate_weight(&w.out_weight, &format!("{prefix}.out_weight"), d * d)?;
            validate_weight(
                &w.ffn_linear1_weight,
                &format!("{prefix}.ffn.linear1.weight"),
                ffn * d,
            )?;
            validate_weight(
                &w.ffn_linear1_bias,
                &format!("{prefix}.ffn.linear1.bias"),
                ffn,
            )?;
            validate_weight(
                &w.ffn_linear2_weight,
                &format!("{prefix}.ffn.linear2.weight"),
                d * ffn,
            )?;
            validate_weight(
                &w.ffn_linear2_bias,
                &format!("{prefix}.ffn.linear2.bias"),
                d,
            )?;
            validate_weight(&w.gamma_1, &format!("{prefix}.gamma_1"), d)?;
            validate_weight(&w.gamma_2, &format!("{prefix}.gamma_2"), d)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Weight map builders
// ---------------------------------------------------------------------------

/// Build weight map for a self-attention transformer layer.
pub fn build_self_attention_weight_map(w: &SelfAttentionLayerWeights) -> HashMap<String, Vec<f32>> {
    let mut m = HashMap::new();
    let eps = vec![LAYER_NORM_EPS];

    m.insert("ln1_eps".to_string(), eps.clone());
    m.insert("ln1_weight".to_string(), w.norm1.weight.clone());
    m.insert("ln1_bias".to_string(), w.norm1.bias.clone());
    m.insert("ln2_eps".to_string(), eps.clone());
    m.insert("ln2_weight".to_string(), w.norm2.weight.clone());
    m.insert("ln2_bias".to_string(), w.norm2.bias.clone());
    m.insert("lnout_eps".to_string(), eps);
    m.insert("lnout_weight".to_string(), w.norm_out.weight.clone());
    m.insert("lnout_bias".to_string(), w.norm_out.bias.clone());

    m.insert("q_weight".to_string(), w.q_weight.clone());
    m.insert("k_weight".to_string(), w.k_weight.clone());
    m.insert("v_weight".to_string(), w.v_weight.clone());
    m.insert("out_weight".to_string(), w.out_weight.clone());

    m.insert(
        "ffn_linear1_weight".to_string(),
        w.ffn_linear1_weight.clone(),
    );
    m.insert("ffn_linear1_bias".to_string(), w.ffn_linear1_bias.clone());
    m.insert(
        "ffn_linear2_weight".to_string(),
        w.ffn_linear2_weight.clone(),
    );
    m.insert("ffn_linear2_bias".to_string(), w.ffn_linear2_bias.clone());

    m.insert("gamma_1".to_string(), w.gamma_1.clone());
    m.insert("gamma_2".to_string(), w.gamma_2.clone());

    m
}

/// Build weight map for a cross-attention transformer layer.
pub fn build_cross_attention_weight_map(
    w: &CrossAttentionLayerWeights,
) -> HashMap<String, Vec<f32>> {
    let mut m = HashMap::new();
    let eps = vec![LAYER_NORM_EPS];

    m.insert("ln1_eps".to_string(), eps.clone());
    m.insert("ln1_weight".to_string(), w.norm1.weight.clone());
    m.insert("ln1_bias".to_string(), w.norm1.bias.clone());
    m.insert("ln2_eps".to_string(), eps.clone());
    m.insert("ln2_weight".to_string(), w.norm2.weight.clone());
    m.insert("ln2_bias".to_string(), w.norm2.bias.clone());
    m.insert("ln3_eps".to_string(), eps.clone());
    m.insert("ln3_weight".to_string(), w.norm3.weight.clone());
    m.insert("ln3_bias".to_string(), w.norm3.bias.clone());
    m.insert("lnout_eps".to_string(), eps);
    m.insert("lnout_weight".to_string(), w.norm_out.weight.clone());
    m.insert("lnout_bias".to_string(), w.norm_out.bias.clone());

    m.insert("q_weight".to_string(), w.q_weight.clone());
    m.insert("k_weight".to_string(), w.k_weight.clone());
    m.insert("v_weight".to_string(), w.v_weight.clone());
    m.insert("out_weight".to_string(), w.out_weight.clone());

    m.insert(
        "ffn_linear1_weight".to_string(),
        w.ffn_linear1_weight.clone(),
    );
    m.insert("ffn_linear1_bias".to_string(), w.ffn_linear1_bias.clone());
    m.insert(
        "ffn_linear2_weight".to_string(),
        w.ffn_linear2_weight.clone(),
    );
    m.insert("ffn_linear2_bias".to_string(), w.ffn_linear2_bias.clone());

    m.insert("gamma_1".to_string(), w.gamma_1.clone());
    m.insert("gamma_2".to_string(), w.gamma_2.clone());

    m
}

#[cfg(test)]
#[path = "demucs_transformer_validate_tests.rs"]
mod tests;
