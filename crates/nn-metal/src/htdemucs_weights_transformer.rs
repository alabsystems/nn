// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Transformer weight loading for HTDemucs.
//!
//! Extracted from `htdemucs_weights.rs` for 500-line compliance.

use crate::demucs_transformer::{
    CrossAttentionLayerWeights, DemucsTransformerWeights, LayerNormWeights,
    SelfAttentionLayerWeights, TransformerLayerWeights,
};

use super::{extract, load_norm, TensorSource, WeightLoadError};
use super::{FFN_DIM, NUM_LAYERS, TRANSFORMER_DIM};

pub(super) fn load_self_attn_layer(
    src: &impl TensorSource,
    prefix: &str,
) -> Result<SelfAttentionLayerWeights, WeightLoadError> {
    Ok(SelfAttentionLayerWeights {
        norm1: load_norm(src, &format!("{prefix}_norm1"))?,
        norm2: load_norm(src, &format!("{prefix}_norm2"))?,
        norm_out: load_norm(src, &format!("{prefix}_norm_out"))?,
        q_weight: extract(src, &format!("{prefix}_q_weight"))?,
        k_weight: extract(src, &format!("{prefix}_k_weight"))?,
        v_weight: extract(src, &format!("{prefix}_v_weight"))?,
        out_weight: extract(src, &format!("{prefix}_out_weight"))?,
        ffn_linear1_weight: extract(src, &format!("{prefix}_ffn1_weight"))?,
        ffn_linear1_bias: extract(src, &format!("{prefix}_ffn1_bias"))?,
        ffn_linear2_weight: extract(src, &format!("{prefix}_ffn2_weight"))?,
        ffn_linear2_bias: extract(src, &format!("{prefix}_ffn2_bias"))?,
        gamma_1: extract(src, &format!("{prefix}_gamma1"))?,
        gamma_2: extract(src, &format!("{prefix}_gamma2"))?,
    })
}

pub(super) fn load_cross_attn_layer(
    src: &impl TensorSource,
    prefix: &str,
) -> Result<CrossAttentionLayerWeights, WeightLoadError> {
    Ok(CrossAttentionLayerWeights {
        norm1: load_norm(src, &format!("{prefix}_norm1"))?,
        norm2: load_norm(src, &format!("{prefix}_norm2"))?,
        norm3: load_norm(src, &format!("{prefix}_norm3"))?,
        norm_out: load_norm(src, &format!("{prefix}_norm_out"))?,
        q_weight: extract(src, &format!("{prefix}_q_weight"))?,
        k_weight: extract(src, &format!("{prefix}_k_weight"))?,
        v_weight: extract(src, &format!("{prefix}_v_weight"))?,
        out_weight: extract(src, &format!("{prefix}_out_weight"))?,
        ffn_linear1_weight: extract(src, &format!("{prefix}_ffn1_weight"))?,
        ffn_linear1_bias: extract(src, &format!("{prefix}_ffn1_bias"))?,
        ffn_linear2_weight: extract(src, &format!("{prefix}_ffn2_weight"))?,
        ffn_linear2_bias: extract(src, &format!("{prefix}_ffn2_bias"))?,
        gamma_1: extract(src, &format!("{prefix}_gamma1"))?,
        gamma_2: extract(src, &format!("{prefix}_gamma2"))?,
    })
}

pub(super) fn load_transformer(
    src: &impl TensorSource,
) -> Result<DemucsTransformerWeights, WeightLoadError> {
    let mut temporal_layers = Vec::with_capacity(NUM_LAYERS);
    for i in 0..NUM_LAYERS {
        let is_cross = i % 2 == 1;
        if is_cross {
            let prefix = format!("xformer_cross{i}");
            temporal_layers.push(TransformerLayerWeights::CrossAttention(
                load_cross_attn_layer(src, &prefix)?,
            ));
        } else {
            let prefix = format!("xformer_self{i}");
            temporal_layers.push(TransformerLayerWeights::SelfAttention(
                load_self_attn_layer(src, &prefix)?,
            ));
        }
    }

    // Spectral layers use dummy zero weights (temporal-only mode).
    // The transformer constructor validates weight sizes, so we provide
    // correctly-sized zero vectors.
    let spectral_layers = build_zero_spectral_layers();

    Ok(DemucsTransformerWeights {
        channel_upsampler_t_weight: extract(src, "xformer_upsample_weight")?,
        channel_upsampler_t_bias: extract(src, "xformer_upsample_bias")?,
        channel_upsampler_s_weight: vec![0.0; TRANSFORMER_DIM * BOTTLENECK_DIM],
        channel_upsampler_s_bias: vec![0.0; TRANSFORMER_DIM],
        norm_in_t: load_norm(src, "xformer_norm_in")?,
        norm_in_s: LayerNormWeights {
            weight: vec![1.0; TRANSFORMER_DIM],
            bias: vec![0.0; TRANSFORMER_DIM],
        },
        temporal_layers,
        spectral_layers,
        channel_downsampler_t_weight: extract(src, "xformer_downsample_weight")?,
        channel_downsampler_t_bias: extract(src, "xformer_downsample_bias")?,
        channel_downsampler_s_weight: vec![0.0; BOTTLENECK_DIM * TRANSFORMER_DIM],
        channel_downsampler_s_bias: vec![0.0; BOTTLENECK_DIM],
    })
}

use super::BOTTLENECK_DIM;

/// Build zero-weight spectral layers for temporal-only mode.
pub(super) fn build_zero_spectral_layers() -> Vec<TransformerLayerWeights> {
    let d = TRANSFORMER_DIM;
    let ffn = FFN_DIM;

    (0..NUM_LAYERS)
        .map(|i| {
            let is_cross = i % 2 == 1;
            let zero_norm = || LayerNormWeights {
                weight: vec![1.0; d],
                bias: vec![0.0; d],
            };
            if is_cross {
                TransformerLayerWeights::CrossAttention(CrossAttentionLayerWeights {
                    norm1: zero_norm(),
                    norm2: zero_norm(),
                    norm3: zero_norm(),
                    norm_out: zero_norm(),
                    q_weight: vec![0.0; d * d],
                    k_weight: vec![0.0; d * d],
                    v_weight: vec![0.0; d * d],
                    out_weight: vec![0.0; d * d],
                    ffn_linear1_weight: vec![0.0; ffn * d],
                    ffn_linear1_bias: vec![0.0; ffn],
                    ffn_linear2_weight: vec![0.0; d * ffn],
                    ffn_linear2_bias: vec![0.0; d],
                    gamma_1: vec![0.0; d],
                    gamma_2: vec![0.0; d],
                })
            } else {
                TransformerLayerWeights::SelfAttention(SelfAttentionLayerWeights {
                    norm1: zero_norm(),
                    norm2: zero_norm(),
                    norm_out: zero_norm(),
                    q_weight: vec![0.0; d * d],
                    k_weight: vec![0.0; d * d],
                    v_weight: vec![0.0; d * d],
                    out_weight: vec![0.0; d * d],
                    ffn_linear1_weight: vec![0.0; ffn * d],
                    ffn_linear1_bias: vec![0.0; ffn],
                    ffn_linear2_weight: vec![0.0; d * ffn],
                    ffn_linear2_bias: vec![0.0; d],
                    gamma_1: vec![0.0; d],
                    gamma_2: vec![0.0; d],
                })
            }
        })
        .collect()
}
