// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared Demucs weight factory functions for nn-metal test files.
//!
//! Consolidates `make_dconv_weights()`, `make_encoder_block_weights()`,
//! `make_encoder_weights()`, `make_layer_norm_weights()`,
//! `make_self_attention_weights()`, `make_cross_attention_weights()`,
//! `make_transformer_weights()`, `make_decoder_block_weights()`,
//! `make_decoder_weights()`, and `make_htdemucs_weights()` which were
//! previously duplicated across htdemucs_tests.rs,
//! demucs_temporal_encoder_tests.rs, and demucs_transformer_tests.rs.
//!
//! Usage from any `#[cfg(test)]` submodule:
//! ```ignore
//! // NOTE: ignore — uses crate-internal paths only valid within nn-metal
//! use crate::demucs_test_common::{make_htdemucs_weights, make_encoder_weights};
//! ```
//!
//! Part of #1204: Metal test helper consolidation.

use crate::demucs_shared::{
    channels_at_depth, AUDIO_CHANNELS, DCONV_COMPRESS, DCONV_DEPTH, DCONV_KERNEL,
    DECODER_OUTPUT_CHANNELS,
};
use crate::demucs_temporal_decoder::{DecoderBlockWeights, DemucsTemporalDecoderWeights};
use crate::demucs_temporal_encoder::{DemucsTemporalEncoderWeights, EncoderBlockWeights};
use crate::demucs_transformer::{
    CrossAttentionLayerWeights, DemucsTransformerWeights, LayerNormWeights,
    SelfAttentionLayerWeights, TransformerLayerWeights,
};
use crate::htdemucs::HTDemucsWeights;
use nn_models::demucs_temporal_weights::DConvSubLayerWeights;
use nn_models::demucs_transformer_constants::{
    BOTTLENECK_DIM, FFN_HIDDEN_DIM, NUM_LAYERS, TRANSFORMER_DIM,
};

// ---------------------------------------------------------------------------
// Constants matching the production config
// ---------------------------------------------------------------------------

pub(crate) const DEPTH: usize = 4;
pub(crate) const KERNEL_SIZE: usize = 8;
pub(crate) const REWRITE_KERNEL: usize = 3;

// ---------------------------------------------------------------------------
// Temporal encoder weight factories
// ---------------------------------------------------------------------------

/// Create minimal valid DConv sub-layer weights for a given channel count.
pub(crate) fn make_dconv_weights(channels: usize) -> DConvSubLayerWeights {
    let compressed = channels / DCONV_COMPRESS;
    let doubled = channels * 2;
    DConvSubLayerWeights {
        conv_compress_weight: vec![0.01; compressed * channels * DCONV_KERNEL],
        conv_compress_bias: vec![0.0; compressed],
        norm_compress_gamma: vec![1.0; compressed],
        norm_compress_beta: vec![0.0; compressed],
        conv_expand_weight: vec![0.01; doubled * compressed],
        conv_expand_bias: vec![0.0; doubled],
        norm_expand_gamma: vec![1.0; doubled],
        norm_expand_beta: vec![0.0; doubled],
        layer_scale: vec![0.1; channels],
    }
}

/// Create minimal valid encoder block weights.
pub(crate) fn make_encoder_block_weights(in_ch: usize, out_ch: usize) -> EncoderBlockWeights {
    let doubled = out_ch * 2;
    EncoderBlockWeights {
        conv_weight: vec![0.01; out_ch * in_ch * KERNEL_SIZE],
        conv_bias: vec![0.0; out_ch],
        dconv: (0..DCONV_DEPTH)
            .map(|_| make_dconv_weights(out_ch))
            .collect(),
        rewrite_weight: vec![0.01; doubled * out_ch],
        rewrite_bias: vec![0.0; doubled],
    }
}

/// Create valid weights for the full 4-block temporal encoder.
pub(crate) fn make_encoder_weights() -> DemucsTemporalEncoderWeights {
    DemucsTemporalEncoderWeights {
        blocks: (0..DEPTH)
            .map(|d| {
                let in_ch = if d == 0 {
                    AUDIO_CHANNELS
                } else {
                    channels_at_depth(d - 1)
                };
                let out_ch = channels_at_depth(d);
                make_encoder_block_weights(in_ch, out_ch)
            })
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// Transformer weight factories
// ---------------------------------------------------------------------------

/// Create minimal valid LayerNorm weights.
pub(crate) fn make_layer_norm_weights(dim: usize) -> LayerNormWeights {
    LayerNormWeights {
        weight: vec![1.0; dim],
        bias: vec![0.0; dim],
    }
}

/// Create minimal valid self-attention layer weights.
///
/// Weights use distinct magnitudes per layer type so the test exercises
/// meaningful (non-degenerate) computation through each sublayer.
pub(crate) fn make_self_attention_weights() -> SelfAttentionLayerWeights {
    let d = TRANSFORMER_DIM;
    let ffn = FFN_HIDDEN_DIM;
    SelfAttentionLayerWeights {
        norm1: make_layer_norm_weights(d),
        norm2: make_layer_norm_weights(d),
        norm_out: make_layer_norm_weights(d),
        q_weight: vec![0.02; d * d],
        k_weight: vec![0.015; d * d],
        v_weight: vec![0.025; d * d],
        out_weight: vec![0.012; d * d],
        ffn_linear1_weight: vec![0.005; ffn * d],
        ffn_linear1_bias: vec![0.001; ffn],
        ffn_linear2_weight: vec![0.008; d * ffn],
        ffn_linear2_bias: vec![0.001; d],
        gamma_1: vec![0.01; d],
        gamma_2: vec![0.01; d],
    }
}

/// Create minimal valid cross-attention layer weights.
///
/// Uses distinct magnitudes from self-attention so alternating layers
/// produce meaningfully different outputs.
pub(crate) fn make_cross_attention_weights() -> CrossAttentionLayerWeights {
    let d = TRANSFORMER_DIM;
    let ffn = FFN_HIDDEN_DIM;
    CrossAttentionLayerWeights {
        norm1: make_layer_norm_weights(d),
        norm2: make_layer_norm_weights(d),
        norm3: make_layer_norm_weights(d),
        norm_out: make_layer_norm_weights(d),
        q_weight: vec![0.018; d * d],
        k_weight: vec![0.022; d * d],
        v_weight: vec![0.02; d * d],
        out_weight: vec![0.014; d * d],
        ffn_linear1_weight: vec![0.006; ffn * d],
        ffn_linear1_bias: vec![0.002; ffn],
        ffn_linear2_weight: vec![0.007; d * ffn],
        ffn_linear2_bias: vec![0.002; d],
        gamma_1: vec![0.012; d],
        gamma_2: vec![0.012; d],
    }
}

/// Create minimal valid transformer weights (temporal + spectral layers).
pub(crate) fn make_transformer_weights() -> DemucsTransformerWeights {
    let d = TRANSFORMER_DIM;
    let b = BOTTLENECK_DIM;

    let make_layers = || -> Vec<TransformerLayerWeights> {
        (0..NUM_LAYERS)
            .map(|i| {
                if i % 2 == 0 {
                    TransformerLayerWeights::SelfAttention(make_self_attention_weights())
                } else {
                    TransformerLayerWeights::CrossAttention(make_cross_attention_weights())
                }
            })
            .collect()
    };

    DemucsTransformerWeights {
        channel_upsampler_t_weight: vec![0.01; d * b],
        channel_upsampler_t_bias: vec![0.001; d],
        channel_upsampler_s_weight: vec![0.012; d * b],
        channel_upsampler_s_bias: vec![0.001; d],
        norm_in_t: make_layer_norm_weights(d),
        norm_in_s: make_layer_norm_weights(d),
        temporal_layers: make_layers(),
        spectral_layers: make_layers(),
        channel_downsampler_t_weight: vec![0.008; b * d],
        channel_downsampler_t_bias: vec![0.001; b],
        channel_downsampler_s_weight: vec![0.009; b * d],
        channel_downsampler_s_bias: vec![0.001; b],
    }
}

// ---------------------------------------------------------------------------
// Temporal decoder weight factories
// ---------------------------------------------------------------------------

/// Create minimal valid decoder block weights.
pub(crate) fn make_decoder_block_weights(block_idx: usize) -> DecoderBlockWeights {
    let encoder_depth = DEPTH - 1 - block_idx;
    let in_ch = channels_at_depth(encoder_depth);
    let out_ch = if encoder_depth == 0 {
        DECODER_OUTPUT_CHANNELS
    } else {
        channels_at_depth(encoder_depth - 1)
    };
    DecoderBlockWeights {
        rewrite_weight: vec![0.01; in_ch * 2 * in_ch * REWRITE_KERNEL],
        rewrite_bias: vec![0.0; in_ch * 2],
        dconv: (0..DCONV_DEPTH)
            .map(|_| make_dconv_weights(in_ch))
            .collect(),
        conv_tr_weight: vec![0.01; in_ch * out_ch * KERNEL_SIZE],
        conv_tr_bias: vec![0.0; out_ch],
    }
}

/// Create valid weights for the full 4-block temporal decoder.
pub(crate) fn make_decoder_weights() -> DemucsTemporalDecoderWeights {
    DemucsTemporalDecoderWeights {
        blocks: (0..DEPTH).map(make_decoder_block_weights).collect(),
    }
}

// ---------------------------------------------------------------------------
// HTDemucs composite weight factory
// ---------------------------------------------------------------------------

/// Create valid weights for the full HTDemucs model (temporal only, no spectral).
///
/// Weights use small constant values (0.01 for weights, 1.0 for norm gamma, 0.0 for biases)
/// and are suitable for contract tests (shape and finiteness validation) but not parity tests.
pub(crate) fn make_htdemucs_weights() -> HTDemucsWeights {
    HTDemucsWeights {
        encoder: make_encoder_weights(),
        transformer: make_transformer_weights(),
        decoder: make_decoder_weights(),
        spectral_encoder: None,
        spectral_decoder: None,
    }
}
