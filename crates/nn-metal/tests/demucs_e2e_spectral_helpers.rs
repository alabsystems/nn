// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Weight factory helpers for HTDemucs spectral contract tests.
//!
//! Extracted from `demucs_e2e_spectral_contract.rs` for the 500-line limit.

#![allow(dead_code, unreachable_pub)]

use nn_metal::HTDemucsWeights;
use nn_models::demucs_spectral_weights::{
    DemucsSpectralDecoderWeights, DemucsSpectralEncoderWeights, SpectralDConvSubLayerWeights,
    SpectralDecoderBlockWeights, SpectralEncDConvSubLayerWeights, SpectralEncoderBlockWeights,
};
use nn_models::demucs_temporal_weights::{
    DConvSubLayerWeights, DecoderBlockWeights, DemucsTemporalDecoderWeights,
    DemucsTemporalEncoderWeights, EncoderBlockWeights,
};
use nn_models::demucs_transformer_weights::{
    CrossAttentionLayerWeights, DemucsTransformerWeights, LayerNormWeights,
    SelfAttentionLayerWeights, TransformerLayerWeights,
};

// Constants duplicated from the parent test file to keep this module
// self-contained (cargo compiles tests/*.rs as standalone binaries).
const AUDIO_CHANNELS: usize = 2;
const DEPTH: usize = 4;
const INITIAL_CHANNELS: usize = 48;
const KERNEL_SIZE: usize = 8;
const REWRITE_KERNEL: usize = 3;
const DCONV_DEPTH: usize = 2;
const DCONV_COMPRESS: usize = 8;
const DCONV_KERNEL: usize = 3;
const DECODER_OUTPUT_CHANNELS: usize = 8;
const TRANSFORMER_DIM: usize = 512;
const BOTTLENECK_DIM: usize = 384;
const FFN_HIDDEN_DIM: usize = 2048;
const NUM_LAYERS: usize = 5;
const SPECTRAL_INPUT_CHANNELS: usize = 4;
const SPECTRAL_OUTPUT_CHANNELS: usize = 16;

fn channels_at_depth(depth: usize) -> usize {
    INITIAL_CHANNELS * (1 << depth)
}

// ---------------------------------------------------------------------------
// Temporal weight factories
// ---------------------------------------------------------------------------

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

fn make_layer_norm_weights(dim: usize) -> LayerNormWeights {
    LayerNormWeights {
        weight: vec![1.0; dim],
        bias: vec![0.0; dim],
    }
}

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

pub(crate) fn make_decoder_weights() -> DemucsTemporalDecoderWeights {
    DemucsTemporalDecoderWeights {
        blocks: (0..DEPTH).map(make_decoder_block_weights).collect(),
    }
}

// ---------------------------------------------------------------------------
// Spectral weight factories
// ---------------------------------------------------------------------------

fn make_spectral_enc_dconv_weights(channels: usize) -> SpectralEncDConvSubLayerWeights {
    let compressed = channels / DCONV_COMPRESS;
    let doubled = channels * 2;
    SpectralEncDConvSubLayerWeights {
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

fn make_spectral_encoder_block_weights(in_ch: usize, out_ch: usize) -> SpectralEncoderBlockWeights {
    let doubled = out_ch * 2;
    SpectralEncoderBlockWeights {
        conv_weight: vec![0.01; out_ch * in_ch * KERNEL_SIZE],
        conv_bias: vec![0.0; out_ch],
        dconv: (0..DCONV_DEPTH)
            .map(|_| make_spectral_enc_dconv_weights(out_ch))
            .collect(),
        rewrite_weight: vec![0.01; doubled * out_ch],
        rewrite_bias: vec![0.0; doubled],
    }
}

fn make_spectral_encoder_weights() -> DemucsSpectralEncoderWeights {
    DemucsSpectralEncoderWeights {
        blocks: (0..DEPTH)
            .map(|d| {
                let in_ch = if d == 0 {
                    SPECTRAL_INPUT_CHANNELS
                } else {
                    channels_at_depth(d - 1)
                };
                let out_ch = channels_at_depth(d);
                make_spectral_encoder_block_weights(in_ch, out_ch)
            })
            .collect(),
        freq_emb_weight: None,
    }
}

fn make_spectral_dec_dconv_weights(channels: usize) -> SpectralDConvSubLayerWeights {
    let compressed = channels / DCONV_COMPRESS;
    let doubled = channels * 2;
    SpectralDConvSubLayerWeights {
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

fn make_spectral_decoder_block_weights(block_idx: usize) -> SpectralDecoderBlockWeights {
    let encoder_depth = DEPTH - 1 - block_idx;
    let in_ch = channels_at_depth(encoder_depth);
    let out_ch = if encoder_depth == 0 {
        SPECTRAL_OUTPUT_CHANNELS
    } else {
        channels_at_depth(encoder_depth - 1)
    };
    SpectralDecoderBlockWeights {
        // Conv2d rewrite: [in_ch*2, in_ch, 3, 3]
        rewrite_weight: vec![0.005; in_ch * 2 * in_ch * 9],
        rewrite_bias: vec![0.0; in_ch * 2],
        dconv: (0..DCONV_DEPTH)
            .map(|_| make_spectral_dec_dconv_weights(in_ch))
            .collect(),
        // ConvTranspose1d: [in_ch, out_ch, KERNEL_SIZE]
        conv_tr_weight: vec![0.01; in_ch * out_ch * KERNEL_SIZE],
        conv_tr_bias: vec![0.0; out_ch],
    }
}

fn make_spectral_decoder_weights() -> DemucsSpectralDecoderWeights {
    DemucsSpectralDecoderWeights {
        blocks: (0..DEPTH)
            .map(make_spectral_decoder_block_weights)
            .collect(),
    }
}

/// Create valid synthetic HTDemucs weights with spectral branch.
pub(crate) fn make_htdemucs_spectral_weights() -> HTDemucsWeights {
    HTDemucsWeights::new(
        make_encoder_weights(),
        make_transformer_weights(),
        make_decoder_weights(),
        Some(make_spectral_encoder_weights()),
        Some(make_spectral_decoder_weights()),
    )
}

// ---------------------------------------------------------------------------
// Deterministic noise generator
// ---------------------------------------------------------------------------

pub(crate) fn deterministic_noise(len: usize) -> Vec<f32> {
    let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64 / u64::MAX as f64) * 0.2 - 0.1) as f32
        })
        .collect()
}
