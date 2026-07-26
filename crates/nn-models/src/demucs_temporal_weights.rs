// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Weight types for the Demucs temporal encoder and decoder.
//!
//! Pure data structs with no backend dependencies. Consumed by both nn-metal
//! (GPU dispatch) and nn-verify (NY composition tests).
//!
//! Extracted from `demucs_temporal_decoder.rs` and `demucs_temporal_encoder.rs`
//! as part of #860.

// ---------------------------------------------------------------------------
// Shared DConv sub-layer weights
// ---------------------------------------------------------------------------

/// Weights for a single DConv sub-layer.
#[derive(Debug, Clone)]
#[must_use = "DConvSubLayerWeights is a data transfer type; pass to DecoderBlockWeights"]
pub struct DConvSubLayerWeights {
    /// Conv1d compress: [compressed, channels, dconv_kernel].
    pub conv_compress_weight: Vec<f32>,
    /// Conv1d compress bias: [compressed].
    pub conv_compress_bias: Vec<f32>,
    /// GroupNorm compress: gamma [compressed], beta [compressed].
    pub norm_compress_gamma: Vec<f32>,
    pub norm_compress_beta: Vec<f32>,
    /// Conv1d expand: [channels*2, compressed, 1].
    pub conv_expand_weight: Vec<f32>,
    /// Conv1d expand bias: [channels*2].
    pub conv_expand_bias: Vec<f32>,
    /// GroupNorm expand: gamma [channels*2], beta [channels*2].
    pub norm_expand_gamma: Vec<f32>,
    pub norm_expand_beta: Vec<f32>,
    /// LayerScale: [channels].
    pub layer_scale: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Temporal decoder weights
// ---------------------------------------------------------------------------

/// Weights for a single decoder block.
#[derive(Debug, Clone)]
#[must_use = "DecoderBlockWeights is a data transfer type; pass to DemucsTemporalDecoderWeights"]
pub struct DecoderBlockWeights {
    /// Rewrite Conv1d: [in_ch*2, in_ch, rewrite_kernel=3].
    pub rewrite_weight: Vec<f32>,
    /// Rewrite Conv1d bias: [in_ch*2].
    pub rewrite_bias: Vec<f32>,
    /// DConv sub-layers (default: 2).
    pub dconv: Vec<DConvSubLayerWeights>,
    /// ConvTranspose1d: [in_ch, out_ch, kernel_size=8].
    pub conv_tr_weight: Vec<f32>,
    /// ConvTranspose1d bias: [out_ch].
    pub conv_tr_bias: Vec<f32>,
}

/// All weights for the temporal decoder (5 blocks for full HTDemucs).
#[derive(Debug, Clone)]
#[must_use = "DemucsTemporalDecoderWeights is a data transfer type; pass to DemucsTemporalDecoder::new()"]
pub struct DemucsTemporalDecoderWeights {
    pub blocks: Vec<DecoderBlockWeights>,
}

// ---------------------------------------------------------------------------
// Temporal encoder weights
// ---------------------------------------------------------------------------

/// Weights for a single encoder block.
#[derive(Debug, Clone)]
#[must_use = "EncoderBlockWeights is a data transfer type; pass to DemucsTemporalEncoderWeights"]
pub struct EncoderBlockWeights {
    /// Conv1d: [out_ch, in_ch, kernel_size=8].
    pub conv_weight: Vec<f32>,
    /// Conv1d bias: [out_ch].
    pub conv_bias: Vec<f32>,
    /// DConv sub-layers (default: 2).
    pub dconv: Vec<DConvSubLayerWeights>,
    /// Rewrite Conv1d: [out_ch*2, out_ch, 1].
    pub rewrite_weight: Vec<f32>,
    /// Rewrite Conv1d bias: [out_ch*2].
    pub rewrite_bias: Vec<f32>,
}

/// All weights for the temporal encoder (5 blocks for full HTDemucs: 4 basic + 1 final).
#[derive(Debug, Clone)]
#[must_use = "DemucsTemporalEncoderWeights is a data transfer type; pass to DemucsTemporalEncoder::new()"]
pub struct DemucsTemporalEncoderWeights {
    pub blocks: Vec<EncoderBlockWeights>,
}
