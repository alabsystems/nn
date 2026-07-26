// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Weight types for the Demucs spectral encoder and decoder.
//!
//! Pure data structs with no backend dependencies. Consumed by both nn-metal
//! (GPU dispatch) and nn-verify (NY composition tests).
//!
//! Extracted from `demucs_spectral_encoder_types.rs` and
//! `demucs_spectral_decoder_types.rs` as part of #860.

// ---------------------------------------------------------------------------
// Spectral encoder weights
// ---------------------------------------------------------------------------

/// Weights for a single DConv sub-layer (spectral encoder variant).
#[derive(Debug, Clone)]
#[must_use = "SpectralEncDConvSubLayerWeights is a data transfer type"]
pub struct SpectralEncDConvSubLayerWeights {
    /// Conv1d compress: [compressed, channels, dconv_kernel].
    pub conv_compress_weight: Vec<f32>,
    pub conv_compress_bias: Vec<f32>,
    pub norm_compress_gamma: Vec<f32>,
    pub norm_compress_beta: Vec<f32>,
    /// Conv1d expand: [channels*2, compressed, 1].
    pub conv_expand_weight: Vec<f32>,
    pub conv_expand_bias: Vec<f32>,
    pub norm_expand_gamma: Vec<f32>,
    pub norm_expand_beta: Vec<f32>,
    /// LayerScale: [channels].
    pub layer_scale: Vec<f32>,
}

/// Weights for a single spectral encoder block.
#[derive(Debug, Clone)]
#[must_use = "SpectralEncoderBlockWeights is a data transfer type"]
pub struct SpectralEncoderBlockWeights {
    /// Main Conv1d: [out_ch, in_ch, kernel_size=8].
    pub conv_weight: Vec<f32>,
    /// Main Conv1d bias: [out_ch].
    pub conv_bias: Vec<f32>,
    /// DConv sub-layers (default: 2).
    pub dconv: Vec<SpectralEncDConvSubLayerWeights>,
    /// Rewrite Conv1d: [out_ch*2, out_ch, 1].
    pub rewrite_weight: Vec<f32>,
    /// Rewrite Conv1d bias: [out_ch*2].
    pub rewrite_bias: Vec<f32>,
}

/// All weights for the spectral encoder basic blocks + optional freq embedding.
///
/// For the full 6-depth HTDemucs, the basic builders handle depths 0-3.
/// Depths 4-5 (deep blocks with LSTM + attention) require separate weight types.
#[derive(Debug, Clone)]
#[must_use = "DemucsSpectralEncoderWeights is a data transfer type"]
pub struct DemucsSpectralEncoderWeights {
    pub blocks: Vec<SpectralEncoderBlockWeights>,
    /// Optional frequency embedding weight: [FREQ_EMB_FEATURES, FREQ_EMB_DIM].
    /// Applied after depth-0 output. `None` to skip.
    pub freq_emb_weight: Option<Vec<f32>>,
}

// ---------------------------------------------------------------------------
// Spectral decoder weights
// ---------------------------------------------------------------------------

/// Weights for a single DConv sub-layer (spectral decoder variant).
#[derive(Debug, Clone)]
#[must_use = "SpectralDConvSubLayerWeights is a data transfer type"]
pub struct SpectralDConvSubLayerWeights {
    /// Conv1d compress: [compressed, channels, dconv_kernel].
    pub conv_compress_weight: Vec<f32>,
    pub conv_compress_bias: Vec<f32>,
    pub norm_compress_gamma: Vec<f32>,
    pub norm_compress_beta: Vec<f32>,
    /// Conv1d expand: [channels*2, compressed, 1].
    pub conv_expand_weight: Vec<f32>,
    pub conv_expand_bias: Vec<f32>,
    pub norm_expand_gamma: Vec<f32>,
    pub norm_expand_beta: Vec<f32>,
    /// LayerScale: [channels].
    pub layer_scale: Vec<f32>,
}

/// Weights for a single spectral decoder block.
#[derive(Debug, Clone)]
#[must_use = "SpectralDecoderBlockWeights is a data transfer type"]
pub struct SpectralDecoderBlockWeights {
    /// Rewrite Conv2d: [in_ch*2, in_ch, 3, 3] (stored as flat [in_ch*2 * in_ch * 9]).
    pub rewrite_weight: Vec<f32>,
    /// Rewrite Conv2d bias: [in_ch*2].
    pub rewrite_bias: Vec<f32>,
    /// DConv sub-layers (default: 2).
    pub dconv: Vec<SpectralDConvSubLayerWeights>,
    /// ConvTranspose1d: [in_ch, out_ch, kernel_size=8].
    pub conv_tr_weight: Vec<f32>,
    /// ConvTranspose1d bias: [out_ch].
    pub conv_tr_bias: Vec<f32>,
}

/// All weights for the spectral decoder basic blocks.
///
/// For the full 6-depth HTDemucs, the basic builders handle depths 0-3.
/// Depths 4-5 (deep blocks) require separate weight types.
#[derive(Debug, Clone)]
#[must_use = "DemucsSpectralDecoderWeights is a data transfer type"]
pub struct DemucsSpectralDecoderWeights {
    pub blocks: Vec<SpectralDecoderBlockWeights>,
}
