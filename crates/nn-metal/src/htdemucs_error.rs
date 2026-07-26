// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error types for the unified HTDemucs model.

use crate::demucs_spectral_decoder::DemucsSpectralDecoderError;
use crate::demucs_spectral_encoder::DemucsSpectralEncoderError;
use crate::demucs_temporal_decoder::DemucsTemporalDecoderError;
use crate::demucs_temporal_encoder::DemucsTemporalEncoderError;
use crate::demucs_transformer::DemucsTransformerError;
use crate::istft::IstftError;

use super::WeightLoadError;

/// Errors from unified HTDemucs construction or forward pass.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HTDemucsError {
    /// Temporal encoder error.
    #[error("encoder: {0}")]
    Encoder(#[from] DemucsTemporalEncoderError),

    /// Transformer error.
    #[error("transformer: {0}")]
    Transformer(#[from] DemucsTransformerError),

    /// Temporal decoder error.
    #[error("decoder: {0}")]
    Decoder(#[from] DemucsTemporalDecoderError),

    /// Spectral encoder error.
    #[error("spectral encoder: {0}")]
    SpectralEncoder(#[from] DemucsSpectralEncoderError),

    /// Spectral decoder error.
    #[error("spectral decoder: {0}")]
    SpectralDecoder(#[from] DemucsSpectralDecoderError),

    /// STFT magnitude input has wrong length.
    #[error("STFT magnitude length {actual} != expected {expected} (spectral encoder requires STFT input)")]
    StftLength { expected: usize, actual: usize },

    /// Input audio has wrong length (must be exactly channels × audio_t).
    #[error("audio length {actual} != expected {expected} (channels={channels} × audio_t)")]
    AudioLength {
        actual: usize,
        expected: usize,
        channels: usize,
    },

    /// Audio too short for the encoder pipeline.
    #[error("audio temporal dim {actual} < minimum {minimum}")]
    AudioTooShort { actual: usize, minimum: usize },

    /// Non-finite values in input audio.
    #[error("non-finite input: {count} NaN/Inf values")]
    NonFiniteInput { count: usize },

    /// Non-finite values in an intermediate stage output.
    #[error("non-finite intermediate ({stage}): {count} NaN/Inf values")]
    NonFiniteIntermediate { stage: &'static str, count: usize },

    /// Non-finite values in final output after denormalization.
    #[error("non-finite output: {count} NaN/Inf values")]
    NonFiniteOutput { count: usize },

    /// Zero-length audio (t == 0) — cannot compute mean or variance.
    #[error("zero-length audio: t=0 produces division by zero in normalization")]
    ZeroLengthAudio,

    /// Normalization produced non-finite output from finite input.
    #[error("normalize overflow: {count} non-finite values in normalized output")]
    NormalizeOverflow { count: usize },

    /// Denormalization data shorter than expected output.
    #[error("denormalize length mismatch: data length {actual} < expected {expected}")]
    DenormalizeLengthMismatch { actual: usize, expected: usize },

    /// iSTFT reconstruction error.
    #[error("istft: {0}")]
    Istft(#[from] IstftError),

    /// One-sided spectral branch: encoder without decoder or vice versa.
    #[error("spectral branch requires both encoder and decoder weights, got only {provided}")]
    OneSidedSpectral {
        /// Which side was provided ("encoder" or "decoder").
        provided: &'static str,
    },

    /// Weight loading error.
    #[error("weight load: {0}")]
    WeightLoad(#[from] WeightLoadError),

    /// GPU tensor operation error (e.g., GPU iSTFT dispatch failure).
    #[error("tensor: {0}")]
    Tensor(#[from] nn_core::TensorError),
}

impl From<HTDemucsError> for nn_core::TensorError {
    fn from(e: HTDemucsError) -> Self {
        let msg = e.to_string();
        Self::backend_failure_with_source(
            nn_core::BackendDomain::Metal,
            nn_core::BackendErrorKind::Other,
            msg,
            e,
        )
    }
}
