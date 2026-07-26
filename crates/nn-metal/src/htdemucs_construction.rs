// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HTDemucs constructor logic — extracted from `htdemucs.rs` for file-size compliance (#1342).

use crate::demucs_spectral_decoder::DemucsSpectralDecoder;
use crate::demucs_spectral_encoder::DemucsSpectralEncoder;
use crate::demucs_temporal_decoder::DemucsTemporalDecoder;
use crate::demucs_temporal_encoder::DemucsTemporalEncoder;
use crate::demucs_transformer::DemucsTransformer;
use crate::istft::{IstftBasis, IstftParams};
use crate::istft_gpu::IstftGpuBasis;

use super::{
    compute_bottleneck_t, compute_encoder_input_lengths, compute_spectral_encoder_freqs, HTDemucs,
    HTDemucsError, HTDemucsWeights, SPECTRAL_BOTTLENECK_F, SPECTRAL_INPUT_CHANNELS,
};

impl HTDemucs {
    pub(super) fn new_inner(
        weights: HTDemucsWeights,
        audio_t: usize,
        stft_f: Option<usize>,
        stft_t: Option<usize>,
    ) -> Result<Self, HTDemucsError> {
        // Guard against underflow in compute_bottleneck_t — audio_t == 0
        // causes unsigned subtraction panic in the encoder shape computation.
        if audio_t < Self::MINIMUM_AUDIO_T {
            return Err(HTDemucsError::AudioTooShort {
                actual: audio_t,
                minimum: Self::MINIMUM_AUDIO_T,
            });
        }

        // Build temporal encoder.
        let encoder = DemucsTemporalEncoder::new(weights.encoder, audio_t)?;
        let bottleneck_t = compute_bottleneck_t(audio_t);

        // Build spectral encoder/decoder if weights are provided.
        let (
            spectral_encoder,
            spectral_decoder,
            istft_basis,
            istft_gpu_basis,
            spectral_seq_len,
            stft_expected_len,
            resolved_f,
            resolved_t,
        ) = match (weights.spectral_encoder, weights.spectral_decoder) {
            (Some(enc_w), Some(dec_w)) => {
                let f = stft_f.unwrap_or(2048);
                let t = stft_t.unwrap_or(bottleneck_t);
                let spec_enc = DemucsSpectralEncoder::new(enc_w, f, t)?;
                // Spectral seq len = F_bottleneck × T (reshaped for transformer).
                let seq_len = SPECTRAL_BOTTLENECK_F * t;
                let expected = SPECTRAL_INPUT_CHANNELS * f * t;

                // Build spectral decoder with encoder freq/time info.
                let encoder_freqs = compute_spectral_encoder_freqs(f);
                let encoder_times = vec![t; 4];
                let spec_dec = DemucsSpectralDecoder::new(dec_w, &encoder_freqs, &encoder_times)?;

                // Pre-compute iSTFT basis for spectral→waveform reconstruction.
                // HTDemucs uses n_fft = (f-1)*2 (f is one-sided bins = n_fft/2+1),
                // hop = n_fft/4, normalized=true, center=true.
                let n_fft = (f - 1) * 2;
                let hop_length = n_fft / 4;
                let istft_params = IstftParams::new(n_fft, hop_length, true, true)?;
                let basis = IstftBasis::new(istft_params)?;

                // Best-effort GPU iSTFT basis upload. Falls back to CPU if
                // Metal context is unavailable (e.g., non-macOS or not init'd).
                let gpu_basis = IstftGpuBasis::from_basis(&basis).ok();

                (
                    Some(spec_enc),
                    Some(spec_dec),
                    Some(basis),
                    gpu_basis,
                    seq_len,
                    expected,
                    f,
                    t,
                )
            }
            (Some(_), None) => {
                return Err(HTDemucsError::OneSidedSpectral {
                    provided: "encoder",
                });
            }
            (None, Some(_)) => {
                return Err(HTDemucsError::OneSidedSpectral {
                    provided: "decoder",
                });
            }
            (None, None) => (None, None, None, None, 1, 0, 0, 0),
        };

        let transformer =
            DemucsTransformer::new(weights.transformer, bottleneck_t, spectral_seq_len)?;

        let encoder_input_lengths = compute_encoder_input_lengths(audio_t);
        let decoder = DemucsTemporalDecoder::new(weights.decoder, &encoder_input_lengths)?;

        Ok(Self {
            encoder,
            transformer,
            decoder,
            spectral_encoder,
            spectral_decoder,
            istft_basis,
            istft_gpu_basis,
            audio_t,
            spectral_seq_len,
            stft_expected_len,
            stft_f: resolved_f,
            stft_t: resolved_t,
        })
    }
}
