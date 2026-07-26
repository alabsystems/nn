// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HTDemucs forward pass implementation.
//!
//! Extracted from `htdemucs.rs` to keep files under 500 lines.
//! Contains all forward methods (CPU and GPU).

use super::{
    check_finite, denormalize_output, normalize_audio, spectral_reconstruct,
    spectral_reconstruct_gpu, EncoderDispatch, HTDemucs, HTDemucsError, AUDIO_CHANNELS,
    BOTTLENECK_DIM,
};
use crate::PipelineCache;
use nn_core::layers::{with_nan_check_policy, NanCheckPolicy};

impl HTDemucs {
    /// Run the temporal-only forward pass (spectral input is zeros).
    ///
    /// `cache`: Metal pipeline cache for GPU dispatch.
    /// `audio`: stereo audio `[2, T]` (flattened, row-major).
    ///
    /// Returns flattened `[OUTPUT_CHANNELS, T]` (4 sources × 2 channels × T).
    pub fn forward(&self, cache: &PipelineCache, audio: &[f32]) -> Result<Vec<f32>, HTDemucsError> {
        self.forward_inner(cache, audio, None, EncoderDispatch::Cpu)
    }

    /// Run the full forward pass with both temporal and spectral branches.
    ///
    /// `stft_mag`: pre-computed STFT magnitude `[4, F, T]` (flattened).
    ///             4 = 2 stereo channels × 2 (real + imaginary).
    ///
    /// The spectral encoder must have been provided at construction time.
    /// Returns flattened `[OUTPUT_CHANNELS, T]` (4 sources × 2 channels × T).
    pub fn forward_with_stft(
        &self,
        cache: &PipelineCache,
        audio: &[f32],
        stft_mag: &[f32],
    ) -> Result<Vec<f32>, HTDemucsError> {
        self.forward_inner(cache, audio, Some(stft_mag), EncoderDispatch::Cpu)
    }

    /// GPU-optimized temporal-only forward pass.
    ///
    /// Same semantics as [`forward`](Self::forward), but uses buffer-to-buffer
    /// dispatch in the temporal encoder (eliminating up to 3 CPU<->GPU
    /// round-trips between encoder blocks). Transformer and decoder use the
    /// standard CPU dispatch path.
    ///
    /// Sub-components (encoder, transformer) create their own
    /// [`with_gpu_scope`](crate::gpu_scope::with_gpu_scope) internally for
    /// per-component batching. The outer forward pass does NOT wrap in
    /// `with_gpu_scope` because `forward_inner` performs CPU readbacks between
    /// components (encoder bottleneck → transformer input), which require
    /// committed GPU data. See #1933, #1912.
    pub fn forward_gpu(
        &self,
        cache: &PipelineCache,
        audio: &[f32],
    ) -> Result<Vec<f32>, HTDemucsError> {
        // Input validation runs OUTSIDE the NaN-skip scope so NaN input is
        // always rejected regardless of the per-stage check policy (#1958).
        let count = crate::count_non_finite(audio);
        if count > 0 {
            return Err(HTDemucsError::NonFiniteInput { count });
        }
        // NaN-skip scope: per-stage checks inside forward_inner are no-ops.
        let output = with_nan_check_policy(NanCheckPolicy::Skip, || {
            self.forward_inner(cache, audio, None, EncoderDispatch::Gpu)
        })?;
        // Defense-in-depth: model-boundary output check runs AFTER NaN-skip scope exits.
        crate::check_non_finite_err(&output, |count| HTDemucsError::NonFiniteOutput { count })?;
        Ok(output)
    }

    /// GPU-optimized forward pass with both temporal and spectral branches.
    ///
    /// Same semantics as [`forward_with_stft`](Self::forward_with_stft), but
    /// uses buffer-to-buffer dispatch in the temporal encoder.
    /// Sub-components create their own GPU scopes internally. See #1933.
    pub fn forward_gpu_with_stft(
        &self,
        cache: &PipelineCache,
        audio: &[f32],
        stft_mag: &[f32],
    ) -> Result<Vec<f32>, HTDemucsError> {
        // Input validation runs OUTSIDE the NaN-skip scope (#1958).
        let count = crate::count_non_finite(audio);
        if count > 0 {
            return Err(HTDemucsError::NonFiniteInput { count });
        }
        // NaN-skip scope: per-stage checks inside forward_inner are no-ops.
        let output = with_nan_check_policy(NanCheckPolicy::Skip, || {
            self.forward_inner(cache, audio, Some(stft_mag), EncoderDispatch::Gpu)
        })?;
        // Defense-in-depth: model-boundary output check runs AFTER NaN-skip scope exits.
        crate::check_non_finite_err(&output, |count| HTDemucsError::NonFiniteOutput { count })?;
        Ok(output)
    }

    fn forward_inner(
        &self,
        cache: &PipelineCache,
        audio: &[f32],
        stft_mag: Option<&[f32]>,
        encoder_dispatch: EncoderDispatch,
    ) -> Result<Vec<f32>, HTDemucsError> {
        // Validate input dimensions.
        let expected_len = AUDIO_CHANNELS * self.audio_t;
        if audio.len() != expected_len {
            return Err(HTDemucsError::AudioLength {
                actual: audio.len(),
                expected: expected_len,
                channels: AUDIO_CHANNELS,
            });
        }

        crate::check_non_finite_err(audio, |count| HTDemucsError::NonFiniteInput { count })?;

        // Step 1: Normalize input audio.
        let (normalized, mean, std_val) = normalize_audio(audio, self.audio_t)?;

        // Step 2: Temporal encoder — dispatch via CPU or GPU buffer-to-buffer.
        let enc_out = match encoder_dispatch {
            EncoderDispatch::Cpu => self.encoder.forward(cache, &normalized)?,
            EncoderDispatch::Gpu => self.encoder.forward_gpu(cache, &normalized)?,
        };

        // AC1: Check encoder output for NaN/Inf before transformer.
        check_finite(&enc_out.bottleneck, "encoder")?;

        // Step 3: Spectral encoder (if present) or zeros.
        // Preserve encoder skips for the spectral decoder.
        let (spectral_input, spec_enc_skips) =
            if let (Some(spec_enc), Some(stft)) = (&self.spectral_encoder, stft_mag) {
                if stft.len() != self.stft_expected_len {
                    return Err(HTDemucsError::StftLength {
                        expected: self.stft_expected_len,
                        actual: stft.len(),
                    });
                }
                let spec_out = spec_enc.forward(cache, stft)?;
                check_finite(&spec_out.bottleneck, "spectral_encoder")?;
                (spec_out.bottleneck, Some(spec_out.skips))
            } else {
                (vec![0.0f32; BOTTLENECK_DIM * self.spectral_seq_len], None)
            };

        // Step 4: Transformer (temporal + spectral cross-attention).
        // GPU dispatch: use buffer-to-buffer to eliminate ~12 CPU round-trips.
        let (temporal_out, spectral_out) = match encoder_dispatch {
            EncoderDispatch::Cpu => {
                self.transformer
                    .forward(cache, &enc_out.bottleneck, &spectral_input)?
            }
            EncoderDispatch::Gpu => {
                self.transformer
                    .forward_gpu(cache, &enc_out.bottleneck, &spectral_input)?
            }
        };

        check_finite(&temporal_out, "transformer")?;

        // Step 5: Temporal decoder with skip connections — dispatch via CPU or GPU.
        // When GPU skip buffers are available (from encoder GPU path), pass them
        // to the decoder for GPU-resident center-trim (eliminates 4 readbacks).
        let decoded = match encoder_dispatch {
            EncoderDispatch::Cpu => self.decoder.forward(cache, &temporal_out, &enc_out.skips)?,
            EncoderDispatch::Gpu => {
                if let Some(ref gpu_skips) = enc_out.skips_gpu {
                    self.decoder.forward_gpu_with_skips(
                        cache,
                        &temporal_out,
                        &enc_out.skips,
                        gpu_skips,
                    )?
                } else {
                    self.decoder
                        .forward_gpu(cache, &temporal_out, &enc_out.skips)?
                }
            }
        };
        check_finite(&decoded, "decoder")?;

        // Step 6: Spectral decoder → iSTFT → waveform (if spectral branch present).
        // GPU dispatch uses GPU-accelerated iSTFT when the GPU basis is available.
        let spectral_waveform = if let (Some(spec_dec), Some(basis), Some(skips)) =
            (&self.spectral_decoder, &self.istft_basis, spec_enc_skips)
        {
            check_finite(&spectral_out, "transformer_spectral")?;
            let spec_decoded = spec_dec.forward(cache, &spectral_out, &skips)?;
            check_finite(&spec_decoded, "spectral_decoder")?;
            let waveform = if let (EncoderDispatch::Gpu, Some(gpu_basis)) =
                (encoder_dispatch, &self.istft_gpu_basis)
            {
                spectral_reconstruct_gpu(
                    &spec_decoded,
                    gpu_basis,
                    cache,
                    self.stft_f,
                    self.stft_t,
                    self.audio_t,
                    mean,
                    std_val,
                )?
            } else {
                spectral_reconstruct(
                    &spec_decoded,
                    basis,
                    self.stft_f,
                    self.stft_t,
                    self.audio_t,
                    mean,
                    std_val,
                )?
            };
            Some(waveform)
        } else {
            None
        };

        // Step 7: Denormalize temporal output.
        let mut output = denormalize_output(&decoded, self.audio_t, mean, std_val)?;

        // Step 8: Sum temporal + spectral waveforms.
        if let Some(spectral) = spectral_waveform {
            for (o, &s) in output.iter_mut().zip(spectral.iter()) {
                *o += s;
            }
        }

        crate::check_non_finite_err(&output, |count| HTDemucsError::NonFiniteOutput { count })?;

        Ok(output)
    }
}
