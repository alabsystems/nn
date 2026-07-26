// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Debug forward-pass variants for HTDemucs.
//!
//! Extracted from `htdemucs.rs` for the 500-line limit.
//! These methods return intermediate tensors for parity debugging.
//!
//! Part of #900 (code-health extraction).

use super::{
    denormalize_output, normalize_audio, HTDemucs, HTDemucsError, AUDIO_CHANNELS, BOTTLENECK_DIM,
};
use crate::PipelineCache;

impl HTDemucs {
    /// Forward pass returning per-block encoder intermediates for parity debugging.
    ///
    /// Returns `(normalized_audio, block_inputs, block_outputs)` where
    /// `block_inputs[i]` is the input to encoder block `i` (after stride padding)
    /// and `block_outputs[i]` is the output of block `i`.
    pub fn forward_encoder_debug(
        &self,
        cache: &PipelineCache,
        audio: &[f32],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>), HTDemucsError> {
        let expected_len = AUDIO_CHANNELS * self.audio_t;
        if audio.len() != expected_len {
            return Err(HTDemucsError::AudioLength {
                actual: audio.len(),
                expected: expected_len,
                channels: AUDIO_CHANNELS,
            });
        }
        let (normalized, _mean, _std_val) = normalize_audio(audio, self.audio_t)?;
        let (_enc_out, block_inputs, block_outputs) =
            self.encoder.forward_debug(cache, &normalized)?;
        Ok((normalized, block_inputs, block_outputs))
    }

    /// Forward pass returning intermediate tensors for debugging.
    ///
    /// Returns `(encoder_bottleneck, transformer_output, final_output)`.
    pub fn forward_debug(
        &self,
        cache: &PipelineCache,
        audio: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), HTDemucsError> {
        let expected_len = AUDIO_CHANNELS * self.audio_t;
        if audio.len() != expected_len {
            return Err(HTDemucsError::AudioLength {
                actual: audio.len(),
                expected: expected_len,
                channels: AUDIO_CHANNELS,
            });
        }
        crate::check_non_finite_err(audio, |count| HTDemucsError::NonFiniteInput { count })?;

        let (normalized, mean, std_val) = normalize_audio(audio, self.audio_t)?;
        let enc_out = self.encoder.forward(cache, &normalized)?;
        let encoder_bottleneck = enc_out.bottleneck.clone();

        let spectral_zeros = vec![0.0f32; BOTTLENECK_DIM * self.spectral_seq_len];
        let (temporal_out, _) =
            self.transformer
                .forward(cache, &enc_out.bottleneck, &spectral_zeros)?;
        let transformer_output = temporal_out.clone();

        let decoded = self.decoder.forward(cache, &temporal_out, &enc_out.skips)?;
        let output = denormalize_output(&decoded, self.audio_t, mean, std_val)?;

        Ok((encoder_bottleneck, transformer_output, output))
    }
}
