// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Result types for the step-by-step Kokoro execution API.
//!
//! Extracted from `compiled_kokoro_steps.rs` to comply with the
//! 450-line production file limit. Part of #2218.

use nn_core::dyn_tensor::DynTensor;

/// Result of [`CompiledKokoro::split_style`](super::CompiledKokoro::split_style):
/// splits `[B, 2*style_dim]` voice embedding into decoder and prosody halves.
#[derive(Debug)]
#[non_exhaustive]
pub struct StyleSplit {
    /// Decoder style `[B, style_dim]` — input to Generator.
    pub decoder_style: DynTensor,
    /// Prosody style `[B, style_dim]` — input to ProsodyPredictor and F0EnergyPredictor.
    pub prosody_style: DynTensor,
}

impl StyleSplit {
    pub(crate) fn new(decoder_style: DynTensor, prosody_style: DynTensor) -> Self {
        Self {
            decoder_style,
            prosody_style,
        }
    }
}

/// Result of [`CompiledKokoro::step_encode`](super::CompiledKokoro::step_encode):
/// PlBert + bert_encoder + TextEncoder.
#[derive(Debug)]
#[non_exhaustive]
pub struct StepEncodeResult {
    /// ALBERT contextual features `[B, d_en, T]` — input to ProsodyPredictor.
    pub bert_features: DynTensor,
    /// TextEncoder features `[B, d_en, T]` — input to length_regulate → Generator.
    pub text_features: DynTensor,
    /// Input sequence length (segment cache key).
    pub seq_len: usize,
}

impl StepEncodeResult {
    pub(crate) fn new(bert_features: DynTensor, text_features: DynTensor, seq_len: usize) -> Self {
        Self {
            bert_features,
            text_features,
            seq_len,
        }
    }
}

/// Result of [`CompiledKokoro::step_predict_prosody`](super::CompiledKokoro::step_predict_prosody).
#[derive(Debug)]
#[non_exhaustive]
pub struct StepProsodyResult {
    /// Duration logits `[B, T, max_dur]` — sigmoid → sum → durations.
    pub dur_logits: DynTensor,
    /// Prosody features `[B, d_en+style_dim, T]` — input to length_regulate → F0EnergyPredictor.
    pub features: DynTensor,
}

impl StepProsodyResult {
    pub(crate) fn new(dur_logits: DynTensor, features: DynTensor) -> Self {
        Self {
            dur_logits,
            features,
        }
    }
}

/// Result of [`CompiledKokoro::step_regulate`](super::CompiledKokoro::step_regulate).
#[derive(Debug)]
#[non_exhaustive]
pub struct StepRegulateResult {
    /// Per-phoneme durations `[B, T]` after sigmoid+sum+scale.
    pub durations: DynTensor,
    /// Aligned prosody features `[B, d_en+style_dim, T_mel]` — input to F0EnergyPredictor.
    pub aligned_dur: DynTensor,
    /// Aligned text features `[B, d_en, T_mel]` — input to Generator.
    pub regulated: DynTensor,
    /// Time steps in mel domain (segment cache key for F0/Generator).
    pub t_mel: usize,
}

impl StepRegulateResult {
    pub(crate) fn new(
        durations: DynTensor,
        aligned_dur: DynTensor,
        regulated: DynTensor,
        t_mel: usize,
    ) -> Self {
        Self {
            durations,
            aligned_dur,
            regulated,
            t_mel,
        }
    }
}

/// Result of [`CompiledKokoro::step_predict_f0_energy`](super::CompiledKokoro::step_predict_f0_energy).
#[derive(Debug)]
#[non_exhaustive]
pub struct StepF0EnergyResult {
    /// Fundamental frequency prediction `[B, 1, 2*T_mel]`.
    pub f0: DynTensor,
    /// Energy envelope prediction `[B, 1, 2*T_mel]`.
    pub energy: DynTensor,
}

impl StepF0EnergyResult {
    pub(crate) fn new(f0: DynTensor, energy: DynTensor) -> Self {
        Self { f0, energy }
    }
}

/// Result of [`CompiledKokoro::step_generate`](super::CompiledKokoro::step_generate).
#[derive(Debug)]
#[non_exhaustive]
pub struct StepGeneratorResult {
    /// Spectral magnitude `[B, n_fft/2+1, T_frames]`.
    pub magnitude: DynTensor,
    /// Spectral phase `[B, n_fft/2+1, T_frames]`.
    pub phase: DynTensor,
}

impl StepGeneratorResult {
    pub(crate) fn new(magnitude: DynTensor, phase: DynTensor) -> Self {
        Self { magnitude, phase }
    }
}

/// Intermediate tensors from the synthesis pipeline.
///
/// Returned by [`CompiledKokoro::synthesize_with_intermediates`](super::CompiledKokoro::synthesize_with_intermediates)
/// alongside the audio and certificate. Useful for visualization, debugging,
/// prosody analysis, and building custom post-processing pipelines.
#[derive(Debug)]
#[non_exhaustive]
pub struct SynthesisIntermediates {
    /// Per-phoneme predicted durations `[B, T]` (mel frames per token).
    pub durations: DynTensor,
    /// Fundamental frequency prediction `[B, 1, 2*T_mel]` (Hz).
    pub f0: DynTensor,
    /// Energy envelope prediction `[B, 1, 2*T_mel]`.
    pub energy: DynTensor,
    /// Time steps in mel domain (derived from duration sum).
    pub t_mel: usize,
}

impl SynthesisIntermediates {
    pub(crate) fn new(
        durations: DynTensor,
        f0: DynTensor,
        energy: DynTensor,
        t_mel: usize,
    ) -> Self {
        Self {
            durations,
            f0,
            energy,
            t_mel,
        }
    }
}
