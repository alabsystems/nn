// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compile-time pipeline shape verification for the CompiledKokoro step API.
//!
//! The 7-step Kokoro pipeline (encode -> prosody -> regulate -> f0_energy ->
//! harmonic_source -> generate -> istft) has implicit shape contracts between
//! steps. When shapes don't match, failures surface as cryptic Metal GPU
//! errors. This module adds type-level verification of those shape contracts
//! using phantom type tags.
//!
//! [`PipelineTensor<Tag>`] is a zero-cost wrapper around [`DynTensor`] that
//! carries a compile-time pipeline position marker. The typed step methods
//! on [`CompiledKokoro`] accept and return `PipelineTensor` values, so the
//! compiler enforces that outputs from one step flow into the correct next
//! step. Passing a `PipelineTensor<EncoderOutput>` where a
//! `PipelineTensor<RegulatedOutput>` is expected is a compile error.
//!
//! # Zero-cost abstraction
//!
//! `PhantomData<Tag>` is a zero-sized type. The tag is erased by the
//! compiler — `PipelineTensor<T>` has the same size and alignment as
//! `DynTensor`. There is no runtime overhead.
//!
//! # Example
//!
//! ```rust,ignore
//! let encode = kokoro.step_encode_typed(&input_ids, &cache)?;
//! let prosody = kokoro.step_predict_prosody_typed(
//!     &encode.bert_features, &style.prosody_style, encode.seq_len, &cache,
//! )?;
//! // This would be a compile error — prosody output is not encoder output:
//! // let bad = kokoro.step_predict_prosody_typed(&prosody.features, ...);
//! ```
//!
//! Part of #3635.

use std::marker::PhantomData;

use nn_core::dyn_tensor::DynTensor;

/// Type-tagged tensor wrapper for CompiledKokoro pipeline positions.
///
/// Zero-cost at runtime — the tag is erased by the compiler.
/// Use the typed step methods on [`CompiledKokoro`] to create and
/// consume these values with compile-time shape contract enforcement.
#[derive(Debug, Clone)]
pub struct PipelineTensor<Tag> {
    pub(crate) inner: DynTensor,
    _tag: PhantomData<Tag>,
}

impl<Tag> PipelineTensor<Tag> {
    /// Wrap a `DynTensor` with a pipeline position tag.
    ///
    /// This is `pub(crate)` — only the typed step methods should create
    /// tagged tensors, ensuring tags are assigned correctly.
    pub(crate) fn new(inner: DynTensor) -> Self {
        Self {
            inner,
            _tag: PhantomData,
        }
    }

    /// Access the inner `DynTensor` by reference.
    #[must_use]
    pub fn inner(&self) -> &DynTensor {
        &self.inner
    }

    /// Consume the wrapper and return the inner `DynTensor`.
    ///
    /// Use this when you need to pass the tensor to untyped APIs or
    /// perform custom modifications between pipeline steps.
    #[must_use]
    pub fn into_inner(self) -> DynTensor {
        self.inner
    }
}

// ---------------------------------------------------------------------------
// Pipeline position markers (zero-sized types)
// ---------------------------------------------------------------------------

/// Marker: encoder input token IDs `[B, T]`.
#[derive(Debug, Clone, Copy)]
pub struct EncoderInput;

/// Marker: ALBERT bert_features `[B, d_en, T]` from segment 0 (PlBert + bert_encoder).
#[derive(Debug, Clone, Copy)]
pub struct BertFeaturesOutput;

/// Marker: TextEncoder features `[B, d_en, T]` from segment 1.
#[derive(Debug, Clone, Copy)]
pub struct TextFeaturesOutput;

/// Marker: prosody duration logits `[B, T, max_dur]` from ProsodyPredictor.
#[derive(Debug, Clone, Copy)]
pub struct ProsodyDurLogitsOutput;

/// Marker: prosody features `[B, d_en+style_dim, T]` from ProsodyPredictor.
#[derive(Debug, Clone, Copy)]
pub struct ProsodyFeaturesOutput;

/// Marker: aligned prosody features `[B, d_en+style_dim, T_mel]` after length_regulate.
#[derive(Debug, Clone, Copy)]
pub struct AlignedDurOutput;

/// Marker: aligned text features `[B, d_en, T_mel]` after length_regulate.
#[derive(Debug, Clone, Copy)]
pub struct RegulatedOutput;

/// Marker: F0 prediction `[B, 1, 2*T_mel]` from F0EnergyPredictor.
#[derive(Debug, Clone, Copy)]
pub struct F0Output;

/// Marker: energy prediction `[B, 1, 2*T_mel]` from F0EnergyPredictor.
#[derive(Debug, Clone, Copy)]
pub struct EnergyOutput;

/// Marker: harmonic source excitation tensor from SineGen.
#[derive(Debug, Clone, Copy)]
pub struct HarmonicSourceOutput;

/// Marker: spectral magnitude `[B, n_fft/2+1, T_frames]` from Generator.
#[derive(Debug, Clone, Copy)]
pub struct MagnitudeOutput;

/// Marker: spectral phase `[B, n_fft/2+1, T_frames]` from Generator.
#[derive(Debug, Clone, Copy)]
pub struct PhaseOutput;

/// Marker: final PCM audio `[1, 1, T_audio]` from iSTFT.
#[derive(Debug, Clone, Copy)]
pub struct IstftOutput;

// ---------------------------------------------------------------------------
// Typed step result types
// ---------------------------------------------------------------------------

/// Typed result of [`CompiledKokoro::step_encode_typed`].
#[derive(Debug)]
#[non_exhaustive]
pub struct TypedEncodeResult {
    /// ALBERT contextual features `[B, d_en, T]`.
    pub bert_features: PipelineTensor<BertFeaturesOutput>,
    /// TextEncoder features `[B, d_en, T]`.
    pub text_features: PipelineTensor<TextFeaturesOutput>,
    /// Input sequence length (segment cache key).
    pub seq_len: usize,
}

/// Typed result of [`CompiledKokoro::step_predict_prosody_typed`].
#[derive(Debug)]
#[non_exhaustive]
pub struct TypedProsodyResult {
    /// Duration logits `[B, T, max_dur]`.
    pub dur_logits: PipelineTensor<ProsodyDurLogitsOutput>,
    /// Prosody features `[B, d_en+style_dim, T]`.
    pub features: PipelineTensor<ProsodyFeaturesOutput>,
}

/// Typed result of [`CompiledKokoro::step_regulate_typed`].
#[derive(Debug)]
#[non_exhaustive]
pub struct TypedRegulateResult {
    /// Per-phoneme durations `[B, T]`.
    pub durations: DynTensor,
    /// Aligned prosody features `[B, d_en+style_dim, T_mel]`.
    pub aligned_dur: PipelineTensor<AlignedDurOutput>,
    /// Aligned text features `[B, d_en, T_mel]`.
    pub regulated: PipelineTensor<RegulatedOutput>,
    /// Time steps in mel domain.
    pub t_mel: usize,
}

/// Typed result of [`CompiledKokoro::step_predict_f0_energy_typed`].
#[derive(Debug)]
#[non_exhaustive]
pub struct TypedF0EnergyResult {
    /// Fundamental frequency prediction `[B, 1, 2*T_mel]`.
    pub f0: PipelineTensor<F0Output>,
    /// Energy envelope prediction `[B, 1, 2*T_mel]`.
    pub energy: PipelineTensor<EnergyOutput>,
}

/// Typed result of [`CompiledKokoro::step_generate_typed`].
#[derive(Debug)]
#[non_exhaustive]
pub struct TypedGeneratorResult {
    /// Spectral magnitude `[B, n_fft/2+1, T_frames]`.
    pub magnitude: PipelineTensor<MagnitudeOutput>,
    /// Spectral phase `[B, n_fft/2+1, T_frames]`.
    pub phase: PipelineTensor<PhaseOutput>,
}

#[cfg(test)]
#[path = "compiled_kokoro_shapes_tests.rs"]
mod tests;
