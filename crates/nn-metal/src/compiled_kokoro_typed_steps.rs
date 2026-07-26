// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Type-safe step methods for [`CompiledKokoro`] with phantom type enforcement.
//!
//! Each method wraps the corresponding untyped step method (in
//! `compiled_kokoro_steps.rs` / `compiled_kokoro_step_regulate.rs`) and
//! returns [`PipelineTensor`]-tagged results. The compiler enforces that
//! outputs flow into the correct next step — passing a
//! `PipelineTensor<BertFeaturesOutput>` where `PipelineTensor<RegulatedOutput>`
//! is expected is a compile error.
//!
//! The untyped methods remain available for callers who need to modify
//! intermediate tensors between steps (the typed API can be escaped via
//! [`PipelineTensor::into_inner`]).
//!
//! Part of #3635.

use nn_core::dyn_tensor::DynTensor;

use crate::cache::PipelineCache;

use super::compiled_kokoro_shapes::{
    AlignedDurOutput, BertFeaturesOutput, EnergyOutput, F0Output, HarmonicSourceOutput,
    IstftOutput, MagnitudeOutput, PhaseOutput, PipelineTensor, ProsodyDurLogitsOutput,
    ProsodyFeaturesOutput, RegulatedOutput, TextFeaturesOutput, TypedEncodeResult,
    TypedF0EnergyResult, TypedGeneratorResult, TypedProsodyResult, TypedRegulateResult,
};
use super::CompiledKokoro;
use super::CompiledKokoroError;

impl CompiledKokoro {
    /// Type-safe Step 1+2: Encode input tokens.
    ///
    /// Wraps [`step_encode`](Self::step_encode) with phantom type tags.
    /// Returns [`TypedEncodeResult`] whose fields can only be passed to
    /// the correct downstream steps.
    pub fn step_encode_typed(
        &mut self,
        input_ids: &DynTensor,
        cache: &PipelineCache,
    ) -> Result<TypedEncodeResult, CompiledKokoroError> {
        let result = self.step_encode(input_ids, cache)?;
        Ok(TypedEncodeResult {
            bert_features: PipelineTensor::new(result.bert_features),
            text_features: PipelineTensor::new(result.text_features),
            seq_len: result.seq_len,
        })
    }

    /// Type-safe Step 3: Run ProsodyPredictor.
    ///
    /// Wraps [`step_predict_prosody`](Self::step_predict_prosody) with phantom
    /// type tags. Accepts `PipelineTensor<BertFeaturesOutput>` — a tensor from
    /// a different pipeline position is a compile error.
    pub fn step_predict_prosody_typed(
        &mut self,
        bert_features: &PipelineTensor<BertFeaturesOutput>,
        prosody_style: &DynTensor,
        seq_len: usize,
        cache: &PipelineCache,
    ) -> Result<TypedProsodyResult, CompiledKokoroError> {
        let result =
            self.step_predict_prosody(bert_features.inner(), prosody_style, seq_len, cache)?;
        Ok(TypedProsodyResult {
            dur_logits: PipelineTensor::new(result.dur_logits),
            features: PipelineTensor::new(result.features),
        })
    }

    /// Type-safe Step 4: Compute durations and run length_regulate.
    ///
    /// Wraps [`step_regulate`](Self::step_regulate) with phantom type tags.
    /// Accepts tagged prosody outputs and text features from the correct
    /// upstream steps.
    pub fn step_regulate_typed(
        &mut self,
        dur_logits: &PipelineTensor<ProsodyDurLogitsOutput>,
        features: &PipelineTensor<ProsodyFeaturesOutput>,
        text_features: &PipelineTensor<TextFeaturesOutput>,
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<TypedRegulateResult, CompiledKokoroError> {
        let result = self.step_regulate(
            dur_logits.inner(),
            features.inner(),
            text_features.inner(),
            speed,
            cache,
        )?;
        Ok(TypedRegulateResult {
            durations: result.durations,
            aligned_dur: PipelineTensor::new(result.aligned_dur),
            regulated: PipelineTensor::new(result.regulated),
            t_mel: result.t_mel,
        })
    }

    /// Type-safe Step 5: Run F0EnergyPredictor.
    ///
    /// Wraps [`step_predict_f0_energy`](Self::step_predict_f0_energy) with
    /// phantom type tags. Accepts `PipelineTensor<AlignedDurOutput>` from
    /// the regulate step.
    pub fn step_predict_f0_energy_typed(
        &mut self,
        aligned_dur: &PipelineTensor<AlignedDurOutput>,
        prosody_style: &DynTensor,
        t_mel: usize,
        cache: &PipelineCache,
    ) -> Result<TypedF0EnergyResult, CompiledKokoroError> {
        let result =
            self.step_predict_f0_energy(aligned_dur.inner(), prosody_style, t_mel, cache)?;
        Ok(TypedF0EnergyResult {
            f0: PipelineTensor::new(result.f0),
            energy: PipelineTensor::new(result.energy),
        })
    }

    /// Type-safe Step 6: Build harmonic source from F0 predictions.
    ///
    /// Wraps [`step_harmonic_source`](Self::step_harmonic_source) with
    /// phantom type tags. Accepts tagged F0 and energy outputs.
    pub fn step_harmonic_source_typed(
        &mut self,
        f0: &PipelineTensor<F0Output>,
        energy: &PipelineTensor<EnergyOutput>,
        t_mel: usize,
        cache: &PipelineCache,
    ) -> Result<PipelineTensor<HarmonicSourceOutput>, CompiledKokoroError> {
        let result = self.step_harmonic_source(f0.inner(), energy.inner(), t_mel, cache)?;
        Ok(PipelineTensor::new(result))
    }

    /// Type-safe Step 7: Run Generator / FullDecoder.
    ///
    /// Wraps [`step_generate`](Self::step_generate) with phantom type tags.
    /// All five inputs are type-checked to come from the correct upstream steps.
    pub fn step_generate_typed(
        &mut self,
        regulated: &PipelineTensor<RegulatedOutput>,
        f0: &PipelineTensor<F0Output>,
        energy: &PipelineTensor<EnergyOutput>,
        decoder_style: &DynTensor,
        har_source: &PipelineTensor<HarmonicSourceOutput>,
        t_mel: usize,
        cache: &PipelineCache,
    ) -> Result<TypedGeneratorResult, CompiledKokoroError> {
        let result = self.step_generate(
            regulated.inner(),
            f0.inner(),
            energy.inner(),
            decoder_style,
            har_source.inner(),
            t_mel,
            cache,
        )?;
        Ok(TypedGeneratorResult {
            magnitude: PipelineTensor::new(result.magnitude),
            phase: PipelineTensor::new(result.phase),
        })
    }

    /// Type-safe Step 8: Run GPU iSTFT.
    ///
    /// Wraps [`step_istft`](Self::step_istft) with phantom type tags.
    /// Accepts tagged magnitude and phase outputs from the generator step.
    pub fn step_istft_typed(
        &mut self,
        magnitude: &PipelineTensor<MagnitudeOutput>,
        phase: &PipelineTensor<PhaseOutput>,
        cache: &PipelineCache,
    ) -> Result<PipelineTensor<IstftOutput>, CompiledKokoroError> {
        let result = self.step_istft(magnitude.inner(), phase.inner(), cache)?;
        Ok(PipelineTensor::new(result))
    }
}
