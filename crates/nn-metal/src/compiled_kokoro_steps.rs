// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Step-by-step execution API for [`CompiledKokoro`].
//!
//! Exposes each pipeline stage as an independent method, allowing callers
//! (e.g. dvoice) to inject prosody hooks between stages. Each step
//! returns intermediate tensors that can be inspected or modified before
//! passing to the next step.
//!
//! # Hook points
//!
//! Between `step_predict_prosody` and `step_regulate`: modify duration logits.
//! Between `step_predict_f0_energy` and `step_harmonic_source`: modify F0/energy.
//! After `step_istft`: per-phoneme volume, break insertion.
//!
//! # Arena safety contract (#2638, #2739, #4264)
//!
//! [`synthesize()`](super::CompiledKokoro::synthesize) runs Steps 1-8 inside
//! `NanCheckPolicy::Skip` (#2981 scope narrowing), so per-step
//! `check_output_finite` calls are no-ops. Model-boundary validation
//! (`any_non_finite()` + `step_verify`) runs outside the Skip scope.
//! Two mechanisms prevent stale reads independent of NaN check flushes:
//!
//! 1. **`to_standalone()` GPU blit** on outputs of `step_encode`/`step_regulate`/
//!    `step_harmonic_source`/`step_istft` — outputs are blit-copied GPU→GPU to
//!    standalone buffers (no CPU roundtrip, #4279) that bypass stale-read detection.
//!    Intermediates use the arena for efficiency; only final outputs are promoted.
//!    `step_regulate` reaches its 4-byte scalar readback via `submit()+sync()`
//!    (GPU prefix-sum, #2911 Phase 2).
//! 2. **`MetalTensorData::new()` compiled segment outputs** — `step_predict_prosody`,
//!    `step_predict_f0_energy`, and `step_generate` outputs set
//!    `arena_generation: None`, making them immune to stale-read detection.
//!
//! # Production blit elision (#4264)
//!
//! The production pipeline (`synthesize`/`synthesize_gpu`/`synthesize_pipelined`)
//! uses `_production` variants of steps 4, 6, and 8 that skip `to_standalone()`
//! blits. This is safe because:
//! - **No flush/sync between step 4 scatter and pipeline exit.** Arena-resident
//!   tensors from `step_regulate` (allocated after the prefix-sum sync) survive
//!   until the pipeline-exit `to_device(&cpu())` flush. `GpuFence::wait()` does
//!   NOT reset the arena.
//! - **GPU dispatch reads raw buffer pointers.** Stale-read detection (#2328)
//!   only fires on `to_device(&cpu())`; GPU-to-GPU consumption is always safe.
//! - **step_istft audio:** The pipeline-exit `flush()` advances arena gen by 1,
//!   placing the audio tensor in the "just-committed batch" safe window.
//! - **step_encode outputs:** Already standalone (`MetalTensorData::new()`), so
//!   `to_standalone()` is a cheap alias (no blit) — not worth eliding.
//!
//! The **public step API** (step_regulate, step_harmonic_source, step_istft)
//! retains all blits for safety. Standalone callers and the chorus path may
//! hold tensors across arena resets and need standalone buffers.
//!
//! **`_no_fence` compiled segment execution** (#2739) — compiled segments
//! use `execute_dyn_no_fence`/`execute_dyn_outputs_no_fence`, encoding GPU
//! work into the lazy batch. GPU work synchronizes at inherent sync points
//! (`step_regulate` prefix-sum readback, pipeline-exit `to_device(&cpu())`).
//!
//! **Standalone callers** (outside `synthesize()`) get active NaN checks by
//! default (`NanCheckPolicy::Always`). Each step's `check_output_finite`
//! triggers a GPU readback flush (~5-10μs). This is useful for debugging.
//!
//! Part of #2527, #2218, #2638, #2739, #2981, #4264.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{check_output_finite, Module};
use nn_core::{DType, TensorError};
use nn_models::kokoro_tts::split_style_embedding;

use nn_tts_verify::Certificate;

use super::{
    check_multi_output, cpu, generator_total_samples, gpu, model_device, seg_cache_miss,
    validate_input_ids, CompiledKokoro, CompiledKokoroError,
};
use super::{
    StepEncodeResult, StepF0EnergyResult, StepGeneratorResult, StepProsodyResult, StyleSplit,
};
use crate::cache::PipelineCache;

impl CompiledKokoro {
    /// Split a `[B, 2*style_dim]` voice embedding into decoder and prosody halves.
    ///
    /// Returns `(decoder_style, prosody_style)`, each `[B, style_dim]`.
    pub fn split_style(&self, style: &DynTensor) -> Result<StyleSplit, CompiledKokoroError> {
        let style_dim = self.shared.config.style_dim;
        let (decoder_style, prosody_style) = split_style_embedding(style, style_dim)?;
        Ok(StyleSplit::new(decoder_style, prosody_style))
    }

    /// Step 1+2: Encode input tokens via PlBert+bert_encoder and TextEncoder.
    ///
    /// PlBert+bert_encoder run as compiled segment 0 (#2744). TextEncoder runs
    /// as compiled segment 1. Both cached by sequence length.
    /// Arena: **standalone** (see [safety contract](self)).
    ///
    /// # Arguments
    ///
    /// * `input_ids` — `[B, T]` token indices (U32 or F32).
    /// * `cache` — Metal pipeline cache.
    pub fn step_encode(
        &mut self,
        input_ids: &DynTensor,
        cache: &PipelineCache,
    ) -> Result<StepEncodeResult, CompiledKokoroError> {
        validate_input_ids(input_ids, self.config().plbert.max_position_embeddings)?;

        // bert_features and text_features must survive across later step
        // executions and any GPU flushes (arena generation resets). Use
        // to_standalone() to GPU-blit outputs to standalone buffers, allowing
        // intermediates to use the arena. (#2574, #2632, #4279)
        //
        // Convert I64 token IDs to F32 on CPU first, then transfer to GPU.
        // to_dtype before to_device: GPU does not support I64 transfer (#2981).
        let dev = model_device(self.shared.model.as_ref());
        let ids_f32 = input_ids.to_dtype(DType::F32)?.to_device(&dev)?;
        let seq_len = ids_f32.dims()[1];

        // Segment 0: PlBert + bert_encoder (compiled GPU, #2744).
        // Position and token-type embeddings depend only on seq_len (the
        // segment cache key). Cache GPU-resident results to eliminate 2
        // CPU→GPU transfers on subsequent calls with the same length (#2912).
        let (pos_emb_gpu, type_emb_gpu) =
            if let Some(cached) = self.plbert_emb_cache.get(&seq_len) {
                cached.clone()
            } else {
                let plbert = self.shared.model()?.plbert();
                let seq_len_u32 =
                    u32::try_from(seq_len).map_err(|_| TensorError::ValueOutOfRange {
                        description: "PlBert seq_len exceeds u32::MAX",
                    })?;
                let position_ids =
                    DynTensor::arange_u32(0, seq_len_u32, &cpu())?.to_device(&dev)?;
                let pos_emb = plbert
                    .position_embeddings()
                    .forward(&position_ids)?
                    .unsqueeze(0)?
                    .to_dtype(DType::F32)?;
                let token_type_ids =
                    DynTensor::zeros(&[seq_len], DType::U32, &cpu())?.to_device(&dev)?;
                let type_emb = plbert
                    .token_type_embeddings()
                    .forward(&token_type_ids)?
                    .unsqueeze(0)?
                    .to_dtype(DType::F32)?;
                let pos_gpu = pos_emb.to_device(&gpu())?;
                let type_gpu = type_emb.to_device(&gpu())?;
                self.plbert_emb_cache
                    .insert(seq_len, (pos_gpu.clone(), type_gpu.clone()));
                (pos_gpu, type_gpu)
            };

        let seg_plbert = self.ensure_seg_plbert(seq_len, &ids_f32, cache)?;
        let ids_gpu = ids_f32.to_device(&gpu())?;
        let bert_features = seg_plbert
            .execute_dyn_no_fence(cache, &[&ids_gpu, &pos_emb_gpu, &type_emb_gpu])
            .map_err(|e| CompiledKokoroError::SegmentExecutionFailed {
                segment: "plbert",
                source: Box::new(e),
            })?;
        check_output_finite(&bert_features, "step:plbert")?;

        // Segment 1: TextEncoder (compiled GPU, uses its own Embedding).
        // Reuse ids_gpu from PlBert — same data, avoids redundant to_device().
        let seg_text = self.ensure_seg_text(seq_len, &ids_f32, cache)?;
        let text_features = seg_text
            .execute_dyn_no_fence(cache, &[&ids_gpu])
            .map_err(|e| CompiledKokoroError::SegmentExecutionFailed {
                segment: "text",
                source: Box::new(e),
            })?;
        check_output_finite(&text_features, "step:text_encoder")?;

        // GPU blit to standalone buffers — no CPU roundtrip (#4279).
        let bert_features = crate::to_standalone::to_standalone(&bert_features)?;
        let text_features = crate::to_standalone::to_standalone(&text_features)?;

        Ok(StepEncodeResult::new(bert_features, text_features, seq_len))
    }

    /// Step 3: Run ProsodyPredictor (compiled GPU, multi-output).
    ///
    /// Returns duration logits and prosody features. Duration logits can be
    /// modified before passing to [`step_regulate`] (prosody hook point).
    /// Arena: **arena-allocated** (see [safety contract](self)).
    ///
    /// # Arguments
    ///
    /// * `bert_features` — `[B, d_en, T]` from [`step_encode`].
    /// * `prosody_style` — `[B, style_dim]` prosody half from [`split_style`].
    /// * `seq_len` — from [`StepEncodeResult::seq_len`].
    /// * `cache` — Metal pipeline cache.
    pub fn step_predict_prosody(
        &mut self,
        bert_features: &DynTensor,
        prosody_style: &DynTensor,
        seq_len: usize,
        cache: &PipelineCache,
    ) -> Result<StepProsodyResult, CompiledKokoroError> {
        let seg_prosody = self.ensure_seg_prosody(seq_len, bert_features, prosody_style, cache)?;
        let style_gpu = prosody_style.to_device(&gpu())?;
        let bert_features_gpu = bert_features.to_device(&gpu())?;
        let outputs = seg_prosody
            .execute_dyn_outputs_no_fence(cache, &[&bert_features_gpu, &style_gpu])
            .map_err(|e| CompiledKokoroError::SegmentExecutionFailed {
                segment: "prosody",
                source: Box::new(e),
            })?;

        check_multi_output(&outputs, 2, "prosody")?;
        check_output_finite(&outputs[0], "step:prosody_dur_logits")?;
        check_output_finite(&outputs[1], "step:prosody_features")?;

        Ok(StepProsodyResult::new(
            outputs[0].clone(),
            outputs[1].clone(),
        ))
    }

    /// Step 5: Run F0EnergyPredictor (compiled GPU, multi-output).
    ///
    /// Returns F0 and energy predictions. These can be modified before passing
    /// to [`step_harmonic_source`] and [`step_generate`] (second hook point).
    /// Arena: **arena-allocated** (see [safety contract](self)).
    ///
    /// # Arguments
    ///
    /// * `aligned_dur` — `[B, d_en+style_dim, T_mel]` from [`StepRegulateResult`].
    /// * `prosody_style` — `[B, style_dim]` prosody half from [`split_style`].
    /// * `t_mel` — from [`StepRegulateResult::t_mel`].
    /// * `cache` — Metal pipeline cache.
    pub fn step_predict_f0_energy(
        &mut self,
        aligned_dur: &DynTensor,
        prosody_style: &DynTensor,
        t_mel: usize,
        cache: &PipelineCache,
    ) -> Result<StepF0EnergyResult, CompiledKokoroError> {
        let seg_f0 = self.ensure_seg_f0(t_mel, aligned_dur, prosody_style, cache)?;
        let aligned_dur_gpu = aligned_dur.to_device(&gpu())?;
        let style_gpu = prosody_style.to_device(&gpu())?;
        let outputs = seg_f0
            .execute_dyn_outputs_no_fence(cache, &[&aligned_dur_gpu, &style_gpu])
            .map_err(|e| CompiledKokoroError::SegmentExecutionFailed {
                segment: "f0",
                source: Box::new(e),
            })?;

        check_multi_output(&outputs, 2, "f0")?;
        check_output_finite(&outputs[0], "step:f0_prediction")?;
        check_output_finite(&outputs[1], "step:energy_prediction")?;

        Ok(StepF0EnergyResult::new(
            outputs[0].clone(),
            outputs[1].clone(),
        ))
    }

    /// Step 6: Build harmonic source from F0 predictions.
    ///
    /// Uses full 9-harmonic SineGen when SourceModule weights are loaded.
    /// Returns `MissingSourceModule` error when weights are absent (#2667).
    /// Arena: **standalone** (see [safety contract](self)).
    ///
    /// # Arguments
    ///
    /// * `f0` — `[B, 1, 2*T_mel]` from [`StepF0EnergyResult`].
    /// * `energy` — `[B, 1, 2*T_mel]` from [`StepF0EnergyResult`].
    /// * `t_mel` — from [`StepRegulateResult::t_mel`].
    /// * `cache` — Metal pipeline cache.
    pub fn step_harmonic_source(
        &mut self,
        f0: &DynTensor,
        energy: &DynTensor,
        _t_mel: usize,
        cache: &PipelineCache,
    ) -> Result<DynTensor, CompiledKokoroError> {
        let n_fft = self.shared.config.n_fft;
        let upsample_rates_product: usize = self.shared.config.upsample_rates.iter().product();
        // harmonic_source must survive across later step executions and any
        // GPU flushes (arena generation resets). Intermediates use the arena;
        // only the final output is blit-copied to standalone. (#2574, #4279)
        let har_source =
            self.build_harmonic_source(f0, energy, n_fft, upsample_rates_product, cache)?;
        check_output_finite(&har_source, "step:harmonic_source")?;
        // GPU blit to standalone buffer — no CPU roundtrip (#4279).
        let har_source = crate::to_standalone::to_standalone(&har_source)?;
        Ok(har_source)
    }

    /// Production-path variant of [`step_harmonic_source`] that skips the
    /// `to_standalone()` GPU blit.
    ///
    /// In the production pipeline, `har_source` is consumed as a GPU input by
    /// `step_generate` (step 7). No `flush()` or `sync()` occurs between steps
    /// 6 and 7, so the arena-resident buffer remains valid. GPU dispatch reads
    /// raw buffer pointers without stale-read detection (#2328 only fires on
    /// `to_device(&cpu())`).
    ///
    /// Saves 1 GPU blit dispatch per synthesis call (#4264).
    pub(crate) fn step_harmonic_source_production(
        &mut self,
        f0: &DynTensor,
        energy: &DynTensor,
        _t_mel: usize,
        cache: &PipelineCache,
    ) -> Result<DynTensor, CompiledKokoroError> {
        let n_fft = self.shared.config.n_fft;
        let upsample_rates_product: usize = self.shared.config.upsample_rates.iter().product();
        let har_source =
            self.build_harmonic_source(f0, energy, n_fft, upsample_rates_product, cache)?;
        check_output_finite(&har_source, "step:harmonic_source")?;
        // Production path: skip blit. har_source is consumed as GPU input
        // by step_generate — no CPU readback, no arena reset before use (#4264).
        Ok(har_source)
    }

    /// Step 7: Run Generator / FullDecoder (compiled GPU, multi-output).
    ///
    /// Produces spectral magnitude and phase for iSTFT.
    /// Arena: **arena-allocated** (see [safety contract](self)).
    ///
    /// # Arguments
    ///
    /// * `regulated` — `[B, d_en, T_mel]` from [`StepRegulateResult`].
    /// * `f0` — `[B, 1, 2*T_mel]` from [`StepF0EnergyResult`].
    /// * `energy` — `[B, 1, 2*T_mel]` from [`StepF0EnergyResult`].
    /// * `decoder_style` — `[B, style_dim]` decoder half from [`split_style`].
    /// * `har_source` — harmonic source from [`step_harmonic_source`].
    /// * `t_mel` — from [`StepRegulateResult::t_mel`].
    /// * `cache` — Metal pipeline cache.
    pub fn step_generate(
        &mut self,
        regulated: &DynTensor,
        f0: &DynTensor,
        energy: &DynTensor,
        decoder_style: &DynTensor,
        har_source: &DynTensor,
        t_mel: usize,
        cache: &PipelineCache,
    ) -> Result<StepGeneratorResult, CompiledKokoroError> {
        // Read scalars before mutable borrow — avoids cloning entire KokoroConfig.
        let upsample_factor: usize = self.shared.config.upsample_rates.iter().product();
        let total_samples = generator_total_samples(t_mel, upsample_factor)?;

        // Compile segment on cache miss — transfer to model weight device
        // for tracing (#2743). LRU cache hits skip recompilation (#2626).
        // ensure_seg_generator uses get() internally (single scan on hit).
        if self.seg_generator.get(total_samples).is_none() {
            let dev = model_device(self.shared.model.as_ref());
            let regulated_dev = regulated.to_device(&dev)?;
            let f0_dev = f0.to_device(&dev)?;
            let energy_dev = energy.to_device(&dev)?;
            let dec_style_dev = decoder_style.to_device(&dev)?;
            let har_dev = har_source.to_device(&dev)?;
            self.ensure_seg_generator(
                total_samples,
                &regulated_dev,
                &f0_dev,
                &energy_dev,
                &dec_style_dev,
                &har_dev,
                cache,
            )?;
        }
        let seg_gen = self
            .seg_generator
            .most_recent()
            .map(|(_, v)| v)
            .ok_or_else(|| seg_cache_miss("generator"))?;

        let dec_style_gpu = decoder_style.to_device(&gpu())?;
        let regulated_gpu = regulated.to_device(&gpu())?;
        // Ensure all inputs are on GPU — callers may have modified tensors on CPU
        // (the whole point of the step API is to allow intermediate modification).
        // to_device is a no-op when already on GPU.
        let f0_gpu = f0.to_device(&gpu())?;
        let energy_gpu = energy.to_device(&gpu())?;
        let har_source_gpu = har_source.to_device(&gpu())?;
        // Fast-half accumulator scope: when the autocast config has both
        // `generator` enabled and `use_fast_half_accumulator` set, FusedResBlock
        // conv kernels use half-precision accumulators (~2x throughput vs ~1.36x
        // for float-accumulator F16). The thread-local flag is read by the fused
        // conv dispatch path in norm_conv_stats / norm_conv_fused.
        let fast_half = self
            .segment_autocast
            .as_ref()
            .is_some_and(|c| c.generator && c.use_fast_half_accumulator);
        let outputs = crate::dyn_tensor_metal::with_fast_half_scope(fast_half, || {
            seg_gen.execute_dyn_outputs_no_fence(
                cache,
                &[
                    &regulated_gpu,
                    &f0_gpu,
                    &energy_gpu,
                    &dec_style_gpu,
                    &har_source_gpu,
                ],
            )
        })
        .map_err(|e| CompiledKokoroError::SegmentExecutionFailed {
            segment: "generator",
            source: Box::new(e),
        })?;

        check_multi_output(&outputs, 2, "generator")?;
        check_output_finite(&outputs[0], "step:magnitude")?;
        check_output_finite(&outputs[1], "step:phase")?;

        Ok(StepGeneratorResult::new(
            outputs[0].clone(),
            outputs[1].clone(),
        ))
    }

    /// Step 8: Run GPU iSTFT on magnitude + phase to produce PCM audio.
    ///
    /// Output is GPU-resident — no flush or CPU readback. The caller
    /// transfers to CPU when needed (e.g., `to_device(&cpu())` in
    /// `synthesize()` at the pipeline boundary).
    ///
    /// # Arguments
    ///
    /// * `magnitude` — from [`StepGeneratorResult`].
    /// * `phase` — from [`StepGeneratorResult`].
    /// * `cache` — Metal pipeline cache.
    pub fn step_istft(
        &mut self,
        magnitude: &DynTensor,
        phase: &DynTensor,
        cache: &PipelineCache,
    ) -> Result<DynTensor, CompiledKokoroError> {
        // Audio tensors must survive across later step executions (chorus
        // shared-encode, GPU batch synth). Intermediates use the arena;
        // only the final clamped PCM is blit-copied to standalone. (#4279)
        let n_fft = self.shared.config.n_fft;
        let audio = self.gpu_istft(magnitude, phase, n_fft, cache)?;
        let audio = audio.clamp(-1.0, 1.0)?;
        // GPU blit to standalone buffer — no CPU roundtrip (#4279).
        Ok(crate::to_standalone::to_standalone(&audio)?)
    }

    /// Production-path variant of [`step_istft`] that skips the
    /// `to_standalone()` GPU blit.
    ///
    /// In the production pipeline, the audio tensor is the LAST GPU step output.
    /// It is consumed by `to_device(&cpu())` at the pipeline exit, which calls
    /// `flush()`. The flush commits all pending GPU work and resets the arena.
    /// After flush, the stale-read check sees alloc_gen=N vs current_gen=N+1
    /// with used_bytes=0 — which is the "just-committed batch" safe case (#2328).
    ///
    /// Saves 1 GPU blit dispatch per synthesis call (#4264).
    pub(crate) fn step_istft_production(
        &mut self,
        magnitude: &DynTensor,
        phase: &DynTensor,
        cache: &PipelineCache,
    ) -> Result<DynTensor, CompiledKokoroError> {
        let n_fft = self.shared.config.n_fft;
        let audio = self.gpu_istft(magnitude, phase, n_fft, cache)?;
        let audio = audio.clamp(-1.0, 1.0)?;
        // Production path: skip blit. Audio is the last GPU output; the
        // pipeline-exit to_device(&cpu()) flushes and reads it safely (#4264).
        Ok(audio)
    }

    /// Step 9: Verify audio quality (hard bounds — < 1ms overhead).
    ///
    /// Transfers audio to CPU and runs 7 hard bound property checks.
    ///
    /// # Arguments
    ///
    /// * `audio` — `[1, 1, T_audio]` PCM at 24 kHz from [`step_istft`].
    pub fn step_verify(&self, audio: &DynTensor) -> Result<Certificate, CompiledKokoroError> {
        let pcm = audio.to_device(&cpu())?.to_flat_vec::<f32>()?;
        let certificate = self
            .shared
            .verifier
            .verify(&pcm)
            .map_err(|e| CompiledKokoroError::VerificationFailed {
                source: Box::new(e),
            })?;

        // Attach CROWN verification evidence when enabled (#4254, #3874).
        if self.crown_verification {
            let result = nn_tts_verify::verify_synthesis_crown_full(
                &certificate,
                &self.crown_config,
                None, // intermediates — populated by step-level callers
            );
            let certificate = certificate.with_crown_evidence(result.moonshot);
            let certificate = if let Some(summary) = result.junction_summary {
                certificate.with_junction_summary(summary)
            } else {
                certificate
            };
            Ok(certificate)
        } else {
            Ok(certificate)
        }
    }
}
