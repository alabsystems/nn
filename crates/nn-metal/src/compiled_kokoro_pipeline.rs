// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Synthesize pipeline for [`CompiledKokoro`].
//!
//! Delegates to the step API (`compiled_kokoro_steps.rs`) for each pipeline
//! stage. The step API is the single source of truth for pipeline logic;
//! this method is a thin orchestrator that chains the steps without hooks.
//!
//! # GPU-resident synthesis (#4251)
//!
//! [`synthesize_gpu_inner`] is the core implementation that returns both a
//! [`GpuAudioHandle`] and the CPU audio tensor (computed once for NaN guard +
//! verification). [`synthesize_gpu`] wraps it dropping the CPU tensor;
//! [`synthesize`] wraps it reusing the CPU tensor directly — no double
//! readback (#4264).
//!
//! # Segment Fusion (#2739)
//!
//! Compiled segments use `execute_dyn_no_fence` / `execute_dyn_outputs_no_fence`,
//! encoding GPU work into the lazy batch without triggering submits.
//!
//! # CPU-GPU Overlap (#4264)
//!
//! `synthesize_gpu_inner` inserts `GpuFence::submit_current()` at strategic
//! pipeline points. Phase 1 (encode, prosody) uses per-step submits. Phase 2
//! (f0, harmonic, generator, iSTFT) uses batched submits: f0+harmonic are
//! encoded into one command buffer, generator+iSTFT into another. This reduces
//! Phase 2 from 3 intermediate submits to 1, saving 2 command buffer creation
//! overheads (~10-50us each on M4 Max). Metal command queue ordering guarantees
//! sequential execution of submitted batches. `GpuFence` does NOT reset the
//! arena, so arena-resident tensors from earlier segments remain valid. The
//! final `to_device(&cpu())` commits the last batch and waits, serving as a
//! global barrier for all prior batches.
//!
//! # NanCheckPolicy scope narrowing (#2981)
//!
//! Steps 1-8 run inside `NanCheckPolicy::Skip`, eliminating ~14 per-step
//! `check_output_finite` GPU→CPU readback flushes. Model-boundary validation
//! (`any_non_finite()` + `step_verify`) runs OUTSIDE the Skip scope and
//! catches NaN/Inf from any step. This matches the pattern used by HTDemucs
//! (`htdemucs_forward.rs`) and Silero VAD (`silero_vad_forward.rs`).
//!
//! Sync points (cache-hit hot path): `step_regulate` 4-byte scalar readback
//! for GPU prefix-sum (#2911) and the pipeline-exit `to_device(&cpu())` which
//! flushes all pending GPU work in a single commit. Steps 1-8 are fully GPU —
//! no mid-pipeline CPU readback. Per-step `check_output_finite` readbacks are
//! eliminated by Skip.
//!
//! `step_harmonic_source`: Fully GPU-native with 2 compiled sub-segments (#1815 D2-D5):
//!  - `seg_sinegen_pre`: F0 → rad_frames + voiced mask (compiled, ~6 dispatches).
//!  - Eager: `cumsum_kahan` barrier (1 dispatch, custom Metal kernel).
//!  - `seg_sinegen_post`: phase → sin → linear+tanh → transpose (compiled, ~7 dispatches).
//!  - Eager: forward STFT via GPU mixed-radix FFT (Good-Thomas PFA 4x5,
//!    [`StftGpuBasis::forward_cat_center_fft`]).
//!
//! No CPU round-trip. (#2691, #2909, #1815)
//!
//! # Production blit elision (#4264)
//!
//! Production pipelines use `_production` step variants that skip
//! `to_standalone()` GPU blits on steps 4 and 6. `synthesize_pipelined` also
//! skips the step 8 blit (no GpuAudioHandle needed). `synthesize_gpu_inner`
//! retains the step 8 blit because [`GpuAudioHandle`] must survive arena
//! resets across subsequent calls. Arena-resident tensors survive because no
//! `flush()`/`sync()` occurs between step 4's scatter outputs and the
//! pipeline-exit `to_device(&cpu())`. Saves 4 GPU blit dispatches on
//! `synthesize`/`synthesize_gpu` (3 in step_regulate + 1 harmonic) and 5 on
//! `synthesize_pipelined` (+ 1 iSTFT).
//! See `compiled_kokoro_steps.rs` doc for the full safety argument.
//!
//! Part of #2739, #2632, #2218, #2928, #2981, #4251, #4264.

use nn_core::layers::{with_nan_check_policy, NanCheckPolicy};

use super::arena_report::{build_arena_report, snapshot_arena_pre, KokoroArenaReport};
use super::*;
use crate::arena::with_decode_scope;
use crate::gpu_fence::GpuFence;

/// Kokoro sample rate in Hz.
const KOKORO_SAMPLE_RATE: u32 = 24000;

/// Internal result from [`CompiledKokoro::synthesize_gpu_inner`].
///
/// Contains both the GPU handle and the already-computed CPU audio tensor,
/// so callers that need CPU audio (e.g., [`synthesize`](CompiledKokoro::synthesize))
/// can reuse it without a second GPU-to-CPU readback (#4264).
struct SynthesizeGpuOutput {
    handle: crate::GpuAudioHandle,
    cpu_audio: DynTensor,
    certificate: Certificate,
}

impl CompiledKokoro {
    /// Synthesize PCM audio from token IDs and style vector, returning a
    /// GPU-resident audio handle.
    ///
    /// The returned [`GpuAudioHandle`] wraps the raw GPU buffer produced by
    /// the iSTFT step. Audio quality verification (certificate) is computed
    /// by reading the audio to CPU internally, but the handle retains the
    /// GPU buffer for callers that want to avoid a second readback (e.g.,
    /// passing audio directly to another GPU pipeline or Metal audio playback).
    ///
    /// # Arguments
    ///
    /// * `input_ids` - `[B, T]` token indices (U32 or F32).
    /// * `style` - `[B, 2*style_dim]` voice embedding (first half = decoder,
    ///   second half = prosody).
    /// * `speed` - Speaking rate multiplier (1.0 = normal).
    /// * `cache` - Metal pipeline cache.
    ///
    /// # Returns
    ///
    /// `(GpuAudioHandle, Certificate)` — the GPU-resident audio buffer at
    /// 24 kHz plus a TTS quality certificate proving 7 hard bound properties.
    /// Call [`GpuAudioHandle::to_cpu()`] or [`GpuAudioHandle::to_cpu_tensor()`]
    /// for explicit CPU readback.
    ///
    /// # Compilation
    ///
    /// On first call (or when input shapes change), segments are traced and
    /// compiled to GPU dispatch plans. Compiled segments are cached by shape
    /// for reuse on subsequent calls.
    ///
    /// # Errors
    ///
    /// Returns [`CompiledKokoroError`] with structured variants:
    /// - [`InvalidSpeed`](CompiledKokoroError::InvalidSpeed) - speed not positive/finite.
    /// - [`SegmentExecutionFailed`](CompiledKokoroError::SegmentExecutionFailed) - GPU segment dispatch error.
    /// - [`VerificationFailed`](CompiledKokoroError::VerificationFailed) - audio quality check error.
    /// - [`Tensor`](CompiledKokoroError::Tensor) - underlying tensor operation error.
    ///
    /// Part of #4251.
    /// Core GPU synthesis returning both GPU handle and CPU audio.
    ///
    /// Both [`synthesize_gpu`] and [`synthesize`] delegate to this method.
    /// The CPU audio tensor is computed once (single GPU flush + readback)
    /// for NaN guard and verification. Callers that need CPU audio reuse it
    /// directly, avoiding the double readback that existed before #4264.
    fn synthesize_gpu_inner(
        &mut self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<SynthesizeGpuOutput, CompiledKokoroError> {
        let style_split = prepare_synthesis_inputs(self, input_ids, style, speed)?;
        // Upload both style halves to GPU once. Step functions call to_device(&gpu())
        // which becomes a no-op when data is already GPU-resident (#2912).
        let decoder_style = style_split.decoder_style.to_device(&gpu())?;
        let prosody_style = style_split.prosody_style.to_device(&gpu())?;

        // Pre-size the arena to avoid growth during the forward pass (#4289).
        // estimate_arena_bytes() returns 0 on first call (no compiled segments yet),
        // so this is a no-op until segments are warmed. On subsequent calls the
        // arena capacity matches the model's actual intermediate buffer needs.
        let arena_estimate = self.estimate_arena_bytes();
        if arena_estimate > 0 {
            let _ = crate::arena::ensure_default_arena_capacity(
                cache.context(),
                arena_estimate,
            );
        }

        // NanCheckPolicy scope narrowing (#2981): Steps 1-8 run inside
        // NanCheckPolicy::Skip, eliminating ~14 per-step check_output_finite
        // GPU→CPU readback flushes. Model-boundary validation (any_non_finite
        // + step_verify) runs OUTSIDE the Skip scope as defense-in-depth.
        // Matches HTDemucs (htdemucs_forward.rs:70) and Silero VAD
        // (silero_vad_forward.rs:92).
        //
        // Arena safety: without_arena wrappers (step_encode, step_regulate,
        // step_harmonic_source) produce standalone buffers. Compiled segment
        // outputs use MetalTensorData::new() with arena_generation: None.
        // Neither depends on check_output_finite flushes for correctness.
        //
        // Sync points: (1) step_regulate prefix-sum 4-byte scalar readback.
        // step_istft is fully GPU — the pipeline-exit to_device(&cpu())
        // commits all pending GPU work in a single flush. SineGen cumsum is
        // GPU-native (Kahan-compensated, #2909).

        // Decode scope: suppress stale-arena-read checks for the entire
        // pipeline. Production Kokoro (D=512) exceeds the 128-encoding auto-
        // flush threshold, triggering mid-pipeline arena resets from sync()
        // (inside step_regulate's submit+sync for prefix-sum readback) and
        // flush() (inside to_device(&cpu()) at pipeline exit). Without the
        // scope, tensors allocated early in the pipeline (generation N) are
        // flagged as stale when gpu_to_cpu runs after the arena has advanced
        // to generation N+K. ObjC ARC keeps Metal buffers alive — the stale
        // check is defense-in-depth, not memory safety. Part of #4264.
        let result = with_decode_scope(||
            (|| -> Result<SynthesizeGpuOutput, CompiledKokoroError> {
                // Steps 1-8: NaN-skip scope eliminates per-step GPU readbacks.
                // check_output_finite inside step functions becomes a no-op.
                let gpu_audio = with_nan_check_policy(NanCheckPolicy::Skip, || {
                    // ---- Pre-readback phase (steps 1-4) ----
                    // ICB replay: when enabled and a cached ICB exists for this
                    // seq_len, the per-CompiledModel ICB replay within each
                    // segment handles the dispatch optimization. The pipeline-
                    // level tracking here records shape keys so the replay buffer
                    // can correlate pre/post-readback phases. (#4264)

                    // Steps 1-2: PlBert + bert_encoder + TextEncoder
                    let enc = self.step_encode(input_ids, cache)?;

                    // Step 3: ProsodyPredictor
                    // Encode and prosody dispatch commands are batched into a
                    // single lazy batch (no intermediate GpuFence::submit_current
                    // between them). Consolidation saves one Metal command buffer
                    // creation (~10-50us on M4 Max). The regulate step below has
                    // its own submit()+sync() which flushes all pending
                    // encode+prosody GPU work before the 4-byte scalar readback.
                    // Metal command queue ordering guarantees correct execution
                    // order within the single command buffer. Part of #4264.
                    let pros = self.step_predict_prosody(
                        &enc.bert_features,
                        &prosody_style,
                        enc.seq_len,
                        cache,
                    )?;

                    // Submit encode+prosody GPU work as a single batch. The GPU
                    // starts executing both segments while the CPU prepares
                    // regulate encoding. One submit instead of two saves one
                    // command buffer creation overhead. Part of #4264
                    // (encode+prosody batch consolidation).
                    let _fence_encode_prosody = GpuFence::submit_current()?;

                    // Step 4: Duration + length_regulate (compiled segment 5).
                    // Production path: skip all blits (#4264).
                    // step_regulate has its own submit()+sync() for the 4-byte
                    // prefix-sum readback. Metal queue ordering ensures the
                    // encode+prosody batch completes before regulate's sync
                    // reads the scalar.
                    let reg = self.step_regulate_production(
                        &pros.dur_logits,
                        &pros.features,
                        &enc.text_features,
                        speed,
                        cache,
                    )?;

                    // ICB replay: notify pre-readback phase complete for shape
                    // tracking. The regulate_scalar_readback sync point splits
                    // the pipeline into two independent replay phases. (#4264)
                    self.notify_pre_readback_complete(enc.seq_len);

                    // ---- Post-readback phase (steps 5-8) ----
                    // Batched submit strategy: f0+harmonic in one command buffer,
                    // generator+iSTFT in another. Reduces Phase 2 from 3
                    // intermediate submits to 1. Part of #4264.

                    // Step 5: F0EnergyPredictor
                    let f0e = self.step_predict_f0_energy(
                        &reg.aligned_dur,
                        &prosody_style,
                        reg.t_mel,
                        cache,
                    )?;

                    // Step 6: Harmonic source — skip blit (#4264).
                    // Batched with f0 in same lazy batch (no intermediate submit).
                    let har_source = self.step_harmonic_source_production(
                        &f0e.f0, &f0e.energy, reg.t_mel, cache,
                    )?;

                    // Single post-readback submit: f0+harmonic together.
                    // Generator is the heaviest segment (~46 dispatches), so
                    // overlapping its CPU encoding with f0+harmonic GPU execution
                    // is the highest-value overlap point.
                    let _fence_f0_harmonic = GpuFence::submit_current()?;

                    // Step 7: Generator / FullDecoder
                    let upsample_factor: usize =
                        self.shared.config.upsample_rates.iter().product();
                    let total_samples =
                        generator_total_samples(reg.t_mel, upsample_factor)?;
                    let generator = self.step_generate(
                        &reg.regulated,
                        &f0e.f0,
                        &f0e.energy,
                        &decoder_style,
                        &har_source,
                        reg.t_mel,
                        cache,
                    )?;

                    // No intermediate submit between generator and iSTFT.
                    // Both stay in the same lazy batch, committed at pipeline exit.
                    // Saves one command buffer creation (~10-50us on M4 Max).

                    // Step 8: GPU iSTFT -> PCM audio.
                    // NOTE: synthesize_gpu_inner returns a GpuAudioHandle that
                    // captures the GPU buffer reference. The audio MUST be
                    // standalone so the handle survives arena resets across
                    // subsequent synthesize() calls. Cannot use _production
                    // variant here. (#4264, #4251)
                    //
                    // The final to_device(&cpu()) below commits this batch and
                    // waits. Metal queue ordering guarantees all prior fence-
                    // submitted batches (encode, prosody, f0+harmonic) and the
                    // current batch (generator+iSTFT) complete together.
                    let audio = self.step_istft(&generator.magnitude, &generator.phase, cache)?;

                    // ICB replay: notify post-readback phase complete. (#4264)
                    self.notify_post_readback_complete(reg.t_mel, total_samples);

                    Ok::<_, CompiledKokoroError>(audio)
                })?;

                // Capture the GPU buffer BEFORE the CPU transfer so callers
                // can use the handle without a second readback (#4251).
                let handle = crate::GpuAudioHandle::from_dyn_tensor(
                    &gpu_audio,
                    KOKORO_SAMPLE_RATE,
                )
                .map_err(|e| {
                    let te: nn_core::TensorError = e.into();
                    CompiledKokoroError::Tensor(Box::new(te))
                })?;

                // Pipeline exit: single GPU→CPU transfer flushes all pending work.
                // This is the only non-regulate sync point — steps 1-8 are fully GPU.
                let cpu_audio = gpu_audio.to_device(&cpu())?;

                // Pipeline exit NaN guard (OUTSIDE Skip scope — defense-in-depth).
                if cpu_audio.any_non_finite()? {
                    let count = cpu_audio
                        .as_cpu_f32()
                        .map(|v| v.iter().filter(|x| !x.is_finite()).count())
                        .unwrap_or(1);
                    return Err(nn_core::TensorError::NonFiniteData {
                        name: "pipeline_output_audio".into(),
                        count,
                    }
                    .into());
                }

                // Step 9: Verify audio quality (already CPU, no-op transfer inside)
                let certificate = self.step_verify(&cpu_audio)?;

                Ok(SynthesizeGpuOutput { handle, cpu_audio, certificate })
            })()
        );

        // Clean up stale GPU commands on error. Without this, uncommitted
        // _no_fence commands from prior successful segments persist in the
        // thread-local lazy batch and contaminate the next synthesize() call.
        if result.is_err() {
            crate::gpu_scope::discard_pending_batch();
        }

        // Auto-release CPU model weights after first successful synthesis (#3079).
        // All 8 segments are now compiled for this shape — GPU buffers are
        // self-contained. Frees ~320 MB of CPU ArrayD<f32> weight data.
        // Silently ignores SharedOwnership (clone_dispatch instances exist).
        if result.is_ok() && self.auto_release && !self.weights_released() {
            let _ = self.release_model_weights();
        }

        result
    }

    /// Synthesize PCM audio from token IDs and style vector, returning a
    /// GPU-resident audio handle.
    ///
    /// The returned [`GpuAudioHandle`] wraps the raw GPU buffer produced by
    /// the iSTFT step. Audio quality verification (certificate) is computed
    /// by reading the audio to CPU internally, but the handle retains the
    /// GPU buffer for callers that want to avoid a second readback (e.g.,
    /// passing audio directly to another GPU pipeline or Metal audio playback).
    ///
    /// # Arguments
    ///
    /// * `input_ids` - `[B, T]` token indices (U32 or F32).
    /// * `style` - `[B, 2*style_dim]` voice embedding (first half = decoder,
    ///   second half = prosody).
    /// * `speed` - Speaking rate multiplier (1.0 = normal).
    /// * `cache` - Metal pipeline cache.
    ///
    /// # Returns
    ///
    /// `(GpuAudioHandle, Certificate)` — the GPU-resident audio buffer at
    /// 24 kHz plus a TTS quality certificate proving 7 hard bound properties.
    /// Call [`GpuAudioHandle::to_cpu()`] or [`GpuAudioHandle::to_cpu_tensor()`]
    /// for explicit CPU readback.
    ///
    /// # Compilation
    ///
    /// On first call (or when input shapes change), segments are traced and
    /// compiled to GPU dispatch plans. Compiled segments are cached by shape
    /// for reuse on subsequent calls.
    ///
    /// # Errors
    ///
    /// Returns [`CompiledKokoroError`] with structured variants:
    /// - [`InvalidSpeed`](CompiledKokoroError::InvalidSpeed) - speed not positive/finite.
    /// - [`SegmentExecutionFailed`](CompiledKokoroError::SegmentExecutionFailed) - GPU segment dispatch error.
    /// - [`VerificationFailed`](CompiledKokoroError::VerificationFailed) - audio quality check error.
    /// - [`Tensor`](CompiledKokoroError::Tensor) - underlying tensor operation error.
    ///
    /// Part of #4251.
    pub fn synthesize_gpu(
        &mut self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<(crate::GpuAudioHandle, Certificate), CompiledKokoroError> {
        let output = self.synthesize_gpu_inner(input_ids, style, speed, cache)?;
        Ok((output.handle, output.certificate))
    }

    /// Synthesize PCM audio from token IDs and style vector.
    ///
    /// Delegates to [`synthesize_gpu_inner`] and reuses the CPU audio tensor
    /// that was already computed for NaN guard + verification. This avoids a
    /// second GPU-to-CPU readback that the previous `synthesize_gpu` +
    /// `handle.to_cpu_tensor()` path incurred (#4264).
    ///
    /// For callers that want to keep audio on GPU (e.g., for Metal audio
    /// playback or passing to another GPU pipeline), use [`synthesize_gpu`]
    /// directly.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - `[B, T]` token indices (U32 or F32).
    /// * `style` - `[B, 2*style_dim]` voice embedding (first half = decoder,
    ///   second half = prosody).
    /// * `speed` - Speaking rate multiplier (1.0 = normal).
    /// * `cache` - Metal pipeline cache.
    ///
    /// # Returns
    ///
    /// `([1, 1, T_audio]` PCM audio at 24 kHz, [`Certificate`]) - the audio
    /// tensor plus a TTS quality certificate proving 7 hard bound properties.
    /// Check `certificate.overall_passed` for aggregate pass/fail.
    ///
    /// # Compilation
    ///
    /// On first call (or when input shapes change), segments are traced and
    /// compiled to GPU dispatch plans. Compiled segments are cached by shape
    /// for reuse on subsequent calls.
    ///
    /// # Errors
    ///
    /// Returns [`CompiledKokoroError`] with structured variants:
    /// - [`InvalidSpeed`](CompiledKokoroError::InvalidSpeed) - speed not positive/finite.
    /// - [`SegmentExecutionFailed`](CompiledKokoroError::SegmentExecutionFailed) - GPU segment dispatch error.
    /// - [`VerificationFailed`](CompiledKokoroError::VerificationFailed) - audio quality check error.
    /// - [`Tensor`](CompiledKokoroError::Tensor) - underlying tensor operation error.
    pub fn synthesize(
        &mut self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<(DynTensor, Certificate), CompiledKokoroError> {
        match self.pipeline_mode {
            PipelineMode::TwoPhase => {
                // Two-phase pipeline: explicit Phase 1 / Phase 2 split at
                // the regulate sync point, with GpuFence submissions for
                // maximum CPU-GPU overlap. Production step variants skip
                // blits. Part of #4264.
                self.synthesize_two_phase(input_ids, style, speed, cache)
            }
            PipelineMode::Sequential => {
                // Sequential pipeline: steps dispatched in order with per-step
                // fence submissions. Simpler execution flow for debugging.
                let output = self.synthesize_gpu_inner(input_ids, style, speed, cache)?;
                Ok((output.cpu_audio, output.certificate))
            }
        }
    }

    /// Synthesize PCM audio with per-synthesis arena utilization tracking.
    ///
    /// Identical to [`synthesize`] but additionally captures arena statistics
    /// (buffer hits/misses, peak usage, pool reuse) as a [`KokoroArenaReport`].
    /// Use this to measure the RTF impact of arena buffer reuse and diagnose
    /// whether the arena is appropriately sized.
    ///
    /// Arena stats are reset before synthesis and captured after, so the
    /// report reflects exactly one synthesis call's allocation behavior.
    ///
    /// # Arguments
    ///
    /// Same as [`synthesize`].
    ///
    /// # Returns
    ///
    /// `(audio, certificate, arena_report)` -- audio and certificate are
    /// identical to [`synthesize`]; `arena_report` contains per-synthesis
    /// allocation metrics.
    ///
    /// Part of #4264.
    pub fn synthesize_with_arena_report(
        &mut self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<(DynTensor, Certificate, KokoroArenaReport), CompiledKokoroError> {
        let pre = snapshot_arena_pre();
        let (audio, cert) = self.synthesize(input_ids, style, speed, cache)?;
        let report = build_arena_report(&pre);
        Ok((audio, cert, report))
    }

    /// Synthesize PCM audio and return intermediate pipeline tensors.
    ///
    /// Identical to [`synthesize`] but additionally returns durations, F0,
    /// and energy predictions via [`SynthesisIntermediates`]. Useful for
    /// visualization, prosody debugging, and custom post-processing.
    ///
    /// The intermediates are transferred to CPU before return so they remain
    /// valid after GPU arena reclamation on the next call.
    ///
    /// # Arguments
    ///
    /// Same as [`synthesize`].
    ///
    /// # Returns
    ///
    /// `(audio, certificate, intermediates)` — audio and certificate are
    /// identical to [`synthesize`]; intermediates contain the raw prosody
    /// predictions from steps 4-5.
    pub fn synthesize_with_intermediates(
        &mut self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<(DynTensor, Certificate, SynthesisIntermediates), CompiledKokoroError> {
        let style_split = prepare_synthesis_inputs(self, input_ids, style, speed)?;
        let decoder_style = style_split.decoder_style.to_device(&gpu())?;
        let prosody_style = style_split.prosody_style.to_device(&gpu())?;

        let result =
            (|| -> Result<(DynTensor, Certificate, SynthesisIntermediates), CompiledKokoroError> {
                let (audio, intermediates) =
                    with_nan_check_policy(NanCheckPolicy::Skip, || -> Result<(DynTensor, SynthesisIntermediates), CompiledKokoroError> {
                        let enc = self.step_encode(input_ids, cache)?;

                        let pros = self.step_predict_prosody(
                            &enc.bert_features,
                            &prosody_style,
                            enc.seq_len,
                            cache,
                        )?;

                        let reg = self.step_regulate(
                            &pros.dur_logits,
                            &pros.features,
                            &enc.text_features,
                            speed,
                            cache,
                        )?;

                        let f0e = self.step_predict_f0_energy(
                            &reg.aligned_dur,
                            &prosody_style,
                            reg.t_mel,
                            cache,
                        )?;

                        // Capture intermediates (CPU copy for survival across arena resets).
                        let intermediates = SynthesisIntermediates::new(
                            reg.durations.to_device(&cpu())?,
                            f0e.f0.to_device(&cpu())?,
                            f0e.energy.to_device(&cpu())?,
                            reg.t_mel,
                        );

                        let har_source =
                            self.step_harmonic_source(&f0e.f0, &f0e.energy, reg.t_mel, cache)?;

                        let generator = self.step_generate(
                            &reg.regulated,
                            &f0e.f0,
                            &f0e.energy,
                            &decoder_style,
                            &har_source,
                            reg.t_mel,
                            cache,
                        )?;

                        let audio =
                            self.step_istft(&generator.magnitude, &generator.phase, cache)?;
                        Ok::<_, CompiledKokoroError>((audio, intermediates))
                    })?;

                // Pipeline exit: single GPU→CPU transfer flushes all pending work.
                let audio = audio.to_device(&cpu())?;

                if audio.any_non_finite()? {
                    let count = audio
                        .as_cpu_f32()
                        .map(|v| v.iter().filter(|x| !x.is_finite()).count())
                        .unwrap_or(1);
                    return Err(nn_core::TensorError::NonFiniteData {
                        name: "pipeline_output_audio".into(),
                        count,
                    }
                    .into());
                }

                let certificate = self.step_verify(&audio)?;
                Ok((audio, certificate, intermediates))
            })();

        if result.is_err() {
            crate::gpu_scope::discard_pending_batch();
        }

        if result.is_ok() && self.auto_release && !self.weights_released() {
            let _ = self.release_model_weights();
        }

        result
    }

    /// Synthesize PCM audio with inline segment-level GPU pipelining via [`GpuFence`].
    ///
    /// **Note:** As of #4264, [`synthesize`] defaults to the two-phase pipeline
    /// (`PipelineMode::TwoPhase`) which provides the same CPU-GPU overlap with
    /// a cleaner Phase 1/Phase 2 structural split. This method remains available
    /// as a legacy entry point that uses inline fence submissions without the
    /// explicit two-phase architecture. Functionally identical output.
    ///
    /// Uses `_production` step variants (step 4, 6, 8) that skip
    /// `to_standalone()` blits, and `GpuFence::submit_current()` at step
    /// boundaries so the GPU executes while the CPU encodes the next segment.
    ///
    /// # Pipelining strategy
    ///
    /// 4 fence submit points: Phase 1 uses per-step submits (encode, prosody);
    /// Phase 2 batches f0+harmonic into one submit and generator+iSTFT into
    /// the pipeline-exit commit. This reduces total submits from 6 to 4,
    /// saving 2 command buffer creation overheads. Each fence is dropped
    /// without waiting — Metal queue ordering handles all data dependencies
    /// between command buffers. The pipeline-exit `to_device(&cpu())` serves
    /// as the global barrier.
    ///
    /// # Arena safety
    ///
    /// [`GpuFence`] submit does NOT reset the activation arena (unlike
    /// `gpu_scope::sync()`). This is safe because:
    /// - `step_encode`, `step_regulate`, `step_harmonic_source`, `step_istft`
    ///   produce standalone buffers (no arena generation).
    /// - Compiled segment outputs use `MetalTensorData::new()` with
    ///   `arena_generation: None` — immune to stale-read detection.
    ///
    /// The pipeline-exit `to_device(&cpu())` call triggers a final `flush()`
    /// which resets the arena after all GPU work completes.
    ///
    /// Part of #4251, #4264.
    pub fn synthesize_pipelined(
        &mut self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<(DynTensor, Certificate), CompiledKokoroError> {
        let style_split = prepare_synthesis_inputs(self, input_ids, style, speed)?;
        let decoder_style = style_split.decoder_style.to_device(&gpu())?;
        let prosody_style = style_split.prosody_style.to_device(&gpu())?;

        // Decode scope: suppress stale-arena-read checks for the entire
        // pipeline, same rationale as synthesize_gpu_inner. Part of #4264.
        let result = with_decode_scope(||
        (|| -> Result<(DynTensor, Certificate), CompiledKokoroError> {
            let audio = with_nan_check_policy(NanCheckPolicy::Skip, || {
                // Steps 1-2: PlBert + bert_encoder + TextEncoder.
                let enc = self.step_encode(input_ids, cache)?;

                // Step 3: ProsodyPredictor.
                // Encode and prosody dispatch commands are batched into a single
                // lazy batch (no intermediate GpuFence::submit_current between
                // them). Consolidation saves one Metal command buffer creation
                // (~10-50us on M4 Max). The regulate step below has its own
                // submit()+sync() which flushes all pending encode+prosody GPU
                // work before the 4-byte scalar readback. Metal command queue
                // ordering guarantees correct execution order within the single
                // command buffer. Part of #4264.
                let pros = self.step_predict_prosody(
                    &enc.bert_features,
                    &prosody_style,
                    enc.seq_len,
                    cache,
                )?;

                // Submit encode+prosody GPU work as a single batch. The GPU
                // starts executing both segments while the CPU prepares regulate
                // encoding. One submit instead of two saves one command buffer
                // creation overhead. Part of #4264 (encode+prosody batch
                // consolidation).
                let _fence_encode_prosody = GpuFence::submit_current()?;

                // Step 4: Duration + length_regulate.
                // step_regulate has its own inherent submit()+sync() for the
                // 4-byte prefix-sum readback. Metal queue ordering ensures the
                // encode+prosody batch completes before regulate's sync reads
                // the scalar. Production path: skip all blits (#4264).
                let reg = self.step_regulate_production(
                    &pros.dur_logits,
                    &pros.features,
                    &enc.text_features,
                    speed,
                    cache,
                )?;

                // ICB replay: notify pre-readback phase complete. (#4264)
                self.notify_pre_readback_complete(enc.seq_len);

                // Step 5: F0EnergyPredictor.
                let f0e = self.step_predict_f0_energy(
                    &reg.aligned_dur,
                    &prosody_style,
                    reg.t_mel,
                    cache,
                )?;

                // Step 6: Harmonic source — skip blit (#4264).
                // Batched with f0 in same lazy batch (no intermediate submit).
                let har_source = self.step_harmonic_source_production(
                    &f0e.f0, &f0e.energy, reg.t_mel, cache,
                )?;

                // Single post-readback submit: f0+harmonic together.
                // Generator is the heaviest segment (~46 dispatches), so
                // overlapping its CPU encoding with f0+harmonic GPU execution
                // is the highest-value overlap point. Reduces Phase 2 from
                // 3 intermediate submits to 1. Part of #4264.
                let _fence_f0_harmonic = GpuFence::submit_current()?;

                // Step 7: Generator / FullDecoder.
                let upsample_factor: usize =
                    self.shared.config.upsample_rates.iter().product();
                let total_samples =
                    generator_total_samples(reg.t_mel, upsample_factor)?;
                let generator = self.step_generate(
                    &reg.regulated,
                    &f0e.f0,
                    &f0e.energy,
                    &decoder_style,
                    &har_source,
                    reg.t_mel,
                    cache,
                )?;

                // No intermediate submit between generator and iSTFT.
                // Both stay in the same lazy batch, committed at pipeline exit.
                // Saves one command buffer creation. Part of #4264.

                // Step 8: GPU iSTFT -> PCM audio — skip blit (#4264).
                // The final to_device(&cpu()) below commits this batch and
                // waits. Metal queue ordering guarantees all prior batches
                // complete first.
                let audio = self.step_istft_production(
                    &generator.magnitude, &generator.phase, cache,
                )?;

                // ICB replay: notify post-readback phase complete. (#4264)
                self.notify_post_readback_complete(reg.t_mel, total_samples);

                Ok::<_, CompiledKokoroError>(audio)
            })?;

            // Pipeline exit: single GPU→CPU transfer flushes all pending work.
            let audio = audio.to_device(&cpu())?;

            // Pipeline exit NaN guard (OUTSIDE Skip scope — defense-in-depth).
            if audio.any_non_finite()? {
                let count = audio
                    .as_cpu_f32()
                    .map(|v| v.iter().filter(|x| !x.is_finite()).count())
                    .unwrap_or(1);
                return Err(nn_core::TensorError::NonFiniteData {
                    name: "pipeline_output_audio".into(),
                    count,
                }
                .into());
            }

            // Step 9: Verify audio quality (already CPU, no-op transfer inside)
            let certificate = self.step_verify(&audio)?;

            Ok((audio, certificate))
        })()
        );

        // Clean up stale GPU commands on error.
        if result.is_err() {
            crate::gpu_scope::discard_pending_batch();
        }

        // Auto-release CPU model weights after first successful synthesis (#3079).
        if result.is_ok() && self.auto_release && !self.weights_released() {
            let _ = self.release_model_weights();
        }

        result
    }
}
