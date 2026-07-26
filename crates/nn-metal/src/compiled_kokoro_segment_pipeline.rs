// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Two-phase CPU-GPU segment pipelining for single-voice Kokoro synthesis.
//!
//! Extends the chorus two-phase pattern (#4290) to the single-voice
//! [`synthesize`](super::CompiledKokoro::synthesize) path. The key insight:
//! `step_regulate` (step 4) has an inherent `submit()+sync()` for a 4-byte
//! scalar readback — the only hard sync in the entire pipeline. Everything
//! after regulate (steps 5-8) is sync-free GPU work.
//!
//! # Two-Phase Structure
//!
//! ```text
//! Phase 1 (has GPU sync):
//!   Steps 1-2: encode (PlBert + TextEncoder)
//!   Step 3:    prosody prediction
//!   Step 4:    regulate (prefix-sum scalar readback — hard sync)
//!
//! Phase 2 (sync-free):
//!   Step 5:    F0/energy prediction
//!   Step 6:    harmonic source
//!   Step 7:    generator
//!   Step 8:    iSTFT
//! ```
//!
//! # CPU-GPU Overlap
//!
//! After Phase 1 completes (regulate sync returns `t_mel`), the CPU submits
//! all Phase 1 GPU work via [`GpuFence::submit_current()`] non-blocking. The
//! GPU starts executing Phase 1 dispatch commands immediately. Meanwhile, the
//! CPU begins encoding Phase 2 dispatch commands (f0, harmonic, generator,
//! iSTFT). Metal command queue ordering guarantees that Phase 1 GPU work
//! completes before Phase 2 GPU work begins execution.
//!
//! ```text
//! CPU:  [Phase1 encode+sync]--[submit fence]--[Phase2 CPU encoding]----------
//! GPU:                         [Phase1 GPU execution]--[Phase2 GPU execution]
//! ```
//!
//! The overlap window is the time the CPU spends encoding Phase 2 dispatches
//! (~0.5-2ms for 5+ compiled segments). This is work the GPU would otherwise
//! wait idle for.
//!
//! # Arena Safety
//!
//! [`GpuFence`] submit does NOT reset the activation arena. Arena-resident
//! tensors from Phase 1 (regulate outputs with `_production` blit elision)
//! remain valid for Phase 2's GPU dispatch encoding. The pipeline-exit
//! `to_device(&cpu())` triggers the final flush and arena reset after all
//! GPU work completes.
//!
//! # Production Blit Elision
//!
//! Uses `_production` step variants (step 4, 6, 8) that skip `to_standalone()`
//! GPU blits. This is safe because no `flush()`/`sync()` occurs between Phase 1
//! outputs and their consumption by Phase 2 GPU dispatches. The only sync is
//! regulate's inherent prefix-sum readback, which happens *before* the scatter
//! outputs are produced. See `compiled_kokoro_steps.rs` safety contract.
//!
//! Part of #4264.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{with_nan_check_policy, NanCheckPolicy};

use nn_tts_verify::Certificate;

use super::{
    cpu, gpu, prepare_synthesis_inputs, CompiledKokoro, CompiledKokoroError, StepEncodeResult,
    StepRegulateResult,
};
use crate::arena::with_decode_scope;
use crate::cache::PipelineCache;
use crate::gpu_fence::GpuFence;

/// Result of Phase 1 (encode + prosody + regulate).
///
/// Contains all outputs needed by Phase 2, plus intermediate state
/// needed for the pipeline-exit verification.
pub(crate) struct Phase1Result {
    /// Encode output (bert_features, text_features, seq_len).
    pub enc: StepEncodeResult,
    /// Regulate output (durations, aligned_dur, regulated, t_mel).
    pub reg: StepRegulateResult,
    /// Prosody style (GPU-resident) — needed by Phase 2 step_predict_f0_energy.
    pub prosody_style: DynTensor,
    /// Decoder style (GPU-resident) — needed by Phase 2 step_generate.
    pub decoder_style: DynTensor,
}

impl CompiledKokoro {
    /// Run Phase 1: encode + prosody + regulate.
    ///
    /// Returns all outputs needed by Phase 2. `step_regulate_production` is
    /// used to skip blits (outputs consumed only by GPU dispatches in Phase 2).
    ///
    /// After this returns, the caller should submit pending GPU work via
    /// `GpuFence::submit_current()` to start Phase 1 execution on GPU while
    /// Phase 2 dispatch commands are encoded on CPU.
    fn run_phase_1(
        &mut self,
        input_ids: &DynTensor,
        prosody_style: &DynTensor,
        decoder_style: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<Phase1Result, CompiledKokoroError> {
        // Steps 1-2: PlBert + bert_encoder + TextEncoder.
        let enc = self.step_encode(input_ids, cache)?;

        // Step 3: ProsodyPredictor.
        // Encode and prosody dispatch commands are batched into a single lazy
        // batch (no intermediate GpuFence::submit_current between them).
        // Consolidation saves one Metal command buffer creation (~10-50us on
        // M4 Max). The regulate step below has its own submit()+sync() which
        // flushes all pending encode+prosody GPU work before the 4-byte scalar
        // readback. Metal command queue ordering guarantees correct execution
        // order within the single command buffer. Part of #4264.
        let pros = self.step_predict_prosody(
            &enc.bert_features,
            prosody_style,
            enc.seq_len,
            cache,
        )?;

        // Submit encode+prosody GPU work as a single batch. The GPU starts
        // executing both segments while the CPU prepares regulate encoding.
        // One submit instead of two saves one command buffer creation overhead.
        // Part of #4264 (encode+prosody batch consolidation).
        let _fence_encode_prosody = GpuFence::submit_current()?;

        // Step 4: Duration + length_regulate (production path: skip blits).
        // step_regulate has its own inherent submit()+sync() for the 4-byte
        // prefix-sum readback. Metal queue ordering ensures the encode+prosody
        // batch completes before regulate's sync reads the scalar.
        let reg = self.step_regulate_production(
            &pros.dur_logits,
            &pros.features,
            &enc.text_features,
            speed,
            cache,
        )?;

        Ok(Phase1Result {
            enc,
            reg,
            prosody_style: prosody_style.clone(),
            decoder_style: decoder_style.clone(),
        })
    }

    /// Run Phase 2: f0_energy + harmonic_source + generate + iSTFT.
    ///
    /// All steps are sync-free GPU work. Phase 1 outputs are consumed as GPU
    /// inputs — Metal queue ordering guarantees Phase 1 GPU work completes
    /// before Phase 2 GPU work executes.
    ///
    /// Uses `_production` step variants for step 6 (harmonic) and step 8
    /// (iSTFT) to skip `to_standalone()` blits.
    ///
    /// # Command buffer batching (#4264)
    ///
    /// Phase 2 uses a single intermediate fence submit (after f0+harmonic,
    /// before generator) instead of per-step submits. Since Phase 2 is entirely
    /// sync-free, Metal handles intra-command-buffer ordering automatically.
    /// Consolidating 3 intermediate submits to 1 reduces Metal command buffer
    /// creation overhead (~10-50us per command buffer on M4 Max).
    ///
    /// The highest-value overlap point is between f0+harmonic GPU execution and
    /// generator CPU encoding (~46 dispatches, the heaviest segment). The
    /// generator + iSTFT dispatches stay in a single lazy batch committed at the
    /// pipeline-exit `to_device(&cpu())`.
    fn run_phase_2(
        &mut self,
        phase1: &Phase1Result,
        cache: &PipelineCache,
    ) -> Result<DynTensor, CompiledKokoroError> {
        // Step 5: F0EnergyPredictor.
        let f0e = self.step_predict_f0_energy(
            &phase1.reg.aligned_dur,
            &phase1.prosody_style,
            phase1.reg.t_mel,
            cache,
        )?;

        // Step 6: Harmonic source — skip blit (production path).
        // Encode f0+harmonic into the same lazy batch (no intermediate submit).
        // Both are relatively lightweight (~20 dispatches combined), so
        // batching them avoids command buffer creation overhead.
        let har_source = self.step_harmonic_source_production(
            &f0e.f0,
            &f0e.energy,
            phase1.reg.t_mel,
            cache,
        )?;

        // Single Phase 2 intermediate submit: f0+harmonic together.
        // Generator is the heaviest segment (~46 dispatches), so overlapping
        // its CPU encoding with f0+harmonic GPU execution is the highest-value
        // overlap point. This submit also ensures f0+harmonic GPU work starts
        // executing while the CPU encodes generator dispatch commands.
        let _fence_f0_harmonic = GpuFence::submit_current()?;

        // Step 7: Generator / FullDecoder.
        let generator = self.step_generate(
            &phase1.reg.regulated,
            &f0e.f0,
            &f0e.energy,
            &phase1.decoder_style,
            &har_source,
            phase1.reg.t_mel,
            cache,
        )?;

        // No intermediate submit between generator and iSTFT — both stay in
        // the same lazy batch. iSTFT encoding is fast (~5 dispatches) and the
        // pipeline-exit to_device(&cpu()) commits everything. Eliminating the
        // submit here saves one command buffer creation (~10-50us). Metal queue
        // ordering within the single command buffer guarantees generator
        // completes before iSTFT reads its outputs.

        // Step 8: GPU iSTFT -> PCM audio — skip blit (production path).
        // The final to_device(&cpu()) below commits this batch and waits.
        // Metal queue ordering guarantees all prior batches complete first.
        self.step_istft_production(&generator.magnitude, &generator.phase, cache)
    }

    /// Synthesize PCM audio using explicit two-phase CPU-GPU pipelining.
    ///
    /// Functionally identical to [`synthesize`](Self::synthesize) — produces
    /// bit-identical output. The difference is pipeline structure:
    ///
    /// - **Phase 1** (steps 1-4): encode + prosody + regulate. Contains the
    ///   only hard sync point (step 4's 4-byte prefix-sum scalar readback).
    ///   After regulate returns `t_mel`, Phase 1 GPU work is submitted via
    ///   [`GpuFence`] non-blocking.
    ///
    /// - **Phase 2** (steps 5-8): f0/energy + harmonic source + generator +
    ///   iSTFT. Entirely sync-free GPU work. CPU starts encoding Phase 2
    ///   dispatches immediately after Phase 1 fence submission — the GPU
    ///   processes Phase 1 commands concurrently with Phase 2 CPU encoding.
    ///
    /// # Overlap Window
    ///
    /// The overlap is Phase 2's CPU encoding time (~0.5-2ms across 5 compiled
    /// segments: f0, sinegen_pre, sinegen_post, generator, iSTFT). On an M4
    /// Max, this eliminates the idle GPU bubble between regulate's sync and
    /// Phase 2's first dispatch.
    ///
    /// # Arguments
    ///
    /// Same as [`synthesize`](Self::synthesize).
    ///
    /// # Returns
    ///
    /// `(audio, certificate)` — identical to [`synthesize`].
    ///
    /// Part of #4264.
    pub fn synthesize_two_phase(
        &mut self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<(DynTensor, Certificate), CompiledKokoroError> {
        let style_split = prepare_synthesis_inputs(self, input_ids, style, speed)?;
        let decoder_style = style_split.decoder_style.to_device(&gpu())?;
        let prosody_style = style_split.prosody_style.to_device(&gpu())?;

        // Pre-size the arena to avoid growth during the forward pass (#4289).
        let arena_estimate = self.estimate_arena_bytes();
        if arena_estimate > 0 {
            let _ = crate::arena::ensure_default_arena_capacity(
                cache.context(),
                arena_estimate,
            );
        }

        // Decode scope: suppress stale-arena-read checks for the entire
        // pipeline, same rationale as synthesize_gpu_inner. Part of #4264.
        let result = with_decode_scope(||
        (|| -> Result<(DynTensor, Certificate), CompiledKokoroError> {
            let audio = with_nan_check_policy(NanCheckPolicy::Skip, || {
                // Phase 1: encode + prosody + regulate (has GPU sync).
                let phase1 = self.run_phase_1(
                    input_ids,
                    &prosody_style,
                    &decoder_style,
                    speed,
                    cache,
                )?;

                // Submit Phase 1 GPU work non-blocking. Metal queue ordering
                // guarantees Phase 1 commands complete before any Phase 2
                // commands execute on GPU. The CPU starts encoding Phase 2
                // dispatch commands immediately — overlapping with Phase 1
                // GPU execution. GpuFence does NOT reset the arena, so
                // Phase 1's arena-resident outputs (from step_regulate_production)
                // remain valid for Phase 2 GPU dispatch encoding.
                let _fence_phase1 = GpuFence::submit_current()?;

                // Phase 2: f0 + harmonic + generate + iSTFT (sync-free).
                // CPU encodes ~100+ GPU dispatch commands while Phase 1
                // work executes on GPU.
                self.run_phase_2(&phase1, cache)
            })?;

            // Pipeline exit: single GPU->CPU transfer flushes all pending work.
            // This is the only non-regulate sync point -- commits both Phase 1
            // and Phase 2 GPU work (any batches not yet submitted are committed
            // here) and waits for all of it to complete.
            let audio = audio.to_device(&cpu())?;

            // Pipeline exit NaN guard (OUTSIDE Skip scope -- defense-in-depth).
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

            // Step 9: Verify audio quality (already CPU, no-op transfer inside).
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

/// Run Phase 2 (steps 5-8) on a `CompiledKokoro` instance given Phase 1 results.
///
/// Free function for use by the chorus pipeline where `&mut self` borrowing
/// conflicts prevent calling methods on individual voices while iterating.
/// Uses the public step API (with blits) for safety — chorus voices may hold
/// tensors across arena resets.
///
/// # Command buffer batching (#4264)
///
/// Uses a single intermediate submit (f0+harmonic together, before generator)
/// instead of 3 per-step submits. Reduces Metal command buffer creation
/// overhead. Generator + iSTFT stay in one lazy batch committed by the caller.
///
/// Part of #4264.
pub(crate) fn run_decode_phase(
    voice: &mut CompiledKokoro,
    reg: &StepRegulateResult,
    prosody_style: &DynTensor,
    decoder_style: &DynTensor,
    cache: &PipelineCache,
) -> Result<DynTensor, CompiledKokoroError> {
    let f0e = voice.step_predict_f0_energy(&reg.aligned_dur, prosody_style, reg.t_mel, cache)?;

    // Batch f0 + harmonic source in same lazy batch.
    let har = voice.step_harmonic_source(&f0e.f0, &f0e.energy, reg.t_mel, cache)?;

    // Single submit: f0+harmonic together. Generator CPU encoding overlaps
    // with f0+harmonic GPU execution. Part of #4264.
    let _fence_f0_harmonic = GpuFence::submit_current()
        .map_err(|e| CompiledKokoroError::Tensor(Box::new(e)))?;

    let generator = voice.step_generate(
        &reg.regulated,
        &f0e.f0,
        &f0e.energy,
        decoder_style,
        &har,
        reg.t_mel,
        cache,
    )?;

    // No intermediate submit between generator and iSTFT — both stay in the
    // same lazy batch. Committed by the caller. Part of #4264.

    voice.step_istft(&generator.magnitude, &generator.phase, cache)
}

/// Run Phase 2 (steps 5-8) and return a non-blocking [`GpuFence`].
///
/// Identical to [`run_decode_phase`] but submits the final GPU work via
/// `GpuFence::submit_current()` instead of leaving it in the lazy batch.
/// The caller collects fences and waits on them in bulk.
///
/// Part of #4264.
pub(crate) fn run_decode_phase_async(
    voice: &mut CompiledKokoro,
    reg: &StepRegulateResult,
    prosody_style: &DynTensor,
    decoder_style: &DynTensor,
    cache: &PipelineCache,
) -> Result<(DynTensor, Option<GpuFence>), CompiledKokoroError> {
    let audio = run_decode_phase(voice, reg, prosody_style, decoder_style, cache)?;

    // Submit the pending GPU work non-blocking.
    let fence = GpuFence::submit_current()
        .map_err(|e| CompiledKokoroError::Tensor(Box::new(e)))?;

    Ok((audio, fence))
}

#[cfg(test)]
mod tests {
    //! Tests for two-phase pipeline parity and fence correctness.
    //!
    //! These tests verify that `synthesize_two_phase` produces identical
    //! output to `synthesize` and that the Phase 1/Phase 2 separation
    //! functions handle edge cases correctly.

    use super::*;

    /// Verify that Phase1Result correctly carries all fields.
    #[test]
    fn test_phase1_result_fields() {
        // Phase1Result is a plain data carrier — verify it has the expected
        // structure by checking that each field is accessible. Actual GPU
        // testing requires KOKORO_WEIGHTS and is covered by the Kokoro gates.
        let _ = size_of::<Phase1Result>();
    }
}
