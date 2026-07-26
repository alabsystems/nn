// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Step 4 (length_regulate) extracted from `compiled_kokoro_steps.rs`.
//!
//! Computes durations from ProsodyPredictor logits and runs
//! `length_regulate` (repeat_interleave) to expand features and text
//! encodings to mel-frame length. Uses GPU prefix-sum (#2911) for
//! buffer allocation with only a 4-byte scalar readback. On the hot path,
//! the readback is reached via `submit()+sync()` rather than `flush()`.
//!
//! ## Compiled segment (#1815 Tier 6 D2b)
//!
//! The elementwise chain (sigmoid → sum → mul_speed → clamp → squeeze →
//! add → floor → clamp_min) is compiled as segment 5. Speed is a tensor
//! input, so the segment caches by `seq_len` without recompiling per speed.
//! Saves ~4-5 GPU encodings vs the previous eager path.
//!
//! Extracted to keep `compiled_kokoro_steps.rs` under 450 lines.
//! Part of #2218, #1815.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::check_output_finite;
use nn_core::DType;

use nn_models::kokoro_error::validate_speed;

use crate::cache::PipelineCache;

use super::{
    check_multi_output, cpu, gpu, CompiledKokoro, CompiledKokoroError, StepRegulateResult,
};

impl CompiledKokoro {
    /// Step 4: Compute durations and run length_regulate.
    ///
    /// This is the **first prosody hook point**: modify `durations` in the
    /// returned result before passing to [`step_predict_f0_energy`] and
    /// [`step_generate`]. For example, apply per-phoneme duration multipliers.
    /// Arena: **standalone** (see [safety contract](super::steps)).
    ///
    /// # GPU prefix-sum (#2911, Phase 2)
    ///
    /// Counts stay on GPU. Shared prefix sum for both repeat_interleave
    /// calls. Only 4-byte scalar readback for output buffer allocation.
    /// Uses `submit()+sync()` on the hot path, so `step_regulate` no longer
    /// contributes a counted flush.
    ///
    /// # Compiled segment (#1815 Tier 6 D2b)
    ///
    /// The duration elementwise chain is compiled as segment 5. Speed is
    /// a tensor input (not baked constant), so a single compilation serves
    /// all speed values for a given `seq_len`.
    ///
    /// # Arguments
    ///
    /// * `dur_logits` — `[B, T, max_dur]` from [`StepProsodyResult`].
    /// * `features` — `[B, d_en+style_dim, T]` from [`StepProsodyResult`].
    /// * `text_features` — `[B, d_en, T]` from [`StepEncodeResult`].
    /// * `speed` — Speaking rate multiplier (1.0 = normal).
    /// * `cache` — Metal pipeline cache.
    pub fn step_regulate(
        &mut self,
        dur_logits: &DynTensor,
        features: &DynTensor,
        text_features: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<StepRegulateResult, CompiledKokoroError> {
        self.step_regulate_inner(dur_logits, features, text_features, speed, cache, true)
    }

    /// Production-path variant of [`step_regulate`] that skips ALL
    /// `to_standalone()` GPU blits (durations, aligned_dur, regulated).
    ///
    /// In the production [`synthesize`](Self::synthesize) /
    /// [`synthesize_gpu`](Self::synthesize_gpu) /
    /// [`synthesize_pipelined`](Self::synthesize_pipelined) paths:
    /// - `durations` is never read (only needed by `synthesize_with_intermediates`).
    /// - `aligned_dur` and `regulated` are consumed as GPU inputs by subsequent
    ///   compiled segments (steps 5, 7). No `flush()` or `sync()` occurs between
    ///   step 4's scatter outputs and their consumption, so the arena-resident
    ///   buffers remain valid. GPU dispatch reads raw buffer pointers without
    ///   stale-read detection (#2328 only fires on `to_device(&cpu())`).
    ///
    /// Saves 3 GPU blit dispatches per synthesis call (#4264).
    ///
    /// The returned tensors are arena-allocated — they may become stale after
    /// an arena reset. Callers MUST NOT call `to_device(&cpu())` on them.
    pub(crate) fn step_regulate_production(
        &mut self,
        dur_logits: &DynTensor,
        features: &DynTensor,
        text_features: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<StepRegulateResult, CompiledKokoroError> {
        self.step_regulate_inner(dur_logits, features, text_features, speed, cache, false)
    }

    /// Core implementation of step 4 (length_regulate).
    ///
    /// When `blit_outputs` is `true`, all output tensors (durations, aligned_dur,
    /// regulated) are promoted to standalone GPU buffers via `to_standalone()`.
    /// When `false` (production fast path), all three are returned as arena-
    /// allocated tensors — saving 3 GPU blit dispatches (#4264). This is safe
    /// because the production pipeline consumes them as GPU inputs only (no
    /// `flush()` or arena reset occurs before consumption).
    fn step_regulate_inner(
        &mut self,
        dur_logits: &DynTensor,
        features: &DynTensor,
        text_features: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
        blit_outputs: bool,
    ) -> Result<StepRegulateResult, CompiledKokoroError> {
        validate_speed(speed).map_err(|_| CompiledKokoroError::InvalidSpeed { value: speed })?;

        let seq_len = dur_logits.dims()[1]; // T phonemes

        // Speed inverse as a [1]-shaped tensor for compiled segment input.
        // broadcast_mul([B,T] * [1]) broadcasts correctly (right-aligned).
        // Created on GPU so trace_seg_regulate (tracing phase) doesn't hit
        // mixed-device errors when dur_logits is already on GPU (#3213).
        let speed_inv = DynTensor::full(&[1], 1.0 / f64::from(speed), DType::F32, &gpu())?;

        // Compile/cache segment 5 (elementwise chain, no model weights).
        // Cache key: seq_len. Speed varies via tensor input, no recompile.
        self.ensure_seg_regulate(seq_len, dur_logits, &speed_inv, cache)?;

        // Check if we have a cached total_repeats for this (seq_len, speed).
        // f32::to_bits() gives exact float key matching — safe because speed
        // is validated above (positive, finite). Part of #4264.
        let cache_key = (seq_len, speed.to_bits());
        let cached_total = self.regulate_total_cache.get(&cache_key).copied();

        // Output tensors must survive across later step executions and any
        // GPU flushes (arena generation resets). Intermediates use the arena;
        // only the 3 outputs are blit-copied to standalone. (#2574, #4279)
        let (durations, aligned_dur, regulated) = {
            let seg = self
                .seg_regulate
                .get(seq_len)
                .ok_or_else(|| super::seg_cache_miss("regulate"))?;

            let dur_logits_gpu = dur_logits.to_device(&gpu())?;
            let speed_inv_gpu = speed_inv.to_device(&gpu())?;

            let outputs = seg
                .execute_dyn_outputs_no_fence(cache, &[&dur_logits_gpu, &speed_inv_gpu])
                .map_err(|e| CompiledKokoroError::SegmentExecutionFailed {
                    segment: "regulate",
                    source: Box::new(e),
                })?;
            check_multi_output(&outputs, 2, "regulate")?;

            // Primary (0): counts_gpu [T], Secondary (1): durations [B, T].
            let counts_gpu = &outputs[0];
            let durations = outputs[1].clone();

            let dim_size = counts_gpu.dims()[0]; // T phonemes

            if dim_size > crate::dyn_tensor_metal::MAX_GPU_PREFIX_SUM {
                // Fallback: unusually long input — use CPU path (#2911).
                crate::gpu_scope::flush()?;
                let counts_cpu = counts_gpu.to_device(&cpu())?;
                let aligned_dur = features.repeat_interleave(2, &counts_cpu)?;
                let regulated = text_features.repeat_interleave(2, &counts_cpu)?;
                (durations, aligned_dur, regulated)
            } else {
                // GPU prefix-sum (#2911 Phase 2): dispatch the scan into the
                // current lazy batch.
                let offsets_buf =
                    crate::dyn_tensor_metal::dispatch_prefix_sum_only(counts_gpu, dim_size)?;

                // Hot-path optimization (#4264): if we have a cached total_repeats
                // for this (seq_len, speed) pair, skip the submit()+sync() GPU
                // pipeline stall. The compiled segment is deterministic for a given
                // seq_len, and speed is an input tensor — so total_repeats is
                // deterministic for the same (seq_len, speed) pair.
                //
                // Cold path: submit+sync to read the scalar total, then cache it.
                let total_repeats = if let Some(cached) = cached_total {
                    // No submit+sync needed — GPU work stays in the lazy batch
                    // and will be committed at the next sync point (pipeline exit
                    // `to_device(&cpu())` or `GpuFence::wait()`). The prefix-sum
                    // dispatch is already encoded in the batch; its result is
                    // consumed by the scatter dispatches below which are also
                    // encoded in the same batch. All GPU-to-GPU data dependencies
                    // are satisfied by Metal's implicit command buffer ordering.
                    cached
                } else {
                    crate::gpu_scope::submit()?;
                    crate::gpu_scope::sync()?;
                    
                    crate::dyn_tensor_metal::read_prefix_sum_total(&offsets_buf, dim_size)?
                };

                // Two scatter dispatches using shared GPU offsets — no additional flushes.
                let aligned_dur = crate::dyn_tensor_metal::gpu_scatter_with_offsets(
                    features,
                    2,
                    &offsets_buf,
                    dim_size,
                    total_repeats,
                )?;
                let regulated = crate::dyn_tensor_metal::gpu_scatter_with_offsets(
                    text_features,
                    2,
                    &offsets_buf,
                    dim_size,
                    total_repeats,
                )?;
                (durations, aligned_dur, regulated)
            }
        };

        // Cache total_repeats for future calls with the same (seq_len, speed).
        // This must be done after the scatter to ensure total_repeats is correct.
        // On the cold path, we read total_repeats from the GPU. On the hot path,
        // we used the cached value. Either way, cache the value for next time.
        // t_mel is derived from total_repeats via the aligned_dur shape.
        if cached_total.is_none() {
            if aligned_dur.dims().len() >= 3 {
                let t_mel_actual = aligned_dur.dims()[2];
                // total_repeats = t_mel for the scatter output dimension.
                // Cache it indexed by (seq_len, speed_bits).
                self.regulate_total_cache.insert(cache_key, t_mel_actual);
            }
        }
        // GPU blit to standalone buffers — no CPU roundtrip (#4279).
        // Production path (blit_outputs=false): skip ALL blits. aligned_dur
        // and regulated are consumed as GPU inputs by steps 5 and 7. No
        // flush()/sync() occurs between scatter outputs and consumption, so
        // arena-resident buffers remain valid. Stale-read detection (#2328)
        // only fires on to_device(&cpu()), which the production path never
        // calls on these intermediates. Saves 3 GPU blit dispatches (#4264).
        //
        // Step API / intermediates path (blit_outputs=true): blit all outputs
        // to standalone so callers can read them back or hold across arena resets.
        let (durations, aligned_dur, regulated) = if blit_outputs {
            let d = crate::to_standalone::to_standalone(&durations)?;
            check_output_finite(&d, "step:durations")?;
            let a = crate::to_standalone::to_standalone(&aligned_dur)?;
            let r = crate::to_standalone::to_standalone(&regulated)?;
            check_output_finite(&a, "step:length_regulate_dur")?;
            check_output_finite(&r, "step:length_regulate_asr")?;
            (d, a, r)
        } else {
            (durations, aligned_dur, regulated)
        };
        if aligned_dur.dims().len() < 3 {
            return Err(nn_core::TensorError::RankMismatch {
                expected: 3,
                actual: aligned_dur.dims().len(),
            }
            .into());
        }
        let t_mel = aligned_dur.dims()[2];
        if t_mel == 0 {
            return Err(nn_core::TensorError::Unsupported(
                "t_mel is 0 after length_regulate — no mel frames produced".into(),
            )
            .into());
        }

        Ok(StepRegulateResult::new(
            durations,
            aligned_dur,
            regulated,
            t_mel,
        ))
    }
}
