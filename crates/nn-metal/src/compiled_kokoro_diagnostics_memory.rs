// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! RSS memory profiling for the Kokoro compiled pipeline (#3079).
//!
//! Extracted from `compiled_kokoro_diagnostics.rs` to keep the parent
//! under the 450-line limit. Contains [`CompiledKokoro::synthesize_with_memory`].

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{with_nan_check_policy, NanCheckPolicy};
use nn_tts_verify::Certificate;

use crate::cache::PipelineCache;

use super::{
    prepare_synthesis_inputs, CompiledKokoro, CompiledKokoroError, DiagnosticOutput,
    MemoryBreakdown, TimingReport,
};

impl CompiledKokoro {
    /// Synthesize with full diagnostics including RSS memory profiling (#3079).
    ///
    /// Like [`synthesize_with_diagnostics`](Self::synthesize_with_diagnostics)
    /// but additionally records process RSS at key pipeline stages using
    /// macOS `mach_task_info`. The RSS data shows where memory is consumed:
    /// before synthesis, after each pipeline step, and after verification.
    ///
    /// Returns `(audio, certificate, diagnostics)` with `diagnostics.rss`
    /// populated with checkpoint data.
    pub fn synthesize_with_memory(
        &mut self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<(DynTensor, Certificate, DiagnosticOutput), CompiledKokoroError> {
        crate::dispatch_stats::reset_counters();
        crate::arena::reset_arena_stats();

        let mut rss = crate::rss::RssTracker::new();
        rss.checkpoint("pre_synthesize");

        let style_split = prepare_synthesis_inputs(self, input_ids, style, speed)?;
        let decoder_style = style_split.decoder_style;
        let prosody_style = style_split.prosody_style;

        let result =
            (|| -> Result<(DynTensor, Certificate, DiagnosticOutput), CompiledKokoroError> {
                let audio = with_nan_check_policy(NanCheckPolicy::Skip, || {
                    let enc = self.step_encode(input_ids, cache)?;
                    rss.checkpoint("after_encode");

                    let pros = self.step_predict_prosody(
                        &enc.bert_features,
                        &prosody_style,
                        enc.seq_len,
                        cache,
                    )?;
                    rss.checkpoint("after_prosody");

                    let reg = self.step_regulate(
                        &pros.dur_logits,
                        &pros.features,
                        &enc.text_features,
                        speed,
                        cache,
                    )?;
                    rss.checkpoint("after_regulate");

                    let f0e = self.step_predict_f0_energy(
                        &reg.aligned_dur,
                        &prosody_style,
                        reg.t_mel,
                        cache,
                    )?;
                    rss.checkpoint("after_f0_energy");

                    let har_source =
                        self.step_harmonic_source(&f0e.f0, &f0e.energy, reg.t_mel, cache)?;
                    rss.checkpoint("after_harmonic");

                    let generator = self.step_generate(
                        &reg.regulated,
                        &f0e.f0,
                        &f0e.energy,
                        &decoder_style,
                        &har_source,
                        reg.t_mel,
                        cache,
                    )?;
                    rss.checkpoint("after_generate");

                    self.step_istft(&generator.magnitude, &generator.phase, cache)
                })?;
                rss.checkpoint("after_istft");

                // Pipeline exit: single GPU→CPU transfer flushes all pending work.
                let audio = audio.to_device(&super::cpu())?;

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
                rss.checkpoint("after_verify");

                // Auto-release CPU model weights after first successful synthesis (#3079).
                // RSS checkpoint before/after shows ~320 MB drop for Kokoro-82M.
                if self.auto_release && !self.weights_released() {
                    rss.checkpoint("before_weight_release");
                    let _ = self.release_model_weights();
                    rss.checkpoint("after_weight_release");
                }

                let stats = crate::dispatch_stats::dispatch_stats();
                let arena_peak_bytes = crate::arena::default_arena_peak_bytes();
                let arena_stats = crate::arena::arena_stats();

                // Build a minimal TimingReport (zero-valued since this path
                // focuses on memory, not timing — use synthesize_with_timing for that).
                let zero = std::time::Duration::ZERO;
                let timing = TimingReport {
                    encode: zero,
                    prosody: zero,
                    regulate: zero,
                    f0_energy: zero,
                    harmonic: zero,
                    generate: zero,
                    istft: zero,
                    verify: zero,
                    total: zero,
                    cache_misses: 0,
                };

                let memory = self.memory_breakdown();

                Ok((
                    audio,
                    certificate,
                    DiagnosticOutput {
                        timing,
                        stats,
                        arena_peak_bytes,
                        arena_stats,
                        rss: Some(rss),
                        memory: Some(memory),
                    },
                ))
            })();

        if result.is_err() {
            crate::gpu_scope::discard_pending_batch();
        }

        result
    }

    /// Estimate peak arena bytes for a single forward pass.
    ///
    /// Sums the maximum `buffer_plan.total_bytes` across all 8 compiled segment
    /// caches. Each segment's buffer plan represents the intermediate GPU memory
    /// needed during that segment's execution. Although the arena resets between
    /// segments (via checkpoint/restore), eager ops between segments also consume
    /// arena space. The estimate uses a 2x headroom multiplier to account for:
    /// - Eager GPU ops between compiled segments (style split, FFT, cumsum)
    /// - 256-byte Metal alignment padding between allocations
    /// - Output tensors that survive into the next segment's input
    ///
    /// Returns 0 if no segments have been compiled yet. Call after at least one
    /// `synthesize()` to populate segment caches, or after `warmup()`.
    ///
    /// Part of #4289.
    #[must_use]
    pub fn estimate_arena_bytes(&self) -> usize {
        let segments = self.all_segments();
        // Segments execute sequentially — peak = max, not sum.
        let max_bytes: usize = segments
            .iter()
            .map(|seg| seg.max_entry_buffer_plan_bytes())
            .max()
            .unwrap_or(0);
        // 2x headroom for eager inter-segment ops + alignment padding.
        max_bytes.saturating_mul(2)
    }

        /// Build a per-domain memory attribution snapshot (#3079 D7).
    ///
    /// Collects GPU weight bytes, arena capacity, pool retained bytes,
    /// planned buffer bytes (per-model intermediate sub-allocation cache),
    /// CPU weight release status, and current process RSS into a single
    /// [`MemoryBreakdown`] for data-driven memory tuning.
    #[must_use]
    pub fn memory_breakdown(&self) -> MemoryBreakdown {
        let arena_stats = crate::arena::arena_stats();
        let planned_buf_bytes = self
            .all_segments()
            .iter()
            .map(|c| c.total_planned_buf_bytes())
            .sum();
        let cached_model_count = self.all_segments().iter().map(|c| c.len()).sum();
        MemoryBreakdown {
            gpu_weight_bytes: self.gpu_weight_bytes(),
            arena_capacity_bytes: crate::arena::arena_capacity(),
            arena_peak_bytes: crate::arena::default_arena_peak_bytes().unwrap_or(0),
            pool_retained_bytes: arena_stats.pool.pooled_bytes,
            planned_buf_bytes,
            cpu_weights_released: self.weights_released(),
            process_rss_bytes: crate::rss::rss_bytes(),
            metal_allocated_bytes: crate::rss::metal_allocated_bytes(),
            cached_model_count,
        }
    }
}
