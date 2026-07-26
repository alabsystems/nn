// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic accessors for [`CompiledKokoro`].
//!
//! Extracted from `compiled_kokoro.rs` for code structure compliance.
//! These methods inspect cached segment state for reporting and testing.
//! Includes [`TimingReport`] for per-stage latency breakdown (#2781).
//!
//! Steps 1-8 run inside `NanCheckPolicy::Skip`, matching the main pipeline
//! in `compiled_kokoro_pipeline.rs` (#2981). Per-step timings measure encoding
//! time, not GPU execution time (GPU work batches in the lazy buffer).
//!
//! Part of #2498, #2781, #2218.

use std::sync::Arc;
use std::time::Instant;

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{with_nan_check_policy, NanCheckPolicy};
use nn_tts_verify::Certificate;

use crate::cache::PipelineCache;
use crate::dispatch_stats::DispatchStats;

use super::{
    cpu, generator_total_samples, prepare_synthesis_inputs, CompiledKokoro, CompiledKokoroError,
};

#[path = "compiled_kokoro_diagnostic_types.rs"]
mod diagnostic_types;
pub use diagnostic_types::{
    DiagnosticOutput, DispatchCensus, DispatchSummary, GpuTimingReport, MemoryBreakdown,
    SegmentCensus, TimingReport,
};

#[path = "compiled_kokoro_diagnostics_memory.rs"]
mod memory;

#[path = "compiled_kokoro_diagnostics_reports.rs"]
mod reports;

impl CompiledKokoro {
    /// Synthesize with GPU dispatch statistics (#2739).
    ///
    /// Returns `(audio, certificate, stats)` where stats has flush/submit/encoding counts.
    pub fn synthesize_with_stats(
        &mut self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<(DynTensor, Certificate, DispatchStats), CompiledKokoroError> {
        crate::dispatch_stats::reset_counters();
        let (audio, cert) = self.synthesize(input_ids, style, speed, cache)?;
        let stats = crate::dispatch_stats::dispatch_stats();
        Ok((audio, cert, stats))
    }

    /// Synthesize with per-stage wall-clock timing (#2781).
    ///
    /// Returns `(audio, certificate, timing)` with per-step latency and cache miss count.
    pub fn synthesize_with_timing(
        &mut self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<(DynTensor, Certificate, TimingReport), CompiledKokoroError> {
        let style_split = prepare_synthesis_inputs(self, input_ids, style, speed)?;
        let decoder_style = style_split.decoder_style;
        let prosody_style = style_split.prosody_style;
        let seq_len = input_ids.dims()[1];
        let upsample_factor: usize = self.config().upsample_rates.iter().product();

        // Pre-pipeline cache miss check for seq_len-keyed segments.
        let mut cache_misses = 0usize;
        if !self.seg_plbert.contains_key(seq_len) {
            cache_misses += 1;
        }
        if !self.seg_text.contains_key(seq_len) {
            cache_misses += 1;
        }
        if !self.seg_prosody.contains_key(seq_len) {
            cache_misses += 1;
        }
        if !self.seg_regulate.contains_key(seq_len) {
            cache_misses += 1;
        }

        // NanCheckPolicy scope narrowing (#2981): Steps 1-8 run inside
        // NanCheckPolicy::Skip. Matches compiled_kokoro_pipeline.rs so timing
        // measurements reflect production behavior. Per-step timings measure
        // encoding time (GPU work batches in lazy buffer until sync points).

        let wall = Instant::now();

        let result = (|| -> Result<(DynTensor, Certificate, TimingReport), CompiledKokoroError> {
            // Steps 1-8: NaN-skip scope eliminates per-step GPU readbacks.
            let (audio, encode, prosody, regulate, f0_energy, harmonic, generate, istft) =
                with_nan_check_policy(
                    NanCheckPolicy::Skip,
                    || -> Result<_, CompiledKokoroError> {
                        let t0 = Instant::now();
                        let enc = self.step_encode(input_ids, cache)?;
                        let encode = t0.elapsed();

                        let t0 = Instant::now();
                        let pros = self.step_predict_prosody(
                            &enc.bert_features,
                            &prosody_style,
                            enc.seq_len,
                            cache,
                        )?;
                        let prosody = t0.elapsed();

                        let t0 = Instant::now();
                        let reg = self.step_regulate(
                            &pros.dur_logits,
                            &pros.features,
                            &enc.text_features,
                            speed,
                            cache,
                        )?;
                        let regulate = t0.elapsed();

                        // Cache keys for f0/generator depend on t_mel (known after regulate).
                        let t_mel = reg.t_mel;
                        let total_samples = generator_total_samples(t_mel, upsample_factor)?;
                        if !self.seg_f0.contains_key(t_mel) {
                            cache_misses += 1;
                        }
                        if !self.seg_generator.contains_key(total_samples) {
                            cache_misses += 1;
                        }

                        let t0 = Instant::now();
                        let f0e = self.step_predict_f0_energy(
                            &reg.aligned_dur,
                            &prosody_style,
                            t_mel,
                            cache,
                        )?;
                        let f0_energy = t0.elapsed();

                        let t0 = Instant::now();
                        let har_source =
                            self.step_harmonic_source(&f0e.f0, &f0e.energy, t_mel, cache)?;
                        let harmonic = t0.elapsed();

                        let t0 = Instant::now();
                        let generator = self.step_generate(
                            &reg.regulated,
                            &f0e.f0,
                            &f0e.energy,
                            &decoder_style,
                            &har_source,
                            t_mel,
                            cache,
                        )?;
                        let generate = t0.elapsed();

                        let t0 = Instant::now();
                        let audio =
                            self.step_istft(&generator.magnitude, &generator.phase, cache)?;
                        let istft = t0.elapsed();

                        Ok((
                            audio, encode, prosody, regulate, f0_energy, harmonic, generate, istft,
                        ))
                    },
                )?;

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

            // Step 9: Verify audio quality (CPU-only, model boundary)
            let t0 = Instant::now();
            let certificate = self.step_verify(&audio)?;
            let verify = t0.elapsed();

            let timing = TimingReport {
                encode,
                prosody,
                regulate,
                f0_energy,
                harmonic,
                generate,
                istft,
                verify,
                total: wall.elapsed(),
                cache_misses,
            };
            Ok((audio, certificate, timing))
        })();

        if result.is_err() {
            crate::gpu_scope::discard_pending_batch();
        }

        result
    }

    /// Synthesize with full diagnostics: timing + dispatch stats + arena peak (#2781, #2914).
    ///
    /// Returns `(audio, certificate, diagnostics)`.
    pub fn synthesize_with_diagnostics(
        &mut self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<(DynTensor, Certificate, DiagnosticOutput), CompiledKokoroError> {
        crate::dispatch_stats::reset_counters();
        crate::arena::reset_arena_stats();
        let (audio, cert, timing) = self.synthesize_with_timing(input_ids, style, speed, cache)?;

        let stats = crate::dispatch_stats::dispatch_stats();
        let arena_peak_bytes = crate::arena::default_arena_peak_bytes();
        let arena_stats = crate::arena::arena_stats();
        Ok((
            audio,
            cert,
            DiagnosticOutput {
                timing,
                stats,
                arena_peak_bytes,
                arena_stats,
                rss: None,
                memory: Some(self.memory_breakdown()),
            },
        ))
    }

    /// Synthesize with per-stage GPU execution timing (#4264).
    ///
    /// Unlike [`synthesize_with_timing`] which measures CPU encoding time,
    /// this method flushes the GPU command buffer after each pipeline step,
    /// waiting for actual GPU completion before starting the next step. The
    /// reported durations include real GPU execution time per segment.
    ///
    /// **This is a profiling tool, not a production path.** The per-step
    /// flushes add significant overhead (1 flush per step instead of 1 total),
    /// defeating lazy batching. Use for:
    /// - Identifying which GPU segments are bottlenecks
    /// - Measuring impact of fusion/dispatch reduction on actual GPU time
    /// - Validating that encoding-time hotspots match GPU-time hotspots
    ///
    /// Returns `(audio, certificate, gpu_timing)`.
    ///
    /// Part of #4264.
    pub fn synthesize_with_gpu_timing(
        &mut self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<(DynTensor, Certificate, GpuTimingReport), CompiledKokoroError> {
        let style_split = prepare_synthesis_inputs(self, input_ids, style, speed)?;
        let decoder_style = style_split.decoder_style;
        let prosody_style = style_split.prosody_style;
        let seq_len = input_ids.dims()[1];
        let upsample_factor: usize = self.config().upsample_rates.iter().product();

        // Pre-pipeline cache miss check.
        let mut cache_misses = 0usize;
        if !self.seg_plbert.contains_key(seq_len) {
            cache_misses += 1;
        }
        if !self.seg_text.contains_key(seq_len) {
            cache_misses += 1;
        }
        if !self.seg_prosody.contains_key(seq_len) {
            cache_misses += 1;
        }
        if !self.seg_regulate.contains_key(seq_len) {
            cache_misses += 1;
        }

        let wall = Instant::now();

        let result =
            (|| -> Result<(DynTensor, Certificate, GpuTimingReport), CompiledKokoroError> {
                // Steps 1-8: NaN-skip scope as in production pipeline.
                // After each step, flush GPU work so timing reflects actual
                // GPU execution, not just CPU encoding.
                let (audio, encode, prosody, regulate, f0_energy, harmonic, generate, istft) =
                    with_nan_check_policy(
                        NanCheckPolicy::Skip,
                        || -> Result<_, CompiledKokoroError> {
                            let t0 = Instant::now();
                            let enc = self.step_encode(input_ids, cache)?;
                            crate::gpu_scope::flush()?;
                            let encode = t0.elapsed();

                            let t0 = Instant::now();
                            let pros = self.step_predict_prosody(
                                &enc.bert_features,
                                &prosody_style,
                                enc.seq_len,
                                cache,
                            )?;
                            crate::gpu_scope::flush()?;
                            let prosody = t0.elapsed();

                            let t0 = Instant::now();
                            let reg = self.step_regulate(
                                &pros.dur_logits,
                                &pros.features,
                                &enc.text_features,
                                speed,
                                cache,
                            )?;
                            // step_regulate already has an inherent submit+sync;
                            // flush any remaining work.
                            crate::gpu_scope::flush()?;
                            let regulate = t0.elapsed();

                            let t_mel = reg.t_mel;
                            let total_samples =
                                generator_total_samples(t_mel, upsample_factor)?;
                            if !self.seg_f0.contains_key(t_mel) {
                                cache_misses += 1;
                            }
                            if !self.seg_generator.contains_key(total_samples) {
                                cache_misses += 1;
                            }

                            let t0 = Instant::now();
                            let f0e = self.step_predict_f0_energy(
                                &reg.aligned_dur,
                                &prosody_style,
                                t_mel,
                                cache,
                            )?;
                            crate::gpu_scope::flush()?;
                            let f0_energy = t0.elapsed();

                            let t0 = Instant::now();
                            let har_source = self.step_harmonic_source(
                                &f0e.f0,
                                &f0e.energy,
                                t_mel,
                                cache,
                            )?;
                            crate::gpu_scope::flush()?;
                            let harmonic = t0.elapsed();

                            let t0 = Instant::now();
                            let generator = self.step_generate(
                                &reg.regulated,
                                &f0e.f0,
                                &f0e.energy,
                                &decoder_style,
                                &har_source,
                                t_mel,
                                cache,
                            )?;
                            crate::gpu_scope::flush()?;
                            let generate = t0.elapsed();

                            let t0 = Instant::now();
                            let audio = self.step_istft(
                                &generator.magnitude,
                                &generator.phase,
                                cache,
                            )?;
                            crate::gpu_scope::flush()?;
                            let istft = t0.elapsed();

                            Ok((
                                audio, encode, prosody, regulate, f0_energy, harmonic,
                                generate, istft,
                            ))
                        },
                    )?;

                // Pipeline exit: all GPU work already flushed per-step above.
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

                let t0 = Instant::now();
                let certificate = self.step_verify(&audio)?;
                let verify = t0.elapsed();

                let timing = GpuTimingReport {
                    encode,
                    prosody,
                    regulate,
                    f0_energy,
                    harmonic,
                    generate,
                    istft,
                    verify,
                    total: wall.elapsed(),
                    cache_misses,
                };
                Ok((audio, certificate, timing))
            })();

        if result.is_err() {
            crate::gpu_scope::discard_pending_batch();
        }

        result
    }

    /// All 8 segment caches as a named array for iteration.
    fn all_segments(&self) -> [&super::segment_cache::SegmentCache; 8] {
        [
            &self.seg_plbert,
            &self.seg_text,
            &self.seg_prosody,
            &self.seg_f0,
            &self.seg_generator,
            &self.seg_regulate,
            &self.seg_sinegen_pre,
            &self.seg_sinegen_post,
        ]
    }

    /// Returns the number of GPU dispatches for the most-recently-used
    /// compiled model in each segment cache.
    #[must_use]
    pub fn total_dispatches(&self) -> usize {
        self.all_segments()
            .iter()
            .filter_map(|c| c.most_recent())
            .map(|(_, s)| s.num_dispatches())
            .sum()
    }

    /// Returns the estimated Metal kernel launch count across compiled segments.
    ///
    /// Unlike `total_dispatches()`, this expands each dispatch plan to count
    /// individual Metal kernel launches (e.g., FusedResBlock → 5-10 dispatches).
    ///
    /// **This is a planner estimate** using `estimated_metal_dispatches()` for
    /// NativeOps and `build_dispatch_plan().len()` for IR ops. It does NOT
    /// include eager-path dispatches (cumsum_kahan, forward STFT, iSTFT).
    /// For the actual runtime dispatch count, use
    /// [`synthesize_with_stats`] → [`DispatchStats::compute_encodings`].
    #[must_use]
    pub fn total_metal_dispatches(&self) -> usize {
        self.all_segments()
            .iter()
            .filter_map(|c| c.most_recent())
            .map(|(_, s)| s.num_metal_dispatches())
            .sum()
    }

    /// Estimated encoding events across all compiled segments.
    ///
    /// Uses [`CompiledModel::num_encoding_events()`] per segment, which counts
    /// compute dispatches (1 per IR Dispatch, `estimated_metal_dispatches()` per
    /// NativeOp) plus blit relocations. This tracks `TOTAL_ENCODINGS + TOTAL_BLITS`
    /// at runtime.
    ///
    /// Excludes eager-path dispatches (cumsum_kahan, iSTFT, harmonic source).
    /// For the actual runtime count, use [`synthesize_with_stats`].
    ///
    /// See #1815 D5.1.
    #[must_use]
    pub fn total_encoding_events(&self) -> usize {
        self.all_segments()
            .iter()
            .filter_map(|c| c.most_recent())
            .map(|(_, s)| s.num_encoding_events())
            .sum()
    }

    /// Returns per-segment dispatch counts for diagnostic reporting.
    ///
    /// Returns a [`DispatchSummary`] with named fields for each pipeline
    /// segment. Segments not yet compiled return 0.
    #[must_use]
    pub fn dispatch_summary(&self) -> DispatchSummary {
        DispatchSummary {
            plbert: self
                .seg_plbert
                .most_recent()
                .map_or(0, |(_, s)| s.num_dispatches()),
            text_encoder: self
                .seg_text
                .most_recent()
                .map_or(0, |(_, s)| s.num_dispatches()),
            prosody: self
                .seg_prosody
                .most_recent()
                .map_or(0, |(_, s)| s.num_dispatches()),
            f0_energy: self
                .seg_f0
                .most_recent()
                .map_or(0, |(_, s)| s.num_dispatches()),
            generator: self
                .seg_generator
                .most_recent()
                .map_or(0, |(_, s)| s.num_dispatches()),
            regulate: self
                .seg_regulate
                .most_recent()
                .map_or(0, |(_, s)| s.num_dispatches()),
            sinegen_pre: self
                .seg_sinegen_pre
                .most_recent()
                .map_or(0, |(_, s)| s.num_dispatches()),
            sinegen_post: self
                .seg_sinegen_post
                .most_recent()
                .map_or(0, |(_, s)| s.num_dispatches()),
        }
    }

    /// Detailed per-segment dispatch census with operation-type breakdown.
    ///
    /// Returns a [`DispatchCensus`] with per-segment counts of NativeOps
    /// (by variant), IR Dispatches (by kernel name), RuntimeOps, and
    /// zero-cost steps. Also identifies adjacent dispatch pairs that are
    /// candidates for fusion.
    ///
    /// This is the primary tool for identifying dispatch reduction
    /// opportunities. Use the output to prioritize new peephole passes.
    ///
    /// Part of #4264.
    #[must_use]
    pub fn dispatch_census(&self) -> DispatchCensus {
        use std::collections::BTreeMap;
        let audit = self.per_segment_step_audit();

        let mut segments = Vec::with_capacity(audit.len());
        let mut total_dispatches = 0usize;
        let mut total_metal = 0usize;
        let mut total_zero_cost = 0usize;
        let mut total_steps = 0usize;

        for (seg_name, steps, dispatches, metal_dispatches) in audit {
            let mut native_counts: BTreeMap<String, usize> = BTreeMap::new();
            let mut ir_counts: BTreeMap<String, usize> = BTreeMap::new();
            let mut runtime_ops = 0usize;
            let mut zero_cost = 0usize;
            let mut fusion_candidates = Vec::new();
            let mut prev_dispatch: Option<String> = None;

            for (_, step_type, detail, _metal) in &steps {
                match *step_type {
                    "NativeOp" => {
                        *native_counts.entry(detail.clone()).or_insert(0) += 1;
                        let label = format!("[NativeOp] {detail}");
                        if let Some(ref prev) = prev_dispatch {
                            fusion_candidates.push((prev.clone(), label.clone()));
                        }
                        prev_dispatch = Some(label);
                    }
                    "Dispatch" => {
                        *ir_counts.entry(detail.clone()).or_insert(0) += 1;
                        let label = format!("[IR] {detail}");
                        if let Some(ref prev) = prev_dispatch {
                            fusion_candidates.push((prev.clone(), label.clone()));
                        }
                        prev_dispatch = Some(label);
                    }
                    "RuntimeOp" => {
                        runtime_ops += 1;
                        let label = format!("[RuntimeOp] {detail}");
                        if let Some(ref prev) = prev_dispatch {
                            fusion_candidates.push((prev.clone(), label.clone()));
                        }
                        prev_dispatch = Some(label);
                    }
                    _ => {
                        zero_cost += 1;
                        prev_dispatch = None;
                    }
                }
            }

            total_dispatches += dispatches;
            total_metal += metal_dispatches;
            total_zero_cost += zero_cost;
            total_steps += steps.len();

            segments.push(SegmentCensus {
                name: seg_name,
                dispatches,
                metal_dispatches,
                native_ops: native_counts.into_iter().collect(),
                ir_dispatches: ir_counts.into_iter().collect(),
                runtime_ops,
                zero_cost,
                total_steps: steps.len(),
                fusion_candidates,
            });
        }

        DispatchCensus {
            segments,
            total_dispatches,
            total_metal_dispatches: total_metal,
            total_zero_cost,
            total_steps,
        }
    }

    /// Total GPU weight buffer bytes across all segment caches.
    ///
    /// Sums `MetalBuffer::len()` for all shared weight buffers in all 8 segment
    /// caches. Returns 0 if no segments have been compiled yet.
    ///
    /// For `clone_dispatch()` instances, this reports the same byte count as
    /// the parent because aliased buffers have the same `len_bytes` — the
    /// underlying GPU memory is shared (ARC reference counting, zero-copy).
    ///
    /// Part of #2740.
    #[must_use]
    pub fn gpu_weight_bytes(&self) -> usize {
        self.all_segments()
            .iter()
            .map(|c| c.shared_weight_bytes())
            .sum()
    }

    /// Number of shared GPU weight buffers across all segment caches.
    #[must_use]
    pub fn gpu_weight_count(&self) -> usize {
        self.all_segments()
            .iter()
            .map(|c| c.shared_weight_count())
            .sum()
    }

    /// Reference count of the shared state (`Arc<SharedKokoroState>`).
    ///
    /// Returns 1 for a standalone instance, N+1 after N `clone_dispatch()` calls.
    /// Used to verify that clones share the same model weights.
    ///
    /// Part of #2740.
    #[must_use]
    pub fn shared_state_refcount(&self) -> usize {
        Arc::strong_count(&self.shared)
    }

    /// Per-segment, per-step audit of all compiled dispatches (#4252).
    ///
    /// Returns `(segment_name, steps, dispatches, metal_dispatches)` for each
    /// compiled segment. `steps` is a `Vec<(step_idx, step_type, detail, metal_count)>`
    /// where `step_type` is one of "Dispatch", "NativeOp", "Passthrough",
    /// "NarrowView", "InputForward", "IdentityPassthrough", "ConstantValue",
    /// "RuntimeOp" and `detail` is the kernel/op name, and `metal_count` is the
    /// estimated Metal kernel launches for that step.
    ///
    /// Part of #4252.
    #[must_use]
    pub fn per_segment_step_audit(
        &self,
    ) -> Vec<(
        String,
        Vec<(usize, &'static str, String, usize)>,
        usize,
        usize,
    )> {
        use nn_dsl::ir::ScalarType;
        use nn_dsl::trace_compile::CompiledStep;
        use nn_dsl::build_dispatch_plan;

        let segments = [
            ("plbert", &self.seg_plbert),
            ("text_encoder", &self.seg_text),
            ("prosody", &self.seg_prosody),
            ("f0_energy", &self.seg_f0),
            ("generator", &self.seg_generator),
            ("regulate", &self.seg_regulate),
            ("sinegen_pre", &self.seg_sinegen_pre),
            ("sinegen_post", &self.seg_sinegen_post),
        ];

        segments
            .into_iter()
            .filter_map(|(name, cache)| {
                cache.most_recent().map(|(_, model)| {
                    let mut step_infos = Vec::new();
                    let dispatches = model.num_dispatches();
                    let metal_dispatches = model.num_metal_dispatches();
                    for (i, step) in model.steps().iter().enumerate() {
                        let (step_type, detail, metal) = match step {
                            CompiledStep::Dispatch { kernel, .. } => {
                                let count = build_dispatch_plan(
                                    kernel.def(),
                                    ScalarType::F32,
                                )
                                .map(|(plan, _)| plan.len())
                                .unwrap_or(1);
                                ("Dispatch", kernel.name().to_string(), count)
                            }
                            CompiledStep::NativeOp { op, .. } => {
                                let count = op.estimated_metal_dispatches();
                                ("NativeOp", op.variant_name().to_string(), count)
                            }
                            CompiledStep::Passthrough { op_name, .. } => {
                                ("Passthrough", op_name.clone(), 0)
                            }
                            CompiledStep::NarrowView { .. } => {
                                ("NarrowView", "narrow".to_string(), 0)
                            }
                            CompiledStep::InputForward => {
                                ("InputForward", "input".to_string(), 0)
                            }
                            CompiledStep::IdentityPassthrough => {
                                ("IdentityPass", "identity".to_string(), 0)
                            }
                            CompiledStep::ConstantValue { value, .. } => {
                                ("ConstantValue", format!("const({value})"), 0)
                            }
                            CompiledStep::RuntimeOp { op } => {
                                ("RuntimeOp", format!("{op:?}"), 1)
                            }
                            _ => ("Unknown", "unknown".to_string(), 0),
                        };
                        step_infos.push((i, step_type, detail, metal));
                    }
                    (name.to_string(), step_infos, dispatches, metal_dispatches)
                })
            })
            .collect()
    }

    /// Count how many of the 8 segment caches have at least one cached entry.
    ///
    /// Returns 0..=8. Used by warmup/precompile tests to verify that all
    /// pipeline stages compiled successfully. Part of #4187.
    #[must_use]
    pub fn total_cached_segments(&self) -> usize {
        let mut count = 0;
        if self.seg_plbert.len() > 0 { count += 1; }
        if self.seg_text.len() > 0 { count += 1; }
        if self.seg_prosody.len() > 0 { count += 1; }
        if self.seg_f0.len() > 0 { count += 1; }
        if self.seg_generator.len() > 0 { count += 1; }
        if self.seg_regulate.len() > 0 { count += 1; }
        if self.seg_sinegen_pre.len() > 0 { count += 1; }
        if self.seg_sinegen_post.len() > 0 { count += 1; }
        count
    }
}
