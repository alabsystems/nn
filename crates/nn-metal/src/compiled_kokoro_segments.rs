// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Segment compilation for [`CompiledKokoro`].
//!
//! Each `ensure_seg_*` checks the per-segment LRU cache. On miss it calls the
//! corresponding `compile_seg_*`, which delegates to [`trace_fns`] for the
//! trace closure and then compiles via `compile_with_shared`.
//!
//! Part of #2465, #2218, #2744.

use std::collections::HashMap;

use nn_core::dyn_tensor::DynTensor;
use nn_core::Result;

use crate::buffer::MetalBuffer;
use crate::cache::PipelineCache;
use crate::compiled_model::CompiledModel;

use nn_core::mixed_precision::MixedPrecisionPolicy;

use super::{model_device, seg_cache_miss, seg_compile_err, CompiledKokoro};

impl CompiledKokoro {
    /// Resolve the autocast policy for a named segment.
    ///
    /// When `segment_autocast` is `Some`, delegates to
    /// [`F16AutocastConfig::policy_for_segment()`] for per-segment control.
    /// Otherwise falls back to the uniform `autocast_policy`.
    ///
    /// Part of #4269.
    fn resolve_segment_autocast(&self, segment_name: &str) -> Option<&MixedPrecisionPolicy> {
        if let Some(ref config) = self.segment_autocast {
            config.policy_for_segment(segment_name)
        } else {
            self.autocast_policy.as_ref()
        }
    }
}

// -- Cache lookup (ensure_seg_*) ----------------------------------------------

impl CompiledKokoro {
    /// Compile or reuse segment 0: PlBert + bert_encoder (token IDs → bert_features).
    ///
    /// PlBert is a 12-layer ALBERT transformer producing contextual embeddings.
    /// The trace system requires all tensor inputs to have trace_node_ids, but
    /// PlBert creates position_ids (via `arange_u32`) and token_type_ids (via
    /// `zeros`) internally — these have no trace_ids. Solution: pre-compute
    /// position and token-type embeddings outside the trace scope as additional
    /// trace inputs. They are constant per seq_len (the cache key).
    ///
    /// bert_encoder (Linear(768, d_en) + transpose) is folded into this segment
    /// to eliminate its eager dispatch overhead.
    ///
    /// Part of #2744, #2218.
    pub(super) fn ensure_seg_plbert(
        &mut self,
        seq_len: usize,
        input_ids: &DynTensor,
        cache: &PipelineCache,
    ) -> Result<&CompiledModel> {
        // Single scan: get() promotes on hit; most_recent() is O(1).
        if self.seg_plbert.get(seq_len).is_none() {
            let shared = self.seg_plbert.shared_weights();
            let compiled = self.compile_seg_plbert(input_ids, cache, shared)?;
            self.seg_plbert.insert(seq_len, compiled);
        }
        self.seg_plbert
            .most_recent()
            .map(|(_, v)| v)
            .ok_or_else(|| seg_cache_miss("plbert"))
    }

    /// Compile or reuse segment 1: TextEncoder (token IDs → features).
    ///
    /// `input_ids`: `[B, T]` token indices, already converted to F32 on CPU.
    /// TextEncoder has its own Embedding layer (architecture correction #5),
    /// so it takes raw token IDs, not bert_encoder output.
    pub(super) fn ensure_seg_text(
        &mut self,
        seq_len: usize,
        input_ids: &DynTensor,
        cache: &PipelineCache,
    ) -> Result<&CompiledModel> {
        if self.seg_text.get(seq_len).is_none() {
            let shared = self.seg_text.shared_weights();
            let compiled = self.compile_seg_text(input_ids, cache, shared)?;
            self.seg_text.insert(seq_len, compiled);
        }
        self.seg_text
            .most_recent()
            .map(|(_, v)| v)
            .ok_or_else(|| seg_cache_miss("text"))
    }

    /// Compile or reuse segment 2: ProsodyPredictor.
    ///
    /// `bert_features`: `[B, d_en, T]` — ALBERT contextual features from
    /// bert_encoder(PlBert output). ProsodyPredictor needs these (not TextEncoder
    /// features) for duration prediction. Fix: #2511.
    pub(super) fn ensure_seg_prosody(
        &mut self,
        seq_len: usize,
        bert_features: &DynTensor,
        style: &DynTensor,
        cache: &PipelineCache,
    ) -> Result<&CompiledModel> {
        if self.seg_prosody.get(seq_len).is_none() {
            let dev = model_device(self.shared.model.as_ref());
            let bert_dev = bert_features.to_device(&dev)?;
            let style_dev = style.to_device(&dev)?;
            let shared = self.seg_prosody.shared_weights();
            let compiled = self.compile_seg_prosody(&bert_dev, &style_dev, cache, shared)?;
            self.seg_prosody.insert(seq_len, compiled);
        }
        self.seg_prosody
            .most_recent()
            .map(|(_, v)| v)
            .ok_or_else(|| seg_cache_miss("prosody"))
    }

    /// Compile or reuse segment 3: F0EnergyPredictor.
    pub(super) fn ensure_seg_f0(
        &mut self,
        t_mel: usize,
        aligned: &DynTensor,
        style: &DynTensor,
        cache: &PipelineCache,
    ) -> Result<&CompiledModel> {
        if self.seg_f0.get(t_mel).is_none() {
            let dev = model_device(self.shared.model.as_ref());
            let aligned_dev = aligned.to_device(&dev)?;
            let style_dev = style.to_device(&dev)?;
            let shared = self.seg_f0.shared_weights();
            let compiled = self.compile_seg_f0(&aligned_dev, &style_dev, cache, shared)?;
            self.seg_f0.insert(t_mel, compiled);
        }
        self.seg_f0
            .most_recent()
            .map(|(_, v)| v)
            .ok_or_else(|| seg_cache_miss("f0"))
    }

    /// Compile or reuse segment 4: Generator.
    ///
    /// `regulated`: `[B, d_en, T_mel]` — TextEncoder features after
    /// length_regulate. FullDecoder receives these (not prosody features).
    /// Fix: #2511.
    pub(super) fn ensure_seg_generator(
        &mut self,
        total_samples: usize,
        regulated: &DynTensor,
        f0: &DynTensor,
        energy: &DynTensor,
        decoder_style: &DynTensor,
        har_source: &DynTensor,
        cache: &PipelineCache,
    ) -> Result<&CompiledModel> {
        if self.seg_generator.get(total_samples).is_none() {
            let shared = self.seg_generator.shared_weights();
            let compiled = self.compile_seg_generator(
                regulated,
                f0,
                energy,
                decoder_style,
                har_source,
                cache,
                shared,
            )?;
            self.seg_generator.insert(total_samples, compiled);
        }
        self.seg_generator
            .most_recent()
            .map(|(_, v)| v)
            .ok_or_else(|| seg_cache_miss("generator"))
    }

    /// Compile or reuse segment 5: Regulate pre-readback elementwise chain.
    ///
    /// Pure elementwise ops (no model weights). Speed is passed as a tensor
    /// input so the segment is reusable across different speed values.
    /// Cache key: `seq_len` (phoneme count T from dur_logits).
    ///
    /// Part of #1815 Tier 6 D2b.
    pub(super) fn ensure_seg_regulate(
        &mut self,
        seq_len: usize,
        dur_logits: &DynTensor,
        speed_inv: &DynTensor,
        cache: &PipelineCache,
    ) -> Result<&CompiledModel> {
        if self.seg_regulate.get(seq_len).is_none() {
            let trace_device = dur_logits.device();
            // Regulate tracing is device-agnostic, but the traced DynTensor ops
            // still require both inputs to live on the same device. Warmup uses
            // CPU-loaded model weights, while the step API often passes GPU
            // logits, so normalize the scalar input to the logits device here.
            let speed_inv_trace = if speed_inv.device() == trace_device {
                speed_inv.clone()
            } else {
                speed_inv.to_device(&trace_device)?
            };
            let shared = self.seg_regulate.shared_weights();
            let compiled =
                self.compile_seg_regulate(dur_logits, &speed_inv_trace, cache, shared)?;
            self.seg_regulate.insert(seq_len, compiled);
        }
        self.seg_regulate
            .most_recent()
            .map(|(_, v)| v)
            .ok_or_else(|| seg_cache_miss("regulate"))
    }

    /// Compile or reuse segment 5a: SineGen pre-cumsum (multi-output).
    ///
    /// `f0`: `[B, T_frames, 1]` — fundamental frequency.
    /// Cache key: `t_frames`. Part of #1815 Tier 6 D2.
    pub(super) fn ensure_seg_sinegen_pre(
        &mut self,
        t_frames: usize,
        f0: &DynTensor,
        upp: usize,
        cache: &PipelineCache,
    ) -> Result<&CompiledModel> {
        if self.seg_sinegen_pre.get(t_frames).is_none() {
            let shared = self.seg_sinegen_pre.shared_weights();
            let compiled = self.compile_seg_sinegen_pre(f0, upp, cache, shared)?;
            self.seg_sinegen_pre.insert(t_frames, compiled);
        }
        self.seg_sinegen_pre
            .most_recent()
            .map(|(_, v)| v)
            .ok_or_else(|| seg_cache_miss("sinegen_pre"))
    }

    /// Compile or reuse segment 5b: SineGen post-cumsum (single-output).
    ///
    /// `cum_gpu`: `[B, T_frames, n_ch]` — cumulative sum from eager step.
    /// `f0_gpu`: `[B, T_frames, 1]` — F0 for voiced mask (folded into trace).
    /// Cache key: `t_frames`. Part of #1815 Tier 6 D3.
    pub(super) fn ensure_seg_sinegen_post(
        &mut self,
        t_frames: usize,
        cum_gpu: &DynTensor,
        f0_gpu: &DynTensor,
        upp: usize,
        voiced_threshold: f64,
        cache: &PipelineCache,
    ) -> Result<&CompiledModel> {
        if self.seg_sinegen_post.get(t_frames).is_none() {
            let shared = self.seg_sinegen_post.shared_weights();
            let compiled =
                self.compile_seg_sinegen_post(cum_gpu, f0_gpu, upp, voiced_threshold, cache, shared)?;
            self.seg_sinegen_post.insert(t_frames, compiled);
        }
        self.seg_sinegen_post
            .most_recent()
            .map(|(_, v)| v)
            .ok_or_else(|| seg_cache_miss("sinegen_post"))
    }
}

// -- Segment trace + compile --------------------------------------------------
//
// Each method delegates to `trace_fns::trace_seg_*()` for the trace closure,
// then compiles via `compile_with_shared()`.

impl CompiledKokoro {
    /// Trace + compile segment 0: PlBert + bert_encoder → bert_features.
    fn compile_seg_plbert(
        &self,
        input_ids: &DynTensor,
        cache: &PipelineCache,
        shared: Option<&HashMap<(usize, String), MetalBuffer>>,
    ) -> Result<CompiledModel> {
        let (_out, graph) = super::trace_fns::trace_seg_plbert(self, input_ids)
            .map_err(|e| seg_compile_err("plbert", e))?;
        compile_with_shared(
            &graph,
            cache,
            shared,
            self.mixed_precision,
            self.resolve_segment_autocast("plbert"),
            self.peephole_configs.get("plbert"),
            self.shape_policy,
        )
        .map_err(|e| seg_compile_err("plbert", e))
    }

    /// Trace + compile segment 1: TextEncoder (token IDs → features).
    fn compile_seg_text(
        &self,
        input_ids: &DynTensor,
        cache: &PipelineCache,
        shared: Option<&HashMap<(usize, String), MetalBuffer>>,
    ) -> Result<CompiledModel> {
        let (_out, graph) = super::trace_fns::trace_seg_text(self, input_ids)
            .map_err(|e| seg_compile_err("text", e))?;
        compile_with_shared(
            &graph,
            cache,
            shared,
            self.mixed_precision,
            self.resolve_segment_autocast("text"),
            self.peephole_configs.get("text"),
            self.shape_policy,
        )
        .map_err(|e| seg_compile_err("text", e))
    }

    /// Trace + compile segment 2: ProsodyPredictor (multi-output).
    fn compile_seg_prosody(
        &self,
        bert_features: &DynTensor,
        style: &DynTensor,
        cache: &PipelineCache,
        shared: Option<&HashMap<(usize, String), MetalBuffer>>,
    ) -> Result<CompiledModel> {
        let (_out, graph) = super::trace_fns::trace_seg_prosody(self, bert_features, style)
            .map_err(|e| seg_compile_err("prosody", e))?;
        compile_with_shared(
            &graph,
            cache,
            shared,
            self.mixed_precision,
            self.resolve_segment_autocast("prosody"),
            self.peephole_configs.get("prosody"),
            self.shape_policy,
        )
        .map_err(|e| seg_compile_err("prosody", e))
    }

    /// Trace + compile segment 3: F0EnergyPredictor (multi-output).
    fn compile_seg_f0(
        &self,
        aligned: &DynTensor,
        style: &DynTensor,
        cache: &PipelineCache,
        shared: Option<&HashMap<(usize, String), MetalBuffer>>,
    ) -> Result<CompiledModel> {
        let (_out, graph) = super::trace_fns::trace_seg_f0(self, aligned, style)
            .map_err(|e| seg_compile_err("f0", e))?;
        compile_with_shared(
            &graph,
            cache,
            shared,
            self.mixed_precision,
            self.resolve_segment_autocast("f0"),
            self.peephole_configs.get("f0"),
            self.shape_policy,
        )
        .map_err(|e| seg_compile_err("f0", e))
    }

    /// Trace + compile segment 5: Regulate (elementwise, multi-output).
    ///
    /// No model weights — segment has 0 weight buffers. Compilation is fast
    /// (just elementwise ops). Mixed-precision/autocast still applied for
    /// consistency with other segments.
    fn compile_seg_regulate(
        &self,
        dur_logits: &DynTensor,
        speed_inv: &DynTensor,
        cache: &PipelineCache,
        shared: Option<&HashMap<(usize, String), MetalBuffer>>,
    ) -> Result<CompiledModel> {
        let max_dur = self.config().max_dur as f64;
        let (_out, graph) = super::trace_fns::trace_seg_regulate(dur_logits, speed_inv, max_dur)
            .map_err(|e| seg_compile_err("regulate", e))?;
        compile_with_shared(
            &graph,
            cache,
            shared,
            self.mixed_precision,
            self.resolve_segment_autocast("regulate"),
            self.peephole_configs.get("regulate"),
            self.shape_policy,
        )
        .map_err(|e| seg_compile_err("regulate", e))
    }

    /// Trace + compile segment 4: Generator (multi-output: magnitude + phase).
    fn compile_seg_generator(
        &self,
        regulated: &DynTensor,
        f0: &DynTensor,
        energy: &DynTensor,
        decoder_style: &DynTensor,
        har_source: &DynTensor,
        cache: &PipelineCache,
        shared: Option<&HashMap<(usize, String), MetalBuffer>>,
    ) -> Result<CompiledModel> {
        let (_out, graph) = super::trace_fns::trace_seg_generator(
            self,
            regulated,
            f0,
            energy,
            decoder_style,
            har_source,
        )
        .map_err(|e| seg_compile_err("generator", e))?;
        compile_with_shared(
            &graph,
            cache,
            shared,
            self.mixed_precision,
            self.resolve_segment_autocast("generator"),
            self.peephole_configs.get("generator"),
            self.shape_policy,
        )
        .map_err(|e| seg_compile_err("generator", e))
    }

    /// Trace + compile segment 5a: SineGen pre-cumsum (single-output).
    fn compile_seg_sinegen_pre(
        &self,
        f0: &DynTensor,
        upp: usize,
        cache: &PipelineCache,
        shared: Option<&HashMap<(usize, String), MetalBuffer>>,
    ) -> Result<CompiledModel> {
        let sm = self.shared.source_module.as_ref().ok_or_else(|| {
            seg_compile_err(
                "sinegen_pre",
                nn_core::TensorError::Unsupported("SourceModule not loaded".into()),
            )
        })?;
        let sg = sm.sine_gen();
        let (_out, graph) =
            super::trace_fns::trace_seg_sinegen_pre(f0, upp, sg.sampling_rate(), sg.n_channels())
                .map_err(|e| seg_compile_err("sinegen_pre", e))?;
        compile_with_shared(
            &graph,
            cache,
            shared,
            self.mixed_precision,
            self.resolve_segment_autocast("sinegen_pre"),
            self.peephole_configs.get("sinegen_pre"),
            self.shape_policy,
        )
        .map_err(|e| seg_compile_err("sinegen_pre", e))
    }

    /// Trace + compile segment 5b: SineGen post-cumsum (single-output).
    ///
    /// Voiced mask computation (unsqueeze→expand→reshape→gt→to_dtype) is now
    /// folded into this compiled segment. Previously 4+ eager dispatches.
    fn compile_seg_sinegen_post(
        &self,
        cum_gpu: &DynTensor,
        f0_gpu: &DynTensor,
        upp: usize,
        voiced_threshold: f64,
        cache: &PipelineCache,
        shared: Option<&HashMap<(usize, String), MetalBuffer>>,
    ) -> Result<CompiledModel> {
        let sm = self.shared.source_module.as_ref().ok_or_else(|| {
            seg_compile_err(
                "sinegen_post",
                nn_core::TensorError::Unsupported("SourceModule not loaded".into()),
            )
        })?;
        let (_out, graph) = super::trace_fns::trace_seg_sinegen_post(
            cum_gpu,
            f0_gpu,
            sm.linear(),
            upp,
            sm.sine_gen().sine_amp(),
            voiced_threshold,
        )
        .map_err(|e| seg_compile_err("sinegen_post", e))?;
        compile_with_shared(
            &graph,
            cache,
            shared,
            self.mixed_precision,
            self.resolve_segment_autocast("sinegen_post"),
            self.peephole_configs.get("sinegen_post"),
            self.shape_policy,
        )
        .map_err(|e| seg_compile_err("sinegen_post", e))
    }
}

// -- Shared-weight compilation helper -----------------------------------------

/// Compile a traced graph, optionally reusing GPU weight buffers (#2630).
/// When `mixed_precision` is true, non-NativeOp steps use F16.
/// When `autocast` is `Some`, uses per-op autocast (all buffers F32). #3085.
/// When `peephole` is `Some`, uses a custom [`PeepholeConfig`] for selective
/// fusion pass control (#3828 Phase 2B).
/// When `shape_policy` is `Polymorphic`, compiled segments accept variable
/// sequence dimensions without recompilation (#3873).
fn compile_with_shared(
    graph: &nn_core::dyn_tensor::trace::ComputationGraph,
    cache: &PipelineCache,
    shared: Option<&HashMap<(usize, String), MetalBuffer>>,
    mixed_precision: bool,
    autocast: Option<&MixedPrecisionPolicy>,
    peephole: Option<&nn_dsl::PeepholeConfig>,
    shape_policy: crate::compiled_model::ShapePolicy,
) -> Result<CompiledModel> {
    let mut b = CompiledModel::builder(graph, cache);
    if let Some(s) = shared {
        b = b.shared_weights(s);
    }
    if let Some(policy) = autocast {
        b = b.autocast(policy.clone());
    } else if mixed_precision {
        b = b.force_dtype(nn_core::DType::F16)?;
    }
    if let Some(config) = peephole {
        b = b.with_peephole_config(config.clone());
    }
    b = b.shape_policy(shape_policy);
    b.build()
}
