// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PeepholeConfig optimizer search for all compiled Kokoro segments.
//!
//! Re-traces each segment to obtain the [`ComputationGraph`], then runs
//! [`optimize_plan_with_cost`] to exhaustively search all 2048 PeepholeConfig
//! combinations per segment. Returns per-segment results showing the optimal
//! config and dispatch count improvement.
//!
//! Part of #3828 Phase 2C.

use std::time::Duration;

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{with_nan_check_policy, Linear, NanCheckPolicy};
use nn_dsl::{optimize_plan_with_cost, CostModel, OptimizationResult};

use crate::cache::PipelineCache;

use super::{gpu, model_device, prepare_synthesis_inputs, CompiledKokoro, CompiledKokoroError};

/// Per-segment optimizer search result.
#[derive(Clone, Debug)]
pub struct SegmentOptimizerResult {
    /// Segment name (e.g., "plbert", "text", "generator").
    pub segment_name: String,
    /// Optimization result with best config, dispatch count, and cost.
    pub optimization: OptimizationResult,
}

/// Trace a segment and run the optimizer search.
///
/// Returns `None` if tracing fails (segment is skipped).
fn optimize_segment(
    name: &str,
    trace_result: nn_core::Result<(DynTensor, nn_core::dyn_tensor::trace::ComputationGraph)>,
    cost_model: &CostModel,
    per_segment_budget: Duration,
) -> Option<SegmentOptimizerResult> {
    let (_out, graph) = match trace_result {
        Ok(v) => v,
        Err(e) => {
            eprintln!("  [optimizer] SKIP {name}: trace failed: {e}");
            return None;
        }
    };

    match optimize_plan_with_cost(&graph, cost_model, per_segment_budget) {
        Ok(result) => Some(SegmentOptimizerResult {
            segment_name: name.to_string(),
            optimization: result,
        }),
        Err(e) => {
            eprintln!("  [optimizer] SKIP {name}: baseline compilation failed: {e}");
            None
        }
    }
}

impl CompiledKokoro {
    /// Run PeepholeConfig optimizer search on all Kokoro segments.
    ///
    /// Traces each segment, then exhaustively searches all 2048 PeepholeConfig
    /// combinations per segment to find the optimal dispatch count. Each segment
    /// gets its own time budget.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - `[B, T]` token indices (same as `synthesize`).
    /// * `style` - `[B, 2*style_dim]` voice embedding (same as `synthesize`).
    /// * `speed` - Speaking rate multiplier (same as `synthesize`).
    /// * `cache` - Metal pipeline cache.
    /// * `per_segment_budget` - Maximum time to spend optimizing each segment.
    ///
    /// Part of #3828 Phase 2C.
    pub fn segment_optimizer_search(
        &mut self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
        per_segment_budget: Duration,
    ) -> Result<Vec<SegmentOptimizerResult>, CompiledKokoroError> {
        let cost_model = CostModel::apple_m4();
        let mut results = Vec::with_capacity(8);

        let style_split = prepare_synthesis_inputs(self, input_ids, style, speed)?;
        let decoder_style = style_split.decoder_style.to_device(&gpu())?;
        let prosody_style = style_split.prosody_style.to_device(&gpu())?;

        // Determine the device where model weights live. Trace inputs must be
        // on this device so forward passes inside `trace_graph` don't hit
        // device-mismatch errors (e.g., GPU inputs vs CPU model weights when
        // `CompiledKokoro::load()` keeps weights on CPU for RSS optimization).
        // The computation graph is device-independent — same structure regardless
        // of whether tracing runs on CPU or GPU. (#4250)
        let trace_dev = model_device(self.shared.model.as_ref());

        with_nan_check_policy(NanCheckPolicy::Skip, || -> Result<(), CompiledKokoroError> {
            // Steps 1-2: Encode (PlBert + TextEncoder).
            let enc = self.step_encode(input_ids, cache)?;

            // Optimize: plbert
            if let Some(r) = optimize_segment(
                "plbert",
                super::trace_fns::trace_seg_plbert(self, input_ids),
                &cost_model,
                per_segment_budget,
            ) {
                results.push(r);
            }

            // Optimize: text
            if let Some(r) = optimize_segment(
                "text",
                super::trace_fns::trace_seg_text(self, input_ids),
                &cost_model,
                per_segment_budget,
            ) {
                results.push(r);
            }

            // Step 3: ProsodyPredictor
            let pros = self.step_predict_prosody(
                &enc.bert_features,
                &prosody_style,
                enc.seq_len,
                cache,
            )?;

            // Optimize: prosody — move inputs to model device for tracing.
            let bert_feat_trace = enc.bert_features.to_device(&trace_dev)?;
            let prosody_style_trace = prosody_style.to_device(&trace_dev)?;
            if let Some(r) = optimize_segment(
                "prosody",
                super::trace_fns::trace_seg_prosody(
                    self,
                    &bert_feat_trace,
                    &prosody_style_trace,
                ),
                &cost_model,
                per_segment_budget,
            ) {
                results.push(r);
            }

            // Step 4: Regulate
            let reg = self.step_regulate(
                &pros.dur_logits,
                &pros.features,
                &enc.text_features,
                speed,
                cache,
            )?;

            // Optimize: regulate — both inputs must be on same device.
            let dur_logits_trace = pros.dur_logits.to_device(&trace_dev)?;
            let speed_inv = DynTensor::full(
                &[1, 1],
                1.0 / f64::from(speed),
                nn_core::DType::F32,
                &trace_dev,
            )?;
            let max_dur = self.config().max_dur as f64;
            if let Some(r) = optimize_segment(
                "regulate",
                super::trace_fns::trace_seg_regulate(&dur_logits_trace, &speed_inv, max_dur),
                &cost_model,
                per_segment_budget,
            ) {
                results.push(r);
            }

            // Step 5: F0/Energy prediction
            let f0e = self.step_predict_f0_energy(
                &reg.aligned_dur,
                &prosody_style,
                reg.t_mel,
                cache,
            )?;

            // Optimize: f0 — move inputs to model device for tracing.
            let aligned_dur_trace = reg.aligned_dur.to_device(&trace_dev)?;
            if let Some(r) = optimize_segment(
                "f0",
                super::trace_fns::trace_seg_f0(
                    self,
                    &aligned_dur_trace,
                    &prosody_style_trace,
                ),
                &cost_model,
                per_segment_budget,
            ) {
                results.push(r);
            }

            // Step 6: Harmonic source
            let har_source =
                self.step_harmonic_source(&f0e.f0, &f0e.energy, reg.t_mel, cache)?;

            // Optimize: sinegen_pre + sinegen_post
            // These segments don't use model weights directly, but inputs
            // must be on a consistent device. Use trace_dev for uniformity.
            //
            // f0 from step_predict_f0_energy is [B, 1, 2*T_mel]. SineGen
            // expects [B, T_frames, 1] (channel-last). Transpose to match
            // build_harmonic_source which does f0_gpu.transpose(1, 2).
            let f0_trace = f0e.f0.to_device(&trace_dev)?.transpose(1, 2)?;
            if let Some(sm) = self.shared.source_module.as_ref() {
                let sg = sm.sine_gen();
                // upp must include hop_length factor, matching
                // build_harmonic_source: source_upsample = product * (n_fft/4).
                let n_fft = self.config().n_fft;
                let hop_length = n_fft / 4;
                let upp: usize = self.config().upsample_rates.iter().product::<usize>() * hop_length;
                if let Some(r) = optimize_segment(
                    "sinegen_pre",
                    super::trace_fns::trace_seg_sinegen_pre(
                        &f0_trace,
                        upp,
                        sg.sampling_rate(),
                        sg.n_channels(),
                    ),
                    &cost_model,
                    per_segment_budget,
                ) {
                    results.push(r);
                }

                let t_frames = f0_trace.dim(1).unwrap_or(1);
                let n_ch = sg.n_channels();
                let batch = f0_trace.dim(0).unwrap_or(1);
                let cum_trace = DynTensor::zeros(
                    &[batch, t_frames, n_ch],
                    nn_core::DType::F32,
                    &trace_dev,
                )?;
                // sinegen_post uses l_linear from SourceModule — move to trace
                // device so weights match traced inputs. (#4250)
                let l_w = sm.linear().weight().to_device(&trace_dev)?;
                let l_b = sm
                    .linear()
                    .bias()
                    .map(|b| b.to_device(&trace_dev))
                    .transpose()?;
                let l_linear_trace = Linear::new(l_w, l_b)?;
                let voiced_threshold = f64::from(sg.voiced_threshold());
                if let Some(r) = optimize_segment(
                    "sinegen_post",
                    super::trace_fns::trace_seg_sinegen_post(
                        &cum_trace,
                        &f0_trace,
                        &l_linear_trace,
                        upp,
                        sg.sine_amp(),
                        voiced_threshold,
                    ),
                    &cost_model,
                    per_segment_budget,
                ) {
                    results.push(r);
                }
            }

            // Optimize: generator — move all inputs to model device.
            // f0 for the generator must be [B, 1, 2*T_mel] (channel-first),
            // NOT the transposed f0_trace [B, 2*T_mel, 1] used by sinegen.
            // FullDecoder::forward expects f0_curve [B, 1, 2T] for its F0_conv
            // (Conv1d with in_channels=1). Passing the transposed shape causes
            // a shape mismatch because dim(1)=2*T_mel != in_channels=1.
            let regulated_trace = reg.regulated.to_device(&trace_dev)?;
            let f0_gen_trace = f0e.f0.to_device(&trace_dev)?;
            let energy_trace = f0e.energy.to_device(&trace_dev)?;
            let decoder_style_trace = decoder_style.to_device(&trace_dev)?;
            let har_source_trace = har_source.to_device(&trace_dev)?;
            if let Some(r) = optimize_segment(
                "generator",
                super::trace_fns::trace_seg_generator(
                    self,
                    &regulated_trace,
                    &f0_gen_trace,
                    &energy_trace,
                    &decoder_style_trace,
                    &har_source_trace,
                ),
                &cost_model,
                per_segment_budget,
            ) {
                results.push(r);
            }

            Ok(())
        })?;

        Ok(results)
    }
}
