// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fusion gap analysis for all compiled Kokoro segments.
//!
//! Re-traces each segment to obtain the [`ComputationGraph`], compiles to a
//! [`CompiledPlan`], and runs [`analyze_fusion_gaps`] + [`CostModel::apple_m4()`]
//! cost estimation. Returns per-segment results for diagnostic reporting.
//!
//! The method runs the pipeline step-by-step to produce the intermediate
//! tensors needed as inputs for downstream segment traces (prosody needs
//! bert_features, generator needs regulated+f0+energy+style+harmonic_source,
//! etc.).
//!
//! Part of #3836.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{with_nan_check_policy, Linear, NanCheckPolicy};
use nn_dsl::{
    analyze_fusion_gaps, compile_trace_to_plan_with_fusion, CostEstimate, CostModel,
    FusionGapAnalysis,
};

use crate::cache::PipelineCache;

use super::{gpu, model_device, prepare_synthesis_inputs, CompiledKokoro, CompiledKokoroError};

/// Per-segment fusion gap analysis result.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SegmentGapAnalysis {
    /// Segment name (e.g., "plbert", "text", "generator").
    pub segment_name: String,
    /// Fusion gap analysis with per-gap blocker classification.
    pub gap_analysis: FusionGapAnalysis,
    /// Roofline-based cost estimate.
    pub cost_estimate: CostEstimate,
    /// Total dispatch steps in the compiled plan.
    pub dispatch_count: usize,
    /// Theoretical minimum dispatches if all closable gaps were fused.
    pub theoretical_minimum: usize,
}

/// Trace a segment, compile to plan, and run gap analysis + cost model.
///
/// Returns `None` if tracing or compilation fails (segment is skipped
/// rather than aborting the entire analysis).
fn analyze_segment(
    name: &str,
    trace_result: nn_core::Result<(DynTensor, nn_core::dyn_tensor::trace::ComputationGraph)>,
    cost_model: &CostModel,
) -> Option<SegmentGapAnalysis> {
    let (_out, graph) = match trace_result {
        Ok(v) => v,
        Err(e) => {
            eprintln!("  [gap_analysis] SKIP {name}: trace failed: {e}");
            return None;
        }
    };

    let plan = match compile_trace_to_plan_with_fusion(&graph) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  [gap_analysis] SKIP {name}: compile failed: {e}");
            return None;
        }
    };

    let gap_analysis = analyze_fusion_gaps(&plan, &graph);
    let cost_estimate = cost_model.estimate(&plan);
    let dispatch_count = gap_analysis.total_dispatches;
    let theoretical_minimum = gap_analysis.theoretical_minimum;

    Some(SegmentGapAnalysis {
        segment_name: name.to_string(),
        gap_analysis,
        cost_estimate,
        dispatch_count,
        theoretical_minimum,
    })
}

impl CompiledKokoro {
    /// Run fusion gap analysis on all 8 Kokoro segments.
    ///
    /// Traces each segment to obtain its [`ComputationGraph`], compiles to a
    /// [`CompiledPlan`] with fusion, and runs [`analyze_fusion_gaps`] +
    /// [`CostModel::apple_m4()`] cost estimation.
    ///
    /// The pipeline is executed step-by-step to produce intermediate tensors
    /// needed by downstream segments. Must be called with the same inputs
    /// used for synthesis. Segments that fail to trace or compile are skipped
    /// (not fatal).
    ///
    /// # Arguments
    ///
    /// * `input_ids` - `[B, T]` token indices (same as `synthesize`).
    /// * `style` - `[B, 2*style_dim]` voice embedding (same as `synthesize`).
    /// * `speed` - Speaking rate multiplier (same as `synthesize`).
    /// * `cache` - Metal pipeline cache.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Synthesize first to compile segments.
    /// let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache)?;
    /// // Then run gap analysis.
    /// let results = kokoro.segment_gap_analysis(&input_ids, &style, 1.0, &cache)?;
    /// for seg in &results {
    ///     eprintln!("{}: {} dispatches, {} theoretical min",
    ///         seg.segment_name, seg.dispatch_count, seg.theoretical_minimum);
    /// }
    /// ```
    ///
    /// Part of #3836.
    pub fn segment_gap_analysis(
        &mut self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<Vec<SegmentGapAnalysis>, CompiledKokoroError> {
        let cost_model = CostModel::apple_m4();
        let mut results = Vec::with_capacity(8);

        let style_split = prepare_synthesis_inputs(self, input_ids, style, speed)?;
        let decoder_style = style_split.decoder_style.to_device(&gpu())?;
        let prosody_style = style_split.prosody_style.to_device(&gpu())?;

        // Determine the device where model weights live so trace inputs match.
        // Same pattern as `ensure_seg_*` in compiled_kokoro_segments.rs. (#4250)
        let trace_dev = model_device(self.shared.model.as_ref());

        // Run the pipeline step-by-step inside NaN-skip scope (same as synthesize),
        // collecting intermediates needed for downstream segment traces.
        with_nan_check_policy(NanCheckPolicy::Skip, || -> Result<(), CompiledKokoroError> {
            // Steps 1-2: Encode (PlBert + TextEncoder).
            let enc = self.step_encode(input_ids, cache)?;

            // Trace + analyze: plbert
            if let Some(r) = analyze_segment(
                "plbert",
                super::trace_fns::trace_seg_plbert(self, input_ids),
                &cost_model,
            ) {
                results.push(r);
            }

            // Trace + analyze: text
            if let Some(r) = analyze_segment(
                "text",
                super::trace_fns::trace_seg_text(self, input_ids),
                &cost_model,
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

            // Trace + analyze: prosody — move inputs to model device.
            let bert_feat_trace = enc.bert_features.to_device(&trace_dev)?;
            let prosody_style_trace = prosody_style.to_device(&trace_dev)?;
            if let Some(r) = analyze_segment(
                "prosody",
                super::trace_fns::trace_seg_prosody(
                    self,
                    &bert_feat_trace,
                    &prosody_style_trace,
                ),
                &cost_model,
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

            // Trace + analyze: regulate — both inputs on same device.
            let dur_logits_trace = pros.dur_logits.to_device(&trace_dev)?;
            let speed_inv = DynTensor::full(
                &[1, 1],
                1.0 / f64::from(speed),
                nn_core::DType::F32,
                &trace_dev,
            )?;
            let max_dur = self.config().max_dur as f64;
            if let Some(r) = analyze_segment(
                "regulate",
                super::trace_fns::trace_seg_regulate(&dur_logits_trace, &speed_inv, max_dur),
                &cost_model,
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

            // Trace + analyze: f0 — move inputs to model device.
            let aligned_dur_trace = reg.aligned_dur.to_device(&trace_dev)?;
            if let Some(r) = analyze_segment(
                "f0",
                super::trace_fns::trace_seg_f0(
                    self,
                    &aligned_dur_trace,
                    &prosody_style_trace,
                ),
                &cost_model,
            ) {
                results.push(r);
            }

            // Step 6: Harmonic source (needed for generator trace)
            let har_source =
                self.step_harmonic_source(&f0e.f0, &f0e.energy, reg.t_mel, cache)?;

            // Trace + analyze: sinegen_pre + sinegen_post
            // Move inputs to trace_dev for consistent device. (#4250)
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
                if let Some(r) = analyze_segment(
                    "sinegen_pre",
                    super::trace_fns::trace_seg_sinegen_pre(
                        &f0_trace,
                        upp,
                        sg.sampling_rate(),
                        sg.n_channels(),
                    ),
                    &cost_model,
                ) {
                    results.push(r);
                }

                // For sinegen_post we need cum from the eager cumsum step.
                // Since step_harmonic_source already ran, we cannot easily
                // extract the intermediate. Trace sinegen_post with synthetic
                // cum tensor that matches the expected shape.
                let t_frames = f0_trace.dim(1).unwrap_or(1);
                let n_ch = sg.n_channels();
                let batch = f0_trace.dim(0).unwrap_or(1);
                let cum_trace = DynTensor::zeros(
                    &[batch, t_frames, n_ch],
                    nn_core::DType::F32,
                    &trace_dev,
                )?;
                // Move l_linear to trace device so weights match inputs. (#4250)
                let l_w = sm.linear().weight().to_device(&trace_dev)?;
                let l_b = sm
                    .linear()
                    .bias()
                    .map(|b| b.to_device(&trace_dev))
                    .transpose()?;
                let l_linear_trace = Linear::new(l_w, l_b)?;
                let voiced_threshold = f64::from(sg.voiced_threshold());
                if let Some(r) = analyze_segment(
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
                ) {
                    results.push(r);
                }
            }

            // Trace + analyze: generator — move all inputs to model device.
            // FullDecoder::forward expects f0_curve as [B, 1, 2*T_mel] (original
            // shape from step_predict_f0_energy), NOT the transposed [B, 2*T_mel, 1]
            // used by sinegen. Using the transposed f0_trace here causes Conv1d
            // weight in_channels mismatch in the F0 downsampling conv. (#4309)
            let regulated_trace = reg.regulated.to_device(&trace_dev)?;
            let f0_gen_trace = f0e.f0.to_device(&trace_dev)?;
            let energy_trace = f0e.energy.to_device(&trace_dev)?;
            let decoder_style_trace = decoder_style.to_device(&trace_dev)?;
            let har_source_trace = har_source.to_device(&trace_dev)?;
            if let Some(r) = analyze_segment(
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
            ) {
                results.push(r);
            }

            Ok(())
        })?;

        Ok(results)
    }
}
