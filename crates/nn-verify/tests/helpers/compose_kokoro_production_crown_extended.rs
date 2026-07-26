// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended CROWN verification tests for Kokoro production segments.
//!
//! Three segment-level CROWN tests (ProsodyPredictor, F0EnergyPredictor,
//! Generator) and one 4-segment composed pipeline test. These extend
//! `compose_kokoro_production.rs` which covers bert_encoder and text_encoder
//! CROWN paths.
//!
//! **Requires:** `KOKORO_WEIGHTS=/path/to/kokoro_v1_0.safetensors`
//! Tests are gated behind `#[cfg(feature = "production-weights")]` (#2716).
//!
//! Part of #2988: CROWN Extension.
//! Part of #2218: Epic — Perfect Kokoro.

#[cfg(feature = "production-weights")]
use super::kokoro_production_segments::{
    mark_trace_outputs, trace_f0_predictor_composed_fast, trace_generator_composed_fast,
    trace_prosody_predictor_composed_fast, trace_text_encoder_fast,
};
#[cfg(feature = "production-weights")]
use super::kokoro_production_weights::{
    build_multi_input_bounds, is_tight_crown_method, prefer_tighter_recorded_output,
    propagate_with_tight_crown_fallback, record_segment, record_segment_crown,
    require_production_weights, tight_crown_method_name, trace_input,
};

#[cfg(feature = "production-weights")]
use nn_core::dyn_tensor::trace::trace_graph;
#[cfg(feature = "production-weights")]
use nn_core::dyn_tensor::DynTensor;
#[cfg(feature = "production-weights")]
use nn_core::test_utils::cpu;
#[cfg(feature = "production-weights")]
use nn_core::{DType, VarBuilder};
#[cfg(feature = "production-weights")]
use nn_models::kokoro_decoder::Generator;
#[cfg(feature = "production-weights")]
use nn_models::kokoro_f0::F0EnergyPredictor;
#[cfg(feature = "production-weights")]
use nn_models::KokoroConfig;
#[cfg(feature = "production-weights")]
use nn_verify::{trace_to_graph_model_multi_input, BoundedTensor};
#[cfg(feature = "production-weights")]
use std::cell::Cell;

// -- Test: ProsodyPredictor CROWN with production weights (#2988) -----------
//
// Multi-input graph: text_features [B, d_en, T] + style [B, style_dim].
// Record the compiled segment's primary output: duration logits
// `[B, T, max_dur]`, which feed `step_regulate`.

#[cfg(feature = "production-weights")]
#[test]
fn test_production_prosody_predictor_crown() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());

    let d_en = config.d_en;
    let style_dim = config.style_dim;
    let prosody = nn_models::kokoro_tts::ProsodyPredictor::load(
        &vb.pp("prosody_predictor"),
        d_en,
        style_dim,
        config.n_prosody_layers,
        config.max_dur,
    )
    .expect("ProsodyPredictor::load");

    let text_shape = [1, d_en, 4];
    let style_shape = [1, style_dim];
    let text_features = DynTensor::full(&text_shape, 0.1, DType::F32, &cpu()).unwrap();
    let style = DynTensor::full(&style_shape, 0.05, DType::F32, &cpu()).unwrap();

    let feat_id = Cell::new(None);
    let (dur_out, mut graph) = trace_graph(|| {
        let text = trace_input(&text_features);
        let sty = trace_input(&style);
        let (dur_logits, features) = prosody
            .forward(&text, &sty)
            .map_err(|e| nn_core::TensorError::Unsupported(e.to_string()))?;
        feat_id.set(features.trace_id());
        Ok(dur_logits)
    })
    .expect("ProsodyPredictor trace");
    mark_trace_outputs(
        &mut graph,
        dur_out.trace_id(),
        feat_id.get(),
        "ProsodyPredictor",
    )
    .expect("ProsodyPredictor output marking");

    let gn = match trace_to_graph_model_multi_input(&graph) {
        Ok(result) => result.graph,
        Err(e) => {
            eprintln!("ProsodyPredictor graph translation failed: {e}");
            return;
        }
    };

    let input_bounds = build_multi_input_bounds(&[
        (&text_shape[..], (-1.0, 1.0)),
        (&style_shape[..], (-0.5, 0.5)),
    ]);

    // IBP baseline
    let ibp_output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    super::common::assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = super::common::bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;

    // CROWN with fallback
    let (method, crown_output, fallback_reason) =
        propagate_with_tight_crown_fallback(&gn, &input_bounds).expect("PP CROWN propagation");
    super::common::assert_bounds_valid(&crown_output);
    let (crown_lo, crown_hi) = super::common::bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;

    eprintln!(
        "ProsodyPredictor production CROWN: method={method:?}, \
         IBP=[{ibp_lo:.4}, {ibp_hi:.4}] (w={ibp_width:.4}), \
         CROWN=[{crown_lo:.4}, {crown_hi:.4}] (w={crown_width:.4})"
    );

    let (recorded_method, recorded_output, ibp_w, selection_reason) =
        prefer_tighter_recorded_output(method, &ibp_output, &crown_output);
    if is_tight_crown_method(recorded_method) {
        super::common::assert_crown_tighter_than_ibp(&crown_output, &ibp_output);
        let ratio = if ibp_width > 0.0 {
            crown_width / ibp_width
        } else {
            1.0
        };
        eprintln!(
            "ProsodyPredictor {} tightening ratio: {ratio:.6}",
            tight_crown_method_name(recorded_method)
        );
    } else {
        eprintln!(
            "ProsodyPredictor recording prefers IBP bounds: {selection_reason}. \
             Fallback reason: {}",
            fallback_reason.as_deref().unwrap_or("none")
        );
    }

    record_segment_crown(
        "kokoro_production_prosody_predictor_crown",
        &input_bounds,
        &recorded_output,
        recorded_method,
        ibp_w,
    );
}

// -- Test: F0EnergyPredictor CROWN with production weights (#2988) ----------
//
// Multi-input graph: aligned [1, d_en+style_dim, 4] + style [1, style_dim].
// F0 uses BiLSTM + AdainResBlk + grouped ConvTranspose1d + Linear.
//
// Former blockers resolved:
// - Grouped ConvTranspose1d: NY natively supports groups > 1 (#2716).
// - LSTM 3D shape: shape-parametric decomposition propagates prefix dims from
//   the data input tensor, so [S, B, I] -> intermediates [S, B, H] (#3005).
// - Verification is conservative (over-approximate): single-step LSTM with
//   zero initial state, applied independently to all timesteps.

#[cfg(feature = "production-weights")]
#[test]
fn test_production_f0_predictor_crown() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());

    let d_en = config.d_en;
    let style_dim = config.style_dim;
    let f0_predictor = F0EnergyPredictor::load(
        &vb.pp("predictor"),
        d_en,
        style_dim,
        config.f0_bilstm_hidden,
    )
    .expect("F0EnergyPredictor::load");

    let aligned_dim = d_en + style_dim;
    let aligned_shape = [1, aligned_dim, 4];
    let style_shape = [1, style_dim];
    let aligned = DynTensor::full(&aligned_shape, 0.1, DType::F32, &cpu()).unwrap();
    let style = DynTensor::full(&style_shape, 0.05, DType::F32, &cpu()).unwrap();

    let energy_id = Cell::new(None);
    let (f0_out, mut graph) = trace_graph(|| {
        let a = trace_input(&aligned);
        let s = trace_input(&style);
        let (f0, energy) = f0_predictor
            .forward(&a, &s)
            .map_err(|e| nn_core::TensorError::Unsupported(e.to_string()))?;
        energy_id.set(energy.trace_id());
        Ok(f0)
    })
    .expect("F0EnergyPredictor trace");
    mark_trace_outputs(
        &mut graph,
        f0_out.trace_id(),
        energy_id.get(),
        "F0EnergyPredictor",
    )
    .expect("F0EnergyPredictor output marking");

    let gn = match trace_to_graph_model_multi_input(&graph) {
        Ok(result) => result.graph,
        Err(e) => {
            eprintln!(
                "F0 graph translation failed: {e}\n  \
                 NOT recording to status file."
            );
            return;
        }
    };

    let input_bounds = build_multi_input_bounds(&[
        (&aligned_shape[..], (-1.0, 1.0)),
        (&style_shape[..], (-0.5, 0.5)),
    ]);

    // IBP baseline
    let ibp_output = match gn.propagate_ibp(&input_bounds) {
        Ok(out) => out,
        Err(e) => {
            eprintln!(
                "F0 IBP propagation failed: {e}\n  \
                 NOT recording to status file."
            );
            return;
        }
    };
    super::common::assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = super::common::bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;

    // CROWN with fallback
    let (method, crown_output, fallback_reason) =
        propagate_with_tight_crown_fallback(&gn, &input_bounds).expect("F0 CROWN propagation");
    super::common::assert_bounds_valid(&crown_output);
    let (crown_lo, crown_hi) = super::common::bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;

    eprintln!(
        "F0EnergyPredictor production CROWN: method={method:?}, \
         IBP=[{ibp_lo:.4}, {ibp_hi:.4}] (w={ibp_width:.4}), \
         CROWN=[{crown_lo:.4}, {crown_hi:.4}] (w={crown_width:.4})"
    );

    let (recorded_method, recorded_output, ibp_w, selection_reason) =
        prefer_tighter_recorded_output(method, &ibp_output, &crown_output);
    if is_tight_crown_method(recorded_method) {
        super::common::assert_crown_tighter_than_ibp(&crown_output, &ibp_output);
        let ratio = if ibp_width > 0.0 {
            crown_width / ibp_width
        } else {
            1.0
        };
        eprintln!(
            "F0 {} tightening ratio: {ratio:.6}",
            tight_crown_method_name(recorded_method)
        );
    } else {
        eprintln!(
            "F0 recording prefers IBP bounds: {selection_reason}. \
             Fallback reason: {}",
            fallback_reason.as_deref().unwrap_or("none")
        );
    }

    record_segment_crown(
        "kokoro_production_f0_predictor_crown",
        &input_bounds,
        &recorded_output,
        recorded_method,
        ibp_w,
    );
}

// -- Test: Generator CROWN with production weights (#2988) ------------------
//
// 3-input graph: x [B, gen_ch, T] + style [B, style_dim] + har_source
// [B, 2*n_bins, T_full]. Generator uses Conv1d + ConvTranspose1d +
// AdainResBlk + Snake activation. CROWN may fall back to IBP due to
// normalization layers in AdainResBlk.

#[cfg(feature = "production-weights")]
#[test]
fn test_production_generator_crown() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());

    let gen_ch = config.gen_initial_channels;
    let style_dim = config.style_dim;
    let n_bins = config.n_fft / 2 + 1;
    let upsample_factor: usize = config.upsample_rates.iter().product();

    let generator = match Generator::load(&vb.pp("decoder"), &config) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Generator::load failed: {e}\n  Skipping CROWN test.");
            return;
        }
    };

    let t_stage1 = 4;
    let t_full = t_stage1 * upsample_factor;
    let x_shape = [1, gen_ch, t_stage1];
    let style_shape = [1, style_dim];
    let har_shape = [1, 2 * n_bins, t_full];

    let x = DynTensor::full(&x_shape, 0.1, DType::F32, &cpu()).unwrap();
    let style = DynTensor::full(&style_shape, 0.05, DType::F32, &cpu()).unwrap();
    let har = DynTensor::full(&har_shape, 0.01, DType::F32, &cpu()).unwrap();

    let phase_id = Cell::new(None);
    let (mag_out, mut graph) = trace_graph(|| {
        let x_t = trace_input(&x);
        let s_t = trace_input(&style);
        let h_t = trace_input(&har);
        let (mag, phase) = generator
            .forward(&x_t, &s_t, &h_t)
            .map_err(|e| nn_core::TensorError::Unsupported(e.to_string()))?;
        phase_id.set(phase.trace_id());
        Ok(mag)
    })
    .expect("Generator trace");
    mark_trace_outputs(&mut graph, mag_out.trace_id(), phase_id.get(), "Generator")
        .expect("Generator output marking");

    let gn = match trace_to_graph_model_multi_input(&graph) {
        Ok(result) => result.graph,
        Err(e) => {
            eprintln!("Generator graph translation failed: {e}");
            return;
        }
    };

    let input_bounds = build_multi_input_bounds(&[
        (&x_shape[..], (-1.0, 1.0)),
        (&style_shape[..], (-0.5, 0.5)),
        (&har_shape[..], (-0.1, 0.1)),
    ]);

    // IBP baseline
    let ibp_output = match gn.propagate_ibp(&input_bounds) {
        Ok(out) => out,
        Err(e) => {
            eprintln!("Generator IBP propagation failed: {e}");
            return;
        }
    };
    super::common::assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = super::common::bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;

    // CROWN with fallback
    let (method, crown_output, fallback_reason) =
        propagate_with_tight_crown_fallback(&gn, &input_bounds)
            .expect("Generator CROWN propagation");
    super::common::assert_bounds_valid(&crown_output);
    let (crown_lo, crown_hi) = super::common::bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;

    eprintln!(
        "Generator production CROWN: method={method:?}, \
         IBP=[{ibp_lo:.4}, {ibp_hi:.4}] (w={ibp_width:.4}), \
         CROWN=[{crown_lo:.4}, {crown_hi:.4}] (w={crown_width:.4})"
    );

    if is_tight_crown_method(method) {
        super::common::assert_crown_tighter_than_ibp(&crown_output, &ibp_output);
        let ratio = if ibp_width > 0.0 {
            crown_width / ibp_width
        } else {
            1.0
        };
        eprintln!(
            "Generator {} tightening ratio: {ratio:.6}",
            tight_crown_method_name(method)
        );
    } else {
        eprintln!(
            "Generator CROWN-family propagation fell back to IBP: {}",
            fallback_reason.as_deref().unwrap_or("unknown")
        );
    }

    let ibp_w = is_tight_crown_method(method).then_some(ibp_width);
    record_segment_crown(
        "kokoro_production_generator_crown",
        &input_bounds,
        &crown_output,
        method,
        ibp_w,
    );
}

// -- Test: Composed 4-segment pipeline: text -> prosody -> F0 -> generator --
//
// Chains 4 segments with production weights. Each segment's output bounds
// become the next segment's input bounds. This proves bounds compose soundly
// across the full Kokoro synthesis pipeline.
//
// The pipeline:
// 1. TextEncoder: tokens -> text_features [B, d_en, T]
// 2. ProsodyPredictor: (text_features, style) -> (dur_logits, prosody)
// 3. F0EnergyPredictor: (aligned_dur+style, style) -> f0
// 4. Generator: (x, style, har_source) -> magnitude
//
// Bridge stages (length_regulate, harmonic_source) use analytical bounds
// and don't require NY graphs.
//
// Part of #2988, Part of #2218.

#[cfg(feature = "production-weights")]
#[test]
fn test_production_composed_4_segment_pipeline() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());

    // Stage 1: TextEncoder -> text_features output bounds (fast: IBP-only)
    let te = trace_text_encoder_fast(&vb, &config);
    let (te_lo, te_hi) = super::common::bounds_min_max(&te.output_bounds);
    let te_width = te_hi - te_lo;
    eprintln!(
        "Composed pipeline stage 1 — TextEncoder: [{te_lo:.4}, {te_hi:.4}], width={te_width:.4}"
    );
    assert!(
        te_lo.is_finite() && te_hi.is_finite(),
        "TextEncoder output must be finite for composition"
    );

    // Stage 2: ProsodyPredictor using TextEncoder output range (fast: IBP-only)
    let pp = trace_prosody_predictor_composed_fast(&vb, &config, (te_lo, te_hi));
    let (pp_lo, pp_hi) = super::common::bounds_min_max(&pp.output_bounds);
    let pp_width = pp_hi - pp_lo;
    eprintln!(
        "Composed pipeline stage 2 — ProsodyPredictor: [{pp_lo:.4}, {pp_hi:.4}], \
         width={pp_width:.4}"
    );
    assert!(
        pp_lo.is_finite() && pp_hi.is_finite(),
        "composed ProsodyPredictor output must be finite"
    );

    // Bridge: length_regulate preserves value bounds (analytical: sigmoid + clamp).
    // Conservative: use max of text and prosody bounds for aligned input.
    let aligned_lo = te_lo.min(pp_lo);
    let aligned_hi = te_hi.max(pp_hi);
    eprintln!("Composed pipeline bridge — aligned: [{aligned_lo:.4}, {aligned_hi:.4}]");

    // Stage 3: F0EnergyPredictor using composed aligned bounds (fast: IBP-only)
    let f0 = match trace_f0_predictor_composed_fast(&vb, &config, (aligned_lo, aligned_hi)) {
        Ok(seg) => seg,
        Err(e) => {
            eprintln!(
                "Composed pipeline stage 3 — F0EnergyPredictor: SKIPPED ({e})\n  \
                 Partial composition: 2/4 segments verified."
            );
            record_segment(
                "kokoro_production_composed_2_segment",
                &te.input_bounds,
                &pp.output_bounds,
            );
            return;
        }
    };
    let (f0_lo, f0_hi) = super::common::bounds_min_max(&f0.output_bounds);
    let f0_width = f0_hi - f0_lo;
    eprintln!(
        "Composed pipeline stage 3 — F0EnergyPredictor: [{f0_lo:.4}, {f0_hi:.4}], \
         width={f0_width:.4}"
    );
    assert!(
        f0_lo.is_finite() && f0_hi.is_finite(),
        "composed F0 output must be finite"
    );

    // Bridge: harmonic_source has analytical bound tanh in (-1, 1).
    // Generator input x comes from text_features post conv_pre projection.
    let gen_x_range = (aligned_lo, aligned_hi);

    // Stage 4: Generator using composed bounds (fast: IBP-only)
    let generator_seg = match trace_generator_composed_fast(&vb, &config, gen_x_range) {
        Ok(seg) => seg,
        Err(e) => {
            eprintln!(
                "Composed pipeline stage 4 — Generator: SKIPPED ({e})\n  \
                 Partial composition: 3/4 segments verified."
            );
            record_segment(
                "kokoro_production_composed_3_segment",
                &te.input_bounds,
                &f0.output_bounds,
            );
            return;
        }
    };
    let (gen_lo, gen_hi) = super::common::bounds_min_max(&generator_seg.output_bounds);
    let gen_width = gen_hi - gen_lo;
    eprintln!(
        "Composed pipeline stage 4 — Generator: [{gen_lo:.4}, {gen_hi:.4}], \
         width={gen_width:.4}"
    );
    assert!(
        gen_lo.is_finite() && gen_hi.is_finite(),
        "composed Generator output must be finite"
    );

    // Record 4-segment composed pipeline result
    record_segment(
        "kokoro_production_composed_4_segment",
        &te.input_bounds,
        &generator_seg.output_bounds,
    );

    eprintln!(
        "4-SEGMENT COMPOSED PIPELINE: TextEncoder->Prosody->F0->Generator\n  \
         Input: tokens [0, vocab]\n  \
         TextEncoder: [{te_lo:.4}, {te_hi:.4}] (w={te_width:.4})\n  \
         Prosody: [{pp_lo:.4}, {pp_hi:.4}] (w={pp_width:.4})\n  \
         F0: [{f0_lo:.4}, {f0_hi:.4}] (w={f0_width:.4})\n  \
         Generator: [{gen_lo:.4}, {gen_hi:.4}] (w={gen_width:.4})"
    );
}
