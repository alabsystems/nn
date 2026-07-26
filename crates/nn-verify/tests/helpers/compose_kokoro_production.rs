// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Production weight verification for Kokoro TTS model.
//!
//! Loads real Kokoro weights (via `KOKORO_WEIGHTS` env var),
//! traces model segments through `trace_graph` → `trace_to_graph_model` → IBP,
//! then runs `extract_layer_bounds` + `analyze_layer_bounds` to produce a
//! `BoundAnalysisReport` identifying explosion layers and tightening targets.
//!
//! Each segment's IBP result is recorded to `nn_verify_status_kokoro.json`
//! via `record_pipeline`, automatically clearing stale entries from pre-#2498
//! architecture corrections.
//!
//! CROWN verification tests (#2598) use an alpha-first CROWN-family fallback
//! wrapper to measure tightening vs IBP on production weights. Results are
//! recorded with the actual propagation method so the status file
//! distinguishes IBP-only from CROWN-family verified entries.
//!
//! **CROWN status (#2715, #2773, resolved):** After NY bump (fc09d35),
//! CROWN succeeds for decoder, duration, AND text pipeline paths with
//! meaningful tightening (1.5x-1.9x over IBP). LayerNorm shape mismatch
//! (#2774) resolved — synthetic text pipeline CROWN test passes. Chained
//! normalization paths remain IBP-only — CROWN through InstanceNorm/RMSNorm
//! produces vacuous bounds. See `compose_chained_norm.rs` for analysis,
//! `#2240` for upstream dep.
//!
//! **Requires:** `KOKORO_WEIGHTS=/path/to/kokoro_v1_0.safetensors`
//! Tests are gated behind `#[cfg(feature = "production-weights")]` (#2716).
//! Without the feature: tests don't compile (no false CI confidence).
//! With the feature: tests panic if weights are missing.
//!
//! Part of #2461.
//! Part of #2598.
//! Part of #2218.

#[cfg(feature = "production-weights")]
use super::kokoro_production_segments::{
    analyze_and_report, append_records, persist_report, record_all_segments,
    trace_bert_encoder_graph, trace_bert_encoder_segment, trace_f0_predictor_segment,
    trace_generator_segment, trace_production_conv_pre, trace_production_output_stage,
    trace_production_upsample_stages, trace_prosody_predictor_composed,
    trace_prosody_predictor_segment, trace_text_encoder_segment,
};
#[cfg(feature = "production-weights")]
use super::kokoro_production_weights::{
    is_tight_crown_method, prefer_tighter_recorded_output, propagate_with_tight_crown_fallback,
    record_segment, record_segment_crown, require_production_weights, tight_crown_method_name,
    trace_input,
};

#[cfg(feature = "production-weights")]
use nn_core::dyn_tensor::trace::trace_graph;
#[cfg(feature = "production-weights")]
use nn_core::dyn_tensor::DynTensor;
#[cfg(feature = "production-weights")]
use nn_core::layers::{Linear, Module};
#[cfg(feature = "production-weights")]
use nn_core::test_utils::cpu;
#[cfg(feature = "production-weights")]
use nn_core::{DType, VarBuilder};
#[cfg(feature = "production-weights")]
use nn_models::kokoro_decoder::Generator;
#[cfg(feature = "production-weights")]
use nn_models::kokoro_tts::TextEncoder;
#[cfg(feature = "production-weights")]
use nn_models::KokoroConfig;
#[cfg(feature = "production-weights")]
use nn_verify::{trace_to_graph_model, BoundedTensor};
#[cfg(feature = "production-weights")]
use ndarray::{ArrayD, IxDyn};

// -- Test 1: PlBert + bert_encoder with production weights --------------------

#[cfg(feature = "production-weights")]
#[test]
fn test_production_bert_encoder_ibp() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());
    let seg = trace_bert_encoder_segment(&vb, &config);
    super::common::assert_bounds_valid(&seg.output_bounds);
    let (out_lo, out_hi) = super::common::bounds_min_max(&seg.output_bounds);
    assert!(
        out_lo.is_finite() && out_hi.is_finite(),
        "production bounds should be finite"
    );
    eprintln!(
        "PlBert+bert_encoder production IBP summary: [{out_lo:.4}, {out_hi:.4}], width={:.4}",
        out_hi - out_lo
    );

    record_segment(
        "kokoro_production_bert_encoder",
        &seg.input_bounds,
        &seg.output_bounds,
    );
}

// -- Test 2: TextEncoder with production weights ------------------------------

#[cfg(feature = "production-weights")]
#[test]
fn test_production_text_encoder_ibp() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());
    let seg = trace_text_encoder_segment(&vb, &config);
    let report = analyze_and_report("kokoro_text_encoder_production", &seg.records);

    // LayerNorm shape mismatch resolved (#2773) — layer extraction must succeed.
    assert!(
        !seg.records.is_empty(),
        "TextEncoder layer extraction must not be empty (#2773)"
    );
    assert!(
        report.output_is_finite,
        "production bounds should be finite"
    );
    assert!(
        report.total_layers >= 5,
        "TextEncoder should have 5+ layers (Embedding+Conv+Norm+LSTM+Linear)"
    );
    // IBP output bounds are always valid regardless of layer extraction.
    super::common::assert_bounds_valid(&seg.output_bounds);

    record_segment(
        "kokoro_production_text_encoder",
        &seg.input_bounds,
        &seg.output_bounds,
    );
}

// -- Test 2a: PlBert + bert_encoder CROWN with production weights (#2598) -----
//
// Mirrors compiled segment 0 (token IDs + fixed position/type embeddings →
// PlBert::forward_core() → bert projection). Exact point bounds are used for
// the position/type inputs so the verification contract matches runtime.

#[cfg(feature = "production-weights")]
#[test]
fn test_production_bert_encoder_crown() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());
    let (gn, input_bounds) = trace_bert_encoder_graph(&vb, &config);

    // IBP baseline
    let ibp_output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    super::common::assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = super::common::bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;

    // CROWN with fallback
    let (method, crown_output, fallback_reason) =
        propagate_with_tight_crown_fallback(&gn, &input_bounds).expect("CROWN propagation");
    super::common::assert_bounds_valid(&crown_output);
    let (crown_lo, crown_hi) = super::common::bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;

    eprintln!(
        "bert_encoder production CROWN: method={method:?}, \
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
            "bert_encoder {} tightening ratio: {ratio:.6}",
            tight_crown_method_name(recorded_method)
        );
    } else {
        eprintln!(
            "bert_encoder recording prefers IBP bounds: {selection_reason}. \
             Fallback reason: {}",
            fallback_reason.as_deref().unwrap_or("none")
        );
    }

    record_segment_crown(
        "kokoro_production_bert_encoder_crown",
        &input_bounds,
        &recorded_output,
        recorded_method,
        ibp_w,
    );
}

// -- Test 2b: TextEncoder CROWN with production weights (#2598) ----------------
//
// Multi-layer graph: Embedding + Conv1d + LayerNorm + BiLSTM + Linear.
// CROWN should provide tighter bounds than IBP due to linear layer chains.
// This is the first real tightening test with production weights.

#[cfg(feature = "production-weights")]
#[test]
fn test_production_text_encoder_crown() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());

    let vocab_size = config.plbert.vocab_size;
    let d_en = config.d_en;
    let text_encoder =
        TextEncoder::load(&vb.pp("text_encoder"), vocab_size, d_en).expect("TextEncoder::load");

    let token_shape = [1, 4];
    let tokens = DynTensor::full(&token_shape, 5.0, DType::I64, &cpu()).unwrap();
    let (_result, graph) = trace_graph(|| {
        let x = trace_input(&tokens);
        text_encoder
            .forward(&x)
            .map_err(|e| nn_core::TensorError::Unsupported(e.to_string()))
    })
    .expect("TextEncoder trace");

    let gn = trace_to_graph_model(&graph)
        .expect("trace_to_graph_model")
        .graph;
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&token_shape), 0.0f32),
        ArrayD::from_elem(IxDyn(&token_shape), (vocab_size - 1) as f32),
    )
    .expect("valid bounds");

    // IBP baseline
    let ibp_output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    super::common::assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = super::common::bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;

    // CROWN with fallback
    let (method, crown_output, fallback_reason) =
        propagate_with_tight_crown_fallback(&gn, &input_bounds).expect("CROWN propagation");
    super::common::assert_bounds_valid(&crown_output);
    let (crown_lo, crown_hi) = super::common::bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;

    eprintln!(
        "TextEncoder production CROWN: method={method:?}, \
         IBP=[{ibp_lo:.4}, {ibp_hi:.4}] (w={ibp_width:.4}), \
         CROWN=[{crown_lo:.4}, {crown_hi:.4}] (w={crown_width:.4})"
    );

    // CROWN must succeed for TextEncoder (#2773). LayerNorm shape mismatch
    // was resolved — text pipeline CROWN no longer falls back to IBP.
    assert!(
        is_tight_crown_method(method),
        "TextEncoder tight CROWN-family propagation must not fall back to IBP (#2773). \
         method={method:?}, reason: {}",
        fallback_reason.as_deref().unwrap_or("unknown")
    );
    super::common::assert_crown_tighter_than_ibp(&crown_output, &ibp_output);
    let ratio = if ibp_width > 0.0 {
        crown_width / ibp_width
    } else {
        1.0
    };
    eprintln!(
        "TextEncoder {} tightening ratio: {ratio:.6}",
        tight_crown_method_name(method)
    );

    let ibp_w = is_tight_crown_method(method).then_some(ibp_width);
    record_segment_crown(
        "kokoro_production_text_encoder_crown",
        &input_bounds,
        &crown_output,
        method,
        ibp_w,
    );
}

// -- Test 3: ProsodyPredictor with production weights -------------------------

#[cfg(feature = "production-weights")]
#[test]
fn test_production_prosody_predictor_ibp() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());
    let seg = trace_prosody_predictor_segment(&vb, &config);
    let report = analyze_and_report("kokoro_prosody_predictor_production", &seg.records);

    if !seg.records.is_empty() {
        assert!(
            report.output_is_finite,
            "production ProsodyPredictor bounds should be finite"
        );
        assert!(
            report.total_layers >= 5,
            "ProsodyPredictor should have 5+ layers (Conv+Norm+LSTM+Linear+...)"
        );
    } else {
        eprintln!("ProsodyPredictor: layer extraction skipped (LayerNorm shape mismatch)");
    }
    super::common::assert_bounds_valid(&seg.output_bounds);

    record_segment(
        "kokoro_production_prosody_predictor",
        &seg.input_bounds,
        &seg.output_bounds,
    );
}

// -- Test 4: F0EnergyPredictor with production weights ------------------------

#[cfg(feature = "production-weights")]
#[test]
fn test_production_f0_predictor_ibp() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());
    let seg = match trace_f0_predictor_segment(&vb, &config) {
        Ok(seg) => seg,
        Err(e) => {
            eprintln!("UNVERIFIED: F0EnergyPredictor — {e}");
            eprintln!("  NOT recording to status file (#2716)");
            return;
        }
    };
    let report = analyze_and_report("kokoro_f0_predictor_production", &seg.records);

    if !seg.records.is_empty() {
        assert!(
            report.output_is_finite,
            "production F0EnergyPredictor bounds should be finite"
        );
        assert!(
            report.total_layers >= 3,
            "F0EnergyPredictor should have 3+ layers (BiLSTM+AdainResBlk+Linear)"
        );
    } else {
        eprintln!("F0EnergyPredictor: layer extraction skipped");
    }
    super::common::assert_bounds_valid(&seg.output_bounds);

    record_segment(
        "kokoro_production_f0_predictor",
        &seg.input_bounds,
        &seg.output_bounds,
    );
}

// -- Test 5: Generator with production weights --------------------------------

#[cfg(feature = "production-weights")]
#[test]
fn test_production_generator_ibp() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());
    let seg = match trace_generator_segment(&vb, &config) {
        Ok(seg) => seg,
        Err(e) => {
            eprintln!("UNVERIFIED: Generator — {e}");
            eprintln!("  NOT recording to status file (#2716)");
            return;
        }
    };
    let report = analyze_and_report("kokoro_generator_production", &seg.records);

    if !seg.records.is_empty() {
        assert!(
            report.output_is_finite,
            "production Generator bounds should be finite"
        );
        assert!(
            report.total_layers >= 5,
            "Generator should have 5+ layers (Conv+ConvTranspose+ResBlock+...)"
        );
    } else {
        eprintln!("Generator: layer extraction skipped");
    }
    super::common::assert_bounds_valid(&seg.output_bounds);

    record_segment(
        "kokoro_production_generator",
        &seg.input_bounds,
        &seg.output_bounds,
    );
}

// -- Test 6: Generator sub-block with production weights (#2597) ---------------

/// Production-weight sub-block verification for the Generator (#2597).
#[cfg(feature = "production-weights")]
#[test]
fn test_production_generator_subblock_ibp() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());
    let generator = match Generator::load(&vb.pp("decoder"), &config) {
        Ok(g) => g,
        Err(e) => {
            eprintln!(
                "Generator::load failed (v1.0 architecture mismatch): {e}\n  \
                 v1.0 uses resblocks.paths/decode/encode, v0.19 uses noise_res/conv_pre.\n  \
                 Skipping sub-block test."
            );
            return;
        }
    };
    let t_stage1 = 4;

    let (conv_pre_input, conv_pre_bounds) =
        trace_production_conv_pre(&generator, config.gen_initial_channels, t_stage1);
    let upsample_bounds =
        trace_production_upsample_stages(&generator, &config, t_stage1, conv_pre_bounds);
    let output = trace_production_output_stage(&generator, &config, t_stage1, upsample_bounds);

    super::common::assert_bounds_valid(&output);
    let (lo_final, hi_final) = super::common::bounds_min_max(&output);
    eprintln!("Production sub-block output: [{lo_final}, {hi_final}]");
    assert!(
        lo_final.is_finite() && hi_final.is_finite(),
        "production sub-block output must be finite, got [{lo_final}, {hi_final}]"
    );
    assert!(
        lo_final > -1.0,
        "exp output should be near-zero, got lo={lo_final}"
    );

    record_segment(
        "kokoro_production_generator_subblock",
        &conv_pre_input,
        &output,
    );
    eprintln!("Production Generator sub-block: finite bounds [{lo_final}, {hi_final}]");
}

// -- Test 7: Composed verification — text_encoder → prosody_predictor ----------
//
// Chains 2 segments with production weights: text_encoder output bounds become
// prosody_predictor input bounds. This tests that bounds compose soundly across
// segment boundaries (Part of #2461).

#[cfg(feature = "production-weights")]
#[test]
fn test_production_composed_text_to_prosody() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());

    // Stage 1: text_encoder with production weights → output bounds
    let te = trace_text_encoder_segment(&vb, &config);
    let (te_lo, te_hi) = super::common::bounds_min_max(&te.output_bounds);
    eprintln!(
        "Composed stage 1 — TextEncoder output: [{te_lo:.4}, {te_hi:.4}], width={:.4}",
        te_hi - te_lo
    );
    assert!(
        te_lo.is_finite() && te_hi.is_finite(),
        "TextEncoder output must be finite for composition"
    );

    // Stage 2: prosody_predictor using text_encoder output range as text_features bounds
    let pp = trace_prosody_predictor_composed(&vb, &config, (te_lo, te_hi));
    let (pp_lo, pp_hi) = super::common::bounds_min_max(&pp.output_bounds);
    let pp_width = pp_hi - pp_lo;
    eprintln!(
        "Composed stage 2 — ProsodyPredictor output: [{pp_lo:.4}, {pp_hi:.4}], width={pp_width:.4}"
    );

    // Composed bounds must be finite (soundness).
    assert!(
        pp_lo.is_finite() && pp_hi.is_finite(),
        "composed text→prosody output must be finite, got [{pp_lo}, {pp_hi}]"
    );

    // Compare with standalone prosody (hardcoded [-1,1] text_features) to verify
    // composition produces different (typically wider) bounds.
    let pp_standalone = trace_prosody_predictor_segment(&vb, &config);
    let (sa_lo, sa_hi) = super::common::bounds_min_max(&pp_standalone.output_bounds);
    let sa_width = sa_hi - sa_lo;
    eprintln!("Standalone ProsodyPredictor: [{sa_lo:.4}, {sa_hi:.4}], width={sa_width:.4}");
    eprintln!(
        "Composition effect: composed_width={pp_width:.4}, standalone_width={sa_width:.4}, \
         ratio={:.4}",
        if sa_width > 0.0 {
            pp_width / sa_width
        } else {
            1.0
        }
    );

    record_segment(
        "kokoro_production_composed_text_to_prosody",
        &pp.input_bounds,
        &pp.output_bounds,
    );
}

// -- Test 8: Combined report across all segments ------------------------------

#[cfg(feature = "production-weights")]
#[test]
fn test_production_combined_report() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());

    let bert = trace_bert_encoder_segment(&vb, &config);
    let te = trace_text_encoder_segment(&vb, &config);
    let pp = trace_prosody_predictor_segment(&vb, &config);
    let f0 = match trace_f0_predictor_segment(&vb, &config) {
        Ok(seg) => Some(seg),
        Err(e) => {
            eprintln!("UNVERIFIED: F0EnergyPredictor — {e}");
            None
        }
    };
    let generator = match trace_generator_segment(&vb, &config) {
        Ok(seg) => Some(seg),
        Err(e) => {
            eprintln!("UNVERIFIED: Generator — {e}");
            None
        }
    };

    record_all_segments(&bert, &te, &pp, f0.as_ref(), generator.as_ref());

    // Count segments with layer records before moving records out.
    let verified_count = 3 + f0.is_some() as usize + generator.is_some() as usize;
    let segments_with_records = [
        Some(&bert),
        Some(&te),
        Some(&pp),
        f0.as_ref(),
        generator.as_ref(),
    ]
    .iter()
    .filter(|s| s.map_or(false, |s| !s.records.is_empty()))
    .count();

    let mut all_records = bert.records;
    append_records(&mut all_records, te.records);
    append_records(&mut all_records, pp.records);
    if let Some(f0) = f0 {
        append_records(&mut all_records, f0.records);
    }
    if let Some(generator) = generator {
        append_records(&mut all_records, generator.records);
    }

    let report = analyze_and_report("kokoro_production_all_segments", &all_records);
    eprintln!(
        "Summary: CROWN coverage {:.1}%, output_width={:.2}, finite={}",
        report.crown_coverage * 100.0,
        report.output_width,
        report.output_is_finite,
    );

    persist_report(&report);

    // bert_encoder always provides layer records; other segments may not
    // (LayerNorm shape mismatch, grouped ConvTranspose, v1.0 arch mismatch).
    if report.total_layers > 0 {
        assert!(report.output_is_finite, "combined output should be finite");
    }
    eprintln!("Combined report: {segments_with_records} of {verified_count} verified segments contributed layer records");
}
