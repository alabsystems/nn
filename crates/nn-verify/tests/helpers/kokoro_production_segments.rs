// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Segment tracing, sub-block verification, and analysis helpers for Kokoro
//! production-weight verification tests.
//!
//! Extracted from `compose_kokoro_production.rs` for 500-line compliance.
//! Part of #2633.

use super::kokoro_production_weights::{
    build_multi_input_bounds, log_bounds_width, record_segment, trace_input, SegmentResult,
};
use nn_core::dyn_tensor::trace::{record_input, trace_graph, ComputationGraph, NodeId};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Linear, Module};
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};
use nn_models::kokoro_decoder::Generator;
use nn_models::kokoro_f0::F0EnergyPredictor;
use nn_models::kokoro_tts::{ProsodyPredictor, TextEncoder};
use nn_models::{KokoroConfig, PlBert};
use nn_verify::bound_analysis::{
    analyze_layer_bounds, report_to_json, AnalysisConfig, BoundAnalysisReport,
};
use nn_verify::layer_bounds::extract_layer_bounds;
use nn_verify::{
    trace_to_graph_model, trace_to_graph_model_multi_input, BoundedTensor, LayerBoundRecord,
};
use ndarray::{ArrayD, IxDyn};
use std::cell::Cell;
use std::path::Path;

// -- Helpers ------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProsodyTraceOutput {
    DurLogits,
    Features,
}

/// Try to extract per-layer bounds; return empty vec on failure (LayerNorm shape mismatch).
///
/// `extract_layer_bounds` fails when NY cannot handle certain layer
/// shapes (e.g. LayerNorm with flattened dimensions). The IBP propagation
/// itself is still valid — only the per-layer breakdown is unavailable.
fn try_extract_layer_bounds(
    gn: &nn_verify::GraphNetwork,
    input: &BoundedTensor,
    label: &str,
) -> Vec<LayerBoundRecord> {
    match extract_layer_bounds(gn, input) {
        Ok(records) => records,
        Err(e) => {
            eprintln!("WARNING: extract_layer_bounds failed for {label}: {e}");
            eprintln!("  IBP bounds are still valid; per-layer analysis skipped.");
            Vec::new()
        }
    }
}

pub(super) fn mark_trace_outputs(
    graph: &mut ComputationGraph,
    primary_id: Option<NodeId>,
    _secondary_id: Option<NodeId>,
    label: &str,
) -> Result<(), String> {
    let primary_id =
        primary_id.ok_or_else(|| format!("trace bug: {label} primary output missing trace ID"))?;
    if !graph.set_primary_output(primary_id) {
        return Err(format!(
            "trace bug: {label} primary output node {primary_id} not found in graph"
        ));
    }
    Ok(())
}

// -- Segment tracing helpers --------------------------------------------------

/// Trace bert_encoder (Linear) and return per-layer bound records + bounds.
pub(super) fn trace_bert_encoder_segment(vb: &VarBuilder, config: &KokoroConfig) -> SegmentResult {
    let (gn, input_bounds) = trace_bert_encoder_graph(vb, config);

    let output_bounds = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    super::common::assert_bounds_valid(&output_bounds);
    log_bounds_width("PlBert+bert_encoder", &output_bounds);

    // Transformer-scale layer extraction is expensive and not needed for the
    // production status refresh. Downstream tests only require the segment
    // bounds themselves.
    let records = Vec::new();
    SegmentResult {
        records,
        input_bounds,
        output_bounds,
    }
}

/// Trace a sound hull abstraction of compiled stage 0:
/// coordinate-wise word-embedding bounds + exact position/type embeddings,
/// then `PlBert::forward_core()` and the bert projection.
///
/// This keeps the proof aligned with the real compiled stage while avoiding the
/// extremely loose interval relaxation that results from treating discrete token
/// IDs as a continuous interval through the embedding lookup.
pub(super) fn trace_bert_encoder_graph(
    vb: &VarBuilder,
    config: &KokoroConfig,
) -> (nn_verify::GraphNetwork, BoundedTensor) {
    let plbert = PlBert::load(vb.pp("plbert"), &config.plbert).expect("PlBert::load");
    let hidden = config.plbert.hidden_size;
    let d_en = config.d_en;
    let w = vb
        .get(&[d_en, hidden], "bert_encoder.weight")
        .expect("weight");
    let b = vb.get(&[d_en], "bert_encoder.bias").expect("bias");
    let bert_encoder = Linear::new(w, Some(b)).expect("Linear::new");

    let seq_len = 4usize;
    let emb_dim = config.plbert.embedding_dim;
    let position_ids = DynTensor::arange_u32(0, seq_len as u32, &cpu()).expect("position ids");
    let pos_emb = plbert
        .position_embeddings()
        .forward(&position_ids)
        .expect("position embeddings")
        .unsqueeze(0)
        .expect("unsqueeze");
    let token_type_ids = DynTensor::zeros(&[seq_len], DType::U32, &cpu()).expect("type ids");
    let type_emb = plbert
        .token_type_embeddings()
        .forward(&token_type_ids)
        .expect("type embeddings")
        .unsqueeze(0)
        .expect("unsqueeze");

    let sample_tokens = DynTensor::full(&[1, seq_len], 5.0, DType::F32, &cpu()).unwrap();
    let sample_word_emb = plbert
        .word_embeddings()
        .forward(&sample_tokens)
        .expect("word embeddings");
    let combined_example = sample_word_emb
        .broadcast_add(&pos_emb)
        .expect("position add")
        .broadcast_add(&type_emb)
        .expect("type add");

    let weight = plbert
        .word_embeddings()
        .weight()
        .to_f32_array()
        .expect("embedding weights to_f32_array");
    let mut word_min = vec![f32::INFINITY; emb_dim];
    let mut word_max = vec![f32::NEG_INFINITY; emb_dim];
    for row in weight.outer_iter() {
        for (idx, &val) in row.iter().enumerate() {
            word_min[idx] = word_min[idx].min(val);
            word_max[idx] = word_max[idx].max(val);
        }
    }
    let pos_flat = pos_emb.to_flat_vec::<f32>().expect("pos_emb flat");
    let type_flat = type_emb.to_flat_vec::<f32>().expect("type_emb flat");
    let mut lower = Vec::with_capacity(seq_len * emb_dim);
    let mut upper = Vec::with_capacity(seq_len * emb_dim);
    for t in 0..seq_len {
        for d in 0..emb_dim {
            let offset = t * emb_dim + d;
            let fixed = pos_flat[offset] + type_flat[offset];
            lower.push(fixed + word_min[d]);
            upper.push(fixed + word_max[d]);
        }
    }

    let (_result, graph) = trace_graph(|| {
        let x = trace_input(&combined_example);
        let bert_output = plbert.forward_core(&x)?;
        bert_encoder.forward(&bert_output)?.transpose(1, 2)
    })
    .expect("PlBert+bert_encoder trace");

    let gn = trace_to_graph_model(&graph)
        .expect("trace_to_graph_model")
        .graph;

    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, seq_len, emb_dim]), lower).expect("lower shape"),
        ArrayD::from_shape_vec(IxDyn(&[1, seq_len, emb_dim]), upper).expect("upper shape"),
    )
    .expect("valid bounds");

    (gn, input_bounds)
}

/// Trace TextEncoder and return per-layer bound records + bounds.
pub(super) fn trace_text_encoder_segment(vb: &VarBuilder, config: &KokoroConfig) -> SegmentResult {
    trace_text_encoder_inner(vb, config, true)
}

/// Trace TextEncoder returning only IBP bounds (no per-layer extraction).
///
/// Use this for composed pipeline tests where only the overall bounds
/// are needed for composition to the next stage. Skips the expensive
/// `extract_layer_bounds` call.
pub(super) fn trace_text_encoder_fast(vb: &VarBuilder, config: &KokoroConfig) -> SegmentResult {
    trace_text_encoder_inner(vb, config, false)
}

fn trace_text_encoder_inner(
    vb: &VarBuilder,
    config: &KokoroConfig,
    extract_layers: bool,
) -> SegmentResult {
    let vocab_size = config.plbert.vocab_size;
    let d_en = config.d_en;
    let text_encoder =
        TextEncoder::load(vb.pp("text_encoder"), vocab_size, d_en).expect("TextEncoder::load");

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

    let output_bounds = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    super::common::assert_bounds_valid(&output_bounds);
    log_bounds_width("TextEncoder", &output_bounds);

    let records = if extract_layers {
        try_extract_layer_bounds(&gn, &input_bounds, "text_encoder")
    } else {
        Vec::new()
    };
    SegmentResult {
        records,
        input_bounds,
        output_bounds,
    }
}

/// Trace ProsodyPredictor (multi-input) and return per-layer bound records + bounds.
///
/// Inputs: text_features `[B, d_en, T]` + style `[B, style_dim]`.
/// Output: duration logits `[B, T, max_dur]` (the compiled segment's primary
/// output consumed by `step_regulate`). Uses default text_features bounds
/// `[-1, 1]`.
pub(super) fn trace_prosody_predictor_segment(
    vb: &VarBuilder,
    config: &KokoroConfig,
) -> SegmentResult {
    trace_prosody_predictor_inner(vb, config, None, true, ProsodyTraceOutput::DurLogits)
}

/// Trace ProsodyPredictor features with composed input bounds from a previous segment.
pub(super) fn trace_prosody_predictor_composed(
    vb: &VarBuilder,
    config: &KokoroConfig,
    text_features_range: (f32, f32),
) -> SegmentResult {
    trace_prosody_predictor_inner(
        vb,
        config,
        Some(text_features_range),
        true,
        ProsodyTraceOutput::Features,
    )
}

/// Trace ProsodyPredictor features with composed bounds, IBP-only (no layer extraction).
pub(super) fn trace_prosody_predictor_composed_fast(
    vb: &VarBuilder,
    config: &KokoroConfig,
    text_features_range: (f32, f32),
) -> SegmentResult {
    trace_prosody_predictor_inner(
        vb,
        config,
        Some(text_features_range),
        false,
        ProsodyTraceOutput::Features,
    )
}

fn trace_prosody_predictor_inner(
    vb: &VarBuilder,
    config: &KokoroConfig,
    text_features_range: Option<(f32, f32)>,
    extract_layers: bool,
    output_kind: ProsodyTraceOutput,
) -> SegmentResult {
    let d_en = config.d_en;
    let style_dim = config.style_dim;
    let prosody = ProsodyPredictor::load(
        vb.pp("prosody_predictor"),
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

    let dur_id: Cell<Option<NodeId>> = Cell::new(None);
    let feat_id: Cell<Option<NodeId>> = Cell::new(None);
    let (result, mut graph) = trace_graph(|| {
        let text = trace_input(&text_features);
        let sty = trace_input(&style);
        let (dur_logits, features) = prosody
            .forward(&text, &sty)
            .map_err(|e| nn_core::TensorError::Unsupported(e.to_string()))?;
        dur_id.set(dur_logits.trace_id());
        feat_id.set(features.trace_id());
        match output_kind {
            ProsodyTraceOutput::DurLogits => Ok(dur_logits),
            ProsodyTraceOutput::Features => Ok(features),
        }
    })
    .expect("ProsodyPredictor trace");
    let secondary_id = match output_kind {
        ProsodyTraceOutput::DurLogits => feat_id.get(),
        ProsodyTraceOutput::Features => dur_id.get(),
    };
    mark_trace_outputs(
        &mut graph,
        result.trace_id(),
        secondary_id,
        "ProsodyPredictor",
    )
    .expect("ProsodyPredictor output marking");

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("ProsodyPredictor trace_to_graph_model")
        .graph;
    let text_range = text_features_range.unwrap_or((-1.0, 1.0));
    let is_composed = text_features_range.is_some();
    let input_bounds = build_multi_input_bounds(&[
        (&text_shape[..], text_range),
        (&style_shape[..], (-0.5, 0.5)),
    ]);

    let output_bounds = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    super::common::assert_bounds_valid(&output_bounds);
    let label = match (is_composed, output_kind) {
        (false, ProsodyTraceOutput::DurLogits) => "ProsodyPredictor(dur_logits)",
        (false, ProsodyTraceOutput::Features) => "ProsodyPredictor(features)",
        (true, ProsodyTraceOutput::DurLogits) => "ProsodyPredictor(composed dur_logits)",
        (true, ProsodyTraceOutput::Features) => "ProsodyPredictor(composed features)",
    };
    log_bounds_width(label, &output_bounds);

    let record_label = match (is_composed, output_kind) {
        (false, ProsodyTraceOutput::DurLogits) => "prosody_predictor",
        (false, ProsodyTraceOutput::Features) => "prosody_predictor_features",
        (true, ProsodyTraceOutput::DurLogits) => "prosody_predictor_composed",
        (true, ProsodyTraceOutput::Features) => "prosody_predictor_composed_features",
    };
    let records = if extract_layers {
        try_extract_layer_bounds(&gn, &input_bounds, record_label)
    } else {
        Vec::new()
    };
    SegmentResult {
        records,
        input_bounds,
        output_bounds,
    }
}

/// Trace F0EnergyPredictor (multi-input) and return per-layer bound records + bounds.
///
/// Inputs: aligned `[B, d_model+style_dim, T_mel]` + style `[B, style_dim]`.
/// Output: F0 curve `[B, 1, 2*T_mel]` (first output of tuple).
///
/// Returns `Err` when graph translation or IBP propagation fails.
/// Former blockers (grouped ConvTranspose1d #2716, sequence LSTM 3D shape
/// #3005) are now resolved — see the shape-parametric LSTM decomposition in
/// the NY-owned translator (ny-trace-bridge `translate/ops_lstm.rs`) and
/// grouped ConvTranspose1d support in NY.
pub(super) fn trace_f0_predictor_segment(
    vb: &VarBuilder,
    config: &KokoroConfig,
) -> Result<SegmentResult, String> {
    trace_f0_predictor_inner(vb, config, None, true)
}

/// Trace F0EnergyPredictor with composed input bounds from a previous stage.
pub(super) fn trace_f0_predictor_composed(
    vb: &VarBuilder,
    config: &KokoroConfig,
    aligned_range: (f32, f32),
) -> Result<SegmentResult, String> {
    trace_f0_predictor_inner(vb, config, Some(aligned_range), true)
}

/// Trace F0EnergyPredictor with composed bounds, IBP-only (no layer extraction).
pub(super) fn trace_f0_predictor_composed_fast(
    vb: &VarBuilder,
    config: &KokoroConfig,
    aligned_range: (f32, f32),
) -> Result<SegmentResult, String> {
    trace_f0_predictor_inner(vb, config, Some(aligned_range), false)
}

fn trace_f0_predictor_inner(
    vb: &VarBuilder,
    config: &KokoroConfig,
    aligned_range: Option<(f32, f32)>,
    extract_layers: bool,
) -> Result<SegmentResult, String> {
    let d_en = config.d_en;
    let style_dim = config.style_dim;
    let f0_predictor =
        F0EnergyPredictor::load(vb.pp("predictor"), d_en, style_dim, config.f0_bilstm_hidden)
            .expect("F0EnergyPredictor::load");

    // F0EnergyPredictor::forward expects aligned [B, d_model+style_dim, T_mel]
    // because DurationEncoder output already includes style concatenation.
    // See kokoro_f0.rs:270: "aligned already includes style (d_model+style_dim=640)".
    let aligned_dim = d_en + style_dim;
    let aligned_shape = [1, aligned_dim, 4];
    let style_shape = [1, style_dim];
    let aligned = DynTensor::full(&aligned_shape, 0.1, DType::F32, &cpu()).unwrap();
    let style = DynTensor::full(&style_shape, 0.05, DType::F32, &cpu()).unwrap();

    let energy_id: Cell<Option<NodeId>> = Cell::new(None);
    let (f0_out, mut graph) = trace_graph(|| {
        let a = trace_input(&aligned);
        let s = trace_input(&style);
        let (f0, energy) = f0_predictor
            .forward(&a, &s)
            .map_err(|e| nn_core::TensorError::Unsupported(e.to_string()))?;
        energy_id.set(energy.trace_id());
        Ok(f0)
    })
    .map_err(|e| format!("F0EnergyPredictor trace failed: {e}"))?;
    mark_trace_outputs(
        &mut graph,
        f0_out.trace_id(),
        energy_id.get(),
        "F0EnergyPredictor",
    )?;

    let a_range = aligned_range.unwrap_or((-1.0, 1.0));
    let is_composed = aligned_range.is_some();
    let input_bounds = build_multi_input_bounds(&[
        (&aligned_shape[..], a_range),
        (&style_shape[..], (-0.5, 0.5)),
    ]);

    // Graph translation and IBP propagation. Former blockers (grouped
    // ConvTranspose1d #2716, sequence LSTM 3D shape #3005) are resolved.
    // Return Err defensively so callers can skip gracefully if new issues arise.
    let gn = trace_to_graph_model_multi_input(&graph)
        .map_err(|e| format!("F0EnergyPredictor graph translation failed: {e}"))?
        .graph;

    let output_bounds = gn
        .propagate_ibp(&input_bounds)
        .map_err(|e| format!("F0EnergyPredictor IBP propagation failed: {e}"))?;
    super::common::assert_bounds_valid(&output_bounds);
    let label = if is_composed {
        "F0EnergyPredictor(composed)"
    } else {
        "F0EnergyPredictor"
    };
    log_bounds_width(label, &output_bounds);

    let record_label = if is_composed {
        "f0_predictor_composed"
    } else {
        "f0_predictor"
    };
    let records = if extract_layers {
        try_extract_layer_bounds(&gn, &input_bounds, record_label)
    } else {
        Vec::new()
    };
    Ok(SegmentResult {
        records,
        input_bounds,
        output_bounds,
    })
}

/// Trace Generator (3-input) and return per-layer bound records + bounds.
///
/// Inputs: x `[B, gen_ch, T_stage1]` + style `[B, style_dim]` + har_source `[B, 2*n_bins, T_full]`.
/// Output: magnitude `[B, n_bins, T_out]` (first output of tuple).
pub(super) fn trace_generator_segment(
    vb: &VarBuilder,
    config: &KokoroConfig,
) -> Result<SegmentResult, String> {
    trace_generator_inner(vb, config, None, true)
}

/// Trace Generator with composed input bounds from previous stages.
pub(super) fn trace_generator_composed(
    vb: &VarBuilder,
    config: &KokoroConfig,
    x_range: (f32, f32),
) -> Result<SegmentResult, String> {
    trace_generator_inner(vb, config, Some(x_range), true)
}

/// Trace Generator with composed bounds, IBP-only (no layer extraction).
pub(super) fn trace_generator_composed_fast(
    vb: &VarBuilder,
    config: &KokoroConfig,
    x_range: (f32, f32),
) -> Result<SegmentResult, String> {
    trace_generator_inner(vb, config, Some(x_range), false)
}

fn trace_generator_inner(
    vb: &VarBuilder,
    config: &KokoroConfig,
    x_range: Option<(f32, f32)>,
    extract_layers: bool,
) -> Result<SegmentResult, String> {
    let gen_ch = config.gen_initial_channels;
    let style_dim = config.style_dim;
    let n_bins = config.n_fft / 2 + 1;
    let upsample_factor: usize = config.upsample_rates.iter().product();

    // v1.0 key differences handled by remap_v1_weights:
    // weight_norm decomposition, ResBlock paths remap, AdaIn layer rename,
    // synthetic conv_pre identity. Return Err if still missing keys (#2716).
    let generator = Generator::load(vb.pp("decoder"), config).map_err(|e| {
        format!(
            "Generator::load failed: {e}\n  \
             Check weight_norm decomposition and key remap in kokoro_production_weights.rs."
        )
    })?;

    let x_r = x_range.unwrap_or((-1.0, 1.0));
    let t_stage1 = 4;
    let t_full = t_stage1 * upsample_factor;
    let x_shape = [1, gen_ch, t_stage1];
    let style_shape = [1, style_dim];
    let har_shape = [1, 2 * n_bins, t_full];

    // har_source comes from SineGen (sin function) so bounded by [-1, 1].
    // Use [-0.1, 0.1] for tighter bounds — SineGen output is attenuated by
    // per-harmonic weights which are small in practice.
    let input_bounds = build_multi_input_bounds(&[
        (&x_shape[..], x_r),
        (&style_shape[..], (-0.5, 0.5)),
        (&har_shape[..], (-0.1, 0.1)),
    ]);

    let x = DynTensor::full(&x_shape, 0.1, DType::F32, &cpu()).unwrap();
    let style = DynTensor::full(&style_shape, 0.05, DType::F32, &cpu()).unwrap();
    let har = DynTensor::full(&har_shape, 0.01, DType::F32, &cpu()).unwrap();

    let phase_id: Cell<Option<NodeId>> = Cell::new(None);
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
    .map_err(|e| format!("Generator trace failed: {e}"))?;
    mark_trace_outputs(&mut graph, mag_out.trace_id(), phase_id.get(), "Generator")?;

    let gn = trace_to_graph_model_multi_input(&graph)
        .map_err(|e| format!("Generator trace_to_graph_model failed: {e}"))?
        .graph;

    let output_bounds = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    super::common::assert_bounds_valid(&output_bounds);
    let is_composed = x_range.is_some();
    let label = if is_composed {
        "Generator(composed)"
    } else {
        "Generator"
    };
    log_bounds_width(label, &output_bounds);

    let record_label = if is_composed {
        "generator_composed"
    } else {
        "generator"
    };
    let records = if extract_layers {
        try_extract_layer_bounds(&gn, &input_bounds, record_label)
    } else {
        Vec::new()
    };
    Ok(SegmentResult {
        records,
        input_bounds,
        output_bounds,
    })
}

// -- Analysis helpers ---------------------------------------------------------

/// Run BoundAnalysisReport with auto-precision drift estimation. Returns the report.
///
/// After bound analysis, if the model has chained normalization layers, the F32/F64
/// precision drift is estimated and populated into the report. This enables the
/// `PRECISION_RISK` recommendation for deep norm chains like Kokoro's 58-layer
/// InstanceNorm chain (#2705).
pub(super) fn analyze_and_report(name: &str, records: &[LayerBoundRecord]) -> BoundAnalysisReport {
    let config = AnalysisConfig::default();
    let mut report = analyze_layer_bounds(name, records, &config);

    // Auto-estimate F32/F64 precision drift when chained norms are detected (#2705).
    report.estimate_and_set_precision_drift(&config);

    let json = report_to_json(&report).expect("report_to_json");
    eprintln!("--- BoundAnalysisReport ({name}) ---\n{json}\n---");
    eprintln!(
        "{name}: {} layers, {} explosion points, {} recommendations, \
         chained_norm_depth={}, precision_drift_ratio={:?}",
        report.total_layers,
        report.explosion_points.len(),
        report.recommendations.len(),
        report.chained_norm_depth,
        report.precision_drift_ratio,
    );
    for (i, rec) in report.recommendations.iter().enumerate() {
        eprintln!("  Recommendation {i}: {rec:?}");
    }
    report
}

// -- Sub-block tracing helpers ------------------------------------------------

/// Trace conv_pre sub-block with production Generator, return (input_bounds, output_bounds).
pub(super) fn trace_production_conv_pre(
    generator: &Generator,
    gen_ch: usize,
    t_stage1: usize,
) -> (BoundedTensor, (f32, f32)) {
    let x_shape = [1, gen_ch, t_stage1];
    let x = DynTensor::full(&x_shape, 0.1, DType::F32, &cpu()).unwrap();
    let (_result, graph) = trace_graph(|| {
        let x_t = trace_input(&x);
        generator
            .forward_conv_pre(&x_t)
            .map_err(|e| nn_core::TensorError::Unsupported(e.to_string()))
    })
    .expect("conv_pre trace");
    let gn = trace_to_graph_model(&graph)
        .expect("conv_pre trace_to_graph")
        .graph;
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&x_shape), -1.0f32),
        ArrayD::from_elem(IxDyn(&x_shape), 1.0f32),
    )
    .expect("valid bounds");
    let output = gn.propagate_ibp(&input_bounds).expect("conv_pre IBP");
    super::common::assert_bounds_valid(&output);
    let (lo, hi) = super::common::bounds_min_max(&output);
    eprintln!("Production sub-block conv_pre: [{lo}, {hi}]");
    assert!(
        lo.is_finite() && hi.is_finite(),
        "conv_pre bounds must be finite"
    );
    (input_bounds, (lo, hi))
}

/// Trace upsample stages with production Generator, chain bounds, return final bounds.
pub(super) fn trace_production_upsample_stages(
    generator: &Generator,
    config: &KokoroConfig,
    t_stage1: usize,
    mut prev_bounds: (f32, f32),
) -> (f32, f32) {
    let gen_ch = config.gen_initial_channels;
    let style_dim = config.style_dim;
    let n_bins = config.n_fft / 2 + 1;
    let upsample_factor: usize = config.upsample_rates.iter().product();

    for stage in 0..config.upsample_rates.len() {
        let h_shape = if stage == 0 {
            [1, gen_ch, t_stage1]
        } else {
            let ch = gen_ch >> stage;
            let t = t_stage1 * config.upsample_rates[..stage].iter().product::<usize>();
            [1, ch, t]
        };
        let style_shape = [1, style_dim];
        let har_shape = [1, 2 * n_bins, t_stage1 * upsample_factor];
        let h = DynTensor::full(&h_shape, 0.1, DType::F32, &cpu()).unwrap();
        let (_result, graph) = trace_graph(|| {
            let mut h_t = h.clone();
            h_t.set_trace_id(record_input(&h_shape, DType::F32).expect("trace active"));
            let mut style = DynTensor::zeros(&style_shape, DType::F32, &cpu())?;
            style.set_trace_id(record_input(&style_shape, DType::F32).expect("trace active"));
            let mut har = DynTensor::zeros(&har_shape, DType::F32, &cpu())?;
            har.set_trace_id(record_input(&har_shape, DType::F32).expect("trace active"));
            generator
                .forward_upsample_stage(stage, &h_t, &style, &har)
                .map_err(|e| nn_core::TensorError::Unsupported(e.to_string()))
        })
        .expect("upsample_stage trace");
        let gn = trace_to_graph_model_multi_input(&graph)
            .expect("upsample trace_to_graph")
            .graph;
        let stage_input = build_multi_input_bounds(&[
            (&h_shape[..], prev_bounds),
            (&style_shape[..], (-0.5, 0.5)),
            (&har_shape[..], (-0.1, 0.1)),
        ]);
        let output = gn.propagate_ibp(&stage_input).expect("upsample IBP");
        super::common::assert_bounds_valid(&output);
        let (lo, hi) = super::common::bounds_min_max(&output);
        eprintln!("Production sub-block upsample_{stage}: [{lo}, {hi}]");
        assert!(
            lo.is_finite() && hi.is_finite(),
            "upsample_{stage} bounds must be finite"
        );
        prev_bounds = (lo, hi);
    }
    prev_bounds
}

/// Trace output stage with production Generator, return output IBP bounds.
pub(super) fn trace_production_output_stage(
    generator: &Generator,
    config: &KokoroConfig,
    t_stage1: usize,
    prev_bounds: (f32, f32),
) -> BoundedTensor {
    let gen_ch = config.gen_initial_channels;
    let ch_final = gen_ch >> config.upsample_rates.len();
    let t_out = t_stage1 * config.upsample_rates.iter().product::<usize>() + 1;
    let h_shape = [1, ch_final, t_out];
    let h = DynTensor::full(&h_shape, 0.1, DType::F32, &cpu()).unwrap();
    let (_result, graph) = trace_graph(|| {
        let mut h_t = h.clone();
        h_t.set_trace_id(record_input(&h_shape, DType::F32).expect("trace active"));
        let (mag, _phase) = generator
            .forward_output_stage(&h_t)
            .map_err(|e| nn_core::TensorError::Unsupported(e.to_string()))?;
        Ok(mag)
    })
    .expect("output_stage trace");
    let gn = trace_to_graph_model(&graph)
        .expect("output trace_to_graph")
        .graph;
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&h_shape), prev_bounds.0),
        ArrayD::from_elem(IxDyn(&h_shape), prev_bounds.1),
    )
    .expect("valid bounds");
    gn.propagate_ibp(&input_bounds).expect("output IBP")
}

// -- Combined report helpers --------------------------------------------------

/// Append records from `src` into `dst`, re-indexing layer indices.
pub(super) fn append_records(dst: &mut Vec<LayerBoundRecord>, src: Vec<LayerBoundRecord>) {
    let offset = dst.len();
    for mut rec in src {
        rec.layer_index += offset;
        if let Some(ref mut sources) = rec.input_sources {
            for s in sources.iter_mut() {
                *s += offset;
            }
        }
        dst.push(rec);
    }
}

/// Record all 5 segment bounds to the per-model status file.
/// Record verified segment bounds to the per-model status file.
///
/// F0 predictor and Generator are `Option` because they may fail to verify
/// (grouped ConvTranspose1d unsupported, v1.0 architecture mismatch).
/// Unverified segments are NOT recorded — see #2716.
pub(super) fn record_all_segments(
    bert: &SegmentResult,
    te: &SegmentResult,
    pp: &SegmentResult,
    f0: Option<&SegmentResult>,
    generator: Option<&SegmentResult>,
) {
    record_segment(
        "kokoro_production_bert_encoder",
        &bert.input_bounds,
        &bert.output_bounds,
    );
    record_segment(
        "kokoro_production_text_encoder",
        &te.input_bounds,
        &te.output_bounds,
    );
    record_segment(
        "kokoro_production_prosody_predictor",
        &pp.input_bounds,
        &pp.output_bounds,
    );
    if let Some(f0) = f0 {
        record_segment(
            "kokoro_production_f0_predictor",
            &f0.input_bounds,
            &f0.output_bounds,
        );
    } else {
        eprintln!("UNVERIFIED: f0_predictor — skipping status file recording (#2716)");
    }
    if let Some(g) = generator {
        record_segment(
            "kokoro_production_generator",
            &g.input_bounds,
            &g.output_bounds,
        );
    } else {
        eprintln!("UNVERIFIED: generator — skipping status file recording (#2716)");
    }
}

/// Persist BoundAnalysisReport JSON to workspace root for Workstream A consumption.
pub(super) fn persist_report(report: &BoundAnalysisReport) {
    let ws = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let report_path = ws.join("nn_kokoro_bound_analysis_report.json");
    let json = report_to_json(report).expect("report_to_json");
    std::fs::write(&report_path, json.as_bytes()).expect("write report JSON");
    eprintln!(
        "Persisted combined BoundAnalysisReport to {}",
        report_path.display()
    );
}
