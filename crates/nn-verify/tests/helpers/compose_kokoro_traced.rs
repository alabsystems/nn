// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Traced Kokoro model verification tests (#2224).
//!
//! These tests trace REAL Kokoro sub-models with `trace_graph()`, translate
//! to NY `GraphNetwork` via `trace_to_graph_model_multi_input()`,
//! and propagate bounds (IBP/CROWN).
//!
//! Phase 2: extends Phase 1b (compose_kokoro_trace_full.rs) by tracing the
//! ProsodyPredictor — a multi-input component requiring the SliceLayer stacking
//! infrastructure from #2377.
//!
//! Part of #2224 (trace real KokoroModel and verify with NY).

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};
use nn_models::kokoro_tts::ProsodyPredictor;
use nn_models::kokoro_tts::TextEncoder;
use nn_verify::trace_to_graph_model_multi_input;
use std::collections::HashMap;

use super::common::bounds_min_max;
use super::common::kokoro_weights::{
    assert_all_finite, propagate_multi_input_ibp, text_encoder_weights, z,
};

// -- Shared verification helpers ----------------------------------------------

/// Count Input nodes in a trace graph.
pub(super) fn count_trace_inputs(graph: &nn_core::dyn_tensor::trace::ComputationGraph) -> usize {
    graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op(), nn_core::dyn_tensor::trace::TraceOp::Input))
        .count()
}

// -- Test-sized Kokoro dimensions --------------------------------------------

/// d_en: encoder dimension (must be even for BiLSTM hidden = d_en/2).
pub(super) const D_EN: usize = 8;
/// PlBert hidden size (bert_encoder maps from this to d_en).
const HIDDEN: usize = 8;
/// Token vocabulary size for TextEncoder.
const VOCAB_SIZE: usize = 16;
/// Style embedding dimension.
pub(super) const STYLE_DIM: usize = 4;

// -- Weight construction helpers ---------------------------------------------

/// Insert a weight tensor with alternating +0.01/-0.01 values (#2428).
///
/// Mixed-sign weights exercise the IBP bound-flip path where negative weights
/// swap lower/upper during interval matmul. Using only positive weights (the
/// `z()` helper) never triggers this path, making `lo <= hi` assertions vacuous.
fn z_mixed(m: &mut HashMap<String, DynTensor>, name: &str, shape: &[usize]) {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| if i % 2 == 0 { 0.01 } else { -0.01 })
        .collect();
    let tensor = DynTensor::from_vec(data, shape, &cpu()).unwrap();
    m.insert(name.to_string(), tensor);
}

/// Build minimal weights for ProsodyPredictor (1 block, d_model=D_EN, style_dim=STYLE_DIM).
///
/// Architecture: DurationEncoder (1× BiLSTM + AdaLayerNorm) + final duration BiLSTM + projection.
/// Weight prefix: `duration.lstms.{i}.*`, `duration.norms.{i}.*`, `duration.duration_proj.*`,
/// `lstm.*` (final BiLSTM).
pub(super) fn prosody_predictor_weights() -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    let hidden = D_EN / 2;
    let lstm_input = D_EN + STYLE_DIM;

    // DurationEncoder block 0: BiLSTM (forward + backward directions)
    z(
        &mut m,
        "duration.lstms.0.weight_ih_l0",
        &[4 * hidden, lstm_input],
    );
    z(
        &mut m,
        "duration.lstms.0.weight_hh_l0",
        &[4 * hidden, hidden],
    );
    z(&mut m, "duration.lstms.0.bias_ih_l0", &[4 * hidden]);
    z(&mut m, "duration.lstms.0.bias_hh_l0", &[4 * hidden]);
    z(
        &mut m,
        "duration.lstms.0.weight_ih_l0_reverse",
        &[4 * hidden, lstm_input],
    );
    z(
        &mut m,
        "duration.lstms.0.weight_hh_l0_reverse",
        &[4 * hidden, hidden],
    );
    z(&mut m, "duration.lstms.0.bias_ih_l0_reverse", &[4 * hidden]);
    z(&mut m, "duration.lstms.0.bias_hh_l0_reverse", &[4 * hidden]);

    // DurationEncoder block 0: AdaLayerNorm (norm + style projection)
    z(&mut m, "duration.norms.0.norm.weight", &[D_EN]);
    z(&mut m, "duration.norms.0.norm.bias", &[D_EN]);
    z(&mut m, "duration.norms.0.fc.weight", &[2 * D_EN, STYLE_DIM]);
    z(&mut m, "duration.norms.0.fc.bias", &[2 * D_EN]);

    // DurationEncoder: duration projection (d_model -> max_dur)
    z(&mut m, "duration.duration_proj.weight", &[50, D_EN]);
    z(&mut m, "duration.duration_proj.bias", &[50]);

    // Final duration BiLSTM (forward + backward directions)
    z(&mut m, "lstm.weight_ih_l0", &[4 * hidden, lstm_input]);
    z(&mut m, "lstm.weight_hh_l0", &[4 * hidden, hidden]);
    z(&mut m, "lstm.bias_ih_l0", &[4 * hidden]);
    z(&mut m, "lstm.bias_hh_l0", &[4 * hidden]);
    z(
        &mut m,
        "lstm.weight_ih_l0_reverse",
        &[4 * hidden, lstm_input],
    );
    z(&mut m, "lstm.weight_hh_l0_reverse", &[4 * hidden, hidden]);
    z(&mut m, "lstm.bias_ih_l0_reverse", &[4 * hidden]);
    z(&mut m, "lstm.bias_hh_l0_reverse", &[4 * hidden]);

    m
}

/// Build ProsodyPredictor weights with mixed-sign weight matrices (#2428).
///
/// Uses `z_mixed` for weight matrices and `z` for biases so that IBP interval
/// matmul exercises the negative-weight bound-flip path (lower/upper swap).
fn prosody_predictor_weights_mixed() -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    let hidden = D_EN / 2;
    let lstm_input = D_EN + STYLE_DIM;

    // DurationEncoder block 0: BiLSTM (mixed-sign weights)
    z_mixed(
        &mut m,
        "duration.lstms.0.weight_ih_l0",
        &[4 * hidden, lstm_input],
    );
    z_mixed(
        &mut m,
        "duration.lstms.0.weight_hh_l0",
        &[4 * hidden, hidden],
    );
    z(&mut m, "duration.lstms.0.bias_ih_l0", &[4 * hidden]);
    z(&mut m, "duration.lstms.0.bias_hh_l0", &[4 * hidden]);
    z_mixed(
        &mut m,
        "duration.lstms.0.weight_ih_l0_reverse",
        &[4 * hidden, lstm_input],
    );
    z_mixed(
        &mut m,
        "duration.lstms.0.weight_hh_l0_reverse",
        &[4 * hidden, hidden],
    );
    z(&mut m, "duration.lstms.0.bias_ih_l0_reverse", &[4 * hidden]);
    z(&mut m, "duration.lstms.0.bias_hh_l0_reverse", &[4 * hidden]);

    // AdaLayerNorm
    z(&mut m, "duration.norms.0.norm.weight", &[D_EN]);
    z(&mut m, "duration.norms.0.norm.bias", &[D_EN]);
    z_mixed(&mut m, "duration.norms.0.fc.weight", &[2 * D_EN, STYLE_DIM]);
    z(&mut m, "duration.norms.0.fc.bias", &[2 * D_EN]);

    // Duration projection (mixed-sign)
    z_mixed(&mut m, "duration.duration_proj.weight", &[50, D_EN]);
    z(&mut m, "duration.duration_proj.bias", &[50]);

    // Final duration BiLSTM (mixed-sign weights)
    z_mixed(&mut m, "lstm.weight_ih_l0", &[4 * hidden, lstm_input]);
    z_mixed(&mut m, "lstm.weight_hh_l0", &[4 * hidden, hidden]);
    z(&mut m, "lstm.bias_ih_l0", &[4 * hidden]);
    z(&mut m, "lstm.bias_hh_l0", &[4 * hidden]);
    z_mixed(
        &mut m,
        "lstm.weight_ih_l0_reverse",
        &[4 * hidden, lstm_input],
    );
    z_mixed(&mut m, "lstm.weight_hh_l0_reverse", &[4 * hidden, hidden]);
    z(&mut m, "lstm.bias_ih_l0_reverse", &[4 * hidden]);
    z(&mut m, "lstm.bias_hh_l0_reverse", &[4 * hidden]);

    m
}

// -- Tests -------------------------------------------------------------------

/// Trace ProsodyPredictor and verify graph conversion + IBP propagation.
///
/// ProsodyPredictor is the first multi-input Kokoro component to be traced:
///   - Input 1: text_features `[B, d_en, T]` (from bert_encoder + TextEncoder)
///   - Input 2: style `[B, style_dim]` (voice embedding)
///
/// Exercises: Conv1d, LayerNorm, Linear (style projection), Narrow, Unsqueeze,
/// Expand, Constant (ones), broadcast_add, broadcast_mul, Cat, Transpose, LSTM,
/// Squeeze — covering 12+ TraceOp variants through the multi-input pipeline.
///
/// IBP propagation enabled after #2413 fixes:
/// - SnakeTensor 1/alpha constant-folded (no weight-only Reciprocal node)
/// - Constant output registered as weight (no BroadcastAdd shape mismatch)
#[test]
fn test_trace_kokoro_prosody_predictor() {
    let weights = prosody_predictor_weights();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let prosody = ProsodyPredictor::load(&vb, D_EN, STYLE_DIM, 1, 50).unwrap();

    let batch = 1;
    let seq_len = 3;
    let text_shape = [batch, D_EN, seq_len];
    let style_shape = [batch, STYLE_DIM];

    let text_features = DynTensor::full(&text_shape, 0.1, DType::F32, &cpu()).unwrap();
    let style = DynTensor::full(&style_shape, 0.05, DType::F32, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut text = text_features.clone();
        let id_text = record_input(&text_shape, DType::F32).unwrap();
        text.set_trace_id(id_text);
        let mut sty = style.clone();
        let id_style = record_input(&style_shape, DType::F32).unwrap();
        sty.set_trace_id(id_style);
        let (dur, _feat) = prosody.forward(&text, &sty)?;
        Ok(dur)
    })
    .unwrap();

    // AC1: Trace captures non-trivial graph.
    let node_count = graph.nodes().len();
    assert!(
        node_count >= 10,
        "expected at least 10 traced nodes for ProsodyPredictor, got {node_count}"
    );

    assert_eq!(
        count_trace_inputs(&graph),
        2,
        "ProsodyPredictor should have 2 inputs"
    );

    // AC2: trace_to_graph_model succeeds now that AdaLayerNorm is decomposed (#2547).
    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("ProsodyPredictor trace→graph should succeed with AdaLayerNorm decomposition")
        .graph;
    assert!(
        gn.num_nodes() > 0,
        "GraphNetwork should have nodes after AdaLayerNorm decomposition"
    );
    eprintln!(
        "ProsodyPredictor: GraphNetwork has {} nodes",
        gn.num_nodes()
    );

    // AC3: IBP propagation produces finite bounds.
    let output = propagate_multi_input_ibp(
        &gn,
        &[
            (&text_shape[..], (-1.0, 1.0)),
            (&style_shape[..], (-0.5, 0.5)),
        ],
    );
    assert_all_finite(&output, "ProsodyPredictor");
    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    // Tightness assertion (#2594): vacuously wide bounds (e.g., [-1e30, 1e30]) fail here.
    // Threshold 1e6 calibrated from traced tests: prosody at test dims produces width < 10.
    assert!(
        width < 1e6,
        "ProsodyPredictor: bounds width {width} exceeds 1e6 (vacuously wide)"
    );
    eprintln!("ProsodyPredictor IBP bounds: [{lo_min}, {hi_max}], width={width:.4}");
}

/// Mixed-sign weight variant of ProsodyPredictor IBP test (#2428).
///
/// Exercises the negative-weight bound-flip path in IBP interval matmul.
/// With positive-only weights (the `z()` helper), `lower_out = W * lower_in`
/// and `upper_out = W * upper_in`. Negative weights swap: `lower_out = W * upper_in`.
/// This test catches bugs where the swap is missing (inverted bounds).
#[test]
fn test_trace_kokoro_prosody_predictor_mixed_weights() {
    let weights = prosody_predictor_weights_mixed();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let prosody = ProsodyPredictor::load(&vb, D_EN, STYLE_DIM, 1, 50).unwrap();

    let batch = 1;
    let seq_len = 3;
    let text_shape = [batch, D_EN, seq_len];
    let style_shape = [batch, STYLE_DIM];

    let text_features = DynTensor::full(&text_shape, 0.1, DType::F32, &cpu()).unwrap();
    let style = DynTensor::full(&style_shape, 0.05, DType::F32, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut text = text_features.clone();
        let id_text = record_input(&text_shape, DType::F32).unwrap();
        text.set_trace_id(id_text);
        let mut sty = style.clone();
        let id_style = record_input(&style_shape, DType::F32).unwrap();
        sty.set_trace_id(id_style);
        let (dur, _feat) = prosody.forward(&text, &sty)?;
        Ok(dur)
    })
    .unwrap();

    // AdaLayerNorm decomposition now supported (#2547).
    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("ProsodyPredictor (mixed) trace→graph should succeed")
        .graph;
    assert!(
        gn.num_nodes() > 0,
        "GraphNetwork should have nodes after AdaLayerNorm decomposition (mixed weights)"
    );
    eprintln!(
        "ProsodyPredictor (mixed): GraphNetwork has {} nodes",
        gn.num_nodes()
    );

    // IBP propagation with mixed-sign weights exercises bound-flip path.
    let output = propagate_multi_input_ibp(
        &gn,
        &[
            (&text_shape[..], (-1.0, 1.0)),
            (&style_shape[..], (-0.5, 0.5)),
        ],
    );
    assert_all_finite(&output, "ProsodyPredictor (mixed)");
    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    // Tightness assertion (#2594): mixed-sign weights may widen bounds vs positive-only.
    assert!(
        width < 1e6,
        "ProsodyPredictor (mixed): bounds width {width} exceeds 1e6 (vacuously wide)"
    );
    eprintln!("ProsodyPredictor (mixed) IBP bounds: [{lo_min}, {hi_max}], width={width:.4}");
}

/// Build text-to-duration pipeline models for tracing.
fn build_text_to_duration_models() -> (TextEncoder, ProsodyPredictor) {
    let te_weights = text_encoder_weights(VOCAB_SIZE, D_EN, 0.01);
    let vb_te = VarBuilder::from_tensors(te_weights, DType::F32, &cpu());
    let text_encoder = TextEncoder::load(&vb_te, VOCAB_SIZE, D_EN).unwrap();
    let pp_weights = prosody_predictor_weights();
    let vb_pp = VarBuilder::from_tensors(pp_weights, DType::F32, &cpu());
    let prosody = ProsodyPredictor::load(&vb_pp, D_EN, STYLE_DIM, 1, 50).unwrap();
    (text_encoder, prosody)
}

/// Trace text-to-duration pipeline: TextEncoder -> ProsodyPredictor.
///
/// End-to-end pipeline crossing single-input to multi-input boundary.
/// Two inputs: token IDs (text) and style (voice embedding).
///
/// IBP propagation enabled after #2413 fixes.
#[test]
fn test_trace_kokoro_text_to_duration_pipeline() {
    let (text_encoder, prosody) = build_text_to_duration_models();
    let batch = 1;
    let seq_len = 3;
    let token_shape = [batch, seq_len];
    let style_shape = [batch, STYLE_DIM];
    let token_ids: Vec<i64> = (0..batch * seq_len)
        .map(|i| (i % VOCAB_SIZE) as i64)
        .collect();
    let tokens = DynTensor::from_vec_i64(token_ids, &token_shape, &cpu()).unwrap();
    let style = DynTensor::full(&style_shape, 0.05, DType::F32, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = tokens.clone();
        let id_x = record_input(&token_shape, DType::I64).unwrap();
        x.set_trace_id(id_x);
        let mut sty = style.clone();
        let id_sty = record_input(&style_shape, DType::F32).unwrap();
        sty.set_trace_id(id_sty);
        let text_features = text_encoder.forward(&x)?;
        let (dur_logits, _features) = prosody.forward(&text_features, &sty)?;
        Ok(dur_logits)
    })
    .unwrap();

    // AC1: Trace captures non-trivial graph.
    let node_count = graph.nodes().len();
    assert!(
        node_count >= 20,
        "expected >= 20 traced nodes, got {node_count}"
    );
    assert_eq!(
        count_trace_inputs(&graph),
        2,
        "pipeline should have 2 inputs"
    );

    // AC2: trace_to_graph_model succeeds now that AdaLayerNorm is decomposed (#2547).
    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("text-to-duration trace→graph should succeed with AdaLayerNorm decomposition")
        .graph;
    assert!(
        gn.num_nodes() > 0,
        "GraphNetwork should have nodes after text-to-duration pipeline decomposition"
    );
    eprintln!(
        "Text-to-duration pipeline: GraphNetwork has {} nodes",
        gn.num_nodes()
    );

    // AC3: IBP propagation produces finite bounds.
    let output = propagate_multi_input_ibp(
        &gn,
        &[
            (&token_shape[..], (0.0, VOCAB_SIZE as f32)),
            (&style_shape[..], (-0.5, 0.5)),
        ],
    );
    assert_all_finite(&output, "Text-to-duration pipeline");
    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    // Tightness assertion (#2594): text-to-duration at test dims produces width ~0.095.
    // Threshold 1e6 is conservative for production dimensions.
    assert!(
        width < 1e6,
        "Text-to-duration: bounds width {width} exceeds 1e6 (vacuously wide)"
    );
    eprintln!("Text-to-duration pipeline IBP bounds: [{lo_min}, {hi_max}], width={width:.4}");
}
