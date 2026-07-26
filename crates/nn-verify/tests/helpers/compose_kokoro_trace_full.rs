// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro trace-to-graph verification tests.
//!
//! These tests trace REAL Kokoro model code (not hand-built graphs) through
//! `trace_graph()` → `trace_to_graph_model()` → IBP propagation.
//!
//! Phase 1b of designs/2026-03-15-segmented-kokoro-trace-verification.md.
//! Part of #2224 (trace real KokoroModel and verify with NY).
//! Part of #2329 (8-op decomposition — covers Flip via BiLSTM).

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Linear, Module};
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};
use nn_models::kokoro_tts::TextEncoder;
use nn_verify::trace_to_graph_model;

use super::common::kokoro_weights::{bert_encoder_weights, text_encoder_weights};
use super::common::{
    assert_bounds_valid, assert_bounds_width, assert_crown_tighter_when_not_fallback,
    bounds_min_max, uniform_bounds,
};

// -- Test-sized Kokoro dimensions (matching nn-models test config) -----------

/// d_en: encoder dimension (must be even for BiLSTM hidden = d_en/2).
const D_EN: usize = 8;
/// PlBert hidden size (bert_encoder maps from this to d_en).
const HIDDEN: usize = 8;
/// Token vocabulary size for TextEncoder.
const VOCAB_SIZE: usize = 16;

// -- Tests --------------------------------------------------------------------

/// Trace bert_encoder (Linear) through trace_to_graph and verify IBP.
///
/// Simplest possible Kokoro trace test: single Linear layer.
/// Verifies the trace → graph → IBP pipeline works for Kokoro components.
#[test]
fn test_trace_kokoro_bert_encoder() {
    let weights = bert_encoder_weights(D_EN, HIDDEN, 0.0);
    let vb = VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let w = vb.get(&[D_EN, HIDDEN], "weight").unwrap();
    let b = vb.get(&[D_EN], "bias").unwrap();
    let bert_encoder = Linear::new(w, Some(b)).unwrap();

    let batch = 1;
    let seq_len = 3;
    let input_shape = [batch, seq_len, HIDDEN];
    let bert_output = DynTensor::full(&input_shape, 0.1, DType::F32, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = bert_output.clone();
        let id = record_input(x.dims(), DType::F32).unwrap();
        x.set_trace_id(id);
        let encoded = bert_encoder.forward(&x)?;
        Ok(encoded)
    })
    .unwrap();

    // Convert trace to NY GraphNetwork.
    let gn = trace_to_graph_model(&graph)
        .expect("trace_to_graph_model should succeed for bert_encoder")
        .graph;

    // Run IBP propagation.
    let input_bounds = uniform_bounds(&input_shape, 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    // With zero weights and zero bias, output bounds should be tight around 0.
    let (lo_min, hi_max) = bounds_min_max(&output);
    assert!(
        lo_min >= -1e-5 && hi_max <= 1e-5,
        "zero-weight Linear should produce near-zero bounds, got [{lo_min}, {hi_max}]"
    );
}

/// Trace TextEncoder (Embedding + Conv + LayerNorm + BiLSTM) through trace_to_graph
/// and verify IBP.
///
/// Traces real Kokoro TextEncoder including:
/// - Embedding (token ID lookup)
/// - Conv1d (3x with LayerNorm + LeakyReLU)
/// - Transpose (dimension reordering for BiLSTM)
/// - Flip (#2329 op — used by BiLSTM backward direction)
/// - Lstm (2x — forward and backward, single composite op per direction #2224)
/// - Cat (BiLSTM output concatenation)
/// - Linear (TextEncoder output projection)
#[test]
fn test_trace_kokoro_text_pipeline() {
    let te_weights = text_encoder_weights(VOCAB_SIZE, D_EN, 0.0);
    let vb_te = VarBuilder::from_tensors(te_weights, DType::F32, &cpu());
    let text_encoder = TextEncoder::load(&vb_te, VOCAB_SIZE, D_EN).unwrap();

    let batch = 1;
    let seq_len = 3;
    let token_shape = [batch, seq_len];
    let token_ids: Vec<i64> = (0..batch * seq_len)
        .map(|i| (i % VOCAB_SIZE) as i64)
        .collect();
    let tokens = DynTensor::from_vec_i64(token_ids, &token_shape, &cpu()).unwrap();

    // Trace the TextEncoder: Embedding → Conv → LayerNorm → BiLSTM → projection.
    let (_result, graph) = trace_graph(|| {
        let mut x = tokens.clone();
        let id = record_input(x.dims(), DType::I64).unwrap();
        x.set_trace_id(id);
        let text_features = text_encoder.forward(&x)?;
        Ok(text_features)
    })
    .unwrap();

    // Verify the trace captured the expected op types.
    let node_count = graph.nodes().len();
    // Expected: 1 Input + 1 Linear + 1 Transpose + (2 Transpose + 2 Lstm +
    //   2 Flip + 1 Cat) BiLSTM + (1 Transpose + 1 Linear + 1 Transpose) proj
    // = ~13+ nodes. Exact count depends on LSTM per-timestep decomposition.
    assert!(
        node_count >= 10,
        "expected at least 10 traced nodes for text pipeline, got {node_count}"
    );

    // Convert trace to NY GraphNetwork.
    let gn = trace_to_graph_model(&graph)
        .expect("trace_to_graph_model should succeed for text pipeline")
        .graph;

    // Run IBP propagation.
    let input_bounds = uniform_bounds(&token_shape, VOCAB_SIZE as f32);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
    // Tightness assertion (#2594): text pipeline IBP should not be vacuously wide.
    assert_bounds_width(&output, 1e6, "text_pipeline_ibp");

    // Bounds should be finite and non-trivial (the pipeline has activations).
    let (lo_min, hi_max) = bounds_min_max(&output);
    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "output bounds must be finite, got [{lo_min}, {hi_max}]"
    );
}

/// CROWN propagation on the text pipeline (tighter bounds than IBP).
///
/// Runs both IBP and CROWN, asserting CROWN produces tighter bounds when
/// it doesn't fall back. Validates the traced graph is compatible with
/// CROWN's backward linear relaxation, not just forward IBP.
#[test]
fn test_trace_kokoro_text_pipeline_crown() {
    let te_weights = text_encoder_weights(VOCAB_SIZE, D_EN, 0.0);
    let vb_te = VarBuilder::from_tensors(te_weights, DType::F32, &cpu());
    let text_encoder = TextEncoder::load(&vb_te, VOCAB_SIZE, D_EN).unwrap();

    let batch = 1;
    let seq_len = 3;
    let token_shape = [batch, seq_len];
    let token_ids: Vec<i64> = (0..batch * seq_len)
        .map(|i| (i % VOCAB_SIZE) as i64)
        .collect();
    let tokens = DynTensor::from_vec_i64(token_ids, &token_shape, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = tokens.clone();
        let id = record_input(x.dims(), DType::I64).unwrap();
        x.set_trace_id(id);
        let text_features = text_encoder.forward(&x)?;
        Ok(text_features)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("trace_to_graph_model")
        .graph;
    let input_bounds = uniform_bounds(&token_shape, VOCAB_SIZE as f32);

    let (method, output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&gn, &input_bounds);

    assert_bounds_valid(&output);
    // Tightness assertion (#2594): CROWN text pipeline width ceiling.
    assert_bounds_width(&output, 1e6, "text_pipeline_crown");

    eprintln!(
        "Kokoro text pipeline CROWN: method={method:?}, fallback={:?}",
        fallback_reason.as_deref().unwrap_or("none")
    );
}

/// Debug test: trace just a single LSTM forward_seq to isolate trace issues.
#[test]
fn test_trace_lstm_forward_seq_only() {
    use nn_core::layers::Lstm;
    let hidden = D_EN / 2; // 4
    let w_ih = DynTensor::zeros(&[4 * hidden, D_EN], DType::F32, &cpu()).unwrap();
    let w_hh = DynTensor::zeros(&[4 * hidden, hidden], DType::F32, &cpu()).unwrap();
    let b_ih = DynTensor::zeros(&[4 * hidden], DType::F32, &cpu()).unwrap();
    let b_hh = DynTensor::zeros(&[4 * hidden], DType::F32, &cpu()).unwrap();
    let lstm = Lstm::new(w_ih, w_hh, Some(b_ih), Some(b_hh), hidden).unwrap();

    let seq_len = 2;
    let batch = 1;
    let input_shape = [seq_len, batch, D_EN];
    let input_data = DynTensor::full(&input_shape, 0.1, DType::F32, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = input_data.clone();
        let id = record_input(x.dims(), DType::F32).unwrap();
        x.set_trace_id(id);
        eprintln!(
            "LSTM input trace_id={:?}, shape={:?}",
            x.trace_id(),
            x.dims()
        );
        let (output, _state) = lstm.forward_seq(&x, None)?;
        eprintln!(
            "LSTM output trace_id={:?}, shape={:?}",
            output.trace_id(),
            output.dims()
        );
        Ok(output)
    })
    .unwrap();

    eprintln!("LSTM-only graph: {} nodes", graph.nodes().len());
    for (i, node) in graph.nodes().iter().enumerate() {
        eprintln!("  node {i}: {:?}", node.op());
    }
    // At least Input + Lstm nodes.
    assert!(
        graph.nodes().len() >= 2,
        "LSTM trace should have at least 2 nodes, got {}",
        graph.nodes().len()
    );
}

/// Debug test: trace BiLstm steps manually to find exact failure.
#[test]
fn test_trace_bilstm_forward_seq_only() {
    use nn_core::layers::Lstm;
    let hidden = D_EN / 2;
    let fwd_lstm = Lstm::new(
        DynTensor::zeros(&[4 * hidden, D_EN], DType::F32, &cpu()).unwrap(),
        DynTensor::zeros(&[4 * hidden, hidden], DType::F32, &cpu()).unwrap(),
        Some(DynTensor::zeros(&[4 * hidden], DType::F32, &cpu()).unwrap()),
        Some(DynTensor::zeros(&[4 * hidden], DType::F32, &cpu()).unwrap()),
        hidden,
    )
    .unwrap();
    let bwd_lstm = Lstm::new(
        DynTensor::zeros(&[4 * hidden, D_EN], DType::F32, &cpu()).unwrap(),
        DynTensor::zeros(&[4 * hidden, hidden], DType::F32, &cpu()).unwrap(),
        Some(DynTensor::zeros(&[4 * hidden], DType::F32, &cpu()).unwrap()),
        Some(DynTensor::zeros(&[4 * hidden], DType::F32, &cpu()).unwrap()),
        hidden,
    )
    .unwrap();

    let seq_len = 2;
    let batch = 1;
    let input_shape = [seq_len, batch, D_EN];
    let input_data = DynTensor::full(&input_shape, 0.1, DType::F32, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = input_data.clone();
        let id = record_input(x.dims(), DType::F32).unwrap();
        x.set_trace_id(id);

        // Step 1: forward LSTM
        let (fwd_out, _fwd_state) = fwd_lstm.forward_seq(&x, None).map_err(|e| {
            eprintln!("FAIL at fwd_lstm.forward_seq: {e}");
            e
        })?;
        eprintln!("fwd_lstm OK, trace_id={:?}", fwd_out.trace_id());

        // Step 2: flip input for backward direction
        eprintln!(
            "pre-flip: x.trace_id={:?}, is_tracing={}",
            x.trace_id(),
            nn_core::dyn_tensor::trace::is_tracing()
        );
        let reversed = x.flip(0).map_err(|e| {
            eprintln!("FAIL at flip: {e}");
            e
        })?;
        eprintln!("flip OK, trace_id={:?}", reversed.trace_id());

        // Step 3: backward LSTM on flipped input
        let (bwd_out_rev, _bwd_state) = bwd_lstm.forward_seq(&reversed, None).map_err(|e| {
            eprintln!("FAIL at bwd_lstm.forward_seq: {e}");
            e
        })?;
        eprintln!("bwd_lstm OK, trace_id={:?}", bwd_out_rev.trace_id());

        // Step 4: flip backward output
        let bwd_out = bwd_out_rev.flip(0).map_err(|e| {
            eprintln!("FAIL at flip(bwd): {e}");
            e
        })?;
        eprintln!("flip(bwd) OK, trace_id={:?}", bwd_out.trace_id());

        // Step 5: cat
        let outputs = DynTensor::cat(&[&fwd_out, &bwd_out], 2).map_err(|e| {
            eprintln!("FAIL at cat: {e}");
            e
        })?;
        eprintln!("cat OK, trace_id={:?}", outputs.trace_id());

        Ok(outputs)
    })
    .unwrap();
    let n = graph.nodes().len();
    eprintln!("BiLSTM manual graph: {n} nodes");
    assert!(
        n >= 6,
        "BiLSTM trace: expected >=6 nodes (Input+2LSTM+2Flip+Cat), got {n}"
    );
}

/// Verify trace graph structure: correct op types and counts.
///
/// With single-composite LSTM recording (#2224), forward_seq records one LSTM
/// op per direction (not per timestep). BiLSTM produces: 2 LSTMs + 2 Flips +
/// 1 Cat. The new TextEncoder (Embedding + 3×Conv1d + 3×LayerNorm + BiLSTM +
/// Linear) produces:
/// - 1 Embedding (token lookup)
/// - 3 Conv1d (convolution blocks)
/// - 1 Linear (output projection)
/// - 9 Transposes (1 post-embed + 3×2 conv/norm + 1 post-LSTM + 1 output)
///   (BiLSTM pre-LSTM uses permute instead of transpose pair)
/// - 2 LSTMs + 2 Flips + 1 Cat (BiLSTM)
#[test]
fn test_trace_kokoro_text_pipeline_structure() {
    let te_weights = text_encoder_weights(VOCAB_SIZE, D_EN, 0.0);
    let vb_te = VarBuilder::from_tensors(te_weights, DType::F32, &cpu());
    let text_encoder = TextEncoder::load(&vb_te, VOCAB_SIZE, D_EN).unwrap();

    let batch = 1;
    let seq_len = 2;
    let token_shape = [batch, seq_len];
    let token_ids: Vec<i64> = (0..batch * seq_len)
        .map(|i| (i % VOCAB_SIZE) as i64)
        .collect();
    let tokens = DynTensor::from_vec_i64(token_ids, &token_shape, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = tokens.clone();
        let id = record_input(x.dims(), DType::I64).unwrap();
        x.set_trace_id(id);
        let text_features = text_encoder.forward(&x).map_err(|e| {
            eprintln!("FAIL at text_encoder.forward: {e}");
            e
        })?;
        eprintln!("text_encoder OK, trace_id={:?}", text_features.trace_id());
        Ok(text_features)
    })
    .unwrap();

    use nn_core::dyn_tensor::trace::TraceOp;
    let mut lstm_count = 0;
    let mut flip_count = 0;
    let mut linear_count = 0;
    let mut conv1d_count = 0;
    let mut embedding_count = 0;
    let mut cat_count = 0;
    let mut transpose_count = 0;
    for node in graph.nodes() {
        match node.op() {
            TraceOp::Lstm { .. } => lstm_count += 1,
            TraceOp::Flip { .. } => flip_count += 1,
            TraceOp::Linear { .. } => linear_count += 1,
            TraceOp::Conv1d { .. } => conv1d_count += 1,
            TraceOp::Embedding { .. } => embedding_count += 1,
            TraceOp::Cat { .. } => cat_count += 1,
            TraceOp::Transpose { .. } => transpose_count += 1,
            _ => {}
        }
    }

    // Single-composite LSTM: 1 per direction = 2 LSTM ops total (#2224).
    assert_eq!(
        lstm_count, 2,
        "expected 2 LSTM ops (1 per BiLSTM direction), got {lstm_count}"
    );
    // 2 flips: reverse input for backward, reverse backward output.
    assert_eq!(flip_count, 2, "expected 2 Flip ops for BiLSTM");
    // 1 embedding: token lookup.
    assert_eq!(
        embedding_count, 1,
        "expected 1 Embedding op, got {embedding_count}"
    );
    // 3 conv1d: one per convolution block.
    assert_eq!(conv1d_count, 3, "expected 3 Conv1d ops, got {conv1d_count}");
    // 1 linear: TextEncoder projection only (no more bert_encoder).
    assert_eq!(linear_count, 1, "expected 1 Linear op, got {linear_count}");
    // 1 cat: BiLSTM concat only (per-timestep stacking is suppressed).
    assert_eq!(
        cat_count, 1,
        "expected 1 Cat op (BiLSTM concat), got {cat_count}"
    );
    // 9 transposes: 1 post-embed + 3×2 conv/norm + 1 post-LSTM + 1 output.
    // (BiLSTM pre-LSTM now uses permute instead of double-transpose.)
    assert_eq!(
        transpose_count, 9,
        "expected 9 Transpose ops, got {transpose_count}"
    );
}
