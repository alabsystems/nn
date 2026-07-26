// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Traced Kokoro model structure + Generator tests.
//!
//! Extracted from `compose_kokoro_traced.rs` to keep both files under
//! the 500-line limit (#2633).
//!
//! - ProsodyPredictor structure test: verifies op types in traced graph
//! - Generator properties: traces vocoder with 3 inputs, verifies IBP
//!
//! Part of #2633, Part of #2224, Part of #2218.

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, TensorError, VarBuilder};
use nn_models::kokoro_decoder::Generator;
use nn_models::kokoro_tts::ProsodyPredictor;
use nn_verify::trace_to_graph_model_multi_input;

use super::common::bounds_min_max;
use super::common::kokoro_weights::{
    assert_all_finite, build_test_generator, propagate_multi_input_ibp, GEN_CH, GEN_N_BINS,
};

// Re-use constants and helpers from the parent traced module.
use super::compose_kokoro_traced::{
    count_trace_inputs, prosody_predictor_weights, D_EN, STYLE_DIM,
};

// -- ProsodyPredictor structure test ------------------------------------------

/// Trace ProsodyPredictor and return the computation graph.
fn trace_prosody_predictor_graph() -> nn_core::dyn_tensor::trace::ComputationGraph {
    let weights = prosody_predictor_weights();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let prosody = ProsodyPredictor::load(&vb, D_EN, STYLE_DIM, 1, 50).unwrap();
    let batch = 1;
    let seq_len = 2;
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
        let (dur_logits, _features) = prosody.forward(&text, &sty)?;
        Ok(dur_logits)
    })
    .unwrap();
    graph
}

#[test]
fn test_trace_kokoro_prosody_predictor_structure() {
    let graph = trace_prosody_predictor_graph();

    use nn_core::dyn_tensor::trace::TraceOp;
    let mut linear_count = 0;
    let mut lstm_count = 0;
    let mut flip_count = 0;
    let mut cat_count = 0;
    let mut transpose_count = 0;

    for node in graph.nodes() {
        match node.op() {
            TraceOp::Linear { .. } => linear_count += 1,
            TraceOp::Lstm { .. } => lstm_count += 1,
            TraceOp::Flip { .. } => flip_count += 1,
            TraceOp::Cat { .. } => cat_count += 1,
            TraceOp::Transpose { .. } => transpose_count += 1,
            _ => {}
        }
    }

    eprintln!("ProsodyPredictor structure:");
    eprintln!("  Linear: {linear_count}");
    eprintln!("  LSTM: {lstm_count}");
    eprintln!("  Flip: {flip_count}");
    eprintln!("  Cat: {cat_count}");
    eprintln!("  Transpose: {transpose_count}");

    // ProsodyPredictor has at least 1 LSTM (forward path).
    assert!(lstm_count >= 1, "expected >= 1 LSTM op, got {lstm_count}");
    // BiLSTM produces Flip + Cat for reverse direction.
    assert!(flip_count >= 1, "expected >= 1 Flip op, got {flip_count}");
    assert!(cat_count >= 1, "expected >= 1 Cat op, got {cat_count}");
    // Multiple transposes for [B, D, T] <-> [B, T, D] conversions.
    assert!(
        transpose_count >= 4,
        "expected >= 4 Transpose ops, got {transpose_count}"
    );
}

// -- Generator property tests -------------------------------------------------

fn build_local_generator() -> Generator {
    build_test_generator(0.01, STYLE_DIM)
}

/// Trace Kokoro Generator (vocoder) and verify trace + IBP propagation.
///
/// Exercises: Conv1d, ConvTranspose1d, ResBlock (Snake activation + AdaIN),
/// noise injection, conv_post, narrow, clamp, exp, sin — the full Generator path.
///
/// Uses multi-input trace with 3 variable inputs (x, style, har_source) so
/// noise_convs Conv1d gets an activation input instead of a constant (#2413 AC4).
#[test]
fn test_trace_kokoro_generator_properties() {
    let generator = build_local_generator();

    let batch = 1;
    let t_in = 8;
    let t_full = 16;
    let input_shape = [batch, GEN_CH, t_in];
    let style_shape = [batch, STYLE_DIM];
    let har_shape = [batch, 2 * GEN_N_BINS, t_full];
    let x = DynTensor::full(&input_shape, 0.1, DType::F32, &cpu()).unwrap();

    // Trace Generator with all 3 inputs as variables so noise_convs Conv1d
    // gets an activation input (har_source) instead of a constant.
    let (_result, graph) = trace_graph(|| {
        let mut inp = x.clone();
        let id_x = record_input(&input_shape, DType::F32).unwrap();
        inp.set_trace_id(id_x);

        let mut style = DynTensor::zeros(&style_shape, DType::F32, &cpu())?;
        let id_style = record_input(&style_shape, DType::F32).unwrap();
        style.set_trace_id(id_style);

        let mut har_source = DynTensor::zeros(&har_shape, DType::F32, &cpu())?;
        let id_har = record_input(&har_shape, DType::F32).unwrap();
        har_source.set_trace_id(id_har);

        let (mag, _phase) = generator
            .forward(&inp, &style, &har_source)
            .map_err(|e| TensorError::Unsupported(e.to_string()))?;
        Ok(mag)
    })
    .unwrap();

    // AC1: Trace captures non-trivial graph.
    let node_count = graph.nodes().len();
    assert!(
        node_count >= 20,
        "expected at least 20 traced nodes for Generator, got {node_count}"
    );
    assert_eq!(
        count_trace_inputs(&graph),
        3,
        "Generator should have 3 verified inputs (x, style, har_source)"
    );
    eprintln!("Generator: traced {node_count} nodes");

    // AC2: trace_to_graph_model succeeds now that FusedAdainResBlock is decomposed (#2547).
    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("Generator trace->graph should succeed with FusedAdainResBlock decomposition")
        .graph;
    assert!(
        gn.num_nodes() > 0,
        "GraphNetwork should have nodes after FusedAdainResBlock decomposition"
    );
    eprintln!("Generator: GraphNetwork has {} nodes", gn.num_nodes());

    // AC3: IBP propagation produces finite bounds (3-input model).
    let output = propagate_multi_input_ibp(
        &gn,
        &[
            (&input_shape[..], (-1.0, 1.0)),
            (&style_shape[..], (-0.5, 0.5)),
            (&har_shape[..], (-1.0, 1.0)),
        ],
    );
    assert_all_finite(&output, "Generator (properties)");
    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    // Tightness assertion (#2594): Generator with Snake + AdaIN may have wider bounds.
    // Threshold 1e6 catches vacuously wide results while allowing IBP over-approximation.
    assert!(
        width < 1e6,
        "Generator: bounds width {width} exceeds 1e6 (vacuously wide)"
    );
    eprintln!("Generator (properties) IBP bounds: [{lo_min}, {hi_max}], width={width:.4}");
}
