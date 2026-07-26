// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Generator (Segment 3: Vocoder) trace-to-graph verification tests.
//!
//! Extracted from `compose_kokoro_trace_full.rs` to keep both files
//! under the 500-line limit (#2633).
//!
//! Traces the Kokoro Generator through `trace_graph()` →
//! `trace_to_graph_model_multi_input()` → IBP propagation.
//!
//! Part of #2633, Part of #2224, Part of #2218.

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, TensorError};
use nn_models::kokoro_decoder::Generator;
use nn_verify::trace_to_graph_model_multi_input;

use super::common::kokoro_weights::{
    assert_all_finite, build_test_generator as build_shared_generator, propagate_multi_input_ibp,
    GEN_CH, GEN_N_BINS,
};
use super::common::{assert_bounds_width, bounds_min_max};

// -- Generator (Segment 3: Vocoder) -------------------------------------------

const GEN_STYLE_DIM: usize = 4;

// -- Generator trace tests ----------------------------------------------------

/// Build a test-scale Generator with zero-fill weights (structure-only tracing).
fn build_local_generator() -> Generator {
    build_shared_generator(0.0, GEN_STYLE_DIM)
}

/// Trace the Kokoro Generator (Segment 3: Vocoder) and verify graph capture + IBP.
///
/// Exercises: Conv1d -> LeakyReLU -> ConvTranspose1d -> noise injection ->
/// ResBlock (FusedAdainResBlock) -> Conv1d -> narrow -> clamp -> exp/sin.
///
/// FusedAdainResBlock decomposition now supported (#2547).
#[test]
fn test_trace_kokoro_generator() {
    let generator = build_local_generator();
    let graph = trace_generator_graph(&generator);

    assert!(
        graph.nodes().len() >= 20,
        "expected >= 20 traced nodes for Generator"
    );

    // FusedAdainResBlock decomposition now supported (#2547).
    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("Generator trace→graph should succeed with FusedAdainResBlock decomposition")
        .graph;
    assert!(
        gn.num_nodes() > 0,
        "GraphNetwork should have nodes after FusedAdainResBlock decomposition"
    );
    eprintln!("Generator: GraphNetwork has {} nodes", gn.num_nodes());

    // IBP propagation: 3-input model (x, style, har_source).
    let t_in = 8;
    let t_full = 16;
    let output = propagate_multi_input_ibp(
        &gn,
        &[
            (&[1, GEN_CH, t_in], (-1.0, 1.0)),
            (&[1, GEN_STYLE_DIM], (-0.5, 0.5)),
            (&[1, 2 * GEN_N_BINS, t_full], (-1.0, 1.0)),
        ],
    );
    assert_all_finite(&output, "Generator");
    // Tightness assertion (#2594): Generator with Snake + AdaIN may have wider bounds
    // than decoder-only pipeline, but should not be vacuously wide.
    assert_bounds_width(&output, 1e6, "generator_ibp");
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Generator IBP bounds: [{lo_min}, {hi_max}]");
}

/// Trace Generator and return the computation graph (shared by structure + IBP tests).
fn trace_generator_graph(generator: &Generator) -> nn_core::dyn_tensor::trace::ComputationGraph {
    let batch = 1;
    let t_in = 8;
    let t_full = 16;
    let x = DynTensor::full(&[batch, GEN_CH, t_in], 0.1, DType::F32, &cpu()).unwrap();
    let style = DynTensor::zeros(&[batch, GEN_STYLE_DIM], DType::F32, &cpu()).unwrap();
    let har_source =
        DynTensor::zeros(&[batch, 2 * GEN_N_BINS, t_full], DType::F32, &cpu()).unwrap();
    let (_result, graph) = trace_graph(|| {
        let mut inp = x.clone();
        let id_x = record_input(inp.dims(), DType::F32).unwrap();
        inp.set_trace_id(id_x);
        let mut sty = style.clone();
        let id_s = record_input(sty.dims(), DType::F32).unwrap();
        sty.set_trace_id(id_s);
        let mut har = har_source.clone();
        let id_h = record_input(har.dims(), DType::F32).unwrap();
        har.set_trace_id(id_h);
        let (magnitude, phase) = generator
            .forward(&inp, &sty, &har)
            .map_err(|e| TensorError::Unsupported(e.to_string()))?;
        DynTensor::cat(&[&magnitude, &phase], 1)
    })
    .unwrap();
    graph
}

/// Verify Generator trace structure: key op types are present.
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
fn test_trace_kokoro_generator_structure() {
    let generator = build_local_generator();
    let graph = trace_generator_graph(&generator);

    use nn_core::dyn_tensor::trace::{KokoroFusedOp, TraceOp};
    let mut counts = [0usize; 7]; // conv1d, conv_tr, fused_resblock, sin, exp, clamp, adain_snake
    for node in graph.nodes() {
        match node.op() {
            TraceOp::Conv1d { .. } => counts[0] += 1,
            TraceOp::ConvTranspose1d { .. } => counts[1] += 1,
            TraceOp::KokoroFused(KokoroFusedOp::FusedAdainResBlock { .. }) => counts[2] += 1,
            TraceOp::Sin => counts[3] += 1,
            TraceOp::Exp => counts[4] += 1,
            TraceOp::Clamp { .. } => counts[5] += 1,
            TraceOp::KokoroFused(KokoroFusedOp::AdainSnake { .. }) => counts[6] += 1,
            _ => {}
        }
    }
    let [conv1d, conv_tr, fused_resblock, sin, exp, clamp, adain_snake] = counts;

    // Conv1d: input_conv + output_conv + noise_conv (3) + ResBlock internal convs
    // (2 per dilation layer: main ResBlock 1 dilation × 2 + noise_res 1 dilation × 2 = 4).
    assert!(conv1d >= 7, "expected >= 7 Conv1d, got {conv1d}");
    assert_eq!(conv_tr, 1, "expected 1 ConvTranspose1d, got {conv_tr}");
    // FusedAdainResBlock: 0 — replaced by decomposed NativeOp path (#2590).
    assert_eq!(
        fused_resblock, 0,
        "expected 0 FusedAdainResBlock (decomposed path), got {fused_resblock}"
    );
    // AdainSnake: 2 per dilation layer (main ResBlock 1 dilation × 2 + noise_res 1 dilation × 2 = 4).
    assert!(
        adain_snake >= 4,
        "expected >= 4 AdainSnake (decomposed ResBlock), got {adain_snake}"
    );
    assert!(sin >= 1, "expected >= 1 Sin (phase generation), got {sin}");
    assert!(exp >= 1, "expected >= 1 Exp (magnitude), got {exp}");
    // log_mag clamp (prevents exp() overflow).
    assert!(
        clamp >= 1,
        "expected >= 1 Clamp (log_mag safety guard), got {clamp}"
    );
    eprintln!(
        "Generator structure: conv1d={conv1d}, conv_tr={conv_tr}, \
         fused_resblock={fused_resblock}, adain_snake={adain_snake}, \
         sin={sin}, exp={exp}, clamp={clamp}"
    );
}
