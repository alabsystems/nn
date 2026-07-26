// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PlBert (ALBERT) trace-based NY verification.
//!
//! Traces `PlBert::forward_core()` — the full transformer pipeline:
//! LayerNorm -> Linear projection -> N x (multi-head attention + residual +
//! LayerNorm + FFN with GELU + residual + LayerNorm).
//!
//! Uses `forward_core()` instead of `forward()` because embedding lookup
//! creates integer-indexed data-dependent shapes incompatible with NY.
//! `forward_core()` takes pre-combined embeddings `[B, T, emb_dim]` as input,
//! covering the entire compute-heavy transformer pipeline.
//!
//! Exercises: Embedding LayerNorm, factorized Linear projection, multi-head
//! self-attention (Q/K/V projections, reshape, transpose, contiguous,
//! variable-variable MatMul, Softmax, output projection), FFN (Linear + GELU +
//! Linear), residual Add connections, post-attention LayerNorm, post-FFN
//! LayerNorm, shared weight iteration (ALBERT cross-layer sharing).
//!
//! Part of #2402: PlBert completely unverified by NY.
//! Part of #2218: Epic — Perfect Kokoro.

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};
use nn_models::{PlBert, PlbertConfig};
use nn_verify::trace_to_graph_model;
use std::collections::HashMap;

use super::common::kokoro_weights::{assert_all_finite, z_fill};
use super::common::{assert_bounds_valid, assert_bounds_width, bounds_min_max, uniform_bounds};

// -- Test dimensions ----------------------------------------------------------

const EMB_DIM: usize = 4;
const HIDDEN: usize = 8;
const NUM_HEADS: usize = 2;
const INTERMEDIATE: usize = 16;
const BATCH: usize = 1;
const SEQ_LEN: usize = 3;

// -- Weight builder -----------------------------------------------------------

const LAYER_PREFIX: &str = "encoder.albert_layer_groups.0.albert_layers.0";

/// Build minimal PlBert weights for tracing.
///
/// Uses small fill values (0.01) for meaningful IBP bounds and
/// LayerNorm weight=1.0 for stable normalization.
fn plbert_weights(num_hidden_layers: usize) -> (HashMap<String, DynTensor>, PlbertConfig) {
    let mut m = HashMap::new();
    let fill = 0.01;

    // Embedding LayerNorm (applied in forward_core)
    m.insert(
        "embeddings.LayerNorm.weight".into(),
        DynTensor::full(&[EMB_DIM], 1.0, DType::F32, &cpu()).unwrap(),
    );
    z_fill(&mut m, "embeddings.LayerNorm.bias", &[EMB_DIM], 0.0);

    // Factorized projection: emb_dim -> hidden_size
    z_fill(
        &mut m,
        "encoder.embedding_hidden_mapping_in.weight",
        &[HIDDEN, EMB_DIM],
        fill,
    );
    z_fill(
        &mut m,
        "encoder.embedding_hidden_mapping_in.bias",
        &[HIDDEN],
        0.0,
    );

    // Attention Q, K, V, dense projections
    for name in &[
        "attention.query",
        "attention.key",
        "attention.value",
        "attention.dense",
    ] {
        z_fill(
            &mut m,
            &format!("{LAYER_PREFIX}.{name}.weight"),
            &[HIDDEN, HIDDEN],
            fill,
        );
        z_fill(
            &mut m,
            &format!("{LAYER_PREFIX}.{name}.bias"),
            &[HIDDEN],
            0.0,
        );
    }

    // Post-attention LayerNorm
    m.insert(
        format!("{LAYER_PREFIX}.attention.LayerNorm.weight"),
        DynTensor::full(&[HIDDEN], 1.0, DType::F32, &cpu()).unwrap(),
    );
    z_fill(
        &mut m,
        &format!("{LAYER_PREFIX}.attention.LayerNorm.bias"),
        &[HIDDEN],
        0.0,
    );

    // FFN: up-project (hidden -> intermediate) + down-project (intermediate -> hidden)
    z_fill(
        &mut m,
        &format!("{LAYER_PREFIX}.ffn.weight"),
        &[INTERMEDIATE, HIDDEN],
        fill,
    );
    z_fill(
        &mut m,
        &format!("{LAYER_PREFIX}.ffn.bias"),
        &[INTERMEDIATE],
        0.0,
    );
    z_fill(
        &mut m,
        &format!("{LAYER_PREFIX}.ffn_output.weight"),
        &[HIDDEN, INTERMEDIATE],
        fill,
    );
    z_fill(
        &mut m,
        &format!("{LAYER_PREFIX}.ffn_output.bias"),
        &[HIDDEN],
        0.0,
    );

    // Post-FFN LayerNorm
    m.insert(
        format!("{LAYER_PREFIX}.full_layer_layer_norm.weight"),
        DynTensor::full(&[HIDDEN], 1.0, DType::F32, &cpu()).unwrap(),
    );
    z_fill(
        &mut m,
        &format!("{LAYER_PREFIX}.full_layer_layer_norm.bias"),
        &[HIDDEN],
        0.0,
    );

    // forward_core doesn't use embeddings, but PlBert::load requires them.
    // Use minimal dummy values.
    z_fill(
        &mut m,
        "embeddings.word_embeddings.weight",
        &[10, EMB_DIM],
        fill,
    );
    z_fill(
        &mut m,
        "embeddings.position_embeddings.weight",
        &[16, EMB_DIM],
        fill,
    );
    z_fill(
        &mut m,
        "embeddings.token_type_embeddings.weight",
        &[2, EMB_DIM],
        fill,
    );

    let mut config = PlbertConfig::default();
    config.vocab_size = 10;
    config.embedding_dim = EMB_DIM;
    config.hidden_size = HIDDEN;
    config.num_attention_heads = NUM_HEADS;
    config.intermediate_size = INTERMEDIATE;
    config.max_position_embeddings = 16;
    config.num_hidden_layers = num_hidden_layers;
    config.layer_norm_eps = 1e-12;

    (m, config)
}

// -- Trace + graph conversion -------------------------------------------------

/// Trace `PlBert::forward_core()` and convert to NY GraphNetwork.
///
/// Returns `(graph_network, input_shape)` for bounds propagation.
fn trace_plbert_core(num_layers: usize) -> (nn_verify::GraphNetwork, [usize; 3]) {
    let (weights, config) = plbert_weights(num_layers);
    let vb = VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let plbert = PlBert::load(&vb, &config).expect("PlBert::load");

    let input_shape = [BATCH, SEQ_LEN, EMB_DIM];
    let combined_emb = DynTensor::full(&input_shape, 0.1, DType::F32, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = combined_emb.clone();
        let id = record_input(&input_shape, DType::F32).unwrap();
        x.set_trace_id(id);
        plbert.forward_core(&x)
    })
    .expect("PlBert::forward_core trace");

    let node_count = graph.nodes().len();
    eprintln!("PlBert forward_core ({num_layers} layers): traced {node_count} nodes");

    let gn = trace_to_graph_model(&graph)
        .expect("trace_to_graph_model for PlBert forward_core")
        .graph;
    assert!(
        gn.num_nodes() > 0,
        "GraphNetwork must have nodes for PlBert"
    );
    eprintln!(
        "PlBert forward_core: GraphNetwork has {} nodes",
        gn.num_nodes()
    );

    (gn, input_shape)
}

// -- Test 1: Single shared layer, IBP -----------------------------------------

/// Trace PlBert forward_core with 1 shared layer and verify IBP bounds.
///
/// Exercises the full ALBERT layer: multi-head attention (Q/K/V + reshape +
/// transpose + variable-variable MatMul + Softmax + output projection) +
/// residual + LayerNorm + FFN (Linear + GELU + Linear) + residual + LayerNorm.
///
/// Part of #2402 AC1: PlBert attention layer verified via trace_graph.
#[test]
fn test_trace_plbert_forward_core_1layer_ibp() {
    let (gn, input_shape) = trace_plbert_core(1);

    let input_bounds = uniform_bounds(&input_shape, 0.5);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");

    // AC2: softmax + LayerNorm + GELU produce finite, non-NaN bounds.
    assert_all_finite(&output, "PlBert_1layer");
    assert_bounds_valid(&output);

    // Bounds should not be vacuously wide (1e6 threshold matches other Kokoro tests).
    assert_bounds_width(&output, 1e6, "PlBert_1layer");

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("PlBert 1-layer IBP: bounds=[{lo_min}, {hi_max}], width={width:.4}");
}

// -- Test 2: Two shared layers (ALBERT weight sharing), IBP -------------------

/// Trace PlBert forward_core with 2 shared layer iterations.
///
/// ALBERT reuses the same weights for each iteration. Two iterations exercises
/// the weight-sharing loop and verifies bounds don't explode through repeated
/// application of the same transformer block.
///
/// Part of #2402 AC1.
#[test]
fn test_trace_plbert_forward_core_2layer_ibp() {
    let (gn, input_shape) = trace_plbert_core(2);

    let input_bounds = uniform_bounds(&input_shape, 0.5);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");

    assert_all_finite(&output, "PlBert_2layer");
    assert_bounds_valid(&output);
    assert_bounds_width(&output, 1e6, "PlBert_2layer");

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("PlBert 2-layer IBP: bounds=[{lo_min}, {hi_max}], width={width:.4}");
}

// -- Test 3: CROWN propagation -----------------------------------------------

/// Verify PlBert with CROWN propagation (tighter than IBP).
///
/// CROWN linearizes through the network layers for tighter bounds.
/// Uses `assert_crown_tighter_when_not_fallback` which handles CROWN
/// fallback to IBP gracefully (e.g., if LayerNorm forces fallback).
///
/// Part of #2402 AC2: PlBert softmax + LayerNorm + GELU verified.
#[test]
fn test_trace_plbert_forward_core_crown() {
    let (gn, input_shape) = trace_plbert_core(1);

    let input_bounds = uniform_bounds(&input_shape, 0.5);

    let (method, crown_output, fallback_reason) =
        super::common::assert_crown_tighter_when_not_fallback(&gn, &input_bounds);

    let (lo_min, hi_max) = bounds_min_max(&crown_output);
    let width = hi_max - lo_min;
    eprintln!(
        "PlBert CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}], width={width:.4}, \
         fallback={:?}",
        fallback_reason.as_deref().unwrap_or("none")
    );
}

// -- Test 4: NaN/overflow safety with extreme inputs --------------------------

/// Verify PlBert produces finite bounds even with wide input ranges.
///
/// Softmax is the primary overflow risk: if attention scores are very large,
/// exp() overflows. LayerNorm has a division-by-zero risk if variance is zero.
/// GELU uses erf() which is bounded but has numerical edge cases.
///
/// Uses +-2.0 input range (wider than typical embeddings) to stress-test.
///
/// Part of #2402 AC2: PlBert softmax + LayerNorm + GELU NaN/overflow safety.
#[test]
fn test_trace_plbert_nan_overflow_safety() {
    let (gn, input_shape) = trace_plbert_core(1);

    // Wide input range to stress-test numerical safety.
    let input_bounds = uniform_bounds(&input_shape, 2.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");

    // Core safety property: all bounds must be finite (no NaN, no Inf).
    let (lo, hi) = output.lower_upper();
    for (idx, (&lo_val, &hi_val)) in lo.iter().zip(hi.iter()).enumerate() {
        assert!(
            lo_val.is_finite(),
            "NaN/overflow safety: PlBert lower at {idx} not finite: {lo_val}"
        );
        assert!(
            hi_val.is_finite(),
            "NaN/overflow safety: PlBert upper at {idx} not finite: {hi_val}"
        );
        assert!(
            lo_val <= hi_val,
            "NaN/overflow safety: bounds inverted at {idx}: lo={lo_val} > hi={hi_val}"
        );
    }

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PlBert NaN safety (input +-2.0): bounds=[{lo_min}, {hi_max}]");
}
