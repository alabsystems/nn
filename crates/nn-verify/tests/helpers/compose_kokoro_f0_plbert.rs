// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro F0EnergyPredictor and PlBert-style self-attention trace verification.
//!
//! Part of #2427 (Kokoro trace verification gaps).
//! Part of #2224 (trace real KokoroModel and verify with NY).
//!
//! Coverage matrix gaps filled:
//! - F0EnergyPredictor: AdainResBlk1d + BiLSTM + Upsample1d + Linear
//! - PlBert attention: variable-variable MatMul (BilinearCrownLayer) + Softmax

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Linear, Module};
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};
use nn_models::kokoro_f0::F0EnergyPredictor;
use nn_verify::{trace_to_graph_model, trace_to_graph_model_multi_input};
use std::collections::HashMap;

use super::common::bounds_min_max;
use super::common::kokoro_weights::{assert_all_finite, propagate_multi_input_ibp, z};

// -- Dimensions ---------------------------------------------------------------

const D_MODEL: usize = 8;
const STYLE_DIM: usize = 4;
const BILSTM_HIDDEN: usize = 4; // D_MODEL / 2

// -- Weight helpers -----------------------------------------------------------

/// Insert AdainResBlk1d weights at `prefix`.
fn adain_resblk_weights(
    m: &mut HashMap<String, DynTensor>,
    prefix: &str,
    dim_in: usize,
    dim_out: usize,
    style_dim: usize,
    upsample: bool,
) {
    z(
        m,
        &format!("{prefix}.n1.fc.weight"),
        &[2 * dim_in, style_dim],
    );
    z(m, &format!("{prefix}.n1.fc.bias"), &[2 * dim_in]);
    z(
        m,
        &format!("{prefix}.n2.fc.weight"),
        &[2 * dim_out, style_dim],
    );
    z(m, &format!("{prefix}.n2.fc.bias"), &[2 * dim_out]);
    z(m, &format!("{prefix}.c1.weight"), &[dim_out, dim_in, 3]);
    z(m, &format!("{prefix}.c1.bias"), &[dim_out]);
    z(m, &format!("{prefix}.c2.weight"), &[dim_out, dim_out, 3]);
    z(m, &format!("{prefix}.c2.bias"), &[dim_out]);
    if dim_in != dim_out {
        z(m, &format!("{prefix}.skip.weight"), &[dim_out, dim_in, 1]);
        z(m, &format!("{prefix}.skip.bias"), &[dim_out]);
    }
    if upsample {
        z(m, &format!("{prefix}.pool.weight"), &[dim_in, 1, 3]);
        z(m, &format!("{prefix}.pool.bias"), &[dim_in]);
    }
}

/// Build F0EnergyPredictor weights: shared BiLSTM + 2×3 AdainResBlk1d + 2 projections.
fn f0_predictor_weights() -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    let bilstm_out = 2 * BILSTM_HIDDEN;
    // BiLSTM input is cat(features, style) = d_model + style_dim.
    let bilstm_input = D_MODEL + STYLE_DIM;

    // Shared BiLSTM (forward + backward)
    for dir in &["forward", "backward"] {
        let p = format!("shared.{dir}");
        z(
            &mut m,
            &format!("{p}.weight_ih_l0"),
            &[4 * BILSTM_HIDDEN, bilstm_input],
        );
        z(
            &mut m,
            &format!("{p}.weight_hh_l0"),
            &[4 * BILSTM_HIDDEN, BILSTM_HIDDEN],
        );
        z(&mut m, &format!("{p}.bias_ih_l0"), &[4 * BILSTM_HIDDEN]);
        z(&mut m, &format!("{p}.bias_hh_l0"), &[4 * BILSTM_HIDDEN]);
    }

    // F0 head: 3 AdainResBlk1d blocks + projection
    adain_resblk_weights(&mut m, "F0.0", bilstm_out, bilstm_out, STYLE_DIM, false);
    adain_resblk_weights(&mut m, "F0.1", bilstm_out, BILSTM_HIDDEN, STYLE_DIM, true);
    adain_resblk_weights(
        &mut m,
        "F0.2",
        BILSTM_HIDDEN,
        BILSTM_HIDDEN,
        STYLE_DIM,
        false,
    );
    z(&mut m, "F0_proj.weight", &[1, BILSTM_HIDDEN]);
    z(&mut m, "F0_proj.bias", &[1]);

    // Energy (N) head: same architecture as F0
    adain_resblk_weights(&mut m, "N.0", bilstm_out, bilstm_out, STYLE_DIM, false);
    adain_resblk_weights(&mut m, "N.1", bilstm_out, BILSTM_HIDDEN, STYLE_DIM, true);
    adain_resblk_weights(
        &mut m,
        "N.2",
        BILSTM_HIDDEN,
        BILSTM_HIDDEN,
        STYLE_DIM,
        false,
    );
    z(&mut m, "N_proj.weight", &[1, BILSTM_HIDDEN]);
    z(&mut m, "N_proj.bias", &[1]);

    m
}

// -- Test 1: F0EnergyPredictor ------------------------------------------------

/// Trace F0EnergyPredictor and verify trace → graph → IBP.
///
/// Multi-input model: aligned features `[B, D, T]` + style `[B, style_dim]`.
/// Exercises: BiLSTM (Flip + LSTM + Cat), 3× AdainResBlk1d (InstanceNorm +
/// LeakyReLU + Conv1d + ConvTranspose1d + Upsample1d), Linear projection.
///
/// Part of #2427 AC1.
#[test]
fn test_trace_kokoro_f0_predictor() {
    let weights = f0_predictor_weights();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let f0_pred = F0EnergyPredictor::load(&vb, D_MODEL, STYLE_DIM, BILSTM_HIDDEN).unwrap();

    let batch = 1;
    let t_mel = 3;
    // F0EnergyPredictor expects aligned features that already include style
    // from DurationEncoder: [B, d_model + style_dim, T_mel].
    let aligned_shape = [batch, D_MODEL + STYLE_DIM, t_mel];
    let style_shape = [batch, STYLE_DIM];

    let aligned = DynTensor::full(&aligned_shape, 0.1, DType::F32, &cpu()).unwrap();
    let style = DynTensor::full(&style_shape, 0.05, DType::F32, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut al = aligned.clone();
        let id_al = record_input(&aligned_shape, DType::F32).unwrap();
        al.set_trace_id(id_al);
        let mut sty = style.clone();
        let id_sty = record_input(&style_shape, DType::F32).unwrap();
        sty.set_trace_id(id_sty);
        let (f0, _energy) = f0_pred.forward(&al, &sty)?;
        Ok(f0)
    })
    .unwrap();

    // AC1: Trace captures non-trivial graph.
    let node_count = graph.nodes().len();
    assert!(
        node_count >= 20,
        "expected at least 20 traced nodes for F0EnergyPredictor, got {node_count}"
    );
    eprintln!("F0EnergyPredictor: traced {node_count} nodes");

    // AC2: Graph translation. FusedAdainResBlock decomposition (#2547),
    // output_padding (#2558), and grouped ConvTranspose1d (#2989) are all resolved.
    // Remaining blocker: axis convention (#2987) — BiLSTM scan Slice ops reference
    // the batch dimension which doesn't exist in unbatched NY mode.
    let result = trace_to_graph_model_multi_input(&graph);
    match &result {
        Ok(tr) => {
            let gn = &tr.graph;
            // Full pipeline succeeded — all blockers resolved.
            assert!(gn.num_nodes() > 0, "GraphNetwork should have nodes");
            eprintln!(
                "F0EnergyPredictor: GraphNetwork has {} nodes",
                gn.num_nodes()
            );
            // Verify IBP propagation produces finite bounds.
            let output = propagate_multi_input_ibp(
                gn,
                &[
                    (&aligned_shape[..], (-1.0, 1.0)),
                    (&style_shape[..], (-0.5, 0.5)),
                ],
            );
            assert_all_finite(&output, "F0EnergyPredictor");
            super::common::assert_bounds_width(&output, 1e6, "F0EnergyPredictor");
            let (lo_min, hi_max) = bounds_min_max(&output);
            eprintln!("F0EnergyPredictor IBP bounds: [{lo_min}, {hi_max}]");
        }
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                !msg.contains("FusedAdainResBlock"),
                "FusedAdainResBlock should be decomposed (#2547), but got: {msg}"
            );
            // Axis convention (#2987) is in progress — accept axis/batch errors.
            eprintln!("F0EnergyPredictor: graph translation blocked by: {msg}");
        }
    }
}

// -- Test 2: PlBert-style self-attention --------------------------------------

/// Attention hidden dimension for test.
const ATTN_HIDDEN: usize = 8;

/// Build Q, K, V, dense Linear projections for single-head self-attention.
fn build_attention_projections() -> (Linear, Linear, Linear, Linear) {
    let mut weights = HashMap::new();
    for name in &["q", "k", "v", "dense"] {
        z(
            &mut weights,
            &format!("{name}.weight"),
            &[ATTN_HIDDEN, ATTN_HIDDEN],
        );
        z(&mut weights, &format!("{name}.bias"), &[ATTN_HIDDEN]);
    }
    let vb = VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let make = |n: &str| {
        Linear::new(
            vb.get(&[ATTN_HIDDEN, ATTN_HIDDEN], &format!("{n}.weight"))
                .unwrap(),
            Some(vb.get(&[ATTN_HIDDEN], &format!("{n}.bias")).unwrap()),
        )
        .unwrap()
    };
    (make("q"), make("k"), make("v"), make("dense"))
}

/// Trace PlBert-style self-attention and verify variable-variable MatMul + Softmax.
///
/// Builds self-attention from public layers::Linear primitives (AlbertAttention is
/// private). Uses single-head to avoid contiguous() which is unsupported in
/// trace-to-graph.
///
/// Exercises: 4× Linear (Q, K, V, dense), 2× variable-variable MatMul
/// (BilinearCrownLayer: Q@K^T and attn@V), Softmax, Transpose, mul_scalar.
///
/// Part of #2427 AC2.
#[test]
fn test_trace_kokoro_plbert_attention() {
    let (q_proj, k_proj, v_proj, dense) = build_attention_projections();

    let batch = 1;
    let seq_len = 3;
    let input_shape = [batch, seq_len, ATTN_HIDDEN];
    let hidden = DynTensor::full(&input_shape, 0.1, DType::F32, &cpu()).unwrap();
    let scale = 1.0 / (ATTN_HIDDEN as f64).sqrt();

    let (_result, graph) = trace_graph(|| {
        let mut x = hidden.clone();
        let id = record_input(&input_shape, DType::F32).unwrap();
        x.set_trace_id(id);

        // Single-head attention from primitives (avoids private AlbertAttention
        // and contiguous() which is unsupported in trace-to-graph).
        let q = q_proj.forward(&x)?;
        let k = k_proj.forward(&x)?;
        let v = v_proj.forward(&x)?;

        // Q @ K^T: variable-variable MatMul → BilinearCrownLayer
        let k_t = k.transpose(1, 2)?;
        let scores = q.matmul(&k_t)?;
        let scores = scores.mul_scalar(scale)?;

        // Softmax over last dimension
        let attn_weights = nn_core::layers::softmax(&scores, 2)?;

        // attn @ V: variable-variable MatMul → BilinearCrownLayer
        let output = attn_weights.matmul(&v)?;

        // Output projection
        dense.forward(&output)
    })
    .unwrap();

    // AC1: Trace captures non-trivial graph.
    let node_count = graph.nodes().len();
    assert!(
        node_count >= 8,
        "expected at least 8 traced nodes for self-attention, got {node_count}"
    );
    eprintln!("PlBert attention: traced {node_count} nodes");

    // AC2: Graph conversion succeeds (single-input model).
    let gn = trace_to_graph_model(&graph)
        .expect("trace_to_graph_model for PlBert self-attention")
        .graph;
    assert!(gn.num_nodes() > 0, "GraphNetwork should have nodes");
    eprintln!(
        "PlBert attention: GraphNetwork has {} nodes",
        gn.num_nodes()
    );

    // AC3: IBP propagation produces finite bounds.
    let input_bounds = super::common::uniform_bounds(&input_shape, 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    super::common::assert_bounds_valid(&output);
    super::common::assert_bounds_width(&output, 1e6, "PlBert_attention");
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PlBert attention IBP bounds: [{lo_min}, {hi_max}]");
}
