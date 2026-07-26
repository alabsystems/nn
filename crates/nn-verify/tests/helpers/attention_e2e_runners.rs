// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, unreachable_pub, clippy::duplicated_attributes)]

//! Monolithic graph builders and pipeline runners for Phase 15–16 attention
//! end-to-end and certificate tests.
//!
//! Extracted from two files per #1978.

use super::common;
use super::lw_builders;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};

// ===========================================================================
// Monolithic graph builders
// ===========================================================================

/// Build a monolithic attention graph containing all 4 layers:
///
///   Layer 1: Q_proj = Q @ W_q → [T, d_k]       (linear projection)
///   Layer 2: Scores = Q_proj @ K^T / √d_k → [T, T]  (score computation)
///   Layer 3: Weights = Softmax(Scores) → [T, T]      (attention weights)
///   Layer 4: Output = Weights @ V → [T, d_k]         (output projection)
///
/// All layers are in a SINGLE graph — NY sees the full computation
/// path and can propagate CROWN bounds end-to-end.
pub fn build_monolithic_attention(
    name: &str,
    seq_len: usize,
    d_model: usize,
    d_k: usize,
) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);

    let q = b.add_input("query", &[seq_len, d_model]);
    let w_q = b.add_input("w_q", &[d_model, d_k]);
    let k = b.add_input("key", &[seq_len, d_k]);
    let v = b.add_input("value", &[seq_len, d_k]);

    let q_proj = b.add_matmul(q, w_q, false, None, &[seq_len, d_k]);
    let scale = 1.0 / (d_k as f32).sqrt();
    let scores = b.add_matmul(q_proj, k, true, Some(scale), &[seq_len, seq_len]);
    let weights = b.add_softmax(scores, -1, &[seq_len, seq_len]);
    let output = b.add_matmul(weights, v, false, None, &[seq_len, d_k]);

    b.build(output).expect("valid monolithic attention graph")
}

/// Build a monolithic 3-layer attention (no projection):
///
///   Layer 1: Scores = Q @ K^T / √d → [T, T]
///   Layer 2: Weights = Softmax(Scores) → [T, T]
///   Layer 3: Output = Weights @ V → [T, d]
pub fn build_monolithic_attention_no_proj(name: &str, seq_len: usize, d: usize) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);

    let q = b.add_input("query", &[seq_len, d]);
    let k = b.add_input("key", &[seq_len, d]);
    let v = b.add_input("value", &[seq_len, d]);

    let scale = 1.0 / (d as f32).sqrt();
    let scores = b.add_matmul(q, k, true, Some(scale), &[seq_len, seq_len]);
    let weights = b.add_softmax(scores, -1, &[seq_len, seq_len]);
    let output = b.add_matmul(weights, v, false, None, &[seq_len, d]);

    b.build(output)
        .expect("valid monolithic attention (no proj)")
}

/// Build a simplified prosody score graph for certificate generation.
///
/// Architecture: hidden (Variable) + PE → attention scores.
pub fn build_prosody_score_graph(
    name: &str,
    seq_len: usize,
    d: usize,
) -> (TensorKernelDef, Vec<usize>) {
    let mut b = TensorBlockBuilder::new(name);

    let hidden = b.add_input("hidden", &[seq_len, d]);
    let pe = b.add_input("pe", &[seq_len, d]);
    let k = b.add_input("key", &[seq_len, d]);

    let q = b.add_binary_add(hidden, pe, &[seq_len, d]);
    let scale = 1.0 / (d as f32).sqrt();
    let scores_shape = [seq_len, seq_len];
    let scores = b.add_matmul(q, k, true, Some(scale), &scores_shape);

    let def = b.build(scores).expect("valid prosody score graph");
    (def, scores_shape.to_vec())
}

// ===========================================================================
// Pipeline runners
// ===========================================================================

/// Run 3-layer layerwise pipeline and return output bounds.
///
/// Uses shared layer builders from `lw_builders`.
pub fn run_layerwise_3layer(
    seq_len: usize,
    d: usize,
    input_bound: f32,
    k_scale: f32,
    v_scale: f32,
) -> BoundedTensor {
    // Layer 1: Score
    let score_def = lw_builders::build_score_layer(&format!("lw15_score_{d}"), seq_len, d);
    let k_tensor = lw_builders::build_k_identity(seq_len, d, k_scale);
    let score_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(k_tensor),
    ];
    let score_graph = tensor_kernel_to_graph(&score_def, &score_bindings).expect("score graph");
    let q_bounds = common::uniform_bounds(&[seq_len, d], input_bound);
    let (_, score_out, _) = nn_verify::propagate_with_crown_fallback(&score_graph, &q_bounds)
        .expect("score propagation");

    // Layer 2: Softmax
    let sm_def = lw_builders::build_softmax_layer(&format!("lw15_sm_{d}"), seq_len);
    let sm_bindings = vec![TensorParamBinding::Variable];
    let sm_graph = tensor_kernel_to_graph(&sm_def, &sm_bindings).expect("softmax graph");
    let (_, sm_out, _) = nn_verify::propagate_with_crown_fallback(&sm_graph, &score_out)
        .expect("softmax propagation");

    // Layer 3: Output
    let out_def = lw_builders::build_output_layer(&format!("lw15_out_{d}"), seq_len, d);
    let v_tensor = lw_builders::build_v_tensor(seq_len, d, v_scale);
    let out_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(v_tensor),
    ];
    let out_graph = tensor_kernel_to_graph(&out_def, &out_bindings).expect("output graph");
    let (_, out_out, _) =
        nn_verify::propagate_with_crown_fallback(&out_graph, &sm_out).expect("output propagation");

    out_out
}
