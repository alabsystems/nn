// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

//! Per-layer decomposition for Kokoro ProsodyPredictor attention score graph.
//!
//! Decomposes the monolithic attention score graph from `kokoro_attn_scaled.rs`
//! into individual layers, each with its own `TensorKernelDef`. This enables
//! `verify_layerwise` (#1762) to apply per-layer CROWN propagation.
//!
//! **Phase 40 result (negative):** Per-layer CROWN produces *wider* bounds than
//! monolithic IBP due to junction bound widening — cross-layer correlations are
//! lost at stage boundaries. See `compose_attention_monotonicity_phase40.rs`.
//!
//! Architecture decomposition:
//! ```text
//!   Layer 0: Conv1d    — [seq_len, d_model] → transpose → Conv1d → transpose → [seq_len, d_model]
//!   Layer 1: LayerNorm — [seq_len, d_model] → LayerNorm → [seq_len, d_model]
//!   Layer 2: AdaLN     — [seq_len, d_model] × style → (1+gamma)*x + beta → [seq_len, d_model]
//!   Layer 3: Gate+Proj — [seq_len, d_model] → sigmoid*tanh gating → linear proj → [seq_len, d_model]
//!   Layer 4: Residual+Attn — [seq_len, d_model] + input → Q+PE → Q@K^T/√d → [seq_len, seq_len]
//! ```
//!
//! Note: Layer 4 requires the *original input* for the residual connection.
//! Since verify_layerwise chains output→input, we handle the residual by
//! making the gating layer output a concatenated [gated; original_input] tensor,
//! and having the residual+attention layer split it back.
//!
//! **Simpler alternative (used here):** Decompose into 3 macro-stages rather
//! than 5 micro-layers. This keeps the residual connection within a single stage:
//!
//! ```text
//!   Stage 0: Encoder — Conv1d → LayerNorm → AdaLN → [seq_len, d_model]
//!   Stage 1: Gate+Proj+Residual — sigmoid*tanh → proj → residual(+input) → [seq_len, d_model]
//!   Stage 2: AttentionScores — (input+PE) @ K^T / √d → [seq_len, seq_len]
//! ```
//!
//! The residual in Stage 1 adds the *input to Stage 1* (which equals the output
//! of Stage 0, verified at the junction). This is correct because the junction
//! verification ensures Stage 0's output bounds contain Stage 1's input.
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 40.

use super::helpers::KokoroDims;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

/// A single layer/stage in the decomposed attention pipeline.
pub(super) type AttnLayerSpec = (TensorKernelDef, Vec<TensorParamBinding>);

/// Style embedding dimension (scaled proportionally).
fn style_dim(dims: &KokoroDims) -> usize {
    (dims.d_model / 2).clamp(4, 128)
}

/// Hidden dimension for LSTM-like gate (d_model / 2).
fn hidden_dim(dims: &KokoroDims) -> usize {
    dims.d_model / 2
}

const CONV_KERNEL: usize = 3;
const CONV_PADDING: usize = 1;

/// Build sinusoidal positional encoding matrix [seq_len, d_model].
fn build_sinusoidal_pe(seq_len: usize, d_model: usize) -> ArrayD<f32> {
    let mut data = vec![0.0f32; seq_len * d_model];
    for t in 0..seq_len {
        for i in 0..d_model / 2 {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * i as f64 / d_model as f64);
            data[t * d_model + 2 * i] = freq.sin() as f32;
            data[t * d_model + 2 * i + 1] = freq.cos() as f32;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[seq_len, d_model]), data).expect("valid PE shape")
}

// Weight constructors — delegated to super::common::weights (Part of #1938).
use super::common::weights;

fn build_encoder_weight(rows: usize, cols: usize, scale: f32) -> ArrayD<f32> {
    weights::encoder_weight(rows, cols, scale)
}

fn build_conv_weight(out_ch: usize, in_ch: usize, kernel: usize, scale: f32) -> ArrayD<f32> {
    weights::conv_weight(out_ch, in_ch, kernel, scale)
}

/// Style embedding expanded to [seq_len, style_dim].
fn build_style_expanded(seq_len: usize, sdim: usize, magnitude: f32) -> ArrayD<f32> {
    let row: Vec<f32> = (0..sdim)
        .map(|i| magnitude * (1.0 + 0.1 * i as f32))
        .collect();
    let data: Vec<f32> = row.iter().cycle().take(seq_len * sdim).copied().collect();
    ArrayD::from_shape_vec(IxDyn(&[seq_len, sdim]), data).expect("valid style shape")
}

/// Style projection weight [style_dim, 2*d_model].
fn build_style_proj_weight(in_dim: usize, out_dim: usize, scale: f32) -> ArrayD<f32> {
    let total = in_dim * out_dim;
    let mut data = vec![0.0f32; total];
    for i in 0..in_dim {
        for j in 0..out_dim {
            data[i * out_dim + j] = scale * 0.01 * ((i * j % 7) as f32 + 0.1);
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[in_dim, out_dim]), data).expect("valid style proj shape")
}

/// Pre-compute style bias: style [T, S] @ W_s [S, H] → [T, H].
fn precompute_style_bias(style: &ArrayD<f32>, w_s: &ArrayD<f32>) -> ArrayD<f32> {
    let s = style
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("style 2D");
    let w = w_s
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("w_s 2D");
    s.dot(&w).into_dyn()
}

// ---------------------------------------------------------------------------
// Stage 0: Encoder (Conv1d + LayerNorm + AdaLayerNorm)
// ---------------------------------------------------------------------------

/// Build encoder stage: Conv1d → LayerNorm → AdaLayerNorm(style).
///
/// Input: [seq_len, d_model] (Variable) → Output: [seq_len, d_model]
///
/// This is a "micro-stage" grouping 3 adjacent operations for tighter CROWN
/// bounds than verifying each individually (design doc D1 mitigation).
pub(super) fn build_stage_encoder(dims: &KokoroDims, enc_scale: f32) -> AttnLayerSpec {
    let d = dims.d_model;
    let t = dims.seq_len;
    let s = style_dim(dims);

    let mut b = TensorBlockBuilder::new("stage_encoder");

    // Inputs
    let raw_input = b.add_input("raw_input", &[t, d]); // 0: Variable
    let style_exp = b.add_input("style_exp", &[t, s]); // 1
    let conv_w = b.add_input("conv_w", &[d, d, CONV_KERNEL]); // 2
    let eps = b.add_input("eps", &[1]); // 3
    let ln_w = b.add_input("ln_w", &[d]); // 4
    let ln_b = b.add_input("ln_b", &[d]); // 5
    let style_proj_w = b.add_input("style_proj_w", &[s, 2 * d]); // 6
    let ones = b.add_input("ones", &[t, d]); // 7

    // Conv1d: channels-first processing
    let input_cf = b.add_transpose(raw_input, &[1, 0], &[d, t]);
    let conv_out = b.add_conv1d(input_cf, conv_w, None, 1, CONV_PADDING, &[d, t]);
    let conv_back = b.add_transpose(conv_out, &[1, 0], &[t, d]);

    // LayerNorm
    let normed = b.add_layer_norm(conv_back, eps, 1, ln_w, ln_b, &[t, d]);

    // AdaLayerNorm: style → gamma, beta
    let style_proj = b.add_matmul(style_exp, style_proj_w, false, None, &[t, 2 * d]);
    let gamma = b.add_narrow(style_proj, 1, 0, d, &[t, d]);
    let beta = b.add_narrow(style_proj, 1, d, d, &[t, d]);
    let scale = b.add_binary_add(ones, gamma, &[t, d]);
    let scaled = b.add_binary_mul(normed, scale, &[t, d]);
    let output = b.add_binary_add(scaled, beta, &[t, d]);

    let def = b.build(output).expect("valid encoder stage graph");

    let style_exp_data = build_style_expanded(t, s, 0.5);
    let conv_w_data = build_conv_weight(d, d, CONV_KERNEL, enc_scale);
    let style_proj_w_data = build_style_proj_weight(s, 2 * d, enc_scale);

    let bindings = vec![
        TensorParamBinding::Variable,                       // raw_input
        TensorParamBinding::ConstantTensor(style_exp_data), // style_exp
        TensorParamBinding::ConstantTensor(conv_w_data),    // conv_w
        TensorParamBinding::ConstantScalar(1e-5),           // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 1.0f32)), // ln_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 0.0f32)), // ln_b
        TensorParamBinding::ConstantTensor(style_proj_w_data), // style_proj_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[t, d]), 1.0f32)), // ones
    ];
    (def, bindings)
}

// ---------------------------------------------------------------------------
// Stage 1: Gate + Projection + Residual
// ---------------------------------------------------------------------------

/// Build gating + projection + residual stage.
///
/// Input: [seq_len, d_model] (Variable = AdaLN output, which also serves
/// as the "original input" for the residual connection in the ProsodyPredictor
/// architecture — the residual adds the *AdaLN output* before gating, not
/// the raw text features).
///
/// Actually, in the real architecture the residual adds the *raw input* before
/// the conv block. For per-layer decomposition, we model the residual as adding
/// the *stage input* (output of Stage 0), which is tight because junction
/// containment ensures the stage input is within verified bounds.
///
/// Output: [seq_len, d_model]
pub(super) fn build_stage_gate_proj_residual(dims: &KokoroDims, enc_scale: f32) -> AttnLayerSpec {
    let d = dims.d_model;
    let t = dims.seq_len;
    let s = style_dim(dims);
    let h = hidden_dim(dims);

    let mut b = TensorBlockBuilder::new("stage_gate_proj_residual");

    // Input (AdaLN output / stage input)
    let ada_out = b.add_input("ada_output", &[t, d]); // 0: Variable
    let gate_wx = b.add_input("gate_wx", &[d, h]); // 1
    let gate_bias = b.add_input("gate_bias", &[t, h]); // 2
    let val_wx = b.add_input("val_wx", &[d, h]); // 3
    let val_bias = b.add_input("val_bias", &[t, h]); // 4
    let out_proj_w = b.add_input("out_proj_w", &[h, d]); // 5

    // LSTM-like gating
    let gate_raw_x = b.add_matmul(ada_out, gate_wx, false, None, &[t, h]);
    let gate_raw = b.add_binary_add(gate_raw_x, gate_bias, &[t, h]);
    let gate = b.add_sigmoid(gate_raw, &[t, h]);

    let val_raw_x = b.add_matmul(ada_out, val_wx, false, None, &[t, h]);
    let val_raw = b.add_binary_add(val_raw_x, val_bias, &[t, h]);
    let val = b.add_tanh(val_raw, &[t, h]);

    let gated = b.add_binary_mul(gate, val, &[t, h]);

    // Output projection + residual
    let projected = b.add_matmul(gated, out_proj_w, false, None, &[t, d]);
    let output = b.add_binary_add(ada_out, projected, &[t, d]);

    let def = b
        .build(output)
        .expect("valid gate+proj+residual stage graph");

    // Build weight bindings
    let full_gate_w = build_encoder_weight(d + s, h, enc_scale * 0.5);
    let full_val_w = build_encoder_weight(d + s, h, enc_scale * 0.5);
    let gate_wx_data = full_gate_w
        .slice(ndarray::s![0..d, ..])
        .to_owned()
        .into_dyn();
    let val_wx_data = full_val_w
        .slice(ndarray::s![0..d, ..])
        .to_owned()
        .into_dyn();
    let gate_ws = full_gate_w
        .slice(ndarray::s![d.., ..])
        .to_owned()
        .into_dyn();
    let val_ws = full_val_w.slice(ndarray::s![d.., ..]).to_owned().into_dyn();

    let style_exp = build_style_expanded(t, s, 0.5);
    let gate_bias_data = precompute_style_bias(&style_exp, &gate_ws);
    let val_bias_data = precompute_style_bias(&style_exp, &val_ws);
    let out_proj_w_data = build_encoder_weight(h, d, enc_scale);

    let bindings = vec![
        TensorParamBinding::Variable,                        // ada_output
        TensorParamBinding::ConstantTensor(gate_wx_data),    // gate_wx
        TensorParamBinding::ConstantTensor(gate_bias_data),  // gate_bias
        TensorParamBinding::ConstantTensor(val_wx_data),     // val_wx
        TensorParamBinding::ConstantTensor(val_bias_data),   // val_bias
        TensorParamBinding::ConstantTensor(out_proj_w_data), // out_proj_w
    ];
    (def, bindings)
}

// ---------------------------------------------------------------------------
// Stage 2: Attention Scores
// ---------------------------------------------------------------------------

/// Build attention score computation stage.
///
/// Input: [seq_len, d_model] (Variable = residual output)
/// Output: [seq_len, seq_len] (pre-softmax attention scores)
///
/// Architecture: Q = input + PE, K = PE, scores = Q @ K^T / √d_model
pub(super) fn build_stage_attention_scores(dims: &KokoroDims, pe_scale: f32) -> AttnLayerSpec {
    let d = dims.d_model;
    let t = dims.seq_len;

    let mut b = TensorBlockBuilder::new("stage_attention_scores");

    // Input (residual output)
    let residual_out = b.add_input("residual_output", &[t, d]); // 0: Variable
    let pe = b.add_input("pe", &[t, d]); // 1
    let k = b.add_input("key", &[t, d]); // 2

    // Q = residual_output + PE
    let q = b.add_binary_add(residual_out, pe, &[t, d]);

    // scores = Q @ K^T / √d_model
    let att_scale = 1.0 / (d as f32).sqrt();
    let scores = b.add_matmul(q, k, true, Some(att_scale), &[t, t]);

    let def = b.build(scores).expect("valid attention scores stage graph");

    let mut pe_data = build_sinusoidal_pe(t, d);
    pe_data.mapv_inplace(|v| v * pe_scale);

    let bindings = vec![
        TensorParamBinding::Variable,                        // residual_output
        TensorParamBinding::ConstantTensor(pe_data.clone()), // pe
        TensorParamBinding::ConstantTensor(pe_data), // key (= PE for identity-like structure)
    ];
    (def, bindings)
}

// ---------------------------------------------------------------------------
// Full layerwise decomposition
// ---------------------------------------------------------------------------

/// Build the ProsodyPredictor attention pipeline as 3 stages for verify_layerwise.
///
/// Returns: [encoder_stage, gate_proj_residual_stage, attention_scores_stage]
pub(super) fn build_attn_layerwise(
    dims: &KokoroDims,
    enc_scale: f32,
    pe_scale: f32,
) -> Vec<AttnLayerSpec> {
    vec![
        build_stage_encoder(dims, enc_scale),
        build_stage_gate_proj_residual(dims, enc_scale),
        build_stage_attention_scores(dims, pe_scale),
    ]
}

/// Evaluate attention monotonicity margin using per-layer CROWN composition.
///
/// Returns `(min_margin, row_margins, propagation_methods, all_finite)`.
pub(super) fn evaluate_layerwise_margin(
    dims: &KokoroDims,
    input_bound: f32,
    enc_scale: f32,
    pe_scale: f32,
) -> (f64, Vec<f64>, Vec<String>, bool) {
    use nn_tts_verify::monotonicity::interpret_attention_monotonicity;

    let stages = build_attn_layerwise(dims, enc_scale, pe_scale);

    let initial_bounds = super::common::uniform_bounds(&[dims.seq_len, dims.d_model], input_bound);

    // Propagate through each stage, collecting per-stage methods and bounds
    let mut current_bounds = initial_bounds;
    let mut methods = Vec::with_capacity(stages.len());

    for (i, (def, bindings)) in stages.iter().enumerate() {
        let graph = nn_verify::tensor_kernel_to_graph(def, bindings)
            .unwrap_or_else(|e| panic!("stage {i} graph build failed: {e}"));

        let (method, output_bounds, _) =
            nn_verify::propagate_with_crown_fallback(&graph, &current_bounds)
                .unwrap_or_else(|e| panic!("stage {i} propagation failed: {e}"));

        let method_str = match method {
            nn_verify::PropMethod::Crown => "CROWN",
            nn_verify::PropMethod::Ibp => "IBP",
            _ => "unknown",
        };
        methods.push(method_str.to_string());

        eprintln!(
            "  Stage {i}: method={method_str}, output_shape={:?}",
            output_bounds.shape()
        );

        current_bounds = output_bounds;
    }

    let (lo, hi) = current_bounds.lower_upper();
    let all_finite = lo.iter().chain(hi.iter()).all(|v| v.is_finite());

    let lo_slice = lo.as_slice().expect("contiguous");
    let hi_slice = hi.as_slice().expect("contiguous");

    // Final output is [seq_len, seq_len] attention scores
    let last_method = methods.last().map(String::as_str).unwrap_or("unknown");
    let cert = interpret_attention_monotonicity(
        lo_slice,
        hi_slice,
        dims.seq_len,
        dims.seq_len,
        f64::from(input_bound),
        last_method,
    )
    .unwrap();

    (cert.min_margin, cert.row_margins, methods, all_finite)
}
