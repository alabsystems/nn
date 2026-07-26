// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

//! Parameterized builder for scaled Kokoro ProsodyPredictor attention score graphs.
//!
//! Extends the D=8 hardcoded builders in `phase8_builders.rs` to accept
//! `KokoroDims` parameters, enabling monotonicity certificate analysis at
//! any model dimension from D=8 to D=512 (production).
//!
//! The graph computes pre-softmax attention scores:
//! ```text
//!   raw_input [seq_len, d_model] (Variable)
//!   → Conv1d → LayerNorm → AdaLayerNorm(style) → gate(sigmoid*tanh) → proj → residual
//!   → Q = residual + PE, K = PE
//!   → scores = Q @ K^T / √d_model → [seq_len, seq_len]
//! ```
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 39.

use super::helpers::KokoroDims;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

/// Style embedding dimension (scaled proportionally: production Kokoro uses 256).
fn style_dim(dims: &KokoroDims) -> usize {
    // Scale style_dim with d_model: min 4, max d_model/2
    (dims.d_model / 2).clamp(4, 128)
}

/// Hidden dimension for LSTM-like gate (d_model / 2).
fn hidden_dim(dims: &KokoroDims) -> usize {
    dims.d_model / 2
}

/// Conv1d kernel size (matches Kokoro ProsodyPredictor: kernel=3).
const CONV_KERNEL: usize = 3;
/// Conv1d padding for same-length output (kernel=3 → padding=1).
const CONV_PADDING: usize = 1;

// ---------------------------------------------------------------------------
// Sinusoidal PE (reusable, parameterized)
// ---------------------------------------------------------------------------

/// Build sinusoidal positional encoding matrix `[seq_len, d_model]`.
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
// Scaled attention score builder
// ---------------------------------------------------------------------------

/// Build a full Kokoro ProsodyPredictor-style block + attention scores,
/// parameterized by `KokoroDims`.
///
/// Architecture:
/// ```text
/// Conv1d → LayerNorm → AdaLayerNorm(style) → gate(sigmoid*tanh) → proj → residual
/// Q = residual + PE, K = PE
/// scores = Q @ K^T / √d_model → [seq_len, seq_len]
/// ```
pub(super) fn build_scaled_attention_scores(dims: &KokoroDims) -> (TensorKernelDef, Vec<usize>) {
    let d = dims.d_model;
    let t = dims.seq_len;
    let s = style_dim(dims);
    let h = hidden_dim(dims);

    let mut b = TensorBlockBuilder::new("attn_scores_scaled_prosody");

    // Inputs (order must match bindings)
    let raw_input = b.add_input("raw_input", &[t, d]); // 0: Variable
    let style_exp = b.add_input("style_exp", &[t, s]); // 1
    let conv_w = b.add_input("conv_w", &[d, d, CONV_KERNEL]); // 2
    let eps = b.add_input("eps", &[1]); // 3
    let ln_w = b.add_input("ln_w", &[d]); // 4
    let ln_b = b.add_input("ln_b", &[d]); // 5
    let style_proj_w = b.add_input("style_proj_w", &[s, 2 * d]); // 6
    let gate_wx = b.add_input("gate_wx", &[d, h]); // 7
    let gate_bias = b.add_input("gate_bias", &[t, h]); // 8
    let val_wx = b.add_input("val_wx", &[d, h]); // 9
    let val_bias = b.add_input("val_bias", &[t, h]); // 10
    let out_proj_w = b.add_input("out_proj_w", &[h, d]); // 11
    let pe = b.add_input("pe", &[t, d]); // 12
    let k = b.add_input("key", &[t, d]); // 13
    let ones = b.add_input("ones", &[t, d]); // 14

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
    let ada_out = b.add_binary_add(scaled, beta, &[t, d]);

    // LSTM-like gating (decomposed from concat)
    let gate_raw_x = b.add_matmul(ada_out, gate_wx, false, None, &[t, h]);
    let gate_raw = b.add_binary_add(gate_raw_x, gate_bias, &[t, h]);
    let gate = b.add_sigmoid(gate_raw, &[t, h]);

    let val_raw_x = b.add_matmul(ada_out, val_wx, false, None, &[t, h]);
    let val_raw = b.add_binary_add(val_raw_x, val_bias, &[t, h]);
    let val = b.add_tanh(val_raw, &[t, h]);

    let gated = b.add_binary_mul(gate, val, &[t, h]);

    // Output projection + residual
    let projected = b.add_matmul(gated, out_proj_w, false, None, &[t, d]);
    let residual_out = b.add_binary_add(raw_input, projected, &[t, d]);

    // Attention scores
    let q = b.add_binary_add(residual_out, pe, &[t, d]);
    let att_scale = 1.0 / (d as f32).sqrt();
    let scores_shape = [t, t];
    let scores = b.add_matmul(q, k, true, Some(att_scale), &scores_shape);

    let def = b
        .build(scores)
        .expect("valid scaled prosody attention scores graph");
    (def, scores_shape.to_vec())
}

/// Bindings for scaled ProsodyPredictor attention scores.
///
/// `enc_scale` controls weight magnitude (near-identity structure).
/// `pe_scale` controls positional encoding amplitude (higher → stronger diagonal dominance).
pub(super) fn scaled_attention_bindings(
    dims: &KokoroDims,
    enc_scale: f32,
    pe_scale: f32,
) -> Vec<TensorParamBinding> {
    let d = dims.d_model;
    let t = dims.seq_len;
    let s = style_dim(dims);
    let h = hidden_dim(dims);

    let style_exp = build_style_expanded(t, s, 0.5);
    let conv_w = build_conv_weight(d, d, CONV_KERNEL, enc_scale);
    let style_proj_w = build_style_proj_weight(s, 2 * d, enc_scale);

    // Decompose full gate weight [d+s, h] into x-part [d, h] and s-part [s, h]
    let full_gate_w = build_encoder_weight(d + s, h, enc_scale * 0.5);
    let full_val_w = build_encoder_weight(d + s, h, enc_scale * 0.5);

    let gate_wx = full_gate_w
        .slice(ndarray::s![0..d, ..])
        .to_owned()
        .into_dyn();
    let val_wx = full_val_w
        .slice(ndarray::s![0..d, ..])
        .to_owned()
        .into_dyn();
    let gate_ws = full_gate_w
        .slice(ndarray::s![d.., ..])
        .to_owned()
        .into_dyn();
    let val_ws = full_val_w.slice(ndarray::s![d.., ..]).to_owned().into_dyn();

    let gate_bias = precompute_style_bias(&style_exp, &gate_ws);
    let val_bias = precompute_style_bias(&style_exp, &val_ws);

    let out_proj_w = build_encoder_weight(h, d, enc_scale);
    let mut pe = build_sinusoidal_pe(t, d);
    pe.mapv_inplace(|v| v * pe_scale);

    vec![
        TensorParamBinding::Variable,                  // raw_input
        TensorParamBinding::ConstantTensor(style_exp), // style_exp
        TensorParamBinding::ConstantTensor(conv_w),    // conv_w
        TensorParamBinding::ConstantScalar(1e-5),      // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 1.0f32)), // ln_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 0.0f32)), // ln_b
        TensorParamBinding::ConstantTensor(style_proj_w), // style_proj_w
        TensorParamBinding::ConstantTensor(gate_wx),   // gate_wx
        TensorParamBinding::ConstantTensor(gate_bias), // gate_bias
        TensorParamBinding::ConstantTensor(val_wx),    // val_wx
        TensorParamBinding::ConstantTensor(val_bias),  // val_bias
        TensorParamBinding::ConstantTensor(out_proj_w), // out_proj_w
        TensorParamBinding::ConstantTensor(pe.clone()), // pe
        TensorParamBinding::ConstantTensor(pe.clone()), // key (same as PE for identity-like structure)
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[t, d]), 1.0f32)), // ones
    ]
}

/// Evaluate monotonicity margin at given dimensions and parameters.
///
/// Returns `(min_margin, row_margins, propagation_method, all_finite)`.
pub(super) fn evaluate_scaled_margin(
    dims: &KokoroDims,
    input_bound: f32,
    enc_scale: f32,
    pe_scale: f32,
) -> (f64, Vec<f64>, String, bool) {
    use nn_tts_verify::monotonicity::interpret_attention_monotonicity;

    let (def, _) = build_scaled_attention_scores(dims);
    let bindings = scaled_attention_bindings(dims, enc_scale, pe_scale);

    let graph = nn_verify::tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let input = super::common::uniform_bounds(&[dims.seq_len, dims.d_model], input_bound);

    let (method, output, _) =
        nn_verify::propagate_with_crown_fallback(&graph, &input).expect("propagation");
    let (lo, hi) = output.lower_upper();

    let all_finite = lo.iter().chain(hi.iter()).all(|v| v.is_finite());

    let lo_slice = lo.as_slice().expect("contiguous");
    let hi_slice = hi.as_slice().expect("contiguous");

    let method_str = match method {
        nn_verify::PropMethod::Crown => "CROWN",
        nn_verify::PropMethod::Ibp => "IBP",
        _ => "unknown",
    };

    let cert = interpret_attention_monotonicity(
        lo_slice,
        hi_slice,
        dims.seq_len,
        dims.seq_len,
        f64::from(input_bound),
        method_str,
    )
    .unwrap();

    (
        cert.min_margin,
        cert.row_margins,
        method_str.to_string(),
        all_finite,
    )
}
