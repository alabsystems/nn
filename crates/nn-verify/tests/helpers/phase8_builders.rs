// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Phase 8 graph builders for full Kokoro ProsodyPredictor architecture.
//!
//! Extends Phase 7 (Conv1d + LayerNorm + Residual) with:
//! - **AdaLayerNorm(style)**: `(1 + gamma(style)) * LayerNorm(x) + beta(style)`
//! - **Style concatenation**: cat(normed, style) along feature dimension
//! - **LSTM-like gating**: Linear→Sigmoid gate structure approximating LSTM
//!
//! The LSTM is modeled as a gated linear layer for NY tractability:
//! real LSTM is sequential over T steps, but for bounds verification we model
//! the gate structure (sigmoid * tanh) which captures the key nonlinearity.
//!
//! Style inputs are pre-expanded to [T, S] (same style per timestep) to avoid
//! broadcast→matmul constant-fold issues in graph translation (requires ≥2D).
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 8.

use super::attn_helpers::{build_sinusoidal_pe, D_MODEL, SEQ_LEN};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

// Re-export Phase 7 helpers for weight construction
use super::phase7_builders::{build_conv_weight, build_encoder_weight};

/// Style embedding dimension (matches Kokoro: 128, scaled down for verification).
pub(super) const STYLE_DIM: usize = 4;

/// Hidden dimension for LSTM-like gate structure (d_model / 2).
const HIDDEN_DIM: usize = D_MODEL / 2;

/// Conv1d kernel size (matches Kokoro ProsodyPredictor: kernel=3).
const CONV_KERNEL: usize = 3;

/// Conv1d padding for same-length output (kernel=3 → padding=1).
const CONV_PADDING: usize = 1;

// ---------------------------------------------------------------------------
// Architecture C: Full ProsodyPredictor block (Conv1d + AdaLayerNorm + gate)
// ---------------------------------------------------------------------------

/// Build a full Kokoro ProsodyPredictor-style block + attention scores.
///
/// Architecture (matches actual Kokoro ProsodyPredictor):
/// ```text
/// Block:
///   Conv1d → AdaLayerNorm(style) → gate(sigmoid*tanh) → proj → residual
/// Attention:
///   Q = residual_out + PE, K = PE
///   scores = Q @ K^T / √D → [T, T]
/// ```
///
/// The Kokoro ProsodyPredictor concatenates `[ada_out, style]` before the
/// LSTM gate. Since NY's Concat translator rejects constant inputs,
/// we decompose `concat([x, s]) @ W` as `x @ W_x + style_bias` where
/// `style_bias = s @ W_s` is pre-computed as a constant tensor. This is
/// mathematically equivalent and avoids the constant-concat limitation.
pub(super) fn build_prosody_predictor_attention_scores() -> (TensorKernelDef, Vec<usize>) {
    let mut b = TensorBlockBuilder::new("attn_scores_prosody_predictor");

    // Inputs (order matters — must match bindings)
    let raw_input = b.add_input("raw_input", &[SEQ_LEN, D_MODEL]); // 0: Variable
    let style_exp = b.add_input("style_exp", &[SEQ_LEN, STYLE_DIM]); // 1: pre-expanded style
    let conv_w = b.add_input("conv_w", &[D_MODEL, D_MODEL, CONV_KERNEL]); // 2
    let eps = b.add_input("eps", &[1]); // 3
    let ln_w = b.add_input("ln_w", &[D_MODEL]); // 4
    let ln_b = b.add_input("ln_b", &[D_MODEL]); // 5
    let style_proj_w = b.add_input("style_proj_w", &[STYLE_DIM, 2 * D_MODEL]); // 6
                                                                               // Decomposed gate weights: ada_out @ W_gate_x + gate_bias
    let gate_wx = b.add_input("gate_wx", &[D_MODEL, HIDDEN_DIM]); // 7
    let gate_bias = b.add_input("gate_bias", &[SEQ_LEN, HIDDEN_DIM]); // 8: pre-computed style @ W_gate_s
    let val_wx = b.add_input("val_wx", &[D_MODEL, HIDDEN_DIM]); // 9
    let val_bias = b.add_input("val_bias", &[SEQ_LEN, HIDDEN_DIM]); // 10: pre-computed style @ W_val_s
    let out_proj_w = b.add_input("out_proj_w", &[HIDDEN_DIM, D_MODEL]); // 11
    let pe = b.add_input("pe", &[SEQ_LEN, D_MODEL]); // 12
    let k = b.add_input("key", &[SEQ_LEN, D_MODEL]); // 13
    let ones = b.add_input("ones", &[SEQ_LEN, D_MODEL]); // 14

    // --- Conv1d: channels-first processing ---
    let input_cf = b.add_transpose(raw_input, &[1, 0], &[D_MODEL, SEQ_LEN]);
    let conv_out = b.add_conv1d(input_cf, conv_w, None, 1, CONV_PADDING, &[D_MODEL, SEQ_LEN]);
    let conv_back = b.add_transpose(conv_out, &[1, 0], &[SEQ_LEN, D_MODEL]);

    // --- AdaLayerNorm ---
    let normed = b.add_layer_norm(conv_back, eps, 1, ln_w, ln_b, &[SEQ_LEN, D_MODEL]);

    // Style projection: [T, S] @ [S, 2D] → [T, 2D]
    let style_proj = b.add_matmul(
        style_exp,
        style_proj_w,
        false,
        None,
        &[SEQ_LEN, 2 * D_MODEL],
    );
    let gamma = b.add_narrow(style_proj, 1, 0, D_MODEL, &[SEQ_LEN, D_MODEL]);
    let beta = b.add_narrow(style_proj, 1, D_MODEL, D_MODEL, &[SEQ_LEN, D_MODEL]);

    // (1 + gamma) * normed + beta
    let scale = b.add_binary_add(ones, gamma, &[SEQ_LEN, D_MODEL]);
    let scaled = b.add_binary_mul(normed, scale, &[SEQ_LEN, D_MODEL]);
    let ada_out = b.add_binary_add(scaled, beta, &[SEQ_LEN, D_MODEL]);

    // --- LSTM-like gating (decomposed from concat) ---
    // Instead of concat([ada_out, style]) @ W, decompose as:
    //   ada_out @ W_x + style_bias   where style_bias = style @ W_s (pre-computed)
    let gate_raw_x = b.add_matmul(ada_out, gate_wx, false, None, &[SEQ_LEN, HIDDEN_DIM]);
    let gate_raw = b.add_binary_add(gate_raw_x, gate_bias, &[SEQ_LEN, HIDDEN_DIM]);
    let gate = b.add_sigmoid(gate_raw, &[SEQ_LEN, HIDDEN_DIM]);

    let val_raw_x = b.add_matmul(ada_out, val_wx, false, None, &[SEQ_LEN, HIDDEN_DIM]);
    let val_raw = b.add_binary_add(val_raw_x, val_bias, &[SEQ_LEN, HIDDEN_DIM]);
    let val = b.add_tanh(val_raw, &[SEQ_LEN, HIDDEN_DIM]);

    // gated = gate * val → [T, H]
    let gated = b.add_binary_mul(gate, val, &[SEQ_LEN, HIDDEN_DIM]);

    // --- Output projection + residual ---
    let projected = b.add_matmul(gated, out_proj_w, false, None, &[SEQ_LEN, D_MODEL]);
    let residual_out = b.add_binary_add(raw_input, projected, &[SEQ_LEN, D_MODEL]);

    // --- Attention scores ---
    let q = b.add_binary_add(residual_out, pe, &[SEQ_LEN, D_MODEL]);
    let att_scale = 1.0 / (D_MODEL as f32).sqrt();
    let scores_shape = [SEQ_LEN, SEQ_LEN];
    let scores = b.add_matmul(q, k, true, Some(att_scale), &scores_shape);

    let def = b
        .build(scores)
        .expect("valid prosody-predictor attention scores graph");
    (def, scores_shape.to_vec())
}

// ---------------------------------------------------------------------------
// Architecture D: Simplified ProsodyPredictor (no gate, style affine only)
// ---------------------------------------------------------------------------

/// Build a simplified ProsodyPredictor: Conv1d + AdaLayerNorm(style) + Linear
/// without the LSTM gate structure. Tests whether style conditioning alone
/// improves monotonicity bounds.
pub(super) fn build_ada_norm_attention_scores() -> (TensorKernelDef, Vec<usize>) {
    let mut b = TensorBlockBuilder::new("attn_scores_ada_norm");

    // Inputs (order matches bindings)
    let raw_input = b.add_input("raw_input", &[SEQ_LEN, D_MODEL]); // 0: Variable
    let style_exp = b.add_input("style_exp", &[SEQ_LEN, STYLE_DIM]); // 1: pre-expanded style
    let conv_w = b.add_input("conv_w", &[D_MODEL, D_MODEL, CONV_KERNEL]); // 2
    let eps = b.add_input("eps", &[1]); // 3
    let ln_w = b.add_input("ln_w", &[D_MODEL]); // 4
    let ln_b = b.add_input("ln_b", &[D_MODEL]); // 5
    let style_proj_w = b.add_input("style_proj_w", &[STYLE_DIM, 2 * D_MODEL]); // 6
    let proj_w = b.add_input("proj_w", &[D_MODEL, D_MODEL]); // 7
    let pe = b.add_input("pe", &[SEQ_LEN, D_MODEL]); // 8
    let k = b.add_input("key", &[SEQ_LEN, D_MODEL]); // 9
    let ones = b.add_input("ones", &[SEQ_LEN, D_MODEL]); // 10

    // Conv1d
    let input_cf = b.add_transpose(raw_input, &[1, 0], &[D_MODEL, SEQ_LEN]);
    let conv_out = b.add_conv1d(input_cf, conv_w, None, 1, CONV_PADDING, &[D_MODEL, SEQ_LEN]);
    let conv_back = b.add_transpose(conv_out, &[1, 0], &[SEQ_LEN, D_MODEL]);

    // LayerNorm
    let normed = b.add_layer_norm(conv_back, eps, 1, ln_w, ln_b, &[SEQ_LEN, D_MODEL]);

    // AdaLayerNorm: style projection → gamma, beta
    let style_proj = b.add_matmul(
        style_exp,
        style_proj_w,
        false,
        None,
        &[SEQ_LEN, 2 * D_MODEL],
    );
    let gamma = b.add_narrow(style_proj, 1, 0, D_MODEL, &[SEQ_LEN, D_MODEL]);
    let beta = b.add_narrow(style_proj, 1, D_MODEL, D_MODEL, &[SEQ_LEN, D_MODEL]);

    let scale_factor = b.add_binary_add(ones, gamma, &[SEQ_LEN, D_MODEL]);
    let scaled = b.add_binary_mul(normed, scale_factor, &[SEQ_LEN, D_MODEL]);
    let ada_out = b.add_binary_add(scaled, beta, &[SEQ_LEN, D_MODEL]);

    // Linear + ReLU + Residual
    let projected = b.add_matmul(ada_out, proj_w, false, None, &[SEQ_LEN, D_MODEL]);
    let activated = b.add_relu(projected, &[SEQ_LEN, D_MODEL]);
    let residual_out = b.add_binary_add(raw_input, activated, &[SEQ_LEN, D_MODEL]);

    // Attention scores
    let q = b.add_binary_add(residual_out, pe, &[SEQ_LEN, D_MODEL]);
    let att_scale = 1.0 / (D_MODEL as f32).sqrt();
    let scores_shape = [SEQ_LEN, SEQ_LEN];
    let scores = b.add_matmul(q, k, true, Some(att_scale), &scores_shape);

    let def = b
        .build(scores)
        .expect("valid ada-norm attention scores graph");
    (def, scores_shape.to_vec())
}

// ---------------------------------------------------------------------------
// Binding constructors
// ---------------------------------------------------------------------------

/// Build style embedding expanded to [T, S]: same style per timestep.
fn build_style_expanded(seq_len: usize, style_dim: usize, magnitude: f32) -> ArrayD<f32> {
    let row: Vec<f32> = (0..style_dim)
        .map(|i| magnitude * (1.0 + 0.1 * i as f32))
        .collect();
    let data: Vec<f32> = row
        .iter()
        .cycle()
        .take(seq_len * style_dim)
        .copied()
        .collect();
    ArrayD::from_shape_vec(IxDyn(&[seq_len, style_dim]), data).expect("valid style shape")
}

/// Build style projection weight [S, 2*D] for gamma/beta.
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

/// Pre-compute `style @ W_s` bias for decomposed concat+matmul.
///
/// Given style [T, S] and W_s [S, H], returns [T, H].
fn precompute_style_bias(style: &ArrayD<f32>, w_s: &ArrayD<f32>) -> ArrayD<f32> {
    let s = style
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("style 2D");
    let w = w_s
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("w_s 2D");
    let result = s.dot(&w);
    result.into_dyn()
}

/// Bindings for Architecture C (full ProsodyPredictor block).
/// Input order: raw_input, style_exp, conv_w, eps, ln_w, ln_b, style_proj_w,
///              gate_wx, gate_bias, val_wx, val_bias, out_proj_w, pe, key, ones
pub(super) fn prosody_predictor_bindings(enc_scale: f32, pe_scale: f32) -> Vec<TensorParamBinding> {
    let style_exp = build_style_expanded(SEQ_LEN, STYLE_DIM, 0.5);
    let conv_w = build_conv_weight(D_MODEL, D_MODEL, CONV_KERNEL, enc_scale);
    let style_proj_w = build_style_proj_weight(STYLE_DIM, 2 * D_MODEL, enc_scale);

    // Decompose the full gate weight [D+S, H] into x-part [D, H] and s-part [S, H]
    let full_gate_w = build_encoder_weight(D_MODEL + STYLE_DIM, HIDDEN_DIM, enc_scale * 0.5);
    let full_val_w = build_encoder_weight(D_MODEL + STYLE_DIM, HIDDEN_DIM, enc_scale * 0.5);

    // Extract x-part: rows 0..D → [D, H]
    let gate_wx = full_gate_w
        .slice(ndarray::s![0..D_MODEL, ..])
        .to_owned()
        .into_dyn();
    let val_wx = full_val_w
        .slice(ndarray::s![0..D_MODEL, ..])
        .to_owned()
        .into_dyn();

    // Extract s-part: rows D..D+S → [S, H]
    let gate_ws = full_gate_w
        .slice(ndarray::s![D_MODEL.., ..])
        .to_owned()
        .into_dyn();
    let val_ws = full_val_w
        .slice(ndarray::s![D_MODEL.., ..])
        .to_owned()
        .into_dyn();

    // Pre-compute style bias: style [T, S] @ W_s [S, H] → [T, H]
    let gate_bias = precompute_style_bias(&style_exp, &gate_ws);
    let val_bias = precompute_style_bias(&style_exp, &val_ws);

    let out_proj_w = build_encoder_weight(HIDDEN_DIM, D_MODEL, enc_scale);
    let mut pe = build_sinusoidal_pe(SEQ_LEN, D_MODEL);
    pe.mapv_inplace(|v| v * pe_scale);

    vec![
        TensorParamBinding::Variable,                  // raw_input
        TensorParamBinding::ConstantTensor(style_exp), // style_exp
        TensorParamBinding::ConstantTensor(conv_w),    // conv_w
        TensorParamBinding::ConstantScalar(1e-5),      // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)), // ln_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)), // ln_b
        TensorParamBinding::ConstantTensor(style_proj_w), // style_proj_w
        TensorParamBinding::ConstantTensor(gate_wx),   // gate_wx
        TensorParamBinding::ConstantTensor(gate_bias), // gate_bias
        TensorParamBinding::ConstantTensor(val_wx),    // val_wx
        TensorParamBinding::ConstantTensor(val_bias),  // val_bias
        TensorParamBinding::ConstantTensor(out_proj_w), // out_proj_w
        TensorParamBinding::ConstantTensor(pe.clone()), // pe
        TensorParamBinding::ConstantTensor(pe.clone()), // key
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[SEQ_LEN, D_MODEL]), 1.0f32)), // ones
    ]
}

/// Bindings for Architecture D (AdaLayerNorm only, no gate).
/// Input order: raw_input, style_exp, conv_w, eps, ln_w, ln_b, style_proj_w,
///              proj_w, pe, key, ones
pub(super) fn ada_norm_bindings(enc_scale: f32, pe_scale: f32) -> Vec<TensorParamBinding> {
    let style_exp = build_style_expanded(SEQ_LEN, STYLE_DIM, 0.5);
    let conv_w = build_conv_weight(D_MODEL, D_MODEL, CONV_KERNEL, enc_scale);
    let style_proj_w = build_style_proj_weight(STYLE_DIM, 2 * D_MODEL, enc_scale);
    let proj_w = build_encoder_weight(D_MODEL, D_MODEL, enc_scale);
    let mut pe = build_sinusoidal_pe(SEQ_LEN, D_MODEL);
    pe.mapv_inplace(|v| v * pe_scale);

    vec![
        TensorParamBinding::Variable,                  // raw_input
        TensorParamBinding::ConstantTensor(style_exp), // style_exp
        TensorParamBinding::ConstantTensor(conv_w),    // conv_w
        TensorParamBinding::ConstantScalar(1e-5),      // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)), // ln_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)), // ln_b
        TensorParamBinding::ConstantTensor(style_proj_w), // style_proj_w
        TensorParamBinding::ConstantTensor(proj_w),    // proj_w
        TensorParamBinding::ConstantTensor(pe.clone()), // pe
        TensorParamBinding::ConstantTensor(pe.clone()), // key
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[SEQ_LEN, D_MODEL]), 1.0f32)), // ones
    ]
}
