// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Dampened-residual ProsodyPredictor builders for Phase 48.
//!
//! Each block computes `text_input + alpha * projection` instead of
//! `text_input + projection`. Alpha < 1.0 reduces the per-block bound
//! widening caused by residual accumulation.

#![allow(dead_code, clippy::duplicated_attributes)]

use super::helpers::KokoroDims;
use super::prosody_scaled::ProsodyDims;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::TensorNodeId;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

/// Build a ProsodyPredictor with dampened residuals: h_new = h + alpha * proj.
///
/// The `alpha` parameter does not affect graph structure (alpha is an input node),
/// only the bindings. It is accepted here for API symmetry with `build_bindings_dampened`.
pub(super) fn build_prosody_dampened(
    dims: &KokoroDims,
    n_blocks: usize,
    _alpha: f32,
) -> (TensorKernelDef, usize) {
    let pd = ProsodyDims::from_kokoro(dims);
    let mut b = TensorBlockBuilder::new("kokoro_prosody_dampened");

    let text_size = pd.d_model * pd.seq_len;
    let flat_input = b.add_input("flat_input", &[pd.flat_input_size()]);
    let text_flat = b.add_narrow(flat_input, 0, 0, text_size, &[text_size]);
    let text_input = b.add_reshape(text_flat, &[pd.d_model, pd.seq_len]);
    let style_input = b.add_narrow(flat_input, 0, text_size, pd.style_dim, &[pd.style_dim]);
    let eps = b.add_input("eps", &[1]);

    let mut h = text_input;
    for idx in 0..n_blocks {
        h = add_prosody_block_dampened(&mut b, h, style_input, eps, &pd, idx);
    }

    let dur_logits = add_duration_projection(&mut b, h, &pd);

    (
        b.build(dur_logits).expect("valid dampened prosody graph"),
        pd.seq_len,
    )
}

/// ProsodyBlock with dampened residual: h_new = text_input + alpha * proj.
fn add_prosody_block_dampened(
    b: &mut TensorBlockBuilder,
    text_input: TensorNodeId,
    style_input: TensorNodeId,
    eps: TensorNodeId,
    pd: &ProsodyDims,
    idx: usize,
) -> TensorNodeId {
    let d = pd.d_model;
    let s = pd.seq_len;
    let sd = pd.style_dim;
    let lh = pd.lstm_hidden;
    let lstm_dim = d + sd;

    // Conv1d: [d, s] → [d, s]
    let conv_w = b.add_input(&format!("b{idx}_conv_w"), &[d, d, 3]);
    let conv_b = b.add_input(&format!("b{idx}_conv_b"), &[d]);
    let conv_b_bc = b.add_broadcast_left(conv_b, &[d, s]);
    let conv_out = b.add_conv1d(text_input, conv_w, None, 1, 1, &[d, s]);
    let conv_biased = b.add_binary_add(conv_out, conv_b_bc, &[d, s]);

    // Transpose: [d, s] → [s, d]
    let conv_t = b.add_transpose(conv_biased, &[1, 0], &[s, d]);

    // AdaLayerNorm
    let style_proj_w = b.add_input(&format!("b{idx}_adaln_w"), &[2 * d, sd]);
    let style_proj_b = b.add_input(&format!("b{idx}_adaln_b"), &[2 * d]);
    let style_rs = b.add_reshape(style_input, &[1, sd]);
    let style_projected = b.add_matmul(style_rs, style_proj_w, true, None, &[1, 2 * d]);
    let style_proj_b_bc = b.add_broadcast(style_proj_b, &[1, 2 * d]);
    let sp_biased = b.add_binary_add(style_projected, style_proj_b_bc, &[1, 2 * d]);
    let gamma_raw = b.add_narrow(sp_biased, 1, 0, d, &[1, d]);
    let beta_raw = b.add_narrow(sp_biased, 1, d, d, &[1, d]);
    let ones = b.add_input(&format!("b{idx}_ones"), &[1, d]);
    let gamma = b.add_binary_add(ones, gamma_raw, &[1, d]);
    let ln_w = b.add_input(&format!("b{idx}_ln_w"), &[d]);
    let ln_b = b.add_input(&format!("b{idx}_ln_b"), &[d]);
    let normed = b.add_layer_norm(conv_t, eps, 1, ln_w, ln_b, &[s, d]);
    let gamma_bc = b.add_broadcast(gamma, &[s, d]);
    let beta_bc = b.add_broadcast(beta_raw, &[s, d]);
    let scaled = b.add_binary_mul(gamma_bc, normed, &[s, d]);
    let adaln_out = b.add_binary_add(scaled, beta_bc, &[s, d]);

    // Concat with style: [s, d] cat [s, sd] → [s, lstm_dim]
    let style_bc = b.add_broadcast(style_rs, &[s, sd]);
    let lstm_input = b.add_concat(&[adaln_out, style_bc], 1, &[s, lstm_dim]);

    // LSTM (single timestep)
    let h0 = b.add_input(&format!("b{idx}_lstm_h0"), &[lh]);
    let c0 = b.add_input(&format!("b{idx}_lstm_c0"), &[lh]);
    let w_ih = b.add_input(&format!("b{idx}_lstm_w_ih"), &[4 * lh, lstm_dim]);
    let w_hh = b.add_input(&format!("b{idx}_lstm_w_hh"), &[4 * lh, lh]);
    let bias = b.add_input(&format!("b{idx}_lstm_bias"), &[4 * lh]);
    let first_step = b.add_narrow(lstm_input, 0, 0, 1, &[1, lstm_dim]);
    let first_sq = b.add_reshape(first_step, &[lstm_dim]);
    let lstm_out = b.add_lstm(first_sq, h0, c0, w_ih, w_hh, Some(bias), &[lh]);

    // Linear projection: [lh] → [d]
    let proj_w = b.add_input(&format!("b{idx}_proj_w"), &[d, lh]);
    let proj_b = b.add_input(&format!("b{idx}_proj_b"), &[d]);
    let lstm_rs = b.add_reshape(lstm_out, &[1, lh]);
    let projected = b.add_matmul(lstm_rs, proj_w, true, None, &[1, d]);
    let proj_b_bc = b.add_broadcast(proj_b, &[1, d]);
    let proj_biased = b.add_binary_add(projected, proj_b_bc, &[1, d]);

    // Broadcast for residual: [1, d] → [d, s]
    let proj_bc = b.add_broadcast(proj_biased, &[s, d]);
    let proj_t = b.add_transpose(proj_bc, &[1, 0], &[d, s]);

    // Dampened residual: text_input + alpha * proj_t
    let alpha_const = b.add_input(&format!("b{idx}_residual_alpha"), &[1]);
    let alpha_bc = b.add_broadcast(alpha_const, &[d, s]);
    let scaled_proj = b.add_binary_mul(proj_t, alpha_bc, &[d, s]);
    b.add_binary_add(text_input, scaled_proj, &[d, s])
}

/// Duration projection: h [d, s] → dur_logits [s]
fn add_duration_projection(
    b: &mut TensorBlockBuilder,
    h: TensorNodeId,
    pd: &ProsodyDims,
) -> TensorNodeId {
    let d = pd.d_model;
    let s = pd.seq_len;
    let h_t = b.add_transpose(h, &[1, 0], &[s, d]);
    let dur_w = b.add_input("dur_proj_w", &[1, d]);
    let dur_b = b.add_input("dur_proj_b", &[1]);
    let dur_2d = b.add_matmul(h_t, dur_w, true, None, &[s, 1]);
    let dur_b_bc = b.add_broadcast(dur_b, &[s, 1]);
    let dur_biased = b.add_binary_add(dur_2d, dur_b_bc, &[s, 1]);
    b.add_reshape(dur_biased, &[s])
}

/// Build bindings for dampened-residual variant (14 per-block + 1 alpha per block).
pub(super) fn build_bindings_dampened(
    dims: &KokoroDims,
    n_blocks: usize,
    weight_mag: f32,
    alpha: f32,
) -> Vec<TensorParamBinding> {
    let pd = ProsodyDims::from_kokoro(dims);
    let d = pd.d_model;
    let sd = pd.style_dim;
    let lh = pd.lstm_hidden;
    let lstm_dim = d + sd;
    let mut bindings = Vec::new();

    // flat_input (Variable)
    bindings.push(TensorParamBinding::Variable);
    // eps
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    for _ in 0..n_blocks {
        // 14 standard per-block bindings (same order as build_bindings)
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[d, d, 3]),
            weight_mag,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[d]),
            0.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[2 * d, sd]),
            weight_mag,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[2 * d]),
            0.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[1, d]),
            1.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[d]),
            1.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[d]),
            0.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[lh]),
            0.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[lh]),
            0.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[4 * lh, lstm_dim]),
            weight_mag,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[4 * lh, lh]),
            weight_mag,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[4 * lh]),
            0.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[d, lh]),
            weight_mag,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[d]),
            0.0f32,
        )));
        // residual_alpha [1]
        bindings.push(TensorParamBinding::ConstantScalar(alpha));
    }

    // Duration projection
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1, d]),
        weight_mag,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.0f32,
    )));

    bindings
}

/// Run a dampened proof and return (lower_bound, is_proven, method_str).
pub(super) fn run_dampened_proof(
    dims: &KokoroDims,
    n_blocks: usize,
    mag: f32,
    ib: f32,
    alpha: f32,
) -> (f64, bool, &'static str) {
    let pd = ProsodyDims::from_kokoro(dims);
    let (def, _) = build_prosody_dampened(dims, n_blocks, alpha);
    let bindings = build_bindings_dampened(dims, n_blocks, mag, alpha);
    super::prosody_n_blocks::run_proof(&def, &bindings, pd.flat_input_size(), ib)
}
