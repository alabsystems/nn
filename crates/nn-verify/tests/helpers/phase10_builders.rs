// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Phase 10 graph builders: multi-block stacked ProsodyPredictor architecture.
//!
//! The real Kokoro ProsodyPredictor uses 3 stacked blocks, each:
//!   Conv1d → AdaLayerNorm(style) → Gate(sigmoid*tanh) → Linear → Residual
//!
//! Phase 8 verified a single block. Phase 10 stacks N blocks to test whether
//! monotonicity proofs compose through deeper architectures.
//!
//! Each block has independent weights but shares the same style embedding
//! (matching the real architecture where style is a per-utterance constant).
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 10.

use super::attn_helpers::{build_sinusoidal_pe, D_MODEL, SEQ_LEN};
use super::phase7_builders::{build_conv_weight, build_encoder_weight};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{TensorKernelDef, TensorNodeId};
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

/// Style embedding dimension (matches Phase 8).
const STYLE_DIM: usize = 4;

/// Hidden dimension for LSTM-like gate structure (d_model / 2).
const HIDDEN_DIM: usize = D_MODEL / 2;

/// Conv1d kernel size (matches Kokoro ProsodyPredictor: kernel=3).
const CONV_KERNEL: usize = 3;

/// Conv1d padding for same-length output (kernel=3 → padding=1).
const CONV_PADDING: usize = 1;

// ---------------------------------------------------------------------------
// Per-block input nodes and graph construction
// ---------------------------------------------------------------------------

/// Weight input nodes for one ProsodyPredictor block.
struct BlockInputs {
    conv_w: TensorNodeId,
    eps: TensorNodeId,
    ln_w: TensorNodeId,
    ln_b: TensorNodeId,
    ones: TensorNodeId,
    gamma: TensorNodeId,
    beta: TensorNodeId,
    gate_wx: TensorNodeId,
    gate_bias: TensorNodeId,
    val_wx: TensorNodeId,
    val_bias: TensorNodeId,
    out_proj_w: TensorNodeId,
}

/// Add 12 weight inputs for one block with the given suffix.
fn add_block_inputs(b: &mut TensorBlockBuilder, suffix: &str) -> BlockInputs {
    BlockInputs {
        conv_w: b.add_input(&format!("conv_w{suffix}"), &[D_MODEL, D_MODEL, CONV_KERNEL]),
        eps: b.add_input(&format!("eps{suffix}"), &[1]),
        ln_w: b.add_input(&format!("ln_w{suffix}"), &[D_MODEL]),
        ln_b: b.add_input(&format!("ln_b{suffix}"), &[D_MODEL]),
        ones: b.add_input(&format!("ones{suffix}"), &[SEQ_LEN, D_MODEL]),
        gamma: b.add_input(&format!("gamma{suffix}"), &[SEQ_LEN, D_MODEL]),
        beta: b.add_input(&format!("beta{suffix}"), &[SEQ_LEN, D_MODEL]),
        gate_wx: b.add_input(&format!("gate_wx{suffix}"), &[D_MODEL, HIDDEN_DIM]),
        gate_bias: b.add_input(&format!("gate_bias{suffix}"), &[SEQ_LEN, HIDDEN_DIM]),
        val_wx: b.add_input(&format!("val_wx{suffix}"), &[D_MODEL, HIDDEN_DIM]),
        val_bias: b.add_input(&format!("val_bias{suffix}"), &[SEQ_LEN, HIDDEN_DIM]),
        out_proj_w: b.add_input(&format!("out_proj_w{suffix}"), &[HIDDEN_DIM, D_MODEL]),
    }
}

/// Build one ProsodyPredictor block: Conv1d → AdaLayerNorm → Gate → Proj → Residual.
fn build_block(b: &mut TensorBlockBuilder, input: TensorNodeId, w: &BlockInputs) -> TensorNodeId {
    let cf = b.add_transpose(input, &[1, 0], &[D_MODEL, SEQ_LEN]);
    let c = b.add_conv1d(cf, w.conv_w, None, 1, CONV_PADDING, &[D_MODEL, SEQ_LEN]);
    let cb = b.add_transpose(c, &[1, 0], &[SEQ_LEN, D_MODEL]);
    let n = b.add_layer_norm(cb, w.eps, 1, w.ln_w, w.ln_b, &[SEQ_LEN, D_MODEL]);
    let s = b.add_binary_add(w.ones, w.gamma, &[SEQ_LEN, D_MODEL]);
    let sc = b.add_binary_mul(n, s, &[SEQ_LEN, D_MODEL]);
    let ada = b.add_binary_add(sc, w.beta, &[SEQ_LEN, D_MODEL]);
    let gx = b.add_matmul(ada, w.gate_wx, false, None, &[SEQ_LEN, HIDDEN_DIM]);
    let gr = b.add_binary_add(gx, w.gate_bias, &[SEQ_LEN, HIDDEN_DIM]);
    let g = b.add_sigmoid(gr, &[SEQ_LEN, HIDDEN_DIM]);
    let vx = b.add_matmul(ada, w.val_wx, false, None, &[SEQ_LEN, HIDDEN_DIM]);
    let vr = b.add_binary_add(vx, w.val_bias, &[SEQ_LEN, HIDDEN_DIM]);
    let v = b.add_tanh(vr, &[SEQ_LEN, HIDDEN_DIM]);
    let gated = b.add_binary_mul(g, v, &[SEQ_LEN, HIDDEN_DIM]);
    let proj = b.add_matmul(gated, w.out_proj_w, false, None, &[SEQ_LEN, D_MODEL]);
    b.add_binary_add(input, proj, &[SEQ_LEN, D_MODEL])
}

// ---------------------------------------------------------------------------
// 2-block stacked ProsodyPredictor architecture
// ---------------------------------------------------------------------------

/// Build a 2-block stacked ProsodyPredictor + attention scores.
///
/// Architecture:
/// ```text
/// Block 1: Conv1d → AdaLayerNorm(style) → Gate → Proj → Residual
/// Block 2: Conv1d → AdaLayerNorm(style) → Gate → Proj → Residual
/// Attention: Q = block2_out + PE, scores = Q @ K^T / √D
/// ```
pub(super) fn build_two_block_prosody_predictor() -> (TensorKernelDef, Vec<usize>) {
    let mut b = TensorBlockBuilder::new("attn_scores_two_block_prosody");

    let raw_input = b.add_input("raw_input", &[SEQ_LEN, D_MODEL]);
    let w1 = add_block_inputs(&mut b, "1");
    let w2 = add_block_inputs(&mut b, "2");
    let pe = b.add_input("pe", &[SEQ_LEN, D_MODEL]);
    let k = b.add_input("key", &[SEQ_LEN, D_MODEL]);

    let h1 = build_block(&mut b, raw_input, &w1);
    let h2 = build_block(&mut b, h1, &w2);

    let q = b.add_binary_add(h2, pe, &[SEQ_LEN, D_MODEL]);
    let att_scale = 1.0 / (D_MODEL as f32).sqrt();
    let scores_shape = [SEQ_LEN, SEQ_LEN];
    let scores = b.add_matmul(q, k, true, Some(att_scale), &scores_shape);

    let def = b
        .build(scores)
        .expect("valid two-block prosody predictor graph");
    (def, scores_shape.to_vec())
}

// ---------------------------------------------------------------------------
// Binding constructors
// ---------------------------------------------------------------------------

/// Build style embedding expanded to [T, S].
fn build_style_expanded(magnitude: f32) -> ArrayD<f32> {
    let row: Vec<f32> = (0..STYLE_DIM)
        .map(|i| magnitude * (1.0 + 0.1 * i as f32))
        .collect();
    let data: Vec<f32> = row
        .iter()
        .cycle()
        .take(SEQ_LEN * STYLE_DIM)
        .copied()
        .collect();
    ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, STYLE_DIM]), data).expect("valid style shape")
}

/// Build style projection weight [S, out_dim].
fn build_style_proj_weight(out_dim: usize, scale: f32) -> ArrayD<f32> {
    let total = STYLE_DIM * out_dim;
    let mut data = vec![0.0f32; total];
    for i in 0..STYLE_DIM {
        for j in 0..out_dim {
            data[i * out_dim + j] = scale * 0.01 * ((i * j % 7) as f32 + 0.1);
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[STYLE_DIM, out_dim]), data).expect("valid style proj shape")
}

/// Pre-compute style @ W_s for decomposed concat+matmul.
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

/// Pre-compute style gamma and beta for one block.
fn precompute_style_gamma_beta(
    style: &ArrayD<f32>,
    style_proj_w: &ArrayD<f32>,
) -> (ArrayD<f32>, ArrayD<f32>) {
    let s = style
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("2D");
    let w = style_proj_w
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("2D");
    let proj = s.dot(&w); // [T, 2*D]
    let gamma = proj
        .slice(ndarray::s![.., 0..D_MODEL])
        .to_owned()
        .into_dyn();
    let beta = proj.slice(ndarray::s![.., D_MODEL..]).to_owned().into_dyn();
    (gamma, beta)
}

/// Build one block's weight bindings (everything except raw_input, pe, key).
fn block_bindings(enc_scale: f32, style: &ArrayD<f32>) -> Vec<TensorParamBinding> {
    let conv_w = build_conv_weight(D_MODEL, D_MODEL, CONV_KERNEL, enc_scale);
    let style_proj_w = build_style_proj_weight(2 * D_MODEL, enc_scale);
    let (gamma, beta) = precompute_style_gamma_beta(style, &style_proj_w);

    // Full gate weights, decomposed into x-part and s-part
    let full_gate_w = build_encoder_weight(D_MODEL + STYLE_DIM, HIDDEN_DIM, enc_scale * 0.5);
    let full_val_w = build_encoder_weight(D_MODEL + STYLE_DIM, HIDDEN_DIM, enc_scale * 0.5);
    let gate_wx = full_gate_w
        .slice(ndarray::s![0..D_MODEL, ..])
        .to_owned()
        .into_dyn();
    let val_wx = full_val_w
        .slice(ndarray::s![0..D_MODEL, ..])
        .to_owned()
        .into_dyn();
    let gate_ws = full_gate_w
        .slice(ndarray::s![D_MODEL.., ..])
        .to_owned()
        .into_dyn();
    let val_ws = full_val_w
        .slice(ndarray::s![D_MODEL.., ..])
        .to_owned()
        .into_dyn();
    let gate_bias = precompute_style_bias(style, &gate_ws);
    let val_bias = precompute_style_bias(style, &val_ws);
    let out_proj_w = build_encoder_weight(HIDDEN_DIM, D_MODEL, enc_scale);

    vec![
        TensorParamBinding::ConstantTensor(conv_w), // conv_w
        TensorParamBinding::ConstantScalar(1e-5),   // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)), // ln_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)), // ln_b
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[SEQ_LEN, D_MODEL]), 1.0f32)), // ones
        TensorParamBinding::ConstantTensor(gamma), // gamma
        TensorParamBinding::ConstantTensor(beta),  // beta
        TensorParamBinding::ConstantTensor(gate_wx), // gate_wx
        TensorParamBinding::ConstantTensor(gate_bias), // gate_bias
        TensorParamBinding::ConstantTensor(val_wx), // val_wx
        TensorParamBinding::ConstantTensor(val_bias), // val_bias
        TensorParamBinding::ConstantTensor(out_proj_w), // out_proj_w
    ]
}

/// Bindings for 2-block stacked ProsodyPredictor.
/// Input order: raw_input, [block1 12 weights], [block2 12 weights], pe, key
pub(super) fn two_block_bindings(enc_scale: f32, pe_scale: f32) -> Vec<TensorParamBinding> {
    let style = build_style_expanded(0.5);
    let mut pe = build_sinusoidal_pe(SEQ_LEN, D_MODEL);
    pe.mapv_inplace(|v| v * pe_scale);

    let mut bindings = vec![TensorParamBinding::Variable]; // raw_input
    bindings.extend(block_bindings(enc_scale, &style)); // block 1 (12 items)
    bindings.extend(block_bindings(enc_scale, &style)); // block 2 (12 items)
    bindings.push(TensorParamBinding::ConstantTensor(pe.clone())); // pe
    bindings.push(TensorParamBinding::ConstantTensor(pe)); // key
    bindings
}
