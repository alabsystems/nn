// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Phase 7 graph builders and binding constructors for Kokoro ProsodyPredictor-style
//! encoder prefixes used in attention monotonicity verification.
//!
//! Extracted from compose_attention_monotonicity_phase7.rs for 500-line compliance.

use super::attn_helpers::{build_sinusoidal_pe, D_MODEL, SEQ_LEN};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

/// Conv1d kernel size (matches Kokoro ProsodyPredictor: kernel=3).
const CONV_KERNEL: usize = 3;

/// Conv1d padding for same-length output (kernel=3 → padding=1).
const CONV_PADDING: usize = 1;

// ---------------------------------------------------------------------------
// Graph builders — Architecture A: ProsodyPredictor-inspired
// ---------------------------------------------------------------------------

/// Build a ProsodyPredictor-inspired encoder prefix + attention scores.
///
/// Architecture:
///   raw_input: Variable [T, D] in [-B, B]
///   Layer 1: Linear(W_1) → ReLU
///   Layer 2: LayerNorm(eps, ln_w, ln_b)
///   Layer 3: Linear(W_2) → ReLU
///   Q = hidden + PE
///   K = PE
///   scores = Q @ K^T / √D → [T, T]
pub(super) fn build_prosody_inspired_attention_scores() -> (TensorKernelDef, Vec<usize>) {
    let mut b = TensorBlockBuilder::new("attn_scores_prosody_inspired");

    let raw_input = b.add_input("raw_input", &[SEQ_LEN, D_MODEL]);
    let w1 = b.add_input("w1", &[D_MODEL, D_MODEL]);
    let w2 = b.add_input("w2", &[D_MODEL, D_MODEL]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_w", &[D_MODEL]);
    let ln_b = b.add_input("ln_b", &[D_MODEL]);
    let pe = b.add_input("pe", &[SEQ_LEN, D_MODEL]);
    let k = b.add_input("key", &[SEQ_LEN, D_MODEL]);

    // Layer 1: Linear → ReLU
    let l1 = b.add_matmul(raw_input, w1, false, None, &[SEQ_LEN, D_MODEL]);
    let l1_act = b.add_relu(l1, &[SEQ_LEN, D_MODEL]);

    // Layer 2: LayerNorm
    let normed = b.add_layer_norm(l1_act, eps, 1, ln_w, ln_b, &[SEQ_LEN, D_MODEL]);

    // Layer 3: Linear → ReLU
    let l2 = b.add_matmul(normed, w2, false, None, &[SEQ_LEN, D_MODEL]);
    let hidden = b.add_relu(l2, &[SEQ_LEN, D_MODEL]);

    // Q = hidden + PE, scores = Q @ K^T / √D
    let q = b.add_binary_add(hidden, pe, &[SEQ_LEN, D_MODEL]);
    let scale = 1.0 / (D_MODEL as f32).sqrt();
    let scores_shape = [SEQ_LEN, SEQ_LEN];
    let scores = b.add_matmul(q, k, true, Some(scale), &scores_shape);

    let def = b
        .build(scores)
        .expect("valid prosody-inspired attention scores graph");
    (def, scores_shape.to_vec())
}

// ---------------------------------------------------------------------------
// Graph builders — Architecture B: Conv1d + LayerNorm + Residual
// ---------------------------------------------------------------------------

/// Build a Conv1d-based ProsodyBlock encoder prefix + attention scores.
///
/// Architecture (matches Kokoro ProsodyPredictor structure):
///   raw_input: Variable [T, D] in [-B, B]
///   Transpose to channels-first: [D, T]
///   Conv1d(W, kernel=3, pad=1): [D, T]
///   Transpose back: [T, D]
///   LayerNorm
///   Linear(W_proj) → ReLU
///   Residual add with raw_input
///   Q = residual_out + PE, K = PE
///   scores = Q @ K^T / √D → [T, T]
pub(super) fn build_conv_block_attention_scores() -> (TensorKernelDef, Vec<usize>) {
    let mut b = TensorBlockBuilder::new("attn_scores_conv_block");

    let raw_input = b.add_input("raw_input", &[SEQ_LEN, D_MODEL]);
    let conv_w = b.add_input("conv_w", &[D_MODEL, D_MODEL, CONV_KERNEL]);
    let proj_w = b.add_input("proj_w", &[D_MODEL, D_MODEL]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_w", &[D_MODEL]);
    let ln_b = b.add_input("ln_b", &[D_MODEL]);
    let pe = b.add_input("pe", &[SEQ_LEN, D_MODEL]);
    let k = b.add_input("key", &[SEQ_LEN, D_MODEL]);

    // Transpose to channels-first: [T, D] → [D, T]
    let input_cf = b.add_transpose(raw_input, &[1, 0], &[D_MODEL, SEQ_LEN]);

    // Conv1d: [D, T] → [D, T] (same-padding)
    let conv_out = b.add_conv1d(input_cf, conv_w, None, 1, CONV_PADDING, &[D_MODEL, SEQ_LEN]);

    // Transpose back: [D, T] → [T, D]
    let conv_back = b.add_transpose(conv_out, &[1, 0], &[SEQ_LEN, D_MODEL]);

    // LayerNorm + Linear + ReLU
    let normed = b.add_layer_norm(conv_back, eps, 1, ln_w, ln_b, &[SEQ_LEN, D_MODEL]);
    let projected = b.add_matmul(normed, proj_w, false, None, &[SEQ_LEN, D_MODEL]);
    let activated = b.add_relu(projected, &[SEQ_LEN, D_MODEL]);

    // Residual connection
    let residual_out = b.add_binary_add(raw_input, activated, &[SEQ_LEN, D_MODEL]);

    // Q = residual_out + PE, K = PE
    let q = b.add_binary_add(residual_out, pe, &[SEQ_LEN, D_MODEL]);
    let scale = 1.0 / (D_MODEL as f32).sqrt();
    let scores_shape = [SEQ_LEN, SEQ_LEN];
    let scores = b.add_matmul(q, k, true, Some(scale), &scores_shape);

    let def = b
        .build(scores)
        .expect("valid conv-block attention scores graph");
    (def, scores_shape.to_vec())
}

/// Build a two-block Conv1d encoder: two Conv1d→LayerNorm→Linear→ReLU→Residual
/// stages before attention score computation.
pub(super) fn build_two_conv_block_attention_scores() -> (TensorKernelDef, Vec<usize>) {
    let mut b = TensorBlockBuilder::new("attn_scores_two_conv_block");

    let raw_input = b.add_input("raw_input", &[SEQ_LEN, D_MODEL]);

    // Block 1 weights
    let conv_w1 = b.add_input("conv_w1", &[D_MODEL, D_MODEL, CONV_KERNEL]);
    let proj_w1 = b.add_input("proj_w1", &[D_MODEL, D_MODEL]);
    let eps1 = b.add_input("eps1", &[1]);
    let ln_w1 = b.add_input("ln_w1", &[D_MODEL]);
    let ln_b1 = b.add_input("ln_b1", &[D_MODEL]);

    // Block 2 weights
    let conv_w2 = b.add_input("conv_w2", &[D_MODEL, D_MODEL, CONV_KERNEL]);
    let proj_w2 = b.add_input("proj_w2", &[D_MODEL, D_MODEL]);
    let eps2 = b.add_input("eps2", &[1]);
    let ln_w2 = b.add_input("ln_w2", &[D_MODEL]);
    let ln_b2 = b.add_input("ln_b2", &[D_MODEL]);

    // Attention inputs
    let pe = b.add_input("pe", &[SEQ_LEN, D_MODEL]);
    let k = b.add_input("key", &[SEQ_LEN, D_MODEL]);

    // --- Block 1 ---
    let cf1 = b.add_transpose(raw_input, &[1, 0], &[D_MODEL, SEQ_LEN]);
    let conv1 = b.add_conv1d(cf1, conv_w1, None, 1, CONV_PADDING, &[D_MODEL, SEQ_LEN]);
    let back1 = b.add_transpose(conv1, &[1, 0], &[SEQ_LEN, D_MODEL]);
    let norm1 = b.add_layer_norm(back1, eps1, 1, ln_w1, ln_b1, &[SEQ_LEN, D_MODEL]);
    let proj1 = b.add_matmul(norm1, proj_w1, false, None, &[SEQ_LEN, D_MODEL]);
    let act1 = b.add_relu(proj1, &[SEQ_LEN, D_MODEL]);
    let h1 = b.add_binary_add(raw_input, act1, &[SEQ_LEN, D_MODEL]);

    // --- Block 2 ---
    let cf2 = b.add_transpose(h1, &[1, 0], &[D_MODEL, SEQ_LEN]);
    let conv2 = b.add_conv1d(cf2, conv_w2, None, 1, CONV_PADDING, &[D_MODEL, SEQ_LEN]);
    let back2 = b.add_transpose(conv2, &[1, 0], &[SEQ_LEN, D_MODEL]);
    let norm2 = b.add_layer_norm(back2, eps2, 1, ln_w2, ln_b2, &[SEQ_LEN, D_MODEL]);
    let proj2 = b.add_matmul(norm2, proj_w2, false, None, &[SEQ_LEN, D_MODEL]);
    let act2 = b.add_relu(proj2, &[SEQ_LEN, D_MODEL]);
    let hidden = b.add_binary_add(h1, act2, &[SEQ_LEN, D_MODEL]);

    // Q = hidden + PE, K = PE
    let q = b.add_binary_add(hidden, pe, &[SEQ_LEN, D_MODEL]);
    let scale = 1.0 / (D_MODEL as f32).sqrt();
    let scores_shape = [SEQ_LEN, SEQ_LEN];
    let scores = b.add_matmul(q, k, true, Some(scale), &scores_shape);

    let def = b
        .build(scores)
        .expect("valid two-conv-block attention scores graph");
    (def, scores_shape.to_vec())
}

// Weight constructors — delegated to super::common::weights (Part of #1938).
use super::common::weights;

/// Build encoder weight matrix with controlled magnitude (near-identity).
pub(super) fn build_encoder_weight(rows: usize, cols: usize, scale: f32) -> ArrayD<f32> {
    weights::encoder_weight(rows, cols, scale)
}

/// Build Conv1d weight tensor [out_ch, in_ch, kernel].
pub(super) fn build_conv_weight(
    out_ch: usize,
    in_ch: usize,
    kernel: usize,
    scale: f32,
) -> ArrayD<f32> {
    weights::conv_weight(out_ch, in_ch, kernel, scale)
}

/// Bindings for Architecture A (prosody-inspired: Linear→ReLU→LN→Linear→ReLU).
pub(super) fn prosody_inspired_bindings(enc_scale: f32, pe_scale: f32) -> Vec<TensorParamBinding> {
    let w1 = build_encoder_weight(D_MODEL, D_MODEL, enc_scale);
    let w2 = build_encoder_weight(D_MODEL, D_MODEL, enc_scale);
    let mut pe = build_sinusoidal_pe(SEQ_LEN, D_MODEL);
    pe.mapv_inplace(|v| v * pe_scale);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w1),
        TensorParamBinding::ConstantTensor(w2),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)),
        TensorParamBinding::ConstantTensor(pe.clone()),
        TensorParamBinding::ConstantTensor(pe),
    ]
}

/// Bindings for Architecture B (single conv block).
pub(super) fn conv_block_bindings(enc_scale: f32, pe_scale: f32) -> Vec<TensorParamBinding> {
    let conv_w = build_conv_weight(D_MODEL, D_MODEL, CONV_KERNEL, enc_scale);
    let proj_w = build_encoder_weight(D_MODEL, D_MODEL, enc_scale);
    let mut pe = build_sinusoidal_pe(SEQ_LEN, D_MODEL);
    pe.mapv_inplace(|v| v * pe_scale);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(conv_w),
        TensorParamBinding::ConstantTensor(proj_w),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)),
        TensorParamBinding::ConstantTensor(pe.clone()),
        TensorParamBinding::ConstantTensor(pe),
    ]
}

/// Bindings for two conv blocks.
pub(super) fn two_conv_block_bindings(enc_scale: f32, pe_scale: f32) -> Vec<TensorParamBinding> {
    let conv_w = build_conv_weight(D_MODEL, D_MODEL, CONV_KERNEL, enc_scale);
    let proj_w = build_encoder_weight(D_MODEL, D_MODEL, enc_scale);
    let mut pe = build_sinusoidal_pe(SEQ_LEN, D_MODEL);
    pe.mapv_inplace(|v| v * pe_scale);
    vec![
        TensorParamBinding::Variable,
        // Block 1
        TensorParamBinding::ConstantTensor(conv_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)),
        // Block 2
        TensorParamBinding::ConstantTensor(conv_w),
        TensorParamBinding::ConstantTensor(proj_w),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)),
        // Attention
        TensorParamBinding::ConstantTensor(pe.clone()),
        TensorParamBinding::ConstantTensor(pe),
    ]
}
