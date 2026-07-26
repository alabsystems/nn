// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for the GLM-OCR full decoder pipeline bounds.
//!
//! Verifies NY IBP and CROWN bound propagation through GLM-OCR
//! decoder pipeline components used for optical character recognition in
//! the dpdf document understanding stack. GLM-OCR uses the GLM-4/5
//! decoder-only architecture.
//!
//! ## Tests (8 tests)
//!
//! 1. **Decoder embedding bounds** — Token + position embedding propagation (IBP)
//! 2. **RMSNorm bound contraction** — Root mean square normalization (IBP + CROWN)
//! 3. **SwiGLU FFN bounds** — gate_proj -> SiLU -> mul(up_proj) -> down_proj (IBP + CROWN)
//! 4. **RoPE attention bounds** — RoPE-enhanced attention score propagation (IBP)
//! 5. **Causal decoder block bounds** — Single decoder block composition (IBP + CROWN)
//! 6. **Two-layer stack bounds** — Two decoder blocks composed (IBP)
//! 7. **Vocabulary projection bounds** — Final logits projection (IBP)
//! 8. **Full pipeline bounds** — End-to-end mini decoder pipeline (IBP + CROWN)
//!
//! Architecture references:
//! - GLM-4V / ChatGLM (THUDM): Decoder-only transformer for OCR
//! - RMSNorm (Zhang & Sennrich, 2019): Root mean square normalization
//! - SwiGLU (Shazeer, 2020): SiLU-gated FFN
//! - GQA (Ainslie et al., 2023): Grouped-query attention
//! - RoPE (Su et al., 2021): Rotary positional embeddings
//!
//! Dimensions (symbolic, small for fast verification):
//! - HIDDEN_DIM=8, FFN_DIM=16, NUM_HEADS=2, NUM_KV_HEADS=2
//! - HEAD_DIM=4, SEQ_LEN=4, VOCAB_SIZE=32
//!
//! Part of #4225: Compose tests for GLM-OCR full decoder pipeline bounds.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorNodeId};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- symbolic (small for fast verification), structurally
// representative of GLM-OCR (production: hidden=1536, FFN=8960,
// heads=12, KV_heads=2, head_dim=128)
// ---------------------------------------------------------------------------

const HIDDEN_DIM: usize = 8;
const FFN_DIM: usize = 16;
const NUM_HEADS: usize = 2;
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 4
const SEQ_LEN: usize = 4;
const VOCAB_SIZE: usize = 32;
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Constant weight tensor binding with small magnitude.
fn weight(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
}

/// Ones tensor binding (for RMSNorm weight parameter).
fn ones(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 1.0f32))
}

/// Scalar epsilon binding for normalization layers.
fn eps_binding() -> TensorParamBinding {
    TensorParamBinding::ConstantScalar(1e-5)
}

/// Sequence-domain input bounds: embeddings in [-range, +range].
fn seq_bounds(seq_len: usize, dim: usize, range: f32) -> BoundedTensor {
    uniform_bounds(&[seq_len, dim], range)
}

/// RoPE cos/sin positional encoding table bounded in [-1, 1].
fn rope_cos_sin(seq_len: usize, head_dim: usize) -> ArrayD<f32> {
    let mut data = vec![0.0f32; seq_len * head_dim];
    for t in 0..seq_len {
        for i in 0..head_dim / 2 {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * i as f64 / head_dim as f64);
            data[t * head_dim + 2 * i] = freq.cos() as f32;
            data[t * head_dim + 2 * i + 1] = freq.sin() as f32;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[seq_len, head_dim]), data).expect("valid RoPE table")
}

/// Add a SwiGLU FFN sub-block to a builder.
///
/// gate_proj -> SiLU * up_proj -> down_proj
/// Input/output: [seq_len, hidden_dim]. Returns output node.
fn add_swiglu_ffn(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    prefix: &str,
) -> TensorNodeId {
    let ffn_shape = [seq_len, ffn_dim];
    let out_shape = [seq_len, hidden_dim];

    let gate_w = b.add_input(&format!("{prefix}_gate_w"), &[ffn_dim, hidden_dim]);
    let up_w = b.add_input(&format!("{prefix}_up_w"), &[ffn_dim, hidden_dim]);
    let down_w = b.add_input(&format!("{prefix}_down_w"), &[hidden_dim, ffn_dim]);

    // Gate branch: gate_proj -> SiLU(x) = x * sigmoid(x)
    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);

    // Up branch
    let up = b.add_linear(input, up_w, None, &ffn_shape);

    // Multiplicative gating + down projection
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    b.add_linear(hidden, down_w, None, &out_shape)
}

/// Push SwiGLU FFN bindings (3 params: gate_w, up_w, down_w).
fn push_swiglu_bindings(bindings: &mut Vec<TensorParamBinding>, hidden_dim: usize, ffn_dim: usize) {
    bindings.push(weight(&[ffn_dim, hidden_dim])); // gate_w
    bindings.push(weight(&[ffn_dim, hidden_dim])); // up_w
    bindings.push(weight(&[hidden_dim, ffn_dim])); // down_w
}

/// Add a single GLM-OCR decoder block to a builder.
///
/// RMSNorm -> Attention -> residual -> RMSNorm -> SwiGLU FFN -> residual
/// Input/output: [seq_len, hidden_dim]. Returns output node.
fn add_decoder_block(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    prefix: &str,
) -> TensorNodeId {
    let shape = [seq_len, hidden_dim];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Pre-attention RMSNorm
    let n1_eps = b.add_input(&format!("{prefix}_n1_eps"), &[1]);
    let n1_w = b.add_input(&format!("{prefix}_n1_w"), &[hidden_dim]);
    let normed1 = b.add_rms_norm(input, n1_eps, 1, n1_w, &shape);

    // Self-attention (Q/K/V + attention + output projection)
    let q_w = b.add_input(&format!("{prefix}_q_w"), &[hidden_dim, hidden_dim]);
    let k_w = b.add_input(&format!("{prefix}_k_w"), &[hidden_dim, hidden_dim]);
    let v_w = b.add_input(&format!("{prefix}_v_w"), &[hidden_dim, hidden_dim]);
    let o_w = b.add_input(&format!("{prefix}_o_w"), &[hidden_dim, hidden_dim]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
    let attn_proj = b.add_linear(attn, o_w, None, &shape);

    // Residual after attention
    let res1 = b.add_binary_add(input, attn_proj, &shape);

    // Pre-FFN RMSNorm
    let n2_eps = b.add_input(&format!("{prefix}_n2_eps"), &[1]);
    let n2_w = b.add_input(&format!("{prefix}_n2_w"), &[hidden_dim]);
    let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &shape);

    // SwiGLU FFN
    let ffn_out = add_swiglu_ffn(b, normed2, seq_len, hidden_dim, ffn_dim, prefix);

    // Residual after FFN
    b.add_binary_add(res1, ffn_out, &shape)
}

/// Push decoder block bindings (12 params: 2 RMSNorm + 4 attention + 3 SwiGLU + eps*2 + w*2).
fn push_decoder_block_bindings(
    bindings: &mut Vec<TensorParamBinding>,
    hidden_dim: usize,
    ffn_dim: usize,
) {
    // RMSNorm 1: eps, weight
    bindings.push(eps_binding());
    bindings.push(ones(&[hidden_dim]));
    // Attention: Q, K, V, O weights
    bindings.push(weight(&[hidden_dim, hidden_dim]));
    bindings.push(weight(&[hidden_dim, hidden_dim]));
    bindings.push(weight(&[hidden_dim, hidden_dim]));
    bindings.push(weight(&[hidden_dim, hidden_dim]));
    // RMSNorm 2: eps, weight
    bindings.push(eps_binding());
    bindings.push(ones(&[hidden_dim]));
    // SwiGLU FFN: gate_w, up_w, down_w
    push_swiglu_bindings(bindings, hidden_dim, ffn_dim);
}

// ===========================================================================
// 1. Token + position embedding bound propagation (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_decoder_embedding_bounds() {
    // Token embedding + position embedding: two Linear projections summed.
    // Simulates: embed(token_ids) + position_encoding -> bounded hidden states.
    let mut b = TensorBlockBuilder::new("glm_ocr_dec_embed");
    let input = b.add_input("token_embed", &[SEQ_LEN, HIDDEN_DIM]);
    let pos_w = b.add_input("pos_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    // Position embedding: Linear projection of input (proxy for learned PE)
    let pos_embed = b.add_linear(input, pos_w, None, &[SEQ_LEN, HIDDEN_DIM]);

    // Sum token + position embeddings
    let out = b.add_binary_add(input, pos_embed, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid embedding kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR decoder embedding IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 2. RMSNorm bound contraction (IBP + CROWN)
// ===========================================================================

#[test]
fn test_glm_ocr_rms_norm_bounds() {
    // RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    // GLM uses RMSNorm instead of LayerNorm in all decoder layers.
    let mut b = TensorBlockBuilder::new("glm_ocr_dec_rmsnorm");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let w = b.add_input("weight", &[HIDDEN_DIM]);

    let out = b.add_rms_norm(input, eps, 1, w, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid RMSNorm kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        eps_binding(),
        ones(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP pass
    let ibp_output = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through RMSNorm");
    assert_bounds_valid(&ibp_output);
    assert_eq!(ibp_output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("GLM-OCR RMSNorm IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());

    // CROWN pass — should be at least as tight as IBP
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    assert_eq!(crown_output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("GLM-OCR RMSNorm CROWN: method={method:?}, bounds=[{clo:.6}, {chi:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 3. SwiGLU FFN bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_glm_ocr_swiglu_ffn_bounds() {
    // SwiGLU: gate_proj -> SiLU -> mul(up_proj) -> down_proj
    // GLM-4/5 FFN with gated linear units.
    let mut b = TensorBlockBuilder::new("glm_ocr_dec_swiglu");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    let out = add_swiglu_ffn(&mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, "ffn");
    let def = b.build(out).expect("valid SwiGLU kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP pass
    let ibp_output = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through SwiGLU");
    assert_bounds_valid(&ibp_output);
    assert_eq!(ibp_output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("GLM-OCR SwiGLU IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());

    // CROWN pass
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    assert_eq!(crown_output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("GLM-OCR SwiGLU CROWN: method={method:?}, bounds=[{clo:.6}, {chi:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 4. RoPE-enhanced attention score bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_rope_attention_bounds() {
    // RoPE rotation applied to Q/K before attention scoring.
    // Simplified: Q_proj -> mul(cos) + mul(sin) -> attention with K/V
    let mut b = TensorBlockBuilder::new("glm_ocr_dec_rope_attn");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let cos_table = b.add_input("cos_table", &[SEQ_LEN, HIDDEN_DIM]);
    let sin_table = b.add_input("sin_table", &[SEQ_LEN, HIDDEN_DIM]);

    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Q projection + RoPE rotation (simplified)
    let q = b.add_linear(input, q_w, None, &shape);
    let q_cos = b.add_binary_mul(q, cos_table, &shape);
    let q_sin = b.add_binary_mul(q, sin_table, &shape);
    let q_rope = b.add_binary_add(q_cos, q_sin, &shape);

    // K projection + RoPE rotation (simplified)
    let k = b.add_linear(input, k_w, None, &shape);
    let k_cos = b.add_binary_mul(k, cos_table, &shape);
    let k_sin = b.add_binary_mul(k, sin_table, &shape);
    let k_rope = b.add_binary_add(k_cos, k_sin, &shape);

    // V projection (no RoPE)
    let v = b.add_linear(input, v_w, None, &shape);

    // Attention: softmax(Q_rope @ K_rope^T / sqrt(d)) @ V
    let out = b.add_attention(
        q_rope,
        k_rope,
        v,
        AttentionMask::Causal,
        Some(scale),
        &shape,
    );
    let def = b.build(out).expect("valid RoPE attention kernel");

    let cos_data = rope_cos_sin(SEQ_LEN, HIDDEN_DIM);
    let sin_data = rope_cos_sin(SEQ_LEN, HIDDEN_DIM);
    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // q_w
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // k_w
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // v_w
        TensorParamBinding::ConstantTensor(cos_data),
        TensorParamBinding::ConstantTensor(sin_data),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR RoPE attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 5. Single causal decoder block compose (IBP + CROWN)
// ===========================================================================

#[test]
fn test_glm_ocr_causal_decoder_block_bounds() {
    // Full decoder block: RMSNorm -> Attention -> residual -> RMSNorm -> SwiGLU -> residual
    let mut b = TensorBlockBuilder::new("glm_ocr_dec_block");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    let out = add_decoder_block(&mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, "blk0");
    let def = b.build(out).expect("valid decoder block kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_decoder_block_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP pass
    let ibp_output = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through decoder block");
    assert_bounds_valid(&ibp_output);
    assert_eq!(ibp_output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("GLM-OCR decoder block IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());

    // CROWN pass
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    assert_eq!(crown_output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("GLM-OCR decoder block CROWN: method={method:?}, bounds=[{clo:.6}, {chi:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 6. Two decoder blocks composed (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_two_layer_stack_bounds() {
    // Two decoder blocks stacked: block0 -> block1
    // Verifies bounds widening through depth is monotonic and finite.
    let mut b = TensorBlockBuilder::new("glm_ocr_dec_2layer");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    let block0_out = add_decoder_block(&mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, "blk0");
    let out = add_decoder_block(&mut b, block0_out, SEQ_LEN, HIDDEN_DIM, FFN_DIM, "blk1");
    let def = b.build(out).expect("valid 2-layer decoder kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_decoder_block_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM); // block 0
    push_decoder_block_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM); // block 1
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through 2-layer decoder");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR 2-layer decoder IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());

    // Verify 2-layer bounds are wider than 1-layer (monotonic widening)
    let graph_1layer = {
        let mut b1 = TensorBlockBuilder::new("glm_ocr_dec_1layer_ref");
        let inp = b1.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
        let out1 = add_decoder_block(&mut b1, inp, SEQ_LEN, HIDDEN_DIM, FFN_DIM, "blk0");
        let def1 = b1.build(out1).expect("valid 1-layer kernel");
        let mut bindings1 = vec![TensorParamBinding::Variable];
        push_decoder_block_bindings(&mut bindings1, HIDDEN_DIM, FFN_DIM);
        tensor_kernel_to_graph(&def1, &bindings1).expect("graph translation")
    };
    let output_1layer = graph_1layer
        .propagate_ibp(&input_bounds)
        .expect("IBP 1-layer");
    let (lo1, hi1) = bounds_min_max(&output_1layer);
    let width_1 = hi1 - lo1;
    let width_2 = hi_max - lo_min;
    eprintln!("GLM-OCR widening: 1-layer width={width_1:.6}, 2-layer width={width_2:.6}");
    // 2-layer should be at least as wide as 1-layer
    assert!(
        width_2 >= width_1 - 1e-4,
        "2-layer bounds should be at least as wide as 1-layer"
    );
}

// ===========================================================================
// 7. Final vocabulary projection bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_vocabulary_projection_bounds() {
    // LM head: RMSNorm -> Linear(hidden_dim, vocab_size) -> logits
    // Output bounds are unbounded logits (not softmax).
    let mut b = TensorBlockBuilder::new("glm_ocr_dec_vocab_proj");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);

    // Final RMSNorm
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &[SEQ_LEN, HIDDEN_DIM]);

    // Linear projection to vocabulary
    let out = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid vocab projection kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        eps_binding(),
        ones(&[HIDDEN_DIM]),
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through vocab projection");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR vocab projection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 8. End-to-end mini pipeline (IBP + CROWN)
// ===========================================================================

#[test]
fn test_glm_ocr_full_pipeline_bounds() {
    // Full mini pipeline: embedding -> decoder block -> RMSNorm -> LM head -> softmax
    // End-to-end bounds propagation through a minimal GLM-OCR decoder.
    let mut b = TensorBlockBuilder::new("glm_ocr_dec_full_pipeline");
    let input = b.add_input("token_embed", &[SEQ_LEN, HIDDEN_DIM]);

    // Position embedding (simplified: linear + add)
    let pos_w = b.add_input("pos_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let pos_embed = b.add_linear(input, pos_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let embedded = b.add_binary_add(input, pos_embed, &[SEQ_LEN, HIDDEN_DIM]);

    // Single decoder block
    let decoded = add_decoder_block(&mut b, embedded, SEQ_LEN, HIDDEN_DIM, FFN_DIM, "blk0");

    // Final RMSNorm
    let final_eps = b.add_input("final_eps", &[1]);
    let final_w = b.add_input("final_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(decoded, final_eps, 1, final_w, &[SEQ_LEN, HIDDEN_DIM]);

    // LM head projection
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);

    // Softmax output distribution
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid full pipeline kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // pos_w
    ];
    push_decoder_block_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM); // decoder block
    bindings.push(eps_binding()); // final_eps
    bindings.push(ones(&[HIDDEN_DIM])); // final_w
    bindings.push(weight(&[VOCAB_SIZE, HIDDEN_DIM])); // lm_w

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP pass
    let ibp_output = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through full pipeline");
    assert_bounds_valid(&ibp_output);
    assert_eq!(ibp_output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("GLM-OCR full pipeline IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());

    // Softmax output should be in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "softmax lower bound should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "softmax upper bound should be <= 1, got {hi_max}"
    );

    // CROWN pass
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    assert_eq!(crown_output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("GLM-OCR full pipeline CROWN: method={method:?}, bounds=[{clo:.6}, {chi:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}
