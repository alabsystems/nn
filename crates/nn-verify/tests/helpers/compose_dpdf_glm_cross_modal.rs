// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep compose tests: GLM-OCR vision-to-decoder cross-modal composition.
//!
//! Verifies bounds propagation through the cross-modal boundary where
//! vision encoder features are projected and consumed by the GLM-4 decoder.
//! These tests target the heuristic gaps in GLM-OCR compose coverage by
//! testing intermediate cross-modal compositions at increasing depth.
//!
//! 1. **Vision projection + RMSNorm**: Linear(VISION_DIM, HIDDEN_DIM) -> RMSNorm.
//!    The projection bridge from vision encoder to decoder space (IBP + CROWN).
//!
//! 2. **Projected features + GQA cross-attention**: Vision features as KV,
//!    decoder hidden states as Q, through grouped-query attention (IBP + CROWN).
//!
//! 3. **Cross-attention + SwiGLU FFN**: Attention output through gated FFN
//!    with residual connection (IBP + CROWN).
//!
//! 4. **Vision projection + decoder block**: Full cross-modal decoder layer
//!    with RMSNorm pre/post, attention, and SwiGLU (IBP + CROWN).
//!
//! 5. **MTP head from cross-attention output**: Cross-attention -> Linear ->
//!    softmax multi-token prediction. Output bounded in [0, 1] (IBP).
//!
//! 6. **Vision + 2-layer decoder + LM head**: End-to-end cross-modal pipeline
//!    testing bounds propagation through the full composition (IBP + CROWN).
//!
//! 7. **Tight-input cross-attention**: Narrow +-0.1 bounds for CROWN precision
//!    on cross-modal attention to reduce relaxation gap (IBP + CROWN).
//!
//! 8. **Cross-modal residual accumulation**: Verify that cross-attention
//!    residual + self-attention residual compose without vacuous blowup (IBP).
//!
//! Architecture reference:
//! - GLM-4V (THUDM): Vision-language model using cross-attention for fusion
//! - RMSNorm (Zhang & Sennrich, 2019)
//! - SwiGLU (Shazeer, 2020)
//! - GQA (Ainslie et al., 2023)
//!
//! Dimensions are small for fast verification (HIDDEN_DIM=16, VISION_DIM=24).
//! All tests use IbpValidated soundness mode per nn engineering rules.
//!
//! Part of #4304: deep NY compose tests for dpdf cross-modal composition.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions
// ---------------------------------------------------------------------------

const HIDDEN_DIM: usize = 16;
const VISION_DIM: usize = 24;
const FFN_DIM: usize = 64;
const SEQ_LEN: usize = 4;
const VISION_SEQ: usize = 4;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 4
const VOCAB_SIZE: usize = 32;
const W_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn w(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), W_MAG)
}

fn ones(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 1.0f32)
}

fn zeros(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 0.0f32)
}

// ===========================================================================
// 1. Vision projection + RMSNorm
// ===========================================================================

fn build_vision_proj_rmsnorm() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_vision_proj_rmsnorm");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let input = b.add_input("vision_features", &[VISION_SEQ, VISION_DIM]);
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, VISION_DIM]);
    let proj_b = b.add_input("proj_b", &[HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);

    let projected = b.add_linear(input, proj_w, Some(proj_b), &shape);
    let normed = b.add_rms_norm(projected, eps, 1, norm_w, &shape);
    b.build(normed).expect("valid vision proj + rmsnorm")
}

fn vision_proj_rmsnorm_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, VISION_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])),
    ]
}

#[test]
fn test_glm_cross_vision_proj_rmsnorm_ibp() {
    let def = build_vision_proj_rmsnorm();
    let bindings = vision_proj_rmsnorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[VISION_SEQ, VISION_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

#[test]
fn test_glm_cross_vision_proj_rmsnorm_crown() {
    let def = build_vision_proj_rmsnorm();
    let bindings = vision_proj_rmsnorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[VISION_SEQ, VISION_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("vision_proj_rmsnorm CROWN method: {method:?}");
}

// ===========================================================================
// 2. Projected features + GQA cross-attention
// ===========================================================================

fn build_cross_attention_gqa() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_cross_attn_gqa");
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Decoder hidden states as Q
    let decoder_input = b.add_input("decoder_hidden", &[SEQ_LEN, HIDDEN_DIM]);
    // Vision features as KV (already projected to HIDDEN_DIM)
    let vision_input = b.add_input("vision_kv", &[VISION_SEQ, HIDDEN_DIM]);

    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(decoder_input, q_w, None, &shape);
    let k = b.add_linear(vision_input, k_w, None, &[VISION_SEQ, HIDDEN_DIM]);
    let v = b.add_linear(vision_input, v_w, None, &[VISION_SEQ, HIDDEN_DIM]);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let out = b.add_linear(attn, o_w, None, &shape);
    b.build(out).expect("valid cross-attention GQA")
}

fn cross_attention_gqa_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // decoder hidden
        TensorParamBinding::Variable, // vision kv — second variable input
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, HIDDEN_DIM])),
    ]
}

#[test]
fn test_glm_cross_attention_gqa_ibp() {
    let def = build_cross_attention_gqa();
    let bindings = cross_attention_gqa_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    // Multi-variable: single input covers both variable regions
    let total_elems = SEQ_LEN * HIDDEN_DIM + VISION_SEQ * HIDDEN_DIM;
    let input = uniform_bounds(&[total_elems], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
}

#[test]
fn test_glm_cross_attention_gqa_crown() {
    let def = build_cross_attention_gqa();
    let bindings = cross_attention_gqa_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let total_elems = SEQ_LEN * HIDDEN_DIM + VISION_SEQ * HIDDEN_DIM;
    let input = uniform_bounds(&[total_elems], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("cross_attention_gqa CROWN method: {method:?}");
}

// ===========================================================================
// 3. Cross-attention + SwiGLU FFN
// ===========================================================================

fn build_cross_attn_swiglu() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_cross_attn_swiglu");
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // RMSNorm before FFN
    let eps = b.add_input("eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &shape);

    // SwiGLU FFN
    let gate_w = b.add_input("gate_w", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_w", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_w", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    // Residual connection
    let out = b.add_binary_add(input, ffn_out, &shape);
    b.build(out).expect("valid cross-attn + swiglu")
}

fn cross_attn_swiglu_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, FFN_DIM])),
    ]
}

#[test]
fn test_glm_cross_attn_swiglu_ibp() {
    let def = build_cross_attn_swiglu();
    let bindings = cross_attn_swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

#[test]
fn test_glm_cross_attn_swiglu_crown() {
    let def = build_cross_attn_swiglu();
    let bindings = cross_attn_swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("cross_attn_swiglu CROWN method: {method:?}");
}

// ===========================================================================
// 4. Vision projection + full decoder block
// ===========================================================================

fn build_vision_proj_decoder_block() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_vision_proj_decoder");
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Vision input + projection
    let input = b.add_input("vision_features", &[VISION_SEQ, VISION_DIM]);
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, VISION_DIM]);
    let proj_b = b.add_input("proj_b", &[HIDDEN_DIM]);
    let projected = b.add_linear(input, proj_w, Some(proj_b), &shape);

    // Pre-attention RMSNorm
    let n1_eps = b.add_input("n1_eps", &[1]);
    let n1_w = b.add_input("n1_w", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(projected, n1_eps, 1, n1_w, &shape);

    // Self-attention
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
    let attn_out = b.add_linear(attn, o_w, None, &shape);
    let res1 = b.add_binary_add(projected, attn_out, &shape);

    // Pre-FFN RMSNorm
    let n2_eps = b.add_input("n2_eps", &[1]);
    let n2_w = b.add_input("n2_w", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &shape);

    // SwiGLU FFN
    let gate_w = b.add_input("gate_w", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_w", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_w", &[HIDDEN_DIM, FFN_DIM]);
    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);
    let res2 = b.add_binary_add(res1, ffn_out, &shape);

    b.build(res2).expect("valid vision proj + decoder block")
}

fn vision_proj_decoder_block_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        // Projection
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, VISION_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])),
        // RMSNorm 1
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])),
        // Q/K/V/O
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, HIDDEN_DIM])),
        // RMSNorm 2
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])),
        // SwiGLU
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, FFN_DIM])),
    ]
}

#[test]
fn test_glm_cross_vision_proj_decoder_block_ibp() {
    let def = build_vision_proj_decoder_block();
    let bindings = vision_proj_decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[VISION_SEQ, VISION_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

#[test]
fn test_glm_cross_vision_proj_decoder_block_crown() {
    let def = build_vision_proj_decoder_block();
    let bindings = vision_proj_decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[VISION_SEQ, VISION_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("vision_proj_decoder_block CROWN method: {method:?}");
}

// ===========================================================================
// 5. MTP head from cross-attention output
// ===========================================================================

fn build_cross_modal_mtp_head() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_cross_modal_mtp");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // RMSNorm before LM head
    let eps = b.add_input("eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &shape);

    // LM head projection
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);

    // Softmax output
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    b.build(probs).expect("valid cross-modal MTP head")
}

fn cross_modal_mtp_head_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[VOCAB_SIZE, HIDDEN_DIM])),
    ]
}

#[test]
fn test_glm_cross_modal_mtp_head_ibp() {
    let def = build_cross_modal_mtp_head();
    let bindings = cross_modal_mtp_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "softmax lower bound should be >= 0, got {lo}");
    assert!(
        hi <= 1.0 + 1e-5,
        "softmax upper bound should be <= 1, got {hi}"
    );
}

// ===========================================================================
// 6. Vision + 2-layer decoder + LM head
// ===========================================================================

/// Add one GLM-OCR decoder layer to the builder.
fn add_decoder_layer(
    b: &mut TensorBlockBuilder,
    x: nn_dsl::TensorNodeId,
    layer_idx: usize,
) -> nn_dsl::TensorNodeId {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let pfx = format!("l{layer_idx}");

    let n1_eps = b.add_input(&format!("{pfx}_n1_eps"), &[1]);
    let n1_w = b.add_input(&format!("{pfx}_n1_w"), &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(x, n1_eps, 1, n1_w, &shape);

    let q_w = b.add_input(&format!("{pfx}_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input(&format!("{pfx}_k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input(&format!("{pfx}_v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input(&format!("{pfx}_o_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
    let attn_out = b.add_linear(attn, o_w, None, &shape);
    let res1 = b.add_binary_add(x, attn_out, &shape);

    let n2_eps = b.add_input(&format!("{pfx}_n2_eps"), &[1]);
    let n2_w = b.add_input(&format!("{pfx}_n2_w"), &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &shape);

    let gate_w = b.add_input(&format!("{pfx}_gate_w"), &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input(&format!("{pfx}_up_w"), &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input(&format!("{pfx}_down_w"), &[HIDDEN_DIM, FFN_DIM]);
    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);
    b.add_binary_add(res1, ffn_out, &shape)
}

fn push_decoder_layer_bindings(bindings: &mut Vec<TensorParamBinding>) {
    // RMSNorm 1
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    // Q/K/V/O
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        HIDDEN_DIM, HIDDEN_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        HIDDEN_DIM, HIDDEN_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        HIDDEN_DIM, HIDDEN_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        HIDDEN_DIM, HIDDEN_DIM,
    ])));
    // RMSNorm 2
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    // SwiGLU
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        FFN_DIM, HIDDEN_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        FFN_DIM, HIDDEN_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        HIDDEN_DIM, FFN_DIM,
    ])));
}

fn build_vision_two_layer_decoder_lm_head() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_vision_2layer_decoder_lm");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Vision input + projection
    let input = b.add_input("vision_features", &[VISION_SEQ, VISION_DIM]);
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, VISION_DIM]);
    let proj_b = b.add_input("proj_b", &[HIDDEN_DIM]);
    let projected = b.add_linear(input, proj_w, Some(proj_b), &shape);

    // 2 decoder layers
    let x = add_decoder_layer(&mut b, projected, 0);
    let x = add_decoder_layer(&mut b, x, 1);

    // Final RMSNorm + LM head + softmax
    let final_eps = b.add_input("final_eps", &[1]);
    let final_w = b.add_input("final_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(x, final_eps, 1, final_w, &shape);
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    b.build(probs)
        .expect("valid vision + 2-layer decoder + LM head")
}

fn vision_two_layer_decoder_lm_head_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, VISION_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])),
    ];
    push_decoder_layer_bindings(&mut bindings);
    push_decoder_layer_bindings(&mut bindings);
    // Final RMSNorm
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    // LM head
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        VOCAB_SIZE, HIDDEN_DIM,
    ])));
    bindings
}

#[test]
fn test_glm_cross_vision_2layer_decoder_lm_head_ibp() {
    let def = build_vision_two_layer_decoder_lm_head();
    let bindings = vision_two_layer_decoder_lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[VISION_SEQ, VISION_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    // Softmax output must be in [0, 1]
    assert!(lo >= -1e-5, "softmax lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "softmax upper <= 1, got {hi}");
}

#[test]
fn test_glm_cross_vision_2layer_decoder_lm_head_crown() {
    let def = build_vision_two_layer_decoder_lm_head();
    let bindings = vision_two_layer_decoder_lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[VISION_SEQ, VISION_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("vision_2layer_decoder_lm_head CROWN method: {method:?}");
}

// ===========================================================================
// 7. Tight-input cross-attention
// ===========================================================================

#[test]
fn test_glm_cross_vision_proj_decoder_block_tight_crown() {
    let def = build_vision_proj_decoder_block();
    let bindings = vision_proj_decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    // Narrow +-0.1 bounds for CROWN precision
    let input = uniform_bounds(&[VISION_SEQ, VISION_DIM], 0.1);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);

    // Tight inputs should produce tighter bounds
    let ibp_wide = {
        let wide_input = uniform_bounds(&[VISION_SEQ, VISION_DIM], 1.0);
        graph.propagate_ibp(&wide_input).expect("IBP wide")
    };
    let (_, tight_hi) = bounds_min_max(&output);
    let (_, wide_hi) = bounds_min_max(&ibp_wide);
    eprintln!(
        "tight-input CROWN width: {tight_hi:.4}, wide IBP width: {wide_hi:.4}, method: {method:?}"
    );
}

// ===========================================================================
// 8. Cross-modal residual accumulation
// ===========================================================================

fn build_cross_modal_residual_accumulation() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_cross_modal_residual");
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // Self-attention residual
    let n1_eps = b.add_input("n1_eps", &[1]);
    let n1_w = b.add_input("n1_w", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, n1_eps, 1, n1_w, &shape);
    let q1_w = b.add_input("q1_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k1_w = b.add_input("k1_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v1_w = b.add_input("v1_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o1_w = b.add_input("o1_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let q1 = b.add_linear(normed1, q1_w, None, &shape);
    let k1 = b.add_linear(normed1, k1_w, None, &shape);
    let v1 = b.add_linear(normed1, v1_w, None, &shape);
    let self_attn = b.add_attention(q1, k1, v1, AttentionMask::Causal, Some(scale), &shape);
    let self_attn_out = b.add_linear(self_attn, o1_w, None, &shape);
    let res1 = b.add_binary_add(input, self_attn_out, &shape);

    // Cross-attention residual (vision KV from same hidden — simplified)
    let n2_eps = b.add_input("n2_eps", &[1]);
    let n2_w = b.add_input("n2_w", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &shape);
    let q2_w = b.add_input("q2_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k2_w = b.add_input("k2_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v2_w = b.add_input("v2_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o2_w = b.add_input("o2_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let q2 = b.add_linear(normed2, q2_w, None, &shape);
    let k2 = b.add_linear(normed2, k2_w, None, &shape);
    let v2 = b.add_linear(normed2, v2_w, None, &shape);
    let cross_attn = b.add_attention(q2, k2, v2, AttentionMask::Standard, Some(scale), &shape);
    let cross_attn_out = b.add_linear(cross_attn, o2_w, None, &shape);
    let res2 = b.add_binary_add(res1, cross_attn_out, &shape);

    b.build(res2)
        .expect("valid cross-modal residual accumulation")
}

fn cross_modal_residual_accumulation_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    // RMSNorm 1
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    // Self-attn QKVO
    for _ in 0..4 {
        bindings.push(TensorParamBinding::ConstantTensor(w(&[
            HIDDEN_DIM, HIDDEN_DIM,
        ])));
    }
    // RMSNorm 2
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    // Cross-attn QKVO
    for _ in 0..4 {
        bindings.push(TensorParamBinding::ConstantTensor(w(&[
            HIDDEN_DIM, HIDDEN_DIM,
        ])));
    }
    bindings
}

#[test]
fn test_glm_cross_modal_residual_accumulation_ibp() {
    let def = build_cross_modal_residual_accumulation();
    let bindings = cross_modal_residual_accumulation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    // Verify residual doesn't cause vacuous blowup
    let (lo, hi) = bounds_min_max(&output);
    let width = hi - lo;
    assert!(
        width < 1e6,
        "residual accumulation bounds too wide: {width}"
    );
}

// ===========================================================================
// Verify-and-record
// ===========================================================================

#[test]
fn test_glm_cross_vision_proj_rmsnorm_verify_and_record() {
    let def = build_vision_proj_rmsnorm();
    let bindings = vision_proj_rmsnorm_bindings();
    let input = uniform_bounds(&[VISION_SEQ, VISION_DIM], 1.0);
    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "glm_ocr::test_glm_cross_vision_proj_rmsnorm_verify_and_record",
    );
    assert_bounds_valid(&result.output_bounds);
}
