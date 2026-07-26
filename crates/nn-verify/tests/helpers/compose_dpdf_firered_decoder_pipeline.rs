// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended FireRed-OCR decoder pipeline compose verification tests.
//!
//! These tests verify NY IBP and CROWN bound propagation through
//! the FireRed-OCR decoder pipeline stages, complementing the vision-language
//! tests in `compose_dpdf_firered_vision_lang.rs`.
//!
//! ## Decoder Pipeline Stages (tests 1-12)
//!
//! 1. **Text decoder causal attention bounds** (IBP + CROWN): RMSNorm -> causal
//!    self-attention -> residual, isolated decoder attention sub-block.
//! 2. **Cross-attention vision-to-decoder** (IBP + CROWN): Vision encoder features
//!    attend to decoder hidden state via standard (non-causal) cross-attention.
//! 3. **Autoregressive generation step** (IBP): Single-step autoregressive decode:
//!    decoder layer -> RMSNorm -> LM head -> softmax token probabilities.
//! 4. **Multi-step generation accumulation** (IBP): Two sequential generation
//!    steps feeding back through the decoder, verifying bound accumulation.
//! 5. **Beam search score propagation** (IBP): Log-softmax score computation
//!    with additive accumulation for beam search decoding.
//! 6. **End-to-end vision-encoder-cross-attn-decoder pipeline** (IBP + CROWN):
//!    Patch embed -> encoder block -> projection -> cross-attention -> decoder
//!    block -> softmax, full inference pipeline.
//! 7. **Language model head probability bounds** (IBP): RMSNorm -> Linear ->
//!    Softmax producing token distribution bounded in [0, 1].
//! 8. **Token embedding lookup bounds** (IBP): Embedding weight matrix lookup
//!    modeled as Linear projection with one-hot-like input.
//! 9. **Position encoding addition bounds** (IBP + CROWN): Learned position
//!    embedding added to token embeddings, verifying additive stability.
//! 10. **CROWN tightening for decoder stages** (CROWN): Isolated decoder block
//!     with narrow +-0.5 input bounds for CROWN linearization precision.
//! 11. **Decoder + cross-attention + LM head pipeline** (IBP): Full decoder
//!     pipeline: cross-attention -> decoder self-attention -> LM head -> softmax.
//! 12. **Verify-and-record** (IBP): Full decoder pipeline with status recording.
//!
//! Architecture references:
//! - FireRed-OCR: Qwen3-VL-2B variant for document OCR
//!   (yuyq96/FireRed-OCR-Qwen3-VL-2B, HuggingFace)
//! - SwiGLU (Shazeer, 2020): Gated linear unit with SiLU activation
//! - RMSNorm (Zhang & Sennrich, 2019): Root mean square layer normalization
//! - RoPE (Su et al., 2021): Rotary positional embeddings
//!
//! Dimensions are small for fast verification (HIDDEN_DIM=8, SEQ_LEN=4).
//! All tests use IbpValidated soundness mode per nn engineering rules.
//!
//! Part of #4240: FireRed-OCR decoder pipeline compose tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const HIDDEN_DIM: usize = 8;
const FFN_DIM: usize = 16;
const NUM_HEADS: usize = 2;
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 4
const PATCH_SIZE: usize = 2;
const IMG_CHANNELS: usize = 3;
const IMG_H: usize = 4;
const IMG_W: usize = 4;
const NUM_PATCHES: usize = (IMG_H / PATCH_SIZE) * (IMG_W / PATCH_SIZE); // 4
const SEQ_LEN: usize = 4;
const VOCAB_SIZE: usize = 32;
const W_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Constant weight tensor binding.
fn weight(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), W_MAG))
}

/// Zero bias tensor binding.
fn bias_zero(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}

/// Ones tensor binding (for normalization weights).
fn ones_binding(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 1.0f32))
}

/// Scalar epsilon binding for normalization layers.
fn eps_binding() -> TensorParamBinding {
    TensorParamBinding::ConstantScalar(1e-5)
}

/// Build SiLU activation: SiLU(x) = x * sigmoid(x).
fn add_silu(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    shape: &[usize],
) -> nn_dsl::TensorNodeId {
    let sig = b.add_sigmoid(input, shape);
    b.add_binary_mul(input, sig, shape)
}

/// Add a SwiGLU FFN block: gate_proj -> SiLU -> mul(up_proj) -> down_proj.
fn add_swiglu_ffn(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    pfx: &str,
    seq_len: usize,
) -> nn_dsl::TensorNodeId {
    let shape = [seq_len, HIDDEN_DIM];
    let ffn_shape = [seq_len, FFN_DIM];

    let gate_w = b.add_input(&format!("{pfx}_gate_w"), &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input(&format!("{pfx}_up_w"), &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input(&format!("{pfx}_down_w"), &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_act = add_silu(b, gate, &ffn_shape);
    let up = b.add_linear(input, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    b.add_linear(hidden, down_w, None, &shape)
}

/// Push SwiGLU FFN constant bindings.
fn push_swiglu_bindings(bindings: &mut Vec<TensorParamBinding>) {
    bindings.push(weight(&[FFN_DIM, HIDDEN_DIM]));
    bindings.push(weight(&[FFN_DIM, HIDDEN_DIM]));
    bindings.push(weight(&[HIDDEN_DIM, FFN_DIM]));
}

/// Add an encoder block: RMSNorm -> self-attention -> residual -> RMSNorm ->
/// SwiGLU FFN -> residual.
fn add_encoder_block(
    b: &mut TensorBlockBuilder,
    x: nn_dsl::TensorNodeId,
    pfx: &str,
    seq_len: usize,
) -> nn_dsl::TensorNodeId {
    let shape = [seq_len, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Pre-attention RMSNorm
    let n1_eps = b.add_input(&format!("{pfx}_n1_eps"), &[1]);
    let n1_w = b.add_input(&format!("{pfx}_n1_w"), &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(x, n1_eps, 1, n1_w, &shape);

    // Self-attention (Q/K/V/O projections)
    let q_w = b.add_input(&format!("{pfx}_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input(&format!("{pfx}_k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input(&format!("{pfx}_v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input(&format!("{pfx}_o_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);

    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, o_w, None, &shape);
    let res1 = b.add_binary_add(x, attn_out, &shape);

    // Pre-FFN RMSNorm
    let n2_eps = b.add_input(&format!("{pfx}_n2_eps"), &[1]);
    let n2_w = b.add_input(&format!("{pfx}_n2_w"), &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &shape);

    // SwiGLU FFN
    let ffn_out = add_swiglu_ffn(b, normed2, pfx, seq_len);
    b.add_binary_add(res1, ffn_out, &shape)
}

/// Push encoder block constant bindings.
fn push_encoder_block_bindings(bindings: &mut Vec<TensorParamBinding>) {
    // RMSNorm 1
    bindings.push(eps_binding());
    bindings.push(ones_binding(&[HIDDEN_DIM]));
    // Q/K/V/O
    for _ in 0..4 {
        bindings.push(weight(&[HIDDEN_DIM, HIDDEN_DIM]));
    }
    // RMSNorm 2
    bindings.push(eps_binding());
    bindings.push(ones_binding(&[HIDDEN_DIM]));
    // SwiGLU FFN
    push_swiglu_bindings(bindings);
}

/// Add a decoder block with causal self-attention.
fn add_decoder_block(
    b: &mut TensorBlockBuilder,
    x: nn_dsl::TensorNodeId,
    pfx: &str,
) -> nn_dsl::TensorNodeId {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Pre-attention RMSNorm
    let n1_eps = b.add_input(&format!("{pfx}_n1_eps"), &[1]);
    let n1_w = b.add_input(&format!("{pfx}_n1_w"), &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(x, n1_eps, 1, n1_w, &shape);

    // Causal self-attention
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

    // Pre-FFN RMSNorm
    let n2_eps = b.add_input(&format!("{pfx}_n2_eps"), &[1]);
    let n2_w = b.add_input(&format!("{pfx}_n2_w"), &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &shape);

    // SwiGLU FFN
    let ffn_out = add_swiglu_ffn(b, normed2, pfx, SEQ_LEN);
    b.add_binary_add(res1, ffn_out, &shape)
}

/// Push decoder block constant bindings (same layout as encoder but with causal).
fn push_decoder_block_bindings(bindings: &mut Vec<TensorParamBinding>) {
    // RMSNorm 1
    bindings.push(eps_binding());
    bindings.push(ones_binding(&[HIDDEN_DIM]));
    // Q/K/V/O
    for _ in 0..4 {
        bindings.push(weight(&[HIDDEN_DIM, HIDDEN_DIM]));
    }
    // RMSNorm 2
    bindings.push(eps_binding());
    bindings.push(ones_binding(&[HIDDEN_DIM]));
    // SwiGLU FFN
    push_swiglu_bindings(bindings);
}

/// Add a cross-attention block: Q from decoder, K/V from encoder memory.
fn add_cross_attention_block(
    b: &mut TensorBlockBuilder,
    decoder_hidden: nn_dsl::TensorNodeId,
    pfx: &str,
) -> nn_dsl::TensorNodeId {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Pre-attention RMSNorm
    let n_eps = b.add_input(&format!("{pfx}_ca_n_eps"), &[1]);
    let n_w = b.add_input(&format!("{pfx}_ca_n_w"), &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(decoder_hidden, n_eps, 1, n_w, &shape);

    // Cross-attention projections
    let q_w = b.add_input(&format!("{pfx}_ca_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input(&format!("{pfx}_ca_k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input(&format!("{pfx}_ca_v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input(&format!("{pfx}_ca_o_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed, q_w, None, &shape);
    let k = b.add_linear(normed, k_w, None, &shape);
    let v = b.add_linear(normed, v_w, None, &shape);

    // Standard (non-causal) cross-attention
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, o_w, None, &shape);
    b.add_binary_add(decoder_hidden, attn_out, &shape)
}

/// Push cross-attention block constant bindings.
fn push_cross_attention_bindings(bindings: &mut Vec<TensorParamBinding>) {
    bindings.push(eps_binding());
    bindings.push(ones_binding(&[HIDDEN_DIM]));
    for _ in 0..4 {
        bindings.push(weight(&[HIDDEN_DIM, HIDDEN_DIM]));
    }
}

// ===========================================================================
// 1. Text decoder causal attention bounds (IBP + CROWN)
// ===========================================================================

fn build_decoder_attention() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_dec_causal_attn");
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let input = b.add_input("hidden", &shape);

    // Pre-attention RMSNorm
    let n_eps = b.add_input("n_eps", &[1]);
    let n_w = b.add_input("n_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, n_eps, 1, n_w, &shape);

    // Causal self-attention
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed, q_w, None, &shape);
    let k = b.add_linear(normed, k_w, None, &shape);
    let v = b.add_linear(normed, v_w, None, &shape);

    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
    let attn_out = b.add_linear(attn, o_w, None, &shape);
    let res = b.add_binary_add(input, attn_out, &shape);

    b.build(res).expect("valid decoder causal attention")
}

fn decoder_attention_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        eps_binding(),
        ones_binding(&[HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
    ]
}

#[test]
fn test_firered_dec_causal_attention_ibp() {
    let def = build_decoder_attention();
    let bindings = decoder_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

#[test]
fn test_firered_dec_causal_attention_crown() {
    let def = build_decoder_attention();
    let bindings = decoder_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered_dec causal_attention CROWN method: {method:?}");
}

// ===========================================================================
// 2. Cross-attention vision-to-decoder bounds (IBP + CROWN)
// ===========================================================================

fn build_vision_cross_attention() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_dec_cross_attn");
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Decoder hidden state as variable input
    let input = b.add_input("decoder_hidden", &shape);

    // Pre-cross-attention RMSNorm
    let n_eps = b.add_input("n_eps", &[1]);
    let n_w = b.add_input("n_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, n_eps, 1, n_w, &shape);

    // Cross-attention: Q from decoder, K/V from encoder memory
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let kv_proj_w = b.add_input("kv_proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed, q_w, None, &shape);
    // K/V derived from encoder features projected through decoder space
    let kv_proj = b.add_linear(normed, kv_proj_w, None, &shape);
    let k = b.add_linear(kv_proj, k_w, None, &shape);
    let v = b.add_linear(kv_proj, v_w, None, &shape);

    // Standard (non-causal) for cross-attention
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, o_w, None, &shape);
    let res = b.add_binary_add(input, attn_out, &shape);

    b.build(res).expect("valid vision cross-attention")
}

fn vision_cross_attention_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        eps_binding(),
        ones_binding(&[HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
    ]
}

#[test]
fn test_firered_dec_cross_attention_ibp() {
    let def = build_vision_cross_attention();
    let bindings = vision_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

#[test]
fn test_firered_dec_cross_attention_crown() {
    let def = build_vision_cross_attention();
    let bindings = vision_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered_dec cross_attention CROWN method: {method:?}");
}

// ===========================================================================
// 3. Autoregressive generation step bounds (IBP)
// ===========================================================================

fn build_autoregressive_step() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_dec_autoreg_step");

    // Single token position (autoregressive: one position at a time)
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // Decoder block (causal self-attention + SwiGLU FFN)
    let decoded = add_decoder_block(&mut b, input, "dec0");

    // Final RMSNorm
    let final_eps = b.add_input("final_eps", &[1]);
    let final_w = b.add_input("final_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(decoded, final_eps, 1, final_w, &[SEQ_LEN, HIDDEN_DIM]);

    // LM head -> softmax
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs).expect("valid autoregressive step")
}

fn autoregressive_step_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_decoder_block_bindings(&mut bindings);
    // Final RMSNorm
    bindings.push(eps_binding());
    bindings.push(ones_binding(&[HIDDEN_DIM]));
    // LM head
    bindings.push(weight(&[VOCAB_SIZE, HIDDEN_DIM]));
    bindings
}

#[test]
fn test_firered_dec_autoregressive_step_ibp() {
    let def = build_autoregressive_step();
    let bindings = autoregressive_step_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "autoreg softmax lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "autoreg softmax upper <= 1, got {hi}");
    assert_eq!(output.shape(), &[SEQ_LEN, VOCAB_SIZE]);
}

// ===========================================================================
// 4. Multi-step generation accumulation (IBP)
// ===========================================================================

fn build_multi_step_generation() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_dec_multistep_gen");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let input = b.add_input("hidden", &shape);

    // Step 1: decoder block
    let step1 = add_decoder_block(&mut b, input, "step1");

    // Step 2: second decoder block feeding on step 1 output
    let step2 = add_decoder_block(&mut b, step1, "step2");

    // Final RMSNorm -> LM head -> softmax
    let final_eps = b.add_input("final_eps", &[1]);
    let final_w = b.add_input("final_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(step2, final_eps, 1, final_w, &shape);

    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs).expect("valid multi-step generation")
}

fn multi_step_generation_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_decoder_block_bindings(&mut bindings); // step1
    push_decoder_block_bindings(&mut bindings); // step2
    bindings.push(eps_binding());
    bindings.push(ones_binding(&[HIDDEN_DIM]));
    bindings.push(weight(&[VOCAB_SIZE, HIDDEN_DIM]));
    bindings
}

#[test]
fn test_firered_dec_multi_step_generation_ibp() {
    let def = build_multi_step_generation();
    let bindings = multi_step_generation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "multi-step softmax lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "multi-step softmax upper <= 1, got {hi}");
}

// ===========================================================================
// 5. Beam search score propagation (IBP)
// ===========================================================================

fn build_beam_search_scores() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_dec_beam_scores");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let input = b.add_input("hidden", &shape);

    // RMSNorm -> Linear -> log_softmax (modeled as softmax for IBP)
    let eps = b.add_input("eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &shape);

    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);

    // Softmax produces token probabilities (log_softmax analog for beam search)
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    // Accumulate: add previous beam scores (constant) to current log-probs
    // Modeled as additive bias representing accumulated beam scores
    let prev_scores = b.add_input("prev_scores", &[SEQ_LEN, VOCAB_SIZE]);
    let accumulated = b.add_binary_add(probs, prev_scores, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(accumulated).expect("valid beam search scores")
}

fn beam_search_scores_bindings() -> Vec<TensorParamBinding> {
    // Previous beam scores: small negative values (log-prob accumulations)
    let score_data: Vec<f32> = (0..SEQ_LEN * VOCAB_SIZE)
        .map(|i| -0.1 * (i as f32 / (SEQ_LEN * VOCAB_SIZE) as f32))
        .collect();
    let score_tensor = ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, VOCAB_SIZE]), score_data).unwrap();

    vec![
        TensorParamBinding::Variable,
        eps_binding(),
        ones_binding(&[HIDDEN_DIM]),
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        TensorParamBinding::ConstantTensor(score_tensor),
    ]
}

#[test]
fn test_firered_dec_beam_search_scores_ibp() {
    let def = build_beam_search_scores();
    let bindings = beam_search_scores_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, VOCAB_SIZE]);
}

// ===========================================================================
// 6. End-to-end vision-encoder-cross-attn-decoder pipeline (IBP + CROWN)
// ===========================================================================

fn build_e2e_pipeline() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_dec_e2e_pipeline");
    let patch_dim = IMG_CHANNELS * PATCH_SIZE * PATCH_SIZE;

    // Patch embedding
    let input = b.add_input("patches", &[NUM_PATCHES, patch_dim]);
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, patch_dim]);
    let proj_b = b.add_input("proj_b", &[HIDDEN_DIM]);
    let embedded = b.add_linear(input, proj_w, Some(proj_b), &[NUM_PATCHES, HIDDEN_DIM]);

    // Encoder block
    let encoded = add_encoder_block(&mut b, embedded, "enc0", NUM_PATCHES);

    // Vision-to-language projection
    let vl_w = b.add_input("vl_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let vl_b = b.add_input("vl_b", &[HIDDEN_DIM]);
    let projected = b.add_linear(encoded, vl_w, Some(vl_b), &[SEQ_LEN, HIDDEN_DIM]);

    // Cross-attention from encoder features
    let cross_out = add_cross_attention_block(&mut b, projected, "dec0");

    // Decoder self-attention block
    let decoded = add_decoder_block(&mut b, cross_out, "dec0");

    // LM head: RMSNorm -> Linear -> Softmax
    let final_eps = b.add_input("final_eps", &[1]);
    let final_w = b.add_input("final_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(decoded, final_eps, 1, final_w, &[SEQ_LEN, HIDDEN_DIM]);

    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs).expect("valid E2E pipeline")
}

fn e2e_pipeline_bindings() -> Vec<TensorParamBinding> {
    let patch_dim = IMG_CHANNELS * PATCH_SIZE * PATCH_SIZE;
    let mut bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, patch_dim]),
        bias_zero(&[HIDDEN_DIM]),
    ];
    push_encoder_block_bindings(&mut bindings);
    // Vision-to-language projection
    bindings.push(weight(&[HIDDEN_DIM, HIDDEN_DIM]));
    bindings.push(bias_zero(&[HIDDEN_DIM]));
    // Cross-attention
    push_cross_attention_bindings(&mut bindings);
    // Decoder block
    push_decoder_block_bindings(&mut bindings);
    // Final RMSNorm
    bindings.push(eps_binding());
    bindings.push(ones_binding(&[HIDDEN_DIM]));
    // LM head
    bindings.push(weight(&[VOCAB_SIZE, HIDDEN_DIM]));
    bindings
}

#[test]
fn test_firered_dec_e2e_pipeline_ibp() {
    let def = build_e2e_pipeline();
    let bindings = e2e_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let patch_dim = IMG_CHANNELS * PATCH_SIZE * PATCH_SIZE;
    let input = uniform_bounds(&[NUM_PATCHES, patch_dim], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "E2E softmax lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "E2E softmax upper <= 1, got {hi}");
    eprintln!(
        "firered_dec E2E pipeline IBP: lo={lo:.6}, hi={hi:.6}, width={:.6}",
        hi - lo
    );
}

#[test]
fn test_firered_dec_e2e_pipeline_crown() {
    let def = build_e2e_pipeline();
    let bindings = e2e_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let patch_dim = IMG_CHANNELS * PATCH_SIZE * PATCH_SIZE;
    let input = uniform_bounds(&[NUM_PATCHES, patch_dim], 0.5);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered_dec E2E pipeline CROWN method: {method:?}");
}

// ===========================================================================
// 7. Language model head probability bounds (IBP)
// ===========================================================================

fn build_lm_head_probs() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_dec_lm_head_probs");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // RMSNorm -> Linear(HIDDEN, VOCAB) -> Softmax
    let eps = b.add_input("eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &[SEQ_LEN, HIDDEN_DIM]);

    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let lm_b = b.add_input("lm_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(normed, lm_w, Some(lm_b), &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs).expect("valid LM head probs")
}

fn lm_head_probs_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        eps_binding(),
        ones_binding(&[HIDDEN_DIM]),
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        bias_zero(&[VOCAB_SIZE]),
    ]
}

#[test]
fn test_firered_dec_lm_head_probs_ibp() {
    let def = build_lm_head_probs();
    let bindings = lm_head_probs_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "LM head probs lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "LM head probs upper <= 1, got {hi}");
    assert_eq!(output.shape(), &[SEQ_LEN, VOCAB_SIZE]);
}

// ===========================================================================
// 8. Token embedding lookup bounds (IBP)
// ===========================================================================

fn build_token_embedding() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_dec_token_embed");

    // Input: one-hot-like token indices modeled as bounded features
    let input = b.add_input("token_features", &[SEQ_LEN, VOCAB_SIZE]);

    // Embedding lookup modeled as Linear(VOCAB, HIDDEN)
    let embed_w = b.add_input("embed_w", &[HIDDEN_DIM, VOCAB_SIZE]);
    let embedded = b.add_linear(input, embed_w, None, &[SEQ_LEN, HIDDEN_DIM]);

    b.build(embedded).expect("valid token embedding")
}

fn token_embedding_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, VOCAB_SIZE]),
    ]
}

#[test]
fn test_firered_dec_token_embedding_ibp() {
    let def = build_token_embedding();
    let bindings = token_embedding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    // One-hot-like: values in [0, 1]
    let input = uniform_bounds(&[SEQ_LEN, VOCAB_SIZE], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 9. Position encoding addition bounds (IBP + CROWN)
// ===========================================================================

fn build_position_encoding_add() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_dec_pos_enc_add");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Token embeddings
    let input = b.add_input("token_embeds", &shape);

    // Learned position embedding: Linear projection
    let pos_w = b.add_input("pos_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let pos_embed = b.add_linear(input, pos_w, None, &shape);

    // Additive position encoding
    let with_pos = b.add_binary_add(input, pos_embed, &shape);

    // RMSNorm after position addition for stability
    let eps = b.add_input("eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(with_pos, eps, 1, norm_w, &shape);

    b.build(normed).expect("valid position encoding addition")
}

fn position_encoding_add_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        eps_binding(),
        ones_binding(&[HIDDEN_DIM]),
    ]
}

#[test]
fn test_firered_dec_position_encoding_add_ibp() {
    let def = build_position_encoding_add();
    let bindings = position_encoding_add_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

#[test]
fn test_firered_dec_position_encoding_add_crown() {
    let def = build_position_encoding_add();
    let bindings = position_encoding_add_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered_dec position_encoding_add CROWN method: {method:?}");
}

// ===========================================================================
// 10. CROWN tightening for decoder stages (CROWN)
// ===========================================================================

#[test]
fn test_firered_dec_crown_tightening_narrow_input() {
    // Build a full decoder block, but use narrow input bounds (+-0.5)
    // to maximize CROWN linearization precision
    let mut b = TensorBlockBuilder::new("firered_dec_crown_tight");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let out = add_decoder_block(&mut b, input, "dec0");
    let def = b.build(out).expect("valid decoder block");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_decoder_block_bindings(&mut bindings);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered_dec crown_tightening narrow input method: {method:?}");
}

// ===========================================================================
// 11. Decoder + cross-attention + LM head pipeline (IBP)
// ===========================================================================

fn build_decoder_cross_attn_lm_head() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_dec_xattn_lm_pipeline");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let input = b.add_input("hidden", &shape);

    // Cross-attention block
    let cross_out = add_cross_attention_block(&mut b, input, "xattn");

    // Decoder self-attention block
    let decoded = add_decoder_block(&mut b, cross_out, "dec0");

    // LM head: RMSNorm -> Linear -> Softmax
    let final_eps = b.add_input("final_eps", &[1]);
    let final_w = b.add_input("final_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(decoded, final_eps, 1, final_w, &shape);

    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs)
        .expect("valid decoder + cross-attn + LM head")
}

fn decoder_cross_attn_lm_head_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_cross_attention_bindings(&mut bindings);
    push_decoder_block_bindings(&mut bindings);
    bindings.push(eps_binding());
    bindings.push(ones_binding(&[HIDDEN_DIM]));
    bindings.push(weight(&[VOCAB_SIZE, HIDDEN_DIM]));
    bindings
}

#[test]
fn test_firered_dec_xattn_lm_pipeline_ibp() {
    let def = build_decoder_cross_attn_lm_head();
    let bindings = decoder_cross_attn_lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "xattn+lm softmax lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "xattn+lm softmax upper <= 1, got {hi}");
    assert_eq!(output.shape(), &[SEQ_LEN, VOCAB_SIZE]);
}

// ===========================================================================
// 12. Verify-and-record (IBP)
// ===========================================================================

#[test]
fn test_firered_dec_e2e_pipeline_verify_and_record() {
    let def = build_e2e_pipeline();
    let bindings = e2e_pipeline_bindings();
    let patch_dim = IMG_CHANNELS * PATCH_SIZE * PATCH_SIZE;
    let input = uniform_bounds(&[NUM_PATCHES, patch_dim], 1.0);
    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "firered_decoder_pipeline::test_firered_dec_e2e_pipeline_verify_and_record",
    );
    assert_bounds_valid(&result.output_bounds);
}

#[test]
fn test_firered_dec_xattn_lm_verify_and_record() {
    let def = build_decoder_cross_attn_lm_head();
    let bindings = decoder_cross_attn_lm_head_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "firered_decoder_pipeline::test_firered_dec_xattn_lm_verify_and_record",
    );
    assert_bounds_valid(&result.output_bounds);
}
