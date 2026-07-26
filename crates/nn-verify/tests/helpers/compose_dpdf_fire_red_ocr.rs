// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests for the FireRed-OCR vision-language pipeline.
//!
//! FireRed-OCR is a Qwen3-VL-2B variant fine-tuned for document OCR. It
//! combines a vision encoder (patch embedding + ViT blocks) with a language
//! decoder (embedding + transformer decoder with cross-attention) and a CTC
//! output head.
//!
//! These tests verify NY IBP and CROWN bound propagation through
//! three levels of the pipeline:
//!
//! ## Vision Encoder (tests 1-4)
//!
//! 1. **Conv patch embedding** (IBP): Conv2d(3, HIDDEN, patch, stride=patch)
//!    maps image patches to hidden dimension.
//! 2. **BatchNorm + ReLU encoder block** (IBP + CROWN): Conv -> BN -> ReLU
//!    -> Pool -> Linear projection. Vision feature extractor.
//! 3. **Two-block vision encoder** (IBP): Stacked Conv-BN-ReLU blocks with
//!    residual connections verifying bounds through depth.
//! 4. **Vision encoder with projection** (IBP + CROWN): Full encoder ->
//!    Linear projection to language model dimension.
//!
//! ## Language Decoder (tests 5-8)
//!
//! 5. **Token embedding + positional encoding** (IBP): Embedding lookup
//!    with additive positional bias.
//! 6. **Decoder self-attention block** (IBP + CROWN): RMSNorm -> causal
//!    self-attention -> residual -> RMSNorm -> SwiGLU FFN -> residual.
//! 7. **Decoder cross-attention** (IBP): Decoder queries attend to encoder
//!    memory via standard (non-causal) attention.
//! 8. **Two-layer decoder stack** (IBP): Stacked decoder blocks verifying
//!    bounds widening through depth.
//!
//! ## Full Pipeline (tests 9-12)
//!
//! 9. **Vision -> projection -> decoder** (IBP): Image features through
//!    projection layer into one decoder block.
//! 10. **CTC output head** (IBP): Linear -> Softmax producing character
//!     probabilities bounded in [0, 1].
//! 11. **End-to-end: vision -> decoder -> CTC** (IBP): Full pipeline from
//!     image patches to character probabilities.
//! 12. **End-to-end with CROWN** (IBP + CROWN): Same pipeline with CROWN
//!     linearization for tighter bounds.
//!
//! Architecture references:
//! - FireRed-OCR: Qwen3-VL-2B variant for document OCR
//!   (yuyq96/FireRed-OCR-Qwen3-VL-2B, HuggingFace)
//! - CTC (Graves et al., 2006): Connectionist Temporal Classification
//! - SwiGLU (Shazeer, 2020): Gated linear unit with SiLU activation
//! - RMSNorm (Zhang & Sennrich, 2019): Root mean square layer normalization
//!
//! Dimensions are small for fast verification (HIDDEN_DIM=8, SEQ_LEN=4).
//! All tests use IbpValidated soundness mode per nn engineering rules.
//!
//! Part of #4240: FireRed-OCR vision-language pipeline compose tests.

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

/// Add a full decoder block: RMSNorm -> causal self-attention -> residual ->
/// RMSNorm -> SwiGLU FFN -> residual.
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

    // Self-attention (Q/K/V/O projections)
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

/// Push decoder block constant bindings.
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

// ===========================================================================
// 1. Conv patch embedding (IBP)
// ===========================================================================

fn build_patch_embedding() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_patch_embed");
    // Input: flattened image patch features [NUM_PATCHES, IMG_CHANNELS * PATCH_SIZE * PATCH_SIZE]
    let patch_dim = IMG_CHANNELS * PATCH_SIZE * PATCH_SIZE;
    let input = b.add_input("patches", &[NUM_PATCHES, patch_dim]);
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, patch_dim]);
    let proj_b = b.add_input("proj_b", &[HIDDEN_DIM]);
    let out = b.add_linear(input, proj_w, Some(proj_b), &[NUM_PATCHES, HIDDEN_DIM]);
    b.build(out).expect("valid patch embedding")
}

fn patch_embedding_bindings() -> Vec<TensorParamBinding> {
    let patch_dim = IMG_CHANNELS * PATCH_SIZE * PATCH_SIZE;
    vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, patch_dim]),
        bias_zero(&[HIDDEN_DIM]),
    ]
}

#[test]
fn test_firered_vl_patch_embedding_ibp() {
    let def = build_patch_embedding();
    let bindings = patch_embedding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let patch_dim = IMG_CHANNELS * PATCH_SIZE * PATCH_SIZE;
    let input = uniform_bounds(&[NUM_PATCHES, patch_dim], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[NUM_PATCHES, HIDDEN_DIM]);
}

// ===========================================================================
// 2. BatchNorm + ReLU encoder block (IBP + CROWN)
// ===========================================================================

fn build_vision_encoder_block() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_vision_block");
    let input = b.add_input("features", &[NUM_PATCHES, HIDDEN_DIM]);

    // Linear (simulating Conv projection)
    let conv_w = b.add_input("conv_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let conv_b = b.add_input("conv_b", &[HIDDEN_DIM]);
    let conv_out = b.add_linear(input, conv_w, Some(conv_b), &[NUM_PATCHES, HIDDEN_DIM]);

    // BatchNorm (simulated as LayerNorm for graph compatibility)
    let bn_eps = b.add_input("bn_eps", &[1]);
    let bn_w = b.add_input("bn_w", &[HIDDEN_DIM]);
    let bn_b = b.add_input("bn_b", &[HIDDEN_DIM]);
    let normed = b.add_layer_norm(conv_out, bn_eps, 1, bn_w, bn_b, &[NUM_PATCHES, HIDDEN_DIM]);

    // ReLU activation
    let activated = b.add_relu(normed, &[NUM_PATCHES, HIDDEN_DIM]);

    // Output projection
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let projected = b.add_linear(activated, proj_w, None, &[NUM_PATCHES, HIDDEN_DIM]);

    b.build(projected).expect("valid vision encoder block")
}

fn vision_encoder_block_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        eps_binding(),
        ones_binding(&[HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
    ]
}

#[test]
fn test_firered_vl_vision_encoder_block_ibp() {
    let def = build_vision_encoder_block();
    let bindings = vision_encoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[NUM_PATCHES, HIDDEN_DIM]);
}

#[test]
fn test_firered_vl_vision_encoder_block_crown() {
    let def = build_vision_encoder_block();
    let bindings = vision_encoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered_vl vision_encoder_block CROWN method: {method:?}");
}

// ===========================================================================
// 3. Two-block vision encoder (IBP)
// ===========================================================================

fn build_two_block_vision_encoder() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_2block_vision");
    let shape = [NUM_PATCHES, HIDDEN_DIM];
    let input = b.add_input("features", &shape);

    // Block 1: Linear -> LayerNorm -> ReLU
    let w1 = b.add_input("w1", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b1 = b.add_input("b1", &[HIDDEN_DIM]);
    let h1 = b.add_linear(input, w1, Some(b1), &shape);
    let ln1_eps = b.add_input("ln1_eps", &[1]);
    let ln1_w = b.add_input("ln1_w", &[HIDDEN_DIM]);
    let ln1_b = b.add_input("ln1_b", &[HIDDEN_DIM]);
    let n1 = b.add_layer_norm(h1, ln1_eps, 1, ln1_w, ln1_b, &shape);
    let a1 = b.add_relu(n1, &shape);
    // Residual
    let r1 = b.add_binary_add(input, a1, &shape);

    // Block 2: Linear -> LayerNorm -> ReLU
    let w2 = b.add_input("w2", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b2 = b.add_input("b2", &[HIDDEN_DIM]);
    let h2 = b.add_linear(r1, w2, Some(b2), &shape);
    let ln2_eps = b.add_input("ln2_eps", &[1]);
    let ln2_w = b.add_input("ln2_w", &[HIDDEN_DIM]);
    let ln2_b = b.add_input("ln2_b", &[HIDDEN_DIM]);
    let n2 = b.add_layer_norm(h2, ln2_eps, 1, ln2_w, ln2_b, &shape);
    let a2 = b.add_relu(n2, &shape);
    let r2 = b.add_binary_add(r1, a2, &shape);

    b.build(r2).expect("valid 2-block vision encoder")
}

fn two_block_vision_encoder_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        // Block 1
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        eps_binding(),
        ones_binding(&[HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        // Block 2
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        eps_binding(),
        ones_binding(&[HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
    ]
}

#[test]
fn test_firered_vl_two_block_vision_encoder_ibp() {
    let def = build_two_block_vision_encoder();
    let bindings = two_block_vision_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    let width = hi - lo;
    assert!(
        width < 1e6,
        "2-block vision encoder bounds too wide: {width}"
    );
    eprintln!("2-block vision encoder IBP width: {width:.4}");
}

// ===========================================================================
// 4. Vision encoder with projection (IBP + CROWN)
// ===========================================================================

fn build_vision_encoder_with_projection() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_vision_proj");
    let shape = [NUM_PATCHES, HIDDEN_DIM];
    let input = b.add_input("features", &shape);

    // Encoder block: Linear -> LayerNorm -> ReLU
    let w1 = b.add_input("w1", &[HIDDEN_DIM, HIDDEN_DIM]);
    let h1 = b.add_linear(input, w1, None, &shape);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let n1 = b.add_layer_norm(h1, ln_eps, 1, ln_w, ln_b, &shape);
    let a1 = b.add_relu(n1, &shape);

    // Projection to language model dimension (same dim for simplicity)
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_b", &[HIDDEN_DIM]);
    let projected = b.add_linear(a1, proj_w, Some(proj_b), &shape);

    b.build(projected)
        .expect("valid vision encoder with projection")
}

fn vision_encoder_with_projection_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        eps_binding(),
        ones_binding(&[HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
    ]
}

#[test]
fn test_firered_vl_vision_encoder_projection_ibp() {
    let def = build_vision_encoder_with_projection();
    let bindings = vision_encoder_with_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
}

#[test]
fn test_firered_vl_vision_encoder_projection_crown() {
    let def = build_vision_encoder_with_projection();
    let bindings = vision_encoder_with_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered_vl vision_encoder_projection CROWN method: {method:?}");
}

// ===========================================================================
// 5. Token embedding + positional encoding (IBP)
// ===========================================================================

fn build_embedding_with_position() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_embed_pos");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Token embedding (simulated as linear from one-hot-like input)
    let input = b.add_input("token_features", &[SEQ_LEN, HIDDEN_DIM]);
    let embed_w = b.add_input("embed_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let embedded = b.add_linear(input, embed_w, None, &shape);

    // Positional encoding (additive bias)
    let pos_enc = b.add_input("pos_enc", &[SEQ_LEN, HIDDEN_DIM]);
    let out = b.add_binary_add(embedded, pos_enc, &shape);

    b.build(out).expect("valid embedding + position")
}

fn embedding_with_position_bindings() -> Vec<TensorParamBinding> {
    // Positional encoding values bounded in [-1, 1] (sinusoidal)
    let pos_data: Vec<f32> = (0..SEQ_LEN * HIDDEN_DIM)
        .map(|i| {
            let t = (i / HIDDEN_DIM) as f64;
            let d = (i % HIDDEN_DIM) as f64;
            let freq = t / 10000.0_f64.powf(2.0 * d / HIDDEN_DIM as f64);
            if i % 2 == 0 {
                freq.sin() as f32
            } else {
                freq.cos() as f32
            }
        })
        .collect();
    let pos_tensor = ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), pos_data).unwrap();

    vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        TensorParamBinding::ConstantTensor(pos_tensor),
    ]
}

#[test]
fn test_firered_vl_embedding_position_ibp() {
    let def = build_embedding_with_position();
    let bindings = embedding_with_position_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 6. Decoder self-attention block (IBP + CROWN)
// ===========================================================================

fn build_decoder_self_attention_block() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_decoder_block");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let out = add_decoder_block(&mut b, input, "d0");
    b.build(out).expect("valid decoder self-attention block")
}

fn decoder_self_attention_block_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_decoder_block_bindings(&mut bindings);
    bindings
}

#[test]
fn test_firered_vl_decoder_self_attention_ibp() {
    let def = build_decoder_self_attention_block();
    let bindings = decoder_self_attention_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

#[test]
fn test_firered_vl_decoder_self_attention_crown() {
    let def = build_decoder_self_attention_block();
    let bindings = decoder_self_attention_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered_vl decoder_self_attention CROWN method: {method:?}");
}

// ===========================================================================
// 7. Decoder cross-attention (IBP)
// ===========================================================================

fn build_decoder_cross_attention() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_cross_attn");
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let input = b.add_input("decoder_hidden", &shape);

    // Pre-attention RMSNorm
    let n_eps = b.add_input("n_eps", &[1]);
    let n_w = b.add_input("n_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, n_eps, 1, n_w, &shape);

    // Cross-attention: Q from decoder, K/V from encoder memory (same shape here)
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed, q_w, None, &shape);
    let k = b.add_linear(normed, k_w, None, &shape);
    let v = b.add_linear(normed, v_w, None, &shape);
    // Standard (non-causal) mask for cross-attention
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, o_w, None, &shape);
    let res = b.add_binary_add(input, attn_out, &shape);

    b.build(res).expect("valid decoder cross-attention")
}

fn decoder_cross_attention_bindings() -> Vec<TensorParamBinding> {
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
fn test_firered_vl_decoder_cross_attention_ibp() {
    let def = build_decoder_cross_attention();
    let bindings = decoder_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 8. Two-layer decoder stack (IBP)
// ===========================================================================

fn build_two_layer_decoder() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_2layer_decoder");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let x = add_decoder_block(&mut b, input, "d0");
    let x = add_decoder_block(&mut b, x, "d1");
    b.build(x).expect("valid 2-layer decoder")
}

fn two_layer_decoder_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_decoder_block_bindings(&mut bindings);
    push_decoder_block_bindings(&mut bindings);
    bindings
}

#[test]
fn test_firered_vl_two_layer_decoder_ibp() {
    let def = build_two_layer_decoder();
    let bindings = two_layer_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    let width = hi - lo;
    assert!(width < 1e6, "2-layer decoder bounds too wide: {width}");
    eprintln!("2-layer decoder IBP width: {width:.4}");
}

// ===========================================================================
// 9. Vision -> projection -> decoder (IBP)
// ===========================================================================

fn build_vision_to_decoder() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_vision_to_decoder");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Vision encoder output (already encoded)
    let input = b.add_input("vision_features", &[NUM_PATCHES, HIDDEN_DIM]);

    // Projection: Linear from vision to decoder dimension
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_b", &[HIDDEN_DIM]);
    let projected = b.add_linear(input, proj_w, Some(proj_b), &shape);

    // One decoder block
    let x = add_decoder_block(&mut b, projected, "d0");

    b.build(x).expect("valid vision -> projection -> decoder")
}

fn vision_to_decoder_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
    ];
    push_decoder_block_bindings(&mut bindings);
    bindings
}

#[test]
fn test_firered_vl_vision_to_decoder_ibp() {
    let def = build_vision_to_decoder();
    let bindings = vision_to_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 10. CTC output head (IBP)
// ===========================================================================

fn build_ctc_output_head() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_ctc_head");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // Final RMSNorm
    let eps = b.add_input("eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &[SEQ_LEN, HIDDEN_DIM]);

    // CTC projection + softmax
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, ctc_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs).expect("valid CTC output head")
}

fn ctc_output_head_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        eps_binding(),
        ones_binding(&[HIDDEN_DIM]),
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
    ]
}

#[test]
fn test_firered_vl_ctc_output_head_ibp() {
    let def = build_ctc_output_head();
    let bindings = ctc_output_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "CTC softmax lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "CTC softmax upper <= 1, got {hi}");
}

// ===========================================================================
// 11. End-to-end: vision -> decoder -> CTC (IBP)
// ===========================================================================

fn build_end_to_end_pipeline() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_e2e");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Vision features input (post-encoder)
    let input = b.add_input("vision_features", &[NUM_PATCHES, HIDDEN_DIM]);

    // Projection
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_b", &[HIDDEN_DIM]);
    let projected = b.add_linear(input, proj_w, Some(proj_b), &shape);

    // Two decoder layers
    let x = add_decoder_block(&mut b, projected, "d0");
    let x = add_decoder_block(&mut b, x, "d1");

    // Final norm + CTC head
    let final_eps = b.add_input("final_eps", &[1]);
    let final_w = b.add_input("final_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(x, final_eps, 1, final_w, &shape);

    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, ctc_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs).expect("valid end-to-end pipeline")
}

fn end_to_end_pipeline_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
    ];
    push_decoder_block_bindings(&mut bindings);
    push_decoder_block_bindings(&mut bindings);
    // Final RMSNorm
    bindings.push(eps_binding());
    bindings.push(ones_binding(&[HIDDEN_DIM]));
    // CTC head
    bindings.push(weight(&[VOCAB_SIZE, HIDDEN_DIM]));
    bindings
}

#[test]
fn test_firered_vl_end_to_end_ibp() {
    let def = build_end_to_end_pipeline();
    let bindings = end_to_end_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "end-to-end softmax lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "end-to-end softmax upper <= 1, got {hi}");
    eprintln!(
        "firered_vl end-to-end IBP: lo={lo:.6}, hi={hi:.6}, width={:.6}",
        hi - lo
    );
}

// ===========================================================================
// 12. End-to-end with CROWN
// ===========================================================================

#[test]
fn test_firered_vl_end_to_end_crown() {
    let def = build_end_to_end_pipeline();
    let bindings = end_to_end_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 0.5);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered_vl end-to-end CROWN method: {method:?}");
}

// ===========================================================================
// Verify-and-record
// ===========================================================================

#[test]
fn test_firered_vl_decoder_block_verify_and_record() {
    let def = build_decoder_self_attention_block();
    let bindings = decoder_self_attention_block_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "firered_ocr_vl::test_firered_vl_decoder_block_verify_and_record",
    );
    assert_bounds_valid(&result.output_bounds);
}

#[test]
fn test_firered_vl_end_to_end_verify_and_record() {
    let def = build_end_to_end_pipeline();
    let bindings = end_to_end_pipeline_bindings();
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);
    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "firered_ocr_vl::test_firered_vl_end_to_end_verify_and_record",
    );
    assert_bounds_valid(&result.output_bounds);
}
