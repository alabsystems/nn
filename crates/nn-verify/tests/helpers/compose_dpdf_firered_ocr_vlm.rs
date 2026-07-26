// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests for the FireRed-OCR VLM pipeline bounds.
//!
//! FireRed-OCR is a vision-language model for document OCR that combines
//! a convolutional vision encoder, a text embedding encoder, cross-modal
//! fusion, multi-scale feature extraction, and CTC/attention decoder heads.
//!
//! These tests verify NY IBP and CROWN bound propagation through
//! 18 distinct pipeline stages covering the full VLM architecture:
//!
//! ## Vision Encoder (tests 1-3)
//!
//! 1. **Conv patch embedding** (IBP + CROWN): Conv2d(3, D, P, stride=P) ->
//!    BatchNorm -> ReLU producing patch features from raw image pixels.
//! 2. **ViT encoder block** (IBP + CROWN): RMSNorm -> self-attention ->
//!    residual -> SwiGLU FFN -> residual with encoder-block composition.
//! 3. **Vision feature projection** (IBP): RMSNorm -> Linear mapping
//!    vision encoder features to cross-modal embedding dimension.
//!
//! ## Text Encoder (tests 4-6)
//!
//! 4. **Token embedding + position** (IBP): Embedding lookup (modeled as
//!    Linear) + additive sinusoidal position encoding.
//! 5. **Text transformer block** (IBP + CROWN): RMSNorm -> causal
//!    self-attention -> residual -> SwiGLU FFN -> residual.
//! 6. **Text encoder 2-layer stack** (IBP): Two stacked transformer blocks
//!    verifying bounds growth through depth.
//!
//! ## Cross-Modal Fusion (tests 7-9)
//!
//! 7. **Additive vision-text fusion** (IBP): Vision features + text features
//!    via element-wise addition producing fused representation.
//! 8. **Gated cross-modal fusion** (IBP + CROWN): Sigmoid gate * vision +
//!    (1 - gate) * text for learned modality weighting.
//! 9. **Cross-attention fusion** (IBP): Text queries attend to vision
//!    encoder memory via standard (non-causal) multi-head attention.
//!
//! ## OCR Decoder (tests 10-12)
//!
//! 10. **Character classification head** (IBP): Linear -> softmax producing
//!     per-position character probabilities bounded in [0, 1].
//! 11. **CTC projection head** (IBP + CROWN): RMSNorm -> Linear(D, vocab)
//!     -> softmax for CTC decoding with character-level probabilities.
//! 12. **Attention decoder head** (IBP): RMSNorm -> causal attention ->
//!     Linear -> softmax for autoregressive character prediction.
//!
//! ## Multi-Scale Features (tests 13-15)
//!
//! 13. **Dual-resolution vision features** (IBP): Two parallel conv paths
//!     (stride-1 + stride-2) with additive feature fusion.
//! 14. **FPN-style feature pyramid** (IBP + CROWN): Three-scale 1x1 conv
//!     branches + concat + merge conv for multi-resolution fusion.
//! 15. **Multi-scale to decoder projection** (IBP): FPN features -> Linear
//!     -> RMSNorm -> decoder-ready representation.
//!
//! ## Full Pipeline (tests 16-18)
//!
//! 16. **Vision-to-CTC pipeline** (IBP): Conv patch embed -> encoder block
//!     -> projection -> CTC head -> softmax end-to-end.
//! 17. **Full VLM pipeline** (IBP + CROWN): Conv patch embed -> encoder ->
//!     projection -> cross-modal fusion -> decoder block -> softmax.
//! 18. **Monotone tightening** (IBP): Narrower input bounds produce narrower
//!     output bounds through the full pipeline.
//!
//! Architecture references:
//! - FireRed-OCR: Qwen3-VL-2B variant for document OCR
//!   (yuyq96/FireRed-OCR-Qwen3-VL-2B, HuggingFace)
//! - CTC (Graves et al., 2006): Connectionist Temporal Classification
//! - SwiGLU (Shazeer, 2020): Gated linear unit with SiLU activation
//! - RMSNorm (Zhang & Sennrich, 2019): Root mean square layer normalization
//! - FPN (Lin et al., 2017): Feature Pyramid Network for multi-scale fusion
//!
//! Dimensions are small for fast verification (HIDDEN_DIM=8, SEQ_LEN=4).
//! All tests use IbpValidated soundness mode per nn engineering rules.
//!
//! Part of #4240: FireRed-OCR VLM pipeline compose tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const HIDDEN_DIM: usize = 8;
const FFN_DIM: usize = 16;
const NUM_HEADS: usize = 2;
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 4
const IMG_CH: usize = 3;
const PATCH_SIZE: usize = 2;
const IMG_H: usize = 4;
const IMG_W: usize = 4;
const NUM_PATCHES: usize = (IMG_H / PATCH_SIZE) * (IMG_W / PATCH_SIZE); // 4
const SEQ_LEN: usize = 4;
const VOCAB_SIZE: usize = 32;
const FPN_CH: usize = 8;
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

/// Ones tensor binding (for normalization weights / BN variance).
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

/// Push decoder block constant bindings.
fn push_decoder_block_bindings(bindings: &mut Vec<TensorParamBinding>) {
    push_encoder_block_bindings(bindings);
}

// ===========================================================================
// 1. Conv patch embedding (IBP + CROWN)
// ===========================================================================

/// Build convolutional patch embedding: Conv2d(3, D, P, stride=P) -> BN -> ReLU.
///
/// Input: [IMG_CH, IMG_H, IMG_W] = [3, 4, 4] image pixels
/// Output: [HIDDEN_DIM, IMG_H/PATCH_SIZE, IMG_W/PATCH_SIZE] = [8, 2, 2]
/// Reshape to [NUM_PATCHES, HIDDEN_DIM] = [4, 8] for transformer input.
fn build_conv_patch_embed() -> TensorKernelDef {
    let conv_out_h = IMG_H / PATCH_SIZE;
    let conv_out_w = IMG_W / PATCH_SIZE;
    let conv_shape = [HIDDEN_DIM, conv_out_h, conv_out_w];
    let flat_shape = [NUM_PATCHES, HIDDEN_DIM];

    let mut b = TensorBlockBuilder::new("firered_vlm_conv_patch_embed");
    let input = b.add_input("image", &[IMG_CH, IMG_H, IMG_W]);

    // Conv2d patch embedding: stride=PATCH_SIZE, kernel=PATCH_SIZE
    let conv_w = b.add_input("conv_w", &[HIDDEN_DIM, IMG_CH, PATCH_SIZE, PATCH_SIZE]);
    let conv_b = b.add_input("conv_b", &[HIDDEN_DIM]);
    let conv = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &conv_shape,
    );

    // BatchNorm + ReLU
    let bn_mean = b.add_input("bn_mean", &[HIDDEN_DIM]);
    let bn_var = b.add_input("bn_var", &[HIDDEN_DIM]);
    let bn_w = b.add_input("bn_w", &[HIDDEN_DIM]);
    let bn_b = b.add_input("bn_b", &[HIDDEN_DIM]);
    let bn_eps = b.add_input("bn_eps", &[1]);
    let bn = b.add_batch_norm(conv, bn_mean, bn_var, bn_w, bn_b, bn_eps, &conv_shape);
    let act = b.add_relu(bn, &conv_shape);

    // Reshape to [NUM_PATCHES, HIDDEN_DIM] for transformer
    let flat = b.add_reshape(act, &flat_shape);

    b.build(flat).expect("valid conv patch embed")
}

fn conv_patch_embed_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, IMG_CH, PATCH_SIZE, PATCH_SIZE]),
        bias_zero(&[HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),    // bn_mean
        ones_binding(&[HIDDEN_DIM]), // bn_var
        ones_binding(&[HIDDEN_DIM]), // bn_w
        bias_zero(&[HIDDEN_DIM]),    // bn_b
        eps_binding(),
    ]
}

#[test]
fn test_firered_vlm_conv_patch_embed_ibp() {
    let def = build_conv_patch_embed();
    let bindings = conv_patch_embed_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[IMG_CH, IMG_H, IMG_W]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[IMG_CH, IMG_H, IMG_W]), 1.0f32),
    )
    .expect("image bounds");
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[NUM_PATCHES, HIDDEN_DIM]);
    let (lo, _hi) = bounds_min_max(&output);
    assert!(lo >= -1e-6, "ReLU lower >= 0, got {lo}");
}

#[test]
fn test_firered_vlm_conv_patch_embed_crown() {
    let def = build_conv_patch_embed();
    let bindings = conv_patch_embed_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[IMG_CH, IMG_H, IMG_W]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[IMG_CH, IMG_H, IMG_W]), 1.0f32),
    )
    .expect("image bounds");
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered_vlm conv_patch_embed CROWN method: {method:?}");
}

// ===========================================================================
// 2. ViT encoder block (IBP + CROWN)
// ===========================================================================

fn build_vit_encoder_block() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vlm_vit_enc_block");
    let input = b.add_input("patch_features", &[NUM_PATCHES, HIDDEN_DIM]);
    let encoded = add_encoder_block(&mut b, input, "enc0", NUM_PATCHES);
    b.build(encoded).expect("valid ViT encoder block")
}

fn vit_encoder_block_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_encoder_block_bindings(&mut bindings);
    bindings
}

#[test]
fn test_firered_vlm_vit_encoder_block_ibp() {
    let def = build_vit_encoder_block();
    let bindings = vit_encoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[NUM_PATCHES, HIDDEN_DIM]);
}

#[test]
fn test_firered_vlm_vit_encoder_block_crown() {
    let def = build_vit_encoder_block();
    let bindings = vit_encoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered_vlm vit_encoder_block CROWN method: {method:?}");
}

// ===========================================================================
// 3. Vision feature projection (IBP)
// ===========================================================================

fn build_vision_projection() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vlm_vision_proj");
    let input = b.add_input("encoder_features", &[NUM_PATCHES, HIDDEN_DIM]);

    // RMSNorm -> Linear projection
    let eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &[NUM_PATCHES, HIDDEN_DIM]);

    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_b", &[HIDDEN_DIM]);
    let projected = b.add_linear(normed, proj_w, Some(proj_b), &[NUM_PATCHES, HIDDEN_DIM]);

    b.build(projected).expect("valid vision projection")
}

fn vision_projection_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        eps_binding(),
        ones_binding(&[HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
    ]
}

#[test]
fn test_firered_vlm_vision_projection_ibp() {
    let def = build_vision_projection();
    let bindings = vision_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[NUM_PATCHES, HIDDEN_DIM]);
}

// ===========================================================================
// 4. Token embedding + position encoding (IBP)
// ===========================================================================

fn build_text_embedding() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("firered_vlm_text_embed");
    let input = b.add_input("token_ids", &shape);

    // Embedding lookup modeled as Linear projection
    let embed_w = b.add_input("embed_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let embedded = b.add_linear(input, embed_w, None, &shape);

    // Additive sinusoidal position encoding
    let pe = b.add_input("pos_enc", &[SEQ_LEN, HIDDEN_DIM]);
    let with_pe = b.add_binary_add(embedded, pe, &shape);

    b.build(with_pe).expect("valid text embedding")
}

fn text_embedding_bindings() -> Vec<TensorParamBinding> {
    // Sinusoidal PE bounded in [-1, 1]
    let pe_data: Vec<f32> = (0..SEQ_LEN * HIDDEN_DIM)
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
    let pe_tensor = ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), pe_data).unwrap();

    vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        TensorParamBinding::ConstantTensor(pe_tensor),
    ]
}

#[test]
fn test_firered_vlm_text_embedding_ibp() {
    let def = build_text_embedding();
    let bindings = text_embedding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 5. Text transformer block (IBP + CROWN)
// ===========================================================================

fn build_text_transformer_block() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vlm_text_block");
    let input = b.add_input("text_hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let out = add_decoder_block(&mut b, input, "text0");
    b.build(out).expect("valid text transformer block")
}

fn text_transformer_block_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_decoder_block_bindings(&mut bindings);
    bindings
}

#[test]
fn test_firered_vlm_text_transformer_block_ibp() {
    let def = build_text_transformer_block();
    let bindings = text_transformer_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

#[test]
fn test_firered_vlm_text_transformer_block_crown() {
    let def = build_text_transformer_block();
    let bindings = text_transformer_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered_vlm text_transformer CROWN method: {method:?}");
}

// ===========================================================================
// 6. Text encoder 2-layer stack (IBP)
// ===========================================================================

fn build_text_encoder_2layer() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vlm_text_2layer");
    let input = b.add_input("text_hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let h1 = add_decoder_block(&mut b, input, "text_l0");
    let h2 = add_decoder_block(&mut b, h1, "text_l1");
    b.build(h2).expect("valid 2-layer text encoder")
}

fn text_encoder_2layer_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_decoder_block_bindings(&mut bindings);
    push_decoder_block_bindings(&mut bindings);
    bindings
}

#[test]
fn test_firered_vlm_text_encoder_2layer_ibp() {
    let def = build_text_encoder_2layer();
    let bindings = text_encoder_2layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 7. Additive vision-text fusion (IBP)
// ===========================================================================

fn build_additive_fusion() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("firered_vlm_additive_fusion");

    // Vision features (projected to seq_len)
    let input = b.add_input("vision_features", &shape);
    let v_w = b.add_input("vis_proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let vision_proj = b.add_linear(input, v_w, None, &shape);

    // Text features (from same input for single-input graph)
    let t_w = b.add_input("text_proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let text_proj = b.add_linear(input, t_w, None, &shape);

    // Additive fusion
    let fused = b.add_binary_add(vision_proj, text_proj, &shape);

    b.build(fused).expect("valid additive fusion")
}

fn additive_fusion_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
    ]
}

#[test]
fn test_firered_vlm_additive_fusion_ibp() {
    let def = build_additive_fusion();
    let bindings = additive_fusion_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 8. Gated cross-modal fusion (IBP + CROWN)
// ===========================================================================

/// Build gated fusion: gate = sigmoid(Linear(x)), out = gate * vision + (1-gate) * text.
///
/// Modeled as: sigmoid(Linear(x)) * Linear_v(x) + (1 - sigmoid(Linear(x))) * Linear_t(x).
/// Since (1-sigmoid) is not directly available, we use: out = text + gate * (vision - text),
/// simplified to: gate * vision_proj + (1-gate) via subtraction workaround.
/// Actually, we directly compute: sigmoid_gate * vis_proj and add bias from text.
fn build_gated_fusion() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("firered_vlm_gated_fusion");

    let input = b.add_input("features", &shape);

    // Gate: sigmoid(Linear(x))
    let gate_w = b.add_input("gate_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let gate_logit = b.add_linear(input, gate_w, None, &shape);
    let gate = b.add_sigmoid(gate_logit, &shape);

    // Vision branch
    let vis_w = b.add_input("vis_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let vis = b.add_linear(input, vis_w, None, &shape);

    // Text branch
    let text_w = b.add_input("text_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let text = b.add_linear(input, text_w, None, &shape);

    // Gated fusion: gate * vision + text (simplified -- gate modulates vision contribution)
    let gated_vis = b.add_binary_mul(gate, vis, &shape);
    let fused = b.add_binary_add(gated_vis, text, &shape);

    b.build(fused).expect("valid gated fusion")
}

fn gated_fusion_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
    ]
}

#[test]
fn test_firered_vlm_gated_fusion_ibp() {
    let def = build_gated_fusion();
    let bindings = gated_fusion_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

#[test]
fn test_firered_vlm_gated_fusion_crown() {
    let def = build_gated_fusion();
    let bindings = gated_fusion_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered_vlm gated_fusion CROWN method: {method:?}");
}

// ===========================================================================
// 9. Cross-attention fusion (IBP)
// ===========================================================================

fn build_cross_attention_fusion() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut b = TensorBlockBuilder::new("firered_vlm_cross_attn_fusion");

    let input = b.add_input("features", &shape);

    // Pre-attention RMSNorm
    let eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &shape);

    // Cross-attention: Q from text, K/V from vision (both from normed input)
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed, q_w, None, &shape);
    let k = b.add_linear(normed, k_w, None, &shape);
    let v = b.add_linear(normed, v_w, None, &shape);

    // Standard (non-causal) attention for cross-modal fusion
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, o_w, None, &shape);
    let fused = b.add_binary_add(input, attn_out, &shape);

    b.build(fused).expect("valid cross-attention fusion")
}

fn cross_attention_fusion_bindings() -> Vec<TensorParamBinding> {
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
fn test_firered_vlm_cross_attention_fusion_ibp() {
    let def = build_cross_attention_fusion();
    let bindings = cross_attention_fusion_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 10. Character classification head (IBP)
// ===========================================================================

fn build_char_classification() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vlm_char_cls");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // Linear -> softmax
    let cls_w = b.add_input("cls_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let cls_b = b.add_input("cls_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(input, cls_w, Some(cls_b), &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs).expect("valid char classification")
}

fn char_classification_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        bias_zero(&[VOCAB_SIZE]),
    ]
}

#[test]
fn test_firered_vlm_char_classification_ibp() {
    let def = build_char_classification();
    let bindings = char_classification_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "softmax lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "softmax upper <= 1, got {hi}");
    assert_eq!(output.shape(), &[SEQ_LEN, VOCAB_SIZE]);
}

// ===========================================================================
// 11. CTC projection head (IBP + CROWN)
// ===========================================================================

fn build_ctc_projection() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vlm_ctc_proj");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // RMSNorm before CTC projection
    let eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &[SEQ_LEN, HIDDEN_DIM]);

    // Linear -> softmax (CTC vocabulary includes blank token)
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(normed, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs).expect("valid CTC projection")
}

fn ctc_projection_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        eps_binding(),
        ones_binding(&[HIDDEN_DIM]),
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        bias_zero(&[VOCAB_SIZE]),
    ]
}

#[test]
fn test_firered_vlm_ctc_projection_ibp() {
    let def = build_ctc_projection();
    let bindings = ctc_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "CTC softmax lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "CTC softmax upper <= 1, got {hi}");
}

#[test]
fn test_firered_vlm_ctc_projection_crown() {
    let def = build_ctc_projection();
    let bindings = ctc_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered_vlm ctc_projection CROWN method: {method:?}");
}

// ===========================================================================
// 12. Attention decoder head (IBP)
// ===========================================================================

fn build_attention_decoder_head() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut b = TensorBlockBuilder::new("firered_vlm_attn_decoder_head");
    let input = b.add_input("hidden", &shape);

    // RMSNorm
    let eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &shape);

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

    // LM head: Linear -> softmax
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(res, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs).expect("valid attention decoder head")
}

fn attention_decoder_head_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        eps_binding(),
        ones_binding(&[HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
    ]
}

#[test]
fn test_firered_vlm_attention_decoder_head_ibp() {
    let def = build_attention_decoder_head();
    let bindings = attention_decoder_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "decoder softmax lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "decoder softmax upper <= 1, got {hi}");
    assert_eq!(output.shape(), &[SEQ_LEN, VOCAB_SIZE]);
}

// ===========================================================================
// 13. Dual-resolution vision features (IBP)
// ===========================================================================

fn build_dual_resolution() -> TensorKernelDef {
    let shape = [NUM_PATCHES, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("firered_vlm_dual_res");
    let input = b.add_input("patches", &shape);

    // High-res path: stride-1 equivalent (linear projection)
    let hi_w = b.add_input("hi_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let hi_out = b.add_linear(input, hi_w, None, &shape);

    // Low-res path: linear + ReLU (stride-2 downsampled then upsampled)
    let lo_w = b.add_input("lo_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let lo_b = b.add_input("lo_b", &[HIDDEN_DIM]);
    let lo_proj = b.add_linear(input, lo_w, Some(lo_b), &shape);
    let lo_out = b.add_relu(lo_proj, &shape);

    // Additive fusion
    let fused = b.add_binary_add(hi_out, lo_out, &shape);

    b.build(fused).expect("valid dual-resolution features")
}

fn dual_resolution_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
    ]
}

#[test]
fn test_firered_vlm_dual_resolution_ibp() {
    let def = build_dual_resolution();
    let bindings = dual_resolution_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[NUM_PATCHES, HIDDEN_DIM]);
}

// ===========================================================================
// 14. FPN-style feature pyramid (IBP + CROWN)
// ===========================================================================

fn build_fpn_feature_pyramid() -> TensorKernelDef {
    let shape = [NUM_PATCHES, HIDDEN_DIM];
    let branch_shape = [NUM_PATCHES, FPN_CH];
    let concat_ch = FPN_CH * 3;
    let concat_shape = [NUM_PATCHES, concat_ch];
    let out_shape = [NUM_PATCHES, HIDDEN_DIM];

    let mut b = TensorBlockBuilder::new("firered_vlm_fpn");
    let input = b.add_input("features", &shape);

    // Three-scale branches (1x1 conv equivalent as Linear)
    let w1 = b.add_input("scale1_w", &[FPN_CH, HIDDEN_DIM]);
    let b1 = b.add_input("scale1_b", &[FPN_CH]);
    let s1 = b.add_linear(input, w1, Some(b1), &branch_shape);
    let s1_act = b.add_relu(s1, &branch_shape);

    let w2 = b.add_input("scale2_w", &[FPN_CH, HIDDEN_DIM]);
    let b2 = b.add_input("scale2_b", &[FPN_CH]);
    let s2 = b.add_linear(input, w2, Some(b2), &branch_shape);
    let s2_act = b.add_relu(s2, &branch_shape);

    let w3 = b.add_input("scale3_w", &[FPN_CH, HIDDEN_DIM]);
    let b3 = b.add_input("scale3_b", &[FPN_CH]);
    let s3 = b.add_linear(input, w3, Some(b3), &branch_shape);
    let s3_act = b.add_relu(s3, &branch_shape);

    // Concat all scales on feature dimension
    let fused = b.add_concat(&[s1_act, s2_act, s3_act], 1, &concat_shape);

    // Merge: Linear(concat_ch -> HIDDEN_DIM)
    let merge_w = b.add_input("merge_w", &[HIDDEN_DIM, concat_ch]);
    let merge_b = b.add_input("merge_b", &[HIDDEN_DIM]);
    let merged = b.add_linear(fused, merge_w, Some(merge_b), &out_shape);

    b.build(merged).expect("valid FPN feature pyramid")
}

fn fpn_feature_pyramid_bindings() -> Vec<TensorParamBinding> {
    let concat_ch = FPN_CH * 3;
    vec![
        TensorParamBinding::Variable,
        // Scale 1
        weight(&[FPN_CH, HIDDEN_DIM]),
        bias_zero(&[FPN_CH]),
        // Scale 2
        weight(&[FPN_CH, HIDDEN_DIM]),
        bias_zero(&[FPN_CH]),
        // Scale 3
        weight(&[FPN_CH, HIDDEN_DIM]),
        bias_zero(&[FPN_CH]),
        // Merge
        weight(&[HIDDEN_DIM, concat_ch]),
        bias_zero(&[HIDDEN_DIM]),
    ]
}

#[test]
fn test_firered_vlm_fpn_feature_pyramid_ibp() {
    let def = build_fpn_feature_pyramid();
    let bindings = fpn_feature_pyramid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[NUM_PATCHES, HIDDEN_DIM]);
}

#[test]
fn test_firered_vlm_fpn_feature_pyramid_crown() {
    let def = build_fpn_feature_pyramid();
    let bindings = fpn_feature_pyramid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered_vlm fpn_feature_pyramid CROWN method: {method:?}");
}

// ===========================================================================
// 15. Multi-scale to decoder projection (IBP)
// ===========================================================================

fn build_multiscale_decoder_proj() -> TensorKernelDef {
    let shape = [NUM_PATCHES, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("firered_vlm_ms_dec_proj");
    let input = b.add_input("fpn_features", &shape);

    // Linear projection
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_b", &[HIDDEN_DIM]);
    let projected = b.add_linear(input, proj_w, Some(proj_b), &shape);

    // RMSNorm for decoder readiness
    let eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(projected, eps, 1, norm_w, &shape);

    b.build(normed)
        .expect("valid multi-scale decoder projection")
}

fn multiscale_decoder_proj_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        eps_binding(),
        ones_binding(&[HIDDEN_DIM]),
    ]
}

#[test]
fn test_firered_vlm_multiscale_decoder_proj_ibp() {
    let def = build_multiscale_decoder_proj();
    let bindings = multiscale_decoder_proj_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[NUM_PATCHES, HIDDEN_DIM]);
}

// ===========================================================================
// 16. Vision-to-CTC pipeline (IBP)
// ===========================================================================

fn build_vision_to_ctc_pipeline() -> TensorKernelDef {
    let patch_dim = IMG_CH * PATCH_SIZE * PATCH_SIZE;
    let mut b = TensorBlockBuilder::new("firered_vlm_vision_to_ctc");

    // Patch embedding (linear)
    let input = b.add_input("patches", &[NUM_PATCHES, patch_dim]);
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, patch_dim]);
    let proj_b = b.add_input("proj_b", &[HIDDEN_DIM]);
    let embedded = b.add_linear(input, proj_w, Some(proj_b), &[NUM_PATCHES, HIDDEN_DIM]);

    // Encoder block
    let encoded = add_encoder_block(&mut b, embedded, "enc0", NUM_PATCHES);

    // Vision projection
    let vp_w = b.add_input("vp_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let projected = b.add_linear(encoded, vp_w, None, &[SEQ_LEN, HIDDEN_DIM]);

    // CTC head: RMSNorm -> Linear -> Softmax
    let ctc_eps = b.add_input("ctc_eps", &[1]);
    let ctc_nw = b.add_input("ctc_nw", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(projected, ctc_eps, 1, ctc_nw, &[SEQ_LEN, HIDDEN_DIM]);

    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, ctc_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs).expect("valid vision-to-CTC pipeline")
}

fn vision_to_ctc_pipeline_bindings() -> Vec<TensorParamBinding> {
    let patch_dim = IMG_CH * PATCH_SIZE * PATCH_SIZE;
    let mut bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, patch_dim]),
        bias_zero(&[HIDDEN_DIM]),
    ];
    push_encoder_block_bindings(&mut bindings);
    // Vision projection
    bindings.push(weight(&[HIDDEN_DIM, HIDDEN_DIM]));
    // CTC head
    bindings.push(eps_binding());
    bindings.push(ones_binding(&[HIDDEN_DIM]));
    bindings.push(weight(&[VOCAB_SIZE, HIDDEN_DIM]));
    bindings
}

#[test]
fn test_firered_vlm_vision_to_ctc_pipeline_ibp() {
    let def = build_vision_to_ctc_pipeline();
    let bindings = vision_to_ctc_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let patch_dim = IMG_CH * PATCH_SIZE * PATCH_SIZE;
    let input = uniform_bounds(&[NUM_PATCHES, patch_dim], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "pipeline softmax lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "pipeline softmax upper <= 1, got {hi}");
    eprintln!(
        "firered_vlm vision-to-CTC pipeline IBP: lo={lo:.6}, hi={hi:.6}, width={:.6}",
        hi - lo
    );
}

// ===========================================================================
// 17. Full VLM pipeline (IBP + CROWN)
// ===========================================================================

fn build_full_vlm_pipeline() -> TensorKernelDef {
    let patch_dim = IMG_CH * PATCH_SIZE * PATCH_SIZE;
    let mut b = TensorBlockBuilder::new("firered_vlm_full_pipeline");

    // Patch embedding
    let input = b.add_input("patches", &[NUM_PATCHES, patch_dim]);
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, patch_dim]);
    let proj_b = b.add_input("proj_b", &[HIDDEN_DIM]);
    let embedded = b.add_linear(input, proj_w, Some(proj_b), &[NUM_PATCHES, HIDDEN_DIM]);

    // Vision encoder block
    let encoded = add_encoder_block(&mut b, embedded, "enc0", NUM_PATCHES);

    // Vision-to-language projection
    let vl_w = b.add_input("vl_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let vl_b = b.add_input("vl_b", &[HIDDEN_DIM]);
    let projected = b.add_linear(encoded, vl_w, Some(vl_b), &[SEQ_LEN, HIDDEN_DIM]);

    // Cross-modal fusion: vision + text via additive Linear projections
    let fuse_w = b.add_input("fuse_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let fuse_proj = b.add_linear(projected, fuse_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let fused = b.add_binary_add(projected, fuse_proj, &[SEQ_LEN, HIDDEN_DIM]);

    // Decoder block
    let decoded = add_decoder_block(&mut b, fused, "dec0");

    // LM head: RMSNorm -> Linear -> Softmax
    let final_eps = b.add_input("final_eps", &[1]);
    let final_w = b.add_input("final_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(decoded, final_eps, 1, final_w, &[SEQ_LEN, HIDDEN_DIM]);

    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs).expect("valid full VLM pipeline")
}

fn full_vlm_pipeline_bindings() -> Vec<TensorParamBinding> {
    let patch_dim = IMG_CH * PATCH_SIZE * PATCH_SIZE;
    let mut bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, patch_dim]),
        bias_zero(&[HIDDEN_DIM]),
    ];
    push_encoder_block_bindings(&mut bindings);
    // Vision-to-language projection
    bindings.push(weight(&[HIDDEN_DIM, HIDDEN_DIM]));
    bindings.push(bias_zero(&[HIDDEN_DIM]));
    // Cross-modal fusion
    bindings.push(weight(&[HIDDEN_DIM, HIDDEN_DIM]));
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
fn test_firered_vlm_full_pipeline_ibp() {
    let def = build_full_vlm_pipeline();
    let bindings = full_vlm_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let patch_dim = IMG_CH * PATCH_SIZE * PATCH_SIZE;
    let input = uniform_bounds(&[NUM_PATCHES, patch_dim], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "full VLM softmax lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "full VLM softmax upper <= 1, got {hi}");
    eprintln!(
        "firered_vlm full pipeline IBP: lo={lo:.6}, hi={hi:.6}, width={:.6}",
        hi - lo
    );
}

#[test]
fn test_firered_vlm_full_pipeline_crown() {
    let def = build_full_vlm_pipeline();
    let bindings = full_vlm_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let patch_dim = IMG_CH * PATCH_SIZE * PATCH_SIZE;
    let input = uniform_bounds(&[NUM_PATCHES, patch_dim], 0.5);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered_vlm full pipeline CROWN method: {method:?}");
}

// ===========================================================================
// 18. Monotone tightening (IBP)
// ===========================================================================

#[test]
fn test_firered_vlm_full_pipeline_monotone_tightening() {
    let def = build_full_vlm_pipeline();
    let bindings = full_vlm_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let patch_dim = IMG_CH * PATCH_SIZE * PATCH_SIZE;

    // Wide input: [-1, 1]
    let wide_input = uniform_bounds(&[NUM_PATCHES, patch_dim], 1.0);
    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");
    let (wide_lo, wide_hi) = bounds_min_max(&wide_output);
    let wide_width = wide_hi - wide_lo;

    // Narrow input: [-0.5, 0.5]
    let narrow_input = uniform_bounds(&[NUM_PATCHES, patch_dim], 0.5);
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("IBP narrow");
    let (narrow_lo, narrow_hi) = bounds_min_max(&narrow_output);
    let narrow_width = narrow_hi - narrow_lo;

    eprintln!(
        "Monotone tightening: wide=[{wide_lo}, {wide_hi}] width={wide_width:.6}, \
         narrow=[{narrow_lo}, {narrow_hi}] width={narrow_width:.6}"
    );

    assert!(
        narrow_width <= wide_width + 1e-6,
        "monotone tightening: narrow width {narrow_width:.6} should be <= wide width {wide_width:.6}"
    );
}

// ===========================================================================
// Verify-and-record
// ===========================================================================

#[test]
fn test_firered_vlm_full_pipeline_verify_and_record() {
    let def = build_full_vlm_pipeline();
    let bindings = full_vlm_pipeline_bindings();
    let patch_dim = IMG_CH * PATCH_SIZE * PATCH_SIZE;
    let input = uniform_bounds(&[NUM_PATCHES, patch_dim], 1.0);
    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "firered_ocr_vlm::test_firered_vlm_full_pipeline_verify_and_record",
    );
    assert_bounds_valid(&result.output_bounds);
}

#[test]
fn test_firered_vlm_ctc_projection_verify_and_record() {
    let def = build_ctc_projection();
    let bindings = ctc_projection_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "firered_ocr_vlm::test_firered_vlm_ctc_projection_verify_and_record",
    );
    assert_bounds_valid(&result.output_bounds);
}
