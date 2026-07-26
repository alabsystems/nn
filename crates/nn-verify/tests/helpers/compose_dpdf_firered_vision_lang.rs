// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests for the FireRed-OCR vision-language pipeline bounds.
//!
//! FireRed-OCR (Qwen3-VL-2B) is a vision-language document OCR model combining
//! a ViT visual encoder, a causal transformer decoder with cross-attention,
//! and character-level prediction heads.
//!
//! These tests verify NY IBP and CROWN bound propagation through
//! 14 distinct pipeline stages:
//!
//! ## Visual Encoder (tests 1-2)
//!
//! 1. **ViT visual encoder feature extraction** (IBP + CROWN): Patch embed ->
//!    RMSNorm -> self-attention -> residual -> SwiGLU FFN -> residual.
//! 2. **Visual token projection to language space** (IBP + CROWN): Encoder
//!    features -> Linear projection to decoder hidden dimension.
//!
//! ## Language Decoder (tests 3-7)
//!
//! 3. **Language decoder self-attention per layer** (IBP + CROWN): RMSNorm ->
//!    causal self-attention -> residual -> RMSNorm -> SwiGLU FFN -> residual.
//! 4. **Cross-attention between visual and text tokens** (IBP): Decoder queries
//!    attend to encoder memory via standard (non-causal) attention with residual.
//! 5. **RoPE position encoding for text tokens** (IBP): Rotary positional
//!    encoding via sin/cos bounded in [-1, 1], applied additively to embeddings.
//! 6. **SwiGLU FFN bounds per decoder layer** (IBP + CROWN): gate_proj -> SiLU
//!    -> mul(up_proj) -> down_proj, isolated SwiGLU sub-block.
//! 7. **Layer norm bounds through the model** (IBP + CROWN): RMSNorm -> Linear
//!    -> RMSNorm sandwich verifying normalization stability through depth.
//!
//! ## Prediction Heads (tests 8-10)
//!
//! 8. **LM head token prediction** (IBP): Linear -> Softmax producing token
//!    probabilities bounded in [0, 1].
//! 9. **OCR character-level prediction** (IBP): Final RMSNorm -> CTC projection
//!    -> Softmax, character probabilities bounded in [0, 1].
//! 10. **Layout-aware position encoding** (IBP): 2D sinusoidal position encoding
//!     (x, y) for spatial layout, bounded in [-1, 1], added to features.
//!
//! ## Multi-Resolution & Pipeline (tests 11-12)
//!
//! 11. **Multi-resolution visual features** (IBP): Two-resolution encoder paths
//!     (stride 1 + stride 2) with additive feature fusion.
//! 12. **Full vision-to-OCR pipeline composition** (IBP + CROWN): Patch embed ->
//!     encoder block -> projection -> decoder block -> CTC head -> softmax.
//!
//! ## Confidence & Ordering (tests 13-14)
//!
//! 13. **Confidence score per character** (IBP): CTC softmax -> max-per-position
//!     confidence via element-wise operations, bounded in [0, 1].
//! 14. **Reading order prediction** (IBP): Decoder features -> Linear -> Sigmoid
//!     pairwise ordering scores bounded in (0, 1).
//!
//! Architecture references:
//! - FireRed-OCR: Qwen3-VL-2B variant for document OCR
//!   (yuyq96/FireRed-OCR-Qwen3-VL-2B, HuggingFace)
//! - CTC (Graves et al., 2006): Connectionist Temporal Classification
//! - SwiGLU (Shazeer, 2020): Gated linear unit with SiLU activation
//! - RMSNorm (Zhang & Sennrich, 2019): Root mean square layer normalization
//! - RoPE (Su et al., 2021): Rotary positional embeddings
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

// ===========================================================================
// 1. ViT visual encoder feature extraction bounds (IBP + CROWN)
// ===========================================================================

fn build_vit_encoder() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_vit_encoder");
    let patch_dim = IMG_CHANNELS * PATCH_SIZE * PATCH_SIZE;

    // Patch embedding: flatten image patches -> linear projection
    let input = b.add_input("patches", &[NUM_PATCHES, patch_dim]);
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, patch_dim]);
    let proj_b = b.add_input("proj_b", &[HIDDEN_DIM]);
    let embedded = b.add_linear(input, proj_w, Some(proj_b), &[NUM_PATCHES, HIDDEN_DIM]);

    // One encoder block: RMSNorm -> attention -> residual -> SwiGLU -> residual
    let encoded = add_encoder_block(&mut b, embedded, "enc0", NUM_PATCHES);

    b.build(encoded).expect("valid ViT encoder")
}

fn vit_encoder_bindings() -> Vec<TensorParamBinding> {
    let patch_dim = IMG_CHANNELS * PATCH_SIZE * PATCH_SIZE;
    let mut bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, patch_dim]),
        bias_zero(&[HIDDEN_DIM]),
    ];
    push_encoder_block_bindings(&mut bindings);
    bindings
}

#[test]
fn test_firered_vl_vit_encoder_ibp() {
    let def = build_vit_encoder();
    let bindings = vit_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let patch_dim = IMG_CHANNELS * PATCH_SIZE * PATCH_SIZE;
    let input = uniform_bounds(&[NUM_PATCHES, patch_dim], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[NUM_PATCHES, HIDDEN_DIM]);
}

#[test]
fn test_firered_vl_vit_encoder_crown() {
    let def = build_vit_encoder();
    let bindings = vit_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let patch_dim = IMG_CHANNELS * PATCH_SIZE * PATCH_SIZE;
    let input = uniform_bounds(&[NUM_PATCHES, patch_dim], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered_vl vit_encoder CROWN method: {method:?}");
}

// ===========================================================================
// 2. Visual token projection to language space bounds (IBP + CROWN)
// ===========================================================================

fn build_visual_projection() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_visual_proj");
    let input = b.add_input("vision_features", &[NUM_PATCHES, HIDDEN_DIM]);

    // RMSNorm before projection
    let norm_eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, norm_eps, 1, norm_w, &[NUM_PATCHES, HIDDEN_DIM]);

    // Linear projection from vision dim to language decoder dim
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_b", &[HIDDEN_DIM]);
    let projected = b.add_linear(normed, proj_w, Some(proj_b), &[NUM_PATCHES, HIDDEN_DIM]);

    b.build(projected).expect("valid visual projection")
}

fn visual_projection_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        eps_binding(),
        ones_binding(&[HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
    ]
}

#[test]
fn test_firered_vl_visual_projection_ibp() {
    let def = build_visual_projection();
    let bindings = visual_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[NUM_PATCHES, HIDDEN_DIM]);
}

#[test]
fn test_firered_vl_visual_projection_crown() {
    let def = build_visual_projection();
    let bindings = visual_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered_vl visual_projection CROWN method: {method:?}");
}

// ===========================================================================
// 3. Language decoder self-attention bounds per layer (IBP + CROWN)
// ===========================================================================

fn build_decoder_layer() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_decoder_layer");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let out = add_decoder_block(&mut b, input, "dec0");
    b.build(out).expect("valid decoder layer")
}

fn decoder_layer_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_decoder_block_bindings(&mut bindings);
    bindings
}

#[test]
fn test_firered_vl_decoder_layer_ibp() {
    let def = build_decoder_layer();
    let bindings = decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

#[test]
fn test_firered_vl_decoder_layer_crown() {
    let def = build_decoder_layer();
    let bindings = decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered_vl decoder_layer CROWN method: {method:?}");
}

// ===========================================================================
// 4. Cross-attention between visual and text tokens (IBP)
// ===========================================================================

fn build_cross_attention() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_cross_attn");
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Decoder hidden state
    let input = b.add_input("decoder_hidden", &shape);

    // Pre-attention RMSNorm
    let n_eps = b.add_input("n_eps", &[1]);
    let n_w = b.add_input("n_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, n_eps, 1, n_w, &shape);

    // Cross-attention: Q from decoder, K/V from encoder memory
    // (Using same input for K/V simplification -- structurally identical)
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

    b.build(res).expect("valid cross-attention")
}

fn cross_attention_bindings() -> Vec<TensorParamBinding> {
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
fn test_firered_vl_cross_attention_ibp() {
    let def = build_cross_attention();
    let bindings = cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 5. RoPE position encoding for text tokens (IBP)
// ===========================================================================

fn build_rope_encoding() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_rope");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Token features
    let input = b.add_input("token_features", &shape);

    // Learned embedding
    let embed_w = b.add_input("embed_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let embedded = b.add_linear(input, embed_w, None, &shape);

    // RoPE: additive sin/cos positional encoding
    // In the graph, we model this as element-wise addition of cos/sin PE values.
    let rope_cos = b.add_input("rope_cos", &[SEQ_LEN, HIDDEN_DIM]);
    let rope_sin = b.add_input("rope_sin", &[SEQ_LEN, HIDDEN_DIM]);

    // x * cos + rotate(x) * sin -- simplified as: x * cos + x * sin
    // (rotation is a permutation that doesn't change bounds)
    let cos_part = b.add_binary_mul(embedded, rope_cos, &shape);
    let sin_part = b.add_binary_mul(embedded, rope_sin, &shape);
    let rope_out = b.add_binary_add(cos_part, sin_part, &shape);

    b.build(rope_out).expect("valid RoPE encoding")
}

fn rope_encoding_bindings() -> Vec<TensorParamBinding> {
    // Generate sin/cos tables bounded in [-1, 1]
    let cos_data: Vec<f32> = (0..SEQ_LEN * HIDDEN_DIM)
        .map(|i| {
            let t = (i / HIDDEN_DIM) as f64;
            let d = (i % HIDDEN_DIM) as f64;
            let freq = t / 10000.0_f64.powf(2.0 * d / HIDDEN_DIM as f64);
            freq.cos() as f32
        })
        .collect();
    let sin_data: Vec<f32> = (0..SEQ_LEN * HIDDEN_DIM)
        .map(|i| {
            let t = (i / HIDDEN_DIM) as f64;
            let d = (i % HIDDEN_DIM) as f64;
            let freq = t / 10000.0_f64.powf(2.0 * d / HIDDEN_DIM as f64);
            freq.sin() as f32
        })
        .collect();

    let cos_tensor = ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), cos_data).unwrap();
    let sin_tensor = ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), sin_data).unwrap();

    vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        TensorParamBinding::ConstantTensor(cos_tensor),
        TensorParamBinding::ConstantTensor(sin_tensor),
    ]
}

#[test]
fn test_firered_vl_rope_encoding_ibp() {
    let def = build_rope_encoding();
    let bindings = rope_encoding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 6. SwiGLU FFN bounds per decoder layer (IBP + CROWN)
// ===========================================================================

fn build_swiglu_isolated() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_swiglu");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let out = add_swiglu_ffn(&mut b, input, "ffn", SEQ_LEN);
    b.build(out).expect("valid SwiGLU FFN")
}

fn swiglu_isolated_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_swiglu_bindings(&mut bindings);
    bindings
}

#[test]
fn test_firered_vl_swiglu_ffn_ibp() {
    let def = build_swiglu_isolated();
    let bindings = swiglu_isolated_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

#[test]
fn test_firered_vl_swiglu_ffn_crown() {
    let def = build_swiglu_isolated();
    let bindings = swiglu_isolated_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered_vl swiglu_ffn CROWN method: {method:?}");
}

// ===========================================================================
// 7. Layer norm bounds through the model (IBP + CROWN)
// ===========================================================================

fn build_rmsnorm_sandwich() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_norm_sandwich");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let input = b.add_input("hidden", &shape);

    // First RMSNorm
    let eps1 = b.add_input("eps1", &[1]);
    let w1 = b.add_input("w1", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, eps1, 1, w1, &shape);

    // Linear
    let lin_w = b.add_input("lin_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let linear_out = b.add_linear(normed1, lin_w, None, &shape);

    // Second RMSNorm
    let eps2 = b.add_input("eps2", &[1]);
    let w2 = b.add_input("w2", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(linear_out, eps2, 1, w2, &shape);

    b.build(normed2).expect("valid RMSNorm sandwich")
}

fn rmsnorm_sandwich_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        eps_binding(),
        ones_binding(&[HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        eps_binding(),
        ones_binding(&[HIDDEN_DIM]),
    ]
}

#[test]
fn test_firered_vl_rmsnorm_sandwich_ibp() {
    let def = build_rmsnorm_sandwich();
    let bindings = rmsnorm_sandwich_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

#[test]
fn test_firered_vl_rmsnorm_sandwich_crown() {
    let def = build_rmsnorm_sandwich();
    let bindings = rmsnorm_sandwich_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered_vl rmsnorm_sandwich CROWN method: {method:?}");
}

// ===========================================================================
// 8. LM head token prediction bounds (IBP)
// ===========================================================================

fn build_lm_head() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_lm_head");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // Final RMSNorm
    let eps = b.add_input("eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &[SEQ_LEN, HIDDEN_DIM]);

    // LM head: Linear -> Softmax
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs).expect("valid LM head")
}

fn lm_head_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        eps_binding(),
        ones_binding(&[HIDDEN_DIM]),
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
    ]
}

#[test]
fn test_firered_vl_lm_head_ibp() {
    let def = build_lm_head();
    let bindings = lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "LM head softmax lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "LM head softmax upper <= 1, got {hi}");
}

// ===========================================================================
// 9. OCR character-level prediction bounds (IBP)
// ===========================================================================

fn build_ocr_char_prediction() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_ocr_char");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // Final RMSNorm
    let eps = b.add_input("eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &[SEQ_LEN, HIDDEN_DIM]);

    // CTC projection to character vocabulary + softmax
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(normed, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs).expect("valid OCR char prediction")
}

fn ocr_char_prediction_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        eps_binding(),
        ones_binding(&[HIDDEN_DIM]),
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        bias_zero(&[VOCAB_SIZE]),
    ]
}

#[test]
fn test_firered_vl_ocr_char_prediction_ibp() {
    let def = build_ocr_char_prediction();
    let bindings = ocr_char_prediction_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "OCR char softmax lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "OCR char softmax upper <= 1, got {hi}");
    assert_eq!(output.shape(), &[SEQ_LEN, VOCAB_SIZE]);
}

// ===========================================================================
// 10. Layout-aware position encoding bounds (IBP)
// ===========================================================================

fn build_layout_position_encoding() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_layout_pos");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Token features
    let input = b.add_input("features", &shape);

    // 2D sinusoidal position encoding (x-axis + y-axis)
    // Modeled as two additive PE components
    let pe_x = b.add_input("pe_x", &[SEQ_LEN, HIDDEN_DIM]);
    let pe_y = b.add_input("pe_y", &[SEQ_LEN, HIDDEN_DIM]);

    let with_x = b.add_binary_add(input, pe_x, &shape);
    let with_xy = b.add_binary_add(with_x, pe_y, &shape);

    b.build(with_xy).expect("valid layout position encoding")
}

fn layout_position_encoding_bindings() -> Vec<TensorParamBinding> {
    // 2D sinusoidal PEs bounded in [-1, 1]
    let pe_x_data: Vec<f32> = (0..SEQ_LEN * HIDDEN_DIM)
        .map(|i| {
            let pos = (i / HIDDEN_DIM) as f64;
            let dim = (i % HIDDEN_DIM) as f64;
            let freq = pos / 10000.0_f64.powf(2.0 * dim / HIDDEN_DIM as f64);
            if i % 2 == 0 {
                freq.sin() as f32
            } else {
                freq.cos() as f32
            }
        })
        .collect();
    let pe_y_data: Vec<f32> = (0..SEQ_LEN * HIDDEN_DIM)
        .map(|i| {
            let pos = (i / HIDDEN_DIM) as f64 + 0.5; // offset for y-axis
            let dim = (i % HIDDEN_DIM) as f64;
            let freq = pos / 10000.0_f64.powf(2.0 * dim / HIDDEN_DIM as f64);
            if i % 2 == 0 {
                freq.cos() as f32
            } else {
                freq.sin() as f32
            }
        })
        .collect();

    let pe_x_tensor = ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), pe_x_data).unwrap();
    let pe_y_tensor = ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), pe_y_data).unwrap();

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe_x_tensor),
        TensorParamBinding::ConstantTensor(pe_y_tensor),
    ]
}

#[test]
fn test_firered_vl_layout_position_encoding_ibp() {
    let def = build_layout_position_encoding();
    let bindings = layout_position_encoding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 11. Multi-resolution visual feature bounds (IBP)
// ===========================================================================

fn build_multi_resolution_features() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_multires");
    let shape = [NUM_PATCHES, HIDDEN_DIM];

    // Input patches at original resolution
    let input = b.add_input("patches", &shape);

    // High-res path: Linear projection (stride 1 equivalent)
    let hi_w = b.add_input("hi_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let hi_out = b.add_linear(input, hi_w, None, &shape);

    // Low-res path: Linear -> ReLU (stride 2 downsampled then upsampled)
    let lo_w = b.add_input("lo_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let lo_b = b.add_input("lo_b", &[HIDDEN_DIM]);
    let lo_proj = b.add_linear(input, lo_w, Some(lo_b), &shape);
    let lo_out = b.add_relu(lo_proj, &shape);

    // Additive feature fusion
    let fused = b.add_binary_add(hi_out, lo_out, &shape);

    b.build(fused).expect("valid multi-resolution features")
}

fn multi_resolution_features_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
    ]
}

#[test]
fn test_firered_vl_multi_resolution_features_ibp() {
    let def = build_multi_resolution_features();
    let bindings = multi_resolution_features_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[NUM_PATCHES, HIDDEN_DIM]);
}

// ===========================================================================
// 12. Full vision-to-OCR pipeline composition (IBP + CROWN)
// ===========================================================================

fn build_full_pipeline() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_full_pipeline");
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

    // Decoder block
    let decoded = add_decoder_block(&mut b, projected, "dec0");

    // CTC head: RMSNorm -> Linear -> Softmax
    let final_eps = b.add_input("final_eps", &[1]);
    let final_w = b.add_input("final_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(decoded, final_eps, 1, final_w, &[SEQ_LEN, HIDDEN_DIM]);

    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, ctc_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs).expect("valid full pipeline")
}

fn full_pipeline_bindings() -> Vec<TensorParamBinding> {
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
    // Decoder block
    push_decoder_block_bindings(&mut bindings);
    // Final RMSNorm
    bindings.push(eps_binding());
    bindings.push(ones_binding(&[HIDDEN_DIM]));
    // CTC head
    bindings.push(weight(&[VOCAB_SIZE, HIDDEN_DIM]));
    bindings
}

#[test]
fn test_firered_vl_full_pipeline_ibp() {
    let def = build_full_pipeline();
    let bindings = full_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let patch_dim = IMG_CHANNELS * PATCH_SIZE * PATCH_SIZE;
    let input = uniform_bounds(&[NUM_PATCHES, patch_dim], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "full pipeline softmax lower >= 0, got {lo}");
    assert!(
        hi <= 1.0 + 1e-5,
        "full pipeline softmax upper <= 1, got {hi}"
    );
    eprintln!(
        "firered_vl full pipeline IBP: lo={lo:.6}, hi={hi:.6}, width={:.6}",
        hi - lo
    );
}

#[test]
fn test_firered_vl_full_pipeline_crown() {
    let def = build_full_pipeline();
    let bindings = full_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let patch_dim = IMG_CHANNELS * PATCH_SIZE * PATCH_SIZE;
    let input = uniform_bounds(&[NUM_PATCHES, patch_dim], 0.5);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("firered_vl full pipeline CROWN method: {method:?}");
}

// ===========================================================================
// 13. Confidence score per character bounds (IBP)
// ===========================================================================

fn build_confidence_score() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_confidence");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // CTC head: RMSNorm -> Linear -> Softmax
    let eps = b.add_input("eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &[SEQ_LEN, HIDDEN_DIM]);

    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, ctc_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    // Confidence: max probability per position
    // Modeled as sigmoid(linear(probs)) to produce per-position score in [0, 1]
    let conf_w = b.add_input("conf_w", &[1, VOCAB_SIZE]);
    let conf_logit = b.add_linear(probs, conf_w, None, &[SEQ_LEN, 1]);
    let confidence = b.add_sigmoid(conf_logit, &[SEQ_LEN, 1]);

    b.build(confidence).expect("valid confidence score")
}

fn confidence_score_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        eps_binding(),
        ones_binding(&[HIDDEN_DIM]),
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        weight(&[1, VOCAB_SIZE]),
    ]
}

#[test]
fn test_firered_vl_confidence_score_ibp() {
    let def = build_confidence_score();
    let bindings = confidence_score_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "confidence sigmoid lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "confidence sigmoid upper <= 1, got {hi}");
    assert_eq!(output.shape(), &[SEQ_LEN, 1]);
}

// ===========================================================================
// 14. Reading order prediction bounds (IBP)
// ===========================================================================

fn build_reading_order() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_vl_reading_order");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Decoder features
    let input = b.add_input("decoder_features", &shape);

    // Pairwise ordering: Linear -> Sigmoid
    // Projects features to pairwise order scores
    let order_w = b.add_input("order_w", &[SEQ_LEN, HIDDEN_DIM]);
    let order_b = b.add_input("order_b", &[SEQ_LEN]);
    let order_logits = b.add_linear(input, order_w, Some(order_b), &[SEQ_LEN, SEQ_LEN]);
    let order_probs = b.add_sigmoid(order_logits, &[SEQ_LEN, SEQ_LEN]);

    b.build(order_probs)
        .expect("valid reading order prediction")
}

fn reading_order_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[SEQ_LEN, HIDDEN_DIM]),
        bias_zero(&[SEQ_LEN]),
    ]
}

#[test]
fn test_firered_vl_reading_order_ibp() {
    let def = build_reading_order();
    let bindings = reading_order_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "reading order sigmoid lower >= 0, got {lo}");
    assert!(
        hi <= 1.0 + 1e-5,
        "reading order sigmoid upper <= 1, got {hi}"
    );
    assert_eq!(output.shape(), &[SEQ_LEN, SEQ_LEN]);
}

// ===========================================================================
// Verify-and-record
// ===========================================================================

#[test]
fn test_firered_vl_full_pipeline_verify_and_record() {
    let def = build_full_pipeline();
    let bindings = full_pipeline_bindings();
    let patch_dim = IMG_CHANNELS * PATCH_SIZE * PATCH_SIZE;
    let input = uniform_bounds(&[NUM_PATCHES, patch_dim], 1.0);
    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "firered_vision_lang::test_firered_vl_full_pipeline_verify_and_record",
    );
    assert_bounds_valid(&result.output_bounds);
}

#[test]
fn test_firered_vl_decoder_layer_verify_and_record() {
    let def = build_decoder_layer();
    let bindings = decoder_layer_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "firered_vision_lang::test_firered_vl_decoder_layer_verify_and_record",
    );
    assert_bounds_valid(&result.output_bounds);
}
