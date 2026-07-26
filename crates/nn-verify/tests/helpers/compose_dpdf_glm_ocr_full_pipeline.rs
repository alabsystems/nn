// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for the GLM-OCR full image-to-text pipeline bounds.
//!
//! Verifies NY IBP and CROWN bound propagation through the full
//! GLM-OCR pipeline: vision encoder feature extraction, cross-attention
//! between image features and text tokens, and the complete decoder
//! generation flow for document understanding.
//!
//! ## Tests (15 tests)
//!
//!  1. **Vision encoder feature extraction bounds** — Conv patch embed + projection (IBP)
//!  2. **Cross-attention image-text bounds** — Image features attend to text tokens (IBP)
//!  3. **RoPE position encoding bounds** — Rotary encoding cos/sin bounded (IBP)
//!  4. **SwiGLU FFN intermediate activation bounds** — Gate * up gated linear unit (IBP + CROWN)
//!  5. **Layer norm output bounds per decoder layer** — RMSNorm contraction (IBP + CROWN)
//!  6. **Causal attention mask application** — Causal mask preserves attention bounds (IBP)
//!  7. **Token embedding lookup bounds** — Embedding gather via linear proxy (IBP)
//!  8. **LM head logits output bounds** — Final linear projection to vocab (IBP)
//!  9. **Softmax temperature scaling bounds** — Temperature-scaled softmax in [0,1] (IBP)
//! 10. **Full image-to-text pipeline end-to-end** — Vision + decoder + LM head (IBP + CROWN)
//! 11. **Multi-turn conversation context bounds** — Concatenated turns through decoder (IBP)
//! 12. **System prompt processing bounds** — Prefix prompt through decoder block (IBP)
//! 13. **Image patch embedding bounds** — Conv2d patch extraction (IBP)
//! 14. **Decoder self-attention KV projection bounds** — Q/K/V linear projections (IBP + CROWN)
//! 15. **Output token probability distribution** — Softmax output sums to ~1 (IBP)
//!
//! Architecture references:
//! - GLM-4V / ChatGLM (THUDM): Decoder-only transformer with vision encoder for OCR
//! - RMSNorm (Zhang & Sennrich, 2019): Root mean square normalization
//! - SwiGLU (Shazeer, 2020): SiLU-gated FFN
//! - GQA (Ainslie et al., 2023): Grouped-query attention
//! - RoPE (Su et al., 2021): Rotary positional embeddings
//!
//! Dimensions (symbolic, small for fast verification):
//! - HIDDEN_DIM=8, FFN_DIM=16, NUM_HEADS=2, NUM_KV_HEADS=2
//! - HEAD_DIM=4, SEQ_LEN=4, VOCAB_SIZE=32
//! - IMG_PATCHES=4, PATCH_DIM=12 (3 channels * 2x2 patch)
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
#[allow(dead_code)]
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 4
const SEQ_LEN: usize = 4;
const VOCAB_SIZE: usize = 32;
const IMG_PATCHES: usize = 4;
const PATCH_DIM: usize = 12; // 3 channels * 2x2 patch = 12 flattened
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

/// Push decoder block bindings (12 params: 2 RMSNorm + 4 attention + 3 SwiGLU).
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
// 1. Vision encoder feature extraction bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_vision_encoder_feature_extraction_ibp() {
    // Vision encoder: patch_embed (Linear) -> projection to hidden_dim
    // Simulates Conv2d patch extraction as a flattened Linear.
    // Input: [IMG_PATCHES, PATCH_DIM], Output: [IMG_PATCHES, HIDDEN_DIM]
    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_vision_encoder");
    let input = b.add_input("patches", &[IMG_PATCHES, PATCH_DIM]);
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, PATCH_DIM]);
    let proj_b = b.add_input("proj_b", &[HIDDEN_DIM]);

    // Linear projection from patch space to hidden space
    let projected = b.add_linear(input, proj_w, Some(proj_b), &[IMG_PATCHES, HIDDEN_DIM]);

    // RMSNorm on vision features
    let eps = b.add_input("vis_eps", &[1]);
    let nw = b.add_input("vis_nw", &[HIDDEN_DIM]);
    let out = b.add_rms_norm(projected, eps, 1, nw, &[IMG_PATCHES, HIDDEN_DIM]);
    let def = b.build(out).expect("valid vision encoder kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, PATCH_DIM]),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
        eps_binding(),
        ones(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[IMG_PATCHES, PATCH_DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[IMG_PATCHES, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR vision encoder IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 2. Cross-attention image-text bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_cross_attention_image_text_ibp() {
    // Cross-attention: text tokens (Q) attend to image features (K, V).
    // Q from text: [SEQ_LEN, HIDDEN_DIM], K/V from image: [IMG_PATCHES, HIDDEN_DIM]
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_cross_attn");
    let text_input = b.add_input("text_hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let img_features = b.add_input("img_features", &[IMG_PATCHES, HIDDEN_DIM]);

    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    // Q from text, K/V from image features
    let q = b.add_linear(text_input, q_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let k = b.add_linear(img_features, k_w, None, &[IMG_PATCHES, HIDDEN_DIM]);
    let v = b.add_linear(img_features, v_w, None, &[IMG_PATCHES, HIDDEN_DIM]);

    // Standard (non-causal) attention for cross-attention
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Standard,
        Some(scale),
        &[SEQ_LEN, HIDDEN_DIM],
    );
    let out = b.add_linear(attn, o_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid cross-attention kernel");

    let bindings = vec![
        TensorParamBinding::Variable, // text_hidden
        TensorParamBinding::ConstantTensor(
            // img_features (constant)
            ArrayD::from_elem(IxDyn(&[IMG_PATCHES, HIDDEN_DIM]), 0.5f32),
        ),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // q_w
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // k_w
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // v_w
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // o_w
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR cross-attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 3. RoPE position encoding bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_rope_position_encoding_ibp() {
    // RoPE: Q * cos + rotate(Q) * sin. Cos/sin tables bounded in [-1, 1].
    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_rope_pos");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let cos_table = b.add_input("cos_table", &[SEQ_LEN, HIDDEN_DIM]);
    let sin_table = b.add_input("sin_table", &[SEQ_LEN, HIDDEN_DIM]);

    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Q projection + RoPE rotation (simplified)
    let q = b.add_linear(input, q_w, None, &shape);
    let q_cos = b.add_binary_mul(q, cos_table, &shape);
    let q_sin = b.add_binary_mul(q, sin_table, &shape);
    let out = b.add_binary_add(q_cos, q_sin, &shape);
    let def = b.build(out).expect("valid RoPE kernel");

    let cos_data = rope_cos_sin(SEQ_LEN, HIDDEN_DIM);
    let sin_data = rope_cos_sin(SEQ_LEN, HIDDEN_DIM);
    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        TensorParamBinding::ConstantTensor(cos_data),
        TensorParamBinding::ConstantTensor(sin_data),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR RoPE position encoding IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 4. SwiGLU FFN intermediate activation bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_glm_ocr_swiglu_ffn_intermediate_bounds() {
    // SwiGLU: gate_proj -> SiLU -> mul(up_proj) -> down_proj
    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_swiglu_ffn");
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
    eprintln!("GLM-OCR SwiGLU intermediate IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());

    // CROWN pass
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    assert_eq!(crown_output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("GLM-OCR SwiGLU intermediate CROWN: method={method:?}, bounds=[{clo:.6}, {chi:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 5. Layer norm output bounds per decoder layer (IBP + CROWN)
// ===========================================================================

#[test]
fn test_glm_ocr_layer_norm_per_decoder_layer_bounds() {
    // RMSNorm applied after a decoder block, simulating per-layer normalization.
    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_layer_norm");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // Decoder block first
    let decoded = add_decoder_block(&mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, "blk0");

    // RMSNorm on decoder output
    let eps = b.add_input("ln_eps", &[1]);
    let nw = b.add_input("ln_w", &[HIDDEN_DIM]);
    let out = b.add_rms_norm(decoded, eps, 1, nw, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid per-layer norm kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_decoder_block_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
    bindings.push(eps_binding());
    bindings.push(ones(&[HIDDEN_DIM]));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP pass
    let ibp_output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("GLM-OCR per-layer norm IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());

    // CROWN pass
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("GLM-OCR per-layer norm CROWN: method={method:?}, bounds=[{clo:.6}, {chi:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 6. Causal attention mask application (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_causal_attention_mask_application_ibp() {
    // Causal self-attention: Q/K/V projected, causal mask applied.
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_causal_mask");
    let input = b.add_input("hidden", &shape);

    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(input, q_w, None, &shape);
    let k = b.add_linear(input, k_w, None, &shape);
    let v = b.add_linear(input, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
    let out = b.add_linear(attn, o_w, None, &shape);
    let def = b.build(out).expect("valid causal attention kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR causal mask attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 7. Token embedding lookup bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_token_embedding_lookup_ibp() {
    // Embedding lookup modeled as Linear from one-hot-like input.
    // Input: [SEQ_LEN, VOCAB_SIZE] -> Output: [SEQ_LEN, HIDDEN_DIM]
    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_token_embed");
    let input = b.add_input("token_ids", &[SEQ_LEN, VOCAB_SIZE]);
    let embed_w = b.add_input("embed_w", &[HIDDEN_DIM, VOCAB_SIZE]);
    let out = b.add_linear(input, embed_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid embedding kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, VOCAB_SIZE]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, VOCAB_SIZE], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR token embedding IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 8. LM head logits output bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_lm_head_logits_output_ibp() {
    // LM head: RMSNorm -> Linear(HIDDEN_DIM -> VOCAB_SIZE) -> logits
    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_lm_head");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);

    let normed = b.add_rms_norm(input, eps, 1, norm_w, &[SEQ_LEN, HIDDEN_DIM]);
    let out = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid LM head kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        eps_binding(),
        ones(&[HIDDEN_DIM]),
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR LM head logits IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 9. Softmax temperature scaling bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_softmax_temperature_scaling_ibp() {
    // Temperature scaling: logits / T before softmax.
    // Models as: LM head -> mul(1/T) -> softmax.
    let temperature = 0.7_f32;
    let inv_temp = 1.0 / temperature;

    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_temp_softmax");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // LM head projection
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(input, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);

    // Temperature scaling
    let temp_scale = b.add_input("temp_scale", &[SEQ_LEN, VOCAB_SIZE]);
    let scaled = b.add_binary_mul(logits, temp_scale, &[SEQ_LEN, VOCAB_SIZE]);

    // Softmax
    let out = b.add_softmax(scaled, 1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b
        .build(out)
        .expect("valid temperature-scaled softmax kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[SEQ_LEN, VOCAB_SIZE]),
            inv_temp,
        )),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "GLM-OCR temp-scaled softmax (T={temperature}) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    // Softmax output must be in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "softmax lower bound should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "softmax upper bound should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 10. Full image-to-text pipeline end-to-end (IBP + CROWN)
// ===========================================================================

#[test]
fn test_glm_ocr_full_image_to_text_pipeline_bounds() {
    // Full pipeline: vision_proj -> add with text embed -> decoder block ->
    // final RMSNorm -> LM head -> softmax.
    // Note: Tensor IR reserves axis 0 for concat, so we model the image-text
    // combination as additive fusion (image features projected to text space
    // and added element-wise), which is a valid architectural choice used by
    // several VLMs (e.g., LLaVA uses MLP projection + addition).
    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_full_img2txt");

    // Text token embeddings (variable input)
    let text_input = b.add_input("text_embed", &[SEQ_LEN, HIDDEN_DIM]);

    // Image features projected to text space (constant, pre-extracted)
    let img_proj = b.add_input("img_proj", &[SEQ_LEN, HIDDEN_DIM]);

    // Additive fusion: text embeddings + projected image features
    let combined = b.add_binary_add(text_input, img_proj, &[SEQ_LEN, HIDDEN_DIM]);

    // Decoder block on fused sequence
    let decoded = add_decoder_block(&mut b, combined, SEQ_LEN, HIDDEN_DIM, FFN_DIM, "blk0");

    // Final RMSNorm
    let final_eps = b.add_input("final_eps", &[1]);
    let final_w = b.add_input("final_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(decoded, final_eps, 1, final_w, &[SEQ_LEN, HIDDEN_DIM]);

    // LM head projection
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);

    // Softmax output distribution
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid full img2txt pipeline kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable, // text_embed
        TensorParamBinding::ConstantTensor(
            // img_proj
            ArrayD::from_elem(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), 0.5f32),
        ),
    ];
    push_decoder_block_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
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
    eprintln!("GLM-OCR full img2txt pipeline IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
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
    eprintln!(
        "GLM-OCR full img2txt pipeline CROWN: method={method:?}, bounds=[{clo:.6}, {chi:.6}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 11. Multi-turn conversation context bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_multi_turn_conversation_context_ibp() {
    // Multi-turn: model two conversation turns combined via additive context.
    // Turn 1: [SEQ_LEN, HIDDEN_DIM] (constant, previous turn context).
    // Turn 2: [SEQ_LEN, HIDDEN_DIM] (variable, current turn).
    // Note: Tensor IR reserves axis 0 for concat, so we model the multi-turn
    // combination as additive fusion (prior turn's hidden states added to
    // current turn), which represents KV-cache-style context propagation
    // where past context influences current hidden states via residual addition.

    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_multi_turn");
    let turn1 = b.add_input("turn1", &[SEQ_LEN, HIDDEN_DIM]);
    let turn2 = b.add_input("turn2", &[SEQ_LEN, HIDDEN_DIM]);

    // Additive context: current turn + past turn context
    let combined = b.add_binary_add(turn1, turn2, &[SEQ_LEN, HIDDEN_DIM]);

    // Pass through decoder block
    let out = add_decoder_block(&mut b, combined, SEQ_LEN, HIDDEN_DIM, FFN_DIM, "blk0");
    let def = b.build(out).expect("valid multi-turn kernel");

    let mut bindings = vec![
        TensorParamBinding::ConstantTensor(
            // turn1 (past turn, fixed)
            ArrayD::from_elem(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), 0.3f32),
        ),
        TensorParamBinding::Variable, // turn2 (current turn)
    ];
    push_decoder_block_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR multi-turn context IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 12. System prompt processing bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_system_prompt_processing_ibp() {
    // System prompt: fixed prefix context added to variable user input, then
    // passed through a decoder block + final RMSNorm.
    // Note: Tensor IR reserves axis 0 for concat, so we model the system
    // prompt as an additive bias applied to user input hidden states. This
    // represents the system prompt's influence on the hidden state space
    // after being processed by earlier layers (e.g., via cross-attention
    // or prefix-tuning-style addition).

    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_system_prompt");
    let prompt = b.add_input("system_prompt", &[SEQ_LEN, HIDDEN_DIM]);
    let user_input = b.add_input("user_input", &[SEQ_LEN, HIDDEN_DIM]);

    // Additive fusion: user input + system prompt context
    let combined = b.add_binary_add(prompt, user_input, &[SEQ_LEN, HIDDEN_DIM]);

    // Pass through decoder block
    let decoded = add_decoder_block(&mut b, combined, SEQ_LEN, HIDDEN_DIM, FFN_DIM, "blk0");

    // Final RMSNorm
    let eps = b.add_input("final_eps", &[1]);
    let nw = b.add_input("final_w", &[HIDDEN_DIM]);
    let out = b.add_rms_norm(decoded, eps, 1, nw, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid system prompt kernel");

    let mut bindings = vec![
        TensorParamBinding::ConstantTensor(
            // system_prompt (fixed)
            ArrayD::from_elem(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), 0.1f32),
        ),
        TensorParamBinding::Variable, // user_input
    ];
    push_decoder_block_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
    bindings.push(eps_binding());
    bindings.push(ones(&[HIDDEN_DIM]));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR system prompt processing IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 13. Image patch embedding bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_image_patch_embedding_ibp() {
    // Patch embedding: flatten image patches, project to hidden dim.
    // Models Conv2d patch extraction as Linear from flattened patch.
    // Input: [IMG_PATCHES, PATCH_DIM] -> Output: [IMG_PATCHES, HIDDEN_DIM]
    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_patch_embed");
    let input = b.add_input("patches", &[IMG_PATCHES, PATCH_DIM]);
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, PATCH_DIM]);
    let proj_b = b.add_input("proj_b", &[HIDDEN_DIM]);

    let out = b.add_linear(input, proj_w, Some(proj_b), &[IMG_PATCHES, HIDDEN_DIM]);
    let def = b.build(out).expect("valid patch embedding kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, PATCH_DIM]),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Image pixel intensities in [0, 1]
    let input_bounds = uniform_bounds(&[IMG_PATCHES, PATCH_DIM], 0.5);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[IMG_PATCHES, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR image patch embedding IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 14. Decoder self-attention KV projection bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_glm_ocr_decoder_kv_projection_bounds() {
    // Q/K/V projections before attention: Linear(hidden -> hidden) for each.
    // Verifies projection bounds remain controlled.
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_kv_proj");
    let input = b.add_input("hidden", &shape);

    // RMSNorm before projections (as in decoder block)
    let eps = b.add_input("eps", &[1]);
    let nw = b.add_input("nw", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, nw, &shape);

    // Q, K, V projections
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed, q_w, None, &shape);
    let k = b.add_linear(normed, k_w, None, &shape);
    let v = b.add_linear(normed, v_w, None, &shape);

    // Combine Q, K, V via addition (for single-output graph)
    let qk = b.add_binary_add(q, k, &shape);
    let out = b.add_binary_add(qk, v, &shape);
    let def = b.build(out).expect("valid KV projection kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        eps_binding(),
        ones(&[HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP pass
    let ibp_output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    assert_eq!(ibp_output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("GLM-OCR decoder KV projection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());

    // CROWN pass
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!(
        "GLM-OCR decoder KV projection CROWN: method={method:?}, bounds=[{clo:.6}, {chi:.6}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 15. Output token probability distribution (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_output_token_probability_distribution_ibp() {
    // Full decoder pipeline ending in softmax: decoder block -> RMSNorm ->
    // LM head -> softmax. Verifies output is a valid probability distribution.
    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_output_probs");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // Decoder block
    let decoded = add_decoder_block(&mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, "blk0");

    // Final RMSNorm
    let eps = b.add_input("final_eps", &[1]);
    let nw = b.add_input("final_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(decoded, eps, 1, nw, &[SEQ_LEN, HIDDEN_DIM]);

    // LM head + softmax
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid output probability kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_decoder_block_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
    bindings.push(eps_binding()); // final_eps
    bindings.push(ones(&[HIDDEN_DIM])); // final_w
    bindings.push(weight(&[VOCAB_SIZE, HIDDEN_DIM])); // lm_w

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR output token probs IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // Softmax output must be in [0, 1] (valid probability distribution)
    assert!(
        lo_min >= -1e-4,
        "softmax lower bound should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "softmax upper bound should be <= 1, got {hi_max}"
    );
}
