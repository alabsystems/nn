// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for FireRed-OCR full vision-language pipeline bounds.
//!
//! Verifies IBP and CROWN bound propagation through the FireRed-OCR
//! vision-language pipeline (SigLIP2 encoder, cross-attention decoder,
//! SwiGLU FFN, RMSNorm, CTC/autoregressive output heads):
//!
//! ## Tests (18 tests)
//!
//! 1.  **SigLIP2 vision encoder patch embedding bounds** (IBP)
//! 2.  **Window attention ViT block bounds** (IBP)
//! 3.  **Vision feature projection to LM dimension** (IBP + CROWN)
//! 4.  **Cross-attention vision-to-text bounds** (IBP)
//! 5.  **SwiGLU FFN in decoder blocks** (IBP + CROWN)
//! 6.  **RMSNorm normalization bounds** (IBP + CROWN)
//! 7.  **Full vision encoder pipeline composition** (IBP)
//! 8.  **Full decoder block pipeline** (IBP)
//! 9.  **Vision-to-language cross-modal pipeline** (IBP)
//! 10. **CTC/autoregressive output logit bounds** (IBP)
//! 11. **Multi-resolution vision feature extraction** (IBP)
//! 12. **LM head projection bounds** (IBP + CROWN)
//! 13. **Residual connections through encoder** (IBP)
//! 14. **Residual connections through decoder** (IBP)
//! 15. **Two-block encoder composition** (IBP + CROWN)
//! 16. **Two-block decoder composition** (IBP)
//! 17. **Embedding + position encoding bounds** (IBP)
//! 18. **End-to-end vision-to-logit pipeline** (IBP)
//!
//! Architecture references:
//! - FireRed-OCR: Qwen3-VL-2B variant for document OCR
//! - SigLIP2 (Zhai et al., 2023): Sigmoid-loss pre-trained ViT encoder
//! - SwiGLU (Shazeer, 2020): Gated linear unit with SiLU activation
//! - RMSNorm (Zhang & Sennrich, 2019): Root mean square layer normalization
//! - CTC (Graves et al., 2006): Connectionist Temporal Classification
//!
//! Dimensions (small for fast verification, structurally representative):
//! - HIDDEN_DIM=4, FFN_DIM=8, NUM_HEADS=2, PATCH_DIM=4,
//!   SEQ_LEN=4, VOCAB_SIZE=6, IMG_PATCHES=4
//!
//! Part of #4196: Compose tests for FireRed-OCR full pipeline.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::AttentionMask;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const HIDDEN_DIM: usize = 4;
const FFN_DIM: usize = 8;
const NUM_HEADS: usize = 2;
const PATCH_DIM: usize = 4;
const SEQ_LEN: usize = 4;
const VOCAB_SIZE: usize = 6;
const IMG_PATCHES: usize = 4;
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Constant weight tensor binding.
fn weight(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
}

/// Zero bias tensor binding.
fn bias_zero(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}

/// Ones tensor binding (for RMSNorm weight).
fn ones(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 1.0f32))
}

/// Scalar epsilon binding.
fn eps_binding() -> TensorParamBinding {
    TensorParamBinding::ConstantScalar(1e-5)
}

/// Sequence-domain input bounds: embeddings in [-range, +range].
fn seq_bounds(seq_len: usize, dim: usize, range: f32) -> BoundedTensor {
    uniform_bounds(&[seq_len, dim], range)
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

/// Build SwiGLU FFN block: gate_proj -> SiLU -> mul(up_proj) -> down_proj.
/// Returns (output_node, bindings_to_append).
fn add_swiglu_ffn(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    prefix: &str,
    seq_len: usize,
    dim: usize,
    ffn_dim: usize,
) -> nn_dsl::TensorNodeId {
    let ffn_shape = [seq_len, ffn_dim];
    let out_shape = [seq_len, dim];

    let gate_w = b.add_input(&format!("{prefix}gate_w"), &[ffn_dim, dim]);
    let up_w = b.add_input(&format!("{prefix}up_w"), &[ffn_dim, dim]);
    let down_w = b.add_input(&format!("{prefix}down_w"), &[dim, ffn_dim]);

    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_act = add_silu(b, gate, &ffn_shape);
    let up = b.add_linear(input, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    b.add_linear(hidden, down_w, None, &out_shape)
}

/// Push SwiGLU FFN bindings (3 params: gate_w, up_w, down_w).
fn push_swiglu_bindings(bindings: &mut Vec<TensorParamBinding>) {
    bindings.push(weight(&[FFN_DIM, HIDDEN_DIM])); // gate_w
    bindings.push(weight(&[FFN_DIM, HIDDEN_DIM])); // up_w
    bindings.push(weight(&[HIDDEN_DIM, FFN_DIM])); // down_w
}

/// Build a single encoder block: RMSNorm -> Attention -> residual -> RMSNorm -> SwiGLU -> residual.
fn add_encoder_block(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    prefix: &str,
    seq_len: usize,
) -> nn_dsl::TensorNodeId {
    let shape = [seq_len, HIDDEN_DIM];
    let head_dim = HIDDEN_DIM / NUM_HEADS;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Pre-attention RMSNorm
    let n1_eps = b.add_input(&format!("{prefix}norm1_eps"), &[1]);
    let n1_w = b.add_input(&format!("{prefix}norm1_w"), &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, n1_eps, 1, n1_w, &shape);

    // Self-attention
    let q_w = b.add_input(&format!("{prefix}q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input(&format!("{prefix}k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input(&format!("{prefix}v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input(&format!("{prefix}out_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);

    // Residual
    let res1 = b.add_binary_add(input, attn_out, &shape);

    // Pre-FFN RMSNorm
    let n2_eps = b.add_input(&format!("{prefix}norm2_eps"), &[1]);
    let n2_w = b.add_input(&format!("{prefix}norm2_w"), &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &shape);

    // SwiGLU FFN
    let ffn_out = add_swiglu_ffn(b, normed2, prefix, seq_len, HIDDEN_DIM, FFN_DIM);

    // Residual
    b.add_binary_add(res1, ffn_out, &shape)
}

/// Push one encoder block's bindings (11 params).
fn push_encoder_block_bindings(bindings: &mut Vec<TensorParamBinding>) {
    bindings.push(eps_binding()); // norm1_eps
    bindings.push(ones(&[HIDDEN_DIM])); // norm1_w
    bindings.push(weight(&[HIDDEN_DIM, HIDDEN_DIM])); // q_w
    bindings.push(weight(&[HIDDEN_DIM, HIDDEN_DIM])); // k_w
    bindings.push(weight(&[HIDDEN_DIM, HIDDEN_DIM])); // v_w
    bindings.push(weight(&[HIDDEN_DIM, HIDDEN_DIM])); // out_w
    bindings.push(eps_binding()); // norm2_eps
    bindings.push(ones(&[HIDDEN_DIM])); // norm2_w
    push_swiglu_bindings(bindings); // gate_w, up_w, down_w
}

/// Build a decoder block with cross-attention.
///
/// RMSNorm -> Self-attention -> residual -> RMSNorm -> Cross-attention ->
/// residual -> RMSNorm -> SwiGLU FFN -> residual.
fn add_decoder_block(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    encoder_memory: nn_dsl::TensorNodeId,
    prefix: &str,
    dec_seq: usize,
    enc_seq: usize,
) -> nn_dsl::TensorNodeId {
    let dec_shape = [dec_seq, HIDDEN_DIM];
    let enc_shape = [enc_seq, HIDDEN_DIM];
    let head_dim = HIDDEN_DIM / NUM_HEADS;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Pre-self-attention RMSNorm
    let n1_eps = b.add_input(&format!("{prefix}sa_norm_eps"), &[1]);
    let n1_w = b.add_input(&format!("{prefix}sa_norm_w"), &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, n1_eps, 1, n1_w, &dec_shape);

    // Self-attention
    let sa_q_w = b.add_input(&format!("{prefix}sa_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let sa_k_w = b.add_input(&format!("{prefix}sa_k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let sa_v_w = b.add_input(&format!("{prefix}sa_v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let sa_out_w = b.add_input(&format!("{prefix}sa_out_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let sa_q = b.add_linear(normed1, sa_q_w, None, &dec_shape);
    let sa_k = b.add_linear(normed1, sa_k_w, None, &dec_shape);
    let sa_v = b.add_linear(normed1, sa_v_w, None, &dec_shape);
    let sa_attn = b.add_attention(
        sa_q,
        sa_k,
        sa_v,
        AttentionMask::Causal,
        Some(scale),
        &dec_shape,
    );
    let sa_out = b.add_linear(sa_attn, sa_out_w, None, &dec_shape);
    let res1 = b.add_binary_add(input, sa_out, &dec_shape);

    // Pre-cross-attention RMSNorm
    let n2_eps = b.add_input(&format!("{prefix}ca_norm_eps"), &[1]);
    let n2_w = b.add_input(&format!("{prefix}ca_norm_w"), &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &dec_shape);

    // Cross-attention (decoder queries attend to encoder keys/values)
    let ca_q_w = b.add_input(&format!("{prefix}ca_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let ca_k_w = b.add_input(&format!("{prefix}ca_k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let ca_v_w = b.add_input(&format!("{prefix}ca_v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let ca_out_w = b.add_input(&format!("{prefix}ca_out_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let ca_q = b.add_linear(normed2, ca_q_w, None, &dec_shape);
    let ca_k = b.add_linear(encoder_memory, ca_k_w, None, &enc_shape);
    let ca_v = b.add_linear(encoder_memory, ca_v_w, None, &enc_shape);
    let ca_attn = b.add_attention(
        ca_q,
        ca_k,
        ca_v,
        AttentionMask::Standard,
        Some(scale),
        &dec_shape,
    );
    let ca_out = b.add_linear(ca_attn, ca_out_w, None, &dec_shape);
    let res2 = b.add_binary_add(res1, ca_out, &dec_shape);

    // Pre-FFN RMSNorm
    let n3_eps = b.add_input(&format!("{prefix}ffn_norm_eps"), &[1]);
    let n3_w = b.add_input(&format!("{prefix}ffn_norm_w"), &[HIDDEN_DIM]);
    let normed3 = b.add_rms_norm(res2, n3_eps, 1, n3_w, &dec_shape);

    // SwiGLU FFN
    let ffn_out = add_swiglu_ffn(b, normed3, prefix, dec_seq, HIDDEN_DIM, FFN_DIM);

    b.add_binary_add(res2, ffn_out, &dec_shape)
}

/// Push one decoder block's bindings (19 params).
fn push_decoder_block_bindings(bindings: &mut Vec<TensorParamBinding>) {
    let norm_w = ones(&[HIDDEN_DIM]);
    let qkvo_w = weight(&[HIDDEN_DIM, HIDDEN_DIM]);

    // Self-attention norm + weights
    bindings.push(eps_binding()); // sa_norm_eps
    bindings.push(norm_w.clone()); // sa_norm_w
    bindings.push(qkvo_w.clone()); // sa_q_w
    bindings.push(qkvo_w.clone()); // sa_k_w
    bindings.push(qkvo_w.clone()); // sa_v_w
    bindings.push(qkvo_w.clone()); // sa_out_w

    // Cross-attention norm + weights
    bindings.push(eps_binding()); // ca_norm_eps
    bindings.push(norm_w.clone()); // ca_norm_w
    bindings.push(qkvo_w.clone()); // ca_q_w
    bindings.push(qkvo_w.clone()); // ca_k_w
    bindings.push(qkvo_w.clone()); // ca_v_w
    bindings.push(qkvo_w); // ca_out_w

    // FFN norm + SwiGLU weights
    bindings.push(eps_binding()); // ffn_norm_eps
    bindings.push(norm_w); // ffn_norm_w
    push_swiglu_bindings(bindings); // gate_w, up_w, down_w
}

// ===========================================================================
// 1. SigLIP2 vision encoder patch embedding bounds (IBP)
// ===========================================================================

#[test]
fn test_firered_full_patch_embedding_ibp() {
    // Patch embedding: Linear projection from PATCH_DIM to HIDDEN_DIM.
    // Models Conv2d patch embed as linear over flattened patches.
    let mut b = TensorBlockBuilder::new("fr_full_patch_embed");
    let input = b.add_input("patches", &[IMG_PATCHES, PATCH_DIM]);
    let proj_w = b.add_input("patch_proj_w", &[HIDDEN_DIM, PATCH_DIM]);
    let proj_b = b.add_input("patch_proj_b", &[HIDDEN_DIM]);
    let out = b.add_linear(input, proj_w, Some(proj_b), &[IMG_PATCHES, HIDDEN_DIM]);
    let def = b.build(out).expect("valid patch embed kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, PATCH_DIM]),
        bias_zero(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IMG_PATCHES, PATCH_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[IMG_PATCHES, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FR full patch embed IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 2. Window attention ViT block bounds (IBP)
// ===========================================================================

#[test]
fn test_firered_full_window_attention_ibp() {
    // Window attention ViT block: RMSNorm -> self-attention -> residual.
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let head_dim = HIDDEN_DIM / NUM_HEADS;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut b = TensorBlockBuilder::new("fr_full_window_attn");
    let input = b.add_input("x", &shape);

    // Pre-attention RMSNorm
    let n_eps = b.add_input("norm_eps", &[1]);
    let n_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, n_eps, 1, n_w, &shape);

    // Self-attention with standard (window) mask
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed, q_w, None, &shape);
    let k = b.add_linear(normed, k_w, None, &shape);
    let v = b.add_linear(normed, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);

    let out = b.add_binary_add(input, attn_out, &shape);
    let def = b.build(out).expect("valid window attention kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        eps_binding(),
        ones(&[HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FR full window attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 3. Vision feature projection to LM dimension (IBP + CROWN)
// ===========================================================================

#[test]
fn test_firered_full_vision_projection_ibp_crown() {
    // Linear projection from vision encoder dimension to LM embedding space.
    let lm_dim = HIDDEN_DIM * 2; // 8
    let mut b = TensorBlockBuilder::new("fr_full_vision_proj");
    let input = b.add_input("encoder_features", &[IMG_PATCHES, HIDDEN_DIM]);
    let proj_w = b.add_input("proj_w", &[lm_dim, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_b", &[lm_dim]);
    let out = b.add_linear(input, proj_w, Some(proj_b), &[IMG_PATCHES, lm_dim]);
    let def = b.build(out).expect("valid vision projection kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[lm_dim, HIDDEN_DIM]),
        bias_zero(&[lm_dim]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(IMG_PATCHES, HIDDEN_DIM, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    assert_eq!(ibp_output.lower_upper().0.shape(), &[IMG_PATCHES, lm_dim]);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("FR full vision projection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("FR full vision projection CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 4. Cross-attention vision-to-text bounds (IBP)
// ===========================================================================

#[test]
fn test_firered_full_cross_attention_ibp() {
    // Decoder queries attend to encoder visual features via cross-attention.
    let dec_shape = [SEQ_LEN, HIDDEN_DIM];
    let enc_shape = [IMG_PATCHES, HIDDEN_DIM];
    let head_dim = HIDDEN_DIM / NUM_HEADS;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut b = TensorBlockBuilder::new("fr_full_cross_attn");
    let dec_input = b.add_input("dec_tokens", &dec_shape);
    let enc_memory = b.add_input("enc_features", &enc_shape);

    let ca_q_w = b.add_input("ca_q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let ca_k_w = b.add_input("ca_k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let ca_v_w = b.add_input("ca_v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let ca_out_w = b.add_input("ca_out_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let ca_q = b.add_linear(dec_input, ca_q_w, None, &dec_shape);
    let ca_k = b.add_linear(enc_memory, ca_k_w, None, &enc_shape);
    let ca_v = b.add_linear(enc_memory, ca_v_w, None, &enc_shape);
    let ca_attn = b.add_attention(
        ca_q,
        ca_k,
        ca_v,
        AttentionMask::Standard,
        Some(scale),
        &dec_shape,
    );
    let out = b.add_linear(ca_attn, ca_out_w, None, &dec_shape);
    let def = b.build(out).expect("valid cross-attention kernel");

    // Encoder memory is a constant (pre-computed vision features).
    let enc_data = ArrayD::from_elem(IxDyn(&enc_shape), 0.5f32);
    let bindings = vec![
        TensorParamBinding::Variable,                 // dec_tokens
        TensorParamBinding::ConstantTensor(enc_data), // enc_features
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),            // ca_q_w
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),            // ca_k_w
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),            // ca_v_w
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),            // ca_out_w
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FR full cross-attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 5. SwiGLU FFN in decoder blocks (IBP + CROWN)
// ===========================================================================

#[test]
fn test_firered_full_swiglu_ffn_ibp_crown() {
    let mut b = TensorBlockBuilder::new("fr_full_swiglu");
    let input = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);
    let ffn_out = add_swiglu_ffn(&mut b, input, "ffn_", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let def = b.build(ffn_out).expect("valid SwiGLU kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_swiglu_bindings(&mut bindings);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("FR full SwiGLU FFN IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("FR full SwiGLU FFN CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 6. RMSNorm normalization bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_firered_full_rmsnorm_ibp_crown() {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("fr_full_rmsnorm");
    let input = b.add_input("x", &shape);
    let eps = b.add_input("eps", &[1]);
    let w = b.add_input("w", &[HIDDEN_DIM]);
    let out = b.add_rms_norm(input, eps, 1, w, &shape);
    let def = b.build(out).expect("valid RMSNorm kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        eps_binding(),
        ones(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 2.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("FR full RMSNorm IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("FR full RMSNorm CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 7. Full vision encoder pipeline composition (IBP)
// ===========================================================================

#[test]
fn test_firered_full_vision_encoder_pipeline_ibp() {
    // Patch embed -> 2 encoder blocks -> final RMSNorm.
    let mut b = TensorBlockBuilder::new("fr_full_vision_encoder");
    let input = b.add_input("patches", &[IMG_PATCHES, PATCH_DIM]);

    // Patch embedding
    let proj_w = b.add_input("patch_proj_w", &[HIDDEN_DIM, PATCH_DIM]);
    let proj_b = b.add_input("patch_proj_b", &[HIDDEN_DIM]);
    let embedded = b.add_linear(input, proj_w, Some(proj_b), &[IMG_PATCHES, HIDDEN_DIM]);

    // 2 encoder blocks
    let l1 = add_encoder_block(&mut b, embedded, "enc0_", IMG_PATCHES);
    let l2 = add_encoder_block(&mut b, l1, "enc1_", IMG_PATCHES);

    // Final RMSNorm
    let fn_eps = b.add_input("final_norm_eps", &[1]);
    let fn_w = b.add_input("final_norm_w", &[HIDDEN_DIM]);
    let out = b.add_rms_norm(l2, fn_eps, 1, fn_w, &[IMG_PATCHES, HIDDEN_DIM]);
    let def = b.build(out).expect("valid vision encoder pipeline kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, PATCH_DIM]),
        bias_zero(&[HIDDEN_DIM]),
    ];
    push_encoder_block_bindings(&mut bindings);
    push_encoder_block_bindings(&mut bindings);
    bindings.push(eps_binding()); // final_norm_eps
    bindings.push(ones(&[HIDDEN_DIM])); // final_norm_w

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = uniform_bounds(&[IMG_PATCHES, PATCH_DIM], 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[IMG_PATCHES, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FR full vision encoder pipeline IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 8. Full decoder block pipeline (IBP)
// ===========================================================================

#[test]
fn test_firered_full_decoder_block_pipeline_ibp() {
    // Single decoder block with cross-attention to constant encoder memory.
    let mut b = TensorBlockBuilder::new("fr_full_decoder_block");
    let dec_input = b.add_input("dec_tokens", &[SEQ_LEN, HIDDEN_DIM]);
    let enc_memory = b.add_input("enc_features", &[IMG_PATCHES, HIDDEN_DIM]);

    let out = add_decoder_block(&mut b, dec_input, enc_memory, "dec0_", SEQ_LEN, IMG_PATCHES);
    let def = b.build(out).expect("valid decoder block kernel");

    let enc_data = ArrayD::from_elem(IxDyn(&[IMG_PATCHES, HIDDEN_DIM]), 0.5f32);
    let mut bindings = vec![
        TensorParamBinding::Variable,                 // dec_tokens
        TensorParamBinding::ConstantTensor(enc_data), // enc_features
    ];
    push_decoder_block_bindings(&mut bindings);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FR full decoder block IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 9. Vision-to-language cross-modal pipeline (IBP)
// ===========================================================================

#[test]
fn test_firered_full_vision_to_language_pipeline_ibp() {
    // Encoder block -> VL projection -> cross-attention (decoder queries
    // attend to projected vision features).
    let lm_dim = HIDDEN_DIM * 2;

    let mut b = TensorBlockBuilder::new("fr_full_v2l_pipeline");
    let vis_input = b.add_input("vis_features", &[IMG_PATCHES, HIDDEN_DIM]);

    // Single encoder block
    let enc_out = add_encoder_block(&mut b, vis_input, "enc_", IMG_PATCHES);

    // VL projection
    let vp_w = b.add_input("vl_proj_w", &[lm_dim, HIDDEN_DIM]);
    let vp_b = b.add_input("vl_proj_b", &[lm_dim]);
    let projected = b.add_linear(enc_out, vp_w, Some(vp_b), &[IMG_PATCHES, lm_dim]);

    // Cross-attention: decoder queries attend to projected vision features.
    // Decoder tokens are modeled as constants (pre-initialized embeddings).
    let dec_input = b.add_input("dec_tokens", &[SEQ_LEN, lm_dim]);
    let head_dim = lm_dim / NUM_HEADS;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let ca_q_w = b.add_input("ca_q_w", &[lm_dim, lm_dim]);
    let ca_k_w = b.add_input("ca_k_w", &[lm_dim, lm_dim]);
    let ca_v_w = b.add_input("ca_v_w", &[lm_dim, lm_dim]);
    let ca_out_w = b.add_input("ca_out_w", &[lm_dim, lm_dim]);

    let ca_q = b.add_linear(dec_input, ca_q_w, None, &[SEQ_LEN, lm_dim]);
    let ca_k = b.add_linear(projected, ca_k_w, None, &[IMG_PATCHES, lm_dim]);
    let ca_v = b.add_linear(projected, ca_v_w, None, &[IMG_PATCHES, lm_dim]);
    let ca_attn = b.add_attention(
        ca_q,
        ca_k,
        ca_v,
        AttentionMask::Standard,
        Some(scale),
        &[SEQ_LEN, lm_dim],
    );
    let out = b.add_linear(ca_attn, ca_out_w, None, &[SEQ_LEN, lm_dim]);
    let def = b.build(out).expect("valid V2L pipeline kernel");

    let dec_data = ArrayD::from_elem(IxDyn(&[SEQ_LEN, lm_dim]), 0.1f32);
    let mut bindings = vec![TensorParamBinding::Variable]; // vis_features
    push_encoder_block_bindings(&mut bindings);
    bindings.push(weight(&[lm_dim, HIDDEN_DIM])); // vl_proj_w
    bindings.push(bias_zero(&[lm_dim])); // vl_proj_b
    bindings.push(TensorParamBinding::ConstantTensor(dec_data)); // dec_tokens
    bindings.push(weight(&[lm_dim, lm_dim])); // ca_q_w
    bindings.push(weight(&[lm_dim, lm_dim])); // ca_k_w
    bindings.push(weight(&[lm_dim, lm_dim])); // ca_v_w
    bindings.push(weight(&[lm_dim, lm_dim])); // ca_out_w

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(IMG_PATCHES, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, lm_dim]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FR full V2L pipeline IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 10. CTC/autoregressive output logit bounds (IBP)
// ===========================================================================

#[test]
fn test_firered_full_ctc_logit_bounds_ibp() {
    // Linear -> Softmax producing character probabilities in [0, 1].
    let mut b = TensorBlockBuilder::new("fr_full_ctc_logits");
    let input = b.add_input("decoder_features", &[SEQ_LEN, HIDDEN_DIM]);
    let lm_w = b.add_input("lm_head_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let lm_b = b.add_input("lm_head_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(input, lm_w, Some(lm_b), &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(probs).expect("valid CTC logit kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        bias_zero(&[VOCAB_SIZE]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FR full CTC logit IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Softmax outputs must be in [0, 1]
    assert!(
        lo_min >= -1e-5,
        "softmax lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-5,
        "softmax upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 11. Multi-resolution vision feature extraction (IBP)
// ===========================================================================

#[test]
fn test_firered_full_multi_resolution_ibp() {
    // Two patch embedding branches at different patch sizes, concatenated.
    // Branch 1: PATCH_DIM -> HIDDEN_DIM (full patches)
    // Branch 2: PATCH_DIM*2 -> HIDDEN_DIM (larger patches, half sequence)
    let half_patches = IMG_PATCHES / 2;
    let large_patch_dim = PATCH_DIM * 2;

    let mut b = TensorBlockBuilder::new("fr_full_multi_res");
    let input1 = b.add_input("patches_fine", &[IMG_PATCHES, PATCH_DIM]);
    let input2 = b.add_input("patches_coarse", &[half_patches, large_patch_dim]);

    // Branch 1 projection
    let w1 = b.add_input("proj1_w", &[HIDDEN_DIM, PATCH_DIM]);
    let b1 = b.add_input("proj1_b", &[HIDDEN_DIM]);
    let fine = b.add_linear(input1, w1, Some(b1), &[IMG_PATCHES, HIDDEN_DIM]);

    // Branch 2 projection
    let w2 = b.add_input("proj2_w", &[HIDDEN_DIM, large_patch_dim]);
    let b2 = b.add_input("proj2_b", &[HIDDEN_DIM]);
    let coarse = b.add_linear(input2, w2, Some(b2), &[half_patches, HIDDEN_DIM]);

    // Concatenate along sequence dimension
    let total_seq = IMG_PATCHES + half_patches;
    let concat = b.add_concat(&[fine, coarse], 0, &[total_seq, HIDDEN_DIM]);
    let def = b.build(concat).expect("valid multi-resolution kernel");

    let coarse_data = ArrayD::from_elem(IxDyn(&[half_patches, large_patch_dim]), 0.5f32);
    let bindings = vec![
        TensorParamBinding::Variable,                    // patches_fine
        TensorParamBinding::ConstantTensor(coarse_data), // patches_coarse
        weight(&[HIDDEN_DIM, PATCH_DIM]),                // proj1_w
        bias_zero(&[HIDDEN_DIM]),                        // proj1_b
        weight(&[HIDDEN_DIM, large_patch_dim]),          // proj2_w
        bias_zero(&[HIDDEN_DIM]),                        // proj2_b
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = uniform_bounds(&[IMG_PATCHES, PATCH_DIM], 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[total_seq, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FR full multi-resolution IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 12. LM head projection bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_firered_full_lm_head_ibp_crown() {
    // RMSNorm -> Linear projection to vocab logits.
    let mut b = TensorBlockBuilder::new("fr_full_lm_head");
    let input = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);
    let n_eps = b.add_input("norm_eps", &[1]);
    let n_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, n_eps, 1, n_w, &[SEQ_LEN, HIDDEN_DIM]);
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let out = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid LM head kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        eps_binding(),
        ones(&[HIDDEN_DIM]),
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    assert_eq!(ibp_output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("FR full LM head IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("FR full LM head CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 13. Residual connections through encoder (IBP)
// ===========================================================================

#[test]
fn test_firered_full_encoder_residual_growth_ibp() {
    // Compare bounds after 1, 2, and 3 encoder blocks.
    let build_n_blocks = |n: usize| -> BoundedTensor {
        let mut b = TensorBlockBuilder::new(&format!("fr_full_enc_res_{n}"));
        let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
        let mut x = input;
        for i in 0..n {
            x = add_encoder_block(&mut b, x, &format!("blk{i}_"), SEQ_LEN);
        }
        let def = b.build(x).expect("valid n-block encoder");
        let mut bindings = vec![TensorParamBinding::Variable];
        for _ in 0..n {
            push_encoder_block_bindings(&mut bindings);
        }
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);
        graph.propagate_ibp(&inp).expect("IBP")
    };

    let out1 = build_n_blocks(1);
    let out2 = build_n_blocks(2);
    let out3 = build_n_blocks(3);
    assert_bounds_valid(&out1);
    assert_bounds_valid(&out2);
    assert_bounds_valid(&out3);

    let (l1, h1) = bounds_min_max(&out1);
    let (l2, h2) = bounds_min_max(&out2);
    let (l3, h3) = bounds_min_max(&out3);
    let w1 = h1 - l1;
    let w2 = h2 - l2;
    let w3 = h3 - l3;

    eprintln!("FR encoder residual: 1-blk={w1:.4}, 2-blk={w2:.4}, 3-blk={w3:.4}");
    assert!(w1.is_finite() && w2.is_finite() && w3.is_finite());
}

// ===========================================================================
// 14. Residual connections through decoder (IBP)
// ===========================================================================

#[test]
fn test_firered_full_decoder_residual_growth_ibp() {
    // Compare bounds after 1 and 2 decoder blocks with cross-attention.
    let build_n_decoder = |n: usize| -> BoundedTensor {
        let mut b = TensorBlockBuilder::new(&format!("fr_full_dec_res_{n}"));
        let dec_input = b.add_input("dec_tokens", &[SEQ_LEN, HIDDEN_DIM]);
        let enc_memory = b.add_input("enc_features", &[IMG_PATCHES, HIDDEN_DIM]);

        let mut x = dec_input;
        for i in 0..n {
            x = add_decoder_block(
                &mut b,
                x,
                enc_memory,
                &format!("dec{i}_"),
                SEQ_LEN,
                IMG_PATCHES,
            );
        }
        let def = b.build(x).expect("valid n-block decoder");

        let enc_data = ArrayD::from_elem(IxDyn(&[IMG_PATCHES, HIDDEN_DIM]), 0.5f32);
        let mut bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(enc_data),
        ];
        for _ in 0..n {
            push_decoder_block_bindings(&mut bindings);
        }
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);
        graph.propagate_ibp(&inp).expect("IBP")
    };

    let out1 = build_n_decoder(1);
    let out2 = build_n_decoder(2);
    assert_bounds_valid(&out1);
    assert_bounds_valid(&out2);

    let (l1, h1) = bounds_min_max(&out1);
    let (l2, h2) = bounds_min_max(&out2);
    let w1 = h1 - l1;
    let w2 = h2 - l2;

    eprintln!("FR decoder residual: 1-blk={w1:.4}, 2-blk={w2:.4}");
    assert!(w1.is_finite() && w2.is_finite());
}

// ===========================================================================
// 15. Two-block encoder composition (IBP + CROWN)
// ===========================================================================

#[test]
fn test_firered_full_two_block_encoder_ibp_crown() {
    let mut b = TensorBlockBuilder::new("fr_full_2blk_enc");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let l1 = add_encoder_block(&mut b, input, "enc0_", SEQ_LEN);
    let l2 = add_encoder_block(&mut b, l1, "enc1_", SEQ_LEN);
    let def = b.build(l2).expect("valid 2-block encoder kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_encoder_block_bindings(&mut bindings);
    push_encoder_block_bindings(&mut bindings);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("FR full 2-block encoder IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("FR full 2-block encoder CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 16. Two-block decoder composition (IBP)
// ===========================================================================

#[test]
fn test_firered_full_two_block_decoder_ibp() {
    let mut b = TensorBlockBuilder::new("fr_full_2blk_dec");
    let dec_input = b.add_input("dec_tokens", &[SEQ_LEN, HIDDEN_DIM]);
    let enc_memory = b.add_input("enc_features", &[IMG_PATCHES, HIDDEN_DIM]);

    let l1 = add_decoder_block(&mut b, dec_input, enc_memory, "dec0_", SEQ_LEN, IMG_PATCHES);
    let l2 = add_decoder_block(&mut b, l1, enc_memory, "dec1_", SEQ_LEN, IMG_PATCHES);
    let def = b.build(l2).expect("valid 2-block decoder kernel");

    let enc_data = ArrayD::from_elem(IxDyn(&[IMG_PATCHES, HIDDEN_DIM]), 0.5f32);
    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(enc_data),
    ];
    push_decoder_block_bindings(&mut bindings);
    push_decoder_block_bindings(&mut bindings);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FR full 2-block decoder IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 17. Embedding + position encoding bounds (IBP)
// ===========================================================================

#[test]
fn test_firered_full_embedding_pos_encoding_ibp() {
    // Patch embedding + additive learned positional encoding.
    let mut b = TensorBlockBuilder::new("fr_full_embed_pos");
    let input = b.add_input("patches", &[IMG_PATCHES, PATCH_DIM]);

    // Patch embedding
    let proj_w = b.add_input("patch_proj_w", &[HIDDEN_DIM, PATCH_DIM]);
    let proj_b = b.add_input("patch_proj_b", &[HIDDEN_DIM]);
    let embedded = b.add_linear(input, proj_w, Some(proj_b), &[IMG_PATCHES, HIDDEN_DIM]);

    // Position encoding: constant additive
    let pos_embed = b.add_input("pos_embed", &[IMG_PATCHES, HIDDEN_DIM]);
    let out = b.add_binary_add(embedded, pos_embed, &[IMG_PATCHES, HIDDEN_DIM]);
    let def = b.build(out).expect("valid embedding + pos encoding kernel");

    let pe_data = ArrayD::from_elem(IxDyn(&[IMG_PATCHES, HIDDEN_DIM]), 0.01f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, PATCH_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        TensorParamBinding::ConstantTensor(pe_data),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = uniform_bounds(&[IMG_PATCHES, PATCH_DIM], 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[IMG_PATCHES, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FR full embed + pos encoding IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 18. End-to-end vision-to-logit pipeline (IBP)
// ===========================================================================

#[test]
fn test_firered_full_end_to_end_vision_to_logit_ibp() {
    // Patch embed -> encoder block -> VL projection -> decoder block
    // (with cross-attention) -> RMSNorm -> LM head -> softmax.
    let lm_dim = HIDDEN_DIM * 2;

    let mut b = TensorBlockBuilder::new("fr_full_e2e_v2logit");

    // Vision encoder: patch embed + 1 encoder block
    let vis_input = b.add_input("patches", &[IMG_PATCHES, PATCH_DIM]);
    let proj_w = b.add_input("patch_proj_w", &[HIDDEN_DIM, PATCH_DIM]);
    let proj_b = b.add_input("patch_proj_b", &[HIDDEN_DIM]);
    let embedded = b.add_linear(vis_input, proj_w, Some(proj_b), &[IMG_PATCHES, HIDDEN_DIM]);
    let enc_out = add_encoder_block(&mut b, embedded, "enc_", IMG_PATCHES);

    // VL projection
    let vp_w = b.add_input("vl_proj_w", &[lm_dim, HIDDEN_DIM]);
    let vp_b = b.add_input("vl_proj_b", &[lm_dim]);
    let vis_projected = b.add_linear(enc_out, vp_w, Some(vp_b), &[IMG_PATCHES, lm_dim]);

    // Decoder: single decoder block with cross-attention to vision.
    // Decoder input and encoder memory share the VL-projected dimension.
    let dec_input = b.add_input("dec_tokens", &[SEQ_LEN, lm_dim]);

    // Simplified decoder: cross-attention + SwiGLU (no self-attention to keep tractable)
    let head_dim_dec = lm_dim / NUM_HEADS;
    let scale_dec = 1.0 / (head_dim_dec as f32).sqrt();

    let ca_q_w = b.add_input("dec_ca_q_w", &[lm_dim, lm_dim]);
    let ca_k_w = b.add_input("dec_ca_k_w", &[lm_dim, lm_dim]);
    let ca_v_w = b.add_input("dec_ca_v_w", &[lm_dim, lm_dim]);
    let ca_out_w = b.add_input("dec_ca_out_w", &[lm_dim, lm_dim]);

    let ca_q = b.add_linear(dec_input, ca_q_w, None, &[SEQ_LEN, lm_dim]);
    let ca_k = b.add_linear(vis_projected, ca_k_w, None, &[IMG_PATCHES, lm_dim]);
    let ca_v = b.add_linear(vis_projected, ca_v_w, None, &[IMG_PATCHES, lm_dim]);
    let ca_attn = b.add_attention(
        ca_q,
        ca_k,
        ca_v,
        AttentionMask::Standard,
        Some(scale_dec),
        &[SEQ_LEN, lm_dim],
    );
    let ca_out = b.add_linear(ca_attn, ca_out_w, None, &[SEQ_LEN, lm_dim]);
    let dec_res = b.add_binary_add(dec_input, ca_out, &[SEQ_LEN, lm_dim]);

    // Final RMSNorm
    let fn_eps = b.add_input("final_norm_eps", &[1]);
    let fn_w = b.add_input("final_norm_w", &[lm_dim]);
    let normed = b.add_rms_norm(dec_res, fn_eps, 1, fn_w, &[SEQ_LEN, lm_dim]);

    // LM head -> softmax
    let lm_w = b.add_input("lm_head_w", &[VOCAB_SIZE, lm_dim]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(probs).expect("valid end-to-end pipeline kernel");

    let dec_data = ArrayD::from_elem(IxDyn(&[SEQ_LEN, lm_dim]), 0.1f32);
    let mut bindings = vec![
        TensorParamBinding::Variable,     // patches
        weight(&[HIDDEN_DIM, PATCH_DIM]), // patch_proj_w
        bias_zero(&[HIDDEN_DIM]),         // patch_proj_b
    ];
    push_encoder_block_bindings(&mut bindings);
    bindings.push(weight(&[lm_dim, HIDDEN_DIM])); // vl_proj_w
    bindings.push(bias_zero(&[lm_dim])); // vl_proj_b
    bindings.push(TensorParamBinding::ConstantTensor(dec_data)); // dec_tokens
    bindings.push(weight(&[lm_dim, lm_dim])); // dec_ca_q_w
    bindings.push(weight(&[lm_dim, lm_dim])); // dec_ca_k_w
    bindings.push(weight(&[lm_dim, lm_dim])); // dec_ca_v_w
    bindings.push(weight(&[lm_dim, lm_dim])); // dec_ca_out_w
    bindings.push(eps_binding()); // final_norm_eps
    bindings.push(ones(&[lm_dim])); // final_norm_w
    bindings.push(weight(&[VOCAB_SIZE, lm_dim])); // lm_head_w

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = uniform_bounds(&[IMG_PATCHES, PATCH_DIM], 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FR full E2E vision-to-logit IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Softmax outputs must be in [0, 1]
    assert!(
        lo_min >= -1e-5,
        "softmax lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-5,
        "softmax upper bound must be <= 1, got {hi_max}"
    );
}
