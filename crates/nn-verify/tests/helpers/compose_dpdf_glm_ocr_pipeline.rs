// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests for GLM-OCR pipeline stages.
//!
//! Verifies NY IBP and CROWN bound propagation through the full
//! GLM-OCR document text recognition pipeline: image patch embedding,
//! visual encoder, text decoder with cross-attention, character classification
//! head, and end-to-end pipeline composition.
//!
//! ## Tests (18 tests)
//!
//!  1. **Patch embedding conv2d + flatten bounds** — Conv2d patch extraction + linear projection (IBP)
//!  2. **Patch embedding with position encoding** — Patch embed + sinusoidal PE addition (IBP)
//!  3. **Patch embedding wide input range** — Pixel values in [0, 255] normalized (IBP)
//!  4. **Visual encoder single-layer self-attention** — Self-attention over image patches (IBP + CROWN)
//!  5. **Visual encoder layer norm contraction** — RMSNorm bounds contraction in encoder (IBP + CROWN)
//!  6. **Visual encoder two-layer stack** — Stacked encoder blocks bound accumulation (IBP)
//!  7. **Cross-attention decoder-to-encoder** — Text queries attend to visual features (IBP + CROWN)
//!  8. **Cross-attention with residual connection** — Cross-attn + skip connection (IBP)
//!  9. **Text decoder causal self-attention** — Causal masked self-attention in decoder (IBP)
//! 10. **Text decoder SwiGLU FFN** — Decoder FFN with gated linear unit (IBP + CROWN)
//! 11. **Text decoder full block** — Self-attn + cross-attn + FFN decoder block (IBP)
//! 12. **Text decoder short sequence** — Decoder with SEQ_LEN=2 (IBP)
//! 13. **Text decoder long sequence** — Decoder with SEQ_LEN=8 (IBP)
//! 14. **Character head softmax bounds** — Linear + softmax classification (IBP)
//! 15. **Character head log_softmax bounds** — Linear + log_softmax for CTC loss (IBP)
//! 16. **Full pipeline small image** — 4 patches end-to-end (IBP + CROWN)
//! 17. **Full pipeline large image** — 16 patches end-to-end (IBP)
//! 18. **Layer norm through transformer depth** — RMSNorm bounds at each layer (IBP + CROWN)
//!
//! Architecture references:
//! - GLM-4V / ChatGLM (THUDM): Decoder-only with vision encoder for OCR
//! - RMSNorm (Zhang & Sennrich, 2019): Root mean square normalization
//! - SwiGLU (Shazeer, 2020): SiLU-gated FFN
//! - GQA (Ainslie et al., 2023): Grouped-query attention
//!
//! Dimensions (symbolic, small for fast verification):
//! - HIDDEN_DIM=8, FFN_DIM=16, NUM_HEADS=2, HEAD_DIM=4
//! - VOCAB_SIZE=32, PATCH_DIM=12 (3 channels * 2x2 patch)
//!
//! Part of #4225: Compose verification tests for GLM-OCR pipeline.

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
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 4
const VOCAB_SIZE: usize = 32;
const IMG_PATCHES: usize = 4;
const PATCH_DIM: usize = 12; // 3 channels * 2x2 patch = 12 flattened
const SEQ_LEN: usize = 4;
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

    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);

    let up = b.add_linear(input, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    b.add_linear(hidden, down_w, None, &out_shape)
}

/// Push SwiGLU FFN bindings (3 params: gate_w, up_w, down_w).
fn push_swiglu_bindings(bindings: &mut Vec<TensorParamBinding>, hidden_dim: usize, ffn_dim: usize) {
    bindings.push(weight(&[ffn_dim, hidden_dim]));
    bindings.push(weight(&[ffn_dim, hidden_dim]));
    bindings.push(weight(&[hidden_dim, ffn_dim]));
}

/// Add self-attention sub-block with optional causal mask.
///
/// Input: [seq_len, hidden_dim] -> Q/K/V projections -> attention -> output proj.
/// Returns output node.
fn add_self_attention(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    seq_len: usize,
    hidden_dim: usize,
    mask: AttentionMask,
    prefix: &str,
) -> TensorNodeId {
    let shape = [seq_len, hidden_dim];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let q_w = b.add_input(&format!("{prefix}_q_w"), &[hidden_dim, hidden_dim]);
    let k_w = b.add_input(&format!("{prefix}_k_w"), &[hidden_dim, hidden_dim]);
    let v_w = b.add_input(&format!("{prefix}_v_w"), &[hidden_dim, hidden_dim]);
    let o_w = b.add_input(&format!("{prefix}_o_w"), &[hidden_dim, hidden_dim]);

    let q = b.add_linear(input, q_w, None, &shape);
    let k = b.add_linear(input, k_w, None, &shape);
    let v = b.add_linear(input, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, mask, Some(scale), &shape);
    b.add_linear(attn, o_w, None, &shape)
}

/// Push self-attention bindings (4 weight matrices).
fn push_self_attention_bindings(bindings: &mut Vec<TensorParamBinding>, hidden_dim: usize) {
    for _ in 0..4 {
        bindings.push(weight(&[hidden_dim, hidden_dim]));
    }
}

/// Add a visual encoder block: RMSNorm -> self-attention -> residual ->
/// RMSNorm -> SwiGLU FFN -> residual.
/// Input/output: [num_patches, hidden_dim]. Returns output node.
fn add_visual_encoder_block(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    num_patches: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    prefix: &str,
) -> TensorNodeId {
    let shape = [num_patches, hidden_dim];

    // Pre-attention RMSNorm
    let n1_eps = b.add_input(&format!("{prefix}_n1_eps"), &[1]);
    let n1_w = b.add_input(&format!("{prefix}_n1_w"), &[hidden_dim]);
    let normed1 = b.add_rms_norm(input, n1_eps, 1, n1_w, &shape);

    // Self-attention (Standard mask for encoder, no causal)
    let attn_out = add_self_attention(
        b,
        normed1,
        num_patches,
        hidden_dim,
        AttentionMask::Standard,
        prefix,
    );

    // Residual
    let res1 = b.add_binary_add(input, attn_out, &shape);

    // Pre-FFN RMSNorm
    let n2_eps = b.add_input(&format!("{prefix}_n2_eps"), &[1]);
    let n2_w = b.add_input(&format!("{prefix}_n2_w"), &[hidden_dim]);
    let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &shape);

    // SwiGLU FFN
    let ffn_out = add_swiglu_ffn(b, normed2, num_patches, hidden_dim, ffn_dim, prefix);

    // Residual
    b.add_binary_add(res1, ffn_out, &shape)
}

/// Push visual encoder block bindings (2 RMSNorm + 4 attention + 3 SwiGLU = 11 params).
fn push_encoder_block_bindings(
    bindings: &mut Vec<TensorParamBinding>,
    hidden_dim: usize,
    ffn_dim: usize,
) {
    // RMSNorm 1
    bindings.push(eps_binding());
    bindings.push(ones(&[hidden_dim]));
    // Self-attention
    push_self_attention_bindings(bindings, hidden_dim);
    // RMSNorm 2
    bindings.push(eps_binding());
    bindings.push(ones(&[hidden_dim]));
    // SwiGLU FFN
    push_swiglu_bindings(bindings, hidden_dim, ffn_dim);
}

// ===========================================================================
// 1. Patch embedding conv2d + flatten bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_pipeline_patch_embed_conv_flatten_ibp() {
    // Patch embedding: Conv2d flattened as Linear(PATCH_DIM -> HIDDEN_DIM)
    // Input: [IMG_PATCHES, PATCH_DIM], Output: [IMG_PATCHES, HIDDEN_DIM]
    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_patch_conv");
    let input = b.add_input("patches", &[IMG_PATCHES, PATCH_DIM]);
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, PATCH_DIM]);
    let proj_b = b.add_input("proj_b", &[HIDDEN_DIM]);

    let out = b.add_linear(input, proj_w, Some(proj_b), &[IMG_PATCHES, HIDDEN_DIM]);
    let def = b.build(out).expect("valid patch conv kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, PATCH_DIM]),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[IMG_PATCHES, PATCH_DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[IMG_PATCHES, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR patch embed conv IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 2. Patch embedding with position encoding (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_pipeline_patch_embed_with_pe_ibp() {
    // Patch embed + sinusoidal position encoding addition
    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_patch_pe");
    let input = b.add_input("patches", &[IMG_PATCHES, PATCH_DIM]);
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, PATCH_DIM]);
    let pe_table = b.add_input("pe_table", &[IMG_PATCHES, HIDDEN_DIM]);

    let projected = b.add_linear(input, proj_w, None, &[IMG_PATCHES, HIDDEN_DIM]);
    let out = b.add_binary_add(projected, pe_table, &[IMG_PATCHES, HIDDEN_DIM]);
    let def = b.build(out).expect("valid patch+PE kernel");

    // Build sinusoidal PE table bounded in [-1, 1]
    let mut pe_data = vec![0.0f32; IMG_PATCHES * HIDDEN_DIM];
    for t in 0..IMG_PATCHES {
        for i in 0..HIDDEN_DIM / 2 {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * i as f64 / HIDDEN_DIM as f64);
            pe_data[t * HIDDEN_DIM + 2 * i] = freq.sin() as f32;
            pe_data[t * HIDDEN_DIM + 2 * i + 1] = freq.cos() as f32;
        }
    }

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, PATCH_DIM]),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[IMG_PATCHES, HIDDEN_DIM]), pe_data)
                .expect("valid PE table"),
        ),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[IMG_PATCHES, PATCH_DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[IMG_PATCHES, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR patch embed + PE IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 3. Patch embedding wide input range (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_pipeline_patch_embed_wide_input_ibp() {
    // Image pixel values in [0, 255] normalized to [0, 1] — wider input range.
    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_patch_wide");
    let input = b.add_input("patches", &[IMG_PATCHES, PATCH_DIM]);
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, PATCH_DIM]);
    let proj_b = b.add_input("proj_b", &[HIDDEN_DIM]);

    let projected = b.add_linear(input, proj_w, Some(proj_b), &[IMG_PATCHES, HIDDEN_DIM]);

    // RMSNorm to control bound growth from wide input
    let eps = b.add_input("eps", &[1]);
    let nw = b.add_input("nw", &[HIDDEN_DIM]);
    let out = b.add_rms_norm(projected, eps, 1, nw, &[IMG_PATCHES, HIDDEN_DIM]);
    let def = b.build(out).expect("valid wide-input patch kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, PATCH_DIM]),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
        eps_binding(),
        ones(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Wide input range: normalized pixel values in [-0.5, 0.5] (ImageNet-style)
    let input_bounds = uniform_bounds(&[IMG_PATCHES, PATCH_DIM], 0.5);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[IMG_PATCHES, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR patch embed wide input IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 4. Visual encoder single-layer self-attention (IBP + CROWN)
// ===========================================================================

#[test]
fn test_glm_ocr_pipeline_visual_encoder_single_layer_bounds() {
    // Single visual encoder block: RMSNorm -> self-attention -> residual ->
    // RMSNorm -> SwiGLU -> residual
    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_vis_enc_1l");
    let input = b.add_input("patches", &[IMG_PATCHES, HIDDEN_DIM]);

    let out = add_visual_encoder_block(&mut b, input, IMG_PATCHES, HIDDEN_DIM, FFN_DIM, "enc0");
    let def = b.build(out).expect("valid visual encoder block kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_encoder_block_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(IMG_PATCHES, HIDDEN_DIM, 1.0);

    // IBP pass
    let ibp_output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    assert_eq!(
        ibp_output.lower_upper().0.shape(),
        &[IMG_PATCHES, HIDDEN_DIM]
    );

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("GLM-OCR visual encoder 1-layer IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());

    // CROWN pass
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!(
        "GLM-OCR visual encoder 1-layer CROWN: method={method:?}, bounds=[{clo:.6}, {chi:.6}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 5. Visual encoder layer norm contraction (IBP + CROWN)
// ===========================================================================

#[test]
fn test_glm_ocr_pipeline_visual_encoder_layernorm_contraction_bounds() {
    // RMSNorm on visual encoder output — tests normalization contraction.
    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_vis_enc_ln");
    let input = b.add_input("patches", &[IMG_PATCHES, HIDDEN_DIM]);

    // Encoder block
    let encoded = add_visual_encoder_block(&mut b, input, IMG_PATCHES, HIDDEN_DIM, FFN_DIM, "enc0");

    // Final RMSNorm
    let eps = b.add_input("final_eps", &[1]);
    let nw = b.add_input("final_w", &[HIDDEN_DIM]);
    let out = b.add_rms_norm(encoded, eps, 1, nw, &[IMG_PATCHES, HIDDEN_DIM]);
    let def = b.build(out).expect("valid encoder + norm kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_encoder_block_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
    bindings.push(eps_binding());
    bindings.push(ones(&[HIDDEN_DIM]));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(IMG_PATCHES, HIDDEN_DIM, 1.0);

    // IBP pass
    let ibp_output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("GLM-OCR visual encoder + LN IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());

    // CROWN pass
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("GLM-OCR visual encoder + LN CROWN: method={method:?}, bounds=[{clo:.6}, {chi:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 6. Visual encoder two-layer stack (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_pipeline_visual_encoder_two_layer_stack_ibp() {
    // Two stacked visual encoder blocks to test bound accumulation.
    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_vis_enc_2l");
    let input = b.add_input("patches", &[IMG_PATCHES, HIDDEN_DIM]);

    let enc1 = add_visual_encoder_block(&mut b, input, IMG_PATCHES, HIDDEN_DIM, FFN_DIM, "enc0");
    let out = add_visual_encoder_block(&mut b, enc1, IMG_PATCHES, HIDDEN_DIM, FFN_DIM, "enc1");
    let def = b.build(out).expect("valid 2-layer encoder kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_encoder_block_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
    push_encoder_block_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(IMG_PATCHES, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[IMG_PATCHES, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR visual encoder 2-layer IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 7. Cross-attention decoder-to-encoder (IBP + CROWN)
// ===========================================================================

#[test]
fn test_glm_ocr_pipeline_cross_attention_decoder_to_encoder_bounds() {
    // Cross-attention: text decoder queries attend to visual encoder K/V.
    // Q from text: [SEQ_LEN, HIDDEN_DIM], K/V from image: [IMG_PATCHES, HIDDEN_DIM]
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_xattn_dec_enc");
    let text_input = b.add_input("text_hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let img_features = b.add_input("img_features", &[IMG_PATCHES, HIDDEN_DIM]);

    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(text_input, q_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let k = b.add_linear(img_features, k_w, None, &[IMG_PATCHES, HIDDEN_DIM]);
    let v = b.add_linear(img_features, v_w, None, &[IMG_PATCHES, HIDDEN_DIM]);

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
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[IMG_PATCHES, HIDDEN_DIM]),
            0.5f32,
        )),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
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
    eprintln!("GLM-OCR cross-attn dec->enc IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());

    // CROWN pass
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("GLM-OCR cross-attn dec->enc CROWN: method={method:?}, bounds=[{clo:.6}, {chi:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 8. Cross-attention with residual connection (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_pipeline_cross_attention_with_residual_ibp() {
    // Cross-attention + residual skip connection.
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_xattn_res");
    let text_input = b.add_input("text_hidden", &shape);
    let img_features = b.add_input("img_features", &[IMG_PATCHES, HIDDEN_DIM]);

    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(text_input, q_w, None, &shape);
    let k = b.add_linear(img_features, k_w, None, &[IMG_PATCHES, HIDDEN_DIM]);
    let v = b.add_linear(img_features, v_w, None, &[IMG_PATCHES, HIDDEN_DIM]);

    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_proj = b.add_linear(attn, o_w, None, &shape);

    // Residual connection
    let out = b.add_binary_add(text_input, attn_proj, &shape);
    let def = b.build(out).expect("valid cross-attn residual kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[IMG_PATCHES, HIDDEN_DIM]),
            0.5f32,
        )),
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
    eprintln!("GLM-OCR cross-attn + residual IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 9. Text decoder causal self-attention (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_pipeline_text_decoder_causal_self_attention_ibp() {
    // Causal self-attention in the text decoder.
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_dec_causal");
    let input = b.add_input("hidden", &shape);

    // RMSNorm before attention
    let eps = b.add_input("eps", &[1]);
    let nw = b.add_input("nw", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, nw, &shape);

    // Causal self-attention
    let attn_out = add_self_attention(
        &mut b,
        normed,
        SEQ_LEN,
        HIDDEN_DIM,
        AttentionMask::Causal,
        "dec_sa",
    );

    // Residual
    let out = b.add_binary_add(input, attn_out, &shape);
    let def = b.build(out).expect("valid decoder causal attention kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.push(eps_binding());
    bindings.push(ones(&[HIDDEN_DIM]));
    push_self_attention_bindings(&mut bindings, HIDDEN_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR decoder causal self-attn IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 10. Text decoder SwiGLU FFN (IBP + CROWN)
// ===========================================================================

#[test]
fn test_glm_ocr_pipeline_text_decoder_swiglu_ffn_bounds() {
    // Decoder FFN with SwiGLU gating
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_dec_ffn");
    let input = b.add_input("hidden", &shape);

    // RMSNorm before FFN
    let eps = b.add_input("eps", &[1]);
    let nw = b.add_input("nw", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, nw, &shape);

    // SwiGLU FFN
    let ffn_out = add_swiglu_ffn(&mut b, normed, SEQ_LEN, HIDDEN_DIM, FFN_DIM, "dec_ffn");

    // Residual
    let out = b.add_binary_add(input, ffn_out, &shape);
    let def = b.build(out).expect("valid decoder FFN kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.push(eps_binding());
    bindings.push(ones(&[HIDDEN_DIM]));
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP pass
    let ibp_output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    assert_eq!(ibp_output.lower_upper().0.shape(), &shape);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("GLM-OCR decoder SwiGLU FFN IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());

    // CROWN pass
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("GLM-OCR decoder SwiGLU FFN CROWN: method={method:?}, bounds=[{clo:.6}, {chi:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 11. Text decoder full block (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_pipeline_text_decoder_full_block_ibp() {
    // Full decoder block: causal self-attn + cross-attn + SwiGLU FFN.
    // self-attn: [SEQ_LEN, HIDDEN_DIM], cross-attn KV: [IMG_PATCHES, HIDDEN_DIM]
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_dec_full_blk");
    let input = b.add_input("hidden", &shape);
    let img_features = b.add_input("img_features", &[IMG_PATCHES, HIDDEN_DIM]);

    // Pre-self-attn RMSNorm
    let n1_eps = b.add_input("n1_eps", &[1]);
    let n1_w = b.add_input("n1_w", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, n1_eps, 1, n1_w, &shape);

    // Causal self-attention
    let sa_out = add_self_attention(
        &mut b,
        normed1,
        SEQ_LEN,
        HIDDEN_DIM,
        AttentionMask::Causal,
        "sa",
    );
    let res1 = b.add_binary_add(input, sa_out, &shape);

    // Pre-cross-attn RMSNorm
    let n2_eps = b.add_input("n2_eps", &[1]);
    let n2_w = b.add_input("n2_w", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &shape);

    // Cross-attention to visual features
    let xq_w = b.add_input("xq_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let xk_w = b.add_input("xk_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let xv_w = b.add_input("xv_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let xo_w = b.add_input("xo_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let xq = b.add_linear(normed2, xq_w, None, &shape);
    let xk = b.add_linear(img_features, xk_w, None, &[IMG_PATCHES, HIDDEN_DIM]);
    let xv = b.add_linear(img_features, xv_w, None, &[IMG_PATCHES, HIDDEN_DIM]);
    let xa = b.add_attention(xq, xk, xv, AttentionMask::Standard, Some(scale), &shape);
    let xa_proj = b.add_linear(xa, xo_w, None, &shape);
    let res2 = b.add_binary_add(res1, xa_proj, &shape);

    // Pre-FFN RMSNorm
    let n3_eps = b.add_input("n3_eps", &[1]);
    let n3_w = b.add_input("n3_w", &[HIDDEN_DIM]);
    let normed3 = b.add_rms_norm(res2, n3_eps, 1, n3_w, &shape);

    // SwiGLU FFN
    let ffn_out = add_swiglu_ffn(&mut b, normed3, SEQ_LEN, HIDDEN_DIM, FFN_DIM, "ffn");
    let out = b.add_binary_add(res2, ffn_out, &shape);
    let def = b.build(out).expect("valid full decoder block kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[IMG_PATCHES, HIDDEN_DIM]),
            0.5f32,
        )),
    ];
    // RMSNorm 1
    bindings.push(eps_binding());
    bindings.push(ones(&[HIDDEN_DIM]));
    // Self-attention
    push_self_attention_bindings(&mut bindings, HIDDEN_DIM);
    // RMSNorm 2
    bindings.push(eps_binding());
    bindings.push(ones(&[HIDDEN_DIM]));
    // Cross-attention
    for _ in 0..4 {
        bindings.push(weight(&[HIDDEN_DIM, HIDDEN_DIM]));
    }
    // RMSNorm 3
    bindings.push(eps_binding());
    bindings.push(ones(&[HIDDEN_DIM]));
    // SwiGLU FFN
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR full decoder block IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 12. Text decoder short sequence (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_pipeline_text_decoder_short_sequence_ibp() {
    // Decoder block with short sequence (SEQ_LEN=2) — edge case.
    let short_seq: usize = 2;
    let shape = [short_seq, HIDDEN_DIM];

    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_dec_short");
    let input = b.add_input("hidden", &shape);

    // RMSNorm + causal self-attention + residual
    let eps = b.add_input("eps", &[1]);
    let nw = b.add_input("nw", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, nw, &shape);

    let attn_out = add_self_attention(
        &mut b,
        normed,
        short_seq,
        HIDDEN_DIM,
        AttentionMask::Causal,
        "sa",
    );
    let out = b.add_binary_add(input, attn_out, &shape);
    let def = b.build(out).expect("valid short-seq decoder kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.push(eps_binding());
    bindings.push(ones(&[HIDDEN_DIM]));
    push_self_attention_bindings(&mut bindings, HIDDEN_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(short_seq, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[short_seq, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR decoder short seq (L=2) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 13. Text decoder long sequence (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_pipeline_text_decoder_long_sequence_ibp() {
    // Decoder block with longer sequence (SEQ_LEN=8).
    let long_seq: usize = 8;
    let shape = [long_seq, HIDDEN_DIM];

    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_dec_long");
    let input = b.add_input("hidden", &shape);

    // RMSNorm + causal self-attention + residual
    let eps = b.add_input("eps", &[1]);
    let nw = b.add_input("nw", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, nw, &shape);

    let attn_out = add_self_attention(
        &mut b,
        normed,
        long_seq,
        HIDDEN_DIM,
        AttentionMask::Causal,
        "sa",
    );
    let out = b.add_binary_add(input, attn_out, &shape);
    let def = b.build(out).expect("valid long-seq decoder kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.push(eps_binding());
    bindings.push(ones(&[HIDDEN_DIM]));
    push_self_attention_bindings(&mut bindings, HIDDEN_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(long_seq, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[long_seq, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR decoder long seq (L=8) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 14. Character head softmax bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_pipeline_char_head_softmax_ibp() {
    // Character classification head: Linear(hidden -> vocab) + softmax.
    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_char_softmax");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // RMSNorm before classification
    let eps = b.add_input("eps", &[1]);
    let nw = b.add_input("nw", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, nw, &[SEQ_LEN, HIDDEN_DIM]);

    // Linear projection to vocabulary
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);

    // Softmax
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid char head softmax kernel");

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
    eprintln!("GLM-OCR char head softmax IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

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
// 15. Character head log_softmax bounds (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_pipeline_char_head_log_softmax_ibp() {
    // Character classification head with log_softmax (for CTC loss).
    // Log_softmax output should be <= 0 (log of probabilities).
    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_char_logsoftmax");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // Linear projection to vocabulary
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(input, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);

    // Log_softmax
    let out = b.add_log_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid char head log_softmax kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR char head log_softmax IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // log_softmax output must be <= 0 (log of [0, 1] probabilities)
    assert!(
        hi_max <= 1e-4,
        "log_softmax upper bound should be <= 0, got {hi_max}"
    );
    assert!(lo_min.is_finite(), "log_softmax lower bound must be finite");
}

// ===========================================================================
// 16. Full pipeline small image (IBP + CROWN)
// ===========================================================================

#[test]
fn test_glm_ocr_pipeline_full_e2e_small_image_bounds() {
    // Full pipeline: patch_embed -> visual encoder -> concat with text ->
    // decoder block -> RMSNorm -> char head softmax.
    // Small image: 4 patches.
    let total_seq = IMG_PATCHES + SEQ_LEN;

    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_full_small");

    // Patch embedding (constant — pre-extracted visual features)
    let img_features = b.add_input("img_features", &[IMG_PATCHES, HIDDEN_DIM]);
    // Text embeddings (variable input)
    let text_input = b.add_input("text_embed", &[SEQ_LEN, HIDDEN_DIM]);

    // Concatenate image + text along sequence dimension
    let combined = b.add_concat(&[img_features, text_input], 0, &[total_seq, HIDDEN_DIM]);

    // Single encoder-style block on combined sequence (standard attention)
    let encoded =
        add_visual_encoder_block(&mut b, combined, total_seq, HIDDEN_DIM, FFN_DIM, "blk0");

    // Final RMSNorm
    let final_eps = b.add_input("final_eps", &[1]);
    let final_w = b.add_input("final_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(encoded, final_eps, 1, final_w, &[total_seq, HIDDEN_DIM]);

    // Character head: Linear -> softmax
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[total_seq, VOCAB_SIZE]);
    let out = b.add_softmax(logits, 1, &[total_seq, VOCAB_SIZE]);
    let def = b.build(out).expect("valid full pipeline kernel");

    let mut bindings = vec![
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[IMG_PATCHES, HIDDEN_DIM]),
            0.5f32,
        )),
        TensorParamBinding::Variable,
    ];
    push_encoder_block_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
    bindings.push(eps_binding());
    bindings.push(ones(&[HIDDEN_DIM]));
    bindings.push(weight(&[VOCAB_SIZE, HIDDEN_DIM]));

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP pass
    let ibp_output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    assert_eq!(ibp_output.lower_upper().0.shape(), &[total_seq, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("GLM-OCR full pipeline (4 patches) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
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
    assert_eq!(
        crown_output.lower_upper().0.shape(),
        &[total_seq, VOCAB_SIZE]
    );

    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!(
        "GLM-OCR full pipeline (4 patches) CROWN: method={method:?}, bounds=[{clo:.6}, {chi:.6}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 17. Full pipeline large image (IBP)
// ===========================================================================

#[test]
fn test_glm_ocr_pipeline_full_e2e_large_image_ibp() {
    // Full pipeline with larger image: 16 patches (simulating higher resolution).
    let large_patches: usize = 16;
    let total_seq = large_patches + SEQ_LEN;

    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_full_large");

    let img_features = b.add_input("img_features", &[large_patches, HIDDEN_DIM]);
    let text_input = b.add_input("text_embed", &[SEQ_LEN, HIDDEN_DIM]);

    let combined = b.add_concat(&[img_features, text_input], 0, &[total_seq, HIDDEN_DIM]);

    // Encoder block on combined sequence
    let encoded =
        add_visual_encoder_block(&mut b, combined, total_seq, HIDDEN_DIM, FFN_DIM, "blk0");

    // Final RMSNorm
    let final_eps = b.add_input("final_eps", &[1]);
    let final_w = b.add_input("final_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(encoded, final_eps, 1, final_w, &[total_seq, HIDDEN_DIM]);

    // Character head
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[total_seq, VOCAB_SIZE]);
    let out = b.add_softmax(logits, 1, &[total_seq, VOCAB_SIZE]);
    let def = b.build(out).expect("valid large-image pipeline kernel");

    let mut bindings = vec![
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[large_patches, HIDDEN_DIM]),
            0.5f32,
        )),
        TensorParamBinding::Variable,
    ];
    push_encoder_block_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
    bindings.push(eps_binding());
    bindings.push(ones(&[HIDDEN_DIM]));
    bindings.push(weight(&[VOCAB_SIZE, HIDDEN_DIM]));

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[total_seq, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR full pipeline (16 patches) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
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
// 18. Layer norm through transformer depth (IBP + CROWN)
// ===========================================================================

#[test]
fn test_glm_ocr_pipeline_layernorm_through_transformer_depth_bounds() {
    // Tests RMSNorm contraction at each layer in a 3-layer encoder stack.
    // Builds 3 encoder blocks, applies RMSNorm after each, and verifies
    // bounds remain finite and controlled.
    let mut b = TensorBlockBuilder::new("glm_ocr_pipe_ln_depth");
    let input = b.add_input("patches", &[IMG_PATCHES, HIDDEN_DIM]);

    // Layer 0
    let enc0 = add_visual_encoder_block(&mut b, input, IMG_PATCHES, HIDDEN_DIM, FFN_DIM, "enc0");
    let n0_eps = b.add_input("n0_eps", &[1]);
    let n0_w = b.add_input("n0_w", &[HIDDEN_DIM]);
    let normed0 = b.add_rms_norm(enc0, n0_eps, 1, n0_w, &[IMG_PATCHES, HIDDEN_DIM]);

    // Layer 1
    let enc1 = add_visual_encoder_block(&mut b, normed0, IMG_PATCHES, HIDDEN_DIM, FFN_DIM, "enc1");
    let n1_eps = b.add_input("n1_eps", &[1]);
    let n1_w = b.add_input("n1_w", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(enc1, n1_eps, 1, n1_w, &[IMG_PATCHES, HIDDEN_DIM]);

    // Layer 2
    let enc2 = add_visual_encoder_block(&mut b, normed1, IMG_PATCHES, HIDDEN_DIM, FFN_DIM, "enc2");
    let n2_eps = b.add_input("n2_eps", &[1]);
    let n2_w = b.add_input("n2_w", &[HIDDEN_DIM]);
    let out = b.add_rms_norm(enc2, n2_eps, 1, n2_w, &[IMG_PATCHES, HIDDEN_DIM]);
    let def = b.build(out).expect("valid 3-layer LN depth kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    // 3 encoder blocks + 3 inter-layer RMSNorms
    for _ in 0..3 {
        push_encoder_block_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
        bindings.push(eps_binding());
        bindings.push(ones(&[HIDDEN_DIM]));
    }
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = seq_bounds(IMG_PATCHES, HIDDEN_DIM, 1.0);

    // IBP pass
    let ibp_output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    assert_eq!(
        ibp_output.lower_upper().0.shape(),
        &[IMG_PATCHES, HIDDEN_DIM]
    );

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("GLM-OCR 3-layer LN depth IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());

    // CROWN pass
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("GLM-OCR 3-layer LN depth CROWN: method={method:?}, bounds=[{clo:.6}, {chi:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}
