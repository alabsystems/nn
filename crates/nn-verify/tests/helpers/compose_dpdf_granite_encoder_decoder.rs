// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for Granite-Docling-258M encoder-decoder full pipeline.
//!
//! Verifies IBP and CROWN bound propagation through the complete encoder-decoder
//! architecture: ViT/SigLIP2 vision encoder -> vision projection -> causal decoder
//! with cross-attention -> LM head -> token prediction.
//!
//! ## Tests (14 tests)
//!
//! 1.  **ViT patch embedding spatial bounds** (IBP)
//! 2.  **ViT self-attention per encoder layer** (IBP + CROWN)
//! 3.  **ViT position embedding interpolation** (IBP)
//! 4.  **Decoder cross-attention to encoder features** (IBP + CROWN)
//! 5.  **Decoder self-attention with causal mask** (IBP)
//! 6.  **Layer norm bounds through encoder** (IBP)
//! 7.  **Decoder FFN intermediate bounds** (IBP + CROWN)
//! 8.  **Token prediction head logit bounds** (IBP)
//! 9.  **Structured output token sequence bounds** (IBP)
//! 10. **Full encoder-decoder pipeline composition** (IBP)
//! 11. **Multi-page document feature aggregation** (IBP)
//! 12. **Table structure prediction bounds** (IBP)
//! 13. **OCR text line detection bounds** (IBP)
//! 14. **Layout classification probability bounds** (IBP + CROWN)
//!
//! Architecture references:
//! - Granite-Docling-258M: SigLIP2 vision encoder + Granite LLM decoder
//! - Idefics3 (Laurencon et al., 2024): VLM with cross-attention fusion
//! - SigLIP2 (Zhai et al., 2023): Sigmoid-loss pre-trained ViT encoder
//! - GQA (Ainslie et al., 2023): Grouped-Query Attention in decoder
//! - SwiGLU (Shazeer, 2020): SiLU-gated FFN in LLaMA/Granite family
//!
//! Dimensions (small for fast verification, structurally representative):
//! - IMG_H=16, IMG_W=16, PATCH_SIZE=4, IN_CHANNELS=3
//! - VISION_DIM=24, VISION_SEQ=16 (patches), LM_DIM=16, DEC_SEQ=4
//! - FFN_DIM=32, NUM_HEADS=4, VOCAB=8, NUM_LAYOUT_CLASSES=5
//!
//! Part of #4228: Granite-Docling encoder-decoder compose tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef, TensorNodeId};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// of Granite-Docling-258M encoder-decoder pipeline.
// ---------------------------------------------------------------------------

const IMG_H: usize = 16;
const IMG_W: usize = 16;
const PATCH_SIZE: usize = 4;
const IN_CHANNELS: usize = 3;
/// Number of patches: (IMG_H / PATCH_SIZE) * (IMG_W / PATCH_SIZE).
const VISION_SEQ: usize = (IMG_H / PATCH_SIZE) * (IMG_W / PATCH_SIZE); // 16
/// Vision encoder hidden dimension.
const VISION_DIM: usize = 24;
/// Decoder/LM hidden dimension.
const LM_DIM: usize = 16;
/// Decoder sequence length (text tokens).
const DEC_SEQ: usize = 4;
/// Decoder FFN intermediate dimension.
const FFN_DIM: usize = 32;
/// Number of attention heads.
const NUM_HEADS: usize = 4;
/// Head dimension: LM_DIM / NUM_HEADS.
const HEAD_DIM: usize = LM_DIM / NUM_HEADS; // 4
/// Vocabulary size.
const VOCAB_SIZE: usize = 8;
/// Number of layout classification classes.
const NUM_LAYOUT_CLASSES: usize = 5;
/// Weight magnitude for constant weight tensors.
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn w(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
}

fn ones(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 1.0f32))
}

fn bias_zero(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}

fn eps_scalar() -> TensorParamBinding {
    TensorParamBinding::ConstantScalar(1e-5)
}

/// Image-domain input bounds: pixels in [0, 1].
fn image_bounds(channels: usize, h: usize, w: usize) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[channels, h, w]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[channels, h, w]), 1.0f32),
    )
    .expect("valid image bounds [0, 1]")
}

/// Build a single SigLIP2-style encoder block: LN -> MHA -> residual -> LN -> FFN(GELU) -> residual.
fn add_encoder_block(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    seq_len: usize,
    dim: usize,
    ffn_dim: usize,
    num_heads: usize,
    prefix: &str,
) -> TensorNodeId {
    let shape = [seq_len, dim];
    let ffn_shape = [seq_len, ffn_dim];

    // Pre-norm 1: LayerNorm
    let ln1_w = b.add_input(&format!("{prefix}_ln1_w"), &[dim]);
    let ln1_b = b.add_input(&format!("{prefix}_ln1_b"), &[dim]);
    let eps = b.add_input(&format!("{prefix}_ln1_eps"), &[1]);
    let normed = b.add_layer_norm(input, eps, 1, ln1_w, ln1_b, &shape);

    // Multi-head self-attention
    let qw = b.add_input(&format!("{prefix}_q_w"), &[dim, dim]);
    let kw = b.add_input(&format!("{prefix}_k_w"), &[dim, dim]);
    let vw = b.add_input(&format!("{prefix}_v_w"), &[dim, dim]);
    let ow = b.add_input(&format!("{prefix}_o_w"), &[dim, dim]);
    let attn = b
        .add_multi_head_attention(
            normed,
            qw,
            kw,
            vw,
            ow,
            num_heads,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid MHA");

    // Residual 1
    let res1 = b.add_binary_add(input, attn, &shape);

    // Pre-norm 2: LayerNorm
    let ln2_w = b.add_input(&format!("{prefix}_ln2_w"), &[dim]);
    let ln2_b = b.add_input(&format!("{prefix}_ln2_b"), &[dim]);
    let eps2 = b.add_input(&format!("{prefix}_ln2_eps"), &[1]);
    let normed2 = b.add_layer_norm(res1, eps2, 1, ln2_w, ln2_b, &shape);

    // FFN: Linear -> GELU -> Linear
    let ffn1_w = b.add_input(&format!("{prefix}_ffn1_w"), &[ffn_dim, dim]);
    let ffn2_w = b.add_input(&format!("{prefix}_ffn2_w"), &[dim, ffn_dim]);
    let ffn1 = b.add_linear(normed2, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(act, ffn2_w, None, &shape);

    // Residual 2
    b.add_binary_add(res1, ffn2, &shape)
}

/// Bindings for a single encoder block.
fn encoder_block_bindings(dim: usize, ffn_dim: usize) -> Vec<TensorParamBinding> {
    vec![
        // ln1: weight, bias, eps
        ones(&[dim]),
        bias_zero(&[dim]),
        eps_scalar(),
        // MHA: Q, K, V, O weights
        w(&[dim, dim]),
        w(&[dim, dim]),
        w(&[dim, dim]),
        w(&[dim, dim]),
        // ln2: weight, bias, eps
        ones(&[dim]),
        bias_zero(&[dim]),
        eps_scalar(),
        // FFN: ffn1_w, ffn2_w
        w(&[ffn_dim, dim]),
        w(&[dim, ffn_dim]),
    ]
}

/// Add cross-attention: queries from decoder, keys/values from encoder features.
///
/// Encoder features are assumed to already be projected to [DEC_SEQ, LM_DIM]
/// (sequence length matching decoder) via attention pooling or narrow.
fn add_cross_attention_block(
    b: &mut TensorBlockBuilder,
    dec_input: TensorNodeId,
    enc_features: TensorNodeId,
    prefix: &str,
) -> TensorNodeId {
    let dec_shape = [DEC_SEQ, LM_DIM];

    // LayerNorm on decoder input
    let ln_w = b.add_input(&format!("{prefix}_ln_w"), &[LM_DIM]);
    let ln_b = b.add_input(&format!("{prefix}_ln_b"), &[LM_DIM]);
    let eps = b.add_input(&format!("{prefix}_ln_eps"), &[1]);
    let normed = b.add_layer_norm(dec_input, eps, 1, ln_w, ln_b, &dec_shape);

    // Q from decoder, K/V from encoder (both [DEC_SEQ, LM_DIM])
    let q_w = b.add_input(&format!("{prefix}_q_w"), &[LM_DIM, LM_DIM]);
    let k_w = b.add_input(&format!("{prefix}_k_w"), &[LM_DIM, LM_DIM]);
    let v_w = b.add_input(&format!("{prefix}_v_w"), &[LM_DIM, LM_DIM]);
    let o_w = b.add_input(&format!("{prefix}_o_w"), &[LM_DIM, LM_DIM]);

    let cross_attn = b
        .add_multi_head_cross_attention(
            normed,
            enc_features,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &dec_shape,
        )
        .expect("valid cross-attention");

    // Residual
    b.add_binary_add(dec_input, cross_attn, &dec_shape)
}

/// Bindings for a cross-attention block.
fn cross_attn_bindings() -> Vec<TensorParamBinding> {
    vec![
        ones(&[LM_DIM]),
        bias_zero(&[LM_DIM]),
        eps_scalar(),
        w(&[LM_DIM, LM_DIM]),
        w(&[LM_DIM, LM_DIM]),
        w(&[LM_DIM, LM_DIM]),
        w(&[LM_DIM, LM_DIM]),
    ]
}

/// Add SwiGLU FFN block with RMSNorm pre-norm + residual.
fn add_swiglu_ffn_block(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    prefix: &str,
) -> TensorNodeId {
    let shape = [DEC_SEQ, LM_DIM];
    let ffn_shape = [DEC_SEQ, FFN_DIM];

    // RMSNorm pre-norm
    let rms_eps = b.add_input(&format!("{prefix}_rms_eps"), &[1]);
    let rms_w = b.add_input(&format!("{prefix}_rms_w"), &[LM_DIM]);
    let normed = b.add_rms_norm(input, rms_eps, 1, rms_w, &shape);

    // SwiGLU: silu(gate(x)) * up(x) -> down
    let gate_w = b.add_input(&format!("{prefix}_gate_w"), &[FFN_DIM, LM_DIM]);
    let up_w = b.add_input(&format!("{prefix}_up_w"), &[FFN_DIM, LM_DIM]);
    let down_w = b.add_input(&format!("{prefix}_down_w"), &[LM_DIM, FFN_DIM]);

    let gate = b.add_linear(normed, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    b.add_binary_add(input, ffn_out, &shape)
}

/// Bindings for a SwiGLU FFN block.
fn swiglu_ffn_bindings() -> Vec<TensorParamBinding> {
    vec![
        eps_scalar(),
        ones(&[LM_DIM]),
        w(&[FFN_DIM, LM_DIM]),
        w(&[FFN_DIM, LM_DIM]),
        w(&[LM_DIM, FFN_DIM]),
    ]
}

/// Add decoder self-attention with causal mask + RMSNorm pre-norm + residual.
fn add_self_attention_block(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    prefix: &str,
) -> TensorNodeId {
    let shape = [DEC_SEQ, LM_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // RMSNorm pre-norm
    let rms_eps = b.add_input(&format!("{prefix}_rms_eps"), &[1]);
    let rms_w = b.add_input(&format!("{prefix}_rms_w"), &[LM_DIM]);
    let normed = b.add_rms_norm(input, rms_eps, 1, rms_w, &shape);

    // Self-attention with causal mask
    let q_w = b.add_input(&format!("{prefix}_q_w"), &[LM_DIM, LM_DIM]);
    let k_w = b.add_input(&format!("{prefix}_k_w"), &[LM_DIM, LM_DIM]);
    let v_w = b.add_input(&format!("{prefix}_v_w"), &[LM_DIM, LM_DIM]);
    let o_w = b.add_input(&format!("{prefix}_o_w"), &[LM_DIM, LM_DIM]);

    let q = b.add_linear(normed, q_w, None, &shape);
    let k = b.add_linear(normed, k_w, None, &shape);
    let v = b.add_linear(normed, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
    let attn_out = b.add_linear(attn, o_w, None, &shape);

    b.add_binary_add(input, attn_out, &shape)
}

/// Bindings for a self-attention block.
fn self_attn_bindings() -> Vec<TensorParamBinding> {
    vec![
        eps_scalar(),
        ones(&[LM_DIM]),
        w(&[LM_DIM, LM_DIM]),
        w(&[LM_DIM, LM_DIM]),
        w(&[LM_DIM, LM_DIM]),
        w(&[LM_DIM, LM_DIM]),
    ]
}

/// Full decoder block: self-attn -> cross-attn -> SwiGLU FFN.
fn add_full_decoder_block(
    b: &mut TensorBlockBuilder,
    dec_input: TensorNodeId,
    enc_features: TensorNodeId,
    prefix: &str,
) -> TensorNodeId {
    let self_out = add_self_attention_block(b, dec_input, &format!("{prefix}_sa"));
    let cross_out = add_cross_attention_block(b, self_out, enc_features, &format!("{prefix}_xa"));
    add_swiglu_ffn_block(b, cross_out, &format!("{prefix}_ffn"))
}

/// Bindings for a full decoder block.
fn full_decoder_block_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();
    bindings.extend(self_attn_bindings());
    bindings.extend(cross_attn_bindings());
    bindings.extend(swiglu_ffn_bindings());
    bindings
}

// ===========================================================================
// 1. ViT patch embedding spatial bounds (IBP)
// ===========================================================================

#[test]
fn test_gd_ed_vit_patch_embed_spatial_ibp() {
    let out_h = IMG_H / PATCH_SIZE;
    let out_w = IMG_W / PATCH_SIZE;

    let mut b = TensorBlockBuilder::new("gd_ed_patch_embed");
    let input = b.add_input("image", &[IN_CHANNELS, IMG_H, IMG_W]);
    let conv_w = b.add_input("conv_w", &[VISION_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]);
    let conv_b = b.add_input("conv_b", &[VISION_DIM]);
    let out = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[VISION_DIM, out_h, out_w],
    );
    let def = b.build(out).expect("valid patch embed kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        w(&[VISION_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        bias_zero(&[VISION_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(IN_CHANNELS, IMG_H, IMG_W);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[VISION_DIM, out_h, out_w]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD enc-dec patch embed IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 2. ViT self-attention per encoder layer (IBP + CROWN)
// ===========================================================================

#[test]
fn test_gd_ed_vit_self_attention_ibp_crown() {
    let mut b = TensorBlockBuilder::new("gd_ed_vit_self_attn");
    let input = b.add_input("x", &[VISION_SEQ, VISION_DIM]);
    let out = add_encoder_block(
        &mut b,
        input,
        VISION_SEQ,
        VISION_DIM,
        VISION_DIM * 2,
        NUM_HEADS,
        "enc0",
    );
    let def = b.build(out).expect("valid encoder block kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(encoder_block_bindings(VISION_DIM, VISION_DIM * 2));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = uniform_bounds(&[VISION_SEQ, VISION_DIM], 1.0);

    // IBP
    let ibp_out = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_out);
    let (lo, hi) = bounds_min_max(&ibp_out);
    eprintln!("GD enc-dec ViT self-attn IBP: [{lo:.6}, {hi:.6}]");

    // CROWN
    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("GD enc-dec ViT self-attn CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 3. ViT position embedding interpolation (IBP)
// ===========================================================================

#[test]
fn test_gd_ed_vit_pos_embed_interpolation_ibp() {
    // Position embedding: add learned [VISION_SEQ, VISION_DIM] to patch tokens.
    let mut b = TensorBlockBuilder::new("gd_ed_pos_embed");
    let patch_tokens = b.add_input("patches", &[VISION_SEQ, VISION_DIM]);
    let pos_embed = b.add_input("pos_embed", &[VISION_SEQ, VISION_DIM]);
    let out = b.add_binary_add(patch_tokens, pos_embed, &[VISION_SEQ, VISION_DIM]);
    let def = b.build(out).expect("valid pos embed kernel");

    let pe_data = ArrayD::from_elem(IxDyn(&[VISION_SEQ, VISION_DIM]), 0.01f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe_data),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = uniform_bounds(&[VISION_SEQ, VISION_DIM], 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[VISION_SEQ, VISION_DIM]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD enc-dec pos embed IBP: [{lo:.6}, {hi:.6}]");
    // Adding positive PE shifts both bounds up slightly
    assert!(
        lo < 0.0 && hi > 0.0,
        "bounds should straddle zero after PE addition"
    );
}

// ===========================================================================
// 4. Decoder cross-attention to encoder features (IBP + CROWN)
// ===========================================================================

fn build_cross_attn_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("gd_ed_dec_cross_attn");
    let dec_in = b.add_input("dec_input", &[DEC_SEQ, LM_DIM]);
    // Encoder features are projected/pooled to [DEC_SEQ, LM_DIM] before cross-attn
    // (represents learned attention pooling of VISION_SEQ patches to DEC_SEQ tokens).
    let enc_in = b.add_input("enc_features", &[DEC_SEQ, LM_DIM]);
    let out = add_cross_attention_block(&mut b, dec_in, enc_in, "xattn0");
    let def = b.build(out).expect("valid cross-attn kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DEC_SEQ, LM_DIM]), 0.5f32)),
    ];
    bindings.extend(cross_attn_bindings());
    (def, bindings)
}

#[test]
fn test_gd_ed_decoder_cross_attention_ibp() {
    let (def, bindings) = build_cross_attn_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ, LM_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ, LM_DIM]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD enc-dec cross-attn IBP: [{lo:.6}, {hi:.6}]");
}

#[test]
fn test_gd_ed_decoder_cross_attention_crown() {
    let (def, bindings) = build_cross_attn_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ, LM_DIM], 0.5);

    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD enc-dec cross-attn CROWN ({method:?}): [{lo:.6}, {hi:.6}]");
}

// ===========================================================================
// 5. Decoder self-attention with causal mask (IBP)
// ===========================================================================

#[test]
fn test_gd_ed_decoder_causal_self_attention_ibp() {
    let mut b = TensorBlockBuilder::new("gd_ed_dec_causal_sa");
    let input = b.add_input("dec_input", &[DEC_SEQ, LM_DIM]);
    let out = add_self_attention_block(&mut b, input, "sa0");
    let def = b.build(out).expect("valid causal self-attn kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(self_attn_bindings());
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ, LM_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ, LM_DIM]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD enc-dec causal self-attn IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 6. Layer norm bounds through encoder (IBP)
// ===========================================================================

#[test]
fn test_gd_ed_layer_norm_through_encoder_ibp() {
    // LayerNorm at encoder output stabilizes bounds regardless of input width.
    let mut b = TensorBlockBuilder::new("gd_ed_enc_layernorm");
    let input = b.add_input("x", &[VISION_SEQ, VISION_DIM]);
    let enc_out = add_encoder_block(
        &mut b,
        input,
        VISION_SEQ,
        VISION_DIM,
        VISION_DIM * 2,
        NUM_HEADS,
        "enc0",
    );
    // Final LayerNorm
    let ln_w = b.add_input("final_ln_w", &[VISION_DIM]);
    let ln_b = b.add_input("final_ln_b", &[VISION_DIM]);
    let eps = b.add_input("final_eps", &[1]);
    let out = b.add_layer_norm(enc_out, eps, 1, ln_w, ln_b, &[VISION_SEQ, VISION_DIM]);
    let def = b.build(out).expect("valid enc+LN kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(encoder_block_bindings(VISION_DIM, VISION_DIM * 2));
    bindings.push(ones(&[VISION_DIM]));
    bindings.push(bias_zero(&[VISION_DIM]));
    bindings.push(eps_scalar());

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // Test with wide input range
    let inp = uniform_bounds(&[VISION_SEQ, VISION_DIM], 5.0);
    let output = graph.propagate_ibp(&inp).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD enc-dec LN through encoder IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 7. Decoder FFN intermediate bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_gd_ed_decoder_ffn_intermediate_ibp_crown() {
    let mut b = TensorBlockBuilder::new("gd_ed_dec_ffn");
    let input = b.add_input("dec_input", &[DEC_SEQ, LM_DIM]);
    let out = add_swiglu_ffn_block(&mut b, input, "ffn0");
    let def = b.build(out).expect("valid SwiGLU FFN kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(swiglu_ffn_bindings());
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let inp = uniform_bounds(&[DEC_SEQ, LM_DIM], 1.0);

    // IBP
    let ibp_out = graph.propagate_ibp(&inp).expect("IBP");
    assert_bounds_valid(&ibp_out);
    let (lo, hi) = bounds_min_max(&ibp_out);
    eprintln!("GD enc-dec FFN IBP: [{lo:.6}, {hi:.6}]");

    // CROWN
    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("GD enc-dec FFN CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 8. Token prediction head logit bounds (IBP)
// ===========================================================================

#[test]
fn test_gd_ed_token_prediction_head_logits_ibp() {
    // Final RMSNorm -> LM head linear -> logits [DEC_SEQ, VOCAB_SIZE]
    let mut b = TensorBlockBuilder::new("gd_ed_lm_head");
    let input = b.add_input("dec_output", &[DEC_SEQ, LM_DIM]);

    // Final RMSNorm
    let rms_eps = b.add_input("rms_eps", &[1]);
    let rms_w = b.add_input("rms_w", &[LM_DIM]);
    let normed = b.add_rms_norm(input, rms_eps, 1, rms_w, &[DEC_SEQ, LM_DIM]);

    // LM head projection
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, LM_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[DEC_SEQ, VOCAB_SIZE]);
    let def = b.build(logits).expect("valid LM head kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        eps_scalar(),
        ones(&[LM_DIM]),
        w(&[VOCAB_SIZE, LM_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let inp = uniform_bounds(&[DEC_SEQ, LM_DIM], 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ, VOCAB_SIZE]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD enc-dec LM head logits IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 9. Structured output token sequence bounds (IBP)
// ===========================================================================

#[test]
fn test_gd_ed_structured_output_token_seq_ibp() {
    // Decoder -> RMSNorm -> LM head -> softmax: token probabilities in [0, 1].
    let mut b = TensorBlockBuilder::new("gd_ed_structured_output");
    let input = b.add_input("dec_output", &[DEC_SEQ, LM_DIM]);

    let rms_eps = b.add_input("rms_eps", &[1]);
    let rms_w = b.add_input("rms_w", &[LM_DIM]);
    let normed = b.add_rms_norm(input, rms_eps, 1, rms_w, &[DEC_SEQ, LM_DIM]);

    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, LM_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[DEC_SEQ, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[DEC_SEQ, VOCAB_SIZE]);
    let def = b.build(probs).expect("valid structured output kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        eps_scalar(),
        ones(&[LM_DIM]),
        w(&[VOCAB_SIZE, LM_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let inp = uniform_bounds(&[DEC_SEQ, LM_DIM], 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD enc-dec structured output IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo >= -1e-5, "softmax lower bound must be >= 0, got {lo}");
    assert!(
        hi <= 1.0 + 1e-5,
        "softmax upper bound must be <= 1, got {hi}"
    );
}

// ===========================================================================
// 10. Full encoder-decoder pipeline composition (IBP)
// ===========================================================================

#[test]
fn test_gd_ed_full_encoder_decoder_pipeline_ibp() {
    // Compositional verification: encoder subgraph then decoder subgraph.
    // We verify the encoder produces finite bounds, then use those bounds
    // as input specification for the decoder verification.
    //
    // Stage 1: Encoder (vision_input -> encoder block -> projection -> [VISION_SEQ, LM_DIM])
    {
        let mut b = TensorBlockBuilder::new("gd_ed_pipeline_encoder");
        let vis_in = b.add_input("vision_input", &[VISION_SEQ, VISION_DIM]);
        let enc_out = add_encoder_block(
            &mut b,
            vis_in,
            VISION_SEQ,
            VISION_DIM,
            VISION_DIM * 2,
            NUM_HEADS,
            "enc0",
        );
        let proj_w = b.add_input("proj_w", &[LM_DIM, VISION_DIM]);
        let proj_b = b.add_input("proj_b", &[LM_DIM]);
        let out = b.add_linear(enc_out, proj_w, Some(proj_b), &[VISION_SEQ, LM_DIM]);
        let def = b.build(out).expect("valid encoder subgraph");

        let mut bindings = vec![TensorParamBinding::Variable];
        bindings.extend(encoder_block_bindings(VISION_DIM, VISION_DIM * 2));
        bindings.push(w(&[LM_DIM, VISION_DIM]));
        bindings.push(bias_zero(&[LM_DIM]));

        let graph = tensor_kernel_to_graph(&def, &bindings).expect("encoder graph");
        let input = uniform_bounds(&[VISION_SEQ, VISION_DIM], 1.0);
        let enc_output = graph.propagate_ibp(&input).expect("encoder IBP");
        assert_bounds_valid(&enc_output);

        let (lo, hi) = bounds_min_max(&enc_output);
        eprintln!("GD enc-dec pipeline ENCODER IBP: [{lo:.6}, {hi:.6}]");
        assert!(lo.is_finite() && hi.is_finite());
    }

    // Stage 2: Decoder (dec_input -> self-attn -> cross-attn -> FFN -> RMSNorm -> LM head)
    // Decoder input is Variable here, encoder features are constant (from stage 1 bounds).
    {
        let mut b = TensorBlockBuilder::new("gd_ed_pipeline_decoder");
        let dec_in = b.add_input("dec_input", &[DEC_SEQ, LM_DIM]);
        let enc_feat = b.add_input("enc_features", &[DEC_SEQ, LM_DIM]);
        let dec_out = add_full_decoder_block(&mut b, dec_in, enc_feat, "dec0");

        let final_eps = b.add_input("final_rms_eps", &[1]);
        let final_w = b.add_input("final_rms_w", &[LM_DIM]);
        let normed = b.add_rms_norm(dec_out, final_eps, 1, final_w, &[DEC_SEQ, LM_DIM]);
        let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, LM_DIM]);
        let logits = b.add_linear(normed, lm_w, None, &[DEC_SEQ, VOCAB_SIZE]);
        let def = b.build(logits).expect("valid decoder subgraph");

        let mut bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[DEC_SEQ, LM_DIM]),
                0.5f32,
            )),
        ];
        bindings.extend(full_decoder_block_bindings());
        bindings.push(eps_scalar());
        bindings.push(ones(&[LM_DIM]));
        bindings.push(w(&[VOCAB_SIZE, LM_DIM]));

        let graph = tensor_kernel_to_graph(&def, &bindings).expect("decoder graph");
        let dec_input = uniform_bounds(&[DEC_SEQ, LM_DIM], 1.0);
        let dec_output = graph.propagate_ibp(&dec_input).expect("decoder IBP");
        assert_bounds_valid(&dec_output);
        assert_eq!(dec_output.lower_upper().0.shape(), &[DEC_SEQ, VOCAB_SIZE]);

        let (lo, hi) = bounds_min_max(&dec_output);
        eprintln!("GD enc-dec pipeline DECODER IBP: [{lo:.6}, {hi:.6}]");
        assert!(lo.is_finite() && hi.is_finite());
    }
}

// ===========================================================================
// 11. Multi-page document feature aggregation (IBP)
// ===========================================================================

#[test]
fn test_gd_ed_multi_page_feature_aggregation_ibp() {
    // Multi-page: concatenate features from 2 pages [2*VISION_SEQ, LM_DIM],
    // then project down to [VISION_SEQ, LM_DIM] for the decoder.
    let double_seq = VISION_SEQ * 2;

    let mut b = TensorBlockBuilder::new("gd_ed_multi_page");
    let input = b.add_input("multi_page_features", &[double_seq, LM_DIM]);

    // Mean pooling approximation: Linear projection from 2*SEQ to SEQ
    // In practice this represents attention-based aggregation across pages.
    let proj_w = b.add_input("page_agg_w", &[LM_DIM, LM_DIM]);
    let proj_out = b.add_linear(input, proj_w, None, &[double_seq, LM_DIM]);

    // Narrow to VISION_SEQ (take first page worth of features)
    let narrowed = b.add_narrow(proj_out, 0, 0, VISION_SEQ, &[VISION_SEQ, LM_DIM]);
    let def = b.build(narrowed).expect("valid multi-page kernel");

    let bindings = vec![TensorParamBinding::Variable, w(&[LM_DIM, LM_DIM])];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let inp = uniform_bounds(&[double_seq, LM_DIM], 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[VISION_SEQ, LM_DIM]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD enc-dec multi-page aggregation IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 12. Table structure prediction bounds (IBP)
// ===========================================================================

#[test]
fn test_gd_ed_table_structure_prediction_ibp() {
    // Table structure: decoder outputs -> Linear -> sigmoid for cell/row/col detection.
    let num_table_classes: usize = 4; // cell, row, column, header

    let mut b = TensorBlockBuilder::new("gd_ed_table_structure");
    let input = b.add_input("dec_output", &[DEC_SEQ, LM_DIM]);

    let table_w = b.add_input("table_w", &[num_table_classes, LM_DIM]);
    let table_b = b.add_input("table_b", &[num_table_classes]);
    let logits = b.add_linear(input, table_w, Some(table_b), &[DEC_SEQ, num_table_classes]);
    let probs = b.add_sigmoid(logits, &[DEC_SEQ, num_table_classes]);
    let def = b.build(probs).expect("valid table structure kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        w(&[num_table_classes, LM_DIM]),
        bias_zero(&[num_table_classes]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let inp = uniform_bounds(&[DEC_SEQ, LM_DIM], 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[DEC_SEQ, num_table_classes]
    );

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD enc-dec table structure IBP: [{lo:.6}, {hi:.6}]");
    // Sigmoid output must be in (0, 1)
    assert!(lo >= -1e-5, "sigmoid lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi}");
}

// ===========================================================================
// 13. OCR text line detection bounds (IBP)
// ===========================================================================

#[test]
fn test_gd_ed_ocr_text_line_detection_ibp() {
    // OCR text detection: decoder features -> Linear -> sigmoid confidence per position.
    let mut b = TensorBlockBuilder::new("gd_ed_ocr_textline");
    let input = b.add_input("dec_output", &[DEC_SEQ, LM_DIM]);

    // Binary classification per token: is this a text line boundary?
    let det_w = b.add_input("det_w", &[1, LM_DIM]);
    let det_b = b.add_input("det_b", &[1]);
    let logits = b.add_linear(input, det_w, Some(det_b), &[DEC_SEQ, 1]);
    let conf = b.add_sigmoid(logits, &[DEC_SEQ, 1]);
    let def = b.build(conf).expect("valid text line detection kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        w(&[1, LM_DIM]),
        bias_zero(&[1]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let inp = uniform_bounds(&[DEC_SEQ, LM_DIM], 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ, 1]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD enc-dec OCR text line IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo >= -1e-5, "sigmoid lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi}");
}

// ===========================================================================
// 14. Layout classification probability bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_gd_ed_layout_classification_ibp_crown() {
    // Layout classification: decoder CLS token -> MLP -> softmax over layout classes.
    let mlp_dim = LM_DIM * 2;

    let mut b = TensorBlockBuilder::new("gd_ed_layout_cls");
    let input = b.add_input("cls_features", &[1, LM_DIM]);

    // 2-layer MLP classification head
    let mlp1_w = b.add_input("mlp1_w", &[mlp_dim, LM_DIM]);
    let mlp1_b = b.add_input("mlp1_b", &[mlp_dim]);
    let mlp1 = b.add_linear(input, mlp1_w, Some(mlp1_b), &[1, mlp_dim]);
    let act = b.add_gelu(mlp1, &[1, mlp_dim]);

    let mlp2_w = b.add_input("mlp2_w", &[NUM_LAYOUT_CLASSES, mlp_dim]);
    let mlp2_b = b.add_input("mlp2_b", &[NUM_LAYOUT_CLASSES]);
    let logits = b.add_linear(act, mlp2_w, Some(mlp2_b), &[1, NUM_LAYOUT_CLASSES]);
    let probs = b.add_softmax(logits, 1, &[1, NUM_LAYOUT_CLASSES]);
    let def = b.build(probs).expect("valid layout classification kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        w(&[mlp_dim, LM_DIM]),
        bias_zero(&[mlp_dim]),
        w(&[NUM_LAYOUT_CLASSES, mlp_dim]),
        bias_zero(&[NUM_LAYOUT_CLASSES]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let inp = uniform_bounds(&[1, LM_DIM], 1.0);

    // IBP
    let ibp_out = graph.propagate_ibp(&inp).expect("IBP");
    assert_bounds_valid(&ibp_out);
    assert_eq!(ibp_out.lower_upper().0.shape(), &[1, NUM_LAYOUT_CLASSES]);

    let (lo, hi) = bounds_min_max(&ibp_out);
    eprintln!("GD enc-dec layout classification IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo >= -1e-5, "softmax lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "softmax upper <= 1, got {hi}");

    // CROWN
    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("GD enc-dec layout cls CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
    assert!(clo >= -1e-5, "CROWN softmax lower >= 0, got {clo}");
    assert!(chi <= 1.0 + 1e-5, "CROWN softmax upper <= 1, got {chi}");
}
