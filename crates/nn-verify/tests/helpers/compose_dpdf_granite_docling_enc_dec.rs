// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for Granite-Docling-258M encoder-decoder full pipeline.
//!
//! Verifies IBP and CROWN bound propagation through the complete Idefics3-based
//! VLM architecture: SigLIP2 vision encoder -> vision projection -> cross-attention
//! decoder -> LM head.
//!
//! ## Tests (15 tests)
//!
//! 1.  **SigLIP2 encoder + vision projection** (IBP)
//! 2.  **Cross-attention: decoder queries, encoder key/value** (IBP + CROWN)
//! 3.  **Cross-attention + SwiGLU FFN block** (IBP + CROWN)
//! 4.  **Decoder block: self-attn -> cross-attn -> SwiGLU** (IBP)
//! 5.  **Decoder block CROWN** (CROWN)
//! 6.  **2-layer decoder with cross-attention** (IBP)
//! 7.  **Full pipeline: encoder -> projection -> decoder -> LM head** (IBP)
//! 8.  **Full pipeline + softmax** (IBP, softmax in [0,1])
//! 9.  **Tight-input cross-attention CROWN** (CROWN)
//! 10. **Widening: 1 vs 2 decoder layers** (IBP)
//! 11. **Monotone tightening through encoder-decoder** (IBP)
//! 12. **Verify-and-record: full encoder-decoder pipeline** (IBP)
//! 13. **Self-attention isolation** (IBP + CROWN)
//! 14. **Token embedding through decoder** (IBP)
//! 15. **Encoder depth widening: 1 vs 2 blocks** (IBP)
//!
//! Architecture references:
//! - Granite-Docling-258M: SigLIP2 vision encoder + Granite LLM decoder
//! - Idefics3 (Laurencon et al., 2024): VLM with cross-attention fusion
//! - SigLIP2 (Zhai et al., 2023): Sigmoid-loss pre-trained ViT encoder
//! - GQA (Ainslie et al., 2023): Grouped-Query Attention in decoder
//! - SwiGLU (Shazeer, 2020): SiLU-gated FFN in LLaMA/Granite family
//!
//! Dimensions (small for fast verification, structurally representative):
//! - VISION_DIM=24, VISION_SEQ=4 (tiny encoder output)
//! - LM_DIM=16, DEC_SEQ=4, FFN_DIM=32, NUM_HEADS=4, VOCAB=8
//!
//! Part of #4228: Granite-Docling encoder-decoder compose tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef, TensorNodeId};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// of Granite-Docling-258M encoder-decoder VLM pipeline.
// ---------------------------------------------------------------------------

/// Vision encoder output sequence length (number of patches).
const VISION_SEQ: usize = 4;
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

/// Build a small SigLIP2-style encoder (2 blocks): [VISION_SEQ, VISION_DIM].
///
/// Pre-norm transformer: LayerNorm -> MHA -> residual -> LayerNorm -> FFN(GELU) -> residual.
fn add_siglip2_encoder_block(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    prefix: &str,
) -> TensorNodeId {
    let shape = [VISION_SEQ, VISION_DIM];
    let ffn_dim = VISION_DIM * 2; // 48
    let ffn_shape = [VISION_SEQ, ffn_dim];
    let num_heads = 4; // VISION_DIM=24, 24/4=6 head_dim

    // Pre-norm 1: LayerNorm
    let ln1_w = b.add_input(&format!("{prefix}_ln1_w"), &[VISION_DIM]);
    let ln1_b = b.add_input(&format!("{prefix}_ln1_b"), &[VISION_DIM]);
    let eps = b.add_input(&format!("{prefix}_ln1_eps"), &[1]);
    let normed = b.add_layer_norm(input, eps, 1, ln1_w, ln1_b, &shape);

    // Multi-head self-attention
    let qw = b.add_input(&format!("{prefix}_q_w"), &[VISION_DIM, VISION_DIM]);
    let kw = b.add_input(&format!("{prefix}_k_w"), &[VISION_DIM, VISION_DIM]);
    let vw = b.add_input(&format!("{prefix}_v_w"), &[VISION_DIM, VISION_DIM]);
    let ow = b.add_input(&format!("{prefix}_o_w"), &[VISION_DIM, VISION_DIM]);
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
    let ln2_w = b.add_input(&format!("{prefix}_ln2_w"), &[VISION_DIM]);
    let ln2_b = b.add_input(&format!("{prefix}_ln2_b"), &[VISION_DIM]);
    let eps2 = b.add_input(&format!("{prefix}_ln2_eps"), &[1]);
    let normed2 = b.add_layer_norm(res1, eps2, 1, ln2_w, ln2_b, &shape);

    // FFN: Linear -> GELU -> Linear
    let ffn1_w = b.add_input(&format!("{prefix}_ffn1_w"), &[ffn_dim, VISION_DIM]);
    let ffn2_w = b.add_input(&format!("{prefix}_ffn2_w"), &[VISION_DIM, ffn_dim]);
    let ffn1 = b.add_linear(normed2, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(act, ffn2_w, None, &shape);

    // Residual 2
    b.add_binary_add(res1, ffn2, &shape)
}

/// Bindings for a single SigLIP2 encoder block.
fn siglip2_block_bindings() -> Vec<TensorParamBinding> {
    let ffn_dim = VISION_DIM * 2;
    vec![
        // ln1: weight, bias, eps
        ones(&[VISION_DIM]),
        bias_zero(&[VISION_DIM]),
        eps_scalar(),
        // MHA: Q, K, V, O weights
        w(&[VISION_DIM, VISION_DIM]),
        w(&[VISION_DIM, VISION_DIM]),
        w(&[VISION_DIM, VISION_DIM]),
        w(&[VISION_DIM, VISION_DIM]),
        // ln2: weight, bias, eps
        ones(&[VISION_DIM]),
        bias_zero(&[VISION_DIM]),
        eps_scalar(),
        // FFN: ffn1_w, ffn2_w
        w(&[ffn_dim, VISION_DIM]),
        w(&[VISION_DIM, ffn_dim]),
    ]
}

/// Add a cross-attention block: queries from decoder, keys/values from encoder.
///
/// LN(dec) -> Q proj, LN(enc) -> K/V proj -> cross-attn -> out_proj -> residual.
fn add_cross_attention_block(
    b: &mut TensorBlockBuilder,
    dec_input: TensorNodeId,
    enc_input: TensorNodeId,
    prefix: &str,
) -> TensorNodeId {
    let dec_shape = [DEC_SEQ, LM_DIM];

    // LayerNorm on decoder input
    let ln_dec_w = b.add_input(&format!("{prefix}_ln_dec_w"), &[LM_DIM]);
    let ln_dec_b = b.add_input(&format!("{prefix}_ln_dec_b"), &[LM_DIM]);
    let eps = b.add_input(&format!("{prefix}_ln_dec_eps"), &[1]);
    let normed_dec = b.add_layer_norm(dec_input, eps, 1, ln_dec_w, ln_dec_b, &dec_shape);

    // Q from decoder, K/V from encoder (projected to LM_DIM)
    let q_w = b.add_input(&format!("{prefix}_q_w"), &[LM_DIM, LM_DIM]);
    let k_w = b.add_input(&format!("{prefix}_k_w"), &[LM_DIM, LM_DIM]);
    let v_w = b.add_input(&format!("{prefix}_v_w"), &[LM_DIM, LM_DIM]);
    let o_w = b.add_input(&format!("{prefix}_o_w"), &[LM_DIM, LM_DIM]);

    let cross_attn = b
        .add_multi_head_cross_attention(
            normed_dec,
            enc_input,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &dec_shape,
        )
        .expect("valid cross-attention");

    // Residual connection
    b.add_binary_add(dec_input, cross_attn, &dec_shape)
}

/// Bindings for a cross-attention block.
fn cross_attn_bindings() -> Vec<TensorParamBinding> {
    vec![
        // LN on decoder: weight, bias, eps
        ones(&[LM_DIM]),
        bias_zero(&[LM_DIM]),
        eps_scalar(),
        // Q, K, V, O weights
        w(&[LM_DIM, LM_DIM]),
        w(&[LM_DIM, LM_DIM]),
        w(&[LM_DIM, LM_DIM]),
        w(&[LM_DIM, LM_DIM]),
    ]
}

/// Add a SwiGLU FFN block with RMSNorm pre-norm + residual.
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
        eps_scalar(),          // rms_eps
        ones(&[LM_DIM]),       // rms_w
        w(&[FFN_DIM, LM_DIM]), // gate_w
        w(&[FFN_DIM, LM_DIM]), // up_w
        w(&[LM_DIM, FFN_DIM]), // down_w
    ]
}

/// Add a self-attention block with RMSNorm pre-norm + residual (decoder self-attn).
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
        eps_scalar(),         // rms_eps
        ones(&[LM_DIM]),      // rms_w
        w(&[LM_DIM, LM_DIM]), // q_w
        w(&[LM_DIM, LM_DIM]), // k_w
        w(&[LM_DIM, LM_DIM]), // v_w
        w(&[LM_DIM, LM_DIM]), // o_w
    ]
}

/// Add a full decoder block: self-attn -> cross-attn -> SwiGLU FFN.
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
// 1. SigLIP2 encoder + vision projection (IBP)
// ===========================================================================

#[test]
fn test_gd_enc_dec_encoder_plus_projection_ibp() {
    let mut b = TensorBlockBuilder::new("gd_enc_dec_encoder_proj");

    // Vision encoder: 2 blocks
    let vis_in = b.add_input("vision_input", &[VISION_SEQ, VISION_DIM]);
    let enc1 = add_siglip2_encoder_block(&mut b, vis_in, "enc0");
    let enc2 = add_siglip2_encoder_block(&mut b, enc1, "enc1");

    // Vision projection: VISION_DIM -> LM_DIM
    let proj_w = b.add_input("proj_w", &[LM_DIM, VISION_DIM]);
    let proj_b = b.add_input("proj_b", &[LM_DIM]);
    let out = b.add_linear(enc2, proj_w, Some(proj_b), &[VISION_SEQ, LM_DIM]);
    let def = b.build(out).expect("valid encoder+proj kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(siglip2_block_bindings());
    bindings.extend(siglip2_block_bindings());
    bindings.push(w(&[LM_DIM, VISION_DIM]));
    bindings.push(bias_zero(&[LM_DIM]));

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[VISION_SEQ, VISION_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[VISION_SEQ, LM_DIM]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD enc+proj IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 2. Cross-attention: decoder queries, encoder key/value (IBP + CROWN)
// ===========================================================================

fn build_cross_attention_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("gd_enc_dec_cross_attn");
    let dec_in = b.add_input("dec_input", &[DEC_SEQ, LM_DIM]);
    let enc_in = b.add_input("enc_features", &[VISION_SEQ, LM_DIM]);
    let out = add_cross_attention_block(&mut b, dec_in, enc_in, "xattn0");
    let def = b.build(out).expect("valid cross-attn kernel");

    // Two variable inputs: dec_input is Variable, enc_features is also Variable
    // For NY, we treat dec_input as the variable and enc_features as constant.
    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VISION_SEQ, LM_DIM]), 0.5f32)),
    ];
    bindings.extend(cross_attn_bindings());

    (def, bindings)
}

#[test]
fn test_gd_enc_dec_cross_attention_ibp() {
    let (def, bindings) = build_cross_attention_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ, LM_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ, LM_DIM]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD cross-attn IBP: [{lo:.6}, {hi:.6}]");
}

#[test]
fn test_gd_enc_dec_cross_attention_crown() {
    let (def, bindings) = build_cross_attention_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ, LM_DIM], 0.5);

    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD cross-attn CROWN ({method:?}): [{lo:.6}, {hi:.6}]");
}

// ===========================================================================
// 3. Cross-attention + SwiGLU FFN block (IBP + CROWN)
// ===========================================================================

fn build_cross_attn_ffn_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("gd_enc_dec_xattn_ffn");
    let dec_in = b.add_input("dec_input", &[DEC_SEQ, LM_DIM]);
    let enc_in = b.add_input("enc_features", &[VISION_SEQ, LM_DIM]);

    let cross_out = add_cross_attention_block(&mut b, dec_in, enc_in, "xattn0");
    let out = add_swiglu_ffn_block(&mut b, cross_out, "ffn0");
    let def = b.build(out).expect("valid xattn+ffn kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VISION_SEQ, LM_DIM]), 0.5f32)),
    ];
    bindings.extend(cross_attn_bindings());
    bindings.extend(swiglu_ffn_bindings());

    (def, bindings)
}

#[test]
fn test_gd_enc_dec_xattn_ffn_ibp() {
    let (def, bindings) = build_cross_attn_ffn_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ, LM_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD xattn+ffn IBP: [{lo:.6}, {hi:.6}]");
}

#[test]
fn test_gd_enc_dec_xattn_ffn_crown() {
    let (def, bindings) = build_cross_attn_ffn_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ, LM_DIM], 0.5);

    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD xattn+ffn CROWN ({method:?}): [{lo:.6}, {hi:.6}]");
}

// ===========================================================================
// 4. Full decoder block: self-attn -> cross-attn -> SwiGLU (IBP)
// ===========================================================================

fn build_full_decoder_block_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("gd_enc_dec_full_dec_blk");
    let dec_in = b.add_input("dec_input", &[DEC_SEQ, LM_DIM]);
    let enc_in = b.add_input("enc_features", &[VISION_SEQ, LM_DIM]);

    let out = add_full_decoder_block(&mut b, dec_in, enc_in, "dec0");
    let def = b.build(out).expect("valid full decoder block kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VISION_SEQ, LM_DIM]), 0.5f32)),
    ];
    bindings.extend(full_decoder_block_bindings());

    (def, bindings)
}

#[test]
fn test_gd_enc_dec_full_decoder_block_ibp() {
    let (def, bindings) = build_full_decoder_block_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ, LM_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ, LM_DIM]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD full decoder block IBP: [{lo:.6}, {hi:.6}]");
}

// ===========================================================================
// 5. Full decoder block CROWN
// ===========================================================================

#[test]
fn test_gd_enc_dec_full_decoder_block_crown() {
    let (def, bindings) = build_full_decoder_block_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ, LM_DIM], 0.5);

    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD full decoder block CROWN ({method:?}): [{lo:.6}, {hi:.6}]");
}

// ===========================================================================
// 6. 2-layer decoder with cross-attention (IBP)
// ===========================================================================

fn build_2layer_decoder_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("gd_enc_dec_2layer_dec");
    let dec_in = b.add_input("dec_input", &[DEC_SEQ, LM_DIM]);
    let enc_in = b.add_input("enc_features", &[VISION_SEQ, LM_DIM]);

    let l1 = add_full_decoder_block(&mut b, dec_in, enc_in, "dec0");
    let l2 = add_full_decoder_block(&mut b, l1, enc_in, "dec1");
    let def = b.build(l2).expect("valid 2-layer decoder kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VISION_SEQ, LM_DIM]), 0.5f32)),
    ];
    bindings.extend(full_decoder_block_bindings());
    bindings.extend(full_decoder_block_bindings());

    (def, bindings)
}

#[test]
fn test_gd_enc_dec_2layer_decoder_ibp() {
    let (def, bindings) = build_2layer_decoder_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ, LM_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD 2-layer decoder IBP: [{lo:.6}, {hi:.6}]");
}

// ===========================================================================
// 7. Full pipeline: encoder -> projection -> decoder -> LM head (IBP)
// ===========================================================================

fn build_full_pipeline_kernel(with_softmax: bool) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let label = if with_softmax {
        "gd_enc_dec_full_pipeline_sm"
    } else {
        "gd_enc_dec_full_pipeline"
    };
    let mut b = TensorBlockBuilder::new(label);

    // Vision encoder input (after patch embedding, for tractability)
    let vis_in = b.add_input("vision_input", &[VISION_SEQ, VISION_DIM]);
    let enc_out = add_siglip2_encoder_block(&mut b, vis_in, "enc0");

    // Vision projection: VISION_DIM -> LM_DIM
    let proj_w = b.add_input("proj_w", &[LM_DIM, VISION_DIM]);
    let proj_b = b.add_input("proj_b", &[LM_DIM]);
    let enc_features = b.add_linear(enc_out, proj_w, Some(proj_b), &[VISION_SEQ, LM_DIM]);

    // Decoder input (text embeddings)
    let dec_in = b.add_input("dec_input", &[DEC_SEQ, LM_DIM]);

    // One decoder block with cross-attention to vision
    let dec_out = add_full_decoder_block(&mut b, dec_in, enc_features, "dec0");

    // Final RMSNorm
    let final_eps = b.add_input("final_rms_eps", &[1]);
    let final_w = b.add_input("final_rms_w", &[LM_DIM]);
    let normed = b.add_rms_norm(dec_out, final_eps, 1, final_w, &[DEC_SEQ, LM_DIM]);

    // LM head: Linear -> vocab logits
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, LM_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[DEC_SEQ, VOCAB_SIZE]);

    let out = if with_softmax {
        b.add_softmax(logits, 1, &[DEC_SEQ, VOCAB_SIZE])
    } else {
        logits
    };
    let def = b.build(out).expect("valid full pipeline kernel");

    // Bindings: vision_input=Variable, enc block, proj, dec_input=Constant, dec block, final norm, lm head
    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(siglip2_block_bindings());
    bindings.push(w(&[LM_DIM, VISION_DIM])); // proj_w
    bindings.push(bias_zero(&[LM_DIM])); // proj_b
                                         // Decoder input is constant (fixed text embedding for verification)
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[DEC_SEQ, LM_DIM]),
        0.1f32,
    )));
    bindings.extend(full_decoder_block_bindings());
    bindings.push(eps_scalar()); // final_rms_eps
    bindings.push(ones(&[LM_DIM])); // final_rms_w
    bindings.push(w(&[VOCAB_SIZE, LM_DIM])); // lm_w

    (def, bindings)
}

#[test]
fn test_gd_enc_dec_full_pipeline_ibp() {
    let (def, bindings) = build_full_pipeline_kernel(false);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[VISION_SEQ, VISION_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ, VOCAB_SIZE]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD full pipeline IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 8. Full pipeline + softmax (IBP, softmax in [0,1])
// ===========================================================================

#[test]
fn test_gd_enc_dec_full_pipeline_softmax_ibp() {
    let (def, bindings) = build_full_pipeline_kernel(true);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[VISION_SEQ, VISION_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD full pipeline + softmax IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo >= -1e-5, "softmax lower bound must be >= 0, got {lo}");
    assert!(
        hi <= 1.0 + 1e-5,
        "softmax upper bound must be <= 1, got {hi}"
    );
}

// ===========================================================================
// 9. Tight-input cross-attention CROWN
// ===========================================================================

#[test]
fn test_gd_enc_dec_tight_cross_attention_crown() {
    let (def, bindings) = build_cross_attention_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    // Narrow +-0.1 bounds for CROWN precision analysis
    let input = uniform_bounds(&[DEC_SEQ, LM_DIM], 0.1);

    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD tight cross-attn CROWN ({method:?}): [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 10. Widening: 1 vs 2 decoder layers (IBP)
// ===========================================================================

#[test]
fn test_gd_enc_dec_widening_1_vs_2_layers() {
    let input = uniform_bounds(&[DEC_SEQ, LM_DIM], 1.0);

    // 1-layer decoder
    let (def1, bind1) = build_full_decoder_block_kernel();
    let g1 = tensor_kernel_to_graph(&def1, &bind1).expect("1-layer graph");
    let out1 = g1.propagate_ibp(&input).expect("1-layer IBP");
    let (lo1, hi1) = bounds_min_max(&out1);
    let width1 = hi1 - lo1;

    // 2-layer decoder
    let (def2, bind2) = build_2layer_decoder_kernel();
    let g2 = tensor_kernel_to_graph(&def2, &bind2).expect("2-layer graph");
    let out2 = g2.propagate_ibp(&input).expect("2-layer IBP");
    let (lo2, hi2) = bounds_min_max(&out2);
    let width2 = hi2 - lo2;

    eprintln!("GD enc-dec widening: 1-layer width={width1:.4}, 2-layer width={width2:.4}");
    eprintln!("  Expansion ratio: {:.2}x", width2 / width1.max(1e-10));

    assert!(
        width1.is_finite() && width2.is_finite(),
        "all widths must be finite"
    );
    // Deeper network should not produce drastically narrower bounds
    assert!(
        width2 >= width1 - 1e-4,
        "2-layer bounds should be >= 1-layer: {width2} vs {width1}"
    );
}

// ===========================================================================
// 11. Monotone tightening through encoder-decoder (IBP)
// ===========================================================================

#[test]
fn test_gd_enc_dec_monotone_tightening() {
    let (def, bindings) = build_full_decoder_block_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // Three input ranges: tight, medium, wide
    let tight = uniform_bounds(&[DEC_SEQ, LM_DIM], 0.5);
    let medium = uniform_bounds(&[DEC_SEQ, LM_DIM], 1.0);
    let wide = uniform_bounds(&[DEC_SEQ, LM_DIM], 2.0);

    let out_tight = graph.propagate_ibp(&tight).expect("IBP tight");
    let out_med = graph.propagate_ibp(&medium).expect("IBP medium");
    let out_wide = graph.propagate_ibp(&wide).expect("IBP wide");
    assert_bounds_valid(&out_tight);
    assert_bounds_valid(&out_med);
    assert_bounds_valid(&out_wide);

    let w_tight = {
        let (lo, hi) = bounds_min_max(&out_tight);
        hi - lo
    };
    let w_med = {
        let (lo, hi) = bounds_min_max(&out_med);
        hi - lo
    };
    let w_wide = {
        let (lo, hi) = bounds_min_max(&out_wide);
        hi - lo
    };

    eprintln!("GD enc-dec monotone: tight={w_tight:.4}, medium={w_med:.4}, wide={w_wide:.4}");

    let eps = 1e-3;
    assert!(
        w_tight <= w_med + eps,
        "tight input should produce tight output: {w_tight} > {w_med} + eps"
    );
    assert!(
        w_med <= w_wide + eps,
        "medium input should produce medium output: {w_med} > {w_wide} + eps"
    );
}

// ===========================================================================
// 12. Verify-and-record: full encoder-decoder pipeline
// ===========================================================================

#[test]
fn test_gd_enc_dec_full_pipeline_verify_and_record() {
    let (def, bindings) = build_full_pipeline_kernel(false);
    let input = uniform_bounds(&[VISION_SEQ, VISION_DIM], 1.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "granite_docling_enc_dec::test_gd_enc_dec_full_pipeline_verify_and_record",
    );
    assert!(result.verification.is_finite);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "GD enc-dec pipeline verify: [{lo:.6}, {hi:.6}], mode={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 13. Self-attention isolation (IBP + CROWN)
// ===========================================================================

/// Tests the decoder self-attention sub-block in isolation to verify that
/// causal-masked self-attention preserves finite bounds and that CROWN
/// can tighten beyond IBP.
#[test]
fn test_gd_enc_dec_self_attention_isolation_ibp() {
    let mut b = TensorBlockBuilder::new("gd_enc_dec_self_attn_iso");
    let input = b.add_input("dec_input", &[DEC_SEQ, LM_DIM]);
    let out = add_self_attention_block(&mut b, input, "sa0");
    let def = b.build(out).expect("valid self-attn isolation kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(self_attn_bindings());
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&[DEC_SEQ, LM_DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ, LM_DIM]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD self-attn isolation IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
}

#[test]
fn test_gd_enc_dec_self_attention_isolation_crown() {
    let mut b = TensorBlockBuilder::new("gd_enc_dec_self_attn_iso_c");
    let input = b.add_input("dec_input", &[DEC_SEQ, LM_DIM]);
    let out = add_self_attention_block(&mut b, input, "sa0");
    let def = b.build(out).expect("valid self-attn isolation kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(self_attn_bindings());
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&[DEC_SEQ, LM_DIM], 0.5);

    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD self-attn isolation CROWN ({method:?}): [{lo:.6}, {hi:.6}]");
}

// ===========================================================================
// 14. Token embedding through decoder (IBP)
// ===========================================================================

/// Simulates text token embedding lookup (modeled as Linear from one-hot-like
/// input) followed by a single decoder block with cross-attention. Verifies
/// end-to-end bounds from vocabulary space through the decoder.
#[test]
fn test_gd_enc_dec_token_embedding_through_decoder_ibp() {
    let mut b = TensorBlockBuilder::new("gd_enc_dec_tok_embed_dec");

    // Token embedding: [DEC_SEQ, VOCAB_SIZE] -> [DEC_SEQ, LM_DIM]
    let tok_in = b.add_input("token_ids", &[DEC_SEQ, VOCAB_SIZE]);
    let embed_w = b.add_input("embed_w", &[LM_DIM, VOCAB_SIZE]);
    let embedded = b.add_linear(tok_in, embed_w, None, &[DEC_SEQ, LM_DIM]);

    // Encoder features (constant, pre-extracted from vision encoder)
    let enc_feat = b.add_input("enc_features", &[VISION_SEQ, LM_DIM]);

    // Decoder block: self-attn -> cross-attn -> SwiGLU FFN
    let decoded = add_full_decoder_block(&mut b, embedded, enc_feat, "dec0");

    // Final RMSNorm
    let final_eps = b.add_input("final_rms_eps", &[1]);
    let final_w = b.add_input("final_rms_w", &[LM_DIM]);
    let out = b.add_rms_norm(decoded, final_eps, 1, final_w, &[DEC_SEQ, LM_DIM]);
    let def = b
        .build(out)
        .expect("valid token-embed-through-decoder kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable, // token_ids
        w(&[LM_DIM, VOCAB_SIZE]),     // embed_w
        TensorParamBinding::ConstantTensor(
            // enc_features (pre-extracted)
            ArrayD::from_elem(IxDyn(&[VISION_SEQ, LM_DIM]), 0.5f32),
        ),
    ];
    bindings.extend(full_decoder_block_bindings());
    bindings.push(eps_scalar()); // final_rms_eps
    bindings.push(ones(&[LM_DIM])); // final_rms_w

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&[DEC_SEQ, VOCAB_SIZE], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ, LM_DIM]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("GD token-embed-through-decoder IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 15. Encoder depth widening: 1 vs 2 blocks (IBP)
// ===========================================================================

/// Tracks bound widening through the vision encoder as depth increases.
/// Compares 1-block vs 2-block SigLIP2 encoder output widths, verifying
/// that deeper encoders produce monotonically non-narrowing bounds.
#[test]
fn test_gd_enc_dec_encoder_depth_widening() {
    let input = uniform_bounds(&[VISION_SEQ, VISION_DIM], 1.0);

    // 1-block encoder + projection
    let (def1, bind1) = {
        let mut b = TensorBlockBuilder::new("gd_enc_depth_1blk");
        let vis_in = b.add_input("vision_input", &[VISION_SEQ, VISION_DIM]);
        let enc = add_siglip2_encoder_block(&mut b, vis_in, "enc0");
        let proj_w = b.add_input("proj_w", &[LM_DIM, VISION_DIM]);
        let out = b.add_linear(enc, proj_w, None, &[VISION_SEQ, LM_DIM]);
        let def = b.build(out).expect("valid 1-block encoder kernel");
        let mut bindings = vec![TensorParamBinding::Variable];
        bindings.extend(siglip2_block_bindings());
        bindings.push(w(&[LM_DIM, VISION_DIM]));
        (def, bindings)
    };

    // 2-block encoder + projection
    let (def2, bind2) = {
        let mut b = TensorBlockBuilder::new("gd_enc_depth_2blk");
        let vis_in = b.add_input("vision_input", &[VISION_SEQ, VISION_DIM]);
        let enc1 = add_siglip2_encoder_block(&mut b, vis_in, "enc0");
        let enc2 = add_siglip2_encoder_block(&mut b, enc1, "enc1");
        let proj_w = b.add_input("proj_w", &[LM_DIM, VISION_DIM]);
        let out = b.add_linear(enc2, proj_w, None, &[VISION_SEQ, LM_DIM]);
        let def = b.build(out).expect("valid 2-block encoder kernel");
        let mut bindings = vec![TensorParamBinding::Variable];
        bindings.extend(siglip2_block_bindings());
        bindings.extend(siglip2_block_bindings());
        bindings.push(w(&[LM_DIM, VISION_DIM]));
        (def, bindings)
    };

    let g1 = tensor_kernel_to_graph(&def1, &bind1).expect("1-block graph");
    let g2 = tensor_kernel_to_graph(&def2, &bind2).expect("2-block graph");

    let out1 = g1.propagate_ibp(&input).expect("1-block IBP");
    let out2 = g2.propagate_ibp(&input).expect("2-block IBP");
    assert_bounds_valid(&out1);
    assert_bounds_valid(&out2);

    let (lo1, hi1) = bounds_min_max(&out1);
    let (lo2, hi2) = bounds_min_max(&out2);
    let width1 = hi1 - lo1;
    let width2 = hi2 - lo2;

    eprintln!("GD encoder depth: 1-block width={width1:.4}, 2-block width={width2:.4}");
    eprintln!("  Expansion ratio: {:.2}x", width2 / width1.max(1e-10));

    assert!(
        width1.is_finite() && width2.is_finite(),
        "all widths must be finite"
    );
    // Deeper encoder should not produce drastically narrower bounds
    assert!(
        width2 >= width1 - 1e-4,
        "2-block bounds should be >= 1-block: {width2} vs {width1}"
    );
}
