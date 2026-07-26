// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for vision encoder feature extraction bounds.
//!
//! Verifies IBP and CROWN bound propagation through vision transformer (ViT)
//! encoder architectures used across dpdf document understanding models:
//! SigLIP2 (Granite-Docling), Qwen3-VL WindowViT, and general ViT encoders.
//!
//! ## Tests (18 tests)
//!
//! 1.  **Patch embedding Conv2d output bounds** (IBP)
//! 2.  **Position embedding addition bounds** (IBP)
//! 3.  **Single encoder block: self-attention + MLP** (IBP + CROWN)
//! 4.  **Multi-head attention bounds within encoder** (IBP)
//! 5.  **LayerNorm before attention bounds** (IBP)
//! 6.  **LayerNorm after FFN bounds** (IBP)
//! 7.  **CLS token output bounds after full encoder** (IBP)
//! 8.  **2-layer ViT encoder bounds** (IBP + CROWN)
//! 9.  **4-layer ViT encoder bounds** (IBP)
//! 10. **Window attention bounds (Qwen3-VL WindowViT)** (IBP)
//! 11. **Multi-scale feature extraction (different resolutions)** (IBP)
//! 12. **Patch merging (reducing spatial resolution)** (IBP)
//! 13. **Feature pyramid from encoder layers** (IBP)
//! 14. **SigLIP2 encoder output bounds** (IBP + CROWN)
//! 15. **Global average pooling after encoder** (IBP)
//! 16. **Image resolution scaling effect on bounds** (IBP)
//! 17. **Encoder with different embedding dimensions** (IBP)
//! 18. **Skip connection from early to late encoder layers** (IBP)
//!
//! Architecture references:
//! - ViT (Dosovitskiy et al., 2020): Vision Transformer with patch embedding
//! - SigLIP2 (Zhai et al., 2023): Sigmoid-loss pre-trained ViT encoder
//! - Qwen3-VL (Alibaba): WindowViT with local attention for vision encoding
//! - Swin Transformer (Liu et al., 2021): Patch merging for hierarchical ViT
//! - Granite-Docling: Document understanding with SigLIP2 vision encoder
//!
//! Dimensions (small for fast verification, structurally representative):
//! - IMG_H=16, IMG_W=16, PATCH_SIZE=4, IN_CHANNELS=3
//! - SEQ_LEN=16 (16/4 * 16/4 = 16 patches), HIDDEN_DIM=32, FFN_DIM=64
//! - NUM_HEADS=4, HEAD_DIM=8
//!
//! Part of #4129: Compose tests for vision encoder feature extraction.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef, TensorNodeId};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const IMG_H: usize = 16;
const IMG_W: usize = 16;
const PATCH_SIZE: usize = 4;
const IN_CHANNELS: usize = 3;
/// Number of patches: (IMG_H / PATCH_SIZE) * (IMG_W / PATCH_SIZE).
const SEQ_LEN: usize = (IMG_H / PATCH_SIZE) * (IMG_W / PATCH_SIZE); // 16
const HIDDEN_DIM: usize = 32;
const FFN_DIM: usize = 64;
const NUM_HEADS: usize = 4;
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

/// Ones tensor binding (for LayerNorm / RMSNorm weight).
fn ones(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 1.0f32))
}

/// Scalar epsilon binding.
fn eps_binding() -> TensorParamBinding {
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

/// Sequence-domain input bounds: embeddings in [-range, +range].
fn seq_bounds(seq_len: usize, dim: usize, range: f32) -> BoundedTensor {
    uniform_bounds(&[seq_len, dim], range)
}

/// Build patch embedding Conv2d kernel: [C_in, H, W] -> [EMBED, H/P, W/P].
fn build_patch_embed_kernel(
    name: &str,
    in_ch: usize,
    embed_dim: usize,
    patch_size: usize,
    img_h: usize,
    img_w: usize,
) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let out_h = img_h / patch_size;
    let out_w = img_w / patch_size;

    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("image", &[in_ch, img_h, img_w]);
    let w = b.add_input("proj_w", &[embed_dim, in_ch, patch_size, patch_size]);
    let bias = b.add_input("proj_b", &[embed_dim]);

    let out = b.add_conv2d(
        input,
        w,
        Some(bias),
        patch_size,
        patch_size,
        0,
        0,
        &[embed_dim, out_h, out_w],
    );
    let def = b.build(out).expect("valid patch embed kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[embed_dim, in_ch, patch_size, patch_size]),
        bias_zero(&[embed_dim]),
    ];

    (def, bindings)
}

/// Add a single encoder block (pre-norm transformer) to a builder.
///
/// LN -> MHA -> residual -> LN -> FFN(GELU) -> residual.
/// Input/output: [SEQ, DIM]. Returns output node.
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

/// Build bindings for a single encoder block.
fn encoder_block_bindings(dim: usize, ffn_dim: usize) -> Vec<TensorParamBinding> {
    vec![
        // ln1: weight, bias, eps
        ones(&[dim]),
        bias_zero(&[dim]),
        eps_binding(),
        // MHA: Q, K, V, O weights
        weight(&[dim, dim]),
        weight(&[dim, dim]),
        weight(&[dim, dim]),
        weight(&[dim, dim]),
        // ln2: weight, bias, eps
        ones(&[dim]),
        bias_zero(&[dim]),
        eps_binding(),
        // FFN: ffn1_w, ffn2_w
        weight(&[ffn_dim, dim]),
        weight(&[dim, ffn_dim]),
    ]
}

// ===========================================================================
// 1. Patch embedding Conv2d output bounds (IBP)
// ===========================================================================

#[test]
fn test_vision_encoder_patch_embed_conv2d_ibp() {
    let (def, bindings) = build_patch_embed_kernel(
        "ve_patch_embed",
        IN_CHANNELS,
        HIDDEN_DIM,
        PATCH_SIZE,
        IMG_H,
        IMG_W,
    );
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(IN_CHANNELS, IMG_H, IMG_W);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let out_h = IMG_H / PATCH_SIZE;
    let out_w = IMG_W / PATCH_SIZE;
    assert_eq!(output.lower_upper().0.shape(), &[HIDDEN_DIM, out_h, out_w]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Patch embed IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 2. Position embedding addition bounds (IBP)
// ===========================================================================

#[test]
fn test_vision_encoder_pos_embed_addition_ibp() {
    let mut b = TensorBlockBuilder::new("ve_pos_embed_add");
    let patch_tokens = b.add_input("patch_tokens", &[SEQ_LEN, HIDDEN_DIM]);
    let pos_embed = b.add_input("pos_embed", &[SEQ_LEN, HIDDEN_DIM]);
    let out = b.add_binary_add(patch_tokens, pos_embed, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid pos embed kernel");

    // Patch tokens bounded; pos_embed is a learned constant.
    let pe_data = ArrayD::from_elem(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), 0.01f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe_data),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pos embed add IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Addition of constant shifts bounds by the constant magnitude
    assert!(lo_min < 0.0, "lower bound shifted by positive PE");
    assert!(hi_max > 0.0, "upper bound shifted by positive PE");
}

// ===========================================================================
// 3. Single encoder block: self-attention + MLP (IBP + CROWN)
// ===========================================================================

#[test]
fn test_vision_encoder_single_block_ibp_crown() {
    let mut b = TensorBlockBuilder::new("ve_single_block");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = add_encoder_block(
        &mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk0",
    );
    let def = b.build(out).expect("valid encoder block kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(encoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP baseline
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Single encoder block IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN should also produce valid bounds
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Single encoder block CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 4. Multi-head attention bounds within encoder (IBP)
// ===========================================================================

#[test]
fn test_vision_encoder_mha_bounds_ibp() {
    let mut b = TensorBlockBuilder::new("ve_mha_only");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let qw = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let kw = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let vw = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let ow = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let attn = b
        .add_multi_head_attention(
            input,
            qw,
            kw,
            vw,
            ow,
            NUM_HEADS,
            AttentionMask::Standard,
            &[SEQ_LEN, HIDDEN_DIM],
        )
        .expect("valid MHA");
    let def = b.build(attn).expect("valid MHA kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MHA-only IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 5. LayerNorm before attention bounds (IBP)
// ===========================================================================

#[test]
fn test_vision_encoder_layernorm_pre_attention_ibp() {
    let mut b = TensorBlockBuilder::new("ve_ln_pre_attn");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let out = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid LN kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        ones(&[HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        eps_binding(),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 2.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("LayerNorm pre-attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 6. LayerNorm after FFN bounds (IBP)
// ===========================================================================

#[test]
fn test_vision_encoder_layernorm_post_ffn_ibp() {
    // LN after FFN output: verifies normalization bounds post-activation
    let mut b = TensorBlockBuilder::new("ve_ln_post_ffn");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);

    // FFN: Linear -> GELU -> Linear
    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, HIDDEN_DIM]);
    let ffn2_w = b.add_input("ffn2_w", &[HIDDEN_DIM, FFN_DIM]);
    let ffn1 = b.add_linear(input, ffn1_w, None, &[SEQ_LEN, FFN_DIM]);
    let act = b.add_gelu(ffn1, &[SEQ_LEN, FFN_DIM]);
    let ffn2 = b.add_linear(act, ffn2_w, None, &[SEQ_LEN, HIDDEN_DIM]);

    // Post-FFN LayerNorm
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let out = b.add_layer_norm(ffn2, eps, 1, ln_w, ln_b, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid post-FFN LN kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, FFN_DIM]),
        ones(&[HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        eps_binding(),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("LayerNorm post-FFN IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 7. CLS token output bounds after full encoder (IBP)
// ===========================================================================

#[test]
fn test_vision_encoder_cls_token_output_ibp() {
    // CLS token: prepend a learned [1, DIM] token, run encoder, extract first row.
    // Model: concat([CLS, patch_tokens]) -> encoder block -> narrow(row 0)
    // Simplified: run encoder on SEQ_LEN+1 tokens, narrow output to [1, DIM].
    let cls_seq = SEQ_LEN + 1;

    let mut b = TensorBlockBuilder::new("ve_cls_token");
    let input = b.add_input("x_with_cls", &[cls_seq, HIDDEN_DIM]);
    let out = add_encoder_block(
        &mut b, input, cls_seq, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk0",
    );

    // Narrow to CLS token: first row [1, HIDDEN_DIM]
    let cls_out = b.add_narrow(out, 0, 0, 1, &[1, HIDDEN_DIM]);
    let def = b.build(cls_out).expect("valid CLS token kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(encoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(cls_seq, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[1, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("CLS token after encoder IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 8. 2-layer ViT encoder bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_vision_encoder_2layer_ibp_crown() {
    let mut b = TensorBlockBuilder::new("ve_2layer");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let mid = add_encoder_block(
        &mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk0",
    );
    let out = add_encoder_block(&mut b, mid, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk1");
    let def = b.build(out).expect("valid 2-layer encoder kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(encoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    bindings.extend(encoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("2-layer encoder IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("2-layer encoder CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 9. 4-layer ViT encoder bounds (IBP)
// ===========================================================================

#[test]
fn test_vision_encoder_4layer_ibp() {
    let mut b = TensorBlockBuilder::new("ve_4layer");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let l1 = add_encoder_block(
        &mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk0",
    );
    let l2 = add_encoder_block(&mut b, l1, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk1");
    let l3 = add_encoder_block(&mut b, l2, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk2");
    let out = add_encoder_block(&mut b, l3, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk3");
    let def = b.build(out).expect("valid 4-layer encoder kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    for _ in 0..4 {
        bindings.extend(encoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    }
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("4-layer encoder IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 10. Window attention bounds (Qwen3-VL WindowViT) (IBP)
// ===========================================================================

#[test]
fn test_vision_encoder_window_attention_ibp() {
    // Window attention: partition SEQ_LEN into WINDOW_SIZE windows,
    // run local MHA within each window. Simulated by running MHA
    // on WINDOW_SIZE tokens (equivalent to one local window).
    let window_size: usize = 4;
    let window_dim = HIDDEN_DIM;

    let mut b = TensorBlockBuilder::new("ve_window_attn");
    let input = b.add_input("window_tokens", &[window_size, window_dim]);
    let qw = b.add_input("q_w", &[window_dim, window_dim]);
    let kw = b.add_input("k_w", &[window_dim, window_dim]);
    let vw = b.add_input("v_w", &[window_dim, window_dim]);
    let ow = b.add_input("o_w", &[window_dim, window_dim]);
    let attn = b
        .add_multi_head_attention(
            input,
            qw,
            kw,
            vw,
            ow,
            NUM_HEADS,
            AttentionMask::Standard,
            &[window_size, window_dim],
        )
        .expect("valid window MHA");

    // Residual
    let out = b.add_binary_add(input, attn, &[window_size, window_dim]);
    let def = b.build(out).expect("valid window attention kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[window_dim, window_dim]),
        weight(&[window_dim, window_dim]),
        weight(&[window_dim, window_dim]),
        weight(&[window_dim, window_dim]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(window_size, window_dim, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Window attention (ws={window_size}) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 11. Multi-scale feature extraction (different resolutions) (IBP)
// ===========================================================================

#[test]
fn test_vision_encoder_multiscale_resolution_ibp() {
    // Two patch embeddings at different resolutions (patch_size=4 vs patch_size=8)
    // produce different sequence lengths. Verify both produce valid bounds.
    let (def4, bind4) =
        build_patch_embed_kernel("ve_multiscale_p4", IN_CHANNELS, HIDDEN_DIM, 4, IMG_H, IMG_W);
    let (def8, bind8) =
        build_patch_embed_kernel("ve_multiscale_p8", IN_CHANNELS, HIDDEN_DIM, 8, IMG_H, IMG_W);

    let graph4 = tensor_kernel_to_graph(&def4, &bind4).expect("graph p4");
    let graph8 = tensor_kernel_to_graph(&def8, &bind8).expect("graph p8");
    let input = image_bounds(IN_CHANNELS, IMG_H, IMG_W);

    let out4 = graph4.propagate_ibp(&input).expect("IBP p4");
    let out8 = graph8.propagate_ibp(&input).expect("IBP p8");
    assert_bounds_valid(&out4);
    assert_bounds_valid(&out8);

    // p=4 produces 4x4=16 patches; p=8 produces 2x2=4 patches
    assert_eq!(out4.lower_upper().0.shape(), &[HIDDEN_DIM, 4, 4]);
    assert_eq!(out8.lower_upper().0.shape(), &[HIDDEN_DIM, 2, 2]);

    let (lo4, hi4) = bounds_min_max(&out4);
    let (lo8, hi8) = bounds_min_max(&out8);
    eprintln!("Multi-scale p=4 IBP: [{lo4:.6}, {hi4:.6}], p=8 IBP: [{lo8:.6}, {hi8:.6}]");
}

// ===========================================================================
// 12. Patch merging (reducing spatial resolution) (IBP)
// ===========================================================================

#[test]
fn test_vision_encoder_patch_merging_ibp() {
    // Patch merging: Linear projection to halve token count.
    // Simulated as: [SEQ_LEN, DIM] -> Linear -> [SEQ_LEN/2, 2*DIM] conceptually,
    // but using reshape + linear: [SEQ_LEN, DIM] -> [SEQ_LEN/2, 2*DIM] -> Linear -> [SEQ_LEN/2, DIM]
    let half_seq = SEQ_LEN / 2;
    let double_dim = HIDDEN_DIM * 2;

    let mut b = TensorBlockBuilder::new("ve_patch_merge");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);

    // Reshape: concatenate adjacent patches -> [SEQ/2, 2*DIM]
    let reshaped = b.add_reshape(input, &[half_seq, double_dim]);

    // Linear projection back to DIM
    let proj_w = b.add_input("merge_w", &[HIDDEN_DIM, double_dim]);
    let proj_b = b.add_input("merge_b", &[HIDDEN_DIM]);
    let out = b.add_linear(reshaped, proj_w, Some(proj_b), &[half_seq, HIDDEN_DIM]);
    let def = b.build(out).expect("valid patch merging kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, double_dim]),
        bias_zero(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[half_seq, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Patch merging IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 13. Feature pyramid from encoder layers (IBP)
// ===========================================================================

#[test]
fn test_vision_encoder_feature_pyramid_ibp() {
    // Feature pyramid: project features from 2 encoder layers to the same dim,
    // then add (like FPN lateral connections).
    // Layer 1 features -> Linear -> proj1
    // Layer 2 features -> Linear -> proj2
    // Output = proj1 + proj2
    let mut b = TensorBlockBuilder::new("ve_feature_pyramid");
    let feat1 = b.add_input("encoder_feat1", &[SEQ_LEN, HIDDEN_DIM]);
    let feat2 = b.add_input("encoder_feat2", &[SEQ_LEN, HIDDEN_DIM]);

    // Project both to common dimension
    let proj1_w = b.add_input("proj1_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj2_w = b.add_input("proj2_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let p1 = b.add_linear(feat1, proj1_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let p2 = b.add_linear(feat2, proj2_w, None, &[SEQ_LEN, HIDDEN_DIM]);

    // Fuse via addition
    let out = b.add_binary_add(p1, p2, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid feature pyramid kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Two variable inputs: multi-variable path expects [num_vars, ...tensor_shape]
    let input = uniform_bounds(&[2, SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Feature pyramid IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 14. SigLIP2 encoder output bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_vision_encoder_siglip2_output_ibp_crown() {
    // SigLIP2 encoder: patch embed (as input) -> 2 encoder blocks -> final LN
    let mut b = TensorBlockBuilder::new("ve_siglip2_output");
    let input = b.add_input("patch_embeds", &[SEQ_LEN, HIDDEN_DIM]);
    let l1 = add_encoder_block(
        &mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk0",
    );
    let l2 = add_encoder_block(&mut b, l1, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk1");

    // Final LayerNorm (post-encoder)
    let ln_w = b.add_input("final_ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("final_ln_b", &[HIDDEN_DIM]);
    let eps = b.add_input("final_eps", &[1]);
    let out = b.add_layer_norm(l2, eps, 1, ln_w, ln_b, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid SigLIP2 encoder kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(encoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    bindings.extend(encoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    bindings.push(ones(&[HIDDEN_DIM])); // final LN weight
    bindings.push(bias_zero(&[HIDDEN_DIM])); // final LN bias
    bindings.push(eps_binding()); // final LN eps
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("SigLIP2 encoder IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("SigLIP2 encoder CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 15. Global average pooling after encoder (IBP)
// ===========================================================================

#[test]
fn test_vision_encoder_global_avg_pool_ibp() {
    // Encoder features [SEQ_LEN, DIM] -> mean over seq -> [1, DIM]
    // Simulated as AvgPool2d with kernel = spatial dims on reshaped features.
    // Alternative: reshape [SEQ, DIM] -> [DIM, H_patch, W_patch] -> AvgPool2d
    let h_patch = IMG_H / PATCH_SIZE; // 4
    let w_patch = IMG_W / PATCH_SIZE; // 4

    let mut b = TensorBlockBuilder::new("ve_global_avg_pool");
    let input = b.add_input("encoder_feat", &[HIDDEN_DIM, h_patch, w_patch]);
    // Global average pool: kernel = full spatial, stride = 1, pad = 0
    let out = b.add_avg_pool_2d(input, h_patch, w_patch, 1, 1, 0, 0, &[HIDDEN_DIM, 1, 1]);
    let def = b.build(out).expect("valid global avg pool kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = uniform_bounds(&[HIDDEN_DIM, h_patch, w_patch], 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[HIDDEN_DIM, 1, 1]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Global avg pool IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Average pooling should tighten bounds (average of [-1,+1] is narrower)
    assert!(
        lo_min >= -1.0 - 1e-4,
        "avg pool should not widen below input range"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "avg pool should not widen above input range"
    );
}

// ===========================================================================
// 16. Image resolution scaling effect on bounds (IBP)
// ===========================================================================

#[test]
fn test_vision_encoder_resolution_scaling_ibp() {
    // Compare bounds from 16x16 vs 32x32 images (same patch_size=4).
    // Larger image = more patches = longer sequence = potentially wider bounds.
    let (def16, bind16) =
        build_patch_embed_kernel("ve_res16", IN_CHANNELS, HIDDEN_DIM, PATCH_SIZE, 16, 16);
    let (def32, bind32) =
        build_patch_embed_kernel("ve_res32", IN_CHANNELS, HIDDEN_DIM, PATCH_SIZE, 32, 32);

    let graph16 = tensor_kernel_to_graph(&def16, &bind16).expect("graph 16x16");
    let graph32 = tensor_kernel_to_graph(&def32, &bind32).expect("graph 32x32");

    let input16 = image_bounds(IN_CHANNELS, 16, 16);
    let input32 = image_bounds(IN_CHANNELS, 32, 32);

    let out16 = graph16.propagate_ibp(&input16).expect("IBP 16x16");
    let out32 = graph32.propagate_ibp(&input32).expect("IBP 32x32");
    assert_bounds_valid(&out16);
    assert_bounds_valid(&out32);

    // 16x16 -> 4x4 patches; 32x32 -> 8x8 patches
    assert_eq!(out16.lower_upper().0.shape(), &[HIDDEN_DIM, 4, 4]);
    assert_eq!(out32.lower_upper().0.shape(), &[HIDDEN_DIM, 8, 8]);

    let (lo16, hi16) = bounds_min_max(&out16);
    let (lo32, hi32) = bounds_min_max(&out32);
    eprintln!("Resolution 16x16 IBP: [{lo16:.6}, {hi16:.6}]");
    eprintln!("Resolution 32x32 IBP: [{lo32:.6}, {hi32:.6}]");

    // Same Conv2d weights + same input range => per-element bounds should be similar
    // (larger image doesn't change per-patch bounds, only spatial output size)
    let width16 = hi16 - lo16;
    let width32 = hi32 - lo32;
    let ratio = width32 / width16;
    eprintln!("Resolution bound width ratio (32/16): {ratio:.4}");
    assert!(
        ratio > 0.5 && ratio < 2.0,
        "per-patch bounds should be similar"
    );
}

// ===========================================================================
// 17. Encoder with different embedding dimensions (IBP)
// ===========================================================================

#[test]
fn test_vision_encoder_different_embed_dims_ibp() {
    // Compare encoder block with DIM=32 vs DIM=64.
    // Larger embedding should produce proportionally wider bounds (more weights).
    let dim_small: usize = 32;
    let dim_large: usize = 64;
    let ffn_small = dim_small * 2;
    let ffn_large = dim_large * 2;
    let heads = 4;

    // Small encoder
    let mut b_s = TensorBlockBuilder::new("ve_dim_small");
    let in_s = b_s.add_input("x", &[SEQ_LEN, dim_small]);
    let out_s = add_encoder_block(&mut b_s, in_s, SEQ_LEN, dim_small, ffn_small, heads, "blk0");
    let def_s = b_s.build(out_s).expect("valid small encoder");
    let mut bind_s = vec![TensorParamBinding::Variable];
    bind_s.extend(encoder_block_bindings(dim_small, ffn_small));

    // Large encoder
    let mut b_l = TensorBlockBuilder::new("ve_dim_large");
    let in_l = b_l.add_input("x", &[SEQ_LEN, dim_large]);
    let out_l = add_encoder_block(&mut b_l, in_l, SEQ_LEN, dim_large, ffn_large, heads, "blk0");
    let def_l = b_l.build(out_l).expect("valid large encoder");
    let mut bind_l = vec![TensorParamBinding::Variable];
    bind_l.extend(encoder_block_bindings(dim_large, ffn_large));

    let graph_s = tensor_kernel_to_graph(&def_s, &bind_s).expect("graph small");
    let graph_l = tensor_kernel_to_graph(&def_l, &bind_l).expect("graph large");

    let inp_s = seq_bounds(SEQ_LEN, dim_small, 1.0);
    let inp_l = seq_bounds(SEQ_LEN, dim_large, 1.0);

    let out_s_result = graph_s.propagate_ibp(&inp_s).expect("IBP small");
    let out_l_result = graph_l.propagate_ibp(&inp_l).expect("IBP large");
    assert_bounds_valid(&out_s_result);
    assert_bounds_valid(&out_l_result);

    let (lo_s, hi_s) = bounds_min_max(&out_s_result);
    let (lo_l, hi_l) = bounds_min_max(&out_l_result);
    eprintln!("Encoder dim={dim_small} IBP: [{lo_s:.6}, {hi_s:.6}]");
    eprintln!("Encoder dim={dim_large} IBP: [{lo_l:.6}, {hi_l:.6}]");
    assert!(lo_s.is_finite() && hi_s.is_finite());
    assert!(lo_l.is_finite() && hi_l.is_finite());
}

// ===========================================================================
// 18. Skip connection from early to late encoder layers (IBP)
// ===========================================================================

#[test]
fn test_vision_encoder_skip_connection_early_late_ibp() {
    // Skip connection: layer 0 output is added to layer 2 output (like DenseNet).
    // blk0 -> blk1 -> blk2, output = blk0_out + blk2_out
    let mut b = TensorBlockBuilder::new("ve_skip_early_late");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);

    let l0 = add_encoder_block(
        &mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk0",
    );
    let l1 = add_encoder_block(&mut b, l0, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk1");
    let l2 = add_encoder_block(&mut b, l1, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk2");

    // Skip connection: early (l0) + late (l2)
    let out = b.add_binary_add(l0, l2, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid skip connection kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    for _ in 0..3 {
        bindings.extend(encoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    }
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Skip early-to-late (l0+l2) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}
