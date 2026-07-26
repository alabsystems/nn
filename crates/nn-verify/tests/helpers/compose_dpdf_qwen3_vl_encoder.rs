// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests for Qwen3-VL vision encoder pipeline bounds (#4231).
//!
//! Covers the ViT-style vision encoder pipeline with LayerNorm (instead of
//! RMSNorm), GELU-activated visual token projection, and multi-scale patch
//! merge. These tests complement the existing RMSNorm-based tests in
//! `compose_dpdf_qwen3_vl_vision_encoder.rs` by exercising the LayerNorm
//! code path used in some Qwen3-VL configurations.
//!
//! ## Tests (12 tests)
//!
//! 1.  **Patch embedding** (IBP) -- Conv2d -> flatten -> Linear projection
//! 2.  **Patch embedding** (CROWN) -- CROWN through patch embed pipeline
//! 3.  **ViT block** (IBP) -- LayerNorm -> Q/K/V -> attention -> residual -> FFN
//! 4.  **ViT block** (CROWN) -- CROWN through full ViT block
//! 5.  **Multi-scale patch merge** (IBP) -- Linear projection across 2 scales
//! 6.  **Multi-scale patch merge** (CROWN) -- CROWN through multi-scale merge
//! 7.  **Visual token projection** (IBP) -- LayerNorm -> Linear -> GELU -> Linear
//! 8.  **Visual token projection** (CROWN) -- CROWN through GELU projection
//! 9.  **Full 2-layer vision encoder** (IBP) -- patch_embed -> 2x ViT -> projection
//! 10. **Full 2-layer vision encoder** (CROWN) -- end-to-end CROWN
//! 11. **Attention + FFN composition** (IBP) -- bounds through attn then FFN
//! 12. **Attention + FFN composition** (CROWN) -- CROWN through attn+FFN
//!
//! Architecture references:
//! - Qwen2-VL / Qwen3-VL (Alibaba): ViT backbone, patch embedding, multi-scale
//! - ViT (Dosovitskiy et al., 2020): patch embed + transformer encoder
//! - GELU (Hendrycks & Gimpel, 2016): Gaussian Error Linear Unit
//!
//! Dimensions (small for fast verification, structurally representative):
//! - IMG_SIZE=8, PATCH_SIZE=4, IN_CHANNELS=3, HIDDEN_DIM=16
//! - SEQ_LEN=4, FFN_DIM=32, NUM_HEADS=4, HEAD_DIM=4
//! - LM_DIM=32 (projection target)
//!
//! Part of #4231: Qwen3-VL vision encoder pipeline compose tests.

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

const IMG_SIZE: usize = 8;
const PATCH_SIZE: usize = 4;
const IN_CHANNELS: usize = 3;
const GRID_SIZE: usize = IMG_SIZE / PATCH_SIZE; // 2
const SEQ_LEN: usize = GRID_SIZE * GRID_SIZE; // 4
const HIDDEN_DIM: usize = 16;
const FFN_DIM: usize = 32;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 4
const LM_DIM: usize = 32;
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn w(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG)
}

fn ones_arr(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 1.0f32)
}

fn zeros_arr(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 0.0f32)
}

fn weight(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(w(shape))
}

fn ones_bind(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ones_arr(shape))
}

fn zeros_bind(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(zeros_arr(shape))
}

fn eps_bind() -> TensorParamBinding {
    TensorParamBinding::ConstantScalar(1e-5)
}

/// Image-domain input bounds: pixels in [0, 1].
fn image_bounds() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 1.0f32),
    )
    .expect("valid image bounds [0, 1]")
}

/// Build a patch embedding subgraph: Conv2d -> flatten -> transpose -> Linear.
///
/// Returns (output_node, number of bindings pushed).
fn add_patch_embed(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    bindings: &mut Vec<TensorParamBinding>,
) -> nn_dsl::TensorNodeId {
    let out_h = IMG_SIZE / PATCH_SIZE;
    let out_w = IMG_SIZE / PATCH_SIZE;

    // Conv2d patch projection
    let conv_w = b.add_input(
        "pe_conv_w",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let conv_b = b.add_input("pe_conv_b", &[HIDDEN_DIM]);
    let conv = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, out_h, out_w],
    );

    // Flatten spatial dims: [HIDDEN_DIM, H', W'] -> [HIDDEN_DIM, SEQ_LEN]
    let flat = b.add_reshape(conv, &[HIDDEN_DIM, SEQ_LEN]);

    // Transpose to [SEQ_LEN, HIDDEN_DIM]
    let tokens = b.add_transpose(flat, &[1, 0], &[SEQ_LEN, HIDDEN_DIM]);

    // Linear projection (refines patch features)
    let proj_w = b.add_input("pe_proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj_b = b.add_input("pe_proj_b", &[HIDDEN_DIM]);
    let out = b.add_linear(tokens, proj_w, Some(proj_b), &[SEQ_LEN, HIDDEN_DIM]);

    bindings.push(weight(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]));
    bindings.push(zeros_bind(&[HIDDEN_DIM]));
    bindings.push(weight(&[HIDDEN_DIM, HIDDEN_DIM]));
    bindings.push(zeros_bind(&[HIDDEN_DIM]));

    out
}

/// Build a ViT encoder block: LayerNorm -> Q/K/V -> attention -> residual -> LayerNorm -> FFN -> residual.
///
/// Uses LayerNorm (not RMSNorm) to exercise the LayerNorm verification path.
fn add_vit_block(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    block_idx: usize,
    bindings: &mut Vec<TensorParamBinding>,
) -> nn_dsl::TensorNodeId {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Pre-attention LayerNorm
    let ln1_eps = b.add_input(&format!("vb{block_idx}_ln1_eps"), &[1]);
    let ln1_w = b.add_input(&format!("vb{block_idx}_ln1_w"), &[HIDDEN_DIM]);
    let ln1_b = b.add_input(&format!("vb{block_idx}_ln1_b"), &[HIDDEN_DIM]);
    let normed1 = b.add_layer_norm(input, ln1_eps, 1, ln1_w, ln1_b, &shape);

    // Q/K/V projections
    let q_w = b.add_input(&format!("vb{block_idx}_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input(&format!("vb{block_idx}_k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input(&format!("vb{block_idx}_v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input(&format!("vb{block_idx}_out_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);

    // Residual after attention
    let res1 = b.add_binary_add(input, attn_out, &shape);

    // Pre-FFN LayerNorm
    let ln2_eps = b.add_input(&format!("vb{block_idx}_ln2_eps"), &[1]);
    let ln2_w = b.add_input(&format!("vb{block_idx}_ln2_w"), &[HIDDEN_DIM]);
    let ln2_b = b.add_input(&format!("vb{block_idx}_ln2_b"), &[HIDDEN_DIM]);
    let normed2 = b.add_layer_norm(res1, ln2_eps, 1, ln2_w, ln2_b, &shape);

    // FFN: Linear -> GELU -> Linear
    let ff1_w = b.add_input(&format!("vb{block_idx}_ff1_w"), &[FFN_DIM, HIDDEN_DIM]);
    let ff1_b = b.add_input(&format!("vb{block_idx}_ff1_b"), &[FFN_DIM]);
    let ff2_w = b.add_input(&format!("vb{block_idx}_ff2_w"), &[HIDDEN_DIM, FFN_DIM]);
    let ff2_b = b.add_input(&format!("vb{block_idx}_ff2_b"), &[HIDDEN_DIM]);

    let hidden = b.add_linear(normed2, ff1_w, Some(ff1_b), &ffn_shape);
    let activated = b.add_gelu(hidden, &ffn_shape);
    let ffn_out = b.add_linear(activated, ff2_w, Some(ff2_b), &shape);

    // Residual after FFN
    let out = b.add_binary_add(res1, ffn_out, &shape);

    // Push bindings: 14 weight params per block
    let qkvo = w(&[HIDDEN_DIM, HIDDEN_DIM]);
    bindings.push(eps_bind()); // ln1_eps
    bindings.push(ones_bind(&[HIDDEN_DIM])); // ln1_w
    bindings.push(zeros_bind(&[HIDDEN_DIM])); // ln1_b
    bindings.push(TensorParamBinding::ConstantTensor(qkvo.clone())); // q_w
    bindings.push(TensorParamBinding::ConstantTensor(qkvo.clone())); // k_w
    bindings.push(TensorParamBinding::ConstantTensor(qkvo.clone())); // v_w
    bindings.push(TensorParamBinding::ConstantTensor(qkvo)); // out_w
    bindings.push(eps_bind()); // ln2_eps
    bindings.push(ones_bind(&[HIDDEN_DIM])); // ln2_w
    bindings.push(zeros_bind(&[HIDDEN_DIM])); // ln2_b
    bindings.push(weight(&[FFN_DIM, HIDDEN_DIM])); // ff1_w
    bindings.push(zeros_bind(&[FFN_DIM])); // ff1_b
    bindings.push(weight(&[HIDDEN_DIM, FFN_DIM])); // ff2_w
    bindings.push(zeros_bind(&[HIDDEN_DIM])); // ff2_b

    out
}

// ===========================================================================
// 1. Patch embedding (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_enc_patch_embed_ibp() {
    let mut b = TensorBlockBuilder::new("qwen3_vl_enc_patch_embed");
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let mut bindings = vec![TensorParamBinding::Variable];
    let out = add_patch_embed(&mut b, input, &mut bindings);
    let def = b.build(out).expect("valid patch embed kernel");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds();

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Patch embed (Conv2d+flatten+Linear) IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 2. Patch embedding (CROWN)
// ===========================================================================

#[test]
fn test_qwen3_vl_enc_patch_embed_crown() {
    let mut b = TensorBlockBuilder::new("qwen3_vl_enc_patch_embed_crown");
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let mut bindings = vec![TensorParamBinding::Variable];
    let out = add_patch_embed(&mut b, input, &mut bindings);
    let def = b.build(out).expect("valid patch embed kernel");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    // Tighter input range for CROWN stability
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 0.25f32),
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 0.75f32),
    )
    .expect("valid bounds");

    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("Patch embed CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
    assert_eq!(crown_out.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 3. ViT block (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_enc_vit_block_ibp() {
    let mut b = TensorBlockBuilder::new("qwen3_vl_enc_vit_block");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let mut bindings = vec![TensorParamBinding::Variable];
    let out = add_vit_block(&mut b, input, 0, &mut bindings);
    let def = b.build(out).expect("valid ViT block kernel");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT block (LN+attn+res+LN+FFN+res) IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 4. ViT block (CROWN)
// ===========================================================================

#[test]
fn test_qwen3_vl_enc_vit_block_crown() {
    let mut b = TensorBlockBuilder::new("qwen3_vl_enc_vit_block_crown");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let mut bindings = vec![TensorParamBinding::Variable];
    let out = add_vit_block(&mut b, input, 0, &mut bindings);
    let def = b.build(out).expect("valid ViT block kernel");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("ViT block CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 5. Multi-scale patch merge (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_enc_multi_scale_merge_ibp() {
    // Multi-scale patch merge: two feature streams from different resolutions
    // are each projected to HIDDEN_DIM and added together.
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let mut b = TensorBlockBuilder::new("qwen3_vl_enc_multi_scale_merge");
    let input = b.add_input("scale1_features", &shape);

    // Scale 1: Linear projection
    let proj1_w = b.add_input("proj1_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj1_b = b.add_input("proj1_b", &[HIDDEN_DIM]);
    let scale1 = b.add_linear(input, proj1_w, Some(proj1_b), &shape);

    // Scale 2: constant features from a coarser resolution, also projected
    let scale2_feat = b.add_input("scale2_feat", &shape);
    let proj2_w = b.add_input("proj2_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj2_b = b.add_input("proj2_b", &[HIDDEN_DIM]);
    let scale2 = b.add_linear(scale2_feat, proj2_w, Some(proj2_b), &shape);

    // Additive merge across scales
    let merged = b.add_binary_add(scale1, scale2, &shape);
    let def = b.build(merged).expect("valid multi-scale merge kernel");

    let scale2_const = ArrayD::from_elem(IxDyn(&shape), 0.1f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        zeros_bind(&[HIDDEN_DIM]),
        TensorParamBinding::ConstantTensor(scale2_const),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        zeros_bind(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Multi-scale patch merge IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 6. Multi-scale patch merge (CROWN)
// ===========================================================================

#[test]
fn test_qwen3_vl_enc_multi_scale_merge_crown() {
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let mut b = TensorBlockBuilder::new("qwen3_vl_enc_multi_scale_merge_crown");
    let input = b.add_input("scale1_features", &shape);

    let proj1_w = b.add_input("proj1_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj1_b = b.add_input("proj1_b", &[HIDDEN_DIM]);
    let scale1 = b.add_linear(input, proj1_w, Some(proj1_b), &shape);

    let scale2_feat = b.add_input("scale2_feat", &shape);
    let proj2_w = b.add_input("proj2_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj2_b = b.add_input("proj2_b", &[HIDDEN_DIM]);
    let scale2 = b.add_linear(scale2_feat, proj2_w, Some(proj2_b), &shape);

    let merged = b.add_binary_add(scale1, scale2, &shape);
    let def = b.build(merged).expect("valid multi-scale merge kernel");

    let scale2_const = ArrayD::from_elem(IxDyn(&shape), 0.1f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        zeros_bind(&[HIDDEN_DIM]),
        TensorParamBinding::ConstantTensor(scale2_const),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        zeros_bind(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("Multi-scale patch merge CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 7. Visual token projection (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_enc_visual_token_proj_ibp() {
    // Visual token projection: LayerNorm -> Linear -> GELU -> Linear
    // Maps encoder output to LM embedding space with nonlinear activation.
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let proj_shape = [SEQ_LEN, LM_DIM];

    let mut b = TensorBlockBuilder::new("qwen3_vl_enc_visual_token_proj");
    let input = b.add_input("encoder_out", &shape);

    // LayerNorm before projection
    let ln_eps = b.add_input("proj_ln_eps", &[1]);
    let ln_w = b.add_input("proj_ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("proj_ln_b", &[HIDDEN_DIM]);
    let normed = b.add_layer_norm(input, ln_eps, 1, ln_w, ln_b, &shape);

    // Linear -> GELU -> Linear
    let fc1_w = b.add_input("fc1_w", &[LM_DIM, HIDDEN_DIM]);
    let fc1_b = b.add_input("fc1_b", &[LM_DIM]);
    let hidden = b.add_linear(normed, fc1_w, Some(fc1_b), &proj_shape);
    let activated = b.add_gelu(hidden, &proj_shape);
    let fc2_w = b.add_input("fc2_w", &[LM_DIM, LM_DIM]);
    let fc2_b = b.add_input("fc2_b", &[LM_DIM]);
    let out = b.add_linear(activated, fc2_w, Some(fc2_b), &proj_shape);
    let def = b.build(out).expect("valid visual token projection kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        eps_bind(),
        ones_bind(&[HIDDEN_DIM]),
        zeros_bind(&[HIDDEN_DIM]),
        weight(&[LM_DIM, HIDDEN_DIM]),
        zeros_bind(&[LM_DIM]),
        weight(&[LM_DIM, LM_DIM]),
        zeros_bind(&[LM_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, LM_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Visual token projection (LN+Linear+GELU+Linear) IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 8. Visual token projection (CROWN)
// ===========================================================================

#[test]
fn test_qwen3_vl_enc_visual_token_proj_crown() {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let proj_shape = [SEQ_LEN, LM_DIM];

    let mut b = TensorBlockBuilder::new("qwen3_vl_enc_visual_token_proj_crown");
    let input = b.add_input("encoder_out", &shape);

    let ln_eps = b.add_input("proj_ln_eps", &[1]);
    let ln_w = b.add_input("proj_ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("proj_ln_b", &[HIDDEN_DIM]);
    let normed = b.add_layer_norm(input, ln_eps, 1, ln_w, ln_b, &shape);

    let fc1_w = b.add_input("fc1_w", &[LM_DIM, HIDDEN_DIM]);
    let fc1_b = b.add_input("fc1_b", &[LM_DIM]);
    let hidden = b.add_linear(normed, fc1_w, Some(fc1_b), &proj_shape);
    let activated = b.add_gelu(hidden, &proj_shape);
    let fc2_w = b.add_input("fc2_w", &[LM_DIM, LM_DIM]);
    let fc2_b = b.add_input("fc2_b", &[LM_DIM]);
    let out = b.add_linear(activated, fc2_w, Some(fc2_b), &proj_shape);
    let def = b.build(out).expect("valid visual token projection kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        eps_bind(),
        ones_bind(&[HIDDEN_DIM]),
        zeros_bind(&[HIDDEN_DIM]),
        weight(&[LM_DIM, HIDDEN_DIM]),
        zeros_bind(&[LM_DIM]),
        weight(&[LM_DIM, LM_DIM]),
        zeros_bind(&[LM_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("Visual token projection CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
    assert_eq!(crown_out.lower_upper().0.shape(), &[SEQ_LEN, LM_DIM]);
}

// ===========================================================================
// 9. Full 2-layer vision encoder (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_enc_full_2layer_ibp() {
    // Full pipeline: patch_embed -> 2x ViT blocks -> visual token projection
    let proj_shape = [SEQ_LEN, LM_DIM];

    let mut b = TensorBlockBuilder::new("qwen3_vl_enc_full_2layer");
    let img = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let mut bindings = vec![TensorParamBinding::Variable];

    // Patch embedding
    let tokens = add_patch_embed(&mut b, img, &mut bindings);

    // 2x ViT blocks
    let l1 = add_vit_block(&mut b, tokens, 0, &mut bindings);
    let l2 = add_vit_block(&mut b, l1, 1, &mut bindings);

    // Visual token projection: LayerNorm -> Linear -> GELU -> Linear
    let ln_eps = b.add_input("final_ln_eps", &[1]);
    let ln_w = b.add_input("final_ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("final_ln_b", &[HIDDEN_DIM]);
    let normed = b.add_layer_norm(l2, ln_eps, 1, ln_w, ln_b, &[SEQ_LEN, HIDDEN_DIM]);

    let fc1_w = b.add_input("proj_fc1_w", &[LM_DIM, HIDDEN_DIM]);
    let fc1_b = b.add_input("proj_fc1_b", &[LM_DIM]);
    let hidden = b.add_linear(normed, fc1_w, Some(fc1_b), &proj_shape);
    let activated = b.add_gelu(hidden, &proj_shape);
    let fc2_w = b.add_input("proj_fc2_w", &[LM_DIM, LM_DIM]);
    let fc2_b = b.add_input("proj_fc2_b", &[LM_DIM]);
    let out = b.add_linear(activated, fc2_w, Some(fc2_b), &proj_shape);

    // Projection bindings
    bindings.push(eps_bind());
    bindings.push(ones_bind(&[HIDDEN_DIM]));
    bindings.push(zeros_bind(&[HIDDEN_DIM]));
    bindings.push(weight(&[LM_DIM, HIDDEN_DIM]));
    bindings.push(zeros_bind(&[LM_DIM]));
    bindings.push(weight(&[LM_DIM, LM_DIM]));
    bindings.push(zeros_bind(&[LM_DIM]));

    let def = b.build(out).expect("valid full 2-layer encoder kernel");
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds();

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, LM_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full 2-layer vision encoder IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 10. Full 2-layer vision encoder (CROWN)
// ===========================================================================

#[test]
fn test_qwen3_vl_enc_full_2layer_crown() {
    // Full pipeline with CROWN: uses tighter input bounds for stability
    let proj_shape = [SEQ_LEN, LM_DIM];

    let mut b = TensorBlockBuilder::new("qwen3_vl_enc_full_2layer_crown");
    let img = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let mut bindings = vec![TensorParamBinding::Variable];

    let tokens = add_patch_embed(&mut b, img, &mut bindings);
    let l1 = add_vit_block(&mut b, tokens, 0, &mut bindings);
    let l2 = add_vit_block(&mut b, l1, 1, &mut bindings);

    let ln_eps = b.add_input("final_ln_eps", &[1]);
    let ln_w = b.add_input("final_ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("final_ln_b", &[HIDDEN_DIM]);
    let normed = b.add_layer_norm(l2, ln_eps, 1, ln_w, ln_b, &[SEQ_LEN, HIDDEN_DIM]);

    let fc1_w = b.add_input("proj_fc1_w", &[LM_DIM, HIDDEN_DIM]);
    let fc1_b = b.add_input("proj_fc1_b", &[LM_DIM]);
    let hidden = b.add_linear(normed, fc1_w, Some(fc1_b), &proj_shape);
    let activated = b.add_gelu(hidden, &proj_shape);
    let fc2_w = b.add_input("proj_fc2_w", &[LM_DIM, LM_DIM]);
    let fc2_b = b.add_input("proj_fc2_b", &[LM_DIM]);
    let out = b.add_linear(activated, fc2_w, Some(fc2_b), &proj_shape);

    bindings.push(eps_bind());
    bindings.push(ones_bind(&[HIDDEN_DIM]));
    bindings.push(zeros_bind(&[HIDDEN_DIM]));
    bindings.push(weight(&[LM_DIM, HIDDEN_DIM]));
    bindings.push(zeros_bind(&[LM_DIM]));
    bindings.push(weight(&[LM_DIM, LM_DIM]));
    bindings.push(zeros_bind(&[LM_DIM]));

    let def = b.build(out).expect("valid full 2-layer encoder kernel");
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    // Tighter input for CROWN stability through deep pipeline
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 0.3f32),
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 0.7f32),
    )
    .expect("valid bounds");

    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("Full 2-layer vision encoder CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 11. Attention + FFN composition (IBP)
// ===========================================================================

#[test]
fn test_qwen3_vl_enc_attn_ffn_composition_ibp() {
    // Tests bounds propagation through attention followed by FFN without
    // the LayerNorm normalization layers -- isolates the attention+FFN
    // composition to verify bound widening behavior.
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut b = TensorBlockBuilder::new("qwen3_vl_enc_attn_ffn_comp");
    let input = b.add_input("x", &shape);

    // Attention: Q/K/V -> softmax attention -> out projection
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(input, q_w, None, &shape);
    let k = b.add_linear(input, k_w, None, &shape);
    let v = b.add_linear(input, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);

    // Residual
    let res = b.add_binary_add(input, attn_out, &shape);

    // FFN: Linear -> GELU -> Linear
    let ff1_w = b.add_input("ff1_w", &[FFN_DIM, HIDDEN_DIM]);
    let ff1_b = b.add_input("ff1_b", &[FFN_DIM]);
    let ff2_w = b.add_input("ff2_w", &[HIDDEN_DIM, FFN_DIM]);
    let ff2_b = b.add_input("ff2_b", &[HIDDEN_DIM]);

    let hidden = b.add_linear(res, ff1_w, Some(ff1_b), &ffn_shape);
    let activated = b.add_gelu(hidden, &ffn_shape);
    let ffn_out = b.add_linear(activated, ff2_w, Some(ff2_b), &shape);

    // Residual after FFN
    let out = b.add_binary_add(res, ffn_out, &shape);
    let def = b.build(out).expect("valid attn+FFN composition kernel");

    let qkvo = w(&[HIDDEN_DIM, HIDDEN_DIM]);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(qkvo.clone()), // q_w
        TensorParamBinding::ConstantTensor(qkvo.clone()), // k_w
        TensorParamBinding::ConstantTensor(qkvo.clone()), // v_w
        TensorParamBinding::ConstantTensor(qkvo),         // out_w
        weight(&[FFN_DIM, HIDDEN_DIM]),                   // ff1_w
        zeros_bind(&[FFN_DIM]),                           // ff1_b
        weight(&[HIDDEN_DIM, FFN_DIM]),                   // ff2_w
        zeros_bind(&[HIDDEN_DIM]),                        // ff2_b
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Attention + FFN composition IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 12. Attention + FFN composition (CROWN)
// ===========================================================================

#[test]
fn test_qwen3_vl_enc_attn_ffn_composition_crown() {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut b = TensorBlockBuilder::new("qwen3_vl_enc_attn_ffn_comp_crown");
    let input = b.add_input("x", &shape);

    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(input, q_w, None, &shape);
    let k = b.add_linear(input, k_w, None, &shape);
    let v = b.add_linear(input, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);

    let res = b.add_binary_add(input, attn_out, &shape);

    let ff1_w = b.add_input("ff1_w", &[FFN_DIM, HIDDEN_DIM]);
    let ff1_b = b.add_input("ff1_b", &[FFN_DIM]);
    let ff2_w = b.add_input("ff2_w", &[HIDDEN_DIM, FFN_DIM]);
    let ff2_b = b.add_input("ff2_b", &[HIDDEN_DIM]);

    let hidden = b.add_linear(res, ff1_w, Some(ff1_b), &ffn_shape);
    let activated = b.add_gelu(hidden, &ffn_shape);
    let ffn_out = b.add_linear(activated, ff2_w, Some(ff2_b), &shape);

    let out = b.add_binary_add(res, ffn_out, &shape);
    let def = b.build(out).expect("valid attn+FFN composition kernel");

    let qkvo = w(&[HIDDEN_DIM, HIDDEN_DIM]);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(qkvo.clone()),
        TensorParamBinding::ConstantTensor(qkvo.clone()),
        TensorParamBinding::ConstantTensor(qkvo.clone()),
        TensorParamBinding::ConstantTensor(qkvo),
        weight(&[FFN_DIM, HIDDEN_DIM]),
        zeros_bind(&[FFN_DIM]),
        weight(&[HIDDEN_DIM, FFN_DIM]),
        zeros_bind(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("Attention + FFN composition CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
}
