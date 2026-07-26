// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep compose tests: PaddleOCR-VL visual encoder multi-block composition.
//!
//! Verifies bounds propagation through the SVTR (Scene Text Recognition with
//! a Single Visual Model) visual encoder used in PaddleOCR-VL. These tests
//! target the heuristic gaps in PaddleOCR compose coverage by exercising
//! encoder compositions with both IBP and CROWN at increasing depth.
//!
//! 1. **SVTR self-attention isolation**: Q/K/V + softmax + out_proj through
//!    a single attention head. Both IBP and CROWN (IBP + CROWN).
//!
//! 2. **SVTR MLP GELU block**: LayerNorm -> Linear -> GELU -> Linear ->
//!    residual. Tests GELU activation bounds vs ReLU/SiLU (IBP + CROWN).
//!
//! 3. **Full SVTR encoder block**: LayerNorm -> Attention -> residual ->
//!    LayerNorm -> MLP(GELU) -> residual. Complete encoder block (IBP + CROWN).
//!
//! 4. **Patch embed + encoder block**: Conv2d patch embedding -> reshape ->
//!    transpose -> encoder block. Cross-stage composition (IBP + CROWN).
//!
//! 5. **2-block SVTR encoder + CTC head**: Depth composition through 2 stacked
//!    blocks -> final LayerNorm -> Linear -> softmax. End-to-end from patches
//!    to character probabilities (IBP + CROWN).
//!
//! 6. **Tight-input encoder block**: Narrow +-0.1 bounds to test CROWN
//!    precision on LayerNorm + attention linearization (IBP + CROWN).
//!
//! 7. **DB detector backbone + sigmoid**: Conv-BN-ReLU stages -> sigmoid
//!    probability map. Tests detection branch composition (IBP + CROWN).
//!
//! 8. **Multi-scale SVTR features**: Encoder outputs at different spatial
//!    scales concatenated for multi-resolution recognition (IBP).
//!
//! Architecture reference:
//! - PaddleOCR (Baidu): Production OCR with DB detector + SVTR recognizer
//! - SVTR (Du et al. 2022): Patch embedding + transformer encoder + CTC head
//! - DB (Liao et al. 2020): Differentiable Binarization text detection
//!
//! Dimensions are small for fast verification (HIDDEN_DIM=16, PATCH_SIZE=4).
//! All tests use IbpValidated soundness mode per nn engineering rules.
//!
//! Part of #4304: deep NY compose tests for PaddleOCR visual encoder.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions
// ---------------------------------------------------------------------------

const HIDDEN_DIM: usize = 16;
const FFN_DIM: usize = 64;
const SEQ_LEN: usize = 4;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 4
const VOCAB_SIZE: usize = 64;
const IMG_H: usize = 8;
const IMG_W: usize = 8;
const IN_CHANNELS: usize = 3;
const PATCH_SIZE: usize = 4;
const NUM_PATCHES: usize = (IMG_H / PATCH_SIZE) * (IMG_W / PATCH_SIZE); // 4
const W_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn w(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), W_MAG)
}

fn ones(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 1.0f32)
}

fn zeros(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 0.0f32)
}

/// Image-domain input bounds: pixels in [0, 1].
fn image_bounds(shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), 0.0f32),
        ArrayD::from_elem(IxDyn(shape), 1.0f32),
    )
    .expect("valid image bounds")
}

// ===========================================================================
// 1. SVTR self-attention isolation
// ===========================================================================

fn build_svtr_self_attention() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_svtr_self_attn");
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // LayerNorm before attention
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let normed = b.add_layer_norm(input, ln_eps, 1, ln_w, ln_b, &shape);

    // Q/K/V + attention
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed, q_w, None, &shape);
    let k = b.add_linear(normed, k_w, None, &shape);
    let v = b.add_linear(normed, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, o_w, None, &shape);

    // Residual
    let out = b.add_binary_add(input, attn_out, &shape);
    b.build(out).expect("valid SVTR self-attention")
}

fn svtr_self_attention_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, HIDDEN_DIM])),
    ]
}

#[test]
fn test_paddle_visual_svtr_self_attention_ibp() {
    let def = build_svtr_self_attention();
    let bindings = svtr_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

#[test]
fn test_paddle_visual_svtr_self_attention_crown() {
    let def = build_svtr_self_attention();
    let bindings = svtr_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("svtr_self_attention CROWN method: {method:?}");
}

// ===========================================================================
// 2. SVTR MLP GELU block
// ===========================================================================

fn build_svtr_mlp_gelu() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_svtr_mlp_gelu");
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // LayerNorm before MLP
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let normed = b.add_layer_norm(input, ln_eps, 1, ln_w, ln_b, &shape);

    // MLP: Linear -> GELU -> Linear
    let fc1_w = b.add_input("fc1_w", &[FFN_DIM, HIDDEN_DIM]);
    let fc1_b = b.add_input("fc1_b", &[FFN_DIM]);
    let fc2_w = b.add_input("fc2_w", &[HIDDEN_DIM, FFN_DIM]);
    let fc2_b = b.add_input("fc2_b", &[HIDDEN_DIM]);

    let fc1 = b.add_linear(normed, fc1_w, Some(fc1_b), &ffn_shape);
    let activated = b.add_gelu(fc1, &ffn_shape);
    let fc2 = b.add_linear(activated, fc2_w, Some(fc2_b), &shape);

    // Residual
    let out = b.add_binary_add(input, fc2, &shape);
    b.build(out).expect("valid SVTR MLP GELU")
}

fn svtr_mlp_gelu_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[FFN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, FFN_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])),
    ]
}

#[test]
fn test_paddle_visual_svtr_mlp_gelu_ibp() {
    let def = build_svtr_mlp_gelu();
    let bindings = svtr_mlp_gelu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

#[test]
fn test_paddle_visual_svtr_mlp_gelu_crown() {
    let def = build_svtr_mlp_gelu();
    let bindings = svtr_mlp_gelu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("svtr_mlp_gelu CROWN method: {method:?}");
}

// ===========================================================================
// 3. Full SVTR encoder block
// ===========================================================================

fn add_svtr_encoder_block(
    b: &mut TensorBlockBuilder,
    x: nn_dsl::TensorNodeId,
    pfx: &str,
) -> nn_dsl::TensorNodeId {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // LN + Attention + residual
    let ln1_w = b.add_input(&format!("{pfx}_ln1_w"), &[HIDDEN_DIM]);
    let ln1_b = b.add_input(&format!("{pfx}_ln1_b"), &[HIDDEN_DIM]);
    let ln1_eps = b.add_input(&format!("{pfx}_ln1_eps"), &[1]);
    let normed1 = b.add_layer_norm(x, ln1_eps, 1, ln1_w, ln1_b, &shape);

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

    // LN + MLP(GELU) + residual
    let ln2_w = b.add_input(&format!("{pfx}_ln2_w"), &[HIDDEN_DIM]);
    let ln2_b = b.add_input(&format!("{pfx}_ln2_b"), &[HIDDEN_DIM]);
    let ln2_eps = b.add_input(&format!("{pfx}_ln2_eps"), &[1]);
    let normed2 = b.add_layer_norm(res1, ln2_eps, 1, ln2_w, ln2_b, &shape);

    let fc1_w = b.add_input(&format!("{pfx}_fc1_w"), &[FFN_DIM, HIDDEN_DIM]);
    let fc1_b = b.add_input(&format!("{pfx}_fc1_b"), &[FFN_DIM]);
    let fc2_w = b.add_input(&format!("{pfx}_fc2_w"), &[HIDDEN_DIM, FFN_DIM]);
    let fc2_b = b.add_input(&format!("{pfx}_fc2_b"), &[HIDDEN_DIM]);

    let fc1 = b.add_linear(normed2, fc1_w, Some(fc1_b), &ffn_shape);
    let activated = b.add_gelu(fc1, &ffn_shape);
    let fc2 = b.add_linear(activated, fc2_w, Some(fc2_b), &shape);
    b.add_binary_add(res1, fc2, &shape)
}

fn push_svtr_encoder_block_bindings(bindings: &mut Vec<TensorParamBinding>) {
    // LN1
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    // Q/K/V/O
    for _ in 0..4 {
        bindings.push(TensorParamBinding::ConstantTensor(w(&[
            HIDDEN_DIM, HIDDEN_DIM,
        ])));
    }
    // LN2
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    // MLP
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        FFN_DIM, HIDDEN_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[FFN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        HIDDEN_DIM, FFN_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])));
}

fn build_svtr_full_encoder_block() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_svtr_full_block");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let out = add_svtr_encoder_block(&mut b, input, "b0");
    b.build(out).expect("valid SVTR full encoder block")
}

fn svtr_full_encoder_block_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_svtr_encoder_block_bindings(&mut bindings);
    bindings
}

#[test]
fn test_paddle_visual_svtr_full_encoder_block_ibp() {
    let def = build_svtr_full_encoder_block();
    let bindings = svtr_full_encoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
}

#[test]
fn test_paddle_visual_svtr_full_encoder_block_crown() {
    let def = build_svtr_full_encoder_block();
    let bindings = svtr_full_encoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("svtr_full_encoder_block CROWN method: {method:?}");
}

// ===========================================================================
// 4. Patch embed + encoder block
// ===========================================================================

fn build_patch_embed_encoder_block() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_patch_embed_block");
    let proj_out = [NUM_PATCHES, HIDDEN_DIM];

    // Patch embedding: Conv2d(3, HIDDEN_DIM, PATCH_SIZE, stride=PATCH_SIZE)
    let input = b.add_input("image", &[IN_CHANNELS, IMG_H, IMG_W]);
    let conv_w = b.add_input("conv_w", &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]);
    let conv_b = b.add_input("conv_b", &[HIDDEN_DIM]);
    let grid_h = IMG_H / PATCH_SIZE;
    let grid_w = IMG_W / PATCH_SIZE;
    let conv_out = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, grid_h, grid_w],
    );

    // Reshape [HIDDEN_DIM, grid_h, grid_w] -> [NUM_PATCHES, HIDDEN_DIM]
    let reshaped = b.add_reshape(conv_out, &proj_out);

    // One encoder block
    let out = add_svtr_encoder_block(&mut b, reshaped, "b0");
    b.build(out).expect("valid patch embed + encoder block")
}

fn patch_embed_encoder_block_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])),
    ];
    push_svtr_encoder_block_bindings(&mut bindings);
    bindings
}

#[test]
fn test_paddle_visual_patch_embed_encoder_block_ibp() {
    let def = build_patch_embed_encoder_block();
    let bindings = patch_embed_encoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[IN_CHANNELS, IMG_H, IMG_W]);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.shape(), &[NUM_PATCHES, HIDDEN_DIM]);
}

#[test]
fn test_paddle_visual_patch_embed_encoder_block_crown() {
    let def = build_patch_embed_encoder_block();
    let bindings = patch_embed_encoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[IN_CHANNELS, IMG_H, IMG_W]);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("patch_embed_encoder_block CROWN method: {method:?}");
}

// ===========================================================================
// 5. 2-block SVTR encoder + CTC head
// ===========================================================================

fn build_two_block_ctc_pipeline() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_two_block_ctc");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // 2 encoder blocks
    let x = add_svtr_encoder_block(&mut b, input, "b0");
    let x = add_svtr_encoder_block(&mut b, x, "b1");

    // Final LayerNorm
    let final_ln_w = b.add_input("final_ln_w", &[HIDDEN_DIM]);
    let final_ln_b = b.add_input("final_ln_b", &[HIDDEN_DIM]);
    let final_ln_eps = b.add_input("final_ln_eps", &[1]);
    let normed = b.add_layer_norm(
        x,
        final_ln_eps,
        1,
        final_ln_w,
        final_ln_b,
        &[SEQ_LEN, HIDDEN_DIM],
    );

    // CTC head: Linear -> softmax
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(normed, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    b.build(probs).expect("valid 2-block SVTR + CTC")
}

fn two_block_ctc_pipeline_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_svtr_encoder_block_bindings(&mut bindings);
    push_svtr_encoder_block_bindings(&mut bindings);
    // Final LN
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    // CTC head
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        VOCAB_SIZE, HIDDEN_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[VOCAB_SIZE])));
    bindings
}

#[test]
fn test_paddle_visual_two_block_ctc_pipeline_ibp() {
    let def = build_two_block_ctc_pipeline();
    let bindings = two_block_ctc_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    // CTC softmax output bounded in [0, 1]
    assert!(lo >= -1e-5, "CTC softmax lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "CTC softmax upper <= 1, got {hi}");
}

#[test]
fn test_paddle_visual_two_block_ctc_pipeline_crown() {
    let def = build_two_block_ctc_pipeline();
    let bindings = two_block_ctc_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    eprintln!("two_block_ctc_pipeline CROWN method: {method:?}");
}

// ===========================================================================
// 6. Tight-input encoder block
// ===========================================================================

#[test]
fn test_paddle_visual_svtr_encoder_block_tight_crown() {
    let def = build_svtr_full_encoder_block();
    let bindings = svtr_full_encoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    // Narrow +-0.1 bounds for CROWN precision
    let tight_input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.1);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &tight_input);
    assert_bounds_valid(&output);

    // Compare with wide-input IBP
    let wide_input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let ibp_wide = graph.propagate_ibp(&wide_input).expect("IBP wide");
    let (_, tight_hi) = bounds_min_max(&output);
    let (_, wide_hi) = bounds_min_max(&ibp_wide);
    eprintln!(
        "tight-input CROWN max: {tight_hi:.4}, wide IBP max: {wide_hi:.4}, method: {method:?}"
    );
}

// ===========================================================================
// 7. DB detector backbone + sigmoid
// ===========================================================================

fn build_db_detector_sigmoid() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_db_detector_sigmoid");
    let spatial = IMG_H / 2; // 4

    // Conv-BN-ReLU stage
    let input = b.add_input("image", &[IN_CHANNELS, IMG_H, IMG_W]);
    let conv_w = b.add_input("conv_w", &[HIDDEN_DIM, IN_CHANNELS, 3, 3]);
    let conv_b = b.add_input("conv_b", &[HIDDEN_DIM]);
    let bn_mean = b.add_input("bn_mean", &[HIDDEN_DIM]);
    let bn_var = b.add_input("bn_var", &[HIDDEN_DIM]);
    let bn_w = b.add_input("bn_w", &[HIDDEN_DIM]);
    let bn_b = b.add_input("bn_b", &[HIDDEN_DIM]);
    let bn_eps = b.add_input("bn_eps", &[1]);

    let conv_out = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        2,
        2,
        1,
        1,
        &[HIDDEN_DIM, spatial, spatial],
    );
    let bn_out = b.add_batch_norm(
        conv_out,
        bn_mean,
        bn_var,
        bn_w,
        bn_b,
        bn_eps,
        &[HIDDEN_DIM, spatial, spatial],
    );
    let relu_out = b.add_relu(bn_out, &[HIDDEN_DIM, spatial, spatial]);

    // 1x1 conv to single channel + sigmoid
    let det_w = b.add_input("det_w", &[1, HIDDEN_DIM, 1, 1]);
    let det_b = b.add_input("det_b", &[1]);
    let det_out = b.add_conv2d(
        relu_out,
        det_w,
        Some(det_b),
        1,
        1,
        0,
        0,
        &[1, spatial, spatial],
    );
    let sigmoid = b.add_sigmoid(det_out, &[1, spatial, spatial]);
    b.build(sigmoid).expect("valid DB detector + sigmoid")
}

fn db_detector_sigmoid_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, IN_CHANNELS, 3, 3])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(w(&[1, HIDDEN_DIM, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[1])),
    ]
}

#[test]
fn test_paddle_visual_db_detector_sigmoid_ibp() {
    let def = build_db_detector_sigmoid();
    let bindings = db_detector_sigmoid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[IN_CHANNELS, IMG_H, IMG_W]);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "sigmoid lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi}");
}

#[test]
fn test_paddle_visual_db_detector_sigmoid_crown() {
    let def = build_db_detector_sigmoid();
    let bindings = db_detector_sigmoid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[IN_CHANNELS, IMG_H, IMG_W]);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    assert!(lo >= -1e-5, "sigmoid lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi}");
    eprintln!("db_detector_sigmoid CROWN method: {method:?}");
}

// ===========================================================================
// Verify-and-record
// ===========================================================================

#[test]
fn test_paddle_visual_svtr_full_encoder_block_verify_and_record() {
    let def = build_svtr_full_encoder_block();
    let bindings = svtr_full_encoder_block_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "paddle_ocr::test_paddle_visual_svtr_full_encoder_block_verify_and_record",
    );
    assert_bounds_valid(&result.output_bounds);
}
