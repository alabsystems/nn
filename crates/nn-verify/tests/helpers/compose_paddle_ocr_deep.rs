// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep NY compose tests for PaddleOCR subgraphs.
//!
//! These tests verify bounds propagation through intermediate-depth compositions
//! of the PaddleOCR pipeline (DB text detector + SVTR text recognizer). They
//! bridge the gap between existing sub-block tests (in `compose_dpdf_paddle_ocr.rs`)
//! and full end-to-end verification by exercising compositions at increasing depth:
//!
//! 1. **Self-attention isolation** — Q/K/V + softmax + out_proj (no LayerNorm).
//!    Tests attention-specific bounds in SVTR encoder (IBP + CROWN).
//!
//! 2. **SVTR full encoder block** — LayerNorm -> Attention -> residual ->
//!    LayerNorm -> GELU MLP -> residual. Full pre-norm transformer block (IBP + CROWN).
//!
//! 3. **Patch embed + one block** — Conv2d patch embedding + single SVTR
//!    encoder block. Cross-stage composition (IBP + CROWN).
//!
//! 4. **Two-block SVTR encoder with final LayerNorm** — depth composition +
//!    post-normalization for recognition output (IBP).
//!
//! 5. **Widening analysis** — 1-block vs 2-block IBP width comparison.
//!    Quantifies bounds growth through depth.
//!
//! 6. **SVTR encoder + CTC softmax** — Full recognition tail: 2 encoder
//!    blocks + final LN + CTC linear head + softmax (IBP).
//!
//! 7. **DB ResNet skip connection** — Conv-BN-ReLU -> Conv-BN + skip add ->
//!    ReLU. ResNet basic block from DB detector backbone (IBP + CROWN).
//!
//! 8. **DB 2-stage backbone + FPN sigmoid** — Multi-scale backbone -> FPN
//!    lateral projection -> sigmoid detection head (IBP).
//!
//! 9. **Tight-input SVTR attention** — Narrow bounds (+-0.1) for CROWN
//!    precision on attention softmax (IBP + CROWN).
//!
//! 10. **Full pipeline: patch embed -> 2 SVTR blocks -> LN -> CTC softmax**
//!     — End-to-end recognition with softmax output in [0, 1] (IBP).
//!
//! 11. **DB backbone widening: 1-stage vs 3-stage** — Spatial downsampling
//!     bounds growth through cascaded Conv-BN-ReLU stages (IBP).
//!
//! 12. **SVTR attention CROWN with position encoding** — Sinusoidal PE +
//!     attention block exercising CROWN linearization (CROWN).
//!
//! Architecture reference:
//! - PaddleOCR (Baidu): Production OCR with DB detector + SVTR recognizer
//! - DB (Liao et al. 2020): Differentiable Binarization for text detection
//! - SVTR (Du et al. 2022): Scene Text Recognition with a Single Visual Model
//! - CTC (Graves et al. 2006): Connectionist Temporal Classification
//!
//! Dimensions are small for fast verification (HIDDEN_DIM=16, SEQ_LEN=4).
//! All tests use IbpValidated soundness mode per nn engineering rules
//! (Sound refuses linearization for normalization layers).
//!
//! Part of #3928: deep NY compose tests for PaddleOCR.

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

/// Sequence length (number of patches).
const SEQ_LEN: usize = 4;
/// Embedding / hidden dimension.
const HIDDEN_DIM: usize = 16;
/// Number of attention heads.
const NUM_HEADS: usize = 4;
/// Head dimension = HIDDEN_DIM / NUM_HEADS.
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 4
/// FFN intermediate dimension (4x hidden).
const FFN_DIM: usize = 64;
/// Image spatial size (square).
const IMG_SIZE: usize = 16;
/// Patch size for embedding.
const PATCH_SIZE: usize = 4;
/// Grid size = IMG_SIZE / PATCH_SIZE.
const GRID_SIZE: usize = IMG_SIZE / PATCH_SIZE; // 4
/// Total patches = GRID_SIZE^2.
const NUM_PATCHES: usize = GRID_SIZE * GRID_SIZE; // 16
/// Input channels (RGB).
const IN_CH: usize = 3;
/// Backbone channels for DB detector.
const BACKBONE_CH: usize = 16;
/// Stage 2 channels.
const STAGE2_CH: usize = 32;
/// Half spatial dimension.
const HALF_IMG: usize = IMG_SIZE / 2; // 8
/// Vocabulary size for CTC head.
const VOCAB_SIZE: usize = 32;
/// Weight magnitude.
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
    .expect("valid image bounds [0, 1]")
}

/// Push a Conv-BN-ReLU stage's bindings (7 params).
fn push_conv_bn_relu_bindings(bindings: &mut Vec<TensorParamBinding>, out_ch: usize, in_ch: usize) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[out_ch, in_ch, 3, 3]),
        W_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[out_ch])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[out_ch])));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[out_ch])));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[out_ch])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[out_ch])));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
}

/// Push SVTR encoder block bindings: LN + Q/K/V/O + LN + FC1/FC2 (12 params).
fn push_svtr_block_bindings(bindings: &mut Vec<TensorParamBinding>) {
    let ln_w = ones(&[HIDDEN_DIM]);
    let ln_b = zeros(&[HIDDEN_DIM]);
    let proj = w(&[HIDDEN_DIM, HIDDEN_DIM]);
    let fc1 = w(&[FFN_DIM, HIDDEN_DIM]);
    let fc2 = w(&[HIDDEN_DIM, FFN_DIM]);

    // Attention sub-block: LN_w, LN_b, eps, Q, K, V, O
    bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(proj.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(proj.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(proj.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(proj));

    // MLP sub-block: LN_w, LN_b, eps, FC1, FC2
    bindings.push(TensorParamBinding::ConstantTensor(ln_w));
    bindings.push(TensorParamBinding::ConstantTensor(ln_b));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(fc1));
    bindings.push(TensorParamBinding::ConstantTensor(fc2));
}

/// Add one SVTR encoder block (attention + MLP) to the builder.
///
/// Returns the output node ID. Adds 12 input nodes to the builder.
fn add_svtr_block(
    b: &mut TensorBlockBuilder,
    x: nn_dsl::TensorNodeId,
    block_idx: usize,
) -> nn_dsl::TensorNodeId {
    add_svtr_block_seq(b, x, block_idx, SEQ_LEN)
}

/// Add one SVTR block operating on a `[seq_len, HIDDEN_DIM]` activation.
///
/// All weights act on the HIDDEN_DIM / FFN_DIM feature axis and are therefore
/// sequence-independent; only the activation shapes vary with `seq_len`. The
/// patch-embed paths use `seq_len == NUM_PATCHES` (the conv patch-grid token
/// count); plain block tests use `seq_len == SEQ_LEN`.
fn add_svtr_block_seq(
    b: &mut TensorBlockBuilder,
    x: nn_dsl::TensorNodeId,
    block_idx: usize,
    seq_len: usize,
) -> nn_dsl::TensorNodeId {
    let shape = [seq_len, HIDDEN_DIM];
    let ffn_shape = [seq_len, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let pfx = format!("b{block_idx}");

    // Attention sub-block
    let ln_a_w = b.add_input(&format!("{pfx}_ln1_w"), &[HIDDEN_DIM]);
    let ln_a_b = b.add_input(&format!("{pfx}_ln1_b"), &[HIDDEN_DIM]);
    let ln_a_eps = b.add_input(&format!("{pfx}_ln1_eps"), &[1]);
    let qw = b.add_input(&format!("{pfx}_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let kw = b.add_input(&format!("{pfx}_k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let vw = b.add_input(&format!("{pfx}_v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let ow = b.add_input(&format!("{pfx}_o_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let normed = b.add_layer_norm(x, ln_a_eps, 1, ln_a_w, ln_a_b, &shape);
    let q = b.add_linear(normed, qw, None, &shape);
    let k = b.add_linear(normed, kw, None, &shape);
    let v = b.add_linear(normed, vw, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let proj = b.add_linear(attn, ow, None, &shape);
    let res_a = b.add_binary_add(x, proj, &shape);

    // MLP sub-block
    let ln_b_w = b.add_input(&format!("{pfx}_ln2_w"), &[HIDDEN_DIM]);
    let ln_b_b = b.add_input(&format!("{pfx}_ln2_b"), &[HIDDEN_DIM]);
    let ln_b_eps = b.add_input(&format!("{pfx}_ln2_eps"), &[1]);
    let fc1 = b.add_input(&format!("{pfx}_fc1_w"), &[FFN_DIM, HIDDEN_DIM]);
    let fc2 = b.add_input(&format!("{pfx}_fc2_w"), &[HIDDEN_DIM, FFN_DIM]);

    let normed_b = b.add_layer_norm(res_a, ln_b_eps, 1, ln_b_w, ln_b_b, &shape);
    let h = b.add_linear(normed_b, fc1, None, &ffn_shape);
    let h_act = b.add_gelu(h, &ffn_shape);
    let mlp = b.add_linear(h_act, fc2, None, &shape);
    b.add_binary_add(res_a, mlp, &shape)
}

/// Add a Conv-BN-ReLU stage to the builder (7 input nodes).
fn add_conv_bn_relu(
    b: &mut TensorBlockBuilder,
    x: nn_dsl::TensorNodeId,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    stride: usize,
    out_h: usize,
    out_w: usize,
) -> nn_dsl::TensorNodeId {
    let cw = b.add_input(&format!("{prefix}_conv_w"), &[out_ch, in_ch, 3, 3]);
    let cb = b.add_input(&format!("{prefix}_conv_b"), &[out_ch]);
    let bm = b.add_input(&format!("{prefix}_bn_mean"), &[out_ch]);
    let bv = b.add_input(&format!("{prefix}_bn_var"), &[out_ch]);
    let bw = b.add_input(&format!("{prefix}_bn_w"), &[out_ch]);
    let bb = b.add_input(&format!("{prefix}_bn_b"), &[out_ch]);
    let eps = b.add_input(&format!("{prefix}_eps"), &[1]);

    let out_shape = [out_ch, out_h, out_w];
    let conv = b.add_conv2d(x, cw, Some(cb), stride, stride, 1, 1, &out_shape);
    let bn = b.add_batch_norm(conv, bm, bv, bw, bb, eps, &out_shape);
    b.add_relu(bn, &out_shape)
}

// ===========================================================================
// 1. Self-attention isolation: Q/K/V + softmax + out_proj
// ===========================================================================

/// Build isolated self-attention (no LayerNorm, no residual).
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_self_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_deep_self_attention");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let input = b.add_input("x", &shape);
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let q = b.add_linear(input, q_w, None, &shape);
    let k = b.add_linear(input, k_w, None, &shape);
    let v = b.add_linear(input, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let out = b.add_linear(attn, out_w, None, &shape);

    b.build(out).expect("valid self-attention kernel")
}

fn self_attention_bindings() -> Vec<TensorParamBinding> {
    let wp = w(&[HIDDEN_DIM, HIDDEN_DIM]);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(wp.clone()),
        TensorParamBinding::ConstantTensor(wp.clone()),
        TensorParamBinding::ConstantTensor(wp.clone()),
        TensorParamBinding::ConstantTensor(wp),
    ]
}

#[test]
fn test_paddle_ocr_deep_self_attention_ibp() {
    let def = build_self_attention_kernel();
    let bindings = self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR deep self-attention IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_paddle_ocr_deep_self_attention_crown() {
    let def = build_self_attention_kernel();
    let bindings = self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR deep self-attention CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 2. SVTR full encoder block: LN -> Attention -> residual -> LN -> MLP -> residual
// ===========================================================================

/// Build a complete SVTR encoder block (attention + MLP with pre-norm + residual).
fn build_full_encoder_block_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_deep_full_encoder_block");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = add_svtr_block(&mut b, input, 0);
    b.build(out).expect("valid full encoder block kernel")
}

fn full_encoder_block_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_svtr_block_bindings(&mut bindings);
    bindings
}

#[test]
fn test_paddle_ocr_deep_full_encoder_block_ibp() {
    let def = build_full_encoder_block_kernel();
    let bindings = full_encoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR deep full encoder block IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_paddle_ocr_deep_full_encoder_block_crown() {
    let def = build_full_encoder_block_kernel();
    let bindings = full_encoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR deep full encoder block CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 3. Patch embed + one encoder block
// ===========================================================================

/// Build patch embedding (Conv2d -> reshape -> transpose) followed by one
/// complete SVTR encoder block. Tests cross-stage boundary composition.
fn build_patch_embed_block_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_deep_patch_embed_block");
    // Patch embedding produces NUM_PATCHES tokens, so the block sequence length
    // here is NUM_PATCHES, not SEQ_LEN.
    let patch_shape = [NUM_PATCHES, HIDDEN_DIM];

    let input = b.add_input("image", &[IN_CH, IMG_SIZE, IMG_SIZE]);
    let patch_w = b.add_input("patch_w", &[HIDDEN_DIM, IN_CH, PATCH_SIZE, PATCH_SIZE]);
    let patch_b = b.add_input("patch_b", &[HIDDEN_DIM]);

    // Patch embedding: Conv2d -> reshape -> transpose
    let conv = b.add_conv2d(
        input,
        patch_w,
        Some(patch_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv, &[HIDDEN_DIM, NUM_PATCHES]);
    // transpose [HIDDEN_DIM, NUM_PATCHES] -> [NUM_PATCHES, HIDDEN_DIM]
    let patches = b.add_transpose(reshaped, &[1, 0], &patch_shape);

    // One encoder block over the NUM_PATCHES-length sequence
    let out = add_svtr_block_seq(&mut b, patches, 0, NUM_PATCHES);

    b.build(out).expect("valid patch_embed+block kernel")
}

fn patch_embed_block_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable, // image
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, IN_CH, PATCH_SIZE, PATCH_SIZE])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])),
    ];
    push_svtr_block_bindings(&mut bindings);
    bindings
}

#[test]
fn test_paddle_ocr_deep_patch_embed_block_ibp() {
    let def = build_patch_embed_block_kernel();
    let bindings = patch_embed_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    // Patch embedding yields NUM_PATCHES tokens.
    assert_eq!(output.lower_upper().0.shape(), &[NUM_PATCHES, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR deep patch_embed+block IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_paddle_ocr_deep_patch_embed_block_crown() {
    let def = build_patch_embed_block_kernel();
    let bindings = patch_embed_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR deep patch_embed+block CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 4. Two-block SVTR encoder + final LayerNorm
// ===========================================================================

/// Build 2 SVTR encoder blocks + final LayerNorm (recognition output path).
fn build_two_block_final_ln_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("paddle_ocr_deep_two_block_final_ln");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let input = b.add_input("x", &shape);
    let mut bindings = vec![TensorParamBinding::Variable];

    let x = add_svtr_block(&mut b, input, 0);
    push_svtr_block_bindings(&mut bindings);
    let x = add_svtr_block(&mut b, x, 1);
    push_svtr_block_bindings(&mut bindings);

    // Final LayerNorm
    let ln_w = b.add_input("final_ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("final_ln_b", &[HIDDEN_DIM]);
    let eps = b.add_input("final_ln_eps", &[1]);
    let out = b.add_layer_norm(x, eps, 1, ln_w, ln_b, &shape);

    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    let def = b.build(out).expect("valid two-block+LN kernel");
    (def, bindings)
}

#[test]
fn test_paddle_ocr_deep_two_block_final_ln_ibp() {
    let (def, bindings) = build_two_block_final_ln_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR deep two-block+final-LN IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 5. Widening analysis: 1-block vs 2-block encoder
// ===========================================================================

/// Build an N-block SVTR encoder for widening comparison.
fn build_n_block_encoder(n: usize) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let name = format!("paddle_ocr_deep_{n}block_widening");
    let mut b = TensorBlockBuilder::new(&name);
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let mut bindings = vec![TensorParamBinding::Variable];

    let mut x = input;
    for i in 0..n {
        x = add_svtr_block(&mut b, x, i);
        push_svtr_block_bindings(&mut bindings);
    }

    let def = b.build(x).expect("valid n-block encoder");
    (def, bindings)
}

#[test]
fn test_paddle_ocr_deep_widening_analysis() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // 1-block
    let (def1, bindings1) = build_n_block_encoder(1);
    let graph1 = tensor_kernel_to_graph(&def1, &bindings1).expect("graph");
    let output1 = graph1.propagate_ibp(&input).expect("IBP 1-block");
    let (lo1, hi1) = bounds_min_max(&output1);
    let width1 = hi1 - lo1;

    // 2-block
    let (def2, bindings2) = build_n_block_encoder(2);
    let graph2 = tensor_kernel_to_graph(&def2, &bindings2).expect("graph");
    let output2 = graph2.propagate_ibp(&input).expect("IBP 2-block");
    let (lo2, hi2) = bounds_min_max(&output2);
    let width2 = hi2 - lo2;

    eprintln!("PaddleOCR deep widening: 1-block width={width1:.4}, 2-block width={width2:.4}");
    eprintln!("  1-block: [{lo1:.4}, {hi1:.4}]");
    eprintln!("  2-block: [{lo2:.4}, {hi2:.4}]");

    assert!(width1.is_finite(), "1-block width not finite");
    assert!(width2.is_finite(), "2-block width not finite");
}

// ===========================================================================
// 6. SVTR encoder + CTC softmax output
// ===========================================================================

/// Build 2-block encoder + final LN + CTC linear head + softmax.
///
/// Output must be in [0, 1] (valid probability distribution).
fn build_encoder_ctc_softmax_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("paddle_ocr_deep_encoder_ctc_softmax");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let input = b.add_input("x", &shape);
    let mut bindings = vec![TensorParamBinding::Variable];

    // 2 encoder blocks
    let mut x = input;
    for i in 0..2 {
        x = add_svtr_block(&mut b, x, i);
        push_svtr_block_bindings(&mut bindings);
    }

    // Final LayerNorm
    let ln_w = b.add_input("final_ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("final_ln_b", &[HIDDEN_DIM]);
    let eps = b.add_input("final_ln_eps", &[1]);
    let normed = b.add_layer_norm(x, eps, 1, ln_w, ln_b, &shape);
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // CTC head: Linear + softmax
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(normed, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        VOCAB_SIZE, HIDDEN_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[VOCAB_SIZE])));

    let def = b.build(out).expect("valid encoder+CTC+softmax kernel");
    (def, bindings)
}

#[test]
fn test_paddle_ocr_deep_encoder_ctc_softmax_ibp() {
    let (def, bindings) = build_encoder_ctc_softmax_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR deep encoder+CTC+softmax IBP: [{lo_min}, {hi_max}]");

    // Softmax output must be in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "softmax lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "softmax upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 7. DB ResNet skip connection block
// ===========================================================================

/// Build a ResNet basic block with skip connection for the DB detector.
///
/// Input: `[BACKBONE_CH, IMG_SIZE, IMG_SIZE]` (Variable).
/// Output: `[BACKBONE_CH, IMG_SIZE, IMG_SIZE]`.
///
/// Path: Conv-BN -> ReLU -> Conv-BN -> Add(input) -> ReLU.
fn build_resnet_skip_block_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_deep_resnet_skip_block");
    let shape = [BACKBONE_CH, IMG_SIZE, IMG_SIZE];

    let input = b.add_input("x", &shape);

    // First conv-BN
    let c1_w = b.add_input("c1_w", &[BACKBONE_CH, BACKBONE_CH, 3, 3]);
    let c1_b = b.add_input("c1_b", &[BACKBONE_CH]);
    let b1_m = b.add_input("b1_mean", &[BACKBONE_CH]);
    let b1_v = b.add_input("b1_var", &[BACKBONE_CH]);
    let b1_w = b.add_input("b1_w", &[BACKBONE_CH]);
    let b1_b = b.add_input("b1_b", &[BACKBONE_CH]);
    let eps1 = b.add_input("eps1", &[1]);

    let conv1 = b.add_conv2d(input, c1_w, Some(c1_b), 1, 1, 1, 1, &shape);
    let bn1 = b.add_batch_norm(conv1, b1_m, b1_v, b1_w, b1_b, eps1, &shape);
    let relu1 = b.add_relu(bn1, &shape);

    // Second conv-BN (no ReLU before skip)
    let c2_w = b.add_input("c2_w", &[BACKBONE_CH, BACKBONE_CH, 3, 3]);
    let c2_b = b.add_input("c2_b", &[BACKBONE_CH]);
    let b2_m = b.add_input("b2_mean", &[BACKBONE_CH]);
    let b2_v = b.add_input("b2_var", &[BACKBONE_CH]);
    let b2_w = b.add_input("b2_w", &[BACKBONE_CH]);
    let b2_b = b.add_input("b2_b", &[BACKBONE_CH]);
    let eps2 = b.add_input("eps2", &[1]);

    let conv2 = b.add_conv2d(relu1, c2_w, Some(c2_b), 1, 1, 1, 1, &shape);
    let bn2 = b.add_batch_norm(conv2, b2_m, b2_v, b2_w, b2_b, eps2, &shape);

    // Skip connection + ReLU
    let skip = b.add_binary_add(input, bn2, &shape);
    let out = b.add_relu(skip, &shape);

    b.build(out).expect("valid ResNet skip block kernel")
}

fn resnet_skip_block_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    // First conv-BN
    push_conv_bn_relu_bindings(&mut bindings, BACKBONE_CH, BACKBONE_CH);
    // Second conv-BN (same structure)
    push_conv_bn_relu_bindings(&mut bindings, BACKBONE_CH, BACKBONE_CH);
    bindings
}

#[test]
fn test_paddle_ocr_deep_resnet_skip_block_ibp() {
    let def = build_resnet_skip_block_kernel();
    let bindings = resnet_skip_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[BACKBONE_CH, IMG_SIZE, IMG_SIZE], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[BACKBONE_CH, IMG_SIZE, IMG_SIZE]
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR deep ResNet skip block IBP: [{lo_min}, {hi_max}]");

    // ReLU clamps lower to >= 0
    assert!(
        lo_min >= -1e-4,
        "ReLU output lower should be >= 0, got {lo_min}"
    );
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_paddle_ocr_deep_resnet_skip_block_crown() {
    let def = build_resnet_skip_block_kernel();
    let bindings = resnet_skip_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[BACKBONE_CH, IMG_SIZE, IMG_SIZE], 2.0);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR deep ResNet skip block CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 8. DB 2-stage backbone + sigmoid detection head
// ===========================================================================

/// Build 2-stage DB backbone (stride-2 downsampling) + sigmoid head.
///
/// Input: `[IN_CH, IMG_SIZE, IMG_SIZE]` (Variable, image in [0, 1]).
/// Output: `[1, HALF_IMG, HALF_IMG]` (probability map in [0, 1]).
fn build_db_backbone_sigmoid_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_deep_db_backbone_sigmoid");

    let input = b.add_input("image", &[IN_CH, IMG_SIZE, IMG_SIZE]);

    // Stage 1: preserve spatial
    let s1 = add_conv_bn_relu(
        &mut b,
        input,
        "s1",
        IN_CH,
        BACKBONE_CH,
        1,
        IMG_SIZE,
        IMG_SIZE,
    );

    // Stage 2: downsample 2x
    let s2 = add_conv_bn_relu(
        &mut b,
        s1,
        "s2",
        BACKBONE_CH,
        STAGE2_CH,
        2,
        HALF_IMG,
        HALF_IMG,
    );

    // 1x1 projection + sigmoid
    let head_w = b.add_input("head_w", &[1, STAGE2_CH, 1, 1]);
    let head_b = b.add_input("head_b", &[1]);
    let head_shape = [1, HALF_IMG, HALF_IMG];
    let proj = b.add_conv2d(s2, head_w, Some(head_b), 1, 1, 0, 0, &head_shape);
    let out = b.add_sigmoid(proj, &head_shape);

    b.build(out).expect("valid DB backbone+sigmoid kernel")
}

fn db_backbone_sigmoid_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_relu_bindings(&mut bindings, BACKBONE_CH, IN_CH);
    push_conv_bn_relu_bindings(&mut bindings, STAGE2_CH, BACKBONE_CH);
    bindings.push(TensorParamBinding::ConstantTensor(w(&[1, STAGE2_CH, 1, 1])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[1])));
    bindings
}

#[test]
fn test_paddle_ocr_deep_db_backbone_sigmoid_ibp() {
    let def = build_db_backbone_sigmoid_kernel();
    let bindings = db_backbone_sigmoid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[1, HALF_IMG, HALF_IMG]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR deep DB backbone+sigmoid IBP: [{lo_min}, {hi_max}]");

    // Sigmoid output must be in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "sigmoid lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "sigmoid upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 9. Tight-input SVTR attention (narrow bounds for CROWN precision)
// ===========================================================================

/// Build SVTR attention block with pre-norm, tested with narrow input bounds.
///
/// Narrow bounds (+-0.1) reduce the relaxation gap in softmax and LayerNorm
/// linearization, allowing CROWN to produce tighter results.
fn build_tight_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_deep_tight_attention");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let input = b.add_input("x", &shape);
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &shape);
    let q = b.add_linear(normed, q_w, None, &shape);
    let k = b.add_linear(normed, k_w, None, &shape);
    let v = b.add_linear(normed, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let proj = b.add_linear(attn, out_w, None, &shape);
    let out = b.add_binary_add(input, proj, &shape);

    b.build(out).expect("valid tight attention kernel")
}

fn tight_attention_bindings() -> Vec<TensorParamBinding> {
    let wp = w(&[HIDDEN_DIM, HIDDEN_DIM]);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(wp.clone()),
        TensorParamBinding::ConstantTensor(wp.clone()),
        TensorParamBinding::ConstantTensor(wp.clone()),
        TensorParamBinding::ConstantTensor(wp),
    ]
}

#[test]
fn test_paddle_ocr_deep_tight_attention_ibp() {
    let def = build_tight_attention_kernel();
    let bindings = tight_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    // Narrow input: +-0.1
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.1);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("PaddleOCR deep tight attention IBP (+-0.1): [{lo_min}, {hi_max}], width={width:.6}");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_paddle_ocr_deep_tight_attention_crown() {
    let def = build_tight_attention_kernel();
    let bindings = tight_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.1);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!(
        "PaddleOCR deep tight attention CROWN ({method:?}): [{lo_min}, {hi_max}], width={width:.6}"
    );
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 10. Full recognition: patch embed -> 2 blocks -> LN -> CTC softmax
// ===========================================================================

/// Build the full SVTR recognition pipeline: image -> patches -> encoder ->
/// CTC softmax output. Tests end-to-end bounds through the entire recognizer.
fn build_full_recognition_pipeline_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("paddle_ocr_deep_full_recognition");
    // Patch embedding produces NUM_PATCHES tokens, so the whole recognition
    // sequence (encoder blocks + final LN + CTC head) runs at length NUM_PATCHES.
    let patch_shape = [NUM_PATCHES, HIDDEN_DIM];

    let input = b.add_input("image", &[IN_CH, IMG_SIZE, IMG_SIZE]);
    let patch_w = b.add_input("patch_w", &[HIDDEN_DIM, IN_CH, PATCH_SIZE, PATCH_SIZE]);
    let patch_b = b.add_input("patch_b", &[HIDDEN_DIM]);

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, IN_CH, PATCH_SIZE, PATCH_SIZE])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])),
    ];

    // Patch embedding
    let conv = b.add_conv2d(
        input,
        patch_w,
        Some(patch_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv, &[HIDDEN_DIM, NUM_PATCHES]);
    // transpose [HIDDEN_DIM, NUM_PATCHES] -> [NUM_PATCHES, HIDDEN_DIM]
    let patches = b.add_transpose(reshaped, &[1, 0], &patch_shape);

    // 2 encoder blocks over the NUM_PATCHES-length sequence
    let mut x = patches;
    for i in 0..2 {
        x = add_svtr_block_seq(&mut b, x, i, NUM_PATCHES);
        push_svtr_block_bindings(&mut bindings);
    }

    // Final LayerNorm
    let ln_w = b.add_input("final_ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("final_ln_b", &[HIDDEN_DIM]);
    let eps = b.add_input("final_ln_eps", &[1]);
    let normed = b.add_layer_norm(x, eps, 1, ln_w, ln_b, &patch_shape);
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // CTC head + softmax
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(normed, ctc_w, Some(ctc_b), &[NUM_PATCHES, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[NUM_PATCHES, VOCAB_SIZE]);
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        VOCAB_SIZE, HIDDEN_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[VOCAB_SIZE])));

    let def = b
        .build(out)
        .expect("valid full recognition pipeline kernel");
    (def, bindings)
}

#[test]
fn test_paddle_ocr_deep_full_recognition_ibp() {
    let (def, bindings) = build_full_recognition_pipeline_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    // Patch embedding yields NUM_PATCHES tokens.
    assert_eq!(output.lower_upper().0.shape(), &[NUM_PATCHES, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR deep full recognition IBP: [{lo_min}, {hi_max}]");

    // Softmax output must be in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "softmax lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "softmax upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 11. DB backbone widening: 1-stage vs 3-stage
// ===========================================================================

/// Build an N-stage DB backbone (Conv-BN-ReLU chain with downsampling).
fn build_n_stage_backbone(n: usize) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let name = format!("paddle_ocr_deep_db_{n}stage_widening");
    let mut b = TensorBlockBuilder::new(&name);

    let input = b.add_input("image", &[IN_CH, IMG_SIZE, IMG_SIZE]);
    let mut bindings = vec![TensorParamBinding::Variable];

    let channels = [BACKBONE_CH, STAGE2_CH, 64_usize];
    let in_channels = [IN_CH, BACKBONE_CH, STAGE2_CH];
    let strides = [1_usize, 2, 2];
    let mut spatial = IMG_SIZE;

    let mut x = input;
    for i in 0..n.min(3) {
        if strides[i] == 2 {
            spatial /= 2;
        }
        x = add_conv_bn_relu(
            &mut b,
            x,
            &format!("s{i}"),
            in_channels[i],
            channels[i],
            strides[i],
            spatial,
            spatial,
        );
        push_conv_bn_relu_bindings(&mut bindings, channels[i], in_channels[i]);
    }

    let def = b.build(x).expect("valid n-stage backbone");
    (def, bindings)
}

#[test]
fn test_paddle_ocr_deep_db_backbone_widening() {
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);

    // 1-stage
    let (def1, bindings1) = build_n_stage_backbone(1);
    let graph1 = tensor_kernel_to_graph(&def1, &bindings1).expect("graph");
    let output1 = graph1.propagate_ibp(&input).expect("IBP 1-stage");
    let (lo1, hi1) = bounds_min_max(&output1);
    let width1 = hi1 - lo1;

    // 3-stage
    let (def3, bindings3) = build_n_stage_backbone(3);
    let graph3 = tensor_kernel_to_graph(&def3, &bindings3).expect("graph");
    let output3 = graph3.propagate_ibp(&input).expect("IBP 3-stage");
    let (lo3, hi3) = bounds_min_max(&output3);
    let width3 = hi3 - lo3;

    eprintln!("PaddleOCR deep DB backbone widening:");
    eprintln!("  1-stage: [{lo1:.4}, {hi1:.4}], width={width1:.4}");
    eprintln!("  3-stage: [{lo3:.4}, {hi3:.4}], width={width3:.4}");

    assert!(width1.is_finite(), "1-stage width not finite");
    assert!(width3.is_finite(), "3-stage width not finite");

    // ReLU at each stage clips negative values
    assert!(lo1 >= -1e-4, "1-stage ReLU lower should be >= 0");
    assert!(lo3 >= -1e-4, "3-stage ReLU lower should be >= 0");
}

// ===========================================================================
// 12. SVTR attention CROWN with sinusoidal position encoding
// ===========================================================================

/// Build SVTR attention block with additive sinusoidal position encoding.
///
/// Tests CROWN linearization when position encoding shifts the input
/// distribution before attention.
fn build_attention_with_pe_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_deep_attention_with_pe");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let input = b.add_input("x", &shape);
    let pos_enc = b.add_input("pos_enc", &shape);

    // Add positional encoding
    let x_pos = b.add_binary_add(input, pos_enc, &shape);

    // Attention block with pre-norm
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let normed = b.add_layer_norm(x_pos, eps, 1, ln_w, ln_b, &shape);
    let q = b.add_linear(normed, q_w, None, &shape);
    let k = b.add_linear(normed, k_w, None, &shape);
    let v = b.add_linear(normed, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let proj = b.add_linear(attn, out_w, None, &shape);
    let out = b.add_binary_add(x_pos, proj, &shape);

    b.build(out).expect("valid attention+PE kernel")
}

fn attention_with_pe_bindings() -> Vec<TensorParamBinding> {
    // Generate sinusoidal PE
    let n = SEQ_LEN * HIDDEN_DIM;
    let mut pe_data = Vec::with_capacity(n);
    for t in 0..SEQ_LEN {
        for d in 0..HIDDEN_DIM / 2 {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * d as f64 / HIDDEN_DIM as f64);
            pe_data.push(freq.sin() as f32);
            pe_data.push(freq.cos() as f32);
        }
    }
    let pos_enc =
        ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), pe_data).expect("valid PE shape");

    let wp = w(&[HIDDEN_DIM, HIDDEN_DIM]);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pos_enc),
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(wp.clone()),
        TensorParamBinding::ConstantTensor(wp.clone()),
        TensorParamBinding::ConstantTensor(wp.clone()),
        TensorParamBinding::ConstantTensor(wp),
    ]
}

#[test]
fn test_paddle_ocr_deep_attention_with_pe_crown() {
    let def = build_attention_with_pe_kernel();
    let bindings = attention_with_pe_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR deep attention+PE CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// Verify-and-record tests
// ===========================================================================

#[test]
fn test_paddle_ocr_deep_full_encoder_block_verify() {
    let def = build_full_encoder_block_kernel();
    let bindings = full_encoder_block_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "paddle_ocr_deep_full_encoder_block",
    );
    assert_eq!(result.num_variables, 1);
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

#[test]
fn test_paddle_ocr_deep_resnet_skip_verify() {
    let def = build_resnet_skip_block_kernel();
    let bindings = resnet_skip_block_bindings();
    let input = uniform_bounds(&[BACKBONE_CH, IMG_SIZE, IMG_SIZE], 2.0);

    let result = verify_and_assert(&def, &bindings, &input, "paddle_ocr_deep_resnet_skip_block");
    assert_eq!(result.num_variables, 1);
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[BACKBONE_CH, IMG_SIZE, IMG_SIZE]);
}
