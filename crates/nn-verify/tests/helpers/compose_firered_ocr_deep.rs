// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep NY compose tests for FireRed-OCR subgraphs.
//!
//! These tests verify bounds propagation through intermediate-depth compositions
//! of the FireRed-OCR pipeline (Qwen3-VL-2B variant with CTC decoding). They
//! bridge the gap between existing sub-block tests (in `compose_dpdf_firered_ocr.rs`)
//! and full end-to-end verification by exercising compositions at increasing depth:
//!
//! 1. **Full encoder block IBP + CROWN**: RMSNorm -> 4-head Attention -> residual
//!    -> RMSNorm -> SwiGLU FFN -> residual. Complete vision encoder block.
//!
//! 2. **2-layer encoder stack IBP**: Depth composition widening through two
//!    stacked encoder blocks. Measures IBP bounds growth.
//!
//! 3. **Patch embed + encoder block IBP + CROWN**: Conv2d(3, D, 2, stride=2)
//!    patch embedding followed by one encoder block. Cross-stage composition.
//!
//! 4. **CTC pipeline IBP**: 2 encoder blocks -> Linear(HIDDEN, VOCAB) -> Softmax.
//!    Character probability output bounded in [0, 1].
//!
//! 5. **Full OCR pipeline IBP**: Patch embed -> 2 encoder blocks -> CTC head ->
//!    softmax. End-to-end from image pixels to character probabilities.
//!
//! 6. **Tight-input CROWN**: Narrow +-0.1 bounds through encoder block for
//!    CROWN precision analysis on RMSNorm + softmax linearization.
//!
//! 7. **Widening analysis IBP**: 1-block vs 2-block encoder IBP width comparison.
//!    Quantifies bounds growth through depth.
//!
//! 8. **RMSNorm -> SwiGLU -> RMSNorm sandwich IBP**: Pre-norm FFN sandwich
//!    testing normalization stability through SwiGLU gating.
//!
//! 9. **Patch embedding + positional encoding IBP**: Patch embed followed by
//!    additive learned positional bias.
//!
//! 10. **Full pipeline CROWN**: End-to-end patch embed -> 2 blocks -> CTC ->
//!     softmax with CROWN linearization for tighter bounds.
//!
//! Architecture reference:
//! - FireRed-OCR: Qwen3-VL-2B variant for document OCR with CTC decoding
//! - Qwen2-VL / Qwen3-VL (Alibaba): Vision-language model with RMSNorm, SwiGLU
//! - SwiGLU (Shazeer, 2020): SiLU-gated FFN
//! - CTC (Graves et al. 2006): Connectionist Temporal Classification
//!
//! Dimensions are small for fast verification (HIDDEN_DIM=16, NUM_HEADS=4).
//! All tests use IbpValidated soundness mode per nn engineering rules
//! (Sound refuses linearization for normalization layers).
//!
//! Part of #3906: deep NY compose tests for FireRed-OCR subgraphs.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// of Qwen3-VL-2B architecture used in FireRed-OCR
// ---------------------------------------------------------------------------

/// Sequence length (number of patches).
const SEQ_LEN: usize = 4;
/// Embedding / hidden dimension.
const HIDDEN_DIM: usize = 16;
/// Number of attention heads.
const NUM_HEADS: usize = 4;
/// Head dimension = HIDDEN_DIM / NUM_HEADS.
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 4
/// FFN intermediate dimension (SwiGLU gate and up projections).
const FFN_DIM: usize = 32;
/// Image spatial size (square).
const IMG_SIZE: usize = 8;
/// Patch size for embedding.
const PATCH_SIZE: usize = 2;
/// Grid size = IMG_SIZE / PATCH_SIZE.
const GRID_SIZE: usize = IMG_SIZE / PATCH_SIZE; // 4
/// Total patches = GRID_SIZE^2.
const NUM_PATCHES: usize = GRID_SIZE * GRID_SIZE; // 16
/// Input channels (RGB).
const IN_CH: usize = 3;
/// OCR vocabulary size (characters + blank token for CTC).
const VOCAB_SIZE: usize = 64;
/// Weight magnitude for bounded verification.
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

/// Push FireRed-OCR encoder block bindings (11 params).
///
/// RMSNorm1(eps, w) + Q/K/V/O projections + RMSNorm2(eps, w) + gate/up/down.
fn push_encoder_block_bindings(bindings: &mut Vec<TensorParamBinding>) {
    let norm_w = ones(&[HIDDEN_DIM]);
    let proj = w(&[HIDDEN_DIM, HIDDEN_DIM]);
    let gate = w(&[FFN_DIM, HIDDEN_DIM]);
    let up = w(&[FFN_DIM, HIDDEN_DIM]);
    let down = w(&[HIDDEN_DIM, FFN_DIM]);

    // RMSNorm1: eps, weight
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone()));
    // Q, K, V, O projections
    bindings.push(TensorParamBinding::ConstantTensor(proj.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(proj.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(proj.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(proj));
    // RMSNorm2: eps, weight
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(norm_w));
    // SwiGLU: gate, up, down
    bindings.push(TensorParamBinding::ConstantTensor(gate));
    bindings.push(TensorParamBinding::ConstantTensor(up));
    bindings.push(TensorParamBinding::ConstantTensor(down));
}

/// Add one FireRed-OCR encoder block to the builder.
///
/// RMSNorm -> Attention -> residual -> RMSNorm -> SwiGLU FFN -> residual.
/// Returns the output node ID. Adds 11 input nodes to the builder.
fn add_encoder_block(
    b: &mut TensorBlockBuilder,
    x: nn_dsl::TensorNodeId,
    block_idx: usize,
) -> nn_dsl::TensorNodeId {
    add_encoder_block_seq(b, x, block_idx, SEQ_LEN)
}

/// Add one encoder block operating on a `[seq_len, HIDDEN_DIM]` activation.
///
/// All weights are sequence-independent (they act on the HIDDEN_DIM / FFN_DIM
/// feature axis), so only the activation shapes vary with `seq_len`. The
/// patch-embed paths use `seq_len == NUM_PATCHES` (the token count produced by
/// the conv patch grid); plain encoder tests use `seq_len == SEQ_LEN`.
fn add_encoder_block_seq(
    b: &mut TensorBlockBuilder,
    x: nn_dsl::TensorNodeId,
    block_idx: usize,
    seq_len: usize,
) -> nn_dsl::TensorNodeId {
    let shape = [seq_len, HIDDEN_DIM];
    let ffn_shape = [seq_len, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let pfx = format!("b{block_idx}");

    // Pre-attention RMSNorm
    let norm1_eps = b.add_input(&format!("{pfx}_norm1_eps"), &[1]);
    let norm1_w = b.add_input(&format!("{pfx}_norm1_w"), &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(x, norm1_eps, 1, norm1_w, &shape);

    // Self-attention: Q, K, V, O projections
    let q_w = b.add_input(&format!("{pfx}_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input(&format!("{pfx}_k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input(&format!("{pfx}_v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input(&format!("{pfx}_o_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, o_w, None, &shape);

    // Residual after attention
    let res1 = b.add_binary_add(x, attn_out, &shape);

    // Pre-FFN RMSNorm
    let norm2_eps = b.add_input(&format!("{pfx}_norm2_eps"), &[1]);
    let norm2_w = b.add_input(&format!("{pfx}_norm2_w"), &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &shape);

    // SwiGLU FFN: gate_proj -> sigmoid -> mul(gate) -> mul(up_proj) -> down_proj
    let gate_w = b.add_input(&format!("{pfx}_gate_w"), &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input(&format!("{pfx}_up_w"), &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input(&format!("{pfx}_down_w"), &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_activated = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_activated, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    // Residual after FFN
    b.add_binary_add(res1, ffn_out, &shape)
}

// ===========================================================================
// 1. Full encoder block: RMSNorm -> Attention -> residual -> SwiGLU -> residual
// ===========================================================================

/// Build one complete FireRed-OCR encoder block.
fn build_full_encoder_block_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_deep_full_encoder_block");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = add_encoder_block(&mut b, input, 0);
    b.build(out).expect("valid full encoder block kernel")
}

fn full_encoder_block_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_encoder_block_bindings(&mut bindings);
    bindings
}

#[test]
fn test_firered_ocr_deep_full_encoder_block_ibp() {
    let def = build_full_encoder_block_kernel();
    let bindings = full_encoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR deep full encoder block IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_firered_ocr_deep_full_encoder_block_crown() {
    let def = build_full_encoder_block_kernel();
    let bindings = full_encoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR deep full encoder block CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 2. 2-layer encoder stack: depth composition widening
// ===========================================================================

/// Build an N-block encoder for depth composition analysis.
fn build_n_block_encoder(n: usize) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let name = format!("firered_ocr_deep_{n}block_encoder");
    let mut b = TensorBlockBuilder::new(&name);
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let mut bindings = vec![TensorParamBinding::Variable];

    let mut x = input;
    for i in 0..n {
        x = add_encoder_block(&mut b, x, i);
        push_encoder_block_bindings(&mut bindings);
    }

    let def = b.build(x).expect("valid n-block encoder");
    (def, bindings)
}

#[test]
fn test_firered_ocr_deep_two_layer_encoder_ibp() {
    let (def, bindings) = build_n_block_encoder(2);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR deep 2-layer encoder IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 3. Patch embed + encoder block: Conv2d(3, D, 2, stride=2) + block
// ===========================================================================

/// Build patch embedding followed by one encoder block.
///
/// Conv2d(3, HIDDEN_DIM, PATCH_SIZE, stride=PATCH_SIZE) -> reshape -> transpose
/// -> encoder block. Tests cross-stage boundary composition from image to
/// transformer features.
fn build_patch_embed_encoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_deep_patch_embed_encoder");
    // Patch embedding produces NUM_PATCHES tokens (the conv patch grid), so the
    // transformer sequence length here is NUM_PATCHES, not SEQ_LEN.
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
    let out = add_encoder_block_seq(&mut b, patches, 0, NUM_PATCHES);

    b.build(out).expect("valid patch_embed+encoder kernel")
}

fn patch_embed_encoder_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable, // image
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, IN_CH, PATCH_SIZE, PATCH_SIZE])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])),
    ];
    push_encoder_block_bindings(&mut bindings);
    bindings
}

#[test]
fn test_firered_ocr_deep_patch_embed_encoder_ibp() {
    let def = build_patch_embed_encoder_kernel();
    let bindings = patch_embed_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    // Patch embedding yields NUM_PATCHES tokens.
    assert_eq!(output.lower_upper().0.shape(), &[NUM_PATCHES, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR deep patch_embed+encoder IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_firered_ocr_deep_patch_embed_encoder_crown() {
    let def = build_patch_embed_encoder_kernel();
    let bindings = patch_embed_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR deep patch_embed+encoder CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 4. CTC pipeline: 2 encoder blocks -> Linear(HIDDEN, VOCAB) -> Softmax
// ===========================================================================

/// Build 2-block encoder + CTC linear head + softmax.
///
/// Output must be in [0, 1] (valid probability distribution).
fn build_ctc_pipeline_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("firered_ocr_deep_ctc_pipeline");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let input = b.add_input("x", &shape);
    let mut bindings = vec![TensorParamBinding::Variable];

    // 2 encoder blocks
    let mut x = input;
    for i in 0..2 {
        x = add_encoder_block(&mut b, x, i);
        push_encoder_block_bindings(&mut bindings);
    }

    // CTC head: Linear + softmax
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(x, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        VOCAB_SIZE, HIDDEN_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[VOCAB_SIZE])));

    let def = b.build(out).expect("valid CTC pipeline kernel");
    (def, bindings)
}

#[test]
fn test_firered_ocr_deep_ctc_pipeline_ibp() {
    let (def, bindings) = build_ctc_pipeline_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR deep CTC pipeline IBP: [{lo_min}, {hi_max}]");

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
// 5. Full OCR pipeline: patch embed -> 2 blocks -> CTC head -> softmax
// ===========================================================================

/// Build the full FireRed-OCR recognition pipeline: image -> patches ->
/// 2 encoder blocks -> CTC softmax output.
fn build_full_ocr_pipeline_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("firered_ocr_deep_full_ocr_pipeline");
    // Patch embedding produces NUM_PATCHES tokens, so the whole recognition
    // sequence (encoder blocks + CTC head) runs at length NUM_PATCHES.
    let patch_shape = [NUM_PATCHES, HIDDEN_DIM];

    let input = b.add_input("image", &[IN_CH, IMG_SIZE, IMG_SIZE]);
    let patch_w = b.add_input("patch_w", &[HIDDEN_DIM, IN_CH, PATCH_SIZE, PATCH_SIZE]);
    let patch_b = b.add_input("patch_b", &[HIDDEN_DIM]);

    let mut bindings = vec![
        TensorParamBinding::Variable, // image
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
        x = add_encoder_block_seq(&mut b, x, i, NUM_PATCHES);
        push_encoder_block_bindings(&mut bindings);
    }

    // CTC head: Linear + softmax
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(x, ctc_w, Some(ctc_b), &[NUM_PATCHES, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[NUM_PATCHES, VOCAB_SIZE]);

    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        VOCAB_SIZE, HIDDEN_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[VOCAB_SIZE])));

    let def = b.build(out).expect("valid full OCR pipeline kernel");
    (def, bindings)
}

#[test]
fn test_firered_ocr_deep_full_ocr_pipeline_ibp() {
    let (def, bindings) = build_full_ocr_pipeline_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    // Patch embedding yields NUM_PATCHES tokens.
    assert_eq!(output.lower_upper().0.shape(), &[NUM_PATCHES, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR deep full OCR pipeline IBP: [{lo_min}, {hi_max}]");

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
// 6. Tight-input CROWN: Narrow +-0.1 bounds through encoder block
// ===========================================================================

/// Narrow input bounds (+-0.1) reduce the relaxation gap in RMSNorm and
/// softmax linearization, allowing CROWN to produce tighter results.

#[test]
fn test_firered_ocr_deep_tight_encoder_ibp() {
    let def = build_full_encoder_block_kernel();
    let bindings = full_encoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.1);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("FireRed-OCR deep tight encoder IBP (+-0.1): [{lo_min}, {hi_max}], width={width:.6}");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_firered_ocr_deep_tight_encoder_crown() {
    let def = build_full_encoder_block_kernel();
    let bindings = full_encoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.1);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!(
        "FireRed-OCR deep tight encoder CROWN ({method:?}): [{lo_min}, {hi_max}], width={width:.6}"
    );
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 7. Widening analysis: 1-block vs 2-block encoder
// ===========================================================================

#[test]
fn test_firered_ocr_deep_widening_analysis() {
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

    eprintln!("FireRed-OCR deep widening: 1-block width={width1:.4}, 2-block width={width2:.4}");
    eprintln!("  1-block: [{lo1:.4}, {hi1:.4}]");
    eprintln!("  2-block: [{lo2:.4}, {hi2:.4}]");

    assert!(width1.is_finite(), "1-block width not finite");
    assert!(width2.is_finite(), "2-block width not finite");
}

// ===========================================================================
// 8. RMSNorm -> SwiGLU -> RMSNorm sandwich
// ===========================================================================

/// Build a pre-norm FFN sandwich: RMSNorm -> SwiGLU FFN -> RMSNorm.
/// Tests normalization stability through SwiGLU gating.
fn build_norm_swiglu_norm_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_deep_norm_swiglu_norm");
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    let input = b.add_input("x", &shape);

    // Pre-FFN RMSNorm
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_w", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    // SwiGLU FFN
    let gate_w = b.add_input("gate_w", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_w", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_w", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed1, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_activated = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed1, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_activated, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    // Post-FFN RMSNorm
    let norm2_eps = b.add_input("norm2_eps", &[1]);
    let norm2_w = b.add_input("norm2_w", &[HIDDEN_DIM]);
    let out = b.add_rms_norm(ffn_out, norm2_eps, 1, norm2_w, &shape);

    b.build(out).expect("valid norm-swiglu-norm kernel")
}

fn norm_swiglu_norm_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, FFN_DIM])),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])),
    ]
}

#[test]
fn test_firered_ocr_deep_norm_swiglu_norm_ibp() {
    let def = build_norm_swiglu_norm_kernel();
    let bindings = norm_swiglu_norm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR deep RMSNorm-SwiGLU-RMSNorm IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 9. Patch embedding + positional encoding
// ===========================================================================

/// Build patch embedding followed by additive positional bias.
///
/// Conv2d patch embed -> reshape -> transpose -> add(pos_bias).
/// Tests that learned positional encoding shifts bounds uniformly.
fn build_patch_embed_pos_enc_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_deep_patch_embed_pos_enc");
    // Patch embedding produces NUM_PATCHES tokens; the positional bias is added
    // per token, so it spans the NUM_PATCHES-length sequence.
    let patch_shape = [NUM_PATCHES, HIDDEN_DIM];

    let input = b.add_input("image", &[IN_CH, IMG_SIZE, IMG_SIZE]);
    let patch_w = b.add_input("patch_w", &[HIDDEN_DIM, IN_CH, PATCH_SIZE, PATCH_SIZE]);
    let patch_b = b.add_input("patch_b", &[HIDDEN_DIM]);
    let pos_bias = b.add_input("pos_bias", &patch_shape);

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

    // Add positional bias
    let out = b.add_binary_add(patches, pos_bias, &patch_shape);

    b.build(out).expect("valid patch_embed+pos_enc kernel")
}

fn patch_embed_pos_enc_bindings() -> Vec<TensorParamBinding> {
    // Generate simple sinusoidal positional bias over the NUM_PATCHES sequence.
    let n = NUM_PATCHES * HIDDEN_DIM;
    let mut pe_data = Vec::with_capacity(n);
    for t in 0..NUM_PATCHES {
        for d in 0..HIDDEN_DIM {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * (d / 2) as f64 / HIDDEN_DIM as f64);
            let val = if d % 2 == 0 {
                freq.sin() as f32
            } else {
                freq.cos() as f32
            };
            pe_data.push(val);
        }
    }
    let pe = ArrayD::from_shape_vec(IxDyn(&[NUM_PATCHES, HIDDEN_DIM]), pe_data).expect("valid PE");

    vec![
        TensorParamBinding::Variable, // image
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, IN_CH, PATCH_SIZE, PATCH_SIZE])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(pe),
    ]
}

#[test]
fn test_firered_ocr_deep_patch_embed_pos_enc_ibp() {
    let def = build_patch_embed_pos_enc_kernel();
    let bindings = patch_embed_pos_enc_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    // Patch embedding yields NUM_PATCHES tokens.
    assert_eq!(output.lower_upper().0.shape(), &[NUM_PATCHES, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR deep patch_embed+pos_enc IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 10. Full pipeline CROWN: patch embed -> 2 blocks -> CTC -> softmax
// ===========================================================================

#[test]
fn test_firered_ocr_deep_full_ocr_pipeline_crown() {
    let (def, bindings) = build_full_ocr_pipeline_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_bounds(&[IN_CH, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR deep full OCR pipeline CROWN ({method:?}): [{lo_min}, {hi_max}]");
    if let Some(r) = &fallback {
        eprintln!("Fallback: {r}");
    }

    assert_bounds_valid(&output);

    // Softmax output must be in [0, 1] even with CROWN
    assert!(
        lo_min >= -1e-4,
        "softmax lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "softmax upper should be <= 1, got {hi_max}"
    );
}
