// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests for PaddleOCR-VL text recognition pipeline bounds.
//!
//! Extends the detection tests in `compose_dpdf_paddle_detection.rs` with
//! recognition pipeline stages: ResNet encoder backbone, CTC/attention decoders,
//! multi-scale feature fusion, beam search output bounds, character embeddings,
//! and full detection-to-recognition composition with end-to-end bounds.
//!
//! ## Tests (14 tests)
//!
//! **Recognition encoder (tests 1-3):**
//! 1. ResNet backbone feature extraction for recognition (IBP)
//! 2. Encoder with LayerNorm normalization (IBP + CROWN)
//! 3. Multi-scale feature fusion via FPN for recognition (IBP)
//!
//! **CTC decoder (tests 4-5):**
//! 4. CTC linear decoder: hidden -> vocab logits -> softmax (IBP)
//! 5. CTC decoder with FFN scoring: hidden -> FFN -> softmax (IBP + CROWN)
//!
//! **Attention decoder (tests 6-7):**
//! 6. Cross-attention decoder: query attends to encoder features (IBP)
//! 7. Attention decoder with causal mask and residual (IBP + CROWN)
//!
//! **Variable-length & embeddings (tests 8-9):**
//! 8. Variable-length recognition with padding handling (IBP)
//! 9. Character embedding output bounds (IBP)
//!
//! **Beam search & confidence (tests 10-11):**
//! 10. Beam search top-k probability bounds (IBP)
//! 11. Recognition confidence scoring: encoder -> FFN -> sigmoid (IBP)
//!
//! **End-to-end pipeline (tests 12-14):**
//! 12. Detection backbone -> recognition encoder composition (IBP)
//! 13. Full recognition pipeline: image patch -> encoder -> CTC decoder (IBP)
//! 14. Full recognition pipeline with CROWN tight bounds (IBP + CROWN)
//!
//! Architecture references:
//! - PaddleOCR (Baidu): Production OCR with DB detector + SVTR recognizer
//! - PP-OCRv4: Latest PaddleOCR version with ResNet backbone
//! - SVTR (Du et al. 2022): Scene Text Recognition with a Single Visual Model
//! - CTC (Graves et al. 2006): Connectionist Temporal Classification
//! - Attention decoder (Bahdanau et al. 2015): Sequence-to-sequence decoding
//!
//! Dimensions (small for fast verification, structurally representative):
//! - IMG=16, BACKBONE_CH=8, FPN_CH=16, MID_CH=16, HIDDEN=32
//! - VOCAB=64, SEQ=8, FFN_DIM=64, NUM_HEADS=4, HEAD_DIM=8
//!
//! Part of #4222: NY compose tests for PaddleOCR recognition pipeline.

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

const IMG: usize = 16;
const IN_CH: usize = 3;
const BACKBONE_CH: usize = 8;
const FPN_CH: usize = 16;
const MID_CH: usize = 16;
const HIDDEN: usize = 32;
const FFN_DIM: usize = 64;
const SEQ: usize = 8;
const NUM_HEADS: usize = 4;
const VOCAB: usize = 64;
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

fn image_bounds(shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), 0.0f32),
        ArrayD::from_elem(IxDyn(shape), 1.0f32),
    )
    .expect("valid image bounds [0, 1]")
}

fn push_conv_bn_bindings(
    bindings: &mut Vec<TensorParamBinding>,
    out_ch: usize,
    in_ch: usize,
    kernel: usize,
) {
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        out_ch, in_ch, kernel, kernel,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[out_ch])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[out_ch])));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[out_ch])));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[out_ch])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[out_ch])));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
}

fn add_conv_bn_relu(
    b: &mut TensorBlockBuilder,
    x: nn_dsl::tensor_ir::TensorNodeId,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    out_h: usize,
    out_w: usize,
) -> nn_dsl::tensor_ir::TensorNodeId {
    let out_shape = [out_ch, out_h, out_w];
    let conv_w = b.add_input(
        &format!("{prefix}_conv_w"),
        &[out_ch, in_ch, kernel, kernel],
    );
    let conv_b = b.add_input(&format!("{prefix}_conv_b"), &[out_ch]);
    let conv = b.add_conv2d(
        x,
        conv_w,
        Some(conv_b),
        stride,
        stride,
        padding,
        padding,
        &out_shape,
    );

    let bn_mean = b.add_input(&format!("{prefix}_bn_mean"), &[out_ch]);
    let bn_var = b.add_input(&format!("{prefix}_bn_var"), &[out_ch]);
    let bn_w = b.add_input(&format!("{prefix}_bn_w"), &[out_ch]);
    let bn_b = b.add_input(&format!("{prefix}_bn_b"), &[out_ch]);
    let eps = b.add_input(&format!("{prefix}_eps"), &[1]);
    let bn = b.add_batch_norm(conv, bn_mean, bn_var, bn_w, bn_b, eps, &out_shape);
    b.add_relu(bn, &out_shape)
}

// ===========================================================================
// 1. ResNet backbone feature extraction for recognition (IBP)
// ===========================================================================

#[test]
fn test_paddle_recog_resnet_encoder_ibp() {
    let spatial = IMG / 2;
    let mut b = TensorBlockBuilder::new("ppocr_recog_resnet_encoder");
    let input = b.add_input("image", &[IN_CH, IMG, IMG]);

    // Stage 1: Conv-BN-ReLU stride-2 downsample
    let s1 = add_conv_bn_relu(
        &mut b,
        input,
        "s1",
        IN_CH,
        BACKBONE_CH,
        3,
        2,
        1,
        spatial,
        spatial,
    );

    // Stage 2: Conv-BN-ReLU maintain spatial
    let s2 = add_conv_bn_relu(
        &mut b,
        s1,
        "s2",
        BACKBONE_CH,
        BACKBONE_CH,
        3,
        1,
        1,
        spatial,
        spatial,
    );

    // Stage 3: Conv-BN-ReLU widen channels for recognition
    let s3 = add_conv_bn_relu(
        &mut b,
        s2,
        "s3",
        BACKBONE_CH,
        MID_CH,
        3,
        1,
        1,
        spatial,
        spatial,
    );

    let def = b.build(s3).expect("valid recognition ResNet encoder");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_bindings(&mut bindings, BACKBONE_CH, IN_CH, 3);
    push_conv_bn_bindings(&mut bindings, BACKBONE_CH, BACKBONE_CH, 3);
    push_conv_bn_bindings(&mut bindings, MID_CH, BACKBONE_CH, 3);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(&[IN_CH, IMG, IMG]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through recognition ResNet encoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[MID_CH, spatial, spatial],
        "encoder output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR recognition ResNet encoder IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "ReLU lower >= 0, got {lo_min}");
}

// ===========================================================================
// 2. Encoder with LayerNorm normalization (IBP + CROWN)
// ===========================================================================

#[test]
fn test_paddle_recog_encoder_layernorm_ibp() {
    let in_shape = [SEQ, HIDDEN];
    let mut b = TensorBlockBuilder::new("ppocr_recog_encoder_ln");
    let input = b.add_input("encoder_out", &in_shape);

    // LayerNorm on encoder features
    let ln_w = b.add_input("ln_w", &[HIDDEN]);
    let ln_b = b.add_input("ln_b", &[HIDDEN]);
    let eps = b.add_input("ln_eps", &[1]);
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &in_shape);

    // Linear projection after normalization
    let proj_w = b.add_input("proj_w", &[HIDDEN, HIDDEN]);
    let proj_b = b.add_input("proj_b", &[HIDDEN]);
    let out = b.add_linear(normed, proj_w, Some(proj_b), &in_shape);

    let def = b.build(out).expect("valid encoder LayerNorm");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN])),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN, HIDDEN])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through encoder LayerNorm");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, HIDDEN]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR encoder LayerNorm IBP: bounds=[{lo_min}, {hi_max}]");
}

#[test]
fn test_paddle_recog_encoder_layernorm_crown() {
    let in_shape = [SEQ, HIDDEN];
    let mut b = TensorBlockBuilder::new("ppocr_recog_encoder_ln_crown");
    let input = b.add_input("encoder_out", &in_shape);

    let ln_w = b.add_input("ln_w", &[HIDDEN]);
    let ln_b = b.add_input("ln_b", &[HIDDEN]);
    let eps = b.add_input("ln_eps", &[1]);
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &in_shape);

    let proj_w = b.add_input("proj_w", &[HIDDEN, HIDDEN]);
    let proj_b = b.add_input("proj_b", &[HIDDEN]);
    let out = b.add_linear(normed, proj_w, Some(proj_b), &in_shape);

    let def = b.build(out).expect("valid encoder LayerNorm CROWN");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN])),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN, HIDDEN])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR encoder LayerNorm CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 3. Multi-scale feature fusion via FPN for recognition (IBP)
// ===========================================================================

#[test]
fn test_paddle_recog_multiscale_fusion_ibp() {
    let spatial = IMG / 2;
    let in_shape = [BACKBONE_CH, spatial, spatial];
    let branch_shape = [FPN_CH, spatial, spatial];
    let concat_ch = FPN_CH * 2;
    let concat_shape = [concat_ch, spatial, spatial];
    let out_shape = [FPN_CH, spatial, spatial];

    let mut b = TensorBlockBuilder::new("ppocr_recog_multiscale_fpn");
    let input = b.add_input("features", &in_shape);

    // Branch 1: 1x1 conv for channel projection
    let w1 = b.add_input("br1_w", &[FPN_CH, BACKBONE_CH, 1, 1]);
    let b1 = b.add_input("br1_b", &[FPN_CH]);
    let br1 = b.add_conv2d(input, w1, Some(b1), 1, 1, 0, 0, &branch_shape);
    let br1_relu = b.add_relu(br1, &branch_shape);

    // Branch 2: 3x3 conv for spatial context
    let w2 = b.add_input("br2_w", &[FPN_CH, BACKBONE_CH, 3, 3]);
    let b2 = b.add_input("br2_b", &[FPN_CH]);
    let br2 = b.add_conv2d(input, w2, Some(b2), 1, 1, 1, 1, &branch_shape);
    let br2_relu = b.add_relu(br2, &branch_shape);

    // Concat + merge
    let fused = b.add_concat(&[br1_relu, br2_relu], 0, &concat_shape);
    let wm = b.add_input("merge_w", &[FPN_CH, concat_ch, 1, 1]);
    let bm = b.add_input("merge_b", &[FPN_CH]);
    let merged = b.add_conv2d(fused, wm, Some(bm), 1, 1, 0, 0, &out_shape);
    let out = b.add_relu(merged, &out_shape);

    let def = b.build(out).expect("valid recognition FPN");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[FPN_CH, BACKBONE_CH, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[FPN_CH])),
        TensorParamBinding::ConstantTensor(w(&[FPN_CH, BACKBONE_CH, 3, 3])),
        TensorParamBinding::ConstantTensor(zeros(&[FPN_CH])),
        TensorParamBinding::ConstantTensor(w(&[FPN_CH, concat_ch, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[FPN_CH])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[BACKBONE_CH, spatial, spatial], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through recognition FPN");

    assert_eq!(output.lower_upper().0.shape(), &[FPN_CH, spatial, spatial]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR recognition FPN IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "ReLU lower >= 0, got {lo_min}");
}

// ===========================================================================
// 4. CTC linear decoder: hidden -> vocab logits -> softmax (IBP)
// ===========================================================================

#[test]
fn test_paddle_recog_ctc_linear_decoder_ibp() {
    let in_shape = [SEQ, HIDDEN];
    let logit_shape = [SEQ, VOCAB];

    let mut b = TensorBlockBuilder::new("ppocr_recog_ctc_linear");
    let input = b.add_input("encoder_out", &in_shape);

    let wl = b.add_input("ctc_w", &[VOCAB, HIDDEN]);
    let bl = b.add_input("ctc_b", &[VOCAB]);
    let logits = b.add_linear(input, wl, Some(bl), &logit_shape);
    let out = b.add_softmax(logits, -1, &logit_shape);

    let def = b.build(out).expect("valid CTC linear decoder");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[VOCAB, HIDDEN])),
        TensorParamBinding::ConstantTensor(zeros(&[VOCAB])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through CTC linear decoder");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ, VOCAB]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR CTC linear decoder IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 5. CTC decoder with FFN scoring: hidden -> FFN -> softmax (IBP + CROWN)
// ===========================================================================

#[test]
fn test_paddle_recog_ctc_ffn_decoder_ibp() {
    let in_shape = [SEQ, HIDDEN];
    let ffn_shape = [SEQ, FFN_DIM];
    let logit_shape = [SEQ, VOCAB];

    let mut b = TensorBlockBuilder::new("ppocr_recog_ctc_ffn");
    let input = b.add_input("encoder_out", &in_shape);

    // FFN scoring: Linear -> ReLU -> Linear -> softmax
    let w1 = b.add_input("ffn_w1", &[FFN_DIM, HIDDEN]);
    let b1 = b.add_input("ffn_b1", &[FFN_DIM]);
    let h = b.add_linear(input, w1, Some(b1), &ffn_shape);
    let h_relu = b.add_relu(h, &ffn_shape);

    let w2 = b.add_input("ffn_w2", &[VOCAB, FFN_DIM]);
    let b2 = b.add_input("ffn_b2", &[VOCAB]);
    let logits = b.add_linear(h_relu, w2, Some(b2), &logit_shape);
    let out = b.add_softmax(logits, -1, &logit_shape);

    let def = b.build(out).expect("valid CTC FFN decoder");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, HIDDEN])),
        TensorParamBinding::ConstantTensor(zeros(&[FFN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[VOCAB, FFN_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[VOCAB])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    // IBP
    let ibp_out = graph
        .propagate_ibp(&input)
        .expect("IBP through CTC FFN decoder");
    assert_eq!(ibp_out.lower_upper().0.shape(), &[SEQ, VOCAB]);
    assert_bounds_valid(&ibp_out);

    let (lo_min, hi_max) = bounds_min_max(&ibp_out);
    eprintln!("PaddleOCR CTC FFN decoder IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "softmax upper <= 1, got {hi_max}");
}

#[test]
fn test_paddle_recog_ctc_ffn_decoder_crown() {
    let in_shape = [SEQ, HIDDEN];
    let ffn_shape = [SEQ, FFN_DIM];
    let logit_shape = [SEQ, VOCAB];

    let mut b = TensorBlockBuilder::new("ppocr_recog_ctc_ffn_crown");
    let input = b.add_input("encoder_out", &in_shape);

    let w1 = b.add_input("ffn_w1", &[FFN_DIM, HIDDEN]);
    let b1 = b.add_input("ffn_b1", &[FFN_DIM]);
    let h = b.add_linear(input, w1, Some(b1), &ffn_shape);
    let h_relu = b.add_relu(h, &ffn_shape);

    let w2 = b.add_input("ffn_w2", &[VOCAB, FFN_DIM]);
    let b2 = b.add_input("ffn_b2", &[VOCAB]);
    let logits = b.add_linear(h_relu, w2, Some(b2), &logit_shape);
    let out = b.add_softmax(logits, -1, &logit_shape);

    let def = b.build(out).expect("valid CTC FFN decoder CROWN");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, HIDDEN])),
        TensorParamBinding::ConstantTensor(zeros(&[FFN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[VOCAB, FFN_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[VOCAB])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR CTC FFN decoder CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    assert!(lo_min >= -1e-6, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 6. Cross-attention decoder: query attends to encoder features (IBP)
// ===========================================================================

#[test]
fn test_paddle_recog_cross_attention_decoder_ibp() {
    let enc_shape = [SEQ, HIDDEN];
    let dec_shape = [SEQ, HIDDEN];
    let scale = 1.0 / ((HIDDEN / NUM_HEADS) as f32).sqrt();

    let mut b = TensorBlockBuilder::new("ppocr_recog_cross_attn_dec");
    let enc_features = b.add_input("encoder_out", &enc_shape);
    let dec_input = b.add_input("decoder_input", &dec_shape);

    // Q from decoder, K/V from encoder
    let wq = b.add_input("wq", &[HIDDEN, HIDDEN]);
    let wk = b.add_input("wk", &[HIDDEN, HIDDEN]);
    let wv = b.add_input("wv", &[HIDDEN, HIDDEN]);
    let wo = b.add_input("wo", &[HIDDEN, HIDDEN]);

    let q = b.add_linear(dec_input, wq, None, &dec_shape);
    let k = b.add_linear(enc_features, wk, None, &enc_shape);
    let v = b.add_linear(enc_features, wv, None, &enc_shape);

    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &dec_shape);
    let out = b.add_linear(attn, wo, None, &dec_shape);

    let def = b.build(out).expect("valid cross-attention decoder");

    let bindings = vec![
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[SEQ, HIDDEN]), 0.5f32)), // encoder features (constant)
        TensorParamBinding::Variable, // decoder input
        TensorParamBinding::ConstantTensor(w(&[HIDDEN, HIDDEN])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN, HIDDEN])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN, HIDDEN])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN, HIDDEN])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cross-attention decoder");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ, HIDDEN]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR cross-attention decoder IBP: bounds=[{lo_min}, {hi_max}]");
}

// ===========================================================================
// 7. Attention decoder with causal mask and residual (IBP + CROWN)
// ===========================================================================

#[test]
fn test_paddle_recog_causal_attention_residual_ibp() {
    let shape = [SEQ, HIDDEN];
    let scale = 1.0 / ((HIDDEN / NUM_HEADS) as f32).sqrt();

    let mut b = TensorBlockBuilder::new("ppocr_recog_causal_attn_res");
    let input = b.add_input("decoder_input", &shape);

    // Self-attention with causal mask
    let wq = b.add_input("wq", &[HIDDEN, HIDDEN]);
    let wk = b.add_input("wk", &[HIDDEN, HIDDEN]);
    let wv = b.add_input("wv", &[HIDDEN, HIDDEN]);
    let wo = b.add_input("wo", &[HIDDEN, HIDDEN]);

    let q = b.add_linear(input, wq, None, &shape);
    let k = b.add_linear(input, wk, None, &shape);
    let v = b.add_linear(input, wv, None, &shape);

    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
    let proj = b.add_linear(attn, wo, None, &shape);

    // Residual connection
    let out = b.add_binary_add(input, proj, &shape);

    let def = b.build(out).expect("valid causal attention + residual");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[HIDDEN, HIDDEN])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN, HIDDEN])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN, HIDDEN])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN, HIDDEN])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    // IBP
    let ibp_out = graph
        .propagate_ibp(&input)
        .expect("IBP through causal attention + residual");
    assert_eq!(ibp_out.lower_upper().0.shape(), &[SEQ, HIDDEN]);
    assert_bounds_valid(&ibp_out);

    let (lo_min, hi_max) = bounds_min_max(&ibp_out);
    eprintln!("PaddleOCR causal attention residual IBP: bounds=[{lo_min}, {hi_max}]");
}

#[test]
fn test_paddle_recog_causal_attention_residual_crown() {
    let shape = [SEQ, HIDDEN];
    let scale = 1.0 / ((HIDDEN / NUM_HEADS) as f32).sqrt();

    let mut b = TensorBlockBuilder::new("ppocr_recog_causal_attn_res_c");
    let input = b.add_input("decoder_input", &shape);

    let wq = b.add_input("wq", &[HIDDEN, HIDDEN]);
    let wk = b.add_input("wk", &[HIDDEN, HIDDEN]);
    let wv = b.add_input("wv", &[HIDDEN, HIDDEN]);
    let wo = b.add_input("wo", &[HIDDEN, HIDDEN]);

    let q = b.add_linear(input, wq, None, &shape);
    let k = b.add_linear(input, wk, None, &shape);
    let v = b.add_linear(input, wv, None, &shape);

    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
    let proj = b.add_linear(attn, wo, None, &shape);
    let out = b.add_binary_add(input, proj, &shape);

    let def = b
        .build(out)
        .expect("valid causal attention + residual CROWN");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[HIDDEN, HIDDEN])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN, HIDDEN])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN, HIDDEN])),
        TensorParamBinding::ConstantTensor(w(&[HIDDEN, HIDDEN])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "PaddleOCR causal attention residual CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 8. Variable-length recognition with padding handling (IBP)
// ===========================================================================

#[test]
fn test_paddle_recog_variable_length_padding_ibp() {
    // Models variable-length input via longer sequence with padding mask effect.
    // Uses LayerNorm which normalizes across the hidden dim, handling padded positions.
    let long_seq = SEQ * 2; // padded sequence
    let in_shape = [long_seq, HIDDEN];
    let logit_shape = [long_seq, VOCAB];

    let mut b = TensorBlockBuilder::new("ppocr_recog_varlen_pad");
    let input = b.add_input("padded_seq", &in_shape);

    // LayerNorm handles variable length by normalizing per-position
    let ln_w = b.add_input("ln_w", &[HIDDEN]);
    let ln_b = b.add_input("ln_b", &[HIDDEN]);
    let eps = b.add_input("ln_eps", &[1]);
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &in_shape);

    // CTC projection
    let wl = b.add_input("ctc_w", &[VOCAB, HIDDEN]);
    let bl = b.add_input("ctc_b", &[VOCAB]);
    let logits = b.add_linear(normed, wl, Some(bl), &logit_shape);
    let out = b.add_softmax(logits, -1, &logit_shape);

    let def = b.build(out).expect("valid variable-length recognition");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN])),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(w(&[VOCAB, HIDDEN])),
        TensorParamBinding::ConstantTensor(zeros(&[VOCAB])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[long_seq, HIDDEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through variable-length recognition");

    assert_eq!(output.lower_upper().0.shape(), &[long_seq, VOCAB]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR variable-length recognition IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 9. Character embedding output bounds (IBP)
// ===========================================================================

#[test]
fn test_paddle_recog_character_embedding_ibp() {
    // Character embedding modeled as Linear from one-hot-like input space.
    let embed_dim = HIDDEN;
    let in_shape = [SEQ, VOCAB];
    let out_shape = [SEQ, embed_dim];

    let mut b = TensorBlockBuilder::new("ppocr_recog_char_embed");
    let input = b.add_input("char_ids", &in_shape);

    let embed_w = b.add_input("embed_w", &[embed_dim, VOCAB]);
    let embed_b = b.add_input("embed_b", &[embed_dim]);
    let out = b.add_linear(input, embed_w, Some(embed_b), &out_shape);

    let def = b.build(out).expect("valid character embedding");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[embed_dim, VOCAB])),
        TensorParamBinding::ConstantTensor(zeros(&[embed_dim])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, VOCAB], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through character embedding");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ, embed_dim]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR character embedding IBP: bounds=[{lo_min}, {hi_max}]");
}

// ===========================================================================
// 10. Beam search top-k probability bounds (IBP)
// ===========================================================================

#[test]
fn test_paddle_recog_beam_search_topk_ibp() {
    // Beam search scoring: hidden -> FFN -> softmax probabilities.
    // Top-k candidates are scored; all outputs remain in [0, 1].
    let in_shape = [SEQ, HIDDEN];
    let ffn_shape = [SEQ, FFN_DIM];
    let logit_shape = [SEQ, VOCAB];

    let mut b = TensorBlockBuilder::new("ppocr_recog_beam_topk");
    let input = b.add_input("hidden", &in_shape);

    let w1 = b.add_input("beam_w1", &[FFN_DIM, HIDDEN]);
    let b1 = b.add_input("beam_b1", &[FFN_DIM]);
    let h = b.add_linear(input, w1, Some(b1), &ffn_shape);
    let h_relu = b.add_relu(h, &ffn_shape);

    let w2 = b.add_input("beam_w2", &[VOCAB, FFN_DIM]);
    let b2 = b.add_input("beam_b2", &[VOCAB]);
    let logits = b.add_linear(h_relu, w2, Some(b2), &logit_shape);
    let out = b.add_softmax(logits, -1, &logit_shape);

    let def = b.build(out).expect("valid beam search top-k");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, HIDDEN])),
        TensorParamBinding::ConstantTensor(zeros(&[FFN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[VOCAB, FFN_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[VOCAB])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through beam search top-k");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ, VOCAB]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR beam search top-k IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 11. Recognition confidence scoring: encoder -> FFN -> sigmoid (IBP)
// ===========================================================================

#[test]
fn test_paddle_recog_confidence_scoring_ibp() {
    // Confidence head: encoder features -> Linear -> sigmoid -> [0, 1] score.
    let in_shape = [SEQ, HIDDEN];
    let score_shape = [SEQ, 1];

    let mut b = TensorBlockBuilder::new("ppocr_recog_confidence");
    let input = b.add_input("encoder_out", &in_shape);

    let ws = b.add_input("conf_w", &[1, HIDDEN]);
    let bs = b.add_input("conf_b", &[1]);
    let logit = b.add_linear(input, ws, Some(bs), &score_shape);
    let out = b.add_sigmoid(logit, &score_shape);

    let def = b.build(out).expect("valid confidence scoring");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[1, HIDDEN])),
        TensorParamBinding::ConstantTensor(zeros(&[1])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through confidence scoring");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ, 1]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR confidence scoring IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 12. Detection backbone -> recognition encoder composition (IBP)
// ===========================================================================

#[test]
fn test_paddle_recog_detect_to_encoder_ibp() {
    let spatial = IMG / 2;

    let mut b = TensorBlockBuilder::new("ppocr_recog_detect_to_encoder");
    let input = b.add_input("image", &[IN_CH, IMG, IMG]);

    // Detection backbone: Conv-BN-ReLU stride-2
    let backbone = add_conv_bn_relu(
        &mut b,
        input,
        "bb",
        IN_CH,
        BACKBONE_CH,
        3,
        2,
        1,
        spatial,
        spatial,
    );

    // Recognition encoder: 1x1 conv channel projection + ReLU
    let rw = b.add_input("recog_w", &[MID_CH, BACKBONE_CH, 1, 1]);
    let rb = b.add_input("recog_b", &[MID_CH]);
    let recog_shape = [MID_CH, spatial, spatial];
    let recog = b.add_conv2d(backbone, rw, Some(rb), 1, 1, 0, 0, &recog_shape);
    let out = b.add_relu(recog, &recog_shape);

    let def = b.build(out).expect("valid detect-to-encoder composition");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_bindings(&mut bindings, BACKBONE_CH, IN_CH, 3);
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        MID_CH,
        BACKBONE_CH,
        1,
        1,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[MID_CH])));

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(&[IN_CH, IMG, IMG]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through detect-to-encoder composition");

    assert_eq!(output.lower_upper().0.shape(), &[MID_CH, spatial, spatial]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR detect-to-encoder IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "ReLU lower >= 0, got {lo_min}");
}

// ===========================================================================
// 13. Full recognition pipeline: image -> encoder -> CTC decoder (IBP)
// ===========================================================================

#[test]
fn test_paddle_recog_full_pipeline_ibp() {
    let spatial = IMG / 2;

    let mut b = TensorBlockBuilder::new("ppocr_recog_full_pipeline");
    let input = b.add_input("image", &[IN_CH, IMG, IMG]);

    // Stage 1: ResNet backbone Conv-BN-ReLU
    let backbone = add_conv_bn_relu(
        &mut b,
        input,
        "bb",
        IN_CH,
        BACKBONE_CH,
        3,
        2,
        1,
        spatial,
        spatial,
    );

    // Stage 2: Recognition 1x1 conv to vocab-sized channel map
    let recog_conv_shape = [VOCAB, spatial, spatial];
    let rw = b.add_input("recog_w", &[VOCAB, BACKBONE_CH, 1, 1]);
    let rb = b.add_input("recog_b", &[VOCAB]);
    let recog_conv = b.add_conv2d(backbone, rw, Some(rb), 1, 1, 0, 0, &recog_conv_shape);

    // Stage 3: Softmax over vocabulary (channel) dimension
    let out = b.add_softmax(recog_conv, 0, &recog_conv_shape);

    let def = b.build(out).expect("valid full recognition pipeline");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_bindings(&mut bindings, BACKBONE_CH, IN_CH, 3);
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        VOCAB,
        BACKBONE_CH,
        1,
        1,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[VOCAB])));

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(&[IN_CH, IMG, IMG]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full recognition pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[VOCAB, spatial, spatial]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR full recognition pipeline IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 14. Full recognition pipeline with CROWN tight bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_paddle_recog_full_pipeline_crown() {
    let spatial = IMG / 2;

    let mut b = TensorBlockBuilder::new("ppocr_recog_full_pipe_crown");
    let input = b.add_input("image", &[IN_CH, IMG, IMG]);

    // Stage 1: ResNet backbone Conv-BN-ReLU
    let backbone = add_conv_bn_relu(
        &mut b,
        input,
        "bb",
        IN_CH,
        BACKBONE_CH,
        3,
        2,
        1,
        spatial,
        spatial,
    );

    // Stage 2: Recognition 1x1 conv to vocab-sized channel map
    let recog_conv_shape = [VOCAB, spatial, spatial];
    let rw = b.add_input("recog_w", &[VOCAB, BACKBONE_CH, 1, 1]);
    let rb = b.add_input("recog_b", &[VOCAB]);
    let recog_conv = b.add_conv2d(backbone, rw, Some(rb), 1, 1, 0, 0, &recog_conv_shape);

    // Stage 3: Softmax over vocabulary (channel) dimension
    let out = b.add_softmax(recog_conv, 0, &recog_conv_shape);

    let def = b.build(out).expect("valid full recognition pipeline CROWN");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_bindings(&mut bindings, BACKBONE_CH, IN_CH, 3);
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        VOCAB,
        BACKBONE_CH,
        1,
        1,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[VOCAB])));

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(&[IN_CH, IMG, IMG]);

    // IBP baseline
    let ibp_out = graph
        .propagate_ibp(&input)
        .expect("IBP through full recognition pipeline");
    assert_bounds_valid(&ibp_out);

    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_out);
    eprintln!("PaddleOCR full recognition pipeline IBP: bounds=[{ibp_lo}, {ibp_hi}]");

    // CROWN pass
    let (method, crown_out, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&crown_out);
    eprintln!(
        "PaddleOCR full recognition pipeline CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    assert!(lo_min >= -1e-6, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "softmax upper <= 1, got {hi_max}");
}
