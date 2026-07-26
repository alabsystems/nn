// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests for PaddleOCR-VL text detection subpipeline bounds.
//!
//! Verifies IBP and CROWN bound propagation through the PaddleOCR-VL text
//! detection and recognition subpipeline, focusing on PP-OCRv4 detection
//! backbone and SVTR recognition head.
//!
//! ## Tests (14 tests)
//!
//! **Detection backbone (tests 1-2):**
//! 1. PP-OCRv4 ResNet backbone feature extraction bounds (IBP)
//! 2. DB (Differentiable Binarization) head output bounds [0,1] (IBP + CROWN)
//!
//! **Recognition encoder (tests 3-4):**
//! 3. SVTR recognition encoder self-attention bounds (IBP + CROWN)
//! 4. CTC decoder output probability bounds (IBP)
//!
//! **Detection outputs (tests 5-7):**
//! 5. Text box regression coordinate bounds (IBP)
//! 6. NMS (Non-Maximum Suppression) score filtering (IBP)
//! 7. Feature pyramid network multi-scale bounds (IBP + CROWN)
//!
//! **Recognition preprocessing (tests 8-9):**
//! 8. Recognition input normalization bounds (IBP)
//! 9. Character vocabulary probability distribution (IBP)
//!
//! **Throughput & auxiliary (tests 10-12):**
//! 10. Batch text detection throughput bounds (IBP)
//! 11. Text orientation classifier bounds (IBP)
//! 12. Text line grouping geometry bounds (IBP)
//!
//! **End-to-end (tests 13-14):**
//! 13. Recognition beam search bounds (IBP)
//! 14. Full detection-to-recognition pipeline (IBP)
//!
//! Architecture references:
//! - PaddleOCR (Baidu): Production OCR with DB detector + SVTR recognizer
//! - PP-OCRv4: Latest PaddleOCR version with ResNet backbone
//! - DB (Liao et al. 2020): Differentiable Binarization for text detection
//! - SVTR (Du et al. 2022): Scene Text Recognition with a Single Visual Model
//! - CTC (Graves et al. 2006): Connectionist Temporal Classification
//!
//! Dimensions (small for fast verification, structurally representative):
//! - IMG=16, BACKBONE_CH=8, FPN_CH=16, MID_CH=16, HIDDEN=32, VOCAB=64, SEQ=8
//!
//! Part of #4222: NY compose tests for PaddleOCR-VL text detection.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
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
const MAP_CH: usize = 1;
const HIDDEN: usize = 32;
const FFN_DIM: usize = 64;
const SEQ: usize = 8;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = HIDDEN / NUM_HEADS; // 8
const VOCAB: usize = 64;
const W_MAG: f32 = 0.02;
const NUM_CLASSES: usize = 2; // text / non-text

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
// 1. PP-OCRv4 ResNet backbone feature extraction bounds (IBP)
// ===========================================================================

fn build_ppocr_resnet_backbone() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("ppocr_resnet_backbone");
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
        IMG / 2,
        IMG / 2,
    );

    // Stage 2: Conv-BN-ReLU same spatial
    let s2 = add_conv_bn_relu(
        &mut b,
        s1,
        "s2",
        BACKBONE_CH,
        BACKBONE_CH,
        3,
        1,
        1,
        IMG / 2,
        IMG / 2,
    );

    b.build(s2).expect("valid PP-OCRv4 ResNet backbone")
}

fn ppocr_resnet_backbone_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_bindings(&mut bindings, BACKBONE_CH, IN_CH, 3);
    push_conv_bn_bindings(&mut bindings, BACKBONE_CH, BACKBONE_CH, 3);
    bindings
}

#[test]
fn test_paddle_detect_resnet_backbone_ibp() {
    let def = build_ppocr_resnet_backbone();
    let bindings = ppocr_resnet_backbone_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(&[IN_CH, IMG, IMG]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PP-OCRv4 ResNet backbone");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[BACKBONE_CH, IMG / 2, IMG / 2],
        "backbone output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PP-OCRv4 ResNet backbone IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "ReLU lower >= 0, got {lo_min}");
}

// ===========================================================================
// 2. DB head output bounds [0,1] (IBP + CROWN)
// ===========================================================================

fn build_db_detection_head() -> TensorKernelDef {
    let spatial = IMG / 2;
    let mid_shape = [MID_CH, spatial, spatial];
    let out_shape = [MAP_CH, spatial, spatial];

    let mut b = TensorBlockBuilder::new("ppocr_db_head");
    let input = b.add_input("features", &[BACKBONE_CH, spatial, spatial]);

    // Conv -> ReLU -> Conv -> sigmoid
    let w1 = b.add_input("head_w1", &[MID_CH, BACKBONE_CH, 3, 3]);
    let b1 = b.add_input("head_b1", &[MID_CH]);
    let conv1 = b.add_conv2d(input, w1, Some(b1), 1, 1, 1, 1, &mid_shape);
    let relu = b.add_relu(conv1, &mid_shape);

    let w2 = b.add_input("head_w2", &[MAP_CH, MID_CH, 1, 1]);
    let b2 = b.add_input("head_b2", &[MAP_CH]);
    let conv2 = b.add_conv2d(relu, w2, Some(b2), 1, 1, 0, 0, &out_shape);

    let out = b.add_sigmoid(conv2, &out_shape);
    b.build(out).expect("valid DB detection head")
}

fn db_detection_head_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[MID_CH, BACKBONE_CH, 3, 3])),
        TensorParamBinding::ConstantTensor(zeros(&[MID_CH])),
        TensorParamBinding::ConstantTensor(w(&[MAP_CH, MID_CH, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[MAP_CH])),
    ]
}

#[test]
fn test_paddle_detect_db_head_ibp() {
    let def = build_db_detection_head();
    let bindings = db_detection_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let spatial = IMG / 2;
    let input = uniform_bounds(&[BACKBONE_CH, spatial, spatial], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DB detection head");

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DB detection head IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1, got {hi_max}");
}

#[test]
fn test_paddle_detect_db_head_crown() {
    let def = build_db_detection_head();
    let bindings = db_detection_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let spatial = IMG / 2;
    let input = uniform_bounds(&[BACKBONE_CH, spatial, spatial], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DB detection head CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    assert!(lo_min >= -1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 3. SVTR recognition encoder self-attention bounds (IBP + CROWN)
// ===========================================================================

fn build_svtr_self_attention() -> TensorKernelDef {
    let in_shape = [SEQ, HIDDEN];
    let out_shape = [SEQ, HIDDEN];

    let mut b = TensorBlockBuilder::new("ppocr_svtr_attention");
    let input = b.add_input("seq", &in_shape);

    // Q, K, V linear projections
    let wq = b.add_input("wq", &[HIDDEN, HIDDEN]);
    let bq = b.add_input("bq", &[HIDDEN]);
    let q = b.add_linear(input, wq, Some(bq), &out_shape);

    let wk = b.add_input("wk", &[HIDDEN, HIDDEN]);
    let bk = b.add_input("bk", &[HIDDEN]);
    let k = b.add_linear(input, wk, Some(bk), &out_shape);

    let wv = b.add_input("wv", &[HIDDEN, HIDDEN]);
    let bv = b.add_input("bv", &[HIDDEN]);
    let v = b.add_linear(input, wv, Some(bv), &out_shape);

    // Self-attention: softmax(QK^T/sqrt(d)) * V
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, None, &out_shape);

    // Output projection
    let wo = b.add_input("wo", &[HIDDEN, HIDDEN]);
    let bo = b.add_input("bo", &[HIDDEN]);
    let proj = b.add_linear(attn, wo, Some(bo), &out_shape);

    // Residual
    let out = b.add_binary_add(input, proj, &out_shape);
    b.build(out).expect("valid SVTR self-attention")
}

fn svtr_self_attention_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    // Q, K, V projections (each: weight + bias)
    for _ in 0..4 {
        bindings.push(TensorParamBinding::ConstantTensor(w(&[HIDDEN, HIDDEN])));
        bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN])));
    }
    bindings
}

#[test]
fn test_paddle_detect_svtr_attention_ibp() {
    let def = build_svtr_self_attention();
    let bindings = svtr_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through SVTR self-attention");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ, HIDDEN]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SVTR self-attention IBP: bounds=[{lo_min}, {hi_max}]");
}

#[test]
fn test_paddle_detect_svtr_attention_crown() {
    let def = build_svtr_self_attention();
    let bindings = svtr_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SVTR self-attention CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 4. CTC decoder output probability bounds (IBP)
// ===========================================================================

fn build_ctc_decoder_output() -> TensorKernelDef {
    let in_shape = [SEQ, HIDDEN];
    let logit_shape = [SEQ, VOCAB];

    let mut b = TensorBlockBuilder::new("ppocr_ctc_decoder");
    let input = b.add_input("encoder_out", &in_shape);

    // Linear projection to vocabulary
    let wl = b.add_input("ctc_w", &[VOCAB, HIDDEN]);
    let bl = b.add_input("ctc_b", &[VOCAB]);
    let logits = b.add_linear(input, wl, Some(bl), &logit_shape);

    // Softmax over vocabulary dimension -> probabilities in [0, 1]
    let out = b.add_softmax(logits, -1, &logit_shape);
    b.build(out).expect("valid CTC decoder output")
}

fn ctc_decoder_output_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[VOCAB, HIDDEN])),
        TensorParamBinding::ConstantTensor(zeros(&[VOCAB])),
    ]
}

#[test]
fn test_paddle_detect_ctc_decoder_ibp() {
    let def = build_ctc_decoder_output();
    let bindings = ctc_decoder_output_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through CTC decoder");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ, VOCAB]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("CTC decoder IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 5. Text box regression coordinate bounds (IBP)
// ===========================================================================

fn build_box_regression() -> TensorKernelDef {
    let spatial = IMG / 2;
    let in_shape = [BACKBONE_CH, spatial, spatial];
    let box_ch = 4; // x, y, w, h
    let mid_shape = [MID_CH, spatial, spatial];
    let out_shape = [box_ch, spatial, spatial];

    let mut b = TensorBlockBuilder::new("ppocr_box_regression");
    let input = b.add_input("features", &in_shape);

    // Conv -> ReLU -> Conv -> sigmoid (normalized coords in [0,1])
    let w1 = b.add_input("box_w1", &[MID_CH, BACKBONE_CH, 3, 3]);
    let b1 = b.add_input("box_b1", &[MID_CH]);
    let conv1 = b.add_conv2d(input, w1, Some(b1), 1, 1, 1, 1, &mid_shape);
    let relu = b.add_relu(conv1, &mid_shape);

    let w2 = b.add_input("box_w2", &[box_ch, MID_CH, 1, 1]);
    let b2 = b.add_input("box_b2", &[box_ch]);
    let conv2 = b.add_conv2d(relu, w2, Some(b2), 1, 1, 0, 0, &out_shape);

    let out = b.add_sigmoid(conv2, &out_shape);
    b.build(out).expect("valid box regression head")
}

fn box_regression_bindings() -> Vec<TensorParamBinding> {
    let box_ch = 4;
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[MID_CH, BACKBONE_CH, 3, 3])),
        TensorParamBinding::ConstantTensor(zeros(&[MID_CH])),
        TensorParamBinding::ConstantTensor(w(&[box_ch, MID_CH, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[box_ch])),
    ]
}

#[test]
fn test_paddle_detect_box_regression_ibp() {
    let def = build_box_regression();
    let bindings = box_regression_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let spatial = IMG / 2;
    let input = uniform_bounds(&[BACKBONE_CH, spatial, spatial], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through box regression");

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Box regression IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 6. NMS score filtering (IBP)
// ===========================================================================

fn build_nms_score_filter() -> TensorKernelDef {
    let spatial = IMG / 2;
    let in_shape = [BACKBONE_CH, spatial, spatial];
    let score_shape = [1, spatial, spatial];

    let mut b = TensorBlockBuilder::new("ppocr_nms_score");
    let input = b.add_input("features", &in_shape);

    // Confidence scoring: Conv -> sigmoid produces per-pixel confidence
    let ws = b.add_input("score_w", &[1, BACKBONE_CH, 1, 1]);
    let bs = b.add_input("score_b", &[1]);
    let conv = b.add_conv2d(input, ws, Some(bs), 1, 1, 0, 0, &score_shape);
    let out = b.add_sigmoid(conv, &score_shape);

    b.build(out).expect("valid NMS score filter")
}

fn nms_score_filter_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[1, BACKBONE_CH, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[1])),
    ]
}

#[test]
fn test_paddle_detect_nms_score_ibp() {
    let def = build_nms_score_filter();
    let bindings = nms_score_filter_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let spatial = IMG / 2;
    let input = uniform_bounds(&[BACKBONE_CH, spatial, spatial], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through NMS score filter");

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("NMS score filter IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 7. Feature pyramid network multi-scale bounds (IBP + CROWN)
// ===========================================================================

fn build_fpn_multiscale() -> TensorKernelDef {
    let spatial = IMG / 2;
    let in_shape = [BACKBONE_CH, spatial, spatial];
    let branch_shape = [FPN_CH, spatial, spatial];
    let out_shape = [FPN_CH, spatial, spatial];

    let mut b = TensorBlockBuilder::new("ppocr_fpn_multiscale");
    let input = b.add_input("features", &in_shape);

    // Branch 1: 1x1 lateral conv + ReLU (fine-grained features)
    let w1 = b.add_input("br1_w", &[FPN_CH, BACKBONE_CH, 1, 1]);
    let b1 = b.add_input("br1_b", &[FPN_CH]);
    let br1 = b.add_conv2d(input, w1, Some(b1), 1, 1, 0, 0, &branch_shape);
    let br1_relu = b.add_relu(br1, &branch_shape);

    // Branch 2: 3x3 conv + ReLU (larger receptive field, coarse features)
    let w2 = b.add_input("br2_w", &[FPN_CH, BACKBONE_CH, 3, 3]);
    let b2 = b.add_input("br2_b", &[FPN_CH]);
    let br2 = b.add_conv2d(input, w2, Some(b2), 1, 1, 1, 1, &branch_shape);
    let br2_relu = b.add_relu(br2, &branch_shape);

    // FPN lateral connection: element-wise add (standard FPN fusion)
    let fused = b.add_binary_add(br1_relu, br2_relu, &branch_shape);

    // Merge 1x1 conv for output smoothing
    let wm = b.add_input("merge_w", &[FPN_CH, FPN_CH, 1, 1]);
    let bm = b.add_input("merge_b", &[FPN_CH]);
    let out = b.add_conv2d(fused, wm, Some(bm), 1, 1, 0, 0, &out_shape);

    b.build(out).expect("valid FPN multi-scale")
}

fn fpn_multiscale_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[FPN_CH, BACKBONE_CH, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[FPN_CH])),
        TensorParamBinding::ConstantTensor(w(&[FPN_CH, BACKBONE_CH, 3, 3])),
        TensorParamBinding::ConstantTensor(zeros(&[FPN_CH])),
        TensorParamBinding::ConstantTensor(w(&[FPN_CH, FPN_CH, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[FPN_CH])),
    ]
}

#[test]
fn test_paddle_detect_fpn_multiscale_ibp() {
    let def = build_fpn_multiscale();
    let bindings = fpn_multiscale_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let spatial = IMG / 2;
    let input = uniform_bounds(&[BACKBONE_CH, spatial, spatial], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FPN multi-scale");

    assert_eq!(output.lower_upper().0.shape(), &[FPN_CH, spatial, spatial]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FPN multi-scale IBP: bounds=[{lo_min}, {hi_max}]");
}

#[test]
fn test_paddle_detect_fpn_multiscale_crown() {
    let def = build_fpn_multiscale();
    let bindings = fpn_multiscale_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let spatial = IMG / 2;
    let input = uniform_bounds(&[BACKBONE_CH, spatial, spatial], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FPN multi-scale CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 8. Recognition input normalization bounds (IBP)
// ===========================================================================

fn build_recognition_normalization() -> TensorKernelDef {
    let in_shape = [SEQ, HIDDEN];
    let out_shape = [SEQ, HIDDEN];

    let mut b = TensorBlockBuilder::new("ppocr_recog_norm");
    let input = b.add_input("recog_input", &in_shape);

    // LayerNorm normalization before recognition encoder
    let ln_w = b.add_input("ln_w", &[HIDDEN]);
    let ln_b = b.add_input("ln_b", &[HIDDEN]);
    let eps = b.add_input("ln_eps", &[1]);
    let out = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &out_shape);

    b.build(out).expect("valid recognition normalization")
}

fn recognition_normalization_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN])),
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN])),
        TensorParamBinding::ConstantScalar(1e-5),
    ]
}

#[test]
fn test_paddle_detect_recognition_norm_ibp() {
    let def = build_recognition_normalization();
    let bindings = recognition_normalization_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through recognition normalization");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ, HIDDEN]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Recognition normalization IBP: bounds=[{lo_min}, {hi_max}]");
}

// ===========================================================================
// 9. Character vocabulary probability distribution (IBP)
// ===========================================================================

fn build_vocab_probability() -> TensorKernelDef {
    let in_shape = [SEQ, HIDDEN];
    let logit_shape = [SEQ, VOCAB];

    let mut b = TensorBlockBuilder::new("ppocr_vocab_prob");
    let input = b.add_input("hidden", &in_shape);

    // Linear -> softmax for character probability distribution
    let wl = b.add_input("vocab_w", &[VOCAB, HIDDEN]);
    let bl = b.add_input("vocab_b", &[VOCAB]);
    let logits = b.add_linear(input, wl, Some(bl), &logit_shape);
    let out = b.add_softmax(logits, -1, &logit_shape);

    b.build(out).expect("valid vocab probability")
}

fn vocab_probability_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[VOCAB, HIDDEN])),
        TensorParamBinding::ConstantTensor(zeros(&[VOCAB])),
    ]
}

#[test]
fn test_paddle_detect_vocab_probability_ibp() {
    let def = build_vocab_probability();
    let bindings = vocab_probability_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through vocab probability");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ, VOCAB]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Vocab probability IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 10. Batch text detection throughput bounds (IBP)
// ===========================================================================

fn build_batch_detection() -> TensorKernelDef {
    let spatial = IMG / 2;
    let mid_shape = [MID_CH, spatial, spatial];
    let out_shape = [MAP_CH, spatial, spatial];

    let mut b = TensorBlockBuilder::new("ppocr_batch_detect");
    let input = b.add_input("batch_features", &[BACKBONE_CH, spatial, spatial]);

    // Lightweight detection head for throughput: Conv -> ReLU -> Conv -> sigmoid
    let w1 = b.add_input("batch_w1", &[MID_CH, BACKBONE_CH, 1, 1]);
    let b1 = b.add_input("batch_b1", &[MID_CH]);
    let conv1 = b.add_conv2d(input, w1, Some(b1), 1, 1, 0, 0, &mid_shape);
    let relu = b.add_relu(conv1, &mid_shape);

    let w2 = b.add_input("batch_w2", &[MAP_CH, MID_CH, 1, 1]);
    let b2 = b.add_input("batch_b2", &[MAP_CH]);
    let conv2 = b.add_conv2d(relu, w2, Some(b2), 1, 1, 0, 0, &out_shape);

    let out = b.add_sigmoid(conv2, &out_shape);
    b.build(out).expect("valid batch detection")
}

fn batch_detection_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[MID_CH, BACKBONE_CH, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[MID_CH])),
        TensorParamBinding::ConstantTensor(w(&[MAP_CH, MID_CH, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[MAP_CH])),
    ]
}

#[test]
fn test_paddle_detect_batch_throughput_ibp() {
    let def = build_batch_detection();
    let bindings = batch_detection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let spatial = IMG / 2;
    let input = uniform_bounds(&[BACKBONE_CH, spatial, spatial], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through batch detection");

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Batch detection IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 11. Text orientation classifier bounds (IBP)
// ===========================================================================

fn build_orientation_classifier() -> TensorKernelDef {
    let spatial = IMG / 2;
    let in_shape = [BACKBONE_CH, spatial, spatial];
    // Use spatial-sized conv to simulate global avg pool + classify in one op
    let cls_shape = [NUM_CLASSES, 1, 1];

    let mut b = TensorBlockBuilder::new("ppocr_orientation_cls");
    let input = b.add_input("features", &in_shape);

    // Conv with kernel=spatial acts as global avg pool + linear classifier
    let wc = b.add_input("cls_w", &[NUM_CLASSES, BACKBONE_CH, spatial, spatial]);
    let bc = b.add_input("cls_b", &[NUM_CLASSES]);
    let logits = b.add_conv2d(input, wc, Some(bc), 1, 1, 0, 0, &cls_shape);

    // Softmax over class dimension (0/180 degree orientation)
    let out = b.add_softmax(logits, 0, &cls_shape);

    b.build(out).expect("valid orientation classifier")
}

fn orientation_classifier_bindings() -> Vec<TensorParamBinding> {
    let spatial = IMG / 2;
    // Pool+classify weights: average-like 1/(spatial*spatial) scaled
    let pool_val = W_MAG / (spatial * spatial) as f32;
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLASSES, BACKBONE_CH, spatial, spatial]),
            pool_val,
        )),
        TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])),
    ]
}

#[test]
fn test_paddle_detect_orientation_cls_ibp() {
    let def = build_orientation_classifier();
    let bindings = orientation_classifier_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let spatial = IMG / 2;
    let input = uniform_bounds(&[BACKBONE_CH, spatial, spatial], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through orientation classifier");

    assert_eq!(output.lower_upper().0.shape(), &[NUM_CLASSES, 1, 1]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Orientation classifier IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 12. Text line grouping geometry bounds (IBP)
// ===========================================================================

fn build_line_grouping() -> TensorKernelDef {
    let spatial = IMG / 2;
    let in_shape = [BACKBONE_CH, spatial, spatial];
    let mid_shape = [MID_CH, spatial, spatial];
    // 2 channels: line angle + line distance
    let geom_ch = 2;
    let out_shape = [geom_ch, spatial, spatial];

    let mut b = TensorBlockBuilder::new("ppocr_line_grouping");
    let input = b.add_input("features", &in_shape);

    // Conv -> ReLU -> Conv -> sigmoid (normalized geometry in [0, 1])
    let w1 = b.add_input("line_w1", &[MID_CH, BACKBONE_CH, 3, 3]);
    let b1 = b.add_input("line_b1", &[MID_CH]);
    let conv1 = b.add_conv2d(input, w1, Some(b1), 1, 1, 1, 1, &mid_shape);
    let relu = b.add_relu(conv1, &mid_shape);

    let w2 = b.add_input("line_w2", &[geom_ch, MID_CH, 1, 1]);
    let b2 = b.add_input("line_b2", &[geom_ch]);
    let conv2 = b.add_conv2d(relu, w2, Some(b2), 1, 1, 0, 0, &out_shape);

    let out = b.add_sigmoid(conv2, &out_shape);
    b.build(out).expect("valid line grouping")
}

fn line_grouping_bindings() -> Vec<TensorParamBinding> {
    let geom_ch = 2;
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[MID_CH, BACKBONE_CH, 3, 3])),
        TensorParamBinding::ConstantTensor(zeros(&[MID_CH])),
        TensorParamBinding::ConstantTensor(w(&[geom_ch, MID_CH, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[geom_ch])),
    ]
}

#[test]
fn test_paddle_detect_line_grouping_ibp() {
    let def = build_line_grouping();
    let bindings = line_grouping_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let spatial = IMG / 2;
    let input = uniform_bounds(&[BACKBONE_CH, spatial, spatial], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through line grouping");

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Line grouping IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 13. Recognition beam search bounds (IBP)
// ===========================================================================

fn build_beam_search_output() -> TensorKernelDef {
    let in_shape = [SEQ, HIDDEN];
    let logit_shape = [SEQ, VOCAB];

    let mut b = TensorBlockBuilder::new("ppocr_beam_search");
    let input = b.add_input("hidden", &in_shape);

    // Linear -> ReLU -> Linear -> softmax
    // Models the scoring function used in beam search decoding
    let w1 = b.add_input("beam_w1", &[FFN_DIM, HIDDEN]);
    let b1 = b.add_input("beam_b1", &[FFN_DIM]);
    let ffn_shape = [SEQ, FFN_DIM];
    let h = b.add_linear(input, w1, Some(b1), &ffn_shape);
    let h_relu = b.add_relu(h, &ffn_shape);

    let w2 = b.add_input("beam_w2", &[VOCAB, FFN_DIM]);
    let b2 = b.add_input("beam_b2", &[VOCAB]);
    let logits = b.add_linear(h_relu, w2, Some(b2), &logit_shape);
    let out = b.add_softmax(logits, -1, &logit_shape);

    b.build(out).expect("valid beam search output")
}

fn beam_search_output_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, HIDDEN])),
        TensorParamBinding::ConstantTensor(zeros(&[FFN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[VOCAB, FFN_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[VOCAB])),
    ]
}

#[test]
fn test_paddle_detect_beam_search_ibp() {
    let def = build_beam_search_output();
    let bindings = beam_search_output_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through beam search");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ, VOCAB]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Beam search IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 14. Full detection-to-recognition pipeline (IBP)
// ===========================================================================

fn build_full_detect_to_recog_pipeline() -> TensorKernelDef {
    let spatial = IMG / 2;

    let mut b = TensorBlockBuilder::new("ppocr_full_pipeline");
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

    // Stage 2: DB detection head -> sigmoid probability map
    let det_mid_shape = [MID_CH, spatial, spatial];
    let det_out_shape = [MAP_CH, spatial, spatial];
    let dw1 = b.add_input("det_w1", &[MID_CH, BACKBONE_CH, 3, 3]);
    let db1 = b.add_input("det_b1", &[MID_CH]);
    let det_conv1 = b.add_conv2d(backbone, dw1, Some(db1), 1, 1, 1, 1, &det_mid_shape);
    let det_relu = b.add_relu(det_conv1, &det_mid_shape);

    let dw2 = b.add_input("det_w2", &[MAP_CH, MID_CH, 1, 1]);
    let db2 = b.add_input("det_b2", &[MAP_CH]);
    let det_conv2 = b.add_conv2d(det_relu, dw2, Some(db2), 1, 1, 0, 0, &det_out_shape);
    let _det_out = b.add_sigmoid(det_conv2, &det_out_shape);

    // Stage 3: Recognition path — channel reduction via 1x1 conv + softmax
    // Models the recognition head extracting character predictions from backbone features
    let recog_conv_shape = [VOCAB, spatial, spatial];
    let rw = b.add_input("recog_w", &[VOCAB, BACKBONE_CH, 1, 1]);
    let rb = b.add_input("recog_b", &[VOCAB]);
    let recog_conv = b.add_conv2d(backbone, rw, Some(rb), 1, 1, 0, 0, &recog_conv_shape);

    // Softmax over vocabulary (channel) dimension
    let out = b.add_softmax(recog_conv, 0, &recog_conv_shape);

    b.build(out)
        .expect("valid full detection-to-recognition pipeline")
}

fn full_detect_to_recog_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    // Backbone Conv-BN
    push_conv_bn_bindings(&mut bindings, BACKBONE_CH, IN_CH, 3);
    // Detection head
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        MID_CH,
        BACKBONE_CH,
        3,
        3,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[MID_CH])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        MAP_CH, MID_CH, 1, 1,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[MAP_CH])));
    // Recognition 1x1 conv
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        VOCAB,
        BACKBONE_CH,
        1,
        1,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[VOCAB])));
    bindings
}

#[test]
fn test_paddle_detect_full_pipeline_ibp() {
    let def = build_full_detect_to_recog_pipeline();
    let bindings = full_detect_to_recog_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(&[IN_CH, IMG, IMG]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full detection-to-recognition pipeline");

    let spatial = IMG / 2;
    assert_eq!(output.lower_upper().0.shape(), &[VOCAB, spatial, spatial]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full pipeline IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "softmax upper <= 1, got {hi_max}");
}
