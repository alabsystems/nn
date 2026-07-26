// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-architecture composition verification tests for dpdf document understanding.
//!
//! Verifies properties that span different model architectures working together in
//! the dpdf pipeline. Unlike `compose_dpdf_cross_model.rs` (pairwise model boundaries)
//! or `compose_dpdf_cross_model_pipeline.rs` (full pipeline chains), these tests
//! verify **architectural invariants** that must hold across heterogeneous models.
//!
//! ## Detection -> Recognition Pipeline (3 tests)
//!
//! 1. Detection sigmoid -> linear -> ReLU -> CTC softmax (IBP)
//! 2. Same pipeline with CROWN linearization
//! 3. Monotone tightening: [0.3, 0.7] tighter than [0, 1]
//!
//! ## Layout -> Table Pipeline (2 tests)
//!
//! 4. Box coords -> query projection -> LayerNorm -> sigmoid structure (IBP)
//! 5. Same pipeline with CROWN
//!
//! ## Vision Encoder Comparison (2 tests)
//!
//! 6. SigLIP2 (Conv2d+LayerNorm) vs Qwen3-VL (Conv2d+RMSNorm) patch embed (IBP)
//! 7. Vision encoder bounds containment (same-magnitude check)
//!
//! ## Multi-Model Softmax Consistency (2 tests)
//!
//! 8. GLM-OCR vs PaddleOCR vs Qwen3-VL softmax heads (IBP)
//! 9. Softmax with varying hidden dims (IBP)
//!
//! ## Shared Backbone Invariant (3 tests)
//!
//! 10. Conv-BN-ReLU vs Conv-BN-Sigmoid backbone comparison (IBP)
//! 11. Detection head vs structure head from same backbone (IBP)
//! 12. Shared backbone CROWN consistency
//!
//! Part of #3956: cross-architecture compose tests for dpdf pipeline verification.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const FEATURE_DIM: usize = 32;
const SEQ_LEN: usize = 4;
const VOCAB_SIZE: usize = 16;
const NUM_CLASSES: usize = 8;
const NUM_ANCHORS: usize = 6;
const BACKBONE_CH: usize = 16;
const IMG_CHANNELS: usize = 3;
const SPATIAL: usize = 4;
const TABLE_CLASSES: usize = 4;
const GLM_HIDDEN: usize = 64;
const QWEN_HIDDEN: usize = 48;
const WEIGHT_MAG: f32 = 0.02;

// ===========================================================================
// Helpers
// ===========================================================================

fn w(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG)
}

fn zeros(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 0.0f32)
}

fn ones(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 1.0f32)
}

fn eps_tensor() -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)
}

fn sigmoid_bounds(shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), 0.0f32),
        ArrayD::from_elem(IxDyn(shape), 1.0f32),
    )
    .expect("valid sigmoid bounds [0, 1]")
}

fn image_bounds(shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), 0.0f32),
        ArrayD::from_elem(IxDyn(shape), 1.0f32),
    )
    .expect("valid image bounds [0, 1]")
}

fn narrowed_bounds(shape: &[usize], lo: f32, hi: f32) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), lo),
        ArrayD::from_elem(IxDyn(shape), hi),
    )
    .expect("valid narrowed bounds")
}

// ===========================================================================
// 1-3. Detection -> Recognition: DocLayout-YOLO -> PaddleOCR CTC
// ===========================================================================

fn build_detection_to_recognition_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("cross_arch_detection_to_recognition");
    // Batch-major [SEQ_LEN, NUM_CLASSES]: nn.Linear contracts the last dim
    // (NUM_CLASSES) against weight [out, in] = [FEATURE_DIM, NUM_CLASSES].
    let input = b.add_input("det_sigmoid", &[SEQ_LEN, NUM_CLASSES]);
    let proj_w = b.add_input("proj_weight", &[FEATURE_DIM, NUM_CLASSES]);
    let proj_b = b.add_input("proj_bias", &[FEATURE_DIM]);
    let features = b.add_linear(input, proj_w, Some(proj_b), &[SEQ_LEN, FEATURE_DIM]);
    let activated = b.add_relu(features, &[SEQ_LEN, FEATURE_DIM]);
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, FEATURE_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(activated, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    b.build(out).expect("valid detection to recognition kernel")
}

fn detection_to_recognition_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[FEATURE_DIM, NUM_CLASSES])),
        TensorParamBinding::ConstantTensor(zeros(&[FEATURE_DIM])),
        TensorParamBinding::ConstantTensor(w(&[VOCAB_SIZE, FEATURE_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[VOCAB_SIZE])),
    ]
}

#[test]
fn test_cross_arch_detection_to_recognition_ibp() {
    let def = build_detection_to_recognition_kernel();
    let bindings = detection_to_recognition_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = sigmoid_bounds(&[SEQ_LEN, NUM_CLASSES]);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cross-arch det->rec IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-4, "softmax lo >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax hi <= 1, got {hi_max}");
}

#[test]
fn test_cross_arch_detection_to_recognition_crown() {
    let def = build_detection_to_recognition_kernel();
    let bindings = detection_to_recognition_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = sigmoid_bounds(&[SEQ_LEN, NUM_CLASSES]);
    let (method, crown_output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&crown_output);
    eprintln!("Cross-arch det->rec CROWN: {method:?} [{lo_min}, {hi_max}] fb={fallback:?}");
    assert!(lo_min >= -1e-4, "CROWN softmax lo >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "CROWN softmax hi <= 1, got {hi_max}");
}

#[test]
fn test_cross_arch_detection_to_recognition_monotone_tightening() {
    let def = build_detection_to_recognition_kernel();
    let bindings = detection_to_recognition_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let wide_input = sigmoid_bounds(&[SEQ_LEN, NUM_CLASSES]);
    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");
    assert_bounds_valid(&wide_output);
    let (wide_lo, wide_hi) = bounds_min_max(&wide_output);

    let tight_input = narrowed_bounds(&[SEQ_LEN, NUM_CLASSES], 0.3, 0.7);
    let tight_output = graph.propagate_ibp(&tight_input).expect("IBP tight");
    assert_bounds_valid(&tight_output);
    let (tight_lo, tight_hi) = bounds_min_max(&tight_output);

    let wide_width = wide_hi - wide_lo;
    let tight_width = tight_hi - tight_lo;
    eprintln!("Monotone: wide={wide_width:.6} tight={tight_width:.6}");
    assert!(
        tight_width <= wide_width + 1e-4,
        "tighter input => tighter output"
    );
}

// ===========================================================================
// 4-5. Layout -> Table: DocLayout-YOLO -> Table Transformer
// ===========================================================================

fn build_layout_to_table_structure_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("cross_arch_layout_to_table_structure");
    let input = b.add_input("box_coords", &[NUM_ANCHORS, 4]);
    let q_w = b.add_input("query_weight", &[FEATURE_DIM, 4]);
    let q_b = b.add_input("query_bias", &[FEATURE_DIM]);
    let queries = b.add_linear(input, q_w, Some(q_b), &[NUM_ANCHORS, FEATURE_DIM]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let ln_scale = b.add_input("ln_scale", &[FEATURE_DIM]);
    let ln_bias = b.add_input("ln_bias", &[FEATURE_DIM]);
    let normed = b.add_layer_norm(
        queries,
        ln_eps,
        1,
        ln_scale,
        ln_bias,
        &[NUM_ANCHORS, FEATURE_DIM],
    );
    let cls_w = b.add_input("cls_weight", &[TABLE_CLASSES, FEATURE_DIM]);
    let cls_b = b.add_input("cls_bias", &[TABLE_CLASSES]);
    let logits = b.add_linear(normed, cls_w, Some(cls_b), &[NUM_ANCHORS, TABLE_CLASSES]);
    let out = b.add_sigmoid(logits, &[NUM_ANCHORS, TABLE_CLASSES]);
    b.build(out)
        .expect("valid layout to table structure kernel")
}

fn layout_to_table_structure_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[FEATURE_DIM, 4])),
        TensorParamBinding::ConstantTensor(zeros(&[FEATURE_DIM])),
        TensorParamBinding::ConstantTensor(eps_tensor()),
        TensorParamBinding::ConstantTensor(ones(&[FEATURE_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[FEATURE_DIM])),
        TensorParamBinding::ConstantTensor(w(&[TABLE_CLASSES, FEATURE_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[TABLE_CLASSES])),
    ]
}

#[test]
fn test_cross_arch_layout_to_table_structure_ibp() {
    let def = build_layout_to_table_structure_kernel();
    let bindings = layout_to_table_structure_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = sigmoid_bounds(&[NUM_ANCHORS, 4]);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cross-arch layout->table IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-4, "sigmoid lo >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid hi <= 1, got {hi_max}");
}

#[test]
fn test_cross_arch_layout_to_table_structure_crown() {
    let def = build_layout_to_table_structure_kernel();
    let bindings = layout_to_table_structure_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = sigmoid_bounds(&[NUM_ANCHORS, 4]);
    let (method, crown_output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&crown_output);
    eprintln!("Cross-arch layout->table CROWN: {method:?} [{lo_min}, {hi_max}] fb={fallback:?}");
    assert!(lo_min >= -1e-4, "CROWN sigmoid lo >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "CROWN sigmoid hi <= 1, got {hi_max}");
}

// ===========================================================================
// 6-7. Vision encoder comparison: SigLIP2 vs Qwen3-VL patch embed
// ===========================================================================

fn build_siglip2_patch_embed_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("cross_arch_siglip2_patch_embed");
    let input = b.add_input("image_patch", &[IMG_CHANNELS, SPATIAL, SPATIAL]);
    let conv_w = b.add_input(
        "conv_weight",
        &[FEATURE_DIM, IMG_CHANNELS, SPATIAL, SPATIAL],
    );
    let conv_b = b.add_input("conv_bias", &[FEATURE_DIM]);
    // Conv2d(stride=SPATIAL, padding=0): [3, S, S] -> [D, 1, 1]
    let embedded = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        SPATIAL,
        SPATIAL,
        0,
        0,
        &[FEATURE_DIM, 1, 1],
    );
    let reshaped = b.add_reshape(embedded, &[1, FEATURE_DIM]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let ln_scale = b.add_input("ln_scale", &[FEATURE_DIM]);
    let ln_bias = b.add_input("ln_bias", &[FEATURE_DIM]);
    let out = b.add_layer_norm(reshaped, ln_eps, 1, ln_scale, ln_bias, &[1, FEATURE_DIM]);
    b.build(out).expect("valid SigLIP2 patch embed kernel")
}

fn siglip2_patch_embed_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[FEATURE_DIM, IMG_CHANNELS, SPATIAL, SPATIAL])),
        TensorParamBinding::ConstantTensor(zeros(&[FEATURE_DIM])),
        TensorParamBinding::ConstantTensor(eps_tensor()),
        TensorParamBinding::ConstantTensor(ones(&[FEATURE_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[FEATURE_DIM])),
    ]
}

fn build_qwen3vl_patch_embed_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("cross_arch_qwen3vl_patch_embed");
    let input = b.add_input("image_patch", &[IMG_CHANNELS, SPATIAL, SPATIAL]);
    let conv_w = b.add_input(
        "conv_weight",
        &[FEATURE_DIM, IMG_CHANNELS, SPATIAL, SPATIAL],
    );
    let conv_b = b.add_input("conv_bias", &[FEATURE_DIM]);
    let embedded = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        SPATIAL,
        SPATIAL,
        0,
        0,
        &[FEATURE_DIM, 1, 1],
    );
    let reshaped = b.add_reshape(embedded, &[1, FEATURE_DIM]);
    let rms_eps = b.add_input("rms_eps", &[1]);
    let rms_scale = b.add_input("rms_scale", &[FEATURE_DIM]);
    let out = b.add_rms_norm(reshaped, rms_eps, 1, rms_scale, &[1, FEATURE_DIM]);
    b.build(out).expect("valid Qwen3-VL patch embed kernel")
}

fn qwen3vl_patch_embed_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[FEATURE_DIM, IMG_CHANNELS, SPATIAL, SPATIAL])),
        TensorParamBinding::ConstantTensor(zeros(&[FEATURE_DIM])),
        TensorParamBinding::ConstantTensor(eps_tensor()),
        TensorParamBinding::ConstantTensor(ones(&[FEATURE_DIM])),
    ]
}

#[test]
fn test_cross_arch_vision_encoder_siglip2_vs_qwen3vl_ibp() {
    let siglip2_graph = tensor_kernel_to_graph(
        &build_siglip2_patch_embed_kernel(),
        &siglip2_patch_embed_bindings(),
    )
    .expect("SigLIP2 graph");
    let qwen3vl_graph = tensor_kernel_to_graph(
        &build_qwen3vl_patch_embed_kernel(),
        &qwen3vl_patch_embed_bindings(),
    )
    .expect("Qwen3-VL graph");

    let input = image_bounds(&[IMG_CHANNELS, SPATIAL, SPATIAL]);
    let siglip2_output = siglip2_graph.propagate_ibp(&input).expect("SigLIP2 IBP");
    let qwen3vl_output = qwen3vl_graph.propagate_ibp(&input).expect("Qwen3-VL IBP");

    assert_bounds_valid(&siglip2_output);
    assert_bounds_valid(&qwen3vl_output);

    let (s_lo, s_hi) = bounds_min_max(&siglip2_output);
    let (q_lo, q_hi) = bounds_min_max(&qwen3vl_output);
    eprintln!("SigLIP2 IBP: [{s_lo}, {s_hi}]");
    eprintln!("Qwen3-VL IBP: [{q_lo}, {q_hi}]");
    assert!(s_lo.is_finite() && s_hi.is_finite(), "SigLIP2 finite");
    assert!(q_lo.is_finite() && q_hi.is_finite(), "Qwen3-VL finite");
}

#[test]
fn test_cross_arch_vision_encoder_bounds_containment() {
    let siglip2_graph = tensor_kernel_to_graph(
        &build_siglip2_patch_embed_kernel(),
        &siglip2_patch_embed_bindings(),
    )
    .expect("SigLIP2 graph");
    let qwen3vl_graph = tensor_kernel_to_graph(
        &build_qwen3vl_patch_embed_kernel(),
        &qwen3vl_patch_embed_bindings(),
    )
    .expect("Qwen3-VL graph");

    let input = image_bounds(&[IMG_CHANNELS, SPATIAL, SPATIAL]);
    let siglip2_output = siglip2_graph.propagate_ibp(&input).expect("SigLIP2 IBP");
    let qwen3vl_output = qwen3vl_graph.propagate_ibp(&input).expect("Qwen3-VL IBP");

    let (s_lo, s_hi) = bounds_min_max(&siglip2_output);
    let (q_lo, q_hi) = bounds_min_max(&qwen3vl_output);
    let s_width = s_hi - s_lo;
    let q_width = q_hi - q_lo;
    eprintln!("Width: SigLIP2={s_width:.4}, Qwen3-VL={q_width:.4}");

    let max_w = s_width.max(q_width);
    let min_w = s_width.min(q_width);
    if min_w > 1e-6 {
        let ratio = max_w / min_w;
        eprintln!("Width ratio: {ratio:.2}x");
        assert!(
            ratio < 100.0,
            "encoders should be same magnitude: ratio={ratio}"
        );
    }
}

// ===========================================================================
// 8-9. Multi-model softmax consistency
// ===========================================================================

fn build_lm_head_softmax_kernel(name: &str, hidden_dim: usize) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("features", &[SEQ_LEN, hidden_dim]);
    let lm_w = b.add_input("lm_weight", &[VOCAB_SIZE, hidden_dim]);
    let lm_b = b.add_input("lm_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(input, lm_w, Some(lm_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    b.build(out).expect("valid LM head softmax kernel")
}

fn lm_head_bindings(hidden_dim: usize) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[VOCAB_SIZE, hidden_dim])),
        TensorParamBinding::ConstantTensor(zeros(&[VOCAB_SIZE])),
    ]
}

#[test]
fn test_cross_arch_multi_model_softmax_consistency_ibp() {
    let models: [(&str, usize); 3] = [
        ("cross_arch_glm_lm_head", FEATURE_DIM),
        ("cross_arch_paddle_ctc_head", FEATURE_DIM),
        ("cross_arch_qwen_lm_head", FEATURE_DIM),
    ];
    let input = uniform_bounds(&[SEQ_LEN, FEATURE_DIM], 2.0);

    for (name, hidden) in &models {
        let def = build_lm_head_softmax_kernel(name, *hidden);
        let bindings = lm_head_bindings(*hidden);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let output = graph.propagate_ibp(&input).expect("IBP");
        assert_bounds_valid(&output);
        let (lo_min, hi_max) = bounds_min_max(&output);
        eprintln!("{name} softmax IBP: [{lo_min}, {hi_max}]");
        assert!(lo_min >= -1e-4, "{name} softmax lo >= 0, got {lo_min}");
        assert!(hi_max <= 1.0 + 1e-4, "{name} softmax hi <= 1, got {hi_max}");
    }
}

#[test]
fn test_cross_arch_softmax_varying_hidden_dims_ibp() {
    let models: [(&str, usize); 3] = [
        ("cross_arch_glm_wide_head", GLM_HIDDEN),
        ("cross_arch_paddle_base_head", FEATURE_DIM),
        ("cross_arch_qwen_mid_head", QWEN_HIDDEN),
    ];

    for (name, hidden) in &models {
        let def = build_lm_head_softmax_kernel(name, *hidden);
        let bindings = lm_head_bindings(*hidden);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let input = uniform_bounds(&[SEQ_LEN, *hidden], 2.0);
        let output = graph.propagate_ibp(&input).expect("IBP");
        assert_bounds_valid(&output);
        let (lo_min, hi_max) = bounds_min_max(&output);
        eprintln!("{name} (hidden={hidden}) softmax IBP: [{lo_min}, {hi_max}]");
        assert!(lo_min >= -1e-4, "{name} softmax lo >= 0, got {lo_min}");
        assert!(hi_max <= 1.0 + 1e-4, "{name} softmax hi <= 1, got {hi_max}");
    }
}

// ===========================================================================
// 10-12. Shared backbone invariant
// ===========================================================================

fn build_conv_bn_relu_backbone_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("cross_arch_conv_bn_relu_backbone");
    let input = b.add_input("image", &[IMG_CHANNELS, SPATIAL, SPATIAL]);
    let conv_w = b.add_input("conv_weight", &[BACKBONE_CH, IMG_CHANNELS, 1, 1]);
    let conv_b = b.add_input("conv_bias", &[BACKBONE_CH]);
    let conv_out = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        1,
        1,
        0,
        0,
        &[BACKBONE_CH, SPATIAL, SPATIAL],
    );
    let bn_mean = b.add_input("bn_mean", &[BACKBONE_CH]);
    let bn_var = b.add_input("bn_var", &[BACKBONE_CH]);
    let bn_scale = b.add_input("bn_scale", &[BACKBONE_CH]);
    let bn_bias = b.add_input("bn_bias", &[BACKBONE_CH]);
    let bn_eps = b.add_input("bn_eps", &[1]);
    let normed = b.add_batch_norm(
        conv_out,
        bn_mean,
        bn_var,
        bn_scale,
        bn_bias,
        bn_eps,
        &[BACKBONE_CH, SPATIAL, SPATIAL],
    );
    let out = b.add_relu(normed, &[BACKBONE_CH, SPATIAL, SPATIAL]);
    b.build(out).expect("valid Conv-BN-ReLU backbone kernel")
}

fn conv_bn_relu_backbone_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[BACKBONE_CH, IMG_CHANNELS, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[BACKBONE_CH])),
        TensorParamBinding::ConstantTensor(zeros(&[BACKBONE_CH])), // bn_mean
        TensorParamBinding::ConstantTensor(ones(&[BACKBONE_CH])),  // bn_var
        TensorParamBinding::ConstantTensor(ones(&[BACKBONE_CH])),  // bn_scale
        TensorParamBinding::ConstantTensor(zeros(&[BACKBONE_CH])), // bn_bias
        TensorParamBinding::ConstantTensor(eps_tensor()),          // bn_eps
    ]
}

fn build_conv_bn_sigmoid_backbone_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("cross_arch_conv_bn_sigmoid_backbone");
    let input = b.add_input("image", &[IMG_CHANNELS, SPATIAL, SPATIAL]);
    let conv_w = b.add_input("conv_weight", &[BACKBONE_CH, IMG_CHANNELS, 1, 1]);
    let conv_b = b.add_input("conv_bias", &[BACKBONE_CH]);
    let conv_out = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        1,
        1,
        0,
        0,
        &[BACKBONE_CH, SPATIAL, SPATIAL],
    );
    let bn_mean = b.add_input("bn_mean", &[BACKBONE_CH]);
    let bn_var = b.add_input("bn_var", &[BACKBONE_CH]);
    let bn_scale = b.add_input("bn_scale", &[BACKBONE_CH]);
    let bn_bias = b.add_input("bn_bias", &[BACKBONE_CH]);
    let bn_eps = b.add_input("bn_eps", &[1]);
    let normed = b.add_batch_norm(
        conv_out,
        bn_mean,
        bn_var,
        bn_scale,
        bn_bias,
        bn_eps,
        &[BACKBONE_CH, SPATIAL, SPATIAL],
    );
    let out = b.add_sigmoid(normed, &[BACKBONE_CH, SPATIAL, SPATIAL]);
    b.build(out).expect("valid Conv-BN-Sigmoid backbone kernel")
}

fn conv_bn_sigmoid_backbone_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[BACKBONE_CH, IMG_CHANNELS, 1, 1])),
        TensorParamBinding::ConstantTensor(zeros(&[BACKBONE_CH])),
        TensorParamBinding::ConstantTensor(zeros(&[BACKBONE_CH])),
        TensorParamBinding::ConstantTensor(ones(&[BACKBONE_CH])),
        TensorParamBinding::ConstantTensor(ones(&[BACKBONE_CH])),
        TensorParamBinding::ConstantTensor(zeros(&[BACKBONE_CH])),
        TensorParamBinding::ConstantTensor(eps_tensor()),
    ]
}

#[test]
fn test_cross_arch_shared_backbone_conv_bn_activation_ibp() {
    let relu_graph = tensor_kernel_to_graph(
        &build_conv_bn_relu_backbone_kernel(),
        &conv_bn_relu_backbone_bindings(),
    )
    .expect("ReLU graph");
    let sigmoid_graph = tensor_kernel_to_graph(
        &build_conv_bn_sigmoid_backbone_kernel(),
        &conv_bn_sigmoid_backbone_bindings(),
    )
    .expect("Sigmoid graph");

    let input = image_bounds(&[IMG_CHANNELS, SPATIAL, SPATIAL]);
    let relu_output = relu_graph.propagate_ibp(&input).expect("ReLU IBP");
    let sigmoid_output = sigmoid_graph.propagate_ibp(&input).expect("Sigmoid IBP");

    assert_bounds_valid(&relu_output);
    assert_bounds_valid(&sigmoid_output);

    let (relu_lo, relu_hi) = bounds_min_max(&relu_output);
    let (sig_lo, sig_hi) = bounds_min_max(&sigmoid_output);
    eprintln!("Conv-BN-ReLU IBP: [{relu_lo}, {relu_hi}]");
    eprintln!("Conv-BN-Sigmoid IBP: [{sig_lo}, {sig_hi}]");
    assert!(relu_lo >= -1e-4, "ReLU lo >= 0, got {relu_lo}");
    assert!(sig_lo >= -1e-4, "Sigmoid lo >= 0, got {sig_lo}");
    assert!(sig_hi <= 1.0 + 1e-4, "Sigmoid hi <= 1, got {sig_hi}");
}

// 11. Backbone -> detection head vs structure head

fn build_backbone_to_detection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("cross_arch_backbone_to_detection");
    // Batch-major [SPATIAL, BACKBONE_CH]: nn.Linear contracts the last dim
    // (BACKBONE_CH) against weight [out, in] = [NUM_CLASSES, BACKBONE_CH].
    let input = b.add_input("backbone_features", &[SPATIAL, BACKBONE_CH]);
    let det_w = b.add_input("det_weight", &[NUM_CLASSES, BACKBONE_CH]);
    let det_b = b.add_input("det_bias", &[NUM_CLASSES]);
    let logits = b.add_linear(input, det_w, Some(det_b), &[SPATIAL, NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[SPATIAL, NUM_CLASSES]);
    b.build(out).expect("valid backbone to detection kernel")
}

fn build_backbone_to_structure_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("cross_arch_backbone_to_structure");
    // Batch-major [SPATIAL, BACKBONE_CH]: nn.Linear contracts the last dim
    // (BACKBONE_CH) against weight [out, in] = [TABLE_CLASSES, BACKBONE_CH].
    let input = b.add_input("backbone_features", &[SPATIAL, BACKBONE_CH]);
    let str_w = b.add_input("structure_weight", &[TABLE_CLASSES, BACKBONE_CH]);
    let str_b = b.add_input("structure_bias", &[TABLE_CLASSES]);
    let logits = b.add_linear(input, str_w, Some(str_b), &[SPATIAL, TABLE_CLASSES]);
    let out = b.add_sigmoid(logits, &[SPATIAL, TABLE_CLASSES]);
    b.build(out).expect("valid backbone to structure kernel")
}

#[test]
fn test_cross_arch_backbone_head_independence_ibp() {
    let det_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[NUM_CLASSES, BACKBONE_CH])),
        TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])),
    ];
    let str_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[TABLE_CLASSES, BACKBONE_CH])),
        TensorParamBinding::ConstantTensor(zeros(&[TABLE_CLASSES])),
    ];
    let det_graph = tensor_kernel_to_graph(&build_backbone_to_detection_kernel(), &det_bindings)
        .expect("det graph");
    let str_graph = tensor_kernel_to_graph(&build_backbone_to_structure_kernel(), &str_bindings)
        .expect("str graph");

    let backbone_input = uniform_bounds(&[SPATIAL, BACKBONE_CH], 1.0);
    let det_output = det_graph.propagate_ibp(&backbone_input).expect("det IBP");
    let str_output = str_graph.propagate_ibp(&backbone_input).expect("str IBP");

    assert_bounds_valid(&det_output);
    assert_bounds_valid(&str_output);

    for (name, output) in [("Detection", &det_output), ("Structure", &str_output)] {
        let (lo_min, hi_max) = bounds_min_max(output);
        eprintln!("{name} head IBP: [{lo_min}, {hi_max}]");
        assert!(lo_min >= -1e-4, "{name} sigmoid lo >= 0, got {lo_min}");
        assert!(hi_max <= 1.0 + 1e-4, "{name} sigmoid hi <= 1, got {hi_max}");
    }
}

// 12. Backbone CROWN consistency

#[test]
fn test_cross_arch_backbone_crown_consistency() {
    let backbone_graph = tensor_kernel_to_graph(
        &build_conv_bn_relu_backbone_kernel(),
        &conv_bn_relu_backbone_bindings(),
    )
    .expect("backbone graph");

    let input = image_bounds(&[IMG_CHANNELS, SPATIAL, SPATIAL]);
    let (method, crown_output, fallback) =
        assert_crown_tighter_when_not_fallback(&backbone_graph, &input);
    assert_bounds_valid(&crown_output);
    let (lo_min, hi_max) = bounds_min_max(&crown_output);
    eprintln!("Backbone CROWN: {method:?} [{lo_min}, {hi_max}] fb={fallback:?}");
    assert!(lo_min >= -1e-4, "CROWN backbone lo >= 0, got {lo_min}");
}
