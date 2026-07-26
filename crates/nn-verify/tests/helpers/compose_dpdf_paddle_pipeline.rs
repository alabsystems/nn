// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for PaddleOCR-VL detection + recognition pipeline bounds.
//!
//! Verifies IBP and CROWN bound propagation through the PaddleOCR-VL pipeline
//! consisting of a DB text detection backbone and SVTR text recognition encoder:
//!
//! ## Tests (20 tests)
//!
//! 1.  **Detection backbone Conv-BN-ReLU feature bounds** (IBP)
//! 2.  **DB text detection FPN feature bounds** (IBP)
//! 3.  **Sigmoid detection head probability [0,1]** (IBP)
//! 4.  **SVTR patch embedding bounds** (IBP)
//! 5.  **SVTR transformer self-attention bounds** (CROWN)
//! 6.  **SVTR feature projection bounds** (IBP)
//! 7.  **CTC log-probability output bounds** (IBP)
//! 8.  **Text region crop bounds** (IBP)
//! 9.  **SVTR encoder feature bounds** (IBP + CROWN)
//! 10. **CTC output length preservation** (IBP)
//! 11. **Detection pipeline end-to-end** (IBP)
//! 12. **Recognition confidence via softmax bounds** (IBP)
//! 13. **NMS IoU filtering bounds** (IBP)
//! 14. **Text line grouping bounds** (IBP)
//! 15. **Multi-scale detection bounds** (IBP)
//! 16. **Full pipeline composition (detection+recognition)** (IBP)
//! 17. **Deep detection backbone bounds** (IBP)
//! 18. **SVTR attention + residual bounds** (CROWN)
//! 19. **CTC log-softmax score bounds** (IBP)
//! 20. **Full pipeline with normalization** (IBP + CROWN)
//!
//! Architecture references:
//! - PaddleOCR (Baidu): Production OCR with DB detector + SVTR recognizer
//! - DB (Liao et al. 2020): Differentiable Binarization for text detection
//! - SVTR (Du et al. 2022): Scene Text Recognition with a Single Visual Model
//! - CTC (Graves et al. 2006): Connectionist Temporal Classification
//!
//! Dimensions (small for fast verification, structurally representative):
//! - IN_CHANNELS=3, IMG_SIZE=4, BACKBONE_CH=4, FPN_CH=4
//! - HIDDEN_DIM=4, SEQ_LEN=4, VOCAB_SIZE=6, NUM_HEADS=2, FFN_DIM=8
//!
//! Part of #4194: Compose tests for PaddleOCR-VL pipeline bounds.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::AttentionMask;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// of PaddleOCR-VL (DB detector + SVTR recognizer)
// ---------------------------------------------------------------------------

const IN_CHANNELS: usize = 3;
const IMG_SIZE: usize = 4;
const BACKBONE_CH: usize = 4;
const FPN_CH: usize = 4;
const HIDDEN_DIM: usize = 4;
const SEQ_LEN: usize = 4;
const VOCAB_SIZE: usize = 6;
const NUM_HEADS: usize = 2;
const FFN_DIM: usize = 8;
const WEIGHT_MAG: f32 = 0.1;

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

/// Ones tensor binding (for LayerNorm weight).
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

// ===========================================================================
// 1. Detection backbone Conv-BN-ReLU feature bounds (IBP)
// ===========================================================================

#[test]
fn test_paddle_pipeline_det_backbone_conv_bn_relu_ibp() {
    let mut b = TensorBlockBuilder::new("paddle_det_conv_bn_relu");
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let conv_w = b.add_input("conv_w", &[BACKBONE_CH, IN_CHANNELS, 3, 3]);
    let conv_b = b.add_input("conv_b", &[BACKBONE_CH]);

    let conv_out = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        1,
        1,
        1,
        1,
        &[BACKBONE_CH, IMG_SIZE, IMG_SIZE],
    );

    // BatchNorm
    let bn_mean = b.add_input("bn_mean", &[BACKBONE_CH]);
    let bn_var = b.add_input("bn_var", &[BACKBONE_CH]);
    let bn_w = b.add_input("bn_w", &[BACKBONE_CH]);
    let bn_b = b.add_input("bn_b", &[BACKBONE_CH]);
    let bn_eps = b.add_input("bn_eps", &[1]);
    let bn_out = b.add_batch_norm(
        conv_out,
        bn_mean,
        bn_var,
        bn_w,
        bn_b,
        bn_eps,
        &[BACKBONE_CH, IMG_SIZE, IMG_SIZE],
    );

    // ReLU
    let out = b.add_relu(bn_out, &[BACKBONE_CH, IMG_SIZE, IMG_SIZE]);
    let def = b.build(out).expect("valid det backbone Conv-BN-ReLU");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[BACKBONE_CH, IN_CHANNELS, 3, 3]),
        bias_zero(&[BACKBONE_CH]),
        bias_zero(&[BACKBONE_CH]), // running_mean
        ones(&[BACKBONE_CH]),      // running_var
        ones(&[BACKBONE_CH]),      // bn weight
        bias_zero(&[BACKBONE_CH]), // bn bias
        eps_binding(),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(IN_CHANNELS, IMG_SIZE, IMG_SIZE);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Paddle det Conv-BN-ReLU IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // ReLU output must be non-negative
    assert!(
        lo_min >= -1e-5,
        "ReLU lower bound must be >= 0, got {lo_min}"
    );
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 2. DB text detection FPN feature bounds (IBP)
// ===========================================================================

#[test]
fn test_paddle_pipeline_db_fpn_feature_ibp() {
    // FPN lateral projection: backbone features -> 1x1 conv -> FPN features
    let mut b = TensorBlockBuilder::new("paddle_db_fpn");
    let input = b.add_input("backbone_features", &[BACKBONE_CH, IMG_SIZE, IMG_SIZE]);
    let proj_w = b.add_input("fpn_proj_w", &[FPN_CH, BACKBONE_CH, 1, 1]);
    let proj_b = b.add_input("fpn_proj_b", &[FPN_CH]);

    let out = b.add_conv2d(
        input,
        proj_w,
        Some(proj_b),
        1,
        1,
        0,
        0,
        &[FPN_CH, IMG_SIZE, IMG_SIZE],
    );
    let def = b.build(out).expect("valid FPN lateral projection");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FPN_CH, BACKBONE_CH, 1, 1]),
        bias_zero(&[FPN_CH]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = uniform_bounds(&[BACKBONE_CH, IMG_SIZE, IMG_SIZE], 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[FPN_CH, IMG_SIZE, IMG_SIZE]
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Paddle DB FPN IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 3. Sigmoid detection head probability [0,1] (IBP)
// ===========================================================================

#[test]
fn test_paddle_pipeline_sigmoid_det_head_ibp() {
    // Detection head: FPN features -> 1x1 conv -> sigmoid -> [0, 1]
    let mut b = TensorBlockBuilder::new("paddle_sigmoid_det");
    let input = b.add_input("fpn_features", &[FPN_CH, IMG_SIZE, IMG_SIZE]);
    let head_w = b.add_input("det_head_w", &[1, FPN_CH, 1, 1]);
    let head_b = b.add_input("det_head_b", &[1]);

    let conv_out = b.add_conv2d(
        input,
        head_w,
        Some(head_b),
        1,
        1,
        0,
        0,
        &[1, IMG_SIZE, IMG_SIZE],
    );
    let out = b.add_sigmoid(conv_out, &[1, IMG_SIZE, IMG_SIZE]);
    let def = b.build(out).expect("valid sigmoid detection head");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[1, FPN_CH, 1, 1]),
        bias_zero(&[1]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = uniform_bounds(&[FPN_CH, IMG_SIZE, IMG_SIZE], 2.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Paddle sigmoid det head IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= -1e-5,
        "sigmoid lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-5,
        "sigmoid upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 4. SVTR patch embedding bounds (IBP)
// ===========================================================================

#[test]
fn test_paddle_pipeline_svtr_patch_embed_ibp() {
    // SVTR patch embedding: Conv2d -> reshape -> transpose
    let patch_size = 2;
    let grid = IMG_SIZE / patch_size; // 2
    let num_patches = grid * grid; // 4

    let mut b = TensorBlockBuilder::new("paddle_svtr_patch_embed");
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let embed_w = b.add_input(
        "embed_w",
        &[HIDDEN_DIM, IN_CHANNELS, patch_size, patch_size],
    );
    let embed_b = b.add_input("embed_b", &[HIDDEN_DIM]);

    let conv_out = b.add_conv2d(
        input,
        embed_w,
        Some(embed_b),
        patch_size,
        patch_size,
        0,
        0,
        &[HIDDEN_DIM, grid, grid],
    );
    // Reshape: [HIDDEN_DIM, grid, grid] -> [HIDDEN_DIM, num_patches]
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, num_patches]);
    // Transpose: [HIDDEN_DIM, num_patches] -> [num_patches, HIDDEN_DIM]
    let out = b.add_transpose(reshaped, &[1, 0], &[num_patches, HIDDEN_DIM]);
    let def = b.build(out).expect("valid SVTR patch embed");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, IN_CHANNELS, patch_size, patch_size]),
        bias_zero(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(IN_CHANNELS, IMG_SIZE, IMG_SIZE);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[num_patches, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Paddle SVTR patch embed IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 5. SVTR transformer self-attention bounds (CROWN)
// ===========================================================================

#[test]
fn test_paddle_pipeline_svtr_self_attention_crown() {
    let mut b = TensorBlockBuilder::new("paddle_svtr_attn");
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
    let def = b.build(attn).expect("valid SVTR attention");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 0.5);

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Paddle SVTR attention CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 6. SVTR feature projection bounds (IBP)
// ===========================================================================

#[test]
fn test_paddle_pipeline_svtr_feature_projection_ibp() {
    // Linear projection from encoder dim to a target dim
    let proj_dim = 8;
    let mut b = TensorBlockBuilder::new("paddle_svtr_proj");
    let input = b.add_input("encoder_out", &[SEQ_LEN, HIDDEN_DIM]);
    let proj_w = b.add_input("proj_w", &[proj_dim, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_b", &[proj_dim]);
    let out = b.add_linear(input, proj_w, Some(proj_b), &[SEQ_LEN, proj_dim]);
    let def = b.build(out).expect("valid SVTR feature projection");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[proj_dim, HIDDEN_DIM]),
        bias_zero(&[proj_dim]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, proj_dim]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Paddle SVTR feature projection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 7. CTC log-probability output bounds (IBP)
// ===========================================================================

#[test]
fn test_paddle_pipeline_ctc_log_prob_ibp() {
    // CTC head: Linear -> log_softmax
    let mut b = TensorBlockBuilder::new("paddle_ctc_log_prob");
    let input = b.add_input("encoder_out", &[SEQ_LEN, HIDDEN_DIM]);
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(input, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_log_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid CTC log-prob");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        bias_zero(&[VOCAB_SIZE]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Paddle CTC log-prob IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // log-softmax output must be <= 0
    assert!(
        hi_max <= 1e-5,
        "log-softmax upper bound must be <= 0, got {hi_max}"
    );
    assert!(lo_min.is_finite(), "lower bound must be finite");
}

// ===========================================================================
// 8. Text region crop bounds (IBP)
// ===========================================================================

#[test]
fn test_paddle_pipeline_text_region_crop_ibp() {
    // Simulate cropped text region: narrow from full feature map
    // [BACKBONE_CH, IMG_SIZE, IMG_SIZE] -> narrow spatial -> [BACKBONE_CH, 2, IMG_SIZE]
    let crop_h = 2;
    let mut b = TensorBlockBuilder::new("paddle_text_crop");
    let input = b.add_input("features", &[BACKBONE_CH, IMG_SIZE, IMG_SIZE]);
    let cropped = b.add_narrow(input, 1, 0, crop_h, &[BACKBONE_CH, crop_h, IMG_SIZE]);

    // Reshape to sequence: [BACKBONE_CH, crop_h * IMG_SIZE]
    let seq_len = crop_h * IMG_SIZE;
    let reshaped = b.add_reshape(cropped, &[BACKBONE_CH, seq_len]);
    let out = b.add_transpose(reshaped, &[1, 0], &[seq_len, BACKBONE_CH]);
    let def = b.build(out).expect("valid text region crop");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = uniform_bounds(&[BACKBONE_CH, IMG_SIZE, IMG_SIZE], 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[seq_len, BACKBONE_CH]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Paddle text region crop IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Narrow + reshape + transpose should preserve bounds width
    assert!(
        lo_min >= -1.0 - 1e-5,
        "crop should not widen bounds beyond input"
    );
    assert!(
        hi_max <= 1.0 + 1e-5,
        "crop should not widen bounds beyond input"
    );
}

// ===========================================================================
// 9. SVTR encoder feature bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_paddle_pipeline_svtr_encoder_block_ibp_crown() {
    // Full SVTR encoder block: LN -> MHA -> residual -> LN -> FFN(GELU) -> residual
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    let mut b = TensorBlockBuilder::new("paddle_svtr_encoder_block");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);

    // Pre-norm 1: LayerNorm
    let ln1_w = b.add_input("ln1_w", &[HIDDEN_DIM]);
    let ln1_b = b.add_input("ln1_b", &[HIDDEN_DIM]);
    let ln1_eps = b.add_input("ln1_eps", &[1]);
    let normed1 = b.add_layer_norm(input, ln1_eps, 1, ln1_w, ln1_b, &shape);

    // MHA
    let qw = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let kw = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let vw = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let ow = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let attn = b
        .add_multi_head_attention(
            normed1,
            qw,
            kw,
            vw,
            ow,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid MHA");

    // Residual 1
    let res1 = b.add_binary_add(input, attn, &shape);

    // Pre-norm 2: LayerNorm
    let ln2_w = b.add_input("ln2_w", &[HIDDEN_DIM]);
    let ln2_b = b.add_input("ln2_b", &[HIDDEN_DIM]);
    let ln2_eps = b.add_input("ln2_eps", &[1]);
    let normed2 = b.add_layer_norm(res1, ln2_eps, 1, ln2_w, ln2_b, &shape);

    // FFN: Linear -> GELU -> Linear
    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, HIDDEN_DIM]);
    let ffn2_w = b.add_input("ffn2_w", &[HIDDEN_DIM, FFN_DIM]);
    let ffn1 = b.add_linear(normed2, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(act, ffn2_w, None, &shape);

    // Residual 2
    let out = b.add_binary_add(res1, ffn2, &shape);
    let def = b.build(out).expect("valid SVTR encoder block");

    let bindings = vec![
        TensorParamBinding::Variable,
        ones(&[HIDDEN_DIM]),               // ln1_w
        bias_zero(&[HIDDEN_DIM]),          // ln1_b
        eps_binding(),                     // ln1_eps
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // Q
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // K
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // V
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // O
        ones(&[HIDDEN_DIM]),               // ln2_w
        bias_zero(&[HIDDEN_DIM]),          // ln2_b
        eps_binding(),                     // ln2_eps
        weight(&[FFN_DIM, HIDDEN_DIM]),    // ffn1_w
        weight(&[HIDDEN_DIM, FFN_DIM]),    // ffn2_w
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Paddle SVTR encoder block IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Paddle SVTR encoder block CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 10. CTC output length preservation (IBP)
// ===========================================================================

#[test]
fn test_paddle_pipeline_ctc_output_length_preservation_ibp() {
    // CTC preserves sequence length: [SEQ_LEN, HIDDEN_DIM] -> [SEQ_LEN, VOCAB_SIZE]
    let mut b = TensorBlockBuilder::new("paddle_ctc_length");
    let input = b.add_input("encoder_out", &[SEQ_LEN, HIDDEN_DIM]);
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let out = b.add_linear(input, ctc_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid CTC linear");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    // Verify sequence length dimension is preserved
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "CTC must preserve sequence length"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Paddle CTC length preservation IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 11. Detection pipeline end-to-end (IBP)
// ===========================================================================

#[test]
fn test_paddle_pipeline_detection_e2e_ibp() {
    // Full detection: Conv-BN-ReLU -> 1x1 conv -> sigmoid
    let mut b = TensorBlockBuilder::new("paddle_det_e2e");
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Backbone Conv-BN-ReLU
    let conv_w = b.add_input("conv_w", &[BACKBONE_CH, IN_CHANNELS, 3, 3]);
    let conv_b = b.add_input("conv_b", &[BACKBONE_CH]);
    let conv_out = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        1,
        1,
        1,
        1,
        &[BACKBONE_CH, IMG_SIZE, IMG_SIZE],
    );
    let bn_mean = b.add_input("bn_mean", &[BACKBONE_CH]);
    let bn_var = b.add_input("bn_var", &[BACKBONE_CH]);
    let bn_w = b.add_input("bn_w", &[BACKBONE_CH]);
    let bn_b = b.add_input("bn_b", &[BACKBONE_CH]);
    let bn_eps = b.add_input("bn_eps", &[1]);
    let bn_out = b.add_batch_norm(
        conv_out,
        bn_mean,
        bn_var,
        bn_w,
        bn_b,
        bn_eps,
        &[BACKBONE_CH, IMG_SIZE, IMG_SIZE],
    );
    let relu_out = b.add_relu(bn_out, &[BACKBONE_CH, IMG_SIZE, IMG_SIZE]);

    // Detection head: 1x1 conv -> sigmoid
    let head_w = b.add_input("det_w", &[1, BACKBONE_CH, 1, 1]);
    let head_b = b.add_input("det_b", &[1]);
    let head_out = b.add_conv2d(
        relu_out,
        head_w,
        Some(head_b),
        1,
        1,
        0,
        0,
        &[1, IMG_SIZE, IMG_SIZE],
    );
    let out = b.add_sigmoid(head_out, &[1, IMG_SIZE, IMG_SIZE]);
    let def = b.build(out).expect("valid detection e2e pipeline");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[BACKBONE_CH, IN_CHANNELS, 3, 3]),
        bias_zero(&[BACKBONE_CH]),
        bias_zero(&[BACKBONE_CH]), // bn mean
        ones(&[BACKBONE_CH]),      // bn var
        ones(&[BACKBONE_CH]),      // bn weight
        bias_zero(&[BACKBONE_CH]), // bn bias
        eps_binding(),
        weight(&[1, BACKBONE_CH, 1, 1]),
        bias_zero(&[1]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(IN_CHANNELS, IMG_SIZE, IMG_SIZE);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Paddle detection e2e IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lower must be >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-5,
        "sigmoid upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 12. Recognition confidence via softmax bounds (IBP)
// ===========================================================================

#[test]
fn test_paddle_pipeline_recognition_softmax_confidence_ibp() {
    // CTC head + softmax: confidence per character
    let mut b = TensorBlockBuilder::new("paddle_recog_confidence");
    let input = b.add_input("encoder_out", &[SEQ_LEN, HIDDEN_DIM]);
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(input, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid recognition softmax");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        bias_zero(&[VOCAB_SIZE]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Paddle recognition softmax IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower must be >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-5,
        "softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 13. NMS IoU filtering bounds (IBP)
// ===========================================================================

#[test]
fn test_paddle_pipeline_nms_iou_filtering_ibp() {
    // NMS modeled as sigmoid-based confidence filtering on detection scores
    // Score filtering: Linear -> sigmoid -> [0, 1] confidence
    let num_boxes = SEQ_LEN;
    let mut b = TensorBlockBuilder::new("paddle_nms_iou");
    let input = b.add_input("det_features", &[num_boxes, BACKBONE_CH]);
    let score_w = b.add_input("score_w", &[1, BACKBONE_CH]);
    let score_b = b.add_input("score_b", &[1]);
    let score_logits = b.add_linear(input, score_w, Some(score_b), &[num_boxes, 1]);
    let out = b.add_sigmoid(score_logits, &[num_boxes, 1]);
    let def = b.build(out).expect("valid NMS score filtering");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[1, BACKBONE_CH]),
        bias_zero(&[1]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = uniform_bounds(&[num_boxes, BACKBONE_CH], 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Paddle NMS IoU filtering IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid score must be >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-5,
        "sigmoid score must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 14. Text line grouping bounds (IBP)
// ===========================================================================

#[test]
fn test_paddle_pipeline_text_line_grouping_ibp() {
    // Text line grouping: aggregate detection features via linear projection
    // Simulate: detected regions pooled -> Linear projection -> features
    let num_lines = 2;
    let mut b = TensorBlockBuilder::new("paddle_line_grouping");
    let input = b.add_input("region_features", &[num_lines, BACKBONE_CH]);
    let group_w = b.add_input("group_w", &[HIDDEN_DIM, BACKBONE_CH]);
    let group_b = b.add_input("group_b", &[HIDDEN_DIM]);
    let proj = b.add_linear(input, group_w, Some(group_b), &[num_lines, HIDDEN_DIM]);
    let out = b.add_relu(proj, &[num_lines, HIDDEN_DIM]);
    let def = b.build(out).expect("valid text line grouping");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, BACKBONE_CH]),
        bias_zero(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = uniform_bounds(&[num_lines, BACKBONE_CH], 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[num_lines, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Paddle text line grouping IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "ReLU output must be non-negative");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 15. Multi-scale detection bounds (IBP)
// ===========================================================================

#[test]
fn test_paddle_pipeline_multi_scale_detection_ibp() {
    // Multi-scale detection: two conv stages at different resolutions
    let half_size = IMG_SIZE / 2; // 2

    let mut b = TensorBlockBuilder::new("paddle_multi_scale_det");
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Stage 1: stride-2 conv -> BN -> ReLU
    let s1_w = b.add_input("s1_w", &[BACKBONE_CH, IN_CHANNELS, 3, 3]);
    let s1_b = b.add_input("s1_b", &[BACKBONE_CH]);
    let s1_conv = b.add_conv2d(
        input,
        s1_w,
        Some(s1_b),
        2,
        2,
        1,
        1,
        &[BACKBONE_CH, half_size, half_size],
    );
    let s1_bn_mean = b.add_input("s1_bn_mean", &[BACKBONE_CH]);
    let s1_bn_var = b.add_input("s1_bn_var", &[BACKBONE_CH]);
    let s1_bn_w = b.add_input("s1_bn_w", &[BACKBONE_CH]);
    let s1_bn_b = b.add_input("s1_bn_b", &[BACKBONE_CH]);
    let s1_bn_eps = b.add_input("s1_bn_eps", &[1]);
    let s1_bn = b.add_batch_norm(
        s1_conv,
        s1_bn_mean,
        s1_bn_var,
        s1_bn_w,
        s1_bn_b,
        s1_bn_eps,
        &[BACKBONE_CH, half_size, half_size],
    );
    let s1_out = b.add_relu(s1_bn, &[BACKBONE_CH, half_size, half_size]);

    // Detection head on stage 1 features: 1x1 conv -> sigmoid
    let head_w = b.add_input("head_w", &[1, BACKBONE_CH, 1, 1]);
    let head_b = b.add_input("head_b", &[1]);
    let head_conv = b.add_conv2d(
        s1_out,
        head_w,
        Some(head_b),
        1,
        1,
        0,
        0,
        &[1, half_size, half_size],
    );
    let out = b.add_sigmoid(head_conv, &[1, half_size, half_size]);
    let def = b.build(out).expect("valid multi-scale detection");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[BACKBONE_CH, IN_CHANNELS, 3, 3]),
        bias_zero(&[BACKBONE_CH]),
        bias_zero(&[BACKBONE_CH]), // s1 bn mean
        ones(&[BACKBONE_CH]),      // s1 bn var
        ones(&[BACKBONE_CH]),      // s1 bn weight
        bias_zero(&[BACKBONE_CH]), // s1 bn bias
        eps_binding(),
        weight(&[1, BACKBONE_CH, 1, 1]),
        bias_zero(&[1]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(IN_CHANNELS, IMG_SIZE, IMG_SIZE);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[1, half_size, half_size]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Paddle multi-scale detection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lower must be >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-5,
        "sigmoid upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 16. Full pipeline composition (detection+recognition) (IBP)
// ===========================================================================

#[test]
fn test_paddle_pipeline_full_det_recog_composition_ibp() {
    // Full pipeline: detection backbone features -> recognition encoder -> CTC softmax
    // Simulated as: Linear(det features) -> LN -> Linear(CTC) -> softmax
    let mut b = TensorBlockBuilder::new("paddle_full_pipeline");
    let input = b.add_input("det_features", &[SEQ_LEN, BACKBONE_CH]);

    // Recognition encoder projection
    let enc_w = b.add_input("enc_w", &[HIDDEN_DIM, BACKBONE_CH]);
    let enc_b = b.add_input("enc_b", &[HIDDEN_DIM]);
    let enc_proj = b.add_linear(input, enc_w, Some(enc_b), &[SEQ_LEN, HIDDEN_DIM]);

    // LayerNorm
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let normed = b.add_layer_norm(enc_proj, ln_eps, 1, ln_w, ln_b, &[SEQ_LEN, HIDDEN_DIM]);

    // CTC head + softmax
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(normed, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid full det+recog pipeline");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, BACKBONE_CH]),
        bias_zero(&[HIDDEN_DIM]),
        ones(&[HIDDEN_DIM]),      // ln_w
        bias_zero(&[HIDDEN_DIM]), // ln_b
        eps_binding(),            // ln_eps
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        bias_zero(&[VOCAB_SIZE]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = uniform_bounds(&[SEQ_LEN, BACKBONE_CH], 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Paddle full det+recog pipeline IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower must be >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-5,
        "softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 17. Deep detection backbone bounds (IBP)
// ===========================================================================

#[test]
fn test_paddle_pipeline_deep_det_backbone_ibp() {
    // 2-layer detection backbone: Conv-BN-ReLU -> Conv-BN-ReLU
    let mut b = TensorBlockBuilder::new("paddle_deep_backbone");
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Layer 1: Conv-BN-ReLU
    let l1_w = b.add_input("l1_w", &[BACKBONE_CH, IN_CHANNELS, 3, 3]);
    let l1_b = b.add_input("l1_b", &[BACKBONE_CH]);
    let l1_conv = b.add_conv2d(
        input,
        l1_w,
        Some(l1_b),
        1,
        1,
        1,
        1,
        &[BACKBONE_CH, IMG_SIZE, IMG_SIZE],
    );
    let l1_bn_mean = b.add_input("l1_bn_mean", &[BACKBONE_CH]);
    let l1_bn_var = b.add_input("l1_bn_var", &[BACKBONE_CH]);
    let l1_bn_w = b.add_input("l1_bn_w", &[BACKBONE_CH]);
    let l1_bn_b = b.add_input("l1_bn_b", &[BACKBONE_CH]);
    let l1_bn_eps = b.add_input("l1_bn_eps", &[1]);
    let l1_bn = b.add_batch_norm(
        l1_conv,
        l1_bn_mean,
        l1_bn_var,
        l1_bn_w,
        l1_bn_b,
        l1_bn_eps,
        &[BACKBONE_CH, IMG_SIZE, IMG_SIZE],
    );
    let l1_out = b.add_relu(l1_bn, &[BACKBONE_CH, IMG_SIZE, IMG_SIZE]);

    // Layer 2: Conv-BN-ReLU
    let l2_w = b.add_input("l2_w", &[BACKBONE_CH, BACKBONE_CH, 3, 3]);
    let l2_b = b.add_input("l2_b", &[BACKBONE_CH]);
    let l2_conv = b.add_conv2d(
        l1_out,
        l2_w,
        Some(l2_b),
        1,
        1,
        1,
        1,
        &[BACKBONE_CH, IMG_SIZE, IMG_SIZE],
    );
    let l2_bn_mean = b.add_input("l2_bn_mean", &[BACKBONE_CH]);
    let l2_bn_var = b.add_input("l2_bn_var", &[BACKBONE_CH]);
    let l2_bn_w = b.add_input("l2_bn_w", &[BACKBONE_CH]);
    let l2_bn_b = b.add_input("l2_bn_b", &[BACKBONE_CH]);
    let l2_bn_eps = b.add_input("l2_bn_eps", &[1]);
    let l2_bn = b.add_batch_norm(
        l2_conv,
        l2_bn_mean,
        l2_bn_var,
        l2_bn_w,
        l2_bn_b,
        l2_bn_eps,
        &[BACKBONE_CH, IMG_SIZE, IMG_SIZE],
    );
    let out = b.add_relu(l2_bn, &[BACKBONE_CH, IMG_SIZE, IMG_SIZE]);
    let def = b.build(out).expect("valid deep detection backbone");

    let bindings = vec![
        TensorParamBinding::Variable,
        // Layer 1
        weight(&[BACKBONE_CH, IN_CHANNELS, 3, 3]),
        bias_zero(&[BACKBONE_CH]),
        bias_zero(&[BACKBONE_CH]), // l1 bn mean
        ones(&[BACKBONE_CH]),      // l1 bn var
        ones(&[BACKBONE_CH]),      // l1 bn weight
        bias_zero(&[BACKBONE_CH]), // l1 bn bias
        eps_binding(),
        // Layer 2
        weight(&[BACKBONE_CH, BACKBONE_CH, 3, 3]),
        bias_zero(&[BACKBONE_CH]),
        bias_zero(&[BACKBONE_CH]), // l2 bn mean
        ones(&[BACKBONE_CH]),      // l2 bn var
        ones(&[BACKBONE_CH]),      // l2 bn weight
        bias_zero(&[BACKBONE_CH]), // l2 bn bias
        eps_binding(),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(IN_CHANNELS, IMG_SIZE, IMG_SIZE);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Paddle deep backbone IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "ReLU output must be non-negative");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 18. SVTR attention + residual bounds (CROWN)
// ===========================================================================

#[test]
fn test_paddle_pipeline_svtr_attention_residual_crown() {
    // SVTR attention + residual: LN -> MHA -> add(input, attn)
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let mut b = TensorBlockBuilder::new("paddle_svtr_attn_residual");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);

    // LayerNorm
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let normed = b.add_layer_norm(input, ln_eps, 1, ln_w, ln_b, &shape);

    // MHA
    let qw = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let kw = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let vw = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let ow = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let attn = b
        .add_multi_head_attention(
            normed,
            qw,
            kw,
            vw,
            ow,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid MHA");

    // Residual
    let out = b.add_binary_add(input, attn, &shape);
    let def = b.build(out).expect("valid SVTR attention + residual");

    let bindings = vec![
        TensorParamBinding::Variable,
        ones(&[HIDDEN_DIM]),               // ln_w
        bias_zero(&[HIDDEN_DIM]),          // ln_b
        eps_binding(),                     // ln_eps
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // Q
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // K
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // V
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // O
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 0.5);

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Paddle SVTR attn+residual CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 19. CTC log-softmax score bounds (IBP)
// ===========================================================================

#[test]
fn test_paddle_pipeline_ctc_log_softmax_score_ibp() {
    // CTC scoring: Linear -> log_softmax for beam search
    let mut b = TensorBlockBuilder::new("paddle_ctc_log_softmax_score");
    let input = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);

    // LayerNorm before CTC head
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let normed = b.add_layer_norm(input, ln_eps, 1, ln_w, ln_b, &[SEQ_LEN, HIDDEN_DIM]);

    // CTC projection + log_softmax
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(normed, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_log_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid CTC log-softmax score");

    let bindings = vec![
        TensorParamBinding::Variable,
        ones(&[HIDDEN_DIM]),      // ln_w
        bias_zero(&[HIDDEN_DIM]), // ln_b
        eps_binding(),            // ln_eps
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        bias_zero(&[VOCAB_SIZE]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Paddle CTC log-softmax score IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // log-softmax output must be <= 0
    assert!(
        hi_max <= 1e-5,
        "log-softmax upper must be <= 0, got {hi_max}"
    );
    assert!(lo_min.is_finite(), "lower bound must be finite");
}

// ===========================================================================
// 20. Full pipeline with normalization (IBP + CROWN)
// ===========================================================================

#[test]
fn test_paddle_pipeline_full_with_normalization_ibp_crown() {
    // Full pipeline: LN -> encoder block -> LN -> CTC softmax
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    let mut b = TensorBlockBuilder::new("paddle_full_normed");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);

    // Input LayerNorm
    let ln0_w = b.add_input("ln0_w", &[HIDDEN_DIM]);
    let ln0_b = b.add_input("ln0_b", &[HIDDEN_DIM]);
    let ln0_eps = b.add_input("ln0_eps", &[1]);
    let normed0 = b.add_layer_norm(input, ln0_eps, 1, ln0_w, ln0_b, &shape);

    // Encoder block: LN -> MHA -> residual -> LN -> FFN -> residual
    let ln1_w = b.add_input("ln1_w", &[HIDDEN_DIM]);
    let ln1_b = b.add_input("ln1_b", &[HIDDEN_DIM]);
    let ln1_eps = b.add_input("ln1_eps", &[1]);
    let normed1 = b.add_layer_norm(normed0, ln1_eps, 1, ln1_w, ln1_b, &shape);

    let qw = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let kw = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let vw = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let ow = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let attn = b
        .add_multi_head_attention(
            normed1,
            qw,
            kw,
            vw,
            ow,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid MHA");
    let res1 = b.add_binary_add(normed0, attn, &shape);

    let ln2_w = b.add_input("ln2_w", &[HIDDEN_DIM]);
    let ln2_b = b.add_input("ln2_b", &[HIDDEN_DIM]);
    let ln2_eps = b.add_input("ln2_eps", &[1]);
    let normed2 = b.add_layer_norm(res1, ln2_eps, 1, ln2_w, ln2_b, &shape);

    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, HIDDEN_DIM]);
    let ffn2_w = b.add_input("ffn2_w", &[HIDDEN_DIM, FFN_DIM]);
    let ffn1 = b.add_linear(normed2, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(act, ffn2_w, None, &shape);
    let res2 = b.add_binary_add(res1, ffn2, &shape);

    // Final LayerNorm
    let ln3_w = b.add_input("ln3_w", &[HIDDEN_DIM]);
    let ln3_b = b.add_input("ln3_b", &[HIDDEN_DIM]);
    let ln3_eps = b.add_input("ln3_eps", &[1]);
    let normed3 = b.add_layer_norm(res2, ln3_eps, 1, ln3_w, ln3_b, &shape);

    // CTC head + softmax
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(normed3, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b
        .build(out)
        .expect("valid full pipeline with normalization");

    let bindings = vec![
        TensorParamBinding::Variable,
        // LN0
        ones(&[HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        eps_binding(),
        // LN1
        ones(&[HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        eps_binding(),
        // MHA: Q, K, V, O
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        // LN2
        ones(&[HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        eps_binding(),
        // FFN
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, FFN_DIM]),
        // LN3
        ones(&[HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        eps_binding(),
        // CTC
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        bias_zero(&[VOCAB_SIZE]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    assert_eq!(ibp_output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Paddle full normed pipeline IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower must be >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-5,
        "softmax upper must be <= 1, got {hi_max}"
    );

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Paddle full normed pipeline CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}
