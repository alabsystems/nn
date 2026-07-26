// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: PaddleOCR subgraph NY composition.
//!
//! Verifies bounds propagation through PaddleOCR sub-blocks used in the
//! dpdf document understanding pipeline for optical character recognition:
//!
//! 1. **DB Conv backbone IBP**: Conv2d -> BatchNorm -> ReLU backbone block
//!    from the DB (Differentiable Binarization) text detector.
//!
//! 2. **DB sigmoid output IBP**: Sigmoid binarization head producing
//!    probability map in [0, 1].
//!
//! 3. **SVTR patch embedding IBP**: Conv2d -> reshape -> transpose patch
//!    embedding for the SVTR (Scene Text Recognition) encoder.
//!
//! 4. **SVTR attention block CROWN**: Multi-head self-attention block in
//!    the SVTR encoder with LayerNorm + attention + residual.
//!
//! 5. **SVTR MLP GELU CROWN**: Feed-forward network with GELU activation
//!    in the SVTR transformer encoder.
//!
//! 6. **CTC linear head IBP**: Linear projection to vocabulary for CTC
//!    (Connectionist Temporal Classification) decoding.
//!
//! 7. **CTC softmax output IBP**: Softmax over vocabulary producing
//!    character probability distribution in [0, 1].
//!
//! 8. **Detection pipeline IBP**: Conv backbone -> sigmoid end-to-end.
//!
//! 9. **Recognition pipeline IBP**: Patch embed -> attention -> linear head.
//!
//! 10. **Full OCR pipeline IBP**: Detection + recognition composed.
//!
//! Architecture references:
//! - PaddleOCR (Baidu): Production OCR system with DB detector + SVTR recognizer
//! - DB (Liao et al. 2020): Differentiable Binarization for text detection
//! - SVTR (Du et al. 2022): Scene Text Recognition with a Single Visual Model
//! - CTC (Graves et al. 2006): Connectionist Temporal Classification
//!
//! Dimensions (small for fast verification):
//! - IMG_SIZE=32, PATCH_SIZE=8, HIDDEN_DIM=64, FFN_DIM=128, SEQ_LEN=16,
//!   NUM_HEADS=4, VOCAB_SIZE=256, BACKBONE_CH=32
//!
//! Part of #3894: NY compose tests for PaddleOCR subgraphs.

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

/// Image height and width (square image for DB detector input).
const IMG_SIZE: usize = 32;
/// Patch size for SVTR patch embedding.
const PATCH_SIZE: usize = 8;
/// Number of patches per spatial dimension.
const GRID_SIZE: usize = IMG_SIZE / PATCH_SIZE; // 4
/// Total number of patches.
const NUM_PATCHES: usize = GRID_SIZE * GRID_SIZE; // 16
/// Input channels (RGB).
const IN_CHANNELS: usize = 3;
/// Backbone output channels for DB detector.
const BACKBONE_CH: usize = 32;
/// Hidden dimension for SVTR transformer encoder.
const HIDDEN_DIM: usize = 64;
/// FFN intermediate dimension for SVTR MLP.
const FFN_DIM: usize = 128;
/// Sequence length for recognition (number of patches).
const SEQ_LEN: usize = NUM_PATCHES; // 16
/// Number of attention heads in SVTR.
const NUM_HEADS: usize = 4;
/// Head dimension = HIDDEN_DIM / NUM_HEADS.
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 16
/// Vocabulary size for CTC head.
const VOCAB_SIZE: usize = 256;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ===========================================================================
// 1. DB Conv backbone block: Conv2d -> BatchNorm -> ReLU
// ===========================================================================

/// Build a DB detector backbone conv block.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image pixels in [0, 1]).
/// Output: `[BACKBONE_CH, IMG_SIZE, IMG_SIZE]` (feature map).
///
/// Conv2d(3, 32, kernel=3, stride=1, padding=1) -> BatchNorm -> ReLU.
/// This is the first stage of the DB ResNet backbone.
fn build_db_conv_backbone_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_db_conv_backbone");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let conv_w = b.add_input("conv_weight", &[BACKBONE_CH, IN_CHANNELS, 3, 3]);
    let conv_bias = b.add_input("conv_bias", &[BACKBONE_CH]);

    // BatchNorm parameters
    let bn_mean = b.add_input("bn_running_mean", &[BACKBONE_CH]);
    let bn_var = b.add_input("bn_running_var", &[BACKBONE_CH]);
    let bn_weight = b.add_input("bn_weight", &[BACKBONE_CH]);
    let bn_bias = b.add_input("bn_bias", &[BACKBONE_CH]);
    let eps = b.add_input("eps", &[1]);

    let out_shape = [BACKBONE_CH, IMG_SIZE, IMG_SIZE];

    // Conv2d: [3, 32, 32] -> [32, 32, 32] (padding=1 preserves spatial dims)
    let conv_out = b.add_conv2d(input, conv_w, Some(conv_bias), 1, 1, 1, 1, &out_shape);

    // BatchNorm
    let bn_out = b.add_batch_norm(
        conv_out, bn_mean, bn_var, bn_weight, bn_bias, eps, &out_shape,
    );

    // ReLU
    let out = b.add_relu(bn_out, &out_shape);

    b.build(out)
        .expect("valid PaddleOCR DB conv backbone kernel")
}

/// Create image-domain input bounds: pixels in [0, 1].
fn image_bounds_01(shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), 0.0f32),
        ArrayD::from_elem(IxDyn(shape), 1.0f32),
    )
    .expect("valid image bounds [0, 1]")
}

/// Bindings for DB conv backbone.
fn db_conv_backbone_bindings() -> Vec<TensorParamBinding> {
    let conv_w = ArrayD::from_elem(IxDyn(&[BACKBONE_CH, IN_CHANNELS, 3, 3]), WEIGHT_MAG);
    let conv_bias = ArrayD::from_elem(IxDyn(&[BACKBONE_CH]), 0.0f32);
    let bn_mean = ArrayD::from_elem(IxDyn(&[BACKBONE_CH]), 0.0f32);
    let bn_var = ArrayD::from_elem(IxDyn(&[BACKBONE_CH]), 1.0f32);
    let bn_weight = ArrayD::from_elem(IxDyn(&[BACKBONE_CH]), 1.0f32);
    let bn_bias = ArrayD::from_elem(IxDyn(&[BACKBONE_CH]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                  // image [3, 32, 32]
        TensorParamBinding::ConstantTensor(conv_w),    // conv_weight
        TensorParamBinding::ConstantTensor(conv_bias), // conv_bias
        TensorParamBinding::ConstantTensor(bn_mean),   // bn_running_mean
        TensorParamBinding::ConstantTensor(bn_var),    // bn_running_var
        TensorParamBinding::ConstantTensor(bn_weight), // bn_weight
        TensorParamBinding::ConstantTensor(bn_bias),   // bn_bias
        TensorParamBinding::ConstantScalar(1e-5),      // eps
    ]
}

/// IBP bounds propagate through DB Conv backbone with [0, 1] image input.
#[test]
fn test_db_conv_backbone_ibp() {
    let def = build_db_conv_backbone_kernel();
    let bindings = db_conv_backbone_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR DB conv backbone");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[BACKBONE_CH, IMG_SIZE, IMG_SIZE],
        "DB conv backbone output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR DB conv backbone IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // ReLU clamps lower to 0
    assert!(
        lo_min >= -1e-4,
        "ReLU output lower should be >= 0, got {lo_min}"
    );
}

// ===========================================================================
// 2. DB sigmoid output: Linear -> Sigmoid probability map
// ===========================================================================

/// Build a DB detector sigmoid binarization head.
///
/// Input: `[BACKBONE_CH, IMG_SIZE, IMG_SIZE]` (Variable, feature map).
/// Output: `[1, IMG_SIZE, IMG_SIZE]` (probability map in [0, 1]).
///
/// Conv2d(32, 1, kernel=1) -> Sigmoid. Projects backbone features to a
/// single-channel probability map for text region binarization.
fn build_db_sigmoid_output_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_db_sigmoid_output");

    let input = b.add_input("features", &[BACKBONE_CH, IMG_SIZE, IMG_SIZE]);
    let conv_w = b.add_input("head_weight", &[1, BACKBONE_CH, 1, 1]);
    let conv_bias = b.add_input("head_bias", &[1]);

    let proj_shape = [1, IMG_SIZE, IMG_SIZE];

    // 1x1 conv projection: [32, 32, 32] -> [1, 32, 32]
    let proj = b.add_conv2d(input, conv_w, Some(conv_bias), 1, 1, 0, 0, &proj_shape);

    // Sigmoid: output in [0, 1]
    let out = b.add_sigmoid(proj, &proj_shape);

    b.build(out)
        .expect("valid PaddleOCR DB sigmoid output kernel")
}

/// Bindings for DB sigmoid output.
fn db_sigmoid_output_bindings() -> Vec<TensorParamBinding> {
    let conv_w = ArrayD::from_elem(IxDyn(&[1, BACKBONE_CH, 1, 1]), WEIGHT_MAG);
    let conv_bias = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                  // features
        TensorParamBinding::ConstantTensor(conv_w),    // head_weight
        TensorParamBinding::ConstantTensor(conv_bias), // head_bias
    ]
}

/// IBP bounds through DB sigmoid output: result must be in [0, 1].
#[test]
fn test_db_sigmoid_output_ibp() {
    let def = build_db_sigmoid_output_kernel();
    let bindings = db_sigmoid_output_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[BACKBONE_CH, IMG_SIZE, IMG_SIZE], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR DB sigmoid output");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[1, IMG_SIZE, IMG_SIZE],
        "DB sigmoid output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR DB sigmoid output IBP (features [-2,2]): bounds=[{lo_min}, {hi_max}]");

    // Sigmoid output must be in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "sigmoid output lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "sigmoid output upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 3. SVTR patch embedding: Conv2d -> reshape -> transpose
// ===========================================================================

/// Build an SVTR patch embedding kernel.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image pixels in [0, 1]).
/// Output: `[NUM_PATCHES, HIDDEN_DIM]` after reshape and transpose.
///
/// Conv2d(3, HIDDEN_DIM, kernel=PATCH_SIZE, stride=PATCH_SIZE) produces
/// `[HIDDEN_DIM, GRID_SIZE, GRID_SIZE]`. Reshape to `[HIDDEN_DIM, NUM_PATCHES]`,
/// then transpose to `[NUM_PATCHES, HIDDEN_DIM]`.
fn build_svtr_patch_embed_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_svtr_patch_embed");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let weight = b.add_input(
        "patch_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let bias = b.add_input("patch_bias", &[HIDDEN_DIM]);

    // Conv2d: [3, 32, 32] -> [64, 4, 4]
    let conv_out = b.add_conv2d(
        input,
        weight,
        Some(bias),
        PATCH_SIZE, // stride_h
        PATCH_SIZE, // stride_w
        0,          // padding_h
        0,          // padding_w
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );

    // Reshape: [64, 4, 4] -> [64, 16]
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, NUM_PATCHES]);

    // Transpose: [64, 16] -> [16, 64]
    let out = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, HIDDEN_DIM]);

    b.build(out)
        .expect("valid PaddleOCR SVTR patch embedding kernel")
}

/// Bindings for SVTR patch embedding.
fn svtr_patch_embed_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // image
        TensorParamBinding::ConstantTensor(w),    // patch_weight
        TensorParamBinding::ConstantTensor(bias), // patch_bias
    ]
}

/// IBP bounds propagate through SVTR patch embedding with [0, 1] image input.
#[test]
fn test_svtr_patch_embed_ibp() {
    let def = build_svtr_patch_embed_kernel();
    let bindings = svtr_patch_embed_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR SVTR patch embedding");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_PATCHES, HIDDEN_DIM],
        "SVTR patch embedding output shape should be [NUM_PATCHES={NUM_PATCHES}, HIDDEN_DIM={HIDDEN_DIM}]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR SVTR patch embedding IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

// ===========================================================================
// 4. SVTR attention block: LayerNorm -> Attention -> residual (CROWN)
// ===========================================================================

/// Build an SVTR self-attention block.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, patch embeddings).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Pre-norm transformer attention block:
///   x_norm = LayerNorm(x)
///   q = Linear(x_norm) -> [SEQ_LEN, HIDDEN_DIM]
///   k = Linear(x_norm) -> [SEQ_LEN, HIDDEN_DIM]
///   v = Linear(x_norm) -> [SEQ_LEN, HIDDEN_DIM]
///   attn_out = Attention(q, k, v)
///   output = x + Linear(attn_out)
fn build_svtr_attention_block_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_svtr_attention_block");

    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let ln_weight = b.add_input("ln_weight", &[HIDDEN_DIM]);
    let ln_bias = b.add_input("ln_bias", &[HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let q_w = b.add_input("q_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Pre-norm: LayerNorm
    let x_norm = b.add_layer_norm(input, eps, 1, ln_weight, ln_bias, &shape);

    // Q/K/V projections
    let q = b.add_linear(x_norm, q_w, None, &shape);
    let k = b.add_linear(x_norm, k_w, None, &shape);
    let v = b.add_linear(x_norm, v_w, None, &shape);

    // Self-attention with softmax
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn_out = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);

    // Output projection
    let proj_out = b.add_linear(attn_out, out_w, None, &shape);

    // Residual connection
    let out = b.add_binary_add(input, proj_out, &shape);

    b.build(out)
        .expect("valid PaddleOCR SVTR attention block kernel")
}

/// Bindings for SVTR attention block.
fn svtr_attention_block_bindings() -> Vec<TensorParamBinding> {
    let ln_weight = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let q_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                  // x
        TensorParamBinding::ConstantTensor(ln_weight), // ln_weight
        TensorParamBinding::ConstantTensor(ln_bias),   // ln_bias
        TensorParamBinding::ConstantScalar(1e-5),      // eps
        TensorParamBinding::ConstantTensor(q_w),       // q_proj_weight
        TensorParamBinding::ConstantTensor(k_w),       // k_proj_weight
        TensorParamBinding::ConstantTensor(v_w),       // v_proj_weight
        TensorParamBinding::ConstantTensor(out_w),     // out_proj_weight
    ]
}

/// CROWN bounds propagate through SVTR attention block.
///
/// LayerNorm requires CROWN linearization. Self-attention with softmax
/// uses McCormick envelope for bilinear terms. Residual connection
/// preserves bounded output.
#[test]
fn test_svtr_attention_block_crown() {
    let def = build_svtr_attention_block_kernel();
    let bindings = svtr_attention_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR SVTR attention block: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 5. SVTR MLP with GELU: Linear -> GELU -> Linear (CROWN)
// ===========================================================================

/// Build an SVTR MLP block with GELU activation.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Pre-norm FFN:
///   x_norm = LayerNorm(x)
///   hidden = GELU(Linear(x_norm))   [SEQ_LEN, FFN_DIM]
///   proj = Linear(hidden)           [SEQ_LEN, HIDDEN_DIM]
///   output = x + proj
fn build_svtr_mlp_gelu_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_svtr_mlp_gelu");

    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let ln_weight = b.add_input("ln_weight", &[HIDDEN_DIM]);
    let ln_bias = b.add_input("ln_bias", &[HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let fc1_w = b.add_input("fc1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let fc2_w = b.add_input("fc2_weight", &[HIDDEN_DIM, FFN_DIM]);

    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // Pre-norm: LayerNorm
    let x_norm = b.add_layer_norm(input, eps, 1, ln_weight, ln_bias, &shape);

    // Up projection -> GELU
    let hidden = b.add_linear(x_norm, fc1_w, None, &ffn_shape);
    let hidden_act = b.add_gelu(hidden, &ffn_shape);

    // Down projection
    let proj = b.add_linear(hidden_act, fc2_w, None, &shape);

    // Residual connection
    let out = b.add_binary_add(input, proj, &shape);

    b.build(out).expect("valid PaddleOCR SVTR MLP GELU kernel")
}

/// Bindings for SVTR MLP GELU.
fn svtr_mlp_gelu_bindings() -> Vec<TensorParamBinding> {
    let ln_weight = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                  // x
        TensorParamBinding::ConstantTensor(ln_weight), // ln_weight
        TensorParamBinding::ConstantTensor(ln_bias),   // ln_bias
        TensorParamBinding::ConstantScalar(1e-5),      // eps
        TensorParamBinding::ConstantTensor(fc1_w),     // fc1_weight
        TensorParamBinding::ConstantTensor(fc2_w),     // fc2_weight
    ]
}

/// CROWN bounds propagate through SVTR MLP with GELU.
///
/// GELU is piecewise-smooth and CROWN-friendly. LayerNorm requires
/// CROWN linearization via IbpValidated mode.
#[test]
fn test_svtr_mlp_gelu_crown() {
    let def = build_svtr_mlp_gelu_kernel();
    let bindings = svtr_mlp_gelu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR SVTR MLP GELU: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 6. CTC linear head: Linear projection to vocabulary
// ===========================================================================

/// Build a CTC linear head kernel.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, encoder output).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (logits per timestep).
///
/// Simple linear projection from hidden dimension to vocabulary size.
fn build_ctc_linear_head_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_ctc_linear_head");

    let input = b.add_input("encoder_output", &[SEQ_LEN, HIDDEN_DIM]);
    let weight = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);

    let out = b.add_linear(input, weight, Some(bias), &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid PaddleOCR CTC linear head kernel")
}

/// Bindings for CTC linear head.
fn ctc_linear_head_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // encoder_output
        TensorParamBinding::ConstantTensor(w),    // ctc_weight
        TensorParamBinding::ConstantTensor(bias), // ctc_bias
    ]
}

/// IBP bounds propagate through CTC linear head.
///
/// Pure linear layer: output bounds scale with weight * input range.
/// With 0.02 weights, [-2, 2] input, D=64: max output ~= 0.02 * 64 * 2 = 2.56.
#[test]
fn test_ctc_linear_head_ibp() {
    let def = build_ctc_linear_head_kernel();
    let bindings = ctc_linear_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR CTC linear head");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "CTC linear head output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR CTC linear head IBP (encoder [-2,2]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Linear with D=64, weight=0.02, input in [-2, 2]:
    // max output = sum(|w_i| * 2.0) = 64 * 0.02 * 2 = 2.56
    assert!(
        hi_max < 10.0,
        "CTC head upper should be < 10 with small weights, got {hi_max}"
    );
}

// ===========================================================================
// 7. CTC softmax output: Linear -> Softmax probability distribution
// ===========================================================================

/// Build a CTC softmax output kernel.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, encoder output).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (character probabilities in [0, 1]).
///
/// Linear head followed by softmax over the vocabulary dimension.
fn build_ctc_softmax_output_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_ctc_softmax_output");

    let input = b.add_input("encoder_output", &[SEQ_LEN, HIDDEN_DIM]);
    let weight = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);

    // Linear projection to logits
    let logits = b.add_linear(input, weight, Some(bias), &[SEQ_LEN, VOCAB_SIZE]);

    // Softmax over vocabulary dimension (axis=-1, i.e., axis 1 for 2D)
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid PaddleOCR CTC softmax output kernel")
}

/// Bindings for CTC softmax output.
fn ctc_softmax_output_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // encoder_output
        TensorParamBinding::ConstantTensor(w),    // ctc_weight
        TensorParamBinding::ConstantTensor(bias), // ctc_bias
    ]
}

/// IBP bounds through CTC softmax: output must be in [0, 1].
#[test]
fn test_ctc_softmax_output_ibp() {
    let def = build_ctc_softmax_output_kernel();
    let bindings = ctc_softmax_output_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR CTC softmax output");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "CTC softmax output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR CTC softmax output IBP (encoder [-2,2]): bounds=[{lo_min}, {hi_max}]");

    // Softmax output must be in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "softmax output lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "softmax output upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 8. Detection pipeline: Conv backbone -> sigmoid (end-to-end)
// ===========================================================================

/// Build the DB detection pipeline: Conv-BN-ReLU backbone -> sigmoid head.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image in [0, 1]).
/// Output: `[1, IMG_SIZE, IMG_SIZE]` (probability map in [0, 1]).
///
/// This is the end-to-end detection path: backbone feature extraction
/// followed by sigmoid binarization.
fn build_detection_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_detection_pipeline");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // --- Backbone: Conv2d -> BatchNorm -> ReLU ---
    let conv_w = b.add_input("conv_weight", &[BACKBONE_CH, IN_CHANNELS, 3, 3]);
    let conv_bias = b.add_input("conv_bias", &[BACKBONE_CH]);
    let bn_mean = b.add_input("bn_running_mean", &[BACKBONE_CH]);
    let bn_var = b.add_input("bn_running_var", &[BACKBONE_CH]);
    let bn_weight = b.add_input("bn_weight", &[BACKBONE_CH]);
    let bn_bias = b.add_input("bn_bias", &[BACKBONE_CH]);
    let eps = b.add_input("eps", &[1]);

    let feat_shape = [BACKBONE_CH, IMG_SIZE, IMG_SIZE];
    let conv_out = b.add_conv2d(input, conv_w, Some(conv_bias), 1, 1, 1, 1, &feat_shape);
    let bn_out = b.add_batch_norm(
        conv_out,
        bn_mean,
        bn_var,
        bn_weight,
        bn_bias,
        eps,
        &feat_shape,
    );
    let relu_out = b.add_relu(bn_out, &feat_shape);

    // --- Head: 1x1 Conv -> Sigmoid ---
    let head_w = b.add_input("head_weight", &[1, BACKBONE_CH, 1, 1]);
    let head_bias = b.add_input("head_bias", &[1]);

    let out_shape = [1, IMG_SIZE, IMG_SIZE];
    let proj = b.add_conv2d(relu_out, head_w, Some(head_bias), 1, 1, 0, 0, &out_shape);
    let out = b.add_sigmoid(proj, &out_shape);

    b.build(out)
        .expect("valid PaddleOCR detection pipeline kernel")
}

/// Bindings for detection pipeline.
fn detection_pipeline_bindings() -> Vec<TensorParamBinding> {
    let conv_w = ArrayD::from_elem(IxDyn(&[BACKBONE_CH, IN_CHANNELS, 3, 3]), WEIGHT_MAG);
    let conv_bias = ArrayD::from_elem(IxDyn(&[BACKBONE_CH]), 0.0f32);
    let bn_mean = ArrayD::from_elem(IxDyn(&[BACKBONE_CH]), 0.0f32);
    let bn_var = ArrayD::from_elem(IxDyn(&[BACKBONE_CH]), 1.0f32);
    let bn_weight = ArrayD::from_elem(IxDyn(&[BACKBONE_CH]), 1.0f32);
    let bn_bias = ArrayD::from_elem(IxDyn(&[BACKBONE_CH]), 0.0f32);
    let head_w = ArrayD::from_elem(IxDyn(&[1, BACKBONE_CH, 1, 1]), WEIGHT_MAG);
    let head_bias = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                  // image
        TensorParamBinding::ConstantTensor(conv_w),    // conv_weight
        TensorParamBinding::ConstantTensor(conv_bias), // conv_bias
        TensorParamBinding::ConstantTensor(bn_mean),   // bn_running_mean
        TensorParamBinding::ConstantTensor(bn_var),    // bn_running_var
        TensorParamBinding::ConstantTensor(bn_weight), // bn_weight
        TensorParamBinding::ConstantTensor(bn_bias),   // bn_bias
        TensorParamBinding::ConstantScalar(1e-5),      // eps
        TensorParamBinding::ConstantTensor(head_w),    // head_weight
        TensorParamBinding::ConstantTensor(head_bias), // head_bias
    ]
}

/// IBP through full detection pipeline: image [0,1] -> probability [0,1].
#[test]
fn test_detection_pipeline_ibp() {
    let def = build_detection_pipeline_kernel();
    let bindings = detection_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR detection pipeline");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[1, IMG_SIZE, IMG_SIZE],
        "detection pipeline output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR detection pipeline IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    // End-to-end: sigmoid clamps to [0, 1]
    assert!(
        lo_min >= -1e-4,
        "detection output lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "detection output upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 9. Recognition pipeline: Patch embed -> MLP -> Linear head
// ===========================================================================

/// Build a simplified recognition pipeline.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image in [0, 1]).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (character logits).
///
/// Patch embedding -> MLP encoder block (Linear -> GELU -> Linear) ->
/// CTC linear head. Uses a single MLP block instead of full attention
/// for IBP tractability.
fn build_recognition_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_recognition_pipeline");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // --- Patch embedding: Conv2d -> reshape -> transpose ---
    let patch_w = b.add_input(
        "patch_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_bias = b.add_input("patch_bias", &[HIDDEN_DIM]);

    let conv_out = b.add_conv2d(
        input,
        patch_w,
        Some(patch_bias),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, NUM_PATCHES]);
    let patches = b.add_transpose(reshaped, &[1, 0], &[SEQ_LEN, HIDDEN_DIM]);

    // --- MLP encoder block: Linear -> GELU -> Linear ---
    let fc1_w = b.add_input("fc1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let fc2_w = b.add_input("fc2_weight", &[HIDDEN_DIM, FFN_DIM]);

    let hidden = b.add_linear(patches, fc1_w, None, &[SEQ_LEN, FFN_DIM]);
    let hidden_act = b.add_gelu(hidden, &[SEQ_LEN, FFN_DIM]);
    let encoded = b.add_linear(hidden_act, fc2_w, None, &[SEQ_LEN, HIDDEN_DIM]);

    // Residual
    let enc_out = b.add_binary_add(patches, encoded, &[SEQ_LEN, HIDDEN_DIM]);

    // --- CTC head: Linear ---
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);

    let out = b.add_linear(enc_out, ctc_w, Some(ctc_bias), &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid PaddleOCR recognition pipeline kernel")
}

/// Bindings for recognition pipeline.
fn recognition_pipeline_bindings() -> Vec<TensorParamBinding> {
    let patch_w = ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let patch_bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let ctc_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let ctc_bias = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                   // image
        TensorParamBinding::ConstantTensor(patch_w),    // patch_weight
        TensorParamBinding::ConstantTensor(patch_bias), // patch_bias
        TensorParamBinding::ConstantTensor(fc1_w),      // fc1_weight
        TensorParamBinding::ConstantTensor(fc2_w),      // fc2_weight
        TensorParamBinding::ConstantTensor(ctc_w),      // ctc_weight
        TensorParamBinding::ConstantTensor(ctc_bias),   // ctc_bias
    ]
}

/// IBP through recognition pipeline: image [0,1] -> character logits.
#[test]
fn test_recognition_pipeline_ibp() {
    let def = build_recognition_pipeline_kernel();
    let bindings = recognition_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR recognition pipeline");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "recognition pipeline output shape should be [SEQ_LEN={SEQ_LEN}, VOCAB_SIZE={VOCAB_SIZE}]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR recognition pipeline IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

// ===========================================================================
// 10. Full OCR pipeline: Detection sigmoid + Recognition softmax
// ===========================================================================

/// Build a full OCR pipeline combining detection and recognition.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image in [0, 1]).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (character probabilities).
///
/// This composes the detection backbone (Conv-BN-ReLU) features into
/// the recognition path. In production PaddleOCR, detection crops text
/// regions which are then fed to recognition. For verification, we model
/// the recognition directly on the image (conservative: full image has
/// wider bounds than a cropped text region).
///
/// Path: Conv backbone -> channel projection -> flatten -> Linear -> GELU ->
///       Linear -> CTC head -> Softmax.
fn build_full_ocr_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_full_pipeline");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // --- Stage 1: Backbone Conv-BN-ReLU ---
    let conv_w = b.add_input("conv_weight", &[BACKBONE_CH, IN_CHANNELS, 3, 3]);
    let conv_bias = b.add_input("conv_bias", &[BACKBONE_CH]);
    let bn_mean = b.add_input("bn_running_mean", &[BACKBONE_CH]);
    let bn_var = b.add_input("bn_running_var", &[BACKBONE_CH]);
    let bn_weight = b.add_input("bn_weight", &[BACKBONE_CH]);
    let bn_bias = b.add_input("bn_bias", &[BACKBONE_CH]);
    let eps = b.add_input("eps", &[1]);

    let feat_shape = [BACKBONE_CH, IMG_SIZE, IMG_SIZE];
    let conv_out = b.add_conv2d(input, conv_w, Some(conv_bias), 1, 1, 1, 1, &feat_shape);
    let bn_out = b.add_batch_norm(
        conv_out,
        bn_mean,
        bn_var,
        bn_weight,
        bn_bias,
        eps,
        &feat_shape,
    );
    let features = b.add_relu(bn_out, &feat_shape);

    // --- Stage 2: Channel reduction + spatial flatten ---
    // Project: [BACKBONE_CH, 32, 32] -> [HIDDEN_DIM, 32, 32] via 1x1 conv
    let proj_w = b.add_input("proj_weight", &[HIDDEN_DIM, BACKBONE_CH, 1, 1]);
    let proj_out = b.add_conv2d(
        features,
        proj_w,
        None,
        1,
        1,
        0,
        0,
        &[HIDDEN_DIM, IMG_SIZE, IMG_SIZE],
    );

    // Reshape: [HIDDEN_DIM, 32, 32] -> [HIDDEN_DIM, 1024]
    let spatial_size = IMG_SIZE * IMG_SIZE;
    let reshaped = b.add_reshape(proj_out, &[HIDDEN_DIM, spatial_size]);

    // Transpose: [HIDDEN_DIM, 1024] -> [1024, HIDDEN_DIM]
    // Then narrow to SEQ_LEN positions for tractability
    let transposed = b.add_transpose(reshaped, &[1, 0], &[spatial_size, HIDDEN_DIM]);
    let seq = b.add_narrow(transposed, 0, 0, SEQ_LEN, &[SEQ_LEN, HIDDEN_DIM]);

    // --- Stage 3: MLP encoder ---
    let fc1_w = b.add_input("fc1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let fc2_w = b.add_input("fc2_weight", &[HIDDEN_DIM, FFN_DIM]);

    let hidden = b.add_linear(seq, fc1_w, None, &[SEQ_LEN, FFN_DIM]);
    let hidden_act = b.add_gelu(hidden, &[SEQ_LEN, FFN_DIM]);
    let encoded = b.add_linear(hidden_act, fc2_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let enc_out = b.add_binary_add(seq, encoded, &[SEQ_LEN, HIDDEN_DIM]);

    // --- Stage 4: CTC head + softmax ---
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);

    let logits = b.add_linear(enc_out, ctc_w, Some(ctc_bias), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid PaddleOCR full OCR pipeline kernel")
}

/// Bindings for full OCR pipeline.
fn full_ocr_pipeline_bindings() -> Vec<TensorParamBinding> {
    let conv_w = ArrayD::from_elem(IxDyn(&[BACKBONE_CH, IN_CHANNELS, 3, 3]), WEIGHT_MAG);
    let conv_bias = ArrayD::from_elem(IxDyn(&[BACKBONE_CH]), 0.0f32);
    let bn_mean = ArrayD::from_elem(IxDyn(&[BACKBONE_CH]), 0.0f32);
    let bn_var = ArrayD::from_elem(IxDyn(&[BACKBONE_CH]), 1.0f32);
    let bn_weight = ArrayD::from_elem(IxDyn(&[BACKBONE_CH]), 1.0f32);
    let bn_bias = ArrayD::from_elem(IxDyn(&[BACKBONE_CH]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, BACKBONE_CH, 1, 1]), WEIGHT_MAG);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let ctc_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let ctc_bias = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                  // image
        TensorParamBinding::ConstantTensor(conv_w),    // conv_weight
        TensorParamBinding::ConstantTensor(conv_bias), // conv_bias
        TensorParamBinding::ConstantTensor(bn_mean),   // bn_running_mean
        TensorParamBinding::ConstantTensor(bn_var),    // bn_running_var
        TensorParamBinding::ConstantTensor(bn_weight), // bn_weight
        TensorParamBinding::ConstantTensor(bn_bias),   // bn_bias
        TensorParamBinding::ConstantScalar(1e-5),      // eps
        TensorParamBinding::ConstantTensor(proj_w),    // proj_weight
        TensorParamBinding::ConstantTensor(fc1_w),     // fc1_weight
        TensorParamBinding::ConstantTensor(fc2_w),     // fc2_weight
        TensorParamBinding::ConstantTensor(ctc_w),     // ctc_weight
        TensorParamBinding::ConstantTensor(ctc_bias),  // ctc_bias
    ]
}

/// IBP through the full OCR pipeline: image [0,1] -> character probs [0,1].
#[test]
fn test_full_ocr_pipeline_ibp() {
    let def = build_full_ocr_pipeline_kernel();
    let bindings = full_ocr_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR full OCR pipeline");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "full OCR pipeline output shape should be [SEQ_LEN={SEQ_LEN}, VOCAB_SIZE={VOCAB_SIZE}]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR full OCR pipeline IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    // Softmax output must be in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "full pipeline output lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "full pipeline output upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// Verify-and-record tests for status tracking
// ===========================================================================

/// Verify and record DB conv backbone.
#[test]
fn test_db_conv_backbone_verify_and_record() {
    let def = build_db_conv_backbone_kernel();
    let bindings = db_conv_backbone_bindings();
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let result = verify_and_assert(&def, &bindings, &input, "paddle_ocr_db_conv_backbone");
    assert_eq!(result.num_variables, 1, "single Variable input (image)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[BACKBONE_CH, IMG_SIZE, IMG_SIZE]);
}

/// Verify and record SVTR patch embedding.
#[test]
fn test_svtr_patch_embed_verify_and_record() {
    let def = build_svtr_patch_embed_kernel();
    let bindings = svtr_patch_embed_bindings();
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let result = verify_and_assert(&def, &bindings, &input, "paddle_ocr_svtr_patch_embed");
    assert_eq!(result.num_variables, 1, "single Variable input (image)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_PATCHES, HIDDEN_DIM]);
}

/// Verify and record CTC softmax output.
#[test]
fn test_ctc_softmax_verify_and_record() {
    let def = build_ctc_softmax_output_kernel();
    let bindings = ctc_softmax_output_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);

    let result = verify_and_assert(&def, &bindings, &input, "paddle_ocr_ctc_softmax_output");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, VOCAB_SIZE]);
}

// ===========================================================================
// 11. SVTR two-block encoder: Attention + MLP chained x2
// ===========================================================================

/// Build a 2-block SVTR encoder: (LayerNorm -> Attention -> residual ->
/// LayerNorm -> MLP GELU -> residual) x 2.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, patch embeddings).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Tests depth-2 composition of transformer blocks through the SVTR
/// encoder, exercising repeated LayerNorm + attention + GELU paths.
fn build_svtr_two_block_encoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_svtr_two_block_encoder");

    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // --- Block 1: Attention ---
    let ln1a_w = b.add_input("b1_ln1_weight", &[HIDDEN_DIM]);
    let ln1a_b = b.add_input("b1_ln1_bias", &[HIDDEN_DIM]);
    let ln1a_eps = b.add_input("b1_ln1_eps", &[1]);
    let b1_qw = b.add_input("b1_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b1_kw = b.add_input("b1_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b1_vw = b.add_input("b1_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b1_ow = b.add_input("b1_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let x_norm1a = b.add_layer_norm(input, ln1a_eps, 1, ln1a_w, ln1a_b, &shape);
    let q1 = b.add_linear(x_norm1a, b1_qw, None, &shape);
    let k1 = b.add_linear(x_norm1a, b1_kw, None, &shape);
    let v1 = b.add_linear(x_norm1a, b1_vw, None, &shape);
    let attn1 = b.add_attention(q1, k1, v1, AttentionMask::Standard, Some(scale), &shape);
    let proj1 = b.add_linear(attn1, b1_ow, None, &shape);
    let res1a = b.add_binary_add(input, proj1, &shape);

    // --- Block 1: MLP ---
    let ln1b_w = b.add_input("b1_ln2_weight", &[HIDDEN_DIM]);
    let ln1b_b = b.add_input("b1_ln2_bias", &[HIDDEN_DIM]);
    let ln1b_eps = b.add_input("b1_ln2_eps", &[1]);
    let b1_fc1 = b.add_input("b1_fc1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let b1_fc2 = b.add_input("b1_fc2_weight", &[HIDDEN_DIM, FFN_DIM]);

    let x_norm1b = b.add_layer_norm(res1a, ln1b_eps, 1, ln1b_w, ln1b_b, &shape);
    let h1 = b.add_linear(x_norm1b, b1_fc1, None, &ffn_shape);
    let h1_act = b.add_gelu(h1, &ffn_shape);
    let mlp1 = b.add_linear(h1_act, b1_fc2, None, &shape);
    let res1b = b.add_binary_add(res1a, mlp1, &shape);

    // --- Block 2: Attention ---
    let ln2a_w = b.add_input("b2_ln1_weight", &[HIDDEN_DIM]);
    let ln2a_b = b.add_input("b2_ln1_bias", &[HIDDEN_DIM]);
    let ln2a_eps = b.add_input("b2_ln1_eps", &[1]);
    let b2_qw = b.add_input("b2_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b2_kw = b.add_input("b2_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b2_vw = b.add_input("b2_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b2_ow = b.add_input("b2_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let x_norm2a = b.add_layer_norm(res1b, ln2a_eps, 1, ln2a_w, ln2a_b, &shape);
    let q2 = b.add_linear(x_norm2a, b2_qw, None, &shape);
    let k2 = b.add_linear(x_norm2a, b2_kw, None, &shape);
    let v2 = b.add_linear(x_norm2a, b2_vw, None, &shape);
    let attn2 = b.add_attention(q2, k2, v2, AttentionMask::Standard, Some(scale), &shape);
    let proj2 = b.add_linear(attn2, b2_ow, None, &shape);
    let res2a = b.add_binary_add(res1b, proj2, &shape);

    // --- Block 2: MLP ---
    let ln2b_w = b.add_input("b2_ln2_weight", &[HIDDEN_DIM]);
    let ln2b_b = b.add_input("b2_ln2_bias", &[HIDDEN_DIM]);
    let ln2b_eps = b.add_input("b2_ln2_eps", &[1]);
    let b2_fc1 = b.add_input("b2_fc1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let b2_fc2 = b.add_input("b2_fc2_weight", &[HIDDEN_DIM, FFN_DIM]);

    let x_norm2b = b.add_layer_norm(res2a, ln2b_eps, 1, ln2b_w, ln2b_b, &shape);
    let h2 = b.add_linear(x_norm2b, b2_fc1, None, &ffn_shape);
    let h2_act = b.add_gelu(h2, &ffn_shape);
    let mlp2 = b.add_linear(h2_act, b2_fc2, None, &shape);
    let out = b.add_binary_add(res2a, mlp2, &shape);

    b.build(out)
        .expect("valid PaddleOCR SVTR two-block encoder kernel")
}

/// Bindings for SVTR two-block encoder.
fn svtr_two_block_encoder_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // x

    // Block 1
    bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone())); // b1_ln1_weight
    bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone())); // b1_ln1_bias
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // b1_ln1_eps
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone())); // b1_q_weight
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone())); // b1_k_weight
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone())); // b1_v_weight
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone())); // b1_out_weight
    bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone())); // b1_ln2_weight
    bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone())); // b1_ln2_bias
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // b1_ln2_eps
    bindings.push(TensorParamBinding::ConstantTensor(fc1_w.clone())); // b1_fc1_weight
    bindings.push(TensorParamBinding::ConstantTensor(fc2_w.clone())); // b1_fc2_weight

    // Block 2
    bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone())); // b2_ln1_weight
    bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone())); // b2_ln1_bias
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // b2_ln1_eps
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone())); // b2_q_weight
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone())); // b2_k_weight
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone())); // b2_v_weight
    bindings.push(TensorParamBinding::ConstantTensor(proj_w)); // b2_out_weight
    bindings.push(TensorParamBinding::ConstantTensor(ln_w)); // b2_ln2_weight
    bindings.push(TensorParamBinding::ConstantTensor(ln_b)); // b2_ln2_bias
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // b2_ln2_eps
    bindings.push(TensorParamBinding::ConstantTensor(fc1_w)); // b2_fc1_weight
    bindings.push(TensorParamBinding::ConstantTensor(fc2_w)); // b2_fc2_weight

    bindings
}

/// IBP bounds propagate through 2 sequential SVTR encoder blocks.
#[test]
fn test_svtr_two_block_encoder_ibp() {
    let def = build_svtr_two_block_encoder_kernel();
    let bindings = svtr_two_block_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR SVTR two-block encoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "SVTR two-block encoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR SVTR two-block encoder IBP (input [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

// ===========================================================================
// 12. SVTR four-block encoder: deeper CROWN test
// ===========================================================================

/// Build a 4-block SVTR encoder for deeper CROWN verification.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Same per-block structure as test 11 but 4 blocks deep.
/// Tests CROWN linearization stability through repeated LayerNorm
/// and attention softmax layers.
fn build_svtr_four_block_encoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_svtr_four_block_encoder");

    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut current = input;

    for block_idx in 0..4 {
        let prefix = format!("b{block_idx}");

        // Attention sub-block
        let ln_a_w = b.add_input(&format!("{prefix}_ln1_w"), &[HIDDEN_DIM]);
        let ln_a_b = b.add_input(&format!("{prefix}_ln1_b"), &[HIDDEN_DIM]);
        let ln_a_eps = b.add_input(&format!("{prefix}_ln1_eps"), &[1]);
        let qw = b.add_input(&format!("{prefix}_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let kw = b.add_input(&format!("{prefix}_k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let vw = b.add_input(&format!("{prefix}_v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let ow = b.add_input(&format!("{prefix}_o_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let normed = b.add_layer_norm(current, ln_a_eps, 1, ln_a_w, ln_a_b, &shape);
        let q = b.add_linear(normed, qw, None, &shape);
        let k = b.add_linear(normed, kw, None, &shape);
        let v = b.add_linear(normed, vw, None, &shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
        let proj = b.add_linear(attn, ow, None, &shape);
        let res_a = b.add_binary_add(current, proj, &shape);

        // MLP sub-block
        let ln_b_w = b.add_input(&format!("{prefix}_ln2_w"), &[HIDDEN_DIM]);
        let ln_b_b = b.add_input(&format!("{prefix}_ln2_b"), &[HIDDEN_DIM]);
        let ln_b_eps = b.add_input(&format!("{prefix}_ln2_eps"), &[1]);
        let fc1 = b.add_input(&format!("{prefix}_fc1_w"), &[FFN_DIM, HIDDEN_DIM]);
        let fc2 = b.add_input(&format!("{prefix}_fc2_w"), &[HIDDEN_DIM, FFN_DIM]);

        let normed_b = b.add_layer_norm(res_a, ln_b_eps, 1, ln_b_w, ln_b_b, &shape);
        let h = b.add_linear(normed_b, fc1, None, &ffn_shape);
        let h_act = b.add_gelu(h, &ffn_shape);
        let mlp = b.add_linear(h_act, fc2, None, &shape);
        current = b.add_binary_add(res_a, mlp, &shape);
    }

    b.build(current)
        .expect("valid PaddleOCR SVTR four-block encoder kernel")
}

/// Bindings for SVTR four-block encoder.
fn svtr_four_block_encoder_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // x

    for _ in 0..4 {
        // Attention sub-block
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        // MLP sub-block
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(fc1_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(fc2_w.clone()));
    }

    bindings
}

/// IBP bounds propagate through 4 sequential SVTR encoder blocks.
///
/// Deeper encoder tests bounds growth through repeated residual
/// connections and normalization layers.
#[test]
fn test_svtr_four_block_encoder_ibp() {
    let def = build_svtr_four_block_encoder_kernel();
    let bindings = svtr_four_block_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR SVTR four-block encoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "SVTR four-block encoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR SVTR four-block encoder IBP (input [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 13. DB backbone three-stage: Conv-BN-ReLU at different spatial scales
// ===========================================================================

/// Intermediate channel count for stages 2 and 3.
const STAGE2_CH: usize = 64;
const STAGE3_CH: usize = 128;
/// Spatial dimensions after stride-2 downsampling.
const HALF_IMG: usize = IMG_SIZE / 2; // 16
const QUARTER_IMG: usize = IMG_SIZE / 4; // 8

/// Build a 3-stage DB backbone: Conv-BN-ReLU at progressively
/// smaller spatial resolutions.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image in [0, 1]).
/// Output: `[STAGE3_CH, QUARTER_IMG, QUARTER_IMG]`.
///
/// Stage 1: Conv(3, 32, k=3, s=1, p=1) -> BN -> ReLU (preserve spatial)
/// Stage 2: Conv(32, 64, k=3, s=2, p=1) -> BN -> ReLU (downsample 2x)
/// Stage 3: Conv(64, 128, k=3, s=2, p=1) -> BN -> ReLU (downsample 2x)
fn build_db_backbone_three_stage_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_db_backbone_three_stage");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // --- Stage 1: [3, 32, 32] -> [32, 32, 32] ---
    let s1_cw = b.add_input("s1_conv_w", &[BACKBONE_CH, IN_CHANNELS, 3, 3]);
    let s1_cb = b.add_input("s1_conv_b", &[BACKBONE_CH]);
    let s1_bm = b.add_input("s1_bn_mean", &[BACKBONE_CH]);
    let s1_bv = b.add_input("s1_bn_var", &[BACKBONE_CH]);
    let s1_bw = b.add_input("s1_bn_w", &[BACKBONE_CH]);
    let s1_bb = b.add_input("s1_bn_b", &[BACKBONE_CH]);
    let s1_eps = b.add_input("s1_eps", &[1]);

    let s1_shape = [BACKBONE_CH, IMG_SIZE, IMG_SIZE];
    let s1_conv = b.add_conv2d(input, s1_cw, Some(s1_cb), 1, 1, 1, 1, &s1_shape);
    let s1_bn = b.add_batch_norm(s1_conv, s1_bm, s1_bv, s1_bw, s1_bb, s1_eps, &s1_shape);
    let s1_out = b.add_relu(s1_bn, &s1_shape);

    // --- Stage 2: [32, 32, 32] -> [64, 16, 16] ---
    let s2_cw = b.add_input("s2_conv_w", &[STAGE2_CH, BACKBONE_CH, 3, 3]);
    let s2_cb = b.add_input("s2_conv_b", &[STAGE2_CH]);
    let s2_bm = b.add_input("s2_bn_mean", &[STAGE2_CH]);
    let s2_bv = b.add_input("s2_bn_var", &[STAGE2_CH]);
    let s2_bw = b.add_input("s2_bn_w", &[STAGE2_CH]);
    let s2_bb = b.add_input("s2_bn_b", &[STAGE2_CH]);
    let s2_eps = b.add_input("s2_eps", &[1]);

    let s2_shape = [STAGE2_CH, HALF_IMG, HALF_IMG];
    let s2_conv = b.add_conv2d(s1_out, s2_cw, Some(s2_cb), 2, 2, 1, 1, &s2_shape);
    let s2_bn = b.add_batch_norm(s2_conv, s2_bm, s2_bv, s2_bw, s2_bb, s2_eps, &s2_shape);
    let s2_out = b.add_relu(s2_bn, &s2_shape);

    // --- Stage 3: [64, 16, 16] -> [128, 8, 8] ---
    let s3_cw = b.add_input("s3_conv_w", &[STAGE3_CH, STAGE2_CH, 3, 3]);
    let s3_cb = b.add_input("s3_conv_b", &[STAGE3_CH]);
    let s3_bm = b.add_input("s3_bn_mean", &[STAGE3_CH]);
    let s3_bv = b.add_input("s3_bn_var", &[STAGE3_CH]);
    let s3_bw = b.add_input("s3_bn_w", &[STAGE3_CH]);
    let s3_bb = b.add_input("s3_bn_b", &[STAGE3_CH]);
    let s3_eps = b.add_input("s3_eps", &[1]);

    let s3_shape = [STAGE3_CH, QUARTER_IMG, QUARTER_IMG];
    let s3_conv = b.add_conv2d(s2_out, s3_cw, Some(s3_cb), 2, 2, 1, 1, &s3_shape);
    let s3_bn = b.add_batch_norm(s3_conv, s3_bm, s3_bv, s3_bw, s3_bb, s3_eps, &s3_shape);
    let out = b.add_relu(s3_bn, &s3_shape);

    b.build(out)
        .expect("valid PaddleOCR DB backbone three-stage kernel")
}

/// Bindings for DB backbone three-stage.
fn db_backbone_three_stage_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // image

    // Stage 1
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH, IN_CHANNELS, 3, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // Stage 2
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH, BACKBONE_CH, 3, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // Stage 3
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE3_CH, STAGE2_CH, 3, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE3_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE3_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE3_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE3_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE3_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    bindings
}

/// IBP through 3-stage DB backbone at different spatial scales.
///
/// Each stage halves spatial resolution while increasing channels.
/// ReLU at each stage clips negative values, bounding growth.
#[test]
fn test_db_backbone_three_stage_ibp() {
    let def = build_db_backbone_three_stage_kernel();
    let bindings = db_backbone_three_stage_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR DB backbone three-stage");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[STAGE3_CH, QUARTER_IMG, QUARTER_IMG],
        "DB backbone three-stage output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR DB backbone three-stage IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    // ReLU clamps lower to >= 0
    assert!(
        lo_min >= -1e-4,
        "ReLU output lower should be >= 0, got {lo_min}"
    );
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 14. DB FPN neck: Multi-scale feature fusion
// ===========================================================================

/// Reduced FPN output channels.
const FPN_CH: usize = 32;

/// Build a DB FPN (Feature Pyramid Network) neck that fuses features
/// from two backbone stages via 1x1 convolution and concatenation.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image in [0, 1]).
/// Output: `[FPN_CH * 2, HALF_IMG, HALF_IMG]` (fused multi-scale features).
///
/// Stage 1: Conv-BN-ReLU at full resolution -> 1x1 conv -> [FPN_CH, 32, 32]
/// Stage 2: Conv-BN-ReLU at half resolution -> 1x1 conv -> [FPN_CH, 16, 16]
/// Upsample stage 2 features: narrow stage 1 to match [FPN_CH, 16, 16]
/// Concat: [FPN_CH + FPN_CH, 16, 16]
fn build_db_fpn_neck_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_db_fpn_neck");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // --- Stage 1: full resolution Conv-BN-ReLU ---
    let s1_cw = b.add_input("s1_conv_w", &[BACKBONE_CH, IN_CHANNELS, 3, 3]);
    let s1_cb = b.add_input("s1_conv_b", &[BACKBONE_CH]);
    let s1_bm = b.add_input("s1_bn_mean", &[BACKBONE_CH]);
    let s1_bv = b.add_input("s1_bn_var", &[BACKBONE_CH]);
    let s1_bw = b.add_input("s1_bn_w", &[BACKBONE_CH]);
    let s1_bb = b.add_input("s1_bn_b", &[BACKBONE_CH]);
    let s1_eps = b.add_input("s1_eps", &[1]);

    let s1_shape = [BACKBONE_CH, IMG_SIZE, IMG_SIZE];
    let s1_conv = b.add_conv2d(input, s1_cw, Some(s1_cb), 1, 1, 1, 1, &s1_shape);
    let s1_bn = b.add_batch_norm(s1_conv, s1_bm, s1_bv, s1_bw, s1_bb, s1_eps, &s1_shape);
    let s1_feat = b.add_relu(s1_bn, &s1_shape);

    // --- Stage 2: half resolution Conv-BN-ReLU ---
    let s2_cw = b.add_input("s2_conv_w", &[STAGE2_CH, BACKBONE_CH, 3, 3]);
    let s2_cb = b.add_input("s2_conv_b", &[STAGE2_CH]);
    let s2_bm = b.add_input("s2_bn_mean", &[STAGE2_CH]);
    let s2_bv = b.add_input("s2_bn_var", &[STAGE2_CH]);
    let s2_bw = b.add_input("s2_bn_w", &[STAGE2_CH]);
    let s2_bb = b.add_input("s2_bn_b", &[STAGE2_CH]);
    let s2_eps = b.add_input("s2_eps", &[1]);

    let s2_shape = [STAGE2_CH, HALF_IMG, HALF_IMG];
    let s2_conv = b.add_conv2d(s1_feat, s2_cw, Some(s2_cb), 2, 2, 1, 1, &s2_shape);
    let s2_bn = b.add_batch_norm(s2_conv, s2_bm, s2_bv, s2_bw, s2_bb, s2_eps, &s2_shape);
    let s2_feat = b.add_relu(s2_bn, &s2_shape);

    // --- FPN lateral 1x1 convolutions ---
    // Project stage 1 to FPN channels, then narrow spatial to match stage 2
    let lat1_w = b.add_input("lat1_w", &[FPN_CH, BACKBONE_CH, 1, 1]);
    let lat1 = b.add_conv2d(
        s1_feat,
        lat1_w,
        None,
        1,
        1,
        0,
        0,
        &[FPN_CH, IMG_SIZE, IMG_SIZE],
    );
    // Simulate downsample: narrow spatial dimensions to HALF_IMG
    let lat1_down = b.add_narrow(lat1, 1, 0, HALF_IMG, &[FPN_CH, HALF_IMG, IMG_SIZE]);
    let lat1_down2 = b.add_narrow(lat1_down, 2, 0, HALF_IMG, &[FPN_CH, HALF_IMG, HALF_IMG]);

    // Project stage 2 to FPN channels
    let lat2_w = b.add_input("lat2_w", &[FPN_CH, STAGE2_CH, 1, 1]);
    let lat2 = b.add_conv2d(
        s2_feat,
        lat2_w,
        None,
        1,
        1,
        0,
        0,
        &[FPN_CH, HALF_IMG, HALF_IMG],
    );

    // Concatenate along channel dimension
    let fused_ch = FPN_CH * 2;
    let out = b.add_concat(&[lat1_down2, lat2], 0, &[fused_ch, HALF_IMG, HALF_IMG]);

    b.build(out).expect("valid PaddleOCR DB FPN neck kernel")
}

/// Bindings for DB FPN neck.
fn db_fpn_neck_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // image

    // Stage 1 Conv-BN-ReLU
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH, IN_CHANNELS, 3, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // Stage 2 Conv-BN-ReLU
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH, BACKBONE_CH, 3, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // FPN lateral convs
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[FPN_CH, BACKBONE_CH, 1, 1]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[FPN_CH, STAGE2_CH, 1, 1]),
        WEIGHT_MAG,
    )));

    bindings
}

/// IBP through DB FPN neck: multi-scale feature fusion.
#[test]
fn test_db_fpn_neck_ibp() {
    let def = build_db_fpn_neck_kernel();
    let bindings = db_fpn_neck_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR DB FPN neck");

    let fused_ch = FPN_CH * 2;
    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[fused_ch, HALF_IMG, HALF_IMG],
        "DB FPN neck output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR DB FPN neck IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 15. DB full detector: backbone -> FPN-style fusion -> sigmoid
// ===========================================================================

/// Build the full DB detector: 2-stage backbone -> channel projection
/// -> sigmoid probability map.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image in [0, 1]).
/// Output: `[1, HALF_IMG, HALF_IMG]` (text probability map in [0, 1]).
///
/// Stage 1: Conv-BN-ReLU at full resolution
/// Stage 2: Conv-BN-ReLU at half resolution (stride=2)
/// Project: 1x1 Conv to single channel
/// Output: Sigmoid binarization
fn build_db_full_detector_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_db_full_detector");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // --- Stage 1: Conv-BN-ReLU ---
    let s1_cw = b.add_input("s1_conv_w", &[BACKBONE_CH, IN_CHANNELS, 3, 3]);
    let s1_cb = b.add_input("s1_conv_b", &[BACKBONE_CH]);
    let s1_bm = b.add_input("s1_bn_mean", &[BACKBONE_CH]);
    let s1_bv = b.add_input("s1_bn_var", &[BACKBONE_CH]);
    let s1_bw = b.add_input("s1_bn_w", &[BACKBONE_CH]);
    let s1_bb = b.add_input("s1_bn_b", &[BACKBONE_CH]);
    let s1_eps = b.add_input("s1_eps", &[1]);

    let s1_shape = [BACKBONE_CH, IMG_SIZE, IMG_SIZE];
    let s1_conv = b.add_conv2d(input, s1_cw, Some(s1_cb), 1, 1, 1, 1, &s1_shape);
    let s1_bn = b.add_batch_norm(s1_conv, s1_bm, s1_bv, s1_bw, s1_bb, s1_eps, &s1_shape);
    let s1_out = b.add_relu(s1_bn, &s1_shape);

    // --- Stage 2: Conv-BN-ReLU (stride 2) ---
    let s2_cw = b.add_input("s2_conv_w", &[STAGE2_CH, BACKBONE_CH, 3, 3]);
    let s2_cb = b.add_input("s2_conv_b", &[STAGE2_CH]);
    let s2_bm = b.add_input("s2_bn_mean", &[STAGE2_CH]);
    let s2_bv = b.add_input("s2_bn_var", &[STAGE2_CH]);
    let s2_bw = b.add_input("s2_bn_w", &[STAGE2_CH]);
    let s2_bb = b.add_input("s2_bn_b", &[STAGE2_CH]);
    let s2_eps = b.add_input("s2_eps", &[1]);

    let s2_shape = [STAGE2_CH, HALF_IMG, HALF_IMG];
    let s2_conv = b.add_conv2d(s1_out, s2_cw, Some(s2_cb), 2, 2, 1, 1, &s2_shape);
    let s2_bn = b.add_batch_norm(s2_conv, s2_bm, s2_bv, s2_bw, s2_bb, s2_eps, &s2_shape);
    let s2_out = b.add_relu(s2_bn, &s2_shape);

    // --- Head: 1x1 Conv -> Sigmoid ---
    let head_w = b.add_input("head_w", &[1, STAGE2_CH, 1, 1]);
    let head_b = b.add_input("head_b", &[1]);

    let head_shape = [1, HALF_IMG, HALF_IMG];
    let proj = b.add_conv2d(s2_out, head_w, Some(head_b), 1, 1, 0, 0, &head_shape);
    let out = b.add_sigmoid(proj, &head_shape);

    b.build(out)
        .expect("valid PaddleOCR DB full detector kernel")
}

/// Bindings for DB full detector.
fn db_full_detector_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // image

    // Stage 1
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH, IN_CHANNELS, 3, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // Stage 2
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH, BACKBONE_CH, 3, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // Head
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1, STAGE2_CH, 1, 1]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.0f32,
    )));

    bindings
}

/// IBP through full DB detector: image [0,1] -> sigmoid probability [0,1].
#[test]
fn test_db_full_detector_ibp() {
    let def = build_db_full_detector_kernel();
    let bindings = db_full_detector_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR DB full detector");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[1, HALF_IMG, HALF_IMG],
        "DB full detector output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR DB full detector IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    // Sigmoid output must be in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "detector output lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "detector output upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 16. SVTR patch-to-CTC: Patch embed -> 2 SVTR blocks -> CTC linear head
// ===========================================================================

/// Build an end-to-end SVTR recognizer from patch embedding to CTC head.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image in [0, 1]).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (character logits).
///
/// Patch embed -> 2 SVTR blocks (attention + MLP each) -> CTC linear head.
fn build_svtr_patch_to_ctc_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_svtr_patch_to_ctc");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // --- Patch embedding ---
    let patch_w = b.add_input(
        "patch_w",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_b = b.add_input("patch_b", &[HIDDEN_DIM]);

    let conv_out = b.add_conv2d(
        input,
        patch_w,
        Some(patch_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, NUM_PATCHES]);
    let patches = b.add_transpose(reshaped, &[1, 0], &[SEQ_LEN, HIDDEN_DIM]);

    // --- Block 1 ---
    let ln1a_w = b.add_input("b1_ln1_w", &[HIDDEN_DIM]);
    let ln1a_b = b.add_input("b1_ln1_b", &[HIDDEN_DIM]);
    let ln1a_eps = b.add_input("b1_ln1_eps", &[1]);
    let b1_qw = b.add_input("b1_qw", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b1_kw = b.add_input("b1_kw", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b1_vw = b.add_input("b1_vw", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b1_ow = b.add_input("b1_ow", &[HIDDEN_DIM, HIDDEN_DIM]);

    let n1a = b.add_layer_norm(patches, ln1a_eps, 1, ln1a_w, ln1a_b, &shape);
    let q1 = b.add_linear(n1a, b1_qw, None, &shape);
    let k1 = b.add_linear(n1a, b1_kw, None, &shape);
    let v1 = b.add_linear(n1a, b1_vw, None, &shape);
    let a1 = b.add_attention(q1, k1, v1, AttentionMask::Standard, Some(scale), &shape);
    let p1 = b.add_linear(a1, b1_ow, None, &shape);
    let r1a = b.add_binary_add(patches, p1, &shape);

    let ln1b_w = b.add_input("b1_ln2_w", &[HIDDEN_DIM]);
    let ln1b_b = b.add_input("b1_ln2_b", &[HIDDEN_DIM]);
    let ln1b_eps = b.add_input("b1_ln2_eps", &[1]);
    let b1_fc1 = b.add_input("b1_fc1", &[FFN_DIM, HIDDEN_DIM]);
    let b1_fc2 = b.add_input("b1_fc2", &[HIDDEN_DIM, FFN_DIM]);

    let n1b = b.add_layer_norm(r1a, ln1b_eps, 1, ln1b_w, ln1b_b, &shape);
    let h1 = b.add_linear(n1b, b1_fc1, None, &ffn_shape);
    let h1a = b.add_gelu(h1, &ffn_shape);
    let m1 = b.add_linear(h1a, b1_fc2, None, &shape);
    let r1b = b.add_binary_add(r1a, m1, &shape);

    // --- Block 2 ---
    let ln2a_w = b.add_input("b2_ln1_w", &[HIDDEN_DIM]);
    let ln2a_b = b.add_input("b2_ln1_b", &[HIDDEN_DIM]);
    let ln2a_eps = b.add_input("b2_ln1_eps", &[1]);
    let b2_qw = b.add_input("b2_qw", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b2_kw = b.add_input("b2_kw", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b2_vw = b.add_input("b2_vw", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b2_ow = b.add_input("b2_ow", &[HIDDEN_DIM, HIDDEN_DIM]);

    let n2a = b.add_layer_norm(r1b, ln2a_eps, 1, ln2a_w, ln2a_b, &shape);
    let q2 = b.add_linear(n2a, b2_qw, None, &shape);
    let k2 = b.add_linear(n2a, b2_kw, None, &shape);
    let v2 = b.add_linear(n2a, b2_vw, None, &shape);
    let a2 = b.add_attention(q2, k2, v2, AttentionMask::Standard, Some(scale), &shape);
    let p2 = b.add_linear(a2, b2_ow, None, &shape);
    let r2a = b.add_binary_add(r1b, p2, &shape);

    let ln2b_w = b.add_input("b2_ln2_w", &[HIDDEN_DIM]);
    let ln2b_b = b.add_input("b2_ln2_b", &[HIDDEN_DIM]);
    let ln2b_eps = b.add_input("b2_ln2_eps", &[1]);
    let b2_fc1 = b.add_input("b2_fc1", &[FFN_DIM, HIDDEN_DIM]);
    let b2_fc2 = b.add_input("b2_fc2", &[HIDDEN_DIM, FFN_DIM]);

    let n2b = b.add_layer_norm(r2a, ln2b_eps, 1, ln2b_w, ln2b_b, &shape);
    let h2 = b.add_linear(n2b, b2_fc1, None, &ffn_shape);
    let h2a = b.add_gelu(h2, &ffn_shape);
    let m2 = b.add_linear(h2a, b2_fc2, None, &shape);
    let enc_out = b.add_binary_add(r2a, m2, &shape);

    // --- CTC head ---
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);
    let out = b.add_linear(enc_out, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid PaddleOCR SVTR patch-to-CTC kernel")
}

/// Bindings for SVTR patch-to-CTC pipeline.
fn svtr_patch_to_ctc_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // image

    // Patch embedding
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM]),
        0.0f32,
    )));

    // Blocks 1 and 2 (same structure)
    for _ in 0..2 {
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(fc1_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(fc2_w.clone()));
    }

    // CTC head
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[VOCAB_SIZE]),
        0.0f32,
    )));

    bindings
}

/// IBP through SVTR patch-to-CTC: image [0,1] -> character logits.
#[test]
fn test_svtr_patch_to_ctc_ibp() {
    let def = build_svtr_patch_to_ctc_kernel();
    let bindings = svtr_patch_to_ctc_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR SVTR patch-to-CTC");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "SVTR patch-to-CTC output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR SVTR patch-to-CTC IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 17. Detect-then-recognize: DB output -> crop simulation -> SVTR -> CTC
// ===========================================================================

/// Build a detect-then-recognize pipeline.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image in [0, 1]).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (character probabilities via softmax).
///
/// Detection: Conv-BN-ReLU backbone -> sigmoid (text probability)
/// Crop simulation: narrow the backbone features (simulates text region crop)
/// Recognition: project features -> MLP -> CTC head -> softmax
fn build_detect_then_recognize_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_detect_then_recognize");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // --- Detection backbone ---
    let d_cw = b.add_input("det_conv_w", &[BACKBONE_CH, IN_CHANNELS, 3, 3]);
    let d_cb = b.add_input("det_conv_b", &[BACKBONE_CH]);
    let d_bm = b.add_input("det_bn_mean", &[BACKBONE_CH]);
    let d_bv = b.add_input("det_bn_var", &[BACKBONE_CH]);
    let d_bw = b.add_input("det_bn_w", &[BACKBONE_CH]);
    let d_bb = b.add_input("det_bn_b", &[BACKBONE_CH]);
    let d_eps = b.add_input("det_eps", &[1]);

    let feat_shape = [BACKBONE_CH, IMG_SIZE, IMG_SIZE];
    let d_conv = b.add_conv2d(input, d_cw, Some(d_cb), 1, 1, 1, 1, &feat_shape);
    let d_bn = b.add_batch_norm(d_conv, d_bm, d_bv, d_bw, d_bb, d_eps, &feat_shape);
    let features = b.add_relu(d_bn, &feat_shape);

    // --- Crop simulation: narrow spatial to a text region ---
    // Take HALF_IMG rows from the features, then narrow columns
    let cropped = b.add_narrow(features, 1, 0, HALF_IMG, &[BACKBONE_CH, HALF_IMG, IMG_SIZE]);
    let cropped2 = b.add_narrow(cropped, 2, 0, HALF_IMG, &[BACKBONE_CH, HALF_IMG, HALF_IMG]);

    // --- Recognition: project to hidden dim, flatten, encode ---
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, BACKBONE_CH, 1, 1]);
    let proj = b.add_conv2d(
        cropped2,
        proj_w,
        None,
        1,
        1,
        0,
        0,
        &[HIDDEN_DIM, HALF_IMG, HALF_IMG],
    );

    // Flatten spatial: [HIDDEN_DIM, 16, 16] -> [HIDDEN_DIM, 256]
    let spatial = HALF_IMG * HALF_IMG;
    let flat = b.add_reshape(proj, &[HIDDEN_DIM, spatial]);
    let trans = b.add_transpose(flat, &[1, 0], &[spatial, HIDDEN_DIM]);

    // Narrow to SEQ_LEN for tractability
    let seq = b.add_narrow(trans, 0, 0, SEQ_LEN, &[SEQ_LEN, HIDDEN_DIM]);

    // --- MLP encoder ---
    let fc1_w = b.add_input("fc1_w", &[FFN_DIM, HIDDEN_DIM]);
    let fc2_w = b.add_input("fc2_w", &[HIDDEN_DIM, FFN_DIM]);

    let h = b.add_linear(seq, fc1_w, None, &[SEQ_LEN, FFN_DIM]);
    let h_act = b.add_gelu(h, &[SEQ_LEN, FFN_DIM]);
    let encoded = b.add_linear(h_act, fc2_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let enc_out = b.add_binary_add(seq, encoded, &[SEQ_LEN, HIDDEN_DIM]);

    // --- CTC head + softmax ---
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(enc_out, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid PaddleOCR detect-then-recognize kernel")
}

/// Bindings for detect-then-recognize pipeline.
fn detect_then_recognize_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // image

    // Detection backbone
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH, IN_CHANNELS, 3, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // Projection
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, BACKBONE_CH, 1, 1]),
        WEIGHT_MAG,
    )));

    // MLP encoder
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[FFN_DIM, HIDDEN_DIM]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, FFN_DIM]),
        WEIGHT_MAG,
    )));

    // CTC head
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[VOCAB_SIZE]),
        0.0f32,
    )));

    bindings
}

/// IBP through detect-then-recognize: image -> detection -> crop -> recognition.
///
/// End-to-end pipeline where detection features are cropped and fed to
/// recognition. Softmax output must be in [0, 1].
#[test]
fn test_detect_then_recognize_ibp() {
    let def = build_detect_then_recognize_kernel();
    let bindings = detect_then_recognize_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR detect-then-recognize");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "detect-then-recognize output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR detect-then-recognize IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

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
// 18. CTC greedy decode bounds: softmax -> argmax-like property
// ===========================================================================

/// Build a CTC greedy decode bounds verification kernel.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, encoder output).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (softmax probabilities).
///
/// CTC greedy decoding takes argmax of the softmax output. We verify
/// that the softmax output is a valid probability distribution: all
/// elements in [0, 1], which guarantees the argmax selects a valid token ID.
///
/// Linear -> Softmax: probability distribution over vocabulary.
fn build_ctc_greedy_decode_bounds_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_ctc_greedy_decode_bounds");

    let input = b.add_input("encoder_out", &[SEQ_LEN, HIDDEN_DIM]);
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);

    let logits = b.add_linear(input, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs)
        .expect("valid PaddleOCR CTC greedy decode bounds kernel")
}

/// Bindings for CTC greedy decode bounds.
fn ctc_greedy_decode_bounds_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)),
    ]
}

/// IBP through CTC greedy decode: softmax output guarantees valid token IDs.
///
/// If softmax output is in [0, 1], then argmax selects a token index in
/// [0, VOCAB_SIZE-1] — a valid token ID. This is the key property for
/// CTC greedy decoding correctness.
#[test]
fn test_ctc_greedy_decode_bounds_ibp() {
    let def = build_ctc_greedy_decode_bounds_kernel();
    let bindings = ctc_greedy_decode_bounds_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR CTC greedy decode bounds");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "CTC greedy decode output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "PaddleOCR CTC greedy decode bounds IBP (encoder [-2,2]): bounds=[{lo_min}, {hi_max}]"
    );

    // Softmax output in [0, 1] guarantees valid token selection
    assert!(
        lo_min >= -1e-4,
        "softmax lower should be >= 0 for valid CTC decode, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "softmax upper should be <= 1 for valid CTC decode, got {hi_max}"
    );
}

// ===========================================================================
// 19. SVTR attention with sinusoidal position encoding
// ===========================================================================

/// Build SVTR attention with additive sinusoidal position encoding.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, patch embeddings).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Adds sinusoidal position encoding to the input before applying
/// the standard SVTR attention block. Position encoding is a constant
/// tensor with values in [-1, 1] (sin/cos).
fn build_svtr_attention_with_position_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_svtr_attention_with_position");

    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let pos_enc = b.add_input("pos_enc", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Add positional encoding
    let x_pos = b.add_binary_add(input, pos_enc, &shape);

    // LayerNorm -> Attention -> residual
    let ln_w = b.add_input("ln_weight", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_bias", &[HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let x_norm = b.add_layer_norm(x_pos, eps, 1, ln_w, ln_b, &shape);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let q = b.add_linear(x_norm, q_w, None, &shape);
    let k = b.add_linear(x_norm, k_w, None, &shape);
    let v = b.add_linear(x_norm, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let proj = b.add_linear(attn, out_w, None, &shape);
    let out = b.add_binary_add(x_pos, proj, &shape);

    b.build(out)
        .expect("valid PaddleOCR SVTR attention with position kernel")
}

/// Bindings for SVTR attention with sinusoidal position encoding.
fn svtr_attention_with_position_bindings() -> Vec<TensorParamBinding> {
    // Generate sinusoidal position encoding
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

    vec![
        TensorParamBinding::Variable,                // x
        TensorParamBinding::ConstantTensor(pos_enc), // pos_enc
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)), // ln_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)), // ln_bias
        TensorParamBinding::ConstantScalar(1e-5),                                            // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // q_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // k_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // v_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // out_weight
    ]
}

/// IBP bounds through SVTR attention with sinusoidal position encoding.
///
/// Position encoding values are in [-1, 1], so the addition shifts input
/// bounds by at most 1.0 in each direction before attention.
#[test]
fn test_svtr_attention_with_position_ibp() {
    let def = build_svtr_attention_with_position_kernel();
    let bindings = svtr_attention_with_position_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR SVTR attention with position");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "SVTR attention with position output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "PaddleOCR SVTR attention with position IBP (input [-1,1]): bounds=[{lo_min}, {hi_max}]"
    );

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 20. DB binarization sigmoid: sigmoid -> threshold comparison
// ===========================================================================

/// Build a DB binarization sigmoid verification kernel.
///
/// Input: `[BACKBONE_CH, IMG_SIZE, IMG_SIZE]` (Variable, backbone features).
/// Output: `[1, IMG_SIZE, IMG_SIZE]` (sigmoid probability map in [0, 1]).
///
/// Conv2d(backbone_ch, 1, kernel=3, pad=1) -> Sigmoid.
/// The binarization step in DB uses a threshold on the sigmoid output.
/// We verify the sigmoid output is in [0, 1], which guarantees the
/// threshold comparison produces valid binary {0, 1} decisions.
fn build_db_binarization_sigmoid_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_db_binarization_sigmoid");

    let input = b.add_input("features", &[BACKBONE_CH, IMG_SIZE, IMG_SIZE]);
    let conv_w = b.add_input("bin_conv_w", &[1, BACKBONE_CH, 3, 3]);
    let conv_b = b.add_input("bin_conv_b", &[1]);

    let out_shape = [1, IMG_SIZE, IMG_SIZE];
    let conv_out = b.add_conv2d(input, conv_w, Some(conv_b), 1, 1, 1, 1, &out_shape);
    let out = b.add_sigmoid(conv_out, &out_shape);

    b.build(out)
        .expect("valid PaddleOCR DB binarization sigmoid kernel")
}

/// Bindings for DB binarization sigmoid.
fn db_binarization_sigmoid_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[1, BACKBONE_CH, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 0.0f32)),
    ]
}

/// IBP through DB binarization sigmoid: output must be in [0, 1].
///
/// The sigmoid codomain is (0, 1). For any finite input, the output
/// is strictly within [0, 1]. This guarantees that the subsequent
/// threshold comparison (binarization) always produces a valid decision.
#[test]
fn test_db_binarization_sigmoid_ibp() {
    let def = build_db_binarization_sigmoid_kernel();
    let bindings = db_binarization_sigmoid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[BACKBONE_CH, IMG_SIZE, IMG_SIZE], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR DB binarization sigmoid");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[1, IMG_SIZE, IMG_SIZE],
        "DB binarization sigmoid output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "PaddleOCR DB binarization sigmoid IBP (features [-2,2]): bounds=[{lo_min}, {hi_max}]"
    );

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
// 21. SVTR MLP-GELU two blocks CROWN
// ===========================================================================

/// Build 2 sequential MLP-GELU blocks for CROWN verification.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Block 1: LayerNorm -> Linear -> GELU -> Linear -> residual
/// Block 2: LayerNorm -> Linear -> GELU -> Linear -> residual
///
/// Tests CROWN linearization through repeated GELU activations with
/// LayerNorm pre-normalization (IbpValidated mode).
fn build_svtr_mlp_gelu_two_blocks_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_svtr_mlp_gelu_two_blocks");

    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // --- Block 1 ---
    let ln1_w = b.add_input("b1_ln_w", &[HIDDEN_DIM]);
    let ln1_b = b.add_input("b1_ln_b", &[HIDDEN_DIM]);
    let ln1_eps = b.add_input("b1_eps", &[1]);
    let b1_fc1 = b.add_input("b1_fc1_w", &[FFN_DIM, HIDDEN_DIM]);
    let b1_fc2 = b.add_input("b1_fc2_w", &[HIDDEN_DIM, FFN_DIM]);

    let n1 = b.add_layer_norm(input, ln1_eps, 1, ln1_w, ln1_b, &shape);
    let h1 = b.add_linear(n1, b1_fc1, None, &ffn_shape);
    let h1_act = b.add_gelu(h1, &ffn_shape);
    let m1 = b.add_linear(h1_act, b1_fc2, None, &shape);
    let r1 = b.add_binary_add(input, m1, &shape);

    // --- Block 2 ---
    let ln2_w = b.add_input("b2_ln_w", &[HIDDEN_DIM]);
    let ln2_b = b.add_input("b2_ln_b", &[HIDDEN_DIM]);
    let ln2_eps = b.add_input("b2_eps", &[1]);
    let b2_fc1 = b.add_input("b2_fc1_w", &[FFN_DIM, HIDDEN_DIM]);
    let b2_fc2 = b.add_input("b2_fc2_w", &[HIDDEN_DIM, FFN_DIM]);

    let n2 = b.add_layer_norm(r1, ln2_eps, 1, ln2_w, ln2_b, &shape);
    let h2 = b.add_linear(n2, b2_fc1, None, &ffn_shape);
    let h2_act = b.add_gelu(h2, &ffn_shape);
    let m2 = b.add_linear(h2_act, b2_fc2, None, &shape);
    let out = b.add_binary_add(r1, m2, &shape);

    b.build(out)
        .expect("valid PaddleOCR SVTR MLP GELU two-blocks kernel")
}

/// Bindings for SVTR MLP-GELU two blocks.
fn svtr_mlp_gelu_two_blocks_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable, // x
        // Block 1
        TensorParamBinding::ConstantTensor(ln_w.clone()),
        TensorParamBinding::ConstantTensor(ln_b.clone()),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(fc1_w.clone()),
        TensorParamBinding::ConstantTensor(fc2_w.clone()),
        // Block 2
        TensorParamBinding::ConstantTensor(ln_w),
        TensorParamBinding::ConstantTensor(ln_b),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(fc1_w),
        TensorParamBinding::ConstantTensor(fc2_w),
    ]
}

/// CROWN bounds propagate through 2 sequential MLP-GELU blocks.
///
/// GELU is piecewise-smooth and CROWN-friendly. Two consecutive blocks
/// test CROWN stability through repeated nonlinear layers.
#[test]
fn test_svtr_mlp_gelu_two_blocks_crown() {
    let def = build_svtr_mlp_gelu_two_blocks_kernel();
    let bindings = svtr_mlp_gelu_two_blocks_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR SVTR MLP GELU two-blocks: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 22. DB backbone 3-stage with sigmoid head: full multi-scale detection
// ===========================================================================

/// Build a 3-stage DB backbone ending with sigmoid binarization head.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image in [0, 1]).
/// Output: `[1, QUARTER_IMG, QUARTER_IMG]` (probability map in [0, 1]).
///
/// Stage 1: Conv(3, 32, k=3, s=1, p=1) -> BN -> ReLU
/// Stage 2: Conv(32, 64, k=3, s=2, p=1) -> BN -> ReLU
/// Stage 3: Conv(64, 128, k=3, s=2, p=1) -> BN -> ReLU
/// Head: Conv(128, 1, k=1) -> Sigmoid
fn build_db_three_stage_sigmoid_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_db_three_stage_sigmoid");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // --- Stage 1: [3, 32, 32] -> [32, 32, 32] ---
    let s1_cw = b.add_input("s1_conv_w", &[BACKBONE_CH, IN_CHANNELS, 3, 3]);
    let s1_cb = b.add_input("s1_conv_b", &[BACKBONE_CH]);
    let s1_bm = b.add_input("s1_bn_mean", &[BACKBONE_CH]);
    let s1_bv = b.add_input("s1_bn_var", &[BACKBONE_CH]);
    let s1_bw = b.add_input("s1_bn_w", &[BACKBONE_CH]);
    let s1_bb = b.add_input("s1_bn_b", &[BACKBONE_CH]);
    let s1_eps = b.add_input("s1_eps", &[1]);

    let s1_shape = [BACKBONE_CH, IMG_SIZE, IMG_SIZE];
    let s1_conv = b.add_conv2d(input, s1_cw, Some(s1_cb), 1, 1, 1, 1, &s1_shape);
    let s1_bn = b.add_batch_norm(s1_conv, s1_bm, s1_bv, s1_bw, s1_bb, s1_eps, &s1_shape);
    let s1_out = b.add_relu(s1_bn, &s1_shape);

    // --- Stage 2: [32, 32, 32] -> [64, 16, 16] ---
    let s2_cw = b.add_input("s2_conv_w", &[STAGE2_CH, BACKBONE_CH, 3, 3]);
    let s2_cb = b.add_input("s2_conv_b", &[STAGE2_CH]);
    let s2_bm = b.add_input("s2_bn_mean", &[STAGE2_CH]);
    let s2_bv = b.add_input("s2_bn_var", &[STAGE2_CH]);
    let s2_bw = b.add_input("s2_bn_w", &[STAGE2_CH]);
    let s2_bb = b.add_input("s2_bn_b", &[STAGE2_CH]);
    let s2_eps = b.add_input("s2_eps", &[1]);

    let s2_shape = [STAGE2_CH, HALF_IMG, HALF_IMG];
    let s2_conv = b.add_conv2d(s1_out, s2_cw, Some(s2_cb), 2, 2, 1, 1, &s2_shape);
    let s2_bn = b.add_batch_norm(s2_conv, s2_bm, s2_bv, s2_bw, s2_bb, s2_eps, &s2_shape);
    let s2_out = b.add_relu(s2_bn, &s2_shape);

    // --- Stage 3: [64, 16, 16] -> [128, 8, 8] ---
    let s3_cw = b.add_input("s3_conv_w", &[STAGE3_CH, STAGE2_CH, 3, 3]);
    let s3_cb = b.add_input("s3_conv_b", &[STAGE3_CH]);
    let s3_bm = b.add_input("s3_bn_mean", &[STAGE3_CH]);
    let s3_bv = b.add_input("s3_bn_var", &[STAGE3_CH]);
    let s3_bw = b.add_input("s3_bn_w", &[STAGE3_CH]);
    let s3_bb = b.add_input("s3_bn_b", &[STAGE3_CH]);
    let s3_eps = b.add_input("s3_eps", &[1]);

    let s3_shape = [STAGE3_CH, QUARTER_IMG, QUARTER_IMG];
    let s3_conv = b.add_conv2d(s2_out, s3_cw, Some(s3_cb), 2, 2, 1, 1, &s3_shape);
    let s3_bn = b.add_batch_norm(s3_conv, s3_bm, s3_bv, s3_bw, s3_bb, s3_eps, &s3_shape);
    let s3_out = b.add_relu(s3_bn, &s3_shape);

    // --- Head: 1x1 Conv -> Sigmoid ---
    let head_w = b.add_input("head_w", &[1, STAGE3_CH, 1, 1]);
    let head_b = b.add_input("head_b", &[1]);

    let head_shape = [1, QUARTER_IMG, QUARTER_IMG];
    let proj = b.add_conv2d(s3_out, head_w, Some(head_b), 1, 1, 0, 0, &head_shape);
    let out = b.add_sigmoid(proj, &head_shape);

    b.build(out)
        .expect("valid PaddleOCR DB three-stage sigmoid kernel")
}

/// Bindings for DB three-stage sigmoid.
fn db_three_stage_sigmoid_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // image

    // Stage 1
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH, IN_CHANNELS, 3, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // Stage 2
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH, BACKBONE_CH, 3, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // Stage 3
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE3_CH, STAGE2_CH, 3, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE3_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE3_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE3_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE3_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE3_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // Head
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1, STAGE3_CH, 1, 1]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.0f32,
    )));

    bindings
}

/// IBP through 3-stage backbone with sigmoid: end-to-end detection with
/// multi-scale feature extraction. Sigmoid output must be in [0, 1].
#[test]
fn test_db_three_stage_sigmoid_ibp() {
    let def = build_db_three_stage_sigmoid_kernel();
    let bindings = db_three_stage_sigmoid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR DB three-stage sigmoid");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[1, QUARTER_IMG, QUARTER_IMG],
        "DB three-stage sigmoid output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR DB three-stage sigmoid IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

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
// 23. DB sigmoid head with multi-level feature fusion (FPN + sigmoid)
// ===========================================================================

/// Build a DB sigmoid head with FPN-style multi-level feature fusion.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image in [0, 1]).
/// Output: `[1, HALF_IMG, HALF_IMG]` (probability map in [0, 1]).
///
/// Two backbone stages produce features at different scales. FPN lateral
/// convolutions project both to FPN_CH channels, then concatenated features
/// are passed through a 1x1 conv and sigmoid to produce the probability map.
fn build_db_fpn_sigmoid_head_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_db_fpn_sigmoid_head");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // --- Stage 1: full resolution Conv-BN-ReLU ---
    let s1_cw = b.add_input("s1_conv_w", &[BACKBONE_CH, IN_CHANNELS, 3, 3]);
    let s1_cb = b.add_input("s1_conv_b", &[BACKBONE_CH]);
    let s1_bm = b.add_input("s1_bn_mean", &[BACKBONE_CH]);
    let s1_bv = b.add_input("s1_bn_var", &[BACKBONE_CH]);
    let s1_bw = b.add_input("s1_bn_w", &[BACKBONE_CH]);
    let s1_bb = b.add_input("s1_bn_b", &[BACKBONE_CH]);
    let s1_eps = b.add_input("s1_eps", &[1]);

    let s1_shape = [BACKBONE_CH, IMG_SIZE, IMG_SIZE];
    let s1_conv = b.add_conv2d(input, s1_cw, Some(s1_cb), 1, 1, 1, 1, &s1_shape);
    let s1_bn = b.add_batch_norm(s1_conv, s1_bm, s1_bv, s1_bw, s1_bb, s1_eps, &s1_shape);
    let s1_feat = b.add_relu(s1_bn, &s1_shape);

    // --- Stage 2: half resolution Conv-BN-ReLU ---
    let s2_cw = b.add_input("s2_conv_w", &[STAGE2_CH, BACKBONE_CH, 3, 3]);
    let s2_cb = b.add_input("s2_conv_b", &[STAGE2_CH]);
    let s2_bm = b.add_input("s2_bn_mean", &[STAGE2_CH]);
    let s2_bv = b.add_input("s2_bn_var", &[STAGE2_CH]);
    let s2_bw = b.add_input("s2_bn_w", &[STAGE2_CH]);
    let s2_bb = b.add_input("s2_bn_b", &[STAGE2_CH]);
    let s2_eps = b.add_input("s2_eps", &[1]);

    let s2_shape = [STAGE2_CH, HALF_IMG, HALF_IMG];
    let s2_conv = b.add_conv2d(s1_feat, s2_cw, Some(s2_cb), 2, 2, 1, 1, &s2_shape);
    let s2_bn = b.add_batch_norm(s2_conv, s2_bm, s2_bv, s2_bw, s2_bb, s2_eps, &s2_shape);
    let s2_feat = b.add_relu(s2_bn, &s2_shape);

    // --- FPN lateral projections ---
    let lat1_w = b.add_input("lat1_w", &[FPN_CH, BACKBONE_CH, 1, 1]);
    let lat1 = b.add_conv2d(
        s1_feat,
        lat1_w,
        None,
        1,
        1,
        0,
        0,
        &[FPN_CH, IMG_SIZE, IMG_SIZE],
    );
    let lat1_down = b.add_narrow(lat1, 1, 0, HALF_IMG, &[FPN_CH, HALF_IMG, IMG_SIZE]);
    let lat1_down2 = b.add_narrow(lat1_down, 2, 0, HALF_IMG, &[FPN_CH, HALF_IMG, HALF_IMG]);

    let lat2_w = b.add_input("lat2_w", &[FPN_CH, STAGE2_CH, 1, 1]);
    let lat2 = b.add_conv2d(
        s2_feat,
        lat2_w,
        None,
        1,
        1,
        0,
        0,
        &[FPN_CH, HALF_IMG, HALF_IMG],
    );

    // Concatenate along channel dimension
    let fused_ch = FPN_CH * 2;
    let fused = b.add_concat(&[lat1_down2, lat2], 0, &[fused_ch, HALF_IMG, HALF_IMG]);

    // --- Sigmoid head: 1x1 Conv on fused features -> Sigmoid ---
    let head_w = b.add_input("head_w", &[1, fused_ch, 1, 1]);
    let head_b = b.add_input("head_b", &[1]);
    let head_shape = [1, HALF_IMG, HALF_IMG];
    let proj = b.add_conv2d(fused, head_w, Some(head_b), 1, 1, 0, 0, &head_shape);
    let out = b.add_sigmoid(proj, &head_shape);

    b.build(out)
        .expect("valid PaddleOCR DB FPN sigmoid head kernel")
}

/// Bindings for DB FPN sigmoid head.
fn db_fpn_sigmoid_head_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // image

    // Stage 1 Conv-BN-ReLU
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH, IN_CHANNELS, 3, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // Stage 2 Conv-BN-ReLU
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH, BACKBONE_CH, 3, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // FPN lateral convs
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[FPN_CH, BACKBONE_CH, 1, 1]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[FPN_CH, STAGE2_CH, 1, 1]),
        WEIGHT_MAG,
    )));

    // Sigmoid head
    let fused_ch = FPN_CH * 2;
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1, fused_ch, 1, 1]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.0f32,
    )));

    bindings
}

/// IBP through DB FPN sigmoid head: multi-level feature fusion -> sigmoid.
///
/// Tests that multi-scale feature fusion followed by sigmoid produces
/// output bounds strictly within [0, 1].
#[test]
fn test_db_fpn_sigmoid_head_ibp() {
    let def = build_db_fpn_sigmoid_head_kernel();
    let bindings = db_fpn_sigmoid_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR DB FPN sigmoid head");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[1, HALF_IMG, HALF_IMG],
        "DB FPN sigmoid head output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR DB FPN sigmoid head IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

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
// 24. SVTR 3-layer attention stack CROWN
// ===========================================================================

/// Build a 3-block SVTR attention-only stack for CROWN verification.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Three sequential attention blocks (LayerNorm -> Q/K/V -> Attention ->
/// output projection -> residual). No MLP blocks, pure attention depth test.
fn build_svtr_three_layer_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_svtr_three_layer_attention");

    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut current = input;

    for block_idx in 0..3 {
        let prefix = format!("b{block_idx}");

        let ln_w = b.add_input(&format!("{prefix}_ln_w"), &[HIDDEN_DIM]);
        let ln_b = b.add_input(&format!("{prefix}_ln_b"), &[HIDDEN_DIM]);
        let ln_eps = b.add_input(&format!("{prefix}_ln_eps"), &[1]);
        let qw = b.add_input(&format!("{prefix}_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let kw = b.add_input(&format!("{prefix}_k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let vw = b.add_input(&format!("{prefix}_v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let ow = b.add_input(&format!("{prefix}_o_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let normed = b.add_layer_norm(current, ln_eps, 1, ln_w, ln_b, &shape);
        let q = b.add_linear(normed, qw, None, &shape);
        let k = b.add_linear(normed, kw, None, &shape);
        let v = b.add_linear(normed, vw, None, &shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
        let proj = b.add_linear(attn, ow, None, &shape);
        current = b.add_binary_add(current, proj, &shape);
    }

    b.build(current)
        .expect("valid PaddleOCR SVTR three-layer attention kernel")
}

/// Bindings for SVTR 3-layer attention stack.
fn svtr_three_layer_attention_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // x

    for _ in 0..3 {
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
    }

    bindings
}

/// CROWN bounds through 3-layer pure attention stack.
///
/// Tests CROWN linearization through repeated LayerNorm + attention softmax
/// without interleaved MLP blocks. Deeper attention-only composition
/// exercises the McCormick envelope for bilinear attention terms.
#[test]
fn test_svtr_three_layer_attention_crown() {
    let def = build_svtr_three_layer_attention_kernel();
    let bindings = svtr_three_layer_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "PaddleOCR SVTR three-layer attention: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 25. SVTR 2-layer attention + MLP stack CROWN
// ===========================================================================

/// Build a 2-block SVTR encoder (attention + MLP each) for CROWN.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Block structure: LayerNorm -> Attention -> residual -> LayerNorm ->
/// MLP (GELU) -> residual. Tests CROWN through interleaved attention and
/// GELU nonlinearities at depth 2.
fn build_svtr_two_layer_attn_mlp_crown_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_svtr_two_layer_attn_mlp_crown");

    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut current = input;

    for block_idx in 0..2 {
        let prefix = format!("b{block_idx}");

        // Attention sub-block
        let ln_a_w = b.add_input(&format!("{prefix}_ln1_w"), &[HIDDEN_DIM]);
        let ln_a_b = b.add_input(&format!("{prefix}_ln1_b"), &[HIDDEN_DIM]);
        let ln_a_eps = b.add_input(&format!("{prefix}_ln1_eps"), &[1]);
        let qw = b.add_input(&format!("{prefix}_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let kw = b.add_input(&format!("{prefix}_k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let vw = b.add_input(&format!("{prefix}_v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let ow = b.add_input(&format!("{prefix}_o_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let normed = b.add_layer_norm(current, ln_a_eps, 1, ln_a_w, ln_a_b, &shape);
        let q = b.add_linear(normed, qw, None, &shape);
        let k = b.add_linear(normed, kw, None, &shape);
        let v = b.add_linear(normed, vw, None, &shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
        let proj = b.add_linear(attn, ow, None, &shape);
        let res_a = b.add_binary_add(current, proj, &shape);

        // MLP sub-block
        let ln_b_w = b.add_input(&format!("{prefix}_ln2_w"), &[HIDDEN_DIM]);
        let ln_b_b = b.add_input(&format!("{prefix}_ln2_b"), &[HIDDEN_DIM]);
        let ln_b_eps = b.add_input(&format!("{prefix}_ln2_eps"), &[1]);
        let fc1 = b.add_input(&format!("{prefix}_fc1_w"), &[FFN_DIM, HIDDEN_DIM]);
        let fc2 = b.add_input(&format!("{prefix}_fc2_w"), &[HIDDEN_DIM, FFN_DIM]);

        let normed_b = b.add_layer_norm(res_a, ln_b_eps, 1, ln_b_w, ln_b_b, &shape);
        let h = b.add_linear(normed_b, fc1, None, &ffn_shape);
        let h_act = b.add_gelu(h, &ffn_shape);
        let mlp = b.add_linear(h_act, fc2, None, &shape);
        current = b.add_binary_add(res_a, mlp, &shape);
    }

    b.build(current)
        .expect("valid PaddleOCR SVTR two-layer attn+MLP CROWN kernel")
}

/// Bindings for SVTR 2-layer attention + MLP CROWN.
fn svtr_two_layer_attn_mlp_crown_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // x

    for _ in 0..2 {
        // Attention sub-block
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        // MLP sub-block
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(fc1_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(fc2_w.clone()));
    }

    bindings
}

/// CROWN bounds through 2-layer attention + MLP stack.
///
/// Tests CROWN linearization through interleaved attention (softmax) and
/// GELU nonlinearities. Deeper than single-block tests 4 and 5.
#[test]
fn test_svtr_two_layer_attn_mlp_crown() {
    let def = build_svtr_two_layer_attn_mlp_crown_kernel();
    let bindings = svtr_two_layer_attn_mlp_crown_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "PaddleOCR SVTR two-layer attn+MLP CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 26. CTC head with softmax CROWN
// ===========================================================================

/// Build a CTC head with softmax for CROWN verification.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, encoder output).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (character probabilities in [0, 1]).
///
/// LayerNorm -> Linear -> Softmax. Tests CROWN through the final
/// normalization + linear + softmax composition in the recognition path.
fn build_ctc_head_softmax_crown_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_ctc_head_softmax_crown");

    let input = b.add_input("encoder_out", &[SEQ_LEN, HIDDEN_DIM]);
    let ln_w = b.add_input("ln_weight", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_bias", &[HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);

    let shape = [SEQ_LEN, HIDDEN_DIM];

    // LayerNorm before projection
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &shape);

    // Linear projection to logits
    let logits = b.add_linear(normed, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);

    // Softmax over vocabulary
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid PaddleOCR CTC head softmax CROWN kernel")
}

/// Bindings for CTC head softmax CROWN.
fn ctc_head_softmax_crown_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)),
    ]
}

/// CROWN bounds through CTC head with LayerNorm + softmax.
///
/// LayerNorm requires CROWN linearization (IbpValidated mode). The
/// softmax output must be in [0, 1] regardless of verification method.
#[test]
fn test_ctc_head_softmax_crown() {
    let def = build_ctc_head_softmax_crown_kernel();
    let bindings = ctc_head_softmax_crown_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "CTC head softmax CROWN output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR CTC head softmax CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

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
// 27. Full detection pipeline CROWN: backbone -> FPN fusion -> sigmoid
// ===========================================================================

/// CROWN bounds through the full DB detection pipeline.
///
/// Reuses the detection pipeline kernel (test 8) but verifies with CROWN
/// instead of IBP, testing whether CROWN produces tighter bounds through
/// the Conv-BN-ReLU -> 1x1 Conv -> Sigmoid composition.
#[test]
fn test_detection_pipeline_crown() {
    let def = build_detection_pipeline_kernel();
    let bindings = detection_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[1, IMG_SIZE, IMG_SIZE],
        "detection pipeline CROWN output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR detection pipeline CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    // Sigmoid output must be in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "detection output lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "detection output upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 28. Full recognition pipeline CROWN
// ===========================================================================

/// CROWN bounds through the recognition pipeline.
///
/// Reuses the recognition pipeline kernel (test 9) but verifies with CROWN.
/// Tests CROWN linearization through patch embedding -> GELU MLP -> Linear.
#[test]
fn test_recognition_pipeline_crown() {
    let def = build_recognition_pipeline_kernel();
    let bindings = recognition_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "recognition pipeline CROWN output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "PaddleOCR recognition pipeline CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 29. Detection + recognition end-to-end IBP
// ===========================================================================

/// Build a full detection + recognition end-to-end pipeline.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image in [0, 1]).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (character probabilities).
///
/// Detection: 2-stage Conv-BN-ReLU backbone
/// Crop: narrow backbone features to simulate text region extraction
/// Recognition: project -> MLP with GELU -> CTC head -> softmax
///
/// This is deeper than test 17 because it uses a 2-stage backbone with
/// stride-2 downsampling before the crop.
fn build_detect_recognize_e2e_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_detect_recognize_e2e");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // --- Detection: 2-stage backbone ---
    // Stage 1: full resolution
    let s1_cw = b.add_input("s1_conv_w", &[BACKBONE_CH, IN_CHANNELS, 3, 3]);
    let s1_cb = b.add_input("s1_conv_b", &[BACKBONE_CH]);
    let s1_bm = b.add_input("s1_bn_mean", &[BACKBONE_CH]);
    let s1_bv = b.add_input("s1_bn_var", &[BACKBONE_CH]);
    let s1_bw = b.add_input("s1_bn_w", &[BACKBONE_CH]);
    let s1_bb = b.add_input("s1_bn_b", &[BACKBONE_CH]);
    let s1_eps = b.add_input("s1_eps", &[1]);

    let s1_shape = [BACKBONE_CH, IMG_SIZE, IMG_SIZE];
    let s1_conv = b.add_conv2d(input, s1_cw, Some(s1_cb), 1, 1, 1, 1, &s1_shape);
    let s1_bn = b.add_batch_norm(s1_conv, s1_bm, s1_bv, s1_bw, s1_bb, s1_eps, &s1_shape);
    let s1_out = b.add_relu(s1_bn, &s1_shape);

    // Stage 2: half resolution
    let s2_cw = b.add_input("s2_conv_w", &[STAGE2_CH, BACKBONE_CH, 3, 3]);
    let s2_cb = b.add_input("s2_conv_b", &[STAGE2_CH]);
    let s2_bm = b.add_input("s2_bn_mean", &[STAGE2_CH]);
    let s2_bv = b.add_input("s2_bn_var", &[STAGE2_CH]);
    let s2_bw = b.add_input("s2_bn_w", &[STAGE2_CH]);
    let s2_bb = b.add_input("s2_bn_b", &[STAGE2_CH]);
    let s2_eps = b.add_input("s2_eps", &[1]);

    let s2_shape = [STAGE2_CH, HALF_IMG, HALF_IMG];
    let s2_conv = b.add_conv2d(s1_out, s2_cw, Some(s2_cb), 2, 2, 1, 1, &s2_shape);
    let s2_bn = b.add_batch_norm(s2_conv, s2_bm, s2_bv, s2_bw, s2_bb, s2_eps, &s2_shape);
    let features = b.add_relu(s2_bn, &s2_shape);

    // --- Crop simulation: narrow to text region ---
    let crop_h = HALF_IMG / 2; // 8
    let cropped = b.add_narrow(features, 1, 0, crop_h, &[STAGE2_CH, crop_h, HALF_IMG]);
    let cropped2 = b.add_narrow(cropped, 2, 0, crop_h, &[STAGE2_CH, crop_h, crop_h]);

    // --- Recognition: project -> flatten -> MLP -> CTC ---
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, STAGE2_CH, 1, 1]);
    let proj = b.add_conv2d(
        cropped2,
        proj_w,
        None,
        1,
        1,
        0,
        0,
        &[HIDDEN_DIM, crop_h, crop_h],
    );

    let spatial = crop_h * crop_h; // 64
    let flat = b.add_reshape(proj, &[HIDDEN_DIM, spatial]);
    let trans = b.add_transpose(flat, &[1, 0], &[spatial, HIDDEN_DIM]);
    let seq = b.add_narrow(trans, 0, 0, SEQ_LEN, &[SEQ_LEN, HIDDEN_DIM]);

    // MLP encoder
    let fc1_w = b.add_input("fc1_w", &[FFN_DIM, HIDDEN_DIM]);
    let fc2_w = b.add_input("fc2_w", &[HIDDEN_DIM, FFN_DIM]);

    let h = b.add_linear(seq, fc1_w, None, &[SEQ_LEN, FFN_DIM]);
    let h_act = b.add_gelu(h, &[SEQ_LEN, FFN_DIM]);
    let encoded = b.add_linear(h_act, fc2_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let enc_out = b.add_binary_add(seq, encoded, &[SEQ_LEN, HIDDEN_DIM]);

    // CTC head + softmax
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(enc_out, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid PaddleOCR detect-recognize e2e kernel")
}

/// Bindings for detect-recognize e2e.
fn detect_recognize_e2e_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // image

    // Stage 1
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH, IN_CHANNELS, 3, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // Stage 2
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH, BACKBONE_CH, 3, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // Projection
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, STAGE2_CH, 1, 1]),
        WEIGHT_MAG,
    )));

    // MLP
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[FFN_DIM, HIDDEN_DIM]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, FFN_DIM]),
        WEIGHT_MAG,
    )));

    // CTC head
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[VOCAB_SIZE]),
        0.0f32,
    )));

    bindings
}

/// IBP through detection + recognition end-to-end with 2-stage backbone.
///
/// Deeper than test 17: uses stride-2 downsampling in the backbone before
/// crop, testing bounds propagation through multi-scale features.
#[test]
fn test_detect_recognize_e2e_ibp() {
    let def = build_detect_recognize_e2e_kernel();
    let bindings = detect_recognize_e2e_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR detect-recognize e2e");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "detect-recognize e2e output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR detect-recognize e2e IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

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
// 30. GELU activation bounds through MLP CROWN (narrow input)
// ===========================================================================

/// CROWN bounds through MLP GELU with narrow input range.
///
/// Uses a tighter input bound [-0.5, 0.5] to test CROWN's GELU
/// linearization with small perturbation regions where CROWN should
/// produce significantly tighter bounds than IBP.
/// The verdict is floored with a top-level plain IBP pass (`floor_with_ibp`), which
/// carries the LayerNorm → FFN Linear L2/Cauchy–Schwarz lever (ny). On this narrow
/// input ([-0.5, 0.5]) the lever's exact CS row bound collapses the FFN box, pulling
/// the achieved width from ~27.4 down to ~4.4 — clearing the <20 target soundly in a
/// deadline-bounded run (~25s). The lever's nominal is O(out + in) per Linear
/// (box-midpoint identity), so it is cheap enough to run by default; the floor bound
/// is deterministic regardless of how far the deadline-limited alpha-CROWN gets.
/// Intersection only tightens; the threshold is NOT weakened.
#[test]
fn test_svtr_mlp_gelu_narrow_input_crown() {
    let def = build_svtr_mlp_gelu_kernel();
    let bindings = svtr_mlp_gelu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "PaddleOCR SVTR MLP GELU narrow input: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // With narrow input, bounds should be tight
    assert!(
        hi_max - lo_min < 20.0,
        "narrow input should produce tight bounds, got width {}",
        hi_max - lo_min
    );
}

// ===========================================================================
// 31. SVTR 3-block encoder with final LayerNorm IBP
// ===========================================================================

/// Build a 3-block SVTR encoder with a final LayerNorm.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// 3 transformer blocks (attention + MLP each) followed by a final
/// LayerNorm, matching the SVTR architecture where the encoder output
/// is normalized before the CTC head.
fn build_svtr_three_block_final_ln_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_svtr_three_block_final_ln");

    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut current = input;

    for block_idx in 0..3 {
        let prefix = format!("b{block_idx}");

        // Attention sub-block
        let ln_a_w = b.add_input(&format!("{prefix}_ln1_w"), &[HIDDEN_DIM]);
        let ln_a_b = b.add_input(&format!("{prefix}_ln1_b"), &[HIDDEN_DIM]);
        let ln_a_eps = b.add_input(&format!("{prefix}_ln1_eps"), &[1]);
        let qw = b.add_input(&format!("{prefix}_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let kw = b.add_input(&format!("{prefix}_k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let vw = b.add_input(&format!("{prefix}_v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let ow = b.add_input(&format!("{prefix}_o_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let normed = b.add_layer_norm(current, ln_a_eps, 1, ln_a_w, ln_a_b, &shape);
        let q = b.add_linear(normed, qw, None, &shape);
        let k = b.add_linear(normed, kw, None, &shape);
        let v = b.add_linear(normed, vw, None, &shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
        let proj = b.add_linear(attn, ow, None, &shape);
        let res_a = b.add_binary_add(current, proj, &shape);

        // MLP sub-block
        let ln_b_w = b.add_input(&format!("{prefix}_ln2_w"), &[HIDDEN_DIM]);
        let ln_b_b = b.add_input(&format!("{prefix}_ln2_b"), &[HIDDEN_DIM]);
        let ln_b_eps = b.add_input(&format!("{prefix}_ln2_eps"), &[1]);
        let fc1 = b.add_input(&format!("{prefix}_fc1_w"), &[FFN_DIM, HIDDEN_DIM]);
        let fc2 = b.add_input(&format!("{prefix}_fc2_w"), &[HIDDEN_DIM, FFN_DIM]);

        let normed_b = b.add_layer_norm(res_a, ln_b_eps, 1, ln_b_w, ln_b_b, &shape);
        let h = b.add_linear(normed_b, fc1, None, &ffn_shape);
        let h_act = b.add_gelu(h, &ffn_shape);
        let mlp = b.add_linear(h_act, fc2, None, &shape);
        current = b.add_binary_add(res_a, mlp, &shape);
    }

    // Final LayerNorm
    let final_ln_w = b.add_input("final_ln_w", &[HIDDEN_DIM]);
    let final_ln_b = b.add_input("final_ln_b", &[HIDDEN_DIM]);
    let final_ln_eps = b.add_input("final_ln_eps", &[1]);
    let out = b.add_layer_norm(current, final_ln_eps, 1, final_ln_w, final_ln_b, &shape);

    b.build(out)
        .expect("valid PaddleOCR SVTR three-block final LN kernel")
}

/// Bindings for SVTR 3-block encoder with final LayerNorm.
fn svtr_three_block_final_ln_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // x

    for _ in 0..3 {
        // Attention sub-block
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        // MLP sub-block
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(fc1_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(fc2_w.clone()));
    }

    // Final LayerNorm
    bindings.push(TensorParamBinding::ConstantTensor(ln_w));
    bindings.push(TensorParamBinding::ConstantTensor(ln_b));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    bindings
}

/// IBP bounds through 3-block SVTR encoder with final LayerNorm.
///
/// The final LayerNorm re-normalizes the output after 3 transformer blocks,
/// testing whether bounds remain finite and non-degenerate through deep
/// composition with a normalizing output layer.
#[test]
fn test_svtr_three_block_final_ln_ibp() {
    let def = build_svtr_three_block_final_ln_kernel();
    let bindings = svtr_three_block_final_ln_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR SVTR three-block final LN");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "SVTR three-block final LN output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "PaddleOCR SVTR three-block final LN IBP (input [-1,1]): bounds=[{lo_min}, {hi_max}]"
    );

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

// ===========================================================================
// 32. SVTR patch-to-CTC with softmax end-to-end IBP
// ===========================================================================

/// Build full SVTR recognizer: patch embed -> 2 blocks -> CTC + softmax.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image in [0, 1]).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (character probabilities in [0, 1]).
///
/// Like test 16 but adds softmax after the CTC linear head, verifying
/// the complete recognition path produces valid probability distributions.
fn build_svtr_patch_to_ctc_softmax_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_svtr_patch_to_ctc_softmax");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // --- Patch embedding ---
    let patch_w = b.add_input(
        "patch_w",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_b = b.add_input("patch_b", &[HIDDEN_DIM]);

    let conv_out = b.add_conv2d(
        input,
        patch_w,
        Some(patch_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, NUM_PATCHES]);
    let patches = b.add_transpose(reshaped, &[1, 0], &[SEQ_LEN, HIDDEN_DIM]);

    // --- Block 1 ---
    let ln1a_w = b.add_input("b1_ln1_w", &[HIDDEN_DIM]);
    let ln1a_b = b.add_input("b1_ln1_b", &[HIDDEN_DIM]);
    let ln1a_eps = b.add_input("b1_ln1_eps", &[1]);
    let b1_qw = b.add_input("b1_qw", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b1_kw = b.add_input("b1_kw", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b1_vw = b.add_input("b1_vw", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b1_ow = b.add_input("b1_ow", &[HIDDEN_DIM, HIDDEN_DIM]);

    let n1a = b.add_layer_norm(patches, ln1a_eps, 1, ln1a_w, ln1a_b, &shape);
    let q1 = b.add_linear(n1a, b1_qw, None, &shape);
    let k1 = b.add_linear(n1a, b1_kw, None, &shape);
    let v1 = b.add_linear(n1a, b1_vw, None, &shape);
    let a1 = b.add_attention(q1, k1, v1, AttentionMask::Standard, Some(scale), &shape);
    let p1 = b.add_linear(a1, b1_ow, None, &shape);
    let r1a = b.add_binary_add(patches, p1, &shape);

    let ln1b_w = b.add_input("b1_ln2_w", &[HIDDEN_DIM]);
    let ln1b_b = b.add_input("b1_ln2_b", &[HIDDEN_DIM]);
    let ln1b_eps = b.add_input("b1_ln2_eps", &[1]);
    let b1_fc1 = b.add_input("b1_fc1", &[FFN_DIM, HIDDEN_DIM]);
    let b1_fc2 = b.add_input("b1_fc2", &[HIDDEN_DIM, FFN_DIM]);

    let n1b = b.add_layer_norm(r1a, ln1b_eps, 1, ln1b_w, ln1b_b, &shape);
    let h1 = b.add_linear(n1b, b1_fc1, None, &ffn_shape);
    let h1a = b.add_gelu(h1, &ffn_shape);
    let m1 = b.add_linear(h1a, b1_fc2, None, &shape);
    let r1b = b.add_binary_add(r1a, m1, &shape);

    // --- Block 2 ---
    let ln2a_w = b.add_input("b2_ln1_w", &[HIDDEN_DIM]);
    let ln2a_b = b.add_input("b2_ln1_b", &[HIDDEN_DIM]);
    let ln2a_eps = b.add_input("b2_ln1_eps", &[1]);
    let b2_qw = b.add_input("b2_qw", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b2_kw = b.add_input("b2_kw", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b2_vw = b.add_input("b2_vw", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b2_ow = b.add_input("b2_ow", &[HIDDEN_DIM, HIDDEN_DIM]);

    let n2a = b.add_layer_norm(r1b, ln2a_eps, 1, ln2a_w, ln2a_b, &shape);
    let q2 = b.add_linear(n2a, b2_qw, None, &shape);
    let k2 = b.add_linear(n2a, b2_kw, None, &shape);
    let v2 = b.add_linear(n2a, b2_vw, None, &shape);
    let a2 = b.add_attention(q2, k2, v2, AttentionMask::Standard, Some(scale), &shape);
    let p2 = b.add_linear(a2, b2_ow, None, &shape);
    let r2a = b.add_binary_add(r1b, p2, &shape);

    let ln2b_w = b.add_input("b2_ln2_w", &[HIDDEN_DIM]);
    let ln2b_b = b.add_input("b2_ln2_b", &[HIDDEN_DIM]);
    let ln2b_eps = b.add_input("b2_ln2_eps", &[1]);
    let b2_fc1 = b.add_input("b2_fc1", &[FFN_DIM, HIDDEN_DIM]);
    let b2_fc2 = b.add_input("b2_fc2", &[HIDDEN_DIM, FFN_DIM]);

    let n2b = b.add_layer_norm(r2a, ln2b_eps, 1, ln2b_w, ln2b_b, &shape);
    let h2 = b.add_linear(n2b, b2_fc1, None, &ffn_shape);
    let h2a = b.add_gelu(h2, &ffn_shape);
    let m2 = b.add_linear(h2a, b2_fc2, None, &shape);
    let enc_out = b.add_binary_add(r2a, m2, &shape);

    // --- CTC head + softmax ---
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(enc_out, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid PaddleOCR SVTR patch-to-CTC softmax kernel")
}

/// Bindings for SVTR patch-to-CTC with softmax.
fn svtr_patch_to_ctc_softmax_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // image

    // Patch embedding
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM]),
        0.0f32,
    )));

    // Blocks 1 and 2
    for _ in 0..2 {
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(fc1_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(fc2_w.clone()));
    }

    // CTC head
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[VOCAB_SIZE]),
        0.0f32,
    )));

    bindings
}

/// IBP through full SVTR recognizer with softmax: image -> character probs.
///
/// End-to-end recognition with softmax output. The softmax guarantees
/// output probabilities in [0, 1] for CTC decoding.
#[test]
fn test_svtr_patch_to_ctc_softmax_ibp() {
    let def = build_svtr_patch_to_ctc_softmax_kernel();
    let bindings = svtr_patch_to_ctc_softmax_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR SVTR patch-to-CTC softmax");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "SVTR patch-to-CTC softmax output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR SVTR patch-to-CTC softmax IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

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
// 33. SVTR MLP-GELU three blocks CROWN
// ===========================================================================

/// Build 3 sequential MLP-GELU blocks for deeper CROWN verification.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Three consecutive LayerNorm -> Linear -> GELU -> Linear -> residual
/// blocks. Tests CROWN stability through deeper GELU nonlinearity chains.
fn build_svtr_mlp_gelu_three_blocks_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_svtr_mlp_gelu_three_blocks");

    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    let mut current = input;

    for block_idx in 0..3 {
        let prefix = format!("b{block_idx}");

        let ln_w = b.add_input(&format!("{prefix}_ln_w"), &[HIDDEN_DIM]);
        let ln_b = b.add_input(&format!("{prefix}_ln_b"), &[HIDDEN_DIM]);
        let ln_eps = b.add_input(&format!("{prefix}_eps"), &[1]);
        let fc1 = b.add_input(&format!("{prefix}_fc1_w"), &[FFN_DIM, HIDDEN_DIM]);
        let fc2 = b.add_input(&format!("{prefix}_fc2_w"), &[HIDDEN_DIM, FFN_DIM]);

        let normed = b.add_layer_norm(current, ln_eps, 1, ln_w, ln_b, &shape);
        let h = b.add_linear(normed, fc1, None, &ffn_shape);
        let h_act = b.add_gelu(h, &ffn_shape);
        let mlp = b.add_linear(h_act, fc2, None, &shape);
        current = b.add_binary_add(current, mlp, &shape);
    }

    b.build(current)
        .expect("valid PaddleOCR SVTR MLP GELU three-blocks kernel")
}

/// Bindings for SVTR MLP-GELU three blocks.
fn svtr_mlp_gelu_three_blocks_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // x

    for _ in 0..3 {
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(fc1_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(fc2_w.clone()));
    }

    bindings
}

/// CROWN bounds through 3 sequential MLP-GELU blocks.
///
/// Deeper than the 2-block test (test 21). Tests whether CROWN
/// linearization remains stable through 3 consecutive GELU activations
/// with LayerNorm pre-normalization.
#[test]
fn test_svtr_mlp_gelu_three_blocks_crown() {
    let def = build_svtr_mlp_gelu_three_blocks_kernel();
    let bindings = svtr_mlp_gelu_three_blocks_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "PaddleOCR SVTR MLP GELU three-blocks: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 38. DB 4-stage ConvBnReLU backbone with progressive downsampling
// ===========================================================================

/// Stage 4 channel count for 4-stage backbone.
const STAGE4_CH: usize = 256;
/// 1/8 image dimension.
const EIGHTH_IMG: usize = IMG_SIZE / 8; // 4

/// Build a full 4-stage DB ResNet-like backbone.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image in [0, 1]).
/// Output: `[STAGE4_CH, EIGHTH_IMG, EIGHTH_IMG]`.
///
/// Stage 1: Conv(3, 32, k=3, s=1, p=1) -> BN -> ReLU  [32, 32, 32]
/// Stage 2: Conv(32, 64, k=3, s=2, p=1) -> BN -> ReLU  [64, 16, 16]
/// Stage 3: Conv(64, 128, k=3, s=2, p=1) -> BN -> ReLU [128, 8, 8]
/// Stage 4: Conv(128, 256, k=3, s=2, p=1) -> BN -> ReLU [256, 4, 4]
fn build_db_backbone_four_stage_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_db_backbone_four_stage");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // --- Stage 1: [3, 32, 32] -> [32, 32, 32] ---
    let s1_cw = b.add_input("s1_conv_w", &[BACKBONE_CH, IN_CHANNELS, 3, 3]);
    let s1_cb = b.add_input("s1_conv_b", &[BACKBONE_CH]);
    let s1_bm = b.add_input("s1_bn_mean", &[BACKBONE_CH]);
    let s1_bv = b.add_input("s1_bn_var", &[BACKBONE_CH]);
    let s1_bw = b.add_input("s1_bn_w", &[BACKBONE_CH]);
    let s1_bb = b.add_input("s1_bn_b", &[BACKBONE_CH]);
    let s1_eps = b.add_input("s1_eps", &[1]);

    let s1_shape = [BACKBONE_CH, IMG_SIZE, IMG_SIZE];
    let s1_conv = b.add_conv2d(input, s1_cw, Some(s1_cb), 1, 1, 1, 1, &s1_shape);
    let s1_bn = b.add_batch_norm(s1_conv, s1_bm, s1_bv, s1_bw, s1_bb, s1_eps, &s1_shape);
    let s1_out = b.add_relu(s1_bn, &s1_shape);

    // --- Stage 2: [32, 32, 32] -> [64, 16, 16] ---
    let s2_cw = b.add_input("s2_conv_w", &[STAGE2_CH, BACKBONE_CH, 3, 3]);
    let s2_cb = b.add_input("s2_conv_b", &[STAGE2_CH]);
    let s2_bm = b.add_input("s2_bn_mean", &[STAGE2_CH]);
    let s2_bv = b.add_input("s2_bn_var", &[STAGE2_CH]);
    let s2_bw = b.add_input("s2_bn_w", &[STAGE2_CH]);
    let s2_bb = b.add_input("s2_bn_b", &[STAGE2_CH]);
    let s2_eps = b.add_input("s2_eps", &[1]);

    let s2_shape = [STAGE2_CH, HALF_IMG, HALF_IMG];
    let s2_conv = b.add_conv2d(s1_out, s2_cw, Some(s2_cb), 2, 2, 1, 1, &s2_shape);
    let s2_bn = b.add_batch_norm(s2_conv, s2_bm, s2_bv, s2_bw, s2_bb, s2_eps, &s2_shape);
    let s2_out = b.add_relu(s2_bn, &s2_shape);

    // --- Stage 3: [64, 16, 16] -> [128, 8, 8] ---
    let s3_cw = b.add_input("s3_conv_w", &[STAGE3_CH, STAGE2_CH, 3, 3]);
    let s3_cb = b.add_input("s3_conv_b", &[STAGE3_CH]);
    let s3_bm = b.add_input("s3_bn_mean", &[STAGE3_CH]);
    let s3_bv = b.add_input("s3_bn_var", &[STAGE3_CH]);
    let s3_bw = b.add_input("s3_bn_w", &[STAGE3_CH]);
    let s3_bb = b.add_input("s3_bn_b", &[STAGE3_CH]);
    let s3_eps = b.add_input("s3_eps", &[1]);

    let s3_shape = [STAGE3_CH, QUARTER_IMG, QUARTER_IMG];
    let s3_conv = b.add_conv2d(s2_out, s3_cw, Some(s3_cb), 2, 2, 1, 1, &s3_shape);
    let s3_bn = b.add_batch_norm(s3_conv, s3_bm, s3_bv, s3_bw, s3_bb, s3_eps, &s3_shape);
    let s3_out = b.add_relu(s3_bn, &s3_shape);

    // --- Stage 4: [128, 8, 8] -> [256, 4, 4] ---
    let s4_cw = b.add_input("s4_conv_w", &[STAGE4_CH, STAGE3_CH, 3, 3]);
    let s4_cb = b.add_input("s4_conv_b", &[STAGE4_CH]);
    let s4_bm = b.add_input("s4_bn_mean", &[STAGE4_CH]);
    let s4_bv = b.add_input("s4_bn_var", &[STAGE4_CH]);
    let s4_bw = b.add_input("s4_bn_w", &[STAGE4_CH]);
    let s4_bb = b.add_input("s4_bn_b", &[STAGE4_CH]);
    let s4_eps = b.add_input("s4_eps", &[1]);

    let s4_shape = [STAGE4_CH, EIGHTH_IMG, EIGHTH_IMG];
    let s4_conv = b.add_conv2d(s3_out, s4_cw, Some(s4_cb), 2, 2, 1, 1, &s4_shape);
    let s4_bn = b.add_batch_norm(s4_conv, s4_bm, s4_bv, s4_bw, s4_bb, s4_eps, &s4_shape);
    let out = b.add_relu(s4_bn, &s4_shape);

    b.build(out)
        .expect("valid PaddleOCR DB backbone four-stage kernel")
}

/// Bindings for DB backbone four-stage.
fn db_backbone_four_stage_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // image

    let channels = [
        (BACKBONE_CH, IN_CHANNELS),
        (STAGE2_CH, BACKBONE_CH),
        (STAGE3_CH, STAGE2_CH),
        (STAGE4_CH, STAGE3_CH),
    ];
    for &(out_ch, in_ch) in &channels {
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[out_ch, in_ch, 3, 3]),
            WEIGHT_MAG,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[out_ch]),
            0.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[out_ch]),
            0.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[out_ch]),
            1.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[out_ch]),
            1.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[out_ch]),
            0.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    }

    bindings
}

/// IBP through 4-stage DB backbone with progressive channel/resolution scaling.
///
/// Tests bounds propagation through the full DB ResNet-like feature extractor.
/// Each stage halves spatial dimensions and increases channels. ReLU at each
/// stage prevents negative bound growth.
#[test]
fn test_db_backbone_four_stage_ibp() {
    let def = build_db_backbone_four_stage_kernel();
    let bindings = db_backbone_four_stage_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR DB backbone four-stage");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[STAGE4_CH, EIGHTH_IMG, EIGHTH_IMG],
        "DB backbone four-stage output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR DB backbone four-stage IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    // ReLU ensures lower >= 0
    assert!(
        lo_min >= -1e-4,
        "ReLU output lower should be >= 0, got {lo_min}"
    );
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 39. DB 4-stage backbone CROWN: tighter bounds through deep backbone
// ===========================================================================

/// CROWN bounds through the full 4-stage DB backbone.
///
/// Tests CROWN linearization stability through 4 sequential Conv-BN-ReLU
/// stages with downsampling. CROWN should produce tighter bounds than IBP
/// due to linear relaxation of ReLU.
#[test]
fn test_db_backbone_four_stage_crown() {
    let def = build_db_backbone_four_stage_kernel();
    let bindings = db_backbone_four_stage_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[STAGE4_CH, EIGHTH_IMG, EIGHTH_IMG],
        "DB backbone four-stage CROWN output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "PaddleOCR DB backbone four-stage CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(
        lo_min >= -1e-4,
        "ReLU output lower should be >= 0, got {lo_min}"
    );
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 40. DB FPN 3-scale fusion: multi-resolution feature pyramid
// ===========================================================================

/// Build DB FPN neck fusing 3 backbone stages (full, half, quarter resolution).
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image in [0, 1]).
/// Output: `[FPN_CH * 3, QUARTER_IMG, QUARTER_IMG]` (fused features).
///
/// Three backbone stages produce features at 32x32, 16x16, 8x8 resolution.
/// Each is projected to FPN_CH channels via 1x1 conv. High-res features are
/// narrowed to match lowest resolution, then concatenated along channels.
fn build_db_fpn_three_scale_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_db_fpn_three_scale");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // --- Stage 1: [3, 32, 32] -> [32, 32, 32] ---
    let s1_cw = b.add_input("s1_conv_w", &[BACKBONE_CH, IN_CHANNELS, 3, 3]);
    let s1_cb = b.add_input("s1_conv_b", &[BACKBONE_CH]);
    let s1_bm = b.add_input("s1_bn_mean", &[BACKBONE_CH]);
    let s1_bv = b.add_input("s1_bn_var", &[BACKBONE_CH]);
    let s1_bw = b.add_input("s1_bn_w", &[BACKBONE_CH]);
    let s1_bb = b.add_input("s1_bn_b", &[BACKBONE_CH]);
    let s1_eps = b.add_input("s1_eps", &[1]);

    let s1_shape = [BACKBONE_CH, IMG_SIZE, IMG_SIZE];
    let s1_conv = b.add_conv2d(input, s1_cw, Some(s1_cb), 1, 1, 1, 1, &s1_shape);
    let s1_bn = b.add_batch_norm(s1_conv, s1_bm, s1_bv, s1_bw, s1_bb, s1_eps, &s1_shape);
    let s1_feat = b.add_relu(s1_bn, &s1_shape);

    // --- Stage 2: [32, 32, 32] -> [64, 16, 16] ---
    let s2_cw = b.add_input("s2_conv_w", &[STAGE2_CH, BACKBONE_CH, 3, 3]);
    let s2_cb = b.add_input("s2_conv_b", &[STAGE2_CH]);
    let s2_bm = b.add_input("s2_bn_mean", &[STAGE2_CH]);
    let s2_bv = b.add_input("s2_bn_var", &[STAGE2_CH]);
    let s2_bw = b.add_input("s2_bn_w", &[STAGE2_CH]);
    let s2_bb = b.add_input("s2_bn_b", &[STAGE2_CH]);
    let s2_eps = b.add_input("s2_eps", &[1]);

    let s2_shape = [STAGE2_CH, HALF_IMG, HALF_IMG];
    let s2_conv = b.add_conv2d(s1_feat, s2_cw, Some(s2_cb), 2, 2, 1, 1, &s2_shape);
    let s2_bn = b.add_batch_norm(s2_conv, s2_bm, s2_bv, s2_bw, s2_bb, s2_eps, &s2_shape);
    let s2_feat = b.add_relu(s2_bn, &s2_shape);

    // --- Stage 3: [64, 16, 16] -> [128, 8, 8] ---
    let s3_cw = b.add_input("s3_conv_w", &[STAGE3_CH, STAGE2_CH, 3, 3]);
    let s3_cb = b.add_input("s3_conv_b", &[STAGE3_CH]);
    let s3_bm = b.add_input("s3_bn_mean", &[STAGE3_CH]);
    let s3_bv = b.add_input("s3_bn_var", &[STAGE3_CH]);
    let s3_bw = b.add_input("s3_bn_w", &[STAGE3_CH]);
    let s3_bb = b.add_input("s3_bn_b", &[STAGE3_CH]);
    let s3_eps = b.add_input("s3_eps", &[1]);

    let s3_shape = [STAGE3_CH, QUARTER_IMG, QUARTER_IMG];
    let s3_conv = b.add_conv2d(s2_feat, s3_cw, Some(s3_cb), 2, 2, 1, 1, &s3_shape);
    let s3_bn = b.add_batch_norm(s3_conv, s3_bm, s3_bv, s3_bw, s3_bb, s3_eps, &s3_shape);
    let s3_feat = b.add_relu(s3_bn, &s3_shape);

    // --- FPN lateral 1x1 convolutions ---
    // Stage 1 -> FPN_CH, narrowed to QUARTER_IMG
    let lat1_w = b.add_input("lat1_w", &[FPN_CH, BACKBONE_CH, 1, 1]);
    let lat1 = b.add_conv2d(
        s1_feat,
        lat1_w,
        None,
        1,
        1,
        0,
        0,
        &[FPN_CH, IMG_SIZE, IMG_SIZE],
    );
    let lat1_d1 = b.add_narrow(lat1, 1, 0, QUARTER_IMG, &[FPN_CH, QUARTER_IMG, IMG_SIZE]);
    let lat1_d2 = b.add_narrow(
        lat1_d1,
        2,
        0,
        QUARTER_IMG,
        &[FPN_CH, QUARTER_IMG, QUARTER_IMG],
    );

    // Stage 2 -> FPN_CH, narrowed to QUARTER_IMG
    let lat2_w = b.add_input("lat2_w", &[FPN_CH, STAGE2_CH, 1, 1]);
    let lat2 = b.add_conv2d(
        s2_feat,
        lat2_w,
        None,
        1,
        1,
        0,
        0,
        &[FPN_CH, HALF_IMG, HALF_IMG],
    );
    let lat2_d1 = b.add_narrow(lat2, 1, 0, QUARTER_IMG, &[FPN_CH, QUARTER_IMG, HALF_IMG]);
    let lat2_d2 = b.add_narrow(
        lat2_d1,
        2,
        0,
        QUARTER_IMG,
        &[FPN_CH, QUARTER_IMG, QUARTER_IMG],
    );

    // Stage 3 -> FPN_CH
    let lat3_w = b.add_input("lat3_w", &[FPN_CH, STAGE3_CH, 1, 1]);
    let lat3 = b.add_conv2d(
        s3_feat,
        lat3_w,
        None,
        1,
        1,
        0,
        0,
        &[FPN_CH, QUARTER_IMG, QUARTER_IMG],
    );

    // Concatenate all scales
    let fused_ch = FPN_CH * 3;
    let out = b.add_concat(
        &[lat1_d2, lat2_d2, lat3],
        0,
        &[fused_ch, QUARTER_IMG, QUARTER_IMG],
    );

    b.build(out)
        .expect("valid PaddleOCR DB FPN three-scale kernel")
}

/// Bindings for DB FPN three-scale.
fn db_fpn_three_scale_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // image

    // Stage 1
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH, IN_CHANNELS, 3, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BACKBONE_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // Stage 2
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH, BACKBONE_CH, 3, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE2_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // Stage 3
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE3_CH, STAGE2_CH, 3, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE3_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE3_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE3_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE3_CH]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[STAGE3_CH]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // FPN lateral convs
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[FPN_CH, BACKBONE_CH, 1, 1]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[FPN_CH, STAGE2_CH, 1, 1]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[FPN_CH, STAGE3_CH, 1, 1]),
        WEIGHT_MAG,
    )));

    bindings
}

/// IBP through DB FPN with 3-scale feature fusion.
///
/// Tests bounds propagation through multi-scale feature pyramid where features
/// from 3 backbone stages are projected, spatially narrowed, and concatenated.
#[test]
fn test_db_fpn_three_scale_ibp() {
    let def = build_db_fpn_three_scale_kernel();
    let bindings = db_fpn_three_scale_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR DB FPN three-scale");

    let fused_ch = FPN_CH * 3;
    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[fused_ch, QUARTER_IMG, QUARTER_IMG],
        "DB FPN three-scale output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR DB FPN three-scale IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 41. DB probability map pipeline: FPN -> 1x1 conv -> sigmoid
// ===========================================================================

/// Build the DB probability map head: 3-stage backbone -> 1x1 conv -> sigmoid.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image in [0, 1]).
/// Output: `[1, QUARTER_IMG, QUARTER_IMG]` (probability map in [0, 1]).
///
/// Uses a 3-stage backbone -> 1x1 conv -> sigmoid. This models the DB
/// text detection probability map output.
fn build_db_prob_map_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_db_prob_map");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // --- Stage 1 ---
    let s1_cw = b.add_input("s1_conv_w", &[BACKBONE_CH, IN_CHANNELS, 3, 3]);
    let s1_cb = b.add_input("s1_conv_b", &[BACKBONE_CH]);
    let s1_bm = b.add_input("s1_bn_mean", &[BACKBONE_CH]);
    let s1_bv = b.add_input("s1_bn_var", &[BACKBONE_CH]);
    let s1_bw = b.add_input("s1_bn_w", &[BACKBONE_CH]);
    let s1_bb = b.add_input("s1_bn_b", &[BACKBONE_CH]);
    let s1_eps = b.add_input("s1_eps", &[1]);

    let s1_shape = [BACKBONE_CH, IMG_SIZE, IMG_SIZE];
    let s1_conv = b.add_conv2d(input, s1_cw, Some(s1_cb), 1, 1, 1, 1, &s1_shape);
    let s1_bn = b.add_batch_norm(s1_conv, s1_bm, s1_bv, s1_bw, s1_bb, s1_eps, &s1_shape);
    let s1_feat = b.add_relu(s1_bn, &s1_shape);

    // --- Stage 2: downsample to HALF_IMG ---
    let s2_cw = b.add_input("s2_conv_w", &[STAGE2_CH, BACKBONE_CH, 3, 3]);
    let s2_cb = b.add_input("s2_conv_b", &[STAGE2_CH]);
    let s2_bm = b.add_input("s2_bn_mean", &[STAGE2_CH]);
    let s2_bv = b.add_input("s2_bn_var", &[STAGE2_CH]);
    let s2_bw = b.add_input("s2_bn_w", &[STAGE2_CH]);
    let s2_bb = b.add_input("s2_bn_b", &[STAGE2_CH]);
    let s2_eps = b.add_input("s2_eps", &[1]);

    let s2_shape = [STAGE2_CH, HALF_IMG, HALF_IMG];
    let s2_conv = b.add_conv2d(s1_feat, s2_cw, Some(s2_cb), 2, 2, 1, 1, &s2_shape);
    let s2_bn = b.add_batch_norm(s2_conv, s2_bm, s2_bv, s2_bw, s2_bb, s2_eps, &s2_shape);
    let s2_feat = b.add_relu(s2_bn, &s2_shape);

    // --- Stage 3: downsample to QUARTER_IMG ---
    let s3_cw = b.add_input("s3_conv_w", &[STAGE3_CH, STAGE2_CH, 3, 3]);
    let s3_cb = b.add_input("s3_conv_b", &[STAGE3_CH]);
    let s3_bm = b.add_input("s3_bn_mean", &[STAGE3_CH]);
    let s3_bv = b.add_input("s3_bn_var", &[STAGE3_CH]);
    let s3_bw = b.add_input("s3_bn_w", &[STAGE3_CH]);
    let s3_bb = b.add_input("s3_bn_b", &[STAGE3_CH]);
    let s3_eps = b.add_input("s3_eps", &[1]);

    let s3_shape = [STAGE3_CH, QUARTER_IMG, QUARTER_IMG];
    let s3_conv = b.add_conv2d(s2_feat, s3_cw, Some(s3_cb), 2, 2, 1, 1, &s3_shape);
    let s3_bn = b.add_batch_norm(s3_conv, s3_bm, s3_bv, s3_bw, s3_bb, s3_eps, &s3_shape);
    let s3_feat = b.add_relu(s3_bn, &s3_shape);

    // --- Probability head: 1x1 conv -> sigmoid ---
    let head_w = b.add_input("head_w", &[1, STAGE3_CH, 1, 1]);
    let head_b = b.add_input("head_b", &[1]);
    let logits = b.add_conv2d(
        s3_feat,
        head_w,
        Some(head_b),
        1,
        1,
        0,
        0,
        &[1, QUARTER_IMG, QUARTER_IMG],
    );
    let out = b.add_sigmoid(logits, &[1, QUARTER_IMG, QUARTER_IMG]);

    b.build(out).expect("valid PaddleOCR DB prob map kernel")
}

/// Bindings for DB probability map pipeline.
fn db_prob_map_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // image

    let stages: &[(usize, usize)] = &[
        (BACKBONE_CH, IN_CHANNELS),
        (STAGE2_CH, BACKBONE_CH),
        (STAGE3_CH, STAGE2_CH),
    ];
    for &(out_ch, in_ch) in stages {
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[out_ch, in_ch, 3, 3]),
            WEIGHT_MAG,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[out_ch]),
            0.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[out_ch]),
            0.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[out_ch]),
            1.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[out_ch]),
            1.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[out_ch]),
            0.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    }

    // Head
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1, STAGE3_CH, 1, 1]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.0f32,
    )));

    bindings
}

/// IBP through DB probability map: backbone -> sigmoid in [0, 1].
///
/// End-to-end test of the DB text detection head producing per-pixel
/// text probability. Sigmoid guarantees output in [0, 1].
#[test]
fn test_db_prob_map_ibp() {
    let def = build_db_prob_map_kernel();
    let bindings = db_prob_map_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR DB probability map");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[1, QUARTER_IMG, QUARTER_IMG],
        "DB prob map output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR DB prob map IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    // Sigmoid guarantees [0, 1]
    assert!(
        lo_min >= -1e-4,
        "prob map lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "prob map upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 42. DB threshold map: backbone -> 1x1 conv -> sigmoid (parallel head)
// ===========================================================================

/// Build the DB threshold map head (parallel to probability map).
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image in [0, 1]).
/// Output: `[1, HALF_IMG, HALF_IMG]` (threshold map in [0, 1]).
///
/// Uses a 2-stage backbone -> 1x1 conv -> sigmoid. In DB, the threshold map
/// is produced from the same backbone features as the probability map, but
/// with an independent head. Both produce sigmoid-bounded outputs.
fn build_db_threshold_map_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_db_threshold_map");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // --- Stage 1 ---
    let s1_cw = b.add_input("s1_conv_w", &[BACKBONE_CH, IN_CHANNELS, 3, 3]);
    let s1_cb = b.add_input("s1_conv_b", &[BACKBONE_CH]);
    let s1_bm = b.add_input("s1_bn_mean", &[BACKBONE_CH]);
    let s1_bv = b.add_input("s1_bn_var", &[BACKBONE_CH]);
    let s1_bw = b.add_input("s1_bn_w", &[BACKBONE_CH]);
    let s1_bb = b.add_input("s1_bn_b", &[BACKBONE_CH]);
    let s1_eps = b.add_input("s1_eps", &[1]);

    let s1_shape = [BACKBONE_CH, IMG_SIZE, IMG_SIZE];
    let s1_conv = b.add_conv2d(input, s1_cw, Some(s1_cb), 1, 1, 1, 1, &s1_shape);
    let s1_bn = b.add_batch_norm(s1_conv, s1_bm, s1_bv, s1_bw, s1_bb, s1_eps, &s1_shape);
    let s1_feat = b.add_relu(s1_bn, &s1_shape);

    // --- Stage 2 ---
    let s2_cw = b.add_input("s2_conv_w", &[STAGE2_CH, BACKBONE_CH, 3, 3]);
    let s2_cb = b.add_input("s2_conv_b", &[STAGE2_CH]);
    let s2_bm = b.add_input("s2_bn_mean", &[STAGE2_CH]);
    let s2_bv = b.add_input("s2_bn_var", &[STAGE2_CH]);
    let s2_bw = b.add_input("s2_bn_w", &[STAGE2_CH]);
    let s2_bb = b.add_input("s2_bn_b", &[STAGE2_CH]);
    let s2_eps = b.add_input("s2_eps", &[1]);

    let s2_shape = [STAGE2_CH, HALF_IMG, HALF_IMG];
    let s2_conv = b.add_conv2d(s1_feat, s2_cw, Some(s2_cb), 2, 2, 1, 1, &s2_shape);
    let s2_bn = b.add_batch_norm(s2_conv, s2_bm, s2_bv, s2_bw, s2_bb, s2_eps, &s2_shape);
    let s2_feat = b.add_relu(s2_bn, &s2_shape);

    // --- Threshold head: 1x1 conv -> sigmoid ---
    let th_w = b.add_input("th_w", &[1, STAGE2_CH, 1, 1]);
    let th_b = b.add_input("th_b", &[1]);
    let logits = b.add_conv2d(
        s2_feat,
        th_w,
        Some(th_b),
        1,
        1,
        0,
        0,
        &[1, HALF_IMG, HALF_IMG],
    );
    let out = b.add_sigmoid(logits, &[1, HALF_IMG, HALF_IMG]);

    b.build(out)
        .expect("valid PaddleOCR DB threshold map kernel")
}

/// Bindings for DB threshold map.
fn db_threshold_map_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // image

    let stages: &[(usize, usize)] = &[(BACKBONE_CH, IN_CHANNELS), (STAGE2_CH, BACKBONE_CH)];
    for &(out_ch, in_ch) in stages {
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[out_ch, in_ch, 3, 3]),
            WEIGHT_MAG,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[out_ch]),
            0.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[out_ch]),
            0.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[out_ch]),
            1.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[out_ch]),
            1.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[out_ch]),
            0.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    }

    // Threshold head
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1, STAGE2_CH, 1, 1]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.0f32,
    )));

    bindings
}

/// IBP through DB threshold map head.
///
/// The threshold map is the second output of the DB detector (alongside the
/// probability map). Both must produce outputs bounded in [0, 1] via sigmoid.
#[test]
fn test_db_threshold_map_ibp() {
    let def = build_db_threshold_map_kernel();
    let bindings = db_threshold_map_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR DB threshold map");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[1, HALF_IMG, HALF_IMG],
        "DB threshold map output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR DB threshold map IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    // Sigmoid guarantees [0, 1]
    assert!(
        lo_min >= -1e-4,
        "threshold map lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "threshold map upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 43. DB differentiable binarization: prob map CROWN
// ===========================================================================

/// CROWN bounds through the DB probability map pipeline.
///
/// Tests CROWN on the full detection head: Conv-BN-ReLU stages -> 1x1 conv
/// -> sigmoid. CROWN should tighten sigmoid bounds relative to IBP.
#[test]
fn test_db_prob_map_crown() {
    let def = build_db_prob_map_kernel();
    let bindings = db_prob_map_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[1, QUARTER_IMG, QUARTER_IMG],
        "DB prob map CROWN output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR DB prob map CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    // Sigmoid [0, 1]
    assert!(
        lo_min >= -1e-4,
        "prob map lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "prob map upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 44. DB full detector verify-and-record
// ===========================================================================

/// Verify and record the DB full detector for status tracking.
#[test]
fn test_db_full_detector_verify_and_record() {
    let def = build_db_full_detector_kernel();
    let bindings = db_full_detector_bindings();
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let result = verify_and_assert(&def, &bindings, &input, "paddle_ocr_db_full_detector");
    assert_eq!(result.num_variables, 1, "single Variable input (image)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[1, HALF_IMG, HALF_IMG]);
}

// ===========================================================================
// 45. SVTR 8-block deep encoder IBP
// ===========================================================================

/// Build an 8-block SVTR encoder for deep verification stress test.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Tests IBP stability through 8 consecutive transformer blocks, each with
/// LayerNorm -> attention -> residual -> LayerNorm -> GELU MLP -> residual.
fn build_svtr_eight_block_encoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_svtr_eight_block_encoder");

    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut current = input;

    for block_idx in 0..8 {
        let prefix = format!("b{block_idx}");

        // Attention sub-block
        let ln_a_w = b.add_input(&format!("{prefix}_ln1_w"), &[HIDDEN_DIM]);
        let ln_a_b = b.add_input(&format!("{prefix}_ln1_b"), &[HIDDEN_DIM]);
        let ln_a_eps = b.add_input(&format!("{prefix}_ln1_eps"), &[1]);
        let qw = b.add_input(&format!("{prefix}_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let kw = b.add_input(&format!("{prefix}_k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let vw = b.add_input(&format!("{prefix}_v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let ow = b.add_input(&format!("{prefix}_o_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let normed = b.add_layer_norm(current, ln_a_eps, 1, ln_a_w, ln_a_b, &shape);
        let q = b.add_linear(normed, qw, None, &shape);
        let k = b.add_linear(normed, kw, None, &shape);
        let v = b.add_linear(normed, vw, None, &shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
        let proj = b.add_linear(attn, ow, None, &shape);
        let res_a = b.add_binary_add(current, proj, &shape);

        // MLP sub-block
        let ln_b_w = b.add_input(&format!("{prefix}_ln2_w"), &[HIDDEN_DIM]);
        let ln_b_b = b.add_input(&format!("{prefix}_ln2_b"), &[HIDDEN_DIM]);
        let ln_b_eps = b.add_input(&format!("{prefix}_ln2_eps"), &[1]);
        let fc1_w = b.add_input(&format!("{prefix}_fc1_w"), &[FFN_DIM, HIDDEN_DIM]);
        let fc2_w = b.add_input(&format!("{prefix}_fc2_w"), &[HIDDEN_DIM, FFN_DIM]);

        let normed2 = b.add_layer_norm(res_a, ln_b_eps, 1, ln_b_w, ln_b_b, &shape);
        let h = b.add_linear(normed2, fc1_w, None, &ffn_shape);
        let h_act = b.add_gelu(h, &ffn_shape);
        let mlp_out = b.add_linear(h_act, fc2_w, None, &shape);
        current = b.add_binary_add(res_a, mlp_out, &shape);
    }

    b.build(current)
        .expect("valid PaddleOCR SVTR eight-block encoder kernel")
}

/// Bindings for SVTR eight-block encoder.
fn svtr_eight_block_encoder_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // x

    for _ in 0..8 {
        // Attention sub-block
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        // MLP sub-block
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(fc1_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(fc2_w.clone()));
    }

    bindings
}

/// IBP through 8-block SVTR encoder.
///
/// Deep stress test: 8 transformer blocks with LayerNorm + attention + GELU.
/// Tests whether IBP bounds remain finite and non-degenerate through repeated
/// normalization and nonlinearity layers.
#[test]
fn test_svtr_eight_block_encoder_ibp() {
    let def = build_svtr_eight_block_encoder_kernel();
    let bindings = svtr_eight_block_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR SVTR eight-block encoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "SVTR eight-block encoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR SVTR eight-block encoder IBP (input [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

// ===========================================================================
// 46. SVTR patch embedding at width 48 (narrow variant)
// ===========================================================================

/// Narrower SVTR hidden dimension for small-model variant.
const NARROW_DIM: usize = 48;
/// Narrower FFN dimension.
const NARROW_FFN: usize = 96;
/// Head dim for narrow model.
const NARROW_HEAD_DIM: usize = NARROW_DIM / NUM_HEADS; // 12

/// Build SVTR patch embedding with narrow (48-dim) hidden size.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image in [0, 1]).
/// Output: `[NUM_PATCHES, NARROW_DIM]`.
///
/// Tests a smaller SVTR variant (PP-OCRv3 mobile) with reduced hidden dimension.
fn build_svtr_patch_embed_narrow_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_svtr_patch_embed_narrow");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let patch_w = b.add_input(
        "patch_w",
        &[NARROW_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_b = b.add_input("patch_b", &[NARROW_DIM]);

    let conv_out = b.add_conv2d(
        input,
        patch_w,
        Some(patch_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[NARROW_DIM, GRID_SIZE, GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv_out, &[NARROW_DIM, NUM_PATCHES]);
    let out = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, NARROW_DIM]);

    b.build(out)
        .expect("valid PaddleOCR SVTR patch embed narrow kernel")
}

/// Bindings for narrow SVTR patch embedding.
fn svtr_patch_embed_narrow_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // image
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NARROW_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NARROW_DIM]), 0.0f32)),
    ]
}

/// IBP through narrow (48-dim) SVTR patch embedding.
///
/// Tests bounds propagation through the mobile-variant patch embedding
/// with reduced hidden dimension.
#[test]
fn test_svtr_patch_embed_narrow_ibp() {
    let def = build_svtr_patch_embed_narrow_kernel();
    let bindings = svtr_patch_embed_narrow_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR SVTR patch embed narrow");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_PATCHES, NARROW_DIM],
        "narrow patch embed output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR SVTR patch embed narrow IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 47. SVTR patch embedding at width 96 (wide variant)
// ===========================================================================

/// Wider SVTR hidden dimension for large-model variant.
const WIDE_DIM: usize = 96;

/// Build SVTR patch embedding with wide (96-dim) hidden size.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image in [0, 1]).
/// Output: `[NUM_PATCHES, WIDE_DIM]`.
///
/// Tests a larger SVTR variant with increased hidden dimension, exercising
/// wider weight matrices during bounds propagation.
fn build_svtr_patch_embed_wide_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_svtr_patch_embed_wide");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let patch_w = b.add_input("patch_w", &[WIDE_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]);
    let patch_b = b.add_input("patch_b", &[WIDE_DIM]);

    let conv_out = b.add_conv2d(
        input,
        patch_w,
        Some(patch_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[WIDE_DIM, GRID_SIZE, GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv_out, &[WIDE_DIM, NUM_PATCHES]);
    let out = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, WIDE_DIM]);

    b.build(out)
        .expect("valid PaddleOCR SVTR patch embed wide kernel")
}

/// Bindings for wide SVTR patch embedding.
fn svtr_patch_embed_wide_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // image
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[WIDE_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[WIDE_DIM]), 0.0f32)),
    ]
}

/// IBP through wide (96-dim) SVTR patch embedding.
///
/// Tests bounds propagation with wider hidden dimension. Wider matrices
/// accumulate more terms per output element, testing IBP over-approximation
/// scaling with dimension.
#[test]
fn test_svtr_patch_embed_wide_ibp() {
    let def = build_svtr_patch_embed_wide_kernel();
    let bindings = svtr_patch_embed_wide_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR SVTR patch embed wide");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_PATCHES, WIDE_DIM],
        "wide patch embed output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR SVTR patch embed wide IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 48. CTC decoding pipeline: encoder -> LN -> linear -> softmax
// ===========================================================================

/// Build the full CTC decoding pipeline: LayerNorm -> linear -> softmax.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, encoder output).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (character probabilities in [0, 1]).
///
/// Models the CTC decoder that takes SVTR encoder output and produces
/// per-timestep character probability distributions via softmax.
fn build_ctc_decode_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_ctc_decode_pipeline");

    let input = b.add_input("encoder_out", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // LayerNorm before CTC head
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let normed = b.add_layer_norm(input, ln_eps, 1, ln_w, ln_b, &shape);

    // Linear projection to vocabulary
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(normed, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);

    // Softmax to probabilities
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid PaddleOCR CTC decode pipeline kernel")
}

/// Bindings for CTC decode pipeline.
fn ctc_decode_pipeline_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // encoder_out
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)),
    ]
}

/// CROWN through CTC decode pipeline with LayerNorm pre-normalization.
///
/// LayerNorm requires CROWN linearization (IbpValidated mode). Tests that
/// CROWN produces valid [0, 1] softmax bounds through the full decode head.
#[test]
fn test_ctc_decode_pipeline_crown() {
    let def = build_ctc_decode_pipeline_kernel();
    let bindings = ctc_decode_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "CTC decode pipeline CROWN output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "PaddleOCR CTC decode pipeline CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    // Softmax output in [0, 1]
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
// 49. GELU approximation bounds: very narrow input range
// ===========================================================================

/// Build a standalone GELU-based MLP for bounds testing.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Linear -> GELU -> Linear with narrow input range to test GELU
/// approximation quality in the region near zero.
fn build_gelu_approx_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_gelu_approx");

    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let fc1_w = b.add_input("fc1_w", &[FFN_DIM, HIDDEN_DIM]);
    let fc2_w = b.add_input("fc2_w", &[HIDDEN_DIM, FFN_DIM]);

    let h = b.add_linear(input, fc1_w, None, &[SEQ_LEN, FFN_DIM]);
    let h_act = b.add_gelu(h, &[SEQ_LEN, FFN_DIM]);
    let out = b.add_linear(h_act, fc2_w, None, &[SEQ_LEN, HIDDEN_DIM]);

    b.build(out).expect("valid PaddleOCR GELU approx kernel")
}

/// Bindings for GELU approximation test.
fn gelu_approx_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // x
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, FFN_DIM]),
            WEIGHT_MAG,
        )),
    ]
}

/// CROWN through GELU MLP with very narrow input range [-0.1, 0.1].
///
/// Near-zero inputs exercise the quasi-linear region of GELU where CROWN
/// linearization should produce very tight bounds. Validates that the GELU
/// approximation does not blow up bounds in the linear region.
#[test]
fn test_gelu_approx_very_narrow_crown() {
    let def = build_gelu_approx_kernel();
    let bindings = gelu_approx_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.1);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "PaddleOCR GELU approx very-narrow input: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Very narrow input should produce tight output bounds
    assert!(
        hi_max - lo_min < 5.0,
        "very narrow input should produce tight bounds, got width {}",
        hi_max - lo_min
    );
}

// ===========================================================================
// 50. SVTR attention + position encoding (sinusoidal additive)
// ===========================================================================

/// Build SVTR attention block with additive position encoding.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Adds a constant sinusoidal position encoding before the attention block,
/// modeling how SVTR adds positional information to patch embeddings.
fn build_svtr_attn_pos_enc_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_svtr_attn_pos_enc");

    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let pos_enc = b.add_input("pos_enc", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Add position encoding
    let x_pos = b.add_binary_add(input, pos_enc, &shape);

    // LayerNorm -> attention -> residual
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let qw = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let kw = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let vw = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let ow = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let normed = b.add_layer_norm(x_pos, ln_eps, 1, ln_w, ln_b, &shape);
    let q = b.add_linear(normed, qw, None, &shape);
    let k = b.add_linear(normed, kw, None, &shape);
    let v = b.add_linear(normed, vw, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let proj = b.add_linear(attn, ow, None, &shape);
    let out = b.add_binary_add(x_pos, proj, &shape);

    b.build(out)
        .expect("valid PaddleOCR SVTR attn pos enc kernel")
}

/// Bindings for SVTR attention with position encoding.
fn svtr_attn_pos_enc_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    // Sinusoidal position encoding has values in roughly [-1, 1]
    let pos_enc = ArrayD::from_elem(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), 0.1f32);

    vec![
        TensorParamBinding::Variable, // x
        TensorParamBinding::ConstantTensor(pos_enc),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w),
    ]
}

/// CROWN through SVTR attention with position encoding.
///
/// The additive position encoding shifts input bounds before attention.
/// Tests that CROWN handles the shifted input domain correctly and produces
/// valid bounds through the full attention + residual block.
#[test]
fn test_svtr_attn_pos_enc_crown() {
    let def = build_svtr_attn_pos_enc_kernel();
    let bindings = svtr_attn_pos_enc_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR SVTR attn+pos_enc CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 51. SVTR 4-block encoder CROWN (attention + MLP, tighter than IBP)
// ===========================================================================

/// CROWN bounds through a 4-block SVTR encoder.
///
/// Reuses the 4-block encoder kernel from test 12 but with CROWN. Tests
/// CROWN stability through 4 consecutive transformer blocks with
/// LayerNorm linearization at each block.
// Retired: ny's batched-CROWN fast-fail (crown_batched.rs) skips the wasted per-node
// CROWN-IBP intermediate collection that was discarded anyway when the fused SelfAttention
// node aborts the batched backward; the bit-identical bound is produced by the fixed-slope
// fallback. ~4s in release (the "~244s" ignore was stale); heavier in debug (the
// fixed-slope CROWN backward over 4 stacked blocks is O(L*S^2)) but sound and passing.
#[test]
fn test_svtr_four_block_encoder_crown() {
    let def = build_svtr_four_block_encoder_kernel();
    let bindings = svtr_four_block_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "SVTR four-block encoder CROWN output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "PaddleOCR SVTR four-block encoder CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 52. Full detection -> recognition end-to-end CROWN
// ===========================================================================

/// CROWN bounds through the full OCR pipeline (detection + recognition).
///
/// Reuses the full OCR pipeline kernel from test 10 but with CROWN.
/// Tests CROWN's ability to handle the composite pipeline where detection
/// feeds into recognition through a simulated crop operation.
#[test]
fn test_full_ocr_pipeline_crown() {
    let def = build_full_ocr_pipeline_kernel();
    let bindings = full_ocr_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "full OCR pipeline CROWN output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR full OCR pipeline CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    // Softmax output in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "OCR output lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "OCR output upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 53. Multi-line OCR composition: N detection crops -> N recognitions
// ===========================================================================

/// Build a 2-line OCR pipeline: two parallel recognition paths.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image in [0, 1]).
/// Output: `[total_seq, VOCAB_SIZE]` (two lines concatenated).
///
/// Simulates multi-line OCR: the image is split into two vertical halves,
/// each processed by a shared patch embedding + CTC head, then concatenated.
fn build_multiline_ocr_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_multiline_ocr");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Shared patch embedding weights
    let patch_w = b.add_input(
        "patch_w",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_b = b.add_input("patch_b", &[HIDDEN_DIM]);

    let half_grid_h = HALF_IMG / PATCH_SIZE; // 2
    let half_patches = half_grid_h * GRID_SIZE; // 8

    // --- Line 1: top half of image ---
    let line1 = b.add_narrow(input, 1, 0, HALF_IMG, &[IN_CHANNELS, HALF_IMG, IMG_SIZE]);
    let line1_conv = b.add_conv2d(
        line1,
        patch_w,
        Some(patch_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, half_grid_h, GRID_SIZE],
    );
    let line1_flat = b.add_reshape(line1_conv, &[HIDDEN_DIM, half_patches]);
    let line1_t = b.add_transpose(line1_flat, &[1, 0], &[half_patches, HIDDEN_DIM]);

    // --- Line 2: bottom half of image ---
    let line2 = b.add_narrow(
        input,
        1,
        HALF_IMG,
        HALF_IMG,
        &[IN_CHANNELS, HALF_IMG, IMG_SIZE],
    );
    let line2_conv = b.add_conv2d(
        line2,
        patch_w,
        Some(patch_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, half_grid_h, GRID_SIZE],
    );
    let line2_flat = b.add_reshape(line2_conv, &[HIDDEN_DIM, half_patches]);
    let line2_t = b.add_transpose(line2_flat, &[1, 0], &[half_patches, HIDDEN_DIM]);

    // --- Shared CTC head ---
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);

    let logits1 = b.add_linear(line1_t, ctc_w, Some(ctc_b), &[half_patches, VOCAB_SIZE]);
    let probs1 = b.add_softmax(logits1, 1, &[half_patches, VOCAB_SIZE]);

    let logits2 = b.add_linear(line2_t, ctc_w, Some(ctc_b), &[half_patches, VOCAB_SIZE]);
    let probs2 = b.add_softmax(logits2, 1, &[half_patches, VOCAB_SIZE]);

    // Concatenate line results
    let total_seq = half_patches * 2;
    let out = b.add_concat(&[probs1, probs2], 0, &[total_seq, VOCAB_SIZE]);

    b.build(out).expect("valid PaddleOCR multiline OCR kernel")
}

/// Bindings for multiline OCR pipeline.
fn multiline_ocr_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // image
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)),
    ]
}

/// IBP through multi-line OCR: two parallel recognition paths.
///
/// Tests bounds propagation through a branching pipeline where the same
/// image is split into regions, each independently recognized, then
/// concatenated. Softmax on each branch independently bounds outputs.
#[test]
fn test_multiline_ocr_ibp() {
    let half_grid_h = HALF_IMG / PATCH_SIZE;
    let half_patches = half_grid_h * GRID_SIZE;
    let total_seq = half_patches * 2;

    let def = build_multiline_ocr_kernel();
    let bindings = multiline_ocr_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR multiline OCR");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[total_seq, VOCAB_SIZE],
        "multiline OCR output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR multiline OCR IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    // Softmax on each line -> [0, 1]
    assert!(
        lo_min >= -1e-4,
        "multiline OCR lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "multiline OCR upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 54. Confidence filtering: CTC softmax -> narrow -> threshold check
// ===========================================================================

/// Build a CTC confidence extraction pipeline: softmax -> narrow to first class.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, encoder output).
/// Output: `[SEQ_LEN, 1]` (narrowed to first class probability).
///
/// Models the confidence extraction step where after CTC softmax, we examine
/// per-timestep confidence by narrowing to a specific class. Here we
/// approximate by narrowing to the first class (index 0).
fn build_ctc_confidence_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("paddle_ocr_ctc_confidence");

    let input = b.add_input("encoder_out", &[SEQ_LEN, HIDDEN_DIM]);

    // Linear -> softmax
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(input, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    // Narrow to first class (simulates argmax selection)
    let out = b.add_narrow(probs, 1, 0, 1, &[SEQ_LEN, 1]);

    b.build(out).expect("valid PaddleOCR CTC confidence kernel")
}

/// Bindings for CTC confidence pipeline.
fn ctc_confidence_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // encoder_out
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)),
    ]
}

/// IBP through CTC confidence extraction: softmax -> narrow.
///
/// After narrowing, per-timestep confidence values should remain in [0, 1]
/// since they are a subset of the softmax output.
#[test]
fn test_ctc_confidence_ibp() {
    let def = build_ctc_confidence_kernel();
    let bindings = ctc_confidence_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR CTC confidence");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[SEQ_LEN, 1],
        "CTC confidence output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR CTC confidence IBP (encoder [-2,2]): bounds=[{lo_min}, {hi_max}]");

    // Subset of softmax -> [0, 1]
    assert!(
        lo_min >= -1e-4,
        "confidence lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "confidence upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 55. SVTR encoder + CTC + softmax full recognizer verify-and-record
// ===========================================================================

/// Verify and record the full SVTR patch-to-CTC-softmax pipeline.
#[test]
fn test_svtr_patch_to_ctc_softmax_verify_and_record() {
    let def = build_svtr_patch_to_ctc_softmax_kernel();
    let bindings = svtr_patch_to_ctc_softmax_bindings();
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "paddle_ocr_svtr_patch_to_ctc_softmax",
    );
    assert_eq!(result.num_variables, 1, "single Variable input (image)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, VOCAB_SIZE]);
}
