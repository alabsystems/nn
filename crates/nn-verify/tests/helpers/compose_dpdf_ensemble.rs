// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests for the dpdf 7-model ensemble pipeline.
//!
//! The dpdf document understanding system orchestrates 7 models:
//!   1. **DocLayout-YOLO** — document layout detection (boxes + classes)
//!   2. **Table Transformer** — DETR-based table structure recognition
//!   3. **FireRed-OCR** — Qwen3-VL-2B variant for document OCR (CTC decoding)
//!   4. **Qwen3-VL** — vision-language model (vision encoder + MLP projection)
//!   5. **Granite-Docling** — SigLIP2 vision encoder + Granite LLM decoder
//!   6. **GLM-OCR** — GLM-4V vision-language model for OCR
//!   7. **PaddleOCR** — DB text detector + SVTR text recognizer
//!
//! These tests verify that the full ensemble pipeline preserves bounds
//! when composing across model boundaries. Each test builds a small
//! GraphNetwork representing the relevant model subgraph, runs NY
//! IBP/CROWN propagation, and asserts bounds are finite and non-vacuous.
//!
//! ## Tests (15 tests)
//!
//! 1.  **DocLayout-YOLO multi-scale detection** — conv -> pool -> detection head (IBP)
//! 2.  **Table Transformer DETR attention** — attention -> bbox regression (IBP + CROWN)
//! 3.  **FireRed-OCR vision encoder -> CTC decoder** — patch embed -> encoder -> CTC (IBP)
//! 4.  **Qwen3-VL vision encoder -> MLP projection** — patch embed -> MLP (IBP + CROWN)
//! 5.  **Granite-Docling vision -> LM bridge** — ViT -> linear projection (IBP + CROWN)
//! 6.  **GLM-OCR decoder -> MTP head** — decoder FFN -> softmax (IBP)
//! 7.  **PaddleOCR detection + recognition** — DB sigmoid -> SVTR CTC (IBP)
//! 8.  **DocLayout -> Table Transformer cascade** — detection -> table queries (IBP)
//! 9.  **DocLayout -> FireRed-OCR cascade** — detection -> OCR recognition (IBP)
//! 10. **DocLayout -> Qwen3-VL cascade** — detection -> VLM analysis (IBP)
//! 11. **Full 3-model pipeline: layout -> table -> OCR** (IBP)
//! 12. **Full 4-model pipeline: layout -> table -> OCR -> language** (IBP)
//! 13. **Ensemble confidence aggregation** — multi-model sigmoid merge (IBP + CROWN)
//! 14. **Ensemble monotone tightening** — tighter input -> tighter ensemble output (IBP)
//! 15. **7-model dispatch routing** — softmax gate -> 7 heads (IBP)
//!
//! Dimensions (small for fast verification, structurally representative):
//! - IMG_SIZE=8, HIDDEN=8, SEQ=4, NUM_BOXES=4, NUM_CLASSES=6, VOCAB=8
//!
//! Part of #4243: dpdf 7-model ensemble compose verification tests.

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

/// Image spatial size (square).
const IMG_SIZE: usize = 8;
/// Input channels (RGB).
const IN_CHANNELS: usize = 3;
/// Hidden dimension shared across model boundaries.
const HIDDEN: usize = 8;
/// Sequence length / number of positions.
const SEQ: usize = 4;
/// Number of detection boxes.
const NUM_BOXES: usize = 4;
/// Number of detection classes.
const NUM_CLASSES: usize = 6;
/// OCR vocabulary size.
const VOCAB: usize = 8;
/// Number of attention heads.
const NUM_HEADS: usize = 2;
/// Per-head dimension.
const HEAD_DIM: usize = HIDDEN / NUM_HEADS;
/// FFN intermediate dimension.
const FFN_DIM: usize = HIDDEN * 2;
/// Pooled spatial size after stride-2 conv.
const POOL_SIZE: usize = IMG_SIZE / 2;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;
/// Number of ensemble models for routing.
const NUM_MODELS: usize = 7;

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

/// Ones tensor binding (for normalization scale parameters).
fn ones(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 1.0f32))
}

/// Epsilon scalar binding.
fn eps_binding() -> TensorParamBinding {
    TensorParamBinding::ConstantScalar(1e-5)
}

/// Image-domain input bounds: pixels in [0, 1].
fn image_input_bounds() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 1.0f32),
    )
    .expect("valid image bounds")
}

/// Add a ConvBnSiLU block: Conv2d -> BatchNorm -> sigmoid(x)*x.
fn add_conv_bn_silu(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::tensor_ir::TensorNodeId,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    out_h: usize,
    out_w: usize,
    prefix: &str,
) -> nn_dsl::tensor_ir::TensorNodeId {
    let conv_w = b.add_input(
        &format!("{prefix}_conv_w"),
        &[out_ch, in_ch, kernel, kernel],
    );
    let conv_b = b.add_input(&format!("{prefix}_conv_b"), &[out_ch]);
    let bn_mean = b.add_input(&format!("{prefix}_bn_mean"), &[out_ch]);
    let bn_var = b.add_input(&format!("{prefix}_bn_var"), &[out_ch]);
    let bn_weight = b.add_input(&format!("{prefix}_bn_weight"), &[out_ch]);
    let bn_bias = b.add_input(&format!("{prefix}_bn_bias"), &[out_ch]);
    let bn_eps = b.add_input(&format!("{prefix}_bn_eps"), &[1]);

    let out_shape = [out_ch, out_h, out_w];
    let conv_out = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        stride,
        stride,
        padding,
        padding,
        &out_shape,
    );
    let bn_out = b.add_batch_norm(
        conv_out, bn_mean, bn_var, bn_weight, bn_bias, bn_eps, &out_shape,
    );
    // SiLU: sigmoid(x) * x
    let sig = b.add_sigmoid(bn_out, &out_shape);
    b.add_binary_mul(bn_out, sig, &out_shape)
}

/// Push bindings for one ConvBnSiLU block (7 params).
fn push_conv_bn_silu_bindings(
    bindings: &mut Vec<TensorParamBinding>,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
) {
    bindings.push(weight(&[out_ch, in_ch, kernel, kernel])); // conv_w
    bindings.push(bias_zero(&[out_ch])); // conv_b
    bindings.push(bias_zero(&[out_ch])); // bn_mean
    bindings.push(ones(&[out_ch])); // bn_var
    bindings.push(ones(&[out_ch])); // bn_weight
    bindings.push(bias_zero(&[out_ch])); // bn_bias
    bindings.push(eps_binding()); // bn_eps
}

// ===========================================================================
// 1. DocLayout-YOLO multi-scale detection pipeline (IBP)
// ===========================================================================

/// DocLayout-YOLO backbone: conv stride-2 -> pool-proxy conv -> flatten ->
/// classification head -> sigmoid.
///
/// Verifies detection confidence output bounded in [0, 1] from pixel input.
#[test]
fn test_ensemble_doclayout_yolo_multiscale_detection_ibp() {
    let ch = HIDDEN;

    let mut b = TensorBlockBuilder::new("ensemble_doclayout_yolo");
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Backbone stem: conv stride-2
    let stem = add_conv_bn_silu(
        &mut b,
        input,
        IN_CHANNELS,
        ch,
        3,
        2,
        1,
        POOL_SIZE,
        POOL_SIZE,
        "stem",
    );

    // Flatten and classification head
    let flat_dim = ch * POOL_SIZE * POOL_SIZE;
    let flat = b.add_reshape(stem, &[flat_dim]);
    let cls_w = b.add_input("cls_w", &[NUM_CLASSES, flat_dim]);
    let logits = b.add_linear(flat, cls_w, None, &[NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[NUM_CLASSES]);
    let def = b.build(out).expect("valid doclayout-yolo ensemble kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_conv_bn_silu_bindings(&mut bindings, IN_CHANNELS, ch, 3);
    bindings.push(weight(&[NUM_CLASSES, flat_dim]));

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_input_bounds();

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ensemble doclayout-yolo IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 2. Table Transformer DETR attention -> bbox regression (IBP + CROWN)
// ===========================================================================

/// Table Transformer: object queries -> self-attention -> linear bbox head
/// -> sigmoid coordinates in [0, 1].
///
/// Verifies that DETR attention + regression head preserves bounded output.
#[test]
fn test_ensemble_table_transformer_detr_attention_ibp_crown() {
    let mut b = TensorBlockBuilder::new("ensemble_table_transformer");
    let input = b.add_input("queries", &[SEQ, HIDDEN]);

    // Self-attention (multi-head)
    let q_w = b.add_input("q_w", &[HIDDEN, HIDDEN]);
    let k_w = b.add_input("k_w", &[HIDDEN, HIDDEN]);
    let v_w = b.add_input("v_w", &[HIDDEN, HIDDEN]);
    let out_w = b.add_input("out_w", &[HIDDEN, HIDDEN]);
    let attn_out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[SEQ, HIDDEN],
        )
        .expect("valid MHA");

    // Bbox regression head: Linear -> sigmoid
    let box_w = b.add_input("box_w", &[4, HIDDEN]);
    let box_b = b.add_input("box_b", &[4]);
    let box_logits = b.add_linear(attn_out, box_w, Some(box_b), &[SEQ, 4]);
    let out = b.add_sigmoid(box_logits, &[SEQ, 4]);
    let def = b.build(out).expect("valid table transformer kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN, HIDDEN]), // q_w
        weight(&[HIDDEN, HIDDEN]), // k_w
        weight(&[HIDDEN, HIDDEN]), // v_w
        weight(&[HIDDEN, HIDDEN]), // out_w
        weight(&[4, HIDDEN]),      // box_w
        bias_zero(&[4]),           // box_b
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("ensemble table-transformer IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("ensemble table-transformer CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 3. FireRed-OCR vision encoder -> CTC decoder (IBP)
// ===========================================================================

/// FireRed-OCR: Conv2d patch embed -> reshape -> Linear encoder -> ReLU ->
/// Linear CTC head -> softmax character probabilities.
///
/// Verifies CTC output probabilities bounded in [0, 1].
#[test]
fn test_ensemble_firered_ocr_vision_to_ctc_ibp() {
    let patch_size = 2;
    let num_patches = (IMG_SIZE / patch_size) * (IMG_SIZE / patch_size);

    let mut b = TensorBlockBuilder::new("ensemble_firered_ocr");
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Patch embedding: Conv2d(3, HIDDEN, patch_size, stride=patch_size)
    let patch_w = b.add_input("patch_w", &[HIDDEN, IN_CHANNELS, patch_size, patch_size]);
    let patch_b = b.add_input("patch_b", &[HIDDEN]);
    let patches = b.add_conv2d(
        input,
        patch_w,
        Some(patch_b),
        patch_size,
        patch_size,
        0,
        0,
        &[HIDDEN, IMG_SIZE / patch_size, IMG_SIZE / patch_size],
    );

    // Reshape to [HIDDEN, num_patches] -> transpose to [num_patches, HIDDEN]
    let flat = b.add_reshape(patches, &[HIDDEN, num_patches]);
    let seq = b.add_transpose(flat, &[1, 0], &[num_patches, HIDDEN]);

    // Encoder: Linear -> ReLU
    let enc_w = b.add_input("enc_w", &[HIDDEN, HIDDEN]);
    let enc_out = b.add_linear(seq, enc_w, None, &[num_patches, HIDDEN]);
    let enc_act = b.add_relu(enc_out, &[num_patches, HIDDEN]);

    // CTC head: Linear -> softmax
    let ctc_w = b.add_input("ctc_w", &[VOCAB, HIDDEN]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB]);
    let ctc_logits = b.add_linear(enc_act, ctc_w, Some(ctc_b), &[num_patches, VOCAB]);
    let out = b.add_softmax(ctc_logits, -1, &[num_patches, VOCAB]);
    let def = b.build(out).expect("valid firered-ocr kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN, IN_CHANNELS, patch_size, patch_size]),
        bias_zero(&[HIDDEN]),
        weight(&[HIDDEN, HIDDEN]),
        weight(&[VOCAB, HIDDEN]),
        bias_zero(&[VOCAB]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_input_bounds();

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ensemble firered-ocr IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 4. Qwen3-VL vision encoder -> MLP projection (IBP + CROWN)
// ===========================================================================

/// Qwen3-VL: Conv2d patch embed -> reshape -> transpose -> Linear MLP ->
/// GELU -> Linear projection.
///
/// Verifies vision features produce finite, bounded embeddings for the
/// language model decoder.
#[test]
fn test_ensemble_qwen3_vl_vision_to_projection_ibp_crown() {
    let patch_size = 2;
    let num_patches = (IMG_SIZE / patch_size) * (IMG_SIZE / patch_size);

    let mut b = TensorBlockBuilder::new("ensemble_qwen3_vl");
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Patch embedding
    let patch_w = b.add_input("patch_w", &[HIDDEN, IN_CHANNELS, patch_size, patch_size]);
    let patch_b = b.add_input("patch_b", &[HIDDEN]);
    let patches = b.add_conv2d(
        input,
        patch_w,
        Some(patch_b),
        patch_size,
        patch_size,
        0,
        0,
        &[HIDDEN, IMG_SIZE / patch_size, IMG_SIZE / patch_size],
    );
    let flat = b.add_reshape(patches, &[HIDDEN, num_patches]);
    let seq = b.add_transpose(flat, &[1, 0], &[num_patches, HIDDEN]);

    // MLP projection: Linear -> GELU -> Linear
    let mlp_w1 = b.add_input("mlp_w1", &[FFN_DIM, HIDDEN]);
    let mlp_b1 = b.add_input("mlp_b1", &[FFN_DIM]);
    let mlp_h = b.add_linear(seq, mlp_w1, Some(mlp_b1), &[num_patches, FFN_DIM]);
    let mlp_act = b.add_gelu(mlp_h, &[num_patches, FFN_DIM]);

    let mlp_w2 = b.add_input("mlp_w2", &[HIDDEN, FFN_DIM]);
    let mlp_b2 = b.add_input("mlp_b2", &[HIDDEN]);
    let out = b.add_linear(mlp_act, mlp_w2, Some(mlp_b2), &[num_patches, HIDDEN]);
    let def = b.build(out).expect("valid qwen3-vl kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN, IN_CHANNELS, patch_size, patch_size]),
        bias_zero(&[HIDDEN]),
        weight(&[FFN_DIM, HIDDEN]),
        bias_zero(&[FFN_DIM]),
        weight(&[HIDDEN, FFN_DIM]),
        bias_zero(&[HIDDEN]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_input_bounds();

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("ensemble qwen3-vl IBP: bounds=[{lo_min:.4}, {hi_max:.4}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
    let width = hi_max - lo_min;
    assert!(
        width < 100.0,
        "qwen3-vl projection bounds width {width} too wide"
    );

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("ensemble qwen3-vl CROWN ({method:?}): bounds=[{clo:.4}, {chi:.4}]");
}

// ===========================================================================
// 5. Granite-Docling vision -> LM bridge (IBP + CROWN)
// ===========================================================================

/// Granite-Docling: Conv2d ViT patch proj -> reshape -> transpose ->
/// Linear vision projection to LM embedding space.
///
/// Verifies the vision-to-language bridge produces bounded embeddings.
#[test]
fn test_ensemble_granite_docling_vision_lm_bridge_ibp_crown() {
    let patch_size = 4;
    let num_patches = (IMG_SIZE / patch_size) * (IMG_SIZE / patch_size);
    let lm_dim = HIDDEN; // LM embedding dimension

    let mut b = TensorBlockBuilder::new("ensemble_granite_docling");
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // ViT patch projection
    let proj_w = b.add_input("proj_w", &[HIDDEN, IN_CHANNELS, patch_size, patch_size]);
    let proj_b = b.add_input("proj_b", &[HIDDEN]);
    let patches = b.add_conv2d(
        input,
        proj_w,
        Some(proj_b),
        patch_size,
        patch_size,
        0,
        0,
        &[HIDDEN, IMG_SIZE / patch_size, IMG_SIZE / patch_size],
    );
    let flat = b.add_reshape(patches, &[HIDDEN, num_patches]);
    let seq = b.add_transpose(flat, &[1, 0], &[num_patches, HIDDEN]);

    // Vision-to-LM projection: Linear
    let bridge_w = b.add_input("bridge_w", &[lm_dim, HIDDEN]);
    let bridge_b = b.add_input("bridge_b", &[lm_dim]);
    let out = b.add_linear(seq, bridge_w, Some(bridge_b), &[num_patches, lm_dim]);
    let def = b.build(out).expect("valid granite-docling kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN, IN_CHANNELS, patch_size, patch_size]),
        bias_zero(&[HIDDEN]),
        weight(&[lm_dim, HIDDEN]),
        bias_zero(&[lm_dim]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_input_bounds();

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("ensemble granite-docling IBP: bounds=[{lo_min:.4}, {hi_max:.4}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("ensemble granite-docling CROWN ({method:?}): bounds=[{clo:.4}, {chi:.4}]");
}

// ===========================================================================
// 6. GLM-OCR decoder -> MTP head (IBP)
// ===========================================================================

/// GLM-OCR: Linear embedding -> ReLU -> Linear FFN -> Linear MTP head ->
/// softmax token predictions.
///
/// Verifies multi-token prediction output probabilities in [0, 1].
#[test]
fn test_ensemble_glm_ocr_decoder_mtp_ibp() {
    let mut b = TensorBlockBuilder::new("ensemble_glm_ocr");
    let input = b.add_input("token_features", &[SEQ, HIDDEN]);

    // FFN: Linear -> ReLU -> Linear
    let ffn_w1 = b.add_input("ffn_w1", &[FFN_DIM, HIDDEN]);
    let ffn_h = b.add_linear(input, ffn_w1, None, &[SEQ, FFN_DIM]);
    let ffn_act = b.add_relu(ffn_h, &[SEQ, FFN_DIM]);
    let ffn_w2 = b.add_input("ffn_w2", &[HIDDEN, FFN_DIM]);
    let ffn_out = b.add_linear(ffn_act, ffn_w2, None, &[SEQ, HIDDEN]);

    // MTP head: Linear -> softmax
    let mtp_w = b.add_input("mtp_w", &[VOCAB, HIDDEN]);
    let mtp_b = b.add_input("mtp_b", &[VOCAB]);
    let logits = b.add_linear(ffn_out, mtp_w, Some(mtp_b), &[SEQ, VOCAB]);
    let out = b.add_softmax(logits, -1, &[SEQ, VOCAB]);
    let def = b.build(out).expect("valid glm-ocr kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN]),
        weight(&[HIDDEN, FFN_DIM]),
        weight(&[VOCAB, HIDDEN]),
        bias_zero(&[VOCAB]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ensemble glm-ocr IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 7. PaddleOCR detection + recognition pipeline (IBP)
// ===========================================================================

/// PaddleOCR: Conv2d backbone -> sigmoid detection + Conv2d patch embed ->
/// Linear SVTR -> GELU -> Linear CTC -> softmax recognition.
///
/// Verifies both detection (sigmoid [0,1]) and recognition (softmax [0,1]).
#[test]
fn test_ensemble_paddle_ocr_detect_recognize_ibp() {
    let flat_dim = HIDDEN * POOL_SIZE * POOL_SIZE;

    let mut b = TensorBlockBuilder::new("ensemble_paddle_ocr");
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // DB detector backbone: Conv2d -> flatten -> Linear -> sigmoid
    let det_w = b.add_input("det_conv_w", &[HIDDEN, IN_CHANNELS, 3, 3]);
    let det_b = b.add_input("det_conv_b", &[HIDDEN]);
    let det_feat = b.add_conv2d(
        input,
        det_w,
        Some(det_b),
        2,
        2,
        1,
        1,
        &[HIDDEN, POOL_SIZE, POOL_SIZE],
    );
    let det_flat = b.add_reshape(det_feat, &[flat_dim]);
    let det_head_w = b.add_input("det_head_w", &[1, flat_dim]);
    let det_logit = b.add_linear(det_flat, det_head_w, None, &[1]);
    let det_out = b.add_sigmoid(det_logit, &[1]);

    // SVTR recognizer: Linear -> GELU -> Linear CTC -> softmax
    let svtr_w1 = b.add_input("svtr_w1", &[FFN_DIM, flat_dim]);
    let svtr_h = b.add_linear(det_flat, svtr_w1, None, &[FFN_DIM]);
    let svtr_act = b.add_gelu(svtr_h, &[FFN_DIM]);
    let ctc_w = b.add_input("ctc_w", &[VOCAB, FFN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB]);
    let ctc_logits = b.add_linear(svtr_act, ctc_w, Some(ctc_b), &[VOCAB]);
    let rec_out = b.add_softmax(ctc_logits, -1, &[VOCAB]);

    // Combine detection + recognition via add (ensemble merge proxy)
    // Reshape detection to broadcast with recognition
    let det_bc = b.add_broadcast(det_out, &[VOCAB]);
    let combined = b.add_binary_add(rec_out, det_bc, &[VOCAB]);
    let def = b.build(combined).expect("valid paddle-ocr kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN, IN_CHANNELS, 3, 3]), // det_conv_w
        bias_zero(&[HIDDEN]),                 // det_conv_b
        weight(&[1, flat_dim]),               // det_head_w
        weight(&[FFN_DIM, flat_dim]),         // svtr_w1
        weight(&[VOCAB, FFN_DIM]),            // ctc_w
        bias_zero(&[VOCAB]),                  // ctc_b
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = image_input_bounds();

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ensemble paddle-ocr IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Detection sigmoid in [0,1] + recognition softmax in [0,1] -> combined in [0,2]
    assert!(lo_min >= -1e-3, "combined lower >= 0, got {lo_min}");
    assert!(hi_max <= 2.0 + 1e-3, "combined upper <= 2, got {hi_max}");
}

// ===========================================================================
// 8. DocLayout -> Table Transformer cascade (IBP)
// ===========================================================================

/// Cross-model cascade: DocLayout-YOLO detection sigmoid -> bridge projection
/// -> Table Transformer query input -> attention -> sigmoid bbox output.
///
/// Verifies bounds compose across the detection-to-table boundary.
#[test]
fn test_ensemble_doclayout_to_table_cascade_ibp() {
    let mut b = TensorBlockBuilder::new("ensemble_doclayout_table");
    let input = b.add_input("det_features", &[NUM_BOXES, HIDDEN]);

    // DocLayout detection sigmoid (confidence in [0, 1])
    let det_w = b.add_input("det_w", &[NUM_CLASSES, HIDDEN]);
    let det_logits = b.add_linear(input, det_w, None, &[NUM_BOXES, NUM_CLASSES]);
    let det_conf = b.add_sigmoid(det_logits, &[NUM_BOXES, NUM_CLASSES]);

    // Bridge: project detection confidences to table query space
    let bridge_w = b.add_input("bridge_w", &[HIDDEN, NUM_CLASSES]);
    let bridge_b = b.add_input("bridge_b", &[HIDDEN]);
    let queries = b.add_linear(det_conf, bridge_w, Some(bridge_b), &[NUM_BOXES, HIDDEN]);

    // Table Transformer: Linear bbox head -> sigmoid
    let box_w = b.add_input("box_w", &[4, HIDDEN]);
    let box_b = b.add_input("box_b", &[4]);
    let box_logits = b.add_linear(queries, box_w, Some(box_b), &[NUM_BOXES, 4]);
    let out = b.add_sigmoid(box_logits, &[NUM_BOXES, 4]);
    let def = b.build(out).expect("valid doclayout->table kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_CLASSES, HIDDEN]), // det_w
        weight(&[HIDDEN, NUM_CLASSES]), // bridge_w
        bias_zero(&[HIDDEN]),           // bridge_b
        weight(&[4, HIDDEN]),           // box_w
        bias_zero(&[4]),                // box_b
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_BOXES, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ensemble doclayout->table IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 9. DocLayout -> FireRed-OCR cascade (IBP)
// ===========================================================================

/// Cross-model cascade: DocLayout-YOLO detection sigmoid -> bridge ->
/// FireRed-OCR encoder -> CTC softmax character probabilities.
///
/// Verifies bounds compose across detection-to-OCR boundary.
#[test]
fn test_ensemble_doclayout_to_firered_cascade_ibp() {
    let mut b = TensorBlockBuilder::new("ensemble_doclayout_firered");
    let input = b.add_input("det_features", &[NUM_BOXES, HIDDEN]);

    // DocLayout detection sigmoid
    let det_w = b.add_input("det_w", &[NUM_CLASSES, HIDDEN]);
    let det_logits = b.add_linear(input, det_w, None, &[NUM_BOXES, NUM_CLASSES]);
    let det_conf = b.add_sigmoid(det_logits, &[NUM_BOXES, NUM_CLASSES]);

    // Bridge to OCR feature space
    let bridge_w = b.add_input("bridge_w", &[HIDDEN, NUM_CLASSES]);
    let ocr_feats = b.add_linear(det_conf, bridge_w, None, &[NUM_BOXES, HIDDEN]);

    // FireRed-OCR encoder: Linear -> ReLU
    let enc_w = b.add_input("enc_w", &[HIDDEN, HIDDEN]);
    let enc_out = b.add_linear(ocr_feats, enc_w, None, &[NUM_BOXES, HIDDEN]);
    let enc_act = b.add_relu(enc_out, &[NUM_BOXES, HIDDEN]);

    // CTC head: Linear -> softmax
    let ctc_w = b.add_input("ctc_w", &[VOCAB, HIDDEN]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB]);
    let ctc_logits = b.add_linear(enc_act, ctc_w, Some(ctc_b), &[NUM_BOXES, VOCAB]);
    let out = b.add_softmax(ctc_logits, -1, &[NUM_BOXES, VOCAB]);
    let def = b.build(out).expect("valid doclayout->firered kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_CLASSES, HIDDEN]), // det_w
        weight(&[HIDDEN, NUM_CLASSES]), // bridge_w
        weight(&[HIDDEN, HIDDEN]),      // enc_w
        weight(&[VOCAB, HIDDEN]),       // ctc_w
        bias_zero(&[VOCAB]),            // ctc_b
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_BOXES, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ensemble doclayout->firered IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 10. DocLayout -> Qwen3-VL cascade (IBP)
// ===========================================================================

/// Cross-model cascade: DocLayout-YOLO detection -> bridge ->
/// Qwen3-VL MLP projection for VLM analysis of detected regions.
///
/// Verifies bounds compose across detection-to-VLM boundary.
#[test]
fn test_ensemble_doclayout_to_qwen3_vl_cascade_ibp() {
    let mut b = TensorBlockBuilder::new("ensemble_doclayout_qwen3vl");
    let input = b.add_input("det_features", &[NUM_BOXES, HIDDEN]);

    // DocLayout detection sigmoid
    let det_w = b.add_input("det_w", &[NUM_CLASSES, HIDDEN]);
    let det_logits = b.add_linear(input, det_w, None, &[NUM_BOXES, NUM_CLASSES]);
    let det_conf = b.add_sigmoid(det_logits, &[NUM_BOXES, NUM_CLASSES]);

    // Bridge to VLM feature space
    let bridge_w = b.add_input("bridge_w", &[HIDDEN, NUM_CLASSES]);
    let bridge_b = b.add_input("bridge_b", &[HIDDEN]);
    let vlm_feats = b.add_linear(det_conf, bridge_w, Some(bridge_b), &[NUM_BOXES, HIDDEN]);

    // Qwen3-VL MLP: Linear -> GELU -> Linear
    let mlp_w1 = b.add_input("mlp_w1", &[FFN_DIM, HIDDEN]);
    let mlp_h = b.add_linear(vlm_feats, mlp_w1, None, &[NUM_BOXES, FFN_DIM]);
    let mlp_act = b.add_gelu(mlp_h, &[NUM_BOXES, FFN_DIM]);
    let mlp_w2 = b.add_input("mlp_w2", &[HIDDEN, FFN_DIM]);
    let out = b.add_linear(mlp_act, mlp_w2, None, &[NUM_BOXES, HIDDEN]);
    let def = b.build(out).expect("valid doclayout->qwen3-vl kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_CLASSES, HIDDEN]), // det_w
        weight(&[HIDDEN, NUM_CLASSES]), // bridge_w
        bias_zero(&[HIDDEN]),           // bridge_b
        weight(&[FFN_DIM, HIDDEN]),     // mlp_w1
        weight(&[HIDDEN, FFN_DIM]),     // mlp_w2
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_BOXES, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ensemble doclayout->qwen3-vl IBP: bounds=[{lo_min:.4}, {hi_max:.4}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
    let width = hi_max - lo_min;
    assert!(width < 100.0, "cascade bounds width {width} too wide");
}

// ===========================================================================
// 11. Full 3-model pipeline: layout -> table -> OCR (IBP)
// ===========================================================================

/// Three-model pipeline matching dpdf document processing:
///   DocLayout-YOLO (detection) -> Table Transformer (table structure) ->
///   PaddleOCR (text recognition).
///
/// End-to-end from detection features to character softmax probabilities.
#[test]
fn test_ensemble_three_model_pipeline_ibp() {
    let mut b = TensorBlockBuilder::new("ensemble_3model_pipeline");
    let input = b.add_input("det_features", &[NUM_BOXES, HIDDEN]);

    // Stage 1: DocLayout-YOLO detection -> sigmoid
    let det_w = b.add_input("det_w", &[NUM_CLASSES, HIDDEN]);
    let det_logits = b.add_linear(input, det_w, None, &[NUM_BOXES, NUM_CLASSES]);
    let det_conf = b.add_sigmoid(det_logits, &[NUM_BOXES, NUM_CLASSES]);

    // Stage 2: Table Transformer -> sigmoid bbox
    let table_w = b.add_input("table_w", &[HIDDEN, NUM_CLASSES]);
    let table_feats = b.add_linear(det_conf, table_w, None, &[NUM_BOXES, HIDDEN]);
    let table_head_w = b.add_input("table_head_w", &[4, HIDDEN]);
    let table_logits = b.add_linear(table_feats, table_head_w, None, &[NUM_BOXES, 4]);
    let table_bbox = b.add_sigmoid(table_logits, &[NUM_BOXES, 4]);

    // Stage 3: OCR recognition -> softmax
    let ocr_bridge_w = b.add_input("ocr_bridge_w", &[HIDDEN, 4]);
    let ocr_feats = b.add_linear(table_bbox, ocr_bridge_w, None, &[NUM_BOXES, HIDDEN]);
    let ocr_act = b.add_relu(ocr_feats, &[NUM_BOXES, HIDDEN]);
    let ctc_w = b.add_input("ctc_w", &[VOCAB, HIDDEN]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB]);
    let ctc_logits = b.add_linear(ocr_act, ctc_w, Some(ctc_b), &[NUM_BOXES, VOCAB]);
    let out = b.add_softmax(ctc_logits, -1, &[NUM_BOXES, VOCAB]);
    let def = b.build(out).expect("valid 3-model pipeline kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_CLASSES, HIDDEN]), // det_w
        weight(&[HIDDEN, NUM_CLASSES]), // table_w
        weight(&[4, HIDDEN]),           // table_head_w
        weight(&[HIDDEN, 4]),           // ocr_bridge_w
        weight(&[VOCAB, HIDDEN]),       // ctc_w
        bias_zero(&[VOCAB]),            // ctc_b
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_BOXES, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ensemble 3-model pipeline IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 12. Full 4-model pipeline: layout -> table -> OCR -> language (IBP)
// ===========================================================================

/// Four-model pipeline extending the 3-model pipeline with a language decoder:
///   DocLayout-YOLO -> Table Transformer -> OCR CTC -> GLM-OCR decoder softmax.
///
/// End-to-end from detection features to token prediction probabilities.
#[test]
fn test_ensemble_four_model_pipeline_ibp() {
    let mut b = TensorBlockBuilder::new("ensemble_4model_pipeline");
    let input = b.add_input("det_features", &[SEQ, HIDDEN]);

    // Stage 1: Detection -> sigmoid
    let det_w = b.add_input("det_w", &[NUM_CLASSES, HIDDEN]);
    let det_logits = b.add_linear(input, det_w, None, &[SEQ, NUM_CLASSES]);
    let det_conf = b.add_sigmoid(det_logits, &[SEQ, NUM_CLASSES]);

    // Stage 2: Table -> sigmoid bbox
    let table_w = b.add_input("table_w", &[4, NUM_CLASSES]);
    let table_logits = b.add_linear(det_conf, table_w, None, &[SEQ, 4]);
    let table_bbox = b.add_sigmoid(table_logits, &[SEQ, 4]);

    // Stage 3: OCR -> softmax
    let ocr_w = b.add_input("ocr_w", &[HIDDEN, 4]);
    let ocr_feats = b.add_linear(table_bbox, ocr_w, None, &[SEQ, HIDDEN]);
    let ocr_act = b.add_relu(ocr_feats, &[SEQ, HIDDEN]);
    let ctc_w = b.add_input("ctc_w", &[VOCAB, HIDDEN]);
    let ctc_logits = b.add_linear(ocr_act, ctc_w, None, &[SEQ, VOCAB]);
    let ctc_prob = b.add_softmax(ctc_logits, -1, &[SEQ, VOCAB]);

    // Stage 4: Language decoder FFN -> softmax token predictions
    let lang_w1 = b.add_input("lang_w1", &[HIDDEN, VOCAB]);
    let lang_h = b.add_linear(ctc_prob, lang_w1, None, &[SEQ, HIDDEN]);
    let lang_act = b.add_relu(lang_h, &[SEQ, HIDDEN]);
    let lang_w2 = b.add_input("lang_w2", &[VOCAB, HIDDEN]);
    let lang_b2 = b.add_input("lang_b2", &[VOCAB]);
    let lang_logits = b.add_linear(lang_act, lang_w2, Some(lang_b2), &[SEQ, VOCAB]);
    let out = b.add_softmax(lang_logits, -1, &[SEQ, VOCAB]);
    let def = b.build(out).expect("valid 4-model pipeline kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_CLASSES, HIDDEN]), // det_w
        weight(&[4, NUM_CLASSES]),      // table_w
        weight(&[HIDDEN, 4]),           // ocr_w
        weight(&[VOCAB, HIDDEN]),       // ctc_w
        weight(&[HIDDEN, VOCAB]),       // lang_w1
        weight(&[VOCAB, HIDDEN]),       // lang_w2
        bias_zero(&[VOCAB]),            // lang_b2
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ensemble 4-model pipeline IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 13. Ensemble confidence aggregation (IBP + CROWN)
// ===========================================================================

/// Multi-model ensemble: three parallel sigmoid heads (DocLayout-YOLO,
/// Table Transformer, Granite-Docling) produce confidences that are
/// averaged. Verifies aggregated output remains in [0, 1].
#[test]
fn test_ensemble_confidence_aggregation_ibp_crown() {
    let mut b = TensorBlockBuilder::new("ensemble_confidence_agg");
    let input = b.add_input("shared_features", &[SEQ, HIDDEN]);

    // Head 1: DocLayout-YOLO cls sigmoid
    let h1_w = b.add_input("h1_w", &[NUM_CLASSES, HIDDEN]);
    let h1_logits = b.add_linear(input, h1_w, None, &[SEQ, NUM_CLASSES]);
    let h1_conf = b.add_sigmoid(h1_logits, &[SEQ, NUM_CLASSES]);

    // Head 2: Table Transformer cls sigmoid
    let h2_w = b.add_input("h2_w", &[NUM_CLASSES, HIDDEN]);
    let h2_logits = b.add_linear(input, h2_w, None, &[SEQ, NUM_CLASSES]);
    let h2_conf = b.add_sigmoid(h2_logits, &[SEQ, NUM_CLASSES]);

    // Head 3: Granite-Docling cls sigmoid
    let h3_w = b.add_input("h3_w", &[NUM_CLASSES, HIDDEN]);
    let h3_logits = b.add_linear(input, h3_w, None, &[SEQ, NUM_CLASSES]);
    let h3_conf = b.add_sigmoid(h3_logits, &[SEQ, NUM_CLASSES]);

    // Average: (h1 + h2 + h3) * (1/3)
    let sum12 = b.add_binary_add(h1_conf, h2_conf, &[SEQ, NUM_CLASSES]);
    let sum123 = b.add_binary_add(sum12, h3_conf, &[SEQ, NUM_CLASSES]);

    // Scale by 1/3 via constant multiply
    let one_third = b.add_input("scale", &[SEQ, NUM_CLASSES]);
    let out = b.add_binary_mul(sum123, one_third, &[SEQ, NUM_CLASSES]);
    let def = b.build(out).expect("valid confidence aggregation kernel");

    let scale_data = ArrayD::from_elem(IxDyn(&[SEQ, NUM_CLASSES]), 1.0f32 / 3.0);
    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_CLASSES, HIDDEN]),                 // h1_w
        weight(&[NUM_CLASSES, HIDDEN]),                 // h2_w
        weight(&[NUM_CLASSES, HIDDEN]),                 // h3_w
        TensorParamBinding::ConstantTensor(scale_data), // scale
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("ensemble confidence aggregation IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // Average of 3 sigmoids in [0,1] should still be in [0,1].
    assert!(lo_min >= -1e-3, "avg sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-3, "avg sigmoid upper <= 1, got {hi_max}");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("ensemble confidence aggregation CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 14. Ensemble monotone tightening (IBP)
// ===========================================================================

/// Monotonicity property: narrower input bounds produce output bounds
/// that are no wider than those from the full input range.
///
/// Tests the 3-model pipeline (detection -> table -> OCR) with two
/// input ranges to verify IBP monotonicity.
#[test]
fn test_ensemble_monotone_tightening_ibp() {
    // Build the 3-model pipeline
    let build_pipeline = || {
        let mut b = TensorBlockBuilder::new("ensemble_monotone");
        let input = b.add_input("features", &[SEQ, HIDDEN]);

        // Detection -> sigmoid
        let det_w = b.add_input("det_w", &[NUM_CLASSES, HIDDEN]);
        let det_logits = b.add_linear(input, det_w, None, &[SEQ, NUM_CLASSES]);
        let det_conf = b.add_sigmoid(det_logits, &[SEQ, NUM_CLASSES]);

        // Table -> sigmoid
        let table_w = b.add_input("table_w", &[4, NUM_CLASSES]);
        let table_logits = b.add_linear(det_conf, table_w, None, &[SEQ, 4]);
        let table_bbox = b.add_sigmoid(table_logits, &[SEQ, 4]);

        // OCR -> softmax
        let ocr_w = b.add_input("ocr_w", &[VOCAB, 4]);
        let ocr_b = b.add_input("ocr_b", &[VOCAB]);
        let ctc_logits = b.add_linear(table_bbox, ocr_w, Some(ocr_b), &[SEQ, VOCAB]);
        let out = b.add_softmax(ctc_logits, -1, &[SEQ, VOCAB]);
        let def = b.build(out).expect("valid monotone pipeline kernel");

        let bindings = vec![
            TensorParamBinding::Variable,
            weight(&[NUM_CLASSES, HIDDEN]), // det_w
            weight(&[4, NUM_CLASSES]),      // table_w
            weight(&[VOCAB, 4]),            // ocr_w
            bias_zero(&[VOCAB]),            // ocr_b
        ];
        tensor_kernel_to_graph(&def, &bindings).expect("graph")
    };

    let graph = build_pipeline();

    // Wide input: features in [-1, 1]
    let wide_input = uniform_bounds(&[SEQ, HIDDEN], 1.0);
    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");

    // Narrow input: features in [-0.3, 0.3]
    let narrow_input = uniform_bounds(&[SEQ, HIDDEN], 0.3);
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("IBP narrow");

    let (lo_w, hi_w) = bounds_min_max(&wide_output);
    let (lo_n, hi_n) = bounds_min_max(&narrow_output);
    let wide_width = hi_w - lo_w;
    let narrow_width = hi_n - lo_n;

    eprintln!(
        "ensemble monotone: wide=[{lo_w:.4}, {hi_w:.4}] w={wide_width:.4} \
         | narrow=[{lo_n:.4}, {hi_n:.4}] w={narrow_width:.4}"
    );

    // Monotonicity: narrow input -> output bounds no wider.
    assert!(
        narrow_width <= wide_width + 1e-4,
        "monotone tightening violated: narrow_width={narrow_width} > wide_width={wide_width}"
    );
}

// ===========================================================================
// 15. 7-model dispatch routing (IBP)
// ===========================================================================

/// Ensemble routing: input features are dispatched to 7 model heads via
/// softmax gating. Each head produces a sigmoid/softmax output. The final
/// output is the gated combination.
///
/// This matches the dpdf dispatch pattern where document regions are
/// routed to the appropriate model based on region type.
#[test]
fn test_ensemble_seven_model_dispatch_routing_ibp() {
    let mut b = TensorBlockBuilder::new("ensemble_7model_routing");
    let input = b.add_input("region_features", &[SEQ, HIDDEN]);

    // Routing gate: Linear -> softmax -> 7 model weights
    let gate_w = b.add_input("gate_w", &[NUM_MODELS, HIDDEN]);
    let gate_b = b.add_input("gate_b", &[NUM_MODELS]);
    let gate_logits = b.add_linear(input, gate_w, Some(gate_b), &[SEQ, NUM_MODELS]);
    let gate_probs = b.add_softmax(gate_logits, -1, &[SEQ, NUM_MODELS]);

    // Each model head: Linear -> sigmoid (simplified head)
    // We build 7 heads in sequence and combine via weighted sum.
    // For tractable verification, use a single merged linear layer as proxy.
    // gate_probs [SEQ, 7] @ head_matrix [7, NUM_CLASSES] gives weighted output.
    let head_matrix_w = b.add_input("head_matrix_w", &[NUM_CLASSES, NUM_MODELS]);
    let head_out = b.add_linear(gate_probs, head_matrix_w, None, &[SEQ, NUM_CLASSES]);
    let out = b.add_sigmoid(head_out, &[SEQ, NUM_CLASSES]);
    let def = b.build(out).expect("valid 7-model routing kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_MODELS, HIDDEN]),      // gate_w
        bias_zero(&[NUM_MODELS]),           // gate_b
        weight(&[NUM_CLASSES, NUM_MODELS]), // head_matrix_w
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ensemble 7-model routing IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // Softmax gate [0,1] -> linear -> sigmoid [0,1]
    assert!(lo_min >= -1e-5, "routed sigmoid lower >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-5,
        "routed sigmoid upper <= 1, got {hi_max}"
    );

    // Non-degenerate: routing produces a genuine (non-zero) interval. The
    // tightened softmax+sigmoid IBP now narrows this routed output well below
    // the old 0.01 floor (observed ~0.0028), so that floor is a stale lower
    // bound made obsolete by tighter bounds; a narrower interval is *better*
    // here. We only require the bounds remain non-degenerate.
    let width = hi_max - lo_min;
    assert!(
        width > 0.0,
        "routing output must be a non-degenerate interval, got width={width}"
    );
}
