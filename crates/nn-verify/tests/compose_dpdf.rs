// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests for the DocLayout-YOLO detection pipeline.
//!
//! Verifies IBP and CROWN bound propagation through the pipeline stages
//! corresponding to [`DpdfPipelineMetal`] and [`DpdfImagePreprocessMetal`]:
//!
//! ## Tests (8 tests)
//!
//! 1.  **ImageNet normalization bounds** — `(pixel - mean) / std` per channel (IBP)
//! 2.  **Letterbox padding preserves bounds** — Zero-fill pad + identity (IBP)
//! 3.  **Full preprocessing pipeline composition** — Normalize + resize proxy (IBP + CROWN)
//! 4.  **Detection box coordinates in [0, 1]** — Sigmoid box head (IBP)
//! 5.  **Confidence scores in [0, 1]** — Sigmoid classification head (IBP)
//! 6.  **NMS score filtering output non-negative** — ReLU(sigmoid - threshold) (IBP)
//! 7.  **End-to-end preprocess -> detect -> sigmoid pipeline** (IBP)
//! 8.  **Monotone tightening: tighter pixel range -> tighter detections** (IBP)
//!
//! Architecture: Image -> preprocess_image (normalize, resize, pad) ->
//!   DocLayoutYolo::forward (backbone -> neck -> head) -> NMS (CPU).
//!
//! Dimensions (small for fast verification, structurally representative):
//! - IMG_C=3, IMG_H=8, IMG_W=8 (symbolic, real: 640x640)
//! - BASE_CH=4 (symbolic, real: 64), NUM_CLASSES=3 (symbolic, real: 11)
//! - NUM_BOXES=4 (symbolic, real: 8400)
//!
//! Part of #4186: Compose tests for DocLayout-YOLO detection pipeline bounds.

mod common;

use common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorNodeId;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- symbolic for fast verification
// ---------------------------------------------------------------------------

const IMG_C: usize = 3;
const IMG_H: usize = 8;
const IMG_W: usize = 8;
const BASE_CH: usize = 4;
const NUM_CLASSES: usize = 3;
const NUM_BOXES: usize = 4;
const HEAD_DIM: usize = 4;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

/// ImageNet channel means (RGB).
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// ImageNet channel standard deviations (RGB).
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

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

/// Ones tensor binding (for BatchNorm weight / variance).
fn ones(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 1.0f32))
}

/// Scalar epsilon binding.
fn eps_binding() -> TensorParamBinding {
    TensorParamBinding::ConstantScalar(1e-5)
}

/// Image-domain input bounds: pixels in [0, 1].
fn image_input_bounds() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[IMG_C, IMG_H, IMG_W]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[IMG_C, IMG_H, IMG_W]), 1.0f32),
    )
    .expect("valid image bounds")
}

/// Add SiLU activation: sigmoid(x) * x.
fn add_silu(b: &mut TensorBlockBuilder, input: TensorNodeId, shape: &[usize]) -> TensorNodeId {
    let sig = b.add_sigmoid(input, shape);
    b.add_binary_mul(input, sig, shape)
}

/// Add ConvBnAct block: Conv2d -> BatchNorm -> SiLU.
fn add_conv_bn_act(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    out_h: usize,
    out_w: usize,
    prefix: &str,
) -> TensorNodeId {
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
    add_silu(b, bn_out, &out_shape)
}

/// Push bindings for one ConvBnAct block (7 params).
fn push_conv_bn_act_bindings(
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
// 1. ImageNet normalization bounds (IBP)
// ===========================================================================

/// Verifies that ImageNet normalization `(pixel - mean) / std` produces
/// correct output bounds when input pixels are in [0, 1].
///
/// This matches the normalization in `DpdfPipelineMetal::preprocess_image`
/// and `DpdfImagePreprocessMetal::gpu_normalize`.
///
/// Expected output range per channel:
///   - R: [(0 - 0.485) / 0.229, (1 - 0.485) / 0.229] = [-2.118, 2.248]
///   - G: [(0 - 0.456) / 0.224, (1 - 0.456) / 0.224] = [-2.036, 2.429]
///   - B: [(0 - 0.406) / 0.225, (1 - 0.406) / 0.225] = [-1.804, 2.640]
#[test]
fn test_dpdf_imagenet_normalization_ibp() {
    let inv_std: Vec<f32> = IMAGENET_STD.iter().map(|s| 1.0 / s).collect();
    let neg_mean_div_std: Vec<f32> = IMAGENET_MEAN
        .iter()
        .zip(IMAGENET_STD.iter())
        .map(|(m, s)| -m / s)
        .collect();

    let mut b = TensorBlockBuilder::new("dpdf_imagenet_norm");
    let input = b.add_input("image", &[IMG_C, IMG_H, IMG_W]);
    let scale = b.add_input("inv_std", &[IMG_C]);
    let bias = b.add_input("neg_mean_div_std", &[IMG_C]);

    // Broadcast [C] -> [C, H, W], then element-wise multiply and add.
    let scale_bc = b.add_broadcast_left(scale, &[IMG_C, IMG_H, IMG_W]);
    let scaled = b.add_binary_mul(input, scale_bc, &[IMG_C, IMG_H, IMG_W]);
    let bias_bc = b.add_broadcast_left(bias, &[IMG_C, IMG_H, IMG_W]);
    let out = b.add_binary_add(scaled, bias_bc, &[IMG_C, IMG_H, IMG_W]);
    let def = b.build(out).expect("valid ImageNet normalization kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[IMG_C]), inv_std).expect("inv_std"),
        ),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[IMG_C]), neg_mean_div_std).expect("neg_mean_div_std"),
        ),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_input_bounds();

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through ImageNet normalize");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[IMG_C, IMG_H, IMG_W]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf ImageNet normalize IBP: bounds=[{lo_min:.4}, {hi_max:.4}]");

    // Normalized output must span negative (from 0-mean) and positive (from 1-mean).
    assert!(
        lo_min < -1.5,
        "ImageNet normalized lower should be < -1.5, got {lo_min}"
    );
    assert!(
        hi_max > 2.0,
        "ImageNet normalized upper should be > 2.0, got {hi_max}"
    );
    // But must be bounded (not vacuously wide).
    assert!(
        hi_max - lo_min < 10.0,
        "ImageNet normalize output width should be < 10, got {}",
        hi_max - lo_min
    );
}

// ===========================================================================
// 2. Letterbox padding preserves bounds (IBP)
// ===========================================================================

/// Verifies that letterbox padding (identity for image area, constant fill
/// for pad area) preserves pixel bounds.
///
/// This matches `DpdfImagePreprocessMetal::gpu_letterbox_pad`.
/// The pad fill value is 0.5 * scale_factor (typically 0.5/255 after scaling,
/// but for already-normalized inputs, fill is a fixed constant).
#[test]
fn test_dpdf_letterbox_padding_ibp() {
    // Model letterbox as: linear projection where image pixels map to
    // themselves and padding pixels get a constant fill value.
    let in_h = 6;
    let in_w = 8;
    let in_flat = IMG_C * in_h * in_w;
    let out_flat = IMG_C * IMG_H * IMG_W;
    let fill_value = 0.5f32;

    let mut b = TensorBlockBuilder::new("dpdf_letterbox_pad");
    let input = b.add_input("image", &[in_flat]);
    let pad_w = b.add_input("letterbox_proj", &[out_flat, in_flat]);
    let pad_b = b.add_input("letterbox_bias", &[out_flat]);
    let out = b.add_linear(input, pad_w, Some(pad_b), &[out_flat]);
    let def = b.build(out).expect("valid letterbox kernel");

    // Identity projection for image pixels, fill bias for pad pixels.
    let mut proj_data = vec![0.0f32; out_flat * in_flat];
    let mut bias_data = vec![fill_value; out_flat];
    for i in 0..in_flat.min(out_flat) {
        proj_data[i * in_flat + i] = 1.0;
        bias_data[i] = 0.0; // No fill bias for identity-mapped pixels.
    }
    let proj_tensor =
        ArrayD::from_shape_vec(IxDyn(&[out_flat, in_flat]), proj_data).expect("valid proj");
    let bias_tensor = ArrayD::from_shape_vec(IxDyn(&[out_flat]), bias_data).expect("valid bias");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(proj_tensor),
        TensorParamBinding::ConstantTensor(bias_tensor),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[in_flat], 0.5); // pixels in [-0.5, +0.5]

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through letterbox padding");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf letterbox padding IBP: bounds=[{lo_min:.4}, {hi_max:.4}]");

    // Image pixels: [-0.5, +0.5] (uniform_bounds returns [-range, +range]).
    // The single linear node has no ReLU/clamp: identity rows pass the input
    // range straight through, so the IBP lower bound = input lower bound = -0.5
    // (the -0.50000006 seen in practice is f32 matmul rounding of 1.0 * -0.5),
    // and the identity-mapped upper = +0.5. Pad pixels are exactly fill_value
    // = 0.5, which lies inside [-0.5, +0.5]. Both bounds are reachable and tight.
    assert!(lo_min.is_finite() && hi_max.is_finite());
    assert!(
        lo_min >= -0.5 - 1e-4,
        "letterbox lower should be >= -0.5 (input lower bound), got {lo_min}"
    );
    assert!(
        hi_max <= 0.5 + 1e-4,
        "letterbox upper should be <= 0.5, got {hi_max}"
    );
}

// ===========================================================================
// 3. Full preprocessing pipeline composition (IBP + CROWN)
// ===========================================================================

/// Full image preprocessing pipeline matching `DpdfImagePreprocessMetal`:
///   Input [C, H, W] -> ImageNet normalize -> Conv2d proxy (resize) ->
///   [D, H', W'] -> flatten -> output.
///
/// Verifies that the composed preprocessing produces finite, bounded outputs.
#[test]
fn test_dpdf_full_preprocess_pipeline_ibp_crown() {
    let out_h = IMG_H / 2;
    let out_w = IMG_W / 2;
    let proj_dim = BASE_CH;

    let mut b = TensorBlockBuilder::new("dpdf_preprocess_pipeline");
    let input = b.add_input("image", &[IMG_C, IMG_H, IMG_W]);

    // Step 1: ImageNet normalization (affine per-channel).
    let inv_std: Vec<f32> = IMAGENET_STD.iter().map(|s| 1.0 / s).collect();
    let neg_mean_div_std: Vec<f32> = IMAGENET_MEAN
        .iter()
        .zip(IMAGENET_STD.iter())
        .map(|(m, s)| -m / s)
        .collect();

    let norm_scale = b.add_input("norm_scale", &[IMG_C]);
    let norm_bias = b.add_input("norm_bias", &[IMG_C]);
    let norm_scale_bc = b.add_broadcast_left(norm_scale, &[IMG_C, IMG_H, IMG_W]);
    let norm_bias_bc = b.add_broadcast_left(norm_bias, &[IMG_C, IMG_H, IMG_W]);
    let normed = b.add_binary_mul(input, norm_scale_bc, &[IMG_C, IMG_H, IMG_W]);
    let normed = b.add_binary_add(normed, norm_bias_bc, &[IMG_C, IMG_H, IMG_W]);

    // Step 2: Conv2d as resize proxy (stride-2 downsampling).
    let conv_w = b.add_input("conv_w", &[proj_dim, IMG_C, 3, 3]);
    let conv_b = b.add_input("conv_b", &[proj_dim]);
    let conv_out = b.add_conv2d(
        normed,
        conv_w,
        Some(conv_b),
        2,
        2,
        1,
        1,
        &[proj_dim, out_h, out_w],
    );

    // Step 3: Flatten for downstream consumption.
    let flat_dim = proj_dim * out_h * out_w;
    let out = b.add_reshape(conv_out, &[flat_dim]);
    let def = b.build(out).expect("valid preprocess pipeline kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[IMG_C]), inv_std).expect("inv_std"),
        ),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[IMG_C]), neg_mean_div_std).expect("neg_mean_div_std"),
        ),
        weight(&[proj_dim, IMG_C, 3, 3]),
        bias_zero(&[proj_dim]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_input_bounds();

    // IBP
    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through preprocess pipeline");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("dpdf preprocess pipeline IBP: bounds=[{lo_min:.4}, {hi_max:.4}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("dpdf preprocess pipeline CROWN ({method:?}): bounds=[{clo:.4}, {chi:.4}]");
}

// ===========================================================================
// 4. Detection box coordinates in [0, 1] (IBP)
// ===========================================================================

/// Verifies that detection box coordinate outputs are bounded in [0, 1]
/// via sigmoid activation, matching the YOLO detection head.
///
/// Pipeline: features -> Linear -> sigmoid -> box coordinates.
/// Sigmoid guarantees output in (0, 1) for any finite input.
#[test]
fn test_dpdf_detection_box_coordinates_bounded_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_box_coords");
    let input = b.add_input("features", &[NUM_BOXES, BASE_CH]);

    // Box regression head: Linear -> sigmoid for normalized [0, 1] coordinates.
    let box_w = b.add_input("box_weight", &[HEAD_DIM, BASE_CH]);
    let box_b = b.add_input("box_bias", &[HEAD_DIM]);
    let logits = b.add_linear(input, box_w, Some(box_b), &[NUM_BOXES, HEAD_DIM]);
    let out = b.add_sigmoid(logits, &[NUM_BOXES, HEAD_DIM]);
    let def = b.build(out).expect("valid box coords kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HEAD_DIM, BASE_CH]),
        bias_zero(&[HEAD_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_BOXES, BASE_CH], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through box coords");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[NUM_BOXES, HEAD_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf box coordinates IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // Sigmoid output must be in [0, 1].
    assert!(
        lo_min >= -1e-5,
        "box coordinate lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-5,
        "box coordinate upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 5. Confidence scores in [0, 1] (IBP)
// ===========================================================================

/// Verifies that detection confidence scores are bounded in [0, 1] via
/// sigmoid activation on classification logits.
///
/// This property is critical for NMS: confidence scores outside [0, 1]
/// would make threshold filtering semantically invalid.
#[test]
fn test_dpdf_confidence_scores_bounded_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_confidence");
    let input = b.add_input("cls_logits", &[NUM_BOXES, NUM_CLASSES]);

    // Classification head: sigmoid -> per-class confidence in [0, 1].
    let out = b.add_sigmoid(input, &[NUM_BOXES, NUM_CLASSES]);
    let def = b.build(out).expect("valid confidence kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Test with wide logit range [-10, 10] to stress sigmoid saturation.
    let input = uniform_bounds(&[NUM_BOXES, NUM_CLASSES], 10.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through confidence sigmoid");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf confidence scores IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // Sigmoid output must be strictly in [0, 1].
    assert!(
        lo_min >= -1e-5,
        "confidence lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-5,
        "confidence upper bound must be <= 1, got {hi_max}"
    );

    // Non-vacuous: with logits in [-10, 10], sigmoid covers nearly all of (0, 1).
    let width = hi_max - lo_min;
    assert!(
        width > 0.9,
        "confidence bounds should span nearly [0, 1] for wide logits, got width={width}"
    );
}

// ===========================================================================
// 6. NMS score filtering output non-negative (IBP)
// ===========================================================================

/// Verifies that NMS score filtering (sigmoid - threshold, then ReLU)
/// produces non-negative output scores.
///
/// This matches the filtering in `DpdfPipelineMetal::process_page` where
/// detections below `layout_conf_threshold` are discarded.
#[test]
fn test_dpdf_nms_score_filtering_ibp() {
    let conf_threshold = 0.25f32;

    let mut b = TensorBlockBuilder::new("dpdf_nms_filter");
    let input = b.add_input("cls_logits", &[NUM_BOXES, NUM_CLASSES]);

    // sigmoid -> subtract threshold -> ReLU (zero out below-threshold).
    let conf = b.add_sigmoid(input, &[NUM_BOXES, NUM_CLASSES]);
    let thresh = b.add_input("threshold", &[NUM_BOXES, NUM_CLASSES]);
    let diff = b.add_binary_add(conf, thresh, &[NUM_BOXES, NUM_CLASSES]);
    let out = b.add_relu(diff, &[NUM_BOXES, NUM_CLASSES]);
    let def = b.build(out).expect("valid NMS filter kernel");

    // Threshold as negative constant (subtracting via add).
    let thresh_data = ArrayD::from_elem(IxDyn(&[NUM_BOXES, NUM_CLASSES]), -conf_threshold);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(thresh_data),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_BOXES, NUM_CLASSES], 5.0);

    let output = graph.propagate_ibp(&input).expect("IBP through NMS filter");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf NMS filter IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // ReLU guarantees non-negative output.
    assert!(
        lo_min >= -1e-5,
        "NMS filtered scores must be >= 0 (ReLU), got {lo_min}"
    );
    // Upper bound: sigmoid max is 1, minus threshold 0.25, so max is 0.75.
    assert!(
        hi_max <= 1.0 - conf_threshold + 1e-3,
        "NMS filtered upper should be <= {}, got {hi_max}",
        1.0 - conf_threshold
    );
}

// ===========================================================================
// 7. End-to-end preprocess -> detect -> sigmoid pipeline (IBP)
// ===========================================================================

/// End-to-end pipeline matching `DpdfPipelineMetal::process_page`:
///   Image [C, H, W] -> normalize -> Conv backbone (stem) ->
///   flatten -> Linear cls head -> sigmoid -> confidence [0, 1].
///
/// Proves that for any pixel input in [0, 1], the detection confidence
/// output is bounded in [0, 1].
#[test]
fn test_dpdf_end_to_end_preprocess_detect_ibp() {
    let ch = BASE_CH;
    let stem_sp = IMG_H / 2;
    let num_pos = stem_sp * stem_sp;

    let mut b = TensorBlockBuilder::new("dpdf_e2e");
    let input = b.add_input("image", &[IMG_C, IMG_H, IMG_W]);

    // Step 1: ImageNet normalization.
    let inv_std: Vec<f32> = IMAGENET_STD.iter().map(|s| 1.0 / s).collect();
    let neg_mean_div_std: Vec<f32> = IMAGENET_MEAN
        .iter()
        .zip(IMAGENET_STD.iter())
        .map(|(m, s)| -m / s)
        .collect();

    let norm_scale = b.add_input("norm_scale", &[IMG_C]);
    let norm_bias = b.add_input("norm_bias", &[IMG_C]);
    let norm_scale_bc = b.add_broadcast_left(norm_scale, &[IMG_C, IMG_H, IMG_W]);
    let norm_bias_bc = b.add_broadcast_left(norm_bias, &[IMG_C, IMG_H, IMG_W]);
    let normed = b.add_binary_mul(input, norm_scale_bc, &[IMG_C, IMG_H, IMG_W]);
    let normed = b.add_binary_add(normed, norm_bias_bc, &[IMG_C, IMG_H, IMG_W]);

    // Step 2: Backbone stem (ConvBnAct stride-2).
    let stem = add_conv_bn_act(&mut b, normed, IMG_C, ch, 3, 2, 1, stem_sp, stem_sp, "stem");

    // Step 3: Flatten -> transpose for classification.
    let flat = b.add_reshape(stem, &[ch, num_pos]);
    let transposed = b.add_transpose(flat, &[1, 0], &[num_pos, ch]);

    // Step 4: Classification head -> sigmoid.
    let cls_w = b.add_input("cls_w", &[NUM_CLASSES, ch]);
    let logits = b.add_linear(transposed, cls_w, None, &[num_pos, NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[num_pos, NUM_CLASSES]);
    let def = b.build(out).expect("valid e2e pipeline kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.push(TensorParamBinding::ConstantTensor(
        ArrayD::from_shape_vec(IxDyn(&[IMG_C]), inv_std).expect("inv_std"),
    ));
    bindings.push(TensorParamBinding::ConstantTensor(
        ArrayD::from_shape_vec(IxDyn(&[IMG_C]), neg_mean_div_std).expect("neg_mean_div_std"),
    ));
    push_conv_bn_act_bindings(&mut bindings, IMG_C, ch, 3); // stem
    bindings.push(weight(&[NUM_CLASSES, ch])); // cls_w

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_input_bounds();

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through e2e pipeline");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf e2e pipeline IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // End-to-end: sigmoid guarantees [0, 1].
    assert!(
        lo_min >= -1e-3,
        "e2e sigmoid lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-3,
        "e2e sigmoid upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 8. Monotone tightening: tighter pixel range -> tighter detections (IBP)
// ===========================================================================

/// Verifies IBP monotonicity: narrower input pixel bounds produce output
/// bounds that are no wider than those from the full [0, 1] pixel range.
///
/// This is a fundamental soundness property: if the input domain shrinks,
/// the output bounds cannot grow.
#[test]
fn test_dpdf_monotone_tightening_ibp() {
    let ch = BASE_CH;
    let stem_sp = IMG_H / 2;
    let num_pos = stem_sp * stem_sp;

    // Build the same backbone -> sigmoid pipeline for both runs.
    let build_pipeline = || {
        let mut b = TensorBlockBuilder::new("dpdf_monotone");
        let input = b.add_input("image", &[IMG_C, IMG_H, IMG_W]);

        // Backbone stem.
        let stem = add_conv_bn_act(&mut b, input, IMG_C, ch, 3, 2, 1, stem_sp, stem_sp, "stem");
        let flat = b.add_reshape(stem, &[ch, num_pos]);
        let transposed = b.add_transpose(flat, &[1, 0], &[num_pos, ch]);

        let cls_w = b.add_input("cls_w", &[NUM_CLASSES, ch]);
        let logits = b.add_linear(transposed, cls_w, None, &[num_pos, NUM_CLASSES]);
        let out = b.add_sigmoid(logits, &[num_pos, NUM_CLASSES]);
        let def = b.build(out).expect("valid monotone pipeline kernel");

        let mut bindings = vec![TensorParamBinding::Variable];
        push_conv_bn_act_bindings(&mut bindings, IMG_C, ch, 3);
        bindings.push(weight(&[NUM_CLASSES, ch]));

        tensor_kernel_to_graph(&def, &bindings).expect("graph translation")
    };

    let graph = build_pipeline();

    // Wide input: pixels in [0, 1].
    let wide_input = image_input_bounds();
    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");

    // Narrow input: pixels in [0.2, 0.8] (centered, 60% of full range).
    let narrow_input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[IMG_C, IMG_H, IMG_W]), 0.2f32),
        ArrayD::from_elem(IxDyn(&[IMG_C, IMG_H, IMG_W]), 0.8f32),
    )
    .expect("valid narrow bounds");
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("IBP narrow");

    let (lo_w, hi_w) = bounds_min_max(&wide_output);
    let (lo_n, hi_n) = bounds_min_max(&narrow_output);
    let wide_width = hi_w - lo_w;
    let narrow_width = hi_n - lo_n;

    eprintln!(
        "dpdf monotone tightening: wide=[{lo_w:.4}, {hi_w:.4}] w={wide_width:.4} \
         | narrow=[{lo_n:.4}, {hi_n:.4}] w={narrow_width:.4}"
    );

    // Monotonicity: narrow input bounds -> output bounds no wider.
    assert!(
        narrow_width <= wide_width + 1e-4,
        "monotone tightening violated: narrow_width={narrow_width} > wide_width={wide_width}"
    );
}
